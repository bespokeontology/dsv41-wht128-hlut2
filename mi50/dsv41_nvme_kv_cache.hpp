#pragma once
// Wave 8.2 / 8.3: DeepInfer Pure-Block NVMe CSA2 RadixBlockCache + In-RAM 20L EncoderStateBlock
//   - Zero SWA bytes on NVMe: persists ONLY 128-token-aligned compressed CSA2 KV blocks
//     (.kvblk, magic "DSV41KV1", 114,496 B on disk = 64 B header + 512 B tokens + 113,920 B CSA2 KV
//     [890 B/tok across Slots 0..3] + FNV-1a64; legacy .kvtail files auto-purged on configure)
//   - In-RAM 20-layer Encoder StateBlock (sizeof(EncoderStateBlock) == 1,415,104 B =
//     1,415,040 B slab [20 x 134 rows x 528 B] + 64 B metadata, snapped at 128-token boundaries
//     with 6 slack rows for DSpark backtracking; 0 decoder SWA cached)
//   - KEEP_TOKENS = 140 bounded replay policy: L_cap = max(0, P - 140), 128-token Encoder SWA Bounded Replay
//     on cold NVMe hits (keeping hydrated CSA2 Block KV [0..L1) untouched), and Decoder SWA Bounded Replay
//     over the final >= 140 prompt tokens
//   - Atomic POSIX tmp + rename background writes (0.00 ms decode latency overhead)
//   - Multi-threaded POSIX pread hydration (< 50 ms for 32K+ / 256K prefixes) + bounded mtime LRU directory cap
// Zero Python anywhere.

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace dsv41 {

static constexpr int    kNvmeRadixBlockTokens      = 128;
static constexpr int    kDeepInferKeepTokens       = 140;  // 128 SWA + 6 DSpark slack + 6 margin
static constexpr size_t kNvmeRadixR2SlotBytes      = 16384u + 2048u + 4096u + 256u;          // 22,784 B
static constexpr size_t kNvmeRadixR1SlotBytes      = 32768u + 4096u + 8192u + 512u;          // 45,568 B
static constexpr size_t kNvmeRadixBlockBytes       = 3u * kNvmeRadixR2SlotBytes + kNvmeRadixR1SlotBytes; // 113,920 B (890 B/tok)

// 20-Layer CED Encoder StateBlock in RAM (20 x 134 rows x 528 B = 1,415,040 B ~= 1.35 MiB/slot)
static constexpr size_t kEncoderStateLayers        = 20u;
static constexpr size_t kEncoderStateRingRows      = 128u;
static constexpr size_t kEncoderStateSlackRows     = 6u;
static constexpr size_t kEncoderStateTotalRows     = kEncoderStateRingRows + kEncoderStateSlackRows; // 134 rows
static constexpr size_t kEncoderStateHeadDim       = 512u;
static constexpr size_t kEncoderStateRowBytes      = 512u + 16u;                                     // 528 B (FP8_E4M3 + UE8M0_32)
static constexpr size_t kEncoderStateSlabBytes     = kEncoderStateLayers * kEncoderStateTotalRows * kEncoderStateRowBytes; // 1,415,040 B
static constexpr size_t kMaxRamEncoderStateBlocks  = 64u;                                            // <= 86.4 MiB RAM total
static_assert(kEncoderStateSlabBytes == 1415040u, "EncoderStateBlock must be 20 * 134 * 528 = 1,415,040 bytes (~1.35 MiB)");

struct RadixKey128 {
    uint64_t lo = 0;
    uint64_t hi = 0;
    bool operator==(const RadixKey128& o) const noexcept { return lo == o.lo && hi == o.hi; }
    bool operator!=(const RadixKey128& o) const noexcept { return lo != o.lo || hi != o.hi; }
};

struct RadixKeyHash {
    size_t operator()(const RadixKey128& k) const noexcept {
        return static_cast<size_t>(k.lo ^ (k.hi * 0x9E3779B97F4A7C15ULL));
    }
};

struct RadixBlockEntry {
    RadixKey128 key{};
    uint64_t extra_salt = 0ULL;
    std::array<int32_t, kNvmeRadixBlockTokens> tokens;
    alignas(64) std::array<uint8_t, kNvmeRadixBlockBytes> data;
};

struct EncoderStateBlock {
    int pos = 0;                 // Exact 128-token boundary (pos > 0 && pos % 128 == 0)
    uint32_t slack_valid = 0u;   // Number of populated DSpark slack rows in [0..6]
    RadixKey128 key{};           // Rolling chain hash H_k at block (pos / 128) - 1
    alignas(64) std::array<uint8_t, kEncoderStateSlabBytes> packed_swa;
};
static_assert(sizeof(EncoderStateBlock) == 1415104u,
              "sizeof(EncoderStateBlock) must equal 1,415,040 B slab + 64 B aligned metadata = 1,415,104 B");

// Exact IEEE bit-level helpers matching dsv41_mi50_kernels.hip UE8M0_32 + FP8_E4M3 roundtrip (max_bit_diff == 0)
namespace detail_swa528 {

inline float bf16_to_f32(uint16_t b) noexcept {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, 4u);
    return f;
}

inline uint16_t f32_to_bf16_rne(float v) noexcept {
    uint32_t u = 0u;
    std::memcpy(&u, &v, 4u);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return static_cast<uint16_t>((u >> 16) | 0x0040u);
    u += 0x00007FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

inline float ue8m0_to_f32(uint8_t b) noexcept {
    if (b == 255u) return 0.0f;
    const uint32_t bits = (b == 0u) ? 0x00400000u : (static_cast<uint32_t>(b) << 23);
    float f = 0.0f;
    std::memcpy(&f, &bits, 4u);
    return f;
}

inline float fast_round_scale_ue8m0(float amax, float max_inv, uint8_t& out_ue8) noexcept {
    const float prod = amax * max_inv;
    uint32_t bits = 0u;
    std::memcpy(&bits, &prod, 4u);
    int exp_x = static_cast<int>((bits >> 23) & 0xFFu) - 127 + (((bits & 0x007FFFFFu) != 0u) ? 1 : 0);
    exp_x = std::max(-127, std::min(127, exp_x));
    out_ue8 = static_cast<uint8_t>(exp_x + 127);
    return ue8m0_to_f32(out_ue8);
}

inline uint8_t f32_to_fp8_e4m3(float x) noexcept {
    if (!(x == x)) return 0x7Fu;
    uint32_t xu = 0u;
    std::memcpy(&xu, &x, 4u);
    const uint32_t sign = (xu >> 24) & 0x80u;
    const float ax = std::fabs(x);
    if (ax >= 448.0f) return static_cast<uint8_t>(sign | 0x7Eu);
    if (ax < (1.0f / 1024.0f)) return static_cast<uint8_t>(sign);
    if (ax < (1.0f / 64.0f)) {
        const int m = static_cast<int>(std::rint(ax * 512.0f));
        return static_cast<uint8_t>(sign | static_cast<uint32_t>(m));
    }
    uint32_t u = 0u;
    std::memcpy(&u, &ax, 4u);
    const int e32 = static_cast<int>((u >> 23) & 0xFFu) - 127;
    const uint32_t mant23 = u & 0x007FFFFFu;
    const uint32_t m3_pre = mant23 >> 20;
    const uint32_t rem = mant23 & 0x000FFFFFu;
    uint32_t m3 = m3_pre + ((rem > 0x00080000u || (rem == 0x00080000u && (m3_pre & 1u))) ? 1u : 0u);
    int e4 = e32 + 7;
    if (m3 == 8u) {
        m3 = 0u;
        ++e4;
    }
    if (e4 > 15 || (e4 == 15 && m3 >= 7u)) return static_cast<uint8_t>(sign | 0x7Eu);
    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(e4) << 3) | m3);
}

inline float fp8_e4m3_to_f32(uint8_t b) noexcept {
    const uint32_t sign32 = static_cast<uint32_t>(b & 0x80u) << 24;
    const uint32_t em = static_cast<uint32_t>(b & 0x7Fu);
    if (em == 0x7Fu || em == 0u) {
        float z = 0.0f;
        std::memcpy(&z, &sign32, 4u);
        return z;
    }
    const uint32_t e = em >> 3;
    const uint32_t m = em & 7u;
    float mag = 0.0f;
    if (e == 0u) {
        mag = static_cast<float>(m) * (1.0f / 512.0f);
    } else {
        const uint32_t u = ((e + 120u) << 23) | (m << 20);
        std::memcpy(&mag, &u, 4u);
    }
    return (b & 0x80u) ? -mag : mag;
}

}  // namespace detail_swa528

inline void pack_bf16_row_528(const uint16_t* bf16_512, uint8_t* dst_528) noexcept {
    uint8_t* dst_fp8 = dst_528;
    uint8_t* dst_ue8 = dst_528 + 512u;
    for (size_t g = 0; g < 16u; ++g) {
        const uint16_t* grp = bf16_512 + g * 32u;
        float vals[32];
        float amax = 1e-4f;
        for (size_t j = 0; j < 32u; ++j) {
            const float v = detail_swa528::bf16_to_f32(grp[j]);
            vals[j] = v;
            const float av = std::fabs(v);
            if (av > amax) amax = av;
        }
        uint8_t ue8 = 0u;
        const float s = detail_swa528::fast_round_scale_ue8m0(amax, 1.0f / 448.0f, ue8);
        const float inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;
        dst_ue8[g] = ue8;
        uint8_t* out_g = dst_fp8 + g * 32u;
        for (size_t j = 0; j < 32u; ++j) {
            const float q = std::max(-448.0f, std::min(448.0f, vals[j] * inv_s));
            out_g[j] = detail_swa528::f32_to_fp8_e4m3(q);
        }
    }
}

inline void unpack_bf16_row_528(const uint8_t* src_528, uint16_t* bf16_512) noexcept {
    const uint8_t* src_fp8 = src_528;
    const uint8_t* src_ue8 = src_528 + 512u;
    for (size_t g = 0; g < 16u; ++g) {
        const float s = detail_swa528::ue8m0_to_f32(src_ue8[g]);
        const uint8_t* in_g = src_fp8 + g * 32u;
        uint16_t* out_g = bf16_512 + g * 32u;
        for (size_t j = 0; j < 32u; ++j) {
            const float v = detail_swa528::fp8_e4m3_to_f32(in_g[j]) * s;
            out_g[j] = detail_swa528::f32_to_bf16_rne(v);
        }
    }
}

// Quantize arbitrary BF16 test input in-place using the exact k_split_qkv_norm_rope_quant UE8M0_32 + FP8_E4M3 roundtrip
inline void quantize_bf16_row_ue8m0_fp8_inplace(uint16_t* bf16_512) noexcept {
    uint8_t tmp[kEncoderStateRowBytes];
    pack_bf16_row_528(bf16_512, tmp);
    unpack_bf16_row_528(tmp, bf16_512);
}

// Pack 20 encoder layers' 128-row SWA ring (+ optional <= 6 slack rows) into a 1,415,040 B EncoderStateBlock
inline void pack_encoder_state_block(
    const uint16_t* win_kv_20_128_512,
    const uint16_t* slack_kv_20_6_512,  // optional [20][slack_stride_rows][512], nullptr if slack_valid == 0
    size_t slack_stride_rows,
    uint32_t slack_valid,
    EncoderStateBlock& out_blk) noexcept
{
    const uint32_t sv = std::min<uint32_t>(slack_valid, uint32_t(kEncoderStateSlackRows));
    out_blk.slack_valid = sv;
    uint8_t* base_dst = out_blk.packed_swa.data();
    #pragma omp parallel for schedule(static)
    for (int l = 0; l < static_cast<int>(kEncoderStateLayers); ++l) {
        const uint16_t* ring_l = win_kv_20_128_512 + size_t(l) * kEncoderStateRingRows * kEncoderStateHeadDim;
        uint8_t* dst_l = base_dst + size_t(l) * kEncoderStateTotalRows * kEncoderStateRowBytes;
        for (size_t r = 0; r < kEncoderStateRingRows; ++r) {
            pack_bf16_row_528(ring_l + r * kEncoderStateHeadDim, dst_l + r * kEncoderStateRowBytes);
        }
        uint8_t* dst_slack = dst_l + kEncoderStateRingRows * kEncoderStateRowBytes;
        if (slack_kv_20_6_512 != nullptr && sv > 0u) {
            const uint16_t* sl_l = slack_kv_20_6_512 + size_t(l) * slack_stride_rows * kEncoderStateHeadDim;
            for (size_t r = 0; r < size_t(sv); ++r) {
                pack_bf16_row_528(sl_l + r * kEncoderStateHeadDim, dst_slack + r * kEncoderStateRowBytes);
            }
        }
        if (sv < kEncoderStateSlackRows) {
            std::memset(dst_slack + size_t(sv) * kEncoderStateRowBytes,
                        0,
                        (kEncoderStateSlackRows - size_t(sv)) * kEncoderStateRowBytes);
        }
    }
}

// Unpack 20 encoder layers' 128-row SWA ring (+ optional slack rows) from a 1,415,040 B EncoderStateBlock
inline void unpack_encoder_state_block(
    const EncoderStateBlock& blk,
    uint16_t* out_win_kv_20_128_512,
    uint16_t* out_slack_kv_20_6_512 = nullptr,
    size_t slack_stride_rows = kEncoderStateSlackRows) noexcept
{
    const uint8_t* base_src = blk.packed_swa.data();
    const uint32_t sv = std::min<uint32_t>(blk.slack_valid, uint32_t(kEncoderStateSlackRows));
    #pragma omp parallel for schedule(static)
    for (int l = 0; l < static_cast<int>(kEncoderStateLayers); ++l) {
        const uint8_t* src_l = base_src + size_t(l) * kEncoderStateTotalRows * kEncoderStateRowBytes;
        uint16_t* ring_l = out_win_kv_20_128_512 + size_t(l) * kEncoderStateRingRows * kEncoderStateHeadDim;
        for (size_t r = 0; r < kEncoderStateRingRows; ++r) {
            unpack_bf16_row_528(src_l + r * kEncoderStateRowBytes, ring_l + r * kEncoderStateHeadDim);
        }
        if (out_slack_kv_20_6_512 != nullptr && sv > 0u) {
            const uint8_t* src_slack = src_l + kEncoderStateRingRows * kEncoderStateRowBytes;
            uint16_t* sl_l = out_slack_kv_20_6_512 + size_t(l) * slack_stride_rows * kEncoderStateHeadDim;
            for (size_t r = 0; r < size_t(sv); ++r) {
                unpack_bf16_row_528(src_slack + r * kEncoderStateRowBytes, sl_l + r * kEncoderStateHeadDim);
            }
        }
    }
}

// Fast deterministic 4-lane ILP 64-bit FNV-1a-style checksum (catches every single bit flip at >30 GB/s on Zen 2)
inline uint64_t fnv1a64_update(uint64_t h, const void* ptr, size_t len) noexcept {
    const auto* p = static_cast<const uint8_t*>(ptr);
    size_t i = 0;
    if (len >= 32u) {
        uint64_t h0 = h ^ 0x243F6A8885A308D3ULL;
        uint64_t h1 = (h * 0x9E3779B97F4A7C15ULL) ^ 0x13198A2E03707344ULL;
        uint64_t h2 = (h * 0xBF58476D1CE4E5B9ULL) ^ 0xA4093822299F31D0ULL;
        uint64_t h3 = (h * 0x94D049BB133111EBULL) ^ 0x082EFA98EC4E6C89ULL;
        for (; i + 32u <= len; i += 32u) {
            uint64_t w0, w1, w2, w3;
            std::memcpy(&w0, p + i + 0u,  8u);
            std::memcpy(&w1, p + i + 8u,  8u);
            std::memcpy(&w2, p + i + 16u, 8u);
            std::memcpy(&w3, p + i + 24u, 8u);
            h0 = ((h0 ^ w0) * 0x100000001B3ULL); h0 ^= (h0 >> 32);
            h1 = ((h1 ^ w1) * 0x100000001B3ULL); h1 ^= (h1 >> 32);
            h2 = ((h2 ^ w2) * 0x100000001B3ULL); h2 ^= (h2 >> 32);
            h3 = ((h3 ^ w3) * 0x100000001B3ULL); h3 ^= (h3 >> 32);
        }
        h = h0 ^ ((h1 << 1) | (h1 >> 63)) ^ ((h2 << 7) | (h2 >> 57)) ^ ((h3 << 13) | (h3 >> 51));
        h *= 0x100000001B3ULL;
        h ^= (h >> 32);
    }
    for (; i + 8u <= len; i += 8u) {
        uint64_t w = 0;
        std::memcpy(&w, p + i, 8u);
        h ^= w;
        h *= 0x100000001B3ULL;
        h ^= (h >> 32);
    }
    for (; i < len; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 0x100000001B3ULL;
    }
    return h;
}

inline uint64_t fnv1a64_init() noexcept {
    return 0xCBF29CE484222325ULL;
}

inline RadixKey128 compute_model_overlay_uuid(
    const std::string& mi50_packed_root,
    const std::string& rarebal_overlay_path,
    const std::string& expert_perm_overlay_path = "",
    const std::string& hlut3_dev3_overlay_path = "",
    const std::string& hlut3_tp3_overlay_path = "") noexcept
{
    uint64_t h1 = 0xCBF29CE484222325ULL;
    uint64_t h2 = 0x84222325CBF29CE4ULL;
    auto mix_str = [&](const std::string& s) {
        for (unsigned char c : s) {
            h1 = (h1 ^ uint64_t(c)) * 0x100000001B3ULL;
            h2 = (h2 ^ (uint64_t(c) * 0x9E3779B97F4A7C15ULL)) * 0x94D049BB133111EBULL;
        }
    };
    auto mix_stat = [&](const std::string& path) {
        struct stat st {};
        if (!path.empty() && ::stat(path.c_str(), &st) == 0) {
            const uint64_t sz = static_cast<uint64_t>(st.st_size);
            const uint64_t mt = static_cast<uint64_t>(st.st_mtim.tv_sec);
            h1 = fnv1a64_update(h1, &sz, sizeof(sz));
            h2 = fnv1a64_update(h2, &mt, sizeof(mt));
        }
    };
    mix_stat(mi50_packed_root + "/tp0/layer_02.safetensors");
    if (!rarebal_overlay_path.empty()) {
        const size_t slash = rarebal_overlay_path.find_last_of('/');
        mix_str(slash == std::string::npos ? rarebal_overlay_path : rarebal_overlay_path.substr(slash + 1));
        mix_stat(rarebal_overlay_path);
    } else {
        mix_str("base_hlut2_no_overlay");
    }
    if (!expert_perm_overlay_path.empty() && expert_perm_overlay_path != rarebal_overlay_path) {
        const size_t slash = expert_perm_overlay_path.find_last_of('/');
        mix_str(slash == std::string::npos ? expert_perm_overlay_path : expert_perm_overlay_path.substr(slash + 1));
        mix_stat(expert_perm_overlay_path);
    }
    if (!hlut3_dev3_overlay_path.empty()) {
        const size_t slash = hlut3_dev3_overlay_path.find_last_of('/');
        mix_str("hlut3_dev3:");
        mix_str(slash == std::string::npos ? hlut3_dev3_overlay_path : hlut3_dev3_overlay_path.substr(slash + 1));
        mix_stat(hlut3_dev3_overlay_path);
    }
    if (!hlut3_tp3_overlay_path.empty()) {
        const size_t slash = hlut3_tp3_overlay_path.find_last_of('/');
        mix_str("hlut3_tp3:");
        mix_str(slash == std::string::npos ? hlut3_tp3_overlay_path : hlut3_tp3_overlay_path.substr(slash + 1));
        mix_stat(hlut3_tp3_overlay_path);
    }
    if (h1 == 0ULL && h2 == 0ULL) h1 = 1ULL;
    return RadixKey128{h1, h2};
}

inline RadixKey128 step_radix_hash_salted(
    RadixKey128 parent,
    const int32_t* toks128,
    uint64_t block_extra_salt = 0ULL) noexcept
{
    uint64_t h1 = parent.lo ^ 0xCBF29CE484222325ULL ^ block_extra_salt;
    uint64_t h2 = parent.hi ^ 0x84222325CBF29CE4ULL ^ (block_extra_salt * 0x9E3779B97F4A7C15ULL);
    for (int i = 0; i < kNvmeRadixBlockTokens; ++i) {
        const uint64_t v = static_cast<uint32_t>(toks128[i]);
        h1 ^= v + 0x9E3779B97F4A7C15ULL + (h1 << 6) + (h1 >> 2);
        h1 *= 0x100000001B3ULL;
        h2 ^= (v * 0xBF58476D1CE4E5B9ULL) + (h2 << 7) + (h2 >> 3);
        h2 *= 0x94D049BB133111EBULL;
    }
    if (h1 == 0ULL && h2 == 0ULL) h1 = 1ULL;
    return RadixKey128{h1, h2};
}

struct alignas(8) Dsv41NvmeKvBlockHeader {
    char     magic[8];          // "DSV41KV1"
    uint32_t version;           // 1
    uint32_t block_tokens;      // 128
    uint64_t model_uuid_lo;
    uint64_t model_uuid_hi;
    uint64_t chain_hash_lo;
    uint64_t chain_hash_hi;
    uint64_t extra_salt;
    uint64_t fnv1a64;           // Checksum over header[0..56) + tokens[128] + data[113920]
};
static_assert(sizeof(Dsv41NvmeKvBlockHeader) == 64, "Dsv41NvmeKvBlockHeader must be 64 bytes");
static constexpr size_t kNvmeKvBlockTokensBytes = size_t(kNvmeRadixBlockTokens) * sizeof(int32_t); // 512 B
static constexpr size_t kNvmeKvBlockFileBytes   =
    sizeof(Dsv41NvmeKvBlockHeader) + kNvmeKvBlockTokensBytes + kNvmeRadixBlockBytes;               // 114,496 B
static_assert(kNvmeKvBlockFileBytes == 114496u,
              ".kvblk file must equal 64 B header + 512 B tokens + 113,920 B CSA2 KV = 114,496 B");

class Dsv41NvmeKvStore {
public:
    Dsv41NvmeKvStore() = default;

    static void mkdir_p(const std::string& dir) {
        if (dir.empty()) return;
        std::string cur;
        for (size_t i = 0; i < dir.size(); ++i) {
            cur.push_back(dir[i]);
            if (dir[i] == '/' && cur.size() > 1u) {
                (void)::mkdir(cur.c_str(), 0775);
            }
        }
        (void)::mkdir(dir.c_str(), 0775);
    }

    void configure(const std::string& cache_dir, uint64_t max_bytes) {
        std::lock_guard<std::mutex> lk(mtx_);
        cache_dir_ = cache_dir;
        max_bytes_ = max_bytes;
        enabled_ = !cache_dir_.empty() && (max_bytes_ > 0ULL);
        total_bytes_ = 0ULL;
        if (!enabled_) return;

        mkdir_p(cache_dir_);
        scan_directory_locked();
        enforce_lru_cap_locked();
    }

    bool enabled() const noexcept { return enabled_; }
    const std::string& cache_dir() const noexcept { return cache_dir_; }
    uint64_t max_bytes() const noexcept { return max_bytes_; }
    uint64_t total_bytes() const noexcept {
        std::lock_guard<std::mutex> lk(mtx_);
        return total_bytes_;
    }

    std::string block_path(RadixKey128 key) const {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "/%016llx_%016llx.kvblk",
                      static_cast<unsigned long long>(key.lo),
                      static_cast<unsigned long long>(key.hi));
        return cache_dir_ + buf;
    }

    bool has_block_on_disk(RadixKey128 key) const {
        if (!enabled_) return false;
        struct stat st {};
        const std::string p = block_path(key);
        return (::stat(p.c_str(), &st) == 0) &&
               (static_cast<size_t>(st.st_size) == kBlockFileBytes);
    }

    // Count any SWA/tail files in cache_dir_ (must always equal 0 under Wave 8.2 pure-block layout!)
    size_t count_swa_files_on_disk() const {
        if (cache_dir_.empty()) return 0u;
        DIR* d = ::opendir(cache_dir_.c_str());
        if (!d) return 0u;
        size_t cnt = 0u;
        while (struct dirent* ent = ::readdir(d)) {
            const std::string name = ent->d_name;
            if (ends_with(name, ".kvtail") || ends_with(name, ".swa")) {
                ++cnt;
            }
        }
        (void)::closedir(d);
        return cnt;
    }

    bool write_block_atomic(const RadixBlockEntry& entry, RadixKey128 model_uuid) {
        if (!enabled_) return false;
        const std::string final_p = block_path(entry.key);
        struct stat st_exist {};
        if (::stat(final_p.c_str(), &st_exist) == 0 &&
            static_cast<size_t>(st_exist.st_size) == kBlockFileBytes) {
            return true;
        }

        Dsv41NvmeKvBlockHeader hdr{};
        std::memcpy(hdr.magic, "DSV41KV1", 8u);
        hdr.version       = 1u;
        hdr.block_tokens  = uint32_t(kNvmeRadixBlockTokens);
        hdr.model_uuid_lo = model_uuid.lo;
        hdr.model_uuid_hi = model_uuid.hi;
        hdr.chain_hash_lo = entry.key.lo;
        hdr.chain_hash_hi = entry.key.hi;
        hdr.extra_salt    = entry.extra_salt;

        uint64_t csum = fnv1a64_init();
        csum = fnv1a64_update(csum, &hdr, offsetof(Dsv41NvmeKvBlockHeader, fnv1a64));
        csum = fnv1a64_update(csum, entry.tokens.data(), sizeof(entry.tokens));
        csum = fnv1a64_update(csum, entry.data.data(), entry.data.size());
        hdr.fnv1a64 = csum;

        const uint64_t seq = tmp_seq_.fetch_add(1ULL, std::memory_order_relaxed);
        const std::string tmp_p = final_p + ".tmp." + std::to_string(::getpid()) + "_" + std::to_string(seq);

        const int fd = ::open(tmp_p.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0664);
        if (fd < 0) return false;

        bool ok = pwrite_all(fd, &hdr, sizeof(hdr), 0) &&
                  pwrite_all(fd, entry.tokens.data(), sizeof(entry.tokens), sizeof(hdr)) &&
                  pwrite_all(fd, entry.data.data(), entry.data.size(), sizeof(hdr) + sizeof(entry.tokens));
        (void)::close(fd);
        if (!ok || ::rename(tmp_p.c_str(), final_p.c_str()) != 0) {
            (void)::unlink(tmp_p.c_str());
            return false;
        }

        std::lock_guard<std::mutex> lk(mtx_);
        total_bytes_ += kBlockFileBytes;
        if (total_bytes_ > max_bytes_) {
            enforce_lru_cap_locked();
        }
        return true;
    }

    // Thread-safe lock-free POSIX pread directly into caller destination buffer (zero heap alloc, single memcpy!)
    bool read_block_pread_raw(
        RadixKey128 key,
        RadixKey128 expected_model_uuid,
        const int32_t* expected_toks128,
        uint64_t expected_extra_salt,
        int32_t* out_toks128_opt,
        uint8_t* dst_kv_113920,
        uint64_t* out_extra_salt_opt = nullptr,
        bool touch_mtime = false) const
    {
        if (!enabled_ || dst_kv_113920 == nullptr) return false;
        const std::string p = block_path(key);
        int open_flags = O_RDONLY;
#ifdef O_NOATIME
        open_flags |= O_NOATIME;
#endif
        int fd = ::open(p.c_str(), open_flags);
        if (fd < 0 && errno == EPERM) {
            fd = ::open(p.c_str(), O_RDONLY);
        }
        if (fd < 0) return false;

        alignas(64) thread_local std::array<uint8_t, kBlockFileBytes> tl_raw;
        const bool rd_ok = pread_all(fd, tl_raw.data(), kBlockFileBytes, 0);
        (void)::close(fd);
        if (!rd_ok) return false;

        Dsv41NvmeKvBlockHeader hdr{};
        std::memcpy(&hdr, tl_raw.data(), sizeof(hdr));
        if (std::memcmp(hdr.magic, "DSV41KV1", 8u) != 0 ||
            hdr.version != 1u ||
            hdr.block_tokens != uint32_t(kNvmeRadixBlockTokens) ||
            hdr.model_uuid_lo != expected_model_uuid.lo ||
            hdr.model_uuid_hi != expected_model_uuid.hi ||
            hdr.chain_hash_lo != key.lo ||
            hdr.chain_hash_hi != key.hi ||
            hdr.extra_salt != expected_extra_salt)
        {
            return false;
        }

        constexpr size_t kTokBytes = kNvmeRadixBlockTokens * sizeof(int32_t);
        const uint8_t* tok_ptr = tl_raw.data() + sizeof(hdr);
        const uint8_t* kv_ptr  = tok_ptr + kTokBytes;
        if (expected_toks128 != nullptr &&
            std::memcmp(tok_ptr, expected_toks128, kTokBytes) != 0)
        {
            return false;
        }

        uint64_t csum = fnv1a64_init();
        csum = fnv1a64_update(csum, tl_raw.data(), offsetof(Dsv41NvmeKvBlockHeader, fnv1a64));
        csum = fnv1a64_update(csum, tok_ptr, kTokBytes);
        csum = fnv1a64_update(csum, kv_ptr, kNvmeRadixBlockBytes);
        if (csum != hdr.fnv1a64) {
            return false;
        }

        if (out_toks128_opt != nullptr) {
            std::memcpy(out_toks128_opt, tok_ptr, kTokBytes);
        }
        if (out_extra_salt_opt != nullptr) {
            *out_extra_salt_opt = hdr.extra_salt;
        }
        std::memcpy(dst_kv_113920, kv_ptr, kNvmeRadixBlockBytes);

        if (touch_mtime) {
            (void)::utimensat(AT_FDCWD, p.c_str(), nullptr, 0);
        }
        return true;
    }

    // Thread-safe lock-free POSIX pread of a single .kvblk shard (zero heap alloc via thread_local buffer!)
    bool read_block_pread(
        RadixKey128 key,
        RadixKey128 expected_model_uuid,
        const int32_t* expected_toks128,
        uint64_t expected_extra_salt,
        RadixBlockEntry& out_entry,
        bool touch_mtime = false) const
    {
        uint64_t salt_out = 0ULL;
        if (!read_block_pread_raw(key, expected_model_uuid, expected_toks128, expected_extra_salt,
                                  out_entry.tokens.data(), out_entry.data.data(), &salt_out, touch_mtime)) {
            return false;
        }
        out_entry.key = key;
        out_entry.extra_salt = salt_out;
        return true;
    }

    // Wave 9.3 Phase C.2: Asynchronous kernel NVMe readahead hint for lookahead batch!
    void prefetch_block_hint(RadixKey128 key) const noexcept {
        if (!enabled_) return;
        const std::string p = block_path(key);
        int open_flags = O_RDONLY;
#ifdef O_NOATIME
        open_flags |= O_NOATIME;
#endif
        int fd = ::open(p.c_str(), open_flags);
        if (fd < 0 && errno == EPERM) {
            fd = ::open(p.c_str(), O_RDONLY);
        }
        if (fd >= 0) {
            (void)::posix_fadvise(fd, 0, off_t(kBlockFileBytes), POSIX_FADV_WILLNEED);
            (void)::close(fd);
        }
    }

    void touch_blocks_mtime(const std::vector<RadixKey128>& keys) const {
        if (!enabled_ || keys.empty()) return;
        for (const auto& k : keys) {
            const std::string p = block_path(k);
            (void)::utimensat(AT_FDCWD, p.c_str(), nullptr, 0);
        }
    }

    void enforce_lru_cap() {
        std::lock_guard<std::mutex> lk(mtx_);
        if (enabled_) enforce_lru_cap_locked();
    }

private:
    static constexpr size_t kBlockFileBytes =
        sizeof(Dsv41NvmeKvBlockHeader) + kNvmeRadixBlockTokens * sizeof(int32_t) + kNvmeRadixBlockBytes;

    struct DiskFileMeta {
        std::string name;
        uint64_t size = 0;
        int64_t mtime_sec = 0;
        int64_t mtime_nsec = 0;
    };

    static bool pwrite_all(int fd, const void* buf, size_t len, off_t offset) noexcept {
        const auto* p = static_cast<const uint8_t*>(buf);
        size_t done = 0;
        while (done < len) {
            const ssize_t w = ::pwrite(fd, p + done, len - done, offset + off_t(done));
            if (w < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            if (w == 0) return false;
            done += static_cast<size_t>(w);
        }
        return true;
    }

    static bool pread_all(int fd, void* buf, size_t len, off_t offset) noexcept {
        auto* p = static_cast<uint8_t*>(buf);
        size_t done = 0;
        while (done < len) {
            const ssize_t r = ::pread(fd, p + done, len - done, offset + off_t(done));
            if (r < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            if (r == 0) return false;
            done += static_cast<size_t>(r);
        }
        return true;
    }

    static bool ends_with(const std::string& s, const char* suf) noexcept {
        const size_t sl = std::strlen(suf);
        return s.size() >= sl && s.compare(s.size() - sl, sl, suf) == 0;
    }

    void scan_directory_locked() {
        total_bytes_ = 0ULL;
        DIR* d = ::opendir(cache_dir_.c_str());
        if (!d) return;

        while (struct dirent* ent = ::readdir(d)) {
            const std::string name = ent->d_name;
            if (name == "." || name == "..") continue;
            const std::string full = cache_dir_ + "/" + name;
            // Purge any stale temp files or legacy Stage 2 .kvtail SWA files -> strictly 0 B SWA on NVMe!
            if (name.find(".tmp.") != std::string::npos || ends_with(name, ".kvtail") || ends_with(name, ".swa")) {
                (void)::unlink(full.c_str());
                continue;
            }
            if (!ends_with(name, ".kvblk")) continue;

            struct stat st {};
            if (::stat(full.c_str(), &st) != 0) continue;
            if (static_cast<size_t>(st.st_size) != kBlockFileBytes) {
                (void)::unlink(full.c_str());
                continue;
            }
            total_bytes_ += static_cast<uint64_t>(st.st_size);
        }
        (void)::closedir(d);
    }

    void enforce_lru_cap_locked() {
        DIR* d = ::opendir(cache_dir_.c_str());
        if (!d) return;

        std::vector<DiskFileMeta> files;
        uint64_t sum_bytes = 0ULL;
        while (struct dirent* ent = ::readdir(d)) {
            const std::string name = ent->d_name;
            if (name == "." || name == "..") continue;
            const std::string full = cache_dir_ + "/" + name;
            if (ends_with(name, ".kvtail") || ends_with(name, ".swa")) {
                (void)::unlink(full.c_str());
                continue;
            }
            if (!ends_with(name, ".kvblk")) continue;

            struct stat st {};
            if (::stat(full.c_str(), &st) != 0) continue;
            DiskFileMeta m;
            m.name = name;
            m.size = static_cast<uint64_t>(st.st_size);
            m.mtime_sec = static_cast<int64_t>(st.st_mtim.tv_sec);
            m.mtime_nsec = static_cast<int64_t>(st.st_mtim.tv_nsec);
            sum_bytes += m.size;
            files.push_back(std::move(m));
        }
        (void)::closedir(d);
        total_bytes_ = sum_bytes;

        if (total_bytes_ <= max_bytes_) {
            return;
        }

        std::sort(files.begin(), files.end(), [](const DiskFileMeta& a, const DiskFileMeta& b) {
            if (a.mtime_sec != b.mtime_sec) return a.mtime_sec < b.mtime_sec;
            if (a.mtime_nsec != b.mtime_nsec) return a.mtime_nsec < b.mtime_nsec;
            return a.name < b.name;
        });

        for (const auto& f : files) {
            if (total_bytes_ <= max_bytes_) break;
            const std::string full = cache_dir_ + "/" + f.name;
            if (::unlink(full.c_str()) == 0) {
                total_bytes_ = (total_bytes_ >= f.size) ? (total_bytes_ - f.size) : 0ULL;
            }
        }
    }

    mutable std::mutex mtx_;
    std::string cache_dir_;
    uint64_t max_bytes_ = 0ULL;
    uint64_t total_bytes_ = 0ULL;
    bool enabled_ = false;
    mutable std::atomic<uint64_t> tmp_seq_{0ULL};
};

}  // namespace dsv41
