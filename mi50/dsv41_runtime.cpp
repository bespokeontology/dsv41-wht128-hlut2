// Pure-C++17 Heterogeneous Runtime for DeepSeek-V4.1-Flash-1017 (Wave 7B Canonical)
//   - 2x Tenstorrent Blackhole P150A (64 GiB GDDR6 + 247.82 MiB pure active L1, TP2 via zero-SDMA /dsv41_p150_ipc): HLUT2 Routed Experts [216..383]
//   - 3x AMD Instinct MI50 (96 GiB HBM2, TP3): CSA2 Sparse Attention, Strict 20L CED Prefill,
//     mHC Sinkhorn, Shared Experts, HLUT2 Routed Experts [0..215] + MTP [0..127], 3-Stage DSpark (M=6 Chain / M=8 Dirac Causal Tree)
//   - EPYC 384 GiB DDR4 + Dual-NVMe: Pre-Faulted 189 GiB Engram + RewindBank + 128-tok RadixBlockCache + SWA Replay
// Zero Python anywhere.

#include "dsv41_runtime.hpp"
#include "../src/dsv41_rarebal.hpp"
#include "dsv41_kolmogorov_tree_verifier.hpp"
#include "dsv41_think_loop_guard.hpp"
#include "dsv41_tree_spec.hpp"

#include <immintrin.h>
#include <sys/stat.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dsv41 {

namespace {

static constexpr int kTpSize            = 3;
static constexpr int kVocabSize         = 129280;
static constexpr int kDim               = 5120;
static constexpr int kHcMult            = 4;
static constexpr int kHcMixDim          = 24;
static constexpr int kHcSinkhornIters   = 20;
static constexpr float kNormEps         = 1e-20f;
static constexpr float kHcEps           = 1e-6f;
static constexpr int kNumLayers         = 40;
static constexpr int kNumMtpLayers      = 3;
static constexpr int kTotalBlocks       = kNumLayers + kNumMtpLayers;  // 43
static constexpr int kWindowSize        = 128;
static constexpr int kHeadDim           = 512;
static constexpr int kQLoraRank         = 1280;
static constexpr int kQkvConcatRows     = kQLoraRank + kHeadDim;      // 1792
static constexpr int kOLoraRank         = 1024;
static constexpr int kIndexHeadDim      = 128;
static constexpr int kIndexTopk         = 512;
static constexpr int kCandSourceLayer   = 20;
static constexpr int kSharedInterTp     = 768;   // 2304 / 3
static constexpr int kBackboneExperts   = 384;
static constexpr int kBackboneTopk      = 6;
static constexpr int kMi50BackboneExp   = 216;   // [0..215] on 3x MI50 TP3, [216..383] on 2x P150A TP2
static constexpr int kMtpExperts        = 128;
static constexpr int kMtpTopk           = 3;
static constexpr float kRouteScale      = 1.5f;
static constexpr int kDSparkBlockSize   = 5;     // M = 1 + 5 = 6 chain verify tokens!
static constexpr int kTree8NumNodes     = 8;     // Width-[3,2,1,1] causal draft tree (M = 8 verify nodes)
static constexpr uint64_t kTree8AncMask64     = 0x9345231309050301ULL;
static constexpr uint32_t kTree8DepthNibbles  = 0x32221110u;
static constexpr uint32_t kTree8ParentNibbles = 0x4211000Fu;
static constexpr float    kTree8Top1ConfThresh = 0.72f;
static constexpr std::array<int32_t, kTree8NumNodes>  kTree8Parents = {-1, 0, 0, 0, 1, 1, 2, 4};
static constexpr std::array<uint32_t, kTree8NumNodes> kTree8Depths  = {0u, 1u, 1u, 1u, 2u, 2u, 2u, 3u};
static constexpr int32_t kDSparkNoiseId = 128799;
static constexpr int kMarkovRank        = 256;
static constexpr int kMaxChunkM         = 512;   // Dumitrescu MR vs MP block law: M=512 fits P150A L1 MR!
static constexpr int kMaxDecChunkM      = 512;   // Decoder M=512 (keeps d_cand_block_mask at 16 MiB!)
static constexpr int kMaxDecM           = 320;   // Padded multiple of 64 for decoder-only taps/heads!
static constexpr int kF32RingMaxM       = 64;    // Dumitrescu mixed-precision FP32 ring + FP32 mHC post -> 1 RNE BF16!
static constexpr size_t kPartF32Floats  = 512u * 25600u; // 13,107,200 floats = 50.00 MiB

// Exact 890 B/tok 128-token RadixBlock layout across Slots 0..2 (r=2, 64 slots) and Slot 3 (r=1, 128 slots):
//   Per r=2 slot (0..2): fp4[64*256=16384] + e4m3[64*32=2048] + ik_fp4[64*64=4096] + ik_ue8[64*4=256] = 22,784 B
//   Slot 3 (r=1):        fp4[128*256=32768] + e4m3[128*32=4096] + ik_fp4[128*64=8192] + ik_ue8[128*4=512] = 45,568 B
//   Total = 3 * 22,784 + 45,568 = 113,920 B (exact 890 B/token!)
static constexpr int kRadixBlockTokens      = 128;
static constexpr size_t kRadixR2SlotBytes   = 16384u + 2048u + 4096u + 256u;   // 22,784 B
static constexpr size_t kRadixR1SlotBytes   = 32768u + 4096u + 8192u + 512u;   // 45,568 B
static constexpr size_t kRadixBlockBytes    = 3u * kRadixR2SlotBytes + kRadixR1SlotBytes; // 113,920 B
static constexpr size_t kMaxDdr4RadixBlocks = 65536u;                          // Up to 8.38M tokens (~7.1 GiB max)
static constexpr size_t kWinKvSlabElems     = size_t(kTotalBlocks) * kWindowSize * kHeadDim; // 2,818,048 BF16 (5.375 MiB)
static constexpr size_t kCompStateSlabElems = 3u * 4u * kHeadDim;              // 6,144 floats (24 KiB)

static constexpr int kCompressRatios[kTotalBlocks] = {
    0, 0,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0
};

inline bool is_kv_source_layer(int l) {
    return l == 2 || l == 8 || l == 14 || l == 20;
}

inline int kv_source_slot_for_layer(int l) {
    if (l < 2 || l >= kNumLayers) return -1;
    if (l < 8)  return 0;  // layer 2 (r=2)
    if (l < 14) return 1;  // layer 8 (r=2)
    if (l < 20) return 2;  // layer 14 (r=2)
    return 3;              // layer 20 (r=1)
}

inline bool is_index_source_layer(int l) {
    return l == 2 || l == 8 || l == 14 || l == 20 || l == 24 || l == 28 || l == 32 || l == 36;
}

inline float bf16_bits_to_f32_host(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

inline uint16_t f32_to_bf16_bits_host(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(uint32_t));
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

static std::string trim_line(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) --b;
    return s.substr(a, b - a);
}

void copy_h2d_fast(void* d_dst, const void* h_src, size_t nbytes) {
    thread_local uint8_t* h_stage = nullptr;
    static constexpr size_t kStageBytes = 32u * 1024u * 1024u;
    if (!h_stage) {
        void* p = nullptr;
        if (hipHostMalloc(&p, kStageBytes, hipHostMallocDefault) == hipSuccess) {
            h_stage = static_cast<uint8_t*>(p);
        } else {
            (void)hipGetLastError();
        }
    }
    if (!h_stage) {
        DSV41_HIP_CHECK(hipMemcpy(d_dst, h_src, nbytes, hipMemcpyHostToDevice));
        return;
    }
    const auto* src_b = static_cast<const uint8_t*>(h_src);
    auto* dst_b = static_cast<uint8_t*>(d_dst);
    for (size_t off = 0; off < nbytes; off += kStageBytes) {
        const size_t chunk = std::min(kStageBytes, nbytes - off);
        std::memcpy(h_stage, src_b + off, chunk);
        DSV41_HIP_CHECK(hipMemcpyAsync(dst_b + off, h_stage, chunk, hipMemcpyHostToDevice, nullptr));
        DSV41_HIP_CHECK(hipStreamSynchronize(nullptr));
    }
}

template <typename T>
T* upload_tensor_to_gpu(const SafetensorsMmap& st, const std::string& name, std::vector<void*>& alloc_tracker) {
    const auto& ent = st.at(name);
    const size_t nbytes = static_cast<size_t>(ent.byte_size());
    void* d_ptr = nullptr;
    DSV41_HIP_CHECK(hipMalloc(&d_ptr, nbytes));
    alloc_tracker.push_back(d_ptr);
    copy_h2d_fast(d_ptr, st.tensor_bytes(name), nbytes);
    return static_cast<T*>(d_ptr);
}

template <typename T>
T* upload_concat2_to_gpu(const SafetensorsMmap& st, const std::string& name_a, const std::string& name_b,
                         std::vector<void*>& alloc_tracker, size_t* out_bytes_a = nullptr) {
    const auto& ea = st.at(name_a);
    const auto& eb = st.at(name_b);
    const size_t ba = static_cast<size_t>(ea.byte_size());
    const size_t bb = static_cast<size_t>(eb.byte_size());
    void* d_ptr = nullptr;
    DSV41_HIP_CHECK(hipMalloc(&d_ptr, ba + bb));
    alloc_tracker.push_back(d_ptr);
    copy_h2d_fast(d_ptr, st.tensor_bytes(name_a), ba);
    copy_h2d_fast(static_cast<uint8_t*>(d_ptr) + ba, st.tensor_bytes(name_b), bb);
    if (out_bytes_a) *out_bytes_a = ba;
    return static_cast<T*>(d_ptr);
}

template <typename T>
T* alloc_gpu_buf(size_t count, std::vector<void*>& alloc_tracker, bool zero_init = true) {
    const size_t nbytes = std::max<size_t>(count * sizeof(T), 64u);
    void* d_ptr = nullptr;
    DSV41_HIP_CHECK(hipMalloc(&d_ptr, nbytes));
    alloc_tracker.push_back(d_ptr);
    if (zero_init) {
        DSV41_HIP_CHECK(hipMemset(d_ptr, 0, nbytes));
    }
    return static_cast<T*>(d_ptr);
}

struct RadixKey128 {
    uint64_t lo = 0;
    uint64_t hi = 0;
    bool operator==(const RadixKey128& o) const noexcept { return lo == o.lo && hi == o.hi; }
};

struct RadixKeyHash {
    size_t operator()(const RadixKey128& k) const noexcept {
        return static_cast<size_t>(k.lo ^ (k.hi * 0x9E3779B97F4A7C15ULL));
    }
};

inline RadixKey128 step_radix_hash(RadixKey128 parent, const int32_t* toks128) noexcept {
    uint64_t h1 = parent.lo ^ 0xCBF29CE484222325ULL;
    uint64_t h2 = parent.hi ^ 0x84222325CBF29CE4ULL;
    for (int i = 0; i < kRadixBlockTokens; ++i) {
        const uint64_t v = static_cast<uint32_t>(toks128[i]);
        h1 ^= v + 0x9E3779B97F4A7C15ULL + (h1 << 6) + (h1 >> 2);
        h1 *= 0x100000001B3ULL;
        h2 ^= (v * 0xBF58476D1CE4E5B9ULL) + (h2 << 7) + (h2 >> 3);
        h2 *= 0x94D049BB133111EBULL;
    }
    return RadixKey128{h1, h2};
}

struct RadixBlockEntry {
    RadixKey128 key{};
    std::array<int32_t, kRadixBlockTokens> tokens{};
    alignas(64) std::array<uint8_t, kRadixBlockBytes> data{};
};

struct RewindSnapshot {
    int pos = 0;
    int win_valid_from_20_39 = 0;
    bool mtp_seeded = false;
    std::vector<int32_t> tokens;
    std::vector<uint16_t> win_kv_slab;    // [43 * 128 * 512] BF16 (5.375 MiB)
    std::vector<float> comp_state_slab;   // [3 * 4 * 512] F32 (24 KiB)
};

struct EngramGpuWeights {
    uint16_t* d_q_w   = nullptr;  // [4, 5120] BF16
    uint16_t* d_k_w   = nullptr;  // [4, 5120] BF16
    uint8_t*  d_wkv_w = nullptr;  // [25600, 2048] FP8
    uint8_t*  d_wkv_s = nullptr;  // [800, 64] UE8M0
};

struct BlockGpuWeights {
    int block_id = 0;         // 0..39 backbone, 40..42 MTP
    int h_cnt = 0;            // 24 on tp0/1, 16 on tp2
    int g_cnt = 0;            // 3 on tp0/1, 2 on tp2
    int compress_ratio = 0;   // 0, 2, or 1
    bool is_kv_source = false;
    bool is_index_source = false;
    int ih_cnt = 0;           // 11 on tp0/1, 10 on tp2
    int n_routed_experts = 0; // 384 or 128
    int top_k = 0;            // 6 or 3
    int mi50_experts = 0;     // 216 or 128

    // Norms + Hyper-Connections
    uint16_t* d_attn_norm = nullptr;
    uint16_t* d_ffn_norm  = nullptr;
    float* d_hc_attn_base  = nullptr;
    float* d_hc_attn_fn    = nullptr;
    float* d_hc_attn_scale = nullptr;
    float* d_hc_ffn_base   = nullptr;
    float* d_hc_ffn_fn     = nullptr;
    float* d_hc_ffn_scale  = nullptr;

    // Attention (wq_a + wkv concatenated into d_wqkv_w [1792, 5120]!)
    uint16_t* d_q_norm    = nullptr;
    uint16_t* d_kv_norm   = nullptr;
    uint8_t*  d_wqkv_w    = nullptr;
    uint8_t*  d_wqkv_s    = nullptr;
    uint8_t*  d_wq_a_w    = nullptr;
    uint8_t*  d_wq_a_s    = nullptr;
    uint8_t*  d_wkv_w     = nullptr;
    uint8_t*  d_wkv_s     = nullptr;
    float*    d_attn_sink = nullptr;
    uint8_t*  d_wq_b_w    = nullptr;
    uint8_t*  d_wq_b_s    = nullptr;
    uint8_t*  d_wo_a_w    = nullptr;
    uint8_t*  d_wo_a_s    = nullptr;
    uint8_t*  d_wo_b_w    = nullptr;
    uint8_t*  d_wo_b_s    = nullptr;

    // Optional Compressor (layers 2, 8, 14, 20)
    uint16_t* d_comp_norm  = nullptr;
    uint16_t* d_comp_wkv   = nullptr;
    uint16_t* d_comp_wgate = nullptr;  // r=2 only (2, 8, 14)

    // Optional Indexer (layers 2, 8, 14, 20, 24, 28, 32, 36)
    uint8_t*  d_idx_wq_b_w       = nullptr;
    uint8_t*  d_idx_wq_b_s       = nullptr;
    uint16_t* d_idx_weights_proj = nullptr;
    uint16_t* d_idx_wk           = nullptr;  // kv_source only
    uint16_t* d_idx_k_norm       = nullptr;  // kv_source only

    // MoE Gate + Shared Expert (w1 + w3 concatenated into d_sh_w13_w [1536, 5120]!) + HLUT2 Routed Experts
    uint16_t* d_gate_w       = nullptr;
    float*    d_gate_bias    = nullptr;
    float*    d_gate_bias_vl = nullptr;
    uint8_t*  d_sh_w13_w     = nullptr;
    uint8_t*  d_sh_w13_s     = nullptr;
    uint8_t*  d_sh_w2_w      = nullptr;
    uint8_t*  d_sh_w2_s      = nullptr;

    uint64_t* d_signs_h         = nullptr;
    uint64_t* d_signs_i         = nullptr;
    uint32_t* d_hlut2_gate_codes = nullptr;
    uint16_t* d_hlut2_gate_sc   = nullptr;
    float*    d_hlut2_gate_cent = nullptr;
    uint32_t* d_hlut2_up_codes  = nullptr;
    uint16_t* d_hlut2_up_sc     = nullptr;
    float*    d_hlut2_up_cent   = nullptr;
    uint32_t* d_hlut2_down_codes = nullptr;
    uint16_t* d_hlut2_down_sc   = nullptr;
    float*    d_hlut2_down_cent = nullptr;

    // MTP extras (markov_head.embed.weight kept in host DDR4, saving 21.05 MiB HBM2/GPU!)
    uint16_t* d_main_norm       = nullptr;  // mtp_0
    uint8_t*  d_main_proj_w     = nullptr;  // mtp_0 [5120, 5120]
    uint8_t*  d_main_proj_s     = nullptr;  // mtp_0 [160, 160]
    uint16_t* d_mtp_final_norm  = nullptr;  // mtp_2 [5120]
    uint16_t* d_conf_proj_w     = nullptr;  // mtp_2 [1, 5376]
    uint16_t* d_markov_head_w   = nullptr;  // mtp_2 [vocab_rows, 256]
};

struct TpRankContext {
    int tp = 0;
    int vocab_offset = 0;
    int vocab_rows = 0;
    int cand_block_stride = 0;
    hipStream_t stream           = nullptr;
    hipStream_t router_stream    = nullptr;
    hipStream_t engram_stream    = nullptr;
    hipStream_t keep_warm_stream = nullptr;
    uint32_t*   h_dpm_flag       = nullptr;
    uint32_t*   d_dpm_flag       = nullptr;
    uint32_t*   d_dpm_uc_scratch = nullptr;
    hipEvent_t ev_ready       = nullptr;
    hipEvent_t ev_done        = nullptr;
    hipEvent_t ev_x_ready     = nullptr;
    hipEvent_t ev_router_done = nullptr;
    hipEvent_t ev_engram_ready[2]{};
    hipEvent_t ev_engram_consumed[2]{};
    hipEvent_t ev_radix_stage[2]{};
    hipEvent_t ev_ar_pushed[4]{};
    hipEvent_t ev_ar_consumed[4]{};
    std::vector<void*> device_allocs;
    std::vector<void*> host_allocs;

    // Top-level Embed / Head / Engram / Blocks
    uint16_t* d_embed_w = nullptr;
    uint16_t* d_norm_w  = nullptr;
    uint16_t* d_head_w  = nullptr;
    EngramGpuWeights engram[2];
    BlockGpuWeights blocks[kTotalBlocks];

    // Persistent KV Caches (contiguous slabs for single-DMA RewindBank!)
    uint16_t* d_win_kv_slab = nullptr;
    uint16_t* d_win_kv[kTotalBlocks]{};
    uint16_t* d_last_win_kv_slab = nullptr;
    uint16_t* d_last_win_kv_bf16[kNumLayers]{};
    float*    d_comp_kv_state_slab = nullptr;
    float*    d_comp_kv_state[3]{};      // [2, 512] F32 for r=2 layers (2, 8, 14)
    float*    d_comp_score_state[3]{};   // [2, 512] F32 for r=2 layers (2, 8, 14)
    float*    d_comp_kv_last[3]{};       // [8, 512] F32 saved during M<=6 verify for rollback
    float*    d_comp_score_last[3]{};    // [8, 512] F32 saved during M<=6 verify for rollback
    uint8_t*  d_comp_kv_fp4[4]{};        // [max_comp, 256] U8 (slots 0..2: r=2, slot 3: r=1)
    uint8_t*  d_comp_kv_e4m3[4]{};       // [max_comp, 32]  U8
    uint8_t*  d_index_k_fp4[4]{};        // [max_comp, 64]  U8 (68 B/slot packed Indexer K!)
    uint8_t*  d_index_k_ue8m0[4]{};      // [max_comp, 4]   U8

    // Workspace Scratch (sized for kMaxChunkM = 512, exact multiple of 64!)
    int32_t*  d_token_ids = nullptr;
    uint16_t* d_h_bf16[2]{};             // [M, 4, 5120] ping-pong
    float*    d_pre_mix_f32[2]{};        // [M, 4] ping-pong
    float*    d_post_mix_f32 = nullptr;  // [M, 4]
    float*    d_comb_mix_f32 = nullptr;  // [M, 16]
    float*    d_hc_rsqrt_f32 = nullptr;  // [M]
    float*    d_hc_mixes_f32 = nullptr;  // [M, 24]

    uint16_t* d_x_bf16    = nullptr;     // [M, 5120]
    uint16_t* d_sub_bf16  = nullptr;     // [M, 5120]
    uint16_t* d_part_bf16 = nullptr;     // [M, 5120]
    float*    d_part_f32  = nullptr;     // [M, 25600] (50.00 MiB — aliased with Indexer part, SplitK, MoE part)
    float*    d_sum_f32   = nullptr;     // [M, 25600] (50.00 MiB — aliased with Indexer sum, MoE sum)

    uint8_t*  d_fp8_act   = nullptr;     // [M, 12288]
    uint8_t*  d_fp8_scale = nullptr;     // [M, 384]

    uint16_t* d_qkv_bf16    = nullptr;   // [M, 1792]
    uint16_t* d_qr_bf16     = nullptr;   // [M, 1280]
    uint16_t* d_q_bf16      = nullptr;   // [M, 24 * 512]
    uint16_t* d_kv_cur_bf16 = nullptr;   // [M, 512]
    uint16_t* d_o_bf16      = nullptr;   // [M, 24 * 512]
    uint16_t* d_wo_mid_bf16 = nullptr;   // [M, 3072]

    int32_t*  d_comp_topk_idxs  = nullptr; // [M, 512]
    uint8_t*  d_cand_block_mask = nullptr; // [M, cand_block_stride] (1 byte per 8 compressed keys!)

    float*    d_comp_kv_tmp_f32    = nullptr; // [M, 512]
    float*    d_comp_score_tmp_f32 = nullptr; // [M, 512]
    uint16_t* d_latent_bf16        = nullptr; // [M, 512]

    uint16_t* d_idx_q_bf16       = nullptr; // [M, 12 * 128]
    uint16_t* d_idx_k_tmp_bf16   = nullptr; // [M, 128]
    uint16_t* d_idx_weights_bf16 = nullptr; // [M, 16]

    uint16_t* d_engram_in_bf16[2]{};        // [2][M, 2048] double-buffered async prefetch

    float*          d_gate_logits_f32 = nullptr; // [M, 384]
    int32_t*        d_topk_ids        = nullptr; // [M, 6]
    float*          d_topk_weights    = nullptr; // [M, 6]
    uint16_t*       d_sh_gateup_bf16  = nullptr; // [M, 1536]
    uint16_t*       d_x_rot_f16       = nullptr; // [M, 5120]
    Mi50RoutedPair* d_routed_pairs    = nullptr; // [M * 6]
    int32_t*        d_active_experts  = nullptr; // [256]
    int32_t*        d_csr_offsets     = nullptr; // [257]
    uint16_t*       d_inter_bf16      = nullptr; // [M * 6, 768]
    uint16_t*       d_inter_rot_f16   = nullptr; // [M * 6, 768]
    float*          d_moe_part_f32    = nullptr; // aliased to d_part_f32
    float*          d_moe_sum_f32     = nullptr; // aliased to d_sum_f32
    uint16_t*       d_p150_out_bf16   = nullptr; // [M, 5120]
    float*          d_ar_recv[4][2]{};           // [slot=0..3][peer=0..1] fine-grained coherent P2P posted-write ring

    // DSpark target hidden & heads (padded to kMaxDecM = 320, multiple of 64!)
    uint16_t* d_target_mean_bf16   = nullptr; // [kMaxDecM, 5120]
    uint16_t* d_main_x_bf16        = nullptr; // [kMaxDecM, 5120]
    float*    d_logits_part_f32    = nullptr; // [64, vocab_rows]
    uint16_t* d_markov_sum_bf16    = nullptr; // [64, 256]
    float*    d_markov_logits_f32  = nullptr; // [64, vocab_rows]
    uint16_t* d_conf_in_bf16       = nullptr; // [64, 5376]
    float*    d_conf_out_f32       = nullptr; // [64]

    // Pinned host staging buffers
    uint16_t*       h_reduce_bf16  = nullptr;
    float*          h_reduce_f32   = nullptr;
    float*          h_logits_part  = nullptr; // [8, vocab_rows]
    uint16_t*       h_engram_bf16[2]{};       // [2][M, 2048]
    int32_t*        h_topk_ids     = nullptr; // [M * 6]
    float*          h_topk_w       = nullptr; // [M * 6]
    uint16_t*       h_x_moe_bf16   = nullptr; // [M * 5120]
    uint8_t*        h_radix_stage  = nullptr; // [8 MiB] pinned Radix DMA buffer 0 on tp=0
    uint8_t*        h_radix_stage2 = nullptr; // [8 MiB] pinned Radix DMA buffer 1 on tp=0
    Mi50RoutedPair* h_routed_pairs = nullptr; // [M * 6] pinned routed pairs on tp=0
    int32_t*        h_active_exp   = nullptr; // [256] pinned active experts on tp=0
    int32_t*        h_csr_offsets  = nullptr; // [257] pinned CSR offsets on tp=0
};

}  // namespace
struct Dsv41Runtime::Impl {
    int max_seq_len = 524288;
    bool p2p_enabled = false;
    uint32_t ar_ping = 0u;
    std::array<TpRankContext, kTpSize> ctx;
    Dsv41EngramEngine engram_cpu;
    Dsv41P150ShmMapping p150_shm;
    bool p150_shm_pinned = false;
    std::array<uint16_t*, kTpSize> d_p150_shm_y_out_bf16{nullptr, nullptr, nullptr};
    uint32_t p150_seq_counter = 0u;
    HostRingAllReduceSlot* h_ar_ring = nullptr;
    uint64_t host_ar_seq = 0u;

    // Coherent mapped pinned DDR4 Markov embedding [129280, 256] BF16 (63.1 MiB in DDR4 -> 0 B HBM2!) + GPU DSpark slot
    uint16_t* h_markov_embed_mapped = nullptr;
    DsparkGpuSampleSlot* h_dspark_ctrl = nullptr;
    uint64_t dspark_sample_seq = 0u;
    std::vector<uint16_t> markov_embed_host_bf16;
    std::mutex markov_host_mtx;

    // Prefix Reuse Hierarchy:
    //   Tier 0: Resident GPU KV [0..cached_tokens.size()) (0.0 ms exact continuation)
    //   Tier 1: RewindBank ring of 32 full SWA + r=2 snapshots (< 0.5 ms single-slab DMA)
    //   Tier 2: DDR4 RadixBlockCache (65,536 x 128-tok blocks @ 890 B/tok) + Dual-NVMe GlobalPrefixCache + SWA Replay
    std::vector<int32_t> cached_tokens;
    int win_valid_from_20_39 = 0;
    bool mtp_window_seeded = false;
    int exported_radix_blocks = 0;
    RadixKey128 exported_radix_key{};

    std::deque<RewindSnapshot> rewind_bank;
    std::unordered_map<RadixKey128, std::shared_ptr<RadixBlockEntry>, RadixKeyHash> radix_map;
    std::deque<RadixKey128> radix_fifo;
    std::string nvme_radix_dirs[2];

    // Async Dual-NVMe Radix writer thread
    std::mutex nvme_mtx;
    std::condition_variable nvme_cv;
    std::deque<std::shared_ptr<RadixBlockEntry>> nvme_write_queue;
    bool stop_workers = false;
    std::thread nvme_writer_thread;

    // Async Engram CPU gather + H2D prefetch worker thread
    struct EngramJob {
        int slot = 0;
        int engram_idx = 0;
        int M = 0;
    };
    std::mutex eg_mtx;
    std::condition_variable eg_cv;
    std::condition_variable eg_done_cv;
    std::deque<EngramJob> eg_queue;
    bool eg_slot_ready[2]{false, false};
    std::thread eg_worker_thread;

    // Reusable host scratch buffers
    std::vector<int64_t> engram_hashes_buf;
    std::vector<Mi50RoutedPair> mi50_pairs_buf;
    std::vector<Mi50RoutedPair> mi50_sorted_pairs_buf;
    std::vector<int32_t> mi50_active_exp_buf;
    std::vector<int32_t> mi50_csr_buf;
    std::array<float, 8 * kVocabSize> host_full_logits{};
    std::vector<double> host_tree_probs_f64;
    std::array<std::vector<float>, kDSparkBlockSize> draft_q_probs;
    std::array<float, kDSparkBlockSize> draft_q_selected{};
    std::array<bool, kDSparkBlockSize> draft_q_materialized{};
    std::vector<RadixKey128> gpu_slot_radix_keys;
    double accum_p150_dev_ms = 0.0;
    double accum_attn_ms = 0.0;
    double accum_hsi_ms = 0.0;
    double accum_mi50_moe_ms = 0.0;
    double accum_p150_wait_ms = 0.0;
    double accum_allreduce_ms = 0.0;
    double accum_engram_wait_ms = 0.0;
    double sub_att_qkv_ms = 0.0;
    double sub_att_sparse_ms = 0.0;
    double sub_att_wo_ms = 0.0;
    double sub_att_ar_ms = 0.0;
    double sub_moe_sh_ms = 0.0;
    double sub_moe_hlut_ms = 0.0;
    double sub_moe_ar_ms = 0.0;
    double sub_mhc_ms = 0.0;
    double sub_commit_ms = 0.0;
    bool sync_timers = false;
    std::unique_ptr<SafetensorsMmap> rarebal_overlay_mmap;
    std::atomic<int> rarebal_mi50_patched_slots{0};
    std::unique_ptr<RarebalLiveCollector> rarebal_collector;

    void reset_stage_timers() {
        accum_p150_dev_ms = 0.0;
        accum_attn_ms = 0.0;
        accum_hsi_ms = 0.0;
        accum_mi50_moe_ms = 0.0;
        accum_p150_wait_ms = 0.0;
        accum_allreduce_ms = 0.0;
        accum_engram_wait_ms = 0.0;
        sub_att_qkv_ms = 0.0;
        sub_att_sparse_ms = 0.0;
        sub_att_wo_ms = 0.0;
        sub_att_ar_ms = 0.0;
        sub_moe_sh_ms = 0.0;
        sub_moe_hlut_ms = 0.0;
        sub_moe_ar_ms = 0.0;
        sub_mhc_ms = 0.0;
        sub_commit_ms = 0.0;
    }

    inline void sync_tp0_if_profiling() {
        if (sync_timers) {
            DSV41_HIP_CHECK(hipSetDevice(0));
            DSV41_HIP_CHECK(hipStreamSynchronize(ctx[0].stream));
        }
    }

    ~Impl() {
        {
            std::lock_guard<std::mutex> lk1(nvme_mtx);
            std::lock_guard<std::mutex> lk2(eg_mtx);
            stop_workers = true;
        }
        nvme_cv.notify_all();
        eg_cv.notify_all();
        if (nvme_writer_thread.joinable()) nvme_writer_thread.join();
        if (eg_worker_thread.joinable()) eg_worker_thread.join();

        for (int tp = 0; tp < kTpSize; ++tp) {
            (void)hipSetDevice(tp);
            if (ctx[tp].h_dpm_flag)    *ctx[tp].h_dpm_flag = 0u;
            if (ctx[tp].keep_warm_stream) (void)hipStreamSynchronize(ctx[tp].keep_warm_stream);
            if (ctx[tp].stream)        (void)hipStreamSynchronize(ctx[tp].stream);
            if (ctx[tp].router_stream) (void)hipStreamSynchronize(ctx[tp].router_stream);
            if (ctx[tp].engram_stream) (void)hipStreamSynchronize(ctx[tp].engram_stream);
            if (ctx[tp].ev_ready)       (void)hipEventDestroy(ctx[tp].ev_ready);
            if (ctx[tp].ev_done)        (void)hipEventDestroy(ctx[tp].ev_done);
            if (ctx[tp].ev_x_ready)     (void)hipEventDestroy(ctx[tp].ev_x_ready);
            if (ctx[tp].ev_router_done) (void)hipEventDestroy(ctx[tp].ev_router_done);
            for (int s = 0; s < 2; ++s) {
                if (ctx[tp].ev_engram_ready[s])    (void)hipEventDestroy(ctx[tp].ev_engram_ready[s]);
                if (ctx[tp].ev_engram_consumed[s]) (void)hipEventDestroy(ctx[tp].ev_engram_consumed[s]);
                if (ctx[tp].ev_radix_stage[s])     (void)hipEventDestroy(ctx[tp].ev_radix_stage[s]);
            }
            for (int s = 0; s < 4; ++s) {
                if (ctx[tp].ev_ar_pushed[s])   (void)hipEventDestroy(ctx[tp].ev_ar_pushed[s]);
                if (ctx[tp].ev_ar_consumed[s]) (void)hipEventDestroy(ctx[tp].ev_ar_consumed[s]);
            }
            if (ctx[tp].keep_warm_stream) (void)hipStreamDestroy(ctx[tp].keep_warm_stream);
            if (ctx[tp].router_stream) (void)hipStreamDestroy(ctx[tp].router_stream);
            if (ctx[tp].engram_stream) (void)hipStreamDestroy(ctx[tp].engram_stream);
            if (ctx[tp].stream)        (void)hipStreamDestroy(ctx[tp].stream);
            for (void* p : ctx[tp].device_allocs) {
                if (p) (void)hipFree(p);
            }
            for (void* p : ctx[tp].host_allocs) {
                if (p) (void)hipHostFree(p);
            }
        }
        if (p150_shm_pinned && p150_shm.ptr) {
            d_p150_shm_y_out_bf16.fill(nullptr);
            (void)hipHostUnregister(p150_shm.ptr);
            p150_shm_pinned = false;
        }
        if (h_dspark_ctrl) {
            (void)hipHostFree(h_dspark_ctrl);
            h_dspark_ctrl = nullptr;
        }
        if (h_markov_embed_mapped) {
            (void)hipHostFree(h_markov_embed_mapped);
            h_markov_embed_mapped = nullptr;
        }
        if (h_ar_ring) {
            (void)hipHostFree(h_ar_ring);
            h_ar_ring = nullptr;
        }
    }

    void start_background_workers() {
        const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                         ? std::string(std::getenv("DSV41_ROOT"))
                                         : std::string(".");
        nvme_radix_dirs[0] = (std::getenv("DSV41_RADIX_DIR_0") != nullptr)
                                 ? std::string(std::getenv("DSV41_RADIX_DIR_0"))
                                 : (root_env + "/radix_cache_nvme0");
        nvme_radix_dirs[1] = (std::getenv("DSV41_RADIX_DIR_1") != nullptr)
                                 ? std::string(std::getenv("DSV41_RADIX_DIR_1"))
                                 : (root_env + "/radix_cache_nvme1");
        (void)::mkdir(nvme_radix_dirs[0].c_str(), 0775);
        (void)::mkdir(nvme_radix_dirs[1].c_str(), 0775);

        nvme_writer_thread = std::thread([this]() {
            while (true) {
                std::shared_ptr<RadixBlockEntry> item;
                {
                    std::unique_lock<std::mutex> lk(nvme_mtx);
                    nvme_cv.wait(lk, [&]() { return stop_workers || !nvme_write_queue.empty(); });
                    if (stop_workers && nvme_write_queue.empty()) break;
                    item = std::move(nvme_write_queue.front());
                    nvme_write_queue.pop_front();
                }
                if (!item) continue;
                const int drive = int(item->key.lo & 1ULL);
                char fn[256];
                std::snprintf(fn, sizeof(fn), "%s/%016llx_%016llx.rblk",
                              nvme_radix_dirs[drive].c_str(),
                              static_cast<unsigned long long>(item->key.lo),
                              static_cast<unsigned long long>(item->key.hi));
                std::ifstream chk(fn, std::ios::binary);
                if (chk.good()) continue;
                const std::string tmp_fn = std::string(fn) + ".tmp";
                std::ofstream out(tmp_fn, std::ios::binary | std::ios::trunc);
                if (out) {
                    out.write(reinterpret_cast<const char*>(item->tokens.data()), sizeof(item->tokens));
                    out.write(reinterpret_cast<const char*>(item->data.data()), item->data.size());
                    out.close();
                    if (out.good()) (void)::rename(tmp_fn.c_str(), fn);
                }
            }
        });

        eg_worker_thread = std::thread([this]() {
            while (true) {
                EngramJob job;
                {
                    std::unique_lock<std::mutex> lk(eg_mtx);
                    eg_cv.wait(lk, [&]() { return stop_workers || !eg_queue.empty(); });
                    if (stop_workers && eg_queue.empty()) break;
                    job = eg_queue.front();
                    eg_queue.pop_front();
                }
                if (engram_cpu.has_table_shard(job.engram_idx) && job.M > 0) {
                    for (int tp = 0; tp < 3; ++tp) {
                        (void)hipSetDevice(tp);
                        (void)hipEventSynchronize(ctx[tp].ev_engram_consumed[job.slot]);
                    }
                    engram_cpu.gather_dequant_tp3(
                        job.engram_idx, engram_hashes_buf.data(), job.M, nullptr,
                        ctx[0].h_engram_bf16[job.slot],
                        ctx[1].h_engram_bf16[job.slot],
                        ctx[2].h_engram_bf16[job.slot]);
                    for (int tp = 0; tp < 3; ++tp) {
                        (void)hipSetDevice(tp);
                        auto& c = ctx[tp];
                        (void)hipMemcpyAsync(c.d_engram_in_bf16[job.slot], c.h_engram_bf16[job.slot],
                                             size_t(job.M) * 2048u * sizeof(uint16_t),
                                             hipMemcpyHostToDevice, c.engram_stream);
                        (void)hipEventRecord(c.ev_engram_ready[job.slot], c.engram_stream);
                    }
                }
                {
                    std::lock_guard<std::mutex> lk(eg_mtx);
                    eg_slot_ready[job.slot] = true;
                }
                eg_done_cv.notify_all();
            }
        });
    }

    template <typename T>
    T* alloc_pinned(int tp, size_t count) {
        void* p = nullptr;
        DSV41_HIP_CHECK(hipHostMalloc(&p, std::max<size_t>(count * sizeof(T), 64u), hipHostMallocDefault));
        std::memset(p, 0, std::max<size_t>(count * sizeof(T), 64u));
        ctx[tp].host_allocs.push_back(p);
        return static_cast<T*>(p);
    }

    void reset_r2_compressor_state_gpu(int tp) {
        DSV41_HIP_CHECK(hipSetDevice(tp));
        std::vector<float> init_slab(kCompStateSlabElems, 0.0f);
        for (int s = 0; s < 3; ++s) {
            float* sc = init_slab.data() + size_t(s * 4 + 2) * kHeadDim;
            std::fill(sc, sc + 2 * kHeadDim, -INFINITY);
        }
        DSV41_HIP_CHECK(hipMemcpyAsync(ctx[tp].d_comp_kv_state_slab, init_slab.data(),
                                       kCompStateSlabElems * sizeof(float), hipMemcpyHostToDevice, ctx[tp].stream));
        DSV41_HIP_CHECK(hipStreamSynchronize(ctx[tp].stream));
    }

    void load_block_weights(int tp, int block_id, const std::string& st_path) {
        SafetensorsMmap st(st_path);
        auto& b = ctx[tp].blocks[block_id];
        auto& allocs = ctx[tp].device_allocs;

        b.block_id = block_id;
        b.compress_ratio = kCompressRatios[block_id];
        b.is_kv_source = (block_id < kNumLayers) && is_kv_source_layer(block_id);
        b.is_index_source = (block_id < kNumLayers) && is_index_source_layer(block_id);
        if (block_id < kNumLayers) {
            b.n_routed_experts = kBackboneExperts;
            b.top_k = kBackboneTopk;
            b.mi50_experts = kMi50BackboneExp;
        } else {
            b.n_routed_experts = kMtpExperts;
            b.top_k = kMtpTopk;
            b.mi50_experts = kMtpExperts;
        }

        b.h_cnt = static_cast<int>(st.at("attn.attn_sink").shape.at(0));
        b.g_cnt = b.h_cnt / 8;

        b.d_attn_norm     = upload_tensor_to_gpu<uint16_t>(st, "attn_norm.weight", allocs);
        b.d_ffn_norm      = upload_tensor_to_gpu<uint16_t>(st, "ffn_norm.weight", allocs);
        b.d_hc_attn_base  = upload_tensor_to_gpu<float>(st, "hc_attn_base", allocs);
        b.d_hc_attn_fn    = upload_tensor_to_gpu<float>(st, "hc_attn_fn", allocs);
        b.d_hc_attn_scale = upload_tensor_to_gpu<float>(st, "hc_attn_scale", allocs);
        b.d_hc_ffn_base   = upload_tensor_to_gpu<float>(st, "hc_ffn_base", allocs);
        b.d_hc_ffn_fn     = upload_tensor_to_gpu<float>(st, "hc_ffn_fn", allocs);
        b.d_hc_ffn_scale  = upload_tensor_to_gpu<float>(st, "hc_ffn_scale", allocs);

        b.d_q_norm = upload_tensor_to_gpu<uint16_t>(st, "attn.q_norm.weight", allocs);
        b.d_kv_norm = upload_tensor_to_gpu<uint16_t>(st, "attn.kv_norm.weight", allocs);
        size_t wqa_bytes = 0, wqas_bytes = 0;
        b.d_wqkv_w = upload_concat2_to_gpu<uint8_t>(st, "attn.wq_a.weight", "attn.wkv.weight", allocs, &wqa_bytes);
        b.d_wqkv_s = upload_concat2_to_gpu<uint8_t>(st, "attn.wq_a.scale",  "attn.wkv.scale",  allocs, &wqas_bytes);
        b.d_wq_a_w = b.d_wqkv_w;
        b.d_wq_a_s = b.d_wqkv_s;
        b.d_wkv_w  = b.d_wqkv_w + wqa_bytes;
        b.d_wkv_s  = b.d_wqkv_s + wqas_bytes;

        b.d_attn_sink = upload_tensor_to_gpu<float>(st, "attn.attn_sink", allocs);
        b.d_wq_b_w    = upload_tensor_to_gpu<uint8_t>(st, "attn.wq_b.weight", allocs);
        b.d_wq_b_s    = upload_tensor_to_gpu<uint8_t>(st, "attn.wq_b.scale", allocs);
        b.d_wo_a_w    = upload_tensor_to_gpu<uint8_t>(st, "attn.wo_a.weight", allocs);
        b.d_wo_a_s    = upload_tensor_to_gpu<uint8_t>(st, "attn.wo_a.scale", allocs);
        b.d_wo_b_w    = upload_tensor_to_gpu<uint8_t>(st, "attn.wo_b.weight", allocs);
        b.d_wo_b_s    = upload_tensor_to_gpu<uint8_t>(st, "attn.wo_b.scale", allocs);

        if (b.is_kv_source) {
            b.d_comp_norm = upload_tensor_to_gpu<uint16_t>(st, "attn.compressor.norm.weight", allocs);
            b.d_comp_wkv  = upload_tensor_to_gpu<uint16_t>(st, "attn.compressor.wkv.weight", allocs);
            if (b.compress_ratio > 1) {
                b.d_comp_wgate = upload_tensor_to_gpu<uint16_t>(st, "attn.compressor.wgate.weight", allocs);
            }
        }

        if (b.is_index_source) {
            b.ih_cnt             = static_cast<int>(st.at("attn.indexer.weights_proj.weight").shape.at(0));
            b.d_idx_wq_b_w       = upload_tensor_to_gpu<uint8_t>(st, "attn.indexer.wq_b.weight", allocs);
            b.d_idx_wq_b_s       = upload_tensor_to_gpu<uint8_t>(st, "attn.indexer.wq_b.scale", allocs);
            b.d_idx_weights_proj = upload_tensor_to_gpu<uint16_t>(st, "attn.indexer.weights_proj.weight", allocs);
            if (b.is_kv_source) {
                b.d_idx_wk     = upload_tensor_to_gpu<uint16_t>(st, "attn.indexer.wk.weight", allocs);
                b.d_idx_k_norm = upload_tensor_to_gpu<uint16_t>(st, "attn.indexer.k_norm.weight", allocs);
            }
        }

        b.d_gate_w    = upload_tensor_to_gpu<uint16_t>(st, "ffn.gate.weight", allocs);
        b.d_gate_bias = upload_tensor_to_gpu<float>(st, "ffn.gate.bias", allocs);
        if (st.contains("ffn.gate.bias_vl")) {
            b.d_gate_bias_vl = upload_tensor_to_gpu<float>(st, "ffn.gate.bias_vl", allocs);
        }
        b.d_sh_w13_w = upload_concat2_to_gpu<uint8_t>(st, "ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight", allocs);
        b.d_sh_w13_s = upload_concat2_to_gpu<uint8_t>(st, "ffn.shared_experts.w1.scale",  "ffn.shared_experts.w3.scale",  allocs);
        b.d_sh_w2_w  = upload_tensor_to_gpu<uint8_t>(st, "ffn.shared_experts.w2.weight", allocs);
        b.d_sh_w2_s  = upload_tensor_to_gpu<uint8_t>(st, "ffn.shared_experts.w2.scale", allocs);

        b.d_signs_h          = upload_tensor_to_gpu<uint64_t>(st, "wht.signs_h", allocs);
        b.d_signs_i          = upload_tensor_to_gpu<uint64_t>(st, "wht.signs_i", allocs);
        b.d_hlut2_gate_codes = upload_tensor_to_gpu<uint32_t>(st, "hlut2.gate_codes", allocs);
        b.d_hlut2_gate_sc    = upload_tensor_to_gpu<uint16_t>(st, "hlut2.gate_scales", allocs);
        b.d_hlut2_gate_cent  = upload_tensor_to_gpu<float>(st, "hlut2.gate_centroids", allocs);
        b.d_hlut2_up_codes   = upload_tensor_to_gpu<uint32_t>(st, "hlut2.up_codes", allocs);
        b.d_hlut2_up_sc      = upload_tensor_to_gpu<uint16_t>(st, "hlut2.up_scales", allocs);
        b.d_hlut2_up_cent    = upload_tensor_to_gpu<float>(st, "hlut2.up_centroids", allocs);
        b.d_hlut2_down_codes = upload_tensor_to_gpu<uint32_t>(st, "hlut2.down_codes", allocs);
        b.d_hlut2_down_sc    = upload_tensor_to_gpu<uint16_t>(st, "hlut2.down_scales", allocs);
        b.d_hlut2_down_cent  = upload_tensor_to_gpu<float>(st, "hlut2.down_centroids", allocs);

        if (rarebal_overlay_mmap) {
            const int n_ov = apply_mi50_rarebal_overlay(
                tp, block_id, b.mi50_experts,
                b.d_hlut2_gate_codes, b.d_hlut2_gate_sc, b.d_hlut2_gate_cent,
                b.d_hlut2_up_codes,   b.d_hlut2_up_sc,   b.d_hlut2_up_cent,
                b.d_hlut2_down_codes, b.d_hlut2_down_sc, b.d_hlut2_down_cent,
                *rarebal_overlay_mmap,
                [](void* d_dst, const void* h_src, size_t nb) { copy_h2d_fast(d_dst, h_src, nb); });
            if (tp == 0 && n_ov > 0) {
                rarebal_mi50_patched_slots.fetch_add(n_ov, std::memory_order_relaxed);
            }
        }

        if (block_id == kNumLayers) {
            b.d_main_norm   = upload_tensor_to_gpu<uint16_t>(st, "main_norm.weight", allocs);
            b.d_main_proj_w = upload_tensor_to_gpu<uint8_t>(st, "main_proj.weight", allocs);
            b.d_main_proj_s = upload_tensor_to_gpu<uint8_t>(st, "main_proj.scale", allocs);
        } else if (block_id == kNumLayers + 2) {
            b.d_mtp_final_norm = upload_tensor_to_gpu<uint16_t>(st, "norm.weight", allocs);
            b.d_conf_proj_w    = upload_tensor_to_gpu<uint16_t>(st, "confidence_head.proj.weight", allocs);
            b.d_markov_head_w  = upload_tensor_to_gpu<uint16_t>(st, "markov_head.head.weight", allocs);
            {
                std::lock_guard<std::mutex> lk(markov_host_mtx);
                if (markov_embed_host_bf16.empty()) {
                    markov_embed_host_bf16.assign(size_t(kVocabSize) * kMarkovRank, 0u);
                }
                const auto& me = st.at("markov_head.embed.weight");
                const size_t me_bytes = static_cast<size_t>(me.byte_size());
                std::memcpy(markov_embed_host_bf16.data() + size_t(ctx[tp].vocab_offset) * kMarkovRank,
                            st.tensor_bytes("markov_head.embed.weight"),
                            me_bytes);
                if (h_markov_embed_mapped) {
                    std::memcpy(h_markov_embed_mapped + size_t(ctx[tp].vocab_offset) * kMarkovRank,
                                st.tensor_bytes("markov_head.embed.weight"),
                                me_bytes);
                }
            }
        }
    }

    void init_tp_rank(int tp, const std::string& mi50_root, int vocab_offset_in) {
        DSV41_HIP_CHECK(hipSetDevice(tp));
        auto& c = ctx[tp];
        c.tp = tp;
        c.vocab_offset = vocab_offset_in;
        c.cand_block_stride = (max_seq_len + 7) / 8;

        int lo_pri = 0, hi_pri = 0;
        DSV41_HIP_CHECK(hipDeviceGetStreamPriorityRange(&lo_pri, &hi_pri));
        DSV41_HIP_CHECK(hipStreamCreateWithFlags(&c.stream, hipStreamNonBlocking));
        if (tp == 0) {
            DSV41_HIP_CHECK(hipStreamCreateWithPriority(&c.router_stream, hipStreamNonBlocking, hi_pri));
        }
        DSV41_HIP_CHECK(hipStreamCreateWithFlags(&c.engram_stream, hipStreamNonBlocking));
        DSV41_HIP_CHECK(hipStreamCreateWithPriority(&c.keep_warm_stream, hipStreamNonBlocking, lo_pri));
        {
            void* hp = nullptr;
            DSV41_HIP_CHECK(hipHostMalloc(&hp, 64, hipHostMallocMapped | hipHostMallocCoherent));
            c.host_allocs.push_back(hp);
            c.h_dpm_flag = static_cast<uint32_t*>(hp);
            *c.h_dpm_flag = 0u;
            void* dp = nullptr;
            DSV41_HIP_CHECK(hipHostGetDevicePointer(&dp, hp, 0));
            c.d_dpm_flag = static_cast<uint32_t*>(dp);

            void* ucp = nullptr;
            DSV41_HIP_CHECK(hipExtMallocWithFlags(&ucp, 65536u, hipDeviceMallocUncached));
            c.device_allocs.push_back(ucp);
            c.d_dpm_uc_scratch = static_cast<uint32_t*>(ucp);
            DSV41_HIP_CHECK(hipMemsetAsync(c.d_dpm_uc_scratch, 0, 65536u, c.stream));
        }

        DSV41_HIP_CHECK(hipEventCreateWithFlags(&c.ev_ready, hipEventDisableTiming));
        DSV41_HIP_CHECK(hipEventCreateWithFlags(&c.ev_done, hipEventDisableTiming));
        DSV41_HIP_CHECK(hipEventCreateWithFlags(&c.ev_x_ready, hipEventDisableTiming));
        DSV41_HIP_CHECK(hipEventCreateWithFlags(&c.ev_router_done, hipEventDisableTiming));
        for (int s = 0; s < 2; ++s) {
            DSV41_HIP_CHECK(hipEventCreateWithFlags(&c.ev_engram_ready[s], hipEventDisableTiming));
            DSV41_HIP_CHECK(hipEventCreateWithFlags(&c.ev_engram_consumed[s], hipEventDisableTiming));
            DSV41_HIP_CHECK(hipEventRecord(c.ev_engram_consumed[s], c.stream));
            DSV41_HIP_CHECK(hipEventCreateWithFlags(&c.ev_radix_stage[s], hipEventDisableTiming));
            DSV41_HIP_CHECK(hipEventRecord(c.ev_radix_stage[s], c.stream));
        }
        for (int s = 0; s < 4; ++s) {
            DSV41_HIP_CHECK(hipEventCreateWithFlags(&c.ev_ar_pushed[s], hipEventDisableTiming));
            DSV41_HIP_CHECK(hipEventCreateWithFlags(&c.ev_ar_consumed[s], hipEventDisableTiming));
            DSV41_HIP_CHECK(hipEventRecord(c.ev_ar_consumed[s], c.stream));
        }

        const std::string tp_dir = mi50_root + "/tp" + std::to_string(tp);
        {
            SafetensorsMmap st_emb(tp_dir + "/embed.safetensors");
            c.vocab_rows = static_cast<int>(st_emb.at("embed.weight").shape.at(0));
            c.d_embed_w = upload_tensor_to_gpu<uint16_t>(st_emb, "embed.weight", c.device_allocs);
        }
        {
            SafetensorsMmap st_hd(tp_dir + "/head.safetensors");
            c.d_norm_w = upload_tensor_to_gpu<uint16_t>(st_hd, "norm.weight", c.device_allocs);
            c.d_head_w = upload_tensor_to_gpu<uint16_t>(st_hd, "head.weight", c.device_allocs);
        }
        for (int e_idx = 0; e_idx < 2; ++e_idx) {
            const int lid = (e_idx == 0) ? 1 : 14;
            SafetensorsMmap st_eg(tp_dir + "/engram_layer_" + std::to_string(lid) + ".safetensors");
            c.engram[e_idx].d_q_w   = upload_tensor_to_gpu<uint16_t>(st_eg, "q_weight", c.device_allocs);
            c.engram[e_idx].d_k_w   = upload_tensor_to_gpu<uint16_t>(st_eg, "k_weight", c.device_allocs);
            c.engram[e_idx].d_wkv_w = upload_tensor_to_gpu<uint8_t>(st_eg, "wkv.weight", c.device_allocs);
            c.engram[e_idx].d_wkv_s = upload_tensor_to_gpu<uint8_t>(st_eg, "wkv.scale", c.device_allocs);
        }

        for (int l = 0; l < kNumLayers; ++l) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "/layer_%02d.safetensors", l);
            load_block_weights(tp, l, tp_dir + buf);
        }
        for (int m = 0; m < kNumMtpLayers; ++m) {
            load_block_weights(tp, kNumLayers + m, tp_dir + "/mtp_" + std::to_string(m) + ".safetensors");
        }

        // Initialize exact FP64-angle register RoPE / YaRN inverse frequencies (< 1 KiB __constant__, 0 B HBM2!)
        init_rope_inv_freq_gpu();

        c.d_win_kv_slab = alloc_gpu_buf<uint16_t>(kWinKvSlabElems, c.device_allocs, true);
        for (int b = 0; b < kTotalBlocks; ++b) {
            c.d_win_kv[b] = c.d_win_kv_slab + size_t(b) * kWindowSize * kHeadDim;
        }
        c.d_last_win_kv_slab = alloc_gpu_buf<uint16_t>(size_t(kNumLayers) * 8u * kHeadDim, c.device_allocs, true);
        for (int l = 0; l < kNumLayers; ++l) {
            c.d_last_win_kv_bf16[l] = c.d_last_win_kv_slab + size_t(l) * 8u * kHeadDim;
        }
        c.d_comp_kv_state_slab = alloc_gpu_buf<float>(kCompStateSlabElems, c.device_allocs, true);
        for (int s = 0; s < 3; ++s) {
            c.d_comp_kv_state[s]    = c.d_comp_kv_state_slab + size_t(s * 4 + 0) * kHeadDim;
            c.d_comp_score_state[s] = c.d_comp_kv_state_slab + size_t(s * 4 + 2) * kHeadDim;
            c.d_comp_kv_last[s]     = alloc_gpu_buf<float>(8u * kHeadDim, c.device_allocs, true);
            c.d_comp_score_last[s]  = alloc_gpu_buf<float>(8u * kHeadDim, c.device_allocs, true);
        }
        for (int s = 0; s < 4; ++s) {
            const size_t max_comp = (s < 3) ? size_t(max_seq_len / 2 + 64) : size_t(max_seq_len + 64);
            c.d_comp_kv_fp4[s]   = alloc_gpu_buf<uint8_t>(max_comp * 256u, c.device_allocs, true);
            c.d_comp_kv_e4m3[s]  = alloc_gpu_buf<uint8_t>(max_comp * 32u,  c.device_allocs, true);
            c.d_index_k_fp4[s]   = alloc_gpu_buf<uint8_t>(max_comp * 64u,  c.device_allocs, true);
            c.d_index_k_ue8m0[s] = alloc_gpu_buf<uint8_t>(max_comp * 4u,   c.device_allocs, true);
        }

        const size_t M = kMaxChunkM;
        c.d_token_ids      = alloc_gpu_buf<int32_t>(M, c.device_allocs);
        c.d_h_bf16[0]      = alloc_gpu_buf<uint16_t>(M * kHcMult * kDim, c.device_allocs);
        c.d_h_bf16[1]      = alloc_gpu_buf<uint16_t>(M * kHcMult * kDim, c.device_allocs);
        c.d_pre_mix_f32[0] = alloc_gpu_buf<float>(M * kHcMult, c.device_allocs);
        c.d_pre_mix_f32[1] = alloc_gpu_buf<float>(M * kHcMult, c.device_allocs);
        c.d_post_mix_f32   = alloc_gpu_buf<float>(M * kHcMult, c.device_allocs);
        c.d_comb_mix_f32   = alloc_gpu_buf<float>(M * 16u, c.device_allocs);
        c.d_hc_rsqrt_f32   = alloc_gpu_buf<float>(M, c.device_allocs);
        c.d_hc_mixes_f32   = alloc_gpu_buf<float>(M * kHcMixDim, c.device_allocs);

        c.d_x_bf16    = alloc_gpu_buf<uint16_t>(M * kDim, c.device_allocs);
        c.d_sub_bf16  = alloc_gpu_buf<uint16_t>(M * kDim, c.device_allocs);
        c.d_part_bf16 = alloc_gpu_buf<uint16_t>(M * kDim, c.device_allocs);
        c.d_part_f32  = alloc_gpu_buf<float>(kPartF32Floats, c.device_allocs);
        c.d_sum_f32   = alloc_gpu_buf<float>(kPartF32Floats, c.device_allocs);

        c.d_fp8_act   = alloc_gpu_buf<uint8_t>(M * 12288u, c.device_allocs);
        c.d_fp8_scale = alloc_gpu_buf<uint8_t>(M * 384u, c.device_allocs);

        c.d_qkv_bf16    = alloc_gpu_buf<uint16_t>(M * kQkvConcatRows, c.device_allocs);
        c.d_qr_bf16     = alloc_gpu_buf<uint16_t>(M * kQLoraRank, c.device_allocs);
        c.d_q_bf16      = alloc_gpu_buf<uint16_t>(M * 24u * kHeadDim, c.device_allocs);
        c.d_kv_cur_bf16 = alloc_gpu_buf<uint16_t>(M * kHeadDim, c.device_allocs);
        c.d_o_bf16      = alloc_gpu_buf<uint16_t>(M * 24u * kHeadDim, c.device_allocs);
        c.d_wo_mid_bf16 = alloc_gpu_buf<uint16_t>(M * 3u * kOLoraRank, c.device_allocs);

        c.d_comp_topk_idxs  = alloc_gpu_buf<int32_t>(M * kIndexTopk, c.device_allocs);
        c.d_cand_block_mask = alloc_gpu_buf<uint8_t>(size_t(kMaxDecChunkM) * size_t(c.cand_block_stride), c.device_allocs);

        c.d_comp_kv_tmp_f32    = alloc_gpu_buf<float>(M * kHeadDim, c.device_allocs);
        c.d_comp_score_tmp_f32 = alloc_gpu_buf<float>(M * kHeadDim, c.device_allocs);
        c.d_latent_bf16        = alloc_gpu_buf<uint16_t>(M * kHeadDim, c.device_allocs);

        c.d_idx_q_bf16       = alloc_gpu_buf<uint16_t>(M * 12u * kIndexHeadDim, c.device_allocs);
        c.d_idx_k_tmp_bf16   = alloc_gpu_buf<uint16_t>(M * kIndexHeadDim, c.device_allocs);
        c.d_idx_weights_bf16 = alloc_gpu_buf<uint16_t>(M * 16u, c.device_allocs);

        c.d_engram_in_bf16[0] = alloc_gpu_buf<uint16_t>(M * 2048u, c.device_allocs);
        c.d_engram_in_bf16[1] = alloc_gpu_buf<uint16_t>(M * 2048u, c.device_allocs);

        c.d_gate_logits_f32 = alloc_gpu_buf<float>(M * kBackboneExperts, c.device_allocs);
        c.d_topk_ids        = alloc_gpu_buf<int32_t>(M * kBackboneTopk, c.device_allocs);
        c.d_topk_weights    = alloc_gpu_buf<float>(M * kBackboneTopk, c.device_allocs);
        // Alias MoE-only buffers onto strictly disjoint Attention-only d_o_bf16 (24 MiB) and d_q_bf16 (24 MiB), saving 21.0 MiB HBM2/GPU!
        c.d_sh_gateup_bf16  = c.d_o_bf16;
        c.d_x_rot_f16       = alloc_gpu_buf<uint16_t>(M * kDim, c.device_allocs);
        c.d_routed_pairs    = alloc_gpu_buf<Mi50RoutedPair>(M * kBackboneTopk, c.device_allocs);
        c.d_active_experts  = alloc_gpu_buf<int32_t>(256u, c.device_allocs);
        c.d_csr_offsets     = alloc_gpu_buf<int32_t>(257u, c.device_allocs);
        c.d_inter_bf16      = c.d_q_bf16;
        c.d_inter_rot_f16   = c.d_q_bf16 + (M * kBackboneTopk * kSharedInterTp);
        c.d_moe_part_f32    = c.d_part_f32;
        c.d_moe_sum_f32     = c.d_sum_f32;
        c.d_p150_out_bf16   = alloc_gpu_buf<uint16_t>(M * kDim, c.device_allocs);

        const size_t dec_M = kMaxDecM;
        c.d_target_mean_bf16  = alloc_gpu_buf<uint16_t>(dec_M * kDim, c.device_allocs);
        c.d_main_x_bf16       = alloc_gpu_buf<uint16_t>(dec_M * kDim, c.device_allocs);
        c.d_logits_part_f32   = alloc_gpu_buf<float>(64u * size_t(c.vocab_rows), c.device_allocs);
        c.d_markov_sum_bf16   = alloc_gpu_buf<uint16_t>(64u * kMarkovRank, c.device_allocs);
        c.d_markov_logits_f32 = alloc_gpu_buf<float>(64u * size_t(c.vocab_rows), c.device_allocs);
        c.d_conf_in_bf16      = alloc_gpu_buf<uint16_t>(64u * (kDim + kMarkovRank), c.device_allocs);
        c.d_conf_out_f32      = alloc_gpu_buf<float>(64u, c.device_allocs);

        c.h_reduce_bf16 = alloc_pinned<uint16_t>(tp, 64u);
        c.h_reduce_f32  = alloc_pinned<float>(tp, 64u);
        c.h_logits_part    = alloc_pinned<float>(tp, 8u * size_t(c.vocab_rows));
        c.h_engram_bf16[0] = alloc_pinned<uint16_t>(tp, M * 2048u);
        c.h_engram_bf16[1] = alloc_pinned<uint16_t>(tp, M * 2048u);
        if (tp == 0) {
            c.h_topk_ids     = alloc_pinned<int32_t>(tp, M * kBackboneTopk);
            c.h_topk_w       = alloc_pinned<float>(tp, M * kBackboneTopk);
            c.h_x_moe_bf16   = alloc_pinned<uint16_t>(tp, M * kDim);
            c.h_radix_stage  = alloc_pinned<uint8_t>(tp, 8u * 1024u * 1024u);
            c.h_radix_stage2 = alloc_pinned<uint8_t>(tp, 8u * 1024u * 1024u);
            c.h_routed_pairs = alloc_pinned<Mi50RoutedPair>(tp, M * kBackboneTopk);
            c.h_active_exp   = alloc_pinned<int32_t>(tp, 256u);
            c.h_csr_offsets  = alloc_pinned<int32_t>(tp, 257u);
        }

        reset_r2_compressor_state_gpu(tp);

        size_t free_b = 0, total_b = 0;
        DSV41_HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
        std::printf("[DSV41_RT] MI50 tp=%d ready: vocab=[%d..%d) h_cnt=%d ih_cnt=%d HBM=%.2f / %.2f GiB used\n",
                    tp, c.vocab_offset, c.vocab_offset + c.vocab_rows,
                    c.blocks[2].h_cnt, c.blocks[2].ih_cnt,
                    double(total_b - free_b) / (1024.0 * 1024.0 * 1024.0),
                    double(total_b) / (1024.0 * 1024.0 * 1024.0));
        std::fflush(stdout);
    }

    // Out-of-place TP3 BF16 all-reduce: dst[tp] = src[0] + src[1] + src[2]
    // 100% crash-proof zero-host-sync coherent pinned DDR4 ring (0 PCIe BAR P2P reads, 0 CPU sync stalls!)
    void all_reduce_tp3_bf16(uint16_t* src0, uint16_t* src1, uint16_t* src2,
                             uint16_t* dst0, uint16_t* dst1, uint16_t* dst2,
                             size_t n_elems) {
        if (n_elems == 0) return;
        sync_tp0_if_profiling();
        const auto tar0 = std::chrono::steady_clock::now();
        uint16_t* srcs[3] = {src0, src1, src2};
        uint16_t* dsts[3] = {dst0, dst1, dst2};
        constexpr size_t kMaxBf16PerSlot = kHostArMaxWords * 2u;
        for (size_t off = 0; off < n_elems; off += kMaxBf16PerSlot) {
            const size_t tile = std::min<size_t>(kMaxBf16PerSlot, n_elems - off);
            const uint64_t seq = ++host_ar_seq;
            HostRingAllReduceSlot* slot = &h_ar_ring[(seq - 1ULL) & (kHostArRingSlots - 1ULL)];
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                launch_host_ring_allreduce_bf16(srcs[tp] + off, dsts[tp] + off, slot, tp, seq, tile, ctx[tp].stream);
            }
        }
        sync_tp0_if_profiling();
        accum_allreduce_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tar0).count();
    }

    // Out-of-place TP3 FP32 all-reduce: dst[tp] = src[0] + src[1] + src[2]
    void all_reduce_tp3_f32(float* src0, float* src1, float* src2,
                            float* dst0, float* dst1, float* dst2,
                            size_t n_elems) {
        if (n_elems == 0) return;
        sync_tp0_if_profiling();
        const auto tar0 = std::chrono::steady_clock::now();
        float* srcs[3] = {src0, src1, src2};
        float* dsts[3] = {dst0, dst1, dst2};
        for (size_t off = 0; off < n_elems; off += kHostArMaxWords) {
            const size_t tile = std::min<size_t>(kHostArMaxWords, n_elems - off);
            const uint64_t seq = ++host_ar_seq;
            HostRingAllReduceSlot* slot = &h_ar_ring[(seq - 1ULL) & (kHostArRingSlots - 1ULL)];
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                launch_host_ring_allreduce_f32(srcs[tp] + off, dsts[tp] + off, slot, tp, seq, tile, ctx[tp].stream);
            }
        }
        sync_tp0_if_profiling();
        accum_allreduce_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tar0).count();
    }

    // Dumitrescu mixed-precision FP32 TP3 HostRingAllReduce + optional P150A BF16 add + FP32 mHC post -> 1 final RNE BF16 (M <= 64)
    void all_reduce_tp3_f32_hc_post(const float* src0, const float* src1, const float* src2,
                                    const uint16_t* p150_0, const uint16_t* p150_1, const uint16_t* p150_2,
                                    int in_h_slot, int out_h_slot, int M) {
        if (M <= 0) return;
        sync_tp0_if_profiling();
        const auto tar0 = std::chrono::steady_clock::now();
        const float* srcs[3] = {src0, src1, src2};
        const uint16_t* p150s[3] = {p150_0, p150_1, p150_2};
        const uint64_t seq = ++host_ar_seq;
        HostRingAllReduceSlot* slot = &h_ar_ring[(seq - 1ULL) & (kHostArRingSlots - 1ULL)];
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            auto& c = ctx[tp];
            launch_host_ring_allreduce_f32_hc_post(
                srcs[tp], p150s[tp], c.d_h_bf16[in_h_slot],
                c.d_post_mix_f32, c.d_comb_mix_f32, c.d_h_bf16[out_h_slot],
                slot, tp, seq, M, c.stream);
        }
        sync_tp0_if_profiling();
        accum_allreduce_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tar0).count();
    }

    // Write `count` BF16 [512] rows from d_src_bf16 into ring d_win_kv[block_id] starting at global `start_pos`
    void write_ring_window_gpu(int tp, int block_id, const uint16_t* d_src_bf16, int start_pos, int count) {
        if (count <= 0) return;
        auto& c = ctx[tp];
        DSV41_HIP_CHECK(hipSetDevice(tp));
        if (count >= kWindowSize) {
            const uint16_t* tail_src = d_src_bf16 + size_t(count - kWindowSize) * kHeadDim;
            const int end_pos = start_pos + count;
            const int cutoff = end_pos % kWindowSize;
            const int first_part = kWindowSize - cutoff;
            if (first_part > 0) {
                DSV41_HIP_CHECK(hipMemcpyAsync(c.d_win_kv[block_id] + size_t(cutoff) * kHeadDim,
                                               tail_src, size_t(first_part) * kHeadDim * sizeof(uint16_t),
                                               hipMemcpyDeviceToDevice, c.stream));
            }
            if (cutoff > 0) {
                DSV41_HIP_CHECK(hipMemcpyAsync(c.d_win_kv[block_id],
                                               tail_src + size_t(first_part) * kHeadDim,
                                               size_t(cutoff) * kHeadDim * sizeof(uint16_t),
                                               hipMemcpyDeviceToDevice, c.stream));
            }
        } else {
            const int slot0 = start_pos % kWindowSize;
            const int first_part = std::min(count, kWindowSize - slot0);
            DSV41_HIP_CHECK(hipMemcpyAsync(c.d_win_kv[block_id] + size_t(slot0) * kHeadDim,
                                           d_src_bf16, size_t(first_part) * kHeadDim * sizeof(uint16_t),
                                           hipMemcpyDeviceToDevice, c.stream));
            if (count > first_part) {
                DSV41_HIP_CHECK(hipMemcpyAsync(c.d_win_kv[block_id],
                                               d_src_bf16 + size_t(first_part) * kHeadDim,
                                               size_t(count - first_part) * kHeadDim * sizeof(uint16_t),
                                               hipMemcpyDeviceToDevice, c.stream));
            }
        }
    }

    // Run Layer 20 Compressor + Indexer.k ONLY (used for strict CED 20-layer encoder chunks!)
    void forward_layer20_compressor_only(int h_slot, int start_pos, int M) {
        constexpr int l20 = 20;
        constexpr int kv_slot = 3;
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            auto& c = ctx[tp];
            const auto& b = c.blocks[l20];
            launch_hc_pre_rmsnorm_bf16(c.d_h_bf16[h_slot], c.d_pre_mix_f32[h_slot], b.d_attn_norm,
                                       c.d_x_bf16, nullptr, M, kNormEps, c.stream);

            // r=1 latent: norm(wkv(x))
            launch_bf16_gemm(c.d_x_bf16, b.d_comp_wkv, c.d_latent_bf16, nullptr, M, kHeadDim, kDim, false, c.stream);
            launch_rmsnorm_bf16(c.d_latent_bf16, b.d_comp_norm, c.d_latent_bf16, M, kHeadDim, kNormEps, c.stream);

            // Packed 68 B/slot Indexer k_cache (uses unrotated latent!)
            launch_bf16_gemm(c.d_latent_bf16, b.d_idx_wk, c.d_idx_k_tmp_bf16, nullptr, M, kIndexHeadDim, kHeadDim, false, c.stream);
            launch_rmsnorm_bf16(c.d_idx_k_tmp_bf16, b.d_idx_k_norm, c.d_idx_k_tmp_bf16, M, kIndexHeadDim, kNormEps, c.stream);
            launch_rope_tail64_bf16(c.d_idx_k_tmp_bf16, true, M, 1, kIndexHeadDim, start_pos, 1, false, c.stream);
            launch_pack_index_k_fp4_ue8m0_32(c.d_idx_k_tmp_bf16, c.d_index_k_fp4[kv_slot], c.d_index_k_ue8m0[kv_slot],
                                             start_pos, M, c.stream);

            // Compress KV cache (RoPE + FP4/E4M3_16)
            launch_rope_tail64_bf16(c.d_latent_bf16, true, M, 1, kHeadDim, start_pos, 1, false, c.stream);
            launch_pack_compress_kv_fp4_e4m3_16(c.d_latent_bf16, c.d_comp_kv_fp4[kv_slot], c.d_comp_kv_e4m3[kv_slot],
                                                start_pos, M, c.stream);
        }
    }

    // Run full Backbone Attention for layer `l` (0..39) + mHC post into `out_h_slot` (single final RNE BF16 for M <= 64!)
    void forward_backbone_attention(
        int l,
        int start_pos,
        int M,
        int in_h_slot,
        int out_h_slot,
        int chunk_swa_floor,
        bool commit_window_immediately,
        bool skip_global_kv_write,
        int& io_shared_k_comp,
        uint64_t tree_anc_mask64 = 0ULL,
        uint32_t tree_depth_nibbles = 0u,
        uint32_t tree_parent_nibbles = 0u)
    {
        const int ratio = kCompressRatios[l];
        const int kv_slot = kv_source_slot_for_layer(l);
        int n_lat = 0;
        const int lat_slot_base = (ratio > 0) ? (start_pos / ratio) : 0;
        const int T_comp = (ratio > 0)
            ? ((tree_anc_mask64 != 0ULL) ? (lat_slot_base + M) : ((start_pos + M) / ratio))
            : 0;
        const int K_comp = (ratio > 0) ? std::min(kIndexTopk, T_comp) : 0;
        const int swa_floor = std::max(chunk_swa_floor, (l >= 20) ? win_valid_from_20_39 : 0);
        sync_tp0_if_profiling();
        const auto t_att0 = std::chrono::steady_clock::now();
        double hsi_delta_ms = 0.0;

        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            auto& c = ctx[tp];
            const auto& b = c.blocks[l];
            const bool use_comp_yarn = (ratio > 0);

            // Fused [wq_a; wkv] [1792, 5120] + fused split_qkv_norm_rope_quant!
            launch_act_quant_ue8m0_32(c.d_x_bf16, c.d_fp8_act, c.d_fp8_scale, nullptr, M, kDim, false, c.stream);
            launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, b.d_wqkv_w, b.d_wqkv_s, c.d_qkv_bf16, nullptr,
                            M, kQkvConcatRows, kDim, false, false, c.stream);
            launch_split_qkv_norm_rope_quant(c.d_qkv_bf16, b.d_q_norm, b.d_kv_norm, use_comp_yarn,
                                             c.d_qr_bf16, c.d_kv_cur_bf16, M, start_pos, kNormEps, c.stream,
                                             tree_anc_mask64, tree_depth_nibbles);

            // Quantize qr [M, 1280] (reused by wq_b AND indexer.wq_b!)
            launch_act_quant_ue8m0_32(c.d_qr_bf16, c.d_fp8_act, c.d_fp8_scale, nullptr, M, kQLoraRank, false, c.stream);
            launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, b.d_wq_b_w, b.d_wq_b_s, c.d_q_bf16, nullptr,
                            M, b.h_cnt * kHeadDim, kQLoraRank, false, false, c.stream);
            launch_rope_tail64_bf16(c.d_q_bf16, use_comp_yarn, M, b.h_cnt, kHeadDim, start_pos, 1, false, c.stream,
                                    tree_anc_mask64, tree_depth_nibbles);

            // Compressor pre-RoPE latent (skipped during SWA Bounded Replay when Global KV already restored!)
            if (b.is_kv_source && !skip_global_kv_write) {
                if (ratio == 1) {
                    launch_bf16_gemm(c.d_x_bf16, b.d_comp_wkv, c.d_latent_bf16, nullptr, M, kHeadDim, kDim, false, c.stream);
                    launch_rmsnorm_bf16(c.d_latent_bf16, b.d_comp_norm, c.d_latent_bf16, M, kHeadDim, kNormEps, c.stream);
                    n_lat = M;
                } else {
                    const int r2_idx = kv_slot;  // 0, 1, 2
                    launch_bf16_gemm(c.d_x_bf16, b.d_comp_wkv, nullptr, c.d_comp_kv_tmp_f32, M, kHeadDim, kDim, true, c.stream);
                    launch_bf16_gemm(c.d_x_bf16, b.d_comp_wgate, nullptr, c.d_comp_score_tmp_f32, M, kHeadDim, kDim, true, c.stream);
                    if (!commit_window_immediately) {
                        DSV41_HIP_CHECK(hipMemcpyAsync(c.d_comp_kv_last[r2_idx], c.d_comp_kv_tmp_f32,
                                                       size_t(M) * kHeadDim * sizeof(float),
                                                       hipMemcpyDeviceToDevice, c.stream));
                        DSV41_HIP_CHECK(hipMemcpyAsync(c.d_comp_score_last[r2_idx], c.d_comp_score_tmp_f32,
                                                       size_t(M) * kHeadDim * sizeof(float),
                                                       hipMemcpyDeviceToDevice, c.stream));
                    }
                    if (tree_anc_mask64 != 0ULL) {
                        launch_compressor_r2_tree_pool(c.d_comp_kv_tmp_f32, c.d_comp_score_tmp_f32,
                                                       c.d_comp_kv_state[r2_idx], c.d_comp_score_state[r2_idx],
                                                       c.d_latent_bf16, start_pos, M,
                                                       tree_depth_nibbles, tree_parent_nibbles, c.stream);
                        n_lat = M;
                    } else {
                        n_lat = launch_compressor_r2_pool(c.d_comp_kv_tmp_f32, c.d_comp_score_tmp_f32,
                                                          c.d_comp_kv_state[r2_idx], c.d_comp_score_state[r2_idx],
                                                          c.d_latent_bf16, start_pos, M, c.stream);
                    }
                    if (n_lat > 0) {
                        launch_rmsnorm_bf16(c.d_latent_bf16, b.d_comp_norm, c.d_latent_bf16, n_lat, kHeadDim, kNormEps, c.stream);
                    }
                }
            }

            // Indexer key update + Q/weights preparation (if index_source)
            if (b.is_index_source) {
                if (b.is_kv_source && !skip_global_kv_write && n_lat > 0) {
                    launch_bf16_gemm(c.d_latent_bf16, b.d_idx_wk, c.d_idx_k_tmp_bf16, nullptr,
                                     n_lat, kIndexHeadDim, kHeadDim, false, c.stream);
                    launch_rmsnorm_bf16(c.d_idx_k_tmp_bf16, b.d_idx_k_norm, c.d_idx_k_tmp_bf16,
                                        n_lat, kIndexHeadDim, kNormEps, c.stream);
                    launch_rope_tail64_bf16(c.d_idx_k_tmp_bf16, true,
                                            n_lat, 1, kIndexHeadDim,
                                            (tree_anc_mask64 != 0ULL) ? start_pos : (lat_slot_base * ratio),
                                            ratio, false, c.stream, tree_anc_mask64, tree_depth_nibbles);
                    launch_pack_index_k_fp4_ue8m0_32(c.d_idx_k_tmp_bf16,
                                                     c.d_index_k_fp4[kv_slot], c.d_index_k_ue8m0[kv_slot],
                                                     lat_slot_base, n_lat, c.stream);
                }
                if (T_comp > 0) {
                    launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, b.d_idx_wq_b_w, b.d_idx_wq_b_s, c.d_idx_q_bf16, nullptr,
                                    M, b.ih_cnt * kIndexHeadDim, kQLoraRank, false, false, c.stream);
                    launch_rope_tail64_bf16(c.d_idx_q_bf16, true,
                                            M, b.ih_cnt, kIndexHeadDim, start_pos, 1, false, c.stream,
                                            tree_anc_mask64, tree_depth_nibbles);
                    launch_fp4_quant_dequant_ue8m0_32_inplace(c.d_idx_q_bf16, M * b.ih_cnt, kIndexHeadDim, c.stream);
                    launch_bf16_gemm(c.d_x_bf16, b.d_idx_weights_proj, c.d_idx_weights_bf16, nullptr,
                                     M, b.ih_cnt, kDim, false, c.stream);
                }
            }
        }
        if (sync_timers) {
            sync_tp0_if_profiling();
            sub_att_qkv_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_att0).count();
        }

        // Sub-batched Indexer score + TP3 all-reduce + candidate/top-k selection aliased onto d_part_f32 / d_sum_f32!
        if (ctx[0].blocks[l].is_index_source) {
            io_shared_k_comp = K_comp;
            if (T_comp > 0) {
                sync_tp0_if_profiling();
                const auto t_hsi0 = std::chrono::steady_clock::now();
                const int mode = (l == kCandSourceLayer) ? 1 : ((l > kCandSourceLayer) ? 2 : 0);
                const size_t floats_per_row = size_t(T_comp) + size_t((T_comp + 7) / 8) + 64u;
                const int max_sub_m = std::max(1, std::min(M, static_cast<int>(kPartF32Floats / floats_per_row)));

                for (int m0 = 0; m0 < M; m0 += max_sub_m) {
                    const int sub_m = std::min(max_sub_m, M - m0);
                    for (int tp = 0; tp < 3; ++tp) {
                        DSV41_HIP_CHECK(hipSetDevice(tp));
                        auto& c = ctx[tp];
                        const auto& b = c.blocks[l];
                        launch_indexer_score(
                            c.d_idx_q_bf16 + size_t(m0) * size_t(b.ih_cnt) * kIndexHeadDim,
                            c.d_index_k_fp4[kv_slot],
                            c.d_index_k_ue8m0[kv_slot],
                            c.d_idx_weights_bf16 + size_t(m0) * size_t(b.ih_cnt),
                            c.d_part_f32,  // Always offset 0 of aliased scratch!
                            sub_m, b.ih_cnt, T_comp, c.stream);
                    }
                    all_reduce_tp3_f32(ctx[0].d_part_f32, ctx[1].d_part_f32, ctx[2].d_part_f32,
                                       ctx[0].d_sum_f32,  ctx[1].d_sum_f32,  ctx[2].d_sum_f32,
                                       size_t(sub_m) * size_t(T_comp));
                    for (int tp = 0; tp < 3; ++tp) {
                        DSV41_HIP_CHECK(hipSetDevice(tp));
                        auto& c = ctx[tp];
                        uint8_t* cand_ptr = (mode == 0)
                            ? nullptr
                            : (c.d_cand_block_mask + size_t(m0) * size_t(c.cand_block_stride));
                        launch_indexer_candidates_and_topk(
                            c.d_sum_f32,  // Always offset 0 of aliased scratch!
                            cand_ptr,
                            c.cand_block_stride,
                            c.d_comp_topk_idxs + size_t(m0) * size_t(K_comp),
                            sub_m, T_comp,
                            (tree_anc_mask64 != 0ULL) ? start_pos : (start_pos + m0),
                            ratio, mode, K_comp, c.stream,
                            tree_anc_mask64, tree_depth_nibbles, m0);
                    }
                }
                sync_tp0_if_profiling();
                hsi_delta_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_hsi0).count();
                accum_hsi_ms += hsi_delta_ms;
            }
        }

        // Pack new compressed KV latents + In-place Ring SWA/CSA2 Sparse Attention + commit ring window + wo_a + wo_b
        const auto t_sp0 = std::chrono::steady_clock::now();
        const int active_k_comp = (ratio > 0) ? io_shared_k_comp : 0;
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            auto& c = ctx[tp];
            const auto& b = c.blocks[l];

            if (b.is_kv_source && !skip_global_kv_write && n_lat > 0) {
                launch_rope_tail64_bf16(c.d_latent_bf16, true,
                                        n_lat, 1, kHeadDim,
                                        (tree_anc_mask64 != 0ULL) ? start_pos : (lat_slot_base * ratio),
                                        ratio, false, c.stream, tree_anc_mask64, tree_depth_nibbles);
                launch_pack_compress_kv_fp4_e4m3_16(c.d_latent_bf16, c.d_comp_kv_fp4[kv_slot], c.d_comp_kv_e4m3[kv_slot],
                                                    lat_slot_base, n_lat, c.stream);
            }

            // CRITICAL INVARIANT: launch_sparse_attn MUST run BEFORE write_ring_window_gpu so historical slots < start_pos
            // in c.d_win_kv[l] are not overwritten by c.d_kv_cur_bf16!
            launch_sparse_attn(
                c.d_q_bf16, b.d_attn_sink,
                c.d_win_kv[l], c.d_kv_cur_bf16,
                start_pos, swa_floor, false,
                (active_k_comp > 0) ? c.d_comp_kv_fp4[kv_slot] : nullptr,
                (active_k_comp > 0) ? c.d_comp_kv_e4m3[kv_slot] : nullptr,
                (active_k_comp > 0) ? c.d_comp_topk_idxs : nullptr,
                active_k_comp,
                (M <= 8) ? c.d_part_f32 : nullptr,
                c.d_o_bf16, M, b.h_cnt, c.stream,
                tree_anc_mask64, tree_depth_nibbles);

            if (commit_window_immediately) {
                write_ring_window_gpu(tp, l, c.d_kv_cur_bf16, start_pos, M);
            } else {
                DSV41_HIP_CHECK(hipMemcpyAsync(c.d_last_win_kv_bf16[l], c.d_kv_cur_bf16,
                                               size_t(M) * kHeadDim * sizeof(uint16_t),
                                               hipMemcpyDeviceToDevice, c.stream));
            }
        }
        if (sync_timers) {
            sync_tp0_if_profiling();
            sub_att_sparse_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sp0).count();
        }
        const auto t_wo0 = std::chrono::steady_clock::now();
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            auto& c = ctx[tp];
            const auto& b = c.blocks[l];
            const bool use_comp_yarn = (ratio > 0);

            // Inverse RoPE on tail 64 dims of o + grouped wo_a + wo_b
            launch_rope_tail64_bf16(c.d_o_bf16, use_comp_yarn, M, b.h_cnt, kHeadDim, start_pos, 1, true, c.stream,
                                    tree_anc_mask64, tree_depth_nibbles);
            launch_act_quant_ue8m0_32(c.d_o_bf16, c.d_fp8_act, c.d_fp8_scale, nullptr,
                                      M, b.g_cnt * 4096, false, c.stream);
            launch_fp8_grouped_wo_a(c.d_fp8_act, c.d_fp8_scale, b.d_wo_a_w, b.d_wo_a_s,
                                    c.d_wo_mid_bf16, M, b.g_cnt, c.stream);
            launch_act_quant_ue8m0_32(c.d_wo_mid_bf16, c.d_fp8_act, c.d_fp8_scale, nullptr,
                                      M, b.g_cnt * kOLoraRank, false, c.stream);
            if (M <= kF32RingMaxM) {
                launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, b.d_wo_b_w, b.d_wo_b_s, nullptr, c.d_part_f32,
                                M, kDim, b.g_cnt * kOLoraRank, true, false, c.stream);
            } else {
                launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, b.d_wo_b_w, b.d_wo_b_s, c.d_part_bf16, nullptr,
                                M, kDim, b.g_cnt * kOLoraRank, false, false, c.stream);
            }
        }
        if (sync_timers) {
            sync_tp0_if_profiling();
            sub_att_wo_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_wo0).count();
        }
        const auto t_aar0 = std::chrono::steady_clock::now();
        if (M <= kF32RingMaxM) {
            all_reduce_tp3_f32_hc_post(ctx[0].d_part_f32, ctx[1].d_part_f32, ctx[2].d_part_f32,
                                       nullptr, nullptr, nullptr, in_h_slot, out_h_slot, M);
        } else {
            all_reduce_tp3_bf16(ctx[0].d_part_bf16, ctx[1].d_part_bf16, ctx[2].d_part_bf16,
                                ctx[0].d_sub_bf16,  ctx[1].d_sub_bf16,  ctx[2].d_sub_bf16,
                                size_t(M) * kDim);
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                auto& c = ctx[tp];
                launch_hc_post(c.d_sub_bf16, c.d_h_bf16[in_h_slot], c.d_post_mix_f32, c.d_comb_mix_f32,
                               c.d_h_bf16[out_h_slot], M, c.stream);
            }
        }
        sync_tp0_if_profiling();
        if (sync_timers) {
            sub_att_ar_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_aar0).count();
        }
        accum_attn_ms += std::max(0.0, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_att0).count() - hsi_delta_ms);
    }

    // Run MoE FFN + mHC post for block `block_id` (0..42):
    //   - MTP blocks (40..42): 100% device-resident pair-dispatched HLUT2 (M<=16) + FP32 ring mHC post!
    //   - Backbone blocks (0..39): fused ffn_norm WHT128 + router_stream P150A overlap + FP32 ring mHC post (M<=64)!
    void forward_block_moe(int block_id, int M, int in_h_slot, int out_h_slot) {
        const auto& b0 = ctx[0].blocks[block_id];
        const int top_k = b0.top_k;
        const bool is_backbone = (block_id < kNumLayers);
        sync_tp0_if_profiling();
        const auto t_moe0 = std::chrono::steady_clock::now();
        double p150_wait_this_ms = 0.0;

        if (!is_backbone) {
            const int P = M * top_k;
            const bool use_pairs = (M <= 16);
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                auto& c = ctx[tp];
                const auto& b = c.blocks[block_id];

                launch_bf16_gemm(c.d_x_bf16, b.d_gate_w, nullptr, c.d_gate_logits_f32,
                                 M, b.n_routed_experts, kDim, true, c.stream);
                launch_moe_gate(c.d_gate_logits_f32, b.d_gate_bias, b.d_gate_bias_vl, nullptr,
                                c.d_topk_ids, c.d_topk_weights,
                                M, b.n_routed_experts, top_k, kRouteScale, c.stream);
                if (tp == 0 && rarebal_collector && rarebal_collector->active) {
                    DSV41_HIP_CHECK(hipMemcpyAsync(c.h_topk_ids, c.d_topk_ids, size_t(M) * top_k * sizeof(int32_t),
                                                   hipMemcpyDeviceToHost, c.stream));
                    if (rarebal_collector->need_x_copy(block_id) && rarebal_collector->h_x_pinned) {
                        DSV41_HIP_CHECK(hipMemcpyAsync(rarebal_collector->h_x_pinned, c.d_x_bf16,
                                                       size_t(M) * kDim * sizeof(uint16_t),
                                                       hipMemcpyDeviceToHost, c.stream));
                    }
                    DSV41_HIP_CHECK(hipStreamSynchronize(c.stream));
                    rarebal_collector->record_routed_batch(block_id, M, top_k, c.h_topk_ids, rarebal_collector->h_x_pinned);
                }
                if (use_pairs) {
                    launch_pack_mtp_routed_pairs(c.d_topk_ids, c.d_topk_weights, c.d_routed_pairs, M, top_k, c.stream);
                } else {
                    launch_build_mtp_csr(c.d_topk_ids, c.d_topk_weights,
                                         c.d_active_experts, c.d_csr_offsets, c.d_routed_pairs,
                                         M, top_k, b.n_routed_experts, c.stream);
                }

                launch_act_quant_ue8m0_32(c.d_x_bf16, c.d_fp8_act, c.d_fp8_scale, nullptr, M, kDim, false, c.stream);
                launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, b.d_sh_w13_w, b.d_sh_w13_s, c.d_sh_gateup_bf16, nullptr,
                                M, 2 * kSharedInterTp, kDim, false, false, c.stream);
                launch_swiglu10_split_quant_ue8m0_32(c.d_sh_gateup_bf16, c.d_fp8_act, c.d_fp8_scale,
                                                     M, kSharedInterTp, c.stream);
                launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, b.d_sh_w2_w, b.d_sh_w2_s, nullptr, c.d_moe_part_f32,
                                M, kDim, kSharedInterTp, true, false, c.stream);

                if (use_pairs) {
                    launch_mi50_routed_hlut2_pairs(
                        c.d_x_rot_f16, c.d_routed_pairs, P,
                        b.d_hlut2_gate_codes, b.d_hlut2_gate_sc, b.d_hlut2_gate_cent,
                        b.d_hlut2_up_codes,   b.d_hlut2_up_sc,   b.d_hlut2_up_cent,
                        b.d_hlut2_down_codes, b.d_hlut2_down_sc, b.d_hlut2_down_cent,
                        b.d_signs_i, c.d_inter_bf16, c.d_inter_rot_f16,
                        c.d_moe_part_f32, c.stream);
                } else {
                    launch_mi50_routed_hlut2_grouped(
                        c.d_x_rot_f16, c.d_active_experts, c.d_csr_offsets, c.d_routed_pairs,
                        M, b.n_routed_experts, P,
                        b.d_hlut2_gate_codes, b.d_hlut2_gate_sc, b.d_hlut2_gate_cent,
                        b.d_hlut2_up_codes,   b.d_hlut2_up_sc,   b.d_hlut2_up_cent,
                        b.d_hlut2_down_codes, b.d_hlut2_down_sc, b.d_hlut2_down_cent,
                        b.d_signs_i, c.d_inter_bf16, c.d_inter_rot_f16,
                        c.d_moe_part_f32, c.stream);
                }
                if (M > kF32RingMaxM) {
                    launch_f32_to_bf16(c.d_moe_part_f32, c.d_part_bf16, size_t(M) * kDim, c.stream);
                }
            }
            if (M <= kF32RingMaxM) {
                all_reduce_tp3_f32_hc_post(ctx[0].d_moe_part_f32, ctx[1].d_moe_part_f32, ctx[2].d_moe_part_f32,
                                           nullptr, nullptr, nullptr, in_h_slot, out_h_slot, M);
            } else {
                all_reduce_tp3_bf16(ctx[0].d_part_bf16, ctx[1].d_part_bf16, ctx[2].d_part_bf16,
                                    ctx[0].d_sub_bf16,  ctx[1].d_sub_bf16,  ctx[2].d_sub_bf16,
                                    size_t(M) * kDim);
                for (int tp = 0; tp < 3; ++tp) {
                    DSV41_HIP_CHECK(hipSetDevice(tp));
                    auto& c = ctx[tp];
                    launch_hc_post(c.d_sub_bf16, c.d_h_bf16[in_h_slot], c.d_post_mix_f32, c.d_comb_mix_f32,
                                   c.d_h_bf16[out_h_slot], M, c.stream);
                }
            }
            return;
        }

        // Wave 7A Backbone MoE: tp=0 router_stream gets 100% CU exclusivity first; NO MI50 shared/routed
        // starts until the P150A [216..383] doorbell is armed strictly after ev_router_done.
        {
            DSV41_HIP_CHECK(hipSetDevice(0));
            auto& c0 = ctx[0];
            const auto& bm0 = c0.blocks[block_id];
            DSV41_HIP_CHECK(hipEventRecord(c0.ev_x_ready, c0.stream));
            DSV41_HIP_CHECK(hipStreamWaitEvent(c0.router_stream, c0.ev_x_ready, 0));
            launch_bf16_gemm(c0.d_x_bf16, bm0.d_gate_w, nullptr, c0.d_gate_logits_f32,
                             M, bm0.n_routed_experts, kDim, true, c0.router_stream);
            launch_moe_gate(c0.d_gate_logits_f32, bm0.d_gate_bias, bm0.d_gate_bias_vl, nullptr,
                            c0.d_topk_ids, c0.d_topk_weights,
                            M, bm0.n_routed_experts, top_k, kRouteScale, c0.router_stream);
            DSV41_HIP_CHECK(hipMemcpyAsync(c0.h_topk_ids, c0.d_topk_ids, size_t(M) * top_k * sizeof(int32_t),
                                           hipMemcpyDeviceToHost, c0.router_stream));
            DSV41_HIP_CHECK(hipMemcpyAsync(c0.h_topk_w, c0.d_topk_weights, size_t(M) * top_k * sizeof(float),
                                           hipMemcpyDeviceToHost, c0.router_stream));
            if (p150_shm.ptr) {
                uint16_t* dst_x = p150_shm_pinned ? p150_shm.ptr->x_in_bf16 : c0.h_x_moe_bf16;
                DSV41_HIP_CHECK(hipMemcpyAsync(dst_x, c0.d_part_bf16, size_t(M) * kDim * sizeof(uint16_t),
                                               hipMemcpyDeviceToHost, c0.router_stream));
            }
            if (rarebal_collector && rarebal_collector->need_x_copy(block_id) && rarebal_collector->h_x_pinned) {
                DSV41_HIP_CHECK(hipMemcpyAsync(rarebal_collector->h_x_pinned, c0.d_x_bf16,
                                               size_t(M) * kDim * sizeof(uint16_t),
                                               hipMemcpyDeviceToHost, c0.router_stream));
            }
            DSV41_HIP_CHECK(hipEventRecord(c0.ev_router_done, c0.router_stream));
        }

        // Wait ONLY on tp=0's exclusive router_stream event so we immediately post to P150A!
        DSV41_HIP_CHECK(hipSetDevice(0));
        DSV41_HIP_CHECK(hipEventSynchronize(ctx[0].ev_router_done));
        if (rarebal_collector && rarebal_collector->active) {
            rarebal_collector->record_routed_batch(block_id, M, top_k, ctx[0].h_topk_ids, rarebal_collector->h_x_pinned);
        }

        mi50_pairs_buf.clear();
        bool any_p150 = false;
        uint32_t posted_p150_seq = 0u;
        auto* shm = p150_shm.ptr;

        const int32_t* ids = ctx[0].h_topk_ids;
        const float*   ws  = ctx[0].h_topk_w;
        for (int r = 0; r < M; ++r) {
            uint32_t p150_slot_cursor = 0u;
            for (int k = 0; k < top_k; ++k) {
                const int32_t eid = ids[r * top_k + k];
                const float w = ws[r * top_k + k];
                if (eid < 0 || w == 0.0f) continue;
                if (eid < b0.mi50_experts) {
                    mi50_pairs_buf.push_back(Mi50RoutedPair{r, eid, w, 0});
                } else if (shm && p150_slot_cursor < DSV41_P150_MAX_SLOTS) {
                    shm->row_slots[r][p150_slot_cursor] = eid - kMi50BackboneExp;
                    shm->row_weights[r][p150_slot_cursor] = w;
                    ++p150_slot_cursor;
                    any_p150 = true;
                }
            }
            if (shm) {
                for (uint32_t s = p150_slot_cursor; s < DSV41_P150_MAX_SLOTS; ++s) {
                    shm->row_slots[r][s] = -1;
                    shm->row_weights[r][s] = 0.0f;
                }
            }
        }

        // Wave 7A: Kick off P150A [216..383] doorbell FIRST over /dev/shm, strictly BEFORE
        // ANY MI50 shared SwiGLU, routed [0..215] HLUT2, or CpuTailPool singleton peel.
        if (any_p150 && shm) {
            if (!p150_shm_pinned) {
                std::memcpy(shm->x_in_bf16, ctx[0].h_x_moe_bf16, size_t(M) * kDim * sizeof(uint16_t));
            }
            shm->op = DSV41_P150_OP_MOE_FORWARD;
            shm->layer = static_cast<uint32_t>(block_id);
            shm->rows = static_cast<uint32_t>(M);
            shm->input_is_wht_rotated = 1u;
            posted_p150_seq = ++p150_seq_counter;
            shm->req_seq.store(posted_p150_seq, std::memory_order_release);
        }

        // Wave 7A: P150A doorbell is armed!
        // 1. Launch MI50 shared SwiGLU FIRST (`accumulate_f32 = false`) to initialize `c.d_moe_part_f32`
        //    on all 3 ranks BEFORE `[0..215]` HLUT2 `atomicAdd`s onto `c.d_moe_part_f32`!
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            auto& c = ctx[tp];
            const auto& b = c.blocks[block_id];
            launch_act_quant_ue8m0_32(c.d_x_bf16, c.d_fp8_act, c.d_fp8_scale, nullptr, M, kDim, false, c.stream);
            launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, b.d_sh_w13_w, b.d_sh_w13_s, c.d_sh_gateup_bf16, nullptr,
                            M, 2 * kSharedInterTp, kDim, false, false, c.stream);
            launch_swiglu10_split_quant_ue8m0_32(c.d_sh_gateup_bf16, c.d_fp8_act, c.d_fp8_scale,
                                                 M, kSharedInterTp, c.stream);
            launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, b.d_sh_w2_w, b.d_sh_w2_s, nullptr, c.d_moe_part_f32,
                            M, kDim, kSharedInterTp, true, false, c.stream);
        }
        if (sync_timers) {
            sync_tp0_if_profiling();
            sub_moe_sh_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_moe0).count();
        }

        // 2. Launch MI50 HLUT2 routed experts [0..215] (`atomicAdd` onto `c.d_moe_part_f32`) overlapping P150A!
        const auto t_hlut0 = std::chrono::steady_clock::now();
        const int P = static_cast<int>(mi50_pairs_buf.size());
        if (P > 0) {
            std::array<int, 256> counts{};
            int num_act = 0;
            for (const auto& pr : mi50_pairs_buf) {
                if (counts[pr.expert]++ == 0) ++num_act;
            }
            const bool use_grouped = (M > 8 && num_act < P);
            if (!use_grouped) {
                std::memcpy(ctx[0].h_routed_pairs, mi50_pairs_buf.data(), size_t(P) * sizeof(Mi50RoutedPair));
            } else {
                int act_idx = 0;
                ctx[0].h_csr_offsets[0] = 0;
                std::array<int, 256> cursor{};
                for (int e = 0; e < b0.mi50_experts; ++e) {
                    if (counts[e] > 0) {
                        cursor[e] = ctx[0].h_csr_offsets[act_idx];
                        ctx[0].h_active_exp[act_idx] = e;
                        ctx[0].h_csr_offsets[act_idx + 1] = cursor[e] + counts[e];
                        ++act_idx;
                    }
                }
                for (const auto& pr : mi50_pairs_buf) {
                    ctx[0].h_routed_pairs[cursor[pr.expert]++] = pr;
                }
            }

            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                auto& c = ctx[tp];
                const auto& b = c.blocks[block_id];
                if (!use_grouped) {
                    DSV41_HIP_CHECK(hipMemcpyAsync(c.d_routed_pairs, ctx[0].h_routed_pairs,
                                                   size_t(P) * sizeof(Mi50RoutedPair),
                                                   hipMemcpyHostToDevice, c.stream));
                    launch_mi50_routed_hlut2_pairs(
                        c.d_x_rot_f16, c.d_routed_pairs, P,
                        b.d_hlut2_gate_codes, b.d_hlut2_gate_sc, b.d_hlut2_gate_cent,
                        b.d_hlut2_up_codes,   b.d_hlut2_up_sc,   b.d_hlut2_up_cent,
                        b.d_hlut2_down_codes, b.d_hlut2_down_sc, b.d_hlut2_down_cent,
                        b.d_signs_i, c.d_inter_bf16, c.d_inter_rot_f16,
                        c.d_moe_part_f32, c.stream);
                } else {
                    DSV41_HIP_CHECK(hipMemcpyAsync(c.d_routed_pairs, ctx[0].h_routed_pairs,
                                                   size_t(P) * sizeof(Mi50RoutedPair),
                                                   hipMemcpyHostToDevice, c.stream));
                    DSV41_HIP_CHECK(hipMemcpyAsync(c.d_active_experts, ctx[0].h_active_exp,
                                                   size_t(num_act) * sizeof(int32_t),
                                                   hipMemcpyHostToDevice, c.stream));
                    DSV41_HIP_CHECK(hipMemcpyAsync(c.d_csr_offsets, ctx[0].h_csr_offsets,
                                                   size_t(num_act + 1) * sizeof(int32_t),
                                                   hipMemcpyHostToDevice, c.stream));
                    launch_mi50_routed_hlut2_grouped(
                        c.d_x_rot_f16, c.d_active_experts, c.d_csr_offsets, c.d_routed_pairs,
                        M, num_act, P,
                        b.d_hlut2_gate_codes, b.d_hlut2_gate_sc, b.d_hlut2_gate_cent,
                        b.d_hlut2_up_codes,   b.d_hlut2_up_sc,   b.d_hlut2_up_cent,
                        b.d_hlut2_down_codes, b.d_hlut2_down_sc, b.d_hlut2_down_cent,
                        b.d_signs_i, c.d_inter_bf16, c.d_inter_rot_f16,
                        c.d_moe_part_f32, c.stream);
                }
            }
        }

        const size_t moe_elems = size_t(M) * kDim;
        if (M <= kF32RingMaxM) {
            if (sync_timers) {
                sync_tp0_if_profiling();
                sub_moe_hlut_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_hlut0).count();
            }
            if (any_p150 && shm && d_p150_shm_y_out_bf16[0] != nullptr) {
                // Wave 7A Zero-SDMA Split HostRing: upload MI50 FP32 partials into coherent DDR4 WHILE P150A finishes,
                // then once shm->resp_seq completes, read mapped shm->y_out_bf16 directly via system-scope atomic loads!
                const uint64_t ar_seq = ++host_ar_seq;
                HostRingAllReduceSlot* ar_slot = &h_ar_ring[(ar_seq - 1ULL) & (kHostArRingSlots - 1ULL)];
                for (int tp = 0; tp < 3; ++tp) {
                    DSV41_HIP_CHECK(hipSetDevice(tp));
                    launch_host_ring_upload_f32(ctx[tp].d_moe_part_f32, ar_slot, tp, ar_seq, M, ctx[tp].stream);
                }

                const auto tw0 = std::chrono::steady_clock::now();
                while (shm->resp_seq.load(std::memory_order_acquire) != posted_p150_seq) {
                    dsv41_cpu_relax();
                }
                p150_wait_this_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw0).count();
                accum_p150_wait_ms += p150_wait_this_ms;
                if (shm->status != 0) {
                    throw std::runtime_error("p150_dsv41_worker returned error status " + std::to_string(shm->status));
                }
                accum_p150_dev_ms += double(shm->last_device_ns) * 1e-6;

                const auto t_mar0 = std::chrono::steady_clock::now();
                for (int tp = 0; tp < 3; ++tp) {
                    DSV41_HIP_CHECK(hipSetDevice(tp));
                    auto& c = ctx[tp];
                    launch_host_ring_finish_f32_hc_post(
                        c.d_moe_part_f32, d_p150_shm_y_out_bf16[tp],
                        c.d_h_bf16[in_h_slot], c.d_post_mix_f32, c.d_comb_mix_f32, c.d_h_bf16[out_h_slot],
                        ar_slot, tp, ar_seq, M, c.stream);
                }
                sync_tp0_if_profiling();
                const double ar_dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_mar0).count();
                accum_allreduce_ms += ar_dt;
                if (sync_timers) {
                    sub_moe_ar_ms += ar_dt;
                }
            } else {
                const uint16_t* p150_ptrs[3] = {nullptr, nullptr, nullptr};
                if (any_p150 && shm) {
                    const auto tw0 = std::chrono::steady_clock::now();
                    while (shm->resp_seq.load(std::memory_order_acquire) != posted_p150_seq) {
                        dsv41_cpu_relax();
                    }
                    p150_wait_this_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw0).count();
                    accum_p150_wait_ms += p150_wait_this_ms;
                    if (shm->status != 0) {
                        throw std::runtime_error("p150_dsv41_worker returned error status " + std::to_string(shm->status));
                    }
                    accum_p150_dev_ms += double(shm->last_device_ns) * 1e-6;
                    for (int tp = 0; tp < 3; ++tp) {
                        DSV41_HIP_CHECK(hipSetDevice(tp));
                        auto& c = ctx[tp];
                        DSV41_HIP_CHECK(hipMemcpyAsync(c.d_p150_out_bf16, shm->y_out_bf16,
                                                       moe_elems * sizeof(uint16_t),
                                                       hipMemcpyHostToDevice, c.stream));
                        p150_ptrs[tp] = c.d_p150_out_bf16;
                    }
                }
                const auto t_mar0 = std::chrono::steady_clock::now();
                all_reduce_tp3_f32_hc_post(ctx[0].d_moe_part_f32, ctx[1].d_moe_part_f32, ctx[2].d_moe_part_f32,
                                           p150_ptrs[0], p150_ptrs[1], p150_ptrs[2],
                                           in_h_slot, out_h_slot, M);
                if (sync_timers) {
                    sub_moe_ar_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_mar0).count();
                }
            }
        } else {
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                launch_f32_to_bf16(ctx[tp].d_moe_part_f32, ctx[tp].d_part_bf16, moe_elems, ctx[tp].stream);
            }
            if (sync_timers) {
                sync_tp0_if_profiling();
                sub_moe_hlut_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_hlut0).count();
            }
            const auto t_mar0 = std::chrono::steady_clock::now();
            all_reduce_tp3_bf16(ctx[0].d_part_bf16, ctx[1].d_part_bf16, ctx[2].d_part_bf16,
                                ctx[0].d_sub_bf16,  ctx[1].d_sub_bf16,  ctx[2].d_sub_bf16,
                                moe_elems);
            if (sync_timers) {
                sub_moe_ar_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_mar0).count();
            }
            if (any_p150 && shm) {
                const auto tw0 = std::chrono::steady_clock::now();
                while (shm->resp_seq.load(std::memory_order_acquire) != posted_p150_seq) {
                    dsv41_cpu_relax();
                }
                p150_wait_this_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw0).count();
                accum_p150_wait_ms += p150_wait_this_ms;
                if (shm->status != 0) {
                    throw std::runtime_error("p150_dsv41_worker returned error status " + std::to_string(shm->status));
                }
                accum_p150_dev_ms += double(shm->last_device_ns) * 1e-6;
                for (int tp = 0; tp < 3; ++tp) {
                    DSV41_HIP_CHECK(hipSetDevice(tp));
                    auto& c = ctx[tp];
                    DSV41_HIP_CHECK(hipMemcpyAsync(c.d_p150_out_bf16, shm->y_out_bf16,
                                                   moe_elems * sizeof(uint16_t),
                                                   hipMemcpyHostToDevice, c.stream));
                    launch_add_bf16_2way_inplace(c.d_sub_bf16, c.d_p150_out_bf16, moe_elems, c.stream);
                }
            }
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                auto& c = ctx[tp];
                launch_hc_post(c.d_sub_bf16, c.d_h_bf16[in_h_slot], c.d_post_mix_f32, c.d_comb_mix_f32,
                               c.d_h_bf16[out_h_slot], M, c.stream);
            }
        }
        sync_tp0_if_profiling();
        accum_mi50_moe_ms += std::max(0.0, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_moe0).count() - p150_wait_this_ms);
    }

    // Enqueue async CPU gather + PCIe H2D prefetch for Engram slot (0=L1, 1=L14)
    void enqueue_engram_prefetch(int slot, int engram_idx, int M) {
        if (!engram_cpu.has_table_shard(engram_idx) || M <= 0) return;
        {
            std::lock_guard<std::mutex> lk(eg_mtx);
            eg_slot_ready[slot] = false;
            eg_queue.push_back(EngramJob{slot, engram_idx, M});
        }
        eg_cv.notify_one();
    }

    // Consume prefetched Engram slot + run FP8 wkv + TP3 all-reduce + gated residual add
    void forward_engram_layer(int slot, int engram_idx, int h_slot, int M) {
        if (!engram_cpu.has_table_shard(engram_idx) || M <= 0) return;
        const auto te0 = std::chrono::steady_clock::now();
        {
            std::unique_lock<std::mutex> lk(eg_mtx);
            eg_done_cv.wait(lk, [&]() { return eg_slot_ready[slot]; });
        }
        accum_engram_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - te0).count();

        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            auto& c = ctx[tp];
            DSV41_HIP_CHECK(hipStreamWaitEvent(c.stream, c.ev_engram_ready[slot], 0));
        }

        for (int m0 = 0; m0 < M; m0 += 512) {
            const int sub_m = std::min(512, M - m0);
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                auto& c = ctx[tp];
                const auto& eg = c.engram[engram_idx];
                launch_act_quant_ue8m0_32(c.d_engram_in_bf16[slot] + size_t(m0) * 2048u,
                                          c.d_fp8_act, c.d_fp8_scale, nullptr,
                                          sub_m, 2048, false, c.stream);
                if (m0 + 512 >= M) {
                    DSV41_HIP_CHECK(hipEventRecord(c.ev_engram_consumed[slot], c.stream));
                }
                launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, eg.d_wkv_w, eg.d_wkv_s, nullptr, c.d_part_f32,
                                sub_m, 25600, 2048, true, false, c.stream);
            }

            all_reduce_tp3_f32(ctx[0].d_part_f32, ctx[1].d_part_f32, ctx[2].d_part_f32,
                               ctx[0].d_sum_f32,  ctx[1].d_sum_f32,  ctx[2].d_sum_f32,
                               size_t(sub_m) * 25600u);

            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                auto& c = ctx[tp];
                const auto& eg = c.engram[engram_idx];
                launch_engram_gate_add(c.d_h_bf16[h_slot] + size_t(m0) * kHcMult * kDim,
                                       c.d_sum_f32, eg.d_q_w, eg.d_k_w, nullptr,
                                       sub_m, kNormEps, c.stream);
            }
        }
    }

    // Forward backbone chunk [start_pos .. start_pos + M - 1]:
    //   - encoder_only == true: strict CED 20-layer early-exit (layers 0..19 + layer 20 Compressor/Indexer.k only!)
    //   - skip_global_kv_write == true: 128-token SWA Bounded Replay (rebuilds d_win_kv without touching Global KV!)
    void forward_backbone_chunk(
        const int32_t* token_ids,
        int start_pos,
        int M,
        bool encoder_only,
        int compute_logits_rows,
        bool commit_window_immediately,
        bool capture_dspark_hidden,
        int chunk_swa_floor = 0,
        bool skip_global_kv_write = false,
        uint64_t tree_anc_mask64 = 0ULL,
        uint32_t tree_depth_nibbles = 0u,
        uint32_t tree_parent_nibbles = 0u,
        const int32_t* tree_parents = nullptr,
        const uint32_t* tree_depths = nullptr)
    {
        if (!skip_global_kv_write && M > 0) {
            const int b_first = start_pos / kRadixBlockTokens;
            const int b_last  = (start_pos + M - 1) / kRadixBlockTokens;
            for (int b = b_first; b <= b_last && size_t(b) < gpu_slot_radix_keys.size(); ++b) {
                gpu_slot_radix_keys[size_t(b)] = RadixKey128{};
            }
        }

        // 1. Compute CPU Engram 4-gram hashes & kick off async DDR4 gather + PCIe H2D for BOTH L1 and L14!
        if (tree_anc_mask64 != 0ULL && tree_parents != nullptr && tree_depths != nullptr) {
            engram_cpu.compute_hashes_tree(token_ids, tree_parents, tree_depths, M, start_pos, engram_hashes_buf);
        } else {
            engram_cpu.compute_hashes(token_ids, M, start_pos, nullptr, engram_hashes_buf);
        }
        enqueue_engram_prefetch(0, 0, M);
        enqueue_engram_prefetch(1, 1, M);

        // 2. Upload token_ids + TP3 Embedding
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            auto& c = ctx[tp];
            DSV41_HIP_CHECK(hipMemcpyAsync(c.d_token_ids, token_ids, size_t(M) * sizeof(int32_t),
                                           hipMemcpyHostToDevice, c.stream));
            launch_tp_embed_bf16(c.d_token_ids, c.d_embed_w, c.d_part_bf16,
                                 M, kDim, c.vocab_offset, c.vocab_rows, c.stream);
        }
        all_reduce_tp3_bf16(ctx[0].d_part_bf16, ctx[1].d_part_bf16, ctx[2].d_part_bf16,
                            ctx[0].d_x_bf16,    ctx[1].d_x_bf16,    ctx[2].d_x_bf16,
                            size_t(M) * kDim);

        int h_slot = 0;
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            auto& c = ctx[tp];
            launch_hc_init(c.d_x_bf16, c.d_h_bf16[0], c.d_pre_mix_f32[0], M, c.stream);
        }

        const int stop_layer = encoder_only ? 20 : kNumLayers;
        int shared_k_comp = 0;

        for (int l = 0; l < stop_layer; ++l) {
            if (l == 1)  forward_engram_layer(0, 0, h_slot, M);
            if (l == 14) forward_engram_layer(1, 1, h_slot, M);

            // Capture DSpark target layer input mean (tp0<-L37, tp1<-L38, tp2<-L39) BEFORE layer forward!
            if (capture_dspark_hidden && l >= 37 && l <= 39) {
                const int cap_rows = std::min(M, kMaxDecM);
                const int cap_off = M - cap_rows;
                const int owner_tp = l - 37;
                DSV41_HIP_CHECK(hipSetDevice(owner_tp));
                launch_hc_mean_bf16(ctx[owner_tp].d_h_bf16[h_slot] + size_t(cap_off) * kHcMult * kDim,
                                    ctx[owner_tp].d_target_mean_bf16,
                                    cap_rows, ctx[owner_tp].stream);
            }

            const int next_slot = h_slot ^ 1;

            // Attention sub-block: hc_mixes + fused hc_pre_rmsnorm
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                auto& c = ctx[tp];
                const auto& b = c.blocks[l];
                launch_hc_mixes(c.d_h_bf16[h_slot], b.d_hc_attn_fn, b.d_hc_attn_scale, b.d_hc_attn_base,
                                c.d_hc_rsqrt_f32, c.d_hc_mixes_f32,
                                c.d_pre_mix_f32[next_slot], c.d_post_mix_f32, c.d_comb_mix_f32,
                                M, kNormEps, kHcSinkhornIters, kHcEps, c.stream);
                launch_hc_pre_rmsnorm_bf16(c.d_h_bf16[h_slot], c.d_pre_mix_f32[h_slot], b.d_attn_norm,
                                           c.d_x_bf16, nullptr, M, kNormEps, c.stream);
            }

            forward_backbone_attention(l, start_pos, M, h_slot, next_slot, chunk_swa_floor,
                                       commit_window_immediately, skip_global_kv_write, shared_k_comp,
                                       tree_anc_mask64, tree_depth_nibbles, tree_parent_nibbles);

            // hc_mixes & fused hc_pre_rmsnorm + register WHT128 before MoE
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                auto& c = ctx[tp];
                const auto& b = c.blocks[l];
                launch_hc_mixes(c.d_h_bf16[next_slot], b.d_hc_ffn_fn, b.d_hc_ffn_scale, b.d_hc_ffn_base,
                                c.d_hc_rsqrt_f32, c.d_hc_mixes_f32,
                                c.d_pre_mix_f32[h_slot], c.d_post_mix_f32, c.d_comb_mix_f32,
                                M, kNormEps, kHcSinkhornIters, kHcEps, c.stream);
                launch_hc_pre_rmsnorm_bf16(c.d_h_bf16[next_slot], c.d_pre_mix_f32[next_slot], b.d_ffn_norm,
                                           c.d_x_bf16, nullptr, M, kNormEps, c.stream,
                                           b.d_signs_h, c.d_x_rot_f16, (tp == 0) ? c.d_part_bf16 : nullptr);
            }

            forward_block_moe(l, M, next_slot, h_slot);
        }

        if (encoder_only) {
            if (!skip_global_kv_write) {
                forward_layer20_compressor_only(h_slot, start_pos, M);
            }
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                DSV41_HIP_CHECK(hipStreamSynchronize(ctx[tp].stream));
            }
            return;
        }

        // Final Head: compute logits for last 1 row or all M<=6 rows into host_full_logits
        if (compute_logits_rows > 0) {
            const int rows_out = std::min(compute_logits_rows, M);
            const int row_off = M - rows_out;
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                auto& c = ctx[tp];
                launch_hc_pre_rmsnorm_bf16(c.d_h_bf16[h_slot] + size_t(row_off) * kHcMult * kDim,
                                           c.d_pre_mix_f32[h_slot] + size_t(row_off) * kHcMult,
                                           c.d_norm_w, c.d_x_bf16, nullptr, rows_out, kNormEps, c.stream);
                launch_bf16_gemm(c.d_x_bf16, c.d_head_w, nullptr, c.d_logits_part_f32,
                                 rows_out, c.vocab_rows, kDim, true, c.stream);
                DSV41_HIP_CHECK(hipMemcpyAsync(c.h_logits_part, c.d_logits_part_f32,
                                               size_t(rows_out) * size_t(c.vocab_rows) * sizeof(float),
                                               hipMemcpyDeviceToHost, c.stream));
            }
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                DSV41_HIP_CHECK(hipStreamSynchronize(ctx[tp].stream));
                for (int r = 0; r < rows_out; ++r) {
                    std::memcpy(host_full_logits.data() + size_t(r) * kVocabSize + size_t(ctx[tp].vocab_offset),
                                ctx[tp].h_logits_part + size_t(r) * size_t(ctx[tp].vocab_rows),
                                size_t(ctx[tp].vocab_rows) * sizeof(float));
                }
            }
        } else {
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                DSV41_HIP_CHECK(hipStreamSynchronize(ctx[tp].stream));
            }
        }
    }

    // Commit ONLY `accepted_A` (1 <= accepted_A <= M_verified) verified tokens to backbone window + r=2 state!
    void commit_verified_backbone_state(int start_pos, int M_verified, int accepted_A) {
        const int new_pos = start_pos + accepted_A;
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            auto& c = ctx[tp];
            for (int l = 0; l < kNumLayers; ++l) {
                write_ring_window_gpu(tp, l, c.d_last_win_kv_bf16[l], start_pos, accepted_A);
            }
            // Restore exact r=2 Compressor partial group state if A < M_verified!
            if (accepted_A < M_verified) {
                if ((new_pos & 1) == 1) {
                    // Trailing token at new_pos - 1 (= start_pos + accepted_A - 1) is EVEN and was row (accepted_A - 1)!
                    const int row_idx = accepted_A - 1;
                    for (int s = 0; s < 3; ++s) {
                        DSV41_HIP_CHECK(hipMemcpyAsync(c.d_comp_kv_state[s],
                                                       c.d_comp_kv_last[s] + size_t(row_idx) * kHeadDim,
                                                       kHeadDim * sizeof(float), hipMemcpyDeviceToDevice, c.stream));
                        DSV41_HIP_CHECK(hipMemcpyAsync(c.d_comp_score_state[s],
                                                       c.d_comp_score_last[s] + size_t(row_idx) * kHeadDim,
                                                       kHeadDim * sizeof(float), hipMemcpyDeviceToDevice, c.stream));
                    }
                }
            }
        }
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            DSV41_HIP_CHECK(hipStreamSynchronize(ctx[tp].stream));
        }
    }

    // Alias-free two-kernel GPU commit of ONLY the verified root-to-leaf path `accepted_nodes[0..accepted_A-1]`
    // across all 40 SWA rings, Global FP4/UE8M0 KV + Indexer K (slots 0..3), r=2 compressor state,
    // DSpark target hidden rows [8..8+accepted_A-1], and CPU Engram `seq_cache_`!
    void commit_verified_tree_backbone_state(
        int start_pos,
        int accepted_A,
        const uint32_t* accepted_nodes,
        const int32_t* accepted_tokens)
    {
        uint32_t acc_nodes_nibbles = 0u;
        for (int d = 0; d < accepted_A; ++d) {
            acc_nodes_nibbles |= ((accepted_nodes[d] & 0xFu) << (4 * d));
        }
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            auto& c = ctx[tp];
            launch_commit_tree_win_kv_and_hidden(
                c.d_win_kv_slab, c.d_last_win_kv_slab, c.d_target_mean_bf16,
                start_pos, accepted_A, acc_nodes_nibbles, c.stream);
            launch_commit_tree_global_kv_and_r2(
                c.d_comp_kv_fp4[0], c.d_comp_kv_e4m3[0], c.d_index_k_fp4[0], c.d_index_k_ue8m0[0],
                c.d_comp_kv_fp4[1], c.d_comp_kv_e4m3[1], c.d_index_k_fp4[1], c.d_index_k_ue8m0[1],
                c.d_comp_kv_fp4[2], c.d_comp_kv_e4m3[2], c.d_index_k_fp4[2], c.d_index_k_ue8m0[2],
                c.d_comp_kv_fp4[3], c.d_comp_kv_e4m3[3], c.d_index_k_fp4[3], c.d_index_k_ue8m0[3],
                c.d_comp_kv_state[0], c.d_comp_score_state[0], c.d_comp_kv_last[0], c.d_comp_score_last[0],
                c.d_comp_kv_state[1], c.d_comp_score_state[1], c.d_comp_kv_last[1], c.d_comp_score_last[1],
                c.d_comp_kv_state[2], c.d_comp_score_state[2], c.d_comp_kv_last[2], c.d_comp_score_last[2],
                start_pos, accepted_A, acc_nodes_nibbles, c.stream);
        }
        engram_cpu.commit_accepted_tokens(accepted_tokens, accepted_A, start_pos);
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            DSV41_HIP_CHECK(hipStreamSynchronize(ctx[tp].stream));
        }
    }

    // Fast AVX2 exact temperature + top-p Borel support filter writing IEEE FP64 target distribution (sum == 1.0)
    static void build_probs_f64_fast(
        const float* logits,
        float temperature,
        float top_p,
        double* __restrict__ p_out)
    {
        if (temperature <= 1e-5f) {
            int32_t best_id = 0;
            float best_val = logits[0];
            for (int i = 1; i < kVocabSize; ++i) {
                if (logits[i] > best_val) {
                    best_val = logits[i];
                    best_id = i;
                }
            }
            std::memset(p_out, 0, size_t(kVocabSize) * sizeof(double));
            p_out[best_id] = 1.0;
            return;
        }

        thread_local std::vector<float> probs_f32(kVocabSize);
        float* __restrict__ probs = probs_f32.data();
        const float inv_t = 1.0f / temperature;
        float max_l = logits[0];
        for (int i = 1; i < kVocabSize; ++i) {
            if (logits[i] > max_l) max_l = logits[i];
        }

        double sum_exp = 0.0;
        {
            const __m256 v_max   = _mm256_set1_ps(max_l);
            const __m256 v_inv_t = _mm256_set1_ps(inv_t);
            const __m256 v_lo    = _mm256_set1_ps(-88.0f);
            const __m256 v_hi    = _mm256_set1_ps(88.0f);
            const __m256 v_l2e   = _mm256_set1_ps(1.4426950408889634f);
            const __m256 v_c1    = _mm256_set1_ps(-0.693145751953125f);
            const __m256 v_c2    = _mm256_set1_ps(-1.428606765330187045e-6f);
            const __m256 v_p0    = _mm256_set1_ps(1.9875691500E-4f);
            const __m256 v_p1    = _mm256_set1_ps(1.3981999507E-3f);
            const __m256 v_p2    = _mm256_set1_ps(8.3334519073E-3f);
            const __m256 v_p3    = _mm256_set1_ps(4.1665795894E-2f);
            const __m256 v_p4    = _mm256_set1_ps(1.6666665459E-1f);
            const __m256 v_p5    = _mm256_set1_ps(5.0000001201E-1f);
            const __m256 v_one   = _mm256_set1_ps(1.0f);
            const __m256i v_127  = _mm256_set1_epi32(127);

            __m256d acc0 = _mm256_setzero_pd();
            __m256d acc1 = _mm256_setzero_pd();
            for (int i = 0; i < kVocabSize; i += 8) {
                const __m256 vx_raw = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(logits + i), v_max), v_inv_t);
                const __m256 mask   = _mm256_cmp_ps(vx_raw, v_lo, _CMP_GT_OQ);
                const __m256 vx     = _mm256_min_ps(_mm256_max_ps(vx_raw, v_lo), v_hi);
                const __m256 vn     = _mm256_round_ps(_mm256_mul_ps(vx, v_l2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                const __m256 vr     = _mm256_fmadd_ps(vn, v_c2, _mm256_fmadd_ps(vn, v_c1, vx));
                __m256 vy = _mm256_fmadd_ps(v_p0, vr, v_p1);
                vy = _mm256_fmadd_ps(vy, vr, v_p2);
                vy = _mm256_fmadd_ps(vy, vr, v_p3);
                vy = _mm256_fmadd_ps(vy, vr, v_p4);
                vy = _mm256_fmadd_ps(vy, vr, v_p5);
                vy = _mm256_fmadd_ps(vy, _mm256_mul_ps(vr, vr), _mm256_add_ps(vr, v_one));
                const __m256i pow2n = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(vn), v_127), 23);
                const __m256 vp     = _mm256_and_ps(_mm256_mul_ps(vy, _mm256_castsi256_ps(pow2n)), mask);
                _mm256_storeu_ps(probs + i, vp);
                acc0 = _mm256_add_pd(acc0, _mm256_cvtps_pd(_mm256_castps256_ps128(vp)));
                acc1 = _mm256_add_pd(acc1, _mm256_cvtps_pd(_mm256_extractf128_ps(vp, 1)));
            }
            alignas(32) double sbuf[4];
            _mm256_store_pd(sbuf, _mm256_add_pd(acc0, acc1));
            sum_exp = (sbuf[0] + sbuf[1]) + (sbuf[2] + sbuf[3]);
        }
        const double inv_sum_f64 = 1.0 / std::max(sum_exp, 1e-30);

        if (top_p > 0.0f && top_p < 1.0f) {
            thread_local std::vector<int32_t> cand;
            cand.clear();
            double cand_mass = 0.0;
            for (int i = 0; i < kVocabSize; ++i) {
                const double p = double(probs[i]) * inv_sum_f64;
                probs[i] = float(p);
                if (p > 1e-7) {
                    cand.push_back(i);
                    cand_mass += p;
                }
            }
            if (cand_mass < double(top_p) || cand.empty()) {
                cand.resize(kVocabSize);
                std::iota(cand.begin(), cand.end(), 0);
            }
            std::sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) {
                return (probs[a] != probs[b]) ? (probs[a] > probs[b]) : (a < b);
            });

            double cum = 0.0;
            size_t keep_n = 0;
            for (; keep_n < cand.size(); ++keep_n) {
                cum += double(probs[cand[keep_n]]);
                if (cum >= double(top_p)) {
                    ++keep_n;
                    break;
                }
            }
            if (keep_n == 0 && !cand.empty()) {
                keep_n = 1;
                cum = double(probs[cand[0]]);
            }

            std::memset(p_out, 0, size_t(kVocabSize) * sizeof(double));
            double exact_kept = 0.0;
            for (size_t j = 0; j < keep_n; ++j) {
                exact_kept += double(probs[cand[j]]);
            }
            const double renorm = 1.0 / std::max(exact_kept, 1e-30);
            for (size_t j = 0; j < keep_n; ++j) {
                p_out[cand[j]] = double(probs[cand[j]]) * renorm;
            }
            return;
        }

        double exact_sum = 0.0;
        for (int i = 0; i < kVocabSize; ++i) {
            const double p = double(probs[i]) * inv_sum_f64;
            p_out[i] = p;
            exact_sum += p;
        }
        const double renorm = 1.0 / std::max(exact_sum, 1e-30);
        for (int i = 0; i < kVocabSize; ++i) {
            p_out[i] *= renorm;
        }
    }

    // Project captured d_target_mean_bf16 [seqlen, 5120] (tp0:L37, tp1:L38, tp2:L39) -> d_main_x_bf16 & seed MTP window!
    void update_mtp_window_from_target_hidden(int start_pos, int seqlen, int row_offset = 0) {
        if (seqlen <= 0) return;
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            auto& c = ctx[tp];
            const auto& m0 = c.blocks[kNumLayers];
            const uint16_t* src_mean = c.d_target_mean_bf16 + size_t(row_offset) * kDim;
            launch_act_quant_ue8m0_32(src_mean, c.d_fp8_act, c.d_fp8_scale, nullptr, seqlen, kDim, false, c.stream);
            launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, m0.d_main_proj_w, m0.d_main_proj_s, nullptr, c.d_part_f32,
                            seqlen, kDim, kDim, true, false, c.stream);
        }
        all_reduce_tp3_f32(ctx[0].d_part_f32, ctx[1].d_part_f32, ctx[2].d_part_f32,
                           ctx[0].d_sum_f32,  ctx[1].d_sum_f32,  ctx[2].d_sum_f32,
                           size_t(seqlen) * kDim);

        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            auto& c = ctx[tp];
            const auto& m0 = c.blocks[kNumLayers];
            launch_f32_to_bf16(c.d_sum_f32, c.d_main_x_bf16, size_t(seqlen) * kDim, c.stream);
            launch_rmsnorm_bf16(c.d_main_x_bf16, m0.d_main_norm, c.d_main_x_bf16, seqlen, kDim, kNormEps, c.stream);

            // Seed all 3 MTP blocks' window_kv_cache!
            launch_act_quant_ue8m0_32(c.d_main_x_bf16, c.d_fp8_act, c.d_fp8_scale, nullptr, seqlen, kDim, false, c.stream);
            for (int m = 0; m < kNumMtpLayers; ++m) {
                const int bid = kNumLayers + m;
                const auto& mb = c.blocks[bid];
                launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, mb.d_wkv_w, mb.d_wkv_s, c.d_kv_cur_bf16, nullptr,
                                seqlen, kHeadDim, kDim, false, false, c.stream);
                launch_rmsnorm_bf16(c.d_kv_cur_bf16, mb.d_kv_norm, c.d_kv_cur_bf16, seqlen, kHeadDim, kNormEps, c.stream);
                launch_rope_tail64_bf16(c.d_kv_cur_bf16, false, seqlen, 1, kHeadDim, start_pos, 1, false, c.stream);
                launch_act_quant_ue8m0_32(c.d_kv_cur_bf16, nullptr, nullptr, c.d_kv_cur_bf16, seqlen, kHeadDim, true, c.stream);
                write_ring_window_gpu(tp, bid, c.d_kv_cur_bf16, start_pos, seqlen);
            }
        }
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            DSV41_HIP_CHECK(hipStreamSynchronize(ctx[tp].stream));
        }
        mtp_window_seeded = true;
    }
    // Fast single-threaded exact temperature + top-p distribution & sampling over 129,280 vocab (zero OpenMP overhead!)
    static int32_t build_probs_and_sample(
        const float* logits,
        float temperature,
        float top_p,
        std::vector<float>& out_probs,
        std::mt19937_64& rng)
    {
        if (temperature <= 1e-5f) {
            int32_t best_id = 0;
            float best_val = logits[0];
            for (int i = 1; i < kVocabSize; ++i) {
                if (logits[i] > best_val) {
                    best_val = logits[i];
                    best_id = i;
                }
            }
            return best_id;
        }

        out_probs.resize(kVocabSize);
        float* __restrict__ probs = out_probs.data();
        const float inv_t = 1.0f / temperature;
        float max_l = logits[0];
        for (int i = 1; i < kVocabSize; ++i) {
            if (logits[i] > max_l) max_l = logits[i];
        }

        double sum_exp = 0.0;
        {
            const __m256 v_max   = _mm256_set1_ps(max_l);
            const __m256 v_inv_t = _mm256_set1_ps(inv_t);
            const __m256 v_lo    = _mm256_set1_ps(-88.0f);
            const __m256 v_hi    = _mm256_set1_ps(88.0f);
            const __m256 v_l2e   = _mm256_set1_ps(1.4426950408889634f);
            const __m256 v_c1    = _mm256_set1_ps(-0.693145751953125f);
            const __m256 v_c2    = _mm256_set1_ps(-1.428606765330187045e-6f);
            const __m256 v_p0    = _mm256_set1_ps(1.9875691500E-4f);
            const __m256 v_p1    = _mm256_set1_ps(1.3981999507E-3f);
            const __m256 v_p2    = _mm256_set1_ps(8.3334519073E-3f);
            const __m256 v_p3    = _mm256_set1_ps(4.1665795894E-2f);
            const __m256 v_p4    = _mm256_set1_ps(1.6666665459E-1f);
            const __m256 v_p5    = _mm256_set1_ps(5.0000001201E-1f);
            const __m256 v_one   = _mm256_set1_ps(1.0f);
            const __m256i v_127  = _mm256_set1_epi32(127);

            __m256d acc0 = _mm256_setzero_pd();
            __m256d acc1 = _mm256_setzero_pd();
            // kVocabSize = 129280 = 16160 * 8 (exact multiple of 8!)
            for (int i = 0; i < kVocabSize; i += 8) {
                const __m256 vx_raw = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(logits + i), v_max), v_inv_t);
                const __m256 mask   = _mm256_cmp_ps(vx_raw, v_lo, _CMP_GT_OQ);
                const __m256 vx     = _mm256_min_ps(_mm256_max_ps(vx_raw, v_lo), v_hi);
                const __m256 vn     = _mm256_round_ps(_mm256_mul_ps(vx, v_l2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                const __m256 vr     = _mm256_fmadd_ps(vn, v_c2, _mm256_fmadd_ps(vn, v_c1, vx));
                __m256 vy = _mm256_fmadd_ps(v_p0, vr, v_p1);
                vy = _mm256_fmadd_ps(vy, vr, v_p2);
                vy = _mm256_fmadd_ps(vy, vr, v_p3);
                vy = _mm256_fmadd_ps(vy, vr, v_p4);
                vy = _mm256_fmadd_ps(vy, vr, v_p5);
                vy = _mm256_fmadd_ps(vy, _mm256_mul_ps(vr, vr), _mm256_add_ps(vr, v_one));
                const __m256i pow2n = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(vn), v_127), 23);
                const __m256 vp     = _mm256_and_ps(_mm256_mul_ps(vy, _mm256_castsi256_ps(pow2n)), mask);
                _mm256_storeu_ps(probs + i, vp);
                acc0 = _mm256_add_pd(acc0, _mm256_cvtps_pd(_mm256_castps256_ps128(vp)));
                acc1 = _mm256_add_pd(acc1, _mm256_cvtps_pd(_mm256_extractf128_ps(vp, 1)));
            }
            alignas(32) double sbuf[4];
            _mm256_store_pd(sbuf, _mm256_add_pd(acc0, acc1));
            sum_exp = (sbuf[0] + sbuf[1]) + (sbuf[2] + sbuf[3]);
        }
        const float inv_sum = float(1.0 / std::max(sum_exp, 1e-30));

        if (top_p > 0.0f && top_p < 1.0f) {
            thread_local std::vector<int32_t> cand;
            cand.clear();
            double cand_mass = 0.0;
            for (int i = 0; i < kVocabSize; ++i) {
                const float p = probs[i] * inv_sum;
                probs[i] = p;
                if (p > 1e-7f) {
                    cand.push_back(i);
                    cand_mass += double(p);
                }
            }
            if (cand_mass < double(top_p) || cand.empty()) {
                cand.resize(kVocabSize);
                std::iota(cand.begin(), cand.end(), 0);
            }
            std::sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) {
                return probs[a] > probs[b];
            });

            double cum = 0.0;
            size_t keep_n = 0;
            for (; keep_n < cand.size(); ++keep_n) {
                cum += double(probs[cand[keep_n]]);
                if (cum >= double(top_p)) {
                    ++keep_n;
                    break;
                }
            }
            if (keep_n == 0 && !cand.empty()) {
                keep_n = 1;
                cum = double(probs[cand[0]]);
            }

            thread_local std::vector<float> kept_vals;
            kept_vals.resize(keep_n);
            const float renorm = float(1.0 / std::max(cum, 1e-30));
            for (size_t j = 0; j < keep_n; ++j) {
                kept_vals[j] = probs[cand[j]] * renorm;
            }
            std::memset(probs, 0, size_t(kVocabSize) * sizeof(float));
            for (size_t j = 0; j < keep_n; ++j) {
                probs[cand[j]] = kept_vals[j];
            }

            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float r = dist(rng);
            float acc = 0.0f;
            for (size_t j = 0; j < keep_n; ++j) {
                acc += kept_vals[j];
                if (r <= acc) return cand[j];
            }
            return cand[keep_n - 1];
        }

        for (int i = 0; i < kVocabSize; ++i) {
            probs[i] *= inv_sum;
        }

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng);
        float acc = 0.0f;
        int32_t last_nz = 0;
        for (int i = 0; i < kVocabSize; ++i) {
            const float p = probs[i];
            if (p > 0.0f) {
                last_nz = i;
                acc += p;
                if (r <= acc) return i;
            }
        }
        return last_nz;
    }

    static void apply_rep_penalty_span(
        float* logits,
        const std::vector<int32_t>& base_hist,
        const int32_t* extra_ptr,
        size_t extra_len,
        float penalty)
    {
        const size_t total_len = base_hist.size() + extra_len;
        if (penalty <= 1.0001f || total_len == 0u) return;
        thread_local std::vector<uint8_t> seen(kVocabSize, 0u);
        const size_t start_total = (total_len > 1024u) ? (total_len - 1024u) : 0u;
        const size_t base_start  = std::min(start_total, base_hist.size());
        const size_t extra_start = (start_total > base_hist.size()) ? (start_total - base_hist.size()) : 0u;
        for (size_t i = base_start; i < base_hist.size(); ++i) {
            const int32_t t = base_hist[i];
            if (t >= 0 && t < kVocabSize && !seen[t]) {
                seen[t] = 1u;
                logits[t] = (logits[t] > 0.0f) ? (logits[t] / penalty) : (logits[t] * penalty);
            }
        }
        for (size_t i = extra_start; i < extra_len; ++i) {
            const int32_t t = extra_ptr[i];
            if (t >= 0 && t < kVocabSize && !seen[t]) {
                seen[t] = 1u;
                logits[t] = (logits[t] > 0.0f) ? (logits[t] / penalty) : (logits[t] * penalty);
            }
        }
        for (size_t i = base_start; i < base_hist.size(); ++i) {
            const int32_t t = base_hist[i];
            if (t >= 0 && t < kVocabSize) seen[t] = 0u;
        }
        for (size_t i = extra_start; i < extra_len; ++i) {
            const int32_t t = extra_ptr[i];
            if (t >= 0 && t < kVocabSize) seen[t] = 0u;
        }
    }

    static void apply_rep_penalty_inplace(float* logits, const std::vector<int32_t>& history, float penalty) {
        apply_rep_penalty_span(logits, history, nullptr, 0u, penalty);
    }

    void run_mtp_blocks_capture_only(int32_t anchor_token, int cur_pos) {
        constexpr int B = kDSparkBlockSize;
        int32_t draft_in[B] = {anchor_token, kDSparkNoiseId, kDSparkNoiseId, kDSparkNoiseId, kDSparkNoiseId};
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            auto& c = ctx[tp];
            DSV41_HIP_CHECK(hipMemcpyAsync(c.d_token_ids, draft_in, B * sizeof(int32_t), hipMemcpyHostToDevice, c.stream));
            launch_tp_embed_bf16(c.d_token_ids, c.d_embed_w, c.d_part_bf16, B, kDim, c.vocab_offset, c.vocab_rows, c.stream);
        }
        all_reduce_tp3_bf16(ctx[0].d_part_bf16, ctx[1].d_part_bf16, ctx[2].d_part_bf16,
                            ctx[0].d_x_bf16,    ctx[1].d_x_bf16,    ctx[2].d_x_bf16, size_t(B) * kDim);
        int h_slot = 0;
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            launch_hc_init(ctx[tp].d_x_bf16, ctx[tp].d_h_bf16[0], ctx[tp].d_pre_mix_f32[0], B, ctx[tp].stream);
        }
        for (int m = 0; m < kNumMtpLayers; ++m) {
            const int bid = kNumLayers + m;
            const int next_slot = h_slot ^ 1;
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                auto& c = ctx[tp];
                const auto& b = c.blocks[bid];
                launch_hc_mixes(c.d_h_bf16[h_slot], b.d_hc_attn_fn, b.d_hc_attn_scale, b.d_hc_attn_base,
                                c.d_hc_rsqrt_f32, c.d_hc_mixes_f32,
                                c.d_pre_mix_f32[next_slot], c.d_post_mix_f32, c.d_comb_mix_f32,
                                B, kNormEps, kHcSinkhornIters, kHcEps, c.stream);
                launch_hc_pre_rmsnorm_bf16(c.d_h_bf16[h_slot], c.d_pre_mix_f32[h_slot], b.d_attn_norm,
                                           c.d_x_bf16, nullptr, B, kNormEps, c.stream);
                launch_act_quant_ue8m0_32(c.d_x_bf16, c.d_fp8_act, c.d_fp8_scale, nullptr, B, kDim, false, c.stream);
                launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, b.d_wqkv_w, b.d_wqkv_s, c.d_qkv_bf16, nullptr,
                                B, kQkvConcatRows, kDim, false, false, c.stream);
                launch_split_qkv_norm_rope_quant(c.d_qkv_bf16, b.d_q_norm, b.d_kv_norm, false,
                                                 c.d_qr_bf16, c.d_kv_cur_bf16, B, cur_pos, kNormEps, c.stream);
                launch_act_quant_ue8m0_32(c.d_qr_bf16, c.d_fp8_act, c.d_fp8_scale, nullptr, B, kQLoraRank, false, c.stream);
                launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, b.d_wq_b_w, b.d_wq_b_s, c.d_q_bf16, nullptr,
                                B, b.h_cnt * kHeadDim, kQLoraRank, false, false, c.stream);
                launch_rope_tail64_bf16(c.d_q_bf16, false, B, b.h_cnt, kHeadDim, cur_pos, 1, false, c.stream);
                launch_sparse_attn(c.d_q_bf16, b.d_attn_sink, c.d_win_kv[bid], c.d_kv_cur_bf16,
                                   cur_pos, 0, true, nullptr, nullptr, nullptr, 0, nullptr,
                                   c.d_o_bf16, B, b.h_cnt, c.stream);
                launch_rope_tail64_bf16(c.d_o_bf16, false, B, b.h_cnt, kHeadDim, cur_pos, 1, true, c.stream);
                launch_act_quant_ue8m0_32(c.d_o_bf16, c.d_fp8_act, c.d_fp8_scale, nullptr, B, b.g_cnt * 4096, false, c.stream);
                launch_fp8_grouped_wo_a(c.d_fp8_act, c.d_fp8_scale, b.d_wo_a_w, b.d_wo_a_s, c.d_wo_mid_bf16, B, b.g_cnt, c.stream);
                launch_act_quant_ue8m0_32(c.d_wo_mid_bf16, c.d_fp8_act, c.d_fp8_scale, nullptr, B, b.g_cnt * kOLoraRank, false, c.stream);
                launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, b.d_wo_b_w, b.d_wo_b_s, nullptr, c.d_part_f32,
                                B, kDim, b.g_cnt * kOLoraRank, true, false, c.stream);
            }
            all_reduce_tp3_f32_hc_post(ctx[0].d_part_f32, ctx[1].d_part_f32, ctx[2].d_part_f32,
                                       nullptr, nullptr, nullptr, h_slot, next_slot, B);
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                auto& c = ctx[tp];
                const auto& b = c.blocks[bid];
                launch_hc_mixes(c.d_h_bf16[next_slot], b.d_hc_ffn_fn, b.d_hc_ffn_scale, b.d_hc_ffn_base,
                                c.d_hc_rsqrt_f32, c.d_hc_mixes_f32,
                                c.d_pre_mix_f32[h_slot], c.d_post_mix_f32, c.d_comb_mix_f32,
                                B, kNormEps, kHcSinkhornIters, kHcEps, c.stream);
                launch_hc_pre_rmsnorm_bf16(c.d_h_bf16[next_slot], c.d_pre_mix_f32[next_slot], b.d_ffn_norm,
                                           c.d_x_bf16, nullptr, B, kNormEps, c.stream,
                                           b.d_signs_h, c.d_x_rot_f16, nullptr);
            }
            forward_block_moe(bid, B, next_slot, h_slot);
        }
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            DSV41_HIP_CHECK(hipStreamSynchronize(ctx[tp].stream));
        }
    }

    void run_rarebal_capture_on_prompt(const std::vector<int32_t>& prompt_ids, const ProgressCallback& on_progress) {
        if (!rarebal_collector || rarebal_collector->targets_satisfied() || prompt_ids.size() < 1024u) return;
        const int P = static_cast<int>(prompt_ids.size());
        auto reset_kv = [&]() {
            clear_gpu_kv_state();
            std::fill(gpu_slot_radix_keys.begin(), gpu_slot_radix_keys.end(), RadixKey128{});
        };
        auto run_span = [&](int span_begin, int span_end, bool check_done) {
            reset_kv();
            int rel_pos = 0;
            for (int gpos = span_begin; gpos + 16 <= span_end; ) {
                if (check_done && rarebal_collector->targets_satisfied()) break;
                const int chunk_m = std::min(256, ((span_end - gpos) / 2) * 2);
                if (chunk_m < 16) break;
                if (rel_pos + chunk_m > 4096) { reset_kv(); rel_pos = 0; }

                rarebal_collector->set_batch_tokens(prompt_ids.data() + gpos, chunk_m);
                forward_backbone_chunk(prompt_ids.data() + gpos, rel_pos, chunk_m, false, 0, true, true, 0, false);

                int prev_seeded = 0;
                for (int off = 4; off + 5 < chunk_m; off += 5) {
                    const int seed_n = (off + 1) - prev_seeded;
                    update_mtp_window_from_target_hidden(rel_pos + prev_seeded, seed_n, prev_seeded);
                    prev_seeded = off + 1;
                    rarebal_collector->set_batch_tokens(prompt_ids.data() + gpos + off + 1, 5);
                    run_mtp_blocks_capture_only(prompt_ids[size_t(gpos + off)], rel_pos + off);
                }
                if (prev_seeded < chunk_m) {
                    update_mtp_window_from_target_hidden(rel_pos + prev_seeded, chunk_m - prev_seeded, prev_seeded);
                }
                rel_pos += chunk_m;
                gpos += chunk_m;
                if (on_progress) {
                    Dsv41ProgressUpdate upd{};
                    upd.phase = Dsv41ProgressPhase::PrefillChunk;
                    upd.prompt_tokens = P;
                    if (!on_progress(upd)) break;
                }
            }
        };

        if (!rarebal_collector->candidates_locked) {
            rarebal_collector->active = true;
            rarebal_collector->pass0_only = true;
            if (P >= 8192) {
                run_span(P / 4, P / 4 + 2048, false);
                run_span((3 * P) / 4 - 2048, (3 * P) / 4, false);
            } else {
                run_span(0, std::min(P, 4096), false);
            }
            rarebal_collector->lock_candidates();
            for (auto& r : rarebal_collector->router_hits) r.fill(0ULL);
            rarebal_collector->total_rows_seen.fill(0ULL);
        }

        rarebal_collector->active = true;
        rarebal_collector->pass0_only = false;
        run_span(0, P, true);
        rarebal_collector->active = false;
        rarebal_collector->flush_to_disk();
        reset_kv();
    }

    // Run 3-stage DSpark draft blocks (mtp_0..2) from `anchor_token` at `cur_pos` (zero staging, zero MoE host sync!),
    // with confidence-gated GPU-resident deterministic M=8 width-[3,2,1,1] causal-tree expansion when p_top1 < 0.72!
    int forward_dspark_draft(
        int32_t anchor_token,
        int cur_pos,
        float temperature,
        float top_p,
        float rep_penalty,
        std::mt19937_64& rng,
        std::array<int32_t, kDSparkBlockSize>& out_drafts,
        bool allow_tree8 = false,
        int rem_budget = 0,
        bool* out_used_tree8 = nullptr,
        std::array<int32_t, kTree8NumNodes>* out_tree_tokens = nullptr)
    {
        if (out_used_tree8) *out_used_tree8 = false;
        constexpr int B = kDSparkBlockSize;  // 5
        int32_t draft_in[B] = {anchor_token, kDSparkNoiseId, kDSparkNoiseId, kDSparkNoiseId, kDSparkNoiseId};

        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            auto& c = ctx[tp];
            DSV41_HIP_CHECK(hipMemcpyAsync(c.d_token_ids, draft_in, B * sizeof(int32_t),
                                           hipMemcpyHostToDevice, c.stream));
            launch_tp_embed_bf16(c.d_token_ids, c.d_embed_w, c.d_part_bf16,
                                 B, kDim, c.vocab_offset, c.vocab_rows, c.stream);
        }
        all_reduce_tp3_bf16(ctx[0].d_part_bf16, ctx[1].d_part_bf16, ctx[2].d_part_bf16,
                            ctx[0].d_x_bf16,    ctx[1].d_x_bf16,    ctx[2].d_x_bf16,
                            size_t(B) * kDim);

        int h_slot = 0;
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            launch_hc_init(ctx[tp].d_x_bf16, ctx[tp].d_h_bf16[0], ctx[tp].d_pre_mix_f32[0], B, ctx[tp].stream);
        }

        // Run mtp_0, mtp_1, mtp_2 (blocks 40, 41, 42) entirely on GPU!
        for (int m = 0; m < kNumMtpLayers; ++m) {
            const int bid = kNumLayers + m;
            const int next_slot = h_slot ^ 1;

            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                auto& c = ctx[tp];
                const auto& b = c.blocks[bid];
                launch_hc_mixes(c.d_h_bf16[h_slot], b.d_hc_attn_fn, b.d_hc_attn_scale, b.d_hc_attn_base,
                                c.d_hc_rsqrt_f32, c.d_hc_mixes_f32,
                                c.d_pre_mix_f32[next_slot], c.d_post_mix_f32, c.d_comb_mix_f32,
                                B, kNormEps, kHcSinkhornIters, kHcEps, c.stream);
                launch_hc_pre_rmsnorm_bf16(c.d_h_bf16[h_slot], c.d_pre_mix_f32[h_slot], b.d_attn_norm,
                                           c.d_x_bf16, nullptr, B, kNormEps, c.stream);

                // Fused [wq_a; wkv] + split_qkv_norm_rope_quant + zero-staging bidirectional DSparkAttention!
                launch_act_quant_ue8m0_32(c.d_x_bf16, c.d_fp8_act, c.d_fp8_scale, nullptr, B, kDim, false, c.stream);
                launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, b.d_wqkv_w, b.d_wqkv_s, c.d_qkv_bf16, nullptr,
                                B, kQkvConcatRows, kDim, false, false, c.stream);
                launch_split_qkv_norm_rope_quant(c.d_qkv_bf16, b.d_q_norm, b.d_kv_norm, false,
                                                 c.d_qr_bf16, c.d_kv_cur_bf16, B, cur_pos, kNormEps, c.stream);

                launch_act_quant_ue8m0_32(c.d_qr_bf16, c.d_fp8_act, c.d_fp8_scale, nullptr, B, kQLoraRank, false, c.stream);
                launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, b.d_wq_b_w, b.d_wq_b_s, c.d_q_bf16, nullptr,
                                B, b.h_cnt * kHeadDim, kQLoraRank, false, false, c.stream);
                launch_rope_tail64_bf16(c.d_q_bf16, false, B, b.h_cnt, kHeadDim, cur_pos, 1, false, c.stream);

                launch_sparse_attn(c.d_q_bf16, b.d_attn_sink,
                                   c.d_win_kv[bid], c.d_kv_cur_bf16,
                                   cur_pos, 0, true,
                                   nullptr, nullptr, nullptr, 0,
                                   nullptr,
                                   c.d_o_bf16, B, b.h_cnt, c.stream);
                launch_rope_tail64_bf16(c.d_o_bf16, false, B, b.h_cnt, kHeadDim, cur_pos, 1, true, c.stream);
                launch_act_quant_ue8m0_32(c.d_o_bf16, c.d_fp8_act, c.d_fp8_scale, nullptr, B, b.g_cnt * 4096, false, c.stream);
                launch_fp8_grouped_wo_a(c.d_fp8_act, c.d_fp8_scale, b.d_wo_a_w, b.d_wo_a_s, c.d_wo_mid_bf16, B, b.g_cnt, c.stream);
                launch_act_quant_ue8m0_32(c.d_wo_mid_bf16, c.d_fp8_act, c.d_fp8_scale, nullptr, B, b.g_cnt * kOLoraRank, false, c.stream);
                launch_fp8_gemm(c.d_fp8_act, c.d_fp8_scale, b.d_wo_b_w, b.d_wo_b_s, nullptr, c.d_part_f32,
                                B, kDim, b.g_cnt * kOLoraRank, true, false, c.stream);
            }
            all_reduce_tp3_f32_hc_post(ctx[0].d_part_f32, ctx[1].d_part_f32, ctx[2].d_part_f32,
                                       nullptr, nullptr, nullptr, h_slot, next_slot, B);

            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                auto& c = ctx[tp];
                const auto& b = c.blocks[bid];
                launch_hc_mixes(c.d_h_bf16[next_slot], b.d_hc_ffn_fn, b.d_hc_ffn_scale, b.d_hc_ffn_base,
                                c.d_hc_rsqrt_f32, c.d_hc_mixes_f32,
                                c.d_pre_mix_f32[h_slot], c.d_post_mix_f32, c.d_comb_mix_f32,
                                B, kNormEps, kHcSinkhornIters, kHcEps, c.stream);
                launch_hc_pre_rmsnorm_bf16(c.d_h_bf16[next_slot], c.d_pre_mix_f32[next_slot], b.d_ffn_norm,
                                           c.d_x_bf16, nullptr, B, kNormEps, c.stream,
                                           b.d_signs_h, c.d_x_rot_f16, nullptr);
            }

            forward_block_moe(bid, B, next_slot, h_slot);
        }

        // Base MTP head logits [5, vocab_rows] on GPU (also saves raw hc_pre in tp=0 d_sub_bf16 for ConfidenceHead
        // and preserves unmutated rows [1..2] at offset +8*vocab_rows for zero-alloc M=8 causal-tree top-B expansion!)
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            auto& c = ctx[tp];
            const auto& m2 = c.blocks[kNumLayers + 2];
            launch_hc_pre_rmsnorm_bf16(c.d_h_bf16[h_slot], c.d_pre_mix_f32[h_slot], m2.d_mtp_final_norm,
                                       c.d_x_bf16, (tp == 0) ? c.d_sub_bf16 : nullptr, B, kNormEps, c.stream);
            launch_bf16_gemm(c.d_x_bf16, c.d_head_w, nullptr, c.d_logits_part_f32,
                             B, c.vocab_rows, kDim, true, c.stream);
            if (allow_tree8 && rem_budget >= 4) {
                DSV41_HIP_CHECK(hipMemcpyAsync(c.d_logits_part_f32 + 8u * size_t(c.vocab_rows),
                                               c.d_logits_part_f32 + 1u * size_t(c.vocab_rows),
                                               2u * size_t(c.vocab_rows) * sizeof(float),
                                               hipMemcpyDeviceToDevice, c.stream));
            }
        }

        // GPU-Resident 5-step DSparkMarkovHead + Sliding-1024 RepPenalty + Borel Two-Tier Top-p/Greedy Sampler + GPU ConfidenceHead
        // Collapses 19 CPU-GPU syncs -> 1 single stream sync and 5.17 MB D2H -> 20 B D2H (0 B extra HBM2)!
        const uint64_t dseq = ++dspark_sample_seq;
        DsparkGpuSampleSlot* ctrl = h_dspark_ctrl;
        ctrl->call_seq = dseq;
        ctrl->anchor_token = anchor_token;
        const int R_hist = static_cast<int>(std::min<size_t>(1024u, cached_tokens.size()));
        ctrl->recent_len = R_hist;
        if (R_hist > 0) {
            std::memcpy(ctrl->recent_hist,
                        cached_tokens.data() + (cached_tokens.size() - size_t(R_hist)),
                        size_t(R_hist) * sizeof(int32_t));
        }
        ctrl->rep_penalty = rep_penalty;
        ctrl->temperature = temperature;
        ctrl->top_p = top_p;
        {
            std::mt19937_64 rng_tmp = rng;
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (int i = 0; i < B; ++i) {
                ctrl->u_rand[i] = (temperature > 1e-5f) ? dist(rng_tmp) : 0.0f;
                ctrl->out_drafts[i] = -1;
                ctrl->out_q_selected[i] = 0.0f;
                ctrl->kept_cnt[i] = 0u;
            }
        }
        ctrl->out_drafted_cnt = 0;
        ctrl->out_fallback_step = -1;
        ctrl->out_root_top1_prob = -1.0f;
        std::atomic_thread_fence(std::memory_order_release);

        for (int i = 0; i < B; ++i) {
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                auto& c = ctx[tp];
                const auto& m2 = c.blocks[kNumLayers + 2];
                uint16_t* emb_dst = c.d_markov_sum_bf16 + size_t(i) * kMarkovRank;
                launch_dspark_prepare_markov_emb(
                    ctrl, h_markov_embed_mapped, emb_dst,
                    dseq, i, Dsv41SpecialTokens::kEos, c.stream);
                launch_bf16_gemm(emb_dst, m2.d_markov_head_w,
                                 nullptr, c.d_markov_logits_f32,
                                 1, c.vocab_rows, kMarkovRank, true, c.stream);
                launch_dspark_step_sample_tp3(
                    ctrl,
                    c.d_logits_part_f32 + size_t(i) * size_t(c.vocab_rows),
                    c.d_markov_logits_f32,
                    dseq, i, tp, c.vocab_offset, c.vocab_rows,
                    Dsv41SpecialTokens::kEos, c.stream);
            }
        }

        // Enqueue ConfidenceHead on tp=0 before the single stream sync (20 B D2H!)
        {
            DSV41_HIP_CHECK(hipSetDevice(0));
            auto& c0 = ctx[0];
            const auto& m2_0 = c0.blocks[kNumLayers + 2];
            launch_dspark_pack_conf_in(ctrl, c0.d_sub_bf16, c0.d_markov_sum_bf16, c0.d_conf_in_bf16, c0.stream);
            launch_bf16_gemm(c0.d_conf_in_bf16, m2_0.d_conf_proj_w, nullptr, c0.d_conf_out_f32,
                             B, 1, kDim + kMarkovRank, true, c0.stream);
            DSV41_HIP_CHECK(hipMemcpyAsync(ctrl->out_conf_logits, c0.d_conf_out_f32,
                                           B * sizeof(float), hipMemcpyDeviceToHost, c0.stream));
            DSV41_HIP_CHECK(hipStreamSynchronize(c0.stream));
        }

        int drafted_cnt = 0;
        const int fb_step = ctrl->out_fallback_step;
        if (fb_step < 0) {
            const float p_root_top1 = (ctrl->out_root_top1_prob >= 0.0f)
                ? ctrl->out_root_top1_prob
                : (1.0f / (1.0f + std::exp(-ctrl->out_conf_logits[0])));
            if (allow_tree8 && out_used_tree8 && out_tree_tokens &&
                rem_budget >= 4 &&
                p_root_top1 < kTree8Top1ConfThresh &&
                ctrl->out_drafts[0] != Dsv41SpecialTokens::kEos)
            {
                // High-entropy confidence gate fired (p_root_top1 < 0.72): extract GPU-resident deterministic
                // width-[3,2,1,1] M=8 causal tree across all 3 MI50 ranks with 1 single stream sync (32 B D2H)!
                ctrl->tree_tokens[0] = anchor_token;
                std::atomic_thread_fence(std::memory_order_release);
                for (int tp = 0; tp < 3; ++tp) {
                    DSV41_HIP_CHECK(hipSetDevice(tp));
                    auto& c = ctx[tp];
                    const auto& m2 = c.blocks[kNumLayers + 2];
                    const size_t vr = size_t(c.vocab_rows);

                    // Stage 0: root top-3 {t1, t2, t3} from row 0 (already contains base_0 + markov_0 + rep_penalty)
                    launch_dspark_tree_topk_tp3(
                        ctrl, c.d_logits_part_f32, nullptr,
                        dseq, 0, 1, 3, -1, -1, tp, c.vocab_offset, c.vocab_rows, c.stream);

                    // Markov GEMM (M=2) for t1 and t2 -> Stages 1 & 2 ({t4, t5} under t1, {t6} under t2)
                    uint16_t* emb_t12 = c.d_markov_sum_bf16 + 8u * kMarkovRank;
                    launch_dspark_prepare_tree_markov_emb(
                        ctrl, h_markov_embed_mapped, emb_t12, dseq, 0, 2, c.stream);
                    launch_bf16_gemm(emb_t12, m2.d_markov_head_w,
                                     nullptr, c.d_markov_logits_f32,
                                     2, c.vocab_rows, kMarkovRank, true, c.stream);
                    launch_dspark_tree_topk_tp3(
                        ctrl, c.d_logits_part_f32 + 8u * vr, c.d_markov_logits_f32,
                        dseq, 1, 4, 2, 1, -1, tp, c.vocab_offset, c.vocab_rows, c.stream);
                    launch_dspark_tree_topk_tp3(
                        ctrl, c.d_logits_part_f32 + 8u * vr, c.d_markov_logits_f32 + vr,
                        dseq, 2, 6, 1, 2, -1, tp, c.vocab_offset, c.vocab_rows, c.stream);

                    // Markov GEMM (M=1) for t4 -> Stage 3 ({t7} under t4)
                    uint16_t* emb_t4 = c.d_markov_sum_bf16 + 10u * kMarkovRank;
                    launch_dspark_prepare_tree_markov_emb(
                        ctrl, h_markov_embed_mapped, emb_t4, dseq, 1, 1, c.stream);
                    launch_bf16_gemm(emb_t4, m2.d_markov_head_w,
                                     nullptr, c.d_markov_logits_f32 + 2u * vr,
                                     1, c.vocab_rows, kMarkovRank, true, c.stream);
                    launch_dspark_tree_topk_tp3(
                        ctrl, c.d_logits_part_f32 + 9u * vr, c.d_markov_logits_f32 + 2u * vr,
                        dseq, 3, 7, 1, 1, 4, tp, c.vocab_offset, c.vocab_rows, c.stream);
                }
                DSV41_HIP_CHECK(hipSetDevice(0));
                DSV41_HIP_CHECK(hipStreamSynchronize(ctx[0].stream));
                for (int m = 0; m < kTree8NumNodes; ++m) {
                    (*out_tree_tokens)[size_t(m)] = ctrl->tree_tokens[m];
                }
                *out_used_tree8 = true;
                return kTree8NumNodes - 1;  // 7 drafted tree candidates (M = 8 verify nodes)
            }

            drafted_cnt = std::max(1, std::min(B, int(ctrl->out_drafted_cnt)));
            if (temperature > 1e-5f) {
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                for (int i = 0; i < drafted_cnt; ++i) (void)dist(rng);
            }
            for (int i = 0; i < drafted_cnt; ++i) {
                out_drafts[i] = ctrl->out_drafts[i];
                draft_q_selected[i] = ctrl->out_q_selected[i];
                draft_q_materialized[i] = false;
            }
            if (drafted_cnt > 1) {
                float cum_conf = 1.0f;
                int keep_b = 1;
                for (int i = 0; i < drafted_cnt; ++i) {
                    const float p_acc = 1.0f / (1.0f + std::exp(-ctrl->out_conf_logits[i]));
                    cum_conf *= p_acc;
                    if (i >= 1 && cum_conf < 0.12f) break;
                    keep_b = i + 1;
                }
                drafted_cnt = keep_b;
            }
            return drafted_cnt;
        }

        // Rare (< 0.05% flat-entropy or top_p == 1.0) lossless host fallback from step `fb_step` onward
        for (int tp = 1; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            DSV41_HIP_CHECK(hipStreamSynchronize(ctx[tp].stream));
        }
        std::array<int32_t, 1 + B> tail_toks{};
        tail_toks[0] = anchor_token;
        size_t tail_len = 1u;
        {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (int i = 0; i < fb_step; ++i) {
                if (temperature > 1e-5f) (void)dist(rng);
                out_drafts[i] = ctrl->out_drafts[i];
                draft_q_selected[i] = ctrl->out_q_selected[i];
                draft_q_materialized[i] = false;
                tail_toks[tail_len++] = out_drafts[i];
                drafted_cnt = i + 1;
            }
        }
        int32_t prev_tok = tail_toks[tail_len - 1];
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            auto& c = ctx[tp];
            DSV41_HIP_CHECK(hipMemcpyAsync(c.h_logits_part,
                                           c.d_logits_part_f32 + size_t(fb_step) * size_t(c.vocab_rows),
                                           size_t(B - fb_step) * size_t(c.vocab_rows) * sizeof(float),
                                           hipMemcpyDeviceToHost, c.stream));
        }
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            DSV41_HIP_CHECK(hipStreamSynchronize(ctx[tp].stream));
            for (int r = fb_step; r < B; ++r) {
                std::memcpy(host_full_logits.data() + size_t(r) * kVocabSize + size_t(ctx[tp].vocab_offset),
                            ctx[tp].h_logits_part + size_t(r - fb_step) * size_t(ctx[tp].vocab_rows),
                            size_t(ctx[tp].vocab_rows) * sizeof(float));
            }
        }

        for (int i = fb_step; i < B; ++i) {
            float* row_logits = host_full_logits.data() + size_t(i) * kVocabSize;
            if (i > fb_step) {
                const uint16_t* emb_row = markov_embed_host_bf16.data() + size_t(prev_tok) * kMarkovRank;
                for (int tp = 0; tp < 3; ++tp) {
                    DSV41_HIP_CHECK(hipSetDevice(tp));
                    auto& c = ctx[tp];
                    const auto& m2 = c.blocks[kNumLayers + 2];
                    DSV41_HIP_CHECK(hipMemcpyAsync(c.d_markov_sum_bf16 + i * kMarkovRank, emb_row,
                                                   kMarkovRank * sizeof(uint16_t),
                                                   hipMemcpyHostToDevice, c.stream));
                    launch_bf16_gemm(c.d_markov_sum_bf16 + i * kMarkovRank, m2.d_markov_head_w,
                                     nullptr, c.d_markov_logits_f32, 1, c.vocab_rows, kMarkovRank, true, c.stream);
                    DSV41_HIP_CHECK(hipMemcpyAsync(c.h_logits_part, c.d_markov_logits_f32,
                                                   size_t(c.vocab_rows) * sizeof(float),
                                                   hipMemcpyDeviceToHost, c.stream));
                }
                for (int tp = 0; tp < 3; ++tp) {
                    DSV41_HIP_CHECK(hipSetDevice(tp));
                    DSV41_HIP_CHECK(hipStreamSynchronize(ctx[tp].stream));
                    const float* bias_part = ctx[tp].h_logits_part;
                    float* dst_slice = row_logits + ctx[tp].vocab_offset;
                    const int vr = ctx[tp].vocab_rows;
                    for (int v = 0; v < vr; ++v) dst_slice[v] += bias_part[v];
                }
                apply_rep_penalty_span(row_logits, cached_tokens, tail_toks.data(), tail_len, rep_penalty);
            }
            const int32_t next_d = build_probs_and_sample(row_logits, temperature, top_p, draft_q_probs[i], rng);
            out_drafts[i] = next_d;
            draft_q_selected[i] = (temperature <= 1e-5f) ? 1.0f : draft_q_probs[i][next_d];
            draft_q_materialized[i] = (temperature > 1e-5f);
            prev_tok = next_d;
            tail_toks[tail_len++] = next_d;
            drafted_cnt = i + 1;
            if (next_d == Dsv41SpecialTokens::kEos) {
                break;
            }
        }

        if (drafted_cnt > 1) {
            DSV41_HIP_CHECK(hipSetDevice(0));
            auto& c0 = ctx[0];
            const auto& m2 = c0.blocks[kNumLayers + 2];
            launch_dspark_pack_conf_in(ctrl, c0.d_sub_bf16, c0.d_markov_sum_bf16, c0.d_conf_in_bf16, c0.stream);
            launch_bf16_gemm(c0.d_conf_in_bf16, m2.d_conf_proj_w, nullptr, c0.d_conf_out_f32,
                             drafted_cnt, 1, kDim + kMarkovRank, true, c0.stream);
            float conf_logits[B]{};
            DSV41_HIP_CHECK(hipMemcpyAsync(conf_logits, c0.d_conf_out_f32,
                                           size_t(drafted_cnt) * sizeof(float), hipMemcpyDeviceToHost, c0.stream));
            DSV41_HIP_CHECK(hipStreamSynchronize(c0.stream));

            float cum_conf = 1.0f;
            int keep_b = 1;
            for (int i = 0; i < drafted_cnt; ++i) {
                const float p_acc = 1.0f / (1.0f + std::exp(-conf_logits[i]));
                cum_conf *= p_acc;
                if (i >= 1 && cum_conf < 0.12f) break;
                keep_b = i + 1;
            }
            drafted_cnt = keep_b;
        }
        return drafted_cnt;
    }

    // Clear persistent GPU SWA + r=2 state to cold start (preserving DDR4 RewindBank & RadixBlockCache!)
    void clear_gpu_kv_state() {
        cached_tokens.clear();
        win_valid_from_20_39 = 0;
        mtp_window_seeded = false;
        exported_radix_blocks = 0;
        exported_radix_key = RadixKey128{};
        for (int tp = 0; tp < 3; ++tp) {
            reset_r2_compressor_state_gpu(tp);
            DSV41_HIP_CHECK(hipSetDevice(tp));
            DSV41_HIP_CHECK(hipMemsetAsync(ctx[tp].d_win_kv_slab, 0,
                                           kWinKvSlabElems * sizeof(uint16_t), ctx[tp].stream));
            DSV41_HIP_CHECK(hipStreamSynchronize(ctx[tp].stream));
        }
    }

    // Save current SWA + r=2 state into 32-entry RewindBank (single-slab 5.375 MiB D2H on tp=0, < 0.4 ms!)
    void save_rewind_snapshot() {
        const int cur_pos = static_cast<int>(cached_tokens.size());
        if (cur_pos <= 0) return;
        for (const auto& s : rewind_bank) {
            if (s.pos == cur_pos && s.tokens == cached_tokens) return;
        }
        RewindSnapshot snap;
        snap.pos = cur_pos;
        snap.win_valid_from_20_39 = win_valid_from_20_39;
        snap.mtp_seeded = mtp_window_seeded;
        snap.tokens = cached_tokens;
        snap.win_kv_slab.resize(kWinKvSlabElems);
        snap.comp_state_slab.resize(kCompStateSlabElems);

        DSV41_HIP_CHECK(hipSetDevice(0));
        DSV41_HIP_CHECK(hipMemcpyAsync(snap.win_kv_slab.data(), ctx[0].d_win_kv_slab,
                                       kWinKvSlabElems * sizeof(uint16_t), hipMemcpyDeviceToHost, ctx[0].stream));
        DSV41_HIP_CHECK(hipMemcpyAsync(snap.comp_state_slab.data(), ctx[0].d_comp_kv_state_slab,
                                       kCompStateSlabElems * sizeof(float), hipMemcpyDeviceToHost, ctx[0].stream));
        DSV41_HIP_CHECK(hipStreamSynchronize(ctx[0].stream));

        if (rewind_bank.size() >= 32u) {
            rewind_bank.pop_front();
        }
        rewind_bank.push_back(std::move(snap));
    }

    // Restore SWA + r=2 state from RewindBank snapshot onto all 3 GPUs (< 0.5 ms!)
    void restore_rewind_snapshot_gpu(const RewindSnapshot& snap, bool also_restore_global_kv_from_radix,
                                     const std::vector<std::shared_ptr<RadixBlockEntry>>& radix_prefix)
    {
        if (also_restore_global_kv_from_radix && !radix_prefix.empty()) {
            restore_radix_blocks_gpu(radix_prefix);
        }
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            DSV41_HIP_CHECK(hipMemcpyAsync(ctx[tp].d_win_kv_slab, snap.win_kv_slab.data(),
                                           kWinKvSlabElems * sizeof(uint16_t), hipMemcpyHostToDevice, ctx[tp].stream));
            DSV41_HIP_CHECK(hipMemcpyAsync(ctx[tp].d_comp_kv_state_slab, snap.comp_state_slab.data(),
                                           kCompStateSlabElems * sizeof(float), hipMemcpyHostToDevice, ctx[tp].stream));
        }
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            DSV41_HIP_CHECK(hipStreamSynchronize(ctx[tp].stream));
        }
        cached_tokens = snap.tokens;
        win_valid_from_20_39 = snap.win_valid_from_20_39;
        mtp_window_seeded = snap.mtp_seeded;
        engram_cpu.compute_hashes(cached_tokens.data(), snap.pos, 0, nullptr, engram_hashes_buf);
        exported_radix_blocks = 0;
        exported_radix_key = RadixKey128{};
        const int full_blks = snap.pos / kRadixBlockTokens;
        for (int b = 0; b < full_blks; ++b) {
            exported_radix_key = step_radix_hash(exported_radix_key, cached_tokens.data() + b * kRadixBlockTokens);
            exported_radix_blocks = b + 1;
        }
    }

    // Export newly completed 128-token blocks [exported_radix_blocks .. cached_tokens.size()/128) to DDR4 + async NVMe!
    void export_new_radix_blocks() {
        const int total_blks = static_cast<int>(cached_tokens.size()) / kRadixBlockTokens;
        if (total_blks <= exported_radix_blocks) return;

        DSV41_HIP_CHECK(hipSetDevice(0));
        auto& c0 = ctx[0];

        while (exported_radix_blocks < total_blks) {
            const int blk = exported_radix_blocks;
            const int32_t* blk_toks = cached_tokens.data() + blk * kRadixBlockTokens;
            const RadixKey128 key = step_radix_hash(exported_radix_key, blk_toks);

            if (radix_map.find(key) == radix_map.end()) {
                auto entry = std::make_shared<RadixBlockEntry>();
                entry->key = key;
                std::memcpy(entry->tokens.data(), blk_toks, kRadixBlockTokens * sizeof(int32_t));

                size_t off = 0;
                for (int s = 0; s < 4; ++s) {
                    const size_t n_slots = (s < 3) ? 64u : 128u;
                    const size_t sb = size_t(blk) * n_slots;
                    DSV41_HIP_CHECK(hipMemcpyAsync(c0.h_radix_stage + off, c0.d_comp_kv_fp4[s] + sb * 256u,
                                                   n_slots * 256u, hipMemcpyDeviceToHost, c0.stream));
                    off += n_slots * 256u;
                    DSV41_HIP_CHECK(hipMemcpyAsync(c0.h_radix_stage + off, c0.d_comp_kv_e4m3[s] + sb * 32u,
                                                   n_slots * 32u, hipMemcpyDeviceToHost, c0.stream));
                    off += n_slots * 32u;
                    DSV41_HIP_CHECK(hipMemcpyAsync(c0.h_radix_stage + off, c0.d_index_k_fp4[s] + sb * 64u,
                                                   n_slots * 64u, hipMemcpyDeviceToHost, c0.stream));
                    off += n_slots * 64u;
                    DSV41_HIP_CHECK(hipMemcpyAsync(c0.h_radix_stage + off, c0.d_index_k_ue8m0[s] + sb * 4u,
                                                   n_slots * 4u, hipMemcpyDeviceToHost, c0.stream));
                    off += n_slots * 4u;
                }
                DSV41_HIP_CHECK(hipStreamSynchronize(c0.stream));
                std::memcpy(entry->data.data(), c0.h_radix_stage, kRadixBlockBytes);

                if (radix_map.size() >= kMaxDdr4RadixBlocks && !radix_fifo.empty()) {
                    radix_map.erase(radix_fifo.front());
                    radix_fifo.pop_front();
                }
                radix_map[key] = entry;
                radix_fifo.push_back(key);

                {
                    std::lock_guard<std::mutex> lk(nvme_mtx);
                    nvme_write_queue.push_back(entry);
                }
                nvme_cv.notify_one();
            }

            if (size_t(blk) < gpu_slot_radix_keys.size()) {
                gpu_slot_radix_keys[size_t(blk)] = key;
            }
            exported_radix_key = key;
            exported_radix_blocks = blk + 1;
        }
    }

    // Lookup longest matching chain of 128-token Radix blocks in DDR4 + Dual-NVMe (up to max_blocks)
    std::vector<std::shared_ptr<RadixBlockEntry>> lookup_radix_chain(const int32_t* prompt_toks, int max_blocks) {
        std::vector<std::shared_ptr<RadixBlockEntry>> hits;
        if (max_blocks <= 0) return hits;
        hits.reserve(max_blocks);
        RadixKey128 key{};
        for (int b = 0; b < max_blocks; ++b) {
            const int32_t* blk_toks = prompt_toks + b * kRadixBlockTokens;
            key = step_radix_hash(key, blk_toks);

            auto it = radix_map.find(key);
            if (it != radix_map.end() &&
                std::memcmp(it->second->tokens.data(), blk_toks, kRadixBlockTokens * sizeof(int32_t)) == 0) {
                hits.push_back(it->second);
                continue;
            }

            // On-demand Dual-NVMe GlobalPrefixCache check
            const int drive = int(key.lo & 1ULL);
            char fn[256];
            std::snprintf(fn, sizeof(fn), "%s/%016llx_%016llx.rblk",
                          nvme_radix_dirs[drive].c_str(),
                          static_cast<unsigned long long>(key.lo),
                          static_cast<unsigned long long>(key.hi));
            std::ifstream in(fn, std::ios::binary);
            if (!in) break;
            auto entry = std::make_shared<RadixBlockEntry>();
            entry->key = key;
            in.read(reinterpret_cast<char*>(entry->tokens.data()), sizeof(entry->tokens));
            if (!in || std::memcmp(entry->tokens.data(), blk_toks, sizeof(entry->tokens)) != 0) break;
            in.read(reinterpret_cast<char*>(entry->data.data()), entry->data.size());
            if (!in) break;

            if (radix_map.size() >= kMaxDdr4RadixBlocks && !radix_fifo.empty()) {
                radix_map.erase(radix_fifo.front());
                radix_fifo.pop_front();
            }
            radix_map[key] = entry;
            radix_fifo.push_back(key);
            hits.push_back(entry);
        }
        return hits;
    }

    // Dirty-Only + Double-Buffered Coalesced PCIe H2D upload of `blocks` (890 B/tok) to all 3 MI50 GPUs!
    // Skips HBM2 slots whose RadixKey128 already matches `gpu_slot_radix_keys[b]` (0 PCIe bytes on cache hit!).
    void restore_radix_blocks_gpu(const std::vector<std::shared_ptr<RadixBlockEntry>>& blocks) {
        const int num_blocks = static_cast<int>(blocks.size());
        if (num_blocks <= 0) return;

        auto is_clean = [&](int b) -> bool {
            const RadixKey128& k = blocks[b]->key;
            return size_t(b) < gpu_slot_radix_keys.size() &&
                   (k.lo != 0ULL || k.hi != 0ULL) &&
                   gpu_slot_radix_keys[size_t(b)] == k;
        };

        uint8_t* stage_bufs[2] = {ctx[0].h_radix_stage, ctx[0].h_radix_stage2 ? ctx[0].h_radix_stage2 : ctx[0].h_radix_stage};
        int ping = 0;
        bool any_uploaded = false;
        int cur = 0;

        while (cur < num_blocks) {
            while (cur < num_blocks && is_clean(cur)) {
                ++cur;
            }
            if (cur >= num_blocks) break;

            const int b0 = cur;
            int nb = 0;
            while (cur < num_blocks && nb < 64 && !is_clean(cur)) {
                ++cur;
                ++nb;
            }

            // Wait for any previous H2D DMAs reading `stage_bufs[ping]` across all 3 ranks before overwriting it
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipEventSynchronize(ctx[tp].ev_radix_stage[ping]));
            }
            uint8_t* stage = stage_bufs[ping];

            size_t stage_off = 0;
            size_t slice_offs[16]{};
            size_t slice_lens[16]{};

            int idx = 0;
            size_t entry_off = 0;
            for (int s = 0; s < 4; ++s) {
                const size_t n_slots = (s < 3) ? 64u : 128u;
                const size_t lens[4] = {n_slots * 256u, n_slots * 32u, n_slots * 64u, n_slots * 4u};
                for (int c = 0; c < 4; ++c) {
                    const size_t chunk_b = lens[c];
                    slice_offs[idx] = stage_off;
                    slice_lens[idx] = size_t(nb) * chunk_b;
                    #pragma omp parallel for num_threads(8) if(nb >= 16)
                    for (int k = 0; k < nb; ++k) {
                        std::memcpy(stage + stage_off + size_t(k) * chunk_b,
                                    blocks[b0 + k]->data.data() + entry_off,
                                    chunk_b);
                    }
                    stage_off += size_t(nb) * chunk_b;
                    entry_off += chunk_b;
                    ++idx;
                }
            }

            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                auto& c = ctx[tp];
                int q = 0;
                for (int s = 0; s < 4; ++s) {
                    const size_t n_slots = (s < 3) ? 64u : 128u;
                    const size_t sb = size_t(b0) * n_slots;
                    DSV41_HIP_CHECK(hipMemcpyAsync(c.d_comp_kv_fp4[s] + sb * 256u, stage + slice_offs[q],
                                                   slice_lens[q], hipMemcpyHostToDevice, c.stream));
                    ++q;
                    DSV41_HIP_CHECK(hipMemcpyAsync(c.d_comp_kv_e4m3[s] + sb * 32u, stage + slice_offs[q],
                                                   slice_lens[q], hipMemcpyHostToDevice, c.stream));
                    ++q;
                    DSV41_HIP_CHECK(hipMemcpyAsync(c.d_index_k_fp4[s] + sb * 64u, stage + slice_offs[q],
                                                   slice_lens[q], hipMemcpyHostToDevice, c.stream));
                    ++q;
                    DSV41_HIP_CHECK(hipMemcpyAsync(c.d_index_k_ue8m0[s] + sb * 4u, stage + slice_offs[q],
                                                   slice_lens[q], hipMemcpyHostToDevice, c.stream));
                    ++q;
                }
                DSV41_HIP_CHECK(hipEventRecord(c.ev_radix_stage[ping], c.stream));
            }

            for (int k = 0; k < nb; ++k) {
                if (size_t(b0 + k) < gpu_slot_radix_keys.size()) {
                    gpu_slot_radix_keys[size_t(b0 + k)] = blocks[b0 + k]->key;
                }
            }
            any_uploaded = true;
            if (ctx[0].h_radix_stage2 != nullptr) {
                ping ^= 1;
            }
        }

        if (any_uploaded) {
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                DSV41_HIP_CHECK(hipStreamSynchronize(ctx[tp].stream));
            }
        }
    }
};

Dsv41Runtime::Dsv41Runtime() : impl_(std::make_unique<Impl>()) {}
Dsv41Runtime::~Dsv41Runtime() = default;

void Dsv41Runtime::reset_session_cache() {
    std::lock_guard<std::mutex> lk(gen_mtx_);
    if (impl_ && ready_) {
        impl_->rewind_bank.clear();
        impl_->clear_gpu_kv_state();
    }
}

void Dsv41Runtime::set_sync_timers(bool enable) {
    std::lock_guard<std::mutex> lk(gen_mtx_);
    if (impl_) {
        impl_->sync_timers = enable;
    }
}

void Dsv41Runtime::clear_all_caches() {
    std::lock_guard<std::mutex> lk(gen_mtx_);
    if (impl_ && ready_) {
        impl_->rewind_bank.clear();
        impl_->radix_map.clear();
        impl_->radix_fifo.clear();
        std::fill(impl_->gpu_slot_radix_keys.begin(), impl_->gpu_slot_radix_keys.end(), RadixKey128{});
        impl_->clear_gpu_kv_state();
    }
}

void Dsv41Runtime::init(
    const std::string& mi50_packed_root,
    const std::string& tokenizer_json_path,
    const std::string& p150_shm_name,
    int max_seq_len,
    const std::string& rarebal_overlay_path,
    const std::string& capture_rarebal_dir)
{
    std::lock_guard<std::mutex> lk(gen_mtx_);
    impl_->max_seq_len = max_seq_len;
    if (!rarebal_overlay_path.empty()) {
        impl_->rarebal_overlay_mmap = std::make_unique<SafetensorsMmap>(rarebal_overlay_path);
        std::printf("[DSV41_RT] Loaded RAREBAL-LDLQ hot-expert overlay mmap: %s (%zu tensors)\n",
                    rarebal_overlay_path.c_str(), impl_->rarebal_overlay_mmap->entries().size());
        std::fflush(stdout);
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::printf("[DSV41_RT] Loading tokenizer & building ICU Engram compressed map (99,092)...\n");
    std::fflush(stdout);
    tokenizer_.load(tokenizer_json_path);

    std::vector<int32_t> comp_map;
    int32_t comp_vocab = 0;
    tokenizer_.build_engram_compressed_map(comp_map, comp_vocab);
    impl_->engram_cpu.init_hash_tables(comp_map, comp_vocab, max_seq_len);

    // Mmap both 94.56 GiB Engram shards concurrently across Dual-NVMe controllers (NVMe0 + NVMe1)
    for (int e_idx = 0; e_idx < 2; ++e_idx) {
        const int lid = (e_idx == 0) ? 1 : 14;
        const std::string meta_p = mi50_packed_root + "/engram_layer_" + std::to_string(lid) + ".table_path";
        std::ifstream fm(meta_p);
        if (!fm) {
            throw std::runtime_error("Missing Engram table_path metadata file: " + meta_p);
        }
        std::string raw_line;
        std::getline(fm, raw_line);
        const std::string shard_path = trim_line(raw_line);
        impl_->engram_cpu.open_table_shard(e_idx, shard_path);
        std::printf("[DSV41_RT] Mmapped Engram layer %d table: %s\n", lid, shard_path.c_str());
        std::fflush(stdout);
    }

    // Check 3x MI50 device count & peer access
    int dev_cnt = 0;
    DSV41_HIP_CHECK(hipGetDeviceCount(&dev_cnt));
    if (dev_cnt < kTpSize) {
        throw std::runtime_error("Expected at least 3 MI50 GPUs, found " + std::to_string(dev_cnt));
    }

    // Strictly ban PCIe P2P BAR peer access on gfx906 without xGMI; use 100% crash-proof coherent DDR4 HostRingAllReduce!
    impl_->p2p_enabled = false;
    DSV41_HIP_CHECK(hipSetDevice(0));
    void* ar_ring_ptr = nullptr;
    DSV41_HIP_CHECK(hipHostMalloc(&ar_ring_ptr, kHostArRingSlots * sizeof(HostRingAllReduceSlot),
                                  hipHostMallocMapped | hipHostMallocCoherent));
    std::memset(ar_ring_ptr, 0, kHostArRingSlots * sizeof(HostRingAllReduceSlot));
    impl_->h_ar_ring = static_cast<HostRingAllReduceSlot*>(ar_ring_ptr);
    impl_->host_ar_seq = 0u;

    void* markov_emb_ptr = nullptr;
    const size_t markov_emb_bytes = size_t(kVocabSize) * size_t(kMarkovRank) * sizeof(uint16_t);
    DSV41_HIP_CHECK(hipHostMalloc(&markov_emb_ptr, markov_emb_bytes,
                                  hipHostMallocMapped | hipHostMallocCoherent));
    std::memset(markov_emb_ptr, 0, markov_emb_bytes);
    impl_->h_markov_embed_mapped = static_cast<uint16_t*>(markov_emb_ptr);

    void* dspark_ctrl_ptr = nullptr;
    DSV41_HIP_CHECK(hipHostMalloc(&dspark_ctrl_ptr, sizeof(DsparkGpuSampleSlot),
                                  hipHostMallocMapped | hipHostMallocCoherent));
    std::memset(dspark_ctrl_ptr, 0, sizeof(DsparkGpuSampleSlot));
    impl_->h_dspark_ctrl = static_cast<DsparkGpuSampleSlot*>(dspark_ctrl_ptr);

    impl_->gpu_slot_radix_keys.assign(size_t(max_seq_len / kRadixBlockTokens + 16), RadixKey128{});
    impl_->host_tree_probs_f64.assign(size_t(kTree8NumNodes) * size_t(kVocabSize), 0.0);
    for (int b = 0; b < kDSparkBlockSize; ++b) {
        impl_->draft_q_probs[b].assign(kVocabSize, 0.0f);
    }

    std::printf("[DSV41_RT] 3x MI50 TP3 AllReduce: %zu-slot Coherent Pinned-DDR4 HostRingAllReduce + GPU-Resident DSpark ENABLED (%.1f MiB, 0 PCIe BAR P2P, 0 B extra HBM2)\n",
                kHostArRingSlots, double(kHostArRingSlots * sizeof(HostRingAllReduceSlot)) / (1024.0 * 1024.0));
    std::fflush(stdout);

    // Compute exact vocab offsets from tp0, tp1, tp2 embed.safetensors headers, then load all 3 GPUs in parallel!
    int v_rows[3]{};
    for (int tp = 0; tp < 3; ++tp) {
        SafetensorsMmap st(mi50_packed_root + "/tp" + std::to_string(tp) + "/embed.safetensors");
        v_rows[tp] = static_cast<int>(st.at("embed.weight").shape.at(0));
    }
    const int v_offs[3] = {0, v_rows[0], v_rows[0] + v_rows[1]};

    // Start background NVMe Radix writer + async Engram worker, and concurrently pre-fault 189 GiB Engram DDR4
    // across both NVMe drives while all 3 MI50 GPUs load HBM2 weights!
    impl_->start_background_workers();

    std::exception_ptr load_err[4]{};
    std::thread loaders[3];
    for (int tp = 0; tp < 3; ++tp) {
        loaders[tp] = std::thread([&, tp]() {
            try {
                impl_->init_tp_rank(tp, mi50_packed_root, v_offs[tp]);
            } catch (...) {
                load_err[tp] = std::current_exception();
            }
        });
    }
    std::thread engram_prefaulter([&]() {
        try {
            impl_->engram_cpu.prefault_all_tables();
        } catch (...) {
            load_err[3] = std::current_exception();
        }
    });

    for (int tp = 0; tp < 3; ++tp) loaders[tp].join();
    engram_prefaulter.join();
    for (int i = 0; i < 4; ++i) {
        if (load_err[i]) std::rethrow_exception(load_err[i]);
    }
    if (impl_->rarebal_overlay_mmap) {
        std::printf("[DSV41_RT] Patched %d MI50 + MTP hot experts (x3 TP3 shards, 0 B extra HBM2) from %s\n",
                    impl_->rarebal_mi50_patched_slots.load(std::memory_order_relaxed),
                    rarebal_overlay_path.c_str());
        std::fflush(stdout);
        impl_->rarebal_overlay_mmap.reset();
    }
    if (!capture_rarebal_dir.empty()) {
        const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                         ? std::string(std::getenv("DSV41_ROOT"))
                                         : std::string(".");
        impl_->rarebal_collector = std::make_unique<RarebalLiveCollector>();
        impl_->rarebal_collector->init(capture_rarebal_dir, tokenizer_,
                                       root_env + "/logs/p150_hot_experts.txt");
    }

    // Attach to p150_dsv41_worker over /dev/shm (waiting up to 120s if worker is still loading 40 layers!)
    std::printf("[DSV41_RT] Attaching to 2x Blackhole P150A worker at %s ...\n", p150_shm_name.c_str());
    std::fflush(stdout);
    bool attached = false;
    for (int attempt = 0; attempt < 240; ++attempt) {
        if (impl_->p150_shm.attach_client(p150_shm_name)) {
            const uint64_t mask = uint64_t(impl_->p150_shm.ptr->loaded_layers_mask_lo) |
                                  (uint64_t(impl_->p150_shm.ptr->loaded_layers_mask_hi) << 32);
            if (mask == ((1ULL << 40) - 1ULL)) {
                attached = true;
                break;
            }
            const uint32_t s = ++impl_->p150_seq_counter;
            impl_->p150_shm.ptr->op = DSV41_P150_OP_PING;
            impl_->p150_shm.ptr->req_seq.store(s, std::memory_order_release);
            while (impl_->p150_shm.ptr->resp_seq.load(std::memory_order_acquire) != s) {
                dsv41_cpu_relax();
            }
            const uint64_t mask2 = uint64_t(impl_->p150_shm.ptr->loaded_layers_mask_lo) |
                                   (uint64_t(impl_->p150_shm.ptr->loaded_layers_mask_hi) << 32);
            if (mask2 == ((1ULL << 40) - 1ULL)) {
                attached = true;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    if (!attached) {
        throw std::runtime_error("Failed to attach to ready 40-layer p150_dsv41_worker on " + p150_shm_name);
    }
    impl_->p150_seq_counter = impl_->p150_shm.ptr->req_seq.load(std::memory_order_relaxed);
    if (hipHostRegister(impl_->p150_shm.ptr, sizeof(*impl_->p150_shm.ptr), hipHostRegisterMapped) == hipSuccess) {
        impl_->p150_shm_pinned = true;
        for (int tp = 0; tp < 3; ++tp) {
            DSV41_HIP_CHECK(hipSetDevice(tp));
            void* dp = nullptr;
            DSV41_HIP_CHECK(hipHostGetDevicePointer(&dp, impl_->p150_shm.ptr, 0));
            impl_->d_p150_shm_y_out_bf16[tp] = static_cast<decltype(impl_->p150_shm.ptr)>(dp)->y_out_bf16;
        }
        DSV41_HIP_CHECK(hipSetDevice(0));
        std::printf("[DSV41_RT] Pinned & mapped /dev/shm P150A mailbox (%zu KiB) across 3x MI50 (zero-SDMA ring enabled)!\n",
                    sizeof(*impl_->p150_shm.ptr) / 1024u);
    } else {
        (void)hipGetLastError();
    }

    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[DSV41_RT] Heterogeneous 3x MI50 + 2x Blackhole P150A + Pre-Faulted 189 GiB Engram READY in %.2f s!\n", dt);
    std::fflush(stdout);
    ready_ = true;
}

std::vector<int32_t> Dsv41Runtime::generate(
    const std::vector<int32_t>& prompt_ids,
    const Dsv41GenParams& params,
    TokenStreamCallback on_token,
    Dsv41GenStats& out_stats,
    ProgressCallback on_progress)
{
    std::lock_guard<std::mutex> lk(gen_mtx_);
    out_stats = Dsv41GenStats{};
    if (!ready_ || prompt_ids.empty()) return {};
    impl_->reset_stage_timers();

    const int P = static_cast<int>(prompt_ids.size());
    if (P >= impl_->max_seq_len - 8) {
        throw std::runtime_error("Prompt length " + std::to_string(P) + " exceeds max_seq_len " + std::to_string(impl_->max_seq_len));
    }

    struct DpmWarmGuard {
        Dsv41Runtime::Impl* impl;
        explicit DpmWarmGuard(Dsv41Runtime::Impl* p) : impl(p) {
            for (int tp = 0; tp < 3; ++tp) {
                DSV41_HIP_CHECK(hipSetDevice(tp));
                auto& c = impl->ctx[tp];
                *c.h_dpm_flag = 1u;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                launch_keep_dpm_warm(c.d_dpm_flag, c.d_dpm_uc_scratch, 16383u, c.keep_warm_stream);
            }
        }
        ~DpmWarmGuard() {
            for (int tp = 0; tp < 3; ++tp) {
                *impl->ctx[tp].h_dpm_flag = 0u;
            }
            std::atomic_thread_fence(std::memory_order_seq_cst);
            for (int tp = 0; tp < 3; ++tp) {
                (void)hipSetDevice(tp);
                (void)hipStreamSynchronize(impl->ctx[tp].keep_warm_stream);
            }
        }
    } dpm_guard(impl_.get());

    out_stats.prompt_tokens = P;
    if (impl_->rarebal_collector) {
        impl_->run_rarebal_capture_on_prompt(prompt_ids, on_progress);
    }
    const int ced_split = (P > kWindowSize) ? (((P - kWindowSize) / 2) * 2) : 0;

    std::mt19937_64 rng;
    if (params.seed != 0) {
        rng.seed(params.seed);
    } else {
        std::random_device rd;
        rng.seed((uint64_t(rd()) << 32) ^ uint64_t(rd()) ^
                 uint64_t(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    }

    // 1. Multi-Tier Prefix Reuse Hierarchy:
    //    Tier 0: Exact in-VRAM LCP continuation (0.00 ms, even OR odd boundary!)
    //    Tier 1: RewindBank 32-snapshot ring (< 0.5 ms single-slab DMA)
    //    Tier 2: DDR4 RadixBlockCache + Dual-NVMe GlobalPrefixCache (890 B/tok) + 128-token SWA Bounded Replay!
    auto mono_ns = []() -> int64_t {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    const auto t_prefill_start = std::chrono::steady_clock::now();
    const int64_t t_reuse_begin_ns = mono_ns();
    size_t lcp = 0;
    const size_t max_match = std::min(impl_->cached_tokens.size(), prompt_ids.size() > 1 ? prompt_ids.size() - 1 : 0);
    while (lcp < max_match && impl_->cached_tokens[lcp] == prompt_ids[lcp]) {
        ++lcp;
    }

    int pos = 0;
    if (!impl_->cached_tokens.empty() && lcp == impl_->cached_tokens.size()) {
        // Tier 0: Exact in-VRAM continuation (including odd/even multi-turn tool loop!)
        pos = static_cast<int>(lcp);
    } else {
        // Find best matching RewindBank snapshot strictly < P
        const RewindSnapshot* best_snap = nullptr;
        for (const auto& s : impl_->rewind_bank) {
            if (s.pos < P && s.pos > (best_snap ? best_snap->pos : 0)) {
                if (std::memcmp(s.tokens.data(), prompt_ids.data(), size_t(s.pos) * sizeof(int32_t)) == 0) {
                    best_snap = &s;
                }
            }
        }

        // Lookup 128-token RadixBlockCache chain up to (P - 1) / 128
        const int max_rblks = (P - 1) / kRadixBlockTokens;
        auto radix_hits = impl_->lookup_radix_chain(prompt_ids.data(), max_rblks);
        const int R = static_cast<int>(radix_hits.size()) * kRadixBlockTokens;
        const int snap_pos = best_snap ? best_snap->pos : 0;

        if (best_snap && snap_pos >= R - kWindowSize) {
            // Tier 1: RewindBank avoids SWA Bounded Replay! Also refresh Global KV if current GPU cache diverged < snap_pos
            const bool need_global_dma = (static_cast<int>(lcp) < snap_pos);
            std::vector<std::shared_ptr<RadixBlockEntry>> snap_rblks;
            const int needed_blks = snap_pos / kRadixBlockTokens;
            if (need_global_dma && static_cast<int>(radix_hits.size()) >= needed_blks) {
                snap_rblks.assign(radix_hits.begin(), radix_hits.begin() + needed_blks);
                impl_->restore_rewind_snapshot_gpu(*best_snap, true, snap_rblks);
                pos = snap_pos;
            } else if (!need_global_dma) {
                impl_->restore_rewind_snapshot_gpu(*best_snap, false, snap_rblks);
                pos = snap_pos;
            }
        }

        if (pos == 0 && R > 0) {
            // Tier 2: Restore Global KV for [0..R) via coalesced PCIe DMA + 128-token SWA Bounded Replay on [R-128..R)!
            impl_->clear_gpu_kv_state();
            impl_->restore_radix_blocks_gpu(radix_hits);
            impl_->engram_cpu.compute_hashes(prompt_ids.data(), R - kWindowSize, 0, nullptr, impl_->engram_hashes_buf);

            const int replay_start = R - kWindowSize;
            const bool replay_enc_only = (R <= ced_split);
            const bool replay_seed_mtp = (!replay_enc_only);
            if (!replay_enc_only) {
                impl_->win_valid_from_20_39 = replay_start;
            }

            impl_->forward_backbone_chunk(
                prompt_ids.data() + replay_start,
                replay_start,
                kWindowSize,
                replay_enc_only,
                0,
                true,             // commit_window_immediately
                replay_seed_mtp,  // capture_dspark_hidden
                replay_start,     // chunk_swa_floor = R - 128!
                true              // skip_global_kv_write = true!
            );
            if (replay_enc_only) {
                impl_->win_valid_from_20_39 = R;
            } else {
                impl_->update_mtp_window_from_target_hidden(replay_start, kWindowSize, 0);
            }

            impl_->cached_tokens.assign(prompt_ids.begin(), prompt_ids.begin() + R);
            impl_->exported_radix_blocks = static_cast<int>(radix_hits.size());
            impl_->exported_radix_key = radix_hits.back()->key;
            pos = R;
            out_stats.radix_hit_tokens = R;
            out_stats.swa_replay_tokens = kWindowSize;
        } else if (pos == 0) {
            impl_->clear_gpu_kv_state();
        }
    }

    const auto t_reuse_end = std::chrono::steady_clock::now();
    const int64_t t_reuse_end_ns = mono_ns();
    out_stats.prefix_reuse_ms = std::chrono::duration<double, std::milli>(t_reuse_end - t_prefill_start).count();
    out_stats.cached_prefix_tokens = pos;
    out_stats.prefilled_tokens = P - pos;
    if (on_progress) {
        Dsv41ProgressUpdate upd{};
        upd.phase = Dsv41ProgressPhase::PrefixDone;
        upd.prompt_tokens = P;
        upd.cached_tokens = pos;
        upd.fresh_tokens_done = 0;
        upd.chunk_tokens = 0;
        upd.chunk_ms = out_stats.prefix_reuse_ms;
        upd.begin_ns = t_reuse_begin_ns;
        upd.end_ns = t_reuse_end_ns;
        if (!on_progress(upd)) {
            return {};
        }
    }

    // 2. Strict 20-Layer CED Early-Exit Prefill over [pos .. P - 1]:
    //    - Phase A [pos .. ced_split): 20-layer Encoder-Only + Layer 20 Compressor/Indexer.k (kMaxChunkM = 512)!
    //    - Phase B [max(pos, ced_split) .. P): Final <= 129 tokens run full 40-layer Decoder + seed DSpark MTP window!
    while (pos < ced_split) {
        const int rem_enc = ced_split - pos;
        int chunk_m = std::min(rem_enc, kMaxChunkM);
        // Align odd `pos` back to an even r=2 boundary on the first encoder chunk!
        if ((pos & 1) != 0) {
            if ((chunk_m & 1) == 0) --chunk_m;
        } else if ((chunk_m & 1) != 0) {
            --chunk_m;
        }
        const int64_t tc0_ns = mono_ns();
        impl_->forward_backbone_chunk(
            prompt_ids.data() + pos,
            pos,
            chunk_m,
            true,   // encoder_only = true!
            0,
            true,
            false
        );
        impl_->cached_tokens.insert(impl_->cached_tokens.end(),
                                    prompt_ids.begin() + pos,
                                    prompt_ids.begin() + pos + chunk_m);
        pos += chunk_m;
        impl_->win_valid_from_20_39 = pos;
        const int64_t tc1_ns = mono_ns();
        if (on_progress) {
            Dsv41ProgressUpdate upd{};
            upd.phase = Dsv41ProgressPhase::PrefillChunk;
            upd.prompt_tokens = P;
            upd.cached_tokens = out_stats.cached_prefix_tokens;
            upd.fresh_tokens_done = pos - out_stats.cached_prefix_tokens;
            upd.chunk_tokens = chunk_m;
            upd.chunk_ms = double(tc1_ns - tc0_ns) / 1e6;
            upd.begin_ns = tc0_ns;
            upd.end_ns = tc1_ns;
            if (!on_progress(upd)) {
                impl_->export_new_radix_blocks();
                return {};
            }
        }
    }

    while (pos < P) {
        const int rem_dec = P - pos;
        int chunk_m = std::min(rem_dec, kMaxDecChunkM);
        if (pos + chunk_m < P && ((pos + chunk_m) & 1) != 0 && chunk_m > 1) {
            --chunk_m;
        }
        const bool is_last_chunk = (pos + chunk_m == P);
        const bool need_mtp_seed = ((pos + chunk_m) > (P - kWindowSize));
        const int64_t tc0_ns = mono_ns();

        impl_->forward_backbone_chunk(
            prompt_ids.data() + pos,
            pos,
            chunk_m,
            false,  // full 40-layer Decoder!
            is_last_chunk ? 1 : 0,
            true,
            need_mtp_seed
        );

        if (need_mtp_seed) {
            const int seed_global_start = std::max(pos, P - kWindowSize);
            const int cap_off = std::max(0, chunk_m - kMaxDecM);
            const int row_off = (seed_global_start - pos) - cap_off;
            const int seed_cnt = (pos + chunk_m) - seed_global_start;
            impl_->update_mtp_window_from_target_hidden(seed_global_start, seed_cnt, row_off);
        }

        impl_->cached_tokens.insert(impl_->cached_tokens.end(),
                                    prompt_ids.begin() + pos,
                                    prompt_ids.begin() + pos + chunk_m);
        pos += chunk_m;
        const int64_t tc1_ns = mono_ns();
        if (on_progress) {
            Dsv41ProgressUpdate upd{};
            upd.phase = Dsv41ProgressPhase::PrefillChunk;
            upd.prompt_tokens = P;
            upd.cached_tokens = out_stats.cached_prefix_tokens;
            upd.fresh_tokens_done = pos - out_stats.cached_prefix_tokens;
            upd.chunk_tokens = chunk_m;
            upd.chunk_ms = double(tc1_ns - tc0_ns) / 1e6;
            upd.begin_ns = tc0_ns;
            upd.end_ns = tc1_ns;
            if (!on_progress(upd)) {
                impl_->export_new_radix_blocks();
                impl_->save_rewind_snapshot();
                return {};
            }
        }
    }

    // Export newly completed 128-token Radix blocks + save post-prefill RewindBank snapshot!
    impl_->export_new_radix_blocks();
    impl_->save_rewind_snapshot();

    // Sample first generated token `cur_anchor` (for position P) from prefill logits!
    std::vector<int32_t> generated;
    generated.reserve(std::min(params.max_new_tokens, 4096));

    float* first_logits = impl_->host_full_logits.data();
    Impl::apply_rep_penalty_inplace(first_logits, impl_->cached_tokens, params.repetition_penalty);
    std::vector<float> target_probs;
    int32_t cur_anchor = Impl::build_probs_and_sample(first_logits, params.temperature, params.top_p, target_probs, rng);

    const auto t_prefill_end = std::chrono::steady_clock::now();
    out_stats.fresh_prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_reuse_end).count();
    out_stats.prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

    if (params.max_new_tokens <= 0) {
        return generated;
    }

    const auto t_decode_start = std::chrono::steady_clock::now();
    impl_->accum_p150_dev_ms = 0.0;
    Dsv41ThinkLoopGuard think_guard;
    bool pending_think_end_check = false;
    std::string pending_think_txt;
    size_t pending_think_te = 0;
    if (params.enable_think_loop_guard && !prompt_ids.empty() &&
        prompt_ids.back() == Dsv41SpecialTokens::kThinkOpen) {
        think_guard.enter_think();
    }
    auto observe_tok_with_guard = [&](int32_t tok) -> ThinkGuardDecision {
        if (!params.enable_think_loop_guard) return ThinkGuardDecision{};
        if (pending_think_end_check) {
            pending_think_end_check = false;
            pending_think_txt += tokenizer_.decode_single(tok, false);
            if (classify_think_end(pending_think_txt, pending_think_te, false) == ThinkEndClass::InlineMention) {
                think_guard.resume_think();
                (void)think_guard.observe_think_token(Dsv41SpecialTokens::kThinkEnd);
            }
            pending_think_txt.clear();
        }
        if (tok == Dsv41SpecialTokens::kThinkOpen && !think_guard.in_think()) {
            think_guard.enter_think();
            return ThinkGuardDecision{};
        }
        if (tok == Dsv41SpecialTokens::kThinkEnd && think_guard.in_think()) {
            std::string txt = tokenizer_.decode(generated, false);
            const size_t tag_len = 8;  // strlen("</think>")
            if (txt.size() >= tag_len && txt.compare(txt.size() - tag_len, tag_len, "</think>") == 0) {
                const size_t te = txt.size() - tag_len;
                const ThinkEndClass cls = classify_think_end(txt, te, false);
                if (cls == ThinkEndClass::InlineMention) {
                    return think_guard.observe_think_token(tok);
                }
                if (cls == ThinkEndClass::NeedMoreLookahead) {
                    pending_think_end_check = true;
                    pending_think_txt = std::move(txt);
                    pending_think_te = te;
                }
            }
            think_guard.exit_think();
            return ThinkGuardDecision{};
        }
        return think_guard.observe_think_token(tok);
    };

    generated.push_back(cur_anchor);
    out_stats.completion_tokens = 1;
    bool keep_going = true;
    if (cur_anchor == Dsv41SpecialTokens::kEos) {
        keep_going = false;
    } else if (on_token && !on_token(cur_anchor)) {
        keep_going = false;
    } else {
        (void)observe_tok_with_guard(cur_anchor);
    }

    auto execute_force_close_think = [&](const ThinkGuardDecision& gd) -> bool {
        const int G = static_cast<int>(generated.size());
        const int rew = std::min<int>(static_cast<int>(gd.rewind_tokens), std::max(0, G - 1));
        const int keep_gen = G - rew;
        generated.resize(static_cast<size_t>(keep_gen));
        const int target_pos = P + keep_gen;

        std::vector<int32_t> close_ids = tokenizer_.encode("\n", false);
        close_ids.push_back(Dsv41SpecialTokens::kThinkEnd);
        const int M_close = static_cast<int>(close_ids.size());
        if (target_pos + M_close + 8 >= impl_->max_seq_len) {
            out_stats.completion_tokens = static_cast<int>(generated.size());
            return false;
        }

        // Rewind to even boundary <= target_pos - kWindowSize so 128-token SWA ring + r=2 state + Global KV
        // and MTP windows are restored 100% bit-exactly in a single forward_backbone_chunk!
        const int replay_start = std::max(0, (target_pos - kWindowSize) & ~1);
        const int total_m = (target_pos - replay_start) + M_close;
        std::vector<int32_t> replay_buf;
        replay_buf.reserve(static_cast<size_t>(total_m));
        for (int p_idx = replay_start; p_idx < target_pos; ++p_idx) {
            const int32_t tid = (p_idx < P) ? prompt_ids[static_cast<size_t>(p_idx)]
                                            : generated[static_cast<size_t>(p_idx - P)];
            replay_buf.push_back(tid);
        }
        for (int32_t cid : close_ids) {
            replay_buf.push_back(cid);
        }

        for (int tp = 0; tp < 3; ++tp) {
            impl_->reset_r2_compressor_state_gpu(tp);
        }
        impl_->win_valid_from_20_39 = std::max(impl_->win_valid_from_20_39, replay_start);

        const auto trw0 = std::chrono::steady_clock::now();
        impl_->forward_backbone_chunk(
            replay_buf.data(),
            replay_start,
            total_m,
            false,         // full 40-layer Decoder
            1,             // compute_logits_rows = 1 (post-</think> logits)
            true,          // commit_window_immediately = true
            true,          // capture_dspark_hidden = true
            replay_start,  // chunk_swa_floor = replay_start (zero stale SWA reads)
            false          // refresh Global KV & r=2 state cleanly from even replay_start
        );
        const int post_close_pos = target_pos + M_close;
        const int mtp_seed_start = std::max(replay_start, post_close_pos - kWindowSize);
        const int cap_off = std::max(0, total_m - kMaxDecM);
        const int mtp_row_off = (mtp_seed_start - replay_start) - cap_off;
        const int mtp_seed_cnt = post_close_pos - mtp_seed_start;
        impl_->update_mtp_window_from_target_hidden(mtp_seed_start, mtp_seed_cnt, mtp_row_off);
        const auto trw1 = std::chrono::steady_clock::now();
        out_stats.dec_backbone_ms += std::chrono::duration<double, std::milli>(trw1 - trw0).count();

        impl_->cached_tokens.clear();
        impl_->cached_tokens.reserve(static_cast<size_t>(post_close_pos + 1024));
        impl_->cached_tokens.insert(impl_->cached_tokens.end(), prompt_ids.begin(), prompt_ids.end());
        impl_->cached_tokens.insert(impl_->cached_tokens.end(), generated.begin(), generated.end());
        for (int32_t cid : close_ids) {
            impl_->cached_tokens.push_back(cid);
            generated.push_back(cid);
        }

        const int clean_full_blks = replay_start / kRadixBlockTokens;
        if (impl_->exported_radix_blocks > clean_full_blks) {
            impl_->exported_radix_blocks = 0;
            impl_->exported_radix_key = RadixKey128{};
            for (int b = 0; b < clean_full_blks; ++b) {
                impl_->exported_radix_key = step_radix_hash(
                    impl_->exported_radix_key, impl_->cached_tokens.data() + b * kRadixBlockTokens);
                impl_->exported_radix_blocks = b + 1;
            }
        }

        out_stats.think_guard_triggered = true;
        out_stats.think_guard_period = gd.detected_period;
        out_stats.think_guard_rewound = static_cast<uint32_t>(rew);
        std::printf("[DSV41_THINK_GUARD] Force-closed <think> loop (period=%u, seen=%u, rewound=%d -> pos=%d) — transitioning to post-</think> decode!\n",
                    gd.detected_period, gd.think_tokens_seen, rew, post_close_pos);
        std::fflush(stdout);

        for (int32_t cid : close_ids) {
            if (on_token && !on_token(cid)) {
                out_stats.completion_tokens = static_cast<int>(generated.size());
                return false;
            }
        }
        if (static_cast<int>(generated.size()) >= params.max_new_tokens) {
            out_stats.completion_tokens = static_cast<int>(generated.size());
            return false;
        }

        const auto tsmp0 = std::chrono::steady_clock::now();
        float* post_logits = impl_->host_full_logits.data();
        Impl::apply_rep_penalty_inplace(post_logits, impl_->cached_tokens, params.repetition_penalty);
        post_logits[Dsv41SpecialTokens::kThinkOpen] = -1e30f;
        post_logits[Dsv41SpecialTokens::kThinkEnd]  = -1e30f;
        cur_anchor = Impl::build_probs_and_sample(post_logits, params.temperature, params.top_p, target_probs, rng);
        out_stats.dec_sample_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tsmp0).count();

        generated.push_back(cur_anchor);
        out_stats.completion_tokens = static_cast<int>(generated.size());
        if (cur_anchor == Dsv41SpecialTokens::kEos || (on_token && !on_token(cur_anchor))) {
            return false;
        }
        return true;
    };

    // 3. Decode Loop: 3-Stage DSpark Speculative Decoding (M <= 6 Chain / M = 8 Causal Tree) or Standard M=1 Fallback
    std::uniform_real_distribution<float> uni01(0.0f, 1.0f);
    std::uniform_real_distribution<double> uni01_f64(0.0, 1.0);
    std::vector<float> residual_probs(kVocabSize);

    while (keep_going && static_cast<int>(generated.size()) < params.max_new_tokens) {
        const int cur_pos = static_cast<int>(impl_->cached_tokens.size());
        if (cur_pos + 8 >= impl_->max_seq_len) break;

        const int rem_budget = params.max_new_tokens - static_cast<int>(generated.size());
        const bool run_spec = params.enable_dspark && impl_->mtp_window_seeded && (rem_budget >= 2);

        if (!run_spec) {
            // Single-token M=1 step
            const auto tb0 = std::chrono::steady_clock::now();
            impl_->forward_backbone_chunk(&cur_anchor, cur_pos, 1, false, 1, true, true);
            impl_->update_mtp_window_from_target_hidden(cur_pos, 1, 0);
            const auto tb1 = std::chrono::steady_clock::now();
            out_stats.dec_backbone_ms += std::chrono::duration<double, std::milli>(tb1 - tb0).count();
            impl_->cached_tokens.push_back(cur_anchor);

            float* l0 = impl_->host_full_logits.data();
            Impl::apply_rep_penalty_inplace(l0, impl_->cached_tokens, params.repetition_penalty);
            const int32_t next_tok = Impl::build_probs_and_sample(l0, params.temperature, params.top_p, target_probs, rng);
            out_stats.dec_sample_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tb1).count();

            generated.push_back(next_tok);
            out_stats.completion_tokens = static_cast<int>(generated.size());
            cur_anchor = next_tok;
            if (next_tok == Dsv41SpecialTokens::kEos || (on_token && !on_token(next_tok))) {
                break;
            }
            const auto gd = observe_tok_with_guard(next_tok);
            if (gd.action == ThinkGuardAction::kForceCloseThink) {
                if (!execute_force_close_think(gd)) {
                    break;
                }
            }
            continue;
        }

        // A. Draft B (1..5) linear chain OR high-entropy M=8 width-[3,2,1,1] causal tree using 3-stage DSpark on 3x MI50!
        bool used_tree8 = false;
        std::array<int32_t, kTree8NumNodes> tree_verify_tokens{};
        const auto td0 = std::chrono::steady_clock::now();
        std::array<int32_t, kDSparkBlockSize> drafts{};
        int B = impl_->forward_dspark_draft(
            cur_anchor, cur_pos, params.temperature, params.top_p, params.repetition_penalty, rng, drafts,
            /*allow_tree8=*/true, rem_budget, &used_tree8, &tree_verify_tokens);
        const auto td1 = std::chrono::steady_clock::now();
        out_stats.dec_dspark_ms += std::chrono::duration<double, std::milli>(td1 - td0).count();

        if (used_tree8) {
            out_stats.dspark_steps += 1;
            out_stats.dspark_tree_steps += 1;
            out_stats.dspark_drafted += 3;  // max root-to-leaf sequential draft depth = 3 (L in [1, 4])

            // B_tree. Verify all M=8 causal-tree nodes in ONE backbone pass across 3x MI50 + 2x P150A!
            const auto tb0 = std::chrono::steady_clock::now();
            impl_->forward_backbone_chunk(
                tree_verify_tokens.data(),
                cur_pos,
                kTree8NumNodes,
                false,
                kTree8NumNodes,
                false,  // commit_window_immediately = false -> deferred alias-free tree commit!
                true,   // capture_dspark_hidden = true -> rows [0..7] captured in d_target_mean_bf16!
                0,
                false,
                kTree8AncMask64,
                kTree8DepthNibbles,
                kTree8ParentNibbles,
                kTree8Parents.data(),
                kTree8Depths.data()
            );
            const auto tb1 = std::chrono::steady_clock::now();
            out_stats.dec_backbone_ms += std::chrono::duration<double, std::milli>(tb1 - tb0).count();

            // C_tree. Lossless Kolmogorov Dirac causal-tree verification (TV <= 2e-16, strictly ancestor-only rep penalty!)
            Dsv41SpecTree8 topo{};
            (void)build_spec_tree_topology(
                uint32_t(kTree8NumNodes),
                uint32_t(cur_pos),
                kTree8Parents.data(),
                tree_verify_tokens.data(),
                &topo
            );

            for (int u = 0; u < kTree8NumNodes; ++u) {
                int32_t anc_toks[4]{};
                int p = u;
                while (p >= 0) {
                    anc_toks[kTree8Depths[size_t(p)]] = tree_verify_tokens[size_t(p)];
                    p = kTree8Parents[size_t(p)];
                }
                const size_t anc_len = size_t(kTree8Depths[size_t(u)] + 1u);
                float* u_logits = impl_->host_full_logits.data() + size_t(u) * size_t(kVocabSize);
                Impl::apply_rep_penalty_span(u_logits, impl_->cached_tokens, anc_toks, anc_len, params.repetition_penalty);
                Impl::build_probs_f64_fast(
                    u_logits,
                    params.temperature,
                    params.top_p,
                    impl_->host_tree_probs_f64.data() + size_t(u) * size_t(kVocabSize)
                );
            }

            std::array<double, kTree8NumNodes> u_draws{};
            for (int d = 0; d < 4; ++d) {
                u_draws[size_t(d)] = (params.temperature > 1e-5f) ? uni01_f64(rng) : 0.0;
            }
            const Dsv41TreeVerifyResult vr = verify_deterministic_tree_u01(
                topo,
                impl_->host_tree_probs_f64.data(),
                size_t(kVocabSize),
                u_draws.data()
            );

            int k_acc = 0;
            bool hit_stop = false;
            bool force_closed_in_burst = false;
            ThinkGuardDecision burst_guard_dec{};
            const int n_draft_acc = std::max(0, int(vr.num_emitted) - 1);

            for (int d = 0; d < n_draft_acc; ++d) {
                const int32_t d_tok = vr.emitted_tokens[size_t(d)];
                ++k_acc;
                out_stats.dspark_accepted += 1;
                out_stats.dspark_tree_accepted += 1;
                generated.push_back(d_tok);
                if (d_tok == Dsv41SpecialTokens::kEos || (on_token && !on_token(d_tok))) {
                    hit_stop = true;
                    break;
                }
                const auto gd = observe_tok_with_guard(d_tok);
                if (gd.action == ThinkGuardAction::kForceCloseThink) {
                    force_closed_in_burst = true;
                    burst_guard_dec = gd;
                    break;
                }
            }

            if (force_closed_in_burst) {
                if (!execute_force_close_think(burst_guard_dec)) {
                    break;
                }
                continue;
            }

            // D_tree. Commit exact `accepted_A = 1 + k_acc` verified root-to-leaf nodes to Backbone + MTP + Engram caches!
            const auto tc0 = std::chrono::steady_clock::now();
            out_stats.dec_sample_ms += std::chrono::duration<double, std::milli>(tc0 - tb1).count();
            const int accepted_A = 1 + k_acc;
            int32_t accepted_prefix_toks[4]{};
            accepted_prefix_toks[0] = cur_anchor;
            for (int d = 0; d < k_acc; ++d) {
                accepted_prefix_toks[1 + d] = vr.emitted_tokens[size_t(d)];
            }
            impl_->commit_verified_tree_backbone_state(
                cur_pos, accepted_A, vr.accepted_nodes.data(), accepted_prefix_toks);
            impl_->update_mtp_window_from_target_hidden(cur_pos, accepted_A, 8);
            const double commit_dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc0).count();
            impl_->sub_commit_ms += commit_dt;
            out_stats.dec_backbone_ms += commit_dt;
            for (int s = 0; s < accepted_A; ++s) {
                impl_->cached_tokens.push_back(accepted_prefix_toks[s]);
            }

            if (hit_stop) {
                out_stats.completion_tokens = static_cast<int>(generated.size());
                break;
            }

            const int32_t bonus_or_resampled = vr.emitted_tokens[size_t(vr.num_emitted - 1u)];
            if (bonus_or_resampled >= 0 && static_cast<int>(generated.size()) < params.max_new_tokens) {
                generated.push_back(bonus_or_resampled);
                out_stats.completion_tokens = static_cast<int>(generated.size());
                cur_anchor = bonus_or_resampled;
                if (bonus_or_resampled == Dsv41SpecialTokens::kEos || (on_token && !on_token(bonus_or_resampled))) {
                    break;
                }
                const auto gd = observe_tok_with_guard(bonus_or_resampled);
                if (gd.action == ThinkGuardAction::kForceCloseThink) {
                    if (!execute_force_close_think(gd)) {
                        break;
                    }
                }
            } else {
                out_stats.completion_tokens = static_cast<int>(generated.size());
                break;
            }
            continue;
        }

        B = std::min(B, std::max(1, rem_budget - 1));

        std::array<int32_t, 1 + kDSparkBlockSize> verify_tokens{};
        verify_tokens[0] = cur_anchor;
        for (int i = 0; i < B; ++i) verify_tokens[1 + i] = drafts[i];
        const int M_ver = 1 + B;

        out_stats.dspark_steps += 1;
        out_stats.dspark_drafted += uint64_t(B);

        // B. Verify all M_ver (2..6) tokens in ONE backbone pass across 3x MI50 + 2x P150A!
        const auto tb0 = std::chrono::steady_clock::now();
        impl_->forward_backbone_chunk(
            verify_tokens.data(),
            cur_pos,
            M_ver,
            false,
            M_ver,
            false,  // commit_window_immediately = false -> clean commit of accepted_A only!
            true    // capture_dspark_hidden = true -> updates MTP window for 0..accepted_A-1!
        );
        const auto tb1 = std::chrono::steady_clock::now();
        out_stats.dec_backbone_ms += std::chrono::duration<double, std::milli>(tb1 - tb0).count();

        // C. Lossless Speculative Rejection Sampling (exact for both greedy temp=0 and stochastic temp+top_p!)
        int k_acc = 0;
        int32_t bonus_or_resampled = -1;
        bool hit_stop = false;
        bool force_closed_in_burst = false;
        ThinkGuardDecision burst_guard_dec{};

        for (int k = 0; k < B; ++k) {
            float* t_logits = impl_->host_full_logits.data() + size_t(k) * kVocabSize;
            Impl::apply_rep_penalty_span(t_logits, impl_->cached_tokens, verify_tokens.data(), size_t(1 + k_acc), params.repetition_penalty);
            const int32_t d_tok = drafts[k];

            if (params.temperature <= 1e-5f) {
                const int32_t greedy_t = Impl::build_probs_and_sample(t_logits, 0.0f, 1.0f, target_probs, rng);
                if (d_tok == greedy_t) {
                    ++k_acc;
                    out_stats.dspark_accepted += 1;
                    generated.push_back(d_tok);
                    if (d_tok == Dsv41SpecialTokens::kEos || (on_token && !on_token(d_tok))) {
                        hit_stop = true;
                        break;
                    }
                    const auto gd = observe_tok_with_guard(d_tok);
                    if (gd.action == ThinkGuardAction::kForceCloseThink) {
                        force_closed_in_burst = true;
                        burst_guard_dec = gd;
                        break;
                    }
                } else {
                    bonus_or_resampled = greedy_t;
                    break;
                }
            } else {
                (void)Impl::build_probs_and_sample(t_logits, params.temperature, params.top_p, target_probs, rng);
                const float p_x = target_probs[d_tok];
                const float q_x = impl_->draft_q_selected[k];
                const float accept_prob = (q_x <= 0.0f) ? ((p_x > 0.0f) ? 1.0f : 0.0f) : std::min(1.0f, p_x / q_x);

                if (uni01(rng) <= accept_prob) {
                    ++k_acc;
                    out_stats.dspark_accepted += 1;
                    generated.push_back(d_tok);
                    if (d_tok == Dsv41SpecialTokens::kEos || (on_token && !on_token(d_tok))) {
                        hit_stop = true;
                        break;
                    }
                    const auto gd = observe_tok_with_guard(d_tok);
                    if (gd.action == ThinkGuardAction::kForceCloseThink) {
                        force_closed_in_burst = true;
                        burst_guard_dec = gd;
                        break;
                    }
                } else {
                    auto& q_probs = impl_->draft_q_probs[k];
                    if (!impl_->draft_q_materialized[k]) {
                        q_probs.assign(kVocabSize, 0.0f);
                        const uint32_t kc = impl_->h_dspark_ctrl->kept_cnt[k];
                        for (uint32_t j = 0; j < kc; ++j) {
                            const int32_t vid = impl_->h_dspark_ctrl->kept_id[k][j];
                            if (vid >= 0 && vid < kVocabSize) {
                                q_probs[size_t(vid)] = impl_->h_dspark_ctrl->kept_q[k][j];
                            }
                        }
                        impl_->draft_q_materialized[k] = true;
                    }
                    // Sample from normalized residual max(0, p(v) - q(v))!
                    double res_sum = 0.0;
                    for (int v = 0; v < kVocabSize; ++v) {
                        const float d = std::max(0.0f, target_probs[v] - q_probs[v]);
                        residual_probs[v] = d;
                        res_sum += double(d);
                    }
                    if (res_sum <= 1e-12) {
                        // Identical distributions up to float rounding -> sample directly from target_probs
                        float r = uni01(rng);
                        float acc = 0.0f;
                        bonus_or_resampled = d_tok;
                        for (int v = 0; v < kVocabSize; ++v) {
                            acc += target_probs[v];
                            if (r <= acc && target_probs[v] > 0.0f) {
                                bonus_or_resampled = v;
                                break;
                            }
                        }
                    } else {
                        const float inv_res = float(1.0 / res_sum);
                        float r = uni01(rng);
                        float acc = 0.0f;
                        int32_t last_nz = 0;
                        bonus_or_resampled = -1;
                        for (int v = 0; v < kVocabSize; ++v) {
                            const float pv = residual_probs[v] * inv_res;
                            if (pv > 0.0f) {
                                last_nz = v;
                                acc += pv;
                                if (r <= acc) {
                                    bonus_or_resampled = v;
                                    break;
                                }
                            }
                        }
                        if (bonus_or_resampled < 0) bonus_or_resampled = last_nz;
                    }
                    break;
                }
            }
        }

        if (force_closed_in_burst) {
            if (!execute_force_close_think(burst_guard_dec)) {
                break;
            }
            continue;
        }

        // If all B draft tokens were accepted and we haven't hit EOS/stop, sample bonus token from row B!
        if (!hit_stop && k_acc == B && static_cast<int>(generated.size()) < params.max_new_tokens) {
            float* bonus_logits = impl_->host_full_logits.data() + size_t(B) * kVocabSize;
            Impl::apply_rep_penalty_span(bonus_logits, impl_->cached_tokens, verify_tokens.data(), size_t(1 + B), params.repetition_penalty);
            bonus_or_resampled = Impl::build_probs_and_sample(
                bonus_logits, params.temperature, params.top_p, target_probs, rng);
        }

        // D. Commit exact `accepted_A = 1 + k_acc` verified tokens to Backbone + MTP caches!
        const auto tc0 = std::chrono::steady_clock::now();
        out_stats.dec_sample_ms += std::chrono::duration<double, std::milli>(tc0 - tb1).count();
        const int accepted_A = 1 + k_acc;
        impl_->commit_verified_backbone_state(cur_pos, M_ver, accepted_A);
        impl_->update_mtp_window_from_target_hidden(cur_pos, accepted_A, 0);
        const double commit_dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc0).count();
        impl_->sub_commit_ms += commit_dt;
        out_stats.dec_backbone_ms += commit_dt;
        for (int s = 0; s < accepted_A; ++s) {
            impl_->cached_tokens.push_back(verify_tokens[s]);
        }

        if (hit_stop) {
            out_stats.completion_tokens = static_cast<int>(generated.size());
            break;
        }

        if (bonus_or_resampled >= 0 && static_cast<int>(generated.size()) < params.max_new_tokens) {
            generated.push_back(bonus_or_resampled);
            out_stats.completion_tokens = static_cast<int>(generated.size());
            cur_anchor = bonus_or_resampled;
            if (bonus_or_resampled == Dsv41SpecialTokens::kEos || (on_token && !on_token(bonus_or_resampled))) {
                break;
            }
            const auto gd = observe_tok_with_guard(bonus_or_resampled);
            if (gd.action == ThinkGuardAction::kForceCloseThink) {
                if (!execute_force_close_think(gd)) {
                    break;
                }
            }
        } else {
            out_stats.completion_tokens = static_cast<int>(generated.size());
            break;
        }
    }

    // Export newly completed 128-token Radix blocks + save post-decode RewindBank snapshot!
    impl_->export_new_radix_blocks();
    impl_->save_rewind_snapshot();

    // Strip trailing <｜end▁of▁sentence｜> from returned token IDs (while keeping completion_tokens accurate)
    if (!generated.empty() && generated.back() == Dsv41SpecialTokens::kEos) {
        generated.pop_back();
    }

    const auto t_decode_end = std::chrono::steady_clock::now();
    out_stats.decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    out_stats.dec_p150_dev_ms = impl_->accum_p150_dev_ms;
    out_stats.t_attn_ms = impl_->accum_attn_ms;
    out_stats.t_hsi_index_ms = impl_->accum_hsi_ms;
    out_stats.t_mi50_moe_ms = impl_->accum_mi50_moe_ms;
    out_stats.t_p150_wait_ms = impl_->accum_p150_wait_ms;
    out_stats.t_allreduce_ms = impl_->accum_allreduce_ms;
    out_stats.t_engram_wait_ms = impl_->accum_engram_wait_ms;
    if (impl_->sync_timers) {
        std::printf("[SUB_PROF] att[qkv=%.0fms sparse=%.0fms wo=%.0fms ar=%.0fms] | moe[sh_rtr=%.0fms hlut=%.0fms ar=%.0fms p150w=%.0fms] | commit=%.0fms\n",
                    impl_->sub_att_qkv_ms, impl_->sub_att_sparse_ms, impl_->sub_att_wo_ms, impl_->sub_att_ar_ms,
                    impl_->sub_moe_sh_ms, impl_->sub_moe_hlut_ms, impl_->sub_moe_ar_ms, impl_->accum_p150_wait_ms,
                    impl_->sub_commit_ms);
        std::fflush(stdout);
    }
    return generated;
}

}  // namespace dsv41
