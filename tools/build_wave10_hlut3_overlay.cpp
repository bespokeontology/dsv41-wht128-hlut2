// Wave 10 Deliverable: 5x MI50 Parallel rocBLAS TARQ RAREBAL-LDLQ + WHT128-HLUT3 Dual-Switchbox Overlay Builder
// SB-M5-P2-192G Topology:
//   - dev3 (0000:85:00.0, 16 GiB, Switch #1): 384/384 DSpark MTP + 720 minimax-balanced backbone HLUT3 (1,104 total)
//   - dev4 (0000:08:00.0, 16 GiB, Switch #2): 1,080 minimax-balanced backbone HLUT3 (0 MTP)
//   - TP3  (C7/88/05, 3x 32 GiB):            1,160 sharded TP3 backbone HLUT3 (+ compacted HLUT2 tail)
//   Total promoted WHT128-HLUT3 experts = 3,344 (3.125 bpw)
//
// Modes:
//   --link-check   : Self-test HLUT3 pack, lossless TP3 <-> unsharded roundtrip, & minimax balancer
//   --mode plan-only : Deterministic minimax dual-switchbox 3,344-expert plan -> manifests/wave10_192g_tier_plan.txt
//                      + optional <cal_dir>/wave9_2_harvest_targets.txt (only newly promoted / uncalibrated experts!)
//   --mode full-wave10 : Reuses verified Wave 9.2 RAREBAL-LDLQ tiles losslessly + solves newly promoted experts
//                        across all 5x MI50 GPUs in parallel with strict Dual-Receipt verification (0 violations).
// 100% Pure C++17 + HIP/rocBLAS + OpenMP + AVX2 — Zero Python Anywhere.

#include "repack_rarebal_helpers.hpp"
#include "../src/hlut3_pack.hpp"
#include "../src/dsv41_hlut3_rarebal.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace dsv41;

namespace {

std::string sha256_file_hex(const std::string& path) {
    static constexpr uint32_t K[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };
    auto rotr = [](uint32_t x, uint32_t n) noexcept { return (x >> n) | (x << (32u - n)); };
    uint32_t h[8] = {
        0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u
    };
    auto compress = [&](const uint8_t blk[64]) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(blk[4 * i]) << 24) | (uint32_t(blk[4 * i + 1]) << 16) |
                   (uint32_t(blk[4 * i + 2]) << 8) | uint32_t(blk[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    };

    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("sha256 cannot open " + path);
    std::vector<uint8_t> buf(4ULL << 20);
    uint64_t total_bytes = 0ULL;
    uint8_t tail[64]{};
    size_t tail_len = 0;
    while (f) {
        f.read(reinterpret_cast<char*>(buf.data()), std::streamsize(buf.size()));
        const size_t n = size_t(f.gcount());
        if (n == 0) break;
        total_bytes += n;
        size_t off = 0;
        if (tail_len > 0) {
            const size_t take = std::min<size_t>(64u - tail_len, n);
            std::memcpy(tail + tail_len, buf.data(), take);
            tail_len += take;
            off += take;
            if (tail_len == 64u) { compress(tail); tail_len = 0; }
        }
        while (off + 64u <= n) {
            compress(buf.data() + off);
            off += 64u;
        }
        if (off < n) {
            tail_len = n - off;
            std::memcpy(tail, buf.data() + off, tail_len);
        }
    }
    tail[tail_len++] = 0x80u;
    if (tail_len > 56u) {
        std::memset(tail + tail_len, 0, 64u - tail_len);
        compress(tail);
        tail_len = 0;
    }
    std::memset(tail + tail_len, 0, 56u - tail_len);
    const uint64_t bits = total_bytes * 8ULL;
    for (int i = 0; i < 8; ++i) tail[56 + i] = uint8_t((bits >> (56 - 8 * i)) & 0xFFu);
    compress(tail);

    char out[65];
    for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
    out[64] = '\0';
    return std::string(out);
}

enum class TargetTier : int {
    kDev3Mtp = 0,
    kDev3Backbone = 1,
    kDev4Backbone = 2,
    kTp3Backbone = 3
};

const char* tier_name(TargetTier t) noexcept {
    switch (t) {
        case TargetTier::kDev3Mtp:      return "DEV3_MTP";
        case TargetTier::kDev3Backbone: return "DEV3_BB";
        case TargetTier::kDev4Backbone: return "DEV4_BB";
        case TargetTier::kTp3Backbone:  return "TP3_BB";
    }
    return "UNKNOWN";
}

struct TargetExpert {
    TargetTier tier = TargetTier::kDev3Mtp;
    int bid = 0;
    int logical_eid = 0;
    int base_phys_eid = 0;
    int local_slot = 0;
    uint64_t hist_hits = 0;
    bool reused_w92 = false;
    std::string bin_path;
};

struct Wave10ReceiptRow {
    TargetTier tier = TargetTier::kDev3Mtp;
    int bid = 0;
    int logical_eid = 0;
    int base_phys_eid = 0;
    int local_slot = 0;
    uint64_t hist_hits = 0;
    int m_cal = 0;
    int m_eval = 0;
    int rare_cal = 0;
    int rare_eval = 0;
    double lam_in = 1.0;
    double lam_mid = 1.0;
    DualReceiptMetrics m2_rb{};
    DualReceiptMetrics m3_hi{};
    DualReceiptMetrics m3_rb{};
    bool used_rarebal = false;
    bool reused_w92 = false;
};

// Bit-exact inverse of pack_hlut3_expert_tp3_sharded: stitches 3 TP3 rank shards back into unsharded tiles
void unshard_hlut3_tp3_to_full(
    const uint8_t* const tp_gate[3],
    const uint8_t* const tp_up[3],
    const uint8_t* const tp_down[3],
    uint8_t* out_gate,
    uint8_t* out_up,
    uint8_t* out_down)
{
    constexpr size_t kNtDown = size_t(kHlut3ModelHidden / 32);    // 160
    constexpr size_t kKtDownRank = size_t(kHlut3ModelTp3Inter / 32); // 24
    constexpr size_t kTileB = sizeof(Hlut3Tile32x32);             // 400
    for (int tp = 0; tp < 3; ++tp) {
        std::memcpy(out_gate + size_t(tp) * kHlut3Tp3ProjBytes, tp_gate[tp], kHlut3Tp3ProjBytes);
        std::memcpy(out_up   + size_t(tp) * kHlut3Tp3ProjBytes, tp_up[tp],   kHlut3Tp3ProjBytes);
        for (size_t nt = 0; nt < kNtDown; ++nt) {
            const uint8_t* src_row = tp_down[tp] + nt * kKtDownRank * kTileB;
            uint8_t* dst_row = out_down + (nt * 3u * kKtDownRank + size_t(tp) * kKtDownRank) * kTileB;
            std::memcpy(dst_row, src_row, kKtDownRank * kTileB);
        }
    }
}

void shard_hlut3_full_to_tp3(
    const uint8_t* full_gate,
    const uint8_t* full_up,
    const uint8_t* full_down,
    uint8_t* tp_gate[3],
    uint8_t* tp_up[3],
    uint8_t* tp_down[3])
{
    constexpr size_t kNtDown = size_t(kHlut3ModelHidden / 32);    // 160
    constexpr size_t kKtDownRank = size_t(kHlut3ModelTp3Inter / 32); // 24
    constexpr size_t kTileB = sizeof(Hlut3Tile32x32);             // 400
    for (int tp = 0; tp < 3; ++tp) {
        std::memcpy(tp_gate[tp], full_gate + size_t(tp) * kHlut3Tp3ProjBytes, kHlut3Tp3ProjBytes);
        std::memcpy(tp_up[tp],   full_up   + size_t(tp) * kHlut3Tp3ProjBytes, kHlut3Tp3ProjBytes);
        for (size_t nt = 0; nt < kNtDown; ++nt) {
            const uint8_t* src_row = full_down + (nt * 3u * kKtDownRank + size_t(tp) * kKtDownRank) * kTileB;
            uint8_t* dst_row = tp_down[tp] + nt * kKtDownRank * kTileB;
            std::memcpy(dst_row, src_row, kKtDownRank * kTileB);
        }
    }
}

struct Wave92ReuseEntry {
    bool is_dev3 = false; // true -> in wave9_2_dev3_hlut3, false -> in wave9_2_tp3_hlut3
    int slot = -1;
    Wave10ReceiptRow receipt{};
};

struct Wave92ReuseBank {
    std::unique_ptr<SafetensorsMmap> dev3_st;
    std::unique_ptr<SafetensorsMmap> tp3_st;
    std::map<std::pair<int, int>, Wave92ReuseEntry> entries; // (bid, logical_eid) -> slot & receipt

    void load(const std::string& dev3_path, const std::string& tp3_path, const std::string& receipt_path) {
        entries.clear();
        dev3_st.reset();
        tp3_st.reset();
        std::error_code ec;
        if (!dev3_path.empty() && std::filesystem::is_regular_file(dev3_path, ec)) {
            dev3_st = std::make_unique<SafetensorsMmap>(dev3_path);
        }
        if (!tp3_path.empty() && std::filesystem::is_regular_file(tp3_path, ec)) {
            tp3_st = std::make_unique<SafetensorsMmap>(tp3_path);
        }

        // Parse prior receipts first
        std::map<std::pair<int, int>, Wave10ReceiptRow> rc_map;
        if (!receipt_path.empty() && std::filesystem::is_regular_file(receipt_path, ec)) {
            std::ifstream rf(receipt_path);
            std::string line;
            while (std::getline(rf, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream iss(line);
                std::string tstr, status;
                Wave10ReceiptRow rr{};
                if (iss >> tstr >> rr.bid >> rr.logical_eid >> rr.base_phys_eid >> rr.local_slot
                        >> rr.hist_hits >> rr.m_cal >> rr.m_eval >> rr.rare_cal >> rr.rare_eval
                        >> rr.lam_in >> rr.lam_mid
                        >> rr.m2_rb.cos_all >> rr.m3_hi.cos_all >> rr.m3_rb.cos_all
                        >> rr.m2_rb.snr_db  >> rr.m3_hi.snr_db  >> rr.m3_rb.snr_db
                        >> rr.m2_rb.mse_rare >> rr.m3_hi.mse_rare >> rr.m3_rb.mse_rare
                        >> status)
                {
                    rr.used_rarebal = (status == "EMITTED_RAREBAL");
                    rr.reused_w92 = true;
                    rc_map[{rr.bid, rr.logical_eid}] = rr;
                }
            }
        }

        // Index dev3 MTP (all 384 always reused) + dev3 backbone (only EMITTED_RAREBAL reused; fallback re-calibrated!)
        if (dev3_st) {
            for (int b = 40; b < 43; ++b) {
                const std::string pfx = "dev3.b" + std::to_string(b) + ".";
                if (!dev3_st->contains(pfx + "gate_tiles")) continue;
                for (int e = 0; e < 128; ++e) {
                    Wave92ReuseEntry ent{};
                    ent.is_dev3 = true;
                    ent.slot = e;
                    auto it = rc_map.find({b, e});
                    if (it != rc_map.end()) ent.receipt = it->second;
                    else { ent.receipt.bid = b; ent.receipt.logical_eid = e; ent.receipt.reused_w92 = true; }
                    entries[{b, e}] = ent;
                }
            }
            for (int l = 0; l < 40; ++l) {
                const std::string pfx = "dev3.b" + std::to_string(l) + ".";
                if (!dev3_st->contains(pfx + "logical_eids")) continue;
                const auto& ei = dev3_st->at(pfx + "logical_eids");
                const int ne = int(ei.byte_size() / sizeof(uint16_t));
                const auto* ptr = reinterpret_cast<const uint16_t*>(dev3_st->tensor_bytes(pfx + "logical_eids"));
                for (int s = 0; s < ne; ++s) {
                    const int le = int(ptr[s]);
                    auto it = rc_map.find({l, le});
                    if (it != rc_map.end() && it->second.used_rarebal) {
                        Wave92ReuseEntry ent{};
                        ent.is_dev3 = true;
                        ent.slot = s;
                        ent.receipt = it->second;
                        entries[{l, le}] = ent;
                    }
                }
            }
        }
        // Index tp3 backbone (only EMITTED_RAREBAL reused)
        if (tp3_st) {
            for (int l = 0; l < 40; ++l) {
                const std::string ename = "tp3.b" + std::to_string(l) + ".logical_eids";
                if (!tp3_st->contains(ename)) continue;
                const auto& ei = tp3_st->at(ename);
                const int ne = int(ei.byte_size() / sizeof(uint16_t));
                const auto* ptr = reinterpret_cast<const uint16_t*>(tp3_st->tensor_bytes(ename));
                for (int s = 0; s < ne; ++s) {
                    const int le = int(ptr[s]);
                    auto it = rc_map.find({l, le});
                    if (it != rc_map.end() && it->second.used_rarebal) {
                        Wave92ReuseEntry ent{};
                        ent.is_dev3 = false;
                        ent.slot = s;
                        ent.receipt = it->second;
                        entries[{l, le}] = ent;
                    }
                }
            }
        }
    }

    bool can_reuse(int bid, int le) const {
        return entries.find({bid, le}) != entries.end();
    }

    void extract_unsharded(int bid, int le, uint8_t* out_g, uint8_t* out_u, uint8_t* out_d) const {
        const auto& ent = entries.at({bid, le});
        if (ent.is_dev3) {
            const std::string pfx = "dev3.b" + std::to_string(bid) + ".";
            const size_t off = size_t(ent.slot) * kHlut3UnshardedProjBytes;
            std::memcpy(out_g, dev3_st->tensor_bytes(pfx + "gate_tiles") + off, kHlut3UnshardedProjBytes);
            std::memcpy(out_u, dev3_st->tensor_bytes(pfx + "up_tiles")   + off, kHlut3UnshardedProjBytes);
            std::memcpy(out_d, dev3_st->tensor_bytes(pfx + "down_tiles") + off, kHlut3UnshardedProjBytes);
        } else {
            const uint8_t* tg[3];
            const uint8_t* tu[3];
            const uint8_t* td[3];
            const size_t off = size_t(ent.slot) * kHlut3Tp3ProjBytes;
            for (int r = 0; r < 3; ++r) {
                const std::string pfx = "tp3.tp" + std::to_string(r) + ".b" + std::to_string(bid) + ".";
                tg[r] = tp3_st->tensor_bytes(pfx + "gate_tiles") + off;
                tu[r] = tp3_st->tensor_bytes(pfx + "up_tiles")   + off;
                td[r] = tp3_st->tensor_bytes(pfx + "down_tiles") + off;
            }
            unshard_hlut3_tp3_to_full(tg, tu, td, out_g, out_u, out_d);
        }
    }
};

void load_block_wht_signs(const std::string& mi50_root, int bid,
                          std::vector<uint64_t>& signs_h, std::vector<uint64_t>& signs_i) {
    char tag[64];
    if (bid < 40) std::snprintf(tag, sizeof(tag), "/layer_%02d.safetensors", bid);
    else          std::snprintf(tag, sizeof(tag), "/mtp_%d.safetensors", bid - 40);
    signs_i.clear();
    for (int tp = 0; tp < 3; ++tp) {
        SafetensorsMmap st(mi50_root + "/tp" + std::to_string(tp) + tag);
        if (tp == 0) {
            const auto* sh = reinterpret_cast<const uint64_t*>(st.tensor_bytes("wht.signs_h"));
            signs_h.assign(sh, sh + 80);
        }
        const auto* si = reinterpret_cast<const uint64_t*>(st.tensor_bytes("wht.signs_i"));
        signs_i.insert(signs_i.end(), si, si + 12);
    }
}

Hlut3QuantMatrix dequant_mxfp4_and_quantize_hlut3_hi(
    const uint8_t* w_mxfp4,
    const uint8_t* s_ue8m0,
    int N,
    int K,
    const std::vector<uint64_t>& signs,
    const Hlut3LloydMaxSolution& sol,
    std::vector<float>& out_w_rot)
{
    static constexpr float kE2M1Lut[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    const int G = K / 128;
    const int half_cols = K / 2;
    const int s_cols = K / 32;
    out_w_rot.resize(size_t(N) * size_t(K));

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < N; ++r) {
        const uint8_t* wr = w_mxfp4 + size_t(r) * half_cols;
        const uint8_t* sr = s_ue8m0 + size_t(r) * s_cols;
        float* dr = out_w_rot.data() + size_t(r) * K;
        for (int sb = 0; sb < s_cols; ++sb) {
            const uint8_t e8 = sr[sb];
            const uint32_t scale_bits = (e8 == 0) ? 0x00400000u : (uint32_t(e8) << 23);
            float scale = 0.0f;
            std::memcpy(&scale, &scale_bits, 4);
            const uint8_t* wb = wr + sb * 16;
            float* db = dr + sb * 32;
            for (int b = 0; b < 16; ++b) {
                const uint8_t byte = wb[b];
                db[2 * b + 0] = kE2M1Lut[byte & 0x0Fu] * scale;
                db[2 * b + 1] = kE2M1Lut[byte >> 4] * scale;
            }
        }
        for (int g = 0; g < G; ++g) {
            avx2_wht128_block(dr + g * 128, &signs[size_t(g) * 2]);
        }
    }

    Hlut3QuantMatrix q;
    q.N = N;
    q.K = K;
    q.nt = N / kHlut3TileRows;
    q.kt = K / kHlut3TileCols;
    q.lloyd = sol;
    q.tiles.assign(size_t(q.nt) * size_t(q.kt), Hlut3Tile32x32{});

    #pragma omp parallel for collapse(2) schedule(static)
    for (int nt_i = 0; nt_i < q.nt; ++nt_i) {
        for (int kt_i = 0; kt_i < q.kt; ++kt_i) {
            hlut3_pack_tile(out_w_rot.data(), K, nt_i * kHlut3TileRows, kt_i * kHlut3TileCols,
                            sol, q.tiles[size_t(nt_i) * size_t(q.kt) + size_t(kt_i)]);
        }
    }
    return q;
}

class GpuHlut3LdlqHelper {
public:
    static constexpr size_t kMaxD = 5120;

    explicit GpuHlut3LdlqHelper(int dev_id) {
        if (hipSetDevice(dev_id) != hipSuccess) return;
        if (hipStreamCreate(&stream_) != hipSuccess) return;
        if (rocblas_create_handle(&handle_) != rocblas_status_success) {
            (void)hipStreamDestroy(stream_);
            stream_ = nullptr;
            return;
        }
        (void)rocblas_set_stream(handle_, stream_);
        if (hipMalloc(&d_rup_,  kMaxD * kMaxD * sizeof(float)) != hipSuccess ||
            hipMalloc(&d_err_,  kMaxD * kMaxD * sizeof(float)) != hipSuccess ||
            hipMalloc(&d_wpan_, kMaxD * 128ULL * sizeof(float)) != hipSuccess) {
            cleanup();
            return;
        }
        ok_ = true;
    }
    ~GpuHlut3LdlqHelper() { cleanup(); }

    void upload_rup(const std::vector<float>& R_up, int K) {
        if (!ok_ || R_up.size() != size_t(K) * size_t(K)) return;
        (void)hipMemcpyAsync(d_rup_, R_up.data(), R_up.size() * sizeof(float), hipMemcpyHostToDevice, stream_);
        (void)hipStreamSynchronize(stream_);
    }

    Hlut3QuantMatrix apply_ldlq(const Hlut3QuantMatrix& q_seed, const float* w_rot, const std::vector<float>& R_up) {
        if (!ok_ || size_t(q_seed.N) > kMaxD || size_t(q_seed.K) > kMaxD) {
            return apply_hlut3_ldlq_with_rup(q_seed, w_rot, R_up, true);
        }
        Hlut3QuantMatrix q = q_seed;
        const int N = q.N;
        const int K = q.K;
        if (R_up.size() != size_t(K) * size_t(K)) return q;
        const int G = K / 128;

        std::vector<float> h_wpan(size_t(N) * 128u);
        std::vector<float> h_epan(size_t(N) * 128u);
        const float one = 1.0f;

        for (int g = 0; g < G; ++g) {
            const int col0 = g * 128;
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < N; ++r) {
                std::memcpy(h_wpan.data() + size_t(r) * 128u, w_rot + size_t(r) * K + col0, 128u * sizeof(float));
            }
            if (col0 > 0) {
                (void)hipMemcpyAsync(d_wpan_, h_wpan.data(), h_wpan.size() * sizeof(float), hipMemcpyHostToDevice, stream_);
                (void)rocblas_sgemm(handle_, rocblas_operation_none, rocblas_operation_none,
                                    128, N, col0, &one, d_rup_ + col0, K, d_err_, K, &one, d_wpan_, 128);
                (void)hipMemcpyAsync(h_wpan.data(), d_wpan_, h_wpan.size() * sizeof(float), hipMemcpyDeviceToHost, stream_);
                (void)hipStreamSynchronize(stream_);
            }

            #pragma omp parallel for schedule(static)
            for (int r = 0; r < N; ++r) {
                float* wp = h_wpan.data() + size_t(r) * 128u;
                float* ep = h_epan.data() + size_t(r) * 128u;
                const float* orig_row = w_rot + size_t(r) * K + col0;
                const int nt_i = r / kHlut3TileRows;
                const int local_r = r % kHlut3TileRows;
                const int scale_g = hlut3_scale_group_for_row(local_r);

                for (int j = 0; j < 128; ++j) {
                    const int col = col0 + j;
                    const int kt_i = col / kHlut3TileCols;
                    const int local_c = col % kHlut3TileCols;
                    Hlut3Tile32x32& tile = q.tiles[size_t(nt_i) * size_t(q.kt) + size_t(kt_i)];
                    const float sigma = fp16_to_f32(tile.scales_f16[scale_g]);
                    const float v = wp[j];
                    const double z = (sigma > 0.0f) ? double(std::fabs(v)) / double(sigma) : 0.0;
                    uint32_t mag = 0u;
                    if (z >= q.lloyd.thresholds[2]) mag = 3u;
                    else if (z >= q.lloyd.thresholds[1]) mag = 2u;
                    else if (z >= q.lloyd.thresholds[0]) mag = 1u;
                    else mag = 0u;
                    const uint32_t sign = std::signbit(v) ? 1u : 0u;
                    hlut3_pack_codes_to_seed_tile(tile, local_r, local_c, mag, sign);

                    const float qv = (sign ? -1.0f : 1.0f) * float(q.lloyd.levels[mag]) * sigma;
                    const float err = orig_row[j] - qv;
                    ep[j] = err;
                    const float* rrow = R_up.data() + size_t(col) * K + col0;
                    for (int jj = j + 1; jj < 128; ++jj) {
                        wp[jj] += err * rrow[jj];
                    }
                }
            }

            if (g + 1 < G) {
                (void)hipMemcpy2DAsync(d_err_ + col0, size_t(K) * sizeof(float),
                                       h_epan.data(), 128u * sizeof(float),
                                       128u * sizeof(float), size_t(N),
                                       hipMemcpyHostToDevice, stream_);
            }
        }
        (void)hipStreamSynchronize(stream_);
        return q;
    }

private:
    bool ok_ = false;
    hipStream_t stream_ = nullptr;
    rocblas_handle handle_ = nullptr;
    float *d_rup_ = nullptr, *d_err_ = nullptr, *d_wpan_ = nullptr;

    void cleanup() {
        if (d_wpan_) { (void)hipFree(d_wpan_); d_wpan_ = nullptr; }
        if (d_err_)  { (void)hipFree(d_err_);  d_err_  = nullptr; }
        if (d_rup_)  { (void)hipFree(d_rup_);  d_rup_  = nullptr; }
        if (handle_) { (void)rocblas_destroy_handle(handle_); handle_ = nullptr; }
        if (stream_) { (void)hipStreamDestroy(stream_); stream_ = nullptr; }
        ok_ = false;
    }
};

class AsyncTargetPrefetcher {
public:
    static constexpr size_t kWin = 64;

    AsyncTargetPrefetcher(HfShardRangeFetcher& fetcher,
                          const std::vector<TargetExpert>& items,
                          const std::vector<size_t>& work_indices,
                          int n_threads = 32)
        : fetcher_(fetcher), items_(items), work_indices_(work_indices)
    {
        for (auto& s : slots_) s.wpos = ~size_t(0);
        if (work_indices_.empty()) return;
        const int nt = std::max(1, n_threads);
        workers_.reserve(size_t(nt));
        for (int t = 0; t < nt; ++t) workers_.emplace_back([this]() { worker_loop(); });
    }
    ~AsyncTargetPrefetcher() { stop(); }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_prod_.notify_all();
        cv_cons_.notify_all();
        for (auto& th : workers_) if (th.joinable()) th.join();
        workers_.clear();
    }

    bool take_next(size_t& out_plan_idx, RawExpertMxfp4& out_raw, std::string& out_err) {
        std::lock_guard<std::mutex> ord_lk(take_mtx_);
        size_t wpos = 0;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            if (stop_ || next_consume_ >= work_indices_.size()) return false;
            wpos = next_consume_++;
            Slot& s = slots_[wpos % kWin];
            cv_cons_.wait(lk, [&]() { return stop_ || (s.ready && s.wpos == wpos); });
            if (stop_ && !(s.ready && s.wpos == wpos)) return false;
            out_raw = std::move(s.raw);
            out_err = std::move(s.err);
            s.ready = false;
            s.wpos = ~size_t(0);
            s.err.clear();
            if (wpos + 1 > consumed_) consumed_ = wpos + 1;
        }
        cv_prod_.notify_all();
        out_plan_idx = work_indices_[wpos];
        return true;
    }

private:
    struct Slot {
        size_t wpos = ~size_t(0);
        bool ready = false;
        RawExpertMxfp4 raw{};
        std::string err{};
    };
    void worker_loop() {
        for (;;) {
            size_t my_wpos = ~size_t(0);
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_prod_.wait(lk, [&]() {
                    return stop_ || (next_issue_ < work_indices_.size() && next_issue_ < consumed_ + kWin &&
                                     !slots_[next_issue_ % kWin].ready);
                });
                if (stop_ || next_issue_ >= work_indices_.size()) return;
                my_wpos = next_issue_++;
            }
            const auto& te = items_[work_indices_[my_wpos]];
            RawExpertMxfp4 fetched{};
            std::string err_msg;
            try {
                fetched = fetcher_.fetch_expert(te.bid, te.logical_eid);
            } catch (const std::exception& ex) {
                err_msg = ex.what();
            }
            {
                std::lock_guard<std::mutex> lk(mtx_);
                Slot& s = slots_[my_wpos % kWin];
                s.wpos = my_wpos;
                s.raw = std::move(fetched);
                s.err = std::move(err_msg);
                s.ready = true;
            }
            cv_cons_.notify_all();
        }
    }

    HfShardRangeFetcher& fetcher_;
    const std::vector<TargetExpert>& items_;
    const std::vector<size_t>& work_indices_;
    std::mutex take_mtx_;
    std::mutex mtx_;
    std::condition_variable cv_prod_;
    std::condition_variable cv_cons_;
    std::vector<Slot> slots_{kWin};
    size_t next_issue_ = 0;
    size_t next_consume_ = 0;
    size_t consumed_ = 0;
    bool stop_ = false;
    std::vector<std::thread> workers_;
};

// Exact 2-opt partition of a single layer's sorted items (hits descending) into D3 (size n3) and D4 (size n - n3)
// minimizing |sum(D3) - sum(D4)|.
struct LayerCand { uint64_t h; int le; int pe; };

std::pair<uint64_t, uint64_t> balance_layer_2opt(
    const std::vector<LayerCand>& items_desc,
    int n3,
    std::vector<uint8_t>& out_is_d3)
{
    const int n = int(items_desc.size());
    out_is_d3.assign(size_t(n), 0u);
    n3 = std::clamp(n3, 0, n);
    uint64_t s3 = 0, s4 = 0;
    for (int i = 0; i < n; ++i) {
        if (i < n3) { out_is_d3[size_t(i)] = 1u; s3 += items_desc[size_t(i)].h; }
        else        { out_is_d3[size_t(i)] = 0u; s4 += items_desc[size_t(i)].h; }
    }
    auto abs_diff = [](uint64_t a, uint64_t b) { return (a >= b) ? (a - b) : (b - a); };
    for (int iter = 0; iter < 512; ++iter) {
        const uint64_t cur = abs_diff(s3, s4);
        uint64_t best = cur;
        int best_i = -1, best_j = -1;
        for (int i = 0; i < n; ++i) {
            if (!out_is_d3[size_t(i)]) continue;
            const uint64_t hi = items_desc[size_t(i)].h;
            for (int j = 0; j < n; ++j) {
                if (out_is_d3[size_t(j)]) continue;
                const uint64_t hj = items_desc[size_t(j)].h;
                const uint64_t ns3 = s3 - hi + hj;
                const uint64_t ns4 = s4 - hj + hi;
                const uint64_t nd = abs_diff(ns3, ns4);
                if (nd < best) {
                    best = nd;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        if (best_i < 0) break;
        const uint64_t hi = items_desc[size_t(best_i)].h;
        const uint64_t hj = items_desc[size_t(best_j)].h;
        out_is_d3[size_t(best_i)] = 0u;
        out_is_d3[size_t(best_j)] = 1u;
        s3 = s3 - hi + hj;
        s4 = s4 - hj + hi;
    }
    return {s3, s4};
}

struct MinimaxPlanStats {
    uint64_t total_dev3_hits = 0;
    uint64_t total_dev4_hits = 0;
    double global_imbalance_pct = 0.0;
    double max_layer_imbalance_pct = 0.0;
    double mean_layer_imbalance_pct = 0.0;
    std::array<uint64_t, 40> layer_dev3_hits{};
    std::array<uint64_t, 40> layer_dev4_hits{};
};

std::vector<TargetExpert> build_wave10_minimax_plan(
    const std::string& hist_path,
    const std::string& perm_ov_path,
    int dev3_bb_target,
    int dev4_bb_target,
    int tp3_bb_target,
    int offload_per_layer_cap,
    int dev3_per_layer_cap,
    int dev4_per_layer_cap,
    int tp3_per_layer_cap,
    std::array<std::vector<uint16_t>, 40>& out_dev3_eids,
    std::array<std::vector<uint16_t>, 40>& out_dev4_eids,
    std::array<std::vector<uint16_t>, 40>& out_tp3_eids,
    MinimaxPlanStats& out_stats)
{
    std::array<std::array<uint16_t, 384>, 40> perm{};
    for (int l = 0; l < 40; ++l) {
        for (int e = 0; e < 384; ++e) perm[size_t(l)][size_t(e)] = uint16_t(e);
    }
    if (!perm_ov_path.empty()) {
        SafetensorsMmap st(perm_ov_path);
        if (!st.contains("expert_perm_u16")) {
            throw std::runtime_error("Missing expert_perm_u16 in " + perm_ov_path);
        }
        std::memcpy(perm.data(), st.tensor_bytes("expert_perm_u16"), sizeof(perm));
    }

    std::array<std::array<uint64_t, 384>, 43> hits{};
    {
        std::ifstream hf(hist_path);
        if (!hf) throw std::runtime_error("Cannot open router_hist file: " + hist_path);
        std::string line;
        while (std::getline(hf, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            int b = -1, e = -1;
            uint64_t h = 0, seen = 0;
            if ((iss >> b >> e >> h >> seen) && b >= 0 && b < 43 && e >= 0 && e < 384) {
                hits[size_t(b)][size_t(e)] = h;
            }
        }
    }

    struct Cand { uint64_t h; int l; int le; int pe; };
    std::vector<Cand> mi50_cands;
    mi50_cands.reserve(40 * 216);
    for (int l = 0; l < 40; ++l) {
        for (int le = 0; le < 384; ++le) {
            const int pe = int(perm[size_t(l)][size_t(le)]);
            if (pe < 216) {
                mi50_cands.push_back(Cand{hits[size_t(l)][size_t(le)], l, le, pe});
            }
        }
    }
    std::sort(mi50_cands.begin(), mi50_cands.end(), [](const Cand& a, const Cand& b) {
        if (a.h != b.h) return a.h > b.h;
        return (a.l != b.l) ? (a.l < b.l) : (a.le < b.le);
    });

    const int offload_total = dev3_bb_target + dev4_bb_target;
    std::array<std::vector<LayerCand>, 40> offload_pool{};
    std::array<std::array<uint8_t, 384>, 40> assigned{};

    int got_offload = 0;
    for (const auto& c : mi50_cands) {
        if (got_offload >= offload_total) break;
        if (int(offload_pool[size_t(c.l)].size()) >= offload_per_layer_cap) continue;
        offload_pool[size_t(c.l)].push_back(LayerCand{c.h, c.le, c.pe});
        assigned[size_t(c.l)][size_t(c.le)] = 1u;
        ++got_offload;
    }

    for (int l = 0; l < 40; ++l) {
        out_dev3_eids[size_t(l)].clear();
        out_dev4_eids[size_t(l)].clear();
        out_tp3_eids[size_t(l)].clear();
    }

    int got_tp3 = 0;
    for (const auto& c : mi50_cands) {
        if (got_tp3 >= tp3_bb_target) break;
        if (assigned[size_t(c.l)][size_t(c.le)] != 0u) continue;
        if (int(out_tp3_eids[size_t(c.l)].size()) >= tp3_per_layer_cap) continue;
        out_tp3_eids[size_t(c.l)].push_back(uint16_t(c.le));
        assigned[size_t(c.l)][size_t(c.le)] = 3u;
        ++got_tp3;
    }

    if (got_offload != offload_total || got_tp3 != tp3_bb_target) {
        throw std::runtime_error("Failed to satisfy Wave 10 offload/TP3 targets under per-layer caps");
    }

    // Step 1: Initial per-layer dev3 quota proportional to offload_pool[l].size(), adjusted to exact dev3_bb_target
    std::array<int, 40> n3{};
    int sum_n3 = 0;
    for (int l = 0; l < 40; ++l) {
        const int nl = int(offload_pool[size_t(l)].size());
        const int lo = std::max(0, nl - dev4_per_layer_cap);
        const int hi = std::min(nl, dev3_per_layer_cap);
        const int q = int(std::llround(double(nl) * double(dev3_bb_target) / double(offload_total)));
        n3[size_t(l)] = std::clamp(q, lo, hi);
        sum_n3 += n3[size_t(l)];
    }
    while (sum_n3 < dev3_bb_target) {
        int best_l = -1;
        double best_gap = -1e18;
        for (int l = 0; l < 40; ++l) {
            const int nl = int(offload_pool[size_t(l)].size());
            if (n3[size_t(l)] >= std::min(nl, dev3_per_layer_cap)) continue;
            const double ideal = double(nl) * double(dev3_bb_target) / double(offload_total);
            const double gap = ideal - double(n3[size_t(l)]);
            if (gap > best_gap) { best_gap = gap; best_l = l; }
        }
        if (best_l < 0) throw std::runtime_error("Cannot increase sum_n3 to dev3_bb_target");
        ++n3[size_t(best_l)];
        ++sum_n3;
    }
    while (sum_n3 > dev3_bb_target) {
        int best_l = -1;
        double best_gap = -1e18;
        for (int l = 0; l < 40; ++l) {
            const int nl = int(offload_pool[size_t(l)].size());
            if (n3[size_t(l)] <= std::max(0, nl - dev4_per_layer_cap)) continue;
            const double ideal = double(nl) * double(dev3_bb_target) / double(offload_total);
            const double gap = double(n3[size_t(l)]) - ideal;
            if (gap > best_gap) { best_gap = gap; best_l = l; }
        }
        if (best_l < 0) throw std::runtime_error("Cannot decrease sum_n3 to dev3_bb_target");
        --n3[size_t(best_l)];
        --sum_n3;
    }

    // Step 2: Run 2-opt minimax balancing per layer + global cross-layer quota exchange
    std::array<std::vector<uint8_t>, 40> is_d3{};
    std::array<uint64_t, 40> s3{}, s4{};
    for (int l = 0; l < 40; ++l) {
        auto [a, b] = balance_layer_2opt(offload_pool[size_t(l)], n3[size_t(l)], is_d3[size_t(l)]);
        s3[size_t(l)] = a;
        s4[size_t(l)] = b;
    }

    auto rel_imbal = [](uint64_t a, uint64_t b) {
        const double avg = 0.5 * double(a + b);
        if (avg <= 0.0) return 0.0;
        const double d = (a >= b) ? double(a - b) : double(b - a);
        return 100.0 * d / avg;
    };

    // Outer quota exchange: move 1 dev3 slot from an over-loaded-dev3 layer to an under-loaded-dev3 layer
    for (int iter = 0; iter < 400; ++iter) {
        double best_gain = 1e-6;
        int best_la = -1, best_lb = -1;
        for (int la = 0; la < 40; ++la) {
            const int na = int(offload_pool[size_t(la)].size());
            if (n3[size_t(la)] >= std::min(na, dev3_per_layer_cap)) continue;
            if (s3[size_t(la)] >= s4[size_t(la)]) continue; // la does not need more dev3 mass
            std::vector<uint8_t> ta;
            auto [a3, a4] = balance_layer_2opt(offload_pool[size_t(la)], n3[size_t(la)] + 1, ta);
            const double old_ia = rel_imbal(s3[size_t(la)], s4[size_t(la)]);
            const double new_ia = rel_imbal(a3, a4);
            if (new_ia >= old_ia) continue;

            for (int lb = 0; lb < 40; ++lb) {
                if (lb == la) continue;
                const int nb = int(offload_pool[size_t(lb)].size());
                if (n3[size_t(lb)] <= std::max(0, nb - dev4_per_layer_cap)) continue;
                std::vector<uint8_t> tb;
                auto [b3, b4] = balance_layer_2opt(offload_pool[size_t(lb)], n3[size_t(lb)] - 1, tb);
                const double old_ib = rel_imbal(s3[size_t(lb)], s4[size_t(lb)]);
                const double new_ib = rel_imbal(b3, b4);
                const double gain = (old_ia + old_ib) - (new_ia + new_ib);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_la = la;
                    best_lb = lb;
                }
            }
        }
        if (best_la < 0) break;
        ++n3[size_t(best_la)];
        --n3[size_t(best_lb)];
        auto [a3, a4] = balance_layer_2opt(offload_pool[size_t(best_la)], n3[size_t(best_la)], is_d3[size_t(best_la)]);
        s3[size_t(best_la)] = a3; s4[size_t(best_la)] = a4;
        auto [b3, b4] = balance_layer_2opt(offload_pool[size_t(best_lb)], n3[size_t(best_lb)], is_d3[size_t(best_lb)]);
        s3[size_t(best_lb)] = b3; s4[size_t(best_lb)] = b4;
    }

    // Collect final assignments and stats
    out_stats = MinimaxPlanStats{};
    double sum_layer_imbal = 0.0;
    for (int l = 0; l < 40; ++l) {
        for (size_t i = 0; i < offload_pool[size_t(l)].size(); ++i) {
            if (is_d3[size_t(l)][i]) out_dev3_eids[size_t(l)].push_back(uint16_t(offload_pool[size_t(l)][i].le));
            else                     out_dev4_eids[size_t(l)].push_back(uint16_t(offload_pool[size_t(l)][i].le));
        }
        std::sort(out_dev3_eids[size_t(l)].begin(), out_dev3_eids[size_t(l)].end());
        std::sort(out_dev4_eids[size_t(l)].begin(), out_dev4_eids[size_t(l)].end());
        std::sort(out_tp3_eids[size_t(l)].begin(), out_tp3_eids[size_t(l)].end());

        out_stats.layer_dev3_hits[size_t(l)] = s3[size_t(l)];
        out_stats.layer_dev4_hits[size_t(l)] = s4[size_t(l)];
        out_stats.total_dev3_hits += s3[size_t(l)];
        out_stats.total_dev4_hits += s4[size_t(l)];
        const double li = rel_imbal(s3[size_t(l)], s4[size_t(l)]);
        out_stats.max_layer_imbalance_pct = std::max(out_stats.max_layer_imbalance_pct, li);
        sum_layer_imbal += li;
    }
    out_stats.mean_layer_imbalance_pct = sum_layer_imbal / 40.0;
    out_stats.global_imbalance_pct = rel_imbal(out_stats.total_dev3_hits, out_stats.total_dev4_hits);

    std::vector<TargetExpert> plan;
    plan.reserve(size_t(384 + dev3_bb_target + dev4_bb_target + tp3_bb_target));

    for (int b = 40; b < 43; ++b) {
        for (int e = 0; e < 128; ++e) {
            plan.push_back(TargetExpert{TargetTier::kDev3Mtp, b, e, e, e, hits[size_t(b)][size_t(e)], false, ""});
        }
    }
    for (int l = 0; l < 40; ++l) {
        for (size_t j = 0; j < out_dev3_eids[size_t(l)].size(); ++j) {
            const int le = int(out_dev3_eids[size_t(l)][j]);
            plan.push_back(TargetExpert{TargetTier::kDev3Backbone, l, le, int(perm[size_t(l)][size_t(le)]),
                                        int(j), hits[size_t(l)][size_t(le)], false, ""});
        }
    }
    for (int l = 0; l < 40; ++l) {
        for (size_t j = 0; j < out_dev4_eids[size_t(l)].size(); ++j) {
            const int le = int(out_dev4_eids[size_t(l)][j]);
            plan.push_back(TargetExpert{TargetTier::kDev4Backbone, l, le, int(perm[size_t(l)][size_t(le)]),
                                        int(j), hits[size_t(l)][size_t(le)], false, ""});
        }
    }
    for (int l = 0; l < 40; ++l) {
        for (size_t j = 0; j < out_tp3_eids[size_t(l)].size(); ++j) {
            const int le = int(out_tp3_eids[size_t(l)][j]);
            plan.push_back(TargetExpert{TargetTier::kTp3Backbone, l, le, int(perm[size_t(l)][size_t(le)]),
                                        int(j), hits[size_t(l)][size_t(le)], false, ""});
        }
    }
    return plan;
}

void write_tier_plan_manifest(
    const std::string& manifest_path,
    const std::vector<TargetExpert>& plan,
    const std::array<std::vector<uint16_t>, 40>& dev3_eids,
    const std::array<std::vector<uint16_t>, 40>& dev4_eids,
    const std::array<std::vector<uint16_t>, 40>& tp3_eids,
    const MinimaxPlanStats& st)
{
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(manifest_path).parent_path(), ec);
    std::ofstream f(manifest_path, std::ios::trunc);
    int n_mtp = 0, n_d3 = 0, n_d4 = 0, n_tp3 = 0, n_reused = 0;
    for (const auto& t : plan) {
        if (t.tier == TargetTier::kDev3Mtp) ++n_mtp;
        else if (t.tier == TargetTier::kDev3Backbone) ++n_d3;
        else if (t.tier == TargetTier::kDev4Backbone) ++n_d4;
        else ++n_tp3;
        if (t.reused_w92) ++n_reused;
    }
    f << "# Wave 10 SB-M5-P2-192G Dual-Switchbox Deterministic Minimax Tier Partition Plan\n"
      << "# total_promoted_hlut3=" << plan.size()
      << " dev3_mtp=" << n_mtp
      << " dev3_backbone=" << n_d3
      << " dev4_backbone=" << n_d4
      << " tp3_backbone=" << n_tp3
      << " reused_w92=" << n_reused
      << " new_to_calibrate=" << (int(plan.size()) - n_reused) << "\n"
      << "# minimax_switch_balance: total_dev3_hits=" << st.total_dev3_hits
      << " total_dev4_hits=" << st.total_dev4_hits
      << " global_imbalance_pct=" << std::fixed << std::setprecision(3) << st.global_imbalance_pct
      << " mean_layer_imbalance_pct=" << st.mean_layer_imbalance_pct
      << " max_layer_imbalance_pct=" << st.max_layer_imbalance_pct << "\n"
      << "# per_layer_summary: layer n_dev3 n_dev4 n_tp3 n_compacted_hlut2 hits_dev3 hits_dev4 imbal_pct\n";
    for (int l = 0; l < 40; ++l) {
        const int nd3 = int(dev3_eids[size_t(l)].size());
        const int nd4 = int(dev4_eids[size_t(l)].size());
        const int ntp = int(tp3_eids[size_t(l)].size());
        const uint64_t h3 = st.layer_dev3_hits[size_t(l)];
        const uint64_t h4 = st.layer_dev4_hits[size_t(l)];
        const double avg = 0.5 * double(h3 + h4);
        const double pct = (avg > 0.0) ? 100.0 * double(h3 >= h4 ? h3 - h4 : h4 - h3) / avg : 0.0;
        f << "# L" << std::setw(2) << std::setfill('0') << l << std::setfill(' ')
          << " dev3=" << nd3 << " dev4=" << nd4 << " tp3=" << ntp
          << " hlut2=" << (216 - nd3 - nd4 - ntp)
          << " h3=" << h3 << " h4=" << h4 << " imbal=" << std::fixed << std::setprecision(2) << pct << "%\n";
    }
    f << "# tier block logical_eid base_phys_eid local_slot router_hits_wave8_5 reused_w92\n";
    for (const auto& t : plan) {
        f << tier_name(t.tier) << " " << t.bid << " " << t.logical_eid << " " << t.base_phys_eid << " "
          << t.local_slot << " " << t.hist_hits << " " << (t.reused_w92 ? 1 : 0) << "\n";
    }
}

bool run_self_link_check() {
    const auto sol = solve_gaussian_lloyd_max_hlut3();
    if (!sol.converged || sol.sqnr_db < 14.6) return false;

    // 1. Verify lossless unsharded <-> TP3-sharded roundtrip is 100% bit-exact!
    std::vector<uint8_t> g0(kHlut3UnshardedProjBytes), u0(kHlut3UnshardedProjBytes), d0(kHlut3UnshardedProjBytes);
    for (size_t i = 0; i < g0.size(); ++i) {
        g0[i] = uint8_t((i * 131u + 17u) & 0xFFu);
        u0[i] = uint8_t((i * 197u + 53u) & 0xFFu);
        d0[i] = uint8_t((i * 251u + 89u) & 0xFFu);
    }
    std::array<std::vector<uint8_t>, 3> tg, tu, td;
    uint8_t *pg[3], *pu[3], *pd[3];
    const uint8_t *cpg[3], *cpu[3], *cpd[3];
    for (int r = 0; r < 3; ++r) {
        tg[size_t(r)].resize(kHlut3Tp3ProjBytes);
        tu[size_t(r)].resize(kHlut3Tp3ProjBytes);
        td[size_t(r)].resize(kHlut3Tp3ProjBytes);
        pg[r] = tg[size_t(r)].data(); pu[r] = tu[size_t(r)].data(); pd[r] = td[size_t(r)].data();
        cpg[r] = pg[r]; cpu[r] = pu[r]; cpd[r] = pd[r];
    }
    shard_hlut3_full_to_tp3(g0.data(), u0.data(), d0.data(), pg, pu, pd);
    std::vector<uint8_t> g1(kHlut3UnshardedProjBytes), u1(kHlut3UnshardedProjBytes), d1(kHlut3UnshardedProjBytes);
    unshard_hlut3_tp3_to_full(cpg, cpu, cpd, g1.data(), u1.data(), d1.data());
    if (g0 != g1 || u0 != u1 || d0 != d1) return false;

    // 2. Verify 2-opt minimax balancer on a skewed Zipf-like 45-expert layer (18 vs 27 split)
    std::vector<LayerCand> lc;
    for (int i = 0; i < 45; ++i) {
        lc.push_back(LayerCand{uint64_t(100000.0 / std::pow(double(i + 2), 0.75)), i, i});
    }
    std::vector<uint8_t> mask;
    auto [s3, s4] = balance_layer_2opt(lc, 18, mask);
    const double imbal = 100.0 * std::fabs(double(s3) - double(s4)) / (0.5 * double(s3 + s4));
    if (imbal > 2.0) return false;

    return kHlut3UnshardedExpertBytes == 13824000u && kHlut3Tp3RankExpertBytes == 4608000u;
}

}  // namespace

int main(int argc, char** argv) {
    ::setenv("OMP_WAIT_POLICY", "PASSIVE", 1);
    ::setenv("GPU_MAX_HW_QUEUES", "8", 0);

    const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                     ? std::string(std::getenv("DSV41_ROOT"))
                                     : std::string(".");
    const std::string w92_dir  = (std::getenv("DSV41_OVERLAY_DIR") != nullptr)
                                     ? std::string(std::getenv("DSV41_OVERLAY_DIR"))
                                     : (root_env + "/wave9_2_overlays");
    const std::string w10_dir  = (std::getenv("DSV41_WAVE10_OVERLAY_DIR") != nullptr)
                                     ? std::string(std::getenv("DSV41_WAVE10_OVERLAY_DIR"))
                                     : (root_env + "/wave10_overlays");
    std::string mode              = "full-wave10";
    std::string mi50_root         = root_env + "/packed_mi50";
    std::string hist_path         = root_env + "/packed_rarebal_overlay_wave8_5/router_hist_wave8_5.txt";
    std::string perm_ov_path      = root_env + "/packed_rarebal_overlay_wave8_5/hot_experts_wave8_5.safetensors";
    std::string reuse_dev3_path   = w92_dir + "/wave9_2_dev3_hlut3.safetensors";
    std::string reuse_tp3_path    = w92_dir + "/wave9_2_tp3_hlut3.safetensors";
    std::string reuse_receipt     = w92_dir + "/wave9_2_receipt_manifest.txt";
    std::string cal_dir           = "/dev/shm/dsv41_wave10_cal";
    std::string raw_cache         = "/dev/shm/dsv41_wave10_raw";
    std::string dev3_out          = w10_dir + "/wave10_dev3_hlut3.safetensors";
    std::string dev4_out          = w10_dir + "/wave10_dev4_hlut3.safetensors";
    std::string tp3_out           = w10_dir + "/wave10_tp3_hlut3.safetensors";
    std::string plan_manifest     = "manifests/wave10_192g_tier_plan.txt";
    std::string receipt_manifest  = "manifests/wave10_receipt_manifest.txt";
    std::string sha256_manifest   = "manifests/SHA256SUMS_WAVE10.txt";
    std::string write_targets_out;
    int dev3_bb_target            = 720;
    int dev4_bb_target            = 1080;
    int tp3_bb_target             = 1160;
    int offload_per_layer_cap     = 56;
    int dev3_per_layer_cap        = 26;
    int dev4_per_layer_cap        = 36;
    int tp3_per_layer_cap         = 40;
    int gpu_workers               = 5;
    int fetch_threads             = 32;
    bool consume_bins             = true;
    bool link_check               = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--link-check") link_check = true;
        else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (a == "--mi50-root" && i + 1 < argc) mi50_root = argv[++i];
        else if (a == "--hist" && i + 1 < argc) hist_path = argv[++i];
        else if (a == "--perm-overlay" && i + 1 < argc) perm_ov_path = argv[++i];
        else if (a == "--reuse-dev3" && i + 1 < argc) reuse_dev3_path = argv[++i];
        else if (a == "--reuse-tp3" && i + 1 < argc) reuse_tp3_path = argv[++i];
        else if (a == "--reuse-receipt" && i + 1 < argc) reuse_receipt = argv[++i];
        else if (a == "--no-reuse") { reuse_dev3_path.clear(); reuse_tp3_path.clear(); reuse_receipt.clear(); }
        else if (a == "--cal-dir" && i + 1 < argc) cal_dir = argv[++i];
        else if (a == "--raw-cache" && i + 1 < argc) raw_cache = argv[++i];
        else if (a == "--dev3-out" && i + 1 < argc) dev3_out = argv[++i];
        else if (a == "--dev4-out" && i + 1 < argc) dev4_out = argv[++i];
        else if (a == "--tp3-out" && i + 1 < argc) tp3_out = argv[++i];
        else if (a == "--plan-manifest" && i + 1 < argc) plan_manifest = argv[++i];
        else if (a == "--receipt-manifest" && i + 1 < argc) receipt_manifest = argv[++i];
        else if (a == "--sha256-manifest" && i + 1 < argc) sha256_manifest = argv[++i];
        else if (a == "--write-harvest-targets" && i + 1 < argc) write_targets_out = argv[++i];
        else if (a == "--dev3-bb" && i + 1 < argc) dev3_bb_target = std::stoi(argv[++i]);
        else if (a == "--dev4-bb" && i + 1 < argc) dev4_bb_target = std::stoi(argv[++i]);
        else if (a == "--tp3-bb" && i + 1 < argc) tp3_bb_target = std::stoi(argv[++i]);
        else if (a == "--offload-per-layer-cap" && i + 1 < argc) offload_per_layer_cap = std::stoi(argv[++i]);
        else if (a == "--dev3-per-layer-cap" && i + 1 < argc) dev3_per_layer_cap = std::stoi(argv[++i]);
        else if (a == "--dev4-per-layer-cap" && i + 1 < argc) dev4_per_layer_cap = std::stoi(argv[++i]);
        else if (a == "--tp3-per-layer-cap" && i + 1 < argc) tp3_per_layer_cap = std::stoi(argv[++i]);
        else if (a == "--gpu-workers" && i + 1 < argc) gpu_workers = std::clamp(std::stoi(argv[++i]), 1, 8);
        else if (a == "--fetch-threads" && i + 1 < argc) fetch_threads = std::clamp(std::stoi(argv[++i]), 1, 64);
        else if (a == "--keep-bins") consume_bins = false;
        else if (a == "--consume-bins") consume_bins = true;
    }

    if (link_check) {
        if (!run_self_link_check()) {
            std::fprintf(stderr, "[WAVE10_BUILD] Self link check FAILED\n");
            return 1;
        }
        std::printf("[WAVE10_BUILD] Link, TP3<->Unsharded bit-exact roundtrip, & minimax balancer check OK\n");
        return 0;
    }

    const auto t_start = std::chrono::steady_clock::now();
    const auto lloyd_sol = solve_gaussian_lloyd_max_hlut3();

    Wave92ReuseBank reuse_bank;
    reuse_bank.load(reuse_dev3_path, reuse_tp3_path, reuse_receipt);

    std::array<std::vector<uint16_t>, 40> dev3_eids{}, dev4_eids{}, tp3_eids{};
    MinimaxPlanStats mm_stats{};
    std::vector<TargetExpert> plan = build_wave10_minimax_plan(
        hist_path, perm_ov_path,
        dev3_bb_target, dev4_bb_target, tp3_bb_target,
        offload_per_layer_cap, dev3_per_layer_cap, dev4_per_layer_cap, tp3_per_layer_cap,
        dev3_eids, dev4_eids, tp3_eids, mm_stats);

    int n_reused = 0, n_new = 0;
    for (auto& t : plan) {
        if (reuse_bank.can_reuse(t.bid, t.logical_eid)) {
            t.reused_w92 = true;
            ++n_reused;
        } else {
            ++n_new;
        }
    }

    write_tier_plan_manifest(plan_manifest, plan, dev3_eids, dev4_eids, tp3_eids, mm_stats);
    if (!write_targets_out.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(write_targets_out).parent_path(), ec);
        std::ofstream tf(write_targets_out, std::ios::trunc);
        tf << "# Wave 10 exact newly-promoted harvest targets (block logical_eid) [reused_w92=" << n_reused
           << ", new=" << n_new << "]\n";
        for (const auto& t : plan) {
            if (!t.reused_w92) tf << t.bid << " " << t.logical_eid << "\n";
        }
    }

    std::printf("[WAVE10_PLAN] SB-M5-P2-192G %zu promoted HLUT3 experts (384 MTP + %d dev3 + %d dev4 + %d TP3) -> %s\n"
                "  Reused verified Wave 9.2 RAREBAL/MTP : %d | Newly promoted to harvest/calibrate : %d\n"
                "  Minimax Switch Balance (Switch#1 dev3 vs Switch#2 dev4):\n"
                "    dev3_hits=%llu  dev4_hits=%llu  global_imbalance=%.3f%%  mean_layer=%.3f%%  max_layer=%.3f%%\n",
                plan.size(), dev3_bb_target, dev4_bb_target, tp3_bb_target, plan_manifest.c_str(),
                n_reused, n_new,
                static_cast<unsigned long long>(mm_stats.total_dev3_hits),
                static_cast<unsigned long long>(mm_stats.total_dev4_hits),
                mm_stats.global_imbalance_pct,
                mm_stats.mean_layer_imbalance_pct,
                mm_stats.max_layer_imbalance_pct);
    std::fflush(stdout);

    if (mode == "plan-only") {
        return (mm_stats.global_imbalance_pct < 3.0) ? 0 : 1;
    }
    if (mode != "full-wave10") {
        std::fprintf(stderr, "[WAVE10_BUILD] Unknown --mode '%s'\n", mode.c_str());
        return 1;
    }

    // Attach harvested .bin paths from cal_dir for newly promoted experts
    int bins_found = 0;
    for (auto& t : plan) {
        if (t.reused_w92) continue;
        char fn[256];
        std::snprintf(fn, sizeof(fn), "%s/b%02d_e%03d.bin", cal_dir.c_str(), t.bid, t.logical_eid);
        std::error_code ec;
        if (std::filesystem::is_regular_file(fn, ec)) {
            t.bin_path = fn;
            ++bins_found;
        }
    }
    std::printf("[WAVE10_BUILD] Found %d / %d harvested .bin activation files in %s\n", bins_found, n_new, cal_dir.c_str());
    std::fflush(stdout);

    std::array<std::vector<uint64_t>, 43> block_signs_h{}, block_signs_i{};
    for (int b = 0; b < 43; ++b) {
        load_block_wht_signs(mi50_root, b, block_signs_h[size_t(b)], block_signs_i[size_t(b)]);
    }

    // Pre-allocate lock-free per-block output tile buffers
    std::array<std::vector<uint8_t>, 43> dev3_gate_buf{}, dev3_up_buf{}, dev3_down_buf{};
    for (int m = 0; m < 3; ++m) {
        const size_t nb = 128ULL * kHlut3UnshardedProjBytes;
        dev3_gate_buf[size_t(40 + m)].resize(nb);
        dev3_up_buf[size_t(40 + m)].resize(nb);
        dev3_down_buf[size_t(40 + m)].resize(nb);
    }
    std::array<std::vector<uint8_t>, 40> dev4_gate_buf{}, dev4_up_buf{}, dev4_down_buf{};
    std::array<std::array<std::vector<uint8_t>, 40>, 3> tp3_gate_buf{}, tp3_up_buf{}, tp3_down_buf{};
    for (int l = 0; l < 40; ++l) {
        const size_t nb3 = dev3_eids[size_t(l)].size() * kHlut3UnshardedProjBytes;
        dev3_gate_buf[size_t(l)].resize(nb3);
        dev3_up_buf[size_t(l)].resize(nb3);
        dev3_down_buf[size_t(l)].resize(nb3);

        const size_t nb4 = dev4_eids[size_t(l)].size() * kHlut3UnshardedProjBytes;
        dev4_gate_buf[size_t(l)].resize(nb4);
        dev4_up_buf[size_t(l)].resize(nb4);
        dev4_down_buf[size_t(l)].resize(nb4);

        for (int r = 0; r < 3; ++r) {
            const size_t nbt = tp3_eids[size_t(l)].size() * kHlut3Tp3ProjBytes;
            tp3_gate_buf[size_t(r)][size_t(l)].resize(nbt);
            tp3_up_buf[size_t(r)][size_t(l)].resize(nbt);
            tp3_down_buf[size_t(r)][size_t(l)].resize(nbt);
        }
    }

    std::vector<Wave10ReceiptRow> receipts(plan.size());

    // Phase 1: Losslessly copy / re-shard all verified Wave 9.2 experts in parallel on CPU (< 2s!)
    std::vector<size_t> work_indices;
    work_indices.reserve(size_t(n_new));
    std::vector<size_t> reuse_indices;
    reuse_indices.reserve(size_t(n_reused));
    for (size_t i = 0; i < plan.size(); ++i) {
        if (plan[i].reused_w92) reuse_indices.push_back(i);
        else                    work_indices.push_back(i);
    }

    #pragma omp parallel for schedule(dynamic, 8)
    for (size_t ri = 0; ri < reuse_indices.size(); ++ri) {
        const size_t idx = reuse_indices[ri];
        const TargetExpert& te = plan[idx];
        std::vector<uint8_t> tmp_g(kHlut3UnshardedProjBytes), tmp_u(kHlut3UnshardedProjBytes), tmp_d(kHlut3UnshardedProjBytes);
        reuse_bank.extract_unsharded(te.bid, te.logical_eid, tmp_g.data(), tmp_u.data(), tmp_d.data());

        if (te.tier == TargetTier::kDev3Mtp || te.tier == TargetTier::kDev3Backbone) {
            const size_t off = size_t(te.local_slot) * kHlut3UnshardedProjBytes;
            std::memcpy(dev3_gate_buf[size_t(te.bid)].data() + off, tmp_g.data(), kHlut3UnshardedProjBytes);
            std::memcpy(dev3_up_buf[size_t(te.bid)].data()   + off, tmp_u.data(), kHlut3UnshardedProjBytes);
            std::memcpy(dev3_down_buf[size_t(te.bid)].data() + off, tmp_d.data(), kHlut3UnshardedProjBytes);
        } else if (te.tier == TargetTier::kDev4Backbone) {
            const size_t off = size_t(te.local_slot) * kHlut3UnshardedProjBytes;
            std::memcpy(dev4_gate_buf[size_t(te.bid)].data() + off, tmp_g.data(), kHlut3UnshardedProjBytes);
            std::memcpy(dev4_up_buf[size_t(te.bid)].data()   + off, tmp_u.data(), kHlut3UnshardedProjBytes);
            std::memcpy(dev4_down_buf[size_t(te.bid)].data() + off, tmp_d.data(), kHlut3UnshardedProjBytes);
        } else {
            const size_t off = size_t(te.local_slot) * kHlut3Tp3ProjBytes;
            uint8_t* pg[3] = { tp3_gate_buf[0][size_t(te.bid)].data() + off,
                               tp3_gate_buf[1][size_t(te.bid)].data() + off,
                               tp3_gate_buf[2][size_t(te.bid)].data() + off };
            uint8_t* pu[3] = { tp3_up_buf[0][size_t(te.bid)].data() + off,
                               tp3_up_buf[1][size_t(te.bid)].data() + off,
                               tp3_up_buf[2][size_t(te.bid)].data() + off };
            uint8_t* pd[3] = { tp3_down_buf[0][size_t(te.bid)].data() + off,
                               tp3_down_buf[1][size_t(te.bid)].data() + off,
                               tp3_down_buf[2][size_t(te.bid)].data() + off };
            shard_hlut3_full_to_tp3(tmp_g.data(), tmp_u.data(), tmp_d.data(), pg, pu, pd);
        }
        Wave10ReceiptRow rr = reuse_bank.entries.at({te.bid, te.logical_eid}).receipt;
        rr.tier = te.tier;
        rr.base_phys_eid = te.base_phys_eid;
        rr.local_slot = te.local_slot;
        rr.hist_hits = te.hist_hits;
        rr.reused_w92 = true;
        receipts[idx] = rr;
    }

    // Release Wave 9.2 mmaps before launching 5-GPU calibration
    reuse_bank.dev3_st.reset();
    reuse_bank.tp3_st.reset();

    const double t_reuse_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("[WAVE10_BUILD] Losslessly transferred %zu verified Wave 9.2 experts in %.2fs. Launching 5x MI50 solver for %zu new experts...\n",
                reuse_indices.size(), t_reuse_s, work_indices.size());
    std::fflush(stdout);

    // Phase 2: Solve all newly promoted experts across all 5x MI50 GPUs in parallel!
    int dev_cnt = 0;
    if (hipGetDeviceCount(&dev_cnt) != hipSuccess || dev_cnt <= 0) dev_cnt = 1;
    const int n_workers = std::min(gpu_workers, dev_cnt);

    std::printf("[WAVE10_BUILD] Active HIP GPU workers: %d / %d visible\n", n_workers, dev_cnt);
    std::fflush(stdout);

    HfShardRangeFetcher fetcher(raw_cache);
    AsyncTargetPrefetcher prefetcher(fetcher, plan, work_indices, fetch_threads);
    std::mutex log_mtx;
    std::atomic<int> solved_cnt{0}, rarebal_cnt{0}, fallback_hi_cnt{0};

    auto run_gpu_worker = [&](int wid) {
        (void)hipSetDevice(wid);
#ifdef _OPENMP
        omp_set_num_threads(std::max(4, 48 / n_workers));
#endif
        GpuRarebalAccel gpu_accel(wid);
        GpuHlut3LdlqHelper gpu_hlut3(wid);

        std::vector<float> w1_rot, w3_rot, w2_rot, scratch_sig;
        for (;;) {
            size_t idx = 0;
            RawExpertMxfp4 raw;
            std::string fetch_err;
            if (!prefetcher.take_next(idx, raw, fetch_err)) break;

            const auto t0 = std::chrono::steady_clock::now();
            const TargetExpert& te = plan[idx];
            if (!fetch_err.empty()) {
                throw std::runtime_error("Fetch failed at b" + std::to_string(te.bid) +
                                         " e" + std::to_string(te.logical_eid) + ": " + fetch_err);
            }

            const auto& signs_h = block_signs_h[size_t(te.bid)];
            const auto& signs_i = block_signs_i[size_t(te.bid)];

            const Hlut3QuantMatrix qg3_hi = dequant_mxfp4_and_quantize_hlut3_hi(
                raw.w1_w.data(), raw.w1_s.data(), DSV41_I, DSV41_H, signs_h, lloyd_sol, w1_rot);
            const Hlut3QuantMatrix qu3_hi = dequant_mxfp4_and_quantize_hlut3_hi(
                raw.w3_w.data(), raw.w3_s.data(), DSV41_I, DSV41_H, signs_h, lloyd_sol, w3_rot);
            const Hlut3QuantMatrix qd3_hi = dequant_mxfp4_and_quantize_hlut3_hi(
                raw.w2_w.data(), raw.w2_s.data(), DSV41_H, DSV41_I, signs_i, lloyd_sol, w2_rot);

            Hlut3QuantMatrix qg_sel = qg3_hi;
            Hlut3QuantMatrix qu_sel = qu3_hi;
            Hlut3QuantMatrix qd_sel = qd3_hi;

            Wave10ReceiptRow rr{};
            rr.tier          = te.tier;
            rr.bid           = te.bid;
            rr.logical_eid   = te.logical_eid;
            rr.base_phys_eid = te.base_phys_eid;
            rr.local_slot    = te.local_slot;
            rr.hist_hits     = te.hist_hits;
            rr.reused_w92    = false;

            CapturedExpertData cap;
            bool has_cap = false;
            if (!te.bin_path.empty()) {
                try {
                    cap = load_captured_expert_bin(te.bin_path);
                    has_cap = true;
                } catch (...) {
                    has_cap = false;
                }
                if (consume_bins) {
                    std::error_code rm_ec;
                    std::filesystem::remove(te.bin_path, rm_ec);
                }
            }

            constexpr int kMEval = 512;
            const int min_cal_rows = (te.tier == TargetTier::kDev3Mtp) ? 2048 : 10240;
            if (has_cap && cap.rows >= min_cal_rows + kMEval) {
                const int M_cal = cap.rows - kMEval;
                const int stride = cap.rows / kMEval;
                std::vector<float> X_cal(size_t(M_cal) * DSV41_H), X_eval(size_t(kMEval) * DSV41_H);
                std::vector<uint8_t> rare_cal(static_cast<size_t>(M_cal)), rare_eval(static_cast<size_t>(kMEval));
                int ic = 0, ie = 0;
                for (int r = 0; r < cap.rows; ++r) {
                    const bool hold = (ie < kMEval) && ((r % stride) == (stride - 1));
                    float* dst = hold ? (X_eval.data() + size_t(ie) * DSV41_H) : (X_cal.data() + size_t(ic) * DSV41_H);
                    if (hold) rare_eval[size_t(ie++)] = cap.rare[size_t(r)];
                    else      rare_cal[size_t(ic++)]  = cap.rare[size_t(r)];
                    const uint16_t* sb = cap.x_bf16.data() + size_t(r) * DSV41_H;
                    for (int j = 0; j < DSV41_H; ++j) dst[j] = bf16_to_f32(sb[j]);
                }
                std::vector<uint16_t>().swap(cap.x_bf16);

                Hlut2RarebalStats st_in{}, st_mid{};
                const auto R_up_in = gpu_accel.build_cholesky_rup(
                    X_cal.data(), M_cal, DSV41_H, signs_h, rare_cal.data(), &st_in);

                if (!R_up_in.empty()) {
                    gpu_hlut3.upload_rup(R_up_in, DSV41_H);
                    const Hlut3QuantMatrix qg3_rb = gpu_hlut3.apply_ldlq(qg3_hi, w1_rot.data(), R_up_in);
                    const Hlut3QuantMatrix qu3_rb = gpu_hlut3.apply_ldlq(qu3_hi, w3_rot.data(), R_up_in);

                    std::vector<float> wg3_rb_r, wu3_rb_r, X_cal_rot(size_t(M_cal) * DSV41_H), y_tmp, Z_cal;
                    unpack_hlut3_to_rot_f32(qg3_rb, wg3_rb_r);
                    unpack_hlut3_to_rot_f32(qu3_rb, wu3_rb_r);
                    wht128_rows(X_cal.data(), X_cal_rot.data(), M_cal, DSV41_H, signs_h);
                    (void)eval_rot_swiglu(X_cal_rot.data(), rare_cal.data(), M_cal,
                                          wg3_rb_r.data(), wu3_rb_r.data(), w2_rot.data(),
                                          signs_i, nullptr, y_tmp, &Z_cal, &gpu_accel);

                    const auto R_up_mid = gpu_accel.build_cholesky_rup(
                        Z_cal.data(), M_cal, DSV41_I, signs_i, rare_cal.data(), &st_mid);

                    if (!R_up_mid.empty()) {
                        gpu_hlut3.upload_rup(R_up_mid, DSV41_I);
                        const Hlut3QuantMatrix qd3_rb = gpu_hlut3.apply_ldlq(qd3_hi, w2_rot.data(), R_up_mid);

                        const Hlut2QuantMatrix qg2_hi = dequant_mxfp4_and_quantize_hlut2_mt(
                            raw.w1_w.data(), raw.w1_s.data(), DSV41_I, DSV41_H, signs_h, y_tmp, scratch_sig);
                        const Hlut2QuantMatrix qu2_hi = dequant_mxfp4_and_quantize_hlut2_mt(
                            raw.w3_w.data(), raw.w3_s.data(), DSV41_I, DSV41_H, signs_h, y_tmp, scratch_sig);
                        const Hlut2QuantMatrix qd2_hi = dequant_mxfp4_and_quantize_hlut2_mt(
                            raw.w2_w.data(), raw.w2_s.data(), DSV41_H, DSV41_I, signs_i, y_tmp, scratch_sig);

                        gpu_accel.upload_rup(R_up_in, DSV41_H);
                        const Hlut2QuantMatrix qg2_rb = gpu_accel.apply_ldlq_preuploaded_rup(qg2_hi, w1_rot.data(), R_up_in);
                        const Hlut2QuantMatrix qu2_rb = gpu_accel.apply_ldlq_preuploaded_rup(qu2_hi, w3_rot.data(), R_up_in);
                        gpu_accel.upload_rup(R_up_mid, DSV41_I);
                        const Hlut2QuantMatrix qd2_rb = gpu_accel.apply_ldlq_preuploaded_rup(qd2_hi, w2_rot.data(), R_up_mid);

                        std::vector<float> X_eval_rot(size_t(kMEval) * DSV41_H), Y_ref, Y_out;
                        wht128_rows(X_eval.data(), X_eval_rot.data(), kMEval, DSV41_H, signs_h);
                        (void)eval_rot_swiglu(X_eval_rot.data(), rare_eval.data(), kMEval,
                                              w1_rot.data(), w3_rot.data(), w2_rot.data(),
                                              signs_i, nullptr, Y_ref, nullptr, &gpu_accel);

                        std::vector<float> wg2_r, wu2_r, wd2_r, wg3_hi_r, wu3_hi_r, wd3_hi_r, wd3_rb_r;
                        dequantize_hlut2_to_rot_f32(qg2_rb, wg2_r);
                        dequantize_hlut2_to_rot_f32(qu2_rb, wu2_r);
                        dequantize_hlut2_to_rot_f32(qd2_rb, wd2_r);
                        unpack_hlut3_to_rot_f32(qg3_hi, wg3_hi_r);
                        unpack_hlut3_to_rot_f32(qu3_hi, wu3_hi_r);
                        unpack_hlut3_to_rot_f32(qd3_hi, wd3_hi_r);
                        unpack_hlut3_to_rot_f32(qd3_rb, wd3_rb_r);

                        const auto m2_rb = eval_rot_swiglu(X_eval_rot.data(), rare_eval.data(), kMEval,
                                                           wg2_r.data(), wu2_r.data(), wd2_r.data(),
                                                           signs_i, &Y_ref, Y_out, nullptr, &gpu_accel);
                        const auto m3_hi = eval_rot_swiglu(X_eval_rot.data(), rare_eval.data(), kMEval,
                                                           wg3_hi_r.data(), wu3_hi_r.data(), wd3_hi_r.data(),
                                                           signs_i, &Y_ref, Y_out, nullptr, &gpu_accel);
                        const auto m3_rb = eval_rot_swiglu(X_eval_rot.data(), rare_eval.data(), kMEval,
                                                           wg3_rb_r.data(), wu3_rb_r.data(), wd3_rb_r.data(),
                                                           signs_i, &Y_ref, Y_out, nullptr, &gpu_accel);

                        const bool pass_dual =
                            std::isfinite(m3_rb.cos_all) && std::isfinite(m3_rb.mse_rare) &&
                            (m3_rb.cos_all > m3_hi.cos_all) && (m3_rb.mse_rare < m3_hi.mse_rare) &&
                            (m3_rb.cos_all > m2_rb.cos_all) && (m3_rb.snr_db > m2_rb.snr_db) &&
                            (m3_rb.mse_rare < m2_rb.mse_rare);

                        rr.m_cal      = M_cal;
                        rr.m_eval     = kMEval;
                        rr.rare_cal   = st_in.m_rare;
                        rr.rare_eval  = m3_rb.rare_cnt;
                        rr.lam_in     = st_in.lambda;
                        rr.lam_mid    = st_mid.lambda;
                        rr.m2_rb      = m2_rb;
                        rr.m3_hi      = m3_hi;
                        rr.m3_rb      = m3_rb;
                        rr.used_rarebal = pass_dual;

                        if (pass_dual) {
                            qg_sel = qg3_rb;
                            qu_sel = qu3_rb;
                            qd_sel = qd3_rb;
                        }
                    }
                }
            }

            // Write selected tiles lock-free into exact tier slot
            if (te.tier == TargetTier::kTp3Backbone) {
                const auto sh = pack_hlut3_expert_tp3_sharded(qg_sel, qu_sel, qd_sel);
                const size_t off = size_t(te.local_slot) * kHlut3Tp3ProjBytes;
                for (int r = 0; r < 3; ++r) {
                    std::memcpy(tp3_gate_buf[size_t(r)][size_t(te.bid)].data() + off, sh[size_t(r)].gate_tiles.data(), kHlut3Tp3ProjBytes);
                    std::memcpy(tp3_up_buf[size_t(r)][size_t(te.bid)].data()   + off, sh[size_t(r)].up_tiles.data(),   kHlut3Tp3ProjBytes);
                    std::memcpy(tp3_down_buf[size_t(r)][size_t(te.bid)].data() + off, sh[size_t(r)].down_tiles.data(), kHlut3Tp3ProjBytes);
                }
            } else if (te.tier == TargetTier::kDev4Backbone) {
                const size_t off = size_t(te.local_slot) * kHlut3UnshardedProjBytes;
                std::memcpy(dev4_gate_buf[size_t(te.bid)].data() + off, qg_sel.tiles.data(), kHlut3UnshardedProjBytes);
                std::memcpy(dev4_up_buf[size_t(te.bid)].data()   + off, qu_sel.tiles.data(), kHlut3UnshardedProjBytes);
                std::memcpy(dev4_down_buf[size_t(te.bid)].data() + off, qd_sel.tiles.data(), kHlut3UnshardedProjBytes);
            } else {
                const size_t off = size_t(te.local_slot) * kHlut3UnshardedProjBytes;
                std::memcpy(dev3_gate_buf[size_t(te.bid)].data() + off, qg_sel.tiles.data(), kHlut3UnshardedProjBytes);
                std::memcpy(dev3_up_buf[size_t(te.bid)].data()   + off, qu_sel.tiles.data(), kHlut3UnshardedProjBytes);
                std::memcpy(dev3_down_buf[size_t(te.bid)].data() + off, qd_sel.tiles.data(), kHlut3UnshardedProjBytes);
            }

            receipts[idx] = rr;
            const int cur = ++solved_cnt;
            if (rr.used_rarebal) ++rarebal_cnt;
            else                 ++fallback_hi_cnt;

            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if ((cur % 16 == 0) || cur == int(work_indices.size())) {
                std::lock_guard<std::mutex> lk(log_mtx);
                const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
                std::printf("[HLUT3_W10 %4d/%4zu gpu%d] %-7s b%02d e%03d | M_cal=%d | cos2=%.5f cos3=%.5f | SNR2=%.2f SNR3=%.2f (+%.2fdB) | %s (%.2fs, tot=%.0fs)\n",
                            cur, work_indices.size(), wid, tier_name(te.tier), te.bid, te.logical_eid, rr.m_cal,
                            rr.m2_rb.cos_all, rr.used_rarebal ? rr.m3_rb.cos_all : rr.m3_hi.cos_all,
                            rr.m2_rb.snr_db, rr.used_rarebal ? rr.m3_rb.snr_db : rr.m3_hi.snr_db,
                            (rr.used_rarebal ? rr.m3_rb.snr_db : rr.m3_hi.snr_db) - rr.m2_rb.snr_db,
                            rr.used_rarebal ? "EMITTED_RAREBAL" : "FALLBACK_HLUT3_HI", dt, el);
                std::fflush(stdout);
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(size_t(n_workers));
    for (int w = 0; w < n_workers; ++w) pool.emplace_back(run_gpu_worker, w);
    for (auto& th : pool) th.join();
    prefetcher.stop();

    std::error_code ec;
    if (raw_cache.rfind("/dev/shm/", 0) == 0) std::filesystem::remove_all(raw_cache, ec);
    if (consume_bins && cal_dir.rfind("/dev/shm/", 0) == 0) std::filesystem::remove_all(cal_dir, ec);

    // Emit wave10_dev3_hlut3.safetensors (14.21 GiB: 384 MTP + 720 minimax-balanced backbone)
    {
        std::filesystem::create_directories(std::filesystem::path(dev3_out).parent_path(), ec);
        StreamingSafetensorsWriter w;
        for (int b = 40; b < 43; ++b) {
            const std::string pfx = "dev3.b" + std::to_string(b) + ".";
            w.add_owned(pfx + "gate_tiles", "U8", {128, 72, 160, 400}, std::move(dev3_gate_buf[size_t(b)]));
            w.add_owned(pfx + "up_tiles",   "U8", {128, 72, 160, 400}, std::move(dev3_up_buf[size_t(b)]));
            w.add_owned(pfx + "down_tiles", "U8", {128, 160, 72, 400}, std::move(dev3_down_buf[size_t(b)]));
        }
        for (int l = 0; l < 40; ++l) {
            const int64_t ne = int64_t(dev3_eids[size_t(l)].size());
            if (ne <= 0) continue;
            const std::string pfx = "dev3.b" + std::to_string(l) + ".";
            std::vector<uint8_t> eid_bytes(size_t(ne) * sizeof(uint16_t));
            std::memcpy(eid_bytes.data(), dev3_eids[size_t(l)].data(), eid_bytes.size());
            w.add_owned(pfx + "logical_eids", "U16", {ne}, std::move(eid_bytes));
            w.add_owned(pfx + "gate_tiles",   "U8",  {ne, 72, 160, 400}, std::move(dev3_gate_buf[size_t(l)]));
            w.add_owned(pfx + "up_tiles",     "U8",  {ne, 72, 160, 400}, std::move(dev3_up_buf[size_t(l)]));
            w.add_owned(pfx + "down_tiles",   "U8",  {ne, 160, 72, 400}, std::move(dev3_down_buf[size_t(l)]));
        }
        w.write_atomic(dev3_out);
    }

    // Emit wave10_dev4_hlut3.safetensors (13.90 GiB: 1,080 minimax-balanced backbone)
    {
        std::filesystem::create_directories(std::filesystem::path(dev4_out).parent_path(), ec);
        StreamingSafetensorsWriter w;
        for (int l = 0; l < 40; ++l) {
            const int64_t ne = int64_t(dev4_eids[size_t(l)].size());
            if (ne <= 0) continue;
            const std::string pfx = "dev4.b" + std::to_string(l) + ".";
            std::vector<uint8_t> eid_bytes(size_t(ne) * sizeof(uint16_t));
            std::memcpy(eid_bytes.data(), dev4_eids[size_t(l)].data(), eid_bytes.size());
            w.add_owned(pfx + "logical_eids", "U16", {ne}, std::move(eid_bytes));
            w.add_owned(pfx + "gate_tiles",   "U8",  {ne, 72, 160, 400}, std::move(dev4_gate_buf[size_t(l)]));
            w.add_owned(pfx + "up_tiles",     "U8",  {ne, 72, 160, 400}, std::move(dev4_up_buf[size_t(l)]));
            w.add_owned(pfx + "down_tiles",   "U8",  {ne, 160, 72, 400}, std::move(dev4_down_buf[size_t(l)]));
        }
        w.write_atomic(dev4_out);
    }

    // Emit wave10_tp3_hlut3.safetensors (14.94 GiB: 1,160 sharded TP3 backbone)
    {
        std::filesystem::create_directories(std::filesystem::path(tp3_out).parent_path(), ec);
        StreamingSafetensorsWriter w;
        for (int l = 0; l < 40; ++l) {
            const int64_t ne = int64_t(tp3_eids[size_t(l)].size());
            if (ne <= 0) continue;
            std::vector<uint8_t> eid_bytes(size_t(ne) * sizeof(uint16_t));
            std::memcpy(eid_bytes.data(), tp3_eids[size_t(l)].data(), eid_bytes.size());
            w.add_owned("tp3.b" + std::to_string(l) + ".logical_eids", "U16", {ne}, std::move(eid_bytes));
            for (int r = 0; r < 3; ++r) {
                const std::string pfx = "tp3.tp" + std::to_string(r) + ".b" + std::to_string(l) + ".";
                w.add_owned(pfx + "gate_tiles", "U8", {ne, 24, 160, 400}, std::move(tp3_gate_buf[size_t(r)][size_t(l)]));
                w.add_owned(pfx + "up_tiles",   "U8", {ne, 24, 160, 400}, std::move(tp3_up_buf[size_t(r)][size_t(l)]));
                w.add_owned(pfx + "down_tiles", "U8", {ne, 160, 24, 400}, std::move(tp3_down_buf[size_t(r)][size_t(l)]));
            }
        }
        w.write_atomic(tp3_out);
    }

    // Verify dual-receipt invariants & write manifests
    int dual_violations = 0;
    int rb_mtp = 0, rb_d3 = 0, rb_d4 = 0, rb_tp3 = 0, rb_reused = 0, rb_new = 0;
    double sum_cos2 = 0.0, sum_cos3_hi = 0.0, sum_cos3_rb = 0.0;
    double sum_snr2 = 0.0, sum_snr3_hi = 0.0, sum_snr3_rb = 0.0;
    for (const auto& r : receipts) {
        if (!r.used_rarebal) continue;
        if (!(r.m3_rb.cos_all > r.m3_hi.cos_all && r.m3_rb.mse_rare < r.m3_hi.mse_rare &&
              r.m3_rb.cos_all > r.m2_rb.cos_all && r.m3_rb.snr_db > r.m2_rb.snr_db &&
              r.m3_rb.mse_rare < r.m2_rb.mse_rare)) {
            ++dual_violations;
        }
        sum_cos2    += r.m2_rb.cos_all;
        sum_cos3_hi += r.m3_hi.cos_all;
        sum_cos3_rb += r.m3_rb.cos_all;
        sum_snr2    += r.m2_rb.snr_db;
        sum_snr3_hi += r.m3_hi.snr_db;
        sum_snr3_rb += r.m3_rb.snr_db;
        if (r.tier == TargetTier::kDev3Mtp) ++rb_mtp;
        else if (r.tier == TargetTier::kDev3Backbone) ++rb_d3;
        else if (r.tier == TargetTier::kDev4Backbone) ++rb_d4;
        else ++rb_tp3;
        if (r.reused_w92) ++rb_reused; else ++rb_new;
    }
    const int tot_rb = rb_mtp + rb_d3 + rb_d4 + rb_tp3;

    {
        std::filesystem::create_directories(std::filesystem::path(receipt_manifest).parent_path(), ec);
        std::ofstream mf(receipt_manifest, std::ios::trunc);
        mf << "# Wave 10 SB-M5-P2-192G WHT128-HLUT3 Dual-Receipt Calibration Manifest\n"
           << "# total_hlut3_experts=" << receipts.size()
           << " emitted_rarebal_hlut3=" << tot_rb
           << " (mtp=" << rb_mtp << ", dev3_bb=" << rb_d3 << ", dev4_bb=" << rb_d4 << ", tp3_bb=" << rb_tp3
           << "; reused_w92=" << rb_reused << ", new_w10=" << rb_new << ")"
           << " fallback_hlut3_hi=" << (int(receipts.size()) - tot_rb)
           << " dual_receipt_violations=" << dual_violations
           << " mean_cos_hlut2_rb=" << std::fixed << std::setprecision(5) << (tot_rb > 0 ? sum_cos2 / tot_rb : 0.0)
           << " mean_cos_hlut3_hi=" << (tot_rb > 0 ? sum_cos3_hi / tot_rb : 0.0)
           << " mean_cos_hlut3_rb=" << (tot_rb > 0 ? sum_cos3_rb / tot_rb : 0.0)
           << " mean_snr_hlut2_rb_db=" << (tot_rb > 0 ? sum_snr2 / tot_rb : 0.0)
           << " mean_snr_hlut3_hi_db=" << (tot_rb > 0 ? sum_snr3_hi / tot_rb : 0.0)
           << " mean_snr_hlut3_rb_db=" << (tot_rb > 0 ? sum_snr3_rb / tot_rb : 0.0)
           << " mean_gain_vs_hlut2_rb_db=" << (tot_rb > 0 ? (sum_snr3_rb - sum_snr2) / tot_rb : 0.0)
           << " mean_gain_vs_hlut3_hi_db=" << (tot_rb > 0 ? (sum_snr3_rb - sum_snr3_hi) / tot_rb : 0.0) << "\n"
           << "# tier block logical_eid base_phys_eid slot hits M_cal M_eval rare_cal rare_eval lam_in lam_mid cos_hlut2_rb cos_hlut3_hi cos_hlut3_rb snr_hlut2_rb snr_hlut3_hi snr_hlut3_rb rare_mse_hlut2_rb rare_mse_hlut3_hi rare_mse_hlut3_rb status\n";
        for (const auto& r : receipts) {
            const char* st_str = r.used_rarebal
                ? (r.reused_w92 ? "REUSED_W92_RAREBAL" : "EMITTED_RAREBAL")
                : (r.reused_w92 ? "REUSED_W92_HI" : "FALLBACK_HLUT3_HI");
            mf << tier_name(r.tier) << " " << r.bid << " " << r.logical_eid << " " << r.base_phys_eid << " "
               << r.local_slot << " " << r.hist_hits << " " << r.m_cal << " " << r.m_eval << " "
               << r.rare_cal << " " << r.rare_eval << " " << std::fixed << std::setprecision(5)
               << r.lam_in << " " << r.lam_mid << " "
               << r.m2_rb.cos_all << " " << r.m3_hi.cos_all << " " << r.m3_rb.cos_all << " "
               << r.m2_rb.snr_db  << " " << r.m3_hi.snr_db  << " " << r.m3_rb.snr_db  << " "
               << std::scientific << r.m2_rb.mse_rare << " " << r.m3_hi.mse_rare << " " << r.m3_rb.mse_rare << " "
               << st_str << "\n";
        }
    }

    const std::string sha_dev3 = sha256_file_hex(dev3_out);
    const std::string sha_dev4 = sha256_file_hex(dev4_out);
    const std::string sha_tp3  = sha256_file_hex(tp3_out);
    {
        std::filesystem::create_directories(std::filesystem::path(sha256_manifest).parent_path(), ec);
        std::ofstream sf(sha256_manifest, std::ios::trunc);
        sf << sha_dev3 << "  " << std::filesystem::path(dev3_out).filename().string() << "\n"
           << sha_dev4 << "  " << std::filesystem::path(dev4_out).filename().string() << "\n"
           << sha_tp3  << "  " << std::filesystem::path(tp3_out).filename().string() << "\n";
    }

    const double tot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("[WAVE10_BUILD] DONE in %.2fs | total=%zu | RAREBAL=%d (mtp=%d, dev3=%d, dev4=%d, tp3=%d; reused=%d, new=%d) | HI_fallback=%d | dual_violations=%d | mean_gain_vs_HLUT2=+%.2f dB\n"
                "  dev3_overlay : %s (sha256=%s)\n"
                "  dev4_overlay : %s (sha256=%s)\n"
                "  tp3_overlay  : %s (sha256=%s)\n",
                tot_s, receipts.size(), tot_rb, rb_mtp, rb_d3, rb_d4, rb_tp3, rb_reused, rb_new,
                int(receipts.size()) - tot_rb, dual_violations,
                (tot_rb > 0 ? (sum_snr3_rb - sum_snr2) / tot_rb : 0.0),
                dev3_out.c_str(), sha_dev3.c_str(),
                dev4_out.c_str(), sha_dev4.c_str(),
                tp3_out.c_str(), sha_tp3.c_str());
    std::fflush(stdout);
    return (dual_violations == 0) ? 0 : 2;
}
