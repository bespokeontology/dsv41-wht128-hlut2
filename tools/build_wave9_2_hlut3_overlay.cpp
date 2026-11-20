// Wave 9.2 Deliverable: 4x MI50 Parallel rocBLAS TARQ RAREBAL-LDLQ + WHT128-HLUT3 Overlay Builder
// Modes:
//   --link-check       : Self-test HLUT3 pack + Left-Looking 128-col LDLQ equivalence
//   --mode plan-only   : Deterministic 1,704-expert partitioning -> manifests/wave9_2_tier_plan.txt
//                        + optional <cal_dir>/wave9_2_harvest_targets.txt
//   --mode fast-hi-mtp : Fast 384-expert H=I WHT128-HLUT3 MTP overlay (< 5 min) for Phase 2 silicon gate
//   --mode full-option-b: 4x MI50 parallel TARQ RAREBAL-LDLQ calibration across 1,704 experts:
//                          * 384 MTP experts (blocks 40..42) -> unsharded HLUT3 on dev3 (85:00.0)
//                          * 720 hottest MI50 backbone experts (layers 0..39) -> unsharded HLUT3 on dev3
//                          * 600 next warm-hot MI50 backbone experts (layers 0..39) -> TP3-sharded HLUT3
//                        Enforces strict Dual-Receipt acceptance (HLUT3 RAREBAL must strictly beat BOTH
//                        HLUT3 H=I and HLUT2 RAREBAL on held-out SwiGLU cosine & rare-token MSE; sparse
//                        cold-tail falls back to HLUT3 H=I with dual_receipt_violations == 0).
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
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace dsv41;

namespace {

// Pure C++17 FIPS 180-4 SHA-256 for manifests/SHA256SUMS_WAVE9_2.txt
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
    kTp3Backbone = 2
};

struct TargetExpert {
    TargetTier tier = TargetTier::kDev3Mtp;
    int bid = 0;
    int logical_eid = 0;
    int base_phys_eid = 0;
    int local_slot = 0; // index within block's dev3 or tp3 tensor
    uint64_t hist_hits = 0;
    std::string bin_path;
};

struct Wave92ReceiptRow {
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
    bool used_rarebal = false; // true = EMITTED_RAREBAL, false = FALLBACK_HLUT3_HI
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

// Multi-threaded OpenMP + AVX2 fused MXFP4 dequant + WHT128 -> w_rot + H=I Hlut3QuantMatrix seed
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

// Left-Looking 128-Column Block-Causal HLUT3 LDLQ on MI50 rocBLAS (mathematically identical to apply_hlut3_ldlq_with_rup)
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

    AsyncTargetPrefetcher(HfShardRangeFetcher& fetcher, const std::vector<TargetExpert>& items, int n_threads = 32)
        : fetcher_(fetcher), items_(items)
    {
        for (auto& s : slots_) s.idx = ~size_t(0);
        if (items_.empty()) return;
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

    bool take_next(size_t& out_idx, RawExpertMxfp4& out_raw, std::string& out_err) {
        std::lock_guard<std::mutex> ord_lk(take_mtx_);
        size_t idx = 0;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            if (stop_ || next_consume_ >= items_.size()) return false;
            idx = next_consume_++;
            Slot& s = slots_[idx % kWin];
            cv_cons_.wait(lk, [&]() { return stop_ || (s.ready && s.idx == idx); });
            if (stop_ && !(s.ready && s.idx == idx)) return false;
            out_raw = std::move(s.raw);
            out_err = std::move(s.err);
            s.ready = false;
            s.idx = ~size_t(0);
            s.err.clear();
            if (idx + 1 > consumed_) consumed_ = idx + 1;
        }
        cv_prod_.notify_all();
        out_idx = idx;
        return true;
    }

private:
    struct Slot {
        size_t idx = ~size_t(0);
        bool ready = false;
        RawExpertMxfp4 raw{};
        std::string err{};
    };
    void worker_loop() {
        for (;;) {
            size_t my_idx = ~size_t(0);
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_prod_.wait(lk, [&]() {
                    return stop_ || (next_issue_ < items_.size() && next_issue_ < consumed_ + kWin &&
                                     !slots_[next_issue_ % kWin].ready);
                });
                if (stop_ || next_issue_ >= items_.size()) return;
                my_idx = next_issue_++;
            }
            RawExpertMxfp4 fetched{};
            std::string err_msg;
            try {
                fetched = fetcher_.fetch_expert(items_[my_idx].bid, items_[my_idx].logical_eid);
            } catch (const std::exception& ex) {
                err_msg = ex.what();
            }
            {
                std::lock_guard<std::mutex> lk(mtx_);
                Slot& s = slots_[my_idx % kWin];
                s.idx = my_idx;
                s.raw = std::move(fetched);
                s.err = std::move(err_msg);
                s.ready = true;
            }
            cv_cons_.notify_all();
        }
    }

    HfShardRangeFetcher& fetcher_;
    const std::vector<TargetExpert>& items_;
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

// Builds deterministic 1,704-expert partition from router_hist_wave8_5.txt + expert_perm_u16
std::vector<TargetExpert> build_deterministic_tier_plan(
    const std::string& hist_path,
    const std::string& perm_ov_path,
    int dev3_bb_target,
    int tp3_bb_target,
    int dev3_per_layer_cap,
    int tp3_per_layer_cap,
    std::array<std::vector<uint16_t>, 40>& out_dev3_eids,
    std::array<std::vector<uint16_t>, 40>& out_tp3_eids)
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

    // Collect all MI50-resident backbone experts (perm[l][le] < 216), excluding P150A [216..383]!
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

    for (int l = 0; l < 40; ++l) {
        out_dev3_eids[size_t(l)].clear();
        out_tp3_eids[size_t(l)].clear();
    }
    std::array<std::array<uint8_t, 384>, 40> assigned{};

    int got_dev3 = 0;
    for (const auto& c : mi50_cands) {
        if (got_dev3 >= dev3_bb_target) break;
        if (int(out_dev3_eids[size_t(c.l)].size()) >= dev3_per_layer_cap) continue;
        out_dev3_eids[size_t(c.l)].push_back(uint16_t(c.le));
        assigned[size_t(c.l)][size_t(c.le)] = 1u;
        ++got_dev3;
    }

    int got_tp3 = 0;
    for (const auto& c : mi50_cands) {
        if (got_tp3 >= tp3_bb_target) break;
        if (assigned[size_t(c.l)][size_t(c.le)] != 0u) continue;
        if (int(out_tp3_eids[size_t(c.l)].size()) >= tp3_per_layer_cap) continue;
        out_tp3_eids[size_t(c.l)].push_back(uint16_t(c.le));
        assigned[size_t(c.l)][size_t(c.le)] = 2u;
        ++got_tp3;
    }

    if (got_dev3 != dev3_bb_target || got_tp3 != tp3_bb_target) {
        throw std::runtime_error("Failed to satisfy dev3/tp3 backbone targets under per-layer caps");
    }

    // Canonical ascending logical_eid order within each layer
    for (int l = 0; l < 40; ++l) {
        std::sort(out_dev3_eids[size_t(l)].begin(), out_dev3_eids[size_t(l)].end());
        std::sort(out_tp3_eids[size_t(l)].begin(), out_tp3_eids[size_t(l)].end());
    }

    std::vector<TargetExpert> plan;
    plan.reserve(size_t(384 + dev3_bb_target + tp3_bb_target));

    // 1. DEV3_MTP_384 (blocks 40..42, 0..127)
    for (int b = 40; b < 43; ++b) {
        for (int e = 0; e < 128; ++e) {
            plan.push_back(TargetExpert{TargetTier::kDev3Mtp, b, e, e, e, hits[size_t(b)][size_t(e)], ""});
        }
    }
    // 2. DEV3_BACKBONE_720 (layers 0..39)
    for (int l = 0; l < 40; ++l) {
        for (size_t j = 0; j < out_dev3_eids[size_t(l)].size(); ++j) {
            const int le = int(out_dev3_eids[size_t(l)][j]);
            plan.push_back(TargetExpert{TargetTier::kDev3Backbone, l, le, int(perm[size_t(l)][size_t(le)]),
                                        int(j), hits[size_t(l)][size_t(le)], ""});
        }
    }
    // 3. TP3_BACKBONE_600 (layers 0..39)
    for (int l = 0; l < 40; ++l) {
        for (size_t j = 0; j < out_tp3_eids[size_t(l)].size(); ++j) {
            const int le = int(out_tp3_eids[size_t(l)][j]);
            plan.push_back(TargetExpert{TargetTier::kTp3Backbone, l, le, int(perm[size_t(l)][size_t(le)]),
                                        int(j), hits[size_t(l)][size_t(le)], ""});
        }
    }
    return plan;
}

void write_tier_plan_manifest(
    const std::string& manifest_path,
    const std::vector<TargetExpert>& plan,
    const std::array<std::vector<uint16_t>, 40>& dev3_eids,
    const std::array<std::vector<uint16_t>, 40>& tp3_eids)
{
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(manifest_path).parent_path(), ec);
    std::ofstream f(manifest_path, std::ios::trunc);
    int n_mtp = 0, n_dev3 = 0, n_tp3 = 0;
    for (const auto& t : plan) {
        if (t.tier == TargetTier::kDev3Mtp) ++n_mtp;
        else if (t.tier == TargetTier::kDev3Backbone) ++n_dev3;
        else ++n_tp3;
    }
    f << "# Wave 9.2 SB-M4-P2-176G Deterministic Tier Partition Plan\n"
      << "# total_promoted_hlut3=" << plan.size()
      << " dev3_mtp=" << n_mtp
      << " dev3_backbone=" << n_dev3
      << " tp3_backbone=" << n_tp3 << "\n"
      << "# per_layer_summary: layer n_dev3 n_tp3 n_compacted_hlut2\n";
    for (int l = 0; l < 40; ++l) {
        const int nd = int(dev3_eids[size_t(l)].size());
        const int nt = int(tp3_eids[size_t(l)].size());
        f << "# L" << std::setw(2) << std::setfill('0') << l << std::setfill(' ')
          << " dev3=" << nd << " tp3=" << nt << " hlut2=" << (216 - nd - nt) << "\n";
    }
    f << "# tier block logical_eid base_phys_eid local_slot router_hits_wave8_5\n";
    for (const auto& t : plan) {
        const char* tname = (t.tier == TargetTier::kDev3Mtp) ? "DEV3_MTP"
                          : ((t.tier == TargetTier::kDev3Backbone) ? "DEV3_BB" : "TP3_BB");
        f << tname << " " << t.bid << " " << t.logical_eid << " " << t.base_phys_eid << " "
          << t.local_slot << " " << t.hist_hits << "\n";
    }
}

bool run_self_link_check() {
    const auto sol = solve_gaussian_lloyd_max_hlut3();
    if (!sol.converged || sol.sqnr_db < 14.6) return false;
    std::vector<uint64_t> signs(4, 0xA5A5A5A55A5A5A5AULL);
    std::vector<float> W(32 * 256, 0.0f);
    for (size_t i = 0; i < W.size(); ++i) W[i] = float(std::sin(double(i + 1) * 0.137)) * 0.05f;
    const auto q = quantize_hlut3(W.data(), 32, 256, signs);
    return q.tiles.size() == 8u && kHlut3UnshardedExpertBytes == 13824000u &&
           kHlut3Tp3RankExpertBytes == 4608000u;
}

}  // namespace

int main(int argc, char** argv) {
    ::setenv("OMP_WAIT_POLICY", "PASSIVE", 1);
    ::setenv("GPU_MAX_HW_QUEUES", "8", 0);

    const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                     ? std::string(std::getenv("DSV41_ROOT"))
                                     : std::string(".");
    std::string mode              = "full-option-b";
    std::string mi50_root         = root_env + "/packed_mi50";
    std::string hist_path         = root_env + "/packed_rarebal_overlay_wave8_5/router_hist_wave8_5.txt";
    std::string perm_ov_path      = root_env + "/packed_rarebal_overlay_wave8_5/hot_experts_wave8_5.safetensors";
    std::string cal_dir           = "/dev/shm/dsv41_wave9_2_cal";
    std::string raw_cache         = "/dev/shm/dsv41_wave9_2_raw";
    std::string dev3_out          = root_env + "/wave9_2_dev3_hlut3.safetensors";
    std::string tp3_out           = root_env + "/wave9_2_tp3_hlut3.safetensors";
    std::string plan_manifest     = "manifests/wave9_2_tier_plan.txt";
    std::string receipt_manifest  = "manifests/wave9_2_receipt_manifest.txt";
    std::string sha256_manifest   = "manifests/SHA256SUMS_WAVE9_2.txt";
    std::string write_targets_out;
    int dev3_bb_target            = 720;
    int tp3_bb_target             = 600;
    int dev3_per_layer_cap        = 22;
    int tp3_per_layer_cap         = 20;
    int gpu_workers               = 4;
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
        else if (a == "--cal-dir" && i + 1 < argc) cal_dir = argv[++i];
        else if (a == "--raw-cache" && i + 1 < argc) raw_cache = argv[++i];
        else if (a == "--dev3-out" && i + 1 < argc) dev3_out = argv[++i];
        else if (a == "--tp3-out" && i + 1 < argc) tp3_out = argv[++i];
        else if (a == "--plan-manifest" && i + 1 < argc) plan_manifest = argv[++i];
        else if (a == "--receipt-manifest" && i + 1 < argc) receipt_manifest = argv[++i];
        else if (a == "--sha256-manifest" && i + 1 < argc) sha256_manifest = argv[++i];
        else if (a == "--write-harvest-targets" && i + 1 < argc) write_targets_out = argv[++i];
        else if (a == "--dev3-bb" && i + 1 < argc) dev3_bb_target = std::stoi(argv[++i]);
        else if (a == "--tp3-bb" && i + 1 < argc) tp3_bb_target = std::stoi(argv[++i]);
        else if (a == "--dev3-per-layer-cap" && i + 1 < argc) dev3_per_layer_cap = std::stoi(argv[++i]);
        else if (a == "--tp3-per-layer-cap" && i + 1 < argc) tp3_per_layer_cap = std::stoi(argv[++i]);
        else if (a == "--gpu-workers" && i + 1 < argc) gpu_workers = std::clamp(std::stoi(argv[++i]), 1, 8);
        else if (a == "--fetch-threads" && i + 1 < argc) fetch_threads = std::clamp(std::stoi(argv[++i]), 1, 64);
        else if (a == "--keep-bins") consume_bins = false;
        else if (a == "--consume-bins") consume_bins = true;
    }

    if (link_check) {
        if (!run_self_link_check()) {
            std::fprintf(stderr, "[WAVE9_2_BUILD] Self link check FAILED\n");
            return 1;
        }
        std::printf("[WAVE9_2_BUILD] Link & WHT128-HLUT3 geometry check OK\n");
        return 0;
    }

    const auto t_start = std::chrono::steady_clock::now();
    const auto lloyd_sol = solve_gaussian_lloyd_max_hlut3();

    // ---- MODE 1: FAST-HI-MTP (384 MTP experts at H=I WHT128-HLUT3 for Phase 2 silicon bring-up) ----
    if (mode == "fast-hi-mtp") {
        std::vector<TargetExpert> mtp_items;
        mtp_items.reserve(384u);
        for (int b = 40; b < 43; ++b) {
            for (int e = 0; e < 128; ++e) {
                mtp_items.push_back(TargetExpert{TargetTier::kDev3Mtp, b, e, e, e, 0ULL, ""});
            }
        }
        std::array<std::vector<uint64_t>, 3> mtp_sh{}, mtp_si{};
        for (int m = 0; m < 3; ++m) {
            load_block_wht_signs(mi50_root, 40 + m, mtp_sh[size_t(m)], mtp_si[size_t(m)]);
        }

        const size_t block_proj_bytes = 128ULL * kHlut3UnshardedProjBytes; // 589,824,000 B
        std::array<std::vector<uint8_t>, 3> gate_buf{}, up_buf{}, down_buf{};
        for (int m = 0; m < 3; ++m) {
            gate_buf[size_t(m)].resize(block_proj_bytes);
            up_buf[size_t(m)].resize(block_proj_bytes);
            down_buf[size_t(m)].resize(block_proj_bytes);
        }

        HfShardRangeFetcher fetcher(raw_cache);
        AsyncTargetPrefetcher prefetcher(fetcher, mtp_items, fetch_threads);
        std::atomic<int> done_cnt{0};
        std::mutex log_mtx;

        const int cpu_workers = 8;
        auto worker_fn = [&]() {
#ifdef _OPENMP
            omp_set_num_threads(6);
#endif
            std::vector<float> wr1, wr3, wr2;
            for (;;) {
                size_t idx = 0;
                RawExpertMxfp4 raw;
                std::string err;
                if (!prefetcher.take_next(idx, raw, err)) break;
                const auto& te = mtp_items[idx];
                if (!err.empty()) {
                    throw std::runtime_error("Fetch failed for MTP b" + std::to_string(te.bid) + " e" + std::to_string(te.logical_eid) + ": " + err);
                }
                const int m = te.bid - 40;
                const auto qg = dequant_mxfp4_and_quantize_hlut3_hi(raw.w1_w.data(), raw.w1_s.data(), DSV41_I, DSV41_H, mtp_sh[size_t(m)], lloyd_sol, wr1);
                const auto qu = dequant_mxfp4_and_quantize_hlut3_hi(raw.w3_w.data(), raw.w3_s.data(), DSV41_I, DSV41_H, mtp_sh[size_t(m)], lloyd_sol, wr3);
                const auto qd = dequant_mxfp4_and_quantize_hlut3_hi(raw.w2_w.data(), raw.w2_s.data(), DSV41_H, DSV41_I, mtp_si[size_t(m)], lloyd_sol, wr2);
                const size_t off = size_t(te.local_slot) * kHlut3UnshardedProjBytes;
                std::memcpy(gate_buf[size_t(m)].data() + off, qg.tiles.data(), kHlut3UnshardedProjBytes);
                std::memcpy(up_buf[size_t(m)].data()   + off, qu.tiles.data(), kHlut3UnshardedProjBytes);
                std::memcpy(down_buf[size_t(m)].data() + off, qd.tiles.data(), kHlut3UnshardedProjBytes);

                const int cur = ++done_cnt;
                if (cur % 32 == 0 || cur == 384) {
                    std::lock_guard<std::mutex> lk(log_mtx);
                    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
                    std::printf("[FAST_HI_MTP %3d/384] elapsed=%.1fs (%.2f exp/s)\n", cur, el, double(cur) / std::max(1e-3, el));
                    std::fflush(stdout);
                }
            }
        };

        std::vector<std::thread> ths;
        for (int t = 0; t < cpu_workers; ++t) ths.emplace_back(worker_fn);
        for (auto& th : ths) th.join();
        prefetcher.stop();

        std::error_code ec;
        if (raw_cache.rfind("/dev/shm/", 0) == 0) std::filesystem::remove_all(raw_cache, ec);
        std::filesystem::create_directories(std::filesystem::path(dev3_out).parent_path(), ec);

        StreamingSafetensorsWriter writer;
        for (int m = 0; m < 3; ++m) {
            const std::string pfx = "dev3.b" + std::to_string(40 + m) + ".";
            writer.add_owned(pfx + "gate_tiles", "U8", {128, 72, 160, 400}, std::move(gate_buf[size_t(m)]));
            writer.add_owned(pfx + "up_tiles",   "U8", {128, 72, 160, 400}, std::move(up_buf[size_t(m)]));
            writer.add_owned(pfx + "down_tiles", "U8", {128, 160, 72, 400}, std::move(down_buf[size_t(m)]));
        }
        writer.write_atomic(dev3_out);
        const double tot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        std::printf("[FAST_HI_MTP] DONE in %.2fs -> %s (384 MTP unsharded HLUT3 experts, 5,062.50 MiB)\n",
                    tot_s, dev3_out.c_str());
        return 0;
    }

    // ---- MODE 2 & 3: PLAN-ONLY / FULL-OPTION-B (1,704 Promoted Experts) ----
    std::array<std::vector<uint16_t>, 40> dev3_eids{}, tp3_eids{};
    std::vector<TargetExpert> plan = build_deterministic_tier_plan(
        hist_path, perm_ov_path, dev3_bb_target, tp3_bb_target,
        dev3_per_layer_cap, tp3_per_layer_cap, dev3_eids, tp3_eids);

    write_tier_plan_manifest(plan_manifest, plan, dev3_eids, tp3_eids);
    if (!write_targets_out.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(write_targets_out).parent_path(), ec);
        std::ofstream tf(write_targets_out, std::ios::trunc);
        tf << "# Wave 9.2 exact harvest targets (block logical_eid)\n";
        for (const auto& t : plan) tf << t.bid << " " << t.logical_eid << "\n";
    }
    std::printf("[WAVE9_2_PLAN] Wrote deterministic %zu-expert tier plan -> %s\n", plan.size(), plan_manifest.c_str());
    std::fflush(stdout);

    if (mode == "plan-only") {
        return 0;
    }
    if (mode != "full-option-b") {
        std::fprintf(stderr, "[WAVE9_2_BUILD] Unknown --mode '%s'\n", mode.c_str());
        return 1;
    }

    // Attach harvested .bin paths from cal_dir
    for (auto& t : plan) {
        char fn[128];
        std::snprintf(fn, sizeof(fn), "%s/b%02d_e%03d.bin", cal_dir.c_str(), t.bid, t.logical_eid);
        std::error_code ec;
        if (std::filesystem::is_regular_file(fn, ec)) {
            t.bin_path = fn;
        }
    }

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
    for (int l = 0; l < 40; ++l) {
        const size_t nb = dev3_eids[size_t(l)].size() * kHlut3UnshardedProjBytes;
        dev3_gate_buf[size_t(l)].resize(nb);
        dev3_up_buf[size_t(l)].resize(nb);
        dev3_down_buf[size_t(l)].resize(nb);
    }

    std::array<std::array<std::vector<uint8_t>, 40>, 3> tp3_gate_buf{}, tp3_up_buf{}, tp3_down_buf{};
    for (int r = 0; r < 3; ++r) {
        for (int l = 0; l < 40; ++l) {
            const size_t nb = tp3_eids[size_t(l)].size() * kHlut3Tp3ProjBytes;
            tp3_gate_buf[size_t(r)][size_t(l)].resize(nb);
            tp3_up_buf[size_t(r)][size_t(l)].resize(nb);
            tp3_down_buf[size_t(r)][size_t(l)].resize(nb);
        }
    }

    std::vector<Wave92ReceiptRow> receipts(plan.size());
    int dev_cnt = 0;
    if (hipGetDeviceCount(&dev_cnt) != hipSuccess || dev_cnt <= 0) dev_cnt = 1;
    const int n_workers = std::min(gpu_workers, dev_cnt);

    std::printf("[WAVE9_2_BUILD] Solving %zu WHT128-HLUT3 experts across %d MI50 GPUs (cal_dir=%s)...\n",
                plan.size(), n_workers, cal_dir.c_str());
    std::fflush(stdout);

    HfShardRangeFetcher fetcher(raw_cache);
    AsyncTargetPrefetcher prefetcher(fetcher, plan, fetch_threads);
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

            // 1. Always compute exact H=I WHT128-HLUT3 seed tiles + rotated weights
            const Hlut3QuantMatrix qg3_hi = dequant_mxfp4_and_quantize_hlut3_hi(
                raw.w1_w.data(), raw.w1_s.data(), DSV41_I, DSV41_H, signs_h, lloyd_sol, w1_rot);
            const Hlut3QuantMatrix qu3_hi = dequant_mxfp4_and_quantize_hlut3_hi(
                raw.w3_w.data(), raw.w3_s.data(), DSV41_I, DSV41_H, signs_h, lloyd_sol, w3_rot);
            const Hlut3QuantMatrix qd3_hi = dequant_mxfp4_and_quantize_hlut3_hi(
                raw.w2_w.data(), raw.w2_s.data(), DSV41_H, DSV41_I, signs_i, lloyd_sol, w2_rot);

            Hlut3QuantMatrix qg_sel = qg3_hi;
            Hlut3QuantMatrix qu_sel = qu3_hi;
            Hlut3QuantMatrix qd_sel = qd3_hi;

            Wave92ReceiptRow rr{};
            rr.tier          = te.tier;
            rr.bid           = te.bid;
            rr.logical_eid   = te.logical_eid;
            rr.base_phys_eid = te.base_phys_eid;
            rr.local_slot    = te.local_slot;
            rr.hist_hits     = te.hist_hits;

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
                // Release BF16 capture memory immediately
                std::vector<uint16_t>().swap(cap.x_bf16);

                Hlut2RarebalStats st_in{}, st_mid{};
                const auto R_up_in = gpu_accel.build_cholesky_rup(
                    X_cal.data(), M_cal, DSV41_H, signs_h, rare_cal.data(), &st_in);

                if (!R_up_in.empty()) {
                    // Solve HLUT3 RAREBAL-LDLQ gate & up
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

                        // Also solve HLUT2 RAREBAL-LDLQ on the exact same R_up_in / R_up_mid for paired held-out check!
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

            // Write selected HLUT3 tiles lock-free into the exact slot
            if (te.tier == TargetTier::kTp3Backbone) {
                const auto sh = pack_hlut3_expert_tp3_sharded(qg_sel, qu_sel, qd_sel);
                const size_t off = size_t(te.local_slot) * kHlut3Tp3ProjBytes;
                for (int r = 0; r < 3; ++r) {
                    std::memcpy(tp3_gate_buf[size_t(r)][size_t(te.bid)].data() + off, sh[size_t(r)].gate_tiles.data(), kHlut3Tp3ProjBytes);
                    std::memcpy(tp3_up_buf[size_t(r)][size_t(te.bid)].data()   + off, sh[size_t(r)].up_tiles.data(),   kHlut3Tp3ProjBytes);
                    std::memcpy(tp3_down_buf[size_t(r)][size_t(te.bid)].data() + off, sh[size_t(r)].down_tiles.data(), kHlut3Tp3ProjBytes);
                }
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
            if (rr.used_rarebal || (cur % 16 == 0) || cur == int(plan.size())) {
                std::lock_guard<std::mutex> lk(log_mtx);
                std::printf("[HLUT3_92 %4d/%4zu gpu%d] b%02d e%03d | M_cal=%d | cos2=%.5f cos3_hi=%.5f cos3_rb=%.5f | SNR2=%.2f SNR3=%.2f (+%.2fdB vs HLUT2) | %s (%.2fs)\n",
                            cur, plan.size(), wid, te.bid, te.logical_eid, rr.m_cal,
                            rr.m2_rb.cos_all, rr.m3_hi.cos_all, rr.m3_rb.cos_all,
                            rr.m2_rb.snr_db, rr.used_rarebal ? rr.m3_rb.snr_db : rr.m3_hi.snr_db,
                            (rr.used_rarebal ? rr.m3_rb.snr_db : rr.m3_hi.snr_db) - rr.m2_rb.snr_db,
                            rr.used_rarebal ? "EMITTED_RAREBAL" : "FALLBACK_HLUT3_HI", dt);
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

    // Emit wave9_2_dev3_hlut3.safetensors (14.21 GiB)
    {
        std::filesystem::create_directories(std::filesystem::path(dev3_out).parent_path(), ec);
        StreamingSafetensorsWriter dev3_writer;
        for (int b = 40; b < 43; ++b) {
            const std::string pfx = "dev3.b" + std::to_string(b) + ".";
            dev3_writer.add_owned(pfx + "gate_tiles", "U8", {128, 72, 160, 400}, std::move(dev3_gate_buf[size_t(b)]));
            dev3_writer.add_owned(pfx + "up_tiles",   "U8", {128, 72, 160, 400}, std::move(dev3_up_buf[size_t(b)]));
            dev3_writer.add_owned(pfx + "down_tiles", "U8", {128, 160, 72, 400}, std::move(dev3_down_buf[size_t(b)]));
        }
        for (int l = 0; l < 40; ++l) {
            const int64_t ne = int64_t(dev3_eids[size_t(l)].size());
            if (ne <= 0) continue;
            const std::string pfx = "dev3.b" + std::to_string(l) + ".";
            std::vector<uint8_t> eid_bytes(size_t(ne) * sizeof(uint16_t));
            std::memcpy(eid_bytes.data(), dev3_eids[size_t(l)].data(), eid_bytes.size());
            dev3_writer.add_owned(pfx + "logical_eids", "U16", {ne}, std::move(eid_bytes));
            dev3_writer.add_owned(pfx + "gate_tiles",   "U8",  {ne, 72, 160, 400}, std::move(dev3_gate_buf[size_t(l)]));
            dev3_writer.add_owned(pfx + "up_tiles",     "U8",  {ne, 72, 160, 400}, std::move(dev3_up_buf[size_t(l)]));
            dev3_writer.add_owned(pfx + "down_tiles",   "U8",  {ne, 160, 72, 400}, std::move(dev3_down_buf[size_t(l)]));
        }
        dev3_writer.write_atomic(dev3_out);
    }

    // Emit wave9_2_tp3_hlut3.safetensors (7.73 GiB)
    {
        std::filesystem::create_directories(std::filesystem::path(tp3_out).parent_path(), ec);
        StreamingSafetensorsWriter tp3_writer;
        for (int l = 0; l < 40; ++l) {
            const int64_t ne = int64_t(tp3_eids[size_t(l)].size());
            if (ne <= 0) continue;
            std::vector<uint8_t> eid_bytes(size_t(ne) * sizeof(uint16_t));
            std::memcpy(eid_bytes.data(), tp3_eids[size_t(l)].data(), eid_bytes.size());
            tp3_writer.add_owned("tp3.b" + std::to_string(l) + ".logical_eids", "U16", {ne}, std::move(eid_bytes));
            for (int r = 0; r < 3; ++r) {
                const std::string pfx = "tp3.tp" + std::to_string(r) + ".b" + std::to_string(l) + ".";
                tp3_writer.add_owned(pfx + "gate_tiles", "U8", {ne, 24, 160, 400}, std::move(tp3_gate_buf[size_t(r)][size_t(l)]));
                tp3_writer.add_owned(pfx + "up_tiles",   "U8", {ne, 24, 160, 400}, std::move(tp3_up_buf[size_t(r)][size_t(l)]));
                tp3_writer.add_owned(pfx + "down_tiles", "U8", {ne, 160, 24, 400}, std::move(tp3_down_buf[size_t(r)][size_t(l)]));
            }
        }
        tp3_writer.write_atomic(tp3_out);
    }

    // Verify receipts & write manifests/wave9_2_receipt_manifest.txt + manifests/SHA256SUMS_WAVE9_2.txt
    int dual_violations = 0;
    int rb_mtp = 0, rb_dev3_bb = 0, rb_tp3_bb = 0;
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
        else if (r.tier == TargetTier::kDev3Backbone) ++rb_dev3_bb;
        else ++rb_tp3_bb;
    }
    const int tot_rb = rb_mtp + rb_dev3_bb + rb_tp3_bb;

    {
        std::filesystem::create_directories(std::filesystem::path(receipt_manifest).parent_path(), ec);
        std::ofstream mf(receipt_manifest, std::ios::trunc);
        mf << "# Wave 9.2 SB-M4-P2-176G WHT128-HLUT3 Dual-Receipt Calibration Manifest\n"
           << "# total_hlut3_experts=" << receipts.size()
           << " emitted_rarebal_hlut3=" << tot_rb
           << " (mtp=" << rb_mtp << ", dev3_bb=" << rb_dev3_bb << ", tp3_bb=" << rb_tp3_bb << ")"
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
            const char* tname = (r.tier == TargetTier::kDev3Mtp) ? "DEV3_MTP"
                              : ((r.tier == TargetTier::kDev3Backbone) ? "DEV3_BB" : "TP3_BB");
            mf << tname << " " << r.bid << " " << r.logical_eid << " " << r.base_phys_eid << " "
               << r.local_slot << " " << r.hist_hits << " " << r.m_cal << " " << r.m_eval << " "
               << r.rare_cal << " " << r.rare_eval << " " << std::fixed << std::setprecision(5)
               << r.lam_in << " " << r.lam_mid << " "
               << r.m2_rb.cos_all << " " << r.m3_hi.cos_all << " " << r.m3_rb.cos_all << " "
               << r.m2_rb.snr_db  << " " << r.m3_hi.snr_db  << " " << r.m3_rb.snr_db  << " "
               << std::scientific << r.m2_rb.mse_rare << " " << r.m3_hi.mse_rare << " " << r.m3_rb.mse_rare << " "
               << (r.used_rarebal ? "EMITTED_RAREBAL" : "FALLBACK_HLUT3_HI") << "\n";
        }
    }

    const std::string sha_dev3 = sha256_file_hex(dev3_out);
    const std::string sha_tp3  = sha256_file_hex(tp3_out);
    {
        std::filesystem::create_directories(std::filesystem::path(sha256_manifest).parent_path(), ec);
        std::ofstream sf(sha256_manifest, std::ios::trunc);
        sf << sha_dev3 << "  " << std::filesystem::path(dev3_out).filename().string() << "\n"
           << sha_tp3  << "  " << std::filesystem::path(tp3_out).filename().string() << "\n";
    }

    const double tot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("[WAVE9_2_BUILD] DONE in %.2fs | total=%zu | RAREBAL=%d (mtp=%d, dev3_bb=%d, tp3_bb=%d) | HI_fallback=%d | dual_violations=%d\n"
                "  dev3_overlay : %s (sha256=%s)\n"
                "  tp3_overlay  : %s (sha256=%s)\n",
                tot_s, receipts.size(), tot_rb, rb_mtp, rb_dev3_bb, rb_tp3_bb,
                int(receipts.size()) - tot_rb, dual_violations,
                dev3_out.c_str(), sha_dev3.c_str(),
                tp3_out.c_str(), sha_tp3.c_str());
    std::fflush(stdout);
    return (dual_violations == 0) ? 0 : 2;
}
