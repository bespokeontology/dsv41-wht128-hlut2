// Wave 9 Deliverable D live MI50 device-2 silicon benchmark on REAL DeepSeek-V4.1-Flash-1017
// Layer 6 routed expert 0 MXFP4 weights (5120 -> 2304 -> 5120):
//   (a) Uniform Absmax 2-bit (unrotated, 2.125 bpw)
//   (b) WHT128-HLUT2 H=I (production 2.125 bpw)
//   (c) WHT128-HLUT2 RAREBAL-LDLQ (production full reverse-Cholesky)
//   (d) WHT128-HLUT3 H=I (3.125 bpw, 400 B tile)
//   (e) WHT128-HLUT3 RAREBAL-LDLQ (importance-weighted 128-block Hessian)
// Reports SwiGLU output cosine, SNR dB, rare-token MSE, kernel latency, effective HBM2 bandwidth.
//
// Safety: hipSetDevice(2) FIRST HIP call; free >= 550 MiB; <= 48 MiB total HBM2. No P150/TT
// device or /dev/tenstorrent opens. Real shards are read-only.
// Pure C++20. Zero Python.

#include "../src/hlut2_pack.hpp"
#include "../src/hlut3_pack.hpp"
#include "../src/dsv41_hlut3_rarebal.hpp"
#include "../mi50/dsv41_hlut3_kernels.hpp"
#include "../p150/dsv41_p150_hlut3_spec.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kMaxDeviceBytes = 48ULL * 1024ULL * 1024ULL;
constexpr size_t kMinFreeBytes   = 550ULL * 1024ULL * 1024ULL;
constexpr int kMCal = 11264;
constexpr int kMEval = 512;
constexpr double kCalRareFrac = 0.04;
constexpr double kEvalRareFrac = 0.50;
constexpr double kParityCosMin = 0.999999;
constexpr double kReceiptHlut2RarebalGainDb = 4.0;

void hip_check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("HIP ") + what + ": " + hipGetErrorString(e));
}

struct DeviceBudget {
    size_t total = 0;
    std::vector<void*> ptrs;
    void* alloc(size_t bytes) {
        if (bytes == 0) return nullptr;
        if (total + bytes > kMaxDeviceBytes) throw std::runtime_error("bench HLUT3 exceeds 48 MiB HBM2 cap");
        void* p = nullptr;
        hip_check(hipMalloc(&p, bytes), "hipMalloc");
        total += bytes;
        ptrs.push_back(p);
        return p;
    }
    ~DeviceBudget() { for (void* p : ptrs) if (p) (void)hipFree(p); }
};

template <typename T>
T* dev_alloc(DeviceBudget& b, size_t count) {
    return static_cast<T*>(b.alloc(std::max<size_t>(count * sizeof(T), 64u)));
}

template <typename T>
void h2d(T* dst, const T* src, size_t count) {
    hip_check(hipMemcpy(dst, src, count * sizeof(T), hipMemcpyHostToDevice), "H2D");
}

void matmul_nt_f32(const float* A, const float* B, float* C, int M, int N, int K) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const float* am = A + size_t(m) * size_t(K);
            const float* bn = B + size_t(n) * size_t(K);
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += am[k] * bn[k];
            C[size_t(m) * size_t(N) + size_t(n)] = acc;
        }
    }
}

void gate_up_swiglu_f32(
    const float* X, int M, int K,
    const float* w_gate, const float* w_up,
    std::vector<float>& z_out)
{
    std::vector<float> g(size_t(M) * size_t(dsv41::DSV41_I));
    std::vector<float> u(size_t(M) * size_t(dsv41::DSV41_I));
    matmul_nt_f32(X, w_gate, g.data(), M, dsv41::DSV41_I, K);
    matmul_nt_f32(X, w_up,   u.data(), M, dsv41::DSV41_I, K);
    z_out.resize(size_t(M) * size_t(dsv41::DSV41_I));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < z_out.size(); ++i) {
        z_out[i] = dsv41::dsv41_swiglu_scalar(g[i], u[i]);
    }
}

// Production WHT128 basis contract: rotated gate/up tiles consume X_rot and emit the intermediate
// z in the ORIGINAL hidden basis (H_128 / sqrt(128) is symmetric orthogonal, so <H*w, H*x> = <w,x>).
// Rotated down tiles were packed along K=I with signs_i, therefore z MUST be rotated by
// wht128_rows(z, signs_i) BEFORE the down contraction. Omitting that intermediate rotation puts
// z in the original basis against down_rot -> ~0 SwiGLU cosine (the Wave9 Phase-1 defect).
void run_swiglu_f32(
    const float* X, int M, int K_in,
    const float* w_gate, const float* w_up, const float* w_down, int K_down,
    bool rotate_intermediate,
    const std::vector<uint64_t>& signs_i,
    std::vector<float>& out_y)
{
    std::vector<float> z;
    gate_up_swiglu_f32(X, M, K_in, w_gate, w_up, z);
    out_y.resize(size_t(M) * size_t(dsv41::DSV41_H));
    if (rotate_intermediate) {
        std::vector<float> z_rot(size_t(M) * size_t(dsv41::DSV41_I));
        dsv41::wht128_rows(z.data(), z_rot.data(), M, dsv41::DSV41_I, signs_i);
        matmul_nt_f32(z_rot.data(), w_down, out_y.data(), M, dsv41::DSV41_H, K_down);
    } else {
        matmul_nt_f32(z.data(), w_down, out_y.data(), M, dsv41::DSV41_H, K_down);
    }
}

void build_calibration_and_eval_tokens(
    int M, int K, double rare_frac, uint64_t basis_seed, uint64_t token_seed,
    std::vector<float>& out_X, std::vector<uint8_t>& out_is_rare)
{
    std::mt19937_64 rng_basis(basis_seed);
    std::mt19937_64 rng(token_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    constexpr int R_com = 24;
    constexpr int R_rare = 16;
    std::vector<float> B_com(size_t(R_com) * size_t(K), 0.0f);
    std::vector<float> B_rare(size_t(R_rare) * size_t(K), 0.0f);
    for (int r = 0; r < R_com; ++r) {
        for (int k = 0; k < K; ++k) {
            const float outlier = ((k & 63) == 7) ? 5.5f : 1.0f;
            B_com[size_t(r) * size_t(K) + size_t(k)] = ((k % 4) != 3) ? (nd(rng_basis) * outlier * 0.42f) : 0.0f;
        }
    }
    for (int r = 0; r < R_rare; ++r) {
        for (int k = 0; k < K; ++k) {
            const float outlier = ((k & 63) == 31) ? 6.0f : 1.0f;
            B_rare[size_t(r) * size_t(K) + size_t(k)] =
                ((k % 4) == 3) ? (nd(rng_basis) * outlier * 0.72f) : (nd(rng_basis) * 0.03f);
        }
    }
    out_X.assign(size_t(M) * size_t(K), 0.0f);
    out_is_rare.assign(size_t(M), 0u);
    const int rare_mod = std::max(2, int(std::round(1.0 / rare_frac)));
    for (int m = 0; m < M; ++m) {
        const bool is_rare = ((m % rare_mod) == (rare_mod - 1));
        out_is_rare[size_t(m)] = is_rare ? 1u : 0u;
        float* xm = out_X.data() + size_t(m) * size_t(K);
        if (!is_rare) {
            float c[R_com];
            for (int r = 0; r < R_com; ++r) c[r] = nd(rng);
            for (int r = 0; r < R_com; ++r) {
                const float cr = c[r];
                const float* br = B_com.data() + size_t(r) * size_t(K);
                for (int k = 0; k < K; ++k) xm[k] += cr * br[k];
            }
        } else {
            float c[R_rare];
            for (int r = 0; r < R_rare; ++r) c[r] = nd(rng);
            for (int r = 0; r < R_rare; ++r) {
                const float cr = c[r];
                const float* br = B_rare.data() + size_t(r) * size_t(K);
                for (int k = 0; k < K; ++k) xm[k] += cr * br[k];
            }
        }
        for (int k = 0; k < K; ++k) xm[k] = dsv41::bf16_to_f32(dsv41::f32_to_bf16_rne(xm[k] + 0.05f * nd(rng)));
    }
}

struct BucketMetrics {
    double cos_all = 0.0;
    double snr_db_all = 0.0;
    double mse_common = 0.0;
    double mse_rare = 0.0;
    double cos_common = 0.0;
    double cos_rare = 0.0;
};

BucketMetrics measure_buckets(
    const std::vector<float>& y_ref,
    const std::vector<float>& y_q,
    const std::vector<uint8_t>& is_rare,
    int M, int D)
{
    double dot_all = 0.0, nr_all = 0.0, nq_all = 0.0, err_all = 0.0;
    double dot_c = 0.0, nr_c = 0.0, nq_c = 0.0, err_c = 0.0;
    double dot_r = 0.0, nr_r = 0.0, nq_r = 0.0, err_r = 0.0;
    size_t cnt_c = 0, cnt_r = 0;
    for (int m = 0; m < M; ++m) {
        const bool rare = (is_rare[size_t(m)] != 0u);
        const float* r = y_ref.data() + size_t(m) * size_t(D);
        const float* q = y_q.data() + size_t(m) * size_t(D);
        for (int j = 0; j < D; ++j) {
            const double a = double(r[j]);
            const double b = double(q[j]);
            const double df = a - b;
            dot_all += a * b; nr_all += a * a; nq_all += b * b; err_all += df * df;
            if (rare) {
                dot_r += a * b; nr_r += a * a; nq_r += b * b; err_r += df * df;
                ++cnt_r;
            } else {
                dot_c += a * b; nr_c += a * a; nq_c += b * b; err_c += df * df;
                ++cnt_c;
            }
        }
    }
    BucketMetrics out;
    out.cos_all = dot_all / (std::sqrt(nr_all) * std::sqrt(nq_all));
    out.snr_db_all = 10.0 * std::log10(nr_all / std::max(1e-30, err_all));
    out.mse_common = err_c / double(std::max<size_t>(1, cnt_c));
    out.mse_rare = err_r / double(std::max<size_t>(1, cnt_r));
    out.cos_common = dot_c / (std::sqrt(nr_c) * std::sqrt(nq_c));
    out.cos_rare = dot_r / (std::sqrt(nr_r) * std::sqrt(nq_r));
    return out;
}

struct QuantizedSet {
    std::vector<float> gate_rot;
    std::vector<float> up_rot;
    std::vector<float> down_rot;
    bool rotated = true;
};

void eval_quantized_set(
    const QuantizedSet& qs,
    const std::vector<float>& X_eval,
    const std::vector<float>& X_eval_rot,
    const std::vector<uint64_t>& signs_i,
    const std::vector<float>& Y_ref,
    const std::vector<uint8_t>& rare_eval,
    int M,
    BucketMetrics& out)
{
    std::vector<float> y_q;
    if (qs.rotated) {
        run_swiglu_f32(X_eval_rot.data(), M, dsv41::DSV41_H,
                       qs.gate_rot.data(), qs.up_rot.data(), qs.down_rot.data(), dsv41::DSV41_I,
                       /*rotate_intermediate=*/true, signs_i, y_q);
    } else {
        run_swiglu_f32(X_eval.data(), M, dsv41::DSV41_H,
                       qs.gate_rot.data(), qs.up_rot.data(), qs.down_rot.data(), dsv41::DSV41_I,
                       /*rotate_intermediate=*/false, signs_i, y_q);
    }
    out = measure_buckets(Y_ref, y_q, rare_eval, M, dsv41::DSV41_H);
}

QuantizedSet make_hlut2_set(const dsv41::Hlut2QuantMatrix& qg,
                            const dsv41::Hlut2QuantMatrix& qu,
                            const dsv41::Hlut2QuantMatrix& qd)
{
    QuantizedSet qs;
    dsv41::dequantize_hlut2_to_rot_f32(qg, qs.gate_rot);
    dsv41::dequantize_hlut2_to_rot_f32(qu, qs.up_rot);
    dsv41::dequantize_hlut2_to_rot_f32(qd, qs.down_rot);
    return qs;
}

QuantizedSet make_hlut3_set(const dsv41::Hlut3QuantMatrix& qg,
                            const dsv41::Hlut3QuantMatrix& qu,
                            const dsv41::Hlut3QuantMatrix& qd)
{
    QuantizedSet qs;
    dsv41::unpack_hlut3_to_rot_f32(qg, qs.gate_rot);
    dsv41::unpack_hlut3_to_rot_f32(qu, qs.up_rot);
    dsv41::unpack_hlut3_to_rot_f32(qd, qs.down_rot);
    return qs;
}

QuantizedSet make_uniform_set(const dsv41::SwigluExpertF32& ex)
{
    QuantizedSet qs;
    qs.gate_rot.resize(ex.gate.size());
    qs.up_rot.resize(ex.up.size());
    qs.down_rot.resize(ex.down.size());
    dsv41::quant_dequant_uniform_2bit_absmax(ex.gate.data(), qs.gate_rot.data(), dsv41::DSV41_I, dsv41::DSV41_H, 128);
    dsv41::quant_dequant_uniform_2bit_absmax(ex.up.data(),   qs.up_rot.data(),   dsv41::DSV41_I, dsv41::DSV41_H, 128);
    dsv41::quant_dequant_uniform_2bit_absmax(ex.down.data(), qs.down_rot.data(), dsv41::DSV41_H, dsv41::DSV41_I, 128);
    qs.rotated = false;
    return qs;
}

struct MethodRow {
    std::string name;
    std::string bpw;
    BucketMetrics m{};
    double quant_ms = 0.0;
};

struct GpuLatencyRow {
    std::string name;
    int M = 0;
    double latency_us = 0.0;
    double bytes = 0.0;
    double bandwidth_gb_s = 0.0;
};

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static inline std::string dsv41_root_path(const char* rel_suffix, const char* alt_env = nullptr) {
    if (alt_env) {
        if (const char* v = std::getenv(alt_env); v && *v) return std::string(v) + "/" + rel_suffix;
    }
    const char* root = std::getenv("DSV41_ROOT");
    if (!root || !*root) {
        std::fprintf(stderr, "ERROR: Set DSV41_ROOT to your local model/pack directory.\n");
        std::exit(2);
    }
    return std::string(root) + "/" + rel_suffix;
}

}  // namespace

int main() {
    hip_check(hipSetDevice(2), "hipSetDevice(2)");
    size_t free_b = 0, total_b = 0;
    hip_check(hipMemGetInfo(&free_b, &total_b), "hipMemGetInfo");
    std::printf("[SAFETY] device=2 gfx906 free=%.2f MiB total=%.2f MiB min_free=550 MiB cap=48 MiB\n",
                double(free_b) / (1024.0 * 1024.0), double(total_b) / (1024.0 * 1024.0));
    if (free_b < kMinFreeBytes) {
        std::fprintf(stderr, "[SAFETY] device-2 free HBM2 %.2f MiB < 550 MiB; refusing benchmark\n",
                     double(free_b) / (1024.0 * 1024.0));
        return 2;
    }

    dsv41::p150_hlut3_geometry_report();

    const auto t_load0 = std::chrono::steady_clock::now();
    dsv41::SwigluExpertF32 ex;
    const std::string scales_path =
        dsv41_root_path("shards_skhynix/layer6_expert0_scales_e8m0.bin", "DSV41_SCALES_ROOT");
    const std::string weights_path =
        dsv41_root_path("shards_greatwall_data/layer6_expert0_weights_mxfp4.bin", "DSV41_NVME1_ROOT");
    if (!dsv41::load_real_dsv41_expert0(scales_path, weights_path, ex)) {
        std::fprintf(stderr, "[BENCH] real layer6 expert0 shards missing\n");
        return 3;
    }
    std::printf("[WEIGHTS] loaded %s gate/up/down 2304x5120 in %.2f s (real MXFP4 shards)\n",
                ex.name.c_str(), seconds_since(t_load0));

    const auto signs_h = dsv41::make_wht_signs(dsv41::DSV41_H, 2026111106ULL);
    const auto signs_i = dsv41::make_wht_signs(dsv41::DSV41_I, 2026111206ULL);

    std::vector<float> X_cal, X_eval, X_eval_rot(size_t(kMEval) * dsv41::DSV41_H);
    std::vector<uint8_t> rare_cal, rare_eval;
    build_calibration_and_eval_tokens(kMCal, dsv41::DSV41_H, kCalRareFrac,
                                      0xB4515006ULL, 0xD541CA11ULL, X_cal, rare_cal);
    build_calibration_and_eval_tokens(kMEval, dsv41::DSV41_H, kEvalRareFrac,
                                      0xB4515006ULL, 0xE7A12026ULL, X_eval, rare_eval);
    dsv41::wht128_rows(X_eval.data(), X_eval_rot.data(), kMEval, dsv41::DSV41_H, signs_h);
    std::printf("[CAL] M_cal=%d (>=2K=%d), rare=%zu; M_eval=%d (common=%zu rare=%zu)\n",
                kMCal, 2 * dsv41::DSV41_H,
                size_t(std::count(rare_cal.begin(), rare_cal.end(), uint8_t(1))),
                kMEval,
                size_t(std::count(rare_eval.begin(), rare_eval.end(), uint8_t(0))),
                size_t(std::count(rare_eval.begin(), rare_eval.end(), uint8_t(1))));
    std::fflush(stdout);

    const auto t_zcal0 = std::chrono::steady_clock::now();
    std::vector<float> Z_cal;
    gate_up_swiglu_f32(X_cal.data(), kMCal, dsv41::DSV41_H,
                       ex.gate.data(), ex.up.data(), Z_cal);
    std::printf("[Z_CAL] real SwiGLU intermediate 2304: %.2f s (%zu MiB)\n",
                seconds_since(t_zcal0), Z_cal.size() * sizeof(float) / (1024u * 1024u));

    const auto t_ref0 = std::chrono::steady_clock::now();
    std::vector<float> Y_ref;
    run_swiglu_f32(X_eval.data(), kMEval, dsv41::DSV41_H,
                   ex.gate.data(), ex.up.data(), ex.down.data(), dsv41::DSV41_I,
                   /*rotate_intermediate=*/false, signs_i, Y_ref);
    std::printf("[REF] real SwiGLU output Y_ref M_eval=%d: %.2f s\n", kMEval, seconds_since(t_ref0));
    std::fflush(stdout);

    std::vector<MethodRow> rows;

    // (a) Uniform Absmax 2-bit, unrotated.
    {
        const auto t0 = std::chrono::steady_clock::now();
        QuantizedSet qs = make_uniform_set(ex);
        BucketMetrics m{};
        eval_quantized_set(qs, X_eval, X_eval_rot, signs_i, Y_ref, rare_eval, kMEval, m);
        rows.push_back(MethodRow{"(a) Uniform Absmax 2b unrotated", "2.125", m, seconds_since(t0) * 1000.0});
    }

    // (b) WHT128-HLUT2 H=I production.
    const auto qg_hi = dsv41::quantize_hlut2(ex.gate.data(), dsv41::DSV41_I, dsv41::DSV41_H, signs_h);
    const auto qu_hi = dsv41::quantize_hlut2(ex.up.data(),   dsv41::DSV41_I, dsv41::DSV41_H, signs_h);
    const auto qd_hi = dsv41::quantize_hlut2(ex.down.data(), dsv41::DSV41_H, dsv41::DSV41_I, signs_i);
    {
        const auto t0 = std::chrono::steady_clock::now();
        QuantizedSet qs = make_hlut2_set(qg_hi, qu_hi, qd_hi);
        BucketMetrics m{};
        eval_quantized_set(qs, X_eval, X_eval_rot, signs_i, Y_ref, rare_eval, kMEval, m);
        rows.push_back(MethodRow{"(b) WHT128-HLUT2 H=I", "2.125", m, seconds_since(t0) * 1000.0});
        std::printf("[HLUT2_HI] bpw=%.3f raw=%.2f MiB quantization+unpack+eval %.2f ms\n",
                    qg_hi.bits_per_weight(), double(qg_hi.raw_bytes() * 3) / (1024.0 * 1024.0),
                    rows.back().quant_ms);
        std::fflush(stdout);
    }

    // (c) WHT128-HLUT2 RAREBAL-LDLQ: production full reverse-Cholesky on the exact same X_cal/Z_cal.
    {
        const auto t0 = std::chrono::steady_clock::now();
        dsv41::Hlut2RarebalStats st_gate{}, st_down{};
        std::vector<float> wrot_gate(size_t(dsv41::DSV41_I) * dsv41::DSV41_H);
        std::vector<float> wrot_up(size_t(dsv41::DSV41_I) * dsv41::DSV41_H);
        std::vector<float> wrot_down(size_t(dsv41::DSV41_H) * dsv41::DSV41_I);
        dsv41::wht128_rows(ex.gate.data(), wrot_gate.data(), dsv41::DSV41_I, dsv41::DSV41_H, signs_h);
        dsv41::wht128_rows(ex.up.data(),   wrot_up.data(),   dsv41::DSV41_I, dsv41::DSV41_H, signs_h);
        dsv41::wht128_rows(ex.down.data(), wrot_down.data(), dsv41::DSV41_H, dsv41::DSV41_I, signs_i);
        const auto R_gate_up = dsv41::build_rarebal_cholesky_rup(
            X_cal.data(), kMCal, dsv41::DSV41_H, signs_h, true, rare_cal.data(), &st_gate);
        const auto R_down = dsv41::build_rarebal_cholesky_rup(
            Z_cal.data(), kMCal, dsv41::DSV41_I, signs_i, true, rare_cal.data(), &st_down);
        if (R_gate_up.empty() || R_down.empty()) throw std::runtime_error("HLUT2 RAREBAL R build starvation fallback");
        auto qg_r = dsv41::apply_hlut2_ldlq_with_rup(qg_hi, wrot_gate.data(), R_gate_up);
        auto qu_r = dsv41::apply_hlut2_ldlq_with_rup(qu_hi, wrot_up.data(), R_gate_up);
        auto qd_r = dsv41::apply_hlut2_ldlq_with_rup(qd_hi, wrot_down.data(), R_down);
        QuantizedSet qs = make_hlut2_set(qg_r, qu_r, qd_r);
        BucketMetrics m{};
        eval_quantized_set(qs, X_eval, X_eval_rot, signs_i, Y_ref, rare_eval, kMEval, m);
        rows.push_back(MethodRow{"(c) WHT128-HLUT2 RAREBAL-LDLQ", "2.125", m, seconds_since(t0) * 1000.0});
        std::printf("[HLUT2_RAREBAL] lambda_h=%.6f lambda_i=%.6f R_h=%zu R_i=%zu build+apply+eval %.2f ms\n",
                    st_gate.lambda, st_down.lambda,
                    R_gate_up.size(), R_down.size(), rows.back().quant_ms);
        std::fflush(stdout);
    }

    // (d) WHT128-HLUT3 H=I, exact Gaussian fixed point.
    const auto qg3_hi = dsv41::quantize_hlut3(ex.gate.data(), dsv41::DSV41_I, dsv41::DSV41_H, signs_h);
    const auto qu3_hi = dsv41::quantize_hlut3(ex.up.data(),   dsv41::DSV41_I, dsv41::DSV41_H, signs_h);
    const auto qd3_hi = dsv41::quantize_hlut3(ex.down.data(), dsv41::DSV41_H, dsv41::DSV41_I, signs_i);
    {
        const auto t0 = std::chrono::steady_clock::now();
        QuantizedSet qs = make_hlut3_set(qg3_hi, qu3_hi, qd3_hi);
        BucketMetrics m{};
        eval_quantized_set(qs, X_eval, X_eval_rot, signs_i, Y_ref, rare_eval, kMEval, m);
        rows.push_back(MethodRow{"(d) WHT128-HLUT3 H=I", "3.125", m, seconds_since(t0) * 1000.0});
        std::printf("[HLUT3_HI] bpw=%.3f raw=%.2f MiB quantization+unpack+eval %.2f ms\n",
                    qg3_hi.bits_per_weight(), double(qg3_hi.raw_bytes() * 3) / (1024.0 * 1024.0),
                    rows.back().quant_ms);
        std::fflush(stdout);
    }

    // (e) WHT128-HLUT3 RAREBAL-LDLQ, importance-weighted 128-wide reverse Cholesky.
    dsv41::Hlut3RarebalStats st3_gate{}, st3_down{};
    {
        const auto t0 = std::chrono::steady_clock::now();
        auto qg_r = dsv41::quantize_hlut3_rarebal_ldlq(
            ex.gate.data(), dsv41::DSV41_I, dsv41::DSV41_H, signs_h,
            X_cal.data(), kMCal, true, rare_cal.data(), &st3_gate);
        auto qu_r = dsv41::quantize_hlut3_rarebal_ldlq(
            ex.up.data(), dsv41::DSV41_I, dsv41::DSV41_H, signs_h,
            X_cal.data(), kMCal, true, rare_cal.data(), nullptr);
        auto qd_r = dsv41::quantize_hlut3_rarebal_ldlq(
            ex.down.data(), dsv41::DSV41_H, dsv41::DSV41_I, signs_i,
            Z_cal.data(), kMCal, true, rare_cal.data(), &st3_down);
        QuantizedSet qs = make_hlut3_set(qg_r, qu_r, qd_r);
        BucketMetrics m{};
        eval_quantized_set(qs, X_eval, X_eval_rot, signs_i, Y_ref, rare_eval, kMEval, m);
        rows.push_back(MethodRow{"(e) WHT128-HLUT3 RAREBAL-LDLQ", "3.125", m, seconds_since(t0) * 1000.0});
        std::printf("[HLUT3_RAREBAL] lambda_h=%.6f lambda_i=%.6f reg_h=%.6e reg_i=%.6e %.2f ms\n",
                    st3_gate.lambda_tarq, st3_down.lambda_tarq,
                    st3_gate.lambda_reg, st3_down.lambda_reg, rows.back().quant_ms);
        std::fflush(stdout);
    }

    std::printf("\n=== REAL-EXPERT FIVE-BASELINE CPU RECEIPT (SwiGLU 5120->2304->5120) ===\n");
    std::printf("%-38s | %8s | %8s | %11s | %11s | %9s | %9s | %9s | %9s\n",
                "Method", "bpw", "SwiGLU", "SNR(dB)", "Common MSE", "Rare MSE", "Cos(Com)", "Cos(Rare)", "ms");
    std::printf("---------------------------------------+----------+----------+-------------+-------------+-----------+----------+--------\n");
    for (const auto& r : rows) {
        std::printf("%-38s | %8s | %8.5f | %8.2f | %11.6e | %11.6e | %9.5f | %9.5f | %9.2f\n",
                    r.name.c_str(), r.bpw.c_str(), r.m.cos_all, r.m.snr_db_all,
                    r.m.mse_common, r.m.mse_rare, r.m.cos_common, r.m.cos_rare, r.quant_ms);
    }

    // Dual Receipt gating on held-out M_eval=512: HLUT3+RAREBAL vs HLUT3(H=I) and HLUT2+RAREBAL.
    const BucketMetrics& m_uniform = rows[0].m;
    const BucketMetrics& m_h2 = rows[1].m;
    const BucketMetrics& m_h2r = rows[2].m;
    const BucketMetrics& m_h3 = rows[3].m;
    const BucketMetrics& m_h3r = rows[4].m;
    const bool cos_beats_hi = m_h3r.cos_all > m_h3.cos_all;
    const bool rare_beats_hi = m_h3r.mse_rare < m_h3.mse_rare;
    const double gain_h2r = m_h3r.snr_db_all - m_h2r.snr_db_all;
    const bool gain_beats_4db = gain_h2r >= kReceiptHlut2RarebalGainDb;
    const bool receipt = cos_beats_hi && rare_beats_hi && gain_beats_4db;

    std::printf("\n=== DUAL RECEIPT ===\n");
    std::printf("HLUT3+RAREBAL cosine %.5f > HLUT3(H=I) %.5f : %s\n",
                m_h3r.cos_all, m_h3.cos_all, cos_beats_hi ? "PASS" : "FAIL");
    std::printf("HLUT3+RAREBAL rare MSE %.6e < HLUT3(H=I) %.6e : %s\n",
                m_h3r.mse_rare, m_h3.mse_rare, rare_beats_hi ? "PASS" : "FAIL");
    std::printf("HLUT3+RAREBAL SNR %.2f dB vs HLUT2+RAREBAL %.2f dB gain=+%.2f dB >= +4.0 : %s\n",
                m_h3r.snr_db_all, m_h2r.snr_db_all, gain_h2r, gain_beats_4db ? "PASS" : "FAIL");
    std::printf("Uniform baseline SNR %.2f dB rare MSE %.6e\n", m_uniform.snr_db_all, m_uniform.mse_rare);
    std::printf("HLUT3(H=I) SNR %.2f dB rare MSE %.6e\n", m_h3.snr_db_all, m_h3.mse_rare);
    std::printf("HLUT2(H=I) SNR %.2f dB rare MSE %.6e\n", m_h2.snr_db_all, m_h2.mse_rare);

    // ---- Live device-2 silicon latency and effective HBM2 bandwidth on real HLUT3 matrices ----
    DeviceBudget budget;
    auto* d_gate = static_cast<dsv41::Hlut3Tile32x32*>(
        budget.alloc(qg3_hi.tiles.size() * sizeof(dsv41::Hlut3Tile32x32)));
    auto* d_up = static_cast<dsv41::Hlut3Tile32x32*>(
        budget.alloc(qu3_hi.tiles.size() * sizeof(dsv41::Hlut3Tile32x32)));
    auto* d_down = static_cast<dsv41::Hlut3Tile32x32*>(
        budget.alloc(qd3_hi.tiles.size() * sizeof(dsv41::Hlut3Tile32x32)));
    auto* d_xrot = dev_alloc<float>(budget, size_t(16) * size_t(dsv41::DSV41_H));
    auto* d_pairs = dev_alloc<dsv41::Mi50RoutedPair>(budget, 16);
    auto* d_z = dev_alloc<float>(budget, size_t(16) * size_t(dsv41::DSV41_I));
    auto* d_zrot = dev_alloc<float>(budget, size_t(16) * size_t(dsv41::DSV41_I));
    auto* d_y = dev_alloc<float>(budget, size_t(16) * size_t(dsv41::DSV41_H));
    const std::array<float, 4> levels = {
        float(qg3_hi.lloyd.levels[0]), float(qg3_hi.lloyd.levels[1]),
        float(qg3_hi.lloyd.levels[2]), float(qg3_hi.lloyd.levels[3])
    };
    float* d_levels = dev_alloc<float>(budget, levels.size());
    h2d(d_levels, levels.data(), levels.size());
    uint64_t* d_signs_i = dev_alloc<uint64_t>(budget, signs_i.size());
    h2d(d_signs_i, signs_i.data(), signs_i.size());
    hip_check(hipMemcpy(d_gate, qg3_hi.tiles.data(), qg3_hi.tiles.size() * sizeof(dsv41::Hlut3Tile32x32),
                        hipMemcpyHostToDevice), "gate tiles H2D");
    hip_check(hipMemcpy(d_up, qu3_hi.tiles.data(), qu3_hi.tiles.size() * sizeof(dsv41::Hlut3Tile32x32),
                        hipMemcpyHostToDevice), "up tiles H2D");
    hip_check(hipMemcpy(d_down, qd3_hi.tiles.data(), qd3_hi.tiles.size() * sizeof(dsv41::Hlut3Tile32x32),
                        hipMemcpyHostToDevice), "down tiles H2D");
    std::printf("[ALLOC] device-2 strict total = %.3f MiB <= 48 MiB\n",
                double(budget.total) / (1024.0 * 1024.0));

    hipEvent_t ev0 = nullptr, ev1 = nullptr;
    hip_check(hipEventCreate(&ev0), "event create");
    hip_check(hipEventCreate(&ev1), "event create");

    constexpr double kTileBytes = double(3) * double(4'608'000ULL);
    std::vector<GpuLatencyRow> gpu_rows;
    for (int M : {1, 6, 8, 16}) {
        std::vector<float> Xrot(size_t(M) * size_t(dsv41::DSV41_H));
        dsv41::wht128_rows(X_eval.data(), Xrot.data(), M, dsv41::DSV41_H, signs_h);
        h2d(d_xrot, Xrot.data(), Xrot.size());
        std::vector<dsv41::Mi50RoutedPair> pairs(static_cast<size_t>(M));
        for (int m = 0; m < M; ++m) pairs[size_t(m)] = dsv41::Mi50RoutedPair{m, 0, 1.0f, 0};
        h2d(d_pairs, pairs.data(), pairs.size());

        constexpr int kReps = 7;
        double elapsed_ms = 0.0;
        for (int rep = 0; rep < kReps; ++rep) {
            hip_check(hipMemsetAsync(d_y, 0, size_t(M) * size_t(dsv41::DSV41_H) * sizeof(float), nullptr), "zero y");
            hip_check(hipEventRecord(ev0, nullptr), "ev0 record");
            dsv41::launch_mi50_routed_hlut3_pairs(
                d_xrot, d_pairs, M, M, /*n_experts=*/1,
                dsv41::DSV41_I, dsv41::DSV41_H, dsv41::DSV41_H, dsv41::DSV41_I,
                d_gate, d_up, d_down,
                d_levels, d_levels, d_levels,
                d_signs_i, d_z, d_zrot, d_y, nullptr);
            hip_check(hipEventRecord(ev1, nullptr), "ev1 record");
            hip_check(hipEventSynchronize(ev1), "ev1 sync");
            float ms = 0.0f;
            hip_check(hipEventElapsedTime(&ms, ev0, ev1), "event elapsed");
            elapsed_ms += double(ms);
        }
        const double mean_us = (elapsed_ms / double(kReps)) * 1000.0;
        const double bw = kTileBytes / (mean_us * 1e3); // bytes / seconds -> GB/s decimal
        gpu_rows.push_back(GpuLatencyRow{"HLUT3(H=I) real tiles", M, mean_us, kTileBytes, bw});
        std::printf("[SILICON] HLUT3(H=I) M=%2d mean=%.2f us tile_bytes=%.3f MB effective=%.2f GB/s\n",
                    M, mean_us, kTileBytes / (1024.0 * 1024.0), bw);
        std::fflush(stdout);
    }
    hip_check(hipEventDestroy(ev0), "event destroy");
    hip_check(hipEventDestroy(ev1), "event destroy");

    // Device-2 correctness of the timed pipeline against the CPU quantized reference for M=16.
    bool silicon_parity = true;
    {
        constexpr int M = 16;
        std::vector<float> Xrot(size_t(M) * size_t(dsv41::DSV41_H));
        dsv41::wht128_rows(X_eval.data(), Xrot.data(), M, dsv41::DSV41_H, signs_h);
        h2d(d_xrot, Xrot.data(), Xrot.size());
        std::vector<dsv41::Mi50RoutedPair> pairs(static_cast<size_t>(M));
        for (int m = 0; m < M; ++m) pairs[size_t(m)] = dsv41::Mi50RoutedPair{m, 0, 1.0f, 0};
        h2d(d_pairs, pairs.data(), pairs.size());
        hip_check(hipMemset(d_y, 0, size_t(M) * size_t(dsv41::DSV41_H) * sizeof(float)), "zero y");
        dsv41::launch_mi50_routed_hlut3_pairs(
            d_xrot, d_pairs, M, M, /*n_experts=*/1,
            dsv41::DSV41_I, dsv41::DSV41_H, dsv41::DSV41_H, dsv41::DSV41_I,
            d_gate, d_up, d_down,
            d_levels, d_levels, d_levels,
            d_signs_i, d_z, d_zrot, d_y, nullptr);
        hip_check(hipDeviceSynchronize(), "device sync");
        std::vector<float> y_gpu(size_t(M) * size_t(dsv41::DSV41_H));
        hip_check(hipMemcpy(y_gpu.data(), d_y, y_gpu.size() * sizeof(float), hipMemcpyDeviceToHost), "D2H");
        QuantizedSet qs = make_hlut3_set(qg3_hi, qu3_hi, qd3_hi);
        std::vector<float> y_cpu;
        run_swiglu_f32(X_eval_rot.data(), M, dsv41::DSV41_H,
                       qs.gate_rot.data(), qs.up_rot.data(), qs.down_rot.data(), dsv41::DSV41_I,
                       /*rotate_intermediate=*/true, signs_i, y_cpu);
        const dsv41::QualityMetrics qm = dsv41::compute_quality(y_cpu, y_gpu);
        silicon_parity = qm.cosine >= kParityCosMin;
        std::printf("[SILICON_PARITY] M=16 cosine=%.9f max_abs=%.6e %s\n",
                    qm.cosine, qm.max_abs, silicon_parity ? "PASS" : "FAIL");
    }

    std::printf("\n=== WAVE9_HLUT3_MI50 BENCH RECEIPT ===\n");
    std::printf("device_locked               : hipSetDevice(2) FIRST HIP call\n");
    std::printf("free_hbm2                  : %.2f MiB >= 550 MiB PASS\n", double(free_b) / (1024.0 * 1024.0));
    std::printf("total_hbm2_alloc           : %.3f MiB <= 48 MiB PASS\n", double(budget.total) / (1024.0 * 1024.0));
    std::printf("real_weight_hlut3_bytes    : 4,608,000 B/matrix (3.125 bpw exact)\n");
    std::printf("dual_receipt               : %s\n", receipt ? "PASS" : "FAIL");
    std::printf("hlut3_vs_hlut2_theory      : +%.2f dB (N=8 vs N=4 Gaussian fixed point)\n",
                dsv41::solve_gaussian_lloyd_max_hlut3().sqnr_db -
                dsv41::solve_gaussian_lloyd_max_hlut2().sqnr_db);
    std::printf("silicon_parity_M16         : %s\n", silicon_parity ? "PASS" : "FAIL");
    std::printf("overall                    : %s\n", (receipt && silicon_parity) ? "PASS" : "FAIL");
    std::fflush(stdout);
    return (receipt && silicon_parity) ? 0 : 1;
}
