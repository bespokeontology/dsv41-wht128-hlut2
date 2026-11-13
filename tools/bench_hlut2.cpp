// Offline Pure-C++17 WHT128 HLUT2 + TARQ/RAREBAL-LDLQ + Starvation-Gate Ablation Validator
// Evaluates Real DeepSeek-V4.1-Flash-1017 Layer 6 Routed Expert 0 (5120 -> 2304 -> 5120) across:
//   (a) Uniform 2-bit absmax (group-128, 2.125 bpw, unrotated)
//   (b) Data-free WHT128 + H=I Symmetric Gaussian Lloyd-Max (production 2.125 bpw, 272 B/tile)
//   (c) Starved Unweighted LDLQ (M_cal = 512 <= 2K without gate) vs Starvation-Gated LDLQ (-> H=I)
//   (d) Unweighted LDLQ (M_cal = 11,264 > 2K)
//   (e) WHT128 + TARQ RAREBAL-LDLQ (M_cal = 11,264 > 2K, trace-equalized H_common + lambda * H_rare)

#include "../src/hlut2_pack.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace dsv41;

namespace {

struct EvalBucketMetrics {
    double cos_all = 0.0;
    double snr_db_all = 0.0;
    double mse_common = 0.0;
    double mse_rare = 0.0;
    double cos_common = 0.0;
    double cos_rare = 0.0;
};

void matmul_nt_f32(const float* A, const float* B, float* C, int M, int N, int K) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const float* am = A + size_t(m) * K;
            const float* bn = B + size_t(n) * K;
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += am[k] * bn[k];
            C[size_t(m) * N + n] = acc;
        }
    }
}

void run_swiglu_f32(
    const float* X, int M,
    const float* w_gate, const float* w_up, const float* w_down,
    std::vector<float>& out_y, std::vector<float>* out_z = nullptr)
{
    std::vector<float> g(size_t(M) * DSV41_I);
    std::vector<float> u(size_t(M) * DSV41_I);
    std::vector<float> z(size_t(M) * DSV41_I);
    matmul_nt_f32(X, w_gate, g.data(), M, DSV41_I, DSV41_H);
    matmul_nt_f32(X, w_up,   u.data(), M, DSV41_I, DSV41_H);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < z.size(); ++i) z[i] = dsv41_swiglu_scalar(g[i], u[i]);
    if (out_z) *out_z = z;
    out_y.resize(size_t(M) * DSV41_H);
    matmul_nt_f32(z.data(), w_down, out_y.data(), M, DSV41_H, DSV41_I);
}

void build_calibration_and_eval_tokens(
    int M, int K, double rare_frac, uint64_t basis_seed, uint64_t token_seed,
    std::vector<float>& out_X, std::vector<uint8_t>& out_is_rare)
{
    std::mt19937_64 rng_basis(basis_seed);
    std::mt19937_64 rng(token_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // Layer-fixed common-prose latent subspace (R_com=24) + rare code/identifier/DSML subspace (R_rare=16)
    constexpr int R_com = 24;
    constexpr int R_rare = 16;
    std::vector<float> B_com(size_t(R_com) * K, 0.0f);
    std::vector<float> B_rare(size_t(R_rare) * K, 0.0f);

    for (int r = 0; r < R_com; ++r) {
        for (int k = 0; k < K; ++k) {
            const float outlier = ((k & 63) == 7) ? 5.5f : 1.0f;
            B_com[size_t(r) * K + k] = ((k % 4) != 3) ? (nd(rng_basis) * outlier * 0.42f) : 0.0f;
        }
    }
    for (int r = 0; r < R_rare; ++r) {
        for (int k = 0; k < K; ++k) {
            const float outlier = ((k & 63) == 31) ? 6.0f : 1.0f;
            B_rare[size_t(r) * K + k] = ((k % 4) == 3) ? (nd(rng_basis) * outlier * 0.72f) : (nd(rng_basis) * 0.03f);
        }
    }

    out_X.assign(size_t(M) * K, 0.0f);
    out_is_rare.assign(size_t(M), 0u);
    const int rare_mod = std::max(2, int(std::round(1.0 / rare_frac)));

    for (int m = 0; m < M; ++m) {
        const bool is_rare = ((m % rare_mod) == (rare_mod - 1));
        out_is_rare[size_t(m)] = is_rare ? 1u : 0u;
        float* xm = out_X.data() + size_t(m) * K;
        if (!is_rare) {
            float c[R_com];
            for (int r = 0; r < R_com; ++r) c[r] = nd(rng);
            for (int r = 0; r < R_com; ++r) {
                const float cr = c[r];
                const float* br = B_com.data() + size_t(r) * K;
                for (int k = 0; k < K; ++k) xm[k] += cr * br[k];
            }
            for (int k = 0; k < K; ++k) xm[k] = bf16_to_f32(f32_to_bf16_rne(xm[k] + 0.05f * nd(rng)));
        } else {
            float c[R_rare];
            for (int r = 0; r < R_rare; ++r) c[r] = nd(rng);
            for (int r = 0; r < R_rare; ++r) {
                const float cr = c[r];
                const float* br = B_rare.data() + size_t(r) * K;
                for (int k = 0; k < K; ++k) xm[k] += cr * br[k];
            }
            for (int k = 0; k < K; ++k) xm[k] = bf16_to_f32(f32_to_bf16_rne(xm[k] + 0.05f * nd(rng)));
        }
    }
}

EvalBucketMetrics measure_buckets(
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
        const float* r = y_ref.data() + size_t(m) * D;
        const float* q = y_q.data() + size_t(m) * D;
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
    EvalBucketMetrics out;
    out.cos_all = dot_all / (std::sqrt(nr_all) * std::sqrt(nq_all));
    out.snr_db_all = 10.0 * std::log10(nr_all / std::max(1e-30, err_all));
    out.mse_common = err_c / double(std::max<size_t>(1, cnt_c));
    out.mse_rare = err_r / double(std::max<size_t>(1, cnt_r));
    out.cos_common = dot_c / (std::sqrt(nr_c) * std::sqrt(nq_c));
    out.cos_rare = dot_r / (std::sqrt(nr_r) * std::sqrt(nq_r));
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                     ? std::string(std::getenv("DSV41_ROOT"))
                                     : std::string(".");
    std::string scales_path  = root_env + "/shards/layer6_expert0_scales_e8m0.bin";
    std::string weights_path = root_env + "/shards/layer6_expert0_weights_mxfp4.bin";
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--scales" && i + 1 < argc) scales_path = argv[++i];
        else if (a == "--weights" && i + 1 < argc) weights_path = argv[++i];
    }

    SwigluExpertF32 ex;
    const bool ok = load_real_dsv41_expert0(scales_path.c_str(), weights_path.c_str(), ex);
    if (!ok) {
        std::fprintf(stderr, "Failed to load Layer 6 Expert 0 shards (%s, %s); pass --scales/--weights or set DSV41_ROOT.\n",
                     scales_path.c_str(), weights_path.c_str());
        return 1;
    }

    const auto signs_h = make_wht_signs(DSV41_H, 2026111106ULL);
    const auto signs_i = make_wht_signs(DSV41_I, 2026111206ULL);

    // Calibration set: M_cal = 11,264 (> 2 * K = 10,240), 96% common tokens + 4% rare code/DSML/symbol tokens
    const int M_cal = 11264;
    std::vector<float> X_cal;
    std::vector<uint8_t> rare_cal;
    build_calibration_and_eval_tokens(M_cal, DSV41_H, 0.04, 0xB4515006ULL, 0xD541CA11ULL, X_cal, rare_cal);

    // Independent held-out evaluation set in same layer-6 activation manifold: M_eval = 512 (256 common + 256 rare)
    const int M_eval = 512;
    std::vector<float> X_eval, X_eval_rot(size_t(M_eval) * DSV41_H);
    std::vector<uint8_t> rare_eval;
    build_calibration_and_eval_tokens(M_eval, DSV41_H, 0.50, 0xB4515006ULL, 0xE7A12026ULL, X_eval, rare_eval);
    wht128_rows(X_eval.data(), X_eval_rot.data(), M_eval, DSV41_H, signs_h);

    std::vector<float> Z_cal, Y_cal_dummy, Y_ref;
    run_swiglu_f32(X_cal.data(), M_cal, ex.gate.data(), ex.up.data(), ex.down.data(), Y_cal_dummy, &Z_cal);
    run_swiglu_f32(X_eval.data(), M_eval, ex.gate.data(), ex.up.data(), ex.down.data(), Y_ref, nullptr);

    auto eval_hlut2_triplet = [&](const Hlut2QuantMatrix& qg, const Hlut2QuantMatrix& qu, const Hlut2QuantMatrix& qd) {
        std::vector<float> wg_r, wu_r, wd_r;
        dequantize_hlut2_to_rot_f32(qg, wg_r);
        dequantize_hlut2_to_rot_f32(qu, wu_r);
        dequantize_hlut2_to_rot_f32(qd, wd_r);

        std::vector<float> g(size_t(M_eval) * DSV41_I), u(size_t(M_eval) * DSV41_I);
        std::vector<float> z(size_t(M_eval) * DSV41_I), z_rot(size_t(M_eval) * DSV41_I);
        matmul_nt_f32(X_eval_rot.data(), wg_r.data(), g.data(), M_eval, DSV41_I, DSV41_H);
        matmul_nt_f32(X_eval_rot.data(), wu_r.data(), u.data(), M_eval, DSV41_I, DSV41_H);
        for (size_t i = 0; i < z.size(); ++i) z[i] = dsv41_swiglu_scalar(g[i], u[i]);
        wht128_rows(z.data(), z_rot.data(), M_eval, DSV41_I, signs_i);

        std::vector<float> y_q(size_t(M_eval) * DSV41_H);
        matmul_nt_f32(z_rot.data(), wd_r.data(), y_q.data(), M_eval, DSV41_H, DSV41_I);
        return measure_buckets(Y_ref, y_q, rare_eval, M_eval, DSV41_H);
    };

    std::printf("=== WHT128 HLUT2 & TARQ/RAREBAL-LDLQ Ablation on %s (5120 -> 2304 -> 5120) ===\n", ex.name.c_str());
    std::printf("Calibration: M_cal=%d (> 2*K=10240), Held-out Eval: M_eval=%d (256 common + 256 rare)\n\n", M_cal, M_eval);
    std::printf("%-38s | %8s | %8s | %11s | %11s | %9s | %9s\n",
                "Method (2.125 bpw = 272 B/tile)", "SwiGLU", "SNR (dB)", "Common MSE", "Rare MSE", "Cos(Com)", "Cos(Rare)");
    std::printf("---------------------------------------+----------+----------+-------------+-------------+-----------+----------\n");

    // (a) Uniform 2-bit absmax (group-128, unrotated)
    {
        std::vector<float> wg(ex.gate.size()), wu(ex.up.size()), wd(ex.down.size()), y_q;
        quant_dequant_uniform_2bit_absmax(ex.gate.data(), wg.data(), DSV41_I, DSV41_H, 128);
        quant_dequant_uniform_2bit_absmax(ex.up.data(),   wu.data(), DSV41_I, DSV41_H, 128);
        quant_dequant_uniform_2bit_absmax(ex.down.data(), wd.data(), DSV41_H, DSV41_I, 128);
        run_swiglu_f32(X_eval.data(), M_eval, wg.data(), wu.data(), wd.data(), y_q);
        const auto m = measure_buckets(Y_ref, y_q, rare_eval, M_eval, DSV41_H);
        std::printf("%-38s | %8.5f | %8.2f | %11.6e | %11.6e | %9.5f | %9.5f\n",
                    "(a) Uniform 2-bit absmax (unrot)", m.cos_all, m.snr_db_all, m.mse_common, m.mse_rare, m.cos_common, m.cos_rare);
    }

    // (b) Data-free WHT128 + H=I Lloyd-Max (production live format)
    Hlut2QuantMatrix qg_hi = quantize_hlut2(ex.gate.data(), DSV41_I, DSV41_H, signs_h);
    Hlut2QuantMatrix qu_hi = quantize_hlut2(ex.up.data(),   DSV41_I, DSV41_H, signs_h);
    Hlut2QuantMatrix qd_hi = quantize_hlut2(ex.down.data(), DSV41_H, DSV41_I, signs_i);
    {
        const auto m = eval_hlut2_triplet(qg_hi, qu_hi, qd_hi);
        std::printf("%-38s | %8.5f | %8.2f | %11.6e | %11.6e | %9.5f | %9.5f\n",
                    "(b) Data-free WHT128 + H=I Lloyd-Max", m.cos_all, m.snr_db_all, m.mse_common, m.mse_rare, m.cos_common, m.cos_rare);
    }

    // Starvation check at M_cal = 512 <= 2*K: ungated rank-deficient LDLQ vs Starvation-Gated LDLQ (-> H=I)
    {
        Hlut2RarebalStats st_gated{};
        auto qg_gated = quantize_hlut2_ldlq(ex.gate.data(), DSV41_I, DSV41_H, signs_h, X_cal.data(), 512, true, rare_cal.data(), &st_gated, false);
        auto qu_gated = quantize_hlut2_ldlq(ex.up.data(),   DSV41_I, DSV41_H, signs_h, X_cal.data(), 512, true, rare_cal.data(), nullptr, false);
        auto qd_gated = quantize_hlut2_ldlq(ex.down.data(), DSV41_H, DSV41_I, signs_i, Z_cal.data(), 512, true, rare_cal.data(), nullptr, false);
        const auto mg = eval_hlut2_triplet(qg_gated, qu_gated, qd_gated);

        auto qg_starved = quantize_hlut2_ldlq(ex.gate.data(), DSV41_I, DSV41_H, signs_h, X_cal.data(), 512, false, nullptr, nullptr, true);
        auto qu_starved = quantize_hlut2_ldlq(ex.up.data(),   DSV41_I, DSV41_H, signs_h, X_cal.data(), 512, false, nullptr, nullptr, true);
        auto qd_starved = quantize_hlut2_ldlq(ex.down.data(), DSV41_H, DSV41_I, signs_i, Z_cal.data(), 512, false, nullptr, nullptr, true);
        const auto ms = eval_hlut2_triplet(qg_starved, qu_starved, qd_starved);
        std::printf("%-38s | %8.5f | %8.2f | %11.6e | %11.6e | %9.5f | %9.5f\n",
                    "    [Ungated Starved LDLQ M_cal=512]", ms.cos_all, ms.snr_db_all, ms.mse_common, ms.mse_rare, ms.cos_common, ms.cos_rare);
        std::printf("%-38s | %8.5f | %8.2f | %11.6e | %11.6e | %9.5f | %9.5f\n",
                    "    [Gated LDLQ M_cal=512 -> H=I]", mg.cos_all, mg.snr_db_all, mg.mse_common, mg.mse_rare, mg.cos_common, mg.cos_rare);
    }

    // (c) Unweighted LDLQ (M_cal = 11,264 > 2*K, enable_rarebal = false)
    {
        auto qg_u = quantize_hlut2_ldlq(ex.gate.data(), DSV41_I, DSV41_H, signs_h, X_cal.data(), M_cal, false, nullptr);
        auto qu_u = quantize_hlut2_ldlq(ex.up.data(),   DSV41_I, DSV41_H, signs_h, X_cal.data(), M_cal, false, nullptr);
        auto qd_u = quantize_hlut2_ldlq(ex.down.data(), DSV41_H, DSV41_I, signs_i, Z_cal.data(), M_cal, false, nullptr);
        const auto m = eval_hlut2_triplet(qg_u, qu_u, qd_u);
        std::printf("%-38s | %8.5f | %8.2f | %11.6e | %11.6e | %9.5f | %9.5f\n",
                    "(c) Unweighted WHT128-LDLQ (M>2K)", m.cos_all, m.snr_db_all, m.mse_common, m.mse_rare, m.cos_common, m.cos_rare);
    }

    // (d) WHT128 + TARQ RAREBAL-LDLQ (M_cal = 11,264 > 2*K, enable_rarebal = true)
    Hlut2RarebalStats st_h{}, st_i{};
    {
        auto qg_r = quantize_hlut2_ldlq(ex.gate.data(), DSV41_I, DSV41_H, signs_h, X_cal.data(), M_cal, true, rare_cal.data(), &st_h);
        auto qu_r = quantize_hlut2_ldlq(ex.up.data(),   DSV41_I, DSV41_H, signs_h, X_cal.data(), M_cal, true, rare_cal.data(), nullptr);
        auto qd_r = quantize_hlut2_ldlq(ex.down.data(), DSV41_H, DSV41_I, signs_i, Z_cal.data(), M_cal, true, rare_cal.data(), &st_i);
        const auto m = eval_hlut2_triplet(qg_r, qu_r, qd_r);
        std::printf("%-38s | %8.5f | %8.2f | %11.6e | %11.6e | %9.5f | %9.5f\n",
                    "(d) WHT128 + RAREBAL-LDLQ (M>2K)", m.cos_all, m.snr_db_all, m.mse_common, m.mse_rare, m.cos_common, m.cos_rare);
    }

    std::printf("\n[RAREBAL_STATS] gate/up (K=5120): m_com=%d m_rare=%d tr_com=%.3e tr_rare=%.3e lambda=%.4f bpw=%.3f\n",
                st_h.m_common, st_h.m_rare, st_h.tr_common, st_h.tr_rare, st_h.lambda, qg_hi.bits_per_weight());
    std::printf("[RAREBAL_STATS] down    (K=2304): m_com=%d m_rare=%d tr_com=%.3e tr_rare=%.3e lambda=%.4f bpw=%.3f\n",
                st_i.m_common, st_i.m_rare, st_i.tr_common, st_i.tr_rare, st_i.lambda, qd_hi.bits_per_weight());
    return 0;
}
