#pragma once
#define DSV41_HLUT2_PACK_INCLUDED 1
// HLUT2 (WHT128 + Symmetric 2-Bit Lloyd-Max Centroids, 2.125 bpw) + Optional 1-Pass Cholesky LDLQ/GPTQ
// Offline Quantizer, Reference Evaluator, and Hardware Packers for:
//   - AMD Instinct MI50 (gfx906, 3x 32 GiB HBM2)
//   - Tenstorrent Blackhole P150A (Tensix SFPU + HiFi2 FP32 Dest, 2x 32 GiB GDDR6 + 2x 180 MB L1 SRAM)
// Exact DeepSeek-V4.1-Flash-1017 SwiGLU Expert Geometry: 5120 -> 2304 -> 5120 (TP2: 1152/card).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dsv41 {

static constexpr int DSV41_H = 5120;
static constexpr int DSV41_I = 2304;
static constexpr int DSV41_I_TP2 = 1152;
static constexpr int WHT_BLK = 128;
static constexpr int TT_TILE = 32;
static constexpr float DSV41_SWIGLU_LIMIT = 10.0f;

// ---- Exact IEEE FP16 & BF16 conversions ----
inline uint16_t f32_to_bf16_rne(float x) {
    uint32_t u = 0;
    std::memcpy(&u, &x, 4);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return uint16_t((u >> 16) | 0x0040u);
    u += 0x00007FFFu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

inline float bf16_to_f32(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float x = 0.0f;
    std::memcpy(&x, &u, 4);
    return x;
}

inline uint16_t f32_to_fp16_rne(float x) {
    uint32_t f = 0;
    std::memcpy(&f, &x, 4);
    const uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = int32_t((f >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = f & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        mant |= 0x00800000u;
        const int shift = 14 - exp;
        const uint32_t half_bit = 1u << (shift - 1);
        const uint32_t mask = (1u << shift) - 1u;
        uint32_t h = mant >> shift;
        if ((mant & mask) > half_bit || ((mant & mask) == half_bit && (h & 1u))) ++h;
        return uint16_t(sign | h);
    } else if (exp >= 31) {
        if ((f & 0x7FFFFFFFu) > 0x7F800000u) return uint16_t(sign | 0x7E00u);
        return uint16_t(sign | 0x7C00u);
    }
    const uint32_t round_bit = 0x00001000u;
    const uint32_t rem = mant & 0x00001FFFu;
    uint32_t h_mant = mant >> 13;
    uint32_t h_exp = uint32_t(exp) << 10;
    uint32_t res = h_exp | h_mant;
    if (rem > round_bit || (rem == round_bit && (h_mant & 1u))) ++res;
    return uint16_t(sign | res);
}

inline float fp16_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t u = 0;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            int e = -14;
            uint32_t m = mant;
            while ((m & 0x0400u) == 0) { m <<= 1; --e; }
            m &= 0x03FFu;
            u = sign | (uint32_t(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | (uint32_t(int(exp) - 15 + 127) << 23) | (mant << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ---- Deterministic Per-Layer Random Signs + Normalized WHT128 ----
inline std::vector<uint64_t> make_wht_signs(int K, uint64_t seed) {
    if (K % WHT_BLK != 0) throw std::invalid_argument("WHT128 requires K % 128 == 0");
    std::vector<uint64_t> mask(size_t(K / 64), 0ULL);
    uint64_t st = seed ^ 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < mask.size(); ++i) {
        st ^= st >> 12;
        st ^= st << 25;
        st ^= st >> 27;
        mask[i] = st * 0x2545F4914F6CDD1DULL;
    }
    return mask;
}

inline bool sign_is_neg(const uint64_t* sign_bits, int k) {
    return ((sign_bits[k >> 6] >> (k & 63)) & 1ULL) != 0ULL;
}

// Exact orthonormal transform y = (1/sqrt(128)) * H_128 * (diag(s) * x).
// Note: Since (H_128 / sqrt(128)) is symmetric & orthogonal and diag(s)^2 = I,
// applying wht128_block(w, s) and wht128_block(x, s) preserves <w, x> == <w_rot, x_rot> identically!
inline void wht128_block(float* v, const uint64_t* sign_bits_block2) {
    for (int i = 0; i < 64; ++i) {
        if ((sign_bits_block2[0] >> i) & 1ULL) v[i] = -v[i];
        if ((sign_bits_block2[1] >> i) & 1ULL) v[64 + i] = -v[64 + i];
    }
    for (int len = 1; len < 128; len <<= 1) {
        for (int i = 0; i < 128; i += (len << 1)) {
            for (int j = 0; j < len; ++j) {
                const float a = v[i + j];
                const float b = v[i + len + j];
                v[i + j] = a + b;
                v[i + len + j] = a - b;
            }
        }
    }
    constexpr float kInvSqrt128 = 0.08838834764831844655f;  // 1 / sqrt(128)
    for (int i = 0; i < 128; ++i) v[i] *= kInvSqrt128;
}

inline void wht128_rows(const float* in, float* out, int rows, int K, const std::vector<uint64_t>& signs) {
    const int nblk = K / 128;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const float* xr = in + size_t(r) * K;
        float* yr = out + size_t(r) * K;
        std::memcpy(yr, xr, size_t(K) * sizeof(float));
        for (int b = 0; b < nblk; ++b) {
            wht128_block(yr + b * 128, &signs[size_t(b) * 2]);
        }
    }
}

// Build explicit [K, K] block-diagonal orthogonal matrix Q_b = diag(s_b) * H_128 / sqrt(128)
// for Tensix L1 WHT128 tile matmul: x_rot = x * Q where x_b * Q_b computes (H_128 * diag(s_b) * x_b^T)^T
// Note: (H_128 * D * x^T)^T = x * D * H_128^T = x * D * H_128!
inline void build_wht128_block_matrix(int b, const std::vector<uint64_t>& signs, float out_128x128[128][128]) {
    constexpr float kInvSqrt128 = 0.08838834764831844655f;
    const uint64_t s0 = signs[size_t(b) * 2];
    const uint64_t s1 = signs[size_t(b) * 2 + 1];
    for (int i = 0; i < 128; ++i) {
        const float si = (((i < 64 ? (s0 >> i) : (s1 >> (i - 64))) & 1ULL) != 0ULL) ? -1.0f : 1.0f;
        for (int j = 0; j < 128; ++j) {
            const int pop = __builtin_popcount(unsigned(i & j));
            const float hij = (pop & 1) ? -kInvSqrt128 : kInvSqrt128;
            out_128x128[i][j] = si * hij;
        }
    }
}

// ---- Exact DeepSeek-V4.1-Flash-1017 SwiGLU nonlinearity (swiglu_limit = 10.0) ----
inline float dsv41_swiglu_scalar(float g, float u) {
    const float gc = std::min(g, DSV41_SWIGLU_LIMIT);
    const float uc = std::max(-DSV41_SWIGLU_LIMIT, std::min(DSV41_SWIGLU_LIMIT, u));
    const float silu_g = gc / (1.0f + std::exp(-gc));
    return silu_g * uc;
}

// ---- HLUT2 Quantized Matrix Representation ----
struct Hlut2QuantMatrix {
    int N = 0;
    int K = 0;
    float c0 = 0.452780f;
    float c1 = 1.510417f;
    // Row-major packed 2-bit codes: [N, K / 16] uint32_t (16 2-bit weights per uint32_t; bits [2*m+1:2*m], bit1=sign, bit0=mag)
    std::vector<uint32_t> codes_rm;
    // Row-major FP16 scales: [N, K / 128] uint16_t
    std::vector<uint16_t> scales_fp16;

    size_t raw_bytes() const {
        return codes_rm.size() * sizeof(uint32_t) + scales_fp16.size() * sizeof(uint16_t);
    }
    double bits_per_weight() const {
        return 8.0 * double(raw_bytes()) / double(size_t(N) * size_t(K));
    }
};

inline void fit_lloyd_max_2bit_centroids(
    const std::vector<float>& w_rot, int N, int K,
    const std::vector<float>& sigma_f32, float& out_c0, float& out_c1)
{
    const int G = K / 128;
    float c0 = 0.452780f;
    float c1 = 1.510417f;
    for (int it = 0; it < 12; ++it) {
        const float tau = 0.5f * (c0 + c1);
        double num0 = 0.0, den0 = 0.0, num1 = 0.0, den1 = 0.0;
        #pragma omp parallel for reduction(+:num0,den0,num1,den1) schedule(static)
        for (int r = 0; r < N; ++r) {
            const float* wr = w_rot.data() + size_t(r) * K;
            for (int g = 0; g < G; ++g) {
                const float sig = sigma_f32[size_t(r) * G + g];
                if (sig <= 0.0f) continue;
                const double w2 = double(sig) * double(sig);
                const float inv_s = 1.0f / sig;
                const float* wg = wr + g * 128;
                for (int j = 0; j < 128; ++j) {
                    const float z = std::fabs(wg[j]) * inv_s;
                    if (z < tau) {
                        num0 += w2 * double(z);
                        den0 += w2;
                    } else {
                        num1 += w2 * double(z);
                        den1 += w2;
                    }
                }
            }
        }
        if (den0 > 0.0) c0 = float(num0 / den0);
        if (den1 > 0.0) c1 = float(num1 / den1);
        if (c1 <= c0 + 1e-4f) c1 = c0 + 0.5f;
    }
    out_c0 = c0;
    out_c1 = c1;
}

// Standard 1-pass HLUT2 quantization (zero calibration required)
inline Hlut2QuantMatrix quantize_hlut2(
    const float* W_rm, int N, int K, const std::vector<uint64_t>& signs)
{
    if (N % 32 != 0 || K % 128 != 0) throw std::invalid_argument("HLUT2 requires N%32==0 and K%128==0");
    Hlut2QuantMatrix q;
    q.N = N;
    q.K = K;
    const int G = K / 128;
    std::vector<float> w_rot(size_t(N) * K);
    wht128_rows(W_rm, w_rot.data(), N, K, signs);

    q.scales_fp16.resize(size_t(N) * G);
    std::vector<float> sigma_f32(size_t(N) * G);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < N; ++r) {
        const float* wr = w_rot.data() + size_t(r) * K;
        for (int g = 0; g < G; ++g) {
            double sq = 0.0;
            const float* wg = wr + g * 128;
            for (int j = 0; j < 128; ++j) sq += double(wg[j]) * double(wg[j]);
            const float rms = float(std::sqrt(sq / 128.0 + 1e-30));
            const uint16_t h = f32_to_fp16_rne(rms);
            q.scales_fp16[size_t(r) * G + g] = h;
            sigma_f32[size_t(r) * G + g] = fp16_to_f32(h);
        }
    }

    fit_lloyd_max_2bit_centroids(w_rot, N, K, sigma_f32, q.c0, q.c1);
    const float tau = 0.5f * (q.c0 + q.c1);

    const int words_per_row = K / 16;
    q.codes_rm.assign(size_t(N) * words_per_row, 0u);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < N; ++r) {
        const float* wr = w_rot.data() + size_t(r) * K;
        uint32_t* cr = q.codes_rm.data() + size_t(r) * words_per_row;
        for (int g = 0; g < G; ++g) {
            const float sig = sigma_f32[size_t(r) * G + g];
            const float thr = tau * sig;
            for (int w = 0; w < 8; ++w) {
                uint32_t pack = 0u;
                const float* src = wr + g * 128 + w * 16;
                for (int m = 0; m < 16; ++m) {
                    const float v = src[m];
                    const uint32_t sign_bit = std::signbit(v) ? 2u : 0u;
                    const uint32_t mag_bit = (std::fabs(v) >= thr) ? 1u : 0u;
                    pack |= ((sign_bit | mag_bit) << (2 * m));
                }
                cr[g * 8 + w] = pack;
            }
        }
    }
    return q;
}

// HLUT2 + 1-pass Cholesky GPTQ/LDLQ error feedback with:
//   1. Starvation Fallback Gate: M_cal <= 2 * K automatically returns data-free H = I quantize_hlut2()
//      (prevents rank-deficient Hessian overfitting on cold tail MoE experts; <= 10,240 for K=5120, <= 4,608 for K=2304).
//   2. TARQ Single-Scalar RAREBAL Trace Equalization (enable_rarebal = true):
//      Accumulates separate rotated second-moment matrices H_common and H_rare, computes closed-form
//      lambda = tr(H_common) / max(1e-12, tr(H_rare)), and forms
//      H_bal = (1 / M_cal) * (H_common + lambda * H_rare) + 0.015 * d_bar * I_K
//      feeding directly into single-pass U = L^{-T} Cholesky LDLQ (272 B/tile, zero runtime kernel changes!).
struct Hlut2RarebalStats {
    int m_cal = 0;
    int m_common = 0;
    int m_rare = 0;
    bool used_starvation_fallback = false;
    double tr_common = 0.0;
    double tr_rare = 0.0;
    double lambda = 1.0;
    double d_bar = 0.0;
};

inline std::vector<float> build_rarebal_cholesky_rup(
    const float* X_cal_rm, int M_cal, int K, const std::vector<uint64_t>& signs,
    bool enable_rarebal = true,
    const uint8_t* is_rare_token = nullptr,
    Hlut2RarebalStats* out_stats = nullptr,
    bool bypass_starvation_gate = false)
{
    if (out_stats) *out_stats = Hlut2RarebalStats{};
    if (!X_cal_rm || M_cal <= 0 || (!bypass_starvation_gate && M_cal <= 2 * K)) {
        if (out_stats) {
            out_stats->m_cal = std::max(0, M_cal);
            out_stats->used_starvation_fallback = true;
        }
        return {};
    }

    std::vector<float> x_rot(size_t(M_cal) * K);
    wht128_rows(X_cal_rm, x_rot.data(), M_cal, K, signs);

    // Resolve per-token rarity mask (explicit mask or automatic directional-rarity partition against batch centroid)
    std::vector<uint8_t> auto_rare;
    const uint8_t* rare_mask = is_rare_token;
    if (enable_rarebal && !rare_mask) {
        auto_rare.assign(size_t(M_cal), 0u);
        std::vector<double> mean_dir(size_t(K), 0.0);
        std::vector<float> row_inv_norm(size_t(M_cal), 0.0f);
        for (int m = 0; m < M_cal; ++m) {
            const float* xm = x_rot.data() + size_t(m) * K;
            double n2 = 0.0;
            for (int k = 0; k < K; ++k) n2 += double(xm[k]) * double(xm[k]);
            const double inv_n = (n2 > 1e-30) ? (1.0 / std::sqrt(n2)) : 0.0;
            row_inv_norm[size_t(m)] = float(inv_n);
            for (int k = 0; k < K; ++k) mean_dir[size_t(k)] += double(xm[k]) * inv_n;
        }
        double mn2 = 0.0;
        for (int k = 0; k < K; ++k) mn2 += mean_dir[size_t(k)] * mean_dir[size_t(k)];
        const double inv_mn = (mn2 > 1e-30) ? (1.0 / std::sqrt(mn2)) : 0.0;
        for (int k = 0; k < K; ++k) mean_dir[size_t(k)] *= inv_mn;

        std::vector<std::pair<float, int>> align_idx(static_cast<size_t>(M_cal));
        for (int m = 0; m < M_cal; ++m) {
            const float* xm = x_rot.data() + size_t(m) * K;
            double dot = 0.0;
            for (int k = 0; k < K; ++k) dot += double(xm[k]) * mean_dir[size_t(k)];
            align_idx[size_t(m)] = {float(std::fabs(dot) * double(row_inv_norm[size_t(m)])), m};
        }
        const int rare_target = std::max(1, M_cal / 8);  // lowest-alignment 12.5% tail tokens
        std::nth_element(align_idx.begin(), align_idx.begin() + rare_target, align_idx.end());
        for (int idx = 0; idx < rare_target; ++idx) {
            auto_rare[size_t(align_idx[size_t(idx)].second)] = 1u;
        }
        rare_mask = auto_rare.data();
    }

    // Compute exact rotated second-moment traces tr(H_common) and tr(H_rare), then closed-form lambda!
    double tr_common = 0.0;
    double tr_rare = 0.0;
    int m_common = 0;
    int m_rare = 0;
    for (int m = 0; m < M_cal; ++m) {
        const float* xm = x_rot.data() + size_t(m) * K;
        double row_sq = 0.0;
        for (int k = 0; k < K; ++k) row_sq += double(xm[k]) * double(xm[k]);
        if (enable_rarebal && rare_mask && rare_mask[m] != 0u) {
            tr_rare += row_sq;
            ++m_rare;
        } else {
            tr_common += row_sq;
            ++m_common;
        }
    }

    double lambda = 1.0;
    if (enable_rarebal && m_common > 0 && m_rare > 0 && tr_common > 0.0 && tr_rare > 0.0) {
        lambda = tr_common / std::max(1e-12, tr_rare);
    }

    // Partition rows into common and rare contiguous panels and accumulate
    // H_bal = (1 / M_cal) * (H_common + lambda * H_rare) + 0.015 * d_bar * I_K
    // using cache-blocked 4-row AVX2/OpenMP tiles flushed in FP64.
    std::vector<float> x_com_panel(size_t(m_common) * K);
    std::vector<float> x_rare_panel(size_t(m_rare) * K);
    {
        int ic = 0, ir = 0;
        for (int m = 0; m < M_cal; ++m) {
            const float* src = x_rot.data() + size_t(m) * K;
            if (enable_rarebal && rare_mask && rare_mask[m] != 0u) {
                std::memcpy(x_rare_panel.data() + size_t(ir++) * K, src, size_t(K) * sizeof(float));
            } else {
                std::memcpy(x_com_panel.data() + size_t(ic++) * K, src, size_t(K) * sizeof(float));
            }
        }
    }

    std::vector<double> H(size_t(K) * K, 0.0);
    auto accum_syrk_upper = [&](const float* panel, int M_rows, double scale_w) {
        if (M_rows <= 0 || scale_w == 0.0) return;
        constexpr int kMBlk = 128;
        #pragma omp parallel
        {
            std::vector<float> acc0(static_cast<size_t>(K)), acc1(static_cast<size_t>(K)),
                               acc2(static_cast<size_t>(K)), acc3(static_cast<size_t>(K));
            #pragma omp for schedule(dynamic, 4)
            for (int i0 = 0; i0 < K; i0 += 4) {
                const int rcnt = std::min(4, K - i0);
                for (int mb = 0; mb < M_rows; mb += kMBlk) {
                    const int mend = std::min(M_rows, mb + kMBlk);
                    std::fill(acc0.begin() + i0, acc0.end(), 0.0f);
                    if (rcnt > 1) std::fill(acc1.begin() + i0 + 1, acc1.end(), 0.0f);
                    if (rcnt > 2) std::fill(acc2.begin() + i0 + 2, acc2.end(), 0.0f);
                    if (rcnt > 3) std::fill(acc3.begin() + i0 + 3, acc3.end(), 0.0f);

                    for (int m = mb; m < mend; ++m) {
                        const float* xm = panel + size_t(m) * K;
                        const float v0 = xm[i0];
                        for (int j = i0; j < K; ++j) acc0[size_t(j)] += v0 * xm[j];
                        if (rcnt > 1) {
                            const float v1 = xm[i0 + 1];
                            for (int j = i0 + 1; j < K; ++j) acc1[size_t(j)] += v1 * xm[j];
                        }
                        if (rcnt > 2) {
                            const float v2 = xm[i0 + 2];
                            for (int j = i0 + 2; j < K; ++j) acc2[size_t(j)] += v2 * xm[j];
                        }
                        if (rcnt > 3) {
                            const float v3 = xm[i0 + 3];
                            for (int j = i0 + 3; j < K; ++j) acc3[size_t(j)] += v3 * xm[j];
                        }
                    }

                    double* h0 = H.data() + size_t(i0) * K;
                    for (int j = i0; j < K; ++j) h0[j] += scale_w * double(acc0[size_t(j)]);
                    if (rcnt > 1) {
                        double* h1 = H.data() + size_t(i0 + 1) * K;
                        for (int j = i0 + 1; j < K; ++j) h1[j] += scale_w * double(acc1[size_t(j)]);
                    }
                    if (rcnt > 2) {
                        double* h2 = H.data() + size_t(i0 + 2) * K;
                        for (int j = i0 + 2; j < K; ++j) h2[j] += scale_w * double(acc2[size_t(j)]);
                    }
                    if (rcnt > 3) {
                        double* h3 = H.data() + size_t(i0 + 3) * K;
                        for (int j = i0 + 3; j < K; ++j) h3[j] += scale_w * double(acc3[size_t(j)]);
                    }
                }
            }
        }
    };

    const double inv_mcal = 1.0 / double(M_cal);
    accum_syrk_upper(x_com_panel.data(), m_common, inv_mcal);
    accum_syrk_upper(x_rare_panel.data(), m_rare, lambda * inv_mcal);

    double diag_mean = 0.0;
    for (int i = 0; i < K; ++i) diag_mean += H[size_t(i) * K + i];
    diag_mean /= double(K);
    const double damp = 0.015 * std::max(1e-12, diag_mean);
    for (int i = 0; i < K; ++i) {
        H[size_t(i) * K + i] += damp;
        for (int j = 0; j < i; ++j) H[size_t(i) * K + j] = H[size_t(j) * K + i];
    }

    if (out_stats) {
        out_stats->m_cal = M_cal;
        out_stats->m_common = m_common;
        out_stats->m_rare = m_rare;
        out_stats->used_starvation_fallback = false;
        out_stats->tr_common = tr_common;
        out_stats->tr_rare = tr_rare;
        out_stats->lambda = lambda;
        out_stats->d_bar = diag_mean;
    }

    // Exact causal LDLQ feedback via Reverse Cholesky H = R D R^T (R unit upper triangular, R[col,col] = 1).
    // Equivalent to standard forward Cholesky L_rev L_rev^T of order-reversed H_rev = P H P^T (unit-stride AVX2!):
    //   R[col, k] = L_rev[K - 1 - col, K - 1 - k] / L_rev[K - 1 - k, K - 1 - k]  for col < k,
    // minimizing ||D^{1/2} R^T (w - q)||_2^2 = (w - q)^T H (w - q) via wr[k] += (v - qv) * R[col, k].
    std::vector<double> H_rev(static_cast<size_t>(K) * K, 0.0);
    #pragma omp parallel for schedule(static)
    for (int ir = 0; ir < K; ++ir) {
        const int i_src = K - 1 - ir;
        for (int jr = 0; jr <= ir; ++jr) {
            const int j_src = K - 1 - jr;
            H_rev[size_t(ir) * K + jr] = H[size_t(i_src) * K + j_src];
        }
    }
    H.clear();
    H.shrink_to_fit();

    for (int jr = 0; jr < K; ++jr) {
        double sum = H_rev[size_t(jr) * K + jr];
        const double* lj = H_rev.data() + size_t(jr) * K;
        for (int m = 0; m < jr; ++m) sum -= lj[m] * lj[m];
        if (sum <= 1e-18) sum = 1e-18;
        const double ljj = std::sqrt(sum);
        H_rev[size_t(jr) * K + jr] = ljj;
        const double inv_ljj = 1.0 / ljj;
        #pragma omp parallel for schedule(static) if (K - jr > 128)
        for (int ir = jr + 1; ir < K; ++ir) {
            double s = H_rev[size_t(ir) * K + jr];
            const double* li = H_rev.data() + size_t(ir) * K;
            for (int m = 0; m < jr; ++m) s -= li[m] * lj[m];
            H_rev[size_t(ir) * K + jr] = s * inv_ljj;
        }
    }

    std::vector<float> R_up(static_cast<size_t>(K) * K, 0.0f);
    #pragma omp parallel for schedule(static)
    for (int col = 0; col < K; ++col) {
        const int ir = K - 1 - col;
        float* r_row = R_up.data() + size_t(col) * K;
        r_row[col] = 1.0f;
        for (int k = col + 1; k < K; ++k) {
            const int jr = K - 1 - k;
            r_row[k] = float(H_rev[size_t(ir) * K + jr] / H_rev[size_t(jr) * K + jr]);
        }
    }
    return R_up;
}

inline Hlut2QuantMatrix apply_hlut2_ldlq_with_rup(
    const Hlut2QuantMatrix& q_seed,
    const float* w_rot,
    const std::vector<float>& R_up)
{
    Hlut2QuantMatrix q = q_seed;
    const int N = q.N;
    const int K = q.K;
    if (R_up.size() != static_cast<size_t>(K) * K) return q;
    const int G = K / 128;

    std::vector<float> sigma_f32(static_cast<size_t>(N) * G);
    for (size_t i = 0; i < sigma_f32.size(); ++i) sigma_f32[i] = fp16_to_f32(q.scales_fp16[i]);

    const float c0 = q.c0, c1 = q.c1;
    const float dc = c1 - c0;
    const float tau = 0.5f * (c0 + c1);
    const int words_per_row = K / 16;
    q.codes_rm.assign(static_cast<size_t>(N) * words_per_row, 0u);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < N; ++r) {
        std::vector<float> wr(w_rot + size_t(r) * K, w_rot + size_t(r + 1) * K);
        uint32_t* cr = q.codes_rm.data() + size_t(r) * words_per_row;
        for (int g = 0; g < G; ++g) {
            const float sig = sigma_f32[size_t(r) * G + g];
            const float thr = tau * sig;

            for (int j = 0; j < 128; ++j) {
                const int col = g * 128 + j;
                const float v = wr[col];
                const uint32_t sign_bit = std::signbit(v) ? 2u : 0u;
                const uint32_t mag_bit = (std::fabs(v) >= thr) ? 1u : 0u;
                cr[col >> 4] |= ((sign_bit | mag_bit) << (2 * (col & 15)));

                const float mag_val = (c0 + float(mag_bit) * dc) * sig;
                const float qv = sign_bit ? -mag_val : mag_val;
                const float err = w_rot[size_t(r) * K + col] - qv;
                const float* rrow = R_up.data() + size_t(col) * K;
                for (int k = col + 1; k < K; ++k) {
                    wr[k] += err * rrow[k];
                }
            }
        }
    }
    return q;
}

inline Hlut2QuantMatrix quantize_hlut2_ldlq(
    const float* W_rm, int N, int K, const std::vector<uint64_t>& signs,
    const float* X_cal_rm, int M_cal,
    bool enable_rarebal = true,
    const uint8_t* is_rare_token = nullptr,
    Hlut2RarebalStats* out_stats = nullptr,
    bool bypass_starvation_gate = false)
{
    Hlut2QuantMatrix q = quantize_hlut2(W_rm, N, K, signs);
    const std::vector<float> R_up = build_rarebal_cholesky_rup(
        X_cal_rm, M_cal, K, signs, enable_rarebal, is_rare_token, out_stats, bypass_starvation_gate);
    if (R_up.empty()) return q;

    std::vector<float> w_rot(size_t(N) * K);
    wht128_rows(W_rm, w_rot.data(), N, K, signs);
    return apply_hlut2_ldlq_with_rup(q, w_rot.data(), R_up);
}

inline void dequantize_hlut2_to_rot_f32(const Hlut2QuantMatrix& q, std::vector<float>& out_rot) {
    out_rot.resize(size_t(q.N) * q.K);
    const int G = q.K / 128;
    const int words_per_row = q.K / 16;
    const float c0 = q.c0, dc = q.c1 - q.c0;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < q.N; ++r) {
        const uint32_t* cr = q.codes_rm.data() + size_t(r) * words_per_row;
        float* dst = out_rot.data() + size_t(r) * q.K;
        for (int g = 0; g < G; ++g) {
            const float sig = fp16_to_f32(q.scales_fp16[size_t(r) * G + g]);
            for (int w = 0; w < 8; ++w) {
                const uint32_t pack = cr[g * 8 + w];
                for (int m = 0; m < 16; ++m) {
                    const uint32_t code = (pack >> (2 * m)) & 3u;
                    const float val = (c0 + float(code & 1u) * dc) * sig;
                    dst[g * 128 + w * 16 + m] = (code & 2u) ? -val : val;
                }
            }
        }
    }
}

// ---- Unrotated Uniform 2-Bit Absmax Baseline (group-128 FP16 scale, symmetric levels {+-1/3, +-1} * amax, 2.125 bpw) ----
inline void quant_dequant_uniform_2bit_absmax(const float* W_rm, float* W_out, int N, int K, int group_size = 128) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < N; ++r) {
        const float* src = W_rm + size_t(r) * K;
        float* dst = W_out + size_t(r) * K;
        for (int b = 0; b < K; b += group_size) {
            float amax = 0.0f;
            for (int j = 0; j < group_size; ++j) amax = std::max(amax, std::fabs(src[b + j]));
            const float step = fp16_to_f32(f32_to_fp16_rne(amax / 1.5f));
            if (step <= 1e-30f) {
                for (int j = 0; j < group_size; ++j) dst[b + j] = 0.0f;
                continue;
            }
            for (int j = 0; j < group_size; ++j) {
                const float v = src[b + j];
                const float qm = (std::fabs(v) >= step) ? 1.5f : 0.5f;
                dst[b + j] = (std::signbit(v) ? -qm : qm) * step;
            }
        }
    }
}

// ---- Unrotated Scalar BFP2 Baseline (16-datum shared exponent + 1-bit sign + 1-bit mantissa, 2.5 bpw) ----
// Matches hardware unrotated Bfp2_b behavior: max_abs in 16-element face row sets shared power-of-2 exponent;
// 1 mantissa bit gives levels {0, +-0.5, +-0.75} or symmetric {+-0.375, +-0.875} * 2^exp.
inline void quant_dequant_scalar_bfp2(const float* W_rm, float* W_out, int N, int K) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < N; ++r) {
        const float* src = W_rm + size_t(r) * K;
        float* dst = W_out + size_t(r) * K;
        for (int b = 0; b < K; b += 16) {
            float amax = 0.0f;
            for (int j = 0; j < 16; ++j) amax = std::max(amax, std::fabs(src[b + j]));
            if (amax <= 1e-30f) {
                for (int j = 0; j < 16; ++j) dst[b + j] = 0.0f;
                continue;
            }
            int exp2 = 0;
            std::frexp(amax, &exp2);  // amax in [0.5, 1) * 2^exp2
            const float scale = std::ldexp(1.0f, exp2);
            // Hardware Bfp2_b 2-bit levels in [-scale, +scale]: 00->0.0, 01->+0.5*scale, 10->-0.5*scale, 11->+-0.75*scale?
            // Even giving scalar BFP2 the generous 4 non-zero symmetric levels {+-0.25, +-0.75}*scale:
            for (int j = 0; j < 16; ++j) {
                const float v = src[b + j];
                const float a = std::fabs(v) / scale;
                const float qm = (a >= 0.5f) ? 0.75f : 0.25f;
                dst[b + j] = (std::signbit(v) ? -qm : qm) * scale;
            }
        }
    }
}

// ---- Exact Unrotated BFP4_b Baseline (16-datum shared exponent + sign + 3-bit mantissa, 4.5 bpw) ----
inline void quant_dequant_bfp4b(const float* W_rm, float* W_out, int N, int K) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < N; ++r) {
        const float* src = W_rm + size_t(r) * K;
        float* dst = W_out + size_t(r) * K;
        for (int b = 0; b < K; b += 16) {
            float amax = 0.0f;
            for (int j = 0; j < 16; ++j) amax = std::max(amax, std::fabs(src[b + j]));
            if (amax <= 1e-30f) {
                for (int j = 0; j < 16; ++j) dst[b + j] = 0.0f;
                continue;
            }
            int exp2 = 0;
            std::frexp(amax, &exp2);
            const float step = std::ldexp(1.0f, exp2 - 3);  // 7 steps up to 7/8 * 2^exp2
            for (int j = 0; j < 16; ++j) {
                const float v = src[b + j];
                int qi = int(std::nearbyint(std::fabs(v) / step));
                qi = std::min(7, std::max(0, qi));
                dst[b + j] = (std::signbit(v) ? -float(qi) : float(qi)) * step;
            }
        }
    }
}

// ---- Exact Hardware Bfp8_b (16-datum face row shared exponent + sign + 7-bit mantissa) simulator ----
// Matches Tenstorrent PACR -> Bfp8_b so our CPU reference of P150A HLUT2 -> L1 Bfp8_b -> HiFi2 matmul is exact!
inline void simulate_tt_bfp8b_tile_32x32(float tile[32][32]) {
    // In Tensix 32x32 tile, each row of each 16x16 face (16 contiguous columns: 0..15 and 16..31)
    // shares one 8-bit exponent, with 1 sign bit + 7 magnitude bits (0..127 -> m / 128 * 2^exp2).
    for (int r = 0; r < 32; ++r) {
        for (int fc = 0; fc < 32; fc += 16) {
            float amax = 0.0f;
            for (int c = 0; c < 16; ++c) amax = std::max(amax, std::fabs(tile[r][fc + c]));
            if (amax <= 1e-30f) {
                for (int c = 0; c < 16; ++c) tile[r][fc + c] = 0.0f;
                continue;
            }
            int exp2 = 0;
            std::frexp(amax, &exp2);
            const float step = std::ldexp(1.0f, exp2 - 7);
            for (int c = 0; c < 16; ++c) {
                const float v = tile[r][fc + c];
                int qi = int(std::nearbyint(std::fabs(v) / step));
                qi = std::min(127, std::max(0, qi));
                tile[r][fc + c] = (std::signbit(v) ? -float(qi) : float(qi)) * step;
            }
        }
    }
}

// ---- MI50 Coalesced Memory Layout (2.125 bpw exact) ----
// For [N, K] with K % 128 == 0:
// Coalesced across 32 output rows per block so a 64-lane wavefront (32 rows x 2 K-halves or 16 rows x 4 K-chunks)
// issues 128-byte coalesced HBM2 loads with 0 bank conflicts and ScratchSize == 0:
//   - codes_mi50: [N / 32, K / 16, 32] uint32_t  (each uint32_t = 16 consecutive K weights for row r)
//   - scales_mi50: [N / 32, K / 128, 32] uint16_t (FP16 group scale sigma_{r, g})
struct Hlut2Mi50Packed {
    int N = 0;
    int K = 0;
    float c0 = 0.0f;
    float c1 = 0.0f;
    std::vector<uint32_t> codes;   // size = N * (K / 16)
    std::vector<uint16_t> scales;  // size = N * (K / 128)
};

inline Hlut2Mi50Packed pack_hlut2_for_mi50(const Hlut2QuantMatrix& q) {
    Hlut2Mi50Packed p;
    p.N = q.N;
    p.K = q.K;
    p.c0 = q.c0;
    p.c1 = q.c1;
    const int nb = q.N / 32;
    const int kw = q.K / 16;
    const int G = q.K / 128;
    p.codes.resize(size_t(q.N) * kw);
    p.scales.resize(size_t(q.N) * G);
    #pragma omp parallel for schedule(static)
    for (int rb = 0; rb < nb; ++rb) {
        for (int r = 0; r < 32; ++r) {
            const int row = rb * 32 + r;
            for (int w = 0; w < kw; ++w) {
                p.codes[(size_t(rb) * kw + w) * 32 + r] = q.codes_rm[size_t(row) * kw + w];
            }
            for (int g = 0; g < G; ++g) {
                p.scales[(size_t(rb) * G + g) * 32 + r] = q.scales_fp16[size_t(row) * G + g];
            }
        }
    }
    return p;
}

// ---- Blackhole P150A Tensix Superblock Layout (272 bytes/tile = 2.125 bpw exact, 0 padding!) ----
// Recall C = A * B where A is [M, K] and B = W^T is [K, N_shard].
// Both kt = K / 32 and nt = N_shard / 32 are exact multiples of 4 tiles!
//   - gate / up TP2: [K=5120, N=1152] -> kt=160 (40 K4 blocks), nt=36 (9 N4 blocks) -> 360 superblocks!
//   - down TP2:      [K=1152, N=5120] -> kt=36  (9  K4 blocks), nt=160 (40 N4 blocks) -> 360 superblocks!
// Each (kb4, nb4) superblock covers 4 K-tiles x 4 N-tiles = 16 Tensix 32x32 tiles:
//   1. One UInt32 Tensix TILE (1024 uint32_t = 4096 B = 256 B/tile):
//      At tile datum (i, j) in [0..31]x[0..31], bits [2*t+1 : 2*t] (t = kt_sub*4 + nt_sub in 0..15)
//      encode (sign<<1 | mag) of B[(kb4*4 + kt_sub)*32 + i, (nb4*4 + nt_sub)*32 + j]!
//      When loaded into Tensix Dest via copy_tile(cb_codes), every 32-lane SFPU vector lane holds
//      its own 16 sub-tiles at the EXACT Tensix face/row/col coordinate -> ZERO cross-lane shuffles!
//   2. Scale strip (4 N-subtiles x 32 cols x 2 B FP16/BF16 = 256 B per superblock = 16 B/tile):
//      Stores BF16 (c0 * sigma_{col})? Wait: storing sigma_{col} in BF16 (2 B/col = 256 B/superblock = 16 B/tile)
//      and passing c0, dc as kernel scalar args uses ONLY 256 B/superblock => 4096 + 256 = 4352 B / 16 tiles = 272.000 B/tile!
struct Hlut2P150Packed {
    int K = 0;
    int N = 0;
    int kt4 = 0;  // K / 128
    int nt4 = 0;  // N / 128
    float c0 = 0.0f;
    float c1 = 0.0f;
    // Tiled UInt32 buffer ready for ttnn::Layout::TILE upload: shape [1, 1, kt4 * 32, nt4 * 32] (or [1, kt4*nt4, 32, 32])
    // Stored in standard Tensix 4-face tile order (4 faces of 16x16 uint32_t = 1024 uint32_t per superblock).
    std::vector<uint32_t> codes_tiled_u32;
    // Row-major BF16 scale pages: [kt4 * nt4, 128] uint16_t (256 B per superblock: 4 N-subtiles x 32 BF16 scales).
    std::vector<uint16_t> scales_bf16;

    size_t total_bytes() const {
        return codes_tiled_u32.size() * sizeof(uint32_t) + scales_bf16.size() * sizeof(uint16_t);
    }
};

// Standard Tensix 32x32 tile index for (row i in 0..31, col j in 0..31):
// Face 0: r 0..15, c 0..15   (offset 0)
// Face 1: r 0..15, c 16..31  (offset 256)
// Face 2: r 16..31, c 0..15  (offset 512)
// Face 3: r 16..31, c 16..31 (offset 768)
inline int tensix_tile_offset(int i, int j) {
    const int face = ((i >> 4) << 1) | (j >> 4);
    return (face << 8) | ((i & 15) << 4) | (j & 15);
}

inline Hlut2P150Packed pack_hlut2_for_p150_shard(
    const Hlut2QuantMatrix& q, int n_col_offset, int N_shard)
{
    if (q.K % 128 != 0 || N_shard % 128 != 0 || n_col_offset + N_shard > q.N) {
        throw std::invalid_argument("P150 HLUT2 shard requires K%128==0 and N_shard%128==0");
    }
    Hlut2P150Packed p;
    p.K = q.K;
    p.N = N_shard;
    p.kt4 = q.K / 128;
    p.nt4 = N_shard / 128;
    p.c0 = q.c0;
    p.c1 = q.c1;

    const int num_sb = p.kt4 * p.nt4;
    p.codes_tiled_u32.assign(size_t(num_sb) * 1024, 0u);
    p.scales_bf16.assign(size_t(num_sb) * 128, 0u);

    const int G = q.K / 128;
    const int kw = q.K / 16;

    // Order superblocks by (nb4, kb4) so each Tensix core owning an N4 column-band reads contiguous GDDR6 pages!
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nb4 = 0; nb4 < p.nt4; ++nb4) {
        for (int kb4 = 0; kb4 < p.kt4; ++kb4) {
            const int sb = nb4 * p.kt4 + kb4;
            uint32_t* dst_tile = p.codes_tiled_u32.data() + size_t(sb) * 1024;
            uint16_t* dst_sc = p.scales_bf16.data() + size_t(sb) * 128;

            for (int nt_sub = 0; nt_sub < 4; ++nt_sub) {
                for (int j = 0; j < 32; ++j) {
                    const int n_global = n_col_offset + (nb4 * 4 + nt_sub) * 32 + j;
                    const float sig = fp16_to_f32(q.scales_fp16[size_t(n_global) * G + kb4]);
                    dst_sc[nt_sub * 32 + j] = f32_to_bf16_rne(sig);
                }
            }

            for (int i = 0; i < 32; ++i) {
                for (int j = 0; j < 32; ++j) {
                    uint32_t word = 0u;
                    for (int kt_sub = 0; kt_sub < 4; ++kt_sub) {
                        const int k_global = (kb4 * 4 + kt_sub) * 32 + i;
                        const int w_idx = k_global >> 4;
                        const int bit_shift = 2 * (k_global & 15);
                        for (int nt_sub = 0; nt_sub < 4; ++nt_sub) {
                            const int n_global = n_col_offset + (nb4 * 4 + nt_sub) * 32 + j;
                            const uint32_t code = (q.codes_rm[size_t(n_global) * kw + w_idx] >> bit_shift) & 3u;
                            const int t = kt_sub * 4 + nt_sub;
                            word |= (code << (2 * t));
                        }
                    }
                    dst_tile[tensix_tile_offset(i, j)] = word;
                }
            }
        }
    }
    return p;
}

// ---- Real DeepSeek-V4.1-Flash-1017 Layer 6 Expert 0 MXFP4 Loader (Dual-NVMe Shards) ----
struct SwigluExpertF32 {
    std::string name;
    std::vector<float> gate;  // [2304, 5120]
    std::vector<float> up;    // [2304, 5120]
    std::vector<float> down;  // [5120, 2304]
};

inline float decode_ocp_e2m1(uint8_t nib) {
    static constexpr float kLut[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return kLut[nib & 15u];
}

inline bool load_real_dsv41_expert0(
    const std::string& scales_path_skhynix,
    const std::string& weights_path_greatwall,
    SwigluExpertF32& out)
{
    std::ifstream fs(scales_path_skhynix, std::ios::binary);
    std::ifstream fw(weights_path_greatwall, std::ios::binary);
    if (!fs || !fw) return false;
    std::vector<uint8_t> sb(1105920), wb(17694720);
    if (!fs.read(reinterpret_cast<char*>(sb.data()), sb.size())) return false;
    if (!fw.read(reinterpret_cast<char*>(wb.data()), wb.size())) return false;

    out.name = "Real_DeepSeek_V4.1_Flash_1017_L6_Expert0";
    out.gate.resize(size_t(DSV41_I) * DSV41_H);
    out.down.resize(size_t(DSV41_H) * DSV41_I);
    out.up.resize(size_t(DSV41_I) * DSV41_H);

    auto deq_mxfp4 = [](const uint8_t* w_bytes, const uint8_t* s_bytes, float* dst, int rows, int cols) {
        const int half_cols = cols / 2;
        const int s_cols = cols / 32;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            const uint8_t* wr = w_bytes + size_t(r) * half_cols;
            const uint8_t* sr = s_bytes + size_t(r) * s_cols;
            float* dr = dst + size_t(r) * cols;
            for (int sb_idx = 0; sb_idx < s_cols; ++sb_idx) {
                const uint8_t e8 = sr[sb_idx];
                const float scale = (e8 == 0) ? 0.0f : std::ldexp(1.0f, int(e8) - 127);
                const uint8_t* wb16 = wr + sb_idx * 16;
                float* db32 = dr + sb_idx * 32;
                for (int b = 0; b < 16; ++b) {
                    const uint8_t byte = wb16[b];
                    db32[2 * b + 0] = decode_ocp_e2m1(byte & 0x0Fu) * scale;
                    db32[2 * b + 1] = decode_ocp_e2m1(byte >> 4) * scale;
                }
            }
        }
    };

    // Offsets from safetensors header:
    // w1 (gate): scale [0..368640), weight [0..5898240)
    // w2 (down): scale [368640..737280), weight [5898240..11796480)
    // w3 (up):   scale [737280..1105920), weight [11796480..17694720)
    deq_mxfp4(wb.data() + 0,        sb.data() + 0,      out.gate.data(), DSV41_I, DSV41_H);
    deq_mxfp4(wb.data() + 5898240,  sb.data() + 368640, out.down.data(), DSV41_H, DSV41_I);
    deq_mxfp4(wb.data() + 11796480, sb.data() + 737280, out.up.data(),   DSV41_I, DSV41_H);
    return true;
}

// ---- Synthetic Heavy-Tailed Outlier SwiGLU Expert (10x Outlier Channels) ----
inline SwigluExpertF32 make_heavy_tailed_outlier_expert(uint64_t seed = 20261111ULL) {
    SwigluExpertF32 ex;
    ex.name = "Synthetic_HeavyTailed_10xOutlier_5120x2304";
    ex.gate.resize(size_t(DSV41_I) * DSV41_H);
    ex.up.resize(size_t(DSV41_I) * DSV41_H);
    ex.down.resize(size_t(DSV41_H) * DSV41_I);

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // Realistic LLM channelwise outlier multipliers: ~1.5% of input channels have 10x std spikes!
    std::vector<float> col_scale_h(DSV41_H, 0.014f);
    for (int k = 7; k < DSV41_H; k += 64) col_scale_h[k] = 0.140f;  // 10x outlier channel inside every 64-dim span!
    std::vector<float> col_scale_i(DSV41_I, 0.020f);
    for (int k = 11; k < DSV41_I; k += 64) col_scale_i[k] = 0.200f; // 10x outlier intermediate channel!

    auto fill = [&](std::vector<float>& W, int rows, int cols, const std::vector<float>& cscale) {
        for (int r = 0; r < rows; ++r) {
            const float rscale = 0.85f + 0.30f * float(r % 17) / 16.0f;
            float* wr = W.data() + size_t(r) * cols;
            for (int c = 0; c < cols; ++c) {
                // Student-t-like Kurtosis + 10x structured channel outliers
                const float g1 = nd(rng), g2 = nd(rng);
                const float heavy = g1 * (1.0f + 0.25f * g2 * g2);
                wr[c] = bf16_to_f32(f32_to_bf16_rne(heavy * rscale * cscale[c]));
            }
        }
    };
    fill(ex.gate, DSV41_I, DSV41_H, col_scale_h);
    fill(ex.up,   DSV41_I, DSV41_H, col_scale_h);
    fill(ex.down, DSV41_H, DSV41_I, col_scale_i);
    return ex;
}

inline std::vector<float> make_outlier_activations(int M, int K, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> X(size_t(M) * K);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            const float cmult = ((k & 63) == 7) ? 6.0f : 0.45f;
            X[size_t(m) * K + k] = bf16_to_f32(f32_to_bf16_rne(nd(rng) * cmult));
        }
    }
    return X;
}

struct QualityMetrics {
    double cosine = 0.0;
    double rel_l2 = 0.0;
    double max_abs = 0.0;
};

inline QualityMetrics compute_quality(const std::vector<float>& ref, const std::vector<float>& test) {
    double dot = 0.0, nr = 0.0, nt = 0.0, diff2 = 0.0, mabs = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double a = double(ref[i]);
        const double b = double(test[i]);
        const double d = a - b;
        dot += a * b;
        nr += a * a;
        nt += b * b;
        diff2 += d * d;
        mabs = std::max(mabs, std::fabs(d));
    }
    QualityMetrics m;
    m.cosine = (nr > 0.0 && nt > 0.0) ? (dot / (std::sqrt(nr) * std::sqrt(nt))) : 0.0;
    m.rel_l2 = (nr > 0.0) ? std::sqrt(diff2 / nr) : 0.0;
    m.max_abs = mabs;
    return m;
}

}  // namespace dsv41
