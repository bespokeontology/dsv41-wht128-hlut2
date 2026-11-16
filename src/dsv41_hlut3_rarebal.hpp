#pragma once
// Wave 9 Deliverable B: TARQ RAREBAL-LDLQ importance-weighted reverse-Cholesky quantizer for
// WHT128-HLUT3. Exact global K-wide Hessian H = X^T W_rare X + lambda_reg * I, reverse-order
// LDLQ error feedback onto the fixed 8-centroid Gaussian Lloyd-Max lattice.
//
// The Wave-9 Phase-1 128-wide BLOCK Hessian ignored all cross-block correlations H[j,k] between
// distinct 128-column panels. On real rare-token activations that over-corrects the lattice and
// catastrophically inflates rare MSE. The production law is the global K-wide Hessian:
//   H = sum_m w_m x_rot[m,:]^T x_rot[m,:] + 0.015 * d_bar * I_K   (d_bar = weighted diag mean)
// followed by the exact order-reversed Cholesky H_rev = P H P^T and
//   R[col,k] = L_rev[K-1-col, K-1-k] / L_rev[K-1-k, K-1-k]   (col <= k)
// The quantizer is 100% bit-exact with src/hlut2_pack.hpp::build_rarebal_cholesky_rup (the
// production Wave-8 TARQ Hessian/cholesky law), then the HLUT3 4-positive-centroid lattice is
// applied with causal error feedback wr[k] += (wr[col] - qv) * R[col,k].
//
// Dual Receipt law:
//   HLUT3+RAREBAL must beat HLUT3(H=I) on held-out SwiGLU cosine AND rare-token MSE,
//   and beat HLUT2+RAREBAL SNR by >= +4.0 dB. Bench tools enforce this on real weights.
//
// No Python. No magic floats beyond the production TARQ damping law 0.015 * d_bar.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hlut3_pack.hpp"

namespace dsv41 {

struct Hlut3RarebalStats {
    int m_cal = 0;
    int m_common = 0;
    int m_rare = 0;
    bool used_starvation_fallback = false;
    double tr_common = 0.0;
    double tr_rare = 0.0;
    double lambda_tarq = 1.0;     // trace equalization lambda = tr_common / tr_rare
    double d_bar = 0.0;           // weighted diagonal mean
    double lambda_reg = 0.0;      // 0.015 * d_bar (documented production damping)
};

inline std::vector<uint8_t> hlut3_auto_rare_mask(
    const float* x_rot,
    int M_cal,
    int K)
{
    std::vector<uint8_t> rare(size_t(M_cal), 0u);
    if (M_cal <= 0 || K <= 0) return rare;
    std::vector<double> mean_dir(size_t(K), 0.0);
    std::vector<float> row_inv_norm(size_t(M_cal), 0.0f);
    for (int m = 0; m < M_cal; ++m) {
        const float* xm = x_rot + size_t(m) * size_t(K);
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

    std::vector<std::pair<float, int>> align(static_cast<size_t>(M_cal));
    for (int m = 0; m < M_cal; ++m) {
        const float* xm = x_rot + size_t(m) * size_t(K);
        double dot = 0.0;
        for (int k = 0; k < K; ++k) dot += double(xm[k]) * mean_dir[size_t(k)];
        align[size_t(m)] = {float(std::fabs(dot) * double(row_inv_norm[size_t(m)])), m};
    }
    const int rare_target = std::max(1, M_cal / 8);  // lowest-alignment 12.5% tail tokens
    std::nth_element(align.begin(), align.begin() + rare_target, align.end());
    for (int i = 0; i < rare_target; ++i) rare[size_t(align[size_t(i)].second)] = 1u;
    return rare;
}

// Overwrite the 2-bit magnitude code and 1-bit sign plane cell. OR-ing onto the seed planes
// (quantize_hlut3's independent H=I codes) would leave stale bits: e.g. seed mag=2 (10b) OR new
// mag=1 (01b) becomes 11b=3. LDLQ MUST clear the cell before writing the final feedback code.
inline void hlut3_pack_codes_to_seed_tile(
    Hlut3Tile32x32& tile,
    int local_row,
    int local_col,
    uint32_t mag,
    uint32_t sign) noexcept
{
    const int idx = hlut3_tile_flat(local_row, local_col);
    const int mw = hlut3_mag_word(idx);
    const int ms = hlut3_mag_shift(idx);
    tile.q_lo2[mw] = (tile.q_lo2[mw] & ~(3u << ms)) | (mag << ms);
    const int sw = hlut3_sign_word(idx);
    const int ss = hlut3_sign_shift(idx);
    tile.q_hi1[sw] = (tile.q_hi1[sw] & ~(1u << ss)) | (sign << ss);
}

// Global K-wide importance-weighted Hessian + exact order-reversed Cholesky. Delegates the
// production Hessian/cholesky accumulation to src/hlut2_pack.hpp::build_rarebal_cholesky_rup
// so both tiers are bit-for-bit equal on the same X_cal, signs, rare mask, and damping law.
inline std::vector<float> build_hlut3_rarebal_cholesky_rup(
    const float* X_cal_rm,
    int M_cal,
    int K,
    const std::vector<uint64_t>& signs,
    bool enable_rarebal = true,
    const uint8_t* is_rare_token = nullptr,
    Hlut3RarebalStats* out_stats = nullptr,
    bool bypass_starvation_gate = false)
{
    if (out_stats) *out_stats = Hlut3RarebalStats{};
    Hlut2RarebalStats st2{};
    std::vector<float> R_up = build_rarebal_cholesky_rup(
        X_cal_rm, M_cal, K, signs, enable_rarebal, is_rare_token, &st2, bypass_starvation_gate);
    if (out_stats) {
        out_stats->m_cal = st2.m_cal;
        out_stats->m_common = st2.m_common;
        out_stats->m_rare = st2.m_rare;
        out_stats->used_starvation_fallback = st2.used_starvation_fallback;
        out_stats->tr_common = st2.tr_common;
        out_stats->tr_rare = st2.tr_rare;
        out_stats->lambda_tarq = st2.lambda;
        out_stats->d_bar = st2.d_bar;
        out_stats->lambda_reg = 0.015 * std::max(1e-12, st2.d_bar);
    }
    return R_up;
}

// Apply the fixed HLUT3 8-centroid Gaussian lattice with global causal LDLQ error feedback.
// Each row starts from the exact rotated weight row. The 32-column tile RMS scale for the row's
// 4-row group is preserved bit-exactly from the H=I seed tiles (272/400 B layout unchanged).
inline Hlut3QuantMatrix apply_hlut3_ldlq_with_rup(
    const Hlut3QuantMatrix& q_seed,
    const float* w_rot,
    const std::vector<float>& R_up,
    bool use_original_residual = true)
{
    Hlut3QuantMatrix q = q_seed;
    const int N = q.N;
    const int K = q.K;
    if (N <= 0 || K <= 0 || w_rot == nullptr ||
        R_up.size() != static_cast<size_t>(K) * static_cast<size_t>(K)) {
        return q;
    }
    if (q.tiles.size() != size_t(q.nt) * size_t(q.kt)) return q;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < N; ++r) {
        std::vector<float> wr(w_rot + size_t(r) * size_t(K),
                              w_rot + size_t(r + 1) * size_t(K));
        const int nt_i = r / kHlut3TileRows;
        const int local_r = r % kHlut3TileRows;
        const int scale_g = hlut3_scale_group_for_row(local_r);
        const float* orig_row = w_rot + size_t(r) * size_t(K);

        for (int col = 0; col < K; ++col) {
            const int kt_i = col / kHlut3TileCols;
            const int local_c = col % kHlut3TileCols;
            Hlut3Tile32x32& tile = q.tiles[size_t(nt_i) * size_t(q.kt) + size_t(kt_i)];
            const float sigma = fp16_to_f32(tile.scales_f16[scale_g]);
            const double z = (sigma > 0.0f) ? double(std::fabs(wr[col])) / double(sigma) : 0.0;
            uint32_t mag = 0u;
            if (z >= q.lloyd.thresholds[2]) mag = 3u;
            else if (z >= q.lloyd.thresholds[1]) mag = 2u;
            else if (z >= q.lloyd.thresholds[0]) mag = 1u;
            else mag = 0u;
            const uint32_t sign = std::signbit(wr[col]) ? 1u : 0u;
            hlut3_pack_codes_to_seed_tile(tile, local_r, local_c, mag, sign);

            const float magnitude = float(q.lloyd.levels[mag]) * sigma;
            const float qv = sign ? -magnitude : magnitude;
            const float err = use_original_residual ? (orig_row[col] - qv) : (wr[col] - qv);
            const float* rrow = R_up.data() + size_t(col) * size_t(K);
            #pragma omp simd
            for (int k = col + 1; k < K; ++k) wr[k] += err * rrow[k];
        }
    }
    return q;
}

inline Hlut3QuantMatrix quantize_hlut3_rarebal_ldlq(
    const float* W_rm,
    int N,
    int K,
    const std::vector<uint64_t>& signs,
    const float* X_cal_rm,
    int M_cal,
    bool enable_rarebal = true,
    const uint8_t* is_rare_token = nullptr,
    Hlut3RarebalStats* out_stats = nullptr,
    bool bypass_starvation_gate = false)
{
    if (out_stats) *out_stats = Hlut3RarebalStats{};
    if (W_rm == nullptr || N <= 0 || K <= 0 || K % 128 != 0 || N % 32 != 0) {
        throw std::invalid_argument("HLUT3 RAREBAL quantize requires K%128==0 and N%32==0");
    }
    if (!X_cal_rm || M_cal <= 0 || (!bypass_starvation_gate && M_cal < 2 * K)) {
        if (out_stats) {
            out_stats->m_cal = std::max(0, M_cal);
            out_stats->used_starvation_fallback = true;
        }
        return quantize_hlut3(W_rm, N, K, signs);
    }

    Hlut3RarebalStats st{};
    const std::vector<float> R_up = build_hlut3_rarebal_cholesky_rup(
        X_cal_rm, M_cal, K, signs, enable_rarebal, is_rare_token, &st, bypass_starvation_gate);
    if (out_stats) *out_stats = st;
    if (R_up.empty()) return quantize_hlut3(W_rm, N, K, signs);

    Hlut3QuantMatrix q = quantize_hlut3(W_rm, N, K, signs);
    std::vector<float> w_rot(size_t(N) * size_t(K));
    wht128_rows(W_rm, w_rot.data(), N, K, signs);
    return apply_hlut3_ldlq_with_rup(q, w_rot.data(), R_up, /*use_original_residual=*/true);
}

}  // namespace dsv41
