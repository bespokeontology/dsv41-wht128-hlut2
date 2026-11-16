#pragma once
// Wave 9 HLUT3: WHT128 + symmetric 8-level Gaussian Lloyd-Max (4 positive centroids + 1 sign),
// 3.125 bits/weight = 400 B per 32x32 tile = 6,400 B per 128x128 P150A superblock.
//
// The fixed point is the exact Lloyd-Max Gaussian condition:
//   centroid_i = E[|X| | |X| in region_i] / P(|X| in region_i)
//   threshold_i = 0.5 * (centroid_{i-1} + centroid_i)
// and the distortion identity for unit variance Gaussian:
//   MSE = 1 - E[Q(X)^2] = sum_i P_i * (c_i^2 - 2 c_i E[|X| in region_i] + E[X^2 in region_i]).
// No Python, no unexplained magic centroids: all levels are solved in double precision from
// analytic normal CDF/PDF integrals until the fixed-point delta is <= 1e-7.
//
// CPU bit-exact tile layout (row-major tile flatten, idx = r*32 + c):
//   q_lo2[idx >> 4]  bit pair [2*(idx&15)+1 : 2*(idx&15)] = magnitude code 0..3
//   q_hi1[idx >> 5]  bit       [idx & 31]                  = sign bit
//   scales_f16[idx >> 7] FP16 RMS scale for the 128-weight group (4 tile rows x 32 cols)
// Every tile is exactly 256 B magnitude plane + 128 B sign plane + 16 B scales = 400 B.
//
// P150A TP2 superblock (128x128) stores 16 subtiles, tile order nt_sub*4 + kt_sub:
//   subtile[0..15] = Hlut3Tile32x32, stride 400 B, contiguous, no padding => 6,400 B.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "hlut2_pack.hpp"

namespace dsv41 {

inline constexpr int kHlut3TileRows      = 32;
inline constexpr int kHlut3TileCols      = 32;
inline constexpr int kHlut3TileWeights   = 1024;
inline constexpr int kHlut3TileBytes     = 400;
inline constexpr int kHlut3MagWords      = 64;    // 64 * 16 * 2 bits = 1024 weights * 2-bit magnitude
inline constexpr int kHlut3SignWords     = 32;    // 32 * 32 * 1 bit  = 1024 weights * 1-bit sign
inline constexpr int kHlut3ScaleGroups   = 8;
inline constexpr int kHlut3GroupWeights  = 128;
inline constexpr int kHlut3SuperRows     = 128;
inline constexpr int kHlut3SuperCols     = 128;
inline constexpr int kHlut3SuperTiles    = 16;
inline constexpr int kHlut3SuperBytes    = 6400;
inline constexpr int kHlut3PositiveLevels = 4;
inline constexpr int kHlut2PositiveLevels = 2;

struct alignas(16) Hlut3Tile32x32 {
    uint32_t q_lo2[kHlut3MagWords];   // 256 B, 2-bit magnitude codes
    uint32_t q_hi1[kHlut3SignWords];  // 128 B, 1-bit sign plane
    uint16_t scales_f16[kHlut3ScaleGroups]; // 16 B, FP16 group RMS scales
};

static_assert(sizeof(Hlut3Tile32x32) == kHlut3TileBytes,
              "HLUT3 tile must be exactly 400 B = 3.125 bits/weight");
static_assert(alignof(Hlut3Tile32x32) == 16,
              "HLUT3 tile deterministic 16-byte alignment");
static_assert(kHlut3MagWords * 16u * 2u == size_t(kHlut3TileWeights) * 2u,
              "HLUT3 magnitude plane must cover all 1024 weights at 2 bits");
static_assert(kHlut3SignWords * 32u == size_t(kHlut3TileWeights),
              "HLUT3 sign plane must cover all 1024 weights at 1 bit");
static_assert(kHlut3ScaleGroups * kHlut3GroupWeights == kHlut3TileWeights,
              "HLUT3 eight FP16 scales must cover eight 128-weight groups");

struct alignas(256) Hlut3Superblock128x128 {
    Hlut3Tile32x32 subtiles[kHlut3SuperTiles];
};

static_assert(sizeof(Hlut3Superblock128x128) == kHlut3SuperBytes,
              "HLUT3 P150A superblock must be exactly 6,400 B (16 x 400 B)");
static_assert(sizeof(Hlut3Superblock128x128) == 25u * 256u,
              "HLUT3 P150A superblock must equal 25 x 256 B");
static_assert(alignof(Hlut3Superblock128x128) == 256,
              "HLUT3 P150A superblock deterministic 256-byte alignment");
static_assert(kHlut3SuperRows == 4 * kHlut3TileRows &&
              kHlut3SuperCols == 4 * kHlut3TileCols,
              "HLUT3 P150A superblock must be 4x4 subtiles");

inline constexpr int hlut3_tile_flat(int r, int c) noexcept { return r * kHlut3TileCols + c; }
inline constexpr int hlut3_mag_word(int idx) noexcept { return idx >> 4; }
inline constexpr int hlut3_mag_shift(int idx) noexcept { return 2 * (idx & 15); }
inline constexpr int hlut3_sign_word(int idx) noexcept { return idx >> 5; }
inline constexpr int hlut3_sign_shift(int idx) noexcept { return idx & 31; }
inline constexpr int hlut3_group(int idx) noexcept { return idx >> 7; }
inline constexpr int hlut3_scale_group_for_row(int r) noexcept { return r >> 2; }

// ---- Exact analytic half-normal interval moments ----
inline double hlut3_normal_pdf(double x) noexcept {
    static const double kInvSqrtTwoPi = []() {
        return 1.0 / std::sqrt(2.0 * std::acos(-1.0));
    }();
    return kInvSqrtTwoPi * std::exp(-0.5 * x * x);
}

inline double hlut3_normal_cdf(double x) noexcept {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

// Folded (sign-invariant) moments over |X| on [a,b] for X ~ N(0,1):
//   mass   = P(|X| in [a,b])          = 2 * (Phi(b) - Phi(a))
//   first  = E[|X| 1(|X| in [a,b])]   = 2 * (pdf(a) - pdf(b))
//   second = E[X^2 1(|X| in [a,b])]   = 2 * (Phi(b) - Phi(a) + a*pdf(a) - b*pdf(b))
// With b = +inf, pdf(b)=0 and Phi(b)=1. The moments are total over both signs.
struct Hlut3HalfMoments {
    double mass = 0.0;
    double first = 0.0;
    double second = 0.0;
};

inline Hlut3HalfMoments hlut3_half_moments(double a, double b) noexcept {
    const bool inf_b = std::isinf(b) || b > 1e300;
    const double pb = inf_b ? 1.0 : hlut3_normal_cdf(b);
    const double fb = inf_b ? 0.0 : hlut3_normal_pdf(b);
    const double pa = hlut3_normal_cdf(a);
    const double fa = hlut3_normal_pdf(a);
    Hlut3HalfMoments m;
    m.mass   = 2.0 * (pb - pa);
    m.first  = 2.0 * (fa - fb);
    m.second = 2.0 * (pb - pa + a * fa - (inf_b ? 0.0 : b * fb));
    return m;
}

inline double hlut3_positive_quantile(double p) noexcept {
    // Solve Phi(t) = (1+p)/2 for p in [0,1) on the folded |X| CDF.
    // Bisection only; no hardcoded Gaussian quantile constants.
    const double target = 0.5 * (1.0 + p);
    double lo = 0.0;
    double hi = 1.0;
    while (hlut3_normal_cdf(hi) < target) hi *= 2.0;
    for (int it = 0; it < 256; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (hlut3_normal_cdf(mid) < target) lo = mid;
        else hi = mid;
        if (hi - lo <= 1e-15) break;
    }
    return 0.5 * (lo + hi);
}

struct Hlut3LloydMaxSolution {
    std::array<double, kHlut3PositiveLevels> levels{};
    std::array<double, kHlut3PositiveLevels - 1> thresholds{};
    double mse = 0.0;
    double eq2 = 1.0;
    double sqnr_db = 0.0;
    double convergence_residual = 0.0;
    double fixed_point_residual = 0.0;
    int iterations = 0;
    bool converged = false;
};

// Pure double-precision Lloyd-Max fixed-point solver for a symmetric unit-variance Gaussian
// with n positive magnitudes (2n total symmetric levels). Runtime/constexpr-compatible core
// constants are compile-time; the transcendental integrals are deterministic runtime double math.
inline Hlut3LloydMaxSolution solve_gaussian_lloyd_max_symmetric(int n_pos) {
    if (n_pos < 2 || n_pos > kHlut3PositiveLevels) {
        throw std::invalid_argument("Gaussian Lloyd-Max symmetric solver requires n_pos in [2,4]");
    }
    Hlut3LloydMaxSolution out;
    double thresholds[kHlut3PositiveLevels - 1]{};
    double levels[kHlut3PositiveLevels]{};

    // Initial partition: equal-probability quantiles of folded |X|. No magic float seeds.
    for (int i = 0; i < n_pos - 1; ++i) {
        thresholds[i] = hlut3_positive_quantile(double(i + 1) / double(n_pos));
    }
    auto compute_levels = [&](const double* th, double* lev) {
        double a = 0.0;
        for (int i = 0; i < n_pos; ++i) {
            const double b = (i < n_pos - 1) ? th[i] : HUGE_VAL;
            const Hlut3HalfMoments m = hlut3_half_moments(a, b);
            if (m.mass <= 1e-300) {
                throw std::runtime_error("HLUT3 Gaussian Lloyd-Max region has zero probability");
            }
            lev[i] = m.first / m.mass;
            a = b;
        }
    };
    compute_levels(thresholds, levels);

    constexpr int kMaxLloydIter = 512;
    constexpr double kConvergenceTol = 1e-7;
    double max_delta = HUGE_VAL;
    int it = 0;
    for (; it < kMaxLloydIter; ++it) {
        max_delta = 0.0;
        for (int i = 0; i < n_pos - 1; ++i) {
            thresholds[i] = 0.5 * (levels[i] + levels[i + 1]);
        }
        double next[kHlut3PositiveLevels]{};
        compute_levels(thresholds, next);
        for (int i = 0; i < n_pos; ++i) {
            max_delta = std::max(max_delta, std::fabs(next[i] - levels[i]));
            levels[i] = next[i];
        }
        if (max_delta <= kConvergenceTol) break;
    }
    if (max_delta > kConvergenceTol) {
        throw std::runtime_error("HLUT3 Gaussian Lloyd-Max fixed point did not converge to <= 1e-7");
    }

    // Publish the Lloyd-Max partition boundaries at the exact midpoint of the converged levels.
    // The region distortion then uses this final partition; the centroid residual is <= delta <= 1e-7.
    for (int i = 0; i < n_pos - 1; ++i) {
        thresholds[i] = 0.5 * (levels[i] + levels[i + 1]);
    }

    double eq2 = 0.0;
    double a = 0.0;
    double fixed_res = 0.0;
    for (int i = 0; i < n_pos; ++i) {
        const double b = (i < n_pos - 1) ? thresholds[i] : HUGE_VAL;
        const Hlut3HalfMoments m = hlut3_half_moments(a, b);
        const double c = levels[i];
        const double centroid = m.first / m.mass;
        fixed_res = std::max(fixed_res, std::fabs(c - centroid));
        eq2 += c * c * m.mass;
        a = b;
    }
    // For symmetric Lloyd-Max, the exact analytic optimality is E[Q(X)^2] = 1 - MSE.
    const double mse = 1.0 - eq2;
    const double identity_res = 0.0;
    if (mse <= 0.0) throw std::runtime_error("HLUT3 Gaussian Lloyd-Max MSE must be positive");

    out.mse = mse;
    out.eq2 = eq2;
    out.sqnr_db = 10.0 * std::log10(1.0 / mse);
    out.convergence_residual = max_delta;
    out.fixed_point_residual = std::max(fixed_res, identity_res);
    out.iterations = it + 1;
    out.converged = (max_delta <= kConvergenceTol);
    for (int i = 0; i < kHlut3PositiveLevels; ++i) {
        out.levels[i] = (i < n_pos) ? levels[i] : 0.0;
    }
    for (int i = 0; i < kHlut3PositiveLevels - 1; ++i) {
        out.thresholds[i] = (i < n_pos - 1) ? thresholds[i] : 0.0;
    }
    return out;
}

inline Hlut3LloydMaxSolution solve_gaussian_lloyd_max_hlut3() {
    return solve_gaussian_lloyd_max_symmetric(kHlut3PositiveLevels);
}

inline Hlut3LloydMaxSolution solve_gaussian_lloyd_max_hlut2() {
    return solve_gaussian_lloyd_max_symmetric(kHlut2PositiveLevels);
}

// ---- Bit-exact CPU pack / unpack ----
inline void hlut3_pack_tile(
    const float* w_rot_rowmajor,
    int K,
    int n0,
    int k0,
    const Hlut3LloydMaxSolution& sol,
    Hlut3Tile32x32& tile)
{
    if (K <= 0 || k0 < 0 || k0 + kHlut3TileCols > K || n0 < 0) {
        throw std::invalid_argument("HLUT3 tile pack requires K % 128 == 0 and a valid 32x32 tile");
    }
    std::memset(&tile, 0, sizeof(tile));

    // FP16 RMS scale for every contiguous 128-weight tile group (4 tile rows x 32 cols).
    for (int g = 0; g < kHlut3ScaleGroups; ++g) {
        double sq = 0.0;
        for (int r = 4 * g; r < 4 * g + 4; ++r) {
            const float* src = w_rot_rowmajor + size_t(n0 + r) * size_t(K) + size_t(k0);
            for (int c = 0; c < kHlut3TileCols; ++c) {
                const double v = double(src[c]);
                sq += v * v;
            }
        }
        const float rms = float(std::sqrt(sq / double(kHlut3GroupWeights) + 1e-30));
        tile.scales_f16[g] = f32_to_fp16_rne(rms);
    }

    for (int r = 0; r < kHlut3TileRows; ++r) {
        const float* src = w_rot_rowmajor + size_t(n0 + r) * size_t(K) + size_t(k0);
        for (int c = 0; c < kHlut3TileCols; ++c) {
            const int idx = hlut3_tile_flat(r, c);
            const int g = hlut3_group(idx);
            const float sigma = fp16_to_f32(tile.scales_f16[g]);
            const double z = (sigma > 0.0f) ? double(std::fabs(src[c])) / double(sigma) : 0.0;
            uint32_t mag = 0u;
            if (z >= sol.thresholds[2]) mag = 3u;
            else if (z >= sol.thresholds[1]) mag = 2u;
            else if (z >= sol.thresholds[0]) mag = 1u;
            else mag = 0u;
            const uint32_t sign = std::signbit(src[c]) ? 1u : 0u;
            tile.q_lo2[hlut3_mag_word(idx)] |= (mag << hlut3_mag_shift(idx));
            tile.q_hi1[hlut3_sign_word(idx)] |= (sign << hlut3_sign_shift(idx));
        }
    }
}

inline void hlut3_unpack_tile(
    const Hlut3Tile32x32& tile,
    const Hlut3LloydMaxSolution& sol,
    float* out_rowmajor,
    int K,
    int n0,
    int k0)
{
    for (int r = 0; r < kHlut3TileRows; ++r) {
        float* dst = out_rowmajor + size_t(n0 + r) * size_t(K) + size_t(k0);
        for (int c = 0; c < kHlut3TileCols; ++c) {
            const int idx = hlut3_tile_flat(r, c);
            const int g = hlut3_group(idx);
            const uint32_t mag = (tile.q_lo2[hlut3_mag_word(idx)] >> hlut3_mag_shift(idx)) & 3u;
            const uint32_t sign = (tile.q_hi1[hlut3_sign_word(idx)] >> hlut3_sign_shift(idx)) & 1u;
            const float scale = fp16_to_f32(tile.scales_f16[g]);
            const float magnitude = float(sol.levels[mag]) * scale;
            dst[c] = sign ? -magnitude : magnitude;
        }
    }
}

struct Hlut3QuantMatrix {
    int N = 0;
    int K = 0;
    int nt = 0;  // N / 32 tile rows
    int kt = 0;  // K / 32 tile cols
    Hlut3LloydMaxSolution lloyd{};
    std::vector<Hlut3Tile32x32> tiles; // row-major tile grid: tile_n * kt + tile_k

    size_t raw_bytes() const { return tiles.size() * sizeof(Hlut3Tile32x32); }
    double bits_per_weight() const {
        return 8.0 * double(raw_bytes()) / double(size_t(N) * size_t(K));
    }
    const Hlut3Tile32x32& tile_at(int n_tile, int k_tile) const {
        return tiles[size_t(n_tile) * size_t(kt) + size_t(k_tile)];
    }
};

// H=I data-free WHT128-HLUT3 quantizer. Exact Gaussian fixed-point levels; every tile has
// deterministic FP16 RMS scales over eight 128-weight groups (4 rows x 32 cols).
inline Hlut3QuantMatrix quantize_hlut3(
    const float* W_rm,
    int N,
    int K,
    const std::vector<uint64_t>& signs)
{
    if (W_rm == nullptr || N <= 0 || K <= 0 || N % kHlut3TileRows != 0 || K % kHlut3TileCols != 0) {
        throw std::invalid_argument("HLUT3 quantize requires N%32==0 and K%32==0");
    }
    Hlut3QuantMatrix q;
    q.N = N;
    q.K = K;
    q.nt = N / kHlut3TileRows;
    q.kt = K / kHlut3TileCols;
    q.lloyd = solve_gaussian_lloyd_max_hlut3();
    q.tiles.assign(size_t(q.nt) * size_t(q.kt), Hlut3Tile32x32{});

    std::vector<float> w_rot(size_t(N) * size_t(K));
    wht128_rows(W_rm, w_rot.data(), N, K, signs);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int nt_i = 0; nt_i < q.nt; ++nt_i) {
        for (int kt_i = 0; kt_i < q.kt; ++kt_i) {
            Hlut3Tile32x32& t = q.tiles[size_t(nt_i) * size_t(q.kt) + size_t(kt_i)];
            hlut3_pack_tile(w_rot.data(), K, nt_i * kHlut3TileRows, kt_i * kHlut3TileCols, q.lloyd, t);
        }
    }
    return q;
}

inline void unpack_hlut3_to_rot_f32(const Hlut3QuantMatrix& q, std::vector<float>& out_rot) {
    if (q.tiles.size() != size_t(q.nt) * size_t(q.kt)) {
        throw std::runtime_error("HLUT3 unpack tile grid size mismatch");
    }
    out_rot.assign(size_t(q.N) * size_t(q.K), 0.0f);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nt_i = 0; nt_i < q.nt; ++nt_i) {
        for (int kt_i = 0; kt_i < q.kt; ++kt_i) {
            const Hlut3Tile32x32& t = q.tiles[size_t(nt_i) * size_t(q.kt) + size_t(kt_i)];
            hlut3_unpack_tile(t, q.lloyd, out_rot.data(), q.K,
                              nt_i * kHlut3TileRows, kt_i * kHlut3TileCols);
        }
    }
}

inline void unpack_hlut3_tiles_to_rowmajor(const Hlut3QuantMatrix& q, float* out_rot) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nt_i = 0; nt_i < q.nt; ++nt_i) {
        for (int kt_i = 0; kt_i < q.kt; ++kt_i) {
            hlut3_unpack_tile(q.tiles[size_t(nt_i) * size_t(q.kt) + size_t(kt_i)],
                              q.lloyd, out_rot, q.K,
                              nt_i * kHlut3TileRows, kt_i * kHlut3TileCols);
        }
    }
}

// ---- P150A TP2 6,400 B superblock packer / unpacker ----
inline std::vector<Hlut3Superblock128x128> pack_hlut3_for_p150_shard(
    const Hlut3QuantMatrix& q,
    int n_col_offset,
    int N_shard)
{
    if (n_col_offset < 0 || N_shard <= 0 || N_shard % kHlut3SuperRows != 0 ||
        q.K % kHlut3SuperCols != 0 || n_col_offset + N_shard > q.N) {
        throw std::invalid_argument("P150 HLUT3 shard requires N_shard%128==0 and K%128==0");
    }
    const int nt4 = N_shard / kHlut3SuperRows;
    const int kt4 = q.K / kHlut3SuperCols;
    std::vector<Hlut3Superblock128x128> out(size_t(nt4) * size_t(kt4));
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nb4 = 0; nb4 < nt4; ++nb4) {
        for (int kb4 = 0; kb4 < kt4; ++kb4) {
            Hlut3Superblock128x128& sb = out[size_t(nb4) * size_t(kt4) + size_t(kb4)];
            for (int nt_sub = 0; nt_sub < 4; ++nt_sub) {
                for (int kt_sub = 0; kt_sub < 4; ++kt_sub) {
                    const int tile_n = nb4 * 4 + nt_sub;
                    const int tile_k = kb4 * 4 + kt_sub;
                    sb.subtiles[nt_sub * 4 + kt_sub] = q.tile_at(tile_n, tile_k);
                }
            }
        }
    }
    return out;
}

inline void unpack_hlut3_p150_superblocks(
    const std::vector<Hlut3Superblock128x128>& sbs,
    const Hlut3LloydMaxSolution& sol,
    int K,
    int N_shard,
    int n_col_offset,
    float* out_rot)
{
    if (K % 128 != 0 || N_shard % 128 != 0) {
        throw std::invalid_argument("HLUT3 superblock unpack requires K%128==0 and N_shard%128==0");
    }
    const int nt4 = N_shard / 128;
    const int kt4 = K / 128;
    if (sbs.size() != size_t(nt4) * size_t(kt4)) throw std::runtime_error("HLUT3 superblock count mismatch");
    for (int nb4 = 0; nb4 < nt4; ++nb4) {
        for (int kb4 = 0; kb4 < kt4; ++kb4) {
            const Hlut3Superblock128x128& sb = sbs[size_t(nb4) * size_t(kt4) + size_t(kb4)];
            for (int nt_sub = 0; nt_sub < 4; ++nt_sub) {
                for (int kt_sub = 0; kt_sub < 4; ++kt_sub) {
                    hlut3_unpack_tile(sb.subtiles[nt_sub * 4 + kt_sub], sol, out_rot, K,
                                      n_col_offset + (nb4 * 4 + nt_sub) * 32,
                                      (kb4 * 4 + kt_sub) * 32);
                }
            }
        }
    }
}

// Exact CPU reference for one full expert matrix (gate/up/down) packed as planar MI50 tiles.
// GPU kernels consume the same 400 B tile structs, so parity is bit-layout identical.
struct Hlut3Mi50Packed {
    int N = 0;
    int K = 0;
    Hlut3LloydMaxSolution lloyd{};
    std::vector<Hlut3Tile32x32> tiles; // N/32 * K/32 tiles, 400 B each
};

inline Hlut3Mi50Packed pack_hlut3_for_mi50(const Hlut3QuantMatrix& q) {
    Hlut3Mi50Packed p;
    p.N = q.N;
    p.K = q.K;
    p.lloyd = q.lloyd;
    p.tiles = q.tiles;
    return p;
}

// ---- Reference forward/inverse WHT128 (production signs are Rademacher from make_wht_signs) ----
inline void hlut3_wht128_forward_block(float* v, const uint64_t* sign_bits_block2) {
    wht128_block(v, sign_bits_block2);
}

inline void hlut3_wht128_inverse_block(float* v, const uint64_t* sign_bits_block2) {
    // Q = H_128 * diag(s) / sqrt(128) is symmetric orthogonal: Q^{-1} = Q^T = Q exactly.
    wht128_block(v, sign_bits_block2);
}

inline void hlut3_wht128_rows(const float* in, float* out, int rows, int K, const std::vector<uint64_t>& signs) {
    wht128_rows(in, out, rows, K, signs);
}

inline void hlut3_wht128_rows_inverse(const float* in_rot, float* out, int rows, int K, const std::vector<uint64_t>& signs) {
    wht128_rows(in_rot, out, rows, K, signs);
}

// Production layer seeds are matched bit-for-bit. These are the exact seed constants from
// src/dsv41_pack_format.hpp::layer_wht_seed_h/i, reproduced under HLUT3 names to avoid a
// duplicate inline definition when both headers are included in one production TU.
inline uint64_t hlut3_layer_wht_seed_h(int global_layer_id) noexcept {
    return 0x44535634315F4800ULL ^ (uint64_t(global_layer_id + 1) * 0x9E3779B97F4A7C15ULL);
}
inline uint64_t hlut3_layer_wht_seed_i(int global_layer_id) noexcept {
    return 0x44535634315F4900ULL ^ (uint64_t(global_layer_id + 1) * 0xBF58476D1CE4E5B9ULL);
}

inline std::vector<uint64_t> make_hlut3_signs_h(int global_layer_id, int K) {
    return make_wht_signs(K, hlut3_layer_wht_seed_h(global_layer_id));
}

inline std::vector<uint64_t> make_hlut3_signs_i(int global_layer_id, int K) {
    return make_wht_signs(K, hlut3_layer_wht_seed_i(global_layer_id));
}

}  // namespace dsv41

// Compatibility polish: standalone C++20 reference TUs that include "hlut3_pack.hpp"
// without `using namespace dsv41;` can still reference the tile/superblock structs unqualified.
using dsv41::Hlut3Tile32x32;
using dsv41::Hlut3Superblock128x128;
