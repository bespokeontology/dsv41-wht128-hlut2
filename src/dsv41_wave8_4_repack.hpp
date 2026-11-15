#pragma once
// Wave 8.4: Lossless Bit-Exact Cross-Tier (3x MI50 TP3 <-> 2x Blackhole P150A TP2) HLUT2 Repack
// & Minimax Frequency-Balanced 384-Expert Placement Optimizer.
// Pure C++17, zero Python anywhere.

#include "dsv41_pack_format.hpp"
#include "safetensors_mmap.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dsv41 {

struct ExpertHlut2Triplet {
    Hlut2QuantMatrix qg; // [2304, 5120]
    Hlut2QuantMatrix qu; // [2304, 5120]
    Hlut2QuantMatrix qd; // [5120, 2304]
};

inline Hlut2QuantMatrix unpack_hlut2_mi50_tp3_row_sliced(
    const std::array<const uint32_t*, 3>& src_codes,
    const std::array<const uint16_t*, 3>& src_scales,
    const std::array<const float*, 3>& src_cent)
{
    Hlut2QuantMatrix q;
    q.N = 2304;
    q.K = 5120;
    q.c0 = src_cent[0][0];
    q.c1 = src_cent[0][1];
    constexpr int kw = 320;
    constexpr int G  = 40;
    q.codes_rm.assign(size_t(q.N) * kw, 0u);
    q.scales_fp16.assign(size_t(q.N) * G, 0u);

    for (int tp = 0; tp < 3; ++tp) {
        for (int rb = 0; rb < 24; ++rb) {
            for (int r = 0; r < 32; ++r) {
                const int row_global = tp * 768 + rb * 32 + r;
                for (int w = 0; w < kw; ++w) {
                    q.codes_rm[size_t(row_global) * kw + w] = src_codes[tp][(size_t(rb) * kw + w) * 32 + r];
                }
                for (int g = 0; g < G; ++g) {
                    q.scales_fp16[size_t(row_global) * G + g] = src_scales[tp][(size_t(rb) * G + g) * 32 + r];
                }
            }
        }
    }
    return q;
}

inline Hlut2QuantMatrix unpack_hlut2_mi50_tp3_col_sliced(
    const std::array<const uint32_t*, 3>& src_codes,
    const std::array<const uint16_t*, 3>& src_scales,
    const std::array<const float*, 3>& src_cent)
{
    Hlut2QuantMatrix q;
    q.N = 5120;
    q.K = 2304;
    q.c0 = src_cent[0][0];
    q.c1 = src_cent[0][1];
    constexpr int kw_full = 144;
    constexpr int G_full  = 18;
    constexpr int kw_tp   = 48;
    constexpr int G_tp    = 6;
    q.codes_rm.assign(size_t(q.N) * kw_full, 0u);
    q.scales_fp16.assign(size_t(q.N) * G_full, 0u);

    for (int tp = 0; tp < 3; ++tp) {
        for (int rb = 0; rb < 160; ++rb) {
            for (int r = 0; r < 32; ++r) {
                const int row = rb * 32 + r;
                for (int w = 0; w < kw_tp; ++w) {
                    q.codes_rm[size_t(row) * kw_full + tp * kw_tp + w] =
                        src_codes[tp][(size_t(rb) * kw_tp + w) * 32 + r];
                }
                for (int g = 0; g < G_tp; ++g) {
                    q.scales_fp16[size_t(row) * G_full + tp * G_tp + g] =
                        src_scales[tp][(size_t(rb) * G_tp + g) * 32 + r];
                }
            }
        }
    }
    return q;
}

inline Hlut2QuantMatrix unpack_hlut2_p150_tp2_row_sliced(
    const std::array<const uint8_t*, 2>& src_pages,
    const std::array<const float*, 2>& src_cent)
{
    Hlut2QuantMatrix q;
    q.N = 2304;
    q.K = 5120;
    q.c0 = src_cent[0][0];
    q.c1 = src_cent[0][1];
    constexpr int kw  = 320;
    constexpr int G   = 40;
    constexpr int nt4 = 9;
    constexpr int kt4 = 40;
    q.codes_rm.assign(size_t(q.N) * kw, 0u);
    q.scales_fp16.assign(size_t(q.N) * G, 0u);

    for (int tp = 0; tp < 2; ++tp) {
        for (int nb4 = 0; nb4 < nt4; ++nb4) {
            for (int kb4 = 0; kb4 < kt4; ++kb4) {
                const int sb = nb4 * kt4 + kb4;
                const uint8_t* page_ptr = src_pages[tp] + size_t(sb) * 4352ULL;
                const auto* scodes  = reinterpret_cast<const uint32_t*>(page_ptr);
                const auto* sscales = reinterpret_cast<const uint16_t*>(page_ptr + 4096);

                for (int nt_sub = 0; nt_sub < 4; ++nt_sub) {
                    for (int j = 0; j < 32; ++j) {
                        const int n_global = tp * 1152 + (nb4 * 4 + nt_sub) * 32 + j;
                        q.scales_fp16[size_t(n_global) * G + kb4] = sscales[nt_sub * 32 + j];
                    }
                }
                for (int i = 0; i < 32; ++i) {
                    for (int j = 0; j < 32; ++j) {
                        const uint32_t word = scodes[tensix_tile_offset(i, j)];
                        for (int kt_sub = 0; kt_sub < 4; ++kt_sub) {
                            const int k_global  = (kb4 * 4 + kt_sub) * 32 + i;
                            const int w_idx     = k_global >> 4;
                            const int bit_shift = 2 * (k_global & 15);
                            for (int nt_sub = 0; nt_sub < 4; ++nt_sub) {
                                const int n_global = tp * 1152 + (nb4 * 4 + nt_sub) * 32 + j;
                                const int t = kt_sub * 4 + nt_sub;
                                const uint32_t code = (word >> (2 * t)) & 3u;
                                q.codes_rm[size_t(n_global) * kw + w_idx] |= (code << bit_shift);
                            }
                        }
                    }
                }
            }
        }
    }
    return q;
}

inline Hlut2QuantMatrix unpack_hlut2_p150_tp2_col_sliced(
    const std::array<const uint8_t*, 2>& src_pages,
    const std::array<const float*, 2>& src_cent)
{
    Hlut2QuantMatrix q;
    q.N = 5120;
    q.K = 2304;
    q.c0 = src_cent[0][0];
    q.c1 = src_cent[0][1];
    constexpr int kw_full = 144;
    constexpr int G_full  = 18;
    constexpr int nt4     = 40;
    constexpr int kt4     = 9;
    q.codes_rm.assign(size_t(q.N) * kw_full, 0u);
    q.scales_fp16.assign(size_t(q.N) * G_full, 0u);

    for (int tp = 0; tp < 2; ++tp) {
        for (int nb4 = 0; nb4 < nt4; ++nb4) {
            for (int kb4 = 0; kb4 < kt4; ++kb4) {
                const int sb = nb4 * kt4 + kb4;
                const uint8_t* page_ptr = src_pages[tp] + size_t(sb) * 4352ULL;
                const auto* scodes  = reinterpret_cast<const uint32_t*>(page_ptr);
                const auto* sscales = reinterpret_cast<const uint16_t*>(page_ptr + 4096);

                for (int nt_sub = 0; nt_sub < 4; ++nt_sub) {
                    for (int j = 0; j < 32; ++j) {
                        const int n_global = (nb4 * 4 + nt_sub) * 32 + j;
                        q.scales_fp16[size_t(n_global) * G_full + tp * 9 + kb4] = sscales[nt_sub * 32 + j];
                    }
                }
                for (int i = 0; i < 32; ++i) {
                    for (int j = 0; j < 32; ++j) {
                        const uint32_t word = scodes[tensix_tile_offset(i, j)];
                        for (int kt_sub = 0; kt_sub < 4; ++kt_sub) {
                            const int k_global  = tp * 1152 + (kb4 * 4 + kt_sub) * 32 + i;
                            const int w_idx     = k_global >> 4;
                            const int bit_shift = 2 * (k_global & 15);
                            for (int nt_sub = 0; nt_sub < 4; ++nt_sub) {
                                const int n_global = (nb4 * 4 + nt_sub) * 32 + j;
                                const int t = kt_sub * 4 + nt_sub;
                                const uint32_t code = (word >> (2 * t)) & 3u;
                                q.codes_rm[size_t(n_global) * kw_full + w_idx] |= (code << bit_shift);
                            }
                        }
                    }
                }
            }
        }
    }
    return q;
}

inline ExpertHlut2Triplet unpack_expert_from_overlay_st(const SafetensorsMmap& ov, int bid, int logical_e) {
    ExpertHlut2Triplet tr;
    if (bid >= 40 || logical_e < 216) {
        std::array<const uint32_t*, 3> gc{}, uc{}, dc{};
        std::array<const uint16_t*, 3> gs{}, us{}, ds{};
        std::array<const float*, 3> gcent{}, ucent{}, dcent{};
        for (int tp = 0; tp < 3; ++tp) {
            const std::string p = "mi50.b" + std::to_string(bid) + ".e" + std::to_string(logical_e) + ".tp" + std::to_string(tp);
            gc[tp]    = reinterpret_cast<const uint32_t*>(ov.tensor_bytes(p + ".gate_codes"));
            gs[tp]    = reinterpret_cast<const uint16_t*>(ov.tensor_bytes(p + ".gate_scales"));
            gcent[tp] = reinterpret_cast<const float*>(ov.tensor_bytes(p + ".gate_centroids"));
            uc[tp]    = reinterpret_cast<const uint32_t*>(ov.tensor_bytes(p + ".up_codes"));
            us[tp]    = reinterpret_cast<const uint16_t*>(ov.tensor_bytes(p + ".up_scales"));
            ucent[tp] = reinterpret_cast<const float*>(ov.tensor_bytes(p + ".up_centroids"));
            dc[tp]    = reinterpret_cast<const uint32_t*>(ov.tensor_bytes(p + ".down_codes"));
            ds[tp]    = reinterpret_cast<const uint16_t*>(ov.tensor_bytes(p + ".down_scales"));
            dcent[tp] = reinterpret_cast<const float*>(ov.tensor_bytes(p + ".down_centroids"));
        }
        tr.qg = unpack_hlut2_mi50_tp3_row_sliced(gc, gs, gcent);
        tr.qu = unpack_hlut2_mi50_tp3_row_sliced(uc, us, ucent);
        tr.qd = unpack_hlut2_mi50_tp3_col_sliced(dc, ds, dcent);
    } else {
        const int le = logical_e - 216;
        std::array<const uint8_t*, 2> gp{}, up{}, dp{};
        std::array<const float*, 2> gc{}, uc{}, dc{};
        for (int tp = 0; tp < 2; ++tp) {
            const std::string p = "p150.l" + std::to_string(bid) + ".e" + std::to_string(le) + ".tp" + std::to_string(tp);
            gp[tp] = ov.tensor_bytes(p + ".gate_pages");
            gc[tp] = reinterpret_cast<const float*>(ov.tensor_bytes(p + ".gate_centroids"));
            up[tp] = ov.tensor_bytes(p + ".up_pages");
            uc[tp] = reinterpret_cast<const float*>(ov.tensor_bytes(p + ".up_centroids"));
            dp[tp] = ov.tensor_bytes(p + ".down_pages");
            dc[tp] = reinterpret_cast<const float*>(ov.tensor_bytes(p + ".down_centroids"));
        }
        tr.qg = unpack_hlut2_p150_tp2_row_sliced(gp, gc);
        tr.qu = unpack_hlut2_p150_tp2_row_sliced(up, uc);
        tr.qd = unpack_hlut2_p150_tp2_col_sliced(dp, dc);
    }
    return tr;
}

inline ExpertHlut2Triplet unpack_expert_from_base_mi50(
    const std::array<const SafetensorsMmap*, 3>& st_tp3, int slot_e)
{
    constexpr size_t kCodesWords  = 245760ULL;
    constexpr size_t kScalesHalfs = 30720ULL;
    std::array<const uint32_t*, 3> gc{}, uc{}, dc{};
    std::array<const uint16_t*, 3> gs{}, us{}, ds{};
    std::array<const float*, 3> gcent{}, ucent{}, dcent{};
    for (int tp = 0; tp < 3; ++tp) {
        gc[tp]    = reinterpret_cast<const uint32_t*>(st_tp3[tp]->tensor_bytes("hlut2.gate_codes"))     + size_t(slot_e) * kCodesWords;
        gs[tp]    = reinterpret_cast<const uint16_t*>(st_tp3[tp]->tensor_bytes("hlut2.gate_scales"))    + size_t(slot_e) * kScalesHalfs;
        gcent[tp] = reinterpret_cast<const float*>(st_tp3[tp]->tensor_bytes("hlut2.gate_centroids"))    + size_t(slot_e) * 2ULL;
        uc[tp]    = reinterpret_cast<const uint32_t*>(st_tp3[tp]->tensor_bytes("hlut2.up_codes"))       + size_t(slot_e) * kCodesWords;
        us[tp]    = reinterpret_cast<const uint16_t*>(st_tp3[tp]->tensor_bytes("hlut2.up_scales"))      + size_t(slot_e) * kScalesHalfs;
        ucent[tp] = reinterpret_cast<const float*>(st_tp3[tp]->tensor_bytes("hlut2.up_centroids"))      + size_t(slot_e) * 2ULL;
        dc[tp]    = reinterpret_cast<const uint32_t*>(st_tp3[tp]->tensor_bytes("hlut2.down_codes"))     + size_t(slot_e) * kCodesWords;
        ds[tp]    = reinterpret_cast<const uint16_t*>(st_tp3[tp]->tensor_bytes("hlut2.down_scales"))    + size_t(slot_e) * kScalesHalfs;
        dcent[tp] = reinterpret_cast<const float*>(st_tp3[tp]->tensor_bytes("hlut2.down_centroids"))    + size_t(slot_e) * 2ULL;
    }
    ExpertHlut2Triplet tr;
    tr.qg = unpack_hlut2_mi50_tp3_row_sliced(gc, gs, gcent);
    tr.qu = unpack_hlut2_mi50_tp3_row_sliced(uc, us, ucent);
    tr.qd = unpack_hlut2_mi50_tp3_col_sliced(dc, ds, dcent);
    return tr;
}

inline ExpertHlut2Triplet unpack_expert_from_base_p150(
    const std::array<const SafetensorsMmap*, 2>& st_tp2, int local_e)
{
    constexpr size_t kProjBytes = 360ULL * 4352ULL;
    std::array<const uint8_t*, 2> gp{}, up{}, dp{};
    std::array<const float*, 2> gc{}, uc{}, dc{};
    for (int tp = 0; tp < 2; ++tp) {
        gp[tp] = st_tp2[tp]->tensor_bytes("hlut2.gate_pages") + size_t(local_e) * kProjBytes;
        gc[tp] = reinterpret_cast<const float*>(st_tp2[tp]->tensor_bytes("hlut2.gate_centroids")) + size_t(local_e) * 2ULL;
        up[tp] = st_tp2[tp]->tensor_bytes("hlut2.up_pages")   + size_t(local_e) * kProjBytes;
        uc[tp] = reinterpret_cast<const float*>(st_tp2[tp]->tensor_bytes("hlut2.up_centroids"))   + size_t(local_e) * 2ULL;
        dp[tp] = st_tp2[tp]->tensor_bytes("hlut2.down_pages") + size_t(local_e) * kProjBytes;
        dc[tp] = reinterpret_cast<const float*>(st_tp2[tp]->tensor_bytes("hlut2.down_centroids")) + size_t(local_e) * 2ULL;
    }
    ExpertHlut2Triplet tr;
    tr.qg = unpack_hlut2_p150_tp2_row_sliced(gp, gc);
    tr.qu = unpack_hlut2_p150_tp2_row_sliced(up, uc);
    tr.qd = unpack_hlut2_p150_tp2_col_sliced(dp, dc);
    return tr;
}

struct PackedOverlayTensor {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> buf;
};

inline void pack_expert_into_overlay_tensors(
    const ExpertHlut2Triplet& tr,
    int bid,
    int phys_e,
    std::vector<PackedOverlayTensor>& out)
{
    if (bid >= 40 || phys_e < 216) {
        out.reserve(out.size() + 27);
        for (int tp = 0; tp < 3; ++tp) {
            std::vector<uint8_t> gc(983040), gs(61440), gcent(8);
            std::vector<uint8_t> uc(983040), us(61440), ucent(8);
            std::vector<uint8_t> dc(983040), ds(61440), dcent(8);
            pack_hlut2_mi50_slot_st(slice_hlut2_rows(tr.qg, tp * 768, 768),
                                    reinterpret_cast<uint32_t*>(gc.data()),
                                    reinterpret_cast<uint16_t*>(gs.data()),
                                    reinterpret_cast<float*>(gcent.data()));
            pack_hlut2_mi50_slot_st(slice_hlut2_rows(tr.qu, tp * 768, 768),
                                    reinterpret_cast<uint32_t*>(uc.data()),
                                    reinterpret_cast<uint16_t*>(us.data()),
                                    reinterpret_cast<float*>(ucent.data()));
            pack_hlut2_mi50_slot_st(slice_hlut2_cols(tr.qd, tp * 768, 768),
                                    reinterpret_cast<uint32_t*>(dc.data()),
                                    reinterpret_cast<uint16_t*>(ds.data()),
                                    reinterpret_cast<float*>(dcent.data()));
            const std::string p = "mi50.b" + std::to_string(bid) + ".e" + std::to_string(phys_e) + ".tp" + std::to_string(tp);
            out.push_back({p + ".gate_codes",     "U32", {24, 320, 32}, std::move(gc)});
            out.push_back({p + ".gate_scales",    "F16", {24, 40, 32},  std::move(gs)});
            out.push_back({p + ".gate_centroids", "F32", {2},           std::move(gcent)});
            out.push_back({p + ".up_codes",       "U32", {24, 320, 32}, std::move(uc)});
            out.push_back({p + ".up_scales",      "F16", {24, 40, 32},  std::move(us)});
            out.push_back({p + ".up_centroids",   "F32", {2},           std::move(ucent)});
            out.push_back({p + ".down_codes",     "U32", {160, 48, 32}, std::move(dc)});
            out.push_back({p + ".down_scales",    "F16", {160, 6, 32},   std::move(ds)});
            out.push_back({p + ".down_centroids", "F32", {2},           std::move(dcent)});
        }
    } else {
        out.reserve(out.size() + 12);
        const int le = phys_e - 216;
        for (int tp = 0; tp < 2; ++tp) {
            std::vector<uint8_t> gp(360ULL * 4352ULL), gc(8);
            std::vector<uint8_t> up(360ULL * 4352ULL), uc(8);
            std::vector<uint8_t> dp(360ULL * 4352ULL), dc(8);
            pack_hlut2_p150_slot_st(tr.qg, tp * 1152, 1152, gp.data(), reinterpret_cast<float*>(gc.data()));
            pack_hlut2_p150_slot_st(tr.qu, tp * 1152, 1152, up.data(), reinterpret_cast<float*>(uc.data()));
            pack_hlut2_p150_slot_st(slice_hlut2_cols(tr.qd, tp * 1152, 1152), 0, 5120, dp.data(), reinterpret_cast<float*>(dc.data()));
            const std::string p = "p150.l" + std::to_string(bid) + ".e" + std::to_string(le) + ".tp" + std::to_string(tp);
            out.push_back({p + ".gate_pages",     "U8",  {360, 4352}, std::move(gp)});
            out.push_back({p + ".gate_centroids", "F32", {2},         std::move(gc)});
            out.push_back({p + ".up_pages",       "U8",  {360, 4352}, std::move(up)});
            out.push_back({p + ".up_centroids",   "F32", {2},         std::move(uc)});
            out.push_back({p + ".down_pages",     "U8",  {360, 4352}, std::move(dp)});
            out.push_back({p + ".down_centroids", "F32", {2},         std::move(dc)});
        }
    }
}

// Silicon-calibrated M=8 (48 routed pairs/layer) batch-aware latency model (milliseconds per layer):
struct Wave84LayerCost {
    double mu_mi50 = 0.0;
    double mu_l1   = 0.0;
    double mu_gddr = 0.0;
    double t_mi50_ms = 0.0;
    double t_p150_ms = 0.0;
    double max_ms    = 0.0;
};

inline double poisson_tail_gt(double lambda, int k) noexcept {
    if (lambda <= 1e-9) return 0.0;
    double term = std::exp(-lambda);
    double cdf  = term;
    for (int i = 1; i <= k; ++i) {
        term *= (lambda / double(i));
        cdf  += term;
    }
    return std::clamp(1.0 - cdf, 0.0, 1.0);
}

inline Wave84LayerCost eval_wave84_layer_cost(double mu_mi50, double mu_l1, double mu_gddr) noexcept {
    Wave84LayerCost c;
    c.mu_mi50 = mu_mi50;
    c.mu_l1   = mu_l1;
    c.mu_gddr = mu_gddr;
    // Exact silicon-calibrated slopes from Wave 8.3 dsv41_fe.log (6,512 M=8 steps, 24,993 tokens):
    //   3x MI50 TP3 HBM2 (3.07 TB/s): 0.285 ms fixed (Shared SwiGLU + router + CSR) + 0.0161 ms/pair (= 29.20 ms / 40L at baseline 27.64 pairs/L)
    c.t_mi50_ms = 0.285 + 0.0161 * mu_mi50;

    //   2x Blackhole P150A TP2 (with x16->FABRIC_1D relay):
    //   0.155 ms relay/all_reduce + 0.145 ms per 16-slot MeshTrace batch + 0.0435 ms per GDDR6 DRAM expert miss (= 49.7 ms / 40L post-relay at baseline)
    const double mu_p150 = mu_l1 + mu_gddr;
    const double p_b1 = poisson_tail_gt(mu_p150, 0);
    const double p_b2 = poisson_tail_gt(mu_p150, 16);
    const double p_b3 = poisson_tail_gt(mu_p150, 32);
    const double e_batches = p_b1 + p_b2 + p_b3;
    c.t_p150_ms = p_b1 * 0.155 + e_batches * 0.145 + 0.0435 * mu_gddr;
    c.max_ms = std::max(c.t_mi50_ms, c.t_p150_ms);
    return c;
}

struct Wave84PlacementPlan {
    std::array<std::array<uint16_t, 384>, 40> perm{};
    // Global top K_L1 experts: (layer, logical_e) -> physical local_e = perm[layer][logical_e] - 216
    std::vector<std::pair<int, int>> l1_logical_pairs;
    std::vector<std::pair<uint16_t, uint16_t>> l1_phys_assignments; // (layer, local_eid in 0..167)
    std::vector<std::pair<int, int>> swapped_logical_pairs;         // all (layer, logical_e) where perm[l][e] != e
    double base_mi50_sum_ms = 0.0;
    double base_p150_sum_ms = 0.0;
    double base_max_sum_ms  = 0.0;
    double opt_mi50_sum_ms  = 0.0;
    double opt_p150_sum_ms  = 0.0;
    double opt_max_sum_ms   = 0.0;
    int total_swapped_pairs = 0;
};

inline Wave84PlacementPlan optimize_wave84_placement(
    const std::array<std::array<double, 384>, 40>& pair_expectation_m8, // expected hits per M=8 step (sums to 48 per layer)
    int k_l1_shards = 18,
    int max_swaps_per_layer = 32)
{
    Wave84PlacementPlan plan;

    // 1. Select global top-K_L1 experts across all 40 x 384 backbone experts for P150A L1 SRAM
    struct GlobCand { double mu; int l; int e; };
    std::vector<GlobCand> glob;
    glob.reserve(40 * 384);
    for (int l = 0; l < 40; ++l) {
        for (int e = 0; e < 384; ++e) {
            glob.push_back({pair_expectation_m8[size_t(l)][size_t(e)], l, e});
        }
    }
    std::sort(glob.begin(), glob.end(), [](const GlobCand& a, const GlobCand& b) {
        if (a.mu != b.mu) return a.mu > b.mu;
        return (a.l != b.l) ? (a.l < b.l) : (a.e < b.e);
    });

    std::array<std::array<bool, 384>, 40> is_l1{};
    for (int i = 0; i < std::min<int>(k_l1_shards, int(glob.size())); ++i) {
        is_l1[size_t(glob[size_t(i)].l)][size_t(glob[size_t(i)].e)] = true;
        plan.l1_logical_pairs.push_back({glob[size_t(i)].l, glob[size_t(i)].e});
    }

    // 2. Per-layer minimax tier assignment + involutive zero-churn permutation
    for (int l = 0; l < 40; ++l) {
        const auto& mu = pair_expectation_m8[size_t(l)];

        double base_mi = 0.0, base_l1 = 0.0, base_gd = 0.0;
        for (int e = 0; e < 216; ++e) base_mi += mu[size_t(e)];
        for (int e = 216; e < 384; ++e) {
            if (is_l1[size_t(l)][size_t(e)]) base_l1 += mu[size_t(e)];
            else                             base_gd += mu[size_t(e)];
        }
        const auto bc = eval_wave84_layer_cost(base_mi, base_l1, base_gd);
        plan.base_mi50_sum_ms += bc.t_mi50_ms;
        plan.base_p150_sum_ms += bc.t_p150_ms;
        plan.base_max_sum_ms  += bc.max_ms;

        // Initial tier assignment: tier[e] == 0 -> MI50 [0..215], tier[e] == 1 -> P150 [216..383]
        std::array<int, 384> tier{};
        for (int e = 0; e < 216; ++e) tier[size_t(e)] = 0;
        for (int e = 216; e < 384; ++e) tier[size_t(e)] = 1;

        // Force any global L1 expert originally in [0..215] onto P150, swapping with the hottest non-L1 [216..383] expert
        int forced_swaps = 0;
        for (int e = 0; e < 216; ++e) {
            if (is_l1[size_t(l)][size_t(e)]) {
                int best_p1 = -1;
                double best_mu = -1.0;
                for (int p = 216; p < 384; ++p) {
                    if (tier[size_t(p)] == 1 && !is_l1[size_t(l)][size_t(p)] && mu[size_t(p)] > best_mu) {
                        best_mu = mu[size_t(p)];
                        best_p1 = p;
                    }
                }
                if (best_p1 >= 0) {
                    tier[size_t(e)]       = 1;
                    tier[size_t(best_p1)] = 0;
                    ++forced_swaps;
                }
            }
        }

        auto calc_cur = [&]() {
            double m_mi = 0.0, m_l1 = 0.0, m_gd = 0.0;
            for (int e = 0; e < 384; ++e) {
                if (tier[size_t(e)] == 0) m_mi += mu[size_t(e)];
                else if (is_l1[size_t(l)][size_t(e)]) m_l1 += mu[size_t(e)];
                else m_gd += mu[size_t(e)];
            }
            return eval_wave84_layer_cost(m_mi, m_l1, m_gd);
        };

        for (int step = forced_swaps; step < max_swaps_per_layer; ++step) {
            const auto cur = calc_cur();
            if (cur.t_p150_ms > cur.t_mi50_ms + 0.002) {
                // Move hottest non-L1 P150 expert -> MI50, coldest MI50 expert -> P150 GDDR6
                int hot_p1 = -1, cold_mi = -1;
                double hot_v = -1.0, cold_v = 1e30;
                for (int e = 0; e < 384; ++e) {
                    if (tier[size_t(e)] == 1 && !is_l1[size_t(l)][size_t(e)] && mu[size_t(e)] > hot_v) {
                        hot_v = mu[size_t(e)]; hot_p1 = e;
                    }
                    if (tier[size_t(e)] == 0 && !is_l1[size_t(l)][size_t(e)] && mu[size_t(e)] < cold_v) {
                        cold_v = mu[size_t(e)]; cold_mi = e;
                    }
                }
                if (hot_p1 < 0 || cold_mi < 0 || hot_v <= cold_v) break;
                tier[size_t(hot_p1)]  = 0;
                tier[size_t(cold_mi)] = 1;
                const auto nxt = calc_cur();
                if (nxt.max_ms < cur.max_ms - 0.0015) {
                    continue;
                }
                tier[size_t(hot_p1)]  = 1;
                tier[size_t(cold_mi)] = 0;
                break;
            } else if (cur.t_mi50_ms > cur.t_p150_ms + 0.015) {
                // Move hottest MI50 expert -> P150 GDDR6, coldest non-L1 P150 expert -> MI50
                int hot_mi = -1, cold_p1 = -1;
                double hot_v = -1.0, cold_v = 1e30;
                for (int e = 0; e < 384; ++e) {
                    if (tier[size_t(e)] == 0 && !is_l1[size_t(l)][size_t(e)] && mu[size_t(e)] > hot_v) {
                        hot_v = mu[size_t(e)]; hot_mi = e;
                    }
                    if (tier[size_t(e)] == 1 && !is_l1[size_t(l)][size_t(e)] && mu[size_t(e)] < cold_v) {
                        cold_v = mu[size_t(e)]; cold_p1 = e;
                    }
                }
                if (hot_mi < 0 || cold_p1 < 0 || hot_v <= cold_v) break;
                tier[size_t(hot_mi)]  = 1;
                tier[size_t(cold_p1)] = 0;
                const auto nxt = calc_cur();
                if (nxt.max_ms < cur.max_ms - 0.0015) {
                    continue;
                }
                tier[size_t(hot_mi)]  = 0;
                tier[size_t(cold_p1)] = 1;
                break;
            } else {
                break;
            }
        }

        const auto fc = calc_cur();
        plan.opt_mi50_sum_ms += fc.t_mi50_ms;
        plan.opt_p150_sum_ms += fc.t_p150_ms;
        plan.opt_max_sum_ms  += fc.max_ms;

        // Build zero-churn involutive bijection perm[l]
        std::vector<int> a_to_p150, b_to_mi50;
        for (int e = 0; e < 216; ++e) if (tier[size_t(e)] == 1) a_to_p150.push_back(e);
        for (int e = 216; e < 384; ++e) if (tier[size_t(e)] == 0) b_to_mi50.push_back(e);
        if (a_to_p150.size() != b_to_mi50.size()) {
            throw std::runtime_error("Tier cardinality mismatch at layer " + std::to_string(l));
        }

        for (int e = 0; e < 384; ++e) plan.perm[size_t(l)][size_t(e)] = static_cast<uint16_t>(e);
        for (size_t i = 0; i < a_to_p150.size(); ++i) {
            const int a = a_to_p150[i];
            const int b = b_to_mi50[i];
            plan.perm[size_t(l)][size_t(a)] = static_cast<uint16_t>(b);
            plan.perm[size_t(l)][size_t(b)] = static_cast<uint16_t>(a);
            plan.swapped_logical_pairs.push_back({l, a});
            plan.swapped_logical_pairs.push_back({l, b});
        }
        plan.total_swapped_pairs += static_cast<int>(a_to_p150.size());
    }

    for (const auto& pr : plan.l1_logical_pairs) {
        const uint16_t phys_e = plan.perm[size_t(pr.first)][size_t(pr.second)];
        if (phys_e < 216u || phys_e >= 384u) {
            throw std::runtime_error("Global L1 expert mapped outside [216..383]");
        }
        plan.l1_phys_assignments.push_back({static_cast<uint16_t>(pr.first), static_cast<uint16_t>(phys_e - 216u)});
    }
    return plan;
}

}  // namespace dsv41
