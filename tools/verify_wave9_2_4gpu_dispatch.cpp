// Wave 9.2 Deliverable: Pure C++17/HIP 4-GPU Numerical Parity & Zero-PCIe-AtomicOp IPC Harness
// Verifies M = {1, 2, 4, 8, 16, 128} across:
//   1. MTP_DEV3_HLUT3 (single-GPU unsharded WHT128-HLUT3 on hlut3_dev=3 + zero-TP3-AllReduce epilogue)
//   2. 3-way concurrent TIER_DEV3_HLUT3 (85:00.0) || TIER_TP3_HLUT3 (C7/88/03) || TIER_TP3_HLUT2 (C7/88/03)
//      combined via zero-PCIe-AtomicOp HostRingAllReduce against exact FP32 reference (cosine >= 0.999999).

#include "../src/hlut3_pack.hpp"
#include "../src/hlut2_pack.hpp"
#include "../mi50/dsv41_hlut3_kernels.hpp"
#include "../mi50/dsv41_mi50_kernels.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dsv41;

namespace {

#define V92_HIP_CHECK(call)                                                                      \
    do {                                                                                         \
        hipError_t _e = (call);                                                                  \
        if (_e != hipSuccess) {                                                                  \
            throw std::runtime_error(std::string(#call ": ") + hipGetErrorString(_e));           \
        }                                                                                        \
    } while (0)

struct ParityStats {
    double cosine = 0.0;
    double max_abs = 0.0;
};

ParityStats compute_parity(const std::vector<float>& ref, const std::vector<float>& got) {
    double dot = 0.0, nr = 0.0, ng = 0.0, max_a = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = double(ref[i]);
        const double g = double(got[i]);
        dot += r * g;
        nr  += r * r;
        ng  += g * g;
        max_a = std::max(max_a, std::fabs(r - g));
    }
    const double cos_v = (nr > 0.0 && ng > 0.0) ? (dot / (std::sqrt(nr) * std::sqrt(ng))) : 1.0;
    return ParityStats{cos_v, max_a};
}

// Packs Hlut2QuantMatrix [N, K] into MI50 tiled layout:
//   codes  : [N/32, K/16, 32] uint32_t
//   scales : [N/32, K/128, 32] uint16_t
void pack_hlut2_mi50_tiled(
    const Hlut2QuantMatrix& q,
    std::vector<uint32_t>& out_codes,
    std::vector<uint16_t>& out_scales,
    float out_cent[2])
{
    const int nb = q.N / 32;
    const int cw = q.K / 16;
    const int G  = q.K / 128;
    out_codes.resize(size_t(nb) * cw * 32u);
    out_scales.resize(size_t(nb) * G * 32u);
    for (int r = 0; r < q.N; ++r) {
        const int b = r / 32;
        const int lr = r % 32;
        for (int w = 0; w < cw; ++w) {
            out_codes[(size_t(b) * cw + w) * 32u + lr] = q.codes_rm[size_t(r) * cw + w];
        }
        for (int g = 0; g < G; ++g) {
            out_scales[(size_t(b) * G + g) * 32u + lr] = q.scales_fp16[size_t(r) * G + g];
        }
    }
    out_cent[0] = q.c0;
    out_cent[1] = q.c1;
}

// Exact host reference for a single expert using dequantized rotated weight matrices [2304,5120] / [5120,2304].
void accum_ref_expert_row(
    const float* x_rot_row,
    const std::vector<float>& wg_rot,
    const std::vector<float>& wu_rot,
    const std::vector<float>& wd_rot,
    const std::vector<uint64_t>& signs_i,
    float route_w,
    float* out_row_f32)
{
    constexpr int H = 5120;
    constexpr int I = 2304;
    std::vector<float> z(I, 0.0f);
    for (int i = 0; i < I; ++i) {
        const float* gr = wg_rot.data() + size_t(i) * H;
        const float* ur = wu_rot.data() + size_t(i) * H;
        double sg = 0.0, su = 0.0;
        for (int k = 0; k < H; ++k) {
            sg += double(x_rot_row[k]) * double(gr[k]);
            su += double(x_rot_row[k]) * double(ur[k]);
        }
        z[size_t(i)] = dsv41_swiglu_scalar(float(sg), float(su));
    }
    std::vector<float> zr(I, 0.0f);
    wht128_rows(z.data(), zr.data(), 1, I, signs_i);
    for (int d = 0; d < H; ++d) {
        const float* dr = wd_rot.data() + size_t(d) * I;
        double sd = 0.0;
        for (int i = 0; i < I; ++i) {
            sd += double(zr[size_t(i)]) * double(dr[i]);
        }
        out_row_f32[d] += float(sd) * route_w;
    }
}

void dequantize_hlut3_to_rot_f32(const Hlut3QuantMatrix& q, std::vector<float>& out_w_rot) {
    out_w_rot.assign(size_t(q.N) * size_t(q.K), 0.0f);
    for (int r = 0; r < q.N; ++r) {
        const int nt = r / 32;
        const int lr = r % 32;
        const int sg = lr >> 2;
        for (int kt = 0; kt < q.kt; ++kt) {
            const Hlut3Tile32x32& tile = q.tiles[size_t(nt) * size_t(q.kt) + size_t(kt)];
            const float sigma = fp16_to_f32(tile.scales_f16[sg]);
            for (int lc = 0; lc < 32; ++lc) {
                const int idx = lr * 32 + lc;
                const uint32_t mag  = (tile.q_lo2[idx >> 4] >> ((idx & 15) * 2)) & 3u;
                const uint32_t sign = (tile.q_hi1[idx >> 5] >> (idx & 31)) & 1u;
                const float val = float(q.lloyd.levels[mag]) * sigma;
                out_w_rot[size_t(r) * size_t(q.K) + size_t(kt * 32 + lc)] = sign ? -val : val;
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--link-check") {
            std::printf("[VERIFY_WAVE9_2_4GPU] Link check OK\n");
            return 0;
        }
    }

    int dev_cnt = 0;
    V92_HIP_CHECK(hipGetDeviceCount(&dev_cnt));
    if (dev_cnt < 4) {
        std::fprintf(stderr, "[VERIFY_WAVE9_2_4GPU] ERROR: requires 4 visible MI50 GPUs (HIP_VISIBLE_DEVICES=0,2,3,1); got %d\n", dev_cnt);
        return 1;
    }

    std::printf("=== WAVE 9.2 4-GPU DISPATCH & ZERO-PCIE-ATOMICOP PARITY HARNESS ===\n");
    for (int d = 0; d < 4; ++d) {
        hipDeviceProp_t prop{};
        V92_HIP_CHECK(hipGetDeviceProperties(&prop, d));
        size_t fb = 0, tb = 0;
        V92_HIP_CHECK(hipSetDevice(d));
        V92_HIP_CHECK(hipMemGetInfo(&fb, &tb));
        std::printf("  GPU[%d] %-16s pci=%04x:%02x:%02x.0 total=%.1f MiB free=%.1f MiB\n",
                    d, prop.gcnArchName, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID,
                    double(tb) / (1024.0 * 1024.0), double(fb) / (1024.0 * 1024.0));
    }

    constexpr int H = 5120;
    constexpr int I = 2304;
    constexpr int I_tp = 768;
    constexpr int kNExpPerTier = 2; // 2 HLUT2 + 2 TP3_HLUT3 + 2 DEV3_HLUT3 = 6 total routed experts

    std::mt19937 rng(20261119u);
    std::normal_distribution<float> nd(0.0f, 0.04f);

    std::vector<uint64_t> signs_h(H / 64), signs_i(I / 64);
    for (auto& w : signs_h) w = (uint64_t(rng()) << 32) | uint64_t(rng());
    for (auto& w : signs_i) w = (uint64_t(rng()) << 32) | uint64_t(rng());

    const auto sol = solve_gaussian_lloyd_max_hlut3();
    const float lv[4] = {float(sol.levels[0]), float(sol.levels[1]), float(sol.levels[2]), float(sol.levels[3])};

    // Build synthetic weights for:
    //   [0..1] TP3_HLUT2
    //   [2..3] TP3_HLUT3
    //   [4..5] DEV3_HLUT3 (also reused as 2 MTP unsharded HLUT3 experts)
    std::array<std::vector<float>, 6> ref_wg_rot{}, ref_wu_rot{}, ref_wd_rot{};

    // Host buffers for TP3_HLUT2 [3 ranks][2 experts]
    std::array<std::vector<uint32_t>, 3> h_hlut2_gc{}, h_hlut2_uc{}, h_hlut2_dc{};
    std::array<std::vector<uint16_t>, 3> h_hlut2_gs{}, h_hlut2_us{}, h_hlut2_ds{};
    std::array<std::vector<float>, 3>    h_hlut2_gm{}, h_hlut2_um{}, h_hlut2_dm{};

    // Host buffers for TP3_HLUT3 [3 ranks][2 experts]
    std::array<std::vector<Hlut3Tile32x32>, 3> h_tp3_gt{}, h_tp3_ut{}, h_tp3_dt{};

    // Host buffers for DEV3_HLUT3 [2 experts]
    std::vector<Hlut3Tile32x32> h_dev3_gt, h_dev3_ut, h_dev3_dt;

    auto gen_matrix = [&](int rows, int cols) {
        std::vector<float> m(size_t(rows) * size_t(cols));
        for (float& v : m) v = nd(rng);
        return m;
    };

    // Tier TP3_HLUT2 [0..1]
    for (int e = 0; e < kNExpPerTier; ++e) {
        const auto w1 = gen_matrix(I, H);
        const auto w3 = gen_matrix(I, H);
        const auto w2 = gen_matrix(H, I);
        ref_wg_rot[size_t(e)].assign(size_t(I) * H, 0.0f);
        ref_wu_rot[size_t(e)].assign(size_t(I) * H, 0.0f);
        ref_wd_rot[size_t(e)].assign(size_t(H) * I, 0.0f);

        for (int tp = 0; tp < 3; ++tp) {
            std::vector<uint64_t> si_tp(signs_i.begin() + tp * 12, signs_i.begin() + (tp + 1) * 12);
            std::vector<float> w1_tp(w1.begin() + size_t(tp * I_tp) * H, w1.begin() + size_t((tp + 1) * I_tp) * H);
            std::vector<float> w3_tp(w3.begin() + size_t(tp * I_tp) * H, w3.begin() + size_t((tp + 1) * I_tp) * H);
            std::vector<float> w2_tp(size_t(H) * I_tp);
            for (int r = 0; r < H; ++r) {
                std::memcpy(w2_tp.data() + size_t(r) * I_tp, w2.data() + size_t(r) * I + size_t(tp) * I_tp, I_tp * sizeof(float));
            }
            const auto qg = quantize_hlut2(w1_tp.data(), I_tp, H, signs_h);
            const auto qu = quantize_hlut2(w3_tp.data(), I_tp, H, signs_h);
            const auto qd = quantize_hlut2(w2_tp.data(), H, I_tp, si_tp);

            std::vector<float> dqg, dqu, dqd;
            dequantize_hlut2_to_rot_f32(qg, dqg);
            dequantize_hlut2_to_rot_f32(qu, dqu);
            dequantize_hlut2_to_rot_f32(qd, dqd);
            std::memcpy(ref_wg_rot[size_t(e)].data() + size_t(tp * I_tp) * H, dqg.data(), dqg.size() * sizeof(float));
            std::memcpy(ref_wu_rot[size_t(e)].data() + size_t(tp * I_tp) * H, dqu.data(), dqu.size() * sizeof(float));
            for (int r = 0; r < H; ++r) {
                std::memcpy(ref_wd_rot[size_t(e)].data() + size_t(r) * I + size_t(tp) * I_tp,
                            dqd.data() + size_t(r) * I_tp, I_tp * sizeof(float));
            }

            std::vector<uint32_t> gc, uc, dc;
            std::vector<uint16_t> gs, us, ds;
            float gm[2], um[2], dm[2];
            pack_hlut2_mi50_tiled(qg, gc, gs, gm);
            pack_hlut2_mi50_tiled(qu, uc, us, um);
            pack_hlut2_mi50_tiled(qd, dc, ds, dm);
            h_hlut2_gc[size_t(tp)].insert(h_hlut2_gc[size_t(tp)].end(), gc.begin(), gc.end());
            h_hlut2_gs[size_t(tp)].insert(h_hlut2_gs[size_t(tp)].end(), gs.begin(), gs.end());
            h_hlut2_gm[size_t(tp)].insert(h_hlut2_gm[size_t(tp)].end(), gm, gm + 2);
            h_hlut2_uc[size_t(tp)].insert(h_hlut2_uc[size_t(tp)].end(), uc.begin(), uc.end());
            h_hlut2_us[size_t(tp)].insert(h_hlut2_us[size_t(tp)].end(), us.begin(), us.end());
            h_hlut2_um[size_t(tp)].insert(h_hlut2_um[size_t(tp)].end(), um, um + 2);
            h_hlut2_dc[size_t(tp)].insert(h_hlut2_dc[size_t(tp)].end(), dc.begin(), dc.end());
            h_hlut2_ds[size_t(tp)].insert(h_hlut2_ds[size_t(tp)].end(), ds.begin(), ds.end());
            h_hlut2_dm[size_t(tp)].insert(h_hlut2_dm[size_t(tp)].end(), dm, dm + 2);
        }
    }

    // Tier TP3_HLUT3 [2..3] and Tier DEV3_HLUT3 [4..5]
    for (int e = 2; e < 6; ++e) {
        const auto w1 = gen_matrix(I, H);
        const auto w3 = gen_matrix(I, H);
        const auto w2 = gen_matrix(H, I);
        const auto qg = quantize_hlut3(w1.data(), I, H, signs_h);
        const auto qu = quantize_hlut3(w3.data(), I, H, signs_h);
        const auto qd = quantize_hlut3(w2.data(), H, I, signs_i);
        dequantize_hlut3_to_rot_f32(qg, ref_wg_rot[size_t(e)]);
        dequantize_hlut3_to_rot_f32(qu, ref_wu_rot[size_t(e)]);
        dequantize_hlut3_to_rot_f32(qd, ref_wd_rot[size_t(e)]);

        if (e < 4) {
            const auto sh = pack_hlut3_expert_tp3_sharded(qg, qu, qd);
            for (int tp = 0; tp < 3; ++tp) {
                h_tp3_gt[size_t(tp)].insert(h_tp3_gt[size_t(tp)].end(), sh[size_t(tp)].gate_tiles.begin(), sh[size_t(tp)].gate_tiles.end());
                h_tp3_ut[size_t(tp)].insert(h_tp3_ut[size_t(tp)].end(), sh[size_t(tp)].up_tiles.begin(),   sh[size_t(tp)].up_tiles.end());
                h_tp3_dt[size_t(tp)].insert(h_tp3_dt[size_t(tp)].end(), sh[size_t(tp)].down_tiles.begin(), sh[size_t(tp)].down_tiles.end());
            }
        } else {
            const auto un = pack_hlut3_expert_unsharded(qg, qu, qd);
            h_dev3_gt.insert(h_dev3_gt.end(), un.gate_tiles.begin(), un.gate_tiles.end());
            h_dev3_ut.insert(h_dev3_ut.end(), un.up_tiles.begin(),   un.up_tiles.end());
            h_dev3_dt.insert(h_dev3_dt.end(), un.down_tiles.begin(), un.down_tiles.end());
        }
    }

    // Allocate per-GPU resources across tp=0,1,2 and dev3=3
    constexpr int kMaxM = 128;
    constexpr int kTopK = 6;
    constexpr int kMaxP = kMaxM * kTopK;

    struct TpTestCtx {
        hipStream_t stream = nullptr;
        uint64_t* d_signs_i = nullptr;
        float* d_levels = nullptr;
        uint16_t* d_x_rot_f16 = nullptr;
        int32_t* d_topk_ids = nullptr;
        float* d_topk_w = nullptr;
        Mi50RoutedPair* d_pairs2 = nullptr;
        Mi50RoutedPair* d_pairs3 = nullptr;
        int32_t* d_act = nullptr;
        int32_t* d_csr = nullptr;
        uint16_t* d_inter_bf16 = nullptr;
        uint16_t* d_inter_rot_f16 = nullptr;
        float* d_z3 = nullptr;
        float* d_zrot3 = nullptr;
        float* d_part_f32 = nullptr;
        float* d_sum_f32 = nullptr;
        uint16_t* d_res_bf16 = nullptr;
        float* d_post_f32 = nullptr;
        float* d_comb_f32 = nullptr;
        uint16_t* d_out_h_bf16 = nullptr;
        uint32_t* d_stage_sync = nullptr;

        uint32_t *d_gc = nullptr, *d_uc = nullptr, *d_dc = nullptr;
        uint16_t *d_gs = nullptr, *d_us = nullptr, *d_ds = nullptr;
        float    *d_gm = nullptr, *d_um = nullptr, *d_dm = nullptr;
        Hlut3Tile32x32 *d_gt3 = nullptr, *d_ut3 = nullptr, *d_dt3 = nullptr;
    };
    std::array<TpTestCtx, 3> tp_ctx{};

    auto alloc_copy = [](const auto& vec) {
        using T = typename std::decay_t<decltype(vec)>::value_type;
        T* p = nullptr;
        V92_HIP_CHECK(hipMalloc(&p, vec.size() * sizeof(T)));
        V92_HIP_CHECK(hipMemcpy(p, vec.data(), vec.size() * sizeof(T), hipMemcpyHostToDevice));
        return p;
    };

    for (int tp = 0; tp < 3; ++tp) {
        V92_HIP_CHECK(hipSetDevice(tp));
        auto& c = tp_ctx[size_t(tp)];
        V92_HIP_CHECK(hipStreamCreate(&c.stream));
        std::vector<uint64_t> si_tp(signs_i.begin() + tp * 12, signs_i.begin() + (tp + 1) * 12);
        c.d_signs_i = alloc_copy(si_tp);
        std::vector<float> lv_v(lv, lv + 4);
        c.d_levels  = alloc_copy(lv_v);

        V92_HIP_CHECK(hipMalloc(&c.d_x_rot_f16, kMaxM * H * sizeof(uint16_t)));
        V92_HIP_CHECK(hipMalloc(&c.d_topk_ids,  kMaxP * sizeof(int32_t)));
        V92_HIP_CHECK(hipMalloc(&c.d_topk_w,    kMaxP * sizeof(float)));
        V92_HIP_CHECK(hipMalloc(&c.d_pairs2,    kMaxP * sizeof(Mi50RoutedPair)));
        V92_HIP_CHECK(hipMalloc(&c.d_pairs3,    kMaxP * sizeof(Mi50RoutedPair)));
        V92_HIP_CHECK(hipMalloc(&c.d_act,       256 * sizeof(int32_t)));
        V92_HIP_CHECK(hipMalloc(&c.d_csr,       257 * sizeof(int32_t)));
        V92_HIP_CHECK(hipMalloc(&c.d_inter_bf16,    kMaxP * I_tp * sizeof(uint16_t)));
        V92_HIP_CHECK(hipMalloc(&c.d_inter_rot_f16, kMaxP * I_tp * sizeof(uint16_t)));
        V92_HIP_CHECK(hipMalloc(&c.d_z3,            kMaxP * I_tp * sizeof(float)));
        V92_HIP_CHECK(hipMalloc(&c.d_zrot3,         kMaxP * I_tp * sizeof(float)));
        V92_HIP_CHECK(hipMalloc(&c.d_part_f32,      kMaxM * H * sizeof(float)));
        V92_HIP_CHECK(hipMalloc(&c.d_sum_f32,       kMaxM * H * sizeof(float)));
        V92_HIP_CHECK(hipMalloc(&c.d_res_bf16,      kMaxM * 4 * H * sizeof(uint16_t)));
        V92_HIP_CHECK(hipMalloc(&c.d_post_f32,      kMaxM * 4 * sizeof(float)));
        V92_HIP_CHECK(hipMalloc(&c.d_comb_f32,      kMaxM * 16 * sizeof(float)));
        V92_HIP_CHECK(hipMalloc(&c.d_out_h_bf16,    kMaxM * 4 * H * sizeof(uint16_t)));
        V92_HIP_CHECK(hipMalloc(&c.d_stage_sync,    64 * sizeof(uint32_t)));
        V92_HIP_CHECK(hipMemset(c.d_stage_sync, 0,  64 * sizeof(uint32_t)));

        c.d_gc = alloc_copy(h_hlut2_gc[size_t(tp)]);
        c.d_gs = alloc_copy(h_hlut2_gs[size_t(tp)]);
        c.d_gm = alloc_copy(h_hlut2_gm[size_t(tp)]);
        c.d_uc = alloc_copy(h_hlut2_uc[size_t(tp)]);
        c.d_us = alloc_copy(h_hlut2_us[size_t(tp)]);
        c.d_um = alloc_copy(h_hlut2_um[size_t(tp)]);
        c.d_dc = alloc_copy(h_hlut2_dc[size_t(tp)]);
        c.d_ds = alloc_copy(h_hlut2_ds[size_t(tp)]);
        c.d_dm = alloc_copy(h_hlut2_dm[size_t(tp)]);

        c.d_gt3 = alloc_copy(h_tp3_gt[size_t(tp)]);
        c.d_ut3 = alloc_copy(h_tp3_ut[size_t(tp)]);
        c.d_dt3 = alloc_copy(h_tp3_dt[size_t(tp)]);
    }

    // Allocate dev3 (GPU 3 = 0000:85:00.0) resources
    V92_HIP_CHECK(hipSetDevice(3));
    hipStream_t dev3_stream = nullptr;
    V92_HIP_CHECK(hipStreamCreate(&dev3_stream));
    uint64_t* d3_signs_i = alloc_copy(signs_i);
    float*    d3_levels  = alloc_copy(std::vector<float>(lv, lv + 4));
    Hlut3Tile32x32* d3_gt = alloc_copy(h_dev3_gt);
    Hlut3Tile32x32* d3_ut = alloc_copy(h_dev3_ut);
    Hlut3Tile32x32* d3_dt = alloc_copy(h_dev3_dt);
    uint16_t* d3_x_rot_f16 = nullptr;
    Mi50RoutedPair* d3_pairs = nullptr;
    int32_t* d3_act = nullptr;
    int32_t* d3_csr = nullptr;
    float *d3_z = nullptr, *d3_zrot = nullptr, *d3_out_f32 = nullptr;
    uint32_t* d3_pub_sync = nullptr;
    V92_HIP_CHECK(hipMalloc(&d3_x_rot_f16, kMaxM * H * sizeof(uint16_t)));
    V92_HIP_CHECK(hipMalloc(&d3_pairs,     kMaxP * sizeof(Mi50RoutedPair)));
    V92_HIP_CHECK(hipMalloc(&d3_act,       256 * sizeof(int32_t)));
    V92_HIP_CHECK(hipMalloc(&d3_csr,       257 * sizeof(int32_t)));
    V92_HIP_CHECK(hipMalloc(&d3_z,         kMaxP * I * sizeof(float)));
    V92_HIP_CHECK(hipMalloc(&d3_zrot,      kMaxP * I * sizeof(float)));
    V92_HIP_CHECK(hipMalloc(&d3_out_f32,   kMaxM * H * sizeof(float)));
    V92_HIP_CHECK(hipMalloc(&d3_pub_sync,  64 * sizeof(uint32_t)));
    V92_HIP_CHECK(hipMemset(d3_pub_sync, 0, 64 * sizeof(uint32_t)));

    // Coherent mapped host DDR4 slots
    V92_HIP_CHECK(hipSetDevice(0));
    HostRingAllReduceSlot* h_ar_ring = nullptr;
    Hlut3OffloadSlot* h_dev3_slot = nullptr;
    float* h_dev3_prefill_y = nullptr;
    V92_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_ar_ring),
                                kHostArRingSlots * sizeof(HostRingAllReduceSlot),
                                hipHostMallocMapped | hipHostMallocCoherent));
    V92_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_dev3_slot),
                                sizeof(Hlut3OffloadSlot),
                                hipHostMallocMapped | hipHostMallocCoherent));
    V92_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_dev3_prefill_y),
                                kMaxM * H * sizeof(float),
                                hipHostMallocMapped | hipHostMallocCoherent));
    std::memset(h_ar_ring, 0, kHostArRingSlots * sizeof(HostRingAllReduceSlot));
    std::memset(h_dev3_slot, 0, sizeof(Hlut3OffloadSlot));

    // Set mHC post to pure identity pass-through into hyper-stream 0 (ps = [1,0,0,0], comb = 0, res = 0)
    // so we can verify both exact FP32 y_out and post-mHC BF16 rank equivalence!
    std::vector<uint16_t> zero_res(kMaxM * 4 * H, 0u);
    std::vector<float> post_one(kMaxM * 4, 0.0f);
    std::vector<float> comb_zero(kMaxM * 16, 0.0f);
    for (int m = 0; m < kMaxM; ++m) post_one[size_t(m) * 4 + 0] = 1.0f;
    for (int tp = 0; tp < 3; ++tp) {
        V92_HIP_CHECK(hipSetDevice(tp));
        auto& c = tp_ctx[size_t(tp)];
        V92_HIP_CHECK(hipMemcpy(c.d_res_bf16, zero_res.data(), zero_res.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        V92_HIP_CHECK(hipMemcpy(c.d_post_f32, post_one.data(), post_one.size() * sizeof(float), hipMemcpyHostToDevice));
        V92_HIP_CHECK(hipMemcpy(c.d_comb_f32, comb_zero.data(), comb_zero.size() * sizeof(float), hipMemcpyHostToDevice));
    }

    uint64_t ar_seq = 0;
    uint32_t dev3_seq = 0;
    bool all_ok = true;

    const int kBatchSizes[6] = {1, 2, 4, 8, 16, 128};
    for (int M : kBatchSizes) {
        std::vector<uint16_t> h_x_rot_f16(size_t(M) * H);
        std::vector<float>    h_x_rot_f32(size_t(M) * H);
        for (size_t i = 0; i < h_x_rot_f16.size(); ++i) {
            const uint16_t h = f32_to_fp16_rne(nd(rng) * 8.0f);
            h_x_rot_f16[i] = h;
            h_x_rot_f32[i] = fp16_to_f32(h);
        }

        // ---- TEST 1: MTP_DEV3_HLUT3 (Zero TP3 AllReduce) ----
        std::vector<int32_t> mtp_ids(size_t(M) * 2);
        std::vector<float>   mtp_w(size_t(M) * 2);
        std::vector<float>   ref_mtp(size_t(M) * H, 0.0f);
        for (int r = 0; r < M; ++r) {
            mtp_ids[size_t(r) * 2 + 0] = 0;
            mtp_ids[size_t(r) * 2 + 1] = 1;
            mtp_w[size_t(r) * 2 + 0]   = 0.62f;
            mtp_w[size_t(r) * 2 + 1]   = 0.38f;
            accum_ref_expert_row(h_x_rot_f32.data() + size_t(r) * H,
                                 ref_wg_rot[4], ref_wu_rot[4], ref_wd_rot[4], signs_i, 0.62f,
                                 ref_mtp.data() + size_t(r) * H);
            accum_ref_expert_row(h_x_rot_f32.data() + size_t(r) * H,
                                 ref_wg_rot[5], ref_wu_rot[5], ref_wd_rot[5], signs_i, 0.38f,
                                 ref_mtp.data() + size_t(r) * H);
        }

        std::vector<float> got_mtp_f32(size_t(M) * H, 0.0f);
        if (M <= kHlut3OffloadMaxRows) {
            const uint32_t seq = ++dev3_seq;
            V92_HIP_CHECK(hipSetDevice(0));
            auto& c0 = tp_ctx[0];
            V92_HIP_CHECK(hipMemcpyAsync(c0.d_x_rot_f16, h_x_rot_f16.data(), h_x_rot_f16.size() * sizeof(uint16_t), hipMemcpyHostToDevice, c0.stream));
            V92_HIP_CHECK(hipMemcpyAsync(c0.d_topk_ids, mtp_ids.data(), mtp_ids.size() * sizeof(int32_t), hipMemcpyHostToDevice, c0.stream));
            V92_HIP_CHECK(hipMemcpyAsync(c0.d_topk_w, mtp_w.data(), mtp_w.size() * sizeof(float), hipMemcpyHostToDevice, c0.stream));
            launch_stage_and_arm_hlut3_dev3(
                h_dev3_slot, c0.d_stage_sync, c0.d_topk_ids, c0.d_topk_w,
                c0.d_x_rot_f16, nullptr, 40u, M, 2, 0, 2, 0u, seq, c0.stream);

            // Launch zero-AllReduce rank epilogues BEFORE dev3 even starts (tests GPU spin-wait on done_seq!)
            for (int tp = 0; tp < 3; ++tp) {
                V92_HIP_CHECK(hipSetDevice(tp));
                auto& c = tp_ctx[size_t(tp)];
                launch_dev3_finish_f32_hc_post(
                    h_dev3_slot->y_out_f32, &h_dev3_slot->done_seq, &h_dev3_slot->rank_done_seq[tp][0], seq,
                    c.d_res_bf16, c.d_post_f32, c.d_comb_f32, c.d_out_h_bf16, M, c.stream);
            }

            while (static_cast<int32_t>(__atomic_load_n(&h_dev3_slot->ready_seq, __ATOMIC_ACQUIRE) - seq) < 0) {}
            V92_HIP_CHECK(hipSetDevice(3));
            launch_dev3_ingest_and_pack(h_dev3_slot, d3_x_rot_f16, nullptr, d3_pairs, M, 2, 0, 2, false, dev3_stream);
            V92_HIP_CHECK(hipMemsetAsync(d3_out_f32, 0, size_t(M) * H * sizeof(float), dev3_stream));
            launch_hlut3_unsharded_pairs_f16(
                d3_x_rot_f16, d3_pairs, M * 2, kMaxP, 2, d3_gt, d3_ut, d3_dt,
                d3_levels, d3_signs_i, d3_z, d3_zrot, d3_out_f32, dev3_stream);
            launch_dev3_publish_f32(d3_out_f32, h_dev3_slot, d3_pub_sync, M, seq, dev3_stream);
            V92_HIP_CHECK(hipStreamSynchronize(dev3_stream));

            for (int tp = 0; tp < 3; ++tp) {
                V92_HIP_CHECK(hipSetDevice(tp));
                V92_HIP_CHECK(hipStreamSynchronize(tp_ctx[size_t(tp)].stream));
            }
            std::memcpy(got_mtp_f32.data(), h_dev3_slot->y_out_f32, got_mtp_f32.size() * sizeof(float));
        } else {
            // M = 128 grouped CSR on dev3
            V92_HIP_CHECK(hipSetDevice(3));
            V92_HIP_CHECK(hipMemcpyAsync(d3_x_rot_f16, h_x_rot_f16.data(), h_x_rot_f16.size() * sizeof(uint16_t), hipMemcpyHostToDevice, dev3_stream));
            int32_t h_act[2] = {0, 1};
            int32_t h_off[3] = {0, M, 2 * M};
            std::vector<Mi50RoutedPair> h_sp(size_t(2 * M));
            for (int r = 0; r < M; ++r) {
                h_sp[size_t(r)]     = Mi50RoutedPair{r, 0, 0.62f, 0};
                h_sp[size_t(M + r)] = Mi50RoutedPair{r, 1, 0.38f, 0};
            }
            V92_HIP_CHECK(hipMemcpyAsync(d3_act, h_act, sizeof(h_act), hipMemcpyHostToDevice, dev3_stream));
            V92_HIP_CHECK(hipMemcpyAsync(d3_csr, h_off, sizeof(h_off), hipMemcpyHostToDevice, dev3_stream));
            V92_HIP_CHECK(hipMemcpyAsync(d3_pairs, h_sp.data(), h_sp.size() * sizeof(Mi50RoutedPair), hipMemcpyHostToDevice, dev3_stream));
            V92_HIP_CHECK(hipMemsetAsync(d3_out_f32, 0, size_t(M) * H * sizeof(float), dev3_stream));
            launch_hlut3_routed_grouped_f16(
                d3_x_rot_f16, d3_act, d3_csr, d3_pairs, 2, 2 * M, 2, I, H, H, I,
                d3_gt, d3_ut, d3_dt, d3_levels, d3_signs_i, d3_z, d3_zrot, d3_out_f32, dev3_stream);
            V92_HIP_CHECK(hipMemcpyAsync(got_mtp_f32.data(), d3_out_f32, got_mtp_f32.size() * sizeof(float), hipMemcpyDeviceToHost, dev3_stream));
            V92_HIP_CHECK(hipStreamSynchronize(dev3_stream));
        }
        const auto st_mtp = compute_parity(ref_mtp, got_mtp_f32);

        // ---- TEST 2: 3-Way Concurrent Backbone (TP3_HLUT2 + TP3_HLUT3 + DEV3_HLUT3) ----
        std::vector<int32_t> bb_ids(size_t(M) * kTopK);
        std::vector<float>   bb_w(size_t(M) * kTopK);
        std::vector<float>   ref_bb(size_t(M) * H, 0.0f);
        const float kw[kTopK] = {0.25f, 0.20f, 0.18f, 0.15f, 0.12f, 0.10f};
        for (int r = 0; r < M; ++r) {
            for (int k = 0; k < kTopK; ++k) {
                bb_ids[size_t(r) * kTopK + k] = k; // hits physical 0,1 (HLUT2), 2,3 (TP3_HLUT3), 4,5 (DEV3_HLUT3)
                bb_w[size_t(r) * kTopK + k]   = kw[k];
                accum_ref_expert_row(h_x_rot_f32.data() + size_t(r) * H,
                                     ref_wg_rot[size_t(k)], ref_wu_rot[size_t(k)], ref_wd_rot[size_t(k)],
                                     signs_i, kw[k], ref_bb.data() + size_t(r) * H);
            }
        }

        for (int tp = 0; tp < 3; ++tp) {
            V92_HIP_CHECK(hipSetDevice(tp));
            auto& c = tp_ctx[size_t(tp)];
            V92_HIP_CHECK(hipMemcpyAsync(c.d_x_rot_f16, h_x_rot_f16.data(), h_x_rot_f16.size() * sizeof(uint16_t), hipMemcpyHostToDevice, c.stream));
            V92_HIP_CHECK(hipMemcpyAsync(c.d_topk_ids,  bb_ids.data(), bb_ids.size() * sizeof(int32_t), hipMemcpyHostToDevice, c.stream));
            V92_HIP_CHECK(hipMemcpyAsync(c.d_topk_w,    bb_w.data(), bb_w.size() * sizeof(float), hipMemcpyHostToDevice, c.stream));
            V92_HIP_CHECK(hipMemsetAsync(c.d_part_f32,  0, size_t(M) * H * sizeof(float), c.stream));
        }

        std::vector<float> got_bb_f32(size_t(M) * H, 0.0f);
        if (M <= 16) {
            const uint32_t seq = ++dev3_seq;
            V92_HIP_CHECK(hipSetDevice(0));
            launch_stage_and_arm_hlut3_dev3(
                h_dev3_slot, tp_ctx[0].d_stage_sync, tp_ctx[0].d_topk_ids, tp_ctx[0].d_topk_w,
                tp_ctx[0].d_x_rot_f16, nullptr, 0u, M, kTopK, 4, 6, 0u, seq, tp_ctx[0].stream);

            for (int tp = 0; tp < 3; ++tp) {
                V92_HIP_CHECK(hipSetDevice(tp));
                auto& c = tp_ctx[size_t(tp)];
                launch_pack_mi50_routed_pairs(c.d_topk_ids, c.d_topk_w, c.d_pairs2, M, kTopK, 2, c.stream);
                launch_mi50_routed_hlut2_pairs(
                    c.d_x_rot_f16, c.d_pairs2, M * kTopK,
                    c.d_gc, c.d_gs, c.d_gm, c.d_uc, c.d_us, c.d_um, c.d_dc, c.d_ds, c.d_dm,
                    c.d_signs_i, c.d_inter_bf16, c.d_inter_rot_f16, c.d_part_f32, c.stream);

                launch_pack_ranged_routed_pairs(c.d_topk_ids, c.d_topk_w, c.d_pairs3, M, kTopK, 2, 4, c.stream);
                launch_hlut3_tp3_pairs_f16(
                    c.d_x_rot_f16, c.d_pairs3, M * kTopK, kMaxP, 2,
                    c.d_gt3, c.d_ut3, c.d_dt3, c.d_levels, c.d_signs_i, c.d_z3, c.d_zrot3,
                    c.d_part_f32, c.stream);
            }

            const uint64_t aseq = ++ar_seq;
            HostRingAllReduceSlot* aslot = &h_ar_ring[(aseq - 1ULL) & (kHostArRingSlots - 1ULL)];
            for (int tp = 0; tp < 3; ++tp) {
                V92_HIP_CHECK(hipSetDevice(tp));
                auto& c = tp_ctx[size_t(tp)];
                launch_host_ring_upload_f32(c.d_part_f32, aslot, tp, aseq, M, c.stream);
                launch_host_ring_finish_f32_hc_post(
                    c.d_part_f32, nullptr, c.d_res_bf16, c.d_post_f32, c.d_comb_f32, c.d_out_h_bf16,
                    aslot, tp, aseq, M, c.stream, nullptr, 0u, nullptr,
                    h_dev3_slot->y_out_f32, &h_dev3_slot->done_seq, seq);
            }

            while (static_cast<int32_t>(__atomic_load_n(&h_dev3_slot->ready_seq, __ATOMIC_ACQUIRE) - seq) < 0) {}
            V92_HIP_CHECK(hipSetDevice(3));
            launch_dev3_ingest_and_pack(h_dev3_slot, d3_x_rot_f16, nullptr, d3_pairs, M, kTopK, 4, 6, false, dev3_stream);
            V92_HIP_CHECK(hipMemsetAsync(d3_out_f32, 0, size_t(M) * H * sizeof(float), dev3_stream));
            launch_hlut3_unsharded_pairs_f16(
                d3_x_rot_f16, d3_pairs, M * kTopK, kMaxP, 2, d3_gt, d3_ut, d3_dt,
                d3_levels, d3_signs_i, d3_z, d3_zrot, d3_out_f32, dev3_stream);
            launch_dev3_publish_f32(d3_out_f32, h_dev3_slot, d3_pub_sync, M, seq, dev3_stream);
            V92_HIP_CHECK(hipStreamSynchronize(dev3_stream));

            // Also compute exact FP32 combined sum via HostRingAllReduce + dev3 to check pre-BF16 FP32 cosine!
            for (int tp = 0; tp < 3; ++tp) {
                V92_HIP_CHECK(hipSetDevice(tp));
                V92_HIP_CHECK(hipStreamSynchronize(tp_ctx[size_t(tp)].stream));
            }
            // Reconstruct exact FP32 sum from uploaded ar_slot->payload[0..2] + h_dev3_slot->y_out_f32
            for (int i = 0; i < M * H; ++i) {
                float s0, s1, s2;
                std::memcpy(&s0, &aslot->payload[0][i], 4);
                std::memcpy(&s1, &aslot->payload[1][i], 4);
                std::memcpy(&s2, &aslot->payload[2][i], 4);
                got_bb_f32[size_t(i)] = (s0 + s1 + s2) + h_dev3_slot->y_out_f32[i];
            }
        } else {
            // M = 128 grouped CSR across TP3_HLUT2 + TP3_HLUT3 + DEV3_HLUT3
            V92_HIP_CHECK(hipSetDevice(3));
            V92_HIP_CHECK(hipMemcpyAsync(d3_x_rot_f16, h_x_rot_f16.data(), h_x_rot_f16.size() * sizeof(uint16_t), hipMemcpyHostToDevice, dev3_stream));
            int32_t h_act[2] = {0, 1};
            int32_t h_off[3] = {0, M, 2 * M};
            std::vector<Mi50RoutedPair> h_sp(size_t(2 * M));
            for (int r = 0; r < M; ++r) {
                h_sp[size_t(r)]     = Mi50RoutedPair{r, 0, kw[4], 0};
                h_sp[size_t(M + r)] = Mi50RoutedPair{r, 1, kw[5], 0};
            }
            V92_HIP_CHECK(hipMemcpyAsync(d3_act, h_act, sizeof(h_act), hipMemcpyHostToDevice, dev3_stream));
            V92_HIP_CHECK(hipMemcpyAsync(d3_csr, h_off, sizeof(h_off), hipMemcpyHostToDevice, dev3_stream));
            V92_HIP_CHECK(hipMemcpyAsync(d3_pairs, h_sp.data(), h_sp.size() * sizeof(Mi50RoutedPair), hipMemcpyHostToDevice, dev3_stream));
            V92_HIP_CHECK(hipMemsetAsync(d3_out_f32, 0, size_t(M) * H * sizeof(float), dev3_stream));
            launch_hlut3_routed_grouped_f16(
                d3_x_rot_f16, d3_act, d3_csr, d3_pairs, 2, 2 * M, 2, I, H, H, I,
                d3_gt, d3_ut, d3_dt, d3_levels, d3_signs_i, d3_z, d3_zrot, d3_out_f32, dev3_stream);
            V92_HIP_CHECK(hipMemcpyAsync(h_dev3_prefill_y, d3_out_f32, size_t(M) * H * sizeof(float), hipMemcpyDeviceToHost, dev3_stream));

            for (int tp = 0; tp < 3; ++tp) {
                V92_HIP_CHECK(hipSetDevice(tp));
                auto& c = tp_ctx[size_t(tp)];
                launch_build_mi50_csr(c.d_topk_ids, c.d_topk_w, c.d_act, c.d_csr, c.d_pairs2, M, kTopK, 2, c.stream);
                launch_mi50_routed_hlut2_grouped(
                    c.d_x_rot_f16, c.d_act, c.d_csr, c.d_pairs2, M, 2, M * kTopK,
                    c.d_gc, c.d_gs, c.d_gm, c.d_uc, c.d_us, c.d_um, c.d_dc, c.d_ds, c.d_dm,
                    c.d_signs_i, c.d_inter_bf16, c.d_inter_rot_f16, c.d_part_f32, c.stream);

                launch_build_ranged_csr(c.d_topk_ids, c.d_topk_w, c.d_act, c.d_csr, c.d_pairs3, M, kTopK, 2, 4, c.stream);
                launch_hlut3_routed_grouped_f16(
                    c.d_x_rot_f16, c.d_act, c.d_csr, c.d_pairs3, 2, M * kTopK, 2,
                    I_tp, H, H, I_tp, c.d_gt3, c.d_ut3, c.d_dt3, c.d_levels, c.d_signs_i,
                    c.d_z3, c.d_zrot3, c.d_part_f32, c.stream);
            }

            V92_HIP_CHECK(hipSetDevice(3));
            V92_HIP_CHECK(hipStreamSynchronize(dev3_stream));
            V92_HIP_CHECK(hipSetDevice(0));
            launch_add_f32_inplace(tp_ctx[0].d_part_f32, h_dev3_prefill_y, size_t(M) * H, tp_ctx[0].stream);

            const size_t total_elems = size_t(M) * H;
            for (size_t off = 0; off < total_elems; off += kHostArMaxWords) {
                const size_t tile = std::min<size_t>(kHostArMaxWords, total_elems - off);
                const uint64_t aseq = ++ar_seq;
                HostRingAllReduceSlot* aslot = &h_ar_ring[(aseq - 1ULL) & (kHostArRingSlots - 1ULL)];
                for (int tp = 0; tp < 3; ++tp) {
                    V92_HIP_CHECK(hipSetDevice(tp));
                    auto& c = tp_ctx[size_t(tp)];
                    launch_host_ring_allreduce_f32(c.d_part_f32 + off, c.d_sum_f32 + off, aslot, tp, aseq, tile, c.stream);
                }
            }
            for (int tp = 0; tp < 3; ++tp) {
                V92_HIP_CHECK(hipSetDevice(tp));
                V92_HIP_CHECK(hipStreamSynchronize(tp_ctx[size_t(tp)].stream));
            }
            V92_HIP_CHECK(hipSetDevice(0));
            V92_HIP_CHECK(hipMemcpy(got_bb_f32.data(), tp_ctx[0].d_sum_f32, got_bb_f32.size() * sizeof(float), hipMemcpyDeviceToHost));
        }

        // Verify rank0 == rank1 == rank2 bit-exactness on out_h_bf16 (for M <= 16) or d_sum_f32 (for M = 128)
        bool ranks_equal = true;
        if (M <= 16) {
            std::array<std::vector<uint16_t>, 3> r_out{};
            for (int tp = 0; tp < 3; ++tp) {
                r_out[size_t(tp)].resize(size_t(M) * H);
                V92_HIP_CHECK(hipSetDevice(tp));
                V92_HIP_CHECK(hipMemcpy(r_out[size_t(tp)].data(), tp_ctx[size_t(tp)].d_out_h_bf16,
                                        size_t(M) * H * sizeof(uint16_t), hipMemcpyDeviceToHost));
            }
            ranks_equal = (r_out[0] == r_out[1]) && (r_out[0] == r_out[2]);
        } else {
            std::vector<float> s1(size_t(M) * H), s2(size_t(M) * H);
            V92_HIP_CHECK(hipSetDevice(1));
            V92_HIP_CHECK(hipMemcpy(s1.data(), tp_ctx[1].d_sum_f32, s1.size() * sizeof(float), hipMemcpyDeviceToHost));
            V92_HIP_CHECK(hipSetDevice(2));
            V92_HIP_CHECK(hipMemcpy(s2.data(), tp_ctx[2].d_sum_f32, s2.size() * sizeof(float), hipMemcpyDeviceToHost));
            ranks_equal = (got_bb_f32 == s1) && (got_bb_f32 == s2);
        }

        const auto st_bb = compute_parity(ref_bb, got_bb_f32);
        const bool pass = (st_mtp.cosine >= 0.999999) && (st_bb.cosine >= 0.999999) && ranks_equal;
        all_ok &= pass;

        std::printf("  [M=%3d] MTP_DEV3 cos=%.9f max_abs=%.3e | 3WAY_BB cos=%.9f max_abs=%.3e | tp_eq=%d -> %s\n",
                    M, st_mtp.cosine, st_mtp.max_abs, st_bb.cosine, st_bb.max_abs,
                    ranks_equal ? 1 : 0, pass ? "PASS" : "FAIL");
        std::fflush(stdout);
    }

    std::printf("=== VERIFY_WAVE9_2_4GPU RESULT: %s ===\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
