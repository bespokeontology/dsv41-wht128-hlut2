// Wave 9 Deliverable D verifier:
//   (1) pure analytic Gaussian Lloyd-Max fixed-point laws and SQNR vs HLUT2,
//   (2) exact 400 B CPU pack/unpack and 6,400 B P150A superblock equivalence,
//   (3) live MI50 device-2 HIP vs CPU reference parity >= 0.999999 for M = 1, 6, 8, 16.
// Safety: hipSetDevice(2) is the first HIP runtime call; free HBM2 must be >= 550 MiB;
// total device allocations are hard-capped <= 48 MiB. No /dev/tenstorrent, no IPC.
// Pure C++20. No Python.

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
constexpr double kParityCosMin   = 0.999999;

void hip_check(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        throw std::runtime_error(std::string("HIP ") + what + ": " + hipGetErrorString(e));
    }
}

struct DeviceBudget {
    size_t total = 0;
    std::vector<void*> ptrs;
    void* alloc(size_t bytes) {
        if (bytes == 0) return nullptr;
        if (total + bytes > kMaxDeviceBytes) {
            throw std::runtime_error("Wave9 verifier HBM2 allocation exceeds strict 48 MiB cap");
        }
        void* p = nullptr;
        hip_check(hipMalloc(&p, bytes), "hipMalloc");
        total += bytes;
        ptrs.push_back(p);
        return p;
    }
    ~DeviceBudget() {
        for (void* p : ptrs) {
            if (p) (void)hipFree(p);
        }
    }
};

template <typename T>
T* dev_alloc(DeviceBudget& b, size_t count) {
    return static_cast<T*>(b.alloc(std::max<size_t>(count * sizeof(T), 64u)));
}

template <typename T>
void h2d(T* dst, const std::vector<T>& src) {
    hip_check(hipMemcpy(dst, src.data(), src.size() * sizeof(T), hipMemcpyHostToDevice), "H2D");
}

template <typename T, std::size_t N>
void h2d(T* dst, const std::array<T, N>& src) {
    hip_check(hipMemcpy(dst, src.data(), src.size() * sizeof(T), hipMemcpyHostToDevice), "H2D");
}

template <typename T>
void d2h(std::vector<T>& dst, const T* src, size_t count) {
    dst.resize(count);
    hip_check(hipMemcpy(dst.data(), src, count * sizeof(T), hipMemcpyDeviceToHost), "D2H");
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

dsv41::QualityMetrics quality(const std::vector<float>& ref, const std::vector<float>& test) {
    return dsv41::compute_quality(ref, test);
}

bool check_solver() {
    const auto s3 = dsv41::solve_gaussian_lloyd_max_hlut3();
    const auto s2 = dsv41::solve_gaussian_lloyd_max_hlut2();
    bool ok = s3.converged && s3.convergence_residual <= 1e-7;
    ok &= s3.mse > 0.0 && std::isfinite(s3.sqnr_db);
    ok &= std::fabs(s3.eq2 - (1.0 - s3.mse)) <= 1e-10;
    ok &= std::fabs(s3.fixed_point_residual) <= 1e-7;
    ok &= s3.levels[0] > 0.0;
    for (int i = 1; i < 4; ++i) ok &= s3.levels[i] > s3.levels[i - 1];
    for (int i = 1; i < 3; ++i) ok &= s3.thresholds[i] > s3.thresholds[i - 1];
    ok &= std::fabs(s3.thresholds[0] - 0.5 * (s3.levels[0] + s3.levels[1])) <= 1e-12;
    ok &= std::fabs(s3.thresholds[1] - 0.5 * (s3.levels[1] + s3.levels[2])) <= 1e-12;
    ok &= std::fabs(s3.thresholds[2] - 0.5 * (s3.levels[2] + s3.levels[3])) <= 1e-12;
    ok &= s3.sqnr_db > s2.sqnr_db;
    std::printf("[MATH] HLUT3 c=[%.12f %.12f %.12f %.12f] t=[%.12f %.12f %.12f]\n",
                s3.levels[0], s3.levels[1], s3.levels[2], s3.levels[3],
                s3.thresholds[0], s3.thresholds[1], s3.thresholds[2]);
    std::printf("[MATH] HLUT3 MSE=%.12f EQ2=%.12f SQNR=%.6f dB iters=%d delta=%.3e fixed=%.3e\n",
                s3.mse, s3.eq2, s3.sqnr_db, s3.iterations, s3.convergence_residual, s3.fixed_point_residual);
    std::printf("[MATH] HLUT2 MSE=%.12f SQNR=%.6f dB | gain=+%.6f dB\n",
                s2.mse, s2.sqnr_db, s3.sqnr_db - s2.sqnr_db);
    return ok;
}

bool check_tile_pack_unpack() {
    std::mt19937_64 rng(0x9A11C3ULL);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const auto sol = dsv41::solve_gaussian_lloyd_max_hlut3();
    std::vector<float> w(size_t(32) * 32);
    for (float& v : w) v = dist(rng);

    dsv41::Hlut3Tile32x32 t0{}, t1{};
    dsv41::hlut3_pack_tile(w.data(), 32, 0, 0, sol, t0);
    dsv41::hlut3_pack_tile(w.data(), 32, 0, 0, sol, t1);
    if (std::memcmp(&t0, &t1, sizeof(t0)) != 0) return false;

    std::vector<float> u(size_t(32) * 32);
    dsv41::hlut3_unpack_tile(t0, sol, u.data(), 32, 0, 0);
    for (int r = 0; r < 32; ++r) {
        for (int c = 0; c < 32; ++c) {
            const int idx = dsv41::hlut3_tile_flat(r, c);
            const int g = dsv41::hlut3_group(idx);
            const float sigma = dsv41::fp16_to_f32(t0.scales_f16[g]);
            const uint32_t mag = (t0.q_lo2[dsv41::hlut3_mag_word(idx)] >> dsv41::hlut3_mag_shift(idx)) & 3u;
            const uint32_t sign = (t0.q_hi1[dsv41::hlut3_sign_word(idx)] >> dsv41::hlut3_sign_shift(idx)) & 1u;
            const float expect = (sign ? -1.0f : 1.0f) * float(sol.levels[mag]) * sigma;
            if (u[size_t(r) * 32 + size_t(c)] != expect) return false;
        }
    }
    std::printf("[PACK] tile deterministic, 400 B, q_lo2/q_hi1/scales exact pack-unpack match\n");
    return true;
}

bool check_superblock_equivalence() {
    constexpr int N = 256;
    constexpr int K = 128;
    std::mt19937_64 rng(0x6A15ULL);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> W(size_t(N) * K);
    for (float& v : W) v = nd(rng);
    const auto signs = dsv41::make_wht_signs(K, 0xB4515009ULL);
    const auto q = dsv41::quantize_hlut3(W.data(), N, K, signs);

    const auto sbs = dsv41::pack_hlut3_for_p150_shard(q, 0, N);
    if (sbs.size() != size_t(N / 128) * size_t(K / 128) ||
        sbs.size() * sizeof(dsv41::Hlut3Superblock128x128) != size_t(N) * size_t(K) * 400ULL / 1024ULL) {
        return false;
    }

    std::vector<float> direct(size_t(N) * K, 0.0f);
    for (int nt_i = 0; nt_i < q.nt; ++nt_i) {
        for (int kt_i = 0; kt_i < q.kt; ++kt_i) {
            dsv41::hlut3_unpack_tile(q.tiles[size_t(nt_i) * size_t(q.kt) + size_t(kt_i)],
                                     q.lloyd, direct.data(), K,
                                     nt_i * 32, kt_i * 32);
        }
    }
    std::vector<float> from_sb(size_t(N) * K, 0.0f);
    dsv41::unpack_hlut3_p150_superblocks(sbs, q.lloyd, K, N, 0, from_sb.data());
    if (std::memcmp(direct.data(), from_sb.data(), direct.size() * sizeof(float)) != 0) return false;
    std::printf("[P150] 16-subtile 6,400 B superblock equivalence over %zu blocks (%zu B exact)\n",
                sbs.size(), sbs.size() * sizeof(dsv41::Hlut3Superblock128x128));
    return true;
}

struct ParityCase {
    int M = 0;
    double cosine = 0.0;
    double max_abs = 0.0;
};

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
    // FIRST HIP runtime call on this process: lock device 2.
    hip_check(hipSetDevice(2), "hipSetDevice(2)");

    size_t free_b = 0, total_b = 0;
    hip_check(hipMemGetInfo(&free_b, &total_b), "hipMemGetInfo");
    std::printf("[SAFETY] device=2 gfx906 free=%.2f MiB total=%.2f MiB min_free=550 MiB cap=48 MiB\n",
                double(free_b) / (1024.0 * 1024.0),
                double(total_b) / (1024.0 * 1024.0));
    if (free_b < kMinFreeBytes) {
        std::fprintf(stderr, "[SAFETY] device-2 free HBM2 %.2f MiB < 550 MiB; refusing HIP workload\n",
                     double(free_b) / (1024.0 * 1024.0));
        return 2;
    }

    bool ok = check_solver();
    ok &= check_tile_pack_unpack();
    ok &= check_superblock_equivalence();

    dsv41::p150_hlut3_geometry_report();

    // Real production shard-derived weights (read-only), H=I HLUT3.
    dsv41::SwigluExpertF32 ex;
    const std::string scales_path =
        dsv41_root_path("shards_skhynix/layer6_expert0_scales_e8m0.bin", "DSV41_SCALES_ROOT");
    const std::string weights_path =
        dsv41_root_path("shards_greatwall_data/layer6_expert0_weights_mxfp4.bin", "DSV41_NVME1_ROOT");
    if (!dsv41::load_real_dsv41_expert0(scales_path, weights_path, ex)) {
        std::fprintf(stderr, "[PARITY] real layer6 expert0 shards missing\n");
        return 3;
    }

    const auto signs_h = dsv41::make_hlut3_signs_h(6, dsv41::DSV41_H);
    const auto signs_i = dsv41::make_hlut3_signs_i(6, dsv41::DSV41_I);
    const auto qg = dsv41::quantize_hlut3(ex.gate.data(), dsv41::DSV41_I, dsv41::DSV41_H, signs_h);
    const auto qu = dsv41::quantize_hlut3(ex.up.data(),   dsv41::DSV41_I, dsv41::DSV41_H, signs_h);
    const auto qd = dsv41::quantize_hlut3(ex.down.data(), dsv41::DSV41_H, dsv41::DSV41_I, signs_i);
    if (qg.raw_bytes() != 4'608'000ULL || qu.raw_bytes() != 4'608'000ULL || qd.raw_bytes() != 4'608'000ULL) {
        std::fprintf(stderr, "[PARITY] real HLUT3 tile bytes are not exact 3.125 bpw\n");
        return 4;
    }

    std::vector<float> gate_rot(size_t(qg.N) * size_t(qg.K));
    std::vector<float> up_rot(size_t(qu.N) * size_t(qu.K));
    std::vector<float> down_rot(size_t(qd.N) * size_t(qd.K));
    dsv41::unpack_hlut3_to_rot_f32(qg, gate_rot);
    dsv41::unpack_hlut3_to_rot_f32(qu, up_rot);
    dsv41::unpack_hlut3_to_rot_f32(qd, down_rot);

    std::printf("[WEIGHTS] real layer6 expert0 2304x5120: HLUT3 gate=%zu B up=%zu B down=%zu B (%.3f bpw)\n",
                qg.raw_bytes(), qu.raw_bytes(), qd.raw_bytes(), qg.bits_per_weight());

    DeviceBudget budget;
    auto* d_gate_tiles = static_cast<dsv41::Hlut3Tile32x32*>(
        budget.alloc(qg.tiles.size() * sizeof(dsv41::Hlut3Tile32x32)));
    auto* d_up_tiles = static_cast<dsv41::Hlut3Tile32x32*>(
        budget.alloc(qu.tiles.size() * sizeof(dsv41::Hlut3Tile32x32)));
    auto* d_down_tiles = static_cast<dsv41::Hlut3Tile32x32*>(
        budget.alloc(qd.tiles.size() * sizeof(dsv41::Hlut3Tile32x32)));
    hip_check(hipMemcpy(d_gate_tiles, qg.tiles.data(), qg.tiles.size() * sizeof(dsv41::Hlut3Tile32x32),
                        hipMemcpyHostToDevice), "gate tiles H2D");
    hip_check(hipMemcpy(d_up_tiles, qu.tiles.data(), qu.tiles.size() * sizeof(dsv41::Hlut3Tile32x32),
                        hipMemcpyHostToDevice), "up tiles H2D");
    hip_check(hipMemcpy(d_down_tiles, qd.tiles.data(), qd.tiles.size() * sizeof(dsv41::Hlut3Tile32x32),
                        hipMemcpyHostToDevice), "down tiles H2D");

    const std::array<float, 4> levels = {
        float(qg.lloyd.levels[0]), float(qg.lloyd.levels[1]),
        float(qg.lloyd.levels[2]), float(qg.lloyd.levels[3])
    };
    float* d_levels = dev_alloc<float>(budget, levels.size());
    h2d(d_levels, levels);

    uint64_t* d_signs_i = dev_alloc<uint64_t>(budget, signs_i.size());
    h2d(d_signs_i, signs_i);

    auto* d_xrot = dev_alloc<float>(budget, size_t(16) * size_t(dsv41::DSV41_H));
    auto* d_pairs = dev_alloc<dsv41::Mi50RoutedPair>(budget, 16);
    auto* d_z = dev_alloc<float>(budget, size_t(16) * size_t(dsv41::DSV41_I));
    auto* d_zrot = dev_alloc<float>(budget, size_t(16) * size_t(dsv41::DSV41_I));
    auto* d_y = dev_alloc<float>(budget, size_t(16) * size_t(dsv41::DSV41_H));

    std::printf("[ALLOC] device-2 strict total = %.3f MiB <= 48 MiB\n",
                double(budget.total) / (1024.0 * 1024.0));

    std::vector<ParityCase> cases;
    for (int M : {1, 6, 8, 16}) {
        std::vector<float> X = dsv41::make_outlier_activations(M, dsv41::DSV41_H, 0xD541CA11ULL + uint64_t(M));
        std::vector<float> Xrot(size_t(M) * size_t(dsv41::DSV41_H));
        dsv41::wht128_rows(X.data(), Xrot.data(), M, dsv41::DSV41_H, signs_h);

        // CPU reference pipeline on the exact unpacked quantized matrices.
        std::vector<float> g(size_t(M) * size_t(dsv41::DSV41_I));
        std::vector<float> u(size_t(M) * size_t(dsv41::DSV41_I));
        std::vector<float> z(size_t(M) * size_t(dsv41::DSV41_I));
        std::vector<float> zrot(size_t(M) * size_t(dsv41::DSV41_I));
        std::vector<float> y_cpu(size_t(M) * size_t(dsv41::DSV41_H));
        matmul_nt_f32(Xrot.data(), gate_rot.data(), g.data(), M, dsv41::DSV41_I, dsv41::DSV41_H);
        matmul_nt_f32(Xrot.data(), up_rot.data(),   u.data(), M, dsv41::DSV41_I, dsv41::DSV41_H);
        for (size_t i = 0; i < z.size(); ++i) z[i] = dsv41::dsv41_swiglu_scalar(g[i], u[i]);
        dsv41::wht128_rows(z.data(), zrot.data(), M, dsv41::DSV41_I, signs_i);
        matmul_nt_f32(zrot.data(), down_rot.data(), y_cpu.data(), M, dsv41::DSV41_H, dsv41::DSV41_I);

        std::vector<dsv41::Mi50RoutedPair> pairs(static_cast<size_t>(M));
        for (int m = 0; m < M; ++m) {
            pairs[size_t(m)] = dsv41::Mi50RoutedPair{m, 0, 1.0f, 0};
        }
        h2d(d_xrot, Xrot);
        h2d(d_pairs, pairs);
        hip_check(hipMemset(d_y, 0, size_t(M) * size_t(dsv41::DSV41_H) * sizeof(float)), "zero y");
        dsv41::launch_mi50_routed_hlut3_pairs(
            d_xrot, d_pairs, M, M, /*n_experts=*/1,
            dsv41::DSV41_I, dsv41::DSV41_H, dsv41::DSV41_H, dsv41::DSV41_I,
            d_gate_tiles, d_up_tiles, d_down_tiles,
            d_levels, d_levels, d_levels,
            d_signs_i, d_z, d_zrot, d_y,
            nullptr);
        hip_check(hipDeviceSynchronize(), "device sync");

        std::vector<float> y_gpu;
        d2h(y_gpu, d_y, size_t(M) * size_t(dsv41::DSV41_H));
        const dsv41::QualityMetrics qm = quality(y_cpu, y_gpu);
        ParityCase pc{M, qm.cosine, qm.max_abs};
        cases.push_back(pc);
        ok &= (qm.cosine >= kParityCosMin);
        std::printf("[PARITY] M=%2d cosine=%.9f max_abs=%.6e %s\n",
                    M, qm.cosine, qm.max_abs, (qm.cosine >= kParityCosMin) ? "PASS" : "FAIL");
    }

    bool parity_all = true;
    double worst_cos = 1.0;
    double worst_abs = 0.0;
    for (const auto& c : cases) {
        parity_all &= (c.cosine >= kParityCosMin);
        worst_cos = std::min(worst_cos, c.cosine);
        worst_abs = std::max(worst_abs, c.max_abs);
    }
    ok &= parity_all;

    std::printf("\n=== WAVE9_HLUT3_VERIFY RECEIPT ===\n");
    std::printf("solver_convergence_le_1e-7 : %s\n", check_solver() ? "PASS" : "FAIL");
    std::printf("tile_400B_pack_unpack_exact : %s\n", check_tile_pack_unpack() ? "PASS" : "FAIL");
    std::printf("p150_6400B_superblock_eq    : %s\n", check_superblock_equivalence() ? "PASS" : "FAIL");
    std::printf("device2_free_hbm2_ge_550MiB : %.2f MiB PASS\n", double(free_b) / (1024.0 * 1024.0));
    std::printf("device2_total_alloc_le_48MiB: %.3f MiB PASS\n", double(budget.total) / (1024.0 * 1024.0));
    std::printf("hip_vs_cpu_parity_M1_6_8_16 : min_cos=%.9f max_abs=%.6e %s\n",
                worst_cos, worst_abs, parity_all ? "PASS" : "FAIL");
    std::printf("real_weight_hlut3_bytes     : 4,608,000 B/matrix (3.125 bpw exact)\n");
    std::printf("overall                     : %s\n", ok ? "PASS" : "FAIL");
    std::fflush(stdout);
    return ok ? 0 : 1;
}
