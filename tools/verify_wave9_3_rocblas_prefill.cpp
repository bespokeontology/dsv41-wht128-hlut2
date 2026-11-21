// Wave 9.3 Phase B Parity & Speed Gate:
// Verifies RocblasPrefillEngine (FP16->FP32 v_dot2_f32_f16 Tensor-Op GEMM) against
// canonical gfx906 custom kernels across:
//   1. Dense FP8 E4M3 + UE8M0_32 GEMM ([512, 12288, 1280], [512, 5120, 3072], [512, 1792, 5120])
//   2. Grouped wo_a FP8 GEMM (M=512, g_cnt=3)
//   3. Dense BF16 GEMM ([512, 384, 5120])
//   4. Routed HLUT2 Expert Bucket (TP3 N_inter=768, m_e=64)
//   5. Routed WHT128-HLUT3 Expert Bucket (TP3 N_inter=768 & dev3 unsharded N_inter=2304, m_e=64)
// Requires cosine >= 0.9999 on every single test! 100% Pure C++17 / HIP — Zero Python.

#include "dsv41_wave9_3_rocblas.hpp"
#include "dsv41_hlut3_kernels.hpp"
#include "dsv41_mi50_kernels.hpp"
#include "../src/hlut2_pack.hpp"
#include "../src/hlut3_pack.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
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

void check_hip(hipError_t e, const char* w) {
    if (e != hipSuccess) throw std::runtime_error(std::string(w) + ": " + hipGetErrorString(e));
}

float bf16_to_f32_h(uint16_t b) noexcept {
    const uint32_t u = uint32_t(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, 4);
    return f;
}

uint16_t f32_to_bf16_h(float f) noexcept {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

uint16_t f32_to_f16_h(float f) noexcept {
    return dsv41::f32_to_fp16_rne(f);
}

double cosine_f32(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += double(a[i]) * double(b[i]);
        na  += double(a[i]) * double(a[i]);
        nb  += double(b[i]) * double(b[i]);
    }
    return dot / (std::sqrt(na * nb) + 1e-18);
}

double cosine_bf16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double fa = double(bf16_to_f32_h(a[i]));
        const double fb = double(bf16_to_f32_h(b[i]));
        dot += fa * fb;
        na  += fa * fa;
        nb  += fb * fb;
    }
    return dot / (std::sqrt(na * nb) + 1e-18);
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--link-check") {
            std::printf("verify_wave9_3_rocblas_prefill: link-check OK\n");
            return 0;
        }
    }

    try {
        check_hip(hipSetDevice(0), "hipSetDevice(0)");
        hipStream_t stream = nullptr;
        check_hip(hipStreamCreate(&stream), "hipStreamCreate");

        RocblasPrefillEngine rb_tp3;
        rb_tp3.init(0, stream, /*is_dev3=*/false);
        std::printf("[W93_VERIFY] TP3 RocblasPrefillEngine allocated %.2f MiB HBM2\n",
                    double(rb_tp3.total_allocated_bytes()) / (1024.0 * 1024.0));

        std::mt19937 rng(42);
        std::normal_distribution<float> nd(0.0f, 0.35f);

        // 1. Dense FP8 E4M3 + UE8M0_32 GEMM Parity & Speed ([512, 12288, 1280] and [512, 1792, 5120])
        struct Fp8Case { const char* name; int M; int N; int K; };
        const Fp8Case fp8_cases[] = {
            {"wqkv [512x1792x5120]",   512,  1792, 5120},
            {"wq_b [512x12288x1280]",  512, 12288, 1280},
            {"wo_b [512x5120x3072]",   512,  5120, 3072},
            {"sh_w13 [512x1536x5120]", 512,  1536, 5120},
        };
        for (const auto& tc : fp8_cases) {
            std::vector<uint8_t> hx(size_t(tc.M) * size_t(tc.K));
            std::vector<uint8_t> hxs(size_t(tc.M) * size_t(tc.K / 32), 124u);
            std::vector<uint8_t> hw(size_t(tc.N) * size_t(tc.K));
            std::vector<uint8_t> hws(size_t(tc.N / 32) * size_t(tc.K / 32), 123u);
            for (auto& b : hx) b = uint8_t((rng() % 112u) | ((rng() & 1u) << 7));
            for (auto& b : hw) b = uint8_t((rng() % 112u) | ((rng() & 1u) << 7));

            uint8_t *dx = nullptr, *dxs = nullptr, *dw = nullptr, *dws = nullptr;
            uint16_t *dy_ref = nullptr, *dy_rb = nullptr;
            check_hip(hipMalloc(&dx, hx.size()), "dx");
            check_hip(hipMalloc(&dxs, hxs.size()), "dxs");
            check_hip(hipMalloc(&dw, hw.size()), "dw");
            check_hip(hipMalloc(&dws, hws.size()), "dws");
            check_hip(hipMalloc(&dy_ref, size_t(tc.M) * size_t(tc.N) * 2u), "dy_ref");
            check_hip(hipMalloc(&dy_rb,  size_t(tc.M) * size_t(tc.N) * 2u), "dy_rb");
            check_hip(hipMemcpy(dx, hx.data(), hx.size(), hipMemcpyHostToDevice), "cp dx");
            check_hip(hipMemcpy(dxs, hxs.data(), hxs.size(), hipMemcpyHostToDevice), "cp dxs");
            check_hip(hipMemcpy(dw, hw.data(), hw.size(), hipMemcpyHostToDevice), "cp dw");
            check_hip(hipMemcpy(dws, hws.data(), hws.size(), hipMemcpyHostToDevice), "cp dws");

            // Warmup + time reference
            launch_fp8_gemm(dx, dxs, dw, dws, dy_ref, nullptr, tc.M, tc.N, tc.K, false, false, stream);
            check_hip(hipStreamSynchronize(stream), "sync ref warm");
            const auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < 5; ++it) {
                launch_fp8_gemm(dx, dxs, dw, dws, dy_ref, nullptr, tc.M, tc.N, tc.K, false, false, stream);
            }
            check_hip(hipStreamSynchronize(stream), "sync ref");
            const double ref_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5.0;

            // Warmup + time rocBLAS
            bool ok = rb_tp3.run_fp8_gemm(dx, dxs, dw, dws, dy_rb, nullptr, tc.M, tc.N, tc.K, false, false, stream);
            if (!ok) throw std::runtime_error(std::string("run_fp8_gemm returned false for ") + tc.name);
            check_hip(hipStreamSynchronize(stream), "sync rb warm");
            const auto t1 = std::chrono::steady_clock::now();
            for (int it = 0; it < 5; ++it) {
                (void)rb_tp3.run_fp8_gemm(dx, dxs, dw, dws, dy_rb, nullptr, tc.M, tc.N, tc.K, false, false, stream);
            }
            check_hip(hipStreamSynchronize(stream), "sync rb");
            const double rb_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / 5.0;

            std::vector<uint16_t> hy_ref(size_t(tc.M) * size_t(tc.N));
            std::vector<uint16_t> hy_rb(size_t(tc.M) * size_t(tc.N));
            check_hip(hipMemcpy(hy_ref.data(), dy_ref, hy_ref.size() * 2u, hipMemcpyDeviceToHost), "d2h ref");
            check_hip(hipMemcpy(hy_rb.data(),  dy_rb,  hy_rb.size() * 2u,  hipMemcpyDeviceToHost), "d2h rb");
            const double cos_v = cosine_bf16(hy_ref, hy_rb);
            std::printf("[FP8] %-24s | cos=%.8f | ref=%.3f ms | rocBLAS=%.3f ms (%.2fx)\n",
                        tc.name, cos_v, ref_ms, rb_ms, ref_ms / std::max(1e-6, rb_ms));
            if (cos_v < 0.9999) throw std::runtime_error(std::string("FP8 cosine failed on ") + tc.name);
            (void)hipFree(dx); (void)hipFree(dxs); (void)hipFree(dw); (void)hipFree(dws);
            (void)hipFree(dy_ref); (void)hipFree(dy_rb);
        }

        // 2. Grouped wo_a FP8 GEMM (M=512, g_cnt=3)
        {
            const int M = 512, g_cnt = 3, K_tot = g_cnt * 4096, N_tot = g_cnt * 1024;
            std::vector<uint8_t> ho(size_t(M) * size_t(K_tot));
            std::vector<uint8_t> hos(size_t(M) * size_t(K_tot / 32), 124u);
            std::vector<uint8_t> hw(size_t(N_tot) * 4096u);
            std::vector<uint8_t> hws(size_t(N_tot / 32) * 128u, 123u);
            for (auto& b : ho) b = uint8_t((rng() % 112u) | ((rng() & 1u) << 7));
            for (auto& b : hw) b = uint8_t((rng() % 112u) | ((rng() & 1u) << 7));

            uint8_t *do_p = nullptr, *dos = nullptr, *dw = nullptr, *dws = nullptr;
            uint16_t *dy_ref = nullptr, *dy_rb = nullptr;
            check_hip(hipMalloc(&do_p, ho.size()), "do");
            check_hip(hipMalloc(&dos, hos.size()), "dos");
            check_hip(hipMalloc(&dw, hw.size()), "dw");
            check_hip(hipMalloc(&dws, hws.size()), "dws");
            check_hip(hipMalloc(&dy_ref, size_t(M) * size_t(N_tot) * 2u), "dy_ref");
            check_hip(hipMalloc(&dy_rb,  size_t(M) * size_t(N_tot) * 2u), "dy_rb");
            check_hip(hipMemcpy(do_p, ho.data(), ho.size(), hipMemcpyHostToDevice), "cp do");
            check_hip(hipMemcpy(dos, hos.data(), hos.size(), hipMemcpyHostToDevice), "cp dos");
            check_hip(hipMemcpy(dw, hw.data(), hw.size(), hipMemcpyHostToDevice), "cp dw");
            check_hip(hipMemcpy(dws, hws.data(), hws.size(), hipMemcpyHostToDevice), "cp dws");

            launch_fp8_grouped_wo_a(do_p, dos, dw, dws, dy_ref, M, g_cnt, stream);
            check_hip(hipStreamSynchronize(stream), "wo_a ref warm");
            const auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < 5; ++it) launch_fp8_grouped_wo_a(do_p, dos, dw, dws, dy_ref, M, g_cnt, stream);
            check_hip(hipStreamSynchronize(stream), "wo_a ref");
            const double ref_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5.0;

            bool ok = rb_tp3.run_fp8_grouped_wo_a(do_p, dos, dw, dws, dy_rb, M, g_cnt, stream);
            if (!ok) throw std::runtime_error("run_fp8_grouped_wo_a returned false");
            check_hip(hipStreamSynchronize(stream), "wo_a rb warm");
            const auto t1 = std::chrono::steady_clock::now();
            for (int it = 0; it < 5; ++it) (void)rb_tp3.run_fp8_grouped_wo_a(do_p, dos, dw, dws, dy_rb, M, g_cnt, stream);
            check_hip(hipStreamSynchronize(stream), "wo_a rb");
            const double rb_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / 5.0;

            std::vector<uint16_t> hy_ref(size_t(M) * size_t(N_tot)), hy_rb(size_t(M) * size_t(N_tot));
            check_hip(hipMemcpy(hy_ref.data(), dy_ref, hy_ref.size() * 2u, hipMemcpyDeviceToHost), "d2h");
            check_hip(hipMemcpy(hy_rb.data(),  dy_rb,  hy_rb.size() * 2u,  hipMemcpyDeviceToHost), "d2h");
            const double cos_v = cosine_bf16(hy_ref, hy_rb);
            std::printf("[WOA] grouped_wo_a [512x3x1024x4096] | cos=%.8f | ref=%.3f ms | rocBLAS=%.3f ms (%.2fx)\n",
                        cos_v, ref_ms, rb_ms, ref_ms / std::max(1e-6, rb_ms));
            if (cos_v < 0.9999) throw std::runtime_error("wo_a cosine failed");
            (void)hipFree(do_p); (void)hipFree(dos); (void)hipFree(dw); (void)hipFree(dws);
            (void)hipFree(dy_ref); (void)hipFree(dy_rb);
        }

        // 3. Dense BF16 GEMM Parity & Speed ([512, 384, 5120])
        {
            const int M = 512, N = 384, K = 5120;
            std::vector<uint16_t> hx(size_t(M) * size_t(K)), hw(size_t(N) * size_t(K));
            for (auto& v : hx) v = f32_to_bf16_h(nd(rng));
            for (auto& v : hw) v = f32_to_bf16_h(nd(rng));
            uint16_t *dx = nullptr, *dw = nullptr;
            float *dy_ref = nullptr, *dy_rb = nullptr;
            check_hip(hipMalloc(&dx, hx.size() * 2u), "dx");
            check_hip(hipMalloc(&dw, hw.size() * 2u), "dw");
            check_hip(hipMalloc(&dy_ref, size_t(M) * size_t(N) * 4u), "dy_ref");
            check_hip(hipMalloc(&dy_rb,  size_t(M) * size_t(N) * 4u), "dy_rb");
            check_hip(hipMemcpy(dx, hx.data(), hx.size() * 2u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dw, hw.data(), hw.size() * 2u, hipMemcpyHostToDevice), "cp");

            launch_bf16_gemm(dx, dw, nullptr, dy_ref, M, N, K, true, stream);
            check_hip(hipStreamSynchronize(stream), "bf16 ref");
            const auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < 5; ++it) launch_bf16_gemm(dx, dw, nullptr, dy_ref, M, N, K, true, stream);
            check_hip(hipStreamSynchronize(stream), "bf16 ref");
            const double ref_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5.0;

            bool ok = rb_tp3.run_bf16_gemm(dx, dw, nullptr, dy_rb, M, N, K, true, stream);
            if (!ok) throw std::runtime_error("run_bf16_gemm returned false");
            check_hip(hipStreamSynchronize(stream), "bf16 rb");
            const auto t1 = std::chrono::steady_clock::now();
            for (int it = 0; it < 5; ++it) (void)rb_tp3.run_bf16_gemm(dx, dw, nullptr, dy_rb, M, N, K, true, stream);
            check_hip(hipStreamSynchronize(stream), "bf16 rb");
            const double rb_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / 5.0;

            std::vector<float> hy_ref(size_t(M) * size_t(N)), hy_rb(size_t(M) * size_t(N));
            check_hip(hipMemcpy(hy_ref.data(), dy_ref, hy_ref.size() * 4u, hipMemcpyDeviceToHost), "d2h");
            check_hip(hipMemcpy(hy_rb.data(),  dy_rb,  hy_rb.size() * 4u,  hipMemcpyDeviceToHost), "d2h");
            const double cos_v = cosine_f32(hy_ref, hy_rb);
            std::printf("[BF16] gate_w [512x384x5120] | cos=%.8f | ref=%.3f ms | rocBLAS=%.3f ms (%.2fx)\n",
                        cos_v, ref_ms, rb_ms, ref_ms / std::max(1e-6, rb_ms));
            if (cos_v < 0.9999) throw std::runtime_error("BF16 cosine failed");
            (void)hipFree(dx); (void)hipFree(dw); (void)hipFree(dy_ref); (void)hipFree(dy_rb);
        }

        // 4. Routed HLUT2 Expert Bucket Parity & Speed (TP3 N_inter=768, m_e=64)
        {
            const int M = 64, Ni = 768, H = 5120;
            std::vector<uint16_t> hx(size_t(M) * H);
            for (auto& v : hx) v = f32_to_f16_h(nd(rng));
            std::vector<uint32_t> hgc(245760u), huc(245760u), hdc(245760u);
            std::vector<uint16_t> hgs(30720u, f32_to_f16_h(0.08f)), hus(30720u, f32_to_f16_h(0.08f)), hds(30720u, f32_to_f16_h(0.08f));
            for (auto& w : hgc) w = rng();
            for (auto& w : huc) w = rng();
            for (auto& w : hdc) w = rng();
            float hcent[2] = {0.45f, 1.51f};
            std::vector<uint64_t> hsigns(12u, 0x55AA33CC11223344ULL);
            std::vector<Mi50RoutedPair> hpairs(M);
            for (int i = 0; i < M; ++i) hpairs[i] = Mi50RoutedPair{i, 0, 0.25f + 0.01f * float(i % 7), 0};
            int32_t hact[1] = {0}, hoff[2] = {0, M};

            uint16_t *dx = nullptr, *dgs = nullptr, *dus = nullptr, *dds = nullptr, *dib = nullptr, *dir = nullptr;
            uint32_t *dgc = nullptr, *duc = nullptr, *ddc = nullptr;
            float *dcent = nullptr, *dout_ref = nullptr, *dout_rb = nullptr;
            uint64_t *dsigns = nullptr;
            Mi50RoutedPair* dpairs = nullptr;
            int32_t *dact = nullptr, *doff = nullptr;
            check_hip(hipMalloc(&dx, hx.size() * 2u), "dx");
            check_hip(hipMalloc(&dgc, hgc.size() * 4u), "dgc"); check_hip(hipMalloc(&duc, huc.size() * 4u), "duc"); check_hip(hipMalloc(&ddc, hdc.size() * 4u), "ddc");
            check_hip(hipMalloc(&dgs, hgs.size() * 2u), "dgs"); check_hip(hipMalloc(&dus, hus.size() * 2u), "dus"); check_hip(hipMalloc(&dds, hds.size() * 2u), "dds");
            check_hip(hipMalloc(&dcent, 8u), "dcent"); check_hip(hipMalloc(&dsigns, 12u * 8u), "dsigns");
            check_hip(hipMalloc(&dpairs, hpairs.size() * sizeof(Mi50RoutedPair)), "dpairs");
            check_hip(hipMalloc(&dact, 4u), "dact"); check_hip(hipMalloc(&doff, 8u), "doff");
            check_hip(hipMalloc(&dib, size_t(M) * Ni * 2u), "dib"); check_hip(hipMalloc(&dir, size_t(M) * Ni * 2u), "dir");
            check_hip(hipMalloc(&dout_ref, size_t(M) * H * 4u), "dout_ref"); check_hip(hipMalloc(&dout_rb, size_t(M) * H * 4u), "dout_rb");

            check_hip(hipMemcpy(dx, hx.data(), hx.size() * 2u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dgc, hgc.data(), hgc.size() * 4u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(duc, huc.data(), huc.size() * 4u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(ddc, hdc.data(), hdc.size() * 4u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dgs, hgs.data(), hgs.size() * 2u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dus, hus.data(), hus.size() * 2u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dds, hds.data(), hds.size() * 2u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dcent, hcent, 8u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dsigns, hsigns.data(), 12u * 8u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dpairs, hpairs.data(), hpairs.size() * sizeof(Mi50RoutedPair), hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dact, hact, 4u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(doff, hoff, 8u, hipMemcpyHostToDevice), "cp");

            check_hip(hipMemsetAsync(dout_ref, 0, size_t(M) * H * 4u, stream), "z");
            launch_mi50_routed_hlut2_grouped(dx, dact, doff, dpairs, M, 1, M, dgc, dgs, dcent, duc, dus, dcent, ddc, dds, dcent, dsigns, dib, dir, dout_ref, stream);
            check_hip(hipStreamSynchronize(stream), "hlut2 ref warm");
            const auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < 5; ++it) {
                check_hip(hipMemsetAsync(dout_ref, 0, size_t(M) * H * 4u, stream), "z");
                launch_mi50_routed_hlut2_grouped(dx, dact, doff, dpairs, M, 1, M, dgc, dgs, dcent, duc, dus, dcent, ddc, dds, dcent, dsigns, dib, dir, dout_ref, stream);
            }
            check_hip(hipStreamSynchronize(stream), "hlut2 ref");
            const double ref_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5.0;

            check_hip(hipMemsetAsync(dout_rb, 0, size_t(M) * H * 4u, stream), "z");
            rb_tp3.run_hlut2_expert_bucket(dx, dpairs, 0, 0, M, dgc, dgs, dcent, duc, dus, dcent, ddc, dds, dcent, dsigns, dout_rb, stream);
            check_hip(hipStreamSynchronize(stream), "hlut2 rb warm");
            const auto t1 = std::chrono::steady_clock::now();
            for (int it = 0; it < 5; ++it) {
                check_hip(hipMemsetAsync(dout_rb, 0, size_t(M) * H * 4u, stream), "z");
                rb_tp3.run_hlut2_expert_bucket(dx, dpairs, 0, 0, M, dgc, dgs, dcent, duc, dus, dcent, ddc, dds, dcent, dsigns, dout_rb, stream);
            }
            check_hip(hipStreamSynchronize(stream), "hlut2 rb");
            const double rb_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / 5.0;

            std::vector<float> hy_ref(size_t(M) * H), hy_rb(size_t(M) * H);
            check_hip(hipMemcpy(hy_ref.data(), dout_ref, hy_ref.size() * 4u, hipMemcpyDeviceToHost), "d2h");
            check_hip(hipMemcpy(hy_rb.data(),  dout_rb,  hy_rb.size() * 4u,  hipMemcpyDeviceToHost), "d2h");
            const double cos_v = cosine_f32(hy_ref, hy_rb);
            std::printf("[HLUT2] TP3 expert bucket (m_e=64, Ni=768) | cos=%.8f | ref=%.3f ms | rocBLAS=%.3f ms (%.2fx)\n",
                        cos_v, ref_ms, rb_ms, ref_ms / std::max(1e-6, rb_ms));
            if (cos_v < 0.9999) throw std::runtime_error("HLUT2 bucket cosine failed");
            (void)hipFree(dx); (void)hipFree(dgc); (void)hipFree(duc); (void)hipFree(ddc);
            (void)hipFree(dgs); (void)hipFree(dus); (void)hipFree(dds); (void)hipFree(dcent);
            (void)hipFree(dsigns); (void)hipFree(dpairs); (void)hipFree(dact); (void)hipFree(doff);
            (void)hipFree(dib); (void)hipFree(dir); (void)hipFree(dout_ref); (void)hipFree(dout_rb);
        }

        // 5. Routed HLUT3 Expert Bucket Parity & Speed (both TP3 Ni=768 and dev3 Ni=2304, m_e=64)
        rb_tp3.shutdown();
        RocblasPrefillEngine rb_dev3;
        rb_dev3.init(0, stream, /*is_dev3=*/true);
        std::printf("[W93_VERIFY] dev3 RocblasPrefillEngine allocated %.2f MiB HBM2\n",
                    double(rb_dev3.total_allocated_bytes()) / (1024.0 * 1024.0));
        for (int Ni : {768, 2304}) {
            const int M = 64, H = 5120;
            const size_t n_tiles = size_t(Ni / 32) * size_t(H / 32);
            std::vector<Hlut3Tile32x32> hgt(n_tiles), hut(n_tiles), hdt(n_tiles);
            auto fill_tiles = [&](std::vector<Hlut3Tile32x32>& v) {
                for (auto& t : v) {
                    for (auto& w : t.q_lo2) w = rng();
                    for (auto& w : t.q_hi1) w = rng();
                    for (auto& s : t.scales_f16) s = f32_to_f16_h(0.06f);
                }
            };
            fill_tiles(hgt); fill_tiles(hut); fill_tiles(hdt);
            const auto sol = solve_gaussian_lloyd_max_hlut3();
            float hlv[4] = {float(sol.levels[0]), float(sol.levels[1]), float(sol.levels[2]), float(sol.levels[3])};
            std::vector<uint16_t> hx(size_t(M) * H);
            for (auto& v : hx) v = f32_to_f16_h(nd(rng));
            std::vector<uint64_t> hsigns(size_t(Ni / 64), 0xA5A55A5A12345678ULL);
            std::vector<Mi50RoutedPair> hpairs(M);
            for (int i = 0; i < M; ++i) hpairs[i] = Mi50RoutedPair{i, 0, 0.3f + 0.01f * float(i % 5), 0};
            int32_t hact[1] = {0}, hoff[2] = {0, M};

            Hlut3Tile32x32 *dgt = nullptr, *dut = nullptr, *ddt = nullptr;
            float *dlv = nullptr, *dz = nullptr, *dzr = nullptr, *dout_ref = nullptr, *dout_rb = nullptr;
            uint16_t* dx = nullptr;
            uint64_t* dsigns = nullptr;
            Mi50RoutedPair* dpairs = nullptr;
            int32_t *dact = nullptr, *doff = nullptr;
            check_hip(hipMalloc(&dgt, n_tiles * sizeof(Hlut3Tile32x32)), "dgt");
            check_hip(hipMalloc(&dut, n_tiles * sizeof(Hlut3Tile32x32)), "dut");
            check_hip(hipMalloc(&ddt, n_tiles * sizeof(Hlut3Tile32x32)), "ddt");
            check_hip(hipMalloc(&dlv, 16u), "dlv");
            check_hip(hipMalloc(&dx, hx.size() * 2u), "dx");
            check_hip(hipMalloc(&dsigns, hsigns.size() * 8u), "dsigns");
            check_hip(hipMalloc(&dpairs, hpairs.size() * sizeof(Mi50RoutedPair)), "dpairs");
            check_hip(hipMalloc(&dact, 4u), "dact"); check_hip(hipMalloc(&doff, 8u), "doff");
            check_hip(hipMalloc(&dz, size_t(M) * Ni * 4u), "dz"); check_hip(hipMalloc(&dzr, size_t(M) * Ni * 4u), "dzr");
            check_hip(hipMalloc(&dout_ref, size_t(M) * H * 4u), "dout_ref"); check_hip(hipMalloc(&dout_rb, size_t(M) * H * 4u), "dout_rb");

            check_hip(hipMemcpy(dgt, hgt.data(), n_tiles * sizeof(Hlut3Tile32x32), hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dut, hut.data(), n_tiles * sizeof(Hlut3Tile32x32), hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(ddt, hdt.data(), n_tiles * sizeof(Hlut3Tile32x32), hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dlv, hlv, 16u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dx, hx.data(), hx.size() * 2u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dsigns, hsigns.data(), hsigns.size() * 8u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dpairs, hpairs.data(), hpairs.size() * sizeof(Mi50RoutedPair), hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dact, hact, 4u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(doff, hoff, 8u, hipMemcpyHostToDevice), "cp");

            check_hip(hipMemsetAsync(dout_ref, 0, size_t(M) * H * 4u, stream), "z");
            launch_hlut3_routed_grouped_f16(dx, dact, doff, dpairs, 1, M, 1, Ni, H, H, Ni, dgt, dut, ddt, dlv, dsigns, dz, dzr, dout_ref, stream);
            check_hip(hipStreamSynchronize(stream), "hlut3 ref warm");
            const auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < 5; ++it) {
                check_hip(hipMemsetAsync(dout_ref, 0, size_t(M) * H * 4u, stream), "z");
                launch_hlut3_routed_grouped_f16(dx, dact, doff, dpairs, 1, M, 1, Ni, H, H, Ni, dgt, dut, ddt, dlv, dsigns, dz, dzr, dout_ref, stream);
            }
            check_hip(hipStreamSynchronize(stream), "hlut3 ref");
            const double ref_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5.0;

            check_hip(hipMemsetAsync(dout_rb, 0, size_t(M) * H * 4u, stream), "z");
            rb_dev3.run_hlut3_expert_bucket(dx, dpairs, 0, 0, M, Ni, dgt, dut, ddt, dlv, dsigns, dout_rb, stream);
            check_hip(hipStreamSynchronize(stream), "hlut3 rb warm");
            const auto t1 = std::chrono::steady_clock::now();
            for (int it = 0; it < 5; ++it) {
                check_hip(hipMemsetAsync(dout_rb, 0, size_t(M) * H * 4u, stream), "z");
                rb_dev3.run_hlut3_expert_bucket(dx, dpairs, 0, 0, M, Ni, dgt, dut, ddt, dlv, dsigns, dout_rb, stream);
            }
            check_hip(hipStreamSynchronize(stream), "hlut3 rb");
            const double rb_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / 5.0;

            std::vector<float> hy_ref(size_t(M) * H), hy_rb(size_t(M) * H);
            check_hip(hipMemcpy(hy_ref.data(), dout_ref, hy_ref.size() * 4u, hipMemcpyDeviceToHost), "d2h");
            check_hip(hipMemcpy(hy_rb.data(),  dout_rb,  hy_rb.size() * 4u,  hipMemcpyDeviceToHost), "d2h");
            const double cos_v = cosine_f32(hy_ref, hy_rb);
            std::printf("[HLUT3] expert bucket (m_e=64, Ni=%4d) | cos=%.8f | ref=%.3f ms | rocBLAS=%.3f ms (%.2fx)\n",
                        Ni, cos_v, ref_ms, rb_ms, ref_ms / std::max(1e-6, rb_ms));
            if (cos_v < 0.9999) throw std::runtime_error("HLUT3 bucket cosine failed");

            // 6. Phase D Decode: Zero-Scratch Grouped v_dot2_f32_f16 HLUT3 (m_e=8) vs Scalar Unsharded/TP3 Pairs
            {
                const int Md = 8;
                int32_t hoff_d[2] = {0, Md};
                check_hip(hipMemcpy(doff, hoff_d, 8u, hipMemcpyHostToDevice), "cp doff d");
                uint16_t* dir16 = nullptr;
                check_hip(hipMalloc(&dir16, size_t(Md) * Ni * 2u), "dir16");

                check_hip(hipMemsetAsync(dout_ref, 0, size_t(Md) * H * 4u, stream), "z");
                if (Ni == 2304) {
                    launch_hlut3_unsharded_pairs_f16(dx, dpairs, Md, 512, 1, dgt, dut, ddt, dlv, dsigns, dz, dzr, dout_ref, stream);
                } else {
                    launch_hlut3_tp3_pairs_f16(dx, dpairs, Md, 512, 1, dgt, dut, ddt, dlv, dsigns, dz, dzr, dout_ref, stream);
                }
                check_hip(hipStreamSynchronize(stream), "fdot2 ref warm");
                const auto td0 = std::chrono::steady_clock::now();
                for (int it = 0; it < 10; ++it) {
                    check_hip(hipMemsetAsync(dout_ref, 0, size_t(Md) * H * 4u, stream), "z");
                    if (Ni == 2304) {
                        launch_hlut3_unsharded_pairs_f16(dx, dpairs, Md, 512, 1, dgt, dut, ddt, dlv, dsigns, dz, dzr, dout_ref, stream);
                    } else {
                        launch_hlut3_tp3_pairs_f16(dx, dpairs, Md, 512, 1, dgt, dut, ddt, dlv, dsigns, dz, dzr, dout_ref, stream);
                    }
                }
                check_hip(hipStreamSynchronize(stream), "fdot2 ref");
                const double ref_d_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count() / 10.0;

                check_hip(hipMemsetAsync(dout_rb, 0, size_t(Md) * H * 4u, stream), "z");
                launch_hlut3_fdot2_grouped_f16(dx, dact, doff, dpairs, nullptr, 1, 1, Ni, dgt, dut, ddt, dlv, dsigns, dir16, dout_rb, stream);
                check_hip(hipStreamSynchronize(stream), "fdot2 warm");
                const auto td1 = std::chrono::steady_clock::now();
                for (int it = 0; it < 10; ++it) {
                    check_hip(hipMemsetAsync(dout_rb, 0, size_t(Md) * H * 4u, stream), "z");
                    launch_hlut3_fdot2_grouped_f16(dx, dact, doff, dpairs, nullptr, 1, 1, Ni, dgt, dut, ddt, dlv, dsigns, dir16, dout_rb, stream);
                }
                check_hip(hipStreamSynchronize(stream), "fdot2");
                const double fd_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td1).count() / 10.0;

                std::vector<float> hd_ref(size_t(Md) * H), hd_fd(size_t(Md) * H);
                check_hip(hipMemcpy(hd_ref.data(), dout_ref, hd_ref.size() * 4u, hipMemcpyDeviceToHost), "d2h");
                check_hip(hipMemcpy(hd_fd.data(),  dout_rb,  hd_fd.size() * 4u,  hipMemcpyDeviceToHost), "d2h");
                const double cos_d = cosine_f32(hd_ref, hd_fd);
                std::printf("[HLUT3-FDOT2] decode grouped (m_e=8, Ni=%4d) | cos=%.8f | ref_pairs=%.3f ms | fdot2_grp=%.3f ms (%.2fx)\n",
                            Ni, cos_d, ref_d_ms, fd_ms, ref_d_ms / std::max(1e-6, fd_ms));
                if (cos_d < 0.9999) throw std::runtime_error("HLUT3 fdot2 decode grouped cosine failed");

                // 7. Wave 9.4 Phase C: INT8+FP16-scale wire quant->dequant->HLUT3->BF16 end-to-end parity (must be >= 0.99995!)
                if (Ni == 2304) {
                    int8_t* d_i8 = nullptr;
                    uint16_t *d_sc16 = nullptr, *d_x_dq = nullptr, *d_out_bf16 = nullptr;
                    check_hip(hipMalloc(&d_i8, size_t(M) * H), "d_i8");
                    check_hip(hipMalloc(&d_sc16, size_t(M) * size_t(H / 128) * 2u), "d_sc16");
                    check_hip(hipMalloc(&d_x_dq, size_t(M) * H * 2u), "d_x_dq");
                    check_hip(hipMalloc(&d_out_bf16, size_t(M) * H * 2u), "d_out_bf16");

                    launch_w94_quant_f16_to_int8_rowwise(dx, d_i8, d_sc16, M, H, stream);
                    launch_w94_dequant_int8_to_f16_rowwise(d_i8, d_sc16, d_x_dq, M, H, stream);
                    check_hip(hipMemsetAsync(dout_rb, 0, size_t(M) * H * 4u, stream), "z i8");
                    rb_dev3.run_hlut3_expert_bucket(d_x_dq, dpairs, 0, 0, M, Ni, dgt, dut, ddt, dlv, dsigns, dout_rb, stream);
                    launch_w93_f32_to_bf16_rne(dout_rb, d_out_bf16, size_t(M) * H, stream);
                    check_hip(hipMemsetAsync(dout_rb, 0, size_t(M) * H * 4u, stream), "z acc");
                    launch_w94_add_bf16_to_f32_inplace(dout_rb, d_out_bf16, size_t(M) * H, stream);
                    check_hip(hipStreamSynchronize(stream), "sync i8");

                    std::vector<float> h_i8(size_t(M) * H), h_ref64(size_t(M) * H);
                    check_hip(hipMemcpy(h_i8.data(), dout_rb, h_i8.size() * 4u, hipMemcpyDeviceToHost), "d2h i8");
                    check_hip(hipMemsetAsync(dout_ref, 0, size_t(M) * H * 4u, stream), "z ref64");
                    rb_dev3.run_hlut3_expert_bucket(dx, dpairs, 0, 0, M, Ni, dgt, dut, ddt, dlv, dsigns, dout_ref, stream);
                    check_hip(hipStreamSynchronize(stream), "sync ref64");
                    check_hip(hipMemcpy(h_ref64.data(), dout_ref, h_ref64.size() * 4u, hipMemcpyDeviceToHost), "d2h ref64");
                    const double cos_i8 = cosine_f32(h_ref64, h_i8);
                    std::printf("[W94-INT8-BLK128-WIRE] dev3 HLUT3 (M=64, Ni=2304) | cos=%.8f (gate >= 0.99995)\n", cos_i8);
                    if (cos_i8 < 0.99995) throw std::runtime_error("Wave 9.4 INT8 wire cosine below 0.99995!");
                    (void)hipFree(d_i8); (void)hipFree(d_sc16); (void)hipFree(d_x_dq); (void)hipFree(d_out_bf16);
                }

                // 8. Wave 9.4 Phase C: m_e = {8, 12, 16, 24, 32, 48} crossover sweep (fdot2 grouped vs rocBLAS)
                const int sweep_ms[6] = {8, 12, 16, 24, 32, 48};
                for (int sm : sweep_ms) {
                    int32_t ho[2] = {0, sm};
                    check_hip(hipMemcpy(doff, ho, 8u, hipMemcpyHostToDevice), "cp sweep doff");
                    uint16_t* dir_s = nullptr;
                    check_hip(hipMalloc(&dir_s, size_t(sm) * Ni * 2u), "dir_s");
                    launch_hlut3_fdot2_grouped_f16(dx, dact, doff, dpairs, nullptr, 1, 1, Ni, dgt, dut, ddt, dlv, dsigns, dir_s, dout_ref, stream);
                    check_hip(hipStreamSynchronize(stream), "sw fd warm");
                    const auto ts0 = std::chrono::steady_clock::now();
                    for (int it = 0; it < 5; ++it) {
                        launch_hlut3_fdot2_grouped_f16(dx, dact, doff, dpairs, nullptr, 1, 1, Ni, dgt, dut, ddt, dlv, dsigns, dir_s, dout_ref, stream);
                    }
                    check_hip(hipStreamSynchronize(stream), "sw fd");
                    const double ms_fd = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts0).count() / 5.0;

                    rb_dev3.run_hlut3_expert_bucket(dx, dpairs, 0, 0, sm, Ni, dgt, dut, ddt, dlv, dsigns, dout_rb, stream);
                    check_hip(hipStreamSynchronize(stream), "sw rb warm");
                    const auto ts1 = std::chrono::steady_clock::now();
                    for (int it = 0; it < 5; ++it) {
                        rb_dev3.run_hlut3_expert_bucket(dx, dpairs, 0, 0, sm, Ni, dgt, dut, ddt, dlv, dsigns, dout_rb, stream);
                    }
                    check_hip(hipStreamSynchronize(stream), "sw rb");
                    const double ms_rb = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts1).count() / 5.0;
                    std::printf("[W94-SWEEP HLUT3 Ni=%4d m_e=%2d] fdot2=%.3f ms | rocBLAS=%.3f ms (%.2fx)\n",
                                Ni, sm, ms_fd, ms_rb, ms_fd / std::max(1e-6, ms_rb));
                    (void)hipFree(dir_s);
                }

                (void)hipFree(dir16);
            }

            (void)hipFree(dgt); (void)hipFree(dut); (void)hipFree(ddt); (void)hipFree(dlv);
            (void)hipFree(dx); (void)hipFree(dsigns); (void)hipFree(dpairs); (void)hipFree(dact);
            (void)hipFree(doff); (void)hipFree(dz); (void)hipFree(dzr); (void)hipFree(dout_ref); (void)hipFree(dout_rb);
        }

        rb_dev3.shutdown();

        // 9. Wave 10 Phase C.1: Large-M (M=2048) Dense FP8 HHHS (wqkv, wq_b, wo_a, wo_b, sh_w13, sh_w2) parity & speed
        {
            rb_tp3.init(0, stream, /*is_dev3=*/false);
            struct Fp8Large { const char* name; int M; int N; int K; bool write_f32; };
            const Fp8Large cases[] = {
                {"wqkv   [2048x1792x5120]",  2048,  1792, 5120, false},
                {"wq_b   [2048x12288x1280]", 2048, 12288, 1280, false},
                {"wo_b   [2048x5120x3072]",  2048,  5120, 3072, false},
                {"sh_w13 [2048x1536x5120]",  2048,  1536, 5120, false},
                {"sh_w2  [2048x5120x768]",   2048,  5120,  768, true},
            };
            for (const auto& tc : cases) {
                std::vector<uint8_t> hx(size_t(tc.M) * size_t(tc.K));
                std::vector<uint8_t> hxs(size_t(tc.M) * size_t(tc.K / 32), 124u);
                std::vector<uint8_t> hw(size_t(tc.N) * size_t(tc.K));
                std::vector<uint8_t> hws(size_t(tc.N / 32) * size_t(tc.K / 32), 123u);
                for (auto& b : hx) b = uint8_t((rng() % 112u) | ((rng() & 1u) << 7));
                for (auto& b : hw) b = uint8_t((rng() % 112u) | ((rng() & 1u) << 7));
                uint8_t *dx = nullptr, *dxs = nullptr, *dw = nullptr, *dws = nullptr;
                check_hip(hipMalloc(&dx, hx.size()), "dx2k");
                check_hip(hipMalloc(&dxs, hxs.size()), "dxs2k");
                check_hip(hipMalloc(&dw, hw.size()), "dw2k");
                check_hip(hipMalloc(&dws, hws.size()), "dws2k");
                check_hip(hipMemcpy(dx, hx.data(), hx.size(), hipMemcpyHostToDevice), "cp dx2k");
                check_hip(hipMemcpy(dxs, hxs.data(), hxs.size(), hipMemcpyHostToDevice), "cp dxs2k");
                check_hip(hipMemcpy(dw, hw.data(), hw.size(), hipMemcpyHostToDevice), "cp dw2k");
                check_hip(hipMemcpy(dws, hws.data(), hws.size(), hipMemcpyHostToDevice), "cp dws2k");
                const size_t out_n = size_t(tc.M) * size_t(tc.N);
                uint16_t *dy16_ref = nullptr, *dy16_rb = nullptr;
                float *dy32_ref = nullptr, *dy32_rb = nullptr;
                if (tc.write_f32) {
                    check_hip(hipMalloc(&dy32_ref, out_n * 4u), "dy32_ref");
                    check_hip(hipMalloc(&dy32_rb,  out_n * 4u), "dy32_rb");
                } else {
                    check_hip(hipMalloc(&dy16_ref, out_n * 2u), "dy16_ref");
                    check_hip(hipMalloc(&dy16_rb,  out_n * 2u), "dy16_rb");
                }
                launch_fp8_gemm(dx, dxs, dw, dws, dy16_ref, dy32_ref, tc.M, tc.N, tc.K, tc.write_f32, false, stream);
                check_hip(hipStreamSynchronize(stream), "w10 fp8 ref warm");
                const auto t0 = std::chrono::steady_clock::now();
                for (int it = 0; it < 3; ++it) {
                    launch_fp8_gemm(dx, dxs, dw, dws, dy16_ref, dy32_ref, tc.M, tc.N, tc.K, tc.write_f32, false, stream);
                }
                check_hip(hipStreamSynchronize(stream), "w10 fp8 ref");
                const double ref_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 3.0;

                if (!rb_tp3.run_fp8_gemm(dx, dxs, dw, dws, dy16_rb, dy32_rb, tc.M, tc.N, tc.K, tc.write_f32, false, stream)) {
                    throw std::runtime_error(std::string("W10 run_fp8_gemm false on ") + tc.name);
                }
                check_hip(hipStreamSynchronize(stream), "w10 fp8 rb warm");
                const auto t1 = std::chrono::steady_clock::now();
                for (int it = 0; it < 3; ++it) {
                    (void)rb_tp3.run_fp8_gemm(dx, dxs, dw, dws, dy16_rb, dy32_rb, tc.M, tc.N, tc.K, tc.write_f32, false, stream);
                }
                check_hip(hipStreamSynchronize(stream), "w10 fp8 rb");
                const double rb_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / 3.0;

                double cos_v = 0.0;
                if (tc.write_f32) {
                    std::vector<float> a(out_n), b(out_n);
                    check_hip(hipMemcpy(a.data(), dy32_ref, out_n * 4u, hipMemcpyDeviceToHost), "d2h32");
                    check_hip(hipMemcpy(b.data(), dy32_rb,  out_n * 4u, hipMemcpyDeviceToHost), "d2h32");
                    cos_v = cosine_f32(a, b);
                } else {
                    std::vector<uint16_t> a(out_n), b(out_n);
                    check_hip(hipMemcpy(a.data(), dy16_ref, out_n * 2u, hipMemcpyDeviceToHost), "d2h16");
                    check_hip(hipMemcpy(b.data(), dy16_rb,  out_n * 2u, hipMemcpyDeviceToHost), "d2h16");
                    cos_v = cosine_bf16(a, b);
                }
                std::printf("[W10-FP8-2K] %-25s | cos=%.8f | custom=%.3f ms | HHHS=%.3f ms (%.2fx)\n",
                            tc.name, cos_v, ref_ms, rb_ms, ref_ms / std::max(1e-6, rb_ms));
                if (cos_v < 0.9999) throw std::runtime_error(std::string("W10 FP8 2K cosine failed on ") + tc.name);
                (void)hipFree(dx); (void)hipFree(dxs); (void)hipFree(dw); (void)hipFree(dws);
                if (dy16_ref) (void)hipFree(dy16_ref);
                if (dy16_rb)  (void)hipFree(dy16_rb);
                if (dy32_ref) (void)hipFree(dy32_ref);
                if (dy32_rb)  (void)hipFree(dy32_rb);
            }
            // Grouped wo_a M=2048
            const int M = 2048, g_cnt = 3, K_tot = g_cnt * 4096, N_tot = g_cnt * 1024;
            std::vector<uint8_t> ho(size_t(M) * size_t(K_tot)), hos(size_t(M) * size_t(K_tot / 32), 124u);
            std::vector<uint8_t> hw(size_t(N_tot) * 4096u), hws(size_t(N_tot / 32) * 128u, 123u);
            for (auto& b : ho) b = uint8_t((rng() % 112u) | ((rng() & 1u) << 7));
            for (auto& b : hw) b = uint8_t((rng() % 112u) | ((rng() & 1u) << 7));
            uint8_t *do_p = nullptr, *dos = nullptr, *dw = nullptr, *dws = nullptr;
            uint16_t *dy_ref = nullptr, *dy_rb = nullptr;
            check_hip(hipMalloc(&do_p, ho.size()), "do2k");
            check_hip(hipMalloc(&dos, hos.size()), "dos2k");
            check_hip(hipMalloc(&dw, hw.size()), "dw2k");
            check_hip(hipMalloc(&dws, hws.size()), "dws2k");
            check_hip(hipMalloc(&dy_ref, size_t(M) * size_t(N_tot) * 2u), "dyr2k");
            check_hip(hipMalloc(&dy_rb,  size_t(M) * size_t(N_tot) * 2u), "dyb2k");
            check_hip(hipMemcpy(do_p, ho.data(), ho.size(), hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dos, hos.data(), hos.size(), hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dw, hw.data(), hw.size(), hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dws, hws.data(), hws.size(), hipMemcpyHostToDevice), "cp");
            launch_fp8_grouped_wo_a(do_p, dos, dw, dws, dy_ref, M, g_cnt, stream);
            check_hip(hipStreamSynchronize(stream), "w10 woa ref warm");
            const auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < 3; ++it) launch_fp8_grouped_wo_a(do_p, dos, dw, dws, dy_ref, M, g_cnt, stream);
            check_hip(hipStreamSynchronize(stream), "w10 woa ref");
            const double ref_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 3.0;
            if (!rb_tp3.run_fp8_grouped_wo_a(do_p, dos, dw, dws, dy_rb, M, g_cnt, stream)) {
                throw std::runtime_error("W10 run_fp8_grouped_wo_a M=2048 false");
            }
            check_hip(hipStreamSynchronize(stream), "w10 woa rb warm");
            const auto t1 = std::chrono::steady_clock::now();
            for (int it = 0; it < 3; ++it) (void)rb_tp3.run_fp8_grouped_wo_a(do_p, dos, dw, dws, dy_rb, M, g_cnt, stream);
            check_hip(hipStreamSynchronize(stream), "w10 woa rb");
            const double rb_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / 3.0;
            std::vector<uint16_t> a(size_t(M) * size_t(N_tot)), b(size_t(M) * size_t(N_tot));
            check_hip(hipMemcpy(a.data(), dy_ref, a.size() * 2u, hipMemcpyDeviceToHost), "d2h");
            check_hip(hipMemcpy(b.data(), dy_rb,  b.size() * 2u, hipMemcpyDeviceToHost), "d2h");
            const double cos_v = cosine_bf16(a, b);
            std::printf("[W10-WOA-2K] grouped_wo_a [2048x3x1024x4096] | cos=%.8f | custom=%.3f ms | HHHS=%.3f ms (%.2fx)\n",
                        cos_v, ref_ms, rb_ms, ref_ms / std::max(1e-6, rb_ms));
            if (cos_v < 0.9999) throw std::runtime_error("W10 wo_a 2K cosine failed");
            (void)hipFree(do_p); (void)hipFree(dos); (void)hipFree(dw); (void)hipFree(dws);
            (void)hipFree(dy_ref); (void)hipFree(dy_rb);
            rb_tp3.shutdown();
        }

        // 10. Wave 10 Phase C.2: Tiled Multi-Query SWA+CSA2 (k_sparse_attn_multiq4) parity vs k_sparse_attn_allheads
        {
            const int M = 256, h_cnt = 24, start_pos = 128, n_comp = 128, T_comp = 256;
            std::vector<uint16_t> hq(size_t(M) * h_cnt * 512u), hwin(128u * 512u), hcur(size_t(M) * 512u);
            std::vector<float> hsink(h_cnt, -2.5f);
            std::vector<uint8_t> hcfp4(size_t(T_comp) * 256u), hce4(size_t(T_comp) * 32u, 0x38u);
            std::vector<int32_t> hcidx(size_t(M) * n_comp);
            for (auto& v : hq)   v = f32_to_bf16_h(nd(rng));
            for (auto& v : hwin) v = f32_to_bf16_h(nd(rng));
            for (auto& v : hcur) v = f32_to_bf16_h(nd(rng));
            for (auto& v : hcfp4) v = uint8_t(rng() & 0xFFu);
            for (int r = 0; r < M; ++r) {
                for (int k = 0; k < n_comp; ++k) hcidx[size_t(r) * n_comp + k] = (r + k) % T_comp;
            }
            uint16_t *dq = nullptr, *dwin = nullptr, *dcur = nullptr, *do_mq = nullptr, *do_ref = nullptr;
            float* dsink = nullptr;
            uint8_t *dcfp4 = nullptr, *dce4 = nullptr;
            int32_t* dcidx = nullptr;
            check_hip(hipMalloc(&dq, hq.size() * 2u), "dq");
            check_hip(hipMalloc(&dwin, hwin.size() * 2u), "dwin");
            check_hip(hipMalloc(&dcur, hcur.size() * 2u), "dcur");
            check_hip(hipMalloc(&dsink, hsink.size() * 4u), "dsink");
            check_hip(hipMalloc(&dcfp4, hcfp4.size()), "dcfp4");
            check_hip(hipMalloc(&dce4, hce4.size()), "dce4");
            check_hip(hipMalloc(&dcidx, hcidx.size() * 4u), "dcidx");
            check_hip(hipMalloc(&do_mq, hq.size() * 2u), "do_mq");
            check_hip(hipMalloc(&do_ref, hq.size() * 2u), "do_ref");
            check_hip(hipMemcpy(dq, hq.data(), hq.size() * 2u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dwin, hwin.data(), hwin.size() * 2u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dcur, hcur.data(), hcur.size() * 2u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dsink, hsink.data(), hsink.size() * 4u, hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dcfp4, hcfp4.data(), hcfp4.size(), hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dce4, hce4.data(), hce4.size(), hipMemcpyHostToDevice), "cp");
            check_hip(hipMemcpy(dcidx, hcidx.data(), hcidx.size() * 4u, hipMemcpyHostToDevice), "cp");

            // Reference: 4 slices of M=64 (exercises k_sparse_attn_allheads!) with rolling ring update
            std::vector<uint16_t> hwin_roll = hwin;
            for (int m0 = 0; m0 < M; m0 += 64) {
                check_hip(hipMemcpy(dwin, hwin_roll.data(), hwin_roll.size() * 2u, hipMemcpyHostToDevice), "cp roll");
                launch_sparse_attn(
                    dq + size_t(m0) * h_cnt * 512u, dsink, dwin, dcur + size_t(m0) * 512u,
                    start_pos + m0, 0, false, dcfp4, dce4, dcidx + size_t(m0) * n_comp, n_comp,
                    nullptr, do_ref + size_t(m0) * h_cnt * 512u, 64, h_cnt, stream);
                check_hip(hipStreamSynchronize(stream), "sync ref slice");
                for (int r = 0; r < 64; ++r) {
                    const int slot = (start_pos + m0 + r) & 127;
                    std::memcpy(&hwin_roll[size_t(slot) * 512u], &hcur[size_t(m0 + r) * 512u], 512u * 2u);
                }
            }
            // Restore original dwin and run M=256 k_sparse_attn_multiq4 in one launch!
            check_hip(hipMemcpy(dwin, hwin.data(), hwin.size() * 2u, hipMemcpyHostToDevice), "cp orig");
            launch_sparse_attn(dq, dsink, dwin, dcur, start_pos, 0, false, dcfp4, dce4, dcidx, n_comp,
                               nullptr, do_mq, M, h_cnt, stream);
            check_hip(hipStreamSynchronize(stream), "sync multiq4");
            std::vector<uint16_t> ho_ref(hq.size()), ho_mq(hq.size());
            check_hip(hipMemcpy(ho_ref.data(), do_ref, hq.size() * 2u, hipMemcpyDeviceToHost), "d2h");
            check_hip(hipMemcpy(ho_mq.data(),  do_mq,  hq.size() * 2u, hipMemcpyDeviceToHost), "d2h");
            const double cos_mq = cosine_bf16(ho_ref, ho_mq);
            std::printf("[W10-MULTIQ4-ATTN] M=%d h=%d SWA128+CSA2(%d) | cos=%.9f (gate >= 0.999999)\n",
                        M, h_cnt, n_comp, cos_mq);
            if (cos_mq < 0.999999) throw std::runtime_error("W10 k_sparse_attn_multiq4 parity failed!");
            (void)hipFree(dq); (void)hipFree(dwin); (void)hipFree(dcur); (void)hipFree(dsink);
            (void)hipFree(dcfp4); (void)hipFree(dce4); (void)hipFree(dcidx); (void)hipFree(do_mq); (void)hipFree(do_ref);
        }

        // 11. Wave 10 Phase C.3: 3-GPU 128-bit Vectorized 4-Stage Pipelined BF16 HostRingAllReduce Bit-Exact Parity
        {
            int ndev = 0;
            check_hip(hipGetDeviceCount(&ndev), "hipGetDeviceCount");
            if (ndev >= 3) {
                const size_t n_elems = 1024u * 5120u; // 10.0 MiB BF16/rank!
                void* slot_raw = nullptr;
                check_hip(hipSetDevice(0), "dev0");
                check_hip(hipHostMalloc(&slot_raw, sizeof(HostRingAllReduceSlot),
                                        hipHostMallocMapped | hipHostMallocCoherent), "ar slot");
                std::memset(slot_raw, 0, sizeof(HostRingAllReduceSlot));
                auto* slot = static_cast<HostRingAllReduceSlot*>(slot_raw);

                std::vector<uint16_t> hsrc[3], hdst(n_elems);
                uint16_t *dsrc[3]{}, *ddst[3]{};
                hipStream_t st[3]{};
                for (int tp = 0; tp < 3; ++tp) {
                    hsrc[tp].resize(n_elems);
                    for (size_t i = 0; i < n_elems; ++i) hsrc[tp][i] = f32_to_bf16_h(nd(rng));
                    check_hip(hipSetDevice(tp), "setdev");
                    check_hip(hipStreamCreate(&st[tp]), "mkst");
                    check_hip(hipMalloc(&dsrc[tp], n_elems * 2u), "dsrc");
                    check_hip(hipMalloc(&ddst[tp], n_elems * 2u), "ddst");
                    check_hip(hipMemcpy(dsrc[tp], hsrc[tp].data(), n_elems * 2u, hipMemcpyHostToDevice), "cp");
                }
                for (int tp = 0; tp < 3; ++tp) {
                    check_hip(hipSetDevice(tp), "sd");
                    launch_host_ring_allreduce_bf16(dsrc[tp], ddst[tp], slot, tp, 1ULL, n_elems, st[tp]);
                }
                for (int tp = 0; tp < 3; ++tp) {
                    check_hip(hipSetDevice(tp), "sd");
                    check_hip(hipStreamSynchronize(st[tp]), "sync ar warm");
                }
                const auto ta0 = std::chrono::steady_clock::now();
                for (int it = 0; it < 5; ++it) {
                    const uint64_t seq = 2ULL + uint64_t(it);
                    for (int tp = 0; tp < 3; ++tp) {
                        check_hip(hipSetDevice(tp), "sd");
                        launch_host_ring_allreduce_bf16(dsrc[tp], ddst[tp], slot, tp, seq, n_elems, st[tp]);
                    }
                    for (int tp = 0; tp < 3; ++tp) {
                        check_hip(hipSetDevice(tp), "sd");
                        check_hip(hipStreamSynchronize(st[tp]), "sync ar");
                    }
                }
                const double ar_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta0).count() / 5.0;

                size_t bit_mismatches = 0;
                for (int tp = 0; tp < 3; ++tp) {
                    check_hip(hipSetDevice(tp), "sd");
                    check_hip(hipMemcpy(hdst.data(), ddst[tp], n_elems * 2u, hipMemcpyDeviceToHost), "d2h");
                    for (size_t i = 0; i < n_elems; ++i) {
                        const float s = (bf16_to_f32_h(hsrc[0][i]) + bf16_to_f32_h(hsrc[1][i])) + bf16_to_f32_h(hsrc[2][i]);
                        if (hdst[i] != f32_to_bf16_h(s)) ++bit_mismatches;
                    }
                }
                const double bw_gbs = (3.0 * 10.0 * 3.0) / (ar_ms * 1.024); // 10 MiB TX + 20 MiB RX per rank across 3 ranks
                std::printf("[W10-AR-PIPE] 3x MI50 BF16 vec4 4-stage pipe (10 MiB/rank) | mismatches=%zu | %.3f ms (agg=%.2f GB/s)\n",
                            bit_mismatches, ar_ms, bw_gbs);
                if (bit_mismatches != 0u) throw std::runtime_error("W10 BF16 vec4_pipe HostRingAllReduce bit mismatch!");
                for (int tp = 0; tp < 3; ++tp) {
                    check_hip(hipSetDevice(tp), "sd");
                    (void)hipFree(dsrc[tp]); (void)hipFree(ddst[tp]); (void)hipStreamDestroy(st[tp]);
                }
                (void)hipHostFree(slot_raw);
                check_hip(hipSetDevice(0), "restore dev0");
            }
        }

        check_hip(hipStreamDestroy(stream), "hipStreamDestroy");
        std::printf("PASS: verify_wave9_3_rocblas_prefill (all cosine >= 0.9999 + Wave 10 Phase C parity)\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
