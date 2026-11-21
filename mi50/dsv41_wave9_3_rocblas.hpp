#pragma once
// Wave 9.3 Phase B: Fixed Pre-Allocated HBM2 Workspace & rocBLAS FP16->FP32 (v_dot2_f32_f16)
// Prefill GEMM Dispatcher for AMD Instinct MI50 (gfx906 Vega 20).
//   - Zero dynamic hipMalloc at runtime!
//   - TP3 rank footprint (tp=0,1,2): 83.5 MiB HBM2 (well below 192 MiB cap, safe at 1M + ViT!)
//   - dev3 footprint (0000:85:00.0) : 95.5 MiB HBM2 (keeps 85:00.0 <= 15,000 MiB used!)
//   - Handles 100% of dense FP8 GEMMs (M >= 64, including wq_b [12288,1280] & wo_b [5120,3072]),
//     grouped wo_a FP8 GEMMs, dense BF16 GEMMs, HLUT2 routed experts (m_e >= K_thresh),
//     and HLUT3 routed experts (both TP3 [768,5120] and unsharded dev3 [2304,5120]).
// 100% Pure C++17 / HIP / rocBLAS — Zero Python Anywhere.

#include "dsv41_wave9_3_prefill_kernels.hpp"
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace dsv41 {

class RocblasPrefillEngine {
public:
    static constexpr int kHidden = 5120;
    static constexpr int kDefaultDenseThreshM  = 64;
    static constexpr int kDefaultExpertThreshM = 16;

    RocblasPrefillEngine() = default;
    ~RocblasPrefillEngine() { shutdown(); }

    RocblasPrefillEngine(const RocblasPrefillEngine&) = delete;
    RocblasPrefillEngine& operator=(const RocblasPrefillEngine&) = delete;

    struct ExpertBucketDesc {
        int eid;
        int p_start;
        int m_e;
    };

    // is_dev3 == true  : unsharded N_inter = 2304 (offload_max_b=1..4 -> 45..180 MiB w_f16)
    // is_dev3 == false : TP3 rank N_inter  =  768 (B=8 strided-batched HHHS -> 120 MiB w_f16)
    void init(int dev_id, hipStream_t stream, bool is_dev3, int max_bucket_m = 0, int offload_max_b = 1) {
        shutdown();
        dev_id_ = dev_id;
        is_dev3_ = is_dev3;
        offload_max_b_ = is_dev3 ? std::clamp(offload_max_b, 1, 4) : 8;
        max_bucket_m_ = (max_bucket_m > 0) ? max_bucket_m : 1024;
        n_inter_ = is_dev3 ? 2304 : 768;

        check_hip(hipSetDevice(dev_id_), "RocblasPrefillEngine hipSetDevice");
        check_rb(rocblas_create_handle(&handle_), "rocblas_create_handle");
        check_rb(rocblas_set_stream(handle_, stream), "rocblas_set_stream");
        check_rb(rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_host), "rocblas_set_pointer_mode");

        // 16 MiB rocBLAS device workspace on both TP3 and dev3/dev4 (enables Tensile split-K!)
        ws_bytes_ = 16ULL << 20;
        check_hip(hipMalloc(&d_rb_ws_, ws_bytes_), "hipMalloc(d_rb_ws_)");
        check_rb(rocblas_set_workspace(handle_, d_rb_ws_, ws_bytes_), "rocblas_set_workspace");

        // Largest unpacked FP16 weight buffer:
        //   dev3/dev4 : offload_max_b_ x [4608, 5120] = 45..180 MiB (enables B=3/4 HHHS strided-batched on Switch#1/2!)
        //   TP3       : 8x [1536, 5120] = 120.0 MiB (enables B=8 HHHS strided-batched rocBLAS!)
        w_f16_elems_ = is_dev3 ? (size_t(offload_max_b_) * 4608ULL * 5120ULL) : (8ULL * 1536ULL * 5120ULL);
        x_f16_elems_ = is_dev3
            ? (size_t(max_bucket_m_) * size_t(kHidden))
            : std::max<size_t>(2048ULL * size_t(kHidden), size_t(max_bucket_m_) * size_t(kHidden));
        y_f32_elems_ = is_dev3
            ? (size_t(max_bucket_m_) * size_t(kHidden))
            : std::max<size_t>(512ULL * 12288ULL, size_t(max_bucket_m_) * size_t(kHidden));
        zrot_f16_elems_ = is_dev3
            ? (size_t(max_bucket_m_) * size_t(n_inter_))
            : std::max<size_t>(2048ULL * size_t(n_inter_), size_t(max_bucket_m_) * size_t(n_inter_));

        check_hip(hipMalloc(&d_w_f16_, w_f16_elems_ * sizeof(uint16_t)), "hipMalloc(d_w_f16_)");
        check_hip(hipMalloc(&d_x_f16_, x_f16_elems_ * sizeof(uint16_t)), "hipMalloc(d_x_f16_)");
        check_hip(hipMalloc(&d_y_f32_, y_f32_elems_ * sizeof(float)),    "hipMalloc(d_y_f32_)");
        check_hip(hipMalloc(&d_zrot_f16_, zrot_f16_elems_ * sizeof(uint16_t)), "hipMalloc(d_zrot_f16_)");
        check_hip(hipMalloc(&d_batch_eids_, 4u * 768u * sizeof(int32_t)), "hipMalloc(d_batch_eids_)");
        check_hip(hipHostMalloc(reinterpret_cast<void**>(&h_batch_eids_), 4u * 768u * sizeof(int32_t), hipHostMallocDefault),
                  "hipHostMalloc(h_batch_eids_)");

        // Pre-warm ONLY the exact HHHS (FP16-out, FP32-compute) & HHSS Tensile shapes needed!
        {
            const float alpha = 1.0f, beta = 0.0f;
            uint16_t* d_y_f16 = reinterpret_cast<uint16_t*>(d_y_f32_);
            auto warm_hhhs = [&](int n_out, int m_rows, int k_in) {
                (void)rocblas_gemm_ex(
                    handle_, rocblas_operation_transpose, rocblas_operation_none,
                    n_out, m_rows, k_in, &alpha,
                    d_w_f16_, rocblas_datatype_f16_r, k_in,
                    d_x_f16_, rocblas_datatype_f16_r, k_in,
                    &beta,
                    d_y_f16, rocblas_datatype_f16_r, n_out,
                    d_y_f16, rocblas_datatype_f16_r, n_out,
                    rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
            };
            auto warm_hhss = [&](int n_out, int m_rows, int k_in) {
                (void)rocblas_gemm_ex(
                    handle_, rocblas_operation_transpose, rocblas_operation_none,
                    n_out, m_rows, k_in, &alpha,
                    d_w_f16_, rocblas_datatype_f16_r, k_in,
                    d_x_f16_, rocblas_datatype_f16_r, k_in,
                    &beta,
                    d_y_f32_, rocblas_datatype_f32_r, n_out,
                    d_y_f32_, rocblas_datatype_f32_r, n_out,
                    rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
            };
            const int warm_ms[5] = {12, 24, 64, 128, 384};
            for (int wm : warm_ms) {
                if (is_dev3_) {
                    warm_hhhs(4608, wm, 5120);
                    warm_hhhs(5120, wm, 2304);
                    if (offload_max_b_ >= 2 && (wm == 64 || wm == 128)) {
                        for (int bb = 2; bb <= offload_max_b_; ++bb) {
                            (void)rocblas_gemm_strided_batched_ex(
                                handle_, rocblas_operation_transpose, rocblas_operation_none,
                                4608, wm, 5120, &alpha,
                                d_w_f16_, rocblas_datatype_f16_r, 5120, rocblas_stride(4608) * 5120,
                                d_x_f16_, rocblas_datatype_f16_r, 5120, rocblas_stride(wm) * 5120,
                                &beta,
                                d_y_f16, rocblas_datatype_f16_r, 4608, rocblas_stride(wm) * 4608,
                                d_y_f16, rocblas_datatype_f16_r, 4608, rocblas_stride(wm) * 4608,
                                bb, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
                            (void)rocblas_gemm_strided_batched_ex(
                                handle_, rocblas_operation_transpose, rocblas_operation_none,
                                5120, wm, 2304, &alpha,
                                d_w_f16_,    rocblas_datatype_f16_r, 2304, rocblas_stride(5120) * 2304,
                                d_zrot_f16_, rocblas_datatype_f16_r, 2304, rocblas_stride(wm) * 2304,
                                &beta,
                                d_y_f16, rocblas_datatype_f16_r, 5120, rocblas_stride(wm) * 5120,
                                d_y_f16, rocblas_datatype_f16_r, 5120, rocblas_stride(wm) * 5120,
                                bb, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
                        }
                    }
                } else {
                    warm_hhhs(1536, wm, 5120);
                    warm_hhhs(5120, wm, 768);
                    warm_hhhs(384,  wm, 5120);
                    if (wm == 64 || wm == 128 || wm == 384) {
                        // Pre-warm HHSS (F32 out) Tensile code objects used by router / comp / wo_a / wo_b
                        // so zero lazy hipModuleLoadData fires while multi-GPU HostRingAllReduce is in flight!
                        warm_hhss(384,  wm, 5120);
                        warm_hhss(512,  wm, 5120);
                        warm_hhss(5120, wm, 3072);
                        warm_hhss(5120, wm, 2048);
                        warm_hhhs(1792, wm, 5120);
                        warm_hhhs(5120, wm, 3072);
                        warm_hhhs(5120, wm, 2048);
                        (void)rocblas_gemm_strided_batched_ex(
                            handle_, rocblas_operation_transpose, rocblas_operation_none,
                            1024, wm, 4096, &alpha,
                            d_w_f16_, rocblas_datatype_f16_r, 4096, 1024LL * 4096LL,
                            d_x_f16_, rocblas_datatype_f16_r, 3 * 4096, 4096LL,
                            &beta,
                            d_y_f32_, rocblas_datatype_f32_r, 3 * 1024, 1024LL,
                            d_y_f32_, rocblas_datatype_f32_r, 3 * 1024, 1024LL,
                            3, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
                    }
                    if (wm == 64 || wm == 128) {
                        (void)rocblas_gemm_strided_batched_ex(
                            handle_, rocblas_operation_transpose, rocblas_operation_none,
                            1536, wm, 5120, &alpha,
                            d_w_f16_, rocblas_datatype_f16_r, 5120, rocblas_stride(1536) * 5120,
                            d_x_f16_, rocblas_datatype_f16_r, 5120, rocblas_stride(wm) * 5120,
                            &beta,
                            d_y_f16, rocblas_datatype_f16_r, 1536, rocblas_stride(wm) * 1536,
                            d_y_f16, rocblas_datatype_f16_r, 1536, rocblas_stride(wm) * 1536,
                            8, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
                        (void)rocblas_gemm_strided_batched_ex(
                            handle_, rocblas_operation_transpose, rocblas_operation_none,
                            5120, wm, 768, &alpha,
                            d_w_f16_, rocblas_datatype_f16_r, 768, rocblas_stride(5120) * 768,
                            d_zrot_f16_, rocblas_datatype_f16_r, 768, rocblas_stride(wm) * 768,
                            &beta,
                            d_y_f16, rocblas_datatype_f16_r, 5120, rocblas_stride(wm) * 5120,
                            d_y_f16, rocblas_datatype_f16_r, 5120, rocblas_stride(wm) * 5120,
                            8, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
                    }
                }
            }
            if (!is_dev3_) {
                // Wave 10 Phase C.1: Pre-warm all dense M>=1024 HHHS shapes (wqkv, wq_b, wo_a, wo_b, sh_w13, sh_w2)
                warm_hhhs(12288, 1024, 1280);
                warm_hhhs(8192,  1024, 1280);
                warm_hhhs(1792,  1024, 5120);
                warm_hhhs(1536,  1024, 5120);
                warm_hhhs(5120,  1024, 768);
                warm_hhhs(5120,  1024, 2048);
                warm_hhhs(5120,  1024, 3072);
                for (int gc = 2; gc <= 3; ++gc) {
                    (void)rocblas_gemm_strided_batched_ex(
                        handle_, rocblas_operation_transpose, rocblas_operation_none,
                        1024, 512, 4096, &alpha,
                        d_w_f16_, rocblas_datatype_f16_r, 4096, 1024LL * 4096LL,
                        d_x_f16_, rocblas_datatype_f16_r, gc * 4096, 4096LL,
                        &beta,
                        d_y_f16, rocblas_datatype_f16_r, gc * 1024, 1024LL,
                        d_y_f16, rocblas_datatype_f16_r, gc * 1024, 1024LL,
                        gc, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
                }
            }
            launch_w93_bf16_to_f16(d_w_f16_, d_x_f16_, 256u, stream);
            launch_w93_f32_to_bf16_rne(d_y_f32_, d_x_f16_, 256u, stream);
            launch_w94_f16_to_bf16_rne(d_y_f16, d_x_f16_, 256u, stream);
            launch_w10_f16_to_f32(d_x_f16_, d_y_f32_, 256u, stream);
            launch_w93_unpack_fp8_to_f16(
                reinterpret_cast<const uint8_t*>(d_w_f16_),
                reinterpret_cast<const uint8_t*>(d_x_f16_),
                d_zrot_f16_, 32, 32, false, stream);
            launch_w94_swiglu10_wht128_f16_to_f16(
                d_y_f16, reinterpret_cast<const uint64_t*>(d_w_f16_), d_zrot_f16_, 1, 128, stream);
            check_hip(hipStreamSynchronize(stream), "prewarm RocblasPrefillEngine");
        }
        active_ = true;
    }

    void shutdown() {
        if (!active_ && handle_ == nullptr) return;
        (void)hipSetDevice(dev_id_);
        if (handle_ != nullptr)      { (void)rocblas_destroy_handle(handle_); handle_ = nullptr; }
        if (d_rb_ws_ != nullptr)     { (void)hipFree(d_rb_ws_);      d_rb_ws_ = nullptr; }
        if (d_w_f16_ != nullptr)     { (void)hipFree(d_w_f16_);      d_w_f16_ = nullptr; }
        if (d_x_f16_ != nullptr)     { (void)hipFree(d_x_f16_);      d_x_f16_ = nullptr; }
        if (d_y_f32_ != nullptr)     { (void)hipFree(d_y_f32_);      d_y_f32_ = nullptr; }
        if (d_zrot_f16_ != nullptr)  { (void)hipFree(d_zrot_f16_);   d_zrot_f16_ = nullptr; }
        if (d_batch_eids_ != nullptr){ (void)hipFree(d_batch_eids_); d_batch_eids_ = nullptr; }
        if (h_batch_eids_ != nullptr){ (void)hipHostFree(h_batch_eids_); h_batch_eids_ = nullptr; }
        active_ = false;
    }

    bool active() const noexcept { return active_; }
    size_t total_allocated_bytes() const noexcept {
        return ws_bytes_ +
               (w_f16_elems_ + x_f16_elems_ + zrot_f16_elems_) * sizeof(uint16_t) +
               y_f32_elems_ * sizeof(float);
    }

    // Dense FP8 E4M3 + UE8M0_32 GEMM via Weight-Unpack-Once->FP16 + row-tiled rocblas_gemm_ex(FP16->FP32).
    // Caller must have hipSetDevice(dev_id_) active.
    bool run_fp8_gemm(
        const uint8_t* d_x_fp8, const uint8_t* d_x_scale,
        const uint8_t* d_w_fp8, const uint8_t* d_w_scale,
        uint16_t* d_out_bf16, float* d_out_f32,
        int M, int N, int K, bool write_f32, bool accumulate_f32, hipStream_t stream)
    {
        if (!active_ || M < kDefaultDenseThreshM || (K & 31) != 0) return false;
        if (size_t(N) * size_t(K) > w_f16_elems_) return false;

        const bool use_hhhs = (!write_f32) || (!accumulate_f32 && M >= 1024);
        int max_m = int(x_f16_elems_ / size_t(K));
        if (use_hhhs) {
            // HHHS writes FP16 into aliased d_y_f32_ (2x elements!) -> hits fast 20.5 TFLOPS Tensile assembly!
            max_m = std::min(max_m, int((y_f32_elems_ * 2u) / size_t(N)));
        }
        if (max_m <= 0) return false;

        // Unpack weight matrix ONCE regardless of M!
        launch_w93_unpack_fp8_to_f16(d_w_fp8, d_w_scale, d_w_f16_, N, K, false, stream);

        const float alpha = 1.0f;
        const float beta  = (write_f32 && accumulate_f32) ? 1.0f : 0.0f;
        const int k_blks  = K >> 5;
        uint16_t* d_y_f16 = reinterpret_cast<uint16_t*>(d_y_f32_);
        check_rb(rocblas_set_stream(handle_, stream), "rocblas_set_stream");

        for (int m0 = 0; m0 < M; m0 += max_m) {
            const int cur_m = std::min(max_m, M - m0);
            launch_w93_unpack_fp8_to_f16(
                d_x_fp8 + size_t(m0) * size_t(K),
                d_x_scale + size_t(m0) * size_t(k_blks),
                d_x_f16_, cur_m, K, true, stream);
            if (!use_hhhs) {
                float* c_dst = d_out_f32 + size_t(m0) * size_t(N);
                check_rb(rocblas_gemm_ex(
                    handle_, rocblas_operation_transpose, rocblas_operation_none,
                    N, cur_m, K, &alpha,
                    d_w_f16_, rocblas_datatype_f16_r, K,
                    d_x_f16_, rocblas_datatype_f16_r, K,
                    &beta,
                    c_dst, rocblas_datatype_f32_r, N,
                    c_dst, rocblas_datatype_f32_r, N,
                    rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0),
                    "rocblas_gemm_ex(fp8_f32)");
            } else {
                check_rb(rocblas_gemm_ex(
                    handle_, rocblas_operation_transpose, rocblas_operation_none,
                    N, cur_m, K, &alpha,
                    d_w_f16_, rocblas_datatype_f16_r, K,
                    d_x_f16_, rocblas_datatype_f16_r, K,
                    &beta,
                    d_y_f16, rocblas_datatype_f16_r, N,
                    d_y_f16, rocblas_datatype_f16_r, N,
                    rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0),
                    "rocblas_gemm_ex(fp8_hhhs)");
                if (write_f32 && d_out_f32 != nullptr) {
                    launch_w10_f16_to_f32(d_y_f16, d_out_f32 + size_t(m0) * size_t(N),
                                          size_t(cur_m) * size_t(N), stream);
                } else if (d_out_bf16 != nullptr) {
                    launch_w94_f16_to_bf16_rne(d_y_f16, d_out_bf16 + size_t(m0) * size_t(N),
                                               size_t(cur_m) * size_t(N), stream);
                }
            }
        }
        return true;
    }

    // Grouped wo_a FP8 GEMM: g_cnt groups (each [M, 4096] * [1024, 4096]^T -> [M, 1024] interleaved into [M, g_cnt*1024])
    // Wave 10 Phase C.1: Upgraded to HHHS (FP16-out, FP32-compute -> 20.5 TFLOPS Tensile + 2x larger max_m!).
    bool run_fp8_grouped_wo_a(
        const uint8_t* d_o_fp8, const uint8_t* d_o_scale,
        const uint8_t* d_wo_a_fp8, const uint8_t* d_wo_a_scale,
        uint16_t* d_out_bf16, int M, int g_cnt, hipStream_t stream)
    {
        if (!active_ || M < kDefaultDenseThreshM || g_cnt <= 0 || g_cnt > 3) return false;
        const int K_tot = g_cnt * 4096;
        const int N_tot = g_cnt * 1024;
        if (size_t(N_tot) * 4096u > w_f16_elems_) return false;
        const int max_m = std::min(int(x_f16_elems_ / size_t(K_tot)),
                                   int((y_f32_elems_ * 2u) / size_t(N_tot)));
        if (max_m <= 0) return false;

        launch_w93_unpack_fp8_to_f16(d_wo_a_fp8, d_wo_a_scale, d_w_f16_, N_tot, 4096, false, stream);

        const float alpha = 1.0f, beta = 0.0f;
        const int k_blks = K_tot >> 5;
        uint16_t* d_y_f16 = reinterpret_cast<uint16_t*>(d_y_f32_);
        check_rb(rocblas_set_stream(handle_, stream), "rocblas_set_stream");

        for (int m0 = 0; m0 < M; m0 += max_m) {
            const int cur_m = std::min(max_m, M - m0);
            launch_w93_unpack_fp8_to_f16(
                d_o_fp8 + size_t(m0) * size_t(K_tot),
                d_o_scale + size_t(m0) * size_t(k_blks),
                d_x_f16_, cur_m, K_tot, true, stream);
            check_rb(rocblas_gemm_strided_batched_ex(
                handle_, rocblas_operation_transpose, rocblas_operation_none,
                1024, cur_m, 4096, &alpha,
                d_w_f16_, rocblas_datatype_f16_r, 4096, 1024LL * 4096LL,
                d_x_f16_, rocblas_datatype_f16_r, K_tot, 4096LL,
                &beta,
                d_y_f16, rocblas_datatype_f16_r, N_tot, 1024LL,
                d_y_f16, rocblas_datatype_f16_r, N_tot, 1024LL,
                g_cnt, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0),
                "rocblas_gemm_strided_batched_ex(wo_a_hhhs)");
            launch_w94_f16_to_bf16_rne(d_y_f16, d_out_bf16 + size_t(m0) * size_t(N_tot),
                                       size_t(cur_m) * size_t(N_tot), stream);
        }
        return true;
    }

    // Dense BF16 GEMM via Weight-Unpack-Once BF16->FP16 + row-tiled rocblas_gemm_ex(FP16->FP32)
    bool run_bf16_gemm(
        const uint16_t* d_x_bf16, const uint16_t* d_w_bf16,
        uint16_t* d_out_bf16, float* d_out_f32,
        int M, int N, int K, bool write_f32, hipStream_t stream)
    {
        if (!active_ || M < kDefaultDenseThreshM || N < 64) return false;
        if (size_t(N) * size_t(K) > w_f16_elems_) return false;
        int max_m = int(x_f16_elems_ / size_t(K));
        if (!write_f32) {
            max_m = std::min(max_m, int(y_f32_elems_ / size_t(N)));
        }
        if (max_m <= 0) return false;

        launch_w93_bf16_to_f16(d_w_bf16, d_w_f16_, size_t(N) * size_t(K), stream);

        const float alpha = 1.0f, beta = 0.0f;
        check_rb(rocblas_set_stream(handle_, stream), "rocblas_set_stream");

        for (int m0 = 0; m0 < M; m0 += max_m) {
            const int cur_m = std::min(max_m, M - m0);
            launch_w93_bf16_to_f16(d_x_bf16 + size_t(m0) * size_t(K), d_x_f16_, size_t(cur_m) * size_t(K), stream);
            float* c_dst = write_f32 ? (d_out_f32 + size_t(m0) * size_t(N)) : d_y_f32_;
            check_rb(rocblas_gemm_ex(
                handle_, rocblas_operation_transpose, rocblas_operation_none,
                N, cur_m, K, &alpha,
                d_w_f16_, rocblas_datatype_f16_r, K,
                d_x_f16_, rocblas_datatype_f16_r, K,
                &beta,
                c_dst, rocblas_datatype_f32_r, N,
                c_dst, rocblas_datatype_f32_r, N,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0),
                "rocblas_gemm_ex(bf16)");
            if (!write_f32 && d_out_bf16 != nullptr) {
                launch_w93_f32_to_bf16_rne(d_y_f32_, d_out_bf16 + size_t(m0) * size_t(N),
                                           size_t(cur_m) * size_t(N), stream);
            }
        }
        return true;
    }

    // Routed HLUT2 Expert Bucket (TP3 N_inter=768, H=5120) via HHHS (FP16-out, FP32-compute v_dot2_f32_f16!)
    void run_hlut2_expert_bucket(
        const uint16_t* d_x_rot_f16,
        const Mi50RoutedPair* d_sorted_pairs,
        int expert_id, int p_start, int m_e,
        const uint32_t* d_gate_codes, const uint16_t* d_gate_sc, const float* d_gate_cent,
        const uint32_t* d_up_codes,   const uint16_t* d_up_sc,   const float* d_up_cent,
        const uint32_t* d_down_codes, const uint16_t* d_down_sc, const float* d_down_cent,
        const uint64_t* d_signs_i,
        float* d_out_accum_f32,
        hipStream_t stream)
    {
        if (!active_ || m_e <= 0) return;
        const int Ni = 768;
        const float alpha = 1.0f, beta = 0.0f;
        uint16_t* d_y_f16 = reinterpret_cast<uint16_t*>(d_y_f32_);
        check_rb(rocblas_set_stream(handle_, stream), "rocblas_set_stream");

        launch_w93_unpack_hlut2_gateup_f16(
            d_gate_codes, d_gate_sc, d_gate_cent,
            d_up_codes, d_up_sc, d_up_cent,
            expert_id, Ni, kHidden, d_w_f16_, stream);

        for (int off = 0; off < m_e; off += max_bucket_m_) {
            const int cur_m = std::min(max_bucket_m_, m_e - off);
            launch_w93_gather_rows_f16(d_x_rot_f16, d_sorted_pairs, p_start + off, cur_m, d_x_f16_, stream);
            check_rb(rocblas_gemm_ex(
                handle_, rocblas_operation_transpose, rocblas_operation_none,
                2 * Ni, cur_m, kHidden, &alpha,
                d_w_f16_, rocblas_datatype_f16_r, kHidden,
                d_x_f16_, rocblas_datatype_f16_r, kHidden,
                &beta,
                d_y_f16, rocblas_datatype_f16_r, 2 * Ni,
                d_y_f16, rocblas_datatype_f16_r, 2 * Ni,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0),
                "rocblas_gemm_ex(hlut2_gu_hhhs)");
            launch_w94_swiglu10_wht128_f16_to_f16(d_y_f16, d_signs_i, d_zrot_f16_, cur_m, Ni, stream);
            launch_w93_unpack_hlut2_expert_f16(d_down_codes, d_down_sc, d_down_cent, expert_id, kHidden, Ni, d_w_f16_, stream);
            check_rb(rocblas_gemm_ex(
                handle_, rocblas_operation_transpose, rocblas_operation_none,
                kHidden, cur_m, Ni, &alpha,
                d_w_f16_,    rocblas_datatype_f16_r, Ni,
                d_zrot_f16_, rocblas_datatype_f16_r, Ni,
                &beta,
                d_y_f16, rocblas_datatype_f16_r, kHidden,
                d_y_f16, rocblas_datatype_f16_r, kHidden,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0),
                "rocblas_gemm_ex(hlut2_down_hhhs)");
            launch_w94_scatter_add_weighted_f16(d_y_f16, d_sorted_pairs, p_start + off, cur_m, d_out_accum_f32, stream);
            if (off + cur_m < m_e) {
                launch_w93_unpack_hlut2_gateup_f16(
                    d_gate_codes, d_gate_sc, d_gate_cent,
                    d_up_codes, d_up_sc, d_up_cent,
                    expert_id, Ni, kHidden, d_w_f16_, stream);
            }
        }
    }

    // Routed WHT128-HLUT3 Expert Bucket (TP3 Ni=768 & dev3 unsharded Ni=2304!) via HHHS (3.16x faster on dev3!)
    void run_hlut3_expert_bucket(
        const uint16_t* d_x_rot_f16,
        const Mi50RoutedPair* d_sorted_pairs,
        int expert_id, int p_start, int m_e, int N_inter,
        const Hlut3Tile32x32* d_gate_tiles,
        const Hlut3Tile32x32* d_up_tiles,
        const Hlut3Tile32x32* d_down_tiles,
        const float* d_levels,
        const uint64_t* d_signs_i,
        float* d_out_accum_f32,
        hipStream_t stream)
    {
        if (!active_ || m_e <= 0) return;
        const float alpha = 1.0f, beta = 0.0f;
        uint16_t* d_y_f16 = reinterpret_cast<uint16_t*>(d_y_f32_);
        check_rb(rocblas_set_stream(handle_, stream), "rocblas_set_stream");

        launch_w93_unpack_hlut3_gateup_f16(
            d_gate_tiles, d_up_tiles, d_levels, expert_id, N_inter, kHidden, d_w_f16_, stream);

        for (int off = 0; off < m_e; off += max_bucket_m_) {
            const int cur_m = std::min(max_bucket_m_, m_e - off);
            // Align Tensile M to multiple of 32 (or 16 for small m); zero-pad directly in gather (0 extra hipMemsetAsync!)
            const int m_pad = (cur_m <= 16) ? 16 : std::min(max_bucket_m_, (cur_m + 31) & ~31);
            launch_w93_gather_rows_f16(d_x_rot_f16, d_sorted_pairs, p_start + off, cur_m, d_x_f16_, stream, m_pad);
            check_rb(rocblas_gemm_ex(
                handle_, rocblas_operation_transpose, rocblas_operation_none,
                2 * N_inter, m_pad, kHidden, &alpha,
                d_w_f16_, rocblas_datatype_f16_r, kHidden,
                d_x_f16_, rocblas_datatype_f16_r, kHidden,
                &beta,
                d_y_f16, rocblas_datatype_f16_r, 2 * N_inter,
                d_y_f16, rocblas_datatype_f16_r, 2 * N_inter,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0),
                "rocblas_gemm_ex(hlut3_gu_hhhs)");
            launch_w94_swiglu10_wht128_f16_to_f16(d_y_f16, d_signs_i, d_zrot_f16_, m_pad, N_inter, stream);
            launch_w93_unpack_hlut3_expert_f16(d_down_tiles, d_levels, expert_id, kHidden, N_inter, d_w_f16_, stream);
            check_rb(rocblas_gemm_ex(
                handle_, rocblas_operation_transpose, rocblas_operation_none,
                kHidden, m_pad, N_inter, &alpha,
                d_w_f16_,    rocblas_datatype_f16_r, N_inter,
                d_zrot_f16_, rocblas_datatype_f16_r, N_inter,
                &beta,
                d_y_f16, rocblas_datatype_f16_r, kHidden,
                d_y_f16, rocblas_datatype_f16_r, kHidden,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0),
                "rocblas_gemm_ex(hlut3_down_hhhs)");
            launch_w94_scatter_add_weighted_f16(d_y_f16, d_sorted_pairs, p_start + off, cur_m, d_out_accum_f32, stream);
            if (off + cur_m < m_e) {
                launch_w93_unpack_hlut3_gateup_f16(
                    d_gate_tiles, d_up_tiles, d_levels, expert_id, N_inter, kHidden, d_w_f16_, stream);
            }
        }
    }

    // Wave 9.4 Phase C.2: B=8 Strided-Batched HHHS HLUT2 Expert Dispatch (3.15x faster than serial F32!)
    // Caller sorts `buckets` by m_e descending so each group of up to 8 experts has tightly matched m_pad!
    void run_hlut2_expert_buckets_batched(
        const uint16_t* d_x_rot_f16,
        const Mi50RoutedPair* d_sorted_pairs,
        const ExpertBucketDesc* buckets,
        int n_buckets,
        const uint32_t* d_gate_codes, const uint16_t* d_gate_sc, const float* d_gate_cent,
        const uint32_t* d_up_codes,   const uint16_t* d_up_sc,   const float* d_up_cent,
        const uint32_t* d_down_codes, const uint16_t* d_down_sc, const float* d_down_cent,
        const uint64_t* d_signs_i,
        float* d_out_accum_f32,
        hipStream_t stream)
    {
        if (!active_ || n_buckets <= 0) return;
        const int Ni = 768;
        const size_t gu_stride = size_t(2 * Ni) * size_t(kHidden);
        const size_t dn_stride = size_t(kHidden) * size_t(Ni);
        const int max_b = is_dev3_
            ? std::min(offload_max_b_, int(w_f16_elems_ / gu_stride))
            : std::min(8, int(w_f16_elems_ / gu_stride));
        if (max_b < 2) {
            for (int i = 0; i < n_buckets; ++i) {
                run_hlut2_expert_bucket(
                    d_x_rot_f16, d_sorted_pairs, buckets[i].eid, buckets[i].p_start, buckets[i].m_e,
                    d_gate_codes, d_gate_sc, d_gate_cent, d_up_codes, d_up_sc, d_up_cent,
                    d_down_codes, d_down_sc, d_down_cent, d_signs_i, d_out_accum_f32, stream);
            }
            return;
        }

        const int cap_n = std::min(n_buckets, 256);
        const size_t slot_off = size_t((batch_meta_slot_++) & 3u) * 768u;
        int32_t* h_base    = h_batch_eids_ + slot_off;
        int32_t* h_eids    = h_base;
        int32_t* h_pstarts = h_base + 256;
        int32_t* h_mes     = h_base + 512;
        for (int i = 0; i < cap_n; ++i) {
            h_eids[i]    = buckets[i].eid;
            h_pstarts[i] = buckets[i].p_start;
            h_mes[i]     = buckets[i].m_e;
        }
        int32_t* d_base    = d_batch_eids_ + slot_off;
        int32_t* d_eids    = d_base;
        int32_t* d_pstarts = d_base + 256;
        int32_t* d_mes     = d_base + 512;
        check_hip(hipMemcpyAsync(d_base, h_base, 768u * sizeof(int32_t),
                                 hipMemcpyHostToDevice, stream), "cp d_batch_meta(hlut2)");
        check_rb(rocblas_set_stream(handle_, stream), "rocblas_set_stream");
        uint16_t* d_y_f16 = reinterpret_cast<uint16_t*>(d_y_f32_);
        const int max_tot_rows = int(std::min<size_t>(x_f16_elems_ / size_t(kHidden),
                                                      (y_f32_elems_ * 2u) / size_t(kHidden)));
        const float alpha = 1.0f, beta = 0.0f;

        int i = 0;
        while (i < n_buckets) {
            if (i >= cap_n) {
                run_hlut2_expert_bucket(
                    d_x_rot_f16, d_sorted_pairs, buckets[i].eid, buckets[i].p_start, buckets[i].m_e,
                    d_gate_codes, d_gate_sc, d_gate_cent, d_up_codes, d_up_sc, d_up_cent,
                    d_down_codes, d_down_sc, d_down_cent, d_signs_i, d_out_accum_f32, stream);
                ++i;
                continue;
            }
            int m_pad = (buckets[i].m_e <= 16) ? 16 : ((buckets[i].m_e + 15) & ~15);
            int b_cur = 1;
            while (b_cur < max_b && i + b_cur < cap_n) {
                const int bm = buckets[i + b_cur].m_e;
                const int bpad = (bm <= 16) ? 16 : ((bm + 15) & ~15);
                const int cand_pad = std::max(m_pad, bpad);
                if ((b_cur + 1) * cand_pad > max_tot_rows) break;
                m_pad = cand_pad;
                ++b_cur;
            }
            if (b_cur == 1) {
                run_hlut2_expert_bucket(
                    d_x_rot_f16, d_sorted_pairs, buckets[i].eid, buckets[i].p_start, buckets[i].m_e,
                    d_gate_codes, d_gate_sc, d_gate_cent, d_up_codes, d_up_sc, d_up_cent,
                    d_down_codes, d_down_sc, d_down_cent, d_signs_i, d_out_accum_f32, stream);
                ++i;
                continue;
            }

            launch_w94_unpack_hlut2_gateup_batch_f16(
                d_gate_codes, d_gate_sc, d_gate_cent,
                d_up_codes,   d_up_sc,   d_up_cent,
                d_eids + i, b_cur, Ni, kHidden, d_w_f16_, stream);
            launch_w94_gather_padded_batch_f16(
                d_x_rot_f16, d_sorted_pairs, d_pstarts + i, d_mes + i, b_cur, m_pad, d_x_f16_, stream);
            check_rb(rocblas_gemm_strided_batched_ex(
                handle_, rocblas_operation_transpose, rocblas_operation_none,
                2 * Ni, m_pad, kHidden, &alpha,
                d_w_f16_, rocblas_datatype_f16_r, kHidden, rocblas_stride(gu_stride),
                d_x_f16_, rocblas_datatype_f16_r, kHidden, rocblas_stride(m_pad) * kHidden,
                &beta,
                d_y_f16, rocblas_datatype_f16_r, 2 * Ni, rocblas_stride(m_pad) * (2 * Ni),
                d_y_f16, rocblas_datatype_f16_r, 2 * Ni, rocblas_stride(m_pad) * (2 * Ni),
                b_cur, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0),
                "rocblas_gemm_strided_batched_ex(hlut2_gu_hhhs)");
            launch_w94_swiglu10_wht128_f16_to_f16(d_y_f16, d_signs_i, d_zrot_f16_, b_cur * m_pad, Ni, stream);
            launch_w94_unpack_hlut2_expert_batch_f16(
                d_down_codes, d_down_sc, d_down_cent,
                d_eids + i, b_cur, kHidden, Ni, d_w_f16_, stream);
            check_rb(rocblas_gemm_strided_batched_ex(
                handle_, rocblas_operation_transpose, rocblas_operation_none,
                kHidden, m_pad, Ni, &alpha,
                d_w_f16_,    rocblas_datatype_f16_r, Ni, rocblas_stride(dn_stride),
                d_zrot_f16_, rocblas_datatype_f16_r, Ni, rocblas_stride(m_pad) * Ni,
                &beta,
                d_y_f16, rocblas_datatype_f16_r, kHidden, rocblas_stride(m_pad) * kHidden,
                d_y_f16, rocblas_datatype_f16_r, kHidden, rocblas_stride(m_pad) * kHidden,
                b_cur, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0),
                "rocblas_gemm_strided_batched_ex(hlut2_dn_hhhs)");
            for (int bi = 0; bi < b_cur; ++bi) {
                launch_w94_scatter_add_weighted_f16(
                    d_y_f16 + size_t(bi) * size_t(m_pad) * size_t(kHidden),
                    d_sorted_pairs, buckets[i + bi].p_start, buckets[i + bi].m_e, d_out_accum_f32, stream);
            }
            i += b_cur;
        }
    }

    // Wave 9.4 Phase C.2: B=8 Strided-Batched HHHS WHT128-HLUT3 Expert Dispatch (TP3 Ni=768, single-pass HHHS on dev3!)
    void run_hlut3_expert_buckets_batched(
        const uint16_t* d_x_rot_f16,
        const Mi50RoutedPair* d_sorted_pairs,
        const ExpertBucketDesc* buckets,
        int n_buckets,
        int N_inter,
        const Hlut3Tile32x32* d_gate_tiles,
        const Hlut3Tile32x32* d_up_tiles,
        const Hlut3Tile32x32* d_down_tiles,
        const float* d_levels,
        const uint64_t* d_signs_i,
        float* d_out_accum_f32,
        hipStream_t stream)
    {
        if (!active_ || n_buckets <= 0) return;
        const size_t gu_stride = size_t(2 * N_inter) * size_t(kHidden);
        const size_t dn_stride = size_t(kHidden) * size_t(N_inter);
        const int max_b = is_dev3_
            ? std::min(offload_max_b_, int(w_f16_elems_ / gu_stride))
            : std::min(8, int(w_f16_elems_ / gu_stride));
        if (max_b < 2) {
            for (int i = 0; i < n_buckets; ++i) {
                run_hlut3_expert_bucket(
                    d_x_rot_f16, d_sorted_pairs, buckets[i].eid, buckets[i].p_start, buckets[i].m_e, N_inter,
                    d_gate_tiles, d_up_tiles, d_down_tiles, d_levels, d_signs_i, d_out_accum_f32, stream);
            }
            return;
        }

        const int cap_n = std::min(n_buckets, 256);
        const size_t slot_off = size_t((batch_meta_slot_++) & 3u) * 768u;
        int32_t* h_base    = h_batch_eids_ + slot_off;
        int32_t* h_eids    = h_base;
        int32_t* h_pstarts = h_base + 256;
        int32_t* h_mes     = h_base + 512;
        for (int i = 0; i < cap_n; ++i) {
            h_eids[i]    = buckets[i].eid;
            h_pstarts[i] = buckets[i].p_start;
            h_mes[i]     = buckets[i].m_e;
        }
        int32_t* d_base    = d_batch_eids_ + slot_off;
        int32_t* d_eids    = d_base;
        int32_t* d_pstarts = d_base + 256;
        int32_t* d_mes     = d_base + 512;
        check_hip(hipMemcpyAsync(d_base, h_base, 768u * sizeof(int32_t),
                                 hipMemcpyHostToDevice, stream), "cp d_batch_meta(hlut3)");
        check_rb(rocblas_set_stream(handle_, stream), "rocblas_set_stream");
        uint16_t* d_y_f16 = reinterpret_cast<uint16_t*>(d_y_f32_);
        const int max_tot_rows = int(std::min<size_t>(x_f16_elems_ / size_t(kHidden),
                                                      (y_f32_elems_ * 2u) / size_t(kHidden)));
        const float alpha = 1.0f, beta = 0.0f;

        int i = 0;
        while (i < n_buckets) {
            if (i >= cap_n) {
                run_hlut3_expert_bucket(
                    d_x_rot_f16, d_sorted_pairs, buckets[i].eid, buckets[i].p_start, buckets[i].m_e, N_inter,
                    d_gate_tiles, d_up_tiles, d_down_tiles, d_levels, d_signs_i, d_out_accum_f32, stream);
                ++i;
                continue;
            }
            int m_pad = (buckets[i].m_e <= 16) ? 16 : ((buckets[i].m_e + 15) & ~15);
            int b_cur = 1;
            while (b_cur < max_b && i + b_cur < cap_n) {
                const int bm = buckets[i + b_cur].m_e;
                const int bpad = (bm <= 16) ? 16 : ((bm + 15) & ~15);
                const int cand_pad = std::max(m_pad, bpad);
                if ((b_cur + 1) * cand_pad > max_tot_rows) break;
                m_pad = cand_pad;
                ++b_cur;
            }
            if (b_cur == 1) {
                run_hlut3_expert_bucket(
                    d_x_rot_f16, d_sorted_pairs, buckets[i].eid, buckets[i].p_start, buckets[i].m_e, N_inter,
                    d_gate_tiles, d_up_tiles, d_down_tiles, d_levels, d_signs_i, d_out_accum_f32, stream);
                ++i;
                continue;
            }

            launch_w94_unpack_hlut3_gateup_batch_f16(
                d_gate_tiles, d_up_tiles, d_levels,
                d_eids + i, b_cur, N_inter, kHidden, d_w_f16_, stream);
            launch_w94_gather_padded_batch_f16(
                d_x_rot_f16, d_sorted_pairs, d_pstarts + i, d_mes + i, b_cur, m_pad, d_x_f16_, stream);
            check_rb(rocblas_gemm_strided_batched_ex(
                handle_, rocblas_operation_transpose, rocblas_operation_none,
                2 * N_inter, m_pad, kHidden, &alpha,
                d_w_f16_, rocblas_datatype_f16_r, kHidden, rocblas_stride(gu_stride),
                d_x_f16_, rocblas_datatype_f16_r, kHidden, rocblas_stride(m_pad) * kHidden,
                &beta,
                d_y_f16, rocblas_datatype_f16_r, 2 * N_inter, rocblas_stride(m_pad) * (2 * N_inter),
                d_y_f16, rocblas_datatype_f16_r, 2 * N_inter, rocblas_stride(m_pad) * (2 * N_inter),
                b_cur, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0),
                "rocblas_gemm_strided_batched_ex(hlut3_gu_hhhs)");
            launch_w94_swiglu10_wht128_f16_to_f16(d_y_f16, d_signs_i, d_zrot_f16_, b_cur * m_pad, N_inter, stream);
            launch_w94_unpack_hlut3_expert_batch_f16(
                d_down_tiles, d_levels,
                d_eids + i, b_cur, kHidden, N_inter, d_w_f16_, stream);
            check_rb(rocblas_gemm_strided_batched_ex(
                handle_, rocblas_operation_transpose, rocblas_operation_none,
                kHidden, m_pad, N_inter, &alpha,
                d_w_f16_,    rocblas_datatype_f16_r, N_inter, rocblas_stride(dn_stride),
                d_zrot_f16_, rocblas_datatype_f16_r, N_inter, rocblas_stride(m_pad) * N_inter,
                &beta,
                d_y_f16, rocblas_datatype_f16_r, kHidden, rocblas_stride(m_pad) * kHidden,
                d_y_f16, rocblas_datatype_f16_r, kHidden, rocblas_stride(m_pad) * kHidden,
                b_cur, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0),
                "rocblas_gemm_strided_batched_ex(hlut3_dn_hhhs)");
            for (int bi = 0; bi < b_cur; ++bi) {
                launch_w94_scatter_add_weighted_f16(
                    d_y_f16 + size_t(bi) * size_t(m_pad) * size_t(kHidden),
                    d_sorted_pairs, buckets[i + bi].p_start, buckets[i + bi].m_e, d_out_accum_f32, stream);
            }
            i += b_cur;
        }
    }

private:
    static void check_hip(hipError_t e, const char* w) {
        if (e != hipSuccess) throw std::runtime_error(std::string(w) + ": " + hipGetErrorString(e));
    }
    static void check_rb(rocblas_status s, const char* w) {
        if (s != rocblas_status_success) {
            throw std::runtime_error(std::string(w) + ": rocblas_status=" + std::to_string(int(s)));
        }
    }

    int dev_id_ = 0;
    bool is_dev3_ = false;
    int offload_max_b_ = 1;
    bool active_ = false;
    int max_bucket_m_ = 1024;
    int n_inter_ = 768;
    rocblas_handle handle_ = nullptr;
    void*     d_rb_ws_      = nullptr;
    uint16_t* d_w_f16_      = nullptr;
    uint16_t* d_x_f16_      = nullptr;
    float*    d_y_f32_      = nullptr;
    uint16_t* d_zrot_f16_   = nullptr;
    int32_t*  d_batch_eids_ = nullptr;
    int32_t*  h_batch_eids_ = nullptr;
    uint32_t  batch_meta_slot_ = 0u;
    size_t ws_bytes_ = 0;
    size_t w_f16_elems_ = 0;
    size_t x_f16_elems_ = 0;
    size_t y_f32_elems_ = 0;
    size_t zrot_f16_elems_ = 0;
};

}  // namespace dsv41
