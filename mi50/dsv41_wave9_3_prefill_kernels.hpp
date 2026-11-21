#pragma once
// Wave 9.3 Phase B: Zero-Scratch gfx906 Unpack, Gather, SwiGLU+WHT128, & Scatter Kernels
// powering Weight-Dequantize-Once + rocBLAS FP16->FP32 (v_dot2_f32_f16) Large-M Prefill!
// Zero ScratchSize, 0 SGPR/VGPR spills, 0 trig, 0 __HIP_MEMORY_SCOPE_SYSTEM RMW atomics.

#include "dsv41_mi50_kernels.hpp"
#include "../src/hlut3_pack.hpp"

#include <hip/hip_runtime.h>
#include <cstdint>

namespace dsv41 {

// Unpack FP8 E4M3 + UE8M0_32 matrix [N, K] -> row-major FP16 [N, K] (clamped safely to FP16 finite range).
// per_row_scale == true  : scale is [N, K/32] (activations)
// per_row_scale == false : scale is [N/32, K/32] (weights)
void launch_w93_unpack_fp8_to_f16(
    const uint8_t* d_fp8,
    const uint8_t* d_ue8m0,
    uint16_t* d_out_f16,
    int N, int K,
    bool per_row_scale,
    hipStream_t stream);

// Convert BF16 [M, K] -> FP16 [M, K] (clamped safely to [-65504, 65504]).
void launch_w93_bf16_to_f16(
    const uint16_t* d_in_bf16,
    uint16_t* d_out_f16,
    size_t n_elems,
    hipStream_t stream);

// Convert FP32 [M, N] -> BF16 [M, N] (RNE).
void launch_w93_f32_to_bf16_rne(
    const float* d_in_f32,
    uint16_t* d_out_bf16,
    size_t n_elems,
    hipStream_t stream);

// Unpack ONE TP3 HLUT2 expert projection [N, K] -> row-major FP16 [N, K]
void launch_w93_unpack_hlut2_expert_f16(
    const uint32_t* d_codes,   // [E, N/32, K/16, 32]
    const uint16_t* d_scales,  // [E, N/32, K/128, 32]
    const float*    d_cent,    // [E, 2]
    int expert_id,
    int N, int K,
    uint16_t* d_out_f16,       // [N, K] row-major FP16
    hipStream_t stream);

// Fused single-launch Gate+Up HLUT2 unpack [2*N_inter, K] -> contiguous row-major FP16 [2*N_inter, K]
void launch_w93_unpack_hlut2_gateup_f16(
    const uint32_t* d_gate_codes, const uint16_t* d_gate_scales, const float* d_gate_cent,
    const uint32_t* d_up_codes,   const uint16_t* d_up_scales,   const float* d_up_cent,
    int expert_id, int N_inter, int K,
    uint16_t* d_out_gu_f16,
    hipStream_t stream);

// Unpack ONE WHT128-HLUT3 expert projection [N, K] -> row-major FP16 [N, K]
// Supports both TP3 [768, 5120] / [5120, 768] and unsharded dev3 [2304, 5120] / [5120, 2304].
void launch_w93_unpack_hlut3_expert_f16(
    const Hlut3Tile32x32* d_tiles, // [E, N/32, K/32]
    const float* d_levels,         // [4]
    int expert_id,
    int N, int K,
    uint16_t* d_out_f16,           // [N, K] row-major FP16
    hipStream_t stream);

// Fused single-launch Gate+Up WHT128-HLUT3 unpack [2*N_inter, K] -> contiguous row-major FP16 [2*N_inter, K]
void launch_w93_unpack_hlut3_gateup_f16(
    const Hlut3Tile32x32* d_gate_tiles,
    const Hlut3Tile32x32* d_up_tiles,
    const float* d_levels,
    int expert_id, int N_inter, int K,
    uint16_t* d_out_gu_f16,
    hipStream_t stream);

// Gather m_e token activation rows from d_x_rot_f16[M, 5120] -> contiguous d_gather_f16[m_pad, 5120] (zeros r >= m_e).
void launch_w93_gather_rows_f16(
    const uint16_t* d_x_rot_f16,
    const Mi50RoutedPair* d_pairs,
    int p_start, int m_e,
    uint16_t* d_gather_f16,
    hipStream_t stream,
    int m_pad = 0);

// Fused Clamped-SwiGLU10 + WHT128(signs_i) -> FP16:
// Input d_gu_f32 is [m_e, 2 * N_inter] where row r has gate[0..N_inter-1] followed by up[0..N_inter-1].
// Output d_zrot_f16 is [m_e, N_inter] FP16.
void launch_w93_swiglu10_wht128_f32_to_f16(
    const float* d_gu_f32,
    const uint64_t* d_signs_i,
    uint16_t* d_zrot_f16,
    int m_e, int N_inter,
    hipStream_t stream);

// Weighted scatter-accumulate from contiguous d_y_f32[m_e, 5120] -> d_out_accum_f32[pr.row, 5120] (local HBM2 atomicAdd).
void launch_w93_scatter_add_weighted_f32(
    const float* d_y_f32,
    const Mi50RoutedPair* d_pairs,
    int p_start, int m_e,
    float* d_out_accum_f32,
    hipStream_t stream);

// ---- Wave 9.4 Phase C: Batched Unpack & dev3 INT8/BF16 Wire Kernels (all ScratchSize == 0) ----

// Batched Gate+Up HLUT2 unpack for B (1..4) experts into contiguous d_out_gu_f16[B, 2*N_inter, K]
void launch_w94_unpack_hlut2_gateup_batch_f16(
    const uint32_t* d_gate_codes, const uint16_t* d_gate_scales, const float* d_gate_cent,
    const uint32_t* d_up_codes,   const uint16_t* d_up_scales,   const float* d_up_cent,
    const int32_t* d_eids, int batch_b, int N_inter, int K,
    uint16_t* d_out_gu_f16, hipStream_t stream);

// Batched Down HLUT2 unpack for B (1..4) experts into contiguous d_out_f16[B, N, K]
void launch_w94_unpack_hlut2_expert_batch_f16(
    const uint32_t* d_codes, const uint16_t* d_scales, const float* d_cent,
    const int32_t* d_eids, int batch_b, int N, int K,
    uint16_t* d_out_f16, hipStream_t stream);

// Batched Gate+Up WHT128-HLUT3 unpack for B (1..4) experts into contiguous d_out_gu_f16[B, 2*N_inter, K]
void launch_w94_unpack_hlut3_gateup_batch_f16(
    const Hlut3Tile32x32* d_gate_tiles, const Hlut3Tile32x32* d_up_tiles,
    const float* d_levels, const int32_t* d_eids, int batch_b, int N_inter, int K,
    uint16_t* d_out_gu_f16, hipStream_t stream);

// Batched Down WHT128-HLUT3 unpack for B (1..4) experts into contiguous d_out_f16[B, N, K]
void launch_w94_unpack_hlut3_expert_batch_f16(
    const Hlut3Tile32x32* d_tiles, const float* d_levels,
    const int32_t* d_eids, int batch_b, int N, int K,
    uint16_t* d_out_f16, hipStream_t stream);

// Row-wise symmetric INT8 + FP16 row-scale quantization of WHT128-rotated FP16 activations [M, K] (K multiple of 4)
void launch_w94_quant_f16_to_int8_rowwise(
    const uint16_t* d_in_f16,
    int8_t* d_out_i8,
    uint16_t* d_out_scales_f16,
    int M, int K,
    hipStream_t stream);

// Row-wise symmetric INT8 + FP16 row-scale dequantization -> FP16 [M, K] (K multiple of 4)
void launch_w94_dequant_int8_to_f16_rowwise(
    const int8_t* d_in_i8,
    const uint16_t* d_in_scales_f16,
    uint16_t* d_out_f16,
    int M, int K,
    hipStream_t stream);

// In-place add BF16 vector onto FP32 accumulator (used for 2x smaller dev3 -> TP0 BF16 output transfer!)
void launch_w94_add_bf16_to_f32_inplace(
    float* d_dst_f32,
    const uint16_t* d_src_bf16,
    size_t n_elems,
    hipStream_t stream);

// Wave 9.4 Phase C.2 (HHHS FP16-out FP32-compute Tensile kernels):
// Convert FP16 -> BF16 RNE (2x less HBM2 read traffic than F32->BF16)
void launch_w94_f16_to_bf16_rne(
    const uint16_t* d_in_f16,
    uint16_t* d_out_bf16,
    size_t n_elems,
    hipStream_t stream);

// Wave 10 Phase C.1: Vectorized FP16 -> FP32 (used for HHHS FP8 GEMMs targeting F32 accumulators like sh_w2)
void launch_w10_f16_to_f32(
    const uint16_t* d_in_f16,
    float* d_out_f32,
    size_t n_elems,
    hipStream_t stream);

// Wave 10 Phase C: Single-pass fused FP32 + up to 3 optional BF16 offload streams (dev3 + dev4 + P150A) -> BF16 RNE
// Eliminates 3x post-AllReduce P150A hipMemcpyAsync H2D copies and 2x FP32 RMW passes on tp=0!
void launch_w10_add_3bf16_to_f32_and_bf16(
    const float* d_base_f32,
    const uint16_t* d_add0_bf16,
    const uint16_t* d_add1_bf16,
    const uint16_t* d_add2_bf16,
    uint16_t* d_out_bf16,
    size_t n_elems,
    hipStream_t stream);

// Fused Clamped-SwiGLU10 + WHT128(signs_i) directly from HHHS FP16 [m_rows, 2*N_inter] -> FP16 [m_rows, N_inter]
void launch_w94_swiglu10_wht128_f16_to_f16(
    const uint16_t* d_gu_f16,
    const uint64_t* d_signs_i,
    uint16_t* d_zrot_f16,
    int m_rows, int N_inter,
    hipStream_t stream);

// Zero-atomic coalesced weighted scatter-add directly from HHHS FP16 [m_e, 5120] -> FP32 [M, 5120]
void launch_w94_scatter_add_weighted_f16(
    const uint16_t* d_y_f16,
    const Mi50RoutedPair* d_pairs,
    int p_start, int m_e,
    float* d_out_accum_f32,
    hipStream_t stream);

// Single-launch batched padded gather for B experts into [B, m_pad, 5120] FP16 (zeros r >= m_e)
void launch_w94_gather_padded_batch_f16(
    const uint16_t* d_x_rot_f16,
    const Mi50RoutedPair* d_pairs,
    const int32_t* d_pstarts,
    const int32_t* d_mes,
    int batch_b, int m_pad,
    uint16_t* d_gather_f16,
    hipStream_t stream);

}  // namespace dsv41
