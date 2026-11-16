#pragma once
// Wave 9 Deliverable C: zero-scratch gfx906 Vega 20 decode kernels for WHT128-HLUT3.
// All launches target the caller's HIP stream (bench tools lock hipSetDevice(2)).
// No device v_sin_f32/v_cos_f32; SwiGLU uses exp only. 0 SGPR/VGPR spills, ScratchSize == 0.

#include <hip/hip_runtime.h>
#include <cstdint>

#include "dsv41_mi50_kernels.hpp"
#include "../src/hlut3_pack.hpp"

namespace dsv41 {

void launch_hlut3_gemv_wht128(
    const float* d_x,                 // [M, K] IEEE FP32 row-major
    const Hlut3Tile32x32* d_tiles,    // [N/32 * K/32] 400 B tiles
    const float* d_levels,            // [4] positive magnitudes
    float* d_y,                       // [M, N] IEEE FP32
    int M, int N, int K,
    hipStream_t stream);

// Expert-aware routed HLUT3 pair pipeline. Tile planes are expert-major:
//   gate/up tiles [n_experts, n_gate/32, k_gate/32] 400 B tiles (rank TP3 slice: 768x5120)
//   down tiles    [n_experts, n_down/32, k_down/32] 400 B tiles (rank TP3 slice: 5120x768)
// Full production geometry uses 2304x5120 gate/up and 5120x2304 down. Every pair selects
// tiles + pr.expert * nt * kt, so multi-expert MTP routing is bit-exact.
void launch_mi50_routed_hlut3_pairs(
    const float* d_x,                 // [M, k_gate] FP32 row-major
    const Mi50RoutedPair* d_pairs,    // [P] routed pairs
    int P, int M, int n_experts,
    int n_gate, int k_gate, int n_down, int k_down,
    const Hlut3Tile32x32* d_gate_tiles,
    const Hlut3Tile32x32* d_up_tiles,
    const Hlut3Tile32x32* d_down_tiles,
    const float* d_gate_levels,
    const float* d_up_levels,
    const float* d_down_levels,
    const uint64_t* d_signs_i,          // [n_gate/64] Rademacher words
    float* d_inter_z_f32,               // [P, n_gate] scratch
    float* d_inter_zrot_f32,            // [P, n_gate] scratch
    float* d_out_accum_f32,             // [M, n_down] caller-zeroed IEEE FP32
    hipStream_t stream);

// FP16 activation -> IEEE FP32 staging for the opt-in HLUT3 MTP overlay path (runtime only).
void launch_hlut3_f16_to_f32(
    const uint16_t* d_in_f16,
    float* d_out_f32,
    int n_elems,
    hipStream_t stream);

}  // namespace dsv41
