#pragma once
// Wave 9.2: Zero-scratch gfx906 Vega 20 WHT128-HLUT3 kernels (unsharded single-GPU 0000:85:00.0
// + TP3-sharded 32 GiB core + grouped CSR batched prefill + zero-PCIe-AtomicOp coherent host IPC).
// 0 SGPR/VGPR spills, ScratchSize == 0, 0 device transcendentals/trig, 0 system-scope RMW atomics.

#include <hip/hip_runtime.h>
#include <cstdint>

#include "dsv41_mi50_kernels.hpp"
#include "../src/hlut3_pack.hpp"

namespace dsv41 {

inline constexpr size_t kHlut3OffloadRingSlots = 8u;
inline constexpr int    kHlut3OffloadMaxRows   = 64;
inline constexpr int    kHlut3OffloadMaxTopk   = 6;
inline constexpr int    kHlut3OffloadHidden    = 5120;

// Cache-line-aligned coherent pinned host DDR4 slot between TP3 coordinator (tp=0) and
// dedicated 4th GPU hlut3_dev=3 (0000:85:00.0). Strictly single-producer release/acquire
// sequence numbers (__ATOMIC_RELEASE / __ATOMIC_ACQUIRE) — ZERO PCIe RMW atomics.
struct Hlut3OffloadSlot {
    alignas(64) uint32_t ready_seq;
    uint32_t block_id;
    uint32_t rows;
    uint32_t top_k;
    int32_t  exp_begin;
    int32_t  exp_end;
    uint32_t include_mtp_shared;
    uint32_t pad0[9];

    alignas(64) uint32_t done_seq;
    uint32_t active_pairs;
    uint32_t pad1[14];

    alignas(64) uint32_t rank_done_seq[3][16];

    alignas(64) int32_t  topk_ids[kHlut3OffloadMaxRows * kHlut3OffloadMaxTopk];
    alignas(64) float    topk_weights[kHlut3OffloadMaxRows * kHlut3OffloadMaxTopk];
    alignas(64) uint16_t x_rot_f16[size_t(kHlut3OffloadMaxRows) * size_t(kHlut3OffloadHidden)];
    alignas(64) uint16_t x_norm_bf16[size_t(kHlut3OffloadMaxRows) * size_t(kHlut3OffloadHidden)];
    alignas(64) float    y_out_f32[size_t(kHlut3OffloadMaxRows) * size_t(kHlut3OffloadHidden)];
};

// ---- Baseline Wave 9 FP32-input HLUT3 kernels (preserved for verify_wave9_hlut3 / bench_hlut3_mi50) ----
void launch_hlut3_gemv_wht128(
    const float* d_x,                 // [M, K] IEEE FP32 row-major
    const Hlut3Tile32x32* d_tiles,    // [N/32 * K/32] 400 B tiles
    const float* d_levels,            // [4] positive magnitudes
    float* d_y,                       // [M, N] IEEE FP32
    int M, int N, int K,
    hipStream_t stream);

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
    float* d_out_accum_f32,             // [M, n_down] caller-initialized IEEE FP32
    hipStream_t stream);

void launch_hlut3_f16_to_f32(
    const uint16_t* d_in_f16,
    float* d_out_f32,
    int n_elems,
    hipStream_t stream);

// ---- Wave 9.2 Direct-FP16 Unsharded (hlut3_dev=3: 2304x5120) & TP3-Sharded (768x5120) Kernels ----
void launch_hlut3_unsharded_pairs_f16(
    const uint16_t* d_x_rot_f16,      // [M, 5120] FP16 WHT128-rotated input
    const Mi50RoutedPair* d_pairs,    // [P] routed pairs
    int P, int max_scratch_pairs, int n_experts,
    const Hlut3Tile32x32* d_gate_tiles,
    const Hlut3Tile32x32* d_up_tiles,
    const Hlut3Tile32x32* d_down_tiles,
    const float* d_levels,
    const uint64_t* d_signs_i,        // [36] Rademacher words for I=2304
    float* d_inter_z_f32,             // [max_scratch_pairs, 2304]
    float* d_inter_zrot_f32,          // [max_scratch_pairs, 2304]
    float* d_out_accum_f32,           // [M, 5120]
    hipStream_t stream);

void launch_hlut3_tp3_pairs_f16(
    const uint16_t* d_x_rot_f16,      // [M, 5120] FP16 WHT128-rotated input
    const Mi50RoutedPair* d_pairs,    // [P] routed pairs
    int P, int max_scratch_pairs, int n_experts,
    const Hlut3Tile32x32* d_gate_tiles,
    const Hlut3Tile32x32* d_up_tiles,
    const Hlut3Tile32x32* d_down_tiles,
    const float* d_levels,
    const uint64_t* d_signs_i,        // [12] Rademacher words for I_tp=768
    float* d_inter_z_f32,             // [max_scratch_pairs, 768]
    float* d_inter_zrot_f32,          // [max_scratch_pairs, 768]
    float* d_out_accum_f32,           // [M, 5120]
    hipStream_t stream);

// Token-grouped 8-row LDS tile-reuse HLUT3 dispatch (8x HBM2 weight reuse for M > 16 / batched prefill).
// Requires batch_pairs <= max_scratch_pairs (caller slices CSR batches to <= max_scratch_pairs).
void launch_hlut3_routed_grouped_f16(
    const uint16_t* d_x_rot_f16,      // [M, k_gate] FP16 rotated input
    const int32_t* d_active_experts,  // [num_act]
    const int32_t* d_csr_offsets,     // [num_act + 1], csr_offsets[num_act] == batch_pairs
    const Mi50RoutedPair* d_sorted_pairs, // [batch_pairs]
    int num_act, int batch_pairs, int n_experts,
    int n_gate, int k_gate, int n_down, int k_down,
    const Hlut3Tile32x32* d_gate_tiles,
    const Hlut3Tile32x32* d_up_tiles,
    const Hlut3Tile32x32* d_down_tiles,
    const float* d_levels,
    const uint64_t* d_signs_i,
    float* d_inter_z_f32,
    float* d_inter_zrot_f32,
    float* d_out_accum_f32,
    hipStream_t stream);

// GPU-side range-filtered pair packer & single-CTA CSR builder for [exp_begin, exp_end):
// Maps physical eid in [exp_begin, exp_end) -> local expert (eid - exp_begin); others -> -1.
void launch_pack_ranged_routed_pairs(
    const int32_t* d_topk_ids,
    const float* d_topk_weights,
    Mi50RoutedPair* d_out_pairs,
    int M, int top_k,
    int exp_begin, int exp_end,
    hipStream_t stream);

void launch_build_ranged_csr(
    const int32_t* d_topk_ids,
    const float* d_topk_weights,
    int32_t* d_active_experts,
    int32_t* d_csr_offsets,
    Mi50RoutedPair* d_sorted_pairs,
    int M, int top_k,
    int exp_begin, int exp_end,
    hipStream_t stream);

// ---- Zero-PCIe-AtomicOp Coherent Host DDR4 IPC Kernels for hlut3_dev=3 (0000:85:00.0) ----
// Runs on tp=0 router_stream: stages topk + x_rot_f16 (+ optional x_norm_bf16 for MTP shared experts),
// synchronizes CTAs via tp=0 local HBM2 d_local_wg_sync (__HIP_MEMORY_SCOPE_AGENT), issues
// __threadfence_system(), and writes slot->ready_seq = req_seq (__ATOMIC_RELEASE).
void launch_stage_and_arm_hlut3_dev3(
    Hlut3OffloadSlot* d_mapped_slot,
    uint32_t* d_local_wg_sync,
    const int32_t* d_topk_ids,
    const float* d_topk_weights,
    const uint16_t* d_x_rot_f16,
    const uint16_t* d_x_norm_bf16,
    uint32_t block_id,
    int M, int top_k,
    int exp_begin, int exp_end,
    uint32_t include_mtp_shared,
    uint32_t req_seq,
    hipStream_t stream);

// Wave 10 Dual-Switchbox: Single-HBM2-read fused staging & simultaneous doorbell arming for
// BOTH dev3 (0000:85:00.0, Switch #1) and dev4 (0000:08:00.0, Switch #2) on backbone M <= 64!
void launch_stage_and_arm_hlut3_dual(
    Hlut3OffloadSlot* d_mapped_slot3,
    Hlut3OffloadSlot* d_mapped_slot4,
    uint32_t* d_local_wg_sync,
    const int32_t* d_topk_ids,
    const float* d_topk_weights,
    const uint16_t* d_x_rot_f16,
    uint32_t block_id,
    int M, int top_k,
    int exp3_begin, int exp3_end,
    int exp4_begin, int exp4_end,
    uint32_t req_seq3,
    uint32_t req_seq4,
    hipStream_t stream);

// Runs on hlut3_dev=3 (0000:85:00.0): ingests x_rot_f16 (+ optional x_norm_bf16) and packs
// ranged [exp_begin, exp_end) pairs from coherent pinned DDR4 slot into 85:00.0 HBM2.
void launch_dev3_ingest_and_pack(
    const Hlut3OffloadSlot* d_mapped_slot,
    uint16_t* d_x_rot_f16_dst,
    uint16_t* d_x_norm_bf16_dst,
    Mi50RoutedPair* d_pairs_dst,
    int M, int top_k,
    int exp_begin, int exp_end,
    bool copy_norm_bf16,
    hipStream_t stream);

// Runs on hlut3_dev=3 (0000:85:00.0): writes d_out_f32 [M, 5120] to coherent pinned DDR4 slot->y_out_f32,
// synchronizes CTAs via 85:00.0 local HBM2 d_local_wg_sync (__HIP_MEMORY_SCOPE_AGENT), issues
// __threadfence_system(), and writes slot->done_seq = done_seq (__ATOMIC_RELEASE).
void launch_dev3_publish_f32(
    const float* d_out_f32,
    Hlut3OffloadSlot* d_mapped_slot,
    uint32_t* d_local_wg_sync,
    int M,
    uint32_t done_seq,
    hipStream_t stream);

// Wave 9.3 Phase D: Zero-Scratch Single-Launch Coherent Ingest + Expert-Grouped CSR Builder on dev3 (M <= 16)
void launch_dev3_ingest_and_pack_csr(
    const Hlut3OffloadSlot* d_mapped_slot,
    uint16_t* d_x_rot_f16_dst,
    uint16_t* d_x_norm_bf16_dst,
    int32_t* d_active_exp_out,
    int32_t* d_csr_offsets_out,
    Mi50RoutedPair* d_pairs_out,
    int32_t* d_num_act_out,
    int M, int top_k,
    int exp_begin, int exp_end,
    bool copy_norm_bf16,
    hipStream_t stream);

// Wave 9.3 Phase D: Zero-Scratch Grouped-By-Expert v_dot2_f32_f16 (__builtin_amdgcn_fdot2)
// Fused Gate+Up+SwiGLU10+WHT128->FP16 + Down+AtomicAdd for BOTH dev3 (N_inter=2304) and TP3 (N_inter=768)!
void launch_hlut3_fdot2_grouped_f16(
    const uint16_t* d_x_rot_f16,
    const int32_t* d_active_experts,
    const int32_t* d_csr_offsets,
    const Mi50RoutedPair* d_sorted_pairs,
    const int32_t* d_num_act_opt,  // optional device count (nullptr => use max_act with active_experts[ae] < 0 sentinel)
    int max_act,
    int n_experts,
    int n_inter,                   // 2304 (dev3 unsharded) or 768 (TP3 sharded)
    const Hlut3Tile32x32* d_gate_tiles,
    const Hlut3Tile32x32* d_up_tiles,
    const Hlut3Tile32x32* d_down_tiles,
    const float* d_levels,
    const uint64_t* d_signs_i,
    uint16_t* d_inter_rot_f16,     // [P_max, n_inter] FP16 scratch
    float* d_out_accum_f32,        // [M, 5120] local HBM2 accum
    hipStream_t stream);

}  // namespace dsv41
