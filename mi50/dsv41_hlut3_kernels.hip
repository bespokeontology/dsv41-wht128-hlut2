// Wave 9.2: Zero-scratch gfx906 Vega 20 WHT128-HLUT3 kernels (unsharded single-GPU 0000:85:00.0
// + TP3-sharded 32 GiB core + grouped CSR batched prefill + zero-PCIe-AtomicOp coherent host IPC).
// 400 B tile planes are unpacked in registers: q_lo2 2-bit magnitude -> 4-centroid select,
// q_hi1 sign bit XOR, FP16 group RMS scale, then IEEE FP32 FMA accumulation.
// Zero dynamic local arrays, ScratchSize == 0, 0 SGPR/VGPR spills, 0 device transcendentals/trig,
// and 0 __HIP_MEMORY_SCOPE_SYSTEM RMW atomics (__HIP_MEMORY_SCOPE_AGENT on local HBM2 only).

#include "dsv41_hlut3_kernels.hpp"
#include <hip/hip_fp16.h>

namespace dsv41 {
namespace {

using hlut3_f16x2_t = _Float16 __attribute__((ext_vector_type(2)));

// Single-cycle hardware v_cvt_f32_f16_e32 via HIP __ushort_as_half / __half2float.
__device__ __forceinline__ float d_hlut3_f16_bits_to_f32(uint16_t h) noexcept {
    return __half2float(__ushort_as_half(h));
}

__device__ __forceinline__ uint16_t d_hlut3_f32_to_f16_bits(float x) noexcept {
    x = fmaxf(-65504.0f, fminf(65504.0f, x));
    return __half_as_ushort(__float2half_rn(x));
}

__device__ __forceinline__ uint32_t d_hlut3_sel_h16(
    uint32_t c, uint32_t h0, uint32_t h1, uint32_t h2, uint32_t h3) noexcept
{
    const uint32_t lo = (c & 1u) ? h1 : h0;
    const uint32_t hi = (c & 1u) ? h3 : h2;
    return (c & 2u) ? hi : lo;
}

__device__ __forceinline__ float d_hlut3_sum16(float v) noexcept {
    v += __shfl_xor(v, 1, 16);
    v += __shfl_xor(v, 2, 16);
    v += __shfl_xor(v, 4, 16);
    v += __shfl_xor(v, 8, 16);
    return v;
}

__device__ __forceinline__ float d_hlut3_swiglu10(float g, float u) noexcept {
    const float gc = fminf(g, 10.0f);
    const float uc = fmaxf(-10.0f, fminf(10.0f, u));
    return (gc / (1.0f + expf(-gc))) * uc;
}

__device__ __forceinline__ float d_hlut3_tile_weight_reg(
    const Hlut3Tile32x32* __restrict__ tile,
    int local_row,
    int lane,
    int scale_group,
    float lv0, float lv1, float lv2, float lv3) noexcept
{
    const uint32_t mag_word = tile->q_lo2[local_row * 2 + (lane >> 4)];
    const uint32_t mag_code = (mag_word >> (2 * (lane & 15))) & 3u;
    const uint32_t sign_bit = (tile->q_hi1[local_row] >> lane) & 1u;
    const float sigma = d_hlut3_f16_bits_to_f32(tile->scales_f16[scale_group]);
    const float lv = (mag_code == 0u) ? lv0 : ((mag_code == 1u) ? lv1 : ((mag_code == 2u) ? lv2 : lv3));
    const float magnitude = lv * sigma;
    return sign_bit ? -magnitude : magnitude;
}

__device__ __forceinline__ float d_hlut3_tile_weight(
    const Hlut3Tile32x32* __restrict__ tile,
    int local_row,
    int lane,
    int scale_group,
    const float* __restrict__ levels) noexcept
{
    return d_hlut3_tile_weight_reg(tile, local_row, lane, scale_group,
                                   levels[0], levels[1], levels[2], levels[3]);
}

__device__ __forceinline__ float d_hlut3_warp_sum(float v) noexcept {
    #pragma unroll
    for (int len = 1; len <= 16; len <<= 1) {
        v += __shfl_xor(v, len);
    }
    return v;
}

}  // namespace

// ---- Baseline FP32-input GEMV & Pair Kernels ----
__global__ __launch_bounds__(256, 8)
void k_hlut3_gemv_wht128(
    const float* __restrict__ x,
    const Hlut3Tile32x32* __restrict__ tiles,
    const float* __restrict__ levels,
    float* __restrict__ y,
    int N, int K)
{
    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int m = int(blockIdx.y);
    const int n = int(blockIdx.x) * 8 + warp;
    if (n >= N) return;

    const int nt = n >> 5;
    const int local_row = n & 31;
    const int scale_group = local_row >> 2;
    const int kt_count = K >> 5;
    const float* __restrict__ xrow = x + size_t(m) * size_t(K);
    const Hlut3Tile32x32* __restrict__ tile_row = tiles + size_t(nt) * size_t(kt_count);

    float acc = 0.0f;
    for (int kt = 0; kt < kt_count; ++kt) {
        const Hlut3Tile32x32* __restrict__ tile = tile_row + kt;
        const float w = d_hlut3_tile_weight(tile, local_row, lane, scale_group, levels);
        acc = fmaf(xrow[kt * 32 + lane], w, acc);
    }
    acc = d_hlut3_warp_sum(acc);
    if (lane == 0) y[size_t(m) * size_t(N) + size_t(n)] = acc;
}

__global__ __launch_bounds__(256, 4)
void k_hlut3_pair_gateup_swiglu(
    const float* __restrict__ x,
    const Mi50RoutedPair* __restrict__ pairs,
    const Hlut3Tile32x32* __restrict__ gate_tiles,
    const Hlut3Tile32x32* __restrict__ up_tiles,
    const float* __restrict__ gate_levels,
    const float* __restrict__ up_levels,
    float* __restrict__ z_out,
    int n_experts, int N, int K)
{
    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int p_idx = int(blockIdx.y);
    const Mi50RoutedPair pr = pairs[p_idx];
    if (pr.expert < 0 || pr.expert >= n_experts || pr.weight == 0.0f) return;

    const int n = int(blockIdx.x) * 8 + warp;
    if (n >= N) return;
    const int nt = n >> 5;
    const int local_row = n & 31;
    const int scale_group = local_row >> 2;
    const int nt_count = N >> 5;
    const int kt_count = K >> 5;
    const size_t tile_base = size_t(pr.expert) * size_t(nt_count) * size_t(kt_count);
    const float* __restrict__ xrow = x + size_t(pr.row) * size_t(K);
    const Hlut3Tile32x32* __restrict__ gtile_row = gate_tiles + tile_base + size_t(nt) * size_t(kt_count);
    const Hlut3Tile32x32* __restrict__ utile_row = up_tiles   + tile_base + size_t(nt) * size_t(kt_count);

    float acc_g = 0.0f, acc_u = 0.0f;
    for (int kt = 0; kt < kt_count; ++kt) {
        const float xv = xrow[kt * 32 + lane];
        const Hlut3Tile32x32* gtile = gtile_row + kt;
        const Hlut3Tile32x32* utile = utile_row + kt;
        const float wg = d_hlut3_tile_weight(gtile, local_row, lane, scale_group, gate_levels);
        const float wu = d_hlut3_tile_weight(utile, local_row, lane, scale_group, up_levels);
        acc_g = fmaf(xv, wg, acc_g);
        acc_u = fmaf(xv, wu, acc_u);
    }
    acc_g = d_hlut3_warp_sum(acc_g);
    acc_u = d_hlut3_warp_sum(acc_u);
    if (lane == 0) {
        z_out[size_t(p_idx) * size_t(N) + size_t(n)] = d_hlut3_swiglu10(acc_g, acc_u);
    }
}

__global__ __launch_bounds__(128, 8)
void k_hlut3_wht128_z_f32(
    const float* __restrict__ z,
    const uint64_t* __restrict__ signs_i,
    float* __restrict__ z_rot,
    int dims)
{
    __shared__ float s_xwave[128];
    const int blk = int(blockIdx.x);
    const int p = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int lane = tid & 63;
    const int wid = tid >> 6;

    const size_t base = size_t(p) * size_t(dims) + size_t(blk) * 128u;
    float v = z[base + tid];
    if ((signs_i[blk * 2 + wid] >> lane) & 1ULL) v = -v;

    #pragma unroll
    for (int len = 1; len <= 32; len <<= 1) {
        const float partner = __shfl_xor(v, len);
        v = (lane & len) ? (partner - v) : (v + partner);
    }
    s_xwave[tid] = v;
    __syncthreads();
    const float a = s_xwave[lane];
    const float b = s_xwave[64 + lane];
    const float res = ((wid == 0) ? (a + b) : (a - b)) * 0.08838834764831844655f;
    z_rot[base + tid] = res;
}

__global__ __launch_bounds__(128, 8)
void k_hlut3_wht128_z_f32_gated(
    const float* __restrict__ z,
    const Mi50RoutedPair* __restrict__ pairs,
    const uint64_t* __restrict__ signs_i,
    float* __restrict__ z_rot,
    int n_experts, int dims)
{
    __shared__ float s_xwave[128];
    const int p = int(blockIdx.y);
    const Mi50RoutedPair pr = pairs[p];
    if (pr.expert < 0 || pr.expert >= n_experts || pr.weight == 0.0f) return;

    const int blk = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int lane = tid & 63;
    const int wid = tid >> 6;

    const size_t base = size_t(p) * size_t(dims) + size_t(blk) * 128u;
    float v = z[base + tid];
    if ((signs_i[blk * 2 + wid] >> lane) & 1ULL) v = -v;

    #pragma unroll
    for (int len = 1; len <= 32; len <<= 1) {
        const float partner = __shfl_xor(v, len);
        v = (lane & len) ? (partner - v) : (v + partner);
    }
    s_xwave[tid] = v;
    __syncthreads();
    const float a = s_xwave[lane];
    const float b = s_xwave[64 + lane];
    const float res = ((wid == 0) ? (a + b) : (a - b)) * 0.08838834764831844655f;
    z_rot[base + tid] = res;
}

__global__ __launch_bounds__(256, 8)
void k_hlut3_pair_down_atomic(
    const float* __restrict__ z_rot,
    const Mi50RoutedPair* __restrict__ pairs,
    const Hlut3Tile32x32* __restrict__ down_tiles,
    const float* __restrict__ down_levels,
    float* __restrict__ out_accum_f32,
    int n_experts, int N_down, int K)
{
    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int p_idx = int(blockIdx.y);
    const Mi50RoutedPair pr = pairs[p_idx];
    if (pr.expert < 0 || pr.expert >= n_experts || pr.weight == 0.0f) return;

    const int n = int(blockIdx.x) * 8 + warp;
    if (n >= N_down) return;
    const int nt = n >> 5;
    const int local_row = n & 31;
    const int scale_group = local_row >> 2;
    const int nt_count = N_down >> 5;
    const int kt_count = K >> 5;
    const size_t tile_base = size_t(pr.expert) * size_t(nt_count) * size_t(kt_count);
    const float* __restrict__ zrow = z_rot + size_t(p_idx) * size_t(K);
    const Hlut3Tile32x32* __restrict__ tile_row = down_tiles + tile_base + size_t(nt) * size_t(kt_count);

    float acc = 0.0f;
    for (int kt = 0; kt < kt_count; ++kt) {
        const Hlut3Tile32x32* __restrict__ tile = tile_row + kt;
        const float w = d_hlut3_tile_weight(tile, local_row, lane, scale_group, down_levels);
        acc = fmaf(zrow[kt * 32 + lane], w, acc);
    }
    acc = d_hlut3_warp_sum(acc);
    if (lane == 0) {
        atomicAdd(&out_accum_f32[size_t(pr.row) * size_t(N_down) + size_t(n)], acc * pr.weight);
    }
}

// ---- Wave 9.2 Unsharded (2304x5120) & TP3-Sharded (768x5120) Direct-FP16 Pair Kernels ----
template <int kN, int kK>
__device__ __forceinline__ void d_hlut3_fixed_gateup_swiglu_f16(
    const uint16_t* __restrict__ x_rot_f16,
    const Mi50RoutedPair* __restrict__ pairs,
    const Hlut3Tile32x32* __restrict__ gate_tiles,
    const Hlut3Tile32x32* __restrict__ up_tiles,
    const float* __restrict__ levels,
    float* __restrict__ z_out,
    int n_experts)
{
    const int p_idx = int(blockIdx.y);
    const Mi50RoutedPair pr = pairs[p_idx];
    if (pr.expert < 0 || pr.expert >= n_experts || pr.weight == 0.0f) return;

    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int n = int(blockIdx.x) * 8 + warp;
    if (n >= kN) return;

    constexpr int kNtCount = kN >> 5;
    constexpr int kKtCount = kK >> 5;
    const int nt = n >> 5;
    const int local_row = n & 31;
    const int scale_group = local_row >> 2;
    const size_t tile_base = size_t(pr.expert) * size_t(kNtCount) * size_t(kKtCount);
    const uint16_t* __restrict__ xrow = x_rot_f16 + size_t(pr.row) * size_t(kK);
    const Hlut3Tile32x32* __restrict__ gtile_row = gate_tiles + tile_base + size_t(nt) * size_t(kKtCount);
    const Hlut3Tile32x32* __restrict__ utile_row = up_tiles   + tile_base + size_t(nt) * size_t(kKtCount);

    const float lv0 = levels[0], lv1 = levels[1], lv2 = levels[2], lv3 = levels[3];
    float acc_g = 0.0f, acc_u = 0.0f;
    for (int kt = 0; kt < kKtCount; ++kt) {
        const float xv = d_hlut3_f16_bits_to_f32(xrow[kt * 32 + lane]);
        const float wg = d_hlut3_tile_weight_reg(gtile_row + kt, local_row, lane, scale_group, lv0, lv1, lv2, lv3);
        const float wu = d_hlut3_tile_weight_reg(utile_row + kt, local_row, lane, scale_group, lv0, lv1, lv2, lv3);
        acc_g = fmaf(xv, wg, acc_g);
        acc_u = fmaf(xv, wu, acc_u);
    }
    acc_g = d_hlut3_warp_sum(acc_g);
    acc_u = d_hlut3_warp_sum(acc_u);
    if (lane == 0) {
        z_out[size_t(p_idx) * size_t(kN) + size_t(n)] = d_hlut3_swiglu10(acc_g, acc_u);
    }
}

template <int kNDown, int kKDown>
__device__ __forceinline__ void d_hlut3_fixed_down_atomic(
    const float* __restrict__ z_rot,
    const Mi50RoutedPair* __restrict__ pairs,
    const Hlut3Tile32x32* __restrict__ down_tiles,
    const float* __restrict__ levels,
    float* __restrict__ out_accum_f32,
    int n_experts)
{
    const int p_idx = int(blockIdx.y);
    const Mi50RoutedPair pr = pairs[p_idx];
    if (pr.expert < 0 || pr.expert >= n_experts || pr.weight == 0.0f) return;

    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int n = int(blockIdx.x) * 8 + warp;
    if (n >= kNDown) return;

    constexpr int kNtCount = kNDown >> 5;
    constexpr int kKtCount = kKDown >> 5;
    const int nt = n >> 5;
    const int local_row = n & 31;
    const int scale_group = local_row >> 2;
    const size_t tile_base = size_t(pr.expert) * size_t(kNtCount) * size_t(kKtCount);
    const float* __restrict__ zrow = z_rot + size_t(p_idx) * size_t(kKDown);
    const Hlut3Tile32x32* __restrict__ tile_row = down_tiles + tile_base + size_t(nt) * size_t(kKtCount);

    const float lv0 = levels[0], lv1 = levels[1], lv2 = levels[2], lv3 = levels[3];
    float acc = 0.0f;
    for (int kt = 0; kt < kKtCount; ++kt) {
        const float w = d_hlut3_tile_weight_reg(tile_row + kt, local_row, lane, scale_group, lv0, lv1, lv2, lv3);
        acc = fmaf(zrow[kt * 32 + lane], w, acc);
    }
    acc = d_hlut3_warp_sum(acc);
    if (lane == 0) {
        atomicAdd(&out_accum_f32[size_t(pr.row) * size_t(kNDown) + size_t(n)], acc * pr.weight);
    }
}

__global__ __launch_bounds__(256, 4)
void k_hlut3_unsharded_gate_up_swiglu(
    const uint16_t* __restrict__ x_rot_f16,
    const Mi50RoutedPair* __restrict__ pairs,
    const Hlut3Tile32x32* __restrict__ gate_tiles,
    const Hlut3Tile32x32* __restrict__ up_tiles,
    const float* __restrict__ levels,
    float* __restrict__ z_out,
    int n_experts)
{
    d_hlut3_fixed_gateup_swiglu_f16<2304, 5120>(x_rot_f16, pairs, gate_tiles, up_tiles, levels, z_out, n_experts);
}

__global__ __launch_bounds__(256, 8)
void k_hlut3_unsharded_down(
    const float* __restrict__ z_rot,
    const Mi50RoutedPair* __restrict__ pairs,
    const Hlut3Tile32x32* __restrict__ down_tiles,
    const float* __restrict__ levels,
    float* __restrict__ out_accum_f32,
    int n_experts)
{
    d_hlut3_fixed_down_atomic<5120, 2304>(z_rot, pairs, down_tiles, levels, out_accum_f32, n_experts);
}

__global__ __launch_bounds__(256, 4)
void k_hlut3_tp3_gate_up_swiglu(
    const uint16_t* __restrict__ x_rot_f16,
    const Mi50RoutedPair* __restrict__ pairs,
    const Hlut3Tile32x32* __restrict__ gate_tiles,
    const Hlut3Tile32x32* __restrict__ up_tiles,
    const float* __restrict__ levels,
    float* __restrict__ z_out,
    int n_experts)
{
    d_hlut3_fixed_gateup_swiglu_f16<768, 5120>(x_rot_f16, pairs, gate_tiles, up_tiles, levels, z_out, n_experts);
}

__global__ __launch_bounds__(256, 8)
void k_hlut3_tp3_down(
    const float* __restrict__ z_rot,
    const Mi50RoutedPair* __restrict__ pairs,
    const Hlut3Tile32x32* __restrict__ down_tiles,
    const float* __restrict__ levels,
    float* __restrict__ out_accum_f32,
    int n_experts)
{
    d_hlut3_fixed_down_atomic<5120, 768>(z_rot, pairs, down_tiles, levels, out_accum_f32, n_experts);
}

// ---- Wave 9.2 Token-Grouped 8-Row LDS Tile-Reuse Kernels (M > 16 / Batched Prefill) ----
__global__ __launch_bounds__(256, 4)
void k_hlut3_grouped_gate_up_swiglu_f16(
    const uint16_t* __restrict__ x_rot_f16,
    const int32_t* __restrict__ active_experts,
    const int32_t* __restrict__ csr_offsets,
    const Mi50RoutedPair* __restrict__ sorted_pairs,
    const Hlut3Tile32x32* __restrict__ gate_tiles,
    const Hlut3Tile32x32* __restrict__ up_tiles,
    const float* __restrict__ levels,
    float* __restrict__ z_out,
    int n_experts, int N_gate, int K_gate)
{
    __shared__ float s_x[8][32];
    const int ae = int(blockIdx.y);
    const int exp_id = active_experts[ae];
    if (exp_id < 0 || exp_id >= n_experts) return;
    const int p_start = csr_offsets[ae];
    const int p_end   = csr_offsets[ae + 1];
    if (p_start >= p_end) return;

    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int n = int(blockIdx.x) * 8 + warp;
    const bool valid_n = (n < N_gate);
    const int nt = n >> 5;
    const int local_row = n & 31;
    const int scale_group = local_row >> 2;
    const int nt_count = N_gate >> 5;
    const int kt_count = K_gate >> 5;
    const size_t tile_base = size_t(exp_id) * size_t(nt_count) * size_t(kt_count);
    const Hlut3Tile32x32* __restrict__ gtile_row = gate_tiles + tile_base + size_t(nt) * size_t(kt_count);
    const Hlut3Tile32x32* __restrict__ utile_row = up_tiles   + tile_base + size_t(nt) * size_t(kt_count);

    const float lv0 = levels[0], lv1 = levels[1], lv2 = levels[2], lv3 = levels[3];
    for (int p_base = p_start; p_base < p_end; p_base += 8) {
        const int B = min(8, p_end - p_base);
        float acc_g[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float acc_u[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const int load_row = (warp < B) ? sorted_pairs[p_base + warp].row : 0;

        for (int kt = 0; kt < kt_count; ++kt) {
            s_x[warp][lane] = (warp < B)
                ? d_hlut3_f16_bits_to_f32(x_rot_f16[size_t(load_row) * size_t(K_gate) + size_t(kt * 32 + lane)])
                : 0.0f;
            float wg = 0.0f, wu = 0.0f;
            if (valid_n) {
                wg = d_hlut3_tile_weight_reg(gtile_row + kt, local_row, lane, scale_group, lv0, lv1, lv2, lv3);
                wu = d_hlut3_tile_weight_reg(utile_row + kt, local_row, lane, scale_group, lv0, lv1, lv2, lv3);
            }
            __syncthreads();
            if (valid_n) {
                #pragma unroll
                for (int b = 0; b < 8; ++b) {
                    const float xv = s_x[b][lane];
                    acc_g[b] = fmaf(xv, wg, acc_g[b]);
                    acc_u[b] = fmaf(xv, wu, acc_u[b]);
                }
            }
            __syncthreads();
        }
        if (valid_n) {
            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                if (b < B) {
                    const float sg = d_hlut3_warp_sum(acc_g[b]);
                    const float su = d_hlut3_warp_sum(acc_u[b]);
                    if (lane == 0) {
                        z_out[size_t(p_base + b) * size_t(N_gate) + size_t(n)] = d_hlut3_swiglu10(sg, su);
                    }
                }
            }
        }
    }
}

__global__ __launch_bounds__(256, 4)
void k_hlut3_grouped_down(
    const float* __restrict__ z_rot,
    const int32_t* __restrict__ active_experts,
    const int32_t* __restrict__ csr_offsets,
    const Mi50RoutedPair* __restrict__ sorted_pairs,
    const Hlut3Tile32x32* __restrict__ down_tiles,
    const float* __restrict__ levels,
    float* __restrict__ out_accum_f32,
    int n_experts, int N_down, int K_down)
{
    __shared__ float s_z[8][32];
    const int ae = int(blockIdx.y);
    const int exp_id = active_experts[ae];
    if (exp_id < 0 || exp_id >= n_experts) return;
    const int p_start = csr_offsets[ae];
    const int p_end   = csr_offsets[ae + 1];
    if (p_start >= p_end) return;

    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int n = int(blockIdx.x) * 8 + warp;
    const bool valid_n = (n < N_down);
    const int nt = n >> 5;
    const int local_row = n & 31;
    const int scale_group = local_row >> 2;
    const int nt_count = N_down >> 5;
    const int kt_count = K_down >> 5;
    const size_t tile_base = size_t(exp_id) * size_t(nt_count) * size_t(kt_count);
    const Hlut3Tile32x32* __restrict__ dtile_row = down_tiles + tile_base + size_t(nt) * size_t(kt_count);

    const float lv0 = levels[0], lv1 = levels[1], lv2 = levels[2], lv3 = levels[3];
    for (int p_base = p_start; p_base < p_end; p_base += 8) {
        const int B = min(8, p_end - p_base);
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        for (int kt = 0; kt < kt_count; ++kt) {
            s_z[warp][lane] = (warp < B)
                ? z_rot[size_t(p_base + warp) * size_t(K_down) + size_t(kt * 32 + lane)]
                : 0.0f;
            float wd = 0.0f;
            if (valid_n) {
                wd = d_hlut3_tile_weight_reg(dtile_row + kt, local_row, lane, scale_group, lv0, lv1, lv2, lv3);
            }
            __syncthreads();
            if (valid_n) {
                #pragma unroll
                for (int b = 0; b < 8; ++b) {
                    acc[b] = fmaf(s_z[b][lane], wd, acc[b]);
                }
            }
            __syncthreads();
        }
        if (valid_n) {
            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                if (b < B) {
                    const float sd = d_hlut3_warp_sum(acc[b]);
                    if (lane == 0) {
                        const Mi50RoutedPair pr = sorted_pairs[p_base + b];
                        if (pr.weight != 0.0f) {
                            atomicAdd(&out_accum_f32[size_t(pr.row) * size_t(N_down) + size_t(n)], sd * pr.weight);
                        }
                    }
                }
            }
        }
    }
}

// ---- Range-Filtered Pair Packer & Single-CTA CSR Builder ----
__global__ __launch_bounds__(256, 8)
void k_pack_ranged_routed_pairs(
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_w,
    Mi50RoutedPair* __restrict__ out_pairs,
    int total_pairs, int top_k, int exp_begin, int exp_end)
{
    const int idx = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (idx >= total_pairs) return;
    const int32_t eid = topk_ids[idx];
    const float   w   = topk_w[idx];
    Mi50RoutedPair p;
    p.row = idx / top_k;
    if (eid >= exp_begin && eid < exp_end && w != 0.0f) {
        p.expert = eid - exp_begin;
        p.weight = w;
    } else {
        p.expert = -1;
        p.weight = 0.0f;
    }
    p.pad = 0;
    out_pairs[idx] = p;
}

__global__ __launch_bounds__(256, 4)
void k_build_ranged_csr(
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_w,
    int32_t* __restrict__ active_experts,
    int32_t* __restrict__ csr_offsets,
    Mi50RoutedPair* __restrict__ sorted_pairs,
    int total_pairs, int top_k, int exp_begin, int exp_end)
{
    __shared__ int32_t s_cnt[256];
    const int tid = int(threadIdx.x);
    const int n_exp = exp_end - exp_begin;
    s_cnt[tid] = 0;
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < total_pairs; ++i) {
            const int e = topk_ids[i] - exp_begin;
            if (e >= 0 && e < n_exp && topk_w[i] != 0.0f) {
                ++s_cnt[e];
            }
        }
        int num_act = 0;
        int off = 0;
        for (int e = 0; e < n_exp; ++e) {
            const int c = s_cnt[e];
            if (c > 0) {
                active_experts[num_act] = e;
                csr_offsets[num_act] = off;
                s_cnt[e] = off;
                off += c;
                ++num_act;
            }
        }
        csr_offsets[num_act] = off;
        for (int e = num_act; e < n_exp; ++e) {
            active_experts[e] = -1;
            csr_offsets[e + 1] = off;
        }
        for (int i = 0; i < total_pairs; ++i) {
            const int e = topk_ids[i] - exp_begin;
            const float w = topk_w[i];
            if (e >= 0 && e < n_exp && w != 0.0f) {
                const int dst = s_cnt[e]++;
                Mi50RoutedPair p;
                p.row    = i / top_k;
                p.expert = e;
                p.weight = w;
                p.pad    = 0;
                sorted_pairs[dst] = p;
            }
        }
        for (int i = off; i < total_pairs; ++i) {
            Mi50RoutedPair p;
            p.row    = 0;
            p.expert = -1;
            p.weight = 0.0f;
            p.pad    = 0;
            sorted_pairs[i] = p;
        }
    }
}

// ---- Zero-PCIe-AtomicOp Coherent Host DDR4 IPC Kernels for hlut3_dev=3 (0000:85:00.0) ----
__global__ __launch_bounds__(256, 4)
void k_stage_and_arm_hlut3_dev3(
    Hlut3OffloadSlot* __restrict__ slot,
    uint32_t* __restrict__ d_local_wg_sync,
    const int32_t* __restrict__ d_topk_ids,
    const float* __restrict__ d_topk_weights,
    const uint16_t* __restrict__ d_x_rot_f16,
    const uint16_t* __restrict__ d_x_norm_bf16,
    uint32_t block_id,
    int M, int top_k,
    int exp_begin, int exp_end,
    uint32_t include_mtp_shared,
    uint32_t req_seq)
{
    const int tid = int(threadIdx.x);
    const int gid = int(blockIdx.x) * 256 + tid;
    const int stride = int(gridDim.x) * 256;

    // Backpressure: ensure slot from req_seq - kHlut3OffloadRingSlots was retired before overwriting.
    if (req_seq > kHlut3OffloadRingSlots && tid == 0) {
        const uint32_t min_done = req_seq - uint32_t(kHlut3OffloadRingSlots);
        while (static_cast<int32_t>(__hip_atomic_load(&slot->done_seq, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) - min_done) < 0 ||
               (include_mtp_shared != 0u &&
                (static_cast<int32_t>(__hip_atomic_load(&slot->rank_done_seq[0][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) - min_done) < 0 ||
                 static_cast<int32_t>(__hip_atomic_load(&slot->rank_done_seq[1][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) - min_done) < 0 ||
                 static_cast<int32_t>(__hip_atomic_load(&slot->rank_done_seq[2][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) - min_done) < 0))) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();

    const int total_pairs = M * top_k;
    for (int i = gid; i < total_pairs; i += stride) {
        __hip_atomic_store(&slot->topk_ids[i], d_topk_ids[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(reinterpret_cast<uint32_t*>(&slot->topk_weights[i]),
                           __float_as_uint(d_topk_weights[i]), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }

    const int n_words = (M * kHlut3OffloadHidden) >> 1;
    const uint32_t* src_rot_u32 = reinterpret_cast<const uint32_t*>(d_x_rot_f16);
    uint32_t* dst_rot_u32 = reinterpret_cast<uint32_t*>(slot->x_rot_f16);
    for (int w = gid; w < n_words; w += stride) {
        __hip_atomic_store(&dst_rot_u32[w], src_rot_u32[w], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }

    if (include_mtp_shared != 0u && d_x_norm_bf16 != nullptr) {
        const uint32_t* src_norm_u32 = reinterpret_cast<const uint32_t*>(d_x_norm_bf16);
        uint32_t* dst_norm_u32 = reinterpret_cast<uint32_t*>(slot->x_norm_bf16);
        for (int w = gid; w < n_words; w += stride) {
            __hip_atomic_store(&dst_norm_u32[w], src_norm_u32[w], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }

    if (gid == 0) {
        __hip_atomic_store(&slot->block_id, block_id, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(&slot->rows, uint32_t(M), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(&slot->top_k, uint32_t(top_k), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(reinterpret_cast<uint32_t*>(&slot->exp_begin), uint32_t(exp_begin), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(reinterpret_cast<uint32_t*>(&slot->exp_end), uint32_t(exp_end), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(&slot->include_mtp_shared, include_mtp_shared, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence_system();
        const uint32_t prev = __hip_atomic_fetch_add(d_local_wg_sync, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (prev + 1u == uint32_t(gridDim.x)) {
            __hip_atomic_store(d_local_wg_sync, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __threadfence_system();
            __hip_atomic_store(&slot->ready_seq, req_seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

__global__ __launch_bounds__(256, 8)
void k_dev3_ingest_and_pack(
    const Hlut3OffloadSlot* __restrict__ slot,
    uint16_t* __restrict__ d_x_rot_f16_dst,
    uint16_t* __restrict__ d_x_norm_bf16_dst,
    Mi50RoutedPair* __restrict__ d_pairs_dst,
    int M, int top_k,
    int exp_begin, int exp_end,
    int copy_norm_bf16)
{
    const int gid = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int stride = int(gridDim.x) * 256;

    const int n_words = (M * kHlut3OffloadHidden) >> 1;
    const uint32_t* src_rot_u32 = reinterpret_cast<const uint32_t*>(slot->x_rot_f16);
    uint32_t* dst_rot_u32 = reinterpret_cast<uint32_t*>(d_x_rot_f16_dst);
    for (int w = gid; w < n_words; w += stride) {
        dst_rot_u32[w] = __hip_atomic_load(&src_rot_u32[w], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    if (copy_norm_bf16 != 0 && d_x_norm_bf16_dst != nullptr) {
        const uint32_t* src_norm_u32 = reinterpret_cast<const uint32_t*>(slot->x_norm_bf16);
        uint32_t* dst_norm_u32 = reinterpret_cast<uint32_t*>(d_x_norm_bf16_dst);
        for (int w = gid; w < n_words; w += stride) {
            dst_norm_u32[w] = __hip_atomic_load(&src_norm_u32[w], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }

    const int total_pairs = M * top_k;
    for (int i = gid; i < total_pairs; i += stride) {
        const int32_t eid = __hip_atomic_load(&slot->topk_ids[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        const uint32_t wu = __hip_atomic_load(reinterpret_cast<const uint32_t*>(&slot->topk_weights[i]),
                                              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        const float w = __uint_as_float(wu);
        Mi50RoutedPair p;
        p.row = i / top_k;
        if (eid >= exp_begin && eid < exp_end && w != 0.0f) {
            p.expert = eid - exp_begin;
            p.weight = w;
        } else {
            p.expert = -1;
            p.weight = 0.0f;
        }
        p.pad = 0;
        d_pairs_dst[i] = p;
    }
}

__global__ __launch_bounds__(256, 4)
void k_dev3_publish_f32(
    const float* __restrict__ d_out_f32,
    Hlut3OffloadSlot* __restrict__ slot,
    uint32_t* __restrict__ d_local_wg_sync,
    int n_elems,
    uint32_t done_seq)
{
    const int tid = int(threadIdx.x);
    const int gid = int(blockIdx.x) * 256 + tid;
    const int stride = int(gridDim.x) * 256;

    uint32_t* dst_u32 = reinterpret_cast<uint32_t*>(slot->y_out_f32);
    for (int i = gid; i < n_elems; i += stride) {
        __hip_atomic_store(&dst_u32[i], __float_as_uint(d_out_f32[i]), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence_system();
        const uint32_t prev = __hip_atomic_fetch_add(d_local_wg_sync, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (prev + 1u == uint32_t(gridDim.x)) {
            __hip_atomic_store(d_local_wg_sync, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __threadfence_system();
            __hip_atomic_store(&slot->done_seq, done_seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

__global__ __launch_bounds__(256, 8)
void k_hlut3_f16_to_f32(
    const uint16_t* __restrict__ in_f16,
    float* __restrict__ out_f32,
    int n)
{
    const int idx = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (idx < n) out_f32[idx] = d_hlut3_f16_bits_to_f32(in_f16[idx]);
}

// ---- Host Launchers ----
void launch_hlut3_gemv_wht128(
    const float* d_x,
    const Hlut3Tile32x32* d_tiles,
    const float* d_levels,
    float* d_y,
    int M, int N, int K,
    hipStream_t stream)
{
    if (M <= 0 || N <= 0 || K <= 0 || (N % 8) != 0 || (K % 32) != 0) return;
    hipLaunchKernelGGL(k_hlut3_gemv_wht128, dim3(unsigned(N / 8), unsigned(M)), dim3(256), 0, stream,
                       d_x, d_tiles, d_levels, d_y, N, K);
}

void launch_mi50_routed_hlut3_pairs(
    const float* d_x,
    const Mi50RoutedPair* d_pairs,
    int P, int M, int n_experts,
    int n_gate, int k_gate, int n_down, int k_down,
    const Hlut3Tile32x32* d_gate_tiles,
    const Hlut3Tile32x32* d_up_tiles,
    const Hlut3Tile32x32* d_down_tiles,
    const float* d_gate_levels,
    const float* d_up_levels,
    const float* d_down_levels,
    const uint64_t* d_signs_i,
    float* d_inter_z_f32,
    float* d_inter_zrot_f32,
    float* d_out_accum_f32,
    hipStream_t stream)
{
    (void)M;
    if (P <= 0 || n_experts <= 0 || n_gate <= 0 || k_gate <= 0 ||
        n_down <= 0 || k_down <= 0 ||
        (n_gate % 128) != 0 || (k_gate % 32) != 0 ||
        (n_down % 8) != 0 || (k_down % 32) != 0) {
        return;
    }
    hipLaunchKernelGGL(k_hlut3_pair_gateup_swiglu,
                       dim3(unsigned(n_gate / 8), unsigned(P)), dim3(256), 0, stream,
                       d_x, d_pairs, d_gate_tiles, d_up_tiles,
                       d_gate_levels, d_up_levels, d_inter_z_f32, n_experts, n_gate, k_gate);
    hipLaunchKernelGGL(k_hlut3_wht128_z_f32_gated,
                       dim3(unsigned(n_gate / 128), unsigned(P)), dim3(128), 0, stream,
                       d_inter_z_f32, d_pairs, d_signs_i, d_inter_zrot_f32, n_experts, n_gate);
    hipLaunchKernelGGL(k_hlut3_pair_down_atomic,
                       dim3(unsigned(n_down / 8), unsigned(P)), dim3(256), 0, stream,
                       d_inter_zrot_f32, d_pairs, d_down_tiles, d_down_levels,
                       d_out_accum_f32, n_experts, n_down, k_down);
}

void launch_hlut3_f16_to_f32(
    const uint16_t* d_in_f16,
    float* d_out_f32,
    int n_elems,
    hipStream_t stream)
{
    if (n_elems <= 0) return;
    hipLaunchKernelGGL(k_hlut3_f16_to_f32, dim3(unsigned((n_elems + 255) / 256)), dim3(256), 0, stream,
                       d_in_f16, d_out_f32, n_elems);
}

void launch_hlut3_unsharded_pairs_f16(
    const uint16_t* d_x_rot_f16,
    const Mi50RoutedPair* d_pairs,
    int P, int max_scratch_pairs, int n_experts,
    const Hlut3Tile32x32* d_gate_tiles,
    const Hlut3Tile32x32* d_up_tiles,
    const Hlut3Tile32x32* d_down_tiles,
    const float* d_levels,
    const uint64_t* d_signs_i,
    float* d_inter_z_f32,
    float* d_inter_zrot_f32,
    float* d_out_accum_f32,
    hipStream_t stream)
{
    if (P <= 0 || max_scratch_pairs <= 0 || n_experts <= 0) return;
    for (int off = 0; off < P; off += max_scratch_pairs) {
        const int cur_p = (P - off < max_scratch_pairs) ? (P - off) : max_scratch_pairs;
        const Mi50RoutedPair* p_sub = d_pairs + off;
        hipLaunchKernelGGL(k_hlut3_unsharded_gate_up_swiglu,
                           dim3(unsigned(2304 / 8), unsigned(cur_p)), dim3(256), 0, stream,
                           d_x_rot_f16, p_sub, d_gate_tiles, d_up_tiles, d_levels, d_inter_z_f32, n_experts);
        hipLaunchKernelGGL(k_hlut3_wht128_z_f32_gated,
                           dim3(unsigned(2304 / 128), unsigned(cur_p)), dim3(128), 0, stream,
                           d_inter_z_f32, p_sub, d_signs_i, d_inter_zrot_f32, n_experts, 2304);
        hipLaunchKernelGGL(k_hlut3_unsharded_down,
                           dim3(unsigned(5120 / 8), unsigned(cur_p)), dim3(256), 0, stream,
                           d_inter_zrot_f32, p_sub, d_down_tiles, d_levels, d_out_accum_f32, n_experts);
    }
}

void launch_hlut3_tp3_pairs_f16(
    const uint16_t* d_x_rot_f16,
    const Mi50RoutedPair* d_pairs,
    int P, int max_scratch_pairs, int n_experts,
    const Hlut3Tile32x32* d_gate_tiles,
    const Hlut3Tile32x32* d_up_tiles,
    const Hlut3Tile32x32* d_down_tiles,
    const float* d_levels,
    const uint64_t* d_signs_i,
    float* d_inter_z_f32,
    float* d_inter_zrot_f32,
    float* d_out_accum_f32,
    hipStream_t stream)
{
    if (P <= 0 || max_scratch_pairs <= 0 || n_experts <= 0) return;
    for (int off = 0; off < P; off += max_scratch_pairs) {
        const int cur_p = (P - off < max_scratch_pairs) ? (P - off) : max_scratch_pairs;
        const Mi50RoutedPair* p_sub = d_pairs + off;
        hipLaunchKernelGGL(k_hlut3_tp3_gate_up_swiglu,
                           dim3(unsigned(768 / 8), unsigned(cur_p)), dim3(256), 0, stream,
                           d_x_rot_f16, p_sub, d_gate_tiles, d_up_tiles, d_levels, d_inter_z_f32, n_experts);
        hipLaunchKernelGGL(k_hlut3_wht128_z_f32_gated,
                           dim3(unsigned(768 / 128), unsigned(cur_p)), dim3(128), 0, stream,
                           d_inter_z_f32, p_sub, d_signs_i, d_inter_zrot_f32, n_experts, 768);
        hipLaunchKernelGGL(k_hlut3_tp3_down,
                           dim3(unsigned(5120 / 8), unsigned(cur_p)), dim3(256), 0, stream,
                           d_inter_zrot_f32, p_sub, d_down_tiles, d_levels, d_out_accum_f32, n_experts);
    }
}

void launch_hlut3_routed_grouped_f16(
    const uint16_t* d_x_rot_f16,
    const int32_t* d_active_experts,
    const int32_t* d_csr_offsets,
    const Mi50RoutedPair* d_sorted_pairs,
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
    hipStream_t stream)
{
    if (num_act <= 0 || batch_pairs <= 0 || n_experts <= 0) return;
    hipLaunchKernelGGL(k_hlut3_grouped_gate_up_swiglu_f16,
                       dim3(unsigned(n_gate / 8), unsigned(num_act)), dim3(256), 0, stream,
                       d_x_rot_f16, d_active_experts, d_csr_offsets, d_sorted_pairs,
                       d_gate_tiles, d_up_tiles, d_levels, d_inter_z_f32,
                       n_experts, n_gate, k_gate);
    hipLaunchKernelGGL(k_hlut3_wht128_z_f32_gated,
                       dim3(unsigned(n_gate / 128), unsigned(batch_pairs)), dim3(128), 0, stream,
                       d_inter_z_f32, d_sorted_pairs, d_signs_i, d_inter_zrot_f32, n_experts, n_gate);
    hipLaunchKernelGGL(k_hlut3_grouped_down,
                       dim3(unsigned(n_down / 8), unsigned(num_act)), dim3(256), 0, stream,
                       d_inter_zrot_f32, d_active_experts, d_csr_offsets, d_sorted_pairs,
                       d_down_tiles, d_levels, d_out_accum_f32,
                       n_experts, n_down, k_down);
}

void launch_pack_ranged_routed_pairs(
    const int32_t* d_topk_ids,
    const float* d_topk_weights,
    Mi50RoutedPair* d_out_pairs,
    int M, int top_k,
    int exp_begin, int exp_end,
    hipStream_t stream)
{
    const int total = M * top_k;
    if (total <= 0 || exp_end <= exp_begin) return;
    hipLaunchKernelGGL(k_pack_ranged_routed_pairs,
                       dim3(unsigned((total + 255) / 256)), dim3(256), 0, stream,
                       d_topk_ids, d_topk_weights, d_out_pairs, total, top_k, exp_begin, exp_end);
}

void launch_build_ranged_csr(
    const int32_t* d_topk_ids,
    const float* d_topk_weights,
    int32_t* d_active_experts,
    int32_t* d_csr_offsets,
    Mi50RoutedPair* d_sorted_pairs,
    int M, int top_k,
    int exp_begin, int exp_end,
    hipStream_t stream)
{
    const int total = M * top_k;
    if (total <= 0 || exp_end <= exp_begin || (exp_end - exp_begin) > 256) return;
    hipLaunchKernelGGL(k_build_ranged_csr, dim3(1), dim3(256), 0, stream,
                       d_topk_ids, d_topk_weights, d_active_experts, d_csr_offsets, d_sorted_pairs,
                       total, top_k, exp_begin, exp_end);
}

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
    hipStream_t stream)
{
    if (d_mapped_slot == nullptr || M <= 0 || M > kHlut3OffloadMaxRows) return;
    hipLaunchKernelGGL(k_stage_and_arm_hlut3_dev3, dim3(8), dim3(256), 0, stream,
                       d_mapped_slot, d_local_wg_sync, d_topk_ids, d_topk_weights,
                       d_x_rot_f16, d_x_norm_bf16, block_id, M, top_k,
                       exp_begin, exp_end, include_mtp_shared, req_seq);
}

void launch_dev3_ingest_and_pack(
    const Hlut3OffloadSlot* d_mapped_slot,
    uint16_t* d_x_rot_f16_dst,
    uint16_t* d_x_norm_bf16_dst,
    Mi50RoutedPair* d_pairs_dst,
    int M, int top_k,
    int exp_begin, int exp_end,
    bool copy_norm_bf16,
    hipStream_t stream)
{
    if (d_mapped_slot == nullptr || M <= 0 || M > kHlut3OffloadMaxRows) return;
    hipLaunchKernelGGL(k_dev3_ingest_and_pack, dim3(16), dim3(256), 0, stream,
                       d_mapped_slot, d_x_rot_f16_dst, d_x_norm_bf16_dst, d_pairs_dst,
                       M, top_k, exp_begin, exp_end, copy_norm_bf16 ? 1 : 0);
}

void launch_dev3_publish_f32(
    const float* d_out_f32,
    Hlut3OffloadSlot* d_mapped_slot,
    uint32_t* d_local_wg_sync,
    int M,
    uint32_t done_seq,
    hipStream_t stream)
{
    if (d_mapped_slot == nullptr || M <= 0 || M > kHlut3OffloadMaxRows) return;
    hipLaunchKernelGGL(k_dev3_publish_f32, dim3(16), dim3(256), 0, stream,
                       d_out_f32, d_mapped_slot, d_local_wg_sync,
                       M * kHlut3OffloadHidden, done_seq);
}

// ---- Wave 9.3 Phase D: Zero-Scratch Single-Launch Ingest + Expert-Grouped CSR on dev3 (M <= 16) ----
__global__ __launch_bounds__(256, 4)
void k_dev3_ingest_and_pack_csr(
    const Hlut3OffloadSlot* __restrict__ slot,
    uint16_t* __restrict__ d_x_rot_f16_dst,
    uint16_t* __restrict__ d_x_norm_bf16_dst,
    int32_t* __restrict__ d_active_exp_out,
    int32_t* __restrict__ d_csr_offsets_out,
    Mi50RoutedPair* __restrict__ d_pairs_out,
    int32_t* __restrict__ d_num_act_out,
    int M, int top_k,
    int exp_begin, int exp_end,
    int copy_norm_bf16)
{
    __shared__ int32_t s_cnt[256];
    __shared__ int32_t s_cur[256];

    const int tid = int(threadIdx.x);
    const int gid = int(blockIdx.x) * 256 + tid;
    const int stride = int(gridDim.x) * 256;

    const int n_words = (M * kHlut3OffloadHidden) >> 1;
    const uint32_t* src_rot_u32 = reinterpret_cast<const uint32_t*>(slot->x_rot_f16);
    uint32_t* dst_rot_u32 = reinterpret_cast<uint32_t*>(d_x_rot_f16_dst);
    for (int w = gid; w < n_words; w += stride) {
        dst_rot_u32[w] = __hip_atomic_load(&src_rot_u32[w], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    if (copy_norm_bf16 != 0 && d_x_norm_bf16_dst != nullptr) {
        const uint32_t* src_norm_u32 = reinterpret_cast<const uint32_t*>(slot->x_norm_bf16);
        uint32_t* dst_norm_u32 = reinterpret_cast<uint32_t*>(d_x_norm_bf16_dst);
        for (int w = gid; w < n_words; w += stride) {
            dst_norm_u32[w] = __hip_atomic_load(&src_norm_u32[w], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }

    if (blockIdx.x != 0) return;

    const int n_exp = exp_end - exp_begin;
    const int total_pairs = M * top_k;
    s_cnt[tid] = 0;
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < total_pairs; ++i) {
            const int32_t eid = __hip_atomic_load(&slot->topk_ids[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            const uint32_t wu = __hip_atomic_load(reinterpret_cast<const uint32_t*>(&slot->topk_weights[i]),
                                                  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            const float w = __uint_as_float(wu);
            if (eid >= exp_begin && eid < exp_end && w != 0.0f) {
                ++s_cnt[eid - exp_begin];
            }
        }
        int num_act = 0;
        int acc = 0;
        d_csr_offsets_out[0] = 0;
        for (int e = 0; e < n_exp; ++e) {
            const int c = s_cnt[e];
            if (c > 0) {
                s_cur[e] = acc;
                d_active_exp_out[num_act] = e;
                acc += c;
                d_csr_offsets_out[num_act + 1] = acc;
                ++num_act;
            }
        }
        d_num_act_out[0] = num_act;
        for (int i = 0; i < total_pairs; ++i) {
            const int32_t eid = __hip_atomic_load(&slot->topk_ids[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            const uint32_t wu = __hip_atomic_load(reinterpret_cast<const uint32_t*>(&slot->topk_weights[i]),
                                                  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            const float w = __uint_as_float(wu);
            if (eid >= exp_begin && eid < exp_end && w != 0.0f) {
                const int le = eid - exp_begin;
                const int dst_pos = s_cur[le]++;
                Mi50RoutedPair pr;
                pr.row = i / top_k;
                pr.expert = le;
                pr.weight = w;
                pr.pad = 0;
                d_pairs_out[dst_pos] = pr;
            }
        }
    }
}

// ---- Wave 9.3 Phase D: Full-GPU-Saturation Grouped v_dot2_f32_f16 Gate+Up+SwiGLU10 (ScratchSize == 0) ----
// Grid = (N_inter / 16, max_act) -> 144 CTAs/expert on dev3 (Ni=2304), 48 CTAs/expert on TP3 (Ni=768)!
// Saturates 100% of all 60 Vega 20 CUs even for 1 single active expert!
__global__ __launch_bounds__(256, 3)
void k_hlut3_fdot2_grouped_gu_f16(
    const uint16_t* __restrict__ x_rot_f16,
    const int32_t* __restrict__ active_experts,
    const int32_t* __restrict__ csr_offsets,
    const Mi50RoutedPair* __restrict__ sorted_pairs,
    const int32_t* __restrict__ d_num_act_opt,
    const Hlut3Tile32x32* __restrict__ gate_tiles,
    const Hlut3Tile32x32* __restrict__ up_tiles,
    const float* __restrict__ levels,
    uint16_t* __restrict__ inter_f16,
    int n_experts, int N_inter)
{
    __shared__ uint32_t s_x[8][17];   // 8 tokens x 16 f16x2 pairs (32 cols) + bank pad = 544 B
    __shared__ int32_t  s_row[8];

    const int ae = int(blockIdx.y);
    if (d_num_act_opt != nullptr && ae >= d_num_act_opt[0]) return;
    const int exp_id = active_experts[ae];
    if (exp_id < 0 || exp_id >= n_experts) return;
    const int p_start = csr_offsets[ae];
    const int p_end   = csr_offsets[ae + 1];
    if (p_start >= p_end) return;

    const int tid    = int(threadIdx.x);
    const int hw     = tid >> 4;       // 0..15 half-warp -> 16 output rows per CTA!
    const int lane16 = tid & 15;       // 0..15 -> 2 adjacent cols per lane = full 32-col tile!
    const int n      = int(blockIdx.x) * 16 + hw;
    const int nt     = n >> 5;
    const int lr     = n & 31;
    const int sg     = lr >> 2;
    const int nt_count = N_inter >> 5;
    constexpr int kKtCount = 160;      // 5120 / 32

    const uint32_t h0 = uint32_t(d_hlut3_f32_to_f16_bits(levels[0]));
    const uint32_t h1 = uint32_t(d_hlut3_f32_to_f16_bits(levels[1]));
    const uint32_t h2 = uint32_t(d_hlut3_f32_to_f16_bits(levels[2]));
    const uint32_t h3 = uint32_t(d_hlut3_f32_to_f16_bits(levels[3]));

    const size_t tile_base = size_t(exp_id) * size_t(nt_count) * size_t(kKtCount);
    const Hlut3Tile32x32* __restrict__ gtile_row = gate_tiles + tile_base + size_t(nt) * size_t(kKtCount);
    const Hlut3Tile32x32* __restrict__ utile_row = up_tiles   + tile_base + size_t(nt) * size_t(kKtCount);
    const uint32_t* x_u32 = reinterpret_cast<const uint32_t*>(x_rot_f16);
    const int lo_idx = lr * 2 + (lane16 >> 3);
    const int lo_sh  = 4 * (lane16 & 7);

    for (int p_base = p_start; p_base < p_end; p_base += 8) {
        const int B = min(8, p_end - p_base);

        if (B == 1) {
            const size_t r0_off = size_t(sorted_pairs[p_base + 0].row) * 2560u + size_t(lane16);
            float ag0 = 0.0f, au0 = 0.0f;
            #pragma unroll 2
            for (int kt = 0; kt < kKtCount; ++kt) {
                const uint32_t xb0 = x_u32[r0_off + size_t(kt * 16)];
                hlut3_f16x2_t xv0;
                __builtin_memcpy(&xv0, &xb0, 4);

                const Hlut3Tile32x32* __restrict__ gt = gtile_row + kt;
                const uint32_t g_lo = gt->q_lo2[lo_idx];
                const uint32_t g_hi = gt->q_hi1[lr];
                const uint32_t g_m0 = d_hlut3_sel_h16((g_lo >> lo_sh) & 3u, h0, h1, h2, h3);
                const uint32_t g_m1 = d_hlut3_sel_h16((g_lo >> (lo_sh + 2)) & 3u, h0, h1, h2, h3);
                const uint32_t g_pk = (g_m0 | (((g_hi >> (2 * lane16)) & 1u) << 15)) |
                                      ((g_m1 << 16) | (((g_hi >> (2 * lane16 + 1)) & 1u) << 31));
                hlut3_f16x2_t wg;
                __builtin_memcpy(&wg, &g_pk, 4);
                ag0 = fmaf(__builtin_amdgcn_fdot2(wg, xv0, 0.0f, false),
                           d_hlut3_f16_bits_to_f32(gt->scales_f16[sg]), ag0);

                const Hlut3Tile32x32* __restrict__ ut = utile_row + kt;
                const uint32_t u_lo = ut->q_lo2[lo_idx];
                const uint32_t u_hi = ut->q_hi1[lr];
                const uint32_t u_m0 = d_hlut3_sel_h16((u_lo >> lo_sh) & 3u, h0, h1, h2, h3);
                const uint32_t u_m1 = d_hlut3_sel_h16((u_lo >> (lo_sh + 2)) & 3u, h0, h1, h2, h3);
                const uint32_t u_pk = (u_m0 | (((u_hi >> (2 * lane16)) & 1u) << 15)) |
                                      ((u_m1 << 16) | (((u_hi >> (2 * lane16 + 1)) & 1u) << 31));
                hlut3_f16x2_t wu;
                __builtin_memcpy(&wu, &u_pk, 4);
                au0 = fmaf(__builtin_amdgcn_fdot2(wu, xv0, 0.0f, false),
                           d_hlut3_f16_bits_to_f32(ut->scales_f16[sg]), au0);
            }
            const float sg_sum = d_hlut3_sum16(ag0);
            const float su_sum = d_hlut3_sum16(au0);
            if (lane16 == 0) {
                inter_f16[size_t(p_base) * size_t(N_inter) + size_t(n)] =
                    d_hlut3_f32_to_f16_bits(d_hlut3_swiglu10(sg_sum, su_sum));
            }
            continue;
        }

        if (B == 2) {
            const size_t r0_off = size_t(sorted_pairs[p_base + 0].row) * 2560u + size_t(lane16);
            const size_t r1_off = size_t(sorted_pairs[p_base + 1].row) * 2560u + size_t(lane16);
            float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;
            #pragma unroll 2
            for (int kt = 0; kt < kKtCount; ++kt) {
                const uint32_t xb0 = x_u32[r0_off + size_t(kt * 16)];
                const uint32_t xb1 = x_u32[r1_off + size_t(kt * 16)];
                hlut3_f16x2_t xv0, xv1;
                __builtin_memcpy(&xv0, &xb0, 4);
                __builtin_memcpy(&xv1, &xb1, 4);

                const Hlut3Tile32x32* __restrict__ gt = gtile_row + kt;
                const uint32_t g_lo = gt->q_lo2[lo_idx];
                const uint32_t g_hi = gt->q_hi1[lr];
                const uint32_t g_m0 = d_hlut3_sel_h16((g_lo >> lo_sh) & 3u, h0, h1, h2, h3);
                const uint32_t g_m1 = d_hlut3_sel_h16((g_lo >> (lo_sh + 2)) & 3u, h0, h1, h2, h3);
                const uint32_t g_pk = (g_m0 | (((g_hi >> (2 * lane16)) & 1u) << 15)) |
                                      ((g_m1 << 16) | (((g_hi >> (2 * lane16 + 1)) & 1u) << 31));
                hlut3_f16x2_t wg;
                __builtin_memcpy(&wg, &g_pk, 4);
                const float g_sig = d_hlut3_f16_bits_to_f32(gt->scales_f16[sg]);
                ag0 = fmaf(__builtin_amdgcn_fdot2(wg, xv0, 0.0f, false), g_sig, ag0);
                ag1 = fmaf(__builtin_amdgcn_fdot2(wg, xv1, 0.0f, false), g_sig, ag1);

                const Hlut3Tile32x32* __restrict__ ut = utile_row + kt;
                const uint32_t u_lo = ut->q_lo2[lo_idx];
                const uint32_t u_hi = ut->q_hi1[lr];
                const uint32_t u_m0 = d_hlut3_sel_h16((u_lo >> lo_sh) & 3u, h0, h1, h2, h3);
                const uint32_t u_m1 = d_hlut3_sel_h16((u_lo >> (lo_sh + 2)) & 3u, h0, h1, h2, h3);
                const uint32_t u_pk = (u_m0 | (((u_hi >> (2 * lane16)) & 1u) << 15)) |
                                      ((u_m1 << 16) | (((u_hi >> (2 * lane16 + 1)) & 1u) << 31));
                hlut3_f16x2_t wu;
                __builtin_memcpy(&wu, &u_pk, 4);
                const float u_sig = d_hlut3_f16_bits_to_f32(ut->scales_f16[sg]);
                au0 = fmaf(__builtin_amdgcn_fdot2(wu, xv0, 0.0f, false), u_sig, au0);
                au1 = fmaf(__builtin_amdgcn_fdot2(wu, xv1, 0.0f, false), u_sig, au1);
            }
            const float sg0 = d_hlut3_sum16(ag0), su0 = d_hlut3_sum16(au0);
            const float sg1 = d_hlut3_sum16(ag1), su1 = d_hlut3_sum16(au1);
            if (lane16 == 0) {
                inter_f16[size_t(p_base + 0) * size_t(N_inter) + size_t(n)] =
                    d_hlut3_f32_to_f16_bits(d_hlut3_swiglu10(sg0, su0));
                inter_f16[size_t(p_base + 1) * size_t(N_inter) + size_t(n)] =
                    d_hlut3_f32_to_f16_bits(d_hlut3_swiglu10(sg1, su1));
            }
            continue;
        }

        if (B <= 4) {
            const size_t r0_off = size_t(sorted_pairs[p_base + 0].row) * 2560u + size_t(lane16);
            const size_t r1_off = size_t(sorted_pairs[p_base + 1].row) * 2560u + size_t(lane16);
            const size_t r2_off = size_t(sorted_pairs[p_base + 2].row) * 2560u + size_t(lane16);
            const size_t r3_off = size_t(sorted_pairs[p_base + ((B == 4) ? 3 : 2)].row) * 2560u + size_t(lane16);
            float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
            float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
            for (int kt = 0; kt < kKtCount; ++kt) {
                const uint32_t xb0 = x_u32[r0_off + size_t(kt * 16)];
                const uint32_t xb1 = x_u32[r1_off + size_t(kt * 16)];
                const uint32_t xb2 = x_u32[r2_off + size_t(kt * 16)];
                const uint32_t xb3 = x_u32[r3_off + size_t(kt * 16)];
                hlut3_f16x2_t xv0, xv1, xv2, xv3;
                __builtin_memcpy(&xv0, &xb0, 4);
                __builtin_memcpy(&xv1, &xb1, 4);
                __builtin_memcpy(&xv2, &xb2, 4);
                __builtin_memcpy(&xv3, &xb3, 4);

                const Hlut3Tile32x32* __restrict__ gt = gtile_row + kt;
                const uint32_t g_lo = gt->q_lo2[lo_idx];
                const uint32_t g_hi = gt->q_hi1[lr];
                const uint32_t g_m0 = d_hlut3_sel_h16((g_lo >> lo_sh) & 3u, h0, h1, h2, h3);
                const uint32_t g_m1 = d_hlut3_sel_h16((g_lo >> (lo_sh + 2)) & 3u, h0, h1, h2, h3);
                const uint32_t g_pk = (g_m0 | (((g_hi >> (2 * lane16)) & 1u) << 15)) |
                                      ((g_m1 << 16) | (((g_hi >> (2 * lane16 + 1)) & 1u) << 31));
                hlut3_f16x2_t wg;
                __builtin_memcpy(&wg, &g_pk, 4);
                const float g_sig = d_hlut3_f16_bits_to_f32(gt->scales_f16[sg]);
                ag0 = fmaf(__builtin_amdgcn_fdot2(wg, xv0, 0.0f, false), g_sig, ag0);
                ag1 = fmaf(__builtin_amdgcn_fdot2(wg, xv1, 0.0f, false), g_sig, ag1);
                ag2 = fmaf(__builtin_amdgcn_fdot2(wg, xv2, 0.0f, false), g_sig, ag2);
                ag3 = fmaf(__builtin_amdgcn_fdot2(wg, xv3, 0.0f, false), g_sig, ag3);

                const Hlut3Tile32x32* __restrict__ ut = utile_row + kt;
                const uint32_t u_lo = ut->q_lo2[lo_idx];
                const uint32_t u_hi = ut->q_hi1[lr];
                const uint32_t u_m0 = d_hlut3_sel_h16((u_lo >> lo_sh) & 3u, h0, h1, h2, h3);
                const uint32_t u_m1 = d_hlut3_sel_h16((u_lo >> (lo_sh + 2)) & 3u, h0, h1, h2, h3);
                const uint32_t u_pk = (u_m0 | (((u_hi >> (2 * lane16)) & 1u) << 15)) |
                                      ((u_m1 << 16) | (((u_hi >> (2 * lane16 + 1)) & 1u) << 31));
                hlut3_f16x2_t wu;
                __builtin_memcpy(&wu, &u_pk, 4);
                const float u_sig = d_hlut3_f16_bits_to_f32(ut->scales_f16[sg]);
                au0 = fmaf(__builtin_amdgcn_fdot2(wu, xv0, 0.0f, false), u_sig, au0);
                au1 = fmaf(__builtin_amdgcn_fdot2(wu, xv1, 0.0f, false), u_sig, au1);
                au2 = fmaf(__builtin_amdgcn_fdot2(wu, xv2, 0.0f, false), u_sig, au2);
                au3 = fmaf(__builtin_amdgcn_fdot2(wu, xv3, 0.0f, false), u_sig, au3);
            }
            const float sg0 = d_hlut3_sum16(ag0), su0 = d_hlut3_sum16(au0);
            const float sg1 = d_hlut3_sum16(ag1), su1 = d_hlut3_sum16(au1);
            const float sg2 = d_hlut3_sum16(ag2), su2 = d_hlut3_sum16(au2);
            const float sg3 = d_hlut3_sum16(ag3), su3 = d_hlut3_sum16(au3);
            if (lane16 == 0) {
                inter_f16[size_t(p_base + 0) * size_t(N_inter) + size_t(n)] = d_hlut3_f32_to_f16_bits(d_hlut3_swiglu10(sg0, su0));
                inter_f16[size_t(p_base + 1) * size_t(N_inter) + size_t(n)] = d_hlut3_f32_to_f16_bits(d_hlut3_swiglu10(sg1, su1));
                inter_f16[size_t(p_base + 2) * size_t(N_inter) + size_t(n)] = d_hlut3_f32_to_f16_bits(d_hlut3_swiglu10(sg2, su2));
                if (B == 4) {
                    inter_f16[size_t(p_base + 3) * size_t(N_inter) + size_t(n)] = d_hlut3_f32_to_f16_bits(d_hlut3_swiglu10(sg3, su3));
                }
            }
            continue;
        }

        if (tid < 8) {
            s_row[tid] = (tid < B) ? sorted_pairs[p_base + tid].row : 0;
        }
        __syncthreads();

        float acc_g[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float acc_u[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        for (int kt = 0; kt < kKtCount; ++kt) {
            if (hw < B) {
                s_x[hw][lane16] = x_u32[size_t(s_row[hw]) * 2560u + size_t(kt * 16 + lane16)];
            } else if (hw < 8) {
                s_x[hw][lane16] = 0u;
            }

            const Hlut3Tile32x32* __restrict__ gt = gtile_row + kt;
            const Hlut3Tile32x32* __restrict__ ut = utile_row + kt;
            const uint32_t g_lo = gt->q_lo2[lo_idx];
            const uint32_t g_hi = gt->q_hi1[lr];
            const uint32_t g_m0 = d_hlut3_sel_h16((g_lo >> lo_sh) & 3u, h0, h1, h2, h3);
            const uint32_t g_m1 = d_hlut3_sel_h16((g_lo >> (lo_sh + 2)) & 3u, h0, h1, h2, h3);
            const uint32_t g_pk = (g_m0 | (((g_hi >> (2 * lane16)) & 1u) << 15)) |
                                  ((g_m1 << 16) | (((g_hi >> (2 * lane16 + 1)) & 1u) << 31));
            hlut3_f16x2_t wg;
            __builtin_memcpy(&wg, &g_pk, 4);
            const float g_sig = d_hlut3_f16_bits_to_f32(gt->scales_f16[sg]);

            const uint32_t u_lo = ut->q_lo2[lo_idx];
            const uint32_t u_hi = ut->q_hi1[lr];
            const uint32_t u_m0 = d_hlut3_sel_h16((u_lo >> lo_sh) & 3u, h0, h1, h2, h3);
            const uint32_t u_m1 = d_hlut3_sel_h16((u_lo >> (lo_sh + 2)) & 3u, h0, h1, h2, h3);
            const uint32_t u_pk = (u_m0 | (((u_hi >> (2 * lane16)) & 1u) << 15)) |
                                  ((u_m1 << 16) | (((u_hi >> (2 * lane16 + 1)) & 1u) << 31));
            hlut3_f16x2_t wu;
            __builtin_memcpy(&wu, &u_pk, 4);
            const float u_sig = d_hlut3_f16_bits_to_f32(ut->scales_f16[sg]);

            __syncthreads();
            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                hlut3_f16x2_t xv;
                const uint32_t xbits = s_x[b][lane16];
                __builtin_memcpy(&xv, &xbits, 4);
                acc_g[b] = fmaf(__builtin_amdgcn_fdot2(wg, xv, 0.0f, false), g_sig, acc_g[b]);
                acc_u[b] = fmaf(__builtin_amdgcn_fdot2(wu, xv, 0.0f, false), u_sig, acc_u[b]);
            }
            __syncthreads();
        }

        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            if (b < B) {
                const float sg_sum = d_hlut3_sum16(acc_g[b]);
                const float su_sum = d_hlut3_sum16(acc_u[b]);
                if (lane16 == 0) {
                    inter_f16[size_t(p_base + b) * size_t(N_inter) + size_t(n)] =
                        d_hlut3_f32_to_f16_bits(d_hlut3_swiglu10(sg_sum, su_sum));
                }
            }
        }
    }
}

// Fast in-place WHT128(signs_i) on active CSR pairs in FP16 (Grid = (N_inter/128, max_act), Block = 128)
__global__ __launch_bounds__(128, 8)
void k_hlut3_wht128_csr_f16_inplace(
    const int32_t* __restrict__ active_experts,
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ d_num_act_opt,
    const uint64_t* __restrict__ signs_i,
    uint16_t* __restrict__ inter_f16,
    int n_experts, int N_inter)
{
    __shared__ float s_wht[128];
    const int ae = int(blockIdx.y);
    if (d_num_act_opt != nullptr && ae >= d_num_act_opt[0]) return;
    const int exp_id = active_experts[ae];
    if (exp_id < 0 || exp_id >= n_experts) return;
    const int p_start = csr_offsets[ae];
    const int p_end   = csr_offsets[ae + 1];
    if (p_start >= p_end) return;

    const int tid    = int(threadIdx.x);
    const int lane64 = tid & 63;
    const int wid64  = tid >> 6;
    const int blk128 = int(blockIdx.x);
    const int col    = blk128 * 128 + tid;
    const uint64_t sw = signs_i[blk128 * 2 + wid64];
    const bool flip   = ((sw >> lane64) & 1ULL) != 0ULL;

    for (int p = p_start; p < p_end; ++p) {
        float v = d_hlut3_f16_bits_to_f32(inter_f16[size_t(p) * size_t(N_inter) + size_t(col)]);
        if (flip) v = -v;
        #pragma unroll
        for (int len = 1; len <= 32; len <<= 1) {
            const float partner = __shfl_xor(v, len, 64);
            v = (lane64 & len) ? (partner - v) : (v + partner);
        }
        s_wht[tid] = v;
        __syncthreads();
        const float a = s_wht[lane64];
        const float b_val = s_wht[64 + lane64];
        const float res = ((wid64 == 0) ? (a + b_val) : (a - b_val)) * 0.08838834764831844655f;
        inter_f16[size_t(p) * size_t(N_inter) + size_t(col)] = d_hlut3_f32_to_f16_bits(res);
        __syncthreads();
    }
}

// ---- Wave 9.3 Phase D: Full-GPU-Saturation Grouped v_dot2_f32_f16 Down Projection + Local-HBM2 Scatter ----
// Grid = (5120 / 16 = 320, max_act), Block = 256 (16 half-warps x 16 lanes = 16 output rows/CTA, 0 sub loop!)
__global__ __launch_bounds__(256, 3)
void k_hlut3_fdot2_grouped_down_f16(
    const uint16_t* __restrict__ inter_rot_f16,
    const int32_t* __restrict__ active_experts,
    const int32_t* __restrict__ csr_offsets,
    const Mi50RoutedPair* __restrict__ sorted_pairs,
    const int32_t* __restrict__ d_num_act_opt,
    const Hlut3Tile32x32* __restrict__ down_tiles,
    const float* __restrict__ levels,
    float* __restrict__ out_accum_f32,
    int n_experts, int K_down)
{
    __shared__ uint32_t s_z[8][17];
    __shared__ int32_t  s_row[8];
    __shared__ float    s_w[8];

    const int ae = int(blockIdx.y);
    if (d_num_act_opt != nullptr && ae >= d_num_act_opt[0]) return;
    const int exp_id = active_experts[ae];
    if (exp_id < 0 || exp_id >= n_experts) return;
    const int p_start = csr_offsets[ae];
    const int p_end   = csr_offsets[ae + 1];
    if (p_start >= p_end) return;

    const int tid      = int(threadIdx.x);
    const int hw       = tid >> 4;
    const int lane16   = tid & 15;
    const int n        = int(blockIdx.x) * 16 + hw; // 0..5119 (320 CTAs -> 100% CU saturation!)
    const int nt       = n >> 5;
    const int lr       = n & 31;
    const int sg       = lr >> 2;
    constexpr int kNtCount = 160;
    const int kt_count = K_down >> 5;
    const int k_u32_stride = K_down >> 1;
    const int lo_idx = lr * 2 + (lane16 >> 3);
    const int lo_sh  = 4 * (lane16 & 7);

    const uint32_t h0 = uint32_t(d_hlut3_f32_to_f16_bits(levels[0]));
    const uint32_t h1 = uint32_t(d_hlut3_f32_to_f16_bits(levels[1]));
    const uint32_t h2 = uint32_t(d_hlut3_f32_to_f16_bits(levels[2]));
    const uint32_t h3 = uint32_t(d_hlut3_f32_to_f16_bits(levels[3]));

    const Hlut3Tile32x32* __restrict__ dtile_row =
        down_tiles + (size_t(exp_id) * size_t(kNtCount) + size_t(nt)) * size_t(kt_count);
    const uint32_t* z_u32 = reinterpret_cast<const uint32_t*>(inter_rot_f16);

    for (int p_base = p_start; p_base < p_end; p_base += 8) {
        const int B = min(8, p_end - p_base);

        if (B == 1) {
            const Mi50RoutedPair pr0 = sorted_pairs[p_base + 0];
            const size_t z0_off = size_t(p_base + 0) * size_t(k_u32_stride) + size_t(lane16);
            float a0 = 0.0f;
            #pragma unroll 2
            for (int kt = 0; kt < kt_count; ++kt) {
                const uint32_t zb0 = z_u32[z0_off + size_t(kt * 16)];
                hlut3_f16x2_t zv0;
                __builtin_memcpy(&zv0, &zb0, 4);
                const Hlut3Tile32x32* __restrict__ dt = dtile_row + kt;
                const uint32_t d_lo = dt->q_lo2[lo_idx];
                const uint32_t d_hi = dt->q_hi1[lr];
                const uint32_t d_m0 = d_hlut3_sel_h16((d_lo >> lo_sh) & 3u, h0, h1, h2, h3);
                const uint32_t d_m1 = d_hlut3_sel_h16((d_lo >> (lo_sh + 2)) & 3u, h0, h1, h2, h3);
                const uint32_t d_pk = (d_m0 | (((d_hi >> (2 * lane16)) & 1u) << 15)) |
                                      ((d_m1 << 16) | (((d_hi >> (2 * lane16 + 1)) & 1u) << 31));
                hlut3_f16x2_t wd;
                __builtin_memcpy(&wd, &d_pk, 4);
                a0 = fmaf(__builtin_amdgcn_fdot2(wd, zv0, 0.0f, false),
                          d_hlut3_f16_bits_to_f32(dt->scales_f16[sg]), a0);
            }
            const float sd0 = d_hlut3_sum16(a0);
            if (lane16 == 0 && pr0.weight != 0.0f) {
                atomicAdd(&out_accum_f32[size_t(pr0.row) * 5120u + size_t(n)], sd0 * pr0.weight);
            }
            continue;
        }

        if (B == 2) {
            const Mi50RoutedPair pr0 = sorted_pairs[p_base + 0];
            const Mi50RoutedPair pr1 = sorted_pairs[p_base + 1];
            const size_t z0_off = size_t(p_base + 0) * size_t(k_u32_stride) + size_t(lane16);
            const size_t z1_off = size_t(p_base + 1) * size_t(k_u32_stride) + size_t(lane16);
            float a0 = 0.0f, a1 = 0.0f;
            #pragma unroll 2
            for (int kt = 0; kt < kt_count; ++kt) {
                const uint32_t zb0 = z_u32[z0_off + size_t(kt * 16)];
                const uint32_t zb1 = z_u32[z1_off + size_t(kt * 16)];
                hlut3_f16x2_t zv0, zv1;
                __builtin_memcpy(&zv0, &zb0, 4);
                __builtin_memcpy(&zv1, &zb1, 4);
                const Hlut3Tile32x32* __restrict__ dt = dtile_row + kt;
                const uint32_t d_lo = dt->q_lo2[lo_idx];
                const uint32_t d_hi = dt->q_hi1[lr];
                const uint32_t d_m0 = d_hlut3_sel_h16((d_lo >> lo_sh) & 3u, h0, h1, h2, h3);
                const uint32_t d_m1 = d_hlut3_sel_h16((d_lo >> (lo_sh + 2)) & 3u, h0, h1, h2, h3);
                const uint32_t d_pk = (d_m0 | (((d_hi >> (2 * lane16)) & 1u) << 15)) |
                                      ((d_m1 << 16) | (((d_hi >> (2 * lane16 + 1)) & 1u) << 31));
                hlut3_f16x2_t wd;
                __builtin_memcpy(&wd, &d_pk, 4);
                const float d_sig = d_hlut3_f16_bits_to_f32(dt->scales_f16[sg]);
                a0 = fmaf(__builtin_amdgcn_fdot2(wd, zv0, 0.0f, false), d_sig, a0);
                a1 = fmaf(__builtin_amdgcn_fdot2(wd, zv1, 0.0f, false), d_sig, a1);
            }
            const float sd0 = d_hlut3_sum16(a0);
            const float sd1 = d_hlut3_sum16(a1);
            if (lane16 == 0) {
                if (pr0.weight != 0.0f) atomicAdd(&out_accum_f32[size_t(pr0.row) * 5120u + size_t(n)], sd0 * pr0.weight);
                if (pr1.weight != 0.0f) atomicAdd(&out_accum_f32[size_t(pr1.row) * 5120u + size_t(n)], sd1 * pr1.weight);
            }
            continue;
        }

        if (B <= 4) {
            const Mi50RoutedPair pr0 = sorted_pairs[p_base + 0];
            const Mi50RoutedPair pr1 = sorted_pairs[p_base + 1];
            const Mi50RoutedPair pr2 = sorted_pairs[p_base + 2];
            const Mi50RoutedPair pr3 = sorted_pairs[p_base + ((B == 4) ? 3 : 2)];
            const size_t z0_off = size_t(p_base + 0) * size_t(k_u32_stride) + size_t(lane16);
            const size_t z1_off = size_t(p_base + 1) * size_t(k_u32_stride) + size_t(lane16);
            const size_t z2_off = size_t(p_base + 2) * size_t(k_u32_stride) + size_t(lane16);
            const size_t z3_off = size_t(p_base + ((B == 4) ? 3 : 2)) * size_t(k_u32_stride) + size_t(lane16);
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int kt = 0; kt < kt_count; ++kt) {
                const uint32_t zb0 = z_u32[z0_off + size_t(kt * 16)];
                const uint32_t zb1 = z_u32[z1_off + size_t(kt * 16)];
                const uint32_t zb2 = z_u32[z2_off + size_t(kt * 16)];
                const uint32_t zb3 = z_u32[z3_off + size_t(kt * 16)];
                hlut3_f16x2_t zv0, zv1, zv2, zv3;
                __builtin_memcpy(&zv0, &zb0, 4);
                __builtin_memcpy(&zv1, &zb1, 4);
                __builtin_memcpy(&zv2, &zb2, 4);
                __builtin_memcpy(&zv3, &zb3, 4);
                const Hlut3Tile32x32* __restrict__ dt = dtile_row + kt;
                const uint32_t d_lo = dt->q_lo2[lo_idx];
                const uint32_t d_hi = dt->q_hi1[lr];
                const uint32_t d_m0 = d_hlut3_sel_h16((d_lo >> lo_sh) & 3u, h0, h1, h2, h3);
                const uint32_t d_m1 = d_hlut3_sel_h16((d_lo >> (lo_sh + 2)) & 3u, h0, h1, h2, h3);
                const uint32_t d_pk = (d_m0 | (((d_hi >> (2 * lane16)) & 1u) << 15)) |
                                      ((d_m1 << 16) | (((d_hi >> (2 * lane16 + 1)) & 1u) << 31));
                hlut3_f16x2_t wd;
                __builtin_memcpy(&wd, &d_pk, 4);
                const float d_sig = d_hlut3_f16_bits_to_f32(dt->scales_f16[sg]);
                a0 = fmaf(__builtin_amdgcn_fdot2(wd, zv0, 0.0f, false), d_sig, a0);
                a1 = fmaf(__builtin_amdgcn_fdot2(wd, zv1, 0.0f, false), d_sig, a1);
                a2 = fmaf(__builtin_amdgcn_fdot2(wd, zv2, 0.0f, false), d_sig, a2);
                a3 = fmaf(__builtin_amdgcn_fdot2(wd, zv3, 0.0f, false), d_sig, a3);
            }
            const float sd0 = d_hlut3_sum16(a0);
            const float sd1 = d_hlut3_sum16(a1);
            const float sd2 = d_hlut3_sum16(a2);
            const float sd3 = d_hlut3_sum16(a3);
            if (lane16 == 0) {
                if (pr0.weight != 0.0f) atomicAdd(&out_accum_f32[size_t(pr0.row) * 5120u + size_t(n)], sd0 * pr0.weight);
                if (pr1.weight != 0.0f) atomicAdd(&out_accum_f32[size_t(pr1.row) * 5120u + size_t(n)], sd1 * pr1.weight);
                if (pr2.weight != 0.0f) atomicAdd(&out_accum_f32[size_t(pr2.row) * 5120u + size_t(n)], sd2 * pr2.weight);
                if (B == 4 && pr3.weight != 0.0f) atomicAdd(&out_accum_f32[size_t(pr3.row) * 5120u + size_t(n)], sd3 * pr3.weight);
            }
            continue;
        }

        if (tid < 8) {
            if (tid < B) {
                const Mi50RoutedPair pr = sorted_pairs[p_base + tid];
                s_row[tid] = pr.row;
                s_w[tid]   = pr.weight;
            } else {
                s_row[tid] = 0;
                s_w[tid]   = 0.0f;
            }
        }
        __syncthreads();

        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int kt = 0; kt < kt_count; ++kt) {
            if (hw < B) {
                s_z[hw][lane16] = z_u32[size_t(p_base + hw) * size_t(k_u32_stride) + size_t(kt * 16 + lane16)];
            } else if (hw < 8) {
                s_z[hw][lane16] = 0u;
            }
            const Hlut3Tile32x32* __restrict__ dt = dtile_row + kt;
            const uint32_t d_lo = dt->q_lo2[lo_idx];
            const uint32_t d_hi = dt->q_hi1[lr];
            const uint32_t d_m0 = d_hlut3_sel_h16((d_lo >> lo_sh) & 3u, h0, h1, h2, h3);
            const uint32_t d_m1 = d_hlut3_sel_h16((d_lo >> (lo_sh + 2)) & 3u, h0, h1, h2, h3);
            const uint32_t d_pk = (d_m0 | (((d_hi >> (2 * lane16)) & 1u) << 15)) |
                                  ((d_m1 << 16) | (((d_hi >> (2 * lane16 + 1)) & 1u) << 31));
            hlut3_f16x2_t wd;
            __builtin_memcpy(&wd, &d_pk, 4);
            const float d_sig = d_hlut3_f16_bits_to_f32(dt->scales_f16[sg]);

            __syncthreads();
            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                hlut3_f16x2_t zv;
                const uint32_t zbits = s_z[b][lane16];
                __builtin_memcpy(&zv, &zbits, 4);
                acc[b] = fmaf(__builtin_amdgcn_fdot2(wd, zv, 0.0f, false), d_sig, acc[b]);
            }
            __syncthreads();
        }

        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            if (b < B) {
                const float sd = d_hlut3_sum16(acc[b]);
                if (lane16 == 0 && s_w[b] != 0.0f) {
                    atomicAdd(&out_accum_f32[size_t(s_row[b]) * 5120u + size_t(n)], sd * s_w[b]);
                }
            }
        }
    }
}

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
    hipStream_t stream)
{
    if (d_mapped_slot == nullptr || M <= 0 || M > kHlut3OffloadMaxRows) return;
    hipLaunchKernelGGL(k_dev3_ingest_and_pack_csr, dim3(16), dim3(256), 0, stream,
                       d_mapped_slot, d_x_rot_f16_dst, d_x_norm_bf16_dst,
                       d_active_exp_out, d_csr_offsets_out, d_pairs_out, d_num_act_out,
                       M, top_k, exp_begin, exp_end, copy_norm_bf16 ? 1 : 0);
}

void launch_hlut3_fdot2_grouped_f16(
    const uint16_t* d_x_rot_f16,
    const int32_t* d_active_experts,
    const int32_t* d_csr_offsets,
    const Mi50RoutedPair* d_sorted_pairs,
    const int32_t* d_num_act_opt,
    int max_act,
    int n_experts,
    int n_inter,
    const Hlut3Tile32x32* d_gate_tiles,
    const Hlut3Tile32x32* d_up_tiles,
    const Hlut3Tile32x32* d_down_tiles,
    const float* d_levels,
    const uint64_t* d_signs_i,
    uint16_t* d_inter_rot_f16,
    float* d_out_accum_f32,
    hipStream_t stream)
{
    if (max_act <= 0 || n_experts <= 0 || n_inter <= 0 || (n_inter & 127) != 0) return;
    hipLaunchKernelGGL(k_hlut3_fdot2_grouped_gu_f16,
                       dim3(unsigned(n_inter >> 4), unsigned(max_act)), dim3(256), 0, stream,
                       d_x_rot_f16, d_active_experts, d_csr_offsets, d_sorted_pairs, d_num_act_opt,
                       d_gate_tiles, d_up_tiles, d_levels, d_inter_rot_f16,
                       n_experts, n_inter);
    hipLaunchKernelGGL(k_hlut3_wht128_csr_f16_inplace,
                       dim3(unsigned(n_inter >> 7), unsigned(max_act)), dim3(128), 0, stream,
                       d_active_experts, d_csr_offsets, d_num_act_opt, d_signs_i, d_inter_rot_f16,
                       n_experts, n_inter);
    hipLaunchKernelGGL(k_hlut3_fdot2_grouped_down_f16,
                       dim3(320u, unsigned(max_act)), dim3(256), 0, stream,
                       d_inter_rot_f16, d_active_experts, d_csr_offsets, d_sorted_pairs, d_num_act_opt,
                       d_down_tiles, d_levels, d_out_accum_f32,
                       n_experts, n_inter);
}

}  // namespace dsv41
