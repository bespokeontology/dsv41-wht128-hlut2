// Wave 9 Deliverable C: gfx906 zero-scratch WHT128-HLUT3 decode kernels.
// 400 B tile planes are unpacked in registers: q_lo2 2-bit magnitude -> 4-centroid select,
// q_hi1 sign bit XOR, FP16 group RMS scale, then IEEE FP32 FMA accumulation.
// Zero dynamic local arrays, zero shared scratch for GEMM kernels => ScratchSize == 0.
// No Python. No v_sin_f32/v_cos_f32; only expf in SwiGLU clamp-10.

#include "dsv41_hlut3_kernels.hpp"

namespace dsv41 {
namespace {

__device__ __forceinline__ float d_hlut3_f16_bits_to_f32(uint16_t h) noexcept {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t exp = (uint32_t(h) >> 10) & 0x1Fu;
    const uint32_t mant = uint32_t(h) & 0x03FFu;
    uint32_t u = 0u;
    if (exp == 0u) {
        if (mant == 0u) {
            u = sign;
        } else {
            int e = -14;
            uint32_t m = mant;
            while ((m & 0x0400u) == 0u) { m <<= 1; --e; }
            m &= 0x03FFu;
            u = sign | (uint32_t(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31u) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | (uint32_t(int(exp) - 15 + 127) << 23) | (mant << 13);
    }
    float f = 0.0f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

__device__ __forceinline__ float d_hlut3_swiglu10(float g, float u) noexcept {
    const float gc = fminf(g, 10.0f);
    const float uc = fmaxf(-10.0f, fminf(10.0f, u));
    return (gc / (1.0f + expf(-gc))) * uc;
}

__device__ __forceinline__ float d_hlut3_tile_weight(
    const Hlut3Tile32x32* __restrict__ tile,
    int local_row,
    int lane,
    int scale_group,
    const float* __restrict__ levels) noexcept
{
    // Two magnitude words per tile row: local_row*2 + lane/16.
    const uint32_t mag_word = tile->q_lo2[local_row * 2 + (lane >> 4)];
    const uint32_t mag_code = (mag_word >> (2 * (lane & 15))) & 3u;
    const uint32_t sign_bit = (tile->q_hi1[local_row] >> lane) & 1u;
    const float sigma = d_hlut3_f16_bits_to_f32(tile->scales_f16[scale_group]);
    const float magnitude = levels[mag_code] * sigma;
    return sign_bit ? -magnitude : magnitude;
}

__device__ __forceinline__ float d_hlut3_warp_sum(float v) noexcept {
    #pragma unroll
    for (int len = 1; len <= 16; len <<= 1) {
        v += __shfl_xor(v, len);
    }
    return v;
}

}  // namespace

// One 8-row warp block per (n, m). 8 warps each compute one output row over K/32 tiles.
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

// Routed gate+up+clamp-10 SwiGLU for each routed pair. Expert-major tile plane selection:
//   tile_base(expert) = expert * (N/32) * (K/32), then nt * (K/32) + kt.
// Output inter_z_f32 [P, N] (rank TP3 slice N=768, full verify N=2304).
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

// Q^{-1} = Q exactly for normalized H_128: inverse WHT128 on the 2304 intermediate row.
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

// Routed down + router-weight epilogue + atomic FP32 accumulation into [M, N_down].
// Expert-major down tiles: base = expert * (N_down/32) * (K/32).
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
    hipLaunchKernelGGL(k_hlut3_wht128_z_f32,
                       dim3(unsigned(n_gate / 128), unsigned(P)), dim3(128), 0, stream,
                       d_inter_z_f32, d_signs_i, d_inter_zrot_f32, n_gate);
    hipLaunchKernelGGL(k_hlut3_pair_down_atomic,
                       dim3(unsigned(n_down / 8), unsigned(P)), dim3(256), 0, stream,
                       d_inter_zrot_f32, d_pairs, d_down_tiles, d_down_levels,
                       d_out_accum_f32, n_experts, n_down, k_down);
}

// FP16 -> FP32 conversion for the opt-in HLUT3 MTP overlay path (runtime TP3 slice activations).
__global__ __launch_bounds__(256, 8)
void k_hlut3_f16_to_f32(
    const uint16_t* __restrict__ in_f16,
    float* __restrict__ out_f32,
    int n)
{
    const int idx = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (idx < n) out_f32[idx] = d_hlut3_f16_bits_to_f32(in_f16[idx]);
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

}  // namespace dsv41
