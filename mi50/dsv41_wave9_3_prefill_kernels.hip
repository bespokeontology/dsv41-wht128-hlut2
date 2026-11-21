// Wave 9.3 Phase B: Zero-Scratch gfx906 Unpack, Gather, Fused SwiGLU10+WHT128, & Scatter Kernels
// Zero dynamic local arrays, ScratchSize == 0, 0 SGPR/VGPR spills, 0 trig, 0 system-scope RMW atomics.

#include "dsv41_wave9_3_prefill_kernels.hpp"
#include <hip/hip_fp16.h>

namespace dsv41 {
namespace {

__device__ __forceinline__ uint16_t d_w93_f32_to_f16_clamped(float x) noexcept {
    x = fmaxf(-65504.0f, fminf(65504.0f, x));
    return __half_as_ushort(__float2half_rn(x));
}

__device__ __forceinline__ float d_w93_f16_to_f32(uint16_t h) noexcept {
    return __half2float(__ushort_as_half(h));
}

__device__ __forceinline__ float d_w93_bf16_to_f32(uint16_t b) noexcept {
    return __uint_as_float(uint32_t(b) << 16);
}

__device__ __forceinline__ uint16_t d_w93_f32_to_bf16_rne(float v) noexcept {
    uint32_t u = __float_as_uint(v);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return uint16_t((u >> 16) | 0x0040u);
    u += 0x00007FFFu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

__device__ __forceinline__ float d_w93_ue8m0_to_f32(uint8_t b) noexcept {
    if (b == 255u) return 0.0f;
    const uint32_t bits = (b == 0u) ? 0x00400000u : (uint32_t(b) << 23);
    return __uint_as_float(bits);
}

__device__ __forceinline__ uint16_t d_w93_fp8_e4m3_to_f16_bits(uint8_t b) noexcept {
    const uint32_t s = uint32_t(b & 0x80u) << 8;
    const uint32_t em = uint32_t(b & 0x7Fu);
    if (em == 0x7Fu) return uint16_t(s);
    const uint32_t e = em >> 3;
    const uint32_t m = em & 7u;
    const uint32_t norm_mag = ((e + 8u) << 10) | (m << 7);
    const uint32_t sub_mag = uint32_t((0x232221201E1C1800ULL >> (m * 8u)) & 0xFFULL) << 8;
    return uint16_t(s | ((e == 0u) ? sub_mag : norm_mag));
}

}  // namespace

// 1. Register-only FP8 E4M3 + UE8M0_32 -> FP16 unpack (0 LDS, 0 __syncthreads(), max occupancy!)
__global__ __launch_bounds__(256, 8)
void k_w93_unpack_fp8_to_f16(
    const uint8_t* __restrict__ fp8,
    const uint8_t* __restrict__ ue8m0,
    uint16_t* __restrict__ out_f16,
    int N, int K, int per_row_scale)
{
    const int tid  = int(threadIdx.x);
    const int lane = tid & 31;       // 0..31 within 32-element UE8M0 block
    const int r_in = tid >> 5;       // 0..7
    const int row  = int(blockIdx.y) * 8 + r_in;
    const int kb   = int(blockIdx.x);
    const int col  = kb * 32 + lane;
    if (row >= N || col >= K) return;

    const int k_blks = K >> 5;
    const int s_row  = (per_row_scale != 0) ? row : (row >> 5);
    const float sc   = d_w93_ue8m0_to_f32(ue8m0[size_t(s_row) * size_t(k_blks) + size_t(kb)]);
    const uint8_t qb = fp8[size_t(row) * size_t(K) + size_t(col)];
    const float qf   = d_w93_f16_to_f32(d_w93_fp8_e4m3_to_f16_bits(qb));
    out_f16[size_t(row) * size_t(K) + size_t(col)] = d_w93_f32_to_f16_clamped(qf * sc);
}

// 2. BF16 -> FP16 clamped conversion
__global__ __launch_bounds__(256, 8)
void k_w93_bf16_to_f16(
    const uint16_t* __restrict__ in_bf16,
    uint16_t* __restrict__ out_f16,
    size_t n_elems)
{
    const size_t idx = size_t(blockIdx.x) * 256u + size_t(threadIdx.x);
    if (idx < n_elems) {
        out_f16[idx] = d_w93_f32_to_f16_clamped(d_w93_bf16_to_f32(in_bf16[idx]));
    }
}

// 3. FP32 -> BF16 RNE conversion
__global__ __launch_bounds__(256, 8)
void k_w93_f32_to_bf16_rne(
    const float* __restrict__ in_f32,
    uint16_t* __restrict__ out_bf16,
    size_t n_elems)
{
    const size_t idx = size_t(blockIdx.x) * 256u + size_t(threadIdx.x);
    if (idx < n_elems) {
        out_bf16[idx] = d_w93_f32_to_bf16_rne(in_f32[idx]);
    }
}

// 4a. Coalesced HLUT2 Expert Matrix Unpack [N, K] -> Row-Major FP16 [N, K]
__global__ __launch_bounds__(256, 4)
void k_w93_unpack_hlut2_expert_f16(
    const uint32_t* __restrict__ codes,   // [E, N/32, K/16, 32]
    const uint16_t* __restrict__ scales,  // [E, N/32, K/128, 32]
    const float*    __restrict__ cent,    // [E, 2]
    int expert_id,
    int N, int K,
    uint16_t* __restrict__ out_f16)       // [N, K]
{
    __shared__ uint32_t s_tile[32][65];   // 64 uint32 (= 128 FP16) + 1 bank-pad -> 8.125 KiB LDS

    const int tid = int(threadIdx.x);
    const int r   = tid & 31;             // 0..31 row within 32-row block
    const int w   = tid >> 5;             // 0..7 word within 128-col group (16 weights/word)
    const int g   = int(blockIdx.x);      // 0 .. K/128 - 1
    const int rb  = int(blockIdx.y);      // 0 .. N/32 - 1

    const int n_blks   = N >> 5;
    const int k_words  = K >> 4;          // K / 16
    const int k_groups = K >> 7;          // K / 128
    const size_t e = size_t(expert_id);
    const size_t exp_code_stride  = size_t(n_blks) * size_t(k_words) * 32u;
    const size_t exp_scale_stride = size_t(n_blks) * size_t(k_groups) * 32u;

    const float c0 = cent[e * 2u + 0u];
    const float c1 = cent[e * 2u + 1u];
    const float sig = d_w93_f16_to_f32(scales[e * exp_scale_stride + (size_t(rb) * size_t(k_groups) + size_t(g)) * 32u + size_t(r)]);
    const float v0 = c0 * sig;
    const float v1 = c1 * sig;

    const uint32_t packed = codes[e * exp_code_stride + (size_t(rb) * size_t(k_words) + size_t(g * 8 + w)) * 32u + size_t(r)];

    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const uint32_t nib = (packed >> (4 * p)) & 0xFu;
        const float mag0 = (nib & 1u) ? v1 : v0;
        const float mag1 = (nib & 4u) ? v1 : v0;
        const float f0   = (nib & 2u) ? -mag0 : mag0;
        const float f1   = (nib & 8u) ? -mag1 : mag1;
        const uint32_t h0 = uint32_t(d_w93_f32_to_f16_clamped(f0));
        const uint32_t h1 = uint32_t(d_w93_f32_to_f16_clamped(f1));
        s_tile[r][w * 8 + p] = h0 | (h1 << 16);
    }
    __syncthreads();

    const int out_w = tid & 15;           // 0..15 (x4 uint32 = 64 uint32 = 128 FP16)
    const int r_base = tid >> 4;          // 0..15
    uint4* dst_base = reinterpret_cast<uint4*>(out_f16 + size_t(rb * 32) * size_t(K) + size_t(g * 128));
    const size_t row_stride_u4 = size_t(K) >> 3;

    #pragma unroll
    for (int step = 0; step < 2; ++step) {
        const int rr = r_base + step * 16;
        uint4 v;
        v.x = s_tile[rr][out_w * 4 + 0];
        v.y = s_tile[rr][out_w * 4 + 1];
        v.z = s_tile[rr][out_w * 4 + 2];
        v.w = s_tile[rr][out_w * 4 + 3];
        dst_base[size_t(rr) * row_stride_u4 + size_t(out_w)] = v;
    }
}

// 4b. Fused single-launch Gate+Up HLUT2 Unpack [2*N_inter, K] -> Row-Major FP16 [2*N_inter, K]
__global__ __launch_bounds__(256, 4)
void k_w93_unpack_hlut2_gateup_f16(
    const uint32_t* __restrict__ gate_codes, const uint16_t* __restrict__ gate_scales, const float* __restrict__ gate_cent,
    const uint32_t* __restrict__ up_codes,   const uint16_t* __restrict__ up_scales,   const float* __restrict__ up_cent,
    int expert_id, int N_inter, int K,
    uint16_t* __restrict__ out_gu_f16)
{
    __shared__ uint32_t s_tile[32][65];

    const int tid = int(threadIdx.x);
    const int r   = tid & 31;
    const int w   = tid >> 5;
    const int g   = int(blockIdx.x);
    const int n_blks = N_inter >> 5;
    const int by  = int(blockIdx.y);
    const bool is_up = (by >= n_blks);
    const int rb  = is_up ? (by - n_blks) : by;

    const uint32_t* codes  = is_up ? up_codes  : gate_codes;
    const uint16_t* scales = is_up ? up_scales : gate_scales;
    const float*    cent   = is_up ? up_cent   : gate_cent;

    const int k_words  = K >> 4;
    const int k_groups = K >> 7;
    const size_t e = size_t(expert_id);
    const size_t exp_code_stride  = size_t(n_blks) * size_t(k_words) * 32u;
    const size_t exp_scale_stride = size_t(n_blks) * size_t(k_groups) * 32u;

    const float c0 = cent[e * 2u + 0u];
    const float c1 = cent[e * 2u + 1u];
    const float sig = d_w93_f16_to_f32(scales[e * exp_scale_stride + (size_t(rb) * size_t(k_groups) + size_t(g)) * 32u + size_t(r)]);
    const float v0 = c0 * sig;
    const float v1 = c1 * sig;

    const uint32_t packed = codes[e * exp_code_stride + (size_t(rb) * size_t(k_words) + size_t(g * 8 + w)) * 32u + size_t(r)];

    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const uint32_t nib = (packed >> (4 * p)) & 0xFu;
        const float mag0 = (nib & 1u) ? v1 : v0;
        const float mag1 = (nib & 4u) ? v1 : v0;
        const float f0   = (nib & 2u) ? -mag0 : mag0;
        const float f1   = (nib & 8u) ? -mag1 : mag1;
        const uint32_t h0 = uint32_t(d_w93_f32_to_f16_clamped(f0));
        const uint32_t h1 = uint32_t(d_w93_f32_to_f16_clamped(f1));
        s_tile[r][w * 8 + p] = h0 | (h1 << 16);
    }
    __syncthreads();

    const int out_w = tid & 15;
    const int r_base = tid >> 4;
    uint4* dst_base = reinterpret_cast<uint4*>(out_gu_f16 + size_t(by * 32) * size_t(K) + size_t(g * 128));
    const size_t row_stride_u4 = size_t(K) >> 3;

    #pragma unroll
    for (int step = 0; step < 2; ++step) {
        const int rr = r_base + step * 16;
        uint4 v;
        v.x = s_tile[rr][out_w * 4 + 0];
        v.y = s_tile[rr][out_w * 4 + 1];
        v.z = s_tile[rr][out_w * 4 + 2];
        v.w = s_tile[rr][out_w * 4 + 3];
        dst_base[size_t(rr) * row_stride_u4 + size_t(out_w)] = v;
    }
}

// 5a. Vectorized Coalesced WHT128-HLUT3 Expert Matrix Unpack [N, K] -> Row-Major FP16 [N, K]
// Uses 16 threads/row x 16 rows (2 steps = 32x32 tile), packing 2 adjacent FP16 -> coalesced uint32_t stores (0 LDS, 0 sync).
__global__ __launch_bounds__(256, 8)
void k_w93_unpack_hlut3_expert_f16(
    const Hlut3Tile32x32* __restrict__ tiles,
    const float* __restrict__ levels,
    int expert_id,
    int N, int K,
    uint16_t* __restrict__ out_f16)
{
    const int tid    = int(threadIdx.x);
    const int lane16 = tid & 15;          // 0..15 -> pair of cols (2*lane16, 2*lane16+1)
    const int r0     = tid >> 4;          // 0..15
    const int kt     = int(blockIdx.x);   // 0 .. K/32 - 1
    const int nt     = int(blockIdx.y);   // 0 .. N/32 - 1

    const int nt_count = N >> 5;
    const int kt_count = K >> 5;
    const size_t tile_idx = (size_t(expert_id) * size_t(nt_count) + size_t(nt)) * size_t(kt_count) + size_t(kt);
    const Hlut3Tile32x32* __restrict__ tile = tiles + tile_idx;

    const float lv0 = levels[0];
    const float lv1 = levels[1];
    const float lv2 = levels[2];
    const float lv3 = levels[3];
    const int word_half = lane16 >> 3;           // 0 or 1
    const int mag_shift = (lane16 & 7) * 4;      // 0,4,8..28
    const int col2      = lane16 * 2;
    uint32_t* __restrict__ out_u32 = reinterpret_cast<uint32_t*>(out_f16);
    const size_t k_u32 = size_t(K >> 1);

    #pragma unroll
    for (int step = 0; step < 2; ++step) {
        const int lr = r0 + step * 16;
        const uint32_t mag_word  = tile->q_lo2[lr * 2 + word_half];
        const uint32_t sign_word = tile->q_hi1[lr];
        const float sigma = d_w93_f16_to_f32(tile->scales_f16[lr >> 2]);
        const float s0 = lv0 * sigma;
        const float s1 = lv1 * sigma;
        const float s2 = lv2 * sigma;
        const float s3 = lv3 * sigma;

        const uint32_t m0 = (mag_word >> mag_shift) & 3u;
        const uint32_t m1 = (mag_word >> (mag_shift + 2)) & 3u;
        const float a0 = (m0 == 0u) ? s0 : ((m0 == 1u) ? s1 : ((m0 == 2u) ? s2 : s3));
        const float a1 = (m1 == 0u) ? s0 : ((m1 == 1u) ? s1 : ((m1 == 2u) ? s2 : s3));
        const float v0 = ((sign_word >> col2) & 1u) ? -a0 : a0;
        const float v1 = ((sign_word >> (col2 + 1)) & 1u) ? -a1 : a1;
        const uint32_t h0 = uint32_t(d_w93_f32_to_f16_clamped(v0));
        const uint32_t h1 = uint32_t(d_w93_f32_to_f16_clamped(v1));
        out_u32[size_t(nt * 32 + lr) * k_u32 + size_t(kt * 16 + lane16)] = h0 | (h1 << 16);
    }
}

// 5b. Vectorized Fused single-launch Gate+Up WHT128-HLUT3 Unpack [2*N_inter, K] -> Row-Major FP16 [2*N_inter, K]
__global__ __launch_bounds__(256, 8)
void k_w93_unpack_hlut3_gateup_f16(
    const Hlut3Tile32x32* __restrict__ gate_tiles,
    const Hlut3Tile32x32* __restrict__ up_tiles,
    const float* __restrict__ levels,
    int expert_id, int N_inter, int K,
    uint16_t* __restrict__ out_gu_f16)
{
    const int tid    = int(threadIdx.x);
    const int lane16 = tid & 15;
    const int r0     = tid >> 4;
    const int kt     = int(blockIdx.x);
    const int nt_count = N_inter >> 5;
    const int by     = int(blockIdx.y);
    const bool is_up = (by >= nt_count);
    const int nt     = is_up ? (by - nt_count) : by;
    const int kt_count = K >> 5;

    const Hlut3Tile32x32* __restrict__ base_tiles = is_up ? up_tiles : gate_tiles;
    const size_t tile_idx = (size_t(expert_id) * size_t(nt_count) + size_t(nt)) * size_t(kt_count) + size_t(kt);
    const Hlut3Tile32x32* __restrict__ tile = base_tiles + tile_idx;

    const float lv0 = levels[0];
    const float lv1 = levels[1];
    const float lv2 = levels[2];
    const float lv3 = levels[3];
    const int word_half = lane16 >> 3;
    const int mag_shift = (lane16 & 7) * 4;
    const int col2      = lane16 * 2;
    uint32_t* __restrict__ out_u32 = reinterpret_cast<uint32_t*>(out_gu_f16);
    const size_t k_u32 = size_t(K >> 1);

    #pragma unroll
    for (int step = 0; step < 2; ++step) {
        const int lr = r0 + step * 16;
        const uint32_t mag_word  = tile->q_lo2[lr * 2 + word_half];
        const uint32_t sign_word = tile->q_hi1[lr];
        const float sigma = d_w93_f16_to_f32(tile->scales_f16[lr >> 2]);
        const float s0 = lv0 * sigma;
        const float s1 = lv1 * sigma;
        const float s2 = lv2 * sigma;
        const float s3 = lv3 * sigma;

        const uint32_t m0 = (mag_word >> mag_shift) & 3u;
        const uint32_t m1 = (mag_word >> (mag_shift + 2)) & 3u;
        const float a0 = (m0 == 0u) ? s0 : ((m0 == 1u) ? s1 : ((m0 == 2u) ? s2 : s3));
        const float a1 = (m1 == 0u) ? s0 : ((m1 == 1u) ? s1 : ((m1 == 2u) ? s2 : s3));
        const float v0 = ((sign_word >> col2) & 1u) ? -a0 : a0;
        const float v1 = ((sign_word >> (col2 + 1)) & 1u) ? -a1 : a1;
        const uint32_t h0 = uint32_t(d_w93_f32_to_f16_clamped(v0));
        const uint32_t h1 = uint32_t(d_w93_f32_to_f16_clamped(v1));
        out_u32[size_t(by * 32 + lr) * k_u32 + size_t(kt * 16 + lane16)] = h0 | (h1 << 16);
    }
}

// 6. Gather m_e token rows from d_x_rot_f16[M, 5120] -> contiguous d_gather_f16[m_e, 5120]
__global__ __launch_bounds__(256, 8)
void k_w93_gather_rows_f16(
    const uint16_t* __restrict__ x_rot_f16,
    const Mi50RoutedPair* __restrict__ pairs,
    int p_start,
    int m_e,
    uint16_t* __restrict__ gather_f16)
{
    const int r    = int(blockIdx.y);
    const int col4 = int(blockIdx.x) * 256 + int(threadIdx.x); // 0..639 uint4s (640 * 8 = 5120 FP16)
    if (col4 >= 640) return;
    uint4* dst = reinterpret_cast<uint4*>(gather_f16 + size_t(r) * 5120u);
    if (r >= m_e) {
        dst[col4] = make_uint4(0u, 0u, 0u, 0u);
        return;
    }
    const int src_row = pairs[p_start + r].row;
    const uint4* src = reinterpret_cast<const uint4*>(x_rot_f16 + size_t(src_row) * 5120u);
    dst[col4] = src[col4];
}

// 7. Fused Clamped-SwiGLU10 + WHT128(signs_i) -> FP16
__global__ __launch_bounds__(128, 8)
void k_w93_swiglu10_wht128_f32_to_f16(
    const float* __restrict__ gu_f32,
    const uint64_t* __restrict__ signs_i,
    uint16_t* __restrict__ zrot_f16,
    int N_inter)
{
    __shared__ float s_xwave[128];
    const int tid    = int(threadIdx.x);
    const int lane   = tid & 63;
    const int wid    = tid >> 6;
    const int blk128 = int(blockIdx.x);
    const int r      = int(blockIdx.y);
    const int col    = blk128 * 128 + tid;

    const size_t row_off = size_t(r) * size_t(2 * N_inter);
    const float g = gu_f32[row_off + size_t(col)];
    const float u = gu_f32[row_off + size_t(N_inter + col)];
    const float gc = fminf(g, 10.0f);
    const float uc = fmaxf(-10.0f, fminf(10.0f, u));
    float v = (gc / (1.0f + expf(-gc))) * uc;

    const uint64_t sw = signs_i[col >> 6];
    if ((sw >> (col & 63)) & 1ULL) v = -v;

    #pragma unroll
    for (int len = 1; len <= 32; len <<= 1) {
        const float partner = __shfl_xor(v, len, 64);
        v = (lane & len) ? (partner - v) : (v + partner);
    }
    s_xwave[tid] = v;
    __syncthreads();

    const float a = s_xwave[lane];
    const float b = s_xwave[64 + lane];
    const float res = ((wid == 0) ? (a + b) : (a - b)) * 0.08838834764831844655f;
    zrot_f16[size_t(r) * size_t(N_inter) + size_t(col)] = d_w93_f32_to_f16_clamped(res);
}

// 8. Zero-Atomic Coalesced float4 Weighted Scatter-Add: d_y_f32[m_e, 5120] -> d_out_accum_f32[pr.row, 5120]
// Within a single expert bucket [p_start, p_start + m_e), every pr.row is distinct (top-k chooses distinct experts/token).
__global__ __launch_bounds__(256, 8)
void k_w93_scatter_add_weighted_f32(
    const float* __restrict__ y_f32,
    const Mi50RoutedPair* __restrict__ pairs,
    int p_start,
    float* __restrict__ out_accum_f32)
{
    const int r    = int(blockIdx.y);
    const int col4 = int(blockIdx.x) * 256 + int(threadIdx.x); // 0..1279 float4s (1280 * 4 = 5120)
    if (col4 >= 1280) return;
    const Mi50RoutedPair pr = pairs[p_start + r];
    const float w = pr.weight;
    if (w == 0.0f) return;
    const float4 yv = reinterpret_cast<const float4*>(y_f32 + size_t(r) * 5120u)[col4];
    float4* dst = reinterpret_cast<float4*>(out_accum_f32 + size_t(pr.row) * 5120u);
    float4 av = dst[col4];
    av.x += yv.x * w;
    av.y += yv.y * w;
    av.z += yv.z * w;
    av.w += yv.w * w;
    dst[col4] = av;
}

// ---- Host Launchers ----
void launch_w93_unpack_fp8_to_f16(
    const uint8_t* d_fp8, const uint8_t* d_ue8m0, uint16_t* d_out_f16,
    int N, int K, bool per_row_scale, hipStream_t stream)
{
    if (N <= 0 || K <= 0 || (K & 31) != 0) return;
    hipLaunchKernelGGL(k_w93_unpack_fp8_to_f16,
                       dim3(unsigned(K >> 5), unsigned((N + 7) >> 3)), dim3(256), 0, stream,
                       d_fp8, d_ue8m0, d_out_f16, N, K, per_row_scale ? 1 : 0);
}

void launch_w93_bf16_to_f16(
    const uint16_t* d_in_bf16, uint16_t* d_out_f16, size_t n_elems, hipStream_t stream)
{
    if (n_elems == 0) return;
    hipLaunchKernelGGL(k_w93_bf16_to_f16,
                       dim3(unsigned((n_elems + 255u) / 256u)), dim3(256), 0, stream,
                       d_in_bf16, d_out_f16, n_elems);
}

void launch_w93_f32_to_bf16_rne(
    const float* d_in_f32, uint16_t* d_out_bf16, size_t n_elems, hipStream_t stream)
{
    if (n_elems == 0) return;
    hipLaunchKernelGGL(k_w93_f32_to_bf16_rne,
                       dim3(unsigned((n_elems + 255u) / 256u)), dim3(256), 0, stream,
                       d_in_f32, d_out_bf16, n_elems);
}

void launch_w93_unpack_hlut2_expert_f16(
    const uint32_t* d_codes, const uint16_t* d_scales, const float* d_cent,
    int expert_id, int N, int K, uint16_t* d_out_f16, hipStream_t stream)
{
    if (N <= 0 || K <= 0 || (N & 31) != 0 || (K & 127) != 0) return;
    hipLaunchKernelGGL(k_w93_unpack_hlut2_expert_f16,
                       dim3(unsigned(K >> 7), unsigned(N >> 5)), dim3(256), 0, stream,
                       d_codes, d_scales, d_cent, expert_id, N, K, d_out_f16);
}

void launch_w93_unpack_hlut2_gateup_f16(
    const uint32_t* d_gate_codes, const uint16_t* d_gate_scales, const float* d_gate_cent,
    const uint32_t* d_up_codes,   const uint16_t* d_up_scales,   const float* d_up_cent,
    int expert_id, int N_inter, int K, uint16_t* d_out_gu_f16, hipStream_t stream)
{
    if (N_inter <= 0 || K <= 0 || (N_inter & 31) != 0 || (K & 127) != 0) return;
    hipLaunchKernelGGL(k_w93_unpack_hlut2_gateup_f16,
                       dim3(unsigned(K >> 7), unsigned((2 * N_inter) >> 5)), dim3(256), 0, stream,
                       d_gate_codes, d_gate_scales, d_gate_cent,
                       d_up_codes, d_up_scales, d_up_cent,
                       expert_id, N_inter, K, d_out_gu_f16);
}

void launch_w93_unpack_hlut3_expert_f16(
    const Hlut3Tile32x32* d_tiles, const float* d_levels,
    int expert_id, int N, int K, uint16_t* d_out_f16, hipStream_t stream)
{
    if (N <= 0 || K <= 0 || (N & 31) != 0 || (K & 31) != 0) return;
    hipLaunchKernelGGL(k_w93_unpack_hlut3_expert_f16,
                       dim3(unsigned(K >> 5), unsigned(N >> 5)), dim3(256), 0, stream,
                       d_tiles, d_levels, expert_id, N, K, d_out_f16);
}

void launch_w93_unpack_hlut3_gateup_f16(
    const Hlut3Tile32x32* d_gate_tiles, const Hlut3Tile32x32* d_up_tiles,
    const float* d_levels, int expert_id, int N_inter, int K,
    uint16_t* d_out_gu_f16, hipStream_t stream)
{
    if (N_inter <= 0 || K <= 0 || (N_inter & 31) != 0 || (K & 31) != 0) return;
    hipLaunchKernelGGL(k_w93_unpack_hlut3_gateup_f16,
                       dim3(unsigned(K >> 5), unsigned((2 * N_inter) >> 5)), dim3(256), 0, stream,
                       d_gate_tiles, d_up_tiles, d_levels, expert_id, N_inter, K, d_out_gu_f16);
}

void launch_w93_gather_rows_f16(
    const uint16_t* d_x_rot_f16, const Mi50RoutedPair* d_pairs,
    int p_start, int m_e, uint16_t* d_gather_f16, hipStream_t stream, int m_pad)
{
    if (m_e <= 0) return;
    const int grid_y = (m_pad > m_e) ? m_pad : m_e;
    hipLaunchKernelGGL(k_w93_gather_rows_f16,
                       dim3(3u, unsigned(grid_y)), dim3(256), 0, stream,
                       d_x_rot_f16, d_pairs, p_start, m_e, d_gather_f16);
}

void launch_w93_swiglu10_wht128_f32_to_f16(
    const float* d_gu_f32, const uint64_t* d_signs_i,
    uint16_t* d_zrot_f16, int m_e, int N_inter, hipStream_t stream)
{
    if (m_e <= 0 || N_inter <= 0 || (N_inter & 127) != 0) return;
    hipLaunchKernelGGL(k_w93_swiglu10_wht128_f32_to_f16,
                       dim3(unsigned(N_inter >> 7), unsigned(m_e)), dim3(128), 0, stream,
                       d_gu_f32, d_signs_i, d_zrot_f16, N_inter);
}

void launch_w93_scatter_add_weighted_f32(
    const float* d_y_f32, const Mi50RoutedPair* d_pairs,
    int p_start, int m_e, float* d_out_accum_f32, hipStream_t stream)
{
    if (m_e <= 0) return;
    hipLaunchKernelGGL(k_w93_scatter_add_weighted_f32,
                       dim3(5u, unsigned(m_e)), dim3(256), 0, stream,
                       d_y_f32, d_pairs, p_start, d_out_accum_f32);
}

// ============================================================================
// Wave 9.4 Phase C: Zero-Scratch Batched HLUT2/HLUT3 Unpack + INT8/BF16 Wire
// ============================================================================

__global__ __launch_bounds__(256, 4)
void k_w94_unpack_hlut2_gateup_batch_f16(
    const uint32_t* __restrict__ gate_codes, const uint16_t* __restrict__ gate_scales, const float* __restrict__ gate_cent,
    const uint32_t* __restrict__ up_codes,   const uint16_t* __restrict__ up_scales,   const float* __restrict__ up_cent,
    const int32_t*  __restrict__ d_eids,
    int N_inter, int K,
    uint16_t* __restrict__ out_gu_f16)
{
    __shared__ uint32_t s_tile[32][65];

    const int tid = int(threadIdx.x);
    const int r   = tid & 31;
    const int w   = tid >> 5;
    const int g   = int(blockIdx.x);
    const int n_blks = N_inter >> 5;
    const int by  = int(blockIdx.y);
    const int bi  = int(blockIdx.z);
    const int expert_id = d_eids[bi];
    const bool is_up = (by >= n_blks);
    const int rb  = is_up ? (by - n_blks) : by;

    const uint32_t* codes  = is_up ? up_codes  : gate_codes;
    const uint16_t* scales = is_up ? up_scales : gate_scales;
    const float*    cent   = is_up ? up_cent   : gate_cent;

    const int k_words  = K >> 4;
    const int k_groups = K >> 7;
    const size_t e = size_t(expert_id);
    const size_t exp_code_stride  = size_t(n_blks) * size_t(k_words) * 32u;
    const size_t exp_scale_stride = size_t(n_blks) * size_t(k_groups) * 32u;

    const float c0 = cent[e * 2u + 0u];
    const float c1 = cent[e * 2u + 1u];
    const float sig = d_w93_f16_to_f32(scales[e * exp_scale_stride + (size_t(rb) * size_t(k_groups) + size_t(g)) * 32u + size_t(r)]);
    const float v0 = c0 * sig;
    const float v1 = c1 * sig;

    const uint32_t packed = codes[e * exp_code_stride + (size_t(rb) * size_t(k_words) + size_t(g * 8 + w)) * 32u + size_t(r)];

    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const uint32_t nib = (packed >> (4 * p)) & 0xFu;
        const float mag0 = (nib & 1u) ? v1 : v0;
        const float mag1 = (nib & 4u) ? v1 : v0;
        const float f0   = (nib & 2u) ? -mag0 : mag0;
        const float f1   = (nib & 8u) ? -mag1 : mag1;
        const uint32_t h0 = uint32_t(d_w93_f32_to_f16_clamped(f0));
        const uint32_t h1 = uint32_t(d_w93_f32_to_f16_clamped(f1));
        s_tile[r][w * 8 + p] = h0 | (h1 << 16);
    }
    __syncthreads();

    const int out_w  = tid & 15;
    const int r_base = tid >> 4;
    const size_t mat_stride = size_t(2 * N_inter) * size_t(K);
    uint4* dst_base = reinterpret_cast<uint4*>(out_gu_f16 + size_t(bi) * mat_stride + size_t(by * 32) * size_t(K) + size_t(g * 128));
    const size_t row_stride_u4 = size_t(K) >> 3;

    #pragma unroll
    for (int step = 0; step < 2; ++step) {
        const int rr = r_base + step * 16;
        uint4 v;
        v.x = s_tile[rr][out_w * 4 + 0];
        v.y = s_tile[rr][out_w * 4 + 1];
        v.z = s_tile[rr][out_w * 4 + 2];
        v.w = s_tile[rr][out_w * 4 + 3];
        dst_base[size_t(rr) * row_stride_u4 + size_t(out_w)] = v;
    }
}

__global__ __launch_bounds__(256, 4)
void k_w94_unpack_hlut2_expert_batch_f16(
    const uint32_t* __restrict__ codes,
    const uint16_t* __restrict__ scales,
    const float*    __restrict__ cent,
    const int32_t*  __restrict__ d_eids,
    int N, int K,
    uint16_t* __restrict__ out_f16)
{
    __shared__ uint32_t s_tile[32][65];

    const int tid = int(threadIdx.x);
    const int r   = tid & 31;
    const int w   = tid >> 5;
    const int g   = int(blockIdx.x);
    const int rb  = int(blockIdx.y);
    const int bi  = int(blockIdx.z);
    const int expert_id = d_eids[bi];

    const int n_blks   = N >> 5;
    const int k_words  = K >> 4;
    const int k_groups = K >> 7;
    const size_t e = size_t(expert_id);
    const size_t exp_code_stride  = size_t(n_blks) * size_t(k_words) * 32u;
    const size_t exp_scale_stride = size_t(n_blks) * size_t(k_groups) * 32u;

    const float c0 = cent[e * 2u + 0u];
    const float c1 = cent[e * 2u + 1u];
    const float sig = d_w93_f16_to_f32(scales[e * exp_scale_stride + (size_t(rb) * size_t(k_groups) + size_t(g)) * 32u + size_t(r)]);
    const float v0 = c0 * sig;
    const float v1 = c1 * sig;

    const uint32_t packed = codes[e * exp_code_stride + (size_t(rb) * size_t(k_words) + size_t(g * 8 + w)) * 32u + size_t(r)];

    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const uint32_t nib = (packed >> (4 * p)) & 0xFu;
        const float mag0 = (nib & 1u) ? v1 : v0;
        const float mag1 = (nib & 4u) ? v1 : v0;
        const float f0   = (nib & 2u) ? -mag0 : mag0;
        const float f1   = (nib & 8u) ? -mag1 : mag1;
        const uint32_t h0 = uint32_t(d_w93_f32_to_f16_clamped(f0));
        const uint32_t h1 = uint32_t(d_w93_f32_to_f16_clamped(f1));
        s_tile[r][w * 8 + p] = h0 | (h1 << 16);
    }
    __syncthreads();

    const int out_w  = tid & 15;
    const int r_base = tid >> 4;
    const size_t mat_stride = size_t(N) * size_t(K);
    uint4* dst_base = reinterpret_cast<uint4*>(out_f16 + size_t(bi) * mat_stride + size_t(rb * 32) * size_t(K) + size_t(g * 128));
    const size_t row_stride_u4 = size_t(K) >> 3;

    #pragma unroll
    for (int step = 0; step < 2; ++step) {
        const int rr = r_base + step * 16;
        uint4 v;
        v.x = s_tile[rr][out_w * 4 + 0];
        v.y = s_tile[rr][out_w * 4 + 1];
        v.z = s_tile[rr][out_w * 4 + 2];
        v.w = s_tile[rr][out_w * 4 + 3];
        dst_base[size_t(rr) * row_stride_u4 + size_t(out_w)] = v;
    }
}

__global__ __launch_bounds__(256, 8)
void k_w94_unpack_hlut3_gateup_batch_f16(
    const Hlut3Tile32x32* __restrict__ gate_tiles,
    const Hlut3Tile32x32* __restrict__ up_tiles,
    const float* __restrict__ levels,
    const int32_t* __restrict__ d_eids,
    int N_inter, int K,
    uint16_t* __restrict__ out_gu_f16)
{
    const int tid    = int(threadIdx.x);
    const int lane16 = tid & 15;
    const int r0     = tid >> 4;
    const int kt     = int(blockIdx.x);
    const int nt_count = N_inter >> 5;
    const int by     = int(blockIdx.y);
    const int bi     = int(blockIdx.z);
    const int expert_id = d_eids[bi];
    const bool is_up = (by >= nt_count);
    const int nt     = is_up ? (by - nt_count) : by;
    const int kt_count = K >> 5;

    const Hlut3Tile32x32* __restrict__ base_tiles = is_up ? up_tiles : gate_tiles;
    const size_t tile_idx = (size_t(expert_id) * size_t(nt_count) + size_t(nt)) * size_t(kt_count) + size_t(kt);
    const Hlut3Tile32x32* __restrict__ tile = base_tiles + tile_idx;

    const float lv0 = levels[0];
    const float lv1 = levels[1];
    const float lv2 = levels[2];
    const float lv3 = levels[3];
    const int word_half = lane16 >> 3;
    const int mag_shift = (lane16 & 7) * 4;
    const int col2      = lane16 * 2;
    const size_t mat_stride_u32 = size_t(2 * N_inter) * size_t(K >> 1);
    uint32_t* __restrict__ out_u32 = reinterpret_cast<uint32_t*>(out_gu_f16) + size_t(bi) * mat_stride_u32;
    const size_t k_u32 = size_t(K >> 1);

    #pragma unroll
    for (int step = 0; step < 2; ++step) {
        const int lr = r0 + step * 16;
        const uint32_t mag_word  = tile->q_lo2[lr * 2 + word_half];
        const uint32_t sign_word = tile->q_hi1[lr];
        const float sigma = d_w93_f16_to_f32(tile->scales_f16[lr >> 2]);
        const float s0 = lv0 * sigma;
        const float s1 = lv1 * sigma;
        const float s2 = lv2 * sigma;
        const float s3 = lv3 * sigma;

        const uint32_t m0 = (mag_word >> mag_shift) & 3u;
        const uint32_t m1 = (mag_word >> (mag_shift + 2)) & 3u;
        const float a0 = (m0 == 0u) ? s0 : ((m0 == 1u) ? s1 : ((m0 == 2u) ? s2 : s3));
        const float a1 = (m1 == 0u) ? s0 : ((m1 == 1u) ? s1 : ((m1 == 2u) ? s2 : s3));
        const float v0 = ((sign_word >> col2) & 1u) ? -a0 : a0;
        const float v1 = ((sign_word >> (col2 + 1)) & 1u) ? -a1 : a1;
        const uint32_t h0 = uint32_t(d_w93_f32_to_f16_clamped(v0));
        const uint32_t h1 = uint32_t(d_w93_f32_to_f16_clamped(v1));
        out_u32[size_t(by * 32 + lr) * k_u32 + size_t(kt * 16 + lane16)] = h0 | (h1 << 16);
    }
}

__global__ __launch_bounds__(256, 8)
void k_w94_unpack_hlut3_expert_batch_f16(
    const Hlut3Tile32x32* __restrict__ tiles,
    const float* __restrict__ levels,
    const int32_t* __restrict__ d_eids,
    int N, int K,
    uint16_t* __restrict__ out_f16)
{
    const int tid    = int(threadIdx.x);
    const int lane16 = tid & 15;
    const int r0     = tid >> 4;
    const int kt     = int(blockIdx.x);
    const int nt     = int(blockIdx.y);
    const int bi     = int(blockIdx.z);
    const int expert_id = d_eids[bi];

    const int nt_count = N >> 5;
    const int kt_count = K >> 5;
    const size_t tile_idx = (size_t(expert_id) * size_t(nt_count) + size_t(nt)) * size_t(kt_count) + size_t(kt);
    const Hlut3Tile32x32* __restrict__ tile = tiles + tile_idx;

    const float lv0 = levels[0];
    const float lv1 = levels[1];
    const float lv2 = levels[2];
    const float lv3 = levels[3];
    const int word_half = lane16 >> 3;
    const int mag_shift = (lane16 & 7) * 4;
    const int col2      = lane16 * 2;
    const size_t mat_stride_u32 = size_t(N) * size_t(K >> 1);
    uint32_t* __restrict__ out_u32 = reinterpret_cast<uint32_t*>(out_f16) + size_t(bi) * mat_stride_u32;
    const size_t k_u32 = size_t(K >> 1);

    #pragma unroll
    for (int step = 0; step < 2; ++step) {
        const int lr = r0 + step * 16;
        const uint32_t mag_word  = tile->q_lo2[lr * 2 + word_half];
        const uint32_t sign_word = tile->q_hi1[lr];
        const float sigma = d_w93_f16_to_f32(tile->scales_f16[lr >> 2]);
        const float s0 = lv0 * sigma;
        const float s1 = lv1 * sigma;
        const float s2 = lv2 * sigma;
        const float s3 = lv3 * sigma;

        const uint32_t m0 = (mag_word >> mag_shift) & 3u;
        const uint32_t m1 = (mag_word >> (mag_shift + 2)) & 3u;
        const float a0 = (m0 == 0u) ? s0 : ((m0 == 1u) ? s1 : ((m0 == 2u) ? s2 : s3));
        const float a1 = (m1 == 0u) ? s0 : ((m1 == 1u) ? s1 : ((m1 == 2u) ? s2 : s3));
        const float v0 = ((sign_word >> col2) & 1u) ? -a0 : a0;
        const float v1 = ((sign_word >> (col2 + 1)) & 1u) ? -a1 : a1;
        const uint32_t h0 = uint32_t(d_w93_f32_to_f16_clamped(v0));
        const uint32_t h1 = uint32_t(d_w93_f32_to_f16_clamped(v1));
        out_u32[size_t(nt * 32 + lr) * k_u32 + size_t(kt * 16 + lane16)] = h0 | (h1 << 16);
    }
}

// Per-128-element (per-WHT128-block) symmetric INT8 + FP16 scale quantization (1 block of 128 elems = 32 threads,
// 8 blocks/CTA = 256 threads; 0 LDS, pure warp-shuffle reduction, cos >= 0.99997 with 40 FP16 scales/row!)
__global__ __launch_bounds__(256, 8)
void k_w94_quant_f16_to_int8_rowwise(
    const uint16_t* __restrict__ in_f16,
    int8_t*         __restrict__ out_i8,
    uint16_t*       __restrict__ out_scales_f16,
    int M, int K)
{
    const int k_blks = K >> 7;  // 40 for K=5120
    const int tid    = int(threadIdx.x);
    const int lane32 = tid & 31;
    const int sub    = tid >> 5;  // 0..7 blocks per CTA
    const int gblk   = int(blockIdx.x) * 8 + sub;
    const int total_blks = M * k_blks;
    if (gblk >= total_blks) return;

    const int elem_off = gblk * 128 + lane32 * 4;
    const uint2 p = *reinterpret_cast<const uint2*>(in_f16 + size_t(elem_off));
    const float f0 = d_w93_f16_to_f32(uint16_t(p.x & 0xFFFFu));
    const float f1 = d_w93_f16_to_f32(uint16_t(p.x >> 16));
    const float f2 = d_w93_f16_to_f32(uint16_t(p.y & 0xFFFFu));
    const float f3 = d_w93_f16_to_f32(uint16_t(p.y >> 16));

    float bmax = fmaxf(fmaxf(fabsf(f0), fabsf(f1)), fmaxf(fabsf(f2), fabsf(f3)));
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        bmax = fmaxf(bmax, __shfl_xor(bmax, off, 32));
    }

    const float sc      = fmaxf(bmax * (1.0f / 127.0f), 1.0e-7f);
    const uint16_t sc16 = d_w93_f32_to_f16_clamped(sc);
    const float sc_eff  = fmaxf(d_w93_f16_to_f32(sc16), 1.0e-7f);
    const float inv_sc  = 1.0f / sc_eff;
    if (lane32 == 0) {
        out_scales_f16[gblk] = sc16;
    }

    const int q0 = max(-127, min(127, __float2int_rn(f0 * inv_sc)));
    const int q1 = max(-127, min(127, __float2int_rn(f1 * inv_sc)));
    const int q2 = max(-127, min(127, __float2int_rn(f2 * inv_sc)));
    const int q3 = max(-127, min(127, __float2int_rn(f3 * inv_sc)));
    *reinterpret_cast<uint32_t*>(out_i8 + size_t(elem_off)) =
        (uint32_t(q0 & 0xFF) << 0)  |
        (uint32_t(q1 & 0xFF) << 8)  |
        (uint32_t(q2 & 0xFF) << 16) |
        (uint32_t(q3 & 0xFF) << 24);
}

// Per-128-element symmetric INT8 + FP16 scale -> FP16 dequantization (0 LDS, 0 sync, coalesced uint32 -> uint2!)
__global__ __launch_bounds__(256, 8)
void k_w94_dequant_int8_to_f16_rowwise(
    const int8_t*   __restrict__ in_i8,
    const uint16_t* __restrict__ in_scales_f16,
    uint16_t*       __restrict__ out_f16,
    int M, int K)
{
    const int k_blks = K >> 7;
    const int tid    = int(threadIdx.x);
    const int lane32 = tid & 31;
    const int sub    = tid >> 5;
    const int gblk   = int(blockIdx.x) * 8 + sub;
    if (gblk >= M * k_blks) return;

    const float sc = d_w93_f16_to_f32(in_scales_f16[gblk]);
    const int elem_off = gblk * 128 + lane32 * 4;
    const uint32_t pk = *reinterpret_cast<const uint32_t*>(in_i8 + size_t(elem_off));
    const float f0 = float(int8_t(pk & 0xFFu))         * sc;
    const float f1 = float(int8_t((pk >> 8) & 0xFFu))  * sc;
    const float f2 = float(int8_t((pk >> 16) & 0xFFu)) * sc;
    const float f3 = float(int8_t((pk >> 24) & 0xFFu)) * sc;
    uint2 out_v;
    out_v.x = uint32_t(d_w93_f32_to_f16_clamped(f0)) | (uint32_t(d_w93_f32_to_f16_clamped(f1)) << 16);
    out_v.y = uint32_t(d_w93_f32_to_f16_clamped(f2)) | (uint32_t(d_w93_f32_to_f16_clamped(f3)) << 16);
    *reinterpret_cast<uint2*>(out_f16 + size_t(elem_off)) = out_v;
}

// In-place add BF16 vector onto FP32 accumulator (coalesced uint2 -> float4 for 2x smaller dev3 output wire!)
__global__ __launch_bounds__(256, 8)
void k_w94_add_bf16_to_f32_inplace(
    float*          __restrict__ dst_f32,
    const uint16_t* __restrict__ src_bf16,
    size_t n_elems)
{
    const size_t idx4 = (size_t(blockIdx.x) * 256u + size_t(threadIdx.x)) * 4u;
    if (idx4 + 3u < n_elems) {
        const uint2 p = *reinterpret_cast<const uint2*>(src_bf16 + idx4);
        float4* d4 = reinterpret_cast<float4*>(dst_f32 + idx4);
        float4 v = *d4;
        v.x += d_w93_bf16_to_f32(uint16_t(p.x & 0xFFFFu));
        v.y += d_w93_bf16_to_f32(uint16_t(p.x >> 16));
        v.z += d_w93_bf16_to_f32(uint16_t(p.y & 0xFFFFu));
        v.w += d_w93_bf16_to_f32(uint16_t(p.y >> 16));
        *d4 = v;
    } else {
        for (size_t i = idx4; i < n_elems && i < idx4 + 4u; ++i) {
            dst_f32[i] += d_w93_bf16_to_f32(src_bf16[i]);
        }
    }
}

void launch_w94_unpack_hlut2_gateup_batch_f16(
    const uint32_t* d_gate_codes, const uint16_t* d_gate_scales, const float* d_gate_cent,
    const uint32_t* d_up_codes,   const uint16_t* d_up_scales,   const float* d_up_cent,
    const int32_t* d_eids, int batch_b, int N_inter, int K,
    uint16_t* d_out_gu_f16, hipStream_t stream)
{
    if (batch_b <= 0 || N_inter <= 0 || K <= 0 || (N_inter & 31) != 0 || (K & 127) != 0) return;
    hipLaunchKernelGGL(k_w94_unpack_hlut2_gateup_batch_f16,
                       dim3(unsigned(K >> 7), unsigned((2 * N_inter) >> 5), unsigned(batch_b)), dim3(256), 0, stream,
                       d_gate_codes, d_gate_scales, d_gate_cent,
                       d_up_codes, d_up_scales, d_up_cent,
                       d_eids, N_inter, K, d_out_gu_f16);
}

void launch_w94_unpack_hlut2_expert_batch_f16(
    const uint32_t* d_codes, const uint16_t* d_scales, const float* d_cent,
    const int32_t* d_eids, int batch_b, int N, int K,
    uint16_t* d_out_f16, hipStream_t stream)
{
    if (batch_b <= 0 || N <= 0 || K <= 0 || (N & 31) != 0 || (K & 127) != 0) return;
    hipLaunchKernelGGL(k_w94_unpack_hlut2_expert_batch_f16,
                       dim3(unsigned(K >> 7), unsigned(N >> 5), unsigned(batch_b)), dim3(256), 0, stream,
                       d_codes, d_scales, d_cent, d_eids, N, K, d_out_f16);
}

void launch_w94_unpack_hlut3_gateup_batch_f16(
    const Hlut3Tile32x32* d_gate_tiles, const Hlut3Tile32x32* d_up_tiles,
    const float* d_levels, const int32_t* d_eids, int batch_b, int N_inter, int K,
    uint16_t* d_out_gu_f16, hipStream_t stream)
{
    if (batch_b <= 0 || N_inter <= 0 || K <= 0 || (N_inter & 31) != 0 || (K & 31) != 0) return;
    hipLaunchKernelGGL(k_w94_unpack_hlut3_gateup_batch_f16,
                       dim3(unsigned(K >> 5), unsigned((2 * N_inter) >> 5), unsigned(batch_b)), dim3(256), 0, stream,
                       d_gate_tiles, d_up_tiles, d_levels, d_eids, N_inter, K, d_out_gu_f16);
}

void launch_w94_unpack_hlut3_expert_batch_f16(
    const Hlut3Tile32x32* d_tiles, const float* d_levels,
    const int32_t* d_eids, int batch_b, int N, int K,
    uint16_t* d_out_f16, hipStream_t stream)
{
    if (batch_b <= 0 || N <= 0 || K <= 0 || (N & 31) != 0 || (K & 31) != 0) return;
    hipLaunchKernelGGL(k_w94_unpack_hlut3_expert_batch_f16,
                       dim3(unsigned(K >> 5), unsigned(N >> 5), unsigned(batch_b)), dim3(256), 0, stream,
                       d_tiles, d_levels, d_eids, N, K, d_out_f16);
}

void launch_w94_quant_f16_to_int8_rowwise(
    const uint16_t* d_in_f16, int8_t* d_out_i8, uint16_t* d_out_scales_f16,
    int M, int K, hipStream_t stream)
{
    if (M <= 0 || K <= 0 || (K & 127) != 0) return;
    const int total_blks = M * (K >> 7);
    hipLaunchKernelGGL(k_w94_quant_f16_to_int8_rowwise,
                       dim3(unsigned((total_blks + 7) / 8)), dim3(256), 0, stream,
                       d_in_f16, d_out_i8, d_out_scales_f16, M, K);
}

void launch_w94_dequant_int8_to_f16_rowwise(
    const int8_t* d_in_i8, const uint16_t* d_in_scales_f16, uint16_t* d_out_f16,
    int M, int K, hipStream_t stream)
{
    if (M <= 0 || K <= 0 || (K & 127) != 0) return;
    const int total_blks = M * (K >> 7);
    hipLaunchKernelGGL(k_w94_dequant_int8_to_f16_rowwise,
                       dim3(unsigned((total_blks + 7) / 8)), dim3(256), 0, stream,
                       d_in_i8, d_in_scales_f16, d_out_f16, M, K);
}

void launch_w94_add_bf16_to_f32_inplace(
    float* d_dst_f32, const uint16_t* d_src_bf16, size_t n_elems, hipStream_t stream)
{
    if (n_elems == 0) return;
    const size_t n4 = (n_elems + 3u) >> 2;
    hipLaunchKernelGGL(k_w94_add_bf16_to_f32_inplace,
                       dim3(unsigned((n4 + 255u) / 256u)), dim3(256), 0, stream,
                       d_dst_f32, d_src_bf16, n_elems);
}

// ---- Wave 9.4 Phase C.2: HHHS (FP16-out / FP32-compute) Zero-Scratch Kernels ----

__global__ __launch_bounds__(256, 8)
void k_w94_f16_to_bf16_rne(
    const uint16_t* __restrict__ in_f16,
    uint16_t*       __restrict__ out_bf16,
    size_t n_elems)
{
    const size_t idx4 = (size_t(blockIdx.x) * 256u + size_t(threadIdx.x)) * 4u;
    if (idx4 + 3u < n_elems) {
        const uint2 p = *reinterpret_cast<const uint2*>(in_f16 + idx4);
        uint2 o;
        const uint32_t b0 = uint32_t(d_w93_f32_to_bf16_rne(d_w93_f16_to_f32(uint16_t(p.x & 0xFFFFu))));
        const uint32_t b1 = uint32_t(d_w93_f32_to_bf16_rne(d_w93_f16_to_f32(uint16_t(p.x >> 16))));
        const uint32_t b2 = uint32_t(d_w93_f32_to_bf16_rne(d_w93_f16_to_f32(uint16_t(p.y & 0xFFFFu))));
        const uint32_t b3 = uint32_t(d_w93_f32_to_bf16_rne(d_w93_f16_to_f32(uint16_t(p.y >> 16))));
        o.x = b0 | (b1 << 16);
        o.y = b2 | (b3 << 16);
        *reinterpret_cast<uint2*>(out_bf16 + idx4) = o;
    } else {
        for (size_t i = idx4; i < n_elems && i < idx4 + 4u; ++i) {
            out_bf16[i] = d_w93_f32_to_bf16_rne(d_w93_f16_to_f32(in_f16[i]));
        }
    }
}

__global__ __launch_bounds__(128, 8)
void k_w94_swiglu10_wht128_f16_to_f16(
    const uint16_t* __restrict__ gu_f16,
    const uint64_t* __restrict__ signs_i,
    uint16_t*       __restrict__ zrot_f16,
    int N_inter)
{
    __shared__ float s_xwave[128];
    const int tid    = int(threadIdx.x);
    const int lane   = tid & 63;
    const int wid    = tid >> 6;
    const int blk128 = int(blockIdx.x);
    const int r      = int(blockIdx.y);
    const int col    = blk128 * 128 + tid;

    const size_t row_off = size_t(r) * size_t(2 * N_inter);
    const float g = d_w93_f16_to_f32(gu_f16[row_off + size_t(col)]);
    const float u = d_w93_f16_to_f32(gu_f16[row_off + size_t(N_inter + col)]);
    const float gc = fminf(g, 10.0f);
    const float uc = fmaxf(-10.0f, fminf(10.0f, u));
    float v = (gc / (1.0f + expf(-gc))) * uc;

    const uint64_t sw = signs_i[col >> 6];
    if ((sw >> (col & 63)) & 1ULL) v = -v;

    #pragma unroll
    for (int len = 1; len <= 32; len <<= 1) {
        const float partner = __shfl_xor(v, len, 64);
        v = (lane & len) ? (partner - v) : (v + partner);
    }
    s_xwave[tid] = v;
    __syncthreads();

    const float a = s_xwave[lane];
    const float b = s_xwave[64 + lane];
    const float res = ((wid == 0) ? (a + b) : (a - b)) * 0.08838834764831844655f;
    zrot_f16[size_t(r) * size_t(N_inter) + size_t(col)] = d_w93_f32_to_f16_clamped(res);
}

__global__ __launch_bounds__(256, 8)
void k_w94_scatter_add_weighted_f16(
    const uint16_t*       __restrict__ y_f16,
    const Mi50RoutedPair* __restrict__ pairs,
    int p_start,
    float*                __restrict__ out_accum_f32)
{
    const int r    = int(blockIdx.y);
    const int col4 = int(blockIdx.x) * 256 + int(threadIdx.x); // 0..1279 quads (1280 * 4 = 5120)
    if (col4 >= 1280) return;
    const Mi50RoutedPair pr = pairs[p_start + r];
    const float w = pr.weight;
    if (w == 0.0f) return;
    const uint2 yp = reinterpret_cast<const uint2*>(y_f16 + size_t(r) * 5120u)[col4];
    float4* dst = reinterpret_cast<float4*>(out_accum_f32 + size_t(pr.row) * 5120u);
    float4 av = dst[col4];
    av.x += d_w93_f16_to_f32(uint16_t(yp.x & 0xFFFFu)) * w;
    av.y += d_w93_f16_to_f32(uint16_t(yp.x >> 16))     * w;
    av.z += d_w93_f16_to_f32(uint16_t(yp.y & 0xFFFFu)) * w;
    av.w += d_w93_f16_to_f32(uint16_t(yp.y >> 16))     * w;
    dst[col4] = av;
}

__global__ __launch_bounds__(256, 8)
void k_w94_gather_padded_batch_f16(
    const uint16_t*       __restrict__ x_rot_f16,
    const Mi50RoutedPair* __restrict__ pairs,
    const int32_t*        __restrict__ d_pstarts,
    const int32_t*        __restrict__ d_mes,
    int m_pad,
    uint16_t*             __restrict__ gather_f16)
{
    const int bi   = int(blockIdx.z);
    const int r    = int(blockIdx.y);
    const int col4 = int(blockIdx.x) * 256 + int(threadIdx.x); // 0..639 uint4s (640 * 8 = 5120)
    if (col4 >= 640) return;
    uint4* dst = reinterpret_cast<uint4*>(gather_f16 + (size_t(bi) * size_t(m_pad) + size_t(r)) * 5120u);
    const int me = d_mes[bi];
    if (r >= me) {
        dst[col4] = make_uint4(0u, 0u, 0u, 0u);
        return;
    }
    const int src_row = pairs[d_pstarts[bi] + r].row;
    const uint4* src = reinterpret_cast<const uint4*>(x_rot_f16 + size_t(src_row) * 5120u);
    dst[col4] = src[col4];
}

__global__ __launch_bounds__(256, 8)
void k_w10_f16_to_f32(
    const uint16_t* __restrict__ in_f16,
    float* __restrict__ out_f32,
    size_t n_elems)
{
    const size_t i4 = size_t(blockIdx.x) * 256u + size_t(threadIdx.x);
    const size_t base = i4 << 2;
    if (base + 4u <= n_elems) {
        const uint2 hv = reinterpret_cast<const uint2*>(in_f16)[i4];
        float4 fv;
        fv.x = d_w93_f16_to_f32(uint16_t(hv.x & 0xFFFFu));
        fv.y = d_w93_f16_to_f32(uint16_t(hv.x >> 16));
        fv.z = d_w93_f16_to_f32(uint16_t(hv.y & 0xFFFFu));
        fv.w = d_w93_f16_to_f32(uint16_t(hv.y >> 16));
        reinterpret_cast<float4*>(out_f32)[i4] = fv;
    } else if (base < n_elems) {
        for (size_t k = base; k < n_elems; ++k) {
            out_f32[k] = d_w93_f16_to_f32(in_f16[k]);
        }
    }
}

void launch_w94_f16_to_bf16_rne(
    const uint16_t* d_in_f16, uint16_t* d_out_bf16, size_t n_elems, hipStream_t stream)
{
    if (n_elems == 0) return;
    const size_t n4 = (n_elems + 3u) >> 2;
    hipLaunchKernelGGL(k_w94_f16_to_bf16_rne,
                       dim3(unsigned((n4 + 255u) / 256u)), dim3(256), 0, stream,
                       d_in_f16, d_out_bf16, n_elems);
}

__global__ __launch_bounds__(256, 8)
void k_w10_add_3bf16_to_f32_and_bf16(
    const float*    __restrict__ base_f32,
    const uint16_t* __restrict__ add0_bf16,
    const uint16_t* __restrict__ add1_bf16,
    const uint16_t* __restrict__ add2_bf16,
    uint16_t*       __restrict__ out_bf16,
    size_t n_elems)
{
    const size_t i4 = size_t(blockIdx.x) * 256u + size_t(threadIdx.x);
    const size_t base = i4 << 2;
    if (base + 4u <= n_elems) {
        float4 v = reinterpret_cast<const float4*>(base_f32)[i4];
        const size_t w0 = i4 << 1;
        // System-scope relaxed loads (pure read-only global_load_dword glc, 0 RMW atomics)
        // bypass stale gfx906 L2 cache lines for host-written mapped /dev/shm & coherent offload buffers!
        if (add0_bf16 != nullptr) {
            const uint32_t* p32 = reinterpret_cast<const uint32_t*>(add0_bf16);
            const uint32_t px = __hip_atomic_load(&p32[w0 + 0u], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            const uint32_t py = __hip_atomic_load(&p32[w0 + 1u], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            v.x += d_w93_bf16_to_f32(uint16_t(px & 0xFFFFu));
            v.y += d_w93_bf16_to_f32(uint16_t(px >> 16));
            v.z += d_w93_bf16_to_f32(uint16_t(py & 0xFFFFu));
            v.w += d_w93_bf16_to_f32(uint16_t(py >> 16));
        }
        if (add1_bf16 != nullptr) {
            const uint32_t* p32 = reinterpret_cast<const uint32_t*>(add1_bf16);
            const uint32_t px = __hip_atomic_load(&p32[w0 + 0u], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            const uint32_t py = __hip_atomic_load(&p32[w0 + 1u], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            v.x += d_w93_bf16_to_f32(uint16_t(px & 0xFFFFu));
            v.y += d_w93_bf16_to_f32(uint16_t(px >> 16));
            v.z += d_w93_bf16_to_f32(uint16_t(py & 0xFFFFu));
            v.w += d_w93_bf16_to_f32(uint16_t(py >> 16));
        }
        if (add2_bf16 != nullptr) {
            const uint32_t* p32 = reinterpret_cast<const uint32_t*>(add2_bf16);
            const uint32_t px = __hip_atomic_load(&p32[w0 + 0u], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            const uint32_t py = __hip_atomic_load(&p32[w0 + 1u], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            v.x += d_w93_bf16_to_f32(uint16_t(px & 0xFFFFu));
            v.y += d_w93_bf16_to_f32(uint16_t(px >> 16));
            v.z += d_w93_bf16_to_f32(uint16_t(py & 0xFFFFu));
            v.w += d_w93_bf16_to_f32(uint16_t(py >> 16));
        }
        uint2 out;
        out.x = uint32_t(d_w93_f32_to_bf16_rne(v.x)) | (uint32_t(d_w93_f32_to_bf16_rne(v.y)) << 16);
        out.y = uint32_t(d_w93_f32_to_bf16_rne(v.z)) | (uint32_t(d_w93_f32_to_bf16_rne(v.w)) << 16);
        reinterpret_cast<uint2*>(out_bf16)[i4] = out;
    } else if (base < n_elems) {
        for (size_t k = base; k < n_elems; ++k) {
            float v = base_f32[k];
            if (add0_bf16 != nullptr) v += d_w93_bf16_to_f32(__hip_atomic_load(&add0_bf16[k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
            if (add1_bf16 != nullptr) v += d_w93_bf16_to_f32(__hip_atomic_load(&add1_bf16[k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
            if (add2_bf16 != nullptr) v += d_w93_bf16_to_f32(__hip_atomic_load(&add2_bf16[k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
            out_bf16[k] = d_w93_f32_to_bf16_rne(v);
        }
    }
}

void launch_w10_f16_to_f32(
    const uint16_t* d_in_f16, float* d_out_f32, size_t n_elems, hipStream_t stream)
{
    if (n_elems == 0) return;
    const size_t n4 = (n_elems + 3u) >> 2;
    hipLaunchKernelGGL(k_w10_f16_to_f32,
                       dim3(unsigned((n4 + 255u) / 256u)), dim3(256), 0, stream,
                       d_in_f16, d_out_f32, n_elems);
}

void launch_w10_add_3bf16_to_f32_and_bf16(
    const float* d_base_f32,
    const uint16_t* d_add0_bf16,
    const uint16_t* d_add1_bf16,
    const uint16_t* d_add2_bf16,
    uint16_t* d_out_bf16,
    size_t n_elems,
    hipStream_t stream)
{
    if (n_elems == 0) return;
    const size_t n4 = (n_elems + 3u) >> 2;
    hipLaunchKernelGGL(k_w10_add_3bf16_to_f32_and_bf16,
                       dim3(unsigned((n4 + 255u) / 256u)), dim3(256), 0, stream,
                       d_base_f32, d_add0_bf16, d_add1_bf16, d_add2_bf16, d_out_bf16, n_elems);
}

void launch_w94_swiglu10_wht128_f16_to_f16(
    const uint16_t* d_gu_f16, const uint64_t* d_signs_i,
    uint16_t* d_zrot_f16, int m_rows, int N_inter, hipStream_t stream)
{
    if (m_rows <= 0 || N_inter <= 0 || (N_inter & 127) != 0) return;
    hipLaunchKernelGGL(k_w94_swiglu10_wht128_f16_to_f16,
                       dim3(unsigned(N_inter >> 7), unsigned(m_rows)), dim3(128), 0, stream,
                       d_gu_f16, d_signs_i, d_zrot_f16, N_inter);
}

void launch_w94_scatter_add_weighted_f16(
    const uint16_t* d_y_f16, const Mi50RoutedPair* d_pairs,
    int p_start, int m_e, float* d_out_accum_f32, hipStream_t stream)
{
    if (m_e <= 0) return;
    hipLaunchKernelGGL(k_w94_scatter_add_weighted_f16,
                       dim3(5u, unsigned(m_e)), dim3(256), 0, stream,
                       d_y_f16, d_pairs, p_start, d_out_accum_f32);
}

void launch_w94_gather_padded_batch_f16(
    const uint16_t* d_x_rot_f16, const Mi50RoutedPair* d_pairs,
    const int32_t* d_pstarts, const int32_t* d_mes,
    int batch_b, int m_pad, uint16_t* d_gather_f16, hipStream_t stream)
{
    if (batch_b <= 0 || m_pad <= 0) return;
    hipLaunchKernelGGL(k_w94_gather_padded_batch_f16,
                       dim3(3u, unsigned(m_pad), unsigned(batch_b)), dim3(256), 0, stream,
                       d_x_rot_f16, d_pairs, d_pstarts, d_mes, m_pad, d_gather_f16);
}

}  // namespace dsv41
