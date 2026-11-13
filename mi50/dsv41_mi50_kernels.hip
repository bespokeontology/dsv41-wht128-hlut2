// Pure-C++17 / AMD HIP (gfx906) Full Kernel Suite for DeepSeek-V4.1-Flash-1017 on 3x MI50 (96 GiB HBM2)
// Zero Python anywhere. Engineered for ScratchSize == 0 across every single kernel.

#include "dsv41_mi50_kernels.hpp"
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace dsv41 {

// Exact FP64-angle register RoPE / YaRN inverse frequencies + Dumitrescu Fn 7 / Fn 28 zero-trig orthogonal Givens tables
// (58.5 KiB total in __constant__ memory; external symbol linkage for HIP runtime; covers pos 0..4,194,303 with 0 device trig!)
__constant__ double c_inv_freq_win_f64[32];
__constant__ double c_inv_freq_comp_f64[32];
__constant__ double c_rope_d0_cos[2][16][32];
__constant__ double c_rope_d0_sin[2][16][32];
__constant__ double c_rope_dk_cos[2][6][7][32];
__constant__ double c_rope_dk_sin[2][6][7][32];

namespace {

using f16x2_t = _Float16 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float d_bf16_to_f32(uint16_t b) {
    return __uint_as_float(uint32_t(b) << 16);
}

__device__ __forceinline__ uint16_t d_f32_to_bf16_rne(float x) {
    uint32_t u = __float_as_uint(x);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return uint16_t((u >> 16) | 0x0040u);
    u += 0x00007FFFu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

__device__ __forceinline__ uint16_t d_f32_to_fp16_bits(float x) {
    _Float16 h = static_cast<_Float16>(x);
    uint16_t u = 0;
    __builtin_memcpy(&u, &h, 2);
    return u;
}

__device__ __forceinline__ float d_fp16_bits_to_f32(uint16_t u) {
    _Float16 h;
    __builtin_memcpy(&h, &u, 2);
    return static_cast<float>(h);
}

// Exact UE8M0 -> FP32 (2^(b - 127), handling b==0 -> 2^-127 IEEE subnormal 0x00400000)
__device__ __forceinline__ float d_ue8m0_to_f32(uint8_t b) {
    if (b == 255u) return 0.0f;
    const uint32_t bits = (b == 0u) ? 0x00400000u : (uint32_t(b) << 23);
    return __uint_as_float(bits);
}

// Exact kernel.py:22-37 fast_round_scale(amax, max_inv) -> (s_f32, ue8m0_byte)
__device__ __forceinline__ float d_fast_round_scale_ue8m0(float amax, float max_inv, uint8_t& out_ue8) {
    const float prod = amax * max_inv;
    const uint32_t bits = __float_as_uint(prod);
    int exp_x = int((bits >> 23) & 0xFFu) - 127 + (((bits & 0x007FFFFFu) != 0u) ? 1 : 0);
    exp_x = max(-127, min(127, exp_x));
    out_ue8 = uint8_t(exp_x + 127);
    return d_ue8m0_to_f32(out_ue8);
}

// Lossless branchless FP8_E4M3 (float8_e4m3fn) -> IEEE FP16 bit pattern!
// Normal (1<=e<=15): exp_fp16 = e + 8, mant_fp16 = m << 7
// Subnormal (e==0, m=0..7): packed high bytes in 0x232221201E1C1800ULL!
__device__ __forceinline__ uint16_t d_fp8_e4m3_to_fp16_bits(uint8_t b) {
    const uint32_t s = uint32_t(b & 0x80u) << 8;
    const uint32_t em = uint32_t(b & 0x7Fu);
    if (em == 0x7Fu) return uint16_t(s);  // E4M3 NaN -> signed 0 guard
    const uint32_t e = em >> 3;
    const uint32_t m = em & 7u;
    const uint32_t norm_mag = ((e + 8u) << 10) | (m << 7);
    const uint32_t sub_mag = uint32_t((0x232221201E1C1800ULL >> (m * 8u)) & 0xFFULL) << 8;
    return uint16_t(s | ((e == 0u) ? sub_mag : norm_mag));
}

__device__ __forceinline__ f16x2_t d_fp8x2_to_f16x2(uint16_t b01) {
    const uint32_t lo = d_fp8_e4m3_to_fp16_bits(uint8_t(b01 & 0xFFu));
    const uint32_t hi = d_fp8_e4m3_to_fp16_bits(uint8_t(b01 >> 8));
    const uint32_t pack = lo | (hi << 16);
    f16x2_t res;
    __builtin_memcpy(&res, &pack, 4);
    return res;
}

__device__ __forceinline__ uint32_t d_lut_fp8x2_to_u32(const uint16_t* s_lut, uint16_t b01) {
    return uint32_t(s_lut[b01 & 0xFFu]) | (uint32_t(s_lut[b01 >> 8]) << 16);
}

__device__ __forceinline__ f16x2_t d_lut_fp8x2_to_f16x2(const uint16_t* s_lut, uint16_t b01) {
    const uint32_t pack = d_lut_fp8x2_to_u32(s_lut, b01);
    f16x2_t res;
    __builtin_memcpy(&res, &pack, 4);
    return res;
}

__device__ __forceinline__ float d_fp8_e4m3_to_f32(uint8_t b) {
    return d_fp16_bits_to_f32(d_fp8_e4m3_to_fp16_bits(b));
}

// Exact FP32 -> float8_e4m3fn (round-to-nearest-even, clamped to [-448, 448])
__device__ __forceinline__ uint8_t d_f32_to_fp8_e4m3(float x) {
    if (!(x == x)) return 0x7Fu;
    const uint32_t sign = (__float_as_uint(x) >> 24) & 0x80u;
    float ax = fabsf(x);
    if (ax >= 448.0f) return uint8_t(sign | 0x7Eu);  // e=15, m=6 -> 448.0
    if (ax < (1.0f / 1024.0f)) return uint8_t(sign); // < 0.5 * 2^-9 -> 0
    if (ax < (1.0f / 64.0f)) {
        // Subnormal range: ax * 512 in [0.5, 8.0), RNE to integer m in 0..8 (m==8 -> e=1, m=0 = 0x08!)
        const float scaled = ax * 512.0f;
        int m = __float2int_rn(scaled);
        return uint8_t(sign | uint32_t(m));
    }
    const uint32_t u = __float_as_uint(ax);
    int e32 = int((u >> 23) & 0xFFu) - 127;
    const uint32_t mant23 = u & 0x007FFFFFu;
    // Round 23-bit mantissa down to 3 bits (shift right by 20) with RNE:
    const uint32_t m3_pre = mant23 >> 20;
    const uint32_t rem = mant23 & 0x000FFFFFu;
    uint32_t m3 = m3_pre + ((rem > 0x00080000u || (rem == 0x00080000u && (m3_pre & 1u))) ? 1u : 0u);
    int e4 = e32 + 7;
    if (m3 == 8u) {
        m3 = 0u;
        ++e4;
    }
    if (e4 > 15 || (e4 == 15 && m3 >= 7u)) return uint8_t(sign | 0x7Eu);
    return uint8_t(sign | (uint32_t(e4) << 3) | m3);
}

// Exact FP32 -> FP4_E2M1 (round-to-nearest-even on {0, 0.5, 1, 1.5, 2, 3, 4, 6})
__device__ __forceinline__ uint8_t d_f32_to_fp4_e2m1(float x) {
    const uint8_t sign = (__float_as_uint(x) >> 28) & 0x8u;
    const float a = fabsf(x);
    uint8_t mag = 0u;
    if (a <= 0.25f)      mag = 0u;
    else if (a < 0.75f)  mag = 1u;
    else if (a <= 1.25f) mag = 2u;
    else if (a < 1.75f)  mag = 3u;
    else if (a <= 2.5f)  mag = 4u;
    else if (a < 3.5f)   mag = 5u;
    else if (a <= 5.0f)  mag = 6u;
    else                 mag = 7u;
    return sign | mag;
}

__device__ __forceinline__ float d_fp4_e2m1_to_f32(uint8_t nib) {
    // 8 positive values * 2 = {0, 1, 2, 3, 4, 6, 8, 12} packed in 4-bit nibbles of 0xC8643210u!
    const uint32_t twice = (0xC8643210u >> ((nib & 7u) * 4u)) & 0xFu;
    const float v = float(twice) * 0.5f;
    return (nib & 8u) ? -v : v;
}

__device__ __forceinline__ float d_swiglu10(float g, float u) {
    const float gc = fminf(g, 10.0f);
    const float uc = fmaxf(-10.0f, fminf(10.0f, u));
    return (gc / (1.0f + __expf(-gc))) * uc;
}

__device__ __forceinline__ float d_wave_sum64(float v) {
    #pragma unroll
    for (int mask = 32; mask >= 1; mask >>= 1) {
        v += __shfl_xor(v, mask, 64);
    }
    return v;
}

__device__ __forceinline__ float d_subwave_sum32(float v) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        v += __shfl_xor(v, mask, 32);
    }
    return v;
}

__device__ __forceinline__ float d_subwave_max32(float v) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        v = fmaxf(v, __shfl_xor(v, mask, 32));
    }
    return v;
}

__device__ __forceinline__ uint32_t d_float_to_ordered_u32(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? (~u) : (u ^ 0x80000000u);
}

// =============================================================================
// 1. Embedding, RMSNorm, Elementwise
// =============================================================================
__global__ __launch_bounds__(256)
void k_tp_embed_bf16(
    const int32_t* __restrict__ token_ids,
    const uint16_t* __restrict__ embed_w,
    uint16_t* __restrict__ out_bf16,
    int D, int vocab_offset, int vocab_rows)
{
    const int m = int(blockIdx.y);
    const int d = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (d >= D) return;
    const int32_t tid = token_ids[m] - vocab_offset;
    uint16_t val = 0u;
    if (tid >= 0 && tid < vocab_rows) {
        val = embed_w[size_t(tid) * size_t(D) + size_t(d)];
    }
    out_bf16[size_t(m) * size_t(D) + size_t(d)] = val;
}

__global__ __launch_bounds__(256)
void k_rmsnorm_bf16(
    const uint16_t* __restrict__ in_bf16,
    const uint16_t* __restrict__ w_bf16,
    uint16_t* __restrict__ out_bf16,
    int D, float eps)
{
    __shared__ float s_wave[4];
    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const uint16_t* row_in = in_bf16 + size_t(m) * size_t(D);
    uint16_t* row_out = out_bf16 + size_t(m) * size_t(D);

    float sq = 0.0f;
    for (int d = tid; d < D; d += 256) {
        const float v = d_bf16_to_f32(row_in[d]);
        sq += v * v;
    }
    sq = d_wave_sum64(sq);
    if ((tid & 63) == 0) s_wave[tid >> 6] = sq;
    __syncthreads();
    const float total_sq = s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3];
    const float rstd = rsqrtf(total_sq * (1.0f / float(D)) + eps);

    for (int d = tid; d < D; d += 256) {
        const float v = d_bf16_to_f32(row_in[d]) * rstd * d_bf16_to_f32(w_bf16[d]);
        row_out[d] = d_f32_to_bf16_rne(v);
    }
}

__global__ __launch_bounds__(256)
void k_f32_to_bf16(const float* __restrict__ in_f32, uint16_t* __restrict__ out_bf16, size_t n) {
    const size_t idx = size_t(blockIdx.x) * 256u + size_t(threadIdx.x);
    if (idx < n) out_bf16[idx] = d_f32_to_bf16_rne(in_f32[idx]);
}

__global__ __launch_bounds__(256)
void k_add_f32_2way_inplace(float* __restrict__ dst, const float* __restrict__ src, size_t n) {
    const size_t idx = size_t(blockIdx.x) * 256u + size_t(threadIdx.x);
    if (idx < n) dst[idx] += src[idx];
}

__global__ __launch_bounds__(256)
void k_add_f32_inplace(float* __restrict__ dst, const float* __restrict__ s1, const float* __restrict__ s2, size_t n) {
    const size_t idx = size_t(blockIdx.x) * 256u + size_t(threadIdx.x);
    if (idx < n) dst[idx] += s1[idx] + s2[idx];
}

__global__ __launch_bounds__(256)
void k_add_f32_3way(float* __restrict__ out, const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, size_t n) {
    const size_t idx = size_t(blockIdx.x) * 256u + size_t(threadIdx.x);
    if (idx < n) out[idx] = a[idx] + b[idx] + c[idx];
}

// 128-bit float4 vectorized P2P 3-way all-reduce (4x PCIe/xGMI burst efficiency!)
__global__ __launch_bounds__(256)
void k_add_f32_3way_vec4(float4* __restrict__ out4, const float4* __restrict__ a4, const float4* __restrict__ b4, const float4* __restrict__ c4, size_t n4) {
    const size_t idx = size_t(blockIdx.x) * 256u + size_t(threadIdx.x);
    if (idx < n4) {
        const float4 va = a4[idx];
        const float4 vb = b4[idx];
        const float4 vc = c4[idx];
        float4 r;
        r.x = va.x + vb.x + vc.x;
        r.y = va.y + vb.y + vc.y;
        r.z = va.z + vb.z + vc.z;
        r.w = va.w + vb.w + vc.w;
        out4[idx] = r;
    }
}

__global__ __launch_bounds__(256)
void k_add_bf16_3way(
    uint16_t* __restrict__ out,
    const uint16_t* __restrict__ a,
    const uint16_t* __restrict__ b,
    const uint16_t* __restrict__ c,
    size_t n)
{
    const size_t idx = size_t(blockIdx.x) * 256u + size_t(threadIdx.x);
    if (idx < n) {
        const float s = d_bf16_to_f32(a[idx]) + d_bf16_to_f32(b[idx]) + d_bf16_to_f32(c[idx]);
        out[idx] = d_f32_to_bf16_rne(s);
    }
}

// 128-bit uint4 (8 BF16s/thread) vectorized P2P 3-way all-reduce (8x PCIe/xGMI burst efficiency!)
__global__ __launch_bounds__(256)
void k_add_bf16_3way_vec8(
    uint4* __restrict__ out8,
    const uint4* __restrict__ a8,
    const uint4* __restrict__ b8,
    const uint4* __restrict__ c8,
    size_t n8)
{
    const size_t idx = size_t(blockIdx.x) * 256u + size_t(threadIdx.x);
    if (idx < n8) {
        const uint4 va = a8[idx];
        const uint4 vb = b8[idx];
        const uint4 vc = c8[idx];
        uint32_t words_a[4] = {va.x, va.y, va.z, va.w};
        uint32_t words_b[4] = {vb.x, vb.y, vb.z, vb.w};
        uint32_t words_c[4] = {vc.x, vc.y, vc.z, vc.w};
        uint32_t words_o[4];
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float lo = d_bf16_to_f32(uint16_t(words_a[w])) +
                             d_bf16_to_f32(uint16_t(words_b[w])) +
                             d_bf16_to_f32(uint16_t(words_c[w]));
            const float hi = d_bf16_to_f32(uint16_t(words_a[w] >> 16)) +
                             d_bf16_to_f32(uint16_t(words_b[w] >> 16)) +
                             d_bf16_to_f32(uint16_t(words_c[w] >> 16));
            words_o[w] = uint32_t(d_f32_to_bf16_rne(lo)) | (uint32_t(d_f32_to_bf16_rne(hi)) << 16);
        }
        out8[idx] = make_uint4(words_o[0], words_o[1], words_o[2], words_o[3]);
    }
}

__global__ __launch_bounds__(256)
void k_add_f32_and_bf16_to_bf16(
    const float* __restrict__ f32_sum,
    const uint16_t* __restrict__ p150_bf16,
    uint16_t* __restrict__ out_bf16,
    size_t n)
{
    const size_t idx = size_t(blockIdx.x) * 256u + size_t(threadIdx.x);
    if (idx < n) {
        float v = f32_sum[idx];
        if (p150_bf16) v += d_bf16_to_f32(p150_bf16[idx]);
        out_bf16[idx] = d_f32_to_bf16_rne(v);
    }
}

// =============================================================================
// 2. Hyper-Connections (mHC: hc_mult=4, dim=5120, hc_dim=20480, mix_hc=24)
// =============================================================================
__global__ __launch_bounds__(256)
void k_hc_init(
    const uint16_t* __restrict__ embed_bf16,
    uint16_t* __restrict__ h_bf16,
    float* __restrict__ pre_mix_f32,
    int M)
{
    const int m = int(blockIdx.y);
    const int d = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (d < 5120) {
        const uint16_t v = embed_bf16[size_t(m) * 5120u + size_t(d)];
        uint16_t* dst = h_bf16 + size_t(m) * 20480u;
        dst[d] = v;
        dst[5120 + d] = v;
        dst[10240 + d] = v;
        dst[15360 + d] = v;
    }
    if (blockIdx.x == 0 && threadIdx.x < 4) {
        pre_mix_f32[m * 4 + threadIdx.x] = (threadIdx.x == 0) ? 1.0f : 0.0f;
    }
}

__global__ __launch_bounds__(256)
void k_hc_mean_bf16(const uint16_t* __restrict__ h_bf16, uint16_t* __restrict__ mean_bf16) {
    const int m = int(blockIdx.y);
    const int d = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (d < 5120) {
        const uint16_t* src = h_bf16 + size_t(m) * 20480u + size_t(d);
        const float avg = 0.25f * (d_bf16_to_f32(src[0]) + d_bf16_to_f32(src[5120]) +
                                   d_bf16_to_f32(src[10240]) + d_bf16_to_f32(src[15360]));
        mean_bf16[size_t(m) * 5120u + size_t(d)] = d_f32_to_bf16_rne(avg);
    }
}

__global__ __launch_bounds__(256)
void k_hc_rms_rsqrt(const uint16_t* __restrict__ h_bf16, float* __restrict__ out_rsqrt, float norm_eps) {
    __shared__ float s_wave[4];
    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const uint16_t* row = h_bf16 + size_t(m) * 20480u;
    float sq = 0.0f;
    for (int i = tid; i < 20480; i += 256) {
        const float v = d_bf16_to_f32(row[i]);
        sq += v * v;
    }
    sq = d_wave_sum64(sq);
    if ((tid & 63) == 0) s_wave[tid >> 6] = sq;
    __syncthreads();
    if (tid == 0) {
        const float tot = s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3];
        out_rsqrt[m] = rsqrtf(tot * (1.0f / 20480.0f) + norm_eps);
    }
}

__global__ __launch_bounds__(256, 4)
void k_hc_proj24(
    const uint16_t* __restrict__ h_bf16,
    const float* __restrict__ hc_fn_f32,
    const float* __restrict__ rsqrt_f32,
    float* __restrict__ mixes_f32,
    int M)
{
    __shared__ float s_wave[8][4];
    const int out_idx = int(blockIdx.x);  // 0..23
    const int m0 = int(blockIdx.y) * 8;
    const int tid = int(threadIdx.x);

    const float* w = hc_fn_f32 + size_t(out_idx) * 20480u;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = tid * 4; i < 20480; i += 1024) {
        const float4 w4 = *reinterpret_cast<const float4*>(w + i);
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (m0 + r < M) {
                const uint2 x4 = *reinterpret_cast<const uint2*>(h_bf16 + size_t(m0 + r) * 20480u + size_t(i));
                acc[r] += d_bf16_to_f32(uint16_t(x4.x))       * w4.x
                        + d_bf16_to_f32(uint16_t(x4.x >> 16)) * w4.y
                        + d_bf16_to_f32(uint16_t(x4.y))       * w4.z
                        + d_bf16_to_f32(uint16_t(x4.y >> 16)) * w4.w;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        const float s = d_wave_sum64(acc[r]);
        if ((tid & 63) == 0) s_wave[r][tid >> 6] = s;
    }
    __syncthreads();
    if (tid < 8 && (m0 + tid) < M) {
        const int gm = m0 + tid;
        const float sum = s_wave[tid][0] + s_wave[tid][1] + s_wave[tid][2] + s_wave[tid][3];
        mixes_f32[gm * 24 + out_idx] = sum * rsqrt_f32[gm];
    }
}

// Exact kernel.py:426-460 hc_split_sinkhorn (1 thread per token m; 4x4 matrix fits in 16 registers!)
__global__ __launch_bounds__(64)
void k_hc_sinkhorn(
    const float* __restrict__ mixes_f32,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ out_pre,
    float* __restrict__ out_post,
    float* __restrict__ out_comb,
    int M, int sinkhorn_iters, float eps)
{
    const int m = int(blockIdx.x) * 64 + int(threadIdx.x);
    if (m >= M) return;

    const float* mx = mixes_f32 + m * 24;
    const float sc0 = hc_scale[0], sc1 = hc_scale[1], sc2 = hc_scale[2];

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float z0 = mx[j] * sc0 + hc_base[j];
        out_pre[m * 4 + j] = (1.0f / (1.0f + __expf(-z0))) + eps;
        const float z1 = mx[4 + j] * sc1 + hc_base[4 + j];
        out_post[m * 4 + j] = 2.0f / (1.0f + __expf(-z1));
    }

    float c[4][4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float rmax = -1e30f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int idx = 8 + j * 4 + k;
            c[j][k] = mx[idx] * sc2 + hc_base[idx];
            rmax = fmaxf(rmax, c[j][k]);
        }
        float rsum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            c[j][k] = __expf(c[j][k] - rmax);
            rsum += c[j][k];
        }
        const float inv_r = 1.0f / rsum;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            c[j][k] = c[j][k] * inv_r + eps;
        }
    }

    // Initial col normalization (comb = comb / (comb.sum(-2) + eps))
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float csum = c[0][k] + c[1][k] + c[2][k] + c[3][k] + eps;
        const float inv_c = 1.0f / csum;
        c[0][k] *= inv_c; c[1][k] *= inv_c; c[2][k] *= inv_c; c[3][k] *= inv_c;
    }

    for (int it = 0; it < sinkhorn_iters - 1; ++it) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float rsum = c[j][0] + c[j][1] + c[j][2] + c[j][3] + eps;
            const float inv_r = 1.0f / rsum;
            c[j][0] *= inv_r; c[j][1] *= inv_r; c[j][2] *= inv_r; c[j][3] *= inv_r;
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float csum = c[0][k] + c[1][k] + c[2][k] + c[3][k] + eps;
            const float inv_c = 1.0f / csum;
            c[0][k] *= inv_c; c[1][k] *= inv_c; c[2][k] *= inv_c; c[3][k] *= inv_c;
        }
    }

    float* dst_c = out_comb + m * 16;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            dst_c[j * 4 + k] = c[j][k];
        }
    }
}

__global__ __launch_bounds__(256, 2)
void k_hc_proj24_bm16(
    const uint16_t* __restrict__ h_bf16,
    const float* __restrict__ hc_fn_f32,
    const float* __restrict__ rsqrt_f32,
    float* __restrict__ mixes_f32,
    int M)
{
    __shared__ float s_wave[16][4];
    const int out_idx = int(blockIdx.x);  // 0..23
    const int m0 = int(blockIdx.y) * 16;
    const int tid = int(threadIdx.x);

    const float* w = hc_fn_f32 + size_t(out_idx) * 20480u;
    float acc[16] = {0.0f};

    for (int i = tid * 4; i < 20480; i += 1024) {
        const float4 w4 = *reinterpret_cast<const float4*>(w + i);
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (m0 + r < M) {
                const uint2 x4 = *reinterpret_cast<const uint2*>(h_bf16 + size_t(m0 + r) * 20480u + size_t(i));
                acc[r] += d_bf16_to_f32(uint16_t(x4.x))       * w4.x
                        + d_bf16_to_f32(uint16_t(x4.x >> 16)) * w4.y
                        + d_bf16_to_f32(uint16_t(x4.y))       * w4.z
                        + d_bf16_to_f32(uint16_t(x4.y >> 16)) * w4.w;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const float s = d_wave_sum64(acc[r]);
        if ((tid & 63) == 0) s_wave[r][tid >> 6] = s;
    }
    __syncthreads();
    if (tid < 16 && (m0 + tid) < M) {
        const int gm = m0 + tid;
        const float sum = s_wave[tid][0] + s_wave[tid][1] + s_wave[tid][2] + s_wave[tid][3];
        mixes_f32[gm * 24 + out_idx] = sum * rsqrt_f32[gm];
    }
}

__global__ __launch_bounds__(256)
void k_hc_pre(const uint16_t* __restrict__ h_bf16, const float* __restrict__ pre_f32, uint16_t* __restrict__ out_bf16) {
    const int m = int(blockIdx.y);
    const int d = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (d >= 5120) return;
    const float p0 = pre_f32[m * 4 + 0];
    const float p1 = pre_f32[m * 4 + 1];
    const float p2 = pre_f32[m * 4 + 2];
    const float p3 = pre_f32[m * 4 + 3];
    const uint16_t* src = h_bf16 + size_t(m) * 20480u + size_t(d);
    const float y = p0 * d_bf16_to_f32(src[0]) +
                    p1 * d_bf16_to_f32(src[5120]) +
                    p2 * d_bf16_to_f32(src[10240]) +
                    p3 * d_bf16_to_f32(src[15360]);
    out_bf16[size_t(m) * 5120u + size_t(d)] = d_f32_to_bf16_rne(y);
}

// Fused hc_pre + RMSNorm (D = 5120 = 20 * 256) + optional 2-unit WHT128: 1 CTA/row, 11 KiB LDS (4 CTAs/CU)!
__global__ __launch_bounds__(256, 4)
void k_hc_pre_rmsnorm_bf16(
    const uint16_t* __restrict__ h_bf16,
    const float*    __restrict__ pre_f32,
    const uint16_t* __restrict__ norm_w_bf16,
    uint16_t*       __restrict__ out_norm_bf16,
    uint16_t*       __restrict__ out_raw_bf16,
    float eps,
    const uint64_t* __restrict__ signs_u64,
    uint16_t*       __restrict__ out_rot_f16,
    uint16_t*       __restrict__ out_rot_bf16)
{
    __shared__ uint16_t s_raw[5120];
    __shared__ float s_wht[2][128];
    __shared__ float s_wave[4];

    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const float p0 = pre_f32[m * 4 + 0];
    const float p1 = pre_f32[m * 4 + 1];
    const float p2 = pre_f32[m * 4 + 2];
    const float p3 = pre_f32[m * 4 + 3];
    const uint16_t* src_row = h_bf16 + size_t(m) * 20480u;

    float sq = 0.0f;
    for (int d = tid; d < 5120; d += 256) {
        const float y = p0 * d_bf16_to_f32(src_row[d]) +
                        p1 * d_bf16_to_f32(src_row[5120 + d]) +
                        p2 * d_bf16_to_f32(src_row[10240 + d]) +
                        p3 * d_bf16_to_f32(src_row[15360 + d]);
        const uint16_t yb = d_f32_to_bf16_rne(y);
        s_raw[d] = yb;
        if (out_raw_bf16) out_raw_bf16[size_t(m) * 5120u + size_t(d)] = yb;
        const float yf = d_bf16_to_f32(yb);
        sq += yf * yf;
    }
    sq = d_wave_sum64(sq);
    if ((tid & 63) == 0) s_wave[tid >> 6] = sq;
    __syncthreads();

    const float total_sq = s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3];
    const float rstd = rsqrtf(total_sq * (1.0f / 5120.0f) + eps);
    uint16_t* dst_row = out_norm_bf16 + size_t(m) * 5120u;
    const bool do_wht = (signs_u64 != nullptr && out_rot_f16 != nullptr);
    for (int d = tid; d < 5120; d += 256) {
        const float v = d_bf16_to_f32(s_raw[d]) * rstd * d_bf16_to_f32(norm_w_bf16[d]);
        const uint16_t nb = d_f32_to_bf16_rne(v);
        dst_row[d] = nb;
        if (do_wht) s_raw[d] = nb;
    }

    if (do_wht) {
        __syncthreads();
        const int unit = tid >> 7;          // 0..1
        const int lane128 = tid & 127;      // 0..127
        const int lane64 = tid & 63;        // 0..63
        const int wid_u = (tid >> 6) & 1;   // 0..1
        uint16_t* dst_f16 = out_rot_f16 + size_t(m) * 5120u;
        uint16_t* dst_bf16 = out_rot_bf16 ? (out_rot_bf16 + size_t(m) * 5120u) : nullptr;

        for (int blk = unit; blk < 40; blk += 2) {
            const int col = blk * 128 + lane128;
            float v = d_bf16_to_f32(s_raw[col]);
            if ((signs_u64[blk * 2 + wid_u] >> lane64) & 1ULL) v = -v;
            #pragma unroll
            for (int len = 1; len <= 32; len <<= 1) {
                const float partner = __shfl_xor(v, len, 64);
                v = (lane64 & len) ? (partner - v) : (v + partner);
            }
            s_wht[unit][lane128] = v;
            __syncthreads();
            const float a = s_wht[unit][lane64];
            const float b = s_wht[unit][64 + lane64];
            const float res = ((wid_u == 0) ? (a + b) : (a - b)) * 0.08838834764831844655f;
            dst_f16[col] = d_f32_to_fp16_bits(res);
            if (dst_bf16) dst_bf16[col] = d_f32_to_bf16_rne(res);
            __syncthreads();
        }
    }
}

__global__ __launch_bounds__(256)
void k_hc_post(
    const uint16_t* __restrict__ sub_bf16,
    const uint16_t* __restrict__ res_bf16,
    const float* __restrict__ post_f32,
    const float* __restrict__ comb_f32,
    uint16_t* __restrict__ out_h_bf16)
{
    const int m = int(blockIdx.y);
    const int d = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (d >= 5120) return;

    const float xv = d_bf16_to_f32(sub_bf16[size_t(m) * 5120u + size_t(d)]);
    const uint16_t* rs = res_bf16 + size_t(m) * 20480u + size_t(d);
    const float r0 = d_bf16_to_f32(rs[0]);
    const float r1 = d_bf16_to_f32(rs[5120]);
    const float r2 = d_bf16_to_f32(rs[10240]);
    const float r3 = d_bf16_to_f32(rs[15360]);

    const float* ps = post_f32 + m * 4;
    const float* cb = comb_f32 + m * 16;
    uint16_t* dst = out_h_bf16 + size_t(m) * 20480u + size_t(d);

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float v = ps[j] * xv + cb[0 * 4 + j] * r0 + cb[1 * 4 + j] * r1 + cb[2 * 4 + j] * r2 + cb[3 * 4 + j] * r3;
        dst[j * 5120] = d_f32_to_bf16_rne(v);
    }
}

// =============================================================================
// 3. Engram Gate & Residual Add (model.py:350-365)
// =============================================================================
__global__ __launch_bounds__(256)
void k_engram_gate_add(
    uint16_t* __restrict__ h_bf16,
    const float* __restrict__ kv_f32,
    const uint16_t* __restrict__ q_weight,
    const uint16_t* __restrict__ k_weight,
    const uint8_t* __restrict__ image_mask,
    float norm_eps)
{
    __shared__ float s_red[3][4];
    const int c = int(blockIdx.x);  // 0..3
    const int m = int(blockIdx.y);  // 0..M-1
    const int tid = int(threadIdx.x);

    if (image_mask && image_mask[m]) return;

    uint16_t* h_row = h_bf16 + (size_t(m) * 4u + size_t(c)) * 5120u;
    const float* key_row = kv_f32 + size_t(m) * 25600u + size_t(c) * 5120u;
    const float* val_row = kv_f32 + size_t(m) * 25600u + 20480u;
    const uint16_t* qw = q_weight + size_t(c) * 5120u;
    const uint16_t* kw = k_weight + size_t(c) * 5120u;

    float sh2 = 0.0f, sk2 = 0.0f, sdot = 0.0f;
    for (int d = tid; d < 5120; d += 256) {
        const float hv = d_bf16_to_f32(h_row[d]);
        const float kv = key_row[d];
        const float wv = d_bf16_to_f32(qw[d]) * d_bf16_to_f32(kw[d]);
        sh2 += hv * hv;
        sk2 += kv * kv;
        sdot += hv * wv * kv;
    }
    sh2 = d_wave_sum64(sh2);
    sk2 = d_wave_sum64(sk2);
    sdot = d_wave_sum64(sdot);
    if ((tid & 63) == 0) {
        const int w = tid >> 6;
        s_red[0][w] = sh2;
        s_red[1][w] = sk2;
        s_red[2][w] = sdot;
    }
    __syncthreads();

    const float tot_h2 = s_red[0][0] + s_red[0][1] + s_red[0][2] + s_red[0][3];
    const float tot_k2 = s_red[1][0] + s_red[1][1] + s_red[1][2] + s_red[1][3];
    const float tot_dot = s_red[2][0] + s_red[2][1] + s_red[2][2] + s_red[2][3];

    const float rstd = rsqrtf(tot_h2 * (1.0f / 5120.0f) + norm_eps) *
                       rsqrtf(tot_k2 * (1.0f / 5120.0f) + norm_eps);
    const float dot = tot_dot * rstd * 0.013975424859373686f;  // 5120^-0.5
    const float s_sqrt = copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot);
    const float gate = 1.0f / (1.0f + __expf(-s_sqrt));

    for (int d = tid; d < 5120; d += 256) {
        const float hv = d_bf16_to_f32(h_row[d]);
        h_row[d] = d_f32_to_bf16_rne(hv + gate * val_row[d]);
    }
}

// =============================================================================
// 4. Quantization & Lossless FP8xFP8 (fdot2 + UE8M0/32) & BF16 GEMV/GEMM
// =============================================================================
__global__ __launch_bounds__(256)
void k_act_quant_ue8m0_32(
    const uint16_t* __restrict__ in_bf16,
    uint8_t* __restrict__ out_fp8,
    uint8_t* __restrict__ out_ue8m0,
    uint16_t* __restrict__ inplace_bf16,
    int total_groups,
    bool inplace)
{
    // 256 threads = 8 groups of 32 lanes!
    const int grp = int(blockIdx.x) * 8 + (int(threadIdx.x) >> 5);
    if (grp >= total_groups) return;
    const int lane = int(threadIdx.x) & 31;
    const size_t idx = size_t(grp) * 32u + size_t(lane);

    const float x = d_bf16_to_f32(in_bf16[idx]);
    const float amax = fmaxf(d_subwave_max32(fabsf(x)), 1e-4f);
    uint8_t ue8 = 0u;
    const float s = d_fast_round_scale_ue8m0(amax, 1.0f / 448.0f, ue8);
    const float q_in = fmaxf(-448.0f, fminf(448.0f, x / s));
    const uint8_t fp8 = d_f32_to_fp8_e4m3(q_in);

    if (inplace) {
        inplace_bf16[idx] = d_f32_to_bf16_rne(d_fp8_e4m3_to_f32(fp8) * s);
    } else {
        out_fp8[idx] = fp8;
        if (lane == 0) out_ue8m0[grp] = ue8;
    }
}

// Coalesced low-M (M <= 8) FP8xFP8 32x32 block-scaled GEMV/GEMM using hardware v_dot2_f32_f16!
// Grid: ((N + 7) / 8, 1), Block: 256 threads = 8 N-rows x 32 contiguous K-lanes (8 FP8 bytes/lane = 256B coalesced!).
template <int MAX_M>
__global__ __launch_bounds__(256, 4)
void k_fp8_gemv_fdot2(
    const uint16_t* __restrict__ x_fp8x2,  // [M, K/2]
    const uint8_t*  __restrict__ x_scale,  // [M, K/32]
    const uint16_t* __restrict__ w_fp8x2,  // [N, K/2]
    const uint8_t*  __restrict__ w_scale,  // [N/32, K/32]
    uint16_t* __restrict__ out_bf16,
    float*    __restrict__ out_f32,
    int M, int N, int K, bool write_f32, bool accum_f32)
{
    __shared__ uint16_t s_lut[256];
    __shared__ uint32_t s_x[MAX_M][128];
    __shared__ float    s_xs[MAX_M][8];

    const int tid = int(threadIdx.x);
    const int nw = tid >> 5;          // 0 .. 7
    const int lane = tid & 31;        // 0 .. 31
    const int n = int(blockIdx.x) * 8 + nw;

    s_lut[tid] = d_fp8_e4m3_to_fp16_bits(uint8_t(tid));
    __syncthreads();

    const int k_blks = K >> 5;        // K / 32
    const uint8_t* w_bytes = (n < N) ? (reinterpret_cast<const uint8_t*>(w_fp8x2) + size_t(n) * size_t(K)) : nullptr;
    const uint8_t* x_bytes = reinterpret_cast<const uint8_t*>(x_fp8x2);
    const uint8_t* ws_row  = (n < N) ? (w_scale + size_t(n >> 5) * size_t(k_blks)) : nullptr;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    const int K256 = K & ~255;
    for (int k_off = 0; k_off < K256; k_off += 256) {
        if (nw < MAX_M && nw < M) {
            const uint2 x8 = *reinterpret_cast<const uint2*>(x_bytes + size_t(nw) * size_t(K) + size_t(k_off + lane * 8));
            s_x[nw][lane * 4 + 0] = d_lut_fp8x2_to_u32(s_lut, uint16_t(x8.x));
            s_x[nw][lane * 4 + 1] = d_lut_fp8x2_to_u32(s_lut, uint16_t(x8.x >> 16));
            s_x[nw][lane * 4 + 2] = d_lut_fp8x2_to_u32(s_lut, uint16_t(x8.y));
            s_x[nw][lane * 4 + 3] = d_lut_fp8x2_to_u32(s_lut, uint16_t(x8.y >> 16));
            if (lane < 8) {
                s_xs[nw][lane] = d_ue8m0_to_f32(x_scale[size_t(nw) * size_t(k_blks) + size_t((k_off >> 5) + lane)]);
            }
        }
        __syncthreads();

        if (n < N) {
            const int kb_local = lane >> 2;
            const int kb = (k_off >> 5) + kb_local;
            const uint2 w8 = *reinterpret_cast<const uint2*>(w_bytes + size_t(k_off + lane * 8));
            const f16x2_t wv0 = d_lut_fp8x2_to_f16x2(s_lut, uint16_t(w8.x));
            const f16x2_t wv1 = d_lut_fp8x2_to_f16x2(s_lut, uint16_t(w8.x >> 16));
            const f16x2_t wv2 = d_lut_fp8x2_to_f16x2(s_lut, uint16_t(w8.y));
            const f16x2_t wv3 = d_lut_fp8x2_to_f16x2(s_lut, uint16_t(w8.y >> 16));
            const float sw = d_ue8m0_to_f32(ws_row[kb]);

            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < M) {
                    const uint4 ux = *reinterpret_cast<const uint4*>(&s_x[m][lane * 4]);
                    f16x2_t xv0, xv1, xv2, xv3;
                    __builtin_memcpy(&xv0, &ux.x, 4);
                    __builtin_memcpy(&xv1, &ux.y, 4);
                    __builtin_memcpy(&xv2, &ux.z, 4);
                    __builtin_memcpy(&xv3, &ux.w, 4);
                    float b = 0.0f;
                    b = __builtin_amdgcn_fdot2(wv0, xv0, b, false);
                    b = __builtin_amdgcn_fdot2(wv1, xv1, b, false);
                    b = __builtin_amdgcn_fdot2(wv2, xv2, b, false);
                    b = __builtin_amdgcn_fdot2(wv3, xv3, b, false);
                    acc[m] += b * (sw * s_xs[m][kb_local]);
                }
            }
        }
        __syncthreads();
    }

    if (n >= N) return;

    for (int k_off = K256; k_off < K; k_off += 32) {
        const int kb = k_off >> 5;
        if (lane < 16) {
            const uint16_t w2 = w_fp8x2[size_t(n) * size_t(K >> 1) + size_t(kb * 16 + lane)];
            const f16x2_t wv = d_lut_fp8x2_to_f16x2(s_lut, w2);
            const float sw = d_ue8m0_to_f32(ws_row[kb]);
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < M) {
                    const uint16_t x2 = x_fp8x2[size_t(m) * size_t(K >> 1) + size_t(kb * 16 + lane)];
                    const float sx = d_ue8m0_to_f32(x_scale[size_t(m) * size_t(k_blks) + size_t(kb)]);
                    acc[m] += __builtin_amdgcn_fdot2(wv, d_lut_fp8x2_to_f16x2(s_lut, x2), 0.0f, false) * (sw * sx);
                }
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float sum = d_subwave_sum32(acc[m]);
            if (lane == 0) {
                const size_t out_idx = size_t(m) * size_t(N) + size_t(n);
                if (write_f32) {
                    if (accum_f32) out_f32[out_idx] += sum;
                    else           out_f32[out_idx] = sum;
                } else {
                    out_bf16[out_idx] = d_f32_to_bf16_rne(sum);
                }
            }
        }
    }
}

// Tiled M > 8 FP8xFP8 GEMM for prefill (BM=32, BN=32, BK=32, 100% coalesced X & W tiles + conflict-free LDS, ScratchSize == 0!)
__global__ __launch_bounds__(256, 2)
void k_fp8_gemm_tiled32(
    const uint16_t* __restrict__ x_fp8x2,
    const uint8_t*  __restrict__ x_scale,
    const uint16_t* __restrict__ w_fp8x2,
    const uint8_t*  __restrict__ w_scale,
    uint16_t* __restrict__ out_bf16,
    float*    __restrict__ out_f32,
    int M, int N, int K, bool write_f32, bool accum_f32)
{
    __shared__ uint32_t s_x[32][16];
    __shared__ uint32_t s_w[32][17];
    __shared__ float    s_xs[32];
    __shared__ float    s_ws;

    const int nb = int(blockIdx.x);   // N tile of 32
    const int mb = int(blockIdx.y);   // M tile of 32
    const int tid = int(threadIdx.x); // 0..255
    const int nl = tid & 31;          // 0..31 (one N column per lane!)
    const int mg = tid >> 5;          // 0..7  (handles 4 M-rows: mg*4 + 0..3)

    const int n = nb * 32 + nl;
    const int m0 = mb * 32 + mg * 4;
    const int k_blks = K >> 5;
    const int k2_row = K >> 1;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int kb = 0; kb < k_blks; ++kb) {
        // Coalesced cooperative load of 32x16 f16x2 X-tile and 32x16 f16x2 W-tile!
        #pragma unroll
        for (int ld = 0; ld < 2; ++ld) {
            const int idx = tid * 2 + ld;
            const int r = idx >> 4;    // 0..31
            const int kp = idx & 15;   // 0..15 (16 adjacent threads read 32 contiguous bytes!)
            const int gm = mb * 32 + r;
            const int gn = nb * 32 + r;
            uint32_t px = 0u, pw = 0u;
            if (gm < M) {
                const f16x2_t xv = d_fp8x2_to_f16x2(x_fp8x2[size_t(gm) * size_t(k2_row) + kb * 16 + kp]);
                __builtin_memcpy(&px, &xv, 4);
            }
            if (gn < N) {
                const f16x2_t wv = d_fp8x2_to_f16x2(w_fp8x2[size_t(gn) * size_t(k2_row) + kb * 16 + kp]);
                __builtin_memcpy(&pw, &wv, 4);
            }
            s_x[r][kp] = px;
            s_w[r][kp] = pw;
        }
        if (tid < 32) {
            const int gm = mb * 32 + tid;
            s_xs[tid] = (gm < M) ? d_ue8m0_to_f32(x_scale[size_t(gm) * size_t(k_blks) + kb]) : 0.0f;
        } else if (tid == 32) {
            s_ws = (nb * 32 < N) ? d_ue8m0_to_f32(w_scale[size_t(nb) * size_t(k_blks) + kb]) : 0.0f;
        }
        __syncthreads();

        if (n < N) {
            const float sw = s_ws;
            float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;

            #pragma unroll
            for (int p = 0; p < 16; ++p) {
                f16x2_t wv, x0, x1, x2, x3;
                const uint32_t uw = s_w[nl][p];
                const uint32_t u0 = s_x[mg * 4 + 0][p];
                const uint32_t u1 = s_x[mg * 4 + 1][p];
                const uint32_t u2 = s_x[mg * 4 + 2][p];
                const uint32_t u3 = s_x[mg * 4 + 3][p];
                __builtin_memcpy(&wv, &uw, 4);
                __builtin_memcpy(&x0, &u0, 4);
                __builtin_memcpy(&x1, &u1, 4);
                __builtin_memcpy(&x2, &u2, 4);
                __builtin_memcpy(&x3, &u3, 4);
                b0 = __builtin_amdgcn_fdot2(wv, x0, b0, false);
                b1 = __builtin_amdgcn_fdot2(wv, x1, b1, false);
                b2 = __builtin_amdgcn_fdot2(wv, x2, b2, false);
                b3 = __builtin_amdgcn_fdot2(wv, x3, b3, false);
            }
            acc0 += b0 * (sw * s_xs[mg * 4 + 0]);
            acc1 += b1 * (sw * s_xs[mg * 4 + 1]);
            acc2 += b2 * (sw * s_xs[mg * 4 + 2]);
            acc3 += b3 * (sw * s_xs[mg * 4 + 3]);
        }
        __syncthreads();
    }

    if (n < N) {
        const float vals[4] = {acc0, acc1, acc2, acc3};
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int gm = m0 + r;
            if (gm < M) {
                const size_t out_idx = size_t(gm) * size_t(N) + size_t(n);
                if (write_f32) {
                    if (accum_f32) out_f32[out_idx] += vals[r];
                    else           out_f32[out_idx] = vals[r];
                } else {
                    out_bf16[out_idx] = d_f32_to_bf16_rne(vals[r]);
                }
            }
        }
    }
}

// High-throughput BM=64, BN=64, BK=32 4x4 fdot2 GEMM for M >= 32 prefill (8.8 KiB LDS, ScratchSize == 0!)
__global__ __launch_bounds__(256, 2)
void k_fp8_gemm_bm64_bn64(
    const uint16_t* __restrict__ x_fp8x2,
    const uint8_t*  __restrict__ x_scale,
    const uint16_t* __restrict__ w_fp8x2,
    const uint8_t*  __restrict__ w_scale,
    uint16_t* __restrict__ out_bf16,
    float*    __restrict__ out_f32,
    int M, int N, int K, bool write_f32, bool accum_f32)
{
    __shared__ uint16_t s_lut[256];
    __shared__ uint32_t s_x[64][17];
    __shared__ uint32_t s_w_t[16][65];
    __shared__ float    s_xs[64];
    __shared__ float    s_ws[2];

    const int nb64 = int(blockIdx.x);
    const int mb64 = int(blockIdx.y);
    const int tid  = int(threadIdx.x);
    const int tx   = tid & 15;   // 0..15 -> coalesced N-cols (j*16 + tx)
    const int ty   = tid >> 4;   // 0..15 -> 4 M-rows (ty*4 + 0..3)

    s_lut[tid] = d_fp8_e4m3_to_fp16_bits(uint8_t(tid));
    __syncthreads();

    const int gn_base = nb64 * 64;
    const int gm_base = mb64 * 64;
    const int k_blks  = K >> 5;
    const int k2_row  = K >> 1;

    float acc[4][4] = {{0.0f, 0.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 0.0f, 0.0f}};

    for (int kb = 0; kb < k_blks; ++kb) {
        {
            const int r  = tid >> 2;          // 0..63
            const int kp = (tid & 3) * 4;     // 0, 4, 8, 12 (8 bytes = 1x uint2 per thread!)
            const int gm = gm_base + r;
            const int gn = gn_base + r;
            uint32_t px0 = 0u, px1 = 0u, px2 = 0u, px3 = 0u;
            uint32_t pw0 = 0u, pw1 = 0u, pw2 = 0u, pw3 = 0u;
            if (gm < M) {
                const uint2 vx = *reinterpret_cast<const uint2*>(x_fp8x2 + size_t(gm) * size_t(k2_row) + kb * 16 + kp);
                px0 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vx.x));
                px1 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vx.x >> 16));
                px2 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vx.y));
                px3 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vx.y >> 16));
            }
            if (gn < N) {
                const uint2 vw = *reinterpret_cast<const uint2*>(w_fp8x2 + size_t(gn) * size_t(k2_row) + kb * 16 + kp);
                pw0 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vw.x));
                pw1 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vw.x >> 16));
                pw2 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vw.y));
                pw3 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vw.y >> 16));
            }
            s_x[r][kp + 0] = px0; s_x[r][kp + 1] = px1; s_x[r][kp + 2] = px2; s_x[r][kp + 3] = px3;
            s_w_t[kp + 0][r] = pw0; s_w_t[kp + 1][r] = pw1; s_w_t[kp + 2][r] = pw2; s_w_t[kp + 3][r] = pw3;
        }
        if (tid < 64) {
            const int gm = gm_base + tid;
            s_xs[tid] = (gm < M) ? d_ue8m0_to_f32(x_scale[size_t(gm) * size_t(k_blks) + kb]) : 0.0f;
        } else if (tid < 66) {
            const int sub_nb = nb64 * 2 + (tid - 64);
            s_ws[tid - 64] = (sub_nb * 32 < N) ? d_ue8m0_to_f32(w_scale[size_t(sub_nb) * size_t(k_blks) + kb]) : 0.0f;
        }
        __syncthreads();

        float b[4][4] = {{0.0f, 0.0f, 0.0f, 0.0f},
                         {0.0f, 0.0f, 0.0f, 0.0f},
                         {0.0f, 0.0f, 0.0f, 0.0f},
                         {0.0f, 0.0f, 0.0f, 0.0f}};
        #pragma unroll
        for (int p = 0; p < 16; ++p) {
            f16x2_t xv[4], wv[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t ux = s_x[ty * 4 + i][p];
                __builtin_memcpy(&xv[i], &ux, 4);
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t uw = s_w_t[p][j * 16 + tx];
                __builtin_memcpy(&wv[j], &uw, 4);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    b[i][j] = __builtin_amdgcn_fdot2(wv[j], xv[i], b[i][j], false);
                }
            }
        }

        const float sw0 = s_ws[0];
        const float sw1 = s_ws[1];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float sx = s_xs[ty * 4 + i];
            acc[i][0] += b[i][0] * (sw0 * sx);
            acc[i][1] += b[i][1] * (sw0 * sx);
            acc[i][2] += b[i][2] * (sw1 * sx);
            acc[i][3] += b[i][3] * (sw1 * sx);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int gm = gm_base + ty * 4 + i;
        if (gm < M) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int gn = gn_base + j * 16 + tx;
                if (gn < N) {
                    const size_t out_idx = size_t(gm) * size_t(N) + size_t(gn);
                    if (write_f32) {
                        if (accum_f32) out_f32[out_idx] += acc[i][j];
                        else           out_f32[out_idx] = acc[i][j];
                    } else {
                        out_bf16[out_idx] = d_f32_to_bf16_rne(acc[i][j]);
                    }
                }
            }
        }
    }
}

// High-throughput BM=128, BN=64, BK=32 8x4 fdot2 GEMM for M >= 128 prefill (2x W-tile reuse, zero bank conflicts, ScratchSize == 0!)
__global__ __launch_bounds__(256, 2)
void k_fp8_gemm_bm128_bn64(
    const uint16_t* __restrict__ x_fp8x2,
    const uint8_t*  __restrict__ x_scale,
    const uint16_t* __restrict__ w_fp8x2,
    const uint8_t*  __restrict__ w_scale,
    uint16_t* __restrict__ out_bf16,
    float*    __restrict__ out_f32,
    int M, int N, int K, bool write_f32, bool accum_f32)
{
    __shared__ uint16_t s_lut[256];
    __shared__ uint32_t s_x[128][17];
    __shared__ uint32_t s_w_t[16][65];
    __shared__ float    s_xs[128];
    __shared__ float    s_ws[2];

    const int nb64 = int(blockIdx.x);
    const int mb128 = int(blockIdx.y);
    const int tid  = int(threadIdx.x);
    const int tx   = tid & 15;   // 0..15 -> 4 coalesced N-cols (j*16 + tx)
    const int ty   = tid >> 4;   // 0..15 -> 8 M-rows (ty*8 + 0..7)

    s_lut[tid] = d_fp8_e4m3_to_fp16_bits(uint8_t(tid));
    __syncthreads();

    const int gn_base = nb64 * 64;
    const int gm_base = mb128 * 128;
    const int k_blks  = K >> 5;
    const int k2_row  = K >> 1;

    float acc[8][4] = {{0.0f}};

    for (int kb = 0; kb < k_blks; ++kb) {
        #pragma unroll
        for (int ld = 0; ld < 2; ++ld) {
            const int r  = (tid >> 2) + ld * 64; // 0..127
            const int kp = (tid & 3) * 4;
            const int gm = gm_base + r;
            uint32_t px0 = 0u, px1 = 0u, px2 = 0u, px3 = 0u;
            if (gm < M) {
                const uint2 vx = *reinterpret_cast<const uint2*>(x_fp8x2 + size_t(gm) * size_t(k2_row) + kb * 16 + kp);
                px0 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vx.x));
                px1 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vx.x >> 16));
                px2 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vx.y));
                px3 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vx.y >> 16));
            }
            s_x[r][kp + 0] = px0; s_x[r][kp + 1] = px1; s_x[r][kp + 2] = px2; s_x[r][kp + 3] = px3;
        }
        {
            const int r  = tid >> 2;
            const int kp = (tid & 3) * 4;
            const int gn = gn_base + r;
            uint32_t pw0 = 0u, pw1 = 0u, pw2 = 0u, pw3 = 0u;
            if (gn < N) {
                const uint2 vw = *reinterpret_cast<const uint2*>(w_fp8x2 + size_t(gn) * size_t(k2_row) + kb * 16 + kp);
                pw0 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vw.x));
                pw1 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vw.x >> 16));
                pw2 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vw.y));
                pw3 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vw.y >> 16));
            }
            s_w_t[kp + 0][r] = pw0; s_w_t[kp + 1][r] = pw1; s_w_t[kp + 2][r] = pw2; s_w_t[kp + 3][r] = pw3;
        }
        if (tid < 128) {
            const int gm = gm_base + tid;
            s_xs[tid] = (gm < M) ? d_ue8m0_to_f32(x_scale[size_t(gm) * size_t(k_blks) + kb]) : 0.0f;
        } else if (tid < 130) {
            const int sub_nb = nb64 * 2 + (tid - 128);
            s_ws[tid - 128] = (sub_nb * 32 < N) ? d_ue8m0_to_f32(w_scale[size_t(sub_nb) * size_t(k_blks) + kb]) : 0.0f;
        }
        __syncthreads();

        float b[8][4] = {{0.0f}};
        #pragma unroll 2
        for (int p = 0; p < 16; ++p) {
            f16x2_t xv[8], wv[4];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t ux = s_x[ty * 8 + i][p];
                __builtin_memcpy(&xv[i], &ux, 4);
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t uw = s_w_t[p][j * 16 + tx];
                __builtin_memcpy(&wv[j], &uw, 4);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    b[i][j] = __builtin_amdgcn_fdot2(wv[j], xv[i], b[i][j], false);
                }
            }
        }

        const float sw0 = s_ws[0];
        const float sw1 = s_ws[1];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float sx = s_xs[ty * 8 + i];
            acc[i][0] += b[i][0] * (sw0 * sx);
            acc[i][1] += b[i][1] * (sw0 * sx);
            acc[i][2] += b[i][2] * (sw1 * sx);
            acc[i][3] += b[i][3] * (sw1 * sx);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int gm = gm_base + ty * 8 + i;
        if (gm < M) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int gn = gn_base + j * 16 + tx;
                if (gn < N) {
                    const size_t out_idx = size_t(gm) * size_t(N) + size_t(gn);
                    if (write_f32) {
                        if (accum_f32) out_f32[out_idx] += acc[i][j];
                        else           out_f32[out_idx] = acc[i][j];
                    } else {
                        out_bf16[out_idx] = d_f32_to_bf16_rne(acc[i][j]);
                    }
                }
            }
        }
    }
}

// Coalesced multi-row grouped block-diagonal wo_a GEMV/GEMM (M <= 8): [M, g_cnt, 4096] x [g_cnt, 1024, 4096] -> [M, g_cnt, 1024] BF16
template <int MAX_M>
__global__ __launch_bounds__(256, 4)
void k_fp8_grouped_wo_a_mr(
    const uint16_t* __restrict__ o_fp8x2,
    const uint8_t*  __restrict__ o_scale,
    const uint16_t* __restrict__ w_fp8x2,
    const uint8_t*  __restrict__ w_scale,
    uint16_t* __restrict__ out_bf16,
    int M, int g_cnt)
{
    __shared__ uint16_t s_lut[256];
    __shared__ uint32_t s_x[MAX_M][128];
    __shared__ float    s_xs[MAX_M][8];

    const int tid = int(threadIdx.x);
    const int nw = tid >> 5;                      // 0..7
    const int lane = tid & 31;                    // 0..31
    const int n_local = int(blockIdx.x) * 8 + nw; // 0..1023
    const int g = int(blockIdx.y);                // 0..g_cnt-1
    const int n_global = g * 1024 + n_local;

    s_lut[tid] = d_fp8_e4m3_to_fp16_bits(uint8_t(tid));
    __syncthreads();

    const uint8_t* w_bytes = reinterpret_cast<const uint8_t*>(w_fp8x2) + size_t(n_global) * 4096u;
    const uint8_t* ws_row  = w_scale + size_t(g * 32 + (n_local >> 5)) * 128u;
    const uint8_t* o_bytes = reinterpret_cast<const uint8_t*>(o_fp8x2);

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (int k_off = 0; k_off < 4096; k_off += 256) {
        if (nw < MAX_M && nw < M) {
            const size_t rg = size_t(nw) * size_t(g_cnt) + size_t(g);
            const uint2 x8 = *reinterpret_cast<const uint2*>(o_bytes + rg * 4096u + size_t(k_off + lane * 8));
            s_x[nw][lane * 4 + 0] = d_lut_fp8x2_to_u32(s_lut, uint16_t(x8.x));
            s_x[nw][lane * 4 + 1] = d_lut_fp8x2_to_u32(s_lut, uint16_t(x8.x >> 16));
            s_x[nw][lane * 4 + 2] = d_lut_fp8x2_to_u32(s_lut, uint16_t(x8.y));
            s_x[nw][lane * 4 + 3] = d_lut_fp8x2_to_u32(s_lut, uint16_t(x8.y >> 16));
            if (lane < 8) {
                s_xs[nw][lane] = d_ue8m0_to_f32(o_scale[rg * 128u + size_t((k_off >> 5) + lane)]);
            }
        }
        __syncthreads();

        const int kb_local = lane >> 2;
        const int kb = (k_off >> 5) + kb_local;
        const uint2 w8 = *reinterpret_cast<const uint2*>(w_bytes + size_t(k_off + lane * 8));
        const f16x2_t wv0 = d_lut_fp8x2_to_f16x2(s_lut, uint16_t(w8.x));
        const f16x2_t wv1 = d_lut_fp8x2_to_f16x2(s_lut, uint16_t(w8.x >> 16));
        const f16x2_t wv2 = d_lut_fp8x2_to_f16x2(s_lut, uint16_t(w8.y));
        const f16x2_t wv3 = d_lut_fp8x2_to_f16x2(s_lut, uint16_t(w8.y >> 16));
        const float sw = d_ue8m0_to_f32(ws_row[kb]);

        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const uint4 ux = *reinterpret_cast<const uint4*>(&s_x[m][lane * 4]);
                f16x2_t xv0, xv1, xv2, xv3;
                __builtin_memcpy(&xv0, &ux.x, 4);
                __builtin_memcpy(&xv1, &ux.y, 4);
                __builtin_memcpy(&xv2, &ux.z, 4);
                __builtin_memcpy(&xv3, &ux.w, 4);
                float b = 0.0f;
                b = __builtin_amdgcn_fdot2(wv0, xv0, b, false);
                b = __builtin_amdgcn_fdot2(wv1, xv1, b, false);
                b = __builtin_amdgcn_fdot2(wv2, xv2, b, false);
                b = __builtin_amdgcn_fdot2(wv3, xv3, b, false);
                acc[m] += b * (sw * s_xs[m][kb_local]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float sum = d_subwave_sum32(acc[m]);
            if (lane == 0) {
                out_bf16[(size_t(m) * size_t(g_cnt) + size_t(g)) * 1024u + size_t(n_local)] = d_f32_to_bf16_rne(sum);
            }
        }
    }
}

// Tiled grouped block-diagonal wo_a GEMM for M > 8 prefill (BM=32, BN=32, BK=32)
__global__ __launch_bounds__(256, 2)
void k_fp8_grouped_wo_a_tiled32(
    const uint16_t* __restrict__ o_fp8x2,
    const uint8_t*  __restrict__ o_scale,
    const uint16_t* __restrict__ w_fp8x2,
    const uint8_t*  __restrict__ w_scale,
    uint16_t* __restrict__ out_bf16,
    int M, int g_cnt)
{
    __shared__ uint32_t s_x[32][16];
    __shared__ uint32_t s_w[32][17];
    __shared__ float    s_xs[32];
    __shared__ float    s_ws;

    const int nb_local = int(blockIdx.x); // 0..31
    const int mb = int(blockIdx.y);       // 0..(M+31)/32 - 1
    const int g = int(blockIdx.z);        // 0..g_cnt - 1
    const int tid = int(threadIdx.x);
    const int nl = tid & 31;
    const int mg = tid >> 5;
    const int n_local = nb_local * 32 + nl;
    const int m0 = mb * 32 + mg * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int kb = 0; kb < 128; ++kb) {
        #pragma unroll
        for (int ld = 0; ld < 2; ++ld) {
            const int idx = tid * 2 + ld;
            const int r = idx >> 4;
            const int kp = idx & 15;
            const int gm = mb * 32 + r;
            const int gn_global = g * 1024 + nb_local * 32 + r;
            uint32_t px = 0u, pw = 0u;
            if (gm < M) {
                const f16x2_t xv = d_fp8x2_to_f16x2(o_fp8x2[(size_t(gm) * size_t(g_cnt) + size_t(g)) * 2048u + kb * 16 + kp]);
                __builtin_memcpy(&px, &xv, 4);
            }
            const f16x2_t wv = d_fp8x2_to_f16x2(w_fp8x2[size_t(gn_global) * 2048u + kb * 16 + kp]);
            __builtin_memcpy(&pw, &wv, 4);
            s_x[r][kp] = px;
            s_w[r][kp] = pw;
        }
        if (tid < 32) {
            const int gm = mb * 32 + tid;
            s_xs[tid] = (gm < M) ? d_ue8m0_to_f32(o_scale[(size_t(gm) * size_t(g_cnt) + size_t(g)) * 128u + kb]) : 0.0f;
        } else if (tid == 32) {
            s_ws = d_ue8m0_to_f32(w_scale[size_t(g * 32 + nb_local) * 128u + kb]);
        }
        __syncthreads();

        const float sw = s_ws;
        float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
        #pragma unroll
        for (int p = 0; p < 16; ++p) {
            f16x2_t wv, x0, x1, x2, x3;
            const uint32_t uw = s_w[nl][p];
            const uint32_t u0 = s_x[mg * 4 + 0][p];
            const uint32_t u1 = s_x[mg * 4 + 1][p];
            const uint32_t u2 = s_x[mg * 4 + 2][p];
            const uint32_t u3 = s_x[mg * 4 + 3][p];
            __builtin_memcpy(&wv, &uw, 4);
            __builtin_memcpy(&x0, &u0, 4);
            __builtin_memcpy(&x1, &u1, 4);
            __builtin_memcpy(&x2, &u2, 4);
            __builtin_memcpy(&x3, &u3, 4);
            b0 = __builtin_amdgcn_fdot2(wv, x0, b0, false);
            b1 = __builtin_amdgcn_fdot2(wv, x1, b1, false);
            b2 = __builtin_amdgcn_fdot2(wv, x2, b2, false);
            b3 = __builtin_amdgcn_fdot2(wv, x3, b3, false);
        }
        acc0 += b0 * (sw * s_xs[mg * 4 + 0]);
        acc1 += b1 * (sw * s_xs[mg * 4 + 1]);
        acc2 += b2 * (sw * s_xs[mg * 4 + 2]);
        acc3 += b3 * (sw * s_xs[mg * 4 + 3]);
        __syncthreads();
    }

    const float vals[4] = {acc0, acc1, acc2, acc3};
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int gm = m0 + r;
        if (gm < M) {
            out_bf16[(size_t(gm) * size_t(g_cnt) + size_t(g)) * 1024u + size_t(n_local)] = d_f32_to_bf16_rne(vals[r]);
        }
    }
}

// High-throughput BM=64, BN=64, BK=32 4x4 fdot2 grouped wo_a GEMM for M >= 32 prefill (ScratchSize == 0!)
__global__ __launch_bounds__(256, 2)
void k_fp8_grouped_wo_a_bm64_bn64(
    const uint16_t* __restrict__ o_fp8x2,
    const uint8_t*  __restrict__ o_scale,
    const uint16_t* __restrict__ w_fp8x2,
    const uint8_t*  __restrict__ w_scale,
    uint16_t* __restrict__ out_bf16,
    int M, int g_cnt)
{
    __shared__ uint16_t s_lut[256];
    __shared__ uint32_t s_x[64][17];
    __shared__ uint32_t s_w_t[16][65];
    __shared__ float    s_xs[64];
    __shared__ float    s_ws[2];

    const int nb64_local = int(blockIdx.x); // 0..15 (1024 / 64)
    const int mb64       = int(blockIdx.y); // 0..(M+63)/64 - 1
    const int g          = int(blockIdx.z); // 0..g_cnt - 1
    const int tid        = int(threadIdx.x);
    const int tx         = tid & 15;
    const int ty         = tid >> 4;
    const int gm_base    = mb64 * 64;
    const int nl_base    = nb64_local * 64;

    s_lut[tid] = d_fp8_e4m3_to_fp16_bits(uint8_t(tid));
    __syncthreads();

    float acc[4][4] = {{0.0f, 0.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 0.0f, 0.0f}};

    for (int kb = 0; kb < 128; ++kb) {
        {
            const int r  = tid >> 2;
            const int kp = (tid & 3) * 4;
            const int gm = gm_base + r;
            const int gn_global = g * 1024 + nl_base + r;
            uint32_t px0 = 0u, px1 = 0u, px2 = 0u, px3 = 0u;
            if (gm < M) {
                const uint2 vx = *reinterpret_cast<const uint2*>(o_fp8x2 + (size_t(gm) * size_t(g_cnt) + size_t(g)) * 2048u + kb * 16 + kp);
                px0 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vx.x));
                px1 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vx.x >> 16));
                px2 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vx.y));
                px3 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vx.y >> 16));
            }
            const uint2 vw = *reinterpret_cast<const uint2*>(w_fp8x2 + size_t(gn_global) * 2048u + kb * 16 + kp);
            const uint32_t pw0 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vw.x));
            const uint32_t pw1 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vw.x >> 16));
            const uint32_t pw2 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vw.y));
            const uint32_t pw3 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vw.y >> 16));
            s_x[r][kp + 0] = px0; s_x[r][kp + 1] = px1; s_x[r][kp + 2] = px2; s_x[r][kp + 3] = px3;
            s_w_t[kp + 0][r] = pw0; s_w_t[kp + 1][r] = pw1; s_w_t[kp + 2][r] = pw2; s_w_t[kp + 3][r] = pw3;
        }
        if (tid < 64) {
            const int gm = gm_base + tid;
            s_xs[tid] = (gm < M) ? d_ue8m0_to_f32(o_scale[(size_t(gm) * size_t(g_cnt) + size_t(g)) * 128u + kb]) : 0.0f;
        } else if (tid < 66) {
            s_ws[tid - 64] = d_ue8m0_to_f32(w_scale[size_t(g * 32 + nb64_local * 2 + (tid - 64)) * 128u + kb]);
        }
        __syncthreads();

        float b[4][4] = {{0.0f, 0.0f, 0.0f, 0.0f},
                         {0.0f, 0.0f, 0.0f, 0.0f},
                         {0.0f, 0.0f, 0.0f, 0.0f},
                         {0.0f, 0.0f, 0.0f, 0.0f}};
        #pragma unroll
        for (int p = 0; p < 16; ++p) {
            f16x2_t xv[4], wv[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t ux = s_x[ty * 4 + i][p];
                __builtin_memcpy(&xv[i], &ux, 4);
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t uw = s_w_t[p][j * 16 + tx];
                __builtin_memcpy(&wv[j], &uw, 4);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    b[i][j] = __builtin_amdgcn_fdot2(wv[j], xv[i], b[i][j], false);
                }
            }
        }

        const float sw0 = s_ws[0];
        const float sw1 = s_ws[1];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float sx = s_xs[ty * 4 + i];
            acc[i][0] += b[i][0] * (sw0 * sx);
            acc[i][1] += b[i][1] * (sw0 * sx);
            acc[i][2] += b[i][2] * (sw1 * sx);
            acc[i][3] += b[i][3] * (sw1 * sx);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int gm = gm_base + ty * 4 + i;
        if (gm < M) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int nl = nl_base + j * 16 + tx;
                out_bf16[(size_t(gm) * size_t(g_cnt) + size_t(g)) * 1024u + size_t(nl)] = d_f32_to_bf16_rne(acc[i][j]);
            }
        }
    }
}

// High-throughput BM=128, BN=64, BK=32 8x4 fdot2 grouped wo_a GEMM for M >= 128 prefill (2x W-tile reuse, ScratchSize == 0!)
__global__ __launch_bounds__(256, 2)
void k_fp8_grouped_wo_a_bm128_bn64(
    const uint16_t* __restrict__ o_fp8x2,
    const uint8_t*  __restrict__ o_scale,
    const uint16_t* __restrict__ w_fp8x2,
    const uint8_t*  __restrict__ w_scale,
    uint16_t* __restrict__ out_bf16,
    int M, int g_cnt)
{
    __shared__ uint16_t s_lut[256];
    __shared__ uint32_t s_x[128][17];
    __shared__ uint32_t s_w_t[16][65];
    __shared__ float    s_xs[128];
    __shared__ float    s_ws[2];

    const int nb64_local = int(blockIdx.x); // 0..15 (1024 / 64)
    const int mb128      = int(blockIdx.y); // 0..(M+127)/128 - 1
    const int g          = int(blockIdx.z); // 0..g_cnt - 1
    const int tid        = int(threadIdx.x);
    const int tx         = tid & 15;
    const int ty         = tid >> 4;
    const int gm_base    = mb128 * 128;
    const int nl_base    = nb64_local * 64;

    s_lut[tid] = d_fp8_e4m3_to_fp16_bits(uint8_t(tid));
    __syncthreads();

    float acc[8][4] = {{0.0f}};

    for (int kb = 0; kb < 128; ++kb) {
        #pragma unroll
        for (int ld = 0; ld < 2; ++ld) {
            const int r  = (tid >> 2) + ld * 64;
            const int kp = (tid & 3) * 4;
            const int gm = gm_base + r;
            uint32_t px0 = 0u, px1 = 0u, px2 = 0u, px3 = 0u;
            if (gm < M) {
                const uint2 vx = *reinterpret_cast<const uint2*>(o_fp8x2 + (size_t(gm) * size_t(g_cnt) + size_t(g)) * 2048u + kb * 16 + kp);
                px0 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vx.x));
                px1 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vx.x >> 16));
                px2 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vx.y));
                px3 = d_lut_fp8x2_to_u32(s_lut, uint16_t(vx.y >> 16));
            }
            s_x[r][kp + 0] = px0; s_x[r][kp + 1] = px1; s_x[r][kp + 2] = px2; s_x[r][kp + 3] = px3;
        }
        {
            const int r  = tid >> 2;
            const int kp = (tid & 3) * 4;
            const int gn_global = g * 1024 + nl_base + r;
            const uint2 vw = *reinterpret_cast<const uint2*>(w_fp8x2 + size_t(gn_global) * 2048u + kb * 16 + kp);
            s_w_t[kp + 0][r] = d_lut_fp8x2_to_u32(s_lut, uint16_t(vw.x));
            s_w_t[kp + 1][r] = d_lut_fp8x2_to_u32(s_lut, uint16_t(vw.x >> 16));
            s_w_t[kp + 2][r] = d_lut_fp8x2_to_u32(s_lut, uint16_t(vw.y));
            s_w_t[kp + 3][r] = d_lut_fp8x2_to_u32(s_lut, uint16_t(vw.y >> 16));
        }
        if (tid < 128) {
            const int gm = gm_base + tid;
            s_xs[tid] = (gm < M) ? d_ue8m0_to_f32(o_scale[(size_t(gm) * size_t(g_cnt) + size_t(g)) * 128u + kb]) : 0.0f;
        } else if (tid < 130) {
            s_ws[tid - 128] = d_ue8m0_to_f32(w_scale[size_t(g * 32 + nb64_local * 2 + (tid - 128)) * 128u + kb]);
        }
        __syncthreads();

        float b[8][4] = {{0.0f}};
        #pragma unroll 2
        for (int p = 0; p < 16; ++p) {
            f16x2_t xv[8], wv[4];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t ux = s_x[ty * 8 + i][p];
                __builtin_memcpy(&xv[i], &ux, 4);
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t uw = s_w_t[p][j * 16 + tx];
                __builtin_memcpy(&wv[j], &uw, 4);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    b[i][j] = __builtin_amdgcn_fdot2(wv[j], xv[i], b[i][j], false);
                }
            }
        }

        const float sw0 = s_ws[0];
        const float sw1 = s_ws[1];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float sx = s_xs[ty * 8 + i];
            acc[i][0] += b[i][0] * (sw0 * sx);
            acc[i][1] += b[i][1] * (sw0 * sx);
            acc[i][2] += b[i][2] * (sw1 * sx);
            acc[i][3] += b[i][3] * (sw1 * sx);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int gm = gm_base + ty * 8 + i;
        if (gm < M) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int nl = nl_base + j * 16 + tx;
                out_bf16[(size_t(gm) * size_t(g_cnt) + size_t(g)) * 1024u + size_t(nl)] = d_f32_to_bf16_rne(acc[i][j]);
            }
        }
    }
}

// Coalesced 128-bit (uint4 = 8 BF16s/lane) multi-row BF16 GEMV/GEMM (M <= 8) with 4 KiB LDS X-tile broadcast!
template <int MAX_M>
__global__ __launch_bounds__(256, 4)
void k_bf16_gemv_mr(
    const uint16_t* __restrict__ x_bf16,
    const uint16_t* __restrict__ w_bf16,
    uint16_t* __restrict__ out_bf16,
    float*    __restrict__ out_f32,
    int M, int N, int K, bool write_f32)
{
    __shared__ uint4 s_x[MAX_M][32];

    const int nw   = int(threadIdx.x) >> 5;
    const int lane = int(threadIdx.x) & 31;
    const int n    = int(blockIdx.x) * 8 + nw;

    const uint16_t* wr = (n < N) ? (w_bf16 + size_t(n) * size_t(K)) : nullptr;
    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    const int K256 = K & ~255;
    for (int k_off = 0; k_off < K256; k_off += 256) {
        if (nw < MAX_M && nw < M) {
            s_x[nw][lane] = *reinterpret_cast<const uint4*>(x_bf16 + size_t(nw) * size_t(K) + size_t(k_off + lane * 8));
        }
        __syncthreads();

        if (n < N) {
            const uint4 w4 = *reinterpret_cast<const uint4*>(wr + k_off + lane * 8);
            const float wf0 = d_bf16_to_f32(uint16_t(w4.x));
            const float wf1 = d_bf16_to_f32(uint16_t(w4.x >> 16));
            const float wf2 = d_bf16_to_f32(uint16_t(w4.y));
            const float wf3 = d_bf16_to_f32(uint16_t(w4.y >> 16));
            const float wf4 = d_bf16_to_f32(uint16_t(w4.z));
            const float wf5 = d_bf16_to_f32(uint16_t(w4.z >> 16));
            const float wf6 = d_bf16_to_f32(uint16_t(w4.w));
            const float wf7 = d_bf16_to_f32(uint16_t(w4.w >> 16));

            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < M) {
                    const uint4 x4 = s_x[m][lane];
                    acc[m] += d_bf16_to_f32(uint16_t(x4.x))       * wf0
                            + d_bf16_to_f32(uint16_t(x4.x >> 16)) * wf1
                            + d_bf16_to_f32(uint16_t(x4.y))       * wf2
                            + d_bf16_to_f32(uint16_t(x4.y >> 16)) * wf3
                            + d_bf16_to_f32(uint16_t(x4.z))       * wf4
                            + d_bf16_to_f32(uint16_t(x4.z >> 16)) * wf5
                            + d_bf16_to_f32(uint16_t(x4.w))       * wf6
                            + d_bf16_to_f32(uint16_t(x4.w >> 16)) * wf7;
                }
            }
        }
        __syncthreads();
    }

    if (n >= N) return;

    const int K8 = K & ~7;
    for (int k = K256 + lane * 8; k < K8; k += 256) {
        const uint4 w4 = *reinterpret_cast<const uint4*>(wr + k);
        const float wf0 = d_bf16_to_f32(uint16_t(w4.x));
        const float wf1 = d_bf16_to_f32(uint16_t(w4.x >> 16));
        const float wf2 = d_bf16_to_f32(uint16_t(w4.y));
        const float wf3 = d_bf16_to_f32(uint16_t(w4.y >> 16));
        const float wf4 = d_bf16_to_f32(uint16_t(w4.z));
        const float wf5 = d_bf16_to_f32(uint16_t(w4.z >> 16));
        const float wf6 = d_bf16_to_f32(uint16_t(w4.w));
        const float wf7 = d_bf16_to_f32(uint16_t(w4.w >> 16));

        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const uint4 x4 = *reinterpret_cast<const uint4*>(x_bf16 + size_t(m) * size_t(K) + size_t(k));
                acc[m] += d_bf16_to_f32(uint16_t(x4.x))       * wf0
                        + d_bf16_to_f32(uint16_t(x4.x >> 16)) * wf1
                        + d_bf16_to_f32(uint16_t(x4.y))       * wf2
                        + d_bf16_to_f32(uint16_t(x4.y >> 16)) * wf3
                        + d_bf16_to_f32(uint16_t(x4.z))       * wf4
                        + d_bf16_to_f32(uint16_t(x4.z >> 16)) * wf5
                        + d_bf16_to_f32(uint16_t(x4.w))       * wf6
                        + d_bf16_to_f32(uint16_t(x4.w >> 16)) * wf7;
            }
        }
    }
    for (int k = K8 + lane; k < K; k += 32) {
        const float wf = d_bf16_to_f32(wr[k]);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) acc[m] += d_bf16_to_f32(x_bf16[size_t(m) * size_t(K) + size_t(k)]) * wf;
        }
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float sum = d_subwave_sum32(acc[m]);
            if (lane == 0) {
                const size_t idx = size_t(m) * size_t(N) + size_t(n);
                if (write_f32) out_f32[idx] = sum;
                else           out_bf16[idx] = d_f32_to_bf16_rne(sum);
            }
        }
    }
}

// Tiled BF16 GEMM for large M prefill (BM=32, BN=32, BK=32, 4 M-rows/thread, 8.25 KiB LDS, ScratchSize == 0!)
__global__ __launch_bounds__(256, 2)
void k_bf16_gemm_tiled32(
    const uint16_t* __restrict__ x_bf16,
    const uint16_t* __restrict__ w_bf16,
    uint16_t* __restrict__ out_bf16,
    float*    __restrict__ out_f32,
    int M, int N, int K, bool write_f32)
{
    __shared__ float s_x[32][33];
    __shared__ float s_w[32][33];

    const int nb = int(blockIdx.x) * 32;
    const int mb = int(blockIdx.y) * 32;
    const int tid = int(threadIdx.x);
    const int nl = tid & 31;           // 0..31 N-col
    const int m0 = (tid >> 5) * 4;     // 0..28 (4 M-rows/thread)

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int kb = 0; kb < K; kb += 32) {
        #pragma unroll
        for (int ld = 0; ld < 4; ++ld) {
            const int idx = tid + ld * 256;
            const int r = idx >> 5;    // 0..31
            const int c = idx & 31;    // 0..31 (32 adjacent threads read 64 contiguous bytes!)
            s_x[r][c] = (mb + r < M && kb + c < K) ? d_bf16_to_f32(x_bf16[size_t(mb + r) * size_t(K) + kb + c]) : 0.0f;
            s_w[r][c] = (nb + r < N && kb + c < K) ? d_bf16_to_f32(w_bf16[size_t(nb + r) * size_t(K) + kb + c]) : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            const float wv = s_w[nl][k];
            acc0 += s_x[m0 + 0][k] * wv;
            acc1 += s_x[m0 + 1][k] * wv;
            acc2 += s_x[m0 + 2][k] * wv;
            acc3 += s_x[m0 + 3][k] * wv;
        }
        __syncthreads();
    }

    if (nb + nl < N) {
        const float vals[4] = {acc0, acc1, acc2, acc3};
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int gm = mb + m0 + r;
            if (gm < M) {
                const size_t out_idx = size_t(gm) * size_t(N) + size_t(nb + nl);
                if (write_f32) out_f32[out_idx] = vals[r];
                else           out_bf16[out_idx] = d_f32_to_bf16_rne(vals[r]);
            }
        }
    }
}

__global__ __launch_bounds__(256)
void k_swiglu10_bf16(
    const uint16_t* __restrict__ gate_bf16,
    const uint16_t* __restrict__ up_bf16,
    uint16_t* __restrict__ out_bf16,
    int n_elems)
{
    const int idx = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (idx < n_elems) {
        const float g = d_bf16_to_f32(gate_bf16[idx]);
        const float u = d_bf16_to_f32(up_bf16[idx]);
        out_bf16[idx] = d_f32_to_bf16_rne(d_swiglu10(g, u));
    }
}

// High-throughput BM=64, BN=64, BK=16 4x4 FP32-FMA BF16 GEMM for M >= 32 prefill (8.5 KiB LDS, ScratchSize == 0!)
__global__ __launch_bounds__(256, 2)
void k_bf16_gemm_bm64_bn64(
    const uint16_t* __restrict__ x_bf16,
    const uint16_t* __restrict__ w_bf16,
    uint16_t* __restrict__ out_bf16,
    float*    __restrict__ out_f32,
    int M, int N, int K, bool write_f32)
{
    __shared__ float s_x[64][17];
    __shared__ float s_w[64][17];

    const int gn_base = int(blockIdx.x) * 64;
    const int gm_base = int(blockIdx.y) * 64;
    const int tid     = int(threadIdx.x);
    const int tx      = tid & 15;
    const int ty      = tid >> 4;

    float acc[4][4] = {{0.0f, 0.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 0.0f, 0.0f}};

    for (int kb = 0; kb < K; kb += 16) {
        #pragma unroll
        for (int ld = 0; ld < 4; ++ld) {
            const int idx = tid + ld * 256;
            const int r   = idx >> 4;   // 0..63
            const int c   = idx & 15;   // 0..15
            s_x[r][c] = (gm_base + r < M && kb + c < K) ? d_bf16_to_f32(x_bf16[size_t(gm_base + r) * size_t(K) + kb + c]) : 0.0f;
            s_w[r][c] = (gn_base + r < N && kb + c < K) ? d_bf16_to_f32(w_bf16[size_t(gn_base + r) * size_t(K) + kb + c]) : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            const float xv0 = s_x[ty * 4 + 0][k];
            const float xv1 = s_x[ty * 4 + 1][k];
            const float xv2 = s_x[ty * 4 + 2][k];
            const float xv3 = s_x[ty * 4 + 3][k];
            const float wv0 = s_w[tx * 4 + 0][k];
            const float wv1 = s_w[tx * 4 + 1][k];
            const float wv2 = s_w[tx * 4 + 2][k];
            const float wv3 = s_w[tx * 4 + 3][k];
            acc[0][0] += xv0 * wv0; acc[0][1] += xv0 * wv1; acc[0][2] += xv0 * wv2; acc[0][3] += xv0 * wv3;
            acc[1][0] += xv1 * wv0; acc[1][1] += xv1 * wv1; acc[1][2] += xv1 * wv2; acc[1][3] += xv1 * wv3;
            acc[2][0] += xv2 * wv0; acc[2][1] += xv2 * wv1; acc[2][2] += xv2 * wv2; acc[2][3] += xv2 * wv3;
            acc[3][0] += xv3 * wv0; acc[3][1] += xv3 * wv1; acc[3][2] += xv3 * wv2; acc[3][3] += xv3 * wv3;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int gm = gm_base + ty * 4 + i;
        if (gm < M) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int gn = gn_base + tx * 4 + j;
                if (gn < N) {
                    const size_t out_idx = size_t(gm) * size_t(N) + size_t(gn);
                    if (write_f32) out_f32[out_idx] = acc[i][j];
                    else           out_bf16[out_idx] = d_f32_to_bf16_rne(acc[i][j]);
                }
            }
        }
    }
}

// High-throughput BM=128, BN=64, BK=16 8x4 FP32-FMA BF16 GEMM for M >= 128 prefill (2x W-tile reuse + uint2 loads, 13.1 KiB LDS!)
__global__ __launch_bounds__(256, 2)
void k_bf16_gemm_bm128_bn64(
    const uint16_t* __restrict__ x_bf16,
    const uint16_t* __restrict__ w_bf16,
    uint16_t* __restrict__ out_bf16,
    float*    __restrict__ out_f32,
    int M, int N, int K, bool write_f32)
{
    __shared__ float s_x[128][17];
    __shared__ float s_w_t[16][65];

    const int gn_base = int(blockIdx.x) * 64;
    const int gm_base = int(blockIdx.y) * 128;
    const int tid     = int(threadIdx.x);
    const int tx      = tid & 15;
    const int ty      = tid >> 4;

    float acc[8][4] = {{0.0f}};

    for (int kb = 0; kb < K; kb += 16) {
        #pragma unroll
        for (int ld = 0; ld < 2; ++ld) {
            const int r = (tid >> 2) + ld * 64; // 0..127
            const int c = (tid & 3) * 4;        // 0, 4, 8, 12
            const int gm = gm_base + r;
            float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
            if (gm < M) {
                if (kb + c + 3 < K && ((K & 3) == 0)) {
                    const uint2 vx = *reinterpret_cast<const uint2*>(x_bf16 + size_t(gm) * size_t(K) + kb + c);
                    x0 = d_bf16_to_f32(uint16_t(vx.x));
                    x1 = d_bf16_to_f32(uint16_t(vx.x >> 16));
                    x2 = d_bf16_to_f32(uint16_t(vx.y));
                    x3 = d_bf16_to_f32(uint16_t(vx.y >> 16));
                } else {
                    if (kb + c + 0 < K) x0 = d_bf16_to_f32(x_bf16[size_t(gm) * size_t(K) + kb + c + 0]);
                    if (kb + c + 1 < K) x1 = d_bf16_to_f32(x_bf16[size_t(gm) * size_t(K) + kb + c + 1]);
                    if (kb + c + 2 < K) x2 = d_bf16_to_f32(x_bf16[size_t(gm) * size_t(K) + kb + c + 2]);
                    if (kb + c + 3 < K) x3 = d_bf16_to_f32(x_bf16[size_t(gm) * size_t(K) + kb + c + 3]);
                }
            }
            s_x[r][c + 0] = x0; s_x[r][c + 1] = x1; s_x[r][c + 2] = x2; s_x[r][c + 3] = x3;
        }
        {
            const int r = tid >> 2;             // 0..63
            const int c = (tid & 3) * 4;
            const int gn = gn_base + r;
            float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f, w3 = 0.0f;
            if (gn < N) {
                if (kb + c + 3 < K && ((K & 3) == 0)) {
                    const uint2 vw = *reinterpret_cast<const uint2*>(w_bf16 + size_t(gn) * size_t(K) + kb + c);
                    w0 = d_bf16_to_f32(uint16_t(vw.x));
                    w1 = d_bf16_to_f32(uint16_t(vw.x >> 16));
                    w2 = d_bf16_to_f32(uint16_t(vw.y));
                    w3 = d_bf16_to_f32(uint16_t(vw.y >> 16));
                } else {
                    if (kb + c + 0 < K) w0 = d_bf16_to_f32(w_bf16[size_t(gn) * size_t(K) + kb + c + 0]);
                    if (kb + c + 1 < K) w1 = d_bf16_to_f32(w_bf16[size_t(gn) * size_t(K) + kb + c + 1]);
                    if (kb + c + 2 < K) w2 = d_bf16_to_f32(w_bf16[size_t(gn) * size_t(K) + kb + c + 2]);
                    if (kb + c + 3 < K) w3 = d_bf16_to_f32(w_bf16[size_t(gn) * size_t(K) + kb + c + 3]);
                }
            }
            s_w_t[c + 0][r] = w0; s_w_t[c + 1][r] = w1; s_w_t[c + 2][r] = w2; s_w_t[c + 3][r] = w3;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            float xv[8], wv[4];
            #pragma unroll
            for (int i = 0; i < 8; ++i) xv[i] = s_x[ty * 8 + i][k];
            #pragma unroll
            for (int j = 0; j < 4; ++j) wv[j] = s_w_t[k][j * 16 + tx];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += xv[i] * wv[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int gm = gm_base + ty * 8 + i;
        if (gm < M) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int gn = gn_base + j * 16 + tx;
                if (gn < N) {
                    const size_t out_idx = size_t(gm) * size_t(N) + size_t(gn);
                    if (write_f32) out_f32[out_idx] = acc[i][j];
                    else           out_bf16[out_idx] = d_f32_to_bf16_rne(acc[i][j]);
                }
            }
        }
    }
}

// Fused SwiGLU10 on concatenated [gate (I), up (I)] + UE8M0_32 activation quantization!
__global__ __launch_bounds__(256)
void k_swiglu10_split_quant_ue8m0_32(
    const uint16_t* __restrict__ gateup_bf16, // [M, 2 * I]
    uint8_t* __restrict__ out_fp8,            // [M, I]
    uint8_t* __restrict__ out_ue8m0,          // [M, I / 32]
    int I, int total_groups)
{
    const int grp = int(blockIdx.x) * 8 + (int(threadIdx.x) >> 5);
    if (grp >= total_groups) return;
    const int lane = int(threadIdx.x) & 31;
    const int groups_per_row = I >> 5;
    const int m = grp / groups_per_row;
    const int c = (grp - m * groups_per_row) * 32 + lane;

    const uint16_t* row = gateup_bf16 + size_t(m) * size_t(2 * I);
    const float g = d_bf16_to_f32(row[c]);
    const float u = d_bf16_to_f32(row[I + c]);
    // Preserve exact intermediate BF16 rounding of standalone k_swiglu10_bf16 -> k_act_quant_ue8m0_32!
    const float x = d_bf16_to_f32(d_f32_to_bf16_rne(d_swiglu10(g, u)));

    const float amax = fmaxf(d_subwave_max32(fabsf(x)), 1e-4f);
    uint8_t ue8 = 0u;
    const float s = d_fast_round_scale_ue8m0(amax, 1.0f / 448.0f, ue8);
    const float q_in = fmaxf(-448.0f, fminf(448.0f, x / s));
    out_fp8[size_t(m) * size_t(I) + size_t(c)] = d_f32_to_fp8_e4m3(q_in);
    if (lane == 0) out_ue8m0[grp] = ue8;
}

// Dumitrescu §3.1.2 Def. 3.2 (Fn 7: real (c, s) with c^2+s^2=1, zero trig calls!) + §3.8.1 (Fn 28: disjoint rotations)
__device__ __forceinline__ void d_rope_sincos_f64_pair(
    int pos, int pair, bool use_comp_yarn, double& out_s, double& out_c)
{
    const int tbl = use_comp_yarn ? 1 : 0;
    const int upos = (pos >= 0) ? pos : -pos;
    double c = c_rope_d0_cos[tbl][upos & 15][pair];
    double s = c_rope_d0_sin[tbl][upos & 15][pair];
    int rem = upos >> 4;
    #pragma unroll
    for (int st = 0; st < 6; ++st) {
        const int dg = rem & 7;
        rem >>= 3;
        if (dg != 0) {
            const double tc = c_rope_dk_cos[tbl][st][dg - 1][pair];
            const double ts = c_rope_dk_sin[tbl][st][dg - 1][pair];
            const double nc = fma(c, tc, -s * ts);
            const double ns = fma(s, tc,  c * ts);
            c = nc;
            s = ns;
        }
    }
    out_c = c;
    out_s = (pos >= 0) ? s : -s;
}

__device__ __forceinline__ void d_rope_sincos_f64(
    int pos, int pair, bool use_comp_yarn, float& out_s, float& out_c)
{
    double sd, cd;
    d_rope_sincos_f64_pair(pos, pair, use_comp_yarn, sd, cd);
    out_s = float(sd);
    out_c = float(cd);
}

// Fused split + q_norm (1280) + kv_norm (512) + exact FP64-angle tail-64 RoPE + inplace UE8M0_32 quant on concatenated [qr; kv] [M, 1792]!
__global__ __launch_bounds__(256)
void k_split_qkv_norm_rope_quant(
    const uint16_t* __restrict__ qkv_bf16,     // [M, 1792]
    const uint16_t* __restrict__ q_norm_bf16,  // [1280]
    const uint16_t* __restrict__ kv_norm_bf16, // [512]
    bool use_comp_yarn,
    uint16_t*       __restrict__ out_qr_bf16,  // [M, 1280]
    uint16_t*       __restrict__ out_kv_bf16,  // [M, 512]
    int pos_base, float eps)
{
    __shared__ float    s_wave[4];
    __shared__ uint16_t s_kv[512];

    const int part = int(blockIdx.x); // 0 = qr (1280), 1 = kv (512)
    const int m    = int(blockIdx.y);
    const int tid  = int(threadIdx.x);

    if (part == 0) {
        const uint16_t* qr_in = qkv_bf16 + size_t(m) * 1792u;
        uint16_t* qr_out = out_qr_bf16 + size_t(m) * 1280u;
        float sq = 0.0f;
        for (int d = tid; d < 1280; d += 256) {
            const float v = d_bf16_to_f32(qr_in[d]);
            sq += v * v;
        }
        sq = d_wave_sum64(sq);
        if ((tid & 63) == 0) s_wave[tid >> 6] = sq;
        __syncthreads();
        const float rstd = rsqrtf((s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3]) * (1.0f / 1280.0f) + eps);
        for (int d = tid; d < 1280; d += 256) {
            qr_out[d] = d_f32_to_bf16_rne(d_bf16_to_f32(qr_in[d]) * rstd * d_bf16_to_f32(q_norm_bf16[d]));
        }
    } else {
        const uint16_t* kv_in = qkv_bf16 + size_t(m) * 1792u + 1280u;
        uint16_t* kv_out = out_kv_bf16 + size_t(m) * 512u;

        // 1. RMSNorm(512) -> BF16 in s_kv
        const float v0 = d_bf16_to_f32(kv_in[tid]);
        const float v1 = d_bf16_to_f32(kv_in[256 + tid]);
        float sq = d_wave_sum64(v0 * v0 + v1 * v1);
        if ((tid & 63) == 0) s_wave[tid >> 6] = sq;
        __syncthreads();
        const float rstd = rsqrtf((s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3]) * (1.0f / 512.0f) + eps);
        s_kv[tid]       = d_f32_to_bf16_rne(v0 * rstd * d_bf16_to_f32(kv_norm_bf16[tid]));
        s_kv[256 + tid] = d_f32_to_bf16_rne(v1 * rstd * d_bf16_to_f32(kv_norm_bf16[256 + tid]));
        __syncthreads();

        // 2. Exact FP64-angle Tail-64 RoPE at pos = pos_base + m
        if (tid < 32) {
            const int pos = pos_base + m;
            float s, c;
            d_rope_sincos_f64(pos, tid, use_comp_yarn, s, c);
            const float x0 = d_bf16_to_f32(s_kv[448 + 2 * tid + 0]);
            const float x1 = d_bf16_to_f32(s_kv[448 + 2 * tid + 1]);
            s_kv[448 + 2 * tid + 0] = d_f32_to_bf16_rne(x0 * c - x1 * s);
            s_kv[448 + 2 * tid + 1] = d_f32_to_bf16_rne(x0 * s + x1 * c);
        }
        __syncthreads();

        // 3. Inplace UE8M0_32 FP8 roundtrip -> out_kv_bf16
        #pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int d = pass * 256 + tid;
            const float x = d_bf16_to_f32(s_kv[d]);
            const float amax = fmaxf(d_subwave_max32(fabsf(x)), 1e-4f);
            uint8_t ue8 = 0u;
            const float s = d_fast_round_scale_ue8m0(amax, 1.0f / 448.0f, ue8);
            const uint8_t fp8 = d_f32_to_fp8_e4m3(fmaxf(-448.0f, fminf(448.0f, x / s)));
            kv_out[d] = d_f32_to_bf16_rne(d_fp8_e4m3_to_f32(fp8) * s);
        }
    }
}

// =============================================================================
// 5. Exact FP64-Angle Register RoPE / YaRN, Compressor, Indexer, and Sparse Attention
// =============================================================================
__global__ __launch_bounds__(32)
void k_rope_tail64_bf16(
    uint16_t* __restrict__ x_bf16,
    bool use_comp_yarn,
    int H, int D, int pos_base, int pos_stride, bool inverse)
{
    const int pair = int(threadIdx.x);  // 0..31
    const int h = int(blockIdx.x);      // 0..H-1
    const int m = int(blockIdx.y);      // 0..M-1
    const int pos = pos_base + m * pos_stride;

    float s_raw, c;
    d_rope_sincos_f64(pos, pair, use_comp_yarn, s_raw, c);
    const float s = inverse ? -s_raw : s_raw;

    uint16_t* tail = x_bf16 + (size_t(m) * size_t(H) + size_t(h)) * size_t(D) + size_t(D - 64);
    const float x0 = d_bf16_to_f32(tail[2 * pair + 0]);
    const float x1 = d_bf16_to_f32(tail[2 * pair + 1]);
    tail[2 * pair + 0] = d_f32_to_bf16_rne(x0 * c - x1 * s);
    tail[2 * pair + 1] = d_f32_to_bf16_rne(x0 * s + x1 * c);
}

// Disjoint 64-row Tile Orthogonal Givens RoPE (Dumitrescu Fn 7 + Fn 28):
// Grid: ((H + 3) / 4, (M + 63) / 64), Block: 256 = 8 M-rows x 32 frequency pairs, 2 KiB LDS (4 CTAs/CU)!
__global__ __launch_bounds__(256, 4)
void k_rope_tail64_givens_bf16(
    uint16_t* __restrict__ x_bf16,
    bool use_comp_yarn,
    int M, int H, int D, int pos_base, bool inverse)
{
    __shared__ double s_anc_c[4][32];
    __shared__ double s_anc_s[4][32];

    const int tid = int(threadIdx.x);
    const int tile64 = int(blockIdx.y);
    const int m_tile0 = tile64 * 64;
    const int h0 = int(blockIdx.x) * 4;
    const int h1 = min(H, h0 + 4);
    const int tbl = use_comp_yarn ? 1 : 0;

    if (tid < 128) {
        const int sub = tid >> 5;
        const int pair = tid & 31;
        double as, ac;
        d_rope_sincos_f64_pair(pos_base + m_tile0 + sub * 16, pair, use_comp_yarn, as, ac);
        s_anc_c[sub][pair] = ac;
        s_anc_s[sub][pair] = as;
    }
    __syncthreads();

    const int pair = tid & 31;
    const int r_lane = tid >> 5;  // 0..7
    for (int pass = 0; pass < 8; ++pass) {
        const int row_local = pass * 8 + r_lane;
        const int m = m_tile0 + row_local;
        if (m >= M) break;

        const int sub = row_local >> 4;
        const int m_lo = row_local & 15;
        const double ac = s_anc_c[sub][pair];
        const double as = s_anc_s[sub][pair];
        const double dc = c_rope_d0_cos[tbl][m_lo][pair];
        const double ds = c_rope_d0_sin[tbl][m_lo][pair];
        const float c = float(fma(ac, dc, -as * ds));
        const float s_raw = float(fma(as, dc, ac * ds));
        const float s = inverse ? -s_raw : s_raw;

        for (int h = h0; h < h1; ++h) {
            uint32_t* tail32 = reinterpret_cast<uint32_t*>(
                x_bf16 + (size_t(m) * size_t(H) + size_t(h)) * size_t(D) + size_t(D - 64));
            const uint32_t pak = tail32[pair];
            const float x0 = __uint_as_float(pak << 16);
            const float x1 = __uint_as_float(pak & 0xFFFF0000u);
            const uint16_t y0 = d_f32_to_bf16_rne(x0 * c - x1 * s);
            const uint16_t y1 = d_f32_to_bf16_rne(x0 * s + x1 * c);
            tail32[pair] = uint32_t(y0) | (uint32_t(y1) << 16);
        }
    }
}

__global__ __launch_bounds__(256)
void k_compressor_r2_pool(
    const float* __restrict__ kv_f32,
    const float* __restrict__ score_f32,
    float* __restrict__ kv_state,
    float* __restrict__ score_state,
    uint16_t* __restrict__ out_pool_bf16,
    int start_pos, int M)
{
    // Grid: (2, 1), Block: 256 -> covers all 512 channels independently across M steps!
    const int d = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (d >= 512) return;

    float k0 = kv_state[d];
    float k1 = kv_state[512 + d];
    float s0 = score_state[d];
    float s1 = score_state[512 + d];

    int emitted = 0;
    for (int m = 0; m < M; ++m) {
        const int pos = start_pos + m;
        const float kv = kv_f32[size_t(m) * 512u + size_t(d)];
        const float sc = score_f32[size_t(m) * 512u + size_t(d)];
        if ((pos & 1) == 0) {
            k0 = kv;
            s0 = sc;
        } else {
            k1 = kv;
            s1 = sc;
            const float smax = fmaxf(s0, s1);
            const float e0 = __expf(s0 - smax);
            const float e1 = __expf(s1 - smax);
            const float inv = 1.0f / (e0 + e1);
            const float pooled = k0 * (e0 * inv) + k1 * (e1 * inv);
            out_pool_bf16[size_t(emitted) * 512u + size_t(d)] = d_f32_to_bf16_rne(pooled);
            ++emitted;
        }
    }
    kv_state[d] = k0;
    kv_state[512 + d] = k1;
    score_state[d] = s0;
    score_state[512 + d] = s1;
}

// Exact compressed KV quantization: block_size=16, scale_dtype=FP8_E4M3 (kernel.py:160-163)
__global__ __launch_bounds__(256)
void k_pack_compress_kv_fp4_e4m3_16(
    const uint16_t* __restrict__ latent_bf16, // [N_lat, 512]
    uint8_t* __restrict__ cache_fp4,          // [max_comp, 256]
    uint8_t* __restrict__ cache_e4m3,         // [max_comp, 32]
    int slot_offset)
{
    // Block: 256 threads = 32 groups of 8 threads (each thread handles 2 elements = 1 packed FP4 byte!)
    const int n = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int grp = tid >> 3;   // 0..31 (512 / 16)
    const int lane = tid & 7;   // 0..7

    const uint16_t* src = latent_bf16 + size_t(n) * 512u + grp * 16 + lane * 2;
    const float v0 = d_bf16_to_f32(src[0]);
    const float v1 = d_bf16_to_f32(src[1]);

    float amax = fmaxf(fabsf(v0), fabsf(v1));
    #pragma unroll
    for (int mask = 4; mask >= 1; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, mask, 8));
    }
    amax = fmaxf(amax, 6.0f * (1.0f / 512.0f));  // 6 * 2^-9 per kernel.py:162
    const uint8_t s_e4m3 = d_f32_to_fp8_e4m3(amax * (1.0f / 6.0f));
    const float s = d_fp8_e4m3_to_f32(s_e4m3);

    const uint8_t q0 = d_f32_to_fp4_e2m1(fmaxf(-6.0f, fminf(6.0f, v0 / s)));
    const uint8_t q1 = d_f32_to_fp4_e2m1(fmaxf(-6.0f, fminf(6.0f, v1 / s)));

    const size_t dst_slot = size_t(slot_offset + n);
    cache_fp4[dst_slot * 256u + grp * 8 + lane] = uint8_t(q0 | (q1 << 4));
    if (lane == 0) {
        cache_e4m3[dst_slot * 32u + grp] = s_e4m3;
    }
}

// Packed 68 B/slot Indexer K cache: 64 B FP4_x2 + 4 B UE8M0 (exact 890 B/tok Global KV!)
__global__ __launch_bounds__(256)
void k_pack_index_k_fp4_ue8m0_32(
    const uint16_t* __restrict__ k_bf16,   // [N_lat, 128]
    uint8_t* __restrict__ cache_fp4,       // [max_comp, 64]
    uint8_t* __restrict__ cache_ue8m0,     // [max_comp, 4]
    int slot_offset, int total_groups)
{
    const int grp = int(blockIdx.x) * 8 + (int(threadIdx.x) >> 5);
    if (grp >= total_groups) return;
    const int lane = int(threadIdx.x) & 31;
    const int n = grp >> 2;  // 4 groups of 32 per 128-dim row
    const int g = grp & 3;

    const float x = d_bf16_to_f32(k_bf16[size_t(grp) * 32u + size_t(lane)]);
    const float amax = fmaxf(d_subwave_max32(fabsf(x)), 7.0529156e-38f);
    uint8_t ue8 = 0u;
    const float s = d_fast_round_scale_ue8m0(amax, 1.0f / 6.0f, ue8);
    const uint32_t nib = uint32_t(d_f32_to_fp4_e2m1(fmaxf(-6.0f, fminf(6.0f, x / s))));
    const uint32_t nib_hi = __shfl_xor(nib, 1, 32);

    const size_t dst_slot = size_t(slot_offset + n);
    if ((lane & 1) == 0) {
        cache_fp4[dst_slot * 64u + size_t(g * 16 + (lane >> 1))] = uint8_t(nib | (nib_hi << 4));
    }
    if (lane == 0) {
        cache_ue8m0[dst_slot * 4u + size_t(g)] = ue8;
    }
}

// Exact Indexer q quantization: block_size=32, scale_dtype=FE8M0, inplace=True (kernel.py:164-171)
__global__ __launch_bounds__(256)
void k_fp4_quant_dequant_ue8m0_32_inplace(uint16_t* __restrict__ x_bf16, int total_groups) {
    const int grp = int(blockIdx.x) * 8 + (int(threadIdx.x) >> 5);
    if (grp >= total_groups) return;
    const int lane = int(threadIdx.x) & 31;
    const size_t idx = size_t(grp) * 32u + size_t(lane);

    const float x = d_bf16_to_f32(x_bf16[idx]);
    // 6 * 2^-126 = 7.0529156e-38f
    const float amax = fmaxf(d_subwave_max32(fabsf(x)), 7.0529156e-38f);
    uint8_t ue8 = 0u;
    const float s = d_fast_round_scale_ue8m0(amax, 1.0f / 6.0f, ue8);
    const uint8_t nib = d_f32_to_fp4_e2m1(fmaxf(-6.0f, fminf(6.0f, x / s)));
    x_bf16[idx] = d_f32_to_bf16_rne(d_fp4_e2m1_to_f32(nib) * s);
}

// Indexer score over packed 68 B/slot Indexer K (4x less HBM2 BW than BF16!):
// Grid: ((T_comp + 255) / 256, M), Block: 256 threads (1 compressed key slot t per thread)
__global__ __launch_bounds__(256, 4)
void k_indexer_score(
    const uint16_t* __restrict__ q_bf16,       // [M, ih_cnt, 128]
    const uint8_t*  __restrict__ k_fp4,        // [T_comp, 64]
    const uint8_t*  __restrict__ k_ue8m0,      // [T_comp, 4]
    const uint16_t* __restrict__ weights_bf16, // [M, ih_cnt]
    float* __restrict__ out_score,             // [M, T_comp]
    int ih_cnt, int T_comp)
{
    __shared__ float s_q[11][128];
    __shared__ float s_w[11];

    const int t = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int m = int(blockIdx.y);
    const int tid = int(threadIdx.x);

    for (int idx = tid; idx < ih_cnt * 128; idx += 256) {
        const int h = idx >> 7;
        const int d = idx & 127;
        s_q[h][d] = d_bf16_to_f32(q_bf16[(size_t(m) * size_t(ih_cnt) + size_t(h)) * 128u + size_t(d)]);
    }
    if (tid < ih_cnt) {
        s_w[tid] = d_bf16_to_f32(weights_bf16[size_t(m) * size_t(ih_cnt) + size_t(tid)]) * 0.015625f;
    }
    __syncthreads();

    if (t >= T_comp) return;

    const uint32_t sc4 = *reinterpret_cast<const uint32_t*>(k_ue8m0 + size_t(t) * 4u);
    const uint4* kp4   = reinterpret_cast<const uint4*>(k_fp4 + size_t(t) * 64u);

    float dots[11];
    #pragma unroll
    for (int h = 0; h < 11; ++h) dots[h] = 0.0f;

    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        const float sc = d_ue8m0_to_f32(uint8_t((sc4 >> (g * 8)) & 0xFFu));
        const uint4 q16 = kp4[g];
        const uint32_t words[4] = {q16.x, q16.y, q16.z, q16.w};
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t pk = words[w];
            const int d = g * 32 + w * 8;
            const float kv0 = d_fp4_e2m1_to_f32(uint8_t((pk >>  0) & 0xFu)) * sc;
            const float kv1 = d_fp4_e2m1_to_f32(uint8_t((pk >>  4) & 0xFu)) * sc;
            const float kv2 = d_fp4_e2m1_to_f32(uint8_t((pk >>  8) & 0xFu)) * sc;
            const float kv3 = d_fp4_e2m1_to_f32(uint8_t((pk >> 12) & 0xFu)) * sc;
            const float kv4 = d_fp4_e2m1_to_f32(uint8_t((pk >> 16) & 0xFu)) * sc;
            const float kv5 = d_fp4_e2m1_to_f32(uint8_t((pk >> 20) & 0xFu)) * sc;
            const float kv6 = d_fp4_e2m1_to_f32(uint8_t((pk >> 24) & 0xFu)) * sc;
            const float kv7 = d_fp4_e2m1_to_f32(uint8_t((pk >> 28) & 0xFu)) * sc;
            #pragma unroll
            for (int h = 0; h < 11; ++h) {
                if (h < ih_cnt) {
                    dots[h] += s_q[h][d + 0] * kv0 + s_q[h][d + 1] * kv1 + s_q[h][d + 2] * kv2 + s_q[h][d + 3] * kv3 +
                               s_q[h][d + 4] * kv4 + s_q[h][d + 5] * kv5 + s_q[h][d + 6] * kv6 + s_q[h][d + 7] * kv7;
                }
            }
        }
    }

    float total = 0.0f;
    #pragma unroll
    for (int h = 0; h < 11; ++h) {
        if (h < ih_cnt) total += fmaxf(dots[h], 0.0f) * s_w[h];
    }
    out_score[size_t(m) * size_t(T_comp) + size_t(t)] = total;
}

// Multi-row FP4_E2M1 + UE8M0 HSI Lightning Indexer (reads each 68 B compressed key ONCE across MAX_M query rows = 6x HBM2 BW cut!):
// Holds the 68 B key in 17 VGPRs and evaluates mr = 0..m_rows-1 sequentially with 11 accumulator VGPRs -> ScratchSize == 0!
template <int MAX_M>
__global__ __launch_bounds__(256, 2)
void k_indexer_score_msplit(
    const uint16_t* __restrict__ q_bf16,       // [M, ih_cnt, 128]
    const uint8_t*  __restrict__ k_fp4,        // [T_comp, 64]
    const uint8_t*  __restrict__ k_ue8m0,      // [T_comp, 4]
    const uint16_t* __restrict__ weights_bf16, // [M, ih_cnt]
    float* __restrict__ out_score,             // [M, T_comp]
    int M, int ih_cnt, int T_comp)
{
    __shared__ uint32_t s_q2[MAX_M][11][64];
    __shared__ float s_w[MAX_M][11];

    const int t = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int m_base = int(blockIdx.y) * MAX_M;
    const int m_rows = min(MAX_M, M - m_base);
    const int tid = int(threadIdx.x);

    const int total_q2 = m_rows * ih_cnt * 64;
    for (int idx = tid; idx < total_q2; idx += 256) {
        const int d2 = idx & 63;
        const int mh = idx >> 6;
        const int mr = mh / ih_cnt;
        const int h = mh - mr * ih_cnt;
        const uint32_t* qp = reinterpret_cast<const uint32_t*>(
            q_bf16 + (size_t(m_base + mr) * size_t(ih_cnt) + size_t(h)) * 128u);
        s_q2[mr][h][d2] = qp[d2];
    }
    if (tid < m_rows * ih_cnt) {
        const int mr = tid / ih_cnt;
        const int h = tid - mr * ih_cnt;
        s_w[mr][h] = d_bf16_to_f32(weights_bf16[size_t(m_base + mr) * size_t(ih_cnt) + size_t(h)]) * 0.015625f;
    }
    __syncthreads();

    if (t >= T_comp) return;

    const uint32_t sc4 = *reinterpret_cast<const uint32_t*>(k_ue8m0 + size_t(t) * 4u);
    const uint4* kp4 = reinterpret_cast<const uint4*>(k_fp4 + size_t(t) * 64u);

    float dots[MAX_M][11];
    #pragma unroll
    for (int mr = 0; mr < MAX_M; ++mr) {
        #pragma unroll
        for (int h = 0; h < 11; ++h) dots[mr][h] = 0.0f;
    }

    #pragma unroll 1
    for (int g = 0; g < 4; ++g) {
        const float sc = d_ue8m0_to_f32(uint8_t((sc4 >> (g * 8)) & 0xFFu));
        const uint4 q16 = kp4[g];
        #pragma unroll 1
        for (int w = 0; w < 4; ++w) {
            const uint32_t pk = (w == 0) ? q16.x : ((w == 1) ? q16.y : ((w == 2) ? q16.z : q16.w));
            const int d2_base = g * 16 + w * 4;
            const float kv0 = d_fp4_e2m1_to_f32(uint8_t((pk >>  0) & 0xFu)) * sc;
            const float kv1 = d_fp4_e2m1_to_f32(uint8_t((pk >>  4) & 0xFu)) * sc;
            const float kv2 = d_fp4_e2m1_to_f32(uint8_t((pk >>  8) & 0xFu)) * sc;
            const float kv3 = d_fp4_e2m1_to_f32(uint8_t((pk >> 12) & 0xFu)) * sc;
            const float kv4 = d_fp4_e2m1_to_f32(uint8_t((pk >> 16) & 0xFu)) * sc;
            const float kv5 = d_fp4_e2m1_to_f32(uint8_t((pk >> 20) & 0xFu)) * sc;
            const float kv6 = d_fp4_e2m1_to_f32(uint8_t((pk >> 24) & 0xFu)) * sc;
            const float kv7 = d_fp4_e2m1_to_f32(uint8_t((pk >> 28) & 0xFu)) * sc;

            #pragma unroll
            for (int mr = 0; mr < MAX_M; ++mr) {
                if (mr < m_rows) {
                    #pragma unroll
                    for (int h = 0; h < 11; ++h) {
                        if (h < ih_cnt) {
                            const uint32_t qp0 = s_q2[mr][h][d2_base + 0];
                            const uint32_t qp1 = s_q2[mr][h][d2_base + 1];
                            const uint32_t qp2 = s_q2[mr][h][d2_base + 2];
                            const uint32_t qp3 = s_q2[mr][h][d2_base + 3];
                            dots[mr][h] += __uint_as_float(qp0 << 16) * kv0 + __uint_as_float(qp0 & 0xFFFF0000u) * kv1 +
                                           __uint_as_float(qp1 << 16) * kv2 + __uint_as_float(qp1 & 0xFFFF0000u) * kv3 +
                                           __uint_as_float(qp2 << 16) * kv4 + __uint_as_float(qp2 & 0xFFFF0000u) * kv5 +
                                           __uint_as_float(qp3 << 16) * kv6 + __uint_as_float(qp3 & 0xFFFF0000u) * kv7;
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int mr = 0; mr < MAX_M; ++mr) {
        if (mr < m_rows) {
            float total = 0.0f;
            #pragma unroll
            for (int h = 0; h < 11; ++h) {
                if (h < ih_cnt) total += fmaxf(dots[mr][h], 0.0f) * s_w[mr][h];
            }
            out_score[size_t(m_base + mr) * size_t(T_comp) + size_t(t)] = total;
        }
    }
}

// Device helper: 4-pass 8-bit MSB->LSB radix select to find exact k_keep-th largest ordered uint32 key!
__device__ uint32_t d_radix_kth_largest(
    const float* __restrict__ scores,
    int len,
    int k_keep,
    uint32_t* s_hist,     // [256]
    uint32_t* s_scalar)   // [2]
{
    const int tid = int(threadIdx.x);
    uint32_t prefix = 0u;
    uint32_t mask = 0u;
    int rem_k = k_keep;

    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - pass * 8;
        s_hist[tid] = 0u;
        __syncthreads();

        for (int i = tid; i < len; i += 256) {
            const uint32_t u = d_float_to_ordered_u32(scores[i]);
            if ((u & mask) == prefix) {
                atomicAdd(&s_hist[(u >> shift) & 0xFFu], 1u);
            }
        }
        __syncthreads();

        if (tid == 0) {
            int acc = 0;
            uint32_t chosen_bucket = 0u;
            for (int b = 255; b >= 0; --b) {
                const int cnt = int(s_hist[b]);
                if (acc + cnt >= rem_k) {
                    chosen_bucket = uint32_t(b);
                    rem_k -= acc;
                    break;
                }
                acc += cnt;
            }
            s_scalar[0] = chosen_bucket;
            s_scalar[1] = uint32_t(rem_k);
        }
        __syncthreads();

        prefix |= (s_scalar[0] << shift);
        mask |= (0xFFu << shift);
        rem_k = int(s_scalar[1]);
    }
    return prefix;
}

// 1 CTA of 256 threads per query row m: handles causal mask + compact 1B/8-key candidate_blocks (Layer 20 / 24..36) + exact top-512 + sort!
__global__ __launch_bounds__(256)
void k_indexer_candidates_and_topk(
    float* __restrict__ score_f32,
    uint8_t* __restrict__ cand_block_mask,
    int cand_block_stride,
    int32_t* __restrict__ out_comp_idxs,
    int T_comp, int start_pos, int ratio, int mode, int K_out,
    float* __restrict__ block_scores_scratch) // [M, ceil(T_comp/8)] when mode==1
{
    __shared__ uint32_t s_hist[256];
    __shared__ uint32_t s_scalar[4];
    __shared__ int32_t  s_sel[512];

    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    float* row = score_f32 + size_t(m) * size_t(T_comp);
    const int compress_len = (start_pos + m + 1) / ratio;
    const float neg_inf = -INFINITY;
    const float pos_inf = INFINITY;

    // 1. Causal mask: t >= compress_len -> -inf
    for (int t = tid; t < T_comp; t += 256) {
        if (t >= compress_len) row[t] = neg_inf;
    }
    __syncthreads();

    // 2. Candidate block producer (Layer 20, mode==1) or consumer (Layers 24..36, mode==2)
    if (mode == 1) {
        const int num_blocks = (T_comp + 7) >> 3;
        float* bscores = block_scores_scratch + size_t(m) * size_t(num_blocks);
        const int last_blk = (compress_len > 0) ? ((compress_len - 1) >> 3) : -1;

        for (int b = tid; b < num_blocks; b += 256) {
            if (b == last_blk) {
                bscores[b] = pos_inf;
            } else {
                float mx = neg_inf;
                const int base = b * 8;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (base + j < T_comp) mx = fmaxf(mx, row[base + j]);
                }
                bscores[b] = mx;
            }
        }
        __syncthreads();

        uint8_t* cblk = cand_block_mask + size_t(m) * size_t(cand_block_stride);
        const int k_blk = min(2048, num_blocks);
        if (num_blocks <= 2048) {
            for (int b = tid; b < num_blocks; b += 256) {
                cblk[b] = (bscores[b] > neg_inf) ? 1u : 0u;
            }
        } else {
            const uint32_t kth_u = d_radix_kth_largest(bscores, num_blocks, k_blk, s_hist, s_scalar);
            const uint32_t neg_inf_u = d_float_to_ordered_u32(neg_inf);
            if (tid == 0) {
                s_scalar[2] = 0u;  // count strictly greater
            }
            __syncthreads();
            for (int b = tid; b < num_blocks; b += 256) {
                const uint32_t u = d_float_to_ordered_u32(bscores[b]);
                if (u > kth_u && u > neg_inf_u) atomicAdd(&s_scalar[2], 1u);
            }
            __syncthreads();
            if (tid == 0) {
                s_scalar[3] = (k_blk > int(s_scalar[2])) ? uint32_t(k_blk - int(s_scalar[2])) : 0u;
            }
            __syncthreads();
            for (int b = tid; b < num_blocks; b += 256) {
                const uint32_t u = d_float_to_ordered_u32(bscores[b]);
                bool keep = false;
                if (u > neg_inf_u) {
                    if (u > kth_u) {
                        keep = true;
                    } else if (u == kth_u) {
                        const uint32_t prev = atomicAdd(&s_scalar[3], 0xFFFFFFFFu);
                        if (int(prev) > 0) keep = true;
                    }
                }
                cblk[b] = keep ? 1u : 0u;
            }
        }
        __syncthreads();
    } else if (mode == 2) {
        const uint8_t* cblk = cand_block_mask + size_t(m) * size_t(cand_block_stride);
        for (int t = tid; t < T_comp; t += 256) {
            if (!cblk[t >> 3]) row[t] = neg_inf;
        }
        __syncthreads();
    }

    // 3. Top-K (K_out = min(512, T_comp)) over row[0..T_comp-1] + sort indices ascending!
    if (K_out <= 0) return;
    const int valid_k = min(K_out, compress_len);
    s_sel[tid] = 0x7FFFFFFF;
    s_sel[256 + tid] = 0x7FFFFFFF;
    __syncthreads();

    if (valid_k > 0) {
        const uint32_t neg_inf_u = d_float_to_ordered_u32(neg_inf);
        const uint32_t kth_u = d_radix_kth_largest(row, compress_len, valid_k, s_hist, s_scalar);

        if (tid == 0) {
            s_scalar[0] = 0u;  // write cursor
            s_scalar[1] = 0u;  // strictly greater count
        }
        __syncthreads();

        for (int t = tid; t < compress_len; t += 256) {
            const uint32_t u = d_float_to_ordered_u32(row[t]);
            if (u > kth_u && u > neg_inf_u) {
                const uint32_t pos = atomicAdd(&s_scalar[0], 1u);
                if (pos < 512u) s_sel[pos] = t;
            }
        }
        __syncthreads();

        if (tid == 0) {
            const int have = int(s_scalar[0]);
            s_scalar[2] = (valid_k > have) ? uint32_t(valid_k - have) : 0u;
        }
        __syncthreads();

        if (s_scalar[2] > 0u) {
            for (int t = tid; t < compress_len; t += 256) {
                const uint32_t u = d_float_to_ordered_u32(row[t]);
                if (u == kth_u && u > neg_inf_u) {
                    const uint32_t rem = atomicAdd(&s_scalar[2], 0xFFFFFFFFu);
                    if (int(rem) > 0) {
                        const uint32_t pos = atomicAdd(&s_scalar[0], 1u);
                        if (pos < 512u) s_sel[pos] = t;
                    }
                }
            }
            __syncthreads();
        }

        // Bitonic sort 512 elements in s_sel ascending across 256 threads
        for (int k = 2; k <= 512; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                const int ixj = tid ^ j;
                if (ixj > tid) {
                    const bool asc = ((tid & k) == 0);
                    const int32_t a = s_sel[tid];
                    const int32_t b = s_sel[ixj];
                    if ((a > b) == asc) {
                        s_sel[tid] = b;
                        s_sel[ixj] = a;
                    }
                }
                const int tid2 = tid + 256;
                const int ixj2 = tid2 ^ j;
                if (ixj2 > tid2) {
                    const bool asc2 = ((tid2 & k) == 0);
                    const int32_t a2 = s_sel[tid2];
                    const int32_t b2 = s_sel[ixj2];
                    if ((a2 > b2) == asc2) {
                        s_sel[tid2] = b2;
                        s_sel[ixj2] = a2;
                    }
                }
                __syncthreads();
            }
        }
    }

    int32_t* dst = out_comp_idxs + size_t(m) * size_t(K_out);
    for (int k = tid; k < K_out; k += 256) {
        const int32_t v = s_sel[k];
        dst[k] = (v < compress_len) ? v : -1;
    }
}

__device__ __forceinline__ void d_load_win_kv_pak(
    const uint16_t* src, int d_lane, uint32_t* kv_pak_slot)
{
    const uint4 v0 = *reinterpret_cast<const uint4*>(src + 0);
    const uint4 v1 = *reinterpret_cast<const uint4*>(src + 8);
    kv_pak_slot[0 * 32 + d_lane] = v0.x;
    kv_pak_slot[1 * 32 + d_lane] = v0.y;
    kv_pak_slot[2 * 32 + d_lane] = v0.z;
    kv_pak_slot[3 * 32 + d_lane] = v0.w;
    kv_pak_slot[4 * 32 + d_lane] = v1.x;
    kv_pak_slot[5 * 32 + d_lane] = v1.y;
    kv_pak_slot[6 * 32 + d_lane] = v1.z;
    kv_pak_slot[7 * 32 + d_lane] = v1.w;
}

__device__ __forceinline__ void d_load_comp_kv_pak(
    const uint8_t* comp_fp4, const uint8_t* comp_e4m3, const uint16_t* s_lut,
    int32_t kv_idx, int d_lane, uint32_t* kv_pak_slot)
{
    const float sc = d_fp16_bits_to_f32(s_lut[comp_e4m3[size_t(kv_idx) * 32u + size_t(d_lane)]]);
    const uint2 pb8 = *reinterpret_cast<const uint2*>(comp_fp4 + size_t(kv_idx) * 256u + d_lane * 8);
    const uint32_t words[2] = {pb8.x, pb8.y};
    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        const uint32_t wd = words[w];
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t byte = (wd >> (b * 8)) & 0xFFu;
            const uint16_t lo = d_f32_to_bf16_rne(d_fp4_e2m1_to_f32(uint8_t(byte & 0xFu)) * sc);
            const uint16_t hi = d_f32_to_bf16_rne(d_fp4_e2m1_to_f32(uint8_t(byte >> 4)) * sc);
            kv_pak_slot[(w * 4 + b) * 32 + d_lane] = uint32_t(lo) | (uint32_t(hi) << 16);
        }
    }
}

__device__ __forceinline__ void d_sparse_attn_step_kv_pak(
    const float (&q_reg)[16],
    float (&acc_o)[16],
    float& m_max,
    float& l_sum,
    const uint32_t* kv_pak_slot,
    int d_lane)
{
    constexpr float kScale = 0.04419417382415922f;  // 512^-0.5
    float kv[16];
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const uint32_t u = kv_pak_slot[p * 32 + d_lane];
        kv[2 * p + 0] = __uint_as_float(u << 16);
        kv[2 * p + 1] = __uint_as_float(u & 0xFFFF0000u);
    }
    float dot = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) dot += q_reg[j] * kv[j];
    dot = d_subwave_sum32(dot) * kScale;

    if (dot <= m_max) {
        const float p = __expf(dot - m_max);
        l_sum += p;
        const float p_cast = d_bf16_to_f32(d_f32_to_bf16_rne(p));
        #pragma unroll
        for (int j = 0; j < 16; ++j) acc_o[j] += p_cast * kv[j];
    } else {
        const float alpha = __expf(m_max - dot);
        l_sum = l_sum * alpha + 1.0f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) acc_o[j] = acc_o[j] * alpha + kv[j];
        m_max = dot;
    }
}

// Zero-Staging In-Place Ring Sparse Attention (M <= 16 / DSpark):
// Conflict-free 8 KiB packed-BF16 LDS + 1-expf lazy online-softmax -> 3 CTAs/CU (12 waves/CU), ScratchSize == 0!
__global__ __launch_bounds__(256, 3)
void k_sparse_attn(
    const uint16_t* __restrict__ q_bf16,
    const float*    __restrict__ attn_sink,
    const uint16_t* __restrict__ win_kv_ring, // [128, 512] persistent ring (before committing cur_kv_bf16!)
    const uint16_t* __restrict__ cur_kv_bf16, // [M, 512]
    int start_pos,
    int swa_floor_pos,
    bool dspark_bidir,
    const uint8_t*  __restrict__ comp_fp4,
    const uint8_t*  __restrict__ comp_e4m3,
    const int32_t*  __restrict__ comp_idxs,
    int n_comp,
    uint16_t*       __restrict__ out_o_bf16,
    int M, int h_cnt)
{
    __shared__ uint16_t s_lut[256];
    __shared__ uint32_t s_kv_pak[8][256];
    __shared__ int32_t  s_valid[8];

    const int m = int(blockIdx.x);
    const int h_base = int(blockIdx.y) * 8;
    const int tid = int(threadIdx.x);
    const int h_local = tid >> 5;   // 0..7
    const int d_lane = tid & 31;    // 0..31
    const int h = h_base + h_local;

    s_lut[tid] = d_fp8_e4m3_to_fp16_bits(uint8_t(tid));
    __syncthreads();

    float q_reg[16];
    float acc_o[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        acc_o[j] = 0.0f;
        q_reg[j] = (h < h_cnt) ? d_bf16_to_f32(q_bf16[(size_t(m) * size_t(h_cnt) + size_t(h)) * 512u + d_lane * 16 + j]) : 0.0f;
    }

    float m_max = -1e30f;
    float l_sum = 0.0f;

    int win_first = 0, hist_valid = 0, n_win_m = 0;
    if (dspark_bidir) {
        win_first = max(swa_floor_pos, max(0, start_pos - 128));
        hist_valid = max(0, start_pos - win_first);
        n_win_m = hist_valid + M;
    } else {
        const int q_pos = start_pos + m;
        win_first = max(swa_floor_pos, max(0, q_pos - 127));
        n_win_m = max(0, q_pos - win_first + 1);
    }

    for (int base = 0; base < n_win_m; base += 8) {
        const int slot = h_local;
        const int kv_ord = base + slot;
        const int valid = (kv_ord < n_win_m) ? 1 : -1;
        if (d_lane == 0) s_valid[slot] = valid;
        if (valid >= 0) {
            const uint16_t* src = nullptr;
            if (dspark_bidir) {
                src = (kv_ord < hist_valid)
                    ? (win_kv_ring + size_t((win_first + kv_ord) & 127) * 512u + d_lane * 16)
                    : (cur_kv_bf16 + size_t(kv_ord - hist_valid) * 512u + d_lane * 16);
            } else {
                const int p = win_first + kv_ord;
                src = (p < start_pos)
                    ? (win_kv_ring + size_t(p & 127) * 512u + d_lane * 16)
                    : (cur_kv_bf16 + size_t(p - start_pos) * 512u + d_lane * 16);
            }
            d_load_win_kv_pak(src, d_lane, s_kv_pak[slot]);
        }
        __syncthreads();

        if (h < h_cnt) {
            #pragma unroll 1
            for (int t = 0; t < 8; ++t) {
                if (s_valid[t] >= 0) d_sparse_attn_step_kv_pak(q_reg, acc_o, m_max, l_sum, s_kv_pak[t], d_lane);
            }
        }
        __syncthreads();
    }

    if (n_comp > 0 && comp_fp4 && comp_e4m3 && comp_idxs) {
        const int32_t* c_idx_row = comp_idxs + size_t(m) * size_t(n_comp);
        for (int base = 0; base < n_comp; base += 8) {
            const int slot = h_local;
            const int32_t kv_idx = (base + slot < n_comp) ? c_idx_row[base + slot] : -1;
            if (d_lane == 0) s_valid[slot] = kv_idx;
            if (kv_idx >= 0) d_load_comp_kv_pak(comp_fp4, comp_e4m3, s_lut, kv_idx, d_lane, s_kv_pak[slot]);
            __syncthreads();

            if (h < h_cnt) {
                #pragma unroll 1
                for (int t = 0; t < 8; ++t) {
                    if (s_valid[t] >= 0) d_sparse_attn_step_kv_pak(q_reg, acc_o, m_max, l_sum, s_kv_pak[t], d_lane);
                }
            }
            __syncthreads();
        }
    }

    if (h < h_cnt) {
        const float m_tot = fmaxf(m_max, attn_sink[h]);
        const float scale = (l_sum > 0.0f) ? __expf(m_max - m_tot) : 0.0f;
        const float denom = l_sum * scale + __expf(attn_sink[h] - m_tot);
        const float inv_l = (denom > 0.0f) ? (scale / denom) : 0.0f;
        uint16_t* dst = out_o_bf16 + (size_t(m) * size_t(h_cnt) + size_t(h)) * 512u + d_lane * 16;
        #pragma unroll
        for (int j = 0; j < 16; ++j) dst[j] = d_f32_to_bf16_rne(acc_o[j] * inv_l);
    }
}

// 8-Head High-Occupancy Prefill Sparse Attention (M > 16, 256 threads = 8 heads/CTA, 3 CTAs/CU = 12 waves/CU, ScratchSize == 0!):
// Uses 4-key batched ILP online softmax (1 acc_o rescale per 4 keys instead of 4!) and zero idle warps on h_cnt = 16 / 24!
__device__ __forceinline__ void d_sparse_attn_step_4kv_pak(
    const float (&q_reg)[16],
    float (&acc_o)[16],
    float& m_max,
    float& l_sum,
    const uint32_t (*kv_pak)[256],
    int t0,
    int d_lane)
{
    constexpr float kScale = 0.04419417382415922f;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const int idx = p * 32 + d_lane;
        const float q0 = q_reg[2 * p + 0];
        const float q1 = q_reg[2 * p + 1];
        const uint32_t u0 = kv_pak[t0 + 0][idx];
        const uint32_t u1 = kv_pak[t0 + 1][idx];
        const uint32_t u2 = kv_pak[t0 + 2][idx];
        const uint32_t u3 = kv_pak[t0 + 3][idx];
        d0 += q0 * __uint_as_float(u0 << 16) + q1 * __uint_as_float(u0 & 0xFFFF0000u);
        d1 += q0 * __uint_as_float(u1 << 16) + q1 * __uint_as_float(u1 & 0xFFFF0000u);
        d2 += q0 * __uint_as_float(u2 << 16) + q1 * __uint_as_float(u2 & 0xFFFF0000u);
        d3 += q0 * __uint_as_float(u3 << 16) + q1 * __uint_as_float(u3 & 0xFFFF0000u);
    }
    d0 = d_subwave_sum32(d0) * kScale;
    d1 = d_subwave_sum32(d1) * kScale;
    d2 = d_subwave_sum32(d2) * kScale;
    d3 = d_subwave_sum32(d3) * kScale;

    const float m_quad = fmaxf(fmaxf(d0, d1), fmaxf(d2, d3));
    const float m_new  = fmaxf(m_max, m_quad);
    if (m_new > m_max) {
        const float alpha = __expf(m_max - m_new);
        l_sum *= alpha;
        #pragma unroll
        for (int j = 0; j < 16; ++j) acc_o[j] *= alpha;
        m_max = m_new;
    }
    const float p0 = __expf(d0 - m_max);
    const float p1 = __expf(d1 - m_max);
    const float p2 = __expf(d2 - m_max);
    const float p3 = __expf(d3 - m_max);
    l_sum += (p0 + p1) + (p2 + p3);

    const float w0 = d_bf16_to_f32(d_f32_to_bf16_rne(p0));
    const float w1 = d_bf16_to_f32(d_f32_to_bf16_rne(p1));
    const float w2 = d_bf16_to_f32(d_f32_to_bf16_rne(p2));
    const float w3 = d_bf16_to_f32(d_f32_to_bf16_rne(p3));
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const int idx = p * 32 + d_lane;
        const uint32_t u0 = kv_pak[t0 + 0][idx];
        const uint32_t u1 = kv_pak[t0 + 1][idx];
        const uint32_t u2 = kv_pak[t0 + 2][idx];
        const uint32_t u3 = kv_pak[t0 + 3][idx];
        acc_o[2 * p + 0] += w0 * __uint_as_float(u0 << 16) + w1 * __uint_as_float(u1 << 16) +
                            w2 * __uint_as_float(u2 << 16) + w3 * __uint_as_float(u3 << 16);
        acc_o[2 * p + 1] += w0 * __uint_as_float(u0 & 0xFFFF0000u) + w1 * __uint_as_float(u1 & 0xFFFF0000u) +
                            w2 * __uint_as_float(u2 & 0xFFFF0000u) + w3 * __uint_as_float(u3 & 0xFFFF0000u);
    }
}

__global__ __launch_bounds__(256, 2)
void k_sparse_attn_allheads(
    const uint16_t* __restrict__ q_bf16,
    const float*    __restrict__ attn_sink,
    const uint16_t* __restrict__ win_kv_ring,
    const uint16_t* __restrict__ cur_kv_bf16,
    int start_pos,
    int swa_floor_pos,
    const uint8_t*  __restrict__ comp_fp4,
    const uint8_t*  __restrict__ comp_e4m3,
    const int32_t*  __restrict__ comp_idxs,
    int n_comp,
    uint16_t*       __restrict__ out_o_bf16,
    int h_cnt)
{
    __shared__ uint16_t s_lut[256];
    __shared__ uint32_t s_kv_pak[8][256]; // 8.0 KiB conflict-free LDS -> 3 CTAs/CU (12 waves/CU)!
    __shared__ int32_t  s_valid[8];

    const int m      = int(blockIdx.x);
    const int h_base = int(blockIdx.y) * 8;
    const int tid    = int(threadIdx.x);
    const int h_loc  = tid >> 5;   // 0..7
    const int h      = h_base + h_loc;
    const int d_lane = tid & 31;

    s_lut[tid] = d_fp8_e4m3_to_fp16_bits(uint8_t(tid));
    __syncthreads();

    float q_reg[16];
    float acc_o[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        acc_o[j] = 0.0f;
        q_reg[j] = (h < h_cnt) ? d_bf16_to_f32(q_bf16[(size_t(m) * size_t(h_cnt) + size_t(h)) * 512u + d_lane * 16 + j]) : 0.0f;
    }

    float m_max = -1e30f;
    float l_sum = 0.0f;

    const int q_pos = start_pos + m;
    const int win_first = max(swa_floor_pos, max(0, q_pos - 127));
    const int n_win_m = max(0, q_pos - win_first + 1);

    for (int base = 0; base < n_win_m; base += 8) {
        const int kv_ord = base + h_loc;
        const int valid = (kv_ord < n_win_m) ? 1 : -1;
        if (d_lane == 0) s_valid[h_loc] = valid;
        if (valid >= 0) {
            const int p = win_first + kv_ord;
            const uint16_t* src = (p < start_pos)
                ? (win_kv_ring + size_t(p & 127) * 512u + d_lane * 16)
                : (cur_kv_bf16 + size_t(p - start_pos) * 512u + d_lane * 16);
            d_load_win_kv_pak(src, d_lane, s_kv_pak[h_loc]);
        }
        __syncthreads();

        if (h < h_cnt) {
            #pragma unroll 1
            for (int t = 0; t < 8; t += 4) {
                if ((s_valid[t + 0] & s_valid[t + 1] & s_valid[t + 2] & s_valid[t + 3]) >= 0) {
                    d_sparse_attn_step_4kv_pak(q_reg, acc_o, m_max, l_sum, s_kv_pak, t, d_lane);
                } else {
                    #pragma unroll 1
                    for (int sub = 0; sub < 4; ++sub) {
                        if (s_valid[t + sub] >= 0) d_sparse_attn_step_kv_pak(q_reg, acc_o, m_max, l_sum, s_kv_pak[t + sub], d_lane);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (n_comp > 0 && comp_fp4 && comp_e4m3 && comp_idxs) {
        const int32_t* c_idx_row = comp_idxs + size_t(m) * size_t(n_comp);
        for (int base = 0; base < n_comp; base += 8) {
            const int32_t kv_idx = (base + h_loc < n_comp) ? c_idx_row[base + h_loc] : -1;
            if (d_lane == 0) s_valid[h_loc] = kv_idx;
            if (kv_idx >= 0) d_load_comp_kv_pak(comp_fp4, comp_e4m3, s_lut, kv_idx, d_lane, s_kv_pak[h_loc]);
            __syncthreads();

            if (h < h_cnt) {
                #pragma unroll 1
                for (int t = 0; t < 8; t += 4) {
                    if ((s_valid[t + 0] & s_valid[t + 1] & s_valid[t + 2] & s_valid[t + 3]) >= 0) {
                        d_sparse_attn_step_4kv_pak(q_reg, acc_o, m_max, l_sum, s_kv_pak, t, d_lane);
                    } else {
                        #pragma unroll 1
                        for (int sub = 0; sub < 4; ++sub) {
                            if (s_valid[t + sub] >= 0) d_sparse_attn_step_kv_pak(q_reg, acc_o, m_max, l_sum, s_kv_pak[t + sub], d_lane);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    if (h < h_cnt) {
        const float m_tot = fmaxf(m_max, attn_sink[h]);
        const float scale = (l_sum > 0.0f) ? __expf(m_max - m_tot) : 0.0f;
        const float denom = l_sum * scale + __expf(attn_sink[h] - m_tot);
        const float inv_l = (denom > 0.0f) ? (scale / denom) : 0.0f;
        uint16_t* dst = out_o_bf16 + (size_t(m) * size_t(h_cnt) + size_t(h)) * 512u + d_lane * 16;
        #pragma unroll
        for (int j = 0; j < 16; ++j) dst[j] = d_f32_to_bf16_rne(acc_o[j] * inv_l);
    }
}

// 8-way Split-K Sparse Attention for M <= 8 Decode & DSpark Verify (3 CTAs/CU = 12 waves/CU, 8 KiB LDS, ScratchSize == 0!):
__global__ __launch_bounds__(256, 3)
void k_sparse_attn_splitk(
    const uint16_t* __restrict__ q_bf16,
    const uint16_t* __restrict__ win_kv_ring,
    const uint16_t* __restrict__ cur_kv_bf16,
    int start_pos,
    int swa_floor_pos,
    const uint8_t*  __restrict__ comp_fp4,
    const uint8_t*  __restrict__ comp_e4m3,
    const int32_t*  __restrict__ comp_idxs,
    int n_comp,
    float*          __restrict__ split_scratch, // [M, 8, h_cnt, 514]
    int h_cnt)
{
    __shared__ uint16_t s_lut[256];
    __shared__ uint32_t s_kv_pak[8][256];
    __shared__ int32_t  s_valid[8];

    const int m       = int(blockIdx.x);
    const int h_base  = int(blockIdx.y) * 8;
    const int split   = int(blockIdx.z); // 0..7
    const int tid     = int(threadIdx.x);
    const int h_local = tid >> 5;
    const int d_lane  = tid & 31;
    const int h       = h_base + h_local;

    s_lut[tid] = d_fp8_e4m3_to_fp16_bits(uint8_t(tid));
    __syncthreads();

    float q_reg[16];
    float acc_o[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        acc_o[j] = 0.0f;
        q_reg[j] = (h < h_cnt) ? d_bf16_to_f32(q_bf16[(size_t(m) * size_t(h_cnt) + size_t(h)) * 512u + d_lane * 16 + j]) : 0.0f;
    }

    const int q_pos = start_pos + m;
    const int win_first = max(swa_floor_pos, max(0, q_pos - 127));
    const int n_win_m = max(0, q_pos - win_first + 1);
    const int win_groups = (n_win_m + 7) >> 3;
    const int comp_groups = (n_comp > 0 && comp_fp4 && comp_e4m3 && comp_idxs) ? ((n_comp + 7) >> 3) : 0;
    const int total_groups = win_groups + comp_groups;

    const int g_begin = (total_groups * split) >> 3;
    const int g_end   = (total_groups * (split + 1)) >> 3;

    float m_max = -1e30f;
    float l_sum = 0.0f;
    const int32_t* c_idx_row = (comp_groups > 0) ? (comp_idxs + size_t(m) * size_t(n_comp)) : nullptr;

    for (int g_idx = g_begin; g_idx < g_end; ++g_idx) {
        const int slot = h_local;
        if (g_idx < win_groups) {
            const int kv_ord = g_idx * 8 + slot;
            const int valid = (kv_ord < n_win_m) ? 1 : -1;
            if (d_lane == 0) s_valid[slot] = valid;
            if (valid >= 0) {
                const int p = win_first + kv_ord;
                const uint16_t* src = (p < start_pos)
                    ? (win_kv_ring + size_t(p & 127) * 512u + d_lane * 16)
                    : (cur_kv_bf16 + size_t(p - start_pos) * 512u + d_lane * 16);
                d_load_win_kv_pak(src, d_lane, s_kv_pak[slot]);
            }
        } else {
            const int base = (g_idx - win_groups) * 8;
            const int32_t kv_idx = (base + slot < n_comp) ? c_idx_row[base + slot] : -1;
            if (d_lane == 0) s_valid[slot] = kv_idx;
            if (kv_idx >= 0) d_load_comp_kv_pak(comp_fp4, comp_e4m3, s_lut, kv_idx, d_lane, s_kv_pak[slot]);
        }
        __syncthreads();

        if (h < h_cnt) {
            #pragma unroll 1
            for (int t = 0; t < 8; ++t) {
                if (s_valid[t] >= 0) d_sparse_attn_step_kv_pak(q_reg, acc_o, m_max, l_sum, s_kv_pak[t], d_lane);
            }
        }
        __syncthreads();
    }

    if (h < h_cnt) {
        float* dst = split_scratch + ((size_t(m) * 8u + size_t(split)) * size_t(h_cnt) + size_t(h)) * 514u;
        #pragma unroll
        for (int j = 0; j < 16; ++j) dst[d_lane * 16 + j] = acc_o[j];
        if (d_lane == 0) {
            dst[512] = m_max;
            dst[513] = l_sum;
        }
    }
}

// Combine 8 Split-K partials + attention sink -> out_o_bf16 [M, h_cnt, 512]
__global__ __launch_bounds__(256)
void k_sparse_attn_reduce_splits(
    const float* __restrict__ split_scratch, // [M, 8, h_cnt, 514]
    const float* __restrict__ attn_sink,     // [h_cnt]
    uint16_t*    __restrict__ out_o_bf16,    // [M, h_cnt, 512]
    int h_cnt)
{
    const int m   = int(blockIdx.x);
    const int h   = int(blockIdx.y);
    const int tid = int(threadIdx.x); // 0..255 -> handles d = tid and d = tid + 256

    const float sink = attn_sink[h];
    float m_tot = sink;
    float m_s[8];
    float l_s[8];
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        const float* sp = split_scratch + ((size_t(m) * 8u + size_t(s)) * size_t(h_cnt) + size_t(h)) * 514u;
        m_s[s] = sp[512];
        l_s[s] = sp[513];
        if (l_s[s] > 0.0f) m_tot = fmaxf(m_tot, m_s[s]);
    }

    float l_tot = __expf(sink - m_tot);
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (l_s[s] > 0.0f) {
            const float sc = __expf(m_s[s] - m_tot);
            const float* sp = split_scratch + ((size_t(m) * 8u + size_t(s)) * size_t(h_cnt) + size_t(h)) * 514u;
            l_tot += l_s[s] * sc;
            acc0 += sp[tid] * sc;
            acc1 += sp[256 + tid] * sc;
        }
    }

    const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    uint16_t* dst = out_o_bf16 + (size_t(m) * size_t(h_cnt) + size_t(h)) * 512u;
    dst[tid]       = d_f32_to_bf16_rne(acc0 * inv_l);
    dst[256 + tid] = d_f32_to_bf16_rne(acc1 * inv_l);
}

// =============================================================================
// 6. MoE Gate + Multi-Row Routed HLUT2 (3x MI50 TP3)
// =============================================================================
__global__ __launch_bounds__(256)
void k_moe_gate(
    const float* __restrict__ logits_f32,
    const float* __restrict__ bias_f32,
    const float* __restrict__ bias_vl_f32,
    const uint8_t* __restrict__ image_mask,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_weights,
    int n_routed, int top_k, float route_scale)
{
    __shared__ float s_raw[384];
    __shared__ float s_biased[384];

    const int m = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const bool is_img = (image_mask && bias_vl_f32 && image_mask[m] != 0u);
    const float* bptr = is_img ? bias_vl_f32 : bias_f32;
    const float* lrow = logits_f32 + size_t(m) * size_t(n_routed);

    for (int e = tid; e < n_routed; e += 256) {
        const float x = lrow[e];
        const float sp = (x > 20.0f) ? x : log1pf(__expf(x));
        const float sc = sqrtf(fmaxf(sp, 0.0f));
        s_raw[e] = sc;
        s_biased[e] = sc + bptr[e];
    }
    __syncthreads();

    if (tid == 0) {
        float w_sum = 0.0f;
        int32_t picked_ids[6];
        float picked_w[6];
        for (int k = 0; k < top_k; ++k) {
            int best_e = 0;
            float best_v = -1e30f;
            for (int e = 0; e < n_routed; ++e) {
                if (s_biased[e] > best_v) {
                    best_v = s_biased[e];
                    best_e = e;
                }
            }
            picked_ids[k] = best_e;
            picked_w[k] = s_raw[best_e];
            w_sum += picked_w[k];
            s_biased[best_e] = -1e30f;
        }
        const float scale = (top_k > 1) ? (route_scale / (w_sum + 1e-20f)) : route_scale;
        for (int k = 0; k < top_k; ++k) {
            out_ids[m * top_k + k] = picked_ids[k];
            out_weights[m * top_k + k] = picked_w[k] * scale;
        }
    }
}

__global__ __launch_bounds__(128, 8)
void k_wht128_bf16_to_f16(
    const uint16_t* __restrict__ in_bf16,
    uint16_t* __restrict__ out_f16,
    const uint64_t* __restrict__ signs64,
    int K)
{
    __shared__ float s_xwave[128];
    const int blk = int(blockIdx.x);
    const int row = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int lane = tid & 63;
    const int wid = tid >> 6;

    const size_t base = size_t(row) * size_t(K) + size_t(blk) * 128u;
    float v = d_bf16_to_f32(in_bf16[base + tid]);
    if ((signs64[blk * 2 + wid] >> lane) & 1ULL) v = -v;

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
    out_f16[base + tid] = d_f32_to_fp16_bits(res);
}

__device__ __forceinline__ void d_make_hlut2_lut_x2(const float* cent, uint32_t& out_h0_x2, uint32_t& out_dh_x2) {
    const uint32_t h0 = d_f32_to_fp16_bits(cent[0]);
    const uint32_t h1 = d_f32_to_fp16_bits(cent[1]);
    out_h0_x2 = h0 | (h0 << 16);
    out_dh_x2 = out_h0_x2 ^ (h1 | (h1 << 16));
}

__device__ __forceinline__ f16x2_t d_unpack_hlut2_pair(uint32_t nib4, uint32_t h0_x2, uint32_t dh_x2) {
    const uint32_t m01 = (nib4 & 1u) | ((nib4 & 4u) << 14);
    const uint32_t mag_bits = h0_x2 ^ ((m01 * 0xFFFFu) & dh_x2);
    const uint32_t sgn_bits = ((nib4 << 14) & 0x00008000u) | ((nib4 << 28) & 0x80000000u);
    const uint32_t out_bits = mag_bits | sgn_bits;
    f16x2_t res;
    __builtin_memcpy(&res, &out_bits, 4);
    return res;
}

// Fused 6-CTA Pair-wise HLUT2 Gate+Up+SwiGLU10 + Register-Butterfly WHT128 (768 -> FP16):
// Grid: (6, P), Block: 256 = 32 N-lanes x 8 K-stripes, 12.25 KiB LDS (3 CTAs/CU, 4x fewer xr reads, 0 inter_bf16 writes!)
__global__ __launch_bounds__(256, 3)
void k_hlut2_pair_swiglu(
    const uint32_t* __restrict__ x_rot_f16x2,  // [M, 2560]
    const Mi50RoutedPair* __restrict__ pairs,  // [P]
    const uint32_t* __restrict__ gate_codes,   // [E, 24, 320, 32] -> 245760 words/expert
    const uint16_t* __restrict__ gate_scales,  // [E, 24, 40, 32]  -> 30720 halfs/expert
    const float*    __restrict__ gate_cent,    // [E, 2]
    const uint32_t* __restrict__ up_codes,
    const uint16_t* __restrict__ up_scales,
    const float*    __restrict__ up_cent,
    const uint64_t* __restrict__ signs_i,      // [12] uint64 (768 dims)
    uint16_t*       __restrict__ inter_rot_f16)// [P, 768] FP16
{
    __shared__ uint32_t s_xr[2560];
    __shared__ float s_red[8][2][32];
    __shared__ uint16_t s_inter[128];

    const int blk128 = int(blockIdx.x); // 0..5
    const int p_idx = int(blockIdx.y);  // 0..P-1
    const int tid = int(threadIdx.x);
    const int r = tid & 31;
    const int ks = tid >> 5;            // 0..7

    const Mi50RoutedPair pr = pairs[p_idx];
    const size_t e = size_t(pr.expert);
    const uint32_t* xr = x_rot_f16x2 + size_t(pr.row) * 2560u;

    for (int idx = tid; idx < 2560; idx += 256) {
        s_xr[idx] = xr[idx];
    }
    __syncthreads();

    uint32_t g_h0, g_dh, u_h0, u_dh;
    d_make_hlut2_lut_x2(gate_cent + e * 2u, g_h0, g_dh);
    d_make_hlut2_lut_x2(up_cent + e * 2u, u_h0, u_dh);

    for (int sub = 0; sub < 4; ++sub) {
        const int rb = blk128 * 4 + sub;
        const uint32_t* gc_base = gate_codes + e * 245760u + size_t(rb) * 320u * 32u + r;
        const uint16_t* gs_base = gate_scales + e * 30720u + size_t(rb) * 40u * 32u + r;
        const uint32_t* uc_base = up_codes + e * 245760u + size_t(rb) * 320u * 32u + r;
        const uint16_t* us_base = up_scales + e * 30720u + size_t(rb) * 40u * 32u + r;

        float acc_g = 0.0f, acc_u = 0.0f;
        for (int g = ks; g < 40; g += 8) {
            const float sig_g = d_fp16_bits_to_f32(gs_base[g * 32]);
            const float sig_u = d_fp16_bits_to_f32(us_base[g * 32]);
            const uint32_t* gc = gc_base + size_t(g * 8) * 32u;
            const uint32_t* uc = uc_base + size_t(g * 8) * 32u;

            float grp_g = 0.0f, grp_u = 0.0f;
            #pragma unroll 4
            for (int w = 0; w < 8; ++w) {
                const uint32_t pg = gc[w * 32];
                const uint32_t pu = uc[w * 32];
                const uint32_t* xb = s_xr + g * 64 + w * 8;
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    f16x2_t xv;
                    const uint32_t xbits = xb[k];
                    __builtin_memcpy(&xv, &xbits, 4);
                    grp_g = __builtin_amdgcn_fdot2(d_unpack_hlut2_pair(pg >> (4 * k), g_h0, g_dh), xv, grp_g, false);
                    grp_u = __builtin_amdgcn_fdot2(d_unpack_hlut2_pair(pu >> (4 * k), u_h0, u_dh), xv, grp_u, false);
                }
            }
            acc_g += grp_g * sig_g;
            acc_u += grp_u * sig_u;
        }

        s_red[ks][0][r] = acc_g;
        s_red[ks][1][r] = acc_u;
        __syncthreads();

        if (ks == 0) {
            const float sg = s_red[0][0][r] + s_red[1][0][r] + s_red[2][0][r] + s_red[3][0][r] +
                             s_red[4][0][r] + s_red[5][0][r] + s_red[6][0][r] + s_red[7][0][r];
            const float su = s_red[0][1][r] + s_red[1][1][r] + s_red[2][1][r] + s_red[3][1][r] +
                             s_red[4][1][r] + s_red[5][1][r] + s_red[6][1][r] + s_red[7][1][r];
            s_inter[sub * 32 + r] = d_f32_to_bf16_rne(d_swiglu10(sg, su));
        }
        __syncthreads();
    }

    if (tid < 128) {
        float* s_wht = &s_red[0][0][0];
        const int lane = tid & 63;
        const int wid = tid >> 6;
        float v = d_bf16_to_f32(s_inter[tid]);
        if ((signs_i[blk128 * 2 + wid] >> lane) & 1ULL) v = -v;
        #pragma unroll
        for (int len = 1; len <= 32; len <<= 1) {
            const float partner = __shfl_xor(v, len, 64);
            v = (lane & len) ? (partner - v) : (v + partner);
        }
        s_wht[tid] = v;
    }
    __syncthreads();
    if (tid < 128) {
        const float* s_wht = &s_red[0][0][0];
        const int lane = tid & 63;
        const int wid = tid >> 6;
        const float a = s_wht[lane];
        const float b = s_wht[64 + lane];
        const float res = ((wid == 0) ? (a + b) : (a - b)) * 0.08838834764831844655f;
        inter_rot_f16[size_t(p_idx) * 768u + blk128 * 128 + tid] = d_f32_to_fp16_bits(res);
    }
}

// Pair-wise HLUT2 Down + atomicAdd into out_accum_f32 [M, 5120]:
// Grid: (160, P), Block: 192 = 32 N-lanes x 6 K-stripes (since K=768 = 6 * 128!).
__global__ __launch_bounds__(192, 4)
void k_hlut2_pair_down_atomic(
    const uint32_t* __restrict__ inter_rot_f16x2, // [P, 384]
    const Mi50RoutedPair* __restrict__ pairs,     // [P]
    const uint32_t* __restrict__ down_codes,      // [E, 160, 48, 32] -> 245760 words/expert
    const uint16_t* __restrict__ down_scales,     // [E, 160, 6, 32]  -> 30720 halfs/expert
    const float*    __restrict__ down_cent,       // [E, 2]
    float*          __restrict__ out_accum_f32)   // [M, 5120]
{
    __shared__ float s_red[6][32];

    const int rb = int(blockIdx.x);    // 0..159
    const int p_idx = int(blockIdx.y); // 0..P-1
    const int tid = int(threadIdx.x);  // 0..191
    const int r = tid & 31;
    const int g = tid >> 5;            // 0..5 (exact 128-group!)

    const Mi50RoutedPair pr = pairs[p_idx];
    const size_t e = size_t(pr.expert);
    const uint32_t* xr = inter_rot_f16x2 + size_t(p_idx) * 384u + g * 64;

    uint32_t d_h0, d_dh;
    d_make_hlut2_lut_x2(down_cent + e * 2u, d_h0, d_dh);

    const uint32_t* dc = down_codes + e * 245760u + (size_t(rb) * 48u + size_t(g * 8)) * 32u + r;
    const float sig_d = d_fp16_bits_to_f32(down_scales[e * 30720u + (size_t(rb) * 6u + size_t(g)) * 32u + r]);

    float grp = 0.0f;
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        const uint32_t pd = dc[w * 32];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            f16x2_t xv;
            const uint32_t xbits = xr[w * 8 + k];
            __builtin_memcpy(&xv, &xbits, 4);
            grp = __builtin_amdgcn_fdot2(d_unpack_hlut2_pair(pd >> (4 * k), d_h0, d_dh), xv, grp, false);
        }
    }
    s_red[g][r] = grp * sig_d;
    __syncthreads();

    if (g == 0) {
        const float sum = (s_red[0][r] + s_red[1][r] + s_red[2][r] +
                           s_red[3][r] + s_red[4][r] + s_red[5][r]) * pr.weight;
        atomicAdd(&out_accum_f32[size_t(pr.row) * 5120u + rb * 32 + r], sum);
    }
}

__device__ __forceinline__ int d_hlut2_swiz4(int idx4) {
    return idx4 ^ (((idx4 >> 4) & 1) << 2);
}

__device__ __forceinline__ void d_fdot4_u4_pair(
    const uint4& vx,
    const f16x2_t (&wg)[4],
    const f16x2_t (&wu)[4],
    float& gg,
    float& gu)
{
    const uint32_t words[4] = {vx.x, vx.y, vx.z, vx.w};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        f16x2_t xv;
        __builtin_memcpy(&xv, &words[i], 4);
        gg = __builtin_amdgcn_fdot2(wg[i], xv, gg, false);
        gu = __builtin_amdgcn_fdot2(wu[i], xv, gu, false);
    }
}

// Fused 6-CTA Grouped HLUT2 Gate+Up+SwiGLU10 + Register-Butterfly WHT128 (BM=8, 16 KiB LDS, ScratchSize == 0!)
__global__ __launch_bounds__(256, 2)
void k_hlut2_grouped_swiglu8(
    const uint32_t* __restrict__ x_rot_f16x2,
    const int32_t*  __restrict__ active_experts,
    const int32_t*  __restrict__ csr_offsets,
    const Mi50RoutedPair* __restrict__ sorted_pairs,
    const uint32_t* __restrict__ gate_codes,
    const uint16_t* __restrict__ gate_scales,
    const float*    __restrict__ gate_cent,
    const uint32_t* __restrict__ up_codes,
    const uint16_t* __restrict__ up_scales,
    const float*    __restrict__ up_cent,
    const uint64_t* __restrict__ signs_i,
    uint16_t*       __restrict__ inter_bf16,
    uint16_t*       __restrict__ inter_rot_f16)
{
    union SharedSwiglu8 {
        uint4 x4[8][128];
        float red[8][2][8][32];
        float wht[2][128];
    };
    __shared__ SharedSwiglu8 sh;

    const int blk128 = int(blockIdx.x);   // 0..5
    const int ae_idx = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int r = tid & 31;
    const int ks = tid >> 5;

    const int p_begin = csr_offsets[ae_idx];
    const int p_end = csr_offsets[ae_idx + 1];
    if (p_begin >= p_end) return;

    const size_t e = size_t(active_experts[ae_idx]);
    uint32_t g_h0, g_dh, u_h0, u_dh;
    d_make_hlut2_lut_x2(gate_cent + e * 2u, g_h0, g_dh);
    d_make_hlut2_lut_x2(up_cent + e * 2u, u_h0, u_dh);

    const int ld_t_off = tid >> 7;        // 0..1
    const int ld_col4  = tid & 127;       // 0..127
    const int ld_sw4   = d_hlut2_swiz4(ld_col4);

    for (int p0 = p_begin; p0 < p_end; p0 += 8) {
        const int cnt = min(8, p_end - p0);
        const int bm  = (cnt <= 4) ? 4 : 8;

        #pragma unroll 1
        for (int sub = 0; sub < 4; ++sub) {
            const int rb = blk128 * 4 + sub;
            const uint32_t* gc_base = gate_codes + e * 245760u + size_t(rb) * 320u * 32u + r;
            const uint16_t* gs_base = gate_scales + e * 30720u + size_t(rb) * 40u * 32u + r;
            const uint32_t* uc_base = up_codes + e * 245760u + size_t(rb) * 320u * 32u + r;
            const uint16_t* us_base = up_scales + e * 30720u + size_t(rb) * 40u * 32u + r;

            float acc_g[8] = {0.0f};
            float acc_u[8] = {0.0f};

            for (int g_base = 0; g_base < 40; g_base += 8) {
                const int g = g_base + ks;
                const uint16_t raw_sig_g = gs_base[g * 32];
                const uint16_t raw_sig_u = us_base[g * 32];
                const uint32_t* gc = gc_base + size_t(g * 8) * 32u;
                const uint32_t* uc = uc_base + size_t(g * 8) * 32u;
                uint32_t pg_reg[4], pu_reg[4];
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    pg_reg[w] = gc[w * 32];
                    pu_reg[w] = uc[w * 32];
                }

                for (int tb = 0; tb < bm; tb += 2) {
                    const int t = tb + ld_t_off;
                    uint4 vx = {0u, 0u, 0u, 0u};
                    if (t < cnt) {
                        const int row_t = sorted_pairs[p0 + t].row;
                        vx = *reinterpret_cast<const uint4*>(x_rot_f16x2 + size_t(row_t) * 2560u + size_t(g_base * 64 + ld_col4 * 4));
                    }
                    sh.x4[t][ld_sw4] = vx;
                }
                __syncthreads();

                const float sig_g = d_fp16_bits_to_f32(raw_sig_g);
                const float sig_u = d_fp16_bits_to_f32(raw_sig_u);

                float gg[8] = {0.0f};
                float gu[8] = {0.0f};
                if (cnt <= 4) {
                    #pragma unroll 1
                    for (int wb = 0; wb < 8; wb += 4) {
                        if (wb > 0) {
                            #pragma unroll
                            for (int w = 0; w < 4; ++w) { pg_reg[w] = gc[(wb + w) * 32]; pu_reg[w] = uc[(wb + w) * 32]; }
                        }
                        #pragma unroll
                        for (int w = 0; w < 4; ++w) {
                            const uint32_t pg = pg_reg[w];
                            const uint32_t pu = pu_reg[w];
                            #pragma unroll
                            for (int kh = 0; kh < 2; ++kh) {
                                const uint32_t pgh = pg >> (16 * kh);
                                const uint32_t puh = pu >> (16 * kh);
                                const f16x2_t wg[4] = {
                                    d_unpack_hlut2_pair(pgh >> 0,  g_h0, g_dh),
                                    d_unpack_hlut2_pair(pgh >> 4,  g_h0, g_dh),
                                    d_unpack_hlut2_pair(pgh >> 8,  g_h0, g_dh),
                                    d_unpack_hlut2_pair(pgh >> 12, g_h0, g_dh)};
                                const f16x2_t wu[4] = {
                                    d_unpack_hlut2_pair(puh >> 0,  u_h0, u_dh),
                                    d_unpack_hlut2_pair(puh >> 4,  u_h0, u_dh),
                                    d_unpack_hlut2_pair(puh >> 8,  u_h0, u_dh),
                                    d_unpack_hlut2_pair(puh >> 12, u_h0, u_dh)};
                                const int sw4 = d_hlut2_swiz4(ks * 16 + (wb + w) * 2 + kh);
                                #pragma unroll
                                for (int t = 0; t < 4; ++t) {
                                    d_fdot4_u4_pair(sh.x4[t][sw4], wg, wu, gg[t], gu[t]);
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        acc_g[t] += gg[t] * sig_g;
                        acc_u[t] += gu[t] * sig_u;
                    }
                } else {
                    #pragma unroll 1
                    for (int wb = 0; wb < 8; wb += 4) {
                        if (wb > 0) {
                            #pragma unroll
                            for (int w = 0; w < 4; ++w) { pg_reg[w] = gc[(wb + w) * 32]; pu_reg[w] = uc[(wb + w) * 32]; }
                        }
                        #pragma unroll
                        for (int w = 0; w < 4; ++w) {
                            const uint32_t pg = pg_reg[w];
                            const uint32_t pu = pu_reg[w];
                            #pragma unroll 1
                            for (int kh = 0; kh < 2; ++kh) {
                                const uint32_t pgh = pg >> (16 * kh);
                                const uint32_t puh = pu >> (16 * kh);
                                const f16x2_t wg[4] = {
                                    d_unpack_hlut2_pair(pgh >> 0,  g_h0, g_dh),
                                    d_unpack_hlut2_pair(pgh >> 4,  g_h0, g_dh),
                                    d_unpack_hlut2_pair(pgh >> 8,  g_h0, g_dh),
                                    d_unpack_hlut2_pair(pgh >> 12, g_h0, g_dh)};
                                const f16x2_t wu[4] = {
                                    d_unpack_hlut2_pair(puh >> 0,  u_h0, u_dh),
                                    d_unpack_hlut2_pair(puh >> 4,  u_h0, u_dh),
                                    d_unpack_hlut2_pair(puh >> 8,  u_h0, u_dh),
                                    d_unpack_hlut2_pair(puh >> 12, u_h0, u_dh)};
                                const int sw4 = d_hlut2_swiz4(ks * 16 + (wb + w) * 2 + kh);
                                #pragma unroll
                                for (int t = 0; t < 8; ++t) {
                                    d_fdot4_u4_pair(sh.x4[t][sw4], wg, wu, gg[t], gu[t]);
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        acc_g[t] += gg[t] * sig_g;
                        acc_u[t] += gu[t] * sig_u;
                    }
                }
                __syncthreads();
            }

            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                sh.red[ks][0][t][r] = acc_g[t];
                sh.red[ks][1][t][r] = acc_u[t];
            }
            __syncthreads();

            if (ks < cnt) {
                const int t = ks;
                const float sg = sh.red[0][0][t][r] + sh.red[1][0][t][r] + sh.red[2][0][t][r] + sh.red[3][0][t][r] +
                                 sh.red[4][0][t][r] + sh.red[5][0][t][r] + sh.red[6][0][t][r] + sh.red[7][0][t][r];
                const float su = sh.red[0][1][t][r] + sh.red[1][1][t][r] + sh.red[2][1][t][r] + sh.red[3][1][t][r] +
                                 sh.red[4][1][t][r] + sh.red[5][1][t][r] + sh.red[6][1][t][r] + sh.red[7][1][t][r];
                inter_bf16[size_t(p0 + t) * 768u + rb * 32 + r] = d_f32_to_bf16_rne(d_swiglu10(sg, su));
            }
            __syncthreads();
        }

        const int unit = tid >> 7;
        const int lane128 = tid & 127;
        const int lane64 = tid & 63;
        const int wid_u = (tid >> 6) & 1;
        const uint64_t sign_w = signs_i[blk128 * 2 + wid_u];
        for (int t_base = 0; t_base < cnt; t_base += 2) {
            const int t = t_base + unit;
            float v = 0.0f;
            if (t < cnt) {
                v = d_bf16_to_f32(inter_bf16[size_t(p0 + t) * 768u + blk128 * 128 + lane128]);
                if ((sign_w >> lane64) & 1ULL) v = -v;
            }
            #pragma unroll
            for (int len = 1; len <= 32; len <<= 1) {
                const float partner = __shfl_xor(v, len, 64);
                v = (lane64 & len) ? (partner - v) : (v + partner);
            }
            sh.wht[unit][lane128] = v;
            __syncthreads();
            if (t < cnt) {
                const float a = sh.wht[unit][lane64];
                const float b = sh.wht[unit][64 + lane64];
                const float res = ((wid_u == 0) ? (a + b) : (a - b)) * 0.08838834764831844655f;
                inter_rot_f16[size_t(p0 + t) * 768u + blk128 * 128 + lane128] = d_f32_to_fp16_bits(res);
            }
            __syncthreads();
        }
    }
}

// Fused 6-CTA Prefill Grouped HLUT2 Gate+Up+SwiGLU10 + Register-Butterfly WHT128 (Adaptive BM=4/8/16, 32 KiB LDS!)
__global__ __launch_bounds__(256, 2)
void k_hlut2_grouped_swiglu16(
    const uint32_t* __restrict__ x_rot_f16x2,
    const int32_t*  __restrict__ active_experts,
    const int32_t*  __restrict__ csr_offsets,
    const Mi50RoutedPair* __restrict__ sorted_pairs,
    const uint32_t* __restrict__ gate_codes,
    const uint16_t* __restrict__ gate_scales,
    const float*    __restrict__ gate_cent,
    const uint32_t* __restrict__ up_codes,
    const uint16_t* __restrict__ up_scales,
    const float*    __restrict__ up_cent,
    const uint64_t* __restrict__ signs_i,
    uint16_t*       __restrict__ inter_bf16,
    uint16_t*       __restrict__ inter_rot_f16)
{
    union SharedSwiglu16 {
        uint4 x4[16][128];
        float red[8][2][16][32];
        float wht[2][128];
    };
    __shared__ SharedSwiglu16 sh;

    const int blk128 = int(blockIdx.x);   // 0..5
    const int ae_idx = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int r = tid & 31;
    const int ks = tid >> 5;

    const int p_begin = csr_offsets[ae_idx];
    const int p_end = csr_offsets[ae_idx + 1];
    if (p_begin >= p_end) return;

    const size_t e = size_t(active_experts[ae_idx]);
    uint32_t g_h0, g_dh, u_h0, u_dh;
    d_make_hlut2_lut_x2(gate_cent + e * 2u, g_h0, g_dh);
    d_make_hlut2_lut_x2(up_cent + e * 2u, u_h0, u_dh);

    const int ld_t_off = tid >> 7;
    const int ld_col4  = tid & 127;
    const int ld_sw4   = d_hlut2_swiz4(ld_col4);

    for (int p0 = p_begin; p0 < p_end; p0 += 16) {
        const int cnt = min(16, p_end - p0);
        const int bm  = (cnt <= 4) ? 4 : ((cnt <= 8) ? 8 : 16);

        #pragma unroll 1
        for (int sub = 0; sub < 4; ++sub) {
            const int rb = blk128 * 4 + sub;
            const size_t code_off = e * 245760u + size_t(rb) * 10240u + size_t(r);
            const size_t sc_off   = e * 30720u + size_t(rb) * 1280u + size_t(r);

            float acc_g[16] = {0.0f};
            float acc_u[16] = {0.0f};

            #pragma unroll 1
            for (int g_base = 0; g_base < 40; g_base += 8) {
                const int g = g_base + ks;
                const uint16_t raw_sig_g = gate_scales[sc_off + size_t(g) * 32u];
                const uint16_t raw_sig_u = up_scales[sc_off + size_t(g) * 32u];
                const size_t g_code_off  = code_off + size_t(g * 8) * 32u;

                for (int tb = 0; tb < bm; tb += 2) {
                    const int t = tb + ld_t_off;
                    uint4 vx = {0u, 0u, 0u, 0u};
                    if (t < cnt) {
                        const int row_t = sorted_pairs[p0 + t].row;
                        vx = *reinterpret_cast<const uint4*>(x_rot_f16x2 + size_t(row_t) * 2560u + size_t(g_base * 64 + ld_col4 * 4));
                    }
                    sh.x4[t][ld_sw4] = vx;
                }
                __syncthreads();

                const float sig_g = d_fp16_bits_to_f32(raw_sig_g);
                const float sig_u = d_fp16_bits_to_f32(raw_sig_u);

                if (cnt <= 4) {
                    float gg[4] = {0.0f}, gu[4] = {0.0f};
                    #pragma unroll 1
                    for (int w = 0; w < 8; ++w) {
                        const uint32_t pg = gate_codes[g_code_off + size_t(w) * 32u];
                        const uint32_t pu = up_codes[g_code_off + size_t(w) * 32u];
                        #pragma unroll
                        for (int kh = 0; kh < 2; ++kh) {
                            const uint32_t pgh = pg >> (16 * kh);
                            const uint32_t puh = pu >> (16 * kh);
                            const f16x2_t wg[4] = {
                                d_unpack_hlut2_pair(pgh >> 0,  g_h0, g_dh),
                                d_unpack_hlut2_pair(pgh >> 4,  g_h0, g_dh),
                                d_unpack_hlut2_pair(pgh >> 8,  g_h0, g_dh),
                                d_unpack_hlut2_pair(pgh >> 12, g_h0, g_dh)};
                            const f16x2_t wu[4] = {
                                d_unpack_hlut2_pair(puh >> 0,  u_h0, u_dh),
                                d_unpack_hlut2_pair(puh >> 4,  u_h0, u_dh),
                                d_unpack_hlut2_pair(puh >> 8,  u_h0, u_dh),
                                d_unpack_hlut2_pair(puh >> 12, u_h0, u_dh)};
                            const int sw4 = d_hlut2_swiz4(ks * 16 + w * 2 + kh);
                            #pragma unroll
                            for (int t = 0; t < 4; ++t) d_fdot4_u4_pair(sh.x4[t][sw4], wg, wu, gg[t], gu[t]);
                        }
                    }
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) { acc_g[t] += gg[t] * sig_g; acc_u[t] += gu[t] * sig_u; }
                } else if (cnt <= 8) {
                    float gg[8] = {0.0f}, gu[8] = {0.0f};
                    #pragma unroll 1
                    for (int w = 0; w < 8; ++w) {
                        const uint32_t pg = gate_codes[g_code_off + size_t(w) * 32u];
                        const uint32_t pu = up_codes[g_code_off + size_t(w) * 32u];
                        #pragma unroll 1
                        for (int kh = 0; kh < 2; ++kh) {
                            const uint32_t pgh = pg >> (16 * kh);
                            const uint32_t puh = pu >> (16 * kh);
                            const f16x2_t wg[4] = {
                                d_unpack_hlut2_pair(pgh >> 0,  g_h0, g_dh),
                                d_unpack_hlut2_pair(pgh >> 4,  g_h0, g_dh),
                                d_unpack_hlut2_pair(pgh >> 8,  g_h0, g_dh),
                                d_unpack_hlut2_pair(pgh >> 12, g_h0, g_dh)};
                            const f16x2_t wu[4] = {
                                d_unpack_hlut2_pair(puh >> 0,  u_h0, u_dh),
                                d_unpack_hlut2_pair(puh >> 4,  u_h0, u_dh),
                                d_unpack_hlut2_pair(puh >> 8,  u_h0, u_dh),
                                d_unpack_hlut2_pair(puh >> 12, u_h0, u_dh)};
                            const int sw4 = d_hlut2_swiz4(ks * 16 + w * 2 + kh);
                            #pragma unroll
                            for (int t = 0; t < 8; ++t) d_fdot4_u4_pair(sh.x4[t][sw4], wg, wu, gg[t], gu[t]);
                        }
                    }
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) { acc_g[t] += gg[t] * sig_g; acc_u[t] += gu[t] * sig_u; }
                } else {
                    float gg[16] = {0.0f}, gu[16] = {0.0f};
                    #pragma unroll 1
                    for (int w = 0; w < 8; ++w) {
                        const uint32_t pg = gate_codes[g_code_off + size_t(w) * 32u];
                        const uint32_t pu = up_codes[g_code_off + size_t(w) * 32u];
                        #pragma unroll 1
                        for (int kh = 0; kh < 2; ++kh) {
                            const uint32_t pgh = pg >> (16 * kh);
                            const uint32_t puh = pu >> (16 * kh);
                            const f16x2_t wg[4] = {
                                d_unpack_hlut2_pair(pgh >> 0,  g_h0, g_dh),
                                d_unpack_hlut2_pair(pgh >> 4,  g_h0, g_dh),
                                d_unpack_hlut2_pair(pgh >> 8,  g_h0, g_dh),
                                d_unpack_hlut2_pair(pgh >> 12, g_h0, g_dh)};
                            const f16x2_t wu[4] = {
                                d_unpack_hlut2_pair(puh >> 0,  u_h0, u_dh),
                                d_unpack_hlut2_pair(puh >> 4,  u_h0, u_dh),
                                d_unpack_hlut2_pair(puh >> 8,  u_h0, u_dh),
                                d_unpack_hlut2_pair(puh >> 12, u_h0, u_dh)};
                            const int sw4 = d_hlut2_swiz4(ks * 16 + w * 2 + kh);
                            #pragma unroll
                            for (int t = 0; t < 16; ++t) d_fdot4_u4_pair(sh.x4[t][sw4], wg, wu, gg[t], gu[t]);
                        }
                    }
                    #pragma unroll
                    for (int t = 0; t < 16; ++t) { acc_g[t] += gg[t] * sig_g; acc_u[t] += gu[t] * sig_u; }
                }
                __syncthreads();
            }

            #pragma unroll
            for (int t = 0; t < 16; ++t) {
                sh.red[ks][0][t][r] = acc_g[t];
                sh.red[ks][1][t][r] = acc_u[t];
            }
            __syncthreads();

            for (int t = ks; t < cnt; t += 8) {
                const float sg = sh.red[0][0][t][r] + sh.red[1][0][t][r] + sh.red[2][0][t][r] + sh.red[3][0][t][r] +
                                 sh.red[4][0][t][r] + sh.red[5][0][t][r] + sh.red[6][0][t][r] + sh.red[7][0][t][r];
                const float su = sh.red[0][1][t][r] + sh.red[1][1][t][r] + sh.red[2][1][t][r] + sh.red[3][1][t][r] +
                                 sh.red[4][1][t][r] + sh.red[5][1][t][r] + sh.red[6][1][t][r] + sh.red[7][1][t][r];
                inter_bf16[size_t(p0 + t) * 768u + rb * 32 + r] = d_f32_to_bf16_rne(d_swiglu10(sg, su));
            }
            __syncthreads();
        }

        const int unit = tid >> 7;
        const int lane128 = tid & 127;
        const int lane64 = tid & 63;
        const int wid_u = (tid >> 6) & 1;
        const uint64_t sign_w = signs_i[blk128 * 2 + wid_u];
        for (int t_base = 0; t_base < cnt; t_base += 2) {
            const int t = t_base + unit;
            float v = 0.0f;
            if (t < cnt) {
                v = d_bf16_to_f32(inter_bf16[size_t(p0 + t) * 768u + blk128 * 128 + lane128]);
                if ((sign_w >> lane64) & 1ULL) v = -v;
            }
            #pragma unroll
            for (int len = 1; len <= 32; len <<= 1) {
                const float partner = __shfl_xor(v, len, 64);
                v = (lane64 & len) ? (partner - v) : (v + partner);
            }
            sh.wht[unit][lane128] = v;
            __syncthreads();
            if (t < cnt) {
                const float a = sh.wht[unit][lane64];
                const float b = sh.wht[unit][64 + lane64];
                const float res = ((wid_u == 0) ? (a + b) : (a - b)) * 0.08838834764831844655f;
                inter_rot_f16[size_t(p0 + t) * 768u + blk128 * 128 + lane128] = d_f32_to_fp16_bits(res);
            }
            __syncthreads();
        }
    }
}

// Grouped HLUT2 Down-Proj with Register-Resident `pd_reg[8]` & Conflict-Free Swizzled uint4 LDS (`4` CTAs/CU, 12 KiB LDS!)
__global__ __launch_bounds__(192, 4)
void k_hlut2_grouped_down8(
    const uint32_t* __restrict__ inter_rot_f16x2,
    const int32_t*  __restrict__ active_experts,
    const int32_t*  __restrict__ csr_offsets,
    const Mi50RoutedPair* __restrict__ sorted_pairs,
    const uint32_t* __restrict__ down_codes,
    const uint16_t* __restrict__ down_scales,
    const float*    __restrict__ down_cent,
    float*          __restrict__ out_accum_f32)
{
    union SharedDown8 {
        uint4 inter4[8][96]; // 12 KiB LDS: 4 CTAs/CU = 12 waves/CU!
        float red[6][8][32]; // 6 KiB LDS
    };
    __shared__ SharedDown8 sh;

    const int rb = int(blockIdx.x);       // 0..159
    const int ae_idx = int(blockIdx.y);   // 0..num_active-1
    const int tid = int(threadIdx.x);     // 0..191
    const int r = tid & 31;
    const int g = tid >> 5;               // 0..5

    const int p_begin = csr_offsets[ae_idx];
    const int p_end = csr_offsets[ae_idx + 1];
    if (p_begin >= p_end) return;

    const size_t e = size_t(active_experts[ae_idx]);
    uint32_t d_h0, d_dh;
    d_make_hlut2_lut_x2(down_cent + e * 2u, d_h0, d_dh);

    const uint32_t* dc = down_codes + e * 245760u + (size_t(rb) * 48u + size_t(g * 8)) * 32u + r;
    const float sig_d = d_fp16_bits_to_f32(down_scales[e * 30720u + (size_t(rb) * 6u + size_t(g)) * 32u + r]);

    uint32_t pd_reg[8];
    #pragma unroll
    for (int w = 0; w < 8; ++w) pd_reg[w] = dc[w * 32];

    const int ld_t_off = (tid >= 96) ? 1 : 0;
    const int ld_col4  = (tid >= 96) ? (tid - 96) : tid;
    const int ld_sw4   = d_hlut2_swiz4(ld_col4);

    for (int p0 = p_begin; p0 < p_end; p0 += 8) {
        const int cnt = min(8, p_end - p0);
        const int bm  = (cnt <= 4) ? 4 : 8;

        for (int tb = 0; tb < bm; tb += 2) {
            const int t = tb + ld_t_off;
            uint4 vi = {0u, 0u, 0u, 0u};
            if (t < cnt) {
                vi = *reinterpret_cast<const uint4*>(inter_rot_f16x2 + size_t(p0 + t) * 384u + size_t(ld_col4 * 4));
            }
            sh.inter4[t][ld_sw4] = vi;
        }
        __syncthreads();

        float grp[8] = {0.0f};
        if (cnt <= 4) {
            #pragma unroll 4
            for (int w = 0; w < 8; ++w) {
                const uint32_t pd = pd_reg[w];
                #pragma unroll
                for (int kh = 0; kh < 2; ++kh) {
                    const uint32_t pdh = pd >> (16 * kh);
                    const f16x2_t wd[4] = {
                        d_unpack_hlut2_pair(pdh >> 0,  d_h0, d_dh),
                        d_unpack_hlut2_pair(pdh >> 4,  d_h0, d_dh),
                        d_unpack_hlut2_pair(pdh >> 8,  d_h0, d_dh),
                        d_unpack_hlut2_pair(pdh >> 12, d_h0, d_dh)};
                    const int sw4 = d_hlut2_swiz4(g * 16 + w * 2 + kh);
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        const uint4 vi = sh.inter4[t][sw4];
                        const uint32_t words[4] = {vi.x, vi.y, vi.z, vi.w};
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            f16x2_t xv;
                            __builtin_memcpy(&xv, &words[i], 4);
                            grp[t] = __builtin_amdgcn_fdot2(wd[i], xv, grp[t], false);
                        }
                    }
                }
            }
            __syncthreads();
            #pragma unroll
            for (int t = 0; t < 4; ++t) sh.red[g][t][r] = grp[t] * sig_d;
        } else {
            #pragma unroll 2
            for (int w = 0; w < 8; ++w) {
                const uint32_t pd = pd_reg[w];
                #pragma unroll
                for (int kh = 0; kh < 2; ++kh) {
                    const uint32_t pdh = pd >> (16 * kh);
                    const f16x2_t wd[4] = {
                        d_unpack_hlut2_pair(pdh >> 0,  d_h0, d_dh),
                        d_unpack_hlut2_pair(pdh >> 4,  d_h0, d_dh),
                        d_unpack_hlut2_pair(pdh >> 8,  d_h0, d_dh),
                        d_unpack_hlut2_pair(pdh >> 12, d_h0, d_dh)};
                    const int sw4 = d_hlut2_swiz4(g * 16 + w * 2 + kh);
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        const uint4 vi = sh.inter4[t][sw4];
                        const uint32_t words[4] = {vi.x, vi.y, vi.z, vi.w};
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            f16x2_t xv;
                            __builtin_memcpy(&xv, &words[i], 4);
                            grp[t] = __builtin_amdgcn_fdot2(wd[i], xv, grp[t], false);
                        }
                    }
                }
            }
            __syncthreads();
            #pragma unroll
            for (int t = 0; t < 8; ++t) sh.red[g][t][r] = grp[t] * sig_d;
        }
        __syncthreads();

        for (int t = g; t < cnt; t += 6) {
            const Mi50RoutedPair pr = sorted_pairs[p0 + t];
            const float sum = (sh.red[0][t][r] + sh.red[1][t][r] + sh.red[2][t][r] +
                               sh.red[3][t][r] + sh.red[4][t][r] + sh.red[5][t][r]) * pr.weight;
            atomicAdd(&out_accum_f32[size_t(pr.row) * 5120u + rb * 32 + r], sum);
        }
        __syncthreads();
    }
}

}  // namespace

// =============================================================================
// Host Launch Wrappers
// =============================================================================

void launch_tp_embed_bf16(
    const int32_t* d_token_ids, const uint16_t* d_embed_w, uint16_t* d_out_bf16,
    int M, int D, int vocab_offset, int vocab_rows, hipStream_t stream)
{
    if (M <= 0) return;
    dim3 grid((D + 255) / 256, M);
    hipLaunchKernelGGL(k_tp_embed_bf16, grid, dim3(256), 0, stream,
                       d_token_ids, d_embed_w, d_out_bf16, D, vocab_offset, vocab_rows);
}

void launch_rmsnorm_bf16(
    const uint16_t* d_in_bf16, const uint16_t* d_weight_bf16, uint16_t* d_out_bf16,
    int M, int D, float eps, hipStream_t stream)
{
    if (M <= 0) return;
    hipLaunchKernelGGL(k_rmsnorm_bf16, dim3(M), dim3(256), 0, stream,
                       d_in_bf16, d_weight_bf16, d_out_bf16, D, eps);
}

void launch_f32_to_bf16(const float* d_in_f32, uint16_t* d_out_bf16, size_t n_elems, hipStream_t stream) {
    if (n_elems == 0) return;
    hipLaunchKernelGGL(k_f32_to_bf16, dim3((n_elems + 255) / 256), dim3(256), 0, stream, d_in_f32, d_out_bf16, n_elems);
}

void launch_add_f32_2way_inplace(float* d_dst, const float* d_src, size_t n_elems, hipStream_t stream) {
    if (n_elems == 0) return;
    hipLaunchKernelGGL(k_add_f32_2way_inplace, dim3((n_elems + 255) / 256), dim3(256), 0, stream, d_dst, d_src, n_elems);
}

void launch_add_f32_inplace(float* d_dst, const float* d_src1, const float* d_src2, size_t n_elems, hipStream_t stream) {
    if (n_elems == 0) return;
    hipLaunchKernelGGL(k_add_f32_inplace, dim3((n_elems + 255) / 256), dim3(256), 0, stream, d_dst, d_src1, d_src2, n_elems);
}

void launch_add_f32_3way(float* d_out_f32, const float* d_a, const float* d_b, const float* d_c, size_t n_elems, hipStream_t stream) {
    if (n_elems == 0) return;
    if ((n_elems & 3u) == 0u &&
        ((reinterpret_cast<uintptr_t>(d_out_f32) | reinterpret_cast<uintptr_t>(d_a) |
          reinterpret_cast<uintptr_t>(d_b) | reinterpret_cast<uintptr_t>(d_c)) & 15u) == 0u) {
        const size_t n4 = n_elems >> 2;
        hipLaunchKernelGGL(k_add_f32_3way_vec4, dim3((n4 + 255) / 256), dim3(256), 0, stream,
                           reinterpret_cast<float4*>(d_out_f32),
                           reinterpret_cast<const float4*>(d_a),
                           reinterpret_cast<const float4*>(d_b),
                           reinterpret_cast<const float4*>(d_c), n4);
    } else {
        hipLaunchKernelGGL(k_add_f32_3way, dim3((n_elems + 255) / 256), dim3(256), 0, stream, d_out_f32, d_a, d_b, d_c, n_elems);
    }
}

__global__ __launch_bounds__(256)
void k_p2p_push_u4_2peers(
    const uint4* __restrict__ src,
    uint4* __restrict__ peer1_recv,
    uint4* __restrict__ peer2_recv,
    size_t n_u4)
{
    const size_t idx = size_t(blockIdx.x) * 256u + size_t(threadIdx.x);
    if (idx < n_u4) {
        const uint4 v = src[idx];
        peer1_recv[idx] = v;
        peer2_recv[idx] = v;
    }
}

__global__ __launch_bounds__(256)
void k_p2p_push_u16_2peers(
    const uint16_t* __restrict__ src,
    uint16_t* __restrict__ peer1_recv,
    uint16_t* __restrict__ peer2_recv,
    size_t n_elems)
{
    const size_t idx = size_t(blockIdx.x) * 256u + size_t(threadIdx.x);
    if (idx < n_elems) {
        const uint16_t v = src[idx];
        peer1_recv[idx] = v;
        peer2_recv[idx] = v;
    }
}

__global__ __launch_bounds__(256)
void k_p2p_push_f32_scalar_2peers(
    const float* __restrict__ src,
    float* __restrict__ peer1_recv,
    float* __restrict__ peer2_recv,
    size_t n_elems)
{
    const size_t idx = size_t(blockIdx.x) * 256u + size_t(threadIdx.x);
    if (idx < n_elems) {
        const float v = src[idx];
        peer1_recv[idx] = v;
        peer2_recv[idx] = v;
    }
}

void launch_p2p_push_bf16_2peers(
    const uint16_t* d_src, uint16_t* d_peer1_recv, uint16_t* d_peer2_recv,
    size_t n_elems, hipStream_t stream)
{
    if (n_elems == 0) return;
    if ((n_elems & 7u) == 0u &&
        ((reinterpret_cast<uintptr_t>(d_src) | reinterpret_cast<uintptr_t>(d_peer1_recv) |
          reinterpret_cast<uintptr_t>(d_peer2_recv)) & 15u) == 0u) {
        const size_t n8 = n_elems >> 3;
        hipLaunchKernelGGL(k_p2p_push_u4_2peers, dim3((n8 + 255) / 256), dim3(256), 0, stream,
                           reinterpret_cast<const uint4*>(d_src),
                           reinterpret_cast<uint4*>(d_peer1_recv),
                           reinterpret_cast<uint4*>(d_peer2_recv), n8);
    } else {
        hipLaunchKernelGGL(k_p2p_push_u16_2peers, dim3((n_elems + 255) / 256), dim3(256), 0, stream,
                           d_src, d_peer1_recv, d_peer2_recv, n_elems);
    }
}

void launch_p2p_push_f32_2peers(
    const float* d_src, float* d_peer1_recv, float* d_peer2_recv,
    size_t n_elems, hipStream_t stream)
{
    if (n_elems == 0) return;
    if ((n_elems & 3u) == 0u &&
        ((reinterpret_cast<uintptr_t>(d_src) | reinterpret_cast<uintptr_t>(d_peer1_recv) |
          reinterpret_cast<uintptr_t>(d_peer2_recv)) & 15u) == 0u) {
        const size_t n4 = n_elems >> 2;
        hipLaunchKernelGGL(k_p2p_push_u4_2peers, dim3((n4 + 255) / 256), dim3(256), 0, stream,
                           reinterpret_cast<const uint4*>(d_src),
                           reinterpret_cast<uint4*>(d_peer1_recv),
                           reinterpret_cast<uint4*>(d_peer2_recv), n4);
    } else {
        hipLaunchKernelGGL(k_p2p_push_f32_scalar_2peers, dim3((n_elems + 255) / 256), dim3(256), 0, stream,
                           d_src, d_peer1_recv, d_peer2_recv, n_elems);
    }
}

__global__ __launch_bounds__(256, 4)
void k_host_ring_allreduce_bf16(
    const uint16_t* __restrict__ d_src,
    uint16_t*       __restrict__ d_dst,
    HostRingAllReduceSlot* __restrict__ slot,
    int rank,
    uint64_t seq,
    size_t n_elems)
{
    const int tid = int(threadIdx.x);
    const size_t stride = size_t(gridDim.x) * 256u;
    const size_t gid0   = size_t(blockIdx.x) * 256u + size_t(tid);
    const int p1 = (rank + 1) % 3;
    const int p2 = (rank + 2) % 3;

    if (seq > kHostArRingSlots && tid == 0) {
        const uint64_t min_done = seq - kHostArRingSlots;
        while (__hip_atomic_load(&slot->done_seq[p1][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < min_done ||
               __hip_atomic_load(&slot->done_seq[p2][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < min_done) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();

    const size_t n_words = (n_elems + 1u) >> 1;
    for (size_t w = gid0; w < n_words; w += stride) {
        const size_t i0 = w << 1;
        uint32_t packed = uint32_t(d_src[i0]);
        if (i0 + 1u < n_elems) packed |= (uint32_t(d_src[i0 + 1u]) << 16);
        __hip_atomic_store(&slot->payload[rank][w], packed, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence_system();
        const uint32_t prev = __hip_atomic_fetch_add(&slot->write_ctas[rank][0], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM);
        if (prev + 1u == uint32_t(gridDim.x)) {
            __hip_atomic_store(&slot->write_ctas[rank][0], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            __hip_atomic_store(&slot->ready_seq[rank][0], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        while (__hip_atomic_load(&slot->ready_seq[p1][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq ||
               __hip_atomic_load(&slot->ready_seq[p2][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();

    for (size_t w = gid0; w < n_words; w += stride) {
        const size_t i0 = w << 1;
        uint32_t u[3];
        uint32_t loc = uint32_t(d_src[i0]);
        if (i0 + 1u < n_elems) loc |= (uint32_t(d_src[i0 + 1u]) << 16);
        u[rank] = loc;
        u[p1]   = __hip_atomic_load(&slot->payload[p1][w], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        u[p2]   = __hip_atomic_load(&slot->payload[p2][w], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

        const float s0 = (d_bf16_to_f32(uint16_t(u[0])) + d_bf16_to_f32(uint16_t(u[1]))) + d_bf16_to_f32(uint16_t(u[2]));
        d_dst[i0] = d_f32_to_bf16_rne(s0);
        if (i0 + 1u < n_elems) {
            const float s1 = (d_bf16_to_f32(uint16_t(u[0] >> 16)) + d_bf16_to_f32(uint16_t(u[1] >> 16))) + d_bf16_to_f32(uint16_t(u[2] >> 16));
            d_dst[i0 + 1u] = d_f32_to_bf16_rne(s1);
        }
    }
    __syncthreads();

    if (tid == 0) {
        const uint32_t prev = __hip_atomic_fetch_add(&slot->read_ctas[rank][0], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM);
        if (prev + 1u == uint32_t(gridDim.x)) {
            __hip_atomic_store(&slot->read_ctas[rank][0], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            __hip_atomic_store(&slot->done_seq[rank][0], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

__global__ __launch_bounds__(256, 4)
void k_host_ring_allreduce_f32(
    const float* __restrict__ d_src,
    float*       __restrict__ d_dst,
    HostRingAllReduceSlot* __restrict__ slot,
    int rank,
    uint64_t seq,
    size_t n_elems)
{
    const int tid = int(threadIdx.x);
    const size_t stride = size_t(gridDim.x) * 256u;
    const size_t gid0   = size_t(blockIdx.x) * 256u + size_t(tid);
    const int p1 = (rank + 1) % 3;
    const int p2 = (rank + 2) % 3;

    if (seq > kHostArRingSlots && tid == 0) {
        const uint64_t min_done = seq - kHostArRingSlots;
        while (__hip_atomic_load(&slot->done_seq[p1][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < min_done ||
               __hip_atomic_load(&slot->done_seq[p2][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < min_done) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();

    for (size_t i = gid0; i < n_elems; i += stride) {
        const uint32_t bits = __float_as_uint(d_src[i]);
        __hip_atomic_store(&slot->payload[rank][i], bits, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence_system();
        const uint32_t prev = __hip_atomic_fetch_add(&slot->write_ctas[rank][0], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM);
        if (prev + 1u == uint32_t(gridDim.x)) {
            __hip_atomic_store(&slot->write_ctas[rank][0], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            __hip_atomic_store(&slot->ready_seq[rank][0], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        while (__hip_atomic_load(&slot->ready_seq[p1][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq ||
               __hip_atomic_load(&slot->ready_seq[p2][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();

    for (size_t i = gid0; i < n_elems; i += stride) {
        float v[3];
        v[rank] = d_src[i];
        v[p1]   = __uint_as_float(__hip_atomic_load(&slot->payload[p1][i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
        v[p2]   = __uint_as_float(__hip_atomic_load(&slot->payload[p2][i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
        d_dst[i] = (v[0] + v[1]) + v[2];
    }
    __syncthreads();

    if (tid == 0) {
        const uint32_t prev = __hip_atomic_fetch_add(&slot->read_ctas[rank][0], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM);
        if (prev + 1u == uint32_t(gridDim.x)) {
            __hip_atomic_store(&slot->read_ctas[rank][0], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            __hip_atomic_store(&slot->done_seq[rank][0], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

void launch_host_ring_allreduce_bf16(
    const uint16_t* d_src, uint16_t* d_dst, HostRingAllReduceSlot* d_mapped_slot,
    int rank, uint64_t seq, size_t n_elems, hipStream_t stream)
{
    if (n_elems == 0) return;
    const size_t n_words = (n_elems + 1u) >> 1;
    const uint32_t num_ctas = (n_words <= 65536u) ? 16u : 32u;
    hipLaunchKernelGGL(k_host_ring_allreduce_bf16, dim3(num_ctas), dim3(256), 0, stream,
                       d_src, d_dst, d_mapped_slot, rank, seq, n_elems);
}

__global__ __launch_bounds__(256, 4)
void k_host_ring_allreduce_f32_hc_post(
    const float*    __restrict__ d_src_f32,
    const uint16_t* __restrict__ d_p150_bf16,
    const uint16_t* __restrict__ d_res_bf16,
    const float*    __restrict__ d_post_f32,
    const float*    __restrict__ d_comb_f32,
    uint16_t*       __restrict__ d_out_h_bf16,
    HostRingAllReduceSlot* __restrict__ slot,
    int rank,
    uint64_t seq,
    int M)
{
    const int tid = int(threadIdx.x);
    const int d   = int(blockIdx.x) * 256 + tid;  // 20 CTAs * 256 = 5120 columns (co-resident <= 60 CUs!)
    const int p1  = (rank + 1) % 3;
    const int p2  = (rank + 2) % 3;

    if (seq > kHostArRingSlots && tid == 0) {
        const uint64_t min_done = seq - kHostArRingSlots;
        while (__hip_atomic_load(&slot->done_seq[p1][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < min_done ||
               __hip_atomic_load(&slot->done_seq[p2][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < min_done) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();

    for (int m = 0; m < M; ++m) {
        const size_t idx = size_t(m) * 5120u + size_t(d);
        const uint32_t bits = __float_as_uint(d_src_f32[idx]);
        __hip_atomic_store(&slot->payload[rank][idx], bits, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence_system();
        const uint32_t prev = __hip_atomic_fetch_add(&slot->write_ctas[rank][0], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM);
        if (prev + 1u == uint32_t(gridDim.x)) {
            __hip_atomic_store(&slot->write_ctas[rank][0], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            __hip_atomic_store(&slot->ready_seq[rank][0], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        while (__hip_atomic_load(&slot->ready_seq[p1][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq ||
               __hip_atomic_load(&slot->ready_seq[p2][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();

    for (int m = 0; m < M; ++m) {
        const size_t idx = size_t(m) * 5120u + size_t(d);
        float v[3];
        v[rank] = d_src_f32[idx];
        v[p1]   = __uint_as_float(__hip_atomic_load(&slot->payload[p1][idx], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
        v[p2]   = __uint_as_float(__hip_atomic_load(&slot->payload[p2][idx], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
        float xv = (v[0] + v[1]) + v[2];
        if (d_p150_bf16 != nullptr) {
            xv += d_bf16_to_f32(d_p150_bf16[idx]);
        }

        const uint16_t* rs = d_res_bf16 + size_t(m) * 20480u + size_t(d);
        const float r0 = d_bf16_to_f32(rs[0]);
        const float r1 = d_bf16_to_f32(rs[5120]);
        const float r2 = d_bf16_to_f32(rs[10240]);
        const float r3 = d_bf16_to_f32(rs[15360]);

        const float* ps = d_post_f32 + m * 4;
        const float* cb = d_comb_f32 + m * 16;
        uint16_t* dst = d_out_h_bf16 + size_t(m) * 20480u + size_t(d);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float h_val = ps[j] * xv + cb[0 * 4 + j] * r0 + cb[1 * 4 + j] * r1 +
                                cb[2 * 4 + j] * r2 + cb[3 * 4 + j] * r3;
            dst[j * 5120] = d_f32_to_bf16_rne(h_val);
        }
    }
    __syncthreads();

    if (tid == 0) {
        const uint32_t prev = __hip_atomic_fetch_add(&slot->read_ctas[rank][0], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM);
        if (prev + 1u == uint32_t(gridDim.x)) {
            __hip_atomic_store(&slot->read_ctas[rank][0], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            __hip_atomic_store(&slot->done_seq[rank][0], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

void launch_host_ring_allreduce_f32(
    const float* d_src, float* d_dst, HostRingAllReduceSlot* d_mapped_slot,
    int rank, uint64_t seq, size_t n_elems, hipStream_t stream)
{
    if (n_elems == 0) return;
    const uint32_t num_ctas = (n_elems <= 65536u) ? 16u : 32u;
    hipLaunchKernelGGL(k_host_ring_allreduce_f32, dim3(num_ctas), dim3(256), 0, stream,
                       d_src, d_dst, d_mapped_slot, rank, seq, n_elems);
}

void launch_host_ring_allreduce_f32_hc_post(
    const float* d_src_f32, const uint16_t* d_p150_bf16,
    const uint16_t* d_res_bf16, const float* d_post_f32, const float* d_comb_f32,
    uint16_t* d_out_h_bf16, HostRingAllReduceSlot* d_mapped_slot,
    int rank, uint64_t seq, int M, hipStream_t stream)
{
    if (M <= 0) return;
    hipLaunchKernelGGL(k_host_ring_allreduce_f32_hc_post, dim3(20), dim3(256), 0, stream,
                       d_src_f32, d_p150_bf16, d_res_bf16, d_post_f32, d_comb_f32,
                       d_out_h_bf16, d_mapped_slot, rank, seq, M);
}

void launch_add_bf16_3way(
    uint16_t* d_out_bf16, const uint16_t* d_a, const uint16_t* d_b, const uint16_t* d_c,
    size_t n_elems, hipStream_t stream)
{
    if (n_elems == 0) return;
    if ((n_elems & 7u) == 0u &&
        ((reinterpret_cast<uintptr_t>(d_out_bf16) | reinterpret_cast<uintptr_t>(d_a) |
          reinterpret_cast<uintptr_t>(d_b) | reinterpret_cast<uintptr_t>(d_c)) & 15u) == 0u) {
        const size_t n8 = n_elems >> 3;
        hipLaunchKernelGGL(k_add_bf16_3way_vec8, dim3((n8 + 255) / 256), dim3(256), 0, stream,
                           reinterpret_cast<uint4*>(d_out_bf16),
                           reinterpret_cast<const uint4*>(d_a),
                           reinterpret_cast<const uint4*>(d_b),
                           reinterpret_cast<const uint4*>(d_c), n8);
    } else {
        hipLaunchKernelGGL(k_add_bf16_3way, dim3((n_elems + 255) / 256), dim3(256), 0, stream, d_out_bf16, d_a, d_b, d_c, n_elems);
    }
}

void launch_add_f32_and_bf16_to_bf16(
    const float* d_routed_tp3_sum, const uint16_t* d_p150_bf16, uint16_t* d_out_bf16,
    size_t n_elems, hipStream_t stream)
{
    if (n_elems == 0) return;
    hipLaunchKernelGGL(k_add_f32_and_bf16_to_bf16, dim3((n_elems + 255) / 256), dim3(256), 0, stream,
                       d_routed_tp3_sum, d_p150_bf16, d_out_bf16, n_elems);
}

void launch_hc_init(
    const uint16_t* d_embed_bf16, uint16_t* d_h_bf16, float* d_pre_mix_f32,
    int M, hipStream_t stream)
{
    if (M <= 0) return;
    hipLaunchKernelGGL(k_hc_init, dim3(20, M), dim3(256), 0, stream, d_embed_bf16, d_h_bf16, d_pre_mix_f32, M);
}

void launch_hc_mean_bf16(const uint16_t* d_h_bf16, uint16_t* d_mean_bf16, int M, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_hc_mean_bf16, dim3(20, M), dim3(256), 0, stream, d_h_bf16, d_mean_bf16);
}

void launch_hc_mixes(
    const uint16_t* d_h_bf16, const float* d_hc_fn_f32, const float* d_hc_scale_f32, const float* d_hc_base_f32,
    float* d_scratch_rsqrt_f32, float* d_scratch_mixes_f32,
    float* d_out_pre_f32, float* d_out_post_f32, float* d_out_comb_f32,
    int M, float norm_eps, int sinkhorn_iters, float hc_eps, hipStream_t stream)
{
    if (M <= 0) return;
    hipLaunchKernelGGL(k_hc_rms_rsqrt, dim3(M), dim3(256), 0, stream, d_h_bf16, d_scratch_rsqrt_f32, norm_eps);
    if (M >= 16) {
        hipLaunchKernelGGL(k_hc_proj24_bm16, dim3(24, (M + 15) / 16), dim3(256), 0, stream,
                           d_h_bf16, d_hc_fn_f32, d_scratch_rsqrt_f32, d_scratch_mixes_f32, M);
    } else {
        hipLaunchKernelGGL(k_hc_proj24, dim3(24, (M + 7) / 8), dim3(256), 0, stream,
                           d_h_bf16, d_hc_fn_f32, d_scratch_rsqrt_f32, d_scratch_mixes_f32, M);
    }
    hipLaunchKernelGGL(k_hc_sinkhorn, dim3((M + 63) / 64), dim3(64), 0, stream,
                       d_scratch_mixes_f32, d_hc_scale_f32, d_hc_base_f32,
                       d_out_pre_f32, d_out_post_f32, d_out_comb_f32, M, sinkhorn_iters, hc_eps);
}

void launch_hc_pre(const uint16_t* d_h_bf16, const float* d_pre_f32, uint16_t* d_out_bf16, int M, hipStream_t stream) {
    if (M <= 0) return;
    hipLaunchKernelGGL(k_hc_pre, dim3(20, M), dim3(256), 0, stream, d_h_bf16, d_pre_f32, d_out_bf16);
}

void launch_hc_pre_rmsnorm_bf16(
    const uint16_t* d_h_bf16, const float* d_pre_f32, const uint16_t* d_norm_w_bf16,
    uint16_t* d_out_norm_bf16, uint16_t* d_out_raw_bf16,
    int M, float eps, hipStream_t stream,
    const uint64_t* d_signs_u64, uint16_t* d_out_rot_f16, uint16_t* d_out_rot_bf16)
{
    if (M <= 0) return;
    hipLaunchKernelGGL(k_hc_pre_rmsnorm_bf16, dim3(M), dim3(256), 0, stream,
                       d_h_bf16, d_pre_f32, d_norm_w_bf16, d_out_norm_bf16, d_out_raw_bf16, eps,
                       d_signs_u64, d_out_rot_f16, d_out_rot_bf16);
}

void launch_hc_post(
    const uint16_t* d_sub_bf16, const uint16_t* d_res_bf16,
    const float* d_post_f32, const float* d_comb_f32, uint16_t* d_out_h_bf16,
    int M, hipStream_t stream)
{
    if (M <= 0) return;
    hipLaunchKernelGGL(k_hc_post, dim3(20, M), dim3(256), 0, stream,
                       d_sub_bf16, d_res_bf16, d_post_f32, d_comb_f32, d_out_h_bf16);
}

void launch_engram_gate_add(
    uint16_t* d_h_bf16, const float* d_kv_f32, const uint16_t* d_q_weight, const uint16_t* d_k_weight,
    const uint8_t* d_image_mask, int M, float norm_eps, hipStream_t stream)
{
    if (M <= 0) return;
    hipLaunchKernelGGL(k_engram_gate_add, dim3(4, M), dim3(256), 0, stream,
                       d_h_bf16, d_kv_f32, d_q_weight, d_k_weight, d_image_mask, norm_eps);
}

void launch_act_quant_ue8m0_32(
    const uint16_t* d_in_bf16, uint8_t* d_out_fp8, uint8_t* d_out_ue8m0, uint16_t* d_inplace_bf16,
    int M, int K, bool inplace_bf16, hipStream_t stream)
{
    if (M <= 0 || K <= 0) return;
    const int total_groups = M * (K / 32);
    hipLaunchKernelGGL(k_act_quant_ue8m0_32, dim3((total_groups + 7) / 8), dim3(256), 0, stream,
                       d_in_bf16, d_out_fp8, d_out_ue8m0, d_inplace_bf16, total_groups, inplace_bf16);
}

void launch_fp8_gemm(
    const uint8_t* d_x_fp8, const uint8_t* d_x_scale,
    const uint8_t* d_w_fp8, const uint8_t* d_w_scale,
    uint16_t* d_out_bf16, float* d_out_f32,
    int M, int N, int K, bool write_f32, bool accumulate_f32, hipStream_t stream)
{
    if (M <= 0 || N <= 0) return;
    const auto* x2 = reinterpret_cast<const uint16_t*>(d_x_fp8);
    const auto* w2 = reinterpret_cast<const uint16_t*>(d_w_fp8);
    if (M == 1) {
        hipLaunchKernelGGL((k_fp8_gemv_fdot2<1>), dim3((N + 7) / 8), dim3(256), 0, stream,
                           x2, d_x_scale, w2, d_w_scale, d_out_bf16, d_out_f32, M, N, K, write_f32, accumulate_f32);
    } else if (M <= 5) {
        hipLaunchKernelGGL((k_fp8_gemv_fdot2<5>), dim3((N + 7) / 8), dim3(256), 0, stream,
                           x2, d_x_scale, w2, d_w_scale, d_out_bf16, d_out_f32, M, N, K, write_f32, accumulate_f32);
    } else if (M <= 6) {
        hipLaunchKernelGGL((k_fp8_gemv_fdot2<6>), dim3((N + 7) / 8), dim3(256), 0, stream,
                           x2, d_x_scale, w2, d_w_scale, d_out_bf16, d_out_f32, M, N, K, write_f32, accumulate_f32);
    } else if (M <= 8) {
        hipLaunchKernelGGL((k_fp8_gemv_fdot2<8>), dim3((N + 7) / 8), dim3(256), 0, stream,
                           x2, d_x_scale, w2, d_w_scale, d_out_bf16, d_out_f32, M, N, K, write_f32, accumulate_f32);
    } else if (M >= 128 && N >= 64) {
        hipLaunchKernelGGL(k_fp8_gemm_bm128_bn64, dim3((N + 63) / 64, (M + 127) / 128), dim3(256), 0, stream,
                           x2, d_x_scale, w2, d_w_scale, d_out_bf16, d_out_f32, M, N, K, write_f32, accumulate_f32);
    } else if (M >= 32 && N >= 64) {
        hipLaunchKernelGGL(k_fp8_gemm_bm64_bn64, dim3((N + 63) / 64, (M + 63) / 64), dim3(256), 0, stream,
                           x2, d_x_scale, w2, d_w_scale, d_out_bf16, d_out_f32, M, N, K, write_f32, accumulate_f32);
    } else {
        hipLaunchKernelGGL(k_fp8_gemm_tiled32, dim3((N + 31) / 32, (M + 31) / 32), dim3(256), 0, stream,
                           x2, d_x_scale, w2, d_w_scale, d_out_bf16, d_out_f32, M, N, K, write_f32, accumulate_f32);
    }
}

void launch_fp8_grouped_wo_a(
    const uint8_t* d_o_fp8, const uint8_t* d_o_scale,
    const uint8_t* d_wo_a_fp8, const uint8_t* d_wo_a_scale,
    uint16_t* d_out_bf16, int M, int g_cnt, hipStream_t stream)
{
    if (M <= 0 || g_cnt <= 0) return;
    const auto* o2 = reinterpret_cast<const uint16_t*>(d_o_fp8);
    const auto* w2 = reinterpret_cast<const uint16_t*>(d_wo_a_fp8);
    if (M == 1) {
        hipLaunchKernelGGL((k_fp8_grouped_wo_a_mr<1>), dim3(128, g_cnt), dim3(256), 0, stream,
                           o2, d_o_scale, w2, d_wo_a_scale, d_out_bf16, M, g_cnt);
    } else if (M <= 5) {
        hipLaunchKernelGGL((k_fp8_grouped_wo_a_mr<5>), dim3(128, g_cnt), dim3(256), 0, stream,
                           o2, d_o_scale, w2, d_wo_a_scale, d_out_bf16, M, g_cnt);
    } else if (M <= 6) {
        hipLaunchKernelGGL((k_fp8_grouped_wo_a_mr<6>), dim3(128, g_cnt), dim3(256), 0, stream,
                           o2, d_o_scale, w2, d_wo_a_scale, d_out_bf16, M, g_cnt);
    } else if (M <= 8) {
        hipLaunchKernelGGL((k_fp8_grouped_wo_a_mr<8>), dim3(128, g_cnt), dim3(256), 0, stream,
                           o2, d_o_scale, w2, d_wo_a_scale, d_out_bf16, M, g_cnt);
    } else if (M >= 128) {
        hipLaunchKernelGGL(k_fp8_grouped_wo_a_bm128_bn64, dim3(16, (M + 127) / 128, g_cnt), dim3(256), 0, stream,
                           o2, d_o_scale, w2, d_wo_a_scale, d_out_bf16, M, g_cnt);
    } else if (M >= 32) {
        hipLaunchKernelGGL(k_fp8_grouped_wo_a_bm64_bn64, dim3(16, (M + 63) / 64, g_cnt), dim3(256), 0, stream,
                           o2, d_o_scale, w2, d_wo_a_scale, d_out_bf16, M, g_cnt);
    } else {
        hipLaunchKernelGGL(k_fp8_grouped_wo_a_tiled32, dim3(32, (M + 31) / 32, g_cnt), dim3(256), 0, stream,
                           o2, d_o_scale, w2, d_wo_a_scale, d_out_bf16, M, g_cnt);
    }
}

void launch_bf16_gemm(
    const uint16_t* d_x_bf16, const uint16_t* d_w_bf16,
    uint16_t* d_out_bf16, float* d_out_f32,
    int M, int N, int K, bool write_f32, hipStream_t stream)
{
    if (M <= 0 || N <= 0) return;
    if (M == 1) {
        hipLaunchKernelGGL((k_bf16_gemv_mr<1>), dim3((N + 7) / 8), dim3(256), 0, stream,
                           d_x_bf16, d_w_bf16, d_out_bf16, d_out_f32, M, N, K, write_f32);
    } else if (M <= 5) {
        hipLaunchKernelGGL((k_bf16_gemv_mr<5>), dim3((N + 7) / 8), dim3(256), 0, stream,
                           d_x_bf16, d_w_bf16, d_out_bf16, d_out_f32, M, N, K, write_f32);
    } else if (M <= 6) {
        hipLaunchKernelGGL((k_bf16_gemv_mr<6>), dim3((N + 7) / 8), dim3(256), 0, stream,
                           d_x_bf16, d_w_bf16, d_out_bf16, d_out_f32, M, N, K, write_f32);
    } else if (M <= 8) {
        hipLaunchKernelGGL((k_bf16_gemv_mr<8>), dim3((N + 7) / 8), dim3(256), 0, stream,
                           d_x_bf16, d_w_bf16, d_out_bf16, d_out_f32, M, N, K, write_f32);
    } else if (N < 16) {
        for (int m0 = 0; m0 < M; m0 += 8) {
            const int mc = std::min(8, M - m0);
            hipLaunchKernelGGL((k_bf16_gemv_mr<8>), dim3((N + 7) / 8), dim3(256), 0, stream,
                               d_x_bf16 + size_t(m0) * size_t(K), d_w_bf16,
                               d_out_bf16 ? (d_out_bf16 + size_t(m0) * size_t(N)) : nullptr,
                               d_out_f32 ? (d_out_f32 + size_t(m0) * size_t(N)) : nullptr,
                               mc, N, K, write_f32);
        }
    } else if (M >= 128 && N >= 64) {
        hipLaunchKernelGGL(k_bf16_gemm_bm128_bn64, dim3((N + 63) / 64, (M + 127) / 128), dim3(256), 0, stream,
                           d_x_bf16, d_w_bf16, d_out_bf16, d_out_f32, M, N, K, write_f32);
    } else if (M >= 32 && N >= 64) {
        hipLaunchKernelGGL(k_bf16_gemm_bm64_bn64, dim3((N + 63) / 64, (M + 63) / 64), dim3(256), 0, stream,
                           d_x_bf16, d_w_bf16, d_out_bf16, d_out_f32, M, N, K, write_f32);
    } else {
        hipLaunchKernelGGL(k_bf16_gemm_tiled32, dim3((N + 31) / 32, (M + 31) / 32), dim3(256), 0, stream,
                           d_x_bf16, d_w_bf16, d_out_bf16, d_out_f32, M, N, K, write_f32);
    }
}

void launch_swiglu10_bf16(
    const uint16_t* d_gate_bf16, const uint16_t* d_up_bf16, uint16_t* d_out_bf16,
    int n_elems, hipStream_t stream)
{
    if (n_elems <= 0) return;
    hipLaunchKernelGGL(k_swiglu10_bf16, dim3((n_elems + 255) / 256), dim3(256), 0, stream,
                       d_gate_bf16, d_up_bf16, d_out_bf16, n_elems);
}

void launch_swiglu10_split_quant_ue8m0_32(
    const uint16_t* d_gateup_bf16, uint8_t* d_out_fp8, uint8_t* d_out_ue8m0,
    int M, int I, hipStream_t stream)
{
    if (M <= 0 || I <= 0) return;
    const int total_groups = M * (I / 32);
    hipLaunchKernelGGL(k_swiglu10_split_quant_ue8m0_32, dim3((total_groups + 7) / 8), dim3(256), 0, stream,
                       d_gateup_bf16, d_out_fp8, d_out_ue8m0, I, total_groups);
}

void launch_split_qkv_norm_rope_quant(
    const uint16_t* d_qkv_bf16, const uint16_t* d_q_norm_bf16, const uint16_t* d_kv_norm_bf16,
    bool use_comp_yarn,
    uint16_t* d_out_qr_bf16, uint16_t* d_out_kv_bf16,
    int M, int pos_base, float eps, hipStream_t stream)
{
    if (M <= 0) return;
    hipLaunchKernelGGL(k_split_qkv_norm_rope_quant, dim3(2, M), dim3(256), 0, stream,
                       d_qkv_bf16, d_q_norm_bf16, d_kv_norm_bf16, use_comp_yarn,
                       d_out_qr_bf16, d_out_kv_bf16, pos_base, eps);
}

void init_rope_inv_freq_gpu()
{
    auto build_freqs = [](double base, int orig_seq, double factor, double beta_fast, double beta_slow,
                          double out_freqs[32]) {
        constexpr int dim = 64;
        constexpr double kPi = 3.141592653589793238462643383279502884;
        for (int i = 0; i < 32; ++i) {
            out_freqs[i] = 1.0 / std::pow(base, double(2 * i) / double(dim));
        }
        if (orig_seq > 0) {
            auto corr_dim = [&](double rot) {
                return double(dim) * std::log(double(orig_seq) / (rot * 2.0 * kPi)) / (2.0 * std::log(base));
            };
            const double low = std::max(std::floor(corr_dim(beta_fast)), 0.0);
            const double high = std::min(std::ceil(corr_dim(beta_slow)), double(dim - 1));
            const double denom = std::max(high - low, 1e-3);
            for (int i = 0; i < 32; ++i) {
                const double ramp = std::max(0.0, std::min(1.0, (double(i) - low) / denom));
                const double smooth = 1.0 - ramp;
                out_freqs[i] = (out_freqs[i] / factor) * (1.0 - smooth) + out_freqs[i] * smooth;
            }
        }
    };

    double fw[32], fc[32];
    build_freqs(10000.0, 0, 16.0, 32.0, 1.0, fw);
    build_freqs(160000.0, 65536, 16.0, 32.0, 1.0, fc);

    const double* f_tbl[2] = {fw, fc};
    double h_d0_cos[2][16][32];
    double h_d0_sin[2][16][32];
    double h_dk_cos[2][6][7][32];
    double h_dk_sin[2][6][7][32];

    for (int tbl = 0; tbl < 2; ++tbl) {
        for (int m = 0; m < 16; ++m) {
            for (int p = 0; p < 32; ++p) {
                const long double ang = static_cast<long double>(m) * static_cast<long double>(f_tbl[tbl][p]);
                h_d0_cos[tbl][m][p] = static_cast<double>(std::cos(ang));
                h_d0_sin[tbl][m][p] = static_cast<double>(std::sin(ang));
            }
        }
        for (int st = 0; st < 6; ++st) {
            const uint64_t radix_step = 16ULL << (3 * st);
            for (int dg = 1; dg <= 7; ++dg) {
                for (int p = 0; p < 32; ++p) {
                    const long double ang = static_cast<long double>(uint64_t(dg) * radix_step) *
                                            static_cast<long double>(f_tbl[tbl][p]);
                    h_dk_cos[tbl][st][dg - 1][p] = static_cast<double>(std::cos(ang));
                    h_dk_sin[tbl][st][dg - 1][p] = static_cast<double>(std::sin(ang));
                }
            }
        }
    }

    DSV41_HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_inv_freq_win_f64), fw, sizeof(fw), 0, hipMemcpyHostToDevice));
    DSV41_HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_inv_freq_comp_f64), fc, sizeof(fc), 0, hipMemcpyHostToDevice));
    DSV41_HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_rope_d0_cos), h_d0_cos, sizeof(h_d0_cos), 0, hipMemcpyHostToDevice));
    DSV41_HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_rope_d0_sin), h_d0_sin, sizeof(h_d0_sin), 0, hipMemcpyHostToDevice));
    DSV41_HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_rope_dk_cos), h_dk_cos, sizeof(h_dk_cos), 0, hipMemcpyHostToDevice));
    DSV41_HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_rope_dk_sin), h_dk_sin, sizeof(h_dk_sin), 0, hipMemcpyHostToDevice));
}

void launch_rope_tail64_bf16(
    uint16_t* d_x_bf16, bool use_comp_yarn,
    int M, int H, int D, int pos_base, int pos_stride, bool inverse, hipStream_t stream)
{
    if (M <= 0 || H <= 0) return;
    if (M >= 16 && pos_stride == 1) {
        hipLaunchKernelGGL(k_rope_tail64_givens_bf16, dim3((H + 3) / 4, (M + 63) / 64), dim3(256), 0, stream,
                           d_x_bf16, use_comp_yarn, M, H, D, pos_base, inverse);
    } else {
        hipLaunchKernelGGL(k_rope_tail64_bf16, dim3(H, M), dim3(32), 0, stream,
                           d_x_bf16, use_comp_yarn, H, D, pos_base, pos_stride, inverse);
    }
}

int launch_compressor_r2_pool(
    const float* d_kv_f32, const float* d_score_f32,
    float* d_kv_state_f32, float* d_score_state_f32,
    uint16_t* d_out_pool_bf16, int start_pos, int M, hipStream_t stream)
{
    if (M <= 0) return 0;
    hipLaunchKernelGGL(k_compressor_r2_pool, dim3(2), dim3(256), 0, stream,
                       d_kv_f32, d_score_f32, d_kv_state_f32, d_score_state_f32, d_out_pool_bf16, start_pos, M);
    return ((start_pos + M) / 2) - (start_pos / 2);
}

void launch_pack_compress_kv_fp4_e4m3_16(
    const uint16_t* d_latent_bf16, uint8_t* d_cache_fp4, uint8_t* d_cache_e4m3,
    int slot_offset, int N_lat, hipStream_t stream)
{
    if (N_lat <= 0) return;
    hipLaunchKernelGGL(k_pack_compress_kv_fp4_e4m3_16, dim3(N_lat), dim3(256), 0, stream,
                       d_latent_bf16, d_cache_fp4, d_cache_e4m3, slot_offset);
}

void launch_pack_index_k_fp4_ue8m0_32(
    const uint16_t* d_k_bf16, uint8_t* d_cache_fp4, uint8_t* d_cache_ue8m0,
    int slot_offset, int N_lat, hipStream_t stream)
{
    if (N_lat <= 0) return;
    const int total_groups = N_lat * 4;
    hipLaunchKernelGGL(k_pack_index_k_fp4_ue8m0_32, dim3((total_groups + 7) / 8), dim3(256), 0, stream,
                       d_k_bf16, d_cache_fp4, d_cache_ue8m0, slot_offset, total_groups);
}

void launch_fp4_quant_dequant_ue8m0_32_inplace(uint16_t* d_x_bf16, int rows, int D, hipStream_t stream) {
    if (rows <= 0 || D <= 0) return;
    const int total_groups = rows * (D / 32);
    hipLaunchKernelGGL(k_fp4_quant_dequant_ue8m0_32_inplace, dim3((total_groups + 7) / 8), dim3(256), 0, stream,
                       d_x_bf16, total_groups);
}

void launch_indexer_score(
    const uint16_t* d_q_bf16, const uint8_t* d_k_fp4, const uint8_t* d_k_ue8m0,
    const uint16_t* d_weights_bf16, float* d_out_score_f32,
    int M, int ih_cnt, int T_comp, hipStream_t stream)
{
    if (M <= 0 || T_comp <= 0) return;
    const uint32_t tx = uint32_t((T_comp + 255) / 256);
    if (M == 1) {
        hipLaunchKernelGGL(k_indexer_score, dim3(tx, 1), dim3(256), 0, stream,
                           d_q_bf16, d_k_fp4, d_k_ue8m0, d_weights_bf16, d_out_score_f32, ih_cnt, T_comp);
    } else if (M == 2) {
        hipLaunchKernelGGL((k_indexer_score_msplit<2>), dim3(tx, 1), dim3(256), 0, stream,
                           d_q_bf16, d_k_fp4, d_k_ue8m0, d_weights_bf16, d_out_score_f32, M, ih_cnt, T_comp);
    } else if (M <= 4) {
        hipLaunchKernelGGL((k_indexer_score_msplit<4>), dim3(tx, 1), dim3(256), 0, stream,
                           d_q_bf16, d_k_fp4, d_k_ue8m0, d_weights_bf16, d_out_score_f32, M, ih_cnt, T_comp);
    } else {
        hipLaunchKernelGGL((k_indexer_score_msplit<6>), dim3(tx, (M + 5) / 6), dim3(256), 0, stream,
                           d_q_bf16, d_k_fp4, d_k_ue8m0, d_weights_bf16, d_out_score_f32, M, ih_cnt, T_comp);
    }
}

void launch_indexer_candidates_and_topk(
    float* d_score_f32, uint8_t* d_cand_block_mask, int cand_block_stride,
    int32_t* d_out_comp_idxs,
    int M, int T_comp, int start_pos, int ratio, int mode, int K_out, hipStream_t stream)
{
    if (M <= 0 || T_comp <= 0 || K_out <= 0) return;
    float* block_scratch = d_score_f32 + size_t(M) * size_t(T_comp);
    hipLaunchKernelGGL(k_indexer_candidates_and_topk, dim3(M), dim3(256), 0, stream,
                       d_score_f32, d_cand_block_mask, cand_block_stride,
                       d_out_comp_idxs, T_comp, start_pos, ratio, mode, K_out, block_scratch);
}

void launch_sparse_attn(
    const uint16_t* d_q_bf16, const float* d_attn_sink_f32,
    const uint16_t* d_win_kv_ring, const uint16_t* d_cur_kv_bf16,
    int start_pos, int swa_floor_pos, bool dspark_bidir,
    const uint8_t* d_comp_fp4, const uint8_t* d_comp_e4m3, const int32_t* d_comp_idxs, int n_comp,
    float* d_splitk_scratch,
    uint16_t* d_out_o_bf16, int M, int h_cnt, hipStream_t stream)
{
    if (M <= 0 || h_cnt <= 0) return;
    const int max_win = std::min(128, std::max(0, start_pos + M - swa_floor_pos));
    if (M <= 8 && !dspark_bidir && d_splitk_scratch != nullptr && (max_win + n_comp) >= 64) {
        hipLaunchKernelGGL(k_sparse_attn_splitk, dim3(M, (h_cnt + 7) / 8, 8), dim3(256), 0, stream,
                           d_q_bf16, d_win_kv_ring, d_cur_kv_bf16, start_pos, swa_floor_pos,
                           d_comp_fp4, d_comp_e4m3, d_comp_idxs, n_comp,
                           d_splitk_scratch, h_cnt);
        hipLaunchKernelGGL(k_sparse_attn_reduce_splits, dim3(M, h_cnt), dim3(256), 0, stream,
                           d_splitk_scratch, d_attn_sink_f32, d_out_o_bf16, h_cnt);
    } else if (M > 16 && !dspark_bidir && h_cnt >= 8) {
        hipLaunchKernelGGL(k_sparse_attn_allheads, dim3(M, (h_cnt + 7) / 8), dim3(256), 0, stream,
                           d_q_bf16, d_attn_sink_f32,
                           d_win_kv_ring, d_cur_kv_bf16, start_pos, swa_floor_pos,
                           d_comp_fp4, d_comp_e4m3, d_comp_idxs, n_comp,
                           d_out_o_bf16, h_cnt);
    } else {
        hipLaunchKernelGGL(k_sparse_attn, dim3(M, (h_cnt + 7) / 8), dim3(256), 0, stream,
                           d_q_bf16, d_attn_sink_f32,
                           d_win_kv_ring, d_cur_kv_bf16, start_pos, swa_floor_pos, dspark_bidir,
                           d_comp_fp4, d_comp_e4m3, d_comp_idxs, n_comp,
                           d_out_o_bf16, M, h_cnt);
    }
}

void launch_moe_gate(
    const float* d_logits_f32, const float* d_bias_f32, const float* d_bias_vl_f32,
    const uint8_t* d_image_mask, int32_t* d_out_ids, float* d_out_weights,
    int M, int n_routed, int top_k, float route_scale, hipStream_t stream)
{
    if (M <= 0) return;
    hipLaunchKernelGGL(k_moe_gate, dim3(M), dim3(256), 0, stream,
                       d_logits_f32, d_bias_f32, d_bias_vl_f32, d_image_mask,
                       d_out_ids, d_out_weights, n_routed, top_k, route_scale);
}

__global__ __launch_bounds__(64)
void k_pack_mtp_routed_pairs(
    const int32_t* __restrict__ topk_ids,
    const float*   __restrict__ topk_w,
    Mi50RoutedPair* __restrict__ out_pairs,
    int total_pairs,
    int top_k)
{
    const int idx = int(blockIdx.x) * 64 + int(threadIdx.x);
    if (idx >= total_pairs) return;
    Mi50RoutedPair p;
    p.row    = idx / top_k;
    p.expert = topk_ids[idx];
    p.weight = topk_w[idx];
    p.pad    = 0;
    out_pairs[idx] = p;
}

void launch_pack_mtp_routed_pairs(
    const int32_t* d_topk_ids, const float* d_topk_weights,
    Mi50RoutedPair* d_out_pairs, int M, int top_k, hipStream_t stream)
{
    const int total_pairs = M * top_k;
    if (total_pairs <= 0) return;
    hipLaunchKernelGGL(k_pack_mtp_routed_pairs, dim3((total_pairs + 63) / 64), dim3(64), 0, stream,
                       d_topk_ids, d_topk_weights, d_out_pairs, total_pairs, top_k);
}

void launch_wht128_bf16_to_f16(
    const uint16_t* d_in_bf16, uint16_t* d_out_f16, const uint64_t* d_signs_u64,
    int rows, int K, hipStream_t stream)
{
    if (rows <= 0 || K <= 0) return;
    hipLaunchKernelGGL(k_wht128_bf16_to_f16, dim3(K / 128, rows), dim3(128), 0, stream,
                       d_in_bf16, d_out_f16, d_signs_u64, K);
}

void launch_mi50_routed_hlut2_pairs(
    const uint16_t* d_x_rot_f16, const Mi50RoutedPair* d_pairs, int P,
    const uint32_t* d_gate_codes, const uint16_t* d_gate_scales, const float* d_gate_centroids,
    const uint32_t* d_up_codes, const uint16_t* d_up_scales, const float* d_up_centroids,
    const uint32_t* d_down_codes, const uint16_t* d_down_scales, const float* d_down_centroids,
    const uint64_t* d_signs_i,
    uint16_t* d_inter_bf16_scratch, uint16_t* d_inter_rot_f16_scratch,
    float* d_out_accum_f32, hipStream_t stream)
{
    (void)d_inter_bf16_scratch;
    if (P <= 0) return;
    hipLaunchKernelGGL(
        k_hlut2_pair_swiglu, dim3(6, P), dim3(256), 0, stream,
        reinterpret_cast<const uint32_t*>(d_x_rot_f16), d_pairs,
        d_gate_codes, d_gate_scales, d_gate_centroids,
        d_up_codes, d_up_scales, d_up_centroids,
        d_signs_i, d_inter_rot_f16_scratch);
    hipLaunchKernelGGL(
        k_hlut2_pair_down_atomic, dim3(160, P), dim3(192), 0, stream,
        reinterpret_cast<const uint32_t*>(d_inter_rot_f16_scratch), d_pairs,
        d_down_codes, d_down_scales, d_down_centroids,
        d_out_accum_f32);
}

void launch_mi50_routed_hlut2_grouped(
    const uint16_t* d_x_rot_f16,
    const int32_t* d_active_experts, const int32_t* d_csr_offsets, const Mi50RoutedPair* d_sorted_pairs,
    int num_active, int P,
    const uint32_t* d_gate_codes, const uint16_t* d_gate_scales, const float* d_gate_centroids,
    const uint32_t* d_up_codes, const uint16_t* d_up_scales, const float* d_up_centroids,
    const uint32_t* d_down_codes, const uint16_t* d_down_scales, const float* d_down_centroids,
    const uint64_t* d_signs_i,
    uint16_t* d_inter_bf16_scratch, uint16_t* d_inter_rot_f16_scratch,
    float* d_out_accum_f32, hipStream_t stream)
{
    if (num_active <= 0 || P <= 0) return;
    if (P <= 640) {
        hipLaunchKernelGGL(
            k_hlut2_grouped_swiglu8, dim3(6, num_active), dim3(256), 0, stream,
            reinterpret_cast<const uint32_t*>(d_x_rot_f16),
            d_active_experts, d_csr_offsets, d_sorted_pairs,
            d_gate_codes, d_gate_scales, d_gate_centroids,
            d_up_codes, d_up_scales, d_up_centroids,
            d_signs_i, d_inter_bf16_scratch, d_inter_rot_f16_scratch);
    } else {
        hipLaunchKernelGGL(
            k_hlut2_grouped_swiglu16, dim3(6, num_active), dim3(256), 0, stream,
            reinterpret_cast<const uint32_t*>(d_x_rot_f16),
            d_active_experts, d_csr_offsets, d_sorted_pairs,
            d_gate_codes, d_gate_scales, d_gate_centroids,
            d_up_codes, d_up_scales, d_up_centroids,
            d_signs_i, d_inter_bf16_scratch, d_inter_rot_f16_scratch);
    }
    hipLaunchKernelGGL(
        k_hlut2_grouped_down8, dim3(160, num_active), dim3(192), 0, stream,
        reinterpret_cast<const uint32_t*>(d_inter_rot_f16_scratch),
        d_active_experts, d_csr_offsets, d_sorted_pairs,
        d_down_codes, d_down_scales, d_down_centroids,
        d_out_accum_f32);
}

__global__ __launch_bounds__(64)
void k_keep_dpm_warm(
    const volatile uint32_t* __restrict__ active_flag,
    volatile uint32_t* __restrict__ uc_scratch,
    uint32_t mask_u32)
{
    const uint32_t tid = blockIdx.x * 64u + threadIdx.x;
    uint32_t acc = tid + 1u;
    uint32_t idx = (tid * 16u) & mask_u32;
    for (uint64_t it = 0; it < 600000000ULL; ++it) {
        if ((it & 63ULL) == 0ULL && *active_flag == 0u) break;
        acc ^= uc_scratch[idx] + uint32_t(it);
        idx = (idx + 256u) & mask_u32;
        __builtin_amdgcn_s_sleep(2);
    }
    if (tid == 0u && acc == 0xFFFFFFFFu) {
        uc_scratch[0] = acc;
    }
}

__global__ __launch_bounds__(256)
void k_add_bf16_2way_inplace(uint16_t* __restrict__ dst, const uint16_t* __restrict__ src, size_t n) {
    const size_t idx = size_t(blockIdx.x) * 256u + size_t(threadIdx.x);
    if (idx < n) {
        dst[idx] = d_f32_to_bf16_rne(d_bf16_to_f32(dst[idx]) + d_bf16_to_f32(src[idx]));
    }
}

void launch_add_bf16_2way_inplace(uint16_t* d_dst_bf16, const uint16_t* d_src_bf16, size_t n_elems, hipStream_t stream) {
    if (n_elems == 0) return;
    hipLaunchKernelGGL(k_add_bf16_2way_inplace, dim3((n_elems + 255) / 256), dim3(256), 0, stream, d_dst_bf16, d_src_bf16, n_elems);
}

__global__ __launch_bounds__(128)
void k_build_mtp_csr(
    const int32_t* __restrict__ topk_ids,
    const float*   __restrict__ topk_w,
    int32_t*       __restrict__ active_experts,
    int32_t*       __restrict__ csr_offsets,
    Mi50RoutedPair* __restrict__ sorted_pairs,
    int total_pairs,
    int top_k,
    int n_experts)
{
    __shared__ int32_t s_cnt[128];
    __shared__ int32_t s_off[129];
    __shared__ int32_t s_act[128];

    const int tid = int(threadIdx.x); // 0..127
    int my_cnt = 0;
    if (tid < n_experts) {
        for (int i = 0; i < total_pairs; ++i) {
            if (topk_ids[i] == tid && topk_w[i] != 0.0f) ++my_cnt;
        }
    }
    s_cnt[tid] = my_cnt;
    __syncthreads();

    if (tid == 0) {
        int num_act = 0;
        int cur = 0;
        for (int e = 0; e < n_experts; ++e) {
            const int c = s_cnt[e];
            if (c > 0) {
                s_act[num_act] = e;
                s_off[num_act] = cur;
                s_cnt[e] = cur;
                cur += c;
                ++num_act;
            } else {
                s_cnt[e] = -1;
            }
        }
        for (int k = num_act; k <= n_experts; ++k) s_off[k] = cur;
        for (int k = num_act; k < n_experts; ++k) s_act[k] = 0;
    }
    __syncthreads();

    if (tid < n_experts) {
        active_experts[tid] = s_act[tid];
        csr_offsets[tid] = s_off[tid];
    }
    if (tid == 0) {
        csr_offsets[n_experts] = s_off[n_experts];
        for (int i = 0; i < total_pairs; ++i) {
            const int e = topk_ids[i];
            const float w = topk_w[i];
            if (e >= 0 && e < n_experts && w != 0.0f) {
                const int dst = s_cnt[e]++;
                Mi50RoutedPair p;
                p.row    = i / top_k;
                p.expert = e;
                p.weight = w;
                p.pad    = 0;
                sorted_pairs[dst] = p;
            }
        }
    }
}

void launch_build_mtp_csr(
    const int32_t* d_topk_ids, const float* d_topk_weights,
    int32_t* d_active_experts, int32_t* d_csr_offsets, Mi50RoutedPair* d_sorted_pairs,
    int M, int top_k, int n_experts, hipStream_t stream)
{
    const int total_pairs = M * top_k;
    if (total_pairs <= 0 || n_experts <= 0 || n_experts > 128) return;
    hipLaunchKernelGGL(k_build_mtp_csr, dim3(1), dim3(128), 0, stream,
                       d_topk_ids, d_topk_weights, d_active_experts, d_csr_offsets, d_sorted_pairs,
                       total_pairs, top_k, n_experts);
}

void launch_keep_dpm_warm(
    const uint32_t* d_active_flag,
    uint32_t* d_uc_scratch,
    uint32_t mask_u32,
    hipStream_t stream)
{
    hipLaunchKernelGGL(k_keep_dpm_warm, dim3(4), dim3(64), 0, stream,
                       d_active_flag, d_uc_scratch, mask_u32);
}

namespace {

__device__ __forceinline__ float d_cephes_exp_f32(float x_raw) {
    if (!(x_raw > -88.0f)) return 0.0f;
    const float x = fminf(fmaxf(x_raw, -88.0f), 88.0f);
    const float n = __builtin_rintf(x * 1.4426950408889634f);
    const float r = fmaf(n, -1.428606765330187045e-6f, fmaf(n, -0.693145751953125f, x));
    float y = fmaf(1.9875691500E-4f, r, 1.3981999507E-3f);
    y = fmaf(y, r, 8.3334519073E-3f);
    y = fmaf(y, r, 4.1665795894E-2f);
    y = fmaf(y, r, 1.6666665459E-1f);
    y = fmaf(y, r, 5.0000001201E-1f);
    y = fmaf(y, r * r, r + 1.0f);
    const uint32_t pow2n_bits = uint32_t(int32_t(n) + 127) << 23;
    return y * __uint_as_float(pow2n_bits);
}

__global__ __launch_bounds__(256)
void k_dspark_prepare_markov_emb(
    DsparkGpuSampleSlot* __restrict__ ctrl,
    const uint16_t*      __restrict__ mapped_markov_embed_bf16,
    uint16_t*            __restrict__ out_markov_emb_bf16,
    uint64_t seq,
    int step_idx,
    int eos_id)
{
    const int tid = int(threadIdx.x);
    if (step_idx > 0 && tid == 0) {
        while (__hip_atomic_load(&ctrl->ready_tok_seq[step_idx - 1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();

    if (step_idx > 0) {
        if (ctrl->out_fallback_step >= 0 || ctrl->out_drafted_cnt < step_idx || ctrl->out_drafts[step_idx - 1] == eos_id) {
            out_markov_emb_bf16[tid] = 0u;
            return;
        }
    }
    const int32_t tok = (step_idx == 0) ? ctrl->anchor_token : ctrl->out_drafts[step_idx - 1];
    out_markov_emb_bf16[tid] = mapped_markov_embed_bf16[size_t(tok) * 256u + size_t(tid)];
}

__global__ __launch_bounds__(256)
void k_dspark_step_sample_tp3(
    DsparkGpuSampleSlot* __restrict__ ctrl,
    float*               __restrict__ base_logits_f32,
    float*               __restrict__ markov_logits_f32,
    uint64_t seq,
    int step_idx,
    int rank,
    int vocab_offset,
    int vocab_rows,
    int eos_id)
{
    union SharedDspark {
        uint32_t seen[1352];
        double   d256[256];
        struct {
            float   val[kDsparkMaxCand];
            int32_t id[kDsparkMaxCand];
        } sort;
    };
    __shared__ SharedDspark sh;
    __shared__ uint32_t s_cnt;
    __shared__ uint32_t s_ovf;
    __shared__ uint32_t s_keep_n;
    __shared__ float    s_renorm;

    const int tid = int(threadIdx.x);

    if (step_idx > 0 && (ctrl->out_fallback_step >= 0 || ctrl->out_drafted_cnt < step_idx ||
                         ctrl->out_drafts[step_idx - 1] == eos_id)) {
        if (rank == 0 && tid == 0) {
            __hip_atomic_store(&ctrl->ready_tok_seq[step_idx], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        return;
    }

    for (int i = tid; i < 1352; i += 256) sh.seen[i] = 0u;
    __syncthreads();

    const float penalty = ctrl->rep_penalty;
    const bool do_rep = (penalty > 1.0001f);
    if (do_rep) {
        const int R = ctrl->recent_len;
        const int hist_start = max(0, (R + step_idx + 1) - 1024);
        for (int k = hist_start + tid; k < R; k += 256) {
            const int32_t t = ctrl->recent_hist[k];
            if (t >= vocab_offset && t < vocab_offset + vocab_rows) {
                const int lt = t - vocab_offset;
                atomicOr(&sh.seen[lt >> 5], 1u << (lt & 31));
            }
        }
        if (tid <= step_idx) {
            const int32_t t = (tid == 0) ? ctrl->anchor_token : ctrl->out_drafts[tid - 1];
            if (t >= vocab_offset && t < vocab_offset + vocab_rows) {
                const int lt = t - vocab_offset;
                atomicOr(&sh.seen[lt >> 5], 1u << (lt & 31));
            }
        }
        __syncthreads();
    }

    float loc_max = -1.0e38f;
    int32_t loc_id = vocab_offset;
    for (int v = tid; v < vocab_rows; v += 256) {
        float lg = base_logits_f32[v] + markov_logits_f32[v];
        if (do_rep && ((sh.seen[v >> 5] >> (v & 31)) & 1u)) {
            lg = (lg > 0.0f) ? (lg / penalty) : (lg * penalty);
        }
        base_logits_f32[v] = lg;
        const int32_t gid = vocab_offset + v;
        if (lg > loc_max || (lg == loc_max && gid < loc_id)) {
            loc_max = lg;
            loc_id  = gid;
        }
    }
    __syncthreads();

    sh.sort.val[tid] = loc_max;
    sh.sort.id[tid]  = loc_id;
    __syncthreads();
    for (int half = 128; half > 0; half >>= 1) {
        if (tid < half) {
            const float vb = sh.sort.val[tid + half];
            const int32_t ib = sh.sort.id[tid + half];
            const float va = sh.sort.val[tid];
            const int32_t ia = sh.sort.id[tid];
            if (vb > va || (vb == va && ib < ia)) {
                sh.sort.val[tid] = vb;
                sh.sort.id[tid]  = ib;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        ctrl->rank_max[step_idx][rank]    = sh.sort.val[0];
        ctrl->rank_argmax[step_idx][rank] = sh.sort.id[0];
        __threadfence_system();
        __hip_atomic_store(&ctrl->ready_max_seq[step_idx][rank], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        while (__hip_atomic_load(&ctrl->ready_max_seq[step_idx][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq ||
               __hip_atomic_load(&ctrl->ready_max_seq[step_idx][1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq ||
               __hip_atomic_load(&ctrl->ready_max_seq[step_idx][2], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();

    const float temp = ctrl->temperature;
    if (temp <= 1e-5f) {
        if (rank == 0 && tid == 0) {
            float g_max = ctrl->rank_max[step_idx][0];
            int32_t g_id = ctrl->rank_argmax[step_idx][0];
            for (int r = 1; r < 3; ++r) {
                const float rm = ctrl->rank_max[step_idx][r];
                if (rm > g_max) {
                    g_max = rm;
                    g_id  = ctrl->rank_argmax[step_idx][r];
                }
            }
            ctrl->kept_cnt[step_idx]       = 1u;
            ctrl->kept_id[step_idx][0]     = g_id;
            ctrl->kept_q[step_idx][0]      = 1.0f;
            ctrl->out_drafts[step_idx]     = g_id;
            ctrl->out_q_selected[step_idx] = 1.0f;
            ctrl->out_drafted_cnt          = step_idx + 1;
            __threadfence_system();
            __hip_atomic_store(&ctrl->ready_tok_seq[step_idx], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        return;
    }

    const float top_p = ctrl->top_p;
    if (!(top_p > 0.0f && top_p < 1.0f)) {
        if (rank == 0 && tid == 0) {
            ctrl->out_fallback_step = step_idx;
            __threadfence_system();
            __hip_atomic_store(&ctrl->ready_tok_seq[step_idx], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        return;
    }

    const float g_max = fmaxf(fmaxf(ctrl->rank_max[step_idx][0], ctrl->rank_max[step_idx][1]), ctrl->rank_max[step_idx][2]);
    const float inv_t = 1.0f / temp;
    double loc_sum_exp = 0.0;
    for (int v = tid; v < vocab_rows; v += 256) {
        const float vp = d_cephes_exp_f32((base_logits_f32[v] - g_max) * inv_t);
        markov_logits_f32[v] = vp;
        loc_sum_exp += double(vp);
    }
    sh.d256[tid] = loc_sum_exp;
    __syncthreads();
    for (int half = 128; half > 0; half >>= 1) {
        if (tid < half) sh.d256[tid] += sh.d256[tid + half];
        __syncthreads();
    }

    if (tid == 0) {
        ctrl->rank_sum_exp[step_idx][rank] = sh.d256[0];
        __threadfence_system();
        __hip_atomic_store(&ctrl->ready_exp_seq[step_idx][rank], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        while (__hip_atomic_load(&ctrl->ready_exp_seq[step_idx][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq ||
               __hip_atomic_load(&ctrl->ready_exp_seq[step_idx][1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq ||
               __hip_atomic_load(&ctrl->ready_exp_seq[step_idx][2], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();

    const double sum_exp = ctrl->rank_sum_exp[step_idx][0] + ctrl->rank_sum_exp[step_idx][1] + ctrl->rank_sum_exp[step_idx][2];
    const float inv_sum = float(1.0 / fmax(sum_exp, 1e-30));

    auto run_cand_tier = [&](float tau, uint64_t cand_tag, bool wait_all) {
        if (tid == 0) {
            s_cnt = 0u;
            s_ovf = 0u;
        }
        __syncthreads();
        double m_acc = 0.0;
        for (int v = tid; v < vocab_rows; v += 256) {
            const float p = markov_logits_f32[v] * inv_sum;
            if (p > tau) {
                m_acc += double(p);
                const uint32_t pos = atomicAdd(&s_cnt, 1u);
                if (pos < uint32_t(kDsparkMaxCand)) {
                    ctrl->rank_cand_id[rank][pos]  = vocab_offset + v;
                    ctrl->rank_cand_exp[rank][pos] = p;
                } else {
                    s_ovf = 1u;
                }
            }
        }
        sh.d256[tid] = m_acc;
        __syncthreads();
        for (int half = 128; half > 0; half >>= 1) {
            if (tid < half) sh.d256[tid] += sh.d256[tid + half];
            __syncthreads();
        }
        if (tid == 0) {
            ctrl->rank_cand_cnt[step_idx][rank]  = min(s_cnt, uint32_t(kDsparkMaxCand));
            ctrl->rank_overflow[step_idx][rank]  = s_ovf;
            ctrl->rank_cand_mass[step_idx][rank] = sh.d256[0];
            __threadfence_system();
            __hip_atomic_store(&ctrl->ready_cand_seq[step_idx][rank], cand_tag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            if (wait_all) {
                while (__hip_atomic_load(&ctrl->ready_cand_seq[step_idx][0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < cand_tag ||
                       __hip_atomic_load(&ctrl->ready_cand_seq[step_idx][1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < cand_tag ||
                       __hip_atomic_load(&ctrl->ready_cand_seq[step_idx][2], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < cand_tag) {
                    __builtin_amdgcn_s_sleep(1);
                }
            }
        }
        __syncthreads();
    };

    run_cand_tier(2.5e-4f, seq * 2ULL, true);
    const double mass1 = ctrl->rank_cand_mass[step_idx][0] + ctrl->rank_cand_mass[step_idx][1] + ctrl->rank_cand_mass[step_idx][2];
    const uint32_t cnt1 = ctrl->rank_cand_cnt[step_idx][0] + ctrl->rank_cand_cnt[step_idx][1] + ctrl->rank_cand_cnt[step_idx][2];
    const uint32_t ovf1 = ctrl->rank_overflow[step_idx][0] | ctrl->rank_overflow[step_idx][1] | ctrl->rank_overflow[step_idx][2];
    if (ovf1 != 0u || cnt1 > uint32_t(kDsparkMaxCand) || mass1 < double(top_p)) {
        run_cand_tier(1.0e-5f, seq * 2ULL + 1ULL, rank == 0);
    }

    if (rank != 0) return;

    const uint32_t c0 = ctrl->rank_cand_cnt[step_idx][0];
    const uint32_t c1 = ctrl->rank_cand_cnt[step_idx][1];
    const uint32_t c2 = ctrl->rank_cand_cnt[step_idx][2];
    const uint32_t total_cnt = c0 + c1 + c2;
    const uint32_t any_ovf = ctrl->rank_overflow[step_idx][0] | ctrl->rank_overflow[step_idx][1] | ctrl->rank_overflow[step_idx][2];
    const double total_mass = ctrl->rank_cand_mass[step_idx][0] + ctrl->rank_cand_mass[step_idx][1] + ctrl->rank_cand_mass[step_idx][2];

    if (any_ovf != 0u || total_cnt == 0u || total_cnt > uint32_t(kDsparkMaxCand) || total_mass < double(top_p)) {
        if (tid == 0) {
            ctrl->out_fallback_step = step_idx;
            __threadfence_system();
            __hip_atomic_store(&ctrl->ready_tok_seq[step_idx], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        return;
    }

    uint32_t np2 = 2u;
    while (np2 < total_cnt) np2 <<= 1u;

    for (uint32_t idx = uint32_t(tid); idx < np2; idx += 256u) {
        if (idx < c0) {
            sh.sort.val[idx] = ctrl->rank_cand_exp[0][idx];
            sh.sort.id[idx]  = ctrl->rank_cand_id[0][idx];
        } else if (idx < c0 + c1) {
            const uint32_t k = idx - c0;
            sh.sort.val[idx] = ctrl->rank_cand_exp[1][k];
            sh.sort.id[idx]  = ctrl->rank_cand_id[1][k];
        } else if (idx < total_cnt) {
            const uint32_t k = idx - (c0 + c1);
            sh.sort.val[idx] = ctrl->rank_cand_exp[2][k];
            sh.sort.id[idx]  = ctrl->rank_cand_id[2][k];
        } else {
            sh.sort.val[idx] = -1.0f;
            sh.sort.id[idx]  = 0x70000000 + int32_t(idx);
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= np2; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t idx = uint32_t(tid); idx < np2; idx += 256u) {
                const uint32_t ixj = idx ^ j;
                if (ixj > idx) {
                    const bool desc = ((idx & k) == 0u);
                    const float vi = sh.sort.val[idx];
                    const float vj = sh.sort.val[ixj];
                    const int32_t ii = sh.sort.id[idx];
                    const int32_t ij = sh.sort.id[ixj];
                    const bool i_better = (vi > vj) || (vi == vj && ii < ij);
                    if (desc != i_better) {
                        sh.sort.val[idx] = vj;
                        sh.sort.id[idx]  = ij;
                        sh.sort.val[ixj] = vi;
                        sh.sort.id[ixj]  = ii;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        double cum = 0.0;
        uint32_t keep_n = 0u;
        for (; keep_n < total_cnt; ++keep_n) {
            cum += double(sh.sort.val[keep_n]);
            if (cum >= double(top_p)) {
                ++keep_n;
                break;
            }
        }
        if (keep_n == 0u) {
            keep_n = 1u;
            cum = double(sh.sort.val[0]);
        }
        const float renorm = float(1.0 / fmax(cum, 1e-30));
        s_keep_n = keep_n;
        s_renorm = renorm;

        const float u = ctrl->u_rand[step_idx];
        float acc = 0.0f;
        uint32_t chosen_j = keep_n - 1u;
        for (uint32_t j = 0u; j < keep_n; ++j) {
            acc += sh.sort.val[j] * renorm;
            if (u <= acc) {
                chosen_j = j;
                break;
            }
        }
        ctrl->kept_cnt[step_idx]       = keep_n;
        ctrl->out_drafts[step_idx]     = sh.sort.id[chosen_j];
        ctrl->out_q_selected[step_idx] = sh.sort.val[chosen_j] * renorm;
        ctrl->out_drafted_cnt          = step_idx + 1;
    }
    __syncthreads();

    const uint32_t keep_n = s_keep_n;
    const float renorm = s_renorm;
    for (uint32_t j = uint32_t(tid); j < keep_n; j += 256u) {
        ctrl->kept_id[step_idx][j] = sh.sort.id[j];
        ctrl->kept_q[step_idx][j]  = sh.sort.val[j] * renorm;
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence_system();
        __hip_atomic_store(&ctrl->ready_tok_seq[step_idx], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

__global__ __launch_bounds__(256)
void k_dspark_pack_conf_in(
    const DsparkGpuSampleSlot* __restrict__ ctrl,
    const uint16_t*            __restrict__ sub_bf16,
    const uint16_t*            __restrict__ markov_sum_bf16,
    uint16_t*                  __restrict__ conf_in_bf16)
{
    (void)ctrl;
    const int row = int(blockIdx.y);  // 0..4
    const int col = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (col >= 5376) return;
    uint16_t v = 0u;
    if (col < 5120) {
        v = sub_bf16[size_t(row) * 5120u + size_t(col)];
    } else {
        v = markov_sum_bf16[size_t(row) * 256u + size_t(col - 5120)];
    }
    conf_in_bf16[size_t(row) * 5376u + size_t(col)] = v;
}

}  // namespace

void launch_dspark_prepare_markov_emb(
    DsparkGpuSampleSlot* d_ctrl, const uint16_t* d_mapped_markov_embed_bf16,
    uint16_t* d_out_markov_emb_bf16, uint64_t seq, int step_idx, int eos_id, hipStream_t stream)
{
    hipLaunchKernelGGL(k_dspark_prepare_markov_emb, dim3(1), dim3(256), 0, stream,
                       d_ctrl, d_mapped_markov_embed_bf16, d_out_markov_emb_bf16, seq, step_idx, eos_id);
}

void launch_dspark_step_sample_tp3(
    DsparkGpuSampleSlot* d_ctrl, float* d_base_logits_row_f32, float* d_markov_logits_f32,
    uint64_t seq, int step_idx, int rank, int vocab_offset, int vocab_rows, int eos_id, hipStream_t stream)
{
    hipLaunchKernelGGL(k_dspark_step_sample_tp3, dim3(1), dim3(256), 0, stream,
                       d_ctrl, d_base_logits_row_f32, d_markov_logits_f32,
                       seq, step_idx, rank, vocab_offset, vocab_rows, eos_id);
}

void launch_dspark_pack_conf_in(
    const DsparkGpuSampleSlot* d_ctrl, const uint16_t* d_sub_bf16,
    const uint16_t* d_markov_sum_bf16, uint16_t* d_conf_in_bf16, hipStream_t stream)
{
    hipLaunchKernelGGL(k_dspark_pack_conf_in, dim3(21, 5), dim3(256), 0, stream,
                       d_ctrl, d_sub_bf16, d_markov_sum_bf16, d_conf_in_bf16);
}

}  // namespace dsv41
