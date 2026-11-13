#pragma once
// Pure-C++17 Streaming Safetensors Writer & TP3/TP2 HLUT2 Partition Spec for DeepSeek-V4.1-Flash-1017
// Zero Python anywhere. Produces 64-byte-aligned .safetensors shards directly mmap-able by SafetensorsMmap.

#include <fcntl.h>
#include <immintrin.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef DSV41_HLUT2_PACK_INCLUDED
#include "hlut2_pack.hpp"
#endif
#include "safetensors_mmap.hpp"

namespace dsv41 {

// Exact heterogeneous topology constants:
// - 3x AMD Instinct MI50 (96 GiB HBM2):
//     * Vocab (129,280): (43104, 43104, 43072) rows across tp = (0, 1, 2)
//     * Attention o_groups (8 groups, 64 heads of 512): (3, 3, 2) groups = (24, 24, 16) heads across tp = (0, 1, 2)
//     * Indexer heads (32 heads of 128): (11, 11, 10) heads across tp = (0, 1, 2)
//     * Shared expert intermediate (2304): (768, 768, 768) across tp = (0, 1, 2)
//     * Routed experts:
//         - Backbone (layers 0..39, 384 experts): experts [0..215] (216) on MI50 TP3 (I=768/card)
//         - DSpark MTP (mtp 0..2, 128 experts):   experts [0..127] (128) on MI50 TP3 (I=768/card)
// - 2x Tenstorrent Blackhole P150A (64 GiB GDDR6 + 360 MB L1):
//     * Backbone routed experts [216..383] (168/layer x 40 layers = 6,720 experts), TP2 (I=1152/card, 272 B/tile)
struct Dsv41Tp3Split {
    static constexpr int vocab_rows(int tp) { return (tp < 2) ? 43104 : 43072; }
    static constexpr int vocab_offset(int tp) { return (tp == 0) ? 0 : (tp == 1 ? 43104 : 86208); }

    static constexpr int o_groups(int tp) { return (tp < 2) ? 3 : 2; }
    static constexpr int o_group_offset(int tp) { return (tp == 0) ? 0 : (tp == 1 ? 3 : 6); }

    static constexpr int attn_heads(int tp) { return o_groups(tp) * 8; }            // 24, 24, 16
    static constexpr int attn_head_offset(int tp) { return o_group_offset(tp) * 8; } // 0, 24, 48

    static constexpr int idx_heads(int tp) { return (tp < 2) ? 11 : 10; }
    static constexpr int idx_head_offset(int tp) { return (tp == 0) ? 0 : (tp == 1 ? 11 : 22); }

    static constexpr int moe_inter_tp3 = 768;   // 2304 / 3 = 6 * 128 = 24 * 32
    static constexpr int moe_inter_tp2 = 1152;  // 2304 / 2 = 9 * 128 = 36 * 32

    static constexpr int mi50_routed_backbone = 216;
    static constexpr int p150_routed_backbone = 168;
    static constexpr int mi50_routed_mtp = 128;
};

inline uint64_t layer_wht_seed_h(int global_layer_id) {
    return 0x44535634315F4800ULL ^ (uint64_t(global_layer_id + 1) * 0x9E3779B97F4A7C15ULL);
}
inline uint64_t layer_wht_seed_i(int global_layer_id) {
    return 0x44535634315F4900ULL ^ (uint64_t(global_layer_id + 1) * 0xBF58476D1CE4E5B9ULL);
}

// Streaming 64-byte-aligned Safetensors writer (buffers metadata, writes tensor payloads directly to disk fd!)
class StreamingSafetensorsWriter {
public:
    struct PendingEntry {
        std::string name;
        std::string dtype;
        std::vector<int64_t> shape;
        const void* ptr = nullptr;
        uint64_t nbytes = 0;
        uint64_t begin_off = 0;
        uint64_t end_off = 0;
    };

    void add_borrowed(
        const std::string& name,
        const std::string& dtype,
        const std::vector<int64_t>& shape,
        const void* data,
        uint64_t nbytes)
    {
        PendingEntry e;
        e.name = name;
        e.dtype = dtype;
        e.shape = shape;
        e.ptr = data;
        e.nbytes = nbytes;
        entries_.push_back(std::move(e));
    }

    void add_owned(
        const std::string& name,
        const std::string& dtype,
        const std::vector<int64_t>& shape,
        std::vector<uint8_t>&& buf)
    {
        const uint64_t nb = buf.size();
        owned_.push_back(std::move(buf));
        add_borrowed(name, dtype, shape, owned_.back().data(), nb);
    }

    void write_atomic(const std::string& final_path) {
        uint64_t cur = 0;
        for (auto& e : entries_) {
            cur = (cur + 63ULL) & ~63ULL;
            e.begin_off = cur;
            e.end_off = cur + e.nbytes;
            cur = e.end_off;
        }

        std::ostringstream js;
        js << "{\"__metadata__\":{\"format\":\"dsv41_hlut2_v1\"}";
        for (const auto& e : entries_) {
            js << ",\"" << e.name << "\":{\"dtype\":\"" << e.dtype << "\",\"shape\":[";
            for (size_t i = 0; i < e.shape.size(); ++i) {
                if (i) js << ",";
                js << e.shape[i];
            }
            js << "],\"data_offsets\":[" << e.begin_off << "," << e.end_off << "]}";
        }
        js << "}";
        std::string hdr = js.str();
        // Pad JSON header so (8 + hdr.size()) % 64 == 0 -> every tensor payload starts at 64-byte file alignment!
        while ((8ULL + hdr.size()) % 64ULL != 0ULL) hdr.push_back(' ');
        const uint64_t hdr_len = hdr.size();

        const std::string tmp_path = final_path + ".tmp";
        int fd = ::open(tmp_path.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_CLOEXEC, 0664);
        if (fd < 0) throw std::runtime_error("StreamingSafetensorsWriter: open failed for " + tmp_path);

        auto write_all = [&](const void* p, size_t n) {
            const uint8_t* b = static_cast<const uint8_t*>(p);
            size_t off = 0;
            while (off < n) {
                const ssize_t w = ::write(fd, b + off, n - off);
                if (w <= 0) {
                    ::close(fd);
                    ::unlink(tmp_path.c_str());
                    throw std::runtime_error("StreamingSafetensorsWriter: write error on " + tmp_path);
                }
                off += size_t(w);
            }
        };

        write_all(&hdr_len, 8);
        write_all(hdr.data(), hdr.size());

        static const uint8_t zeros[64] = {0};
        uint64_t written_payload = 0;
        for (const auto& e : entries_) {
            if (e.begin_off > written_payload) {
                write_all(zeros, size_t(e.begin_off - written_payload));
                written_payload = e.begin_off;
            }
            if (e.nbytes > 0) {
                write_all(e.ptr, size_t(e.nbytes));
                written_payload += e.nbytes;
            }
        }
        if (::fsync(fd) != 0) {
            ::close(fd);
            ::unlink(tmp_path.c_str());
            throw std::runtime_error("StreamingSafetensorsWriter: fsync failed for " + tmp_path);
        }
        ::close(fd);

        // Verify written file via SafetensorsMmap before atomic rename!
        {
            SafetensorsMmap verify(tmp_path);
            if (verify.entries().size() != entries_.size()) {
                ::unlink(tmp_path.c_str());
                throw std::runtime_error("StreamingSafetensorsWriter: verification entry count mismatch");
            }
        }

        if (::rename(tmp_path.c_str(), final_path.c_str()) != 0) {
            ::unlink(tmp_path.c_str());
            throw std::runtime_error("StreamingSafetensorsWriter: rename failed to " + final_path);
        }
    }

private:
    std::vector<PendingEntry> entries_;
    std::vector<std::vector<uint8_t>> owned_;
};

// Slice Hlut2QuantMatrix along rows [n_off .. n_off + n_len)
inline Hlut2QuantMatrix slice_hlut2_rows(const Hlut2QuantMatrix& q, int n_off, int n_len) {
    Hlut2QuantMatrix out;
    out.N = n_len;
    out.K = q.K;
    out.c0 = q.c0;
    out.c1 = q.c1;
    const int kw = q.K / 16;
    const int G = q.K / 128;
    out.codes_rm.assign(
        q.codes_rm.begin() + size_t(n_off) * kw,
        q.codes_rm.begin() + size_t(n_off + n_len) * kw);
    out.scales_fp16.assign(
        q.scales_fp16.begin() + size_t(n_off) * G,
        q.scales_fp16.begin() + size_t(n_off + n_len) * G);
    return out;
}

// Slice Hlut2QuantMatrix along reduction cols [k_off .. k_off + k_len) (k_off & k_len multiples of 128)
inline Hlut2QuantMatrix slice_hlut2_cols(const Hlut2QuantMatrix& q, int k_off, int k_len) {
    Hlut2QuantMatrix out;
    out.N = q.N;
    out.K = k_len;
    out.c0 = q.c0;
    out.c1 = q.c1;
    const int kw_in = q.K / 16, kw_out = k_len / 16, kw_off = k_off / 16;
    const int g_in = q.K / 128, g_out = k_len / 128, g_off = k_off / 128;
    out.codes_rm.resize(size_t(q.N) * kw_out);
    out.scales_fp16.resize(size_t(q.N) * g_out);
    for (int r = 0; r < q.N; ++r) {
        std::memcpy(
            out.codes_rm.data() + size_t(r) * kw_out,
            q.codes_rm.data() + size_t(r) * kw_in + kw_off,
            size_t(kw_out) * sizeof(uint32_t));
        std::memcpy(
            out.scales_fp16.data() + size_t(r) * g_out,
            q.scales_fp16.data() + size_t(r) * g_in + g_off,
            size_t(g_out) * sizeof(uint16_t));
    }
    return out;
}

// Fast single-thread (caller-parallelized across experts!) AVX2 + F16C WHT128 + HLUT2 quantizer
inline void avx2_wht128_block(float* v, const uint64_t* sign_bits_block2) {
    for (int i = 0; i < 64; ++i) {
        if ((sign_bits_block2[0] >> i) & 1ULL) v[i] = -v[i];
        if ((sign_bits_block2[1] >> i) & 1ULL) v[64 + i] = -v[64 + i];
    }
    // Stages len = 1, 2, 4 scalar/compiler-vectorized inside 8-float lanes
    for (int len = 1; len < 8; len <<= 1) {
        for (int i = 0; i < 128; i += (len << 1)) {
            for (int j = 0; j < len; ++j) {
                const float a = v[i + j];
                const float b = v[i + len + j];
                v[i + j] = a + b;
                v[i + len + j] = a - b;
            }
        }
    }
#if defined(__AVX2__)
    // Stages len = 8, 16, 32, 64 in 8-wide __m256 registers!
    for (int len = 8; len < 128; len <<= 1) {
        for (int i = 0; i < 128; i += (len << 1)) {
            for (int j = 0; j < len; j += 8) {
                const __m256 a = _mm256_loadu_ps(v + i + j);
                const __m256 b = _mm256_loadu_ps(v + i + len + j);
                _mm256_storeu_ps(v + i + j, _mm256_add_ps(a, b));
                _mm256_storeu_ps(v + i + len + j, _mm256_sub_ps(a, b));
            }
        }
    }
    const __m256 s = _mm256_set1_ps(0.08838834764831844655f);
    for (int i = 0; i < 128; i += 8) {
        _mm256_storeu_ps(v + i, _mm256_mul_ps(_mm256_loadu_ps(v + i), s));
    }
#else
    for (int len = 8; len < 128; len <<= 1) {
        for (int i = 0; i < 128; i += (len << 1)) {
            for (int j = 0; j < len; ++j) {
                const float a = v[i + j], b = v[i + len + j];
                v[i + j] = a + b;
                v[i + len + j] = a - b;
            }
        }
    }
    for (int i = 0; i < 128; ++i) v[i] *= 0.08838834764831844655f;
#endif
}

// Single-expert fused MXFP4-dequant + AVX2-WHT128 + 8-iter Lloyd-Max HLUT2 quantizer (thread-local, zero OpenMP nesting!)
inline Hlut2QuantMatrix dequant_mxfp4_and_quantize_hlut2_st(
    const uint8_t* w_mxfp4,
    const uint8_t* s_ue8m0,
    int N,
    int K,
    const std::vector<uint64_t>& signs,
    std::vector<float>& scratch_w_rot,
    std::vector<float>& scratch_sigma)
{
    static constexpr float kE2M1Lut[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };

    Hlut2QuantMatrix q;
    q.N = N;
    q.K = K;
    const int G = K / 128;
    const int half_cols = K / 2;
    const int s_cols = K / 32;

    scratch_w_rot.resize(size_t(N) * K);
    scratch_sigma.resize(size_t(N) * G);
    q.scales_fp16.resize(size_t(N) * G);

    for (int r = 0; r < N; ++r) {
        const uint8_t* wr = w_mxfp4 + size_t(r) * half_cols;
        const uint8_t* sr = s_ue8m0 + size_t(r) * s_cols;
        float* dr = scratch_w_rot.data() + size_t(r) * K;
        for (int sb = 0; sb < s_cols; ++sb) {
            const uint8_t e8 = sr[sb];
            const uint32_t scale_bits = (e8 == 0) ? 0x00400000u : (uint32_t(e8) << 23);
            float scale = 0.0f;
            std::memcpy(&scale, &scale_bits, 4);
            const uint8_t* wb = wr + sb * 16;
            float* db = dr + sb * 32;
            for (int b = 0; b < 16; ++b) {
                const uint8_t byte = wb[b];
                db[2 * b + 0] = kE2M1Lut[byte & 0x0Fu] * scale;
                db[2 * b + 1] = kE2M1Lut[byte >> 4] * scale;
            }
        }
        for (int g = 0; g < G; ++g) {
            float* wg = dr + g * 128;
            avx2_wht128_block(wg, &signs[size_t(g) * 2]);
            double sq = 0.0;
            for (int j = 0; j < 128; ++j) sq += double(wg[j]) * double(wg[j]);
            const float rms = float(std::sqrt(sq * (1.0 / 128.0) + 1e-30));
            const uint16_t h = f32_to_fp16_rne(rms);
            q.scales_fp16[size_t(r) * G + g] = h;
            scratch_sigma[size_t(r) * G + g] = fp16_to_f32(h);
        }
    }

    float c0 = 0.4528f, c1 = 1.5104f;
    for (int it = 0; it < 8; ++it) {
        const float tau = 0.5f * (c0 + c1);
        double num0 = 0.0, den0 = 0.0, num1 = 0.0, den1 = 0.0;
        for (int r = 0; r < N; ++r) {
            const float* wr = scratch_w_rot.data() + size_t(r) * K;
            for (int g = 0; g < G; ++g) {
                const float sig = scratch_sigma[size_t(r) * G + g];
                if (sig <= 0.0f) continue;
                const double w2 = double(sig) * double(sig);
                const float inv_s = 1.0f / sig;
                const float* wg = wr + g * 128;
                for (int j = 0; j < 128; ++j) {
                    const float z = std::fabs(wg[j]) * inv_s;
                    if (z < tau) { num0 += w2 * double(z); den0 += w2; }
                    else         { num1 += w2 * double(z); den1 += w2; }
                }
            }
        }
        if (den0 > 0.0) c0 = float(num0 / den0);
        if (den1 > 0.0) c1 = float(num1 / den1);
        if (c1 <= c0 + 1e-4f) c1 = c0 + 0.5f;
    }
    // Alternating coordinate-descent refinement of per-group FP16 scales & (c0, c1) centroids
    const int words_per_row = K / 16;
    q.codes_rm.assign(size_t(N) * words_per_row, 0u);
    for (int outer = 0; outer < 3; ++outer) {
        const float tau = 0.5f * (c0 + c1);
        double n0 = 0.0, d0 = 0.0, n1 = 0.0, d1 = 0.0;
        for (int r = 0; r < N; ++r) {
            const float* wr = scratch_w_rot.data() + size_t(r) * K;
            uint32_t* cr = q.codes_rm.data() + size_t(r) * words_per_row;
            for (int g = 0; g < G; ++g) {
                const float* wg = wr + g * 128;
                float sig = scratch_sigma[size_t(r) * G + g];
                uint8_t mag[128];
                for (int it = 0; it < 2; ++it) {
                    const float thr = tau * sig;
                    double uw = 0.0, uu = 0.0;
                    for (int j = 0; j < 128; ++j) {
                        const float a = std::fabs(wg[j]);
                        const uint8_t m = (a >= thr) ? 1u : 0u;
                        mag[j] = m;
                        const float u = m ? c1 : c0;
                        uw += double(u) * double(a);
                        uu += double(u) * double(u);
                    }
                    if (uu > 1e-30) {
                        const uint16_t sh = f32_to_fp16_rne(float(uw / uu));
                        sig = fp16_to_f32(sh);
                    }
                }
                const uint16_t sh = f32_to_fp16_rne(sig);
                q.scales_fp16[size_t(r) * G + g] = sh;
                sig = fp16_to_f32(sh);
                scratch_sigma[size_t(r) * G + g] = sig;

                if (outer == 2) {
                    for (int w = 0; w < 8; ++w) {
                        uint32_t pack = 0u;
                        for (int m = 0; m < 16; ++m) {
                            const int j = w * 16 + m;
                            const uint32_t sb = std::signbit(wg[j]) ? 2u : 0u;
                            pack |= ((sb | uint32_t(mag[j])) << (2 * m));
                        }
                        cr[g * 8 + w] = pack;
                    }
                } else if (sig > 0.0f) {
                    const double s2 = double(sig) * double(sig);
                    for (int j = 0; j < 128; ++j) {
                        const float a = std::fabs(wg[j]);
                        if (mag[j] == 0) { n0 += double(sig) * double(a); d0 += s2; }
                        else             { n1 += double(sig) * double(a); d1 += s2; }
                    }
                }
            }
        }
        if (outer < 2) {
            if (d0 > 0.0) c0 = float(n0 / d0);
            if (d1 > 0.0) c1 = float(n1 / d1);
            if (c1 <= c0 + 1e-4f) c1 = c0 + 0.5f;
        }
    }
    q.c0 = c0;
    q.c1 = c1;
    return q;
}

// Exact MI50 (TP3) & Blackhole P150A (TP2, FP16 scales, 272 B/tile = 4352 B/page) slot packers
inline void pack_hlut2_mi50_slot_st(
    const Hlut2QuantMatrix& q,
    uint32_t* dst_codes,   // [N/32, K/16, 32]
    uint16_t* dst_scales,  // [N/32, K/128, 32]
    float* dst_centroids)  // [2]
{
    dst_centroids[0] = q.c0;
    dst_centroids[1] = q.c1;
    const int nb = q.N / 32;
    const int kw = q.K / 16;
    const int G = q.K / 128;
    for (int rb = 0; rb < nb; ++rb) {
        for (int r = 0; r < 32; ++r) {
            const int row = rb * 32 + r;
            for (int w = 0; w < kw; ++w) {
                dst_codes[(size_t(rb) * kw + w) * 32 + r] = q.codes_rm[size_t(row) * kw + w];
            }
            for (int g = 0; g < G; ++g) {
                dst_scales[(size_t(rb) * G + g) * 32 + r] = q.scales_fp16[size_t(row) * G + g];
            }
        }
    }
}

inline void pack_hlut2_p150_slot_st(
    const Hlut2QuantMatrix& q,
    int n_col_offset,
    int N_shard,
    uint8_t* dst_pages,    // [(N_shard/128)*(K/128) * 4352]
    float* dst_centroids)  // [2]
{
    dst_centroids[0] = q.c0;
    dst_centroids[1] = q.c1;
    const int kt4 = q.K / 128;
    const int nt4 = N_shard / 128;
    const int G = q.K / 128;
    const int kw = q.K / 16;

    for (int nb4 = 0; nb4 < nt4; ++nb4) {
        for (int kb4 = 0; kb4 < kt4; ++kb4) {
            const int sb = nb4 * kt4 + kb4;
            uint8_t* page_ptr = dst_pages + size_t(sb) * 4352ULL;
            uint32_t* dst_codes = reinterpret_cast<uint32_t*>(page_ptr);
            uint16_t* dst_scales = reinterpret_cast<uint16_t*>(page_ptr + 4096);

            for (int nt_sub = 0; nt_sub < 4; ++nt_sub) {
                for (int j = 0; j < 32; ++j) {
                    const int n_global = n_col_offset + (nb4 * 4 + nt_sub) * 32 + j;
                    dst_scales[nt_sub * 32 + j] = q.scales_fp16[size_t(n_global) * G + kb4];
                }
            }
            std::memset(dst_codes, 0, 4096);
            for (int i = 0; i < 32; ++i) {
                for (int j = 0; j < 32; ++j) {
                    uint32_t word = 0u;
                    for (int kt_sub = 0; kt_sub < 4; ++kt_sub) {
                        const int k_global = (kb4 * 4 + kt_sub) * 32 + i;
                        const int w_idx = k_global >> 4;
                        const int bit_shift = 2 * (k_global & 15);
                        for (int nt_sub = 0; nt_sub < 4; ++nt_sub) {
                            const int n_global = n_col_offset + (nb4 * 4 + nt_sub) * 32 + j;
                            const uint32_t code = (q.codes_rm[size_t(n_global) * kw + w_idx] >> bit_shift) & 3u;
                            const int t = kt_sub * 4 + nt_sub;
                            word |= (code << (2 * t));
                        }
                    }
                    dst_codes[tensix_tile_offset(i, j)] = word;
                }
            }
        }
    }
}

}  // namespace dsv41
