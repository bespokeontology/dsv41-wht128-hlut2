#pragma once
// Modular I/O, HTTP Byte-Range Raw MXFP4 Fetcher, and MI50 rocBLAS + AVX2 Rotated SwiGLU / LDLQ Accelerator
// for Wave 8 Track 8C Expanded Hot-Expert RAREBAL-LDLQ Repack.
// Pure C++17 + OpenMP + AVX2 + HIP/rocBLAS, zero Python anywhere.

#include "../src/hlut2_pack.hpp"
#include "../src/dsv41_pack_format.hpp"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <fcntl.h>
#include <immintrin.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dsv41 {

struct CapturedExpertData {
    int block_id = 0;
    int eid = 0;
    bool pinned_p150 = false;
    int rows = 0;
    uint64_t router_hits = 0;
    std::vector<uint8_t> rare;
    std::vector<uint16_t> x_bf16;
};

inline CapturedExpertData load_captured_expert_bin(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("Cannot open captured bin: " + path);
    struct Hdr { uint32_t magic; int32_t bid; int32_t eid; int32_t rows; uint32_t pin; uint32_t pad; uint64_t hits; } hdr{};
    auto rd = [&](void* p, size_t n) {
        auto* b = static_cast<uint8_t*>(p);
        for (size_t o = 0; o < n;) {
            const ssize_t r = ::read(fd, b + o, n - o);
            if (r <= 0) { ::close(fd); throw std::runtime_error("Short read on " + path); }
            o += size_t(r);
        }
    };
    rd(&hdr, sizeof(hdr));
    if (hdr.magic != 0x5242414Cu || hdr.rows <= 0) {
        ::close(fd);
        throw std::runtime_error("Invalid RAREBAL capture header in " + path);
    }
    CapturedExpertData out;
    out.block_id = hdr.bid;
    out.eid = hdr.eid;
    out.pinned_p150 = (hdr.pin != 0u);
    out.rows = hdr.rows;
    out.router_hits = hdr.hits;
    out.rare.resize(size_t(hdr.rows));
    out.x_bf16.resize(size_t(hdr.rows) * DSV41_H);
    rd(out.rare.data(), out.rare.size());
    rd(out.x_bf16.data(), out.x_bf16.size() * sizeof(uint16_t));
    ::close(fd);
    return out;
}

struct RawExpertMxfp4 {
    std::vector<uint8_t> w1_w, w1_s, w3_w, w3_s, w2_w, w2_s;
};

class HfShardRangeFetcher {
public:
    explicit HfShardRangeFetcher(const std::string& cache_dir) : cache_dir_(cache_dir) {
        std::error_code ec;
        std::filesystem::create_directories(cache_dir_, ec);
    }

    RawExpertMxfp4 fetch_expert(int block_id, int eid) {
        const int shard_idx = (block_id < 40) ? (3 + block_id) : (44 + (block_id - 40));
        const ShardMeta& sm = get_shard_meta(shard_idx);
        const std::string pfx = (block_id < 40)
            ? ("layers." + std::to_string(block_id) + ".ffn.experts." + std::to_string(eid))
            : ("mtp." + std::to_string(block_id - 40) + ".ffn.experts." + std::to_string(eid));

        const std::string names[6] = {
            pfx + ".w1.weight", pfx + ".w1.scale",
            pfx + ".w3.weight", pfx + ".w3.scale",
            pfx + ".w2.weight", pfx + ".w2.scale"
        };
        std::pair<uint64_t, uint64_t> offs[6];
        uint64_t min_b = ~0ULL, max_e = 0ULL;
        for (int i = 0; i < 6; ++i) {
            offs[i] = parse_offsets(sm.header_json, names[i]);
            min_b = std::min(min_b, offs[i].first);
            max_e = std::max(max_e, offs[i].second);
        }

        RawExpertMxfp4 out;
        std::vector<uint8_t>* dsts[6] = {&out.w1_w, &out.w1_s, &out.w3_w, &out.w3_s, &out.w2_w, &out.w2_s};
        if (max_e > min_b && (max_e - min_b) <= (32ULL << 20)) {
            const auto span = curl_range(sm.cdn_url, sm.hf_url, 8ULL + sm.hdr_len + min_b, 8ULL + sm.hdr_len + max_e - 1ULL);
            for (int i = 0; i < 6; ++i) {
                const uint64_t rel = offs[i].first - min_b;
                const uint64_t len = offs[i].second - offs[i].first;
                dsts[i]->assign(span.begin() + rel, span.begin() + rel + len);
            }
        } else {
            // Single curl process reusing 1 HTTP/2 TLS connection across all 6 byte ranges!
            curl_multi_ranges_keepalive(cache_dir_, block_id, eid, sm.cdn_url, sm.hf_url, sm.hdr_len, offs, dsts);
        }
        return out;
    }

private:
    struct ShardMeta {
        uint64_t hdr_len = 0;
        std::string hf_url;
        std::string cdn_url;
        std::string header_json;
    };
    std::string cache_dir_;
    std::mutex mtx_;
    std::unordered_map<int, ShardMeta> meta_;

    static void curl_multi_ranges_keepalive(
        const std::string& cache_dir, int bid, int eid,
        const std::string& cdn, const std::string& hf, uint64_t hdr_len,
        const std::pair<uint64_t, uint64_t> offs[6],
        std::vector<uint8_t>* dsts[6])
    {
        static std::atomic<uint64_t> seq{0};
        const uint64_t tag = ++seq;
        std::string tmp_files[6];
        for (int i = 0; i < 6; ++i) {
            tmp_files[i] = cache_dir + "/mr_" + std::to_string(bid) + "_" + std::to_string(eid) +
                           "_" + std::to_string(tag) + "_" + std::to_string(i) + ".tmp";
        }
        auto cleanup_tmps = [&]() {
            std::error_code ec;
            for (int i = 0; i < 6; ++i) std::filesystem::remove(tmp_files[i], ec);
        };

        for (int att = 0; att < 8; ++att) {
            const std::string& u = (att == 0 && !cdn.empty()) ? cdn : hf;
            std::ostringstream cmd;
            cmd << "/usr/bin/curl -sSL --fail --http2 --connect-timeout 15 --speed-time 30 --speed-limit 65536";
            for (int i = 0; i < 6; ++i) {
                const uint64_t b0 = 8ULL + hdr_len + offs[i].first;
                const uint64_t b1 = 8ULL + hdr_len + offs[i].second - 1ULL;
                if (i > 0) cmd << " --next -sSL --fail --http2 --connect-timeout 15 --speed-time 30 --speed-limit 65536";
                cmd << " -r " << b0 << "-" << b1 << " -o \"" << tmp_files[i] << "\" \"" << u << "\"";
            }
            FILE* p = ::popen(cmd.str().c_str(), "r");
            if (!p) continue;
            const int rc = ::pclose(p);
            if (rc != 0) { cleanup_tmps(); continue; }

            bool all_ok = true;
            for (int i = 0; i < 6; ++i) {
                const uint64_t want = offs[i].second - offs[i].first;
                std::ifstream f(tmp_files[i], std::ios::binary);
                if (!f) { all_ok = false; break; }
                dsts[i]->resize(want);
                if (!f.read(reinterpret_cast<char*>(dsts[i]->data()), std::streamsize(want))) {
                    all_ok = false;
                    break;
                }
            }
            cleanup_tmps();
            if (all_ok) return;
        }
        cleanup_tmps();
        throw std::runtime_error("HTTP multi-range fetch failed on " + hf);
    }

    static std::vector<uint8_t> curl_range(const std::string& cdn, const std::string& hf, uint64_t b0, uint64_t b1) {
        const uint64_t want = b1 - b0 + 1ULL;
        for (int att = 0; att < 8; ++att) {
            const std::string& u = (att == 0 && !cdn.empty()) ? cdn : hf;
            std::ostringstream cmd;
            cmd << "/usr/bin/curl -sSL --fail --http2 --connect-timeout 15 --speed-time 30 --speed-limit 65536 -r "
                << b0 << "-" << b1 << " \"" << u << "\"";
            FILE* p = ::popen(cmd.str().c_str(), "r");
            if (!p) continue;
            std::vector<uint8_t> buf(want);
            size_t got = 0;
            while (got < want) {
                const size_t r = std::fread(buf.data() + got, 1, want - got, p);
                if (r == 0) break;
                got += r;
            }
            const int rc = ::pclose(p);
            if (rc == 0 && got == want) return buf;
        }
        throw std::runtime_error("HTTP range fetch failed on " + hf);
    }

    static std::string resolve_cdn(const std::string& hf_url) {
        std::string cmd = "/usr/bin/curl -sSL --fail --connect-timeout 15 -r 0-0 -o /dev/null -w \"%{url_effective}\" \"" + hf_url + "\"";
        FILE* p = ::popen(cmd.c_str(), "r");
        if (!p) return hf_url;
        char b[4096]; std::string out;
        while (std::fgets(b, sizeof(b), p)) out += b;
        const int rc = ::pclose(p);
        while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) out.pop_back();
        return (rc == 0 && out.rfind("https://", 0) == 0) ? out : hf_url;
    }

    static std::pair<uint64_t, uint64_t> parse_offsets(const std::string& js, const std::string& key) {
        const std::string needle = "\"" + key + "\":";
        const size_t p0 = js.find(needle);
        if (p0 == std::string::npos) throw std::runtime_error("Missing tensor in HF header: " + key);
        const size_t pd = js.find("\"data_offsets\":[", p0);
        if (pd == std::string::npos || pd - p0 > 256) throw std::runtime_error("Missing data_offsets for " + key);
        size_t cur = pd + 16;
        uint64_t a = 0, b = 0;
        while (cur < js.size() && js[cur] >= '0' && js[cur] <= '9') a = a * 10ULL + uint64_t(js[cur++] - '0');
        if (cur < js.size() && js[cur] == ',') ++cur;
        while (cur < js.size() && js[cur] >= '0' && js[cur] <= '9') b = b * 10ULL + uint64_t(js[cur++] - '0');
        if (b <= a) throw std::runtime_error("Invalid data_offsets for " + key);
        return {a, b};
    }

    const ShardMeta& get_shard_meta(int shard_idx) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            auto it = meta_.find(shard_idx);
            if (it != meta_.end()) return it->second;
        }

        char sname[64];
        std::snprintf(sname, sizeof(sname), "model-%05d-of-00048.safetensors", shard_idx);
        ShardMeta sm;
        sm.hf_url = std::string("https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/") + sname;
        sm.cdn_url = resolve_cdn(sm.hf_url);

        const std::string hdr_cache = cache_dir_ + "/" + sname + ".hdr";
        std::ifstream cf(hdr_cache, std::ios::binary);
        if (cf.is_open()) {
            sm.header_json.assign((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
            sm.hdr_len = sm.header_json.size();
        } else {
            const auto b8 = curl_range(sm.cdn_url, sm.hf_url, 0, 7);
            std::memcpy(&sm.hdr_len, b8.data(), 8);
            const auto hb = curl_range(sm.cdn_url, sm.hf_url, 8, 8 + sm.hdr_len - 1);
            sm.header_json.assign(reinterpret_cast<const char*>(hb.data()), hb.size());
            const std::string tmp_hdr = hdr_cache + ".tmp." + std::to_string(reinterpret_cast<uintptr_t>(&sm));
            std::ofstream of(tmp_hdr, std::ios::binary | std::ios::trunc);
            of.write(sm.header_json.data(), std::streamsize(sm.header_json.size()));
            of.close();
            std::error_code ec;
            std::filesystem::rename(tmp_hdr, hdr_cache, ec);
        }

        std::lock_guard<std::mutex> lk(mtx_);
        auto it = meta_.find(shard_idx);
        if (it != meta_.end()) return it->second;
        return meta_.emplace(shard_idx, std::move(sm)).first->second;
    }
};

// Multi-threaded OpenMP + AVX2 fused MXFP4-dequant + WHT128 + 8-iter Lloyd-Max HLUT2 seed quantizer
inline Hlut2QuantMatrix dequant_mxfp4_and_quantize_hlut2_mt(
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

    #pragma omp parallel for schedule(static)
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
        #pragma omp parallel for reduction(+:num0,den0,num1,den1) schedule(static)
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

    const int words_per_row = K / 16;
    q.codes_rm.assign(size_t(N) * words_per_row, 0u);
    for (int outer = 0; outer < 3; ++outer) {
        const float tau = 0.5f * (c0 + c1);
        double n0 = 0.0, d0 = 0.0, n1 = 0.0, d1 = 0.0;
        #pragma omp parallel for reduction(+:n0,d0,n1,d1) schedule(static)
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

struct DualReceiptMetrics {
    double cos_all = 0.0;
    double snr_db = 0.0;
    double mse_all = 0.0;
    double mse_common = 0.0;
    double mse_rare = 0.0;
    int rare_cnt = 0;
};

inline void matmul_nt_avx2(const float* A, const float* B, float* C, int M, int N, int K) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m0 = 0; m0 < M; m0 += 4) {
        for (int n0 = 0; n0 < N; n0 += 2) {
            const int mr = std::min(4, M - m0);
            const int nr = std::min(2, N - n0);
            if (mr == 4 && nr == 2) {
                const float *a0 = A + size_t(m0) * K, *a1 = a0 + K, *a2 = a1 + K, *a3 = a2 + K;
                const float *b0 = B + size_t(n0) * K, *b1 = b0 + K;
                __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
                __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
                __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
                __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
                for (int k = 0; k < K; k += 8) {
                    const __m256 vb0 = _mm256_loadu_ps(b0 + k), vb1 = _mm256_loadu_ps(b1 + k);
                    const __m256 va0 = _mm256_loadu_ps(a0 + k), va1 = _mm256_loadu_ps(a1 + k);
                    const __m256 va2 = _mm256_loadu_ps(a2 + k), va3 = _mm256_loadu_ps(a3 + k);
                    c00 = _mm256_fmadd_ps(va0, vb0, c00); c01 = _mm256_fmadd_ps(va0, vb1, c01);
                    c10 = _mm256_fmadd_ps(va1, vb0, c10); c11 = _mm256_fmadd_ps(va1, vb1, c11);
                    c20 = _mm256_fmadd_ps(va2, vb0, c20); c21 = _mm256_fmadd_ps(va2, vb1, c21);
                    c30 = _mm256_fmadd_ps(va3, vb0, c30); c31 = _mm256_fmadd_ps(va3, vb1, c31);
                }
                auto hsum = [](__m256 v) {
                    alignas(32) float t[8];
                    _mm256_store_ps(t, v);
                    return (t[0] + t[1]) + (t[2] + t[3]) + (t[4] + t[5]) + (t[6] + t[7]);
                };
                C[size_t(m0 + 0) * N + n0 + 0] = hsum(c00); C[size_t(m0 + 0) * N + n0 + 1] = hsum(c01);
                C[size_t(m0 + 1) * N + n0 + 0] = hsum(c10); C[size_t(m0 + 1) * N + n0 + 1] = hsum(c11);
                C[size_t(m0 + 2) * N + n0 + 0] = hsum(c20); C[size_t(m0 + 2) * N + n0 + 1] = hsum(c21);
                C[size_t(m0 + 3) * N + n0 + 0] = hsum(c30); C[size_t(m0 + 3) * N + n0 + 1] = hsum(c31);
            } else {
                for (int dm = 0; dm < mr; ++dm) {
                    for (int dn = 0; dn < nr; ++dn) {
                        const float* a = A + size_t(m0 + dm) * K;
                        const float* b = B + size_t(n0 + dn) * K;
                        __m256 acc = _mm256_setzero_ps();
                        for (int k = 0; k < K; k += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc);
                        alignas(32) float t[8]; _mm256_store_ps(t, acc);
                        C[size_t(m0 + dm) * N + n0 + dn] = (t[0] + t[1]) + (t[2] + t[3]) + (t[4] + t[5]) + (t[6] + t[7]);
                    }
                }
            }
        }
    }
}

// MI50 rocBLAS + Explicit 16-wide AVX2 FP64 Cholesky + Left-Looking 128-Block LDLQ Accelerator
class GpuRarebalAccel {
public:
    static constexpr size_t kMaxM = 16384;
    static constexpr size_t kMaxD = 5120;

    explicit GpuRarebalAccel(int device_id = 0) {
        if (hipSetDevice(device_id) != hipSuccess) return;
        if (hipStreamCreate(&stream_) != hipSuccess) return;
        if (rocblas_create_handle(&handle_) != rocblas_status_success) {
            (void)hipStreamDestroy(stream_);
            stream_ = nullptr;
            return;
        }
        (void)rocblas_set_stream(handle_, stream_);
        if (hipMalloc(&d_a_,    kMaxM * kMaxD * sizeof(float)) != hipSuccess ||
            hipMalloc(&d_b_,    kMaxD * kMaxD * sizeof(float)) != hipSuccess ||
            hipMalloc(&d_c_,    kMaxM * kMaxD * sizeof(float)) != hipSuccess ||
            hipMalloc(&d_rup_,  kMaxD * kMaxD * sizeof(float)) != hipSuccess ||
            hipMalloc(&d_err_,  kMaxD * kMaxD * sizeof(float)) != hipSuccess ||
            hipMalloc(&d_wpan_, kMaxD * 128ULL * sizeof(float)) != hipSuccess ||
            hipMalloc(&d_hrev_, kMaxD * kMaxD * sizeof(double)) != hipSuccess)
        {
            cleanup();
            return;
        }
        ok_ = true;
    }

    ~GpuRarebalAccel() { cleanup(); }
    bool ok() const noexcept { return ok_; }

    // Row-major C[M, N] = A[M, K] * B[N, K]^T
    void matmul_nt(const float* A, const float* B, float* C, int M, int N, int K) {
        if (!ok_ || size_t(M) > kMaxM || size_t(N) > kMaxD || size_t(K) > kMaxD) {
            matmul_nt_avx2(A, B, C, M, N, K);
            return;
        }
        (void)hipMemcpyAsync(d_a_, A, size_t(M) * K * sizeof(float), hipMemcpyHostToDevice, stream_);
        (void)hipMemcpyAsync(d_b_, B, size_t(N) * K * sizeof(float), hipMemcpyHostToDevice, stream_);
        const float one = 1.0f, zero = 0.0f;
        (void)rocblas_sgemm(handle_, rocblas_operation_transpose, rocblas_operation_none,
                            N, M, K, &one, d_b_, K, d_a_, K, &zero, d_c_, N);
        (void)hipMemcpyAsync(C, d_c_, size_t(M) * N * sizeof(float), hipMemcpyDeviceToHost, stream_);
        (void)hipStreamSynchronize(stream_);
    }

    std::vector<float> build_cholesky_rup(
        const float* X_cal_rm, int M_cal, int K,
        const std::vector<uint64_t>& signs,
        const uint8_t* rare_mask,
        Hlut2RarebalStats* out_stats)
    {
        if (!ok_ || M_cal <= 2 * K || size_t(M_cal) > kMaxM || size_t(K) > kMaxD) {
            return build_rarebal_cholesky_rup(X_cal_rm, M_cal, K, signs, true, rare_mask, out_stats);
        }

        std::vector<float> x_rot(size_t(M_cal) * K);
        wht128_rows(X_cal_rm, x_rot.data(), M_cal, K, signs);

        double tr_common = 0.0, tr_rare = 0.0;
        int m_common = 0, m_rare = 0;
        #pragma omp parallel for reduction(+:tr_common,tr_rare,m_common,m_rare) schedule(static)
        for (int m = 0; m < M_cal; ++m) {
            const float* xm = x_rot.data() + size_t(m) * K;
            double row_sq = 0.0;
            for (int k = 0; k < K; ++k) row_sq += double(xm[k]) * double(xm[k]);
            if (rare_mask && rare_mask[m] != 0u) {
                tr_rare += row_sq;
                ++m_rare;
            } else {
                tr_common += row_sq;
                ++m_common;
            }
        }

        double lambda = 1.0;
        if (m_common > 0 && m_rare > 0 && tr_common > 0.0 && tr_rare > 0.0) {
            lambda = tr_common / std::max(1e-12, tr_rare);
        }

        std::vector<float> x_com(size_t(m_common) * K), x_rare(size_t(m_rare) * K);
        {
            int ic = 0, ir = 0;
            for (int m = 0; m < M_cal; ++m) {
                const float* src = x_rot.data() + size_t(m) * K;
                if (rare_mask && rare_mask[m] != 0u) {
                    std::memcpy(x_rare.data() + size_t(ir++) * K, src, size_t(K) * sizeof(float));
                } else {
                    std::memcpy(x_com.data() + size_t(ic++) * K, src, size_t(K) * sizeof(float));
                }
            }
        }

        const float inv_mcal = float(1.0 / double(M_cal));
        bool gram_init = false;
        if (m_common > 0) {
            (void)hipMemcpyAsync(d_a_, x_com.data(), size_t(m_common) * K * sizeof(float), hipMemcpyHostToDevice, stream_);
            const float alpha = inv_mcal, beta = 0.0f;
            (void)rocblas_sgemm(handle_, rocblas_operation_none, rocblas_operation_transpose,
                                K, K, m_common, &alpha, d_a_, K, d_a_, K, &beta, d_b_, K);
            gram_init = true;
        }
        if (m_rare > 0) {
            (void)hipMemcpyAsync(d_a_, x_rare.data(), size_t(m_rare) * K * sizeof(float), hipMemcpyHostToDevice, stream_);
            const float alpha = float(lambda * double(inv_mcal));
            const float beta = gram_init ? 1.0f : 0.0f;
            (void)rocblas_sgemm(handle_, rocblas_operation_none, rocblas_operation_transpose,
                                K, K, m_rare, &alpha, d_a_, K, d_a_, K, &beta, d_b_, K);
        }

        std::vector<float> h_gram(size_t(K) * K);
        (void)hipMemcpyAsync(h_gram.data(), d_b_, size_t(K) * K * sizeof(float), hipMemcpyDeviceToHost, stream_);
        (void)hipStreamSynchronize(stream_);

        double diag_mean = 0.0;
        for (int i = 0; i < K; ++i) diag_mean += double(h_gram[size_t(i) * K + i]);
        diag_mean /= double(K);
        const double damp = 0.015 * std::max(1e-12, diag_mean);

        if (out_stats) {
            out_stats->m_cal = M_cal;
            out_stats->m_common = m_common;
            out_stats->m_rare = m_rare;
            out_stats->used_starvation_fallback = false;
            out_stats->tr_common = tr_common;
            out_stats->tr_rare = tr_rare;
            out_stats->lambda = lambda;
            out_stats->d_bar = diag_mean;
        }

        // Blocked B=128 Order-Reversed FP64 Cholesky L_rev L_rev^T = P H P^T on Vega 20 (gfx906 1:2 FP64 rocblas_dgemm)!
        std::vector<double> H_rev(size_t(K) * K, 0.0);
        #pragma omp parallel for schedule(static)
        for (int ir = 0; ir < K; ++ir) {
            const int i_src = K - 1 - ir;
            for (int jr = 0; jr < K; ++jr) {
                const int j_src = K - 1 - jr;
                const double sym = 0.5 * (double(h_gram[size_t(i_src) * K + j_src]) +
                                          double(h_gram[size_t(j_src) * K + i_src]));
                H_rev[size_t(ir) * K + jr] = sym + ((ir == jr) ? damp : 0.0);
            }
        }
        h_gram.clear();
        h_gram.shrink_to_fit();

        (void)hipMemcpyAsync(d_hrev_, H_rev.data(), H_rev.size() * sizeof(double), hipMemcpyHostToDevice, stream_);
        (void)hipStreamSynchronize(stream_);

        constexpr int kBlk = 128;
        const int num_blk = K / kBlk;
        std::vector<double> h_pan(size_t(K) * kBlk);
        const double neg_one = -1.0, pos_one = 1.0;

        for (int b = 0; b < num_blk; ++b) {
            const int j0 = b * kBlk;
            const int rem = K - j0;
            (void)hipMemcpy2DAsync(h_pan.data(), kBlk * sizeof(double),
                                   d_hrev_ + size_t(j0) * K + j0, size_t(K) * sizeof(double),
                                   kBlk * sizeof(double), size_t(rem),
                                   hipMemcpyDeviceToHost, stream_);
            (void)hipStreamSynchronize(stream_);

            for (int jj = 0; jj < kBlk; ++jj) {
                double* rj = h_pan.data() + size_t(jj) * kBlk;
                double sq = 0.0;
                for (int m = 0; m < jj; ++m) sq += rj[m] * rj[m];
                const double ljj = std::sqrt(std::max(1e-18, rj[jj] - sq));
                rj[jj] = ljj;
                const double inv_ljj = 1.0 / ljj;
                for (int r = jj + 1; r < kBlk; ++r) {
                    double* rr = h_pan.data() + size_t(r) * kBlk;
                    double dot = 0.0;
                    for (int m = 0; m < jj; ++m) dot += rr[m] * rj[m];
                    rr[jj] = (rr[jj] - dot) * inv_ljj;
                }
                for (int r = 0; r < jj; ++r) h_pan[size_t(r) * kBlk + jj] = 0.0;
            }

            if (rem > kBlk) {
                #pragma omp parallel for schedule(static)
                for (int r = kBlk; r < rem; ++r) {
                    double* rr = h_pan.data() + size_t(r) * kBlk;
                    for (int jj = 0; jj < kBlk; ++jj) {
                        const double* rj = h_pan.data() + size_t(jj) * kBlk;
                        double dot = 0.0;
                        for (int m = 0; m < jj; ++m) dot += rr[m] * rj[m];
                        rr[jj] = (rr[jj] - dot) / rj[jj];
                    }
                }
            }

            (void)hipMemcpy2DAsync(d_hrev_ + size_t(j0) * K + j0, size_t(K) * sizeof(double),
                                   h_pan.data(), kBlk * sizeof(double),
                                   kBlk * sizeof(double), size_t(rem),
                                   hipMemcpyHostToDevice, stream_);

            const int rem2 = rem - kBlk;
            if (rem2 > 0) {
                const int j1 = j0 + kBlk;
                const double* d_l21 = d_hrev_ + size_t(j1) * K + j0;
                double* d_h22 = d_hrev_ + size_t(j1) * K + j1;
                (void)rocblas_dgemm(handle_, rocblas_operation_transpose, rocblas_operation_none,
                                    rem2, rem2, kBlk, &neg_one, d_l21, K, d_l21, K, &pos_one, d_h22, K);
            }
        }

        (void)hipMemcpyAsync(H_rev.data(), d_hrev_, H_rev.size() * sizeof(double), hipMemcpyDeviceToHost, stream_);
        (void)hipStreamSynchronize(stream_);

        std::vector<float> R_up(size_t(K) * K, 0.0f);
        #pragma omp parallel for schedule(static)
        for (int col = 0; col < K; ++col) {
            const int ir = K - 1 - col;
            float* r_row = R_up.data() + size_t(col) * K;
            r_row[col] = 1.0f;
            for (int k = col + 1; k < K; ++k) {
                const int jr = K - 1 - k;
                r_row[k] = float(H_rev[size_t(ir) * K + jr] / H_rev[size_t(jr) * K + jr]);
            }
        }
        return R_up;
    }

    void upload_rup(const std::vector<float>& R_up, int K) {
        if (!ok_ || R_up.size() != size_t(K) * K) return;
        (void)hipMemcpyAsync(d_rup_, R_up.data(), R_up.size() * sizeof(float), hipMemcpyHostToDevice, stream_);
        (void)hipStreamSynchronize(stream_);
    }

    // Exact Left-Looking 128-Column Block-Causal LDLQ using pre-uploaded d_rup_ on MI50!
    Hlut2QuantMatrix apply_ldlq_preuploaded_rup(
        const Hlut2QuantMatrix& q_seed,
        const float* w_rot,
        const std::vector<float>& R_up)
    {
        if (!ok_ || size_t(q_seed.N) > kMaxD || size_t(q_seed.K) > kMaxD) {
            return apply_hlut2_ldlq_with_rup(q_seed, w_rot, R_up);
        }
        Hlut2QuantMatrix q = q_seed;
        const int N = q.N;
        const int K = q.K;
        if (R_up.size() != size_t(K) * K) return q;
        const int G = K / 128;

        std::vector<float> sigma_f32(size_t(N) * G);
        for (size_t i = 0; i < sigma_f32.size(); ++i) sigma_f32[i] = fp16_to_f32(q.scales_fp16[i]);

        const float c0 = q.c0, c1 = q.c1;
        const float dc = c1 - c0;
        const float tau = 0.5f * (c0 + c1);
        const int words_per_row = K / 16;
        q.codes_rm.assign(size_t(N) * words_per_row, 0u);

        std::vector<float> h_wpan(size_t(N) * 128);
        std::vector<float> h_epan(size_t(N) * 128);
        const float one = 1.0f;

        for (int g = 0; g < G; ++g) {
            const int col0 = g * 128;
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < N; ++r) {
                std::memcpy(h_wpan.data() + size_t(r) * 128, w_rot + size_t(r) * K + col0, 128 * sizeof(float));
            }
            if (col0 > 0) {
                (void)hipMemcpyAsync(d_wpan_, h_wpan.data(), h_wpan.size() * sizeof(float), hipMemcpyHostToDevice, stream_);
                (void)rocblas_sgemm(handle_, rocblas_operation_none, rocblas_operation_none,
                                    128, N, col0, &one, d_rup_ + col0, K, d_err_, K, &one, d_wpan_, 128);
                (void)hipMemcpyAsync(h_wpan.data(), d_wpan_, h_wpan.size() * sizeof(float), hipMemcpyDeviceToHost, stream_);
                (void)hipStreamSynchronize(stream_);
            }

            #pragma omp parallel for schedule(static)
            for (int r = 0; r < N; ++r) {
                float* wp = h_wpan.data() + size_t(r) * 128;
                float* ep = h_epan.data() + size_t(r) * 128;
                uint32_t* cr = q.codes_rm.data() + size_t(r) * words_per_row;
                const float sig = sigma_f32[size_t(r) * G + g];
                const float thr = tau * sig;
                const float* wr_orig = w_rot + size_t(r) * K + col0;

                for (int j = 0; j < 128; ++j) {
                    const int col = col0 + j;
                    const float v = wp[j];
                    const uint32_t sign_bit = std::signbit(v) ? 2u : 0u;
                    const uint32_t mag_bit = (std::fabs(v) >= thr) ? 1u : 0u;
                    cr[col >> 4] |= ((sign_bit | mag_bit) << (2 * (col & 15)));

                    const float mag_val = (c0 + float(mag_bit) * dc) * sig;
                    const float qv = sign_bit ? -mag_val : mag_val;
                    const float err = wr_orig[j] - qv;
                    ep[j] = err;
                    const float* rrow = R_up.data() + size_t(col) * K + col0;
                    for (int jj = j + 1; jj < 128; ++jj) {
                        wp[jj] += err * rrow[jj];
                    }
                }
            }

            if (g + 1 < G) {
                (void)hipMemcpy2DAsync(d_err_ + col0, size_t(K) * sizeof(float),
                                       h_epan.data(), 128 * sizeof(float),
                                       128 * sizeof(float), size_t(N),
                                       hipMemcpyHostToDevice, stream_);
            }
        }
        (void)hipStreamSynchronize(stream_);
        return q;
    }

private:
    bool ok_ = false;
    hipStream_t stream_ = nullptr;
    rocblas_handle handle_ = nullptr;
    float *d_a_ = nullptr, *d_b_ = nullptr, *d_c_ = nullptr;
    float *d_rup_ = nullptr, *d_err_ = nullptr, *d_wpan_ = nullptr;
    double *d_hrev_ = nullptr;

    void cleanup() {
        if (d_hrev_) { (void)hipFree(d_hrev_); d_hrev_ = nullptr; }
        if (d_wpan_) { (void)hipFree(d_wpan_); d_wpan_ = nullptr; }
        if (d_err_)  { (void)hipFree(d_err_);  d_err_  = nullptr; }
        if (d_rup_)  { (void)hipFree(d_rup_);  d_rup_  = nullptr; }
        if (d_c_)    { (void)hipFree(d_c_);    d_c_    = nullptr; }
        if (d_b_)    { (void)hipFree(d_b_);    d_b_    = nullptr; }
        if (d_a_)    { (void)hipFree(d_a_);    d_a_    = nullptr; }
        if (handle_) { (void)rocblas_destroy_handle(handle_); handle_ = nullptr; }
        if (stream_) { (void)hipStreamDestroy(stream_); stream_ = nullptr; }
        ok_ = false;
    }
};

inline DualReceiptMetrics eval_rot_swiglu(
    const float* X_rot, const uint8_t* rare_mask, int M,
    const float* w1_rot, const float* w3_rot, const float* w2_rot,
    const std::vector<uint64_t>& signs_i,
    const std::vector<float>* ref_y,
    std::vector<float>& out_y,
    std::vector<float>* out_z = nullptr,
    GpuRarebalAccel* gpu = nullptr)
{
    std::vector<float> g(size_t(M) * DSV41_I), u(size_t(M) * DSV41_I), z(size_t(M) * DSV41_I), zr(size_t(M) * DSV41_I);
    if (gpu && gpu->ok()) {
        gpu->matmul_nt(X_rot, w1_rot, g.data(), M, DSV41_I, DSV41_H);
        gpu->matmul_nt(X_rot, w3_rot, u.data(), M, DSV41_I, DSV41_H);
    } else {
        matmul_nt_avx2(X_rot, w1_rot, g.data(), M, DSV41_I, DSV41_H);
        matmul_nt_avx2(X_rot, w3_rot, u.data(), M, DSV41_I, DSV41_H);
    }
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < z.size(); ++i) z[i] = dsv41_swiglu_scalar(g[i], u[i]);
    if (out_z) *out_z = z;
    wht128_rows(z.data(), zr.data(), M, DSV41_I, signs_i);
    out_y.resize(size_t(M) * DSV41_H);
    if (gpu && gpu->ok()) {
        gpu->matmul_nt(zr.data(), w2_rot, out_y.data(), M, DSV41_H, DSV41_I);
    } else {
        matmul_nt_avx2(zr.data(), w2_rot, out_y.data(), M, DSV41_H, DSV41_I);
    }

    DualReceiptMetrics dm{};
    if (!ref_y) return dm;
    double dot = 0.0, nr = 0.0, nq = 0.0, ea = 0.0, ec = 0.0, er = 0.0;
    size_t cc = 0, cr = 0;
    for (int m = 0; m < M; ++m) {
        const bool is_r = (rare_mask[m] != 0u);
        if (is_r) ++dm.rare_cnt;
        for (int j = 0; j < DSV41_H; ++j) {
            const double r = (*ref_y)[size_t(m) * DSV41_H + j];
            const double q = out_y[size_t(m) * DSV41_H + j];
            const double d2 = (r - q) * (r - q);
            dot += r * q; nr += r * r; nq += q * q; ea += d2;
            if (is_r) { er += d2; ++cr; } else { ec += d2; ++cc; }
        }
    }
    dm.cos_all    = (nr > 0.0 && nq > 0.0) ? (dot / (std::sqrt(nr) * std::sqrt(nq))) : 0.0;
    dm.snr_db     = 10.0 * std::log10(std::max(1e-30, nr) / std::max(1e-30, ea));
    dm.mse_all    = ea / double(size_t(M) * DSV41_H);
    dm.mse_common = ec / double(std::max<size_t>(1, cc));
    dm.mse_rare   = (cr > 0) ? (er / double(cr)) : dm.mse_all;
    return dm;
}

}  // namespace dsv41
