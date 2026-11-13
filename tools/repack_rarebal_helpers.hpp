#pragma once
// Modular I/O, HTTP Byte-Range Raw MXFP4 Fetcher, and AVX2 Rotated SwiGLU Evaluator for Wave 6B RAREBAL-LDLQ Repack
// Pure C++17, zero Python anywhere.

#include "../src/hlut2_pack.hpp"
#include "../src/dsv41_pack_format.hpp"

#include <fcntl.h>
#include <immintrin.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
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
            for (int i = 0; i < 6; ++i) {
                *dsts[i] = curl_range(sm.cdn_url, sm.hf_url, 8ULL + sm.hdr_len + offs[i].first, 8ULL + sm.hdr_len + offs[i].second - 1ULL);
            }
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

    static std::vector<uint8_t> curl_range(const std::string& cdn, const std::string& hf, uint64_t b0, uint64_t b1) {
        const uint64_t want = b1 - b0 + 1ULL;
        for (int att = 0; att < 8; ++att) {
            const std::string& u = (att == 0 && !cdn.empty()) ? cdn : hf;
            std::ostringstream cmd;
            cmd << "/usr/bin/curl -sSL --fail --connect-timeout 15 --speed-time 30 --speed-limit 65536 -r "
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
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = meta_.find(shard_idx);
        if (it != meta_.end()) return it->second;

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
            std::ofstream of(hdr_cache, std::ios::binary | std::ios::trunc);
            of.write(sm.header_json.data(), std::streamsize(sm.header_json.size()));
        }
        return meta_.emplace(shard_idx, std::move(sm)).first->second;
    }
};

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

inline DualReceiptMetrics eval_rot_swiglu(
    const float* X_rot, const uint8_t* rare_mask, int M,
    const float* w1_rot, const float* w3_rot, const float* w2_rot,
    const std::vector<uint64_t>& signs_i,
    const std::vector<float>* ref_y,
    std::vector<float>& out_y,
    std::vector<float>* out_z = nullptr)
{
    std::vector<float> g(size_t(M) * DSV41_I), u(size_t(M) * DSV41_I), z(size_t(M) * DSV41_I), zr(size_t(M) * DSV41_I);
    matmul_nt_avx2(X_rot, w1_rot, g.data(), M, DSV41_I, DSV41_H);
    matmul_nt_avx2(X_rot, w3_rot, u.data(), M, DSV41_I, DSV41_H);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < z.size(); ++i) z[i] = dsv41_swiglu_scalar(g[i], u[i]);
    if (out_z) *out_z = z;
    wht128_rows(z.data(), zr.data(), M, DSV41_I, signs_i);
    out_y.resize(size_t(M) * DSV41_H);
    matmul_nt_avx2(zr.data(), w2_rot, out_y.data(), M, DSV41_H, DSV41_I);

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
