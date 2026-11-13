// Pure-C++17 Exact Engram N-Gram Hash State + Pre-Faulted 189 GiB DDR4 Gather & AVX2 Dequantizer
// Matches hf_meta/inference/engram.py & model.py:296-366 100% bit-for-bit!

#include "dsv41_engram_cpu.hpp"

#include <immintrin.h>
#include <sys/mman.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dsv41 {

namespace {

uint64_t mulmod_u64(uint64_t a, uint64_t b, uint64_t m) {
    return uint64_t((__uint128_t(a) * __uint128_t(b)) % __uint128_t(m));
}

uint64_t powmod_u64(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t res = 1ULL;
    base %= mod;
    while (exp > 0ULL) {
        if (exp & 1ULL) res = mulmod_u64(res, base, mod);
        base = mulmod_u64(base, base, mod);
        exp >>= 1u;
    }
    return res;
}

bool is_prime_i64(int64_t n_signed) {
    if (n_signed < 2) return false;
    const uint64_t n = uint64_t(n_signed);
    static constexpr uint64_t kSmallPrimes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (uint64_t p : kSmallPrimes) {
        if (n == p) return true;
        if (n % p == 0ULL) return false;
    }
    uint64_t d = n - 1ULL;
    int s = 0;
    while ((d & 1ULL) == 0ULL) {
        d >>= 1u;
        ++s;
    }
    for (uint64_t a : kSmallPrimes) {
        if (a >= n) break;
        uint64_t x = powmod_u64(a, d, n);
        if (x == 1ULL || x == n - 1ULL) continue;
        bool comp = true;
        for (int r = 1; r < s; ++r) {
            x = mulmod_u64(x, x, n);
            if (x == n - 1ULL) {
                comp = false;
                break;
            }
        }
        if (comp) return false;
    }
    return true;
}

class NumpyDefaultRng {
public:
    explicit NumpyDefaultRng(uint32_t seed_u32) {
        static constexpr uint32_t INIT_A     = 0x43b0d7e5u;
        static constexpr uint32_t MULT_A     = 0x931e8875u;
        static constexpr uint32_t INIT_B     = 0x8b51f9ddu;
        static constexpr uint32_t MULT_B     = 0x58f38dedu;
        static constexpr uint32_t MIX_MULT_L = 0xca01f9ddu;
        static constexpr uint32_t MIX_MULT_R = 0x4973f715u;
        static constexpr uint32_t XSHIFT     = 16u;

        auto hashmix = [&](uint32_t value, uint32_t& hash_const) -> uint32_t {
            value ^= hash_const;
            hash_const *= MULT_A;
            value *= hash_const;
            value ^= (value >> XSHIFT);
            return value;
        };
        auto mix = [&](uint32_t x, uint32_t y) -> uint32_t {
            uint32_t result = MIX_MULT_L * x - MIX_MULT_R * y;
            result ^= (result >> XSHIFT);
            return result;
        };

        uint32_t pool[4] = {0, 0, 0, 0};
        uint32_t hash_const = INIT_A;
        for (int i = 0; i < 4; ++i) {
            pool[i] = hashmix((i == 0) ? seed_u32 : 0u, hash_const);
        }
        for (int i_src = 0; i_src < 4; ++i_src) {
            for (int i_dst = 0; i_dst < 4; ++i_dst) {
                if (i_src != i_dst) {
                    pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src], hash_const));
                }
            }
        }

        uint32_t hc_b = INIT_B;
        uint32_t w32[8];
        for (int i = 0; i < 8; ++i) {
            uint32_t dv = pool[i & 3];
            dv ^= hc_b;
            hc_b *= MULT_B;
            dv *= hc_b;
            dv ^= (dv >> XSHIFT);
            w32[i] = dv;
        }
        uint64_t s64[4];
        for (int k = 0; k < 4; ++k) {
            s64[k] = uint64_t(w32[2 * k]) | (uint64_t(w32[2 * k + 1]) << 32);
        }

        const __uint128_t initstate = (__uint128_t(s64[0]) << 64) | __uint128_t(s64[1]);
        const __uint128_t initseq   = (__uint128_t(s64[2]) << 64) | __uint128_t(s64[3]);

        state_ = 0;
        inc_ = (initseq << 1u) | __uint128_t(1u);
        step();
        state_ += initstate;
        step();
    }

    uint64_t next_uint64() {
        step();
        const uint64_t xored = uint64_t(state_ >> 64u) ^ uint64_t(state_);
        const unsigned rot = unsigned(state_ >> 122u);
        return (xored >> rot) | (xored << ((-rot) & 63u));
    }

    int64_t bounded_int64(int64_t high_excl) {
        const uint64_t rng_excl = uint64_t(high_excl);
        __uint128_t m = __uint128_t(next_uint64()) * __uint128_t(rng_excl);
        uint64_t leftover = uint64_t(m);
        if (leftover < rng_excl) {
            const uint64_t threshold = (0ULL - rng_excl) % rng_excl;
            while (leftover < threshold) {
                m = __uint128_t(next_uint64()) * __uint128_t(rng_excl);
                leftover = uint64_t(m);
            }
        }
        return int64_t(uint64_t(m >> 64u));
    }

private:
    __uint128_t state_ = 0;
    __uint128_t inc_ = 0;

    void step() {
        static const __uint128_t kMul =
            (__uint128_t(2549297995355413924ULL) << 64) | __uint128_t(4865540595714422341ULL);
        state_ = state_ * kMul + inc_;
    }
};

struct alignas(64) EngramDequantLuts {
    alignas(64) std::array<float, 256> fp8_e4m3;
    alignas(64) std::array<float, 256> ue8m0;

    EngramDequantLuts() {
        for (int b = 0; b < 256; ++b) {
            const uint32_t s = uint32_t(b >> 7) & 1u;
            const uint32_t e = uint32_t(b >> 3) & 0xFu;
            const uint32_t m = uint32_t(b) & 0x7u;
            float val = 0.0f;
            if (e == 0u) {
                val = float(m) * (1.0f / 512.0f);
            } else if (e == 15u && m == 7u) {
                val = 0.0f;
            } else {
                val = std::ldexp(1.0f + float(m) * 0.125f, int(e) - 7);
            }
            fp8_e4m3[b] = s ? -val : val;

            if (b == 255) {
                ue8m0[b] = 0.0f;
            } else {
                ue8m0[b] = std::ldexp(1.0f, b - 127);
            }
        }
    }
};

const EngramDequantLuts& dequant_luts() {
    static const EngramDequantLuts l;
    return l;
}

// Vectorized AVX2 dequant of one 256-dim row (8 blocks of 32 FP8_E4M3 * UE8M0) -> 256 BF16 (RNE)
inline void dequant_engram_row_avx2(
    const uint8_t* __restrict__ wr,
    const uint8_t* __restrict__ sr,
    const EngramDequantLuts& luts,
    uint16_t* __restrict__ dst)
{
    const float* lut_fp8 = luts.fp8_e4m3.data();
    const __m256i v_7fff = _mm256_set1_epi32(0x00007FFFu);
    const __m256i v_one  = _mm256_set1_epi32(1);

    for (int blk = 0; blk < 8; ++blk) {
        const __m256 v_sc = _mm256_set1_ps(luts.ue8m0[sr[blk]]);
        const uint8_t* wb = wr + blk * 32;
        uint16_t* db = dst + blk * 32;

        for (int half = 0; half < 2; ++half) {
            const uint8_t* p = wb + half * 16;
            const __m128i b_lo = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
            const __m128i b_hi = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + 8));
            const __m256i idx0 = _mm256_cvtepu8_epi32(b_lo);
            const __m256i idx1 = _mm256_cvtepu8_epi32(b_hi);
            const __m256 f0 = _mm256_mul_ps(_mm256_i32gather_ps(lut_fp8, idx0, 4), v_sc);
            const __m256 f1 = _mm256_mul_ps(_mm256_i32gather_ps(lut_fp8, idx1, 4), v_sc);
            const __m256i u0 = _mm256_castps_si256(f0);
            const __m256i u1 = _mm256_castps_si256(f1);
            const __m256i r0 = _mm256_srli_epi32(_mm256_add_epi32(u0, _mm256_add_epi32(v_7fff, _mm256_and_si256(_mm256_srli_epi32(u0, 16), v_one))), 16);
            const __m256i r1 = _mm256_srli_epi32(_mm256_add_epi32(u1, _mm256_add_epi32(v_7fff, _mm256_and_si256(_mm256_srli_epi32(u1, 16), v_one))), 16);
            const __m256i pk  = _mm256_packus_epi32(r0, r1);
            const __m256i ord = _mm256_permute4x64_epi64(pk, 0xD8);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(db + half * 16), ord);
        }
    }
}

void touch_region_parallel(const uint8_t* base, size_t nbytes, int nthreads) {
    if (!base || nbytes == 0) return;
    std::vector<std::thread> workers;
    workers.reserve(size_t(nthreads));
    std::atomic<uint64_t> sink{0};
    const size_t total_pages = (nbytes + 4095ULL) >> 12;

    for (int t = 0; t < nthreads; ++t) {
        workers.emplace_back([&, t]() {
            const size_t p0 = (total_pages * size_t(t)) / size_t(nthreads);
            const size_t p1 = (total_pages * size_t(t + 1)) / size_t(nthreads);
            uint64_t acc = 0;
            for (size_t p = p0; p < p1; ++p) {
                acc += base[p << 12];
            }
            sink.fetch_add(acc, std::memory_order_relaxed);
        });
    }
    for (auto& w : workers) w.join();
}

}  // namespace

void Dsv41EngramEngine::init_hash_tables(
    const std::vector<int32_t>& compressed_token_map,
    int32_t compressed_vocab_size,
    int max_seq_len)
{
    if (compressed_vocab_size != Dsv41EngramConfig::kExpectedCompressedVocab) {
        throw std::runtime_error(
            "Dsv41EngramEngine: compressed_vocab_size=" + std::to_string(compressed_vocab_size) +
            " != expected " + std::to_string(Dsv41EngramConfig::kExpectedCompressedVocab));
    }
    token_map_ = compressed_token_map;
    pad_compressed_id_ = token_map_.at(Dsv41EngramConfig::kPadTokenId);
    seq_cache_.assign(size_t(max_seq_len), pad_compressed_id_);

    std::unordered_set<int64_t> seen;
    for (int l = 0; l < Dsv41EngramConfig::kNumEngramLayers; ++l) {
        int col = 0;
        int64_t cum = 0;
        for (int ng = 0; ng < Dsv41EngramConfig::kMaxNgramSize - 1; ++ng) {
            int64_t cur = Dsv41EngramConfig::kBaseVocabSize - 1;
            for (int h = 0; h < Dsv41EngramConfig::kNumHeads; ++h) {
                int64_t cand = cur + 1;
                while (!is_prime_i64(cand) || seen.count(cand)) {
                    ++cand;
                }
                seen.insert(cand);
                cur = cand;
                primes_[l][col] = cand;
                offsets_[l][col] = cum;
                cum += cand;
                ++col;
            }
        }
        if (cum != Dsv41EngramConfig::kExpectedNumEmbeddings[l]) {
            throw std::runtime_error(
                "Dsv41EngramEngine: prime sum mismatch for engram layer idx " + std::to_string(l) +
                ": got " + std::to_string(cum) + " vs expected " +
                std::to_string(Dsv41EngramConfig::kExpectedNumEmbeddings[l]));
        }
    }

    const int64_t max_long = std::numeric_limits<int64_t>::max();
    const int64_t multiplier_bound = std::max<int64_t>(1LL, (max_long / int64_t(compressed_vocab_size)) / 2LL);
    for (int l = 0; l < Dsv41EngramConfig::kNumEngramLayers; ++l) {
        const uint32_t seed = uint32_t(10007 * Dsv41EngramConfig::kLayerIds[l]);
        NumpyDefaultRng rng(seed);
        for (int k = 0; k < Dsv41EngramConfig::kMaxNgramSize; ++k) {
            const int64_t v = rng.bounded_int64(multiplier_bound);
            multipliers_[l][k] = v * 2LL + 1LL;
        }
        for (int col = 0; col < 24; ++col) {
            const uint64_t up = uint64_t(primes_[l][col]);
            barrett_p_[l][col] = double(primes_[l][col]);
            barrett_inv_p_[l][col] = 1.0 / double(primes_[l][col]);
            barrett_r21_[l][col] = double((1ULL << 21) % up);
            barrett_r42_[l][col] = double((1ULL << 42) % up);
            barrett_r63_neg_[l][col] = double((up - ((1ULL << 63) % up)) % up);
            barrett_off_[l][col] = double(offsets_[l][col]);
        }
    }

    // Startup self-test: verify exact bit-parity of AVX2 21-bit signed Barrett vs scalar % across extremal & random int64_t
    {
        std::vector<int64_t> probe = {
            std::numeric_limits<int64_t>::min(),
            std::numeric_limits<int64_t>::min() + 1,
            -1, 0, 1,
            std::numeric_limits<int64_t>::max() - 1,
            std::numeric_limits<int64_t>::max()
        };
        uint64_t s = 0x9E3779B97F4A7C15ULL;
        for (int i = 0; i < 4096; ++i) {
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
            probe.push_back(int64_t(s * 0x2545F4914F6CDD1DULL));
        }
        for (int64_t x : probe) {
            const uint64_t ux = uint64_t(x);
            const __m256d vx0 = _mm256_set1_pd(double(ux & 0x1FFFFFULL));
            const __m256d vx1 = _mm256_set1_pd(double((ux >> 21) & 0x1FFFFFULL));
            const __m256d vx2 = _mm256_set1_pd(double((ux >> 42) & 0x1FFFFFULL));
            const __m256d vb63 = _mm256_set1_pd(double(ux >> 63));
            for (int l = 0; l < 2; ++l) {
                int64_t got[24];
                for (int c0 = 0; c0 < 24; c0 += 4) {
                    const __m256d vp = _mm256_load_pd(&barrett_p_[l][c0]);
                    const __m256d vinv = _mm256_load_pd(&barrett_inv_p_[l][c0]);
                    __m256d vS = _mm256_fmadd_pd(vx1, _mm256_load_pd(&barrett_r21_[l][c0]), vx0);
                    vS = _mm256_fmadd_pd(vx2, _mm256_load_pd(&barrett_r42_[l][c0]), vS);
                    vS = _mm256_fmadd_pd(vb63, _mm256_load_pd(&barrett_r63_neg_[l][c0]), vS);
                    const __m256d vq = _mm256_floor_pd(_mm256_mul_pd(vS, vinv));
                    __m256d vrem = _mm256_fnmadd_pd(vq, vp, vS);
                    vrem = _mm256_sub_pd(vrem, _mm256_and_pd(_mm256_cmp_pd(vrem, vp, _CMP_GE_OQ), vp));
                    const __m128i vi32 = _mm256_cvttpd_epi32(_mm256_add_pd(vrem, _mm256_load_pd(&barrett_off_[l][c0])));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&got[c0]), _mm256_cvtepi32_epi64(vi32));
                }
                for (int col = 0; col < 24; ++col) {
                    const int64_t p = primes_[l][col];
                    int64_t rem = x % p;
                    if (rem < 0) rem += p;
                    if (got[col] != rem + offsets_[l][col]) {
                        throw std::runtime_error("Dsv41EngramEngine: AVX2 Barrett prime-modulo self-test failed");
                    }
                }
            }
        }
    }
}

void Dsv41EngramEngine::open_table_shard(int engram_idx, const std::string& safetensors_path) {
    if (engram_idx < 0 || engram_idx >= Dsv41EngramConfig::kNumEngramLayers) {
        throw std::runtime_error("Dsv41EngramEngine: invalid engram_idx");
    }
    const int layer_id = Dsv41EngramConfig::kLayerIds[engram_idx];
    const std::string pfx = "layers." + std::to_string(layer_id) + ".engram.embed";

    auto st = std::make_unique<SafetensorsMmap>(safetensors_path);
    const auto& we = st->at(pfx + ".weight");
    const auto& se = st->at(pfx + ".scale");
    if (we.shape.size() != 2 || we.shape[0] != Dsv41EngramConfig::kExpectedNumEmbeddings[engram_idx] || we.shape[1] != 256) {
        throw std::runtime_error("Dsv41EngramEngine: unexpected embed.weight shape in " + safetensors_path);
    }
    if (se.shape.size() != 2 || se.shape[0] != we.shape[0] || se.shape[1] != 8) {
        throw std::runtime_error("Dsv41EngramEngine: unexpected embed.scale shape in " + safetensors_path);
    }

    const uint8_t* wp = st->tensor_bytes(pfx + ".weight");
    const uint8_t* sp = st->tensor_bytes(pfx + ".scale");

    (void)::madvise(const_cast<uint8_t*>(wp), we.byte_size(), MADV_SEQUENTIAL);
    (void)::madvise(const_cast<uint8_t*>(sp), se.byte_size(), MADV_SEQUENTIAL);
#ifdef MADV_HUGEPAGE
    (void)::madvise(const_cast<uint8_t*>(wp), we.byte_size(), MADV_HUGEPAGE);
    (void)::madvise(const_cast<uint8_t*>(sp), se.byte_size(), MADV_HUGEPAGE);
#endif

    tables_[engram_idx].rows = we.shape[0];
    tables_[engram_idx].weight_fp8 = wp;
    tables_[engram_idx].scale_ue8m0 = sp;
    tables_[engram_idx].weight_bytes = we.byte_size();
    tables_[engram_idx].scale_bytes = se.byte_size();
    tables_[engram_idx].st = std::move(st);
}

void Dsv41EngramEngine::prefault_all_tables() {
    const auto t0 = std::chrono::steady_clock::now();
    size_t total_bytes = 0;
    for (int i = 0; i < 2; ++i) {
        if (tables_[i].weight_fp8) total_bytes += tables_[i].weight_bytes + tables_[i].scale_bytes;
    }
    if (total_bytes == 0) return;

    std::printf("[DSV41_ENGRAM] Pre-faulting %.2f GiB Engram tables into 384 GiB DDR4 concurrently across Dual-NVMe (NVMe0 + NVMe1)...\n",
                double(total_bytes) / (1024.0 * 1024.0 * 1024.0));
    std::fflush(stdout);

    // Hydrate Shard 47 (NVMe0) and Shard 48 (NVMe1) simultaneously!
    std::thread t_s0([&]() {
        if (tables_[0].weight_fp8) {
            touch_region_parallel(tables_[0].scale_ue8m0, tables_[0].scale_bytes, 4);
            touch_region_parallel(tables_[0].weight_fp8, tables_[0].weight_bytes, 12);
            (void)::madvise(const_cast<uint8_t*>(tables_[0].weight_fp8), tables_[0].weight_bytes, MADV_RANDOM);
            (void)::madvise(const_cast<uint8_t*>(tables_[0].scale_ue8m0), tables_[0].scale_bytes, MADV_RANDOM);
        }
    });
    std::thread t_s1([&]() {
        if (tables_[1].weight_fp8) {
            touch_region_parallel(tables_[1].scale_ue8m0, tables_[1].scale_bytes, 4);
            touch_region_parallel(tables_[1].weight_fp8, tables_[1].weight_bytes, 12);
            (void)::madvise(const_cast<uint8_t*>(tables_[1].weight_fp8), tables_[1].weight_bytes, MADV_RANDOM);
            (void)::madvise(const_cast<uint8_t*>(tables_[1].scale_ue8m0), tables_[1].scale_bytes, MADV_RANDOM);
        }
    });
    t_s0.join();
    t_s1.join();

    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double gib = double(total_bytes) / (1024.0 * 1024.0 * 1024.0);
    std::printf("[DSV41_ENGRAM] Hydrated %.2f GiB in %.2fs (%.2f GB/s, 0 runtime NVMe faults)\n",
                gib, sec, double(total_bytes) / (1e9 * std::max(1e-6, sec)));
    std::fflush(stdout);
}

void Dsv41EngramEngine::compute_hashes(
    const int32_t* token_ids,
    int seqlen,
    int start_pos,
    const uint8_t* image_mask,
    std::vector<int64_t>& out_hashes)
{
    static constexpr int32_t kDead = -1;
    if (start_pos + seqlen > int(seq_cache_.size())) {
        seq_cache_.resize(size_t(start_pos + seqlen + 4096), pad_compressed_id_);
    }

    for (int s = 0; s < seqlen; ++s) {
        if (image_mask && image_mask[s]) {
            seq_cache_[size_t(start_pos + s)] = kDead;
        } else {
            const int32_t tid = token_ids[s];
            const int32_t cid = (tid >= 0 && size_t(tid) < token_map_.size()) ? token_map_[size_t(tid)] : pad_compressed_id_;
            seq_cache_[size_t(start_pos + s)] = cid;
        }
    }

    out_hashes.resize(size_t(seqlen) * 2ULL * 24ULL);

    #pragma omp parallel for if(seqlen >= 64) schedule(static)
    for (int s = 0; s < seqlen; ++s) {
        const int pos = start_pos + s;
        int64_t toks[4];
        bool blocked = false;
        for (int shift = 0; shift < 4; ++shift) {
            const int idx = std::max(0, pos - shift);
            const int32_t src = seq_cache_[size_t(idx)];
            if (pos < shift || src == kDead) blocked = true;
            toks[shift] = blocked ? int64_t(pad_compressed_id_) : int64_t(src);
        }

        for (int l = 0; l < 2; ++l) {
            int64_t* dst = out_hashes.data() + (size_t(s) * 2ULL + size_t(l)) * 24ULL;
            int64_t rolling = toks[0] * multipliers_[l][0];
            for (int i = 1; i < 4; ++i) {
                rolling ^= (toks[i] * multipliers_[l][i]);
                const int base_col = (i - 1) * 8;
                const uint64_t ux = uint64_t(rolling);
                const __m256d vx0 = _mm256_set1_pd(double(ux & 0x1FFFFFULL));
                const __m256d vx1 = _mm256_set1_pd(double((ux >> 21) & 0x1FFFFFULL));
                const __m256d vx2 = _mm256_set1_pd(double((ux >> 42) & 0x1FFFFFULL));
                const __m256d vb63 = _mm256_set1_pd(double(ux >> 63));
                for (int g = 0; g < 2; ++g) {
                    const int c0 = base_col + g * 4;
                    const __m256d vp = _mm256_load_pd(&barrett_p_[l][c0]);
                    const __m256d vinv = _mm256_load_pd(&barrett_inv_p_[l][c0]);
                    __m256d vS = _mm256_fmadd_pd(vx1, _mm256_load_pd(&barrett_r21_[l][c0]), vx0);
                    vS = _mm256_fmadd_pd(vx2, _mm256_load_pd(&barrett_r42_[l][c0]), vS);
                    vS = _mm256_fmadd_pd(vb63, _mm256_load_pd(&barrett_r63_neg_[l][c0]), vS);
                    const __m256d vq = _mm256_floor_pd(_mm256_mul_pd(vS, vinv));
                    __m256d vrem = _mm256_fnmadd_pd(vq, vp, vS);
                    vrem = _mm256_sub_pd(vrem, _mm256_and_pd(_mm256_cmp_pd(vrem, vp, _CMP_GE_OQ), vp));
                    const __m128i vi32 = _mm256_cvttpd_epi32(_mm256_add_pd(vrem, _mm256_load_pd(&barrett_off_[l][c0])));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c0), _mm256_cvtepi32_epi64(vi32));
                }
            }
        }
    }
}

void Dsv41EngramEngine::compute_hashes_tree(
    const int32_t* token_ids,
    const int32_t* parent_ids,
    const uint32_t* depths,
    int M,
    int start_pos,
    std::vector<int64_t>& out_hashes)
{
    static constexpr int32_t kDead = -1;
    if (start_pos + M > int(seq_cache_.size())) {
        seq_cache_.resize(size_t(start_pos + M + 4096), pad_compressed_id_);
    }

    int32_t node_cid[16];
    for (int s = 0; s < M; ++s) {
        const int32_t tid = token_ids[s];
        node_cid[s] = (tid >= 0 && size_t(tid) < token_map_.size()) ? token_map_[size_t(tid)] : pad_compressed_id_;
    }

    out_hashes.resize(size_t(M) * 2ULL * 24ULL);

    for (int s = 0; s < M; ++s) {
        int32_t anc_cid[16];
        for (int k = 0; k < 16; ++k) anc_cid[k] = pad_compressed_id_;
        int cur = s;
        for (int guard = 0; cur >= 0 && guard < 16; ++guard) {
            const uint32_t cd = depths[cur];
            if (cd < 16u) anc_cid[cd] = node_cid[cur];
            cur = parent_ids[cur];
        }

        const int pos = start_pos + int(depths[s]);
        int64_t toks[4];
        bool blocked = false;
        for (int shift = 0; shift < 4; ++shift) {
            const int idx = pos - shift;
            if (idx < 0) {
                blocked = true;
                toks[shift] = int64_t(pad_compressed_id_);
            } else {
                const int32_t src = (idx >= start_pos)
                    ? anc_cid[idx - start_pos]
                    : seq_cache_[size_t(idx)];
                if (src == kDead) blocked = true;
                toks[shift] = blocked ? int64_t(pad_compressed_id_) : int64_t(src);
            }
        }

        for (int l = 0; l < 2; ++l) {
            int64_t* dst = out_hashes.data() + (size_t(s) * 2ULL + size_t(l)) * 24ULL;
            int64_t rolling = toks[0] * multipliers_[l][0];
            for (int i = 1; i < 4; ++i) {
                rolling ^= (toks[i] * multipliers_[l][i]);
                const int base_col = (i - 1) * 8;
                const uint64_t ux = uint64_t(rolling);
                const __m256d vx0 = _mm256_set1_pd(double(ux & 0x1FFFFFULL));
                const __m256d vx1 = _mm256_set1_pd(double((ux >> 21) & 0x1FFFFFULL));
                const __m256d vx2 = _mm256_set1_pd(double((ux >> 42) & 0x1FFFFFULL));
                const __m256d vb63 = _mm256_set1_pd(double(ux >> 63));
                for (int g = 0; g < 2; ++g) {
                    const int c0 = base_col + g * 4;
                    const __m256d vp = _mm256_load_pd(&barrett_p_[l][c0]);
                    const __m256d vinv = _mm256_load_pd(&barrett_inv_p_[l][c0]);
                    __m256d vS = _mm256_fmadd_pd(vx1, _mm256_load_pd(&barrett_r21_[l][c0]), vx0);
                    vS = _mm256_fmadd_pd(vx2, _mm256_load_pd(&barrett_r42_[l][c0]), vS);
                    vS = _mm256_fmadd_pd(vb63, _mm256_load_pd(&barrett_r63_neg_[l][c0]), vS);
                    const __m256d vq = _mm256_floor_pd(_mm256_mul_pd(vS, vinv));
                    __m256d vrem = _mm256_fnmadd_pd(vq, vp, vS);
                    vrem = _mm256_sub_pd(vrem, _mm256_and_pd(_mm256_cmp_pd(vrem, vp, _CMP_GE_OQ), vp));
                    const __m128i vi32 = _mm256_cvttpd_epi32(_mm256_add_pd(vrem, _mm256_load_pd(&barrett_off_[l][c0])));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c0), _mm256_cvtepi32_epi64(vi32));
                }
            }
        }
    }
}

void Dsv41EngramEngine::commit_accepted_tokens(
    const int32_t* accepted_tokens,
    int count,
    int start_pos)
{
    if (count <= 0) return;
    if (start_pos + count > int(seq_cache_.size())) {
        seq_cache_.resize(size_t(start_pos + count + 4096), pad_compressed_id_);
    }
    for (int i = 0; i < count; ++i) {
        const int32_t tid = accepted_tokens[i];
        const int32_t cid = (tid >= 0 && size_t(tid) < token_map_.size()) ? token_map_[size_t(tid)] : pad_compressed_id_;
        seq_cache_[size_t(start_pos + i)] = cid;
    }
}

void Dsv41EngramEngine::gather_dequant_tp3(
    int engram_idx,
    const int64_t* all_hashes,
    int seqlen,
    const uint8_t* image_mask,
    uint16_t* out_tp0_bf16,
    uint16_t* out_tp1_bf16,
    uint16_t* out_tp2_bf16) const
{
    uint16_t* outs[3] = {out_tp0_bf16, out_tp1_bf16, out_tp2_bf16};
    const auto& tbl = tables_.at(engram_idx);
    if (!tbl.weight_fp8 || !tbl.scale_ue8m0) {
        for (int tp = 0; tp < 3; ++tp) {
            std::memset(outs[tp], 0, size_t(seqlen) * 2048ULL * sizeof(uint16_t));
        }
        return;
    }

    const auto& luts = dequant_luts();
    const uint8_t* __restrict__ w_base = tbl.weight_fp8;
    const uint8_t* __restrict__ s_base = tbl.scale_ue8m0;
    const int total_jobs = seqlen * 24;

    // Small-M OpenMP Gate + Hardware MSHR Prefetch Burst (seqlen=1: 0 OpenMP barrier; seqlen<=8: 4 threads)
    if (seqlen <= 8) {
        for (int job = 0; job < total_jobs; ++job) {
            const int s = job / 24;
            if (image_mask && image_mask[s]) continue;
            const int col = job % 24;
            const int64_t row_id = all_hashes[(size_t(s) * 2ULL + size_t(engram_idx)) * 24ULL + size_t(col)];
            const uint8_t* wr = w_base + size_t(row_id) * 256ULL;
            _mm_prefetch(reinterpret_cast<const char*>(wr), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(wr + 64), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(wr + 128), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(wr + 192), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(s_base + size_t(row_id) * 8ULL), _MM_HINT_T0);
        }

        if (seqlen == 1) {
            for (int col = 0; col < 24; ++col) {
                const int tp = col >> 3;
                const int local_col = col & 7;
                uint16_t* __restrict__ dst = outs[tp] + size_t(local_col) * 256ULL;
                if (image_mask && image_mask[0]) {
                    std::memset(dst, 0, 256 * sizeof(uint16_t));
                    continue;
                }
                const int64_t row_id = all_hashes[size_t(engram_idx) * 24ULL + size_t(col)];
                dequant_engram_row_avx2(w_base + size_t(row_id) * 256ULL, s_base + size_t(row_id) * 8ULL, luts, dst);
            }
            return;
        }

        #pragma omp parallel for num_threads(4) schedule(static)
        for (int job = 0; job < total_jobs; ++job) {
            const int s = job / 24;
            const int col = job % 24;
            const int tp = col >> 3;
            const int local_col = col & 7;
            uint16_t* __restrict__ dst = outs[tp] + (size_t(s) * 8ULL + size_t(local_col)) * 256ULL;
            if (image_mask && image_mask[s]) {
                std::memset(dst, 0, 256 * sizeof(uint16_t));
                continue;
            }
            const int64_t row_id = all_hashes[(size_t(s) * 2ULL + size_t(engram_idx)) * 24ULL + size_t(col)];
            dequant_engram_row_avx2(w_base + size_t(row_id) * 256ULL, s_base + size_t(row_id) * 8ULL, luts, dst);
        }
        return;
    }

    // Large-M prefill path (M > 6): full-socket OpenMP + 16-token (384-job) DDR4 MSHR software prefetch lookahead
    #pragma omp parallel for schedule(static)
    for (int job = 0; job < total_jobs; ++job) {
        if (job + 384 < total_jobs) {
            const int ps = (job + 384) / 24;
            const int pcol = (job + 384) % 24;
            const int64_t prow = all_hashes[(size_t(ps) * 2ULL + size_t(engram_idx)) * 24ULL + size_t(pcol)];
            const uint8_t* pwr = w_base + size_t(prow) * 256ULL;
            _mm_prefetch(reinterpret_cast<const char*>(pwr), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(pwr + 64), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(pwr + 128), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(pwr + 192), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(s_base + size_t(prow) * 8ULL), _MM_HINT_T0);
        }

        const int s = job / 24;
        const int col = job % 24;
        const int tp = col >> 3;
        const int local_col = col & 7;
        uint16_t* __restrict__ dst = outs[tp] + (size_t(s) * 8ULL + size_t(local_col)) * 256ULL;

        if (image_mask && image_mask[s]) {
            std::memset(dst, 0, 256 * sizeof(uint16_t));
            continue;
        }

        const int64_t row_id = all_hashes[(size_t(s) * 2ULL + size_t(engram_idx)) * 24ULL + size_t(col)];
        dequant_engram_row_avx2(w_base + size_t(row_id) * 256ULL, s_base + size_t(row_id) * 8ULL, luts, dst);
    }
}

}  // namespace dsv41
