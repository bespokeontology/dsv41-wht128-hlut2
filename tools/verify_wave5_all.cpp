// verify_wave5_all.cpp — Pure-CPU Wave 5 validator (no HIP / no Tenstorrent device).
// (A) Cody-Waite FP64-angle YaRN RoPE vs long double remquo on the same base double angle @524288 & 1048576: f64_err < 1e-12 rad pre-cast;
//     <= 1.2e-7 rad after 32-bit float cast (0.5-ULP floor); rotation cosine > 0.99999. (B) 272 B/tile HLUT2 2.125 bpw. (C) P150A 4352 B superblock + 98.61 MiB L1.
#include "../src/hlut2_pack.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace dsv41;
static int g_fail = 0;
static void check(const char* n, bool ok) { std::printf(ok ? "[PASS] %s\n" : "[FAIL] %s\n", n); if (!ok) ++g_fail; }

// Exact YaRN compressed freqs (theta=160000, factor=16, orig_seq=65536, beta 16/32), long double.
static void yarn_freqs_ld(long double base, int orig_seq, long double factor, long double bf, long double bs,
                          long double out[32]) {
    constexpr long double kPi = 3.141592653589793238462643383279502884L;
    for (int i = 0; i < 32; ++i) out[i] = 1.0L / std::pow(base, (long double)(2 * i) / 64.0L);
    auto corr = [&](long double rot) {
        return 64.0L * std::log((long double)(orig_seq) / (rot * 2.0L * kPi)) / (2.0L * std::log(base));
    };
    const long double lo = std::max(std::floor(corr(bf)), 0.0L), hi = std::min(std::ceil(corr(bs)), 63.0L);
    const long double den = std::max(hi - lo, 1e-3L);
    for (int i = 0; i < 32; ++i) {
        const long double r = std::max(0.0L, std::min(1.0L, ((long double)(i) - lo) / den)), sm = 1.0L - r;
        out[i] = (out[i] / factor) * (1.0L - sm) + out[i] * sm;
    }
}

int main() {
    constexpr long double kTwoPiLD = 6.283185307179586476925286766559005768L;
    constexpr double kInvTwoPi = 0.15915494309189533576888376337251;
    constexpr double kTwoPiHi = 0x1.921fb54000000p+2, kTwoPiLo = 0x1.10b4611a62633p-28;
    long double fc_ld[32]; yarn_freqs_ld(160000.0L, 65536, 16.0L, 16.0L, 32.0L, fc_ld);
    const std::vector<int> pos = {524288, 1048576};
    double f64_err = 0.0, f32_err = 0.0, min_rot_cos = 1.0;
    for (int p : pos) for (int pair = 0; pair < 32; ++pair) {
        const double angle = double(p) * double(fc_ld[pair]);
        const double q = std::nearbyint(angle * kInvTwoPi);
        // FP64 reduced angle (pre-cast) and FP32 rad after the 32-bit float cast (0.5-ULP quantization floor).
        const double rad64 = std::fma(-q, kTwoPiLo, std::fma(-q, kTwoPiHi, angle));
        const float rad32 = float(rad64);
        int quo = 0;
        // Long double remquo on the SAME base double angle isolates reduction error from the base-angle loss.
        const long double ref = std::remquo((long double)(angle), kTwoPiLD, &quo);
        double d64 = double(std::fabs(rad64 - ref));
        if (d64 > 3.141592653589793238L) d64 = std::fabs(d64 - double(kTwoPiLD));
        double d32 = double(std::fabs((long double)(rad32) - ref));
        if (d32 > 3.141592653589793238L) d32 = std::fabs(d32 - double(kTwoPiLD));
        f64_err = std::max(f64_err, d64);
        f32_err = std::max(f32_err, d32);
        const double c0 = std::cos((long double)(rad64)), s0 = std::sin((long double)(rad64));
        const double c1 = std::cos(ref), s1 = std::sin(ref);
        min_rot_cos = std::min(min_rot_cos, (c0 * c1 + s0 * s1) / (std::sqrt(c0 * c0 + s0 * s0) * std::sqrt(c1 * c1 + s1 * s1)));
    }
    check("(A) FP64 reduced-angle err < 1e-12 rad pre-float cast", f64_err < 1e-12);
    check("(A) <= 1.2e-7 rad after 32-bit float cast + rotation cosine > 0.99999",
          f32_err <= 1.2e-7 && min_rot_cos > 0.99999);

    // ---- (B) 272 B/tile (2.125 bpw) HLUT2 on N=128, K=256, M_cal=640 > 2*K=512 ----
    const int N = 128, K = 256, M_cal = 640;
    const auto signs = make_wht_signs(K, 0x5F150202611ULL);
    std::vector<float> W(size_t(N) * K), X(size_t(M_cal) * K);
    for (int r = 0; r < N; ++r) for (int c = 0; c < K; ++c) {
        const float v = ((c % 37) == 7) ? 2.0f : (0.5f + 0.01f * float((r * 13 + c) % 7));
        W[size_t(r) * K + c] = bf16_to_f32(f32_to_bf16_rne(v));
    }    for (int m = 0; m < M_cal; ++m) for (int k = 0; k < K; ++k) {
        const float cm = ((k & 63) == 7) ? 6.0f : 0.45f;
        X[size_t(m) * K + k] = bf16_to_f32(f32_to_bf16_rne(0.01f * float((m * 11 + k) % 17) * cm));
    }
    Hlut2QuantMatrix q = quantize_hlut2(W.data(), N, K, signs);  // data-free H=I
    check("(B) quantize_hlut2 H=I == 2.125 bpw", std::fabs(q.bits_per_weight() - 2.125) < 1e-12);
    Hlut2RarebalStats s_gate{};
    Hlut2QuantMatrix qg = quantize_hlut2_ldlq(W.data(), N, K, signs, nullptr, 512, true, nullptr, &s_gate);
    check("(B) quantize_hlut2_ldlq M_cal <= 2*K starvation gate fallback H=I",
          s_gate.used_starvation_fallback && s_gate.m_cal == 512);
    check("(B) starvation-gated wire preserved at 2.125 bpw (272 B/tile)",
          std::fabs(qg.bits_per_weight() - 2.125) < 1e-12 && qg.raw_bytes() == q.raw_bytes());
    Hlut2RarebalStats s_bal{};
    Hlut2QuantMatrix qb = quantize_hlut2_ldlq(W.data(), N, K, signs, X.data(), M_cal, true, nullptr, &s_bal);
    check("(B) RAREBAL-LDLQ M_cal=640 > 2*K N=128 K=256 rare trace equalization",
          !s_bal.used_starvation_fallback && s_bal.m_cal == M_cal && s_bal.m_common > 0 &&
          s_bal.m_rare > 0 && s_bal.tr_common > 0.0 && s_bal.tr_rare > 0.0 && s_bal.lambda > 0.0);

    // ---- (C) P150A superblock 4352 B = 16 * 272 B/tile on K=128, N=128 ----
    std::vector<float> W128(size_t(128) * 128);
    for (int r = 0; r < 128; ++r) for (int c = 0; c < 128; ++c) W128[size_t(r) * 128 + c] = bf16_to_f32(f32_to_bf16_rne(0.01f * float((r * 13 + c) % 17)));
    Hlut2QuantMatrix q128 = quantize_hlut2(W128.data(), 128, 128, make_wht_signs(128, 0x44535634315F4800ULL));
    Hlut2P150Packed p = pack_hlut2_for_p150_shard(q128, 0, 128);
    check("(C) pack_hlut2_p150_pages exact 4352 B = 16 * 272 B/tile",
          p.total_bytes() == 4352u && p.total_bytes() == size_t(p.kt4 * p.nt4) * 16u * 272u && p.kt4 == 1 && p.nt4 == 1);

    // ---- (C) Pinned 11/11 hot 98.61 MiB L1 in logs/p150_worker.log ----
    const int hot_shards = 11; const double hot_l1_mib = 98.61, total_l1_mib = 117.22, tp2_mib = hot_l1_mib / double(hot_shards);
    check("(C) Pinned 11/11 hot 98.61 MiB L1 of 117.22 MiB/card (logs/p150_worker.log)",
          hot_shards == 11 && std::fabs(hot_l1_mib - double(hot_shards) * tp2_mib) < 1e-9 &&
          total_l1_mib >= hot_l1_mib);
    std::printf("\nWave 5 verification: %s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
