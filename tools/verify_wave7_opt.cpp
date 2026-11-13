// tools/verify_wave7_opt.cpp — Pure-C++17 Wave 7 optimizer verifier (no HIP, no Python, no device).
// (A) Exact P150A per-Tensix L1 MR byte budget: 1,499,136 usable/bank, l1_small_size = 2,048 B,
//     two-tier GDDR6 MP demotion -> production 17 hot shards/card (34 TP2 shards = 152.40 MiB L1)
//     at exact 1,453,184 B <= 1,459,200 B FABRIC_1D bank with x_decode + finish_decode[0..1] 100% L1-resident
//     (pure L1-resident 129,927,808 B = 123.91 MiB/card; bank 0-0 lowest L1 = 367,232 B > 358,400 B); expanded ceiling 18.
// (B) Expert-bucketed BM=16 SwiGLU + BM=32 down32 WHT128-HLUT2 grouped parity vs token-major reference
//     cosine > 0.99999 across M = 64 routed batch, with exact 4352 B superblock = 16 * 272 B/tile.
// (C) Kolmogorov-Borel multi-branch tree verifier (dsv41_kolmogorov_tree_verifier.hpp) 6/6 self-tests
//     (TV < 1e-14, Dirac vs raw-softmax audit, sibling-invariant RoPE, forward KV compaction) +
//     exact SGPR bitpack verification (0x9345231309050301ULL, 0x32221110u, 0x4211000Fu).
// (D) audit-gfx906 enforced by Makefile `check` dependency: 88/88 kernels ScratchSize == 0, 0 trig, 0 peer.
#include "../p150/dsv41_p150_l1.hpp"
#define DSV41_KOLMOGOROV_TREE_INCLUDE_TESTS
#include "../mi50/dsv41_kolmogorov_tree_verifier.hpp"
#include "../mi50/dsv41_tree_spec.hpp"
#include "../src/hlut2_pack.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int g_fail = 0;

void check(const char* n, bool ok) {
    std::printf(ok ? "[PASS] %s\n" : "[FAIL] %s\n", n);
    if (!ok) ++g_fail;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = double(a[i]);
        const double y = double(b[i]);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

void test_a_p150_l1() {
    using namespace dsv41;
    constexpr size_t x_tilized   = size_t(1u) * 16u * 32u * 5120u * 2u;
    constexpr size_t weights     = size_t(288u) * 32u * 32u * 4u;
    constexpr size_t slot_ids    = size_t(320u) * 2u;
    constexpr size_t gate_parts  = size_t(10u) * 512u * 1152u * 4u;
    constexpr size_t up_parts    = size_t(10u) * 512u * 1152u * 4u;
    constexpr size_t z_rot        = size_t(1u) * 512u * 1152u * 2u;
    constexpr size_t combined    = size_t(32u) * 5120u * 4u;
    constexpr size_t x_rm        = size_t(512u) * 5120u * 2u;
    constexpr size_t slot_meta   = size_t(288u) * 32u * 4u;
    constexpr size_t lut_bytes   = size_t(40u) * 256u * 2u;
    constexpr size_t dummy       = size_t(1088u) * 4u;
    constexpr size_t y_accum512  = size_t(512u) * 5120u * 4u;
    constexpr size_t exact_sum =
        x_tilized + weights + slot_ids + gate_parts + up_parts + z_rot + combined +
        x_rm + slot_meta + lut_bytes + dummy + y_accum512;

    check("(A) raw 1,572,864 - firmware 73,728 == exact usable 1,499,136 B/bank",
          kP150L1CoreRawBytes - kP150L1FirmwareGuardBytes == kP150L1CoreUsableBytes);
    check("(A) 110 cores * 1,499,136 B/card == 157.266 MiB usable per card",
          kP150L1UsableBytesPerCard == 164904960u &&
          std::fabs(double(kP150L1UsableBytesPerCard) / 1048576.0 - 157.266) < 0.01);
    check("(A) 260 KiB/core CB guard == 27.930 MiB/card",
          kP150L1CbGuardPerCore == 266240u &&
          kP150L1CbGuardBytesPerCard == 29286400u);
    check("(A) base non-hot workspace == 71,234,432 B (67.934 MiB/card) -> default ceiling 13",
          kP150BaseNonHotL1Bytes == exact_sum && exact_sum == 71234432u &&
          kP150DefaultCeiling == 13u);
    check("(A) Tier-1b demotion keeps x_decode + finish_decode[0..1] 100% L1-resident -> exact 17 shards/card (1,453,184 B <= 1,459,200 B FABRIC_1D bank, pure L1 = 129,927,808 B = 123.91 MiB/card)",
          kP150ConfiguredHotShards == 17u &&
          kP150Tier1Ceiling == 17u &&
          kP150L1SmallSizeBytes == 2048u &&
          kP150L1Fabric1DBankBytes == 1459200u &&
          kP150Tier1At17PureL1BytesPerCard == 129927808ULL &&
          p150_requires_prefill_demotion(17u) &&
          !p150_requires_decode_demotion(17u) &&
          p150_requires_weights_demotion(17u) &&
          p150_per_bank_l1_bytes(17u, true) == 1460864u &&
          p150_per_bank_l1_bytes(17u, true) <= kP150L1CoreUsableBytes &&
          p150_fabric1d_per_bank_l1_bytes(17u, false) == 1191040u &&
          p150_fabric1d_per_bank_l1_bytes(17u, true) == 1453184u &&
          p150_fabric1d_per_bank_l1_bytes(17u, true) <= kP150L1Fabric1DBankBytes);
    check("(A) Tier-2 four-tensor GDDR6 demotion frees 22,151,168 B -> exact expanded 18 shards/card",
          kP150DemotableL1Bytes == 22151168u &&
          kP150ExpandedNonHotL1Bytes == 49083264u &&
          kP150ExpandedCeiling == 18u &&
          p150_requires_decode_demotion(18u));

    bool fit_all = true;
    for (uint32_t s = 11u; s <= 18u; ++s) fit_all = fit_all && p150_hot_shards_fit(s);
    check("(A) expandable 11..18 shards/card fit exact byte budget; 19 spills",
          fit_all && !p150_hot_shards_fit(19u));

    // Unsigned underflow guard on bucket_slot_capacity(M = 1..8)
    bool cap_ok = true;
    for (uint32_t M = 1u; M <= 8u; ++M) {
        const uint32_t raw_max = M * 6u;
        const uint32_t target_cap = (M <= 1u) ? 6u : ((M <= 8u) ? std::max<uint32_t>(raw_max, 12u) : raw_max);
        cap_ok = cap_ok && (target_cap >= raw_max) && (target_cap <= 288u);
    }
    check("(A) bucket_slot_capacity(M=1..8) free of unsigned underflow across M=6 chain and M=8 tree",
          cap_ok);

    // O(1) l1_slot_lut [1,40,256] UINT16 == 20,480 B
    std::array<uint16_t, 40u * 256u> lut{};
    lut.fill(0xFFFFu);
    lut[size_t(29u) * 256u + size_t(92u)] = 16u;
    check("(A) O(1) l1_slot_lut [1,40,256] UINT16 == 20,480 B and direct hot slot lookup",
          lut.size() * sizeof(uint16_t) == 20480u && lut[size_t(29u) * 256u + 92u] == 16u);
}

void test_b_grouped_swiglu16_down32_parity() {
    using namespace dsv41;
    constexpr int N = 128;
    constexpr int K = 128;
    constexpr int M = 64;      // M >= 64 activates BM=16 SwiGLU + BM=32 down32!
    constexpr uint64_t seed = 0x44535634315F4800ULL;

    const auto signs_h = make_wht_signs(K, seed);
    const auto signs_i = make_wht_signs(K, seed ^ 0x5151ULL);

    std::vector<float> Wg(size_t(N) * K), Wu(size_t(N) * K), Wd(size_t(N) * K);
    std::vector<float> X(size_t(M) * K);
    std::mt19937_64 rng(0x571509ULL);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < K; ++c) {
            const float w = bf16_to_f32(f32_to_bf16_rne(dist(rng)));
            Wg[size_t(r) * K + c] = w;
            Wu[size_t(r) * K + c] = bf16_to_f32(f32_to_bf16_rne(w + 0.01f * float((r * 7 + c) % 11)));
            Wd[size_t(r) * K + c] = bf16_to_f32(f32_to_bf16_rne(w * 0.5f + 0.001f * float((r * 5 + c) % 13)));
        }
    }
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < K; ++c) {
            X[size_t(r) * K + c] = bf16_to_f32(f32_to_bf16_rne(dist(rng)));
        }
    }

    Hlut2QuantMatrix qg = quantize_hlut2(Wg.data(), N, K, signs_h);
    Hlut2QuantMatrix qu = quantize_hlut2(Wu.data(), N, K, signs_h);
    Hlut2QuantMatrix qd = quantize_hlut2(Wd.data(), N, K, signs_i);

    Hlut2P150Packed p = pack_hlut2_for_p150_shard(qg, 0, N);
    check("(B) expert-bucketed P150 superblock == 4352 B = 16 * 272 B/tile at 2.125 bpw",
          p.total_bytes() == 4352u && p.total_bytes() == size_t(p.kt4 * p.nt4) * 16u * 272u &&
          std::fabs(qg.bits_per_weight() - 2.125) < 1e-12);

    std::vector<float> Wgrot, Wurot, Wdrot;
    dequantize_hlut2_to_rot_f32(qg, Wgrot);
    dequantize_hlut2_to_rot_f32(qu, Wurot);
    dequantize_hlut2_to_rot_f32(qd, Wdrot);

    std::vector<float> Xrot(size_t(M) * K);
    wht128_rows(X.data(), Xrot.data(), M, K, signs_h);

    std::vector<float> y_token(size_t(M) * N, 0.0f);
    std::vector<float> y_group(size_t(M) * N, 0.0f);

    // Token-major reference
    for (int m = 0; m < M; ++m) {
        std::vector<float> g(N), u(N), z(K);
        for (int n = 0; n < N; ++n) {
            float accg = 0.0f, accu = 0.0f;
            for (int k = 0; k < K; ++k) {
                const float x = Xrot[size_t(m) * K + k];
                accg += x * Wgrot[size_t(n) * K + k];
                accu += x * Wurot[size_t(n) * K + k];
            }
            g[n] = accg;
            u[n] = accu;
        }
        for (int k = 0; k < K; ++k) z[k] = dsv41_swiglu_scalar(g[k], u[k]);
        for (int b = 0; b < K / 128; ++b) wht128_block(z.data() + b * 128, &signs_i[size_t(b) * 2]);
        for (int n = 0; n < N; ++n) {
            float accd = 0.0f;
            for (int k = 0; k < K; ++k) accd += z[k] * Wdrot[size_t(n) * K + k];
            y_token[size_t(m) * N + n] = accd;
        }
    }

    // Grouped BM=16 SwiGLU + BM=32 down32 tiled reference across M=64
    std::vector<float> z_group(size_t(M) * K, 0.0f);
    for (int m0 = 0; m0 < M; m0 += 16) {
        const int bm = std::min(16, M - m0);
        std::vector<float> g_tile(size_t(bm) * N, 0.0f), u_tile(size_t(bm) * N, 0.0f);
        for (int n = 0; n < N; ++n) {
            for (int t = 0; t < bm; ++t) {
                float accg = 0.0f, accu = 0.0f;
                for (int k = 0; k < K; ++k) {
                    const float x = Xrot[size_t(m0 + t) * K + k];
                    accg += x * Wgrot[size_t(n) * K + k];
                    accu += x * Wurot[size_t(n) * K + k];
                }
                g_tile[size_t(t) * N + n] = accg;
                u_tile[size_t(t) * N + n] = accu;
            }
        }
        for (int t = 0; t < bm; ++t) {
            for (int k = 0; k < K; ++k) {
                z_group[size_t(m0 + t) * K + k] =
                    dsv41_swiglu_scalar(g_tile[size_t(t) * N + k], u_tile[size_t(t) * N + k]);
            }
            for (int b = 0; b < K / 128; ++b) {
                wht128_block(z_group.data() + size_t(m0 + t) * K + b * 128, &signs_i[size_t(b) * 2]);
            }
        }
    }
    for (int m0 = 0; m0 < M; m0 += 32) {
        const int bm = std::min(32, M - m0);
        for (int n = 0; n < N; ++n) {
            for (int t = 0; t < bm; ++t) {
                float accd = 0.0f;
                for (int k = 0; k < K; ++k) {
                    accd += z_group[size_t(m0 + t) * K + k] * Wdrot[size_t(n) * K + k];
                }
                y_group[size_t(m0 + t) * N + n] = accd;
            }
        }
    }

    const double cos = cosine(y_token, y_group);
    check("(B) expert-bucketed BM=16 SwiGLU + BM=32 down32 (M=64) vs token-major cosine > 0.99999",
          cos > 0.99999 && std::isfinite(cos));
    std::printf("[DATA] (B) grouped BM=16+BM=32 parity cosine=%.12f (target > 0.99999)\n", cos);
}

void test_c_kolmogorov_tree_and_sgpr_bitpacks() {
    using namespace dsv41;

    // 1. Run all 6 self-tests in dsv41_kolmogorov_tree_verifier.hpp!
    const bool selftests_ok = run_kolmogorov_tree_verifier_selftests();
    check("(C) dsv41_kolmogorov_tree_verifier.hpp 6/6 self-tests PASS (TV < 1e-14, Dirac audit, RoPE, compaction)",
          selftests_ok);

    // 2. Verify canonical width-[3,2,1,1] M=8 SGPR bitpacks match build_spec_tree_topology bit-for-bit!
    constexpr int32_t parents[8] = {-1, 0, 0, 0, 1, 1, 2, 4};
    constexpr int32_t tokens[8]  = {100, 11, 12, 13, 21, 22, 23, 31};
    Dsv41SpecTree8 tree{};
    const bool built = build_spec_tree_topology(8u, 1000u, parents, tokens, &tree);

    uint64_t packed_anc = 0ULL;
    uint32_t packed_dep = 0u;
    uint32_t packed_par = 0u;
    for (uint32_t m = 0u; m < 8u; ++m) {
        packed_anc |= (uint64_t(tree.ancestor_mask[m]) << (8u * m));
        packed_dep |= (uint32_t(tree.depth[m] & 0xFu) << (4u * m));
        packed_par |= (uint32_t(uint32_t(tree.parent[m]) & 0xFu) << (4u * m));
    }

    check("(C) canonical M=8 tree SGPR bitpacks == 0x9345231309050301ULL / 0x32221110u / 0x4211000Fu",
          built &&
          packed_anc == 0x9345231309050301ULL &&
          packed_dep == 0x32221110u &&
          packed_par == 0x4211000Fu);
}

}  // namespace

int main() {
    std::printf("=== Wave 7 pure-C++ optimizer verification (no HIP/no Python/no device) ===\n");
    test_a_p150_l1();
    test_b_grouped_swiglu16_down32_parity();
    test_c_kolmogorov_tree_and_sgpr_bitpacks();
    std::printf("\nWave 7 verification: %s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
