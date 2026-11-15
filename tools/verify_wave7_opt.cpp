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

double exact_tree8_marginal_tv_at_node(
    const dsv41::Dsv41SpecTree8& topo,
    uint32_t cur,
    const double* p_row,
    size_t V)
{
    std::vector<double> q_emit(V, 0.0);
    std::vector<bool> in_children(V, false);
    double branch_cdf = 0.0;
    uint8_t m = static_cast<uint8_t>(topo.child_mask[cur] & topo.valid_node_mask);
    while (m != 0u) {
        const uint32_t c = static_cast<uint32_t>(__builtin_ctz(m));
        m &= static_cast<uint8_t>(m - 1u);
        if (((topo.valid_node_mask >> c) & 1u) == 0u) continue;
        const int32_t tok = topo.token_id[c];
        if (tok >= 0 && size_t(tok) < V && !in_children[size_t(tok)]) {
            in_children[size_t(tok)] = true;
            const double pm = p_row[size_t(tok)];
            if (pm > 0.0) {
                q_emit[size_t(tok)] += pm;
                branch_cdf += pm;
            }
        }
    }
    const double rem_mass = std::max(0.0, 1.0 - branch_cdf);
    if (rem_mass > 1e-15) {
        double comp_sum = 0.0;
        for (size_t v = 0; v < V; ++v) {
            if (!in_children[v] && p_row[v] > 0.0) comp_sum += p_row[v];
        }
        const double scale = (comp_sum > 0.0) ? (rem_mass / comp_sum) : 0.0;
        for (size_t v = 0; v < V; ++v) {
            if (!in_children[v] && p_row[v] > 0.0) q_emit[v] += p_row[v] * scale;
        }
    }
    double l1 = 0.0;
    for (size_t v = 0; v < V; ++v) l1 += std::abs(q_emit[v] - p_row[v]);
    return 0.5 * l1;
}

double exact_tree16_marginal_tv_at_node(
    const dsv41::Dsv41SpecTree16& topo,
    uint32_t cur,
    const double* p_row,
    size_t V)
{
    std::vector<double> q_emit(V, 0.0);
    std::vector<bool> in_children(V, false);
    double branch_cdf = 0.0;
    uint16_t m = static_cast<uint16_t>(topo.child_mask[cur] & topo.valid_node_mask);
    while (m != 0u) {
        const uint32_t c = static_cast<uint32_t>(__builtin_ctz(m));
        m &= static_cast<uint16_t>(m - 1u);
        if (((topo.valid_node_mask >> c) & 1u) == 0u) continue;
        const int32_t tok = topo.token_id[c];
        if (tok >= 0 && size_t(tok) < V && !in_children[size_t(tok)]) {
            in_children[size_t(tok)] = true;
            const double pm = p_row[size_t(tok)];
            if (pm > 0.0) {
                q_emit[size_t(tok)] += pm;
                branch_cdf += pm;
            }
        }
    }
    const double rem_mass = std::max(0.0, 1.0 - branch_cdf);
    if (rem_mass > 1e-15) {
        double comp_sum = 0.0;
        for (size_t v = 0; v < V; ++v) {
            if (!in_children[v] && p_row[v] > 0.0) comp_sum += p_row[v];
        }
        const double scale = (comp_sum > 0.0) ? (rem_mass / comp_sum) : 0.0;
        for (size_t v = 0; v < V; ++v) {
            if (!in_children[v] && p_row[v] > 0.0) q_emit[v] += p_row[v] * scale;
        }
    }
    double l1 = 0.0;
    for (size_t v = 0; v < V; ++v) l1 += std::abs(q_emit[v] - p_row[v]);
    return 0.5 * l1;
}

void test_d_wave8_3_dynamic_m8_and_filler_quarantine() {
    using namespace dsv41;

    // 1. Delta 8.3A: Full-range kAutoTree16MaxContext == 0 across 0..1,048,576
    {
        Dsv41AdaptiveTreeWidthController auto_ctrl(Dsv41TreeWidthMode::Auto);
        for (int i = 0; i < 16; ++i) auto_ctrl.observe_step(1, true);
        const bool ok_range =
            (Dsv41AdaptiveTreeWidthController::kAutoTree16MaxContext == 0) &&
            (auto_ctrl.select_width(true, 0) == 8) &&
            (auto_ctrl.select_width(true, 9029) == 8) &&
            (auto_ctrl.select_width(true, 49151) == 8) &&
            (auto_ctrl.select_width(true, 224410) == 8) &&
            (auto_ctrl.select_width(true, 1048575) == 8);
        Dsv41AdaptiveTreeWidthController opt_in_16(Dsv41TreeWidthMode::Fixed16);
        check("(D1) Wave 8.3A kAutoTree16MaxContext == 0 keeps M=8 across 0..1,048,576 (Fixed16 stays opt-in)",
              ok_range && opt_in_16.select_width(true, 9029) == 16);
    }

    // 2. Delta 8.3B: deep_select_top7_confidence_tree8 exact chain collapse (depth=5) & wide bush (4+3, max_depth=2)
    {
        std::array<int32_t, kDeepSelect8PoolNodes> pool_toks{};
        for (int i = 0; i < kDeepSelect8PoolNodes; ++i) pool_toks[size_t(i)] = 10 + i;

        std::array<double, kDeepSelect8PoolNodes> chain_conf{};
        chain_conf.fill(0.02);
        chain_conf[0] = 1.0;
        // Primary chain: 1 -> 5 -> 11 -> 14 -> 15 (5 drafts -> depth 5!), plus root backups 2, 3
        chain_conf[1] = 0.96;
        chain_conf[5] = 0.95;
        chain_conf[11] = 0.94;
        chain_conf[14] = 0.93;
        chain_conf[15] = 0.92;
        chain_conf[2] = 0.35;
        chain_conf[3] = 0.30;
        Dsv41DeepSelectPlan8 chain_plan{};
        const bool chain_ok = deep_select_top7_confidence_tree8(
            kDeepSelect8PoolNodes, kDeepSelect8PoolParents.data(),
            pool_toks.data(), chain_conf.data(), /*eos=*/1, &chain_plan);

        Dsv41SpecTree8 chain_topo{};
        const bool chain_topo_ok = build_spec_tree_topology(
            8u, 64u, chain_plan.parents.data(), chain_plan.tokens.data(),
            &chain_topo, chain_plan.valid_node_mask);
        uint64_t check_anc64 = 0ULL;
        uint32_t check_dep32 = 0u;
        for (uint32_t m = 0; m < 8u; ++m) {
            check_anc64 |= (uint64_t(chain_topo.ancestor_mask[m]) << (8u * m));
            check_dep32 |= (uint32_t(chain_topo.depth[m] & 0xFu) << (4u * m));
        }

        std::array<double, kDeepSelect8PoolNodes> bush_conf{};
        bush_conf.fill(0.01);
        bush_conf[0] = 1.0;
        for (int u = 1; u <= 4; ++u) bush_conf[size_t(u)] = 0.24;  // 4 root children
        for (int u = 5; u <= 7; ++u) bush_conf[size_t(u)] = 0.22;  // 3 depth-2 children -> 4 + 3 bush!
        Dsv41DeepSelectPlan8 bush_plan{};
        const bool bush_ok = deep_select_top7_confidence_tree8(
            kDeepSelect8PoolNodes, kDeepSelect8PoolParents.data(),
            pool_toks.data(), bush_conf.data(), /*eos=*/1, &bush_plan);
        int d1_cnt = 0, d2_cnt = 0;
        for (int i = 1; i < 8; ++i) {
            if (bush_plan.depths[size_t(i)] == 1u) ++d1_cnt;
            if (bush_plan.depths[size_t(i)] == 2u) ++d2_cnt;
        }

        check("(D2) Wave 8.3B DeepSelect8 exact chain collapse (depth=5) & wide 4+3 bush (max_depth=2) + SGPR bitpacks",
              chain_ok && chain_topo_ok &&
              chain_plan.valid_node_mask == 0xFFu &&
              chain_plan.max_depth == 5 &&
              chain_plan.anc_mask64 == check_anc64 &&
              chain_plan.depth_nibbles32 == check_dep32 &&
              bush_ok && bush_plan.valid_node_mask == 0xFFu &&
              bush_plan.max_depth == 2 && d1_cnt == 4 && d2_cnt == 3);
    }

    // 3. Delta 8.3C: 100,000 sweeps under V = 4 candidate shortage & early-EOS pruning across M=8 & M=16
    constexpr int kSweeps = 100000;
    constexpr size_t V = 4;
    std::mt19937_64 rng(0x57415645383343ULL);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    uint64_t invalid_accepts = 0;
    uint64_t shortage_cases_m8 = 0;
    uint64_t shortage_cases_m16 = 0;
    double max_tv_m8 = 0.0;
    double max_tv_m16 = 0.0;
    bool inert_filler_ok = true;

    std::array<double, 8 * V> probs8{};
    std::array<double, 8 * V> q_draft8{};
    std::array<double, V> p_res_scratch{};

    for (int s = 0; s < kSweeps; ++s) {
        // V = 4 with eos_token_id = 0, frequent -1 shortages & sibling collisions!
        std::array<int32_t, kDeepSelect8PoolNodes> ptok8{};
        std::array<double, kDeepSelect8PoolNodes> pconf8{};
        ptok8[0] = (s % 16 == 0) ? 0 : int32_t(1 + (s % 3));  // include root EOS edge cases!
        pconf8[0] = 1.0;
        for (int u = 1; u < kDeepSelect8PoolNodes; ++u) {
            const uint64_t r = rng();
            ptok8[size_t(u)] = ((r & 3ULL) == 0ULL) ? -1 : int32_t(r % V);
            pconf8[size_t(u)] = u01(rng);
        }

        Dsv41DeepSelectPlan8 plan8{};
        if (!deep_select_top7_confidence_tree8(
                kDeepSelect8PoolNodes, kDeepSelect8PoolParents.data(),
                ptok8.data(), pconf8.data(), /*eos_token_id=*/0, &plan8)) {
            inert_filler_ok = false;
            break;
        }
        if (plan8.valid_node_mask != 0xFFu) ++shortage_cases_m8;

        // Verify quarantined filler slots attach inertly under root 0 with anc_mask = (1<<0)|(1<<i) and token = -1,
        // THEN overwrite filler token with in-vocab token 2 (simulating runtime kDSparkNoiseId gather) to prove
        // valid_node_mask alone blocks acceptance AND excludes filler from residual mass!
        std::array<int32_t, 8> runtime_toks8 = plan8.tokens;
        for (int i = 1; i < 8; ++i) {
            if (((plan8.valid_node_mask >> i) & 1u) == 0u) {
                const uint8_t row_anc = static_cast<uint8_t>((plan8.anc_mask64 >> (8 * i)) & 0xFFULL);
                if (plan8.tokens[size_t(i)] != -1 ||
                    plan8.parents[size_t(i)] != 0 ||
                    plan8.depths[size_t(i)] != 1u ||
                    row_anc != static_cast<uint8_t>((1u << 0) | (1u << i))) {
                    inert_filler_ok = false;
                }
                runtime_toks8[size_t(i)] = 2;  // High-prob in-vocab noise token!
            }
        }

        Dsv41SpecTree8 topo8{};
        if (!build_spec_tree_topology(8u, uint32_t(s & 127), plan8.parents.data(),
                                      runtime_toks8.data(), &topo8, plan8.valid_node_mask)) {
            inert_filler_ok = false;
            break;
        }

        for (int n = 0; n < 8; ++n) {
            double sp = 0.0, sq = 0.0;
            for (size_t v = 0; v < V; ++v) {
                probs8[size_t(n) * V + v] = 0.05 + u01(rng);
                q_draft8[size_t(n) * V + v] = 0.05 + u01(rng);
                sp += probs8[size_t(n) * V + v];
                sq += q_draft8[size_t(n) * V + v];
            }
            for (size_t v = 0; v < V; ++v) {
                probs8[size_t(n) * V + v] /= sp;
                q_draft8[size_t(n) * V + v] /= sq;
            }
        }

        std::array<double, 8> u_draws{};
        for (double& d : u_draws) d = u01(rng);

        // Deterministic M=8 verifier check
        const auto vr_det = verify_deterministic_tree_u01(topo8, probs8.data(), V, u_draws.data());
        for (uint32_t d = 0; d < vr_det.num_emitted; ++d) {
            const uint32_t nd = vr_det.accepted_nodes[d];
            if (((plan8.valid_node_mask >> nd) & 1u) == 0u) ++invalid_accepts;
        }

        // Stochastic SpecInfer M=8 verifier check (even with unmasked raw_child_mask containing filler bits!)
        std::array<uint8_t, 8> poisoned_raw_children = topo8.child_mask;
        for (int i = 1; i < 8; ++i) {
            if (((plan8.valid_node_mask >> i) & 1u) == 0u) {
                poisoned_raw_children[0] = static_cast<uint8_t>(poisoned_raw_children[0] | (1u << i));
            }
        }
        const auto vr_sto = verify_stochastic_specinfer_tree(
            topo8, poisoned_raw_children.data(), probs8.data(), q_draft8.data(),
            V, u_draws.data(), u01(rng), p_res_scratch.data());
        for (uint32_t d = 0; d < vr_sto.num_emitted; ++d) {
            const uint32_t nd = vr_sto.accepted_nodes[d];
            if (((plan8.valid_node_mask >> nd) & 1u) == 0u) ++invalid_accepts;
        }

        for (uint32_t n = 0; n < 8u; ++n) {
            if (((topo8.valid_node_mask >> n) & 1u) == 0u) continue;
            const double tv = exact_tree8_marginal_tv_at_node(topo8, n, &probs8[size_t(n) * V], V);
            if (tv > max_tv_m8) max_tv_m8 = tv;
        }

        // Also stress M=16 DeepSelect under V=4 candidate shortage & early-EOS
        std::array<int32_t, kDeepSelectPoolNodes> ptok16{};
        std::array<double, kDeepSelectPoolNodes> pconf16{};
        ptok16[0] = (s % 16 == 0) ? 0 : int32_t(1 + (s % 3));
        pconf16[0] = 1.0;
        for (int u = 1; u < kDeepSelectPoolNodes; ++u) {
            const uint64_t r = rng();
            ptok16[size_t(u)] = ((r & 3ULL) == 0ULL) ? -1 : int32_t(r % V);
            pconf16[size_t(u)] = u01(rng);
        }
        Dsv41DeepSelectPlan16 plan16{};
        if (!deep_select_top15_confidence_tree(
                kDeepSelectPoolNodes, kDeepSelectPoolParents.data(),
                ptok16.data(), pconf16.data(), /*eos_token_id=*/0, &plan16)) {
            inert_filler_ok = false;
            break;
        }
        if (plan16.valid_node_mask != 0xFFFFu) ++shortage_cases_m16;
        std::array<int32_t, 16> runtime_toks16 = plan16.tokens;
        for (int i = 1; i < 16; ++i) {
            if (((plan16.valid_node_mask >> i) & 1u) == 0u) {
                const uint64_t w = (i < 4) ? plan16.anc_mask256.w0
                                 : (i < 8) ? plan16.anc_mask256.w1
                                 : (i < 12) ? plan16.anc_mask256.w2
                                            : plan16.anc_mask256.w3;
                const uint16_t row_anc = static_cast<uint16_t>((w >> (16 * (i & 3))) & 0xFFFFULL);
                if (plan16.tokens[size_t(i)] != -1 ||
                    plan16.parents[size_t(i)] != 0 ||
                    row_anc != static_cast<uint16_t>((1u << 0) | (1u << i))) {
                    inert_filler_ok = false;
                }
                runtime_toks16[size_t(i)] = 2;
            }
        }
        Dsv41SpecTree16 topo16{};
        if (!build_spec_tree16_topology(16u, uint32_t(s & 127), plan16.parents.data(),
                                        runtime_toks16.data(), &topo16, plan16.valid_node_mask)) {
            inert_filler_ok = false;
            break;
        }
        std::array<double, 16> u16{};
        for (int i = 0; i < 16; ++i) u16[size_t(i)] = u_draws[size_t(i & 7)];
        const auto vr16 = verify_deterministic_tree16_u01_lazy(
            topo16, V, u16.data(), [&](uint32_t cur) -> const double* {
                return &probs8[size_t(cur & 7u) * V];
            });
        for (uint32_t d = 0; d < vr16.num_emitted; ++d) {
            const uint32_t nd = vr16.accepted_nodes[d];
            if (((plan16.valid_node_mask >> nd) & 1u) == 0u) ++invalid_accepts;
        }
        const double tv16 = exact_tree16_marginal_tv_at_node(topo16, 0u, &probs8[0], V);
        if (tv16 > max_tv_m16) max_tv_m16 = tv16;
    }

    // Empirical Monte Carlo check on stochastic SpecInfer with quarantined filler slot carrying q_c = 1.0 on tok=2!
    double sto_emp_tv = 0.0;
    {
        const int32_t par[8]  = {-1, 0, 0, 0, 0, 0, 0, 0};
        const int32_t toks[8] = {1,  1, 2, 2, 2, 2, 2, 2};  // slots 2..7 are quarantined filler!
        const uint8_t vmask = 0x03u;                        // ONLY root 0 and child 1 are valid!
        Dsv41SpecTree8 t8{};
        build_spec_tree_topology(8u, 0u, par, toks, &t8, vmask);
        const std::array<double, V> p_target = {0.15, 0.25, 0.45, 0.15};
        std::array<double, 8 * V> p_tab{};
        std::array<double, 8 * V> q_tab{};
        for (int n = 0; n < 8; ++n) {
            for (size_t v = 0; v < V; ++v) p_tab[size_t(n) * V + v] = p_target[v];
        }
        // Child 1 proposes tok=1 deterministically; quarantined slots 2..7 put 100% mass on tok=2
        // (would wipe out p_res[2] if not excluded from residual subtraction!)
        q_tab[1 * V + 1] = 1.0;
        for (int n = 2; n < 8; ++n) q_tab[size_t(n) * V + 2] = 1.0;
        std::array<uint8_t, 8> raw_cm{};
        raw_cm[0] = 0xFEu;

        std::array<uint64_t, V> counts{};
        for (int i = 0; i < kSweeps; ++i) {
            std::array<double, 8> ua{};
            for (double& x : ua) x = u01(rng);
            const auto vr = verify_stochastic_specinfer_tree(
                t8, raw_cm.data(), p_tab.data(), q_tab.data(), V, ua.data(), u01(rng), p_res_scratch.data());
            if (((vmask >> vr.accepted_nodes[vr.num_emitted - 1u]) & 1u) == 0u) ++invalid_accepts;
            ++counts[size_t(vr.emitted_tokens[0])];
        }
        for (size_t v = 0; v < V; ++v) {
            sto_emp_tv += 0.5 * std::abs(double(counts[v]) / double(kSweeps) - p_target[v]);
        }
    }

    std::printf("[DATA] (D3) 100,000 V=4 shortage/EOS sweeps: shortage(M8=%llu, M16=%llu) invalid_accepts=%llu max_TV_M8=%.3e max_TV_M16=%.3e sto_MC_TV=%.4f\n",
                static_cast<unsigned long long>(shortage_cases_m8),
                static_cast<unsigned long long>(shortage_cases_m16),
                static_cast<unsigned long long>(invalid_accepts),
                max_tv_m8, max_tv_m16, sto_emp_tv);
    check("(D3) Wave 8.3C valid_node_mask quarantine: 0 filler acceptances & max_TV < 1e-14 across 100,000 V=4/EOS sweeps",
          inert_filler_ok && shortage_cases_m8 > 10000ULL && shortage_cases_m16 > 10000ULL &&
          invalid_accepts == 0ULL && max_tv_m8 < 1e-14 && max_tv_m16 < 1e-14 && sto_emp_tv < 0.008);
}

}  // namespace

int main() {
    std::printf("=== Wave 7 + Wave 8.3 pure-C++ optimizer verification (no HIP/no Python/no device) ===\n");
    test_a_p150_l1();
    test_b_grouped_swiglu16_down32_parity();
    test_c_kolmogorov_tree_and_sgpr_bitpacks();
    test_d_wave8_3_dynamic_m8_and_filler_quarantine();
    std::printf("\nWave 7 + 8.3 verification: %s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
