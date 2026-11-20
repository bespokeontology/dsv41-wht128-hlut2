// Wave 9.3 Phase A: Cycle-Exact Per-Stage Bottleneck Profiler for SB-M4-P2-176G
// Profiles on live 4x MI50 (C7:00.0, 88:00.0, 03:00.0, 85:00.0) + 2x Blackhole P150A:
//   1. Cold Prefill (M = 512, 2048, 4096 fresh tokens, reuse = 0): real wall tok/s + sync_timers breakdown
//   2. Decode Bursts (DSpark Tree Step, 64 gen tokens) at 8K, 64K, and 256K active CSA2 context
// Writes zero-PII Markdown table to --out-md (default: docs/wave9_3_baseline_profile.md).
// 100% Pure C++17 / HIP — Zero Python Anywhere.

#include "../mi50/dsv41_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct PrefillRow {
    int tokens = 0;
    double wall_ms = 0.0;
    double wall_tps = 0.0;
    double sync_wall_ms = 0.0;
    dsv41::Dsv41StageBreakdown bd{};
};

struct DecodeRow {
    int ctx_len = 0;
    int gen_tokens = 0;
    uint64_t steps = 0;
    double decode_ms = 0.0;
    double tok_s = 0.0;
    double tau = 0.0;
    double accept_pct = 0.0;
    double bb_ms = 0.0;
    double dspark_ms = 0.0;
    double sample_ms = 0.0;
    double attn_ms = 0.0;
    double hsi_ms = 0.0;
    double moe_ms = 0.0;
    double ar_ms = 0.0;
    double p150_wait_ms = 0.0;
    double p150_dev_ms = 0.0;
    double dev3_busy_ms = 0.0;
    double engram_ms = 0.0;
};

std::vector<int32_t> build_fresh_prompt(const dsv41::Dsv41Tokenizer& tok, int target_tokens, int nonce) {
    std::string body = "Session profile-nonce-" + std::to_string(nonce) +
                       ": Analyze heterogeneous AMD Instinct MI50 gfx906 and Blackhole P150A scheduling.\n";
    int sec = 1;
    std::vector<int32_t> ids = tok.encode(body, true);
    while (int(ids.size()) < target_tokens) {
        std::string chunk =
            "Section " + std::to_string(sec++) + " (nonce=" + std::to_string(nonce) + "): "
            "In asymmetric SB-M4-P2-176G inference, 384 DSpark MTP experts and 720 hottest backbone experts "
            "execute unsharded WHT128-HLUT3 tiles on hlut3_dev=3 while 600 warm-hot backbone experts execute "
            "sharded WHT128-HLUT3 on TP3 alongside 7,320 compacted HLUT2 experts and 168 P150A experts over "
            "4x 800G FABRIC_1D with zero PCIe system-scope RMW atomics across PEX88048.\n";
        auto sub = tok.encode(chunk, false);
        ids.insert(ids.end(), sub.begin(), sub.end());
    }
    ids.resize(size_t(target_tokens));
    return ids;
}

}  // namespace

int main(int argc, char** argv) {
    ::setenv("OMP_WAIT_POLICY", "PASSIVE", 1);
    ::setenv("GPU_MAX_HW_QUEUES", "8", 0);

    const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                     ? std::string(std::getenv("DSV41_ROOT"))
                                     : std::string(".");
    const std::string ov_dir   = (std::getenv("DSV41_OVERLAY_DIR") != nullptr)
                                     ? std::string(std::getenv("DSV41_OVERLAY_DIR"))
                                     : (root_env + "/wave9_2_overlays");
    std::string overlay       = root_env + "/packed_rarebal_overlay_wave8_5/hot_experts_wave8_5.safetensors";
    std::string dev3_overlay  = ov_dir + "/wave9_2_dev3_hlut3.safetensors";
    std::string tp3_overlay   = ov_dir + "/wave9_2_tp3_hlut3.safetensors";
    std::string out_md        = "docs/wave9_3_baseline_profile.md";
    std::string label         = "Wave 9.2 Baseline (SB-M4-P2-176G)";
    int max_seq_len           = 1048576;
    int superchunk_m          = 512;
    int decode_gen_tokens     = 64;
    bool rocblas_prefill      = true;
    int rocblas_expert_thresh = 16;
    int rocblas_hlut2_thresh  = 48;
    float tree8_conf_thresh   = 0.72f;
    int chain_max_b           = 5;
    bool check_prefill_cos    = false;
    bool link_check           = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--link-check") link_check = true;
        else if (a == "--overlay" && i + 1 < argc) overlay = argv[++i];
        else if (a == "--hlut3-dev3-overlay" && i + 1 < argc) dev3_overlay = argv[++i];
        else if (a == "--hlut3-tp3-overlay" && i + 1 < argc) tp3_overlay = argv[++i];
        else if (a == "--out-md" && i + 1 < argc) out_md = argv[++i];
        else if (a == "--label" && i + 1 < argc) label = argv[++i];
        else if (a == "--max-seq-len" && i + 1 < argc) max_seq_len = std::stoi(argv[++i]);
        else if (a == "--superchunk-m" && i + 1 < argc) superchunk_m = std::stoi(argv[++i]);
        else if (a == "--decode-tokens" && i + 1 < argc) decode_gen_tokens = std::stoi(argv[++i]);
        else if (a == "--no-rocblas-prefill") rocblas_prefill = false;
        else if (a == "--rocblas-prefill") rocblas_prefill = true;
        else if (a == "--rocblas-expert-thresh" && i + 1 < argc) rocblas_expert_thresh = std::stoi(argv[++i]);
        else if (a == "--rocblas-hlut2-thresh" && i + 1 < argc) rocblas_hlut2_thresh = std::stoi(argv[++i]);
        else if (a == "--tree8-conf-thresh" && i + 1 < argc) tree8_conf_thresh = std::stof(argv[++i]);
        else if (a == "--chain-max-b" && i + 1 < argc) chain_max_b = std::stoi(argv[++i]);
        else if (a == "--check-prefill-cos") check_prefill_cos = true;
    }

    if (link_check) {
        std::printf("[PROFILE_W93] Link check OK.\n");
        return 0;
    }

    dsv41::Dsv41Runtime rt;
    rt.set_expert_perm_overlay(overlay);
    if (!dev3_overlay.empty()) rt.set_hlut3_dev3_overlay(dev3_overlay, 3);
    if (!tp3_overlay.empty())  rt.set_hlut3_tp3_overlay(tp3_overlay);
    rt.set_rocblas_prefill(rocblas_prefill);
    rt.set_rocblas_expert_thresh(rocblas_expert_thresh);
    rt.set_rocblas_hlut2_thresh(rocblas_hlut2_thresh);
    rt.set_dspark_tree8_conf_thresh(tree8_conf_thresh);
    rt.set_dspark_chain_max_b(chain_max_b);
    rt.set_csa2_shared_tree8(true);
    rt.set_fused_mhc_norm(true);
    rt.set_fused_dspark_epilogue(true);
    rt.init(root_env + "/packed_mi50",
            root_env + "/hf_meta/tokenizer.json",
            "/dsv41_p150_ipc",
            max_seq_len,
            overlay,
            "",
            root_env + "/packed_vit/dsv41_vit_bf16.safetensors");
    rt.set_dual_stream_overlap(true);
    rt.set_prefill_superchunk_m(superchunk_m);
    rt.set_dspark_tree_width(0);
    rt.set_dspark_static_m8(false);

    const auto mem = rt.query_gpu_mem_margin();
    std::printf("[PROFILE_W93] Free HBM2 @ 1M+ViT: tp0=%.1f tp1=%.1f tp2=%.1f dev3=%.1f/%.1f MiB\n",
                mem.free_mib[0], mem.free_mib[1], mem.free_mib[2], mem.dev3_free_mib, mem.dev3_total_mib);
    std::fflush(stdout);

    // Optional full-40L prefill logits cosine parity & determinism diagnostic
    if (check_prefill_cos) {
        auto ids_cos = build_fresh_prompt(rt.tokenizer(), 512, 999);
        dsv41::Dsv41GenParams gp_c{};
        gp_c.max_new_tokens = 1;
        gp_c.temperature = 0.0f;
        gp_c.enable_dspark = false;
        auto run_once = [&](bool rb, bool sync_t) {
            dsv41::Dsv41GenStats st_x{};
            rt.clear_all_caches();
            rt.set_rocblas_prefill(rb);
            rt.set_sync_timers(sync_t);
            (void)rt.generate(ids_cos, gp_c, [](int32_t) { return true; }, st_x);
            return rt.last_logits_row0();
        };
        auto cmp = [](const char* tag, const std::vector<float>& a, const std::vector<float>& b) {
            double dot = 0.0, na = 0.0, nb = 0.0, max_d = 0.0;
            int ta = 0, tb = 0;
            for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
                dot += double(a[i]) * double(b[i]);
                na  += double(a[i]) * double(a[i]);
                nb  += double(b[i]) * double(b[i]);
                max_d = std::max(max_d, std::abs(double(a[i]) - double(b[i])));
                if (a[i] > a[size_t(ta)]) ta = int(i);
                if (b[i] > b[size_t(tb)]) tb = int(i);
            }
            const double cs = (na > 0.0 && nb > 0.0) ? (dot / std::sqrt(na * nb)) : 0.0;
            std::printf("[PROFILE_W93][PARITY %s] cos=%.8f max_d=%.4f top1=%d(%.2f)/%d(%.2f) match=%d\n",
                        tag, cs, max_d, ta, a[size_t(ta)], tb, b[size_t(tb)], int(ta == tb));
            std::fflush(stdout);
        };
        const auto r0sa = run_once(false, true);
        const auto r0sb = run_once(false, true);
        const auto r1sa = run_once(true,  true);
        const auto r1sb = run_once(true,  true);
        const auto r1aa = run_once(true,  false);
        const auto r1ab = run_once(true,  false);
        cmp("rb0_sync_a vs rb0_sync_b", r0sa, r0sb);
        cmp("rb1_sync_a vs rb1_sync_b", r1sa, r1sb);
        cmp("rb1_async_a vs rb1_async_b", r1aa, r1ab);
        cmp("rb0_sync vs rb1_sync", r0sa, r1sa);
        cmp("rb1_sync vs rb1_async", r1sa, r1aa);
        return 0;
    }

    // 1. Cold Prefill Breakdown (M = 512, 2048, 4096 fresh tokens, reuse = 0)
    const int prefill_lens[3] = {512, 2048, 4096};
    std::vector<PrefillRow> pf_rows;
    int nonce_seq = 100;
    for (int plen : prefill_lens) {
        PrefillRow pr{};
        pr.tokens = plen;
        dsv41::Dsv41GenParams gp{};
        gp.max_new_tokens = 0;
        gp.temperature = 0.0f;
        gp.enable_dspark = false;

        // Pass A: True un-instrumented wall throughput (sync_timers = false)
        rt.clear_all_caches();
        rt.set_sync_timers(false);
        auto ids_a = build_fresh_prompt(rt.tokenizer(), plen, nonce_seq++);
        dsv41::Dsv41GenStats st_a{};
        (void)rt.generate(ids_a, gp, [](int32_t) { return true; }, st_a);
        pr.wall_ms  = st_a.fresh_prefill_ms;
        pr.wall_tps = (pr.wall_ms > 0.0) ? (1000.0 * double(st_a.prefilled_tokens) / pr.wall_ms) : 0.0;

        // Pass B: Synchronized per-stage breakdown (sync_timers = true)
        rt.clear_all_caches();
        rt.set_sync_timers(true);
        rt.reset_stage_breakdown();
        auto ids_b = build_fresh_prompt(rt.tokenizer(), plen, nonce_seq++);
        dsv41::Dsv41GenStats st_b{};
        (void)rt.generate(ids_b, gp, [](int32_t) { return true; }, st_b);
        pr.sync_wall_ms = st_b.fresh_prefill_ms;
        pr.bd = rt.query_stage_breakdown();
        rt.set_sync_timers(false);

        std::printf("[PROFILE_W93][PREFILL M=%4d] wall=%.1fms (%.1f tok/s) | sync=%.1fms | "
                    "qkv=%.1f hsi=%.1f sp=%.1f wo=%.1f a_ar=%.1f | "
                    "rtr=%.1f sh=%.1f hl2=%.1f hl3=%.1f d3w=%.1f(d3b=%.1f) p150w=%.1f(p150d=%.1f) m_ar=%.1f eg=%.1f\n",
                    plen, pr.wall_ms, pr.wall_tps, pr.sync_wall_ms,
                    pr.bd.att_qkv_ms, pr.bd.att_hsi_ms, pr.bd.att_sparse_ms, pr.bd.att_wo_ms, pr.bd.att_ar_ms,
                    pr.bd.moe_router_ms, pr.bd.moe_shared_ms, pr.bd.moe_tp3_hlut2_ms, pr.bd.moe_tp3_hlut3_ms,
                    pr.bd.moe_dev3_wait_ms, pr.bd.dev3_dev_busy_ms,
                    pr.bd.moe_p150_wait_ms, pr.bd.p150_dev_ms, pr.bd.moe_ar_ms, pr.bd.engram_wait_ms);
        std::fflush(stdout);
        pf_rows.push_back(pr);
    }

    // 2. Decode Burst Breakdown across 8K, 64K, and 256K active CSA2 contexts (using zero-stall GPU RTC!)
    const int dec_ctxs[3] = {8192, 65536, 262144};
    std::vector<DecodeRow> dec_rows;
    for (int clen : dec_ctxs) {
        rt.set_sync_timers(false);
        rt.seed_synthetic_context_for_profile(clen);
        rt.reset_stage_breakdown();

        // Append a short continuation prompt matching the exact seeded cached prefix?
        // Easier: build a prompt whose cached prefix matches via a tiny generate after seeding,
        // or append 16 tokens directly on top of the seeded context:
        // Since cached_tokens is populated by seed_synthetic_context_for_profile, let's drive
        // generate() with cached_tokens + 4 continuation tokens? Wait: cached_tokens is inside Impl.
        // Instead, seed_synthetic_context_for_profile leaves a deterministic token pattern (unit_toks[b&3]),
        // so construct the exact same prefix + 4 tokens to hit Tier-0 0.00ms LCP!
        const std::string seed_txt =
            "In heterogeneous AMD Instinct MI50 gfx906 and Tenstorrent Blackhole P150A inference for DeepSeek-V4.1-Flash, "
            "CSA2 sparse attention selects top-64 compressed blocks over 1M tokens while WHT128-HLUT3 executes 3-bit "
            "Gaussian Lloyd-Max tiles with zero scratch memory and zero PCIe system-scope RMW atomics. ";
        std::vector<int32_t> unit_toks;
        while (unit_toks.size() < 512u) {
            auto piece = rt.tokenizer().encode(seed_txt + "Segment " + std::to_string(unit_toks.size()) + ".\n", false);
            unit_toks.insert(unit_toks.end(), piece.begin(), piece.end());
        }
        unit_toks.resize(512u);

        const int n_pop_blks = (clen - 256) / 128;
        std::vector<int32_t> full_prompt;
        full_prompt.reserve(size_t(clen) + 32u);
        for (int b = 0; b < n_pop_blks; ++b) {
            full_prompt.insert(full_prompt.end(),
                               unit_toks.begin() + size_t(b & 3) * 128u,
                               unit_toks.begin() + size_t(b & 3) * 128u + 128u);
        }
        full_prompt.insert(full_prompt.end(), unit_toks.begin(), unit_toks.begin() + 256);
        auto tail_q = rt.tokenizer().encode("\nSummarize the key hardware invariants:\n", false);
        full_prompt.insert(full_prompt.end(), tail_q.begin(), tail_q.end());

        dsv41::Dsv41GenParams gp{};
        gp.max_new_tokens = decode_gen_tokens;
        gp.temperature = 0.0f;
        gp.enable_dspark = true;
        dsv41::Dsv41GenStats st{};
        (void)rt.generate(full_prompt, gp, [](int32_t) { return true; }, st);
        const auto bd = rt.query_stage_breakdown();

        DecodeRow dr{};
        dr.ctx_len      = clen;
        dr.gen_tokens   = st.completion_tokens;
        dr.steps        = st.dspark_steps;
        dr.decode_ms    = st.decode_ms;
        dr.tok_s        = (st.decode_ms > 0.0) ? (1000.0 * double(st.completion_tokens) / st.decode_ms) : 0.0;
        dr.tau          = (st.dspark_steps > 0) ? (1.0 + double(st.dspark_accepted) / double(st.dspark_steps)) : 1.0;
        dr.accept_pct   = (st.dspark_drafted > 0) ? (100.0 * double(st.dspark_accepted) / double(st.dspark_drafted)) : 0.0;
        dr.bb_ms        = st.dec_backbone_ms;
        dr.dspark_ms    = st.dec_dspark_ms;
        dr.sample_ms    = st.dec_sample_ms;
        dr.attn_ms      = st.t_attn_ms;
        dr.hsi_ms       = st.t_hsi_index_ms;
        dr.moe_ms       = st.t_mi50_moe_ms;
        dr.ar_ms        = st.t_allreduce_ms;
        dr.p150_wait_ms = st.t_p150_wait_ms;
        dr.p150_dev_ms  = st.dec_p150_dev_ms;
        dr.dev3_busy_ms = bd.dev3_dev_busy_ms;
        dr.engram_ms    = st.t_engram_wait_ms;

        std::printf("[PROFILE_W93][DECODE ctx=%6d] gen=%d in %.1fms (%.2f tok/s | steps=%llu tree=%llu tau=%.2f acc=%.1f%%) | "
                    "bb=%.1fms (attn=%.1f hsi=%.1f moe=%.1f p150w=%.1f[dev=%.1f] ar=%.1f eg=%.1f) | "
                    "dsp=%.1fms smp=%.1fms dev3_busy=%.1fms (reuse=%d)\n",
                    clen, dr.gen_tokens, dr.decode_ms, dr.tok_s,
                    static_cast<unsigned long long>(dr.steps),
                    static_cast<unsigned long long>(st.dspark_tree_steps), dr.tau, dr.accept_pct,
                    dr.bb_ms, dr.attn_ms, dr.hsi_ms, dr.moe_ms, dr.p150_wait_ms, dr.p150_dev_ms, dr.ar_ms, dr.engram_ms,
                    dr.dspark_ms, dr.sample_ms, dr.dev3_busy_ms, st.cached_prefix_tokens);
        std::fflush(stdout);
        dec_rows.push_back(dr);
    }

    // Write clean zero-PII Markdown breakdown report
    std::error_code ec;
    const auto parent = std::filesystem::path(out_md).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent, ec);
    std::ofstream f(out_md, std::ios::trunc);
    f << "# Wave 9.3 Phase A — Cycle-Exact Per-Stage Bottleneck Profile (" << label << ")\n\n";
    f << "- **Topology**: `4× MI50 (C7:00.0 32G, 88:00.0 32G, 03:00.0 32G, 85:00.0 16G = 112 GiB)` + `2× Blackhole P150A (64 GiB)` = **`176 GiB`**\n";
    f << "- **HBM2 Free @ `max_seq_len = " << max_seq_len << "` + `ViT`**: `tp0 = "
      << std::fixed << std::setprecision(1) << mem.free_mib[0] << " MiB`, `tp1 = " << mem.free_mib[1]
      << " MiB`, `tp2 = " << mem.free_mib[2] << " MiB`, `dev3 = " << mem.dev3_free_mib << " / " << mem.dev3_total_mib << " MiB`\n\n";

    f << "## 1. Cold Prefill Breakdown (`reuse = 0`, `M = 512, 2048, 4096`)\n\n";
    f << "| `M` | Wall (`ms`) | **Wall `tok/s`** | Sync Total (`ms`) | QKV/Proj (`ms`) | HSI (`ms`) | CSA2+SWA (`ms`) | Wo (`ms`) | Attn AR (`ms`) | Router (`ms`) | Shared (`ms`) | TP3 HLUT2 (`ms`) | TP3 HLUT3 (`ms`) | `dev3` Wait / Busy (`ms`) | `P150A` Wait / Dev (`ms`) | MoE AR (`ms`) | Engram (`ms`) |\n";
    f << "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
    for (const auto& r : pf_rows) {
        const double denom = std::max(1.0, r.sync_wall_ms);
        auto pct = [&](double v) {
            std::ostringstream os;
            os << std::fixed << std::setprecision(1) << v << " (" << std::setprecision(0) << (100.0 * v / denom) << "%)";
            return os.str();
        };
        f << "| `" << r.tokens << "` | `" << std::setprecision(1) << r.wall_ms << "` | **`" << r.wall_tps << "`** | `"
          << r.sync_wall_ms << "` | " << pct(r.bd.att_qkv_ms) << " | " << pct(r.bd.att_hsi_ms)
          << " | " << pct(r.bd.att_sparse_ms) << " | " << pct(r.bd.att_wo_ms) << " | " << pct(r.bd.att_ar_ms)
          << " | " << pct(r.bd.moe_router_ms) << " | " << pct(r.bd.moe_shared_ms)
          << " | " << pct(r.bd.moe_tp3_hlut2_ms) << " | " << pct(r.bd.moe_tp3_hlut3_ms)
          << " | " << std::setprecision(1) << r.bd.moe_dev3_wait_ms << " / " << r.bd.dev3_dev_busy_ms
          << " | " << r.bd.moe_p150_wait_ms << " / " << r.bd.p150_dev_ms
          << " | " << pct(r.bd.moe_ar_ms) << " | " << pct(r.bd.engram_wait_ms) << " |\n";
    }

    f << "\n## 2. Decode Burst Breakdown (`DSpark` Tree Speculative Decode @ `8K`, `64K`, `256K`)\n\n";
    f << "| Active Context | Gen Toks | Steps | `tau` | Accept `%` | **Decode `tok/s`** | `T_step` (`ms`) | Backbone (`ms`) | Attn (`ms`) | HSI (`ms`) | MoE (`ms`) | `P150A` Wait / Dev (`ms`) | AllReduce (`ms`) | `DSpark` (`ms`) | `dev3` Busy (`ms`) | Engram (`ms`) |\n";
    f << "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
    for (const auto& d : dec_rows) {
        const double tstep = (d.steps > 0) ? (d.decode_ms / double(d.steps)) : d.decode_ms;
        f << "| `" << d.ctx_len << "` | " << d.gen_tokens << " | " << d.steps
          << " | `" << std::setprecision(2) << d.tau << "` | `" << std::setprecision(1) << d.accept_pct << "%`"
          << " | **`" << std::setprecision(2) << d.tok_s << "`** | `" << tstep << "`"
          << " | " << std::setprecision(1) << d.bb_ms << " | " << d.attn_ms << " | " << d.hsi_ms
          << " | " << d.moe_ms << " | " << d.p150_wait_ms << " / " << d.p150_dev_ms
          << " | " << d.ar_ms << " | " << d.dspark_ms << " | " << d.dev3_busy_ms << " | " << d.engram_ms << " |\n";
    }
    f.close();
    std::printf("[PROFILE_W93] Wrote breakdown report -> %s\n", out_md.c_str());
    return 0;
}
