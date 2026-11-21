// Wave 10 Phase C: Fine-Grained Sub-Stage Bottleneck Profiler for SB-M5-P2-192G Dual-Switchbox
// Profiles on live 5x MI50 (C7:00.0, 88:00.0, 05:00.0, 85:00.0, 08:00.0) + 2x Blackhole P150A:
//   1. Interactive Small-Tail Prefill: M = 256 fresh tokens on top of 340K (348,160) cached KV
//   2. Cold Prefill (M = 2048, 4096 fresh tokens, reuse = 0): wall tok/s + sync_timers breakdown
//   3. Decode Bursts (DSpark M_verify=7 + 3x MTP, 64 gen tokens) at 8K, 64K, 160K, 340K active CSA2 context
// Reports BOTH dev3 (Switch #1) and dev4 (Switch #2) busy/wait times + all 5 MI50 free HBM2 margins.
// 100% Pure C++17 / HIP — Zero Python Anywhere.

#include "../mi50/dsv41_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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
    std::string tag;
    int ctx_prefix = 0;
    int tokens = 0;
    int cached_hit = 0;
    double wall_ms = 0.0;
    double wall_tps = 0.0;
    double sync_wall_ms = 0.0;
    dsv41::Dsv41StageBreakdown bd{};
};

struct DecodeRow {
    int ctx_len = 0;
    int gen_tokens = 0;
    uint64_t steps = 0;
    uint64_t tree_steps = 0;
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
    double dev4_busy_ms = 0.0;
    double engram_ms = 0.0;
    int cached_prefix = 0;
};

std::vector<int32_t> build_fresh_prompt(const dsv41::Dsv41Tokenizer& tok, int target_tokens, int nonce) {
    std::string body = "Session w10-profile-nonce-" + std::to_string(nonce) +
                       ": Analyze dual-switchbox 5x AMD Instinct MI50 gfx906 and 2x Blackhole P150A scheduling.\n";
    int sec = 1;
    std::vector<int32_t> ids = tok.encode(body, true);
    while (int(ids.size()) < target_tokens) {
        std::string chunk =
            "Section " + std::to_string(sec++) + " (nonce=" + std::to_string(nonce) + "): "
            "In dual-switchbox SB-M5-P2-192G inference, 384 DSpark MTP experts and 720 minimax-balanced backbone "
            "experts execute unsharded WHT128-HLUT3 tiles on hlut3_dev=3 (Switch #1) concurrently with 1,080 "
            "minimax-balanced backbone experts on hlut3_dev=4 (Switch #2), 1,160 TP3 HLUT3 experts, compacted "
            "TP3 HLUT2, and 6,720 P150A experts over 4x 800G FABRIC_1D with zero PCIe system-scope RMW atomics.\n";
        auto sub = tok.encode(chunk, false);
        ids.insert(ids.end(), sub.begin(), sub.end());
    }
    ids.resize(size_t(target_tokens));
    return ids;
}

std::vector<int32_t> build_seeded_prefix(const dsv41::Dsv41Tokenizer& tok, int ctx_len) {
    const std::string seed_txt =
        "In heterogeneous AMD Instinct MI50 gfx906 and Tenstorrent Blackhole P150A inference for DeepSeek-V4.1-Flash, "
        "CSA2 sparse attention selects top-64 compressed blocks over 1M tokens while WHT128-HLUT3 executes 3-bit "
        "Gaussian Lloyd-Max tiles with zero scratch memory and zero PCIe system-scope RMW atomics. ";
    std::vector<int32_t> unit_toks;
    while (unit_toks.size() < 512u) {
        auto piece = tok.encode(seed_txt + "Segment " + std::to_string(unit_toks.size()) + ".\n", false);
        unit_toks.insert(unit_toks.end(), piece.begin(), piece.end());
    }
    unit_toks.resize(512u);

    const int n_pop_blks = (ctx_len - 256) / 128;
    std::vector<int32_t> prefix;
    prefix.reserve(size_t(ctx_len));
    for (int b = 0; b < n_pop_blks; ++b) {
        prefix.insert(prefix.end(),
                      unit_toks.begin() + size_t(b & 3) * 128u,
                      unit_toks.begin() + size_t(b & 3) * 128u + 128u);
    }
    prefix.insert(prefix.end(), unit_toks.begin(), unit_toks.begin() + 256);
    return prefix;
}

}  // namespace

int main(int argc, char** argv) {
    ::setenv("OMP_WAIT_POLICY", "PASSIVE", 1);
    ::setenv("GPU_MAX_HW_QUEUES", "8", 0);

    const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                     ? std::string(std::getenv("DSV41_ROOT"))
                                     : std::string(".");
    const std::string ov_dir   = (std::getenv("DSV41_WAVE10_OVERLAY_DIR") != nullptr)
                                     ? std::string(std::getenv("DSV41_WAVE10_OVERLAY_DIR"))
                                     : (root_env + "/wave10_overlays");
    std::string overlay       = root_env + "/packed_rarebal_overlay_wave8_5/hot_experts_wave8_5.safetensors";
    std::string dev3_overlay  = ov_dir + "/wave10_dev3_hlut3.safetensors";
    std::string dev4_overlay  = ov_dir + "/wave10_dev4_hlut3.safetensors";
    std::string tp3_overlay   = ov_dir + "/wave10_tp3_hlut3.safetensors";
    std::string out_md        = "docs/wave10_profile.md";
    std::string label         = "Wave 10 SB-M5-P2-192G Dual-Switchbox";
    int max_seq_len           = 1048576;
    int superchunk_m          = 2048;
    int decode_gen_tokens     = 64;
    bool rocblas_prefill      = true;
    int rocblas_expert_thresh = 20;
    int rocblas_hlut2_thresh  = 24;
    float tree8_conf_thresh   = 0.72f;
    int chain_max_b           = 5;
    bool skip_decode          = false;
    bool skip_prefill         = false;
    bool link_check           = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--link-check") link_check = true;
        else if (a == "--overlay" && i + 1 < argc) overlay = argv[++i];
        else if (a == "--hlut3-dev3-overlay" && i + 1 < argc) dev3_overlay = argv[++i];
        else if (a == "--hlut3-dev4-overlay" && i + 1 < argc) dev4_overlay = argv[++i];
        else if (a == "--no-hlut3-dev4-overlay") dev4_overlay.clear();
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
        else if (a == "--skip-decode") skip_decode = true;
        else if (a == "--skip-prefill") skip_prefill = true;
    }

    if (link_check) {
        std::printf("[PROFILE_W10] Link check OK.\n");
        return 0;
    }

    dsv41::Dsv41Runtime rt;
    rt.set_expert_perm_overlay(overlay);
    if (!dev3_overlay.empty()) rt.set_hlut3_dev3_overlay(dev3_overlay, 3);
    if (!dev4_overlay.empty()) rt.set_hlut3_dev4_overlay(dev4_overlay, 4);
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
    std::printf("[PROFILE_W10] Free HBM2 @ 1M+ViT: tp0=%.1f tp1=%.1f tp2=%.1f dev3=%.1f/%.1f dev4=%.1f/%.1f MiB\n",
                mem.free_mib[0], mem.free_mib[1], mem.free_mib[2],
                mem.dev3_free_mib, mem.dev3_total_mib,
                mem.dev4_free_mib, mem.dev4_total_mib);
    std::fflush(stdout);

    std::vector<PrefillRow> pf_rows;
    int nonce_seq = 1000;

    if (!skip_prefill) {
        // 1A. Interactive Small-Tail Prefill: M = 256 fresh tokens on top of 340K (348,160) cached KV!
        {
            const int ctx_340k = 348160;
            const int tail_m = 256;
            PrefillRow pr{};
            pr.tag = "M=256 @ 340K KV";
            pr.ctx_prefix = ctx_340k;
            pr.tokens = tail_m;

            dsv41::Dsv41GenParams gp{};
            gp.max_new_tokens = 0;
            gp.temperature = 0.0f;
            gp.enable_dspark = false;
            const auto base_prefix = build_seeded_prefix(rt.tokenizer(), ctx_340k);

            rt.set_sync_timers(false);
            rt.seed_synthetic_context_for_profile(ctx_340k);
            auto tail_a = build_fresh_prompt(rt.tokenizer(), tail_m, nonce_seq++);
            std::vector<int32_t> full_a = base_prefix;
            full_a.insert(full_a.end(), tail_a.begin(), tail_a.end());
            dsv41::Dsv41GenStats st_a{};
            (void)rt.generate(full_a, gp, [](int32_t) { return true; }, st_a);
            pr.cached_hit = st_a.cached_prefix_tokens;
            pr.wall_ms = st_a.fresh_prefill_ms;
            pr.wall_tps = (pr.wall_ms > 0.0) ? (1000.0 * double(st_a.prefilled_tokens) / pr.wall_ms) : 0.0;

            rt.set_sync_timers(false);
            rt.seed_synthetic_context_for_profile(ctx_340k);
            rt.set_sync_timers(true);
            rt.reset_stage_breakdown();
            auto tail_b = build_fresh_prompt(rt.tokenizer(), tail_m, nonce_seq++);
            std::vector<int32_t> full_b = base_prefix;
            full_b.insert(full_b.end(), tail_b.begin(), tail_b.end());
            dsv41::Dsv41GenStats st_b{};
            (void)rt.generate(full_b, gp, [](int32_t) { return true; }, st_b);
            pr.sync_wall_ms = st_b.fresh_prefill_ms;
            pr.bd = rt.query_stage_breakdown();
            rt.set_sync_timers(false);

            std::printf("[PROFILE_W10][TAIL M=%4d @ 340K reuse=%d] wall=%.1fms (%.1f tok/s) | sync=%.1fms | "
                        "qkv=%.1f hsi=%.1f sp=%.1f wo=%.1f a_ar=%.1f | "
                        "rtr=%.1f sh=%.1f hl2=%.1f hl3=%.1f d3w=%.1f(d3b=%.1f) d4w=%.1f(d4b=%.1f) p150w=%.1f(p150d=%.1f) m_ar=%.1f eg=%.1f\n",
                        tail_m, pr.cached_hit, pr.wall_ms, pr.wall_tps, pr.sync_wall_ms,
                        pr.bd.att_qkv_ms, pr.bd.att_hsi_ms, pr.bd.att_sparse_ms, pr.bd.att_wo_ms, pr.bd.att_ar_ms,
                        pr.bd.moe_router_ms, pr.bd.moe_shared_ms, pr.bd.moe_tp3_hlut2_ms, pr.bd.moe_tp3_hlut3_ms,
                        pr.bd.moe_dev3_wait_ms, pr.bd.dev3_dev_busy_ms,
                        pr.bd.moe_dev4_wait_ms, pr.bd.dev4_dev_busy_ms,
                        pr.bd.moe_p150_wait_ms, pr.bd.p150_dev_ms, pr.bd.moe_ar_ms, pr.bd.engram_wait_ms);
            std::fflush(stdout);
            pf_rows.push_back(pr);
        }

        // 1B. Cold Prefill (reuse = 0, M = 2048, 4096)
        const int prefill_lens[2] = {2048, 4096};
        for (int plen : prefill_lens) {
            PrefillRow pr{};
            pr.tag = "Cold M=" + std::to_string(plen);
            pr.ctx_prefix = 0;
            pr.tokens = plen;
            dsv41::Dsv41GenParams gp{};
            gp.max_new_tokens = 0;
            gp.temperature = 0.0f;
            gp.enable_dspark = false;

            rt.clear_all_caches();
            rt.set_sync_timers(false);
            auto ids_a = build_fresh_prompt(rt.tokenizer(), plen, nonce_seq++);
            dsv41::Dsv41GenStats st_a{};
            (void)rt.generate(ids_a, gp, [](int32_t) { return true; }, st_a);
            pr.cached_hit = st_a.cached_prefix_tokens;
            pr.wall_ms  = st_a.fresh_prefill_ms;
            pr.wall_tps = (pr.wall_ms > 0.0) ? (1000.0 * double(st_a.prefilled_tokens) / pr.wall_ms) : 0.0;

            rt.clear_all_caches();
            rt.set_sync_timers(true);
            rt.reset_stage_breakdown();
            auto ids_b = build_fresh_prompt(rt.tokenizer(), plen, nonce_seq++);
            dsv41::Dsv41GenStats st_b{};
            (void)rt.generate(ids_b, gp, [](int32_t) { return true; }, st_b);
            pr.sync_wall_ms = st_b.fresh_prefill_ms;
            pr.bd = rt.query_stage_breakdown();
            rt.set_sync_timers(false);

            std::printf("[PROFILE_W10][COLD M=%4d] wall=%.1fms (%.1f tok/s) | sync=%.1fms | "
                        "qkv=%.1f hsi=%.1f sp=%.1f wo=%.1f a_ar=%.1f | "
                        "rtr=%.1f sh=%.1f hl2=%.1f hl3=%.1f d3w=%.1f(d3b=%.1f) d4w=%.1f(d4b=%.1f) p150w=%.1f(p150d=%.1f) m_ar=%.1f eg=%.1f\n",
                        plen, pr.wall_ms, pr.wall_tps, pr.sync_wall_ms,
                        pr.bd.att_qkv_ms, pr.bd.att_hsi_ms, pr.bd.att_sparse_ms, pr.bd.att_wo_ms, pr.bd.att_ar_ms,
                        pr.bd.moe_router_ms, pr.bd.moe_shared_ms, pr.bd.moe_tp3_hlut2_ms, pr.bd.moe_tp3_hlut3_ms,
                        pr.bd.moe_dev3_wait_ms, pr.bd.dev3_dev_busy_ms,
                        pr.bd.moe_dev4_wait_ms, pr.bd.dev4_dev_busy_ms,
                        pr.bd.moe_p150_wait_ms, pr.bd.p150_dev_ms, pr.bd.moe_ar_ms, pr.bd.engram_wait_ms);
            std::fflush(stdout);
            pf_rows.push_back(pr);
        }
    }

    // 2. Decode Burst Breakdown across 8K, 64K, 160K, 340K, 544K active CSA2 contexts
    std::vector<DecodeRow> dec_rows;
    if (!skip_decode) {
        const int dec_ctxs[5] = {8192, 65536, 163840, 348160, 557056};
        for (int clen : dec_ctxs) {
            rt.set_sync_timers(false);
            rt.seed_synthetic_context_for_profile(clen);
            rt.reset_stage_breakdown();

            std::vector<int32_t> full_prompt = build_seeded_prefix(rt.tokenizer(), clen);
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
            dr.ctx_len       = clen;
            dr.gen_tokens    = st.completion_tokens;
            dr.steps         = st.dspark_steps;
            dr.tree_steps    = st.dspark_tree_steps;
            dr.decode_ms     = st.decode_ms;
            dr.tok_s         = (st.decode_ms > 0.0) ? (1000.0 * double(st.completion_tokens) / st.decode_ms) : 0.0;
            dr.tau           = (st.dspark_steps > 0) ? (1.0 + double(st.dspark_accepted) / double(st.dspark_steps)) : 1.0;
            dr.accept_pct    = (st.dspark_drafted > 0) ? (100.0 * double(st.dspark_accepted) / double(st.dspark_drafted)) : 0.0;
            dr.bb_ms         = st.dec_backbone_ms;
            dr.dspark_ms     = st.dec_dspark_ms;
            dr.sample_ms     = st.dec_sample_ms;
            dr.attn_ms       = st.t_attn_ms;
            dr.hsi_ms        = st.t_hsi_index_ms;
            dr.moe_ms        = st.t_mi50_moe_ms;
            dr.ar_ms         = st.t_allreduce_ms;
            dr.p150_wait_ms  = st.t_p150_wait_ms;
            dr.p150_dev_ms   = st.dec_p150_dev_ms;
            dr.dev3_busy_ms  = bd.dev3_dev_busy_ms;
            dr.dev4_busy_ms  = bd.dev4_dev_busy_ms;
            dr.engram_ms     = st.t_engram_wait_ms;
            dr.cached_prefix = st.cached_prefix_tokens;

            const double tstep = (dr.steps > 0) ? (dr.decode_ms / double(dr.steps)) : dr.decode_ms;
            const double inv_s = (dr.steps > 0) ? (1.0 / double(dr.steps)) : 1.0;
            std::printf("[PROFILE_W10][DECODE ctx=%6d] gen=%d in %.1fms (%.2f tok/s | steps=%llu T_step=%.2fms tau=%.2f acc=%.1f%%) | "
                        "per-step: bb=%.2f (attn=%.2f hsi=%.2f moe=%.2f p150w=%.2f[dev=%.2f] ar=%.2f eg=%.2f) "
                        "mtp=%.2f smp=%.2f dev3=%.2f dev4=%.2f (reuse=%d)\n",
                        clen, dr.gen_tokens, dr.decode_ms, dr.tok_s,
                        static_cast<unsigned long long>(dr.steps), tstep, dr.tau, dr.accept_pct,
                        dr.bb_ms * inv_s, dr.attn_ms * inv_s, dr.hsi_ms * inv_s, dr.moe_ms * inv_s,
                        dr.p150_wait_ms * inv_s, dr.p150_dev_ms * inv_s, dr.ar_ms * inv_s, dr.engram_ms * inv_s,
                        dr.dspark_ms * inv_s, dr.sample_ms * inv_s, dr.dev3_busy_ms * inv_s, dr.dev4_busy_ms * inv_s,
                        dr.cached_prefix);
            std::fflush(stdout);
            dec_rows.push_back(dr);
        }
    }

    std::error_code ec;
    const auto parent = std::filesystem::path(out_md).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent, ec);
    std::ofstream f(out_md, std::ios::trunc);
    f << "# Wave 10 SB-M5-P2-192G — Fine-Grained Sub-Stage Profile (" << label << ")\n\n"
      << "- **HBM2 Free @ 1M + ViT**: `tp0 = " << std::fixed << std::setprecision(1) << mem.free_mib[0]
      << " MiB`, `tp1 = " << mem.free_mib[1] << " MiB`, `tp2 = " << mem.free_mib[2]
      << " MiB`, `dev3 = " << mem.dev3_free_mib << " MiB`, `dev4 = " << mem.dev4_free_mib << " MiB`\n\n";

    if (!pf_rows.empty()) {
        f << "## 1. Prefill & Tool-Tail Breakdown\n\n"
          << "| Scenario | Wall ms | Wall tok/s | qkv | hsi | sparse | wo | a_ar | rtr | sh | hl2 | hl3 | d3w(d3b) | d4w(d4b) | p150w(p150d) | m_ar | eg |\n"
          << "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
        for (const auto& pr : pf_rows) {
            f << "| " << pr.tag << " | " << pr.wall_ms << " | **" << pr.wall_tps << "** | "
              << pr.bd.att_qkv_ms << " | " << pr.bd.att_hsi_ms << " | " << pr.bd.att_sparse_ms << " | "
              << pr.bd.att_wo_ms << " | " << pr.bd.att_ar_ms << " | " << pr.bd.moe_router_ms << " | "
              << pr.bd.moe_shared_ms << " | " << pr.bd.moe_tp3_hlut2_ms << " | " << pr.bd.moe_tp3_hlut3_ms << " | "
              << pr.bd.moe_dev3_wait_ms << "(" << pr.bd.dev3_dev_busy_ms << ") | "
              << pr.bd.moe_dev4_wait_ms << "(" << pr.bd.dev4_dev_busy_ms << ") | "
              << pr.bd.moe_p150_wait_ms << "(" << pr.bd.p150_dev_ms << ") | "
              << pr.bd.moe_ar_ms << " | " << pr.bd.engram_wait_ms << " |\n";
        }
        f << "\n";
    }
    if (!dec_rows.empty()) {
        f << "## 2. Decode Breakdown (`DSpark` Dynamic Tree / Chain)\n\n"
          << "| Active Context | tok/s | T_step ms | tau | acc% | attn | hsi | moe | p150w(dev) | ar | mtp | dev3 | dev4 |\n"
          << "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
        for (const auto& dr : dec_rows) {
            const double tstep = (dr.steps > 0) ? (dr.decode_ms / double(dr.steps)) : dr.decode_ms;
            const double inv_s = (dr.steps > 0) ? (1.0 / double(dr.steps)) : 1.0;
            f << "| " << dr.ctx_len << " | **" << std::setprecision(2) << dr.tok_s << "** | "
              << tstep << " | " << dr.tau << " | " << std::setprecision(1) << dr.accept_pct << "% | "
              << std::setprecision(2) << dr.attn_ms * inv_s << " | " << dr.hsi_ms * inv_s << " | "
              << dr.moe_ms * inv_s << " | " << dr.p150_wait_ms * inv_s << "(" << dr.p150_dev_ms * inv_s << ") | "
              << dr.ar_ms * inv_s << " | " << dr.dspark_ms * inv_s << " | "
              << dr.dev3_busy_ms * inv_s << " | " << dr.dev4_busy_ms * inv_s << " |\n";
        }
    }
    std::printf("[PROFILE_W10] Wrote breakdown -> %s\n", out_md.c_str());
    return 0;
}
