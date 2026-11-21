#pragma once
// Pure-C++17 Heterogeneous Inference Runtime for DeepSeek-V4.1-Flash-1017 (Speed Wave 3)
//   - 3x AMD Instinct MI50 (96 GiB HBM2, TP3): Attention (CSA2 + Strict CED), mHC, Shared Experts,
//     Routed HLUT2 Experts [0..215] + MTP [0..127], 3-Stage DSpark Speculative Draft (M=6)
//   - 2x Tenstorrent Blackhole P150A (64 GiB GDDR6, TP2 over /dsv41_p150_ipc): Routed HLUT2 Experts [216..383]
//   - EPYC 384 GiB DDR4 + Dual-NVMe: Pre-Faulted 189 GiB Engram + RewindBank + 128-tok RadixBlockCache + SWA Replay
// Zero Python anywhere.

#include "dsv41_engram_cpu.hpp"
#include "dsv41_mi50_kernels.hpp"
#include "dsv41_tokenizer_dsml.hpp"
#include "dsv41_vit.hpp"
#include "../p150/dsv41_p150_shm.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace dsv41 {

struct Dsv41GenParams {
    int max_new_tokens = 8192;
    float temperature = 1.0f;
    float top_p = 0.95f;
    float repetition_penalty = 1.0f;
    bool enable_dspark = true;
    bool enable_think_loop_guard = true;
    uint64_t seed = 0;
};

struct Dsv41GenStats {
    int prompt_tokens = 0;
    int cached_prefix_tokens = 0;
    int radix_hit_tokens = 0;
    int nvme_hit_tokens = 0;
    int swa_replay_tokens = 0;
    int prefilled_tokens = 0;
    int completion_tokens = 0;
    int images_encoded = 0;
    double vit_encode_ms = 0.0;
    double nvme_hydrate_ms = 0.0;
    double prefix_reuse_ms = 0.0;
    double fresh_prefill_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double dec_dspark_ms = 0.0;
    double dec_backbone_ms = 0.0;
    double dec_p150_dev_ms = 0.0;
    double dec_sample_ms = 0.0;
    double t_attn_ms = 0.0;
    double t_hsi_index_ms = 0.0;
    double t_mi50_moe_ms = 0.0;
    double t_p150_wait_ms = 0.0;
    double t_allreduce_ms = 0.0;
    double t_engram_wait_ms = 0.0;
    double overlap_saved_ms = 0.0;
    uint64_t dspark_steps = 0;
    uint64_t dspark_drafted = 0;
    uint64_t dspark_accepted = 0;
    uint64_t dspark_tree_steps = 0;
    uint64_t dspark_tree_accepted = 0;
    uint64_t dspark_tree_dyn8_steps = 0;
    uint64_t dspark_tree_dyn8_accepted = 0;
    uint64_t dspark_tree16_steps = 0;
    uint64_t dspark_tree16_accepted = 0;
    bool think_guard_triggered = false;
    uint32_t think_guard_period = 0;
    uint32_t think_guard_rewound = 0;
};

using TokenStreamCallback = std::function<bool(int32_t token_id)>;

enum class Dsv41ProgressPhase : uint8_t {
    PrefixDone = 0,
    PrefillChunk = 1,
};

struct Dsv41ProgressUpdate {
    Dsv41ProgressPhase phase = Dsv41ProgressPhase::PrefixDone;
    int prompt_tokens = 0;
    int cached_tokens = 0;
    int fresh_tokens_done = 0;
    int chunk_tokens = 0;
    double chunk_ms = 0.0;
    int64_t begin_ns = 0;
    int64_t end_ns = 0;
};

using ProgressCallback = std::function<bool(const Dsv41ProgressUpdate&)>;

struct Dsv41GpuMemMargin {
    double free_mib[3]{0.0, 0.0, 0.0};
    double total_mib[3]{0.0, 0.0, 0.0};
    double dev3_free_mib = 0.0;
    double dev3_total_mib = 0.0;
    bool dev3_active = false;
    double dev4_free_mib = 0.0;
    double dev4_total_mib = 0.0;
    bool dev4_active = false;
    bool pointer_audit_64bit_ok = false;
    size_t max_comp_slots[4]{};
    size_t hot_comp_slots[4]{};
    size_t ext_comp_slots[4]{};
};

struct Dsv41StageBreakdown {
    double att_qkv_ms = 0.0;
    double att_hsi_ms = 0.0;
    double att_sparse_ms = 0.0;
    double att_wo_ms = 0.0;
    double att_ar_ms = 0.0;
    double moe_router_ms = 0.0;
    double moe_shared_ms = 0.0;
    double moe_tp3_hlut2_ms = 0.0;
    double moe_tp3_hlut3_ms = 0.0;
    double moe_dev3_wait_ms = 0.0;
    double moe_dev4_wait_ms = 0.0;
    double moe_p150_wait_ms = 0.0;
    double moe_ar_ms = 0.0;
    double mhc_ms = 0.0;
    double commit_ms = 0.0;
    double engram_wait_ms = 0.0;
    double p150_dev_ms = 0.0;
    double dev3_dev_busy_ms = 0.0;
    uint64_t dev3_dispatches = 0;
    double dev4_dev_busy_ms = 0.0;
    uint64_t dev4_dispatches = 0;
};

struct Dsv411MDecodeBurstRow {
    int burst_idx = 0;
    int pos_start = 0;
    int pos_end = 0;
    int accepted_tokens = 0;
    double step_ms = 0.0;
    double tok_s = 0.0;
    double attn_ms = 0.0;
    double hsi_ms = 0.0;
    double moe_ms = 0.0;
    double p150w_ms = 0.0;
    double ar_ms = 0.0;
    double dsp_ms = 0.0;
    double smp_ms = 0.0;
    double egw_ms = 0.0;
};

struct Dsv411MGateReceipt {
    Dsv41GpuMemMargin mem_margin{};
    bool engram_prefetch_overlap_m512_ok = false;
    bool engram_prefetch_overlap_m1024_ok = false;
    double engram_prefetch_wait_m512_ms = 0.0;
    double engram_prefetch_wait_m1024_ms = 0.0;
    double populate_1m_s = 0.0;
    double prefill_tail_ms = 0.0;
    int prefill_tail_tokens = 0;
    uint64_t nonfinite_count = 0;
    bool int32_overflow_free = false;
    double hsi_top64_overlap_pct = 0.0;
    bool needle_128_retrieved = false;
    bool needle_524288_retrieved = false;
    bool needle_900000_retrieved = false;
    double csa2_vdot2_vs_fp32_cos = 0.0;
    double csa2_vdot2_vs_fp32_max_abs = 0.0;
    int decode_bursts_completed = 0;
    int decode_final_pos = 0;
    int decode_total_emitted = 0;
    double decode_mean_step_ms = 0.0;
    double decode_agg_tok_s = 0.0;
    std::vector<Dsv411MDecodeBurstRow> bursts;
    int nvme_blocks_flushed = 0;
    size_t nvme_block_bytes = 0;
    size_t nvme_swa_disk_bytes = 0;
    bool nvme_crc32c_ok = false;
    double nvme_cold_hydrate_ms = 0.0;
    int nvme_swa_replay_tokens = 0;
    double nvme_enc_cos = 0.0;
    double nvme_enc_max_abs = 0.0;
    double nvme_dec_cos = 0.0;
    double nvme_dec_max_abs = 0.0;
};

class Dsv41Runtime {
public:
    Dsv41Runtime();
    ~Dsv41Runtime();

    Dsv41Runtime(const Dsv41Runtime&) = delete;
    Dsv41Runtime& operator=(const Dsv41Runtime&) = delete;

    void init(
        const std::string& mi50_packed_root,
        const std::string& tokenizer_json_path,
        const std::string& p150_shm_name = "/dsv41_p150_ipc",
        int max_seq_len = 524288,
        const std::string& rarebal_overlay_path = "",
        const std::string& capture_rarebal_dir = "",
        const std::string& vit_weights_path = "");

    bool ready() const noexcept { return ready_; }
    bool vision_ready() const noexcept;
    const Dsv41Tokenizer& tokenizer() const { return tokenizer_; }

    void reset_session_cache();
    void set_sync_timers(bool enable);
    void set_dual_stream_overlap(bool enable);
    void set_fused_mhc_norm(bool enable);
    bool fused_mhc_norm() const noexcept;
    void set_fused_dspark_epilogue(bool enable);
    bool fused_dspark_epilogue() const noexcept;
    void set_prefill_superchunk_m(int super_m);
    void set_dspark_tree_width(int width_mode);  // 0 = auto, 8 = fixed M=8, 16 = fixed M=16
    void set_dspark_tree16_max_context(int max_ctx);
    void set_dspark_static_m8(bool enable);
    void set_dspark_dyn8_max_depth(int max_depth);
    void set_expert_perm_overlay(const std::string& overlay_path);
    // Opt-in Wave-9 HLUT3 MTP overlay: load 400 B/tile HLUT3 tile planes for DSpark MTP blocks
    // 40..42 only when --hlut3-mtp-overlay <path> is supplied. When unset, the default production
    // 2.125 bpw HLUT2 MTP footprint is byte-identical and 0 B extra HBM2 is allocated.
    void set_hlut3_mtp_overlay(const std::string& overlay_path);
    // Wave 9.2 SB-M4-P2-176G asymmetric 4x MI50 + 2x P150A WHT128-HLUT3 overlays:
    //   --hlut3-dev3-overlay: 384 unsharded MTP + up to 720 unsharded backbone experts on hlut3_dev=3 (0000:85:00.0)
    //   --hlut3-tp3-overlay : 600+ TP3-sharded backbone experts promoted in-place on tp=0,1,2 (C7/88/03)
    void set_hlut3_dev3_overlay(const std::string& overlay_path, int dev_id = 3);
    void set_hlut3_dev4_overlay(const std::string& overlay_path, int dev_id = 4);
    void set_hlut3_tp3_overlay(const std::string& overlay_path);
    void set_rocblas_prefill(bool enable);
    void set_rocblas_expert_thresh(int thresh);
    void set_rocblas_hlut2_thresh(int thresh);
    void set_dspark_tree8_conf_thresh(float thresh);
    void set_dspark_chain_max_b(int max_b);
    void set_csa2_shared_tree8(bool enable);
    void set_nvme_kv_cache(const std::string& cache_dir, double max_gb = 32.0);
    void flush_nvme_kv_cache();
    void clear_all_caches();
    std::vector<float> last_logits_row0() const;
    bool capture_rarebal_corpus(const std::vector<int32_t>& corpus_ids);
    bool capture_rarebal_multimodal(
        const std::vector<int32_t>& raw_prompt_ids,
        const std::vector<Dsv41ImageSource>& images);
    Dsv41GpuMemMargin query_gpu_mem_margin();
    Dsv41StageBreakdown query_stage_breakdown() const;
    void reset_stage_breakdown();
    void seed_synthetic_context_for_profile(int ctx_len);
    Dsv411MGateReceipt run_wave8_5_1m_e2e_gate(const std::string& nvme_tmp_dir);

    // Thread-safe generation with multi-tier KV prefix reuse, strict CED prefill, and 3-stage DSpark (M=6)
    std::vector<int32_t> generate(
        const std::vector<int32_t>& prompt_ids,
        const Dsv41GenParams& params,
        TokenStreamCallback on_token,
        Dsv41GenStats& out_stats,
        ProgressCallback on_progress = nullptr,
        const std::vector<Dsv41ImageSource>* images = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Dsv41Tokenizer tokenizer_;
    std::mutex gen_mtx_;
    bool ready_ = false;
};

}  // namespace dsv41
