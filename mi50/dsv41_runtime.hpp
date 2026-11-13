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
    int swa_replay_tokens = 0;
    int prefilled_tokens = 0;
    int completion_tokens = 0;
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
    uint64_t dspark_steps = 0;
    uint64_t dspark_drafted = 0;
    uint64_t dspark_accepted = 0;
    uint64_t dspark_tree_steps = 0;
    uint64_t dspark_tree_accepted = 0;
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
        const std::string& capture_rarebal_dir = "");

    bool ready() const noexcept { return ready_; }
    const Dsv41Tokenizer& tokenizer() const { return tokenizer_; }

    void reset_session_cache();
    void set_sync_timers(bool enable);
    void clear_all_caches();

    // Thread-safe generation with multi-tier KV prefix reuse, strict CED prefill, and 3-stage DSpark (M=6)
    std::vector<int32_t> generate(
        const std::vector<int32_t>& prompt_ids,
        const Dsv41GenParams& params,
        TokenStreamCallback on_token,
        Dsv41GenStats& out_stats,
        ProgressCallback on_progress = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Dsv41Tokenizer tokenizer_;
    std::mutex gen_mtx_;
    bool ready_ = false;
};

}  // namespace dsv41
