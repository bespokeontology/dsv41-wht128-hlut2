#pragma once
// Pure-C++17 Tokenizer, ICU Engram Vocab Compressor (99,092 exact), V4.1 Chat Renderer & DSML Parser
// Zero Python anywhere. Links against system libpcre2-8 and libicuuc/libicui18n.

#include "dsv41_json.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace dsv41 {

struct Dsv41ImageSource;

struct Dsv41SpecialTokens {
    static constexpr int32_t kBos       = 0;       // <｜begin▁of▁sentence｜>
    static constexpr int32_t kEos       = 1;       // <｜end▁of▁sentence｜>
    static constexpr int32_t kPad       = 2;       // <｜▁pad▁｜>
    static constexpr int32_t kSystem    = 128799;  // <｜System｜> (also DSpark noise token!)
    static constexpr int32_t kUser      = 128803;  // <｜User｜>
    static constexpr int32_t kAssistant = 128804;  // <｜Assistant｜>
    static constexpr int32_t kThinkOpen = 128821;  // <think>
    static constexpr int32_t kThinkEnd       = 128822;  // </think>
    static constexpr int32_t kDsml           = 128825;  // ｜DSML｜
    static constexpr int32_t kLatestReminder = 128828;  // <｜latest_reminder｜>
    static constexpr int32_t kImage          = 129264;  // <｜deepseek_image｜>
};

// Segment-structured prompt representation (deepseek-recipe token-ID anti-injection):
// Structural/image markers are carried as exact integer token IDs (`Kind::TokenId`), while
// ordinary user/tool/assistant text is carried as `Kind::OrdinaryText` and encoded strictly
// with base-vocabulary BPE (`parse_special = false`).
struct PromptSegment {
    enum class Kind : uint8_t { OrdinaryText = 0, TokenId = 1 };
    Kind kind = Kind::OrdinaryText;
    int32_t token_id = -1;
    int32_t repeat = 1;
    std::string text;

    static PromptSegment txt(std::string s) {
        PromptSegment seg;
        seg.kind = Kind::OrdinaryText;
        seg.text = std::move(s);
        return seg;
    }
    static PromptSegment tok(int32_t id, int32_t count = 1) {
        PromptSegment seg;
        seg.kind = Kind::TokenId;
        seg.token_id = id;
        seg.repeat = count;
        return seg;
    }
};

struct Dsv41ParsedToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

struct Dsv41ParsedCompletion {
    std::string reasoning_content;
    std::string content;
    std::vector<Dsv41ParsedToolCall> tool_calls;
};

class Dsv41Tokenizer {
public:
    Dsv41Tokenizer();
    ~Dsv41Tokenizer();

    Dsv41Tokenizer(const Dsv41Tokenizer&) = delete;
    Dsv41Tokenizer& operator=(const Dsv41Tokenizer&) = delete;

    void load(const std::string& tokenizer_json_path);

    size_t vocab_size() const { return id_to_token_.size(); }

    std::vector<int32_t> encode(const std::string& text, bool parse_special = true) const;
    // OpenMP-parallel segment BPE (`parse_special = false` on OrdinaryText, direct ID insertion on TokenId)
    std::vector<int32_t> encode_segments(const std::vector<PromptSegment>& segments) const;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special_tokens = false) const;
    std::string decode_single(int32_t id, bool skip_special_tokens = false) const;
    // O(1) precomputed raw byte-level payload for cross-token UTF-8 streaming without U+FFFD splits
    std::string_view raw_bytes_single(int32_t id, bool skip_special_tokens = false) const;
    const std::string& id_to_token(int32_t id) const;
    int32_t token_to_id(const std::string& tok) const;

    // Builds exact 129,280 -> 99,092 compressed token map per engram.py:build_compressed_token_map
    void build_engram_compressed_map(std::vector<int32_t>& out_lookup, int32_t& out_compressed_vocab_size) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<std::string> id_to_token_;
    std::vector<std::string> id_to_raw_bytes_;
    std::vector<uint8_t> id_is_special_;
    std::unordered_map<std::string, int32_t> token_to_id_;
};

// Renders OpenAI chat request into anti-injection `PromptSegment` sequence per deepseek-recipe:
// structural & image markers are emitted as `Kind::TokenId`; all ordinary text is `Kind::OrdinaryText`.
std::vector<PromptSegment> render_dsv41_chat_segments(
    const JsonValue& messages,
    const JsonValue* tools,
    const JsonValue* response_format,
    const JsonValue* reasoning_effort_val,
    std::string& out_thinking_mode,
    int& out_budget,
    std::vector<Dsv41ImageSource>* out_images = nullptr,
    int prefix_base_effort = 0);

enum class ThinkEndClass {
    NeedMoreLookahead,
    InlineMention,
    RealClose
};

// Distinguishes genuine structural `</think>` boundaries from inline code/prose mentions inside `<think>`
// (e.g. `"\n</think>"`, `` `</think>` ``, `'</think>'`, `post-</think>`, `(</think>)`, `the </think> tag`).
ThinkEndClass classify_think_end(const std::string& text, size_t te, bool is_eof);

// Parses raw model completion text (excluding <｜end▁of▁sentence｜>) into reasoning_content, content,
// and OpenAI-formatted tool_calls per hf_meta/encoding/encoding.py:parse_message_from_completion_text!
Dsv41ParsedCompletion parse_dsv41_completion(
    const std::string& raw_text,
    const std::string& thinking_mode,
    uint64_t call_id_seed = 1,
    size_t forced_think_end_pos = std::string::npos);

struct Dsv41ToolStreamDelta {
    size_t index = 0;
    std::string id;
    std::string name;
    std::string arguments_delta;
    bool is_start = false;
    bool is_end = false;
};

// Incremental streaming filter for SSE: emits reasoning_content deltas during <think>...</think>,
// content deltas after </think>, and incremental tool_calls deltas during <｜DSML｜ calls>...!
class Dsv41StreamParser {
public:
    explicit Dsv41StreamParser(const std::string& thinking_mode);

    // Token-ID-gated streaming parser (deepseek-recipe):
    //   - Exits `<think>` strictly on `token_id == Dsv41SpecialTokens::kThinkEnd` (128822)
    //   - Enters `DSML` tool parsing strictly on `token_id == Dsv41SpecialTokens::kDsml` (128825)
    //     matched via incremental KMP automaton
    //   - Buffers incomplete UTF-8 multi-byte tails across tokens and streams live `arguments_delta`
    //     JSON fragments without full-object buffering
    void feed_token(
        int32_t token_id,
        std::string_view raw_piece,
        std::string& out_reasoning_delta,
        std::string& out_content_delta,
        std::vector<Dsv41ToolStreamDelta>* out_tool_deltas = nullptr,
        uint64_t call_id_seed = 1);

    void feed(
        const std::string& piece,
        std::string& out_reasoning_delta,
        std::string& out_content_delta,
        std::vector<Dsv41ToolStreamDelta>* out_tool_deltas = nullptr,
        uint64_t call_id_seed = 1);
    void force_close_thinking(std::string& out_reasoning_delta);
    void finish(
        std::string& out_reasoning_delta,
        std::string& out_content_delta,
        Dsv41ParsedCompletion& out_final,
        uint64_t call_id_seed,
        std::vector<Dsv41ToolStreamDelta>* out_tool_deltas = nullptr);

    const std::string& raw_accumulated() const { return full_raw_; }
    const std::vector<PromptSegment>& assistant_segments() const { return assistant_segs_; }
    bool in_thinking() const { return state_ == State::InThinking; }
    bool in_tool_calls() const { return state_ == State::InToolCalls; }
    bool should_stop_generation() const { return tool_block_closed_ || tool_runaway_; }

private:
    void scan_incremental_tools(std::vector<Dsv41ToolStreamDelta>* out_tool_deltas, uint64_t call_id_seed);
    void step_token_gated_dsml_bytes(
        std::string_view utf8_chunk,
        std::vector<Dsv41ToolStreamDelta>* out_tool_deltas,
        uint64_t call_id_seed);
    void on_token_gated_dsml_marker(
        std::string& out_content_delta,
        std::vector<Dsv41ToolStreamDelta>* out_tool_deltas,
        uint64_t call_id_seed);

    enum class State { InThinking, InContent, ConfirmingDsmlOpen, InToolCalls };
    enum class DsmlSub {
        BetweenInvokes,
        ReadingInvokeHeader,
        BetweenParams,
        ReadingParamHeader,
        StreamingStringParamValue,
        BufferingJsonParamValue,
        ReadingCloseTag
    };

    std::string thinking_mode_;
    State state_;
    std::string full_raw_;
    std::string pending_;
    bool has_real_think_end_ = false;
    size_t real_think_end_pos_ = std::string::npos;
    size_t tool_scan_pos_ = 0;
    size_t next_tool_idx_ = 0;
    bool open_invoke_started_ = false;
    size_t open_invoke_body_pos_ = 0;
    std::string open_invoke_name_;
    size_t tool_bytes_ = 0;
    bool tool_block_closed_ = false;
    bool tool_runaway_ = false;

    // Token-ID-gated KMP & incremental JSON delta state
    bool token_gated_mode_ = false;
    std::string utf8_tail_;
    std::string reasoning_acc_;
    std::string content_acc_;
    std::vector<Dsv41ParsedToolCall> parsed_tools_;
    std::vector<PromptSegment> assistant_segs_;
    std::string dsml_pre_ws_;
    std::string dsml_open_buf_;
    int kmp_q_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    DsmlSub dsml_sub_ = DsmlSub::BetweenInvokes;
    std::string dsml_tag_buf_;
    std::string param_val_pending_;
    std::string cur_tool_id_;
    std::string cur_tool_name_;
    std::string cur_tool_json_acc_;
    size_t cur_tool_param_count_ = 0;
    std::string cur_param_name_;
    bool cur_param_is_string_ = true;
    bool close_is_invoke_ = false;
    bool close_is_wrapper_ = false;
};

}  // namespace dsv41
