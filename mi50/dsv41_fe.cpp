// Pure-C++17 OpenAI + Anthropic Compatible HTTP + SSE Server for DeepSeek-V4.1-Flash-1017
// Drives Dsv41Runtime (3x MI50 + 2x Blackhole P150A + Dual-NVMe Engram + DeepSeek-ViT + Dynamic DeepSelect8 M=8).
// Zero Python anywhere. Supports /health, /v1/models, /v1/tokenize, and /v1/chat/completions
// with streaming reasoning_content, content, DSML -> OpenAI tool_calls, and client disconnect abort.

#include "dsv41_anthropic_messages.hpp"
#include "dsv41_live_progress.hpp"
#include "dsv41_runtime.hpp"

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <random>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

volatile std::sig_atomic_t g_stop_server = 0;
std::atomic<int> g_listen_fd{-1};
void on_sig(int) {
    g_stop_server = 1;
    const int fd = g_listen_fd.exchange(-1, std::memory_order_relaxed);
    if (fd >= 0) (void)::shutdown(fd, SHUT_RDWR);
}

bool send_all(int fd, const char* buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = ::send(fd, buf + off, len - off, MSG_NOSIGNAL);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        off += static_cast<size_t>(n);
    }
    return true;
}

bool send_str(int fd, const std::string& s) {
    return send_all(fd, s.data(), s.size());
}

bool client_disconnected(int fd) {
    char c = 0;
    ssize_t r = ::recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT);
    if (r == 0) return true;  // Clean FIN from peer
    if (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return true;
    return false;
}

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
};

bool read_http_request(int fd, HttpRequest& req) {
    std::string raw;
    raw.reserve(8192);
    char tmp[8192];
    size_t hdr_end = std::string::npos;
    while (hdr_end == std::string::npos) {
        ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        raw.append(tmp, static_cast<size_t>(n));
        hdr_end = raw.find("\r\n\r\n");
        if (raw.size() > 64ULL * 1024ULL * 1024ULL) return false;
    }

    const std::string hdr_block = raw.substr(0, hdr_end);
    const size_t line_end = hdr_block.find("\r\n");
    const std::string req_line = (line_end == std::string::npos) ? hdr_block : hdr_block.substr(0, line_end);
    {
        std::istringstream iss(req_line);
        iss >> req.method >> req.path;
    }
    const size_t qpos = req.path.find('?');
    if (qpos != std::string::npos) req.path = req.path.substr(0, qpos);

    size_t content_len = 0;
    bool chunked = false;
    {
        std::istringstream hs(hdr_block);
        std::string line;
        std::getline(hs, line);
        while (std::getline(hs, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t colon = line.find(':');
            if (colon == std::string::npos) continue;
            std::string k = line.substr(0, colon);
            for (char& ch : k) {
                if (ch >= 'A' && ch <= 'Z') ch = char(ch - 'A' + 'a');
            }
            std::string v = line.substr(colon + 1);
            size_t a = 0;
            while (a < v.size() && (v[a] == ' ' || v[a] == '\t')) ++a;
            v = v.substr(a);
            if (k == "content-length") {
                content_len = static_cast<size_t>(std::strtoull(v.c_str(), nullptr, 10));
            } else if (k == "transfer-encoding" && v.find("chunked") != std::string::npos) {
                chunked = true;
            }
        }
    }

    const size_t body_start = hdr_end + 4;
    if (!chunked) {
        req.body = raw.substr(body_start);
        while (req.body.size() < content_len) {
            ssize_t n = ::recv(fd, tmp, std::min(sizeof(tmp), content_len - req.body.size()), 0);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                return false;
            }
            req.body.append(tmp, static_cast<size_t>(n));
        }
        if (req.body.size() > content_len) req.body.resize(content_len);
    } else {
        std::string buf = raw.substr(body_start);
        req.body.clear();
        while (true) {
            size_t crlf = buf.find("\r\n");
            while (crlf == std::string::npos) {
                ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
                if (n <= 0) return false;
                buf.append(tmp, static_cast<size_t>(n));
                crlf = buf.find("\r\n");
            }
            const size_t chunk_sz = static_cast<size_t>(std::strtoull(buf.substr(0, crlf).c_str(), nullptr, 16));
            buf.erase(0, crlf + 2);
            if (chunk_sz == 0) break;
            while (buf.size() < chunk_sz + 2) {
                ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
                if (n <= 0) return false;
                buf.append(tmp, static_cast<size_t>(n));
            }
            req.body.append(buf.data(), chunk_sz);
            buf.erase(0, chunk_sz + 2);
        }
    }
    return true;
}

void send_json_response(int fd, int status, const std::string& status_text, const std::string& json_body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << " " << status_text << "\r\n"
        << "Content-Type: application/json; charset=utf-8\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Access-Control-Allow-Headers: Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta, anthropic-dangerous-direct-browser-access\r\n"
        << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        << "Connection: close\r\n"
        << "Content-Length: " << json_body.size() << "\r\n\r\n"
        << json_body;
    (void)send_str(fd, oss.str());
}

std::atomic<uint64_t> g_req_counter{1};

const std::string& boot_nonce_hex() {
    static const std::string kNonce = []() {
        std::random_device rd;
        const uint64_t v = (uint64_t(rd()) << 32) ^ uint64_t(rd()) ^
                           uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
        char buf[20];
        std::snprintf(buf, sizeof(buf), "%012llx", static_cast<unsigned long long>(v & 0xFFFFFFFFFFFFULL));
        return std::string(buf);
    }();
    return kNonce;
}

uint64_t boot_call_seed() {
    static const uint64_t kSeed = []() {
        std::random_device rd;
        const uint64_t r = (uint64_t(rd()) % 900000ULL) + 100000ULL;
        return r * 100000ULL;
    }();
    return kSeed;
}

dsv41::JsonValue make_usage_json(int64_t prompt_toks, int64_t completion_toks, int64_t cached_toks) {
    const int64_t clamped_cached = std::clamp<int64_t>(cached_toks, 0, std::max<int64_t>(0, prompt_toks));
    const int64_t miss_toks = std::max<int64_t>(0, prompt_toks - clamped_cached);

    dsv41::JsonValue details = dsv41::JsonValue::make_object();
    details.set_key("cached_tokens", dsv41::JsonValue::make_int(clamped_cached));

    dsv41::JsonValue u = dsv41::JsonValue::make_object();
    u.set_key("prompt_tokens", dsv41::JsonValue::make_int(prompt_toks));
    u.set_key("completion_tokens", dsv41::JsonValue::make_int(completion_toks));
    u.set_key("total_tokens", dsv41::JsonValue::make_int(prompt_toks + completion_toks));
    u.set_key("prompt_tokens_details", std::move(details));
    u.set_key("prompt_cache_hit_tokens", dsv41::JsonValue::make_int(clamped_cached));
    u.set_key("prompt_cache_miss_tokens", dsv41::JsonValue::make_int(miss_toks));
    return u;
}

struct FeServerConfig {
    std::string default_model_id = "DeepSeek-V4.1-Flash-1017";
    std::string search_url;
    int max_seq_len = 1048576;
    int default_reasoning_effort = dsv41::kDsv41DefaultReasoningEffort; // 100 (Max)
    int default_max_tokens = dsv41::kDsv41DefaultMaxTokens;             // 262144
    int max_output_tokens = dsv41::kDsv41ArchMaxOutputTokens;           // 393216
    int min_thinking_max_tokens = 0;
};

void log_fe_turn_telemetry(
    uint64_t req_num,
    const char* tag,
    const dsv41::Dsv41GenStats& stats,
    size_t tool_count,
    bool thinking_enabled,
    int reasoning_effort,
    int max_new_tokens,
    int server_search_reqs = 0)
{
    const double reuse_tps = (stats.cached_prefix_tokens > 0)
                                 ? (1000.0 * stats.cached_prefix_tokens / std::max(stats.prefix_reuse_ms, 0.01))
                                 : 0.0;
    const double fresh_tps = (stats.fresh_prefill_ms > 0.0 && stats.prefilled_tokens > 0)
                                 ? (1000.0 * stats.prefilled_tokens / stats.fresh_prefill_ms)
                                 : 0.0;
    const double dec_tps = (stats.decode_ms > 0.0) ? (1000.0 * stats.completion_tokens / stats.decode_ms) : 0.0;
    const double acc_pct = (stats.dspark_drafted > 0)
                               ? (100.0 * double(stats.dspark_accepted) / double(stats.dspark_drafted))
                               : 0.0;
    const double tau = (stats.dspark_steps > 0)
                           ? (1.0 + double(stats.dspark_accepted) / double(stats.dspark_steps))
                           : 1.0;
    const uint64_t tree_m16 = stats.dspark_tree16_steps;
    const uint64_t tree_m8 = (stats.dspark_tree_steps >= tree_m16)
                                 ? (stats.dspark_tree_steps - tree_m16)
                                 : 0ULL;
    const uint64_t tree_dyn8 = stats.dspark_tree_dyn8_steps;
    char effort_buf[16];
    if (thinking_enabled && reasoning_effort > 0) {
        std::snprintf(effort_buf, sizeof(effort_buf), "%d", reasoning_effort);
    } else {
        std::snprintf(effort_buf, sizeof(effort_buf), "off");
    }
    char extra_note[224] = "";
    int off = 0;
    if (stats.images_encoded > 0 && size_t(off) < sizeof(extra_note)) {
        off += std::snprintf(extra_note + off, sizeof(extra_note) - size_t(off),
                             " | img=%d(%.1fms)", stats.images_encoded, stats.vit_encode_ms);
    }
    if (server_search_reqs > 0 && size_t(off) < sizeof(extra_note)) {
        off += std::snprintf(extra_note + off, sizeof(extra_note) - size_t(off), " | srv_search=%d", server_search_reqs);
    }
    if (stats.think_guard_triggered && size_t(off) < sizeof(extra_note)) {
        std::snprintf(extra_note + off, sizeof(extra_note) - size_t(off), " | ThinkGuard(rew=%u,p=%u)",
                      stats.think_guard_rewound, stats.think_guard_period);
    }
    std::printf("[DSV41_FE] #%llu %s OK | effort=%s max_new=%d tree_m8=%llu(dyn=%llu) tree_m16=%llu | "
                "prompt=%d (reuse=%d [radix=%d nvme_hit=%d (%.2fms) swa_rep=%d] in %.2fms=%.0f tok/s | fresh=%d in %.1fms=%.1f tok/s) | "
                "gen=%d in %.1fms=%.2f tok/s [bb=%.0fms p150=%.0fms dsp=%.0fms smp=%.0fms steps=%llu tau=%.2f] | "
                "stages[attn=%.0fms hsi=%.0fms moe=%.0fms p150w=%.0fms ovlp=%.0fms ar=%.0fms egw=%.1fms] | "
                "DSpark acc=%llu/%llu (%.1f%%, tree=%llu/%llu, dyn8=%llu/%llu, tree16=%llu/%llu) | tools=%zu%s\n",
                static_cast<unsigned long long>(req_num), tag,
                effort_buf, max_new_tokens,
                static_cast<unsigned long long>(tree_m8),
                static_cast<unsigned long long>(tree_dyn8),
                static_cast<unsigned long long>(tree_m16),
                stats.prompt_tokens,
                stats.cached_prefix_tokens, stats.radix_hit_tokens,
                stats.nvme_hit_tokens, stats.nvme_hydrate_ms, stats.swa_replay_tokens,
                stats.prefix_reuse_ms, reuse_tps,
                stats.prefilled_tokens, stats.fresh_prefill_ms, fresh_tps,
                stats.completion_tokens, stats.decode_ms, dec_tps,
                stats.dec_backbone_ms, stats.dec_p150_dev_ms, stats.dec_dspark_ms, stats.dec_sample_ms,
                static_cast<unsigned long long>(stats.dspark_steps), tau,
                stats.t_attn_ms, stats.t_hsi_index_ms, stats.t_mi50_moe_ms, stats.t_p150_wait_ms,
                stats.overlap_saved_ms, stats.t_allreduce_ms, stats.t_engram_wait_ms,
                static_cast<unsigned long long>(stats.dspark_accepted),
                static_cast<unsigned long long>(stats.dspark_drafted),
                acc_pct,
                static_cast<unsigned long long>(stats.dspark_tree_accepted),
                static_cast<unsigned long long>(stats.dspark_tree_steps),
                static_cast<unsigned long long>(stats.dspark_tree_dyn8_accepted),
                static_cast<unsigned long long>(stats.dspark_tree_dyn8_steps),
                static_cast<unsigned long long>(stats.dspark_tree16_accepted),
                static_cast<unsigned long long>(stats.dspark_tree16_steps),
                tool_count, extra_note);
    std::fflush(stdout);
}

void handle_connection(int fd, dsv41::Dsv41Runtime* rt, const FeServerConfig& cfg) {
    const std::string& default_model_id = cfg.default_model_id;
    const std::string& search_url = cfg.search_url;
    int one = 1;
    (void)::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    HttpRequest req;
    if (!read_http_request(fd, req)) {
        ::close(fd);
        return;
    }

    if (req.method == "OPTIONS") {
        send_json_response(fd, 200, "OK", "{}");
        ::close(fd);
        return;
    }

    if (req.path == "/health" || req.path == "/v1/health") {
        dsv41::JsonValue obj = dsv41::JsonValue::make_object();
        obj.set_key("status", rt->ready() ? "ok" : "loading");
        obj.set_key("ready", rt->ready());
        obj.set_key("vision", rt->vision_ready());
        obj.set_key("model", default_model_id);
        send_json_response(fd, rt->ready() ? 200 : 503, rt->ready() ? "OK" : "Service Unavailable", obj.dumps_compact());
        ::close(fd);
        return;
    }

    if (req.path == "/v1/models" || req.path == "/models") {
        const int64_t now_ts = static_cast<int64_t>(std::time(nullptr));
        dsv41::JsonValue data = dsv41::JsonValue::make_array();
        for (const char* mid : {default_model_id.c_str(), "dsv41-hlut2", "DeepSeek-V4.1-Flash-1017"}) {
            bool dup = false;
            for (const auto& existing : data.arr_val) {
                if (existing.at("id").as_str() == mid) dup = true;
            }
            if (dup) continue;
            dsv41::JsonValue m = dsv41::JsonValue::make_object();
            m.set_key("id", std::string(mid));
            m.set_key("object", "model");
            m.set_key("created", now_ts);
            m.set_key("owned_by", "deepseek-local-hlut2");
            data.push_back(std::move(m));
        }
        dsv41::JsonValue root = dsv41::JsonValue::make_object();
        root.set_key("object", "list");
        root.set_key("data", std::move(data));
        send_json_response(fd, 200, "OK", root.dumps_compact());
        ::close(fd);
        return;
    }

    if (req.path == "/v1/reset_cache" && req.method == "POST") {
        rt->reset_session_cache();
        send_json_response(fd, 200, "OK", "{\"status\":\"gpu_and_rewind_cleared_radix_preserved\"}");
        ::close(fd);
        return;
    }

    if (req.path == "/v1/clear_all_caches" && req.method == "POST") {
        rt->clear_all_caches();
        send_json_response(fd, 200, "OK", "{\"status\":\"all_ram_and_gpu_caches_cleared\"}");
        ::close(fd);
        return;
    }

    if (req.path == "/v1/profile_sync" && req.method == "POST") {
        const bool en = (req.body.find("true") != std::string::npos || req.body.find("1") != std::string::npos);
        rt->set_sync_timers(en);
        send_json_response(fd, 200, "OK", en ? "{\"sync_timers\":true}" : "{\"sync_timers\":false}");
        ::close(fd);
        return;
    }

    if (req.path == "/v1/tokenize" && req.method == "POST") {
        try {
            dsv41::JsonValue j = dsv41::parse_json(req.body);
            std::string text;
            std::vector<dsv41::Dsv41ImageSource> tok_images;
            std::vector<int32_t> ids;
            if (j.contains("messages") && j.at("messages").is_array()) {
                const auto oreq = dsv41::translate_openai_to_dsv41(
                    j, default_model_id, cfg.default_reasoning_effort,
                    cfg.max_output_tokens, cfg.default_max_tokens, cfg.min_thinking_max_tokens);
                std::string tmode;
                int budget = oreq.reasoning_effort;
                const std::vector<dsv41::PromptSegment> segs =
                    dsv41::build_dsv41_openai_prompt_segments(oreq, tmode, budget, &tok_images);
                ids = rt->tokenizer().encode_segments(segs);
            } else {
                if (j.contains("prompt") && j.at("prompt").is_string()) {
                    text = j.at("prompt").as_str();
                } else if (j.contains("text") && j.at("text").is_string()) {
                    text = j.at("text").as_str();
                }
                const bool parse_special = j.contains("parse_special") ? j.at("parse_special").as_bool(false) : false;
                ids = rt->tokenizer().encode(text, parse_special);
            }
            if (!tok_images.empty()) {
                std::vector<int32_t> exp_ids;
                std::vector<uint8_t> img_mask;
                std::vector<dsv41::Dsv41PreparedImage> prep;
                dsv41::expand_deepseek_vl_prompt(ids, tok_images, exp_ids, img_mask, prep);
                ids = std::move(exp_ids);
            }
            dsv41::JsonValue tok_arr = dsv41::JsonValue::make_array();
            for (int32_t id : ids) tok_arr.push_back(int64_t(id));
            dsv41::JsonValue resp = dsv41::JsonValue::make_object();
            resp.set_key("count", int64_t(ids.size()));
            resp.set_key("tokens", std::move(tok_arr));
            send_json_response(fd, 200, "OK", resp.dumps_compact());
        } catch (const std::exception& e) {
            dsv41::JsonValue err = dsv41::JsonValue::make_object();
            err.set_key("error", std::string(e.what()));
            send_json_response(fd, 400, "Bad Request", err.dumps_compact());
        }
        ::close(fd);
        return;
    }

    if ((req.path == "/v1/chat/completions" || req.path == "/chat/completions") && req.method == "POST") {
        try {
            const uint64_t req_num = g_req_counter.fetch_add(1);
            const int64_t created_ts = static_cast<int64_t>(std::time(nullptr));
            const std::string cmpl_id = "chatcmpl-dsv41-" + boot_nonce_hex() + "-" + std::to_string(req_num);
            const uint64_t call_seed = boot_call_seed() + req_num * 100ULL + 1ULL;

            const dsv41::JsonValue body = dsv41::parse_json(req.body);
            if (!body.contains("messages") || !body.at("messages").is_array()) {
                throw std::runtime_error("Missing 'messages' array in chat/completions request");
            }

            const dsv41::OpenAiTranslatedRequest oreq = dsv41::translate_openai_to_dsv41(
                body, default_model_id,
                cfg.default_reasoning_effort,
                cfg.max_output_tokens,
                cfg.default_max_tokens,
                cfg.min_thinking_max_tokens);

            int effective_max_new = oreq.max_tokens;
            std::string sync_json;
            dsv41::Dsv41GenStats agg_stats{};
            int srv_search_reqs = 0;
            size_t total_tools = 0;

            if (oreq.stream) {
                const std::string sse_hdr =
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/event-stream; charset=utf-8\r\n"
                    "Cache-Control: no-cache\r\n"
                    "Connection: close\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "Access-Control-Allow-Headers: Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta, anthropic-dangerous-direct-browser-access\r\n"
                    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\r\n";
                if (!send_str(fd, sse_hdr)) {
                    ::close(fd);
                    return;
                }

                std::mutex sse_mtx;
                std::atomic<bool> turn_done{false};
                std::atomic<bool> aborted{false};

                auto sse_write = [&](const std::string& payload) -> bool {
                    std::lock_guard<std::mutex> lk(sse_mtx);
                    if (aborted.load(std::memory_order_relaxed)) return false;
                    if (!send_str(fd, payload)) {
                        aborted.store(true, std::memory_order_release);
                        return false;
                    }
                    return true;
                };

                auto make_chunk_envelope = [&](dsv41::JsonValue delta, const char* finish_reason) {
                    dsv41::JsonValue choice = dsv41::JsonValue::make_object();
                    choice.set_key("index", int64_t(0));
                    choice.set_key("delta", std::move(delta));
                    if (finish_reason) choice.set_key("finish_reason", std::string(finish_reason));
                    else choice.set_key("finish_reason", dsv41::JsonValue());

                    dsv41::JsonValue choices = dsv41::JsonValue::make_array();
                    choices.push_back(std::move(choice));

                    dsv41::JsonValue env = dsv41::JsonValue::make_object();
                    env.set_key("id", cmpl_id);
                    env.set_key("object", "chat.completion.chunk");
                    env.set_key("created", created_ts);
                    env.set_key("model", oreq.model);
                    env.set_key("choices", std::move(choices));
                    return "data: " + env.dumps_compact() + "\n\n";
                };

                // Initial empty role delta (preserves true client TTFT until first real token delta)
                dsv41::JsonValue role_delta = dsv41::JsonValue::make_object();
                role_delta.set_key("role", "assistant");
                role_delta.set_key("content", "");
                if (!sse_write(make_chunk_envelope(std::move(role_delta), nullptr))) {
                    ::close(fd);
                    return;
                }

                dsv41::Dsv41LiveProgress live_progress(true);
                auto emit_progress = [&](bool is_final) -> bool {
                    if (aborted.load(std::memory_order_acquire)) return false;
                    if (!is_final) {
                        if ((!live_progress.input_ready.load(std::memory_order_acquire) &&
                             live_progress.count.load(std::memory_order_acquire) == 0) ||
                            live_progress.end_ns.load(std::memory_order_acquire) != 0) {
                            return true;
                        }
                    }
                    uint64_t serviced = 0, total = 0, cached = 0, gen = 0;
                    dsv41::JsonValue prog = live_progress.sample(cmpl_id, is_final, &serviced, &total, &cached, &gen);

                    dsv41::JsonValue env = dsv41::JsonValue::make_object();
                    env.set_key("id", cmpl_id);
                    env.set_key("object", "chat.completion.chunk");
                    env.set_key("created", created_ts);
                    env.set_key("model", oreq.model);
                    env.set_key("choices", dsv41::JsonValue::make_array());
                    env.set_key("x_flash_progress", std::move(prog));
                    if (oreq.include_usage && total > 0 && serviced >= total) {
                        env.set_key("usage", make_usage_json(int64_t(total), int64_t(gen), int64_t(cached)));
                    }
                    return sse_write("data: " + env.dumps_compact() + "\n\n");
                };

                std::thread hb([&]() {
                    int ticks = 0;
                    while (!turn_done.load(std::memory_order_acquire) && !aborted.load(std::memory_order_acquire)) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        if (turn_done.load(std::memory_order_acquire) || aborted.load(std::memory_order_acquire)) break;
                        if (!emit_progress(false)) break;
                        if (++ticks >= 150) {
                            ticks = 0;
                            if (!live_progress.input_ready.load(std::memory_order_acquire)) {
                                dsv41::JsonValue keepalive = dsv41::JsonValue::make_object();
                                keepalive.set_key("content", "");
                                if (!sse_write(make_chunk_envelope(std::move(keepalive), nullptr))) break;
                            }
                        }
                    }
                });
                struct HbJoinGuard {
                    std::atomic<bool>& done;
                    std::thread& th;
                    void stop_and_join() {
                        done.store(true, std::memory_order_release);
                        if (th.joinable()) th.join();
                    }
                    ~HbJoinGuard() { stop_and_join(); }
                } hb_guard{turn_done, hb};

                std::atomic<uint64_t> gen_tok_cnt{0};
                int sub_step_idx = 0;
                const bool ok = dsv41::run_openai_chat_completions_turn(
                    oreq,
                    rt->tokenizer(),
                    search_url,
                    cmpl_id,
                    created_ts,
                    call_seed,
                    [&](const std::vector<int32_t>& prompt_ids,
                        int max_new_tokens,
                        dsv41::TokenStreamCallback on_tok,
                        dsv41::Dsv41GenStats& out_stats,
                        const std::vector<dsv41::Dsv41ImageSource>* images) -> std::vector<int32_t> {
                        const int clamped = dsv41::clamp_max_new_tokens_to_context(
                            max_new_tokens, int(prompt_ids.size()), cfg.max_seq_len, cfg.max_output_tokens);
                        if (sub_step_idx == 0) effective_max_new = clamped;
                        ++sub_step_idx;

                        dsv41::Dsv41GenParams params;
                        params.max_new_tokens = clamped;
                        params.temperature = oreq.temperature;
                        params.top_p = oreq.top_p;
                        params.repetition_penalty = oreq.repetition_penalty;
                        params.seed = oreq.seed;
                        params.enable_dspark = oreq.enable_dspark;
                        params.enable_think_loop_guard = oreq.enable_think_loop_guard;

                        return rt->generate(
                            prompt_ids,
                            params,
                            [&](int32_t tok_id) -> bool {
                                const uint64_t tc = gen_tok_cnt.fetch_add(1, std::memory_order_relaxed) + 1;
                                live_progress.observe_token(tc);
                                return on_tok(tok_id);
                            },
                            out_stats,
                            [&](const dsv41::Dsv41ProgressUpdate& upd) -> bool {
                                if (aborted.load(std::memory_order_relaxed) || client_disconnected(fd)) {
                                    aborted.store(true, std::memory_order_release);
                                    return false;
                                }
                                if (upd.phase == dsv41::Dsv41ProgressPhase::PrefixDone) {
                                    live_progress.on_prefix_done(
                                        uint64_t(std::max(0, upd.prompt_tokens)),
                                        uint64_t(std::max(0, upd.cached_tokens)),
                                        upd.begin_ns,
                                        upd.end_ns);
                                } else {
                                    live_progress.on_prefill_chunk(
                                        uint64_t(std::max(0, upd.chunk_tokens)),
                                        upd.chunk_ms,
                                        upd.begin_ns,
                                        upd.end_ns);
                                }
                                return emit_progress(false);
                            },
                            images);
                    },
                    [&]() -> bool {
                        if (aborted.load(std::memory_order_relaxed) || client_disconnected(fd)) {
                            aborted.store(true, std::memory_order_release);
                            return true;
                        }
                        return false;
                    },
                    sse_write,
                    sync_json,
                    agg_stats,
                    srv_search_reqs,
                    total_tools,
                    [&]() {
                        live_progress.finish();
                        hb_guard.stop_and_join();
                        if (!aborted.load(std::memory_order_acquire)) {
                            (void)emit_progress(true);
                        }
                    });

                live_progress.finish();
                hb_guard.stop_and_join();

                if (ok && !aborted.load(std::memory_order_acquire)) {
                    log_fe_turn_telemetry(
                        req_num, "stream", agg_stats, total_tools,
                        oreq.thinking_enabled, oreq.reasoning_effort, effective_max_new, srv_search_reqs);
                } else {
                    std::printf("[DSV41_FE] #%llu stream aborted by client disconnect after %d tokens\n",
                                static_cast<unsigned long long>(req_num), agg_stats.completion_tokens);
                    std::fflush(stdout);
                }
            } else {
                int sub_step_idx = 0;
                const bool ok = dsv41::run_openai_chat_completions_turn(
                    oreq,
                    rt->tokenizer(),
                    search_url,
                    cmpl_id,
                    created_ts,
                    call_seed,
                    [&](const std::vector<int32_t>& prompt_ids,
                        int max_new_tokens,
                        dsv41::TokenStreamCallback on_tok,
                        dsv41::Dsv41GenStats& out_stats,
                        const std::vector<dsv41::Dsv41ImageSource>* images) -> std::vector<int32_t> {
                        const int clamped = dsv41::clamp_max_new_tokens_to_context(
                            max_new_tokens, int(prompt_ids.size()), cfg.max_seq_len, cfg.max_output_tokens);
                        if (sub_step_idx == 0) effective_max_new = clamped;
                        ++sub_step_idx;

                        dsv41::Dsv41GenParams params;
                        params.max_new_tokens = clamped;
                        params.temperature = oreq.temperature;
                        params.top_p = oreq.top_p;
                        params.repetition_penalty = oreq.repetition_penalty;
                        params.seed = oreq.seed;
                        params.enable_dspark = oreq.enable_dspark;
                        params.enable_think_loop_guard = oreq.enable_think_loop_guard;
                        return rt->generate(prompt_ids, params, std::move(on_tok), out_stats, nullptr, images);
                    },
                    [&]() -> bool { return client_disconnected(fd); },
                    [](const std::string&) -> bool { return true; },
                    sync_json,
                    agg_stats,
                    srv_search_reqs,
                    total_tools);

                if (ok) {
                    send_json_response(fd, 200, "OK", sync_json);
                    log_fe_turn_telemetry(
                        req_num, "sync", agg_stats, total_tools,
                        oreq.thinking_enabled, oreq.reasoning_effort, effective_max_new, srv_search_reqs);
                }
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[DSV41_FE] Request error: %s\n", e.what());
            dsv41::JsonValue err_inner = dsv41::JsonValue::make_object();
            err_inner.set_key("message", std::string(e.what()));
            err_inner.set_key("type", "server_error");
            dsv41::JsonValue err = dsv41::JsonValue::make_object();
            err.set_key("error", std::move(err_inner));
            send_json_response(fd, 500, "Internal Server Error", err.dumps_compact());
        }
        ::close(fd);
        return;
    }

    if ((req.path == "/anthropic/v1/messages" || req.path == "/v1/messages" || req.path == "/messages") &&
        req.method == "POST")
    {
        try {
            const uint64_t req_num = g_req_counter.fetch_add(1);
            const std::string msg_id = "msg_dsv41_" + boot_nonce_hex() + "_" + std::to_string(req_num);
            const uint64_t call_seed = boot_call_seed() + req_num * 100ULL + 1ULL;

            const dsv41::JsonValue body = dsv41::parse_json(req.body);
            if (!body.contains("messages") || !body.at("messages").is_array()) {
                throw std::runtime_error("Missing 'messages' array in Anthropic messages request");
            }
            const dsv41::AnthropicTranslatedRequest treq = dsv41::translate_anthropic_to_dsv41(
                body, default_model_id,
                cfg.default_reasoning_effort,
                cfg.max_output_tokens,
                cfg.default_max_tokens,
                cfg.min_thinking_max_tokens);

            if (treq.stream) {
                const std::string sse_hdr =
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/event-stream; charset=utf-8\r\n"
                    "Cache-Control: no-cache\r\n"
                    "Connection: close\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "Access-Control-Allow-Headers: Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta, anthropic-dangerous-direct-browser-access\r\n"
                    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\r\n";
                if (!send_str(fd, sse_hdr)) {
                    ::close(fd);
                    return;
                }
            }

            std::mutex sse_mtx;
            std::atomic<bool> aborted{false};
            auto sse_send = [&](const std::string& payload) -> bool {
                std::lock_guard<std::mutex> lk(sse_mtx);
                if (aborted.load(std::memory_order_relaxed)) return false;
                if (!send_str(fd, payload)) {
                    aborted.store(true, std::memory_order_release);
                    return false;
                }
                return true;
            };

            std::string sync_json;
            dsv41::Dsv41GenStats agg_stats{};
            int srv_search_reqs = 0;
            size_t total_tools = 0;
            int effective_max_new = treq.max_tokens;
            int sub_step_idx = 0;

            const bool ok = dsv41::run_anthropic_messages_turn(
                treq,
                rt->tokenizer(),
                search_url,
                msg_id,
                call_seed,
                [&](const std::vector<int32_t>& prompt_ids,
                    int max_new_tokens,
                    dsv41::TokenStreamCallback on_tok,
                    dsv41::Dsv41GenStats& out_stats,
                    const std::vector<dsv41::Dsv41ImageSource>* images) -> std::vector<int32_t> {
                    const int clamped = dsv41::clamp_max_new_tokens_to_context(
                        max_new_tokens, int(prompt_ids.size()), cfg.max_seq_len, cfg.max_output_tokens);
                    if (sub_step_idx == 0) effective_max_new = clamped;
                    ++sub_step_idx;

                    dsv41::Dsv41GenParams params;
                    params.max_new_tokens = clamped;
                    params.temperature = treq.has_explicit_temperature ? std::max(0.0f, treq.temperature) : 1.0f;
                    params.top_p = std::clamp(treq.top_p, 0.01f, 1.0f);
                    params.seed = treq.seed;
                    params.enable_dspark = true;
                    return rt->generate(prompt_ids, params, std::move(on_tok), out_stats, nullptr, images);
                },
                [&]() -> bool {
                    if (aborted.load(std::memory_order_relaxed) || client_disconnected(fd)) {
                        aborted.store(true, std::memory_order_release);
                        return true;
                    }
                    return false;
                },
                sse_send,
                sync_json,
                agg_stats,
                srv_search_reqs,
                total_tools);

            if (ok && !aborted.load(std::memory_order_acquire)) {
                if (!treq.stream) {
                    send_json_response(fd, 200, "OK", sync_json);
                }
                log_fe_turn_telemetry(
                    req_num, treq.stream ? "anthropic-stream" : "anthropic-sync",
                    agg_stats, total_tools,
                    treq.thinking_enabled, treq.reasoning_effort, effective_max_new, srv_search_reqs);
            } else {
                std::printf("[DSV41_FE] #%llu anthropic turn aborted after %d tokens\n",
                            static_cast<unsigned long long>(req_num), agg_stats.completion_tokens);
                std::fflush(stdout);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[DSV41_FE] Anthropic request error: %s\n", e.what());
            dsv41::JsonValue err_inner = dsv41::JsonValue::make_object();
            err_inner.set_key("type", "invalid_request_error");
            err_inner.set_key("message", std::string(e.what()));
            dsv41::JsonValue err = dsv41::JsonValue::make_object();
            err.set_key("type", "error");
            err.set_key("error", std::move(err_inner));
            send_json_response(fd, 400, "Bad Request", err.dumps_compact());
        }
        ::close(fd);
        return;
    }

    send_json_response(fd, 404, "Not Found", "{\"error\":\"not found\"}");
    ::close(fd);
}

}  // namespace

int main(int argc, char** argv) {
    ::setenv("OMP_WAIT_POLICY", "PASSIVE", 1);
    ::setenv("GPU_MAX_HW_QUEUES", "8", 0);
    try {
        const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                         ? std::string(std::getenv("DSV41_ROOT"))
                                         : std::string(".");
        std::string host = "127.0.0.1";
        int port = (std::getenv("DSV41_PORT") != nullptr) ? std::stoi(std::getenv("DSV41_PORT")) : 8000;
        std::string mi50_root = root_env + "/packed_mi50";
        std::string tok_path  = root_env + "/hf_meta/tokenizer.json";
        std::string p150_shm  = "/dsv41_p150_ipc";
        std::string rarebal_overlay;
        std::string expert_perm_overlay;
        std::string hlut3_mtp_overlay;
        std::string hlut3_dev3_overlay;
        std::string hlut3_tp3_overlay;
        int hlut3_dev3_id = 3;
        if (const char* w9_dir = std::getenv("DSV41_WAVE9_OVERLAY_DIR")) {
            if (*w9_dir != '\0') {
                const std::string c_dev3 = std::string(w9_dir) + "/wave9_2_dev3_hlut3.safetensors";
                const std::string c_tp3  = std::string(w9_dir) + "/wave9_2_tp3_hlut3.safetensors";
                if (std::ifstream(c_dev3, std::ios::binary).good()) hlut3_dev3_overlay = c_dev3;
                if (std::ifstream(c_tp3, std::ios::binary).good())  hlut3_tp3_overlay  = c_tp3;
            }
        }
        bool disable_expert_perm = false;
        std::string capture_rarebal_dir;
        std::string vit_weights;
        bool vit_explicit = false;
        {
            const std::string def_vit = root_env + "/packed_vit/dsv41_vit_bf16.safetensors";
            std::ifstream vf(def_vit, std::ios::binary);
            if (vf.good()) vit_weights = def_vit;
        }
        std::string nvme_kv_cache_dir = root_env + "/nvme_kv_cache";
        double nvme_kv_cache_gb = 32.0;
        bool dual_overlap        = true;
        int dspark_tree_width    = 0;  // 0 = auto, 8 = fixed M=8, 16 = fixed M=16
        int dspark_tree16_max_ctx = 0; // Wave 8.3A: default 0 (disables under-48K 4-batch M=16 drag)
        bool dspark_static_m8    = false;
        int dspark_dyn8_max_depth = 5;
        bool csa2_shared_tree8   = true;
        bool fused_mhc_norm      = true;
        bool fused_dspark_epilogue = true;
        bool rocblas_prefill     = true;
        int rocblas_expert_thresh = 18;
        int rocblas_hlut2_thresh  = 20;
        int prefill_superchunk_m  = 2048;
        float tree8_conf_thresh   = 0.72f;
        int chain_max_b           = 5;

        FeServerConfig srv_cfg;
        for (const char* env_k : {"SEARCH_URL", "DSV41_SEARCH_URL", "DSV41_SEARXNG_URL"}) {
            if (const char* env_s = std::getenv(env_k)) {
                if (*env_s != '\0') {
                    srv_cfg.search_url = env_s;
                    break;
                }
            }
        }

        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--link-check") {
                std::cout << "[DSV41_FE] Link check OK\n";
                return 0;
            } else if (arg == "--model-dir" && i + 1 < argc) {
                const std::string md = argv[++i];
                mi50_root = md + "/packed_mi50";
                tok_path  = md + "/hf_meta/tokenizer.json";
                nvme_kv_cache_dir = md + "/nvme_kv_cache";
                if (!vit_explicit) {
                    const std::string cand_vit = md + "/packed_vit/dsv41_vit_bf16.safetensors";
                    std::ifstream vf(cand_vit, std::ios::binary);
                    vit_weights = vf.good() ? cand_vit : std::string();
                }
            } else if (arg == "--host" && i + 1 < argc) {
                host = argv[++i];
            } else if (arg == "--port" && i + 1 < argc) {
                port = std::stoi(argv[++i]);
            } else if (arg == "--mi50-root" && i + 1 < argc) {
                mi50_root = argv[++i];
            } else if (arg == "--tokenizer" && i + 1 < argc) {
                tok_path = argv[++i];
            } else if (arg == "--p150-shm" && i + 1 < argc) {
                p150_shm = argv[++i];
            } else if (arg == "--model-id" && i + 1 < argc) {
                srv_cfg.default_model_id = argv[++i];
            } else if (arg == "--max-seq-len" && i + 1 < argc) {
                srv_cfg.max_seq_len = std::stoi(argv[++i]);
            } else if (arg == "--default-reasoning-effort" && i + 1 < argc) {
                srv_cfg.default_reasoning_effort = dsv41::parse_reasoning_effort_cli(argv[++i]);
            } else if (arg == "--default-max-tokens" && i + 1 < argc) {
                srv_cfg.default_max_tokens = std::clamp(std::stoi(argv[++i]), 1, dsv41::kDsv41ArchMaxOutputTokens);
            } else if (arg == "--max-output-tokens" && i + 1 < argc) {
                srv_cfg.max_output_tokens = std::clamp(std::stoi(argv[++i]), 1, dsv41::kDsv41ArchMaxOutputTokens);
            } else if (arg == "--min-thinking-max-tokens" && i + 1 < argc) {
                srv_cfg.min_thinking_max_tokens = std::max(0, std::stoi(argv[++i]));
            } else if (arg == "--rarebal-overlay" && i + 1 < argc) {
                rarebal_overlay = argv[++i];
            } else if (arg == "--no-rarebal-overlay") {
                rarebal_overlay.clear();
            } else if (arg == "--expert-perm-overlay" && i + 1 < argc) {
                expert_perm_overlay = argv[++i];
                disable_expert_perm = false;
            } else if (arg == "--no-expert-perm-overlay") {
                expert_perm_overlay.clear();
                disable_expert_perm = true;
            } else if (arg == "--hlut3-mtp-overlay" && i + 1 < argc) {
                hlut3_mtp_overlay = argv[++i];
            } else if (arg == "--no-hlut3-mtp-overlay") {
                hlut3_mtp_overlay.clear();
            } else if (arg == "--hlut3-dev3-overlay" && i + 1 < argc) {
                hlut3_dev3_overlay = argv[++i];
            } else if (arg == "--no-hlut3-dev3-overlay") {
                hlut3_dev3_overlay.clear();
            } else if (arg == "--hlut3-dev3-id" && i + 1 < argc) {
                hlut3_dev3_id = std::stoi(argv[++i]);
            } else if (arg == "--hlut3-tp3-overlay" && i + 1 < argc) {
                hlut3_tp3_overlay = argv[++i];
            } else if (arg == "--no-hlut3-tp3-overlay") {
                hlut3_tp3_overlay.clear();
            } else if (arg == "--rocblas-prefill") {
                rocblas_prefill = true;
            } else if (arg == "--no-rocblas-prefill") {
                rocblas_prefill = false;
            } else if (arg == "--rocblas-expert-thresh" && i + 1 < argc) {
                rocblas_expert_thresh = std::stoi(argv[++i]);
            } else if (arg == "--rocblas-hlut2-thresh" && i + 1 < argc) {
                rocblas_hlut2_thresh = std::stoi(argv[++i]);
            } else if (arg == "--prefill-superchunk-m" && i + 1 < argc) {
                prefill_superchunk_m = std::stoi(argv[++i]);
            } else if (arg == "--tree8-conf-thresh" && i + 1 < argc) {
                tree8_conf_thresh = std::stof(argv[++i]);
            } else if (arg == "--chain-max-b" && i + 1 < argc) {
                chain_max_b = std::stoi(argv[++i]);
            } else if (arg == "--csa2-shared-tree8") {
                csa2_shared_tree8 = true;
            } else if (arg == "--no-csa2-shared-tree8") {
                csa2_shared_tree8 = false;
            } else if (arg == "--fused-mhc-norm") {
                fused_mhc_norm = true;
            } else if (arg == "--no-fused-mhc-norm") {
                fused_mhc_norm = false;
            } else if (arg == "--fused-dspark-epilogue") {
                fused_dspark_epilogue = true;
            } else if (arg == "--no-fused-dspark-epilogue") {
                fused_dspark_epilogue = false;
            } else if (arg == "--capture-rarebal" && i + 1 < argc) {
                capture_rarebal_dir = argv[++i];
            } else if (arg == "--vit-weights" && i + 1 < argc) {
                vit_weights = argv[++i];
                vit_explicit = true;
            } else if (arg == "--no-vit") {
                vit_weights.clear();
                vit_explicit = true;
            } else if (arg == "--nvme-kv-cache-dir" && i + 1 < argc) {
                nvme_kv_cache_dir = argv[++i];
            } else if (arg == "--nvme-kv-cache-gb" && i + 1 < argc) {
                nvme_kv_cache_gb = std::stod(argv[++i]);
            } else if (arg == "--no-nvme-kv-cache") {
                nvme_kv_cache_dir.clear();
                nvme_kv_cache_gb = 0.0;
            } else if (arg == "--dual-stream-overlap") {
                dual_overlap = true;
            } else if (arg == "--no-dual-stream-overlap") {
                dual_overlap = false;
            } else if (arg == "--dspark-tree-width" && i + 1 < argc) {
                const std::string w(argv[++i]);
                if (w == "8") dspark_tree_width = 8;
                else if (w == "16") dspark_tree_width = 16;
                else dspark_tree_width = 0;
            } else if (arg == "--dspark-tree16-max-ctx" && i + 1 < argc) {
                dspark_tree16_max_ctx = std::max(0, std::stoi(argv[++i]));
            } else if (arg == "--dspark-static-m8") {
                dspark_static_m8 = true;
            } else if (arg == "--dspark-dynamic-m8") {
                dspark_static_m8 = false;
            } else if (arg == "--dspark-dyn8-max-depth" && i + 1 < argc) {
                dspark_dyn8_max_depth = std::clamp(std::stoi(argv[++i]), 2, 5);
            } else if (arg == "--search-url" && i + 1 < argc) {
                srv_cfg.search_url = argv[++i];
            }
        }
        srv_cfg.default_max_tokens = std::min(srv_cfg.default_max_tokens, srv_cfg.max_output_tokens);

        std::signal(SIGINT, on_sig);
        std::signal(SIGTERM, on_sig);
        std::signal(SIGPIPE, SIG_IGN);

        dsv41::Dsv41Runtime runtime;
        runtime.set_nvme_kv_cache(nvme_kv_cache_dir, nvme_kv_cache_gb);
        if (disable_expert_perm) {
            runtime.set_expert_perm_overlay("");
        } else if (!expert_perm_overlay.empty()) {
            runtime.set_expert_perm_overlay(expert_perm_overlay);
        }
        runtime.set_csa2_shared_tree8(csa2_shared_tree8);
        runtime.set_fused_mhc_norm(fused_mhc_norm);
        runtime.set_fused_dspark_epilogue(fused_dspark_epilogue);
        runtime.set_hlut3_mtp_overlay(hlut3_mtp_overlay);
        if (!hlut3_dev3_overlay.empty()) {
            runtime.set_hlut3_dev3_overlay(hlut3_dev3_overlay, hlut3_dev3_id);
        }
        if (!hlut3_tp3_overlay.empty()) {
            runtime.set_hlut3_tp3_overlay(hlut3_tp3_overlay);
        }
        runtime.set_rocblas_prefill(rocblas_prefill);
        runtime.set_rocblas_expert_thresh(rocblas_expert_thresh);
        runtime.set_rocblas_hlut2_thresh(rocblas_hlut2_thresh);
        runtime.set_dspark_tree8_conf_thresh(tree8_conf_thresh);
        runtime.set_dspark_chain_max_b(chain_max_b);
        runtime.init(mi50_root, tok_path, p150_shm, srv_cfg.max_seq_len, rarebal_overlay, capture_rarebal_dir, vit_weights);
        runtime.set_prefill_superchunk_m(prefill_superchunk_m);
        runtime.set_dual_stream_overlap(dual_overlap);
        runtime.set_dspark_tree_width(dspark_tree_width);
        runtime.set_dspark_tree16_max_context(dspark_tree16_max_ctx);
        runtime.set_dspark_static_m8(dspark_static_m8);
        runtime.set_dspark_dyn8_max_depth(dspark_dyn8_max_depth);
        runtime.set_fused_mhc_norm(fused_mhc_norm);
        runtime.set_fused_dspark_epilogue(fused_dspark_epilogue);

        const int srv_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (srv_fd < 0) throw std::runtime_error("socket() failed");
        g_listen_fd.store(srv_fd, std::memory_order_relaxed);

        int reuse = 1;
        (void)::setsockopt(srv_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
            ::close(srv_fd);
            throw std::runtime_error("Invalid bind address: " + host);
        }
        if (::bind(srv_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            ::close(srv_fd);
            throw std::runtime_error("bind() failed on " + host + ":" + std::to_string(port));
        }
        if (::listen(srv_fd, 64) != 0) {
            ::close(srv_fd);
            throw std::runtime_error("listen() failed");
        }

        std::printf("[DSV41_FE] OpenAI + Anthropic HTTP/SSE server listening on http://%s:%d (/v1 + /anthropic/v1/messages, model=%s, max_seq_len=%d, effort=%d, max_out=%d/%d, dual_ovlp=%d, tree_w=%s, dyn_m8=%d[d<=%d,t16_ctx=%d], csa2_sh8=%d, mhc_fuse=%d, dsp_ep=%d, vit=%s, nvme_kv=%s(%.1fG), search=%s)\n",
                    host.c_str(), port, srv_cfg.default_model_id.c_str(), srv_cfg.max_seq_len,
                    srv_cfg.default_reasoning_effort, srv_cfg.default_max_tokens, srv_cfg.max_output_tokens,
                    int(dual_overlap),
                    (dspark_tree_width == 8 ? "8" : (dspark_tree_width == 16 ? "16" : "auto")),
                    dspark_static_m8 ? 0 : 1, dspark_dyn8_max_depth, dspark_tree16_max_ctx,
                    int(csa2_shared_tree8),
                    int(fused_mhc_norm),
                    int(fused_dspark_epilogue),
                    runtime.vision_ready() ? "ready" : "off",
                    nvme_kv_cache_dir.empty() ? "off" : nvme_kv_cache_dir.c_str(),
                    nvme_kv_cache_gb,
                    srv_cfg.search_url.empty() ? "off" : srv_cfg.search_url.c_str());
        std::fflush(stdout);

        while (!g_stop_server) {
            sockaddr_in cli{};
            socklen_t clen = sizeof(cli);
            const int cfd = ::accept(srv_fd, reinterpret_cast<sockaddr*>(&cli), &clen);
            if (cfd < 0) {
                if (errno == EINTR && !g_stop_server) continue;
                break;
            }
            std::thread(handle_connection, cfd, &runtime, srv_cfg).detach();
        }

        runtime.flush_nvme_kv_cache();
        ::close(srv_fd);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[DSV41_FE] Fatal error: %s\n", e.what());
        return 1;
    }
}
