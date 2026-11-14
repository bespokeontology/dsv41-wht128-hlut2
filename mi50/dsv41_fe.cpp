// Pure-C++17 OpenAI-Compatible HTTP + SSE Server for DeepSeek-V4.1-Flash-1017 on 127.0.0.1:8042
// Drives Dsv41Runtime (3x MI50 + 2x Blackhole P150A + Dual-NVMe Engram + 3-Stage DSpark M=6).
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
void on_sig(int) { g_stop_server = 1; }

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

// Extract complete UTF-8 prefix from `pending_bytes`, leaving any trailing incomplete 2/3/4-byte sequence in `pending_bytes`
std::string pop_complete_utf8(std::string& pending_bytes) {
    if (pending_bytes.empty()) return {};
    const size_t n = pending_bytes.size();
    size_t check = n;
    for (size_t back = 1; back <= 3 && back <= n; ++back) {
        const uint8_t b = static_cast<uint8_t>(pending_bytes[n - back]);
        if ((b & 0x80u) == 0u) {
            break;  // ASCII byte is complete
        }
        if ((b & 0xC0u) == 0xC0u) {
            size_t expected = 1;
            if ((b & 0xE0u) == 0xC0u) expected = 2;
            else if ((b & 0xF0u) == 0xE0u) expected = 3;
            else if ((b & 0xF8u) == 0xF0u) expected = 4;
            if (back < expected) {
                check = n - back;  // Incomplete multi-byte sequence at tail!
            }
            break;
        }
    }
    std::string out = pending_bytes.substr(0, check);
    pending_bytes.erase(0, check);
    return out;
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

void log_fe_turn_telemetry(
    uint64_t req_num,
    const char* tag,
    const dsv41::Dsv41GenStats& stats,
    size_t tool_count,
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
    char extra_note[128] = "";
    int off = 0;
    if (server_search_reqs > 0) {
        off += std::snprintf(extra_note + off, sizeof(extra_note) - size_t(off), " | srv_search=%d", server_search_reqs);
    }
    if (stats.think_guard_triggered && size_t(off) < sizeof(extra_note)) {
        std::snprintf(extra_note + off, sizeof(extra_note) - size_t(off), " | ThinkGuard(rew=%u,p=%u)",
                      stats.think_guard_rewound, stats.think_guard_period);
    }
    std::printf("[DSV41_FE] #%llu %s OK | prompt=%d (reuse=%d [radix=%d] in %.2fms=%.0f tok/s | fresh=%d in %.1fms=%.1f tok/s) | "
                "gen=%d in %.1fms=%.2f tok/s [bb=%.0fms p150=%.0fms dsp=%.0fms smp=%.0fms steps=%llu tau=%.2f] | "
                "stages[attn=%.0fms hsi=%.0fms moe=%.0fms p150w=%.0fms ovlp=%.0fms ar=%.0fms egw=%.1fms] | "
                "DSpark acc=%llu/%llu (%.1f%%, tree=%llu/%llu, tree16=%llu/%llu) | tools=%zu%s\n",
                static_cast<unsigned long long>(req_num), tag,
                stats.prompt_tokens,
                stats.cached_prefix_tokens, stats.radix_hit_tokens, stats.prefix_reuse_ms, reuse_tps,
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
                static_cast<unsigned long long>(stats.dspark_tree16_accepted),
                static_cast<unsigned long long>(stats.dspark_tree16_steps),
                tool_count, extra_note);
    std::fflush(stdout);
}

void handle_connection(int fd, dsv41::Dsv41Runtime* rt, const std::string& default_model_id, const std::string& search_url) {
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
            if (j.contains("prompt") && j.at("prompt").is_string()) {
                text = j.at("prompt").as_str();
            } else if (j.contains("text") && j.at("text").is_string()) {
                text = j.at("text").as_str();
            } else if (j.contains("messages") && j.at("messages").is_array()) {
                std::string tmode;
                int budget = 75;
                text = dsv41::render_dsv41_chat_prompt(
                    j.at("messages"), j.find("tools"), j.find("response_format"), j.find("reasoning_effort"),
                    tmode, budget);
            }
            const auto ids = rt->tokenizer().encode(text, true);
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

            dsv41::JsonValue body = dsv41::parse_json(req.body);
            if (!body.contains("messages") || !body.at("messages").is_array()) {
                throw std::runtime_error("Missing 'messages' array in chat/completions request");
            }

            std::string model_name = default_model_id;
            if (body.contains("model") && body.at("model").is_string() && !body.at("model").as_str().empty()) {
                model_name = body.at("model").as_str();
            }

            // Respect tool_choice == "none"
            const dsv41::JsonValue* tools_ptr = body.find("tools");
            if (body.contains("tool_choice") && body.at("tool_choice").is_string() &&
                body.at("tool_choice").as_str() == "none") {
                tools_ptr = nullptr;
            }

            // Extract reasoning_effort (checking top-level and chat_template_kwargs, defaulting empty to "55" sweet spot!)
            dsv41::JsonValue effort_override;
            const dsv41::JsonValue* effort_ptr = body.find("reasoning_effort");
            if ((!effort_ptr || effort_ptr->is_null() || (effort_ptr->is_string() && effort_ptr->as_str().empty())) &&
                body.contains("chat_template_kwargs") && body.at("chat_template_kwargs").is_object()) {
                const auto& ctk = body.at("chat_template_kwargs");
                if (ctk.contains("enable_thinking") && ctk.at("enable_thinking").is_bool() && !ctk.at("enable_thinking").as_bool()) {
                    effort_override = dsv41::JsonValue("none");
                    effort_ptr = &effort_override;
                } else if (ctk.contains("reasoning_effort") && !ctk.at("reasoning_effort").is_null()) {
                    effort_ptr = ctk.find("reasoning_effort");
                }
            }
            if (!effort_ptr || effort_ptr->is_null() || (effort_ptr->is_string() && effort_ptr->as_str().empty())) {
                effort_override = dsv41::JsonValue("55");
                effort_ptr = &effort_override;
            }

            std::string thinking_mode;
            int thinking_budget = 55;
            const std::string prompt_text = dsv41::render_dsv41_chat_prompt(
                body.at("messages"), tools_ptr, body.find("response_format"), effort_ptr,
                thinking_mode, thinking_budget);
            const std::vector<int32_t> prompt_ids = rt->tokenizer().encode(prompt_text, true);

            dsv41::Dsv41GenParams params;
            params.temperature = 0.95f;
            params.top_p = 0.95f;
            params.max_new_tokens = 131072;
            params.enable_dspark = true;

            if (body.contains("temperature") && body.at("temperature").is_number()) {
                params.temperature = static_cast<float>(std::max(0.0, body.at("temperature").as_number()));
            }
            if (body.contains("top_p") && body.at("top_p").is_number()) {
                params.top_p = static_cast<float>(std::clamp(body.at("top_p").as_number(), 0.01, 1.0));
            }
            if (body.contains("repetition_penalty") && body.at("repetition_penalty").is_number()) {
                params.repetition_penalty = static_cast<float>(std::max(1.0, body.at("repetition_penalty").as_number()));
            }
            if (body.contains("max_completion_tokens") && body.at("max_completion_tokens").is_number()) {
                params.max_new_tokens = static_cast<int>(std::clamp<int64_t>(body.at("max_completion_tokens").as_int(), 1, 131072));
            } else if (body.contains("max_tokens") && body.at("max_tokens").is_number()) {
                params.max_new_tokens = static_cast<int>(std::clamp<int64_t>(body.at("max_tokens").as_int(), 1, 131072));
            }
            if (body.contains("seed") && body.at("seed").is_number()) {
                params.seed = static_cast<uint64_t>(body.at("seed").as_int());
            }
            if (body.contains("dspark") && body.at("dspark").is_bool()) {
                params.enable_dspark = body.at("dspark").as_bool();
            }
            if (body.contains("think_loop_guard") && body.at("think_loop_guard").is_bool()) {
                params.enable_think_loop_guard = body.at("think_loop_guard").as_bool();
            }

            const bool stream = body.contains("stream") && body.at("stream").is_bool() && body.at("stream").as_bool();
            bool include_usage = true;
            if (body.contains("stream_options") && body.at("stream_options").is_object()) {
                const auto& so = body.at("stream_options");
                if (so.contains("include_usage") && so.at("include_usage").is_bool()) {
                    include_usage = so.at("include_usage").as_bool();
                }
            }

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
                env.set_key("model", model_name);
                env.set_key("choices", std::move(choices));
                return "data: " + env.dumps_compact() + "\n\n";
            };

            if (stream) {
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
                    env.set_key("model", model_name);
                    env.set_key("choices", dsv41::JsonValue::make_array());
                    env.set_key("x_flash_progress", std::move(prog));
                    if (include_usage && total > 0 && serviced >= total) {
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

                dsv41::Dsv41StreamParser sp(thinking_mode);
                std::string utf8_pending;
                uint64_t gen_tok_cnt = 0;
                bool any_streamed_tools = false;
                size_t streamed_tool_cnt = 0;
                std::vector<dsv41::Dsv41ToolStreamDelta> tool_deltas;

                auto emit_tool_deltas = [&](const std::vector<dsv41::Dsv41ToolStreamDelta>& deltas) -> bool {
                    for (const auto& td : deltas) {
                        any_streamed_tools = true;
                        if (td.index + 1 > streamed_tool_cnt) streamed_tool_cnt = td.index + 1;

                        dsv41::JsonValue fn = dsv41::JsonValue::make_object();
                        if (td.is_start) {
                            fn.set_key("name", td.name);
                            fn.set_key("arguments", "");
                        } else {
                            fn.set_key("arguments", td.arguments_delta);
                        }

                        dsv41::JsonValue item = dsv41::JsonValue::make_object();
                        item.set_key("index", int64_t(td.index));
                        if (td.is_start) {
                            item.set_key("id", td.id);
                            item.set_key("type", "function");
                        }
                        item.set_key("function", std::move(fn));

                        dsv41::JsonValue tc_arr = dsv41::JsonValue::make_array();
                        tc_arr.push_back(std::move(item));

                        dsv41::JsonValue d = dsv41::JsonValue::make_object();
                        d.set_key("tool_calls", std::move(tc_arr));
                        if (!sse_write(make_chunk_envelope(std::move(d), nullptr))) {
                            return false;
                        }
                    }
                    return true;
                };

                dsv41::Dsv41GenStats stats;
                const auto gen_ids = rt->generate(
                    prompt_ids,
                    params,
                    [&](int32_t tok_id) -> bool {
                        if (aborted.load(std::memory_order_relaxed) || client_disconnected(fd)) {
                            aborted.store(true, std::memory_order_release);
                            return false;
                        }
                        ++gen_tok_cnt;
                        live_progress.observe_token(gen_tok_cnt);

                        if (tok_id == dsv41::Dsv41SpecialTokens::kThinkEnd && !utf8_pending.empty()) {
                            utf8_pending.clear();
                        }
                        if (stats.think_guard_triggered &&
                            tok_id == dsv41::Dsv41SpecialTokens::kThinkEnd &&
                            sp.in_thinking()) {
                            std::string r_delta;
                            sp.force_close_thinking(r_delta);
                            if (!r_delta.empty()) {
                                dsv41::JsonValue d = dsv41::JsonValue::make_object();
                                d.set_key("reasoning_content", r_delta);
                                if (!sse_write(make_chunk_envelope(std::move(d), nullptr))) {
                                    return false;
                                }
                            }
                            return true;
                        }
                        utf8_pending += rt->tokenizer().decode_single(tok_id, false);
                        const std::string piece = pop_complete_utf8(utf8_pending);
                        if (!piece.empty()) {
                            std::string r_delta, c_delta;
                            sp.feed(piece, r_delta, c_delta, &tool_deltas, call_seed);
                            if (!r_delta.empty()) {
                                dsv41::JsonValue d = dsv41::JsonValue::make_object();
                                d.set_key("reasoning_content", r_delta);
                                if (!sse_write(make_chunk_envelope(std::move(d), nullptr))) {
                                    return false;
                                }
                            }
                            if (!c_delta.empty()) {
                                dsv41::JsonValue d = dsv41::JsonValue::make_object();
                                d.set_key("content", c_delta);
                                if (!sse_write(make_chunk_envelope(std::move(d), nullptr))) {
                                    return false;
                                }
                            }
                            if (!tool_deltas.empty() && !emit_tool_deltas(tool_deltas)) {
                                return false;
                            }
                            if (sp.should_stop_generation()) {
                                return false;
                            }
                        }
                        return true;
                    },
                    stats,
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
                    });

                live_progress.finish();
                hb_guard.stop_and_join();

                if (!aborted.load(std::memory_order_acquire)) {
                    (void)emit_progress(true);

                    if (!utf8_pending.empty()) {
                        std::string r_d, c_d;
                        sp.feed(utf8_pending, r_d, c_d, &tool_deltas, call_seed);
                        utf8_pending.clear();
                        if (!r_d.empty()) {
                            dsv41::JsonValue d = dsv41::JsonValue::make_object();
                            d.set_key("reasoning_content", r_d);
                            (void)sse_write(make_chunk_envelope(std::move(d), nullptr));
                        }
                        if (!c_d.empty()) {
                            dsv41::JsonValue d = dsv41::JsonValue::make_object();
                            d.set_key("content", c_d);
                            (void)sse_write(make_chunk_envelope(std::move(d), nullptr));
                        }
                        if (!tool_deltas.empty()) {
                            (void)emit_tool_deltas(tool_deltas);
                        }
                    }

                    std::string r_tail, c_tail;
                    dsv41::Dsv41ParsedCompletion parsed_final;
                    sp.finish(r_tail, c_tail, parsed_final, call_seed, &tool_deltas);

                    if (!r_tail.empty()) {
                        dsv41::JsonValue d = dsv41::JsonValue::make_object();
                        d.set_key("reasoning_content", r_tail);
                        (void)sse_write(make_chunk_envelope(std::move(d), nullptr));
                    }
                    if (!c_tail.empty()) {
                        dsv41::JsonValue d = dsv41::JsonValue::make_object();
                        d.set_key("content", c_tail);
                        (void)sse_write(make_chunk_envelope(std::move(d), nullptr));
                    }
                    if (!tool_deltas.empty()) {
                        (void)emit_tool_deltas(tool_deltas);
                    }

                    // Prevent harness EMPTY_RESPONSE auto-retry on pure whitespace/EOS stop
                    if (!any_streamed_tools && parsed_final.tool_calls.empty() && parsed_final.content.empty() &&
                        parsed_final.reasoning_content.empty()) {
                        dsv41::JsonValue d = dsv41::JsonValue::make_object();
                        d.set_key("content", " ");
                        (void)sse_write(make_chunk_envelope(std::move(d), nullptr));
                    }

                    const char* fin_reason = "stop";
                    if (any_streamed_tools) {
                        fin_reason = "tool_calls";
                    } else if (!parsed_final.tool_calls.empty()) {
                        fin_reason = "tool_calls";
                        dsv41::JsonValue tc_arr = dsv41::JsonValue::make_array();
                        for (size_t i = 0; i < parsed_final.tool_calls.size(); ++i) {
                            const auto& tc = parsed_final.tool_calls[i];
                            dsv41::JsonValue fn = dsv41::JsonValue::make_object();
                            fn.set_key("name", tc.name);
                            fn.set_key("arguments", tc.arguments_json);

                            dsv41::JsonValue item = dsv41::JsonValue::make_object();
                            item.set_key("index", int64_t(i));
                            item.set_key("id", tc.id);
                            item.set_key("type", "function");
                            item.set_key("function", std::move(fn));
                            tc_arr.push_back(std::move(item));
                        }
                        dsv41::JsonValue d = dsv41::JsonValue::make_object();
                        d.set_key("tool_calls", std::move(tc_arr));
                        (void)sse_write(make_chunk_envelope(std::move(d), nullptr));
                    } else if (stats.completion_tokens >= params.max_new_tokens) {
                        fin_reason = "length";
                    }

                    (void)sse_write(make_chunk_envelope(dsv41::JsonValue::make_object(), fin_reason));

                    if (include_usage) {
                        dsv41::JsonValue env = dsv41::JsonValue::make_object();
                        env.set_key("id", cmpl_id);
                        env.set_key("object", "chat.completion.chunk");
                        env.set_key("created", created_ts);
                        env.set_key("model", model_name);
                        env.set_key("choices", dsv41::JsonValue::make_array());
                        env.set_key("usage", make_usage_json(
                                                 int64_t(stats.prompt_tokens),
                                                 int64_t(stats.completion_tokens),
                                                 int64_t(stats.cached_prefix_tokens)));
                        (void)sse_write("data: " + env.dumps_compact() + "\n\n");
                    }
                    (void)sse_write("data: [DONE]\n\n");

                    log_fe_turn_telemetry(
                        req_num, "stream", stats,
                        std::max(parsed_final.tool_calls.size(), streamed_tool_cnt));
                } else {
                    std::printf("[DSV41_FE] #%llu stream aborted by client disconnect after %d tokens\n",
                                static_cast<unsigned long long>(req_num), stats.completion_tokens);
                    std::fflush(stdout);
                }
            } else {
                // Non-streaming chat/completions
                dsv41::Dsv41GenStats stats;
                const auto gen_ids = rt->generate(
                    prompt_ids, params,
                    [&](int32_t) -> bool { return !client_disconnected(fd); },
                    stats);
                const std::string raw_text = rt->tokenizer().decode(gen_ids, false);
                const auto parsed = dsv41::parse_dsv41_completion(raw_text, thinking_mode, call_seed);

                dsv41::JsonValue msg = dsv41::JsonValue::make_object();
                msg.set_key("role", "assistant");
                if (parsed.content.empty() && !parsed.tool_calls.empty()) {
                    msg.set_key("content", dsv41::JsonValue());
                } else {
                    msg.set_key("content", parsed.content);
                }
                if (!parsed.reasoning_content.empty()) {
                    msg.set_key("reasoning_content", parsed.reasoning_content);
                }
                const char* fin_reason = "stop";
                if (!parsed.tool_calls.empty()) {
                    fin_reason = "tool_calls";
                    dsv41::JsonValue tc_arr = dsv41::JsonValue::make_array();
                    for (const auto& tc : parsed.tool_calls) {
                        dsv41::JsonValue fn = dsv41::JsonValue::make_object();
                        fn.set_key("name", tc.name);
                        fn.set_key("arguments", tc.arguments_json);

                        dsv41::JsonValue item = dsv41::JsonValue::make_object();
                        item.set_key("id", tc.id);
                        item.set_key("type", "function");
                        item.set_key("function", std::move(fn));
                        tc_arr.push_back(std::move(item));
                    }
                    msg.set_key("tool_calls", std::move(tc_arr));
                } else if (stats.completion_tokens >= params.max_new_tokens) {
                    fin_reason = "length";
                }

                dsv41::JsonValue choice = dsv41::JsonValue::make_object();
                choice.set_key("index", int64_t(0));
                choice.set_key("message", std::move(msg));
                choice.set_key("finish_reason", std::string(fin_reason));

                dsv41::JsonValue choices = dsv41::JsonValue::make_array();
                choices.push_back(std::move(choice));

                dsv41::JsonValue root = dsv41::JsonValue::make_object();
                root.set_key("id", cmpl_id);
                root.set_key("object", "chat.completion");
                root.set_key("created", created_ts);
                root.set_key("model", model_name);
                root.set_key("choices", std::move(choices));
                root.set_key("usage", make_usage_json(
                                          int64_t(stats.prompt_tokens),
                                          int64_t(stats.completion_tokens),
                                          int64_t(stats.cached_prefix_tokens)));

                send_json_response(fd, 200, "OK", root.dumps_compact());
                log_fe_turn_telemetry(req_num, "sync", stats, parsed.tool_calls.size());
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
            const dsv41::AnthropicTranslatedRequest treq = dsv41::translate_anthropic_to_dsv41(body, default_model_id);

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

            const bool ok = dsv41::run_anthropic_messages_turn(
                treq,
                rt->tokenizer(),
                search_url,
                msg_id,
                call_seed,
                [&](const std::vector<int32_t>& prompt_ids,
                    int max_new_tokens,
                    dsv41::TokenStreamCallback on_tok,
                    dsv41::Dsv41GenStats& out_stats) -> std::vector<int32_t> {
                    dsv41::Dsv41GenParams params;
                    params.max_new_tokens = std::clamp(max_new_tokens, 1, 131072);
                    params.temperature = treq.has_explicit_temperature ? std::max(0.0f, treq.temperature) : 0.95f;
                    params.top_p = std::clamp(treq.top_p, 0.01f, 1.0f);
                    params.seed = treq.seed;
                    params.enable_dspark = true;
                    return rt->generate(prompt_ids, params, std::move(on_tok), out_stats);
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
                    agg_stats, total_tools, srv_search_reqs);
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
    try {
        const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                         ? std::string(std::getenv("DSV41_ROOT"))
                                         : std::string(".");
        std::string host = "127.0.0.1";
        int port = 8042;
        std::string mi50_root = root_env + "/packed_mi50";
        std::string tok_path  = root_env + "/hf_meta/tokenizer.json";
        std::string p150_shm  = "/dsv41_p150_ipc";
        std::string model_id  = "DeepSeek-V4.1-Flash-1017";
        std::string rarebal_overlay;
        std::string capture_rarebal_dir;
        std::string search_url;
        int max_seq_len        = 1048576;
        bool dual_overlap      = true;
        int dspark_tree_width  = 0;  // 0 = auto, 8 = fixed M=8, 16 = fixed M=16

        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--link-check") {
                std::cout << "[DSV41_FE] Link check OK\n";
                return 0;
            } else if (arg == "--model-dir" && i + 1 < argc) {
                const std::string md = argv[++i];
                mi50_root = md + "/packed_mi50";
                tok_path  = md + "/hf_meta/tokenizer.json";
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
                model_id = argv[++i];
            } else if (arg == "--max-seq-len" && i + 1 < argc) {
                max_seq_len = std::stoi(argv[++i]);
            } else if (arg == "--rarebal-overlay" && i + 1 < argc) {
                rarebal_overlay = argv[++i];
            } else if (arg == "--no-rarebal-overlay") {
                rarebal_overlay.clear();
            } else if (arg == "--capture-rarebal" && i + 1 < argc) {
                capture_rarebal_dir = argv[++i];
            } else if (arg == "--dual-stream-overlap") {
                dual_overlap = true;
            } else if (arg == "--no-dual-stream-overlap") {
                dual_overlap = false;
            } else if (arg == "--dspark-tree-width" && i + 1 < argc) {
                const std::string w(argv[++i]);
                if (w == "8") dspark_tree_width = 8;
                else if (w == "16") dspark_tree_width = 16;
                else dspark_tree_width = 0;
            } else if (arg == "--search-url" && i + 1 < argc) {
                search_url = argv[++i];
            }
        }

        std::signal(SIGINT, on_sig);
        std::signal(SIGTERM, on_sig);
        std::signal(SIGPIPE, SIG_IGN);

        dsv41::Dsv41Runtime runtime;
        runtime.init(mi50_root, tok_path, p150_shm, max_seq_len, rarebal_overlay, capture_rarebal_dir);
        runtime.set_dual_stream_overlap(dual_overlap);
        runtime.set_dspark_tree_width(dspark_tree_width);

        const int srv_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (srv_fd < 0) throw std::runtime_error("socket() failed");

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

        std::printf("[DSV41_FE] OpenAI + Anthropic HTTP/SSE server listening on http://%s:%d (/v1 + /anthropic/v1/messages, model=%s, max_seq_len=%d, dual_ovlp=%d, tree_w=%s, search=%s)\n",
                    host.c_str(), port, model_id.c_str(), max_seq_len, int(dual_overlap),
                    (dspark_tree_width == 8 ? "8" : (dspark_tree_width == 16 ? "16" : "auto")),
                    search_url.empty() ? "off" : search_url.c_str());
        std::fflush(stdout);

        while (!g_stop_server) {
            sockaddr_in cli{};
            socklen_t clen = sizeof(cli);
            const int cfd = ::accept(srv_fd, reinterpret_cast<sockaddr*>(&cli), &clen);
            if (cfd < 0) {
                if (errno == EINTR) continue;
                break;
            }
            std::thread(handle_connection, cfd, &runtime, model_id, search_url).detach();
        }

        ::close(srv_fd);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[DSV41_FE] Fatal error: %s\n", e.what());
        return 1;
    }
}
