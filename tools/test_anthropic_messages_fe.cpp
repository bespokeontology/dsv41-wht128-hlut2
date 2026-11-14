// Wave 8E Deterministic Verification Suite:
//   1. Anthropic /anthropic/v1/messages JSON -> DeepSeek-V4.1 prompt & DSML schema translation
//   2. Streaming SSE (<think> -> thinking_delta + signature_delta, text_delta, DSML -> client tool_use + input_json_delta)
//   3. Mid-turn pure-C++ POSIX socket --search-url loop (server_tool_use + web_search_tool_result + 0-re-prefill LCP continuation)
//   4. Multi-turn RadixBlockCache token-prefix identity across echoed server-tool history
// Pure C++17, zero Python anywhere.

#include "../mi50/dsv41_anthropic_messages.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace dsv41;

namespace {

#define CHECK_TRUE(cond, msg)                                                           \
    do {                                                                                \
        if (!(cond)) {                                                                  \
            std::fprintf(stderr, "[FAIL] %s (%s:%d)\n", (msg), __FILE__, __LINE__);     \
            return false;                                                               \
        }                                                                               \
    } while (0)

struct EphemeralSearchServer {
    int srv_fd = -1;
    int port = 0;
    std::atomic<bool> stop{false};
    std::atomic<int> hits{0};
    std::string last_request_line;
    std::thread worker;

    bool start() {
        srv_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (srv_fd < 0) return false;
        int one = 1;
        (void)::setsockopt(srv_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = 0;
        (void)::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        if (::bind(srv_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            ::close(srv_fd);
            srv_fd = -1;
            return false;
        }
        if (::listen(srv_fd, 16) != 0) {
            ::close(srv_fd);
            srv_fd = -1;
            return false;
        }
        sockaddr_in bound{};
        socklen_t blen = sizeof(bound);
        if (::getsockname(srv_fd, reinterpret_cast<sockaddr*>(&bound), &blen) != 0) {
            ::close(srv_fd);
            srv_fd = -1;
            return false;
        }
        port = ntohs(bound.sin_port);

        worker = std::thread([this]() {
            while (!stop.load(std::memory_order_acquire)) {
                sockaddr_in cli{};
                socklen_t clen = sizeof(cli);
                int cfd = ::accept(srv_fd, reinterpret_cast<sockaddr*>(&cli), &clen);
                if (cfd < 0) break;
                char buf[4096];
                ssize_t r = ::recv(cfd, buf, sizeof(buf) - 1, 0);
                if (r > 0) {
                    buf[r] = '\0';
                    std::string req(buf, size_t(r));
                    size_t eol = req.find("\r\n");
                    if (eol != std::string::npos) last_request_line = req.substr(0, eol);
                }
                hits.fetch_add(1, std::memory_order_relaxed);
                const std::string json_body =
                    "{\"results\":["
                    "{\"title\":\"Tenstorrent Blackhole P150A Architecture\","
                    "\"url\":\"https://docs.tenstorrent.com/aibs/blackhole/index.html\","
                    "\"content\":\"Blackhole P150A provides 110 active Tensix cores with 1.5 MiB L1 SRAM per core and 32 GB GDDR6.\"}"
                    "]}";
                std::string http_resp =
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: application/json\r\n"
                    "Content-Length: " + std::to_string(json_body.size()) + "\r\n"
                    "Connection: close\r\n\r\n" + json_body;
                (void)::send(cfd, http_resp.data(), http_resp.size(), MSG_NOSIGNAL);
                ::close(cfd);
            }
        });
        return true;
    }

    ~EphemeralSearchServer() {
        stop.store(true, std::memory_order_release);
        if (srv_fd >= 0) {
            ::shutdown(srv_fd, SHUT_RDWR);
            ::close(srv_fd);
            srv_fd = -1;
        }
        if (worker.joinable()) worker.join();
    }
};

bool test_translation_and_client_tool_sse(const Dsv41Tokenizer& tok) {
    const std::string req_json = R"JSON({
        "model": "DeepSeek-V4.1-Flash-1017",
        "max_tokens": 2048,
        "stream": true,
        "thinking": {"type": "enabled", "budget_tokens": 12000},
        "system": [{"type": "text", "text": "You are a systems engineer."}],
        "tools": [
            {
                "name": "read_kernel_file",
                "description": "Read a C++ HIP kernel source file.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }
        ],
        "messages": [
            {"role": "user", "content": "Inspect mi50/dsv41_mi50_kernels.hip."}
        ]
    })JSON";

    const JsonValue body = parse_json(req_json);
    const AnthropicTranslatedRequest treq = translate_anthropic_to_dsv41(body, "dsv41");
    CHECK_TRUE(treq.stream, "stream flag parsed");
    CHECK_TRUE(treq.thinking_mode == "thinking", "thinking enabled");
    CHECK_TRUE(treq.reasoning_effort.as_int() == 75, "budget_tokens=12000 maps to effort=75");
    CHECK_TRUE(treq.openai_tools.arr_val.size() == 1, "custom tool translated");

    // Simulate model streaming <think> + DSML client tool invoke (search_url="" -> client tool_use!)
    const std::string dsml = "\xEF\xBD\x9C" "DSML\xEF\xBD\x9C";
    const std::string mock_completion =
        "Let us read the kernel file first.</think>\n\n<" + dsml + " calls>\n<" + dsml +
        " invoke name=\"read_kernel_file\">\n<" + dsml +
        " parameter name=\"path\" string=\"true\">mi50/dsv41_mi50_kernels.hip</" + dsml +
        " parameter>\n</" + dsml + " invoke>\n</" + dsml + " calls>";
    const std::vector<int32_t> mock_ids = tok.encode(mock_completion, true);

    std::vector<std::string> sse_frames;
    std::string sync_json;
    Dsv41GenStats agg_stats{};
    int srv_reqs = 0;
    size_t n_tools = 0;

    const bool ok = run_anthropic_messages_turn(
        treq,
        tok,
        /*search_url=*/"",
        "msg_test_client_tool",
        1001ULL,
        [&](const std::vector<int32_t>& prompt_ids,
            int /*max_new_tokens*/,
            TokenStreamCallback on_tok,
            Dsv41GenStats& out_stats) -> std::vector<int32_t> {
            out_stats.prompt_tokens = int(prompt_ids.size());
            out_stats.cached_prefix_tokens = 128;
            std::vector<int32_t> emitted;
            for (int32_t id : mock_ids) {
                emitted.push_back(id);
                out_stats.completion_tokens = int(emitted.size());
                if (!on_tok(id)) break;
            }
            return emitted;
        },
        []() { return false; },
        [&](const std::string& frame) {
            sse_frames.push_back(frame);
            return true;
        },
        sync_json,
        agg_stats,
        srv_reqs,
        n_tools);

    CHECK_TRUE(ok, "run_anthropic_messages_turn client tool SSE ok");
    CHECK_TRUE(srv_reqs == 0, "0 server search requests when --search-url unset");
    CHECK_TRUE(n_tools == 1, "1 client tool call parsed");

    std::string all_sse;
    for (const auto& f : sse_frames) all_sse += f;
    CHECK_TRUE(all_sse.find("event: message_start") != std::string::npos, "message_start emitted");
    CHECK_TRUE(all_sse.find("\"thinking_delta\"") != std::string::npos, "thinking_delta emitted");
    CHECK_TRUE(all_sse.find("\"signature_delta\"") != std::string::npos, "signature_delta emitted");
    CHECK_TRUE(all_sse.find("\"type\":\"tool_use\"") != std::string::npos, "client tool_use block emitted");
    CHECK_TRUE(all_sse.find("\"input_json_delta\"") != std::string::npos, "input_json_delta emitted");
    CHECK_TRUE(all_sse.find("mi50/dsv41_mi50_kernels.hip") != std::string::npos, "tool path argument streamed");
    CHECK_TRUE(all_sse.find("\"stop_reason\":\"tool_use\"") != std::string::npos, "stop_reason is tool_use");
    CHECK_TRUE(all_sse.find("event: message_stop") != std::string::npos, "message_stop emitted");

    std::printf("[PASS] Case 1 & 2: Anthropic JSON -> DSV41 translation + streaming <think>/DSML -> tool_use SSE\n");
    return true;
}

bool test_server_search_loop_and_radix_prefix(const Dsv41Tokenizer& tok) {
    EphemeralSearchServer mock_srv;
    CHECK_TRUE(mock_srv.start(), "ephemeral POSIX search server started");
    const std::string search_url = "http://127.0.0.1:" + std::to_string(mock_srv.port);

    const std::string req_json = R"JSON({
        "model": "DeepSeek-V4.1-Flash-1017",
        "max_tokens": 4096,
        "stream": true,
        "thinking": {"type": "enabled", "budget_tokens": 8192},
        "tools": [
            {"type": "web_search_20250305", "name": "web_search"}
        ],
        "messages": [
            {"role": "user", "content": "How much L1 SRAM per Tensix core does Blackhole P150A have?"}
        ]
    })JSON";

    const AnthropicTranslatedRequest treq = translate_anthropic_to_dsv41(parse_json(req_json), "dsv41");
    const std::string dsml = "\xEF\xBD\x9C" "DSML\xEF\xBD\x9C";

    const std::string step0_raw =
        "Let me search the Blackhole P150A specs.</think>\n\n<" + dsml + " calls>\n<" + dsml +
        " invoke name=\"deepseek_web_search\">\n<" + dsml +
        " parameter name=\"query\" string=\"true\">Blackhole P150A L1 SRAM</" + dsml +
        " parameter>\n</" + dsml + " invoke>\n</" + dsml + " calls>";
    const std::string step1_raw =
        "The documentation confirms 1.5 MiB L1 per Tensix core across 110 cores.</think>"
        "Each active Tensix core on Tenstorrent Blackhole P150A has 1.5 MiB of L1 SRAM.";

    const std::vector<int32_t> step0_ids = tok.encode(step0_raw, true);
    const std::vector<int32_t> step1_ids = tok.encode(step1_raw, true);

    std::vector<std::vector<int32_t>> seen_prompts;
    std::vector<int32_t> step0_active_seq;
    int step_idx = 0;
    std::string all_sse;
    std::string sync_json;
    Dsv41GenStats agg_stats{};
    int srv_reqs = 0;
    size_t n_tools = 0;

    const bool ok = run_anthropic_messages_turn(
        treq,
        tok,
        search_url,
        "msg_test_srv_search",
        2001ULL,
        [&](const std::vector<int32_t>& prompt_ids,
            int /*max_new_tokens*/,
            TokenStreamCallback on_tok,
            Dsv41GenStats& out_stats) -> std::vector<int32_t> {
            seen_prompts.push_back(prompt_ids);
            out_stats.prompt_tokens = int(prompt_ids.size());
            const auto& src = (step_idx == 0) ? step0_ids : step1_ids;
            std::vector<int32_t> gen;
            for (int32_t id : src) {
                gen.push_back(id);
                out_stats.completion_tokens = int(gen.size());
                if (!on_tok(id)) break;
            }
            if (step_idx == 0) {
                step0_active_seq = prompt_ids;
                step0_active_seq.insert(step0_active_seq.end(), gen.begin(), gen.end());
            } else {
                // Simulate Dsv41Runtime Tier-0 LCP prefix hit!
                size_t lcp = 0;
                while (lcp < step0_active_seq.size() && lcp < prompt_ids.size() &&
                       step0_active_seq[lcp] == prompt_ids[lcp]) {
                    ++lcp;
                }
                out_stats.cached_prefix_tokens = int(lcp);
            }
            ++step_idx;
            return gen;
        },
        []() { return false; },
        [&](const std::string& frame) {
            all_sse += frame;
            return true;
        },
        sync_json,
        agg_stats,
        srv_reqs,
        n_tools);

    CHECK_TRUE(ok, "server-side search mid-turn SSE loop ok");
    CHECK_TRUE(mock_srv.hits.load() == 1, "mock POSIX search server hit once");
    CHECK_TRUE(mock_srv.last_request_line.find("q=Blackhole+P150A+L1+SRAM") != std::string::npos,
               "URL-encoded search query received");
    CHECK_TRUE(srv_reqs == 1, "srv_search_reqs == 1");
    CHECK_TRUE(seen_prompts.size() == 2, "2 generation sub-steps executed in single HTTP turn");

    // Verify 100% Tier-0 LCP token prefix match between sub-step 0 active_seq and sub-step 1 prompt_ids!
    CHECK_TRUE(seen_prompts[1].size() > step0_active_seq.size(), "sub-step 1 extends sub-step 0 sequence");
    for (size_t i = 0; i < step0_active_seq.size(); ++i) {
        CHECK_TRUE(seen_prompts[1][i] == step0_active_seq[i], "100% zero-re-prefill LCP token match at mid-turn resume");
    }

    // Parse every SSE data frame to verify 100% valid JSON and reconstruct streamed thinking + text!
    std::string concat_thinking;
    std::string concat_text;
    size_t pos = 0;
    while (pos < all_sse.size()) {
        const size_t dpos = all_sse.find("data: ", pos);
        if (dpos == std::string::npos) break;
        const size_t eol = all_sse.find('\n', dpos + 6);
        if (eol == std::string::npos) break;
        const JsonValue ev = parse_json(all_sse.substr(dpos + 6, eol - (dpos + 6)));
        if (ev.get("type").as_str() == "content_block_delta") {
            const JsonValue& d = ev.get("delta");
            if (d.get("type").as_str() == "thinking_delta") concat_thinking += d.get("thinking").as_str();
            else if (d.get("type").as_str() == "text_delta") concat_text += d.get("text").as_str();
        }
        pos = eol + 1;
    }

    CHECK_TRUE(all_sse.find("\"type\":\"server_tool_use\"") != std::string::npos, "server_tool_use emitted");
    CHECK_TRUE(all_sse.find("\"name\":\"web_search\"") != std::string::npos, "public web_search tool name preserved");
    CHECK_TRUE(all_sse.find("\"type\":\"web_search_tool_result\"") != std::string::npos, "web_search_tool_result emitted");
    CHECK_TRUE(concat_thinking.find("Let me search the Blackhole P150A specs.") != std::string::npos &&
               concat_thinking.find("The documentation confirms 1.5 MiB L1 per Tensix core across 110 cores.") != std::string::npos,
               "both pre-search and post-search thinking blocks streamed");
    CHECK_TRUE(concat_text == "Each active Tensix core on Tenstorrent Blackhole P150A has 1.5 MiB of L1 SRAM.",
               "post-search answer text streamed exact");
    CHECK_TRUE(all_sse.find("\"stop_reason\":\"end_turn\"") != std::string::npos, "stop_reason is end_turn");
    CHECK_TRUE(all_sse.find("\"web_search_requests\":1") != std::string::npos, "usage.server_tool_use reported");

    // Case 4: Verify Turn 2 multi-turn echoed server-tool history produces 100% identical RadixBlockCache token prefix!
    const std::string turn2_json = R"JSON({
        "model": "DeepSeek-V4.1-Flash-1017",
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 8192},
        "tools": [
            {"type": "web_search_20250305", "name": "web_search"}
        ],
        "messages": [
            {"role": "user", "content": "How much L1 SRAM per Tensix core does Blackhole P150A have?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me search the Blackhole P150A specs."},
                    {"type": "server_tool_use", "id": "call_dsv41_2001_0", "name": "web_search", "input": {"query": "Blackhole P150A L1 SRAM"}},
                    {"type": "web_search_tool_result", "tool_use_id": "call_dsv41_2001_0", "content": [
                        {"type": "web_search_result", "title": "Tenstorrent Blackhole P150A Architecture",
                         "url": "https://docs.tenstorrent.com/aibs/blackhole/index.html",
                         "encrypted_content": "Blackhole P150A provides 110 active Tensix cores with 1.5 MiB L1 SRAM per core and 32 GB GDDR6."}
                    ]},
                    {"type": "thinking", "thinking": "The documentation confirms 1.5 MiB L1 per Tensix core across 110 cores."},
                    {"type": "text", "text": "Each active Tensix core on Tenstorrent Blackhole P150A has 1.5 MiB of L1 SRAM."}
                ]
            },
            {"role": "user", "content": "What is that across 110 cores?"}
        ]
    })JSON";

    const AnthropicTranslatedRequest t2 = translate_anthropic_to_dsv41(parse_json(turn2_json), "dsv41");
    std::string tm2 = t2.thinking_mode;
    int tb2 = 55;
    const std::string turn2_prompt = render_dsv41_chat_prompt(
        t2.openai_messages, &t2.openai_tools, nullptr, &t2.reasoning_effort, tm2, tb2);
    const std::vector<int32_t> turn2_ids = tok.encode(turn2_prompt, true);

    std::vector<int32_t> expected_turn1_full = seen_prompts[1];
    expected_turn1_full.insert(expected_turn1_full.end(), step1_ids.begin(), step1_ids.end());
    CHECK_TRUE(turn2_ids.size() > expected_turn1_full.size(), "Turn 2 extends full Turn 1 sequence");
    for (size_t i = 0; i < expected_turn1_full.size(); ++i) {
        CHECK_TRUE(turn2_ids[i] == expected_turn1_full[i],
                   "Turn 2 echoed server-tool history matches Turn 1 token-for-token (100% RadixBlockCache hit)");
    }

    std::printf("[PASS] Case 3 & 4: Mid-turn POSIX --search-url loop + 100%% Tier-0 LCP & Turn-2 RadixBlockCache prefix parity (%zu tokens)\n",
                expected_turn1_full.size());
    return true;
}

std::string write_minimal_fallback_tokenizer() {
    std::string direct(256, 0);
    for (int b = 33; b <= 126; ++b) direct[size_t(b)] = 1;
    for (int b = 161; b <= 172; ++b) direct[size_t(b)] = 1;
    for (int b = 174; b <= 255; ++b) direct[size_t(b)] = 1;

    std::ostringstream js;
    js << "{\"model\":{\"vocab\":{";
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        const uint32_t cp = direct[size_t(b)] ? uint32_t(b) : uint32_t(256 + n++);
        std::string u8;
        if (cp <= 0x7Fu) {
            u8.push_back(char(cp));
        } else {
            u8.push_back(char(0xC0u | (cp >> 6)));
            u8.push_back(char(0x80u | (cp & 0x3Fu)));
        }
        if (b > 0) js << ",";
        js << JsonValue(u8).dumps_compact() << ":" << (3 + b);
    }
    js << "},\"merges\":[]},"
       << "\"pre_tokenizer\":{\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":\"\\\\S+|\\\\s+\"}}]},"
       << "\"added_tokens\":["
       << "{\"id\":0,\"content\":\"<\\uFF5Cbegin\\u2581of\\u2581sentence\\uFF5C>\",\"special\":true},"
       << "{\"id\":1,\"content\":\"<\\uFF5Cend\\u2581of\\u2581sentence\\uFF5C>\",\"special\":true},"
       << "{\"id\":2,\"content\":\"<\\uFF5C\\u2581pad\\u2581\\uFF5C>\",\"special\":true},"
       << "{\"id\":128799,\"content\":\"<\\uFF5CSystem\\uFF5C>\",\"special\":true},"
       << "{\"id\":128803,\"content\":\"<\\uFF5CUser\\uFF5C>\",\"special\":true},"
       << "{\"id\":128804,\"content\":\"<\\uFF5CAssistant\\uFF5C>\",\"special\":true},"
       << "{\"id\":128821,\"content\":\"<think>\",\"special\":true},"
       << "{\"id\":128822,\"content\":\"</think>\",\"special\":true},"
       << "{\"id\":128825,\"content\":\"\\uFF5CDSML\\uFF5C\",\"special\":true},"
       << "{\"id\":129264,\"content\":\"<\\uFF5Cdeepseek_image\\uFF5C>\",\"special\":true}"
       << "]}";

    const std::string tmp_path = "/tmp/dsv41_minimal_tok_" + std::to_string(::getpid()) + ".json";
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    out << js.str();
    out.close();
    return tmp_path;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                     ? std::string(std::getenv("DSV41_ROOT"))
                                     : std::string(".");
    std::string tok_path = root_env + "/hf_meta/tokenizer.json";
    if (argc > 1) tok_path = argv[1];

    bool using_tmp = false;
    std::error_code ec;
    if (!std::filesystem::exists(tok_path, ec)) {
        tok_path = write_minimal_fallback_tokenizer();
        using_tmp = true;
    }

    Dsv41Tokenizer tok;
    tok.load(tok_path);
    if (using_tmp) {
        std::filesystem::remove(tok_path, ec);
    }
    std::printf("[TEST_ANTHROPIC_FE] Loaded tokenizer (%zu vocab%s)\n",
                tok.vocab_size(), using_tmp ? ", synthetic ByteLevel fallback" : "");

    if (!test_translation_and_client_tool_sse(tok)) return 1;
    if (!test_server_search_loop_and_radix_prefix(tok)) return 1;

    std::printf("[TEST_ANTHROPIC_FE] ALL TRACK 8E TESTS PASSED.\n");
    return 0;
}
