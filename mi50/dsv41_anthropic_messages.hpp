#pragma once
// Wave 8E: Native Pure-C++ Anthropic Messages API (/anthropic/v1/messages & /v1/messages)
// + Pluggable Mid-Turn Server-Side deepseek_web_search / deepseek_read_url Loop (0 Python Proxies)
// Preserves deterministic DeepSeek-V4.1-Flash-1017 prompt prefixes for 100% RadixBlockCache hits.

#include "dsv41_json.hpp"
#include "dsv41_runtime.hpp"
#include "dsv41_tokenizer_dsml.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dsv41 {

struct AnthropicTranslatedRequest {
    std::string model = "DeepSeek-V4.1-Flash-1017-WHT128-HLUT2";
    int max_tokens = 16384;
    float temperature = 0.6f;
    float top_p = 0.95f;
    bool has_explicit_temperature = false;
    bool stream = false;
    uint64_t seed = 0;
    std::string thinking_mode = "thinking";
    JsonValue reasoning_effort = JsonValue::make_int(55);
    JsonValue openai_messages = JsonValue::make_array();
    JsonValue openai_tools = JsonValue::make_array();
    std::vector<std::string> stop_sequences;
    // Maps internal DSML function name (e.g. "deepseek_web_search") -> public Anthropic tool name (e.g. "web_search" or "deepseek_web_search")
    std::unordered_map<std::string, std::string> server_tool_public_names;
};

struct ServerSearchResult {
    bool ok = false;
    std::string error_code;
    std::string tool_result_text;      // Injected into <tool_result>...</tool_result>
    JsonValue anthropic_result_items; // Array of {"type":"web_search_result", ...} for web_search_tool_result
};

namespace detail_anthropic {

inline std::string url_encode(const std::string& s) {
    static const char kHex[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(s.size() * 3);
    for (unsigned char c : s) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~') {
            out.push_back(char(c));
        } else if (c == ' ') {
            out.push_back('+');
        } else {
            out.push_back('%');
            out.push_back(kHex[(c >> 4) & 0xF]);
            out.push_back(kHex[c & 0xF]);
        }
    }
    return out;
}

inline bool parse_http_url(const std::string& url, std::string& host, int& port, std::string& base_path) {
    std::string u = url;
    if (u.rfind("http://", 0) == 0) {
        u.erase(0, 7);
    } else if (u.rfind("https://", 0) == 0) {
        return false;  // Local POSIX HTTP search sidecar (e.g. http://127.0.0.1:port)
    }
    size_t slash = u.find('/');
    std::string authority = (slash == std::string::npos) ? u : u.substr(0, slash);
    base_path = (slash == std::string::npos) ? "" : u.substr(slash);
    while (!base_path.empty() && base_path.back() == '/') base_path.pop_back();

    size_t colon = authority.rfind(':');
    if (colon != std::string::npos) {
        host = authority.substr(0, colon);
        port = std::atoi(authority.substr(colon + 1).c_str());
    } else {
        host = authority;
        port = 80;
    }
    return !host.empty() && port > 0 && port <= 65535;
}

inline std::string decode_chunked_body(const std::string& raw) {
    std::string out;
    size_t pos = 0;
    while (pos < raw.size()) {
        size_t crlf = raw.find("\r\n", pos);
        if (crlf == std::string::npos) break;
        std::string hex_len = raw.substr(pos, crlf - pos);
        size_t semi = hex_len.find(';');
        if (semi != std::string::npos) hex_len.resize(semi);
        char* endp = nullptr;
        unsigned long chunk_sz = std::strtoul(hex_len.c_str(), &endp, 16);
        pos = crlf + 2;
        if (chunk_sz == 0) break;
        if (pos + chunk_sz > raw.size()) {
            out.append(raw.data() + pos, raw.size() - pos);
            break;
        }
        out.append(raw.data() + pos, chunk_sz);
        pos += chunk_sz;
        if (pos + 2 <= raw.size() && raw[pos] == '\r' && raw[pos + 1] == '\n') pos += 2;
    }
    return out;
}

inline bool posix_http_get(const std::string& host, int port, const std::string& path_and_query,
                           int& out_status, std::string& out_body, int timeout_sec = 10)
{
    out_status = 0;
    out_body.clear();

    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = nullptr;
    const std::string port_str = std::to_string(port);
    if (::getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0 || !res) {
        return false;
    }

    int fd = -1;
    for (struct addrinfo* p = res; p != nullptr; p = p->ai_next) {
        fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        struct timeval tv{};
        tv.tv_sec = timeout_sec;
        tv.tv_usec = 0;
        (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(res);
    if (fd < 0) return false;

    std::string req = "GET " + (path_and_query.empty() ? std::string("/") : path_and_query) +
                      " HTTP/1.1\r\nHost: " + host + ":" + port_str +
                      "\r\nAccept: application/json, text/plain;q=0.9, */*;q=0.8\r\n"
                      "User-Agent: dsv41-fe/wave8\r\nConnection: close\r\n\r\n";

    size_t sent = 0;
    while (sent < req.size()) {
        ssize_t w = ::send(fd, req.data() + sent, req.size() - sent, MSG_NOSIGNAL);
        if (w <= 0) {
            ::close(fd);
            return false;
        }
        sent += size_t(w);
    }

    std::string resp;
    char buf[8192];
    for (;;) {
        ssize_t r = ::recv(fd, buf, sizeof(buf), 0);
        if (r <= 0) break;
        resp.append(buf, size_t(r));
        if (resp.size() > (4ULL << 20)) break;  // Cap observation payload at 4 MiB
    }
    ::close(fd);

    size_t hdr_end = resp.find("\r\n\r\n");
    size_t sep_len = 4;
    if (hdr_end == std::string::npos) {
        hdr_end = resp.find("\n\n");
        sep_len = 2;
    }
    if (hdr_end == std::string::npos) return false;

    const std::string hdr = resp.substr(0, hdr_end);
    size_t sp1 = hdr.find(' ');
    if (sp1 != std::string::npos) {
        out_status = std::atoi(hdr.c_str() + sp1 + 1);
    }
    std::string raw_body = resp.substr(hdr_end + sep_len);
    std::string hdr_low = hdr;
    for (char& c : hdr_low) c = char(std::tolower(static_cast<unsigned char>(c)));
    if (hdr_low.find("transfer-encoding: chunked") != std::string::npos) {
        out_body = decode_chunked_body(raw_body);
    } else {
        out_body = std::move(raw_body);
    }
    return out_status >= 200 && out_status < 300;
}

inline std::string extract_anthropic_block_text(const JsonValue& content) {
    if (content.is_string()) return content.s_val;
    if (content.is_array()) {
        std::string out;
        for (const auto& b : content.arr_val) {
            if (b.is_string()) {
                if (!out.empty()) out += "\n";
                out += b.s_val;
            } else if (b.is_object()) {
                const std::string& bt = b.get("type").as_str();
                if (bt == "text") {
                    if (!out.empty()) out += "\n";
                    out += b.get("text").as_str();
                } else if (bt == "web_search_result") {
                    if (!out.empty()) out += "\n\n";
                    const std::string title = b.get("title").as_str();
                    const std::string url = b.get("url").as_str();
                    std::string snippet = b.get("content").as_str();
                    if (snippet.empty()) snippet = b.get("snippet").as_str();
                    if (snippet.empty()) snippet = b.get("encrypted_content").as_str();
                    out += "[" + title + "](" + url + ")\n" + snippet;
                }
            }
        }
        return out;
    }
    if (content.is_object()) {
        return content.dumps_compact();
    }
    return "";
}

inline JsonValue make_builtin_search_schema(const std::string& fn_name) {
    JsonValue tool = JsonValue::make_object();
    tool.set("type", JsonValue::make_string("function"));
    JsonValue fn = JsonValue::make_object();
    fn.set("name", JsonValue::make_string(fn_name));
    fn.set("description", JsonValue::make_string(
        "Search the web for up-to-date facts, documentation, or references and return top ranked snippets and URLs."));
    JsonValue params = JsonValue::make_object();
    params.set("type", JsonValue::make_string("object"));
    JsonValue props = JsonValue::make_object();
    JsonValue qprop = JsonValue::make_object();
    qprop.set("type", JsonValue::make_string("string"));
    qprop.set("description", JsonValue::make_string("Search query string."));
    props.set("query", std::move(qprop));
    params.set("properties", std::move(props));
    JsonValue req = JsonValue::make_array();
    req.push(JsonValue::make_string("query"));
    params.set("required", std::move(req));
    fn.set("parameters", std::move(params));
    tool.set("function", std::move(fn));
    return tool;
}

inline JsonValue make_builtin_read_url_schema(const std::string& fn_name) {
    JsonValue tool = JsonValue::make_object();
    tool.set("type", JsonValue::make_string("function"));
    JsonValue fn = JsonValue::make_object();
    fn.set("name", JsonValue::make_string(fn_name));
    fn.set("description", JsonValue::make_string(
        "Fetch and read the text content of a specific URL."));
    JsonValue params = JsonValue::make_object();
    params.set("type", JsonValue::make_string("object"));
    JsonValue props = JsonValue::make_object();
    JsonValue uprop = JsonValue::make_object();
    uprop.set("type", JsonValue::make_string("string"));
    uprop.set("description", JsonValue::make_string("Target HTTP/HTTPS URL to read."));
    props.set("url", std::move(uprop));
    params.set("properties", std::move(props));
    JsonValue req = JsonValue::make_array();
    req.push(JsonValue::make_string("url"));
    params.set("required", std::move(req));
    fn.set("parameters", std::move(params));
    tool.set("function", std::move(fn));
    return tool;
}

inline int map_anthropic_budget_to_effort(int64_t budget_tokens) {
    if (budget_tokens <= 0) return 0;
    if (budget_tokens <= 1536) return 35;
    if (budget_tokens <= 4096) return 50;
    if (budget_tokens <= 8192) return 55;
    if (budget_tokens <= 16384) return 75;
    return 100;
}

}  // namespace detail_anthropic

inline AnthropicTranslatedRequest translate_anthropic_to_dsv41(
    const JsonValue& req,
    const std::string& default_model_id)
{
    AnthropicTranslatedRequest out;
    out.model = req.get("model").as_str(default_model_id);
    if (out.model.empty()) out.model = default_model_id;
    out.max_tokens = std::max(1, int(req.get("max_tokens").as_int(16384)));
    out.stream = req.get("stream").as_bool(false);
    if (req.has("temperature") && !req.get("temperature").is_null()) {
        out.temperature = float(req.get("temperature").as_number(0.6));
        out.has_explicit_temperature = true;
    }
    out.top_p = float(req.get("top_p").as_number(0.95));
    if (req.has("seed") && !req.get("seed").is_null()) {
        out.seed = static_cast<uint64_t>(req.get("seed").as_int(0));
    }
    if (req.get("stop_sequences").is_array()) {
        for (const auto& s : req.get("stop_sequences").arr_val) {
            if (s.is_string() && !s.s_val.empty()) out.stop_sequences.push_back(s.s_val);
        }
    }

    // Thinking mode resolution:
    //   - Anthropic `thinking: {"type": "enabled", "budget_tokens": N}` -> thinking mode with mapped effort
    //   - Anthropic `thinking: {"type": "disabled"}` -> chat mode (`effort = 0`)
    //   - Omitted `thinking`: default to DeepSeek-V4.1 thinking mode (effort=55) unless tool_choice forces specific non-thinking
    if (req.get("thinking").is_object()) {
        const JsonValue& th = req.get("thinking");
        const std::string ttype = th.get("type").as_str("enabled");
        if (ttype == "disabled" || ttype == "off" || ttype == "none") {
            out.thinking_mode = "chat";
            out.reasoning_effort = JsonValue::make_int(0);
        } else {
            out.thinking_mode = "thinking";
            const int64_t btok = th.get("budget_tokens").as_int(8192);
            out.reasoning_effort = JsonValue::make_int(detail_anthropic::map_anthropic_budget_to_effort(btok));
        }
    } else if (req.has("reasoning_effort")) {
        out.reasoning_effort = req.get("reasoning_effort");
    }

    // 1. System prompt
    if (req.has("system") && !req.get("system").is_null()) {
        const std::string sys_txt = detail_anthropic::extract_anthropic_block_text(req.get("system"));
        if (!sys_txt.empty()) {
            JsonValue sys_msg = JsonValue::make_object();
            sys_msg.set("role", JsonValue::make_string("system"));
            sys_msg.set("content", JsonValue::make_string(sys_txt));
            out.openai_messages.push(std::move(sys_msg));
        }
    }

    // 2. Tools array (custom input_schema tools + built-in deepseek_web_search / deepseek_read_url / web_search_20250305)
    std::unordered_set<std::string> seen_tools;
    if (req.get("tools").is_array()) {
        for (const auto& t : req.get("tools").arr_val) {
            if (!t.is_object()) continue;
            const std::string ttype = t.get("type").as_str();
            const std::string tname = t.get("name").as_str();

            if (ttype == "deepseek_web_search" || tname == "deepseek_web_search" ||
                ttype.rfind("web_search", 0) == 0 || tname == "web_search")
            {
                const std::string dsml_name = (tname == "web_search" && ttype != "deepseek_web_search")
                    ? "deepseek_web_search" : (tname.empty() ? "deepseek_web_search" : tname);
                const std::string pub_name = tname.empty() ? "deepseek_web_search" : tname;
                out.server_tool_public_names[dsml_name] = pub_name;
                out.server_tool_public_names["deepseek_web_search"] = pub_name;
                if (seen_tools.insert(dsml_name).second) {
                    if (t.has("input_schema") && t.get("input_schema").is_object()) {
                        JsonValue wrap = JsonValue::make_object();
                        wrap.set("type", JsonValue::make_string("function"));
                        JsonValue fn = JsonValue::make_object();
                        fn.set("name", JsonValue::make_string(dsml_name));
                        fn.set("description", JsonValue::make_string(t.get("description").as_str("Search the web.")));
                        fn.set("parameters", t.get("input_schema"));
                        wrap.set("function", std::move(fn));
                        out.openai_tools.push(std::move(wrap));
                    } else {
                        out.openai_tools.push(detail_anthropic::make_builtin_search_schema(dsml_name));
                    }
                }
                continue;
            }

            if (ttype == "deepseek_read_url" || tname == "deepseek_read_url" ||
                ttype.rfind("web_fetch", 0) == 0 || tname == "read_url")
            {
                const std::string dsml_name = tname.empty() ? "deepseek_read_url" : tname;
                out.server_tool_public_names[dsml_name] = dsml_name;
                out.server_tool_public_names["deepseek_read_url"] = dsml_name;
                if (seen_tools.insert(dsml_name).second) {
                    if (t.has("input_schema") && t.get("input_schema").is_object()) {
                        JsonValue wrap = JsonValue::make_object();
                        wrap.set("type", JsonValue::make_string("function"));
                        JsonValue fn = JsonValue::make_object();
                        fn.set("name", JsonValue::make_string(dsml_name));
                        fn.set("description", JsonValue::make_string(t.get("description").as_str("Read URL content.")));
                        fn.set("parameters", t.get("input_schema"));
                        wrap.set("function", std::move(fn));
                        out.openai_tools.push(std::move(wrap));
                    } else {
                        out.openai_tools.push(detail_anthropic::make_builtin_read_url_schema(dsml_name));
                    }
                }
                continue;
            }

            if (t.has("function") && t.get("function").is_object()) {
                const std::string fn_n = t.get("function").get("name").as_str();
                if (!fn_n.empty() && seen_tools.insert(fn_n).second) {
                    out.openai_tools.push(t);
                }
                continue;
            }

            if (!tname.empty() && seen_tools.insert(tname).second) {
                JsonValue wrap = JsonValue::make_object();
                wrap.set("type", JsonValue::make_string("function"));
                JsonValue fn = JsonValue::make_object();
                fn.set("name", JsonValue::make_string(tname));
                if (t.has("description")) fn.set("description", t.get("description"));
                if (t.has("input_schema") && t.get("input_schema").is_object()) {
                    fn.set("parameters", t.get("input_schema"));
                } else {
                    JsonValue empty_schema = JsonValue::make_object();
                    empty_schema.set("type", JsonValue::make_string("object"));
                    empty_schema.set("properties", JsonValue::make_object());
                    fn.set("parameters", std::move(empty_schema));
                }
                wrap.set("function", std::move(fn));
                out.openai_tools.push(std::move(wrap));
            }
        }
    }

    // 3. Messages array -> OpenAI/DSV41 messages (preserving multi-turn thinking, tool_use, tool_result, and server_tool_use!)
    if (req.get("messages").is_array()) {
        for (const auto& m : req.get("messages").arr_val) {
            if (!m.is_object()) continue;
            const std::string role = m.get("role").as_str();
            const JsonValue& content = m.get("content");

            if (role == "user") {
                if (content.is_string()) {
                    JsonValue um = JsonValue::make_object();
                    um.set("role", JsonValue::make_string("user"));
                    um.set("content", content);
                    out.openai_messages.push(std::move(um));
                } else if (content.is_array()) {
                    JsonValue cbs = JsonValue::make_array();
                    for (const auto& b : content.arr_val) {
                        if (!b.is_object()) continue;
                        const std::string bt = b.get("type").as_str();
                        if (bt == "tool_result" || bt == "web_search_tool_result") {
                            JsonValue tr = JsonValue::make_object();
                            tr.set("type", JsonValue::make_string("tool_result"));
                            tr.set("tool_use_id", JsonValue::make_string(b.get("tool_use_id").as_str()));
                            std::string txt = detail_anthropic::extract_anthropic_block_text(b.get("content"));
                            if (b.get("is_error").as_bool(false) && !txt.empty()) {
                                txt = "Error: " + txt;
                            }
                            tr.set("content", JsonValue::make_string(std::move(txt)));
                            cbs.push(std::move(tr));
                        } else if (bt == "text") {
                            JsonValue tb = JsonValue::make_object();
                            tb.set("type", JsonValue::make_string("text"));
                            tb.set("text", JsonValue::make_string(b.get("text").as_str()));
                            cbs.push(std::move(tb));
                        }
                    }
                    JsonValue um = JsonValue::make_object();
                    um.set("role", JsonValue::make_string("user"));
                    um.set("content_blocks", std::move(cbs));
                    out.openai_messages.push(std::move(um));
                }
            } else if (role == "assistant") {
                if (content.is_string()) {
                    JsonValue am = JsonValue::make_object();
                    am.set("role", JsonValue::make_string("assistant"));
                    am.set("content", content);
                    am.set("reasoning_content", JsonValue::make_string(""));
                    out.openai_messages.push(std::move(am));
                } else if (content.is_array()) {
                    std::string cur_thinking;
                    std::string cur_text;
                    JsonValue cur_tcs = JsonValue::make_array();
                    bool has_pending_assistant = false;

                    auto flush_assistant = [&]() {
                        if (!has_pending_assistant && cur_thinking.empty() && cur_text.empty() && cur_tcs.arr_val.empty()) {
                            return;
                        }
                        JsonValue am = JsonValue::make_object();
                        am.set("role", JsonValue::make_string("assistant"));
                        am.set("content", JsonValue::make_string(cur_text));
                        am.set("reasoning_content", JsonValue::make_string(cur_thinking));
                        if (!cur_tcs.arr_val.empty()) {
                            am.set("tool_calls", std::move(cur_tcs));
                            cur_tcs = JsonValue::make_array();
                        }
                        out.openai_messages.push(std::move(am));
                        cur_thinking.clear();
                        cur_text.clear();
                        has_pending_assistant = false;
                    };

                    for (const auto& b : content.arr_val) {
                        if (!b.is_object()) continue;
                        const std::string bt = b.get("type").as_str();
                        if (bt == "thinking") {
                            cur_thinking += b.get("thinking").as_str();
                            has_pending_assistant = true;
                        } else if (bt == "text") {
                            cur_text += b.get("text").as_str();
                            has_pending_assistant = true;
                        } else if (bt == "tool_use" || bt == "server_tool_use") {
                            std::string tname = b.get("name").as_str();
                            if (bt == "server_tool_use" && tname == "web_search" &&
                                seen_tools.count("deepseek_web_search") > 0 && seen_tools.count("web_search") == 0) {
                                tname = "deepseek_web_search";
                            }
                            JsonValue tc = JsonValue::make_object();
                            tc.set("id", JsonValue::make_string(b.get("id").as_str()));
                            tc.set("type", JsonValue::make_string("function"));
                            JsonValue fn = JsonValue::make_object();
                            fn.set("name", JsonValue::make_string(tname));
                            fn.set("arguments", b.has("input") ? b.get("input") : JsonValue::make_object());
                            tc.set("function", std::move(fn));
                            cur_tcs.push(std::move(tc));
                            has_pending_assistant = true;
                        } else if (bt == "web_search_tool_result" || bt == "tool_result") {
                            // Inline server-tool result inside an assistant turn: split into assistant(tool_calls) + tool(result)
                            // so Turn N+1 renders the exact same DSML + <tool_result> token prefix as Turn N!
                            flush_assistant();
                            JsonValue tm = JsonValue::make_object();
                            tm.set("role", JsonValue::make_string("tool"));
                            tm.set("tool_call_id", JsonValue::make_string(b.get("tool_use_id").as_str()));
                            tm.set("content", JsonValue::make_string(
                                detail_anthropic::extract_anthropic_block_text(b.get("content"))));
                            out.openai_messages.push(std::move(tm));
                        }
                    }
                    flush_assistant();
                }
            } else if (role == "system") {
                JsonValue sm = JsonValue::make_object();
                sm.set("role", JsonValue::make_string("system"));
                sm.set("content", JsonValue::make_string(detail_anthropic::extract_anthropic_block_text(content)));
                out.openai_messages.push(std::move(sm));
            }
        }
    }

    return out;
}

inline bool is_builtin_server_search_name(const std::string& name) {
    return name == "deepseek_web_search" || name == "web_search";
}

inline bool is_builtin_server_read_name(const std::string& name) {
    return name == "deepseek_read_url" || name == "read_url" || name == "web_fetch";
}

inline ServerSearchResult execute_server_search_or_read(
    const std::string& search_base_url,
    const std::string& tool_name,
    const std::string& args_json_str)
{
    ServerSearchResult res;
    res.anthropic_result_items = JsonValue::make_array();
    if (search_base_url.empty()) {
        res.error_code = "unavailable";
        return res;
    }

    std::string host;
    int port = 80;
    std::string base_path;
    if (!detail_anthropic::parse_http_url(search_base_url, host, port, base_path)) {
        res.error_code = "invalid_search_url";
        res.tool_result_text = "Error: invalid --search-url configuration.";
        return res;
    }

    JsonValue args;
    if (!JsonParser::try_parse(args_json_str, args) || !args.is_object()) {
        args = JsonValue::make_object();
    }

    std::string req_path;
    if (is_builtin_server_search_name(tool_name)) {
        const std::string query = args.get("query").as_str();
        const std::string endpoint = base_path.empty() ? "/search" : base_path;
        req_path = endpoint + "?q=" + detail_anthropic::url_encode(query) + "&format=json";
    } else if (is_builtin_server_read_name(tool_name)) {
        const std::string target_url = args.get("url").as_str();
        // If target_url points directly to the configured host:port, fetch its path directly; otherwise proxy via /fetch?url=
        std::string thost;
        int tport = 0;
        std::string tpath;
        if (detail_anthropic::parse_http_url(target_url, thost, tport, tpath) && thost == host && tport == port) {
            req_path = tpath.empty() ? "/" : tpath;
        } else {
            const std::string endpoint = base_path.empty() ? "/fetch" : (base_path + "/fetch");
            req_path = endpoint + "?url=" + detail_anthropic::url_encode(target_url) + "&format=json";
        }
    } else {
        res.error_code = "unsupported_tool";
        return res;
    }

    int status = 0;
    std::string body;
    if (!detail_anthropic::posix_http_get(host, port, req_path, status, body)) {
        res.error_code = "unavailable";
        res.tool_result_text = "Search backend unavailable (HTTP status " + std::to_string(status) + ").";
        return res;
    }

    res.ok = true;
    JsonValue parsed;
    if (JsonParser::try_parse(body, parsed) && parsed.is_object()) {
        if (parsed.get("results").is_array()) {
            const auto& arr = parsed.get("results").arr_val;
            const size_t lim = std::min<size_t>(arr.size(), 8);
            for (size_t i = 0; i < lim; ++i) {
                const JsonValue& item = arr[i];
                if (!item.is_object()) continue;
                const std::string title = item.get("title").as_str("Result " + std::to_string(i + 1));
                const std::string url = item.get("url").as_str();
                std::string snippet = item.get("content").as_str();
                if (snippet.empty()) snippet = item.get("snippet").as_str();
                if (snippet.empty()) snippet = item.get("text").as_str();

                JsonValue wr = JsonValue::make_object();
                wr.set("type", JsonValue::make_string("web_search_result"));
                wr.set("title", JsonValue::make_string(title));
                wr.set("url", JsonValue::make_string(url));
                wr.set("encrypted_content", JsonValue::make_string(snippet));
                if (item.has("page_age") && item.get("page_age").is_string()) {
                    wr.set("page_age", item.get("page_age"));
                }
                res.anthropic_result_items.push(std::move(wr));

                if (!res.tool_result_text.empty()) res.tool_result_text += "\n\n";
                res.tool_result_text += "[" + title + "](" + url + ")\n" + snippet;
            }
            if (res.tool_result_text.empty()) {
                res.tool_result_text = "No results found.";
            }
            return res;
        }
        if (parsed.has("content") && parsed.get("content").is_string()) {
            const std::string title = parsed.get("title").as_str(args.get("url").as_str("Document"));
            const std::string url = parsed.get("url").as_str(args.get("url").as_str());
            const std::string text = parsed.get("content").as_str();
            JsonValue wr = JsonValue::make_object();
            wr.set("type", JsonValue::make_string("web_search_result"));
            wr.set("title", JsonValue::make_string(title));
            wr.set("url", JsonValue::make_string(url));
            wr.set("encrypted_content", JsonValue::make_string(text));
            res.anthropic_result_items.push(std::move(wr));
            res.tool_result_text = "[" + title + "](" + url + ")\n" + text;
            return res;
        }
    }

    // Plain-text fallback
    if (body.size() > 16384) body.resize(16384);
    res.tool_result_text = body;
    JsonValue wr = JsonValue::make_object();
    wr.set("type", JsonValue::make_string("web_search_result"));
    wr.set("title", JsonValue::make_string(is_builtin_server_read_name(tool_name) ? "Fetched URL" : "Search Result"));
    wr.set("url", JsonValue::make_string(args.get("url").as_str(search_base_url)));
    wr.set("encrypted_content", JsonValue::make_string(body));
    res.anthropic_result_items.push(std::move(wr));
    return res;
}

inline std::string format_anthropic_sse_event(const char* event_name, const JsonValue& data_obj) {
    std::string out = "event: ";
    out += event_name;
    out += "\ndata: ";
    out += data_obj.dumps_compact();
    out += "\n\n";
    return out;
}

class AnthropicSseStateMachine {
public:
    using SendFn = std::function<bool(const std::string&)>;

    AnthropicSseStateMachine(SendFn send_fn, std::string msg_id, std::string model)
        : send_(std::move(send_fn)), msg_id_(std::move(msg_id)), model_(std::move(model)) {}

    bool emit_message_start(int input_tokens, int cache_read_tokens) {
        if (started_) return true;
        started_ = true;
        JsonValue ev = JsonValue::make_object();
        ev.set("type", JsonValue::make_string("message_start"));
        JsonValue msg = JsonValue::make_object();
        msg.set("id", JsonValue::make_string(msg_id_));
        msg.set("type", JsonValue::make_string("message"));
        msg.set("role", JsonValue::make_string("assistant"));
        msg.set("model", JsonValue::make_string(model_));
        msg.set("content", JsonValue::make_array());
        msg.set("stop_reason", JsonValue::make_null());
        msg.set("stop_sequence", JsonValue::make_null());
        JsonValue usage = JsonValue::make_object();
        usage.set("input_tokens", JsonValue::make_int(input_tokens));
        usage.set("cache_creation_input_tokens", JsonValue::make_int(0));
        usage.set("cache_read_input_tokens", JsonValue::make_int(cache_read_tokens));
        usage.set("output_tokens", JsonValue::make_int(0));
        msg.set("usage", std::move(usage));
        ev.set("message", std::move(msg));
        return send_(format_anthropic_sse_event("message_start", ev));
    }

    bool emit_thinking_delta(const std::string& delta) {
        if (delta.empty()) return true;
        if (open_kind_ != BlockKind::Thinking) {
            if (!close_open_block()) return false;
            cur_idx_ = next_idx_++;
            open_kind_ = BlockKind::Thinking;
            JsonValue ev = JsonValue::make_object();
            ev.set("type", JsonValue::make_string("content_block_start"));
            ev.set("index", JsonValue::make_int(cur_idx_));
            JsonValue cb = JsonValue::make_object();
            cb.set("type", JsonValue::make_string("thinking"));
            cb.set("thinking", JsonValue::make_string(""));
            ev.set("content_block", std::move(cb));
            if (!send_(format_anthropic_sse_event("content_block_start", ev))) return false;
        }
        JsonValue ev = JsonValue::make_object();
        ev.set("type", JsonValue::make_string("content_block_delta"));
        ev.set("index", JsonValue::make_int(cur_idx_));
        JsonValue d = JsonValue::make_object();
        d.set("type", JsonValue::make_string("thinking_delta"));
        d.set("thinking", JsonValue::make_string(delta));
        ev.set("delta", std::move(d));
        return send_(format_anthropic_sse_event("content_block_delta", ev));
    }

    bool emit_text_delta(const std::string& delta) {
        if (delta.empty()) return true;
        if (open_kind_ != BlockKind::Text) {
            if (!close_open_block()) return false;
            cur_idx_ = next_idx_++;
            open_kind_ = BlockKind::Text;
            JsonValue ev = JsonValue::make_object();
            ev.set("type", JsonValue::make_string("content_block_start"));
            ev.set("index", JsonValue::make_int(cur_idx_));
            JsonValue cb = JsonValue::make_object();
            cb.set("type", JsonValue::make_string("text"));
            cb.set("text", JsonValue::make_string(""));
            ev.set("content_block", std::move(cb));
            if (!send_(format_anthropic_sse_event("content_block_start", ev))) return false;
        }
        JsonValue ev = JsonValue::make_object();
        ev.set("type", JsonValue::make_string("content_block_delta"));
        ev.set("index", JsonValue::make_int(cur_idx_));
        JsonValue d = JsonValue::make_object();
        d.set("type", JsonValue::make_string("text_delta"));
        d.set("text", JsonValue::make_string(delta));
        ev.set("delta", std::move(d));
        return send_(format_anthropic_sse_event("content_block_delta", ev));
    }

    bool emit_tool_start(const std::string& id, const std::string& public_name, bool is_server_tool) {
        if (!close_open_block()) return false;
        cur_idx_ = next_idx_++;
        open_kind_ = BlockKind::Tool;
        JsonValue ev = JsonValue::make_object();
        ev.set("type", JsonValue::make_string("content_block_start"));
        ev.set("index", JsonValue::make_int(cur_idx_));
        JsonValue cb = JsonValue::make_object();
        cb.set("type", JsonValue::make_string(is_server_tool ? "server_tool_use" : "tool_use"));
        cb.set("id", JsonValue::make_string(id));
        cb.set("name", JsonValue::make_string(public_name));
        cb.set("input", JsonValue::make_object());
        ev.set("content_block", std::move(cb));
        return send_(format_anthropic_sse_event("content_block_start", ev));
    }

    bool emit_tool_input_json_and_close(const std::string& partial_json) {
        if (open_kind_ != BlockKind::Tool) return true;
        if (!partial_json.empty()) {
            JsonValue ev = JsonValue::make_object();
            ev.set("type", JsonValue::make_string("content_block_delta"));
            ev.set("index", JsonValue::make_int(cur_idx_));
            JsonValue d = JsonValue::make_object();
            d.set("type", JsonValue::make_string("input_json_delta"));
            d.set("partial_json", JsonValue::make_string(partial_json));
            ev.set("delta", std::move(d));
            if (!send_(format_anthropic_sse_event("content_block_delta", ev))) return false;
        }
        return close_open_block();
    }

    bool emit_web_search_tool_result(const std::string& tool_use_id, const ServerSearchResult& sr) {
        if (!close_open_block()) return false;
        const int idx = next_idx_++;
        JsonValue ev = JsonValue::make_object();
        ev.set("type", JsonValue::make_string("content_block_start"));
        ev.set("index", JsonValue::make_int(idx));
        JsonValue cb = JsonValue::make_object();
        cb.set("type", JsonValue::make_string("web_search_tool_result"));
        cb.set("tool_use_id", JsonValue::make_string(tool_use_id));
        if (sr.ok) {
            cb.set("content", sr.anthropic_result_items);
        } else {
            JsonValue err = JsonValue::make_object();
            err.set("type", JsonValue::make_string("web_search_tool_result_error"));
            err.set("error_code", JsonValue::make_string(sr.error_code.empty() ? "unavailable" : sr.error_code));
            cb.set("content", std::move(err));
        }
        ev.set("content_block", std::move(cb));
        if (!send_(format_anthropic_sse_event("content_block_start", ev))) return false;

        JsonValue stop_ev = JsonValue::make_object();
        stop_ev.set("type", JsonValue::make_string("content_block_stop"));
        stop_ev.set("index", JsonValue::make_int(idx));
        return send_(format_anthropic_sse_event("content_block_stop", stop_ev));
    }

    bool close_open_block() {
        if (open_kind_ == BlockKind::None) return true;
        if (open_kind_ == BlockKind::Thinking) {
            JsonValue sig = JsonValue::make_object();
            sig.set("type", JsonValue::make_string("content_block_delta"));
            sig.set("index", JsonValue::make_int(cur_idx_));
            JsonValue d = JsonValue::make_object();
            d.set("type", JsonValue::make_string("signature_delta"));
            d.set("signature", JsonValue::make_string("dsv41-wht128-hlut2"));
            sig.set("delta", std::move(d));
            if (!send_(format_anthropic_sse_event("content_block_delta", sig))) return false;
        }
        JsonValue stop_ev = JsonValue::make_object();
        stop_ev.set("type", JsonValue::make_string("content_block_stop"));
        stop_ev.set("index", JsonValue::make_int(cur_idx_));
        open_kind_ = BlockKind::None;
        return send_(format_anthropic_sse_event("content_block_stop", stop_ev));
    }

    bool emit_finish(const std::string& anthropic_stop_reason, int output_tokens, int server_web_search_reqs) {
        if (!close_open_block()) return false;
        JsonValue md = JsonValue::make_object();
        md.set("type", JsonValue::make_string("message_delta"));
        JsonValue d = JsonValue::make_object();
        d.set("stop_reason", JsonValue::make_string(anthropic_stop_reason));
        d.set("stop_sequence", JsonValue::make_null());
        md.set("delta", std::move(d));
        JsonValue usage = JsonValue::make_object();
        usage.set("output_tokens", JsonValue::make_int(output_tokens));
        if (server_web_search_reqs > 0) {
            JsonValue stu = JsonValue::make_object();
            stu.set("web_search_requests", JsonValue::make_int(server_web_search_reqs));
            usage.set("server_tool_use", std::move(stu));
        }
        md.set("usage", std::move(usage));
        if (!send_(format_anthropic_sse_event("message_delta", md))) return false;

        JsonValue ms = JsonValue::make_object();
        ms.set("type", JsonValue::make_string("message_stop"));
        return send_(format_anthropic_sse_event("message_stop", ms));
    }

    bool is_started() const { return started_; }

private:
    enum class BlockKind { None, Thinking, Text, Tool };
    SendFn send_;
    std::string msg_id_;
    std::string model_;
    bool started_ = false;
    BlockKind open_kind_ = BlockKind::None;
    int cur_idx_ = -1;
    int next_idx_ = 0;
};

// Appends exact DeepSeek-V4.1 tool-continuation suffix onto `prompt_str` after a mid-turn server-side tool call,
// preserving 100% token-level prefix identity with Dsv41Runtime::active_seq_ids_ (0 re-prefill penalty!).
inline void append_dsv41_midturn_tool_observation(
    std::string& prompt_str,
    const std::string& raw_step_completion,
    const std::vector<std::string>& observations,
    const std::string& thinking_mode)
{
    static const char* kEosTok       = "<\xEF\xBD\x9C" "end\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>";
    static const char* kUserTok      = "<\xEF\xBD\x9C" "User\xEF\xBD\x9C>";
    static const char* kAssistantTok = "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>";
    const std::string eos(kEosTok);

    std::string clean_raw = raw_step_completion;
    if (clean_raw.size() >= eos.size() &&
        clean_raw.compare(clean_raw.size() - eos.size(), eos.size(), eos) == 0) {
        clean_raw.erase(clean_raw.size() - eos.size());
    }
    prompt_str += clean_raw;
    prompt_str += kEosTok;
    prompt_str += kUserTok;
    for (size_t i = 0; i < observations.size(); ++i) {
        if (i) prompt_str += "\n\n";
        prompt_str += "<tool_result>";
        prompt_str += observations[i];
        prompt_str += "</tool_result>";
    }
    prompt_str += kAssistantTok;
    prompt_str += (thinking_mode == "thinking") ? "<think>" : "</think>";
}

using AnthropicGenerateStepFn = std::function<std::vector<int32_t>(
    const std::vector<int32_t>& prompt_ids,
    int max_new_tokens,
    TokenStreamCallback on_token,
    Dsv41GenStats& out_stats)>;

inline bool run_anthropic_messages_turn(
    const AnthropicTranslatedRequest& treq,
    const Dsv41Tokenizer& tokenizer,
    const std::string& search_url,
    const std::string& msg_id,
    uint64_t call_seed_base,
    const AnthropicGenerateStepFn& generate_step,
    const std::function<bool()>& is_aborted,
    const AnthropicSseStateMachine::SendFn& sse_send,
    std::string& out_sync_json,
    Dsv41GenStats& out_agg_stats,
    int& out_server_search_reqs,
    size_t& out_total_tools)
{
    out_sync_json.clear();
    out_agg_stats = Dsv41GenStats{};
    out_server_search_reqs = 0;
    out_total_tools = 0;

    std::string thinking_mode = treq.thinking_mode;
    int thinking_budget = 55;
    const JsonValue* tools_ptr = treq.openai_tools.arr_val.empty() ? nullptr : &treq.openai_tools;
    std::string prompt_str = render_dsv41_chat_prompt(
        treq.openai_messages, tools_ptr, nullptr, &treq.reasoning_effort,
        thinking_mode, thinking_budget);

    auto pop_utf8 = [](std::string& pending_bytes) -> std::string {
        if (pending_bytes.empty()) return {};
        const size_t n = pending_bytes.size();
        size_t check = n;
        for (size_t back = 1; back <= 3 && back <= n; ++back) {
            const uint8_t b = static_cast<uint8_t>(pending_bytes[n - back]);
            if ((b & 0x80u) == 0u) break;
            if ((b & 0xC0u) == 0xC0u) {
                size_t expected = 1;
                if ((b & 0xE0u) == 0xC0u) expected = 2;
                else if ((b & 0xF0u) == 0xE0u) expected = 3;
                else if ((b & 0xF8u) == 0xF0u) expected = 4;
                if (back < expected) check = n - back;
                break;
            }
        }
        std::string res = pending_bytes.substr(0, check);
        pending_bytes.erase(0, check);
        return res;
    };

    auto resolve_tool = [&](const std::string& dsml_name) -> std::pair<std::string, bool> {
        const bool srv = !search_url.empty() &&
                         (is_builtin_server_search_name(dsml_name) || is_builtin_server_read_name(dsml_name));
        auto it = treq.server_tool_public_names.find(dsml_name);
        const std::string pub = (it != treq.server_tool_public_names.end()) ? it->second : dsml_name;
        return {pub, srv};
    };

    AnthropicSseStateMachine sse(sse_send, msg_id, treq.model);
    JsonValue sync_blocks = JsonValue::make_array();
    int remaining_tokens = std::max(1, treq.max_tokens);
    std::string final_stop_reason = "end_turn";
    std::string matched_stop_seq;
    bool has_matched_stop_seq = false;

    for (int sub = 0; sub < 8 && remaining_tokens > 0; ++sub) {
        const std::vector<int32_t> prompt_ids = tokenizer.encode(prompt_str, true);
        const int step_budget = remaining_tokens;
        const uint64_t step_seed = call_seed_base + uint64_t(sub) * 16ULL;

        Dsv41StreamParser sp(thinking_mode);
        std::string utf8_pending;
        std::string step_content_accum;
        std::vector<Dsv41ToolStreamDelta> tool_deltas;
        Dsv41GenStats step_stats{};
        bool client_abort = false;

        auto emit_td_list = [&](const std::vector<Dsv41ToolStreamDelta>& deltas) -> bool {
            for (const auto& td : deltas) {
                auto [pub_name, is_srv] = resolve_tool(td.name);
                if (td.is_start) {
                    if (!sse.emit_tool_start(td.id, pub_name, is_srv)) return false;
                } else {
                    if (!sse.emit_tool_input_json_and_close(td.arguments_delta)) return false;
                }
            }
            return true;
        };

        (void)generate_step(
            prompt_ids,
            step_budget,
            [&](int32_t tok_id) -> bool {
                if (is_aborted()) {
                    client_abort = true;
                    return false;
                }
                if (treq.stream && !sse.is_started()) {
                    const int p_toks = (step_stats.prompt_tokens > 0) ? step_stats.prompt_tokens : int(prompt_ids.size());
                    if (!sse.emit_message_start(p_toks, step_stats.cached_prefix_tokens)) {
                        client_abort = true;
                        return false;
                    }
                }
                if (tok_id == Dsv41SpecialTokens::kThinkEnd && !utf8_pending.empty()) {
                    utf8_pending.clear();
                }
                if (step_stats.think_guard_triggered &&
                    tok_id == Dsv41SpecialTokens::kThinkEnd &&
                    sp.in_thinking())
                {
                    std::string r_delta;
                    sp.force_close_thinking(r_delta);
                    if (treq.stream && !r_delta.empty() && !sse.emit_thinking_delta(r_delta)) {
                        client_abort = true;
                        return false;
                    }
                    return true;
                }

                utf8_pending += tokenizer.decode_single(tok_id, false);
                const std::string piece = pop_utf8(utf8_pending);
                if (!piece.empty()) {
                    std::string r_delta, c_delta;
                    sp.feed(piece, r_delta, c_delta, &tool_deltas, step_seed);
                    if (treq.stream && !r_delta.empty() && !sse.emit_thinking_delta(r_delta)) {
                        client_abort = true;
                        return false;
                    }
                    if (!c_delta.empty()) {
                        step_content_accum += c_delta;
                        for (const auto& sseq : treq.stop_sequences) {
                            const size_t pos = step_content_accum.find(sseq);
                            if (pos != std::string::npos) {
                                has_matched_stop_seq = true;
                                matched_stop_seq = sseq;
                                const size_t prev_len = step_content_accum.size() - c_delta.size();
                                if (pos > prev_len) {
                                    c_delta.resize(pos - prev_len);
                                } else {
                                    c_delta.clear();
                                }
                                break;
                            }
                        }
                        if (treq.stream && !c_delta.empty() && !sse.emit_text_delta(c_delta)) {
                            client_abort = true;
                            return false;
                        }
                        if (has_matched_stop_seq) return false;
                    }
                    if (treq.stream && !tool_deltas.empty() && !emit_td_list(tool_deltas)) {
                        client_abort = true;
                        return false;
                    }
                    if (sp.should_stop_generation()) {
                        return false;
                    }
                }
                return true;
            },
            step_stats);

        if (client_abort || is_aborted()) return false;

        if (treq.stream && !sse.is_started()) {
            const int p_toks = (step_stats.prompt_tokens > 0) ? step_stats.prompt_tokens : int(prompt_ids.size());
            if (!sse.emit_message_start(p_toks, step_stats.cached_prefix_tokens)) return false;
        }

        if (!utf8_pending.empty()) {
            std::string r_d, c_d;
            sp.feed(utf8_pending, r_d, c_d, &tool_deltas, step_seed);
            utf8_pending.clear();
            if (treq.stream) {
                if (!r_d.empty() && !sse.emit_thinking_delta(r_d)) return false;
                if (!c_d.empty() && !has_matched_stop_seq && !sse.emit_text_delta(c_d)) return false;
                if (!tool_deltas.empty() && !emit_td_list(tool_deltas)) return false;
            }
        }

        std::string r_fin, c_fin;
        Dsv41ParsedCompletion parsed_step;
        sp.finish(r_fin, c_fin, parsed_step, step_seed, &tool_deltas);
        if (treq.stream) {
            if (!r_fin.empty() && !sse.emit_thinking_delta(r_fin)) return false;
            if (!c_fin.empty() && !has_matched_stop_seq && !sse.emit_text_delta(c_fin)) return false;
            if (!tool_deltas.empty() && !emit_td_list(tool_deltas)) return false;
        }

        // Accumulate stats across mid-turn search sub-steps
        if (sub == 0) {
            out_agg_stats.prompt_tokens        = (step_stats.prompt_tokens > 0) ? step_stats.prompt_tokens : int(prompt_ids.size());
            out_agg_stats.cached_prefix_tokens = step_stats.cached_prefix_tokens;
            out_agg_stats.radix_hit_tokens     = step_stats.radix_hit_tokens;
            out_agg_stats.swa_replay_tokens    = step_stats.swa_replay_tokens;
            out_agg_stats.prefilled_tokens     = step_stats.prefilled_tokens;
            out_agg_stats.prefix_reuse_ms      = step_stats.prefix_reuse_ms;
            out_agg_stats.fresh_prefill_ms     = step_stats.fresh_prefill_ms;
            out_agg_stats.prefill_ms           = step_stats.prefill_ms;
        }
        out_agg_stats.completion_tokens     += step_stats.completion_tokens;
        out_agg_stats.decode_ms             += step_stats.decode_ms;
        out_agg_stats.dec_dspark_ms         += step_stats.dec_dspark_ms;
        out_agg_stats.dec_backbone_ms       += step_stats.dec_backbone_ms;
        out_agg_stats.dec_p150_dev_ms       += step_stats.dec_p150_dev_ms;
        out_agg_stats.dec_sample_ms         += step_stats.dec_sample_ms;
        out_agg_stats.t_attn_ms             += step_stats.t_attn_ms;
        out_agg_stats.t_hsi_index_ms        += step_stats.t_hsi_index_ms;
        out_agg_stats.t_mi50_moe_ms         += step_stats.t_mi50_moe_ms;
        out_agg_stats.t_p150_wait_ms        += step_stats.t_p150_wait_ms;
        out_agg_stats.t_allreduce_ms        += step_stats.t_allreduce_ms;
        out_agg_stats.t_engram_wait_ms      += step_stats.t_engram_wait_ms;
        out_agg_stats.overlap_saved_ms      += step_stats.overlap_saved_ms;
        out_agg_stats.dspark_steps          += step_stats.dspark_steps;
        out_agg_stats.dspark_drafted        += step_stats.dspark_drafted;
        out_agg_stats.dspark_accepted       += step_stats.dspark_accepted;
        out_agg_stats.dspark_tree_steps     += step_stats.dspark_tree_steps;
        out_agg_stats.dspark_tree_accepted  += step_stats.dspark_tree_accepted;
        out_agg_stats.dspark_tree16_steps    += step_stats.dspark_tree16_steps;
        out_agg_stats.dspark_tree16_accepted += step_stats.dspark_tree16_accepted;
        if (step_stats.think_guard_triggered) {
            out_agg_stats.think_guard_triggered = true;
            out_agg_stats.think_guard_period    = step_stats.think_guard_period;
            out_agg_stats.think_guard_rewound  += step_stats.think_guard_rewound;
        }

        remaining_tokens -= step_stats.completion_tokens;
        out_total_tools += parsed_step.tool_calls.size();

        if (!treq.stream) {
            if (!parsed_step.reasoning_content.empty()) {
                JsonValue tb = JsonValue::make_object();
                tb.set("type", JsonValue::make_string("thinking"));
                tb.set("thinking", JsonValue::make_string(parsed_step.reasoning_content));
                tb.set("signature", JsonValue::make_string("dsv41-wht128-hlut2"));
                sync_blocks.push(std::move(tb));
            }
            std::string txt = parsed_step.content;
            if (has_matched_stop_seq) {
                const size_t p = txt.find(matched_stop_seq);
                if (p != std::string::npos) txt.resize(p);
            }
            if (!txt.empty() || parsed_step.tool_calls.empty()) {
                if (!txt.empty() || sync_blocks.arr_val.empty()) {
                    JsonValue cb = JsonValue::make_object();
                    cb.set("type", JsonValue::make_string("text"));
                    cb.set("text", JsonValue::make_string(txt));
                    sync_blocks.push(std::move(cb));
                }
            }
            for (const auto& tc : parsed_step.tool_calls) {
                auto [pub_name, is_srv] = resolve_tool(tc.name);
                JsonValue inp;
                if (!JsonParser::try_parse(tc.arguments_json, inp) || !inp.is_object()) {
                    inp = JsonValue::make_object();
                }
                JsonValue ub = JsonValue::make_object();
                ub.set("type", JsonValue::make_string(is_srv ? "server_tool_use" : "tool_use"));
                ub.set("id", JsonValue::make_string(tc.id));
                ub.set("name", JsonValue::make_string(pub_name));
                ub.set("input", std::move(inp));
                sync_blocks.push(std::move(ub));
            }
        }

        if (has_matched_stop_seq) {
            final_stop_reason = "stop_sequence";
            break;
        }
        if (parsed_step.tool_calls.empty()) {
            final_stop_reason = (step_stats.completion_tokens >= step_budget) ? "max_tokens" : "end_turn";
            break;
        }

        // Check if all invoked tools in this step are server-side search/read tools with --search-url configured
        bool all_server_tools = !search_url.empty();
        for (const auto& tc : parsed_step.tool_calls) {
            if (!resolve_tool(tc.name).second) {
                all_server_tools = false;
                break;
            }
        }
        if (!all_server_tools || remaining_tokens <= 0) {
            final_stop_reason = (parsed_step.tool_calls.empty()) ? "max_tokens" : "tool_use";
            break;
        }

        // Execute server-side search/fetch in pure C++ POSIX sockets and continue mid-turn with 0 re-prefill!
        std::vector<std::string> observations;
        observations.reserve(parsed_step.tool_calls.size());
        for (const auto& tc : parsed_step.tool_calls) {
            const ServerSearchResult sr = execute_server_search_or_read(search_url, tc.name, tc.arguments_json);
            ++out_server_search_reqs;
            observations.push_back(sr.tool_result_text);
            if (treq.stream) {
                if (!sse.emit_web_search_tool_result(tc.id, sr)) return false;
            } else {
                JsonValue wr = JsonValue::make_object();
                wr.set("type", JsonValue::make_string("web_search_tool_result"));
                wr.set("tool_use_id", JsonValue::make_string(tc.id));
                if (sr.ok) {
                    wr.set("content", sr.anthropic_result_items);
                } else {
                    JsonValue err = JsonValue::make_object();
                    err.set("type", JsonValue::make_string("web_search_tool_result_error"));
                    err.set("error_code", JsonValue::make_string(sr.error_code.empty() ? "unavailable" : sr.error_code));
                    wr.set("content", std::move(err));
                }
                sync_blocks.push(std::move(wr));
            }
        }
        append_dsv41_midturn_tool_observation(prompt_str, sp.raw_accumulated(), observations, thinking_mode);
    }

    if (treq.stream) {
        return sse.emit_finish(final_stop_reason, out_agg_stats.completion_tokens, out_server_search_reqs);
    }

    JsonValue root = JsonValue::make_object();
    root.set("id", JsonValue::make_string(msg_id));
    root.set("type", JsonValue::make_string("message"));
    root.set("role", JsonValue::make_string("assistant"));
    root.set("model", JsonValue::make_string(treq.model));
    root.set("content", std::move(sync_blocks));
    root.set("stop_reason", JsonValue::make_string(final_stop_reason));
    root.set("stop_sequence", has_matched_stop_seq ? JsonValue::make_string(matched_stop_seq) : JsonValue::make_null());
    JsonValue usage = JsonValue::make_object();
    usage.set("input_tokens", JsonValue::make_int(out_agg_stats.prompt_tokens));
    usage.set("cache_creation_input_tokens", JsonValue::make_int(0));
    usage.set("cache_read_input_tokens", JsonValue::make_int(out_agg_stats.cached_prefix_tokens));
    usage.set("output_tokens", JsonValue::make_int(out_agg_stats.completion_tokens));
    if (out_server_search_reqs > 0) {
        JsonValue stu = JsonValue::make_object();
        stu.set("web_search_requests", JsonValue::make_int(out_server_search_reqs));
        usage.set("server_tool_use", std::move(stu));
    }
    root.set("usage", std::move(usage));
    out_sync_json = root.dumps_compact();
    return true;
}

}  // namespace dsv41
