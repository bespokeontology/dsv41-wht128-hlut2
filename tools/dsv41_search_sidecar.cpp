// Pure-C++17 SearXNG-Compatible Local HTTP JSON Search & URL Reader Sidecar (Zero Python)
// Serves GET /search?q=...&format=json and GET /fetch?url=...&format=json on 127.0.0.1:${DSV41_SEARCH_PORT:-8080}
// Dynamically loads system /usr/lib/x86_64-linux-gnu/libcurl.so.4 via dlopen for HTTPS upstream fetches.

#include "../mi50/dsv41_json.hpp"

#include <arpa/inet.h>
#include <dlfcn.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

volatile std::sig_atomic_t g_stop = 0;
void on_sig(int) { g_stop = 1; }

// Minimal libcurl C ABI constants & function pointers (via dlopen("libcurl.so.4"))
using CURL = void;
using CURLcode = int;
constexpr int kCurlOptUrl            = 10002;
constexpr int kCurlOptWriteFunction  = 20011;
constexpr int kCurlOptWriteData      = 10001;
constexpr int kCurlOptUserAgent      = 10018;
constexpr int kCurlOptFollowLocation = 52;
constexpr int kCurlOptMaxRedirs      = 68;
constexpr int kCurlOptTimeoutMs      = 155;
constexpr int kCurlOptConnectTimeoutMs = 156;
constexpr int kCurlOptAcceptEncoding = 10102;
constexpr int kCurlOptNoSignal       = 99;

struct LibCurlApi {
    void* handle = nullptr;
    CURLcode (*global_init)(long) = nullptr;
    CURL* (*easy_init)() = nullptr;
    CURLcode (*easy_setopt)(CURL*, int, ...) = nullptr;
    CURLcode (*easy_perform)(CURL*) = nullptr;
    void (*easy_cleanup)(CURL*) = nullptr;

    bool init() {
        const char* candidates[] = {
            "libcurl.so.4",
            "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
            "libcurl-gnutls.so.4",
        };
        for (const char* path : candidates) {
            handle = ::dlopen(path, RTLD_NOW | RTLD_LOCAL);
            if (handle) break;
        }
        if (!handle) return false;
        global_init  = reinterpret_cast<decltype(global_init)>(::dlsym(handle, "curl_global_init"));
        easy_init    = reinterpret_cast<decltype(easy_init)>(::dlsym(handle, "curl_easy_init"));
        easy_setopt  = reinterpret_cast<decltype(easy_setopt)>(::dlsym(handle, "curl_easy_setopt"));
        easy_perform = reinterpret_cast<decltype(easy_perform)>(::dlsym(handle, "curl_easy_perform"));
        easy_cleanup = reinterpret_cast<decltype(easy_cleanup)>(::dlsym(handle, "curl_easy_cleanup"));
        if (!easy_init || !easy_setopt || !easy_perform || !easy_cleanup) return false;
        if (global_init) (void)global_init(3L);
        return true;
    }
};

size_t curl_write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
    const size_t n = size * nmemb;
    auto* out = static_cast<std::string*>(userdata);
    if (out->size() + n <= (2ULL << 20)) {
        out->append(ptr, n);
    }
    return n;
}

bool https_get(const LibCurlApi& curl, const std::string& url, std::string& out_body, long timeout_ms = 6000) {
    out_body.clear();
    if (!curl.easy_init) return false;
    CURL* eh = curl.easy_init();
    if (!eh) return false;
    (void)curl.easy_setopt(eh, kCurlOptUrl, url.c_str());
    (void)curl.easy_setopt(eh, kCurlOptWriteFunction, &curl_write_cb);
    (void)curl.easy_setopt(eh, kCurlOptWriteData, &out_body);
    (void)curl.easy_setopt(eh, kCurlOptUserAgent,
                           "dsv41-search/1.0 (https://github.com/bespokeontology/dsv41-wht128-hlut2)");
    (void)curl.easy_setopt(eh, kCurlOptFollowLocation, 1L);
    (void)curl.easy_setopt(eh, kCurlOptMaxRedirs, 5L);
    (void)curl.easy_setopt(eh, kCurlOptConnectTimeoutMs, 2500L);
    (void)curl.easy_setopt(eh, kCurlOptTimeoutMs, timeout_ms);
    (void)curl.easy_setopt(eh, kCurlOptAcceptEncoding, "");
    (void)curl.easy_setopt(eh, kCurlOptNoSignal, 1L);
    const CURLcode rc = curl.easy_perform(eh);
    curl.easy_cleanup(eh);
    return rc == 0 && !out_body.empty();
}

std::string url_decode(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '+') {
            out.push_back(' ');
        } else if (s[i] == '%' && i + 2 < s.size()) {
            auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            const int hi = hex(s[i + 1]);
            const int lo = hex(s[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(char((hi << 4) | lo));
                i += 2;
                continue;
            }
            out.push_back(s[i]);
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

std::string url_encode(const std::string& s) {
    static const char kHex[] = "0123456789ABCDEF";
    std::string out;
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

std::string get_query_param(const std::string& query_str, const std::string& key) {
    size_t pos = 0;
    while (pos < query_str.size()) {
        size_t amp = query_str.find('&', pos);
        if (amp == std::string::npos) amp = query_str.size();
        const std::string kv = query_str.substr(pos, amp - pos);
        const size_t eq = kv.find('=');
        if (eq != std::string::npos && kv.substr(0, eq) == key) {
            return url_decode(kv.substr(eq + 1));
        }
        pos = amp + 1;
    }
    return {};
}

std::string strip_html_and_entities(const std::string& html) {
    std::string raw;
    raw.reserve(html.size());
    bool in_tag = false;
    for (size_t i = 0; i < html.size(); ++i) {
        if (!in_tag && html[i] == '<') {
            auto starts_ci = [&](const char* tag) {
                const size_t tl = std::strlen(tag);
                if (i + tl > html.size()) return false;
                for (size_t k = 0; k < tl; ++k) {
                    if (std::tolower(static_cast<unsigned char>(html[i + k])) != tag[k]) return false;
                }
                return true;
            };
            if (starts_ci("<script")) {
                size_t end_s = html.find("</script", i + 7);
                if (end_s == std::string::npos) end_s = html.find("</SCRIPT", i + 7);
                if (end_s == std::string::npos) break;
                size_t gt = html.find('>', end_s);
                i = (gt == std::string::npos) ? html.size() - 1 : gt;
                continue;
            }
            if (starts_ci("<style")) {
                size_t end_s = html.find("</style", i + 6);
                if (end_s == std::string::npos) end_s = html.find("</STYLE", i + 6);
                if (end_s == std::string::npos) break;
                size_t gt = html.find('>', end_s);
                i = (gt == std::string::npos) ? html.size() - 1 : gt;
                continue;
            }
            in_tag = true;
            raw.push_back(' ');
            continue;
        }
        if (in_tag) {
            if (html[i] == '>') in_tag = false;
            continue;
        }
        if (html[i] == '&') {
            if (html.compare(i, 5, "&amp;") == 0) { raw.push_back('&'); i += 4; continue; }
            if (html.compare(i, 4, "&lt;") == 0) { raw.push_back('<'); i += 3; continue; }
            if (html.compare(i, 4, "&gt;") == 0) { raw.push_back('>'); i += 3; continue; }
            if (html.compare(i, 6, "&quot;") == 0) { raw.push_back('"'); i += 5; continue; }
            if (html.compare(i, 6, "&nbsp;") == 0) { raw.push_back(' '); i += 5; continue; }
            if (html.compare(i, 5, "&#39;") == 0 || html.compare(i, 6, "&#x27;") == 0) {
                raw.push_back('\'');
                i += (html[i + 2] == 'x' ? 5 : 4);
                continue;
            }
        }
        raw.push_back(html[i]);
    }
    std::string collapsed;
    collapsed.reserve(raw.size());
    bool prev_sp = true;
    for (char c : raw) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!prev_sp) {
                collapsed.push_back(' ');
                prev_sp = true;
            }
        } else {
            collapsed.push_back(c);
            prev_sp = false;
        }
    }
    if (!collapsed.empty() && collapsed.back() == ' ') collapsed.pop_back();
    return collapsed;
}

std::string unwrap_ddg_redirect(const std::string& href) {
    const size_t uddg = href.find("uddg=");
    if (uddg != std::string::npos) {
        size_t end = href.find('&', uddg + 5);
        if (end == std::string::npos) end = href.size();
        return url_decode(href.substr(uddg + 5, end - (uddg + 5)));
    }
    if (href.rfind("//", 0) == 0) return "https:" + href;
    return href;
}

void parse_ddg_html_results(const std::string& html, dsv41::JsonValue& results_arr) {
    size_t pos = 0;
    while (results_arr.arr_val.size() < 8) {
        const size_t a_cls = html.find("result__a", pos);
        if (a_cls == std::string::npos) break;
        const size_t a_open = html.rfind("<a", a_cls);
        const size_t a_gt   = html.find('>', a_cls);
        const size_t a_end  = html.find("</a>", a_gt == std::string::npos ? a_cls : a_gt);
        if (a_open == std::string::npos || a_gt == std::string::npos || a_end == std::string::npos) break;

        std::string href;
        const std::string tag_hdr = html.substr(a_open, a_gt - a_open);
        const size_t hp = tag_hdr.find("href=\"");
        if (hp != std::string::npos) {
            const size_t he = tag_hdr.find('"', hp + 6);
            if (he != std::string::npos) href = unwrap_ddg_redirect(tag_hdr.substr(hp + 6, he - (hp + 6)));
        }
        const std::string title = strip_html_and_entities(html.substr(a_gt + 1, a_end - (a_gt + 1)));

        std::string snippet;
        const size_t sn_cls = html.find("result__snippet", a_end);
        const size_t next_a = html.find("result__a", a_end + 4);
        if (sn_cls != std::string::npos && (next_a == std::string::npos || sn_cls < next_a)) {
            const size_t sn_gt = html.find('>', sn_cls);
            size_t sn_end = html.find("</a>", sn_gt == std::string::npos ? sn_cls : sn_gt);
            const size_t sn_div = html.find("</div>", sn_gt == std::string::npos ? sn_cls : sn_gt);
            if (sn_div != std::string::npos && (sn_end == std::string::npos || sn_div < sn_end)) sn_end = sn_div;
            if (sn_gt != std::string::npos && sn_end != std::string::npos) {
                snippet = strip_html_and_entities(html.substr(sn_gt + 1, sn_end - (sn_gt + 1)));
            }
        }
        if (!title.empty() && !href.empty()) {
            dsv41::JsonValue item = dsv41::JsonValue::make_object();
            item.set("title", dsv41::JsonValue::make_string(title));
            item.set("url", dsv41::JsonValue::make_string(href));
            item.set("content", dsv41::JsonValue::make_string(snippet));
            results_arr.push(std::move(item));
        }
        pos = a_end + 4;
    }
}

void append_hn_algolia_results(const LibCurlApi& curl, const std::string& query, dsv41::JsonValue& results_arr) {
    std::string body;
    const std::string hn_url =
        "https://hn.algolia.com/api/v1/search?hitsPerPage=5&query=" + url_encode(query);
    if (!https_get(curl, hn_url, body, 4000)) return;
    dsv41::JsonValue j;
    if (!dsv41::JsonParser::try_parse(body, j) || !j.is_object()) return;
    const dsv41::JsonValue& hits = j.get("hits");
    if (!hits.is_array()) return;
    for (const auto& h : hits.arr_val) {
        if (results_arr.arr_val.size() >= 8) break;
        std::string title = h.get("title").as_str();
        if (title.empty()) title = h.get("story_title").as_str();
        std::string url = h.get("url").as_str();
        if (url.empty()) {
            const std::string oid = h.get("objectID").as_str();
            if (!oid.empty()) url = "https://news.ycombinator.com/item?id=" + oid;
        }
        std::string snip = strip_html_and_entities(h.get("story_text").as_str());
        if (snip.empty()) snip = strip_html_and_entities(h.get("comment_text").as_str());
        if (snip.empty()) {
            snip = title + " (points: " + std::to_string(h.get("points").as_int(0)) +
                   ", author: " + h.get("author").as_str() + ")";
        }
        if (!title.empty() && !url.empty()) {
            dsv41::JsonValue item = dsv41::JsonValue::make_object();
            item.set("title", dsv41::JsonValue::make_string(title));
            item.set("url", dsv41::JsonValue::make_string(url));
            item.set("content", dsv41::JsonValue::make_string(snip));
            results_arr.push(std::move(item));
        }
    }
}

void append_github_repo_results(const LibCurlApi& curl, const std::string& query, dsv41::JsonValue& results_arr) {
    std::string body;
    const std::string gh_url =
        "https://api.github.com/search/repositories?per_page=3&q=" + url_encode(query);
    if (!https_get(curl, gh_url, body, 4000)) return;
    dsv41::JsonValue j;
    if (!dsv41::JsonParser::try_parse(body, j) || !j.is_object()) return;
    const dsv41::JsonValue& items = j.get("items");
    if (!items.is_array()) return;
    for (const auto& it : items.arr_val) {
        if (results_arr.arr_val.size() >= 8) break;
        const std::string full_name = it.get("full_name").as_str();
        const std::string html_url  = it.get("html_url").as_str();
        const std::string desc      = it.get("description").as_str();
        if (full_name.empty() || html_url.empty()) continue;
        dsv41::JsonValue item = dsv41::JsonValue::make_object();
        item.set("title", dsv41::JsonValue::make_string(full_name + " (GitHub)"));
        item.set("url", dsv41::JsonValue::make_string(html_url));
        item.set("content", dsv41::JsonValue::make_string(desc));
        results_arr.push(std::move(item));
    }
}

void append_wikipedia_fallback(const LibCurlApi& curl, const std::string& query, dsv41::JsonValue& results_arr) {
    std::string body;
    const std::string wiki_url =
        "https://en.wikipedia.org/w/api.php?action=query&list=search&utf8=1&format=json&srlimit=5&srsearch=" +
        url_encode(query);
    if (!https_get(curl, wiki_url, body, 4500)) return;
    dsv41::JsonValue j;
    if (!dsv41::JsonParser::try_parse(body, j) || !j.is_object()) return;
    const dsv41::JsonValue& sarr = j.get("query").get("search");
    if (!sarr.is_array()) return;
    for (const auto& el : sarr.arr_val) {
        if (results_arr.arr_val.size() >= 8) break;
        const std::string title = el.get("title").as_str();
        const std::string snip  = strip_html_and_entities(el.get("snippet").as_str());
        if (title.empty()) continue;
        std::string slug = title;
        for (char& c : slug) {
            if (c == ' ') c = '_';
        }
        dsv41::JsonValue item = dsv41::JsonValue::make_object();
        item.set("title", dsv41::JsonValue::make_string(title + " - Wikipedia"));
        item.set("url", dsv41::JsonValue::make_string("https://en.wikipedia.org/wiki/" + url_encode(slug)));
        item.set("content", dsv41::JsonValue::make_string(snip));
        results_arr.push(std::move(item));
    }
}

void send_http_json(int fd, int code, const std::string& body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << code << (code == 200 ? " OK" : " Error") << "\r\n"
        << "Content-Type: application/json; charset=utf-8\r\n"
        << "Connection: close\r\n"
        << "Content-Length: " << body.size() << "\r\n\r\n"
        << body;
    const std::string s = oss.str();
    size_t sent = 0;
    while (sent < s.size()) {
        ssize_t n = ::send(fd, s.data() + sent, s.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) break;
        sent += size_t(n);
    }
}

void handle_client(int fd, const LibCurlApi* curl) {
    char buf[8192];
    ssize_t r = ::recv(fd, buf, sizeof(buf) - 1, 0);
    if (r <= 0) {
        ::close(fd);
        return;
    }
    buf[r] = '\0';
    std::string req_line(buf, size_t(r));
    const size_t eol = req_line.find("\r\n");
    if (eol != std::string::npos) req_line.resize(eol);

    std::string method, target;
    {
        std::istringstream iss(req_line);
        iss >> method >> target;
    }
    std::string path = target;
    std::string query_str;
    const size_t qpos = target.find('?');
    if (qpos != std::string::npos) {
        path = target.substr(0, qpos);
        query_str = target.substr(qpos + 1);
    }

    if (path == "/health") {
        send_http_json(fd, 200, "{\"status\":\"ok\",\"service\":\"dsv41_search_sidecar\"}");
        ::close(fd);
        return;
    }

    if (path == "/search") {
        const std::string q = get_query_param(query_str, "query").empty()
                                  ? get_query_param(query_str, "q")
                                  : get_query_param(query_str, "query");
        dsv41::JsonValue results = dsv41::JsonValue::make_array();
        if (!q.empty() && curl->handle) {
            std::string html;
            if (https_get(*curl, "https://html.duckduckgo.com/html/?q=" + url_encode(q), html, 3500)) {
                parse_ddg_html_results(html, results);
            }
            if (results.arr_val.size() < 4) {
                append_wikipedia_fallback(*curl, q, results);
            }
            if (results.arr_val.size() < 6) {
                append_hn_algolia_results(*curl, q, results);
            }
            if (results.arr_val.size() < 8) {
                append_github_repo_results(*curl, q, results);
            }
        }
        dsv41::JsonValue root = dsv41::JsonValue::make_object();
        root.set("query", dsv41::JsonValue::make_string(q));
        root.set("results", std::move(results));
        send_http_json(fd, 200, root.dumps_compact());
        ::close(fd);
        return;
    }

    if (path == "/fetch") {
        const std::string url = get_query_param(query_str, "url");
        std::string raw;
        bool ok = false;
        if (!url.empty() && curl->handle) {
            ok = https_get(*curl, url, raw, 6000);
        }
        dsv41::JsonValue root = dsv41::JsonValue::make_object();
        root.set("url", dsv41::JsonValue::make_string(url));
        if (ok) {
            std::string text = strip_html_and_entities(raw);
            if (text.size() > 16384) text.resize(16384);
            root.set("title", dsv41::JsonValue::make_string(url));
            root.set("content", dsv41::JsonValue::make_string(text));
            send_http_json(fd, 200, root.dumps_compact());
        } else {
            root.set("content", dsv41::JsonValue::make_string("Unable to fetch target URL: " + url));
            send_http_json(fd, 200, root.dumps_compact());
        }
        ::close(fd);
        return;
    }

    send_http_json(fd, 404, "{\"error\":\"not found\"}");
    ::close(fd);
}

}  // namespace

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = (std::getenv("DSV41_SEARCH_PORT") != nullptr)
                   ? std::atoi(std::getenv("DSV41_SEARCH_PORT"))
                   : 8080;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--link-check") {
            std::cout << "[DSV41_SEARCH] Link check OK\n";
            return 0;
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        }
    }

    std::signal(SIGINT, on_sig);
    std::signal(SIGTERM, on_sig);
    std::signal(SIGPIPE, SIG_IGN);

    LibCurlApi curl;
    const bool curl_ok = curl.init();

    const int srv_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv_fd < 0) {
        std::perror("socket");
        return 1;
    }
    int one = 1;
    (void)::setsockopt(srv_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        ::close(srv_fd);
        return 1;
    }
    if (::bind(srv_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::perror("bind");
        ::close(srv_fd);
        return 1;
    }
    if (::listen(srv_fd, 64) != 0) {
        std::perror("listen");
        ::close(srv_fd);
        return 1;
    }

    std::printf("[DSV41_SEARCH] Pure-C++ SearXNG-compatible JSON endpoint listening on http://%s:%d (libcurl=%d)\n",
                host.c_str(), port, int(curl_ok));
    std::fflush(stdout);

    while (!g_stop) {
        sockaddr_in cli{};
        socklen_t clen = sizeof(cli);
        const int cfd = ::accept(srv_fd, reinterpret_cast<sockaddr*>(&cli), &clen);
        if (cfd < 0) continue;
        std::thread(handle_client, cfd, &curl).detach();
    }
    ::close(srv_fd);
    return 0;
}
