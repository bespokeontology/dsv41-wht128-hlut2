// Pure-C++17 Tokenizer, ICU Engram Vocab Compressor (99,092 exact), V4.1 Chat Renderer & DSML Parser
// Zero Python anywhere.

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <unicode/locid.h>
#include <unicode/normalizer2.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>

#include "dsv41_tokenizer_dsml.hpp"
#include "dsv41_vit.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace dsv41 {

namespace {

// Exact GPT-2 ByteLevel byte <-> unicode mapping
struct ByteLevelTables {
    std::array<std::string, 256> byte_to_utf8;
    std::unordered_map<uint32_t, uint8_t> cp_to_byte;

    ByteLevelTables() {
        std::array<bool, 256> direct{};
        for (int b = 33; b <= 126; ++b) direct[b] = true;
        for (int b = 161; b <= 172; ++b) direct[b] = true;
        for (int b = 174; b <= 255; ++b) direct[b] = true;

        int n = 0;
        for (int b = 0; b < 256; ++b) {
            uint32_t cp = direct[b] ? uint32_t(b) : uint32_t(256 + n++);
            std::string u8;
            if (cp <= 0x7Fu) {
                u8.push_back(char(cp));
            } else {
                u8.push_back(char(0xC0u | (cp >> 6)));
                u8.push_back(char(0x80u | (cp & 0x3Fu)));
            }
            byte_to_utf8[b] = u8;
            cp_to_byte[cp] = uint8_t(b);
        }
    }
};

const ByteLevelTables& byte_tables() {
    static const ByteLevelTables t;
    return t;
}

// Decode one UTF-8 codepoint starting at s[i]; advances i. Returns 0xFFFD on malformed sequence
// advancing by maximal subpart per WHATWG / Rust String::from_utf8_lossy!
uint32_t next_utf8_lossy(const std::string& s, size_t& i, bool* had_error = nullptr) {
    const size_t n = s.size();
    const uint8_t b0 = uint8_t(s[i]);
    if (b0 < 0x80u) {
        ++i;
        return b0;
    }
    if (b0 >= 0xC2u && b0 <= 0xDFu) {
        if (i + 1 < n && (uint8_t(s[i + 1]) & 0xC0u) == 0x80u) {
            uint32_t cp = ((uint32_t(b0 & 0x1Fu) << 6) | uint32_t(uint8_t(s[i + 1]) & 0x3Fu));
            i += 2;
            return cp;
        }
        ++i;
        if (had_error) *had_error = true;
        return 0xFFFDu;
    }
    if (b0 >= 0xE0u && b0 <= 0xEFu) {
        const uint8_t min_b1 = (b0 == 0xE0u) ? 0xA0u : 0x80u;
        const uint8_t max_b1 = (b0 == 0xEDu) ? 0x9Fu : 0xBFu;
        if (i + 1 < n) {
            const uint8_t b1 = uint8_t(s[i + 1]);
            if (b1 >= min_b1 && b1 <= max_b1) {
                if (i + 2 < n && (uint8_t(s[i + 2]) & 0xC0u) == 0x80u) {
                    uint32_t cp = (uint32_t(b0 & 0x0Fu) << 12) |
                                  (uint32_t(b1 & 0x3Fu) << 6) |
                                  uint32_t(uint8_t(s[i + 2]) & 0x3Fu);
                    i += 3;
                    return cp;
                }
                i += 2;
                if (had_error) *had_error = true;
                return 0xFFFDu;
            }
        }
        ++i;
        if (had_error) *had_error = true;
        return 0xFFFDu;
    }
    if (b0 >= 0xF0u && b0 <= 0xF4u) {
        const uint8_t min_b1 = (b0 == 0xF0u) ? 0x90u : 0x80u;
        const uint8_t max_b1 = (b0 == 0xF4u) ? 0x8Fu : 0xBFu;
        if (i + 1 < n) {
            const uint8_t b1 = uint8_t(s[i + 1]);
            if (b1 >= min_b1 && b1 <= max_b1) {
                if (i + 2 < n && (uint8_t(s[i + 2]) & 0xC0u) == 0x80u) {
                    if (i + 3 < n && (uint8_t(s[i + 3]) & 0xC0u) == 0x80u) {
                        uint32_t cp = (uint32_t(b0 & 0x07u) << 18) |
                                      (uint32_t(b1 & 0x3Fu) << 12) |
                                      (uint32_t(uint8_t(s[i + 2]) & 0x3Fu) << 6) |
                                      uint32_t(uint8_t(s[i + 3]) & 0x3Fu);
                        i += 4;
                        return cp;
                    }
                    i += 3;
                    if (had_error) *had_error = true;
                    return 0xFFFDu;
                }
                i += 2;
                if (had_error) *had_error = true;
                return 0xFFFDu;
            }
        }
        ++i;
        if (had_error) *had_error = true;
        return 0xFFFDu;
    }
    ++i;
    if (had_error) *had_error = true;
    return 0xFFFDu;
}

void append_cp_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7Fu) {
        out.push_back(char(cp));
    } else if (cp <= 0x7FFu) {
        out.push_back(char(0xC0u | (cp >> 6)));
        out.push_back(char(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out.push_back(char(0xE0u | (cp >> 12)));
        out.push_back(char(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(char(0x80u | (cp & 0x3Fu)));
    } else {
        out.push_back(char(0xF0u | (cp >> 18)));
        out.push_back(char(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(char(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(char(0x80u | (cp & 0x3Fu)));
    }
}

std::string bytes_to_utf8_lossy(const std::string& raw_bytes, bool* out_had_fffd = nullptr) {
    std::string out;
    out.reserve(raw_bytes.size());
    bool had_err = false;
    size_t i = 0;
    while (i < raw_bytes.size()) {
        const uint32_t cp = next_utf8_lossy(raw_bytes, i, &had_err);
        if (cp == 0xFFFDu) had_err = true;
        append_cp_utf8(out, cp);
    }
    if (out_had_fffd) *out_had_fffd = had_err;
    return out;
}

struct AddedTokenEntry {
    int32_t id = -1;
    std::string content;
    bool special = false;
};

}  // namespace

struct Dsv41Tokenizer::Impl {
    std::vector<pcre2_code*> split_regexes;
    std::vector<AddedTokenEntry> added_sorted;  // sorted by content.size() desc
    std::vector<uint8_t> is_added_id;
    std::unordered_map<std::string, uint32_t> merge_ranks;  // "left\x1fright" -> rank

    ~Impl() {
        for (pcre2_code* re : split_regexes) {
            if (re) pcre2_code_free(re);
        }
    }

    static std::string make_pair_key(const std::string& a, const std::string& b) {
        std::string k;
        k.reserve(a.size() + b.size() + 1);
        k.append(a);
        k.push_back('\x1f');
        k.append(b);
        return k;
    }

    void apply_split_isolated(pcre2_code* re, const std::vector<std::string>& in_pieces, std::vector<std::string>& out_pieces) const {
        out_pieces.clear();
        pcre2_match_data* md = pcre2_match_data_create_from_pattern(re, nullptr);
        for (const std::string& piece : in_pieces) {
            if (piece.empty()) continue;
            size_t offset = 0;
            const size_t len = piece.size();
            while (offset < len) {
                const int rc = pcre2_match(
                    re,
                    reinterpret_cast<PCRE2_SPTR>(piece.data()),
                    len,
                    offset,
                    0,
                    md,
                    nullptr);
                if (rc < 0) {
                    out_pieces.push_back(piece.substr(offset));
                    break;
                }
                PCRE2_SIZE* ovec = pcre2_get_ovector_pointer(md);
                const size_t m0 = ovec[0];
                const size_t m1 = ovec[1];
                if (m0 > offset) {
                    out_pieces.push_back(piece.substr(offset, m0 - offset));
                }
                if (m1 > m0) {
                    out_pieces.push_back(piece.substr(m0, m1 - m0));
                    offset = m1;
                } else {
                    // Guard against zero-length match
                    size_t next = offset + 1;
                    while (next < len && (uint8_t(piece[next]) & 0xC0u) == 0x80u) ++next;
                    out_pieces.push_back(piece.substr(offset, next - offset));
                    offset = next;
                }
            }
        }
        pcre2_match_data_free(md);
    }

    void bpe_encode_piece(
        const std::string& raw_piece,
        const std::unordered_map<std::string, int32_t>& vocab,
        std::vector<int32_t>& out_ids) const
    {
        if (raw_piece.empty()) return;
        const auto& bt = byte_tables();
        std::vector<std::string> word;
        word.reserve(raw_piece.size());
        for (unsigned char b : raw_piece) {
            word.push_back(bt.byte_to_utf8[b]);
        }

        if (word.size() == 1) {
            auto it = vocab.find(word[0]);
            if (it != vocab.end()) out_ids.push_back(it->second);
            return;
        }

        while (word.size() > 1) {
            uint32_t best_rank = std::numeric_limits<uint32_t>::max();
            size_t best_idx = 0;
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                auto it = merge_ranks.find(make_pair_key(word[i], word[i + 1]));
                if (it != merge_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = i;
                }
            }
            if (best_rank == std::numeric_limits<uint32_t>::max()) break;

            const std::string first = word[best_idx];
            const std::string second = word[best_idx + 1];
            const std::string merged = first + second;

            std::vector<std::string> next_word;
            next_word.reserve(word.size() - 1);
            size_t i = 0;
            while (i < word.size()) {
                if (i + 1 < word.size() && word[i] == first && word[i + 1] == second) {
                    next_word.push_back(merged);
                    i += 2;
                } else {
                    next_word.push_back(std::move(word[i]));
                    ++i;
                }
            }
            word = std::move(next_word);
        }

        for (const std::string& sym : word) {
            auto it = vocab.find(sym);
            if (it != vocab.end()) {
                out_ids.push_back(it->second);
            }
        }
    }
};

Dsv41Tokenizer::Dsv41Tokenizer() : impl_(std::make_unique<Impl>()) {}
Dsv41Tokenizer::~Dsv41Tokenizer() = default;

void Dsv41Tokenizer::load(const std::string& tokenizer_json_path) {
    std::ifstream in(tokenizer_json_path, std::ios::binary);
    if (!in) throw std::runtime_error("Dsv41Tokenizer: cannot open " + tokenizer_json_path);
    std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    JsonValue root = JsonParser::parse(raw);

    // 1. Parse model.vocab
    const JsonValue& model = root.get("model");
    const JsonValue& vocab = model.get("vocab");
    if (!vocab.is_object()) throw std::runtime_error("Dsv41Tokenizer: missing model.vocab");

    int32_t max_id = -1;
    for (const auto& kv : vocab.obj_val) {
        const int32_t id = int32_t(kv.second.as_int(-1));
        if (id > max_id) max_id = id;
    }
    const JsonValue& added_tokens = root.get("added_tokens");
    if (added_tokens.is_array()) {
        for (const auto& at : added_tokens.arr_val) {
            const int32_t id = int32_t(at.get("id").as_int(-1));
            if (id > max_id) max_id = id;
        }
    }
    if (max_id < 0) throw std::runtime_error("Dsv41Tokenizer: empty vocab");

    id_to_token_.assign(size_t(max_id + 1), "");
    id_is_special_.assign(size_t(max_id + 1), 0u);
    impl_->is_added_id.assign(size_t(max_id + 1), 0u);
    token_to_id_.clear();
    token_to_id_.reserve(size_t(max_id + 1) * 2);

    for (const auto& kv : vocab.obj_val) {
        const int32_t id = int32_t(kv.second.as_int());
        id_to_token_[size_t(id)] = kv.first;
        token_to_id_[kv.first] = id;
    }

    // 2. Parse added_tokens
    impl_->added_sorted.clear();
    if (added_tokens.is_array()) {
        for (const auto& at : added_tokens.arr_val) {
            AddedTokenEntry e;
            e.id = int32_t(at.get("id").as_int());
            e.content = at.get("content").as_str();
            e.special = at.get("special").as_bool(false);
            if (e.id >= 0 && size_t(e.id) < id_to_token_.size()) {
                id_to_token_[size_t(e.id)] = e.content;
                id_is_special_[size_t(e.id)] = e.special ? 1u : 0u;
                impl_->is_added_id[size_t(e.id)] = 1u;
                token_to_id_[e.content] = e.id;
                impl_->added_sorted.push_back(std::move(e));
            }
        }
    }
    std::stable_sort(
        impl_->added_sorted.begin(),
        impl_->added_sorted.end(),
        [](const AddedTokenEntry& a, const AddedTokenEntry& b) {
            return a.content.size() > b.content.size();
        });

    // 3. Parse all Split(Regex) pre_tokenizers in sequence!
    for (pcre2_code* re : impl_->split_regexes) {
        if (re) pcre2_code_free(re);
    }
    impl_->split_regexes.clear();
    const JsonValue& pre = root.get("pre_tokenizer");
    const JsonValue& pre_list = pre.get("pretokenizers");
    if (pre_list.is_array()) {
        for (const auto& pt : pre_list.arr_val) {
            if (pt.get("type").as_str() == "Split") {
                const std::string pat = pt.get("pattern").get("Regex").as_str();
                if (!pat.empty()) {
                    int errcode = 0;
                    PCRE2_SIZE erroff = 0;
                    pcre2_code* re = pcre2_compile(
                        reinterpret_cast<PCRE2_SPTR>(pat.data()),
                        pat.size(),
                        PCRE2_UTF | PCRE2_UCP,
                        &errcode,
                        &erroff,
                        nullptr);
                    if (!re) {
                        throw std::runtime_error("Dsv41Tokenizer: PCRE2 compile failed at offset " +
                                                 std::to_string(erroff) + " for pattern: " + pat);
                    }
                    (void)pcre2_jit_compile(re, PCRE2_JIT_COMPLETE);
                    impl_->split_regexes.push_back(re);
                }
            }
        }
    }
    if (impl_->split_regexes.empty()) {
        throw std::runtime_error("Dsv41Tokenizer: found 0 Split regexes in pre_tokenizer");
    }

    // 4. Parse model.merges (both "a b" and ["a", "b"] supported)
    impl_->merge_ranks.clear();
    const JsonValue& merges = model.get("merges");
    if (!merges.is_array()) throw std::runtime_error("Dsv41Tokenizer: missing model.merges");
    impl_->merge_ranks.reserve(merges.arr_val.size() * 2);
    for (uint32_t r = 0; r < uint32_t(merges.arr_val.size()); ++r) {
        const JsonValue& m = merges.arr_val[r];
        if (m.is_string()) {
            const size_t sp = m.s_val.find(' ');
            if (sp != std::string::npos) {
                impl_->merge_ranks.emplace(
                    Impl::make_pair_key(m.s_val.substr(0, sp), m.s_val.substr(sp + 1)), r);
            }
        } else if (m.is_array() && m.arr_val.size() == 2) {
            impl_->merge_ranks.emplace(
                Impl::make_pair_key(m.arr_val[0].as_str(), m.arr_val[1].as_str()), r);
        }
    }

    // 5. Precompute raw byte-level payloads per token ID for O(1) zero-alloc UTF-8 streaming
    const auto& bt = byte_tables();
    id_to_raw_bytes_.assign(id_to_token_.size(), std::string{});
    for (size_t tid = 0; tid < id_to_token_.size(); ++tid) {
        const std::string& tok = id_to_token_[tid];
        if (impl_->is_added_id[tid]) {
            id_to_raw_bytes_[tid] = tok;
            continue;
        }
        std::string rb;
        rb.reserve(tok.size());
        size_t i = 0;
        while (i < tok.size()) {
            const size_t prev = i;
            const uint32_t cp = next_utf8_lossy(tok, i);
            auto it = bt.cp_to_byte.find(cp);
            if (it != bt.cp_to_byte.end()) {
                rb.push_back(char(it->second));
            } else {
                rb.append(tok.data() + prev, i - prev);
            }
        }
        id_to_raw_bytes_[tid] = std::move(rb);
    }
}

const std::string& Dsv41Tokenizer::id_to_token(int32_t id) const {
    static const std::string kEmpty;
    if (id < 0 || size_t(id) >= id_to_token_.size()) return kEmpty;
    return id_to_token_[size_t(id)];
}

int32_t Dsv41Tokenizer::token_to_id(const std::string& tok) const {
    auto it = token_to_id_.find(tok);
    return (it == token_to_id_.end()) ? -1 : it->second;
}

std::vector<int32_t> Dsv41Tokenizer::encode(const std::string& text, bool parse_special) const {
    std::vector<int32_t> out;
    if (text.empty()) return out;

    auto encode_ordinary_span = [&](const std::string& span) {
        if (span.empty()) return;
        std::vector<std::string> pieces = {span};
        std::vector<std::string> next_pieces;
        for (pcre2_code* re : impl_->split_regexes) {
            impl_->apply_split_isolated(re, pieces, next_pieces);
            pieces.swap(next_pieces);
        }
        for (const std::string& p : pieces) {
            impl_->bpe_encode_piece(p, token_to_id_, out);
        }
    };

    if (!parse_special || impl_->added_sorted.empty()) {
        encode_ordinary_span(text);
        return out;
    }

    // Leftmost-longest scan over added/special tokens.
    // Note: in DeepSeek-V4.1, all added tokens start with '<' EXCEPT "｜DSML｜" (starts with 0xEF 0xBD 0x9C)!
    size_t cursor = 0;
    const size_t n = text.size();
    while (cursor < n) {
        size_t best_pos = std::string::npos;
        const AddedTokenEntry* best_entry = nullptr;

        // Fast candidate search: scan forward to next '<' or '｜' (0xEF 0xBD 0x9C)
        size_t probe = cursor;
        while (probe < n) {
            const unsigned char c = static_cast<unsigned char>(text[probe]);
            if (c == '<' || (c == 0xEFu && probe + 2 < n &&
                             uint8_t(text[probe + 1]) == 0xBDu && uint8_t(text[probe + 2]) == 0x9Cu)) {
                for (const auto& e : impl_->added_sorted) {
                    const size_t elen = e.content.size();
                    if (probe + elen <= n && std::memcmp(text.data() + probe, e.content.data(), elen) == 0) {
                        best_pos = probe;
                        best_entry = &e;
                        break;  // added_sorted is sorted descending by length -> longest match at earliest pos!
                    }
                }
                if (best_entry) break;
            }
            ++probe;
        }

        if (!best_entry) {
            encode_ordinary_span(text.substr(cursor));
            break;
        }
        if (best_pos > cursor) {
            encode_ordinary_span(text.substr(cursor, best_pos - cursor));
        }
        out.push_back(best_entry->id);
        cursor = best_pos + best_entry->content.size();
    }
    return out;
}

namespace {

struct TokSliceHashKey {
    uint64_t h1 = 0;
    uint64_t h2 = 0;
    size_t len = 0;
    bool operator==(const TokSliceHashKey& o) const noexcept {
        return h1 == o.h1 && h2 == o.h2 && len == o.len;
    }
};

struct TokSliceHashKeyFn {
    size_t operator()(const TokSliceHashKey& k) const noexcept {
        return static_cast<size_t>(k.h1 ^ (k.h2 * 0x9E3779B97F4A7C15ULL) ^ k.len);
    }
};

struct TokSliceCacheEntry {
    std::string text;
    std::vector<int32_t> ids;
};

struct TokSliceCache {
    static constexpr size_t kMaxEntries = 8192u;
    std::mutex mtx;
    std::unordered_map<TokSliceHashKey, TokSliceCacheEntry, TokSliceHashKeyFn> map;
    std::deque<TokSliceHashKey> fifo;
    const void* owner_tok = nullptr;

    static TokSliceHashKey hash_sv(std::string_view sv) noexcept {
        uint64_t h1 = 0xcbf29ce484222325ULL;
        uint64_t h2 = 0x84222325cbf29ce4ULL ^ uint64_t(sv.size());
        const auto* p = reinterpret_cast<const uint8_t*>(sv.data());
        size_t i = 0;
        const size_t n = sv.size();
        for (; i + 8u <= n; i += 8u) {
            uint64_t w = 0;
            std::memcpy(&w, p + i, 8u);
            h1 ^= w;
            h1 *= 0x100000001b3ULL;
            h2 ^= (w + 0x9e3779b97f4a7c15ULL);
            h2 = (h2 << 13) | (h2 >> 51);
            h2 *= 0xbf58476d1ce4e5b9ULL;
        }
        for (; i < n; ++i) {
            h1 ^= uint64_t(p[i]);
            h1 *= 0x100000001b3ULL;
            h2 ^= uint64_t(p[i]) * 0x9e3779b97f4a7c15ULL;
            h2 = (h2 << 7) | (h2 >> 57);
        }
        return TokSliceHashKey{h1, h2, n};
    }

    bool lookup_locked(const void* tok_ptr, const TokSliceHashKey& k, std::string_view sv, std::vector<int32_t>& out) {
        if (owner_tok != tok_ptr) return false;
        auto it = map.find(k);
        if (it == map.end() || it->second.text.size() != sv.size()) return false;
        if (std::memcmp(it->second.text.data(), sv.data(), sv.size()) != 0) return false;
        out = it->second.ids;
        return true;
    }

    void insert_locked(const void* tok_ptr, const TokSliceHashKey& k, std::string_view sv, const std::vector<int32_t>& ids) {
        if (owner_tok != tok_ptr) {
            map.clear();
            fifo.clear();
            owner_tok = tok_ptr;
        }
        if (map.find(k) != map.end()) return;
        if (map.size() >= kMaxEntries && !fifo.empty()) {
            map.erase(fifo.front());
            fifo.pop_front();
        }
        TokSliceCacheEntry e;
        e.text.assign(sv.data(), sv.size());
        e.ids = ids;
        map.emplace(k, std::move(e));
        fifo.push_back(k);
    }
};

TokSliceCache& global_tok_slice_cache() {
    static TokSliceCache cache;
    return cache;
}

}  // namespace

std::vector<int32_t> Dsv41Tokenizer::encode_segments(const std::vector<PromptSegment>& segments) const {
    if (segments.empty()) return {};

    size_t total_ordinary_bytes = 0;
    for (const auto& seg : segments) {
        if (seg.kind == PromptSegment::Kind::OrdinaryText) {
            total_ordinary_bytes += seg.text.size();
        }
    }

    auto& scache = global_tok_slice_cache();
    if (total_ordinary_bytes < 8192u) {
        std::vector<int32_t> out;
        out.reserve(segments.size() + total_ordinary_bytes / 3u);
        for (const auto& seg : segments) {
            if (seg.kind == PromptSegment::Kind::TokenId) {
                if (seg.token_id >= 0 && seg.repeat > 0) {
                    out.insert(out.end(), size_t(seg.repeat), seg.token_id);
                }
            } else if (!seg.text.empty()) {
                std::vector<int32_t> sub;
                bool hit = false;
                if (seg.text.size() >= 64u) {
                    const auto hk = TokSliceCache::hash_sv(seg.text);
                    std::lock_guard<std::mutex> lk(scache.mtx);
                    hit = scache.lookup_locked(this, hk, seg.text, sub);
                }
                if (!hit) {
                    sub = encode(seg.text, /*parse_special=*/false);
                    if (seg.text.size() >= 64u) {
                        const auto hk = TokSliceCache::hash_sv(seg.text);
                        std::lock_guard<std::mutex> lk(scache.mtx);
                        scache.insert_locked(this, hk, seg.text, sub);
                    }
                }
                out.insert(out.end(), sub.begin(), sub.end());
            }
        }
        return out;
    }

    struct WorkSlice {
        bool is_tok = false;
        bool cache_hit = false;
        int32_t token_id = -1;
        int32_t repeat = 0;
        std::string_view sv;
        TokSliceHashKey hk{};
    };

    std::vector<WorkSlice> slices;
    slices.reserve(segments.size() + total_ordinary_bytes / 2048u + 8u);
    static constexpr size_t kTargetSliceBytes = 2048u;

    for (const auto& seg : segments) {
        if (seg.kind == PromptSegment::Kind::TokenId) {
            if (seg.token_id >= 0 && seg.repeat > 0) {
                slices.push_back(WorkSlice{true, false, seg.token_id, seg.repeat, {}, {}});
            }
            continue;
        }
        const std::string& s = seg.text;
        if (s.empty()) continue;
        if (s.size() <= kTargetSliceBytes * 2u) {
            slices.push_back(WorkSlice{false, false, -1, 0, std::string_view(s), {}});
            continue;
        }
        size_t start = 0;
        while (start < s.size()) {
            if (s.size() - start <= kTargetSliceBytes) {
                slices.push_back(WorkSlice{false, false, -1, 0, std::string_view(s.data() + start, s.size() - start), {}});
                break;
            }
            size_t p = start + kTargetSliceBytes;
            while (p < s.size()) {
                const unsigned char cur = static_cast<unsigned char>(s[p]);
                if (s[p - 1] == '\n' && cur > 0x20u && (cur & 0xC0u) != 0x80u) {
                    break;
                }
                ++p;
            }
            slices.push_back(WorkSlice{false, false, -1, 0, std::string_view(s.data() + start, p - start), {}});
            start = p;
        }
    }

    std::vector<std::vector<int32_t>> encoded_slices(slices.size());
    const int n_slices = static_cast<int>(slices.size());
    int miss_cnt = 0;
    {
        for (int i = 0; i < n_slices; ++i) {
            auto& ws = slices[size_t(i)];
            if (!ws.is_tok && ws.sv.size() >= 32u && ws.sv.size() <= 16384u) {
                ws.hk = TokSliceCache::hash_sv(ws.sv);
            }
        }
        std::lock_guard<std::mutex> lk(scache.mtx);
        for (int i = 0; i < n_slices; ++i) {
            auto& ws = slices[size_t(i)];
            if (!ws.is_tok && ws.sv.size() >= 32u && ws.sv.size() <= 16384u) {
                if (scache.lookup_locked(this, ws.hk, ws.sv, encoded_slices[size_t(i)])) {
                    ws.cache_hit = true;
                } else {
                    ++miss_cnt;
                }
            } else if (!ws.is_tok && !ws.sv.empty()) {
                ++miss_cnt;
            }
        }
    }

    if (miss_cnt > 0) {
        #pragma omp parallel for schedule(dynamic, 1) if(miss_cnt >= 2)
        for (int i = 0; i < n_slices; ++i) {
            const auto& ws = slices[size_t(i)];
            if (!ws.is_tok && !ws.cache_hit && !ws.sv.empty()) {
                encoded_slices[size_t(i)] = encode(std::string(ws.sv), /*parse_special=*/false);
            }
        }
        std::lock_guard<std::mutex> lk(scache.mtx);
        for (int i = 0; i < n_slices; ++i) {
            const auto& ws = slices[size_t(i)];
            if (!ws.is_tok && !ws.cache_hit && ws.sv.size() >= 32u && ws.sv.size() <= 16384u) {
                scache.insert_locked(this, ws.hk, ws.sv, encoded_slices[size_t(i)]);
            }
        }
    }

    size_t total_ids = 0;
    for (size_t i = 0; i < slices.size(); ++i) {
        total_ids += slices[i].is_tok ? size_t(slices[i].repeat) : encoded_slices[i].size();
    }
    std::vector<int32_t> out;
    out.reserve(total_ids);
    for (size_t i = 0; i < slices.size(); ++i) {
        if (slices[i].is_tok) {
            out.insert(out.end(), size_t(slices[i].repeat), slices[i].token_id);
        } else {
            const auto& v = encoded_slices[i];
            out.insert(out.end(), v.begin(), v.end());
        }
    }
    return out;
}

std::string_view Dsv41Tokenizer::raw_bytes_single(int32_t id, bool skip_special_tokens) const {
    if (id < 0 || size_t(id) >= id_to_raw_bytes_.size()) return {};
    if (skip_special_tokens && id_is_special_[size_t(id)]) return {};
    return id_to_raw_bytes_[size_t(id)];
}

std::string Dsv41Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special_tokens) const {
    std::string raw_bytes;
    raw_bytes.reserve(ids.size() * 4);
    for (int32_t id : ids) {
        const std::string_view rb = raw_bytes_single(id, skip_special_tokens);
        if (!rb.empty()) raw_bytes.append(rb.data(), rb.size());
    }
    return bytes_to_utf8_lossy(raw_bytes);
}

std::string Dsv41Tokenizer::decode_single(int32_t id, bool skip_special_tokens) const {
    const std::string_view rb = raw_bytes_single(id, skip_special_tokens);
    if (rb.empty()) return {};
    return bytes_to_utf8_lossy(std::string(rb));
}

void Dsv41Tokenizer::build_engram_compressed_map(
    std::vector<int32_t>& out_lookup,
    int32_t& out_compressed_vocab_size) const
{
    UErrorCode status = U_ZERO_ERROR;
    const icu::Normalizer2* nfkc = icu::Normalizer2::getNFKCInstance(status);
    const icu::Normalizer2* nfd = icu::Normalizer2::getNFDInstance(status);
    if (U_FAILURE(status) || !nfkc || !nfd) {
        throw std::runtime_error("Dsv41Tokenizer: ICU Normalizer2 initialization failed");
    }

    const auto& bt = byte_tables();
    const size_t V = id_to_token_.size();
    out_lookup.assign(V, 0);
    std::unordered_map<std::string, int32_t> key_to_new;
    key_to_new.reserve(V);

    for (size_t token_id = 0; token_id < V; ++token_id) {
        const std::string& raw_tok = id_to_token_[token_id];
        bool had_fffd = false;
        std::string text;
        if (impl_->is_added_id[token_id]) {
            text = raw_tok;
            if (text.find("\xEF\xBF\xBD") != std::string::npos) had_fffd = true;
        } else {
            std::string raw_bytes;
            raw_bytes.reserve(raw_tok.size());
            size_t i = 0;
            while (i < raw_tok.size()) {
                const size_t prev = i;
                const uint32_t cp = next_utf8_lossy(raw_tok, i);
                auto it = bt.cp_to_byte.find(cp);
                if (it != bt.cp_to_byte.end()) {
                    raw_bytes.push_back(char(it->second));
                } else {
                    raw_bytes.append(raw_tok.data() + prev, i - prev);
                }
            }
            text = bytes_to_utf8_lossy(raw_bytes, &had_fffd);
        }

        std::string key;
        if (had_fffd) {
            key = raw_tok;
        } else {
            // Exact engram.py normalizer sequence:
            //   1. NFKC()
            //   2. NFD()
            //   3. StripAccents()  -> remove General_Category == Mn (U_NON_SPACING_MARK)
            //   4. Lowercase()     -> Unicode root-locale lowercase
            //   5. Replace(Regex("[ \t\r\n]+"), " ")
            //   6. Replace(Regex("^ $"), "\ue000")
            //   7. Strip()         -> strip Unicode White_Space (UCHAR_WHITE_SPACE) left & right
            //   8. Replace("\ue000", " ")
            UErrorCode st = U_ZERO_ERROR;
            const icu::UnicodeString u_in = icu::UnicodeString::fromUTF8(icu::StringPiece(text.data(), int32_t(text.size())));
            icu::UnicodeString u_nfkc = nfkc->normalize(u_in, st);
            icu::UnicodeString u_nfd = nfd->normalize(u_nfkc, st);

            icu::UnicodeString u_stripped_mn;
            for (int32_t idx = 0; idx < u_nfd.length();) {
                const UChar32 c = u_nfd.char32At(idx);
                idx += U16_LENGTH(c);
                if ((U_GET_GC_MASK(c) & U_GC_M_MASK) == 0) {
                    u_stripped_mn.append(c);
                }
            }
            u_stripped_mn.toLower(icu::Locale::getRoot());

            // Step 5: collapse [ \t\r\n]+ -> " "
            icu::UnicodeString u_ws;
            bool in_ascii_ws = false;
            for (int32_t idx = 0; idx < u_stripped_mn.length();) {
                const UChar32 c = u_stripped_mn.char32At(idx);
                idx += U16_LENGTH(c);
                if (c == 0x20 || c == 0x09 || c == 0x0D || c == 0x0A) {
                    if (!in_ascii_ws) {
                        u_ws.append(UChar32(0x20));
                        in_ascii_ws = true;
                    }
                } else {
                    in_ascii_ws = false;
                    u_ws.append(c);
                }
            }

            // Step 6: Replace(Regex("^ $"), "\ue000")
            if (u_ws.length() == 1 && u_ws.char32At(0) == 0x20) {
                u_ws.setTo(UChar32(0xE000));
            }

            // Step 7: Strip() using Unicode White_Space property (Rust char::is_whitespace)
            int32_t left = 0;
            int32_t right = u_ws.length();
            while (left < right) {
                const UChar32 c = u_ws.char32At(left);
                if (u_hasBinaryProperty(c, UCHAR_WHITE_SPACE)) left += U16_LENGTH(c);
                else break;
            }
            while (right > left) {
                const UChar32 c = u_ws.char32At(right - 1);
                // Handle supplementary chars cleanly via moveIndex32
                const int32_t prev_idx = u_ws.moveIndex32(right, -1);
                const UChar32 cp = u_ws.char32At(prev_idx);
                (void)c;
                if (u_hasBinaryProperty(cp, UCHAR_WHITE_SPACE)) right = prev_idx;
                else break;
            }
            icu::UnicodeString u_trimmed = u_ws.tempSubStringBetween(left, right);

            // Step 8: Replace("\ue000", " ")
            icu::UnicodeString u_final;
            for (int32_t idx = 0; idx < u_trimmed.length();) {
                const UChar32 c = u_trimmed.char32At(idx);
                idx += U16_LENGTH(c);
                u_final.append(c == 0xE000 ? UChar32(0x20) : c);
            }

            std::string normalized;
            u_final.toUTF8String(normalized);
            key = normalized.empty() ? text : normalized;
        }

        auto [it, inserted] = key_to_new.emplace(key, int32_t(key_to_new.size()));
        (void)inserted;
        out_lookup[token_id] = it->second;
    }

    out_compressed_vocab_size = int32_t(key_to_new.size());
}

// =============================================================================
// V4.1 Chat Prompt Rendering & DSML Tool Schema / Completion Parsing
// Exact port of hf_meta/encoding/encoding.py
// =============================================================================

namespace {

static const char* kEosTok       = "<\xEF\xBD\x9C" "end\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>";
static const char* kThinkEnd     = "</think>";
static const char* kDsmlTok      = "\xEF\xBD\x9C" "DSML\xEF\xBD\x9C";

bool is_image_json_block(const JsonValue& b) {
    if (!b.is_object()) return false;
    const std::string& bt = b.get("type").as_str();
    return bt == "image" || bt == "image_url" || bt == "input_image";
}

Dsv41ImageSource extract_image_source_from_json_block(const JsonValue& b) {
    const std::string& bt = b.get("type").as_str();
    if (bt == "image_url") {
        const JsonValue& iu = b.get("image_url");
        const std::string url = iu.is_string() ? iu.s_val : iu.get("url").as_str();
        if (url.empty()) {
            throw std::runtime_error("image_url block missing url");
        }
        return load_image_source_from_spec(url);
    }
    if (bt == "input_image") {
        if (b.has("image_url")) {
            const JsonValue& iu = b.get("image_url");
            const std::string url = iu.is_string() ? iu.s_val : iu.get("url").as_str();
            if (!url.empty()) return load_image_source_from_spec(url);
        }
        if (b.has("url") && b.get("url").is_string() && !b.get("url").s_val.empty()) {
            return load_image_source_from_spec(b.get("url").s_val);
        }
        if (b.has("image_data") && b.get("image_data").is_string() && !b.get("image_data").s_val.empty()) {
            return make_image_source_from_base64(b.get("image_data").s_val, b.get("media_type").as_str());
        }
        throw std::runtime_error("input_image block does not contain a valid source");
    }
    if (b.has("source") && b.get("source").is_object()) {
        const JsonValue& src = b.get("source");
        const std::string st = src.get("type").as_str();
        if (st == "base64" || (!src.get("data").as_str().empty() && st != "url")) {
            return make_image_source_from_base64(src.get("data").as_str(), src.get("media_type").as_str());
        }
        if (st == "url" || !src.get("url").as_str().empty()) {
            return load_image_source_from_spec(src.get("url").as_str());
        }
    }
    if (b.has("data") && b.get("data").is_string() && !b.get("data").s_val.empty()) {
        return make_image_source_from_base64(b.get("data").s_val, b.get("media_type").as_str());
    }
    if (b.has("url") && b.get("url").is_string() && !b.get("url").s_val.empty()) {
        return load_image_source_from_spec(b.get("url").s_val);
    }
    throw std::runtime_error("Image block does not contain a valid source");
}

struct PromptSegmentBuilder {
    std::vector<PromptSegment> segs;

    void push_tok(int32_t id, int32_t count = 1) {
        if (id < 0 || count <= 0) return;
        if (!segs.empty() && segs.back().kind == PromptSegment::Kind::TokenId && segs.back().token_id == id) {
            segs.back().repeat += count;
        } else {
            segs.push_back(PromptSegment::tok(id, count));
        }
    }

    void push_txt(std::string_view s) {
        if (s.empty()) return;
        if (!segs.empty() && segs.back().kind == PromptSegment::Kind::OrdinaryText) {
            segs.back().text.append(s.data(), s.size());
        } else {
            segs.push_back(PromptSegment::txt(std::string(s)));
        }
    }
};

std::pair<std::string, std::string> split_tool_name(const std::string& raw_name, const std::string& explicit_ns = "") {
    const size_t pos = raw_name.find("::");
    if (pos != std::string::npos) {
        return {raw_name.substr(0, pos), raw_name.substr(pos + 2)};
    }
    return {explicit_ns, raw_name};
}

std::string tool_name_for_encoding(const JsonValue& fn_obj) {
    std::string ns;
    if (const JsonValue* nsv = fn_obj.find("namespace")) {
        if (nsv->is_string()) ns = nsv->s_val;
        else if (nsv->is_object()) ns = nsv->get("name").as_str();
    }
    auto [res_ns, bare] = split_tool_name(fn_obj.get("name").as_str(), ns);
    return res_ns.empty() ? bare : (res_ns + "::" + bare);
}

std::vector<JsonValue> tools_from_openai_format(const JsonValue& tools) {
    std::vector<JsonValue> out;
    if (!tools.is_array()) return out;
    for (const auto& t : tools.arr_val) {
        JsonValue fn = t.has("function") ? t.get("function") : t;
        if (!fn.is_object()) continue;
        if (t.has("namespace") && !t.get("namespace").is_null()) {
            fn.set("namespace", t.get("namespace"));
        }
        fn.set("name", JsonValue::make_string(tool_name_for_encoding(fn)));
        if (const JsonValue* nsv = fn.find("namespace")) {
            if (nsv->is_object() && !nsv->get("description").as_str().empty()) {
                const std::string desc = nsv->get("description").as_str() + "\n" + fn.get("description").as_str();
                fn.set("description", JsonValue::make_string(desc));
            }
            fn.erase("namespace");
        }
        out.push_back(std::move(fn));
    }
    return out;
}

void append_tools_block_segments(PromptSegmentBuilder& bld, const std::vector<JsonValue>& fns) {
    std::string schemas;
    for (size_t i = 0; i < fns.size(); ++i) {
        if (i) schemas.push_back('\n');
        schemas += fns[i].dumps_py();
    }
    bld.push_txt("## Tools\n\n");
    bld.push_txt("You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<");
    bld.push_tok(Dsv41SpecialTokens::kDsml);
    bld.push_txt(" calls>\" block like the following:\n\n<");
    bld.push_tok(Dsv41SpecialTokens::kDsml);
    bld.push_txt(" calls>\n<");
    bld.push_tok(Dsv41SpecialTokens::kDsml);
    bld.push_txt(" invoke name=\"$TOOL_NAME\">\n<");
    bld.push_tok(Dsv41SpecialTokens::kDsml);
    bld.push_txt(" parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</");
    bld.push_tok(Dsv41SpecialTokens::kDsml);
    bld.push_txt(" parameter>\n...\n</");
    bld.push_tok(Dsv41SpecialTokens::kDsml);
    bld.push_txt(" invoke>\n<");
    bld.push_tok(Dsv41SpecialTokens::kDsml);
    bld.push_txt(" invoke name=\"$TOOL_NAME2\">\n...\n</");
    bld.push_tok(Dsv41SpecialTokens::kDsml);
    bld.push_txt(" invoke>\n</");
    bld.push_tok(Dsv41SpecialTokens::kDsml);
    bld.push_txt(" calls>\n\n");
    bld.push_txt("String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n");
    bld.push_txt("If thinking_mode is enabled (triggered by ");
    bld.push_tok(Dsv41SpecialTokens::kThinkOpen);
    bld.push_txt("), you MUST output your complete reasoning inside ");
    bld.push_tok(Dsv41SpecialTokens::kThinkOpen);
    bld.push_txt("...");
    bld.push_tok(Dsv41SpecialTokens::kThinkEnd);
    bld.push_txt(" BEFORE any tool calls or final response.\n\n");
    bld.push_txt("Otherwise, output directly after ");
    bld.push_tok(Dsv41SpecialTokens::kThinkEnd);
    bld.push_txt(" with tool calls or final response.\n\n");
    bld.push_txt("### Available Tool Schemas\n\n");
    bld.push_txt(schemas);
    bld.push_txt("\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n");
}

void append_arguments_to_dsml_segments(PromptSegmentBuilder& bld, const JsonValue& raw_args) {
    JsonValue args = raw_args;
    for (int iter = 0; iter < 2; ++iter) {
        if (args.is_string()) {
            JsonValue parsed;
            if (JsonParser::try_parse(args.s_val, parsed)) {
                args = std::move(parsed);
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if (!args.is_object()) {
        JsonValue wrap = JsonValue::make_object();
        wrap.set("arguments", raw_args);
        args = std::move(wrap);
    }

    for (size_t i = 0; i < args.obj_val.size(); ++i) {
        if (i) bld.push_txt("\n");
        const std::string& k = args.obj_val[i].first;
        const JsonValue& v = args.obj_val[i].second;
        const bool is_str = v.is_string();
        bld.push_txt("<");
        bld.push_tok(Dsv41SpecialTokens::kDsml);
        bld.push_txt(" parameter name=\"");
        bld.push_txt(k);
        bld.push_txt("\" string=\"");
        bld.push_txt(is_str ? "true" : "false");
        bld.push_txt("\">");
        if (is_str) {
            bld.push_txt(v.s_val);
        } else {
            bld.push_txt(v.dumps_py());
        }
        bld.push_txt("</");
        bld.push_tok(Dsv41SpecialTokens::kDsml);
        bld.push_txt(" parameter>");
    }
}

int resolve_reasoning_budget(const JsonValue* effort_val, std::string& io_thinking_mode) {
    // Official DeepSeek-V4.1 Max recipe default = 100 ("low"=50, "medium"=65, "high"=75, "xhigh"=90, "max"=100, or int 1..100; 0/"none" -> chat mode)
    if (!effort_val || effort_val->is_null()) return (io_thinking_mode == "chat") ? 0 : 100;
    if (effort_val->is_bool()) {
        if (!effort_val->b_val) {
            io_thinking_mode = "chat";
            return 0;
        }
        return 100;
    }
    if (effort_val->is_int() || effort_val->is_number()) {
        const int v = int(effort_val->as_int(100));
        if (v <= 0) {
            io_thinking_mode = "chat";
            return 0;
        }
        return std::max(1, std::min(100, v));
    }
    if (effort_val->is_string()) {
        std::string s = effort_val->s_val;
        size_t a = 0;
        while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
        size_t b = s.size();
        while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
        s = s.substr(a, b - a);
        for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
        if (s.empty()) return (io_thinking_mode == "chat") ? 0 : 100;
        if (s == "none" || s == "off" || s == "disabled" || s == "false" || s == "chat" || s == "0") {
            io_thinking_mode = "chat";
            return 0;
        }
        if (s == "minimal") return 35;
        if (s == "low") return 50;
        if (s == "medium" || s == "med") return 65;
        if (s == "high") return 75;
        if (s == "xhigh" || s == "x-high" || s == "very_high") return 90;
        if (s == "max" || s == "ultrathink") return 100;
        char* endp = nullptr;
        const long iv = std::strtol(s.c_str(), &endp, 10);
        if (endp && *endp == '\0') {
            if (iv <= 0) {
                io_thinking_mode = "chat";
                return 0;
            }
            return int(std::max(1L, std::min(100L, iv)));
        }
    }
    return (io_thinking_mode == "chat") ? 0 : 100;
}

int find_last_user_index(const std::vector<JsonValue>& msgs) {
    for (int idx = int(msgs.size()) - 1; idx >= 0; --idx) {
        const std::string& role = msgs[size_t(idx)].get("role").as_str();
        if (role == "user" || (role == "system" && idx > 0)) {
            const JsonValue* cb = msgs[size_t(idx)].find("content_blocks");
            if (cb && cb->is_array() && !cb->arr_val.empty()) {
                bool has_user_block = false;
                for (const auto& b : cb->arr_val) {
                    const std::string& bt = b.get("type").as_str();
                    if (bt == "text" || bt == "image" || bt == "image_url" || bt == "input_image") {
                        has_user_block = true;
                        break;
                    }
                }
                if (!has_user_block) continue;
            }
            return idx;
        }
    }
    return -1;
}

std::string extract_text_content(const JsonValue& c) {
    if (c.is_string()) return c.s_val;
    if (c.is_array()) {
        std::string joined;
        for (const auto& b : c.arr_val) {
            if (b.is_object() && b.get("type").as_str() == "text") {
                if (!joined.empty()) joined += "\n\n";
                joined += b.get("text").as_str();
            } else if (b.is_string()) {
                if (!joined.empty()) joined += "\n\n";
                joined += b.s_val;
            }
        }
        return joined;
    }
    return "";
}

}  // namespace

std::vector<PromptSegment> render_dsv41_chat_segments(
    const JsonValue& raw_messages,
    const JsonValue* tools,
    const JsonValue* response_format,
    const JsonValue* reasoning_effort_val,
    std::string& out_thinking_mode,
    int& out_budget,
    std::vector<Dsv41ImageSource>* out_images,
    int prefix_base_effort)
{
    if (out_images) out_images->clear();

    if (out_thinking_mode != "chat" && out_thinking_mode != "thinking") {
        out_thinking_mode = "thinking";
    }
    out_budget = resolve_reasoning_budget(reasoning_effort_val, out_thinking_mode);

    // 1. Copy messages and attach tools/response_format to leading system message (prepending empty system msg if needed)
    std::vector<JsonValue> input_msgs;
    if (raw_messages.is_array()) input_msgs = raw_messages.arr_val;

    const bool has_tools = (tools && tools->is_array() && !tools->arr_val.empty());
    const bool has_resp_fmt = (response_format && response_format->is_object() && !response_format->obj_val.empty() &&
                               response_format->get("type").as_str() != "text");

    if ((has_tools || has_resp_fmt) && (input_msgs.empty() || input_msgs[0].get("role").as_str() != "system")) {
        JsonValue sys = JsonValue::make_object();
        sys.set("role", JsonValue::make_string("system"));
        sys.set("content", JsonValue::make_string(""));
        input_msgs.insert(input_msgs.begin(), std::move(sys));
    }
    if (!input_msgs.empty() && input_msgs[0].get("role").as_str() == "system") {
        if (has_tools && !input_msgs[0].has("tools")) input_msgs[0].set("tools", *tools);
        if (has_resp_fmt && !input_msgs[0].has("response_format")) input_msgs[0].set("response_format", *response_format);
    }

    // 2. merge_tool_messages (encoding.py:184-230)
    std::vector<JsonValue> merged;
    for (const auto& msg_in : input_msgs) {
        JsonValue msg = msg_in;
        std::string role = msg.get("role").as_str();
        if (role == "developer") {
            role = "system";
            msg.set("role", JsonValue::make_string("system"));
        }
        if (role == "tool") {
            JsonValue tb = JsonValue::make_object();
            tb.set("type", JsonValue::make_string("tool_result"));
            tb.set("tool_use_id", JsonValue::make_string(msg.get("tool_call_id").as_str()));
            tb.set("content", msg.has("content") ? msg.get("content") : JsonValue::make_string(""));
            if (!merged.empty() && merged.back().get("role").as_str() == "user" && merged.back().has("content_blocks")) {
                merged.back().find_mut("content_blocks")->push(std::move(tb));
            } else {
                JsonValue u = JsonValue::make_object();
                u.set("role", JsonValue::make_string("user"));
                JsonValue arr = JsonValue::make_array();
                arr.push(std::move(tb));
                u.set("content_blocks", std::move(arr));
                merged.push_back(std::move(u));
            }
        } else if (role == "user") {
            JsonValue cbs = JsonValue::make_array();
            if (msg.has("content_blocks") && msg.get("content_blocks").is_array()) {
                cbs = msg.get("content_blocks");
            } else if (msg.get("content").is_array()) {
                cbs = msg.get("content");
            } else {
                JsonValue txt = JsonValue::make_object();
                txt.set("type", JsonValue::make_string("text"));
                txt.set("text", JsonValue::make_string(msg.get("content").as_str()));
                cbs.push(std::move(txt));
            }
            if (!merged.empty() && merged.back().get("role").as_str() == "user" &&
                merged.back().has("content_blocks") && !merged.back().has("task")) {
                JsonValue* dst = merged.back().find_mut("content_blocks");
                for (auto& el : cbs.arr_val) dst->push(std::move(el));
            } else {
                msg.set("content_blocks", std::move(cbs));
                merged.push_back(std::move(msg));
            }
        } else {
            merged.push_back(std::move(msg));
        }
    }

    // 3. sort_tool_results_by_call_order (encoding.py:233-266)
    std::unordered_map<std::string, int> call_order;
    for (auto& msg : merged) {
        const std::string& role = msg.get("role").as_str();
        if (role == "assistant" && msg.get("tool_calls").is_array()) {
            call_order.clear();
            const auto& tcs = msg.get("tool_calls").arr_val;
            for (int i = 0; i < int(tcs.size()); ++i) {
                std::string id = tcs[size_t(i)].get("id").as_str();
                if (id.empty()) id = tcs[size_t(i)].get("function").get("id").as_str();
                if (!id.empty()) call_order[id] = i;
            }
        } else if (role == "user" && msg.get("content_blocks").is_array() && !call_order.empty()) {
            auto& blks = msg.find_mut("content_blocks")->arr_val;
            std::vector<JsonValue> tool_blks;
            for (const auto& b : blks) {
                if (b.get("type").as_str() == "tool_result") tool_blks.push_back(b);
            }
            if (tool_blks.size() > 1) {
                std::stable_sort(tool_blks.begin(), tool_blks.end(), [&](const JsonValue& a, const JsonValue& b) {
                    const auto ia = call_order.find(a.get("tool_use_id").as_str());
                    const auto ib = call_order.find(b.get("tool_use_id").as_str());
                    const int oa = (ia == call_order.end()) ? 0 : ia->second;
                    const int ob = (ib == call_order.end()) ? 0 : ib->second;
                    return oa < ob;
                });
                size_t si = 0;
                for (auto& b : blks) {
                    if (b.get("type").as_str() == "tool_result") b = std::move(tool_blks[si++]);
                }
            }
        }
    }

    // 4. drop_thinking resolution (encoding.py:785-792): never drop thinking when tools are present!
    bool any_msg_tools = false;
    for (const auto& m : merged) {
        if (m.has("tools") && m.get("tools").is_array() && !m.get("tools").arr_val.empty()) {
            any_msg_tools = true;
            break;
        }
    }
    const bool drop_thinking = !any_msg_tools;
    const int last_user_idx = find_last_user_index(merged);

    if (out_thinking_mode == "thinking" && drop_thinking) {
        for (int idx = 0; idx < int(merged.size()); ++idx) {
            if (idx < last_user_idx && merged[size_t(idx)].get("role").as_str() == "assistant") {
                merged[size_t(idx)].erase("reasoning_content");
                merged[size_t(idx)].erase("thinking");
            }
        }
    }

    // 5. Render full prompt as anti-injection PromptSegments starting with BOS (deepseek-recipe + encoding.py)
    PromptSegmentBuilder bld;
    bld.push_tok(Dsv41SpecialTokens::kBos);

    for (int index = 0; index < int(merged.size()); ++index) {
        const JsonValue& msg = merged[size_t(index)];
        const std::string role = msg.get("role").as_str();

        std::string effort_prefix;
        if (index == 0 && out_thinking_mode == "thinking") {
            const int hdr_effort = (prefix_base_effort >= 1 && prefix_base_effort <= 100)
                ? prefix_base_effort : out_budget;
            effort_prefix = "Reasoning Effort: " + std::to_string(hdr_effort) +
                            " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
        }
        if (index == 0 && (!effort_prefix.empty() || role == "system")) {
            bld.push_tok(Dsv41SpecialTokens::kSystem);
        }
        bld.push_txt(effort_prefix);

        if (role == "system") {
            if (index > 0) bld.push_tok(Dsv41SpecialTokens::kSystem);
            bld.push_txt(extract_text_content(msg.get("content")));
            if (msg.get("tools").is_array() && !msg.get("tools").arr_val.empty()) {
                bld.push_txt("\n\n");
                append_tools_block_segments(bld, tools_from_openai_format(msg.get("tools")));
            }
            if (msg.get("response_format").is_object() && !msg.get("response_format").obj_val.empty()) {
                bld.push_txt("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
                bld.push_txt(msg.get("response_format").dumps_py());
            }
        } else if (role == "user") {
            bld.push_tok(Dsv41SpecialTokens::kUser);
            if (msg.get("content_blocks").is_array()) {
                const auto& blks = msg.get("content_blocks").arr_val;
                for (size_t bi = 0; bi < blks.size(); ++bi) {
                    if (bi) bld.push_txt("\n\n");
                    const auto& b = blks[bi];
                    const std::string bt = b.get("type").as_str();
                    if (bt == "text") {
                        bld.push_txt(b.get("text").as_str());
                    } else if (is_image_json_block(b)) {
                        if (out_images) out_images->push_back(extract_image_source_from_json_block(b));
                        bld.push_tok(Dsv41SpecialTokens::kImage);
                    } else if (bt == "tool_result") {
                        bld.push_txt("<tool_result>");
                        const JsonValue& tc = b.get("content");
                        if (tc.is_array()) {
                            for (size_t si = 0; si < tc.arr_val.size(); ++si) {
                                if (si) bld.push_txt("\n\n");
                                const JsonValue& sub = tc.arr_val[si];
                                if (sub.is_string()) {
                                    bld.push_txt(sub.s_val);
                                } else if (sub.is_object() && sub.get("type").as_str() == "text") {
                                    bld.push_txt(sub.get("text").as_str());
                                } else if (is_image_json_block(sub)) {
                                    if (out_images) out_images->push_back(extract_image_source_from_json_block(sub));
                                    bld.push_tok(Dsv41SpecialTokens::kImage);
                                } else if (sub.is_object()) {
                                    bld.push_txt("[Unsupported " + sub.get("type").as_str() + "]");
                                }
                            }
                        } else {
                            bld.push_txt(extract_text_content(tc));
                        }
                        bld.push_txt("</tool_result>");
                    } else {
                        bld.push_txt("[Unsupported " + bt + "]");
                    }
                }
            } else {
                bld.push_txt(extract_text_content(msg.get("content")));
            }
        } else if (role == "latest_reminder") {
            bld.push_tok(Dsv41SpecialTokens::kLatestReminder);
            bld.push_txt(extract_text_content(msg.get("content")));
        } else if (role == "assistant") {
            if (out_thinking_mode == "thinking") {
                const bool prev_has_task = (index > 0 && merged[size_t(index - 1)].has("task") &&
                                            !merged[size_t(index - 1)].get("task").is_null());
                if (!prev_has_task && (!drop_thinking || index > last_user_idx)) {
                    const std::string rc = msg.has("reasoning_content") && !msg.get("reasoning_content").is_null()
                                               ? msg.get("reasoning_content").as_str()
                                               : msg.get("thinking").as_str();
                    bld.push_txt(rc);
                    bld.push_tok(Dsv41SpecialTokens::kThinkEnd);
                }
            }
            bld.push_txt(extract_text_content(msg.get("content")));
            if (msg.get("tool_calls").is_array() && !msg.get("tool_calls").arr_val.empty()) {
                bld.push_txt("\n\n<");
                bld.push_tok(Dsv41SpecialTokens::kDsml);
                bld.push_txt(" calls>\n");
                const auto& tcs = msg.get("tool_calls").arr_val;
                for (size_t ti = 0; ti < tcs.size(); ++ti) {
                    if (ti) bld.push_txt("\n");
                    const JsonValue& tc = tcs[ti];
                    const JsonValue& fn = tc.has("function") ? tc.get("function") : tc;
                    std::string ns = tc.get("namespace").as_str();
                    if (ns.empty()) ns = fn.get("namespace").as_str();
                    auto [rns, bare] = split_tool_name(fn.get("name").as_str(), ns);
                    const std::string qname = rns.empty() ? bare : (rns + "::" + bare);
                    bld.push_txt("<");
                    bld.push_tok(Dsv41SpecialTokens::kDsml);
                    bld.push_txt(" invoke name=\"");
                    bld.push_txt(qname);
                    bld.push_txt("\">\n");
                    append_arguments_to_dsml_segments(bld, fn.get("arguments"));
                    bld.push_txt("\n</");
                    bld.push_tok(Dsv41SpecialTokens::kDsml);
                    bld.push_txt(" invoke>");
                }
                bld.push_txt("\n</");
                bld.push_tok(Dsv41SpecialTokens::kDsml);
                bld.push_txt(" calls>");
            }
            if (!msg.get("wo_eos").as_bool(false)) {
                bld.push_tok(Dsv41SpecialTokens::kEos);
            }
        }

        if (index + 1 < int(merged.size())) {
            const std::string& next_role = merged[size_t(index + 1)].get("role").as_str();
            if (next_role != "assistant" && next_role != "latest_reminder") {
                continue;
            }
        }

        if (role == "user" || (role == "system" && index > 0)) {
            bld.push_tok(Dsv41SpecialTokens::kAssistant);
            if ((!drop_thinking && out_thinking_mode == "thinking") ||
                (drop_thinking && out_thinking_mode == "thinking" && index >= last_user_idx)) {
                bld.push_tok(Dsv41SpecialTokens::kThinkOpen);
            } else {
                bld.push_tok(Dsv41SpecialTokens::kThinkEnd);
            }
        }
    }

    // Wave 9.4 Phase E: Prefix-preserving active-turn Reasoning Effort override via native <｜latest_reminder｜>
    // when effort differs from the fixed index==0 header (keeps tokens [0..N_history] 100.0% bit-identical across
    // max <-> high <-> medium <-> low toggles -> >99% RadixBlockCache hit rate, 0 cold re-prefill!).
    if (out_thinking_mode == "thinking" &&
        prefix_base_effort >= 1 && prefix_base_effort <= 100 &&
        out_budget != prefix_base_effort && !merged.empty())
    {
        const std::string effort_tail =
            "Reasoning Effort: " + std::to_string(out_budget) +
            " (range 1-100, the higher the value, the more thorough the reasoning)";
        if (merged.back().get("role").as_str() == "latest_reminder") {
            bld.push_txt("\n" + effort_tail);
        } else {
            bld.push_tok(Dsv41SpecialTokens::kLatestReminder);
            bld.push_txt(effort_tail);
        }
    }

    return bld.segs;
}

namespace {

size_t find_dsml_variant(
    const std::string& text,
    size_t pos,
    std::initializer_list<const char*> suffixes,
    bool closing,
    size_t* matched_len = nullptr)
{
    size_t best_pos = std::string::npos;
    size_t best_len = 0;
    const std::string prefix = std::string(closing ? "</" : "<") + kDsmlTok;
    for (const char* suf : suffixes) {
        const std::string pat = prefix + suf;
        const size_t p = text.find(pat, pos);
        if (p != std::string::npos && (best_pos == std::string::npos || p < best_pos)) {
            best_pos = p;
            best_len = pat.size();
        }
    }
    if (matched_len) *matched_len = best_len;
    return best_pos;
}

std::string decode_dsml_invoke_body_to_json(const std::string& text, size_t pcur, size_t inv_limit) {
    std::string args_json = "{";
    bool first_param = true;
    while (pcur < inv_limit) {
        size_t po_len = 0;
        const size_t po = find_dsml_variant(text, pcur, {" parameter", "parameter"}, false, &po_len);
        if (po == std::string::npos || po >= inv_limit) break;

        const size_t tag_gt = text.find('>', po + po_len);
        if (tag_gt == std::string::npos || tag_gt >= inv_limit) break;

        const size_t pk0 = text.find("name=\"", po + po_len);
        if (pk0 == std::string::npos || pk0 >= tag_gt) break;
        const size_t pk1 = text.find('"', pk0 + 6);
        if (pk1 == std::string::npos || pk1 >= tag_gt) break;
        const std::string p_name = text.substr(pk0 + 6, pk1 - (pk0 + 6));

        std::string p_is_str = "true";
        const size_t ps0 = text.find("string=\"", po + po_len);
        if (ps0 != std::string::npos && ps0 < tag_gt) {
            const size_t ps1 = text.find('"', ps0 + 8);
            if (ps1 != std::string::npos && ps1 < tag_gt) {
                p_is_str = text.substr(ps0 + 8, ps1 - (ps0 + 8));
            }
        }

        const size_t val_start = tag_gt + 1;
        size_t pc_len = 0;
        const size_t pc = find_dsml_variant(text, val_start, {" parameter>", "parameter>"}, true, &pc_len);
        if (pc == std::string::npos || pc > inv_limit) break;
        const std::string p_val = text.substr(val_start, pc - val_start);

        if (!first_param) args_json += ", ";
        first_param = false;
        JsonValue::append_escaped_str(args_json, p_name);
        args_json += ": ";
        if (p_is_str == "true") {
            JsonValue::append_escaped_str(args_json, p_val);
        } else {
            JsonValue tmp;
            if (JsonParser::try_parse(p_val, tmp)) {
                args_json += p_val;
            } else {
                JsonValue::append_escaped_str(args_json, p_val);
            }
        }
        pcur = pc + pc_len;
    }
    args_json += "}";
    return args_json;
}

}  // namespace

ThinkEndClass classify_think_end(const std::string& text, size_t te, bool is_eof) {
    const size_t tag_len = std::strlen(kThinkEnd);
    const size_t after_pos = te + tag_len;

    // 1. Locate start of current line before `te`
    size_t line_start = te;
    while (line_start > 0 && text[line_start - 1] != '\n') {
        --line_start;
    }

    // 2. Check if `te` is inside an unclosed Markdown fenced code block (``` or ~~~) within <think>
    bool in_fence = false;
    char fence_ch = 0;
    size_t scan = 0;
    while (scan < line_start) {
        const size_t ls = scan;
        size_t le = text.find('\n', ls);
        if (le == std::string::npos || le >= line_start) le = line_start;
        size_t p = ls;
        while (p < le && (text[p] == ' ' || text[p] == '\t')) ++p;
        if (p + 3 <= le &&
            ((text[p] == '`' && text[p + 1] == '`' && text[p + 2] == '`') ||
             (text[p] == '~' && text[p + 1] == '~' && text[p + 2] == '~'))) {
            const char ch = text[p];
            if (!in_fence) {
                in_fence = true;
                fence_ch = ch;
            } else if (ch == fence_ch) {
                size_t q = p + 3;
                while (q < le && text[q] == ch) ++q;
                while (q < le && (text[q] == ' ' || text[q] == '\t' || text[q] == '\r')) ++q;
                if (q == le) in_fence = false;
            }
        }
        scan = (le < line_start) ? (le + 1) : line_start;
    }
    if (in_fence) {
        return ThinkEndClass::InlineMention;
    }

    // 3. Inspect same-line prefix `text[line_start .. te)` for unclosed quotes/backticks or code escapes
    int backtick_cnt = 0;
    int dquote_cnt = 0;
    bool has_non_ws = false;
    for (size_t i = line_start; i < te; ++i) {
        const char c = text[i];
        if (c != ' ' && c != '\t' && c != '\r') has_non_ws = true;
        if (c == '\\' && i + 1 < te) {
            ++i;
            continue;
        }
        if (c == '`') ++backtick_cnt;
        else if (c == '"') ++dquote_cnt;
    }
    if ((backtick_cnt & 1) != 0 || (dquote_cnt & 1) != 0) {
        return ThinkEndClass::InlineMention;
    }

    if (te > line_start) {
        const char prev = text[te - 1];
        if (prev == '\\') {
            return ThinkEndClass::InlineMention;
        }
        if ((prev == 'n' || prev == 'r' || prev == 't') && te >= line_start + 2 && text[te - 2] == '\\') {
            return ThinkEndClass::InlineMention;
        }
        if (prev == '-' || prev == '=' || prev == '(' || prev == '[' || prev == '{' ||
            prev == '<' || prev == '/' || prev == '+' || prev == '|' || prev == '_') {
            return ThinkEndClass::InlineMention;
        }
    }

    // 4. Require >= 1 byte lookahead after </think> unless at EOF
    if (after_pos >= text.size()) {
        return is_eof ? ThinkEndClass::RealClose : ThinkEndClass::NeedMoreLookahead;
    }

    const std::string eos(kEosTok);
    const size_t rem = text.size() - after_pos;
    const size_t cmp_n = std::min(eos.size(), rem);
    if (text.compare(after_pos, cmp_n, eos, 0, cmp_n) == 0) {
        if (rem < eos.size() && !is_eof) {
            return ThinkEndClass::NeedMoreLookahead;
        }
        return ThinkEndClass::RealClose;
    }

    const char next = text[after_pos];
    if (next == ')' || next == ']' || next == '}' || next == ',' || next == ';') {
        return ThinkEndClass::InlineMention;
    }
    if (next == '\'' && te > line_start && text[te - 1] == '\'') {
        return ThinkEndClass::InlineMention;
    }
    if (has_non_ws && te > line_start) {
        const char prev = text[te - 1];
        if ((prev == ' ' || prev == '\t') &&
            (next == ' ' || next == '\t' || next == '.' || next == ':' || next == '?' || next == '!')) {
            return ThinkEndClass::InlineMention;
        }
    }

    return ThinkEndClass::RealClose;
}

Dsv41ParsedCompletion parse_dsv41_completion(
    const std::string& raw_text,
    const std::string& thinking_mode,
    uint64_t call_id_seed,
    size_t forced_think_end_pos)
{
    Dsv41ParsedCompletion out;
    std::string text = raw_text;

    const std::string eos(kEosTok);
    if (text.size() >= eos.size() && text.compare(text.size() - eos.size(), eos.size(), eos) == 0) {
        text.erase(text.size() - eos.size());
    }

    size_t pos = 0;
    const size_t tag_len = std::strlen(kThinkEnd);
    if (thinking_mode == "thinking") {
        size_t real_te = std::string::npos;
        if (forced_think_end_pos != std::string::npos &&
            forced_think_end_pos + tag_len <= text.size() &&
            text.compare(forced_think_end_pos, tag_len, kThinkEnd) == 0) {
            real_te = forced_think_end_pos;
        } else {
            size_t search_from = pos;
            while (search_from < text.size()) {
                const size_t cand = text.find(kThinkEnd, search_from);
                if (cand == std::string::npos) break;
                if (classify_think_end(text, cand, true) == ThinkEndClass::RealClose) {
                    real_te = cand;
                    break;
                }
                search_from = cand + tag_len;
            }
        }
        if (real_te != std::string::npos) {
            out.reasoning_content = text.substr(pos, real_te - pos);
            pos = real_te + tag_len;
        } else {
            out.reasoning_content = text.substr(pos);
            return out;
        }
    }

    size_t open_len = 0;
    size_t cs = find_dsml_variant(
        text, pos, {" calls>", "calls>", "function_calls>", " function_calls>"}, false, &open_len);
    if (cs == std::string::npos) {
        // Also recover bare <｜DSML｜ invoke...> if model omitted outer wrapper
        cs = find_dsml_variant(text, pos, {" invoke", "invoke"}, false, nullptr);
        open_len = 0;
    }
    if (cs == std::string::npos) {
        // Quarantine orphan bare <｜DSML｜ parameter...> arriving without an enclosing invoke block
        const size_t ps = find_dsml_variant(text, pos, {" parameter", "parameter"}, false, nullptr);
        if (ps != std::string::npos) {
            out.content = text.substr(pos, ps - pos);
            if (out.content.size() >= 2 && out.content.compare(out.content.size() - 2, 2, "\n\n") == 0) {
                out.content.erase(out.content.size() - 2);
            } else if (!out.content.empty() && out.content.back() == '\n') {
                out.content.pop_back();
            }
            return out;
        }
        out.content = text.substr(pos);
        return out;
    }

    out.content = text.substr(pos, cs - pos);
    if (out.content.size() >= 2 && out.content.compare(out.content.size() - 2, 2, "\n\n") == 0) {
        out.content.erase(out.content.size() - 2);
    } else if (!out.content.empty() && out.content.back() == '\n') {
        out.content.pop_back();
    }

    size_t cur = cs + open_len;
    const size_t limit = text.size();

    uint64_t call_idx = 0;
    while (cur < limit) {
        size_t inv_open_len = 0;
        const size_t inv = find_dsml_variant(text, cur, {" invoke", "invoke"}, false, &inv_open_len);
        if (inv == std::string::npos || inv >= limit) break;
        const size_t inv_gt = text.find('>', inv + inv_open_len);
        if (inv_gt == std::string::npos || inv_gt >= limit) break;
        const size_t name_q0 = text.find("name=\"", inv + inv_open_len);
        if (name_q0 == std::string::npos || name_q0 >= inv_gt) break;
        const size_t name_q1 = text.find('"', name_q0 + 6);
        if (name_q1 == std::string::npos || name_q1 >= inv_gt) break;

        const std::string raw_tool_name = text.substr(name_q0 + 6, name_q1 - (name_q0 + 6));
        auto [ns, bare_name] = split_tool_name(raw_tool_name);
        (void)ns;

        size_t inv_close_len = 0;
        const size_t inv_end = find_dsml_variant(text, inv_gt + 1, {" invoke>", "invoke>"}, true, &inv_close_len);
        if (inv_end == std::string::npos || inv_end > limit) break;

        std::string args_json = decode_dsml_invoke_body_to_json(text, inv_gt + 1, inv_end);

        Dsv41ParsedToolCall tc;
        char id_buf[64];
        std::snprintf(id_buf, sizeof(id_buf), "call_dsv41_%llu_%llu",
                      static_cast<unsigned long long>(call_id_seed),
                      static_cast<unsigned long long>(call_idx++));
        tc.id = id_buf;
        tc.name = bare_name;
        tc.arguments_json = std::move(args_json);
        out.tool_calls.push_back(std::move(tc));

        cur = inv_end + inv_close_len;
        while (cur < limit && (text[cur] == ' ' || text[cur] == '\n' || text[cur] == '\r' || text[cur] == '\t')) ++cur;
        size_t next_inv_len = 0;
        const size_t next_inv = find_dsml_variant(text, cur, {" invoke", "invoke"}, false, &next_inv_len);
        if (next_inv != cur) break;
    }
    return out;
}

namespace {

void append_json_escaped_inner(std::string& out, std::string_view sv) {
    for (unsigned char c : sv) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20u) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", unsigned(c));
                    out += buf;
                } else {
                    out.push_back(char(c));
                }
                break;
        }
    }
}

std::string json_quote_str(std::string_view sv) {
    std::string out;
    out.reserve(sv.size() + 4);
    out.push_back('"');
    append_json_escaped_inner(out, sv);
    out.push_back('"');
    return out;
}

std::string pop_complete_utf8_bytes(std::string& pending_bytes) {
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
    if (check == 0) return {};
    std::string complete_raw = pending_bytes.substr(0, check);
    pending_bytes.erase(0, check);
    return bytes_to_utf8_lossy(complete_raw);
}

void push_assistant_seg_tok(std::vector<PromptSegment>& segs, int32_t id) {
    if (id < 0) return;
    if (!segs.empty() && segs.back().kind == PromptSegment::Kind::TokenId && segs.back().token_id == id) {
        segs.back().repeat += 1;
    } else {
        segs.push_back(PromptSegment::tok(id, 1));
    }
}

void push_assistant_seg_txt(std::vector<PromptSegment>& segs, std::string_view sv) {
    if (sv.empty()) return;
    if (!segs.empty() && segs.back().kind == PromptSegment::Kind::OrdinaryText) {
        segs.back().text.append(sv.data(), sv.size());
    } else {
        segs.push_back(PromptSegment::txt(std::string(sv)));
    }
}

struct DsmlOpenKmpSpec {
    static constexpr int kNumPats = 8;
    const char* pats[kNumPats] = {
        " calls>",
        "calls>",
        " function_calls>",
        "function_calls>",
        " invoke",
        "invoke",
        " parameter",
        "parameter"
    };
    bool is_wrapper[kNumPats] = {true, true, true, true, false, false, false, false};
    bool is_orphan_param[kNumPats] = {false, false, false, false, false, false, true, true};
    int lens[kNumPats] = {};
    int pi[kNumPats][24] = {};

    DsmlOpenKmpSpec() {
        for (int p = 0; p < kNumPats; ++p) {
            const int m = static_cast<int>(std::strlen(pats[p]));
            lens[p] = m;
            pi[p][0] = 0;
            int k = 0;
            for (int i = 1; i < m; ++i) {
                while (k > 0 && pats[p][k] != pats[p][i]) {
                    k = pi[p][k - 1];
                }
                if (pats[p][k] == pats[p][i]) ++k;
                pi[p][i] = k;
            }
        }
    }

    int step(int p, int q, char ch) const {
        while (q > 0 && (q >= lens[p] || pats[p][q] != ch)) {
            q = pi[p][q - 1];
        }
        if (q < lens[p] && pats[p][q] == ch) ++q;
        return q;
    }
};

const DsmlOpenKmpSpec& dsml_open_kmp() {
    static const DsmlOpenKmpSpec kSpec;
    return kSpec;
}

std::string trim_leading_ascii_ws(std::string_view sv) {
    size_t i = 0;
    while (i < sv.size() && (sv[i] == ' ' || sv[i] == '\n' || sv[i] == '\r' || sv[i] == '\t')) ++i;
    return std::string(sv.substr(i));
}

}  // namespace

Dsv41StreamParser::Dsv41StreamParser(const std::string& thinking_mode)
    : thinking_mode_(thinking_mode),
      state_(thinking_mode == "thinking" ? State::InThinking : State::InContent) {}

void Dsv41StreamParser::step_token_gated_dsml_bytes(
    std::string_view utf8_chunk,
    std::vector<Dsv41ToolStreamDelta>* out_tool_deltas,
    uint64_t call_id_seed)
{
    static constexpr size_t kMaxSingleToolBytes = 196608ULL;  // 192 KiB runaway guard per invoke
    if (utf8_chunk.empty() || tool_block_closed_ || tool_runaway_) return;

    std::string work(utf8_chunk);
    while (!work.empty() && !tool_block_closed_ && !tool_runaway_) {
        if (dsml_sub_ == DsmlSub::BetweenInvokes) {
            dsml_tag_buf_ += work;
            work.clear();
            const std::string trimmed = trim_leading_ascii_ws(dsml_tag_buf_);
            if (trimmed.empty() || trimmed == "<" || trimmed == "</") {
                return;
            }
            if (next_tool_idx_ > 0) {
                tool_block_closed_ = true;
            }
            return;
        }

        if (dsml_sub_ == DsmlSub::ReadingInvokeHeader) {
            dsml_tag_buf_ += work;
            work.clear();
            const size_t gt = dsml_tag_buf_.find('>');
            if (gt == std::string::npos) {
                if (dsml_tag_buf_.size() > 4096u) tool_runaway_ = true;
                return;
            }
            const std::string hdr = dsml_tag_buf_.substr(0, gt);
            work = dsml_tag_buf_.substr(gt + 1);
            dsml_tag_buf_.clear();
            if (trim_leading_ascii_ws(hdr).rfind("invoke", 0) != 0) {
                tool_block_closed_ = true;
                return;
            }

            const size_t nq0 = hdr.find("name=\"");
            if (nq0 == std::string::npos) {
                dsml_sub_ = DsmlSub::BetweenInvokes;
                continue;
            }
            const size_t nq1 = hdr.find('"', nq0 + 6);
            if (nq1 == std::string::npos) {
                dsml_sub_ = DsmlSub::BetweenInvokes;
                continue;
            }
            auto [ns, bare_name] = split_tool_name(hdr.substr(nq0 + 6, nq1 - (nq0 + 6)));
            (void)ns;

            char id_buf[64];
            std::snprintf(id_buf, sizeof(id_buf), "call_dsv41_%llu_%llu",
                          static_cast<unsigned long long>(call_id_seed),
                          static_cast<unsigned long long>(next_tool_idx_));
            cur_tool_id_ = id_buf;
            cur_tool_name_ = bare_name;
            cur_tool_json_acc_.clear();
            cur_tool_param_count_ = 0;
            open_invoke_started_ = true;
            open_invoke_name_ = bare_name;
            tool_bytes_ = 0;
            dsml_sub_ = DsmlSub::BetweenParams;

            if (out_tool_deltas) {
                out_tool_deltas->push_back(Dsv41ToolStreamDelta{
                    next_tool_idx_, cur_tool_id_, cur_tool_name_, "", /*is_start=*/true, /*is_end=*/false});
            }
            continue;
        }

        if (dsml_sub_ == DsmlSub::BetweenParams) {
            dsml_tag_buf_ += work;
            work.clear();
            return;
        }

        if (dsml_sub_ == DsmlSub::ReadingParamHeader) {
            dsml_tag_buf_ += work;
            work.clear();
            const size_t gt = dsml_tag_buf_.find('>');
            if (gt == std::string::npos) {
                if (dsml_tag_buf_.size() > 4096u) tool_runaway_ = true;
                return;
            }
            const std::string hdr = dsml_tag_buf_.substr(0, gt);
            work = dsml_tag_buf_.substr(gt + 1);
            dsml_tag_buf_.clear();

            cur_param_name_ = "value";
            const size_t pk0 = hdr.find("name=\"");
            if (pk0 != std::string::npos) {
                const size_t pk1 = hdr.find('"', pk0 + 6);
                if (pk1 != std::string::npos) {
                    cur_param_name_ = hdr.substr(pk0 + 6, pk1 - (pk0 + 6));
                }
            }
            cur_param_is_string_ = true;
            const size_t ps0 = hdr.find("string=\"");
            if (ps0 != std::string::npos) {
                const size_t ps1 = hdr.find('"', ps0 + 8);
                if (ps1 != std::string::npos) {
                    cur_param_is_string_ = (hdr.substr(ps0 + 8, ps1 - (ps0 + 8)) != "false");
                }
            }
            param_val_pending_.clear();

            std::string prefix = (cur_tool_param_count_ == 0) ? "{" : ", ";
            prefix += json_quote_str(cur_param_name_);
            prefix += cur_param_is_string_ ? ": \"" : ": ";
            cur_tool_json_acc_ += prefix;
            if (out_tool_deltas) {
                out_tool_deltas->push_back(Dsv41ToolStreamDelta{
                    next_tool_idx_, cur_tool_id_, cur_tool_name_, std::move(prefix), false, false});
            }
            dsml_sub_ = cur_param_is_string_ ? DsmlSub::StreamingStringParamValue : DsmlSub::BufferingJsonParamValue;
            continue;
        }

        if (dsml_sub_ == DsmlSub::StreamingStringParamValue) {
            tool_bytes_ += work.size();
            if (tool_bytes_ > kMaxSingleToolBytes) {
                tool_runaway_ = true;
                return;
            }
            param_val_pending_ += work;
            work.clear();

            size_t hold = 0;
            if (param_val_pending_.size() >= 2 &&
                param_val_pending_[param_val_pending_.size() - 2] == '<' &&
                param_val_pending_[param_val_pending_.size() - 1] == '/') {
                hold = 2;
            } else if (!param_val_pending_.empty() && param_val_pending_.back() == '<') {
                hold = 1;
            }
            const size_t emit_n = param_val_pending_.size() - hold;
            if (emit_n > 0) {
                std::string esc;
                esc.reserve(emit_n + 8);
                append_json_escaped_inner(esc, std::string_view(param_val_pending_.data(), emit_n));
                param_val_pending_.erase(0, emit_n);
                cur_tool_json_acc_ += esc;
                if (out_tool_deltas && !esc.empty()) {
                    out_tool_deltas->push_back(Dsv41ToolStreamDelta{
                        next_tool_idx_, cur_tool_id_, cur_tool_name_, std::move(esc), false, false});
                }
            }
            return;
        }

        if (dsml_sub_ == DsmlSub::BufferingJsonParamValue) {
            tool_bytes_ += work.size();
            if (tool_bytes_ > kMaxSingleToolBytes) {
                tool_runaway_ = true;
                return;
            }
            param_val_pending_ += work;
            work.clear();
            return;
        }

        if (dsml_sub_ == DsmlSub::ReadingCloseTag) {
            dsml_tag_buf_ += work;
            work.clear();
            const size_t gt = dsml_tag_buf_.find('>');
            if (gt == std::string::npos) {
                if (dsml_tag_buf_.size() > 512u) tool_block_closed_ = true;
                return;
            }
            work = dsml_tag_buf_.substr(gt + 1);
            dsml_tag_buf_.clear();

            if (close_is_wrapper_) {
                tool_block_closed_ = true;
                return;
            }
            if (close_is_invoke_) {
                std::string close_json = (cur_tool_param_count_ == 0) ? "{}" : "}";
                cur_tool_json_acc_ += close_json;
                parsed_tools_.push_back(Dsv41ParsedToolCall{cur_tool_id_, cur_tool_name_, cur_tool_json_acc_});
                if (out_tool_deltas) {
                    out_tool_deltas->push_back(Dsv41ToolStreamDelta{
                        next_tool_idx_, cur_tool_id_, cur_tool_name_, std::move(close_json), false, /*is_end=*/true});
                }
                open_invoke_started_ = false;
                tool_bytes_ = 0;
                ++next_tool_idx_;
                dsml_sub_ = DsmlSub::BetweenInvokes;
            } else {
                dsml_sub_ = DsmlSub::BetweenParams;
            }
            continue;
        }
    }
}

void Dsv41StreamParser::on_token_gated_dsml_marker(
    std::string& out_content_delta,
    std::vector<Dsv41ToolStreamDelta>* out_tool_deltas,
    uint64_t /*call_id_seed*/)
{
    if (state_ == State::InContent) {
        if (!pending_.empty() && pending_.back() == '<') {
            pending_.pop_back();
            dsml_pre_ws_.clear();
            if (pending_.size() >= 2 && pending_.compare(pending_.size() - 2, 2, "\n\n") == 0) {
                dsml_pre_ws_ = "\n\n";
                pending_.erase(pending_.size() - 2);
            } else if (!pending_.empty() && pending_.back() == '\n') {
                dsml_pre_ws_ = "\n";
                pending_.pop_back();
            }
            if (!pending_.empty()) {
                out_content_delta += pending_;
                content_acc_ += pending_;
                pending_.clear();
            }
            dsml_open_buf_.clear();
            for (int& q : kmp_q_) q = 0;
            state_ = State::ConfirmingDsmlOpen;
        } else {
            pending_ += kDsmlTok;
            out_content_delta += pending_;
            content_acc_ += pending_;
            pending_.clear();
        }
        return;
    }

    if (state_ == State::ConfirmingDsmlOpen) {
        // Unexpected back-to-back ｜DSML｜ inside unconfirmed header -> flush literal
        std::string restored = dsml_pre_ws_ + "<" + kDsmlTok + dsml_open_buf_ + kDsmlTok;
        dsml_pre_ws_.clear();
        dsml_open_buf_.clear();
        state_ = State::InContent;
        out_content_delta += restored;
        content_acc_ += restored;
        return;
    }

    if (state_ != State::InToolCalls || tool_block_closed_ || tool_runaway_) return;

    if (dsml_sub_ == DsmlSub::BetweenInvokes) {
        const std::string trimmed = trim_leading_ascii_ws(dsml_tag_buf_);
        dsml_tag_buf_.clear();
        if (trimmed == "<") {
            dsml_sub_ = DsmlSub::ReadingInvokeHeader;
        } else if (trimmed == "</") {
            dsml_sub_ = DsmlSub::ReadingCloseTag;
            close_is_invoke_ = false;
            close_is_wrapper_ = true;
        } else {
            tool_block_closed_ = true;
        }
        return;
    }

    if (dsml_sub_ == DsmlSub::BetweenParams) {
        const std::string trimmed = trim_leading_ascii_ws(dsml_tag_buf_);
        dsml_tag_buf_.clear();
        if (trimmed == "<") {
            dsml_sub_ = DsmlSub::ReadingParamHeader;
        } else if (trimmed == "</") {
            dsml_sub_ = DsmlSub::ReadingCloseTag;
            close_is_invoke_ = true;
            close_is_wrapper_ = false;
        } else {
            tool_block_closed_ = true;
        }
        return;
    }

    if (dsml_sub_ == DsmlSub::StreamingStringParamValue) {
        if (param_val_pending_.size() >= 2 &&
            param_val_pending_[param_val_pending_.size() - 2] == '<' &&
            param_val_pending_[param_val_pending_.size() - 1] == '/') {
            param_val_pending_.erase(param_val_pending_.size() - 2);
            std::string delta;
            if (!param_val_pending_.empty()) {
                append_json_escaped_inner(delta, param_val_pending_);
                param_val_pending_.clear();
            }
            delta.push_back('"');
            cur_tool_json_acc_ += delta;
            if (out_tool_deltas) {
                out_tool_deltas->push_back(Dsv41ToolStreamDelta{
                    next_tool_idx_, cur_tool_id_, cur_tool_name_, std::move(delta), false, false});
            }
            ++cur_tool_param_count_;
            dsml_tag_buf_.clear();
            dsml_sub_ = DsmlSub::ReadingCloseTag;
            close_is_invoke_ = false;
            close_is_wrapper_ = false;
        } else {
            param_val_pending_ += kDsmlTok;
            std::string esc;
            append_json_escaped_inner(esc, param_val_pending_);
            param_val_pending_.clear();
            cur_tool_json_acc_ += esc;
            if (out_tool_deltas && !esc.empty()) {
                out_tool_deltas->push_back(Dsv41ToolStreamDelta{
                    next_tool_idx_, cur_tool_id_, cur_tool_name_, std::move(esc), false, false});
            }
        }
        return;
    }

    if (dsml_sub_ == DsmlSub::BufferingJsonParamValue) {
        if (param_val_pending_.size() >= 2 &&
            param_val_pending_[param_val_pending_.size() - 2] == '<' &&
            param_val_pending_[param_val_pending_.size() - 1] == '/') {
            param_val_pending_.erase(param_val_pending_.size() - 2);
            JsonValue tmp;
            std::string val_json;
            if (JsonParser::try_parse(param_val_pending_, tmp)) {
                val_json = param_val_pending_;
            } else {
                val_json = json_quote_str(param_val_pending_);
            }
            param_val_pending_.clear();
            cur_tool_json_acc_ += val_json;
            if (out_tool_deltas) {
                out_tool_deltas->push_back(Dsv41ToolStreamDelta{
                    next_tool_idx_, cur_tool_id_, cur_tool_name_, std::move(val_json), false, false});
            }
            ++cur_tool_param_count_;
            dsml_tag_buf_.clear();
            dsml_sub_ = DsmlSub::ReadingCloseTag;
            close_is_invoke_ = false;
            close_is_wrapper_ = false;
        } else {
            param_val_pending_ += kDsmlTok;
        }
        return;
    }
}

void Dsv41StreamParser::feed_token(
    int32_t token_id,
    std::string_view raw_piece,
    std::string& out_reasoning_delta,
    std::string& out_content_delta,
    std::vector<Dsv41ToolStreamDelta>* out_tool_deltas,
    uint64_t call_id_seed)
{
    out_reasoning_delta.clear();
    out_content_delta.clear();
    if (out_tool_deltas) out_tool_deltas->clear();
    token_gated_mode_ = true;

    if (token_id == Dsv41SpecialTokens::kEos) {
        if (state_ == State::InToolCalls) {
            tool_block_closed_ = true;
        }
        return;
    }

    // Flush any pending multi-byte UTF-8 tail before structural special tokens
    auto flush_incomplete_utf8 = [&]() -> std::string {
        if (utf8_tail_.empty()) return {};
        std::string flushed = bytes_to_utf8_lossy(utf8_tail_);
        utf8_tail_.clear();
        push_assistant_seg_txt(assistant_segs_, flushed);
        full_raw_ += flushed;
        return flushed;
    };

    // 1. Strict Token-ID-gated `</think>` exit (ordinary BPE spelling "</think>" NEVER exits <think>!)
    if (token_id == Dsv41SpecialTokens::kThinkEnd) {
        const std::string tail = flush_incomplete_utf8();
        push_assistant_seg_tok(assistant_segs_, Dsv41SpecialTokens::kThinkEnd);
        if (state_ == State::InThinking) {
            if (!tail.empty()) {
                out_reasoning_delta += tail;
                reasoning_acc_ += tail;
            }
            if (!pending_.empty()) {
                out_reasoning_delta += pending_;
                reasoning_acc_ += pending_;
                pending_.clear();
            }
            has_real_think_end_ = true;
            real_think_end_pos_ = full_raw_.size();
            full_raw_ += kThinkEnd;
            state_ = State::InContent;
            return;
        }
        // After </think>, a literal 128822 token behaves as literal "</think>"
        full_raw_ += kThinkEnd;
        if (state_ == State::InToolCalls) {
            step_token_gated_dsml_bytes(kThinkEnd, out_tool_deltas, call_id_seed);
            return;
        }
        pending_ += tail;
        pending_ += kThinkEnd;
    } else if (token_id == Dsv41SpecialTokens::kDsml) {
        // 2. Strict Token-ID-gated `｜DSML｜` marker (ordinary BPE spelling "｜DSML｜" NEVER triggers tool mode!)
        const std::string tail = flush_incomplete_utf8();
        push_assistant_seg_tok(assistant_segs_, Dsv41SpecialTokens::kDsml);
        full_raw_ += kDsmlTok;
        if (state_ == State::InThinking) {
            if (!tail.empty()) {
                out_reasoning_delta += tail;
                reasoning_acc_ += tail;
            }
            out_reasoning_delta += kDsmlTok;
            reasoning_acc_ += kDsmlTok;
            return;
        }
        if (!tail.empty()) {
            if (state_ == State::InToolCalls) {
                step_token_gated_dsml_bytes(tail, out_tool_deltas, call_id_seed);
            } else {
                pending_ += tail;
            }
        }
        on_token_gated_dsml_marker(out_content_delta, out_tool_deltas, call_id_seed);
        return;
    } else if (token_id == Dsv41SpecialTokens::kThinkOpen ||
               token_id == Dsv41SpecialTokens::kSystem ||
               token_id == Dsv41SpecialTokens::kUser ||
               token_id == Dsv41SpecialTokens::kAssistant ||
               token_id == Dsv41SpecialTokens::kLatestReminder ||
               token_id == Dsv41SpecialTokens::kImage ||
               token_id == Dsv41SpecialTokens::kBos) {
        const std::string tail = flush_incomplete_utf8();
        push_assistant_seg_tok(assistant_segs_, token_id);
        full_raw_.append(raw_piece.data(), raw_piece.size());
        if (state_ == State::InThinking) {
            if (!tail.empty()) {
                out_reasoning_delta += tail;
                reasoning_acc_ += tail;
            }
            out_reasoning_delta.append(raw_piece.data(), raw_piece.size());
            reasoning_acc_.append(raw_piece.data(), raw_piece.size());
            return;
        }
        if (state_ == State::InToolCalls) {
            if (!tail.empty()) step_token_gated_dsml_bytes(tail, out_tool_deltas, call_id_seed);
            step_token_gated_dsml_bytes(raw_piece, out_tool_deltas, call_id_seed);
            return;
        }
        pending_ += tail;
        pending_.append(raw_piece.data(), raw_piece.size());
    } else {
        if (raw_piece.empty()) return;
        utf8_tail_.append(raw_piece.data(), raw_piece.size());
        const std::string utf8_chunk = pop_complete_utf8_bytes(utf8_tail_);
        if (utf8_chunk.empty()) return;

        push_assistant_seg_txt(assistant_segs_, utf8_chunk);
        full_raw_ += utf8_chunk;

        if (state_ == State::InThinking) {
            out_reasoning_delta += utf8_chunk;
            reasoning_acc_ += utf8_chunk;
            return;
        }

        if (state_ == State::ConfirmingDsmlOpen) {
            const auto& kmp = dsml_open_kmp();
            for (size_t idx = 0; idx < utf8_chunk.size(); ++idx) {
                const char ch = utf8_chunk[idx];
                dsml_open_buf_.push_back(ch);
                const int cur_len = static_cast<int>(dsml_open_buf_.size());
                int matched_pat = -1;
                bool any_prefix = false;
                for (int p = 0; p < DsmlOpenKmpSpec::kNumPats; ++p) {
                    kmp_q_[p] = kmp.step(p, kmp_q_[p], ch);
                    if (kmp_q_[p] == cur_len) {
                        any_prefix = true;
                        if (cur_len == kmp.lens[p]) {
                            matched_pat = p;
                            break;
                        }
                    }
                }
                if (matched_pat >= 0) {
                    const std::string_view rem(utf8_chunk.data() + idx + 1, utf8_chunk.size() - (idx + 1));
                    state_ = State::InToolCalls;
                    if (kmp.is_orphan_param[matched_pat]) {
                        tool_block_closed_ = true;
                        dsml_tag_buf_.clear();
                    } else if (kmp.is_wrapper[matched_pat]) {
                        dsml_sub_ = DsmlSub::BetweenInvokes;
                        dsml_tag_buf_.clear();
                    } else {
                        dsml_sub_ = DsmlSub::ReadingInvokeHeader;
                        dsml_tag_buf_ = dsml_open_buf_;
                    }
                    dsml_pre_ws_.clear();
                    dsml_open_buf_.clear();
                    if (!rem.empty() && !tool_block_closed_) {
                        step_token_gated_dsml_bytes(rem, out_tool_deltas, call_id_seed);
                    }
                    return;
                }
                if (!any_prefix) {
                    // False alarm after `<｜DSML｜`: restore held bytes cleanly into content!
                    pending_ += dsml_pre_ws_;
                    pending_ += "<";
                    pending_ += kDsmlTok;
                    pending_ += dsml_open_buf_;
                    if (idx + 1 < utf8_chunk.size()) {
                        pending_.append(utf8_chunk.data() + idx + 1, utf8_chunk.size() - (idx + 1));
                    }
                    dsml_pre_ws_.clear();
                    dsml_open_buf_.clear();
                    state_ = State::InContent;
                    break;
                }
            }
            if (state_ == State::ConfirmingDsmlOpen) return;
        } else if (state_ == State::InToolCalls) {
            step_token_gated_dsml_bytes(utf8_chunk, out_tool_deltas, call_id_seed);
            return;
        } else {
            pending_ += utf8_chunk;
        }
    }

    // In `State::InContent`, hold at most 1..3 trailing bytes matching a prefix of {"\n\n<", "\n<", "<"}
    if (state_ == State::InContent && !pending_.empty()) {
        size_t hold = 0;
        const size_t n = pending_.size();
        if (n >= 3 && pending_.compare(n - 3, 3, "\n\n<") == 0) {
            hold = 3;
        } else if (n >= 2 && (pending_.compare(n - 2, 2, "\n\n") == 0 || pending_.compare(n - 2, 2, "\n<") == 0)) {
            hold = 2;
        } else if (pending_.back() == '\n' || pending_.back() == '<') {
            hold = 1;
        }
        const size_t emit_n = n - hold;
        if (emit_n > 0) {
            out_content_delta.append(pending_.data(), emit_n);
            content_acc_.append(pending_.data(), emit_n);
            pending_.erase(0, emit_n);
        }
    }
}

void Dsv41StreamParser::scan_incremental_tools(
    std::vector<Dsv41ToolStreamDelta>* out_tool_deltas,
    uint64_t call_id_seed)
{
    static constexpr size_t kMaxSingleToolBytes = 196608ULL;  // 192 KiB runaway guard per buffered tool invoke
    const std::string eos(kEosTok);

    if (tool_bytes_ > kMaxSingleToolBytes) {
        tool_runaway_ = true;
        return;
    }

    while (tool_scan_pos_ < full_raw_.size() && !tool_block_closed_) {
        if (!open_invoke_started_) {
            size_t ws = tool_scan_pos_;
            while (ws < full_raw_.size() &&
                   (full_raw_[ws] == ' ' || full_raw_[ws] == '\n' || full_raw_[ws] == '\r' || full_raw_[ws] == '\t')) {
                ++ws;
            }
            if (ws >= full_raw_.size()) return;

            if (next_tool_idx_ == 0) {
                size_t open_wrap_len = 0;
                const size_t ow = find_dsml_variant(
                    full_raw_, ws, {" calls>", "calls>", "function_calls>", " function_calls>"}, false, &open_wrap_len);
                if (ow == ws) {
                    tool_scan_pos_ = ws + open_wrap_len;
                    continue;
                }
            }

            size_t close_len = 0;
            const size_t bc = find_dsml_variant(
                full_raw_, ws, {" calls>", "calls>", "function_calls>", " function_calls>", " tool_calls>", "tool_calls>"}, true, &close_len);
            if (bc == ws ||
                (ws + eos.size() <= full_raw_.size() && full_raw_.compare(ws, eos.size(), eos) == 0) ||
                full_raw_.find(eos, ws) != std::string::npos) {
                tool_block_closed_ = true;
                return;
            }

            if (next_tool_idx_ > 0) {
                const std::string p_inv1 = std::string("<") + kDsmlTok + " invoke";
                const std::string p_inv2 = std::string("<") + kDsmlTok + "invoke";
                const size_t rem = full_raw_.size() - ws;
                const bool pref1 = (rem < p_inv1.size()) && (p_inv1.compare(0, rem, full_raw_, ws, rem) == 0);
                const bool pref2 = (rem < p_inv2.size()) && (p_inv2.compare(0, rem, full_raw_, ws, rem) == 0);
                if (pref1 || pref2) return;

                const bool match1 = (rem >= p_inv1.size()) && (full_raw_.compare(ws, p_inv1.size(), p_inv1) == 0);
                const bool match2 = (rem >= p_inv2.size()) && (full_raw_.compare(ws, p_inv2.size(), p_inv2) == 0);
                if (!match1 && !match2) {
                    if (rem >= 2 && full_raw_[ws] == '<' && full_raw_[ws + 1] == '/') {
                        if (full_raw_.find('>', ws + 2) == std::string::npos && rem < 64) {
                            return;
                        }
                    } else if (rem == 1 && full_raw_[ws] == '<') {
                        return;
                    }
                    tool_block_closed_ = true;
                    return;
                }
            }

            size_t inv_open_len = 0;
            const size_t inv = find_dsml_variant(full_raw_, ws, {" invoke", "invoke"}, false, &inv_open_len);
            if (inv == std::string::npos) return;
            const size_t inv_gt = full_raw_.find('>', inv + inv_open_len);
            if (inv_gt == std::string::npos) return;
            const size_t name_q0 = full_raw_.find("name=\"", inv + inv_open_len);
            if (name_q0 == std::string::npos || name_q0 >= inv_gt) {
                tool_scan_pos_ = inv_gt + 1;
                continue;
            }
            const size_t name_q1 = full_raw_.find('"', name_q0 + 6);
            if (name_q1 == std::string::npos || name_q1 >= inv_gt) {
                tool_scan_pos_ = inv_gt + 1;
                continue;
            }
            auto [ns, bare_name] = split_tool_name(full_raw_.substr(name_q0 + 6, name_q1 - (name_q0 + 6)));
            (void)ns;
            open_invoke_started_ = true;
            open_invoke_name_ = bare_name;
            open_invoke_body_pos_ = inv_gt + 1;
            tool_scan_pos_ = open_invoke_body_pos_;

            if (out_tool_deltas) {
                char id_buf[64];
                std::snprintf(id_buf, sizeof(id_buf), "call_dsv41_%llu_%llu",
                              static_cast<unsigned long long>(call_id_seed),
                              static_cast<unsigned long long>(next_tool_idx_));
                out_tool_deltas->push_back(Dsv41ToolStreamDelta{next_tool_idx_, id_buf, bare_name, "", true, false});
            }
        } else {
            size_t inv_close_len = 0;
            const size_t inv_end = find_dsml_variant(
                full_raw_, open_invoke_body_pos_, {" invoke>", "invoke>"}, true, &inv_close_len);
            if (inv_end == std::string::npos) {
                if (full_raw_.size() - open_invoke_body_pos_ > kMaxSingleToolBytes) {
                    tool_runaway_ = true;
                }
                return;
            }
            if (out_tool_deltas) {
                char id_buf[64];
                std::snprintf(id_buf, sizeof(id_buf), "call_dsv41_%llu_%llu",
                              static_cast<unsigned long long>(call_id_seed),
                              static_cast<unsigned long long>(next_tool_idx_));
                std::string args_json = decode_dsml_invoke_body_to_json(full_raw_, open_invoke_body_pos_, inv_end);
                out_tool_deltas->push_back(Dsv41ToolStreamDelta{
                    next_tool_idx_, id_buf, open_invoke_name_, std::move(args_json), false, true});
            }
            open_invoke_started_ = false;
            tool_bytes_ = 0;
            ++next_tool_idx_;
            tool_scan_pos_ = inv_end + inv_close_len;
        }
    }
}

void Dsv41StreamParser::feed(
    const std::string& piece,
    std::string& out_reasoning_delta,
    std::string& out_content_delta,
    std::vector<Dsv41ToolStreamDelta>* out_tool_deltas,
    uint64_t call_id_seed)
{
    out_reasoning_delta.clear();
    out_content_delta.clear();
    if (out_tool_deltas) out_tool_deltas->clear();
    if (piece.empty()) return;

    full_raw_ += piece;
    if (state_ == State::InToolCalls) {
        tool_bytes_ += piece.size();
        scan_incremental_tools(out_tool_deltas, call_id_seed);
        return;
    }

    pending_ += piece;

    while (state_ == State::InThinking) {
        const std::string tag(kThinkEnd);
        const size_t te = pending_.find(tag);
        if (te != std::string::npos) {
            const size_t raw_te = full_raw_.size() - pending_.size() + te;
            const ThinkEndClass cls = classify_think_end(full_raw_, raw_te, false);
            if (cls == ThinkEndClass::InlineMention) {
                out_reasoning_delta += pending_.substr(0, te + tag.size());
                pending_.erase(0, te + tag.size());
                continue;
            }
            if (cls == ThinkEndClass::NeedMoreLookahead) {
                if (te > 0) {
                    out_reasoning_delta += pending_.substr(0, te);
                    pending_.erase(0, te);
                }
                return;
            }
            out_reasoning_delta += pending_.substr(0, te);
            pending_.erase(0, te + tag.size());
            has_real_think_end_ = true;
            real_think_end_pos_ = raw_te;
            state_ = State::InContent;
            break;
        } else {
            size_t hold = 0;
            for (size_t k = std::min(pending_.size(), tag.size() - 1); k >= 1; --k) {
                if (pending_.compare(pending_.size() - k, k, tag, 0, k) == 0) {
                    hold = k;
                    break;
                }
            }
            const size_t emit_n = pending_.size() - hold;
            if (emit_n > 0) {
                out_reasoning_delta += pending_.substr(0, emit_n);
                pending_.erase(0, emit_n);
            }
            return;
        }
    }

    if (state_ == State::InContent && !pending_.empty()) {
        const std::string dsml_open = std::string("<") + kDsmlTok;
        const size_t dp = pending_.find(dsml_open);
        if (dp != std::string::npos) {
            const size_t raw_dp = full_raw_.size() - pending_.size() + dp;
            std::string before = pending_.substr(0, dp);
            if (before.size() >= 2 && before.compare(before.size() - 2, 2, "\n\n") == 0) {
                before.erase(before.size() - 2);
            } else if (!before.empty() && before.back() == '\n') {
                before.pop_back();
            }
            out_content_delta += before;
            pending_.clear();
            state_ = State::InToolCalls;
            size_t open_len = 0;
            const size_t wrapper = (raw_dp < full_raw_.size())
                ? find_dsml_variant(full_raw_, raw_dp, {" calls>", "calls>", "function_calls>", " function_calls>"}, false, &open_len)
                : std::string::npos;
            if (wrapper == raw_dp) {
                tool_scan_pos_ = raw_dp + open_len;
            } else {
                tool_scan_pos_ = (raw_dp < full_raw_.size()) ? raw_dp : full_raw_.size();
            }
            scan_incremental_tools(out_tool_deltas, call_id_seed);
            return;
        }

        const std::string hold_pat1 = std::string("\n\n<") + kDsmlTok;
        const std::string hold_pat2 = std::string("<") + kDsmlTok;
        const std::string hold_pat3(kEosTok);

        const size_t eos_pos = pending_.find(hold_pat3);
        if (eos_pos != std::string::npos) {
            out_content_delta += pending_.substr(0, eos_pos);
            pending_.clear();
            return;
        }

        size_t hold = 0;
        auto check_hold = [&](const std::string& pat) {
            for (size_t k = std::min(pending_.size(), pat.size() - 1); k >= 1; --k) {
                if (pending_.compare(pending_.size() - k, k, pat, 0, k) == 0) {
                    if (k > hold) hold = k;
                    break;
                }
            }
        };
        check_hold(hold_pat1);
        check_hold(hold_pat2);
        check_hold(hold_pat3);

        const size_t emit_n = pending_.size() - hold;
        if (emit_n > 0) {
            out_content_delta += pending_.substr(0, emit_n);
            pending_.erase(0, emit_n);
        }
    }
}

void Dsv41StreamParser::force_close_thinking(std::string& out_reasoning_delta) {
    out_reasoning_delta.clear();
    if (state_ != State::InThinking) return;
    if (!utf8_tail_.empty()) {
        const std::string tail = bytes_to_utf8_lossy(utf8_tail_);
        utf8_tail_.clear();
        push_assistant_seg_txt(assistant_segs_, tail);
        full_raw_ += tail;
        out_reasoning_delta += tail;
        reasoning_acc_ += tail;
    }
    const std::string tag(kThinkEnd);
    const size_t raw_te = full_raw_.size();
    full_raw_ += tag;
    push_assistant_seg_tok(assistant_segs_, Dsv41SpecialTokens::kThinkEnd);
    if (!pending_.empty()) {
        out_reasoning_delta += pending_;
        reasoning_acc_ += pending_;
        pending_.clear();
    }
    has_real_think_end_ = true;
    real_think_end_pos_ = raw_te;
    state_ = State::InContent;
}

void Dsv41StreamParser::finish(
    std::string& out_reasoning_delta,
    std::string& out_content_delta,
    Dsv41ParsedCompletion& out_final,
    uint64_t call_id_seed,
    std::vector<Dsv41ToolStreamDelta>* out_tool_deltas)
{
    out_reasoning_delta.clear();
    out_content_delta.clear();
    if (out_tool_deltas) out_tool_deltas->clear();

    if (token_gated_mode_) {
        if (!utf8_tail_.empty()) {
            const std::string tail = bytes_to_utf8_lossy(utf8_tail_);
            utf8_tail_.clear();
            push_assistant_seg_txt(assistant_segs_, tail);
            full_raw_ += tail;
            if (state_ == State::InThinking) {
                out_reasoning_delta += tail;
                reasoning_acc_ += tail;
            } else if (state_ == State::InToolCalls) {
                step_token_gated_dsml_bytes(tail, out_tool_deltas, call_id_seed);
            } else {
                pending_ += tail;
            }
        }
        if (state_ == State::ConfirmingDsmlOpen) {
            pending_ += dsml_pre_ws_;
            pending_ += "<";
            pending_ += kDsmlTok;
            pending_ += dsml_open_buf_;
            dsml_pre_ws_.clear();
            dsml_open_buf_.clear();
            state_ = State::InContent;
        }
        if (state_ == State::InThinking && !pending_.empty()) {
            out_reasoning_delta += pending_;
            reasoning_acc_ += pending_;
            pending_.clear();
        } else if (state_ == State::InContent && !pending_.empty()) {
            out_content_delta += pending_;
            content_acc_ += pending_;
            pending_.clear();
        } else if (state_ == State::InToolCalls && open_invoke_started_) {
            std::string close_tail;
            if (dsml_sub_ == DsmlSub::StreamingStringParamValue) {
                if (!param_val_pending_.empty()) {
                    append_json_escaped_inner(close_tail, param_val_pending_);
                    param_val_pending_.clear();
                }
                close_tail += "\"}";
            } else if (dsml_sub_ == DsmlSub::BufferingJsonParamValue) {
                close_tail += json_quote_str(param_val_pending_);
                param_val_pending_.clear();
                close_tail += "}";
            } else if (cur_tool_param_count_ == 0) {
                close_tail = "{\"_truncated\":true}";
            } else {
                close_tail = "}";
            }
            cur_tool_json_acc_ += close_tail;
            parsed_tools_.push_back(Dsv41ParsedToolCall{cur_tool_id_, cur_tool_name_, cur_tool_json_acc_});
            if (out_tool_deltas) {
                out_tool_deltas->push_back(Dsv41ToolStreamDelta{
                    next_tool_idx_, cur_tool_id_, cur_tool_name_, std::move(close_tail), false, /*is_end=*/true});
            }
            open_invoke_started_ = false;
            ++next_tool_idx_;
        }

        out_final.reasoning_content = reasoning_acc_;
        out_final.content = content_acc_;
        out_final.tool_calls = parsed_tools_;
        return;
    }

    const std::string eos(kEosTok);
    if (pending_.size() >= eos.size() && pending_.compare(pending_.size() - eos.size(), eos.size(), eos) == 0) {
        pending_.erase(pending_.size() - eos.size());
    }

    if (state_ == State::InThinking && !pending_.empty()) {
        const std::string tag(kThinkEnd);
        const size_t te = pending_.find(tag);
        if (te != std::string::npos) {
            size_t raw_base = full_raw_.size();
            if (raw_base >= eos.size() && full_raw_.compare(raw_base - eos.size(), eos.size(), eos) == 0) {
                raw_base -= eos.size();
            }
            const size_t raw_te = (raw_base >= pending_.size()) ? (raw_base - pending_.size() + te) : te;
            if (classify_think_end(full_raw_, raw_te, true) == ThinkEndClass::RealClose) {
                out_reasoning_delta += pending_.substr(0, te);
                pending_.erase(0, te + tag.size());
                has_real_think_end_ = true;
                real_think_end_pos_ = raw_te;
                state_ = State::InContent;
            }
        }
    }

    if (state_ == State::InThinking && !pending_.empty()) {
        out_reasoning_delta += pending_;
        pending_.clear();
    } else if (state_ == State::InContent && !pending_.empty()) {
        out_content_delta = pending_;
        pending_.clear();
    } else if (state_ == State::InToolCalls) {
        // Check if outer wrapper just finished arriving at end of stream
        if (tool_scan_pos_ <= full_raw_.size() && !open_invoke_started_) {
            size_t open_len = 0;
            const std::string dsml_open = std::string("<") + kDsmlTok;
            const size_t raw_dp = full_raw_.find(dsml_open);
            if (raw_dp != std::string::npos && tool_scan_pos_ == raw_dp) {
                const size_t w = find_dsml_variant(
                    full_raw_, raw_dp, {" calls>", "calls>", "function_calls>", " function_calls>"}, false, &open_len);
                if (w == raw_dp) tool_scan_pos_ = raw_dp + open_len;
            }
        }
        scan_incremental_tools(out_tool_deltas, call_id_seed);
        if (open_invoke_started_ && out_tool_deltas) {
            // Cleanly close truncated/runaway invoke with valid JSON so client never stalls on half-open tool_call!
            char id_buf[64];
            std::snprintf(id_buf, sizeof(id_buf), "call_dsv41_%llu_%llu",
                          static_cast<unsigned long long>(call_id_seed),
                          static_cast<unsigned long long>(next_tool_idx_));
            out_tool_deltas->push_back(Dsv41ToolStreamDelta{
                next_tool_idx_, id_buf, open_invoke_name_, "{\"_truncated\":true}", false, true});
            open_invoke_started_ = false;
            ++next_tool_idx_;
        }
    }

    out_final = parse_dsv41_completion(
        full_raw_,
        thinking_mode_,
        call_id_seed,
        has_real_think_end_ ? real_think_end_pos_ : std::string::npos);
}

}  // namespace dsv41
