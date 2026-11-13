// Pure-C++17 Tokenizer, ICU Engram Vocab Compressor (99,092 exact), V4.1 Chat Renderer & DSML Parser
// Zero Python anywhere.

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <unicode/locid.h>
#include <unicode/normalizer2.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>

#include "dsv41_tokenizer_dsml.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>

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

std::string Dsv41Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special_tokens) const {
    const auto& bt = byte_tables();
    std::string raw_bytes;
    raw_bytes.reserve(ids.size() * 4);

    for (int32_t id : ids) {
        if (id < 0 || size_t(id) >= id_to_token_.size()) continue;
        if (skip_special_tokens && id_is_special_[size_t(id)]) continue;
        const std::string& tok = id_to_token_[size_t(id)];
        if (impl_->is_added_id[size_t(id)]) {
            raw_bytes.append(tok);
            continue;
        }
        size_t i = 0;
        while (i < tok.size()) {
            const size_t prev = i;
            const uint32_t cp = next_utf8_lossy(tok, i);
            auto it = bt.cp_to_byte.find(cp);
            if (it != bt.cp_to_byte.end()) {
                raw_bytes.push_back(char(it->second));
            } else {
                raw_bytes.append(tok.data() + prev, i - prev);
            }
        }
    }
    return bytes_to_utf8_lossy(raw_bytes);
}

std::string Dsv41Tokenizer::decode_single(int32_t id, bool skip_special_tokens) const {
    return decode(std::vector<int32_t>{id}, skip_special_tokens);
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

static const char* kBosTok       = "<\xEF\xBD\x9C" "begin\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>";
static const char* kEosTok       = "<\xEF\xBD\x9C" "end\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>";
static const char* kSystemTok    = "<\xEF\xBD\x9C" "System\xEF\xBD\x9C>";
static const char* kUserTok      = "<\xEF\xBD\x9C" "User\xEF\xBD\x9C>";
static const char* kAssistantTok = "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>";
static const char* kReminderTok  = "<\xEF\xBD\x9C" "latest_reminder\xEF\xBD\x9C>";
static const char* kThinkStart   = "<think>";
static const char* kThinkEnd     = "</think>";
static const char* kDsmlTok      = "\xEF\xBD\x9C" "DSML\xEF\xBD\x9C";

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

std::string render_tools_block(const std::vector<JsonValue>& fns) {
    std::string schemas;
    for (size_t i = 0; i < fns.size(); ++i) {
        if (i) schemas.push_back('\n');
        schemas += fns[i].dumps_py();
    }
    std::string s;
    s.reserve(schemas.size() + 1024);
    s += "## Tools\n\n";
    s += "You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<";
    s += kDsmlTok;
    s += " calls>\" block like the following:\n\n<";
    s += kDsmlTok;
    s += " calls>\n<";
    s += kDsmlTok;
    s += " invoke name=\"$TOOL_NAME\">\n<";
    s += kDsmlTok;
    s += " parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</";
    s += kDsmlTok;
    s += " parameter>\n...\n</";
    s += kDsmlTok;
    s += " invoke>\n<";
    s += kDsmlTok;
    s += " invoke name=\"$TOOL_NAME2\">\n...\n</";
    s += kDsmlTok;
    s += " invoke>\n</";
    s += kDsmlTok;
    s += " calls>\n\n";
    s += "String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n";
    s += "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n";
    s += "Otherwise, output directly after </think> with tool calls or final response.\n\n";
    s += "### Available Tool Schemas\n\n";
    s += schemas;
    s += "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";
    return s;
}

std::string encode_arguments_to_dsml(const JsonValue& raw_args) {
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

    std::string out;
    for (size_t i = 0; i < args.obj_val.size(); ++i) {
        if (i) out.push_back('\n');
        const std::string& k = args.obj_val[i].first;
        const JsonValue& v = args.obj_val[i].second;
        const bool is_str = v.is_string();
        out += "<";
        out += kDsmlTok;
        out += " parameter name=\"";
        out += k;
        out += "\" string=\"";
        out += (is_str ? "true" : "false");
        out += "\">";
        out += (is_str ? v.s_val : v.dumps_py());
        out += "</";
        out += kDsmlTok;
        out += " parameter>";
    }
    return out;
}

int resolve_reasoning_budget(const JsonValue* effort_val, std::string& io_thinking_mode) {
    // Interactive default budget = 55 per Speed Wave 3 directive (supports "low"=50, "medium"=55, "high"=75, "max"=100, or int 1..100; 0/"none" -> chat mode)
    if (!effort_val || effort_val->is_null()) return 55;
    if (effort_val->is_int() || effort_val->is_number()) {
        const int v = int(effort_val->as_int(55));
        if (v <= 0) {
            io_thinking_mode = "chat";
            return 0;
        }
        return std::max(1, std::min(100, v));
    }
    if (effort_val->is_string()) {
        std::string s = effort_val->s_val;
        for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
        if (s.empty()) return 55;
        if (s == "none" || s == "off" || s == "disabled" || s == "chat" || s == "0") {
            io_thinking_mode = "chat";
            return 0;
        }
        if (s == "minimal") return 35;
        if (s == "low") return 50;
        if (s == "medium") return 55;
        if (s == "high") return 75;
        if (s == "max" || s == "xhigh") return 100;
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
    return 55;
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
                    if (bt == "text" || bt == "image" || bt == "image_url") {
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

std::string render_dsv41_chat_prompt(
    const JsonValue& raw_messages,
    const JsonValue* tools,
    const JsonValue* response_format,
    const JsonValue* reasoning_effort_val,
    std::string& out_thinking_mode,
    int& out_budget)
{
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
            }
        }
    }

    // 5. Render full prompt starting with BOS (encoding.py:783, 585-742)
    std::string prompt = kBosTok;
    for (int index = 0; index < int(merged.size()); ++index) {
        const JsonValue& msg = merged[size_t(index)];
        const std::string role = msg.get("role").as_str();

        std::string effort_prefix;
        if (index == 0 && out_thinking_mode == "thinking") {
            effort_prefix = "Reasoning Effort: " + std::to_string(out_budget) +
                            " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
        }
        if (index == 0 && (!effort_prefix.empty() || role == "system")) {
            prompt += kSystemTok;
        }
        prompt += effort_prefix;

        if (role == "system") {
            if (index > 0) prompt += kSystemTok;
            prompt += extract_text_content(msg.get("content"));
            if (msg.get("tools").is_array() && !msg.get("tools").arr_val.empty()) {
                prompt += "\n\n";
                prompt += render_tools_block(tools_from_openai_format(msg.get("tools")));
            }
            if (msg.get("response_format").is_object() && !msg.get("response_format").obj_val.empty()) {
                prompt += "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n";
                prompt += msg.get("response_format").dumps_py();
            }
        } else if (role == "user") {
            prompt += kUserTok;
            if (msg.get("content_blocks").is_array()) {
                const auto& blks = msg.get("content_blocks").arr_val;
                for (size_t bi = 0; bi < blks.size(); ++bi) {
                    if (bi) prompt += "\n\n";
                    const auto& b = blks[bi];
                    const std::string bt = b.get("type").as_str();
                    if (bt == "text") {
                        prompt += b.get("text").as_str();
                    } else if (bt == "tool_result") {
                        prompt += "<tool_result>";
                        prompt += extract_text_content(b.get("content"));
                        prompt += "</tool_result>";
                    }
                }
            } else {
                prompt += extract_text_content(msg.get("content"));
            }
        } else if (role == "latest_reminder") {
            prompt += kReminderTok;
            prompt += extract_text_content(msg.get("content"));
        } else if (role == "assistant") {
            std::string thinking_part;
            if (out_thinking_mode == "thinking") {
                if (!drop_thinking || index > last_user_idx) {
                    thinking_part = msg.get("reasoning_content").as_str() + kThinkEnd;
                }
            }
            std::string tc_content;
            if (msg.get("tool_calls").is_array() && !msg.get("tool_calls").arr_val.empty()) {
                tc_content += "\n\n<";
                tc_content += kDsmlTok;
                tc_content += " calls>\n";
                const auto& tcs = msg.get("tool_calls").arr_val;
                for (size_t ti = 0; ti < tcs.size(); ++ti) {
                    if (ti) tc_content.push_back('\n');
                    const JsonValue& tc = tcs[ti];
                    const JsonValue& fn = tc.has("function") ? tc.get("function") : tc;
                    std::string ns = tc.get("namespace").as_str();
                    if (ns.empty()) ns = fn.get("namespace").as_str();
                    auto [rns, bare] = split_tool_name(fn.get("name").as_str(), ns);
                    const std::string qname = rns.empty() ? bare : (rns + "::" + bare);
                    tc_content += "<";
                    tc_content += kDsmlTok;
                    tc_content += " invoke name=\"";
                    tc_content += qname;
                    tc_content += "\">\n";
                    tc_content += encode_arguments_to_dsml(fn.get("arguments"));
                    tc_content += "\n</";
                    tc_content += kDsmlTok;
                    tc_content += " invoke>";
                }
                tc_content += "\n</";
                tc_content += kDsmlTok;
                tc_content += " calls>";
            }
            prompt += thinking_part;
            prompt += extract_text_content(msg.get("content"));
            prompt += tc_content;
            if (!msg.get("wo_eos").as_bool(false)) {
                prompt += kEosTok;
            }
        }

        if (index + 1 < int(merged.size())) {
            const std::string& next_role = merged[size_t(index + 1)].get("role").as_str();
            if (next_role != "assistant" && next_role != "latest_reminder") {
                continue;
            }
        }

        if (role == "user" || (role == "system" && index > 0)) {
            prompt += kAssistantTok;
            if ((!drop_thinking && out_thinking_mode == "thinking") ||
                (drop_thinking && out_thinking_mode == "thinking" && index >= last_user_idx)) {
                prompt += kThinkStart;
            } else {
                prompt += kThinkEnd;
            }
        }
    }

    return prompt;
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

Dsv41StreamParser::Dsv41StreamParser(const std::string& thinking_mode)
    : thinking_mode_(thinking_mode),
      state_(thinking_mode == "thinking" ? State::InThinking : State::InContent) {}

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

            // Once >= 1 <｜DSML｜invoke>...</｜DSML｜invoke> has completed, the ONLY valid next DSML construct
            // after whitespace is a parallel <｜DSML｜ invoke...> / <｜DSML｜invoke...>; any closing tag or
            // non-invoke continuation closes the tool block immediately without runaway generation!
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
                            return;  // Wait for '>' of closing wrapper tag so stream cuts cleanly right at '>'
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
                out_tool_deltas->push_back(Dsv41ToolStreamDelta{next_tool_idx_, id_buf, bare_name, "", true});
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
                    next_tool_idx_, id_buf, open_invoke_name_, std::move(args_json), false});
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
    const std::string tag(kThinkEnd);
    const size_t raw_te = full_raw_.size();
    full_raw_ += tag;
    if (!pending_.empty()) {
        out_reasoning_delta = pending_;
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
                next_tool_idx_, id_buf, open_invoke_name_, "{\"_truncated\":true}", false});
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
