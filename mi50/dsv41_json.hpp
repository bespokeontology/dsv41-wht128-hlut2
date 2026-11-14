#pragma once
// Pure-C++17 Zero-Dependency Order-Preserving JSON Parser & Serializer for DeepSeek-V4.1-Flash-1017
// Preserves exact object insertion order and int64 vs double vs bool vs string types for:
//   - OpenAI /v1/chat/completions, /v1/models, /v1/tokenize HTTP & SSE payloads
//   - Exact Python-compatible json.dumps(..., ensure_ascii=False) in DSML tool schema rendering

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dsv41 {

struct JsonValue {
    enum Type { Null, Bool, Int, Double, String, Array, Object };

    Type type = Null;
    bool b_val = false;
    int64_t i_val = 0;
    double d_val = 0.0;
    std::string s_val;
    std::vector<JsonValue> arr_val;
    std::vector<std::pair<std::string, JsonValue>> obj_val;

    JsonValue() = default;
    JsonValue(bool b) : type(Bool), b_val(b) {}
    JsonValue(int64_t i) : type(Int), i_val(i), d_val(double(i)) {}
    JsonValue(int i) : type(Int), i_val(int64_t(i)), d_val(double(i)) {}
    JsonValue(double d) : type(Double), i_val(int64_t(d)), d_val(d) {}
    JsonValue(const char* s) : type(String), s_val(s ? s : "") {}
    JsonValue(std::string s) : type(String), s_val(std::move(s)) {}
    static JsonValue make_null() { return JsonValue(); }
    static JsonValue make_bool(bool b) { JsonValue v; v.type = Bool; v.b_val = b; return v; }
    static JsonValue make_int(int64_t i) { JsonValue v; v.type = Int; v.i_val = i; v.d_val = double(i); return v; }
    static JsonValue make_double(double d) { JsonValue v; v.type = Double; v.d_val = d; v.i_val = int64_t(d); return v; }
    static JsonValue make_string(std::string s) { JsonValue v; v.type = String; v.s_val = std::move(s); return v; }
    static JsonValue make_array() { JsonValue v; v.type = Array; return v; }
    static JsonValue make_object() { JsonValue v; v.type = Object; return v; }

    bool is_null() const { return type == Null; }
    bool is_bool() const { return type == Bool; }
    bool is_int() const { return type == Int; }
    bool is_number() const { return type == Int || type == Double; }
    bool is_string() const { return type == String; }
    bool is_array() const { return type == Array; }
    bool is_object() const { return type == Object; }

    double as_double(double def = 0.0) const {
        if (type == Double) return d_val;
        if (type == Int) return double(i_val);
        return def;
    }
    double as_number(double def = 0.0) const { return as_double(def); }
    int64_t as_int(int64_t def = 0) const {
        if (type == Int) return i_val;
        if (type == Double) return int64_t(d_val);
        return def;
    }
    bool as_bool(bool def = false) const {
        if (type == Bool) return b_val;
        return def;
    }
    const std::string& as_str() const {
        static const std::string kEmpty;
        return (type == String) ? s_val : kEmpty;
    }
    std::string as_str(const std::string& def) const {
        return (type == String) ? s_val : def;
    }

    bool has(const std::string& key) const {
        if (type != Object) return false;
        for (const auto& kv : obj_val) {
            if (kv.first == key) return true;
        }
        return false;
    }

    const JsonValue* find(const std::string& key) const {
        if (type != Object) return nullptr;
        for (const auto& kv : obj_val) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }

    JsonValue* find_mut(const std::string& key) {
        if (type != Object) return nullptr;
        for (auto& kv : obj_val) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }

    const JsonValue& get(const std::string& key) const {
        static const JsonValue kNull;
        const JsonValue* p = find(key);
        return p ? *p : kNull;
    }

    bool contains(const std::string& key) const { return has(key); }
    const JsonValue& at(const std::string& key) const { return get(key); }
    void set_key(const std::string& key, JsonValue val) { set(key, std::move(val)); }
    void push_back(JsonValue val) { push(std::move(val)); }

    void set(const std::string& key, JsonValue val) {
        if (type != Object) {
            type = Object;
            obj_val.clear();
        }
        for (auto& kv : obj_val) {
            if (kv.first == key) {
                kv.second = std::move(val);
                return;
            }
        }
        obj_val.emplace_back(key, std::move(val));
    }

    void erase(const std::string& key) {
        if (type != Object) return;
        for (auto it = obj_val.begin(); it != obj_val.end(); ++it) {
            if (it->first == key) {
                obj_val.erase(it);
                return;
            }
        }
    }

    void push(JsonValue val) {
        if (type != Array) {
            type = Array;
            arr_val.clear();
        }
        arr_val.push_back(std::move(val));
    }

    static void append_escaped_str(std::string& out, const std::string& s) {
        out.push_back('"');
        for (unsigned char c : s) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (c < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", unsigned(c));
                        out += buf;
                    } else {
                        out.push_back(char(c));
                    }
                    break;
            }
        }
        out.push_back('"');
    }

    void dump_into(std::string& out, bool py_separators) const {
        const char* comma = py_separators ? ", " : ",";
        const char* colon = py_separators ? ": " : ":";
        switch (type) {
            case Null:
                out += "null";
                break;
            case Bool:
                out += (b_val ? "true" : "false");
                break;
            case Int:
                out += std::to_string(i_val);
                break;
            case Double: {
                if (!std::isfinite(d_val)) {
                    out += "null";
                } else {
                    char buf[64];
                    std::snprintf(buf, sizeof(buf), "%.16g", d_val);
                    // Preserve float decimal point like Python json.dumps when integer-valued Double
                    bool has_dot_or_exp = false;
                    for (const char* p = buf; *p; ++p) {
                        if (*p == '.' || *p == 'e' || *p == 'E') { has_dot_or_exp = true; break; }
                    }
                    out += buf;
                    if (!has_dot_or_exp) out += ".0";
                }
                break;
            }
            case String:
                append_escaped_str(out, s_val);
                break;
            case Array:
                out.push_back('[');
                for (size_t i = 0; i < arr_val.size(); ++i) {
                    if (i) out += comma;
                    arr_val[i].dump_into(out, py_separators);
                }
                out.push_back(']');
                break;
            case Object:
                out.push_back('{');
                for (size_t i = 0; i < obj_val.size(); ++i) {
                    if (i) out += comma;
                    append_escaped_str(out, obj_val[i].first);
                    out += colon;
                    obj_val[i].second.dump_into(out, py_separators);
                }
                out.push_back('}');
                break;
        }
    }

    std::string dumps_compact() const {
        std::string out;
        out.reserve(128);
        dump_into(out, false);
        return out;
    }

    // Matches Python json.dumps(v, ensure_ascii=False)
    std::string dumps_py() const {
        std::string out;
        out.reserve(128);
        dump_into(out, true);
        return out;
    }
};

class JsonParser {
public:
    static JsonValue parse(const std::string& src) {
        JsonParser p(src.data(), src.size());
        JsonValue v = p.parse_value();
        p.skip_ws();
        if (p.pos_ != p.len_) {
            throw std::runtime_error("JSON trailing characters at byte " + std::to_string(p.pos_));
        }
        return v;
    }

    static bool try_parse(const std::string& src, JsonValue& out) {
        try {
            out = parse(src);
            return true;
        } catch (...) {
            return false;
        }
    }

private:
    const char* s_;
    size_t len_;
    size_t pos_ = 0;

    JsonParser(const char* s, size_t len) : s_(s), len_(len) {}

    void skip_ws() {
        while (pos_ < len_) {
            const char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') ++pos_;
            else break;
        }
    }

    static void append_utf8(std::string& out, uint32_t cp) {
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

    uint32_t parse_hex4() {
        if (pos_ + 4 > len_) throw std::runtime_error("JSON truncated \\u escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = s_[pos_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
            else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
            else throw std::runtime_error("JSON invalid hex digit");
        }
        return v;
    }

    std::string parse_string() {
        if (pos_ >= len_ || s_[pos_] != '"') throw std::runtime_error("JSON expected '\"'");
        ++pos_;
        std::string out;
        while (pos_ < len_) {
            const char c = s_[pos_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (pos_ >= len_) throw std::runtime_error("JSON unterminated escape");
                const char esc = s_[pos_++];
                switch (esc) {
                    case '"':  out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/'); break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'u': {
                        uint32_t cp = parse_hex4();
                        if (cp >= 0xD800u && cp <= 0xDBFFu && pos_ + 2 <= len_ &&
                            s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
                            pos_ += 2;
                            const uint32_t lo = parse_hex4();
                            if (lo >= 0xDC00u && lo <= 0xDFFFu) {
                                cp = 0x10000u + (((cp - 0xD800u) << 10) | (lo - 0xDC00u));
                            }
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default:
                        out.push_back(esc);
                        break;
                }
            } else {
                out.push_back(c);
            }
        }
        throw std::runtime_error("JSON unterminated string");
    }

    JsonValue parse_number() {
        const size_t start = pos_;
        if (pos_ < len_ && s_[pos_] == '-') ++pos_;
        while (pos_ < len_ && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
        bool is_float = false;
        if (pos_ < len_ && s_[pos_] == '.') {
            is_float = true;
            ++pos_;
            while (pos_ < len_ && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
        }
        if (pos_ < len_ && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
            is_float = true;
            ++pos_;
            if (pos_ < len_ && (s_[pos_] == '+' || s_[pos_] == '-')) ++pos_;
            while (pos_ < len_ && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
        }
        const std::string num_s(s_ + start, pos_ - start);
        if (!is_float) {
            char* endp = nullptr;
            const long long iv = std::strtoll(num_s.c_str(), &endp, 10);
            if (endp && *endp == '\0') return JsonValue::make_int(int64_t(iv));
        }
        char* endp = nullptr;
        const double dv = std::strtod(num_s.c_str(), &endp);
        if (!endp || *endp != '\0') throw std::runtime_error("JSON invalid number: " + num_s);
        return JsonValue::make_double(dv);
    }

    JsonValue parse_value() {
        skip_ws();
        if (pos_ >= len_) throw std::runtime_error("JSON unexpected EOF");
        const char c = s_[pos_];
        if (c == '"') return JsonValue::make_string(parse_string());
        if (c == '{') {
            ++pos_;
            JsonValue obj = JsonValue::make_object();
            skip_ws();
            if (pos_ < len_ && s_[pos_] == '}') { ++pos_; return obj; }
            while (true) {
                skip_ws();
                std::string k = parse_string();
                skip_ws();
                if (pos_ >= len_ || s_[pos_] != ':') throw std::runtime_error("JSON expected ':'");
                ++pos_;
                JsonValue v = parse_value();
                obj.obj_val.emplace_back(std::move(k), std::move(v));
                skip_ws();
                if (pos_ < len_ && s_[pos_] == ',') { ++pos_; continue; }
                if (pos_ < len_ && s_[pos_] == '}') { ++pos_; break; }
                throw std::runtime_error("JSON expected ',' or '}'");
            }
            return obj;
        }
        if (c == '[') {
            ++pos_;
            JsonValue arr = JsonValue::make_array();
            skip_ws();
            if (pos_ < len_ && s_[pos_] == ']') { ++pos_; return arr; }
            while (true) {
                arr.arr_val.push_back(parse_value());
                skip_ws();
                if (pos_ < len_ && s_[pos_] == ',') { ++pos_; continue; }
                if (pos_ < len_ && s_[pos_] == ']') { ++pos_; break; }
                throw std::runtime_error("JSON expected ',' or ']'");
            }
            return arr;
        }
        if (c == 't' && pos_ + 4 <= len_ && std::memcmp(s_ + pos_, "true", 4) == 0) {
            pos_ += 4;
            return JsonValue::make_bool(true);
        }
        if (c == 'f' && pos_ + 5 <= len_ && std::memcmp(s_ + pos_, "false", 5) == 0) {
            pos_ += 5;
            return JsonValue::make_bool(false);
        }
        if (c == 'n' && pos_ + 4 <= len_ && std::memcmp(s_ + pos_, "null", 4) == 0) {
            pos_ += 4;
            return JsonValue::make_null();
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            return parse_number();
        }
        throw std::runtime_error(std::string("JSON unexpected char: ") + c);
    }
};

inline JsonValue parse_json(const std::string& src) {
    return JsonParser::parse(src);
}

}  // namespace dsv41
