// Wave 9.4 standalone pure-C++17 engine & ISA static auditor.
//   - Parses every .s file under <root>/build/tools: kernel count, private segment
//     fixed size, SGPR/VGPR spill counts, device trig mnemonics, SLC atomics.
//   - Scans <root>/mi50/*.hpp and *.hip (plus supplemental *.cpp for the tokenizer
//     slice cache): system-scope RMW atomics, kMaxChunkM, TokSliceCache capacity,
//     compute_model_overlay_uuid path-argument count.
//   - --self-test injects synthetic violations (scratch > 0, SLC atomic,
//     system-scope fetch_add, spills, trig, invariant drift) and proves detection.
// Zero Python anywhere. C++17 only.
//
// Build: g++ -O3 -std=c++17 -Wall -Wextra -Werror w94_engine_auditor.cpp -o w94_engine_auditor

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace w94audit {

namespace fs = std::filesystem;

// Wave 9.4 final-profile receipt: "164/164 zero-scratch" compiled gfx906 kernels.
constexpr std::size_t kExpectedGfx906Kernels = 164u;
constexpr long long    kExpectedMaxChunkM    = 4096ll;
constexpr long long    kExpectedTokSliceCap  = 8192ll;
constexpr std::size_t  kExpectedUuidArgs     = 5u;
constexpr const char*  kDefaultRoot          = ".";
constexpr std::string_view kGfx906Marker     = "amdgcn-amd-amdhsa--gfx906";

// ---------------------------------------------------------------------------
// Generic string helpers
// ---------------------------------------------------------------------------

inline bool is_space_char(unsigned char c) { return std::isspace(c) != 0; }
inline bool is_ident_char(unsigned char c) { return std::isalnum(c) != 0 || c == '_'; }

inline std::string_view trim(std::string_view v) {
    std::size_t b = 0;
    while (b < v.size() && is_space_char(static_cast<unsigned char>(v[b]))) ++b;
    std::size_t e = v.size();
    while (e > b && is_space_char(static_cast<unsigned char>(v[e - 1u]))) --e;
    return v.substr(b, e - b);
}

inline std::string trim_copy(std::string_view v) {
    const std::string_view t = trim(v);
    return std::string(t);
}

inline bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

inline std::string_view after_prefix(std::string_view s, std::string_view p) {
    return trim(s.substr(p.size()));
}

inline std::string_view first_word(std::string_view s) {
    std::size_t i = 0;
    while (i < s.size() && !is_space_char(static_cast<unsigned char>(s[i]))) ++i;
    return s.substr(0, i);
}

inline bool parse_leading_i64(std::string_view s, long long& out) {
    const std::string_view t = trim(s);
    if (t.empty()) return false;
    long long v = 0;
    const char* begin = t.data();
    const char* end = t.data() + t.size();
    const std::from_chars_result r = std::from_chars(begin, end, v);
    if (r.ec != std::errc() || r.ptr == begin) return false;
    out = v;
    return true;
}

inline std::vector<std::string_view> split_tokens(std::string_view s) {
    std::vector<std::string_view> out;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() &&
               (is_space_char(static_cast<unsigned char>(s[i])) || s[i] == ',')) {
            ++i;
        }
        if (i >= s.size()) break;
        const std::size_t b = i;
        while (i < s.size() &&
               !is_space_char(static_cast<unsigned char>(s[i])) && s[i] != ',') {
            ++i;
        }
        out.push_back(s.substr(b, i - b));
    }
    return out;
}

inline bool contains_substr(std::string_view hay, std::string_view needle) {
    return hay.find(needle) != std::string_view::npos;
}

inline bool has_bounded_token(std::string_view hay, std::string_view tok) {
    std::size_t p = 0;
    while ((p = hay.find(tok, p)) != std::string_view::npos) {
        const bool left = (p == 0u) || !is_ident_char(static_cast<unsigned char>(hay[p - 1u]));
        const std::size_t e = p + tok.size();
        const bool right = (e >= hay.size()) ||
                           !is_ident_char(static_cast<unsigned char>(hay[e]));
        if (left && right) return true;
        p = e;
    }
    return false;
}

inline std::size_t line_number_at(std::string_view text, std::size_t off) {
    std::size_t line = 1u;
    const std::size_t lim = std::min(off, text.size());
    for (std::size_t i = 0; i < lim; ++i) {
        if (text[i] == '\n') ++line;
    }
    return line;
}

inline std::string compact_ws(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    bool pending_space = false;
    for (char c : s) {
        if (is_space_char(static_cast<unsigned char>(c))) {
            pending_space = !out.empty();
        } else {
            if (pending_space) out.push_back(' ');
            pending_space = false;
            out.push_back(c);
        }
    }
    return out;
}

inline std::string join_values(const std::vector<long long>& values) {
    std::string s;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0u) s += ", ";
        s += std::to_string(values[i]);
    }
    return s;
}

// ---------------------------------------------------------------------------
// Comment / string-literal stripper. Length and newline offsets are preserved,
// so line numbers remain valid while comments and literals cannot trigger
// false positives (e.g. the "Zero __HIP_MEMORY_SCOPE_SYSTEM" architectural
// comments or a stale "kMaxChunkM = 512" note).
// ---------------------------------------------------------------------------

enum class LexState { Normal, LineComment, BlockComment, StringLit, CharLit };

inline std::string strip_comments_and_literals(std::string_view in) {
    std::string out(in);
    LexState st = LexState::Normal;
    std::size_t i = 0;
    const std::size_t n = in.size();
    while (i < n) {
        const char c = in[i];
        switch (st) {
            case LexState::Normal:
                if (c == '/' && i + 1u < n && in[i + 1u] == '/') {
                    out[i] = ' ';
                    out[i + 1u] = ' ';
                    i += 2u;
                    st = LexState::LineComment;
                } else if (c == '/' && i + 1u < n && in[i + 1u] == '*') {
                    out[i] = ' ';
                    out[i + 1u] = ' ';
                    i += 2u;
                    st = LexState::BlockComment;
                } else if (c == '"') {
                    out[i] = ' ';
                    ++i;
                    st = LexState::StringLit;
                } else if (c == '\'') {
                    out[i] = ' ';
                    ++i;
                    st = LexState::CharLit;
                } else {
                    ++i;
                }
                break;
            case LexState::LineComment:
                if (c == '\n') {
                    st = LexState::Normal;
                } else {
                    out[i] = ' ';
                }
                ++i;
                break;
            case LexState::BlockComment:
                if (c == '*' && i + 1u < n && in[i + 1u] == '/') {
                    out[i] = ' ';
                    out[i + 1u] = ' ';
                    i += 2u;
                    st = LexState::Normal;
                } else {
                    if (c != '\n') out[i] = ' ';
                    ++i;
                }
                break;
            case LexState::StringLit:
            case LexState::CharLit:
                if (c == '\\' && i + 1u < n) {
                    out[i] = ' ';
                    out[i + 1u] = (in[i + 1u] == '\n') ? '\n' : ' ';
                    i += 2u;
                } else if ((st == LexState::StringLit && c == '"') ||
                           (st == LexState::CharLit && c == '\'')) {
                    out[i] = ' ';
                    ++i;
                    st = LexState::Normal;
                } else {
                    if (c != '\n') out[i] = ' ';
                    ++i;
                }
                break;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// ISA assembly audit
// ---------------------------------------------------------------------------

struct AsmKernelInfo {
    std::string name;
    bool directive_seen = false;   // .amdhsa_kernel <name>
    bool metadata_seen  = false;   // .amdgpu_metadata .name: <name>
    std::vector<long long> private_values;  // top directive + msgpack descriptor
    std::vector<long long> sgpr_values;     // .sgpr_spill_count (both syntaxes)
    std::vector<long long> vgpr_values;     // .vgpr_spill_count (both syntaxes)
};

struct AsmFileReport {
    std::string path;
    std::string filename;
    bool gfx906_target = false;
    std::size_t kernel_directives = 0u;
    std::size_t kernel_names      = 0u;
    std::size_t scratch_markers   = 0u;
    std::size_t scratch_zero      = 0u;
    std::size_t trig_count        = 0u;
    std::size_t slc_atomic_count  = 0u;
    std::size_t private_zero      = 0u;
    std::size_t sgpr_zero         = 0u;
    std::size_t vgpr_zero         = 0u;
    std::map<std::string, AsmKernelInfo> kernels;
    std::vector<std::string> violations;
    bool pass = false;
};

inline bool is_atomic_mnemonic(std::string_view t) {
    static const char* const kPrefixes[] = {
        "global_atomic_", "flat_atomic_", "buffer_atomic_", "ds_atomic_", "s_atomic_",
        "global_atomic",  "flat_atomic",  "buffer_atomic", "ds_atomic"
    };
    for (const char* p : kPrefixes) {
        if (starts_with(t, p)) return true;
    }
    return false;
}

inline bool extract_directive_value(std::string_view line, std::string_view dir,
                                    long long& out) {
    if (!starts_with(line, dir)) return false;
    std::string_view rest = line.substr(dir.size());
    if (!rest.empty() && rest.front() == ':') rest.remove_prefix(1u);
    rest = trim(rest);
    if (rest.empty()) return false;
    return parse_leading_i64(rest, out);
}

inline void push_field_values(std::vector<long long>& dst, long long v) {
    dst.push_back(v);
}

inline bool all_zero(const std::vector<long long>& v) {
    if (v.empty()) return false;
    for (long long x : v) {
        if (x != 0ll) return false;
    }
    return true;
}

inline bool has_nonzero(const std::vector<long long>& v) {
    for (long long x : v) {
        if (x != 0ll) return true;
    }
    return false;
}

inline bool has_conflicting_values(const std::vector<long long>& v) {
    if (v.empty()) return false;
    const long long first = v.front();
    for (long long x : v) {
        if (x != first) return true;
    }
    return false;
}

AsmFileReport scan_asm_text(std::string_view text, const std::string& display_path) {
    AsmFileReport r;
    r.path = display_path;
    r.filename = fs::path(display_path).filename().string();
    r.gfx906_target = text.find(kGfx906Marker) != std::string_view::npos;

    std::string current_name;
    std::size_t line_no = 0u;
    std::size_t pos = 0u;

    while (pos < text.size()) {
        std::size_t nl = text.find('\n', pos);
        const std::size_t end = (nl == std::string_view::npos) ? text.size() : nl;
        const std::string_view raw = text.substr(pos, end - pos);
        const std::string_view t = trim(raw);
        ++line_no;

        // Kernel identity: top-level directive.
        if (starts_with(t, ".amdhsa_kernel")) {
            const std::string_view name = first_word(after_prefix(t, ".amdhsa_kernel"));
            if (name.empty()) {
                r.violations.push_back("line " + std::to_string(line_no) +
                                       ": empty .amdhsa_kernel directive");
            } else {
                current_name.assign(name);
                AsmKernelInfo& k = r.kernels[current_name];
                k.name = current_name;
                k.directive_seen = true;
                ++r.kernel_directives;
            }
        } else if (starts_with(t, ".name:")) {
            // Kernel identity: msgpack descriptor (.amdgpu_metadata).
            const std::string_view name = after_prefix(t, ".name:");
            if (name.empty()) {
                r.violations.push_back("line " + std::to_string(line_no) +
                                       ": empty metadata .name:");
            } else {
                current_name.assign(name);
                AsmKernelInfo& k = r.kernels[current_name];
                k.name = current_name;
                k.metadata_seen = true;
                ++r.kernel_names;
            }
        } else {
            long long v = 0;
            bool handled = false;
            if (extract_directive_value(t, ".amdhsa_private_segment_fixed_size", v) ||
                extract_directive_value(t, ".private_segment_fixed_size:", v)) {
                handled = true;
                if (current_name.empty()) {
                    r.violations.push_back("line " + std::to_string(line_no) +
                                           ": private segment metadata outside kernel");
                } else {
                    push_field_values(r.kernels[current_name].private_values, v);
                }
            }
            if (extract_directive_value(t, ".sgpr_spill_count:", v) ||
                extract_directive_value(t, ".sgpr_spill_count", v)) {
                handled = true;
                if (current_name.empty()) {
                    r.violations.push_back("line " + std::to_string(line_no) +
                                           ": sgpr spill metadata outside kernel");
                } else {
                    push_field_values(r.kernels[current_name].sgpr_values, v);
                }
            }
            if (extract_directive_value(t, ".vgpr_spill_count:", v) ||
                extract_directive_value(t, ".vgpr_spill_count", v)) {
                handled = true;
                if (current_name.empty()) {
                    r.violations.push_back("line " + std::to_string(line_no) +
                                           ": vgpr spill metadata outside kernel");
                } else {
                    push_field_values(r.kernels[current_name].vgpr_values, v);
                }
            }
            (void)handled;

            // llvm .AMDGPU.csdata kernel-info scratch marker.
            if (starts_with(t, "; ScratchSize:")) {
                std::string_view rest = t.substr(std::string_view("; ScratchSize:").size());
                rest = trim(rest);
                long long sv = 0;
                if (parse_leading_i64(rest, sv)) {
                    ++r.scratch_markers;
                    if (sv == 0ll) {
                        ++r.scratch_zero;
                    } else {
                        r.violations.push_back("line " + std::to_string(line_no) +
                                               ": ScratchSize=" + std::to_string(sv) +
                                               " (must be 0)");
                    }
                } else {
                    r.violations.push_back("line " + std::to_string(line_no) +
                                           ": unparsable ScratchSize marker");
                }
            }
        }

        // Instruction-level audit. Assembly comment lines are skipped so a
        // comment mention cannot masquerade as an instruction.
        if (!t.empty() && t.front() != ';') {
            bool line_atomic = false;
            bool line_slc    = false;
            for (std::string_view tok : split_tokens(raw)) {
                if (tok == "v_sin_f32" || tok == "v_cos_f32") {
                    ++r.trig_count;
                    r.violations.push_back("line " + std::to_string(line_no) +
                                           ": device trig instruction " +
                                           std::string(tok));
                }
                if (is_atomic_mnemonic(tok)) line_atomic = true;
                if (tok == "slc") line_slc = true;
            }
            if (line_atomic && line_slc) {
                ++r.slc_atomic_count;
                r.violations.push_back("line " + std::to_string(line_no) +
                                       ": SLC (system-coherent) atomic instruction");
            }
        }

        if (nl == std::string_view::npos) break;
        pos = nl + 1u;
    }

    // Per-kernel completeness and zero-value verification.
    for (const auto& entry : r.kernels) {
        const AsmKernelInfo& k = entry.second;
        if (!k.directive_seen) {
            r.violations.push_back("kernel " + k.name +
                                   ": msgpack descriptor without .amdhsa_kernel directive");
        }
        if (!k.metadata_seen) {
            r.violations.push_back("kernel " + k.name +
                                   ": missing .amdgpu_metadata kernel descriptor");
        }

        const auto check_field = [&](const std::vector<long long>& vals,
                                     const char* field_name, std::size_t& zero_count) {
            if (vals.empty()) {
                r.violations.push_back("kernel " + k.name + ": missing " + field_name);
                return;
            }
            if (has_nonzero(vals)) {
                r.violations.push_back("kernel " + k.name + ": nonzero " + field_name +
                                       " value(s) [" + join_values(vals) + "]");
            }
            if (has_conflicting_values(vals)) {
                r.violations.push_back("kernel " + k.name + ": conflicting " + field_name +
                                       " metadata [" + join_values(vals) + "]");
            }
            if (all_zero(vals)) ++zero_count;
        };

        check_field(k.private_values, "amdhsa_private_segment_fixed_size", r.private_zero);
        check_field(k.sgpr_values, "sgpr_spill_count", r.sgpr_zero);
        check_field(k.vgpr_values, "vgpr_spill_count", r.vgpr_zero);
    }

    if (r.kernel_directives == 0u) {
        r.violations.push_back("no gfx906 kernels found");
    }
    if (!r.gfx906_target) {
        r.violations.push_back("not a gfx906 assembly bundle");
    }
    r.pass = r.violations.empty() && r.gfx906_target && r.kernel_directives > 0u;
    return r;
}

// ---------------------------------------------------------------------------
// Filesystem enumeration
// ---------------------------------------------------------------------------

std::vector<fs::path> list_files(const fs::path& dir,
                                 std::initializer_list<std::string_view> extensions,
                                 std::vector<std::string>& errors) {
    std::vector<fs::path> out;
    std::error_code ec;
    fs::directory_iterator it(dir, ec);
    if (ec) {
        errors.push_back("directory not readable: " + dir.string() + " (" + ec.message() + ")");
        return out;
    }
    const fs::directory_iterator end_it;
    while (it != end_it) {
        const fs::directory_entry& ent = *it;
        std::error_code fec;
        if (ent.is_regular_file(fec)) {
            const std::string ext = ent.path().extension().string();
            for (std::string_view want : extensions) {
                if (ext == want) {
                    out.push_back(ent.path());
                    break;
                }
            }
        }
        it.increment(ec);
        if (ec) {
            errors.push_back("directory iteration failure: " + dir.string() +
                             " (" + ec.message() + ")");
            break;
        }
    }
    std::sort(out.begin(), out.end(), [](const fs::path& a, const fs::path& b) {
        return a.filename().string() < b.filename().string();
    });
    return out;
}

bool read_text_file(const fs::path& path, std::string& out, std::vector<std::string>& errors) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        errors.push_back("cannot open " + path.string());
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (in.bad()) {
        errors.push_back("read failure " + path.string());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Source-level system-scope RMW scan
// ---------------------------------------------------------------------------

struct RmwHit {
    std::size_t line = 0u;
    std::string snippet;
    std::string token;
};

struct RmwTokenSpec {
    const char* text;
    bool system_intrinsic;  // *_system HIP builtins are system-scope by definition
};

// Longest forms first. Identifier-boundary checks prevent nested matches
// (e.g. "fetch_add" inside "__hip_atomic_fetch_add" or "atomicAdd" inside
// "atomicAdd_system").
static const RmwTokenSpec kRmwTokens[] = {
    {"__hip_atomic_compare_exchange_strong", false},
    {"__hip_atomic_compare_exchange_weak",   false},
    {"__hip_atomic_fetch_add",               false},
    {"__hip_atomic_fetch_sub",               false},
    {"__hip_atomic_exchange",                false},
    {"atomic_compare_exchange_strong_explicit", false},
    {"atomic_compare_exchange_weak_explicit",   false},
    {"atomic_fetch_add_explicit",            false},
    {"atomic_fetch_sub_explicit",            false},
    {"atomic_fetch_or_explicit",             false},
    {"atomic_fetch_and_explicit",            false},
    {"atomic_fetch_xor_explicit",            false},
    {"atomicAdd_system",                     true},
    {"atomicSub_system",                     true},
    {"atomicCAS_system",                     true},
    {"atomicExch_system",                    true},
    {"atomicInc_system",                     true},
    {"atomicDec_system",                     true},
    {"atomicAnd_system",                     true},
    {"atomicOr_system",                      true},
    {"atomicXor_system",                     true},
    {"atomicMax_system",                     true},
    {"atomicMin_system",                     true},
    {"atomicAdd",                            false},
    {"atomicSub",                            false},
    {"atomicCAS",                            false},
    {"atomicExch",                           false},
    {"atomicMin",                            false},
    {"atomicMax",                            false},
    {"atomicAnd",                            false},
    {"atomicOr",                             false},
    {"atomicXor",                            false},
    {"atomicInc",                            false},
    {"atomicDec",                            false},
    {"fetch_add",                            false},
    {"fetch_sub",                            false},
    {"compare_exchange_strong",              false},
    {"compare_exchange_weak",                false},
    {"compare_exchange",                     false},
    {"exchange",                             false},
};

inline std::size_t find_matching_paren(std::string_view code, std::size_t open) {
    int depth = 0;
    for (std::size_t i = open; i < code.size(); ++i) {
        if (code[i] == '(') {
            ++depth;
        } else if (code[i] == ')') {
            --depth;
            if (depth == 0) return i;
        }
    }
    return std::string_view::npos;
}

inline std::string snippet_at(std::string_view code, std::size_t off) {
    std::size_t b = code.rfind('\n', off);
    b = (b == std::string_view::npos) ? 0u : b + 1u;
    std::size_t e = code.find('\n', off);
    if (e == std::string_view::npos) e = code.size();
    std::string s = trim_copy(code.substr(b, e - b));
    if (s.size() > 160u) s.resize(160u);
    return s;
}

std::vector<RmwHit> scan_system_scope_rmw(std::string_view code) {
    std::vector<RmwHit> hits;
    std::set<std::size_t> seen_call_positions;
    for (const RmwTokenSpec& spec : kRmwTokens) {
        const std::string_view tok(spec.text);
        std::size_t p = 0;
        while ((p = code.find(tok, p)) != std::string_view::npos) {
            const std::size_t e = p + tok.size();
            const bool left = (p == 0u) ||
                              !is_ident_char(static_cast<unsigned char>(code[p - 1u]));
            const bool right = (e >= code.size()) ||
                               !is_ident_char(static_cast<unsigned char>(code[e]));
            if (left && right) {
                std::size_t q = e;
                while (q < code.size() &&
                       is_space_char(static_cast<unsigned char>(code[q]))) {
                    ++q;
                }
                bool call_args_ok = false;
                std::string_view args;
                std::size_t key = p;
                if (q < code.size() && code[q] == '(') {
                    const std::size_t close = find_matching_paren(code, q);
                    if (close != std::string_view::npos) {
                        key = q;
                        args = code.substr(q + 1u, close - q - 1u);
                        call_args_ok = true;
                    }
                }
                bool hit = spec.system_intrinsic;
                if (!hit && call_args_ok &&
                    has_bounded_token(args, "__HIP_MEMORY_SCOPE_SYSTEM")) {
                    hit = true;
                }
                if (hit && seen_call_positions.insert(key).second) {
                    RmwHit h;
                    h.line = line_number_at(code, p);
                    h.snippet = snippet_at(code, p);
                    h.token = spec.text;
                    hits.push_back(std::move(h));
                }
            }
            p = e;
        }
    }
    std::sort(hits.begin(), hits.end(), [](const RmwHit& a, const RmwHit& b) {
        return a.line < b.line;
    });
    return hits;
}

// ---------------------------------------------------------------------------
// Source invariant parsers (kMaxChunkM, TokSliceCache, overlay UUID signature)
// ---------------------------------------------------------------------------

struct InvariantParse {
    bool found = false;
    bool valid = false;
    std::vector<long long> values;
    std::string source;
    std::size_t line = 0u;
};

InvariantParse parse_kmaxchunkm(std::string_view code, const std::string& source) {
    InvariantParse r;
    r.source = source;
    const std::string_view tok = "kMaxChunkM";
    std::size_t p = 0;
    while ((p = code.find(tok, p)) != std::string_view::npos) {
        const std::size_t e = p + tok.size();
        const bool left = (p == 0u) ||
                          !is_ident_char(static_cast<unsigned char>(code[p - 1u]));
        const bool right = (e >= code.size()) ||
                           !is_ident_char(static_cast<unsigned char>(code[e]));
        if (left && right) {
            std::size_t i = e;
            while (i < code.size() &&
                   is_space_char(static_cast<unsigned char>(code[i]))) {
                ++i;
            }
            if (i < code.size() && code[i] == '=') {
                ++i;
                while (i < code.size() &&
                       is_space_char(static_cast<unsigned char>(code[i]))) {
                    ++i;
                }
                long long v = 0;
                if (parse_leading_i64(code.substr(i), v)) {
                    r.values.push_back(v);
                    if (!r.found) {
                        r.found = true;
                        r.line = line_number_at(code, p);
                    }
                }
            }
        }
        p = e;
    }
    r.valid = r.found && !r.values.empty();
    for (long long v : r.values) {
        if (v != kExpectedMaxChunkM) r.valid = false;
    }
    return r;
}

InvariantParse parse_tokslice_capacity(std::string_view code, const std::string& source) {
    InvariantParse r;
    r.source = source;
    const std::string_view struct_name = "struct TokSliceCache";
    const std::size_t struct_pos = code.find(struct_name);
    if (struct_pos == std::string_view::npos) return r;

    std::size_t search_from = struct_pos + struct_name.size();
    const std::string_view tok = "kMaxEntries";
    std::size_t p = code.find(tok, search_from);
    if (p == std::string_view::npos) return r;

    const std::size_t e = p + tok.size();
    std::size_t i = e;
    while (i < code.size() &&
           is_space_char(static_cast<unsigned char>(code[i]))) {
        ++i;
    }
    if (i >= code.size() || code[i] != '=') return r;
    ++i;
    while (i < code.size() &&
           is_space_char(static_cast<unsigned char>(code[i]))) {
        ++i;
    }
    long long v = 0;
    if (!parse_leading_i64(code.substr(i), v)) return r;
    r.found = true;
    r.line = line_number_at(code, p);
    r.values.push_back(v);
    r.valid = (v == kExpectedTokSliceCap);
    return r;
}

struct UuidParse {
    bool found = false;
    bool valid = false;
    std::size_t arg_count = 0u;
    bool all_std_string = false;
    bool all_path_named = false;
    std::string signature;
    std::string source;
    std::size_t line = 0u;
};

inline std::vector<std::string_view> split_top_level_params(std::string_view s) {
    std::vector<std::string_view> out;
    int paren = 0;
    int angle = 0;
    int brace = 0;
    int bracket = 0;
    std::size_t start = 0u;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        switch (c) {
            case '(': ++paren; break;
            case ')': if (paren > 0) --paren; break;
            case '<': ++angle; break;
            case '>': if (angle > 0) --angle; break;
            case '{': ++brace; break;
            case '}': if (brace > 0) --brace; break;
            case '[': ++bracket; break;
            case ']': if (bracket > 0) --bracket; break;
            case ',':
                if (paren == 0 && angle == 0 && brace == 0 && bracket == 0) {
                    out.push_back(trim(s.substr(start, i - start)));
                    start = i + 1u;
                }
                break;
            default: break;
        }
    }
    out.push_back(trim(s.substr(start)));
    if (out.size() == 1u && out.front().empty()) out.clear();
    return out;
}

UuidParse parse_overlay_uuid(std::string_view code, const std::string& source) {
    UuidParse r;
    r.source = source;
    const std::string_view tok = "compute_model_overlay_uuid";
    std::size_t p = 0;
    bool any = false;
    bool all_valid = true;

    while ((p = code.find(tok, p)) != std::string_view::npos) {
        const std::size_t e = p + tok.size();
        const bool left = (p == 0u) ||
                          !is_ident_char(static_cast<unsigned char>(code[p - 1u]));
        const bool right = (e >= code.size()) ||
                           !is_ident_char(static_cast<unsigned char>(code[e]));
        if (left && right) {
            std::size_t q = e;
            while (q < code.size() &&
                   is_space_char(static_cast<unsigned char>(code[q]))) {
                ++q;
            }
            if (q < code.size() && code[q] == '(') {
                const std::size_t close = find_matching_paren(code, q);
                if (close != std::string_view::npos) {
                    const std::string_view params =
                        code.substr(q + 1u, close - q - 1u);
                    const std::vector<std::string_view> parts =
                        split_top_level_params(params);
                    bool all_string = true;
                    bool all_path = true;
                    for (std::string_view part : parts) {
                        if (!contains_substr(part, "std::string")) all_string = false;
                        if (!contains_substr(part, "path") &&
                            !contains_substr(part, "root")) {
                            all_path = false;
                        }
                    }
                    const bool this_valid =
                        (parts.size() == kExpectedUuidArgs) && all_string && all_path;
                    if (!any) {
                        r.found = true;
                        r.line = line_number_at(code, p);
                        r.arg_count = parts.size();
                        r.all_std_string = all_string;
                        r.all_path_named = all_path;
                        r.signature = compact_ws(params);
                        if (r.signature.size() > 220u) r.signature.resize(220u);
                    }
                    any = true;
                    if (!this_valid) all_valid = false;
                }
            }
        }
        p = e;
    }
    r.valid = any && all_valid;
    return r;
}

// ---------------------------------------------------------------------------
// Check 14: k_sparse_attn_allheads 4-key s_valid guard must be pure bitwise OR,
// with zero bitwise AND operators in the guard condition.
// ---------------------------------------------------------------------------

inline std::size_t find_bounded_token(std::string_view hay, std::string_view tok,
                                      std::size_t from = 0u) {
    std::size_t p = from;
    while ((p = hay.find(tok, p)) != std::string_view::npos) {
        const bool left = (p == 0u) ||
                          !is_ident_char(static_cast<unsigned char>(hay[p - 1u]));
        const std::size_t e = p + tok.size();
        const bool right = (e >= hay.size()) ||
                           !is_ident_char(static_cast<unsigned char>(hay[e]));
        if (left && right) return p;
        p = e;
    }
    return std::string_view::npos;
}

inline std::size_t find_matching_brace(std::string_view code, std::size_t open) {
    int depth = 0;
    for (std::size_t i = open; i < code.size(); ++i) {
        if (code[i] == '{') {
            ++depth;
        } else if (code[i] == '}') {
            --depth;
            if (depth == 0) return i;
        }
    }
    return std::string_view::npos;
}

inline std::string strip_all_ws(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (!is_space_char(static_cast<unsigned char>(c))) out.push_back(c);
    }
    return out;
}

struct GuardInspect {
    bool valid = false;
    std::size_t bitwise_or_count = 0u;
    std::size_t bitwise_and_count = 0u;
};

inline GuardInspect inspect_allheads_guard(std::string_view condition) {
    GuardInspect g;
    const std::string norm = strip_all_ws(condition);
    const bool has_terms =
        norm.find("s_valid[t+0]") != std::string::npos &&
        norm.find("s_valid[t+1]") != std::string::npos &&
        norm.find("s_valid[t+2]") != std::string::npos &&
        norm.find("s_valid[t+3]") != std::string::npos;
    for (std::size_t i = 0; i < norm.size(); ++i) {
        const char c = norm[i];
        if (c == '|') {
            const bool logical =
                (i > 0u && norm[i - 1u] == '|') ||
                (i + 1u < norm.size() && norm[i + 1u] == '|');
            if (!logical) ++g.bitwise_or_count;
        } else if (c == '&') {
            const bool logical =
                (i > 0u && norm[i - 1u] == '&') ||
                (i + 1u < norm.size() && norm[i + 1u] == '&');
            if (!logical) ++g.bitwise_and_count;
        }
    }
    g.valid = has_terms && g.bitwise_or_count == 3u && g.bitwise_and_count == 0u;
    return g;
}

struct AllheadsGuardParse {
    bool found = false;
    bool valid = false;
    std::size_t guard_count = 0u;
    std::size_t or_guard_count = 0u;
    std::size_t bitwise_and_count = 0u;
    std::size_t line = 0u;
    std::string source;
};

AllheadsGuardParse parse_allheads_guard(std::string_view code, const std::string& source) {
    AllheadsGuardParse r;
    r.source = source;
    const std::string_view tok = "k_sparse_attn_allheads";
    std::size_t search = 0u;
    while (true) {
        const std::size_t p = find_bounded_token(code, tok, search);
        if (p == std::string_view::npos) break;
        search = p + tok.size();

        std::size_t q = p + tok.size();
        while (q < code.size() &&
               is_space_char(static_cast<unsigned char>(code[q]))) {
            ++q;
        }
        if (q >= code.size() || code[q] != '(') continue;
        const std::size_t sig_close = find_matching_paren(code, q);
        if (sig_close == std::string_view::npos) continue;

        std::size_t b = sig_close + 1u;
        while (b < code.size() &&
               is_space_char(static_cast<unsigned char>(code[b]))) {
            ++b;
        }
        if (b >= code.size() || code[b] != '{') continue;
        const std::size_t body_close = find_matching_brace(code, b);
        if (body_close == std::string_view::npos) continue;

        const std::string_view body = code.substr(b + 1u, body_close - b - 1u);
        r.found = true;
        r.line = line_number_at(code, p);

        std::size_t gp = 0u;
        while ((gp = body.find("s_valid", gp)) != std::string_view::npos) {
            const std::size_t lb = body.find('[', gp + 7u);
            if (lb == std::string_view::npos) break;
            const std::size_t rb = body.find(']', lb + 1u);
            if (rb == std::string_view::npos) break;
            const std::string index = strip_all_ws(body.substr(lb + 1u, rb - lb - 1u));
            if (index != "t+0") {
                gp = rb + 1u;
                continue;
            }

            const std::size_t line_start = [&]() {
                const std::size_t nl = body.rfind('\n', gp);
                return (nl == std::string_view::npos) ? 0u : nl + 1u;
            }();
            const std::size_t if_pos = body.find("if", line_start);
            if (if_pos != std::string_view::npos && if_pos < gp) {
                const bool if_left = (if_pos == 0u) ||
                    !is_ident_char(static_cast<unsigned char>(body[if_pos - 1u]));
                const std::size_t if_end = if_pos + 2u;
                const bool if_right = (if_end >= body.size()) ||
                    !is_ident_char(static_cast<unsigned char>(body[if_end]));
                if (if_left && if_right) {
                    std::size_t ip = if_end;
                    while (ip < body.size() &&
                           is_space_char(static_cast<unsigned char>(body[ip]))) {
                        ++ip;
                    }
                    if (ip < body.size() && body[ip] == '(') {
                        const std::size_t cond_close = find_matching_paren(body, ip);
                        if (cond_close != std::string_view::npos &&
                            gp > ip && gp < cond_close) {
                            const std::string_view cond =
                                body.substr(ip + 1u, cond_close - ip - 1u);
                            const GuardInspect gi = inspect_allheads_guard(cond);
                            ++r.guard_count;
                            r.bitwise_and_count += gi.bitwise_and_count;
                            if (gi.valid) ++r.or_guard_count;
                        }
                    }
                }
            }
            gp = rb + 1u;
        }
        break;  // First actual definition is authoritative.
    }
    r.valid = r.found && r.guard_count > 0u &&
              r.or_guard_count == r.guard_count && r.bitwise_and_count == 0u;
    return r;
}

// ---------------------------------------------------------------------------
// Check 15: src/hlut3_pack.hpp static_assert(sizeof(Hlut3Tile32x32) == 400),
// resolving either the literal 400 or the kHlut3TileBytes constant.
// ---------------------------------------------------------------------------

struct TileAssertParse {
    bool found = false;
    bool struct_found = false;
    bool constant_found = false;
    bool assert_found = false;
    long long constant_value = -1;
    long long assert_value = -1;
    std::size_t assert_count = 0u;
    bool valid = false;
    std::size_t line = 0u;
    std::string source;
};

TileAssertParse parse_tile_size_assert(std::string_view code, const std::string& source) {
    TileAssertParse r;
    r.source = source;
    r.struct_found = has_bounded_token(code, "Hlut3Tile32x32");

    // Resolve inline constexpr int kHlut3TileBytes = 400 (or any declared value).
    const std::string_view ctok = "kHlut3TileBytes";
    std::size_t cp = 0u;
    while ((cp = find_bounded_token(code, ctok, cp)) != std::string_view::npos) {
        std::size_t q = cp + ctok.size();
        while (q < code.size() &&
               is_space_char(static_cast<unsigned char>(code[q]))) {
            ++q;
        }
        if (q < code.size() && code[q] == '=') {
            ++q;
            while (q < code.size() &&
                   is_space_char(static_cast<unsigned char>(code[q]))) {
                ++q;
            }
            long long v = 0;
            if (parse_leading_i64(code.substr(q), v)) {
                r.constant_found = true;
                r.constant_value = v;
                break;
            }
        }
        cp += ctok.size();
    }

    const std::string_view sizeof_tok = "sizeof(Hlut3Tile32x32)";
    const std::string_view sa_tok = "static_assert";
    std::size_t sp = 0u;
    bool all_400 = true;
    bool any = false;
    while ((sp = code.find(sizeof_tok, sp)) != std::string_view::npos) {
        const std::size_t sa = code.rfind(sa_tok, sp);
        bool in_static_assert = false;
        if (sa != std::string_view::npos) {
            const bool sa_left = (sa == 0u) ||
                !is_ident_char(static_cast<unsigned char>(code[sa - 1u]));
            const std::size_t sa_end = sa + sa_tok.size();
            const bool sa_right = (sa_end >= code.size()) ||
                !is_ident_char(static_cast<unsigned char>(code[sa_end]));
            if (sa_left && sa_right) {
                const std::string_view between = code.substr(sa, sp - sa);
                if (between.find(';') == std::string_view::npos &&
                    between.find('(') != std::string_view::npos) {
                    in_static_assert = true;
                }
            }
        }
        if (in_static_assert) {
            std::size_t q = sp + sizeof_tok.size();
            while (q < code.size() &&
                   is_space_char(static_cast<unsigned char>(code[q]))) {
                ++q;
            }
            if (q + 1u < code.size() && code[q] == '=' && code[q + 1u] == '=') {
                q += 2u;
                while (q < code.size() &&
                       is_space_char(static_cast<unsigned char>(code[q]))) {
                    ++q;
                }
                long long v = -1;
                bool ok = false;
                if (q < code.size() &&
                    std::isdigit(static_cast<unsigned char>(code[q])) != 0) {
                    ok = parse_leading_i64(code.substr(q), v);
                } else {
                    std::size_t id_end = q;
                    while (id_end < code.size() &&
                           is_ident_char(static_cast<unsigned char>(code[id_end]))) {
                        ++id_end;
                    }
                    const std::string_view id = code.substr(q, id_end - q);
                    if (id == ctok && r.constant_found) {
                        v = r.constant_value;
                        ok = true;
                    }
                }
                if (ok) {
                    any = true;
                    ++r.assert_count;
                    r.assert_value = v;
                    r.line = line_number_at(code, sp);
                    if (v != 400ll) all_400 = false;
                }
            }
        }
        sp += sizeof_tok.size();
    }
    r.assert_found = any;
    r.found = any;
    r.valid = r.struct_found && r.assert_found && r.assert_count > 0u && all_400;
    return r;
}

// ---------------------------------------------------------------------------
// Aggregate audits
// ---------------------------------------------------------------------------

struct AsmAudit {
    std::vector<AsmFileReport> files;
    std::size_t total_files        = 0u;
    std::size_t total_directives   = 0u;
    std::size_t total_names        = 0u;
    std::size_t total_scratch      = 0u;
    std::size_t total_scratch_zero = 0u;
    std::size_t total_private_zero = 0u;
    std::size_t total_sgpr_zero    = 0u;
    std::size_t total_vgpr_zero    = 0u;
    std::size_t total_trig         = 0u;
    std::size_t total_slc          = 0u;
    std::size_t arch_gfx906_files  = 0u;
    bool ok = false;
};

bool run_asm_audit(const fs::path& root, AsmAudit& a, std::vector<std::string>& errors) {
    const fs::path tools_dir = root / "build" / "tools";
    const std::vector<fs::path> paths =
        list_files(tools_dir, {std::string_view(".s")}, errors);
    if (paths.empty()) {
        errors.push_back("no .s files under " + tools_dir.string());
        return false;
    }

    a.files.clear();
    for (const fs::path& p : paths) {
        std::string text;
        if (!read_text_file(p, text, errors)) continue;
        AsmFileReport rep = scan_asm_text(text, p.string());
        a.total_files += 1u;
        a.total_directives += rep.kernel_directives;
        a.total_names += rep.kernel_names;
        a.total_scratch += rep.scratch_markers;
        a.total_scratch_zero += rep.scratch_zero;
        a.total_private_zero += rep.private_zero;
        a.total_sgpr_zero += rep.sgpr_zero;
        a.total_vgpr_zero += rep.vgpr_zero;
        a.total_trig += rep.trig_count;
        a.total_slc += rep.slc_atomic_count;
        if (rep.gfx906_target) ++a.arch_gfx906_files;
        if (!rep.pass) {
            errors.push_back("ISA violations in " + rep.filename);
        }
        a.files.push_back(std::move(rep));
    }

    if (a.files.size() != paths.size()) {
        errors.push_back("not every .s file was read");
    }

    bool all_files_pass = true;
    for (const AsmFileReport& rep : a.files) {
        if (!rep.pass) all_files_pass = false;
    }
    const bool total_ok = (a.total_directives == kExpectedGfx906Kernels);
    const bool parity_ok = (a.total_directives == a.total_names);
    const bool scratch_ok = (a.total_scratch == a.total_directives) &&
                            (a.total_scratch_zero == a.total_scratch);
    const bool private_ok = (a.total_private_zero == a.total_directives);
    const bool sgpr_ok = (a.total_sgpr_zero == a.total_directives);
    const bool vgpr_ok = (a.total_vgpr_zero == a.total_directives);
    const bool trig_ok = (a.total_trig == 0u);
    const bool slc_ok = (a.total_slc == 0u);
    const bool arch_ok = (a.arch_gfx906_files == a.total_files);

    a.ok = all_files_pass && total_ok && parity_ok && scratch_ok &&
           private_ok && sgpr_ok && vgpr_ok && trig_ok && slc_ok && arch_ok;
    if (!a.ok) errors.push_back("aggregate ISA audit failed");
    return true;
}

struct SourceAudit {
    std::size_t scanned_hpp_hip = 0u;
    std::size_t scanned_cpp     = 0u;
    std::vector<RmwHit> rmw_hits;
    InvariantParse kmax;
    InvariantParse tokcache;
    UuidParse uuid;
    AllheadsGuardParse allheads;
    TileAssertParse tile_assert;
    std::size_t scanned_src = 0u;
    bool ok = false;
};

bool run_source_audit(const fs::path& root, SourceAudit& s,
                      std::vector<std::string>& errors) {
    const fs::path mi50_dir = root / "mi50";
    const std::vector<fs::path> core_paths =
        list_files(mi50_dir, {std::string_view(".hpp"), std::string_view(".hip")}, errors);
    if (core_paths.empty()) {
        errors.push_back("no .hpp/.hip files under " + mi50_dir.string());
        return false;
    }

    bool all_read = true;
    for (const fs::path& p : core_paths) {
        std::string raw;
        if (!read_text_file(p, raw, errors)) {
            all_read = false;
            continue;
        }
        ++s.scanned_hpp_hip;
        const std::string code = strip_comments_and_literals(raw);
        const std::string source = p.string();

        std::vector<RmwHit> hits = scan_system_scope_rmw(code);
        for (RmwHit& h : hits) s.rmw_hits.push_back(std::move(h));

        InvariantParse km = parse_kmaxchunkm(code, source);
        if (km.found) {
            s.kmax.found = true;
            s.kmax.source = source;
            s.kmax.line = km.line;
            for (long long v : km.values) s.kmax.values.push_back(v);
        }
        UuidParse up = parse_overlay_uuid(code, source);
        if (up.found) {
            s.uuid = std::move(up);
        }
        AllheadsGuardParse ag = parse_allheads_guard(code, source);
        if (ag.found) {
            s.allheads = std::move(ag);
        }
    }

    // TokSliceCache is defined in dsv41_tokenizer_dsml.cpp, not in a .hpp/.hip.
    // Supplemental .cpp scan is therefore mandatory to verify its capacity.
    const std::vector<fs::path> cpp_paths =
        list_files(mi50_dir, {std::string_view(".cpp")}, errors);
    for (const fs::path& p : cpp_paths) {
        std::string raw;
        if (!read_text_file(p, raw, errors)) {
            all_read = false;
            continue;
        }
        ++s.scanned_cpp;
        const std::string code = strip_comments_and_literals(raw);
        InvariantParse tc = parse_tokslice_capacity(code, p.string());
        if (tc.found) s.tokcache = std::move(tc);
    }

    // Check 15 source: src/hlut3_pack.hpp asserts the 400 B HLUT3 tile size.
    const std::vector<fs::path> src_paths =
        list_files(root / "src", {std::string_view(".hpp")}, errors);
    if (src_paths.empty()) {
        errors.push_back("no .hpp files under " + (root / "src").string());
        all_read = false;
    }
    for (const fs::path& p : src_paths) {
        std::string raw;
        if (!read_text_file(p, raw, errors)) {
            all_read = false;
            continue;
        }
        ++s.scanned_src;
        const std::string code = strip_comments_and_literals(raw);
        TileAssertParse ta = parse_tile_size_assert(code, p.string());
        if (ta.found) s.tile_assert = std::move(ta);
    }

    s.kmax.valid = s.kmax.found && !s.kmax.values.empty();
    for (long long v : s.kmax.values) {
        if (v != kExpectedMaxChunkM) s.kmax.valid = false;
    }

    const bool rmw_ok = s.rmw_hits.empty();
    const bool kmax_ok = s.kmax.found && s.kmax.valid;
    const bool tokcache_ok = s.tokcache.found && s.tokcache.valid;
    const bool uuid_ok = s.uuid.found && s.uuid.valid &&
                         (s.uuid.arg_count == kExpectedUuidArgs);
    const bool allheads_ok = s.allheads.found && s.allheads.valid;
    const bool tile_ok = s.tile_assert.found && s.tile_assert.valid;
    s.ok = all_read && rmw_ok && kmax_ok && tokcache_ok && uuid_ok &&
           allheads_ok && tile_ok;
    if (!s.ok) errors.push_back("source invariant audit failed");
    return true;
}

// ---------------------------------------------------------------------------
// Structured report rows
// ---------------------------------------------------------------------------

struct CheckRow {
    std::string check;
    std::string scope;
    std::string expected;
    std::string observed;
    bool pass = false;
};

std::string fraction(std::size_t zero, std::size_t total) {
    return std::to_string(zero) + "/" + std::to_string(total);
}

std::vector<CheckRow> build_asm_rows(const AsmAudit& a) {
    std::vector<CheckRow> rows;
    rows.push_back({"total gfx906 kernels", "build/tools/*.s",
                    std::to_string(kExpectedGfx906Kernels),
                    std::to_string(a.total_directives),
                    a.total_directives == kExpectedGfx906Kernels});
    rows.push_back({"kernel directive/descriptor parity", "build/tools/*.s",
                    "equal",
                    std::to_string(a.total_directives) + " / " +
                        std::to_string(a.total_names),
                    a.total_directives == a.total_names});
    rows.push_back({"amdhsa_private_segment_fixed_size == 0", "build/tools/*.s",
                    fraction(a.total_directives, a.total_directives),
                    fraction(a.total_private_zero, a.total_directives),
                    a.total_private_zero == a.total_directives});
    rows.push_back({"sgpr_spill_count == 0", "build/tools/*.s",
                    fraction(a.total_directives, a.total_directives),
                    fraction(a.total_sgpr_zero, a.total_directives),
                    a.total_sgpr_zero == a.total_directives});
    rows.push_back({"vgpr_spill_count == 0", "build/tools/*.s",
                    fraction(a.total_directives, a.total_directives),
                    fraction(a.total_vgpr_zero, a.total_directives),
                    a.total_vgpr_zero == a.total_directives});
    rows.push_back({"ScratchSize == 0 (kernel-info)", "build/tools/*.s",
                    fraction(a.total_directives, a.total_directives),
                    fraction(a.total_scratch_zero, a.total_scratch) +
                        " (" + std::to_string(a.total_scratch) + " markers)",
                    a.total_scratch == a.total_directives &&
                        a.total_scratch_zero == a.total_scratch});
    rows.push_back({"device trig instructions", "build/tools/*.s",
                    "0", std::to_string(a.total_trig), a.total_trig == 0u});
    rows.push_back({"SLC atomics", "build/tools/*.s",
                    "0", std::to_string(a.total_slc), a.total_slc == 0u});
    rows.push_back({"gfx906 assembly bundles", "build/tools/*.s",
                    fraction(a.total_files, a.total_files),
                    fraction(a.arch_gfx906_files, a.total_files),
                    a.arch_gfx906_files == a.total_files});
    return rows;
}

std::vector<CheckRow> build_source_rows(const SourceAudit& s) {
    std::vector<CheckRow> rows;

    std::string rmw_observed = std::to_string(s.rmw_hits.size());
    rows.push_back({"system-scope RMW atomics", "mi50/*.hpp + *.hip (" +
                                                 std::to_string(s.scanned_hpp_hip) +
                                                 " files)",
                    "0", rmw_observed, s.rmw_hits.empty()});

    std::string kmax_observed = s.kmax.found ? join_values(s.kmax.values) : "not found";
    rows.push_back({"kMaxChunkM == 4096", "mi50/*.hpp + *.hip",
                    std::to_string(kExpectedMaxChunkM), kmax_observed,
                    s.kmax.found && s.kmax.valid});

    std::string tok_observed = s.tokcache.found ? join_values(s.tokcache.values) : "not found";
    if (s.tokcache.found) {
        tok_observed += " (" + fs::path(s.tokcache.source).filename().string() + ")";
    }
    rows.push_back({"TokSliceCache capacity == 8192", "mi50/*.cpp (supplemental)",
                    std::to_string(kExpectedTokSliceCap), tok_observed,
                    s.tokcache.found && s.tokcache.valid});

    std::string uuid_observed = s.uuid.found
        ? std::to_string(s.uuid.arg_count) + " args (std::string=" +
              (s.uuid.all_std_string ? "yes" : "no") + ", path/root names=" +
              (s.uuid.all_path_named ? "yes" : "no") + ")"
        : "not found";
    rows.push_back({"compute_model_overlay_uuid path args == 5",
                    s.uuid.found
                        ? fs::path(s.uuid.source).filename().string()
                        : "mi50/*.hpp + *.hip",
                    std::to_string(kExpectedUuidArgs), uuid_observed,
                    s.uuid.found && s.uuid.valid &&
                        s.uuid.arg_count == kExpectedUuidArgs});

    // Check 14: k_sparse_attn_allheads 4-key guard is pure bitwise OR (AND == 0).
    std::string allheads_observed = s.allheads.found
        ? "guards=" + std::to_string(s.allheads.guard_count) +
              " OR-only=" + std::to_string(s.allheads.or_guard_count) +
              " bitwise-AND=" + std::to_string(s.allheads.bitwise_and_count)
        : "not found";
    rows.push_back({"k_sparse_attn_allheads 4-key OR-only guard",
                    s.allheads.found
                        ? fs::path(s.allheads.source).filename().string()
                        : "mi50/*.hip",
                    "OR-only; AND=0", allheads_observed,
                    s.allheads.found && s.allheads.valid});

    // Check 15: Hlut3Tile32x32 static_assert(sizeof == 400).
    std::string tile_observed = s.tile_assert.found
        ? "asserts=" + std::to_string(s.tile_assert.assert_count) +
              " sizeof=" + std::to_string(s.tile_assert.assert_value) +
              " constant=" + (s.tile_assert.constant_found
                                  ? std::to_string(s.tile_assert.constant_value)
                                  : std::string("n/a"))
        : "not found";
    rows.push_back({"Hlut3Tile32x32 sizeof==400 static_assert",
                    s.tile_assert.found
                        ? fs::path(s.tile_assert.source).filename().string()
                        : "src/hlut3_pack.hpp",
                    "400", tile_observed,
                    s.tile_assert.found && s.tile_assert.valid});
    return rows;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

void print_asm_table(const AsmAudit& a) {
    std::printf("\n=== TABLE 1: GFX906 ISA PER-FILE ===\n");
    std::printf("%-29s %-8s %8s %12s %12s %12s %12s %6s %6s %-7s\n",
                "file", "arch", "kernels", "priv==0", "sgpr==0", "vgpr==0",
                "scratch==0", "trig", "slc", "verdict");
    for (const AsmFileReport& r : a.files) {
        std::printf("%-29s %-8s %8zu %12s %12s %12s %12s %6zu %6zu %-7s\n",
                    r.filename.c_str(),
                    r.gfx906_target ? "gfx906" : "other",
                    r.kernel_directives,
                    fraction(r.private_zero, r.kernel_directives).c_str(),
                    fraction(r.sgpr_zero, r.kernel_directives).c_str(),
                    fraction(r.vgpr_zero, r.kernel_directives).c_str(),
                    fraction(r.scratch_zero, r.scratch_markers).c_str(),
                    r.trig_count,
                    r.slc_atomic_count,
                    r.pass ? "PASS" : "FAIL");
    }
    std::printf("%-29s %-8s %8zu %12s %12s %12s %12s %6zu %6zu %-7s\n",
                "TOTAL",
                fraction(a.arch_gfx906_files, a.total_files).c_str(),
                a.total_directives,
                fraction(a.total_private_zero, a.total_directives).c_str(),
                fraction(a.total_sgpr_zero, a.total_directives).c_str(),
                fraction(a.total_vgpr_zero, a.total_directives).c_str(),
                fraction(a.total_scratch_zero, a.total_scratch).c_str(),
                a.total_trig,
                a.total_slc,
                a.ok ? "PASS" : "FAIL");
}

void print_check_table(const char* title, const std::vector<CheckRow>& rows,
                       std::size_t& passed, std::size_t& total) {
    std::printf("\n=== %s ===\n", title);
    std::printf("%-42s %-31s %-22s %-36s %-7s\n",
                "check", "scope", "expected", "observed", "result");
    for (const CheckRow& r : rows) {
        std::printf("%-42s %-31s %-22s %-36s %-7s\n",
                    r.check.c_str(), r.scope.c_str(), r.expected.c_str(),
                    r.observed.c_str(), r.pass ? "PASS" : "FAIL");
        ++total;
        if (r.pass) ++passed;
    }
}

// ---------------------------------------------------------------------------
// Self-test: synthetic violation injection + clean-control false-positive proof
// ---------------------------------------------------------------------------

struct SelfTestRow {
    std::string id;
    std::string injected;
    bool caught = false;
    bool clean_control = false;
};

inline bool violations_contain(const std::vector<std::string>& violations,
                               std::string_view needle) {
    for (const std::string& v : violations) {
        if (v.find(needle) != std::string::npos) return true;
    }
    return false;
}

std::string synthetic_asm(bool bad_scratch, bool bad_sgpr, bool bad_vgpr,
                          bool bad_scratch_marker, bool bad_slc, bool bad_trig) {
    std::string out;
    out += "\t.amdgcn_target \"amdgcn-amd-amdhsa--gfx906\"\n";
    out += "\t.amdhsa_kernel demo_kernel\n";
    out += "\t\t.amdhsa_private_segment_fixed_size ";
    out += (bad_scratch ? "4096" : "0");
    out += "\n\t.end_amdhsa_kernel\n";
    out += "; ScratchSize: ";
    out += (bad_scratch_marker ? "256" : "0");
    out += "\n\t.amdgpu_metadata\n";
    out += "---\namdhsa.kernels:\n";
    out += "  - .args: []\n";
    out += "    .name:           demo_kernel\n";
    out += "    .private_segment_fixed_size: ";
    out += (bad_scratch ? "4096" : "0");
    out += "\n    .sgpr_spill_count: ";
    out += (bad_sgpr ? "7" : "0");
    out += "\n    .vgpr_spill_count: ";
    out += (bad_vgpr ? "9" : "0");
    out += "\n    .wavefront_size: 64\n";
    out += bad_trig ? "\tv_sin_f32 v1, v2\n" : "\tv_add_f32 v1, v2, v3\n";
    out += bad_slc
        ? "\tglobal_atomic_add v1, v0, v1, s[10:11] glc slc\n"
        : "\tglobal_atomic_add v1, v0, v1, s[10:11] glc\n";
    return out;
}

int run_self_test() {
    std::printf("\n=== W94 ENGINE AUDITOR -- SELF-TEST ===\n");
    std::printf("Synthetic negative injections must be caught; clean controls must "
                "produce zero violations.\n");
    std::printf("\n=== TABLE S: SELF-TEST CASES ===\n");
    std::printf("%-23s %-35s %-9s %-15s %-7s\n",
                "case", "injected violation", "detected", "clean control", "verdict");

    std::vector<SelfTestRow> rows;

    const auto add_case = [&](const std::string& id, const std::string& injected,
                              bool caught, bool clean_ok) {
        SelfTestRow row;
        row.id = id;
        row.injected = injected;
        row.caught = caught;
        row.clean_control = clean_ok;
        rows.push_back(std::move(row));
    };

    // 1. Scratch (private segment) > 0.
    {
        const AsmFileReport bad = scan_asm_text(
            synthetic_asm(true, false, false, false, false, false), "<synthetic>");
        const AsmFileReport good = scan_asm_text(
            synthetic_asm(false, false, false, false, false, false), "<synthetic>");
        add_case("ASM_SCRATCH_GT0",
                 ".amdhsa_private_segment_fixed_size = 4096",
                 violations_contain(bad.violations,
                                    "amdhsa_private_segment_fixed_size"),
                 good.violations.empty() && good.pass);
    }
    // 2. SGPR spill > 0.
    {
        const AsmFileReport bad = scan_asm_text(
            synthetic_asm(false, true, false, false, false, false), "<synthetic>");
        const AsmFileReport good = scan_asm_text(
            synthetic_asm(false, false, false, false, false, false), "<synthetic>");
        add_case("ASM_SGPR_SPILL_GT0",
                 ".sgpr_spill_count: 7",
                 violations_contain(bad.violations, "sgpr_spill_count"),
                 good.violations.empty() && good.pass);
    }
    // 3. VGPR spill > 0.
    {
        const AsmFileReport bad = scan_asm_text(
            synthetic_asm(false, false, true, false, false, false), "<synthetic>");
        const AsmFileReport good = scan_asm_text(
            synthetic_asm(false, false, false, false, false, false), "<synthetic>");
        add_case("ASM_VGPR_SPILL_GT0",
                 ".vgpr_spill_count: 9",
                 violations_contain(bad.violations, "vgpr_spill_count"),
                 good.violations.empty() && good.pass);
    }
    // 4. ScratchSize marker > 0.
    {
        const AsmFileReport bad = scan_asm_text(
            synthetic_asm(false, false, false, true, false, false), "<synthetic>");
        const AsmFileReport good = scan_asm_text(
            synthetic_asm(false, false, false, false, false, false), "<synthetic>");
        add_case("ASM_SCRATCHSIZE_GT0",
                 "; ScratchSize: 256",
                 violations_contain(bad.violations, "ScratchSize"),
                 good.violations.empty() && good.pass);
    }
    // 5. Injected SLC atomic.
    {
        const AsmFileReport bad = scan_asm_text(
            synthetic_asm(false, false, false, false, true, false), "<synthetic>");
        const AsmFileReport good = scan_asm_text(
            synthetic_asm(false, false, false, false, false, false), "<synthetic>");
        add_case("ASM_SLC_ATOMIC",
                 "global_atomic_add ... glc slc",
                 bad.slc_atomic_count == 1u &&
                     violations_contain(bad.violations, "SLC"),
                 good.slc_atomic_count == 0u && good.violations.empty() && good.pass);
    }
    // 6. Injected device trig.
    {
        const AsmFileReport bad = scan_asm_text(
            synthetic_asm(false, false, false, false, false, true), "<synthetic>");
        const AsmFileReport good = scan_asm_text(
            synthetic_asm(false, false, false, false, false, false), "<synthetic>");
        add_case("ASM_TRIG_INSTRUCTION",
                 "v_sin_f32 v1, v2",
                 bad.trig_count == 1u &&
                     violations_contain(bad.violations, "trig instruction"),
                 good.trig_count == 0u && good.violations.empty() && good.pass);
    }
    // 7. Injected system-scope fetch_add + atomicAdd_system.
    {
        const std::string bad_src =
            "__hip_atomic_fetch_add(&ctr, 1u, __ATOMIC_ACQ_REL, "
            "__HIP_MEMORY_SCOPE_SYSTEM);\n"
            "atomicAdd_system(&f, 1.0f);\n";
        const std::vector<RmwHit> bad_hits =
            scan_system_scope_rmw(strip_comments_and_literals(bad_src));

        const std::string good_src =
            "// architectural comment: __hip_atomic_fetch_add(..., "
            "__HIP_MEMORY_SCOPE_SYSTEM)\n"
            "__hip_atomic_fetch_add(&ctr, 1u, __ATOMIC_ACQ_REL, "
            "__HIP_MEMORY_SCOPE_AGENT);\n"
            "atomicAdd(&f, 1.0f);\n"
            "__hip_atomic_load(&g, __ATOMIC_ACQUIRE, "
            "__HIP_MEMORY_SCOPE_SYSTEM);\n";
        const std::vector<RmwHit> good_hits =
            scan_system_scope_rmw(strip_comments_and_literals(good_src));

        add_case("SRC_SYSTEM_SCOPE_FETCH_ADD",
                 "__hip_atomic_fetch_add(..., SCOPE_SYSTEM) + atomicAdd_system",
                 bad_hits.size() == 2u, good_hits.empty());
    }
    // 8. kMaxChunkM drift.
    {
        const std::string bad_src =
            "// stale note: kMaxChunkM = 512\n"
            "static constexpr int kMaxChunkM = 1024;\n";
        const InvariantParse bad = parse_kmaxchunkm(
            strip_comments_and_literals(bad_src), "<synthetic>");
        const InvariantParse good = parse_kmaxchunkm(
            strip_comments_and_literals(
                "static constexpr int kMaxChunkM = 4096;\n"),
            "<synthetic>");
        add_case("INV_KMAXCHUNKM_DRIFT",
                 "kMaxChunkM = 1024 (stale comment 512 ignored)",
                 bad.found && !bad.valid, good.found && good.valid);
    }
    // 9. TokSliceCache capacity drift.
    {
        const std::string bad_src =
            "struct TokSliceCache {\n"
            "    static constexpr size_t kMaxEntries = 4096u;\n"
            "};\n";
        const InvariantParse bad = parse_tokslice_capacity(bad_src, "<synthetic>");
        const InvariantParse good = parse_tokslice_capacity(
            "struct TokSliceCache {\n"
            "    static constexpr size_t kMaxEntries = 8192u;\n"
            "};\n",
            "<synthetic>");
        add_case("INV_TOKSLICE_CAP_DRIFT",
                 "TokSliceCache::kMaxEntries = 4096",
                 bad.found && !bad.valid, good.found && good.valid);
    }
    // 10. Overlay UUID argument count drift.
    {
        const std::string bad_src =
            "inline RadixKey128 compute_model_overlay_uuid(\n"
            "    const std::string& mi50_packed_root,\n"
            "    const std::string& rarebal_overlay_path,\n"
            "    const std::string& expert_perm_overlay_path,\n"
            "    const std::string& hlut3_dev3_overlay_path) noexcept\n";
        const UuidParse bad = parse_overlay_uuid(bad_src, "<synthetic>");
        const std::string good_src =
            "inline RadixKey128 compute_model_overlay_uuid(\n"
            "    const std::string& mi50_packed_root,\n"
            "    const std::string& rarebal_overlay_path,\n"
            "    const std::string& expert_perm_overlay_path,\n"
            "    const std::string& hlut3_dev3_overlay_path,\n"
            "    const std::string& hlut3_tp3_overlay_path) noexcept\n";
        const UuidParse good = parse_overlay_uuid(good_src, "<synthetic>");
        add_case("INV_OVERLAY_UUID_ARGS",
                 "compute_model_overlay_uuid with 4 path args",
                 bad.found && !bad.valid, good.found && good.valid);
    }
    // 11. Check 14 negative: injected bitwise AND in the 4-key allheads guard.
    {
        const std::string bad_src =
            "__global__ void k_sparse_attn_allheads(int x) {\n"
            "    if ((s_valid[t + 0] & s_valid[t + 1] & s_valid[t + 2] & "
            "s_valid[t + 3]) >= 0) { use(t); }\n"
            "}\n";
        const AllheadsGuardParse bad =
            parse_allheads_guard(bad_src, "<synthetic>");
        const std::string good_src =
            "__global__ void k_sparse_attn_allheads(int x) {\n"
            "    if ((s_valid[t + 0] | s_valid[t + 1] | s_valid[t + 2] | "
            "s_valid[t + 3]) >= 0) { use(t); }\n"
            "}\n";
        const AllheadsGuardParse good =
            parse_allheads_guard(good_src, "<synthetic>");
        add_case("GUARD_ALLHEADS_AND_INJECT",
                 "s_valid[t+0..3] guard joined by bitwise AND (&)",
                 bad.found && !bad.valid && bad.bitwise_and_count == 3u,
                 good.found && good.valid && good.or_guard_count == 1u);
    }
    // 12. Check 15 negative: injected HLUT3 tile size drift (constant 416).
    {
        const std::string bad_src =
            "inline constexpr int kHlut3TileBytes = 416;\n"
            "struct Hlut3Tile32x32 { unsigned char b[416]; };\n"
            "static_assert(sizeof(Hlut3Tile32x32) == kHlut3TileBytes, "
            "\"HLUT3 tile\");\n";
        const TileAssertParse bad =
            parse_tile_size_assert(bad_src, "<synthetic>");
        const std::string good_src =
            "inline constexpr int kHlut3TileBytes = 400;\n"
            "struct Hlut3Tile32x32 { unsigned char b[400]; };\n"
            "static_assert(sizeof(Hlut3Tile32x32) == kHlut3TileBytes, "
            "\"HLUT3 tile\");\n";
        const TileAssertParse good =
            parse_tile_size_assert(good_src, "<synthetic>");
        add_case("TILE_HLUT3_SIZE_DRIFT",
                 "static_assert sizeof(Hlut3Tile32x32) == kHlut3TileBytes=416",
                 bad.found && !bad.valid && bad.assert_value == 416ll,
                 good.found && good.valid && good.assert_value == 400ll);
    }

    std::size_t passed = 0u;
    for (const SelfTestRow& r : rows) {
        const bool ok = r.caught && r.clean_control;
        if (ok) ++passed;
        std::printf("%-23s %-35s %-9s %-15s %-7s\n",
                    r.id.c_str(), r.injected.c_str(),
                    r.caught ? "YES" : "NO",
                    r.clean_control ? "clean" : "FALSE-POS",
                    ok ? "PASS" : "FAIL");
    }

    const std::size_t total = rows.size();
    std::printf("\nSELF-TEST SUMMARY: %zu/%zu injected-violation cases PASS\n",
                passed, total);
    std::printf("SELF-TEST OVERALL VERDICT: %s\n", (passed == total) ? "PASS" : "FAIL");
    return (passed == total) ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Live audit
// ---------------------------------------------------------------------------

int run_live_audit(const fs::path& root) {
    std::printf("\n=== W94 ENGINE AUDITOR -- LIVE AUDIT ===\n");
    std::printf("root: %s\n", root.string().c_str());

    std::vector<std::string> errors;
    AsmAudit asm_audit;
    SourceAudit src_audit;

    const bool asm_ran = run_asm_audit(root, asm_audit, errors);
    const bool src_ran = run_source_audit(root, src_audit, errors);

    if (asm_ran) print_asm_table(asm_audit);

    std::vector<CheckRow> rows = build_asm_rows(asm_audit);
    const std::vector<CheckRow> src_rows = build_source_rows(src_audit);
    rows.insert(rows.end(), src_rows.begin(), src_rows.end());

    std::size_t passed = 0u;
    std::size_t total = 0u;
    print_check_table("TABLE 2: STRUCTURED FINDINGS", rows, passed, total);

    if (!errors.empty()) {
        std::printf("\n=== DIAGNOSTICS (%zu) ===\n", errors.size());
        std::size_t shown = 0u;
        for (const std::string& e : errors) {
            if (shown++ >= 40u) {
                std::printf("... %zu more diagnostics suppressed\n",
                            errors.size() - shown + 1u);
                break;
            }
            std::printf("  %s\n", e.c_str());
        }
    }

    const bool overall = asm_ran && src_ran && asm_audit.ok && src_audit.ok &&
                         (passed == total);
    std::printf("\nOVERALL VERDICT: %s (%zu/%zu checks PASS)\n",
                overall ? "PASS" : "FAIL", passed, total);
    return overall ? 0 : 1;
}

}  // namespace w94audit

int main(int argc, char** argv) {
    bool self_test = false;
    std::string root = w94audit::kDefaultRoot;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--self-test") {
            self_test = true;
        } else if (arg == "--root") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "w94_engine_auditor: --root requires a path\n");
                return 2;
            }
            root = argv[++i];
        } else if (arg == "--help") {
            std::printf("usage: w94_engine_auditor [--self-test] [--root PATH]\n"
                        "  default root: %s\n", w94audit::kDefaultRoot);
            return 0;
        } else {
            std::fprintf(stderr, "w94_engine_auditor: unknown argument '%s'\n",
                         arg.c_str());
            return 2;
        }
    }

    if (self_test) return w94audit::run_self_test();
    return w94audit::run_live_audit(w94audit::fs::path(root));
}
