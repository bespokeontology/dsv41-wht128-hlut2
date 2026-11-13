#pragma once
// Pure-C++17 Zero-Dependency safetensors mmap Parser & Writer (Zero Python Anywhere):
//   Format: [8-byte little-endian uint64_t header_len][JSON header][raw tensor payload]
// Supports zero-copy mmap views across BF16, F16, F32, U8, I8, F8_E8M0, F4 (MXFP4) tensors.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace dsv41 {

struct SafetensorEntry {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin_offset = 0;
    uint64_t end_offset = 0;

    uint64_t byte_size() const { return end_offset - begin_offset; }
};

class SafetensorsMmap {
public:
    SafetensorsMmap() = default;
    explicit SafetensorsMmap(const std::string& path) { open(path); }
    ~SafetensorsMmap() { close(); }

    SafetensorsMmap(const SafetensorsMmap&) = delete;
    SafetensorsMmap& operator=(const SafetensorsMmap&) = delete;

    SafetensorsMmap(SafetensorsMmap&& o) noexcept
        : fd_(o.fd_), size_(o.size_), data_base_(o.data_base_), map_(o.map_), entries_(std::move(o.entries_)) {
        o.fd_ = -1;
        o.size_ = 0;
        o.data_base_ = 0;
        o.map_ = nullptr;
    }

    void open(const std::string& path) {
        close();
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("SafetensorsMmap: failed to open " + path);
        struct stat st {};
        if (::fstat(fd_, &st) != 0 || st.st_size < 8) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("SafetensorsMmap: invalid file size for " + path);
        }
        size_ = static_cast<size_t>(st.st_size);
        map_ = static_cast<const uint8_t*>(::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0));
        if (map_ == MAP_FAILED) {
            map_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("SafetensorsMmap: mmap failed for " + path);
        }
        uint64_t hdr_len = 0;
        std::memcpy(&hdr_len, map_, 8);
        if (8ULL + hdr_len > size_) {
            close();
            throw std::runtime_error("SafetensorsMmap: header length exceeds file size in " + path);
        }
        data_base_ = 8ULL + hdr_len;
        parse_header(reinterpret_cast<const char*>(map_ + 8), static_cast<size_t>(hdr_len));
    }

    void close() {
        if (map_) {
            ::munmap(const_cast<uint8_t*>(map_), size_);
            map_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        size_ = 0;
        data_base_ = 0;
        entries_.clear();
    }

    bool contains(const std::string& name) const { return entries_.find(name) != entries_.end(); }

    const SafetensorEntry& at(const std::string& name) const {
        auto it = entries_.find(name);
        if (it == entries_.end()) throw std::runtime_error("SafetensorsMmap: missing tensor " + name);
        return it->second;
    }

    const uint8_t* tensor_bytes(const std::string& name) const {
        const auto& e = at(name);
        if (data_base_ + e.end_offset > size_) {
            throw std::runtime_error("SafetensorsMmap: tensor offset out of bounds for " + name);
        }
        return map_ + data_base_ + e.begin_offset;
    }

    const std::map<std::string, SafetensorEntry>& entries() const { return entries_; }

private:
    int fd_ = -1;
    size_t size_ = 0;
    uint64_t data_base_ = 0;
    const uint8_t* map_ = nullptr;
    std::map<std::string, SafetensorEntry> entries_;

    static void skip_ws(const char* s, size_t n, size_t& p) {
        while (p < n && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
    }

    static std::string parse_string(const char* s, size_t n, size_t& p) {
        skip_ws(s, n, p);
        if (p >= n || s[p] != '"') throw std::runtime_error("SafetensorsMmap: expected '\"'");
        ++p;
        std::string out;
        while (p < n && s[p] != '"') {
            if (s[p] == '\\' && p + 1 < n) {
                ++p;
                switch (s[p]) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default: out.push_back(s[p]); break;
                }
            } else {
                out.push_back(s[p]);
            }
            ++p;
        }
        if (p >= n || s[p] != '"') throw std::runtime_error("SafetensorsMmap: unterminated string");
        ++p;
        return out;
    }

    static uint64_t parse_u64(const char* s, size_t n, size_t& p) {
        skip_ws(s, n, p);
        uint64_t v = 0;
        bool any = false;
        while (p < n && std::isdigit(static_cast<unsigned char>(s[p]))) {
            v = v * 10ULL + uint64_t(s[p] - '0');
            ++p;
            any = true;
        }
        if (!any) throw std::runtime_error("SafetensorsMmap: expected integer");
        return v;
    }

    static void skip_value(const char* s, size_t n, size_t& p) {
        skip_ws(s, n, p);
        if (p >= n) return;
        if (s[p] == '"') {
            (void)parse_string(s, n, p);
        } else if (s[p] == '{') {
            ++p;
            int depth = 1;
            while (p < n && depth > 0) {
                if (s[p] == '"') {
                    (void)parse_string(s, n, p);
                    continue;
                }
                if (s[p] == '{') ++depth;
                else if (s[p] == '}') --depth;
                ++p;
            }
        } else if (s[p] == '[') {
            ++p;
            int depth = 1;
            while (p < n && depth > 0) {
                if (s[p] == '"') {
                    (void)parse_string(s, n, p);
                    continue;
                }
                if (s[p] == '[') ++depth;
                else if (s[p] == ']') --depth;
                ++p;
            }
        } else {
            while (p < n && s[p] != ',' && s[p] != '}' && s[p] != ']') ++p;
        }
    }

    void parse_header(const char* s, size_t n) {
        size_t p = 0;
        skip_ws(s, n, p);
        if (p >= n || s[p] != '{') throw std::runtime_error("SafetensorsMmap: header must start with '{'");
        ++p;
        while (true) {
            skip_ws(s, n, p);
            if (p >= n) break;
            if (s[p] == '}') { ++p; break; }
            std::string key = parse_string(s, n, p);
            skip_ws(s, n, p);
            if (p >= n || s[p] != ':') throw std::runtime_error("SafetensorsMmap: expected ':'");
            ++p;
            skip_ws(s, n, p);
            if (key == "__metadata__") {
                skip_value(s, n, p);
            } else {
                if (p >= n || s[p] != '{') throw std::runtime_error("SafetensorsMmap: tensor entry must be '{'");
                ++p;
                SafetensorEntry ent;
                ent.name = key;
                while (true) {
                    skip_ws(s, n, p);
                    if (p < n && s[p] == '}') { ++p; break; }
                    std::string fkey = parse_string(s, n, p);
                    skip_ws(s, n, p);
                    if (p >= n || s[p] != ':') throw std::runtime_error("SafetensorsMmap: expected ':' in tensor");
                    ++p;
                    skip_ws(s, n, p);
                    if (fkey == "dtype") {
                        ent.dtype = parse_string(s, n, p);
                    } else if (fkey == "shape") {
                        if (p >= n || s[p] != '[') throw std::runtime_error("SafetensorsMmap: shape must be '['");
                        ++p;
                        while (true) {
                            skip_ws(s, n, p);
                            if (p < n && s[p] == ']') { ++p; break; }
                            ent.shape.push_back(static_cast<int64_t>(parse_u64(s, n, p)));
                            skip_ws(s, n, p);
                            if (p < n && s[p] == ',') ++p;
                        }
                    } else if (fkey == "data_offsets") {
                        if (p >= n || s[p] != '[') throw std::runtime_error("SafetensorsMmap: offsets must be '['");
                        ++p;
                        ent.begin_offset = parse_u64(s, n, p);
                        skip_ws(s, n, p);
                        if (p >= n || s[p] != ',') throw std::runtime_error("SafetensorsMmap: expected ',' in offsets");
                        ++p;
                        ent.end_offset = parse_u64(s, n, p);
                        skip_ws(s, n, p);
                        if (p >= n || s[p] != ']') throw std::runtime_error("SafetensorsMmap: expected ']' in offsets");
                        ++p;
                    } else {
                        skip_value(s, n, p);
                    }
                    skip_ws(s, n, p);
                    if (p < n && s[p] == ',') ++p;
                }
                entries_.emplace(ent.name, std::move(ent));
            }
            skip_ws(s, n, p);
            if (p < n && s[p] == ',') ++p;
        }
    }
};

// Materialize a valid standalone .safetensors shard from extracted real DeepSeek-V4.1-Flash-1017
// layers.6.ffn.experts.0 byte ranges so SafetensorsMmap exercises full zero-copy mmap parsing!
inline bool ensure_real_expert0_safetensors(
    const std::string& scales_path_skhynix,
    const std::string& weights_path_greatwall,
    const std::string& out_safetensors_path)
{
    struct stat st {};
    if (::stat(out_safetensors_path.c_str(), &st) == 0 && st.st_size > 18800000) {
        return true;
    }
    std::ifstream fs(scales_path_skhynix, std::ios::binary);
    std::ifstream fw(weights_path_greatwall, std::ios::binary);
    if (!fs || !fw) return false;
    std::vector<uint8_t> sb(1105920), wb(17694720);
    if (!fs.read(reinterpret_cast<char*>(sb.data()), sb.size())) return false;
    if (!fw.read(reinterpret_cast<char*>(wb.data()), wb.size())) return false;

    // 6 tensors: w1.weight, w1.scale, w2.weight, w2.scale, w3.weight, w3.scale
    // Payload layout: [w1.w (5898240)][w1.s (368640)][w2.w (5898240)][w2.s (368640)][w3.w (5898240)][w3.s (368640)]
    std::string hdr =
        "{"
        "\"__metadata__\":{\"source\":\"deepseek-ai/DeepSeek-V4.1-Flash-1017/model-00009-of-00048.safetensors\"},"
        "\"model.layers.6.mlp.experts.0.gate_proj.weight\":{\"dtype\":\"U8\",\"shape\":[2304,2560],\"data_offsets\":[0,5898240]},"
        "\"model.layers.6.mlp.experts.0.gate_proj.weight_scale_inv\":{\"dtype\":\"U8\",\"shape\":[2304,160],\"data_offsets\":[5898240,6266880]},"
        "\"model.layers.6.mlp.experts.0.down_proj.weight\":{\"dtype\":\"U8\",\"shape\":[5120,1152],\"data_offsets\":[6266880,12165120]},"
        "\"model.layers.6.mlp.experts.0.down_proj.weight_scale_inv\":{\"dtype\":\"U8\",\"shape\":[5120,72],\"data_offsets\":[12165120,12533760]},"
        "\"model.layers.6.mlp.experts.0.up_proj.weight\":{\"dtype\":\"U8\",\"shape\":[2304,2560],\"data_offsets\":[12533760,18432000]},"
        "\"model.layers.6.mlp.experts.0.up_proj.weight_scale_inv\":{\"dtype\":\"U8\",\"shape\":[2304,160],\"data_offsets\":[18432000,18800640]}"
        "}";
    while (hdr.size() % 8 != 0) hdr.push_back(' ');
    const uint64_t hdr_len = static_cast<uint64_t>(hdr.size());

    std::ofstream out(out_safetensors_path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(&hdr_len), 8);
    out.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
    out.write(reinterpret_cast<const char*>(wb.data() + 0),        5898240);
    out.write(reinterpret_cast<const char*>(sb.data() + 0),        368640);
    out.write(reinterpret_cast<const char*>(wb.data() + 5898240),  5898240);
    out.write(reinterpret_cast<const char*>(sb.data() + 368640),   368640);
    out.write(reinterpret_cast<const char*>(wb.data() + 11796480), 5898240);
    out.write(reinterpret_cast<const char*>(sb.data() + 737280),   368640);
    return out.good();
}

}  // namespace dsv41
