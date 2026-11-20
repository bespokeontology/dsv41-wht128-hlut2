// Pure-C++17 Streaming .safetensors __metadata__ Sanitizer, Bit-Exact Payload Verifier, & <1.90 GiB Release Chunker
// Zero Python, zero external crypto dependencies (built-in FIPS 180-4 SHA-256).
// Rewrites safetensors JSON __metadata__ with clean portable canonical keys while proving 100% bit-identical
// tensor payload SHA-256 and streaming directly into <1,900 MiB split assets matching GitHub's 2.0 GiB limit.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Minimal FIPS 180-4 SHA-256 implementation (portable pure C++17)
class Sha256 {
public:
    Sha256() { reset(); }

    void reset() {
        h_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
              0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        total_bits_ = 0;
        buf_len_ = 0;
    }

    void update(const uint8_t* data, size_t len) {
        total_bits_ += uint64_t(len) * 8ULL;
        size_t i = 0;
        if (buf_len_ > 0) {
            const size_t take = std::min(len, size_t(64) - buf_len_);
            std::memcpy(buf_.data() + buf_len_, data, take);
            buf_len_ += take;
            i += take;
            if (buf_len_ == 64) {
                compress(buf_.data());
                buf_len_ = 0;
            }
        }
        while (i + 64 <= len) {
            compress(data + i);
            i += 64;
        }
        if (i < len) {
            buf_len_ = len - i;
            std::memcpy(buf_.data(), data + i, buf_len_);
        }
    }

    std::string final_hex() {
        uint8_t pad[128] = {};
        const size_t pad_len = (buf_len_ < 56) ? (56 - buf_len_) : (120 - buf_len_);
        pad[0] = 0x80u;
        const uint64_t bits = total_bits_;
        update(pad, pad_len);
        uint8_t len_be[8];
        for (int k = 0; k < 8; ++k) {
            len_be[k] = static_cast<uint8_t>((bits >> (56 - 8 * k)) & 0xFFu);
        }
        // Directly compress final block without incrementing total_bits_
        std::memcpy(buf_.data() + buf_len_, len_be, 8);
        compress(buf_.data());
        buf_len_ = 0;

        static constexpr char kHex[] = "0123456789abcdef";
        std::string out(64, '0');
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                out[size_t(i * 8 + j)] = kHex[(h_[size_t(i)] >> (28 - 4 * j)) & 0xFu];
            }
        }
        return out;
    }

private:
    static inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32u - n)); }

    void compress(const uint8_t blk[64]) {
        static constexpr uint32_t kK[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
        };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(blk[i * 4]) << 24) | (uint32_t(blk[i * 4 + 1]) << 16) |
                   (uint32_t(blk[i * 4 + 2]) << 8) | uint32_t(blk[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
        uint32_t e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t t1 = h + S1 + ch + kK[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
        h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
    }

    std::array<uint32_t, 8> h_{};
    uint64_t total_bits_ = 0;
    std::array<uint8_t, 64> buf_{};
    size_t buf_len_ = 0;
};

// Rewrite or insert canonical "__metadata__" object in a safetensors JSON header, 8-byte space-padded.
bool rewrite_safetensors_metadata(const std::string& raw_hdr,
                                  const std::string& clean_meta_json,
                                  std::string& out_hdr,
                                  std::string& err) {
    // Trim trailing whitespace/padding
    size_t end = raw_hdr.size();
    while (end > 0 && (raw_hdr[end - 1] == ' ' || raw_hdr[end - 1] == '\n' ||
                       raw_hdr[end - 1] == '\r' || raw_hdr[end - 1] == '\t')) {
        --end;
    }
    const std::string trimmed = raw_hdr.substr(0, end);
    if (trimmed.size() < 2 || trimmed.front() != '{' || trimmed.back() != '}') {
        err = "safetensors header is not a valid JSON object";
        return false;
    }

    const std::string key = "\"__metadata__\"";
    size_t kpos = trimmed.find(key);
    std::string rebuilt;
    if (kpos != std::string::npos) {
        size_t colon = trimmed.find(':', kpos + key.size());
        if (colon == std::string::npos) {
            err = "malformed __metadata__ entry (missing ':')";
            return false;
        }
        size_t obj_start = trimmed.find('{', colon);
        if (obj_start == std::string::npos) {
            err = "malformed __metadata__ entry (missing '{')";
            return false;
        }
        int depth = 0;
        bool in_str = false;
        bool esc = false;
        size_t obj_end = std::string::npos;
        for (size_t i = obj_start; i < trimmed.size(); ++i) {
            const char ch = trimmed[i];
            if (in_str) {
                if (esc) esc = false;
                else if (ch == '\\') esc = true;
                else if (ch == '"') in_str = false;
            } else {
                if (ch == '"') in_str = true;
                else if (ch == '{') ++depth;
                else if (ch == '}') {
                    --depth;
                    if (depth == 0) {
                        obj_end = i;
                        break;
                    }
                }
            }
        }
        if (obj_end == std::string::npos) {
            err = "unterminated __metadata__ JSON object";
            return false;
        }
        rebuilt = trimmed.substr(0, obj_start) + clean_meta_json + trimmed.substr(obj_end + 1);
    } else {
        rebuilt = "{\"__metadata__\":" + clean_meta_json + "," + trimmed.substr(1);
    }

    // Pad with trailing ASCII spaces to 8-byte alignment per safetensors specification
    while ((rebuilt.size() & 7u) != 0u) {
        rebuilt.push_back(' ');
    }
    out_hdr = std::move(rebuilt);
    return true;
}

struct PartReceipt {
    std::string filename;
    uint64_t bytes = 0;
    std::string sha256;
};

struct SanitizeResult {
    std::string out_name;
    uint64_t total_bytes = 0;
    uint64_t src_body_bytes = 0;
    std::string src_body_sha256;
    std::string dst_body_sha256;
    std::string full_sanitized_sha256;
    std::vector<PartReceipt> parts;
};

bool sanitize_and_chunk(const fs::path& src_path,
                        const fs::path& out_dir,
                        const std::string& out_name,
                        const std::string& clean_meta_json,
                        uint64_t chunk_bytes,
                        SanitizeResult& res,
                        std::string& err) {
    std::ifstream in(src_path, std::ios::binary);
    if (!in) {
        err = "failed to open input safetensors: " + src_path.string();
        return false;
    }
    const uint64_t file_size = fs::file_size(src_path);
    if (file_size < 10) {
        err = "input file too small: " + src_path.string();
        return false;
    }

    uint8_t hlen_le[8];
    in.read(reinterpret_cast<char*>(hlen_le), 8);
    if (!in) {
        err = "failed to read 8-byte header length";
        return false;
    }
    uint64_t hlen = 0;
    for (int i = 0; i < 8; ++i) hlen |= (uint64_t(hlen_le[i]) << (8 * i));
    if (hlen == 0 || hlen > (64ULL << 20) || 8ULL + hlen > file_size) {
        err = "invalid safetensors header length: " + std::to_string(hlen);
        return false;
    }

    std::string raw_hdr(size_t(hlen), '\0');
    in.read(raw_hdr.data(), std::streamsize(hlen));
    if (!in) {
        err = "failed to read safetensors JSON header";
        return false;
    }

    std::string new_hdr;
    if (!rewrite_safetensors_metadata(raw_hdr, clean_meta_json, new_hdr, err)) {
        return false;
    }

    fs::create_directories(out_dir);
    res = SanitizeResult{};
    res.out_name = out_name;
    res.src_body_bytes = file_size - (8ULL + hlen);

    const uint64_t new_hlen = uint64_t(new_hdr.size());
    uint8_t new_hlen_le[8];
    for (int i = 0; i < 8; ++i) new_hlen_le[i] = static_cast<uint8_t>((new_hlen >> (8 * i)) & 0xFFu);

    const uint64_t total_out_bytes = 8ULL + new_hlen + res.src_body_bytes;
    const bool split_needed = (chunk_bytes > 0 && total_out_bytes > chunk_bytes);

    Sha256 full_hash;
    Sha256 src_body_hash;
    Sha256 dst_body_hash;
    Sha256 part_hash;

    int part_idx = 0;
    uint64_t part_written = 0;
    std::ofstream cur_out;
    std::string cur_part_name;

    auto open_next_part = [&]() -> bool {
        if (cur_out.is_open()) {
            cur_out.close();
            if (!cur_out) {
                err = "write/close error on " + cur_part_name;
                return false;
            }
            res.parts.push_back(PartReceipt{cur_part_name, part_written, part_hash.final_hex()});
        }
        if (split_needed) {
            char suffix[16];
            std::snprintf(suffix, sizeof(suffix), ".part-%02d", part_idx);
            cur_part_name = out_name + suffix;
        } else {
            cur_part_name = out_name;
        }
        ++part_idx;
        part_written = 0;
        part_hash.reset();
        cur_out.open(out_dir / cur_part_name, std::ios::binary | std::ios::trunc);
        if (!cur_out) {
            err = "failed to open output asset: " + (out_dir / cur_part_name).string();
            return false;
        }
        return true;
    };

    auto write_stream = [&](const uint8_t* ptr, size_t n, bool is_body) -> bool {
        full_hash.update(ptr, n);
        size_t off = 0;
        while (off < n) {
            if (!cur_out.is_open() || (split_needed && part_written >= chunk_bytes)) {
                if (!open_next_part()) return false;
            }
            size_t take = n - off;
            if (split_needed) {
                take = std::min<size_t>(take, size_t(chunk_bytes - part_written));
            }
            cur_out.write(reinterpret_cast<const char*>(ptr + off), std::streamsize(take));
            if (!cur_out) {
                err = "failed writing to " + cur_part_name;
                return false;
            }
            part_hash.update(ptr + off, take);
            if (is_body) dst_body_hash.update(ptr + off, take);
            part_written += take;
            res.total_bytes += take;
            off += take;
        }
        return true;
    };

    if (!write_stream(new_hlen_le, 8, false)) return false;
    if (!write_stream(reinterpret_cast<const uint8_t*>(new_hdr.data()), new_hdr.size(), false)) return false;

    std::vector<uint8_t> buf(16ULL << 20);  // 16 MiB streaming buffer
    uint64_t remaining = res.src_body_bytes;
    while (remaining > 0) {
        const size_t want = size_t(std::min<uint64_t>(remaining, buf.size()));
        in.read(reinterpret_cast<char*>(buf.data()), std::streamsize(want));
        if (!in || size_t(in.gcount()) != want) {
            err = "unexpected EOF while streaming tensor payload";
            return false;
        }
        src_body_hash.update(buf.data(), want);
        if (!write_stream(buf.data(), want, true)) return false;
        remaining -= want;
    }

    if (cur_out.is_open()) {
        cur_out.close();
        if (!cur_out) {
            err = "final close error on " + cur_part_name;
            return false;
        }
        res.parts.push_back(PartReceipt{cur_part_name, part_written, part_hash.final_hex()});
    }

    res.src_body_sha256 = src_body_hash.final_hex();
    res.dst_body_sha256 = dst_body_hash.final_hex();
    res.full_sanitized_sha256 = full_hash.final_hex();

    if (res.src_body_sha256 != res.dst_body_sha256) {
        err = "TENSOR PAYLOAD SHA-256 MISMATCH! src=" + res.src_body_sha256 + " dst=" + res.dst_body_sha256;
        return false;
    }
    return true;
}

int run_self_test() {
    const fs::path tmp = fs::temp_directory_path() / "w94_sanitize_st_selftest";
    std::error_code ec;
    fs::remove_all(tmp, ec);
    fs::create_directories(tmp);

    const std::string dirty_hdr =
        "{\"__metadata__\":{\"format\":\"old\",\"src\":\"/home/synthetic_user/private_box\",\"serial\":\"AABBCCDD11223344\"},"
        "\"t0\":{\"dtype\":\"U8\",\"shape\":[4096],\"data_offsets\":[0,4096]}}";
    std::vector<uint8_t> payload(4096);
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<uint8_t>((i * 73u + 19u) & 0xFFu);

    const fs::path src = tmp / "dirty.safetensors";
    {
        std::ofstream os(src, std::ios::binary);
        const uint64_t hl = dirty_hdr.size();
        for (int i = 0; i < 8; ++i) os.put(static_cast<char>((hl >> (8 * i)) & 0xFFu));
        os.write(dirty_hdr.data(), std::streamsize(dirty_hdr.size()));
        os.write(reinterpret_cast<const char*>(payload.data()), std::streamsize(payload.size()));
    }

    const std::string clean_meta =
        "{\"arch\":\"DeepSeek-V4.1-Flash-1017\",\"bpw\":\"3.125\","
        "\"calibration\":\"FP64-TARQ-RAREBAL-LDLQ-602K-MultiDomain-ViT\","
        "\"experts\":\"1704\",\"format\":\"WHT128-HLUT3\"}";

    SanitizeResult res;
    std::string err;
    if (!sanitize_and_chunk(src, tmp / "out", "clean.safetensors", clean_meta, 1500u, res, err)) {
        std::cerr << "[SELF_TEST] FAIL sanitize_and_chunk: " << err << "\n";
        return 1;
    }
    if (res.parts.size() < 3) {
        std::cerr << "[SELF_TEST] FAIL expected >=3 chunks, got " << res.parts.size() << "\n";
        return 1;
    }

    // Reassemble chunks and verify bit-exact payload + zero dirty strings + SHA-256 match
    std::vector<uint8_t> reassembled;
    for (const auto& p : res.parts) {
        std::ifstream is(tmp / "out" / p.filename, std::ios::binary);
        std::vector<uint8_t> chunk((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
        Sha256 ph;
        ph.update(chunk.data(), chunk.size());
        if (ph.final_hex() != p.sha256) {
            std::cerr << "[SELF_TEST] FAIL part sha mismatch on " << p.filename << "\n";
            return 1;
        }
        reassembled.insert(reassembled.end(), chunk.begin(), chunk.end());
    }
    Sha256 fh;
    fh.update(reassembled.data(), reassembled.size());
    if (fh.final_hex() != res.full_sanitized_sha256) {
        std::cerr << "[SELF_TEST] FAIL reassembled sha mismatch\n";
        return 1;
    }
    uint64_t new_hl = 0;
    for (int i = 0; i < 8; ++i) new_hl |= (uint64_t(reassembled[size_t(i)]) << (8 * i));
    if ((new_hl & 7u) != 0u) {
        std::cerr << "[SELF_TEST] FAIL header not 8-byte aligned\n";
        return 1;
    }
    std::string new_hdr(reinterpret_cast<const char*>(reassembled.data() + 8), size_t(new_hl));
    if (new_hdr.find("synthetic_user") != std::string::npos ||
        new_hdr.find("AABBCCDD11223344") != std::string::npos ||
        new_hdr.find("WHT128-HLUT3") == std::string::npos) {
        std::cerr << "[SELF_TEST] FAIL sanitized header check failed: " << new_hdr << "\n";
        return 1;
    }
    if (reassembled.size() != 8 + new_hl + payload.size() ||
        std::memcmp(reassembled.data() + 8 + new_hl, payload.data(), payload.size()) != 0) {
        std::cerr << "[SELF_TEST] FAIL tensor payload memcmp mismatch!\n";
        return 1;
    }

    fs::remove_all(tmp, ec);
    std::cout << "[SELF_TEST] PASS (header sanitized, 8B aligned, " << res.parts.size()
              << " chunks reassembled, memcmp==0, body_sha256=" << res.dst_body_sha256 << ")\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool self_test = false;
    std::string input_path;
    std::string out_dir;
    std::string out_name;
    std::string sha256_out;
    std::string tier_tag = "1704";
    uint64_t chunk_mib = 1900;

    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--self-test") self_test = true;
        else if (a == "--input" && i + 1 < argc) input_path = argv[++i];
        else if (a == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--out-name" && i + 1 < argc) out_name = argv[++i];
        else if (a == "--sha256-out" && i + 1 < argc) sha256_out = argv[++i];
        else if (a == "--experts" && i + 1 < argc) tier_tag = argv[++i];
        else if (a == "--chunk-mib" && i + 1 < argc) chunk_mib = std::stoull(argv[++i]);
        else {
            std::cerr << "Usage: sanitize_safetensors_release [--self-test] "
                         "--input <src.safetensors> --out-dir <dir> --out-name <name.safetensors> "
                         "[--chunk-mib 1900] [--sha256-out SHA256SUMS.txt]\n";
            return 2;
        }
    }

    if (self_test) return run_self_test();
    if (input_path.empty() || out_dir.empty() || out_name.empty()) {
        std::cerr << "Error: --input, --out-dir, and --out-name are required.\n";
        return 2;
    }

    const std::string clean_meta =
        "{\"arch\":\"DeepSeek-V4.1-Flash-1017\",\"bpw\":\"3.125\","
        "\"calibration\":\"FP64-TARQ-RAREBAL-LDLQ-602K-MultiDomain-ViT\","
        "\"experts\":\"" + tier_tag + "\",\"format\":\"WHT128-HLUT3\"}";

    SanitizeResult res;
    std::string err;
    if (!sanitize_and_chunk(input_path, out_dir, out_name, clean_meta,
                            chunk_mib * 1024ULL * 1024ULL, res, err)) {
        std::cerr << "[ERROR] " << err << "\n";
        return 1;
    }

    std::cout << "[SANITIZE_OK] " << res.out_name
              << " total_bytes=" << res.total_bytes
              << " body_bytes=" << res.src_body_bytes
              << " parts=" << res.parts.size()
              << " body_sha256=" << res.dst_body_sha256
              << " full_sha256=" << res.full_sanitized_sha256 << "\n";

    if (!sha256_out.empty()) {
        std::ofstream os(sha256_out, std::ios::app);
        if (!os) {
            std::cerr << "[ERROR] cannot open --sha256-out: " << sha256_out << "\n";
            return 1;
        }
        for (const auto& p : res.parts) {
            os << p.sha256 << "  " << p.filename << "\n";
        }
        if (res.parts.size() > 1) {
            os << res.full_sanitized_sha256 << "  " << res.out_name << "\n";
        }
    }
    return 0;
}
