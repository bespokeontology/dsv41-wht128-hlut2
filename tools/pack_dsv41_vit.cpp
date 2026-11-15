// Pure-C++17 DeepSeek-V4.1-Flash-1017 Vision Tower + Aligner + Delimiter Packer (Zero Python)
// Extracts all 266 BF16 multimodal tensors (vision.patch_embed.*, vision.blocks.0..31.*,
// vision.norm.weight, aligner.w1.*, aligner.w2.*, image_start, image_end, image_newline)
// from either raw 1017 shards (--raw-dir: model-00001/00002-of-00048.safetensors) or
// packed_mi50 (--packed-mi50-dir: vision.safetensors + tp0/embed.safetensors) into
// $DSV41_ROOT/packed_vit/dsv41_vit_bf16.safetensors in contiguous execution order
// so each 24.51 MiB ViT block streams in a single coalesced PCIe SDMA burst!
// Do NOT quantize ViT or Aligner to 2-bit: 100% lossless BF16 preserved bit-for-bit.

#include "../src/safetensors_mmap.hpp"

#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct TensorSpec {
    std::string name;
    std::vector<int64_t> expected_shape;
    int shard_group; // 0 = vision/aligner shard, 1 = delimiter/embed shard
};

std::vector<TensorSpec> build_canonical_vit_specs() {
    std::vector<TensorSpec> specs;
    specs.reserve(266);

    // 1. PatchEmbed + Final Norm
    specs.push_back({"vision.patch_embed.proj.weight", {1024, 588}, 0});
    specs.push_back({"vision.patch_embed.proj.bias",   {1024},      0});
    specs.push_back({"vision.norm.weight",             {1024},      0});

    // 2. Boundary delimiters ([5120] BF16)
    specs.push_back({"image_start",   {5120}, 1});
    specs.push_back({"image_end",     {5120}, 1});
    specs.push_back({"image_newline", {5120}, 1});

    // 3. 32 ViT blocks laid out contiguously per block (25,702,400 B/block = 24.5117 MiB/block)
    for (int l = 0; l < 32; ++l) {
        const std::string p = "vision.blocks." + std::to_string(l) + ".";
        specs.push_back({p + "norm1.weight",     {1024},       0});
        specs.push_back({p + "attn.wqkv.weight", {3072, 1024}, 0});
        specs.push_back({p + "attn.wqkv.bias",   {3072},       0});
        specs.push_back({p + "attn.wo.weight",   {1024, 1024}, 0});
        specs.push_back({p + "attn.wo.bias",     {1024},       0});
        specs.push_back({p + "norm2.weight",     {1024},       0});
        specs.push_back({p + "mlp.w1.weight",    {5632, 1024}, 0});
        specs.push_back({p + "mlp.w2.weight",    {1024, 2816}, 0});
    }

    // 4. 2-Layer GELU Aligner (9216 -> 5120 -> 5120)
    specs.push_back({"aligner.w1.weight", {5120, 9216}, 0});
    specs.push_back({"aligner.w1.bias",   {5120},       0});
    specs.push_back({"aligner.w2.weight", {5120, 5120}, 0});
    specs.push_back({"aligner.w2.bias",   {5120},       0});

    return specs;
}

bool file_exists(const std::string& p) {
    struct stat st {};
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

void ensure_dir(const std::string& dir) {
    if (dir.empty()) return;
    ::mkdir(dir.c_str(), 0775);
}

uint16_t f32_to_bf16_rne(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(uint32_t));
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
        return static_cast<uint16_t>((u >> 16) | 0x0040u);
    }
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x00007FFFu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

} // namespace

int main(int argc, char** argv) {
    const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                     ? std::string(std::getenv("DSV41_ROOT"))
                                     : std::string(".");
    std::string raw_dir;
    std::string packed_mi50_dir = root_env + "/packed_mi50";
    std::string out_dir         = root_env + "/packed_vit";
    bool link_check = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--raw-dir" && i + 1 < argc) {
            raw_dir = argv[++i];
        } else if (a == "--packed-mi50-dir" && i + 1 < argc) {
            packed_mi50_dir = argv[++i];
        } else if (a == "--out-dir" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (a == "--link-check") {
            link_check = true;
        } else {
            std::fprintf(stderr, "Usage: %s [--raw-dir <dir>] [--packed-mi50-dir <dir>] [--out-dir <dir>] [--link-check]\n", argv[0]);
            return 2;
        }
    }

    if (link_check) {
        std::printf("[PACK_VIT] link-check OK\n");
        return 0;
    }

    std::string shard0_path;
    std::string shard1_path;
    if (!raw_dir.empty() &&
        file_exists(raw_dir + "/model-00001-of-00048.safetensors") &&
        file_exists(raw_dir + "/model-00002-of-00048.safetensors"))
    {
        shard0_path = raw_dir + "/model-00001-of-00048.safetensors";
        shard1_path = raw_dir + "/model-00002-of-00048.safetensors";
    } else if (file_exists(packed_mi50_dir + "/vision.safetensors") &&
               file_exists(packed_mi50_dir + "/tp0/embed.safetensors"))
    {
        shard0_path = packed_mi50_dir + "/vision.safetensors";
        shard1_path = packed_mi50_dir + "/tp0/embed.safetensors";
    } else {
        std::fprintf(stderr, "[PACK_VIT] ERROR: could not locate source shards in --raw-dir='%s' or --packed-mi50-dir='%s'\n",
                     raw_dir.c_str(), packed_mi50_dir.c_str());
        return 1;
    }

    std::printf("[PACK_VIT] Opening source shards:\n  [0] %s\n  [1] %s\n",
                shard0_path.c_str(), shard1_path.c_str());

    dsv41::SafetensorsMmap st0(shard0_path);
    dsv41::SafetensorsMmap st1(shard1_path);

    const auto specs = build_canonical_vit_specs();
    struct PreparedSlice {
        std::string name;
        std::vector<int64_t> shape;
        const uint8_t* raw_ptr = nullptr;
        std::vector<uint16_t> converted_bf16;
        uint64_t byte_len = 0;
        uint64_t begin_off = 0;
        uint64_t end_off = 0;
    };

    std::vector<PreparedSlice> slices;
    slices.reserve(specs.size());
    uint64_t cur_off = 0;

    for (const auto& sp : specs) {
        const dsv41::SafetensorsMmap& src = st0.contains(sp.name) ? st0 : st1;
        if (!src.contains(sp.name)) {
            std::fprintf(stderr, "[PACK_VIT] ERROR: missing tensor '%s'\n", sp.name.c_str());
            return 1;
        }
        const auto& ent = src.at(sp.name);
        if (ent.shape != sp.expected_shape) {
            std::fprintf(stderr, "[PACK_VIT] ERROR: shape mismatch for '%s'\n", sp.name.c_str());
            return 1;
        }

        uint64_t numel = 1;
        for (int64_t d : ent.shape) numel *= static_cast<uint64_t>(d);
        const uint64_t bf16_bytes = numel * 2ULL;

        PreparedSlice ps;
        ps.name = sp.name;
        ps.shape = ent.shape;
        ps.byte_len = bf16_bytes;
        ps.begin_off = cur_off;
        ps.end_off = cur_off + bf16_bytes;
        cur_off = ps.end_off;

        if (ent.dtype == "BF16") {
            if (ent.byte_size() != bf16_bytes) {
                std::fprintf(stderr, "[PACK_VIT] ERROR: byte size mismatch for '%s'\n", sp.name.c_str());
                return 1;
            }
            ps.raw_ptr = src.tensor_bytes(sp.name);
        } else if (ent.dtype == "F32") {
            if (ent.byte_size() != numel * 4ULL) {
                std::fprintf(stderr, "[PACK_VIT] ERROR: F32 byte size mismatch for '%s'\n", sp.name.c_str());
                return 1;
            }
            const auto* fp = reinterpret_cast<const float*>(src.tensor_bytes(sp.name));
            ps.converted_bf16.resize(numel);
            for (uint64_t k = 0; k < numel; ++k) {
                ps.converted_bf16[k] = f32_to_bf16_rne(fp[k]);
            }
        } else {
            std::fprintf(stderr, "[PACK_VIT] ERROR: unsupported dtype '%s' for '%s'\n",
                         ent.dtype.c_str(), sp.name.c_str());
            return 1;
        }
        slices.push_back(std::move(ps));
    }

    std::ostringstream hdr_ss;
    hdr_ss << "{\"__metadata__\":{\"format\":\"dsv41_vit_bf16_v1\",\"model\":\"DeepSeek-V4.1-Flash-1017\",\"layers\":\"32\"}";
    for (const auto& ps : slices) {
        hdr_ss << ",\"" << ps.name << "\":{\"dtype\":\"BF16\",\"shape\":[";
        for (size_t j = 0; j < ps.shape.size(); ++j) {
            if (j) hdr_ss << ",";
            hdr_ss << ps.shape[j];
        }
        hdr_ss << "],\"data_offsets\":[" << ps.begin_off << "," << ps.end_off << "]}";
    }
    hdr_ss << "}";
    std::string hdr = hdr_ss.str();
    while (hdr.size() % 8u != 0u) hdr.push_back(' ');
    const uint64_t hdr_len = static_cast<uint64_t>(hdr.size());

    ensure_dir(out_dir);
    const std::string out_path = out_dir + "/dsv41_vit_bf16.safetensors";
    const std::string tmp_path = out_path + ".tmp." + std::to_string(::getpid());

    {
        std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
        if (!ofs) {
            std::fprintf(stderr, "[PACK_VIT] ERROR: cannot open '%s' for writing\n", tmp_path.c_str());
            return 1;
        }
        ofs.write(reinterpret_cast<const char*>(&hdr_len), 8);
        ofs.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
        for (const auto& ps : slices) {
            const char* p = ps.raw_ptr
                ? reinterpret_cast<const char*>(ps.raw_ptr)
                : reinterpret_cast<const char*>(ps.converted_bf16.data());
            ofs.write(p, static_cast<std::streamsize>(ps.byte_len));
            if (!ofs) {
                std::fprintf(stderr, "[PACK_VIT] ERROR: write failed on '%s'\n", ps.name.c_str());
                ::unlink(tmp_path.c_str());
                return 1;
            }
        }
        ofs.flush();
    }

    if (::rename(tmp_path.c_str(), out_path.c_str()) != 0) {
        ::unlink(tmp_path.c_str());
        std::fprintf(stderr, "[PACK_VIT] ERROR: rename failed -> '%s'\n", out_path.c_str());
        return 1;
    }

    // Immediate mmap verification of packed shard
    dsv41::SafetensorsMmap verify_st(out_path);
    const uint64_t b0_begin = verify_st.at("vision.blocks.0.norm1.weight").begin_offset;
    const uint64_t b0_end   = verify_st.at("vision.blocks.0.mlp.w2.weight").end_offset;
    const uint64_t block_stride = b0_end - b0_begin;
    for (int l = 0; l < 32; ++l) {
        const std::string p = "vision.blocks." + std::to_string(l) + ".";
        const uint64_t lb = verify_st.at(p + "norm1.weight").begin_offset;
        const uint64_t le = verify_st.at(p + "mlp.w2.weight").end_offset;
        if (le - lb != block_stride || lb != b0_begin + static_cast<uint64_t>(l) * block_stride) {
            std::fprintf(stderr, "[PACK_VIT] ERROR: block %d not contiguous!\n", l);
            return 1;
        }
    }

    std::printf("[PACK_VIT] Packed %zu BF16 tensors (%.2f MiB payload, %.2f MiB/block contiguous) -> %s\n",
                slices.size(),
                double(cur_off) / (1024.0 * 1024.0),
                double(block_stride) / (1024.0 * 1024.0),
                out_path.c_str());
    return 0;
}
