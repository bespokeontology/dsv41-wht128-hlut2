#pragma once
// Pure-C++17 / AMD HIP (gfx906) Native DeepSeek-ViT + 3x3 Pixel-Unshuffle + 2-Layer GELU Aligner
// for DeepSeek-V4.1-Flash-1017 (Wave 8.1 Stage 3).
//   - Zero Python anywhere: pure-C++ SHA-256, Base64, PNG/WebP/JPEG/BMP/PPM/GIF/TIFF decoding via dlopen
//   - Bit-for-bit image_processor.py plan_image_grid + ImageOps.pad(..., color=(127,127,127)) bicubic resize
//   - Zero HBM2 starvation: 925.58 MiB BF16 ViT+Aligner shard hosted in tp=2 coherent pinned DDR4,
//     double-buffered over copy_stream + compute_stream through a ~254 MiB tp=2 HBM2 workspace!

#include "dsv41_mi50_kernels.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dsv41 {

static constexpr int32_t kDsv41ImageTokenId = 129264;  // <｜deepseek_image｜>

enum class Dsv41ImgTokenType : int8_t {
    kText         = -1,
    kImageStart   = 0,
    kImage        = 1,
    kImageNewline = 2,
    kImageEnd     = 3,
};

struct Dsv41ImageSource {
    // Either compressed image bytes (PNG/JPEG/WebP/BMP/GIF/PPM/TIFF):
    std::vector<uint8_t> raw_bytes;
    std::string media_type;
    // Or pre-decoded RGB8 pixels (used by deterministic synthetic parity tests):
    std::vector<uint8_t> rgb8;
    int width  = 0;
    int height = 0;
};

struct Dsv41GridPlan {
    int best_width  = 0;
    int best_height = 0;
    int n_vit_h     = 0;
    int n_vit_w     = 0;
    int n_llm_h     = 0;
    int n_llm_w     = 0;

    int num_vit_patches() const noexcept { return n_vit_h * n_vit_w; }
    int num_llm_patch_tokens() const noexcept { return n_llm_h * n_llm_w; }
    int span_tokens() const noexcept { return n_llm_h * (n_llm_w + 1) + 2; }
};

struct Dsv41PreparedImage {
    int span_start = 0;  // Start index in expanded prompt_ids
    int span_len   = 0;  // n_llm_h * (n_llm_w + 1) + 2
    Dsv41GridPlan grid{};
    std::array<uint8_t, 32> sha256{};
    uint64_t cache_salt = 0;            // Non-zero 64-bit folded SHA-256 salt for RadixBlockCache & NVMe KV
    std::vector<uint16_t> patches_bf16; // [n_vit_h * n_vit_w, 588] BF16 ([c=0..2][dy=0..13][dx=0..13])
    std::vector<uint16_t> merged_bf16;  // [span_len, 5120] BF16 (populated by Dsv41VitEngine::encode_prepared_image)
};

// Pure-C++ SHA-256 (FIPS 180-4) and non-zero 64-bit salt folder
std::array<uint8_t, 32> sha256_bytes(const uint8_t* data, size_t len);
uint64_t fold_sha256_salt(const std::array<uint8_t, 32>& digest) noexcept;
std::string sha256_hex(const std::array<uint8_t, 32>& digest);

// Pure-C++ Base64 & Image Source Loader (data:image/...;base64,..., file://, local path, http://)
std::vector<uint8_t> decode_base64_bytes(const std::string& b64);
std::string encode_base64_bytes(const uint8_t* data, size_t len);
Dsv41ImageSource load_image_source_from_spec(const std::string& url_or_data_or_path);
Dsv41ImageSource make_image_source_from_base64(const std::string& b64, const std::string& media_type = "");

// Pure-C++ Multi-Format RGB8 Decoder & PNG Encoder (via dlopen libpng16/libwebp/libgdk_pixbuf + built-in PPM/BMP)
void decode_image_bytes_to_rgb8(
    const uint8_t* bytes,
    size_t len,
    std::vector<uint8_t>& out_rgb8,
    int& out_width,
    int& out_height);

std::vector<uint8_t> encode_rgb8_to_png_bytes(
    const uint8_t* rgb8,
    int width,
    int height);

// Exact image_processor.py grid planner, bicubic ImageOps.pad, patch extractor, and prompt span expander
Dsv41GridPlan plan_deepseek_vl_grid(
    int orig_width,
    int orig_height,
    int min_pixels       = 295936,
    int patch_size       = 14,
    int downsample_ratio = 3,
    int max_n_token      = 1024);

std::vector<Dsv41ImgTokenType> build_deepseek_vl_token_types(int n_llm_h, int n_llm_w);

Dsv41PreparedImage prepare_deepseek_vl_image(const Dsv41ImageSource& src);

void expand_deepseek_vl_prompt(
    const std::vector<int32_t>& raw_prompt_ids,
    const std::vector<Dsv41ImageSource>& image_sources,
    std::vector<int32_t>& out_expanded_ids,
    std::vector<uint8_t>& out_image_mask,
    std::vector<Dsv41PreparedImage>& out_prepared_images);

// Native DeepSeek-ViT (32L, 1024d, 16h, 2816i, 2D-RoPE) + 3x3 Unshuffle + 2L GELU Aligner
class Dsv41VitEngine {
public:
    Dsv41VitEngine();
    ~Dsv41VitEngine();

    Dsv41VitEngine(const Dsv41VitEngine&) = delete;
    Dsv41VitEngine& operator=(const Dsv41VitEngine&) = delete;

    void init(const std::string& packed_vit_safetensors_path, int hip_device_id = 2);
    bool ready() const noexcept;
    int device_id() const noexcept;

    // Runs PatchEmbed -> num_layers ViT blocks -> vision.norm -> 3x3 unshuffle -> Aligner -> span merge on GPU
    // Populates img.merged_bf16 ([img.span_len, 5120] BF16) and returns elapsed GPU wall time in ms.
    double encode_prepared_image(Dsv41PreparedImage& img, int num_layers = 32);

    // Exact OpenMP FP32 CPU reference using the loaded BF16 weights (for verify_wave8_1_vit parity checks)
    void encode_prepared_image_cpu_fp32_ref(
        const Dsv41PreparedImage& img,
        std::vector<float>& out_merged_f32,
        int num_layers = 32) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dsv41
