// Pure-C++17 / AMD HIP (gfx906) Native DeepSeek-ViT + 3x3 Pixel-Unshuffle + 2-Layer GELU Aligner
// for DeepSeek-V4.1-Flash-1017 (Wave 8.1 Stage 3).
// Zero Python anywhere.

#include "dsv41_vit.hpp"
#include "../src/safetensors_mmap.hpp"

#include <arpa/inet.h>
#include <dlfcn.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dsv41 {

namespace {

static constexpr int kVitDim        = 1024;
static constexpr int kVitHeads      = 16;
static constexpr int kVitHeadDim    = 64;
static constexpr int kVitRopeDim    = 32;
static constexpr int kVitInterDim   = 2816;
static constexpr int kVitLayers     = 32;
static constexpr int kPatchDim      = 3 * 14 * 14;  // 588
static constexpr int kUnshufDim     = kVitDim * 9;  // 9216
static constexpr int kLlmDim        = 5120;
static constexpr int kMaxVitPatches = 9216;         // up to 96x96 (1344x1344)
static constexpr int kMaxLlmPatches = 1088;         // up to 32x34

inline uint16_t host_f32_to_bf16_rne(float f) noexcept {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(uint32_t));
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
        return static_cast<uint16_t>((u >> 16) | 0x0040u);
    }
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x00007FFFu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

inline float host_bf16_to_f32(uint16_t b) noexcept {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

struct ScopedHipDevice {
    int prev = 0;
    explicit ScopedHipDevice(int target) {
        DSV41_HIP_CHECK(hipGetDevice(&prev));
        if (prev != target) {
            DSV41_HIP_CHECK(hipSetDevice(target));
        }
    }
    ~ScopedHipDevice() {
        (void)hipSetDevice(prev);
    }
};

// ============================================================================
// Pure-C++ FIPS 180-4 SHA-256
// ============================================================================
inline uint32_t rotr32(uint32_t x, uint32_t n) noexcept {
    return (x >> n) | (x << (32u - n));
}

void sha256_transform(uint32_t state[8], const uint8_t block[64]) noexcept {
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
        w[i] = (static_cast<uint32_t>(block[i * 4 + 0]) << 24) |
               (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
               static_cast<uint32_t>(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t temp1 = h + S1 + ch + kK[i] + w[i];
        const uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

inline uint64_t splitmix64_step(uint64_t z) noexcept {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// ============================================================================
// Runtime dlopen Image Decoders (libpng16 + libwebp + libgdk_pixbuf-2.0)
// ============================================================================
struct LibPngApi {
    // Matches verified x86_64 libpng16.so.16 png_image ABI layout (+64B safety tail)
    struct PngImageAbi {
        void*    opaque;
        uint32_t version;
        uint32_t width;
        uint32_t height;
        uint32_t format;
        uint32_t flags;
        uint32_t colormap_entries;
        uint32_t warning_or_error;
        char     message[64];
        uint8_t  abi_guard_pad[64];
    };

    using BeginReadMemFn = int (*)(PngImageAbi*, const void*, size_t);
    using FinishReadFn   = int (*)(PngImageAbi*, const void*, void*, int32_t, void*);
    using WriteToMemFn   = int (*)(PngImageAbi*, void*, size_t*, int, const void*, int32_t, const void*);
    using FreeFn         = void (*)(PngImageAbi*);

    void* handle = nullptr;
    BeginReadMemFn begin_read_mem = nullptr;
    FinishReadFn   finish_read    = nullptr;
    WriteToMemFn   write_to_mem   = nullptr;
    FreeFn         free_img       = nullptr;
    bool ok = false;

    static const LibPngApi& instance() {
        static LibPngApi api = []() {
            LibPngApi a;
            a.handle = ::dlopen("libpng16.so.16", RTLD_LAZY | RTLD_LOCAL);
            if (!a.handle) a.handle = ::dlopen("/usr/lib/x86_64-linux-gnu/libpng16.so.16", RTLD_LAZY | RTLD_LOCAL);
            if (a.handle) {
                a.begin_read_mem = reinterpret_cast<BeginReadMemFn>(::dlsym(a.handle, "png_image_begin_read_from_memory"));
                a.finish_read    = reinterpret_cast<FinishReadFn>(::dlsym(a.handle, "png_image_finish_read"));
                a.write_to_mem   = reinterpret_cast<WriteToMemFn>(::dlsym(a.handle, "png_image_write_to_memory"));
                a.free_img       = reinterpret_cast<FreeFn>(::dlsym(a.handle, "png_image_free"));
                a.ok = (a.begin_read_mem && a.finish_read && a.free_img);
            }
            return a;
        }();
        return api;
    }
};

struct LibWebpApi {
    using DecodeRgbFn = uint8_t* (*)(const uint8_t*, size_t, int*, int*);
    using WebpFreeFn  = void (*)(void*);

    void* handle = nullptr;
    DecodeRgbFn decode_rgb = nullptr;
    WebpFreeFn  webp_free  = nullptr;
    bool ok = false;

    static const LibWebpApi& instance() {
        static LibWebpApi api = []() {
            LibWebpApi a;
            a.handle = ::dlopen("libwebp.so.7", RTLD_LAZY | RTLD_LOCAL);
            if (!a.handle) a.handle = ::dlopen("/usr/lib/x86_64-linux-gnu/libwebp.so.7", RTLD_LAZY | RTLD_LOCAL);
            if (a.handle) {
                a.decode_rgb = reinterpret_cast<DecodeRgbFn>(::dlsym(a.handle, "WebPDecodeRGB"));
                a.webp_free  = reinterpret_cast<WebpFreeFn>(::dlsym(a.handle, "WebPFree"));
                a.ok = (a.decode_rgb && a.webp_free);
            }
            return a;
        }();
        return api;
    }
};

struct LibGdkPixbufApi {
    struct GErrorAbi {
        uint32_t domain;
        int32_t  code;
        char*    message;
    };

    using LoaderNewFn   = void* (*)(void);
    using LoaderWriteFn = int (*)(void*, const uint8_t*, size_t, GErrorAbi**);
    using LoaderCloseFn = int (*)(void*, GErrorAbi**);
    using GetPixbufFn   = void* (*)(void*);
    using GetIntFn      = int (*)(const void*);
    using GetPixelsFn   = const uint8_t* (*)(const void*);
    using UnrefFn       = void (*)(void*);
    using ErrorFreeFn   = void (*)(GErrorAbi*);

    void* gdk_handle  = nullptr;
    void* gobj_handle = nullptr;
    void* glib_handle = nullptr;
    LoaderNewFn   loader_new   = nullptr;
    LoaderWriteFn loader_write = nullptr;
    LoaderCloseFn loader_close = nullptr;
    GetPixbufFn   get_pixbuf   = nullptr;
    GetIntFn      get_width    = nullptr;
    GetIntFn      get_height   = nullptr;
    GetIntFn      get_stride   = nullptr;
    GetIntFn      get_channels = nullptr;
    GetPixelsFn   get_pixels   = nullptr;
    UnrefFn       g_unref      = nullptr;
    ErrorFreeFn   g_err_free   = nullptr;
    bool ok = false;

    static const LibGdkPixbufApi& instance() {
        static LibGdkPixbufApi api = []() {
            LibGdkPixbufApi a;
            a.glib_handle = ::dlopen("libglib-2.0.so.0", RTLD_LAZY | RTLD_GLOBAL);
            if (!a.glib_handle) a.glib_handle = ::dlopen("/usr/lib/x86_64-linux-gnu/libglib-2.0.so.0", RTLD_LAZY | RTLD_GLOBAL);
            a.gobj_handle = ::dlopen("libgobject-2.0.so.0", RTLD_LAZY | RTLD_GLOBAL);
            if (!a.gobj_handle) a.gobj_handle = ::dlopen("/usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0", RTLD_LAZY | RTLD_GLOBAL);
            a.gdk_handle  = ::dlopen("libgdk_pixbuf-2.0.so.0", RTLD_LAZY | RTLD_LOCAL);
            if (!a.gdk_handle) a.gdk_handle = ::dlopen("/usr/lib/x86_64-linux-gnu/libgdk_pixbuf-2.0.so.0", RTLD_LAZY | RTLD_LOCAL);
            if (a.gdk_handle && a.gobj_handle) {
                a.loader_new   = reinterpret_cast<LoaderNewFn>(::dlsym(a.gdk_handle, "gdk_pixbuf_loader_new"));
                a.loader_write = reinterpret_cast<LoaderWriteFn>(::dlsym(a.gdk_handle, "gdk_pixbuf_loader_write"));
                a.loader_close = reinterpret_cast<LoaderCloseFn>(::dlsym(a.gdk_handle, "gdk_pixbuf_loader_close"));
                a.get_pixbuf   = reinterpret_cast<GetPixbufFn>(::dlsym(a.gdk_handle, "gdk_pixbuf_loader_get_pixbuf"));
                a.get_width    = reinterpret_cast<GetIntFn>(::dlsym(a.gdk_handle, "gdk_pixbuf_get_width"));
                a.get_height   = reinterpret_cast<GetIntFn>(::dlsym(a.gdk_handle, "gdk_pixbuf_get_height"));
                a.get_stride   = reinterpret_cast<GetIntFn>(::dlsym(a.gdk_handle, "gdk_pixbuf_get_rowstride"));
                a.get_channels = reinterpret_cast<GetIntFn>(::dlsym(a.gdk_handle, "gdk_pixbuf_get_n_channels"));
                a.get_pixels   = reinterpret_cast<GetPixelsFn>(::dlsym(a.gdk_handle, "gdk_pixbuf_get_pixels"));
                a.g_unref      = reinterpret_cast<UnrefFn>(::dlsym(a.gobj_handle, "g_object_unref"));
                if (a.glib_handle) {
                    a.g_err_free = reinterpret_cast<ErrorFreeFn>(::dlsym(a.glib_handle, "g_error_free"));
                }
                a.ok = (a.loader_new && a.loader_write && a.loader_close && a.get_pixbuf &&
                        a.get_width && a.get_height && a.get_stride && a.get_channels &&
                        a.get_pixels && a.g_unref);
            }
            return a;
        }();
        return api;
    }
};

bool try_decode_ppm_or_bmp(
    const uint8_t* bytes,
    size_t len,
    std::vector<uint8_t>& out_rgb8,
    int& out_w,
    int& out_h)
{
    // Binary PPM (P6) or PGM (P5)
    if (len > 7 && bytes[0] == 'P' && (bytes[1] == '6' || bytes[1] == '5')) {
        const bool is_rgb = (bytes[1] == '6');
        size_t p = 2;
        auto next_int = [&](int& val) -> bool {
            while (p < len) {
                if (bytes[p] == '#') {
                    while (p < len && bytes[p] != '\n') ++p;
                } else if (bytes[p] <= ' ') {
                    ++p;
                } else {
                    break;
                }
            }
            if (p >= len || bytes[p] < '0' || bytes[p] > '9') return false;
            int v = 0;
            while (p < len && bytes[p] >= '0' && bytes[p] <= '9') {
                v = v * 10 + (bytes[p] - '0');
                ++p;
            }
            val = v;
            return true;
        };
        int w = 0, h = 0, maxv = 0;
        if (next_int(w) && next_int(h) && next_int(maxv) && w > 0 && h > 0 && maxv == 255 && p < len) {
            ++p;  // single whitespace byte after maxval
            const size_t need = static_cast<size_t>(w) * static_cast<size_t>(h) * (is_rgb ? 3u : 1u);
            if (p + need <= len) {
                out_w = w;
                out_h = h;
                out_rgb8.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 3u);
                if (is_rgb) {
                    std::memcpy(out_rgb8.data(), bytes + p, need);
                } else {
                    for (size_t i = 0; i < need; ++i) {
                        out_rgb8[i * 3 + 0] = bytes[p + i];
                        out_rgb8[i * 3 + 1] = bytes[p + i];
                        out_rgb8[i * 3 + 2] = bytes[p + i];
                    }
                }
                return true;
            }
        }
    }
    // Uncompressed 24-bit / 32-bit BMP ("BM")
    if (len >= 54 && bytes[0] == 'B' && bytes[1] == 'M') {
        auto rd32 = [&](size_t off) -> uint32_t {
            uint32_t v = 0;
            std::memcpy(&v, bytes + off, 4);
            return v;
        };
        auto rd16 = [&](size_t off) -> uint16_t {
            uint16_t v = 0;
            std::memcpy(&v, bytes + off, 2);
            return v;
        };
        const uint32_t pix_off = rd32(10);
        const int32_t  w       = static_cast<int32_t>(rd32(18));
        const int32_t  h_raw   = static_cast<int32_t>(rd32(22));
        const uint16_t bpp     = rd16(28);
        const uint32_t comp    = rd32(30);
        const bool top_down    = (h_raw < 0);
        const int32_t  h       = top_down ? -h_raw : h_raw;
        if (w > 0 && h > 0 && (bpp == 24 || bpp == 32) && comp == 0 && pix_off < len) {
            const size_t bytes_pp = bpp / 8u;
            const size_t row_stride = ((static_cast<size_t>(w) * bytes_pp + 3u) & ~3u);
            if (pix_off + row_stride * static_cast<size_t>(h) <= len) {
                out_w = w;
                out_h = h;
                out_rgb8.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 3u);
                for (int y = 0; y < h; ++y) {
                    const int sy = top_down ? y : (h - 1 - y);
                    const uint8_t* srow = bytes + pix_off + static_cast<size_t>(sy) * row_stride;
                    uint8_t* drow = out_rgb8.data() + static_cast<size_t>(y) * static_cast<size_t>(w) * 3u;
                    for (int x = 0; x < w; ++x) {
                        drow[x * 3 + 0] = srow[x * bytes_pp + 2];  // R
                        drow[x * 3 + 1] = srow[x * bytes_pp + 1];  // G
                        drow[x * 3 + 2] = srow[x * bytes_pp + 0];  // B
                    }
                }
                return true;
            }
        }
    }
    return false;
}

// ============================================================================
// Exact Pillow Catmull-Rom Bicubic Resizer (ImagingResample.c parity)
// ============================================================================
inline double pil_bicubic_filter(double x) noexcept {
    x = std::fabs(x);
    if (x < 1.0) {
        return ((1.5 * x - 2.5) * x) * x + 1.0;
    }
    if (x < 2.0) {
        return (((-0.5 * x + 2.5) * x - 4.0) * x) + 2.0;
    }
    return 0.0;
}

struct ResampleTapRow {
    int xmin = 0;
    int count = 0;
    std::vector<float> weights;
};

std::vector<ResampleTapRow> build_bicubic_taps(int in_size, int out_size) {
    std::vector<ResampleTapRow> rows(static_cast<size_t>(out_size));
    const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
    const double filterscale = std::max(1.0, scale);
    const double support = 2.0 * filterscale;
    const double inv_fscale = 1.0 / filterscale;

    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (static_cast<double>(xx) + 0.5) * scale;
        int xmin = static_cast<int>(std::floor(center - support + 0.5));
        int xmax = static_cast<int>(std::floor(center + support + 0.5));
        if (xmin < 0) xmin = 0;
        if (xmax > in_size) xmax = in_size;
        const int cnt = std::max(0, xmax - xmin);
        ResampleTapRow& r = rows[static_cast<size_t>(xx)];
        r.xmin = xmin;
        r.count = cnt;
        r.weights.resize(static_cast<size_t>(cnt));
        double wsum = 0.0;
        for (int i = 0; i < cnt; ++i) {
            const double w = pil_bicubic_filter((static_cast<double>(xmin + i) - center + 0.5) * inv_fscale);
            r.weights[static_cast<size_t>(i)] = static_cast<float>(w);
            wsum += w;
        }
        if (wsum != 0.0) {
            const float inv_sum = static_cast<float>(1.0 / wsum);
            for (int i = 0; i < cnt; ++i) r.weights[static_cast<size_t>(i)] *= inv_sum;
        }
    }
    return rows;
}

std::vector<uint8_t> resize_rgb8_pil_bicubic(
    const uint8_t* src,
    int in_w,
    int in_h,
    int out_w,
    int out_h)
{
    if (in_w == out_w && in_h == out_h) {
        return std::vector<uint8_t>(src, src + static_cast<size_t>(in_w) * static_cast<size_t>(in_h) * 3u);
    }

    const auto x_taps = build_bicubic_taps(in_w, out_w);
    const auto y_taps = build_bicubic_taps(in_h, out_h);

    // Pass 1: horizontal resample -> uint8 intermediate (matching Pillow 2-pass uint8 behavior)
    std::vector<uint8_t> horiz(static_cast<size_t>(in_h) * static_cast<size_t>(out_w) * 3u);
    #pragma omp parallel for schedule(static) if (in_h * out_w >= 16384)
    for (int y = 0; y < in_h; ++y) {
        const uint8_t* srow = src + static_cast<size_t>(y) * static_cast<size_t>(in_w) * 3u;
        uint8_t* drow = horiz.data() + static_cast<size_t>(y) * static_cast<size_t>(out_w) * 3u;
        for (int xx = 0; xx < out_w; ++xx) {
            const auto& tap = x_taps[static_cast<size_t>(xx)];
            float sr = 0.0f, sg = 0.0f, sb = 0.0f;
            for (int k = 0; k < tap.count; ++k) {
                const float w = tap.weights[static_cast<size_t>(k)];
                const uint8_t* px = srow + static_cast<size_t>(tap.xmin + k) * 3u;
                sr += w * static_cast<float>(px[0]);
                sg += w * static_cast<float>(px[1]);
                sb += w * static_cast<float>(px[2]);
            }
            drow[xx * 3 + 0] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::floor(sr + 0.5f)), 0, 255));
            drow[xx * 3 + 1] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::floor(sg + 0.5f)), 0, 255));
            drow[xx * 3 + 2] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::floor(sb + 0.5f)), 0, 255));
        }
    }

    // Pass 2: vertical resample -> uint8 output
    std::vector<uint8_t> out(static_cast<size_t>(out_h) * static_cast<size_t>(out_w) * 3u);
    #pragma omp parallel for schedule(static) if (out_h * out_w >= 16384)
    for (int yy = 0; yy < out_h; ++yy) {
        const auto& tap = y_taps[static_cast<size_t>(yy)];
        uint8_t* drow = out.data() + static_cast<size_t>(yy) * static_cast<size_t>(out_w) * 3u;
        for (int x = 0; x < out_w; ++x) {
            float sr = 0.0f, sg = 0.0f, sb = 0.0f;
            for (int k = 0; k < tap.count; ++k) {
                const float w = tap.weights[static_cast<size_t>(k)];
                const uint8_t* px = horiz.data() +
                                    (static_cast<size_t>(tap.xmin + k) * static_cast<size_t>(out_w) + static_cast<size_t>(x)) * 3u;
                sr += w * static_cast<float>(px[0]);
                sg += w * static_cast<float>(px[1]);
                sb += w * static_cast<float>(px[2]);
            }
            drow[x * 3 + 0] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::floor(sr + 0.5f)), 0, 255));
            drow[x * 3 + 1] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::floor(sg + 0.5f)), 0, 255));
            drow[x * 3 + 2] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::floor(sb + 0.5f)), 0, 255));
        }
    }
    return out;
}

std::vector<uint8_t> http_get_binary_bytes(const std::string& url) {
    // Minimal POSIX HTTP/1.1 client for http://host[:port]/path
    if (url.rfind("http://", 0) != 0) {
        throw std::runtime_error("Only http://, data:, or local file paths supported in pure-C++ image loader: " + url);
    }
    const std::string rest = url.substr(7);
    const size_t slash = rest.find('/');
    const std::string host_port = (slash == std::string::npos) ? rest : rest.substr(0, slash);
    const std::string path = (slash == std::string::npos) ? "/" : rest.substr(slash);
    std::string host = host_port;
    std::string port = "80";
    const size_t colon = host_port.find(':');
    if (colon != std::string::npos) {
        host = host_port.substr(0, colon);
        port = host_port.substr(colon + 1);
    }

    struct addrinfo hints {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = nullptr;
    if (::getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res) {
        throw std::runtime_error("Cannot resolve image host: " + host);
    }
    int fd = -1;
    for (auto* p = res; p; p = p->ai_next) {
        fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        struct timeval tv { 10, 0 };
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(res);
    if (fd < 0) {
        throw std::runtime_error("Failed to connect to image URL: " + url);
    }

    const std::string req = "GET " + path + " HTTP/1.1\r\nHost: " + host_port + "\r\nConnection: close\r\n\r\n";
    size_t sent = 0;
    while (sent < req.size()) {
        const ssize_t n = ::send(fd, req.data() + sent, req.size() - sent, 0);
        if (n <= 0) {
            ::close(fd);
            throw std::runtime_error("HTTP send failed for image URL: " + url);
        }
        sent += static_cast<size_t>(n);
    }

    std::vector<uint8_t> raw;
    uint8_t buf[16384];
    for (;;) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        raw.insert(raw.end(), buf, buf + n);
    }
    ::close(fd);

    const std::string view(reinterpret_cast<const char*>(raw.data()), raw.size());
    const size_t hdr_end = view.find("\r\n\r\n");
    if (hdr_end == std::string::npos || view.find(" 200") == std::string::npos) {
        throw std::runtime_error("Non-200 HTTP response loading image URL: " + url);
    }
    return std::vector<uint8_t>(raw.begin() + static_cast<ptrdiff_t>(hdr_end + 4), raw.end());
}

void compute_2d_rope_cos_sin(int n_h, int n_w, std::vector<float>& out_cos, std::vector<float>& out_sin) {
    const int N = n_h * n_w;
    out_cos.resize(static_cast<size_t>(N) * kVitRopeDim);
    out_sin.resize(static_cast<size_t>(N) * kVitRopeDim);

    float inv_freq[16];
    for (int k = 0; k < 16; ++k) {
        inv_freq[k] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * k) / static_cast<float>(kVitRopeDim));
    }
    for (int ph = 0; ph < n_h; ++ph) {
        for (int pw = 0; pw < n_w; ++pw) {
            const size_t base = static_cast<size_t>(ph * n_w + pw) * kVitRopeDim;
            for (int k = 0; k < 16; ++k) {
                const float ah = static_cast<float>(ph) * inv_freq[k];
                const float aw = static_cast<float>(pw) * inv_freq[k];
                out_cos[base + k]      = std::cos(ah);
                out_sin[base + k]      = std::sin(ah);
                out_cos[base + 16 + k] = std::cos(aw);
                out_sin[base + 16 + k] = std::sin(aw);
            }
        }
    }
}

}  // namespace

// ============================================================================
// Public Helpers
// ============================================================================

std::array<uint8_t, 32> sha256_bytes(const uint8_t* data, size_t len) {
    uint32_t state[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    size_t off = 0;
    while (off + 64u <= len) {
        sha256_transform(state, data + off);
        off += 64u;
    }

    uint8_t tail[128]{};
    const size_t rem = len - off;
    if (rem > 0) {
        std::memcpy(tail, data + off, rem);
    }
    tail[rem] = 0x80u;
    const size_t total_blocks = (rem >= 56u) ? 2u : 1u;
    const uint64_t bit_len = static_cast<uint64_t>(len) * 8ULL;
    for (int i = 0; i < 8; ++i) {
        tail[total_blocks * 64u - 1u - static_cast<size_t>(i)] =
            static_cast<uint8_t>((bit_len >> (i * 8)) & 0xFFu);
    }
    sha256_transform(state, tail);
    if (total_blocks == 2u) {
        sha256_transform(state, tail + 64u);
    }

    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i) {
        out[i * 4 + 0] = static_cast<uint8_t>((state[i] >> 24) & 0xFFu);
        out[i * 4 + 1] = static_cast<uint8_t>((state[i] >> 16) & 0xFFu);
        out[i * 4 + 2] = static_cast<uint8_t>((state[i] >> 8) & 0xFFu);
        out[i * 4 + 3] = static_cast<uint8_t>(state[i] & 0xFFu);
    }
    return out;
}

uint64_t fold_sha256_salt(const std::array<uint8_t, 32>& digest) noexcept {
    uint64_t lanes[4]{};
    std::memcpy(lanes, digest.data(), 32);
    uint64_t acc = 0xD5741F1A58081FA1ULL;
    for (int i = 0; i < 4; ++i) {
        acc = splitmix64_step(acc ^ lanes[i]);
    }
    return (acc == 0ULL) ? 0x9E3779B97F4A7C15ULL : acc;
}

std::string sha256_hex(const std::array<uint8_t, 32>& digest) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string s(64, '0');
    for (size_t i = 0; i < 32; ++i) {
        s[i * 2 + 0] = kHex[(digest[i] >> 4) & 0xFu];
        s[i * 2 + 1] = kHex[digest[i] & 0xFu];
    }
    return s;
}

std::vector<uint8_t> decode_base64_bytes(const std::string& b64) {
    static const std::array<int8_t, 256> kLut = []() {
        std::array<int8_t, 256> t{};
        t.fill(-1);
        for (int c = 'A'; c <= 'Z'; ++c) t[static_cast<size_t>(c)] = static_cast<int8_t>(c - 'A');
        for (int c = 'a'; c <= 'z'; ++c) t[static_cast<size_t>(c)] = static_cast<int8_t>(c - 'a' + 26);
        for (int c = '0'; c <= '9'; ++c) t[static_cast<size_t>(c)] = static_cast<int8_t>(c - '0' + 52);
        t[static_cast<size_t>('+')] = 62;
        t[static_cast<size_t>('-')] = 62;  // URL-safe base64 support
        t[static_cast<size_t>('/')] = 63;
        t[static_cast<size_t>('_')] = 63;
        return t;
    }();

    std::vector<uint8_t> out;
    out.reserve((b64.size() * 3u) / 4u + 4u);
    uint32_t val = 0;
    int valb = -8;
    for (unsigned char c : b64) {
        if (c <= ' ') continue;
        if (c == '=') break;
        const int8_t d = kLut[c];
        if (d < 0) {
            throw std::runtime_error("Invalid character in base64 image payload");
        }
        val = (val << 6) | static_cast<uint32_t>(d);
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<uint8_t>((val >> valb) & 0xFFu));
            valb -= 8;
        }
    }
    return out;
}

std::string encode_base64_bytes(const uint8_t* data, size_t len) {
    static constexpr char kEnc[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2u) / 3u) * 4u);
    size_t i = 0;
    while (i + 3u <= len) {
        const uint32_t n = (static_cast<uint32_t>(data[i]) << 16) |
                           (static_cast<uint32_t>(data[i + 1]) << 8) |
                           static_cast<uint32_t>(data[i + 2]);
        out.push_back(kEnc[(n >> 18) & 63u]);
        out.push_back(kEnc[(n >> 12) & 63u]);
        out.push_back(kEnc[(n >> 6) & 63u]);
        out.push_back(kEnc[n & 63u]);
        i += 3u;
    }
    if (i < len) {
        uint32_t n = static_cast<uint32_t>(data[i]) << 16;
        if (i + 1u < len) n |= static_cast<uint32_t>(data[i + 1]) << 8;
        out.push_back(kEnc[(n >> 18) & 63u]);
        out.push_back(kEnc[(n >> 12) & 63u]);
        out.push_back((i + 1u < len) ? kEnc[(n >> 6) & 63u] : '=');
        out.push_back('=');
    }
    return out;
}

Dsv41ImageSource make_image_source_from_base64(const std::string& b64, const std::string& media_type) {
    Dsv41ImageSource s;
    s.raw_bytes = decode_base64_bytes(b64);
    s.media_type = media_type;
    return s;
}

Dsv41ImageSource load_image_source_from_spec(const std::string& spec) {
    if (spec.empty()) {
        throw std::runtime_error("Empty image URL/path specification");
    }
    Dsv41ImageSource s;
    if (spec.rfind("data:", 0) == 0) {
        const size_t comma = spec.find(',');
        if (comma == std::string::npos) {
            throw std::runtime_error("Malformed data: image URI (missing comma)");
        }
        const std::string hdr = spec.substr(5, comma - 5);
        if (hdr.find(";base64") == std::string::npos) {
            throw std::runtime_error("Unsupported non-base64 data: image URI: " + hdr);
        }
        const size_t semi = hdr.find(';');
        s.media_type = (semi == std::string::npos) ? hdr : hdr.substr(0, semi);
        s.raw_bytes = decode_base64_bytes(spec.substr(comma + 1));
        return s;
    }
    if (spec.rfind("http://", 0) == 0) {
        s.raw_bytes = http_get_binary_bytes(spec);
        return s;
    }
    std::string path = spec;
    if (path.rfind("file://", 0) == 0) {
        path = path.substr(7);
    }
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("Cannot open local image file: " + path);
    }
    const std::streamsize sz = ifs.tellg();
    if (sz <= 0) {
        throw std::runtime_error("Empty or unreadable local image file: " + path);
    }
    ifs.seekg(0, std::ios::beg);
    s.raw_bytes.resize(static_cast<size_t>(sz));
    if (!ifs.read(reinterpret_cast<char*>(s.raw_bytes.data()), sz)) {
        throw std::runtime_error("Failed reading local image file: " + path);
    }
    return s;
}

void decode_image_bytes_to_rgb8(
    const uint8_t* bytes,
    size_t len,
    std::vector<uint8_t>& out_rgb8,
    int& out_width,
    int& out_height)
{
    if (!bytes || len < 8) {
        throw std::runtime_error("Image byte buffer too small to decode");
    }

    // 1. PNG via libpng16.so.16 (PNG_FORMAT_RGBA = 3 -> unpremultiplied drop alpha matching PIL convert("RGB"))
    static constexpr uint8_t kPngSig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (std::memcmp(bytes, kPngSig, 8) == 0) {
        const auto& png = LibPngApi::instance();
        if (png.ok) {
            LibPngApi::PngImageAbi img{};
            img.version = 1u;  // PNG_IMAGE_VERSION
            if (png.begin_read_mem(&img, bytes, len) != 0 && img.width > 0 && img.height > 0) {
                const int w = static_cast<int>(img.width);
                const int h = static_cast<int>(img.height);
                img.format = 3u;  // PNG_FORMAT_RGBA
                std::vector<uint8_t> rgba(static_cast<size_t>(w) * static_cast<size_t>(h) * 4u);
                const int ok = png.finish_read(&img, nullptr, rgba.data(), 0, nullptr);
                png.free_img(&img);
                if (ok != 0) {
                    out_width = w;
                    out_height = h;
                    out_rgb8.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 3u);
                    const size_t npix = static_cast<size_t>(w) * static_cast<size_t>(h);
                    for (size_t i = 0; i < npix; ++i) {
                        out_rgb8[i * 3 + 0] = rgba[i * 4 + 0];
                        out_rgb8[i * 3 + 1] = rgba[i * 4 + 1];
                        out_rgb8[i * 3 + 2] = rgba[i * 4 + 2];
                    }
                    return;
                }
            } else {
                png.free_img(&img);
            }
        }
    }

    // 2. WebP via libwebp.so.7
    if (len >= 12 && std::memcmp(bytes, "RIFF", 4) == 0 && std::memcmp(bytes + 8, "WEBP", 4) == 0) {
        const auto& wp = LibWebpApi::instance();
        if (wp.ok) {
            int w = 0, h = 0;
            uint8_t* decoded = wp.decode_rgb(bytes, len, &w, &h);
            if (decoded && w > 0 && h > 0) {
                out_width = w;
                out_height = h;
                const size_t nbytes = static_cast<size_t>(w) * static_cast<size_t>(h) * 3u;
                out_rgb8.assign(decoded, decoded + nbytes);
                wp.webp_free(decoded);
                return;
            }
            if (decoded) wp.webp_free(decoded);
        }
    }

    // 3. Built-in PPM / PGM / uncompressed BMP
    if (try_decode_ppm_or_bmp(bytes, len, out_rgb8, out_width, out_height)) {
        return;
    }

    // 4. General multi-format fallback (JPEG, GIF, TIFF, BMP, PNG, WebP) via libgdk_pixbuf-2.0.so.0
    const auto& gdk = LibGdkPixbufApi::instance();
    if (gdk.ok) {
        static std::mutex gdk_mtx;
        std::lock_guard<std::mutex> lk(gdk_mtx);
        void* loader = gdk.loader_new();
        if (loader) {
            LibGdkPixbufApi::GErrorAbi* err = nullptr;
            const int w_ok = gdk.loader_write(loader, bytes, len, &err);
            if (err && gdk.g_err_free) { gdk.g_err_free(err); err = nullptr; }
            const int c_ok = gdk.loader_close(loader, &err);
            if (err && gdk.g_err_free) { gdk.g_err_free(err); err = nullptr; }
            void* pix = (w_ok && c_ok) ? gdk.get_pixbuf(loader) : nullptr;
            if (pix) {
                const int w = gdk.get_width(pix);
                const int h = gdk.get_height(pix);
                const int stride = gdk.get_stride(pix);
                const int nch = gdk.get_channels(pix);
                const uint8_t* pixels = gdk.get_pixels(pix);
                if (w > 0 && h > 0 && (nch == 3 || nch == 4) && pixels) {
                    out_width = w;
                    out_height = h;
                    out_rgb8.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 3u);
                    for (int y = 0; y < h; ++y) {
                        const uint8_t* srow = pixels + static_cast<size_t>(y) * static_cast<size_t>(stride);
                        uint8_t* drow = out_rgb8.data() + static_cast<size_t>(y) * static_cast<size_t>(w) * 3u;
                        if (nch == 3) {
                            std::memcpy(drow, srow, static_cast<size_t>(w) * 3u);
                        } else {
                            for (int x = 0; x < w; ++x) {
                                drow[x * 3 + 0] = srow[x * 4 + 0];
                                drow[x * 3 + 1] = srow[x * 4 + 1];
                                drow[x * 3 + 2] = srow[x * 4 + 2];
                            }
                        }
                    }
                    gdk.g_unref(loader);
                    return;
                }
            }
            gdk.g_unref(loader);
        }
    }

    throw std::runtime_error("Unsupported or corrupted image payload (" + std::to_string(len) + " bytes)");
}

std::vector<uint8_t> encode_rgb8_to_png_bytes(const uint8_t* rgb8, int width, int height) {
    if (!rgb8 || width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid RGB8 buffer passed to encode_rgb8_to_png_bytes");
    }
    const auto& png = LibPngApi::instance();
    if (!png.ok || !png.write_to_mem) {
        throw std::runtime_error("libpng16.so.16 png_image_write_to_memory unavailable");
    }
    LibPngApi::PngImageAbi img{};
    img.version = 1u;
    img.width   = static_cast<uint32_t>(width);
    img.height  = static_cast<uint32_t>(height);
    img.format  = 2u;  // PNG_FORMAT_RGB
    size_t mem_bytes = 0;
    if (png.write_to_mem(&img, nullptr, &mem_bytes, 0, rgb8, 0, nullptr) == 0 || mem_bytes == 0) {
        png.free_img(&img);
        throw std::runtime_error("png_image_write_to_memory size query failed");
    }
    std::vector<uint8_t> out(mem_bytes);
    if (png.write_to_mem(&img, out.data(), &mem_bytes, 0, rgb8, 0, nullptr) == 0) {
        png.free_img(&img);
        throw std::runtime_error("png_image_write_to_memory encode failed");
    }
    png.free_img(&img);
    out.resize(mem_bytes);
    return out;
}

// ============================================================================
// Exact image_processor.py Grid Planner & Patch Preprocessor
// ============================================================================

Dsv41GridPlan plan_deepseek_vl_grid(
    int orig_width,
    int orig_height,
    int min_pixels,
    int patch_size,
    int downsample_ratio,
    int max_n_token)
{
    if (orig_width <= 0 || orig_height <= 0) {
        throw std::runtime_error("Invalid image dimensions in plan_deepseek_vl_grid");
    }
    int width  = orig_width;
    int height = orig_height;
    const int p = patch_size;

    const int64_t area = static_cast<int64_t>(width) * static_cast<int64_t>(height);
    if (area > 0 && area < static_cast<int64_t>(min_pixels)) {
        const double ratio = std::sqrt(static_cast<double>(min_pixels) / static_cast<double>(area));
        width  = static_cast<int>(static_cast<double>(width) * ratio);
        height = static_cast<int>(static_cast<double>(height) * ratio);
    }

    int best_width  = ((width + p - 1) / p) * p;
    int best_height = ((height + p - 1) / p) * p;

    auto llm_grid = [&](int bh, int bw) -> std::pair<int, int> {
        const int vh = bh / p;
        const int vw = bw / p;
        return {(vh + downsample_ratio - 1) / downsample_ratio,
                (vw + downsample_ratio - 1) / downsample_ratio};
    };
    auto num_img_toks = [&](int lh, int lw) -> int {
        return lh * (lw + 1) + 2;
    };

    auto [n_llm_h, n_llm_w] = llm_grid(best_height, best_width);
    if (num_img_toks(n_llm_h, n_llm_w) > max_n_token) {
        // Exact solve_resize_ratio from image_processor.py:46-58
        const double r = static_cast<double>(height) / static_cast<double>(width);
        const double max_w_float = std::sqrt(static_cast<double>(max_n_token - 2) / r + 0.25) - 0.5;
        const double max_h_float = max_w_float * r;
        const int cell = p * downsample_ratio;
        if (max_w_float < 1.0) {
            best_height = ((max_n_token - 2) / 2) * cell;
            best_width  = cell;
        } else if (max_h_float < 1.0) {
            best_height = cell;
            best_width  = (max_n_token - 3) * cell;
        } else {
            const double beta = std::min(
                std::floor(max_w_float) * static_cast<double>(cell) / static_cast<double>(width),
                std::floor(max_h_float) * static_cast<double>(cell) / static_cast<double>(height));
            best_height = static_cast<int>(std::floor(static_cast<double>(height) * beta / static_cast<double>(p))) * p;
            best_width  = static_cast<int>(std::floor(static_cast<double>(width) * beta / static_cast<double>(p))) * p;
        }
        best_height = std::max(p, best_height);
        best_width  = std::max(p, best_width);
        std::tie(n_llm_h, n_llm_w) = llm_grid(best_height, best_width);
    }

    Dsv41GridPlan plan;
    plan.best_width  = best_width;
    plan.best_height = best_height;
    plan.n_vit_h     = best_height / p;
    plan.n_vit_w     = best_width / p;
    plan.n_llm_h     = n_llm_h;
    plan.n_llm_w     = n_llm_w;
    return plan;
}

std::vector<Dsv41ImgTokenType> build_deepseek_vl_token_types(int n_llm_h, int n_llm_w) {
    std::vector<Dsv41ImgTokenType> types;
    types.reserve(static_cast<size_t>(n_llm_h * (n_llm_w + 1) + 2));
    types.push_back(Dsv41ImgTokenType::kImageStart);
    for (int rh = 0; rh < n_llm_h; ++rh) {
        for (int rw = 0; rw < n_llm_w; ++rw) {
            types.push_back(Dsv41ImgTokenType::kImage);
        }
        types.push_back(Dsv41ImgTokenType::kImageNewline);
    }
    types.push_back(Dsv41ImgTokenType::kImageEnd);
    return types;
}

Dsv41PreparedImage prepare_deepseek_vl_image(const Dsv41ImageSource& src) {
    Dsv41PreparedImage out;

    std::vector<uint8_t> decoded_rgb;
    const uint8_t* rgb_ptr = nullptr;
    int orig_w = 0;
    int orig_h = 0;

    if (!src.raw_bytes.empty()) {
        out.sha256 = sha256_bytes(src.raw_bytes.data(), src.raw_bytes.size());
        decode_image_bytes_to_rgb8(src.raw_bytes.data(), src.raw_bytes.size(), decoded_rgb, orig_w, orig_h);
        rgb_ptr = decoded_rgb.data();
    } else if (!src.rgb8.empty() && src.width > 0 && src.height > 0 &&
               src.rgb8.size() == static_cast<size_t>(src.width) * static_cast<size_t>(src.height) * 3u) {
        out.sha256 = sha256_bytes(src.rgb8.data(), src.rgb8.size());
        rgb_ptr = src.rgb8.data();
        orig_w = src.width;
        orig_h = src.height;
    } else {
        throw std::runtime_error("Dsv41ImageSource has neither raw_bytes nor valid rgb8 buffer");
    }

    out.cache_salt = fold_sha256_salt(out.sha256);
    out.grid = plan_deepseek_vl_grid(orig_w, orig_h);
    out.span_len = out.grid.span_tokens();

    const int bw = out.grid.best_width;
    const int bh = out.grid.best_height;

    // Exact PIL.ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    const double scale = std::min(static_cast<double>(bw) / static_cast<double>(orig_w),
                                  static_cast<double>(bh) / static_cast<double>(orig_h));
    const int new_w = std::max(1, static_cast<int>(std::nearbyint(static_cast<double>(orig_w) * scale)));
    const int new_h = std::max(1, static_cast<int>(std::nearbyint(static_cast<double>(orig_h) * scale)));
    const std::vector<uint8_t> resized = resize_rgb8_pil_bicubic(rgb_ptr, orig_w, orig_h, new_w, new_h);

    std::vector<uint8_t> padded(static_cast<size_t>(bh) * static_cast<size_t>(bw) * 3u, 127u);
    const int pad_left = (bw - new_w) / 2;
    const int pad_top  = (bh - new_h) / 2;
    for (int y = 0; y < new_h && (pad_top + y) < bh; ++y) {
        const int copy_w = std::min(new_w, bw - pad_left);
        if (copy_w > 0) {
            std::memcpy(
                padded.data() + (static_cast<size_t>(pad_top + y) * static_cast<size_t>(bw) + static_cast<size_t>(pad_left)) * 3u,
                resized.data() + static_cast<size_t>(y) * static_cast<size_t>(new_w) * 3u,
                static_cast<size_t>(copy_w) * 3u);
        }
    }

    // Normalize ((u8 / 255.0f - 0.5f) / 0.5f).to(bf16) and extract [n_vit_h * n_vit_w, 3, 14, 14]
    uint16_t norm_lut[256];
    for (int v = 0; v < 256; ++v) {
        const float x = ((static_cast<float>(v) / 255.0f) - 0.5f) / 0.5f;
        norm_lut[v] = host_f32_to_bf16_rne(x);
    }

    const int n_vit_h = out.grid.n_vit_h;
    const int n_vit_w = out.grid.n_vit_w;
    const int N = n_vit_h * n_vit_w;
    out.patches_bf16.resize(static_cast<size_t>(N) * kPatchDim);

    #pragma omp parallel for schedule(static) if (N >= 256)
    for (int idx = 0; idx < N; ++idx) {
        const int vh = idx / n_vit_w;
        const int vw = idx % n_vit_w;
        uint16_t* dst = out.patches_bf16.data() + static_cast<size_t>(idx) * kPatchDim;
        for (int dy = 0; dy < 14; ++dy) {
            const uint8_t* row = padded.data() +
                                 (static_cast<size_t>(vh * 14 + dy) * static_cast<size_t>(bw) +
                                  static_cast<size_t>(vw * 14)) * 3u;
            for (int dx = 0; dx < 14; ++dx) {
                const int pix_off = dy * 14 + dx;
                dst[0 * 196 + pix_off] = norm_lut[row[dx * 3 + 0]];
                dst[1 * 196 + pix_off] = norm_lut[row[dx * 3 + 1]];
                dst[2 * 196 + pix_off] = norm_lut[row[dx * 3 + 2]];
            }
        }
    }

    return out;
}

void expand_deepseek_vl_prompt(
    const std::vector<int32_t>& raw_prompt_ids,
    const std::vector<Dsv41ImageSource>& image_sources,
    std::vector<int32_t>& out_expanded_ids,
    std::vector<uint8_t>& out_image_mask,
    std::vector<Dsv41PreparedImage>& out_prepared_images)
{
    size_t placeholder_cnt = 0;
    for (int32_t t : raw_prompt_ids) {
        if (t == kDsv41ImageTokenId) ++placeholder_cnt;
    }
    if (placeholder_cnt != image_sources.size()) {
        throw std::runtime_error(
            "Image placeholder count mismatch: prompt has " + std::to_string(placeholder_cnt) +
            " <｜deepseek_image｜> tokens, but request supplied " + std::to_string(image_sources.size()) + " images");
    }

    out_prepared_images.clear();
    if (placeholder_cnt == 0) {
        out_expanded_ids = raw_prompt_ids;
        out_image_mask.assign(raw_prompt_ids.size(), 0u);
        return;
    }

    out_prepared_images.resize(placeholder_cnt);
    for (size_t i = 0; i < placeholder_cnt; ++i) {
        out_prepared_images[i] = prepare_deepseek_vl_image(image_sources[i]);
    }

    size_t total_len = raw_prompt_ids.size() - placeholder_cnt;
    for (const auto& img : out_prepared_images) {
        total_len += static_cast<size_t>(img.span_len);
    }

    out_expanded_ids.clear();
    out_expanded_ids.reserve(total_len);
    out_image_mask.clear();
    out_image_mask.reserve(total_len);

    size_t img_idx = 0;
    for (int32_t t : raw_prompt_ids) {
        if (t != kDsv41ImageTokenId) {
            out_expanded_ids.push_back(t);
            out_image_mask.push_back(0u);
        } else {
            auto& img = out_prepared_images[img_idx++];
            img.span_start = static_cast<int>(out_expanded_ids.size());
            out_expanded_ids.insert(out_expanded_ids.end(), static_cast<size_t>(img.span_len), kDsv41ImageTokenId);
            out_image_mask.insert(out_image_mask.end(), static_cast<size_t>(img.span_len), 1u);
        }
    }
}

// ============================================================================
// Dsv41VitEngine (Pinned DDR4 + tp=2 Double-Buffered Zero-Scratch HIP Runtime)
// ============================================================================

struct Dsv41VitEngine::Impl {
    struct BlockRelOffsets {
        size_t norm1_w = 0;
        size_t wqkv_w  = 0;
        size_t wqkv_b  = 0;
        size_t wo_w    = 0;
        size_t wo_b    = 0;
        size_t norm2_w = 0;
        size_t mlp_w1  = 0;
        size_t mlp_w2  = 0;
    };

    int device_id = 2;
    bool is_ready = false;
    mutable std::mutex mtx;

    uint8_t* h_pinned_weights = nullptr;
    size_t   pinned_bytes     = 0;
    uint16_t* h_pinned_out    = nullptr;

    // Host pointers inside h_pinned_weights
    const uint16_t* h_patch_proj_w  = nullptr;
    const uint16_t* h_patch_proj_b  = nullptr;
    const uint16_t* h_vision_norm_w = nullptr;
    const uint16_t* h_image_start   = nullptr;
    const uint16_t* h_image_end     = nullptr;
    const uint16_t* h_image_newline = nullptr;
    const uint8_t*  h_blocks_base[kVitLayers]{};
    BlockRelOffsets blk_rel{};
    size_t          vit_block_bytes = 0;
    const uint16_t* h_aligner_w1_w  = nullptr;
    const uint16_t* h_aligner_w1_b  = nullptr;
    const uint16_t* h_aligner_w2_w  = nullptr;
    const uint16_t* h_aligner_w2_b  = nullptr;

    // Device buffers on tp = device_id
    uint8_t*  d_static_const  = nullptr;
    uint16_t* d_patch_proj_w  = nullptr;
    uint16_t* d_patch_proj_b  = nullptr;
    uint16_t* d_vision_norm_w = nullptr;
    uint16_t* d_aligner_w1_b  = nullptr;
    uint16_t* d_aligner_w2_b  = nullptr;

    static constexpr size_t kStageSlotBytes = 27ULL << 20;  // 27 MiB per double-buffer slot
    uint8_t*  d_w_stage[2]{nullptr, nullptr};
    uint16_t* d_x_bf16        = nullptr;  // [9216, 1024] BF16 (18 MiB)
    uint16_t* d_norm_bf16     = nullptr;  // [9216, 1024] BF16 (18 MiB)
    uint16_t* d_qkv_bf16      = nullptr;  // [3, 9216, 1024] BF16 (54 MiB)
    float*    d_f32_scratch   = nullptr;  // [9216, 3072] FP32 (108 MiB)
    float*    d_cos_sin_f32   = nullptr;  // [2, 9216, 32] FP32 (2.25 MiB)

    hipStream_t compute_stream = nullptr;
    hipStream_t copy_stream    = nullptr;
    hipEvent_t  stage_ready_ev[2]{nullptr, nullptr};
    hipEvent_t  stage_free_ev[2]{nullptr, nullptr};

    int cached_rope_h = 0;
    int cached_rope_w = 0;

    ~Impl() {
        if (!is_ready && !h_pinned_weights && !d_static_const) return;
        ScopedHipDevice guard(device_id);
        for (int s = 0; s < 2; ++s) {
            if (stage_ready_ev[s]) (void)hipEventDestroy(stage_ready_ev[s]);
            if (stage_free_ev[s])  (void)hipEventDestroy(stage_free_ev[s]);
            if (d_w_stage[s])      (void)hipFree(d_w_stage[s]);
        }
        if (compute_stream) (void)hipStreamDestroy(compute_stream);
        if (copy_stream)    (void)hipStreamDestroy(copy_stream);
        if (d_cos_sin_f32)  (void)hipFree(d_cos_sin_f32);
        if (d_f32_scratch)  (void)hipFree(d_f32_scratch);
        if (d_qkv_bf16)     (void)hipFree(d_qkv_bf16);
        if (d_norm_bf16)    (void)hipFree(d_norm_bf16);
        if (d_x_bf16)       (void)hipFree(d_x_bf16);
        if (d_static_const) (void)hipFree(d_static_const);
        if (h_pinned_out)   (void)hipHostFree(h_pinned_out);
        if (h_pinned_weights) (void)hipHostFree(h_pinned_weights);
    }
};

Dsv41VitEngine::Dsv41VitEngine() : impl_(std::make_unique<Impl>()) {}
Dsv41VitEngine::~Dsv41VitEngine() = default;

bool Dsv41VitEngine::ready() const noexcept { return impl_ && impl_->is_ready; }
int Dsv41VitEngine::device_id() const noexcept { return impl_ ? impl_->device_id : -1; }

void Dsv41VitEngine::init(const std::string& packed_vit_safetensors_path, int hip_device_id) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (impl_->is_ready) return;

    const auto t0 = std::chrono::steady_clock::now();
    impl_->device_id = hip_device_id;
    ScopedHipDevice guard(hip_device_id);

    SafetensorsMmap st(packed_vit_safetensors_path);
    const uint64_t end_off = st.at("aligner.w2.bias").end_offset;
    impl_->pinned_bytes = static_cast<size_t>(end_off);

    DSV41_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&impl_->h_pinned_weights),
                                  impl_->pinned_bytes, hipHostMallocDefault));
    DSV41_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&impl_->h_pinned_out),
                                  static_cast<size_t>(kMaxLlmPatches) * kLlmDim * sizeof(uint16_t),
                                  hipHostMallocDefault));

    // Parallel memcpy from mmap into coherent pinned DDR4 so page-table entries are pre-faulted
    const uint8_t* mmap_payload = st.tensor_bytes("vision.patch_embed.proj.weight") -
                                  st.at("vision.patch_embed.proj.weight").begin_offset;
    #pragma omp parallel for schedule(static)
    for (int chunk = 0; chunk < 64; ++chunk) {
        const size_t b0 = (impl_->pinned_bytes * static_cast<size_t>(chunk)) / 64u;
        const size_t b1 = (impl_->pinned_bytes * static_cast<size_t>(chunk + 1)) / 64u;
        if (b1 > b0) {
            std::memcpy(impl_->h_pinned_weights + b0, mmap_payload + b0, b1 - b0);
        }
    }

    auto h_u16 = [&](const std::string& name) -> const uint16_t* {
        return reinterpret_cast<const uint16_t*>(impl_->h_pinned_weights + st.at(name).begin_offset);
    };

    impl_->h_patch_proj_w  = h_u16("vision.patch_embed.proj.weight");
    impl_->h_patch_proj_b  = h_u16("vision.patch_embed.proj.bias");
    impl_->h_vision_norm_w = h_u16("vision.norm.weight");
    impl_->h_image_start   = h_u16("image_start");
    impl_->h_image_end     = h_u16("image_end");
    impl_->h_image_newline = h_u16("image_newline");
    impl_->h_aligner_w1_w  = h_u16("aligner.w1.weight");
    impl_->h_aligner_w1_b  = h_u16("aligner.w1.bias");
    impl_->h_aligner_w2_w  = h_u16("aligner.w2.weight");
    impl_->h_aligner_w2_b  = h_u16("aligner.w2.bias");

    const uint64_t b0_base = st.at("vision.blocks.0.norm1.weight").begin_offset;
    const uint64_t b0_end  = st.at("vision.blocks.0.mlp.w2.weight").end_offset;
    impl_->vit_block_bytes = static_cast<size_t>(b0_end - b0_base);
    impl_->blk_rel.norm1_w = static_cast<size_t>(st.at("vision.blocks.0.norm1.weight").begin_offset - b0_base);
    impl_->blk_rel.wqkv_w  = static_cast<size_t>(st.at("vision.blocks.0.attn.wqkv.weight").begin_offset - b0_base);
    impl_->blk_rel.wqkv_b  = static_cast<size_t>(st.at("vision.blocks.0.attn.wqkv.bias").begin_offset - b0_base);
    impl_->blk_rel.wo_w    = static_cast<size_t>(st.at("vision.blocks.0.attn.wo.weight").begin_offset - b0_base);
    impl_->blk_rel.wo_b    = static_cast<size_t>(st.at("vision.blocks.0.attn.wo.bias").begin_offset - b0_base);
    impl_->blk_rel.norm2_w = static_cast<size_t>(st.at("vision.blocks.0.norm2.weight").begin_offset - b0_base);
    impl_->blk_rel.mlp_w1  = static_cast<size_t>(st.at("vision.blocks.0.mlp.w1.weight").begin_offset - b0_base);
    impl_->blk_rel.mlp_w2  = static_cast<size_t>(st.at("vision.blocks.0.mlp.w2.weight").begin_offset - b0_base);

    for (int l = 0; l < kVitLayers; ++l) {
        const std::string p = "vision.blocks." + std::to_string(l) + ".";
        const uint64_t lb = st.at(p + "norm1.weight").begin_offset;
        const uint64_t le = st.at(p + "mlp.w2.weight").end_offset;
        if (le - lb != impl_->vit_block_bytes) {
            throw std::runtime_error("Non-uniform ViT block stride at layer " + std::to_string(l));
        }
        impl_->h_blocks_base[l] = impl_->h_pinned_weights + lb;
    }

    // Allocate persistent tp = device_id workspace (~255 MiB total -> leaves ~705 MiB free HBM2 on tp=2!)
    static constexpr size_t kStaticBytes = 2ULL << 20;
    DSV41_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&impl_->d_static_const), kStaticBytes));
    for (int s = 0; s < 2; ++s) {
        DSV41_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&impl_->d_w_stage[s]), Impl::kStageSlotBytes));
        DSV41_HIP_CHECK(hipEventCreateWithFlags(&impl_->stage_ready_ev[s], hipEventDisableTiming));
        DSV41_HIP_CHECK(hipEventCreateWithFlags(&impl_->stage_free_ev[s], hipEventDisableTiming));
    }
    DSV41_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&impl_->d_x_bf16),
                              static_cast<size_t>(kMaxVitPatches) * kVitDim * sizeof(uint16_t)));
    DSV41_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&impl_->d_norm_bf16),
                              static_cast<size_t>(kMaxVitPatches) * kVitDim * sizeof(uint16_t)));
    DSV41_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&impl_->d_qkv_bf16),
                              3ULL * static_cast<size_t>(kMaxVitPatches) * kVitDim * sizeof(uint16_t)));
    DSV41_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&impl_->d_f32_scratch),
                              static_cast<size_t>(kMaxVitPatches) * (3 * kVitDim) * sizeof(float)));
    DSV41_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&impl_->d_cos_sin_f32),
                              2ULL * static_cast<size_t>(kMaxVitPatches) * kVitRopeDim * sizeof(float)));
    DSV41_HIP_CHECK(hipStreamCreateWithFlags(&impl_->compute_stream, hipStreamNonBlocking));
    DSV41_HIP_CHECK(hipStreamCreateWithFlags(&impl_->copy_stream, hipStreamNonBlocking));

    // Upload small static tensors (patch_embed, vision.norm, aligner biases) once
    size_t cur = 0;
    auto push_static = [&](const uint16_t* h_src, size_t n_u16) -> uint16_t* {
        cur = (cur + 255u) & ~255u;
        const size_t b = n_u16 * sizeof(uint16_t);
        if (cur + b > kStaticBytes) throw std::runtime_error("d_static_const overflow");
        uint16_t* d_ptr = reinterpret_cast<uint16_t*>(impl_->d_static_const + cur);
        DSV41_HIP_CHECK(hipMemcpyAsync(d_ptr, h_src, b, hipMemcpyHostToDevice, impl_->compute_stream));
        cur += b;
        return d_ptr;
    };
    impl_->d_patch_proj_w  = push_static(impl_->h_patch_proj_w,  kVitDim * kPatchDim);
    impl_->d_patch_proj_b  = push_static(impl_->h_patch_proj_b,  kVitDim);
    impl_->d_vision_norm_w = push_static(impl_->h_vision_norm_w, kVitDim);
    impl_->d_aligner_w1_b  = push_static(impl_->h_aligner_w1_b,  kLlmDim);
    impl_->d_aligner_w2_b  = push_static(impl_->h_aligner_w2_b,  kLlmDim);
    DSV41_HIP_CHECK(hipStreamSynchronize(impl_->compute_stream));

    size_t free_b = 0, total_b = 0;
    DSV41_HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
    impl_->is_ready = true;

    const double dt_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr,
                 "[DSV41_VIT] Native DeepSeek-ViT + 3x3 Aligner READY on MI50 tp=%d in %.2fs "
                 "(pinned_DDR4=%.2f MiB, HBM2_free=%.1f MiB)\n",
                 hip_device_id, dt_s,
                 static_cast<double>(impl_->pinned_bytes) / (1024.0 * 1024.0),
                 static_cast<double>(free_b) / (1024.0 * 1024.0));
}

double Dsv41VitEngine::encode_prepared_image(Dsv41PreparedImage& img, int num_layers) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->is_ready) {
        throw std::runtime_error("Dsv41VitEngine::encode_prepared_image called before init()");
    }
    if (num_layers < 0 || num_layers > kVitLayers) {
        throw std::runtime_error("Invalid num_layers in encode_prepared_image");
    }

    const int n_vit_h = img.grid.n_vit_h;
    const int n_vit_w = img.grid.n_vit_w;
    const int N       = n_vit_h * n_vit_w;
    const int n_llm_h = img.grid.n_llm_h;
    const int n_llm_w = img.grid.n_llm_w;
    const int L       = n_llm_h * n_llm_w;
    if (N <= 0 || N > kMaxVitPatches || L <= 0 || L > kMaxLlmPatches) {
        throw std::runtime_error("Prepared image grid exceeds ViT workspace limits");
    }

    const auto t0 = std::chrono::steady_clock::now();
    ScopedHipDevice guard(impl_->device_id);

    float* d_cos = impl_->d_cos_sin_f32;
    float* d_sin = impl_->d_cos_sin_f32 + static_cast<size_t>(kMaxVitPatches) * kVitRopeDim;
    if (impl_->cached_rope_h != n_vit_h || impl_->cached_rope_w != n_vit_w) {
        std::vector<float> h_cos, h_sin;
        compute_2d_rope_cos_sin(n_vit_h, n_vit_w, h_cos, h_sin);
        DSV41_HIP_CHECK(hipMemcpyAsync(d_cos, h_cos.data(), h_cos.size() * sizeof(float),
                                       hipMemcpyHostToDevice, impl_->compute_stream));
        DSV41_HIP_CHECK(hipMemcpyAsync(d_sin, h_sin.data(), h_sin.size() * sizeof(float),
                                       hipMemcpyHostToDevice, impl_->compute_stream));
        impl_->cached_rope_h = n_vit_h;
        impl_->cached_rope_w = n_vit_w;
    }

    // Upload patches [N, 588] BF16 into d_norm_bf16
    DSV41_HIP_CHECK(hipMemcpyAsync(impl_->d_norm_bf16, img.patches_bf16.data(),
                                   img.patches_bf16.size() * sizeof(uint16_t),
                                   hipMemcpyHostToDevice, impl_->compute_stream));

    // Pre-stage ViT blocks 0 and 1 on copy_stream
    for (int pre = 0; pre < std::min(2, num_layers); ++pre) {
        DSV41_HIP_CHECK(hipMemcpyAsync(impl_->d_w_stage[pre], impl_->h_blocks_base[pre],
                                       impl_->vit_block_bytes, hipMemcpyHostToDevice, impl_->copy_stream));
        DSV41_HIP_CHECK(hipEventRecord(impl_->stage_ready_ev[pre], impl_->copy_stream));
    }

    // 1. PatchEmbed: d_norm_bf16 [N, 588] -> d_x_bf16 [N, 1024]
    launch_bf16_gemm(impl_->d_norm_bf16, impl_->d_patch_proj_w,
                     nullptr, impl_->d_f32_scratch, N, kVitDim, kPatchDim,
                     /*write_f32=*/true, impl_->compute_stream);
    launch_vit_linear_epilogue_bf16(impl_->d_f32_scratch, impl_->d_patch_proj_b,
                                    nullptr, impl_->d_x_bf16, N, kVitDim,
                                    /*apply_exact_gelu=*/false, impl_->compute_stream);

    uint16_t* d_q = impl_->d_qkv_bf16;
    uint16_t* d_k = impl_->d_qkv_bf16 + static_cast<size_t>(kMaxVitPatches) * kVitDim;
    uint16_t* d_v = impl_->d_qkv_bf16 + 2ULL * static_cast<size_t>(kMaxVitPatches) * kVitDim;

    // 2. 32 Bidirectional 2D-RoPE ViT Blocks
    for (int l = 0; l < num_layers; ++l) {
        const int s = l & 1;
        DSV41_HIP_CHECK(hipStreamWaitEvent(impl_->compute_stream, impl_->stage_ready_ev[s], 0));

        const uint8_t* base      = impl_->d_w_stage[s];
        const auto* d_b_norm1_w  = reinterpret_cast<const uint16_t*>(base + impl_->blk_rel.norm1_w);
        const auto* d_b_wqkv_w   = reinterpret_cast<const uint16_t*>(base + impl_->blk_rel.wqkv_w);
        const auto* d_b_wqkv_b   = reinterpret_cast<const uint16_t*>(base + impl_->blk_rel.wqkv_b);
        const auto* d_b_wo_w     = reinterpret_cast<const uint16_t*>(base + impl_->blk_rel.wo_w);
        const auto* d_b_wo_b     = reinterpret_cast<const uint16_t*>(base + impl_->blk_rel.wo_b);
        const auto* d_b_norm2_w  = reinterpret_cast<const uint16_t*>(base + impl_->blk_rel.norm2_w);
        const auto* d_b_mlp_w1   = reinterpret_cast<const uint16_t*>(base + impl_->blk_rel.mlp_w1);
        const auto* d_b_mlp_w2   = reinterpret_cast<const uint16_t*>(base + impl_->blk_rel.mlp_w2);

        // Attention sub-layer
        launch_rmsnorm_bf16(impl_->d_x_bf16, d_b_norm1_w, impl_->d_norm_bf16, N, kVitDim, 1e-6f, impl_->compute_stream);
        launch_bf16_gemm(impl_->d_norm_bf16, d_b_wqkv_w, nullptr, impl_->d_f32_scratch,
                         N, 3 * kVitDim, kVitDim, /*write_f32=*/true, impl_->compute_stream);
        launch_vit_split_qkv_2drope_bf16(impl_->d_f32_scratch, d_b_wqkv_b, d_cos, d_sin,
                                         d_q, d_k, d_v, N, impl_->compute_stream);
        launch_vit_bidir_flash_attn_bf16(d_q, d_k, d_v, impl_->d_norm_bf16, N, impl_->compute_stream);
        launch_bf16_gemm(impl_->d_norm_bf16, d_b_wo_w, nullptr, impl_->d_f32_scratch,
                         N, kVitDim, kVitDim, /*write_f32=*/true, impl_->compute_stream);
        launch_vit_linear_epilogue_bf16(impl_->d_f32_scratch, d_b_wo_b, impl_->d_x_bf16, impl_->d_x_bf16,
                                        N, kVitDim, /*apply_exact_gelu=*/false, impl_->compute_stream);

        // MLP sub-layer (unclamped SwiGLU, bias=False)
        launch_rmsnorm_bf16(impl_->d_x_bf16, d_b_norm2_w, impl_->d_norm_bf16, N, kVitDim, 1e-6f, impl_->compute_stream);
        uint16_t* d_gateup_bf16 = reinterpret_cast<uint16_t*>(impl_->d_f32_scratch);
        launch_bf16_gemm(impl_->d_norm_bf16, d_b_mlp_w1, d_gateup_bf16, nullptr,
                         N, 2 * kVitInterDim, kVitDim, /*write_f32=*/false, impl_->compute_stream);
        launch_vit_swiglu_split_bf16(d_gateup_bf16, impl_->d_qkv_bf16, N, kVitInterDim, impl_->compute_stream);
        launch_bf16_gemm(impl_->d_qkv_bf16, d_b_mlp_w2, nullptr, impl_->d_f32_scratch,
                         N, kVitDim, kVitInterDim, /*write_f32=*/true, impl_->compute_stream);
        launch_vit_linear_epilogue_bf16(impl_->d_f32_scratch, nullptr, impl_->d_x_bf16, impl_->d_x_bf16,
                                        N, kVitDim, /*apply_exact_gelu=*/false, impl_->compute_stream);

        DSV41_HIP_CHECK(hipEventRecord(impl_->stage_free_ev[s], impl_->compute_stream));
        if (l + 2 < num_layers) {
            DSV41_HIP_CHECK(hipStreamWaitEvent(impl_->copy_stream, impl_->stage_free_ev[s], 0));
            DSV41_HIP_CHECK(hipMemcpyAsync(impl_->d_w_stage[s], impl_->h_blocks_base[l + 2],
                                           impl_->vit_block_bytes, hipMemcpyHostToDevice, impl_->copy_stream));
            DSV41_HIP_CHECK(hipEventRecord(impl_->stage_ready_ev[s], impl_->copy_stream));
        }
    }

    // 3. Final vision.norm + 3x3 pixel-unshuffle ([N, 1024] -> [L, 9216])
    launch_rmsnorm_bf16(impl_->d_x_bf16, impl_->d_vision_norm_w, impl_->d_norm_bf16, N, kVitDim, 1e-6f, impl_->compute_stream);
    uint16_t* d_unshuf_bf16 = reinterpret_cast<uint16_t*>(
        reinterpret_cast<uint8_t*>(impl_->d_f32_scratch) + (64ULL << 20));  // +64 MiB offset inside 108 MiB buffer
    launch_vit_unshuffle3x3_bf16(impl_->d_norm_bf16, d_unshuf_bf16, n_vit_h, n_vit_w, kVitDim, impl_->compute_stream);

    // 4. Aligner w1 ([5120, 9216] + bias + exact GELU) streamed in 4x1280 row tiles (22.5 MiB/tile)
    //    + Aligner w2 ([5120, 5120] + bias) streamed in 2x2560 row tiles (25.0 MiB/tile)
    uint16_t* d_a1_full   = impl_->d_qkv_bf16;
    uint16_t* d_a2_full   = d_a1_full + static_cast<size_t>(kMaxLlmPatches) * kLlmDim;
    uint16_t* d_chunk_tmp = d_a2_full + static_cast<size_t>(kMaxLlmPatches) * kLlmDim;

    static constexpr int kW1TileRows = 1280;  // 4 tiles * 1280 = 5120
    const size_t w1_tile_bytes = static_cast<size_t>(kW1TileRows) * kUnshufDim * sizeof(uint16_t);
    for (int t = 0; t < 2; ++t) {
        if (num_layers > 0) {
            // Wait for last ViT blocks using slot t to finish before reusing d_w_stage[t]
            DSV41_HIP_CHECK(hipStreamWaitEvent(impl_->copy_stream, impl_->stage_free_ev[t], 0));
        }
        DSV41_HIP_CHECK(hipMemcpyAsync(
            impl_->d_w_stage[t],
            impl_->h_aligner_w1_w + static_cast<size_t>(t * kW1TileRows) * kUnshufDim,
            w1_tile_bytes, hipMemcpyHostToDevice, impl_->copy_stream));
        DSV41_HIP_CHECK(hipEventRecord(impl_->stage_ready_ev[t], impl_->copy_stream));
    }
    for (int t = 0; t < 4; ++t) {
        const int s  = t & 1;
        const int n0 = t * kW1TileRows;
        DSV41_HIP_CHECK(hipStreamWaitEvent(impl_->compute_stream, impl_->stage_ready_ev[s], 0));
        launch_bf16_gemm(d_unshuf_bf16, reinterpret_cast<const uint16_t*>(impl_->d_w_stage[s]),
                         nullptr, impl_->d_f32_scratch, L, kW1TileRows, kUnshufDim,
                         /*write_f32=*/true, impl_->compute_stream);
        launch_vit_linear_epilogue_bf16(impl_->d_f32_scratch, impl_->d_aligner_w1_b + n0,
                                        nullptr, d_chunk_tmp, L, kW1TileRows,
                                        /*apply_exact_gelu=*/true, impl_->compute_stream);
        DSV41_HIP_CHECK(hipMemcpy2DAsync(
            d_a1_full + n0, kLlmDim * sizeof(uint16_t),
            d_chunk_tmp, kW1TileRows * sizeof(uint16_t),
            kW1TileRows * sizeof(uint16_t), static_cast<size_t>(L),
            hipMemcpyDeviceToDevice, impl_->compute_stream));
        DSV41_HIP_CHECK(hipEventRecord(impl_->stage_free_ev[s], impl_->compute_stream));

        if (t + 2 < 4) {
            DSV41_HIP_CHECK(hipStreamWaitEvent(impl_->copy_stream, impl_->stage_free_ev[s], 0));
            DSV41_HIP_CHECK(hipMemcpyAsync(
                impl_->d_w_stage[s],
                impl_->h_aligner_w1_w + static_cast<size_t>((t + 2) * kW1TileRows) * kUnshufDim,
                w1_tile_bytes, hipMemcpyHostToDevice, impl_->copy_stream));
            DSV41_HIP_CHECK(hipEventRecord(impl_->stage_ready_ev[s], impl_->copy_stream));
        }
    }

    static constexpr int kW2TileRows = 2560;  // 2 tiles * 2560 = 5120
    const size_t w2_tile_bytes = static_cast<size_t>(kW2TileRows) * kLlmDim * sizeof(uint16_t);
    for (int t = 0; t < 2; ++t) {
        DSV41_HIP_CHECK(hipStreamWaitEvent(impl_->copy_stream, impl_->stage_free_ev[t], 0));
        DSV41_HIP_CHECK(hipMemcpyAsync(
            impl_->d_w_stage[t],
            impl_->h_aligner_w2_w + static_cast<size_t>(t * kW2TileRows) * kLlmDim,
            w2_tile_bytes, hipMemcpyHostToDevice, impl_->copy_stream));
        DSV41_HIP_CHECK(hipEventRecord(impl_->stage_ready_ev[t], impl_->copy_stream));
    }
    for (int t = 0; t < 2; ++t) {
        const int n0 = t * kW2TileRows;
        DSV41_HIP_CHECK(hipStreamWaitEvent(impl_->compute_stream, impl_->stage_ready_ev[t], 0));
        launch_bf16_gemm(d_a1_full, reinterpret_cast<const uint16_t*>(impl_->d_w_stage[t]),
                         nullptr, impl_->d_f32_scratch, L, kW2TileRows, kLlmDim,
                         /*write_f32=*/true, impl_->compute_stream);
        launch_vit_linear_epilogue_bf16(impl_->d_f32_scratch, impl_->d_aligner_w2_b + n0,
                                        nullptr, d_chunk_tmp, L, kW2TileRows,
                                        /*apply_exact_gelu=*/false, impl_->compute_stream);
        DSV41_HIP_CHECK(hipMemcpy2DAsync(
            d_a2_full + n0, kLlmDim * sizeof(uint16_t),
            d_chunk_tmp, kW2TileRows * sizeof(uint16_t),
            kW2TileRows * sizeof(uint16_t), static_cast<size_t>(L),
            hipMemcpyDeviceToDevice, impl_->compute_stream));
        DSV41_HIP_CHECK(hipEventRecord(impl_->stage_free_ev[t], impl_->compute_stream));
    }

    // 5. Copy [L, 5120] BF16 Aligner output to host pinned buffer & assemble span [span_len, 5120]
    DSV41_HIP_CHECK(hipMemcpyAsync(impl_->h_pinned_out, d_a2_full,
                                   static_cast<size_t>(L) * kLlmDim * sizeof(uint16_t),
                                   hipMemcpyDeviceToHost, impl_->compute_stream));
    DSV41_HIP_CHECK(hipStreamSynchronize(impl_->compute_stream));

    const int span_len = n_llm_h * (n_llm_w + 1) + 2;
    img.span_len = span_len;
    img.merged_bf16.resize(static_cast<size_t>(span_len) * kLlmDim);

    // [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]
    std::memcpy(img.merged_bf16.data(), impl_->h_image_start, kLlmDim * sizeof(uint16_t));
    size_t dst_row = 1;
    for (int rh = 0; rh < n_llm_h; ++rh) {
        std::memcpy(img.merged_bf16.data() + dst_row * kLlmDim,
                    impl_->h_pinned_out + static_cast<size_t>(rh * n_llm_w) * kLlmDim,
                    static_cast<size_t>(n_llm_w) * kLlmDim * sizeof(uint16_t));
        dst_row += static_cast<size_t>(n_llm_w);
        std::memcpy(img.merged_bf16.data() + dst_row * kLlmDim,
                    impl_->h_image_newline, kLlmDim * sizeof(uint16_t));
        ++dst_row;
    }
    std::memcpy(img.merged_bf16.data() + dst_row * kLlmDim, impl_->h_image_end, kLlmDim * sizeof(uint16_t));

    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// ============================================================================
// Pure-FP32 OpenMP CPU Ground-Truth Reference (for verify_wave8_1_vit)
// ============================================================================

void Dsv41VitEngine::encode_prepared_image_cpu_fp32_ref(
    const Dsv41PreparedImage& img,
    std::vector<float>& out_merged_f32,
    int num_layers) const
{
    if (!impl_ || !impl_->is_ready) {
        throw std::runtime_error("encode_prepared_image_cpu_fp32_ref called before init()");
    }
    const int n_vit_h = img.grid.n_vit_h;
    const int n_vit_w = img.grid.n_vit_w;
    const int N       = n_vit_h * n_vit_w;
    const int n_llm_h = img.grid.n_llm_h;
    const int n_llm_w = img.grid.n_llm_w;
    const int L       = n_llm_h * n_llm_w;

    auto cpu_linear = [](const float* x, const uint16_t* w_bf16, const uint16_t* b_bf16,
                         float* y, int M, int N_out, int K_in) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N_out; ++n) {
                const float* xr = x + static_cast<size_t>(m) * K_in;
                const uint16_t* wr = w_bf16 + static_cast<size_t>(n) * K_in;
                double acc = b_bf16 ? static_cast<double>(host_bf16_to_f32(b_bf16[n])) : 0.0;
                for (int k = 0; k < K_in; ++k) {
                    acc += static_cast<double>(xr[k]) * static_cast<double>(host_bf16_to_f32(wr[k]));
                }
                y[static_cast<size_t>(m) * N_out + n] = static_cast<float>(acc);
            }
        }
    };

    auto cpu_rmsnorm = [](const float* x, const uint16_t* w_bf16, float* y, int M, int D, float eps) {
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; ++m) {
            const float* xr = x + static_cast<size_t>(m) * D;
            float* yr = y + static_cast<size_t>(m) * D;
            double sq = 0.0;
            for (int d = 0; d < D; ++d) sq += static_cast<double>(xr[d]) * static_cast<double>(xr[d]);
            const float rstd = 1.0f / std::sqrt(static_cast<float>(sq / static_cast<double>(D)) + eps);
            for (int d = 0; d < D; ++d) {
                yr[d] = xr[d] * rstd * host_bf16_to_f32(w_bf16[d]);
            }
        }
    };

    std::vector<float> patches_f32(img.patches_bf16.size());
    for (size_t i = 0; i < img.patches_bf16.size(); ++i) {
        patches_f32[i] = host_bf16_to_f32(img.patches_bf16[i]);
    }

    std::vector<float> x(static_cast<size_t>(N) * kVitDim);
    cpu_linear(patches_f32.data(), impl_->h_patch_proj_w, impl_->h_patch_proj_b, x.data(), N, kVitDim, kPatchDim);

    std::vector<float> cos_v, sin_v;
    compute_2d_rope_cos_sin(n_vit_h, n_vit_w, cos_v, sin_v);

    std::vector<float> x_norm(static_cast<size_t>(N) * kVitDim);
    std::vector<float> qkv(static_cast<size_t>(N) * 3 * kVitDim);
    std::vector<float> q(static_cast<size_t>(N) * kVitDim);
    std::vector<float> k(static_cast<size_t>(N) * kVitDim);
    std::vector<float> v(static_cast<size_t>(N) * kVitDim);
    std::vector<float> attn_o(static_cast<size_t>(N) * kVitDim);
    std::vector<float> proj_o(static_cast<size_t>(N) * kVitDim);
    std::vector<float> gateup(static_cast<size_t>(N) * 2 * kVitInterDim);
    std::vector<float> mlp_mid(static_cast<size_t>(N) * kVitInterDim);

    for (int l = 0; l < num_layers; ++l) {
        const uint8_t* base     = impl_->h_blocks_base[l];
        const auto* b_norm1_w   = reinterpret_cast<const uint16_t*>(base + impl_->blk_rel.norm1_w);
        const auto* b_wqkv_w    = reinterpret_cast<const uint16_t*>(base + impl_->blk_rel.wqkv_w);
        const auto* b_wqkv_b    = reinterpret_cast<const uint16_t*>(base + impl_->blk_rel.wqkv_b);
        const auto* b_wo_w      = reinterpret_cast<const uint16_t*>(base + impl_->blk_rel.wo_w);
        const auto* b_wo_b      = reinterpret_cast<const uint16_t*>(base + impl_->blk_rel.wo_b);
        const auto* b_norm2_w   = reinterpret_cast<const uint16_t*>(base + impl_->blk_rel.norm2_w);
        const auto* b_mlp_w1    = reinterpret_cast<const uint16_t*>(base + impl_->blk_rel.mlp_w1);
        const auto* b_mlp_w2    = reinterpret_cast<const uint16_t*>(base + impl_->blk_rel.mlp_w2);

        cpu_rmsnorm(x.data(), b_norm1_w, x_norm.data(), N, kVitDim, 1e-6f);
        cpu_linear(x_norm.data(), b_wqkv_w, b_wqkv_b, qkv.data(), N, 3 * kVitDim, kVitDim);

        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const float* rqkv = qkv.data() + static_cast<size_t>(n) * 3 * kVitDim;
            const float* rc   = cos_v.data() + static_cast<size_t>(n) * kVitRopeDim;
            const float* rs   = sin_v.data() + static_cast<size_t>(n) * kVitRopeDim;
            for (int h = 0; h < kVitHeads; ++h) {
                for (int d = 0; d < 32; ++d) {
                    const int o1 = h * 64 + d;
                    const int o2 = h * 64 + 32 + d;
                    const float q1 = rqkv[o1], q2 = rqkv[o2];
                    const float k1 = rqkv[1024 + o1], k2 = rqkv[1024 + o2];
                    q[static_cast<size_t>(n) * 1024 + o1] = q1 * rc[d] - q2 * rs[d];
                    q[static_cast<size_t>(n) * 1024 + o2] = q2 * rc[d] + q1 * rs[d];
                    k[static_cast<size_t>(n) * 1024 + o1] = k1 * rc[d] - k2 * rs[d];
                    k[static_cast<size_t>(n) * 1024 + o2] = k2 * rc[d] + k1 * rs[d];
                    v[static_cast<size_t>(n) * 1024 + o1] = rqkv[2048 + o1];
                    v[static_cast<size_t>(n) * 1024 + o2] = rqkv[2048 + o2];
                }
            }
        }

        // Exact FP32 Bidirectional SDPA
        #pragma omp parallel for collapse(2) schedule(static)
        for (int h = 0; h < kVitHeads; ++h) {
            for (int qi = 0; qi < N; ++qi) {
                std::vector<float> scores(static_cast<size_t>(N));
                float m_max = -1.0e30f;
                const float* qh = q.data() + static_cast<size_t>(qi) * 1024 + h * 64;
                for (int ki = 0; ki < N; ++ki) {
                    const float* kh = k.data() + static_cast<size_t>(ki) * 1024 + h * 64;
                    float dot = 0.0f;
                    for (int d = 0; d < 64; ++d) dot += qh[d] * kh[d];
                    dot *= 0.125f;
                    scores[static_cast<size_t>(ki)] = dot;
                    if (dot > m_max) m_max = dot;
                }
                float denom = 0.0f;
                for (int ki = 0; ki < N; ++ki) {
                    scores[static_cast<size_t>(ki)] = std::exp(scores[static_cast<size_t>(ki)] - m_max);
                    denom += scores[static_cast<size_t>(ki)];
                }
                const float inv_d = 1.0f / denom;
                float* oh = attn_o.data() + static_cast<size_t>(qi) * 1024 + h * 64;
                for (int d = 0; d < 64; ++d) oh[d] = 0.0f;
                for (int ki = 0; ki < N; ++ki) {
                    const float p = scores[static_cast<size_t>(ki)] * inv_d;
                    const float* vh = v.data() + static_cast<size_t>(ki) * 1024 + h * 64;
                    for (int d = 0; d < 64; ++d) oh[d] += p * vh[d];
                }
            }
        }

        cpu_linear(attn_o.data(), b_wo_w, b_wo_b, proj_o.data(), N, kVitDim, kVitDim);
        for (size_t i = 0; i < x.size(); ++i) x[i] += proj_o[i];

        cpu_rmsnorm(x.data(), b_norm2_w, x_norm.data(), N, kVitDim, 1e-6f);
        cpu_linear(x_norm.data(), b_mlp_w1, nullptr, gateup.data(), N, 2 * kVitInterDim, kVitDim);
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const float* gr = gateup.data() + static_cast<size_t>(n) * 2 * kVitInterDim;
            const float* ur = gr + kVitInterDim;
            float* mr = mlp_mid.data() + static_cast<size_t>(n) * kVitInterDim;
            for (int j = 0; j < kVitInterDim; ++j) {
                const float g = gr[j];
                mr[j] = (g / (1.0f + std::exp(-g))) * ur[j];
            }
        }
        cpu_linear(mlp_mid.data(), b_mlp_w2, nullptr, proj_o.data(), N, kVitDim, kVitInterDim);
        for (size_t i = 0; i < x.size(); ++i) x[i] += proj_o[i];
    }

    cpu_rmsnorm(x.data(), impl_->h_vision_norm_w, x_norm.data(), N, kVitDim, 1e-6f);

    // 3x3 pixel-unshuffle (channel-outer F.unfold order)
    std::vector<float> unshuf(static_cast<size_t>(L) * kUnshufDim, 0.0f);
    for (int rh = 0; rh < n_llm_h; ++rh) {
        for (int rw = 0; rw < n_llm_w; ++rw) {
            float* drow = unshuf.data() + static_cast<size_t>(rh * n_llm_w + rw) * kUnshufDim;
            for (int c = 0; c < kVitDim; ++c) {
                for (int dy = 0; dy < 3; ++dy) {
                    for (int dx = 0; dx < 3; ++dx) {
                        const int sh = rh * 3 + dy;
                        const int sw = rw * 3 + dx;
                        if (sh < n_vit_h && sw < n_vit_w) {
                            drow[c * 9 + dy * 3 + dx] =
                                x_norm[static_cast<size_t>(sh * n_vit_w + sw) * kVitDim + c];
                        }
                    }
                }
            }
        }
    }

    std::vector<float> a1(static_cast<size_t>(L) * kLlmDim);
    std::vector<float> a2(static_cast<size_t>(L) * kLlmDim);
    cpu_linear(unshuf.data(), impl_->h_aligner_w1_w, impl_->h_aligner_w1_b, a1.data(), L, kLlmDim, kUnshufDim);
    for (float& v_el : a1) {
        v_el = 0.5f * v_el * (1.0f + std::erf(v_el * 0.7071067811865475244f));
    }
    cpu_linear(a1.data(), impl_->h_aligner_w2_w, impl_->h_aligner_w2_b, a2.data(), L, kLlmDim, kLlmDim);

    const int span_len = n_llm_h * (n_llm_w + 1) + 2;
    out_merged_f32.resize(static_cast<size_t>(span_len) * kLlmDim);
    auto copy_delim = [&](size_t row, const uint16_t* src_bf16) {
        float* d = out_merged_f32.data() + row * kLlmDim;
        for (int c = 0; c < kLlmDim; ++c) d[c] = host_bf16_to_f32(src_bf16[c]);
    };
    copy_delim(0, impl_->h_image_start);
    size_t dst_row = 1;
    for (int rh = 0; rh < n_llm_h; ++rh) {
        std::memcpy(out_merged_f32.data() + dst_row * kLlmDim,
                    a2.data() + static_cast<size_t>(rh * n_llm_w) * kLlmDim,
                    static_cast<size_t>(n_llm_w) * kLlmDim * sizeof(float));
        dst_row += static_cast<size_t>(n_llm_w);
        copy_delim(dst_row++, impl_->h_image_newline);
    }
    copy_delim(dst_row, impl_->h_image_end);
}

}  // namespace dsv41
