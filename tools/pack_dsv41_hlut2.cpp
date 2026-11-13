// Pure-C++17 Streaming 48-Shard Packer for DeepSeek-V4.1-Flash-1017 (Zero Python Anywhere)
// Watches $DSV41_ROOT/shard_status/model-XXXXX-of-00048.safetensors.ready,
// converts backbone + DSpark shards (00001..00046) on the fly across EPYC AVX2 threads into:
//   - $DSV41_ROOT/packed_mi50/tp{0,1,2}/*.safetensors  (3x MI50 HBM2)
//   - $DSV41_ROOT/packed_p150/tp{0,1}/*.safetensors    (2x Blackhole P150A GDDR6)
// Verifies every output shard via SafetensorsMmap + inline Layer Expert 0 SwiGLU cosine check,
// and unlinks raw 00001..00046 .safetensors shards immediately while keeping Engram 00047 & 00048 intact!

#include "../src/dsv41_pack_format.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace dsv41;

namespace {

void mkdir_p(const std::string& p) {
    std::error_code ec;
    std::filesystem::create_directories(p, ec);
}

bool file_exists(const std::string& p) {
    struct stat st {};
    return ::stat(p.c_str(), &st) == 0;
}

std::string read_first_line(const std::string& p) {
    std::ifstream in(p);
    std::string line;
    std::getline(in, line);
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) line.pop_back();
    return line;
}

// Slice 2D row-major tensor along rows [r_off .. r_off + r_len) (zero-copy pointer + byte size!)
struct BorrowedSlice {
    const uint8_t* ptr = nullptr;
    uint64_t nbytes = 0;
    std::vector<int64_t> shape;
    std::string dtype;
};

BorrowedSlice slice_rows_borrowed(const SafetensorsMmap& st, const std::string& name, int64_t r_off, int64_t r_len) {
    const auto& e = st.at(name);
    if (e.shape.empty() || r_off < 0 || r_off + r_len > e.shape[0]) {
        throw std::runtime_error("slice_rows_borrowed out of range on " + name);
    }
    const uint64_t row_bytes = e.byte_size() / uint64_t(e.shape[0]);
    BorrowedSlice b;
    b.ptr = st.tensor_bytes(name) + uint64_t(r_off) * row_bytes;
    b.nbytes = uint64_t(r_len) * row_bytes;
    b.shape = e.shape;
    b.shape[0] = r_len;
    b.dtype = e.dtype;
    return b;
}

// Slice 2D row-major tensor along columns [c_off .. c_off + c_len) (strided copy into owned buffer)
std::vector<uint8_t> slice_cols_2d(
    const SafetensorsMmap& st, const std::string& name, int64_t c_off, int64_t c_len,
    std::vector<int64_t>& out_shape, std::string& out_dtype)
{
    const auto& e = st.at(name);
    if (e.shape.size() != 2 || c_off < 0 || c_off + c_len > e.shape[1]) {
        throw std::runtime_error("slice_cols_2d out of range on " + name);
    }
    const int64_t rows = e.shape[0];
    const int64_t cols = e.shape[1];
    const uint64_t elem_bytes = e.byte_size() / uint64_t(rows * cols);
    const uint64_t src_stride = uint64_t(cols) * elem_bytes;
    const uint64_t dst_stride = uint64_t(c_len) * elem_bytes;

    std::vector<uint8_t> out(uint64_t(rows) * dst_stride);
    const uint8_t* src = st.tensor_bytes(name) + uint64_t(c_off) * elem_bytes;
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        std::memcpy(out.data() + uint64_t(r) * dst_stride, src + uint64_t(r) * src_stride, dst_stride);
    }
    out_shape = {rows, c_len};
    out_dtype = e.dtype;
    return out;
}

void add_replicated(StreamingSafetensorsWriter& w, const SafetensorsMmap& st, const std::string& src_name, const std::string& dst_name) {
    const auto& e = st.at(src_name);
    w.add_borrowed(dst_name, e.dtype, e.shape, st.tensor_bytes(src_name), e.byte_size());
}

void add_row_shard(
    StreamingSafetensorsWriter& w, const SafetensorsMmap& st,
    const std::string& src_name, const std::string& dst_name, int64_t r_off, int64_t r_len)
{
    auto b = slice_rows_borrowed(st, src_name, r_off, r_len);
    w.add_borrowed(dst_name, b.dtype, b.shape, b.ptr, b.nbytes);
}

void add_col_shard_2d(
    StreamingSafetensorsWriter& w, const SafetensorsMmap& st,
    const std::string& src_name, const std::string& dst_name, int64_t c_off, int64_t c_len)
{
    std::vector<int64_t> shp;
    std::string dt;
    auto buf = slice_cols_2d(st, src_name, c_off, c_len, shp, dt);
    w.add_owned(dst_name, dt, shp, std::move(buf));
}

// pack_hlut2_mi50_slot_st & pack_hlut2_p150_slot_st live in dsv41_pack_format.hpp

// Inline real-weight SwiGLU verification on Expert 0 of every layer
QualityMetrics verify_expert0_swiglu_inline(
    const SafetensorsMmap& st,
    const std::string& pfx,
    const Hlut2QuantMatrix& q_w1,
    const Hlut2QuantMatrix& q_w3,
    const Hlut2QuantMatrix& q_w2,
    const std::vector<uint64_t>& signs_h,
    const std::vector<uint64_t>& signs_i)
{
    static constexpr float kE2M1[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    auto deq_mat = [&](const std::string& wname, const std::string& sname, int N, int K) {
        const uint8_t* wb = st.tensor_bytes(wname);
        const uint8_t* sb = st.tensor_bytes(sname);
        std::vector<float> out(size_t(N) * K);
        for (int r = 0; r < N; ++r) {
            for (int g = 0; g < K / 32; ++g) {
                const uint8_t e8 = sb[size_t(r) * (K / 32) + g];
                const uint32_t bits = (e8 == 0) ? 0x00400000u : (uint32_t(e8) << 23);
                float sc = 0.0f;
                std::memcpy(&sc, &bits, 4);
                for (int b = 0; b < 16; ++b) {
                    const uint8_t byte = wb[size_t(r) * (K / 2) + g * 16 + b];
                    out[size_t(r) * K + g * 32 + 2 * b + 0] = kE2M1[byte & 0x0Fu] * sc;
                    out[size_t(r) * K + g * 32 + 2 * b + 1] = kE2M1[byte >> 4] * sc;
                }
            }
        }
        return out;
    };

    const auto w1_ref = deq_mat(pfx + ".ffn.experts.0.w1.weight", pfx + ".ffn.experts.0.w1.scale", 2304, 5120);
    const auto w3_ref = deq_mat(pfx + ".ffn.experts.0.w3.weight", pfx + ".ffn.experts.0.w3.scale", 2304, 5120);
    const auto w2_ref = deq_mat(pfx + ".ffn.experts.0.w2.weight", pfx + ".ffn.experts.0.w2.scale", 5120, 2304);

    std::vector<float> w1_rot, w3_rot, w2_rot;
    dequantize_hlut2_to_rot_f32(q_w1, w1_rot);
    dequantize_hlut2_to_rot_f32(q_w3, w3_rot);
    dequantize_hlut2_to_rot_f32(q_w2, w2_rot);

    const int M = 4;
    const auto X = make_outlier_activations(M, 5120, 20261111ULL);
    std::vector<float> X_rot(size_t(M) * 5120);
    wht128_rows(X.data(), X_rot.data(), M, 5120, signs_h);

    std::vector<float> y_ref(size_t(M) * 5120, 0.0f);
    std::vector<float> y_hlut(size_t(M) * 5120, 0.0f);

    for (int m = 0; m < M; ++m) {
        std::vector<float> z_ref(2304), z_hlut(2304), z_hlut_rot(2304);
        for (int i = 0; i < 2304; ++i) {
            double g0 = 0.0, u0 = 0.0, g1 = 0.0, u1 = 0.0;
            for (int k = 0; k < 5120; ++k) {
                g0 += double(w1_ref[size_t(i) * 5120 + k]) * double(X[size_t(m) * 5120 + k]);
                u0 += double(w3_ref[size_t(i) * 5120 + k]) * double(X[size_t(m) * 5120 + k]);
                g1 += double(w1_rot[size_t(i) * 5120 + k]) * double(X_rot[size_t(m) * 5120 + k]);
                u1 += double(w3_rot[size_t(i) * 5120 + k]) * double(X_rot[size_t(m) * 5120 + k]);
            }
            z_ref[i] = dsv41_swiglu_scalar(float(g0), float(u0));
            z_hlut[i] = dsv41_swiglu_scalar(float(g1), float(u1));
        }
        wht128_rows(z_hlut.data(), z_hlut_rot.data(), 1, 2304, signs_i);
        for (int h = 0; h < 5120; ++h) {
            double o0 = 0.0, o1 = 0.0;
            for (int i = 0; i < 2304; ++i) {
                o0 += double(w2_ref[size_t(h) * 2304 + i]) * double(z_ref[i]);
                o1 += double(w2_rot[size_t(h) * 2304 + i]) * double(z_hlut_rot[i]);
            }
            y_ref[size_t(m) * 5120 + h] = float(o0);
            y_hlut[size_t(m) * 5120 + h] = float(o1);
        }
    }
    return compute_quality(y_ref, y_hlut);
}

void pack_moe_block_shard(
    int shard_idx,
    const std::string& raw_path,
    const std::string& mi50_root,
    const std::string& p150_root,
    bool is_mtp)
{
    const auto t0 = std::chrono::steady_clock::now();
    SafetensorsMmap st(raw_path);

    const int local_id = is_mtp ? (shard_idx - 44) : (shard_idx - 3);
    const int global_layer_id = is_mtp ? (40 + local_id) : local_id;
    const std::string pfx = is_mtp ? ("mtp." + std::to_string(local_id)) : ("layers." + std::to_string(local_id));
    char tag_buf[32];
    if (is_mtp) std::snprintf(tag_buf, sizeof(tag_buf), "mtp_%d", local_id);
    else        std::snprintf(tag_buf, sizeof(tag_buf), "layer_%02d", local_id);
    const std::string tag(tag_buf);

    const int total_experts = is_mtp ? 128 : 384;
    const int mi50_experts = is_mtp ? 128 : Dsv41Tp3Split::mi50_routed_backbone;
    const int p150_experts = is_mtp ? 0 : Dsv41Tp3Split::p150_routed_backbone;

    const auto signs_h = make_wht_signs(5120, layer_wht_seed_h(global_layer_id));
    const auto signs_i = make_wht_signs(2304, layer_wht_seed_i(global_layer_id));

    // Preallocate MI50 TP3 HLUT2 expert banks:
    //   gate/up per card: N=768 (nb=24), K=5120 (kw=320, G=40)
    //     codes:  [mi50_experts, 24, 320, 32] uint32 = 245,760 uint32/expert (983,040 B)
    //     scales: [mi50_experts, 24, 40, 32]  uint16 = 30,720 uint16/expert  (61,440 B)
    //   down per card: N=5120 (nb=160), K=768 (kw=48, G=6)
    //     codes:  [mi50_experts, 160, 48, 32] uint32 = 245,760 uint32/expert (983,040 B)
    //     scales: [mi50_experts, 160, 6, 32]  uint16 = 30,720 uint16/expert  (61,440 B)
    constexpr size_t kMi50CodesWords = 245760ULL;
    constexpr size_t kMi50ScalesHalfs = 30720ULL;

    std::array<std::vector<uint32_t>, 3> mi50_gate_codes, mi50_up_codes, mi50_down_codes;
    std::array<std::vector<uint16_t>, 3> mi50_gate_scales, mi50_up_scales, mi50_down_scales;
    std::array<std::vector<float>, 3> mi50_gate_cent, mi50_up_cent, mi50_down_cent;

    for (int tp = 0; tp < 3; ++tp) {
        mi50_gate_codes[tp].resize(size_t(mi50_experts) * kMi50CodesWords);
        mi50_up_codes[tp].resize(size_t(mi50_experts) * kMi50CodesWords);
        mi50_down_codes[tp].resize(size_t(mi50_experts) * kMi50CodesWords);
        mi50_gate_scales[tp].resize(size_t(mi50_experts) * kMi50ScalesHalfs);
        mi50_up_scales[tp].resize(size_t(mi50_experts) * kMi50ScalesHalfs);
        mi50_down_scales[tp].resize(size_t(mi50_experts) * kMi50ScalesHalfs);
        mi50_gate_cent[tp].resize(size_t(mi50_experts) * 2);
        mi50_up_cent[tp].resize(size_t(mi50_experts) * 2);
        mi50_down_cent[tp].resize(size_t(mi50_experts) * 2);
    }

    // Preallocate P150A TP2 HLUT2 expert banks (backbone only, 168 experts):
    //   360 superblocks * 4352 B = 1,566,720 B per projection per expert per card!
    constexpr size_t kP150ProjBytes = 360ULL * 4352ULL;
    std::array<std::vector<uint8_t>, 2> p150_gate_pages, p150_up_pages, p150_down_pages;
    std::array<std::vector<float>, 2> p150_gate_cent, p150_up_cent, p150_down_cent;
    if (p150_experts > 0) {
        for (int tp = 0; tp < 2; ++tp) {
            p150_gate_pages[tp].resize(size_t(p150_experts) * kP150ProjBytes);
            p150_up_pages[tp].resize(size_t(p150_experts) * kP150ProjBytes);
            p150_down_pages[tp].resize(size_t(p150_experts) * kP150ProjBytes);
            p150_gate_cent[tp].resize(size_t(p150_experts) * 2);
            p150_up_cent[tp].resize(size_t(p150_experts) * 2);
            p150_down_cent[tp].resize(size_t(p150_experts) * 2);
        }
    }

    Hlut2QuantMatrix exp0_w1, exp0_w3, exp0_w2;

    #pragma omp parallel
    {
        std::vector<float> scratch_rot, scratch_sig;
        #pragma omp for schedule(dynamic, 1)
        for (int e = 0; e < total_experts; ++e) {
            const std::string epfx = pfx + ".ffn.experts." + std::to_string(e);
            Hlut2QuantMatrix qw1 = dequant_mxfp4_and_quantize_hlut2_st(
                st.tensor_bytes(epfx + ".w1.weight"),
                st.tensor_bytes(epfx + ".w1.scale"),
                2304, 5120, signs_h, scratch_rot, scratch_sig);
            Hlut2QuantMatrix qw3 = dequant_mxfp4_and_quantize_hlut2_st(
                st.tensor_bytes(epfx + ".w3.weight"),
                st.tensor_bytes(epfx + ".w3.scale"),
                2304, 5120, signs_h, scratch_rot, scratch_sig);
            Hlut2QuantMatrix qw2 = dequant_mxfp4_and_quantize_hlut2_st(
                st.tensor_bytes(epfx + ".w2.weight"),
                st.tensor_bytes(epfx + ".w2.scale"),
                5120, 2304, signs_i, scratch_rot, scratch_sig);

            if (e == 0) {
                exp0_w1 = qw1;
                exp0_w3 = qw3;
                exp0_w2 = qw2;
            }

            if (e < mi50_experts) {
                for (int tp = 0; tp < 3; ++tp) {
                    const auto g_sub = slice_hlut2_rows(qw1, tp * 768, 768);
                    const auto u_sub = slice_hlut2_rows(qw3, tp * 768, 768);
                    const auto d_sub = slice_hlut2_cols(qw2, tp * 768, 768);
                    pack_hlut2_mi50_slot_st(
                        g_sub,
                        mi50_gate_codes[tp].data() + size_t(e) * kMi50CodesWords,
                        mi50_gate_scales[tp].data() + size_t(e) * kMi50ScalesHalfs,
                        mi50_gate_cent[tp].data() + size_t(e) * 2);
                    pack_hlut2_mi50_slot_st(
                        u_sub,
                        mi50_up_codes[tp].data() + size_t(e) * kMi50CodesWords,
                        mi50_up_scales[tp].data() + size_t(e) * kMi50ScalesHalfs,
                        mi50_up_cent[tp].data() + size_t(e) * 2);
                    pack_hlut2_mi50_slot_st(
                        d_sub,
                        mi50_down_codes[tp].data() + size_t(e) * kMi50CodesWords,
                        mi50_down_scales[tp].data() + size_t(e) * kMi50ScalesHalfs,
                        mi50_down_cent[tp].data() + size_t(e) * 2);
                }
            } else {
                const int slot = e - mi50_experts;
                for (int tp = 0; tp < 2; ++tp) {
                    pack_hlut2_p150_slot_st(
                        qw1, tp * 1152, 1152,
                        p150_gate_pages[tp].data() + size_t(slot) * kP150ProjBytes,
                        p150_gate_cent[tp].data() + size_t(slot) * 2);
                    pack_hlut2_p150_slot_st(
                        qw3, tp * 1152, 1152,
                        p150_up_pages[tp].data() + size_t(slot) * kP150ProjBytes,
                        p150_up_cent[tp].data() + size_t(slot) * 2);
                    const auto d_sub = slice_hlut2_cols(qw2, tp * 1152, 1152);
                    pack_hlut2_p150_slot_st(
                        d_sub, 0, 5120,
                        p150_down_pages[tp].data() + size_t(slot) * kP150ProjBytes,
                        p150_down_cent[tp].data() + size_t(slot) * 2);
                }
            }
        }
    }

    // Inline real-weight accuracy sanity check on Expert 0 (single-matrix W1 >= 0.91 and 3-matrix SwiGLU >= 0.80)
    const auto qm = verify_expert0_swiglu_inline(st, pfx, exp0_w1, exp0_w3, exp0_w2, signs_h, signs_i);
    if (!std::isfinite(qm.cosine) || qm.cosine < 0.80) {
        throw std::runtime_error("HLUT2 Expert 0 verification failed on " + tag + ": cosine=" + std::to_string(qm.cosine));
    }

    // Write MI50 TP3 shards
    for (int tp = 0; tp < 3; ++tp) {
        StreamingSafetensorsWriter w;

        // Replicated norms, mHC, wq_a, wkv, gate
        add_replicated(w, st, pfx + ".attn_norm.weight", "attn_norm.weight");
        add_replicated(w, st, pfx + ".ffn_norm.weight", "ffn_norm.weight");
        add_replicated(w, st, pfx + ".hc_attn_base", "hc_attn_base");
        add_replicated(w, st, pfx + ".hc_attn_fn", "hc_attn_fn");
        add_replicated(w, st, pfx + ".hc_attn_scale", "hc_attn_scale");
        add_replicated(w, st, pfx + ".hc_ffn_base", "hc_ffn_base");
        add_replicated(w, st, pfx + ".hc_ffn_fn", "hc_ffn_fn");
        add_replicated(w, st, pfx + ".hc_ffn_scale", "hc_ffn_scale");

        add_replicated(w, st, pfx + ".attn.q_norm.weight", "attn.q_norm.weight");
        add_replicated(w, st, pfx + ".attn.kv_norm.weight", "attn.kv_norm.weight");
        add_replicated(w, st, pfx + ".attn.wq_a.weight", "attn.wq_a.weight");
        add_replicated(w, st, pfx + ".attn.wq_a.scale", "attn.wq_a.scale");
        add_replicated(w, st, pfx + ".attn.wkv.weight", "attn.wkv.weight");
        add_replicated(w, st, pfx + ".attn.wkv.scale", "attn.wkv.scale");

        // TP3 attention heads & o_groups: (24, 24, 16) heads = (3, 3, 2) o_groups
        const int h_off = Dsv41Tp3Split::attn_head_offset(tp);
        const int h_cnt = Dsv41Tp3Split::attn_heads(tp);
        const int g_off = Dsv41Tp3Split::o_group_offset(tp);
        const int g_cnt = Dsv41Tp3Split::o_groups(tp);

        add_row_shard(w, st, pfx + ".attn.attn_sink", "attn.attn_sink", h_off, h_cnt);
        add_row_shard(w, st, pfx + ".attn.wq_b.weight", "attn.wq_b.weight", h_off * 512, h_cnt * 512);
        add_row_shard(w, st, pfx + ".attn.wq_b.scale", "attn.wq_b.scale", (h_off * 512) / 32, (h_cnt * 512) / 32);
        add_row_shard(w, st, pfx + ".attn.wo_a.weight", "attn.wo_a.weight", g_off * 1024, g_cnt * 1024);
        add_row_shard(w, st, pfx + ".attn.wo_a.scale", "attn.wo_a.scale", (g_off * 1024) / 32, (g_cnt * 1024) / 32);
        add_col_shard_2d(w, st, pfx + ".attn.wo_b.weight", "attn.wo_b.weight", g_off * 1024, g_cnt * 1024);
        add_col_shard_2d(w, st, pfx + ".attn.wo_b.scale", "attn.wo_b.scale", (g_off * 1024) / 32, (g_cnt * 1024) / 32);

        // Optional Compressor (layers 2, 8, 14, 20)
        if (st.contains(pfx + ".attn.compressor.wkv.weight")) {
            add_replicated(w, st, pfx + ".attn.compressor.norm.weight", "attn.compressor.norm.weight");
            add_replicated(w, st, pfx + ".attn.compressor.wkv.weight", "attn.compressor.wkv.weight");
            if (st.contains(pfx + ".attn.compressor.wgate.weight")) {
                add_replicated(w, st, pfx + ".attn.compressor.wgate.weight", "attn.compressor.wgate.weight");
            }
        }

        // Optional Indexer (layers 2, 8, 14, 20, 24, 28, 32, 36): (11, 11, 10) heads
        if (st.contains(pfx + ".attn.indexer.wq_b.weight")) {
            const int ih_off = Dsv41Tp3Split::idx_head_offset(tp);
            const int ih_cnt = Dsv41Tp3Split::idx_heads(tp);
            add_row_shard(w, st, pfx + ".attn.indexer.wq_b.weight", "attn.indexer.wq_b.weight", ih_off * 128, ih_cnt * 128);
            add_row_shard(w, st, pfx + ".attn.indexer.wq_b.scale", "attn.indexer.wq_b.scale", (ih_off * 128) / 32, (ih_cnt * 128) / 32);
            add_row_shard(w, st, pfx + ".attn.indexer.weights_proj.weight", "attn.indexer.weights_proj.weight", ih_off, ih_cnt);
            if (st.contains(pfx + ".attn.indexer.wk.weight")) {
                add_replicated(w, st, pfx + ".attn.indexer.wk.weight", "attn.indexer.wk.weight");
                add_replicated(w, st, pfx + ".attn.indexer.k_norm.weight", "attn.indexer.k_norm.weight");
            }
        }

        // Gate & TP3 Shared Expert (I = 2304 -> 768/card)
        add_replicated(w, st, pfx + ".ffn.gate.weight", "ffn.gate.weight");
        add_replicated(w, st, pfx + ".ffn.gate.bias", "ffn.gate.bias");
        if (st.contains(pfx + ".ffn.gate.bias_vl")) {
            add_replicated(w, st, pfx + ".ffn.gate.bias_vl", "ffn.gate.bias_vl");
        }
        add_row_shard(w, st, pfx + ".ffn.shared_experts.w1.weight", "ffn.shared_experts.w1.weight", tp * 768, 768);
        add_row_shard(w, st, pfx + ".ffn.shared_experts.w1.scale", "ffn.shared_experts.w1.scale", tp * 24, 24);
        add_row_shard(w, st, pfx + ".ffn.shared_experts.w3.weight", "ffn.shared_experts.w3.weight", tp * 768, 768);
        add_row_shard(w, st, pfx + ".ffn.shared_experts.w3.scale", "ffn.shared_experts.w3.scale", tp * 24, 24);
        add_col_shard_2d(w, st, pfx + ".ffn.shared_experts.w2.weight", "ffn.shared_experts.w2.weight", tp * 768, 768);
        add_col_shard_2d(w, st, pfx + ".ffn.shared_experts.w2.scale", "ffn.shared_experts.w2.scale", tp * 24, 24);

        // MTP extra heads (mtp.0 main_proj/main_norm; mtp.2 norm/confidence_head/markov_head)
        if (is_mtp && local_id == 0) {
            add_replicated(w, st, "mtp.0.main_norm.weight", "main_norm.weight");
            add_col_shard_2d(w, st, "mtp.0.main_proj.weight", "main_proj.weight", tp * 5120, 5120);
            add_col_shard_2d(w, st, "mtp.0.main_proj.scale", "main_proj.scale", tp * 160, 160);
        }
        if (is_mtp && local_id == 2) {
            add_replicated(w, st, "mtp.2.norm.weight", "norm.weight");
            add_replicated(w, st, "mtp.2.confidence_head.proj.weight", "confidence_head.proj.weight");
            const int v_off = Dsv41Tp3Split::vocab_offset(tp);
            const int v_cnt = Dsv41Tp3Split::vocab_rows(tp);
            add_row_shard(w, st, "mtp.2.markov_head.embed.weight", "markov_head.embed.weight", v_off, v_cnt);
            add_row_shard(w, st, "mtp.2.markov_head.head.weight", "markov_head.head.weight", v_off, v_cnt);
        }

        // WHT128 sign masks
        const std::vector<uint64_t> signs_i_sub(signs_i.begin() + tp * 12, signs_i.begin() + (tp + 1) * 12);
        w.add_borrowed("wht.signs_h", "U64", {int64_t(signs_h.size())}, signs_h.data(), signs_h.size() * 8ULL);
        std::vector<uint8_t> si_bytes(signs_i_sub.size() * 8ULL);
        std::memcpy(si_bytes.data(), signs_i_sub.data(), si_bytes.size());
        w.add_owned("wht.signs_i", "U64", {int64_t(signs_i_sub.size())}, std::move(si_bytes));

        // Contiguous HLUT2 expert banks
        w.add_borrowed("hlut2.gate_codes", "U32", {mi50_experts, 24, 320, 32}, mi50_gate_codes[tp].data(), mi50_gate_codes[tp].size() * 4ULL);
        w.add_borrowed("hlut2.gate_scales", "F16", {mi50_experts, 24, 40, 32}, mi50_gate_scales[tp].data(), mi50_gate_scales[tp].size() * 2ULL);
        w.add_borrowed("hlut2.gate_centroids", "F32", {mi50_experts, 2}, mi50_gate_cent[tp].data(), mi50_gate_cent[tp].size() * 4ULL);

        w.add_borrowed("hlut2.up_codes", "U32", {mi50_experts, 24, 320, 32}, mi50_up_codes[tp].data(), mi50_up_codes[tp].size() * 4ULL);
        w.add_borrowed("hlut2.up_scales", "F16", {mi50_experts, 24, 40, 32}, mi50_up_scales[tp].data(), mi50_up_scales[tp].size() * 2ULL);
        w.add_borrowed("hlut2.up_centroids", "F32", {mi50_experts, 2}, mi50_up_cent[tp].data(), mi50_up_cent[tp].size() * 4ULL);

        w.add_borrowed("hlut2.down_codes", "U32", {mi50_experts, 160, 48, 32}, mi50_down_codes[tp].data(), mi50_down_codes[tp].size() * 4ULL);
        w.add_borrowed("hlut2.down_scales", "F16", {mi50_experts, 160, 6, 32}, mi50_down_scales[tp].data(), mi50_down_scales[tp].size() * 2ULL);
        w.add_borrowed("hlut2.down_centroids", "F32", {mi50_experts, 2}, mi50_down_cent[tp].data(), mi50_down_cent[tp].size() * 4ULL);

        const std::string out_dir = mi50_root + "/tp" + std::to_string(tp);
        mkdir_p(out_dir);
        w.write_atomic(out_dir + "/" + tag + ".safetensors");
    }

    // Write P150A TP2 shards (backbone layers 0..39)
    if (p150_experts > 0) {
        for (int tp = 0; tp < 2; ++tp) {
            StreamingSafetensorsWriter w;
            const std::vector<uint64_t> signs_i_sub(signs_i.begin() + tp * 18, signs_i.begin() + (tp + 1) * 18);
            w.add_borrowed("wht.signs_h", "U64", {int64_t(signs_h.size())}, signs_h.data(), signs_h.size() * 8ULL);
            std::vector<uint8_t> si_bytes(signs_i_sub.size() * 8ULL);
            std::memcpy(si_bytes.data(), signs_i_sub.data(), si_bytes.size());
            w.add_owned("wht.signs_i", "U64", {int64_t(signs_i_sub.size())}, std::move(si_bytes));

            w.add_borrowed("hlut2.gate_pages", "U8", {p150_experts, 360, 4352}, p150_gate_pages[tp].data(), p150_gate_pages[tp].size());
            w.add_borrowed("hlut2.gate_centroids", "F32", {p150_experts, 2}, p150_gate_cent[tp].data(), p150_gate_cent[tp].size() * 4ULL);

            w.add_borrowed("hlut2.up_pages", "U8", {p150_experts, 360, 4352}, p150_up_pages[tp].data(), p150_up_pages[tp].size());
            w.add_borrowed("hlut2.up_centroids", "F32", {p150_experts, 2}, p150_up_cent[tp].data(), p150_up_cent[tp].size() * 4ULL);

            w.add_borrowed("hlut2.down_pages", "U8", {p150_experts, 360, 4352}, p150_down_pages[tp].data(), p150_down_pages[tp].size());
            w.add_borrowed("hlut2.down_centroids", "F32", {p150_experts, 2}, p150_down_cent[tp].data(), p150_down_cent[tp].size() * 4ULL);

            const std::string out_dir = p150_root + "/tp" + std::to_string(tp);
            mkdir_p(out_dir);
            w.write_atomic(out_dir + "/" + tag + ".safetensors");
        }
    }

    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[DSV41_PACK] Packed %s (shard %02d/48) in %.2fs | Expert0 SwiGLU cos=%.5f rel_l2=%.4f (c0=%.4f c1=%.4f)\n",
                tag.c_str(), shard_idx, dt, qm.cosine, qm.rel_l2, exp0_w1.c0, exp0_w1.c1);
    std::fflush(stdout);
}

void pack_single_shard(
    int shard_idx,
    const std::string& raw_path,
    const std::string& mi50_root,
    const std::string& p150_root)
{
    if (shard_idx == 1) {
        // Vision + Aligner (0.90 GiB): verify via SafetensorsMmap and move/copy to mi50_root/vision.safetensors
        {
            SafetensorsMmap st(raw_path);
            if (st.entries().empty()) throw std::runtime_error("Empty vision shard");
        }
        mkdir_p(mi50_root);
        const std::string dst = mi50_root + "/vision.safetensors";
        if (raw_path != dst) {
            std::filesystem::copy_file(raw_path, dst + ".tmp", std::filesystem::copy_options::overwrite_existing);
            std::filesystem::rename(dst + ".tmp", dst);
        }
        std::printf("[DSV41_PACK] Packed vision.safetensors (shard 01/48)\n");
        std::fflush(stdout);
    } else if (shard_idx == 2) {
        SafetensorsMmap st(raw_path);
        for (int tp = 0; tp < 3; ++tp) {
            StreamingSafetensorsWriter w;
            add_row_shard(w, st, "embed.weight", "embed.weight",
                          Dsv41Tp3Split::vocab_offset(tp), Dsv41Tp3Split::vocab_rows(tp));
            if (st.contains("image_start")) {
                add_replicated(w, st, "image_start", "image_start");
                add_replicated(w, st, "image_end", "image_end");
                add_replicated(w, st, "image_newline", "image_newline");
            }
            const std::string out_dir = mi50_root + "/tp" + std::to_string(tp);
            mkdir_p(out_dir);
            w.write_atomic(out_dir + "/embed.safetensors");
        }
        std::printf("[DSV41_PACK] Packed embed.safetensors (shard 02/48)\n");
        std::fflush(stdout);
    } else if (shard_idx >= 3 && shard_idx <= 42) {
        pack_moe_block_shard(shard_idx, raw_path, mi50_root, p150_root, false);
    } else if (shard_idx == 43) {
        SafetensorsMmap st(raw_path);
        for (int tp = 0; tp < 3; ++tp) {
            StreamingSafetensorsWriter w;
            add_replicated(w, st, "norm.weight", "norm.weight");
            add_row_shard(w, st, "head.weight", "head.weight",
                          Dsv41Tp3Split::vocab_offset(tp), Dsv41Tp3Split::vocab_rows(tp));
            const std::string out_dir = mi50_root + "/tp" + std::to_string(tp);
            mkdir_p(out_dir);
            w.write_atomic(out_dir + "/head.safetensors");
        }
        std::printf("[DSV41_PACK] Packed head.safetensors (shard 43/48)\n");
        std::fflush(stdout);
    } else if (shard_idx >= 44 && shard_idx <= 46) {
        pack_moe_block_shard(shard_idx, raw_path, mi50_root, p150_root, true);
    } else if (shard_idx == 47 || shard_idx == 48) {
        // Engram shards (94.56 GiB each): keep raw .safetensors intact on NVMe for DDR4 mmap,
        // and extract GPU-resident q_weight, k_weight, and TP3 row-parallel wkv.{weight,scale}!
        const int layer_id = (shard_idx == 47) ? 1 : 14;
        const std::string pfx = "layers." + std::to_string(layer_id) + ".engram";
        SafetensorsMmap st(raw_path);
        for (int tp = 0; tp < 3; ++tp) {
            StreamingSafetensorsWriter w;
            add_replicated(w, st, pfx + ".q_weight", "q_weight");
            add_replicated(w, st, pfx + ".k_weight", "k_weight");
            add_col_shard_2d(w, st, pfx + ".wkv.weight", "wkv.weight", tp * 2048, 2048);
            add_col_shard_2d(w, st, pfx + ".wkv.scale", "wkv.scale", tp * 64, 64);
            const std::string out_dir = mi50_root + "/tp" + std::to_string(tp);
            mkdir_p(out_dir);
            w.write_atomic(out_dir + "/engram_layer_" + std::to_string(layer_id) + ".safetensors");
        }
        std::ofstream tp_file(mi50_root + "/engram_layer_" + std::to_string(layer_id) + ".table_path", std::ios::trunc);
        tp_file << raw_path << "\n";
        tp_file.close();
        std::printf("[DSV41_PACK] Indexed Engram layer %d (%s kept intact for DDR4 mmap!)\n", layer_id, raw_path.c_str());
        std::fflush(stdout);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string root = (std::getenv("DSV41_ROOT") != nullptr)
                           ? std::string(std::getenv("DSV41_ROOT"))
                           : std::string(".");
    std::string mi50_root;
    std::string p150_root;
    bool watch_mode = false;
    int max_shard = 48;
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--watch") watch_mode = true;
        else if (a == "--once") watch_mode = false;
        else if (a == "--model-dir" && i + 1 < argc) root = argv[++i];
        else if (a == "--mi50-out" && i + 1 < argc) mi50_root = argv[++i];
        else if (a == "--p150-out" && i + 1 < argc) p150_root = argv[++i];
        else max_shard = std::atoi(a.c_str());
    }

    const std::string status_dir = root + "/shard_status";
    if (mi50_root.empty()) mi50_root = root + "/packed_mi50";
    if (p150_root.empty()) {
        p150_root = (std::getenv("DSV41_P150_ROOT") != nullptr)
                        ? std::string(std::getenv("DSV41_P150_ROOT"))
                        : (root + "/packed_p150");
    }
    mkdir_p(status_dir);
    mkdir_p(mi50_root);
    mkdir_p(p150_root);

    std::printf("[DSV41_PACK] Streaming Pure-C++17 Packer started (watch=%d, max_shard=%d)\n",
                int(watch_mode), max_shard);
    std::fflush(stdout);

    while (true) {
        int packed_count = 0;
        bool did_work = false;
        for (int idx = 1; idx <= max_shard; ++idx) {
            char sname[64];
            std::snprintf(sname, sizeof(sname), "model-%05d-of-00048.safetensors", idx);
            const std::string packed_marker = status_dir + "/" + sname + ".packed";
            if (file_exists(packed_marker)) {
                ++packed_count;
                continue;
            }
            const std::string ready_marker = status_dir + "/" + sname + ".ready";
            if (!file_exists(ready_marker)) continue;

            const std::string raw_path = read_first_line(ready_marker);
            if (raw_path.empty() || !file_exists(raw_path)) continue;

            pack_single_shard(idx, raw_path, mi50_root, p150_root);

            // Reclaim raw backbone/MTP shards 00001..00046 immediately after verified packing;
            // NEVER unlink Engram shards 00047 & 00048!
            if (idx <= 46) {
                ::unlink(raw_path.c_str());
                ::unlink(ready_marker.c_str());
            }

            std::ofstream pm(packed_marker, std::ios::trunc);
            pm << "ok\n";
            pm.close();
            ++packed_count;
            did_work = true;
        }

        if (!watch_mode || packed_count >= max_shard) {
            std::printf("[DSV41_PACK] Completed %d/%d shards.\n", packed_count, max_shard);
            std::fflush(stdout);
            break;
        }
        if (!did_work) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    }
    return 0;
}
