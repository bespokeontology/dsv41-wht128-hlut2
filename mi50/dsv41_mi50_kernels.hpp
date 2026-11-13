#pragma once
// Pure-C++17 / AMD HIP (gfx906) Full Kernel Suite for DeepSeek-V4.1-Flash-1017 on 3x MI50 (96 GiB HBM2)
// Zero Python anywhere. Every kernel engineered for ScratchSize == 0 (zero VGPR spills to scratch).

#include <hip/hip_runtime.h>
#include <cstdint>

namespace dsv41 {

#define DSV41_HIP_CHECK(call)                                                                     \
    do {                                                                                          \
        hipError_t err__ = (call);                                                                \
        if (err__ != hipSuccess) {                                                                \
            throw std::runtime_error(std::string("HIP error at ") + __FILE__ + ":" +              \
                                     std::to_string(__LINE__) + ": " + hipGetErrorString(err__)); \
        }                                                                                         \
    } while (0)

// Host-side routed pair descriptor uploaded per MoE call
struct Mi50RoutedPair {
    int32_t row;       // 0 .. M-1
    int32_t expert;    // 0 .. mi50_experts-1
    float   weight;    // scaled router weight
    int32_t pad;
};

// 1. Embedding & RMSNorm & Conversions
void launch_tp_embed_bf16(
    const int32_t* d_token_ids,   // [M]
    const uint16_t* d_embed_w,    // [vocab_rows, D] BF16
    uint16_t* d_out_bf16,         // [M, D] BF16
    int M, int D, int vocab_offset, int vocab_rows,
    hipStream_t stream);

void launch_rmsnorm_bf16(
    const uint16_t* d_in_bf16,    // [M, D]
    const uint16_t* d_weight_bf16,// [D]
    uint16_t* d_out_bf16,         // [M, D]
    int M, int D, float eps,
    hipStream_t stream);

void launch_f32_to_bf16(
    const float* d_in_f32,
    uint16_t* d_out_bf16,
    size_t n_elems,
    hipStream_t stream);

void launch_add_f32_2way_inplace(
    float* d_dst,
    const float* d_src,
    size_t n_elems,
    hipStream_t stream);

void launch_add_f32_inplace(
    float* d_dst,
    const float* d_src1,
    const float* d_src2,
    size_t n_elems,
    hipStream_t stream);

void launch_add_f32_3way(
    float* d_out_f32,
    const float* d_a,
    const float* d_b,
    const float* d_c,
    size_t n_elems,
    hipStream_t stream);

void launch_add_bf16_3way(
    uint16_t* d_out_bf16,
    const uint16_t* d_a,
    const uint16_t* d_b,
    const uint16_t* d_c,
    size_t n_elems,
    hipStream_t stream);

void launch_add_bf16_2way_inplace(
    uint16_t* d_dst_bf16,
    const uint16_t* d_src_bf16,
    size_t n_elems,
    hipStream_t stream);

// Posted-Write PCIe P2P Push: reads local HBM2 `d_src` once and writes posted 128-bit stores to both peers' HBM2 recv buffers!
void launch_p2p_push_bf16_2peers(
    const uint16_t* d_src,
    uint16_t* d_peer1_recv,
    uint16_t* d_peer2_recv,
    size_t n_elems,
    hipStream_t stream);

void launch_p2p_push_f32_2peers(
    const float* d_src,
    float* d_peer1_recv,
    float* d_peer2_recv,
    size_t n_elems,
    hipStream_t stream);

// Zero-CPU-sync Pinned Coherent System-Memory TP3 Ring AllReduce (0 bytes HBM2, 0 PCIe P2P BAR reads, bit-identical (r0+r1)+r2 sum!)
static constexpr size_t kHostArRingSlots = 8u;
static constexpr size_t kHostArMaxWords  = 2621440u;  // 10.0 MiB/rank = up to 5,242,880 BF16 (M=1024) or 2,621,440 FP32 elems!

struct HostRingAllReduceSlot {
    alignas(64) uint32_t write_ctas[3][16];
    alignas(64) uint32_t read_ctas[3][16];
    alignas(64) uint64_t ready_seq[3][8];
    alignas(64) uint64_t done_seq[3][8];
    alignas(64) uint32_t payload[3][kHostArMaxWords];
};

void launch_host_ring_allreduce_bf16(
    const uint16_t* d_src,
    uint16_t* d_dst,
    HostRingAllReduceSlot* d_mapped_slot,
    int rank,
    uint64_t seq,
    size_t n_elems,
    hipStream_t stream);

void launch_host_ring_allreduce_f32(
    const float* d_src,
    float* d_dst,
    HostRingAllReduceSlot* d_mapped_slot,
    int rank,
    uint64_t seq,
    size_t n_elems,
    hipStream_t stream);

// Dumitrescu mixed-precision FP32 TP3 HostRingAllReduce + optional P150A BF16 add + FP32 mHC post -> single final RNE BF16!
void launch_host_ring_allreduce_f32_hc_post(
    const float* d_src_f32,         // [M, 5120] FP32 local shard
    const uint16_t* d_p150_bf16,    // [M, 5120] BF16 optional P150A output (nullptr for Attn / MTP / non-P150)
    const uint16_t* d_res_bf16,     // [M, 4, 5120] BF16 residual hyper-stream
    const float* d_post_f32,        // [M, 4] FP32 post mix
    const float* d_comb_f32,        // [M, 16] FP32 comb mix
    uint16_t* d_out_h_bf16,         // [M, 4, 5120] BF16 updated hyper-stream
    HostRingAllReduceSlot* d_mapped_slot,
    int rank,
    uint64_t seq,
    int M,
    hipStream_t stream);

// Split Phase-1 upload + Phase-2 finish for Zero-SDMA P150A + TP3 MoE overlap:
// Phase 1 streams local [M, 5120] FP32 partials into HostRingAllReduceSlot while P150A executes;
// Phase 2 reads mapped /dsv41_p150_ipc y_out_bf16 directly via system-scope atomic loads (0 hipMemcpyAsync!).
void launch_host_ring_upload_f32(
    const float* d_src_f32,         // [M, 5120] FP32 local shard
    HostRingAllReduceSlot* d_mapped_slot,
    int rank,
    uint64_t seq,
    int M,
    hipStream_t stream);

void launch_host_ring_finish_f32_hc_post(
    const float* d_src_f32,         // [M, 5120] FP32 local shard
    const uint16_t* d_p150_bf16,    // [M, 5120] BF16 mapped /dsv41_p150_ipc y_out_bf16
    const uint16_t* d_res_bf16,     // [M, 4, 5120] BF16 residual hyper-stream
    const float* d_post_f32,        // [M, 4] FP32 post mix
    const float* d_comb_f32,        // [M, 16] FP32 comb mix
    uint16_t* d_out_h_bf16,         // [M, 4, 5120] BF16 updated hyper-stream
    HostRingAllReduceSlot* d_mapped_slot,
    int rank,
    uint64_t seq,
    int M,
    hipStream_t stream);

void launch_add_f32_and_bf16_to_bf16(
    const float* d_routed_tp3_sum,  // [M, 5120] FP32 (sum of 3 MI50 routed + shared partials)
    const uint16_t* d_p150_bf16,    // [M, 5120] BF16 (optional P150A expert output, may be nullptr)
    uint16_t* d_out_bf16,           // [M, 5120] BF16
    size_t n_elems,
    hipStream_t stream);

// 2. Hyper-Connections (mHC: hc_mult=4, dim=5120, mix_hc=24)
void launch_hc_init(
    const uint16_t* d_embed_bf16, // [M, 5120]
    uint16_t* d_h_bf16,           // [M, 4, 5120]
    float* d_pre_mix_f32,         // [M, 4] -> [1, 0, 0, 0]
    int M,
    hipStream_t stream);

void launch_hc_mean_bf16(
    const uint16_t* d_h_bf16,     // [M, 4, 5120]
    uint16_t* d_mean_bf16,        // [M, 5120]
    int M,
    hipStream_t stream);

void launch_hc_mixes(
    const uint16_t* d_h_bf16,     // [M, 4, 5120]
    const float* d_hc_fn_f32,     // [24, 20480]
    const float* d_hc_scale_f32,  // [3]
    const float* d_hc_base_f32,   // [24]
    float* d_scratch_rsqrt_f32,   // [M]
    float* d_scratch_mixes_f32,   // [M, 24]
    float* d_out_pre_f32,         // [M, 4]
    float* d_out_post_f32,        // [M, 4]
    float* d_out_comb_f32,        // [M, 16] ([M, 4, 4])
    int M, float norm_eps, int sinkhorn_iters, float hc_eps,
    hipStream_t stream);

void launch_hc_pre(
    const uint16_t* d_h_bf16,     // [M, 4, 5120]
    const float* d_pre_f32,       // [M, 4]
    uint16_t* d_out_bf16,         // [M, 5120]
    int M,
    hipStream_t stream);

// Fused hc_pre + RMSNorm (5120) + optional register-butterfly WHT128 (d_signs_u64 -> d_out_rot_f16 / d_out_rot_bf16)
void launch_hc_pre_rmsnorm_bf16(
    const uint16_t* d_h_bf16,       // [M, 4, 5120]
    const float* d_pre_f32,         // [M, 4]
    const uint16_t* d_norm_w_bf16,  // [5120]
    uint16_t* d_out_norm_bf16,      // [M, 5120]
    uint16_t* d_out_raw_bf16,       // [M, 5120] optional (may be nullptr)
    int M, float eps,
    hipStream_t stream,
    const uint64_t* d_signs_u64 = nullptr,  // [80] optional WHT128 signs for ffn_norm
    uint16_t* d_out_rot_f16 = nullptr,      // [M, 5120] optional FP16 rotated output
    uint16_t* d_out_rot_bf16 = nullptr);    // [M, 5120] optional BF16 rotated output (tp=0 -> P150A)

void launch_hc_post(
    const uint16_t* d_sub_bf16,   // [M, 5120]
    const uint16_t* d_res_bf16,   // [M, 4, 5120]
    const float* d_post_f32,      // [M, 4]
    const float* d_comb_f32,      // [M, 16]
    uint16_t* d_out_h_bf16,       // [M, 4, 5120]
    int M,
    hipStream_t stream);

// 3. Engram Gate & Residual Add (layers 1 & 14)
void launch_engram_gate_add(
    uint16_t* d_h_bf16,           // [M, 4, 5120] in/out
    const float* d_kv_f32,        // [M, 25600] (TP3-summed wkv output)
    const uint16_t* d_q_weight,   // [4, 5120] BF16
    const uint16_t* d_k_weight,   // [4, 5120] BF16
    const uint8_t* d_image_mask,  // [M] optional (nullptr = all text)
    int M, float norm_eps,
    hipStream_t stream);

// 4. Quantization + FP8xFP8 (32x32 UE8M0) & BF16 GEMV/GEMM
void launch_act_quant_ue8m0_32(
    const uint16_t* d_in_bf16,    // [M, K]
    uint8_t* d_out_fp8,           // [M, K] (when !inplace_bf16)
    uint8_t* d_out_ue8m0,         // [M, K/32] (when !inplace_bf16)
    uint16_t* d_inplace_bf16,     // [M, K] (when inplace_bf16)
    int M, int K, bool inplace_bf16,
    hipStream_t stream);

void launch_fp8_gemm(
    const uint8_t* d_x_fp8,       // [M, K]
    const uint8_t* d_x_scale,     // [M, K/32] UE8M0
    const uint8_t* d_w_fp8,       // [N, K]
    const uint8_t* d_w_scale,     // [N/32, K/32] UE8M0
    uint16_t* d_out_bf16,         // [M, N] (if !write_f32)
    float* d_out_f32,             // [M, N] (if write_f32)
    int M, int N, int K, bool write_f32, bool accumulate_f32,
    hipStream_t stream);

void launch_fp8_grouped_wo_a(
    const uint8_t* d_o_fp8,       // [M, g_cnt * 4096]
    const uint8_t* d_o_scale,     // [M, g_cnt * 128] UE8M0
    const uint8_t* d_wo_a_fp8,    // [g_cnt * 1024, 4096]
    const uint8_t* d_wo_a_scale,  // [g_cnt * 32, 128] UE8M0
    uint16_t* d_out_bf16,         // [M, g_cnt * 1024]
    int M, int g_cnt,
    hipStream_t stream);

void launch_bf16_gemm(
    const uint16_t* d_x_bf16,     // [M, K]
    const uint16_t* d_w_bf16,     // [N, K]
    uint16_t* d_out_bf16,         // [M, N] (if !write_f32)
    float* d_out_f32,             // [M, N] (if write_f32)
    int M, int N, int K, bool write_f32,
    hipStream_t stream);

void launch_swiglu10_bf16(
    const uint16_t* d_gate_bf16,  // [M, I]
    const uint16_t* d_up_bf16,    // [M, I]
    uint16_t* d_out_bf16,         // [M, I]
    int n_elems,
    hipStream_t stream);

// Fused SwiGLU10 on concatenated [gate (I), up (I)] + UE8M0_32 activation quantization for sh_w2
void launch_swiglu10_split_quant_ue8m0_32(
    const uint16_t* d_gateup_bf16,// [M, 2 * I] (I multiple of 32)
    uint8_t* d_out_fp8,           // [M, I]
    uint8_t* d_out_ue8m0,         // [M, I / 32]
    int M, int I,
    hipStream_t stream);

// Fused split + q_norm (1280) + kv_norm (512) + exact FP64-angle tail-64 RoPE + inplace UE8M0_32 quant on concatenated [qr; kv] [M, 1792]
void launch_split_qkv_norm_rope_quant(
    const uint16_t* d_qkv_bf16,     // [M, 1792] ([0..1279]=qr, [1280..1791]=kv)
    const uint16_t* d_q_norm_bf16,  // [1280]
    const uint16_t* d_kv_norm_bf16, // [512]
    bool use_comp_yarn,             // false=window (theta=10000), true=compressed YaRN (theta=160000, factor=16)
    uint16_t* d_out_qr_bf16,        // [M, 1280]
    uint16_t* d_out_kv_bf16,        // [M, 512]
    int M, int pos_base, float eps,
    hipStream_t stream,
    uint64_t tree_anc_mask64 = 0ULL,
    uint32_t tree_depth_nibbles = 0u);

// 5. Exact FP64-Angle Register RoPE / YaRN (< 1 KiB __constant__), Compressor, Indexer, and Sparse Attention (CSA2 + DSpark)
void init_rope_inv_freq_gpu();

void launch_rope_tail64_bf16(
    uint16_t* d_x_bf16,           // [M, H, D] (D >= 64; rotates [..., D-64 : D])
    bool use_comp_yarn,           // false=window (theta=10000), true=compressed YaRN (theta=160000, factor=16)
    int M, int H, int D, int pos_base, int pos_stride, bool inverse,
    hipStream_t stream,
    uint64_t tree_anc_mask64 = 0ULL,
    uint32_t tree_depth_nibbles = 0u);

int launch_compressor_r2_pool(
    const float* d_kv_f32,        // [M, 512]
    const float* d_score_f32,     // [M, 512]
    float* d_kv_state_f32,        // [2, 512] persistent per-layer state
    float* d_score_state_f32,     // [2, 512] persistent per-layer state
    uint16_t* d_out_pool_bf16,    // [max(1, M), 512] pre-norm pooled BF16
    int start_pos, int M,
    hipStream_t stream);

// Speculative M <= 8 causal-tree r=2 pooling (pools parent->child odd pairs without mutating persistent kv_state/score_state!)
void launch_compressor_r2_tree_pool(
    const float* d_kv_f32,        // [M, 512]
    const float* d_score_f32,     // [M, 512]
    const float* d_kv_state_f32,  // [2, 512] persistent per-layer state (read-only)
    const float* d_score_state_f32,
    uint16_t* d_out_pool_bf16,    // [M, 512]
    int start_pos, int M,
    uint32_t tree_depth_nibbles,
    uint32_t tree_parent_nibbles,
    hipStream_t stream);

void launch_pack_compress_kv_fp4_e4m3_16(
    const uint16_t* d_latent_bf16,// [N_lat, 512] (post-RoPE!)
    uint8_t* d_cache_fp4,         // [max_comp, 256] (writes starting at slot_offset)
    uint8_t* d_cache_e4m3,        // [max_comp, 32]  (writes starting at slot_offset)
    int slot_offset, int N_lat,
    hipStream_t stream);

// Packed 68 B/slot Indexer K cache: 64 B FP4_x2 + 4 B UE8M0 (exact 890 B/tok CSA2 Global KV!)
void launch_pack_index_k_fp4_ue8m0_32(
    const uint16_t* d_k_bf16,     // [N_lat, 128] (post-RoPE!)
    uint8_t* d_cache_fp4,         // [max_comp, 64]
    uint8_t* d_cache_ue8m0,       // [max_comp, 4]
    int slot_offset, int N_lat,
    hipStream_t stream);

void launch_fp4_quant_dequant_ue8m0_32_inplace(
    uint16_t* d_x_bf16,           // [rows, D] BF16 (D multiple of 32)
    int rows, int D,
    hipStream_t stream);

void launch_indexer_score(
    const uint16_t* d_q_bf16,       // [M, ih_cnt, 128] (post-RoPE, post-FP4 quant/dequant)
    const uint8_t*  d_k_fp4,        // [max_comp, 64]
    const uint8_t*  d_k_ue8m0,      // [max_comp, 4]
    const uint16_t* d_weights_bf16, // [M, ih_cnt]
    float* d_out_score_f32,         // [M, T_comp]
    int M, int ih_cnt, int T_comp,
    hipStream_t stream);

void launch_indexer_candidates_and_topk(
    float* d_score_f32,             // [M, T_comp] (TP3 all-reduced; upper tail holds M * ceil(T_comp/8) scratch!)
    uint8_t* d_cand_block_mask,     // [M, cand_block_stride] (1 byte per 8 compressed keys!)
    int cand_block_stride,
    int32_t* d_out_comp_idxs,       // [M, K_out]  (0..T_comp-1 or -1, sorted ascending!)
    int M, int T_comp, int start_pos, int ratio,
    int mode,                       // 0 = plain topk, 1 = layer 20 candidate source + topk, 2 = candidate consumer + topk
    int K_out,                      // min(512, T_comp)
    hipStream_t stream,
    uint64_t tree_anc_mask64 = 0ULL,
    uint32_t tree_depth_nibbles = 0u,
    int q_row_offset = 0);

// Zero-staging in-place ring SWA + Top-512 CSA2 + automatic Split-K for M <= 8 (+ zero-alloc SGPR causal-tree support!)
void launch_sparse_attn(
    const uint16_t* d_q_bf16,       // [M, h_cnt, 512]
    const float* d_attn_sink_f32,   // [h_cnt]
    const uint16_t* d_win_kv_ring,  // [128, 512] persistent ring buffer (BEFORE committing d_cur_kv_bf16!)
    const uint16_t* d_cur_kv_bf16,  // [M, 512] current step/chunk KV
    int start_pos,
    int swa_floor_pos,              // 0 normally; replay_start_pos during SWA Bounded Replay
    bool dspark_bidir,              // false = causal CSA2, true = DSpark bidirectional draft
    const uint8_t* d_comp_fp4,      // [max_comp, 256] (optional, nullptr if n_comp==0)
    const uint8_t* d_comp_e4m3,     // [max_comp, 32]  (optional, nullptr if n_comp==0)
    const int32_t* d_comp_idxs,     // [M, n_comp] (0..T_comp-1 or -1)
    int n_comp,
    float* d_splitk_scratch,        // optional scratch (>= M * 8 * h_cnt * 514 floats) for M<=8 Split-K
    uint16_t* d_out_o_bf16,         // [M, h_cnt, 512]
    int M, int h_cnt,
    hipStream_t stream,
    uint64_t tree_anc_mask64 = 0ULL,
    uint32_t tree_depth_nibbles = 0u);

// Accepted-branch gather & Global KV / r=2 state compaction for M <= 8 causal-tree verification (0 extra HBM2!)
void launch_commit_tree_win_kv_and_hidden(
    uint16_t* d_win_kv_slab,            // [43, 128, 512]
    const uint16_t* d_last_win_kv_slab, // [40, 8, 512]
    uint16_t* d_target_mean_bf16,       // [>=16, 5120] (gathers accepted_nodes[0..A-1] -> rows [8..8+A-1])
    int start_pos,
    int accepted_A,
    uint32_t acc_nodes_nibbles,
    hipStream_t stream);

void launch_commit_tree_global_kv_and_r2(
    uint8_t* d_comp_fp4_0, uint8_t* d_comp_e4m3_0, uint8_t* d_idx_fp4_0, uint8_t* d_idx_ue8_0,
    uint8_t* d_comp_fp4_1, uint8_t* d_comp_e4m3_1, uint8_t* d_idx_fp4_1, uint8_t* d_idx_ue8_1,
    uint8_t* d_comp_fp4_2, uint8_t* d_comp_e4m3_2, uint8_t* d_idx_fp4_2, uint8_t* d_idx_ue8_2,
    uint8_t* d_comp_fp4_3, uint8_t* d_comp_e4m3_3, uint8_t* d_idx_fp4_3, uint8_t* d_idx_ue8_3,
    float* d_kv_state_0, float* d_sc_state_0, const float* d_kv_last_0, const float* d_sc_last_0,
    float* d_kv_state_1, float* d_sc_state_1, const float* d_kv_last_1, const float* d_sc_last_1,
    float* d_kv_state_2, float* d_sc_state_2, const float* d_kv_last_2, const float* d_sc_last_2,
    int start_pos,
    int accepted_A,
    uint32_t acc_nodes_nibbles,
    hipStream_t stream);

// 6. MoE Gate + Multi-Row Routed HLUT2 (3x MI50 TP3)
void launch_moe_gate(
    const float* d_logits_f32,    // [M, n_routed] (x.float() @ gate.weight.float()^T)
    const float* d_bias_f32,      // [n_routed]
    const float* d_bias_vl_f32,   // [n_routed] (optional)
    const uint8_t* d_image_mask,  // [M] (optional)
    int32_t* d_out_ids,           // [M, top_k]
    float* d_out_weights,         // [M, top_k]
    int M, int n_routed, int top_k, float route_scale,
    hipStream_t stream);

// Device-resident MTP pair packer: builds Mi50RoutedPair [M * top_k] directly on GPU with zero host sync!
void launch_pack_mtp_routed_pairs(
    const int32_t* d_topk_ids,    // [M, top_k]
    const float*   d_topk_weights,// [M, top_k]
    Mi50RoutedPair* d_out_pairs,  // [M * top_k]
    int M, int top_k,
    hipStream_t stream);

// Device-resident 1-CTA MTP CSR builder (<= 128 experts, zero host sync) for grouped HLUT2 execution!
void launch_build_mtp_csr(
    const int32_t* d_topk_ids,      // [M, top_k]
    const float*   d_topk_weights,  // [M, top_k]
    int32_t* d_active_experts,      // [n_experts]
    int32_t* d_csr_offsets,         // [n_experts + 1]
    Mi50RoutedPair* d_sorted_pairs, // [M * top_k]
    int M, int top_k, int n_experts,
    hipStream_t stream);

void launch_wht128_bf16_to_f16(
    const uint16_t* d_in_bf16,    // [rows, K]
    uint16_t* d_out_f16,          // [rows, K]
    const uint64_t* d_signs_u64,  // [K / 64]
    int rows, int K,
    hipStream_t stream);

void launch_mi50_routed_hlut2_pairs(
    const uint16_t* d_x_rot_f16,       // [M, 5120] FP16
    const Mi50RoutedPair* d_pairs,     // [P]
    int P,
    const uint32_t* d_gate_codes,      // [E_local, 24, 320, 32]
    const uint16_t* d_gate_scales,     // [E_local, 24, 40, 32]
    const float*    d_gate_centroids,  // [E_local, 2]
    const uint32_t* d_up_codes,        // [E_local, 24, 320, 32]
    const uint16_t* d_up_scales,       // [E_local, 24, 40, 32]
    const float*    d_up_centroids,    // [E_local, 2]
    const uint32_t* d_down_codes,      // [E_local, 160, 48, 32]
    const uint16_t* d_down_scales,     // [E_local, 160, 6, 32]
    const float*    d_down_centroids,  // [E_local, 2]
    const uint64_t* d_signs_i,         // [12] uint64 (for 768 dims)
    uint16_t* d_inter_bf16_scratch,    // [P, 768]
    uint16_t* d_inter_rot_f16_scratch, // [P, 768]
    float* d_out_accum_f32,            // [M, 5120] FP32 (atomicAdd into existing buffer!)
    hipStream_t stream);

// Grouped expert execution for large prefill chunks (M > 8): BM=32 expert-major for M >= 64 prefill,
// BM=8/16 adaptive below 64. Every unpacked 272 B HLUT2 tile is reused across 32 token rows.
void launch_mi50_routed_hlut2_grouped(
    const uint16_t* d_x_rot_f16,       // [M, 5120] FP16
    const int32_t* d_active_experts,   // [num_active] expert IDs
    const int32_t* d_csr_offsets,      // [num_active + 1] offsets into sorted_pairs
    const Mi50RoutedPair* d_sorted_pairs, // [P] sorted by expert
    int M, int num_active, int P,
    const uint32_t* d_gate_codes,
    const uint16_t* d_gate_scales,
    const float*    d_gate_centroids,
    const uint32_t* d_up_codes,
    const uint16_t* d_up_scales,
    const float*    d_up_centroids,
    const uint32_t* d_down_codes,
    const uint16_t* d_down_scales,
    const float*    d_down_centroids,
    const uint64_t* d_signs_i,
    uint16_t* d_inter_bf16_scratch,    // [P, 768]
    uint16_t* d_inter_rot_f16_scratch, // [P, 768]
    float* d_out_accum_f32,            // [M, 5120] FP32
    hipStream_t stream);

// Rootless Vega20 auto-DPM clock keeper (active ONLY during generate(), 0% idle power when inactive!)
void launch_keep_dpm_warm(
    const uint32_t* d_active_flag,
    uint32_t* d_uc_scratch,
    uint32_t mask_u32,
    hipStream_t stream);

// 7. GPU-Resident 5-Step DSpark Markov + Sliding RepPenalty + Borel Full-Partition Top-p/Greedy Sampler (1 sync instead of 19!)
static constexpr int kDsparkMaxCand = 4096;

struct DsparkGpuSampleSlot {
    uint64_t call_seq;
    int32_t  anchor_token;
    int32_t  recent_len;
    int32_t  recent_hist[1024];
    float    rep_penalty;
    float    temperature;
    float    top_p;
    float    u_rand[5];

    alignas(64) uint64_t ready_max_seq[5][3];
    alignas(64) uint64_t ready_exp_seq[5][3];
    alignas(64) uint64_t ready_cand_seq[5][3];
    alignas(64) uint64_t ready_tok_seq[5];
    float    rank_max[5][3];
    int32_t  rank_argmax[5][3];
    double   rank_sum_exp[5][3];
    double   rank_cand_mass[5][3];
    uint32_t rank_cand_cnt[5][3];
    uint32_t rank_overflow[5][3];

    int32_t  out_drafts[5];
    float    out_q_selected[5];
    int32_t  out_drafted_cnt;
    int32_t  out_fallback_step;
    float    out_conf_logits[5];
    float    out_root_top1_prob;

    // Deterministic top-B M=8 tree extraction (width [3, 2, 1, 1], Dirac delta proposals)
    alignas(64) uint64_t ready_tree_rank_seq[4][3];
    alignas(64) uint64_t ready_tree_stage_seq[4];
    float    tree_rank_val[4][3][3];
    int32_t  tree_rank_id[4][3][3];
    int32_t  tree_tokens[8];

    uint32_t kept_cnt[5];
    int32_t  kept_id[5][kDsparkMaxCand];
    float    kept_q[5][kDsparkMaxCand];

    int32_t  rank_cand_id[3][kDsparkMaxCand];
    float    rank_cand_exp[3][kDsparkMaxCand];
};

void launch_dspark_prepare_markov_emb(
    DsparkGpuSampleSlot* d_ctrl,
    const uint16_t* d_mapped_markov_embed_bf16, // [129280, 256] coherent pinned host BF16
    uint16_t* d_out_markov_emb_bf16,            // [256] local HBM2 row for step i
    uint64_t seq,
    int step_idx,
    int eos_id,
    hipStream_t stream);

void launch_dspark_step_sample_tp3(
    DsparkGpuSampleSlot* d_ctrl,
    float* d_base_logits_row_f32,               // [vocab_rows] in-place (base + markov + rep_penalty)
    float* d_markov_logits_f32,                 // [vocab_rows] in: markov logits, scratch: exp values
    uint64_t seq,
    int step_idx,
    int rank,
    int vocab_offset,
    int vocab_rows,
    int eos_id,
    hipStream_t stream);

void launch_dspark_pack_conf_in(
    const DsparkGpuSampleSlot* d_ctrl,
    const uint16_t* d_sub_bf16,                 // [5, 5120]
    const uint16_t* d_markov_sum_bf16,          // [5, 256]
    uint16_t* d_conf_in_bf16,                   // [5, 5376]
    hipStream_t stream);

void launch_dspark_prepare_tree_markov_emb(
    DsparkGpuSampleSlot* d_ctrl,
    const uint16_t* d_mapped_markov_embed_bf16, // [129280, 256] coherent pinned host BF16
    uint16_t* d_out_markov_emb_bf16,            // [num_rows, 256] local HBM2
    uint64_t seq,
    int wait_stage,                             // 0 = wait stage 0 (reads tree_tokens[1], tree_tokens[2]); 1 = wait stage 1 (reads tree_tokens[4])
    int num_rows,                               // 2 for stage 1, 1 for stage 2
    hipStream_t stream);

void launch_dspark_tree_topk_tp3(
    DsparkGpuSampleSlot* d_ctrl,
    const float* d_base_logits_row_f32,         // [vocab_rows] (for stage 0: already rep-penalized step-0 logits; for stages 1..3: saved raw base row)
    const float* d_markov_logits_row_f32,       // nullptr for stage 0; [vocab_rows] for stages 1..3
    uint64_t seq,
    int tree_stage,                             // 0..3
    int out_slot,                               // 1 (B=3), 4 (B=2), 6 (B=1), 7 (B=1)
    int B_out,                                  // 1..3
    int anc_slot1,                              // -1 or tree_tokens index (e.g. 1 or 2)
    int anc_slot2,                              // -1 or tree_tokens index (e.g. 4)
    int rank,
    int vocab_offset,
    int vocab_rows,
    hipStream_t stream);

}  // namespace dsv41
