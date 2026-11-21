#pragma once
// Wave 9.2: Lock-free coherent pinned host DDR4 channel & dedicated 4th-GPU worker engine
// for hlut3_dev = 3 (0000:85:00.0, 16 GiB switchbox MI50).
//
// Architectural invariants:
//   1. Hard runtime HBM2 guard <= 15,360 MiB (15.0 GiB) projected total usage on hlut3_dev=3.
//   2. Zero __HIP_MEMORY_SCOPE_SYSTEM RMW atomics anywhere across PCIe: intra-GPU CTA barriers
//      live in local HBM2 (__HIP_MEMORY_SCOPE_AGENT), synchronized across PCIe strictly via
//      __threadfence_system() + single-producer release/acquire sequence numbers.
//   3. Executes 100% of MTP FFN (FP8 shared_experts + 128 unsharded WHT128-HLUT3 routed experts
//      for blocks 40..42, 0 TP3 MoE all-reduce) and 720 hottest MI50 backbone experts (layers 0..39)
//      concurrently alongside TP3 (C7/88/03) and 2x Blackhole P150A.

#include "dsv41_hlut3_kernels.hpp"
#include "dsv41_wave9_3_rocblas.hpp"
#include "../src/safetensors_mmap.hpp"

#include <hip/hip_runtime.h>
#include <immintrin.h>
#include <sched.h>
#include <pthread.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace dsv41 {

class Hlut3Dev3Engine {
public:
    static constexpr int      kDevIdDefault       = 3;
    static constexpr int      kNumBackboneLayers  = 40;
    static constexpr int      kNumMtpBlocks       = 3;
    static constexpr int      kTotalBlocks        = 43;
    static constexpr int      kHidden             = 5120;
    static constexpr int      kInter              = 2304;
    static constexpr int      kTp3Inter           = 768;
    static constexpr int      kMaxChunkM          = 4096;
    static constexpr int      kMaxMtpShChunkM     = 512;
    static constexpr int      kMaxScratchPairs    = 512;
    static constexpr uint64_t kHardUsedCapBytes   = 15360ULL * 1024ULL * 1024ULL; // 15,360 MiB (15.0 GiB)

    struct BlockWeightsDev3 {
        int n_experts = 0;
        std::vector<uint16_t> logical_eids;     // [n_experts] for backbone blocks 0..39
        Hlut3Tile32x32* d_gate_tiles = nullptr; // [n_experts, 72, 160]
        Hlut3Tile32x32* d_up_tiles   = nullptr; // [n_experts, 72, 160]
        Hlut3Tile32x32* d_down_tiles = nullptr; // [n_experts, 160, 72]
        uint64_t*       d_signs_i    = nullptr; // [36] full 2304-dim WHT128 Rademacher words

        // Unsharded MTP shared_experts (populated for blocks 40..42 so MTP needs 0 TP3 MoE AllReduce)
        uint8_t* d_sh_w13_w = nullptr; // [4608, 5120] FP8
        uint8_t* d_sh_w13_s = nullptr; // [144, 160]   UE8M0
        uint8_t* d_sh_w2_w  = nullptr; // [5120, 2304] FP8
        uint8_t* d_sh_w2_s  = nullptr; // [160, 72]    UE8M0
    };

    Hlut3Dev3Engine() = default;
    ~Hlut3Dev3Engine() { shutdown(); }

    Hlut3Dev3Engine(const Hlut3Dev3Engine&) = delete;
    Hlut3Dev3Engine& operator=(const Hlut3Dev3Engine&) = delete;

    bool active() const noexcept { return active_; }
    bool has_mtp() const noexcept { return has_mtp_; }
    bool has_backbone() const noexcept { return total_backbone_experts_ > 0; }
    int  dev_id() const noexcept { return dev_id_; }
    int  total_mtp_experts() const noexcept { return total_mtp_experts_; }
    int  total_backbone_experts() const noexcept { return total_backbone_experts_; }
    int  block_expert_count(int bid) const noexcept {
        return (bid >= 0 && bid < kTotalBlocks) ? blocks_[size_t(bid)].n_experts : 0;
    }
    const std::vector<uint16_t>& block_logical_eids(int bid) const {
        return blocks_.at(size_t(bid)).logical_eids;
    }

    Hlut3OffloadSlot* mtp_fast_slot(uint32_t seq) noexcept {
        return &h_mtp_ring_[(seq - 1u) & uint32_t(kHlut3OffloadRingSlots - 1u)];
    }
    Hlut3OffloadSlot* bb_fast_slot(uint32_t seq) noexcept {
        return &h_bb_ring_[(seq - 1u) & uint32_t(kHlut3OffloadRingSlots - 1u)];
    }

    uint32_t next_mtp_fast_seq() noexcept { return ++mtp_seq_issued_; }
    uint32_t next_bb_fast_seq() noexcept { return ++bb_seq_issued_; }

    uint64_t dispatches() const noexcept { return stat_dispatches_.load(std::memory_order_relaxed); }
    double   busy_ms() const noexcept { return double(stat_busy_ns_.load(std::memory_order_relaxed)) * 1e-6; }

    // Initializes hlut3 offload GPU (dev3=85:00.0 Switch #1 or dev4=08:00.0 Switch #2), enforces <= 15,360 MiB
    // hard cap, uploads weights, and starts the dedicated coherent host-ring polling worker thread.
    void init(const std::string& overlay_path,
              const std::string& mi50_packed_root,
              int dev_id = kDevIdDefault,
              bool enable_rocblas_prefill = true,
              int rocblas_expert_thresh = RocblasPrefillEngine::kDefaultExpertThreshM,
              int pin_core = 24,
              const std::string& tensor_prefix = "")
    {
        shutdown();
        dev_id_ = dev_id;
        pin_core_ = pin_core;
        enable_rocblas_prefill_ = enable_rocblas_prefill;
        rocblas_expert_thresh_ = std::max(4, rocblas_expert_thresh);

        int dev_count = 0;
        if (hipGetDeviceCount(&dev_count) != hipSuccess || dev_id_ < 0 || dev_id_ >= dev_count) {
            throw std::runtime_error("Hlut3Dev3Engine requires HIP device " + std::to_string(dev_id_) +
                                     " (set HIP_VISIBLE_DEVICES=0,2,3,1,4); visible count=" + std::to_string(dev_count));
        }

        SafetensorsMmap ov(overlay_path);

        // Auto-detect tensor prefix ("dev3" vs "dev4") if not explicitly supplied
        std::string pfx_root = tensor_prefix;
        if (pfx_root.empty()) {
            pfx_root = "dev3";
            for (int bid = 0; bid < kTotalBlocks; ++bid) {
                if (ov.contains("dev4.b" + std::to_string(bid) + ".gate_tiles")) {
                    pfx_root = "dev4";
                    break;
                }
            }
        }
        tag_ = (pfx_root == "dev4") ? "DSV41_DEV4" : "DSV41_DEV3";

        // 1. Inspect overlay contents & compute exact projected HBM2 usage BEFORE allocating!
        uint64_t tile_bytes_needed = 0ULL;
        int mtp_blocks_found = 0;
        int bb_experts_found = 0;
        std::array<int, kTotalBlocks> exp_counts{};

        for (int bid = 0; bid < kTotalBlocks; ++bid) {
            const std::string pfx = pfx_root + ".b" + std::to_string(bid) + ".";
            if (!ov.contains(pfx + "gate_tiles") ||
                !ov.contains(pfx + "up_tiles") ||
                !ov.contains(pfx + "down_tiles")) {
                continue;
            }
            const size_t gb = size_t(ov.at(pfx + "gate_tiles").byte_size());
            const size_t ub = size_t(ov.at(pfx + "up_tiles").byte_size());
            const size_t db = size_t(ov.at(pfx + "down_tiles").byte_size());
            if (gb == 0 || gb != ub || gb != db || (gb % kHlut3UnshardedProjBytes) != 0) {
                throw std::runtime_error("Invalid " + pfx_root + " HLUT3 tile tensor byte size at block " + std::to_string(bid));
            }
            const int ne = int(gb / kHlut3UnshardedProjBytes);
            if (bid >= kNumBackboneLayers) {
                if (ne != 128) {
                    throw std::runtime_error("MTP " + pfx_root + " HLUT3 block " + std::to_string(bid) + " must have 128 experts");
                }
                ++mtp_blocks_found;
            } else {
                if (!ov.contains(pfx + "logical_eids") ||
                    size_t(ov.at(pfx + "logical_eids").byte_size()) != size_t(ne) * sizeof(uint16_t)) {
                    throw std::runtime_error("Missing/invalid logical_eids for " + pfx_root + " backbone block " + std::to_string(bid));
                }
                bb_experts_found += ne;
            }
            exp_counts[size_t(bid)] = ne;
            tile_bytes_needed += 3ULL * uint64_t(gb);
        }

        if (mtp_blocks_found != 0 && mtp_blocks_found != kNumMtpBlocks) {
            throw std::runtime_error(pfx_root + " overlay must provide either 0 or all 3 MTP blocks (40..42)");
        }

        const bool need_mtp_scratch = (mtp_blocks_found == kNumMtpBlocks);
        const uint64_t mtp_shared_bytes = need_mtp_scratch
            ? uint64_t(kNumMtpBlocks) * (2ULL * 2304ULL * 5120ULL + 2ULL * 72ULL * 160ULL +
                                         5120ULL * 2304ULL + 160ULL * 72ULL)
            : 0ULL;
        const uint64_t mtp_scratch_bytes = need_mtp_scratch
            ? (uint64_t(kMaxMtpShChunkM) * kHidden * sizeof(uint16_t) +
               uint64_t(kMaxMtpShChunkM) * (kHidden + 160ULL) +
               uint64_t(kMaxMtpShChunkM) * (2 * kInter) * sizeof(uint16_t))
            : 0ULL;
        check_hip(hipSetDevice(dev_id_), "hipSetDevice(offload)");
        size_t free_b = 0, total_b = 0;
        check_hip(hipMemGetInfo(&free_b, &total_b), "hipMemGetInfo(offload)");
        const uint64_t base_used = (total_b >= free_b) ? uint64_t(total_b - free_b) : 0ULL;

        // Empirical gfx906 finding: unsharded N_inter=2304 (4608x5120) already saturates all 60 CUs via Split-K
        // at B=1 (beats strided-batched B=4 by -216 ms @ M=4096 and preserves +135 MiB free HBM2 on dev3/dev4!).
        const int chosen_offload_max_b = 1;
        const uint64_t base_scratch_no_rb =
            uint64_t(kMaxChunkM) * kHidden * sizeof(uint16_t) +
            mtp_scratch_bytes +
            uint64_t(kMaxChunkM) * kHidden * sizeof(float) +
            uint64_t(kMaxScratchPairs) * kInter * sizeof(float) +
            1024ULL * 1024ULL;
        const uint64_t rb_bytes = enable_rocblas_prefill_ ? ((60ULL + 45ULL * uint64_t(chosen_offload_max_b)) << 20) : 0ULL;
        const uint64_t scratch_bytes = base_scratch_no_rb + rb_bytes;
        const uint64_t projected_used = base_used + tile_bytes_needed + mtp_shared_bytes + scratch_bytes;

        if (projected_used > kHardUsedCapBytes || free_b < (tile_bytes_needed + mtp_shared_bytes + scratch_bytes + (512ULL << 20))) {
            char msg[384];
            std::snprintf(msg, sizeof(msg),
                          "[%s] ABORT: projected HBM2 usage %.2f MiB exceeds 15,360.00 MiB hard guard on dev=%d (total=%.2f MiB, free=%.2f MiB)",
                          tag_.c_str(), double(projected_used) / (1024.0 * 1024.0), dev_id_,
                          double(total_b) / (1024.0 * 1024.0), double(free_b) / (1024.0 * 1024.0));
            throw std::runtime_error(msg);
        }

        int lo_pri = 0, hi_pri = 0;
        check_hip(hipDeviceGetStreamPriorityRange(&lo_pri, &hi_pri), "hipDeviceGetStreamPriorityRange(offload)");
        check_hip(hipStreamCreateWithPriority(&stream_, hipStreamNonBlocking, hi_pri), "hipStreamCreate(offload)");

        // 2. Allocate coherent mapped pinned host DDR4 rings & prefill staging buffers
        if (need_mtp_scratch) {
            void* h_mtp_raw = nullptr;
            check_hip(hipHostMalloc(&h_mtp_raw, kHlut3OffloadRingSlots * sizeof(Hlut3OffloadSlot),
                                    hipHostMallocMapped | hipHostMallocCoherent),
                      "hipHostMalloc(h_mtp_ring_)");
            std::memset(h_mtp_raw, 0, kHlut3OffloadRingSlots * sizeof(Hlut3OffloadSlot));
            h_mtp_ring_ = static_cast<Hlut3OffloadSlot*>(h_mtp_raw);
        }
        void* h_bb_raw  = nullptr;
        check_hip(hipHostMalloc(&h_bb_raw, kHlut3OffloadRingSlots * sizeof(Hlut3OffloadSlot),
                                hipHostMallocMapped | hipHostMallocCoherent),
                  "hipHostMalloc(h_bb_ring_)");
        std::memset(h_bb_raw,  0, kHlut3OffloadRingSlots * sizeof(Hlut3OffloadSlot));
        h_bb_ring_  = static_cast<Hlut3OffloadSlot*>(h_bb_raw);

        check_hip(hipHostMalloc(reinterpret_cast<void**>(&h_prefill_x_rot_f16_),
                                size_t(kMaxChunkM) * kHidden * sizeof(uint16_t), hipHostMallocDefault),
                  "hipHostMalloc(h_prefill_x_rot_f16_)");
        check_hip(hipHostMalloc(reinterpret_cast<void**>(&h_prefill_x_i8_),
                                size_t(kMaxChunkM) * kHidden * sizeof(int8_t), hipHostMallocDefault),
                  "hipHostMalloc(h_prefill_x_i8_)");
        check_hip(hipHostMalloc(reinterpret_cast<void**>(&h_prefill_x_scales_f16_),
                                size_t(kMaxChunkM) * size_t(kHidden / 128) * sizeof(uint16_t), hipHostMallocDefault),
                  "hipHostMalloc(h_prefill_x_scales_f16_)");
        if (need_mtp_scratch) {
            check_hip(hipHostMalloc(reinterpret_cast<void**>(&h_prefill_x_norm_bf16_),
                                    size_t(kMaxMtpShChunkM) * kHidden * sizeof(uint16_t), hipHostMallocDefault),
                      "hipHostMalloc(h_prefill_x_norm_bf16_)");
        }
        check_hip(hipHostMalloc(reinterpret_cast<void**>(&h_prefill_pairs_),
                                size_t(kMaxChunkM) * kHlut3OffloadMaxTopk * sizeof(Mi50RoutedPair), hipHostMallocDefault),
                  "hipHostMalloc(h_prefill_pairs_)");
        check_hip(hipHostMalloc(reinterpret_cast<void**>(&h_prefill_y_out_f32_),
                                size_t(kMaxChunkM) * kHidden * sizeof(float),
                                hipHostMallocMapped | hipHostMallocCoherent),
                  "hipHostMalloc(h_prefill_y_out_f32_)");
        check_hip(hipHostMalloc(reinterpret_cast<void**>(&h_prefill_y_out_bf16_),
                                size_t(kMaxChunkM) * kHidden * sizeof(uint16_t),
                                hipHostMallocMapped | hipHostMallocCoherent),
                  "hipHostMalloc(h_prefill_y_out_bf16_)");

        // 3. Allocate device scratch on offload GPU
        const auto sol = solve_gaussian_lloyd_max_hlut3();
        const float lv[4] = {float(sol.levels[0]), float(sol.levels[1]),
                             float(sol.levels[2]), float(sol.levels[3])};
        d_levels_        = alloc_dev<float>(4u);
        check_hip(hipMemcpy(d_levels_, lv, sizeof(lv), hipMemcpyHostToDevice), "copy d_levels_");

        d_x_rot_f16_     = alloc_dev<uint16_t>(size_t(kMaxChunkM) * kHidden);
        d_x_scales_f16_  = alloc_dev<uint16_t>(size_t(kMaxChunkM) * size_t(kHidden / 128));
        if (need_mtp_scratch) {
            d_x_norm_bf16_   = alloc_dev<uint16_t>(size_t(kMaxMtpShChunkM) * kHidden);
            d_fp8_act_       = alloc_dev<uint8_t>(size_t(kMaxMtpShChunkM) * kHidden);
            d_fp8_scale_     = alloc_dev<uint8_t>(size_t(kMaxMtpShChunkM) * (kHidden / 32));
            d_sh_gateup_bf16_= alloc_dev<uint16_t>(size_t(kMaxMtpShChunkM) * 2u * kInter);
        }
        d_out_f32_       = alloc_dev<float>(size_t(kMaxChunkM) * kHidden);
        d_inter_z_f32_   = alloc_dev<float>(size_t(kMaxScratchPairs) * kInter);
        d_pairs_         = alloc_dev<Mi50RoutedPair>(size_t(kMaxChunkM) * kHlut3OffloadMaxTopk);
        d_active_exp_    = alloc_dev<int32_t>(256u);
        d_csr_offsets_   = alloc_dev<int32_t>(257u);
        d_num_act_       = alloc_dev<int32_t>(4u);
        d_publish_sync_  = alloc_dev<uint32_t>(2u * kHlut3OffloadRingSlots * 16u);
        check_hip(hipMemset(d_publish_sync_, 0, 2u * kHlut3OffloadRingSlots * 16u * sizeof(uint32_t)), "zero d_publish_sync_");

        // 4. Upload per-block full 2304-dim WHT128 signs_i, HLUT3 tiles, and MTP unsharded FP8 shared_experts
        for (int bid = 0; bid < kTotalBlocks; ++bid) {
            const int ne = exp_counts[size_t(bid)];
            if (ne <= 0) continue;
            auto& b = blocks_[size_t(bid)];
            b.n_experts = ne;

            std::vector<uint64_t> signs_i_full;
            signs_i_full.reserve(36u);
            char tag[64];
            if (bid < kNumBackboneLayers) std::snprintf(tag, sizeof(tag), "/layer_%02d.safetensors", bid);
            else                          std::snprintf(tag, sizeof(tag), "/mtp_%d.safetensors", bid - kNumBackboneLayers);

            std::array<std::unique_ptr<SafetensorsMmap>, 3> tp_st;
            for (int r = 0; r < 3; ++r) {
                tp_st[size_t(r)] = std::make_unique<SafetensorsMmap>(
                    mi50_packed_root + "/tp" + std::to_string(r) + tag);
                const auto* si = reinterpret_cast<const uint64_t*>(tp_st[size_t(r)]->tensor_bytes("wht.signs_i"));
                signs_i_full.insert(signs_i_full.end(), si, si + 12);
            }
            b.d_signs_i = alloc_dev<uint64_t>(36u);
            check_hip(hipMemcpy(b.d_signs_i, signs_i_full.data(), 36u * sizeof(uint64_t), hipMemcpyHostToDevice),
                      "copy d_signs_i");

            const std::string pfx = pfx_root + ".b" + std::to_string(bid) + ".";
            if (bid < kNumBackboneLayers) {
                const auto* eids_ptr = reinterpret_cast<const uint16_t*>(ov.tensor_bytes(pfx + "logical_eids"));
                b.logical_eids.assign(eids_ptr, eids_ptr + ne);
            }

            const size_t n_tiles = size_t(ne) * kHlut3UnshardedProjTiles;
            const size_t proj_b  = n_tiles * sizeof(Hlut3Tile32x32);
            b.d_gate_tiles = alloc_dev<Hlut3Tile32x32>(n_tiles);
            b.d_up_tiles   = alloc_dev<Hlut3Tile32x32>(n_tiles);
            b.d_down_tiles = alloc_dev<Hlut3Tile32x32>(n_tiles);
            check_hip(hipMemcpy(b.d_gate_tiles, ov.tensor_bytes(pfx + "gate_tiles"), proj_b, hipMemcpyHostToDevice), "copy gate_tiles");
            check_hip(hipMemcpy(b.d_up_tiles,   ov.tensor_bytes(pfx + "up_tiles"),   proj_b, hipMemcpyHostToDevice), "copy up_tiles");
            check_hip(hipMemcpy(b.d_down_tiles, ov.tensor_bytes(pfx + "down_tiles"), proj_b, hipMemcpyHostToDevice), "copy down_tiles");

            if (bid >= kNumBackboneLayers) {
                // Stitch TP3-sliced FP8 shared_experts into unsharded [4608, 5120] w13 and [5120, 2304] w2
                constexpr size_t kW1RankBytes = size_t(kTp3Inter) * kHidden;        // 3,932,160 B
                constexpr size_t kS1RankBytes = size_t(kTp3Inter / 32) * (kHidden / 32); // 3,840 B
                std::vector<uint8_t> w13_w(6u * kW1RankBytes);
                std::vector<uint8_t> w13_s(6u * kS1RankBytes);
                for (int r = 0; r < 3; ++r) {
                    std::memcpy(w13_w.data() + size_t(r) * kW1RankBytes,
                                tp_st[size_t(r)]->tensor_bytes("ffn.shared_experts.w1.weight"), kW1RankBytes);
                    std::memcpy(w13_w.data() + size_t(3 + r) * kW1RankBytes,
                                tp_st[size_t(r)]->tensor_bytes("ffn.shared_experts.w3.weight"), kW1RankBytes);
                    std::memcpy(w13_s.data() + size_t(r) * kS1RankBytes,
                                tp_st[size_t(r)]->tensor_bytes("ffn.shared_experts.w1.scale"), kS1RankBytes);
                    std::memcpy(w13_s.data() + size_t(3 + r) * kS1RankBytes,
                                tp_st[size_t(r)]->tensor_bytes("ffn.shared_experts.w3.scale"), kS1RankBytes);
                }
                std::vector<uint8_t> w2_w(size_t(kHidden) * kInter);
                std::vector<uint8_t> w2_s(size_t(kHidden / 32) * (kInter / 32));
                for (int row = 0; row < kHidden; ++row) {
                    for (int r = 0; r < 3; ++r) {
                        const uint8_t* src = reinterpret_cast<const uint8_t*>(
                            tp_st[size_t(r)]->tensor_bytes("ffn.shared_experts.w2.weight")) + size_t(row) * kTp3Inter;
                        std::memcpy(w2_w.data() + size_t(row) * kInter + size_t(r) * kTp3Inter, src, kTp3Inter);
                    }
                }
                for (int rb = 0; rb < kHidden / 32; ++rb) {
                    for (int r = 0; r < 3; ++r) {
                        const uint8_t* src = reinterpret_cast<const uint8_t*>(
                            tp_st[size_t(r)]->tensor_bytes("ffn.shared_experts.w2.scale")) + size_t(rb) * (kTp3Inter / 32);
                        std::memcpy(w2_s.data() + size_t(rb) * (kInter / 32) + size_t(r) * (kTp3Inter / 32),
                                    src, kTp3Inter / 32);
                    }
                }
                b.d_sh_w13_w = alloc_dev<uint8_t>(w13_w.size());
                b.d_sh_w13_s = alloc_dev<uint8_t>(w13_s.size());
                b.d_sh_w2_w  = alloc_dev<uint8_t>(w2_w.size());
                b.d_sh_w2_s  = alloc_dev<uint8_t>(w2_s.size());
                check_hip(hipMemcpy(b.d_sh_w13_w, w13_w.data(), w13_w.size(), hipMemcpyHostToDevice), "copy mtp w13_w");
                check_hip(hipMemcpy(b.d_sh_w13_s, w13_s.data(), w13_s.size(), hipMemcpyHostToDevice), "copy mtp w13_s");
                check_hip(hipMemcpy(b.d_sh_w2_w,  w2_w.data(),  w2_w.size(),  hipMemcpyHostToDevice), "copy mtp w2_w");
                check_hip(hipMemcpy(b.d_sh_w2_s,  w2_s.data(),  w2_s.size(),  hipMemcpyHostToDevice), "copy mtp w2_s");
            }
        }

        static std::mutex rb_prewarm_mtx;
        {
            std::lock_guard<std::mutex> rb_lk(rb_prewarm_mtx);
            check_hip(hipSetDevice(dev_id_), "hipSetDevice(rb_init)");
            if (enable_rocblas_prefill_) {
                rb_prefill_.init(dev_id_, stream_, /*is_dev3=*/true, /*max_bucket_m=*/1024, chosen_offload_max_b);
            }
        }

        size_t free_after = 0, total_after = 0;
        check_hip(hipMemGetInfo(&free_after, &total_after), "hipMemGetInfo(dev3 after)");
        const uint64_t actual_used = (total_after >= free_after) ? uint64_t(total_after - free_after) : 0ULL;
        if (actual_used > kHardUsedCapBytes) {
            shutdown();
            throw std::runtime_error("[" + tag_ + "] ABORT: actual HBM2 used exceeded 15,360 MiB hard guard");
        }

        has_mtp_ = (mtp_blocks_found == kNumMtpBlocks);
        total_mtp_experts_ = mtp_blocks_found * 128;
        total_backbone_experts_ = bb_experts_found;

        // Pre-warm HIP fatbin code objects on offload GPU at boot before worker_ or DpmWarmGuard starts!
        {
            std::lock_guard<std::mutex> rb_lk(rb_prewarm_mtx);
            check_hip(hipSetDevice(dev_id_), "hipSetDevice(prewarm)");
            int warm_bid = -1;
            for (int bid = 0; bid < kTotalBlocks; ++bid) {
                if (blocks_[size_t(bid)].n_experts > 0) { warm_bid = bid; break; }
            }
            if (warm_bid >= 0) {
                const auto& wb = blocks_[size_t(warm_bid)];
                Hlut3OffloadSlot* warm_slot = h_mtp_ring_ ? &h_mtp_ring_[0] : &h_bb_ring_[0];
                warm_slot->topk_ids[0] = -1;
                warm_slot->topk_weights[0] = 0.0f;
                launch_dev3_ingest_and_pack_csr(
                    warm_slot, d_x_rot_f16_, d_x_norm_bf16_,
                    d_active_exp_, d_csr_offsets_, d_pairs_, d_num_act_,
                    1, 1, 0, wb.n_experts, (wb.d_sh_w13_w != nullptr), stream_);
                if (wb.d_sh_w13_w != nullptr && d_x_norm_bf16_ != nullptr) {
                    run_mtp_shared_fp8(wb, 1);
                } else if (d_x_norm_bf16_ != nullptr) {
                    launch_act_quant_ue8m0_32(d_x_norm_bf16_, d_fp8_act_, d_fp8_scale_, nullptr, 1, kHidden, false, stream_);
                }
                launch_hlut3_fdot2_grouped_f16(
                    d_x_rot_f16_, d_active_exp_, d_csr_offsets_, d_pairs_, d_num_act_,
                    1, wb.n_experts, kInter,
                    wb.d_gate_tiles, wb.d_up_tiles, wb.d_down_tiles,
                    d_levels_, wb.d_signs_i,
                    reinterpret_cast<uint16_t*>(d_inter_z_f32_), d_out_f32_, stream_);
                if (rb_prefill_.active()) {
                    Mi50RoutedPair dummy_pr{0, 0, 0.0f, 0};
                    check_hip(hipMemcpyAsync(d_pairs_, &dummy_pr, sizeof(dummy_pr), hipMemcpyHostToDevice, stream_), "cp dummy");
                    rb_prefill_.run_hlut3_expert_bucket(
                        d_x_rot_f16_, d_pairs_, 0, 0, 1, kInter,
                        wb.d_gate_tiles, wb.d_up_tiles, wb.d_down_tiles,
                        d_levels_, wb.d_signs_i, d_out_f32_, stream_);
                }
                launch_dev3_publish_f32(d_out_f32_, warm_slot, d_publish_sync_, 1, 1u, stream_);
                check_hip(hipStreamSynchronize(stream_), "prewarm offload kernels");
                if (h_mtp_ring_) std::memset(h_mtp_ring_, 0, kHlut3OffloadRingSlots * sizeof(Hlut3OffloadSlot));
                std::memset(h_bb_ring_,  0, kHlut3OffloadRingSlots * sizeof(Hlut3OffloadSlot));
                check_hip(hipMemset(d_publish_sync_, 0, 2u * kHlut3OffloadRingSlots * 16u * sizeof(uint32_t)), "re-zero d_publish_sync_");
                check_hip(hipStreamSynchronize(nullptr), "sync stream0 after offload re-zero");
            }
        }

        active_ = true;
        stop_worker_.store(false, std::memory_order_relaxed);
        mtp_seq_issued_ = 0u;
        bb_seq_issued_  = 0u;
        prefill_req_seq_.store(0u, std::memory_order_relaxed);
        prefill_done_seq_.store(0u, std::memory_order_relaxed);

        worker_ = std::thread([this]() { worker_loop(); });

        std::printf("[%s] Offload GPU hlut3_dev=%d (core=%d, rb_B=%d) READY: mtp_experts=%d backbone_experts=%d HBM=%.2f / %.2f GiB used (free=%.1f MiB, cap=15360 MiB)\n",
                    tag_.c_str(), dev_id_, pin_core_, chosen_offload_max_b, total_mtp_experts_, total_backbone_experts_,
                    double(actual_used) / double(1ULL << 30),
                    double(total_after) / double(1ULL << 30),
                    double(free_after) / (1024.0 * 1024.0));
        std::fflush(stdout);
        check_hip(hipSetDevice(0), "restore hipSetDevice(0)");
    }

    int8_t*   prefill_x_i8_staging() noexcept { return h_prefill_x_i8_; }
    uint16_t* prefill_x_scales_staging() noexcept { return h_prefill_x_scales_f16_; }

    // Async prefill submission for M > 64 (up to kMaxChunkM = 4096) using legacy FP16/BF16 staging (MTP / fallback).
    uint32_t submit_prefill_async(
        int block_id,
        int M,
        const uint16_t* h_x_rot,
        const uint16_t* h_x_norm_bf16,
        const Mi50RoutedPair* h_local_pairs,
        int n_pairs,
        bool include_mtp_shared,
        bool x_is_bf16 = false)
    {
        if (!active_ || M <= 0 || M > kMaxChunkM) {
            throw std::runtime_error("submit_prefill_async invalid M or inactive Hlut3Dev3Engine");
        }
        std::memcpy(h_prefill_x_rot_f16_, h_x_rot, size_t(M) * kHidden * sizeof(uint16_t));
        if (include_mtp_shared && h_x_norm_bf16 != nullptr) {
            std::memcpy(h_prefill_x_norm_bf16_, h_x_norm_bf16, size_t(M) * kHidden * sizeof(uint16_t));
        }
        if (n_pairs > 0 && h_local_pairs != nullptr) {
            std::memcpy(h_prefill_pairs_, h_local_pairs, size_t(n_pairs) * sizeof(Mi50RoutedPair));
        }
        prefill_block_id_  = block_id;
        prefill_rows_      = M;
        prefill_n_pairs_   = n_pairs;
        prefill_mtp_sh_    = include_mtp_shared;
        prefill_x_is_bf16_ = x_is_bf16;
        prefill_x_is_int8_ = false;
        prefill_out_bf16_  = false;
        const uint32_t seq = prefill_req_seq_.load(std::memory_order_relaxed) + 1u;
        prefill_req_seq_.store(seq, std::memory_order_release);
        return seq;
    }

    // Wave 9.4 Phase C: Zero-CPU-activation-copy INT8+FP16-scale H2D + BF16 D2H async prefill submission!
    // Caller (TP0 router_stream) has ALREADY DMA'd [M, 5120] INT8 into prefill_x_i8_staging() and
    // [M] FP16 row scales into prefill_x_scales_staging() before ev_router_done fired (0 ms CPU activation copy!).
    uint32_t submit_prefill_int8_async(
        int block_id,
        int M,
        const Mi50RoutedPair* h_local_pairs,
        int n_pairs,
        bool out_bf16 = true)
    {
        return submit_prefill_int8_shared_async(
            block_id, M, h_prefill_x_i8_, h_prefill_x_scales_f16_, h_local_pairs, n_pairs, out_bf16);
    }

    // Wave 10 Phase B: Zero-duplicate shared INT8-BLK128 input staging read concurrently by dev3 & dev4!
    uint32_t submit_prefill_int8_shared_async(
        int block_id,
        int M,
        const int8_t* shared_x_i8,
        const uint16_t* shared_x_scales_f16,
        const Mi50RoutedPair* h_local_pairs,
        int n_pairs,
        bool out_bf16 = true)
    {
        if (!active_ || M <= 0 || M > kMaxChunkM) {
            throw std::runtime_error("submit_prefill_int8_shared_async invalid M or inactive Hlut3Dev3Engine");
        }
        if (n_pairs > 0 && h_local_pairs != nullptr) {
            std::memcpy(h_prefill_pairs_, h_local_pairs, size_t(n_pairs) * sizeof(Mi50RoutedPair));
        }
        prefill_block_id_     = block_id;
        prefill_rows_         = M;
        prefill_n_pairs_      = n_pairs;
        prefill_mtp_sh_       = false;
        prefill_x_is_bf16_    = false;
        prefill_x_is_int8_    = true;
        prefill_out_bf16_     = out_bf16;
        prefill_x_i8_src_     = shared_x_i8 ? shared_x_i8 : h_prefill_x_i8_;
        prefill_x_scales_src_ = shared_x_scales_f16 ? shared_x_scales_f16 : h_prefill_x_scales_f16_;
        const uint32_t seq = prefill_req_seq_.load(std::memory_order_relaxed) + 1u;
        prefill_req_seq_.store(seq, std::memory_order_release);
        return seq;
    }

    const float* wait_prefill_output(uint32_t seq) {
        while (static_cast<int32_t>(prefill_done_seq_.load(std::memory_order_acquire) - seq) < 0) {
            _mm_pause();
        }
        return h_prefill_y_out_f32_;
    }

    const uint16_t* wait_prefill_output_bf16(uint32_t seq) {
        while (static_cast<int32_t>(prefill_done_seq_.load(std::memory_order_acquire) - seq) < 0) {
            _mm_pause();
        }
        return h_prefill_y_out_bf16_;
    }

    void shutdown() {
        if (worker_.joinable()) {
            stop_worker_.store(true, std::memory_order_release);
            worker_.join();
        }
        rb_prefill_.shutdown();
        if (!dev_allocs_.empty() || stream_ != nullptr) {
            (void)hipSetDevice(dev_id_);
            if (stream_ != nullptr) {
                (void)hipStreamSynchronize(stream_);
                (void)hipStreamDestroy(stream_);
                stream_ = nullptr;
            }
            for (void* p : dev_allocs_) {
                if (p != nullptr) (void)hipFree(p);
            }
            dev_allocs_.clear();
        }
        if (h_mtp_ring_ != nullptr) { (void)hipHostFree(h_mtp_ring_); h_mtp_ring_ = nullptr; }
        if (h_bb_ring_  != nullptr) { (void)hipHostFree(h_bb_ring_);  h_bb_ring_  = nullptr; }
        if (h_prefill_x_rot_f16_ != nullptr)    { (void)hipHostFree(h_prefill_x_rot_f16_);    h_prefill_x_rot_f16_ = nullptr; }
        if (h_prefill_x_i8_ != nullptr)         { (void)hipHostFree(h_prefill_x_i8_);         h_prefill_x_i8_ = nullptr; }
        if (h_prefill_x_scales_f16_ != nullptr) { (void)hipHostFree(h_prefill_x_scales_f16_); h_prefill_x_scales_f16_ = nullptr; }
        if (h_prefill_x_norm_bf16_ != nullptr)  { (void)hipHostFree(h_prefill_x_norm_bf16_);  h_prefill_x_norm_bf16_ = nullptr; }
        if (h_prefill_pairs_ != nullptr)        { (void)hipHostFree(h_prefill_pairs_);        h_prefill_pairs_ = nullptr; }
        if (h_prefill_y_out_f32_ != nullptr)    { (void)hipHostFree(h_prefill_y_out_f32_);    h_prefill_y_out_f32_ = nullptr; }
        if (h_prefill_y_out_bf16_ != nullptr)   { (void)hipHostFree(h_prefill_y_out_bf16_);   h_prefill_y_out_bf16_ = nullptr; }
        d_x_norm_bf16_ = nullptr;
        d_fp8_act_ = nullptr;
        d_fp8_scale_ = nullptr;
        d_sh_gateup_bf16_ = nullptr;
        blocks_ = {};
        active_ = false;
        has_mtp_ = false;
        total_mtp_experts_ = 0;
        total_backbone_experts_ = 0;
    }

private:
    static void check_hip(hipError_t err, const char* what) {
        if (err != hipSuccess) {
            throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
        }
    }

    template <typename T>
    T* alloc_dev(size_t count) {
        if (count == 0) return nullptr;
        void* p = nullptr;
        check_hip(hipMalloc(&p, count * sizeof(T)), "hipMalloc(dev3)");
        dev_allocs_.push_back(p);
        return static_cast<T*>(p);
    }

    void run_mtp_shared_fp8(const BlockWeightsDev3& b, int M) {
        launch_act_quant_ue8m0_32(d_x_norm_bf16_, d_fp8_act_, d_fp8_scale_, nullptr, M, kHidden, false, stream_);
        if (M < 1024 || !rb_prefill_.run_fp8_gemm(d_fp8_act_, d_fp8_scale_, b.d_sh_w13_w, b.d_sh_w13_s,
                                      d_sh_gateup_bf16_, nullptr, M, 2 * kInter, kHidden, false, false, stream_)) {
            launch_fp8_gemm(d_fp8_act_, d_fp8_scale_, b.d_sh_w13_w, b.d_sh_w13_s, d_sh_gateup_bf16_, nullptr,
                            M, 2 * kInter, kHidden, false, false, stream_);
        }
        launch_swiglu10_split_quant_ue8m0_32(d_sh_gateup_bf16_, d_fp8_act_, d_fp8_scale_, M, kInter, stream_);
        if (M < 1024 || !rb_prefill_.run_fp8_gemm(d_fp8_act_, d_fp8_scale_, b.d_sh_w2_w, b.d_sh_w2_s,
                                      nullptr, d_out_f32_, M, kHidden, kInter, true, false, stream_)) {
            launch_fp8_gemm(d_fp8_act_, d_fp8_scale_, b.d_sh_w2_w, b.d_sh_w2_s, nullptr, d_out_f32_,
                            M, kHidden, kInter, true, false, stream_);
        }
    }

    void process_fast_slot(Hlut3OffloadSlot* slot, uint32_t seq, size_t sync_base_off) {
        const auto t0 = std::chrono::steady_clock::now();
        const int bid       = int(slot->block_id);
        const int M         = int(slot->rows);
        const int top_k     = int(slot->top_k);
        const int exp_begin = int(slot->exp_begin);
        const int exp_end   = int(slot->exp_end);
        const bool mtp_sh   = (slot->include_mtp_shared != 0u);
        const auto& b       = blocks_[size_t(bid)];

        // Wave 9.3 Phase D: Single-launch coherent ingest + LDS expert-grouped CSR builder +
        // zero-scratch v_dot2_f32_f16 (__builtin_amdgcn_fdot2) fused Gate+Up+SwiGLU+WHT128 + Down!
        launch_dev3_ingest_and_pack_csr(
            slot, d_x_rot_f16_, d_x_norm_bf16_,
            d_active_exp_, d_csr_offsets_, d_pairs_, d_num_act_,
            M, top_k, exp_begin, exp_end, mtp_sh, stream_);

        if (mtp_sh && b.d_sh_w13_w != nullptr) {
            run_mtp_shared_fp8(b, M);
        } else {
            (void)hipMemsetAsync(d_out_f32_, 0, size_t(M) * kHidden * sizeof(float), stream_);
        }

        if (b.n_experts > 0 && exp_end > exp_begin) {
            const int P = M * top_k;
            const int max_act = std::min(b.n_experts, P);
            launch_hlut3_fdot2_grouped_f16(
                d_x_rot_f16_, d_active_exp_, d_csr_offsets_, d_pairs_, d_num_act_,
                max_act, b.n_experts, kInter,
                b.d_gate_tiles, b.d_up_tiles, b.d_down_tiles,
                d_levels_, b.d_signs_i,
                reinterpret_cast<uint16_t*>(d_inter_z_f32_), d_out_f32_, stream_);
        }

        uint32_t* d_sync = d_publish_sync_ + sync_base_off + ((seq - 1u) & uint32_t(kHlut3OffloadRingSlots - 1u)) * 16u;
        launch_dev3_publish_f32(d_out_f32_, slot, d_sync, M, seq, stream_);
        // Wave 9.3 Phase D.3: Non-blocking AQL doorbell flush (0 CPU sync stall between slots; in-order stream_ + 8-slot ring!)
        (void)hipStreamQuery(stream_);

        const auto dt_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count());
        stat_dispatches_.fetch_add(1ULL, std::memory_order_relaxed);
        stat_busy_ns_.fetch_add(dt_ns, std::memory_order_relaxed);
    }

    void worker_loop() {
        // Wave 9.3 / Wave 10: Pin offload worker thread (dev3 -> core 24, p150 -> core 25, dev4 -> core 26)
        if (pin_core_ >= 0) {
            cpu_set_t cs;
            CPU_ZERO(&cs);
            CPU_SET(pin_core_, &cs);
            (void)::pthread_setaffinity_np(::pthread_self(), sizeof(cs), &cs);
        }
        (void)hipSetDevice(dev_id_);
        uint32_t next_mtp  = 1u;
        uint32_t next_bb   = 1u;
        uint32_t next_pref = 1u;

        std::vector<int32_t> h_act;
        std::vector<int32_t> h_off;
        std::vector<Mi50RoutedPair> h_sorted;
        std::vector<Mi50RoutedPair> h_rb_pairs;
        std::vector<Mi50RoutedPair> flat_sorted;
        std::vector<RocblasPrefillEngine::ExpertBucketDesc> rb_descs;
        h_act.reserve(256u);
        h_off.reserve(257u);
        h_sorted.reserve(kMaxScratchPairs);
        h_rb_pairs.reserve(size_t(kMaxChunkM) * kHlut3OffloadMaxTopk);
        flat_sorted.resize(size_t(kMaxChunkM) * kHlut3OffloadMaxTopk);
        rb_descs.reserve(256u);

        while (!stop_worker_.load(std::memory_order_acquire)) {
            if (h_mtp_ring_ != nullptr) {
                Hlut3OffloadSlot* mslot = &h_mtp_ring_[(next_mtp - 1u) & uint32_t(kHlut3OffloadRingSlots - 1u)];
                if (static_cast<int32_t>(__atomic_load_n(&mslot->ready_seq, __ATOMIC_ACQUIRE) - next_mtp) >= 0) {
                    process_fast_slot(mslot, next_mtp, 0u);
                    ++next_mtp;
                    continue;
                }
            }

            Hlut3OffloadSlot* bslot = &h_bb_ring_[(next_bb - 1u) & uint32_t(kHlut3OffloadRingSlots - 1u)];
            if (static_cast<int32_t>(__atomic_load_n(&bslot->ready_seq, __ATOMIC_ACQUIRE) - next_bb) >= 0) {
                process_fast_slot(bslot, next_bb, kHlut3OffloadRingSlots * 16u);
                ++next_bb;
                continue;
            }

            const uint32_t pseq = prefill_req_seq_.load(std::memory_order_acquire);
            if (static_cast<int32_t>(pseq - next_pref) >= 0) {
                const auto t0 = std::chrono::steady_clock::now();
                const int bid       = prefill_block_id_;
                const int M         = prefill_rows_;
                const int n_pairs   = prefill_n_pairs_;
                const bool mtp_sh   = prefill_mtp_sh_;
                const bool x_bf16   = prefill_x_is_bf16_;
                const bool x_int8   = prefill_x_is_int8_;
                const bool out_bf16 = prefill_out_bf16_;
                const auto& b       = blocks_[size_t(bid)];

                if (x_int8) {
                    // Alias d_out_f32_ (80 MiB, unused before hipMemsetAsync!) for 2x smaller INT8 H2D wire -> 0 B extra HBM2!
                    int8_t* d_i8_alias = reinterpret_cast<int8_t*>(d_out_f32_);
                    (void)hipMemcpyAsync(d_i8_alias, prefill_x_i8_src_,
                                         size_t(M) * kHidden * sizeof(int8_t), hipMemcpyHostToDevice, stream_);
                    (void)hipMemcpyAsync(d_x_scales_f16_, prefill_x_scales_src_,
                                         size_t(M) * size_t(kHidden / 128) * sizeof(uint16_t), hipMemcpyHostToDevice, stream_);
                    launch_w94_dequant_int8_to_f16_rowwise(d_i8_alias, d_x_scales_f16_, d_x_rot_f16_, M, kHidden, stream_);
                } else {
                    (void)hipMemcpyAsync(d_x_rot_f16_, h_prefill_x_rot_f16_,
                                         size_t(M) * kHidden * sizeof(uint16_t), hipMemcpyHostToDevice, stream_);
                    if (x_bf16) {
                        launch_w93_bf16_to_f16(d_x_rot_f16_, d_x_rot_f16_, size_t(M) * kHidden, stream_);
                    }
                }
                if (mtp_sh && b.d_sh_w13_w != nullptr) {
                    (void)hipMemcpyAsync(d_x_norm_bf16_, h_prefill_x_norm_bf16_,
                                         size_t(M) * kHidden * sizeof(uint16_t), hipMemcpyHostToDevice, stream_);
                    run_mtp_shared_fp8(b, M);
                } else {
                    (void)hipMemsetAsync(d_out_f32_, 0, size_t(M) * kHidden * sizeof(float), stream_);
                }

                if (n_pairs > 0 && b.n_experts > 0) {
                    // Wave 9.4 Phase C: Flat pre-allocated O(n) counting sort (0 heap allocations per layer!)
                    const int ne_lim = std::min(b.n_experts, 256);
                    std::array<int, 256> counts{};
                    std::array<int, 257> offsets{};
                    std::array<int, 256> cursors{};
                    for (int i = 0; i < n_pairs; ++i) {
                        const auto& pr = h_prefill_pairs_[i];
                        if (pr.expert >= 0 && pr.expert < ne_lim && pr.weight != 0.0f) {
                            ++counts[size_t(pr.expert)];
                        }
                    }
                    int total_valid = 0;
                    for (int e = 0; e < ne_lim; ++e) {
                        offsets[size_t(e)] = total_valid;
                        cursors[size_t(e)] = total_valid;
                        total_valid += counts[size_t(e)];
                    }
                    offsets[size_t(ne_lim)] = total_valid;
                    for (int i = 0; i < n_pairs; ++i) {
                        const auto& pr = h_prefill_pairs_[i];
                        if (pr.expert >= 0 && pr.expert < ne_lim && pr.weight != 0.0f) {
                            flat_sorted[size_t(cursors[size_t(pr.expert)]++)] = pr;
                        }
                    }

                    // Dispatch hot expert buckets (m_e >= rocblas_expert_thresh_) via B=3/4 Strided-Batched HHHS rocBLAS!
                    const bool use_rb = (rb_prefill_.active() && M >= RocblasPrefillEngine::kDefaultDenseThreshM);
                    if (use_rb) {
                        h_rb_pairs.clear();
                        rb_descs.clear();
                        for (int e = 0; e < ne_lim; ++e) {
                            const int cnt = counts[size_t(e)];
                            if (cnt >= rocblas_expert_thresh_) {
                                rb_descs.push_back(RocblasPrefillEngine::ExpertBucketDesc{e, 0, cnt});
                                counts[size_t(e)] = 0;
                            }
                        }
                        if (!rb_descs.empty()) {
                            std::sort(rb_descs.begin(), rb_descs.end(),
                                      [](const auto& a, const auto& b) {
                                          return (a.m_e != b.m_e) ? (a.m_e > b.m_e) : (a.eid < b.eid);
                                      });
                            for (auto& bk : rb_descs) {
                                bk.p_start = int(h_rb_pairs.size());
                                const int src0 = offsets[size_t(bk.eid)];
                                h_rb_pairs.insert(h_rb_pairs.end(),
                                                  flat_sorted.begin() + src0,
                                                  flat_sorted.begin() + src0 + bk.m_e);
                            }
                            (void)hipMemcpyAsync(d_pairs_, h_rb_pairs.data(),
                                                 h_rb_pairs.size() * sizeof(Mi50RoutedPair),
                                                 hipMemcpyHostToDevice, stream_);
                            rb_prefill_.run_hlut3_expert_buckets_batched(
                                d_x_rot_f16_, d_pairs_,
                                rb_descs.data(), int(rb_descs.size()), kInter,
                                b.d_gate_tiles, b.d_up_tiles, b.d_down_tiles,
                                d_levels_, b.d_signs_i, d_out_f32_, stream_);
                        }
                    }

                    // Remaining sparse tail experts (m_e < rocblas_expert_thresh_) via grouped HLUT3 CSR
                    std::array<int, 256> rem_pos{};
                    for (int e = 0; e < ne_lim; ++e) rem_pos[size_t(e)] = offsets[size_t(e)];
                    for (int e = 0; e < ne_lim;) {
                        h_act.clear();
                        h_off.clear();
                        h_sorted.clear();
                        h_off.push_back(0);
                        while (e < ne_lim) {
                            const int rem = counts[size_t(e)];
                            if (rem <= 0) { ++e; continue; }
                            if (int(h_sorted.size()) + rem <= kMaxScratchPairs) {
                                h_act.push_back(e);
                                const int p0 = rem_pos[size_t(e)];
                                h_sorted.insert(h_sorted.end(),
                                                flat_sorted.begin() + p0,
                                                flat_sorted.begin() + p0 + rem);
                                h_off.push_back(int(h_sorted.size()));
                                counts[size_t(e)] = 0;
                                ++e;
                            } else if (h_sorted.empty()) {
                                h_act.push_back(e);
                                const int p0 = rem_pos[size_t(e)];
                                h_sorted.insert(h_sorted.end(),
                                                flat_sorted.begin() + p0,
                                                flat_sorted.begin() + p0 + kMaxScratchPairs);
                                rem_pos[size_t(e)] += kMaxScratchPairs;
                                counts[size_t(e)] -= kMaxScratchPairs;
                                h_off.push_back(int(h_sorted.size()));
                                break;
                            } else {
                                break;
                            }
                        }
                        if (!h_sorted.empty()) {
                            (void)hipMemcpyAsync(d_active_exp_, h_act.data(), h_act.size() * sizeof(int32_t), hipMemcpyHostToDevice, stream_);
                            (void)hipMemcpyAsync(d_csr_offsets_, h_off.data(), h_off.size() * sizeof(int32_t), hipMemcpyHostToDevice, stream_);
                            (void)hipMemcpyAsync(d_pairs_, h_sorted.data(), h_sorted.size() * sizeof(Mi50RoutedPair), hipMemcpyHostToDevice, stream_);
                            launch_hlut3_fdot2_grouped_f16(
                                d_x_rot_f16_, d_active_exp_, d_csr_offsets_, d_pairs_, nullptr,
                                int(h_act.size()), b.n_experts, kInter,
                                b.d_gate_tiles, b.d_up_tiles, b.d_down_tiles,
                                d_levels_, b.d_signs_i,
                                reinterpret_cast<uint16_t*>(d_inter_z_f32_), d_out_f32_, stream_);
                        }
                    }
                }

                if (out_bf16) {
                    // Alias dead d_x_rot_f16_ (40 MiB) to return 2x smaller BF16 output over PCIe switchbox!
                    launch_w93_f32_to_bf16_rne(d_out_f32_, d_x_rot_f16_, size_t(M) * kHidden, stream_);
                    (void)hipMemcpyAsync(h_prefill_y_out_bf16_, d_x_rot_f16_,
                                         size_t(M) * kHidden * sizeof(uint16_t), hipMemcpyDeviceToHost, stream_);
                } else {
                    (void)hipMemcpyAsync(h_prefill_y_out_f32_, d_out_f32_,
                                         size_t(M) * kHidden * sizeof(float), hipMemcpyDeviceToHost, stream_);
                }
                (void)hipStreamSynchronize(stream_);
                prefill_done_seq_.store(next_pref, std::memory_order_release);

                const auto dt_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0).count());
                stat_dispatches_.fetch_add(1ULL, std::memory_order_relaxed);
                stat_busy_ns_.fetch_add(dt_ns, std::memory_order_relaxed);
                ++next_pref;
                continue;
            }

            _mm_pause();
        }
    }

    int dev_id_ = kDevIdDefault;
    int pin_core_ = 24;
    std::string tag_ = "DSV41_DEV3";
    bool active_ = false;
    bool has_mtp_ = false;
    bool enable_rocblas_prefill_ = true;
    int rocblas_expert_thresh_ = RocblasPrefillEngine::kDefaultExpertThreshM;
    int total_mtp_experts_ = 0;
    int total_backbone_experts_ = 0;
    RocblasPrefillEngine rb_prefill_;

    hipStream_t stream_ = nullptr;
    std::vector<void*> dev_allocs_;
    std::array<BlockWeightsDev3, kTotalBlocks> blocks_{};

    float*          d_levels_         = nullptr;
    uint16_t*       d_x_rot_f16_      = nullptr;
    uint16_t*       d_x_scales_f16_   = nullptr;
    uint16_t*       d_x_norm_bf16_    = nullptr;
    uint8_t*        d_fp8_act_        = nullptr;
    uint8_t*        d_fp8_scale_      = nullptr;
    uint16_t*       d_sh_gateup_bf16_ = nullptr;
    float*          d_out_f32_        = nullptr;
    float*          d_inter_z_f32_    = nullptr;
    Mi50RoutedPair* d_pairs_          = nullptr;
    int32_t*        d_active_exp_     = nullptr;
    int32_t*        d_csr_offsets_    = nullptr;
    int32_t*        d_num_act_        = nullptr;
    uint32_t*       d_publish_sync_   = nullptr;

    Hlut3OffloadSlot* h_mtp_ring_        = nullptr;
    Hlut3OffloadSlot* h_bb_ring_         = nullptr;
    uint32_t          mtp_seq_issued_    = 0u;
    uint32_t          bb_seq_issued_     = 0u;

    uint16_t*       h_prefill_x_rot_f16_    = nullptr;
    int8_t*         h_prefill_x_i8_         = nullptr;
    uint16_t*       h_prefill_x_scales_f16_ = nullptr;
    const int8_t*   prefill_x_i8_src_       = nullptr;
    const uint16_t* prefill_x_scales_src_   = nullptr;
    uint16_t*       h_prefill_x_norm_bf16_  = nullptr;
    Mi50RoutedPair* h_prefill_pairs_        = nullptr;
    float*          h_prefill_y_out_f32_    = nullptr;
    uint16_t*       h_prefill_y_out_bf16_   = nullptr;
    int             prefill_block_id_       = 0;
    int             prefill_rows_           = 0;
    int             prefill_n_pairs_        = 0;
    bool            prefill_mtp_sh_         = false;
    bool            prefill_x_is_bf16_      = false;
    bool            prefill_x_is_int8_      = false;
    bool            prefill_out_bf16_       = false;
    std::atomic<uint32_t> prefill_req_seq_{0u};
    std::atomic<uint32_t> prefill_done_seq_{0u};

    std::atomic<bool> stop_worker_{false};
    std::thread worker_;
    std::atomic<uint64_t> stat_dispatches_{0ULL};
    std::atomic<uint64_t> stat_busy_ns_{0ULL};
};

}  // namespace dsv41
