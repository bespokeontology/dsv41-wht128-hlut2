// Wave 8.5 Track 8.5C: True 1,048,576-Token (1M) Context End-to-End Silicon Verification Gate
// Boots full production heterogeneous runtime simultaneously:
//   - max_seq_len = 1,048,576 (tiered HBM2 + coherent mapped-DDR4 CSA2 Global KV + HSI Indexer K)
//   - Dsv41VitEngine (253.45 MiB resident on tp=2)
//   - 2x Tenstorrent Blackhole P150A (18/18 L1 shards/card = 36 TP2 shards + Card #0 PCIe x16 -> FABRIC_1D relay)
//   - Wave 8.5 2,000+ TARQ RAREBAL-LDLQ + 445-swap frequency-balanced overlay (0 B extra runtime VRAM)
//   - 188.83 GiB pre-faulted DDR4 Engram + double-buffered M=512 / M=1024 prefill overlap
//   - GPU_MAX_HW_QUEUES=8 + Wave 8.5B cross-sublayer mHC & DSpark epilogue fusion
// Executes all 4 mandatory Track 8.5C gates via Dsv41Runtime::run_wave8_5_1m_e2e_gate and emits WAVE8_5_1M_E2E_RECEIPT.md!
// 100% Pure C++17 / gfx906 HIP — Zero Python Anywhere.

#include "../mi50/dsv41_runtime.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <unistd.h>

namespace {

#define CHECK_1M(cond, msg)                                                                   \
    do {                                                                                      \
        if (!(cond)) {                                                                        \
            std::fprintf(stderr, "[WAVE8_5_1M_FAIL] %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

void write_receipt_markdown(
    const std::string& out_md,
    const dsv41::Dsv411MGateReceipt& r,
    const std::string& overlay_path,
    double total_wall_s)
{
    std::ofstream f(out_md, std::ios::trunc);
    CHECK_1M(f.is_open(), ("Failed to open receipt output: " + out_md).c_str());

    f << "# Wave 8.5 — 1,048,576-Token (`1M`) End-to-End Silicon Verification Receipt\n\n";
    f << "- **Runtime Topology**: `3x AMD Instinct MI50 32GB (gfx906 TP3)` + `2x Tenstorrent Blackhole P150A (TP2, 18/18 L1 + x16->FABRIC_1D)` + `188.83 GiB Locked DDR4 Engram` + `Dsv41VitEngine (tp=2)`\n";
    f << "- **Active Overlay**: `" << overlay_path << "`\n";
    f << "- **Context Ceiling (`max_seq_len`)**: `1,048,576` tokens (`8,192` x `128`-token Radix/NVMe blocks)\n";
    f << "- **Total E2E Gate Wall Time**: `" << std::fixed << std::setprecision(2) << total_wall_s << " s`\n\n";

    f << "## 1. Simultaneous 1M VRAM Headroom & 64-Bit Pointer Audit\n\n";
    f << "| Rank | Device Role | Free HBM2 (`MiB`) | Total HBM2 (`MiB`) | `>= 200 MiB` Gate | `uint64_t` Pointer Audit |\n";
    f << "| :--- | :--- | ---: | ---: | :---: | :---: |\n";
    for (int tp = 0; tp < 3; ++tp) {
        f << "| `tp=" << tp << "` | " << (tp == 2 ? "`Backbone + MTP + Dsv41VitEngine`" : "`Backbone + MTP`")
          << " | `" << std::fixed << std::setprecision(2) << r.mem_margin.free_mib[tp]
          << "` | `" << r.mem_margin.total_mib[tp]
          << "` | `" << (r.mem_margin.free_mib[tp] >= 200.0 ? "PASS" : "FAIL")
          << "` | `" << (r.mem_margin.pointer_audit_64bit_ok ? "PASS" : "FAIL") << "` |\n";
    }
    f << "\n- **Tiered CSA2 Slot Partition (`tp=0..2`)**:\n";
    for (int s = 0; s < 4; ++s) {
        f << "  - Slot `" << s << "` (`r=" << (s < 3 ? 2 : 1) << "`): `max_comp=" << r.mem_margin.max_comp_slots[s]
          << "` (`HBM2 hot=" << r.mem_margin.hot_comp_slots[s]
          << "`, `Coherent DDR4 ext=" << r.mem_margin.ext_comp_slots[s] << "`)\n";
    }

    f << "\n## 2. Double-Buffered Prefill Engram Overlap & 1M HSI / CSA2 Numerical Stability\n\n";
    f << "- **`M = 512` Prefill Double-Buffered Engram Wait**: `" << std::setprecision(4) << r.engram_prefetch_wait_m512_ms
      << " ms/chunk` (`" << (r.engram_prefetch_overlap_m512_ok ? "PASS" : "FAIL") << "`)\n";
    f << "- **`M = 1024` Superchunk Prefill + P150A `M = 1024` `x16` Relay Wait**: `" << r.engram_prefetch_wait_m1024_ms
      << " ms/superchunk` (`" << (r.engram_prefetch_overlap_m1024_ok ? "PASS" : "FAIL") << "`)\n";
    f << "- **1M Chunked Prefill (`[1,046,528 .. 1,048,543]`, `2,016` tokens)**: `" << std::setprecision(2)
      << r.prefill_tail_ms << " ms`, `int32_overflow_free = " << (r.int32_overflow_free ? "true" : "false")
      << "`, `nonfinite_count = " << r.nonfinite_count << "`\n";
    f << "- **1M `HSI` `v_dot2_f32_f16` (`clamp=false`) vs `FP32` Top-64 Overlap (`8,192` blocks)**: `"
      << std::setprecision(3) << r.hsi_top64_overlap_pct << "%` (`>= 99.8%` gate)\n";
    f << "- **Planted Needle Retrieval Across 1M Span**: `pos=128 (blk 1): " << (r.needle_128_retrieved ? "HIT" : "MISS")
      << "`, `pos=524,288 (blk 4096): " << (r.needle_524288_retrieved ? "HIT" : "MISS")
      << "`, `pos=900,000 (blk 7031): " << (r.needle_900000_retrieved ? "HIT" : "MISS") << "`\n";
    f << "- **1M Tiered `CSA2` `v_dot2_f32_f16` vs `FP32` Output Parity**: `cosine = "
      << std::fixed << std::setprecision(8) << r.csa2_vdot2_vs_fp32_cos
      << "`, `max_abs_diff = " << std::scientific << std::setprecision(3) << r.csa2_vdot2_vs_fp32_max_abs << "`\n";

    f << "\n## 3. 32-Burst `Dynamic DeepSelect8` (`M = 8`) Decode Across `1,048,544 -> 1,048,575` (`1M - 1`)\n\n";
    f << "- **Completed Bursts**: `" << r.decode_bursts_completed << "` | **Final Verified Position**: `" << r.decode_final_pos
      << "` (`1,048,575 == 1M - 1`) | **Total Emitted**: `" << r.decode_total_emitted << "` tokens\n";
    f << "- **Mean Burst Wall Time**: `" << std::fixed << std::setprecision(2) << r.decode_mean_step_ms
      << " ms/step` | **Aggregate 1M Throughput**: `" << r.decode_agg_tok_s << " tok/s`\n\n";
    f << "| Burst | `start_pos` | `acc` | `ms/step` | `tok/s` | `attn` | `hsi` | `moe` | `p150w` | `ar` | `dsp` | `smp` | `egw` |\n";
    f << "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
    for (const auto& b : r.bursts) {
        f << "| " << b.burst_idx
          << " | `" << b.pos_start << "`"
          << " | " << b.accepted_tokens
          << " | " << std::fixed << std::setprecision(2) << b.step_ms
          << " | " << b.tok_s
          << " | " << b.attn_ms
          << " | " << b.hsi_ms
          << " | " << b.moe_ms
          << " | " << b.p150w_ms
          << " | " << b.ar_ms
          << " | " << b.dsp_ms
          << " | " << b.smp_ms
          << " | " << b.egw_ms << " |\n";
    }

    f << "\n## 4. 1M Boundary & Tail Pure-Block NVMe `.kvblk` Cold Hydration & `Encoder SWA` Replay\n\n";
    f << "- **Flushed Boundary + Tail Blocks**: `" << r.nvme_blocks_flushed
      << "` (`blocks {0, 1, 4095, 4096, 7031, 8184..8191}`)\n";
    f << "- **On-Disk `.kvblk` Footprint**: `" << r.nvme_block_bytes
      << " B/block` (`576 B` header + `113,920 B` payload = `890 B/tok`), **`"
      << r.nvme_swa_disk_bytes << " B` disk SWA**\n";
    f << "- **Cold NVMe Hydration + Hardware `CRC32C`**: `" << (r.nvme_crc32c_ok ? "PASS" : "FAIL")
      << "` (`" << std::fixed << std::setprecision(2) << r.nvme_cold_hydrate_ms
      << " ms` across `" << r.nvme_blocks_flushed << "` blocks)\n";
    f << "- **`128`-Token `Encoder SWA Bounded Replay` (`[1,048,448 .. 1,048,575]`)**: `swa_replay_tokens = "
      << r.nvme_swa_replay_tokens << "`, `enc_cos = " << std::fixed << std::setprecision(8) << r.nvme_enc_cos
      << "` (`max_abs = " << std::scientific << std::setprecision(3) << r.nvme_enc_max_abs << "`)\n";
    f << "- **`40L` Decoder Continuation Parity Post-Hydration**: `dec_cos = "
      << std::fixed << std::setprecision(8) << r.nvme_dec_cos
      << "` (`max_abs = " << std::scientific << std::setprecision(3) << r.nvme_dec_max_abs << "`)\n";
}

}  // namespace

int main(int argc, char** argv) {
    ::setenv("OMP_WAIT_POLICY", "PASSIVE", 1);
    ::setenv("GPU_MAX_HW_QUEUES", "8", 0);

    const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                     ? std::string(std::getenv("DSV41_ROOT"))
                                     : std::string(".");
    bool link_check = false;
    std::string overlay = root_env + "/packed_rarebal_overlay_wave8_5/hot_experts_wave8_5.safetensors";
    std::string nvme_tmp = root_env + "/tmp/verify_w85_1m_kv";
    std::string receipt_out = root_env + "/WAVE8_5_1M_E2E_RECEIPT.md";

    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--link-check") link_check = true;
        else if (a == "--overlay" && i + 1 < argc) overlay = argv[++i];
        else if (a == "--nvme-tmp" && i + 1 < argc) nvme_tmp = argv[++i];
        else if (a == "--receipt-out" && i + 1 < argc) receipt_out = argv[++i];
    }

    if (link_check) {
        std::printf("[WAVE8_5_1M_E2E] Link check OK.\n");
        return 0;
    }

    std::error_code ec;
    if (!std::filesystem::is_regular_file(overlay, ec)) {
        const std::string fallback = root_env + "/packed_rarebal_overlay_wave8_4/hot_experts_wave8_4_perm.safetensors";
        CHECK_1M(std::filesystem::is_regular_file(fallback, ec), "Neither Wave 8.5 nor Wave 8.4 overlay found!");
        overlay = fallback;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::printf("[WAVE8_5_1M_E2E] Booting full 1,048,576-token Dsv41Runtime + ViT + 18/18 P150A L1 + 189 GiB Engram...\n");
    std::fflush(stdout);

    dsv41::Dsv41Runtime rt;
    rt.set_expert_perm_overlay(overlay);
    rt.set_csa2_shared_tree8(true);
    rt.set_fused_mhc_norm(true);
    rt.set_fused_dspark_epilogue(true);
    rt.init(root_env + "/packed_mi50",
            root_env + "/hf_meta/tokenizer.json",
            "/dsv41_p150_ipc",
            1048576,
            overlay,
            "",
            root_env + "/packed_vit/dsv41_vit_bf16.safetensors");
    rt.set_dual_stream_overlap(true);
    rt.set_dspark_tree_width(8);
    rt.set_dspark_static_m8(false);
    CHECK_1M(rt.ready() && rt.vision_ready(), "Dsv41Runtime + Dsv41VitEngine failed to initialize at max_seq_len=1,048,576");

    const dsv41::Dsv411MGateReceipt rec = rt.run_wave8_5_1m_e2e_gate(nvme_tmp);
    std::filesystem::remove_all(nvme_tmp, ec);

    const double tot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    write_receipt_markdown(receipt_out, rec, overlay, tot_s);

    std::printf("[WAVE8_5_1M_E2E] Summary (%.2fs):\n"
                "  Free HBM2 @ 1M + ViT : %.2f / %.2f / %.2f MiB (64-bit ptr audit=%d)\n"
                "  Engram overlap       : M=512 %.4f ms (%d) | M=1024 %.4f ms (%d)\n"
                "  1M HSI & CSA2        : top64_overlap=%.3f%% | needles=%d/%d/%d | CSA2 cos=%.8f (max_abs=%.3e) | nonfinite=%zu\n"
                "  1M Decode (32x M=8)  : final_pos=%d | %.2f ms/step | %.2f tok/s (%d bursts)\n"
                "  1M NVMe .kvblk       : %zu B/blk (%zu B disk SWA) | CRC32C=%d | enc_cos=%.8f | dec_cos=%.8f\n"
                "  Receipt written      : %s\n",
                tot_s,
                rec.mem_margin.free_mib[0], rec.mem_margin.free_mib[1], rec.mem_margin.free_mib[2],
                int(rec.mem_margin.pointer_audit_64bit_ok),
                rec.engram_prefetch_wait_m512_ms, int(rec.engram_prefetch_overlap_m512_ok),
                rec.engram_prefetch_wait_m1024_ms, int(rec.engram_prefetch_overlap_m1024_ok),
                rec.hsi_top64_overlap_pct,
                int(rec.needle_128_retrieved), int(rec.needle_524288_retrieved), int(rec.needle_900000_retrieved),
                rec.csa2_vdot2_vs_fp32_cos, rec.csa2_vdot2_vs_fp32_max_abs, rec.nonfinite_count,
                rec.decode_final_pos, rec.decode_mean_step_ms, rec.decode_agg_tok_s, rec.decode_bursts_completed,
                rec.nvme_block_bytes, rec.nvme_swa_disk_bytes, int(rec.nvme_crc32c_ok),
                rec.nvme_enc_cos, rec.nvme_dec_cos,
                receipt_out.c_str());
    std::fflush(stdout);

    // Enforce every mandatory Track 8.5C gate!
    CHECK_1M(rec.mem_margin.free_mib[0] >= 200.0 &&
             rec.mem_margin.free_mib[1] >= 200.0 &&
             rec.mem_margin.free_mib[2] >= 200.0,
             "Failed >= 200.0 MiB free HBM2 per MI50 rank at max_seq_len=1,048,576 + ViT!");
    CHECK_1M(rec.mem_margin.pointer_audit_64bit_ok, "64-bit pointer offset audit failed!");
    CHECK_1M(rec.engram_prefetch_overlap_m512_ok && rec.engram_prefetch_overlap_m1024_ok,
             "Double-buffered M=512 / M=1024 prefill Engram + P150A x16 overlap gate failed!");
    CHECK_1M(rec.int32_overflow_free && rec.nonfinite_count == 0u,
             "1M prefill/decode produced non-finite activations/logits or int32 index overflow!");
    CHECK_1M(rec.hsi_top64_overlap_pct >= 99.8,
             "1M HSI v_dot2_f32_f16 top-64 block overlap fell below 99.8% vs FP32!");
    CHECK_1M(rec.needle_128_retrieved && rec.needle_524288_retrieved && rec.needle_900000_retrieved,
             "Planted needle retrieval failed across [128, 524288, 900000] at 1M context!");
    CHECK_1M(rec.csa2_vdot2_vs_fp32_cos >= 0.99999,
             "1M tiered CSA2 output cosine fell below 0.99999 vs FP32!");
    CHECK_1M(rec.decode_bursts_completed == 32 && rec.decode_final_pos == 1048575,
             "Failed to complete 32 Dynamic DeepSelect8 bursts reaching exact 1,048,575 (1M - 1) ceiling!");
    CHECK_1M(rec.nvme_block_bytes == 114496u && rec.nvme_swa_disk_bytes == 0u && rec.nvme_crc32c_ok,
             "1M NVMe .kvblk byte layout / 0 B SWA / CRC32C check failed!");
    CHECK_1M(rec.nvme_swa_replay_tokens == 128 && rec.nvme_enc_cos >= 0.99999999 && rec.nvme_dec_cos >= 0.99999999,
             "1M NVMe cold hydration + 128-token Encoder SWA replay failed bit-exact 1.00000000 parity!");

    std::printf("[WAVE8_5_1M_E2E] ALL 1,048,576-TOKEN (1M) SILICON GATES PASSED.\n");
    return 0;
}
