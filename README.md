# `2× Tenstorrent Blackhole P150A` + `WHT128-HLUT2` (`2.125 bpw`): Native C++ Inference Engine for 748.6B `DeepSeek-V4.1-Flash-1017` (Wave 6C Canonical)

**Co-Engineered by [**`bespokeontology`**](https://github.com/bespokeontology) & [**Gemini (Google DeepMind)**](https://deepmind.google/technologies/gemini/)** · **100% Pure TT-Metal C++20 / `gfx906` HIP / Host C++17 (`0` Python Files)**  
📄 **[6-Page Illustrated Whitepaper (PDF)](docs/WHT128_HLUT2_Blackhole_P150A_Whitepaper.pdf)** · 📘 **[Full Engineering Report (`docs/WHT128_HLUT2_ENGINEERING_REPORT.md`)](docs/WHT128_HLUT2_ENGINEERING_REPORT.md)**

---

## Overview

This repository contains the clean-room, zero-Python production inference engine for **[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** (**`748.6B` total parameters**: `559.7B` `40`-layer [`CSA2`](https://zartbot.github.io/blog/model_arch/dsv41flash_arch/en.html) MoE + `188.83 GiB` external [`Engram`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/engram.py) `N`-gram memory + `3`-stage [`DSpark`](https://rocm.docs.amd.com/projects/atom/en/latest/deepseek_v41_dspark.html) speculative draft) co-designed across heterogeneous non-CUDA silicon:

1. **[`2× Tenstorrent Blackhole P150A`](https://docs.tenstorrent.com/aibs/blackhole/index.html) (Lead Sparse MoE Tier):** `220` compute Tensix cores (`2 × 110`), `64 GiB` GDDR6 (`MP`), and **`2× 180 MB = 360 MB` on-die Tensix L1 SRAM** (`234.48 MiB` active `MR`, `98.61 MiB` pinned hot expert shards), cross-wired card-to-card via **`4× 800G` QSFP-DD cables** (`3.2 Tbps` `FABRIC_1D` TP2), with **one `P150A` running on a narrow `PCIe x4` host riser**.
2. **`3× AMD Instinct MI50` (Dense Backbone, `CSA2`, & `DSpark` Tier):** `96 GiB` HBM2 (`Vega20 gfx906` TP3, `80/80` HIP kernels `ScratchSize == 0`, `0` PCIe BAR P2P reads via coherent pinned-`DDR4` `FP32` `HostRingAllReduce`).
3. **`AMD EPYC` + Dual NVMe (`Engram`, Prefix Cache, & `AVX2` Tail Tier):** `384 GiB` DDR4 hosting `188.83 GiB` pre-faulted native `FP8 E4M3` `Engram` tables (`248.98 GB/s` warm hydration in `0.81 s`), 4-tier stateless `CSA2` prefix cache (`524,288` `512K` YaRN context), and `29`-thread `AVX2` `CpuTailPool`.

### Headline Measured Performance (Live `512K`-Context Agentic Engineering Sessions)

* **`47.5 – 52.2 tok/s`** sustained `DSpark` (`M = 6`) decode across `58.4K – 91.2K` context (`96.7%` token acceptance, `tau = 5.66`)
* **`38.9 – 43.6 tok/s`** [`DSML`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/encoding/README.md) C++ tool-call generation (`tau = 4.40 – 5.08`, `80.4% – 86.1%` acceptance)
* **`37.10 tok/s`** sustained `10,024`-token `<think>` mathematical/C++ derivations (**`+37.6%` speedup** from Wave 6B's `56`-expert [`TARQ` `RAREBAL-LDLQ`](https://arxiv.org/pdf/2605.27808) overlay at `0 B` extra `MI50` HBM2)
* **`53.64 – 62.31 ms/step`** across all `40` `2× Blackhole P150A` MoE layers (`0.78 ms` traced `M = 1`, `parity_cos = 1.0000000` — making the `PCIe x4` riser invisible)
* **`2.44B tok/s`** Tier-0 in-VRAM `LCP` reuse · **`2.00M – 3.06M tok/s`** cross-session `144.4K` switch (`72.26 ms`) · **`107,556 – 184,914 tok/s`** warm `RadixBlockCache` hydration
* **`0` false positives & `0` inline-`</think>` spills** across `26,072+` formal C++ `<think>` audit tokens under Wave 6C `Dsv41ThinkLoopGuard` + stateful `CSA2`/`SWA`/`MTP` rewind

---

## How `WHT128-HLUT2` (`2.125 bpw`) Preserves Full-Precision `748.6B` Fidelity

### 1. Subsystem Full-Precision Preservation (`Only 1.56% Active Sparse Capacity Compressed`)
Uniform 4-bit quantization of `559.7B` transformer parameters (`~280 GiB`) cannot fit in `160 GiB` total accelerator DRAM (`64 GiB` `P150A` GDDR6 + `96 GiB` `MI50` HBM2), while naive 2-bit quantization across attention or shared layers collapses reasoning. `WHT128-HLUT2` (`2.125 bpw`) compresses **exclusively** the `384` sparse conditional routed SwiGLU experts (`6 / 384 = 1.56%` active per token), preserving **full or lossless precision on every always-active and state-bearing subsystem**:

* **`BF16` / `FP32` (`16–32 bpw`)** on embeddings, `RMSNorm`, [`mHC` (`n = 4` Sinkhorn)](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json), `384`-way routers, `LM Head`, and Wave 6 `FP32` residual/AllReduce streams;
* **`8.01 bpw` (`cos > 0.9999`)** on all `40` [`CSA2` attention layers](https://zartbot.github.io/blog/model_arch/dsv41flash_arch/en.html), `HSI` indexers, `40` always-active shared SwiGLU experts, and `3` [`DSpark`](https://rocm.docs.amd.com/projects/atom/en/latest/deepseek_v41_dspark.html) dense heads;
* **Bitwise-exact native `8.25 bpw` (`FP8 E4M3 + FP16` scale, zero re-quantization)** on the `188.83 GiB` external [`Engram`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/engram.py) `N`-gram memory in `DDR4`.

### 2. Full-Precision `FP32` Compute & On-Die SRAM Dequantization Across Heterogeneous Silicon
Stock CUDA 2-bit schemes fail across heterogeneous tile/NoC (`Blackhole`) vs SIMT (`gfx906`) silicon: [`QTIP`](https://arxiv.org/html/2406.11235v1)/[`EXL3`](https://github.com/turboderp-org/exllamav3) sequential bitshift trellises stall `32`-wide Tensix [`SFPU` lanes](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/VectorUnit.md); [`QuIP#` `E8P`](https://arxiv.org/abs/2402.04396) lattice codebooks require per-lane scatter gathers absent in Tensix tile-row `SFPLOAD` and trigger `gfx906` LDS bank conflicts; and unrotated `Bfp2_b` (`2.50 bpw`) exceeds `160 GiB` DRAM by `24.6 GiB` while hitting Blackhole LLK tile-face bugs ([`#58712`](https://github.com/tenstorrent/tt-metal/issues/58712), [`#56682`](https://github.com/tenstorrent/tt-metal/issues/56682/linked_closing_reference?reference_location=REPO_ISSUES_INDEX), [`#57683`](https://github.com/tenstorrent/tt-metal/issues/57683)).

`WHT128-HLUT2` uses **one unified mathematical representation** (`WHT128` Rademacher incoherence + symmetric 2-bit [PolarQuant Lloyd-Max](https://arxiv.org/html/2603.29078v1) + [`TARQ` `RAREBAL-LDLQ`](https://arxiv.org/pdf/2605.27808)) mapped natively to each chip's SRAM and `FP32` accumulators:
* **On `2× Blackhole P150A` (Tensix On-Die L1 SRAM `MR` + `FP32` `Dst`):** `16` logical `32×32` 2-bit subtiles pack into one zero-padding **`4,352 B` `UInt32` 4-face NoC superblock** (`272 B/tile`). Each `P150A` pins **`11/11` hot expert shards (`22` TP2 shards = `98.61 MiB`, `234.48 MiB` total L1 `MR` across both cards alongside `512`-row `x_rm_l1` and `FP32` `y_accum_f32`) directly inside `20+ TB/s` on-die Tensix L1 SRAM** via `O(1)` `l1_slot_lut`. Tensix `SFPU` (`MOD0_FMT_INT32`) expands 2-bit codes via a branchless 1-bit affine FMA into `2 * sb_cnt`-deep L1 circular buffers, feeding `HiFi2` `MVMUL` into **full `FP32` `8`-tile [`Dst` accumulators](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/Dst.md) (`dst_full_sync_en = true`, `UnpackToDestFp32`)** and L1 `FP32` `y_accum_f32`.
* **On `3× AMD Instinct MI50` (`gfx906` `64 KiB` XOR-Swizzled LDS + `FP32` Dot/Residual):** Identical `2.125 bpw` tiles unpack in registers in `4` ALU instructions (`0` LUT gathers, `80/80` kernels `ScratchSize == 0`) into `v_dot2_f32_f16` with `FP32` group scales and Wave 6 **`FP32` `HostRingAllReduce + mHC post`** before a single IEEE RNE round to `BF16`.

---

## `4× 800G` Inter-Card Fabric & Overcoming the `PCIe x4` Riser on One `Blackhole P150A`

The two `Blackhole P150A` cards are cross-wired card-to-card via **four `800G` QSFP-DD cables** (`3.2 Tbps` aggregate `FABRIC_1D` TP2 mesh across Cages P1–P4), while **one `P150A` connects to the EPYC motherboard through a narrow `PCIe x4` riser**. Three architectural invariants prevent the `x4` riser from ever bottlenecking inference:

1. **Zero Runtime Weight Streaming Over PCIe:** All `168` routed experts/layer (`[216..383]`) across `40` layers + `24` `RAREBAL-LDLQ` overlays stay permanently resident in `P150A` `GDDR6` `MP` and `98.61 MiB` on-die L1 `MR`.
2. **`4× 800G` Direct Fabric Absorbs TP2 Traffic:** Inter-card TP2 synchronization and mesh collectives traverse the `4× 800G` QSFP-DD links directly rather than hair-pinning through the `PCIe x4` riser and EPYC root complex.
3. **`1.5 GiB` `MeshTrace` + `60 KiB` Compact Readback + `CpuTailPool`:** On-device `MeshTrace` eliminates host kernel-dispatch chatter over `x4`, decode ships only compact `M = 6` activations with a single-wave **`60 KiB` readback**, and `CpuTailPool` peels singleton tail experts onto `EPYC` `AVX2` threads in parallel.

---

## Wave 6, Wave 6B (`TARQ RAREBAL-LDLQ`), & Wave 6C (`ThinkLoopGuard` + Stateful Rewind)

* **Wave 6 (Kolmogorov / Dumitrescu Upgrades):** `FP32` `HostRingAllReduce + mHC post` with single RNE round; zero-trig `FP64` Givens `512K` YaRN RoPE (`0` device trig instructions); GPU-resident `DSpark` Borel full-partition renormalization (`5.65 – 6.30 ms/step`); `P150A` `512`-row `MR`/`MP` + `O(1)` `l1_slot_lut`; `AVX2` Barrett `21`-bit `Engram` (`248.98 GB/s`); dirty-only `RadixBlockCache` (`144.4K` switch in `72.26 ms`).
* **Wave 6B (`56`-Expert `TARQ RAREBAL-LDLQ` Overlay):** `56/56 EMITTED` hot experts (`24` `P150A` incl. `8/11` L1-pinned + `20` `MI50` + `12` `DSpark` `mtp.0..2`, with `3` lower-frequency `P150A` L1 pairs automatically held at `H = I` by the `2K` starvation gate) achieving `+2.90` to `+6.14 dB` SNR gain (`+37.6%` sustained `<think>` decode speedup to `37.10 tok/s`). Distributed as a GitHub Release asset (`hot_experts_rarebal.safetensors`, `503 MB`).
* **Wave 6C (`Dsv41ThinkLoopGuard` + Even-Aligned `SWA`/`r = 2`/`MTP` Rewind + Inline-`</think>` Guard):** Passive `O(1)` `<think>` loop observer (`mi50/dsv41_think_loop_guard.hpp`, `45.94 ns/tok`, `0` logit mutation), hardware-exact rewind snapping `replay_start = max(0, (target_pos - 128) & ~1)` to reset `CSA2` `r = 2` state, rebuild `128`-token `SWA` rings, re-seed `128`-row `DSpark` `MTP` rings, and trim `RadixBlockCache` (`mi50/dsv41_runtime.cpp:3420–3565`), plus context-aware `classify_think_end()` (`mi50/dsv41_tokenizer_dsml.cpp`) preventing quoted `"\n</think>"` mentions from spilling reasoning into content.

---

## Repository Structure (`p150/` Lead Hierarchy)

```text
├── p150/                  # 2x Tenstorrent Blackhole P150A TT-Metal C++20 FABRIC_1D TP2 engine & 13 Tensix kernels
│   ├── kernels/           # dsv41_hlut2_writer_b.cpp (O(1) l1_slot_lut + 2*sb_cnt CBs), dsv41_hlut2_compute.cpp, ...
│   ├── p150_dsv41_engine.cpp
│   └── p150_dsv41_worker.cpp
├── src/                   # WHT128-HLUT2 (2.125 bpw) + Reverse-Cholesky LDLQ + TARQ RAREBAL & safetensors mmap
│   ├── hlut2_pack.hpp
│   ├── dsv41_pack_format.hpp
│   ├── dsv41_rarebal.hpp
│   └── safetensors_mmap.hpp
├── mi50/                  # 3x AMD Instinct MI50 (gfx906 TP3) zero-spill HIP kernels, runtime, Engram, & :8042 server
│   ├── dsv41_mi50_kernels.hip
│   ├── dsv41_runtime.cpp
│   ├── dsv41_engram_cpu.cpp
│   ├── dsv41_think_loop_guard.hpp
│   ├── dsv41_tokenizer_dsml.cpp
│   └── dsv41_fe.cpp
├── tools/                 # Pure-C++17 offline shard packer, RAREBAL repacker, ablation bench, & Wave 5/6 self-tests
│   ├── pack_dsv41_hlut2.cpp
│   ├── repack_hot_experts_rarebal.cpp
│   ├── repack_rarebal_helpers.hpp
│   ├── bench_hlut2.cpp
│   ├── verify_wave5_all.cpp
│   └── verify_wave6_math.cpp
├── docs/
│   ├── WHT128_HLUT2_Blackhole_P150A_Whitepaper.pdf
│   └── WHT128_HLUT2_ENGINEERING_REPORT.md
└── manifests/
    ├── rarebal_receipt_manifest.txt
    ├── CANONICAL_RELEASE.json
    └── SHA256SUMS.txt
```

---

## Build, Self-Test, & Launch

```bash
# 1. Build offline tools, self-tests, 3x MI50 runtime/server, and (when TT_METAL_HOME is set) 2x P150A worker
export DSV41_ROOT=/path/to/dsv41-model-workspace
export TT_METAL_HOME=/opt/tt-metal   # Tenstorrent Blackhole TT-Metal SDK install
make -j12

# 2. Run Wave 5, Wave 6, and Wave 6C mathematical + parser + gfx906 zero-scratch self-tests
make check
make audit-gfx906

# 3. Download `hot_experts_rarebal.safetensors` from the v1.0.0-wave6c GitHub Release and verify SHA-256
sha256sum -c SHA256SUMS.txt

# 4. Launch 2x Blackhole P150A TP2 worker first, then 3x MI50 + 189 GiB Engram HTTP/SSE server
./bin/p150_dsv41_worker --model-dir "$DSV41_ROOT" --rarebal-overlay hot_experts_rarebal.safetensors --reuse-shm &
OMP_WAIT_POLICY=PASSIVE ./bin/dsv41_fe --model-dir "$DSV41_ROOT" --rarebal-overlay hot_experts_rarebal.safetensors --port 8042 --max-seq-len 524288
```

## Credits & Attribution

* **Hardware Architecture, Systems Direction, & Heterogeneous Co-Design:** [**`bespokeontology`**](https://github.com/bespokeontology)
* **AI Pair-Programming, Kernel Co-Engineering, & Mathematical Verification:** [**Gemini (Google DeepMind)**](https://deepmind.google/technologies/gemini/) — TT-Metal C++20 Tensix dataflow/SFPU/HiFi2 kernels, zero-spill `gfx906` HIP kernels, `WHT128-HLUT2` + `TARQ RAREBAL-LDLQ` C++17 packers, Kolmogorov–Dumitrescu Wave 6 math, and Wave 6C `ThinkLoopGuard` + stateful `CSA2`/`SWA`/`MTP` rewind.

## License

Released under the [MIT License](LICENSE) (`Copyright (c) 2026 bespokeontology`).
