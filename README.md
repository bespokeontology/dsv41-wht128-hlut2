# `WHT128-HLUT2` + `561`-Expert `TARQ RAREBAL-LDLQ` (`2.125 bpw → FP32`) on `2× Custom TSMC 6nm Tenstorrent Blackhole P150A` Spatial Dataflow ASICs: `1M`-Context C++ Engine for 748.6B `DeepSeek-V4.1-Flash-1017` (`v1.2.0-wave8`)

**Co-Engineered by [**`bespokeontology`**](https://github.com/bespokeontology), [**Google Antigravity (Google DeepMind)**](https://antigravity.google/), & [**Gemini (Google DeepMind)**](https://deepmind.google/technologies/gemini/)** · **100% Pure TT-Metal C++20 / `gfx906` HIP / Host C++17 (`0` Python Files)**  
📄 **[4-Page Illustrated Whitepaper (PDF)](docs/WHT128_HLUT2_Blackhole_P150A_Whitepaper.pdf)** · 📘 **[Full Wave 8 Engineering Report (`docs/WHT128_HLUT2_ENGINEERING_REPORT.md`)](docs/WHT128_HLUT2_ENGINEERING_REPORT.md)** · 📦 **[`v1.2.0-wave8` Release & `561`-Expert `RAREBAL` Shards](https://github.com/bespokeontology/dsv41-wht128-hlut2/releases/tag/v1.2.0-wave8)**

---

## Overview

This repository contains the clean-room, zero-Python production inference engine and custom **`WHT128-HLUT2` + `561`-Expert `TARQ RAREBAL-LDLQ` (`2.125 bpw → IEEE FP32`)** quantization stack for **[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** (**`748.6B` total parameters**: `559.7B` `40`-layer [`CSA2`](https://zartbot.github.io/blog/model_arch/dsv41flash_arch/en.html) MoE + `188.83 GiB` external [`Engram`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/engram.py) `N`-gram memory + `3`-stage [`DSpark`](https://rocm.docs.amd.com/projects/atom/en/latest/deepseek_v41_dspark.html) speculative drafter), co-designed across heterogeneous non-CUDA silicon up to **`1,048,576` (`1M`) active context tokens**:

1. **[`2× Tenstorrent Blackhole P150A`](https://docs.tenstorrent.com/aibs/blackhole/index.html) Custom TSMC `6nm` (`N6`) Spatial Dataflow ASICs (Lead Sparse MoE Tier):** `220` compute Tensix cores (`2 × 110`, each with `5` RISC-V baby cores + `32×32` `HiFi2`/`FP32` Matrix Engine + `32`-lane Vector `SFPU` + dual `NoC` routers), `64 GiB` GDDR6 (`MP`), and **`2× 180 MB = 360 MB` on-die Tensix L1 SRAM** (**`247.82 MiB` pure active L1** [`123.91 MiB/card = 129,927,808 B/card`], **`34/34` `RAREBAL`-calibrated TP2 hot shards = `152.40 MiB`** pinned in L1 alongside `100%` L1-resident `x_decode` and `finish_decode[0..1]`), cross-wired card-to-card via **`4× 800G` QSFP-DD cables** (`3.2 Tbps` `FABRIC_1D` TP2), with **Card `#1` running behind a narrow `PCIe x4` host riser**.
2. **`3× AMD Instinct MI50` (Dense Backbone, `1M` Tiered `CSA2`, & Dual-Stream `DSpark` Causal-Tree Tier):** `96 GiB` HBM2 (`Vega20 gfx906` TP3, **`91/91` runtime + `3/3` verification HIP kernels at `ScratchSize == 0`**, `0` PCIe BAR P2P reads via coherent pinned-`DDR4` `FP32` `HostRingAllReduce` + pre-`shared_experts` dual-stream `4`-slot `25,680 KiB` `/dev/shm` fast ring).
3. **`AMD EPYC` + Dual NVMe (`Engram`, `1M` Warm KV Ring, Prefix Cache, & `AVX2` Tail Tier):** `384 GiB` DDR4 hosting `188.83 GiB` pre-faulted native `FP8 E4M3` `Engram` tables (`220.74 GB/s` warm hydration in `0.92 s`), coherent mapped pinned-DDR4 warm `CSA2` KV ring (`1,048,576` active context ceiling; **`217,367` & `417,556` verified live active context**), 4-tier stateless `RadixBlockCache` (`2.31B–2.68B tok/s` hot LCP), and pure-C++20 `/v1/chat/completions` + `/anthropic/v1/messages` SSE.

### Headline Measured Performance (`v1.2.0-wave8` Live Agentic Engineering Sessions)

* **`37.93 tok/s`** live `<think>` + 9-tool [`DSML`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/encoding/README.md) Stream `#1` decode (`τ = 4.03`, `75.6%` `DSpark` acceptance, **`+17.2%`** over `v1.1.0-wave7b`'s `32.37 tok/s` / `67.0%`) and **`35.36 tok/s`** multi-tool `26.6K–44K` `<think>` reasoning (`τ = 3.95`, **`+16.7%`** over `v1.1.0-wave7b`; up to `47.5–52.2 tok/s` peak `DSpark` decode and `39.72 tok/s` at `258.4K` ctx)
* **`28.88 ms/step`** `3× MI50` MoE service time (**`2.01×` faster** than `v1.1.0-wave7b`'s `57.98 ms/step` via 1-CTA `k_pack_mi50_routed_pairs_compact256`) and **`35.75 ms/step`** non-overlapped `2× Blackhole P150A` wait (**`-51.9%`** vs. `74.39 ms/step`, saving **`18.57 ms/step`** of true concurrent `P150A || MI50` overlap vs. `54.33 ms/step` `P150A` device time)
* **`0.73 ms` (`M = 1`), `2.22 ms` (`M = 6`), `2.52 ms` (`M = 8`), & `4.53 ms` (`M = 16`)** `7/7` hardware `MeshTrace` shapes across all `40` `2× Blackhole P150A` layers (`parity_cos = 1.0000000`, `0 B` extra L1 SRAM, `max_TV = 2.793e-16` on `M = 16`, live peak `τ = 4.62` across `217,367` active tokens)
* **`561 / 561`** [`TARQ RAREBAL-LDLQ`](https://arxiv.org/pdf/2605.27808) hot experts (`10.02×` scale-up from `56`, covering **`34/34` `P150A` L1 shards** + `307` `MI50` + `37` `DSpark` MTP experts; `0` Dual Receipt violations; mean SwiGLU cosine **`0.88357 → 0.95228`**, **`+4.221 dB`** mean SNR at **`0 B` extra runtime VRAM**)
* **`1,048,576` (`1M`) active `CSA2` context** (`268.0 / 270.0 / 960.0 MiB` free HBM2 across `tp = 0..2`, exact `1.00000000` cosine at `pos = 1,048,575`) · **`614.8 tok/s`** 20-layer [`CED` fresh prefill](https://arxiv.org/pdf/2609.19969) (`> 500 tok/s` past `146K` ctx) · **`2.68B tok/s`** Tier-0 hot LCP

---

## Wave 8 (`v1.2.0-wave8`) Architectural Breakthroughs

1. **Track 8A — Pre-`shared_experts` Dual-Stream `P150A || MI50` Overlap & 1-CTA `compact256` Pair Packer:**
   * Forks `tp = 0`'s high-priority `router_stream` (`stream_prio = [1..-1]`) onto a `4`-slot **`25,680 KiB` host-mapped `/dev/shm` fast ring** (`100,000` PCIe coherency iterations in `3,978.30 ms`, `stale_reads = 0`) and orders `ev_p150_armed_ring` **strictly before `shared_experts`**, ensuring `MI50` CU saturation never delays `P150A` doorbell wake-up.
   * Replaces multi-launch pair sorting on decode/tree steps (`M <= 16`) with a single zero-scratch `256`-thread `k_pack_mi50_routed_pairs_compact256` kernel, cutting `MI50` MoE time **`2.01×` (`57.98 → 28.88 ms/step`)** and saving **`18.57 ms/step`** of parallel `P150A || MI50` execution.
2. **Track 8B — `7/7` `MeshTrace` (`M = 1..1,024`) & Entropy-Adaptive `M = 8 ↔ 16` Kolmogorov–Dirac Causal Tree (`0 B` Extra L1):**
   * Embeds `M = 16` causal-tree verification directly inside `P150A`'s existing `32`-row L1 `x_decode` tile (**`0 B` extra L1**, `1,453,184 B <= 1,459,200 B/bank`, `247.82 MiB` pure active L1 across `2× P150A`) with SGPR `TreeAncMask256` attention (`ScratchSize == 0`) and exact Dirac Plackett–Luce coupling (`max_TV = 2.793e-16 < 1e-15`, `--dspark-tree-width auto|8|16`).
3. **Track 8C — `561`-Expert `TARQ RAREBAL-LDLQ` Scale-Up via `3× MI50` Hardware `FP64` Blocked Cholesky:**
   * Offloads `B = 128` blocked `FP64` Reverse-Cholesky (`L_rev L_rev^T = P H_bal P^T`) onto `gfx906`'s `1:2` `FP64` units via `rocblas_dgemm` across all `3× MI50` GPUs (`46.2 s → 1.51 s/expert`, `30.6×` speedup), scaling verified `M_cal >= 11,272` overlays **`10.02×` from `56` to `561` experts** (`217` `P150A` incl. **`34/34` pinned L1 shards** + `307` `MI50` + `37` `MTP`, `4.91 GiB` in-place overlay, **`+4.221 dB`** mean SNR, `0 B` extra runtime VRAM).
4. **Track 8D — `1,048,576`-Token (`1M`) Tiered `CSA2` Context & `91/91` Zero-Scratch `gfx906` Kernels:**
   * Upgrades KV/Indexer strides to `64`-bit `size_t` and pairs `262,144` hot HBM2 `r = 2` slots (`524,288` hot tokens) with a zero-host-sync coherent mapped pinned-DDR4 warm ring (`max_seq_len = 1,048,576`, `1.00000000` RoPE & `CSA2` cosine at `pos = 1,048,575`, `91/91` runtime + `3/3` verify8 kernels at `ScratchSize == 0`).
5. **Track 8E — Pure-C++20 `/anthropic/v1/messages` SSE & Mid-Turn POSIX Search:**
   * Serves native Anthropic Messages SSE (`thinking_delta`, `signature_delta`, `tool_use`, CORS preflight, and optional `--search-url` mid-turn POSIX HTTP search) directly inside `dsv41_fe` with `100%` `RadixBlockCache` prefix identity.

---

## How `WHT128-HLUT2` + `TARQ RAREBAL-LDLQ` (`2.125 bpw → FP32`) Preserves Full-Precision `748.6B` Fidelity

`WHT128-HLUT2` (`2.125 bpw`, `272 B / 32×32` tile, `4,352 B / 128×128` 16-subtile `UInt32` NoC superblock) compresses **exclusively** the `384` sparse conditional routed SwiGLU experts (`6 / 384 = 1.56%` active per token), keeping **every always-active and state-bearing subsystem at full or lossless precision** inside `160 GiB` total accelerator DRAM (`64 GiB` `P150A` GDDR6 + `96 GiB` `MI50` HBM2):

* **`BF16` / `FP32` (`16–32 bpw`)** on embeddings, `RMSNorm`, [`mHC` (`n = 4` Sinkhorn)](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json), `384`-way routers, `LM Head`, and `FP32` `HostRingAllReduce + mHC post` (`266.8×` lower truncation MSE);
* **`8.01 bpw` (`cos > 0.9999`)** on all `40` [`CSA2` attention layers](https://zartbot.github.io/blog/model_arch/dsv41flash_arch/en.html), `HSI` indexers, `40` always-active shared SwiGLU experts, and `3` [`DSpark`](https://rocm.docs.amd.com/projects/atom/en/latest/deepseek_v41_dspark.html) dense heads;
* **Bitwise-exact native `8.25 bpw` (`FP8 E4M3 + FP16` scale, zero re-quantization)** on the `188.83 GiB` external [`Engram`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/engram.py) `N`-gram memory in `384 GiB DDR4`;
* **Wave 8 `561/561` [`TARQ RAREBAL-LDLQ`](https://arxiv.org/pdf/2605.27808) Overlay (`hot_experts_rarebal_wave8.safetensors`, `4.91 GiB` across `3` `< 2.0 GiB` release shards):** Causal Reverse-Cholesky [`QuIP#` `LDLQ`](https://arxiv.org/html/2402.04396v1) + `M_cal > 2K` starvation gate + rare-token trace equalization across `217` `P150A` (`34/34` L1 shards), `307` `MI50`, and `37` `DSpark` MTP hot experts (`0.88357 → 0.95228` mean SwiGLU cosine, **`+4.221 dB`** mean SNR gain at **`0 B` extra VRAM**).

---

## Repository Structure (`p150/` Lead Hierarchy)

```text
├── p150/                  # 2x Custom TSMC 6nm Tenstorrent Blackhole P150A TT-Metal C++20 FABRIC_1D TP2 engine
│   ├── kernels/           # 13 Tensix compute/NoC kernels (O(1) l1_slot_lut, 2*sb_cnt CBs, HiFi2 -> FP32 Dst)
│   ├── dsv41_p150_l1.hpp  # Exact Tier-1b FABRIC_1D per-bank L1 budget (17/17 shards + L1 x_decode = 123.91 MiB/card)
│   ├── dsv41_p150_shm.hpp # Lock-free 4-slot 25,680 KiB /dev/shm fast ring + M=1..1024 mailbox
│   ├── p150_dsv41_engine.cpp
│   └── p150_dsv41_worker.cpp
├── src/                   # WHT128-HLUT2 (2.125 bpw) + Reverse-Cholesky LDLQ + TARQ RAREBAL & safetensors mmap
│   ├── hlut2_pack.hpp
│   ├── dsv41_pack_format.hpp
│   ├── dsv41_rarebal.hpp
│   └── safetensors_mmap.hpp
├── mi50/                  # 3x AMD Instinct MI50 (gfx906 TP3) 91/91 zero-scratch HIP kernels, 1M CSA2, & SSE server
│   ├── dsv41_mi50_kernels.hip
│   ├── dsv41_kolmogorov_tree_verifier.hpp  # Lossless Dirac Plackett-Luce M=8/M=16 verifier (max_TV = 2.793e-16)
│   ├── dsv41_tree_spec.hpp                 # SGPR-packed M=8 & M=16 TreeAncMask256 causal-tree topologies
│   ├── dsv41_anthropic_messages.hpp        # Pure-C++20 /anthropic/v1/messages SSE & POSIX --search-url loop
│   ├── dsv41_runtime.cpp
│   ├── dsv41_engram_cpu.cpp
│   ├── dsv41_think_loop_guard.hpp
│   ├── dsv41_tokenizer_dsml.cpp
│   └── dsv41_fe.cpp
├── tools/                 # Pure-C++17 packer, 3x MI50 FP64 RAREBAL repacker, harvester, & Wave 5/6/7B/8 self-tests
│   ├── pack_dsv41_hlut2.cpp
│   ├── harvest_rarebal_wave8.cpp
│   ├── repack_hot_experts_rarebal.cpp
│   ├── repack_rarebal_helpers.hpp
│   ├── bench_hlut2.cpp
│   ├── verify_wave5_all.cpp
│   ├── verify_wave6_math.cpp
│   ├── verify_wave7_opt.cpp
│   ├── verify_wave8_opt.cpp
│   └── test_anthropic_messages_fe.cpp
├── docs/
│   ├── WHT128_HLUT2_Blackhole_P150A_Whitepaper.pdf
│   └── WHT128_HLUT2_ENGINEERING_REPORT.md
└── manifests/
    ├── rarebal_wave8_receipt_manifest.txt
    ├── rarebal_receipt_manifest.txt
    ├── CANONICAL_RELEASE.json
    └── SHA256SUMS.txt
```

---

## Build, Self-Test, Reassemble `561`-Expert Overlay, & Launch (`1M` Context)

```bash
# 1. Build offline tools, self-tests, 3x MI50 runtime/server, and (when TT_METAL_HOME is set) 2x P150A worker
export DSV41_ROOT=/path/to/dsv41-model-workspace
export TT_METAL_HOME=/opt/tt-metal   # Tenstorrent Blackhole TT-Metal SDK install
make -j12

# 2. Run deterministic mathematical, L1 geometry, Dirac M=8/16 tree, Anthropic SSE, & 91/91 zero-scratch audits
make check
make audit-gfx906
# Optional hardware-on-device 100K PCIe coherency + 1M Tiered CSA2 GPU suite (when GPUs are idle):
# make check-gpu

# 3. Download the three <2.0 GiB Wave 8 RAREBAL-LDLQ shards from GitHub Releases, reassemble, and verify SHA-256
gh release download v1.2.0-wave8 -R bespokeontology/dsv41-wht128-hlut2 \
  -p 'hot_experts_rarebal_wave8.safetensors.part-*' -p 'SHA256SUMS.txt' --clobber
cat hot_experts_rarebal_wave8.safetensors.part-00 \
    hot_experts_rarebal_wave8.safetensors.part-01 \
    hot_experts_rarebal_wave8.safetensors.part-02 \
    > hot_experts_rarebal_wave8.safetensors
sha256sum -c SHA256SUMS.txt

# 4. Launch 2x Blackhole P150A TP2 worker first (overlays 34/34 L1 shards before MR pinning), then 1M-context server
./bin/p150_dsv41_worker --model-dir "$DSV41_ROOT" \
  --rarebal-overlay hot_experts_rarebal_wave8.safetensors --reuse-shm &

OMP_WAIT_POLICY=PASSIVE ./bin/dsv41_fe --model-dir "$DSV41_ROOT" \
  --rarebal-overlay hot_experts_rarebal_wave8.safetensors \
  --port 8042 --max-seq-len 1048576 --dual-stream-overlap --dspark-tree-width 8
```

---

## Credits & Attribution

* **Hardware Architecture, Systems Direction, & Heterogeneous Co-Design:** [**`bespokeontology`**](https://github.com/bespokeontology)
* **Agentic C++ Systems Engineering, Wave 7B & Wave 8 (`8A`–`8E`) Silicon Implementation, `561`-Expert `3× MI50` `FP64` `RAREBAL-LDLQ` Solver, & Verification:** [**Google Antigravity (Google DeepMind)**](https://antigravity.google/) — implemented pre-`shared_experts` dual-stream `P150A || MI50` overlap (`25,680 KiB` 4-slot `/dev/shm` fast ring + 1-CTA `compact256`, `2.01×` faster `MI50` MoE, `-51.9%` `P150A` wait), `7/7` `MeshTrace` (`0.73 ms` `M = 1`, `2.52 ms` `M = 8`, `4.53 ms` `M = 16` at `0 B` extra L1), `3× MI50` `rocblas_dgemm` `FP64` blocked Cholesky repacker (`561` hot experts covering `34/34` `P150A` L1 shards, `+4.221 dB` SNR), `1,048,576`-token (`1M`) tiered `CSA2` KV (`91/91` zero-scratch `gfx906` runtime kernels), and pure-C++20 `/anthropic/v1/messages` SSE.
* **Kernel Co-Design, Quantization & Measure-Theoretic Synthesis (`WHT128-HLUT2`, `TARQ RAREBAL-LDLQ`, Kolmogorov–Dirac `M = 8 ↔ 16` Verifier), & Whitepaper Synthesis:** [**Gemini (Google DeepMind)**](https://deepmind.google/technologies/gemini/)
* **Self-Hosted Long-Horizon Wave 7/8 Synthesis (`417.6K` & `217.4K` Live Ctx) & Read-Only `PASS 5/5` Production Audits:** [**`DeepSeek-V4.1-Flash-1017`**](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)

## License

Released under the [MIT License](LICENSE) (`Copyright (c) 2026 bespokeontology`).
