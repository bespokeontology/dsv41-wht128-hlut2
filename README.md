# `2× Tenstorrent Blackhole P150A` + `WHT128-HLUT2` (`2.125 bpw`): Native C++ Inference Engine for 748.6B `DeepSeek-V4.1-Flash-1017` (`v1.1.0-wave7b` Canonical)

**Co-Engineered by [**`bespokeontology`**](https://github.com/bespokeontology), [**Google Antigravity (Google DeepMind)**](https://antigravity.google/), & [**Gemini (Google DeepMind)**](https://deepmind.google/technologies/gemini/)** · **100% Pure TT-Metal C++20 / `gfx906` HIP / Host C++17 (`0` Python Files)**  
📄 **[4-Page Illustrated Whitepaper (PDF)](docs/WHT128_HLUT2_Blackhole_P150A_Whitepaper.pdf)** · 📘 **[Full Wave 7B Engineering Report (`docs/WHT128_HLUT2_ENGINEERING_REPORT.md`)](docs/WHT128_HLUT2_ENGINEERING_REPORT.md)**

---

## Overview

This repository contains the clean-room, zero-Python production inference engine for **[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** (**`748.6B` total parameters**: `559.7B` `40`-layer [`CSA2`](https://zartbot.github.io/blog/model_arch/dsv41flash_arch/en.html) MoE + `188.83 GiB` external [`Engram`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/engram.py) `N`-gram memory + `3`-stage [`DSpark`](https://rocm.docs.amd.com/projects/atom/en/latest/deepseek_v41_dspark.html) speculative drafter) co-designed across heterogeneous non-CUDA silicon:

1. **[`2× Tenstorrent Blackhole P150A`](https://docs.tenstorrent.com/aibs/blackhole/index.html) (Lead Sparse MoE Tier):** `220` compute Tensix cores (`2 × 110`), `64 GiB` GDDR6 (`MP`), and **`2× 180 MB = 360 MB` on-die Tensix L1 SRAM** (**`247.82 MiB` pure active L1** [`123.91 MiB/card = 129,927,808 B/card`], **`34` TP2 hot shards = `152.40 MiB`** pinned in L1 alongside `100%` L1-resident `x_decode` and `finish_decode[0..1]`), cross-wired card-to-card via **`4× 800G` QSFP-DD cables** (`3.2 Tbps` `FABRIC_1D` TP2), with **one `P150A` running on a narrow `PCIe x4` host riser**.
2. **`3× AMD Instinct MI50` (Dense Backbone, `CSA2`, & `M = 8` `DSpark` Causal-Tree Tier):** `96 GiB` HBM2 (`Vega20 gfx906` TP3, **`88/88` HIP kernels `ScratchSize == 0`** including `BM = 32` `k_hlut2_grouped_down32`, `0` PCIe BAR P2P reads via coherent pinned-`DDR4` `FP32` `HostRingAllReduce` + zero-SDMA `20,544 KiB` `/dev/shm` mailbox).
3. **`AMD EPYC` + Dual NVMe (`Engram`, Prefix Cache, & `AVX2` Tail Tier):** `384 GiB` DDR4 hosting `188.83 GiB` pre-faulted native `FP8 E4M3` `Engram` tables (`218.11–248.98 GB/s` warm hydration in `0.81–0.93 s`), 4-tier stateless `CSA2` prefix cache (`524,288` `512K` YaRN context, **`417,556` verified live active context**), and `29`-thread `AVX2` `CpuTailPool`.

### Headline Measured Performance (`v1.1.0-wave7b` Live Agentic Engineering Sessions)

* **`614.8 tok/s`** 20-layer [`CED` fresh prefill](https://arxiv.org/pdf/2609.19969) across `17,510` fresh tokens at `27.9K` ctx (**`583.3 tok/s`** across `28,868` fresh tokens at `61.2K` ctx; **`> 500 tok/s` sustained past `146K` context** — `+22%` to `+53%` over Wave 6C)
* **`30.30 tok/s`** sustained `12,528`-token `<think>` reasoning at `44K–57K` ctx (`tau = 3.45`, `tree = 1452/1201`), **`28.11–31.91 tok/s`** at `97K–122K` ctx (`+45.8%` to `+61.8%` over Wave 6C), and **`24.73–28.55 tok/s`** at `160K–173K` ctx (`+39.5%` to `+49.5%` via `M = 8` Dirac causal-tree speculation)
* **`34.23 – 36.84 tok/s`** matched-context [`DSML`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/encoding/README.md) tool turns (`+38.3%` over Wave 6C; up to `47.5–52.2 tok/s` peak `DSpark` decode and `39.72 tok/s` at `258.4K` ctx)
* **`53.84 – 58.09 ms/step`** across all `40` `2× Blackhole P150A` MoE layers (`0.81 ms` traced `M = 1`, `2.32 ms` traced `M = 6`, **`2.58 ms` traced `M = 8`**, `parity_cos = 1.0000000` — with unhidden host wait `p150w` reduced to **`52.01–53.78 ms/step`** via concurrent MI50 + zero-SDMA overlap)
* **`2.68B tok/s`** Tier-0 in-VRAM `LCP` reuse · **`383,424 tok/s`** warm `171.5K`-token `RadixBlockCache` hydration (`447.34 ms`, `+45.2%`) · **`1.998M tok/s`** dirty-only cross-session switch (`72.26 ms` @ `144K`)

---

## Wave 7B (`v1.1.0-wave7b`) Architectural Breakthroughs

1. **Track 7A — `34` Pinned `Blackhole P150A` Hot Shards (`152.40 MiB` L1) + `100%` L1-Resident Decode + Zero-SDMA Mailbox:**
   * Expands static hot-expert pinning from `11` to **`17` shards/card (`34` TP2 shards = `159,805,440 B = 152.40 MiB` L1)** inside [`FABRIC_1D`](https://github.com/tenstorrent/tt-metal) (`kP150L1SmallSizeBytes = 2,048 B`, `kP150L1Fabric1DBankBytes = 1,459,200 B`, Tier-1b per-bank usage = **`1,453,184 B <= 1,459,200 B`**).
   * Keeps **`x_decode` (`[1, 1, 32, 5120]` BF16 = `327,680 B` = `4,096 B/bank`)** and **`finish_decode[0..1]` (`typecast` + `FABRIC_1D` `all_reduce` = `655,360 B` = `8,192 B/bank`) `100%` L1-resident across all `40` decode layers**, staging only `finish_decode[2]` (`untilize` `ROW_MAJOR` PCIe host-read buffer = `10,240 B` page on bank `0-0`) in GDDR6 with **`+8,832 B` post-trace zero-clash hardware safety margin** (`367,232 B > 358,400 B`, `0` `TT_THROW`, exact pure L1 = **`129,927,808 B = 123.91 MiB/card`**).
   * Maps `/dsv41_p150_ipc` (`21,037,248 B = 20,544 KiB`) across all `3× MI50` GPUs via `hipHostRegisterMapped` and posts the P150A doorbell strictly ahead of MI50 `[0..215]` + partial TP3 ring uploads (`0` `hipMemcpyAsync` SDMA launches on `M <= 64`, hiding **`3.41–5.71 ms/step`** of wait time).
2. **Track 7B — Zero-Scratch `BM = 32` `k_hlut2_grouped_down32` (`88/88` `gfx906` Kernels `ScratchSize == 0`):**
   * Pairs `k_hlut2_grouped_swiglu16` (`BM = 16`) with a VGPR-promoted `192`-thread (`3` `wave64` / `6 × 32`-lane) `k_hlut2_grouped_down32` kernel (`48 KiB` swizzled `uint4` LDS, register-cached `pd_reg[8]`, `32`-row `2.125 bpw` tile reuse, `cosine = 1.000000000000` vs. token-major reference), accelerating `20`-layer `CED` fresh prefill to **`614.8 tok/s`** and sustaining **`> 500 tok/s` past `146K` context**.
3. **Track 7C — SGPR-Packed `M = 8` `DSpark` Causal-Tree Attention & Kolmogorov–Dirac Verifier (`max_TV = 1.735e-16`):**
   * Dynamically upgrades `DSpark` drafting to a width-`[3, 2, 1, 1]` `8`-node causal tree when root confidence drops (`p_root_top1 < 0.72f`), packing ancestor visibility (`0x9345231309050301ULL`), depth nibbles (`0x32221110u`), and parent nibbles (`0x4211000Fu`) into scalar SGPRs (`0 B` scratch) with sibling-invariant `base_pos + depth[i]` Givens RoPE and alias-free forward KV compaction.
   * Verifies deterministic top-$B$ branches via exact Dirac proposals $q_{u,j} = \delta_{x_j}$ and Plackett–Luce residual renormalization (`mi50/dsv41_kolmogorov_tree_verifier.hpp`, `max_TV = 1.735e-16`, `17.11 ns/step`), preventing the `0.40` TV bias of naive deterministic `min(1, p/q)`.

---

## How `WHT128-HLUT2` (`2.125 bpw`) Preserves Full-Precision `748.6B` Fidelity

`WHT128-HLUT2` (`2.125 bpw`, `272 B / 32×32` tile, `4,352 B / 128×128` 16-subtile `UInt32` NoC superblock) compresses **exclusively** the `384` sparse conditional routed SwiGLU experts (`6 / 384 = 1.56%` active per token), keeping **every always-active and state-bearing subsystem at full or lossless precision** inside `160 GiB` total accelerator DRAM (`64 GiB` `P150A` GDDR6 + `96 GiB` `MI50` HBM2):

* **`BF16` / `FP32` (`16–32 bpw`)** on embeddings, `RMSNorm`, [`mHC` (`n = 4` Sinkhorn)](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json), `384`-way routers, `LM Head`, and Wave 6 `FP32` `HostRingAllReduce + mHC post` (`266.8×` lower truncation MSE);
* **`8.01 bpw` (`cos > 0.9999`)** on all `40` [`CSA2` attention layers](https://zartbot.github.io/blog/model_arch/dsv41flash_arch/en.html), `HSI` indexers, `40` always-active shared SwiGLU experts, and `3` [`DSpark`](https://rocm.docs.amd.com/projects/atom/en/latest/deepseek_v41_dspark.html) dense heads;
* **Bitwise-exact native `8.25 bpw` (`FP8 E4M3 + FP16` scale, zero re-quantization)** on the `188.83 GiB` external [`Engram`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/engram.py) `N`-gram memory in `384 GiB DDR4`;
* **Wave 6B `56/56` [`TARQ RAREBAL-LDLQ`](https://arxiv.org/pdf/2605.27808) Overlay (`hot_experts_rarebal.safetensors`, `503 MB`):** Causal Reverse-Cholesky [`QuIP#` `LDLQ`](https://arxiv.org/html/2402.04396v1) + `2K` starvation gate + trace equalization across `24` `P150A`, `20` `MI50`, and `12` `DSpark` MTP hot experts (`+2.90` to `+6.14 dB` held-out SNR gain at `0 B` extra VRAM).

---

## Repository Structure (`p150/` Lead Hierarchy)

```text
├── p150/                  # 2x Tenstorrent Blackhole P150A TT-Metal C++20 FABRIC_1D TP2 engine & 13 Tensix kernels
│   ├── kernels/           # dsv41_hlut2_writer_b.cpp (O(1) l1_slot_lut + 2*sb_cnt CBs), dsv41_hlut2_compute.cpp, ...
│   ├── dsv41_p150_l1.hpp  # Exact Tier-1b FABRIC_1D per-bank L1 budget (17/17 shards + L1 x_decode = 123.91 MiB/card)
│   ├── dsv41_p150_shm.hpp # Lock-free M=1..1024 zero-SDMA POSIX /dev/shm mailbox (20,544 KiB)
│   ├── p150_dsv41_engine.cpp
│   └── p150_dsv41_worker.cpp
├── src/                   # WHT128-HLUT2 (2.125 bpw) + Reverse-Cholesky LDLQ + TARQ RAREBAL & safetensors mmap
│   ├── hlut2_pack.hpp
│   ├── dsv41_pack_format.hpp
│   ├── dsv41_rarebal.hpp
│   └── safetensors_mmap.hpp
├── mi50/                  # 3x AMD Instinct MI50 (gfx906 TP3) 88/88 zero-scratch HIP kernels, runtime, & :8042 server
│   ├── dsv41_mi50_kernels.hip
│   ├── dsv41_kolmogorov_tree_verifier.hpp  # Lossless Dirac Plackett-Luce M=8 tree verifier (max_TV = 1.735e-16)
│   ├── dsv41_tree_spec.hpp                 # SGPR-packed M=8 width-[3,2,1,1] causal-tree bitmasks & nibbles
│   ├── dsv41_runtime.cpp
│   ├── dsv41_engram_cpu.cpp
│   ├── dsv41_think_loop_guard.hpp
│   ├── dsv41_tokenizer_dsml.cpp
│   └── dsv41_fe.cpp
├── tools/                 # Pure-C++17 offline packer, RAREBAL repacker, ablation bench, & Wave 5/6/7B self-tests
│   ├── pack_dsv41_hlut2.cpp
│   ├── repack_hot_experts_rarebal.cpp
│   ├── repack_rarebal_helpers.hpp
│   ├── bench_hlut2.cpp
│   ├── verify_wave5_all.cpp
│   ├── verify_wave6_math.cpp
│   └── verify_wave7_opt.cpp
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

# 2. Run Wave 5, Wave 6, Wave 6C, and Wave 7B mathematical + L1 geometry + Dirac tree + gfx906 zero-scratch self-tests
make check
make audit-gfx906

# 3. Download `hot_experts_rarebal.safetensors` from GitHub Releases and verify SHA-256
sha256sum -c SHA256SUMS.txt

# 4. Launch 2x Blackhole P150A TP2 worker first, then 3x MI50 + 189 GiB Engram HTTP/SSE server
./bin/p150_dsv41_worker --model-dir "$DSV41_ROOT" --rarebal-overlay hot_experts_rarebal.safetensors --reuse-shm &
OMP_WAIT_POLICY=PASSIVE ./bin/dsv41_fe --model-dir "$DSV41_ROOT" --rarebal-overlay hot_experts_rarebal.safetensors --port 8042 --max-seq-len 524288
```

---

## Credits & Attribution

* **Hardware Architecture, Systems Direction, & Heterogeneous Co-Design:** [**`bespokeontology`**](https://github.com/bespokeontology)
* **Agentic C++ Systems Engineering, Wave 7B Tier-1b L1 Calibration, Kernel Integration, & Verification:** [**Google Antigravity (Google DeepMind)**](https://antigravity.google/) — restored `100%` L1 residency for `x_decode` + `finish_decode[0..1]` with GDDR6 `finish_decode[2]` `ROW_MAJOR` staging (`+8,832 B` post-trace bank `0-0` safety margin, `123.91 MiB/card` pure active L1), integrated `88/88` zero-scratch `gfx906` `k_hlut2_grouped_down32` + SGPR `M = 8` Dirac causal-tree speculation (`max_TV = 1.735e-16`), and verified Wave 7B end-to-end.
* **Kernel Co-Design, Quantization & Measure-Theoretic Synthesis (`WHT128-HLUT2`, `TARQ RAREBAL-LDLQ`, Kolmogorov–Dirac Verifier), & Whitepaper Synthesis:** [**Gemini (Google DeepMind)**](https://deepmind.google/technologies/gemini/)
* **Self-Hosted Long-Horizon Wave 7 Synthesis (`202` Steps @ `417.6K` Ctx) & Read-Only `PASS 5/5` Audit:** [**`DeepSeek-V4.1-Flash-1017`**](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)

## License

Released under the [MIT License](LICENSE) (`Copyright (c) 2026 bespokeontology`).
