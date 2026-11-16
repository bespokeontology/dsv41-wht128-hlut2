# `WHT128-HLUT2` + `2,182`-Expert `TARQ RAREBAL-LDLQ` (`2.125 bpw → FP32`) on `2× Custom TSMC 6nm Tenstorrent Blackhole P150A` ASICs: `1M`-Context Multimodal C++ Engine for 763B `DeepSeek-V4.1-Flash-1017` (`v1.5.0-wave8.5`)

**Authors:** [**`bespokeontology`**](https://github.com/bespokeontology) — Heterogeneous Silicon & Quantization Co-Design Engineering · **100% Pure [`TT-Metal`](https://github.com/tenstorrent/tt-metal) C++20 / `gfx906` HIP / Host C++17 (`0` Python Files, `-Werror`)**  
📄 **[4-Page Illustrated Whitepaper (PDF)](docs/WHT128_HLUT2_Blackhole_P150A_Whitepaper.pdf)** · 📘 **[Full `v1.5.0-wave8.5` Engineering Report (`docs/WHT128_HLUT2_ENGINEERING_REPORT.md`)](docs/WHT128_HLUT2_ENGINEERING_REPORT.md)** · 🧾 **[`1M` E2E Silicon Receipt (`docs/WAVE8_5_1M_E2E_RECEIPT.md`)](docs/WAVE8_5_1M_E2E_RECEIPT.md)** · 🧪 **[Self-Hosted Wave 9 `WHT128-HLUT3` Report (`docs/WAVE9_HLUT3_SELFHOST_REPORT.md`)](docs/WAVE9_HLUT3_SELFHOST_REPORT.md)** · 🚀 **[`v1.4.0-wave8.4` Release](https://github.com/bespokeontology/dsv41-wht128-hlut2/releases/tag/v1.4.0-wave8.4)**

---

## `v1.5.0-wave8.5` Release Highlights (Wave 8.5 Canonical + Self-Hosted Wave 9 `WHT128-HLUT3` Foundation)

* **Wave 8.5 — `2,182`-Expert Starvation-Safe [`TARQ RAREBAL-LDLQ`](https://arxiv.org/pdf/2605.27808) (`99.47%` `DSpark` Hit Coverage), Cross-Sublayer `mHC` + `DSpark` Epilogue Fusion (`116/116` Zero-Scratch `gfx906`), & Full `1M` E2E Silicon Gate:** Expands [`TARQ RAREBAL-LDLQ`](https://arxiv.org/pdf/2605.27808) over a `2.05M`-token zero-PII multimodal (`DeepSeek-ViT`) + C++20/HIP/TT-Metal + formal math + DSML corpus (`M_cal >= 10,240`, `M_eval = 512`) to **`2,182` calibrated experts** (`1,996` main backbone + **`186` [`DSpark`](https://rocm.docs.amd.com/projects/atom/en/latest/deepseek_v41_dspark.html) `mtp.0..2` drafters = `99.47%` draft hit coverage** + **`18/18` (`36/36` TP2) `P150A` L1 shards**; `dual_receipt_violations = 0`, mean SwiGLU cosine `0.88344 → 0.95456`, **`+4.63 dB`** mean SNR gain, `0 B` extra VRAM). Fuses `HostRingAllReduce + mHC post/res + next-sublayer H_pre contraction + RMSNorm + FP8/INT8/WHT128`, 1-launch `20,480 → 24 + 20-iter Sinkhorn`, `inv_rope_tail64 + UE8M0/32`, `DSpark ConfidenceHead` projection, `L1v` `__ATOMIC_RELEASE`/`__ATOMIC_ACQUIRE` cache-key fences, and `DSML` bare-parameter quarantine (`116/116` compiled `gfx906` kernels at `ScratchSize == 0`), cutting short-context backbone latency **`-9.81 ms/step` (`97.39 → 87.58 ms/step`)** and delivering **`58.58 tok/s`** (`95.0%` `DSpark` acc, `τ = 5.70`) at `23.2K` ctx, **`623.0 tok/s`** fresh prefill, **`45.88 tok/s`** at `230.8K` ctx, **`42.05 tok/s`** at `288.2K` ctx, **`40.11 tok/s`** at `360.0K` ctx, **`34.26 tok/s`** at `390.2K` ctx, **`451,094 tok/s`** cold NVMe `.kvblk` hydration of **`391,936` tokens in `0.869 s`** (`0 B` disk SWA), and full **`1,048,576`-token (`1M`) E2E silicon verification in `59.06 s`** (`268 / 270 / 696 MiB` free HBM2 w/ ViT, `100.000%` `HSI` top-64 overlap across `8,192` blocks, `CSA2` `cosine = 1.00000000`, `3/3` needles, `32` `M = 8` bursts to `pos = 1,048,575`, `6.74 ms` cold `.kvblk` hydration + `128`-tok SWA replay `enc_cos = 1.00000000` / `dec_cos = 1.00000000`).
* **Self-Hosted Wave 9 — `WHT128-HLUT3` (`3.125 bpw`, `400 B` Tile / `6,400 B` `P150A` Superblock) Built Live by `DeepSeek-V4.1-Flash-1017` (`177` Tool Steps to `395.7K` Context):** Authored, debugged, compiled under `-Werror`, and verified on live `gfx906` silicon by the live Wave 8.5 engine itself across **`177` cumulative agentic tool steps** up to **`395,743` active context tokens** spanning two mid-campaign cold NVMe `.kvblk` restarts (`232.1K` and `391.9K` tokens). Solves the exact 8-level symmetric [Gaussian Lloyd-Max](https://web.stanford.edu/class/ee398a/handouts/papers/Max%20-%20Quantizing%20for%20Min%20Distortion.pdf) fixed point (`c = {0.24509418, 0.75600528, 1.34390928, 2.15194570}`, `SQNR_3 = 14.6158 dB`, **`+5.3155 dB`** over `HLUT2`), achieves **`0.98541` held-out SwiGLU cosine (`15.38 dB` SNR, `+5.16 dB` over `HLUT2+RAREBAL`)** and **`0.999999999` GPU-vs-CPU parity** on real Layer-6 routed `MXFP4` weights, fits **`11` hot shards/card (`1,413,248 B <= 1,459,200 B`, `+45,952 B` margin)** in `P150A` `FABRIC_1D` L1 SRAM, adds opt-in `--hlut3-mtp-overlay <path>` (`0 B` HBM2 delta when unset), and provides deterministic `4×–7× MI50` + `2×–3× P150A` PCIe switchbox planning (`plan_wave9_switchbox`).
* **Wave 8.4 — Cross-Tier `384`-Expert Minimax Balancer (`445` Swaps), `18/18` `P150A` L1 Shards + `x16 -> FABRIC_1D` Relay, & `v_dot2_f32_f16` `CSA2`:** Rebalances all `40 × 384` sparse routed experts across `P150A` L1 SRAM (`18` hottest/layer = `161.37 MiB`), `3× MI50` `3.07 TB/s` HBM2 (`216` warm-hot/layer), and `2× P150A` GDDR6 (`150` coldest long-tail/layer) via bit-exact `TP3 <-> TP2` repacking (`445` swapped pairs, logical-first `k_moe_gate` bias invariance `cos = 1.000000000`), relays Card `#1` traffic through Card `#0` `PCIe x16 -> 4× 800G FABRIC_1D` (`7/7` `MeshTrace` `parity_cos == 1.0000000`), and pairs pre-`dlopen` [ROCm `GPU_MAX_HW_QUEUES=8`](https://codimd.web.cern.ch/G3gNEdGjQEaH8_QcpW3bMQ/slide) with [`gfx906` `v_dot2_f32_f16`](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX906.html) `HSI` scoring and 10-split register-Q [`CSA2`](https://zartbot.github.io/blog/model_arch/dsv41flash_arch/en.html) `shared-tree8`.
* **Wave 8.3 — Single-Pass `ConfidenceHead` `Dynamic DeepSelect8` (`M = 8` SGPR Causal Tree Across `0–1M` Context):** `deep_select_top7_confidence_tree8` (`Dsv41DeepSelectPlan8`) scores a `16`-node (`depths 0..5`) single-pass `DSpark` candidate pool (`4` MTP micro-batches `B = 3, 2, 1, 1` = `7` rows, `+0.34 ms` drafter cost, `0 B` extra HBM2) via `confidence_head.proj.weight` (`[1, 5,376]`) and compiles dynamic `64`-bit SGPR `anc_mask64`, `depth_nibbles32`, `parent_nibbles32`, and `valid_node_mask` (`0` invalid filler acceptances across `100,000` shortage sweeps, **`max_TV_M8 = 2.463e-16 < 1e-14`**).
* **Wave 8.1 & 8.2 — Native Pure-C++/HIP `DeepSeek-ViT` (`253.45 MiB`), [`DeepInfer`](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md) Pure-Block Cold NVMe `CSA2` KV (`0 B` Disk SWA), & `PromptSegment` Anti-Injection:** Streams `BF16` `DeepSeek-ViT` on `tp = 2` (`268.0 / 270.0 / 696.0 MiB` free HBM2 at `1M`), persists `114,496 B` `.kvblk` records (`0 B` disk SWA, `128`-token bounded `Encoder SWA` cold replay `enc_cos = 1.00000000`, `dec_cos = 1.00000000`), enforces `PromptSegment` type-level anti-injection (`0` structural ID leaks on `18,476 B` hostile C++ payloads), and cuts host sampling `smp` **`-81%`**.

---

## Heterogeneous Silicon Architecture (`2× Blackhole P150A` + `3× MI50` + `EPYC` Dual-NVMe)

This repository contains the clean-room, zero-Python production multimodal inference engine and custom **`WHT128-HLUT2` + `2,182`-Expert `TARQ RAREBAL-LDLQ` (`2.125 bpw → IEEE FP32`)** quantization & cross-tier MoE balancer stack for **[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** up to **`1,048,576` (`1M`) active context tokens**:

1. **[`2× Tenstorrent Blackhole P150A`](https://docs.tenstorrent.com/aibs/blackhole/index.html) Custom TSMC `6nm` (`N6`) Spatial Dataflow ASICs (Lead L1-Hot & Long-Tail Sparse MoE Tier):** `220` compute Tensix cores (`2 × 110`, each with `5` RISC-V baby cores + `32×32` `HiFi2`/`FP32` Matrix Engine + `32`-lane Vector `SFPU` + dual `NoC` routers), `64 GiB` GDDR6 (`MP`), and **`2× 180 MB = 360 MB` on-die Tensix L1 SRAM** (**`18/18` pinned hot shards/card = `36/36` TP2 shards = `161.37 MiB` pure L1 weights** with **`+11,264 B` bank `0-0` margin** alongside `100%` L1-resident `x_decode` and `finish_decode[0..1]`; `7/7` `MeshTrace` shapes `parity_cos == 1.0000000` at `0.72 ms` `M = 1`, `2.18 ms` `M = 6`, `3.47 ms` `M = 8`, `4.59 ms` `M = 16`), cross-wired via **`4× 800G` QSFP-DD cables** (`3.2 Tbps` `FABRIC_1D` TP2) with **Card `#0` `PCIe x16 -> FABRIC_1D` relay (`--p150-x16-relay`)** bypassing Card `#1`'s narrow `PCIe x4` host riser (`623.0 tok/s` fresh prefill).
2. **`3× AMD Instinct MI50` (Warm-Hot `216`-Expert MoE, Fused `mHC` Backbone, `DeepSeek-ViT`, `1M` Tiered `CSA2`, `Dynamic DeepSelect8`, & Opt-In `HLUT3` Tier):** `96 GiB` HBM2 (`3.07 TB/s` aggregate, `Vega20 gfx906` TP3, **`116/116` compiled `gfx906` kernels [`106` runtime+ViT + `5` `HLUT3` + `5` verifiers] at `ScratchSize == 0`**, pre-`dlopen` `GPU_MAX_HW_QUEUES=8`, `0` PCIe BAR P2P reads via coherent pinned-`DDR4` `FP32` `HostRingAllReduce`, cross-sublayer fused `mHC` + `RMSNorm` + `DSpark` epilogue [`87.58 ms/step`], logical-first `d_expert_perm[40 * 384]` router remap [`445` cross-tier swaps, `cos = 1.000000000`], `55`-VGPR `v_dot2_f32_f16` `k_indexer_score_msplit<8>`, 10-split register-Q `CSA2` `shared-tree8`, `253.45 MiB` streamed `BF16` `DeepSeek-ViT` on `tp = 2`, `Dynamic DeepSelect8` backed by **`186` `RAREBAL` `mtp.0..2` drafter experts [`99.47%` hit coverage]**, and opt-in `--hlut3-mtp-overlay` `400 B/tile` decode kernels).
3. **`AMD EPYC` + Dual NVMe (`Engram`, Sub-Second `392K` Cold NVMe `.kvblk` Persistence, `1M` Warm KV Ring, & SSE Tier):** `384 GiB` DDR4 hosting `188.83 GiB` pre-faulted native `FP8 E4M3` `Engram` tables (`207.69 GB/s` warm hydration in `0.98 s`, `0.0009 ms/chunk` `M = 512` double-buffered prefill overlap), `114,496 B` `.kvblk` pure-block cold NVMe `CSA2` persistence (`0 B` disk SWA, **`391,936` tokens hydrated in `0.869 s` at `451,094 tok/s`**), `64`-block (`86.371 MiB`) RAM `EncoderStateBlock` ring, coherent mapped pinned-DDR4 warm `CSA2` KV (**`1,048,576` (`1M`) verified end-to-end on silicon**; **`395,743` multimodal + C++20/HIP agentic & `417,556` text verified live context**), and pure-C++ `/v1/chat/completions` + `/anthropic/v1/messages` SSE + POSIX search sidecar.

---

## Repository Structure (`p150/` Lead Hierarchy)

```text
├── p150/                  # 2x Custom TSMC 6nm Tenstorrent Blackhole P150A TT-Metal C++20 FABRIC_1D TP2 engine
│   ├── kernels/           # 13 Tensix compute/NoC kernels (O(1) l1_slot_lut, x16->FABRIC_1D relay, HiFi2 -> FP32 Dst)
│   ├── dsv41_p150_l1.hpp  # Tier-2 FABRIC_1D per-bank L1 budget (18/18 hot shards = 161.37 MiB + +11,264 B margin)
│   ├── dsv41_p150_hlut3_spec.hpp # Wave 9: 6,400 B HLUT3 superblock & 11-shard (1,413,248 B <= 1,459,200 B) L1 spec
│   ├── dsv41_p150_shm.hpp # Lock-free 4-slot 25,680 KiB /dev/shm fast ring + M=1..1024 mailbox
│   ├── p150_dsv41_engine.cpp
│   └── p150_dsv41_worker.cpp
├── src/                   # WHT128-HLUT2 (2.125 bpw), Wave 9 WHT128-HLUT3 (3.125 bpw), RAREBAL-LDLQ, & mmap
│   ├── hlut2_pack.hpp
│   ├── hlut3_pack.hpp
│   ├── dsv41_hlut3_rarebal.hpp
│   ├── dsv41_pack_format.hpp
│   ├── dsv41_rarebal.hpp
│   ├── dsv41_wave8_4_repack.hpp
│   └── safetensors_mmap.hpp
├── mi50/                  # 3x MI50 (gfx906 TP3) 111/111 runtime+HLUT3 zero-scratch HIP kernels, ViT, NVMe KV, & SSE
│   ├── dsv41_mi50_kernels.hip              # Fused mHC/Sinkhorn/DSpark + v_dot2_f32_f16 HSI + 10-split CSA2 tree8
│   ├── dsv41_hlut3_kernels.{hpp,hip}       # Wave 9: Zero-scratch 400 B/tile WHT128-HLUT3 TP3 decode kernels
│   ├── dsv41_vit.{hpp,cpp}                 # Wave 8.1: Pure-C++/HIP DeepSeek-ViT + 2D-RoPE + 3x3 unshuffle + Aligner
│   ├── dsv41_nvme_kv_cache.hpp             # Wave 8.2: 114,496 B .kvblk pure-block cold NVMe CSA2 KV (0 B disk SWA)
│   ├── dsv41_kolmogorov_tree_verifier.hpp  # Wave 8.3: Dynamic DeepSelect8 + valid_node_mask (max_TV = 2.463e-16)
│   ├── dsv41_tree_spec.hpp                 # Dynamic 64-bit SGPR M=8 & TreeAncMask256 M=16 causal-tree topologies
│   ├── dsv41_anthropic_messages.hpp        # PromptSegment anti-injection + /anthropic/v1/messages + multimodal spans
│   ├── dsv41_runtime.cpp
│   ├── dsv41_engram_cpu.cpp
│   ├── dsv41_think_loop_guard.hpp
│   ├── dsv41_tokenizer_dsml.cpp
│   └── dsv41_fe.cpp
├── tools/                 # Pure-C++17 packers, 2,182-expert FP64 RAREBAL solver, & 1M silicon self-tests
│   ├── build_wave8_5_rarebal.cpp           # Wave 8.5A: 2,182-expert starvation-safe FP64 TARQ RAREBAL-LDLQ solver
│   ├── build_wave8_4_expert_perm.cpp       # Wave 8.4: Cross-tier 384-expert minimax balancer (445 swaps)
│   ├── harvest_rarebal_wave8.cpp           # Zero-PII 2.05M-token multimodal + C++20/HIP/math/DSML harvester
│   ├── verify_wave8_5.cpp                  # Wave 8.5: 2,182-expert audit + fused mHC / Sinkhorn / DSpark parity
│   ├── verify_wave8_5_1m_e2e.cpp           # Wave 8.5C: Full 1,048,576-token (1M) E2E silicon verification suite
│   ├── verify_wave9_hlut3.cpp              # Wave 9: Analytic Lloyd-Max + 400B/6400B pack + MI50 HLUT3 parity
│   ├── bench_hlut3_mi50.cpp                # Wave 9: 5-baseline real MXFP4 SwiGLU Dual Receipt + MI50 bench
│   ├── plan_wave9_switchbox.cpp            # Wave 9: Deterministic 4x-7x MI50 + 2x-3x P150A switchbox planner
│   ├── verify_wave8_4.cpp
│   ├── pack_dsv41_hlut2.cpp
│   ├── pack_dsv41_vit.cpp
│   ├── repack_hot_experts_rarebal.cpp
│   ├── dsv41_search_sidecar.cpp
│   ├── verify_wave5_all.cpp
│   ├── verify_wave6_math.cpp
│   ├── verify_wave7_opt.cpp
│   ├── verify_wave8_opt.cpp
│   ├── verify_wave8_1_nvme_kv.cpp
│   ├── verify_wave8_1_vit.cpp
│   ├── verify_wave8_2_deepinfer.cpp
│   └── test_anthropic_messages_fe.cpp
├── scripts/               # Parameterized production launchers (GPU_MAX_HW_QUEUES=8, --p150-x16-relay, fused mHC)
│   ├── run_p150_worker.sh
│   └── run_dsv41_fe.sh
├── docs/
│   ├── WHT128_HLUT2_Blackhole_P150A_Whitepaper.pdf
│   ├── WHT128_HLUT2_ENGINEERING_REPORT.md
│   ├── WAVE8_5_1M_E2E_RECEIPT.md
│   └── WAVE9_HLUT3_SELFHOST_REPORT.md
└── manifests/
    ├── wave8_5_receipt_manifest.txt
    ├── WAVE8_5_RELEASE_RECEIPT.md
    ├── wave9_verify.log
    ├── wave9_bench_hlut3.log
    ├── wave9_switchbox.log
    ├── wave8_4_receipt_manifest.txt
    ├── WAVE8_4_RELEASE_RECEIPT.md
    ├── CANONICAL_RELEASE.json
    └── SHA256SUMS.txt
```

---

## Build, Self-Test (`116/116` Zero-Scratch `gfx906`), & Launch (`1M` Multimodal Context)

```bash
# 1. Build offline tools, self-tests, 3x MI50 + ViT + HLUT3 runtime/server, and (when TT_METAL_HOME is set) 2x P150A worker
export DSV41_ROOT=/path/to/dsv41-model-workspace
export TT_METAL_HOME=/opt/tt-metal   # Tenstorrent Blackhole TT-Metal SDK install
make -j12

# 2. Run pure host-CPU mathematical, Tier-2 L1 geometry, switchbox planner, & 116/116 zero-scratch gfx906 audits
make check
make audit-gfx906
# Optional hardware-on-device PCIe coherency, ViT parity, Cold NVMe .kvblk, fused mHC, & 1M E2E GPU suite (when GPUs are idle):
# make check-gpu
# GPU_MAX_HW_QUEUES=8 ./bin/verify_wave8_5_1m_e2e

# 3. Pack contiguous BF16 DeepSeek-ViT weights once (lossless BF16, 0 Python)
./bin/pack_dsv41_vit --packed-mi50-dir "$DSV41_ROOT/packed_mi50" --out-dir "$DSV41_ROOT/packed_vit"

# 4. Launch 2x Blackhole P150A TP2 worker (18/18 L1 shards + x16->FABRIC_1D relay), search sidecar, and 1M multimodal server
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-8}"

./bin/p150_dsv41_worker --model-dir "$DSV41_ROOT" \
  --rarebal-overlay "$DSV41_ROOT/packed_rarebal_overlay_wave8_5/hot_experts_wave8_5.safetensors" \
  --p150-l1-shards 18 --p150-x16-relay --reuse-shm &

./bin/dsv41_search_sidecar --host 127.0.0.1 --port "${DSV41_SEARCH_PORT:-8080}" &

OMP_WAIT_POLICY=PASSIVE ./bin/dsv41_fe --model-dir "$DSV41_ROOT" \
  --rarebal-overlay "$DSV41_ROOT/packed_rarebal_overlay_wave8_5/hot_experts_wave8_5.safetensors" \
  --expert-perm-overlay "$DSV41_ROOT/packed_rarebal_overlay_wave8_5/hot_experts_wave8_5.safetensors" \
  --vit-weights "$DSV41_ROOT/packed_vit/dsv41_vit_bf16.safetensors" \
  --nvme-kv-cache-dir "$DSV41_ROOT/nvme_kv_cache" --nvme-kv-cache-gb 32 \
  --search-url "${SEARCH_URL:-http://127.0.0.1:8080}" \
  --port "${DSV41_PORT:-8000}" --max-seq-len 1048576 \
  --dual-stream-overlap --csa2-shared-tree8 --fused-mhc-norm --fused-dspark-epilogue --dspark-tree-width auto
```

---

## Attribution & License

* **Authors & Maintainer:** [**`bespokeontology`**](https://github.com/bespokeontology) — Heterogeneous Silicon & Quantization Co-Design Engineering
* **Base Multimodal Model (`763B`-Class):** [**`deepseek-ai/DeepSeek-V4.1-Flash-1017`**](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
* **License:** Released under the [MIT License](LICENSE) (`Copyright (c) 2026 bespokeontology`).
