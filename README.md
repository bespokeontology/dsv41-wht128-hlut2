# `WHT128-HLUT2` + `778`-Expert `TARQ RAREBAL-LDLQ` (`2.125 bpw → FP32`) on `2× Custom TSMC 6nm Tenstorrent Blackhole P150A` ASICs: `1M`-Context Multimodal C++ Engine for 763B `DeepSeek-V4.1-Flash-1017` (`v1.4.0-wave8.4`)

**Authors:** [**`bespokeontology`**](https://github.com/bespokeontology) — Heterogeneous Silicon & Quantization Co-Design Engineering · **100% Pure [`TT-Metal`](https://github.com/tenstorrent/tt-metal) C++20 / `gfx906` HIP / Host C++17 (`0` Python Files, `-Werror`)**  
📄 **[4-Page Illustrated Whitepaper (PDF)](docs/WHT128_HLUT2_Blackhole_P150A_Whitepaper.pdf)** · 📘 **[Full `v1.4.0-wave8.4` Engineering Report (`docs/WHT128_HLUT2_ENGINEERING_REPORT.md`)](docs/WHT128_HLUT2_ENGINEERING_REPORT.md)** · 🚀 **[`v1.4.0-wave8.4` Release](https://github.com/bespokeontology/dsv41-wht128-hlut2/releases/tag/v1.4.0-wave8.4)** · 📦 **[`778`-Expert `Wave 8.4` Balanced Overlay (`v1.4.0-wave8.4`)](https://github.com/bespokeontology/dsv41-wht128-hlut2/releases/tag/v1.4.0-wave8.4)**

---

## `v1.4.0-wave8.4` Release Highlights (Wave 8.4 Canonical: `384`-Expert Cross-Tier Balancer + `778` `RAREBAL` Experts + `18/18` `P150A` L1 Relay)

* **Wave 8.4 — Cross-Tier `384`-Expert Minimax Balancer (`445` Swaps), `778`-Expert [`TARQ RAREBAL-LDLQ`](https://arxiv.org/pdf/2605.27808) (`2.97×` MTP), `18/18` `P150A` L1 Shards + `x16 -> FABRIC_1D` Relay, & `99/99` Zero-Scratch `gfx906` Kernels:** Rebalances all `40 × 384` sparse routed experts by empirical activation frequency across `P150A` L1 SRAM (`18` hottest/layer), `3× MI50` `3.07 TB/s` HBM2 (`216` warm-hot/layer), and `2× P150A` GDDR6 (`150` coldest long-tail/layer) via bit-exact `TP3 <-> TP2` repacking (`445` swapped pairs, `565` base repacked experts, logical-first `k_moe_gate` bias invariance `cos = 1.000000000`, `0` routing diffs, `0 B` extra VRAM). Expands [`TARQ RAREBAL-LDLQ`](https://arxiv.org/pdf/2605.27808) to **`778` starvation-safe experts** (`668` backbone + **`110` [`DSpark`](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/) `mtp.0..2` drafters** + **`18/18` (`36/36` TP2 = `161.37 MiB`)** pinned `P150A` L1 shards w/ `+11,264 B` bank `0-0` margin; `0` Dual Receipt violations), relays Card `#1` traffic through Card `#0` `PCIe x16 -> 4× 800G FABRIC_1D` (`7/7` `MeshTrace` `parity_cos == 1.0000000`, **`547.69 tok/s`** fresh `225K` multimodal prefill, **`+21.8%`**), and combines pre-`dlopen` [ROCm `GPU_MAX_HW_QUEUES=8`](https://codimd.web.cern.ch/G3gNEdGjQEaH8_QcpW3bMQ/slide), [`gfx906` `v_dot2_f32_f16`](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX906.html) `HSI` scoring, and 10-split register-Q [`CSA2`](https://zartbot.github.io/blog/model_arch/dsv41flash_arch/en.html) `shared-tree8` (`99/99` `ScratchSize == 0`) — delivering **`0.623 s` (`381,290 tok/s`)** cold NVMe hydration of **`237,568` tokens**, **`38.91 tok/s`** post-cold-reload decode (`+36.8%` vs. [`v1.3.0-wave8.3`](https://github.com/bespokeontology/dsv41-wht128-hlut2/releases/tag/v1.3.0-wave8.3)), **`36.22 tok/s`** steady multimodal tool decode at `232.7K` (`+10.3%`), **`25.47–31.19 tok/s`** deep `T = 1.0` `<think>` synthesis at `237.0K–238.5K` (`+20.1%`, `<think>` `DSpark` acc **`+8.1 pp`** `59.5% -> 67.6%`), and **`40.18 tok/s`** (`83.6%` warm `DSpark` acc, `97.39 ms/step`) under `25K` context.
* **Wave 8.3 — Single-Pass `ConfidenceHead` `Dynamic DeepSelect8` (`M = 8` SGPR Causal Tree Across `0–1M` Context):** `deep_select_top7_confidence_tree8` (`Dsv41DeepSelectPlan8`) scores a `16`-node (`depths 0..5`) single-pass `DSpark` candidate pool (`4` MTP micro-batches `B = 3, 2, 1, 1` = `7` rows, `+0.34 ms` drafter cost, `0 B` extra HBM2) via `confidence_head.proj.weight` (`[1, 5,376]`) and compiles dynamic `64`-bit SGPR `anc_mask64`, `depth_nibbles32`, `parent_nibbles32`, and `valid_node_mask` (`0` invalid filler acceptances across `100,000` shortage sweeps, **`max_TV_M8 = 2.463e-16 < 1e-14`**).
* **Wave 8.2 — [`DeepInfer`](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md) Pure-Block Cold NVMe `CSA2` KV (`0 B` Disk SWA), `PromptSegment` Anti-Injection & `-81%` Lazy Softmax:** Persists `128`-token `.kvblk` records (`114,496 B` = `64 B` `Dsv41NvmeKvBlockHeader` + `512 B` token IDs + `113,920 B` `CSA2` KV; `0` `.kvtail`/`.swa` files on disk) paired with a `64`-block (`86.371 MiB`) RAM `EncoderStateBlock` ring and `128`-token bounded `Encoder SWA` cold replay (`enc_cos = 1.00000000`, `dec_cos = 1.00000000`), enforces C++ type-level anti-injection via `PromptSegment` (`0` structural ID leaks on `18,476 B` hostile C++ payloads), and cuts host sampling `smp` **`-81%` (`7.04 → 1.17–1.37 ms/step`)**.
* **Wave 8.1 — Native Pure-C++/HIP `DeepSeek-ViT` Vision (`Dsv41VitEngine`, `253.45 MiB` HBM2 on `tp = 2`):** Executes the full **`763B` multimodal [`DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** (`748.6B` language + vision encoder) in pure C++/HIP with double-buffered pinned-DDR4 weight streaming, `2D-RoPE`, `3×3` pixel-unshuffle, 2-layer MLP projector, image router bias, and [`Engram`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/engram.py) image-token masking while preserving **`266.0 / 270.0 / 696.0 MiB` free HBM2** across `tp = 0..2` at `max_seq_len = 1,048,576`.

---

## Heterogeneous Silicon Architecture (`2× Blackhole P150A` + `3× MI50` + `EPYC` Dual-NVMe)

This repository contains the clean-room, zero-Python production multimodal inference engine and custom **`WHT128-HLUT2` + `778`-Expert `TARQ RAREBAL-LDLQ` (`2.125 bpw → IEEE FP32`)** quantization & cross-tier MoE balancer stack for **[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** up to **`1,048,576` (`1M`) active context tokens**:

1. **[`2× Tenstorrent Blackhole P150A`](https://docs.tenstorrent.com/aibs/blackhole/index.html) Custom TSMC `6nm` (`N6`) Spatial Dataflow ASICs (Lead L1-Hot & Long-Tail Sparse MoE Tier):** `220` compute Tensix cores (`2 × 110`, each with `5` RISC-V baby cores + `32×32` `HiFi2`/`FP32` Matrix Engine + `32`-lane Vector `SFPU` + dual `NoC` routers), `64 GiB` GDDR6 (`MP`), and **`2× 180 MB = 360 MB` on-die Tensix L1 SRAM** (**`18/18` pinned hot shards/card = `36/36` TP2 shards = `161.37 MiB` pure L1 weights** with **`+11,264 B` bank `0-0` margin** alongside `100%` L1-resident `x_decode` and `finish_decode[0..1]`; `7/7` `MeshTrace` shapes `parity_cos == 1.0000000` at `0.72 ms` `M = 1`, `2.18 ms` `M = 6`, `3.47 ms` `M = 8`, `4.59 ms` `M = 16`), cross-wired via **`4× 800G` QSFP-DD cables** (`3.2 Tbps` `FABRIC_1D` TP2) with **Card `#0` `PCIe x16 -> FABRIC_1D` relay (`--p150-x16-relay`)** bypassing Card `#1`'s narrow `PCIe x4` host riser (`547.69 tok/s` fresh `225K` multimodal prefill, `+21.8%`).
2. **`3× AMD Instinct MI50` (Warm-Hot `216`-Expert MoE, Dense Backbone, `DeepSeek-ViT`, `1M` Tiered `CSA2`, & `Dynamic DeepSelect8` Tier):** `96 GiB` HBM2 (`3.07 TB/s` aggregate, `Vega20 gfx906` TP3, **`99/99` runtime + ViT + `3/3` `verify8` + `1/1` `verify8_4` HIP kernels at `ScratchSize == 0`**, pre-`dlopen` `GPU_MAX_HW_QUEUES=8`, `0` PCIe BAR P2P reads via coherent pinned-`DDR4` `FP32` `HostRingAllReduce`, logical-first `d_expert_perm[40 * 384]` router remap [`445` cross-tier swaps, `cos = 1.000000000`], `55`-VGPR `v_dot2_f32_f16` `k_indexer_score_msplit<8>`, 10-split register-Q `CSA2` `shared-tree8`, `253.45 MiB` streamed `BF16` `DeepSeek-ViT` on `tp = 2`, and `Dynamic DeepSelect8` backed by `110` `RAREBAL` `mtp.0..2` drafter experts).
3. **`AMD EPYC` + Dual NVMe (`Engram`, `0.62 s` Cold NVMe `.kvblk` Persistence, `1M` Warm KV Ring, & SSE Tier):** `384 GiB` DDR4 hosting `188.83 GiB` pre-faulted native `FP8 E4M3` `Engram` tables (`220.74 GB/s` warm hydration in `0.92 s`), `114,496 B` `.kvblk` pure-block cold NVMe `CSA2` persistence (`0 B` disk SWA, **`237,568` multimodal tokens hydrated in `0.623 s` at `381,290 tok/s`**), `64`-block (`86.371 MiB`) RAM `EncoderStateBlock` ring, coherent mapped pinned-DDR4 warm `CSA2` KV (`1,048,576` ceiling; **`238,489` multimodal & `417,556` text verified live context**), and pure-C++ `/v1/chat/completions` + `/anthropic/v1/messages` SSE + POSIX search sidecar.

---

## Repository Structure (`p150/` Lead Hierarchy)

```text
├── p150/                  # 2x Custom TSMC 6nm Tenstorrent Blackhole P150A TT-Metal C++20 FABRIC_1D TP2 engine
│   ├── kernels/           # 13 Tensix compute/NoC kernels (O(1) l1_slot_lut, x16->FABRIC_1D relay, HiFi2 -> FP32 Dst)
│   ├── dsv41_p150_l1.hpp  # Tier-2 FABRIC_1D per-bank L1 budget (18/18 hot shards = 161.37 MiB + +11,264 B margin)
│   ├── dsv41_p150_shm.hpp # Lock-free 4-slot 25,680 KiB /dev/shm fast ring + M=1..1024 mailbox
│   ├── p150_dsv41_engine.cpp
│   └── p150_dsv41_worker.cpp
├── src/                   # WHT128-HLUT2 (2.125 bpw), TARQ RAREBAL-LDLQ, TP3<->TP2 repacker, & safetensors mmap
│   ├── hlut2_pack.hpp
│   ├── dsv41_pack_format.hpp
│   ├── dsv41_rarebal.hpp
│   ├── dsv41_wave8_4_repack.hpp
│   └── safetensors_mmap.hpp
├── mi50/                  # 3x MI50 (gfx906 TP3) 99/99 zero-scratch HIP kernels, DeepSeek-ViT, Cold NVMe KV, & SSE
│   ├── dsv41_mi50_kernels.hip              # v_dot2_f32_f16 HSI + 10-split CSA2 shared-tree8 + logical-first perm
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
├── tools/                 # Pure-C++17 packers, 384-expert minimax balancer, 3x MI50 FP64 RAREBAL, & self-tests
│   ├── build_wave8_4_expert_perm.cpp       # Wave 8.4: Cross-tier 384-expert minimax balancer + 778-expert RAREBAL
│   ├── harvest_rarebal_wave8.cpp
│   ├── verify_wave8_4.cpp                  # Wave 8.4: TP3<->TP2 bit-exactness, logical-first router, L1 & CSA2 gates
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
├── scripts/               # Parameterized production launchers (GPU_MAX_HW_QUEUES=8, --p150-x16-relay, CSA2 tree8)
│   ├── run_p150_worker.sh
│   └── run_dsv41_fe.sh
├── docs/
│   ├── WHT128_HLUT2_Blackhole_P150A_Whitepaper.pdf
│   └── WHT128_HLUT2_ENGINEERING_REPORT.md
└── manifests/
    ├── wave8_4_receipt_manifest.txt
    ├── WAVE8_4_RELEASE_RECEIPT.md
    ├── rarebal_wave8_receipt_manifest.txt
    ├── CANONICAL_RELEASE.json
    └── SHA256SUMS.txt
```

---

## Build, Self-Test, Reassemble `778`-Expert Balanced Overlay, & Launch (`1M` Multimodal Context)

```bash
# 1. Build offline tools, self-tests, 3x MI50 + ViT runtime/server, and (when TT_METAL_HOME is set) 2x P150A worker
export DSV41_ROOT=/path/to/dsv41-model-workspace
export TT_METAL_HOME=/opt/tt-metal   # Tenstorrent Blackhole TT-Metal SDK install
make -j12

# 2. Run pure host-CPU mathematical, Tier-2 L1 geometry, TP3<->TP2 bit-exactness, & 99/99 zero-scratch gfx906 audits
make check
make audit-gfx906
./bin/verify_wave8_4 --link-check
# Optional hardware-on-device PCIe coherency, ViT parity, Cold NVMe .kvblk, & 1M CSA2 GPU suite (when GPUs are idle):
# make check-gpu

# 3. Download the 7-part Wave 8.4 384-expert permuted + 778-expert RAREBAL-LDLQ overlay (< 1.9 GiB/chunk, 11.73 GiB total)
gh release download v1.4.0-wave8.4 -R bespokeontology/dsv41-wht128-hlut2 \
  -p 'hot_experts_wave8_4_perm.safetensors.part-*' -p 'SHA256SUMS.txt' --clobber
cat hot_experts_wave8_4_perm.safetensors.part-* > hot_experts_wave8_4_perm.safetensors && sha256sum -c SHA256SUMS.txt
# Full overlay SHA-256: fb896094d895cc9bf37f85be8b2c1efb4db472d5ddf6d613e1cd5a382a2a2496

# 4. Pack contiguous BF16 DeepSeek-ViT weights once (lossless BF16, 0 Python)
./bin/pack_dsv41_vit --packed-mi50-dir "$DSV41_ROOT/packed_mi50" --out-dir "$DSV41_ROOT/packed_vit"

# 5. Launch 2x Blackhole P150A TP2 worker (18/18 L1 shards + x16->FABRIC_1D relay), search sidecar, and 1M multimodal server
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-8}"

./bin/p150_dsv41_worker --model-dir "$DSV41_ROOT" \
  --rarebal-overlay hot_experts_wave8_4_perm.safetensors \
  --p150-l1-shards 18 --p150-x16-relay --reuse-shm &

./bin/dsv41_search_sidecar --host 127.0.0.1 --port "${DSV41_SEARCH_PORT:-8080}" &

OMP_WAIT_POLICY=PASSIVE ./bin/dsv41_fe --model-dir "$DSV41_ROOT" \
  --rarebal-overlay hot_experts_wave8_4_perm.safetensors \
  --expert-perm-overlay hot_experts_wave8_4_perm.safetensors \
  --vit-weights "$DSV41_ROOT/packed_vit/dsv41_vit_bf16.safetensors" \
  --nvme-kv-cache-dir "$DSV41_ROOT/nvme_kv_cache" --nvme-kv-cache-gb 32 \
  --search-url "${SEARCH_URL:-http://127.0.0.1:8080}" \
  --port "${DSV41_PORT:-8000}" --max-seq-len 1048576 \
  --dual-stream-overlap --csa2-shared-tree8 --dspark-tree-width auto
```

---

## Attribution & License

* **Authors & Maintainer:** [**`bespokeontology`**](https://github.com/bespokeontology) — Heterogeneous Silicon & Quantization Co-Design Engineering
* **Base Multimodal Model (`763B`-Class):** [**`deepseek-ai/DeepSeek-V4.1-Flash-1017`**](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
* **License:** Released under the [MIT License](LICENSE) (`Copyright (c) 2026 bespokeontology`).
