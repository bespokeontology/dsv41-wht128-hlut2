# `WHT128-HLUT2` + `561`-Expert `TARQ RAREBAL-LDLQ` (`2.125 bpw → FP32`) on `2× Custom TSMC 6nm Tenstorrent Blackhole P150A` ASICs: `1M`-Context Multimodal C++ Engine for 763B `DeepSeek-V4.1-Flash-1017` (`v1.3.0-wave8.3`)

**Co-Engineered by [**`bespokeontology`**](https://github.com/bespokeontology), [**Google Antigravity (Google DeepMind)**](https://antigravity.google/), & [**Gemini (Google DeepMind)**](https://deepmind.google/technologies/gemini/)** · **100% Pure TT-Metal C++20 / `gfx906` HIP / Host C++17 (`0` Python Files, `-Werror`)**  
📄 **[4-Page Illustrated Whitepaper (PDF)](docs/WHT128_HLUT2_Blackhole_P150A_Whitepaper.pdf)** · 📘 **[Full `v1.3.0-wave8.3` Engineering Report (`docs/WHT128_HLUT2_ENGINEERING_REPORT.md`)](docs/WHT128_HLUT2_ENGINEERING_REPORT.md)** · 🚀 **[`v1.3.0-wave8.3` Release](https://github.com/bespokeontology/dsv41-wht128-hlut2/releases/tag/v1.3.0-wave8.3)** · 📦 **[`561`-Expert `RAREBAL` Shards (`v1.2.0-wave8`)](https://github.com/bespokeontology/dsv41-wht128-hlut2/releases/tag/v1.2.0-wave8)**

---

## `v1.3.0-wave8.3` Release Highlights (Wave 8.3 Canonical: Vision + `0.72 s` Cold NVMe `CSA2` KV + `Dynamic DeepSelect8`)

* **Wave 8.1 — Native Pure-C++/HIP `DeepSeek-ViT` Vision (`Dsv41VitEngine`, `253.45 MiB` HBM2 on `tp = 2`):** Unlocks the full **`763B`-class multimodal [`DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** (`748.6B` text + `DSpark` MTP + `32`-block `BF16` vision encoder & projector) in pure C++/HIP with zero Python dependencies. Streams double-buffered ViT layer weights from pinned DDR4 onto `MI50` `tp = 2` (`0 B` added on `tp = 0, 1`, `0` changes to `p150_dsv41_worker`), running `2D-RoPE`, `3×3` pixel-unshuffle, the 2-layer MLP projector, image router bias (`ffn.gate.bias_vl`), and [`Engram`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/engram.py) image-token masking while preserving **`268.0 / 270.0 / 698.0 MiB` free HBM2** across `tp = 0..2` at `max_seq_len = 1,048,576` (`96/96` runtime + ViT `gfx906` kernels at `ScratchSize == 0`).
* **Wave 8.2 — [`DeepInfer`](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md) Pure-Block Cold NVMe `CSA2` KV (`0 B` Disk SWA), `PromptSegment` Anti-Injection & `-81%` Lazy Softmax:** Persists `128`-token `.kvblk` records (`114,496 B` = `64 B` `Dsv41NvmeKvBlockHeader` + `512 B` token IDs + `113,920 B` [`CSA2`](https://zartbot.github.io/blog/model_arch/dsv41flash_arch/en.html) KV; `0` `.kvtail`/`.swa` files on disk) paired with a `64`-block (`86.371 MiB`) RAM `EncoderStateBlock` ring and `128`-token bounded `Encoder SWA` cold replay (`L_cap = P - 140`, snapping outside partial `<｜deepseek_image｜>` spans; `enc_cos = 1.00000000`, `dec_cos = 1.00000000`) — hydrating **`220,544`–`224,384` multimodal tokens across cold daemon restarts in `0.632–0.722 s`** (`109.2–226.3 ms` raw NVMe read up to **`1.71 GB/s`**, **`~630–720×` faster** than `~7.6 min` cold prefill). Enforces C++ type-level anti-injection via `PromptSegment` (`parse_special = false` ordinary text + integer-ID-gated `128822` `</think>` / `128825` `｜DSML｜` `DsmlOpenKmpSpec` streaming, `0` leaks on `18,476 B` hostile C++ payloads) and slashes host sampling `smp` **`-81%` (`7.04 → 1.17–1.37 ms/step`)** via lazy visited-path tree softmax (`1.06/16` nodes) + async CPU post-processing.
* **Wave 8.3 — Single-Pass `ConfidenceHead` `Dynamic DeepSelect8` (`M = 8` SGPR Causal Tree Across `0–1M` Context):** `deep_select_top7_confidence_tree8` (`Dsv41DeepSelectPlan8`) scores a `16`-node (`depths 0..5`) single-pass [`DSpark`](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/) candidate pool (`4` MTP micro-batches `B = 3, 2, 1, 1` = `7` rows, `+0.34 ms` drafter cost, `0 B` extra HBM2) using `confidence_head.proj.weight` (`[1, 5,376]`) and compiles dynamic `64`-bit SGPR `anc_mask64`, `depth_nibbles32`, `parent_nibbles32`, and `valid_node_mask` (`0` invalid filler acceptances across `100,000` shortage sweeps, **`max_TV_M8 = 2.463e-16 < 1e-14`**). Delivers **`40.03 tok/s`** (`2.48×` vs. `4`-batch `M = 16`'s `16.17 tok/s`) at `8.7K` Effort-100 context, beats static SGPR `M = 8` head-to-head (`35.22` vs. `33.23 tok/s`, **`+6.0%`**; **`+26.3%`** accepted tree drafts/step) across `25K` generated tokens, and reaches **`32.85 tok/s`** (`+13.4%` over static `M = 8`, **`2.50×`** over `M = 16`, `84.2%` acceptance, `0.28 ms` `814.6M tok/s` GPU slot hit) at **`228,225` multimodal tokens** (`232,604` peak verified multimodal context).
* **Overlay Shards Compatibility (`0` Re-Download Required):** `v1.3.0-wave8.3` uses the identical `561`-expert `hot_experts_rarebal_wave8.safetensors` (`4.91 GiB`, `SHA-256 bf78446e551207e5d2a9b8061b1299df67359e2a565764b6e092ee26ea47b237`) published in three `< 2 GiB` chunks on [`v1.2.0-wave8`](https://github.com/bespokeontology/dsv41-wht128-hlut2/releases/tag/v1.2.0-wave8) (`cat hot_experts_rarebal_wave8.safetensors.part-* > hot_experts_rarebal_wave8.safetensors`).

---

## Heterogeneous Silicon Architecture (`2× Blackhole P150A` + `3× MI50` + `EPYC` Dual-NVMe)

This repository contains the clean-room, zero-Python production multimodal inference engine and custom **`WHT128-HLUT2` + `561`-Expert `TARQ RAREBAL-LDLQ` (`2.125 bpw → IEEE FP32`)** quantization stack for **[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** up to **`1,048,576` (`1M`) active context tokens**:

1. **[`2× Tenstorrent Blackhole P150A`](https://docs.tenstorrent.com/aibs/blackhole/index.html) Custom TSMC `6nm` (`N6`) Spatial Dataflow ASICs (Lead Sparse MoE Tier):** `220` compute Tensix cores (`2 × 110`, each with `5` RISC-V baby cores + `32×32` `HiFi2`/`FP32` Matrix Engine + `32`-lane Vector `SFPU` + dual `NoC` routers), `64 GiB` GDDR6 (`MP`), and **`2× 180 MB = 360 MB` on-die Tensix L1 SRAM** (**`247.82 MiB` pure active L1** [`123.91 MiB/card = 129,927,808 B/card`], **`34/34` `RAREBAL`-calibrated TP2 hot shards = `152.40 MiB`** pinned in L1 alongside `100%` L1-resident `x_decode` and `finish_decode[0..1]`, `7/7` `MeshTrace` shapes at `0.73 ms` `M = 1`, `2.52 ms` `M = 8`, `4.53 ms` `M = 16` with `0 B` extra L1), cross-wired card-to-card via **`4× 800G` QSFP-DD cables** (`3.2 Tbps` `FABRIC_1D` TP2), with **Card `#1` running behind a narrow `PCIe x4` host riser**.
2. **`3× AMD Instinct MI50` (Dense Backbone, `DeepSeek-ViT`, `1M` Tiered `CSA2`, & `Dynamic DeepSelect8` Tier):** `96 GiB` HBM2 (`Vega20 gfx906` TP3, **`96/96` runtime + ViT + `3/3` `verify8` HIP kernels at `ScratchSize == 0`**, `0` PCIe BAR P2P reads via coherent pinned-`DDR4` `FP32` `HostRingAllReduce`, pre-`shared_experts` dual-stream `4`-slot `25,680 KiB` `/dev/shm` fast ring saving `18.57 ms/step` parallel overlap, `253.45 MiB` streamed `BF16` `DeepSeek-ViT` on `tp = 2`, and single-pass `ConfidenceHead` `Dynamic DeepSelect8` SGPR causal tree).
3. **`AMD EPYC` + Dual NVMe (`Engram`, `0.72 s` Cold NVMe `.kvblk` Persistence, `1M` Warm KV Ring, & SSE Tier):** `384 GiB` DDR4 hosting `188.83 GiB` pre-faulted native `FP8 E4M3` `Engram` tables (`220.74 GB/s` warm hydration in `0.92 s`), `114,496 B` `.kvblk` pure-block cold NVMe `CSA2` persistence (`0 B` disk SWA, `224,384` multimodal tokens hydrated in `0.722 s` at up to `1.71 GB/s`), `64`-block (`86.371 MiB`) RAM `EncoderStateBlock` ring, coherent mapped pinned-DDR4 warm `CSA2` KV (`1,048,576` ceiling; **`232,604` multimodal & `417,556` text verified live context**), and pure-C++ `/v1/chat/completions` + `/anthropic/v1/messages` SSE + POSIX search sidecar.

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
├── mi50/                  # 3x MI50 (gfx906 TP3) 96/96 zero-scratch HIP kernels, DeepSeek-ViT, Cold NVMe KV, & SSE
│   ├── dsv41_mi50_kernels.hip
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
├── tools/                 # Pure-C++17 packers, 3x MI50 FP64 RAREBAL repacker, search sidecar, & Wave 5–8.3 self-tests
│   ├── pack_dsv41_hlut2.cpp
│   ├── pack_dsv41_vit.cpp
│   ├── harvest_rarebal_wave8.cpp
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

## Build, Self-Test, Reassemble `561`-Expert Overlay, & Launch (`1M` Multimodal Context)

```bash
# 1. Build offline tools, self-tests, 3x MI50 + ViT runtime/server, and (when TT_METAL_HOME is set) 2x P150A worker
export DSV41_ROOT=/path/to/dsv41-model-workspace
export TT_METAL_HOME=/opt/tt-metal   # Tenstorrent Blackhole TT-Metal SDK install
make -j12

# 2. Run pure host-CPU mathematical, L1 geometry, Dynamic DeepSelect8, PromptSegment anti-injection, & 96/96 zero-scratch audits
make check
make audit-gfx906
# Optional hardware-on-device PCIe coherency, ViT parity, Cold NVMe .kvblk, & 1M CSA2 GPU suite (when GPUs are idle):
# make check-gpu

# 3. Download the immutable 3-part 561-expert RAREBAL-LDLQ overlay from v1.2.0-wave8 and reassemble (4.91 GiB)
gh release download v1.2.0-wave8 -R bespokeontology/dsv41-wht128-hlut2 \
  -p 'hot_experts_rarebal_wave8.safetensors.part-*' --clobber
cat hot_experts_rarebal_wave8.safetensors.part-* > hot_experts_rarebal_wave8.safetensors
echo "bf78446e551207e5d2a9b8061b1299df67359e2a565764b6e092ee26ea47b237  hot_experts_rarebal_wave8.safetensors" | sha256sum -c -

# 4. Pack contiguous BF16 DeepSeek-ViT weights once (lossless BF16, 0 Python)
./bin/pack_dsv41_vit --packed-mi50-dir "$DSV41_ROOT/packed_mi50" --out-dir "$DSV41_ROOT/packed_vit"

# 5. Launch 2x Blackhole P150A TP2 worker, optional search sidecar, and 1M-context multimodal server (Dynamic DeepSelect8 default)
./bin/p150_dsv41_worker --model-dir "$DSV41_ROOT" \
  --rarebal-overlay hot_experts_rarebal_wave8.safetensors --reuse-shm &

./bin/dsv41_search_sidecar --host 127.0.0.1 --port "${DSV41_SEARCH_PORT:-8080}" &

OMP_WAIT_POLICY=PASSIVE ./bin/dsv41_fe --model-dir "$DSV41_ROOT" \
  --rarebal-overlay hot_experts_rarebal_wave8.safetensors \
  --vit-weights "$DSV41_ROOT/packed_vit/dsv41_vit_bf16.safetensors" \
  --nvme-kv-cache-dir "$DSV41_ROOT/nvme_kv_cache" --nvme-kv-cache-gb 32 \
  --search-url "${SEARCH_URL:-http://127.0.0.1:8080}" \
  --port "${DSV41_PORT:-8000}" --max-seq-len 1048576 \
  --dual-stream-overlap --dspark-tree-width auto
```

---

## Credits & Attribution

* **Hardware Architecture, Systems Direction, & Heterogeneous Co-Design:** [**`bespokeontology`**](https://github.com/bespokeontology)
* **Agentic C++ Systems Engineering, Waves 7B–8.3 Silicon Implementation (`DeepSeek-ViT`, `0.72 s` Pure-Block Cold NVMe `CSA2` KV, `PromptSegment` Isolation, `Dynamic DeepSelect8`, `561`-Expert `FP64` `RAREBAL-LDLQ`), & Verification:** [**Google Antigravity (Google DeepMind)**](https://antigravity.google/)
* **Kernel Co-Design, Quantization & Measure-Theoretic Synthesis (`WHT128-HLUT2`, `TARQ RAREBAL-LDLQ`, `DeepInfer` Pure-Block Replay, Kolmogorov–Dirac `Dynamic DeepSelect8`), & Whitepaper Synthesis:** [**Gemini (Google DeepMind)**](https://deepmind.google/technologies/gemini/)
* **Self-Hosted Long-Horizon Multimodal & Code Audits (`232.6K` Multimodal & `417.6K` Text Live Context):** [**`DeepSeek-V4.1-Flash-1017`**](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)

## License

Released under the [MIT License](LICENSE) (`Copyright (c) 2026 bespokeontology`).
