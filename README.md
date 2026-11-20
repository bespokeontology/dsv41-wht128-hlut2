# `WHT128-HLUT3` (`3.125 bpw`, `1,704` `TARQ RAREBAL-LDLQ` Experts) + `WHT128-HLUT2` on `4× MI50` + `2× Blackhole P150A` (`176 GiB` Switchbox): `1M`-Context Multimodal Pure-C++ Engine for 763B `DeepSeek-V4.1-Flash-1017` (`v2.0.0-wave9.4-hlut3`)

**Authors:** [**`bespokeontology`**](https://github.com/bespokeontology) — Heterogeneous Silicon & Quantization Co-Design Engineering · **100% Pure [`TT-Metal`](https://github.com/tenstorrent/tt-metal) C++20 / `gfx906` HIP / Host C++17 (`0` Python Files, `-Wall -Wextra -Werror`, `164/164` Zero-Scratch `gfx906` Kernels)**  
🚀 **[Wave 9.4 Ultra Engineering Report (`docs/WAVE9_4_ULTRA_PERF_REPORT.md`)](docs/WAVE9_4_ULTRA_PERF_REPORT.md)** · 🧾 **[Wave 9.4 `1M` E2E Silicon Receipt (`docs/WAVE9_4_1M_E2E_RECEIPT.md`)](docs/WAVE9_4_1M_E2E_RECEIPT.md)** · ⚙️ **[Wave 9.2 `176G` Switchbox Report (`docs/WAVE9_2_176G_SWITCHBOX_REPORT.md`)](docs/WAVE9_2_176G_SWITCHBOX_REPORT.md)** · ⚡ **[Wave 9.3 Perf Report (`docs/WAVE9_3_OVERNIGHT_PERF_REPORT.md`)](docs/WAVE9_3_OVERNIGHT_PERF_REPORT.md)** · 📄 **[Illustrated Whitepaper PDF (`docs/WHT128_HLUT2_Blackhole_P150A_Whitepaper.pdf`)](docs/WHT128_HLUT2_Blackhole_P150A_Whitepaper.pdf)** · 🏷️ **[`v1.5.0-wave8.5` Baseline](https://github.com/bespokeontology/dsv41-wht128-hlut2/releases/tag/v1.5.0-wave8.5)**

---

## `v2.0.0-wave9.4-hlut3` Release Highlights (Waves 9.2 → 9.4 Ultra)

* **`WHT128-HLUT3` (`3.125 bpw`, `400 B` `32×32` Tile, `+5.3 to +6.1 dB` SNR over `HLUT2`) on `SB-M4-P2-176G` (`4× MI50` + `2× Blackhole P150A` = `176 GiB`):** Promotes **`1,704`** hottest experts calibrated via `FP64` [`TARQ RAREBAL-LDLQ`](https://arxiv.org/pdf/2605.27808) (`602K`-token multi-domain + `DeepSeek-ViT` corpus) into exact 8-level symmetric Gaussian Lloyd-Max `WHT128-HLUT3` (`c = {0.24509418, 0.75600528, 1.34390928, 2.15194570}`, `+5.32 dB` theoretical / `+5.3 to +6.1 dB` empirical SNR over `HLUT2`):
  - **Tier-0 Asymmetric `dev3` MTP Offload (`384/384` `DSpark` drafters = `100%` coverage):** Moves all `384` MTP routed experts unsharded onto a dedicated 4th `16 GiB` MI50 (`dev3`, e.g. `0000:85:00.0`), reclaiming `+1,204 MiB/rank` on every `32 GiB` TP3 card and eliminating TP3 all-reduce on MTP MoE.
  - **Tier-1 Parallel `dev3` Backbone Offload (`720` hottest MI50 backbone experts):** Runs concurrently with TP3 + P150A MoE over a per-128-block **`INT8 + FP16-scale` H2D** and **`BF16` D2H** wire (`cos = 0.99995590`, `0 B` extra HBM2).
  - **Tier-2 In-Place TP3 Backbone Promotion (`600` sharded TP3 backbone experts):** Packed cleanly into freed TP3 HBM2 alongside `7,320` compacted `HLUT2` MI50 experts and `6,720` `P150A` `HLUT2` experts (`[216..383]` invariant).
* **Zero-PCIe-AtomicOp Broadcom `PEX88048` Switchbox Stability:** Eliminates `100%` of GPU-side `__HIP_MEMORY_SCOPE_SYSTEM` RMW atomics / `slc` atomics targeting pinned host DDR4. Every `HostRingAllReduce` CTA barrier (`d_write_cta`/`d_read_cta`) runs exclusively in each GPU's local HBM2 with `__HIP_MEMORY_SCOPE_AGENT`, delivering **`0` `ERREVENT_ATHUB_INTERRUPT` / `0` `CmpltAbrt` / `0` AER errors** across two MI50s hosted behind a non-AtomicOpsRouting Broadcom `PEX88048` PCIe switchbox throughout `544K`-token live sessions and full `1M` E2E runs.
* **Wave 9.4 Ultra Throughput Stack (`P150A` + `MI50` + `CSA2` + Prefix-Preserving Effort):**
  - **`2× Blackhole P150A` (`2.15×` device / `74×` host-wait speedup at `M=4096`):** `DSV41_P150_MAX_ROWS=4096` (`SHM_VERSION=4`), `kBatchSlots=24`, Expert-Grouped Contiguous Best-Fit Batch Packer, and fixed `op_traces` MeshTrace cache eviction (`9/9` silicon self-checks `parity_cos = 1.0000000`, `1,241.7 ms` P150A device time / `7.6 ms` wait at `M=4096`).
  - **`4× MI50` `HHHS` `B=8` Strided-Batched `rocBLAS` & `128`-Aligned CED Chunking:** `FP16`-out / `FP32` `v_dot2_f32_f16` `HHHS` dispatch (`rocblas_expert_thresh=18`, `rocblas_hlut2_thresh=20`), `kMaxChunkM=4096`, `128`-aligned single-chunk tail collapse, cutting `COLD M=4096` wall latency **`5,882.9 → 4,604.1 ms` (`889.6 tok/s`, peak `913.9 tok/s` live cold GUI prefill)**.
  - **Sub-Second `340K` Interactive Tool-Tail & `2.17×` Faster `340K` Decode Attention:** Compact `mode == 2` candidate scoring (`21.25×` less compute at `340K`, `64×` at `1M`), 60-CU Split-KV exact block-max + radix select, and TP0-only FP16-wire reduction (`6×` less PCIe traffic) with **100.000% bit-exact parity (`0/4096` diffs)** — cutting **`M=256 @ 340K` from `1,731.2 → 991.0 ms` (`1.75×`, `258.3 tok/s`)** and **`340K` decode attention from `78.53 → 36.23 ms` (`2.17×`, `HSI = 0.24–0.31 ms/step`)**.
  - **Prefix-Preserving `reasoningEffort` Switching:** Fixed `prefix_base_effort` header (`index == 0`) paired with active-turn `<｜latest_reminder｜>` (`128828`) overrides preserves **`100.00%` (`Medium → Low` @ `266K`)** and **`99.73%` (`Low → Medium` @ `543K`)** KV prefix reuse with **zero cold re-prefill** on both OpenAI `/v1/chat/completions` and Anthropic `/anthropic/v1/messages`.

---

## Verified Wave 9.4 Ultra Benchmarks (`SB-M4-P2-176G`, Stock `225 W` MI50 Power Cap)

### 1. Prefill & Interactive Tool-Tail Comparison (Wave 9.3 vs. Wave 9.4 Ultra)

| Scenario | Cached KV | Fresh `M` | Wave 9.3 Wall (`ms` / `tok/s`) | Wave 9.4 Ultra Wall (`ms` / `tok/s`) | Speedup | Key Stage Reduction |
| :--- | ---: | ---: | :---: | :---: | :---: | :--- |
| **Tool-Tail `M=256 @ 340K`** | `348,160` | `256` | `1,731.2 ms` / `147.9 tok/s` | **`991.0 ms` / `258.3 tok/s`** | **`1.75×`** | HSI `725.1 → 174.3 ms` (`4.16×`), `dev3` wait `205.2 → 34.4 ms` |
| **Cold Prefill `M=2048`** | `0` | `2,048` | `3,358.8 ms` / `609.7 tok/s` | **`2,744.6 ms` / `746.2 tok/s`** | **`1.22×`** | P150A dev `1,704.0 → 846.4 ms`, wait `363.2 → 5.2 ms` (`70×`) |
| **Cold Prefill `M=4096`** | `0` | `4,096` | `5,882.9 ms` / `696.3 tok/s` | **`4,604.1 ms` / `889.6 tok/s`** | **`1.28×`** | P150A `2,665.8 → 1,241.7 ms` (`2.15×`), `dev3` `2,306.5 → 1,207.5 ms` |
| **Live Cold GUI Prefill** | `0` | `8,981` | `~700 tok/s` | **`9.83 s` / `913.9 tok/s`** | **`1.31×`** | Full live SSE prompt ingestion |

### 2. Decode Scaling (`T_step` / `Attn` Across `8K → 1M`) & Live `544K` GUI Stopwatch Telemetry

Both **honest end-to-end wall-clock throughput** (`generated_tokens / generation_elapsed_s` over **`135,548` real generated tokens** across **`108` LLM steps / `152` tool calls** in the live `:3081` GUI) and **internal step latency (`T_step`, `Attn/step`, `tau`)** are reported side-by-side:

| Regime / Context Band | Measurement Source | True Wall-Clock Avg (`tok/s`) | Peak (`tok/s`) | `T_step` (`ms`) | `Attn/step` (`ms`) | Avg `tau` | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`8K` Synthetic Burst** | `profile_wave9_4_breakdown` | **`52.15 tok/s`** | `52.15` | `102.27 ms` | `27.87 ms` | `5.25` | `94.4%` DSpark acc |
| **`<= 160K` Live GUI** (`12.8K` gen) | `:3081` GUI (`15` reqs) | **`31.69 tok/s`** | **`54.46 tok/s`** (`11K`) | `108.54 ms` | `30.84 ms` (`64K`) | `3.44` | Peak cold ingest **`913.9 tok/s`** |
| **`160K–350K` Live GUI** (`77.5K` gen) | `:3081` GUI (`26` reqs) | **`35.86 tok/s`** | **`56.00 tok/s`** (`230K`) | `113.43 ms` | **`36.23 ms`** (`340K`, **`2.17×`**) | `4.07` | Includes **`29,546`-tok** single-shot C++ gen @ **`44.74 tok/s`** & **`54.14 tok/s`** @ `319K` |
| **`350K–544K` Live GUI** (`45.2K` gen) | `:3081` GUI (`67` reqs) | **`31.95 tok/s`** | **`52.37 tok/s`** (`408K`) | `121.05 ms` | `~38 ms` | `3.87` | **`51.37 tok/s`** @ `487K`, **`50.26 tok/s`** @ `533K`, `HSI = 0.25 ms/step` |
| **Cold NVMe `.kvblk` + Live `DeepSeek-ViT`** | `:3081` Turn 4 (`544,210` ctx) | **`37.73 tok/s`** | `37.73` | `114.4 ms` | `37.7 ms` | `4.32` | **`543,744` tok reloaded cold from NVMe in `366.15 ms` (`1.49M tok/s` raw / `384K tok/s` full HBM2+SWA)** + `516.9 ms` ViT (`4/4` chart bars read exact!) |
| **`1,048,576` (`1M`) E2E Silicon Gate** | `verify_wave8_5_1m_e2e` | **PASS** | `~41.5` (`tau=5.4`) | **`131.0 ms`** | **`51.0 ms`** | — | Free HBM2: **`1,738 / 1,740 / 2,164 MiB`** + `dev3` **`1,134 MiB`**; `100.000%` HSI recall; `3/3` needles HIT; `CSA2 cos=1.00000000`; `CRC32C=PASS` |

---

## Download, Reassemble, Verify, Build & Launch (`v2.0.0-wave9.4-hlut3`)

### 1. Download & Reassemble Sanitized `WHT128-HLUT3` (`1,704`-Expert) Release Overlays

```bash
export DSV41_WEIGHTS_DIR=/path/to/dsv41-weights
export DSV41_OVERLAY_DIR="$DSV41_WEIGHTS_DIR/wave9_2_overlays"
mkdir -p "$DSV41_OVERLAY_DIR" && cd "$DSV41_OVERLAY_DIR"

# Download all release assets for v2.0.0-wave9.4-hlut3
gh release download v2.0.0-wave9.4-hlut3 --repo bespokeontology/dsv41-wht128-hlut2

# Reassemble <1.90 GiB split chunks into canonical safetensors overlays
cat wave9_2_dev3_hlut3.safetensors.part-* > wave9_2_dev3_hlut3.safetensors
cat wave9_2_tp3_hlut3.safetensors.part-*  > wave9_2_tp3_hlut3.safetensors

# Verify cryptographic SHA-256 integrity of every split part, reassembled overlay, and manifest
sha256sum -c SHA256SUMS.txt
```

### 2. Clean Pure-C++ Build & Verification (`164/164` Zero-Scratch `gfx906`, `0` Python, `0` PII)

```bash
export DSV41_ROOT="$(pwd)"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export TT_METAL_HOME="${TT_METAL_HOME:-/opt/tt-metal}"

# Build MI50 runtime/server, all pure-C++17 tools, and (when TT_METAL_HOME is present) 2x P150A worker
make -j12

# Run complete host verification, 164/164 zero-scratch gfx906 ISA audit, engine invariant audit, & zero-PII self-tests
make -j12 check
./bin/audit_zero_pii --self-test
./bin/audit_zero_pii --root .
```

### 3. Launch `SB-M4-P2-176G` (`4× MI50` + `2× Blackhole P150A`) on `:8042`

```bash
export DSV41_ROOT="$(pwd)"
export DSV41_WEIGHTS_DIR=/path/to/dsv41-weights
export DSV41_OVERLAY_DIR="$DSV41_WEIGHTS_DIR/wave9_2_overlays"
export DSV41_KV_CACHE_DIR="$DSV41_WEIGHTS_DIR/nvme_kv_cache"
export DSV41_PORT=8042
export DSV41_DEFAULT_EFFORT=65       # Prefix-preserving default effort (65 = Medium)
export GPU_MAX_HW_QUEUES=8

# 1. Start 2x Blackhole P150A TP2 worker (40/40 layers, kBatchSlots=24, x16->FABRIC_1D relay)
./scripts/run_p150_worker.sh &

# 2. Start optional pure-C++ POSIX search sidecar
./bin/dsv41_search_sidecar --host 127.0.0.1 --port "${DSV41_SEARCH_PORT:-8888}" &

# 3. Start 4x MI50 + DeepSeek-ViT + 1M NVMe CSA2 KV + OpenAI/Anthropic SSE server
./scripts/run_dsv41_fe.sh
```

---

## Attribution & License

* **Authors & Maintainer:** [**`bespokeontology`**](https://github.com/bespokeontology) — Heterogeneous Silicon & Quantization Co-Design Engineering
* **Base Multimodal Model (`763B`-Class):** [**`deepseek-ai/DeepSeek-V4.1-Flash-1017`**](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
* **License:** Released under the [MIT License](LICENSE) (`Copyright (c) 2026 bespokeontology`).
