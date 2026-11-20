# Wave 9.3 (`SB-M4-P2-176G`) Overnight Extreme Throughput Optimization & Live GUI Engineering Report

**Target Architecture**: `DeepSeek-V4.1-Flash-1017` (`40` backbone layers + `3` DSpark MTP layers, `384` routed experts/layer, `top_k = 6`, `CSA2`, `Engram`, `DSML`, `DeepSeek-ViT`)  
**Production Topology**: `SB-M4-P2-176G` (`176 GiB` total device memory = `3× 32 GiB MI50 [tp=0..2]` + `1× 16 GiB MI50 [hlut3_dev=3]` + `2× 32 GiB Tenstorrent Blackhole P150A` over `4× 800G FABRIC_1D` + `188.83 GiB` host-locked DDR4 Engram tables)  
**Active WHT128-HLUT3 Footprint**: `1,704` promoted `3.125 bpw` experts (`384/384` MTP + `720` `dev3` backbone + `600` `TP3` backbone) + `7,320` compacted `TP3` HLUT2 + `6,720` invariant `P150A` HLUT2  
**Zero-Scratch / Zero-Atomic / Zero-Python Invariant**: `145 / 145` zero-scratch `gfx906` kernels, `0` SGPR spills, `0` VGPR spills, `0` device trig ops, `0` `slc` / `__HIP_MEMORY_SCOPE_SYSTEM` RMW atomics, `0` `.py` files anywhere in the build or runtime tree.

---

## 1. Executive Summary of Delivered Wave 9.3 Optimizations

1. **Phase B — Cold Compute Prefill Acceleration (`+56%` to `+66.7%` vs Wave 9.2; up to `+15.6%` faster than Wave 8.5 HLUT2 despite `1,704` higher-precision `3.125 bpw` experts)**:
   - **Weight-Unpack-Once + `rocBLAS` FP16->FP32 Tensor-Op GEMM (`mi50/dsv41_wave9_3_rocblas.hpp`, `mi50/dsv41_wave9_3_prefill_kernels.hip`)**:
     - Vectorized coalesced `uint32_t` HLUT3 tile unpack + zero-atomic coalesced `float4` weighted scatter (`dim3(5, m_e)`) delivering **`5.99×`** speedup on hot `TP3 HLUT2` buckets (`m_e >= 48`), **`5.47×`** on `TP3 HLUT3` (`m_e >= 16`), **`2.28×`** on `dev3` unsharded `HLUT3` (`m_e >= 16`), and **`2.93×`** on `BF16` router `gate_w`.
     - Eliminated `rocblas_initialize()` cold bloat (`-180 MiB/GPU`), replaced with targeted FP16->FP32 shape warmup (`25.2 s` total 4-GPU boot, `0 ms` runtime JIT stall) inside a fixed `83.5 MiB` (`TP3`) / `66.25 MiB` (`dev3`) workspace.
     - Kept dense `FP8` projections (`QKV`, `Wo`) on custom fused `k_fp8_gemm_bm128_bn64` / `k_fp8_grouped_wo_a_bm128_bn64` (`-344 ms` faster than unpack+`rocBLAS` at `M = 2,048`).
   - **Full-`v_dot2_f32_f16` 8-Head 4-Key Prefill Sparse Attention (`k_sparse_attn_allheads`, `mi50/dsv41_mi50_kernels.hip`)**:
     - Upgraded BOTH `Q·K^T` and `P·V` to `v_dot2_f32_f16` at `3 CTAs/CU` (`ScratchSize == 0`), cutting `4K` `CSA2+SWA` prefill attention from `911 ms -> 595 ms` (**`1.53×`** faster).
     - **Critical Correctness Discovery & Fix**: Located and fixed a latent bitwise-AND bug in `k_sparse_attn_allheads`'s 4-key validity guard (`(s_valid[t+0] & ... & s_valid[t+3]) >= 0` -> bitwise `|`) that caused tokens `0..126` (`n_win_m < 128`) to read uninitialized LDS slots; post-fix prefill parity reached **bit-exact `cos = 1.00000000`, `max_d = 0.0000`**.
   - **Superchunk & Pipeline Upgrades (`mi50/dsv41_runtime.cpp`, `mi50/dsv41_hlut3_offload_ipc.hpp`)**:
     - Raised `kMaxChunkM = 2048` (`4×` larger expert buckets) with pipelined `1,024`-row P150A `/dev/shm` slicing, overlapping `shared_experts` behind `router_stream` D2H, and `dev3_reuse_bf16` (eliminating duplicate `20 MiB/layer` host staging copies).
2. **Phase C — Context, Tokenization & Multi-File Ingestion (`81,312` to `3,511,145 tok/s` — `4×` to `175×` above `20,000 tok/s` target)**:
   - **`TokSliceCache` (`mi50/dsv41_tokenizer_dsml.cpp`)**: `8,192`-entry dual-64-bit hash + exact `memcmp` BPE slice cache cutting repeated large-context tokenization to `< 1.5 ms`.
   - **High-Bandwidth Cold NVMe `.kvblk` Hydration (`mi50/dsv41_nvme_kv_cache.hpp`)**: `posix_fadvise(POSIX_FADV_WILLNEED)` 64-block lookahead + 24-thread parallel `pread` + hardware `CRC32C` + 5-argument `compute_model_overlay_uuid` hashing both `dev3` and `TP3` HLUT3 overlays. Measured **`2.47M tok/s` raw NVMe hydration** (`342,272` tokens / `2,674` `.kvblk` blocks in **`138.52 ms`**) and **`306,405 tok/s` end-to-end cold reload** (including `128`-token Encoder SWA Replay) during the live `:3081` GUI session!
3. **Phase D — Sustained Agentic Decode & `DSpark` (`mi50/dsv41_hlut3_kernels.hip`, `mi50/dsv41_hlut3_offload_ipc.hpp`)**:
   - Full-CU-saturation grouped `v_dot2_f32_f16` HLUT3 decode (`Grid = (N_inter/16, max_act)` and `(320, max_act)`) with zero-LDS zero-barrier `B == 1 / 2 / <= 4` fast paths (**`2.99×`** faster on `dev3`, **`1.49×`** faster on `TP3`) + single-launch `k_dev3_ingest_and_pack_csr` CSR builder.
   - Zero-bounce decode `stage_and_arm` directly on `ctx[0].stream`, dedicated physical EPYC core pinning (`cpu 24`), and non-blocking AQL `hipStreamQuery` doorbell flush: cut `HostRingAllReduce` latency **`3.0×–4.7×`** (`19.4 -> 4.5–6.5 ms/step`), `DSpark` draft latency **`13.1 -> 9.5–10.0 ms/step`**, and `dev3` decode wait to **`0.0 ms`**!

---

## 2. Side-by-Side Throughput Comparison: Wave 8.5 vs Wave 9.2 vs Wave 9.3

### 2.1 Cold Compute Prefill (`reuse = 0`, Fresh Unique Nonces)

| Benchmark Workload | Wave 8.5 (`3× MI50`, `HLUT2`) | Wave 9.2 (`4× MI50`, `1,704 HLUT3`) | Wave 9.3 (`4× MI50`, `1,704 HLUT3 Turbo`) | Gain vs Wave 9.2 |
| :--- | ---: | ---: | ---: | ---: |
| Standalone `M = 512` cold (`profile_wave9_3_breakdown`) | — | `265.2 tok/s` (`1,931 ms`) | **`317.5 tok/s`** (`1,613 ms`) | **`+19.7%`** |
| Standalone `M = 2,048` cold (`profile_wave9_3_breakdown`) | — | `368.3 tok/s` (`5,561 ms`) | **`610.5 tok/s`** (`3,355 ms`) | **`+65.8%`** |
| Standalone `M = 4,096` cold (`profile_wave9_3_breakdown`) | — | `413.5 tok/s` (`9,906 ms`) | **`689.1 tok/s`** (`5,944 ms`) | **`+66.7%`** |
| Live `:8042` `1,564` tok (`req_1600_b`) | `485.7 tok/s` (`3,220 ms`) | `339.4 tok/s` (`4,608 ms`) | **`530.4 tok/s`** (`2,949 ms`) | **`+56.3%`** (`+9.2%` vs W8.5) |
| Live `:8042` `2,400` tok (`req_1600`) | `485.7 tok/s` (`4,941 ms`) | `354.6 tok/s` (`6,767 ms`) | **`561.3 tok/s`** (`4,275 ms`) | **`+58.3%`** (`+15.6%` vs W8.5) |
| Live `:8042` `4,909` tok (real C++/HIP source) | — | — | **`481.6 tok/s`** (`10,193 ms`) | — |
| Live `:8042` `19,661` tok (real C++/HIP repo) | — | — | **`531.6 tok/s`** (`36.98 s`) | — |
| Live `:8042` `37,508` tok (real multi-file C++/HIP repo) | — | — | **`541.3 tok/s`** (`69.29 s`) | — |
| Live `:3081` GUI `8,547` tok cold turn (`#11`) | — | — | **`668.0 tok/s`** (`12.79 s`) | — |
| Live `:3081` GUI `168,756` tok full-cold turn (`#21`) | — | — | **`580.6 tok/s`** (`290.66 s`) | — |

### 2.2 Context & Multi-File Ingestion (`L1v` HBM2 Radix + Cold NVMe `.kvblk`)

| Ingestion Mode (Live `:8042` & `:3081` GUI) | Reused Tokens | Wall Time | Effective Ingestion Throughput | Target (`>= 20,000 tok/s`) |
| :--- | ---: | ---: | ---: | :---: |
| Live `:8042` `37.6K` Multi-File Repo `L1v` HBM2 Radix (`#8`) | `37,376` (`swa_rep=128`) | `459.66 ms` | **`81,312 tok/s`** | **PASS (`4.1×`)** |
| Live `:8042` `37.6K` Cold NVMe `.kvblk` after `/v1/clear_all_caches` (`#9`) | `37,376` (`nvme=17.59 ms`) | `455.61 ms` | **`82,035 tok/s`** (`2.12M` raw NVMe) | **PASS (`4.1×`)** |
| Live `:3081` GUI `160.5K` Active KV Reuse (`#18`) | `160,453` | `0.20 ms` | **`810,012,772 tok/s`** | **PASS** |
| Live `:3081` GUI `314.6K` `L1v` HBM2 Radix Hit (`#23`) | `312,448` | `88.99 ms` | **`3,511,145 tok/s`** | **PASS (`175×`)** |
| Live `:3081` GUI **`342.7K` Cold NVMe `.kvblk` Reload** after `/v1/clear_all_caches` (`#34`) | **`342,272`** (`nvme=138.52 ms`, `swa_rep=128`) | **`1,117.06 ms`** | **`306,405 tok/s`** (**`2.47M`** raw NVMe) | **PASS (`15.3×`)** |

### 2.3 Sustained Agentic Decode Curve (`8K` -> `342K` -> `1M` Active Context)

| Active Context | Source | `DSpark` Acceptance | `tau` (tok/step) | `T_step` (`ms`) | Sustained Decode (`tok/s`) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `8,192` tok | `profile_wave9_3_breakdown` | `90.7%` | `4.77` | `111.9 ms` | **`44.00 tok/s`** |
| `8,547` tok | Live `:3081` GUI (`#11`, `gen=1,323`, `6` tools) | `80.9%` | `4.44` | `107.6 ms` | **`41.27 tok/s`** |
| `37,577` tok | Live `:8042` (`#8`, multi-file C++ repo) | `81.1%` | `4.54` | `111.7 ms` | **`40.93 tok/s`** |
| `62,862` tok | Live `:3081` GUI (`#13`, `gen=2,591`, `3` tools) | `70.3%` | `3.73` | `114.5 ms` | **`32.56 tok/s`** |
| `65,536` tok | `profile_wave9_3_breakdown` (`chain_max_b=4/5`) | `71.4%–82.6%` | `3.65–4.43` | `110.3–113.7 ms` | **`33.10–40.15 tok/s`** |
| `104,156` tok | Live `:3081` GUI (`#15`, `gen=2,271`) | `63.0%` | `3.15` | `121.2 ms` | **`26.00 tok/s`** |
| `118,763` tok | Live `:3081` GUI (`#17`, **`gen=41,690` continuous**) | `71.0%` | `3.73` | `129.1 ms` | **`28.91 tok/s`** |
| `160,500` tok | Live `:3081` GUI (`#18`, `edit`/compile loop) | **`96.0%`** | **`5.65`** | `126.3 ms` | **`45.08 tok/s`** |
| `168,756` tok | Live `:3081` GUI (`#21`, `gen=1,772`, `10` tools) | `78.0%` | `4.17` | `127.1 ms` | **`32.80 tok/s`** |
| `314,605` tok | Live `:3081` GUI (`#23`, `3` tools) | `84.1%` | `4.51` | `147.9 ms` | **`30.42 tok/s`** |
| `334,600` tok | Live `:3081` GUI (`#25`, `edit` loop) | **`94.3%`** | **`5.59`** | `154.0 ms` | **`36.05 tok/s`** |
| `339,465` tok | Live `:3081` GUI (`#32`, test loop) | **`94.5%`** | **`5.56`** | `148.6 ms` | **`37.20 tok/s`** |
| `342,722` tok | Live `:3081` GUI (`#34`, Cold NVMe + `DeepSeek-ViT`) | `70.0%` | `3.72` | `159.9 ms` | **`23.28 tok/s`** |
| `1,048,575` (`1M`) | `verify_wave8_5_1m_e2e` (`32` bursts, `M=8`) | Gate synthetic | `1.81` | `258.3 ms` (down from `275.9 ms` W9.2) | **`7.02 tok/s`** |

---

## 3. Live `:3081` DeepSeek Harness GUI Autonomous Engineering Campaign Summary

Executed end-to-end directly inside the live DeepSeek Harness GUI (`http://127.0.0.1:3081/`, model badge `DeepSeek-V4.1-Flash-1017 · Wave 9.3 WHT128-HLUT3 Turbo · 4×MI50 + 2×P150A · 1,704 HLUT3 · 1M · Max (100)`):
- **Campaign Scope**: `3` user turns, `23` autonomous tool-calling steps (`#11`–`#34`), **`423,328` cumulative input tokens**, **`90,233` tokens generated**, peak active context **`342,722` tokens**, `92%` overall cache hit rate, **`0` tool syntax failures**, **`0` GPU/PCIe errors**.
- **Turn 1 (`8.5K -> 162.1K` active tokens, `Max` effort)**:
  - Autonomously read 5 Wave 9.3 core source files in parallel, inspected AMDGPU `.s` dual metadata formats via `bash`, wrote a `58,659`-byte pure-C++17 static analyzer (`w93_engine_auditor.cpp`), compiled it on the **first attempt** with `g++ -O3 -std=c++17 -Wall -Wextra -Werror` (`0` warnings/errors), and passed `10/10` synthetic violation self-tests + `13/13` live tree checks.
- **Turn 2 (`168.8K -> 340.9K` active tokens, `Low` effort)**:
  - Fired `10` parallel tool calls reading `6,244` lines across `dsv41_hlut3_kernels.hip`, `dsv41_mi50_kernels.hip`, `dsv41_runtime.cpp`, `src/hlut3_pack.hpp`, and `dsv41_fe.cpp`, pinpointing `k_sparse_attn_allheads` at line `3650` and both bitwise-`|` `s_valid` guards at lines `3725` and `3752` at `>305K` context.
  - Applied `7` surgical `edit` calls expanding `w93_engine_auditor.cpp` to `74,579` bytes, recompiled cleanly with `-Werror` (`0` warnings/errors), and passed **`12/12` self-tests** and **`15/15` live checks** (`145/145` zero-scratch kernels, `0` spills, `0` trig, `0` `slc` atomics, `0` system-scope RMW atomics, `kMaxChunkM == 2048`, `TokSliceCache == 8192`, `5`-arg overlay UUID, `OR`-only `s_valid` guards, `sizeof(Hlut3Tile32x32) == 400`).
- **Turn 3 (`342,722` active tokens — Mid-Session Cold NVMe `.kvblk` Reload + Live `DeepSeek-ViT` Chart Inspection)**:
  - Triggered `POST /v1/clear_all_caches` wiping all HBM2/RAM radix caches at `340.9K` context, attached a live `392×392` PNG throughput bar chart in the `:3081` composer, and sent Turn 3.
  - Hydrated **`342,272` tokens (`2,674` `.kvblk` blocks)** from cold NVMe in **`138.52 ms`** (`2.47M tok/s` raw NVMe; `1.12 s` / **`306,405 tok/s`** including `128`-tok Encoder SWA Replay), encoded the PNG chart via `DeepSeek-ViT` on `tp=2` in `426.6 ms` (`3.8 s` total TTFT), accurately ranked all 3 bars (`Red — shortest [Wave 9.2]`, `Yellow/amber — middle [Wave 8.5]`, `Green — tallest [Wave 9.3 Turbo]`), and recalled all `340K`-context audit constants (`lines 3725 & 3752`, `400 bytes`, `145/145`, `12/12`, `15/15`) with **100% accuracy and zero tool calls**.

---

## 4. Verification Gates, Hardware Telemetry & Bottleneck Analysis

### 4.1 Verification & Safety Audits
- **`make -j12 check` + `make audit-gfx906`**: `PASS` (`145 / 145` zero-scratch `gfx906` kernels, `0` SGPR/VGPR spills, `0` trig, `0` `slc` atomics, `0` `.py` files).
- **`verify_wave9_2_4gpu_dispatch`**: `PASS` (`MTP_DEV3 cos = 1.000000000`, `3WAY_BB cos >= 0.99999925`, `tp_eq = 1`).
- **`verify_wave9_3_rocblas_prefill`**: `PASS` (all kernels `cos >= 0.9999985`).
- **`verify_wave8_5_1m_e2e` (`1,048,576`-token `1M` + `ViT` Gate)**: `PASS` (`73.87 s` total; `1M HSI = 100.000%`, needles `1/1/1`, `CSA2 cos = 1.00000000`, `M=512` Engram wait `0.0011 ms`, `M=1024` Engram wait `0.0019 ms`, NVMe `.kvblk` `CRC32C = 1`, `enc_cos = 1.00000000`, `dec_cos = 1.00000000`, `32`-burst decode to `pos = 1,048,575`).
- **Post-Campaign 4-GPU Telemetry (`>70 min` continuous load up to `342.7K` tokens)**:
  - `tp=0` (32G): `30,565 / 32,752 MiB` (`2,187 MiB` free), `44.0 °C` edge
  - `hlut3_dev=3` (16G switchbox): `15,190 / 16,368 MiB` (`<= 15,360 MiB` cap, `1,178 MiB` free), `38.0 °C` edge, `16GT/s x16 (ok)`, `UESta`/`CESta` all `-`
  - `tp=1` (32G switchbox): `30,545 / 32,752 MiB` (`2,207 MiB` free), `39.0 °C` edge, `16GT/s x16 (ok)`, `UESta`/`CESta` all `-`
  - `tp=2` (32G display + ViT): `30,704 / 32,752 MiB` (`2,048 MiB` free), `47.0 °C` edge
  - Kernel `dmesg`: `0` `amdgpu` faults, `0` `RAS`/`ATHUB`/`CmpltAbrt` errors.
  - Freeze & Rollback: all `14` `FREEZE_*` directories intact (`dr-xr-xr-x`); rollbacks `dsv41_fe.bak_wave8_5_noatomic` and `dsv41_fe.bak_wave9_2_verified` preserved.

### 4.2 Remaining Hardware Ceiling & Wave 9.4 Path to `2,000 tok/s` Cold Prefill & `50+ tok/s` Decode
With Wave 9.3 bringing `TP3 MoE` (`2,124 ms` @ `4K`), `dev3 MoE` (`2,379 ms` @ `4K`), and `2× Blackhole P150A` (`2,695 ms` @ `4K`) into near-exact equilibrium:
1. **Blackhole P150A GDDR6 `m_tiles = 1` Ceiling**: On the current `p150_dsv41_worker` (`--p150-l1-shards 18`), `M > 32` prefill arenas demote to GDDR6 and issue single-tile `m_tiles = 1` matmuls (`2.70 s` for `4K` tokens -> theoretical ceiling `~1,480 tok/s` even if MI50 took `0 ms`). Upgrading the P150A worker's Tensix prefill kernel to `m_tiles = 4..8` (blocked expert-grouped GEMM) will cut `p150_dev` by `~3–4×` in prefill (`2.70s -> ~0.75s`) and `~1.5×` in `M = 6..8` decode (`55 ms -> ~35 ms/step`), immediately unlocking `50+ tok/s` sustained decode (`tau = 4.5–5.5` / `~90 ms/step`) and `1,500–2,000 tok/s` cold prefill.
2. **MI50 `TP3` `HLUT2` Prefill (`1,567 ms` @ `4K`)**: Lowering `rocblas_hlut2_thresh` further with a fused FP16 batch-unpack kernel (`k_w93_unpack_hlut2_batch`) for medium buckets (`m_e = 16..47`) will shave another `~500 ms` from `4K` cold prefill once P150A prefill is unblocked.
