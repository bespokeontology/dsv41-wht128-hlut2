# Wave 9.4 Ultra (`SB-M4-P2-176G`) — Exhaustive Throughput, Long-Context, & Prefix-Preserving Engineering Report

**Architecture**: `DeepSeek-V4.1-Flash-1017` (`CSA2` Sparse Attention, `DSpark` 3-layer MTP Speculative Decoding, `Engram` 189 GiB DDR4 Hash Tables, Native `DeepSeek-ViT` + 3×3 Aligner, `DSML` Tool Calling)  
**Hardware Topology (`SB-M4-P2-176G`)**: `4× AMD Instinct MI50` (`gfx906`, `3× 32 GiB` TP3 + `1× 16 GiB` dedicated asymmetric `dev3` = `112 GiB` HBM2, two cards hosted behind a non-AtomicOpsRouting Broadcom `PEX88048` PCIe switchbox) + `2× Tenstorrent Blackhole P150A` (`64 GiB` GDDR6 over `4× 800G` `TT-Metal` `FABRIC_1D` TP2) = **`176 GiB` Accelerator Memory**  
**Precision Tier (`WHT128-HLUT3` 3.125 bpw + `WHT128-HLUT2` 2.125 bpw)**: `1,704` `FP64-TARQ-RAREBAL-LDLQ` calibrated `WHT128-HLUT3` (`400 B` `32×32` tile) experts (`384/384` MTP drafters + `720` unsharded `dev3` backbone + `600` sharded TP3 backbone) + `7,320` compacted TP3 `HLUT2` experts + `6,720` P150A `HLUT2` experts (`[216..383]` invariant).  
**Hygiene Invariants**: `100%` pure C++20/C++17/HIP/TT-Metal C++ (`0` `.py` files anywhere), `-Wall -Wextra -Werror`, **`164/164` zero-scratch `gfx906` kernels** (`ScratchSize == 0`, `0` SGPR spills, `0` VGPR spills, `0` device trig), **`0` PCIe system-scope RMW atomics / `slc` atomics** across all 4 MI50s.

---

## 1. Executive Summary — Wave 9.3 Baseline vs. Wave 9.4 Ultra

All benchmark timings below report **both** honest end-to-end wall-clock throughput (`tokens / wall_elapsed_s`) and internal per-step timing (`T_step`, `tau`, acceptance rate) on identical `SB-M4-P2-176G` hardware at stock `225 W` MI50 power caps.

### 1.1 Prefill & Interactive Tool-Tail Sub-Stage Improvements

| Scenario | Metric | Wave 9.3 Baseline | Wave 9.4 Ultra | Speedup / Delta |
| :--- | :--- | ---: | ---: | ---: |
| **Interactive Tool-Tail (`M=256 @ 340K KV`)** | **Wall Latency (`ms`)** | `1,731.2 ms` | **`991.0 ms`** | **`1.75×` (`-740.2 ms`, sub-1s!)** |
| `M=256 @ 340K KV` | **Wall `tok/s`** | `147.9 tok/s` | **`258.3 tok/s`** | **`+74.6%`** |
| `M=256 @ 340K KV` | CSA2 HSI Indexer (`ms`) | `725.1 ms` | **`174.3 ms`** | **`4.16×`** |
| `M=256 @ 340K KV` | `dev3` Wait / Busy (`ms`) | `205.2 / 515.3 ms` | **`34.4 / 309.6 ms`** | **`5.97×` wait / `1.66×` busy** |
| **Cold Prefill (`M=2048`)** | **Wall Latency (`ms`)** | `3,358.8 ms` | **`2,744.6 ms`** | **`1.22×` (`-614.2 ms`)** |
| `Cold M=2048` | **Wall `tok/s`** | `609.7 tok/s` | **`746.2 tok/s`** | **`+22.4%`** |
| `Cold M=2048` | `P150A` Wait / Device (`ms`) | `363.2 / 1,704.0 ms` | **`5.2 / 846.4 ms`** | **`69.8×` wait / `2.01×` dev** |
| **Cold Prefill (`M=4096`)** | **Wall Latency (`ms`)** | `5,882.9 ms` | **`4,604.1 ms`** | **`1.28×` (`-1,278.8 ms`)** |
| `Cold M=4096` | **Wall `tok/s`** | `696.3 tok/s` | **`889.6 tok/s`** | **`+27.8%`** |
| `Cold M=4096` | `P150A` Wait / Device (`ms`) | `563.7 / 2,665.8 ms` | **`7.6 / 1,241.7 ms`** | **`74.2×` wait / `2.15×` dev** |
| `Cold M=4096` | `dev3` Wait / Busy (`ms`) | `487.8 / 2,306.5 ms` | **`50.5 / 1,207.5 ms`** | **`9.66×` wait / `1.91×` busy** |
| **Live Cold GUI Prefill (`M=8,981`)** | **Wall `tok/s`** | `~700 tok/s` | **`913.9 tok/s`** | **`+30.6%` (`9.83 s` wall)** |

### 1.2 Decode Step Latency (`T_step`) & Attention Scaling (`8K -> 1M` Context)

Note: Fixed-burst `64`-token synthetic decode throughput depends on stochastic `DSpark` acceptance `tau` on a given seed; **architecture-intrinsic step latency `T_step` and `Attn/step`** isolate true engine speed, while the **Live `:3081` GUI Campaign (Section 3)** reports honest true stopwatch throughput over **`135,548` real generated tokens** up to **`544,210` active tokens**.

| Active Context | Wave 9.3 `T_step` (`ms`) | Wave 9.4 `T_step` (`ms`) | Wave 9.3 `Attn/step` (`ms`) | Wave 9.4 `Attn/step` (`ms`) | `Attn` Speedup | `T_step` Speedup | Synthetic `tok/s` (`tau`) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `8,192` | `105.83 ms` | **`102.27 ms`** | `27.36 ms` | `27.87 ms` | `0.98×` | `1.03×` | **`52.15 tok/s`** (`tau=5.25`, `94.4%` acc) |
| `65,536` | `114.10 ms` | **`104.48 ms`** | `37.49 ms` | **`30.84 ms`** | **`1.22×`** | **`1.09×`** | `26.63 tok/s` (`tau=2.74`, `53.3%` acc) |
| `163,840` | `134.51 ms` | **`117.92 ms`** | `54.69 ms` | **`34.10 ms`** | **`1.60×`** | **`1.14×`** | `23.60 tok/s` (`tau=2.70`, `44.8%` acc) |
| `348,160` | `156.70 ms` | **`120.21 ms`** | `78.53 ms` | **`36.23 ms`** | **`2.17×`** | **`1.30×`** | `28.02 tok/s` (`tau=3.32`, `62.0%` acc) |
| `1,048,576` (`1M` E2E) | — | **`131.0 ms`** | — | **`51.0 ms`** | — | — | `41.5 tok/s` @ `tau=5.4` (est.) |

---

## 2. Engineering Changes Delivered (Phases B–E)

### Phase B — `2× Blackhole P150A` Multi-Row Compact Prefill & Trace-Cache Fix
- Raised `DSV41_P150_MAX_ROWS` from `2048 -> 4096` (`SHM_VERSION = 4`), expanded `kBatchSlots = 24` (`kMaxCompactSlots = 640`, `11` hot L1 shards/card with `256 KiB/core` L1 guard), and implemented an **Expert-Grouped Contiguous Best-Fit Batch Packer** in `p150/p150_dsv41_engine.cpp` + `p150/kernels/dsv41_hlut2_writer_b.cpp`.
- **Root-caused & fixed a critical `op_traces` cache eviction bug**: `p150_dsv41_engine.cpp` previously checked `op_traces.size() < 96u` *before* calling `op_traces.find(key)`, permanently disabling all TT-Metal MeshTraces (prefill and decode) once 96 unique shapes accumulated. Moving `op_traces.find(key)` ahead of insertion and raising the cap to `256u` reclaimed `~535 ms` on `M=4096` prefill and `~27 ms/step` on decode.
- Verified **9/9 P150A silicon self-checks PASS** (`M = 1, 6, 7, 8, 16, 64, 512, 2048, 4096`; `parity_cos = 1.0000000`, `fp32_cos = 0.999912..0.999917`). Cut `COLD M=4096` P150A device time **`2,665.8 -> 1,241.7 ms` (`2.15×`)** and host wait **`563.7 -> 7.6 ms` (`74.2×`)**.

### Phase C — `4× MI50` `HHHS` `B=8` Strided-Batched rocBLAS, `INT8` `dev3` Wire, & `128`-Aligned CED
- Migrated `RocblasPrefillEngine` routed-expert GEMMs to **`HHHS`** (`FP16` out, `FP32` `v_dot2_f32_f16` compute) with `B=8` strided-batched dispatch (`16`-aligned `m_pad` on TP3, `32`-aligned on `dev3`, 4-slot metadata ring) and calibrated `rocblas_expert_thresh = 18` / `rocblas_hlut2_thresh = 20`.
- Added per-128-block **`INT8 + FP16-scale` H2D** and **`BF16` D2H** wire compression on the `tp0 <-> dev3` PCIe offload link (`cos = 0.99995590 >= 0.99995`, `0 B` extra HBM2 via disjoint buffer aliasing).
- Expanded `kMaxChunkM = 4096` and snapped `eff_ced_split` to 128-aligned boundaries (collapsing `P - pos <= 384` into a single 40-layer decoder chunk, eliminating redundant 20-layer encoder tail passes). Cut `COLD M=4096` `dev3` busy time **`2,306.5 -> 1,207.5 ms` (`1.91×`)** and wait **`487.8 -> 50.5 ms` (`9.66×`)**.

### Phase D — Sub-Second `340K` CSA2 HSI Indexer & Zero-Atomic TP0 Reduction
- Implemented **Compact `mode == 2` Candidate Scoring** (`k_w94_indexer_score_cand` + `k_w94_indexer_cand_topk`) scoring only the `<= 16,384` candidate tokens selected at layer 20 (`21.25×` less compute at `340K`, `64×` at `1M`).
- Added **60-CU Split-KV Exact Block-Max + Radix Select** (`k_w94_indexer_causal_and_blockmax` + `k_w94_indexer_splitkv_select`) for `T_comp > 16,384`, `k_indexer_score_msplit<10>` for `M > 8`, **TP0-only FP16-wire reduction** (`k_host_ring_reduce_tp0_f16_wire` — `6×` less PCIe traffic vs TP3 FP32 allreduce), and tiny **TP0->TP1,2 `u32` broadcast** (`k_host_ring_bcast_tp0_u32`).
- Verified **100.000% bit-exact parity (`0/4096` index diffs)** across `8K–340K` linear and causal-tree modes on silicon. Cut `M=256 @ 340K` HSI **`725.1 -> 174.3 ms` (`4.16×`)** and `340K` decode attention **`78.53 -> 36.23 ms` (`2.17×`)**.

### Phase E — Prefix-Preserving `reasoningEffort` Switching
- Decoupled prompt prefix rendering (`index == 0` fixed `prefix_base_effort`, default `65` / `Medium` in live frontend) from active-turn effort overrides injected cleanly via native `<｜latest_reminder｜>` (`token 128828`) after `<｜Assistant｜><think>` *only* when `out_budget != prefix_base_effort`.
- Verified **100.0% history token identity** across `low`/`medium`/`high`/`max` toggles on both OpenAI `/v1/chat/completions` and Anthropic `/anthropic/v1/messages` endpoints in unit tests (`test_anthropic_messages_fe`) and live GUI runs.

---

## 3. Live `:3081` Harness GUI Campaign (`4` Turns · `108` LLM Steps · `152` Tool Calls · `544,210` Peak Active Tokens)

A full end-to-end autonomous engineering campaign was driven live through the DeepSeek Harness GUI (`:3081` -> `:8042` Wave 9.4 Ultra frontend) without interruption, tasking `DeepSeek-V4.1-Flash-1017` to inspect 18 Wave 9.4 source/doc files, author a 1,300+ line pure-C++17 release verifier (`w94_release_verifier.cpp`), compile with `-Wall -Wextra -Werror`, pass `21/21` synthetic self-tests + live repo & release-readiness audits, switch effort dynamically in both directions, and reload cold from NVMe `.kvblk` with a live `DeepSeek-ViT` chart attachment.

### 3.1 True Stopwatch Wall-Clock vs. Internal `T_step` / `tau` Telemetry

| Context Band | Completed Requests | Total Generated Tokens | **True Stopwatch Avg (`tok/s`)** | **Peak Request (`tok/s`)** | Avg `T_step` (`ms`) | Avg `tau` | Peak Fresh Ingest (`tok/s`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `<= 160K` | `15` | `12,821` | **`31.69 tok/s`** | **`54.46 tok/s`** (`11K` ctx) | `108.54 ms` | `3.44` | **`913.9 tok/s`** (`8,981` cold tok) |
| `160K – 350K` | `26` | `77,538` | **`35.86 tok/s`** | **`56.00 tok/s`** (`230K` ctx) | `113.43 ms` | `4.07` | **`753.0 tok/s`** (`52,711` tok @ `319K`) |
| `350K – 544K` | `67` | `45,189` | **`31.95 tok/s`** | **`52.37 tok/s`** (`408K` ctx) | `121.05 ms` | `3.87` | **`715.7 tok/s`** (`56,203` tok @ `375K`) |
| **Full Session (`0 -> 544K`)** | **`108`** | **`135,548`** | **`34.05 tok/s`** | **`56.00 tok/s`** (`230K` ctx) | **`115.48 ms`** | **`3.93`** | **`913.9 tok/s`** (`666.0` avg) |

**Notable Milestone Live GUI Requests**:
- **Request `#1` (Cold `8,981`-tok prompt)**: `913.9 tok/s` cold prefill, `46.02 tok/s` decode (`tau=4.79`).
- **Request `#20` (`194.9K -> 224.4K` single-shot `95 KB` C++17 file generation)**: **`29,546` continuous generated tokens** in `660.4 s` = **`44.74 tok/s` sustained** (`tau=5.14`, `86.9%` DSpark acceptance), compiled cleanly with `-Wall -Wextra -Werror` on the first try!
- **Request `#24` (`229,934` active tokens)**: **`56.00 tok/s` decode** (`tau=5.77`, `100.0%` DSpark acceptance, `785 ms` tool-tail TTFT).
- **Request `#41` (`319,335` active tokens)**: `52,711` fresh tokens ingested @ **`753.0 tok/s`**, **`54.14 tok/s` decode** (`tau=5.89`, `98.5%` acceptance, `hsi = 0.26 ms/step`).
- **Request `#43` (`407,826` active tokens)**: `32,530` fresh tokens ingested @ **`709.1 tok/s`**, **`52.37 tok/s` decode** (`tau=5.75`, `98.0%` acceptance, `hsi = 0.24 ms/step`).
- **Request `#54` (`486,688` active tokens)**: **`51.37 tok/s` decode** (`tau=5.80`, `97.3%` acceptance, `hsi = 0.26 ms/step`).
- **Request `#97` (`533,414` active tokens)**: **`50.26 tok/s` decode** (`tau=5.58`, `95.2%` acceptance, `hsi = 0.27 ms/step`).

### 3.2 Live Bidirectional Prefix-Preserving Effort Switch & Cold NVMe `.kvblk` + `DeepSeek-ViT` Proof

| Event | Request `#` | Active Context | Prefix Reused (`tokens`) | Reuse `%` | Restore Time / Effective Rate | Decode `tok/s` | Outcome |
| :--- | :---: | ---: | ---: | ---: | :--- | ---: | :--- |
| **Switch `Medium (65) -> Low (50)`** | `#39` | `266,440` | `265,494 / 265,494` prior | **`100.00%`** (`99.64%` tot) | `0.47 ms` (**`568M tok/s`** hot) | **`48.39 tok/s`** | **PASS (`0` re-prefill)** |
| **Switch `Low (50) -> Medium (65)`** | `#108` | `543,185` | `541,696 / 543,185` | **`99.73%`** | `504.64 ms` (**`1.07M tok/s`**) | **`37.04 tok/s`** | **PASS (`0` re-prefill, `4/4` exact recall)** |
| **Cold NVMe `.kvblk` Reload + Live `DeepSeek-ViT` Chart** (after `POST /v1/clear_all_caches`) | `#109` | `544,210` | **`543,744`** (`nvme_hit=543,744`) | **`99.91%`** | **`366.15 ms` NVMe (`1.49M tok/s`)** / `1,414.6 ms` full HBM2+SWA (`384K tok/s`) | **`37.73 tok/s`** (`3.1s` TTFT, `img=1` `516.9 ms`) | **PASS (`4/4` chart bars visually read & cross-checked exact!)** |

**Overall Session Cache Reuse**: **`98.89%`** (`41,447,486 / 41,912,011` prompt tokens reused across `108` steps).

---

## 4. Verification Gates & Hardware Health Receipt

| Gate | Status | Result |
| :--- | :---: | :--- |
| `make -j12 check` + `make audit-gfx906` | **PASS** | `164/164` zero-scratch `gfx906` kernels (`0` SGPR/VGPR spills, `0` device trig, `0` `slc`/system RMW atomics) |
| `tools/w94_engine_auditor` (`--self-test` & `--root`) | **PASS** | `12/12` synthetic negative tests PASS; `15/15` live tree checks PASS |
| `verify_wave9_2_4gpu_dispatch` | **PASS** | `6/6` PASS (`MTP_DEV3 cos = 1.000000000`, `3WAY_BB cos = 0.999999300`) |
| `verify_wave9_3_rocblas_prefill` | **PASS** | `INT8-BLK128-WIRE cos = 0.99995590`, `HLUT3 5.38×`, `HLUT2 4.64×` |
| `verify_wave8_5_1m_e2e` (`1,048,576` tokens + ViT + NVMe `.kvblk`) | **PASS** | Free HBM2: `tp0=1,738`, `tp1=1,740`, `tp2=2,164`, `dev3=1,134 MiB`; `100.000%` HSI recall; `3/3` needles HIT; `CSA2 cos=1.00000000`; `CRC32C=PASS` |
| Switchbox PCIe Health (`0000:85:00.0` & `0000:88:00.0`) after `544K` GUI run | **PASS** | `DevSta/UESta/CESta` 100% clean (`0` `ERREVENT_ATHUB_INTERRUPT`, `0` `CmpltAbrt`, `0` AER); temps `43–58 °C` |
| Self-hosted `DeepSeek-V4.1-Flash` Verifier (`w94_release_verifier`) | **PASS** | `21/21` self-tests PASS; `6/6` base audit PASS; `0` `.py` files anywhere |
