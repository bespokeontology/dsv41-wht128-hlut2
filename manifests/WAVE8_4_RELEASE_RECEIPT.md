# Canonical Release Receipt — `v1.4.0-wave8.4` (`Balanced` Cross-Tier `384`-Expert Placement + `778` `RAREBAL` Experts + `18/18` `P150A` L1 Relay)

- **Repository:** [`bespokeontology/dsv41-wht128-hlut2`](https://github.com/bespokeontology/dsv41-wht128-hlut2)
- **Release Tag:** `v1.4.0-wave8.4` (superseding `v1.3.0-wave8.3`)
- **Toolchain & Hygiene:** `100%` Pure C++20 / C++17 / TT-Metal / `gfx906` HIP (`0` Python files, `-Wall -Wextra -Werror` clean)

## Static & Deterministic Verification Receipts

1. **Zero-Scratch `gfx906` Static ISA Audit (`make audit-gfx906`):**
   - Runtime + `DeepSeek-ViT` `gfx906` kernels: **`99 / 99`** at `ScratchSize: 0` (`v_dot2_f32_f16` `k_indexer_score_msplit<8>` + 10-split register-Q `CSA2` `shared-tree8`)
   - `verify_wave8_opt` `gfx906` kernels: **`3 / 3`** at `ScratchSize: 0`
   - `verify_wave8_4` `gfx906` kernels: **`1 / 1`** at `ScratchSize: 0`
   - Device transcendental instructions (`v_sin_f32` / `v_cos_f32`): **`0`**
   - PCIe BAR P2P calls (`hipDeviceEnablePeerAccess`): **`0`**
   - Python files in repository: **`0`**
2. **Wave 8.4 Cross-Tier Minimax Balancer, `778`-Expert `RAREBAL-LDLQ`, & `P150A` `18`-Shard L1 Geometry (`verify_wave8_4`):**
   - `2× Blackhole P150A` Tier-2 `FABRIC_1D` L1 budget: **`18 / 18` shards/card** (`36` TP2 shards = **`161.37 MiB` pure L1 weights**, `1,243,136 B/bank + 204,800 B CB = 1,447,936 / 1,459,200 B` = **`+11,264 B` bank `0-0` margin**; `7/7` `MeshTrace` shapes `parity_cos == 1.0000000`, `stale_reads == 0`).
   - `384`-Expert `TP3 <-> TP2` Minimax Placement (`445` swapped pairs, `565` base repacked experts): `k_moe_gate` logical-first bias/top-6 invariance **`cos = 1.000000000`** (`max_diff = 2.980e-08`, `0` routing diffs, `0 B` extra VRAM).
   - `778`-Expert `3× MI50` `FP64` `TARQ RAREBAL-LDLQ` Overlay (`668` main + **`110` `DSpark` `mtp.0..2`** + **`18/18` `P150A` L1**): **`0` Dual Receipt violations** (`mean cos 0.88527 -> 0.95523`, **`+4.57 dB`** mean SNR overall, **`+5.45 dB`** mean SNR on `mtp.0..2`, `-63.0%` rare-token MSE).
3. **Wave 7, Wave 8.2 & Wave 8.3 Speculative & Protocol Suites (`verify_wave7_opt`, `test_anthropic_messages_fe`):**
   - `Dynamic DeepSelect8` `100,000` shortage sweeps: **`invalid_accepts = 0`**, **`max_TV_M8 = 2.463e-16 < 1e-14`**.
   - `18,476 B` hostile C++ zero-injection + BPE `"</"` + `"think>"` stream isolation: **`0` structural ID leaks**.

## Verified Silicon Telemetry (`2× Blackhole P150A` + `3× AMD Instinct MI50`)

- **Short Effort-100 Max Agentic Decode (`~8.7K` ctx):** **`40.18 tok/s`** (`97.39–97.81 ms/step`, **`83.6%` warm `DSpark` acceptance**, `τ = 4.55`, `p150w = 32.69 ms/step`).
- **Fresh `225,065`-Token Multimodal Prefill:** **`547.69 tok/s`** (**`+21.8%`** vs. `v1.3.0-wave8.3` via Card `#0` `PCIe x16 -> 4× 800G FABRIC_1D` relay + `GPU_MAX_HW_QUEUES=8`).
- **Cold NVMe `.kvblk` Multimodal Hydration (`0 B` Disk SWA):** **`237,568` tokens hydrated in `0.623 s`** (`623.06 ms`, `156.62 ms` raw NVMe read, **`381,290 tok/s`**, `4,977` `.kvblk` blocks) followed by **`38.91 tok/s`** post-cold-reload decode (`+36.8%` vs. `v1.3.0-wave8.3`).
- **Steady-State & Deep `<think>` Multimodal Decode (`232.7K → 238.5K` ctx):** **`36.22 tok/s`** steady multimodal tool decode (`+10.3%`, live `ViT` `691.3 ms`) and **`25.47–31.19 tok/s`** deep `T = 1.0` `<think>` synthesis (`+20.1%`, `<think>` `DSpark` acceptance **`+8.1 pp`** `59.5% -> 67.6%`).
