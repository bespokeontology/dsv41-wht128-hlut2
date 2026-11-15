# Canonical Release Receipt — `v1.3.0-wave8.3` (`Dynamic DeepSelect8` + `DeepSeek-ViT` + Cold NVMe `CSA2` KV)

- **Repository:** [`bespokeontology/dsv41-wht128-hlut2`](https://github.com/bespokeontology/dsv41-wht128-hlut2)
- **Release Tag:** `v1.3.0-wave8.3` (superseding `v1.2.0-wave8`, sharing immutable `hot_experts_rarebal_wave8.safetensors.part-00..02` overlay shards)
- **Toolchain & Hygiene:** `100%` Pure C++20 / C++17 / TT-Metal / `gfx906` HIP (`0` Python files, `-Wall -Wextra -Werror` clean)

## Static & Deterministic Verification Receipts

1. **Zero-Scratch `gfx906` Static ISA Audit (`make audit-gfx906`):**
   - Runtime + `DeepSeek-ViT` `gfx906` kernels: **`96 / 96`** at `ScratchSize: 0`
   - `verify_wave8_opt` `gfx906` kernels: **`3 / 3`** at `ScratchSize: 0`
   - Device transcendental instructions (`v_sin_f32` / `v_cos_f32`): **`0`**
   - PCIe BAR P2P calls (`hipDeviceEnablePeerAccess`): **`0`**
   - Python files in repository: **`0`**
2. **Wave 7 + Wave 8.3 Deterministic Optimizer & Speculative Tree Suite (`verify_wave7_opt`):**
   - `2× Blackhole P150A` Tier-1b `FABRIC_1D` L1 budget: **`17 / 17` shards/card** (`1,453,184 B <= 1,459,200 B/bank`, **`129,927,808 B = 123.91 MiB/card`, `247.82 MiB` pure active L1** across `2` cards, `0 B` extra L1 for `M = 16`).
   - Wave 8.3A `kAutoTree16MaxContext == 0`: keeps `M = 8` across `0..1,048,576` (`Fixed16` remains opt-in via `--dspark-tree-width 16`).
   - Wave 8.3B `deep_select_top7_confidence_tree8`: verified exact depth-`5` chain collapse and wide `4 + 3` depth-`2` bush + `64`-bit SGPR bitpacks (`anc_mask64`, `depth_nibbles32`, `parent_nibbles32`).
   - Wave 8.3C `valid_node_mask` quarantine (`100,000` `V = 4` shortage + post-EOS sweeps): **`invalid_accepts = 0`**, **`max_TV_M8 = 2.463e-16 < 1e-14`**, `max_TV_M16 = 2.359e-16 < 1e-14`.
3. **Stage 1, Track 8E & Delta 8.3D Protocol + Anti-Injection Suite (`test_anthropic_messages_fe`):**
   - `encoding.py` `Reasoning Effort: 100` / `drop_thinking = false` / reminder parity: **PASS**
   - `/anthropic/v1/messages` + OpenAI multimodal `<｜deepseek_image｜>` (`129264`) span expansion & call-order `tool_result` sorting: **PASS**
   - Delta 8.3D `18,476 B` (`>= 16 KiB`) hostile C++ zero-injection + BPE `"</"` + `"think>"` stream isolation (`0` injected `128803`/`128804`/`129264`/`128822`/`128825`): **PASS**

## Verified Silicon Telemetry (`2× Blackhole P150A` + `3× AMD Instinct MI50`)

- **Short Effort-100 Multimodal/Code Decode (`~8.7K` ctx):** **`40.03 tok/s`** (`#1`) / **`39.77 tok/s`** (`#3`), `τ = 4.19–4.32`, `76.7%` acceptance (**`2.48×`** vs. `4`-batch `M = 16`).
- **Matched `25K`-Output A/B (`--dspark-static-m8` vs. `Dynamic DeepSelect8`):** `33.23 → 35.22 tok/s` (**`+6.0%`** net `tok/s`, **`+26.3%`** accepted tree drafts/step `1.118 → 1.411`, `64.7% → 72.3%` acceptance, `+0.34 ms` drafter cost).
- **Cold NVMe `.kvblk` Multimodal Hydration (`0 B` Disk SWA):** **`220,544` tokens in `0.632 s`** (`109.23 ms` raw NVMe @ **`1.71 GB/s`**) and **`224,384` tokens in `0.722 s`** (`226.26 ms` raw NVMe + `128`-token Encoder SWA replay, `disk_swa_files = 0`).
- **Steady-State Long Multimodal Decode (`228,225 → 232,604` ctx):** **`32.85 tok/s`** (`τ = 4.40`, **`84.2%` acceptance**, `0.28 ms` `814.6M tok/s` GPU slot hit; **`+13.4%`** over static `M = 8` and **`2.50×`** over `M = 16`).
