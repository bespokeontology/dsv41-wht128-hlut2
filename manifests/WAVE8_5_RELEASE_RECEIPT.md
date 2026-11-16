# Wave 8.5 + Self-Hosted Wave 9 `v1.5.0-wave8.5` Canonical Release Receipt

* **Release Tag**: `v1.5.0-wave8.5` (superseding `v1.4.0-wave8.4` and `v1.3.0-wave8.3`)
* **Model**: `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`763B`-class Multimodal: `559.7B` 40L MoE + `~14B` 3-Stage DSpark MTP + `188.83 GiB` FP8 Engram + `14.4B` BF16 DeepSeek-ViT)
* **Wave 8.5 Calibrated Overlay**: `hot_experts_wave8_5.safetensors` (`24,720,594,696` bytes = `23.02 GiB`, `14,043` tensors + `expert_perm.u16` `[40, 384]`, split into `13` `< 1,900 MiB` release chunks `.part-00` .. `.part-12`)
  * **Total Emitted `TARQ RAREBAL-LDLQ` Experts**: **`2,182`** (`1,996` main backbone + **`186` `DSpark` `mtp.0..2` drafters = `99.47%` draft hit coverage**, `1,404` newly solved in Wave 8.5 + `778` preserved from Wave 8.4)
  * **Pinned `2× Blackhole P150A` L1 Shards**: **`18/18` per card (`36/36` TP2 = `161.37 MiB` pure L1 weights**, `+11,264 B` bank `0-0` margin, Card `#0` `PCIe x16 -> 4× 800G FABRIC_1D` relay)
  * **Dual Receipt Violations**: **`0 / 2,182`** (`M_cal >= 10,240`, `M_eval = 512`, `0x5242414Du` header + non-finite BF16 scrub)
  * **Held-Out SwiGLU Quality**: Mean cosine `0.88344 → 0.95456`, mean SNR `6.37 dB → 11.00 dB` (**`+4.63 dB`** overall; **`+5.34 dB`** on `DSpark` MTP)
* **Cross-Sublayer `mHC` + `DSpark` Epilogue Fusion, `L1v` Fence & `DSML` Guard (`116/116` Zero-Scratch `gfx906` Kernels)**:
  * `106/106` runtime + ViT + `5/5` Wave 9 `HLUT3` + `3/3` `verify8` + `1/1` `verify8_4` + `1/1` `verify8_5` = **`116/116` compiled `gfx906` kernels at `ScratchSize: 0`**, `0` SGPR/VGPR spills, `0` `v_sin_f32`/`v_cos_f32`
  * Short-context decode step latency: **`97.39 ms → 87.58 ms/step` (`-9.81 ms/step`)**
  * Short-context Effort-100 throughput: **`58.58 tok/s`** (`23.2K` ctx, `95.0%` `DSpark` acc, `τ = 5.70`) and **`55.12 tok/s`** (`9.3K` ctx)
  * Fresh prefill throughput: **`623.0 tok/s`** (`0.0009 ms/chunk` `M = 512` & `0.0020 ms/superchunk` `M = 1024` double-buffered `Engram` overlap)
  * Long-horizon self-hosted Wave 9 `WHT128-HLUT3` agentic throughput (`177` cumulative live tool steps):
    * `230.8K` ctx: **`45.88 tok/s`** (`94.5%` `DSpark` acc, `τ = 5.58`)
    * `288.2K` ctx (after `0.704 s` cold NVMe reload at `409,368 tok/s`): **`42.05 tok/s`** (`93.4%` acc, `τ = 5.53`)
    * `360.0K` ctx: **`40.11 tok/s`** (`92.7%` acc, `τ = 5.44`)
    * `390.2K` ctx: **`34.26 tok/s`** (`86.6%` acc, `τ = 5.00`)
    * `391,936`-token (`3,062`-block) second cold NVMe `.kvblk` reload: **`0.869 s` (`265.58 ms` raw NVMe read, `451,094 tok/s`, `0 B` disk SWA)**, continuing cleanly through **`395,743` active context tokens**
* **Full `1,048,576`-Token (`1M`) E2E Silicon Receipt (`docs/WAVE8_5_1M_E2E_RECEIPT.md`, `59.06 s` `PASS`)**:
  * Simultaneous `1M` + `DeepSeek-ViT` free HBM2: **`268.00 / 270.00 / 696.00 MiB`** (`>= 200 MiB` `PASS`, `64`-bit pointer audit `PASS`)
  * `1M` `HSI` `v_dot2_f32_f16` top-`64` overlap across `8,192` blocks: **`100.000%` (`PASS`)**, `3/3` needles retrieved (`pos = 128`, `524,288`, `900,000`)
  * `1M` Tiered `CSA2` `v_dot2_f32_f16` vs `FP32` parity: **`cosine = 1.00000000`**, `max_abs_diff = 0.000e+00`, `nonfinite_count = 0`
  * `32`-burst `Dynamic DeepSelect8` (`M = 8`) decode through `pos = 1,048,575`: **`229.38 ms/step`**, **`7.63 tok/s`** (`56` tokens emitted)
  * Cold NVMe `.kvblk` boundary/tail hydration (`0 B` disk SWA): **`6.74 ms`**, `CRC32C = PASS`, `128`-token `Encoder SWA` replay **`enc_cos = 1.00000000`**, `40L` decoder continuation **`dec_cos = 1.00000000`**
* **Self-Hosted Wave 9 `WHT128-HLUT3` (`3.125 bpw`) Foundation (`docs/WAVE9_HLUT3_SELFHOST_REPORT.md`, `PASS`)**:
  * Analytic 8-level symmetric Gaussian Lloyd-Max fixed point: `c = {0.24509418, 0.75600528, 1.34390928, 2.15194570}`, `SQNR_3 = 14.6158 dB` (**`+5.3155 dB`** over `HLUT2`)
  * Real Layer-6 Expert-0 `MXFP4` Dual Receipt (`bench_hlut3_mi50`): `HLUT3+RAREBAL` SwiGLU `cosine = 0.98541 > 0.98404`, rare `MSE = 4.4961e-04 < 4.8856e-04`, `SNR = 15.38 dB` (**`+5.16 dB >= +4.0 dB`** over `HLUT2+RAREBAL`), GPU-vs-CPU parity **`cosine = 0.999999999`**
  * `P150A` `6,400 B` superblock L1 fit & PCIe switchbox planner (`plan_wave9_switchbox`): **`11` hot shards/card (`1,413,248 B <= 1,459,200 B`, `+45,952 B` margin)**, opt-in `--hlut3-mtp-overlay <path>` (`0 B` HBM2 delta when unset)
