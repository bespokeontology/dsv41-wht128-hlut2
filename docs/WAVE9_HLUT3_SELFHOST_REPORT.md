# Wave 9 WHT128-HLUT3 Selfhost Engineering Report

**Workspace:** `${DSV41_WAVE9_WORKTREE}/`  
**Scope:** Wave 9 WHT128-HLUT3 3.125 bpw, 400 B/tile, 6,400 B P150A superblock.  
**Result:** `overall PASS` on all three required binaries, full clean rebuild, and ISA audit.

---

## 1. Executive Receipt

| Deliverable | Required binary / artifact | Exit / receipt | Result |
|---|---|---:|---|
| D1 real-expert fidelity | `./bin/bench_hlut3_mi50` | `exit 0`, `overall PASS` | **PASS** |
| D1 strict Dual Receipt | bench receipt: cosine, rare MSE, SNR gain | 3/3 gates PASS | **PASS** |
| D1 MI50 device-2 M=16 silicon parity | bench: `cosine=1.000000000` | `>= 0.999999` | **PASS** |
| D2 switchbox planner | `./bin/plan_wave9_switchbox` | `exit 0`, `overall PASS` | **PASS** |
| D3 opt-in HLUT3 MTP overlay wiring | runtime + FE compile/link clean; default 0 B delta | unset no-op | **PASS** |
| D4 analytic/struct verifier | `./bin/verify_wave9_hlut3` | `exit 0`, `overall PASS` | **PASS** |
| D4 clean rebuild | `make clean && make -j12 all && make audit-gfx906` | `exit 0`; 0 warnings, 0 errors | **PASS** |
| D4 zero Python | whole workspace | `0` `.py` files | **PASS** |
| D4 gfx906 ISA audit | 116 compiled kernels across 5 `.s` files | 116/116 scratch 0, spills 0, 0 trig | **PASS** |

The final build log is `wave9_build.log`; the three required run logs are:
`build/tools/wave9_verify.log`,
`build/tools/wave9_bench_hlut3.log`,
`build/tools/wave9_switchbox.log`.

No process opened `/dev/tenstorrent/*` or `/dsv41_p150_ipc`; no FREEZE/packed directory was modified.
`verify_wave9_hlut3` and `bench_hlut3_mi50` both call `hipSetDevice(2)` as the first HIP runtime call and
allocate 14.090 MiB total HBM2, strictly <= 48 MiB. The planner is pure host C++ (no HIP allocation).

---

## 2. Phase-1 Audit and Real Defects Fixed

### 2.1 Real benchmark rotation defect

The prior `build/tools/wave9_bench_hlut3.log` reported ~0 SwiGLU cosine for every rotated baseline.
Root cause: the CPU evaluator rotated the weights and the input activations, but never rotated the
**SwiGLU intermediate Z** back into the I=2304 rotated basis before the down projection.

Production basis law:

- `gate_rot = WHT128(gate, signs_h)` rotates the K=H=5120 axis.
- `up_rot   = WHT128(up,   signs_h)` rotates the K=H axis.
- `x_rot    = WHT128(x,    signs_h)`.
- `g[m,n] = sum_k x_rot[m,k] * gate_rot[n,k] = sum_j x[m,j] * gate[n,j]`
  because `H_128/sqrt(128)` is symmetric orthogonal. Therefore gate/up outputs are already in the
  **original hidden basis**.
- `down_rot = WHT128(down, signs_i)` rotates the K=I=2304 axis.
- Hence the down contraction must consume `z_rot = WHT128(z, signs_i)`, not original-basis `z`.

Fix in `tools/bench_hlut3_mi50.cpp`:
`run_swiglu_f32(... rotate_intermediate, signs_i, ...)` rotates `Z` with `wht128_rows(Z, signs_i)`
when `QuantizedSet.rotated == true`. All five real-expert baselines now evaluate on the correct basis.
The GPU pipeline already did this correctly; the CPU reference now mirrors it exactly.

### 2.2 HLUT3 LDLQ stale-code overwrite

`hlut3_pack_codes_to_seed_tile` originally OR-ed the new 2-bit magnitude / 1-bit sign onto the seed
planes produced by independent `quantize_hlut3`. OR cannot replace stale bits:
seed mag=2 (`10b`) OR new mag=1 (`01b`) => `11b`=3, wrong magnitude; seed sign=1 OR new sign=0 stays 1.

Fix in `src/dsv41_hlut3_rarebal.hpp`: mask-clear then set every cell:

```
q_lo2[word] = (q_lo2[word] & ~(3u << shift)) | (mag << shift)
q_hi1[word] = (q_hi1[word] & ~(1u << shift)) | (sign << shift)
```

Every tile cell is now overwritten exactly once with the feedback-quantized code.

### 2.3 HLUT3 RAREBAL block-Hessian defect

The Phase-1 quantizer used an independent 128-wide **block-diagonal** Hessian per 128-column panel.
That ignored all cross-block correlations `H[j,k]` between distinct panels. On real rare-token
activations, importance weighting amplifies rare rows, so block-local feedback over-corrected and
catastrophically inflated rare MSE (prior SNR -1.86 dB, rare MSE 3.11e+02).

Fix: HLUT3 RAREBAL now uses the **global K-wide** importance-weighted Hessian + order-reversed
Cholesky, delegated bit-for-bit to the production builder
`src/hlut2_pack.hpp::build_rarebal_cholesky_rup`:

```
H_bal = (1/M_cal) * (X_com^T X_com + lambda_tarq * X_rare^T X_rare) + 0.015 * d_bar * I_K
lambda_tarq = tr_common / tr_rare
```

Then `H_rev = P H P^T`, lower Cholesky `L_rev L_rev^T`, and
`R[col,k] = L_rev[K-1-col, K-1-k] / L_rev[K-1-k, K-1-k]` for `col <= k`, `R[col,col]=1`.
The applied residual law reproduces the production Wave-8 HLUT2 LDLQ contract exactly:
`err = w_orig[col] - qv(wr[col])`, propagated `wr[k] += err * R[col,k]`.
**The production HLUT2 code was not modified**, so baseline (c) remains the exact production contract.

### 2.4 HLUT3 pair kernels: expert selection and TP3 slice geometry

The Phase-1 pair kernels ignored `pr.expert` and hardcoded full geometry `2304x5120` gate/up and
`5120x2304` down. Fixes in `mi50/dsv41_hlut3_kernels.hip` / `.hpp`:

- expert-major tile base: `expert * (N/32) * (K/32)`, then `nt * (K/32) + kt`;
- bounds check `pr.expert >= n_experts` => skip;
- parameterized `n_gate, k_gate, n_down, k_down`:
  full verifier/bench uses `2304,5120,5120,2304`; runtime TP3 rank slice uses `768,5120,5120,768`;
- new FP16 -> FP32 staging kernel for the opt-in runtime overlay path (`launch_hlut3_f16_to_f32`).

---

## 3. Mathematical Derivations

### 3.1 Folded half-normal moments (X ~ N(0,1))

For `|X|` in `[a,b]`, total over both signs:

```
mass   = P(|X| in [a,b])       = 2 * (Phi(b) - Phi(a))
first  = E[|X| 1(|X| in [a,b])] = 2 * (pdf(a) - pdf(b))
second = E[X^2 1(|X| in [a,b])] = 2 * (Phi(b) - Phi(a) + a*pdf(a) - b*pdf(b))
```

With `b = +inf`: `pdf(b)=0`, `Phi(b)=1`. No magic quantile constants; quantiles by bisection.

### 3.2 Symmetric Gaussian Lloyd-Max fixed point (4 positive centroids + sign)

For `n_pos=4`, thresholds `t_i = 0.5*(c_i + c_{i+1})`, centroids
`c_i = first_i / mass_i` over `[t_{i-1}, t_i]` (`t_{-1}=0`, `t_3=+inf`). Iterate until
`max |c_i^{new} - c_i^{old}| <= 1e-7`; final partition is published at the midpoint boundaries.
Converged exactly at iteration 101, residual `9.062e-08`, fixed-point residual `7.930e-08`.

Solved values:

```
HLUT3 levels     c = [0.245094008523, 0.756004815833, 1.343908643574, 2.151945077793]
HLUT3 thresholds t = [0.500549412178, 1.049956729703, 1.747926860683]
HLUT3 MSE = 0.034547865738  EQ2 = 0.965452134262  SQNR = 14.615788 dB
HLUT2 MSE = 0.117481905322  SQNR = 9.300290 dB      gain = +5.315498 dB
```

### 3.3 Exact distortion identity

Region distortion for centroid `c_i`:

```
D_i = E[X^2 | region i] - 2 c_i E[|X| | region i] + c_i^2 P_i
    = second_i - 2 c_i first_i + c_i^2 mass_i
```

Because `c_i = first_i/mass_i`, `-2 c_i first_i = -2 c_i^2 mass_i`, so `D_i = second_i - c_i^2 mass_i`.
Summing all regions and using `sum second_i = E[X^2] = 1`:

```
MSE = 1 - sum_i c_i^2 * mass_i = 1 - EQ2
```

Verifier asserts `|EQ2 - (1 - MSE)| <= 1e-10`. No fitted/magic centroids.

### 3.4 Bit layout and byte identities

Tile flatten `idx = r*32 + c`:

- magnitude: `q_lo2[idx >> 4]`, bits `[2*(idx&15)+1 : 2*(idx&15)]` = code 0..3;
- sign: `q_hi1[idx >> 5]`, bit `[idx & 31]`;
- scale: `scales_f16[idx >> 7]`, FP16 RMS over 128 weights (4 rows x 32 cols).

1024 weights per tile:

```
magnitude = 1024 * 2 bits / 8 = 256 B   (64 uint32)
sign      = 1024 * 1 bit  / 8 = 128 B   (32 uint32)
scales    = 1024 * 16 bits / 128 = 16 B (8 uint16)
total     = 400 B                       = 3.125 bits/weight
```

Superblock `128x128` = 16 subtiles × 400 B = 6,400 B = 25 × 256 B.
Static asserts in `src/hlut3_pack.hpp` prove 400 B, 6400 B, align 16/256, tile coverage.

### 3.5 HLUT3 vs HLUT2 tile planes

HLUT2 at 2.125 bpw: 2 magnitude/sign bits + 16 bits/128 scale = 2.125; its MI50 planar 2-bit
code plane is 256 B/tile and its P150A 272 B/tile superblock packs magnitude plus BF16 scales.
HLUT3 adds a separate 128 B sign plane, so it is exactly 400 B/tile and 6,400 B/superblock.

### 3.6 TP3 per-rank promotion contract

Published Wave-9 contract bytes and exact MiB identities (integer factorization, no floats):

```
HLUT3 per-rank TP3 expert = 295 * 16,384 B = 4,833,280 B = 295/64 MiB = 4.609375 MiB  (display 4.609)
HLUT2 per-rank TP3 expert = 12,839 * 256 B = 3,286,784 B = 12,839/4096 MiB = 3.134521484375 MiB (display 3.134)
promotion delta            = 6,041 * 256 B  = 1,546,496 B = 6,041/4096 MiB = 1.474853515625 MiB (approx +1.475)
top-32 MTP delta          = 32 * 6,041 * 256 B = 49,487,872 B = 6,041/128 MiB = 47.1953125 MiB (approx +47.19)
```

Static asserts in `tools/plan_wave9_switchbox.cpp` bind all four constants.
The full TP3 expert used for raw VRAM ceilings is `3 * 4,833,280 = 14,499,840 B`.
The P150A TP2 expert is exactly `6,912,000 B = 3 * 360 * 6,400 B`.

---

## 4. Struct Layouts

### 4.1 `dsv41::Hlut3Tile32x32` (alignas 16, exactly 400 B)

| Field | Count | Bytes | Role |
|---|---:|---:|---|
| `q_lo2` | 64 `uint32_t` | 256 | 2-bit magnitude codes, 1024 weights |
| `q_hi1` | 32 `uint32_t` | 128 | 1-bit sign plane, 1024 weights |
| `scales_f16` | 8 `uint16_t` | 16 | FP16 RMS per 128-weight group |

### 4.2 `dsv41::Hlut3Superblock128x128` (alignas 256, exactly 6,400 B)

```
subtiles[16] = Hlut3Tile32x32;   // nt_sub*4 + kt_sub order, stride 400 B, no padding
sizeof == 6400 == 25 * 256
```

### 4.3 Quantizer structs

- `Hlut3LloydMaxSolution`: `levels[4]`, `thresholds[3]`, `mse`, `eq2`, `sqnr_db`,
  `convergence_residual`, `fixed_point_residual`, `iterations`, `converged`.
- `Hlut3QuantMatrix`: `N`, `K`, `nt=N/32`, `kt=K/32`, `lloyd`, planar `tiles[nt*kt]` (400 B each),
  `raw_bytes()`, `bits_per_weight()`, `tile_at()`.
- `Hlut3Mi50Packed`: `N`, `K`, `lloyd`, `tiles` — bit-layout identical to CPU reference.
- `Hlut3RarebalStats`: `m_cal/m_common/m_rare`, starvation flag, `tr_common`, `tr_rare`,
  `lambda_tarq`, `d_bar`, `lambda_reg`.

### 4.4 Runtime opt-in overlay fields

`BlockGpuWeights` adds, only populated when the overlay path is supplied:

```
bool has_hlut3;
Hlut3Tile32x32* d_hlut3_gate_tiles;  // [128, 24 nt, 160 kt] 400 B tiles
Hlut3Tile32x32* d_hlut3_up_tiles;
Hlut3Tile32x32* d_hlut3_down_tiles;  // [128, 160 nt, 24 kt] 400 B tiles
```

`TpRankContext` adds `d_hlut3_levels[4]`, `d_hlut3_z[48,768]`, `d_hlut3_zrot[48,768]`,
`d_x_rot_f32[16,5120]`. All null / 0 B when `--hlut3-mtp-overlay` is unset.

---

## 5. gfx906 ISA Audit

Audit command: `make audit-gfx906` compiled disassembly of every compiled gfx906 kernel source.

| Assembly file | Kernels | ScratchSize 0 | `.sgpr_spill_count: 0` | `.vgpr_spill_count: 0` | trig (`v_sin_f32`/`v_cos_f32`) |
|---|---:|---:|---:|---:|---:|
| `build/tools/wave8_gfx906.s` | 106 | 106 | 106 | 106 | 0 |
| `build/tools/verify_wave8_gfx906.s` | 3 | 3 | 3 | 3 | 0 |
| `build/tools/verify_wave8_4_gfx906.s` | 1 | 1 | 1 | 1 | 0 |
| `build/tools/verify_wave8_5_gfx906.s` | 1 | 1 | 1 | 1 | 0 |
| `build/tools/hlut3_gfx906.s` | 5 | 5 | 5 | 5 | 0 |
| **TOTAL** | **116** | **116** | **116** | **116** | **0** |

`ScratchSize: 0` = dynamic shared/scratch bytes 0; `.sgpr_spill_count: 0` and
`.vgpr_spill_count: 0` = 0 SGPR/VGPR spills. All zero-scratch checks are equality tests, not samples.
HLUT3 kernels are: GEMV, pair gate+up SwiGLU, inverse WHT128 Z, down atomic, and the opt-in
FP16->FP32 conversion kernel; all 5 report ScratchSize 0 / 0 spills / 0 trig.

---

## 6. Real-Expert Five-Baseline CPU Receipt (final run)

Real weights: `${DSV41_ROOT}/shards_skhynix/layer6_expert0_scales_e8m0.bin`
and `${DSV41_ROOT}/shards_greatwall_data/layer6_expert0_weights_mxfp4.bin`
(read-only), `2304x5120` gate/up/down MXFP4. Calibration `M_cal=11264`, eval `M_eval=512`
(common 256, rare 256). **All five baselines PASS** (finite, exact bpw measured metrics).

| Method | bpw | SwiGLU cosine | SNR dB | Common MSE | Rare MSE | Cos(Com) | Cos(Rare) | ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (a) Uniform Absmax 2b unrotated | 2.125 | 0.72958 | -1.29 | 6.564229e+01 | 2.144041e+02 | 0.73298 | 0.72891 | 632.02 |
| (b) WHT128-HLUT2 H=I | 2.125 | 0.83828 | 5.27 | 1.297911e+01 | 4.883565e+01 | 0.83968 | 0.83794 | 570.08 |
| (c) WHT128-HLUT2 RAREBAL-LDLQ | 2.125 | 0.97472 | 13.02 | 2.024816e+00 | 8.355175e+00 | 0.97674 | 0.97419 | 6375.74 |
| (d) WHT128-HLUT3 H=I | 3.125 | 0.95149 | 10.24 | 4.126834e+00 | 1.555821e+01 | 0.95190 | 0.95138 | 757.82 |
| (e) WHT128-HLUT3 RAREBAL-LDLQ | 3.125 | 0.99310 | 18.62 | 5.257226e-01 | 2.333337e+00 | 0.99400 | 0.99286 | 10185.34 |

Strict Dual Receipt gates:

```
HLUT3+RAREBAL cosine 0.99310 > HLUT3(H=I) 0.95149                         : PASS
HLUT3+RAREBAL rare MSE 2.333337e+00 < HLUT3(H=I) 1.555821e+01            : PASS
HLUT3+RAREBAL SNR 18.62 dB vs HLUT2+RAREBAL 13.02 dB gain +5.60 dB >= +4.0 : PASS
```

Theory lattice gain (N=8 vs N=4 Gaussian fixed point): `+5.315498 dB` verifier; `+5.32 dB` bench display.

---

## 7. MI50 Device-2 Silicon Latency / Effective HBM2 GB/s (M=1,6,8,16)

Locked device 2: free HBM2 696.00 MiB >= 550 MiB; total device allocation 14.090 MiB <= 48 MiB.
Tile bytes = 3 matrices × 4,608,000 B = 13,824,000 B = 13.18359375 MiB (display 13.184 MB).
Effective GB/s = 13,824,000 B / mean_us / 1e3 (decimal GB/s).

| M | mean latency us | tile bytes | effective GB/s |
|---:|---:|---:|---:|
| 1 | 463.22 | 13,824,000 | 29.84 |
| 6 | 1471.69 | 13,824,000 | 9.39 |
| 8 | 1832.10 | 13,824,000 | 7.55 |
| 16 | 3486.85 | 13,824,000 | 3.96 |

Device-2 CPU/GPU parity: M=16 `cosine=1.000000000`, `max_abs=1.525879e-04` => PASS `>= 0.999999`.
Verifier parity M=1/6/8/16: cosine 1.000000000, max_abs 1.525879e-05 / 1.335144e-05 /
1.692772e-05 / 1.668930e-05, min cosine 1.0 => PASS.

---

## 8. P150A 6,400 B Superblock L1 Math

- Tile stride: 400 B; subtiles: 16; superblock: 6,400 B = 25 × 256 B.
- Gate/up TP2 shard: `K=5120 -> kt4=40`, `N_shard=1152 -> nt4=9`, 360 superblocks;
  360 × 6,400 = **2,304,000 B**.
- Down TP2 shard: `K=1152 -> kt4=9`, `N_shard=5120 -> nt4=40`, 360 superblocks;
  360 × 6,400 = **2,304,000 B**.
- TP2 expert (gate+up+down): 3 × 360 × 6,400 = **6,912,000 B = 6.591796875 MiB**.

Exact FABRIC_1D per-bank budget at 11 HLUT3 hot shards/card:

```
slot_ids        ceil(1/110)      * 640   =     640
gate_partials   ceil(5760/110)=53 * 4096 = 217,088
up_partials     ceil(5760/110)=53 * 4096 = 217,088
z_rot_bf16      ceil(576/110)=6  * 2048 =  12,288
combined_f32    ceil(160/110)=2  * 4096 =   8,192
l1_slot_lut     ceil(40/110)=1   * 512  =     512
x_decode        ceil(160/110)=2  * 2048 =   4,096
3 * hot shards  ceil(3960/110)=36* 6400 *3 = 691,200
256 KiB/core CB guard ceil(7040/110)=64 * 4096 = 262,144
-------------------------------------------------------
total per bank = 1,413,248 B <= 1,459,200 B ceiling, margin 45,952 B
```

At 12 hot shards/card: `1,490,048 B > 1,459,200 B`, overflow **30,848 B** => 11 is the exact live ceiling.
11 hot shards/card = 11 × 6,912,000 = **76,032,000 B = 72.509765625 MiB** (display 72.51).
TP2 hot pair = 22 shards = **152,064,000 B = 145.01953125 MiB**.
110 Tensix banks/card; TT-Metal InterleavedBuffer page rounding:
`ceil(360/110)=4 pages/bank/projection`; 4 × 110 × 6,400 = 2,816,000 B vs 360 × 6,400 = 2,304,000 B,
padding **512,000 B = 80 × 6,400 B**. `p150_hlut3_geometry_report` printed the exact contract.

---

## 9. Switchbox Capacity Tables (4x-7x MI50 + 2x-3x P150A)

### 9.1 VRAM / TP3 / TP2 rank slots

32 GiB per MI50 HBM2 card and 32 GiB per P150A GDDR6 card. `PCIe slots = MI50 + P150A`.
`TP3 groups = floor(MI50/3)`, `TP2 groups = floor(P150A/2)`.

| cfg | MI50 | P150 | PCIe | MI50 GiB | P150 GiB | Total GiB | TP3 groups | TP3 rank slots | TP2 groups | TP2 rank slots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SB-M4-P2 | 4 | 2 | 6 | 128 | 64 | 192 | 1 | 4 | 1 | 2 |
| SB-M4-P3 | 4 | 3 | 7 | 128 | 96 | 224 | 1 | 4 | 1 | 3 |
| SB-M5-P2 | 5 | 2 | 7 | 160 | 64 | 224 | 1 | 5 | 1 | 2 |
| SB-M5-P3 | 5 | 3 | 8 | 160 | 96 | 256 | 1 | 5 | 1 | 3 |
| SB-M6-P2 | 6 | 2 | 8 | 192 | 64 | 256 | 2 | 6 | 1 | 2 |
| SB-M6-P3 | 6 | 3 | 9 | 192 | 96 | 288 | 2 | 6 | 1 | 3 |
| SB-M7-P2 | 7 | 2 | 9 | 224 | 64 | 288 | 2 | 7 | 1 | 2 |
| SB-M7-P3 | 7 | 3 | 10 | 224 | 96 | 320 | 2 | 7 | 1 | 3 |

### 9.2 Raw VRAM expert ceilings, P150A L1 fit, per-rank top-N MTP promotion

Raw ceilings are weights-only (`MI50 total / 14,499,840 B full TP3 HLUT3 expert`,
`P150A total / 6,912,000 B TP2 expert`); runtime attention/shared/MTP reserves excluded.
L1 bank bytes are identical across all configs: 1,413,248 B, margin 45,952 B, fit PASS.

| cfg | TP3raw | TP2raw | L1 hot shards | L1 bank | margin | fit | T8 MiB | T16 MiB | T32 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SB-M4-P2 | 9478 | 9942 | 22 | 1,413,248 | 45,952 | PASS | 11.798828 | 23.597656 | 47.195312 |
| SB-M4-P3 | 9478 | 14913 | 33 | 1,413,248 | 45,952 | PASS | 11.798828 | 23.597656 | 47.195312 |
| SB-M5-P2 | 11848 | 9942 | 22 | 1,413,248 | 45,952 | PASS | 11.798828 | 23.597656 | 47.195312 |
| SB-M5-P3 | 11848 | 14913 | 33 | 1,413,248 | 45,952 | PASS | 11.798828 | 23.597656 | 47.195312 |
| SB-M6-P2 | 14217 | 9942 | 22 | 1,413,248 | 45,952 | PASS | 11.798828 | 23.597656 | 47.195312 |
| SB-M6-P3 | 14217 | 14913 | 33 | 1,413,248 | 45,952 | PASS | 11.798828 | 23.597656 | 47.195312 |
| SB-M7-P2 | 16587 | 9942 | 22 | 1,413,248 | 45,952 | PASS | 11.798828 | 23.597656 | 47.195312 |
| SB-M7-P3 | 16587 | 14913 | 33 | 1,413,248 | 45,952 | PASS | 11.798828 | 23.597656 | 47.195312 |

### 9.3 Top-N DSpark MTP HLUT3 overlay promotion (per TP3 rank)

| N | bytes | exact MiB | display MiB | note |
|---:|---:|---:|---:|---|
| 8 | 12,371,968 | 11.798828125 | 11.799 | subset |
| 16 | 24,743,936 | 23.59765625 | 23.598 | subset |
| 32 | 49,487,872 | 47.1953125 | 47.195 | **top-32 PASS (approx +47.19 MiB/rank)** |

### 9.4 Top-32 MTP overlay totals across TP3 ranks

| cfg | rank bytes | total MiB |
|---|---:|---:|
| SB-M4-P2 / SB-M4-P3 | 49,487,872 | 188.781250 |
| SB-M5-P2 / SB-M5-P3 | 49,487,872 | 235.976562 |
| SB-M6-P2 / SB-M6-P3 | 49,487,872 | 283.171875 |
| SB-M7-P2 / SB-M7-P3 | 49,487,872 | 330.367188 |

All arithmetic is integer-exact bytes; no floats in byte identities; MiB divisions are 1/2^20.

---

## 10. Opt-In `--hlut3-mtp-overlay <path>` Wiring

- `mi50/dsv41_runtime.hpp`: `void set_hlut3_mtp_overlay(const std::string&)`.
- `mi50/dsv41_runtime.cpp`: Impl `hlut3_mtp_overlay_path`; loader only for MTP blocks 40..42.
  Safetensors contract names (expert-major TP3 rank slices):

```
hlut3.b<bid>.gate_tiles  UINT8 [128, 24, 160, 400] = 196,608,000 B
hlut3.b<bid>.up_tiles    UINT8 [128, 24, 160, 400] = 196,608,000 B
hlut3.b<bid>.down_tiles  UINT8 [128, 160, 24, 400] = 196,608,000 B
```

Per block 589,824,000 B; 3 MTP blocks total 1,769,472,000 B per rank. Loader guards free HBM2 >=
tiles + scratch + 550 MiB; if insufficient it refuses and leaves HLUT2 unchanged. It uploads the exact
analytic Gaussian levels and allocates 48-pair 768-dimension scratch. The MTP dispatch swaps to:

```
launch_hlut3_f16_to_f32(d_x_rot_f16, d_x_rot_f32, M*5120)
launch_mi50_routed_hlut3_pairs(d_x_rot_f32, pairs, P, M, 128,
                               768, 5120, 5120, 768, ...)
```

only when `use_pairs && b.has_hlut3`; the HLUT2 path stays byte-identical otherwise.

- `mi50/dsv41_fe.cpp`: CLI `--hlut3-mtp-overlay <path>` / `--no-hlut3-mtp-overlay`, setter before init.
- Unset path: loader returns immediately -> **0 B allocation, 0 footprint change, default 2.125 bpw**.
- No `/dev/tenstorrent` or `/dsv41_p150_ipc` is opened by the loader (it reads the overlay path only).

---

## 11. Makefile Build / Audit Wiring

- `Makefile` `TOOLS_BINS` adds `verify_wave9_hlut3`, `bench_hlut3_mi50`, `plan_wave9_switchbox`.
- Wave9 compile/link rules added with `HIP_WALL_FLAGS` (-Wall -Wextra -Werror) and `WAVE9_LDFLAGS`.
- `TOOL_CXXFLAGS` now carries `-Werror`; `mi50/Makefile` HIPFLAGS carries
  `-Wall -Wextra -Werror -Wno-unused-parameter`.
- `mi50/Makefile` `OBJS` adds `dsv41_hlut3_kernels.o`, so `all` compiles the HLUT3 HIP object and
  links `dsv41_fe` + all runtime tools; `build/mi50/dsv41_hlut3_kernels.o` (28,256 B) exists.
- Existing runtime-link tools that reference the new launches (`harvest_rarebal_wave8`,
  `verify_wave8_2_deepinfer`, `verify_wave8_4`, `verify_wave8_5`, `verify_wave8_5_1m_e2e`)
  now link `dsv41_hlut3_kernels.o`, keeping link clean.
- `audit-gfx906` compiles `mi50/dsv41_hlut3_kernels.hip` to `build/tools/hlut3_gfx906.s`
  and audits ScratchSize, `.sgpr_spill_count`, `.vgpr_spill_count`, and device trig
  across all 5 assembly files. Both `all` and `audit-gfx906` are wired.

---

## 12. Clean Rebuild and Verification Receipts

### 12.1 Build

```
make clean && make -j12 all && make audit-gfx906
```

Receipt:

```
build_exit = 0
warning:   = 0
error:     = 0
.py files  = 0
TOTAL gfx906 kernels            = 116
TOTAL zero-scratch kernels      = 116
TOTAL sgpr_spill_count markers  = 116
TOTAL zero sgpr spills          = 116
TOTAL vgpr_spill_count markers  = 116
TOTAL zero vgpr spills          = 116
TOTAL device trig instructions   = 0
```

### 12.2 Required run receipts

```
./bin/verify_wave9_hlut3      exit 0, overall PASS
GPU_MAX_HW_QUEUES=8 ./bin/bench_hlut3_mi50  exit 0, overall PASS
./bin/plan_wave9_switchbox    exit 0, overall PASS
```

Verifier PASS lines:

```
solver_convergence_le_1e-7 : PASS (delta 9.062e-08, fixed 7.930e-08)
tile_400B_pack_unpack_exact : PASS
p150_6400B_superblock_eq    : PASS (16 subtiles, 12,800 B over 2 blocks)
device2_free_hbm2_ge_550MiB : 696.00 MiB PASS
device2_total_alloc_le_48MiB: 14.090 MiB PASS
hip_vs_cpu_parity_M1_6_8_16 : min_cos=1.000000000 max_abs=1.692772e-05 PASS
real_weight_hlut3_bytes     : 4,608,000 B/matrix (3.125 bpw exact)
overall                     : PASS
```

Bench PASS lines:

```
device_locked               : hipSetDevice(2) FIRST HIP call
free_hbm2                  : 696.00 MiB >= 550 MiB PASS
total_hbm2_alloc           : 14.090 MiB <= 48 MiB PASS
real_weight_hlut3_bytes    : 4,608,000 B/matrix (3.125 bpw exact)
dual_receipt               : PASS
hlut3_vs_hlut2_theory      : +5.32 dB (N=8 vs N=4 Gaussian fixed point)
silicon_parity_M16         : PASS
overall                    : PASS
```

Planner PASS lines:

```
configs_4x7_mi50_2x3_p150 : 8 PASS
tile_400B_superblock_6400B : PASS
tp3_rank_hlut3_4833280B   : PASS (4.609 MiB exact 4.609375)
tp3_rank_hlut2_3286784B   : PASS (3.134 MiB exact 3.134521484375)
promotion_delta_1546496B  : PASS (1.474853515625 MiB approx +1.475/rank)
top32_mtp_49487872B       : PASS (47.1953125 MiB approx +47.19/rank)
p150_l1_bank_fit_11_shards : PASS (1,413,248 <= 1,459,200)
overall                   : PASS
```

---

## 13. Files Changed / Removed

Changed:

- `src/dsv41_hlut3_rarebal.hpp` — clear-set code overwrite; global importance-weighted reverse
  Cholesky (production HLUT2 builder delegation); HLUT3 lattice causal LDLQ.
- `tools/bench_hlut3_mi50.cpp` — intermediate Z rotation before down; expert-aware parameterized
  launch calls; correct CPU parity reference.
- `tools/verify_wave9_hlut3.cpp` — parameterized expert-aware launch geometry (n_experts=1,
  full 2304/5120/5120/2304).
- `mi50/dsv41_hlut3_kernels.hpp` / `.hip` — expert-major tiles, parameterized TP3 slice geometry,
  bounds check, FP16->FP32 staging kernel.
- `mi50/dsv41_runtime.hpp` / `.cpp` — `set_hlut3_mtp_overlay`, loader, TP3 rank scratch, MTP dispatch.
- `mi50/dsv41_fe.cpp` — `--hlut3-mtp-overlay <path>` / `--no-hlut3-mtp-overlay`.
- `Makefile` — Wave9 bins/rules, `-Werror`, audit hlut3 + spill checks, audit totals.
- `mi50/Makefile` — HIP `-Wall -Wextra -Werror`, HLUT3 object in OBJS, runtime dependency.
- New `tools/plan_wave9_switchbox.cpp`.
- New `WAVE9_HLUT3_SELFHOST_REPORT.md` (this file).

Removed temporary scratch probes:

```
tools/hip_device2_probe.cpp
tools/solver_probe.cpp
tools/p150_hlut3_probe.cpp
```

(A temporary `tools/w9_debug.cpp` LDLQ diagnostic was also created and removed after diagnosis;
no `.py` file existed at any point.)

---

## 14. Verification Notes / Honest Scope

- HLUT2 production quantizer/LDLQ code was not altered; baseline (c) uses the exact production
  Wave-8 code. HLUT3 delegates to the same Hessian/cholesky builder to keep dual-tier bit parity.
- The HLUT3 residual law deliberately mirrors production HLUT2 (`err = original - qv`); it is the
  documented dual-tier contract that delivered the PASS gain, not an arbitrary rewrite.
- The Phase-1 block Hessian is removed from the applied path; only the global K-wide Hessian is used.
- The runtime overlay loader was compiled/linked clean but **not executed** against live devices 0/1
  or `/dsv41_p150_ipc`; only device 2 was exercised by the required verifier and benchmark.
- All published primary contract bytes (4,833,280 / 3,286,784 / delta / top-32) are integer-exact
  and static-asserted; the raw tile plane geometry remains 400 B/tile and 6,400 B/superblock.
- Zero Python: `find . -name '*.py' | wc -l` = 0 after clean rebuild.
- Zero warnings: `grep -cE 'warning:' wave9_build.log` = 0; `error:` = 0.

**Final status: PASS.**
