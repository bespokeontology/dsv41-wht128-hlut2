# Wave 9.2 / 8.5 — 1,048,576-Token (`1M`) End-to-End Silicon Verification Receipt

- **Runtime Topology**: `3x AMD Instinct MI50 32GB (gfx906 TP3)` + `1x AMD Instinct MI50 16GB (hlut3_dev=3 WHT128-HLUT3)` + `2x Tenstorrent Blackhole P150A (TP2, 18/18 L1 + x16->FABRIC_1D)` + `188.83 GiB Locked DDR4 Engram` + `Dsv41VitEngine (tp=2)`
- **Active Overlay**: `./packed_rarebal_overlay_wave8_5/hot_experts_wave8_5.safetensors`
- **Active `hlut3_dev=3` Overlay**: `${DSV41_WAVE9_OVERLAY_DIR}/wave9_2_dev3_hlut3.safetensors`
- **Active `TP3` `HLUT3` Overlay**: `${DSV41_WAVE9_OVERLAY_DIR}/wave9_2_tp3_hlut3.safetensors`
- **Context Ceiling (`max_seq_len`)**: `1,048,576` tokens (`8,192` x `128`-token Radix/NVMe blocks)
- **Total E2E Gate Wall Time**: `393.97 s`

## 1. Simultaneous 1M VRAM Headroom & 64-Bit Pointer Audit

| Rank | Device Role | Free HBM2 (`MiB`) | Total HBM2 (`MiB`) | `>= 200 MiB` Gate | `uint64_t` Pointer Audit |
| :--- | :--- | ---: | ---: | :---: | :---: |
| `tp=0` | `Backbone` | `1738.00` | `32752.00` | `PASS` | `PASS` |
| `tp=1` | `Backbone` | `1740.00` | `32752.00` | `PASS` | `PASS` |
| `tp=2` | `Backbone + Dsv41VitEngine` | `2164.00` | `32752.00` | `PASS` | `PASS` |
| `hlut3_dev=3` | `Unsharded MTP (384) + Backbone (720) WHT128-HLUT3` | `1134.00` | `16368.00` | `PASS` | `PASS` |

- **Tiered CSA2 Slot Partition (`tp=0..2`)**:
  - Slot `0` (`r=2`): `max_comp=524288` (`HBM2 hot=262144`, `Coherent DDR4 ext=262144`)
  - Slot `1` (`r=2`): `max_comp=524288` (`HBM2 hot=262144`, `Coherent DDR4 ext=262144`)
  - Slot `2` (`r=2`): `max_comp=524288` (`HBM2 hot=262144`, `Coherent DDR4 ext=262144`)
  - Slot `3` (`r=1`): `max_comp=1048576` (`HBM2 hot=524288`, `Coherent DDR4 ext=524288`)

## 2. Double-Buffered Prefill Engram Overlap & 1M HSI / CSA2 Numerical Stability

- **`M = 512` Prefill Double-Buffered Engram Wait**: `0.0004 ms/chunk` (`PASS`)
- **`M = 1024` Superchunk Prefill + P150A `M = 1024` `x16` Relay Wait**: `0.0008 ms/superchunk` (`PASS`)
- **1M Chunked Prefill (`[1,046,528 .. 1,048,543]`, `2,016` tokens)**: `6056.68 ms`, `int32_overflow_free = true`, `nonfinite_count = 0`
- **1M `HSI` `v_dot2_f32_f16` (`clamp=false`) vs `FP32` Top-64 Overlap (`8,192` blocks)**: `100.000%` (`>= 99.8%` gate)
- **Planted Needle Retrieval Across 1M Span**: `pos=128 (blk 1): HIT`, `pos=524,288 (blk 4096): HIT`, `pos=900,000 (blk 7031): HIT`
- **1M Tiered `CSA2` `v_dot2_f32_f16` vs `FP32` Output Parity**: `cosine = 1.00000000`, `max_abs_diff = 0.000e+00`

## 3. 32-Burst `Dynamic DeepSelect8` (`M = 8`) Decode Across `1,048,544 -> 1,048,575` (`1M - 1`)

- **Completed Bursts**: `32` | **Final Verified Position**: `1048575` (`1,048,575 == 1M - 1`) | **Total Emitted**: `56` tokens
- **Mean Burst Wall Time**: `117.35 ms/step` | **Aggregate 1M Throughput**: `14.91 tok/s`

| Burst | `start_pos` | `acc` | `ms/step` | `tok/s` | `attn` | `hsi` | `moe` | `p150w` | `ar` | `dsp` | `smp` | `egw` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `1048544` | 2 | 139.67 | 14.32 | 53.03 | 0.27 | 37.78 | 22.80 | 8.65 | 12.69 | 0.33 | 0.00 |
| 1 | `1048546` | 2 | 134.90 | 14.83 | 52.26 | 0.28 | 37.17 | 23.03 | 7.80 | 10.06 | 0.30 | 0.08 |
| 2 | `1048548` | 2 | 129.07 | 15.50 | 51.64 | 0.28 | 36.38 | 19.31 | 6.82 | 9.99 | 0.29 | 0.00 |
| 3 | `1048550` | 2 | 132.25 | 15.12 | 51.26 | 0.27 | 37.26 | 20.60 | 8.11 | 10.69 | 0.29 | 0.06 |
| 4 | `1048552` | 2 | 133.05 | 15.03 | 51.32 | 0.29 | 39.21 | 21.03 | 6.84 | 10.15 | 0.29 | 0.00 |
| 5 | `1048554` | 2 | 133.28 | 15.01 | 51.03 | 0.28 | 38.50 | 20.50 | 7.62 | 11.10 | 0.32 | 0.00 |
| 6 | `1048556` | 2 | 129.27 | 15.47 | 52.44 | 0.29 | 38.18 | 17.97 | 5.78 | 10.61 | 0.27 | 0.24 |
| 7 | `1048558` | 1 | 135.74 | 7.37 | 53.08 | 0.31 | 37.15 | 23.96 | 5.90 | 11.42 | 0.17 | 0.00 |
| 8 | `1048559` | 2 | 134.20 | 14.90 | 52.42 | 0.28 | 38.21 | 21.18 | 6.61 | 11.04 | 0.28 | 0.00 |
| 9 | `1048561` | 2 | 138.38 | 14.45 | 52.71 | 0.30 | 38.08 | 25.09 | 6.43 | 11.68 | 0.32 | 0.00 |
| 10 | `1048563` | 2 | 129.86 | 15.40 | 53.16 | 0.29 | 36.41 | 18.77 | 6.42 | 10.18 | 0.35 | 0.04 |
| 11 | `1048565` | 2 | 130.54 | 15.32 | 51.56 | 0.31 | 36.34 | 21.15 | 6.79 | 10.26 | 0.28 | 0.00 |
| 12 | `1048567` | 2 | 129.99 | 15.39 | 52.01 | 0.28 | 36.71 | 20.06 | 6.21 | 10.10 | 0.30 | 0.18 |
| 13 | `1048569` | 1 | 65.54 | 15.26 | 27.32 | 0.32 | 18.64 | 12.26 | 4.92 | 0.00 | 0.09 | 0.00 |
| 14 | `1048570` | 1 | 63.90 | 15.65 | 27.16 | 0.27 | 17.85 | 11.74 | 4.64 | 0.00 | 0.09 | 0.00 |
| 15 | `1048571` | 1 | 64.64 | 15.47 | 27.13 | 0.25 | 18.33 | 11.85 | 4.86 | 0.00 | 0.09 | 0.00 |
| 16 | `1048572` | 1 | 62.59 | 15.98 | 27.04 | 0.25 | 18.38 | 10.23 | 4.65 | 0.00 | 0.09 | 0.00 |
| 17 | `1048573` | 1 | 63.58 | 15.73 | 27.41 | 0.26 | 18.04 | 11.21 | 4.61 | 0.00 | 0.09 | 0.00 |
| 18 | `1048574` | 1 | 63.73 | 15.69 | 26.88 | 0.28 | 17.61 | 12.10 | 4.63 | 0.00 | 0.09 | 0.00 |
| 19 | `1048575` | 1 | 64.07 | 15.61 | 27.17 | 0.29 | 17.82 | 12.07 | 4.51 | 0.00 | 0.09 | 0.00 |
| 20 | `1048568` | 2 | 131.41 | 15.22 | 51.00 | 0.26 | 39.60 | 19.81 | 7.34 | 9.68 | 0.31 | 0.07 |
| 21 | `1048568` | 2 | 131.04 | 15.26 | 51.21 | 0.29 | 39.68 | 19.16 | 7.19 | 9.69 | 0.29 | 0.00 |
| 22 | `1048568` | 2 | 131.64 | 15.19 | 51.81 | 0.29 | 39.69 | 19.32 | 7.05 | 10.05 | 0.29 | 0.00 |
| 23 | `1048568` | 2 | 131.26 | 15.24 | 51.12 | 0.28 | 39.55 | 19.28 | 7.52 | 9.87 | 0.28 | 0.00 |
| 24 | `1048568` | 2 | 131.08 | 15.26 | 50.91 | 0.28 | 39.59 | 19.32 | 7.26 | 10.05 | 0.28 | 0.00 |
| 25 | `1048568` | 2 | 131.78 | 15.18 | 51.72 | 0.27 | 39.69 | 19.23 | 7.33 | 9.94 | 0.28 | 0.36 |
| 26 | `1048568` | 2 | 131.74 | 15.18 | 51.81 | 0.27 | 39.64 | 19.41 | 6.91 | 10.06 | 0.29 | 0.00 |
| 27 | `1048568` | 2 | 131.43 | 15.22 | 51.20 | 0.27 | 39.53 | 19.42 | 7.10 | 10.17 | 0.28 | 0.00 |
| 28 | `1048568` | 2 | 130.88 | 15.28 | 50.85 | 0.28 | 39.60 | 19.34 | 7.16 | 10.01 | 0.28 | 0.00 |
| 29 | `1048568` | 2 | 131.80 | 15.17 | 51.35 | 0.28 | 39.68 | 19.37 | 7.52 | 9.92 | 0.28 | 0.29 |
| 30 | `1048568` | 2 | 131.64 | 15.19 | 51.55 | 0.29 | 39.57 | 19.50 | 6.96 | 10.08 | 0.29 | 0.00 |
| 31 | `1048568` | 2 | 131.11 | 15.25 | 51.35 | 0.27 | 39.62 | 19.27 | 7.14 | 9.84 | 0.28 | 0.00 |

## 4. 1M Boundary & Tail Pure-Block NVMe `.kvblk` Cold Hydration & `Encoder SWA` Replay

- **Flushed Boundary + Tail Blocks**: `13` (`blocks {0, 1, 4095, 4096, 7031, 8184..8191}`)
- **On-Disk `.kvblk` Footprint**: `114496 B/block` (`576 B` header + `113,920 B` payload = `890 B/tok`), **`0 B` disk SWA**
- **Cold NVMe Hydration + Hardware `CRC32C`**: `PASS` (`3.28 ms` across `13` blocks)
- **`128`-Token `Encoder SWA Bounded Replay` (`[1,048,448 .. 1,048,575]`)**: `swa_replay_tokens = 128`, `enc_cos = 1.00000000` (`max_abs = 0.000e+00`)
- **`40L` Decoder Continuation Parity Post-Hydration**: `dec_cos = 1.00000000` (`max_abs = 0.000e+00`)
