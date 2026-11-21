# Wave 10 (`SB-M5-P2-192G`) — 1,048,576-Token (`1M`) End-to-End Silicon Verification Receipt

- **Runtime Topology**: `3x AMD Instinct MI50 32GB (gfx906 TP3)` + `2x AMD Instinct MI50 16GB (hlut3_dev=3 Switch #1 + hlut3_dev=4 Switch #2 WHT128-HLUT3)` + `2x Tenstorrent Blackhole P150A (TP2, 18/18 L1 + x16->FABRIC_1D)` + `188.83 GiB Locked DDR4 Engram` + `Dsv41VitEngine (tp=2)`
- **Active Overlay**: `${DSV41_ROOT}/packed_rarebal_overlay_wave8_5/hot_experts_wave8_5.safetensors`
- **Active `hlut3_dev=3` Overlay**: `${DSV41_WAVE10_OVERLAY_DIR}/wave10_dev3_hlut3.safetensors`
- **Active `hlut3_dev=4` Overlay**: `${DSV41_WAVE10_OVERLAY_DIR}/wave10_dev4_hlut3.safetensors`
- **Active `TP3` `HLUT3` Overlay**: `${DSV41_WAVE10_OVERLAY_DIR}/wave10_tp3_hlut3.safetensors`
- **Context Ceiling (`max_seq_len`)**: `1,048,576` tokens (`8,192` x `128`-token Radix/NVMe blocks)
- **Total E2E Gate Wall Time**: `68.81 s`

## 1. Simultaneous 1M VRAM Headroom & 64-Bit Pointer Audit

| Rank | Device Role | Free HBM2 (`MiB`) | Total HBM2 (`MiB`) | Headroom Gate | `uint64_t` Pointer Audit |
| :--- | :--- | ---: | ---: | :---: | :---: |
| `tp=0` | `Backbone` | `4184.00` | `32752.00` | `PASS` | `PASS` |
| `tp=1` | `Backbone` | `4186.00` | `32752.00` | `PASS` | `PASS` |
| `tp=2` | `Backbone + Dsv41VitEngine` | `4610.00` | `32752.00` | `PASS` | `PASS` |
| `hlut3_dev=3` | `Switch #1 Unsharded MTP (384) + Backbone (720) WHT128-HLUT3` | `1152.00` | `16368.00` | `PASS` | `PASS` |
| `hlut3_dev=4` | `Switch #2 Unsharded Backbone (1,080) WHT128-HLUT3` | `1578.00` | `16368.00` | `PASS` | `PASS` |

- **Tiered CSA2 Slot Partition (`tp=0..2`)**:
  - Slot `0` (`r=2`): `max_comp=524288` (`HBM2 hot=262144`, `Coherent DDR4 ext=262144`)
  - Slot `1` (`r=2`): `max_comp=524288` (`HBM2 hot=262144`, `Coherent DDR4 ext=262144`)
  - Slot `2` (`r=2`): `max_comp=524288` (`HBM2 hot=262144`, `Coherent DDR4 ext=262144`)
  - Slot `3` (`r=1`): `max_comp=1048576` (`HBM2 hot=524288`, `Coherent DDR4 ext=524288`)

## 2. Double-Buffered Prefill Engram Overlap & 1M HSI / CSA2 Numerical Stability

- **`M = 512` Prefill Double-Buffered Engram Wait**: `0.0006 ms/chunk` (`PASS`)
- **`M = 1024` Superchunk Prefill + P150A `M = 1024` `x16` Relay Wait**: `0.0017 ms/superchunk` (`PASS`)
- **1M Chunked Prefill (`[1,046,528 .. 1,048,543]`, `2,016` tokens)**: `6749.57 ms`, `int32_overflow_free = true`, `nonfinite_count = 0`
- **1M `HSI` `v_dot2_f32_f16` (`clamp=false`) vs `FP32` Top-64 Overlap (`8,192` blocks)**: `100.000%` (`>= 99.8%` gate)
- **Planted Needle Retrieval Across 1M Span**: `pos=128 (blk 1): HIT`, `pos=524,288 (blk 4096): HIT`, `pos=900,000 (blk 7031): HIT`
- **1M Tiered `CSA2` `v_dot2_f32_f16` vs `FP32` Output Parity**: `cosine = 1.00000000`, `max_abs_diff = 0.000e+00`

## 3. 32-Burst `Dynamic DeepSelect8` (`M = 8`) Decode Across `1,048,544 -> 1,048,575` (`1M - 1`)

- **Completed Bursts**: `32` | **Final Verified Position**: `1048575` (`1,048,575 == 1M - 1`) | **Total Emitted**: `54` tokens
- **Mean Burst Wall Time**: `130.16 ms/step` | **Aggregate 1M Throughput**: `12.97 tok/s`

| Burst | `start_pos` | `acc` | `ms/step` | `tok/s` | `attn` | `hsi` | `moe` | `p150w` | `ar` | `dsp` | `smp` | `egw` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `1048544` | 1 | 142.00 | 7.04 | 54.00 | 0.24 | 37.61 | 17.49 | 17.23 | 11.74 | 0.17 | 0.00 |
| 1 | `1048545` | 1 | 141.56 | 7.06 | 53.96 | 0.26 | 37.54 | 18.51 | 17.30 | 10.24 | 0.15 | 0.00 |
| 2 | `1048546` | 2 | 143.24 | 13.96 | 57.59 | 0.30 | 39.50 | 17.75 | 14.76 | 9.23 | 0.30 | 0.00 |
| 3 | `1048548` | 2 | 139.90 | 14.30 | 55.68 | 0.70 | 38.54 | 16.12 | 14.26 | 9.97 | 0.34 | 0.00 |
| 4 | `1048550` | 2 | 145.50 | 13.75 | 56.66 | 0.34 | 39.76 | 17.17 | 16.70 | 10.14 | 0.32 | 0.00 |
| 5 | `1048552` | 2 | 145.35 | 13.76 | 55.45 | 0.28 | 38.48 | 19.52 | 17.17 | 9.29 | 0.32 | 0.00 |
| 6 | `1048554` | 2 | 146.43 | 13.66 | 56.99 | 0.28 | 39.72 | 17.47 | 16.65 | 10.59 | 0.39 | 0.00 |
| 7 | `1048556` | 2 | 142.23 | 14.06 | 55.56 | 0.28 | 40.25 | 14.66 | 16.73 | 10.01 | 0.31 | 0.00 |
| 8 | `1048558` | 1 | 143.26 | 6.98 | 56.43 | 0.28 | 37.52 | 20.40 | 13.77 | 10.59 | 0.19 | 0.00 |
| 9 | `1048559` | 1 | 143.23 | 6.98 | 58.76 | 0.27 | 36.86 | 14.27 | 16.83 | 12.05 | 0.17 | 0.00 |
| 10 | `1048560` | 2 | 146.52 | 13.65 | 55.74 | 0.30 | 41.52 | 22.03 | 11.77 | 10.47 | 0.31 | 0.00 |
| 11 | `1048562` | 2 | 141.24 | 14.16 | 55.93 | 0.28 | 37.31 | 21.51 | 12.21 | 9.78 | 0.31 | 0.00 |
| 12 | `1048564` | 2 | 138.75 | 14.41 | 55.30 | 0.28 | 38.31 | 17.75 | 12.85 | 9.55 | 0.32 | 0.00 |
| 13 | `1048566` | 2 | 140.15 | 14.27 | 55.02 | 0.29 | 38.15 | 16.82 | 15.82 | 9.68 | 0.32 | 0.00 |
| 14 | `1048568` | 2 | 144.76 | 13.82 | 55.12 | 0.29 | 37.68 | 20.92 | 15.83 | 10.10 | 0.32 | 0.00 |
| 15 | `1048570` | 1 | 65.28 | 15.32 | 28.00 | 0.65 | 16.73 | 12.06 | 5.64 | 0.00 | 0.09 | 0.00 |
| 16 | `1048571` | 1 | 67.04 | 14.92 | 27.66 | 0.53 | 18.60 | 11.75 | 6.43 | 0.00 | 0.09 | 0.00 |
| 17 | `1048572` | 1 | 66.67 | 15.00 | 29.14 | 0.29 | 16.85 | 12.46 | 5.77 | 0.00 | 0.09 | 0.00 |
| 18 | `1048573` | 1 | 67.37 | 14.84 | 29.15 | 0.27 | 19.23 | 10.73 | 5.42 | 0.00 | 0.09 | 0.00 |
| 19 | `1048574` | 1 | 67.95 | 14.72 | 29.09 | 0.27 | 16.45 | 13.94 | 5.97 | 0.00 | 0.10 | 0.00 |
| 20 | `1048575` | 1 | 67.79 | 14.75 | 28.65 | 0.27 | 17.27 | 13.70 | 5.51 | 0.00 | 0.09 | 0.00 |
| 21 | `1048568` | 2 | 146.72 | 13.63 | 55.10 | 0.29 | 41.59 | 18.67 | 17.19 | 9.87 | 0.32 | 0.00 |
| 22 | `1048568` | 2 | 147.48 | 13.56 | 55.27 | 0.32 | 41.49 | 18.41 | 17.73 | 10.01 | 0.32 | 0.00 |
| 23 | `1048568` | 2 | 148.85 | 13.44 | 56.65 | 0.28 | 40.87 | 19.51 | 17.60 | 9.85 | 0.32 | 0.00 |
| 24 | `1048568` | 2 | 148.41 | 13.48 | 56.97 | 0.30 | 40.94 | 18.79 | 17.49 | 9.95 | 0.32 | 0.00 |
| 25 | `1048568` | 2 | 150.27 | 13.31 | 56.80 | 0.29 | 40.12 | 21.74 | 17.12 | 9.99 | 0.31 | 0.00 |
| 26 | `1048568` | 2 | 148.42 | 13.48 | 56.36 | 0.29 | 41.05 | 19.21 | 16.74 | 10.71 | 0.32 | 0.00 |
| 27 | `1048568` | 2 | 146.38 | 13.66 | 55.21 | 0.30 | 39.69 | 19.83 | 17.07 | 9.90 | 0.32 | 0.00 |
| 28 | `1048568` | 2 | 145.75 | 13.72 | 54.31 | 0.29 | 38.87 | 20.38 | 17.81 | 10.04 | 0.31 | 0.00 |
| 29 | `1048568` | 2 | 145.99 | 13.70 | 53.91 | 0.30 | 38.84 | 20.27 | 18.46 | 10.17 | 0.31 | 0.00 |
| 30 | `1048568` | 2 | 144.86 | 13.81 | 53.90 | 0.30 | 38.81 | 20.15 | 17.64 | 10.18 | 0.31 | 0.00 |
| 31 | `1048568` | 2 | 145.69 | 13.73 | 55.06 | 0.29 | 39.07 | 19.86 | 17.30 | 9.95 | 0.32 | 0.00 |

## 4. 1M Boundary & Tail Pure-Block NVMe `.kvblk` Cold Hydration & `Encoder SWA` Replay

- **Flushed Boundary + Tail Blocks**: `13` (`blocks {0, 1, 4095, 4096, 7031, 8184..8191}`)
- **On-Disk `.kvblk` Footprint**: `114496 B/block` (`576 B` header + `113,920 B` payload = `890 B/tok`), **`0 B` disk SWA**
- **Cold NVMe Hydration + Hardware `CRC32C`**: `PASS` (`3.21 ms` across `13` blocks)
- **`128`-Token `Encoder SWA Bounded Replay` (`[1,048,448 .. 1,048,575]`)**: `swa_replay_tokens = 128`, `enc_cos = 1.00000000` (`max_abs = 0.000e+00`)
- **`40L` Decoder Continuation Parity Post-Hydration**: `dec_cos = 1.00000000` (`max_abs = 0.000e+00`)
