# Wave 9.2 (`SB-M4-P2-176G`) Asymmetric 4× MI50 + 2× Blackhole P150A `WHT128-HLUT3` Switchbox Promotion Report

**Architecture Baseline**: `DeepSeek-V4.1-Flash-1017` (`40` backbone layers + `3` DSpark MTP layers, `384` routed experts/layer, `top_k = 6`, `CSA2`, `Engram`, `DSML`, `DeepSeek-ViT`)  
**Hardware Fabric**: `SB-M4-P2-176G` (`176 GiB` total device memory = `3× 32 GiB MI50 [tp=0..2]` + `1× 16 GiB MI50 [hlut3_dev=3]` + `2× 32 GiB Tenstorrent Blackhole P150A` over `4× 800G FABRIC_1D`, with `188.83 GiB` host-locked DDR4 Engram tables)  
**Zero-PCIe-AtomicOp Invariant**: `100%` of inter-GPU synchronization uses local-HBM2 `__HIP_MEMORY_SCOPE_AGENT` counters + coherent host-DDR4 release/acquire stores (`0` `__HIP_MEMORY_SCOPE_SYSTEM` RMW atomics, `0` `slc` atomics across PEX88048).

---

## 1. Asymmetric 176 GiB Expert Tiering & `1,704`-Expert `TARQ` `RAREBAL-LDLQ` Calibration

| Tier | Placement | Format (`bpw`) | Experts Promoted | Weight Footprint | Role |
| :--- | :--- | :--- | ---: | ---: | :--- |
| **Tier-0 (MTP Drafter)** | `hlut3_dev=3` (16 GiB switchbox MI50) | Unsharded `WHT128-HLUT3` (`3.125 bpw`) | `384 / 384` (`100%`) | `4.94 GiB` | Eliminates TP3 AllReduce on MTP routed MoE; reclaims `+1,203.7 MiB/rank` on TP3 |
| **Tier-1 (Hottest Backbone)** | `hlut3_dev=3` (16 GiB switchbox MI50) | Unsharded `WHT128-HLUT3` (`3.125 bpw`) | `720` (`18/layer × 40`) | `9.27 GiB` | Parallel 4th-GPU MoE lane overlapping TP3 and 2× P150A |
| **Tier-2 (Warm-Hot Backbone)** | `TP3` (`tp=0..2` 32 GiB MI50s) | Sharded `WHT128-HLUT3` (`3.125 bpw`) | `600` (`15/layer × 40`) | `2.57 GiB/rank` | High-precision TP3 resident experts replacing compacted HLUT2 slots |
| **Tier-3 (Compacted MI50)** | `TP3` (`tp=0..2` 32 GiB MI50s) | Sharded `WHT128-HLUT2` (`2.125 bpw`) | `7,320` (`183/layer × 40`) | Compacted in-place | Zero-overhead fallback backbone experts |
| **Tier-4 (P150A Invariant)** | `2× Blackhole P150A` (`[216..383]`) | `WHT128-HLUT2` (`18/18` L1 + GDDR6) | `6,720` (`168/layer × 40`) | `64 GiB` GDDR6 | Preserved invariant across Wave 8.5 -> 9.2 -> 9.3 |

### Calibration Quality Receipts (`build_wave9_2_hlut3_overlay`, pure C++/HIP)
- **Total Promoted Experts**: `1,704` (`384` MTP + `720` `dev3` backbone + `600` `TP3` backbone)
- **Solver Outcomes**: `1,442` `EMITTED_RAREBAL` + `262` `FALLBACK_HLUT3_HI` (`0` dual-receipt violations)
- **Mean Reconstruction Fidelity**: Cosine `0.95379` (`HLUT2`) -> **`0.98672`** (`WHT128-HLUT3`), mean SNR gain **`+5.456 dB`**
- **Verified Overlay SHA-256**:
  - `${DSV41_WAVE9_OVERLAY_DIR}/wave9_2_dev3_hlut3.safetensors`: `e110c6e6a5bc8ee8bb6354846a19af88996e325d0ea280826eeff27890338e9d`
  - `${DSV41_WAVE9_OVERLAY_DIR}/wave9_2_tp3_hlut3.safetensors`: `bbe60fd632ecec378a4ef834c70ec0c919a861df2d0b49ec9cdd8995f2804de8`

---

## 2. Silicon Gate & 4-GPU VRAM Margin Verification (`1,048,576` Context + `DeepSeek-ViT`)

- **4-GPU Dispatch Parity (`verify_wave9_2_4gpu_dispatch`, `M = 1, 2, 4, 8, 16, 128`)**: `PASS` (`MTP_DEV3 cos = 1.000000000`, `3WAY_BB cos >= 0.99999925`, `tp_eq = 1`)
- **Full `1M` + `ViT` Silicon Gate (`verify_wave8_5_1m_e2e`)**: `PASS` (`125.47 s` wall time; `1M HSI top64 = 100.000%`, planted needles `1/1/1`, `CSA2 cos = 1.00000000`, `32`-burst decode to `pos = 1,048,575` @ `275.87 ms/step`, NVMe `.kvblk` `CRC32C = 1`, `enc_cos = 1.00000000`, `dec_cos = 1.00000000`, `M=512` Engram wait `0.0007 ms`, `M=1024` Engram wait `0.0014 ms`)
- **Live `:8042` 4-GPU HBM2 & Thermal Telemetry** (`1,048,576`-token ceiling + `DeepSeek-ViT` active, boot `25.67 s`):

| Device | Role | Used HBM2 (`MiB`) | Free HBM2 (`MiB`) | Total (`MiB`) | Peak Edge Temp (`°C`) | PCIe Link & AER (`UESta`/`CESta`) |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| `tp=0` (32G) | Backbone + 200 TP3 HLUT3 | `30,147` | `2,605` (`2,700` init) | `32,752` | `56.0` | Direct x16 (`0` errors) |
| `tp=1` (32G switchbox) | Backbone + 200 TP3 HLUT3 | `30,129` | `2,623` (`2,702` init) | `32,752` | `50.0` | `16GT/s x16 (ok)`, all `-` |
| `tp=2` (32G display) | Backbone + 200 TP3 HLUT3 + ViT | `30,231` | `2,521` (`3,128` init) | `32,752` | `56.0` | Direct x16 (`0` errors) |
| `hlut3_dev=3` (16G switchbox) | 384 MTP + 720 BB HLUT3 | `15,075` (`<= 15,360` cap) | `1,293` (`1,348` init) | `16,368` | `50.0` | `16GT/s x16 (ok)`, all `-` |

- **Kernel `dmesg` Health**: `0` `amdgpu` faults, `0` `RAS`/`ATHUB`/`CmpltAbrt` interrupts.

---

## 3. Live `:8042` Baseline Throughput Comparison (Wave 8.5 vs Wave 9.2)

| Workload (Live `:8042`, `effort=100`, `reuse=0` cold) | Wave 8.5 (`3× MI50`, `HLUT2`) | Wave 9.2 (`4× MI50`, `1,704 HLUT3`) | Notes |
| :--- | :--- | :--- | :--- |
| `1,564`-tok Cold Prefill (`req_1600_b`) | `485.7 tok/s` (`3,220 ms`) | `339.4 tok/s` (`4,608 ms`) | Unoptimized `dev3` serial HLUT3 prefill; target of Wave 9.3 |
| `2,400`-tok Cold Prefill (`req_1600`) | `485.7 tok/s` (`4,941 ms`) | `354.6 tok/s` (`6,767 ms`) | Unoptimized `dev3` serial HLUT3 prefill; target of Wave 9.3 |
| Decode (`1,564`-tok prompt, `gen=256`) | `39.03 tok/s` (`tau=3.85`, `72.3%`) | `30.28 tok/s` (`tau=3.43`, `64.3%`) | Unoptimized `dev3` pair-wise HLUT3 + host sync; target of Wave 9.3 |
| Decode (`2,400`-tok prompt, `gen=256`) | `32.15 tok/s` (`tau=3.23`, `61.1%`) | `29.27 tok/s` (`tau=3.27`, `64.1%`) | Higher DSpark acceptance (`+3.0 pp`) from `384/384` HLUT3 MTP |
| Multi-turn C++ Turn 1 (`prompt=124`, `gen=384`) | — | `23.66 tok/s` (`tau=2.99`, `60.9%`) | `ar = 3,949 ms` baseline |
| Multi-turn C++ Turn 2 (`prompt=590`, `gen=384`) | — | `32.87 tok/s` (`tau=4.12`, `75.5%`) | Strong `DSpark` draft acceptance (`75.5%`) |
| Free HBM2 on `tp=0/1/2` (`1M + ViT`) | `266 / 268 / 694 MiB` | **`2,700 / 2,702 / 3,128 MiB`** | **`+2.43 GiB/rank`** headroom unlocked on every 32 GiB MI50 |

- **Rollback Preservation**: Previous Wave 8.5 binary preserved at `${DSV41_ROOT}/bin/dsv41_fe.bak_wave8_5_noatomic`; verified Wave 9.2 binary preserved at `${DSV41_ROOT}/bin/dsv41_fe.bak_wave9_2_verified`.
