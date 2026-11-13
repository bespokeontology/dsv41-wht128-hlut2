#pragma once
// Pure C++20 exact Tensix L1 geometry + two-tier configurable hot-shard capacity for
// 2x Tenstorrent Blackhole P150A (grid 11x10 = 110 Tensix cores per card).
//
// Per-core physical L1 (Blackhole TENSIX_L1_SIZE) = 1,572,864 B.
// ARC firmware/guard reserve = 73,728 B (72 KiB) -> exact usable 1,499,136 B/bank.
// Bottom CB headroom guard = 260 KiB/core -> 266,240 B/core (62 x 4352 B = 269,824 B/bank).
//
// Persistent per-card base non-hot workspace (71,234,432 B = 67.934 MiB/card):
//   x_tilized [1,16,32,5120] BF16  =  5,242,880 B (16-slot M>32 prefill gather arena)
//   weights_tiled [1,288,32,32] F32=  1,179,648 B
//   slot_ids [1,1,1,320] UINT16    =        640 B
//   gate_partials [1,10,512,1152]  = 23,592,960 B
//   up_partials   [1,10,512,1152]  = 23,592,960 B
//   z_rot_bf16    [1,1,512,1152]   =  1,179,648 B
//   combined_f32  [1,1,32,5120]    =    655,360 B
//   x_rm_l1       [1,1,512,5120]   =  5,242,880 B (M>32 prefill staging)
//   slot_meta_l1  [1,1,288,32]     =     36,864 B
//   l1_slot_lut   [1,40,256] U16   =     20,480 B
//   dummy_hot_l1  [1,1,1088] U32   =      4,352 B
//   arena512 y_accum_f32 [512,5120]= 10,485,760 B (M>32 prefill accumulator)
//
// Two-Tier GDDR6 MP Demotion Law:
//   - Base Tier (11..13 shards/card): 0 demotions (71,234,432 B non-hot L1 -> ceiling 13).
//   - Tier 1a (14..16 shards/card):
//       Demotes M > 32 prefill arenas + metadata (x_rm_l1 5.00 MiB + arena512 10.00 MiB +
//       16-slot prefill x_tilized 5.00 MiB + slot_meta_l1 36 KiB + dummy_hot_l1 4.25 KiB)
//       while keeping BOTH x_decode [1,1,32,5120] BF16 (327,680 B = 4,096 B/bank) AND
//       weights_tiled (1,179,648 B = 12,288 B/bank) in 20+ TB/s Tensix L1.
//   - Tier 1b (17 shards/card, Production Target = 17 -> 34 TP2 shards = 152.40 MiB L1):
//       Additionally places scalar combine weights_tiled AND the final PCIe host-read staging
//       output finish_decode[2] (untilize ROW_MAJOR: 32 x 10,240 B = 327,680 B = 10,240 B on bank 0-0)
//       in GDDR6 MP while keeping x_decode [1,1,32,5120] BF16 (327,680 B = 4,096 B/bank) +
//       finish_decode[0..1] (typecast + FABRIC_1D all_reduce = 655,360 B = 8,192 B/bank)
//       100% L1-resident across all 40 decode layers!
//       Exact standalone per-bank usage at 17 shards + 62-page CB guard = 1,460,864 B <= 1,499,136 B/bank;
//       exact FABRIC_1D (l1_small_size = 2,048 B -> 1,459,200 B/bank) probe usage = 1,453,184 B <= 1,459,200 B/bank
//       (post-trace runtime bank 0-0 lowest L1 address = 367,232 B > 358,400 B static CB end = +8,832 B margin;
//        pure L1-resident workspace/card = 129,927,808 B = 123.91 MiB/card).
//   - Tier 2 (18 shards/card, Optional Max Ceiling):
//       Additionally demotes x_decode to mesh-replicated GDDR6 MP (49,083,264 B non-hot L1 -> ceiling 18).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace dsv41 {

inline constexpr uint32_t kP150L1CoreRawBytes          = 1'572'864u;
inline constexpr uint32_t kP150L1FirmwareGuardBytes    = 73'728u;
inline constexpr uint32_t kP150L1CoreUsableBytes       = 1'499'136u;
inline constexpr uint32_t kP150L1Fabric1DReserveBytes  = 37'888u;    // FABRIC_1D router reserve (CB base = 111,616 B)
inline constexpr uint32_t kP150L1SmallSizeBytes        = 2'048u;     // Isolates ttnn::all_reduce CCL semaphores out of main L1
inline constexpr uint32_t kP150L1Fabric1DBankBytes     = 1'459'200u; // Exact live FABRIC_1D main-L1 bank ceiling (1,499,136 - 37,888 - 2,048)
inline constexpr uint32_t kP150L1PeakStaticCbPerCore   = 246'784u;   // Peak static CB across all kernels (is_down: ends at 358,400 B)
inline constexpr uint32_t kP150L1CbGuardPerCore        = 266'240u;   // 260 KiB/core standalone guard
inline constexpr uint32_t kP150L1Fabric1DCbGuardPerCore= 262'144u;   // 256 KiB/core (covers 246,784 B down CB + 12,288 B finish_decode)
inline constexpr uint32_t kP150L1CoresPerCard          = 110u;

inline constexpr size_t kP150L1CbGuardBytesPerCard = size_t(kP150L1CoresPerCard) * size_t(kP150L1CbGuardPerCore);
inline constexpr size_t kP150L1UsableBytesPerCard  = size_t(kP150L1CoresPerCard) * size_t(kP150L1CoreUsableBytes);

inline constexpr size_t kP150HotShardBytes = size_t(360u) * size_t(4352u) * 3u; // gate+up+down = 4,700,160 B

inline constexpr size_t kP150DecodeActL1Bytes =
    size_t(1u) * 1u * 32u * 5120u * 2u;             // x_decode [1,1,32,5120] BF16 = 327,680 B

inline constexpr size_t kP150FinishDecodeL1Bytes =
    size_t(3u) * kP150DecodeActL1Bytes;             // finish_decode[0..2] BF16 L1 (11..16 shards) = 983,040 B

inline constexpr size_t kP150Tier1At17FinishL1Bytes =
    size_t(2u) * kP150DecodeActL1Bytes;             // finish_decode[0..1] L1 at 17 shards (untilize[2] in GDDR6) = 655,360 B

inline constexpr size_t kP150BaseNonHotL1Bytes =
    size_t(1u) * 16u * 32u * 5120u * 2u +           // x_tilized BF16 (5,242,880 B)
    size_t(288u) * 32u * 32u * 4u +                 // weights_tiled FP32 (1,179,648 B)
    size_t(320u) * 2u +                             // slot_ids UINT16 (640 B)
    size_t(10u) * 512u * 1152u * 4u +               // gate_partials FP32 (23,592,960 B)
    size_t(10u) * 512u * 1152u * 4u +               // up_partials FP32 (23,592,960 B)
    size_t(1u) * 512u * 1152u * 2u +                // z_rot_bf16 (1,179,648 B)
    size_t(32u) * 5120u * 4u +                      // combined_f32 (655,360 B)
    size_t(512u) * 5120u * 2u +                     // x_rm_l1 BF16 (5,242,880 B)
    size_t(288u) * 32u * 4u +                       // slot_meta_l1 UINT32 (36,864 B)
    size_t(40u) * 256u * 2u +                       // l1_slot_lut UINT16 (20,480 B)
    size_t(1088u) * 4u +                            // dummy_hot_l1 UINT32 (4,352 B)
    size_t(512u) * 5120u * 4u;                      // arena512 y_accum_f32 (10,485,760 B)

// Tier 1a (14..16 shards/card): demotes M > 32 prefill arenas, keeps x_decode + weights_tiled in L1.
inline constexpr size_t kP150PrefillOnlyDemotableBytes =
    size_t(512u) * 5120u * 2u +                     // x_rm_l1 -> replicated GDDR6 (5,242,880 B)
    size_t(512u) * 5120u * 4u;                      // arena512 -> sharded GDDR6 (10,485,760 B)

inline constexpr size_t kP150Tier1NonHotL1Bytes =
    kP150BaseNonHotL1Bytes - kP150PrefillOnlyDemotableBytes -
    (size_t(1u) * 16u * 32u * 5120u * 2u) + kP150DecodeActL1Bytes; // 50,590,592 B (Tier 1a nominal)

// Exact pure L1-resident bytes/card at production 17 shards/card (x_decode + finish_decode[0..1] + 17 hot shards, 0 DRAM buffers):
inline constexpr size_t kP150Tier1At17PureL1BytesPerCard =
    size_t(320u) * 2u +                             // slot_ids (640 B)
    kP150DecodeActL1Bytes +                         // x_decode (327,680 B)
    size_t(10u) * 512u * 1152u * 4u +               // gate_partials (23,592,960 B)
    size_t(10u) * 512u * 1152u * 4u +               // up_partials (23,592,960 B)
    size_t(1u) * 512u * 1152u * 2u +                // z_rot_bf16 (1,179,648 B)
    size_t(32u) * 5120u * 4u +                      // combined_f32 (655,360 B)
    kP150Tier1At17FinishL1Bytes +                   // finish_decode[0..1] (655,360 B)
    size_t(40u) * 256u * 2u +                       // l1_slot_lut (20,480 B)
    size_t(17u) * kP150HotShardBytes;               // 17 hot shards (79,902,720 B) -> total 129,927,808 B (123.909 MiB/card)

// Tier 2 (18 shards/card): full four-tensor GDDR6 demotion.
inline constexpr size_t kP150DemotableL1Bytes =
    size_t(1u) * 16u * 32u * 5120u * 2u +           // x_tilized -> replicated GDDR6
    size_t(288u) * 32u * 32u * 4u +                 // weights_tiled -> replicated GDDR6
    size_t(512u) * 5120u * 2u +                     // x_rm_l1 -> replicated GDDR6
    size_t(512u) * 5120u * 4u;                      // arena512 -> sharded GDDR6

inline constexpr size_t kP150ExpandedNonHotL1Bytes = kP150BaseNonHotL1Bytes - kP150DemotableL1Bytes;

inline constexpr uint32_t kP150MinHotShards              = 11u;
inline constexpr uint32_t kP150DefaultHotShards          = 17u;
inline constexpr uint32_t kP150ExpandedDemotionThreshold = 13u;
inline constexpr uint32_t kP150Tier1Ceiling              = 17u;

inline constexpr uint32_t kP150DefaultCeiling = static_cast<uint32_t>(
    (kP150L1UsableBytesPerCard - kP150L1CbGuardBytesPerCard - kP150BaseNonHotL1Bytes) / kP150HotShardBytes);

inline constexpr uint32_t kP150ExpandedCeiling = static_cast<uint32_t>(
    (kP150L1UsableBytesPerCard - kP150L1CbGuardBytesPerCard - kP150ExpandedNonHotL1Bytes) / kP150HotShardBytes);

inline constexpr uint32_t kP150MaxHotL1Shards = std::min<uint32_t>(kP150ExpandedCeiling, 18u);

static_assert(kP150L1CoreRawBytes - kP150L1FirmwareGuardBytes == kP150L1CoreUsableBytes,
              "P150 L1 firmware guard must reduce raw 1,572,864 B to exact 1,499,136 B");
static_assert(kP150DefaultCeiling == 13u,
              "P150 default non-hot geometry must fit exactly 13 hot shards/card");
static_assert((kP150L1UsableBytesPerCard - kP150L1CbGuardBytesPerCard -
               (kP150BaseNonHotL1Bytes - kP150PrefillOnlyDemotableBytes)) / kP150HotShardBytes == 17u,
              "P150 two-buffer prefill demotion must yield 80,112,768 B / 4,700,160 B == 17 shards/card");
static_assert(kP150ExpandedCeiling == 18u,
              "P150 Tier-2 GDDR6 demotion geometry must fit exact zero-spill 18 hot shards/card");

#if !defined(P150_MAX_HOT_L1_SHARDS)
#define P150_MAX_HOT_L1_SHARDS 17
#endif

inline constexpr uint32_t kP150ConfiguredHotShards =
    static_cast<uint32_t>(P150_MAX_HOT_L1_SHARDS);

static_assert(P150_MAX_HOT_L1_SHARDS >= 11 && P150_MAX_HOT_L1_SHARDS <= 18,
              "P150_MAX_HOT_L1_SHARDS must stay in [11..18]");
static_assert(kP150ConfiguredHotShards <= kP150MaxHotL1Shards,
              "P150_MAX_HOT_L1_SHARDS exceeds exact zero-spill hardware ceiling");

inline constexpr bool p150_requires_prefill_demotion(uint32_t requested) noexcept {
    return requested > kP150ExpandedDemotionThreshold && requested <= kP150MaxHotL1Shards;
}

inline constexpr bool p150_requires_decode_demotion(uint32_t requested) noexcept {
    return requested > kP150Tier1Ceiling && requested <= kP150MaxHotL1Shards;
}

inline constexpr bool p150_requires_weights_demotion(uint32_t requested) noexcept {
    return requested >= kP150Tier1Ceiling && requested <= kP150MaxHotL1Shards;
}

// Exact TT-Metal InterleavedBuffer per-bank page ceiling across 110 L1 banks (standalone 260 KiB/core guard).
inline constexpr size_t p150_per_bank_l1_bytes(uint32_t requested, bool include_guard = true) noexcept {
    const auto bank_pages = [](size_t pages, size_t page_bytes) constexpr -> size_t {
        return ((pages + 109u) / 110u) * page_bytes;
    };
    const bool prefill_demote = p150_requires_prefill_demotion(requested);
    const bool decode_demote  = p150_requires_decode_demotion(requested);
    const bool weights_demote = p150_requires_weights_demotion(requested);
    size_t b =
        bank_pages(1u, 640u) +                  // slot_ids
        bank_pages(5760u, 4096u) +              // gate_partials
        bank_pages(5760u, 4096u) +              // up_partials
        bank_pages(576u, 2048u) +               // z_rot_bf16
        bank_pages(160u, 4096u) +               // combined_f32
        bank_pages(40u, 512u);                  // l1_slot_lut
    if (!prefill_demote) {
        b += bank_pages(288u, 128u);            // slot_meta_l1 in L1 only when !prefill_demote
        b += bank_pages(1u, 4352u);             // dummy_hot_l1 in L1 only when !prefill_demote
        b += bank_pages(2560u, 2048u);          // 16-slot x_tilized in L1
        b += bank_pages(2560u, 2048u);          // x_rm_l1 in L1
        b += bank_pages(2560u, 4096u);          // arena512 y_accum_f32 in L1
    } else if (!decode_demote) {
        b += bank_pages(160u, 2048u);           // 1-slot x_decode stays 100% L1-resident at 14..17!
    }
    if (!weights_demote) {
        b += bank_pages(288u, 4096u);           // weights_tiled in L1 for 11..16
    }
    b += 3u * bank_pages(size_t(requested) * 360u, 4352u); // hot gate+up+down
    if (include_guard) {
        const size_t guard_pages = (size_t(kP150L1CoresPerCard) * 260u * 1024u) / 4352u;
        b += bank_pages(guard_pages, 4352u);
    }
    return b;
}

static_assert(p150_per_bank_l1_bytes(17u, true) == 1'460'864u &&
              p150_per_bank_l1_bytes(17u, true) <= kP150L1CoreUsableBytes,
              "17 hot shards + 260 KiB/core CB guard + L1 x_decode must equal 1,460,864 B <= 1,499,136 B/bank");

// Exact live FABRIC_1D (l1_small_size = 2,048 B) per-bank L1 allocation across 110 banks (1,459,200 B/bank).
inline constexpr size_t p150_fabric1d_per_bank_l1_bytes(uint32_t requested, bool include_guard = true) noexcept {
    const auto bank_pages = [](size_t pages, size_t page_bytes) constexpr -> size_t {
        return ((pages + 109u) / 110u) * page_bytes;
    };
    const bool prefill_demote = p150_requires_prefill_demotion(requested);
    const bool decode_demote  = p150_requires_decode_demotion(requested);
    const bool weights_demote = p150_requires_weights_demotion(requested);
    size_t b =
        bank_pages(1u, 640u) +                  // slot_ids
        bank_pages(5760u, 4096u) +              // gate_partials
        bank_pages(5760u, 4096u) +              // up_partials
        bank_pages(576u, 2048u) +               // z_rot_bf16
        bank_pages(160u, 4096u) +               // combined_f32
        bank_pages(40u, 512u);                  // l1_slot_lut
    if (!prefill_demote) {
        b += bank_pages(288u, 128u);            // slot_meta_l1 in L1 only when !prefill_demote
        b += bank_pages(1u, 4352u);             // dummy_hot_l1 in L1 only when !prefill_demote
        b += bank_pages(2560u, 2048u);          // 16-slot x_tilized in L1
        b += bank_pages(2560u, 2048u);          // x_rm_l1 in L1
        b += bank_pages(2560u, 4096u);          // arena512 y_accum_f32 in L1
    } else if (!decode_demote) {
        b += bank_pages(160u, 2048u);           // 1-slot x_decode stays 100% L1-resident at 14..17!
    }
    if (!weights_demote) {
        b += bank_pages(288u, 4096u);           // weights_tiled in L1 for 11..16
    }
    b += 3u * bank_pages(size_t(requested) * 360u, 4352u); // hot gate+up+down
    if (include_guard) {
        const size_t guard_pages = (size_t(kP150L1CoresPerCard) * 256u * 1024u) / 4096u;
        b += bank_pages(guard_pages, 4096u);
    }
    return b;
}

static_assert(kP150Tier1At17PureL1BytesPerCard == 129'927'808ULL,
              "Pure L1-resident workspace at 17 hot shards must equal exact 129,927,808 B (123.909 MiB/card)");
static_assert(kP150L1CoreUsableBytes - kP150L1Fabric1DReserveBytes - kP150L1SmallSizeBytes == kP150L1Fabric1DBankBytes,
              "FABRIC_1D router reserve (37,888 B) + l1_small_size (2,048 B) must reduce 1,499,136 B to exact 1,459,200 B/bank");
static_assert(p150_fabric1d_per_bank_l1_bytes(17u, false) == 1'191'040u,
              "17 hot shards + L1 x_decode must allocate exact 1,191,040 B/bank before finish_decode (367,232 B > 358,400 B CB end)");
static_assert(p150_fabric1d_per_bank_l1_bytes(17u, true) == 1'453'184u &&
              p150_fabric1d_per_bank_l1_bytes(17u, true) <= kP150L1Fabric1DBankBytes,
              "17 hot shards + 256 KiB/core CB guard + L1 x_decode must equal 1,453,184 B <= 1,459,200 B FABRIC_1D bank");

inline constexpr bool p150_requires_demotion(uint32_t requested) noexcept {
    return p150_requires_prefill_demotion(requested);
}

inline constexpr bool p150_hot_shards_fit(uint32_t requested) noexcept {
    if (requested < 1u || requested > 18u) return false;
    if (requested <= kP150ExpandedDemotionThreshold) return requested <= kP150DefaultCeiling;
    if (requested <= kP150Tier1Ceiling) return p150_fabric1d_per_bank_l1_bytes(requested, true) <= kP150L1Fabric1DBankBytes;
    return requested <= kP150ExpandedCeiling;
}

static_assert(p150_hot_shards_fit(kP150ConfiguredHotShards),
              "P150_MAX_HOT_L1_SHARDS does not fit the exact per-Tensix L1 byte budget");

inline void p150_l1_geometry_report() noexcept {
    std::printf("[P150_L1GEOM] raw/core=%u firmware=%u usable/core=%u l1_small=%u fabric1d_bank=%u guard/core=%u cores/card=%u\n",
                unsigned(kP150L1CoreRawBytes), unsigned(kP150L1FirmwareGuardBytes),
                unsigned(kP150L1CoreUsableBytes), unsigned(kP150L1SmallSizeBytes),
                unsigned(kP150L1Fabric1DBankBytes),
                unsigned(kP150L1Fabric1DCbGuardPerCore), unsigned(kP150L1CoresPerCard));
    std::printf("[P150_L1GEOM] usable/card=%.3f MiB guard/card=%.3f MiB base_nonhot=%.3f MiB tier1_nonhot=%.3f MiB tier2_nonhot=%.3f MiB pure_L1_17=%.3f MiB shard=%.3f MiB\n",
                double(kP150L1UsableBytesPerCard) / (1024.0 * 1024.0),
                double(size_t(kP150L1CoresPerCard) * kP150L1Fabric1DCbGuardPerCore) / (1024.0 * 1024.0),
                double(kP150BaseNonHotL1Bytes) / (1024.0 * 1024.0),
                double(kP150Tier1NonHotL1Bytes) / (1024.0 * 1024.0),
                double(kP150ExpandedNonHotL1Bytes) / (1024.0 * 1024.0),
                double(kP150Tier1At17PureL1BytesPerCard) / (1024.0 * 1024.0),
                double(kP150HotShardBytes) / (1024.0 * 1024.0));
    std::printf("[P150_L1GEOM] configured=%u default_ceiling=%u tier1_ceiling=%u expanded_ceiling=%u prefill_demote=%d decode_demote=%d weights_demote=%d fabric1d_bank_17=%zu\n",
                unsigned(kP150ConfiguredHotShards), unsigned(kP150DefaultCeiling),
                unsigned(kP150Tier1Ceiling), unsigned(kP150ExpandedCeiling),
                int(p150_requires_prefill_demotion(kP150ConfiguredHotShards)),
                int(p150_requires_decode_demotion(kP150ConfiguredHotShards)),
                int(p150_requires_weights_demotion(kP150ConfiguredHotShards)),
                p150_fabric1d_per_bank_l1_bytes(17u, true));
    std::fflush(stdout);
}

}  // namespace dsv41
