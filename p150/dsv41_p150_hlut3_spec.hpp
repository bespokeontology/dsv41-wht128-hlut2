#pragma once
// Wave 9 Deliverable D: TT-Metal compile-checked P150A TP2 HLUT3 superblock contract.
// Exact 6,400 B = 16 x 400 B superblock, 360 superblocks per gate/up/down shard, 1,459,200 B
// FABRIC_1D L1 bank fit math across 110 Tensix banks.
//
// The 128x128 superblock covers 4 x 4 32x32 subtiles, subtile order nt_sub*4 + kt_sub.
// TT-Metal InterleavedBuffer page ceiling: every gate/up/down shard is 360 superblocks.
// Hot-shard capacity for HLUT3 at the live FABRIC_1D bank ceiling is exactly 11 shards/card.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#if defined(DSV41_P150_TT_METAL)
#include <tt-metalium/tt_metal.hpp>
#endif

#include "../src/hlut3_pack.hpp"
#include "dsv41_p150_l1.hpp"

namespace dsv41 {

inline constexpr uint32_t kP150Hlut3TileRows       = 32;
inline constexpr uint32_t kP150Hlut3TileCols       = 32;
inline constexpr uint32_t kP150Hlut3TileBytes      = 400;
inline constexpr uint32_t kP150Hlut3SuperRows      = 128;
inline constexpr uint32_t kP150Hlut3SuperCols      = 128;
inline constexpr uint32_t kP150Hlut3SuperTiles     = 16;
inline constexpr uint32_t kP150Hlut3SuperBytes     = 6400;
inline constexpr uint32_t kP150Hlut3SuperPageWords = kP150Hlut3SuperBytes / 4u; // 1600 U32 words
inline constexpr uint32_t kP150Hlut3SuperPageBytes = kP150Hlut3SuperBytes;

// TP2 expert shard geometry: H=5120 -> I=2304 -> H=5120, split K/N by 2.
// gate/up: [K=5120, N_shard=1152], kt4=40, nt4=9 -> 360 superblocks.
// down:    [K=1152, N_shard=5120], kt4=9,  nt4=40 -> 360 superblocks.
inline constexpr uint32_t kP150Hlut3GateUpK       = 5120;
inline constexpr uint32_t kP150Hlut3GateUpNShard  = 1152;
inline constexpr uint32_t kP150Hlut3GateUpKt4     = kP150Hlut3GateUpK / 128;
inline constexpr uint32_t kP150Hlut3GateUpNt4     = kP150Hlut3GateUpNShard / 128;
inline constexpr uint32_t kP150Hlut3GateUpSuper   = kP150Hlut3GateUpKt4 * kP150Hlut3GateUpNt4;
static_assert(kP150Hlut3GateUpSuper == 360u, "P150 HLUT3 gate/up shard must be 360 superblocks");
static_assert(uint64_t(kP150Hlut3GateUpSuper) * kP150Hlut3SuperBytes == 2'304'000ULL,
              "P150 HLUT3 gate/up shard must equal exact 2,304,000 B = 1152 x 5120 x 3.125 bits");
static_assert(uint64_t(kP150Hlut3GateUpSuper) * kP150Hlut3SuperBytes ==
              uint64_t(kP150Hlut3GateUpNShard) * uint64_t(kP150Hlut3GateUpK) * 400ULL / 1024ULL,
              "P150 HLUT3 superblock bytes must be bit-exact 400 B/tile contraction");

inline constexpr uint32_t kP150Hlut3DownK        = 1152;
inline constexpr uint32_t kP150Hlut3DownNShard   = 5120;
inline constexpr uint32_t kP150Hlut3DownKt4      = kP150Hlut3DownK / 128;
inline constexpr uint32_t kP150Hlut3DownNt4      = kP150Hlut3DownNShard / 128;
inline constexpr uint32_t kP150Hlut3DownSuper    = kP150Hlut3DownKt4 * kP150Hlut3DownNt4;
static_assert(kP150Hlut3DownSuper == 360u, "P150 HLUT3 down shard must be 360 superblocks");
static_assert(uint64_t(kP150Hlut3DownSuper) * kP150Hlut3SuperBytes == 2'304'000ULL,
              "P150 HLUT3 down shard must equal exact 2,304,000 B = 5120 x 1152 x 3.125 bits");

inline constexpr uint32_t kP150Hlut3ShardSuperblocks = 360u;
inline constexpr uint32_t kP150Hlut3Projections      = 3u; // gate + up + down
inline constexpr uint64_t kP150Hlut3ExpertBytes =
    uint64_t(kP150Hlut3Projections) * uint64_t(kP150Hlut3ShardSuperblocks) * uint64_t(kP150Hlut3SuperBytes);
static_assert(kP150Hlut3ExpertBytes == 6'912'000ULL,
              "P150 HLUT3 TP2 expert (gate+up+down) must equal exact 6,912,000 B = 6.912 MB");
static_assert(kP150Hlut3ExpertBytes == 13'824'000ULL / 2ULL,
              "P150 HLUT3 TP2 expert must be exact half of the 13,824,000 B full TP3 expert");

inline constexpr size_t kP150Hlut3BaseNonHotL1Bytes =
    size_t(320u) * 2u +                             // slot_ids UINT16 (640 B)
    size_t(10u) * 512u * 1152u * 4u +               // gate_partials FP32 (23,592,960 B)
    size_t(10u) * 512u * 1152u * 4u +               // up_partials FP32 (23,592,960 B)
    size_t(1u) * 512u * 1152u * 2u +                // z_rot_bf16 (1,179,648 B)
    size_t(32u) * 5120u * 4u +                      // combined_f32 (655,360 B)
    size_t(40u) * 256u * 2u +                       // l1_slot_lut UINT16 (20,480 B)
    size_t(160u) * 2048u;                           // x_decode BF16 L1-resident (327,680 B)

inline constexpr size_t kP150Hlut3FabricCbGuardBytes =
    size_t(kP150L1CoresPerCard) * size_t(kP150L1Fabric1DCbGuardPerCore); // 28,835,840 B (256 KiB/core)

inline constexpr size_t p150_hlut3_fabric1d_per_bank_l1_bytes(uint32_t requested, bool include_guard = true) noexcept {
    const auto bank_pages = [](size_t pages, size_t page_bytes) constexpr -> size_t {
        return ((pages + 109u) / 110u) * page_bytes;
    };
    size_t b =
        bank_pages(1u, 640u) +                  // slot_ids
        bank_pages(5760u, 4096u) +              // gate_partials
        bank_pages(5760u, 4096u) +              // up_partials
        bank_pages(576u, 2048u) +               // z_rot_bf16
        bank_pages(160u, 4096u) +               // combined_f32
        bank_pages(40u, 512u) +                 // l1_slot_lut
        bank_pages(160u, 2048u);                // x_decode kept L1-resident for requested <= 17
    b += 3u * bank_pages(size_t(requested) * 360u, size_t(kP150Hlut3SuperBytes)); // gate+up+down superblocks
    if (include_guard) {
        const size_t guard_pages = (size_t(kP150L1CoresPerCard) * 256u * 1024u) / 4096u;
        b += bank_pages(guard_pages, 4096u);
    }
    return b;
}

inline constexpr size_t kP150Hlut3BankBytesAt11 =
    p150_hlut3_fabric1d_per_bank_l1_bytes(11u, true);
inline constexpr size_t kP150Hlut3BankBytesAt12 =
    p150_hlut3_fabric1d_per_bank_l1_bytes(12u, true);
static_assert(kP150Hlut3BankBytesAt11 == 1'413'248u,
              "HLUT3 11 hot shards + 256 KiB/core CB guard + L1 x_decode must equal exact 1,413,248 B");
static_assert(kP150Hlut3BankBytesAt11 <= kP150L1Fabric1DBankBytes,
              "HLUT3 11 hot shards must fit exact 1,459,200 B FABRIC_1D bank (+45,952 B margin)");
static_assert(kP150Hlut3BankBytesAt12 == 1'490'048u &&
              kP150Hlut3BankBytesAt12 > kP150L1Fabric1DBankBytes,
              "HLUT3 12 hot shards must exceed exact 1,459,200 B FABRIC_1D bank by 30,848 B");
static_assert(kP150L1Fabric1DBankBytes == 1'459'200u,
              "HLUT3 contract must bind the exact live 1,459,200 B FABRIC_1D main-L1 bank ceiling");

inline constexpr uint32_t kP150Hlut3MaxHotShards = 11u;
inline constexpr uint32_t kP150Hlut3Tp2HotShards = 2u * kP150Hlut3MaxHotShards; // 22 TP2 shards
inline constexpr size_t kP150Hlut3HotBytesPerCard =
    size_t(kP150Hlut3MaxHotShards) * size_t(kP150Hlut3ExpertBytes); // 76,032,000 B (72.52 MiB/card)
inline constexpr size_t kP150Hlut3HotBytesBothCards = 2u * kP150Hlut3HotBytesPerCard; // 152,064,000 B

static_assert(kP150Hlut3HotBytesPerCard == 76'032'000ULL,
              "HLUT3 11 hot TP2 experts/card must equal exact 76,032,000 B");
static_assert(kP150Hlut3Tp2HotShards == 22u,
              "HLUT3 11 shards/card x 2 cards must expose exactly 22 TP2 hot slots");

inline constexpr uint32_t kP150Hlut3SuperTilesPerProjection = 360u;
inline constexpr uint32_t kP150Hlut3SuperPagesPerProjection =
    (kP150Hlut3SuperTilesPerProjection + kP150L1CoresPerCard - 1u) / kP150L1CoresPerCard;
static_assert(kP150Hlut3SuperPagesPerProjection == 4u,
              "360 superblocks across 110 banks must map exactly 4 interleaved pages/bank/projection");
static_assert(uint64_t(kP150Hlut3SuperPagesPerProjection) * kP150L1CoresPerCard * kP150Hlut3SuperBytes ==
              uint64_t(kP150Hlut3SuperTilesPerProjection) * kP150Hlut3SuperBytes + 80u * 6400u,
              "TT-Metal interleaved page rounding must expose exactly 80 padded superblock pages (512,000 B)");

// TT-Metal L1 reader/unpacker view. The pointer is an L1 InterleavedBuffer page base; each
// 128x128 superblock is 6,400 B and the tile stride is exactly 400 B, no padding.
#if defined(DSV41_P150_TT_METAL)
struct P150Hlut3SuperblockL1Reader {
    tt::tt_metal::CoreCoord core{};
    const Hlut3Superblock128x128* l1_page = nullptr;
    uint32_t bank = 0;
    uint32_t projection = 0;
    uint32_t superblock_idx = 0;
    uint32_t superblocks_per_projection = kP150Hlut3SuperTilesPerProjection;
    uint32_t tile_stride = kP150Hlut3TileBytes;
};
#endif

inline const Hlut3Tile32x32& p150_hlut3_superblock_tile(
    const Hlut3Superblock128x128& sb,
    uint32_t nt_sub,
    uint32_t kt_sub) noexcept
{
    return sb.subtiles[nt_sub * 4u + kt_sub];
}

inline void p150_hlut3_unpack_superblock_tiles(
    const Hlut3Superblock128x128& sb,
    Hlut3Tile32x32 out_tiles[16]) noexcept
{
    for (uint32_t nt_sub = 0; nt_sub < 4u; ++nt_sub) {
        for (uint32_t kt_sub = 0; kt_sub < 4u; ++kt_sub) {
            out_tiles[nt_sub * 4u + kt_sub] = sb.subtiles[nt_sub * 4u + kt_sub];
        }
    }
}

inline void p150_hlut3_geometry_report() noexcept {
    std::printf("[P150_HLUT3] tile=%u B superblock=%u B superblocks/shard=%u shard=%llu B expert=%llu B\n",
                unsigned(kP150Hlut3TileBytes), unsigned(kP150Hlut3SuperBytes),
                unsigned(kP150Hlut3ShardSuperblocks),
                static_cast<unsigned long long>(2'304'000ULL),
                static_cast<unsigned long long>(kP150Hlut3ExpertBytes));
    std::printf("[P150_HLUT3] bank=%u hot_shards=%u bank_at11=%zu margin=%zu bank_at12=%zu overflow=%zu hot/card=%.2f MiB\n",
                unsigned(kP150L1Fabric1DBankBytes), unsigned(kP150Hlut3MaxHotShards),
                kP150Hlut3BankBytesAt11, size_t(kP150L1Fabric1DBankBytes) - kP150Hlut3BankBytesAt11,
                kP150Hlut3BankBytesAt12, kP150Hlut3BankBytesAt12 - kP150L1Fabric1DBankBytes,
                double(kP150Hlut3HotBytesPerCard) / (1024.0 * 1024.0));
    std::printf("[P150_HLUT3] pages/proj/bank=%u bank_pages=%u expert_stride=%u hot_tp2=%u\n",
                unsigned(kP150Hlut3SuperPagesPerProjection),
                unsigned((kP150Hlut3SuperTilesPerProjection + kP150L1CoresPerCard - 1u) / kP150L1CoresPerCard),
                unsigned(kP150Hlut3ExpertBytes), unsigned(kP150Hlut3Tp2HotShards));
    std::fflush(stdout);
}

}  // namespace dsv41
