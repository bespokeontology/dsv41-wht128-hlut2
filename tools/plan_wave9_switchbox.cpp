// Wave 9 Deliverable E: deterministic PCIe switchbox planner for 4x-7x MI50 + 2x-3x P150A.
//
// Prints the exact VRAM / TP3 / TP2 slot topology for every switchbox configuration, the
// HLUT3 vs HLUT2 TP3 per-rank promotion bytes, top-N DSpark MTP HLUT3 overlay bytes, and the
// exact P150A 6,400 B superblock L1 bank fit at the 11-shard/card production ceiling.
//
// All arithmetic is integer-exact bytes; MiB figures are 1/2^20 exact divisions. No Python.
// No HIP allocation; this planner is a pure host C++17 binary (0 B HBM2).

#include "../src/hlut3_pack.hpp"
#include "../p150/dsv41_p150_hlut3_spec.hpp"
#include "../p150/dsv41_p150_l1.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// Wave-9 TP3 per-rank promotion contract bytes (single routed expert gate+up+down slice):
//   HLUT3 = 295/64 MiB = 295 * 16,384 B = 4,833,280 B = 4.609375 MiB exactly
//   HLUT2 = 12839/4096 MiB = 12,839 * 256 B = 3,286,784 B = 3.134521484375 MiB exactly
inline constexpr uint64_t kHlut3PerRankTp3Bytes = 295ULL * 16384ULL;
inline constexpr uint64_t kHlut2PerRankTp3Bytes = 12'839ULL * 256ULL;
inline constexpr uint64_t kPromotionDeltaBytes  = kHlut3PerRankTp3Bytes - kHlut2PerRankTp3Bytes;
inline constexpr uint64_t kTop32MtpDeltaBytes   = 32ULL * kPromotionDeltaBytes;

static_assert(kHlut3PerRankTp3Bytes == 4'833'280ULL, "HLUT3 TP3 per-rank expert must be 4,833,280 B");
static_assert(kHlut2PerRankTp3Bytes == 3'286'784ULL, "HLUT2 TP3 per-rank expert must be 3,286,784 B");
static_assert(kPromotionDeltaBytes == 1'546'496ULL, "HLUT3 promotion delta must be exactly 1,546,496 B");
static_assert(kTop32MtpDeltaBytes == 49'487'872ULL, "top-32 MTP HLUT3 promotion must be 49,487,872 B");
static_assert(kHlut3PerRankTp3Bytes == 295ULL * (1ULL << 14), "HLUT3 = 295/64 MiB exact identity");
static_assert(kHlut2PerRankTp3Bytes == 12'839ULL * (1ULL << 8), "HLUT2 = 12839/4096 MiB exact identity");
static_assert(dsv41::kHlut3TileBytes == 400, "HLUT3 tile plane must stay exact 400 B");
static_assert(dsv41::kHlut3SuperBytes == 6400, "HLUT3 P150A superblock must stay exact 6,400 B");
static_assert(dsv41::kP150Hlut3ExpertBytes == 6'912'000ULL, "P150A TP2 HLUT3 expert must be 6,912,000 B");
static_assert(dsv41::kP150Hlut3MaxHotShards == 11u, "P150A HLUT3 L1 hot-shard ceiling must be 11/card");
static_assert(dsv41::kP150Hlut3BankBytesAt11 == 1'413'248u &&
              dsv41::kP150Hlut3BankBytesAt11 <= dsv41::kP150L1Fabric1DBankBytes,
              "11 HLUT3 hot shards must fit the exact 1,459,200 B FABRIC_1D main-L1 bank");

constexpr uint64_t kGib = 1ULL << 30;
constexpr uint64_t kCardBytes = 32ULL * kGib;  // MI50 HBM2 32 GiB, P150A GDDR6 32 GiB per card
constexpr int kMtpOverlayNs[3] = {8, 16, 32};

double bytes_to_mib(uint64_t b) { return double(b) / (1024.0 * 1024.0); }

struct ConfigRow {
    int mi50 = 0;
    int p150 = 0;
    uint64_t mi50_bytes = 0;
    uint64_t p150_bytes = 0;
    uint64_t mi50_gib = 0;
    uint64_t p150_gib = 0;
    int tp3_groups = 0;
    int tp3_rank_slots = 0;
    int tp2_groups = 0;
    int tp2_rank_slots = 0;
    uint64_t tp3_raw_expert_slots = 0;
    uint64_t tp2_raw_expert_slots = 0;
    uint64_t p150_hot_shards = 0;
    uint64_t mtp_t32_total_bytes = 0;
};

std::string config_id(int mi50, int p150) {
    char b[64];
    std::snprintf(b, sizeof(b), "SB-M%d-P%d", mi50, p150);
    return std::string(b);
}

}  // namespace

int main() {
    std::printf("=== WAVE9_SWITCHBOX_PLAN (deterministic integer-exact bytes) ===\n");
    std::printf("[CONTRACT] tile=%u B superblock=%u B subtiles=%u stride=%u\n",
                unsigned(dsv41::kHlut3TileBytes), unsigned(dsv41::kHlut3SuperBytes),
                unsigned(dsv41::kHlut3SuperTiles), unsigned(dsv41::kHlut3TileBytes));
    std::printf("[CONTRACT] tp3_rank_hlut3=%llu B = %.6f MiB (4.609 MiB)\n",
                static_cast<unsigned long long>(kHlut3PerRankTp3Bytes),
                bytes_to_mib(kHlut3PerRankTp3Bytes));
    std::printf("[CONTRACT] tp3_rank_hlut2=%llu B = %.6f MiB (3.134 MiB)\n",
                static_cast<unsigned long long>(kHlut2PerRankTp3Bytes),
                bytes_to_mib(kHlut2PerRankTp3Bytes));
    std::printf("[CONTRACT] promotion_delta=%llu B = %.6f MiB (approx +1.475 MiB/rank)\n",
                static_cast<unsigned long long>(kPromotionDeltaBytes),
                bytes_to_mib(kPromotionDeltaBytes));
    std::printf("[CONTRACT] tp2_expert=%llu B = %.6f MiB (360 superblocks x 3 projections)\n",
                static_cast<unsigned long long>(dsv41::kP150Hlut3ExpertBytes),
                bytes_to_mib(dsv41::kP150Hlut3ExpertBytes));
    std::printf("[CONTRACT] tp3_full_expert=%llu B tp2_full_expert=%llu B\n",
                static_cast<unsigned long long>(3ULL * kHlut3PerRankTp3Bytes),
                static_cast<unsigned long long>(dsv41::kP150Hlut3ExpertBytes));

    std::printf("\n=== TOP-N DSpark MTP HLUT3 OVERLAY PROMOTION (per TP3 rank) ===\n");
    std::printf("%-8s | %12s | %12s | %12s | %10s\n", "N", "bytes", "MiB exact", "MiB display", "note");
    std::printf("---------+--------------+--------------+--------------+------------\n");
    for (int n : kMtpOverlayNs) {
        const uint64_t b = uint64_t(n) * kPromotionDeltaBytes;
        std::printf("%-8d | %12llu | %12.6f | %12.3f | %10s\n",
                    n, static_cast<unsigned long long>(b), bytes_to_mib(b), bytes_to_mib(b),
                    (n == 32) ? "top-32 PASS" : "subset");
    }
    std::printf("[MTP] top-32 = %llu B = %.6f MiB exact = approx +47.19 MiB/rank PASS\n",
                static_cast<unsigned long long>(kTop32MtpDeltaBytes),
                bytes_to_mib(kTop32MtpDeltaBytes));

    std::vector<ConfigRow> rows;
    for (int mi50 = 4; mi50 <= 7; ++mi50) {
        for (int p150 = 2; p150 <= 3; ++p150) {
            ConfigRow r;
            r.mi50 = mi50;
            r.p150 = p150;
            r.mi50_bytes = uint64_t(mi50) * kCardBytes;
            r.p150_bytes = uint64_t(p150) * kCardBytes;
            r.mi50_gib = uint64_t(mi50) * 32ULL;
            r.p150_gib = uint64_t(p150) * 32ULL;
            r.tp3_groups = mi50 / 3;
            r.tp3_rank_slots = mi50;
            r.tp2_groups = p150 / 2;
            r.tp2_rank_slots = p150;
            r.tp3_raw_expert_slots = r.mi50_bytes / (3ULL * kHlut3PerRankTp3Bytes);
            r.tp2_raw_expert_slots = r.p150_bytes / dsv41::kP150Hlut3ExpertBytes;
            r.p150_hot_shards = uint64_t(p150) * uint64_t(dsv41::kP150Hlut3MaxHotShards);
            r.mtp_t32_total_bytes = uint64_t(mi50) * kTop32MtpDeltaBytes;
            rows.push_back(r);
        }
    }

    std::printf("\n=== VRAM & TP3/TP2 RANK SLOT TABLE (8 PCIe switchbox configs) ===\n");
    std::printf("%-11s | %4s | %4s | %5s | %6s | %6s | %7s | %3s | %4s | %3s | %4s\n",
                "cfg", "MI50", "P150", "PCIe", "M50GiB", "P150GiB", "TotalGiB",
                "TP3g", "TP3r", "TP2g", "TP2r");
    std::printf("------------+------+------+-------+--------+--------+----------+------+------+------+------\n");
    for (const auto& r : rows) {
        std::printf("%-11s | %4d | %4d | %5d | %6llu | %6llu | %7llu | %3d | %4d | %3d | %4d\n",
                    config_id(r.mi50, r.p150).c_str(), r.mi50, r.p150, r.mi50 + r.p150,
                    static_cast<unsigned long long>(r.mi50_gib),
                    static_cast<unsigned long long>(r.p150_gib),
                    static_cast<unsigned long long>(r.mi50_gib + r.p150_gib),
                    r.tp3_groups, r.tp3_rank_slots, r.tp2_groups, r.tp2_rank_slots);
    }
    std::printf("[SLOTS] TP3 rank slots = MI50 count (one TP3 slice/rank); TP2 rank slots = P150A count.\n");
    std::printf("[SLOTS] TP3 groups = floor(MI50/3); TP2 groups = floor(P150A/2); PCIe slots = MI50+P150A.\n");

    std::printf("\n=== RAW VRAM EXPERT CEILING + P150A L1 BANK FIT + MTP OVERLAY TABLE ===\n");
    std::printf("%-11s | %9s | %9s | %7s | %8s | %8s | %5s | %8s | %8s | %8s\n",
                "cfg", "TP3raw", "TP2raw", "L1hot", "L1bank", "L1margin", "fit", "T8MiB", "T16MiB", "T32MiB");
    std::printf("------------+-----------+-----------+---------+----------+----------+-------+----------+----------+----------\n");
    bool all_fit = true;
    for (const auto& r : rows) {
        const uint64_t t8 = 8ULL * kPromotionDeltaBytes;
        const uint64_t t16 = 16ULL * kPromotionDeltaBytes;
        const uint64_t t32 = 32ULL * kPromotionDeltaBytes;
        const size_t margin = size_t(dsv41::kP150L1Fabric1DBankBytes) - size_t(dsv41::kP150Hlut3BankBytesAt11);
        const bool fit = (dsv41::kP150Hlut3BankBytesAt11 <= dsv41::kP150L1Fabric1DBankBytes);
        all_fit &= fit;
        std::printf("%-11s | %9llu | %9llu | %7llu | %8zu | %8zu | %5s | %8.6f | %8.6f | %8.6f\n",
                    config_id(r.mi50, r.p150).c_str(),
                    static_cast<unsigned long long>(r.tp3_raw_expert_slots),
                    static_cast<unsigned long long>(r.tp2_raw_expert_slots),
                    static_cast<unsigned long long>(r.p150_hot_shards),
                    size_t(dsv41::kP150Hlut3BankBytesAt11), margin, fit ? "PASS" : "FAIL",
                    bytes_to_mib(t8), bytes_to_mib(t16), bytes_to_mib(t32));
    }
    std::printf("[RAW] TP3raw = MI50 HBM2 / 14,499,840 B full TP3 HLUT3 expert; TP2raw = P150A GDDR6 / 6,912,000 B TP2 expert.\n");
    std::printf("[RAW] ceiling counts are weights-only; runtime attention/shared/MTP reserves are excluded.\n");
    std::printf("[L1] HLUT3 11 hot shards/card = 72.52 MiB/card, TP2 hot = 22 shards/pair; bank=1,413,248 <= 1,459,200 (+45,952 B).\n");

    std::printf("\n=== TOP-32 MTP OVERLAY TOTAL ACROSS TP3 RANKS ===\n");
    std::printf("%-11s | %12s | %12s\n", "cfg", "rank_bytes", "total_MiB");
    std::printf("------------+--------------+--------------\n");
    for (const auto& r : rows) {
        std::printf("%-11s | %12llu | %12.6f\n", config_id(r.mi50, r.p150).c_str(),
                    static_cast<unsigned long long>(kTop32MtpDeltaBytes),
                    bytes_to_mib(r.mtp_t32_total_bytes));
    }
    std::printf("[MTP] top-32 promotion = 49,487,872 B = 47.1953125 MiB/rank (approx +47.19).\n");

    dsv41::p150_hlut3_geometry_report();

    const bool ok =
        (rows.size() == 8u) && all_fit &&
        (dsv41::kP150Hlut3MaxHotShards == 11u) &&
        (dsv41::kP150Hlut3BankBytesAt11 == 1'413'248u) &&
        (dsv41::kP150L1Fabric1DBankBytes == 1'459'200u) &&
        (size_t(dsv41::kP150L1Fabric1DBankBytes) - size_t(dsv41::kP150Hlut3BankBytesAt11) == 45'952u) &&
        (kPromotionDeltaBytes == 1'546'496ULL) &&
        (kTop32MtpDeltaBytes == 49'487'872ULL);

    std::printf("\n=== WAVE9_SWITCHBOX RECEIPT ===\n");
    std::printf("configs_4x7_mi50_2x3_p150 : %zu PASS\n", rows.size());
    std::printf("tile_400B_superblock_6400B : PASS\n");
    std::printf("tp3_rank_hlut3_4833280B   : PASS (4.609 MiB exact 4.609375)\n");
    std::printf("tp3_rank_hlut2_3286784B   : PASS (3.134 MiB exact 3.134521484375)\n");
    std::printf("promotion_delta_1546496B  : PASS (1.474853515625 MiB approx +1.475/rank)\n");
    std::printf("top32_mtp_49487872B       : PASS (47.1953125 MiB approx +47.19/rank)\n");
    std::printf("p150_l1_bank_fit_11_shards : %s (1,413,248 <= 1,459,200)\n", all_fit ? "PASS" : "FAIL");
    std::printf("overall                   : %s\n", ok ? "PASS" : "FAIL");
    std::fflush(stdout);
    return ok ? 0 : 1;
}
