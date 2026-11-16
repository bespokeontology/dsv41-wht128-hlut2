#pragma once
// Wave 8.5A: Multi-Domain Live Coding/DSML/Math/Multimodal Router Histogram + Streaming Routed Activation Collector
// & Zero-HBM2 MI50 Overlay Patcher.
// Pure C++17, zero Python anywhere.

#include "../mi50/dsv41_tokenizer_dsml.hpp"
#include "safetensors_mmap.hpp"

#include <hip/hip_runtime.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dsv41 {

inline uint8_t classify_rare_token_piece(const std::string& s, int32_t id) {
    if (id <= 2 || id >= 128798) return 1u;
    if (s.empty()) return 0u;
    if (s.find("DSML") != std::string::npos || s.find("\xEF\xBD\x9C") != std::string::npos) return 1u;

    bool has_digit = false, has_under = false, has_code_sym = false;
    bool has_lower = false, has_upper = false, upper_after0 = false;
    size_t first_non_ws = std::string::npos;
    for (size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c > ' ' && first_non_ws == std::string::npos) first_non_ws = i;
        if (c >= '0' && c <= '9') has_digit = true;
        else if (c == '_') has_under = true;
        else if (c >= 'a' && c <= 'z') has_lower = true;
        else if (c >= 'A' && c <= 'Z') {
            has_upper = true;
            if (first_non_ws != std::string::npos && i > first_non_ws) upper_after0 = true;
        } else if (std::strchr("{}[]()<>+*/%&|^~!=;:#\\@$`", int(c)) != nullptr) {
            has_code_sym = true;
        }
    }
    if (has_digit || has_under || has_code_sym) return 1u;
    if (first_non_ws == std::string::npos) return 0u;
    if (s.find("->") != std::string::npos || s.find("--") != std::string::npos || s.find("-=") != std::string::npos) return 1u;
    if (has_lower && upper_after0) return 1u;  // camelCase / PascalCase identifier

    std::string t = s.substr(first_non_ws);
    while (!t.empty() && static_cast<unsigned char>(t.back()) <= ' ') t.pop_back();
    if (t == "-") return 1u;
    if (t.size() >= 2 && has_upper && !has_lower) {
        bool all_ascii = true;
        for (unsigned char c : t) if (c >= 128) all_ascii = false;
        if (all_ascii) return 1u;  // ALL_CAPS macro/acronym
    }
    static const char* const kCppKw[] = {
        "const", "constexpr", "inline", "struct", "class", "void", "int", "float", "double",
        "bool", "auto", "return", "static", "template", "typename", "namespace", "sizeof", "nullptr"
    };
    for (const char* kw : kCppKw) {
        if (t == kw) return 1u;
    }
    return 0u;
}

struct RarebalLiveCollector {
    static constexpr int kTotalBlocks  = 43;
    static constexpr int kMaxExperts   = 384;
    static constexpr int kDim          = 5120;
    static constexpr int kMinNeedRows  = 10784;  // 10,272 (>= 2*K = 10,240) cal + 512 held-out eval
    static constexpr int kMaxStoreRows = 10880;  // 10,368 cal + 512 held-out eval (106.25 MiB/bin)
    static constexpr int kStageFlush   = 256;    // Stream 2.5 MiB chunks directly to disk (< 4 GiB total RAM across 1,620+ bins!)
    static constexpr int kDefaultMainCap = 1480; // Wave 8.5A: +778 preserved from Wave 8.4 -> 2,150-2,500+ total RAREBAL!
    static constexpr int kMtpCandCap     = 384;  // Wave 8.5A: ALL 3 x 128 = 384 DSpark MTP (mtp.0..2) experts!

    struct SpillMount {
        std::string dir;
        int max_bins = 20000;
        int used_bins = 0;
    };

    struct CandBuf {
        int block_id = 0;
        int eid = 0;
        bool pinned_p150 = false;
        bool finalized_on_disk = false;
        int fd = -1;
        int rows = 0;
        int rare_count_cached = 0;
        std::string bin_path;
        std::vector<uint8_t> rare;
        std::vector<uint16_t> stage_x_bf16;
    };

    std::string out_dir;
    std::vector<SpillMount> spill_mounts;
    int main_cand_cap = kDefaultMainCap;
    int min_target_ready = 0;
    std::vector<uint8_t> vocab_is_rare;
    std::array<std::array<uint64_t, kMaxExperts>, kTotalBlocks> router_hits{};
    std::array<uint64_t, kTotalBlocks> total_rows_seen{};
    std::array<std::array<int16_t, kMaxExperts>, kTotalBlocks> cand_slot{};
    std::array<std::array<bool, kMaxExperts>, kTotalBlocks> already_wave8{};
    uint64_t block_need_mask = 0ULL;
    std::vector<std::pair<int, int>> pinned_p150_pairs;
    std::vector<std::pair<int, int>> priority_main_pairs;
    std::vector<CandBuf> cands;
    uint16_t* h_x_pinned = nullptr;
    std::array<uint8_t, 512> cur_row_rare{};
    bool active = false;
    bool pass0_only = false;
    bool candidates_locked = false;
    int turns_captured = 0;

    ~RarebalLiveCollector() {
        for (auto& cb : cands) {
            if (cb.fd >= 0) {
                ::close(cb.fd);
                cb.fd = -1;
            }
        }
        if (h_x_pinned) (void)hipHostFree(h_x_pinned);
    }

    static bool write_all_fd(int fd, const void* p, size_t n) noexcept {
        const auto* b = static_cast<const uint8_t*>(p);
        for (size_t o = 0; o < n;) {
            const ssize_t w = ::write(fd, b + o, n - o);
            if (w <= 0) return false;
            o += size_t(w);
        }
        return true;
    }

    static bool pwrite_all_fd(int fd, const void* p, size_t n, off_t off) noexcept {
        const auto* b = static_cast<const uint8_t*>(p);
        for (size_t o = 0; o < n;) {
            const ssize_t w = ::pwrite(fd, b + o, n - o, off + static_cast<off_t>(o));
            if (w <= 0) return false;
            o += size_t(w);
        }
        return true;
    }

    void parse_spill_dirs_env() {
        spill_mounts.clear();
        const char* env_sp = std::getenv("DSV41_SPILL_DIRS");
        if (!env_sp || !*env_sp) env_sp = std::getenv("DSV41_RAREBAL_SPILL_DIRS");
        if (env_sp && *env_sp) {
            std::stringstream ss(env_sp);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (item.empty()) continue;
                const size_t colon = item.rfind(':');
                SpillMount sm;
                if (colon != std::string::npos && colon > 0) {
                    sm.dir = item.substr(0, colon);
                    sm.max_bins = std::max(1, std::atoi(item.substr(colon + 1).c_str()));
                } else {
                    sm.dir = item;
                    sm.max_bins = 20000;
                }
                std::error_code ec;
                std::filesystem::create_directories(sm.dir, ec);
                spill_mounts.push_back(std::move(sm));
            }
        }
        if (spill_mounts.empty()) {
            SpillMount sm;
            if (!out_dir.empty()) {
                sm.dir = out_dir;
            } else {
                const char* td = std::getenv("TMPDIR");
                sm.dir = std::string((td && *td) ? td : "/tmp") + "/dsv41_rarebal_spill";
            }
            sm.max_bins = 20000;
            spill_mounts.push_back(std::move(sm));
        }
        if (const char* env_cap = std::getenv("DSV41_RAREBAL_MAIN_CAP")) {
            const int v = std::atoi(env_cap);
            if (v > 0) main_cand_cap = v;
        }
        if (const char* env_min = std::getenv("DSV41_RAREBAL_MIN_READY")) {
            const int v = std::atoi(env_min);
            if (v > 0) min_target_ready = v;
        }
    }

    std::string assign_bin_path(int block_id, int eid) {
        char fn[128];
        std::snprintf(fn, sizeof(fn), "/b%02d_e%03d.bin", block_id, eid);
        for (auto& sm : spill_mounts) {
            if (sm.used_bins < sm.max_bins) {
                ++sm.used_bins;
                return sm.dir + fn;
            }
        }
        ++spill_mounts.back().used_bins;
        return spill_mounts.back().dir + fn;
    }

    void init(const std::string& dir, const Dsv41Tokenizer& tok, const std::string& hot_txt) {
        out_dir = dir;
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);
        parse_spill_dirs_env();
        for (auto& r : cand_slot) r.fill(-1);

        const size_t V = tok.vocab_size();
        vocab_is_rare.assign(V, 0u);
        size_t rare_v = 0;
        for (size_t i = 0; i < V; ++i) {
            const uint8_t r = classify_rare_token_piece(tok.decode_single(int32_t(i), false), int32_t(i));
            vocab_is_rare[i] = r;
            rare_v += r;
        }
        std::ifstream hf(hot_txt);
        if (hf.is_open()) {
            int l = 0, e = 0;
            uint64_t dh = 0, ph = 0;
            while (hf >> l >> e >> dh >> ph) {
                if (l >= 0 && l < 40 && e >= 0 && e < 168) pinned_p150_pairs.push_back({l, 216 + e});
                else if (l >= 0 && l < 40 && e >= 168 && e < 384) pinned_p150_pairs.push_back({l, e});
            }
        }

        const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                         ? std::string(std::getenv("DSV41_ROOT"))
                                         : std::string(".");
        // Load existing Dual-Receipt-verified experts from Wave 8.4 (and Wave 8 fallback) so Wave 8.5
        // dedicates 100% of new spill slots to UNCALIBRATED hot backbone + MTP experts (+ L1 re-verification)!
        const std::string prev_manifests[] = {
            root_env + "/packed_rarebal_overlay_wave8_5/wave8_5_stage1_manifest.txt",
            root_env + "/packed_rarebal_overlay_wave8_5/wave8_5_receipt_manifest.txt",
            root_env + "/packed_rarebal_overlay_wave8_4/wave8_4_receipt_manifest.txt",
            root_env + "/packed_rarebal_overlay_wave8/rarebal_wave8_receipt_manifest.txt"
        };
        size_t preserved_cnt = 0;
        for (const std::string& mp : prev_manifests) {
            std::ifstream wm(mp);
            if (!wm.is_open()) continue;
            std::string line;
            while (std::getline(wm, line)) {
                if (line.empty() || line[0] == '#') continue;
                if (line.find("EMITTED") == std::string::npos) continue;
                std::istringstream iss(line);
                int b = -1, e = -1;
                if ((iss >> b >> e) && b >= 0 && b < kTotalBlocks && e >= 0 && e < kMaxExperts) {
                    if (!already_wave8[size_t(b)][size_t(e)]) {
                        already_wave8[size_t(b)][size_t(e)] = true;
                        ++preserved_cnt;
                    }
                }
            }
            if (preserved_cnt > 0) break;
        }

        // Load exact P150A L1 logical assignments + per-layer top-18 P150A priority pairs if Wave 8.4 overlay exists
        const std::string w84_ov_path = root_env + "/packed_rarebal_overlay_wave8_4/hot_experts_wave8_4_perm.safetensors";
        if (std::filesystem::is_regular_file(w84_ov_path, ec)) {
            try {
                SafetensorsMmap w84_ov(w84_ov_path);
                if (w84_ov.contains("p150.hot_l1_logical")) {
                    const auto& info = w84_ov.at("p150.hot_l1_logical");
                    const size_t k_l1 = info.shape.empty() ? 0u : static_cast<size_t>(info.shape[0]);
                    const auto* lu16 = reinterpret_cast<const uint16_t*>(w84_ov.tensor_bytes("p150.hot_l1_logical"));
                    pinned_p150_pairs.clear();
                    for (size_t i = 0; i < k_l1; ++i) {
                        const int lb = int(lu16[2 * i + 0]);
                        const int le = int(lu16[2 * i + 1]);
                        if (lb >= 0 && lb < 40 && le >= 0 && le < 384) {
                            pinned_p150_pairs.push_back({lb, le});
                        }
                    }
                }
                if (w84_ov.contains("expert_perm_u16")) {
                    const auto* pu16 = reinterpret_cast<const uint16_t*>(w84_ov.tensor_bytes("expert_perm_u16"));
                    // Prioritize per-layer P150A [216..233] slots (40 x 18 = 720 slots) ahead of generic tail!
                    for (int b = 0; b < 40; ++b) {
                        for (int le = 0; le < 384; ++le) {
                            const int pe = int(pu16[size_t(b) * 384u + size_t(le)]);
                            if (pe >= 216 && pe < 234) {
                                priority_main_pairs.push_back({b, le});
                            }
                        }
                    }
                }
            } catch (...) {
            }
        }

        // Optional explicit target list override ("block logical_eid is_l1_pinned")
        for (const std::string& tf_name : {out_dir + "/wave8_5_harvest_targets.txt", out_dir + "/wave8_4_harvest_targets.txt"}) {
            std::ifstream tf(tf_name);
            if (!tf.is_open()) continue;
            std::string line;
            while (std::getline(tf, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream iss(line);
                int b = -1, e = -1, pin = 0;
                if ((iss >> b >> e >> pin) && b >= 0 && b < 40 && e >= 0 && e < 384) {
                    if (pin != 0) pinned_p150_pairs.push_back({b, e});
                    else          priority_main_pairs.push_back({b, e});
                }
            }
            break;
        }

        void* hp = nullptr;
        if (hipHostMalloc(&hp, 512ULL * kDim * sizeof(uint16_t), hipHostMallocDefault) == hipSuccess) {
            h_x_pinned = static_cast<uint16_t*>(hp);
        }
        std::printf("[RAREBAL_CAL] Armed Wave 8.5 streaming collector -> %s (mounts=%zu, main_cap=%d, preserved=%zu, vocab_rare=%zu/%zu, p150_l1=%zu)\n",
                    out_dir.c_str(), spill_mounts.size(), main_cand_cap, preserved_cnt, rare_v, V, pinned_p150_pairs.size());
        std::fflush(stdout);

        // Seed from cumulative router_hist.txt if present so candidate selection starts immediately with zero Pass-0 waste!
        const std::string seed_hists[] = {
            root_env + "/packed_rarebal_overlay_wave8_5/router_hist_wave8_5.txt",
            root_env + "/packed_rarebal_overlay_wave8_4/router_hist_wave8_4.txt",
            root_env + "/packed_rarebal_overlay_wave8/router_hist.txt"
        };
        for (const std::string& shp : seed_hists) {
            std::ifstream sh(shp);
            if (!sh.is_open()) continue;
            std::string line;
            size_t loaded = 0;
            while (std::getline(sh, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream iss(line);
                int b = -1, e = -1;
                uint64_t h = 0, seen = 0;
                if ((iss >> b >> e >> h >> seen) && b >= 0 && b < kTotalBlocks && e >= 0 && e < kMaxExperts) {
                    router_hits[size_t(b)][size_t(e)] = h;
                    ++loaded;
                }
            }
            if (loaded >= 15000u) {
                lock_candidates();
                for (auto& r : router_hits) r.fill(0ULL);
                total_rows_seen.fill(0ULL);
                break;
            }
        }
    }

    void set_batch_tokens(const int32_t* ids, int M) noexcept {
        const int lim = std::min(M, int(cur_row_rare.size()));
        for (int i = 0; i < lim; ++i) {
            const int32_t t = ids[i];
            cur_row_rare[size_t(i)] = (t >= 0 && size_t(t) < vocab_is_rare.size()) ? vocab_is_rare[size_t(t)] : 1u;
        }
    }

    bool need_x_copy(int block_id) const noexcept {
        return active && !pass0_only && candidates_locked && ((block_need_mask >> block_id) & 1ULL) != 0ULL;
    }

    void refresh_block_mask(int block_id) noexcept {
        bool any = false;
        const int n_exp = (block_id < 40) ? 384 : 128;
        for (int e = 0; e < n_exp; ++e) {
            const int s = cand_slot[size_t(block_id)][size_t(e)];
            if (s >= 0 && cands[size_t(s)].rows < kMaxStoreRows) { any = true; break; }
        }
        if (any) block_need_mask |= (1ULL << block_id);
        else     block_need_mask &= ~(1ULL << block_id);
    }

    struct StreamHdr {
        uint32_t magic;
        int32_t  bid;
        int32_t  eid;
        int32_t  rows;
        uint32_t pin;
        uint32_t pad;
        uint64_t hits;
    };

    bool ensure_cand_fd(CandBuf& cb) {
        if (cb.fd >= 0) return true;
        if (cb.bin_path.empty()) cb.bin_path = assign_bin_path(cb.block_id, cb.eid);
        cb.fd = ::open(cb.bin_path.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_CLOEXEC, 0664);
        if (cb.fd < 0) return false;
        StreamHdr hdr{0x5242414Du, cb.block_id, cb.eid, 0, uint32_t(cb.pinned_p150), 0u, 0ULL};
        if (!write_all_fd(cb.fd, &hdr, sizeof(hdr))) {
            ::close(cb.fd);
            cb.fd = -1;
            return false;
        }
        return true;
    }

    void flush_stage_rows(CandBuf& cb) {
        if (cb.stage_x_bf16.empty()) return;
        if (!ensure_cand_fd(cb)) {
            std::vector<uint16_t>().swap(cb.stage_x_bf16);
            return;
        }
        (void)write_all_fd(cb.fd, cb.stage_x_bf16.data(), cb.stage_x_bf16.size() * sizeof(uint16_t));
        cb.stage_x_bf16.clear();
    }

    void finalize_cand_bin(CandBuf& cb, bool release_ram) {
        if (cb.rows <= 0) return;
        if (cb.finalized_on_disk) return;
        flush_stage_rows(cb);
        if (cb.fd < 0 && !ensure_cand_fd(cb)) return;

        int rcnt = 0;
        for (uint8_t u : cb.rare) rcnt += int(u != 0u);
        cb.rare_count_cached = rcnt;

        // Streaming layout 0x5242414Du: [Hdr 32B][x_bf16 rows*5120*2B][rare rows*1B]
        const off_t rare_off = static_cast<off_t>(sizeof(StreamHdr)) +
                               static_cast<off_t>(size_t(cb.rows) * size_t(kDim) * sizeof(uint16_t));
        (void)pwrite_all_fd(cb.fd, cb.rare.data(), cb.rare.size(), rare_off);
        if (::ftruncate(cb.fd, rare_off + static_cast<off_t>(cb.rare.size())) != 0) {}

        StreamHdr hdr{0x5242414Du, cb.block_id, cb.eid, cb.rows, uint32_t(cb.pinned_p150), 0u,
                      router_hits[size_t(cb.block_id)][size_t(cb.eid)]};
        (void)pwrite_all_fd(cb.fd, &hdr, sizeof(hdr), 0);

        if (release_ram) {
            ::close(cb.fd);
            cb.fd = -1;
            cb.finalized_on_disk = true;
            std::vector<uint8_t>().swap(cb.rare);
            std::vector<uint16_t>().swap(cb.stage_x_bf16);
        }
    }

    void lock_candidates() {
        if (candidates_locked) return;
        int total_mount_cap = 0;
        for (const auto& sm : spill_mounts) total_mount_cap += sm.max_bins;

        // Count how many uncalibrated MTP experts exist so we reserve 100% of their slots first!
        int need_mtp = 0;
        for (int b = 40; b < 43; ++b) {
            for (int e = 0; e < 128; ++e) {
                if (!already_wave8[size_t(b)][size_t(e)]) ++need_mtp;
            }
        }
        need_mtp = std::min(need_mtp, kMtpCandCap);
        const int eff_main_cap = std::max(0, std::min(main_cand_cap, total_mount_cap - need_mtp));

        auto add_cand = [&](int b, int e, bool pin) {
            if (cand_slot[size_t(b)][size_t(e)] >= 0) return false;
            const int16_t idx = static_cast<int16_t>(cands.size());
            cand_slot[size_t(b)][size_t(e)] = idx;
            CandBuf cb;
            cb.block_id = b;
            cb.eid = e;
            cb.pinned_p150 = pin;
            cb.bin_path = assign_bin_path(b, e);
            cb.rare.reserve(512);
            cb.stage_x_bf16.reserve(size_t(kStageFlush) * kDim);
            cands.push_back(std::move(cb));
            block_need_mask |= (1ULL << b);
            return true;
        };

        // 1. P150A L1-pinned experts (re-verify / fill any missing L1 slots first)
        for (const auto& pr : pinned_p150_pairs) {
            if (int(cands.size()) >= eff_main_cap) break;
            if (!already_wave8[size_t(pr.first)][size_t(pr.second)]) add_cand(pr.first, pr.second, true);
        }

        // 2. Sort priority_main_pairs (per-layer top-18 P150A [216..233] slots) descending by hits
        std::sort(priority_main_pairs.begin(), priority_main_pairs.end(), [&](const auto& a, const auto& b) {
            const uint64_t ha = router_hits[size_t(a.first)][size_t(a.second)];
            const uint64_t hb = router_hits[size_t(b.first)][size_t(b.second)];
            return (ha != hb) ? (ha > hb) : (a < b);
        });
        for (const auto& pr : priority_main_pairs) {
            if (int(cands.size()) >= eff_main_cap) break;
            if (router_hits[size_t(pr.first)][size_t(pr.second)] < 45000ULL) continue;
            if (!already_wave8[size_t(pr.first)][size_t(pr.second)]) add_cand(pr.first, pr.second, false);
        }

        // 3. Globally hottest uncalibrated MI50 [0..215] + P150A [234..383] experts sorted descending by hits
        struct Ranked { uint64_t h; int b; int e; };
        std::vector<Ranked> all_main;
        all_main.reserve(15360);
        for (int b = 0; b < 40; ++b) {
            for (int e = 0; e < 384; ++e) {
                if (already_wave8[size_t(b)][size_t(e)]) continue;
                const uint64_t h = router_hits[size_t(b)][size_t(e)];
                if (h == 0ULL) continue;
                all_main.push_back({h, b, e});
            }
        }
        std::sort(all_main.begin(), all_main.end(), [](const Ranked& a, const Ranked& b) {
            return (a.h != b.h) ? (a.h > b.h) : (a.b != b.b ? a.b < b.b : a.e < b.e);
        });
        for (const auto& x : all_main) {
            if (int(cands.size()) >= eff_main_cap) break;
            add_cand(x.b, x.e, false);
        }

        // 4. All uncalibrated DSpark MTP (mtp.0..2) drafter experts
        int mtp_cnt = 0;
        for (int b = 40; b < 43; ++b) {
            for (int e = 0; e < 128; ++e) {
                if (mtp_cnt >= need_mtp) break;
                if (already_wave8[size_t(b)][size_t(e)]) continue;
                if (add_cand(b, e, false)) ++mtp_cnt;
            }
        }
        candidates_locked = true;
        std::printf("[RAREBAL_CAL] Locked %zu NEW Wave 8.5 candidates (%d main + %d MTP across %zu spill mounts)\n",
                    cands.size(), int(cands.size()) - mtp_cnt, mtp_cnt, spill_mounts.size());
        std::fflush(stdout);
    }

    void record_routed_batch(int block_id, int M, int top_k, const int32_t* topk_ids, const uint16_t* x_bf16) {
        total_rows_seen[size_t(block_id)] += uint64_t(M);
        const bool copy_x = need_x_copy(block_id) && (x_bf16 != nullptr);
        bool any_full = false;
        for (int r = 0; r < M; ++r) {
            const uint8_t r_flag = cur_row_rare[size_t(std::min(r, int(cur_row_rare.size()) - 1))];
            for (int k = 0; k < top_k; ++k) {
                const int32_t eid = topk_ids[r * top_k + k];
                if (eid < 0 || eid >= kMaxExperts) continue;
                ++router_hits[size_t(block_id)][size_t(eid)];
                if (copy_x) {
                    const int s = cand_slot[size_t(block_id)][size_t(eid)];
                    if (s >= 0 && cands[size_t(s)].rows < kMaxStoreRows) {
                        auto& cb = cands[size_t(s)];
                        cb.rare.push_back(r_flag);
                        const uint16_t* row_ptr = x_bf16 + size_t(r) * kDim;
                        cb.stage_x_bf16.insert(cb.stage_x_bf16.end(), row_ptr, row_ptr + kDim);
                        ++cb.rows;
                        if (cb.rows == kMaxStoreRows) {
                            finalize_cand_bin(cb, true);
                            any_full = true;
                        } else if (cb.stage_x_bf16.size() >= size_t(kStageFlush) * kDim) {
                            flush_stage_rows(cb);
                        }
                    }
                }
            }
        }
        if (any_full) refresh_block_mask(block_id);
    }

    bool targets_satisfied(int* out_main_ok = nullptr, int* out_pin_ok = nullptr, int* out_mtp_ok = nullptr) const noexcept {
        int m_ok = 0, p_ok = 0, p_tot = 0, mtp_ok = 0, mtp_tot = 0;
        for (const auto& cb : cands) {
            if (cb.pinned_p150) ++p_tot;
            if (cb.block_id >= 40) ++mtp_tot;
            if (cb.rows >= kMinNeedRows) {
                if (cb.block_id < 40) { ++m_ok; if (cb.pinned_p150) ++p_ok; }
                else ++mtp_ok;
            }
        }
        if (out_main_ok) *out_main_ok = m_ok;
        if (out_pin_ok)  *out_pin_ok  = p_ok;
        if (out_mtp_ok)  *out_mtp_ok  = mtp_ok;
        if (cands.empty()) return false;
        if (min_target_ready > 0) {
            return (p_ok == p_tot) && ((m_ok + mtp_ok) >= min_target_ready);
        }
        return (p_ok == p_tot) && (mtp_ok == mtp_tot) && ((m_ok + mtp_ok) == int(cands.size()));
    }

    void flush_to_disk() {
        ++turns_captured;
        {
            std::ofstream hf(out_dir + "/router_hist.txt", std::ios::trunc);
            hf << "# block_id eid router_hits total_rows_seen\n";
            for (int b = 0; b < kTotalBlocks; ++b) {
                const int n_exp = (b < 40) ? 384 : 128;
                for (int e = 0; e < n_exp; ++e) {
                    hf << b << " " << e << " " << router_hits[size_t(b)][size_t(e)]
                       << " " << total_rows_seen[size_t(b)] << "\n";
                }
            }
        }
        std::ofstream cf(out_dir + "/candidates.txt", std::ios::trunc);
        int m_ok = 0, p_ok = 0, mtp_ok = 0;
        (void)targets_satisfied(&m_ok, &p_ok, &mtp_ok);
        for (auto& cb : cands) {
            if (cb.rows > 0 && !cb.finalized_on_disk) {
                finalize_cand_bin(cb, false);
            }
            const int rare_cnt = cb.finalized_on_disk ? cb.rare_count_cached : [&]() {
                int rc = 0;
                for (uint8_t u : cb.rare) rc += int(u != 0u);
                return rc;
            }();
            cf << cb.block_id << " " << cb.eid << " " << int(cb.pinned_p150) << " "
               << cb.rows << " " << rare_cnt << " " << router_hits[size_t(cb.block_id)][size_t(cb.eid)]
               << " " << cb.bin_path << "\n";
        }
        std::printf("[RAREBAL_CAL] Flushed turn #%d -> ready(>=%d): main=%d (p150_pinned=%d/%zu) mtp=%d (total_ready=%d)\n",
                    turns_captured, kMinNeedRows, m_ok, p_ok, pinned_p150_pairs.size(), mtp_ok, m_ok + mtp_ok);
        std::fflush(stdout);
    }
};

inline int apply_mi50_rarebal_overlay(
    int tp, int block_id, int mi50_experts,
    uint32_t* d_gc, uint16_t* d_gs, float* d_gcent,
    uint32_t* d_uc, uint16_t* d_us, float* d_ucent,
    uint32_t* d_dc, uint16_t* d_ds, float* d_dcent,
    const SafetensorsMmap& ov,
    const std::function<void(void*, const void*, size_t)>& copy_h2d)
{
    constexpr size_t kCodesWords  = 24ULL * 320ULL * 32ULL;  // 245,760 uint32_t = 983,040 B
    constexpr size_t kScalesHalfs = 24ULL * 40ULL * 32ULL;   //  30,720 uint16_t =  61,440 B
    int patched = 0;
    for (int e = 0; e < mi50_experts; ++e) {
        const std::string pfx = "mi50.b" + std::to_string(block_id) + ".e" + std::to_string(e) + ".tp" + std::to_string(tp);
        if (!ov.contains(pfx + ".gate_codes")) continue;
        copy_h2d(d_gc + size_t(e) * kCodesWords,  ov.tensor_bytes(pfx + ".gate_codes"),     kCodesWords * 4ULL);
        copy_h2d(d_gs + size_t(e) * kScalesHalfs, ov.tensor_bytes(pfx + ".gate_scales"),    kScalesHalfs * 2ULL);
        copy_h2d(d_gcent + size_t(e) * 2ULL,      ov.tensor_bytes(pfx + ".gate_centroids"), 8ULL);

        copy_h2d(d_uc + size_t(e) * kCodesWords,  ov.tensor_bytes(pfx + ".up_codes"),       kCodesWords * 4ULL);
        copy_h2d(d_us + size_t(e) * kScalesHalfs, ov.tensor_bytes(pfx + ".up_scales"),      kScalesHalfs * 2ULL);
        copy_h2d(d_ucent + size_t(e) * 2ULL,      ov.tensor_bytes(pfx + ".up_centroids"),   8ULL);

        copy_h2d(d_dc + size_t(e) * kCodesWords,  ov.tensor_bytes(pfx + ".down_codes"),     kCodesWords * 4ULL);
        copy_h2d(d_ds + size_t(e) * kScalesHalfs, ov.tensor_bytes(pfx + ".down_scales"),    kScalesHalfs * 2ULL);
        copy_h2d(d_dcent + size_t(e) * 2ULL,      ov.tensor_bytes(pfx + ".down_centroids"), 8ULL);
        ++patched;
    }
    return patched;
}

}  // namespace dsv41
