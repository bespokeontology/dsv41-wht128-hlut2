#pragma once
// Wave 6B: Live Coding/DSML/Math Router Histogram + Routed Activation Collector & Zero-HBM2 MI50 Overlay Patcher
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
    static constexpr int kMinNeedRows  = 11776;  // 11,264 (> 2*K=10,240) cal + 512 held-out eval
    static constexpr int kMaxStoreRows = 12032;  // 11,520 cal + 512 held-out eval
    static constexpr int kMainCandCap  = 640;    // Wave 8 Track 8C: expanded main backbone hot-expert cap
    static constexpr int kMtpCandCap   = 80;     // Wave 8 Track 8C: expanded DSpark MTP (mtp.0..2) hot-expert cap

    struct CandBuf {
        int block_id = 0;
        int eid = 0;
        bool pinned_p150 = false;
        bool flushed_on_disk = false;
        int rows = 0;
        int rare_count_cached = 0;
        std::vector<uint8_t> rare;
        std::vector<uint16_t> x_bf16;
    };

    std::string out_dir;
    std::vector<uint8_t> vocab_is_rare;
    std::array<std::array<uint64_t, kMaxExperts>, kTotalBlocks> router_hits{};
    std::array<uint64_t, kTotalBlocks> total_rows_seen{};
    std::array<std::array<int16_t, kMaxExperts>, kTotalBlocks> cand_slot{};
    uint64_t block_need_mask = 0ULL;
    std::vector<std::pair<int, int>> pinned_p150_pairs;
    std::vector<CandBuf> cands;
    uint16_t* h_x_pinned = nullptr;
    std::array<uint8_t, 320> cur_row_rare{};
    bool active = false;
    bool pass0_only = false;
    bool candidates_locked = false;
    int turns_captured = 0;

    ~RarebalLiveCollector() {
        if (h_x_pinned) (void)hipHostFree(h_x_pinned);
    }

    void init(const std::string& dir, const Dsv41Tokenizer& tok, const std::string& hot_txt) {
        out_dir = dir;
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);
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
            }
        }
        void* hp = nullptr;
        if (hipHostMalloc(&hp, 320ULL * kDim * sizeof(uint16_t), hipHostMallocDefault) == hipSuccess) {
            h_x_pinned = static_cast<uint16_t*>(hp);
        }
        std::printf("[RAREBAL_CAL] Armed live collector -> %s (vocab_rare=%zu/%zu, p150_l1_pinned=%zu)\n",
                    out_dir.c_str(), rare_v, V, pinned_p150_pairs.size());
        std::fflush(stdout);

        // If a full-corpus router_hist.txt seed already exists, lock globally top candidates immediately (skip Pass 0)!
        const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                         ? std::string(std::getenv("DSV41_ROOT"))
                                         : std::string(".");
        std::ifstream sh(root_env + "/packed_rarebal_overlay_wave8/router_hist.txt");
        if (sh.is_open()) {
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
            }
        }
    }

    void set_batch_tokens(const int32_t* ids, int M) noexcept {
        const int lim = std::min(M, 320);
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

    void write_cand_bin(CandBuf& cb, bool release_ram) {
        if (cb.rows <= 0) return;
        if (!cb.flushed_on_disk) {
            int rcnt = 0;
            for (uint8_t u : cb.rare) rcnt += int(u != 0u);
            cb.rare_count_cached = rcnt;
        }
        char fn[128];
        std::snprintf(fn, sizeof(fn), "/b%02d_e%03d.bin", cb.block_id, cb.eid);
        int fd = ::open((out_dir + fn).c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0664);
        if (fd >= 0) {
            struct Hdr { uint32_t magic; int32_t bid; int32_t eid; int32_t rows; uint32_t pin; uint32_t pad; uint64_t hits; }
                hdr{0x5242414Cu, cb.block_id, cb.eid, cb.rows, uint32_t(cb.pinned_p150), 0u, router_hits[size_t(cb.block_id)][size_t(cb.eid)]};
            auto wr = [&](const void* p, size_t n) {
                const auto* b = static_cast<const uint8_t*>(p);
                for (size_t o = 0; o < n;) { const ssize_t w = ::write(fd, b + o, n - o); if (w <= 0) break; o += size_t(w); }
            };
            wr(&hdr, sizeof(hdr));
            wr(cb.rare.data(), cb.rare.size());
            wr(cb.x_bf16.data(), cb.x_bf16.size() * sizeof(uint16_t));
            ::close(fd);
            if (release_ram) {
                cb.flushed_on_disk = true;
                std::vector<uint8_t>().swap(cb.rare);
                std::vector<uint16_t>().swap(cb.x_bf16);
            }
        }
    }

    void lock_candidates() {
        if (candidates_locked) return;
        auto add_cand = [&](int b, int e, bool pin) {
            if (cand_slot[size_t(b)][size_t(e)] >= 0) return false;
            const int16_t idx = static_cast<int16_t>(cands.size());
            cand_slot[size_t(b)][size_t(e)] = idx;
            CandBuf cb;
            cb.block_id = b; cb.eid = e; cb.pinned_p150 = pin;
            cands.push_back(std::move(cb));
            block_need_mask |= (1ULL << b);
            return true;
        };
        for (const auto& pr : pinned_p150_pairs) add_cand(pr.first, pr.second, true);

        struct Ranked { uint64_t h; int b; int e; };
        std::vector<Ranked> mi50_r, p150_r, all_main, mtp_r;
        for (int b = 0; b < 40; ++b) {
            for (int e = 0; e < 384; ++e) {
                const uint64_t h = router_hits[size_t(b)][size_t(e)];
                if (h == 0ULL) continue;
                all_main.push_back({h, b, e});
                if (e < 216) mi50_r.push_back({h, b, e});
                else         p150_r.push_back({h, b, e});
            }
        }
        auto cmp = [](const Ranked& a, const Ranked& b) { return (a.h != b.h) ? (a.h > b.h) : (a.b != b.b ? a.b < b.b : a.e < b.e); };
        std::sort(mi50_r.begin(), mi50_r.end(), cmp);
        std::sort(p150_r.begin(), p150_r.end(), cmp);
        std::sort(all_main.begin(), all_main.end(), cmp);

        int mi50_cnt = 0, p150_cnt = int(cands.size());
        for (const auto& x : mi50_r) { if (mi50_cnt >= 240) break; if (add_cand(x.b, x.e, false)) ++mi50_cnt; }
        for (const auto& x : p150_r) { if (p150_cnt >= 240) break; if (add_cand(x.b, x.e, false)) ++p150_cnt; }
        for (const auto& x : all_main) {
            if (int(cands.size()) >= kMainCandCap) break;
            add_cand(x.b, x.e, false);
        }
        for (int b = 40; b < 43; ++b) {
            for (int e = 0; e < 128; ++e) {
                if (router_hits[size_t(b)][size_t(e)] > 0ULL) mtp_r.push_back({router_hits[size_t(b)][size_t(e)], b, e});
            }
        }
        std::sort(mtp_r.begin(), mtp_r.end(), cmp);
        int mtp_cnt = 0;
        for (const auto& x : mtp_r) {
            if (mtp_cnt >= kMtpCandCap) break;
            if (add_cand(x.b, x.e, false)) ++mtp_cnt;
        }
        candidates_locked = true;
        std::printf("[RAREBAL_CAL] Locked %zu hot candidates (%d main [p150_pinned=%zu] + %d MTP)\n",
                    cands.size(), int(cands.size()) - mtp_cnt, pinned_p150_pairs.size(), mtp_cnt);
        std::fflush(stdout);
    }

    void record_routed_batch(int block_id, int M, int top_k, const int32_t* topk_ids, const uint16_t* x_bf16) {
        total_rows_seen[size_t(block_id)] += uint64_t(M);
        const bool copy_x = need_x_copy(block_id) && (x_bf16 != nullptr);
        bool any_full = false;
        for (int r = 0; r < M; ++r) {
            const uint8_t r_flag = cur_row_rare[size_t(r)];
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
                        cb.x_bf16.insert(cb.x_bf16.end(), row_ptr, row_ptr + kDim);
                        if (++cb.rows == kMaxStoreRows) {
                            write_cand_bin(cb, true);
                            any_full = true;
                        }
                    }
                }
            }
        }
        if (any_full) refresh_block_mask(block_id);
    }

    bool targets_satisfied(int* out_main_ok = nullptr, int* out_pin_ok = nullptr, int* out_mtp_ok = nullptr) const noexcept {
        int m_ok = 0, p_ok = 0, p_tot = 0, mtp_ok = 0;
        for (const auto& cb : cands) {
            if (cb.pinned_p150) ++p_tot;
            if (cb.rows >= kMinNeedRows) {
                if (cb.block_id < 40) { ++m_ok; if (cb.pinned_p150) ++p_ok; }
                else ++mtp_ok;
            }
        }
        if (out_main_ok) *out_main_ok = m_ok;
        if (out_pin_ok)  *out_pin_ok  = p_ok;
        if (out_mtp_ok)  *out_mtp_ok  = mtp_ok;
        return (p_ok == p_tot) && ((m_ok + mtp_ok) >= 560);
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
            if (cb.rows > 0 && !cb.flushed_on_disk) {
                write_cand_bin(cb, false);
            }
            const int rare_cnt = cb.flushed_on_disk ? cb.rare_count_cached : [&]() {
                int rc = 0;
                for (uint8_t u : cb.rare) rc += int(u != 0u);
                return rc;
            }();
            char fn[128];
            std::snprintf(fn, sizeof(fn), "/b%02d_e%03d.bin", cb.block_id, cb.eid);
            cf << cb.block_id << " " << cb.eid << " " << int(cb.pinned_p150) << " "
               << cb.rows << " " << rare_cnt << " " << router_hits[size_t(cb.block_id)][size_t(cb.eid)]
               << " " << (out_dir + fn) << "\n";
        }
        std::printf("[RAREBAL_CAL] Flushed turn #%d -> ready(>11776): main=%d (p150_pinned=%d/%zu) mtp=%d (total_ready=%d)\n",
                    turns_captured, m_ok, p_ok, pinned_p150_pairs.size(), mtp_ok, m_ok + mtp_ok);
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
