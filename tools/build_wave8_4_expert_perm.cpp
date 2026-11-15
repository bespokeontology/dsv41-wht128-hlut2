// Wave 8.4 Tracks 8.4A + 8.4B:
//   1. Minimax Frequency-Balanced 384-Expert Placement Optimizer (3x MI50 HBM2 <-> 2x Blackhole P150A L1 + GDDR6)
//   2. Full Starvation-Safe TARQ RAREBAL-LDLQ Solver (B=128 FP64 Cholesky across 3x MI50) on DSpark MTP (mtp.0..2)
//      + Promoted L1/Hot Backbone Experts (0 Dual Receipt Violations)
//   3. Lossless Bit-Exact Cross-Tier (TP3 <-> TP2) Merger & Repacker -> hot_experts_wave8_4_perm.safetensors
// Pure C++17 + HIP/rocBLAS + OpenMP + AVX2, zero Python anywhere.

#include "repack_rarebal_helpers.hpp"
#include "../src/dsv41_wave8_4_repack.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace dsv41;

namespace {

struct CandEntry {
    int bid = 0;
    int eid = 0;
    bool pinned_p150 = false;
    int rows = 0;
    int rare_rows = 0;
    uint64_t hits = 0;
    std::string bin_path;
};

struct ReceiptRow {
    size_t order_idx = 0;
    int bid = 0;
    int eid = 0;      // logical expert ID (0..383 for backbone, 0..127 for MTP)
    int phys_eid = 0; // physical expert ID after perm[bid][eid]
    bool pinned_p150 = false;
    uint64_t hits = 0;
    int m_cal = 0;
    int m_eval = 0;
    int m_rare_cal = 0;
    int m_rare_eval = 0;
    double lam_in = 1.0;
    double lam_mid = 1.0;
    DualReceiptMetrics hi{};
    DualReceiptMetrics rb{};
    bool passed = false;
    std::string source_tag; // "WAVE8_4_NEW" or "WAVE8_PRESERVED"
};

void load_block_wht_signs(const std::string& mi50_root, int bid,
                          std::vector<uint64_t>& signs_h, std::vector<uint64_t>& signs_i) {
    char tag[64];
    if (bid < 40) std::snprintf(tag, sizeof(tag), "/layer_%02d.safetensors", bid);
    else          std::snprintf(tag, sizeof(tag), "/mtp_%d.safetensors", bid - 40);
    signs_i.clear();
    for (int tp = 0; tp < 3; ++tp) {
        SafetensorsMmap st(mi50_root + "/tp" + std::to_string(tp) + tag);
        if (tp == 0) {
            const auto* sh = reinterpret_cast<const uint64_t*>(st.tensor_bytes("wht.signs_h"));
            signs_h.assign(sh, sh + 80);
        }
        const auto* si = reinterpret_cast<const uint64_t*>(st.tensor_bytes("wht.signs_i"));
        signs_i.insert(signs_i.end(), si, si + 12);
    }
}

class AsyncRawPrefetcher {
public:
    static constexpr size_t kWin = 64;

    AsyncRawPrefetcher(HfShardRangeFetcher& fetcher, const std::vector<CandEntry>& items, int n_threads = 32)
        : fetcher_(fetcher), items_(items)
    {
        for (auto& s : slots_) s.idx = ~size_t(0);
        if (items_.empty()) return;
        const int nt = std::max(1, n_threads);
        workers_.reserve(size_t(nt));
        for (int t = 0; t < nt; ++t) {
            workers_.emplace_back([this]() { worker_loop(); });
        }
    }

    ~AsyncRawPrefetcher() { stop(); }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_prod_.notify_all();
        cv_cons_.notify_all();
        for (auto& th : workers_) {
            if (th.joinable()) th.join();
        }
        workers_.clear();
    }

    bool take_next(size_t& out_idx, RawExpertMxfp4& out_raw, std::string& out_err,
                   double max_minutes, std::chrono::steady_clock::time_point t_start)
    {
        std::lock_guard<std::mutex> take_guard(take_order_mtx_);
        size_t idx = 0;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            if (stop_ || next_consume_ >= items_.size()) return false;
            idx = next_consume_;
            const double elapsed_min =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() / 60.0;
            if (!items_[idx].pinned_p150 && elapsed_min >= max_minutes) {
                stop_ = true;
                lk.unlock();
                cv_prod_.notify_all();
                cv_cons_.notify_all();
                return false;
            }
            ++next_consume_;
            Slot& s = slots_[idx % kWin];
            cv_cons_.wait(lk, [&]() { return stop_ || (s.ready && s.idx == idx); });
            if (stop_ && !(s.ready && s.idx == idx)) return false;
            out_raw = std::move(s.raw);
            out_err = std::move(s.err);
            s.ready = false;
            s.idx = ~size_t(0);
            s.err.clear();
            if (idx + 1 > consumed_) consumed_ = idx + 1;
        }
        cv_prod_.notify_all();
        out_idx = idx;
        return true;
    }

private:
    struct Slot {
        size_t idx = ~size_t(0);
        bool ready = false;
        RawExpertMxfp4 raw{};
        std::string err{};
    };

    void worker_loop() {
        for (;;) {
            size_t my_idx = ~size_t(0);
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_prod_.wait(lk, [&]() {
                    return stop_ || (next_issue_ < items_.size() && next_issue_ < consumed_ + kWin &&
                                     !slots_[next_issue_ % kWin].ready);
                });
                if (stop_ || next_issue_ >= items_.size()) return;
                my_idx = next_issue_++;
            }

            RawExpertMxfp4 fetched{};
            std::string err_msg;
            try {
                fetched = fetcher_.fetch_expert(items_[my_idx].bid, items_[my_idx].eid);
            } catch (const std::exception& ex) {
                err_msg = ex.what();
            }

            {
                std::lock_guard<std::mutex> lk(mtx_);
                Slot& s = slots_[my_idx % kWin];
                s.idx = my_idx;
                s.raw = std::move(fetched);
                s.err = std::move(err_msg);
                s.ready = true;
            }
            cv_cons_.notify_all();
        }
    }

    HfShardRangeFetcher& fetcher_;
    const std::vector<CandEntry>& items_;
    std::mutex take_order_mtx_;
    std::mutex mtx_;
    std::condition_variable cv_prod_;
    std::condition_variable cv_cons_;
    Slot slots_[kWin];
    size_t next_issue_ = 0;
    size_t next_consume_ = 0;
    size_t consumed_ = 0;
    bool stop_ = false;
    std::vector<std::thread> workers_;
};

void accumulate_router_hist_file(
    const std::string& path,
    std::array<std::array<uint64_t, 384>, 43>& hits,
    std::array<uint64_t, 43>& rows_seen)
{
    std::ifstream hf(path);
    if (!hf.is_open()) return;
    std::string line;
    std::array<uint64_t, 43> file_seen{};
    while (std::getline(hf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int b = -1, e = -1;
        uint64_t h = 0, s = 0;
        if ((iss >> b >> e >> h >> s) && b >= 0 && b < 43 && e >= 0 && e < 384) {
            hits[size_t(b)][size_t(e)] += h;
            file_seen[size_t(b)] = std::max(file_seen[size_t(b)], s);
        }
    }
    for (int b = 0; b < 43; ++b) rows_seen[size_t(b)] += file_seen[size_t(b)];
}

bool run_self_link_check() {
    std::array<std::array<double, 384>, 40> mu{};
    for (int l = 0; l < 40; ++l) {
        double s = 0.0;
        for (int e = 0; e < 384; ++e) {
            // Synthetic skewed distribution with heavy tail in [216..383] on even layers
            const double w = (e >= 216 && (l % 2 == 0)) ? (4.0 / double(1 + (e - 216))) : (1.0 / double(4 + (e % 64)));
            mu[size_t(l)][size_t(e)] = w;
            s += w;
        }
        for (int e = 0; e < 384; ++e) mu[size_t(l)][size_t(e)] *= (48.0 / s);
    }
    const auto plan = optimize_wave84_placement(mu, 18, 20);
    if (plan.l1_phys_assignments.size() != 18u) return false;
    if (!(plan.opt_max_sum_ms < plan.base_max_sum_ms)) return false;
    for (int l = 0; l < 40; ++l) {
        std::array<uint8_t, 384> seen{};
        for (int e = 0; e < 384; ++e) {
            const uint16_t pe = plan.perm[size_t(l)][size_t(e)];
            if (pe >= 384u || seen[pe] != 0u) return false;
            seen[pe] = 1u;
            if (plan.perm[size_t(l)][pe] != uint16_t(e)) return false; // exact involution!
        }
    }

    // Verify bit-exact TP3 <-> TP2 round-trip on a full [2304, 5120] and [5120, 2304] HLUT2 triplet
    ExpertHlut2Triplet orig;
    orig.qg.N = 2304; orig.qg.K = 5120; orig.qg.c0 = 0.271828f; orig.qg.c1 = 1.618034f;
    orig.qg.codes_rm.resize(2304ULL * 320ULL);
    orig.qg.scales_fp16.resize(2304ULL * 40ULL);
    for (size_t i = 0; i < orig.qg.codes_rm.size(); ++i) orig.qg.codes_rm[i] = uint32_t(i * 2654435761ULL);
    for (size_t i = 0; i < orig.qg.scales_fp16.size(); ++i) orig.qg.scales_fp16[i] = uint16_t(0x3800u + (i & 0x3FFu));
    orig.qu = orig.qg;
    orig.qd.N = 5120; orig.qd.K = 2304; orig.qd.c0 = 0.314159f; orig.qd.c1 = 1.414213f;
    orig.qd.codes_rm.resize(5120ULL * 144ULL);
    orig.qd.scales_fp16.resize(5120ULL * 18ULL);
    for (size_t i = 0; i < orig.qd.codes_rm.size(); ++i) orig.qd.codes_rm[i] = uint32_t((i + 17ULL) * 2246822519ULL);
    for (size_t i = 0; i < orig.qd.scales_fp16.size(); ++i) orig.qd.scales_fp16[i] = uint16_t(0x3800u + ((i * 7u) & 0x3FFu));

    std::vector<PackedOverlayTensor> mi50_t, p150_t;
    pack_expert_into_overlay_tensors(orig, 3, 42, mi50_t);
    pack_expert_into_overlay_tensors(orig, 3, 250, p150_t);

    std::array<const uint32_t*, 3> gc{}, dc{};
    std::array<const uint16_t*, 3> gs{}, ds{};
    std::array<const float*, 3> gcent{}, dcent{};
    for (int tp = 0; tp < 3; ++tp) {
        gc[tp]    = reinterpret_cast<const uint32_t*>(mi50_t[size_t(tp * 9 + 0)].buf.data());
        gs[tp]    = reinterpret_cast<const uint16_t*>(mi50_t[size_t(tp * 9 + 1)].buf.data());
        gcent[tp] = reinterpret_cast<const float*>(mi50_t[size_t(tp * 9 + 2)].buf.data());
        dc[tp]    = reinterpret_cast<const uint32_t*>(mi50_t[size_t(tp * 9 + 6)].buf.data());
        ds[tp]    = reinterpret_cast<const uint16_t*>(mi50_t[size_t(tp * 9 + 7)].buf.data());
        dcent[tp] = reinterpret_cast<const float*>(mi50_t[size_t(tp * 9 + 8)].buf.data());
    }
    const auto ug_mi = unpack_hlut2_mi50_tp3_row_sliced(gc, gs, gcent);
    const auto ud_mi = unpack_hlut2_mi50_tp3_col_sliced(dc, ds, dcent);

    std::array<const uint8_t*, 2> gp{}, dp{};
    std::array<const float*, 2> p_gc{}, p_dc{};
    for (int tp = 0; tp < 2; ++tp) {
        gp[tp]   = p150_t[size_t(tp * 6 + 0)].buf.data();
        p_gc[tp] = reinterpret_cast<const float*>(p150_t[size_t(tp * 6 + 1)].buf.data());
        dp[tp]   = p150_t[size_t(tp * 6 + 4)].buf.data();
        p_dc[tp] = reinterpret_cast<const float*>(p150_t[size_t(tp * 6 + 5)].buf.data());
    }
    const auto ug_p1 = unpack_hlut2_p150_tp2_row_sliced(gp, p_gc);
    const auto ud_p1 = unpack_hlut2_p150_tp2_col_sliced(dp, p_dc);

    if (ug_mi.codes_rm != orig.qg.codes_rm || ug_mi.scales_fp16 != orig.qg.scales_fp16) return false;
    if (ud_mi.codes_rm != orig.qd.codes_rm || ud_mi.scales_fp16 != orig.qd.scales_fp16) return false;
    if (ug_p1.codes_rm != orig.qg.codes_rm || ug_p1.scales_fp16 != orig.qg.scales_fp16) return false;
    if (ud_p1.codes_rm != orig.qd.codes_rm || ud_p1.scales_fp16 != orig.qd.scales_fp16) return false;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    ::setenv("OMP_WAIT_POLICY", "PASSIVE", 1);
    ::setenv("GPU_MAX_HW_QUEUES", "8", 0);
    const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                     ? std::string(std::getenv("DSV41_ROOT"))
                                     : std::string(".");
    std::string base_hist     = root_env + "/packed_rarebal_overlay_wave8/router_hist.txt";
    std::string prev_overlay  = root_env + "/packed_rarebal_overlay_wave8/hot_experts_rarebal_wave8.safetensors";
    std::string prev_manifest = root_env + "/packed_rarebal_overlay_wave8/rarebal_wave8_receipt_manifest.txt";
    std::string cal_dir       = "/dev/shm/dsv41_rarebal_cal_wave8_4";
    std::string mi50_root     = root_env + "/packed_mi50";
    std::string p150_root     = (std::getenv("DSV41_P150_ROOT") != nullptr)
                                    ? std::string(std::getenv("DSV41_P150_ROOT"))
                                    : (root_env + "/packed_p150");
    std::string out_dir       = root_env + "/packed_rarebal_overlay_wave8_4";
    std::string out_name      = "hot_experts_wave8_4_perm.safetensors";
    std::string manifest_name = "wave8_4_receipt_manifest.txt";
    std::string raw_cache     = "/dev/shm/dsv41_rarebal_raw_wave8_4";
    int k_l1_shards           = 18;
    int max_swaps_per_layer   = 32;
    int max_main_new          = 400;
    int max_mtp_new           = 384;
    int gpu_workers           = 3;
    double max_minutes        = 45.0;
    bool consume_bins         = false;
    bool plan_only            = false;
    bool link_check           = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--base-hist" && i + 1 < argc) base_hist = argv[++i];
        else if (a == "--prev-overlay" && i + 1 < argc) prev_overlay = argv[++i];
        else if (a == "--prev-manifest" && i + 1 < argc) prev_manifest = argv[++i];
        else if (a == "--cal-dir" && i + 1 < argc) cal_dir = argv[++i];
        else if (a == "--mi50-root" && i + 1 < argc) mi50_root = argv[++i];
        else if (a == "--p150-root" && i + 1 < argc) p150_root = argv[++i];
        else if (a == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--out-name" && i + 1 < argc) out_name = argv[++i];
        else if (a == "--manifest-name" && i + 1 < argc) manifest_name = argv[++i];
        else if (a == "--raw-cache" && i + 1 < argc) raw_cache = argv[++i];
        else if (a == "--k-l1-shards" && i + 1 < argc) k_l1_shards = std::clamp(std::stoi(argv[++i]), 1, 18);
        else if (a == "--max-swaps-per-layer" && i + 1 < argc) max_swaps_per_layer = std::clamp(std::stoi(argv[++i]), 0, 168);
        else if (a == "--max-main-new" && i + 1 < argc) max_main_new = std::stoi(argv[++i]);
        else if (a == "--max-mtp-new" && i + 1 < argc) max_mtp_new = std::stoi(argv[++i]);
        else if (a == "--gpu-workers" && i + 1 < argc) gpu_workers = std::clamp(std::stoi(argv[++i]), 1, 3);
        else if (a == "--max-minutes" && i + 1 < argc) max_minutes = std::stod(argv[++i]);
        else if (a == "--consume-bins") consume_bins = true;
        else if (a == "--plan-only") plan_only = true;
        else if (a == "--link-check") link_check = true;
    }

    if (link_check) {
        if (!run_self_link_check()) {
            std::fprintf(stderr, "[WAVE8_4_PERM] Self link & bit-exact round-trip check FAILED\n");
            return 1;
        }
        std::printf("[WAVE8_4_PERM] Link & bit-exact TP3<->TP2 round-trip check OK\n");
        return 0;
    }

    std::array<std::array<uint64_t, 384>, 43> hits{};
    std::array<uint64_t, 43> rows_seen{};
    accumulate_router_hist_file(base_hist, hits, rows_seen);
    accumulate_router_hist_file(cal_dir + "/router_hist.txt", hits, rows_seen);

    std::array<std::array<double, 384>, 40> mu_m8{};
    for (int l = 0; l < 40; ++l) {
        uint64_t tot_h = 0;
        for (int e = 0; e < 384; ++e) tot_h += hits[size_t(l)][size_t(e)];
        const double scale = (tot_h > 0ULL) ? (48.0 / double(tot_h)) : (48.0 / 384.0);
        for (int e = 0; e < 384; ++e) {
            mu_m8[size_t(l)][size_t(e)] = (tot_h > 0ULL) ? (double(hits[size_t(l)][size_t(e)]) * scale) : scale;
        }
    }

    const Wave84PlacementPlan plan = optimize_wave84_placement(mu_m8, k_l1_shards, max_swaps_per_layer);
    std::printf("[WAVE8_4_PERM] Optimized 40x384 placement (K_L1=%d -> %d TP2 shards, swapped_pairs=%d [%d moved experts / 15360]):\n",
                k_l1_shards, k_l1_shards * 2, plan.total_swapped_pairs, plan.total_swapped_pairs * 2);
    std::printf("[WAVE8_4_PERM]   40-Layer E[T_mi50]: %.2f ms -> %.2f ms | E[T_p150]: %.2f ms -> %.2f ms | sum max(T_mi50, T_p150): %.2f ms -> %.2f ms (saved %.2f ms/burst!)\n",
                plan.base_mi50_sum_ms, plan.opt_mi50_sum_ms,
                plan.base_p150_sum_ms, plan.opt_p150_sum_ms,
                plan.base_max_sum_ms,  plan.opt_max_sum_ms,
                plan.base_max_sum_ms - plan.opt_max_sum_ms);
    std::fflush(stdout);

    std::error_code ec;
    std::filesystem::create_directories(cal_dir, ec);
    {
        std::ofstream tf(cal_dir + "/wave8_4_harvest_targets.txt", std::ios::trunc);
        tf << "# block_id logical_eid is_l1_pinned expected_m8_hits\n";
        for (const auto& pr : plan.l1_logical_pairs) {
            tf << pr.first << " " << pr.second << " 1 " << std::fixed << std::setprecision(5)
               << mu_m8[size_t(pr.first)][size_t(pr.second)] << "\n";
        }
        auto sw = plan.swapped_logical_pairs;
        std::sort(sw.begin(), sw.end(), [&](const auto& a, const auto& b) {
            const double ma = mu_m8[size_t(a.first)][size_t(a.second)];
            const double mb = mu_m8[size_t(b.first)][size_t(b.second)];
            return (ma != mb) ? (ma > mb) : (a < b);
        });
        for (const auto& pr : sw) {
            tf << pr.first << " " << pr.second << " 0 " << std::fixed << std::setprecision(5)
               << mu_m8[size_t(pr.first)][size_t(pr.second)] << "\n";
        }
    }

    if (plan_only) {
        std::printf("[WAVE8_4_PERM] Wrote harvest target list -> %s/wave8_4_harvest_targets.txt (L1=%zu, swapped=%zu)\n",
                    cal_dir.c_str(), plan.l1_logical_pairs.size(), plan.swapped_logical_pairs.size());
        return 0;
    }

    std::filesystem::create_directories(out_dir, ec);
    const auto t_start = std::chrono::steady_clock::now();

    std::array<std::array<bool, 384>, 40> is_l1_logical{};
    for (const auto& pr : plan.l1_logical_pairs) {
        is_l1_logical[size_t(pr.first)][size_t(pr.second)] = true;
    }

    // 1. Solve TARQ RAREBAL-LDLQ across newly harvested Wave 8.4 candidates (M_rows >= 11,776) on 3x MI50
    std::vector<CandEntry> raw_cands;
    {
        std::ifstream cf(cal_dir + "/candidates.txt");
        if (cf.is_open()) {
            CandEntry ce;
            int pin = 0;
            while (cf >> ce.bid >> ce.eid >> pin >> ce.rows >> ce.rare_rows >> ce.hits >> ce.bin_path) {
                ce.pinned_p150 = (pin != 0) || (ce.bid < 40 && is_l1_logical[size_t(ce.bid)][size_t(ce.eid)]);
                if (ce.rows >= 10752 && std::filesystem::is_regular_file(ce.bin_path, ec)) {
                    raw_cands.push_back(ce);
                } else if (consume_bins && !ce.bin_path.empty()) {
                    std::filesystem::remove(ce.bin_path, ec);
                }
            }
        }
    }
    std::sort(raw_cands.begin(), raw_cands.end(), [](const CandEntry& a, const CandEntry& b) {
        if (a.pinned_p150 != b.pinned_p150) return a.pinned_p150 > b.pinned_p150;
        if (a.hits != b.hits) return a.hits > b.hits;
        return (a.bid != b.bid) ? (a.bid < b.bid) : (a.eid < b.eid);
    });

    std::vector<CandEntry> all_cands;
    int sel_main = 0, sel_mtp = 0;
    for (const auto& ce : raw_cands) {
        if (ce.bid < 40) {
            if (sel_main >= max_main_new && !ce.pinned_p150) {
                if (consume_bins) std::filesystem::remove(ce.bin_path, ec);
                continue;
            }
            ++sel_main;
        } else {
            if (sel_mtp >= max_mtp_new) {
                if (consume_bins) std::filesystem::remove(ce.bin_path, ec);
                continue;
            }
            ++sel_mtp;
        }
        all_cands.push_back(ce);
    }

    StreamingSafetensorsWriter writer;
    std::vector<ReceiptRow> receipts;
    std::array<std::array<bool, 384>, 43> emitted_logical{};
    std::mutex out_mtx;
    std::atomic<int> new_main_ok{0}, new_mtp_ok{0};

    if (!all_cands.empty()) {
        int dev_cnt = 0;
        if (hipGetDeviceCount(&dev_cnt) != hipSuccess || dev_cnt <= 0) dev_cnt = 1;
        const int n_workers = std::min(gpu_workers, dev_cnt);

        std::printf("[WAVE8_4_REPACK] Solving NEW Wave 8.4 RAREBAL-LDLQ candidates: %zu (main=%d, mtp=%d) across %d MI50 GPUs...\n",
                    all_cands.size(), sel_main, sel_mtp, n_workers);
        std::fflush(stdout);

        std::array<std::vector<uint64_t>, 43> block_signs_h{};
        std::array<std::vector<uint64_t>, 43> block_signs_i{};
        for (int b = 0; b < 43; ++b) {
            load_block_wht_signs(mi50_root, b, block_signs_h[size_t(b)], block_signs_i[size_t(b)]);
        }

        HfShardRangeFetcher fetcher(raw_cache);
        AsyncRawPrefetcher prefetcher(fetcher, all_cands, 32);

        auto run_gpu_worker = [&](int wid) {
            (void)hipSetDevice(wid);
#ifdef _OPENMP
            omp_set_num_threads(std::max(4, 48 / n_workers));
#endif
            GpuRarebalAccel gpu_accel(wid);
            for (;;) {
                size_t idx = 0;
                RawExpertMxfp4 raw;
                std::string fetch_err;
                if (!prefetcher.take_next(idx, raw, fetch_err, max_minutes, t_start)) break;

                const CandEntry& ce = all_cands[idx];
                const bool is_mtp = (ce.bid >= 40);
                const int phys_e = is_mtp ? ce.eid : int(plan.perm[size_t(ce.bid)][size_t(ce.eid)]);
                const auto t0 = std::chrono::steady_clock::now();

                if (!fetch_err.empty()) {
                    std::error_code ferr_ec;
                    if (consume_bins) std::filesystem::remove(ce.bin_path, ferr_ec);
                    std::lock_guard<std::mutex> lk(out_mtx);
                    std::fprintf(stderr, "[WAVE8_4_REPACK] WARN fetch b%02d e%03d failed: %s\n", ce.bid, ce.eid, fetch_err.c_str());
                    continue;
                }

                std::error_code loc_ec;
                const CapturedExpertData cap = load_captured_expert_bin(ce.bin_path);
                if (consume_bins) std::filesystem::remove(ce.bin_path, loc_ec);

                constexpr int kMEval = 512;
                const int M_cal = cap.rows - kMEval;
                if (M_cal < 10240) continue;

                const int stride = cap.rows / kMEval;
                std::vector<float> X_cal(size_t(M_cal) * DSV41_H), X_eval(size_t(kMEval) * DSV41_H);
                std::vector<uint8_t> rare_cal(static_cast<size_t>(M_cal)), rare_eval(static_cast<size_t>(kMEval));
                int ic = 0, ie = 0;
                for (int r = 0; r < cap.rows; ++r) {
                    const bool hold = (ie < kMEval) && ((r % stride) == (stride - 1));
                    float* dst = hold ? (X_eval.data() + size_t(ie) * DSV41_H) : (X_cal.data() + size_t(ic) * DSV41_H);
                    if (hold) rare_eval[size_t(ie++)] = cap.rare[size_t(r)];
                    else      rare_cal[size_t(ic++)]  = cap.rare[size_t(r)];
                    const uint16_t* sb = cap.x_bf16.data() + size_t(r) * DSV41_H;
                    for (int j = 0; j < DSV41_H; ++j) dst[j] = bf16_to_f32(sb[j]);
                }

                const auto& signs_h = block_signs_h[size_t(ce.bid)];
                const auto& signs_i = block_signs_i[size_t(ce.bid)];

                std::vector<float> w1_rot, w3_rot, w2_rot, scratch_sig;
                const Hlut2QuantMatrix qg_hi = dequant_mxfp4_and_quantize_hlut2_mt(raw.w1_w.data(), raw.w1_s.data(), DSV41_I, DSV41_H, signs_h, w1_rot, scratch_sig);
                const Hlut2QuantMatrix qu_hi = dequant_mxfp4_and_quantize_hlut2_mt(raw.w3_w.data(), raw.w3_s.data(), DSV41_I, DSV41_H, signs_h, w3_rot, scratch_sig);
                const Hlut2QuantMatrix qd_hi = dequant_mxfp4_and_quantize_hlut2_mt(raw.w2_w.data(), raw.w2_s.data(), DSV41_H, DSV41_I, signs_i, w2_rot, scratch_sig);

                Hlut2RarebalStats st_in{}, st_mid{};
                const auto R_up_in = gpu_accel.build_cholesky_rup(X_cal.data(), M_cal, DSV41_H, signs_h, rare_cal.data(), &st_in);
                if (R_up_in.empty()) continue;
                gpu_accel.upload_rup(R_up_in, DSV41_H);
                const Hlut2QuantMatrix qg_rb = gpu_accel.apply_ldlq_preuploaded_rup(qg_hi, w1_rot.data(), R_up_in);
                const Hlut2QuantMatrix qu_rb = gpu_accel.apply_ldlq_preuploaded_rup(qu_hi, w3_rot.data(), R_up_in);

                std::vector<float> wg_rb_r, wu_rb_r, X_cal_rot(size_t(M_cal) * DSV41_H), y_tmp, Z_cal;
                dequantize_hlut2_to_rot_f32(qg_rb, wg_rb_r);
                dequantize_hlut2_to_rot_f32(qu_rb, wu_rb_r);
                wht128_rows(X_cal.data(), X_cal_rot.data(), M_cal, DSV41_H, signs_h);
                (void)eval_rot_swiglu(X_cal_rot.data(), rare_cal.data(), M_cal, wg_rb_r.data(), wu_rb_r.data(), w2_rot.data(), signs_i, nullptr, y_tmp, &Z_cal, &gpu_accel);

                const auto R_up_mid = gpu_accel.build_cholesky_rup(Z_cal.data(), M_cal, DSV41_I, signs_i, rare_cal.data(), &st_mid);
                if (R_up_mid.empty()) continue;
                gpu_accel.upload_rup(R_up_mid, DSV41_I);
                const Hlut2QuantMatrix qd_rb = gpu_accel.apply_ldlq_preuploaded_rup(qd_hi, w2_rot.data(), R_up_mid);

                std::vector<float> X_eval_rot(size_t(kMEval) * DSV41_H), Y_ref, Y_hi, Y_rb, wg_hi_r, wu_hi_r, wd_hi_r, wd_rb_r;
                wht128_rows(X_eval.data(), X_eval_rot.data(), kMEval, DSV41_H, signs_h);
                (void)eval_rot_swiglu(X_eval_rot.data(), rare_eval.data(), kMEval, w1_rot.data(), w3_rot.data(), w2_rot.data(), signs_i, nullptr, Y_ref, nullptr, &gpu_accel);
                dequantize_hlut2_to_rot_f32(qg_hi, wg_hi_r); dequantize_hlut2_to_rot_f32(qu_hi, wu_hi_r); dequantize_hlut2_to_rot_f32(qd_hi, wd_hi_r);
                dequantize_hlut2_to_rot_f32(qd_rb, wd_rb_r);
                const auto m_hi = eval_rot_swiglu(X_eval_rot.data(), rare_eval.data(), kMEval, wg_hi_r.data(), wu_hi_r.data(), wd_hi_r.data(), signs_i, &Y_ref, Y_hi, nullptr, &gpu_accel);
                const auto m_rb = eval_rot_swiglu(X_eval_rot.data(), rare_eval.data(), kMEval, wg_rb_r.data(), wu_rb_r.data(), wd_rb_r.data(), signs_i, &Y_ref, Y_rb, nullptr, &gpu_accel);

                const bool pass_dual = (m_rb.cos_all > m_hi.cos_all) && (m_rb.mse_rare < m_hi.mse_rare);
                ReceiptRow rr{idx, ce.bid, ce.eid, phys_e, ce.pinned_p150, ce.hits, M_cal, kMEval,
                              st_in.m_rare, m_rb.rare_cnt, st_in.lambda, st_mid.lambda, m_hi, m_rb, pass_dual, "WAVE8_4_NEW"};

                std::vector<PackedOverlayTensor> local_tensors;
                if (pass_dual) {
                    ExpertHlut2Triplet tr{qg_rb, qu_rb, qd_rb};
                    pack_expert_into_overlay_tensors(tr, ce.bid, phys_e, local_tensors);
                }

                const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                {
                    std::lock_guard<std::mutex> lk(out_mtx);
                    receipts.push_back(rr);
                    if (pass_dual) {
                        emitted_logical[size_t(ce.bid)][size_t(ce.eid)] = true;
                        for (auto& te : local_tensors) {
                            writer.add_owned(te.name, te.dtype, te.shape, std::move(te.buf));
                        }
                        if (is_mtp) ++new_mtp_ok;
                        else        ++new_main_ok;
                    }
                    std::printf("[RAREBAL84 %3zu/%3zu gpu%d] b%02d e%03d->p%03d%s | M_cal=%d | cos %.5f->%.5f | SNR +%.2fdB | rareMSE %.3e->%.3e | %s (%.2fs)\n",
                                idx + 1, all_cands.size(), wid, ce.bid, ce.eid, phys_e, ce.pinned_p150 ? "[L1]" : "    ",
                                M_cal, m_hi.cos_all, m_rb.cos_all, m_rb.snr_db - m_hi.snr_db,
                                m_hi.mse_rare, m_rb.mse_rare, pass_dual ? "EMIT" : "SKIP", dt);
                    std::fflush(stdout);
                }
            }
        };

        std::vector<std::thread> pool;
        pool.reserve(size_t(n_workers));
        for (int w = 0; w < n_workers; ++w) pool.emplace_back(run_gpu_worker, w);
        for (auto& th : pool) if (th.joinable()) th.join();
        prefetcher.stop();
    }

    // 2. Merge 100% of existing Wave 8 Dual-Receipt-verified experts (re-targeting TP3 <-> TP2 losslessly when perm[b][e] != e!)
    int preserved_wave8 = 0, preserved_retargeted = 0;
    if (std::filesystem::is_regular_file(prev_overlay, ec) && std::filesystem::is_regular_file(prev_manifest, ec)) {
        SafetensorsMmap prev_ov(prev_overlay);
        std::ifstream pm(prev_manifest);
        std::string line;
        size_t base_order = 100000u;
        while (std::getline(pm, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            ReceiptRow rr{};
            int pin_i = 0;
            double d_snr = 0.0;
            std::string status;
            if (!(iss >> rr.bid >> rr.eid >> pin_i >> rr.hits >> rr.m_cal >> rr.m_eval
                      >> rr.m_rare_cal >> rr.m_rare_eval >> rr.lam_in >> rr.lam_mid
                      >> rr.hi.cos_all >> rr.rb.cos_all >> rr.hi.snr_db >> rr.rb.snr_db >> d_snr
                      >> rr.hi.mse_rare >> rr.rb.mse_rare >> status)) {
                continue;
            }
            if (status != "EMITTED") continue;
            if (!(rr.rb.cos_all > rr.hi.cos_all && rr.rb.mse_rare < rr.hi.mse_rare)) continue;
            if (emitted_logical[size_t(rr.bid)][size_t(rr.eid)]) continue;

            const bool is_mtp = (rr.bid >= 40);
            const int phys_e = is_mtp ? rr.eid : int(plan.perm[size_t(rr.bid)][size_t(rr.eid)]);
            rr.order_idx = base_order++;
            rr.phys_eid = phys_e;
            rr.pinned_p150 = (!is_mtp && is_l1_logical[size_t(rr.bid)][size_t(rr.eid)]);
            rr.passed = true;
            rr.source_tag = "WAVE8_PRESERVED";

            if (phys_e == rr.eid) {
                // Fast zero-repack copy directly from Wave 8 overlay!
                if (is_mtp || phys_e < 216) {
                    for (int tp = 0; tp < 3; ++tp) {
                        const std::string p = "mi50.b" + std::to_string(rr.bid) + ".e" + std::to_string(phys_e) + ".tp" + std::to_string(tp);
                        auto cp = [&](const std::string& suf, const char* dt, std::vector<int64_t> sh) {
                            const auto& info = prev_ov.at(p + suf);
                            const uint8_t* ptr = prev_ov.tensor_bytes(p + suf);
                            writer.add_owned(p + suf, dt, std::move(sh), std::vector<uint8_t>(ptr, ptr + info.byte_size()));
                        };
                        cp(".gate_codes", "U32", {24, 320, 32}); cp(".gate_scales", "F16", {24, 40, 32}); cp(".gate_centroids", "F32", {2});
                        cp(".up_codes",   "U32", {24, 320, 32}); cp(".up_scales",   "F16", {24, 40, 32}); cp(".up_centroids",   "F32", {2});
                        cp(".down_codes", "U32", {160, 48, 32}); cp(".down_scales", "F16", {160, 6, 32});  cp(".down_centroids", "F32", {2});
                    }
                } else {
                    const int le = phys_e - 216;
                    for (int tp = 0; tp < 2; ++tp) {
                        const std::string p = "p150.l" + std::to_string(rr.bid) + ".e" + std::to_string(le) + ".tp" + std::to_string(tp);
                        auto cp = [&](const std::string& suf, const char* dt, std::vector<int64_t> sh) {
                            const auto& info = prev_ov.at(p + suf);
                            const uint8_t* ptr = prev_ov.tensor_bytes(p + suf);
                            writer.add_owned(p + suf, dt, std::move(sh), std::vector<uint8_t>(ptr, ptr + info.byte_size()));
                        };
                        cp(".gate_pages", "U8", {360, 4352}); cp(".gate_centroids", "F32", {2});
                        cp(".up_pages",   "U8", {360, 4352}); cp(".up_centroids",   "F32", {2});
                        cp(".down_pages", "U8", {360, 4352}); cp(".down_centroids", "F32", {2});
                    }
                }
            } else {
                const ExpertHlut2Triplet tr = unpack_expert_from_overlay_st(prev_ov, rr.bid, rr.eid);
                std::vector<PackedOverlayTensor> tens;
                pack_expert_into_overlay_tensors(tr, rr.bid, phys_e, tens);
                for (auto& te : tens) writer.add_owned(te.name, te.dtype, te.shape, std::move(te.buf));
                ++preserved_retargeted;
            }
            emitted_logical[size_t(rr.bid)][size_t(rr.eid)] = true;
            receipts.push_back(rr);
            ++preserved_wave8;
        }
    }

    // 3. Losslessly repack any remaining uncalibrated cross-tier swapped experts from local Dual NVMe (packed_mi50 & packed_p150)
    int base_swapped_repacked = 0;
    for (int l = 0; l < 40; ++l) {
        std::vector<int> need_logical;
        for (int e = 0; e < 384; ++e) {
            if (plan.perm[size_t(l)][size_t(e)] != uint16_t(e) && !emitted_logical[size_t(l)][size_t(e)]) {
                need_logical.push_back(e);
            }
        }
        if (need_logical.empty()) continue;

        char tag[64];
        std::snprintf(tag, sizeof(tag), "/layer_%02d.safetensors", l);
        SafetensorsMmap mi0(mi50_root + "/tp0" + tag);
        SafetensorsMmap mi1(mi50_root + "/tp1" + tag);
        SafetensorsMmap mi2(mi50_root + "/tp2" + tag);
        SafetensorsMmap p10(p150_root + "/tp0" + tag);
        SafetensorsMmap p11(p150_root + "/tp1" + tag);
        const std::array<const SafetensorsMmap*, 3> st_tp3{&mi0, &mi1, &mi2};
        const std::array<const SafetensorsMmap*, 2> st_tp2{&p10, &p11};

        std::vector<std::vector<PackedOverlayTensor>> layer_out(need_logical.size());
        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t idx = 0; idx < need_logical.size(); ++idx) {
            const int le = need_logical[idx];
            const int pe = int(plan.perm[size_t(l)][size_t(le)]);
            const ExpertHlut2Triplet tr = (le < 216)
                ? unpack_expert_from_base_mi50(st_tp3, le)
                : unpack_expert_from_base_p150(st_tp2, le - 216);
            pack_expert_into_overlay_tensors(tr, l, pe, layer_out[idx]);
        }
        for (size_t idx = 0; idx < need_logical.size(); ++idx) {
            for (auto& te : layer_out[idx]) {
                writer.add_owned(te.name, te.dtype, te.shape, std::move(te.buf));
            }
            ++base_swapped_repacked;
        }
    }

    // 4. Emit expert_perm_u16 [40, 384] U16 + p150.hot_l1_assignments [K_L1, 2] U16 + p150.hot_l1_logical [K_L1, 2] U16
    {
        std::vector<uint8_t> perm_bytes(40ULL * 384ULL * sizeof(uint16_t));
        std::memcpy(perm_bytes.data(), plan.perm.data(), perm_bytes.size());
        writer.add_owned("expert_perm_u16", "U16", {40, 384}, std::move(perm_bytes));

        std::vector<uint8_t> l1_phys_bytes(plan.l1_phys_assignments.size() * 2ULL * sizeof(uint16_t));
        auto* p_u16 = reinterpret_cast<uint16_t*>(l1_phys_bytes.data());
        for (size_t i = 0; i < plan.l1_phys_assignments.size(); ++i) {
            p_u16[2 * i + 0] = plan.l1_phys_assignments[i].first;
            p_u16[2 * i + 1] = plan.l1_phys_assignments[i].second;
        }
        writer.add_owned("p150.hot_l1_assignments", "U16",
                         {static_cast<int64_t>(plan.l1_phys_assignments.size()), 2},
                         std::move(l1_phys_bytes));

        std::vector<uint8_t> l1_log_bytes(plan.l1_logical_pairs.size() * 2ULL * sizeof(uint16_t));
        auto* l_u16 = reinterpret_cast<uint16_t*>(l1_log_bytes.data());
        for (size_t i = 0; i < plan.l1_logical_pairs.size(); ++i) {
            l_u16[2 * i + 0] = static_cast<uint16_t>(plan.l1_logical_pairs[i].first);
            l_u16[2 * i + 1] = static_cast<uint16_t>(plan.l1_logical_pairs[i].second);
        }
        writer.add_owned("p150.hot_l1_logical", "U16",
                         {static_cast<int64_t>(plan.l1_logical_pairs.size()), 2},
                         std::move(l1_log_bytes));
    }

    const std::string st_out = out_dir + "/" + out_name;
    writer.write_atomic(st_out);

    // Count verified totals & L1 RAREBAL coverage
    int tot_rb_main = 0, tot_rb_mtp = 0, tot_rb_l1 = 0, dual_violations = 0;
    for (const auto& r : receipts) {
        if (!r.passed) continue;
        if (!(r.rb.cos_all > r.hi.cos_all && r.rb.mse_rare < r.hi.mse_rare)) ++dual_violations;
        if (r.bid >= 40) ++tot_rb_mtp;
        else {
            ++tot_rb_main;
            if (is_l1_logical[size_t(r.bid)][size_t(r.eid)]) ++tot_rb_l1;
        }
    }

    std::sort(receipts.begin(), receipts.end(), [](const ReceiptRow& a, const ReceiptRow& b) {
        if (a.passed != b.passed) return a.passed > b.passed;
        if (a.pinned_p150 != b.pinned_p150) return a.pinned_p150 > b.pinned_p150;
        return a.order_idx < b.order_idx;
    });

    {
        std::ofstream mf(out_dir + "/" + manifest_name, std::ios::trunc);
        mf << "# Wave 8.4 Frequency-Balanced + Expanded TARQ RAREBAL-LDLQ Overlay Receipt Manifest\n"
           << "# emitted_rarebal_total=" << (tot_rb_main + tot_rb_mtp)
           << " emitted_rarebal_main=" << tot_rb_main
           << " emitted_p150_l1_pinned=" << tot_rb_l1 << "/" << k_l1_shards
           << " emitted_p150_l1_tp2_shards=" << (tot_rb_l1 * 2) << "/" << (k_l1_shards * 2)
           << " emitted_rarebal_mtp=" << tot_rb_mtp
           << " dual_receipt_violations=" << dual_violations
           << " swapped_pairs=" << plan.total_swapped_pairs
           << " base_swapped_repacked=" << base_swapped_repacked << "\n"
           << "# block logical_eid phys_eid pinned hits M_cal M_eval rare_cal rare_eval lam_in lam_mid cos_hi cos_rb snr_hi snr_rb d_snr_db rare_mse_hi rare_mse_rb status source\n";
        for (const auto& r : receipts) {
            mf << r.bid << " " << r.eid << " " << r.phys_eid << " " << int(r.pinned_p150) << " "
               << r.hits << " " << r.m_cal << " " << r.m_eval << " "
               << r.m_rare_cal << " " << r.m_rare_eval << " " << std::fixed << std::setprecision(5)
               << r.lam_in << " " << r.lam_mid << " " << r.hi.cos_all << " " << r.rb.cos_all << " "
               << r.hi.snr_db << " " << r.rb.snr_db << " " << (r.rb.snr_db - r.hi.snr_db) << " "
               << std::scientific << r.hi.mse_rare << " " << r.rb.mse_rare << " "
               << (r.passed ? "EMITTED" : "SKIPPED") << " " << r.source_tag << "\n";
        }
    }

    // Clean up raw MXFP4 range cache if inside /dev/shm
    if (raw_cache.rfind("/dev/shm/", 0) == 0) {
        std::filesystem::remove_all(raw_cache, ec);
    }

    const double tot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("[WAVE8_4_REPACK] DONE in %.2fs -> %s\n"
                "  RAREBAL-verified experts : %d total (main=%d [new=%d, preserved=%d, retargeted=%d], MTP=%d [new=%d], P150_L1=%d/%d -> %d/%d TP2 shards, violations=%d)\n"
                "  Base cross-tier repacked : %d uncalibrated swapped experts (100%% lossless NVMe TP3<->TP2)\n",
                tot_s, st_out.c_str(),
                tot_rb_main + tot_rb_mtp, tot_rb_main, new_main_ok.load(), preserved_wave8, preserved_retargeted,
                tot_rb_mtp, new_mtp_ok.load(), tot_rb_l1, k_l1_shards, tot_rb_l1 * 2, k_l1_shards * 2, dual_violations,
                base_swapped_repacked);
    return (dual_violations == 0 && tot_rb_l1 == k_l1_shards) ? 0 : 2;
}
