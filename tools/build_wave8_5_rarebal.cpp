// Wave 8.5 Track 8.5A:
//   2,000+ Expert Starvation-Safe TARQ RAREBAL-LDLQ Solver (B=128 FP64 Cholesky across 3x MI50)
//   in Invariant Wave 8.4 Permuted Layout (d_expert_perm[40 * 384], 445 cross-tier swaps, 18/18 P150A L1 shards).
//   Enforces M_cal >= 10,240 (gate/up), M_cal >= 4,608 (down), M_eval == 512, and dual_receipt_violations == 0,
//   strictly beating both H = I and any existing Wave 8.4 RAREBAL entry (else carrying forward Wave 8.4 unchanged).
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
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace dsv41;

namespace {

struct CandEntry {
    int bid = 0;
    int eid = 0;
    int prio_tier = 3; // 0 = P150A L1 (18/18), 1 = per-layer P150A [216..233] (720), 2 = DSpark mtp.0..2, 3 = warm-hot MI50/P150A
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
    int phys_eid = 0; // physical expert ID after invariant perm[bid][eid]
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
    std::string source_tag; // "WAVE8_5_NEW" or "WAVE8_4_PRESERVED"
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
    std::vector<Slot> slots_{kWin};
    size_t next_issue_ = 0;
    size_t next_consume_ = 0;
    size_t consumed_ = 0;
    bool stop_ = false;
    std::vector<std::thread> workers_;
};

void copy_phys_expert_from_overlay(StreamingSafetensorsWriter& writer, const SafetensorsMmap& src_ov, int bid, int phys_e) {
    const bool is_mtp = (bid >= 40);
    if (is_mtp || phys_e < 216) {
        for (int tp = 0; tp < 3; ++tp) {
            const std::string p = "mi50.b" + std::to_string(bid) + ".e" + std::to_string(phys_e) + ".tp" + std::to_string(tp);
            auto cp = [&](const std::string& suf, const char* dt, std::vector<int64_t> sh) {
                const auto& info = src_ov.at(p + suf);
                const uint8_t* ptr = src_ov.tensor_bytes(p + suf);
                writer.add_owned(p + suf, dt, std::move(sh), std::vector<uint8_t>(ptr, ptr + info.byte_size()));
            };
            cp(".gate_codes", "U32", {24, 320, 32}); cp(".gate_scales", "F16", {24, 40, 32}); cp(".gate_centroids", "F32", {2});
            cp(".up_codes",   "U32", {24, 320, 32}); cp(".up_scales",   "F16", {24, 40, 32}); cp(".up_centroids",   "F32", {2});
            cp(".down_codes", "U32", {160, 48, 32}); cp(".down_scales", "F16", {160, 6, 32});  cp(".down_centroids", "F32", {2});
        }
    } else {
        const int le = phys_e - 216;
        for (int tp = 0; tp < 2; ++tp) {
            const std::string p = "p150.l" + std::to_string(bid) + ".e" + std::to_string(le) + ".tp" + std::to_string(tp);
            auto cp = [&](const std::string& suf, const char* dt, std::vector<int64_t> sh) {
                const auto& info = src_ov.at(p + suf);
                const uint8_t* ptr = src_ov.tensor_bytes(p + suf);
                writer.add_owned(p + suf, dt, std::move(sh), std::vector<uint8_t>(ptr, ptr + info.byte_size()));
            };
            cp(".gate_pages", "U8", {360, 4352}); cp(".gate_centroids", "F32", {2});
            cp(".up_pages",   "U8", {360, 4352}); cp(".up_centroids",   "F32", {2});
            cp(".down_pages", "U8", {360, 4352}); cp(".down_centroids", "F32", {2});
        }
    }
}

bool run_self_link_check() {
    const std::string tmp_bin = "/dev/shm/test_w85_stream_bin_" + std::to_string(::getpid()) + ".bin";
    int fd = ::open(tmp_bin.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) return false;
    struct Hdr { uint32_t magic; int32_t bid; int32_t eid; int32_t rows; uint32_t pin; uint32_t pad; uint64_t hits; }
        hdr{0x5242414Du, 7, 42, 4, 1u, 0u, 12345ULL};
    std::vector<uint16_t> x(4ULL * DSV41_H, 0x3C00u);
    x[5] = 0x7FC1u; // NaN -> must sanitize to 0
    std::vector<uint8_t> r = {1u, 0u, 1u, 0u};
    if (::write(fd, &hdr, sizeof(hdr)) != ssize_t(sizeof(hdr))) { ::close(fd); return false; }
    if (::write(fd, x.data(), x.size() * 2u) != ssize_t(x.size() * 2u)) { ::close(fd); return false; }
    if (::write(fd, r.data(), r.size()) != ssize_t(r.size())) { ::close(fd); return false; }
    ::close(fd);

    const CapturedExpertData cap = load_captured_expert_bin(tmp_bin);
    std::error_code ec;
    std::filesystem::remove(tmp_bin, ec);
    return cap.block_id == 7 && cap.eid == 42 && cap.rows == 4 && cap.pinned_p150 &&
           cap.x_bf16[0] == 0x3C00u && cap.x_bf16[5] == 0u && cap.rare == r;
}

}  // namespace

int main(int argc, char** argv) {
    ::setenv("OMP_WAIT_POLICY", "PASSIVE", 1);
    ::setenv("GPU_MAX_HW_QUEUES", "8", 0);
    const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                     ? std::string(std::getenv("DSV41_ROOT"))
                                     : std::string(".");
    std::string prev_overlay  = root_env + "/packed_rarebal_overlay_wave8_4/hot_experts_wave8_4_perm.safetensors";
    std::string prev_manifest = root_env + "/packed_rarebal_overlay_wave8_4/wave8_4_receipt_manifest.txt";
    std::string cal_dir       = "/dev/shm/dsv41_rarebal_cal_wave8_5";
    std::string mi50_root     = root_env + "/packed_mi50";
    std::string p150_root     = root_env + "/packed_p150";
    std::string out_dir       = root_env + "/packed_rarebal_overlay_wave8_5";
    std::string out_name      = "hot_experts_wave8_5.safetensors";
    std::string manifest_name = "wave8_5_receipt_manifest.txt";
    std::string raw_cache     = "/dev/shm/dsv41_rarebal_raw_wave8_5";
    int max_main_new          = 1600;
    int max_mtp_new           = 384;
    int gpu_workers           = 3;
    double max_minutes        = 75.0;
    bool consume_bins         = false;
    bool link_check           = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--prev-overlay" && i + 1 < argc) prev_overlay = argv[++i];
        else if (a == "--prev-manifest" && i + 1 < argc) prev_manifest = argv[++i];
        else if (a == "--cal-dir" && i + 1 < argc) cal_dir = argv[++i];
        else if (a == "--mi50-root" && i + 1 < argc) mi50_root = argv[++i];
        else if (a == "--p150-root" && i + 1 < argc) p150_root = argv[++i];
        else if (a == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--out-name" && i + 1 < argc) out_name = argv[++i];
        else if (a == "--manifest-name" && i + 1 < argc) manifest_name = argv[++i];
        else if (a == "--raw-cache" && i + 1 < argc) raw_cache = argv[++i];
        else if (a == "--max-main-new" && i + 1 < argc) max_main_new = std::stoi(argv[++i]);
        else if (a == "--max-mtp-new" && i + 1 < argc) max_mtp_new = std::stoi(argv[++i]);
        else if (a == "--gpu-workers" && i + 1 < argc) gpu_workers = std::clamp(std::stoi(argv[++i]), 1, 3);
        else if (a == "--max-minutes" && i + 1 < argc) max_minutes = std::stod(argv[++i]);
        else if (a == "--consume-bins") consume_bins = true;
        else if (a == "--link-check") link_check = true;
    }

    if (link_check) {
        if (!run_self_link_check()) {
            std::fprintf(stderr, "[WAVE8_5_RAREBAL] Self link & streaming .bin check FAILED\n");
            return 1;
        }
        std::printf("[WAVE8_5_RAREBAL] Link & 0x5242414Du streaming .bin sanitizer check OK\n");
        return 0;
    }

    std::error_code ec;
    if (!std::filesystem::is_regular_file(prev_overlay, ec) || !std::filesystem::is_regular_file(prev_manifest, ec)) {
        std::fprintf(stderr, "[WAVE8_5_RAREBAL] ERROR: Missing Wave 8.4 baseline overlay/manifest (%s)\n", prev_overlay.c_str());
        return 1;
    }

    SafetensorsMmap prev_ov(prev_overlay);
    if (!prev_ov.contains("expert_perm_u16") || !prev_ov.contains("p150.hot_l1_assignments") ||
        !prev_ov.contains("p150.hot_l1_logical")) {
        std::fprintf(stderr, "[WAVE8_5_RAREBAL] ERROR: Wave 8.4 overlay missing expert_perm_u16 / L1 metadata\n");
        return 1;
    }

    // 1. Load invariant Wave 8.4 permutation d_expert_perm[40 * 384] (445 swaps) + 18/18 P150A L1 assignments
    std::array<std::array<uint16_t, 384>, 40> perm{};
    std::memcpy(perm.data(), prev_ov.tensor_bytes("expert_perm_u16"), 40ULL * 384ULL * sizeof(uint16_t));
    int swapped_pairs = 0;
    for (int l = 0; l < 40; ++l) {
        for (int e = 0; e < 384; ++e) {
            if (perm[size_t(l)][size_t(e)] > uint16_t(e)) ++swapped_pairs;
        }
    }

    const auto& l1_phys_info = prev_ov.at("p150.hot_l1_assignments");
    const int k_l1_shards = l1_phys_info.shape.empty() ? 0 : static_cast<int>(l1_phys_info.shape[0]);
    const uint8_t* l1_phys_raw = prev_ov.tensor_bytes("p150.hot_l1_assignments");
    const uint8_t* l1_log_raw  = prev_ov.tensor_bytes("p150.hot_l1_logical");
    const auto* l1_log_u16     = reinterpret_cast<const uint16_t*>(l1_log_raw);

    std::array<std::array<bool, 384>, 40> is_l1_logical{};
    for (int i = 0; i < k_l1_shards; ++i) {
        const int lb = int(l1_log_u16[2 * i + 0]);
        const int le = int(l1_log_u16[2 * i + 1]);
        if (lb >= 0 && lb < 40 && le >= 0 && le < 384) is_l1_logical[size_t(lb)][size_t(le)] = true;
    }

    // 2. Load existing Wave 8.4 Dual-Receipt-verified entries for strict comparison / lossless carry-forward
    std::array<std::array<bool, 384>, 43> has_prev_emitted{};
    std::vector<ReceiptRow> prev_rows(43ULL * 384ULL);
    int prev_emitted_cnt = 0;
    {
        std::ifstream pm(prev_manifest);
        std::string line;
        size_t ord = 100000u;
        while (std::getline(pm, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            ReceiptRow rr{};
            int pin_i = 0;
            double d_snr = 0.0;
            std::string status, src;
            if (!(iss >> rr.bid >> rr.eid >> rr.phys_eid >> pin_i >> rr.hits >> rr.m_cal >> rr.m_eval
                      >> rr.m_rare_cal >> rr.m_rare_eval >> rr.lam_in >> rr.lam_mid
                      >> rr.hi.cos_all >> rr.rb.cos_all >> rr.hi.snr_db >> rr.rb.snr_db >> d_snr
                      >> rr.hi.mse_rare >> rr.rb.mse_rare >> status >> src)) {
                continue;
            }
            if (status != "EMITTED") continue;
            if (!(rr.rb.cos_all > rr.hi.cos_all && rr.rb.mse_rare < rr.hi.mse_rare)) continue;
            if (rr.bid < 0 || rr.bid >= 43 || rr.eid < 0 || rr.eid >= 384) continue;
            rr.order_idx = ord++;
            rr.pinned_p150 = (rr.bid < 40 && is_l1_logical[size_t(rr.bid)][size_t(rr.eid)]);
            rr.passed = true;
            rr.source_tag = "WAVE8_4_PRESERVED";
            has_prev_emitted[size_t(rr.bid)][size_t(rr.eid)] = true;
            prev_rows[size_t(rr.bid) * 384ULL + size_t(rr.eid)] = rr;
            ++prev_emitted_cnt;
        }
    }

    std::filesystem::create_directories(out_dir, ec);
    const auto t_start = std::chrono::steady_clock::now();

    // 3. Load & prioritize Wave 8.5 harvested candidates (strict M_cal >= 10,240 + M_eval == 512 -> rows >= 10,752!)
    std::vector<CandEntry> raw_cands;
    {
        std::ifstream cf(cal_dir + "/candidates.txt");
        if (cf.is_open()) {
            CandEntry ce;
            int pin = 0;
            while (cf >> ce.bid >> ce.eid >> pin >> ce.rows >> ce.rare_rows >> ce.hits >> ce.bin_path) {
                const bool is_mtp = (ce.bid >= 40);
                const int phys_e = is_mtp ? ce.eid : int(perm[size_t(ce.bid)][size_t(ce.eid)]);
                ce.pinned_p150 = (!is_mtp && is_l1_logical[size_t(ce.bid)][size_t(ce.eid)]);
                if (ce.pinned_p150) ce.prio_tier = 0;
                else if (!is_mtp && phys_e >= 216 && phys_e < 234) ce.prio_tier = 1;
                else if (is_mtp) ce.prio_tier = 2;
                else ce.prio_tier = 3;

                if (ce.rows >= 10752 && std::filesystem::is_regular_file(ce.bin_path, ec)) {
                    raw_cands.push_back(ce);
                } else if (consume_bins && !ce.bin_path.empty()) {
                    std::filesystem::remove(ce.bin_path, ec);
                }
            }
        }
    }

    std::sort(raw_cands.begin(), raw_cands.end(), [](const CandEntry& a, const CandEntry& b) {
        if (a.prio_tier != b.prio_tier) return a.prio_tier < b.prio_tier;
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

        std::printf("[WAVE8_5_RAREBAL] Solving %zu starvation-safe Wave 8.5 candidates (main=%d, mtp=%d; +%d Wave 8.4 baseline) across %d MI50 GPUs...\n",
                    all_cands.size(), sel_main, sel_mtp, prev_emitted_cnt, n_workers);
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
                const int phys_e = is_mtp ? ce.eid : int(perm[size_t(ce.bid)][size_t(ce.eid)]);
                const auto t0 = std::chrono::steady_clock::now();

                if (!fetch_err.empty()) {
                    std::error_code ferr_ec;
                    if (consume_bins) std::filesystem::remove(ce.bin_path, ferr_ec);
                    std::lock_guard<std::mutex> lk(out_mtx);
                    std::fprintf(stderr, "[WAVE8_5_RAREBAL] WARN fetch b%02d e%03d failed: %s\n", ce.bid, ce.eid, fetch_err.c_str());
                    continue;
                }

                CapturedExpertData cap;
                bool load_ok = true;
                try {
                    cap = load_captured_expert_bin(ce.bin_path);
                } catch (const std::exception& ex) {
                    load_ok = false;
                    std::lock_guard<std::mutex> lk(out_mtx);
                    std::fprintf(stderr, "[WAVE8_5_RAREBAL] WARN bin load b%02d e%03d failed: %s\n", ce.bid, ce.eid, ex.what());
                }
                if (consume_bins) {
                    std::error_code rm_ec;
                    std::filesystem::remove(ce.bin_path, rm_ec);
                }
                if (!load_ok) continue;

                constexpr int kMEval = 512;
                const int M_cal = cap.rows - kMEval;
                // Strict non-starvation gates: M_cal >= 10,240 (gate/up = 2*5120) and M_cal >= 4,608 (down = 2*2304), M_eval == 512
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

                // Strict Dual-Receipt Gate: must beat BOTH H = I AND any existing Wave 8.4 RAREBAL entry!
                bool pass_dual = std::isfinite(m_rb.cos_all) && std::isfinite(m_rb.mse_rare) &&
                                 (m_rb.cos_all > m_hi.cos_all) && (m_rb.mse_rare < m_hi.mse_rare);
                if (pass_dual && has_prev_emitted[size_t(ce.bid)][size_t(ce.eid)]) {
                    const auto& pr = prev_rows[size_t(ce.bid) * 384ULL + size_t(ce.eid)];
                    if (!(m_rb.cos_all > pr.rb.cos_all && m_rb.mse_rare < pr.rb.mse_rare)) {
                        pass_dual = false; // Wave 8.4 achieved higher cos / lower rare-MSE -> carry forward Wave 8.4 unchanged!
                    }
                }

                ReceiptRow rr{idx, ce.bid, ce.eid, phys_e, ce.pinned_p150, ce.hits, M_cal, kMEval,
                              st_in.m_rare, m_rb.rare_cnt, st_in.lambda, st_mid.lambda, m_hi, m_rb, pass_dual, "WAVE8_5_NEW"};

                std::vector<PackedOverlayTensor> local_tensors;
                if (pass_dual) {
                    ExpertHlut2Triplet tr{qg_rb, qu_rb, qd_rb};
                    pack_expert_into_overlay_tensors(tr, ce.bid, phys_e, local_tensors);
                }

                const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                {
                    std::lock_guard<std::mutex> lk(out_mtx);
                    if (pass_dual || !has_prev_emitted[size_t(ce.bid)][size_t(ce.eid)]) {
                        receipts.push_back(rr);
                    }
                    if (pass_dual) {
                        emitted_logical[size_t(ce.bid)][size_t(ce.eid)] = true;
                        for (auto& te : local_tensors) {
                            writer.add_owned(te.name, te.dtype, te.shape, std::move(te.buf));
                        }
                        if (is_mtp) ++new_mtp_ok;
                        else        ++new_main_ok;
                    }
                    std::printf("[RAREBAL85 %4zu/%4zu gpu%d] b%02d e%03d->p%03d[T%d] | M_cal=%d | cos %.5f->%.5f | SNR +%.2fdB | rareMSE %.3e->%.3e | %s (%.2fs)\n",
                                idx + 1, all_cands.size(), wid, ce.bid, ce.eid, phys_e, ce.prio_tier,
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

    // 4. Carry forward 100% of verified Wave 8.4 RAREBAL entries not strictly superseded by Wave 8.5 (zero repack needed!)
    int preserved_w84 = 0;
    for (int b = 0; b < 43; ++b) {
        const int n_exp = (b < 40) ? 384 : 128;
        for (int e = 0; e < n_exp; ++e) {
            if (!has_prev_emitted[size_t(b)][size_t(e)] || emitted_logical[size_t(b)][size_t(e)]) continue;
            const ReceiptRow& rr = prev_rows[size_t(b) * 384ULL + size_t(e)];
            copy_phys_expert_from_overlay(writer, prev_ov, b, rr.phys_eid);
            emitted_logical[size_t(b)][size_t(e)] = true;
            receipts.push_back(rr);
            ++preserved_w84;
        }
    }

    // 5. Carry forward remaining uncalibrated cross-tier swapped experts (perm[l][e] != e) directly from Wave 8.4 overlay!
    int base_swapped_copied = 0;
    for (int l = 0; l < 40; ++l) {
        for (int e = 0; e < 384; ++e) {
            const int pe = int(perm[size_t(l)][size_t(e)]);
            if (pe != e && !emitted_logical[size_t(l)][size_t(e)]) {
                copy_phys_expert_from_overlay(writer, prev_ov, l, pe);
                ++base_swapped_copied;
            }
        }
    }

    // 6. Emit invariant expert_perm_u16 [40, 384] + p150.hot_l1_assignments [18, 2] + p150.hot_l1_logical [18, 2]
    {
        std::vector<uint8_t> perm_bytes(40ULL * 384ULL * sizeof(uint16_t));
        std::memcpy(perm_bytes.data(), perm.data(), perm_bytes.size());
        writer.add_owned("expert_perm_u16", "U16", {40, 384}, std::move(perm_bytes));

        const size_t l1_bytes_len = static_cast<size_t>(k_l1_shards) * 2ULL * sizeof(uint16_t);
        writer.add_owned("p150.hot_l1_assignments", "U16", {k_l1_shards, 2},
                         std::vector<uint8_t>(l1_phys_raw, l1_phys_raw + l1_bytes_len));
        writer.add_owned("p150.hot_l1_logical", "U16", {k_l1_shards, 2},
                         std::vector<uint8_t>(l1_log_raw, l1_log_raw + l1_bytes_len));
    }

    // Clean up raw MXFP4 range cache BEFORE writing the final safetensors overlay!
    if (raw_cache.rfind("/dev/shm/", 0) == 0) {
        std::filesystem::remove_all(raw_cache, ec);
    }

    const std::string st_out = out_dir + "/" + out_name;
    writer.write_atomic(st_out);

    int tot_rb_main = 0, tot_rb_mtp = 0, tot_rb_l1 = 0, dual_violations = 0;
    double sum_dsnr = 0.0, sum_cos_hi = 0.0, sum_cos_rb = 0.0;
    for (const auto& r : receipts) {
        if (!r.passed) continue;
        if (!(r.rb.cos_all > r.hi.cos_all && r.rb.mse_rare < r.hi.mse_rare)) ++dual_violations;
        sum_dsnr += (r.rb.snr_db - r.hi.snr_db);
        sum_cos_hi += r.hi.cos_all;
        sum_cos_rb += r.rb.cos_all;
        if (r.bid >= 40) ++tot_rb_mtp;
        else {
            ++tot_rb_main;
            if (is_l1_logical[size_t(r.bid)][size_t(r.eid)]) ++tot_rb_l1;
        }
    }
    const int tot_emitted = tot_rb_main + tot_rb_mtp;

    std::sort(receipts.begin(), receipts.end(), [](const ReceiptRow& a, const ReceiptRow& b) {
        if (a.passed != b.passed) return a.passed > b.passed;
        if (a.pinned_p150 != b.pinned_p150) return a.pinned_p150 > b.pinned_p150;
        return a.order_idx < b.order_idx;
    });

    {
        std::ofstream mf(out_dir + "/" + manifest_name, std::ios::trunc);
        mf << "# Wave 8.5 2,000+ Expanded Starvation-Safe TARQ RAREBAL-LDLQ Overlay Receipt Manifest\n"
           << "# emitted_rarebal_total=" << tot_emitted
           << " emitted_rarebal_main=" << tot_rb_main
           << " emitted_p150_l1_pinned=" << tot_rb_l1 << "/" << k_l1_shards
           << " emitted_p150_l1_tp2_shards=" << (tot_rb_l1 * 2) << "/" << (k_l1_shards * 2)
           << " emitted_rarebal_mtp=" << tot_rb_mtp
           << " dual_receipt_violations=" << dual_violations
           << " swapped_pairs=" << swapped_pairs
           << " base_swapped_repacked=" << base_swapped_copied
           << " mean_cos_hi=" << std::fixed << std::setprecision(5) << (tot_emitted > 0 ? sum_cos_hi / tot_emitted : 0.0)
           << " mean_cos_rb=" << (tot_emitted > 0 ? sum_cos_rb / tot_emitted : 0.0)
           << " mean_dsnr_db=" << (tot_emitted > 0 ? sum_dsnr / tot_emitted : 0.0) << "\n"
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

    const double tot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("[WAVE8_5_RAREBAL] DONE in %.2fs -> %s\n"
                "  RAREBAL-verified experts : %d total (main=%d [new=%d, preserved=%d], MTP=%d [new=%d], P150_L1=%d/%d -> %d/%d TP2 shards, violations=%d)\n"
                "  Mean held-out metrics    : cos %.5f -> %.5f | mean +%.2f dB SNR\n"
                "  Base cross-tier copied   : %d uncalibrated swapped experts (invariant 445-swap layout)\n",
                tot_s, st_out.c_str(),
                tot_emitted, tot_rb_main, new_main_ok.load(), preserved_w84,
                tot_rb_mtp, new_mtp_ok.load(), tot_rb_l1, k_l1_shards, tot_rb_l1 * 2, k_l1_shards * 2, dual_violations,
                tot_emitted > 0 ? sum_cos_hi / tot_emitted : 0.0,
                tot_emitted > 0 ? sum_cos_rb / tot_emitted : 0.0,
                tot_emitted > 0 ? sum_dsnr / tot_emitted : 0.0,
                base_swapped_copied);
    return (dual_violations == 0 && tot_rb_l1 == k_l1_shards && tot_emitted >= 2000) ? 0 : 2;
}
