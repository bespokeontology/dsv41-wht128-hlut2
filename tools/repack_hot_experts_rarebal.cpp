// Wave 8 Track 8C: Expanded Hot-Expert TARQ RAREBAL-LDLQ Offline Repack Overlay Builder
// Pure C++17 + 3x MI50 rocBLAS + OpenMP + AVX2 + 32-Thread Async HTTP/2 Range Prefetcher, zero Python anywhere.
// Enforces M_cal >= 11,264 (> 2*K) + M_eval = 512 held-out real routed activations AND Mandatory Dual Receipt Gate,
// prioritizing 17/17 (34/34 TP2 shards) P150A L1-pinned hot experts first and scaling to 512-2,000+ hot experts.

#include "repack_rarebal_helpers.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <thread>

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
    int eid = 0;
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

}  // namespace

int main(int argc, char** argv) {
    const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                     ? std::string(std::getenv("DSV41_ROOT"))
                                     : std::string(".");
    std::string cal_dir       = "/dev/shm/dsv41_rarebal_cal_wave8";
    std::string mi50_root     = root_env + "/packed_mi50";
    std::string out_dir       = root_env + "/packed_rarebal_overlay_wave8";
    std::string out_name      = "hot_experts_rarebal_wave8.safetensors";
    std::string manifest_name = "rarebal_wave8_receipt_manifest.txt";
    std::string raw_cache     = "/dev/shm/dsv41_rarebal_raw";
    int max_main              = 640;
    int max_mtp               = 80;
    int gpu_workers           = 3;
    double max_minutes        = 45.0;
    bool consume_bins         = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--cal-dir" && i + 1 < argc) cal_dir = argv[++i];
        else if (a == "--mi50-root" && i + 1 < argc) mi50_root = argv[++i];
        else if (a == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--out-name" && i + 1 < argc) out_name = argv[++i];
        else if (a == "--manifest-name" && i + 1 < argc) manifest_name = argv[++i];
        else if (a == "--raw-cache" && i + 1 < argc) raw_cache = argv[++i];
        else if (a == "--max-main" && i + 1 < argc) max_main = std::stoi(argv[++i]);
        else if (a == "--max-mtp" && i + 1 < argc) max_mtp = std::stoi(argv[++i]);
        else if (a == "--gpu-workers" && i + 1 < argc) gpu_workers = std::max(1, std::min(3, std::stoi(argv[++i])));
        else if (a == "--max-minutes" && i + 1 < argc) max_minutes = std::stod(argv[++i]);
        else if (a == "--consume-bins") consume_bins = true;
    }

    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);

    std::vector<CandEntry> raw_cands;
    {
        std::ifstream cf(cal_dir + "/candidates.txt");
        if (!cf) { std::fprintf(stderr, "Missing %s/candidates.txt\n", cal_dir.c_str()); return 1; }
        CandEntry ce; int pin = 0;
        while (cf >> ce.bid >> ce.eid >> pin >> ce.rows >> ce.rare_rows >> ce.hits >> ce.bin_path) {
            ce.pinned_p150 = (pin != 0);
            if (ce.rows >= 11776) raw_cands.push_back(ce);  // Strict M_cal >= 11,264 + M_eval = 512 starvation gate!
        }
    }
    // Order: 17/17 P150A L1-pinned hot shards FIRST, then strictly descending router hits across all 43 blocks!
    std::sort(raw_cands.begin(), raw_cands.end(), [](const CandEntry& a, const CandEntry& b) {
        if (a.pinned_p150 != b.pinned_p150) return a.pinned_p150 > b.pinned_p150;
        if (a.hits != b.hits) return a.hits > b.hits;
        return (a.bid != b.bid) ? (a.bid < b.bid) : (a.eid < b.eid);
    });

    std::vector<CandEntry> all_cands;
    all_cands.reserve(raw_cands.size());
    int sel_main = 0, sel_mtp = 0;
    for (const auto& ce : raw_cands) {
        if (ce.bid < 40) {
            if (sel_main >= max_main && !ce.pinned_p150) continue;
            ++sel_main;
        } else {
            if (sel_mtp >= max_mtp) continue;
            ++sel_mtp;
        }
        all_cands.push_back(ce);
    }

    int dev_cnt = 0;
    if (hipGetDeviceCount(&dev_cnt) != hipSuccess || dev_cnt <= 0) dev_cnt = 1;
    const int n_workers = std::min(gpu_workers, dev_cnt);

    std::printf("[RAREBAL_REPACK] Candidates passing M_rows >= 11,776: %zu (selected=%zu: main=%d, mtp=%d, mi50_workers=%d, max_min=%.1f)\n",
                raw_cands.size(), all_cands.size(), sel_main, sel_mtp, n_workers, max_minutes);
    std::fflush(stdout);

    std::array<std::vector<uint64_t>, 43> block_signs_h{};
    std::array<std::vector<uint64_t>, 43> block_signs_i{};
    for (int b = 0; b < 43; ++b) {
        load_block_wht_signs(mi50_root, b, block_signs_h[size_t(b)], block_signs_i[size_t(b)]);
    }

    const auto t_start = std::chrono::steady_clock::now();
    HfShardRangeFetcher fetcher(raw_cache);
    AsyncRawPrefetcher prefetcher(fetcher, all_cands, 32);

    std::mutex out_mtx;
    StreamingSafetensorsWriter writer;
    std::vector<ReceiptRow> receipts;
    std::atomic<int> emitted_main{0}, emitted_pinned{0}, emitted_mtp{0};

    auto run_gpu_worker = [&](int wid) {
        (void)hipSetDevice(wid);
#ifdef _OPENMP
        omp_set_num_threads(std::max(4, 48 / n_workers));
#endif
        GpuRarebalAccel gpu_accel(wid);
        {
            std::lock_guard<std::mutex> lk(out_mtx);
            std::printf("[RAREBAL_REPACK] Worker #%d bound to MI50 GPU %d (rocBLAS=%s, omp_threads=%d)\n",
                        wid, wid, gpu_accel.ok() ? "ON" : "CPU_FALLBACK",
#ifdef _OPENMP
                        std::max(4, 48 / n_workers)
#else
                        1
#endif
            );
            std::fflush(stdout);
        }

        for (;;) {
            size_t idx = 0;
            RawExpertMxfp4 raw;
            std::string fetch_err;
            if (!prefetcher.take_next(idx, raw, fetch_err, max_minutes, t_start)) break;

            const CandEntry& ce = all_cands[idx];
            const bool is_mtp = (ce.bid >= 40);
            const auto t0 = std::chrono::steady_clock::now();

            if (!fetch_err.empty()) {
                std::lock_guard<std::mutex> lk(out_mtx);
                std::fprintf(stderr, "[RAREBAL_REPACK] WARN fetch b%02d e%03d failed: %s\n", ce.bid, ce.eid, fetch_err.c_str());
                continue;
            }

            std::error_code loc_ec;
            const CapturedExpertData cap = load_captured_expert_bin(ce.bin_path);
            if (consume_bins) {
                std::filesystem::remove(ce.bin_path, loc_ec);
            }
            constexpr int kMEval = 512;
            const int M_cal = cap.rows - kMEval;
            if (M_cal < 11264) continue;

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

            // Mandatory Dual Receipt Gate on held-out M_eval = 512 real routed activations
            std::vector<float> X_eval_rot(size_t(kMEval) * DSV41_H), Y_ref, Y_hi, Y_rb, wg_hi_r, wu_hi_r, wd_hi_r, wd_rb_r;
            wht128_rows(X_eval.data(), X_eval_rot.data(), kMEval, DSV41_H, signs_h);
            (void)eval_rot_swiglu(X_eval_rot.data(), rare_eval.data(), kMEval, w1_rot.data(), w3_rot.data(), w2_rot.data(), signs_i, nullptr, Y_ref, nullptr, &gpu_accel);
            dequantize_hlut2_to_rot_f32(qg_hi, wg_hi_r); dequantize_hlut2_to_rot_f32(qu_hi, wu_hi_r); dequantize_hlut2_to_rot_f32(qd_hi, wd_hi_r);
            dequantize_hlut2_to_rot_f32(qd_rb, wd_rb_r);
            const auto m_hi = eval_rot_swiglu(X_eval_rot.data(), rare_eval.data(), kMEval, wg_hi_r.data(), wu_hi_r.data(), wd_hi_r.data(), signs_i, &Y_ref, Y_hi, nullptr, &gpu_accel);
            const auto m_rb = eval_rot_swiglu(X_eval_rot.data(), rare_eval.data(), kMEval, wg_rb_r.data(), wu_rb_r.data(), wd_rb_r.data(), signs_i, &Y_ref, Y_rb, nullptr, &gpu_accel);

            const bool pass_dual = (m_rb.cos_all > m_hi.cos_all) && (m_rb.mse_rare < m_hi.mse_rare);
            ReceiptRow rr{idx, ce.bid, ce.eid, ce.pinned_p150, ce.hits, M_cal, kMEval, st_in.m_rare, m_rb.rare_cnt, st_in.lambda, st_mid.lambda, m_hi, m_rb, pass_dual};

            struct PackedEntry {
                std::string name;
                std::string dtype;
                std::vector<int64_t> shape;
                std::vector<uint8_t> buf;
            };
            std::vector<PackedEntry> local_tensors;
            if (pass_dual) {
                if (is_mtp || ce.eid < 216) {
                    local_tensors.reserve(27);
                    for (int tp = 0; tp < 3; ++tp) {
                        std::vector<uint8_t> gc(983040), gs(61440), gcent(8), uc(983040), us(61440), ucent(8), dc(983040), ds(61440), dcent(8);
                        pack_hlut2_mi50_slot_st(slice_hlut2_rows(qg_rb, tp * 768, 768), reinterpret_cast<uint32_t*>(gc.data()), reinterpret_cast<uint16_t*>(gs.data()), reinterpret_cast<float*>(gcent.data()));
                        pack_hlut2_mi50_slot_st(slice_hlut2_rows(qu_rb, tp * 768, 768), reinterpret_cast<uint32_t*>(uc.data()), reinterpret_cast<uint16_t*>(us.data()), reinterpret_cast<float*>(ucent.data()));
                        pack_hlut2_mi50_slot_st(slice_hlut2_cols(qd_rb, tp * 768, 768), reinterpret_cast<uint32_t*>(dc.data()), reinterpret_cast<uint16_t*>(ds.data()), reinterpret_cast<float*>(dcent.data()));
                        const std::string p = "mi50.b" + std::to_string(ce.bid) + ".e" + std::to_string(ce.eid) + ".tp" + std::to_string(tp);
                        local_tensors.push_back({p + ".gate_codes", "U32", {24, 320, 32}, std::move(gc)});
                        local_tensors.push_back({p + ".gate_scales", "F16", {24, 40, 32}, std::move(gs)});
                        local_tensors.push_back({p + ".gate_centroids", "F32", {2}, std::move(gcent)});
                        local_tensors.push_back({p + ".up_codes", "U32", {24, 320, 32}, std::move(uc)});
                        local_tensors.push_back({p + ".up_scales", "F16", {24, 40, 32}, std::move(us)});
                        local_tensors.push_back({p + ".up_centroids", "F32", {2}, std::move(ucent)});
                        local_tensors.push_back({p + ".down_codes", "U32", {160, 48, 32}, std::move(dc)});
                        local_tensors.push_back({p + ".down_scales", "F16", {160, 6, 32}, std::move(ds)});
                        local_tensors.push_back({p + ".down_centroids", "F32", {2}, std::move(dcent)});
                    }
                } else {
                    local_tensors.reserve(12);
                    const int le = ce.eid - 216;
                    for (int tp = 0; tp < 2; ++tp) {
                        std::vector<uint8_t> gp(360ULL * 4352ULL), gc(8), up(360ULL * 4352ULL), uc(8), dp(360ULL * 4352ULL), dc(8);
                        pack_hlut2_p150_slot_st(qg_rb, tp * 1152, 1152, gp.data(), reinterpret_cast<float*>(gc.data()));
                        pack_hlut2_p150_slot_st(qu_rb, tp * 1152, 1152, up.data(), reinterpret_cast<float*>(uc.data()));
                        pack_hlut2_p150_slot_st(slice_hlut2_cols(qd_rb, tp * 1152, 1152), 0, 5120, dp.data(), reinterpret_cast<float*>(dc.data()));
                        const std::string p = "p150.l" + std::to_string(ce.bid) + ".e" + std::to_string(le) + ".tp" + std::to_string(tp);
                        local_tensors.push_back({p + ".gate_pages", "U8", {360, 4352}, std::move(gp)});
                        local_tensors.push_back({p + ".gate_centroids", "F32", {2}, std::move(gc)});
                        local_tensors.push_back({p + ".up_pages", "U8", {360, 4352}, std::move(up)});
                        local_tensors.push_back({p + ".up_centroids", "F32", {2}, std::move(uc)});
                        local_tensors.push_back({p + ".down_pages", "U8", {360, 4352}, std::move(dp)});
                        local_tensors.push_back({p + ".down_centroids", "F32", {2}, std::move(dc)});
                    }
                }
            }

            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            {
                std::lock_guard<std::mutex> lk(out_mtx);
                receipts.push_back(rr);
                for (auto& te : local_tensors) {
                    writer.add_owned(te.name, te.dtype, te.shape, std::move(te.buf));
                }
                if (pass_dual) {
                    if (is_mtp) ++emitted_mtp;
                    else { ++emitted_main; if (ce.pinned_p150) ++emitted_pinned; }
                }
                std::printf("[RAREBAL %3zu/%3zu gpu%d] b%02d e%03d%s | M_cal=%d lam=(%.2f,%.2f) | cos %.5f->%.5f | SNR %.2f->%.2f dB (+%.2f) | rareMSE %.3e->%.3e | %s (%.2fs)\n",
                            idx + 1, all_cands.size(), wid, ce.bid, ce.eid, ce.pinned_p150 ? "[L1]" : "    ",
                            M_cal, st_in.lambda, st_mid.lambda,
                            m_hi.cos_all, m_rb.cos_all, m_hi.snr_db, m_rb.snr_db, m_rb.snr_db - m_hi.snr_db,
                            m_hi.mse_rare, m_rb.mse_rare, pass_dual ? "EMIT" : "SKIP", dt);
                std::fflush(stdout);
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(size_t(n_workers));
    for (int w = 0; w < n_workers; ++w) {
        pool.emplace_back(run_gpu_worker, w);
    }
    for (auto& th : pool) {
        if (th.joinable()) th.join();
    }
    prefetcher.stop();

    std::sort(receipts.begin(), receipts.end(), [](const ReceiptRow& a, const ReceiptRow& b) {
        return a.order_idx < b.order_idx;
    });

    const std::string st_out = out_dir + "/" + out_name;
    writer.write_atomic(st_out);
    std::filesystem::copy_file(cal_dir + "/router_hist.txt", out_dir + "/router_hist.txt",
                               std::filesystem::copy_options::overwrite_existing, ec);

    const int n_main = emitted_main.load(), n_pin = emitted_pinned.load(), n_mtp = emitted_mtp.load();
    std::ofstream mf(out_dir + "/" + manifest_name, std::ios::trunc);
    mf << "# Wave 8 Track 8C Expanded TARQ RAREBAL-LDLQ Hot-Expert Overlay Receipt Manifest\n"
       << "# emitted_total=" << (n_main + n_mtp)
       << " emitted_main=" << n_main
       << " emitted_p150_l1_pinned=" << n_pin
       << " emitted_p150_l1_tp2_shards=" << (n_pin * 2)
       << " emitted_mtp=" << n_mtp << "\n"
       << "# block eid pinned hits M_cal M_eval rare_cal rare_eval lam_in lam_mid cos_hi cos_rb snr_hi snr_rb d_snr_db rare_mse_hi rare_mse_rb status\n";
    for (const auto& r : receipts) {
        mf << r.bid << " " << r.eid << " " << int(r.pinned_p150) << " " << r.hits << " " << r.m_cal << " " << r.m_eval << " "
           << r.m_rare_cal << " " << r.m_rare_eval << " " << std::fixed << std::setprecision(5)
           << r.lam_in << " " << r.lam_mid << " " << r.hi.cos_all << " " << r.rb.cos_all << " "
           << r.hi.snr_db << " " << r.rb.snr_db << " " << (r.rb.snr_db - r.hi.snr_db) << " "
           << std::scientific << r.hi.mse_rare << " " << r.rb.mse_rare << " " << (r.passed ? "EMITTED" : "SKIPPED") << "\n";
    }
    const double tot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("[RAREBAL_REPACK] DONE in %.2fs -> %s (total=%d: main=%d [p150_l1=%d -> %d TP2 shards], mtp=%d)\n",
                tot_s, st_out.c_str(), n_main + n_mtp, n_main,
                n_pin, n_pin * 2, n_mtp);
    return 0;
}
