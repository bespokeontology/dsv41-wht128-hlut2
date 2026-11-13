// Wave 6B: Selective Hot-Expert TARQ RAREBAL-LDLQ Offline Repack Overlay Builder
// Pure C++17 + OpenMP + AVX2, zero Python anywhere, <= 250 lines.
// Enforces M_cal >= 11,264 (> 2*K) + M_eval = 512 held-out real routed activations AND Mandatory Dual Receipt Gate.

#include "repack_rarebal_helpers.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

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

}  // namespace

int main(int argc, char** argv) {
    const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                     ? std::string(std::getenv("DSV41_ROOT"))
                                     : std::string(".");
    std::string cal_dir   = "/dev/shm/dsv41_rarebal_cal";
    std::string mi50_root = root_env + "/packed_mi50";
    std::string out_dir   = root_env + "/packed_rarebal_overlay";
    std::string raw_cache = "/dev/shm/dsv41_rarebal_raw";
    int max_main = 44;
    int max_mtp  = 12;

    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--cal-dir" && i + 1 < argc) cal_dir = argv[++i];
        else if (a == "--mi50-root" && i + 1 < argc) mi50_root = argv[++i];
        else if (a == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--raw-cache" && i + 1 < argc) raw_cache = argv[++i];
        else if (a == "--max-main" && i + 1 < argc) max_main = std::stoi(argv[++i]);
        else if (a == "--max-mtp" && i + 1 < argc) max_mtp = std::stoi(argv[++i]);
    }

    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);

    std::vector<CandEntry> all_cands;
    {
        std::ifstream cf(cal_dir + "/candidates.txt");
        if (!cf) { std::fprintf(stderr, "Missing %s/candidates.txt\n", cal_dir.c_str()); return 1; }
        CandEntry ce; int pin = 0;
        while (cf >> ce.bid >> ce.eid >> pin >> ce.rows >> ce.rare_rows >> ce.hits >> ce.bin_path) {
            ce.pinned_p150 = (pin != 0);
            if (ce.rows >= 11776) all_cands.push_back(ce);  // Strict M_cal >= 11,264 + M_eval = 512 starvation gate!
        }
    }
    std::sort(all_cands.begin(), all_cands.end(), [](const CandEntry& a, const CandEntry& b) {
        const bool am = (a.bid < 40), bm = (b.bid < 40);
        if (am != bm) return am > bm;
        if (a.pinned_p150 != b.pinned_p150) return a.pinned_p150 > b.pinned_p150;
        return a.hits > b.hits;
    });

    std::printf("[RAREBAL_REPACK] Candidates passing M_rows >= 11,776 (M_cal >= 11,264 + M_eval = 512): %zu\n", all_cands.size());
    std::fflush(stdout);

    HfShardRangeFetcher fetcher(raw_cache);
    StreamingSafetensorsWriter writer;
    std::vector<ReceiptRow> receipts;
    int emitted_main = 0, emitted_pinned = 0, emitted_mtp = 0;
    const auto t_start = std::chrono::steady_clock::now();

    for (const auto& ce : all_cands) {
        const bool is_mtp = (ce.bid >= 40);
        if (!is_mtp && emitted_main >= max_main) continue;
        if (is_mtp && emitted_mtp >= max_mtp) continue;

        const auto t0 = std::chrono::steady_clock::now();
        const CapturedExpertData cap = load_captured_expert_bin(ce.bin_path);
        constexpr int kMEval = 512;
        const int M_cal = cap.rows - kMEval;
        if (M_cal < 11264) continue;

        // Disjoint uniform-stride held-out split across full C++/DSML/Math trajectory
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

        std::vector<uint64_t> signs_h, signs_i;
        load_block_wht_signs(mi50_root, ce.bid, signs_h, signs_i);
        const RawExpertMxfp4 raw = fetcher.fetch_expert(ce.bid, ce.eid);

        std::vector<float> w1_rot, w3_rot, w2_rot, scratch_sig;
        const Hlut2QuantMatrix qg_hi = dequant_mxfp4_and_quantize_hlut2_st(raw.w1_w.data(), raw.w1_s.data(), DSV41_I, DSV41_H, signs_h, w1_rot, scratch_sig);
        const Hlut2QuantMatrix qu_hi = dequant_mxfp4_and_quantize_hlut2_st(raw.w3_w.data(), raw.w3_s.data(), DSV41_I, DSV41_H, signs_h, w3_rot, scratch_sig);
        const Hlut2QuantMatrix qd_hi = dequant_mxfp4_and_quantize_hlut2_st(raw.w2_w.data(), raw.w2_s.data(), DSV41_H, DSV41_I, signs_i, w2_rot, scratch_sig);

        // Shared R_up_in across gate (w1) and up (w3), then R_up_mid on quantized SwiGLU intermediate Z_cal!
        Hlut2RarebalStats st_in{}, st_mid{};
        const auto R_up_in = build_rarebal_cholesky_rup(X_cal.data(), M_cal, DSV41_H, signs_h, true, rare_cal.data(), &st_in);
        if (R_up_in.empty()) continue;
        const Hlut2QuantMatrix qg_rb = apply_hlut2_ldlq_with_rup(qg_hi, w1_rot.data(), R_up_in);
        const Hlut2QuantMatrix qu_rb = apply_hlut2_ldlq_with_rup(qu_hi, w3_rot.data(), R_up_in);

        std::vector<float> wg_rb_r, wu_rb_r, X_cal_rot(size_t(M_cal) * DSV41_H), y_tmp, Z_cal;
        dequantize_hlut2_to_rot_f32(qg_rb, wg_rb_r);
        dequantize_hlut2_to_rot_f32(qu_rb, wu_rb_r);
        wht128_rows(X_cal.data(), X_cal_rot.data(), M_cal, DSV41_H, signs_h);
        (void)eval_rot_swiglu(X_cal_rot.data(), rare_cal.data(), M_cal, wg_rb_r.data(), wu_rb_r.data(), w2_rot.data(), signs_i, nullptr, y_tmp, &Z_cal);

        const auto R_up_mid = build_rarebal_cholesky_rup(Z_cal.data(), M_cal, DSV41_I, signs_i, true, rare_cal.data(), &st_mid);
        if (R_up_mid.empty()) continue;
        const Hlut2QuantMatrix qd_rb = apply_hlut2_ldlq_with_rup(qd_hi, w2_rot.data(), R_up_mid);

        // Mandatory Dual Receipt Gate on held-out M_eval = 512 real routed activations
        std::vector<float> X_eval_rot(size_t(kMEval) * DSV41_H), Y_ref, Y_hi, Y_rb, wg_hi_r, wu_hi_r, wd_hi_r, wd_rb_r;
        wht128_rows(X_eval.data(), X_eval_rot.data(), kMEval, DSV41_H, signs_h);
        (void)eval_rot_swiglu(X_eval_rot.data(), rare_eval.data(), kMEval, w1_rot.data(), w3_rot.data(), w2_rot.data(), signs_i, nullptr, Y_ref);
        dequantize_hlut2_to_rot_f32(qg_hi, wg_hi_r); dequantize_hlut2_to_rot_f32(qu_hi, wu_hi_r); dequantize_hlut2_to_rot_f32(qd_hi, wd_hi_r);
        dequantize_hlut2_to_rot_f32(qd_rb, wd_rb_r);
        const auto m_hi = eval_rot_swiglu(X_eval_rot.data(), rare_eval.data(), kMEval, wg_hi_r.data(), wu_hi_r.data(), wd_hi_r.data(), signs_i, &Y_ref, Y_hi);
        const auto m_rb = eval_rot_swiglu(X_eval_rot.data(), rare_eval.data(), kMEval, wg_rb_r.data(), wu_rb_r.data(), wd_rb_r.data(), signs_i, &Y_ref, Y_rb);

        const bool pass_dual = (m_rb.cos_all > m_hi.cos_all) && (m_rb.mse_rare < m_hi.mse_rare);
        ReceiptRow rr{ce.bid, ce.eid, ce.pinned_p150, ce.hits, M_cal, kMEval, st_in.m_rare, m_rb.rare_cnt, st_in.lambda, st_mid.lambda, m_hi, m_rb, pass_dual};
        receipts.push_back(rr);

        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("[RAREBAL] b%02d e%03d%s | M_cal=%d lam=(%.2f,%.2f) | cos %.5f->%.5f | SNR %.2f->%.2f dB (+%.2f) | rareMSE %.3e->%.3e | %s (%.2fs)\n",
                    ce.bid, ce.eid, ce.pinned_p150 ? "[L1]" : "    ", M_cal, st_in.lambda, st_mid.lambda,
                    m_hi.cos_all, m_rb.cos_all, m_hi.snr_db, m_rb.snr_db, m_rb.snr_db - m_hi.snr_db,
                    m_hi.mse_rare, m_rb.mse_rare, pass_dual ? "EMIT" : "SKIP", dt);
        std::fflush(stdout);
        if (!pass_dual) continue;

        if (is_mtp || ce.eid < 216) {
            for (int tp = 0; tp < 3; ++tp) {
                std::vector<uint8_t> gc(983040), gs(61440), gcent(8), uc(983040), us(61440), ucent(8), dc(983040), ds(61440), dcent(8);
                pack_hlut2_mi50_slot_st(slice_hlut2_rows(qg_rb, tp * 768, 768), reinterpret_cast<uint32_t*>(gc.data()), reinterpret_cast<uint16_t*>(gs.data()), reinterpret_cast<float*>(gcent.data()));
                pack_hlut2_mi50_slot_st(slice_hlut2_rows(qu_rb, tp * 768, 768), reinterpret_cast<uint32_t*>(uc.data()), reinterpret_cast<uint16_t*>(us.data()), reinterpret_cast<float*>(ucent.data()));
                pack_hlut2_mi50_slot_st(slice_hlut2_cols(qd_rb, tp * 768, 768), reinterpret_cast<uint32_t*>(dc.data()), reinterpret_cast<uint16_t*>(ds.data()), reinterpret_cast<float*>(dcent.data()));
                const std::string p = "mi50.b" + std::to_string(ce.bid) + ".e" + std::to_string(ce.eid) + ".tp" + std::to_string(tp);
                writer.add_owned(p + ".gate_codes", "U32", {24, 320, 32}, std::move(gc));
                writer.add_owned(p + ".gate_scales", "F16", {24, 40, 32}, std::move(gs));
                writer.add_owned(p + ".gate_centroids", "F32", {2}, std::move(gcent));
                writer.add_owned(p + ".up_codes", "U32", {24, 320, 32}, std::move(uc));
                writer.add_owned(p + ".up_scales", "F16", {24, 40, 32}, std::move(us));
                writer.add_owned(p + ".up_centroids", "F32", {2}, std::move(ucent));
                writer.add_owned(p + ".down_codes", "U32", {160, 48, 32}, std::move(dc));
                writer.add_owned(p + ".down_scales", "F16", {160, 6, 32}, std::move(ds));
                writer.add_owned(p + ".down_centroids", "F32", {2}, std::move(dcent));
            }
        } else {
            const int le = ce.eid - 216;
            for (int tp = 0; tp < 2; ++tp) {
                std::vector<uint8_t> gp(360ULL * 4352ULL), gc(8), up(360ULL * 4352ULL), uc(8), dp(360ULL * 4352ULL), dc(8);
                pack_hlut2_p150_slot_st(qg_rb, tp * 1152, 1152, gp.data(), reinterpret_cast<float*>(gc.data()));
                pack_hlut2_p150_slot_st(qu_rb, tp * 1152, 1152, up.data(), reinterpret_cast<float*>(uc.data()));
                pack_hlut2_p150_slot_st(slice_hlut2_cols(qd_rb, tp * 1152, 1152), 0, 5120, dp.data(), reinterpret_cast<float*>(dc.data()));
                const std::string p = "p150.l" + std::to_string(ce.bid) + ".e" + std::to_string(le) + ".tp" + std::to_string(tp);
                writer.add_owned(p + ".gate_pages", "U8", {360, 4352}, std::move(gp));
                writer.add_owned(p + ".gate_centroids", "F32", {2}, std::move(gc));
                writer.add_owned(p + ".up_pages", "U8", {360, 4352}, std::move(up));
                writer.add_owned(p + ".up_centroids", "F32", {2}, std::move(uc));
                writer.add_owned(p + ".down_pages", "U8", {360, 4352}, std::move(dp));
                writer.add_owned(p + ".down_centroids", "F32", {2}, std::move(dc));
            }
        }
        if (is_mtp) ++emitted_mtp;
        else { ++emitted_main; if (ce.pinned_p150) ++emitted_pinned; }
    }

    const std::string st_out = out_dir + "/hot_experts_rarebal.safetensors";
    writer.write_atomic(st_out);
    std::filesystem::copy_file(cal_dir + "/router_hist.txt", out_dir + "/router_hist.txt",
                               std::filesystem::copy_options::overwrite_existing, ec);

    std::ofstream mf(out_dir + "/rarebal_receipt_manifest.txt", std::ios::trunc);
    mf << "# Wave 6B Selective RAREBAL-LDLQ Hot-Expert Overlay Receipt Manifest\n"
       << "# emitted_main=" << emitted_main << " emitted_p150_l1_pinned=" << emitted_pinned << " emitted_mtp=" << emitted_mtp << "\n"
       << "# block eid pinned hits M_cal M_eval rare_cal rare_eval lam_in lam_mid cos_hi cos_rb snr_hi snr_rb d_snr_db rare_mse_hi rare_mse_rb status\n";
    for (const auto& r : receipts) {
        mf << r.bid << " " << r.eid << " " << int(r.pinned_p150) << " " << r.hits << " " << r.m_cal << " " << r.m_eval << " "
           << r.m_rare_cal << " " << r.m_rare_eval << " " << std::fixed << std::setprecision(5)
           << r.lam_in << " " << r.lam_mid << " " << r.hi.cos_all << " " << r.rb.cos_all << " "
           << r.hi.snr_db << " " << r.rb.snr_db << " " << (r.rb.snr_db - r.hi.snr_db) << " "
           << std::scientific << r.hi.mse_rare << " " << r.rb.mse_rare << " " << (r.passed ? "EMITTED" : "SKIPPED") << "\n";
    }
    const double tot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("[RAREBAL_REPACK] DONE in %.2fs -> %s (main=%d [p150_l1=%d], mtp=%d)\n",
                tot_s, st_out.c_str(), emitted_main, emitted_pinned, emitted_mtp);
    return 0;
}
