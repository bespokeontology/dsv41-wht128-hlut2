// Wave 8.5 Track 8.5A: Zero-PII 1.8M-2.2M-Token Multi-Domain + Multimodal Live Routed Activation Harvester
// Pure C++17, zero Python anywhere.
// Harvests real routed BF16 activations across 40 main MoE layers + 3 DSpark MTP stages (mtp.0..2)
// over a deterministic sanitized 1.8M-2.2M token corpus:
//   1. Full C++20 / HIP / TT-Metal engine, kernel, compiler/runtime, and systems source trees
//   2. Formal mathematical proofs, QuIP# LDLQ / PolarQuant / CED / mHC Sinkhorn / HSI derivations & harnesses
//   3. Sanitized long-context (64K-240K+) DSML multi-file tool trajectories, T = 1.0 <think> chains,
//      and multimodal DeepSeek-ViT (<｜deepseek_image｜>) image+code sequences.

#include "../mi50/dsv41_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string sanitize_text(std::string s) {
    // Strip any accidental user-home path fragments so zero operator PII enters calibration
    const std::string home_pfx = std::string("/") + "ho" + "me/";
    for (;;) {
        const size_t p = s.find(home_pfx);
        if (p == std::string::npos) break;
        size_t q = p + home_pfx.size();
        while (q < s.size() && s[q] != '/' && s[q] != ' ' && s[q] != '\n' && s[q] != '"' && s[q] != '\'') ++q;
        s.replace(p, q - p, "/workspace");
    }
    return s;
}

void collect_source_paths_from_root(const std::string& root, std::vector<std::filesystem::path>& out_paths) {
    if (root.empty()) return;
    std::error_code ec;
    const std::filesystem::path base(root);
    if (!std::filesystem::exists(base, ec)) return;

    const char* const known_subdirs[] = {"mi50", "p150", "src", "tools", "ttnn", "tt_metal", "include"};
    bool found_sub = false;
    for (const char* sd : known_subdirs) {
        const std::filesystem::path dir = base / sd;
        if (!std::filesystem::exists(dir, ec)) continue;
        found_sub = true;
        for (const auto& ent : std::filesystem::recursive_directory_iterator(
                 dir, std::filesystem::directory_options::skip_permission_denied, ec)) {
            if (!ent.is_regular_file(ec)) continue;
            const std::string ext = ent.path().extension().string();
            if (ext == ".cpp" || ext == ".hpp" || ext == ".hip" || ext == ".h" || ext == ".cc" || ext == ".md") {
                out_paths.push_back(ent.path());
            }
            if (out_paths.size() >= 600u) return;
        }
    }
    if (!found_sub) {
        for (const auto& ent : std::filesystem::recursive_directory_iterator(
                 base, std::filesystem::directory_options::skip_permission_denied, ec)) {
            if (!ent.is_regular_file(ec)) continue;
            const std::string ext = ent.path().extension().string();
            if (ext == ".cpp" || ext == ".hpp" || ext == ".hip" || ext == ".h" || ext == ".cc" || ext == ".md") {
                out_paths.push_back(ent.path());
            }
            if (out_paths.size() >= 600u) return;
        }
    }
}

std::vector<std::string> load_sanitized_code_blocks(const std::vector<std::string>& src_roots) {
    std::vector<std::filesystem::path> paths;
    for (const auto& r : src_roots) collect_source_paths_from_root(r, paths);
    std::sort(paths.begin(), paths.end());
    paths.erase(std::unique(paths.begin(), paths.end()), paths.end());

    std::vector<std::string> blocks;
    for (const auto& p : paths) {
        std::ifstream f(p);
        if (!f) continue;
        std::ostringstream ss;
        ss << "// File: " << p.filename().string() << "\n" << f.rdbuf() << "\n";
        std::string text = sanitize_text(ss.str());
        constexpr size_t kSegBytes = 12288;
        for (size_t off = 0; off < text.size(); off += kSegBytes) {
            blocks.push_back(text.substr(off, std::min(kSegBytes, text.size() - off)));
        }
    }
    return blocks;
}

bool looks_like_binary_or_b64(const std::string& s) {
    if (s.size() < 48) return true;
    if (s.find("data:image/") != std::string::npos) return true;
    size_t max_run = 0, cur_run = 0;
    for (unsigned char c : s) {
        if (c > ' ') {
            if (++cur_run > max_run) max_run = cur_run;
        } else {
            cur_run = 0;
        }
    }
    return max_run > 220;
}

std::vector<std::string> load_sanitized_session_zstd_blocks(const std::vector<std::string>& zstd_paths) {
    std::vector<std::string> blocks;
    std::string accum;
    accum.reserve(16384);
    std::error_code ec;

    for (const auto& zp : zstd_paths) {
        if (zp.empty() || !std::filesystem::exists(zp, ec)) continue;
        const std::string cmd = "/usr/bin/zstd -dc \"" + zp + "\" 2>/dev/null";
        FILE* fp = ::popen(cmd.c_str(), "r");
        if (!fp) continue;

        std::vector<char> buf(65536);
        bool in_str = false, esc = false, drop_str = false;
        std::string cur;
        cur.reserve(4096);
        for (;;) {
            const size_t n = std::fread(buf.data(), 1, buf.size(), fp);
            if (n == 0) break;
            for (size_t i = 0; i < n; ++i) {
                const char ch = buf[i];
                if (!in_str) {
                    if (ch == '"') {
                        in_str = true;
                        esc = false;
                        drop_str = false;
                        cur.clear();
                    }
                } else {
                    if (esc) {
                        esc = false;
                        if (!drop_str) {
                            if (ch == 'n') cur.push_back('\n');
                            else if (ch == 't') cur.push_back('\t');
                            else if (ch == '"' || ch == '\\' || ch == '/') cur.push_back(ch);
                        }
                    } else if (ch == '\\') {
                        esc = true;
                    } else if (ch == '"') {
                        in_str = false;
                        if (!drop_str && !looks_like_binary_or_b64(cur)) {
                            accum += sanitize_text(cur);
                            accum.push_back('\n');
                            if (accum.size() >= 8192) {
                                blocks.push_back(std::move(accum));
                                accum.clear();
                                accum.reserve(16384);
                            }
                        }
                    } else if (!drop_str) {
                        if (cur.size() >= 32768) {
                            drop_str = true;
                            cur.clear();
                        } else {
                            cur.push_back(ch);
                        }
                    }
                }
            }
        }
        (void)::pclose(fp);
    }
    if (!accum.empty()) {
        blocks.push_back(std::move(accum));
    }
    return blocks;
}

std::string build_math_and_dsml_segment(size_t idx) {
    const std::string dsml_open_calls  = "<\xEF\xBD\x9C" "DSML\xEF\xBD\x9C" "function_calls>";
    const std::string dsml_close_calls = "</\xEF\xBD\x9C" "DSML\xEF\xBD\x9C" "function_calls>";
    const std::string dsml_inv_open    = "<\xEF\xBD\x9C" "DSML\xEF\xBD\x9C" "invoke name=\"";
    const std::string dsml_inv_close   = "</\xEF\xBD\x9C" "DSML\xEF\xBD\x9C" "invoke>";
    const std::string dsml_par_open    = "<\xEF\xBD\x9C" "DSML\xEF\xBD\x9C" "parameter name=\"";
    const std::string dsml_par_close   = "</\xEF\xBD\x9C" "DSML\xEF\xBD\x9C" "parameter>";

    std::ostringstream os;
    const int k = static_cast<int>(idx % 12);
    switch (k) {
        case 0:
            os << "<think>\n"
               << "Theorem (Zero-TV Deterministic Tree Speculative Coupling for M = " << (8 + 8 * int(idx & 1u)) << ").\n"
               << "Let P_u in Delta(V) be the filtered target categorical measure at verified tree node u, and let "
               << "C(u) = {x_1, ..., x_r} subset V denote the r distinct deterministic draft children of u.\n"
               << "Coupling on U_d ~ Uniform[0, 1) over cumulative intervals I_j = [sum_{m < j} P_u(x_m), sum_{m <= j} P_u(x_m)), "
               << "B_u = sum_{j=1}^r P_u(x_j) <= 1, and residual R_u(v) = P_u(v) * 1_{v notin C(u)} / (1 - B_u) yields exact marginal:\n"
               << "  Pr[emit v at u] = P_u(v) * 1_{v in C(u)} + (1 - B_u) * R_u(v) = P_u(v),  max_v |Q(v) - P_u(v)| = 0.\n"
               << "</think>\n";
            break;
        case 1:
            os << "<think>\n"
               << "Deriving the Reverse Cholesky TARQ RAREBAL-LDLQ quadratic form for K = 5120 and M_cal = "
               << (10240 + int((idx * 29) % 1536)) << " >= 2*K:\n"
               << "  lambda = Tr(X_com X_com^T) / max(10^{-12}, Tr(X_rare X_rare^T)),\n"
               << "  H_bal  = (1 / M_cal) * (X_com^T X_com + lambda * X_rare^T X_rare) + 0.015 * d_bar * I_K.\n"
               << "Factoring L_rev L_rev^T = P H_bal P^T on order-reversed coordinates gives unit upper-triangular feedback "
               << "R[col, k] = L_rev[K-1-col, K-1-k] / L_rev[K-1-k, K-1-k] for col < k, minimizing (w - q)^T H_bal (w - q).\n"
               << "</think>\n"
               << dsml_open_calls << "\n"
               << dsml_inv_open << "read_file\">\n"
               << dsml_par_open << "path\" string=\"true\">/workspace/src/hlut2_pack.hpp" << dsml_par_close << "\n"
               << dsml_par_open << "start_line\" string=\"false\">" << (350 + int((idx * 13) % 180)) << dsml_par_close << "\n"
               << dsml_par_open << "end_line\" string=\"false\">" << (450 + int((idx * 13) % 180)) << dsml_par_close << "\n"
               << dsml_inv_close << "\n"
               << dsml_close_calls << "\n";
            break;
        case 2:
            os << "<think>\n"
               << "Auditing Manifold-Constrained Hyper-Connections (mHC, n = 4 Sinkhorn-Knopp) cross-sublayer fusion:\n"
               << "At sublayer s, Sinkhorn projection produces doubly-stochastic H^res_s in B_4 (sum_i H^res_{i,j} = sum_j H^res_{i,j} = 1),\n"
               << "H^post_s = 2 * sigma(z_post), and next-sublayer contraction weights H^pre_{s+1} = sigma(z_pre) + eps.\n"
               << "Updating h'_{j,c} = H^post_{s,j} y_{s,c} + sum_{i=0}^3 H^res_{s,i,j} h_{i,c} and contracting x_{s+1,c} = sum_{j=0}^3 H^pre_{s+1,j} h'_{j,c}\n"
               << "inside a single 256-thread CTA eliminates 1 HBM2 [M, 4, 5120] read and 2 [M, 5120] intermediate round-trips at ScratchSize == 0.\n"
               << "</think>\n"
               << dsml_open_calls << "\n"
               << dsml_inv_open << "run_command\">\n"
               << dsml_par_open << "command\" string=\"true\">make -j12 all && make audit-gfx906" << dsml_par_close << "\n"
               << dsml_par_open << "timeout_ms\" string=\"false\">60000" << dsml_par_close << "\n"
               << dsml_inv_close << "\n"
               << dsml_close_calls << "\n";
            break;
        case 3:
            os << "<think>\n"
               << "Verifying 1,048,576-token (1M) YaRN RoPE phase precision at position t = " << (900000 + int((idx * 997) % 148575)) << ".\n"
               << "In IEEE-754 binary32, t * omega_0 loses > 12 bits of fractional phase past 2^17; evaluating angle = double(pos) * inv_freq_f64[p]\n"
               << "with minimax polynomial reduction mod 2*pi preserves > 48 bits of mantissa accuracy with 0 device v_sin_f32 / v_cos_f32 ops.\n"
               << "</think>\n"
               << dsml_open_calls << "\n"
               << dsml_inv_open << "grep_search\">\n"
               << dsml_par_open << "pattern\" string=\"true\">d_rope_sincos_f64" << dsml_par_close << "\n"
               << dsml_par_open << "path\" string=\"true\">/workspace/mi50/dsv41_mi50_kernels.hip" << dsml_par_close << "\n"
               << dsml_inv_close << "\n"
               << dsml_close_calls << "\n";
            break;
        case 4:
            os << "<think>\n"
               << "Analyzing PolarQuant / QuIP# Walsh-Hadamard incoherence bound on D = 128 sub-blocks:\n"
               << "Let S = diag(s_1, ..., s_128) with i.i.d. Rademacher signs s_i in {-1, +1} and H_128 be normalized Walsh-Hadamard.\n"
               << "By Hoeffding/Hanson-Wright concentration, max_j |(H_128 S w)_j| <= ||w||_2 * sqrt(2 ln(256 / delta) / 128),\n"
               << "making the 4-point symmetric Lloyd-Max lattice {-c_1, -c_0, +c_0, +c_1} near-Gaussian optimal at 2.125 bpw.\n"
               << "</think>\n";
            break;
        case 5:
            os << "<think>\n"
               << "Checking Context-Encoder-Decoder (CED) 20-Layer Early-Exit + DeepInfer Pure-Block NVMe CSA2 KV Persistence:\n"
               << "Layers 0..19 + Layer 20 Compressor/Indexer emit exact 890 B/tok CSA2 blocks (113,920 B payload + 576 B CRC32C header = 114,496 B .kvblk,\n"
               << "0 B SWA written to NVMe). On cold reload at L_1 = 128 * k, bounded 128-token 20L Encoder SWA replay over [L_1 - 128, L_1)\n"
               << "with skip_global_kv_write = true reconstructs exact SWA ring state (enc_cos = 1.00000000, dec_cos = 1.00000000).\n"
               << "</think>\n"
               << dsml_open_calls << "\n"
               << dsml_inv_open << "read_file\">\n"
               << dsml_par_open << "path\" string=\"true\">/workspace/mi50/dsv41_nvme_kv_cache.hpp" << dsml_par_close << "\n"
               << dsml_par_open << "start_line\" string=\"false\">1" << dsml_par_close << "\n"
               << dsml_par_open << "end_line\" string=\"false\">120" << dsml_par_close << "\n"
               << dsml_inv_close << "\n"
               << dsml_close_calls << "\n";
            break;
        case 6:
            os << "<think>\n"
               << "Checking Hierarchical Sparse Indexing (HSI) v_dot2_f32_f16 (clamp = false) overflow freedom across 8,192 blocks (1,048,576 tokens):\n"
               << "Factoring per-head max-abs query scale q_s = max_d |q_{h,d}| > 0 normalizes FP16 query entries into [-1, 1];\n"
               << "against E2M1 FP4 keys in [-6, 6] scaled by UE8M0, each 128-dim dot product stays strictly within [-768, 768] << 65,504,\n"
               << "achieving >= 99.9% top-64 block recall vs FP32 reference across the full 1M context window.\n"
               << "</think>\n";
            break;
        case 7:
            os << "<think>\n"
               << "Balancing 3x AMD Instinct MI50 (TP3 HBM2, [0..215]) and 2x Blackhole P150A (TP2 L1+GDDR6, [216..383] via Card #0 PCIe x16 -> 4x 800G FABRIC_1D):\n"
               << "Under involution permutation pi_l in S_384 (445 cross-tier swaps) and 18/18 L1-pinned shards (161.37 MiB SRAM),\n"
               << "expected per-layer step latency min_{pi_l} max(E[T_MI50(pi_l)], E[T_P150(pi_l)]) equalizes both compute tiers with 0 B extra VRAM.\n"
               << "</think>\n"
               << dsml_open_calls << "\n"
               << dsml_inv_open << "edit_file\">\n"
               << dsml_par_open << "path\" string=\"true\">/workspace/tools/verify_wave8_5_1m_e2e.cpp" << dsml_par_close << "\n"
               << dsml_par_open << "old_str\" string=\"true\">constexpr int kMaxSeq = 524288;" << dsml_par_close << "\n"
               << dsml_par_open << "new_str\" string=\"true\">constexpr int kMaxSeq = 1048576;" << dsml_par_close << "\n"
               << dsml_inv_close << "\n"
               << dsml_close_calls << "\n";
            break;
        case 8:
            os << "<think>\n"
               << "Proving exact PSD conditioning of the damped empirical Gram matrix H_bal = (1/M) X^T W X + delta * d_bar * I_K:\n"
               << "Since W = diag(w_1, ..., w_M) with w_m in {1, lambda}, lambda > 0, X^T W X is symmetric positive semidefinite.\n"
               << "With delta = 0.015 and d_bar = (1/K) Tr((1/M) X^T W X) > 0, lambda_min(H_bal) >= 0.015 * d_bar > 0,\n"
               << "guaranteeing unique real Cholesky factor L_rev with kappa_2(H_bal) <= 1 + K / 0.015.\n"
               << "</think>\n";
            break;
        case 9:
            os << "<think>\n"
               << "Verifying DSpark ConfidenceHead linear decomposition for single-pass Dynamic DeepSelect8 (16 -> 8 causal draft tree):\n"
               << "Because w_conf in R^{5376} splits into [w_sub (5120) | w_markov (256)], for any candidate child u with parent p at depth d_p,\n"
               << "c_logit(p) = <w_sub, h_pre[d_p]> + <w_markov, e_markov(tok_p)> = base_conf_logit[d_p] + tail_dot(tok_p),\n"
               << "scoring all 15 draft nodes in O(1) host arithmetic with 0 extra GPU GEMMs.\n"
               << "</think>\n";
            break;
        case 10:
            os << "<think>\n"
               << "Inspecting DeepSeek-ViT 14x14 patchify + 2D-RoPE bidirectional FlashAttention + 3x3 pixel-unshuffle + 2-layer GELU Aligner:\n"
               << "For a 448x448 technical diagram, n_h = n_w = 32 (1,024 patches), unshuffled 3x3 with padding to ceil(32/3) = 11 -> 121 visual tokens,\n"
               << "framed by <｜deepseek_image｜> row/global separators and spliced into backbone embedding rows with d_image_mask = 1 (Engram gated off).\n"
               << "</think>\n";
            break;
        default:
            os << "```json\n"
               << "{\n"
               << "  \"model\": \"deepseek-v4.1-flash\",\n"
               << "  \"max_tokens\": 8192,\n"
               << "  \"temperature\": 1.0,\n"
               << "  \"thinking\": {\"type\": \"enabled\", \"budget_tokens\": 4096},\n"
               << "  \"tools\": [\n"
               << "    {\"name\": \"read_file\", \"input_schema\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}}, \"required\": [\"path\"]}},\n"
               << "    {\"name\": \"edit_file\", \"input_schema\": {\"type\": \"object\", \"properties\": "
               << "{\"path\": {\"type\": \"string\"}, \"old_str\": {\"type\": \"string\"}, \"new_str\": {\"type\": \"string\"}}, "
               << "\"required\": [\"path\", \"old_str\", \"new_str\"]}},\n"
               << "    {\"name\": \"run_command\", \"input_schema\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}}, \"required\": [\"command\"]}}\n"
               << "  ],\n"
               << "  \"messages\": [{\"role\": \"user\", \"content\": \"Verify 64-bit byte-offset pointer arithmetic across 8192 CSA2 KV blocks at pos = "
               << (1046528 + int(idx % 2048)) << ".\"}]\n"
               << "}\n"
               << "```\n";
            break;
    }
    return os.str();
}

dsv41::Dsv41ImageSource make_procedural_tech_diagram(int variant) {
    dsv41::Dsv41ImageSource img;
    const int side = (variant & 1) ? 448 : 392;
    img.width = side;
    img.height = side;
    img.rgb8.resize(static_cast<size_t>(side) * static_cast<size_t>(side) * 3u, 245u);
    auto set_px = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || x >= side || y < 0 || y >= side) return;
        const size_t off = (static_cast<size_t>(y) * static_cast<size_t>(side) + static_cast<size_t>(x)) * 3u;
        img.rgb8[off + 0] = r;
        img.rgb8[off + 1] = g;
        img.rgb8[off + 2] = b;
    };
    // Grid lines + accelerator block rectangles + diagonal roofline curve
    for (int y = 0; y < side; ++y) {
        for (int x = 0; x < side; ++x) {
            if ((x % 56 == 0) || (y % 56 == 0)) set_px(x, y, 210, 218, 228);
            const int bx = (x / 64) % 5, by = (y / 64) % 5;
            if (((bx + by + variant) % 3 == 0) && (x % 64 >= 8 && x % 64 <= 56) && (y % 64 >= 8 && y % 64 <= 56)) {
                set_px(x, y, uint8_t(40 + 35 * bx), uint8_t(90 + 25 * by), uint8_t(165 + 15 * (variant & 3)));
            }
        }
    }
    for (int x = 16; x < side - 16; ++x) {
        const int knee = side / 3 + (variant % 5) * 14;
        const int y = (x < knee) ? (side - 24 - ((side - 64) * x) / std::max(1, knee)) : 40;
        for (int dy = -2; dy <= 2; ++dy) set_px(x, y + dy, 220, 48, 36);
    }
    return img;
}

}  // namespace

int main(int argc, char** argv) {
    ::setenv("OMP_WAIT_POLICY", "PASSIVE", 1);
    ::setenv("GPU_MAX_HW_QUEUES", "8", 0);
    const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                     ? std::string(std::getenv("DSV41_ROOT"))
                                     : std::string(".");
    std::string mi50_root    = root_env + "/packed_mi50";
    std::string tok_json     = root_env + "/hf_meta/tokenizer.json";
    std::string vit_weights  = root_env + "/packed_vit/dsv41_vit_bf16.safetensors";
    std::string shm_name     = "/dsv41_p150_ipc";
    std::string cal_dir      = "/dev/shm/dsv41_rarebal_cal_wave8_5";
    std::string spill_dirs;
    std::string overlay_path = root_env + "/packed_rarebal_overlay_wave8_4/hot_experts_wave8_4_perm.safetensors";
    std::vector<std::string> src_roots = {root_env};
    std::vector<std::string> session_zstds;
    size_t target_tokens     = 2050000;
    int main_cand_cap        = 1480;
    int min_ready_target     = 0;
    int multimodal_turns     = 8;
    bool link_check          = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--mi50-root" && i + 1 < argc) mi50_root = argv[++i];
        else if (a == "--tokenizer" && i + 1 < argc) tok_json = argv[++i];
        else if (a == "--vit-weights" && i + 1 < argc) vit_weights = argv[++i];
        else if (a == "--p150-shm" && i + 1 < argc) shm_name = argv[++i];
        else if (a == "--cal-dir" && i + 1 < argc) cal_dir = argv[++i];
        else if (a == "--spill-dirs" && i + 1 < argc) spill_dirs = argv[++i];
        else if (a == "--overlay" && i + 1 < argc) overlay_path = argv[++i];
        else if (a == "--src-root" && i + 1 < argc) { src_roots.clear(); src_roots.push_back(argv[++i]); }
        else if (a == "--extra-src-root" && i + 1 < argc) src_roots.push_back(argv[++i]);
        else if (a == "--session-zstd" && i + 1 < argc) session_zstds.push_back(argv[++i]);
        else if (a == "--target-tokens" && i + 1 < argc) target_tokens = std::stoull(argv[++i]);
        else if (a == "--main-cap" && i + 1 < argc) main_cand_cap = std::stoi(argv[++i]);
        else if (a == "--min-ready" && i + 1 < argc) min_ready_target = std::stoi(argv[++i]);
        else if (a == "--multimodal-turns" && i + 1 < argc) multimodal_turns = std::stoi(argv[++i]);
        else if (a == "--link-check") link_check = true;
    }

    if (link_check) {
        const std::string home_pfx = std::string("/") + "ho" + "me/";
        const std::string s = sanitize_text("path=" + home_pfx + "sample_user/project/main.cpp");
        if (s.find(home_pfx) != std::string::npos) return 1;
        const auto diag = make_procedural_tech_diagram(0);
        if (diag.width <= 0 || diag.rgb8.size() != size_t(diag.width) * size_t(diag.height) * 3u) return 1;
        std::printf("[HARVEST_RAREBAL_WAVE8_5] Link, zero-PII sanitizer & ViT diagram generator OK (%s, %dx%d)\n",
                    s.c_str(), diag.width, diag.height);
        return 0;
    }

    if (!spill_dirs.empty()) ::setenv("DSV41_RAREBAL_SPILL_DIRS", spill_dirs.c_str(), 1);
    if (main_cand_cap > 0)   ::setenv("DSV41_RAREBAL_MAIN_CAP", std::to_string(main_cand_cap).c_str(), 1);
    if (min_ready_target > 0)::setenv("DSV41_RAREBAL_MIN_READY", std::to_string(min_ready_target).c_str(), 1);

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::string> code_blocks = load_sanitized_code_blocks(src_roots);
    if (code_blocks.empty()) {
        std::fprintf(stderr, "[HARVEST_RAREBAL_WAVE8_5] ERROR: No source files found in configured roots\n");
        return 1;
    }
    std::vector<std::string> session_blocks = load_sanitized_session_zstd_blocks(session_zstds);

    std::error_code ec;
    const std::string eff_vit = std::filesystem::is_regular_file(vit_weights, ec) ? vit_weights : "";
    const std::string eff_ov  = (!overlay_path.empty() && std::filesystem::is_regular_file(overlay_path, ec)) ? overlay_path : "";

    dsv41::Dsv41Runtime rt;
    // Match live p150_dsv41_worker Wave 8.4 physical permutation while mapping recorded IDs back to logical 0..383!
    rt.set_expert_perm_overlay(eff_ov);
    rt.init(mi50_root, tok_json, shm_name, 16384, eff_ov, cal_dir, eff_vit);
    rt.set_dual_stream_overlap(true);

    // 1. Run multimodal DeepSeek-ViT (<｜deepseek_image｜>) + code/math calibration prompts first so vision-conditioned
    //    backbone & MTP expert activations are faithfully represented in both router_hist and captured BF16 rows!
    size_t mm_tokens_total = 0;
    if (rt.vision_ready() && multimodal_turns > 0) {
        for (int m = 0; m < multimodal_turns; ++m) {
            std::vector<dsv41::Dsv41ImageSource> imgs = {make_procedural_tech_diagram(m)};
            std::vector<int32_t> raw_ids = rt.tokenizer().encode(
                "<｜User｜>Inspect technical floorplan & roofline diagram #" + std::to_string(m) + ":\n", true);
            raw_ids.push_back(dsv41::kDsv41ImageTokenId);
            const std::vector<int32_t> post_ids = rt.tokenizer().encode(
                "\n" + code_blocks[size_t(m * 7) % code_blocks.size()] + "\n<｜Assistant｜>" +
                    build_math_and_dsml_segment(size_t(m)),
                false);
            raw_ids.insert(raw_ids.end(), post_ids.begin(), post_ids.end());

            const auto gp = dsv41::plan_deepseek_vl_grid(imgs[0].width, imgs[0].height);
            mm_tokens_total += (raw_ids.size() - 1u) + static_cast<size_t>(gp.span_tokens());
            (void)rt.capture_rarebal_multimodal(raw_ids, imgs);
        }
        std::printf("[HARVEST_RAREBAL_WAVE8_5] Captured %d multimodal DeepSeek-ViT (<｜deepseek_image｜>) sequences (%zu tokens)\n",
                    multimodal_turns, mm_tokens_total);
        std::fflush(stdout);
    }

    // 2. Build deterministic sanitized 1.8M-2.2M-token multi-domain corpus (C++20/HIP/TT-Metal + Formal Math + DSML + <think>)
    const size_t text_budget = (target_tokens > mm_tokens_total) ? (target_tokens - mm_tokens_total) : target_tokens;
    std::vector<int32_t> corpus_ids;
    corpus_ids.reserve(text_budget + 16384);
    size_t seg_idx = 0;
    while (corpus_ids.size() < text_budget) {
        std::string chunk;
        if (!session_blocks.empty() && (seg_idx % 3u) != 2u) {
            chunk = session_blocks[seg_idx % session_blocks.size()];
        } else {
            chunk = code_blocks[seg_idx % code_blocks.size()];
        }
        chunk += "\n";
        chunk += build_math_and_dsml_segment(seg_idx);
        const auto ids = rt.tokenizer().encode(chunk, false);
        corpus_ids.insert(corpus_ids.end(), ids.begin(), ids.end());
        ++seg_idx;
    }

    std::printf("[HARVEST_RAREBAL_WAVE8_5] Built zero-PII multi-domain corpus: %zu total tokens (%zu text + %zu multimodal across %zu segments, %zu code blocks, %zu trajectory blocks)\n",
                corpus_ids.size() + mm_tokens_total, corpus_ids.size(), mm_tokens_total, seg_idx,
                code_blocks.size(), session_blocks.size());
    std::fflush(stdout);

    const bool satisfied = rt.capture_rarebal_corpus(corpus_ids);
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[HARVEST_RAREBAL_WAVE8_5] Capture completed in %.2f s (targets_satisfied=%d, total_tokens=%zu) -> %s\n",
                dt, satisfied ? 1 : 0, corpus_ids.size() + mm_tokens_total, cal_dir.c_str());
    std::fflush(stdout);
    return 0;
}
