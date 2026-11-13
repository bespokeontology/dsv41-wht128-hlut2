// Pure C++20 2x Blackhole P150A Lock-Free /dev/shm Worker Daemon for DeepSeek-V4.1-Flash-1017
// Zero Python anywhere. Serves M=1 decode, M=6 DSpark verify, and M=64..512 prefill over /dsv41_p150_ipc.

#include "p150_dsv41_engine.hpp"
#include "dsv41_p150_l1.hpp"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

namespace {
volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }
}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string root_env = (std::getenv("DSV41_ROOT") != nullptr)
                                         ? std::string(std::getenv("DSV41_ROOT"))
                                         : std::string(".");
        std::string packed_root = (std::getenv("DSV41_P150_ROOT") != nullptr)
                                      ? std::string(std::getenv("DSV41_P150_ROOT"))
                                      : (root_env + "/packed_p150");
        std::string kernels_dir = root_env + "/p150/kernels";
        std::string shm_name    = "/dsv41_p150_ipc";
        std::string rarebal_overlay;
        bool selfcheck_only     = false;
        uint32_t selfcheck_layer = 0u;
        bool enable_trace       = true;
        bool reuse_shm          = false;
        uint32_t max_layers     = 40u;
        uint32_t hot_shards     = dsv41::kP150ConfiguredHotShards;

        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--link-check") {
                dsv41::p150_l1_geometry_report();
                if (!dsv41::p150_hot_shards_fit(hot_shards)) {
                    std::fprintf(stderr, "[P150_DSV41] Hot-shard geometry link check FAILED for %u\n",
                                 unsigned(hot_shards));
                    return 2;
                }
                std::cout << "[P150_DSV41] Link check OK (no devices opened)\n";
                return 0;
            } else if (arg == "--model-dir" && i + 1 < argc) {
                const std::string md = argv[++i];
                packed_root = md + "/packed_p150";
                kernels_dir = md + "/p150/kernels";
            } else if (arg == "--selfcheck-layer0") {
                selfcheck_only = true;
                selfcheck_layer = 0u;
            } else if (arg == "--selfcheck-layer" && i + 1 < argc) {
                selfcheck_only = true;
                selfcheck_layer = static_cast<uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--packed-root" && i + 1 < argc) {
                packed_root = argv[++i];
            } else if (arg == "--kernels-dir" && i + 1 < argc) {
                kernels_dir = argv[++i];
            } else if (arg == "--shm" && i + 1 < argc) {
                shm_name = argv[++i];
            } else if (arg == "--max-layers" && i + 1 < argc) {
                max_layers = static_cast<uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--no-trace") {
                enable_trace = false;
            } else if (arg == "--reuse-shm") {
                reuse_shm = true;
            } else if (arg == "--hot-shards" && i + 1 < argc) {
                hot_shards = std::clamp<uint32_t>(static_cast<uint32_t>(std::stoul(argv[++i])),
                                                  dsv41::kP150MinHotShards,
                                                  dsv41::kP150MaxHotL1Shards);
            } else if (arg == "--rarebal-overlay" && i + 1 < argc) {
                rarebal_overlay = argv[++i];
            } else if (arg == "--no-rarebal-overlay") {
                rarebal_overlay.clear();
            }
        }

        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);

        dsv41::P150Dsv41Engine engine(packed_root, kernels_dir, enable_trace, rarebal_overlay, hot_shards);

        if (selfcheck_only) {
            const bool ok = engine.selfcheck_layer(selfcheck_layer);
            return ok ? 0 : 2;
        }

        dsv41::Dsv41P150ShmMapping shm;
        if (!(reuse_shm ? shm.reuse_or_create_server(shm_name) : shm.create_server(shm_name))) {
            throw std::runtime_error("Failed to create /dev/shm channel: " + shm_name);
        }

        auto publish_mask = [&]() {
            const uint64_t m = engine.loaded_layers_mask();
            shm.ptr->loaded_layers_mask_lo = static_cast<uint32_t>(m & 0xFFFFFFFFULL);
            shm.ptr->loaded_layers_mask_hi = static_cast<uint32_t>(m >> 32);
        };

        const uint32_t initial_cnt = engine.load_available_layers(max_layers);
        if (initial_cnt > 0u && !engine.selfcheck_layer(0u)) {
            throw std::runtime_error("Startup P150 Wave 6 selfcheck_layer(0) failed parity_cos >= 0.99999 gate!");
        }
        publish_mask();
        std::atomic_thread_fence(std::memory_order_release);
        shm.ptr->ready = 1u;

        std::printf("[P150_DSV41] Serving %s (pid=%d, initial_layers=%u/%u, L1/card=%.2f MiB, reuse_shm=%d)\n",
                    shm_name.c_str(), int(::getpid()), initial_cnt, max_layers,
                    double(engine.persistent_l1_bytes_per_card()) / (1024.0 * 1024.0), int(reuse_shm));
        std::fflush(stdout);

        uint32_t last_seq = shm.ptr->resp_seq.load(std::memory_order_acquire);
        uint64_t idle_spins = 0ULL;
        auto last_scan = std::chrono::steady_clock::now();

        while (!g_stop) {
            const uint32_t seq = shm.ptr->req_seq.load(std::memory_order_acquire);
            if (seq == last_seq) {
                dsv41::dsv41_cpu_relax();
                if ((++idle_spins & 0x3FFFFFFULL) == 0ULL) {
                    const auto now = std::chrono::steady_clock::now();
                    if (engine.loaded_layers_mask() != ((max_layers >= 64u) ? ~0ULL : ((1ULL << max_layers) - 1ULL)) &&
                        std::chrono::duration_cast<std::chrono::seconds>(now - last_scan).count() >= 5) {
                        engine.load_available_layers(max_layers);
                        publish_mask();
                        last_scan = now;
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
                continue;
            }
            idle_spins = 0ULL;
            last_seq = seq;

            const uint32_t op = shm.ptr->op;
            if (op == dsv41::DSV41_P150_OP_SHUTDOWN) {
                shm.ptr->status = 0;
                shm.ptr->resp_seq.store(seq, std::memory_order_release);
                break;
            } else if (op == dsv41::DSV41_P150_OP_PING) {
                publish_mask();
                shm.ptr->status = 0;
                shm.ptr->last_device_ns = 0;
                shm.ptr->resp_seq.store(seq, std::memory_order_release);
                continue;
            } else if (op == dsv41::DSV41_P150_OP_MOE_FORWARD) {
                try {
                    const uint64_t ns = engine.forward_moe(
                        shm.ptr->layer,
                        shm.ptr->rows,
                        shm.ptr->input_is_wht_rotated != 0u,
                        shm.ptr->row_slots,
                        shm.ptr->row_weights,
                        shm.ptr->x_in_bf16,
                        shm.ptr->y_out_bf16);
                    publish_mask();
                    shm.ptr->last_device_ns = ns;
                    shm.ptr->status = 0;
                } catch (const std::exception& ex) {
                    std::fprintf(stderr, "[P150_DSV41] forward_moe error: %s\n", ex.what());
                    shm.ptr->status = -1;
                }
                shm.ptr->resp_seq.store(seq, std::memory_order_release);
            } else {
                shm.ptr->status = -2;
                shm.ptr->resp_seq.store(seq, std::memory_order_release);
            }
        }

        std::printf("[P150_DSV41] Worker exiting cleanly.\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[P150_DSV41] Fatal error: %s\n", e.what());
        return 1;
    }
}
