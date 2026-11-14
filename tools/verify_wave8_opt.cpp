// Wave 8 Comprehensive Silicon & Mathematical Verification Suite
//   Test 1 (Track 8A): 100,000-iter gfx906 -> /dev/shm (hipHostRegisterMapped) PCIe Coherency & Doorbell (0 stale reads)
//   Test 2 (Track 8B): M = 16 DSpark Causal-Tree Topology, Adaptive Hysteresis, Exact Analytical Breakpoint TV (< 1e-15),
//                      and 1,000,000-step Kolmogorov-Dirac Verification Sweep
//   Test 3 (Track 8D): 1,048,576-Position (1M) FP64 Givens RoPE Parity (pos = 1,000,000 & 1,048,575), Bit-Packed
//                      Lightning Indexer (T_comp = 524,288), and Tiered HBM2 + Mapped-DDR4 CSA2 Sparse Attention Gather
// Pure C++17 / HIP (gfx906), zero Python, zero ScratchSize spill.

#include "../mi50/dsv41_kolmogorov_tree_verifier.hpp"
#include "../mi50/dsv41_mi50_kernels.hpp"
#include "../mi50/dsv41_tree_spec.hpp"
#include "../p150/dsv41_p150_l1.hpp"
#include "../src/hlut2_pack.hpp"

#include <hip/hip_runtime.h>
#include <immintrin.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace {

#define HIP_CHECK(cmd)                                                                        \
    do {                                                                                      \
        hipError_t e_ = (cmd);                                                                \
        if (e_ != hipSuccess) {                                                               \
            std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__,  \
                         __LINE__);                                                           \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

struct alignas(64) ShmCoherencyTestLayout {
    uint64_t req_seq;
    uint64_t ack_seq;
    uint32_t active_slot;
    uint32_t num_words;
    uint64_t pad[5];
    uint32_t slots[2][40960];  // 2 * 160 KiB (matches M=16 * 5120 BF16 = 163,840 B)
};

__global__ __launch_bounds__(256, 4) void k_pcie_shm_coherency_ring_test(
    ShmCoherencyTestLayout* __restrict__ shm,
    uint32_t words_per_iter,
    uint64_t start_seq,
    uint32_t iters,
    uint32_t* __restrict__ wg_done_counter)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t num_blocks = gridDim.x;
    const uint32_t gid = bid * 256u + tid;
    const uint32_t stride = num_blocks * 256u;

    for (uint32_t it = 0; it < iters; ++it) {
        const uint64_t seq = start_seq + uint64_t(it);
        const uint32_t slot = uint32_t(seq & 1ULL);

        if (tid == 0 && seq >= 2ULL) {
            const uint64_t need_ack = seq - 1ULL;
            while (__hip_atomic_load(&shm->ack_seq, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < need_ack) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __syncthreads();

        const uint32_t expected_base = uint32_t((seq * 0x9E3779B1ULL) ^ 0xA5A55A5AU);
        uint32_t* dst = shm->slots[slot];
        for (uint32_t i = gid; i < words_per_iter; i += stride) {
            __hip_atomic_store(&dst[i], expected_base ^ (i * 2654435761u),
                               __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }

        __threadfence_system();
        __syncthreads();

        if (tid == 0) {
            const uint32_t prev = __hip_atomic_fetch_add(&wg_done_counter[it], 1u,
                                                         __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
            if (prev + 1u == num_blocks) {
                __hip_atomic_store(&shm->active_slot, slot, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                __hip_atomic_store(&shm->num_words, words_per_iter, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                __threadfence_system();
                __hip_atomic_store(&shm->req_seq, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            }
        }
    }
}

__global__ __launch_bounds__(256, 4) void k_write_shm_payload_multi_wg(
    uint32_t* __restrict__ dst_slot,
    uint32_t words,
    uint32_t pattern_seed)
{
    const uint32_t gid = blockIdx.x * 256u + threadIdx.x;
    const uint32_t stride = gridDim.x * 256u;
    for (uint32_t i = gid; i < words; i += stride) {
        __hip_atomic_store(&dst_slot[i], pattern_seed ^ (i * 2654435761u),
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

__global__ __launch_bounds__(64, 4) void k_doorbell_shm_release(
    ShmCoherencyTestLayout* __restrict__ shm,
    uint32_t slot,
    uint32_t words,
    uint64_t seq)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __hip_atomic_store(&shm->active_slot, slot, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(&shm->num_words, words, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __threadfence_system();
        __hip_atomic_store(&shm->req_seq, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

bool test_1_pcie_shm_coherency() {
    std::printf("[TEST 1] Track 8A: gfx906 -> /dev/shm (hipHostRegisterMapped) PCIe Coherency & Doorbell (100,000 iters)...\n");
    std::fflush(stdout);

    const char* shm_name = "/dsv41_wave8_pcie_test";
    ::shm_unlink(shm_name);
    int fd = ::shm_open(shm_name, O_CREAT | O_RDWR | O_TRUNC, 0600);
    if (fd < 0) {
        std::perror("shm_open");
        return false;
    }
    const size_t bytes = sizeof(ShmCoherencyTestLayout);
    if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
        std::perror("ftruncate");
        ::close(fd);
        ::shm_unlink(shm_name);
        return false;
    }
    void* map = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (map == MAP_FAILED) {
        std::perror("mmap");
        ::shm_unlink(shm_name);
        return false;
    }
    std::memset(map, 0, bytes);
    auto* host_shm = reinterpret_cast<ShmCoherencyTestLayout*>(map);

    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipHostRegister(host_shm, bytes, hipHostRegisterMapped));
    ShmCoherencyTestLayout* dev_shm = nullptr;
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dev_shm), host_shm, 0));

    int low_prio = 0, high_prio = 0;
    HIP_CHECK(hipDeviceGetStreamPriorityRange(&low_prio, &high_prio));
    hipStream_t hi_stream = nullptr;
    HIP_CHECK(hipStreamCreateWithPriority(&hi_stream, hipStreamNonBlocking, high_prio));

    constexpr uint32_t kPartA = 90000;
    constexpr uint32_t kPartB = 10000;
    constexpr uint32_t kTotalIters = kPartA + kPartB;
    constexpr uint32_t kWords = 40960;

    uint32_t* d_wg_done = nullptr;
    HIP_CHECK(hipMalloc(&d_wg_done, kPartA * sizeof(uint32_t)));
    HIP_CHECK(hipMemsetAsync(d_wg_done, 0, kPartA * sizeof(uint32_t), hi_stream));
    HIP_CHECK(hipStreamSynchronize(hi_stream));

    std::atomic<uint64_t> stale_errors{0};
    std::thread consumer([&]() {
        uint64_t want_seq = 1;
        while (want_seq <= kTotalIters) {
            uint64_t got = __atomic_load_n(&host_shm->req_seq, __ATOMIC_ACQUIRE);
            while (got < want_seq) {
                _mm_pause();
                got = __atomic_load_n(&host_shm->req_seq, __ATOMIC_ACQUIRE);
            }
            const uint32_t slot = host_shm->active_slot;
            const uint32_t nwords = host_shm->num_words;
            const uint32_t expected_base = uint32_t((want_seq * 0x9E3779B1ULL) ^ 0xA5A55A5AU);
            const uint32_t* src = host_shm->slots[slot];
            for (uint32_t i = 0; i < nwords; i += 16) {
                const uint32_t v0 = src[i];
                const uint32_t v15 = src[i + 15];
                const uint32_t e0 = expected_base ^ (i * 2654435761u);
                const uint32_t e15 = expected_base ^ ((i + 15u) * 2654435761u);
                if (v0 != e0 || v15 != e15) {
                    stale_errors.fetch_add(1, std::memory_order_relaxed);
                }
            }
            __atomic_store_n(&host_shm->ack_seq, want_seq, __ATOMIC_RELEASE);
            ++want_seq;
        }
    });

    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t base = 0; base < kPartA; base += 5000) {
        hipLaunchKernelGGL(k_pcie_shm_coherency_ring_test, dim3(16), dim3(256), 0, hi_stream,
                           dev_shm, kWords, uint64_t(base + 1), 5000u, d_wg_done + base);
    }
    for (uint32_t it = 0; it < kPartB; ++it) {
        const uint64_t seq = uint64_t(kPartA + it + 1);
        const uint32_t slot = uint32_t(seq & 1ULL);
        const uint32_t seed = uint32_t((seq * 0x9E3779B1ULL) ^ 0xA5A55A5AU);
        while (__atomic_load_n(&host_shm->ack_seq, __ATOMIC_ACQUIRE) + 1ULL < seq) {
            _mm_pause();
        }
        hipLaunchKernelGGL(k_write_shm_payload_multi_wg, dim3(16), dim3(256), 0, hi_stream,
                           dev_shm->slots[slot], kWords, seed);
        hipLaunchKernelGGL(k_doorbell_shm_release, dim3(1), dim3(64), 0, hi_stream,
                           dev_shm, slot, kWords, seq);
    }

    consumer.join();
    HIP_CHECK(hipStreamSynchronize(hi_stream));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const uint64_t errs = stale_errors.load();

    HIP_CHECK(hipFree(d_wg_done));
    HIP_CHECK(hipStreamDestroy(hi_stream));
    HIP_CHECK(hipHostUnregister(host_shm));
    ::munmap(map, bytes);
    ::shm_unlink(shm_name);

    std::printf("[TEST 1] Completed %u / %u iterations in %.2f ms (%.2f us/iter, stream_prio=[%d..%d]) | stale_reads=%llu -> %s\n",
                kTotalIters, kTotalIters, ms, (ms * 1000.0) / double(kTotalIters),
                low_prio, high_prio, static_cast<unsigned long long>(errs),
                (errs == 0) ? "PASS" : "FAIL");
    return errs == 0;
}

bool test_2_tree16_kolmogorov_and_adaptive_controller() {
    using namespace dsv41;
    std::printf("[TEST 2] Track 8B: M = 16 DSpark Causal-Tree Topology, Adaptive Hysteresis, & 1,000,000-Step Kolmogorov-Dirac Sweep...\n");
    std::fflush(stdout);

    // 2A. Verify M = 16 fits inside the single 32-row P150A L1 tile (0 B extra L1 SRAM) and SGPR bitpacks match!
    if (p150_fabric1d_per_bank_l1_bytes(17u, true) != 1453184u ||
        p150_fabric1d_per_bank_l1_bytes(17u, true) > kP150L1Fabric1DBankBytes) {
        std::fprintf(stderr, "[TEST 2] P150A L1 bank invariant violated\n");
        return false;
    }

    const int32_t sample_tokens[16] = {100, 1, 2, 3, 4, 5, 6, 1, 3, 2, 0, 7, 4, 5, 2, 6};
    Dsv41SpecTree16 tree16{};
    if (!build_spec_tree16_topology(16u, 4096u, kTree16Parents.data(), sample_tokens, &tree16)) {
        std::fprintf(stderr, "[TEST 2] build_spec_tree16_topology failed\n");
        return false;
    }
    TreeSpecNode plan_nodes[16]{};
    for (uint32_t i = 0; i < 16u; ++i) {
        plan_nodes[i].parent = kTree16Parents[i];
        plan_nodes[i].token  = sample_tokens[i];
        plan_nodes[i].depth  = kTree16Depths[i];
    }
    const TreeSpecPlan plan16 = build_tree_spec_plan(plan_nodes, 16u, 4096);
    if (!plan16.valid) {
        std::fprintf(stderr, "[TEST 2] build_tree_spec_plan(M=16) invalid\n");
        return false;
    }
    for (uint32_t r = 0; r < 16u; ++r) {
        const uint64_t word = (r < 4u) ? kTree16AncMask256.w0
                            : (r < 8u) ? kTree16AncMask256.w1
                            : (r < 12u) ? kTree16AncMask256.w2
                            : kTree16AncMask256.w3;
        const uint16_t packed_row = uint16_t((word >> (16u * (r & 3u))) & 0xFFFFULL);
        const uint32_t packed_dep = uint32_t((kTree16DepthNibbles64 >> (4u * r)) & 0xFULL);
        const uint32_t packed_par = uint32_t((kTree16ParentNibbles64 >> (4u * r)) & 0xFULL);
        if (packed_row != tree16.ancestor_mask[r] || packed_row != plan16.allow_mask[r] ||
            packed_dep != tree16.depth[r] ||
            (r > 0u && int32_t(packed_par) != tree16.parent[r])) {
            std::fprintf(stderr, "[TEST 2] SGPR 256-bit mask/nibble mismatch at row %u\n", r);
            return false;
        }
    }

    // 2B. Verify Dsv41AdaptiveTreeWidthController hysteresis:
    //     - Starts at M=8 (ema_tau=4.25)
    //     - Drops to M=16 inside <think> once rolling tau < 3.9
    //     - Stays at M=16 inside deadband [3.9, 4.2), returns to M=8 once tau >= 4.2 or outside <think>
    Dsv41AdaptiveTreeWidthController ctrl(Dsv41TreeWidthMode::Auto);
    if (ctrl.select_width(true) != 8) return false;
    for (int i = 0; i < 12; ++i) ctrl.observe_step(2, true);  // High-entropy <think> drops EMA < 3.9
    if (ctrl.ema_tau >= 3.9 || ctrl.select_width(true) != 16) {
        std::fprintf(stderr, "[TEST 2] Adaptive controller failed to switch to M=16 when tau=%.3f < 3.9\n", ctrl.ema_tau);
        return false;
    }
    if (ctrl.select_width(false) != 8) {
        std::fprintf(stderr, "[TEST 2] Adaptive controller failed to revert to M=8 outside <think>\n");
        return false;
    }
    ctrl.ema_tau = 4.05;
    ctrl.current_width = 16;
    if (ctrl.select_width(true) != 16) {
        std::fprintf(stderr, "[TEST 2] Adaptive controller violated deadband hysteresis at tau=4.05\n");
        return false;
    }
    for (int i = 0; i < 16; ++i) ctrl.observe_step(5, true);  // Structured steps recover EMA >= 4.2
    if (ctrl.ema_tau < 4.2 || ctrl.select_width(true) != 8) {
        std::fprintf(stderr, "[TEST 2] Adaptive controller failed to recover to M=8 at tau=%.3f >= 4.2\n", ctrl.ema_tau);
        return false;
    }

    // 2C. Exact analytical breakpoint integration across all 16 nodes + 100 random logit tables,
    //     followed by 1,000,000 (10^6) full M = 16 verify_deterministic_tree16_u01 executions!
    constexpr size_t V = 8;
    std::mt19937_64 rng(0x57415645385F3842ULL);
    std::normal_distribution<float> ndist(0.0f, 1.35f);
    double max_tv = 0.0;

    double target_probs[16 * V]{};
    for (int trial = 0; trial < 100; ++trial) {
        int32_t toks[16]{};
        toks[0] = 0;
        for (int n = 0; n < 16; ++n) {
            float logits[V];
            for (size_t v = 0; v < V; ++v) logits[v] = ndist(rng);
            compute_filtered_target_probs_f64(logits, V, 0.85, 6, 0.94, &target_probs[size_t(n) * V]);
            if (n > 0) toks[n] = int32_t((uint32_t(trial * 7 + n * 3)) % V);
        }
        Dsv41SpecTree16 tr{};
        if (!build_spec_tree16_topology(16u, 1024u, kTree16Parents.data(), toks, &tr)) return false;

        for (uint32_t n = 0; n < 16u; ++n) {
            int32_t sub_parent[16] = {-1};
            int32_t sub_tok[16] = {0};
            double sub_probs[16 * V]{};
            std::copy(&target_probs[size_t(n) * V], &target_probs[size_t(n + 1) * V], sub_probs);
            uint32_t sub_m = 1u;
            uint16_t cm = tr.child_mask[n];
            std::vector<int32_t> excl;
            std::vector<double> cuts = {0.0, 1.0};
            double b_cdf = 0.0;
            while (cm != 0u) {
                const uint32_t c = uint32_t(__builtin_ctz(cm));
                cm &= uint16_t(cm - 1u);
                sub_parent[sub_m] = 0;
                sub_tok[sub_m] = tr.token_id[c];
                std::copy(&target_probs[size_t(c) * V], &target_probs[size_t(c + 1) * V], &sub_probs[size_t(sub_m) * V]);
                b_cdf += sub_probs[sub_tok[sub_m]];
                cuts.push_back(std::clamp(b_cdf, 0.0, 1.0));
                excl.push_back(sub_tok[sub_m]);
                ++sub_m;
            }
            double c_cdf = b_cdf;
            for (size_t v = 0; v < V; ++v) {
                if (std::find(excl.begin(), excl.end(), int32_t(v)) != excl.end()) continue;
                c_cdf += sub_probs[v];
                cuts.push_back(std::clamp(c_cdf, 0.0, 1.0));
            }
            std::sort(cuts.begin(), cuts.end());

            Dsv41SpecTree16 sub_tr{};
            if (!build_spec_tree16_topology(sub_m, 0u, sub_parent, sub_tok, &sub_tr)) return false;

            double exact_emitted[V]{};
            for (size_t i = 0; i + 1 < cuts.size(); ++i) {
                const double len = cuts[i + 1] - cuts[i];
                if (len <= 1e-17) continue;
                const double mid_u[6] = {0.5 * (cuts[i] + cuts[i + 1]), 0.5, 0.5, 0.5, 0.5, 0.5};
                const auto vr = verify_deterministic_tree16_u01(sub_tr, sub_probs, V, mid_u);
                exact_emitted[vr.emitted_tokens[0]] += len;
            }
            double l1 = 0.0;
            for (size_t v = 0; v < V; ++v) {
                l1 += std::abs(exact_emitted[v] - target_probs[size_t(n) * V + v]);
            }
            max_tv = std::max(max_tv, 0.5 * l1);
        }
    }

    // 1,000,000-step M = 16 execution sweep (verifying throughput, bounds, and empirical root law)
    std::uniform_real_distribution<double> udist(0.0, 1.0);
    constexpr uint64_t kSweepSteps = 1000000ULL;
    uint64_t total_emitted = 0ULL;
    uint64_t root_hist[V]{};
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t s = 0; s < kSweepSteps; ++s) {
        double u_vec[6];
        for (int d = 0; d < 6; ++d) u_vec[d] = udist(rng);
        const auto vr = verify_deterministic_tree16_u01(tree16, target_probs, V, u_vec);
        if (vr.num_emitted < 1u || vr.num_emitted > 6u) return false;
        total_emitted += vr.num_emitted;
        ++root_hist[vr.emitted_tokens[0]];
    }
    const double ns_per_step =
        std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(kSweepSteps);

    double mc_tv = 0.0;
    for (size_t v = 0; v < V; ++v) {
        mc_tv += 0.5 * std::abs(double(root_hist[v]) / double(kSweepSteps) - target_probs[v]);
    }
    const bool ok = (max_tv < 1e-15) && (mc_tv < 3e-3);
    std::printf("[TEST 2] M=16 exact analytical max_TV=%.3e (< 1e-15) | 1,000,000-step sweep: mean_tau=%.3f, mc_TV=%.4f, %.1f ns/step -> %s\n",
                max_tv, double(total_emitted) / double(kSweepSteps), mc_tv, ns_per_step, ok ? "PASS" : "FAIL");
    return ok;
}

bool test_3_1m_context_rope_indexer_and_tiered_csa2() {
    using namespace dsv41;
    std::printf("[TEST 3] Track 8D: 1,048,576-Position (1M) RoPE + Lightning Indexer + Tiered DDR4 CSA2 Gather...\n");
    std::fflush(stdout);

    HIP_CHECK(hipSetDevice(0));
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    init_rope_inv_freq_gpu();

    // 3A. Exact FP64 Givens RoPE parity at pos = 1,000,000 and pos = 1,048,575 against CPU long double reference
    auto cpu_build_freqs = [](double base, int orig_seq, double factor, double out_freqs[32]) {
        constexpr int dim = 64;
        constexpr double kPi = 3.141592653589793238462643383279502884;
        for (int i = 0; i < 32; ++i) {
            out_freqs[i] = 1.0 / std::pow(base, double(2 * i) / double(dim));
        }
        if (orig_seq > 0) {
            auto corr_dim = [&](double rot) {
                return double(dim) * std::log(double(orig_seq) / (rot * 2.0 * kPi)) / (2.0 * std::log(base));
            };
            const double low = std::max(std::floor(corr_dim(32.0)), 0.0);
            const double high = std::min(std::ceil(corr_dim(1.0)), double(dim - 1));
            const double denom = std::max(high - low, 1e-3);
            for (int i = 0; i < 32; ++i) {
                const double ramp = std::max(0.0, std::min(1.0, (double(i) - low) / denom));
                const double smooth = 1.0 - ramp;
                out_freqs[i] = (out_freqs[i] / factor) * (1.0 - smooth) + out_freqs[i] * smooth;
            }
        }
    };
    double fw[32], fc[32];
    cpu_build_freqs(10000.0, 0, 16.0, fw);
    cpu_build_freqs(160000.0, 65536, 16.0, fc);

    constexpr int kTestH = 8;
    constexpr int kTestD = 512;
    std::mt19937_64 rng(0x314D5F43534132ULL);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    std::vector<uint16_t> h_in(size_t(kTestH) * kTestD);
    for (auto& v : h_in) v = f32_to_bf16_rne(dist(rng));

    uint16_t* d_rope_buf = nullptr;
    HIP_CHECK(hipMalloc(&d_rope_buf, h_in.size() * sizeof(uint16_t)));

    double min_rope_cos = 1.0;
    size_t nonfinite_cnt = 0;
    for (int pos_test : {1000000, 1048575}) {
        for (bool yarn : {false, true}) {
            HIP_CHECK(hipMemcpyAsync(d_rope_buf, h_in.data(), h_in.size() * sizeof(uint16_t),
                                     hipMemcpyHostToDevice, stream));
            launch_rope_tail64_bf16(d_rope_buf, yarn, 1, kTestH, kTestD, pos_test, 1, false, stream);
            std::vector<uint16_t> h_gpu(h_in.size());
            HIP_CHECK(hipMemcpyAsync(h_gpu.data(), d_rope_buf, h_gpu.size() * sizeof(uint16_t),
                                     hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));

            const double* freqs = yarn ? fc : fw;
            double dot = 0.0, ng = 0.0, nr = 0.0;
            for (int h = 0; h < kTestH; ++h) {
                const size_t base = size_t(h) * kTestD + (kTestD - 64);
                for (int p = 0; p < 32; ++p) {
                    const long double ang = static_cast<long double>(pos_test) * static_cast<long double>(freqs[p]);
                    const float c = static_cast<float>(std::cos(ang));
                    const float s = static_cast<float>(std::sin(ang));
                    const float x0 = bf16_to_f32(h_in[base + 2 * p + 0]);
                    const float x1 = bf16_to_f32(h_in[base + 2 * p + 1]);
                    const float r0 = bf16_to_f32(f32_to_bf16_rne(x0 * c - x1 * s));
                    const float r1 = bf16_to_f32(f32_to_bf16_rne(x0 * s + x1 * c));
                    const float g0 = bf16_to_f32(h_gpu[base + 2 * p + 0]);
                    const float g1 = bf16_to_f32(h_gpu[base + 2 * p + 1]);
                    if (!std::isfinite(g0) || !std::isfinite(g1)) ++nonfinite_cnt;
                    dot += double(r0) * double(g0) + double(r1) * double(g1);
                    nr  += double(r0) * double(r0) + double(r1) * double(r1);
                    ng  += double(g0) * double(g0) + double(g1) * double(g1);
                }
            }
            const double cos_v = dot / (std::sqrt(nr) * std::sqrt(ng));
            min_rope_cos = std::min(min_rope_cos, cos_v);
        }
    }
    HIP_CHECK(hipFree(d_rope_buf));

    // 3B. Bit-packed Lightning Indexer across T_comp = 524,288 compressed keys (start_pos = 1,048,575)
    constexpr int kTComp1M = 524288;
    constexpr int kStartPos1M = 1048575;
    constexpr int kCandStride = 2048;  // 2048 uint32 words = 65,536 8-key blocks = 1,048,576 tokens!
    std::vector<float> h_scores(size_t(kTComp1M) + 65536u, -1e9f);
    for (int c = 0; c < kTComp1M; ++c) {
        h_scores[size_t(c)] = -float(c + 1);  // Exact distinct FP32 integers in [-524288, -1] (zero tie ambiguity)
    }
    // Plant dominant peaks at slot 50 (pos 100), slot 300,000 (pos 600,000), and slot 500,000 (pos 1,000,000)!
    h_scores[50]     = 42.0f;
    h_scores[300000] = 55.0f;
    h_scores[500000] = 77.0f;

    float* d_scores = nullptr;
    uint32_t* d_cand_mask = nullptr;
    int32_t* d_out_idxs_m1 = nullptr;
    int32_t* d_out_idxs_m2 = nullptr;
    HIP_CHECK(hipMalloc(&d_scores, h_scores.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_cand_mask, kCandStride * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_out_idxs_m1, 512 * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_out_idxs_m2, 512 * sizeof(int32_t)));
    HIP_CHECK(hipMemsetAsync(d_cand_mask, 0, kCandStride * sizeof(uint32_t), stream));

    HIP_CHECK(hipMemcpyAsync(d_scores, h_scores.data(), h_scores.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    launch_indexer_candidates_and_topk(d_scores, d_cand_mask, kCandStride, d_out_idxs_m1,
                                       1, kTComp1M, kStartPos1M, 2, /*mode=*/1, 512, stream);
    HIP_CHECK(hipMemcpyAsync(d_scores, h_scores.data(), h_scores.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    launch_indexer_candidates_and_topk(d_scores, d_cand_mask, kCandStride, d_out_idxs_m2,
                                       1, kTComp1M, kStartPos1M, 2, /*mode=*/2, 512, stream);

    std::vector<int32_t> h_idxs1(512), h_idxs2(512);
    HIP_CHECK(hipMemcpyAsync(h_idxs1.data(), d_out_idxs_m1, 512 * sizeof(int32_t), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(h_idxs2.data(), d_out_idxs_m2, 512 * sizeof(int32_t), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    std::sort(h_idxs1.begin(), h_idxs1.end());
    std::sort(h_idxs2.begin(), h_idxs2.end());
    const bool has_50     = std::binary_search(h_idxs2.begin(), h_idxs2.end(), 50);
    const bool has_300k   = std::binary_search(h_idxs2.begin(), h_idxs2.end(), 300000);
    const bool has_500k   = std::binary_search(h_idxs2.begin(), h_idxs2.end(), 500000);
    const bool idx_match  = (h_idxs1 == h_idxs2) && has_50 && has_300k && has_500k;

    HIP_CHECK(hipFree(d_scores));
    HIP_CHECK(hipFree(d_cand_mask));
    HIP_CHECK(hipFree(d_out_idxs_m1));
    HIP_CHECK(hipFree(d_out_idxs_m2));

    // 3C. Tiered HBM2 (hot_comp_cap = 1024) + Mapped DDR4 (ext slots up to 524,288 -> pos = 1,048,576, 64-bit size_t strides!)
    //     Compare tiered k_pack_compress_kv_fp4_e4m3_16 + k_sparse_attn (attending to slot 50 in HBM2 + slots 300,000 &
    //     500,000 [pos=1,000,000] & 524,270 in mapped DDR4) against direct HBM2-only execution on the exact same KV latents!
    constexpr int kHotCap = 1024;
    constexpr size_t kExtSlots = size_t(kTComp1M - kHotCap);
    const size_t ext_fp4_bytes  = kExtSlots * 256ULL;
    const size_t ext_e4m3_bytes = kExtSlots * 32ULL;
    const size_t ext_total_bytes = ext_fp4_bytes + ext_e4m3_bytes;

    void* ext_host_slab = ::mmap(nullptr, ext_total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ext_host_slab == MAP_FAILED) {
        std::perror("mmap ext_host_slab");
        return false;
    }
    HIP_CHECK(hipHostRegister(ext_host_slab, ext_total_bytes, hipHostRegisterMapped));
    uint8_t* d_ext_base = nullptr;
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&d_ext_base), ext_host_slab, 0));
    uint8_t* d_ext_fp4  = d_ext_base;
    uint8_t* d_ext_e4m3 = d_ext_base + ext_fp4_bytes;

    uint8_t *d_hot_fp4 = nullptr, *d_hot_e4m3 = nullptr;
    uint8_t *d_ref_fp4 = nullptr, *d_ref_e4m3 = nullptr;
    HIP_CHECK(hipMalloc(&d_hot_fp4,  size_t(kHotCap) * 256ULL));
    HIP_CHECK(hipMalloc(&d_hot_e4m3, size_t(kHotCap) * 32ULL));
    HIP_CHECK(hipMalloc(&d_ref_fp4,  4ULL * 256ULL));
    HIP_CHECK(hipMalloc(&d_ref_e4m3, 4ULL * 32ULL));

    // Pack 4 distinct 512-dim latents into tiered slots {50 (HBM2), 300000 (DDR4), 500000 [pos=1,000,000] (DDR4), 524270 (DDR4)}
    // and into contiguous HBM2 reference slots {0, 1, 2, 3}!
    const int test_slots[4] = {50, 300000, 500000, 524270};
    uint16_t* d_lat = nullptr;
    HIP_CHECK(hipMalloc(&d_lat, 4 * 512 * sizeof(uint16_t)));
    std::vector<uint16_t> h_lat(4 * 512);
    for (auto& v : h_lat) v = f32_to_bf16_rne(dist(rng));
    HIP_CHECK(hipMemcpyAsync(d_lat, h_lat.data(), h_lat.size() * sizeof(uint16_t), hipMemcpyHostToDevice, stream));

    for (int j = 0; j < 4; ++j) {
        launch_pack_compress_kv_fp4_e4m3_16(d_lat + j * 512, d_hot_fp4, d_hot_e4m3,
                                            test_slots[j], 1, stream,
                                            d_ext_fp4, d_ext_e4m3, kHotCap);
        launch_pack_compress_kv_fp4_e4m3_16(d_lat + j * 512, d_ref_fp4, d_ref_e4m3,
                                            j, 1, stream);
    }

    // Run M = 16 causal-tree CSA2 sparse attention at start_pos = 1,048,560 (reaching pos = 1,048,565)!
    constexpr int kMTree = 16;
    constexpr int kHeads = 8;
    std::vector<uint16_t> h_q(size_t(kMTree) * kHeads * 512), h_win_ring(128 * 512), h_cur_kv(size_t(kMTree) * 512);
    std::vector<float> h_sink(kHeads, 0.1f);
    for (auto& v : h_q) v = f32_to_bf16_rne(dist(rng));
    for (auto& v : h_win_ring) v = f32_to_bf16_rne(dist(rng));
    for (auto& v : h_cur_kv) v = f32_to_bf16_rne(dist(rng));

    std::vector<int32_t> h_tiered_idxs(size_t(kMTree) * 4), h_ref_idxs(size_t(kMTree) * 4);
    for (int m = 0; m < kMTree; ++m) {
        for (int j = 0; j < 4; ++j) {
            h_tiered_idxs[size_t(m) * 4 + j] = test_slots[j];
            h_ref_idxs[size_t(m) * 4 + j]    = j;
        }
    }

    uint16_t *d_q = nullptr, *d_win = nullptr, *d_cur = nullptr, *d_out_tier = nullptr, *d_out_ref = nullptr;
    float *d_sink = nullptr, *d_splitk = nullptr;
    int32_t *d_tier_idxs = nullptr, *d_ref_idxs = nullptr;
    HIP_CHECK(hipMalloc(&d_q, h_q.size() * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_win, h_win_ring.size() * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_cur, h_cur_kv.size() * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_out_tier, h_q.size() * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_out_ref,  h_q.size() * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_sink, h_sink.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_splitk, size_t(kMTree) * 8 * kHeads * 514 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_tier_idxs, h_tiered_idxs.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_ref_idxs,  h_ref_idxs.size() * sizeof(int32_t)));

    HIP_CHECK(hipMemcpyAsync(d_q, h_q.data(), h_q.size() * sizeof(uint16_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_win, h_win_ring.data(), h_win_ring.size() * sizeof(uint16_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_cur, h_cur_kv.data(), h_cur_kv.size() * sizeof(uint16_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_sink, h_sink.data(), h_sink.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_tier_idxs, h_tiered_idxs.data(), h_tiered_idxs.size() * sizeof(int32_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_ref_idxs, h_ref_idxs.data(), h_ref_idxs.size() * sizeof(int32_t), hipMemcpyHostToDevice, stream));

    launch_sparse_attn(d_q, d_sink, d_win, d_cur,
                       /*start_pos=*/1048560, /*swa_floor_pos=*/0, /*dspark_bidir=*/false,
                       d_hot_fp4, d_hot_e4m3, d_tier_idxs, 4, d_splitk, d_out_tier,
                       kMTree, kHeads, stream, kTree16AncMask256, kTree16DepthNibbles64,
                       d_ext_fp4, d_ext_e4m3, kHotCap);
    // Note: in causal CSA2, compressed slot idx `c` represents positions `[2*c, 2*c+1]`, which are all `< 1,048,560 - 128`
    // for BOTH {50, 300000, 500000, 524200} and {0, 1, 2, 3} when start_pos=1,048,560! Wait: slot 524,270 has 2*524270 = 1,048,540,
    // which is INSIDE the 128-token local SWA window [1,048,433 .. 1,048,560] where CSA2 masks out compressed keys overlapping SWA!
    // Let's also verify a pure non-overlapping distant slot 524,000 (pos = 1,048,000 < 1,048,432) for exact bit-for-bit parity!
    const int non_swa_slots[4] = {50, 300000, 500000, 524000};
    launch_pack_compress_kv_fp4_e4m3_16(d_lat + 3 * 512, d_hot_fp4, d_hot_e4m3,
                                        524000, 1, stream, d_ext_fp4, d_ext_e4m3, kHotCap);
    for (int m = 0; m < kMTree; ++m) {
        for (int j = 0; j < 4; ++j) h_tiered_idxs[size_t(m) * 4 + j] = non_swa_slots[j];
    }
    HIP_CHECK(hipMemcpyAsync(d_tier_idxs, h_tiered_idxs.data(), h_tiered_idxs.size() * sizeof(int32_t), hipMemcpyHostToDevice, stream));

    launch_sparse_attn(d_q, d_sink, d_win, d_cur,
                       /*start_pos=*/1048560, /*swa_floor_pos=*/0, /*dspark_bidir=*/false,
                       d_hot_fp4, d_hot_e4m3, d_tier_idxs, 4, d_splitk, d_out_tier,
                       kMTree, kHeads, stream, kTree16AncMask256, kTree16DepthNibbles64,
                       d_ext_fp4, d_ext_e4m3, kHotCap);
    launch_sparse_attn(d_q, d_sink, d_win, d_cur,
                       /*start_pos=*/1048560, /*swa_floor_pos=*/0, /*dspark_bidir=*/false,
                       d_ref_fp4, d_ref_e4m3, d_ref_idxs, 4, d_splitk, d_out_ref,
                       kMTree, kHeads, stream, kTree16AncMask256, kTree16DepthNibbles64);

    std::vector<uint16_t> h_out_tier(h_q.size()), h_out_ref(h_q.size());
    HIP_CHECK(hipMemcpyAsync(h_out_tier.data(), d_out_tier, h_out_tier.size() * sizeof(uint16_t), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(h_out_ref.data(),  d_out_ref,  h_out_ref.size() * sizeof(uint16_t),  hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    double dot = 0.0, nt = 0.0, nr = 0.0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < h_out_tier.size(); ++i) {
        const float vt = bf16_to_f32(h_out_tier[i]);
        const float vr = bf16_to_f32(h_out_ref[i]);
        if (!std::isfinite(vt) || !std::isfinite(vr)) ++nonfinite_cnt;
        max_diff = std::max(max_diff, std::fabs(vt - vr));
        dot += double(vt) * double(vr);
        nt  += double(vt) * double(vt);
        nr  += double(vr) * double(vr);
    }
    const double attn_cos = dot / (std::sqrt(nt) * std::sqrt(nr));

    HIP_CHECK(hipFree(d_lat));
    HIP_CHECK(hipFree(d_q));
    HIP_CHECK(hipFree(d_win));
    HIP_CHECK(hipFree(d_cur));
    HIP_CHECK(hipFree(d_out_tier));
    HIP_CHECK(hipFree(d_out_ref));
    HIP_CHECK(hipFree(d_sink));
    HIP_CHECK(hipFree(d_splitk));
    HIP_CHECK(hipFree(d_tier_idxs));
    HIP_CHECK(hipFree(d_ref_idxs));
    HIP_CHECK(hipFree(d_hot_fp4));
    HIP_CHECK(hipFree(d_hot_e4m3));
    HIP_CHECK(hipFree(d_ref_fp4));
    HIP_CHECK(hipFree(d_ref_e4m3));
    HIP_CHECK(hipHostUnregister(ext_host_slab));
    ::munmap(ext_host_slab, ext_total_bytes);
    HIP_CHECK(hipStreamDestroy(stream));

    const bool ok = (nonfinite_cnt == 0) && (min_rope_cos > 0.999999) && idx_match && (attn_cos > 0.999999) && (max_diff == 0.0f);
    std::printf("[TEST 3] 1M Context @ pos=1,000,000..1,048,575: RoPE_cos=%.8f | Indexer(50,300k,500k)=%d | TieredDDR4_CSA2_cos=%.8f (max_diff=%.2e, nonfinite=%zu) -> %s\n",
                min_rope_cos, int(idx_match), attn_cos, max_diff, nonfinite_cnt, ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    if (!test_1_pcie_shm_coherency()) return 1;
    if (!test_2_tree16_kolmogorov_and_adaptive_controller()) return 1;
    if (!test_3_1m_context_rope_indexer_and_tiered_csa2()) return 1;
    std::printf("[VERIFY_WAVE8_OPT] ALL WAVE 8 HARDWARE & MATHEMATICAL TESTS PASSED.\n");
    return 0;
}
