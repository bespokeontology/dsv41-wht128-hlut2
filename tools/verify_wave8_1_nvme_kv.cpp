// Wave 8.2 Delta 8.2B Verification Suite: DeepInfer Pure-Block NVMe CSA2 KV + 20L RAM EncoderStateBlock
// Verifies:
//   1. 33,024-token (258 x 128-tok .kvblk = 28.03 MiB, 0 B SWA on NVMe, legacy .kvtail auto-purge)
//      cold restart QD=16 POSIX pread + 3x MI50 HBM2 hydration (< 50 ms, max_diff == 0, cos == 1.00000000)
//   2. Lossless 20-layer in-RAM EncoderStateBlock (20 x 134 x 528 B = 1,415,040 B ~= 1.35 MiB,
//      128 ring rows + 6 DSpark slack rows, max_bit_diff == 0, cos == 1.00000000)
//   3. Single-bit FNV-1a64 corruption rejection (.kvblk) + model_overlay_uuid / extra_salt isolation
//   4. Bounded mtime LRU directory eviction cap
// 100% Pure C++17 / HIP — Zero Python Anywhere.

#include "dsv41_nvme_kv_cache.hpp"
#include "dsv41_mi50_kernels.hpp"

#include <hip/hip_runtime.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define HIP_ASSERT(call)                                                                      \
    do {                                                                                      \
        hipError_t err__ = (call);                                                            \
        if (err__ != hipSuccess) {                                                            \
            std::fprintf(stderr, "[HIP_ERR] %s:%d: %s\n", __FILE__, __LINE__,                 \
                         hipGetErrorString(err__));                                           \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

void rm_rf_dir(const std::string& dir) {
    DIR* d = ::opendir(dir.c_str());
    if (d) {
        while (struct dirent* ent = ::readdir(d)) {
            const std::string n = ent->d_name;
            if (n == "." || n == "..") continue;
            (void)::unlink((dir + "/" + n).c_str());
        }
        (void)::closedir(d);
    }
    (void)::rmdir(dir.c_str());
}

double bf16_slab_cosine(const uint16_t* a, const uint16_t* b, size_t n, uint32_t& out_max_bit_diff) {
    out_max_bit_diff = 0u;
    long double dot = 0.0L, na = 0.0L, nb = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t diff = (a[i] > b[i]) ? uint32_t(a[i] - b[i]) : uint32_t(b[i] - a[i]);
        if (diff > out_max_bit_diff) out_max_bit_diff = diff;
        uint32_t ua = uint32_t(a[i]) << 16;
        uint32_t ub = uint32_t(b[i]) << 16;
        float fa = 0.0f, fb = 0.0f;
        std::memcpy(&fa, &ua, 4u);
        std::memcpy(&fb, &ub, 4u);
        if (std::isfinite(fa) && std::isfinite(fb)) {
            dot += static_cast<long double>(fa) * static_cast<long double>(fb);
            na  += static_cast<long double>(fa) * static_cast<long double>(fa);
            nb  += static_cast<long double>(fb) * static_cast<long double>(fb);
        }
    }
    if (na <= 0.0L || nb <= 0.0L) return 0.0;
    return static_cast<double>(dot / std::sqrt(na * nb));
}

double byte_slab_cosine(const uint8_t* a, const uint8_t* b, size_t n, uint32_t& out_max_diff) {
    out_max_diff = 0u;
    long double dot = 0.0L, na = 0.0L, nb = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t d = (a[i] > b[i]) ? uint32_t(a[i] - b[i]) : uint32_t(b[i] - a[i]);
        if (d > out_max_diff) out_max_diff = d;
        const long double va = static_cast<long double>(int(a[i]) - 128);
        const long double vb = static_cast<long double>(int(b[i]) - 128);
        dot += va * vb;
        na  += va * va;
        nb  += vb * vb;
    }
    if (na <= 0.0L || nb <= 0.0L) return 0.0;
    return static_cast<double>(dot / std::sqrt(na * nb));
}

}  // namespace

int main() {
    using namespace dsv41;
    const std::string test_dir = "/tmp/dsv41_nvme_kv_verify_" +
                                 std::to_string(::getpid());
    rm_rf_dir(test_dir);

    // -------------------------------------------------------------------------
    // Gate 1: Pure-Block 33,024-token (258 x 128-tok .kvblk, 0 B SWA on NVMe) -> 3x MI50 HBM2
    // -------------------------------------------------------------------------
    constexpr int kNumBlocks   = 258;                                       // 33,024 tokens in .kvblk shards
    constexpr int kTurn2Prompt = kNumBlocks * 128 + kDeepInferKeepTokens;   // 33,164 tokens (L_cap = 33,024!)

    std::mt19937_64 rng(0xD54182B1ULL);
    std::vector<int32_t> prompt_toks(static_cast<size_t>(kTurn2Prompt));
    for (size_t i = 0; i < prompt_toks.size(); ++i) {
        prompt_toks[i] = static_cast<int32_t>(100 + (rng() % 120000));
    }

    const RadixKey128 model_uuid{0x561E8F1A20261114ULL, 0xCA2BA100D5410082ULL};
    std::vector<RadixBlockEntry> ref_blocks(static_cast<size_t>(kNumBlocks));
    RadixKey128 cur_key = model_uuid;
    for (int b = 0; b < kNumBlocks; ++b) {
        auto& blk = ref_blocks[size_t(b)];
        const int32_t* bt = prompt_toks.data() + b * kNvmeRadixBlockTokens;
        cur_key = step_radix_hash_salted(cur_key, bt, 0ULL);
        blk.key = cur_key;
        blk.extra_salt = 0ULL;
        std::memcpy(blk.tokens.data(), bt, kNvmeRadixBlockTokens * sizeof(int32_t));
        uint64_t* w64 = reinterpret_cast<uint64_t*>(blk.data.data());
        for (size_t q = 0; q < kNvmeRadixBlockBytes / 8u; ++q) {
            w64[q] = rng();
        }
    }

    {
        Dsv41NvmeKvStore writer_store;
        writer_store.configure(test_dir, 32ULL << 30);
        for (int b = 0; b < kNumBlocks; ++b) {
            if (!writer_store.write_block_atomic(ref_blocks[size_t(b)], model_uuid)) {
                std::fprintf(stderr, "[FAIL] write_block_atomic failed at block %d\n", b);
                return 1;
            }
        }
        // Plant a stale legacy .kvtail file and confirm cold restart purges it to 0 SWA bytes on disk!
        const std::string stale_tail = test_dir + "/tail_00033024_deadbeef_cafebabe.kvtail";
        const int sfd = ::open(stale_tail.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0664);
        if (sfd >= 0) {
            const char dummy[64] = "DSV41TL1_LEGACY_SWA";
            const ssize_t wr = ::write(sfd, dummy, sizeof(dummy));
            (void)wr;
            (void)::close(sfd);
        }
        const int dfd = ::open(test_dir.c_str(), O_RDONLY);
        if (dfd >= 0) {
            (void)::syncfs(dfd);
            (void)::close(dfd);
        }
    }

    int dev_cnt = 0;
    HIP_ASSERT(hipGetDeviceCount(&dev_cnt));
    const int n_gpus = std::min(3, dev_cnt);
    if (n_gpus < 1) {
        std::fprintf(stderr, "[FAIL] Expected at least 1 HIP GPU\n");
        return 1;
    }

    const size_t total_radix_bytes = size_t(kNumBlocks) * kNvmeRadixBlockBytes;
    constexpr size_t kEncRingElems  = kEncoderStateLayers * kEncoderStateRingRows * kEncoderStateHeadDim;
    constexpr size_t kEncSlackElems = kEncoderStateLayers * kEncoderStateSlackRows * kEncoderStateHeadDim;
    const size_t enc_win_bytes      = kEncRingElems * sizeof(uint16_t);

    std::vector<uint8_t*>    d_radix_kv(static_cast<size_t>(n_gpus), nullptr);
    std::vector<uint16_t*>   d_enc_win_kv(static_cast<size_t>(n_gpus), nullptr);
    std::vector<uint8_t*>    h_pinned(static_cast<size_t>(n_gpus), nullptr);
    std::vector<hipStream_t> streams(static_cast<size_t>(n_gpus), nullptr);
    std::vector<hipStream_t> warm_streams(static_cast<size_t>(n_gpus), nullptr);
    std::vector<uint32_t*>   h_dpm_flag(static_cast<size_t>(n_gpus), nullptr);
    std::vector<uint32_t*>   d_dpm_flag(static_cast<size_t>(n_gpus), nullptr);
    std::vector<uint32_t*>   d_dpm_scratch(static_cast<size_t>(n_gpus), nullptr);
    const size_t pinned_slab_bytes = total_radix_bytes + enc_win_bytes;
    int low_prio = 0, high_prio = 0;
    HIP_ASSERT(hipDeviceGetStreamPriorityRange(&low_prio, &high_prio));
    for (int tp = 0; tp < n_gpus; ++tp) {
        HIP_ASSERT(hipSetDevice(tp));
        HIP_ASSERT(hipHostMalloc(reinterpret_cast<void**>(&h_pinned[size_t(tp)]),
                                 pinned_slab_bytes, hipHostMallocDefault));
        std::memset(h_pinned[size_t(tp)], 0, pinned_slab_bytes);
        HIP_ASSERT(hipStreamCreateWithFlags(&streams[size_t(tp)], hipStreamNonBlocking));
        HIP_ASSERT(hipStreamCreateWithPriority(&warm_streams[size_t(tp)], hipStreamNonBlocking, low_prio));
        HIP_ASSERT(hipHostMalloc(reinterpret_cast<void**>(&h_dpm_flag[size_t(tp)]), 64u,
                                 hipHostMallocMapped | hipHostMallocCoherent));
        HIP_ASSERT(hipHostGetDevicePointer(reinterpret_cast<void**>(&d_dpm_flag[size_t(tp)]),
                                           h_dpm_flag[size_t(tp)], 0));
        *h_dpm_flag[size_t(tp)] = 1u;
        HIP_ASSERT(hipMalloc(reinterpret_cast<void**>(&d_dpm_scratch[size_t(tp)]), 65536u));
        HIP_ASSERT(hipMalloc(reinterpret_cast<void**>(&d_radix_kv[size_t(tp)]), total_radix_bytes));
        HIP_ASSERT(hipMalloc(reinterpret_cast<void**>(&d_enc_win_kv[size_t(tp)]), enc_win_bytes));
        HIP_ASSERT(hipMemset(d_radix_kv[size_t(tp)], 0, total_radix_bytes));
        HIP_ASSERT(hipMemset(d_enc_win_kv[size_t(tp)], 0, enc_win_bytes));
        dsv41::launch_keep_dpm_warm(d_dpm_flag[size_t(tp)], d_dpm_scratch[size_t(tp)], 16383u, warm_streams[size_t(tp)]);
    }

    #pragma omp parallel num_threads(16)
    {
        asm volatile("" ::: "memory");
    }
    for (int tp = 0; tp < n_gpus; ++tp) {
        HIP_ASSERT(hipSetDevice(tp));
        HIP_ASSERT(hipMemcpyAsync(d_radix_kv[size_t(tp)], h_pinned[size_t(tp)], total_radix_bytes,
                                  hipMemcpyHostToDevice, streams[size_t(tp)]));
        HIP_ASSERT(hipMemsetAsync(d_radix_kv[size_t(tp)], 0, total_radix_bytes, streams[size_t(tp)]));
        HIP_ASSERT(hipStreamSynchronize(streams[size_t(tp)]));
    }

    // Cold-restart store instance -> verify 0 SWA files on disk & time pipelined 64-block QD=16 NVMe pread + 3x MI50 SDMA!
    Dsv41NvmeKvStore cold_store;
    cold_store.configure(test_dir, 32ULL << 30);
    if (cold_store.count_swa_files_on_disk() != 0u) {
        std::fprintf(stderr, "[FAIL] Expected 0 SWA/.kvtail files on NVMe after cold configure(), found %zu\n",
                     cold_store.count_swa_files_on_disk());
        return 1;
    }

    // Pre-warm thread_local TLS buffers on the 16-thread OpenMP pool (matches persistent runtime pool)
    #pragma omp parallel for schedule(static) num_threads(16)
    for (int t = 0; t < 16; ++t) {
        alignas(64) std::array<uint8_t, kNvmeRadixBlockBytes> warm_kv;
        (void)cold_store.read_block_pread_raw(
            ref_blocks[0].key, model_uuid, prompt_toks.data(), 0ULL, nullptr, warm_kv.data(), nullptr, false);
    }

    const int L_cap = std::max(0, kTurn2Prompt - kDeepInferKeepTokens);
    const int max_rblks = L_cap / kNvmeRadixBlockTokens;
    if (max_rblks != kNumBlocks) {
        std::fprintf(stderr, "[FAIL] KEEP_TOKENS=140 L_cap block mismatch (%d != %d)\n", max_rblks, kNumBlocks);
        return 1;
    }

    const auto t_hyd0 = std::chrono::steady_clock::now();
    std::vector<RadixKey128> lookup_keys(static_cast<size_t>(max_rblks));
    RadixKey128 lk_k = model_uuid;
    for (int b = 0; b < max_rblks; ++b) {
        lk_k = step_radix_hash_salted(lk_k, prompt_toks.data() + b * kNvmeRadixBlockTokens, 0ULL);
        lookup_keys[size_t(b)] = lk_k;
    }

    int ok_count = 0;
    constexpr int kBatchBlks = 64;
    for (int b0 = 0; b0 < max_rblks; b0 += kBatchBlks) {
        const int nb = std::min(kBatchBlks, max_rblks - b0);
        int batch_ok = 0;
        #pragma omp parallel for schedule(static) num_threads(16) reduction(+:batch_ok)
        for (int k = 0; k < nb; ++k) {
            const int b = b0 + k;
            alignas(64) std::array<uint8_t, kNvmeRadixBlockBytes> local_kv;
            if (cold_store.read_block_pread_raw(
                    lookup_keys[size_t(b)], model_uuid,
                    prompt_toks.data() + b * kNvmeRadixBlockTokens, 0ULL,
                    nullptr, local_kv.data(), nullptr, false))
            {
                for (int tp = 0; tp < n_gpus; ++tp) {
                    std::memcpy(h_pinned[size_t(tp)] + size_t(b) * kNvmeRadixBlockBytes,
                                local_kv.data(), kNvmeRadixBlockBytes);
                }
                batch_ok += 1;
            }
        }
        ok_count += batch_ok;
        const size_t byte_off = size_t(b0) * kNvmeRadixBlockBytes;
        const size_t byte_len = size_t(nb) * kNvmeRadixBlockBytes;
        for (int tp = 0; tp < n_gpus; ++tp) {
            HIP_ASSERT(hipSetDevice(tp));
            HIP_ASSERT(hipMemcpyAsync(d_radix_kv[size_t(tp)] + byte_off, h_pinned[size_t(tp)] + byte_off,
                                      byte_len, hipMemcpyHostToDevice, streams[size_t(tp)]));
        }
    }
    const auto t_io_done = std::chrono::steady_clock::now();
    if (ok_count != kNumBlocks) {
        std::fprintf(stderr, "[FAIL] Hydrated %d / %d blocks\n", ok_count, kNumBlocks);
        return 1;
    }

    for (int tp = 0; tp < n_gpus; ++tp) {
        HIP_ASSERT(hipSetDevice(tp));
        HIP_ASSERT(hipStreamSynchronize(streams[size_t(tp)]));
    }
    const auto t_dma_done = std::chrono::steady_clock::now();
    const double io_ms = std::chrono::duration<double, std::milli>(t_io_done - t_hyd0).count();
    const double dma_ms = std::chrono::duration<double, std::milli>(t_dma_done - t_io_done).count();
    const double hydrate_ms = std::chrono::duration<double, std::milli>(t_dma_done - t_hyd0).count();

    std::vector<uint8_t> gpu_radix_out(total_radix_bytes);
    HIP_ASSERT(hipSetDevice(n_gpus - 1));
    HIP_ASSERT(hipMemcpy(gpu_radix_out.data(), d_radix_kv[size_t(n_gpus - 1)], total_radix_bytes, hipMemcpyDeviceToHost));

    std::vector<uint8_t> ref_all_radix(total_radix_bytes);
    for (int b = 0; b < kNumBlocks; ++b) {
        std::memcpy(ref_all_radix.data() + size_t(b) * kNvmeRadixBlockBytes,
                    ref_blocks[size_t(b)].data.data(), kNvmeRadixBlockBytes);
    }

    uint32_t max_kv_diff = 0u;
    const double kv_cos = byte_slab_cosine(ref_all_radix.data(), gpu_radix_out.data(), total_radix_bytes, max_kv_diff);

    std::printf("[VERIFY_NVME_KV] Gate 1: %d toks (%d .kvblk = %.2f MiB, disk_swa_files=%zu) -> %dx MI50 HBM2 in %.2f ms (io=%.2fms, dma=%.2fms) | "
                "kv_max_diff=%u kv_cos=%.8f\n",
                kNumBlocks * kNvmeRadixBlockTokens, kNumBlocks,
                double(total_radix_bytes) / (1024.0 * 1024.0),
                cold_store.count_swa_files_on_disk(),
                n_gpus, hydrate_ms, io_ms, dma_ms,
                max_kv_diff, kv_cos);

    if (max_kv_diff != 0u || std::fabs(kv_cos - 1.0) > 1e-12 || hydrate_ms >= 50.0) {
        std::fprintf(stderr, "[FAIL] Gate 1 failed (hydrate_ms=%.2f, max_kv_diff=%u)\n", hydrate_ms, max_kv_diff);
        return 1;
    }

    // -------------------------------------------------------------------------
    // Gate 1B: Lossless 20-Layer RAM EncoderStateBlock (20 x 134 x 528 B = 1.35 MiB, 0 decoder SWA)
    // -------------------------------------------------------------------------
    {
        std::vector<uint16_t> ref_ring(kEncRingElems);
        std::vector<uint16_t> ref_slack(kEncSlackElems);
        for (size_t i = 0; i < kEncRingElems; ++i) {
            const float v = float(int(rng() & 0xFFFFu) - 32768) * (1.0f / 8192.0f);
            ref_ring[i] = detail_swa528::f32_to_bf16_rne(v);
        }
        for (size_t i = 0; i < kEncSlackElems; ++i) {
            const float v = float(int(rng() & 0xFFFFu) - 32768) * (1.0f / 8192.0f);
            ref_slack[i] = detail_swa528::f32_to_bf16_rne(v);
        }
        // Apply the exact k_split_qkv_norm_rope_quant UE8M0_32 + FP8_E4M3 roundtrip that populates d_win_kv[0..19]
        for (size_t r = 0; r < kEncoderStateLayers * kEncoderStateRingRows; ++r) {
            quantize_bf16_row_ue8m0_fp8_inplace(ref_ring.data() + r * kEncoderStateHeadDim);
        }
        for (size_t r = 0; r < kEncoderStateLayers * kEncoderStateSlackRows; ++r) {
            quantize_bf16_row_ue8m0_fp8_inplace(ref_slack.data() + r * kEncoderStateHeadDim);
        }

        EncoderStateBlock enc_blk{};
        enc_blk.pos = kNumBlocks * kNvmeRadixBlockTokens;
        enc_blk.key = lookup_keys.back();
        const auto t_pk0 = std::chrono::steady_clock::now();
        pack_encoder_state_block(ref_ring.data(), ref_slack.data(), kEncoderStateSlackRows, 6u, enc_blk);

        std::vector<uint16_t> out_ring(kEncRingElems, 0u);
        std::vector<uint16_t> out_slack(kEncSlackElems, 0u);
        unpack_encoder_state_block(enc_blk, out_ring.data(), out_slack.data(), kEncoderStateSlackRows);
        for (int tp = 0; tp < n_gpus; ++tp) {
            std::memcpy(h_pinned[size_t(tp)] + total_radix_bytes, out_ring.data(), enc_win_bytes);
        }
        for (int tp = 0; tp < n_gpus; ++tp) {
            HIP_ASSERT(hipSetDevice(tp));
            HIP_ASSERT(hipMemcpyAsync(d_enc_win_kv[size_t(tp)], h_pinned[size_t(tp)] + total_radix_bytes,
                                      enc_win_bytes, hipMemcpyHostToDevice, streams[size_t(tp)]));
        }
        for (int tp = 0; tp < n_gpus; ++tp) {
            HIP_ASSERT(hipSetDevice(tp));
            HIP_ASSERT(hipStreamSynchronize(streams[size_t(tp)]));
        }
        const double enc_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_pk0).count();

        std::vector<uint16_t> gpu_ring_out(kEncRingElems, 0u);
        HIP_ASSERT(hipSetDevice(n_gpus - 1));
        HIP_ASSERT(hipMemcpy(gpu_ring_out.data(), d_enc_win_kv[size_t(n_gpus - 1)], enc_win_bytes, hipMemcpyDeviceToHost));

        uint32_t max_ring_diff = 0u, max_slack_diff = 0u;
        const double ring_cos  = bf16_slab_cosine(ref_ring.data(), gpu_ring_out.data(), kEncRingElems, max_ring_diff);
        const double slack_cos = bf16_slab_cosine(ref_slack.data(), out_slack.data(), kEncSlackElems, max_slack_diff);
        std::printf("[VERIFY_NVME_KV] Gate 1B: 20L EncoderStateBlock (%zu B = %.2f MiB, 128+6 rows) pack+DMA in %.2f ms | "
                    "ring_max_diff=%u ring_cos=%.8f | slack_max_diff=%u slack_cos=%.8f\n",
                    kEncoderStateSlabBytes, double(kEncoderStateSlabBytes) / (1024.0 * 1024.0), enc_ms,
                    max_ring_diff, ring_cos, max_slack_diff, slack_cos);
        if (max_ring_diff != 0u || max_slack_diff != 0u ||
            std::fabs(ring_cos - 1.0) > 1e-12 || std::fabs(slack_cos - 1.0) > 1e-12)
        {
            std::fprintf(stderr, "[FAIL] Gate 1B EncoderStateBlock lossless invariant failed!\n");
            return 1;
        }
    }

    // -------------------------------------------------------------------------
    // Gate 2: Checksum Bit-Flip Rejection + Overlay UUID & Image Salt Isolation
    // -------------------------------------------------------------------------
    {
        RadixBlockEntry tmp_out;
        const RadixKey128 wrong_uuid{model_uuid.lo ^ 1ULL, model_uuid.hi};
        if (cold_store.read_block_pread(lookup_keys[0], wrong_uuid, prompt_toks.data(), 0ULL, tmp_out, false)) {
            std::fprintf(stderr, "[FAIL] Gate 2: accepted wrong model_overlay_uuid!\n");
            return 1;
        }
        if (cold_store.read_block_pread(lookup_keys[0], model_uuid, prompt_toks.data(), 0x1234ULL, tmp_out, false)) {
            std::fprintf(stderr, "[FAIL] Gate 2: accepted mismatched extra_salt!\n");
            return 1;
        }

        // Flip 1 bit in block 0 payload on disk and confirm FNV-1a64 catches it
        const std::string p0 = cold_store.block_path(lookup_keys[0]);
        const int fd = ::open(p0.c_str(), O_RDWR);
        uint8_t byte_val = 0;
        if (fd < 0 || ::pread(fd, &byte_val, 1, 1024) != 1) {
            std::fprintf(stderr, "[FAIL] Gate 2: could not open block 0 for bit flip\n");
            return 1;
        }
        byte_val ^= 0x04u;
        if (::pwrite(fd, &byte_val, 1, 1024) != 1) {
            (void)::close(fd);
            return 1;
        }
        (void)::close(fd);

        if (cold_store.read_block_pread(lookup_keys[0], model_uuid, prompt_toks.data(), 0ULL, tmp_out, false)) {
            std::fprintf(stderr, "[FAIL] Gate 2: accepted corrupted .kvblk after 1-bit flip!\n");
            return 1;
        }
        std::printf("[VERIFY_NVME_KV] Gate 2: FNV-1a64 bit-flip rejection (.kvblk) + UUID/salt isolation PASS\n");
    }

    // -------------------------------------------------------------------------
    // Gate 3: Bounded mtime LRU Directory Eviction Cap
    // -------------------------------------------------------------------------
    const std::string lru_dir = test_dir + "_lru";
    rm_rf_dir(lru_dir);
    {
        constexpr uint64_t kCapBytes = 1536ULL * 1024ULL;  // 1.5 MiB (~13 .kvblk shards max)
        Dsv41NvmeKvStore lru_store;
        lru_store.configure(lru_dir, kCapBytes);

        for (int b = 0; b < 12; ++b) {
            (void)lru_store.write_block_atomic(ref_blocks[size_t(b)], model_uuid);
            const std::string bp = lru_store.block_path(ref_blocks[size_t(b)].key);
            struct timespec ts[2]{};
            ts[0].tv_sec = 1700000000 + b;
            ts[1].tv_sec = 1700000000 + b;
            (void)::utimensat(AT_FDCWD, bp.c_str(), ts, 0);
        }
        lru_store.touch_blocks_mtime({ref_blocks[0].key});
        for (int b = 12; b < 24; ++b) {
            const std::string bp = lru_store.block_path(ref_blocks[size_t(b)].key);
            (void)lru_store.write_block_atomic(ref_blocks[size_t(b)], model_uuid);
            struct timespec ts[2]{};
            ts[0].tv_sec = 1700000000 + b;
            ts[1].tv_sec = 1700000000 + b;
            (void)::utimensat(AT_FDCWD, bp.c_str(), ts, 0);
        }
        lru_store.enforce_lru_cap();
        const uint64_t final_sz = lru_store.total_bytes();
        const bool blk0_alive   = lru_store.has_block_on_disk(ref_blocks[0].key);
        const bool blk1_evicted = !lru_store.has_block_on_disk(ref_blocks[1].key);
        std::printf("[VERIFY_NVME_KV] Gate 3: LRU cap=%llu B -> final=%llu B (blk0_alive=%d, blk1_evicted=%d, swa_files=%zu) PASS\n",
                    static_cast<unsigned long long>(kCapBytes),
                    static_cast<unsigned long long>(final_sz),
                    int(blk0_alive), int(blk1_evicted),
                    lru_store.count_swa_files_on_disk());
        if (final_sz > kCapBytes || !blk0_alive || !blk1_evicted || lru_store.count_swa_files_on_disk() != 0u) {
            std::fprintf(stderr, "[FAIL] Gate 3 LRU eviction invariant violated\n");
            return 1;
        }
    }

    for (int tp = 0; tp < n_gpus; ++tp) {
        *h_dpm_flag[size_t(tp)] = 0u;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    for (int tp = 0; tp < n_gpus; ++tp) {
        HIP_ASSERT(hipSetDevice(tp));
        HIP_ASSERT(hipStreamSynchronize(warm_streams[size_t(tp)]));
        HIP_ASSERT(hipStreamDestroy(warm_streams[size_t(tp)]));
        HIP_ASSERT(hipFree(d_dpm_scratch[size_t(tp)]));
        HIP_ASSERT(hipHostFree(h_dpm_flag[size_t(tp)]));
        HIP_ASSERT(hipFree(d_radix_kv[size_t(tp)]));
        HIP_ASSERT(hipFree(d_enc_win_kv[size_t(tp)]));
        HIP_ASSERT(hipStreamDestroy(streams[size_t(tp)]));
        HIP_ASSERT(hipHostFree(h_pinned[size_t(tp)]));
    }
    rm_rf_dir(test_dir);
    rm_rf_dir(test_dir + "_lru");

    std::printf("[VERIFY_NVME_KV] ALL WAVE 8.2 PURE-BLOCK NVME CSA2 KV & 20L STATEBLOCK GATES PASSED\n");
    return 0;
}
