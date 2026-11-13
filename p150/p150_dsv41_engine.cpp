// Pure C++20 2x Tenstorrent Blackhole P150A TP2 HLUT2 MoE Engine Implementation (Speed Wave 3)
// Zero Python anywhere.
// Key Hardware Optimizations:
//   1. Zero-softfloat RV32IM writer_b + hardware SFPMUL scale multiply in TRISC_MATH (~10x faster Phase 1).
//   2. L1 Hot-Expert SRAM Pinning on Layers 0..9 + MeshTrace replay for both decode and prefill.
//   3. On-Chip Gather/Tilize + Scatter/Accumulate Expert-Compact m_tiles=1 Pipeline for M = 33..512 prefill
//      (unpacks each active expert once per 512-token chunk, 5 MiB H2D/D2H, no 64-row re-unpacking!).
//   4. Persistent 4-thread AVX2 WHT128 rotation pool for M > 8 + zero-overhead inline path for M <= 8.

#include "p150_dsv41_engine.hpp"
#include "../src/dsv41_pack_format.hpp"

#include <ttnn/device.hpp>
#include <ttnn/cluster.hpp>
#include <ttnn/config.hpp>
#include <ttnn/distributed/api.hpp>
#include <ttnn/tensor/tensor_ops.hpp>
#include <ttnn/operations/generic/generic_op.hpp>
#include <ttnn/operations/ccl/all_reduce/all_reduce.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/data_movement/untilize/untilize.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/allocation_context.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <sys/mman.h>
#include <thread>
#include <tuple>
#include <vector>

namespace dsv41 {

using namespace tt::tt_metal;

namespace {

struct TraceStorageContextGuard {
    TraceStorageContextGuard()  { tt::tt_metal::push_allocation_context("trace_storage"); }
    ~TraceStorageContextGuard() { tt::tt_metal::pop_allocation_context(); }
    TraceStorageContextGuard(const TraceStorageContextGuard&) = delete;
    TraceStorageContextGuard& operator=(const TraceStorageContextGuard&) = delete;
};

static constexpr uint32_t kBatchSlots      = 16u;    // 16 expert slots/launch (fits 95%+ of M=6 decode in 1 batch & 25% fewer prefill launches!)
static constexpr uint32_t kMaxCompactSlots = 288u;   // 18 * 16 = 288 >= 272 max compact slots for M=512
static constexpr uint32_t kSlotIdsBytes    = 640u;   // [0..575]=288 uint16_t slot_ids, [576..639]=64B layer_desc
static constexpr uint32_t kNumHotL1Layers  = 0u;     // Keep full L1 bottom free for CBs + 16-slot CoreWorkspace1!

void check_no_existing_p150_owner() {
    namespace fs = std::filesystem;
    const std::string self_pid = std::to_string(::getpid());
    for (const auto& proc : fs::directory_iterator("/proc")) {
        const auto pid = proc.path().filename().string();
        if (pid.empty() || pid.find_first_not_of("0123456789") != std::string::npos || pid == self_pid) continue;
        std::error_code ec;
        fs::directory_iterator fd_it(proc.path() / "fd", fs::directory_options::skip_permission_denied, ec);
        if (ec) continue;
        for (const auto& entry : fd_it) {
            const auto link = fs::read_symlink(entry.path(), ec);
            if (ec) { ec.clear(); continue; }
            if (link.string().rfind("/dev/tenstorrent/", 0) == 0) {
                throw std::runtime_error("P150 device already owned by PID " + pid);
            }
        }
    }
}

struct LockedBuffer {
    void* ptr = nullptr;
    size_t size = 0;

    explicit LockedBuffer(size_t nbytes) : size((nbytes + 4095ULL) & ~4095ULL) {
        if (::posix_memalign(&ptr, 4096, size) != 0) throw std::bad_alloc();
        (void)::mlock(ptr, size);
        std::memset(ptr, 0, size);
    }
    ~LockedBuffer() {
        if (ptr) {
            (void)::munlock(ptr, size);
            ::free(ptr);
        }
    }
    LockedBuffer(const LockedBuffer&) = delete;
    LockedBuffer& operator=(const LockedBuffer&) = delete;
};

class HostAvxPool {
public:
    static constexpr uint32_t kWorkers = 7u;  // + caller thread = 8 parallel AVX2 streams

    HostAvxPool() {
        for (uint32_t i = 0; i < kWorkers; ++i) {
            workers_.emplace_back([this, i]() { worker_loop(i + 1u); });
        }
    }
    ~HostAvxPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    template <typename Fn>
    void parallel_rows(uint32_t total_rows, Fn&& fn) {
        if (total_rows <= 2u) {
            fn(0u, total_rows);
            return;
        }
        fn_ = [&](uint32_t tid) {
            const uint32_t n = kWorkers + 1u;
            const uint32_t r0 = (total_rows * tid) / n;
            const uint32_t r1 = (total_rows * (tid + 1u)) / n;
            if (r0 < r1) fn(r0, r1);
        };
        pending_.store(kWorkers, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(mu_);
            ++gen_;
        }
        cv_.notify_all();
        fn_(0u);
        while (pending_.load(std::memory_order_acquire) != 0u) {
            dsv41_cpu_relax();
        }
    }

private:
    void worker_loop(uint32_t tid) {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&]() { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
            }
            fn_(tid);
            pending_.fetch_sub(1u, std::memory_order_acq_rel);
        }
    }

    std::mutex mu_;
    std::condition_variable cv_;
    uint64_t gen_ = 0;
    bool stop_ = false;
    std::atomic<uint32_t> pending_{0};
    std::function<void(uint32_t)> fn_;
    std::vector<std::thread> workers_;
};

CoreRangeSet make_core_set(const std::vector<CoreCoord>& coords) {
    std::set<CoreRange> ranges;
    for (const auto& c : coords) ranges.emplace(c, c);
    return CoreRangeSet(ranges);
}

CBDescriptor make_cb(uint32_t id, uint32_t pages, uint32_t page_bytes, tt::DataFormat fmt, const CoreRangeSet& cores) {
    CBDescriptor cb;
    cb.total_size = pages * page_bytes;
    cb.core_ranges = cores;
    cb.format_descriptors.push_back(CBFormatDescriptor{static_cast<uint8_t>(id), fmt, page_bytes, {}, {}});
    return cb;
}

std::vector<uint32_t> make_accessor_args(std::initializer_list<const ttnn::Tensor*> tensors) {
    std::vector<uint32_t> out;
    for (const auto* t : tensors) {
        const auto a = TensorAccessorArgs(t->buffer()).get_compile_time_args();
        out.insert(out.end(), a.begin(), a.end());
    }
    return out;
}

KernelDescriptor make_kernel(
    const std::string& path,
    const CoreRangeSet& cores,
    KernelDescriptor::ConfigDescriptor config,
    std::vector<uint32_t> ct_args = {})
{
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("Missing P150 kernel file: " + path);
    }
    KernelDescriptor kd;
    kd.kernel_source = path;
    kd.source_type = KernelDescriptor::SourceType::FILE_PATH;
    kd.core_ranges = cores;
    kd.config = std::move(config);
    kd.compile_time_args = std::move(ct_args);
    return kd;
}

ttnn::Tensor run_generic_preserving_topology(
    const std::vector<ttnn::Tensor>& tensors,
    const ProgramDescriptor& program)
{
    if (tensors.empty()) throw std::invalid_argument("run_generic_preserving_topology: empty tensor list");
    const auto topo = tensors.back().tensor_topology();
    auto res = ttnn::generic_op(tensors, program);
    if (res.buffer() != tensors.back().buffer()) {
        throw std::runtime_error("ttnn::generic_op replaced caller-owned output buffer");
    }
    res.update_tensor_topology(topo);
    return res;
}

ttnn::Tensor alloc_l1(
    distributed::MeshDevice& mesh,
    const Shape& shape,
    DataType dtype,
    Layout layout,
    bool sharded_axis0)
{
    return ttnn::create_device_tensor(
        TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), ttnn::L1_MEMORY_CONFIG)),
        &mesh,
        sharded_axis0
            ? TensorTopology::create_sharded_tensor_topology(mesh.shape(), 0)
            : TensorTopology::create_fully_replicated_tensor_topology(mesh.shape()));
}

ttnn::Tensor alloc_dram_sharded0(
    distributed::MeshDevice& mesh,
    const Shape& shape,
    DataType dtype,
    Layout layout)
{
    return ttnn::create_device_tensor(
        TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), ttnn::DRAM_MEMORY_CONFIG)),
        &mesh,
        TensorTopology::create_sharded_tensor_topology(mesh.shape(), 0));
}

ttnn::Tensor alloc_dram_replicated(
    distributed::MeshDevice& mesh,
    const Shape& shape,
    DataType dtype,
    Layout layout)
{
    return ttnn::create_device_tensor(
        TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), ttnn::DRAM_MEMORY_CONFIG)),
        &mesh,
        TensorTopology::create_fully_replicated_tensor_topology(mesh.shape()));
}

void write_sharded_pair_blocking(
    distributed::MeshDevice& mesh,
    ttnn::Tensor& dst,
    const void* tp0_data,
    const void* tp1_data)
{
    auto borrowed = std::shared_ptr<distributed::MeshBuffer>(
        const_cast<distributed::MeshBuffer*>(&dst.mesh_buffer()), [](auto*) {});
    mesh.mesh_command_queue().enqueue_write_shards(
        borrowed,
        {distributed::ShardDataTransfer{distributed::MeshCoordinate{0, 0}}.host_data(const_cast<void*>(tp0_data)),
         distributed::ShardDataTransfer{distributed::MeshCoordinate{0, 1}}.host_data(const_cast<void*>(tp1_data))},
        true);
}

struct ResidentLayerBank {
    uint32_t layer = 0;
    std::string tp0_path;
    std::string tp1_path;
    std::vector<uint64_t> signs_h;  // [80] uint64_t for H=5120 (40 blocks of 128)
    ttnn::Tensor gate_pages;        // DRAM sharded [2, 168*360, 1088] UINT32 (4352 B/page)
    ttnn::Tensor up_pages;          // DRAM sharded [2, 168*360, 1088] UINT32 (4352 B/page)
    ttnn::Tensor down_pages;        // DRAM sharded [2, 168*360, 1088] UINT32 (4352 B/page)
    ttnn::Tensor centroids;         // DRAM sharded [2, 168, 16] FLOAT32 (64 B/page: gc0, g_dc, uc0, u_dc, dc0, d_dc)
    ttnn::Tensor wht_q_i;           // DRAM sharded [2, 9, 128, 128] BFLOAT16 TILE (144 tiles/card)

    std::array<uint16_t, DSV41_P150_NUM_EXPERTS> hot_slot_for_eid = []() {
        std::array<uint16_t, DSV41_P150_NUM_EXPERTS> a{};
        a.fill(0xFFFFu);
        return a;
    }();

    // Persistent read-only OS page-cache mmaps for zero-RAM concurrent EPYC AVX2 tail-expert offload!
    std::array<std::unique_ptr<SafetensorsMmap>, 2> st_mmap{};
    std::array<const uint8_t*, 2> gp_mmap{nullptr, nullptr};
    std::array<const uint8_t*, 2> up_mmap{nullptr, nullptr};
    std::array<const uint8_t*, 2> dp_mmap{nullptr, nullptr};
    std::array<const float*, 2>   gc_mmap{nullptr, nullptr};
    std::array<const float*, 2>   uc_mmap{nullptr, nullptr};
    std::array<const float*, 2>   dc_mmap{nullptr, nullptr};
    std::array<std::array<uint64_t, 18>, 2> signs_i_mmap{};

    // Active per-expert page & centroid indirection (base mmap for cold experts, RAREBAL overlay mmap for hot experts!)
    std::array<std::array<const uint8_t*, DSV41_P150_NUM_EXPERTS>, 2> gp_exp{};
    std::array<std::array<const uint8_t*, DSV41_P150_NUM_EXPERTS>, 2> up_exp{};
    std::array<std::array<const uint8_t*, DSV41_P150_NUM_EXPERTS>, 2> dp_exp{};
    std::array<std::array<float, DSV41_P150_NUM_EXPERTS * 2u>, 2> gc_act{};
    std::array<std::array<float, DSV41_P150_NUM_EXPERTS * 2u>, 2> uc_act{};
    std::array<std::array<float, DSV41_P150_NUM_EXPERTS * 2u>, 2> dc_act{};
};

struct CoreWorkspace1 {
    uint32_t m_tiles_ws = kBatchSlots;      // 16 expert slots batched in L1 per kernel launch
    uint32_t max_slots = kMaxCompactSlots;  // 288 slots (serves both M<=32 decode & M<=512 compact prefill!)
    uint32_t k_splits_up = 10;
    uint32_t k_splits_down = 2;
    bool x_tilized_dram = false;    // Tier 1 (14..18): demotes 16-slot M>32 prefill gather arena (5.0 MiB) to replicated GDDR6
    bool x_decode_dram = false;     // Tier 2 (18 only): demotes 1-slot M<=32 decode activation tile (320 KiB) to replicated GDDR6
    bool weights_dram = false;      // Tier 1b (17..18): demotes scalar combine weights (1.125 MiB) to GDDR6 under FABRIC_1D

    ttnn::Tensor x_tilized;         // Replicated [1, 16, 32, 5120] BFLOAT16 TILE (5.0 MiB M>32 prefill gather)
    ttnn::Tensor x_decode;          // Replicated [1, 1, 32, 5120] BFLOAT16 TILE (320 KiB dedicated L1 decode tile at 14..17!)
    ttnn::Tensor slot_ids;          // L1 replicated [1, 1, 1, kSlotIdsBytes / 2u] UINT16 ROW_MAJOR (640 B)
    ttnn::Tensor weights_tiled;     // Replicated [1, 288, 32, 32] FLOAT32 TILE (1.125 MiB)
    ttnn::Tensor gate_partials;     // L1 sharded    [2, k_splits_up, 16*32, 1152] FLOAT32 TILE
    ttnn::Tensor up_partials;       // L1 sharded    [2, k_splits_up, 16*32, 1152] FLOAT32 TILE
    ttnn::Tensor z_rot_bf16;        // L1 sharded    [2, 1, 16*32, 1152] BFLOAT16 TILE
    ttnn::Tensor down_partials;     // L1 sharded    [2, k_splits_down, 16*32, 5120] FLOAT32 TILE
    ttnn::Tensor combined_f32;      // L1 sharded    [2, 1, 32, 5120] FLOAT32 TILE (M<=32 decode output)
    std::array<ttnn::Tensor, 3> finish_decode;

    CoreWorkspace1(distributed::MeshDevice& mesh, uint32_t k_up, uint32_t k_dn,
                   bool dram_prefill_x, bool dram_decode_x, bool dram_weights)
        : k_splits_up(k_up),
          k_splits_down(k_dn),
          x_tilized_dram(dram_prefill_x),
          x_decode_dram(dram_decode_x),
          weights_dram(dram_weights),
          x_tilized(dram_prefill_x
                        ? alloc_dram_replicated(mesh, Shape({1, kBatchSlots, 32u, 5120u}), DataType::BFLOAT16, Layout::TILE)
                        : alloc_l1(mesh, Shape({1, kBatchSlots, 32u, 5120u}), DataType::BFLOAT16, Layout::TILE, false)),
          x_decode((dram_prefill_x && !dram_decode_x)
                       ? alloc_l1(mesh, Shape({1, 1u, 32u, 5120u}), DataType::BFLOAT16, Layout::TILE, false)
                       : x_tilized),
          slot_ids(alloc_l1(mesh, Shape({1, 1, 1, kSlotIdsBytes / 2u}), DataType::UINT16, Layout::ROW_MAJOR, false)),
          weights_tiled(dram_weights
                            ? alloc_dram_replicated(mesh, Shape({1, kMaxCompactSlots, 32u, 32u}), DataType::FLOAT32, Layout::TILE)
                            : alloc_l1(mesh, Shape({1, kMaxCompactSlots, 32u, 32u}), DataType::FLOAT32, Layout::TILE, false)),
          gate_partials(alloc_l1(mesh, Shape({1, k_up, kBatchSlots * 32u, 1152u}), DataType::FLOAT32, Layout::TILE, true)),
          up_partials(alloc_l1(mesh, Shape({1, k_up, kBatchSlots * 32u, 1152u}), DataType::FLOAT32, Layout::TILE, true)),
          z_rot_bf16(alloc_l1(mesh, Shape({1, 1, kBatchSlots * 32u, 1152u}), DataType::BFLOAT16, Layout::TILE, true)),
          down_partials(gate_partials),
          combined_f32(alloc_l1(mesh, Shape({1, 1, 32u, 5120u}), DataType::FLOAT32, Layout::TILE, true))
    {
        if (gate_partials.buffer()->size() < size_t(k_dn) * kBatchSlots * 160u * 4096ULL) {
            throw std::runtime_error("gate_partials smaller than down_partials requirement");
        }
    }

    const MemoryConfig& finish_untilize_mem_cfg() const {
        return weights_dram ? ttnn::DRAM_MEMORY_CONFIG : ttnn::L1_MEMORY_CONFIG;
    }

    size_t l1_bytes_per_card() const {
        size_t bytes = slot_ids.buffer()->size() +
                       gate_partials.buffer()->size() + up_partials.buffer()->size() +
                       z_rot_bf16.buffer()->size() + combined_f32.buffer()->size();
        if (!x_tilized_dram) {
            bytes += x_tilized.buffer()->size();
        } else if (!x_decode_dram) {
            bytes += x_decode.buffer()->size();
        }
        if (!weights_dram) bytes += weights_tiled.buffer()->size();
        for (const auto& t : finish_decode) {
            if (t.is_allocated() && t.buffer() != nullptr && t.buffer()->is_l1()) {
                bytes += t.buffer()->size();
            }
        }
        return bytes;
    }
};

struct CompactPrefillArena {
    uint32_t max_rows = 512;        // Dumitrescu MR vs MP block law: y_accum_f32 in fast on-chip L1 MR, streamed finish_tensors in GDDR6 MP!
    bool use_dram = false;
    ttnn::Tensor y_accum_f32;       // Sharded [2, 1, max_rows, 5120] FLOAT32 TILE (used as 5 x 4096 B pages/row in L1 MR!)
    std::array<ttnn::Tensor, 2> finish_tensors;  // [0]=typecast BF16 (GDDR6 MP), [1]=all_reduce BF16 (GDDR6 MP)

    CompactPrefillArena(distributed::MeshDevice& mesh, uint32_t mrows, bool dram)
        : max_rows(mrows),
          use_dram(dram),
          y_accum_f32(dram
              ? alloc_dram_sharded0(mesh, Shape({1, 1, mrows, 5120u}), DataType::FLOAT32, Layout::TILE)
              : alloc_l1(mesh, Shape({1, 1, mrows, 5120u}), DataType::FLOAT32, Layout::TILE, true))
    {}

    const MemoryConfig& mem_cfg() const {
        return use_dram ? ttnn::DRAM_MEMORY_CONFIG : ttnn::L1_MEMORY_CONFIG;
    }

    const MemoryConfig& finish_mem_cfg() const {
        return ttnn::DRAM_MEMORY_CONFIG;
    }

    size_t l1_bytes_per_card() const {
        return use_dram ? 0ULL : y_accum_f32.buffer()->size();
    }
};

uint32_t choose_decode_slot_bucket(uint32_t active) {
    static constexpr uint32_t kBuckets[] = {4u, 8u, 12u, 16u, 20u, 24u, 32u};
    for (uint32_t b : kBuckets) {
        if (active <= b) return b;
    }
    return std::min<uint32_t>(((active + kBatchSlots - 1u) / kBatchSlots) * kBatchSlots, kMaxCompactSlots);
}

uint32_t choose_compact_slot_bucket(uint32_t active, uint32_t mode) {
    if (mode == 1u) {
        static constexpr uint32_t kBuckets128[] = {32u, 64u, 96u, 128u, 160u, 192u, 224u};
        for (uint32_t b : kBuckets128) {
            if (active <= b) return b;
        }
        return std::min<uint32_t>(((active + kBatchSlots - 1u) / kBatchSlots) * kBatchSlots, kMaxCompactSlots);
    }
    static constexpr uint32_t kBuckets512[] = {
        32u, 64u, 96u, 128u, 160u, 192u, 224u, 256u, kMaxCompactSlots};
    for (uint32_t b : kBuckets512) {
        if (active <= b) return b;
    }
    return kMaxCompactSlots;
}

template <int MR>
inline void avx2_p150_proj_tmpl(
    const uint8_t* __restrict__ pages_base,
    float c0,
    float c1,
    int nt4,
    int kt4,
    const float* const* __restrict__ x_rows,
    float* const* __restrict__ y_rows)
{
    const __m256 v_lut = _mm256_setr_ps(c0, c1, -c0, -c1, c0, c1, -c0, -c1);
    for (int nb4 = 0; nb4 < nt4; ++nb4) {
        for (int kb4 = 0; kb4 < kt4; ++kb4) {
            const uint8_t* page = pages_base + size_t(nb4 * kt4 + kb4) * 4352ULL;
            const uint32_t* codes = reinterpret_cast<const uint32_t*>(page);
            const uint16_t* scales = reinterpret_cast<const uint16_t*>(page + 4096);

            for (int j_grp = 0; j_grp < 4; ++j_grp) {
                const int col_face_off = ((j_grp >> 1) << 8) + ((j_grp & 1) << 3);
                __m256 sb_acc[MR][4];
                for (int m = 0; m < MR; ++m) {
                    for (int ns = 0; ns < 4; ++ns) sb_acc[m][ns] = _mm256_setzero_ps();
                }

                for (int i = 0; i < 32; ++i) {
                    const int tile_idx = ((i >> 4) << 9) + ((i & 15) << 4) + col_face_off;
                    const __m256i w_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + tile_idx));

                    auto step_kt = [&](auto kt_c) __attribute__((always_inline)) {
                        constexpr int ks = decltype(kt_c)::value;
                        const int k_idx = (kb4 * 4 + ks) * 32 + i;
                        __m256 xv[MR];
                        for (int m = 0; m < MR; ++m) xv[m] = _mm256_set1_ps(x_rows[m][k_idx]);

                        auto step_nt = [&](auto nt_c) __attribute__((always_inline)) {
                            constexpr int ns = decltype(nt_c)::value;
                            constexpr int sh = 2 * (ks * 4 + ns);
                            const __m256i idx = (sh == 0) ? w_vec : _mm256_srli_epi32(w_vec, sh);
                            const __m256 w_val = _mm256_permutevar_ps(v_lut, idx);
                            for (int m = 0; m < MR; ++m) {
                                sb_acc[m][ns] = _mm256_fmadd_ps(xv[m], w_val, sb_acc[m][ns]);
                            }
                        };
                        step_nt(std::integral_constant<int, 0>{});
                        step_nt(std::integral_constant<int, 1>{});
                        step_nt(std::integral_constant<int, 2>{});
                        step_nt(std::integral_constant<int, 3>{});
                    };
                    step_kt(std::integral_constant<int, 0>{});
                    step_kt(std::integral_constant<int, 1>{});
                    step_kt(std::integral_constant<int, 2>{});
                    step_kt(std::integral_constant<int, 3>{});
                }

                for (int ns = 0; ns < 4; ++ns) {
                    const int sc_off = ns * 32 + j_grp * 8;
                    const int out_col = (nb4 * 4 + ns) * 32 + j_grp * 8;
                    const __m256 sig = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(scales + sc_off)));
                    if (kb4 == 0) {
                        for (int m = 0; m < MR; ++m) {
                            _mm256_storeu_ps(y_rows[m] + out_col, _mm256_mul_ps(sb_acc[m][ns], sig));
                        }
                    } else {
                        for (int m = 0; m < MR; ++m) {
                            const __m256 prev = _mm256_loadu_ps(y_rows[m] + out_col);
                            _mm256_storeu_ps(y_rows[m] + out_col, _mm256_fmadd_ps(sb_acc[m][ns], sig, prev));
                        }
                    }
                }
            }
        }
    }
}

struct TailExpertDesc {
    uint16_t eid;
    uint16_t cnt;
    uint32_t pair_offset;
    uint16_t rows[32];
    float    weights[32];
};

inline void run_cpu_tail_shard_avx2(
    const ResidentLayerBank& bank,
    uint32_t tp,
    const TailExpertDesc& desc,
    const float* x_rot_f32,
    float* pair_out_base)
{
    const uint32_t e = desc.eid;
    const uint8_t* gp = bank.gp_exp[tp][e];
    const uint8_t* up = bank.up_exp[tp][e];
    const uint8_t* dp = bank.dp_exp[tp][e];
    const float gc0 = bank.gc_act[tp][2u * e + 0u], gc1 = bank.gc_act[tp][2u * e + 1u];
    const float uc0 = bank.uc_act[tp][2u * e + 0u], uc1 = bank.uc_act[tp][2u * e + 1u];
    const float dc0 = bank.dc_act[tp][2u * e + 0u], dc1 = bank.dc_act[tp][2u * e + 1u];
    const uint64_t* si = bank.signs_i_mmap[tp].data();

    alignas(32) float g_buf[3][1152];
    alignas(32) float u_buf[3][1152];
    alignas(32) float z_buf[3][1152];

    uint32_t pos = 0;
    while (pos < desc.cnt) {
        const uint32_t mr = std::min<uint32_t>(3u, desc.cnt - pos);
        const float* x_ptrs[3];
        float* g_ptrs[3];
        float* u_ptrs[3];
        float* z_ptrs[3];
        float* d_ptrs[3];
        for (uint32_t m = 0; m < mr; ++m) {
            x_ptrs[m] = x_rot_f32 + size_t(desc.rows[pos + m]) * DSV41_P150_H;
            g_ptrs[m] = g_buf[m];
            u_ptrs[m] = u_buf[m];
            z_ptrs[m] = z_buf[m];
            d_ptrs[m] = pair_out_base + size_t(desc.pair_offset + pos + m) * DSV41_P150_H;
        }

        if (mr == 3u) {
            avx2_p150_proj_tmpl<3>(gp, gc0, gc1, 9, 40, x_ptrs, g_ptrs);
            avx2_p150_proj_tmpl<3>(up, uc0, uc1, 9, 40, x_ptrs, u_ptrs);
        } else if (mr == 2u) {
            avx2_p150_proj_tmpl<2>(gp, gc0, gc1, 9, 40, x_ptrs, g_ptrs);
            avx2_p150_proj_tmpl<2>(up, uc0, uc1, 9, 40, x_ptrs, u_ptrs);
        } else {
            avx2_p150_proj_tmpl<1>(gp, gc0, gc1, 9, 40, x_ptrs, g_ptrs);
            avx2_p150_proj_tmpl<1>(up, uc0, uc1, 9, 40, x_ptrs, u_ptrs);
        }

        for (uint32_t m = 0; m < mr; ++m) {
            for (int k = 0; k < 1152; ++k) z_buf[m][k] = dsv41_swiglu_scalar(g_buf[m][k], u_buf[m][k]);
            for (int b = 0; b < 9; ++b) avx2_wht128_block(z_buf[m] + b * 128, &si[b * 2]);
        }

        const float* z_cptrs[3] = {z_buf[0], z_buf[1], z_buf[2]};
        if (mr == 3u) {
            avx2_p150_proj_tmpl<3>(dp, dc0, dc1, 40, 9, z_cptrs, d_ptrs);
        } else if (mr == 2u) {
            avx2_p150_proj_tmpl<2>(dp, dc0, dc1, 40, 9, z_cptrs, d_ptrs);
        } else {
            avx2_p150_proj_tmpl<1>(dp, dc0, dc1, 40, 9, z_cptrs, d_ptrs);
        }

        for (uint32_t m = 0; m < mr; ++m) {
            const __m256 vw = _mm256_set1_ps(desc.weights[pos + m]);
            float* dst = d_ptrs[m];
            for (uint32_t j = 0; j < DSV41_P150_H; j += 8u) {
                _mm256_storeu_ps(dst + j, _mm256_mul_ps(_mm256_loadu_ps(dst + j), vw));
            }
        }
        pos += mr;
    }
}

class CpuTailPool {
public:
    static constexpr uint32_t kWorkers = 28u;  // 28 workers + caller = 29 dedicated Zen 2 physical cores (zero SMT FPU contention!)

    CpuTailPool() {
        for (uint32_t i = 0; i < kWorkers; ++i) {
            workers_.emplace_back([this]() { worker_loop(); });
        }
    }
    ~CpuTailPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    void launch_async(uint32_t num_items, std::function<void(uint32_t)> item_fn) {
        item_fn_ = std::move(item_fn);
        total_items_ = num_items;
        next_item_.store(0u, std::memory_order_relaxed);
        active_workers_.store(kWorkers, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(mu_);
            ++gen_;
        }
        cv_.notify_all();
    }

    void wait_complete() {
        for (;;) {
            const uint32_t idx = next_item_.fetch_add(1u, std::memory_order_relaxed);
            if (idx >= total_items_) break;
            item_fn_(idx);
        }
        while (active_workers_.load(std::memory_order_acquire) != 0u) {
            dsv41_cpu_relax();
        }
    }

private:
    void worker_loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&]() { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
            }
            for (;;) {
                const uint32_t idx = next_item_.fetch_add(1u, std::memory_order_relaxed);
                if (idx >= total_items_) break;
                item_fn_(idx);
            }
            active_workers_.fetch_sub(1u, std::memory_order_acq_rel);
        }
    }

    std::mutex mu_;
    std::condition_variable cv_;
    uint64_t gen_ = 0;
    bool stop_ = false;
    uint32_t total_items_ = 0;
    std::atomic<uint32_t> next_item_{0};
    std::atomic<uint32_t> active_workers_{0};
    std::function<void(uint32_t)> item_fn_;
    std::vector<std::thread> workers_;
};

const std::array<std::array<uint16_t, 32>, 32>& tile_offset_lut() {
    static const auto lut = []() {
        std::array<std::array<uint16_t, 32>, 32> t{};
        for (int r = 0; r < 32; ++r) {
            for (int c = 0; c < 32; ++c) {
                t[r][c] = static_cast<uint16_t>(tensix_tile_offset(r, c));
            }
        }
        return t;
    }();
    return lut;
}

inline void bf16_row_to_f32_avx2(const uint16_t* src_bf16, float* dst_f32) {
    for (uint32_t j = 0; j < DSV41_P150_H; j += 8u) {
        const __m128i b16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_bf16 + j));
        const __m256i u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(b16), 16);
        _mm256_store_ps(dst_f32 + j, _mm256_castsi256_ps(u32));
    }
}

inline void f32_row_to_bf16_avx2(const float* src_f32, uint16_t* dst_bf16) {
    const __m256i v_7fff = _mm256_set1_epi32(0x00007FFFu);
    const __m256i v_one  = _mm256_set1_epi32(1);
    for (uint32_t j = 0; j < DSV41_P150_H; j += 16u) {
        const __m256i u0 = _mm256_castps_si256(_mm256_load_ps(src_f32 + j));
        const __m256i u1 = _mm256_castps_si256(_mm256_load_ps(src_f32 + j + 8u));
        const __m256i r0 = _mm256_srli_epi32(_mm256_add_epi32(u0, _mm256_add_epi32(v_7fff, _mm256_and_si256(_mm256_srli_epi32(u0, 16), v_one))), 16);
        const __m256i r1 = _mm256_srli_epi32(_mm256_add_epi32(u1, _mm256_add_epi32(v_7fff, _mm256_and_si256(_mm256_srli_epi32(u1, 16), v_one))), 16);
        const __m256i pk = _mm256_packus_epi32(r0, r1);
        const __m256i ord = _mm256_permute4x64_epi64(pk, 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_bf16 + j), ord);
    }
}

inline void rotate_row_avx2_to_bf16(
    const uint16_t* src_row,
    const uint64_t* signs_h,
    uint16_t* dst_bf16_row,
    float* optional_dst_f32_row = nullptr)
{
    alignas(32) float local_rot[DSV41_P150_H];
    float* rot_row = optional_dst_f32_row ? optional_dst_f32_row : local_rot;
    bf16_row_to_f32_avx2(src_row, rot_row);
    for (int b = 0; b < 40; ++b) {
        avx2_wht128_block(rot_row + b * 128, &signs_h[b * 2]);
    }
    f32_row_to_bf16_avx2(rot_row, dst_bf16_row);
}

}  // namespace

struct P150Dsv41Engine::Impl {
    std::string packed_root;
    std::string kernels_dir;
    bool enable_trace = true;
    std::unique_ptr<SafetensorsMmap> rarebal_overlay_mmap;
    uint32_t rarebal_p150_patched = 0;

    std::shared_ptr<distributed::MeshDevice> mesh;
    CoreCoord grid{0, 0};
    uint32_t total_cores = 0;
    uint32_t k_splits_up = 12;
    uint32_t k_splits_down = 2;
    uint32_t requested_hot_shards = kP150DefaultHotShards;
    bool expanded_l1_demotion = false;
    bool decode_l1_demotion = false;
    bool weights_l1_demotion = false;

    std::unique_ptr<CoreWorkspace1> ws1;
    bool x_rm_dram = false;                       // expanded >13 demotes this 5.0 MiB compact prefill arena to GDDR6
    std::optional<ttnn::Tensor> x_rm_l1;           // L1 replicated [1, 1, 512, 5120] BFLOAT16 TILE (5.0 MiB/card)
    std::optional<ttnn::Tensor> slot_meta_l1;      // L1 replicated [1, 1, 288, 32] UINT32 ROW_MAJOR (36 KiB/card)
    std::optional<ttnn::Tensor> l1_slot_lut;       // L1 replicated [1, 40, 256] UINT16 ROW_MAJOR (20 KiB/card O(1) hot LUT!)
    std::optional<ttnn::Tensor> dummy_hot_l1;      // L1 sharded    [2, 1, 1088] UINT32 ROW_MAJOR (4352 B/card)
    std::unique_ptr<CompactPrefillArena> arena128; // L1 MR compact arena for M = 33..128 (y_accum in L1, finish in GDDR6 MP!)
    std::unique_ptr<CompactPrefillArena> arena512; // L1 MR compact arena for M = 129..512 (y_accum in L1, finish in GDDR6 MP!)

    HostAvxPool avx_pool;
    CpuTailPool tail_pool;
    LockedBuffer host_x_tilized{160u * 2048u};                      // [32, 5120] BF16 tiled = 320 KiB
    LockedBuffer host_x_rm{1024u * 5120u * sizeof(uint16_t)};       // [1024, 5120] BF16 row-major = 10.0 MiB
    LockedBuffer host_x_rot_f32{1024u * 5120u * sizeof(float)};     // [1024, 5120] FP32 rotated = 20.0 MiB
    LockedBuffer host_tail_pairs_f32{1024u * 2u * 5120u * sizeof(float)}; // [2, 1024, 5120] FP32 tail pairs = 40.0 MiB
    LockedBuffer host_slot_ids{kSlotIdsBytes};                      // 640 B
    LockedBuffer host_slot_lut{DSV41_P150_NUM_LAYERS * 256u * sizeof(uint16_t)}; // 20 KiB O(1) hot LUT
    LockedBuffer host_slot_meta{288u * 128u};                       // 36 KiB
    LockedBuffer host_weights{kMaxCompactSlots * 4096u};            // 288 * 4 KiB = 1.125 MiB
    LockedBuffer host_out_rm{1024u * 5120u * sizeof(uint16_t)};     // [1024, 5120] BF16 row-major = 10.0 MiB

    std::array<std::optional<ResidentLayerBank>, DSV41_P150_NUM_LAYERS> layers;
    uint64_t loaded_mask = 0ULL;
    size_t pinned_hot_l1_bytes = 0;

    // Priority 3: Global static L1 Hot-Expert Shard Arena (~22-26 TP2 shards across both P150A cards!)
    uint32_t num_hot_shards = 0;
    std::optional<ttnn::Tensor> hot_gate_l1;
    std::optional<ttnn::Tensor> hot_up_l1;
    std::optional<ttnn::Tensor> hot_down_l1;

    struct HotAssignment {
        uint16_t layer;
        uint16_t eid;
        bool operator==(const HotAssignment& o) const noexcept { return layer == o.layer && eid == o.eid; }
    };
    std::vector<HotAssignment> current_hot_assignments;
    std::array<std::array<uint64_t, DSV41_P150_NUM_EXPERTS>, DSV41_P150_NUM_LAYERS> decode_expert_hits{};
    std::array<std::array<uint64_t, DSV41_P150_NUM_EXPERTS>, DSV41_P150_NUM_LAYERS> prefill_expert_hits{};
    uint64_t decode_layer_calls = 0;
    uint64_t decode_total_p150_slots = 0;
    uint64_t decode_hot_l1_hits = 0;
    std::string hot_experts_path = (std::getenv("DSV41_ROOT") != nullptr
                                        ? std::string(std::getenv("DSV41_ROOT"))
                                        : std::string(".")) + "/logs/p150_hot_experts.txt";
    std::string expert_hist_path = (std::getenv("DSV41_ROOT") != nullptr
                                        ? std::string(std::getenv("DSV41_ROOT"))
                                        : std::string(".")) + "/logs/p150_expert_hist.txt";

    struct TraceSlot {
        std::optional<distributed::MeshTraceId> id;
        bool warmed = false;
        bool skipped = false;
    };
    // Layer-agnostic global MeshTraces keyed ONLY by (mode, scheduled_slots)!
    // mode: 0 = decode (M<=32), 1 = compact128 (M=33..128), 2 = compact512 (M=129..1024)
    std::map<std::pair<uint32_t, uint32_t>, TraceSlot> op_traces;  // (mode, bucket)
    std::array<TraceSlot, 3> reduce_traces;  // indexed by mode 0, 1, 2
    uint32_t trace_region_bytes = 0;
    uint32_t dirty_mt0 = 0;
    bool programs_prewarmed = false;

    Impl(const std::string& root, const std::string& kdir, bool trace, const std::string& ov_path,
         uint32_t hot_l1_shards)
        : packed_root(root), kernels_dir(kdir), enable_trace(trace)
    {
        requested_hot_shards = std::clamp<uint32_t>(hot_l1_shards, 1u, kP150MaxHotL1Shards);
        expanded_l1_demotion = p150_requires_prefill_demotion(requested_hot_shards);
        decode_l1_demotion   = p150_requires_decode_demotion(requested_hot_shards);
        weights_l1_demotion  = p150_requires_weights_demotion(requested_hot_shards);
        if (!p150_hot_shards_fit(requested_hot_shards)) {
            throw std::runtime_error("P150 hot-shard request exceeds exact per-Tensix L1 byte budget");
        }

        if (!ov_path.empty()) {
            rarebal_overlay_mmap = std::make_unique<SafetensorsMmap>(ov_path);
            std::printf("[P150_DSV41] Mmapped RAREBAL-LDLQ hot-expert overlay: %s (%zu tensors)\n",
                        ov_path.c_str(), rarebal_overlay_mmap->entries().size());
            std::fflush(stdout);
        }
        check_no_existing_p150_owner();
        ttnn::CONFIG.set<"throw_exception_on_fallback">(true);
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
        const auto dispatch = ttnn::device::create_cluster_aware_dispatch_config();
        trace_region_bytes = enable_trace ? (1536u << 20) : 0u;
        mesh = ttnn::distributed::open_mesh_device(
            ttnn::MeshShape{1, 2}, kP150L1SmallSizeBytes, trace_region_bytes, 1, dispatch, std::nullopt, {0, 1});

        grid = mesh->compute_with_storage_grid_size();
        total_cores = static_cast<uint32_t>(grid.x * grid.y);
        if (total_cores < 80u) {
            throw std::runtime_error("Blackhole P150A compute grid too small: " + std::to_string(total_cores));
        }
        k_splits_up   = (total_cores >= 90u) ? 10u : 8u;
        k_splits_down = (total_cores >= 120u) ? 3u : 2u;

        ws1 = std::make_unique<CoreWorkspace1>(*mesh, k_splits_up, k_splits_down,
                                                 expanded_l1_demotion, decode_l1_demotion, weights_l1_demotion);
        x_rm_dram = expanded_l1_demotion;
        x_rm_l1 = x_rm_dram
            ? alloc_dram_replicated(*mesh, Shape({1, 1, 512u, 5120u}), DataType::BFLOAT16, Layout::TILE)
            : alloc_l1(*mesh, Shape({1, 1, 512u, 5120u}), DataType::BFLOAT16, Layout::TILE, false);
        slot_meta_l1 = expanded_l1_demotion
            ? alloc_dram_replicated(*mesh, Shape({1, 1, 288u, 32u}), DataType::UINT32, Layout::ROW_MAJOR)
            : alloc_l1(*mesh, Shape({1, 1, 288u, 32u}), DataType::UINT32, Layout::ROW_MAJOR, false);
        l1_slot_lut = alloc_l1(*mesh, Shape({1, DSV41_P150_NUM_LAYERS, 256u}), DataType::UINT16, Layout::ROW_MAJOR, false);
        dummy_hot_l1 = expanded_l1_demotion
            ? alloc_dram_sharded0(*mesh, Shape({1, 1u, 1088u}), DataType::UINT32, Layout::ROW_MAJOR)
            : alloc_l1(*mesh, Shape({1, 1u, 1088u}), DataType::UINT32, Layout::ROW_MAJOR, true);

        // Dumitrescu MR vs MP block law: Tier 1 (14..17) demotes M>32 prefill arenas + scalar combine weights
        // while keeping x_decode [1,1,32,5120] BF16 + finish_decode 100% in L1 MR under FABRIC_1D!
        arena128 = std::make_unique<CompactPrefillArena>(*mesh, 128u, true);
        arena512 = std::make_unique<CompactPrefillArena>(*mesh, 512u, expanded_l1_demotion);

        // Allocate maximum safe static L1 Hot-Expert Shard Arena at top of L1 while guaranteeing >= 256 KiB/core bottom CB headroom!
        const uint32_t guard_pages = (total_cores * 256u * 1024u) / 4096u;
        for (uint32_t k_try = requested_hot_shards; k_try >= 8u; --k_try) {
            try {
                auto hg = alloc_l1(*mesh, Shape({1, k_try * 360u, 1088u}), DataType::UINT32, Layout::ROW_MAJOR, true);
                auto hu = alloc_l1(*mesh, Shape({1, k_try * 360u, 1088u}), DataType::UINT32, Layout::ROW_MAJOR, true);
                auto hd = alloc_l1(*mesh, Shape({1, k_try * 360u, 1088u}), DataType::UINT32, Layout::ROW_MAJOR, true);
                auto cb_guard = alloc_l1(*mesh, Shape({1, guard_pages, 1024u}), DataType::UINT32, Layout::ROW_MAJOR, true);
                (void)cb_guard;
                hot_gate_l1 = std::move(hg);
                hot_up_l1   = std::move(hu);
                hot_down_l1 = std::move(hd);
                num_hot_shards = k_try;
                pinned_hot_l1_bytes = hot_gate_l1->buffer()->size() +
                                      hot_up_l1->buffer()->size() +
                                      hot_down_l1->buffer()->size();
                break;
            } catch (...) {
                hot_gate_l1.reset();
                hot_up_l1.reset();
                hot_down_l1.reset();
            }
        }

        // Initialize O(1) L1 hot-expert indirection LUT to 0xFFFF across all 40 layers
        std::memset(host_slot_lut.ptr, 0xFF, host_slot_lut.size);
        const distributed::MeshCoordinateRange init_devs(mesh->shape());
        auto& cq = mesh->mesh_command_queue();
        cq.enqueue_write_shard_to_sub_grid(
            l1_slot_lut->mesh_buffer(), host_slot_lut.ptr, init_devs, false, BufferRegion(0, host_slot_lut.size));

        // Pre-allocate persistent output tensors at startup (typecast -> BF16 all_reduce cuts Fabric traffic 2x!)
        auto run_red_decode = [&]() {
            auto keep = ws1->finish_decode;
            (void)keep;
            ws1->finish_decode[0] = ttnn::typecast(ws1->combined_f32, DataType::BFLOAT16, ttnn::L1_MEMORY_CONFIG);
            ws1->finish_decode[1] = ttnn::all_reduce(
                ws1->finish_decode[0], 1, std::nullopt, ttnn::L1_MEMORY_CONFIG, 1, tt::tt_fabric::Topology::Linear);
            ws1->finish_decode[2] = ttnn::untilize(ws1->finish_decode[1], ws1->finish_untilize_mem_cfg());
        };
        auto run_red_arena = [&](CompactPrefillArena& ar) {
            auto keep = ar.finish_tensors;
            (void)keep;
            ar.finish_tensors[0] = ttnn::typecast(ar.y_accum_f32, DataType::BFLOAT16, ar.finish_mem_cfg());
            ar.finish_tensors[1] = ttnn::all_reduce(
                ar.finish_tensors[0], 1, std::nullopt, ar.finish_mem_cfg(), 1, tt::tt_fabric::Topology::Linear);
        };

        run_red_decode();
        run_red_arena(*arena128);
        run_red_arena(*arena512);
        cq.finish();

        p150_l1_geometry_report();
        std::printf("[P150_DSV41] Opened 1x2 Blackhole P150A Mesh (grid=%ux%u=%u cores, FABRIC_1D, hot_shards=%u/card [%u TP2 shards = %.2f MiB L1], requested=%u, demotion=%d, total_L1/card=%.2f MiB, trace=%d)\n",
                    unsigned(grid.x), unsigned(grid.y), total_cores,
                    num_hot_shards, num_hot_shards * 2u,
                    double(pinned_hot_l1_bytes * 2ULL) / (1024.0 * 1024.0),
                    unsigned(requested_hot_shards), int(expanded_l1_demotion),
                    double(workspace_l1_bytes_per_card()) / (1024.0 * 1024.0),
                    int(enable_trace));
        std::fflush(stdout);
    }

    ~Impl() {
        if (decode_layer_calls > 0ULL) {
            save_expert_hist_files(current_hot_assignments);
        }
        if (mesh) {
            mesh->mesh_command_queue().finish();
            for (auto& [k, tr] : op_traces) {
                if (tr.id) mesh->release_mesh_trace(*tr.id);
            }
            for (auto& tr : reduce_traces) {
                if (tr.id) mesh->release_mesh_trace(*tr.id);
            }
            hot_down_l1.reset();
            hot_up_l1.reset();
            hot_gate_l1.reset();
            arena512.reset();
            arena128.reset();
            dummy_hot_l1.reset();
            l1_slot_lut.reset();
            slot_meta_l1.reset();
            x_rm_l1.reset();
            ws1.reset();
            for (auto& l : layers) l.reset();
            mesh->mesh_command_queue().finish();
            ttnn::distributed::close_mesh_device(mesh);
        }
    }

    size_t workspace_l1_bytes_per_card() const {
        size_t bytes = ws1->l1_bytes_per_card() +
                       l1_slot_lut->buffer()->size() +
                       arena128->l1_bytes_per_card() +
                       arena512->l1_bytes_per_card() +
                       pinned_hot_l1_bytes;
        if (!expanded_l1_demotion) {
            bytes += slot_meta_l1->buffer()->size() + dummy_hot_l1->buffer()->size();
        }
        if (!x_rm_dram) bytes += x_rm_l1->buffer()->size();
        return bytes;
    }

    CoreCoord core_at(uint32_t idx) const {
        return CoreCoord{idx % grid.x, idx / grid.x};
    }

    static bool is_hot_expert(const ResidentLayerBank& bank, uint16_t eid) noexcept {
        return eid < DSV41_P150_NUM_EXPERTS && bank.hot_slot_for_eid[eid] != 0xFFFFu;
    }

    void write_layer_desc_to_slot_ids(const ResidentLayerBank& bank) {
        auto* ldesc = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(host_slot_ids.ptr) + 576u);
        ldesc[0] = bank.gate_pages.buffer()->address();
        ldesc[1] = bank.up_pages.buffer()->address();
        ldesc[2] = bank.down_pages.buffer()->address();
        ldesc[3] = bank.centroids.buffer()->address();
        ldesc[4] = bank.wht_q_i.buffer()->address();
        ldesc[5] = hot_gate_l1.has_value() ? hot_gate_l1->buffer()->address() : dummy_hot_l1->buffer()->address();
        ldesc[6] = hot_up_l1.has_value()   ? hot_up_l1->buffer()->address()   : dummy_hot_l1->buffer()->address();
        ldesc[7] = hot_down_l1.has_value() ? hot_down_l1->buffer()->address() : dummy_hot_l1->buffer()->address();
        ldesc[8] = l1_slot_lut->buffer()->address();
        ldesc[9] = bank.layer;
        for (uint32_t k = 10u; k < 16u; ++k) {
            ldesc[k] = 0u;
        }
    }

    void apply_hot_assignments(const std::vector<HotAssignment>& requested) {
        if (num_hot_shards == 0u || !hot_gate_l1.has_value() || !l1_slot_lut.has_value()) return;
        constexpr size_t kExpertBytes = 360ULL * 4352ULL;

        std::vector<HotAssignment> valid;
        valid.reserve(num_hot_shards);
        for (const auto& a : requested) {
            if (valid.size() >= num_hot_shards) break;
            if (a.layer >= DSV41_P150_NUM_LAYERS || a.eid >= DSV41_P150_NUM_EXPERTS) continue;
            if (!layers[a.layer].has_value()) continue;
            bool dup = false;
            for (const auto& v : valid) {
                if (v == a) { dup = true; break; }
            }
            if (dup) continue;
            valid.push_back(a);
        }
        if (valid.empty() || valid == current_hot_assignments) return;

        const size_t arena_bytes = size_t(num_hot_shards) * kExpertBytes;
        std::vector<uint8_t> stage0(arena_bytes, 0u);
        std::vector<uint8_t> stage1(arena_bytes, 0u);

        mesh->mesh_command_queue().finish();
        for (int proj = 0; proj < 3; ++proj) {
            for (size_t k = 0; k < valid.size(); ++k) {
                const auto& bank = *layers[valid[k].layer];
                const uint16_t e = valid[k].eid;
                const uint8_t* s0 = (proj == 0) ? bank.gp_exp[0][e] : ((proj == 1) ? bank.up_exp[0][e] : bank.dp_exp[0][e]);
                const uint8_t* s1 = (proj == 0) ? bank.gp_exp[1][e] : ((proj == 1) ? bank.up_exp[1][e] : bank.dp_exp[1][e]);
                std::memcpy(stage0.data() + k * kExpertBytes, s0, kExpertBytes);
                std::memcpy(stage1.data() + k * kExpertBytes, s1, kExpertBytes);
            }
            ttnn::Tensor& dst = (proj == 0) ? *hot_gate_l1 : ((proj == 1) ? *hot_up_l1 : *hot_down_l1);
            write_sharded_pair_blocking(*mesh, dst, stage0.data(), stage1.data());
        }

        for (auto& l_opt : layers) {
            if (l_opt.has_value()) l_opt->hot_slot_for_eid.fill(0xFFFFu);
        }
        std::memset(host_slot_lut.ptr, 0xFF, host_slot_lut.size);
        auto* lut_u16 = static_cast<uint16_t*>(host_slot_lut.ptr);
        for (size_t k = 0; k < valid.size(); ++k) {
            const uint16_t l = valid[k].layer;
            const uint16_t e = valid[k].eid;
            layers[l]->hot_slot_for_eid[e] = static_cast<uint16_t>(k);
            lut_u16[size_t(l) * 256u + size_t(e)] = static_cast<uint16_t>(k);
        }
        const distributed::MeshCoordinateRange all_devs(mesh->shape());
        mesh->mesh_command_queue().enqueue_write_shard_to_sub_grid(
            l1_slot_lut->mesh_buffer(), host_slot_lut.ptr, all_devs, true, BufferRegion(0, host_slot_lut.size));
        current_hot_assignments = valid;

        std::printf("[P150_HOT_L1] Pinned %zu/%u hot (layer, local_eid -> global_eid) pairs via O(1) l1_slot_lut (%.2f MiB total TP2 L1):",
                    valid.size(), num_hot_shards, double(valid.size() * kExpertBytes * 6ULL) / (1024.0 * 1024.0));
        for (const auto& a : valid) {
            std::printf(" (L%02u,e%u->%u)", unsigned(a.layer), unsigned(a.eid), unsigned(DSV41_P150_EXPERT_BASE + a.eid));
        }
        std::printf("\n");
        std::fflush(stdout);
    }

    std::vector<HotAssignment> rank_top_decode_experts() const {
        struct Cand { uint64_t hits; uint16_t layer; uint16_t eid; };
        std::vector<Cand> cands;
        cands.reserve(DSV41_P150_NUM_LAYERS * DSV41_P150_NUM_EXPERTS);
        for (uint32_t l = 0; l < DSV41_P150_NUM_LAYERS; ++l) {
            if (!layers[l].has_value()) continue;
            for (uint32_t e = 0; e < DSV41_P150_NUM_EXPERTS; ++e) {
                const uint64_t h = decode_expert_hits[l][e] * 4ULL + prefill_expert_hits[l][e];
                if (h > 0ULL) cands.push_back(Cand{h, static_cast<uint16_t>(l), static_cast<uint16_t>(e)});
            }
        }
        std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
            if (a.hits != b.hits) return a.hits > b.hits;
            if (a.layer != b.layer) return a.layer > b.layer;
            return a.eid < b.eid;
        });
        std::vector<HotAssignment> out;
        out.reserve(num_hot_shards);
        for (const auto& c : cands) {
            if (out.size() >= num_hot_shards) break;
            out.push_back(HotAssignment{c.layer, c.eid});
        }
        return out;
    }

    void save_expert_hist_files(const std::vector<HotAssignment>& hot_list) const {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(hot_experts_path).parent_path(), ec);
        if (!hot_list.empty()) {
            std::ofstream hf(hot_experts_path, std::ios::trunc);
            for (const auto& a : hot_list) {
                hf << a.layer << " " << a.eid << " " << decode_expert_hits[a.layer][a.eid]
                   << " " << prefill_expert_hits[a.layer][a.eid] << "\n";
            }
        }
        std::ofstream pf(expert_hist_path, std::ios::trunc);
        const double hit_pct = (decode_total_p150_slots > 0ULL)
            ? (100.0 * double(decode_hot_l1_hits) / double(decode_total_p150_slots)) : 0.0;
        pf << "# P150A Router Activation Histogram (decode_layer_calls=" << decode_layer_calls
           << " total_decode_p150_slots=" << decode_total_p150_slots
           << " hot_l1_hits=" << decode_hot_l1_hits
           << " hot_hit_pct=" << hit_pct << "%)\n";
        for (size_t k = 0; k < hot_list.size(); ++k) {
            const auto& a = hot_list[k];
            pf << "PINNED[" << k << "] layer=" << a.layer
               << " local_eid=" << a.eid << " global_eid=" << (DSV41_P150_EXPERT_BASE + a.eid)
               << " decode_hits=" << decode_expert_hits[a.layer][a.eid]
               << " prefill_slots=" << prefill_expert_hits[a.layer][a.eid] << "\n";
        }
    }

    void seed_initial_hot_experts() {
        if (num_hot_shards == 0u) return;
        std::vector<HotAssignment> seed;
        std::ifstream hf(hot_experts_path);
        if (hf.is_open()) {
            uint32_t l = 0, e = 0;
            uint64_t dh = 0, ph = 0;
            while (seed.size() < num_hot_shards && (hf >> l >> e >> dh >> ph)) {
                if (l < DSV41_P150_NUM_LAYERS && e < DSV41_P150_NUM_EXPERTS && layers[l].has_value()) {
                    seed.push_back(HotAssignment{static_cast<uint16_t>(l), static_cast<uint16_t>(e)});
                }
            }
        }
        // Fallback cold-start seed: decoder layers 20..39 expert 0 until live auto-promotion fires at 10 decode steps!
        for (uint32_t l = 20u; l < DSV41_P150_NUM_LAYERS && seed.size() < num_hot_shards; ++l) {
            if (layers[l].has_value()) seed.push_back(HotAssignment{static_cast<uint16_t>(l), 0u});
        }
        for (uint32_t l = 1u; l < 20u && seed.size() < num_hot_shards; ++l) {
            if (layers[l].has_value()) seed.push_back(HotAssignment{static_cast<uint16_t>(l), 0u});
        }
        apply_hot_assignments(seed);
    }

    void capture_reduce_traces_once() {
        if (!enable_trace || reduce_traces[0].id.has_value()) return;
        TraceStorageContextGuard trace_alloc_guard;
        auto& cq = mesh->mesh_command_queue();
        cq.finish();
        {
            for (auto& t : ws1->finish_decode) t = ttnn::Tensor();
            cq.finish();
            const auto tid = mesh->begin_mesh_trace(cq);
            ws1->finish_decode[0] = ttnn::typecast(ws1->combined_f32, DataType::BFLOAT16, ttnn::L1_MEMORY_CONFIG);
            ws1->finish_decode[1] = ttnn::all_reduce(
                ws1->finish_decode[0], 1, std::nullopt, ttnn::L1_MEMORY_CONFIG, 1, tt::tt_fabric::Topology::Linear);
            ws1->finish_decode[2] = ttnn::untilize(ws1->finish_decode[1], ws1->finish_untilize_mem_cfg());
            mesh->end_mesh_trace(cq, tid);
            reduce_traces[0].id = tid;
            reduce_traces[0].warmed = true;
        }
        auto cap_arena = [&](CompactPrefillArena& ar, TraceSlot& slot) {
            for (auto& t : ar.finish_tensors) t = ttnn::Tensor();
            cq.finish();
            const auto tid = mesh->begin_mesh_trace(cq);
            ar.finish_tensors[0] = ttnn::typecast(ar.y_accum_f32, DataType::BFLOAT16, ar.finish_mem_cfg());
            ar.finish_tensors[1] = ttnn::all_reduce(
                ar.finish_tensors[0], 1, std::nullopt, ar.finish_mem_cfg(), 1, tt::tt_fabric::Topology::Linear);
            mesh->end_mesh_trace(cq, tid);
            slot.id = tid;
            slot.warmed = true;
        };
        cap_arena(*arena128, reduce_traces[1]);
        cap_arena(*arena512, reduce_traces[2]);
        cq.finish();
    }

    void maybe_promote_hot_experts() {
        if (num_hot_shards == 0u) return;
        if (decode_layer_calls != 1560ULL && (decode_layer_calls % 15600ULL) != 0ULL) {
            return;
        }
        const auto ranked = rank_top_decode_experts();
        if (ranked.empty()) return;
        uint64_t cur_cov = 0ULL, new_cov = 0ULL;
        for (const auto& a : current_hot_assignments) cur_cov += decode_expert_hits[a.layer][a.eid];
        for (const auto& a : ranked)                  new_cov += decode_expert_hits[a.layer][a.eid];
        if (current_hot_assignments.empty() || new_cov > cur_cov + (cur_cov / 7ULL)) {
            apply_hot_assignments(ranked);
        }
        save_expert_hist_files(current_hot_assignments);
    }

    void prewarm_all_kernels_untraced(const ResidentLayerBank& bank) {
        if (programs_prewarmed) return;
        programs_prewarmed = true;

        auto* slot_u16 = static_cast<uint16_t*>(host_slot_ids.ptr);
        for (uint32_t i = 0; i < 288u; ++i) slot_u16[i] = (i == 0u) ? 0u : 0xFFFFu;
        write_layer_desc_to_slot_ids(bank);
        std::memset(host_slot_meta.ptr, 0, 288u * 128u);
        std::memset(host_weights.ptr, 0, kBatchSlots * 4096u);
        std::memset(host_x_tilized.ptr, 0, 160u * 2048u);

        auto& cq = mesh->mesh_command_queue();
        const distributed::MeshCoordinateRange all_devs(mesh->shape());
        cq.enqueue_write_shard_to_sub_grid(ws1->slot_ids.mesh_buffer(), host_slot_ids.ptr, all_devs, false, BufferRegion(0, kSlotIdsBytes));
        cq.enqueue_write_shard_to_sub_grid(slot_meta_l1->mesh_buffer(), host_slot_meta.ptr, all_devs, false, BufferRegion(0, kBatchSlots * 128u));
        cq.enqueue_write_shard_to_sub_grid(ws1->weights_tiled.mesh_buffer(), host_weights.ptr, all_devs, false, BufferRegion(0, kBatchSlots * 4096u));
        cq.enqueue_write_shard_to_sub_grid(ws1->x_decode.mesh_buffer(), host_x_tilized.ptr, all_devs, false, BufferRegion(0, 160u * 2048u));
        if (ws1->x_decode.buffer()->address() != ws1->x_tilized.buffer()->address() || ws1->x_tilized_dram != ws1->x_decode_dram) {
            cq.enqueue_write_shard_to_sub_grid(ws1->x_tilized.mesh_buffer(), host_x_tilized.ptr, all_devs, false, BufferRegion(0, 160u * 2048u));
        }

        // Prewarm multi-slot (bcnt = 4, 8, 12, 16) decode + compact kernels before any trace capture!
        for (uint32_t bcnt : {4u, 8u, 12u, kBatchSlots}) {
            enqueue_hlut2_projection(bank, ws1->x_decode, bank.gate_pages, ws1->up_partials, 0u, bcnt, 0u, 3u);
            enqueue_swiglu_wht128(bank, bcnt);
            enqueue_hlut2_projection(bank, ws1->z_rot_bf16, bank.down_pages, ws1->down_partials, 0u, bcnt, 36u, 2u);
            enqueue_ksplit_combine_decode(0u, bcnt);
        }
        enqueue_hlut2_projection(bank, ws1->x_tilized, bank.gate_pages, ws1->up_partials, 0u, kBatchSlots, 160u, 3u);

        enqueue_zero_pages(arena128->y_accum_f32, 128u * 5u);
        enqueue_zero_pages(arena512->y_accum_f32, 512u * 5u);
        enqueue_compact_gather_tilize(0u, kBatchSlots);
        enqueue_compact_scatter_combine(arena128->y_accum_f32, 0u, kBatchSlots);
        enqueue_compact_scatter_combine(arena512->y_accum_f32, 0u, kBatchSlots);
        cq.finish();
    }

    void merge_p150_and_tail_to_bf16(
        uint32_t chunk_rows,
        uint32_t total_tail_pairs,
        const std::vector<TailExpertDesc>& tail_descs,
        uint16_t* y_chunk_bf16)
    {
        std::array<int32_t, DSV41_P150_MAX_ROWS> row_head;
        std::fill_n(row_head.data(), chunk_rows, -1);
        std::array<int32_t, 1024> pair_next;
        for (const auto& d : tail_descs) {
            for (uint32_t k = 0; k < d.cnt; ++k) {
                const uint32_t p = d.pair_offset + k;
                const uint16_t r = d.rows[k];
                pair_next[p] = row_head[r];
                row_head[r] = static_cast<int32_t>(p);
            }
        }
        const auto* out_rm_u16 = static_cast<const uint16_t*>(host_out_rm.ptr);
        const auto* tp0_pairs  = static_cast<const float*>(host_tail_pairs_f32.ptr);
        const auto* tp1_pairs  = tp0_pairs + size_t(total_tail_pairs) * DSV41_P150_H;

        avx_pool.parallel_rows(chunk_rows, [&](uint32_t r0, uint32_t r1) {
            alignas(32) float acc[DSV41_P150_H];
            for (uint32_t r = r0; r < r1; ++r) {
                const uint16_t* src_bf16 = out_rm_u16 + size_t(r) * DSV41_P150_H;
                uint16_t* dst_bf16       = y_chunk_bf16 + size_t(r) * DSV41_P150_H;
                int32_t p = row_head[r];
                if (p < 0) {
                    std::memcpy(dst_bf16, src_bf16, DSV41_P150_H * sizeof(uint16_t));
                    continue;
                }
                bf16_row_to_f32_avx2(src_bf16, acc);
                while (p >= 0) {
                    const float* a0 = tp0_pairs + size_t(uint32_t(p)) * DSV41_P150_H;
                    const float* a1 = tp1_pairs + size_t(uint32_t(p)) * DSV41_P150_H;
                    for (uint32_t j = 0; j < DSV41_P150_H; j += 8u) {
                        __m256 v = _mm256_load_ps(acc + j);
                        v = _mm256_add_ps(v, _mm256_add_ps(_mm256_loadu_ps(a0 + j), _mm256_loadu_ps(a1 + j)));
                        _mm256_store_ps(acc + j, v);
                    }
                    p = pair_next[uint32_t(p)];
                }
                f32_row_to_bf16_avx2(acc, dst_bf16);
            }
        });
    }

    bool ensure_layer_loaded(uint32_t layer) {
        if (layer >= DSV41_P150_NUM_LAYERS) return false;
        if (layers[layer].has_value()) return true;

        char tag[32];
        std::snprintf(tag, sizeof(tag), "layer_%02u.safetensors", layer);
        const std::string p0 = packed_root + "/tp0/" + tag;
        const std::string p1 = packed_root + "/tp1/" + tag;
        if (!std::filesystem::is_regular_file(p0) || !std::filesystem::is_regular_file(p1)) {
            return false;
        }

        const auto t0 = std::chrono::steady_clock::now();
        ResidentLayerBank bank;
        bank.layer = layer;
        bank.tp0_path = p0;
        bank.tp1_path = p1;
        bank.st_mmap[0] = std::make_unique<SafetensorsMmap>(p0);
        bank.st_mmap[1] = std::make_unique<SafetensorsMmap>(p1);
        const SafetensorsMmap& st0 = *bank.st_mmap[0];
        const SafetensorsMmap& st1 = *bank.st_mmap[1];

        constexpr size_t kExpertBytes = 360ULL * 4352ULL;
        for (int tp = 0; tp < 2; ++tp) {
            const SafetensorsMmap& st = *bank.st_mmap[tp];
            bank.gp_mmap[tp] = st.tensor_bytes("hlut2.gate_pages");
            bank.up_mmap[tp] = st.tensor_bytes("hlut2.up_pages");
            bank.dp_mmap[tp] = st.tensor_bytes("hlut2.down_pages");
            bank.gc_mmap[tp] = reinterpret_cast<const float*>(st.tensor_bytes("hlut2.gate_centroids"));
            bank.uc_mmap[tp] = reinterpret_cast<const float*>(st.tensor_bytes("hlut2.up_centroids"));
            bank.dc_mmap[tp] = reinterpret_cast<const float*>(st.tensor_bytes("hlut2.down_centroids"));
            const auto* si = reinterpret_cast<const uint64_t*>(st.tensor_bytes("wht.signs_i"));
            std::memcpy(bank.signs_i_mmap[tp].data(), si, 18u * sizeof(uint64_t));
            for (uint32_t e = 0; e < DSV41_P150_NUM_EXPERTS; ++e) {
                bank.gp_exp[tp][e] = bank.gp_mmap[tp] + size_t(e) * kExpertBytes;
                bank.up_exp[tp][e] = bank.up_mmap[tp] + size_t(e) * kExpertBytes;
                bank.dp_exp[tp][e] = bank.dp_mmap[tp] + size_t(e) * kExpertBytes;
                bank.gc_act[tp][2u * e + 0u] = bank.gc_mmap[tp][2u * e + 0u];
                bank.gc_act[tp][2u * e + 1u] = bank.gc_mmap[tp][2u * e + 1u];
                bank.uc_act[tp][2u * e + 0u] = bank.uc_mmap[tp][2u * e + 0u];
                bank.uc_act[tp][2u * e + 1u] = bank.uc_mmap[tp][2u * e + 1u];
                bank.dc_act[tp][2u * e + 0u] = bank.dc_mmap[tp][2u * e + 0u];
                bank.dc_act[tp][2u * e + 1u] = bank.dc_mmap[tp][2u * e + 1u];
            }
        }

        std::vector<uint32_t> ov_eids;
        if (rarebal_overlay_mmap) {
            for (uint32_t e = 0; e < DSV41_P150_NUM_EXPERTS; ++e) {
                const std::string base = "p150.l" + std::to_string(layer) + ".e" + std::to_string(e) + ".";
                if (!rarebal_overlay_mmap->contains(base + "tp0.gate_pages")) continue;
                for (int tp = 0; tp < 2; ++tp) {
                    const std::string pfx = base + "tp" + std::to_string(tp) + ".";
                    bank.gp_exp[tp][e] = rarebal_overlay_mmap->tensor_bytes(pfx + "gate_pages");
                    bank.up_exp[tp][e] = rarebal_overlay_mmap->tensor_bytes(pfx + "up_pages");
                    bank.dp_exp[tp][e] = rarebal_overlay_mmap->tensor_bytes(pfx + "down_pages");
                    const auto* gc = reinterpret_cast<const float*>(rarebal_overlay_mmap->tensor_bytes(pfx + "gate_centroids"));
                    const auto* uc = reinterpret_cast<const float*>(rarebal_overlay_mmap->tensor_bytes(pfx + "up_centroids"));
                    const auto* dc = reinterpret_cast<const float*>(rarebal_overlay_mmap->tensor_bytes(pfx + "down_centroids"));
                    bank.gc_act[tp][2u * e + 0u] = gc[0]; bank.gc_act[tp][2u * e + 1u] = gc[1];
                    bank.uc_act[tp][2u * e + 0u] = uc[0]; bank.uc_act[tp][2u * e + 1u] = uc[1];
                    bank.dc_act[tp][2u * e + 0u] = dc[0]; bank.dc_act[tp][2u * e + 1u] = dc[1];
                }
                ov_eids.push_back(e);
                ++rarebal_p150_patched;
            }
        }

        const auto* sh_ptr = reinterpret_cast<const uint64_t*>(st0.tensor_bytes("wht.signs_h"));
        bank.signs_h.assign(sh_ptr, sh_ptr + 80);

        constexpr uint32_t kNumPagesPerCard = DSV41_P150_NUM_EXPERTS * 360u;
        constexpr size_t kExpectedPageBytes = size_t(kNumPagesPerCard) * 4352ULL;
        if (st0.at("hlut2.gate_pages").byte_size() != kExpectedPageBytes ||
            st1.at("hlut2.gate_pages").byte_size() != kExpectedPageBytes) {
            throw std::runtime_error("Unexpected hlut2.gate_pages size in " + p0);
        }

        bank.gate_pages = alloc_dram_sharded0(*mesh, Shape({1, kNumPagesPerCard, 1088}), DataType::UINT32, Layout::ROW_MAJOR);
        bank.up_pages   = alloc_dram_sharded0(*mesh, Shape({1, kNumPagesPerCard, 1088}), DataType::UINT32, Layout::ROW_MAJOR);
        bank.down_pages = alloc_dram_sharded0(*mesh, Shape({1, kNumPagesPerCard, 1088}), DataType::UINT32, Layout::ROW_MAJOR);

        if (ov_eids.empty()) {
            write_sharded_pair_blocking(*mesh, bank.gate_pages, bank.gp_mmap[0], bank.gp_mmap[1]);
            write_sharded_pair_blocking(*mesh, bank.up_pages,   bank.up_mmap[0], bank.up_mmap[1]);
            write_sharded_pair_blocking(*mesh, bank.down_pages, bank.dp_mmap[0], bank.dp_mmap[1]);
        } else {
            std::vector<uint8_t> stage0(kExpectedPageBytes), stage1(kExpectedPageBytes);
            for (int proj = 0; proj < 3; ++proj) {
                const uint8_t* b0 = (proj == 0) ? bank.gp_mmap[0] : ((proj == 1) ? bank.up_mmap[0] : bank.dp_mmap[0]);
                const uint8_t* b1 = (proj == 0) ? bank.gp_mmap[1] : ((proj == 1) ? bank.up_mmap[1] : bank.dp_mmap[1]);
                std::memcpy(stage0.data(), b0, kExpectedPageBytes);
                std::memcpy(stage1.data(), b1, kExpectedPageBytes);
                for (uint32_t e : ov_eids) {
                    const uint8_t* o0 = (proj == 0) ? bank.gp_exp[0][e] : ((proj == 1) ? bank.up_exp[0][e] : bank.dp_exp[0][e]);
                    const uint8_t* o1 = (proj == 0) ? bank.gp_exp[1][e] : ((proj == 1) ? bank.up_exp[1][e] : bank.dp_exp[1][e]);
                    std::memcpy(stage0.data() + size_t(e) * kExpertBytes, o0, kExpertBytes);
                    std::memcpy(stage1.data() + size_t(e) * kExpertBytes, o1, kExpertBytes);
                }
                ttnn::Tensor& dst = (proj == 0) ? bank.gate_pages : ((proj == 1) ? bank.up_pages : bank.down_pages);
                write_sharded_pair_blocking(*mesh, dst, stage0.data(), stage1.data());
            }
        }

        // Precompute [gc0, gc1 - gc0, uc0, uc1 - uc0, dc0, dc1 - dc0] so writer_b needs ZERO FP32 math!
        std::array<std::vector<float>, 2> cent_host;
        for (int tp = 0; tp < 2; ++tp) {
            const float* gc = bank.gc_act[tp].data();
            const float* uc = bank.uc_act[tp].data();
            const float* dc = bank.dc_act[tp].data();
            cent_host[tp].assign(DSV41_P150_NUM_EXPERTS * 16u, 0.0f);
            for (uint32_t e = 0; e < DSV41_P150_NUM_EXPERTS; ++e) {
                float* row = cent_host[tp].data() + e * 16u;
                row[0] = gc[e * 2u + 0u];
                row[1] = gc[e * 2u + 1u] - row[0];
                row[2] = uc[e * 2u + 0u];
                row[3] = uc[e * 2u + 1u] - row[2];
                row[4] = dc[e * 2u + 0u];
                row[5] = dc[e * 2u + 1u] - row[4];
            }
        }
        bank.centroids = alloc_dram_sharded0(*mesh, Shape({1, DSV41_P150_NUM_EXPERTS, 16}), DataType::FLOAT32, Layout::ROW_MAJOR);
        write_sharded_pair_blocking(*mesh, bank.centroids, cent_host[0].data(), cent_host[1].data());

        // Build per-card tilized WHT128 Q_i [1, 9, 128, 128] BFLOAT16 TILE
        std::array<std::vector<uint16_t>, 2> wht_tiles;
        for (int tp = 0; tp < 2; ++tp) {
            const SafetensorsMmap& st = (tp == 0) ? st0 : st1;
            const auto* si_ptr = reinterpret_cast<const uint64_t*>(st.tensor_bytes("wht.signs_i"));
            const std::vector<uint64_t> signs_i_sub(si_ptr, si_ptr + 18);
            wht_tiles[tp].assign(9u * 16u * 1024u, 0u);
            float q_mat[128][128];
            for (int nb4 = 0; nb4 < 9; ++nb4) {
                build_wht128_block_matrix(nb4, signs_i_sub, q_mat);
                for (int kt = 0; kt < 4; ++kt) {
                    for (int nt = 0; nt < 4; ++nt) {
                        uint16_t* dst_tile = wht_tiles[tp].data() + ((nb4 * 16) + kt * 4 + nt) * 1024;
                        for (int r = 0; r < 32; ++r) {
                            for (int c = 0; c < 32; ++c) {
                                dst_tile[tensix_tile_offset(r, c)] = f32_to_bf16_rne(q_mat[kt * 32 + r][nt * 32 + c]);
                            }
                        }
                    }
                }
            }
        }
        bank.wht_q_i = alloc_dram_sharded0(*mesh, Shape({1, 9, 128, 128}), DataType::BFLOAT16, Layout::TILE);
        write_sharded_pair_blocking(*mesh, bank.wht_q_i, wht_tiles[0].data(), wht_tiles[1].data());

        layers[layer].emplace(std::move(bank));
        loaded_mask |= (1ULL << layer);
        prewarm_all_kernels_untraced(*layers[layer]);

        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("[P150_DSV41] Loaded layer_%02u (168 experts x TP2 = 753.6 MiB/card GDDR6) in %.2fs\n", layer, sec);
        std::fflush(stdout);
        return true;
    }

    void enqueue_hlut2_projection(
        const ResidentLayerBank& bank,
        const ttnn::Tensor& act_l1,
        const ttnn::Tensor& pages_dram,
        ttnn::Tensor& partials_l1,
        uint32_t slot_start,
        uint32_t batch_cnt,
        uint32_t act_slot_stride,
        uint32_t proj_idx)
    {
        const bool is_down      = (proj_idx == 2u);
        const bool is_fused_gu  = (proj_idx == 3u);
        const uint32_t num_proj = is_fused_gu ? 2u : 1u;
        const uint32_t keep_act = (act_slot_stride == 0u) ? 1u : 0u;
        const uint32_t nt4 = is_down ? 40u : 9u;
        const uint32_t kt4 = is_down ? 9u  : 40u;
        const uint32_t k_splits = is_down ? ws1->k_splits_down : ws1->k_splits_up;
        const uint32_t nt_total = nt4 * 4u;
        const uint32_t kt_total = kt4 * 4u;
        const ttnn::Tensor& hot_t = is_down
            ? (hot_down_l1.has_value() ? *hot_down_l1 : *dummy_hot_l1)
            : (hot_gate_l1.has_value() ? *hot_gate_l1 : *dummy_hot_l1);
        const ttnn::Tensor& partials0 = is_fused_gu ? ws1->gate_partials : partials_l1;

        struct CoreJob {
            CoreCoord coord;
            uint32_t nb4;
            uint32_t ks;
            uint32_t kb4_start;
            uint32_t num_sb;
        };
        std::vector<CoreJob> jobs;
        jobs.reserve(nt4 * k_splits);
        std::map<uint32_t, std::vector<CoreCoord>> sb_groups;
        std::vector<CoreCoord> all_coords;
        all_coords.reserve(nt4 * k_splits);

        uint32_t c_idx = 0;
        for (uint32_t nb4 = 0; nb4 < nt4; ++nb4) {
            uint32_t kb4_cur = 0;
            for (uint32_t ks = 0; ks < k_splits; ++ks) {
                const uint32_t sb_cnt = (kt4 / k_splits) + (ks < (kt4 % k_splits) ? 1u : 0u);
                const CoreCoord c = core_at(c_idx++);
                jobs.push_back(CoreJob{c, nb4, ks, kb4_cur, sb_cnt});
                all_coords.push_back(c);
                sb_groups[sb_cnt].push_back(c);
                kb4_cur += sb_cnt;
            }
        }

        const CoreRangeSet all_cores = make_core_set(all_coords);
        auto reader = make_kernel(
            kernels_dir + "/dsv41_hlut2_reader_a.cpp",
            all_cores,
            ReaderConfigDescriptor{},
            make_accessor_args({&act_l1}));
        auto writer = make_kernel(
            kernels_dir + "/dsv41_hlut2_writer_b.cpp",
            all_cores,
            WriterConfigDescriptor{},
            make_accessor_args({&pages_dram, &bank.centroids, &ws1->slot_ids, &partials0, &hot_t, &(*l1_slot_lut)}));

        for (const auto& j : jobs) {
            reader.runtime_args.emplace_back(
                j.coord,
                std::vector<uint32_t>{
                    act_l1.buffer()->address(),
                    kt_total,
                    j.kb4_start * 4u,
                    j.num_sb * 4u,
                    batch_cnt,
                    act_slot_stride});
            writer.runtime_args.emplace_back(
                j.coord,
                std::vector<uint32_t>{
                    ws1->slot_ids.buffer()->address(),
                    partials0.buffer()->address(),
                    slot_start,
                    proj_idx,
                    j.nb4,
                    kt4,
                    j.kb4_start,
                    j.num_sb,
                    j.ks,
                    nt_total,
                    ws1->m_tiles_ws,
                    kSlotIdsBytes,
                    batch_cnt,
                    partials_l1.buffer()->address()});
        }

        ProgramDescriptor prog;
        prog.kernels = {std::move(reader), std::move(writer)};

        for (const auto& [sb_cnt, coords] : sb_groups) {
            const CoreRangeSet grp_cores = make_core_set(coords);
            ComputeConfigDescriptor cfg;
            cfg.math_fidelity = MathFidelity::HiFi2;
            cfg.math_approx_mode = false;
            cfg.fp32_dest_acc_en = true;
            cfg.dst_full_sync_en = true;
            cfg.bfp8_pack_precise = false;
            cfg.unpack_to_dest_mode.resize(64, UnpackToDestMode::Default);
            cfg.unpack_to_dest_mode[4] = UnpackToDestMode::UnpackToDestFp32;
            cfg.unpack_to_dest_mode[5] = UnpackToDestMode::UnpackToDestFp32;

            auto compute = make_kernel(kernels_dir + "/dsv41_hlut2_compute.cpp", grp_cores, cfg);
            for (const auto& c : coords) {
                compute.runtime_args.emplace_back(c, std::vector<uint32_t>{sb_cnt, batch_cnt, num_proj, keep_act});
            }
            prog.kernels.push_back(std::move(compute));

            const uint32_t cb0_pages = (keep_act != 0u || is_down) ? (sb_cnt * 4u) : (2u * sb_cnt * 4u);
            prog.cbs.push_back(make_cb(0, cb0_pages, 2048u, tt::DataFormat::Float16_b, grp_cores));
            prog.cbs.push_back(make_cb(1, sb_cnt * 16u, 1088u, tt::DataFormat::Bfp8_b, grp_cores));
            prog.cbs.push_back(make_cb(4, 2u * sb_cnt, 4096u, tt::DataFormat::UInt32, grp_cores));
            prog.cbs.push_back(make_cb(5, 2u * sb_cnt, 4096u, tt::DataFormat::Float32, grp_cores));
            prog.cbs.push_back(make_cb(6, 1u, 4096u, tt::DataFormat::UInt32, grp_cores));
            prog.cbs.push_back(make_cb(16, 8u, 4096u, tt::DataFormat::Float32, grp_cores));
        }

        if (is_fused_gu) {
            const ttnn::Tensor& hot_up = hot_up_l1.has_value() ? *hot_up_l1 : *dummy_hot_l1;
            run_generic_preserving_topology({act_l1, ws1->slot_ids, pages_dram, bank.up_pages, bank.centroids, hot_t, hot_up, *l1_slot_lut, ws1->gate_partials, partials_l1}, prog);
        } else {
            run_generic_preserving_topology({act_l1, ws1->slot_ids, pages_dram, bank.centroids, hot_t, *l1_slot_lut, partials_l1}, prog);
        }
    }

    void enqueue_swiglu_wht128(const ResidentLayerBank& bank, uint32_t batch_cnt) {
        const uint32_t max_groups = total_cores / 9u;
        const uint32_t m_per_grp  = (batch_cnt + max_groups - 1u) / max_groups;
        const uint32_t ngroups    = (batch_cnt + m_per_grp - 1u) / m_per_grp;
        const uint32_t ncores     = 9u * ngroups;
        std::vector<CoreCoord> coords;
        coords.reserve(ncores);
        for (uint32_t idx = 0; idx < ncores; ++idx) coords.push_back(core_at(idx));
        const CoreRangeSet cores = make_core_set(coords);

        auto reader = make_kernel(
            kernels_dir + "/dsv41_swiglu_wht128_reader.cpp",
            cores,
            ReaderConfigDescriptor{},
            make_accessor_args({&ws1->gate_partials, &ws1->up_partials, &bank.wht_q_i, &ws1->slot_ids}));
        auto writer = make_kernel(
            kernels_dir + "/dsv41_swiglu_wht128_writer.cpp",
            cores,
            WriterConfigDescriptor{},
            make_accessor_args({&ws1->z_rot_bf16}));

        ComputeConfigDescriptor cfg;
        cfg.math_fidelity = MathFidelity::HiFi2;
        cfg.math_approx_mode = false;
        cfg.fp32_dest_acc_en = true;
        cfg.dst_full_sync_en = false;
        cfg.unpack_to_dest_mode.resize(64, UnpackToDestMode::Default);
        cfg.unpack_to_dest_mode[0] = UnpackToDestMode::UnpackToDestFp32;
        cfg.unpack_to_dest_mode[1] = UnpackToDestMode::UnpackToDestFp32;
        auto compute = make_kernel(kernels_dir + "/dsv41_swiglu_wht128_compute.cpp", cores, cfg);

        for (uint32_t g = 0; g < ngroups; ++g) {
            const uint32_t m_start = g * m_per_grp;
            const uint32_t m_cnt   = std::min<uint32_t>(m_per_grp, batch_cnt - m_start);
            for (uint32_t nb4 = 0; nb4 < 9u; ++nb4) {
                const CoreCoord& c = coords[g * 9u + nb4];
                reader.runtime_args.emplace_back(
                    c,
                    std::vector<uint32_t>{
                        ws1->gate_partials.buffer()->address(),
                        ws1->up_partials.buffer()->address(),
                        ws1->slot_ids.buffer()->address(),
                        nb4,
                        ws1->k_splits_up,
                        ws1->m_tiles_ws,
                        m_start,
                        m_cnt,
                        kSlotIdsBytes});
                writer.runtime_args.emplace_back(
                    c,
                    std::vector<uint32_t>{ws1->z_rot_bf16.buffer()->address(), nb4, m_start, m_cnt});
                compute.runtime_args.emplace_back(c, std::vector<uint32_t>{ws1->k_splits_up, m_cnt});
            }
        }

        ProgramDescriptor prog;
        prog.kernels = {std::move(reader), std::move(writer), std::move(compute)};
        prog.cbs = {
            make_cb(0, ws1->k_splits_up, 4096u, tt::DataFormat::Float32, cores),
            make_cb(1, ws1->k_splits_up, 4096u, tt::DataFormat::Float32, cores),
            make_cb(2, 16u, 2048u, tt::DataFormat::Float16_b, cores),
            make_cb(24, 4u, 2048u, tt::DataFormat::Float16_b, cores),
            make_cb(16, 8u, 2048u, tt::DataFormat::Float16_b, cores),
        };
        run_generic_preserving_topology({ws1->gate_partials, ws1->up_partials, bank.wht_q_i, ws1->slot_ids, ws1->z_rot_bf16}, prog);
    }

    void enqueue_ksplit_combine_decode(uint32_t slot_start, uint32_t batch_cnt) {
        constexpr uint32_t total_out_tiles = 160u;
        const uint32_t ncores = std::min(total_out_tiles, total_cores);
        const uint32_t acc_prev = (slot_start > 0u) ? 1u : 0u;

        std::vector<CoreCoord> coords;
        coords.reserve(ncores);
        for (uint32_t i = 0; i < ncores; ++i) coords.push_back(core_at(i));
        const CoreRangeSet cores = make_core_set(coords);

        auto reader = make_kernel(
            kernels_dir + "/dsv41_ksplit_combine_reader.cpp",
            cores,
            ReaderConfigDescriptor{},
            make_accessor_args({&ws1->down_partials, &ws1->weights_tiled, &ws1->combined_f32}));
        auto writer = make_kernel(
            kernels_dir + "/dsv41_ksplit_combine_writer.cpp",
            cores,
            WriterConfigDescriptor{},
            make_accessor_args({&ws1->combined_f32}));

        ComputeConfigDescriptor cfg;
        cfg.math_fidelity = MathFidelity::HiFi4;
        cfg.math_approx_mode = false;
        cfg.fp32_dest_acc_en = true;
        cfg.dst_full_sync_en = false;
        cfg.unpack_to_dest_mode.resize(64, UnpackToDestMode::Default);
        cfg.unpack_to_dest_mode[0] = UnpackToDestMode::UnpackToDestFp32;
        cfg.unpack_to_dest_mode[1] = UnpackToDestMode::UnpackToDestFp32;
        cfg.unpack_to_dest_mode[2] = UnpackToDestMode::UnpackToDestFp32;
        auto compute = make_kernel(kernels_dir + "/dsv41_ksplit_combine_compute.cpp", cores, cfg);

        uint32_t first_tile = 0;
        for (uint32_t i = 0; i < ncores; ++i) {
            const uint32_t cnt = (total_out_tiles / ncores) + (i < (total_out_tiles % ncores) ? 1u : 0u);
            reader.runtime_args.emplace_back(
                coords[i],
                std::vector<uint32_t>{
                    ws1->down_partials.buffer()->address(),
                    ws1->weights_tiled.buffer()->address(),
                    ws1->combined_f32.buffer()->address(),
                    slot_start,
                    ws1->k_splits_down,
                    ws1->m_tiles_ws,
                    first_tile,
                    cnt,
                    acc_prev,
                    batch_cnt});
            writer.runtime_args.emplace_back(
                coords[i],
                std::vector<uint32_t>{ws1->combined_f32.buffer()->address(), first_tile, cnt});
            compute.runtime_args.emplace_back(
                coords[i],
                std::vector<uint32_t>{ws1->k_splits_down, cnt, acc_prev, batch_cnt});
            first_tile += cnt;
        }

        ProgramDescriptor prog;
        prog.kernels = {std::move(reader), std::move(writer), std::move(compute)};
        prog.cbs = {
            make_cb(0, 2u, 4096u, tt::DataFormat::Float32, cores),
            make_cb(1, 2u, 4096u, tt::DataFormat::Float32, cores),
            make_cb(2, 2u, 4096u, tt::DataFormat::Float32, cores),
            make_cb(16, 2u, 4096u, tt::DataFormat::Float32, cores),
        };
        run_generic_preserving_topology({ws1->down_partials, ws1->weights_tiled, ws1->combined_f32}, prog);
    }

    void enqueue_zero_pages(ttnn::Tensor& y_accum, uint32_t total_pages) {
        const uint32_t ncores = total_cores;
        std::vector<CoreCoord> coords;
        coords.reserve(ncores);
        for (uint32_t i = 0; i < ncores; ++i) coords.push_back(core_at(i));
        const CoreRangeSet cores = make_core_set(coords);

        auto writer = make_kernel(
            kernels_dir + "/dsv41_zero_pages_writer.cpp",
            cores,
            WriterConfigDescriptor{},
            make_accessor_args({&y_accum}));

        uint32_t cur = 0;
        for (uint32_t i = 0; i < ncores; ++i) {
            const uint32_t cnt = (total_pages / ncores) + (i < (total_pages % ncores) ? 1u : 0u);
            writer.runtime_args.emplace_back(coords[i], std::vector<uint32_t>{y_accum.buffer()->address(), cur, cnt});
            cur += cnt;
        }

        ProgramDescriptor prog;
        prog.kernels = {std::move(writer)};
        prog.cbs = {make_cb(0, 1u, 4096u, tt::DataFormat::UInt32, cores)};
        run_generic_preserving_topology({*dummy_hot_l1, y_accum}, prog);
    }

    void enqueue_compact_gather_tilize(uint32_t slot_start, uint32_t batch_cnt) {
        constexpr uint32_t ncores = 80u;
        std::vector<CoreCoord> coords;
        coords.reserve(ncores);
        for (uint32_t i = 0; i < ncores; ++i) coords.push_back(core_at(i));
        const CoreRangeSet cores = make_core_set(coords);

        auto writer = make_kernel(
            kernels_dir + "/dsv41_compact_gather_tilize.cpp",
            cores,
            WriterConfigDescriptor{},
            make_accessor_args({&(*x_rm_l1), &(*slot_meta_l1), &ws1->x_tilized}));

        for (uint32_t i = 0; i < ncores; ++i) {
            writer.runtime_args.emplace_back(
                coords[i],
                std::vector<uint32_t>{
                    x_rm_l1->buffer()->address(),
                    slot_meta_l1->buffer()->address(),
                    ws1->x_tilized.buffer()->address(),
                    slot_start,
                    i,
                    batch_cnt});
        }

        ProgramDescriptor prog;
        prog.kernels = {std::move(writer)};
        prog.cbs = {make_cb(0, 1u, 10240u, tt::DataFormat::UInt32, cores)};
        run_generic_preserving_topology({*x_rm_l1, *slot_meta_l1, ws1->x_tilized}, prog);
    }

    void enqueue_compact_scatter_combine(ttnn::Tensor& y_accum, uint32_t slot_start, uint32_t batch_cnt) {
        constexpr uint32_t ncores = 80u;
        std::vector<CoreCoord> coords;
        coords.reserve(ncores);
        for (uint32_t i = 0; i < ncores; ++i) coords.push_back(core_at(i));
        const CoreRangeSet cores = make_core_set(coords);

        auto reader = make_kernel(
            kernels_dir + "/dsv41_compact_scatter_reader.cpp",
            cores,
            ReaderConfigDescriptor{},
            make_accessor_args({&ws1->down_partials, &ws1->weights_tiled, &y_accum, &(*slot_meta_l1)}));
        auto writer = make_kernel(
            kernels_dir + "/dsv41_compact_scatter_writer.cpp",
            cores,
            WriterConfigDescriptor{},
            make_accessor_args({&y_accum, &(*slot_meta_l1)}));

        ComputeConfigDescriptor cfg;
        cfg.math_fidelity = MathFidelity::HiFi4;
        cfg.math_approx_mode = false;
        cfg.fp32_dest_acc_en = true;
        cfg.dst_full_sync_en = false;
        cfg.unpack_to_dest_mode.resize(64, UnpackToDestMode::Default);
        cfg.unpack_to_dest_mode[0] = UnpackToDestMode::UnpackToDestFp32;
        cfg.unpack_to_dest_mode[1] = UnpackToDestMode::UnpackToDestFp32;
        cfg.unpack_to_dest_mode[2] = UnpackToDestMode::UnpackToDestFp32;
        auto compute = make_kernel(kernels_dir + "/dsv41_ksplit_combine_compute.cpp", cores, cfg);

        for (uint32_t i = 0; i < ncores; ++i) {
            reader.runtime_args.emplace_back(
                coords[i],
                std::vector<uint32_t>{
                    ws1->down_partials.buffer()->address(),
                    ws1->weights_tiled.buffer()->address(),
                    y_accum.buffer()->address(),
                    slot_meta_l1->buffer()->address(),
                    slot_start,
                    ws1->k_splits_down,
                    i,
                    ws1->m_tiles_ws,
                    batch_cnt});
            writer.runtime_args.emplace_back(
                coords[i],
                std::vector<uint32_t>{
                    y_accum.buffer()->address(),
                    slot_meta_l1->buffer()->address(),
                    slot_start,
                    i,
                    batch_cnt});
            compute.runtime_args.emplace_back(
                coords[i],
                std::vector<uint32_t>{ws1->k_splits_down, 2u * batch_cnt, 1u, 1u});
        }

        ProgramDescriptor prog;
        prog.kernels = {std::move(reader), std::move(writer), std::move(compute)};
        prog.cbs = {
            make_cb(0, 2u * ws1->k_splits_down, 4096u, tt::DataFormat::Float32, cores),
            make_cb(1, 2u, 4096u, tt::DataFormat::Float32, cores),
            make_cb(2, 2u, 4096u, tt::DataFormat::Float32, cores),
            make_cb(5, 1u, 8448u, tt::DataFormat::UInt32, cores),
            make_cb(6, 1u, 8448u, tt::DataFormat::UInt32, cores),
            make_cb(7, 16u, 64u,  tt::DataFormat::UInt32, cores),
            make_cb(16, 2u, 4096u, tt::DataFormat::Float32, cores),
        };
        run_generic_preserving_topology({ws1->down_partials, ws1->weights_tiled, *slot_meta_l1, y_accum}, prog);
    }

    template <typename ExecFn, typename PrepReduceFn, typename ReduceFn>
    void run_with_optional_trace(
        [[maybe_unused]] uint32_t layer,
        uint32_t mode,
        uint32_t scheduled_slots,
        uint64_t est_trace_bytes,
        ExecFn&& exec_fn,
        PrepReduceFn&& prep_reduce_fn,
        ReduceFn&& reduce_fn)
    {
        TraceStorageContextGuard trace_alloc_guard;
        auto& cq = mesh->mesh_command_queue();
        const bool trace_this = enable_trace &&
                                (mode <= 2u) &&
                                (op_traces.size() < 64u);
        TraceSlot* op_tr = trace_this ? &op_traces[{mode, scheduled_slots}] : nullptr;
        TraceSlot* red_tr = enable_trace ? &reduce_traces[mode] : nullptr;

        const bool red_ready = (!red_tr) || red_tr->id.has_value() || red_tr->skipped;
        if (op_tr && red_ready && op_tr->warmed && !op_tr->skipped) {
            if (!op_tr->id) {
                if (uint64_t(mesh->get_trace_buffers_size()) + est_trace_bytes + (16ULL << 20) <= uint64_t(trace_region_bytes)) {
                    cq.finish();
                    const auto tid = mesh->begin_mesh_trace(cq);
                    try {
                        exec_fn();
                        mesh->end_mesh_trace(cq, tid);
                    } catch (...) {
                        try { mesh->end_mesh_trace(cq, tid); } catch (...) {}
                        try { mesh->release_mesh_trace(tid); } catch (...) {}
                        throw;
                    }
                    op_tr->id = tid;
                } else {
                    op_tr->skipped = true;
                }
            }
            if (op_tr->id) {
                mesh->replay_mesh_trace(cq, *op_tr->id, false);
            } else {
                exec_fn();
            }
        } else {
            exec_fn();
            if (op_tr && red_ready) op_tr->warmed = true;
        }

        if (red_tr && red_tr->warmed && !red_tr->skipped) {
            if (!red_tr->id) {
                if (uint64_t(mesh->get_trace_buffers_size()) + (8ULL << 20) <= uint64_t(trace_region_bytes)) {
                    cq.finish();
                    prep_reduce_fn();
                    cq.finish();
                    const auto tid = mesh->begin_mesh_trace(cq);
                    try {
                        reduce_fn();
                        mesh->end_mesh_trace(cq, tid);
                    } catch (...) {
                        try { mesh->end_mesh_trace(cq, tid); } catch (...) {}
                        try { mesh->release_mesh_trace(tid); } catch (...) {}
                        throw;
                    }
                    red_tr->id = tid;
                } else {
                    red_tr->skipped = true;
                }
            }
            if (red_tr->id) {
                mesh->replay_mesh_trace(cq, *red_tr->id, false);
            } else {
                prep_reduce_fn();
                reduce_fn();
            }
        } else {
            prep_reduce_fn();
            reduce_fn();
            if (red_tr) red_tr->warmed = true;
        }
    }

    // Fast path for M = 1..32 (Decode & DSpark M = 6) with Hybrid CpuTailPool single-batch cap!
    void run_decode_small(
        const ResidentLayerBank& bank,
        uint32_t chunk_rows,
        bool input_is_wht_rotated,
        const int32_t row_slots[][DSV41_P150_MAX_SLOTS],
        const float row_weights[][DSV41_P150_MAX_SLOTS],
        const uint16_t* x_chunk_bf16,
        uint16_t* y_chunk_bf16)
    {
        struct SmallPair { uint16_t row; float weight; };
        std::array<SmallPair, 32> exp_pairs[DSV41_P150_NUM_EXPERTS];
        std::array<uint8_t, DSV41_P150_NUM_EXPERTS> exp_cnt{};
        std::vector<uint16_t> active_experts;
        active_experts.reserve(48);

        for (uint32_t r = 0; r < chunk_rows; ++r) {
            for (uint32_t s = 0; s < DSV41_P150_MAX_SLOTS; ++s) {
                const int32_t eid = row_slots[r][s];
                const float w = row_weights[r][s];
                if (eid >= 0 && uint32_t(eid) < DSV41_P150_NUM_EXPERTS && w != 0.0f) {
                    uint8_t& c = exp_cnt[uint32_t(eid)];
                    if (c == 0u) {
                        active_experts.push_back(static_cast<uint16_t>(eid));
                        ++decode_expert_hits[bank.layer][uint32_t(eid)];
                    }
                    if (c < 32u) exp_pairs[uint32_t(eid)][c++] = SmallPair{static_cast<uint16_t>(r), w};
                }
            }
        }
        ++decode_layer_calls;

        if (active_experts.empty()) {
            std::memset(y_chunk_bf16, 0, size_t(chunk_rows) * DSV41_P150_H * sizeof(uint16_t));
            return;
        }

        std::vector<TailExpertDesc> tail_descs;
        uint32_t total_tail_pairs = 0u;
        uint32_t p150_count = static_cast<uint32_t>(active_experts.size());
        // Wave 7A: M=2..8 DSpark verify (M=6 chain & M=8 tree) keeps <= 32 experts inside P150A MeshTrace;
        // only non-hot overflow (p150_count > 32) is peeled asynchronously to the singleton CpuTailPool so the
        // 28-worker DDR4 peel overlaps P150A rather than stalls it. M=1 and M>8 stay 100% on-chip.
        const uint32_t target_cap = (chunk_rows > 1u && chunk_rows <= 8u) ? 32u : 0u;
        if (target_cap > 0u && p150_count > target_cap) {
            uint32_t need_peel = p150_count - target_cap;
            std::array<bool, DSV41_P150_NUM_EXPERTS> peeled{};
            tail_descs.reserve(need_peel);
            for (uint8_t max_c = 1u; max_c <= 2u && need_peel > 0u; ++max_c) {
                for (uint16_t eid : active_experts) {
                    if (need_peel == 0u) break;
                    if (is_hot_expert(bank, eid)) continue;
                    const uint8_t cnt = exp_cnt[eid];
                    if (!peeled[eid] && cnt == max_c && total_tail_pairs + cnt <= 1024u) {
                        TailExpertDesc td{};
                        td.eid = eid;
                        td.cnt = cnt;
                        td.pair_offset = total_tail_pairs;
                        for (uint8_t k = 0u; k < cnt; ++k) {
                            td.rows[k] = exp_pairs[eid][k].row;
                            td.weights[k] = exp_pairs[eid][k].weight;
                        }
                        total_tail_pairs += cnt;
                        peeled[eid] = true;
                        tail_descs.push_back(td);
                        --need_peel;
                    }
                }
            }
            if (!tail_descs.empty()) {
                std::vector<uint16_t> kept;
                kept.reserve(active_experts.size() - tail_descs.size());
                for (uint16_t eid : active_experts) {
                    if (!peeled[eid]) kept.push_back(eid);
                }
                active_experts.swap(kept);
                p150_count = static_cast<uint32_t>(active_experts.size());
            }
        }
        decode_total_p150_slots += p150_count;
        for (uint16_t eid : active_experts) {
            if (is_hot_expert(bank, eid)) ++decode_hot_l1_hits;
        }
        const bool use_tail = !tail_descs.empty();
        auto* x_tiles   = static_cast<uint16_t*>(host_x_tilized.ptr);
        auto* x_rot_f32 = static_cast<float*>(host_x_rot_f32.ptr);
        const auto& t_lut = tile_offset_lut();

        for (uint32_t r = chunk_rows; r < dirty_mt0; ++r) {
            const auto& row_off = t_lut[r];
            for (uint32_t kt = 0; kt < 160u; ++kt) {
                uint16_t* dst_tile = x_tiles + kt * 1024u;
                for (int c = 0; c < 32; ++c) dst_tile[row_off[c]] = 0u;
            }
        }
        dirty_mt0 = chunk_rows;

        avx_pool.parallel_rows(chunk_rows, [&](uint32_t r0, uint32_t r1) {
            alignas(32) uint16_t bf16_row[DSV41_P150_H];
            for (uint32_t r = r0; r < r1; ++r) {
                const auto& row_off = t_lut[r];
                const uint16_t* src_row = x_chunk_bf16 + size_t(r) * DSV41_P150_H;
                const uint16_t* use_row = src_row;
                float* rot_dst = use_tail ? (x_rot_f32 + size_t(r) * DSV41_P150_H) : nullptr;
                if (!input_is_wht_rotated) {
                    rotate_row_avx2_to_bf16(src_row, bank.signs_h.data(), bf16_row, rot_dst);
                    use_row = bf16_row;
                } else if (rot_dst) {
                    bf16_row_to_f32_avx2(src_row, rot_dst);
                }
                for (uint32_t kt = 0; kt < 160u; ++kt) {
                    uint16_t* dst_tile = x_tiles + kt * 1024u;
                    const uint16_t* sb = use_row + kt * 32u;
                    for (int c = 0; c < 32; ++c) dst_tile[row_off[c]] = sb[c];
                }
            }
        });

        if (use_tail) {
            auto* tail_out_base = static_cast<float*>(host_tail_pairs_f32.ptr);
            const uint32_t n_desc = static_cast<uint32_t>(tail_descs.size());
            tail_pool.launch_async(n_desc * 2u, [&, tail_out_base, n_desc](uint32_t item) {
                const uint32_t tp = item & 1u;
                const uint32_t di = item >> 1u;
                if (di < n_desc) {
                    float* out_tp = tail_out_base + size_t(tp) * size_t(total_tail_pairs) * DSV41_P150_H;
                    run_cpu_tail_shard_avx2(bank, tp, tail_descs[di], x_rot_f32, out_tp);
                }
            });
        }

        const uint32_t scheduled_slots = choose_decode_slot_bucket(p150_count);
        auto* slot_u16 = static_cast<uint16_t*>(host_slot_ids.ptr);
        for (uint32_t i = 0; i < 288u; ++i) slot_u16[i] = 0xFFFFu;
        for (uint32_t i = 0; i < p150_count; ++i) slot_u16[i] = active_experts[i];
        write_layer_desc_to_slot_ids(bank);

        const size_t weights_bytes = size_t(scheduled_slots) * 4096ULL;
        auto* w_f32 = static_cast<float*>(host_weights.ptr);
        std::memset(w_f32, 0, weights_bytes);
        for (uint32_t i = 0; i < p150_count; ++i) {
            const uint16_t eid = active_experts[i];
            const uint8_t cnt = exp_cnt[eid];
            float* dst_tile = w_f32 + i * 1024u;
            for (uint8_t k = 0; k < cnt; ++k) {
                const auto& row_off = t_lut[exp_pairs[eid][k].row];
                const float w = exp_pairs[eid][k].weight;
                for (int c = 0; c < 32; ++c) dst_tile[row_off[c]] += w;
            }
        }

        const size_t out_bytes = size_t(chunk_rows) * DSV41_P150_H * sizeof(uint16_t);
        auto& cq = mesh->mesh_command_queue();
        const distributed::MeshCoordinateRange all_devs(mesh->shape());
        cq.enqueue_write_shard_to_sub_grid(ws1->x_decode.mesh_buffer(), host_x_tilized.ptr, all_devs, false, BufferRegion(0, 160u * 2048u));
        cq.enqueue_write_shard_to_sub_grid(ws1->slot_ids.mesh_buffer(), host_slot_ids.ptr, all_devs, false, BufferRegion(0, kSlotIdsBytes));
        cq.enqueue_write_shard_to_sub_grid(ws1->weights_tiled.mesh_buffer(), host_weights.ptr, all_devs, false, BufferRegion(0, weights_bytes));

        run_with_optional_trace(
            bank.layer,
            0u,
            scheduled_slots,
            uint64_t((scheduled_slots + kBatchSlots - 1u) / kBatchSlots) * 98304ULL + (2ULL << 20),
            [&]() {
                for (uint32_t s = 0; s < scheduled_slots; s += kBatchSlots) {
                    const uint32_t bcnt = std::min<uint32_t>(kBatchSlots, scheduled_slots - s);
                    enqueue_hlut2_projection(bank, ws1->x_decode, bank.gate_pages, ws1->up_partials, s, bcnt, 0u, 3u);
                    enqueue_swiglu_wht128(bank, bcnt);
                    enqueue_hlut2_projection(bank, ws1->z_rot_bf16, bank.down_pages, ws1->down_partials, s, bcnt, 36u, 2u);
                    enqueue_ksplit_combine_decode(s, bcnt);
                }
            },
            [&]() {
                for (auto& t : ws1->finish_decode) t = ttnn::Tensor();
            },
            [&]() {
                ws1->finish_decode[0] = ttnn::typecast(ws1->combined_f32, DataType::BFLOAT16, ttnn::L1_MEMORY_CONFIG);
                ws1->finish_decode[1] = ttnn::all_reduce(
                    ws1->finish_decode[0], 1, std::nullopt, ttnn::L1_MEMORY_CONFIG, 1, tt::tt_fabric::Topology::Linear);
                ws1->finish_decode[2] = ttnn::untilize(ws1->finish_decode[1], ws1->finish_untilize_mem_cfg());
            });

        const auto& out_rm = ws1->finish_decode[2];
        auto borrowed_out = std::shared_ptr<distributed::MeshBuffer>(
            const_cast<distributed::MeshBuffer*>(&out_rm.mesh_buffer()), [](auto*) {});
        cq.enqueue_read_shards(
            {distributed::ShardDataTransfer{distributed::MeshCoordinate{0, 0}}
                 .host_data(host_out_rm.ptr)
                 .region(BufferRegion(0, out_bytes))},
            borrowed_out,
            !use_tail);

        if (use_tail) {
            tail_pool.wait_complete();
            cq.finish();
            merge_p150_and_tail_to_bf16(chunk_rows, total_tail_pairs, tail_descs, y_chunk_bf16);
        } else {
            std::memcpy(y_chunk_bf16, host_out_rm.ptr, out_bytes);
        }
        maybe_promote_hot_experts();
    }

    // High-throughput Expert-Compact m_tiles=1 path for M = 33..512 (Prefill & CED 128-token tail)
    void run_prefill_compact(
        const ResidentLayerBank& bank,
        uint32_t chunk_rows,
        bool input_is_wht_rotated,
        const int32_t row_slots[][DSV41_P150_MAX_SLOTS],
        const float row_weights[][DSV41_P150_MAX_SLOTS],
        const uint16_t* x_chunk_bf16,
        uint16_t* y_chunk_bf16)
    {
        struct RoutePair {
            uint16_t row;
            float weight;
        };
        std::array<std::vector<RoutePair>, DSV41_P150_NUM_EXPERTS> buckets;
        uint32_t total_pairs = 0;

        for (uint32_t r = 0; r < chunk_rows; ++r) {
            for (uint32_t s = 0; s < DSV41_P150_MAX_SLOTS; ++s) {
                const int32_t eid = row_slots[r][s];
                const float w = row_weights[r][s];
                if (eid >= 0 && uint32_t(eid) < DSV41_P150_NUM_EXPERTS && w != 0.0f) {
                    buckets[uint32_t(eid)].push_back(RoutePair{static_cast<uint16_t>(r), w});
                    ++total_pairs;
                }
            }
        }

        if (total_pairs == 0u) {
            std::memset(y_chunk_bf16, 0, size_t(chunk_rows) * DSV41_P150_H * sizeof(uint16_t));
            return;
        }

        const uint32_t mode = (chunk_rows <= 128u) ? 1u : 2u;
        const uint32_t zero_rows = (chunk_rows <= 128u) ? 128u : 512u;

        // Priority 2: Peel ONLY <= 6 single-token (n == 1) non-hot experts unless rem_slots > kMaxCompactSlots (288),
        // eliminating the 42-expert DDR4 AVX2 stall during M = 1,024 CED prefill!
        uint32_t total_compact_slots = 0u;
        std::vector<uint16_t> tail_cands;
        tail_cands.reserve(DSV41_P150_NUM_EXPERTS);
        const uint32_t max_tail_cnt = 1u;
        for (uint32_t eid = 0; eid < DSV41_P150_NUM_EXPERTS; ++eid) {
            const size_t n = buckets[eid].size();
            if (n == 0u) continue;
            const uint32_t cslots = static_cast<uint32_t>((n + 31u) / 32u);
            total_compact_slots += cslots;
            prefill_expert_hits[bank.layer][eid] += cslots;
            if (n <= 32u) tail_cands.push_back(static_cast<uint16_t>(eid));
        }
        std::sort(tail_cands.begin(), tail_cands.end(), [&](uint16_t a, uint16_t b) {
            if (buckets[a].size() != buckets[b].size()) return buckets[a].size() < buckets[b].size();
            return a < b;
        });

        std::array<bool, DSV41_P150_NUM_EXPERTS> is_tail{};
        std::vector<TailExpertDesc> tail_descs;
        uint32_t total_tail_pairs = 0u;
        const uint32_t max_peel_experts = (mode == 1u) ? 4u : 6u;
        uint32_t rem_slots = total_compact_slots;
        for (uint16_t eid : tail_cands) {
            const uint32_t n = static_cast<uint32_t>(buckets[eid].size());
            const bool force_fit = (rem_slots > kMaxCompactSlots);
            if (!force_fit) {
                if (rem_slots <= kBatchSlots) break;
                if (n > max_tail_cnt || tail_descs.size() >= max_peel_experts) break;
                if (is_hot_expert(bank, eid)) continue;
            }
            if (total_tail_pairs + n > 1024u) break;
            TailExpertDesc td{};
            td.eid = eid;
            td.cnt = static_cast<uint16_t>(n);
            td.pair_offset = total_tail_pairs;
            for (uint32_t k = 0; k < n; ++k) {
                td.rows[k] = buckets[eid][k].row;
                td.weights[k] = buckets[eid][k].weight;
            }
            total_tail_pairs += n;
            is_tail[eid] = true;
            tail_descs.push_back(td);
            --rem_slots;
        }
        const bool use_tail = !tail_descs.empty();
        auto* dst_rm    = static_cast<uint16_t*>(host_x_rm.ptr);
        auto* x_rot_f32 = static_cast<float*>(host_x_rot_f32.ptr);
        const size_t x_rm_bytes = size_t(chunk_rows) * DSV41_P150_H * sizeof(uint16_t);

        // Parallel AVX2 WHT128 rotation into BOTH host_x_rm (BF16) and host_x_rot_f32 (FP32) BEFORE launching CPU tail!
        avx_pool.parallel_rows(chunk_rows, [&](uint32_t r0, uint32_t r1) {
            for (uint32_t r = r0; r < r1; ++r) {
                const uint16_t* src_row = x_chunk_bf16 + size_t(r) * DSV41_P150_H;
                uint16_t* row_bf16      = dst_rm + size_t(r) * DSV41_P150_H;
                float* rot_dst          = use_tail ? (x_rot_f32 + size_t(r) * DSV41_P150_H) : nullptr;
                if (!input_is_wht_rotated) {
                    rotate_row_avx2_to_bf16(src_row, bank.signs_h.data(), row_bf16, rot_dst);
                } else {
                    std::memcpy(row_bf16, src_row, DSV41_P150_H * sizeof(uint16_t));
                    if (rot_dst) bf16_row_to_f32_avx2(src_row, rot_dst);
                }
            }
        });

        if (use_tail) {
            auto* tail_out_base = static_cast<float*>(host_tail_pairs_f32.ptr);
            const uint32_t n_desc = static_cast<uint32_t>(tail_descs.size());
            tail_pool.launch_async(n_desc * 2u, [&, tail_out_base, n_desc](uint32_t item) {
                const uint32_t tp = item & 1u;
                const uint32_t di = item >> 1u;
                if (di < n_desc) {
                    float* out_tp = tail_out_base + size_t(tp) * size_t(total_tail_pairs) * DSV41_P150_H;
                    run_cpu_tail_shard_avx2(bank, tp, tail_descs[di], x_rot_f32, out_tp);
                }
            });
        }

        // Collect remaining P150A CompactSlots and greedily pack disjoint-row slots into each 16-slot batch!
        struct PendingSlot {
            uint16_t eid;
            uint16_t off;
            uint16_t cnt;
            bool placed;
            std::array<uint64_t, 16> mask;
        };
        std::vector<PendingSlot> pending;
        pending.reserve(rem_slots);
        for (uint32_t eid = 0; eid < DSV41_P150_NUM_EXPERTS; ++eid) {
            if (is_tail[eid]) continue;
            const auto& vec = buckets[eid];
            for (size_t off = 0; off < vec.size(); off += 32u) {
                const uint16_t cnt = static_cast<uint16_t>(std::min<size_t>(32u, vec.size() - off));
                PendingSlot ps{static_cast<uint16_t>(eid), static_cast<uint16_t>(off), cnt, false, {}};
                for (uint16_t k = 0; k < cnt; ++k) {
                    const uint16_t r = vec[off + k].row;
                    ps.mask[r >> 6] |= (1ULL << (r & 63u));
                }
                pending.push_back(ps);
            }
        }
        std::sort(pending.begin(), pending.end(), [](const PendingSlot& a, const PendingSlot& b) {
            if (a.cnt != b.cnt) return a.cnt > b.cnt;
            return a.eid < b.eid;
        });

        auto* slot_u16   = static_cast<uint16_t*>(host_slot_ids.ptr);
        auto* meta_bytes = static_cast<uint8_t*>(host_slot_meta.ptr);
        auto* w_f32      = static_cast<float*>(host_weights.ptr);
        for (uint32_t i = 0; i < 288u; ++i) slot_u16[i] = 0xFFFFu;
        write_layer_desc_to_slot_ids(bank);

        const auto& t_lut = tile_offset_lut();
        const uint32_t active_slots = static_cast<uint32_t>(pending.size());
        std::array<uint64_t, 16> inflight_row_mask{};
        uint32_t first_unplaced = 0u;

        for (uint32_t s = 0; s < active_slots; ++s) {
            if ((s % kBatchSlots) == 0u) inflight_row_mask.fill(0ULL);
            while (first_unplaced < active_slots && pending[first_unplaced].placed) ++first_unplaced;

            int32_t chosen = -1;
            for (uint32_t idx = first_unplaced; idx < active_slots; ++idx) {
                if (pending[idx].placed) continue;
                const auto& m = pending[idx].mask;
                uint64_t overlap = 0ULL;
                for (int w = 0; w < 16; ++w) overlap |= (inflight_row_mask[w] & m[w]);
                if (overlap == 0ULL) {
                    chosen = static_cast<int32_t>(idx);
                    break;
                }
            }

            uint32_t wait_barrier = 0u;
            if (chosen < 0) {
                chosen = static_cast<int32_t>(first_unplaced);
                wait_barrier = ((s % kBatchSlots) > 0u) ? 1u : 0u;
            }
            PendingSlot& ps = pending[uint32_t(chosen)];
            ps.placed = true;
            if (wait_barrier != 0u) {
                inflight_row_mask = ps.mask;
            } else {
                for (int m = 0; m < 16; ++m) inflight_row_mask[m] |= ps.mask[m];
            }

            slot_u16[s] = ps.eid;
            uint8_t* mptr = meta_bytes + size_t(s) * 128ULL;
            std::memset(mptr, 0, 128u);
            reinterpret_cast<uint32_t*>(mptr)[0] = ps.cnt;
            reinterpret_cast<uint32_t*>(mptr)[1] = wait_barrier;
            uint16_t* rptr = reinterpret_cast<uint16_t*>(mptr + 8u);

            float* wtile = w_f32 + size_t(s) * 1024ULL;
            std::memset(wtile, 0, 4096u);
            const auto& vec = buckets[ps.eid];
            for (uint32_t k = 0; k < ps.cnt; ++k) {
                rptr[k] = vec[ps.off + k].row;
                const float wk = vec[ps.off + k].weight;
                const auto& row_off = t_lut[k];
                for (int c = 0; c < 32; ++c) wtile[row_off[c]] = wk;
            }
        }

        CompactPrefillArena& arena = (mode == 1u) ? *arena128 : *arena512;
        const uint32_t scheduled_slots = choose_compact_slot_bucket(active_slots, mode);
        if (scheduled_slots > active_slots) {
            std::memset(meta_bytes + size_t(active_slots) * 128ULL, 0, size_t(scheduled_slots - active_slots) * 128ULL);
            std::memset(w_f32 + size_t(active_slots) * 1024ULL, 0, size_t(scheduled_slots - active_slots) * 4096ULL);
        }

        if (!ws1->x_tilized_dram || ws1->x_decode_dram) {
            dirty_mt0 = 32u;  // gather_tilize overwrites ws1->x_tilized only when x_decode aliases it
        }

        auto& cq = mesh->mesh_command_queue();
        const distributed::MeshCoordinateRange all_devs(mesh->shape());
        cq.enqueue_write_shard_to_sub_grid(x_rm_l1->mesh_buffer(), host_x_rm.ptr, all_devs, false, BufferRegion(0, x_rm_bytes));
        cq.enqueue_write_shard_to_sub_grid(ws1->slot_ids.mesh_buffer(), host_slot_ids.ptr, all_devs, false, BufferRegion(0, kSlotIdsBytes));
        cq.enqueue_write_shard_to_sub_grid(slot_meta_l1->mesh_buffer(), host_slot_meta.ptr, all_devs, false, BufferRegion(0, size_t(scheduled_slots) * 128ULL));
        cq.enqueue_write_shard_to_sub_grid(ws1->weights_tiled.mesh_buffer(), host_weights.ptr, all_devs, false, BufferRegion(0, size_t(scheduled_slots) * 4096ULL));

        run_with_optional_trace(
            bank.layer,
            mode,
            scheduled_slots,
            uint64_t((scheduled_slots + kBatchSlots - 1u) / kBatchSlots) * 114688ULL + (4ULL << 20),
            [&]() {
                enqueue_zero_pages(arena.y_accum_f32, zero_rows * 5u);
                for (uint32_t s = 0; s < scheduled_slots; s += kBatchSlots) {
                    const uint32_t bcnt = std::min<uint32_t>(kBatchSlots, scheduled_slots - s);
                    enqueue_compact_gather_tilize(s, bcnt);
                    enqueue_hlut2_projection(bank, ws1->x_tilized, bank.gate_pages, ws1->up_partials, s, bcnt, 160u, 3u);
                    enqueue_swiglu_wht128(bank, bcnt);
                    enqueue_hlut2_projection(bank, ws1->z_rot_bf16, bank.down_pages, ws1->down_partials, s, bcnt, 36u, 2u);
                    enqueue_compact_scatter_combine(arena.y_accum_f32, s, bcnt);
                }
            },
            [&]() {
                for (auto& t : arena.finish_tensors) t = ttnn::Tensor();
            },
            [&]() {
                arena.finish_tensors[0] = ttnn::typecast(arena.y_accum_f32, DataType::BFLOAT16, arena.finish_mem_cfg());
                arena.finish_tensors[1] = ttnn::all_reduce(
                    arena.finish_tensors[0], 1, std::nullopt, arena.finish_mem_cfg(), 1, tt::tt_fabric::Topology::Linear);
            });

        const auto& out_rm = arena.finish_tensors[1];
        auto borrowed_out = std::shared_ptr<distributed::MeshBuffer>(
            const_cast<distributed::MeshBuffer*>(&out_rm.mesh_buffer()), [](auto*) {});
        cq.enqueue_read_shards(
            {distributed::ShardDataTransfer{distributed::MeshCoordinate{0, 0}}
                 .host_data(host_out_rm.ptr)
                 .region(BufferRegion(0, x_rm_bytes))},
            borrowed_out,
            !use_tail);

        if (use_tail) {
            tail_pool.wait_complete();
            cq.finish();
            merge_p150_and_tail_to_bf16(chunk_rows, total_tail_pairs, tail_descs, y_chunk_bf16);
        } else {
            std::memcpy(y_chunk_bf16, host_out_rm.ptr, x_rm_bytes);
        }
    }
};

P150Dsv41Engine::P150Dsv41Engine(
    const std::string& packed_p150_root,
    const std::string& kernels_dir,
    bool enable_trace,
    const std::string& rarebal_overlay_path,
    uint32_t hot_l1_shards)
    : impl_(std::make_unique<Impl>(packed_p150_root, kernels_dir, enable_trace, rarebal_overlay_path, hot_l1_shards))
{}

P150Dsv41Engine::~P150Dsv41Engine() = default;

bool P150Dsv41Engine::ensure_layer_loaded(uint32_t layer) {
    return impl_->ensure_layer_loaded(layer);
}

uint32_t P150Dsv41Engine::load_available_layers(uint32_t max_layers) {
    uint32_t cnt = 0;
    const uint32_t lim = std::min<uint32_t>(max_layers, DSV41_P150_NUM_LAYERS);
    for (uint32_t l = 0; l < lim; ++l) {
        if (impl_->ensure_layer_loaded(l)) ++cnt;
    }
    if (impl_->rarebal_overlay_mmap && impl_->rarebal_p150_patched > 0u) {
        std::printf("[P150_DSV41] Overlaid %u P150A hot experts (x2 TP2 shards) into GDDR6 MP BEFORE L1 MR pinning!\n",
                    impl_->rarebal_p150_patched);
        std::fflush(stdout);
    }
    if (impl_->current_hot_assignments.empty() && cnt > 0u) {
        impl_->seed_initial_hot_experts();
    }
    if (cnt > 0u) {
        impl_->capture_reduce_traces_once();
    }
    return cnt;
}

uint64_t P150Dsv41Engine::loaded_layers_mask() const noexcept {
    return impl_->loaded_mask;
}

size_t P150Dsv41Engine::persistent_l1_bytes_per_card() const noexcept {
    return impl_->workspace_l1_bytes_per_card();
}

uint64_t P150Dsv41Engine::forward_moe(
    uint32_t layer,
    uint32_t rows,
    bool input_is_wht_rotated,
    const int32_t row_slots[][DSV41_P150_MAX_SLOTS],
    const float row_weights[][DSV41_P150_MAX_SLOTS],
    const uint16_t* x_in_bf16,
    uint16_t* y_out_bf16)
{
    if (rows == 0u || rows > DSV41_P150_MAX_ROWS) {
        throw std::invalid_argument("forward_moe: invalid rows=" + std::to_string(rows));
    }
    if (!impl_->ensure_layer_loaded(layer)) {
        throw std::runtime_error("forward_moe: layer not available on disk: " + std::to_string(layer));
    }
    const auto& bank = *impl_->layers[layer];
    const auto t0 = std::chrono::steady_clock::now();

    if (rows <= 32u) {
        impl_->run_decode_small(bank, rows, input_is_wht_rotated, row_slots, row_weights, x_in_bf16, y_out_bf16);
    } else {
        for (uint32_t r0 = 0; r0 < rows; r0 += 512u) {
            const uint32_t chunk = std::min<uint32_t>(512u, rows - r0);
            if (chunk <= 32u) {
                impl_->run_decode_small(
                    bank,
                    chunk,
                    input_is_wht_rotated,
                    row_slots + r0,
                    row_weights + r0,
                    x_in_bf16 + size_t(r0) * DSV41_P150_H,
                    y_out_bf16 + size_t(r0) * DSV41_P150_H);
            } else {
                impl_->run_prefill_compact(
                    bank,
                    chunk,
                    input_is_wht_rotated,
                    row_slots + r0,
                    row_weights + r0,
                    x_in_bf16 + size_t(r0) * DSV41_P150_H,
                    y_out_bf16 + size_t(r0) * DSV41_P150_H);
            }
        }
    }

    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}

namespace {

void cpu_ref_p150_proj(
    const uint8_t* pages_base,
    float c0,
    float c1,
    int nt4,
    int kt4,
    const float* x_rot,
    float* y_out)
{
    const int N = nt4 * 128;
    std::fill(y_out, y_out + N, 0.0f);
    for (int nb4 = 0; nb4 < nt4; ++nb4) {
        for (int kb4 = 0; kb4 < kt4; ++kb4) {
            const uint8_t* page = pages_base + size_t(nb4 * kt4 + kb4) * 4352ULL;
            const uint32_t* codes = reinterpret_cast<const uint32_t*>(page);
            const uint16_t* scales = reinterpret_cast<const uint16_t*>(page + 4096);

            float sig[4][32];
            for (int nt_sub = 0; nt_sub < 4; ++nt_sub) {
                for (int j = 0; j < 32; ++j) {
                    sig[nt_sub][j] = fp16_to_f32(scales[nt_sub * 32 + j]);
                }
            }

            for (int i = 0; i < 32; ++i) {
                for (int j = 0; j < 32; ++j) {
                    const uint32_t w = codes[tensix_tile_offset(i, j)];
                    for (int kt_sub = 0; kt_sub < 4; ++kt_sub) {
                        const float x_val = x_rot[(kb4 * 4 + kt_sub) * 32 + i];
                        for (int nt_sub = 0; nt_sub < 4; ++nt_sub) {
                            const int t = kt_sub * 4 + nt_sub;
                            const uint32_t code = (w >> (2 * t)) & 3u;
                            const float mag = (code & 1u) ? c1 : c0;
                            const float sgn = (code & 2u) ? -1.0f : 1.0f;
                            y_out[(nb4 * 4 + nt_sub) * 32 + j] += x_val * (sgn * mag * sig[nt_sub][j]);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace

bool P150Dsv41Engine::selfcheck_layer(uint32_t layer) {
    if (!ensure_layer_loaded(layer)) {
        std::fprintf(stderr, "[P150_SELFCHECK] Layer %u not found\n", layer);
        return false;
    }
    const auto saved_hot = impl_->current_hot_assignments;
    const auto& bank = *impl_->layers[layer];
    SafetensorsMmap st0(bank.tp0_path);
    SafetensorsMmap st1(bank.tp1_path);

    const uint32_t test_batches[] = {1u, 6u, 8u, 64u, 512u, 1024u};
    bool all_ok = true;

    for (uint32_t M : test_batches) {
        std::vector<uint16_t> x_bf16(size_t(M) * DSV41_P150_H);
        std::vector<uint16_t> x_rot_bf16(size_t(M) * DSV41_P150_H);
        std::vector<uint16_t> y_base(size_t(M) * DSV41_P150_H, 0);
        std::vector<uint16_t> y_bf16(size_t(M) * DSV41_P150_H, 0);
        std::vector<std::array<int32_t, DSV41_P150_MAX_SLOTS>> slots_v(M);
        std::vector<std::array<float, DSV41_P150_MAX_SLOTS>> weights_v(M);

        uint64_t rng = 0x123456789ABCDEF0ULL ^ M;
        auto next_f = [&]() {
            rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
            const uint64_t u = rng * 0x2545F4914F6CDD1DULL;
            return (float(int32_t(u >> 40)) / float(1 << 23)) - 0.5f;
        };

        for (uint32_t r = 0; r < M; ++r) {
            for (uint32_t j = 0; j < DSV41_P150_H; ++j) {
                x_bf16[size_t(r) * DSV41_P150_H + j] = f32_to_bf16_rne(next_f());
            }
            rotate_row_avx2_to_bf16(
                x_bf16.data() + size_t(r) * DSV41_P150_H,
                bank.signs_h.data(),
                x_rot_bf16.data() + size_t(r) * DSV41_P150_H);
            slots_v[r].fill(-1);
            weights_v[r].fill(0.0f);
            slots_v[r][0] = int32_t((r * 7u + 0u) % DSV41_P150_NUM_EXPERTS);
            slots_v[r][1] = int32_t((r * 7u + 41u) % DSV41_P150_NUM_EXPERTS);
            slots_v[r][2] = int32_t((r * 7u + 113u) % DSV41_P150_NUM_EXPERTS);
            weights_v[r][0] = 0.55f;
            weights_v[r][1] = 0.30f;
            weights_v[r][2] = 0.15f;
        }

        const auto* slots_ptr   = reinterpret_cast<const int32_t(*)[DSV41_P150_MAX_SLOTS]>(slots_v.data());
        const auto* weights_ptr = reinterpret_cast<const float(*)[DSV41_P150_MAX_SLOTS]>(weights_v.data());

        // 1. Baseline GDDR6 (unpinned e0/e41/e113) + non-pre-rotated (input_is_wht_rotated = false)
        impl_->apply_hot_assignments({{static_cast<uint16_t>(layer), 167u}});
        uint64_t ns_first = forward_moe(layer, M, false, slots_ptr, weights_ptr, x_bf16.data(), y_base.data());

        // 2. Wave 6 O(1) l1_slot_lut hot-L1 + pre-rotated (input_is_wht_rotated = true) + MR/MP 512-row L1 arena + MeshTrace!
        impl_->apply_hot_assignments({
            {static_cast<uint16_t>(layer), 0u},
            {static_cast<uint16_t>(layer), 41u},
            {static_cast<uint16_t>(layer), 113u}});
        (void)forward_moe(layer, M, true, slots_ptr, weights_ptr, x_rot_bf16.data(), y_bf16.data());
        (void)forward_moe(layer, M, true, slots_ptr, weights_ptr, x_rot_bf16.data(), y_bf16.data());
        (void)forward_moe(layer, M, true, slots_ptr, weights_ptr, x_rot_bf16.data(), y_bf16.data());
        uint64_t ns_traced = forward_moe(layer, M, true, slots_ptr, weights_ptr, x_rot_bf16.data(), y_bf16.data());

        uint32_t nonfinite = 0;
        double pdot = 0.0, pna = 0.0, pnb = 0.0;
        for (uint32_t r = 0; r < M; ++r) {
            for (uint32_t j = 0; j < DSV41_P150_H; ++j) {
                const float v  = bf16_to_f32(y_bf16[size_t(r) * DSV41_P150_H + j]);
                const float vb = bf16_to_f32(y_base[size_t(r) * DSV41_P150_H + j]);
                if (!std::isfinite(v)) ++nonfinite;
                pdot += double(v) * double(vb);
                pna  += double(v) * double(v);
                pnb  += double(vb) * double(vb);
            }
        }
        const double cos_parity = (pna > 0.0 && pnb > 0.0) ? (pdot / (std::sqrt(pna) * std::sqrt(pnb))) : 1.0;

        // Verify exact CPU FP32 reference on boundary & interior rows (including r=511/512 chunk boundary!)
        std::vector<uint32_t> check_indices;
        if (M <= 8u) {
            for (uint32_t r = 0; r < M; ++r) check_indices.push_back(r);
        } else {
            check_indices = {0u, 1u, 31u, 32u, M / 2u, M - 2u, M - 1u};
        }

        double dot = 0.0, na = 0.0, nb = 0.0;
        for (uint32_t r : check_indices) {
            std::vector<float> xr(DSV41_P150_H);
            for (uint32_t j = 0; j < DSV41_P150_H; ++j) xr[j] = bf16_to_f32(x_bf16[size_t(r) * DSV41_P150_H + j]);
            for (int b = 0; b < 40; ++b) avx2_wht128_block(xr.data() + b * 128, &bank.signs_h[b * 2]);

            std::vector<float> y_ref(DSV41_P150_H, 0.0f);
            for (int tp = 0; tp < 2; ++tp) {
                const SafetensorsMmap& st = (tp == 0) ? st0 : st1;
                const auto* si_ptr = reinterpret_cast<const uint64_t*>(st.tensor_bytes("wht.signs_i"));
                const std::vector<uint64_t> si_sub(si_ptr, si_ptr + 18);
                const float* gc = bank.gc_act[tp].data();
                const float* uc = bank.uc_act[tp].data();
                const float* dc = bank.dc_act[tp].data();

                for (uint32_t s = 0; s < 3u; ++s) {
                    const int e = slots_v[r][s];
                    const float w = weights_v[r][s];
                    std::vector<float> g_out(1152), u_out(1152), z(1152), d_out(DSV41_P150_H);
                    cpu_ref_p150_proj(bank.gp_exp[tp][uint32_t(e)], gc[2 * e], gc[2 * e + 1], 9, 40, xr.data(), g_out.data());
                    cpu_ref_p150_proj(bank.up_exp[tp][uint32_t(e)], uc[2 * e], uc[2 * e + 1], 9, 40, xr.data(), u_out.data());
                    for (int k = 0; k < 1152; ++k) z[k] = dsv41_swiglu_scalar(g_out[k], u_out[k]);
                    for (int b = 0; b < 9; ++b) avx2_wht128_block(z.data() + b * 128, &si_sub[b * 2]);
                    cpu_ref_p150_proj(bank.dp_exp[tp][uint32_t(e)], dc[2 * e], dc[2 * e + 1], 40, 9, z.data(), d_out.data());
                    for (uint32_t j = 0; j < DSV41_P150_H; ++j) y_ref[j] += w * d_out[j];
                }
            }

            for (uint32_t j = 0; j < DSV41_P150_H; ++j) {
                const double a = y_ref[j];
                const double b = bf16_to_f32(y_bf16[size_t(r) * DSV41_P150_H + j]);
                dot += a * b;
                na  += a * a;
                nb  += b * b;
            }
        }

        const double cos_fp32 = (na > 0.0 && nb > 0.0) ? (dot / (std::sqrt(na) * std::sqrt(nb))) : 0.0;
        const bool pass = (nonfinite == 0u) && (cos_parity >= 0.99999) && (cos_fp32 >= 0.995);
        std::printf("[P150_SELFCHECK] layer=%u M=%4u | nonfinite=%u parity_cos=%.7f fp32_cos=%.6f | first=%.2f ms traced=%.2f ms -> %s\n",
                    layer, M, nonfinite, cos_parity, cos_fp32, double(ns_first) / 1e6, double(ns_traced) / 1e6, pass ? "PASS" : "FAIL");
        std::fflush(stdout);
        all_ok = all_ok && pass;
    }

    // Reset profiling counters & restore saved 11-shard hot assignments cleanly
    for (auto& row : impl_->decode_expert_hits) row.fill(0ULL);
    for (auto& row : impl_->prefill_expert_hits) row.fill(0ULL);
    impl_->decode_layer_calls = 0ULL;
    impl_->decode_total_p150_slots = 0ULL;
    impl_->decode_hot_l1_hits = 0ULL;
    if (!saved_hot.empty()) {
        impl_->apply_hot_assignments(saved_hot);
    }
    return all_ok;
}

}  // namespace dsv41
