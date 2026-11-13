#pragma once
// Pure C++17/20 Lock-Free POSIX /dev/shm IPC Channel between 3x MI50 (dsv41_fe) and 2x Blackhole P150A (p150_dsv41_worker)
// Zero Python anywhere. Supports M = 1 .. 512 token rows (M=1 decode, M=6 DSpark verify, M=64..512 chunked prefill).

#include <atomic>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace dsv41 {

static constexpr uint32_t DSV41_P150_SHM_MAGIC   = 0x44343150u;  // 'D41P'
static constexpr uint32_t DSV41_P150_SHM_VERSION = 2u;
static constexpr uint32_t DSV41_P150_MAX_ROWS    = 1024u;
static constexpr uint32_t DSV41_P150_MAX_SLOTS   = 8u;           // top-6 routed experts padded to 8
static constexpr uint32_t DSV41_P150_H           = 5120u;
static constexpr uint32_t DSV41_P150_EXPERT_BASE = 216u;         // Global experts [216..383] -> local [0..167]
static constexpr uint32_t DSV41_P150_NUM_EXPERTS = 168u;
static constexpr uint32_t DSV41_P150_NUM_LAYERS  = 40u;          // Backbone layers 0..39

enum Dsv41P150Op : uint32_t {
    DSV41_P150_OP_MOE_FORWARD = 1u,
    DSV41_P150_OP_PING        = 2u,
    DSV41_P150_OP_SHUTDOWN    = 99u,
};

struct alignas(64) Dsv41P150ShmChannel {
    uint32_t magic;
    uint32_t version;
    int32_t  worker_pid;
    uint32_t loaded_layers_mask_lo;  // bitmask of layers 0..31 loaded
    uint32_t loaded_layers_mask_hi;  // bitmask of layers 32..39 loaded
    uint32_t ready;

    alignas(64) std::atomic<uint32_t> req_seq;
    uint32_t op;                     // Dsv41P150Op
    uint32_t layer;                  // 0..39
    uint32_t rows;                   // 1..512
    uint32_t input_is_wht_rotated;   // 1 if x_in_bf16 already has WHT128(signs_h) applied, 0 if raw BF16

    alignas(64) std::atomic<uint32_t> resp_seq;
    int32_t  status;                 // 0 = ok, <0 = error
    uint64_t last_device_ns;

    // Per-row local P150 expert slots [0..167] (-1 = unused) and scaled FP32 router weights
    alignas(64) int32_t row_slots[DSV41_P150_MAX_ROWS][DSV41_P150_MAX_SLOTS];
    alignas(64) float   row_weights[DSV41_P150_MAX_ROWS][DSV41_P150_MAX_SLOTS];

    // Row-major BF16 activations [rows, 5120] and TP2-summed BF16 expert outputs [rows, 5120]
    alignas(64) uint16_t x_in_bf16[DSV41_P150_MAX_ROWS * DSV41_P150_H];
    alignas(64) uint16_t y_out_bf16[DSV41_P150_MAX_ROWS * DSV41_P150_H];
};

inline void dsv41_cpu_relax() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

class Dsv41P150ShmMapping {
public:
    Dsv41P150ShmChannel* ptr = nullptr;
    int fd = -1;
    bool owner = false;
    std::string shm_name;

    Dsv41P150ShmMapping() = default;
    Dsv41P150ShmMapping(const Dsv41P150ShmMapping&) = delete;
    Dsv41P150ShmMapping& operator=(const Dsv41P150ShmMapping&) = delete;

    ~Dsv41P150ShmMapping() { close_mapping(); }

    void close_mapping() noexcept {
        if (ptr) {
            ::munmap(ptr, sizeof(Dsv41P150ShmChannel));
            ptr = nullptr;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        if (owner && !shm_name.empty()) {
            ::shm_unlink(shm_name.c_str());
            owner = false;
        }
    }

    bool create_server(const std::string& name = "/dsv41_p150_ipc") {
        close_mapping();
        shm_name = name;
        ::shm_unlink(shm_name.c_str());
        fd = ::shm_open(shm_name.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (fd < 0) return false;
        if (::ftruncate(fd, sizeof(Dsv41P150ShmChannel)) != 0) {
            close_mapping();
            return false;
        }
        void* m = ::mmap(nullptr, sizeof(Dsv41P150ShmChannel), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) {
            close_mapping();
            return false;
        }
        (void)::mlock(m, sizeof(Dsv41P150ShmChannel));
        std::memset(m, 0, sizeof(Dsv41P150ShmChannel));
        ptr = static_cast<Dsv41P150ShmChannel*>(m);
        owner = true;
        ptr->worker_pid = static_cast<int32_t>(::getpid());
        ptr->version = DSV41_P150_SHM_VERSION;
        ptr->req_seq.store(0u, std::memory_order_relaxed);
        ptr->resp_seq.store(0u, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        ptr->magic = DSV41_P150_SHM_MAGIC;
        return true;
    }

    bool reuse_or_create_server(const std::string& name = "/dsv41_p150_ipc") {
        close_mapping();
        shm_name = name;
        fd = ::shm_open(shm_name.c_str(), O_RDWR | O_CLOEXEC, 0600);
        if (fd >= 0) {
            void* m = ::mmap(nullptr, sizeof(Dsv41P150ShmChannel), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (m != MAP_FAILED) {
                auto* ch = static_cast<Dsv41P150ShmChannel*>(m);
                if (ch->magic == DSV41_P150_SHM_MAGIC && ch->version == DSV41_P150_SHM_VERSION) {
                    (void)::mlock(m, sizeof(Dsv41P150ShmChannel));
                    ptr = ch;
                    owner = false;
                    ptr->worker_pid = static_cast<int32_t>(::getpid());
                    return true;
                }
                ::munmap(m, sizeof(Dsv41P150ShmChannel));
            }
            ::close(fd);
            fd = -1;
        }
        return create_server(name);
    }

    bool attach_client(const std::string& name = "/dsv41_p150_ipc") {
        close_mapping();
        shm_name = name;
        fd = ::shm_open(shm_name.c_str(), O_RDWR | O_CLOEXEC, 0600);
        if (fd < 0) return false;
        void* m = ::mmap(nullptr, sizeof(Dsv41P150ShmChannel), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) {
            close_mapping();
            return false;
        }
        auto* ch = static_cast<Dsv41P150ShmChannel*>(m);
        if (ch->magic != DSV41_P150_SHM_MAGIC || ch->version != DSV41_P150_SHM_VERSION || !ch->ready) {
            ::munmap(m, sizeof(Dsv41P150ShmChannel));
            ::close(fd);
            fd = -1;
            return false;
        }
        ptr = ch;
        owner = false;
        return true;
    }
};

}  // namespace dsv41
