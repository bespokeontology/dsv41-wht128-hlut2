// Blackhole P150A down_proj K-Split Reduction + Router-Weighted FP32 Combine Compute Kernel
// Sums k_splits FP32 partial tiles in Dest, scales by per-row router weight (suppressing 0 * NaN),
// optionally accumulates onto previous active expert sum, and packs FP32 to CB 16.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel::sfpu {
template <int iterations>
inline void scale_zero_safe(uint32_t val_dst, uint32_t wgt_dst, uint32_t out_dst) {
    for (int i = 0; i < iterations; ++i) {
        const sfpi::vFloat a = sfpi::dst_reg[val_dst * 32u];
        const sfpi::vFloat w = sfpi::dst_reg[wgt_dst * 32u];
        sfpi::vFloat r = 0.0f;
        v_if (w != 0.0f) { r = a * w; } v_endif;
        sfpi::dst_reg[out_dst * 32u] = r;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    const uint32_t k_splits        = get_arg_val<uint32_t>(0);
    const uint32_t tile_cnt        = get_arg_val<uint32_t>(1);
    const uint32_t accumulate_prev = get_arg_val<uint32_t>(2);
    const uint32_t slots_per_out   = get_arg_val<uint32_t>(3);  // batch_cnt for decode, 1 for compact_scatter

    compute_kernel_hw_startup(0, 1, 16);

    for (uint32_t i = 0; i < tile_cnt; ++i) {
        tile_regs_acquire();

        for (uint32_t b = 0; b < slots_per_out; ++b) {
            const uint32_t p_dst = (b == 0u) ? 0u : 1u;
            const uint32_t w_dst = (b == 0u) ? 1u : 2u;

            for (uint32_t ks = 0; ks < k_splits; ++ks) {
                cb_wait_front(0, 1);
                copy_init(0);
                if (ks == 0u) {
                    copy_tile(0, 0, p_dst);
                } else {
                    copy_tile(0, 0, 2);
                    add_binary_tile_init();
                    add_binary_tile(p_dst, 2, p_dst);
                }
                cb_pop_front(0, 1);
            }

            cb_wait_front(1, 1);
            copy_init(1);
            copy_tile(1, 0, w_dst);
            cb_pop_front(1, 1);

            MATH((SFPU_BINARY_INIT(unused)));
            MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, true, scale_zero_safe, (8), p_dst, w_dst, p_dst, VectorMode::RC)));
            if (b > 0u) {
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
        }

        if (accumulate_prev != 0u) {
            cb_wait_front(2, 1);
            copy_init(2);
            copy_tile(2, 0, 2);
            cb_pop_front(2, 1);

            add_binary_tile_init();
            add_binary_tile(2, 0, 0);
        }

        tile_regs_commit();
        cb_reserve_back(16, 1);
        tile_regs_wait();
        pack_tile(0, 16);
        tile_regs_release();
        cb_push_back(16, 1);
    }
}
