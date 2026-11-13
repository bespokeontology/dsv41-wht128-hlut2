// Blackhole P150A Fused K-Split Reduction + Clamped SwiGLU(10.0) + L1 4x4 Tile WHT128 Compute Kernel
// Batches k_splits tiles in CB 0 / CB 1 and eliminates redundant srca format reconfigs inside K-split loop!

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/clamped_silu_glu.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    const uint32_t k_splits = get_arg_val<uint32_t>(0);
    const uint32_t m_cnt    = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(0, 1, 24);
    cb_wait_front(2, 16);

    for (uint32_t dm = 0; dm < m_cnt; ++dm) {
        // Phase A: FP32 K-split reduction + DeepSeek-V4 clamped_silu_glu_tile -> CB 24 (Float16_b)
        reconfig_full_operand(0, 1);
        pack_reconfig_data_format(24);
        cb_reserve_back(24, 4);

        for (uint32_t nt_sub = 0; nt_sub < 4u; ++nt_sub) {
            cb_wait_front(0, k_splits);
            cb_wait_front(1, k_splits);

            tile_regs_acquire();
            reconfig_data_format_srca(0);
            copy_init(0);
            copy_tile(0, 0, 0);
            copy_tile(1, 0, 1);

            add_binary_tile_init();
            for (uint32_t ks = 1u; ks < k_splits; ++ks) {
                copy_tile(0, ks, 2);
                copy_tile(1, ks, 3);
                add_binary_tile(0, 2, 0);
                add_binary_tile(1, 3, 1);
            }

            clamped_silu_glu_tile_init();
            clamped_silu_glu_tile(0, 1, 0);
            tile_regs_commit();

            cb_pop_front(0, k_splits);
            cb_pop_front(1, k_splits);

            tile_regs_wait();
            pack_tile(0, 24);
            tile_regs_release();
        }
        cb_push_back(24, 4);
        cb_wait_front(24, 4);

        // Phase B: L1 WHT128 block transform via [1, 4] x [4, 4] matmul_block -> CB 16 (Float16_b)
        reconfig_full_operand<SrcOrder::Reverse>(24, 2);
        pack_reconfig_data_format(24, 16);
        matmul_block_init(24, 2, 0, 4, 1, 4);

        tile_regs_acquire();
        for (uint32_t k = 0; k < 4u; ++k) {
            matmul_block(24, 2, k, k * 4u, 0, 0, 4, 1, 4);
        }
        tile_regs_commit();

        cb_reserve_back(16, 4);
        tile_regs_wait();
        pack_block(0, 16, 4);
        tile_regs_release();
        cb_push_back(16, 4);

        cb_pop_front(24, 4);
    }

    cb_pop_front(2, 16);
}
