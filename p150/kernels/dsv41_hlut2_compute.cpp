// Blackhole P150A Wave 9.4 Wrap-Safe Multi-Tile Weight-Stationary (rt_dim=2 FPU) HLUT2 Pre-Unpack + Matmul Compute Kernel
//   1. Reads exact slot_ids run plan from CB 6 via get_tile_address(6, 0) (mailbox-synced UNPACK/MATH/PACK, 0 B extra L1).
//   2. Skips empty 0xFFFF padding slots completely (0 unpacks, 0 matmuls).
//   3. Unified UInt32 format across CB 4 & CB 5 eliminates 6x/superblock reconfig_data_format_srca + copy_init stalls.
//   4. Wrap-safe rt_dim=2 (all 8 FP32 Dest tiles): tracks cb0_half so rt_dim=2 runs ONLY when CB 0 is at half 0,
//      and packs CB 16 strictly in 4-tile halves (0 wrap overflow ever!).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel::sfpu {

template <int KT_SUB, int NT_SUB>
sfpi_inline sfpi::vFloat decode_hlut2_vec(
    const sfpi::vUInt& w,
    const sfpi::vFloat& base,
    const sfpi::vFloat& step) {
    constexpr uint32_t sh    = static_cast<uint32_t>(2 * (KT_SUB * 4 + NT_SUB));
    constexpr uint32_t mask0 = 1u << sh;
    sfpi::vFloat v = base;
    v_if ((w & mask0) != 0u) { v = v + step; } v_endif;
    if constexpr (sh == 30u) {
        return sfpi::copysgn(v, sfpi::as<sfpi::vSMag>(w));
    } else {
        return sfpi::copysgn(v, sfpi::as<sfpi::vSMag>(w << (30u - sh)));
    }
}

template <int NT_SUB, int OUT_BASE, int COL_FACE, int ODD>
__attribute__((noinline)) void unpack_nt_4kt() {
    constexpr int sig_slot = 224 + COL_FACE * 8 + NT_SUB * 2 + ODD;
    constexpr int face_top = COL_FACE * 8 + ODD;
    constexpr int face_bot = 16 + COL_FACE * 8 + ODD;

    const sfpi::vFloat sig  = sfpi::dst_reg[sig_slot];
    const sfpi::vFloat c0_v = sfpi::dst_reg[240 + ODD];
    const sfpi::vFloat dc_v = sfpi::dst_reg[242 + ODD];
    const sfpi::vFloat base = sig * c0_v;
    const sfpi::vFloat step = sig * dc_v;

#pragma GCC unroll 4
    for (int r_grp = 0; r_grp < 4; ++r_grp) {
        const int row_off    = face_top + 2 * r_grp;
        const sfpi::vInt w_i = sfpi::dst_reg[192 + row_off].mode<sfpi::DataLayout::I32>();
        const sfpi::vUInt w  = sfpi::as<sfpi::vUInt>(w_i);
        sfpi::dst_reg[(OUT_BASE + 0) * 32 + row_off] = decode_hlut2_vec<0, NT_SUB>(w, base, step);
        sfpi::dst_reg[(OUT_BASE + 1) * 32 + row_off] = decode_hlut2_vec<1, NT_SUB>(w, base, step);
        sfpi::dst_reg[(OUT_BASE + 2) * 32 + row_off] = decode_hlut2_vec<2, NT_SUB>(w, base, step);
        sfpi::dst_reg[(OUT_BASE + 3) * 32 + row_off] = decode_hlut2_vec<3, NT_SUB>(w, base, step);
    }

#pragma GCC unroll 4
    for (int r_grp = 0; r_grp < 4; ++r_grp) {
        const int row_off    = face_bot + 2 * r_grp;
        const sfpi::vInt w_i = sfpi::dst_reg[192 + row_off].mode<sfpi::DataLayout::I32>();
        const sfpi::vUInt w  = sfpi::as<sfpi::vUInt>(w_i);
        sfpi::dst_reg[(OUT_BASE + 0) * 32 + row_off] = decode_hlut2_vec<0, NT_SUB>(w, base, step);
        sfpi::dst_reg[(OUT_BASE + 1) * 32 + row_off] = decode_hlut2_vec<1, NT_SUB>(w, base, step);
        sfpi::dst_reg[(OUT_BASE + 2) * 32 + row_off] = decode_hlut2_vec<2, NT_SUB>(w, base, step);
        sfpi::dst_reg[(OUT_BASE + 3) * 32 + row_off] = decode_hlut2_vec<3, NT_SUB>(w, base, step);
    }
}

template <int NT_SUB, int KT_BASE, int OUT_BASE, int COL_FACE, int ODD>
__attribute__((noinline)) void unpack_nt_2kt() {
    constexpr int sig_slot = 224 + COL_FACE * 8 + NT_SUB * 2 + ODD;
    constexpr int face_top = COL_FACE * 8 + ODD;
    constexpr int face_bot = 16 + COL_FACE * 8 + ODD;

    const sfpi::vFloat sig  = sfpi::dst_reg[sig_slot];
    const sfpi::vFloat c0_v = sfpi::dst_reg[240 + ODD];
    const sfpi::vFloat dc_v = sfpi::dst_reg[242 + ODD];
    const sfpi::vFloat base = sig * c0_v;
    const sfpi::vFloat step = sig * dc_v;

#pragma GCC unroll 4
    for (int r_grp = 0; r_grp < 4; ++r_grp) {
        const int row_off    = face_top + 2 * r_grp;
        const sfpi::vInt w_i = sfpi::dst_reg[192 + row_off].mode<sfpi::DataLayout::I32>();
        const sfpi::vUInt w  = sfpi::as<sfpi::vUInt>(w_i);
        sfpi::dst_reg[(OUT_BASE + 0) * 32 + row_off] = decode_hlut2_vec<KT_BASE + 0, NT_SUB>(w, base, step);
        sfpi::dst_reg[(OUT_BASE + 1) * 32 + row_off] = decode_hlut2_vec<KT_BASE + 1, NT_SUB>(w, base, step);
    }

#pragma GCC unroll 4
    for (int r_grp = 0; r_grp < 4; ++r_grp) {
        const int row_off    = face_bot + 2 * r_grp;
        const sfpi::vInt w_i = sfpi::dst_reg[192 + row_off].mode<sfpi::DataLayout::I32>();
        const sfpi::vUInt w  = sfpi::as<sfpi::vUInt>(w_i);
        sfpi::dst_reg[(OUT_BASE + 0) * 32 + row_off] = decode_hlut2_vec<KT_BASE + 0, NT_SUB>(w, base, step);
        sfpi::dst_reg[(OUT_BASE + 1) * 32 + row_off] = decode_hlut2_vec<KT_BASE + 1, NT_SUB>(w, base, step);
    }
}

}  // namespace ckernel::sfpu
#endif

#define CALL_4KT(NT_SUB, OUT_BASE) \
    MATH((::ckernel::sfpu::unpack_nt_4kt<NT_SUB, OUT_BASE, 0, 0>())); \
    MATH((::ckernel::sfpu::unpack_nt_4kt<NT_SUB, OUT_BASE, 0, 1>())); \
    MATH((::ckernel::sfpu::unpack_nt_4kt<NT_SUB, OUT_BASE, 1, 0>())); \
    MATH((::ckernel::sfpu::unpack_nt_4kt<NT_SUB, OUT_BASE, 1, 1>()))

#define CALL_2KT(NT_SUB, KT_BASE, OUT_BASE) \
    MATH((::ckernel::sfpu::unpack_nt_2kt<NT_SUB, KT_BASE, OUT_BASE, 0, 0>())); \
    MATH((::ckernel::sfpu::unpack_nt_2kt<NT_SUB, KT_BASE, OUT_BASE, 0, 1>())); \
    MATH((::ckernel::sfpu::unpack_nt_2kt<NT_SUB, KT_BASE, OUT_BASE, 1, 0>())); \
    MATH((::ckernel::sfpu::unpack_nt_2kt<NT_SUB, KT_BASE, OUT_BASE, 1, 1>()))

inline void copy_codes_and_scales_to_67_fast() {
    copy_tile(4, 0, 6);
    copy_tile(5, 0, 7);
}

inline void unpack_superblock_16tiles(uint32_t sb_base) {
    tile_regs_acquire();
    copy_codes_and_scales_to_67_fast();
    MATH((_llk_math_eltwise_sfpu_start_(0)));
    CALL_4KT(0, 0);
    CALL_2KT(1, 0, 4);
    MATH((_llk_math_eltwise_sfpu_done_()));
    tile_regs_commit();

    tile_regs_wait();
    pack_tile<true>(0, 1, sb_base + 0u);
    pack_tile<true>(1, 1, sb_base + 4u);
    pack_tile<true>(2, 1, sb_base + 8u);
    pack_tile<true>(3, 1, sb_base + 12u);
    pack_tile<true>(4, 1, sb_base + 1u);
    pack_tile<true>(5, 1, sb_base + 5u);
    tile_regs_release();

    tile_regs_acquire();
    copy_codes_and_scales_to_67_fast();
    MATH((_llk_math_eltwise_sfpu_start_(0)));
    CALL_2KT(1, 2, 0);
    CALL_4KT(2, 2);
    MATH((_llk_math_eltwise_sfpu_done_()));
    tile_regs_commit();

    tile_regs_wait();
    pack_tile<true>(0, 1, sb_base + 9u);
    pack_tile<true>(1, 1, sb_base + 13u);
    pack_tile<true>(2, 1, sb_base + 2u);
    pack_tile<true>(3, 1, sb_base + 6u);
    pack_tile<true>(4, 1, sb_base + 10u);
    pack_tile<true>(5, 1, sb_base + 14u);
    tile_regs_release();

    tile_regs_acquire();
    copy_codes_and_scales_to_67_fast();
    MATH((_llk_math_eltwise_sfpu_start_(0)));
    CALL_4KT(3, 0);
    MATH((_llk_math_eltwise_sfpu_done_()));
    tile_regs_commit();

    tile_regs_wait();
    pack_tile<true>(0, 1, sb_base + 3u);
    pack_tile<true>(1, 1, sb_base + 7u);
    pack_tile<true>(2, 1, sb_base + 11u);
    pack_tile<true>(3, 1, sb_base + 15u);
    tile_regs_release();
}

void kernel_main() {
    const uint32_t num_sb        = get_arg_val<uint32_t>(0);
    const uint32_t batch_cnt     = get_arg_val<uint32_t>(1);
    const uint32_t num_projs     = get_arg_val<uint32_t>(2);  // 2 for fused gate+up, 1 for down
    const uint32_t keep_act      = get_arg_val<uint32_t>(3);  // 1 when act_slot_stride == 0
    const uint32_t slot_start    = get_arg_val<uint32_t>(4);  // batch offset into slot_ids
    const uint32_t kt_core       = num_sb * 4u;
    const uint32_t total_w_tiles = num_sb * 16u;

    cb_wait_front(6, 1);
    const uint32_t scratch_addr = get_tile_address(6, 0);
    const volatile uint16_t* slot_u16 = reinterpret_cast<const volatile uint16_t*>(scratch_addr);
    uint16_t local_eids[24];
    bool any_valid = false;
    for (uint32_t b = 0u; b < batch_cnt; ++b) {
        const uint16_t e = slot_u16[slot_start + b];
        local_eids[b] = e;
        if (e < 0xFFFEu) any_valid = true;
    }
    if (!any_valid) {
        return;
    }

    compute_kernel_hw_startup(4, 4, 1);
    reconfig_data_format_srca(4);
    copy_init(4);
    MATH((SFPU_BINARY_INIT(unused)));
    bool need_sfpu_reconfig = false;

    if (keep_act != 0u) {
        cb_wait_front(0, kt_core);
    }

    auto unpack_weights_to_cb1 = [&]() __attribute__((always_inline)) {
        if (need_sfpu_reconfig) {
            reconfig_full_operand(4, 4);
            pack_reconfig_data_format(1);
            reconfig_data_format_srca(4);
            copy_init(4);
            MATH((SFPU_BINARY_INIT(unused)));
        }
        cb_reserve_back(1, total_w_tiles);
        for (uint32_t sb = 0u; sb < num_sb; ++sb) {
            cb_wait_front(4, 1);
            cb_wait_front(5, 1);
            unpack_superblock_16tiles(sb * 16u);
            cb_pop_front(4, 1);
            cb_pop_front(5, 1);
        }
        cb_push_back(1, total_w_tiles);
        cb_wait_front(1, total_w_tiles);
    };

    auto matmul_one_strip = [&]() __attribute__((always_inline)) {
        tile_regs_acquire();
        for (uint32_t inner_dim_idx = 0u; inner_dim_idx < kt_core; ++inner_dim_idx) {
            matmul_block(0, 1, inner_dim_idx, inner_dim_idx * 4u, 0, 0, 4, 1, kt_core);
        }
        tile_regs_commit();

        cb_reserve_back(16, 4);
        tile_regs_wait();
        pack_block(0, 16, 4);
        tile_regs_release();
        cb_push_back(16, 4);
    };

    // Pack strictly in two 4-tile halves into 8-tile CB 16 so CB 16 NEVER wraps mid-pack!
    auto matmul_two_strips = [&]() __attribute__((always_inline)) {
        tile_regs_acquire();
        for (uint32_t inner_dim_idx = 0u; inner_dim_idx < kt_core; ++inner_dim_idx) {
            matmul_block(0, 1, inner_dim_idx, inner_dim_idx * 4u, 0, 0, 4, 2, kt_core);
        }
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(16, 4);
        pack_block(0, 16, 4);
        cb_push_back(16, 4);
        cb_reserve_back(16, 4);
        pack_block(4, 16, 4);
        cb_push_back(16, 4);
        tile_regs_release();
    };

    const bool can_rt2 = (keep_act == 0u && num_projs == 2u);
    uint32_t cb0_half = 0u;
    uint32_t b0 = 0u;
    while (b0 < batch_cnt) {
        const uint32_t expert_slot = local_eids[b0];
        if (expert_slot >= 0xFFFEu) {
            ++b0;
            continue;
        }
        uint32_t run_len = 1u;
        while (b0 + run_len < batch_cnt && local_eids[b0 + run_len] == expert_slot) {
            ++run_len;
        }

        const bool hold_2 = (can_rt2 && run_len == 2u && cb0_half == 0u);
        if (keep_act == 0u && run_len == 1u) {
            cb_wait_front(0, kt_core);
        } else if (hold_2) {
            cb_wait_front(0, 2u * kt_core);
        }

        for (uint32_t p = 0u; p < num_projs; ++p) {
            unpack_weights_to_cb1();

            reconfig_full_operand<SrcOrder::Reverse>(0, 1);
            pack_reconfig_data_format(1, 16);
            need_sfpu_reconfig = true;

            if (keep_act != 0u || run_len == 1u) {
                matmul_block_init(0, 1, 0, 4, 1, kt_core);
                for (uint32_t r = 0u; r < run_len; ++r) {
                    matmul_one_strip();
                }
            } else if (hold_2) {
                matmul_block_init(0, 1, 0, 4, 2, kt_core);
                matmul_two_strips();
            } else if (can_rt2) {
                uint32_t cur_rt = 0u;
                uint32_t r = 0u;
                while (r < run_len) {
                    if (cb0_half == 0u && r + 2u <= run_len) {
                        if (cur_rt != 2u) {
                            matmul_block_init(0, 1, 0, 4, 2, kt_core);
                            cur_rt = 2u;
                        }
                        cb_wait_front(0, 2u * kt_core);
                        matmul_two_strips();
                        cb_pop_front(0, 2u * kt_core);
                        r += 2u;
                    } else {
                        if (cur_rt != 1u) {
                            matmul_block_init(0, 1, 0, 4, 1, kt_core);
                            cur_rt = 1u;
                        }
                        cb_wait_front(0, kt_core);
                        matmul_one_strip();
                        cb_pop_front(0, kt_core);
                        cb0_half ^= 1u;
                        r += 1u;
                    }
                }
            } else {
                matmul_block_init(0, 1, 0, 4, 1, kt_core);
                for (uint32_t r = 0u; r < run_len; ++r) {
                    cb_wait_front(0, kt_core);
                    matmul_one_strip();
                    cb_pop_front(0, kt_core);
                }
            }

            cb_pop_front(1, total_w_tiles);
        }

        if (keep_act == 0u && run_len == 1u) {
            cb_pop_front(0, kt_core);
            if (can_rt2) cb0_half ^= 1u;
        } else if (hold_2) {
            cb_pop_front(0, 2u * kt_core);
        }

        b0 += run_len;
    }

    if (keep_act != 0u) {
        cb_pop_front(0, kt_core);
    }
}
