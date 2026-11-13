// Blackhole P150A Expert-Compact L1 Row Gather + Tilizer (RISCV_0 / NOC_1 across 80 cores)
// Core core_i in [0..79] gathers cnt <= 32 BF16 rows x 64 cols ([64*core_i .. 64*core_i+63], 128 B/row)
// from x_rm_l1 (page = row * 5 + (core_i >> 4), byte_off = (core_i & 15) * 128), zeroes rows [cnt..31],
// tilizes into 2 standard 4-face 32x32 BF16 tiles, and writes tiles (2*core_i, 2*core_i+1) to ws1.x_tilized.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_rm_addr      = get_arg_val<uint32_t>(0);
    const uint32_t slot_meta_addr = get_arg_val<uint32_t>(1);
    const uint32_t x_tilized_addr = get_arg_val<uint32_t>(2);
    const uint32_t slot_start     = get_arg_val<uint32_t>(3);
    const uint32_t core_i         = get_arg_val<uint32_t>(4);  // 0..79
    const uint32_t batch_cnt      = get_arg_val<uint32_t>(5);  // 1..16

    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto m_args = TensorAccessorArgs<x_args.next_compile_time_args_offset(), x_args.next_common_runtime_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<m_args.next_compile_time_args_offset(), m_args.next_common_runtime_args_offset()>();

    const auto x_rm_acc = TensorAccessor(x_args, x_rm_addr, 2048);
    const auto meta_acc = TensorAccessor(m_args, slot_meta_addr, 128);
    const auto out_acc  = TensorAccessor(o_args, x_tilized_addr, 2048);

    // CB 0: 10240 B scratch ([0..127]=meta, [128..4223]=gather_buf 32x128B, [4224..6271]=tile0, [6272..8319]=tile1)
    cb_reserve_back(0, 1);
    const uint32_t base_l1   = get_write_ptr(0);
    const uint32_t gather_l1 = base_l1 + 128u;
    const uint32_t tile0_l1  = base_l1 + 4224u;
    const uint32_t tile1_l1  = base_l1 + 6272u;

    const uint32_t page_grp = core_i >> 4;
    const uint32_t byte_off = (core_i & 15u) * 128u;

    for (uint32_t b = 0; b < batch_cnt; ++b) {
        const uint32_t slot_idx = slot_start + b;
        noc_async_read(meta_acc.get_noc_addr(slot_idx, 0), base_l1, 128u);
        noc_async_read_barrier();

        const volatile uint32_t* meta_u32 = reinterpret_cast<const volatile uint32_t*>(base_l1);
        const uint32_t cnt = meta_u32[0];
        const volatile uint16_t* row_idx = reinterpret_cast<const volatile uint16_t*>(base_l1 + 8u);

        for (uint32_t r = 0; r < cnt; ++r) {
            const uint32_t src_row = row_idx[r];
            noc_async_read(x_rm_acc.get_noc_addr(src_row * 5u + page_grp, byte_off), gather_l1 + r * 128u, 128u);
        }

        uint32_t* dst0 = reinterpret_cast<uint32_t*>(tile0_l1);
        uint32_t* dst1 = reinterpret_cast<uint32_t*>(tile1_l1);
        const uint32_t top_lim = (cnt < 16u) ? 16u : 32u;
        for (uint32_t r = cnt; r < top_lim; ++r) {
            uint32_t* z = reinterpret_cast<uint32_t*>(gather_l1 + r * 128u);
            for (uint32_t w = 0; w < 32u; ++w) z[w] = 0u;
        }
        if (cnt > 0u) {
            noc_async_read_barrier();
        }

        const uint32_t* src_u32 = reinterpret_cast<const uint32_t*>(gather_l1);
        for (uint32_t fr = 0; fr < 16u; ++fr) {
            const uint32_t* s_top = src_u32 + fr * 32u;
            const uint32_t d_off  = fr * 8u;
            for (uint32_t c2 = 0; c2 < 8u; ++c2) {
                dst0[0u   + d_off + c2] = s_top[0u  + c2];
                dst0[128u + d_off + c2] = s_top[8u  + c2];
                dst1[0u   + d_off + c2] = s_top[16u + c2];
                dst1[128u + d_off + c2] = s_top[24u + c2];
            }
        }
        if (cnt <= 16u) {
            for (uint32_t w = 256u; w < 512u; ++w) {
                dst0[w] = 0u;
                dst1[w] = 0u;
            }
        } else {
            for (uint32_t fr = 0; fr < 16u; ++fr) {
                const uint32_t* s_bot = src_u32 + (16u + fr) * 32u;
                const uint32_t d_off  = fr * 8u;
                for (uint32_t c2 = 0; c2 < 8u; ++c2) {
                    dst0[256u + d_off + c2] = s_bot[0u  + c2];
                    dst0[384u + d_off + c2] = s_bot[8u  + c2];
                    dst1[256u + d_off + c2] = s_bot[16u + c2];
                    dst1[384u + d_off + c2] = s_bot[24u + c2];
                }
            }
        }

        const uint32_t out_base = b * 160u + 2u * core_i;
        noc_async_write_tile(out_base + 0u, out_acc, tile0_l1);
        noc_async_write_tile(out_base + 1u, out_acc, tile1_l1);
        noc_async_writes_flushed();
    }
    noc_async_write_barrier();
}
