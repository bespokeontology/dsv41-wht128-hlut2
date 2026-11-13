// Blackhole P150A Expert-Compact K-Split + Scatter-Accumulate Writer (RISCV_0 / NOC_1 across 80 cores)
// Untilizes the cnt <= 32 active rows from the 2 updated 32x32 FP32 tiles in CB 16 and scatters 256 B/row
// back into y_accum_f32[row_idx[r], 64*core_i .. 64*core_i+63] (0 writes when cnt == 0 on padding slots!).

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_accum_addr   = get_arg_val<uint32_t>(0);
    const uint32_t slot_meta_addr = get_arg_val<uint32_t>(1);
    const uint32_t slot_start     = get_arg_val<uint32_t>(2);
    const uint32_t core_i         = get_arg_val<uint32_t>(3);  // 0..79
    const uint32_t batch_cnt      = get_arg_val<uint32_t>(4);  // 1..12

    constexpr auto y_args = TensorAccessorArgs<0>();
    constexpr auto m_args = TensorAccessorArgs<y_args.next_compile_time_args_offset(), y_args.next_common_runtime_args_offset()>();

    const auto y_acc    = TensorAccessor(y_args, y_accum_addr, 4096);
    const auto meta_acc = TensorAccessor(m_args, slot_meta_addr, 128);

    // CB 6: 8448 B scratch ([0..127]=meta, [256..8447]=scatter_buf 32x256B)
    cb_reserve_back(6, 1);
    const uint32_t meta_l1    = get_write_ptr(6);
    const uint32_t scatter_l1 = meta_l1 + 256u;
    const uint32_t page_grp   = core_i >> 4;
    const uint32_t byte_off   = (core_i & 15u) * 256u;

    for (uint32_t b = 0; b < batch_cnt; ++b) {
        const uint32_t slot_idx = slot_start + b;
        noc_async_read(meta_acc.get_noc_addr(slot_idx, 0), meta_l1, 128u);
        noc_async_read_barrier();

        const volatile uint32_t* meta_u32 = reinterpret_cast<const volatile uint32_t*>(meta_l1);
        const uint32_t cnt = meta_u32[0];
        const volatile uint16_t* row_idx = reinterpret_cast<const volatile uint16_t*>(meta_l1 + 8u);

        cb_wait_front(16, 2);
        if (cnt > 0u) {
            const uint32_t rd_l1 = get_read_ptr(16);
            const uint32_t* tile0 = reinterpret_cast<const uint32_t*>(rd_l1);
            const uint32_t* tile1 = reinterpret_cast<const uint32_t*>(rd_l1 + 4096u);

            for (uint32_t r = 0; r < cnt; ++r) {
                uint32_t* row_out = reinterpret_cast<uint32_t*>(scatter_l1 + r * 256u);
                const uint32_t f_base = ((r >= 16u) ? 512u : 0u) + (r & 15u) * 16u;
                for (uint32_t c = 0; c < 16u; ++c) {
                    row_out[0u  + c] = tile0[f_base + c];
                    row_out[16u + c] = tile0[f_base + 256u + c];
                    row_out[32u + c] = tile1[f_base + c];
                    row_out[48u + c] = tile1[f_base + 256u + c];
                }
                const uint32_t dst_row = row_idx[r];
                noc_async_write(scatter_l1 + r * 256u, y_acc.get_noc_addr(dst_row * 5u + page_grp, byte_off), 256u);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(16, 2);

        if (b + 1u < batch_cnt) {
            cb_reserve_back(7, 1);
            cb_push_back(7, 1);
        }
    }
}
