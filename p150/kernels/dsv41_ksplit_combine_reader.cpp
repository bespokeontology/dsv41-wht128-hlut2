// Blackhole P150A down_proj K-Split Reduction + Weighted Expert Combine Reader (RISCV_1 / NOC_0)
// Streams k_splits FP32 partial tiles (CB 0), per-row router weight tile (CB 1),
// and optional previous expert accumulator tile (CB 2).

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t partials_addr   = get_arg_val<uint32_t>(0);
    const uint32_t weights_addr    = get_arg_val<uint32_t>(1);
    const uint32_t combined_addr   = get_arg_val<uint32_t>(2);
    const uint32_t slot_start      = get_arg_val<uint32_t>(3);
    const uint32_t k_splits        = get_arg_val<uint32_t>(4);  // 2
    const uint32_t m_tiles_ws      = get_arg_val<uint32_t>(5);  // 12
    const uint32_t first_tile      = get_arg_val<uint32_t>(6);
    const uint32_t tile_cnt        = get_arg_val<uint32_t>(7);
    const uint32_t accumulate_prev = get_arg_val<uint32_t>(8);  // 0 on first batch, 1 thereafter
    const uint32_t batch_cnt       = get_arg_val<uint32_t>(9);  // 1..12

    constexpr auto p_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<p_args.next_compile_time_args_offset(), p_args.next_common_runtime_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<w_args.next_compile_time_args_offset(), w_args.next_common_runtime_args_offset()>();

    const auto part_acc = TensorAccessor(p_args, partials_addr, 4096);
    const auto wgt_acc  = TensorAccessor(w_args, weights_addr, 4096);
    const auto comb_acc = TensorAccessor(c_args, combined_addr, 4096);

    constexpr uint32_t nt_total = 160u;
    for (uint32_t i = 0; i < tile_cnt; ++i) {
        const uint32_t out_tile = first_tile + i;
        const uint32_t nt = out_tile;

        for (uint32_t b = 0; b < batch_cnt; ++b) {
            for (uint32_t ks = 0; ks < k_splits; ++ks) {
                cb_reserve_back(0, 1);
                const uint32_t p_tile = (ks * m_tiles_ws + b) * nt_total + nt;
                noc_async_read_tile(p_tile, part_acc, get_write_ptr(0));
                noc_async_read_barrier();
                cb_push_back(0, 1);
            }

            cb_reserve_back(1, 1);
            noc_async_read_tile(slot_start + b, wgt_acc, get_write_ptr(1));
            noc_async_read_barrier();
            cb_push_back(1, 1);
        }

        if (accumulate_prev != 0u) {
            cb_reserve_back(2, 1);
            noc_async_read_tile(out_tile, comb_acc, get_write_ptr(2));
            noc_async_read_barrier();
            cb_push_back(2, 1);
        }
    }
}
