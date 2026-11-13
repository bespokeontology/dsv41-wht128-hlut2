// Blackhole P150A down_proj K-Split Reduction + Weighted Expert Combine Writer (RISCV_0 / NOC_1)
// Writes updated FP32 combined_f32 tiles from CB 16 back to L1 combined_f32.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t combined_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_tile    = get_arg_val<uint32_t>(1);
    const uint32_t tile_cnt      = get_arg_val<uint32_t>(2);

    constexpr auto c_args = TensorAccessorArgs<0>();
    const auto comb_acc = TensorAccessor(c_args, combined_addr, 4096);

    for (uint32_t i = 0; i < tile_cnt; ++i) {
        cb_wait_front(16, 1);
        noc_async_write_tile(first_tile + i, comb_acc, get_read_ptr(16));
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
