// Blackhole P150A Activation Strip Reader (RISCV_1 / NOC_0)
// Streams [M_tiles, Kt_core] Float16_b (BF16, 2048 B/tile) activation tiles from L1 act_tilized into CB 0.
// When act_slot_stride == 0 (decode gate+up), reads x_tilized ONCE per launch instead of batch_cnt times!

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t act_addr        = get_arg_val<uint32_t>(0);
    const uint32_t kt_total        = get_arg_val<uint32_t>(1);  // 160 for gate/up, 36 for down
    const uint32_t kt_start        = get_arg_val<uint32_t>(2);  // kb4_start * 4
    const uint32_t kt_core         = get_arg_val<uint32_t>(3);  // num_sb * 4 (16 for gate/up, 16..20 for down)
    const uint32_t batch_cnt       = get_arg_val<uint32_t>(4);  // 1..16 expert slots per launch
    const uint32_t act_slot_stride = get_arg_val<uint32_t>(5);  // 0 for decode gate+up, kt_total otherwise
    (void)kt_total;

    constexpr auto act_args = TensorAccessorArgs<0>();
    const auto act_acc = TensorAccessor(act_args, act_addr, 2048);

    const uint32_t iters = (act_slot_stride == 0u) ? (batch_cnt > 0u ? 1u : 0u) : batch_cnt;
    for (uint32_t b = 0; b < iters; ++b) {
        cb_reserve_back(0, kt_core);
        const uint32_t dst_base = get_write_ptr(0);
        const uint32_t row_tile_base = b * act_slot_stride + kt_start;
        for (uint32_t k = 0; k < kt_core; ++k) {
            noc_async_read_tile(row_tile_base + k, act_acc, dst_base + k * 2048u);
        }
        noc_async_read_barrier();
        cb_push_back(0, kt_core);
    }
}
