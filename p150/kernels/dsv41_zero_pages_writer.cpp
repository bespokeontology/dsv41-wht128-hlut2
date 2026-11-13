// Blackhole P150A L1 Page Zero-Initializer (RISCV_0 / NOC_1 across 80 cores)
// Zeroes [first_page .. first_page + page_cnt - 1] 4096 B pages of y_accum_f32 in ~1.5 us.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr   = get_arg_val<uint32_t>(0);
    const uint32_t first_page = get_arg_val<uint32_t>(1);
    const uint32_t page_cnt   = get_arg_val<uint32_t>(2);

    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto dst_acc = TensorAccessor(dst_args, dst_addr, 4096);

    cb_reserve_back(0, 1);
    const uint32_t zero_l1 = get_write_ptr(0);
    uint32_t* zp = reinterpret_cast<uint32_t*>(zero_l1);
    for (uint32_t i = 0; i < 1024u; ++i) {
        zp[i] = 0u;
    }

    for (uint32_t p = 0; p < page_cnt; ++p) {
        noc_async_write_tile(first_page + p, dst_acc, zero_l1);
    }
    noc_async_write_barrier();
}
