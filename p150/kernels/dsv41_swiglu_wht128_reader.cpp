// Blackhole P150A Layer-Agnostic SwiGLU(10.0) + WHT128 Reader (RISCV_1 / NOC_0)
// Batches all k_splits (10) gate + up partial tile reads per (dm, nt_sub) behind ONE barrier (10x fewer barriers!).

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t gate_addr      = get_arg_val<uint32_t>(0);
    const uint32_t up_addr        = get_arg_val<uint32_t>(1);
    const uint32_t slot_ids_addr  = get_arg_val<uint32_t>(2);
    const uint32_t nb4            = get_arg_val<uint32_t>(3);  // 0..8
    const uint32_t k_splits       = get_arg_val<uint32_t>(4);  // 10
    const uint32_t m_tiles_total  = get_arg_val<uint32_t>(5);  // 16
    const uint32_t m_start        = get_arg_val<uint32_t>(6);
    const uint32_t m_cnt          = get_arg_val<uint32_t>(7);
    const uint32_t slot_ids_bytes = get_arg_val<uint32_t>(8);  // 640 B

    constexpr auto g_args = TensorAccessorArgs<0>();
    constexpr auto u_args = TensorAccessorArgs<g_args.next_compile_time_args_offset(), g_args.next_common_runtime_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<u_args.next_compile_time_args_offset(), u_args.next_common_runtime_args_offset()>();
    constexpr auto i_args = TensorAccessorArgs<w_args.next_compile_time_args_offset(), w_args.next_common_runtime_args_offset()>();

    const auto gate_acc = TensorAccessor(g_args, gate_addr, 4096);
    const auto up_acc   = TensorAccessor(u_args, up_addr, 4096);
    const auto ids_acc  = TensorAccessor(i_args, slot_ids_addr, slot_ids_bytes);

    cb_reserve_back(2, 16);
    const uint32_t wht_l1 = get_write_ptr(2);

    // Fetch 64-byte layer_desc from slot_ids[576..639] to extract dynamic wht_q_i DRAM address (ldesc[4])!
    noc_async_read(ids_acc.get_noc_addr(0, 576u), wht_l1, 64u);
    noc_async_read_barrier();
    const uint32_t wht_addr = reinterpret_cast<const volatile uint32_t*>(wht_l1)[4];
    const auto wht_acc = TensorAccessor(w_args, wht_addr, 2048);

    // Load 4x4 = 16 Float16_b tiles of WHT128 Q_i[nb4] into CB 2 once for all m_cnt slots!
    const uint32_t wht_base_tile = nb4 * 16u;
    for (uint32_t t = 0; t < 16u; ++t) {
        noc_async_read_tile(wht_base_tile + t, wht_acc, wht_l1 + t * 2048u);
    }
    noc_async_read_barrier();
    cb_push_back(2, 16);

    constexpr uint32_t nt_total = 36u;
    const uint32_t ks_stride = m_tiles_total * nt_total;
    for (uint32_t dm = 0; dm < m_cnt; ++dm) {
        const uint32_t m_base = (m_start + dm) * nt_total + nb4 * 4u;
        for (uint32_t nt_sub = 0; nt_sub < 4u; ++nt_sub) {
            cb_reserve_back(0, k_splits);
            cb_reserve_back(1, k_splits);
            const uint32_t g_wr = get_write_ptr(0);
            const uint32_t u_wr = get_write_ptr(1);
            uint32_t tile_id = m_base + nt_sub;
            for (uint32_t ks = 0; ks < k_splits; ++ks) {
                noc_async_read_tile(tile_id, gate_acc, g_wr + ks * 4096u);
                noc_async_read_tile(tile_id, up_acc,   u_wr + ks * 4096u);
                tile_id += ks_stride;
            }
            noc_async_read_barrier();
            cb_push_back(0, k_splits);
            cb_push_back(1, k_splits);
        }
    }
}
