// Blackhole P150A Wave 9.4 Activation Strip Reader (RISCV_1 / NOC_0)
// Streams [M_tiles, Kt_core] Float16_b (BF16, 2048 B/tile) activation tiles from L1/DRAM act_tilized into CB 0.
// Wrap-safe rt_dim=2 double-strip streaming for gate+up (num_projs==2, CB 0 = 2*kt_core):
//   - Always reserves/pushes kt_core tiles at a time (0 mid-CB wrap overflow ever!).
//   - Tracks 1-bit cb0_half so rt_dim=2 runs ONLY when [0..kt_core-1] and [kt_core..2*kt_core-1] are contiguous!
//   - When run_len==2 and cb0_half==0, reads BOTH M-tiles ONCE across gate AND up (2x less x_tilized GDDR6 BW!).

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t act_addr        = get_arg_val<uint32_t>(0);
    const uint32_t kt_total        = get_arg_val<uint32_t>(1);  // 160 for gate/up, 36 for down
    const uint32_t kt_start        = get_arg_val<uint32_t>(2);  // kb4_start * 4
    const uint32_t kt_core         = get_arg_val<uint32_t>(3);  // num_sb * 4 (16 for gate/up, 16..20 for down)
    const uint32_t batch_cnt       = get_arg_val<uint32_t>(4);  // 1..16 expert slots per launch
    const uint32_t act_slot_stride = get_arg_val<uint32_t>(5);  // 0 for decode gate+up, kt_total otherwise
    const uint32_t slot_start      = get_arg_val<uint32_t>(6);  // batch offset into slot_ids
    const uint32_t num_projs       = get_arg_val<uint32_t>(7);  // 2 for fused gate+up, 1 for down
    (void)kt_total;

    constexpr auto act_args = TensorAccessorArgs<0>();
    const auto act_acc = TensorAccessor(act_args, act_addr, 2048);

    cb_wait_front(6, 1);
    const volatile uint16_t* slot_u16 = reinterpret_cast<const volatile uint16_t*>(get_read_ptr(6));
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

    auto push_strip1 = [&](uint32_t b_idx) __attribute__((always_inline)) {
        cb_reserve_back(0, kt_core);
        const uint32_t dst_base = get_write_ptr(0);
        const uint32_t row_tile_base = b_idx * act_slot_stride + kt_start;
        for (uint32_t k = 0u; k < kt_core; ++k) {
            noc_async_read_tile(row_tile_base + k, act_acc, dst_base + k * 2048u);
        }
        noc_async_read_barrier();
        cb_push_back(0, kt_core);
    };

    if (act_slot_stride == 0u) {
        push_strip1(0u);
        return;
    }

    const bool can_rt2 = (num_projs == 2u);
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

        if (run_len == 1u) {
            push_strip1(b0);
            if (can_rt2) cb0_half ^= 1u;
        } else if (can_rt2 && run_len == 2u && cb0_half == 0u) {
            // Contiguous [0..kt_core-1] + [kt_core..2*kt_core-1] reused across BOTH gate & up!
            push_strip1(b0);
            push_strip1(b0 + 1u);
            // compute pops 2*kt_core -> cb0_half remains 0!
        } else if (can_rt2) {
            for (uint32_t p = 0u; p < num_projs; ++p) {
                uint32_t r = 0u;
                while (r < run_len) {
                    if (cb0_half == 0u && r + 2u <= run_len) {
                        push_strip1(b0 + r);
                        push_strip1(b0 + r + 1u);
                        r += 2u;
                    } else {
                        push_strip1(b0 + r);
                        cb0_half ^= 1u;
                        r += 1u;
                    }
                }
            }
        } else {
            for (uint32_t r = 0u; r < run_len; ++r) {
                push_strip1(b0 + r);
            }
        }
        b0 += run_len;
    }
}
