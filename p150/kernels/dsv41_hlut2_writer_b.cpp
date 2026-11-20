// Blackhole P150A Wave 9.4 Multi-Tile Weight-Stationary 2-SB Burst Pipelined HLUT2 Weight Feeder & Partial Writer
//   1. Publishes slot_ids via CB 6 (0 B extra L1!) so reader_a and compute share exact run plans.
//   2. Contiguous identical expert_slot runs [b0 .. b0 + run_len - 1] read GDDR6/L1 weights ONCE into CB 1.
//   3. Empty padding slots (expert_slot >= 0xFFFE) are skipped with 0 GDDR6 reads, 0 CB4/5 fills, 0 compute, 0 writes.
//   4. 2-superblock parallel burst pipelining over 4-page CB4/CB5 ring overlaps next 2-SB GDDR6 burst with scalar scale unpack!

#include "api/dataflow/dataflow_api.h"

static inline __attribute__((always_inline)) uint32_t fp16_to_f32_bits_fast(uint32_t h) {
    const uint32_t mag = h & 0x7FFFu;
    const uint32_t sgn = (h & 0x8000u) << 16;
    if (__builtin_expect((mag & 0x7C00u) != 0u && (mag & 0x7C00u) != 0x7C00u, 1)) {
        return sgn | ((mag + 0x1C000u) << 13);
    }
    if (mag == 0u) return sgn;
    if ((mag & 0x7C00u) == 0u) {
        uint32_t e = 113u;
        uint32_t m = mag;
        while ((m & 0x0400u) == 0u) {
            m <<= 1;
            --e;
        }
        return sgn | (e << 23) | ((m & 0x03FFu) << 13);
    }
    return sgn | 0x7F800000u | ((mag & 0x03FFu) << 13);
}

void kernel_main() {
    const uint32_t slot_ids_addr   = get_arg_val<uint32_t>(0);
    const uint32_t partials0_addr  = get_arg_val<uint32_t>(1);
    const uint32_t slot_start      = get_arg_val<uint32_t>(2);
    const uint32_t proj_idx        = get_arg_val<uint32_t>(3);   // 3=fused gate+up, 2=down
    const uint32_t nb4             = get_arg_val<uint32_t>(4);   // 0..8 (gate/up) or 0..39 (down)
    const uint32_t kt4_total       = get_arg_val<uint32_t>(5);   // 40 (gate/up) or 9 (down)
    const uint32_t kb4_start       = get_arg_val<uint32_t>(6);
    const uint32_t num_sb          = get_arg_val<uint32_t>(7);   // <= 5
    const uint32_t ks              = get_arg_val<uint32_t>(8);
    const uint32_t nt_total        = get_arg_val<uint32_t>(9);   // 36 (gate/up) or 160 (down)
    const uint32_t m_tiles_ws      = get_arg_val<uint32_t>(10);  // 16 (workspace slot stride per ks)
    const uint32_t slot_ids_bytes  = get_arg_val<uint32_t>(11);  // 1344 B ([0..1279]=640 slot_ids, [1280..1343]=layer_desc)
    const uint32_t batch_cnt       = get_arg_val<uint32_t>(12);  // 1..16 expert slots per launch
    const uint32_t partials1_addr  = get_arg_val<uint32_t>(13);  // up_partials when proj_idx == 3

    constexpr auto p_args = TensorAccessorArgs<0>();
    constexpr auto c_args = TensorAccessorArgs<p_args.next_compile_time_args_offset(), p_args.next_common_runtime_args_offset()>();
    constexpr auto i_args = TensorAccessorArgs<c_args.next_compile_time_args_offset(), c_args.next_common_runtime_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<i_args.next_compile_time_args_offset(), i_args.next_common_runtime_args_offset()>();
    constexpr auto h_args = TensorAccessorArgs<o_args.next_compile_time_args_offset(), o_args.next_common_runtime_args_offset()>();
    constexpr auto l_args = TensorAccessorArgs<h_args.next_compile_time_args_offset(), h_args.next_common_runtime_args_offset()>();

    const bool fused_gu       = (proj_idx == 3u);
    const uint32_t num_projs  = fused_gu ? 2u : 1u;
    const uint32_t base_proj  = fused_gu ? 0u : proj_idx;

    const auto ids_acc  = TensorAccessor(i_args, slot_ids_addr, slot_ids_bytes);
    const auto out_acc0 = TensorAccessor(o_args, partials0_addr, 4096);
    const auto out_acc1 = TensorAccessor(o_args, partials1_addr, 4096);

    cb_reserve_back(6, 1);
    const uint32_t scratch_ptr = get_write_ptr(6);
    noc_async_read(ids_acc.get_noc_addr(0, 0), scratch_ptr, slot_ids_bytes);
    noc_async_read_barrier();
    cb_push_back(6, 1);

    uint16_t local_eids[24];
    const volatile uint16_t* slot_u16 = reinterpret_cast<const volatile uint16_t*>(scratch_ptr);
    bool any_valid = false;
    for (uint32_t b = 0u; b < batch_cnt; ++b) {
        const uint16_t e = slot_u16[slot_start + b];
        local_eids[b] = e;
        if (e < 0xFFFEu) any_valid = true;
    }
    if (!any_valid) {
        return;
    }

    const volatile uint32_t* ldesc = reinterpret_cast<const volatile uint32_t*>(scratch_ptr + (slot_ids_bytes - 64u));
    const uint32_t centroids_addr  = ldesc[3];
    const uint32_t lut_addr        = ldesc[8];
    const uint32_t layer_idx       = ldesc[9];

    const auto pages_acc0    = TensorAccessor(p_args, ldesc[base_proj], 4352);
    const auto pages_acc1    = TensorAccessor(p_args, ldesc[fused_gu ? 1u : base_proj], 4352);
    const auto centroids_acc = TensorAccessor(c_args, centroids_addr, 64);
    const auto hot_acc0      = TensorAccessor(h_args, ldesc[5u + base_proj], 4352);
    const auto hot_acc1      = TensorAccessor(h_args, ldesc[5u + (fused_gu ? 1u : base_proj)], 4352);
    const auto lut_acc       = TensorAccessor(l_args, lut_addr, 512);

    const uint32_t lut_l1 = scratch_ptr + 2816u;
    noc_async_read(lut_acc.get_noc_addr(layer_idx, 0), lut_l1, 512u);
    noc_async_read_barrier();
    const volatile uint16_t* hot_lut = reinterpret_cast<const volatile uint16_t*>(lut_l1);

    const uint32_t local_sb_off  = nb4 * kt4_total + kb4_start;
    const uint32_t cent_l1       = scratch_ptr + slot_ids_bytes;
    const uint32_t raw_scales_0  = cent_l1 + 64u;
    const uint32_t col_tile_base = nb4 * 4u;

    cb_reserve_back(4, 1);
    cb_reserve_back(5, 1);
    const uint32_t cb4_base = get_write_ptr(4);
    const uint32_t cb5_base = get_write_ptr(5);
    uint32_t cb_wr_idx = 0u;
    bool page_tail_zeroed[4] = {false, false, false, false};

    auto drain_out_step = [&](uint32_t b_done, uint32_t p_done) __attribute__((always_inline)) {
        cb_wait_front(16, 4);
        const uint32_t rd_l1 = get_read_ptr(16);
        const uint32_t out_tile_base = (ks * m_tiles_ws + b_done) * nt_total + col_tile_base;
        if (p_done == 0u) {
            for (uint32_t n = 0; n < 4u; ++n) {
                noc_async_write(rd_l1 + n * 4096u, out_acc0.get_noc_addr(out_tile_base + n, 0), 4096u);
            }
        } else {
            for (uint32_t n = 0; n < 4u; ++n) {
                noc_async_write(rd_l1 + n * 4096u, out_acc1.get_noc_addr(out_tile_base + n, 0), 4096u);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(16, 4);
    };

    uint32_t prev_b = 0u;
    uint32_t prev_p = 0u;
    bool has_prev   = false;

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

        const uint16_t hs = hot_lut[expert_slot];
        const int32_t hot_slot = (hs != 0xFFFFu) ? static_cast<int32_t>(hs) : -1;
        const uint32_t dram_sb_base = expert_slot * 360u + local_sb_off;
        const uint32_t hot_sb_base  = (hot_slot >= 0) ? (uint32_t(hot_slot) * 360u + local_sb_off) : 0u;

        for (uint32_t p = 0u; p < num_projs; ++p) {
            const uint32_t cur_proj = base_proj + p;
            uint64_t c0_u64 = 0ULL;
            uint64_t dc_u64 = 0ULL;

            auto issue_burst = [&](uint32_t sb_start, uint32_t cnt, uint32_t ring_pos) __attribute__((always_inline)) {
                if (p == 0u && sb_start == 0u) {
                    noc_async_read(centroids_acc.get_noc_addr(expert_slot, 0), cent_l1, 64u);
                }
                for (uint32_t c = 0u; c < cnt; ++c) {
                    const uint32_t sb = sb_start + c;
                    const uint32_t codes_ptr = cb4_base + ((ring_pos + c) & 3u) * 4096u;
                    if (p == 0u) {
                        if (hot_slot >= 0) {
                            const uint32_t hot_pid = hot_sb_base + sb;
                            noc_async_read(hot_acc0.get_noc_addr(hot_pid, 0), codes_ptr, 4096u);
                            noc_async_read(hot_acc0.get_noc_addr(hot_pid, 4096u), raw_scales_0 + sb * 256u, 256u);
                        } else {
                            const uint32_t page_id = dram_sb_base + sb;
                            noc_async_read(pages_acc0.get_noc_addr(page_id, 0), codes_ptr, 4096u);
                            noc_async_read(pages_acc0.get_noc_addr(page_id, 4096u), raw_scales_0 + sb * 256u, 256u);
                        }
                    } else {
                        if (hot_slot >= 0) {
                            const uint32_t hot_pid = hot_sb_base + sb;
                            noc_async_read(hot_acc1.get_noc_addr(hot_pid, 0), codes_ptr, 4096u);
                            noc_async_read(hot_acc1.get_noc_addr(hot_pid, 4096u), raw_scales_0 + sb * 256u, 256u);
                        } else {
                            const uint32_t page_id = dram_sb_base + sb;
                            noc_async_read(pages_acc1.get_noc_addr(page_id, 0), codes_ptr, 4096u);
                            noc_async_read(pages_acc1.get_noc_addr(page_id, 4096u), raw_scales_0 + sb * 256u, 256u);
                        }
                    }
                }
            };

            // 2-superblock parallel burst pipelining over 4-page CB4/CB5 ring!
            const uint32_t chunk0 = (num_sb >= 2u) ? 2u : num_sb;
            cb_reserve_back(4, chunk0);
            cb_reserve_back(5, chunk0);
            issue_burst(0u, chunk0, cb_wr_idx);
            noc_async_read_barrier();

            uint32_t sb0 = 0u;
            uint32_t cur_chunk = chunk0;
            while (sb0 < num_sb) {
                if (sb0 == 0u) {
                    const volatile uint32_t* cent_u32 = reinterpret_cast<const volatile uint32_t*>(cent_l1);
                    const uint32_t c0_u32 = cent_u32[cur_proj * 2u + 0u];
                    const uint32_t dc_u32 = cent_u32[cur_proj * 2u + 1u];
                    c0_u64 = uint64_t(c0_u32) | (uint64_t(c0_u32) << 32);
                    dc_u64 = uint64_t(dc_u32) | (uint64_t(dc_u32) << 32);
                }

                const uint32_t next_sb0 = sb0 + cur_chunk;
                const uint32_t next_rem = (next_sb0 < num_sb) ? (num_sb - next_sb0) : 0u;
                const uint32_t next_chunk = (next_rem >= 2u) ? 2u : next_rem;
                if (next_chunk > 0u) {
                    cb_reserve_back(4, cur_chunk + next_chunk);
                    cb_reserve_back(5, cur_chunk + next_chunk);
                    issue_burst(next_sb0, next_chunk, (cb_wr_idx + cur_chunk) & 3u);
                }

                for (uint32_t c = 0u; c < cur_chunk; ++c) {
                    const uint32_t sb = sb0 + c;
                    const uint32_t pg = (cb_wr_idx + c) & 3u;
                    const volatile uint32_t* hs32 = reinterpret_cast<const volatile uint32_t*>(raw_scales_0 + sb * 256u);
                    uint64_t* su64 = reinterpret_cast<uint64_t*>(cb5_base + pg * 4096u);

                    for (uint32_t nt_sub = 0; nt_sub < 4u; ++nt_sub) {
                        uint64_t* f0_row = su64 + (nt_sub * 4u) * 8u;
                        uint64_t* f1_row = su64 + 128u + (nt_sub * 4u) * 8u;
                        const volatile uint32_t* hs_lo32 = hs32 + nt_sub * 16u;
                        const volatile uint32_t* hs_hi32 = hs_lo32 + 8u;
                        #pragma GCC unroll 8
                        for (uint32_t c2 = 0; c2 < 8u; ++c2) {
                            const uint32_t plo = hs_lo32[c2];
                            const uint32_t phi = hs_hi32[c2];
                            const uint64_t ulo = uint64_t(fp16_to_f32_bits_fast(plo & 0xFFFFu)) |
                                                 (uint64_t(fp16_to_f32_bits_fast(plo >> 16)) << 32);
                            const uint64_t uhi = uint64_t(fp16_to_f32_bits_fast(phi & 0xFFFFu)) |
                                                 (uint64_t(fp16_to_f32_bits_fast(phi >> 16)) << 32);
                            f0_row[0u  + c2] = ulo;
                            f0_row[8u  + c2] = ulo;
                            f0_row[16u + c2] = ulo;
                            f0_row[24u + c2] = ulo;
                            f1_row[0u  + c2] = uhi;
                            f1_row[8u  + c2] = uhi;
                            f1_row[16u + c2] = uhi;
                            f1_row[24u + c2] = uhi;
                        }
                    }

                    uint64_t* f2 = su64 + 256u;
                    for (uint32_t idx = 0; idx < 32u; ++idx) {
                        f2[idx]       = c0_u64;
                        f2[32u + idx] = dc_u64;
                    }
                    if (!page_tail_zeroed[pg]) {
                        for (uint32_t idx = 320u; idx < 512u; ++idx) {
                            su64[idx] = 0ULL;
                        }
                        page_tail_zeroed[pg] = true;
                    }
                }

                cb_push_back(4, cur_chunk);
                cb_push_back(5, cur_chunk);
                cb_wr_idx = (cb_wr_idx + cur_chunk) & 3u;

                if (next_chunk > 0u) {
                    noc_async_read_barrier();
                }
                sb0 = next_sb0;
                cur_chunk = next_chunk;
            }

            if (has_prev) {
                drain_out_step(prev_b, prev_p);
            }
            for (uint32_t r = 0u; r + 1u < run_len; ++r) {
                drain_out_step(b0 + r, p);
            }
            prev_b   = b0 + run_len - 1u;
            prev_p   = p;
            has_prev = true;
        }

        b0 += run_len;
    }

    if (has_prev) {
        drain_out_step(prev_b, prev_p);
        noc_async_write_barrier();
    }
}
