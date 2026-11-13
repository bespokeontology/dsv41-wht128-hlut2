#pragma once
// Pure C++20 2x Tenstorrent Blackhole P150A TP2 HLUT2 MoE Engine for DeepSeek-V4.1-Flash-1017
// Zero Python anywhere. Executes routed experts [216..383] (168 experts/layer x 40 layers, I=1152/card)
// using Two-Phase cb_w_b8 SFPU Pre-Unpack + FP32 Dest Matmul + L1 SwiGLU(10.0) + L1 WHT128 + FABRIC_1D TP2 AllReduce.

#include "dsv41_p150_shm.hpp"
#include <cstdint>
#include <memory>
#include <string>

namespace dsv41 {

class P150Dsv41Engine {
public:
    P150Dsv41Engine(
        const std::string& packed_p150_root,
        const std::string& kernels_dir,
        bool enable_trace = true,
        const std::string& rarebal_overlay_path = "");
    ~P150Dsv41Engine();

    P150Dsv41Engine(const P150Dsv41Engine&) = delete;
    P150Dsv41Engine& operator=(const P150Dsv41Engine&) = delete;

    // Loads layer [0..39] if tp0 & tp1 .safetensors exist on disk and not yet loaded. Returns true if resident.
    bool ensure_layer_loaded(uint32_t layer);

    // Scans layers [0..max_layers-1] and loads all available shards. Returns count of resident layers.
    uint32_t load_available_layers(uint32_t max_layers = 40u);

    uint64_t loaded_layers_mask() const noexcept;
    size_t persistent_l1_bytes_per_card() const noexcept;

    // Executes MoE forward for `rows` (1..512) on `layer` (0..39).
    // Writes TP2-summed BF16 [rows, 5120] into `y_out_bf16` and returns device execution time in ns.
    uint64_t forward_moe(
        uint32_t layer,
        uint32_t rows,
        bool input_is_wht_rotated,
        const int32_t row_slots[][DSV41_P150_MAX_SLOTS],
        const float row_weights[][DSV41_P150_MAX_SLOTS],
        const uint16_t* x_in_bf16,
        uint16_t* y_out_bf16);

    // Runs silicon verification on a loaded layer (M=1 decode, M=6 DSpark, M=64 prefill) vs exact CPU HLUT2 reference.
    bool selfcheck_layer(uint32_t layer = 0u);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dsv41
