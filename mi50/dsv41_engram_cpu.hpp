#pragma once
// Pure-C++17 Exact Engram N-Gram Hash State + Pre-Faulted 189 GiB DDR4 Gather & AVX2 Dequantizer
// Matches hf_meta/inference/engram.py & model.py:296-366 100% bit-for-bit.
// Speed Wave 3:
//   - Concurrent Dual-NVMe (NVMe0 + NVMe1) startup pre-fault into 384 GiB DDR4
//   - Small-M OpenMP gate (seqlen=1 inline, seqlen<=6 4 threads, seqlen>6 full socket) + _mm_prefetch + AVX2 RNE
//   - Direct seq_cache_ snapshot/restore accessors for RewindBank & RadixBlockCache

#include "../src/safetensors_mmap.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dsv41 {

struct Dsv41EngramConfig {
    static constexpr int kNumEngramLayers = 2;
    static constexpr int kLayerIds[2] = {1, 14};
    static constexpr int64_t kExpectedNumEmbeddings[2] = {384006168LL, 384016682LL};
    static constexpr int kMaxNgramSize = 4;
    static constexpr int kNumHeads = 8;
    static constexpr int kNumHashCols = (kMaxNgramSize - 1) * kNumHeads;  // 24
    static constexpr int kColsPerTp = kNumHashCols / 3;                    // 8
    static constexpr int kHeadDim = 256;
    static constexpr int kEmbedDimPerTp = kColsPerTp * kHeadDim;           // 2048
    static constexpr int64_t kBaseVocabSize = 16000000LL;
    static constexpr int32_t kPadTokenId = 2;
    static constexpr int32_t kExpectedCompressedVocab = 99092;
};

class Dsv41EngramEngine {
public:
    Dsv41EngramEngine() = default;
    ~Dsv41EngramEngine() = default;

    void init_hash_tables(const std::vector<int32_t>& compressed_token_map, int32_t compressed_vocab_size, int max_seq_len = 262144);

    void open_table_shard(int engram_idx, const std::string& safetensors_path);

    // Hydrates all 189 GiB of Shard 47 + Shard 48 weight_fp8 & scale_ue8m0 pages into DDR4 concurrently across both NVMe drives!
    void prefault_all_tables();

    bool has_table_shard(int engram_idx) const {
        return engram_idx >= 0 && engram_idx < 2 && tables_[engram_idx].weight_fp8 != nullptr;
    }

    void compute_hashes(
        const int32_t* token_ids,
        int seqlen,
        int start_pos,
        const uint8_t* image_mask,
        std::vector<int64_t>& out_hashes);

    // Causal-tree 4-gram hash evaluation for M <= 8 speculative tree nodes without mutating seq_cache_ with cousins
    void compute_hashes_tree(
        const int32_t* token_ids,
        const int32_t* parent_ids,
        const uint32_t* depths,
        int M,
        int start_pos,
        std::vector<int64_t>& out_hashes);

    // Commit verified branch tokens [0..count-1] into seq_cache_[start_pos .. start_pos + count - 1]
    void commit_accepted_tokens(
        const int32_t* accepted_tokens,
        int count,
        int start_pos);

    void gather_dequant_tp3(
        int engram_idx,
        const int64_t* all_hashes,
        int seqlen,
        const uint8_t* image_mask,
        uint16_t* out_tp0_bf16,
        uint16_t* out_tp1_bf16,
        uint16_t* out_tp2_bf16) const;

    void ensure_seq_cache_capacity(int min_len) {
        if (min_len > int(seq_cache_.size())) {
            seq_cache_.resize(size_t(min_len + 4096), pad_compressed_id_);
        }
    }
    const int32_t* seq_cache_data() const { return seq_cache_.data(); }
    int32_t* seq_cache_data() { return seq_cache_.data(); }
    int32_t pad_compressed_id() const { return pad_compressed_id_; }

    const std::array<std::array<int64_t, 4>, 2>& multipliers() const { return multipliers_; }
    const std::array<std::array<int64_t, 24>, 2>& primes() const { return primes_; }
    const std::array<std::array<int64_t, 24>, 2>& offsets() const { return offsets_; }

private:
    struct MmapTable {
        std::unique_ptr<SafetensorsMmap> st;
        const uint8_t* weight_fp8 = nullptr;   // [num_embeddings, 256] F8_E4M3
        const uint8_t* scale_ue8m0 = nullptr;  // [num_embeddings, 8]   F8_E8M0
        size_t weight_bytes = 0;
        size_t scale_bytes = 0;
        int64_t rows = 0;
    };

    std::vector<int32_t> token_map_;
    int32_t pad_compressed_id_ = 0;
    std::array<std::array<int64_t, 24>, 2> primes_{};
    std::array<std::array<int64_t, 24>, 2> offsets_{};
    std::array<std::array<int64_t, 4>, 2> multipliers_{};
    alignas(32) double barrett_p_[2][24]{};
    alignas(32) double barrett_inv_p_[2][24]{};
    alignas(32) double barrett_r21_[2][24]{};
    alignas(32) double barrett_r42_[2][24]{};
    alignas(32) double barrett_r63_neg_[2][24]{};
    alignas(32) double barrett_off_[2][24]{};
    std::vector<int32_t> seq_cache_;
    std::array<MmapTable, 2> tables_;
};

}  // namespace dsv41
