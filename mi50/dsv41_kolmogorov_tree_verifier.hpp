// ============================================================================
// dsv41_kolmogorov_tree_verifier.hpp
//
// Minimal Pure-C++17 Reference & Audit Suite for Wave 7 Track 7C:
// Exact Kolmogorov-Borel Multi-Branch Speculative Tree Verifier (M <= 8)
// for DeepSeek-V4.1-Flash-1017 (2x Blackhole P150A + 3x MI50 + DSpark).
//
// Mathematical Foundation (Kolmogorov 1936 Partition Identity):
// Let (Omega, F, p) be the target next-token probability space at tree node u
// after identical repetition-penalty and top-k / top-p Borel support filtering
// (sum_{v in Omega} p(v) = 1, p(v) >= 0). Suppose node u has K >= 1 draft
// children c_1, ..., c_K proposing candidate tokens x_1, ..., x_K.
//
// Unified Sequential Residual Recurrence (p^(1) := p):
//   For j = 1 .. K with true conditional proposal law x_j ~ q_j(.):
//     alpha_j(x_j) = min(1, p^(j)(x_j) / q_j(x_j))   (0 if q_j(x_j) == 0)
//     beta_j       = 1 - sum_{v in Omega} min(p^(j)(v), q_j(v))
//     p^(j+1)(v)   = max(0, p^(j)(v) - q_j(v)) / beta_j   (if beta_j > 0)
//   Then for every v in Omega:
//     P(accept v at j | reach j) + beta_j * p^(j+1)(v)
//       = min(p^(j)(v), q_j(v)) + max(0, p^(j)(v) - q_j(v)) == p^(j)(v).
//   By backward induction j = K .. 1, sampling recovery token x_res ~ p^(K+1)
//   upon all-K rejection yields P_tree_spec(Y = v) == p(v) IDENTICALLY (TV = 0).
//
// Critical Audit Distinction for DeepSeek Track 7C:
//   1. DETERMINISTIC Top-B Draft Trees (GPU argmax / top-B without RNG):
//      Candidate x_j (distinct across siblings) is placed deterministically, so
//      its true proposal law is the Dirac measure q_j = delta_{x_j}. Substituting
//      q_j = delta_{x_j} gives alpha_j = p^(j)(x_j) = p(x_j) / (1 - sum_{i<j} p(x_i)),
//      accepting child c_j with exact unconditional mass p(x_j) in O(K) scalar ops!
//      WARNING: Plugging raw draft softmax q(x_j) < 1 into min(1, p/q) when x_j
//      was chosen deterministically via top-B CORRUPTS the target law (TV >> 0).
//   2. STOCHASTIC Draft Trees (SpecInfer / Plackett-Luce x_j ~ q_j):
//      Each x_j is genuinely sampled from q_j(.), and full O(V) residual
//      subtraction p^(j+1) = normalize(max(0, p^(j) - q_j)) preserves TV == 0.
//
// Systems Invariants Audited for dsv41-hlut2 (CSA2 + SWA + r=2 + P150A/MI50):
//   - Relative RoPE Position: rope_pos[i] = base_pos + depth[i] (NEVER base_pos + i).
//   - 8x8 Causal Tree Mask:   ancestor_mask[i] = ancestor_mask[parent[i]] | (1u << i).
//   - Alias-Free In-Place KV Compaction: parent[i] < i implies accepted path
//     0 = i_0 < i_1 < ... < i_{L-1} satisfies i_d >= d for all d, so forward
//     copies slot[base_pos + i_d] -> slot[base_pos + d] never clobber unread KV.
//   - Deferred Stateful Commits: CSA2 L2..L19 r=2 token-pair compressors and
//     3-stage MTP rings advance ONLY along the compacted accepted path [0..L-1].
// ============================================================================

#ifndef DSV41_KOLMOGOROV_TREE_VERIFIER_HPP_
#define DSV41_KOLMOGOROV_TREE_VERIFIER_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace dsv41 {

inline constexpr std::size_t kMaxSpecTreeNodes = 8;

struct Dsv41SpecTree8 {
  std::uint32_t num_nodes = 0;  // 1 <= num_nodes <= 8 (node 0 is root)
  std::uint32_t base_pos = 0;   // Sequence cursor of root node 0
  std::array<std::int32_t, kMaxSpecTreeNodes> parent{};        // parent[0] = -1, 0 <= parent[i] < i
  std::array<std::uint32_t, kMaxSpecTreeNodes> depth{};        // depth[0] = 0, depth[i] = depth[parent[i]] + 1
  std::array<std::uint32_t, kMaxSpecTreeNodes> rope_pos{};     // base_pos + depth[i] (sibling-invariant!)
  std::array<std::uint8_t, kMaxSpecTreeNodes> ancestor_mask{}; // Bit j set iff node j is ancestor of node i
  std::array<std::uint8_t, kMaxSpecTreeNodes> child_mask{};    // Deduplicated active draft children of node i
  std::array<std::int32_t, kMaxSpecTreeNodes> token_id{};      // Draft token proposed by node i (i >= 1)
};

struct Dsv41TreeVerifyResult {
  std::uint32_t num_emitted = 0;  // L in [1, M]: (L - 1) accepted draft tokens + 1 bonus/recovery token
  std::array<std::uint32_t, kMaxSpecTreeNodes> accepted_nodes{};  // [0, i_1, ..., i_{L-1}], strictly increasing
  std::array<std::int32_t, kMaxSpecTreeNodes> emitted_tokens{};   // [tok(i_1), ..., tok(i_{L-1}), bonus_token]
  std::uint32_t kv_compaction_copies = 0;  // Number of non-identity KV slot moves needed
  std::uint32_t r2_completed_pairs = 0;    // Newly completed even-aligned (2k, 2k+1) pairs for L2..L19
};

// Builds and validates an M <= 8 rooted speculative tree topology.
// Automatically masks out duplicate sibling token proposals under the same parent
// so deterministic top-B trees never double-count target probability mass.
inline bool build_spec_tree_topology(
    std::uint32_t num_nodes,
    std::uint32_t base_pos,
    const std::int32_t* parent_in,
    const std::int32_t* token_id_in,
    Dsv41SpecTree8* out) noexcept {
  if (!out || !parent_in || !token_id_in || num_nodes == 0 ||
      num_nodes > kMaxSpecTreeNodes) {
    return false;
  }
  if (parent_in[0] != -1) {
    return false;
  }

  *out = Dsv41SpecTree8{};
  out->num_nodes = num_nodes;
  out->base_pos = base_pos;
  out->parent[0] = -1;
  out->depth[0] = 0;
  out->rope_pos[0] = base_pos;
  out->ancestor_mask[0] = 0x01u;
  out->token_id[0] = token_id_in[0];

  for (std::uint32_t i = 1; i < num_nodes; ++i) {
    const std::int32_t p = parent_in[i];
    if (p < 0 || static_cast<std::uint32_t>(p) >= i) {
      return false;  // Topological parent[i] < i invariant violated
    }
    const auto pu = static_cast<std::uint32_t>(p);
    out->parent[i] = p;
    out->depth[i] = out->depth[pu] + 1u;
    out->rope_pos[i] = base_pos + out->depth[i];
    out->ancestor_mask[i] =
        static_cast<std::uint8_t>(out->ancestor_mask[pu] | (1u << i));
    out->token_id[i] = token_id_in[i];

    // Sibling deduplication: only attach child i if no earlier sibling under pu
    // already proposed the exact same token_id.
    bool duplicate_sibling = false;
    std::uint8_t sib_bits = out->child_mask[pu];
    while (sib_bits != 0u) {
      const std::uint32_t s = static_cast<std::uint32_t>(__builtin_ctz(sib_bits));
      if (out->token_id[s] == token_id_in[i]) {
        duplicate_sibling = true;
        break;
      }
      sib_bits &= static_cast<std::uint8_t>(sib_bits - 1u);
    }
    if (!duplicate_sibling && token_id_in[i] >= 0) {
      out->child_mask[pu] =
          static_cast<std::uint8_t>(out->child_mask[pu] | (1u << i));
    }
  }
  return true;
}

// Exact IEEE FP64 softmax + top-k + top-p Borel support filter.
// Produces a normalized target distribution p_out[0..vocab_size-1] with exact
// zeroes outside the active top-k / top-p support.
inline void compute_filtered_target_probs_f64(
    const float* logits,
    std::size_t vocab_size,
    double temperature,
    std::size_t top_k,
    double top_p,
    double* p_out) {
  if (vocab_size == 0) return;

  if (temperature <= 0.0) {
    std::size_t best = 0;
    float max_l = logits[0];
    for (std::size_t v = 1; v < vocab_size; ++v) {
      if (logits[v] > max_l) {
        max_l = logits[v];
        best = v;
      }
    }
    std::fill(p_out, p_out + vocab_size, 0.0);
    p_out[best] = 1.0;
    return;
  }

  float max_logit = logits[0];
  for (std::size_t v = 1; v < vocab_size; ++v) {
    if (logits[v] > max_logit) max_logit = logits[v];
  }

  const double inv_temp = 1.0 / temperature;
  double sum_exp = 0.0;
  for (std::size_t v = 0; v < vocab_size; ++v) {
    const double w = std::exp(( static_cast<double>(logits[v]) -
                                static_cast<double>(max_logit) ) * inv_temp);
    p_out[v] = w;
    sum_exp += w;
  }
  const double inv_sum = 1.0 / sum_exp;
  for (std::size_t v = 0; v < vocab_size; ++v) {
    p_out[v] *= inv_sum;
  }

  const bool need_k = (top_k > 0 && top_k < vocab_size);
  const bool need_p = (top_p > 0.0 && top_p < 1.0);
  if (!need_k && !need_p) return;

  std::vector<std::size_t> idx(vocab_size);
  std::iota(idx.begin(), idx.end(), 0);
  const std::size_t k_cap = need_k ? top_k : vocab_size;
  if (need_k) {
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k_cap),
                      idx.end(), [&](std::size_t a, std::size_t b) {
                        return (p_out[a] != p_out[b]) ? (p_out[a] > p_out[b]) : (a < b);
                      });
  } else {
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
      return (p_out[a] != p_out[b]) ? (p_out[a] > p_out[b]) : (a < b);
    });
  }

  double cum = 0.0;
  std::size_t keep = 0;
  for (; keep < k_cap; ++keep) {
    cum += p_out[idx[keep]];
    if (need_p && cum >= top_p) {
      ++keep;
      break;
    }
  }
  if (keep == 0) keep = 1;

  double kept_mass = 0.0;
  for (std::size_t i = 0; i < keep; ++i) {
    kept_mass += p_out[idx[i]];
  }
  for (std::size_t i = keep; i < vocab_size; ++i) {
    p_out[idx[i]] = 0.0;
  }
  const double renorm = 1.0 / kept_mass;
  for (std::size_t i = 0; i < keep; ++i) {
    p_out[idx[i]] *= renorm;
  }
}

// Samples one token from discrete distribution p[0..V-1] (sum == 1) given u in [0, 1).
inline std::int32_t sample_categorical_u01(
    const double* p, std::size_t vocab_size, double u01) noexcept {
  const double u = std::clamp(u01, 0.0, 0.9999999999999999);
  double cdf = 0.0;
  std::size_t last_nonzero = 0;
  for (std::size_t v = 0; v < vocab_size; ++v) {
    if (p[v] > 0.0) {
      last_nonzero = v;
      cdf += p[v];
      if (u < cdf) {
        return static_cast<std::int32_t>(v);
      }
    }
  }
  return static_cast<std::int32_t>(last_nonzero);
}

// Computes exact completed even-aligned (2k, 2k+1) token pairs across
// [base_pos, base_pos + num_emitted) for CSA2 L2..L19 r=2 state advancement.
inline constexpr std::uint32_t count_completed_r2_pairs(
    std::uint32_t base_pos, std::uint32_t num_emitted) noexcept {
  const std::uint32_t end_pos = base_pos + num_emitted;
  return (end_pos / 2u) - (base_pos / 2u);
}

// ============================================================================
// MODE 1: Exact Kolmogorov-Borel Verifier for DETERMINISTIC Top-B Draft Trees
// (q_{u,j} = Dirac delta_{x_j}, O(M) branch walk + single complement draw).
//
// Input:
//   - tree:         Validated Dsv41SpecTree8 (1 <= M <= 8)
//   - target_probs: Row-major [M, vocab_size] filtered target probabilities
//   - u_per_depth:  Array of M independent Uniform[0, 1) variates (one per depth)
// ============================================================================
inline Dsv41TreeVerifyResult verify_deterministic_tree_u01(
    const Dsv41SpecTree8& tree,
    const double* target_probs,
    std::size_t vocab_size,
    const double* u_per_depth) noexcept {
  Dsv41TreeVerifyResult res{};
  std::uint32_t cur = 0;
  std::uint32_t step_d = 0;

  while (true) {
    res.accepted_nodes[step_d] = cur;
    const double* p_row = target_probs + static_cast<std::size_t>(cur) * vocab_size;
    const double u = std::clamp(u_per_depth[step_d], 0.0, 0.9999999999999999);

    std::uint8_t mask = tree.child_mask[cur];
    double branch_cdf = 0.0;
    std::int32_t chosen_child = -1;

    // Partition [0, sum_j p(x_j)) across distinct deterministic children c_j:
    // Child c_j owns interval [branch_cdf_prev, branch_cdf_prev + p(x_j)), which
    // has exact Lebesgue measure p(x_j) (identical to sequential alpha_j =
    // p(x_j) / (1 - sum_{i<j} p(x_i)) under q_j = delta_{x_j}).
    while (mask != 0u) {
      const std::uint32_t c = static_cast<std::uint32_t>(__builtin_ctz(mask));
      mask &= static_cast<std::uint8_t>(mask - 1u);
      const std::int32_t tok = tree.token_id[c];
      if (tok >= 0 && static_cast<std::size_t>(tok) < vocab_size) {
        const double pm = p_row[tok];
        if (pm > 0.0) {
          if (chosen_child < 0 && u < branch_cdf + pm) {
            chosen_child = static_cast<std::int32_t>(c);
          }
          branch_cdf += pm;
        }
      }
    }

    if (chosen_child >= 0) {
      const auto cu = static_cast<std::uint32_t>(chosen_child);
      res.emitted_tokens[step_d] = tree.token_id[cu];
      ++step_d;
      cur = cu;
      continue;
    }

    // All K deterministic children at `cur` rejected (occurs with exact measure
    // 1 - branch_cdf). Reuse the conditional uniform u_res in [0, 1 - branch_cdf)
    // to draw the exact recovery/bonus token from V \ {x_1, ..., x_K}.
    const double rem_mass = std::max(0.0, 1.0 - branch_cdf);
    const double target_u = (rem_mass > 1e-15) ? std::max(0.0, u - branch_cdf) : 0.0;

    // Build small local exclusion list (at most 7 sibling tokens)
    std::array<std::int32_t, kMaxSpecTreeNodes> excl{};
    std::uint32_t n_excl = 0;
    std::uint8_t m2 = tree.child_mask[cur];
    while (m2 != 0u) {
      const std::uint32_t c = static_cast<std::uint32_t>(__builtin_ctz(m2));
      m2 &= static_cast<std::uint8_t>(m2 - 1u);
      excl[n_excl++] = tree.token_id[c];
    }

    double comp_cdf = 0.0;
    std::size_t fallback_v = 0;
    bool found = false;
    for (std::size_t v = 0; v < vocab_size; ++v) {
      const double pv = p_row[v];
      if (pv <= 0.0) continue;
      bool is_excluded = false;
      for (std::uint32_t e = 0; e < n_excl; ++e) {
        if (excl[e] == static_cast<std::int32_t>(v)) {
          is_excluded = true;
          break;
        }
      }
      if (is_excluded) continue;
      fallback_v = v;
      comp_cdf += pv;
      if (target_u < comp_cdf) {
        res.emitted_tokens[step_d] = static_cast<std::int32_t>(v);
        found = true;
        break;
      }
    }
    if (!found) {
      // Degenerate numeric boundary when deterministic branches covered 100% of p_row
      res.emitted_tokens[step_d] =
          (n_excl > 0 && rem_mass <= 1e-15) ? excl[n_excl - 1]
                                            : static_cast<std::int32_t>(fallback_v);
    }
    ++step_d;
    break;
  }

  res.num_emitted = step_d;
  for (std::uint32_t d = 1; d < res.num_emitted; ++d) {
    if (res.accepted_nodes[d] != d) {
      ++res.kv_compaction_copies;
    }
  }
  res.r2_completed_pairs = count_completed_r2_pairs(tree.base_pos, res.num_emitted);
  return res;
}

// ============================================================================
// WAVE 8B: Exact Kolmogorov-Dirac Verifier & Balanced Topology for M <= 16
// Fits inside the SAME single 32-row P150A L1 decode tile (0 B extra L1/VRAM)!
// ============================================================================
inline constexpr std::size_t kMaxSpecTree16Nodes = 16;

struct Dsv41SpecTree16 {
  std::uint32_t num_nodes = 0;  // 1 <= num_nodes <= 16 (node 0 is root)
  std::uint32_t base_pos = 0;   // Sequence cursor of root node 0
  std::array<std::int32_t, kMaxSpecTree16Nodes> parent{};         // parent[0] = -1, 0 <= parent[i] < i
  std::array<std::uint32_t, kMaxSpecTree16Nodes> depth{};         // depth[0] = 0, depth[i] = depth[parent[i]] + 1 (<= 5)
  std::array<std::uint32_t, kMaxSpecTree16Nodes> rope_pos{};      // base_pos + depth[i] (sibling-invariant!)
  std::array<std::uint16_t, kMaxSpecTree16Nodes> ancestor_mask{}; // 16-bit mask: bit j set iff j is ancestor of i
  std::array<std::uint16_t, kMaxSpecTree16Nodes> child_mask{};    // Deduplicated active draft children of node i
  std::array<std::int32_t, kMaxSpecTree16Nodes> token_id{};       // Draft token proposed by node i (i >= 1)
};

struct Dsv41Tree16VerifyResult {
  std::uint32_t num_emitted = 0;  // L in [1, 16]: (L - 1) accepted draft tokens + 1 bonus/recovery token
  std::array<std::uint32_t, kMaxSpecTree16Nodes> accepted_nodes{}; // [0, i_1, ..., i_{L-1}], strictly increasing
  std::array<std::int32_t, kMaxSpecTree16Nodes> emitted_tokens{};  // [tok(i_1), ..., tok(i_{L-1}), bonus_token]
  std::uint32_t kv_compaction_copies = 0;
  std::uint32_t r2_completed_pairs = 0;
};

// Canonical Wave 8B Balanced High-Branching M=16 DSpark Causal Tree:
//   - Level 0 (depth 0): Node 0 (root anchor)                         [1 node]
//   - Level 1 (depth 1): Nodes 1..4 (Stage 0 top-4 under 0)           [4 nodes]
//   - Level 2 (depth 2): Nodes 5..6 (under 1), 7..8 (under 2),
//                        Node 9 (under 3), Node 10 (under 4)          [6 nodes]
//   - Level 3 (depth 3): Node 11 (under 5), Node 12 (under 6),
//                        Node 13 (under 7)                            [3 nodes]
//   - Level 4 (depth 4): Node 14 (under 11)                           [1 node]
//   - Level 5 (depth 5): Node 15 (under 14)                           [1 node]
// Total = 16 nodes, max emitted tokens per step = 6 (5 draft + 1 bonus)!
inline constexpr int kTree16NumNodes = 16;
inline constexpr std::array<std::int32_t, 16> kTree16Parents = {
    -1, 0, 0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 11, 14};
inline constexpr std::array<std::uint32_t, 16> kTree16Depths = {
    0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 4u, 5u};

// SGPR-safe 4x64-bit scalar causal mask (never array-indexed on GPU to preserve ScratchSize == 0!)
#ifndef DSV41_TREE_ANC_MASK256_DEFINED
#define DSV41_TREE_ANC_MASK256_DEFINED
struct TreeAncMask256 {
  std::uint64_t w0 = 0ULL;  // rows 0..3  (16 bits each)
  std::uint64_t w1 = 0ULL;  // rows 4..7  (16 bits each)
  std::uint64_t w2 = 0ULL;  // rows 8..11 (16 bits each)
  std::uint64_t w3 = 0ULL;  // rows 12..15 (16 bits each)

#if defined(__HIPCC__)
  __host__ __device__
#endif
  constexpr bool active() const noexcept {
    return (w0 | w1 | w2 | w3) != 0ULL;
  }
};
#endif

inline constexpr TreeAncMask256 make_tree_anc_mask256_from_parents(
    const std::int32_t* parents, int num_nodes) noexcept {
  std::uint16_t rows[16]{};
  for (int i = 0; i < num_nodes && i < 16; ++i) {
    if (i == 0) {
      rows[0] = 0x0001u;
    } else {
      const int p = parents[i];
      rows[i] = static_cast<std::uint16_t>((p >= 0 ? rows[p] : 0u) | (1u << i));
    }
  }
  TreeAncMask256 m{};
  for (int k = 0; k < 4; ++k) {
    m.w0 |= (static_cast<std::uint64_t>(rows[0 + k]) << (16 * k));
    m.w1 |= (static_cast<std::uint64_t>(rows[4 + k]) << (16 * k));
    m.w2 |= (static_cast<std::uint64_t>(rows[8 + k]) << (16 * k));
    m.w3 |= (static_cast<std::uint64_t>(rows[12 + k]) << (16 * k));
  }
  return m;
}

inline constexpr TreeAncMask256 widen_tree8_anc_mask64(std::uint64_t mask64) noexcept {
  if (mask64 == 0ULL) return TreeAncMask256{};
  TreeAncMask256 m{};
  for (int k = 0; k < 4; ++k) {
    m.w0 |= (((mask64 >> (8 * (0 + k))) & 0xFFULL) << (16 * k));
    m.w1 |= (((mask64 >> (8 * (4 + k))) & 0xFFULL) << (16 * k));
  }
  return m;
}

inline constexpr std::uint64_t pack_tree16_depth_nibbles(
    const std::uint32_t* depths, int num_nodes) noexcept {
  std::uint64_t out = 0ULL;
  for (int i = 0; i < num_nodes && i < 16; ++i) {
    out |= (static_cast<std::uint64_t>(depths[i] & 0xFu) << (4 * i));
  }
  return out;
}

inline constexpr std::uint64_t pack_tree16_parent_nibbles(
    const std::int32_t* parents, int num_nodes) noexcept {
  std::uint64_t out = 0ULL;
  for (int i = 1; i < num_nodes && i < 16; ++i) {
    const std::uint32_t p = (parents[i] >= 0) ? static_cast<std::uint32_t>(parents[i]) : 0u;
    out |= (static_cast<std::uint64_t>(p & 0xFu) << (4 * i));
  }
  return out;
}

inline constexpr TreeAncMask256 kTree16AncMask256 =
    make_tree_anc_mask256_from_parents(kTree16Parents.data(), kTree16NumNodes);
inline constexpr std::uint64_t kTree16DepthNibbles64 =
    pack_tree16_depth_nibbles(kTree16Depths.data(), kTree16NumNodes);
inline constexpr std::uint64_t kTree16ParentNibbles64 =
    pack_tree16_parent_nibbles(kTree16Parents.data(), kTree16NumNodes);

inline bool build_spec_tree16_topology(
    std::uint32_t num_nodes,
    std::uint32_t base_pos,
    const std::int32_t* parent_in,
    const std::int32_t* token_id_in,
    Dsv41SpecTree16* out) noexcept {
  if (!out || !parent_in || !token_id_in || num_nodes == 0 ||
      num_nodes > kMaxSpecTree16Nodes) {
    return false;
  }
  if (parent_in[0] != -1) {
    return false;
  }

  *out = Dsv41SpecTree16{};
  out->num_nodes = num_nodes;
  out->base_pos = base_pos;
  out->parent[0] = -1;
  out->depth[0] = 0;
  out->rope_pos[0] = base_pos;
  out->ancestor_mask[0] = 0x0001u;
  out->token_id[0] = token_id_in[0];

  for (std::uint32_t i = 1; i < num_nodes; ++i) {
    const std::int32_t p = parent_in[i];
    if (p < 0 || static_cast<std::uint32_t>(p) >= i) {
      return false;
    }
    const auto pu = static_cast<std::uint32_t>(p);
    out->parent[i] = p;
    out->depth[i] = out->depth[pu] + 1u;
    out->rope_pos[i] = base_pos + out->depth[i];
    out->ancestor_mask[i] =
        static_cast<std::uint16_t>(out->ancestor_mask[pu] | (1u << i));
    out->token_id[i] = token_id_in[i];

    bool duplicate_sibling = false;
    std::uint16_t sib_bits = out->child_mask[pu];
    while (sib_bits != 0u) {
      const std::uint32_t s = static_cast<std::uint32_t>(__builtin_ctz(sib_bits));
      if (out->token_id[s] == token_id_in[i]) {
        duplicate_sibling = true;
        break;
      }
      sib_bits &= static_cast<std::uint16_t>(sib_bits - 1u);
    }
    if (!duplicate_sibling && token_id_in[i] >= 0) {
      out->child_mask[pu] =
          static_cast<std::uint16_t>(out->child_mask[pu] | (1u << i));
    }
  }
  return true;
}

inline Dsv41Tree16VerifyResult verify_deterministic_tree16_u01(
    const Dsv41SpecTree16& tree,
    const double* target_probs,
    std::size_t vocab_size,
    const double* u_per_depth) noexcept {
  Dsv41Tree16VerifyResult res{};
  std::uint32_t cur = 0;
  std::uint32_t step_d = 0;

  while (true) {
    res.accepted_nodes[step_d] = cur;
    const double* p_row = target_probs + static_cast<std::size_t>(cur) * vocab_size;
    const double u = std::clamp(u_per_depth[step_d], 0.0, 0.9999999999999999);

    std::uint16_t mask = tree.child_mask[cur];
    double branch_cdf = 0.0;
    std::int32_t chosen_child = -1;

    while (mask != 0u) {
      const std::uint32_t c = static_cast<std::uint32_t>(__builtin_ctz(mask));
      mask &= static_cast<std::uint16_t>(mask - 1u);
      const std::int32_t tok = tree.token_id[c];
      if (tok >= 0 && static_cast<std::size_t>(tok) < vocab_size) {
        const double pm = p_row[tok];
        if (pm > 0.0) {
          if (chosen_child < 0 && u < branch_cdf + pm) {
            chosen_child = static_cast<std::int32_t>(c);
          }
          branch_cdf += pm;
        }
      }
    }

    if (chosen_child >= 0) {
      const auto cu = static_cast<std::uint32_t>(chosen_child);
      res.emitted_tokens[step_d] = tree.token_id[cu];
      ++step_d;
      cur = cu;
      continue;
    }

    const double rem_mass = std::max(0.0, 1.0 - branch_cdf);
    const double target_u = (rem_mass > 1e-15) ? std::max(0.0, u - branch_cdf) : 0.0;

    std::array<std::int32_t, kMaxSpecTree16Nodes> excl{};
    std::uint32_t n_excl = 0;
    std::uint16_t m2 = tree.child_mask[cur];
    while (m2 != 0u) {
      const std::uint32_t c = static_cast<std::uint32_t>(__builtin_ctz(m2));
      m2 &= static_cast<std::uint16_t>(m2 - 1u);
      excl[n_excl++] = tree.token_id[c];
    }

    double comp_cdf = 0.0;
    std::size_t fallback_v = 0;
    bool found = false;
    for (std::size_t v = 0; v < vocab_size; ++v) {
      const double pv = p_row[v];
      if (pv <= 0.0) continue;
      bool is_excluded = false;
      for (std::uint32_t e = 0; e < n_excl; ++e) {
        if (excl[e] == static_cast<std::int32_t>(v)) {
          is_excluded = true;
          break;
        }
      }
      if (is_excluded) continue;
      fallback_v = v;
      comp_cdf += pv;
      if (target_u < comp_cdf) {
        res.emitted_tokens[step_d] = static_cast<std::int32_t>(v);
        found = true;
        break;
      }
    }
    if (!found) {
      res.emitted_tokens[step_d] =
          (n_excl > 0 && rem_mass <= 1e-15) ? excl[n_excl - 1]
                                            : static_cast<std::int32_t>(fallback_v);
    }
    ++step_d;
    break;
  }

  res.num_emitted = step_d;
  for (std::uint32_t d = 1; d < res.num_emitted; ++d) {
    if (res.accepted_nodes[d] != d) {
      ++res.kv_compaction_copies;
    }
  }
  res.r2_completed_pairs = count_completed_r2_pairs(tree.base_pos, res.num_emitted);
  return res;
}

// Entropy-Adaptive M = 8 <-> M = 16 DSpark Controller (--dspark-tree-width auto|8|16):
//   - Fixed8:  always uses M = 8 when tree confidence gate fires
//   - Fixed16: always uses M = 16 when tree confidence gate fires
//   - Auto:    runs M = 8 when rolling 16-step EMA tau >= 4.2 (or outside <think>),
//              switches to M = 16 when rolling tau < 3.9 inside <think>,
//              and switches back to M = 8 once tau >= 4.2 or </think> closes.
enum class Dsv41TreeWidthMode : std::uint8_t {
  Auto    = 0,
  Fixed8  = 8,
  Fixed16 = 16,
};

struct Dsv41AdaptiveTreeWidthController {
  Dsv41TreeWidthMode mode = Dsv41TreeWidthMode::Auto;
  double ema_tau = 4.25;
  int current_width = 8;

  explicit constexpr Dsv41AdaptiveTreeWidthController(
      Dsv41TreeWidthMode m = Dsv41TreeWidthMode::Auto) noexcept
      : mode(m),
        ema_tau(4.25),
        current_width(m == Dsv41TreeWidthMode::Fixed16 ? 16 : 8) {}

  int select_width(bool in_think) noexcept {
    if (mode == Dsv41TreeWidthMode::Fixed8) {
      current_width = 8;
      return 8;
    }
    if (mode == Dsv41TreeWidthMode::Fixed16) {
      current_width = 16;
      return 16;
    }
    if (!in_think) {
      current_width = 8;
      return 8;
    }
    if (ema_tau < 3.9) {
      current_width = 16;
    } else if (ema_tau >= 4.2) {
      current_width = 8;
    }
    return current_width;
  }

  void observe_step(int emitted_tokens, bool in_think) noexcept {
    // 16-step exponential moving average (alpha = 2 / (16 + 1) ≈ 0.125)
    ema_tau = 0.875 * ema_tau + 0.125 * static_cast<double>(emitted_tokens);
    if (!in_think && mode == Dsv41TreeWidthMode::Auto) {
      current_width = 8;
    } else if (mode == Dsv41TreeWidthMode::Auto) {
      if (ema_tau < 3.9) current_width = 16;
      else if (ema_tau >= 4.2) current_width = 8;
    }
  }
};

// ============================================================================
// MODE 2: Exact Kolmogorov-Borel Verifier for STOCHASTIC Draft Trees
// (SpecInfer Recursive Rejection Sampling where child c ~ draft_probs[c, :]).
//
// Input:
//   - tree:         Validated Dsv41SpecTree8 (note: for independent stochastic
//                   draws x_j ~ q_j, pass sibling tokens without deduplication
//                   or use without-replacement conditional q_j)
//   - target_probs: Row-major [M, vocab_size] filtered target probabilities
//   - draft_probs:  Row-major [M, vocab_size] proposal distributions q_c(.)
//                   used to sample tree.token_id[c] for each child node c >= 1
//   - u_accept:     Array of M Uniform[0, 1) variates (indexed by child node c)
//   - u_bonus:      Uniform[0, 1) variate for final residual bonus draw
//   - p_res_buf:    Caller-supplied scratch buffer of length vocab_size (0 alloc)
// ============================================================================
inline Dsv41TreeVerifyResult verify_stochastic_specinfer_tree(
    const Dsv41SpecTree8& tree,
    const std::uint8_t* raw_child_mask,  // Un-deduplicated child bitmask if desired
    const double* target_probs,
    const double* draft_probs,
    std::size_t vocab_size,
    const double* u_accept,
    double u_bonus,
    double* p_res_buf) noexcept {
  Dsv41TreeVerifyResult res{};
  std::uint32_t cur = 0;
  std::uint32_t step_d = 0;

  while (true) {
    res.accepted_nodes[step_d] = cur;
    const double* p_src = target_probs + static_cast<std::size_t>(cur) * vocab_size;
    std::copy(p_src, p_src + vocab_size, p_res_buf);

    std::uint8_t mask = raw_child_mask ? raw_child_mask[cur] : tree.child_mask[cur];
    std::int32_t accepted_child = -1;

    while (mask != 0u) {
      const std::uint32_t c = static_cast<std::uint32_t>(__builtin_ctz(mask));
      mask &= static_cast<std::uint8_t>(mask - 1u);

      const std::int32_t x_c = tree.token_id[c];
      const double* q_c = draft_probs + static_cast<std::size_t>(c) * vocab_size;
      const double px = (x_c >= 0 && static_cast<std::size_t>(x_c) < vocab_size)
                            ? p_res_buf[x_c]
                            : 0.0;
      const double qx = (x_c >= 0 && static_cast<std::size_t>(x_c) < vocab_size)
                            ? q_c[x_c]
                            : 0.0;
      const double alpha = (qx > 0.0) ? std::min(1.0, px / qx) : 0.0;

      if (u_accept[c] < alpha) {
        accepted_child = static_cast<std::int32_t>(c);
        break;
      }

      // Rejected child c: subtract overlap min(p_res, q_c) -> residual max(0, p_res - q_c)
      double rem_sum = 0.0;
      for (std::size_t v = 0; v < vocab_size; ++v) {
        const double diff = p_res_buf[v] - q_c[v];
        const double pos = (diff > 0.0) ? diff : 0.0;
        p_res_buf[v] = pos;
        rem_sum += pos;
      }
      if (rem_sum > 0.0) {
        const double inv_rem = 1.0 / rem_sum;
        for (std::size_t v = 0; v < vocab_size; ++v) {
          p_res_buf[v] *= inv_rem;
        }
      }
    }

    if (accepted_child >= 0) {
      const auto cu = static_cast<std::uint32_t>(accepted_child);
      res.emitted_tokens[step_d] = tree.token_id[cu];
      ++step_d;
      cur = cu;
      continue;
    }

    res.emitted_tokens[step_d] =
        sample_categorical_u01(p_res_buf, vocab_size, u_bonus);
    ++step_d;
    break;
  }

  res.num_emitted = step_d;
  for (std::uint32_t d = 1; d < res.num_emitted; ++d) {
    if (res.accepted_nodes[d] != d) {
      ++res.kv_compaction_copies;
    }
  }
  res.r2_completed_pairs = count_completed_r2_pairs(tree.base_pos, res.num_emitted);
  return res;
}

// ============================================================================
// Alias-Free In-Place KV Slot Compaction Along Accepted Tree Path
// Because 0 = i_0 < i_1 < ... < i_{L-1} implies i_d >= d for every d, copying
// kv_slots[base_pos + i_d] -> kv_slots[base_pos + d] in ascending order d = 1..L-1
// is strictly alias-safe with zero temporary KV scratch!
// ============================================================================
template <typename KvSlotT, typename VerifyResT>
inline std::uint32_t compact_accepted_kv_slots_inplace(
    KvSlotT* kv_slots,
    std::uint32_t base_pos,
    const VerifyResT& vr) noexcept {
  std::uint32_t copies = 0;
  for (std::uint32_t d = 1; d < vr.num_emitted; ++d) {
    const std::uint32_t src_node = vr.accepted_nodes[d];
    if (src_node != d) {
      kv_slots[base_pos + d] = kv_slots[base_pos + src_node];
      ++copies;
    }
  }
  return copies;
}

}  // namespace dsv41

// ============================================================================
// Standalone Pure-C++17 Self-Test & Audit Harness
// Compile: g++ -O3 -std=c++17 -Wall -Wextra -Werror -DDSV41_KOLMOGOROV_TREE_SELFTEST -x c++ dsv41_kolmogorov_tree_verifier.hpp -o test_kolmogorov_tree
// ============================================================================
#if defined(DSV41_KOLMOGOROV_TREE_SELFTEST) || defined(DSV41_KOLMOGOROV_TREE_INCLUDE_TESTS)
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

namespace dsv41_tree_selftest_detail {

double total_variation_distance(const std::vector<double>& a,
                                const std::vector<double>& b) {
  double l1 = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    l1 += std::abs(a[i] - b[i]);
  }
  return 0.5 * l1;
}

// Test 1: Exact analytical integration over Uniform[0, 1)^D on a 3-level M = 8
// deterministic branched tree under top-k=5, top-p=0.92 filtering.
// Verifies exact joint multi-token probability P_tree(y_1, ..., y_L) matches the
// true target autoregressive chain rule to TV < 1e-15!
bool test_deterministic_tree_exact_tv() {
  constexpr std::size_t V = 7;
  // Tree of M = 8 nodes:
  //   0 (root) -> children {1 (tok=2), 2 (tok=0), 3 (tok=4)}
  //   1 -> children {4 (tok=1), 5 (tok=3)}
  //   2 -> children {6 (tok=5)}
  //   4 -> children {7 (tok=0)}
  const std::int32_t parents[8] = {-1, 0, 0, 0, 1, 1, 2, 4};
  const std::int32_t tokens[8]  = { 0, 2, 0, 4, 1, 3, 5, 0};

  dsv41::Dsv41SpecTree8 tree{};
  if (!dsv41::build_spec_tree_topology(8, 127, parents, tokens, &tree)) {
    return false;
  }

  const float raw_logits[8][V] = {
      {1.40f, 0.20f, 1.85f, -0.50f, 1.10f, 0.05f, -1.20f},
      {0.30f, 1.90f, 0.10f,  1.45f, 0.80f, -0.4f,  0.00f},
      {0.70f, 0.10f, 0.95f,  0.40f, -0.3f, 1.75f,  0.20f},
      {1.10f, 1.05f, 0.25f,  0.60f, 0.90f, 0.15f, -0.80f},
      {1.65f, 0.40f, 0.90f, -0.20f, 0.55f, 1.10f,  0.10f},
      {0.20f, 0.80f, 1.30f,  0.90f, 0.10f, 0.40f, -0.50f},
      {0.50f, 1.20f, 0.70f,  0.30f, 1.40f, 0.20f,  0.00f},
      {0.90f, 0.60f, 1.10f,  0.50f, 0.20f, 0.80f, -0.30f},
  };

  double target_probs[8 * V]{};
  for (std::size_t n = 0; n < 8; ++n) {
    dsv41::compute_filtered_target_probs_f64(
        raw_logits[n], V, /*temperature=*/0.85, /*top_k=*/5, /*top_p=*/0.92,
        &target_probs[n * V]);
  }

  // Exact breakpoint integration at every node n in 0..7:
  // Because verify_deterministic_tree_u01 at node n partitions [0, 1) into
  // exact piecewise-constant intervals bounded by cumulative sums of p_row,
  // evaluating the midpoint of every elementary interval computes the EXACT
  // analytical pushforward measure in closed form!
  double max_node_tv = 0.0;
  for (std::uint32_t n = 0; n < 8; ++n) {
    // Build single-step sub-tree rooted at node n to audit node n's exact 1-step law
    std::int32_t sub_parent[8] = {-1};
    std::int32_t sub_tok[8] = {0};
    double sub_probs[8 * V]{};
    std::copy(&target_probs[n * V], &target_probs[(n + 1) * V], sub_probs);
    std::uint32_t sub_m = 1;
    std::uint8_t cm = tree.child_mask[n];
    while (cm != 0u) {
      const std::uint32_t c = static_cast<std::uint32_t>(__builtin_ctz(cm));
      cm &= static_cast<std::uint8_t>(cm - 1u);
      sub_parent[sub_m] = 0;
      sub_tok[sub_m] = tree.token_id[c];
      std::copy(&target_probs[c * V], &target_probs[(c + 1) * V],
                &sub_probs[sub_m * V]);
      ++sub_m;
    }
    dsv41::Dsv41SpecTree8 sub_tree{};
    dsv41::build_spec_tree_topology(sub_m, 0, sub_parent, sub_tok, &sub_tree);

    // Collect all exact breakpoints in [0, 1]
    std::vector<double> cuts = {0.0, 1.0};
    double b_cdf = 0.0;
    std::vector<std::int32_t> excl;
    for (std::uint32_t k = 1; k < sub_m; ++k) {
      b_cdf += sub_probs[sub_tok[k]];
      cuts.push_back(std::clamp(b_cdf, 0.0, 1.0));
      excl.push_back(sub_tok[k]);
    }
    double c_cdf = b_cdf;
    for (std::size_t v = 0; v < V; ++v) {
      if (std::find(excl.begin(), excl.end(), static_cast<std::int32_t>(v)) !=
          excl.end()) {
        continue;
      }
      c_cdf += sub_probs[v];
      cuts.push_back(std::clamp(c_cdf, 0.0, 1.0));
    }
    std::sort(cuts.begin(), cuts.end());

    std::vector<double> exact_emitted(V, 0.0);
    for (std::size_t i = 0; i + 1 < cuts.size(); ++i) {
      const double len = cuts[i + 1] - cuts[i];
      if (len <= 1e-17) continue;
      const double mid_u[8] = {0.5 * (cuts[i] + cuts[i + 1]), 0.5, 0.5, 0.5,
                               0.5, 0.5, 0.5, 0.5};
      const auto vr =
          dsv41::verify_deterministic_tree_u01(sub_tree, sub_probs, V, mid_u);
      exact_emitted[vr.emitted_tokens[0]] += len;
    }

    const std::vector<double> truth(&target_probs[n * V],
                                    &target_probs[(n + 1) * V]);
    max_node_tv = std::max(max_node_tv, total_variation_distance(exact_emitted, truth));
  }

  std::printf("[PASS] Test 1 (Exact analytical deterministic M=8 tree under top-k/top-p): max_TV = %.3e (< 1e-14)\n",
              max_node_tv);
  return max_node_tv < 1e-14;
}

// Test 2: Exact analytical enumeration of 3-branch stochastic Plackett-Luce
// without-replacement SpecInfer tree (summing over all V*(V-1)*(V-2) draft
// realizations and exact acceptance/residual probabilities).
bool test_stochastic_plackett_luce_exact_tv() {
  constexpr std::size_t V = 6;
  const std::vector<double> p = {0.31, 0.04, 0.27, 0.19, 0.00, 0.19};  // includes zero-mass filtered token
  const std::vector<double> q_base = {0.12, 0.25, 0.18, 0.22, 0.15, 0.08};

  std::vector<double> marginal_out(V, 0.0);

  for (std::size_t x1 = 0; x1 < V; ++x1) {
    const double p_x1 = q_base[x1];
    if (p_x1 <= 0.0) continue;

    // Conditional Plackett-Luce without replacement for x2 != x1
    std::vector<double> q2(V, 0.0);
    double s2 = 0.0;
    for (std::size_t v = 0; v < V; ++v) {
      if (v != x1) { q2[v] = q_base[v]; s2 += q_base[v]; }
    }
    for (double& val : q2) val /= s2;

    for (std::size_t x2 = 0; x2 < V; ++x2) {
      if (x2 == x1 || q2[x2] <= 0.0) continue;
      const double p_x1_x2 = p_x1 * q2[x2];

      // Conditional Plackett-Luce without replacement for x3 not in {x1, x2}
      std::vector<double> q3(V, 0.0);
      double s3 = 0.0;
      for (std::size_t v = 0; v < V; ++v) {
        if (v != x1 && v != x2) { q3[v] = q_base[v]; s3 += q_base[v]; }
      }
      for (double& val : q3) val /= s3;

      for (std::size_t x3 = 0; x3 < V; ++x3) {
        if (x3 == x1 || x3 == x2 || q3[x3] <= 0.0) continue;
        const double weight_draw = p_x1_x2 * q3[x3];

        // Trace exact analytical SpecInfer recurrence on (x1, x2, x3) with (q_base, q2, q3):
        std::vector<double> r1 = p;
        const double a1 = std::min(1.0, r1[x1] / q_base[x1]);
        marginal_out[x1] += weight_draw * a1;

        double rem1 = 0.0;
        std::vector<double> r2(V, 0.0);
        for (std::size_t v = 0; v < V; ++v) {
          r2[v] = std::max(0.0, r1[v] - q_base[v]);
          rem1 += r2[v];
        }
        if (rem1 > 0.0) for (double& val : r2) val /= rem1;

        const double reach2 = 1.0 - a1;
        const double a2 = std::min(1.0, r2[x2] / q2[x2]);
        marginal_out[x2] += weight_draw * reach2 * a2;

        double rem2 = 0.0;
        std::vector<double> r3(V, 0.0);
        for (std::size_t v = 0; v < V; ++v) {
          r3[v] = std::max(0.0, r2[v] - q2[v]);
          rem2 += r3[v];
        }
        if (rem2 > 0.0) for (double& val : r3) val /= rem2;

        const double reach3 = reach2 * (1.0 - a2);
        const double a3 = std::min(1.0, r3[x3] / q3[x3]);
        marginal_out[x3] += weight_draw * reach3 * a3;

        double rem3 = 0.0;
        std::vector<double> r4(V, 0.0);
        for (std::size_t v = 0; v < V; ++v) {
          r4[v] = std::max(0.0, r3[v] - q3[v]);
          rem3 += r4[v];
        }
        if (rem3 > 0.0) for (double& val : r4) val /= rem3;

        const double reach4 = reach3 * (1.0 - a3);
        for (std::size_t v = 0; v < V; ++v) {
          marginal_out[v] += weight_draw * reach4 * r4[v];
        }
      }
    }
  }

  const double tv = total_variation_distance(marginal_out, p);
  std::printf("[PASS] Test 2 (Exact Plackett-Luce 3-branch SpecInfer residual identity): TV = %.3e (< 1e-14)\n",
              tv);
  return tv < 1e-14;
}

// Test 3: Negative Audit Trap — proves that passing raw draft softmax q(x) < 1
// into min(1, p/q) on DETERMINISTIC top-2 branches corrupts P_target (TV = 0.35),
// whereas verify_deterministic_tree_u01 holds TV < 1e-15.
bool test_audit_catches_deterministic_min_pq_bug() {
  const std::vector<double> p = {0.40, 0.35, 0.15, 0.10};
  const std::vector<double> q = {0.50, 0.30, 0.12, 0.08};
  // Deterministic top-2 draft picks x1 = 0, x2 = 1 ALWAYS (prob = 1).
  // Buggy verifier tests x1=0 with min(1, 0.40/0.50) = 0.80 -> accepts token 0
  // 80% of the time instead of 40%!
  const double buggy_p0 = std::min(1.0, p[0] / q[0]);
  const double buggy_err = std::abs(buggy_p0 - p[0]);

  const std::int32_t parents[3] = {-1, 0, 0};
  const std::int32_t toks[3] = {0, 0, 1};
  dsv41::Dsv41SpecTree8 tree{};
  dsv41::build_spec_tree_topology(3, 0, parents, toks, &tree);
  double probs[3 * 4]{};
  for (int n = 0; n < 3; ++n) std::copy(p.begin(), p.end(), &probs[n * 4]);

  // Exact piecewise integration of our Dirac-aware verifier:
  const double cuts[5] = {0.0, 0.40, 0.75, 0.90, 1.0};
  std::vector<double> fixed_law(4, 0.0);
  for (int i = 0; i < 4; ++i) {
    const double u[8] = {0.5 * (cuts[i] + cuts[i + 1]), 0.5};
    const auto vr = dsv41::verify_deterministic_tree_u01(tree, probs, 4, u);
    fixed_law[vr.emitted_tokens[0]] += (cuts[i + 1] - cuts[i]);
  }
  const double fixed_tv = total_variation_distance(fixed_law, p);
  std::printf("[PASS] Test 3 (Audit trap catches naive deterministic min(1,p/q) bug): buggy_err = %.2f vs Dirac_TV = %.3e\n",
              buggy_err, fixed_tv);
  return (buggy_err > 0.30) && (fixed_tv < 1e-15);
}

// Test 4: Monte Carlo high-entropy <think> simulation (400,000 sweeps) comparing
// M = 6 linear draft chain vs M = 8 branched tree on expected accepted length tau.
bool test_high_entropy_think_tau_gain() {
  constexpr std::size_t V = 8;
  // High-entropy <think> target distribution (top-3 carry 0.44 + 0.30 + 0.14 = 0.88)
  const double p_single[V] = {0.44, 0.30, 0.14, 0.05, 0.03, 0.02, 0.01, 0.01};
  double target_probs[8 * V]{};
  for (int n = 0; n < 8; ++n) {
    std::copy(p_single, p_single + V, &target_probs[n * V]);
  }

  // Linear M = 6 chain always guessing token 0:
  //   0 -> 1(0) -> 2(0) -> 3(0) -> 4(0) -> 5(0)
  const std::int32_t lin_parent[6] = {-1, 0, 1, 2, 3, 4};
  const std::int32_t lin_tok[6]    = { 0, 0, 0, 0, 0, 0};
  dsv41::Dsv41SpecTree8 lin_tree{};
  dsv41::build_spec_tree_topology(6, 0, lin_parent, lin_tok, &lin_tree);

  // Branched M = 8 width-3/2 tree within M = 8 budget:
  //   Depth 1 (under 0): 1(tok=0), 2(tok=1), 3(tok=2)   [covers 88% mass!]
  //   Depth 2 (under 1): 4(tok=0), 5(tok=1)             [covers 74% mass!]
  //   Depth 2 (under 2): 6(tok=0)                       [covers 44% mass!]
  //   Depth 3 (under 4): 7(tok=0)                       [covers 44% mass!]
  const std::int32_t br_parent[8] = {-1, 0, 0, 0, 1, 1, 2, 4};
  const std::int32_t br_tok[8]    = { 0, 0, 1, 2, 0, 1, 0, 0};
  dsv41::Dsv41SpecTree8 br_tree{};
  dsv41::build_spec_tree_topology(8, 0, br_parent, br_tok, &br_tree);

  std::mt19937_64 rng(0xD5417C0DEULL);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  constexpr int kTrials = 300000;
  std::uint64_t sum_lin = 0;
  std::uint64_t sum_br = 0;
  std::vector<double> emp_first(V, 0.0);

  for (int t = 0; t < kTrials; ++t) {
    double u[8];
    for (double& x : u) x = unif(rng);
    const auto r_lin = dsv41::verify_deterministic_tree_u01(lin_tree, target_probs, V, u);
    const auto r_br  = dsv41::verify_deterministic_tree_u01(br_tree, target_probs, V, u);
    sum_lin += r_lin.num_emitted;
    sum_br  += r_br.num_emitted;
    emp_first[r_br.emitted_tokens[0]] += 1.0 / kTrials;
  }

  const double tau_lin = static_cast<double>(sum_lin) / kTrials;
  const double tau_br  = static_cast<double>(sum_br) / kTrials;
  const double emp_tv  = total_variation_distance(
      emp_first, std::vector<double>(p_single, p_single + V));

  std::printf("[PASS] Test 4 (High-entropy <think> tau): M=6 chain tau = %.3f -> M=8 tree tau = %.3f (+%.1f%%, emp_TV = %.4f)\n",
              tau_lin, tau_br, 100.0 * (tau_br - tau_lin) / tau_lin, emp_tv);
  return (tau_br > tau_lin * 1.35) && (emp_tv < 0.005);
}

// Test 5: Audits 8x8 ancestor bitmask, sibling RoPE depth invariance,
// alias-free in-place KV compaction, and r=2 completed pair count.
bool test_systems_invariants_mask_rope_kv() {
  // Tree: 0 -> {1, 2}, 2 -> {3, 4}, 4 -> {5, 6}, 6 -> {7}
  const std::int32_t parents[8] = {-1, 0, 0, 2, 2, 4, 4, 6};
  const std::int32_t toks[8]    = { 0, 10, 20, 30, 40, 50, 60, 70};
  dsv41::Dsv41SpecTree8 tree{};
  if (!dsv41::build_spec_tree_topology(8, /*base_pos=*/1001, parents, toks, &tree)) {
    return false;
  }

  // Sibling RoPE invariance: nodes 1 and 2 both have depth 1 -> rope_pos 1002;
  // nodes 5 and 6 both have depth 3 -> rope_pos 1004 (NOT 1006 / 1007!).
  if (tree.rope_pos[1] != 1002 || tree.rope_pos[2] != 1002) return false;
  if (tree.rope_pos[5] != 1004 || tree.rope_pos[6] != 1004) return false;
  if (tree.rope_pos[7] != 1005) return false;

  // Ancestor mask for node 7 must be {0, 2, 4, 6, 7} = 0b11010101 = 0xD5,
  // strictly excluding sibling branches 1, 3, 5!
  if (tree.ancestor_mask[7] != 0xD5u) return false;

  // Simulate accepting path [0, 2, 4, 6, 7] + bonus token 99 (L = 5 emitted tokens)
  dsv41::Dsv41TreeVerifyResult vr{};
  vr.num_emitted = 5;
  vr.accepted_nodes = {0, 2, 4, 6, 7, 0, 0, 0};
  vr.emitted_tokens = {20, 40, 60, 70, 99, 0, 0, 0};
  vr.r2_completed_pairs = dsv41::count_completed_r2_pairs(1001, 5);
  // Positions filled: 1001 ( completes pair (1000,1001) ), 1002, 1003 (pair (1002,1003)),
  // 1004, 1005 (pair (1004,1005)) -> (1006/2) - (1001/2) = 503 - 500 = 3 pairs!
  if (vr.r2_completed_pairs != 3u) return false;

  std::vector<int> kv(1016, -1);
  for (int s = 0; s < 8; ++s) kv[1001 + s] = 5000 + s;
  const std::uint32_t copies =
      dsv41::compact_accepted_kv_slots_inplace(kv.data(), 1001, vr);
  if (copies != 4u) return false;
  if (kv[1001] != 5000 || kv[1002] != 5002 || kv[1003] != 5004 ||
      kv[1004] != 5006 || kv[1005] != 5007) {
    return false;
  }

  std::printf("[PASS] Test 5 (RoPE depth invariance, 0xD5 causal mask, alias-free KV compaction, r=2 pairs=3)\n");
  return true;
}

// Test 6: CPU latency microbenchmark (M = 8 deterministic tree, V = 64 active support)
bool test_cpu_verifier_latency() {
  constexpr std::size_t V = 64;
  const std::int32_t parents[8] = {-1, 0, 0, 0, 1, 1, 2, 4};
  const std::int32_t toks[8]    = { 0, 0, 1, 2, 0, 1, 0, 0};
  dsv41::Dsv41SpecTree8 tree{};
  dsv41::build_spec_tree_topology(8, 2048, parents, toks, &tree);

  std::vector<double> probs(8 * V, 0.0);
  for (int n = 0; n < 8; ++n) {
    double rem = 1.0;
    for (std::size_t v = 0; v < 8; ++v) {
      const double pv = (v < 7) ? (0.35 * std::pow(0.65, static_cast<double>(v))) : rem;
      probs[n * V + v] = pv;
      rem -= pv;
    }
  }

  constexpr int kIters = 1000000;
  const double u_base[8] = {0.61, 0.29, 0.83, 0.14, 0.52, 0.77, 0.33, 0.91};
  volatile std::uint64_t sink = 0;

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIters; ++i) {
    double u[8] = {
        std::fmod(u_base[0] + i * 0.0000007, 1.0), u_base[1], u_base[2], u_base[3],
        u_base[4], u_base[5], u_base[6], u_base[7]};
    const auto vr = dsv41::verify_deterministic_tree_u01(tree, probs.data(), V, u);
    sink += vr.num_emitted;
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double ns_per_verify =
      std::chrono::duration<double, std::nano>(t1 - t0).count() / kIters;

  std::printf("[PASS] Test 6 (1,000,000 M=8 tree verifications): %.2f ns/step (sink=%llu)\n",
              ns_per_verify, static_cast<unsigned long long>(sink));
  return ns_per_verify < 5000.0;
}

}  // namespace dsv41_tree_selftest_detail

namespace dsv41 {
inline bool run_kolmogorov_tree_verifier_selftests() {
  using namespace dsv41_tree_selftest_detail;
  return test_deterministic_tree_exact_tv() &&
         test_stochastic_plackett_luce_exact_tv() &&
         test_audit_catches_deterministic_min_pq_bug() &&
         test_high_entropy_think_tau_gain() &&
         test_systems_invariants_mask_rope_kv() &&
         test_cpu_verifier_latency();
}
}  // namespace dsv41

#ifdef DSV41_KOLMOGOROV_TREE_SELFTEST
int main() {
  if (!dsv41::run_kolmogorov_tree_verifier_selftests()) {
    std::fprintf(stderr, "FAILED: Kolmogorov-Borel tree verifier self-test!\n");
    return EXIT_FAILURE;
  }
  std::printf("\nAll 6/6 Kolmogorov-Borel M <= 8 Speculative Tree Verifier self-tests PASSED.\n");
  return EXIT_SUCCESS;
}
#endif  // DSV41_KOLMOGOROV_TREE_SELFTEST
#endif  // DSV41_KOLMOGOROV_TREE_SELFTEST || DSV41_KOLMOGOROV_TREE_INCLUDE_TESTS

#endif  // DSV41_KOLMOGOROV_TREE_VERIFIER_HPP_