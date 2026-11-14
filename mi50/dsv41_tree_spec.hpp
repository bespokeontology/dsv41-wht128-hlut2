#pragma once
// Pure C++20 Kolmogorov-Borel speculative tree verifier for DeepSeek-V4.1-Flash-1017 DSpark.
//
// (1) Exact causal tree mask for M <= 8 branched `mtp.0..2` draft nodes: each node
//     attends ONLY its root + parent chain + itself, with per-node depth position ids.
// (2) Borel residual rejection sampler: accept draft token x with min(1, p(x)/q(x));
//     on rejection sample normalized residual max(0, p(y)-q(y)) over the same target
//     top-p/top-k support.  Under a shared draft q at a node this is exactly
//     Kolmogorov-Borel P_tree_spec == P_target, no truncation shift.
//
// The target p and draft q are FP64 probability vectors over a supplied support n.
// No Python anywhere. No device/scratch work in this header.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace dsv41 {

inline constexpr uint32_t kTreeSpecMaxNodes = 16u;     // M <= 16 branched DSpark verify (M=8 & M=16)
inline constexpr uint32_t kTreeSpecMaxDepth = 5u;      // mtp + Markov chain depth <= 5 (up to 6 emitted tokens)

struct TreeSpecNode {
    int32_t parent = -1;    // node index parent; -1 only root node 0
    int32_t token  = -1;    // token id at this node
    uint32_t depth = 0;     // 0 root anchor, 1..5 draft tree levels
};

struct TreeSpecPlan {
    uint32_t node_count = 0;
    std::array<uint16_t, kTreeSpecMaxNodes> allow_mask{}; // bit j allowed if node j is root/parent-chain/self
    std::array<int32_t, kTreeSpecMaxNodes> pos_id{};      // root_pos + depth
    std::array<uint32_t, kTreeSpecMaxNodes> sibling{};    // sibling ordinal at parent
    bool valid = false;
};

inline bool tree_spec_plan_valid(const TreeSpecNode* nodes, uint32_t count) noexcept {
    if (nodes == nullptr || count == 0u || count > kTreeSpecMaxNodes) return false;
    if (nodes[0].parent != -1 || nodes[0].depth != 0u) return false;
    for (uint32_t i = 0u; i < count; ++i) {
        if (nodes[i].depth > kTreeSpecMaxDepth) return false;
        if (nodes[i].token < 0) return false;
        if (i != 0u) {
            if (nodes[i].parent < 0 || uint32_t(nodes[i].parent) >= i) return false;
            if (nodes[i].depth != nodes[uint32_t(nodes[i].parent)].depth + 1u) return false;
        }
    }
    return true;
}

inline TreeSpecPlan build_tree_spec_plan(
    const TreeSpecNode* nodes,
    uint32_t count,
    int32_t root_pos) noexcept
{
    TreeSpecPlan plan{};
    plan.node_count = count;
    if (!tree_spec_plan_valid(nodes, count) || root_pos < 0) return plan;

    for (uint32_t i = 0u; i < count; ++i) {
        uint16_t mask = uint16_t(1u << i);      // itself
        uint32_t ancestor = i;
        for (uint32_t d = 0u; d < nodes[i].depth; ++d) {
            ancestor = uint32_t(nodes[ancestor].parent);
            mask |= uint16_t(1u << ancestor);
        }
        plan.allow_mask[i] = mask;
        plan.pos_id[i] = root_pos + int32_t(nodes[i].depth);
    }

    std::array<uint32_t, kTreeSpecMaxNodes> seen_at_parent{};
    for (uint32_t i = 1u; i < count; ++i) {
        const uint32_t p = uint32_t(nodes[i].parent);
        plan.sibling[i] = seen_at_parent[p]++;
    }
    plan.valid = true;
    return plan;
}

inline uint32_t tree_spec_sibling_count(const TreeSpecNode* nodes, uint32_t count, uint32_t parent) noexcept {
    if (nodes == nullptr || parent >= count) return 0u;
    uint32_t s = 0u;
    for (uint32_t i = 1u; i < count; ++i) {
        if (uint32_t(nodes[i].parent) == parent) ++s;
    }
    return s;
}

inline uint32_t tree_spec_children(
    const TreeSpecNode* nodes,
    uint32_t count,
    uint32_t parent,
    uint32_t* out,
    uint32_t cap) noexcept
{
    if (nodes == nullptr || out == nullptr || parent >= count) return 0u;
    uint32_t n = 0u;
    for (uint32_t i = 1u; i < count; ++i) {
        if (uint32_t(nodes[i].parent) == parent && n < cap) out[n++] = i;
    }
    return n;
}

inline bool tree_spec_allowed(const TreeSpecPlan& plan, uint32_t row, uint32_t ancestor_row) noexcept {
    if (!plan.valid || row >= plan.node_count || ancestor_row >= plan.node_count) return false;
    return (plan.allow_mask[row] & uint16_t(1u << ancestor_row)) != 0u;
}

// Exact top-k truncation with renormalization over the kept support (k >= 1).
inline uint32_t truncate_top_k(
    float* probs,
    uint32_t n,
    uint32_t top_k,
    uint32_t* kept_ids) noexcept
{
    if (probs == nullptr || kept_ids == nullptr || n == 0u) return 0u;
    top_k = std::max<uint32_t>(1u, std::min<uint32_t>(top_k, n));
    std::vector<uint32_t> cand(n);
    for (uint32_t i = 0u; i < n; ++i) cand[i] = i;
    std::sort(cand.begin(), cand.end(), [&](uint32_t a, uint32_t b) {
        if (probs[a] != probs[b]) return probs[a] > probs[b];
        return a < b;
    });
    double sum = 0.0;
    std::vector<float> kept_vals(top_k);
    for (uint32_t j = 0u; j < top_k; ++j) {
        kept_ids[j] = cand[j];
        kept_vals[j] = probs[cand[j]];
        sum += double(probs[cand[j]]);
    }
    for (uint32_t i = 0u; i < n; ++i) probs[i] = 0.0f;
    if (sum <= 0.0) return 0u;
    const double inv = 1.0 / sum;
    for (uint32_t j = 0u; j < top_k; ++j) {
        probs[kept_ids[j]] = float(double(kept_vals[j]) * inv);
    }
    return top_k;
}

// Exact top-p truncation with renormalization over the kept support (p in (0,1)).
inline uint32_t truncate_top_p(
    float* probs,
    uint32_t n,
    float top_p,
    uint32_t* kept_ids) noexcept
{
    if (probs == nullptr || kept_ids == nullptr || n == 0u) return 0u;
    if (top_p <= 0.0f || top_p >= 1.0f) return 0u;
    std::vector<uint32_t> cand;
    cand.reserve(n);
    double total = 0.0;
    for (uint32_t i = 0u; i < n; ++i) {
        if (probs[i] > 0.0f) {
            cand.push_back(i);
            total += double(probs[i]);
        }
    }
    if (cand.empty() || total <= 0.0) return 0u;
    std::sort(cand.begin(), cand.end(), [&](uint32_t a, uint32_t b) {
        if (probs[a] != probs[b]) return probs[a] > probs[b];
        return a < b;
    });
    double cum = 0.0;
    uint32_t keep = 0u;
    for (; keep < cand.size(); ++keep) {
        cum += double(probs[cand[keep]]) / total;
        if (cum >= double(top_p)) { ++keep; break; }
    }
    if (keep == 0u) { keep = 1u; cum = double(probs[cand[0]]) / total; }
    const double inv = 1.0 / std::max(cum, 1e-300);
    for (uint32_t j = 0u; j < keep; ++j) {
        kept_ids[j] = cand[j];
        probs[cand[j]] = float(double(probs[cand[j]]) / total * inv);
    }
    for (uint32_t i = 0u; i < n; ++i) {
        bool retained = false;
        for (uint32_t j = 0u; j < keep; ++j) {
            if (kept_ids[j] == i) { retained = true; break; }
        }
        if (!retained) probs[i] = 0.0f;
    }
    return keep;
}

inline double borel_accept_prob(double p_x, double q_x) noexcept {
    if (p_x <= 0.0 || q_x <= 0.0) return 0.0;
    return std::min(1.0, p_x / q_x);
}

inline bool borel_accept(double p_x, double q_x, double u) noexcept {
    return u <= borel_accept_prob(p_x, q_x);
}

inline double borel_residual_mass(
    const double* p,
    const double* q,
    uint32_t n) noexcept
{
    if (p == nullptr || q == nullptr) return 0.0;
    double s = 0.0;
    for (uint32_t i = 0u; i < n; ++i) s += std::max(0.0, p[i] - q[i]);
    return s;
}

inline double borel_residual_prob(
    const double* p,
    const double* q,
    uint32_t n,
    uint32_t i,
    double mass) noexcept
{
    if (p == nullptr || q == nullptr || i >= n || mass <= 0.0) return 0.0;
    return std::max(0.0, p[i] - q[i]) / mass;
}

inline int32_t borel_sample_residual(
    const double* p,
    const double* q,
    uint32_t n,
    std::mt19937_64& rng) noexcept
{
    if (p == nullptr || q == nullptr || n == 0u) return -1;
    const double mass = borel_residual_mass(p, q, n);
    if (mass <= 0.0) return -1;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double r = dist(rng);
    double acc = 0.0;
    int32_t last = -1;
    for (uint32_t i = 0u; i < n; ++i) {
        const double v = std::max(0.0, p[i] - q[i]) / mass;
        if (v > 0.0) {
            last = int32_t(i);
            acc += v;
            if (r <= acc) return int32_t(i);
        }
    }
    return last;
}

// Exact conditional next-token distribution from one sibling proposal:
//   P_tree_spec(y) = min(p(y), q(y)) + max(0, p(y)-q(y)) = P_target(y).
inline double borel_verified_prob(
    const double* p,
    const double* q,
    uint32_t n,
    uint32_t y) noexcept
{
    if (p == nullptr || q == nullptr || y >= n) return 0.0;
    return std::min(p[y], q[y]) + std::max(0.0, p[y] - q[y]);
}

// Pick ONE sibling proposal uniformly at random (independent of target), then run
// the exact Borel accept/reject. Because every sibling proposal at a node shares
// the same conditional draft q, the accepted branch distribution is exactly P_target.
inline uint32_t borel_choose_sibling(
    uint32_t sibling_count,
    std::mt19937_64& rng) noexcept
{
    if (sibling_count == 0u) return 0u;
    std::uniform_int_distribution<uint32_t> dist(0u, sibling_count - 1u);
    return dist(rng);
}

// TV distance between P_tree_spec and P_target for a closed support n.
inline double borel_tree_tv(
    const double* p,
    const double* q,
    uint32_t n) noexcept
{
    if (p == nullptr || q == nullptr || n == 0u) return 1.0;
    double tv = 0.0;
    for (uint32_t y = 0u; y < n; ++y) tv += std::fabs(borel_verified_prob(p, q, n, y) - p[y]);
    return tv;
}

// One-node exact verification: draft x sampled from q, accepted with p/q,
// rejected residual max(p-q).  Enumerate output probability for each y.
// Exact unconditional node output probability after draft x is sampled from q:
//   accepted x -> min(p(y), q(y)); rejected -> residual max(0, p(y)-q(y)).
// The rejection mass equals residual mass, so P_tree_spec(y) = min(p(y),q(y))
// + max(0,p(y)-q(y)) = P_target(y) for every y (x cancels out).
inline double borel_node_output_prob(
    const double* p,
    const double* q,
    uint32_t n,
    uint32_t /*x*/,
    uint32_t y) noexcept
{
    if (p == nullptr || q == nullptr || y >= n) return 0.0;
    return std::min(p[y], q[y]) + std::max(0.0, p[y] - q[y]);
}

// Node output TV against target p under a shared draft q.
inline double borel_node_tv(
    const double* p,
    const double* q,
    uint32_t n,
    uint32_t /*x*/) noexcept
{
    if (p == nullptr || q == nullptr || n == 0u) return 1.0;
    double tv = 0.0;
    for (uint32_t y = 0u; y < n; ++y) {
        tv += std::fabs(borel_node_output_prob(p, q, n, 0u, y) - p[y]);
    }
    return tv;
}

}  // namespace dsv41
