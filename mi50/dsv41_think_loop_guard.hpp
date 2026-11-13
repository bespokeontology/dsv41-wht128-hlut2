// mi50/dsv41_think_loop_guard.hpp
// Minimal Zero-Quality-Loss <think> Stream Repetition Guard for dsv41_fe.cpp
// 100% Pure C++17, Zero Heap Allocations on Hot Path, Zero Sampling/Logit Mutation.
//
// Compile standalone self-test:
//   g++ -O3 -std=c++17 -Wall -Wextra -Werror -x c++ -DDSV41_THINK_GUARD_SELFTEST mi50/dsv41_think_loop_guard.hpp -o /tmp/test_think_guard && /tmp/test_think_guard

#ifndef DSV41_THINK_LOOP_GUARD_HPP_
#define DSV41_THINK_LOOP_GUARD_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

namespace dsv41 {

struct ThinkLoopConfig {
  // Detector A: Exact periodic token cycle t[i] == t[i - p] for p in [1 .. kMaxPeriod].
  // Fires only when BOTH consecutive matched tokens >= min_periodic_tokens AND cycles >= min_cycles.
  uint32_t min_periodic_tokens = 128; // e.g., 64x 2-tok "Hmm.\n" or 16x 8-tok phrase
  uint32_t min_periodic_cycles = 16;  // Prevents firing on legitimate 3-4x code/table templates

  // Detector B: Aperiodic low-entropy stutter (e.g., random mix of "Hmm", ".", " Wait", "\n")
  // over a 384-token window after warmup, catching non-exact filler loops while sparing
  // 56-row numerical manifest audits (which always carry >= 25+ distinct digit/identifier tokens).
  uint32_t entropy_window = 384;
  uint32_t max_stutter_unique_ids = 8;

  // Number of initial loop tokens to keep before rewinding CSA2 KV cursor on ForceCloseThink.
  uint32_t keep_prefix_tokens_on_rewind = 4;
};

enum class ThinkGuardAction : uint8_t {
  kContinue = 0,
  kForceCloseThink = 1, // Rewind KV tail + inject "</think>" so model emits tool call or answer
};

struct ThinkGuardDecision {
  ThinkGuardAction action = ThinkGuardAction::kContinue;
  uint32_t detected_period = 0; // >0 if triggered by Detector A, 0 if Detector B
  uint32_t rewind_tokens = 0;   // Stateless CSA2 KV slots to roll back before injecting </think>
  uint32_t think_tokens_seen = 0;
};

class Dsv41ThinkLoopGuard {
 public:
  static constexpr uint32_t kRingCap = 512; // Power of two >= entropy_window & min_periodic_tokens
  static constexpr uint32_t kRingMask = kRingCap - 1u;
  static constexpr uint32_t kMaxPeriod = 32;
  static constexpr uint32_t kHashSlots = 1024; // Open-addressed refcount table for O(1) unique count

  explicit Dsv41ThinkLoopGuard(ThinkLoopConfig cfg = {}) : cfg_(cfg) { reset(); }

  void reset() noexcept {
    ring_.fill(0);
    period_run_.fill(0);
    ht_key_.fill(kEmptyKey);
    ht_cnt_.fill(0);
    think_pos_ = 0;
    win_unique_ = 0;
    in_think_ = false;
    forced_closed_ = false;
  }

  void enter_think() noexcept {
    if (!forced_closed_) {
      in_think_ = true;
      think_pos_ = 0;
      win_unique_ = 0;
      period_run_.fill(0);
      ht_key_.fill(kEmptyKey);
      ht_cnt_.fill(0);
    }
  }

  void exit_think() noexcept { in_think_ = false; }
  void resume_think() noexcept {
    if (!forced_closed_) in_think_ = true;
  }
  bool in_think() const noexcept { return in_think_; }
  bool was_forced_closed() const noexcept { return forced_closed_; }

  // Passive O(1) per-token observer called ONLY on tokens inside <think>...</think>.
  // Touches zero logits and zero RNG states during normal reasoning.
  ThinkGuardDecision observe_think_token(int32_t token_id) noexcept {
    ThinkGuardDecision dec{};
    if (!in_think_ || forced_closed_) return dec;

    const uint32_t uid = static_cast<uint32_t>(token_id);
    const uint32_t pos = think_pos_++;
    dec.think_tokens_seen = think_pos_;

    // 1. Update sliding-window unique token counter (window = cfg_.entropy_window)
    if (pos >= cfg_.entropy_window) {
      const uint32_t old_uid = ring_[(pos - cfg_.entropy_window) & kRingMask];
      dec_unique(old_uid);
    }
    inc_unique(uid);

    // 2. Update O(kMaxPeriod) exact periodic run lengths before overwriting ring slot
    uint32_t triggered_p = 0;
    const uint32_t max_p = std::min<uint32_t>(kMaxPeriod, pos);
    for (uint32_t p = 1; p <= max_p; ++p) {
      if (ring_[(pos - p) & kRingMask] == uid) {
        const uint32_t run = ++period_run_[p];
        const uint32_t total_span = run + p;
        if (run >= cfg_.min_periodic_tokens &&
            total_span >= p * cfg_.min_periodic_cycles) {
          triggered_p = p;
          break;
        }
      } else {
        period_run_[p] = 0;
      }
    }
    ring_[pos & kRingMask] = uid;

    if (triggered_p > 0) {
      in_think_ = false;
      forced_closed_ = true;
      const uint32_t total_loop_toks = period_run_[triggered_p] + triggered_p;
      const uint32_t keep = std::min(total_loop_toks, std::max(triggered_p, cfg_.keep_prefix_tokens_on_rewind));
      dec.action = ThinkGuardAction::kForceCloseThink;
      dec.detected_period = triggered_p;
      dec.rewind_tokens = total_loop_toks - keep;
      return dec;
    }

    // 3. Check aperiodic vocabulary collapse ("Hmm. Wait. Hmm, wait... ")
    if (think_pos_ >= cfg_.entropy_window && win_unique_ <= cfg_.max_stutter_unique_ids) {
      in_think_ = false;
      forced_closed_ = true;
      dec.action = ThinkGuardAction::kForceCloseThink;
      dec.detected_period = 0;
      dec.rewind_tokens = cfg_.entropy_window - std::min(cfg_.entropy_window, cfg_.keep_prefix_tokens_on_rewind);
      return dec;
    }

    return dec;
  }

 private:
  static constexpr uint32_t kEmptyKey = 0xFFFFFFFFu;

  void inc_unique(uint32_t key) noexcept {
    uint32_t slot = (key * 2654435761u) & (kHashSlots - 1u);
    while (true) {
      if (ht_key_[slot] == key) {
        ++ht_cnt_[slot];
        return;
      }
      if (ht_key_[slot] == kEmptyKey) {
        ht_key_[slot] = key;
        ht_cnt_[slot] = 1;
        ++win_unique_;
        return;
      }
      slot = (slot + 1u) & (kHashSlots - 1u);
    }
  }

  void dec_unique(uint32_t key) noexcept {
    uint32_t slot = (key * 2654435761u) & (kHashSlots - 1u);
    while (ht_key_[slot] != kEmptyKey) {
      if (ht_key_[slot] == key) {
        if (--ht_cnt_[slot] == 0) {
          --win_unique_;
          erase_and_rehash(slot);
        }
        return;
      }
      slot = (slot + 1u) & (kHashSlots - 1u);
    }
  }

  void erase_and_rehash(uint32_t empty_slot) noexcept {
    uint32_t cur = (empty_slot + 1u) & (kHashSlots - 1u);
    while (ht_key_[cur] != kEmptyKey) {
      const uint32_t ideal = (ht_key_[cur] * 2654435761u) & (kHashSlots - 1u);
      const bool should_move = (empty_slot <= cur)
                                   ? (ideal <= empty_slot || ideal > cur)
                                   : (ideal <= empty_slot && ideal > cur);
      if (should_move) {
        ht_key_[empty_slot] = ht_key_[cur];
        ht_cnt_[empty_slot] = ht_cnt_[cur];
        empty_slot = cur;
      }
      cur = (cur + 1u) & (kHashSlots - 1u);
    }
    ht_key_[empty_slot] = kEmptyKey;
    ht_cnt_[empty_slot] = 0;
  }

  ThinkLoopConfig cfg_{};
  std::array<uint32_t, kRingCap> ring_{};
  std::array<uint32_t, kMaxPeriod + 1> period_run_{};
  std::array<uint32_t, kHashSlots> ht_key_{};
  std::array<uint16_t, kHashSlots> ht_cnt_{};
  uint32_t think_pos_ = 0;
  uint32_t win_unique_ = 0;
  bool in_think_ = false;
  bool forced_closed_ = false;
};

} // namespace dsv41

#ifdef DSV41_THINK_GUARD_SELFTEST
#include <chrono>
#include <iostream>
#include <vector>

int main() {
  using namespace dsv41;

  // Test 1: Exact 2-token "Hmm.\n" loop triggers cleanly and computes exact rewind count
  {
    Dsv41ThinkLoopGuard g;
    g.enter_think();
    for (int i = 0; i < 40; ++i) {
      if (g.observe_think_token(1000 + i).action != ThinkGuardAction::kContinue) return 1;
    }
    ThinkGuardDecision trig{};
    int emitted = 0;
    for (int i = 0; i < 500; ++i) {
      ++emitted;
      trig = g.observe_think_token((i & 1) ? 4021 : 8812); // "Hmm" + ".\n"
      if (trig.action == ThinkGuardAction::kForceCloseThink) break;
    }
    if (trig.action != ThinkGuardAction::kForceCloseThink || trig.detected_period != 2 || emitted != 130) {
      std::cerr << "FAIL Test 1: emitted=" << emitted << " p=" << trig.detected_period << "\n";
      return 1;
    }
    std::cout << "[PASS] Test 1 (2-token 'Hmm.' loop): caught at " << emitted
              << " loop tokens (vs 20,671), rewind_tokens=" << trig.rewind_tokens << "\n";
  }

  // Test 2: Aperiodic 5-word stutter ("Hmm", ".", " Wait", ",", "\n" in pseudo-random order)
  {
    Dsv41ThinkLoopGuard g;
    g.enter_think();
    const int32_t stutter_vocab[5] = {11, 22, 33, 44, 55};
    uint64_t s = 0xCAFEBABEULL;
    ThinkGuardDecision trig{};
    int emitted = 0;
    for (int i = 0; i < 600; ++i) {
      s = s * 6364136223846793005ULL + 1ULL;
      ++emitted;
      trig = g.observe_think_token(stutter_vocab[(s >> 33) % 5]);
      if (trig.action == ThinkGuardAction::kForceCloseThink) break;
    }
    if (trig.action != ThinkGuardAction::kForceCloseThink || emitted != 384) {
      std::cerr << "FAIL Test 2: emitted=" << emitted << "\n";
      return 1;
    }
    std::cout << "[PASS] Test 2 (aperiodic 5-token hesitation stutter): caught at "
              << emitted << " tokens, rewind_tokens=" << trig.rewind_tokens << "\n";
  }

  // Test 3: Structured 60-row manifest table audit MUST NEVER false-trigger!
  {
    Dsv41ThinkLoopGuard g;
    g.enter_think();
    for (int row = 1; row <= 60; ++row) {
      const int32_t row_toks[18] = {
          501, 502, static_cast<int32_t>(700 + row), 503, static_cast<int32_t>(900 + (row * 7) % 384),
          504, 505, static_cast<int32_t>(1200 + (row % 19)), 506, 507,
          static_cast<int32_t>(1400 + (row * 3) % 29), 508, 509, 510,
          static_cast<int32_t>(1600 + (row % 11)), 511, 512, 513};
      for (int32_t t : row_toks) {
        if (g.observe_think_token(t).action != ThinkGuardAction::kContinue) {
          std::cerr << "FAIL Test 3: false positive on manifest row " << row << "\n";
          return 1;
        }
      }
    }
    std::cout << "[PASS] Test 3 (60-row structured manifest audit = 1,080 tokens): 0 false positives\n";
  }

  // Test 4: 10,024-token mathematical <think> derivation latency & zero false positives
  {
    Dsv41ThinkLoopGuard g;
    g.enter_think();
    uint64_t s = 0x20261112ULL;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10024; ++i) {
      s = s * 2862933555777941757ULL + 3037000493ULL;
      const int32_t tok = static_cast<int32_t>((s >> 32) % 4096);
      if (g.observe_think_token(tok).action != ThinkGuardAction::kContinue) {
        std::cerr << "FAIL Test 4: false positive at token " << i << "\n";
        return 1;
      }
    }
    const double ns_per_tok =
        std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / 10024.0;
    std::cout << "[PASS] Test 4 (10,024-token reasoning stream): 0 false positives ("
              << ns_per_tok << " ns/token CPU overhead)\n";
  }

  return 0;
}
#endif // DSV41_THINK_GUARD_SELFTEST

#endif // DSV41_THINK_LOOP_GUARD_HPP_
