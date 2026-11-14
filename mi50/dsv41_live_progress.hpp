#pragma once
// Pure-C++17 Real-Time Native Telemetry Tracker (`x_flash_progress` v1) for DeepSeek-V4.1-Flash-1017 HTTP/SSE Server
// Emits rolling token-rate histograms (`flashTokenTotals` / `x_flash_progress`) over OpenAI and Anthropic SSE streams.
// Counts all Input (cached + fresh prefill chunks) and Output (reasoning + text + buffered DSML tool calls) in real time.

#include "dsv41_json.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace dsv41 {

struct Dsv41LiveProgress {
    struct Activity {
        const uint64_t ms;
        uint64_t index = 0;
        uint64_t current = 0;
        uint64_t last = 0;
        uint64_t last_active = 0;
        uint64_t active_buckets = 0;
        uint64_t active_tokens = 0;
        std::map<uint64_t, uint64_t> histogram;

        explicit Activity(uint64_t width_ms) : ms(width_ms) {}

        void close_until(uint64_t next) {
            if (next <= index) return;
            if (current > 0) {
                ++active_buckets;
                active_tokens += current;
                ++histogram[current];
                last_active = current;
            }
            last = (next == index + 1) ? current : 0;
            current = 0;
            index = next;
        }

        uint64_t at_rank(uint64_t rank) const {
            for (const auto& kv : histogram) {
                if (rank < kv.second) return kv.first;
                rank -= kv.second;
            }
            return 0;
        }

        JsonValue view(double elapsed_ms) const {
            JsonValue a = JsonValue::make_object();
            a.set_key("window_ms", JsonValue::make_int(int64_t(ms)));
            a.set_key("completed_buckets", JsonValue::make_int(int64_t(index)));
            a.set_key("active_buckets", JsonValue::make_int(int64_t(active_buckets)));
            a.set_key("active_tokens", JsonValue::make_int(int64_t(active_tokens)));
            a.set_key("active_ms", JsonValue::make_int(int64_t(active_buckets * ms)));
            a.set_key("inactive_ms", JsonValue::make_int(int64_t((index - active_buckets) * ms)));
            a.set_key("partial_tokens", JsonValue::make_int(int64_t(current)));
            a.set_key("partial_ms", JsonValue::make_double(std::max(0.0, elapsed_ms - double(index * ms))));
            a.set_key("last_bucket_tokens", JsonValue::make_int(int64_t(last)));
            a.set_key("last_active_tokens", JsonValue::make_int(int64_t(last_active)));
            a.set_key("duty_cycle", JsonValue::make_double(elapsed_ms > 0.0 ? double(active_buckets * ms) / elapsed_ms : 0.0));
            auto rate = [&](double n) { return JsonValue::make_double(n * 1000.0 / double(ms)); };
            a.set_key("active_tps", active_buckets ? rate(double(active_tokens) / double(active_buckets)) : JsonValue::make_null());
            a.set_key("mean_tps", active_buckets ? rate(double(active_tokens) / double(active_buckets)) : JsonValue::make_null());
            a.set_key("median_tps", active_buckets
                                        ? rate(double(at_rank((active_buckets - 1) / 2) + at_rank(active_buckets / 2)) * 0.5)
                                        : JsonValue::make_null());
            a.set_key("p95_tps", active_buckets ? rate(double(at_rank((95 * active_buckets + 99) / 100 - 1))) : JsonValue::make_null());
            a.set_key("max_tps", active_buckets ? rate(double(histogram.rbegin()->first)) : JsonValue::make_null());
            return a;
        }
    };

    struct WorkInterval {
        uint64_t seq = 0;
        uint64_t cached = 0;
        uint64_t fresh = 0;
        uint64_t generated = 0;
        int64_t begin = 0;
        int64_t end = 0;
    };

    const bool enabled;
    std::atomic<uint64_t> count{0};
    std::atomic<int64_t> first_ns{0};
    std::atomic<int64_t> end_ns{0};
    std::atomic<bool> input_ready{false};

    uint64_t input_total = 0;
    uint64_t input_cached = 0;
    std::atomic<uint64_t> input_fresh{0};

    std::mutex mtx_;
    std::array<Activity, 4> activity{{Activity(100), Activity(250), Activity(500), Activity(1000)}};
    std::vector<WorkInterval> pending_work;
    uint64_t work_seq = 0;
    uint64_t work_cached = 0;
    uint64_t work_fresh = 0;
    uint64_t work_generated = 0;
    uint64_t unmeasured_work = 0;
    int64_t work_epoch = 0;
    int64_t work_active_ns = 0;
    int64_t last_prefill_end_ns = 0;
    int64_t cached_active_ns = 0;
    int64_t fresh_active_ns = 0;
    int64_t generated_active_ns = 0;

    uint64_t prefill_tokens = 0;
    uint64_t prefill_sample_tokens = 0;
    double prefill_elapsed_ms = 0.0;
    double prefill_sample_ms = 0.0;
    uint64_t live_work_tokens = 0;
    int64_t live_work_ns = 0;
    bool prefill_active = false;
    bool final_partial_recorded = false;

    explicit Dsv41LiveProgress(bool on = true) : enabled(on), work_epoch(now_ns()) {}

    static int64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    void add_work_locked(uint64_t cached, uint64_t fresh, uint64_t generated, int64_t begin, int64_t end) {
        if (!cached && !fresh && !generated) return;
        if (work_epoch <= 0) work_epoch = (begin > 0) ? begin : now_ns();
        const int64_t b = std::max<int64_t>(begin, work_epoch);
        const int64_t e = std::max<int64_t>(end, b + 1);
        const int64_t dur = e - b;
        pending_work.push_back({++work_seq, cached, fresh, generated, b, e});
        work_cached += cached;
        work_fresh += fresh;
        work_generated += generated;
        work_active_ns += dur;
        if (cached) cached_active_ns += dur;
        if (fresh) fresh_active_ns += dur;
        if (generated) generated_active_ns += dur;
    }

    void close_activity_locked(Activity& a, uint64_t next) {
        if (a.ms == 100 && next > a.index && a.current > 0) {
            const int64_t start = first_ns.load(std::memory_order_relaxed) + int64_t(a.index * 100000000ULL);
            add_work_locked(0, 0, a.current, start, start + 100000000LL);
        }
        a.close_until(next);
    }

    void on_prefix_done(uint64_t total_prompt, uint64_t cached, int64_t begin_ns, int64_t done_ns) {
        if (!enabled) return;
        std::lock_guard<std::mutex> lk(mtx_);
        input_total = total_prompt;
        input_cached = std::min<uint64_t>(cached, total_prompt);
        if (begin_ns > 0 && begin_ns < work_epoch) work_epoch = begin_ns;
        if (input_cached > 0) {
            add_work_locked(input_cached, 0, 0, begin_ns, done_ns);
        }
        last_prefill_end_ns = std::max<int64_t>(done_ns, work_epoch);
        prefill_active = (input_cached < input_total);
        input_ready.store(true, std::memory_order_release);
    }

    void on_prefill_chunk(uint64_t chunk_tokens, double chunk_ms, int64_t begin_ns, int64_t done_ns) {
        if (!enabled || chunk_tokens == 0) return;
        std::lock_guard<std::mutex> lk(mtx_);
        const uint64_t max_fresh = (input_total > input_cached) ? (input_total - input_cached) : 0;
        const uint64_t cur_fresh = input_fresh.load(std::memory_order_relaxed);
        const uint64_t add_fresh = (max_fresh > cur_fresh) ? std::min<uint64_t>(chunk_tokens, max_fresh - cur_fresh) : 0;
        prefill_tokens += add_fresh;
        prefill_elapsed_ms += std::max(0.0, chunk_ms);
        prefill_sample_tokens = chunk_tokens;
        prefill_sample_ms = std::max(0.001, chunk_ms);
        if (add_fresh > 0) {
            input_fresh.store(cur_fresh + add_fresh, std::memory_order_release);
            add_work_locked(0, add_fresh, 0, begin_ns, done_ns);
        }
        last_prefill_end_ns = std::max<int64_t>(done_ns, last_prefill_end_ns);
        prefill_active = ((cur_fresh + add_fresh) < max_fresh);
    }

    void observe_token(uint64_t native_count) {
        if (!enabled) return;
        std::lock_guard<std::mutex> lk(mtx_);
        const int64_t ts = now_ns();
        const uint64_t prior = count.load(std::memory_order_relaxed);
        if (native_count <= prior) return;
        if (prior == 0) {
            const uint64_t max_fresh = (input_total > input_cached) ? (input_total - input_cached) : 0;
            const uint64_t cur_fresh = input_fresh.load(std::memory_order_relaxed);
            if (max_fresh > cur_fresh) {
                const uint64_t rem = max_fresh - cur_fresh;
                input_fresh.store(max_fresh, std::memory_order_release);
                add_work_locked(0, rem, 0, last_prefill_end_ns > 0 ? last_prefill_end_ns : work_epoch, ts);
            }
            prefill_active = false;
            first_ns.store(ts, std::memory_order_relaxed);
        }
        const int64_t start = first_ns.load(std::memory_order_relaxed);
        if (start > 0) {
            const uint64_t elapsed_ns = uint64_t(std::max<int64_t>(0, ts - start));
            for (auto& a : activity) {
                close_activity_locked(a, elapsed_ns / (a.ms * 1000000ULL));
                a.current += (native_count - prior);
            }
        }
        count.store(native_count, std::memory_order_release);
    }

    void finish() {
        if (!enabled) return;
        std::lock_guard<std::mutex> lk(mtx_);
        const int64_t ts = now_ns();
        prefill_active = false;
        end_ns.store(ts, std::memory_order_release);
        const int64_t start = first_ns.load(std::memory_order_relaxed);
        if (start <= 0 || final_partial_recorded) return;
        const uint64_t elapsed_ns = uint64_t(std::max<int64_t>(0, ts - start));
        for (auto& a : activity) {
            close_activity_locked(a, elapsed_ns / (a.ms * 1000000ULL));
        }
        const auto& a0 = activity[0];
        if (a0.current > 0) {
            add_work_locked(0, 0, a0.current, start + int64_t(a0.index * 100000000ULL), ts);
        }
        final_partial_recorded = true;
    }

    JsonValue sample(const std::string& request_id, bool is_final, uint64_t* out_serviced = nullptr,
                     uint64_t* out_total = nullptr, uint64_t* out_cached = nullptr, uint64_t* out_gen = nullptr) {
        std::lock_guard<std::mutex> lk(mtx_);
        const int64_t ts = now_ns();
        const uint64_t n = count.load(std::memory_order_acquire);
        const int64_t start = first_ns.load(std::memory_order_relaxed);
        const int64_t end = end_ns.load(std::memory_order_acquire);
        const int64_t active_end = (end > 0) ? end : ts;
        const int64_t elapsed_ns = (start > 0) ? std::max<int64_t>(0, active_end - start) : 0;

        JsonValue p = JsonValue::make_object();
        p.set_key("version", JsonValue::make_int(1));
        p.set_key("request_id", request_id);
        p.set_key("generated_tokens", JsonValue::make_int(int64_t(n)));
        p.set_key("generation_elapsed_ms", JsonValue::make_double(double(elapsed_ns) / 1e6));
        p.set_key("generation_window_ms", JsonValue::make_int(100));

        JsonValue scales = JsonValue::make_array();
        for (auto& a : activity) {
            close_activity_locked(a, uint64_t(elapsed_ns) / (a.ms * 1000000ULL));
            scales.push_back(a.view(double(elapsed_ns) / 1e6));
        }
        p.set_key("generation_activity", std::move(scales));

        JsonValue work = JsonValue::make_object();
        JsonValue intervals = JsonValue::make_array();
        uint64_t newly_serviced = 0;
        int64_t newly_active_ns = 0;
        for (const auto& iv : pending_work) {
            JsonValue item = JsonValue::make_object();
            item.set_key("seq", JsonValue::make_int(int64_t(iv.seq)));
            item.set_key("start_ns", JsonValue::make_int(std::max<int64_t>(0, iv.begin - work_epoch)));
            item.set_key("end_ns", JsonValue::make_int(std::max<int64_t>(1, iv.end - work_epoch)));
            item.set_key("cached_tokens", JsonValue::make_int(int64_t(iv.cached)));
            item.set_key("fresh_tokens", JsonValue::make_int(int64_t(iv.fresh)));
            item.set_key("generated_tokens", JsonValue::make_int(int64_t(iv.generated)));
            intervals.push_back(std::move(item));
            newly_serviced += iv.cached + iv.fresh + iv.generated;
            newly_active_ns += (iv.end - iv.begin);
        }
        if (newly_serviced > 0) {
            live_work_tokens = newly_serviced;
            live_work_ns = newly_active_ns;
        }
        pending_work.clear();

        work.set_key("intervals", std::move(intervals));
        work.set_key("interval_count", JsonValue::make_int(int64_t(work_seq)));
        work.set_key("cached_tokens", JsonValue::make_int(int64_t(work_cached)));
        work.set_key("fresh_tokens", JsonValue::make_int(int64_t(work_fresh)));
        work.set_key("generated_tokens", JsonValue::make_int(int64_t(work_generated)));
        work.set_key("unmeasured_tokens", JsonValue::make_int(int64_t(unmeasured_work)));
        work.set_key("active_ns", JsonValue::make_int(std::max<int64_t>(0, work_active_ns)));
        work.set_key("request_wall_ns", JsonValue::make_int(work_epoch > 0 ? std::max<int64_t>(0, active_end - work_epoch) : 0));
        work.set_key("cached_active_ns", JsonValue::make_int(std::max<int64_t>(0, cached_active_ns)));
        work.set_key("fresh_active_ns", JsonValue::make_int(std::max<int64_t>(0, fresh_active_ns)));
        work.set_key("generated_active_ns", JsonValue::make_int(std::max<int64_t>(0, generated_active_ns)));
        work.set_key("live_tokens", JsonValue::make_int(int64_t(live_work_tokens)));
        work.set_key("live_active_ns", JsonValue::make_int(std::max<int64_t>(0, live_work_ns)));
        work.set_key("new_work", JsonValue::make_bool(newly_serviced > 0));
        work.set_key("input_processing", JsonValue::make_bool(prefill_active));
        p.set_key("token_work", std::move(work));

        if (activity[0].index > 0) {
            p.set_key("generation_window_end_ms", JsonValue::make_int(int64_t(activity[0].index * 100ULL)));
            p.set_key("generation_window_tokens", JsonValue::make_int(int64_t(activity[0].last)));
        }
        p.set_key("phase", is_final ? "paused" : (n > 0 ? "generating" : "prefill"));
        p.set_key("prefill_tokens", JsonValue::make_int(int64_t(prefill_tokens)));
        p.set_key("prefill_elapsed_ms", JsonValue::make_double(prefill_elapsed_ms));
        p.set_key("prefill_sample_tokens", JsonValue::make_int(int64_t(prefill_sample_tokens)));
        p.set_key("prefill_sample_ms", JsonValue::make_double(prefill_sample_ms));

        const uint64_t fresh = input_fresh.load(std::memory_order_acquire);
        const uint64_t serviced = input_ready.load(std::memory_order_acquire) ? (input_cached + fresh) : 0;
        p.set_key("input_tokens_total", JsonValue::make_int(int64_t(input_total)));
        p.set_key("cached_input_tokens", JsonValue::make_int(int64_t(input_cached)));
        p.set_key("fresh_input_tokens", JsonValue::make_int(int64_t(fresh)));
        p.set_key("input_serviced_tokens", JsonValue::make_int(int64_t(serviced)));
        p.set_key("timing_boundary", "frontend_first_native_token_to_engine_reply");

        if (out_serviced) *out_serviced = serviced;
        if (out_total) *out_total = input_total;
        if (out_cached) *out_cached = input_cached;
        if (out_gen) *out_gen = n;
        return p;
    }
};

}  // namespace dsv41
