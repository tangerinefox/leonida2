// Process-wide counts of calls recorded and replay attempts. A command list can replay more than once.
#pragma once
#include <atomic>
#include <cstdio>

namespace leonida::indirect_stats {
enum class Event { recorded, emitted, record_skip, replay_skip };
struct Counts {
    std::atomic<unsigned long long> values[4]{};
    void report() const {
        std::fprintf(stderr, "[rt] indirect graphics: recorded=%llu emitted=%llu record_skips=%llu replay_skips=%llu\n",
                     values[0].load(), values[1].load(), values[2].load(), values[3].load());
    }
    ~Counts() {
        for (const auto& value : values) {
            if (value.load()) {
                report();
                break;
            }
        }
    }
};
inline void note(Event event) {
    static Counts counts;
    const auto n = 1 + counts.values[static_cast<unsigned>(event)].fetch_add(1, std::memory_order_relaxed);
    if ((n & (n - 1)) == 0) counts.report();
}
// One flag per reason, shared across resolver instantiations.
enum Reason { compound, disabled, pipeline, root_layout, viewport, mesh, topology, first_instance, entry, robust_index,
              count_limit, multi_draw, count_feature, raw_count_limit, argument_usage, count_usage,
              divisor, pending_pipeline, index_view, index_range, stateless, reason_count };
inline void skip(Event event, Reason reason) {
    static constexpr const char* names[]{"compound signature", "opt-out", "missing pipeline", "missing root signature",
        "empty viewport", "mesh indirect", "topology", "drawIndirectFirstInstance", "missing Vulkan entry point",
        "robust sized index binding", "maxDrawIndirectCount", "multiDrawIndirect", "drawIndirectCount",
        "counted draw requires maxDrawIndirectCount == UINT32_MAX", "argument buffer lacks indirect creation usage",
        "count buffer lacks indirect creation usage", "nonunit vertex divisor", "pipeline pending",
        "invalid index view", "index view outside buffer or misaligned", "legacy command has no graphics state"};
    static_assert(sizeof(names) / sizeof(*names) == reason_count);
    static std::atomic_flag shown[reason_count]{};
    if (!shown[reason].test_and_set(std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] indirect graphics skipped: %s\n", names[reason]);
    note(event);
}
}
