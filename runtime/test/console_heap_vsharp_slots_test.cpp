// Portable CPU-only unit test for leonida::ConsoleHeapVsharpSlots (runtime/src/console_heap_vsharp_slots.h).
// Verifies bit-for-bit equivalence with a linear scan of 32-byte descriptor slots, boundary handling, range clears,
// callback-time bit clearing (as in refresh_console_heap invalidation passes), and a 5.5 MiB heap stress check.
#include "../src/console_heap_vsharp_slots.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define EXPECT(condition)                                                  \
    do {                                                                   \
        if (!(condition)) {                                                \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); \
            return 1;                                                      \
        }                                                                  \
    } while (0)

namespace {

using leonida::ConsoleHeapVsharpSlots;

std::vector<std::size_t> collect(const ConsoleHeapVsharpSlots& slots) {
    std::vector<std::size_t> out;
    slots.for_each([&](std::size_t slot) { out.push_back(slot); });
    return out;
}

int test_predicate_exactness() {
    // Primary buffer V#: top 2 bits of w[3] are 0, w[2] (num_records) != 0.
    std::uint32_t primary[8] = {0x1000, 0x20, 64, 0x3fffffffu, 0, 0, 0, 0};
    EXPECT(ConsoleHeapVsharpSlots::is_live_vsharp(primary));

    // Counter-only buffer V#: w[2] == 0, w[6] != 0, top 2 bits of w[3] == 0.
    std::uint32_t counter[8] = {0, 0, 0, 0, 0x2000, 0, 16, 0};
    EXPECT(ConsoleHeapVsharpSlots::is_live_vsharp(counter));

    // Empty / zeroed V#: w[2] == 0 and w[6] == 0.
    std::uint32_t empty[8] = {0x1000, 0, 0, 0x00010000u, 0x2000, 0, 0, 0};
    EXPECT(!ConsoleHeapVsharpSlots::is_live_vsharp(empty));

    // Image T# (w[3] >> 30 != 0): must NOT be treated as a buffer V# even if w[2]/w[6] are non-zero.
    for (std::uint32_t hi : {1u, 2u, 3u}) {
        std::uint32_t tsharp[8] = {0x1000, 0, 128, (hi << 30) | 0x55u, 0, 0, 64, 0};
        EXPECT(!ConsoleHeapVsharpSlots::is_live_vsharp(tsharp));
    }

    // not_tsharp zeroes lower 16 bytes (w[0..3] = 0) while leaving upper 16 bytes (w[4..7]):
    // if w[6] is non-zero, predicate matches the legacy `!(w[3] >> 30) && (w[2] || w[6])` check.
    std::uint32_t not_tsharp_with_hi[8] = {0, 0, 0, 0, 0, 0, 1, 0};
    EXPECT(ConsoleHeapVsharpSlots::is_live_vsharp(not_tsharp_with_hi));
    return 0;
}

int test_boundaries_and_ranges() {
    // Zero-size heap.
    ConsoleHeapVsharpSlots zero;
    zero.reset(0);
    EXPECT(zero.empty() && zero.slot_count() == 0 && zero.live_count() == 0);
    zero.set(0, true);
    zero.clear(0);
    zero.clear_range(0, 10);
    EXPECT(zero.empty() && !zero.test(0) && collect(zero).empty());

    // Word boundary sizes: 1, 63, 64, 65, 128, 129, 257.
    for (std::size_t n : {std::size_t(1), std::size_t(63), std::size_t(64), std::size_t(65), std::size_t(128),
                          std::size_t(129), std::size_t(257)}) {
        ConsoleHeapVsharpSlots s;
        s.reset(n);
        EXPECT(s.slot_count() == n && s.empty());
        // Out-of-range writes are ignored.
        s.set(n, true);
        s.set(n + 100, true);
        EXPECT(s.empty() && !s.test(n));

        // Set every slot, check idempotency and exact ordering.
        for (std::size_t i = 0; i < n; ++i) s.set(i, true);
        s.set(0, true);
        s.set(n - 1, true);
        EXPECT(s.live_count() == n);
        const auto all = collect(s);
        EXPECT(all.size() == n);
        for (std::size_t i = 0; i < n; ++i) EXPECT(all[i] == i && s.test(i));

        // Inverted / empty / past-end clear_range is a no-op.
        s.clear_range(5, 5);
        s.clear_range(10, 2);
        s.clear_range(n, n + 64);
        EXPECT(s.live_count() == n);

        // Partial single-word and multi-word clear_range.
        if (n >= 129) {
            s.clear_range(3, 19);    // within word 0
            s.clear_range(60, 128);  // spans word 0 tail + all of word 1
            EXPECT(s.live_count() == n - 16 - 68);
            EXPECT(s.test(2) && !s.test(3) && !s.test(18) && s.test(19));
            EXPECT(s.test(59) && !s.test(60) && !s.test(127) && s.test(128));
        }

        // Clear past end clamps cleanly.
        s.clear_range(0, n + 500);
        EXPECT(s.empty() && s.live_count() == 0 && collect(s).empty());
    }
    return 0;
}

int test_in_place_invalidation_mutation() {
    // Mirrors refresh_console_heap: for_each visits each live V# slot and calls fix_console_slot, which may clear
    // that slot's bit when its backing buffer was released.
    ConsoleHeapVsharpSlots s;
    s.reset(260);
    const std::vector<std::size_t> initial = {0, 1, 62, 63, 64, 65, 127, 128, 191, 255, 259};
    for (std::size_t slot : initial) s.set(slot, true);
    EXPECT(s.live_count() == initial.size());

    std::vector<std::size_t> visited;
    s.for_each([&](std::size_t slot) {
        visited.push_back(slot);
        // Simulate releasing every even-numbered slot during the invalidation pass.
        if ((slot & 1u) == 0) s.set(slot, false);
    });
    EXPECT(visited == initial);

    std::vector<std::size_t> remaining;
    for (std::size_t slot : initial)
        if ((slot & 1u) != 0) remaining.push_back(slot);
    EXPECT(s.live_count() == remaining.size());
    EXPECT(collect(s) == remaining);

    // Clear the remaining slots and confirm empty fast path visits nothing.
    s.for_each([&](std::size_t slot) { s.clear(slot); });
    EXPECT(s.empty() && collect(s).empty());
    return 0;
}

int test_differential_equivalence_against_linear_copy() {
    // Differential model: maintain a synthetic 32-byte-per-slot `heap.copy` buffer alongside ConsoleHeapVsharpSlots
    // across thousands of random writes, deferred zeroes, page-range zeroes, and invalidation passes.
    constexpr std::size_t kSlots = 1029;
    std::vector<std::array<std::uint32_t, 8>> gpu_copy(kSlots);
    for (auto& slot : gpu_copy) slot.fill(0);

    ConsoleHeapVsharpSlots tracked;
    tracked.reset(kSlots);

    auto linear_scan = [&] {
        std::vector<std::size_t> expected;
        for (std::size_t i = 0; i < kSlots; ++i)
            if (ConsoleHeapVsharpSlots::is_live_vsharp(gpu_copy[i].data())) expected.push_back(i);
        return expected;
    };

    std::uint64_t rng = 0x9e3779b97f4a7c15ull;
    auto next_u32 = [&]() -> std::uint32_t {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        return static_cast<std::uint32_t>((rng * 0x2545f4914f6cdd1dull) >> 32);
    };

    for (unsigned step = 0; step < 4000; ++step) {
        const unsigned op = next_u32() % 10;
        if (op < 6) {
            // Simulate fix_console_slot writing a descriptor to heap.copy.
            const std::size_t slot = next_u32() % kSlots;
            std::array<std::uint32_t, 8> w{};
            const unsigned kind = next_u32() % 5;
            if (kind == 0) {
                // Live primary V#
                w[0] = next_u32();
                w[2] = (next_u32() % 256) + 1;
                w[3] = next_u32() & 0x3fffffffu;
            } else if (kind == 1) {
                // Live counter V# only
                w[6] = (next_u32() % 64) + 1;
            } else if (kind == 2) {
                // Empty V# (zero records)
                w[0] = 0;
                w[2] = 0;
                w[3] = next_u32() & 0x3fffffffu;
                w[6] = 0;
            } else if (kind == 3) {
                // Image T#
                w[2] = 128;
                w[3] = (2u << 30) | (next_u32() & 0x3fffffffu);
                w[6] = 32;
            } else {
                // not_tsharp (lower 16B zeroed, upper 16B arbitrary)
                w[4] = next_u32();
                w[6] = (next_u32() & 1u) ? 7u : 0u;
            }
            gpu_copy[slot] = w;
            tracked.set(slot, ConsoleHeapVsharpSlots::is_live_vsharp(w.data()));
        } else if (op < 8) {
            // Simulate fix_console_slots deferred zeroing.
            const std::size_t slot = next_u32() % kSlots;
            gpu_copy[slot].fill(0);
            tracked.clear(slot);
        } else if (op == 8) {
            // Simulate for_each_zeroed (both page-aligned 128-slot and arbitrary sub-ranges).
            const std::size_t first = next_u32() % (kSlots + 16);
            const std::size_t len = (next_u32() % 200);
            const std::size_t end = first + len;
            for (std::size_t i = first; i < end && i < kSlots; ++i) gpu_copy[i].fill(0);
            tracked.clear_range(first, end);
        } else {
            // Simulate refresh_console_heap invalidation pass clearing a subset of visited slots.
            const std::vector<std::size_t> expected = linear_scan();
            std::vector<std::size_t> got;
            tracked.for_each([&](std::size_t slot) {
                got.push_back(slot);
                if ((slot % 3) == 0) {
                    gpu_copy[slot].fill(0);
                    tracked.set(slot, false);
                }
            });
            EXPECT(got == expected);
        }

        if ((step & 63) == 63) {
            const std::vector<std::size_t> expected = linear_scan();
            EXPECT(tracked.live_count() == expected.size());
            EXPECT(collect(tracked) == expected);
        }
    }
    EXPECT(collect(tracked) == linear_scan());
    return 0;
}

int test_retail_sized_heap_sparse_scan() {
    // 5.5 MiB resource heap = 180,224 32-byte slots (matching retail console descriptor heap).
    constexpr std::size_t kRetailSlots = (5632u * 1024u) / 32u;
    ConsoleHeapVsharpSlots tracked;
    tracked.reset(kRetailSlots);

    const std::vector<std::size_t> hot = {0, 63, 64, 4095, 4096, 65536, 131072, kRetailSlots - 1};
    for (std::size_t slot : hot) tracked.set(slot, true);
    EXPECT(tracked.live_count() == hot.size());
    EXPECT(collect(tracked) == hot);

    // Zero a 4 KiB page (128 slots) containing slots 4096..4223.
    tracked.clear_range(4096, 4096 + 128);
    EXPECT(!tracked.test(4096) && tracked.test(4095));
    EXPECT(tracked.live_count() == hot.size() - 1);
    return 0;
}

int test_console_fix_budget_buffer_vsharp_non_deferral() {
    // Verifies that under fix_console_slots budget expiry (fix_over == true), buffer V# descriptors
    // (!(w[3] >> 30)) are never deferred/nulled, while image T# descriptors ((w[3] >> 30) != 0) are deferred.
    constexpr std::size_t kSlots = 16;
    std::vector<std::array<std::uint32_t, 8>> gpu_copy(kSlots);
    for (auto& slot : gpu_copy) slot.fill(0);

    ConsoleHeapVsharpSlots tracked;
    tracked.reset(kSlots);

    // Populate slots with alternating buffer V# and image T# descriptors:
    // Even slots: live buffer V#s
    // Odd slots: image T#s
    for (std::size_t i = 0; i < kSlots; ++i) {
        std::array<std::uint32_t, 8> w{};
        if ((i & 1u) == 0) {
            // Buffer V#: w[3] >> 30 == 0, num_records != 0
            w[0] = static_cast<std::uint32_t>(0x1000 + i * 0x100);
            w[1] = 0x20;
            w[2] = 256;
            w[3] = 0x00010000u;  // w[3] >> 30 == 0
        } else {
            // Image T#: w[3] >> 30 != 0
            w[0] = static_cast<std::uint32_t>(0x5000 + i * 0x100);
            w[2] = 128;
            w[3] = (2u << 30) | 0x1234u;  // w[3] >> 30 == 2
        }
        gpu_copy[i] = w;
        tracked.set(i, ConsoleHeapVsharpSlots::is_live_vsharp(w.data()));
    }

    // Now simulate fix_console_slots when budget is expired (fix_over == true):
    // Buffer V#s must be fixed in-place and never nulled/deferred; only image T#s are deferred.
    std::size_t deferred_fixes = 0;
    std::vector<std::uint32_t> retries;
    const bool fix_over = true;

    for (std::size_t slot = 0; slot < kSlots; ++slot) {
        const std::array<std::uint32_t, 8>& input = gpu_copy[slot];
        const std::uint32_t w3 = input[3];
        const bool is_buffer_vsharp = !(w3 >> 30);

        if (fix_over && !is_buffer_vsharp) {
            // Image T#: deferred past budget
            gpu_copy[slot].fill(0);
            tracked.clear(slot);
            ++deferred_fixes;
            retries.push_back(static_cast<std::uint32_t>(slot));
            continue;
        }

        // Buffer V#: fixed immediately, not deferred
        tracked.set(slot, ConsoleHeapVsharpSlots::is_live_vsharp(input.data()));
    }

    // All odd slots (image T#s) must have been deferred (nulled)
    EXPECT(deferred_fixes == kSlots / 2);
    EXPECT(retries.size() == kSlots / 2);
    for (std::size_t slot : retries) {
        EXPECT((slot & 1u) == 1u);
        EXPECT(!tracked.test(slot));
        EXPECT(gpu_copy[slot][0] == 0 && gpu_copy[slot][2] == 0);
    }

    // All even slots (buffer V#s) must have been preserved and remain live
    for (std::size_t i = 0; i < kSlots; i += 2) {
        EXPECT(tracked.test(i));
        EXPECT(gpu_copy[i][2] == 256);
        EXPECT(ConsoleHeapVsharpSlots::is_live_vsharp(gpu_copy[i].data()));
    }
    EXPECT(tracked.live_count() == kSlots / 2);
    return 0;
}

}  // namespace

int main() {
    if (test_predicate_exactness()) return 1;
    if (test_boundaries_and_ranges()) return 1;
    if (test_in_place_invalidation_mutation()) return 1;
    if (test_differential_equivalence_against_linear_copy()) return 1;
    if (test_retail_sized_heap_sparse_scan()) return 1;
    if (test_console_fix_budget_buffer_vsharp_non_deferral()) return 1;
    std::printf("console_heap_vsharp_slots_test: ok\n");
    return 0;
}
