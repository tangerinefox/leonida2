// Portable CPU-only unit test for leonida::ConsoleHeapCopy (runtime/src/console_heap_copy.h): the host-RAM mirror of a
// console heap's translated copy. A plain buffer stands in for the copy in the GPU address arena. Checks that every
// write and zero reaches both, that the null guard past the heap stays out of the mirror, that mismatches() finds a
// write that bypassed the helpers (the LEONIDA_HEAP_MIRROR_CHECK case), and that without a mirror view() is the copy.
// Then times a filtered release sweep over a large heap reading the mirror (the work refresh_console_heap does per
// rescan with LEONIDA_CONSOLE_RELEASE_FILTER=1), for reference; no game data.
#include "../src/console_heap_copy.h"
#include "../src/console_heap_vsharp_slots.h"
#include "../src/descriptor_range_changes.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define EXPECT(condition)                                                     \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); \
            return 1;                                                         \
        }                                                                     \
    } while (0)

namespace {

using leonida::ConsoleHeapCopy;
constexpr std::size_t kSlot = 32;

int test_writes_reach_both() {
    constexpr std::size_t slots = 256, bytes = slots * kSlot, guard = 4096;
    std::vector<std::uint8_t> gpu(bytes + guard, 0xCD);
    ConsoleHeapCopy copy;
    EXPECT(!copy);
    copy.attach(gpu.data(), bytes, true);
    EXPECT(copy && copy.mirrored() && copy.gpu() == gpu.data() && copy.bytes() == bytes);
    EXPECT(copy.view() != gpu.data());
    // Creation: the copy and its guard zeroed; the mirror covers the heap only.
    copy.zero(0, bytes + guard);
    for (const auto b : gpu) EXPECT(b == 0);
    std::vector<std::size_t> differ;
    EXPECT(copy.mismatches(kSlot, 8, differ) == 0 && differ.empty());

    std::uint32_t w[8] = {0x1000, 0x20, 64, 0x3fffu, 0, 0, 0, 0};
    for (std::size_t s = 0; s < slots; s += 3) {
        w[0] = std::uint32_t(0x1000 + s * 0x40);
        copy.write(s * kSlot, w, sizeof(w));
    }
    copy.zero(9 * kSlot, kSlot);
    copy.zero(64 * kSlot, 32 * kSlot);  // a zeroed range of slots
    // A sampler heap's whole range copied at once.
    std::vector<std::uint8_t> run(5 * kSlot + 7, 0x5A);
    copy.write(200 * kSlot + 3, run.data(), run.size());
    EXPECT(std::memcmp(copy.view(), gpu.data(), bytes) == 0);
    EXPECT(copy.mismatches(kSlot, 8, differ) == 0);
    // Past the heap: the copy's guard only, nothing out of the mirror's bounds.
    copy.zero(bytes - kSlot, kSlot + guard);
    std::uint32_t tail[8];
    std::memcpy(tail, copy.view() + bytes - kSlot, sizeof(tail));
    for (const auto t : tail) EXPECT(t == 0);
    EXPECT(copy.mismatches(kSlot, 8, differ) == 0);
    std::puts("writes and zeros reach the copy and the mirror; the guard stays out of the mirror");
    return 0;
}

int test_mismatch_found() {
    constexpr std::size_t slots = 8192, bytes = slots * kSlot;  // 256 KiB: more than one compare block
    std::vector<std::uint8_t> gpu(bytes, 0);
    ConsoleHeapCopy copy;
    copy.attach(gpu.data(), bytes, true);
    // Writes that bypass the helpers (a missed write site, or the GPU writing the copy).
    gpu[5 * kSlot + 4] = 1;
    gpu[4000 * kSlot] = 2;
    gpu[4001 * kSlot + 31] = 3;
    gpu[8191 * kSlot + 16] = 4;
    std::vector<std::size_t> differ;
    EXPECT(copy.mismatches(kSlot, 3, differ) == 4);
    EXPECT(differ.size() == 3 && differ[0] == 5 && differ[1] == 4000 && differ[2] == 4001);
    EXPECT(copy.mismatches(kSlot, 8, differ) == 4 && differ.size() == 4 && differ[3] == 8191);
    std::puts("mismatches() finds writes that bypassed the mirror, in slot order");
    return 0;
}

int test_without_mirror() {
    constexpr std::size_t bytes = 64 * kSlot;
    std::vector<std::uint8_t> gpu(bytes, 0);
    ConsoleHeapCopy copy;
    copy.attach(gpu.data(), bytes, false);  // the copy is host RAM already
    EXPECT(!copy.mirrored() && copy.view() == gpu.data());
    const std::uint32_t w[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    copy.write(3 * kSlot, w, sizeof(w));
    EXPECT(!std::memcmp(gpu.data() + 3 * kSlot, w, sizeof(w)));
    std::vector<std::size_t> differ;
    EXPECT(copy.mismatches(kSlot, 8, differ) == 0);

    copy.attach(gpu.data(), bytes, true);
    EXPECT(copy.mirrored());
    copy.drop_mirror();  // the GPU pass writes the copy from now on
    EXPECT(!copy.mirrored() && copy.view() == gpu.data());
    std::puts("without a mirror view() is the copy");
    return 0;
}

// The filtered release sweep of refresh_console_heap over a heap of 576k slots (18 MiB) with 470k live V# slots, one
// released range touching about 2%: the time per sweep with its words read from host RAM.
int bench_filtered_sweep() {
    constexpr std::size_t slots = 576 * 1024, bytes = slots * kSlot;
    std::vector<std::uint8_t> previous(bytes, 0), gpu(bytes, 0);
    ConsoleHeapCopy copy;
    copy.attach(gpu.data(), bytes, true);
    leonida::ConsoleHeapVsharpSlots live;
    live.reset(slots);
    constexpr std::uint64_t kBuffers = 4096, kBufferBytes = 1 << 20, kArena = 1ull << 32;
    for (std::size_t s = 0; s < slots; ++s) {
        if (s % 11 == 10) continue;  // T#s and empty slots
        const std::uint64_t base = 0x100000 + (s * 2654435761u % kBuffers) * kBufferBytes + (s % 64) * 256;
        std::uint32_t w[8] = {std::uint32_t(base), std::uint32_t(base >> 32) & 0xffffu, 256, 0x00027fac, 0, 0, 0, 0};
        std::memcpy(previous.data() + s * kSlot, w, sizeof(w));
        copy.write(s * kSlot, w, sizeof(w));
        live.set(s, leonida::ConsoleHeapVsharpSlots::is_live_vsharp(w));
    }
    leonida::DescriptorRanges backing, aliases;
    for (std::uint64_t b = 0; b < kBuffers; b += 50)  // about 2% of the buffers released
        backing.ranges.push_back({kArena + 0x100000 + b * kBufferBytes, kArena + 0x100000 + (b + 1) * kBufferBytes - 1});
    backing.merge();
    const auto bda = [&](std::uint64_t base) { return kArena + base; };
    const auto affected = [&](const std::uint32_t* src, const std::uint32_t* dst) {
        if ((src[3] >> 30) || !src[2]) return false;
        const std::uint64_t base = std::uint64_t(src[0]) | (std::uint64_t(src[1] & 0xffff) << 32);
        const std::uint64_t fixed = std::uint64_t(dst[0]) | (std::uint64_t(dst[1] & 0xffff) << 32);
        const auto stride = (src[1] >> 16) & 0x3fff;
        const std::uint64_t n = std::uint64_t(src[2]) * (stride ? stride : 1);
        return aliases.overlaps(base, 1) || backing.overlaps(bda(base), n) || (dst[2] && backing.overlaps(bda(fixed), n));
    };
    constexpr int kRuns = 9;
    double best = 1e30, first = 0;
    std::size_t hits = 0;
    for (int run = 0; run < kRuns; ++run) {
        hits = 0;
        const auto start = std::chrono::steady_clock::now();
        const std::uint8_t* const translated = copy.view();
        live.for_each([&](std::size_t slot) {
            const auto at = slot * kSlot;
            const auto* src = reinterpret_cast<const std::uint32_t*>(previous.data() + at);
            const auto* dst = reinterpret_cast<const std::uint32_t*>(translated + at);
            if (affected(src, dst) || affected(src + 4, dst + 4)) ++hits;
        });
        const double took = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (!run) first = took;
        best = std::min(best, took);
    }
    EXPECT(hits > 0 && hits < live.live_count() / 10);
    std::printf("filtered sweep over %zu live slots of %zu MiB, %zu affected: %.2f ms (best of %d, first %.2f), words "
                "from the host mirror\n",
                live.live_count(), bytes >> 20, hits, best, kRuns, first);
    return 0;
}

}  // namespace

int main() {
    if (test_writes_reach_both()) return 1;
    if (test_mismatch_found()) return 1;
    if (test_without_mirror()) return 1;
    if (bench_filtered_sweep()) return 1;
    std::puts("ok");
    return 0;
}
