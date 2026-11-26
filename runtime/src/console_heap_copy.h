#pragma once
// A console resource heap's translated copy (DeviceState::ConsoleHeap::copy): the bytes shaders read, and a host-RAM
// mirror of them that the CPU reads instead.
//
// The copy lies in the GPU address arena when the import of host memory is not used (arena.mapped + offset), and the
// arena prefers DEVICE_LOCAL | HOST_VISIBLE memory (ReBAR): a CPU read of it is an uncached read across PCIe. The
// release sweep of refresh_console_heap with LEONIDA_CONSOLE_RELEASE_FILTER=1 reads words 0-2 of both V# halves of
// every live V# slot (about 470k in a game run) to decide whether the slot needs fixing, and that read cost
// 0.9-1.0 s a sweep. Every CPU write goes through write() and zero(), which write both the copy and the mirror, so
// view() (the mirror, or the copy when there is none) holds what the GPU reads.
//
// No mirror when the copy is already host RAM (the import of the heap's own scratch), when LEONIDA_HEAP_MIRROR=0, or
// once the copy is written by the GPU pass (LEONIDA_CONSOLE_HEAP_GPU: drop_mirror(); the CPU neither writes nor reads
// the copy then). The mirror covers the heap's bytes only, not the null guard after them (written once, at creation).
//
// LEONIDA_HEAP_MIRROR_CHECK=1: refresh_console_heap compares the mirror with the copy at every release sweep
// (mismatches()) and logs the first slots that differ.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace leonida {

class ConsoleHeapCopy {
public:
    // `gpu`: the copy shaders read; `bytes`: the heap's bytes (the mirror's size, when `mirrored`).
    void attach(std::uint8_t* gpu, std::size_t bytes, bool mirrored) {
        gpu_ = gpu;
        bytes_ = bytes;
        mirror_.clear();
        mirror_.shrink_to_fit();
        if (gpu && mirrored) mirror_.assign(bytes, 0);
    }
    // The mirror goes; view() is the copy from now on.
    void drop_mirror() {
        mirror_.clear();
        mirror_.shrink_to_fit();
    }

    explicit operator bool() const noexcept { return gpu_ != nullptr; }
    std::uint8_t* gpu() const noexcept { return gpu_; }
    bool mirrored() const noexcept { return !mirror_.empty(); }
    std::size_t bytes() const noexcept { return bytes_; }

    // What the CPU reads: the mirror, or the copy itself.
    const std::uint8_t* view() const noexcept { return mirror_.empty() ? gpu_ : mirror_.data(); }

    // [at, at + n) of the copy, and of the mirror where it reaches.
    void write(std::size_t at, const void* bytes, std::size_t n) {
        std::memcpy(gpu_ + at, bytes, n);
        if (const std::size_t m = mirror_span(at, n)) std::memcpy(mirror_.data() + at, bytes, m);
    }
    void zero(std::size_t at, std::size_t n) {
        std::memset(gpu_ + at, 0, n);
        if (const std::size_t m = mirror_span(at, n)) std::memset(mirror_.data() + at, 0, m);
    }

    // LEONIDA_HEAP_MIRROR_CHECK: the slots of `slot_bytes` whose mirror bytes differ from the copy's, at most `limit`
    // of them in `out` (in ascending order); returns how many slots differ in all. 0 without a mirror.
    std::size_t mismatches(std::size_t slot_bytes, std::size_t limit, std::vector<std::size_t>& out) const {
        out.clear();
        if (mirror_.empty() || !slot_bytes) return 0;
        // Read the copy in large blocks: one pass over device memory, compared in host RAM.
        constexpr std::size_t kBlock = std::size_t(1) << 16;
        std::vector<std::uint8_t> block;
        std::size_t count = 0;
        const std::size_t step = std::max(slot_bytes, kBlock / slot_bytes * slot_bytes);
        for (std::size_t at = 0; at < bytes_; at += step) {
            const std::size_t n = std::min(step, bytes_ - at);
            block.resize(n);
            std::memcpy(block.data(), gpu_ + at, n);
            if (!std::memcmp(block.data(), mirror_.data() + at, n)) continue;
            for (std::size_t s = 0; s < n; s += slot_bytes) {
                const std::size_t k = std::min(slot_bytes, n - s);
                if (!std::memcmp(block.data() + s, mirror_.data() + at + s, k)) continue;
                if (out.size() < limit) out.push_back((at + s) / slot_bytes);
                ++count;
            }
        }
        return count;
    }

private:
    std::size_t mirror_span(std::size_t at, std::size_t n) const noexcept {
        if (mirror_.empty() || at >= mirror_.size()) return 0;
        return std::min(n, mirror_.size() - at);
    }

    std::uint8_t* gpu_ = nullptr;
    std::size_t bytes_ = 0;
    std::vector<std::uint8_t> mirror_;
};

}  // namespace leonida
