#pragma once
// Host-RAM bitset of console resource-heap slots whose GPU copy (`heap.copy`) currently holds a live buffer V#
// (`!(w[3] >> 30) && (w[2] || w[6])`).
//
// When a buffer backing or title VA alias is released, `refresh_console_heap` must re-run `fix_console_slot` on every
// slot that might still point at the released allocation. Scanning `heap.copy` linearly reads every 32-byte slot of
// the heap (4-5.5+ MiB): when `heap.copy` sits in `arena.mapped` (uncached write-combined GPU memory), each release
// triggers ~180k uncached PCIe reads. Tracking the live V# slots in host RAM keeps invalidation scans strictly in CPU
// cache (zero reads from `heap.copy`) and skips empty and T# slots 64 at a time.

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace leonida {

class ConsoleHeapVsharpSlots {
public:
    // Matches the descriptor predicate for a non-empty buffer V# (primary or counter half) in a 32-byte slot.
    static constexpr bool is_live_vsharp(const std::uint32_t* w) noexcept {
        return !(w[3] >> 30) && (w[2] || w[6]);
    }

    void reset(std::size_t slot_count) {
        slot_count_ = slot_count;
        live_count_ = 0;
        words_.assign((slot_count + 63) / 64, 0);
    }

    void set(std::size_t slot, bool live) noexcept {
        if (slot >= slot_count_) return;
        std::uint64_t& word = words_[slot >> 6];
        const std::uint64_t mask = std::uint64_t{1} << (slot & 63);
        const bool was = (word & mask) != 0;
        if (live == was) return;
        if (live) {
            word |= mask;
            ++live_count_;
        } else {
            word &= ~mask;
            --live_count_;
        }
    }

    void clear(std::size_t slot) noexcept { set(slot, false); }

    // Clears every slot in [first, end), clamping to slot_count().
    void clear_range(std::size_t first, std::size_t end) noexcept {
        if (first >= end || first >= slot_count_ || live_count_ == 0) return;
        end = (std::min)(end, slot_count_);
        std::size_t w = first >> 6;
        const std::size_t last_w = (end - 1) >> 6;
        const unsigned first_bit = static_cast<unsigned>(first & 63);
        const unsigned end_bit = static_cast<unsigned>(end & 63);
        const std::uint64_t head_mask = ~std::uint64_t{0} << first_bit;
        const std::uint64_t tail_mask = end_bit ? ((std::uint64_t{1} << end_bit) - 1) : ~std::uint64_t{0};
        if (w == last_w) {
            const std::uint64_t mask = head_mask & tail_mask;
            const std::uint64_t cleared = words_[w] & mask;
            if (cleared) {
                words_[w] &= ~mask;
                live_count_ -= static_cast<std::size_t>(std::popcount(cleared));
            }
            return;
        }
        if (const std::uint64_t cleared = words_[w] & head_mask) {
            words_[w] &= ~head_mask;
            live_count_ -= static_cast<std::size_t>(std::popcount(cleared));
        }
        for (++w; w < last_w; ++w) {
            if (const std::uint64_t cleared = words_[w]) {
                words_[w] = 0;
                live_count_ -= static_cast<std::size_t>(std::popcount(cleared));
            }
        }
        if (const std::uint64_t cleared = words_[last_w] & tail_mask) {
            words_[last_w] &= ~tail_mask;
            live_count_ -= static_cast<std::size_t>(std::popcount(cleared));
        }
    }

    // Calls fn(slot) in ascending slot order for every set bit present when its 64-slot word is entered.
    // Safe if fn mutates the bit for `slot` (e.g., fix_console_slot clearing an invalidated V#).
    template <class F>
    void for_each(F&& fn) const {
        if (live_count_ == 0) return;
        for (std::size_t w = 0; w < words_.size(); ++w) {
            std::uint64_t bits = words_[w];
            if (!bits) continue;
            const std::size_t base = w << 6;
            while (bits) {
                const unsigned bit = static_cast<unsigned>(std::countr_zero(bits));
                bits &= bits - 1;
                fn(base + bit);
            }
        }
    }

    bool test(std::size_t slot) const noexcept {
        return slot < slot_count_ && ((words_[slot >> 6] >> (slot & 63)) & 1u) != 0;
    }

    std::size_t slot_count() const noexcept { return slot_count_; }
    std::size_t live_count() const noexcept { return live_count_; }
    bool empty() const noexcept { return live_count_ == 0; }

private:
    std::size_t slot_count_ = 0;
    std::size_t live_count_ = 0;
    std::vector<std::uint64_t> words_;
};

}  // namespace leonida
