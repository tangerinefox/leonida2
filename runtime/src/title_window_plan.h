#pragma once
// Where a title-window mapping sits and how much of a stream pool one mapping serves, without a device
// (docs/d3d12_vulkan.md section 3.3 "Title-chosen addresses").
//
// The device's resolve_title_memory makes the mapping; these decide its range first, so a test can count mappings
// without Vulkan. Two rules matter here:
//   - a window is page-aligned, capped at one request's maximum and charged to LEONIDA_TITLE_WINDOWS_MB; one that
//     does not fit is not made, and the address is backed another way or written as a zero-size view, never raw;
//   - a stream pool (the title's CPU view paired with a GPU-only alias, gfxgeom section 3) is one reservation of
//     16-24 GiB, so it cannot be one allocation: one mapping serves a page-aligned readable run of it, and every
//     request inside that run reuses it instead of asking for a 128 KiB window each.
#include <algorithm>
#include <cstdint>

namespace leonida {

// A page-aligned mapping of the title's memory: [start, start + size).
struct TitleWindowRange {
    std::uint64_t start{};
    std::uint64_t size{};
};

// `size` bytes more fit in the title-window budget [used, cap).
inline bool title_window_fits(std::uint64_t cap, std::uint64_t used, std::uint64_t size) {
    return used <= cap && size <= cap - used;
}

// The page-aligned range a window made for [va, va + bytes) covers: inside the free span [lo, hi), aligned to
// `align`, at most `most` bytes of the request, and charged against [used, cap). False when the span holds no page
// for it or the budget has no room.
inline bool title_window_range(std::uint64_t va, std::uint64_t bytes, std::uint64_t lo, std::uint64_t hi,
                               std::uint64_t align, std::uint64_t most, std::uint64_t cap, std::uint64_t used,
                               TitleWindowRange& out) {
    if (!align || hi <= va) return false;
    const std::uint64_t want = std::clamp<std::uint64_t>(bytes ? bytes : 1, 1, most);
    const std::uint64_t start = (std::max)(va & ~(align - 1), lo);
    std::uint64_t end = va > UINT64_MAX - want - align ? UINT64_MAX : (va + want + align - 1) & ~(align - 1);
    end = (std::min)(end, hi);
    if (end <= start) return false;
    const TitleWindowRange range{start, end - start};
    if (!title_window_fits(cap, used, range.size)) return false;
    out = range;
    return true;
}

// The extent of a stream pool one mapping serves for the request at `va`: from its page to the end of the readable
// run (`readable_end`, exclusive), at most `most` bytes, inside the pool [pool_lo, pool_hi). The same extent comes
// back for every request inside it, so each extent costs one mapping, not one per request.
inline bool title_pool_extent(std::uint64_t pool_lo, std::uint64_t pool_hi, std::uint64_t va,
                              std::uint64_t readable_end, std::uint64_t align, std::uint64_t most,
                              TitleWindowRange& out) {
    if (!align || pool_hi <= pool_lo || va < pool_lo || va >= pool_hi) return false;
    const std::uint64_t start = (std::max)(va & ~(align - 1), pool_lo);
    std::uint64_t end = (std::min)(readable_end, pool_hi);
    if (most) end = (std::min)(end, start > UINT64_MAX - most ? UINT64_MAX : start + most);
    end = end > UINT64_MAX - (align - 1) ? UINT64_MAX : (end + align - 1) & ~(align - 1);
    end = (std::min)(end, pool_hi);
    if (end <= va) return false;
    out = {start, end - start};
    return true;
}

}  // namespace leonida
