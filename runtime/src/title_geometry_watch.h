#pragma once

#include "title_page_watch.h"
#include <array>
#include <vector>

namespace leonida::geometry_writes {

constexpr std::uintptr_t tile_size = 65536, page_size = 4096;
using Aliases = void (*)(std::uintptr_t, std::vector<std::uintptr_t>&);
inline std::atomic<Aliases> aliases{};

struct Cell {
    std::atomic<std::uint64_t> version{1}, mapping{1};
    std::array<DWORD, tile_size / page_size> restore{};
    unsigned refs = 0;
};
struct State {
    std::mutex mutex;
    host_heap::map<std::uintptr_t, Cell> cells;
    std::atomic<std::uint64_t> serial{1};
    std::atomic<std::size_t> count{0};
};
inline State& state() { static State s; return s; }

inline Cell* acquire(std::uintptr_t va) {
    auto& s = state();
    std::lock_guard lock(s.mutex);
    auto& c = s.cells[va];
    ++c.refs;
    s.count.store(s.cells.size(), std::memory_order_release);
    return &c;
}

// Restore only protections this watch installed. A guest remap or explicit new protection supersedes them.
inline DWORD read_only(DWORD protection) {
    switch (protection & 255u) {
    case PAGE_READWRITE: case PAGE_WRITECOPY: return (protection & ~255u) | PAGE_READONLY;
    case PAGE_EXECUTE_READWRITE: case PAGE_EXECUTE_WRITECOPY: return (protection & ~255u) | PAGE_EXECUTE_READ;
    default: return 0;
    }
}
// True when it made a page writable again.
inline bool reopen(std::uintptr_t va, Cell& c) {
    bool opened = false;
    for (std::size_t i = 0; i < c.restore.size(); ++i) {
        if (!c.restore[i]) continue;
        void* p = reinterpret_cast<void*>(va + i * page_size);
        MEMORY_BASIC_INFORMATION m{};
        DWORD old{};
        if (!VirtualQuery(p, &m, sizeof m)) continue;
        if (m.State == MEM_COMMIT && m.Protect == read_only(c.restore[i])) {
            if (!VirtualProtect(p, page_size, c.restore[i], &old)) continue;
            opened = true;
        }
        c.restore[i] = 0;
    }
    return opened;
}
inline void release(std::uintptr_t va) {
    auto& s = state();
    std::lock_guard lock(s.mutex);
    auto it = s.cells.find(va);
    if (it != s.cells.end() && !--it->second.refs) {
        // A page reopened here may also be one a slot-133 shadow keeps read-only (both guard the title's geometry):
        // it is writable now with no fault to tell that shadow, which then walks its pages again.
        if (reopen(va, it->second)) title_memory_map::changed(va, va + tile_size);
        s.cells.erase(it);
        s.count.store(s.cells.size(), std::memory_order_release);
    }
}
enum class Event { write, open, fault, mapping, replaced };
// `restored` (open and fault): set when a tile's pages were made writable again. A tile reopens all of its pages, and
// a page of it a slot-133 shadow also keeps read-only (both can guard one page: each queries, then protects, under its
// own lock) is writable then without that shadow knowing; the caller has the shadows reopen the whole tiles too.
inline bool notify(std::uintptr_t begin, std::uintptr_t end, Event event, bool* restored = nullptr) {
    auto& s = state();
    if (!s.count.load(std::memory_order_acquire) || end <= begin) return false;
    std::lock_guard lock(s.mutex);
    auto it = s.cells.lower_bound(begin & ~(tile_size - 1));
    if (it == s.cells.end() || it->first >= end) return false;
    const auto version = s.serial.load(std::memory_order_relaxed) + 1;
    bool opened = false;
    for (; it != s.cells.end() && it->first < end; ++it) {
        auto& cell = it->second;
        if (event == Event::open || event == Event::fault) {
            if (reopen(it->first, cell) && restored) *restored = true;
            if (event == Event::fault) {
                MEMORY_BASIC_INFORMATION m{};
                opened = VirtualQuery(reinterpret_cast<void*>(begin), &m, sizeof m) &&
                         m.State == MEM_COMMIT && !(m.Protect & PAGE_GUARD) && read_only(m.Protect);
            }
        }
        if (event == Event::replaced) {
            for (unsigned i = 0; i < cell.restore.size(); ++i) {
                auto p = it->first + i * page_size;
                if (p < end && p + page_size > begin) cell.restore[i] = 0;
            }
        }
        if (event == Event::mapping || event == Event::replaced) cell.mapping.store(version, std::memory_order_release);
        cell.version.store(version, std::memory_order_release);
    }
    s.serial.store(version, std::memory_order_release);
    return opened;
}

// Arm under the same lock as the write-fault path. A kernel destination may only be guarded again after its pin ends.
inline bool arm(std::uintptr_t va, Cell& c) {
    auto& s = state();
    // Querying an OVERLAPPED can fault into xmem. Never do it with the watch lock held.
    const auto version = c.version.load(std::memory_order_acquire);
    bool pinned = false;
    kernel_writes::for_each_pin(va, va + tile_size, [&](auto, auto) { pinned = true; });
    std::lock_guard lock(s.mutex);
    if (pinned || version != c.version.load(std::memory_order_relaxed)) return false;
    for (std::size_t i = 0; i < c.restore.size();) {
        auto p = va + i * page_size;
        MEMORY_BASIC_INFORMATION m{};
        if (!VirtualQuery(reinterpret_cast<void*>(p), &m, sizeof m)) return false;
        const auto end = (std::min)(va + tile_size, reinterpret_cast<std::uintptr_t>(m.BaseAddress) + m.RegionSize);
        if (end <= p) return false;
        const auto next = (end - va) / page_size;
        const DWORD guarded = read_only(m.Protect);
        if (m.State == MEM_COMMIT && !(m.Protect & PAGE_GUARD) && guarded) {
            DWORD old{};
            if (!VirtualProtect(reinterpret_cast<void*>(p), end - p, guarded, &old)) return false;
            // What the query saw, not `old`: VirtualProtect returns the first page's protection only, and a slot-133
            // shadow (which guards these pages under its own lock) may have made that page read-only in between. A
            // read-only `old` then marked every page of the run as read-only already, so reopen() never gave them back
            // and they stayed read-only with no guard to answer for them: title writes and kernel writes into them
            // failed for good. Such a page has two guards now; reopening it tells the shadows (notify's `restored`).
            for (auto j = i; j < next; ++j) c.restore[j] = m.Protect;
        }
        i = next;
    }
    return true;
}
} // namespace leonida::geometry_writes
