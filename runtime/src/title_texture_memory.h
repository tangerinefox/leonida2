#pragma once
#include <windows.h>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace leonida {

// Keep negative probes until the title changes the map or a writer reports this footprint.
class TextureReadability {
public:
    template<class Probe>
    bool check(std::uint64_t generation, bool cache_failure, Probe probe) {
        if (cache_failure && pending_ && !retry_ && generation_ == generation) return false;
        generation_ = generation;  // snapshot before the probe so a concurrent map change is noticed
        retry_ = false;
        pending_ = !probe();
        return !pending_;
    }
    void defer() { pending_ = true; }  // mapping disappeared between the probe and the read
    void retry() { retry_ = true; }
    bool pending() const { return pending_; }
private:
    std::uint64_t generation_ = 0;
    bool pending_ = false, retry_ = false;
};

using TexturePageCommitter = bool (*)(const void*, std::size_t);

// Check the whole footprint. A committed first page says nothing about a streamed mip farther on.
inline bool texture_memory_readable(std::uintptr_t base, std::uint64_t size, TexturePageCommitter commit = nullptr) {
    if (!base || !size || size > (std::numeric_limits<std::uintptr_t>::max)() - base) return false;
    const auto end = base + size;
    for (auto at = base; at < end;) {
        MEMORY_BASIC_INFORMATION region{};
        const auto* address = reinterpret_cast<const void*>(at);
        if (!VirtualQuery(address, &region, sizeof(region))) return false;
        // A persistent Map can name an xmem section whose physical pages commit on first touch.
        // Only xmem may commit them. A bare reservation or an inaccessible page must still defer.
        if (commit && region.State == MEM_RESERVE && region.Type == MEM_MAPPED) {
            if (!commit(address, end - at) || !VirtualQuery(address, &region, sizeof(region))) return false;
        }
        if (region.State != MEM_COMMIT || (region.Protect & PAGE_GUARD)) return false;
        switch (region.Protect & 0xff) {
        case PAGE_READONLY: case PAGE_READWRITE: case PAGE_WRITECOPY:
        case PAGE_EXECUTE_READ: case PAGE_EXECUTE_READWRITE: case PAGE_EXECUTE_WRITECOPY: break;
        default: return false;
        }
        const auto start = reinterpret_cast<std::uintptr_t>(region.BaseAddress);
        if (at < start || at - start >= region.RegionSize) return false;
        const auto available = region.RegionSize - (at - start);
        if (available >= end - at) return true;
        at += available;
    }
    return true;
}

}  // namespace leonida
