#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace leonida {
// A separate cursor per console heap. The owning backing/alias registry locks record() and read().
struct DescriptorRanges {
    struct Range { uint64_t first, last; }; // inclusive, including UINT64_MAX
    std::vector<Range> ranges;
    void merge() {
        std::sort(ranges.begin(), ranges.end(), [](auto a, auto b) { return a.first < b.first; });
        size_t n = 0;
        for (auto r : ranges) {
            if (n && r.first <= ranges[n-1].last)
                ranges[n-1].last = (std::max)(ranges[n-1].last, r.last);
            else ranges[n++] = r;
        }
        ranges.resize(n);
    }
    bool overlaps(uint64_t base, uint64_t bytes) const {
        if (!bytes) return false;
        auto it = std::lower_bound(ranges.begin(), ranges.end(), base,
                                   [](auto r, uint64_t v) { return r.last < v; });
        return it != ranges.end() && (it->first <= base || it->first - base < bytes);
    }
};
class DescriptorRangeChanges {
    std::array<DescriptorRanges::Range, 4096> ring_{};
    uint64_t serial_{};
public:
    void record(uint64_t start, uint64_t bytes) {
        if (bytes) {
            ring_[serial_++ % ring_.size()] = {start, start + (std::min)(UINT64_MAX-start, bytes-1)};
        }
    }
    bool read(uint64_t& cursor, DescriptorRanges& out) const {
        const auto previous = cursor;
        cursor = serial_;
        if (serial_ - previous > ring_.size()) return false;
        for (auto i = previous; i != serial_; ++i) out.ranges.push_back(ring_[i % ring_.size()]);
        return true;
    }
};
}
