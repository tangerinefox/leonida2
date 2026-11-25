#pragma once
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace leonida {

using TitleWriteRanges = std::vector<std::pair<std::uintptr_t, std::uintptr_t>>;

// Sorted, disjoint half-open intervals let each texture find its first possible writer by end address.
inline void merge_title_writes(TitleWriteRanges& writes) {
    std::sort(writes.begin(), writes.end());
    std::size_t count = 0;
    for (const auto [begin, end] : writes) {
        if (begin >= end) continue;
        if (count && begin <= writes[count - 1].second)
            writes[count - 1].second = std::max(writes[count - 1].second, end);
        else
            writes[count++] = {begin, end};
    }
    writes.resize(count);
}

inline bool title_write_overlaps(const TitleWriteRanges& writes, std::uintptr_t begin, std::uintptr_t end) {
    const auto it = std::upper_bound(writes.begin(), writes.end(), begin,
                                     [](auto address, const auto& range) { return address < range.second; });
    return begin < end && it != writes.end() && it->first < end;
}

// Poll unmarked textures in placement order. A time or byte limit resumes the next pass at
// the following address. The cursor also survives insertion or removal between passes.
template<class Registry, class Visit, class Exhausted>
void poll_title_textures(Registry& textures, std::uint64_t& cursor, Visit visit, Exhausted exhausted) {
    auto it = textures.lower_bound(cursor);
    for (auto left = textures.size(); left; --left) {
        if (it == textures.end()) it = textures.begin();
        auto& [address, texture] = *it++;
        cursor = address + 1;
        if (!texture->mapped && !texture->stale) visit(*texture);
        if (exhausted()) break;  // advance at least one registry entry even if a deadline already passed
    }
}

}  // namespace leonida
