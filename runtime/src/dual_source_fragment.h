#pragma once

// Dual-source blending (RT-DUAL-SOURCE-BLEND). D3D12 takes the second blend source from SV_Target1, which the
// translated pixel shader declares as the output at Location 1. Vulkan takes it from Location 0, Index 1
// (VK_BLEND_FACTOR_SRC1_*, dualSrcBlend). For a pipeline whose blend state uses a SRC1 factor, the fragment module's
// Location 1 output is moved to Location 0 and given Index 1. Standard library only.

#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace leonida::dual_source {

// Rewrites `words` in place and returns true, or returns false and leaves them unchanged when the module is not one
// this rewrite understands: not SPIR-V, no Output variable at Location 1, an output at Location 2 or above (Vulkan has
// one dual-source attachment), or an Index decoration already present.
inline bool remap(std::vector<std::uint32_t>& words) {
    constexpr std::uint32_t kOpDecorate = 71, kOpVariable = 59, kOpTypePointer = 32;
    constexpr std::uint32_t kLocation = 30, kIndex = 32, kBuiltIn = 11, kStorageOutput = 3;
    if (words.size() < 5 || words[0] != 0x07230203) return false;
    std::map<std::uint32_t, std::size_t> location_at;  // variable id -> word index of its Location literal
    std::map<std::uint32_t, std::uint32_t> location_of;
    std::set<std::uint32_t> output_pointers, outputs, builtins;
    for (std::size_t at = 5; at < words.size();) {
        const std::uint32_t n = words[at] >> 16, op = words[at] & 0xffffu;
        if (!n || n > words.size() - at) return false;
        if (op == kOpDecorate && n == 4 && words[at + 2] == kLocation) {
            location_at[words[at + 1]] = at + 3;
            location_of[words[at + 1]] = words[at + 3];
        } else if (op == kOpDecorate && n >= 3 && words[at + 2] == kIndex) {
            return false;
        } else if (op == kOpDecorate && n >= 3 && words[at + 2] == kBuiltIn) {
            builtins.insert(words[at + 1]);
        } else if (op == kOpTypePointer && n == 4 && words[at + 2] == kStorageOutput) {
            output_pointers.insert(words[at + 1]);
        } else if (op == kOpVariable && n >= 4 && words[at + 3] == kStorageOutput) {
            outputs.insert(words[at + 2]);
        }
        at += n;
    }
    std::uint32_t second = 0;
    for (const std::uint32_t id : outputs) {
        if (builtins.count(id)) continue;
        const auto loc = location_of.find(id);
        if (loc == location_of.end()) continue;
        if (loc->second >= 2) return false;
        if (loc->second == 1) second = id;
    }
    if (!second) return false;
    const std::size_t literal = location_at[second];
    words[literal] = 0;
    // OpDecorate %second Index 1, right after its Location decoration (annotations stay together).
    const std::vector<std::uint32_t> index{(4u << 16) | kOpDecorate, second, kIndex, 1};
    words.insert(words.begin() + static_cast<std::ptrdiff_t>(literal + 1), index.begin(), index.end());
    return true;
}

}  // namespace leonida::dual_source
