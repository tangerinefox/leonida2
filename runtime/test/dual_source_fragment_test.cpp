// Dual-source blending (RT-DUAL-SOURCE-BLEND): the fragment module rewrite that moves SV_Target1 from Location 1 to
// Location 0, Index 1. Hand-built SPIR-V word streams only (the declarations the rewrite reads); no GPU, no game data.
#include "../src/dual_source_fragment.h"

#include <cstdio>
#include <initializer_list>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

using Words = std::vector<std::uint32_t>;

void op(Words& w, std::uint32_t code, std::initializer_list<std::uint32_t> args) {
    w.push_back(std::uint32_t(args.size() + 1) << 16 | code);
    w.insert(w.end(), args);
}

constexpr std::uint32_t kDecorate = 71, kTypeFloat = 22, kTypeVector = 23, kTypePointer = 32, kVariable = 59;
constexpr std::uint32_t kLocation = 30, kIndex = 32, kBuiltIn = 11, kFragDepth = 22, kInput = 1, kOutput = 3;

// Output vec4 %10 at Location 0 and %11 at `second` (none if 0xff), FragDepth %12, an input %13 at Location 1, and
// extra decorations from `extra`.
Words module(std::uint32_t second, std::initializer_list<std::initializer_list<std::uint32_t>> extra = {}) {
    Words w{0x07230203, 0x00010000, 0, 20, 0};
    op(w, kDecorate, {10, kLocation, 0});
    if (second != 0xff) op(w, kDecorate, {11, kLocation, second});
    op(w, kDecorate, {12, kBuiltIn, kFragDepth});
    op(w, kDecorate, {13, kLocation, 1});
    for (const auto& e : extra) op(w, kDecorate, e);
    op(w, kTypeFloat, {2, 32});
    op(w, kTypeVector, {3, 2, 4});
    op(w, kTypePointer, {4, kOutput, 3});
    op(w, kTypePointer, {5, kOutput, 2});
    op(w, kTypePointer, {6, kInput, 3});
    op(w, kVariable, {4, 10, kOutput});
    if (second != 0xff) op(w, kVariable, {4, 11, kOutput});
    op(w, kVariable, {5, 12, kOutput});
    op(w, kVariable, {6, 13, kInput});
    return w;
}

// The literal of `id`'s first decoration of `kind`, or -1.
long decoration(const Words& w, std::uint32_t id, std::uint32_t kind) {
    for (std::size_t at = 5; at < w.size(); at += w[at] >> 16)
        if ((w[at] & 0xffff) == kDecorate && w[at + 1] == id && w[at + 2] == kind) return long(w[at + 3]);
    return -1;
}

}  // namespace

int main() {
    Words w = module(1);
    const std::size_t before = w.size();
    check(leonida::dual_source::remap(w), "SV_Target0 at Location 0 and SV_Target1 at Location 1: rewritten");
    check(decoration(w, 11, kLocation) == 0 && decoration(w, 11, kIndex) == 1,
          "SV_Target1 is at Location 0, Index 1");
    check(decoration(w, 10, kLocation) == 0 && decoration(w, 10, kIndex) == -1, "SV_Target0 keeps Location 0, no Index");
    check(decoration(w, 13, kLocation) == 1, "an input at Location 1 is not touched");
    check(w.size() == before + 4, "one OpDecorate added, nothing else");

    for (const auto& [m, what] : {std::pair{module(0xff), "no second colour output"},
                                  std::pair{module(2), "an output at Location 2"},
                                  std::pair{module(1, {{11, kIndex, 0}}), "an Index decoration already present"}}) {
        Words copy = m;
        check(!leonida::dual_source::remap(copy) && copy == m, what);
    }
    Words not_spirv{1, 2, 3, 4, 5, 6};
    check(!leonida::dual_source::remap(not_spirv), "not SPIR-V");
    Words truncated = module(1);
    truncated[5] = (40u << 16) | kDecorate;
    check(!leonida::dual_source::remap(truncated), "an instruction past the end");

    std::printf("%s\n", g_failures ? "FAILED" : "all passed");
    return g_failures ? 1 : 0;
}
