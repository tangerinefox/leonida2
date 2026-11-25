#pragma once

// LOCAL DIAGNOSTIC (not for upstream): fsdraw probe. Follows the title's fullscreen draws back from the backbuffer:
// what each pipeline is (shaders, target formats, blend, write mask), what the draw's descriptor tables hold on the
// CPU and what the GPU address arena holds at the same VA (the shader's heap_base + VA), which T#s have a bindless
// entry, and which render targets the present-probe should read back as well.

#include "title_va_alias.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace leonida::fsdraw {

struct State {
    std::mutex mutex;
    std::unordered_map<std::uintptr_t, std::string> pipelines;  // graphics PSO id -> description
    std::set<std::array<std::uint32_t, 8>> tsharps;             // T#s that have a bindless entry
    const std::uint8_t* arena = nullptr;                        // the GPU address arena's host mapping
    std::uint64_t arena_size = 0;
    std::vector<std::uint64_t> readback;  // render targets of fullscreen draws (besides the presented source)
    std::uint64_t presented_source = 0;
    std::unordered_map<std::uint64_t, std::uint64_t> table_origin;  // translated table VA -> title VA
    std::unordered_map<std::uint64_t, std::pair<std::uint64_t, std::uint32_t>> cbv_origin;  // V# base -> location, size
};

inline State& state() {
    static State s;
    return s;
}

inline void note_pipeline(std::uintptr_t id, std::string text) {
    std::lock_guard lock(state().mutex);
    state().pipelines[id] = std::move(text);
}

inline std::size_t pipeline_count() {  // LOCAL DIAGNOSTIC (mem probe)
    std::lock_guard lock(state().mutex);
    return state().pipelines.size();
}

inline std::string pipeline(std::uintptr_t id) {
    std::lock_guard lock(state().mutex);
    const auto it = state().pipelines.find(id);
    return it == state().pipelines.end() ? std::string("(not registered)") : it->second;
}

inline void note_tsharp(const std::uint32_t* words) {
    std::array<std::uint32_t, 8> key{};
    std::memcpy(key.data(), words, sizeof(key));
    std::lock_guard lock(state().mutex);
    state().tsharps.insert(key);
}

inline bool has_tsharp(const std::uint32_t* words) {
    std::array<std::uint32_t, 8> key{};
    std::memcpy(key.data(), words, sizeof(key));
    std::lock_guard lock(state().mutex);
    return state().tsharps.count(key) != 0;
}

inline void note_arena(const std::uint8_t* mapped, std::uint64_t size) {
    std::lock_guard lock(state().mutex);
    state().arena = mapped;
    state().arena_size = size;
}

// A test or application may create multiple devices. Never inspect a released
// mapping; destroying an older device must leave the current mapping alone.
inline void forget_arena(const std::uint8_t* mapped) {
    std::lock_guard lock(state().mutex);
    if (state().arena == mapped) {
        state().arena = nullptr;
        state().arena_size = 0;
    }
}

// A render target a fullscreen draw wrote: the present-probe reads it back with the source (at most 4 kept).
inline void note_target(std::uint64_t id) {
    if (!id) return;
    std::lock_guard lock(state().mutex);
    auto& r = state().readback;
    for (const std::uint64_t x : r)
        if (x == id) return;
    if (r.size() < 4) r.push_back(id);
}

inline std::vector<std::uint64_t> targets() {
    std::lock_guard lock(state().mutex);
    return state().readback;
}

inline void note_table(std::uint64_t translated, std::uint64_t original) {
    std::lock_guard lock(state().mutex);
    auto& m = state().table_origin;
    if (m.size() > 8192) m.clear();
    m[translated] = original;
}

inline std::uint64_t table_origin(std::uint64_t va) {
    std::lock_guard lock(state().mutex);
    const auto it = state().table_origin.find(va);
    return it == state().table_origin.end() ? va : it->second;
}

inline void note_cbv(std::uint64_t base, std::uint64_t location, std::uint32_t size) {
    std::lock_guard lock(state().mutex);
    auto& m = state().cbv_origin;
    if (m.size() > 8192) m.clear();
    m[base] = {location, size};
}

inline bool cbv_origin(std::uint64_t base, std::uint64_t& location) {
    std::lock_guard lock(state().mutex);
    const auto it = state().cbv_origin.find(base);
    if (it == state().cbv_origin.end()) return false;
    location = it->second.first;
    return true;
}

inline std::string hex_words(const std::uint32_t* w, std::size_t n) {
    std::string s;
    char b[16];
    for (std::size_t i = 0; i < n; ++i) {
        std::snprintf(b, sizeof b, i ? " %08x" : "%08x", w[i]);
        s += b;
    }
    return s;
}

// The 8 dwords at arena offset `va`, if the arena holds them.
inline bool arena_words(std::uint64_t va, std::uint32_t* out, std::size_t n) {
    std::lock_guard lock(state().mutex);
    const State& s = state();
    if (!s.arena || va + n * 4 > s.arena_size || va + n * 4 < va) return false;
    std::memcpy(out, s.arena + va, n * 4);
    return true;
}

inline std::uint64_t arena_size() {
    std::lock_guard lock(state().mutex);
    return state().arena_size;
}

// Logs one descriptor table: `cpu` is the layer's host view of the slot at `va` (null when no heap holds `va`).
// For each slot: the CPU bytes, the arena bytes at the same VA (what a shader's heap_base + VA loads), and what the
// CPU bytes are (null, V# with its base, or T# with whether it has a bindless entry). A V# whose base is inside the
// arena has its first 16 dwords printed from the arena.
inline void describe_table(const char* tag, std::uint64_t va, std::uint32_t slots, const std::uint8_t* cpu,
                           const char* heap_kind) {
    const std::uint32_t shown = slots < 4 ? slots : 4;
    std::fprintf(stderr, "[rt] fsdraw: %s table va 0x%llx slots %u heap %s (arena size 0x%llx)\n", tag,
                 static_cast<unsigned long long>(va), slots, heap_kind, static_cast<unsigned long long>(arena_size()));
    for (std::uint32_t i = 0; i < shown; ++i) {
        const std::uint64_t at = va + std::uint64_t(i) * 32;
        std::uint32_t c[8]{}, g[8]{};
        const bool have_cpu = cpu != nullptr;
        if (have_cpu) std::memcpy(c, cpu + std::size_t(i) * 32, sizeof c);
        const bool have_gpu = arena_words(at, g, 8);
        const bool same = have_cpu && have_gpu && std::memcmp(c, g, sizeof c) == 0;
        std::fprintf(stderr, "[rt] fsdraw: %s   slot %u cpu [%s] gpu [%s]%s\n", tag, i,
                     have_cpu ? hex_words(c, 8).c_str() : "no heap", have_gpu ? hex_words(g, 8).c_str() : "OUT OF ARENA",
                     have_cpu && have_gpu ? (same ? " (same)" : " (DIFFER)") : "");
        const std::uint32_t* w = have_cpu ? c : have_gpu ? g : nullptr;
        if (!w) continue;
        if (!(w[0] | w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7])) {
            std::fprintf(stderr, "[rt] fsdraw: %s   slot %u is null\n", tag, i);
            continue;
        }
        if (!(w[3] >> 30)) {  // buffer V# (TYPE in dword 3 bits 30-31 is 0)
            const std::uint64_t base = std::uint64_t(w[0]) | (std::uint64_t(w[1] & 0xFFFFu) << 32);
            const std::uint32_t stride = (w[1] >> 16) & 0x3FFFu;
            std::uint32_t data[16]{};
            const bool in_arena = arena_words(base, data, 16);
            std::size_t nonzero = 0;
            if (in_arena) {
                std::vector<std::uint32_t> all(std::min<std::uint32_t>(w[2] ? w[2] : 256, 4096) / 4);
                if (!all.empty() && arena_words(base, all.data(), all.size()))
                    for (const std::uint32_t x : all) nonzero += x != 0;
            }
            std::uint64_t location = 0;
            const bool made = cbv_origin(base, location);
            std::fprintf(stderr, "[rt] fsdraw: %s   slot %u V# base 0x%llx stride %u records %u -> %s [%s] nonzero dwords "
                                 "%zu%s\n",
                         tag, i, static_cast<unsigned long long>(base), stride, w[2],
                         in_arena ? "arena" : "outside the arena buffer", in_arena ? hex_words(data, 16).c_str() : "",
                         nonzero, made ? "" : " (no CreateConstantBufferView made this V#)");
            // The CB as the title wrote it, when its location is the title's memory (CPU view): an upload buffer's
            // VA, also outside the arena, is no CPU address.
            std::uint64_t title_lo = 0, title_hi = 0;
            if (made && !in_arena && location && title_va_aliases().title_memory(location, title_lo, title_hi) &&
                title_hi - location >= 64) {
                std::uint32_t cb[16]{};
                std::memcpy(cb, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(location)), sizeof cb);
                std::fprintf(stderr, "[rt] fsdraw: %s   slot %u CB at title 0x%llx: [%s]\n", tag, i,
                             static_cast<unsigned long long>(location), hex_words(cb, 16).c_str());
            }
        } else {
            std::fprintf(stderr, "[rt] fsdraw: %s   slot %u T# bindless entry %s\n", tag, i,
                         has_tsharp(w) ? "yes" : "NO (a lookup misses: index 0, zeros)");
        }
    }
}

// LEONIDA_FSDRAW_POKE_CB=1 (experiment): before a submission, the CB that a bound table's slot-0 V# names gets a
// known colour in dwords 8..11 (what PS_BlitColour multiplies the vertex colour by), written into the title's own
// copy. A non-zero render target afterwards proves table -> V# -> CB -> pixel shader -> target works on the GPU.
inline bool poke_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("LEONIDA_FSDRAW_POKE_CB");
        return v && *v && *v != '0';
    }();
    return on;
}

inline void poke_cb(const std::uint8_t* table_cpu) {
    if (!table_cpu || !poke_enabled()) return;
    std::uint32_t w[4];
    std::memcpy(w, table_cpu, sizeof w);
    if (!(w[0] | w[1] | w[2] | w[3]) || (w[3] >> 30) || w[2] < 48) return;
    const std::uint64_t base = std::uint64_t(w[0]) | (std::uint64_t(w[1] & 0xFFFFu) << 32);
    std::uint64_t location = 0;
    if (!cbv_origin(base, location) || !location) return;
    const float colour[4] = {1.0f, 0.5f, 0.25f, 1.0f};
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(location + 32)), colour, sizeof colour);
    static std::atomic<std::uint64_t> poked{0};
    const std::uint64_t n = ++poked;
    if (n <= 4 || n % 600 == 0)
        std::fprintf(stderr, "[rt] fsdraw: POKE #%llu: CB at title 0x%llx dwords 8..11 = 1.0 0.5 0.25 1.0\n",
                     static_cast<unsigned long long>(n), static_cast<unsigned long long>(location));
}

} // namespace leonida::fsdraw
