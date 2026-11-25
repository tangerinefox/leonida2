#pragma once

#include "../src/command_list.h"
#include "../src/fsdraw_probe.h"
#include "../src/title_va_alias.h"
#include "../src/title_page_watch.h"
#include <windows.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace rt::compute_probe {
inline std::string_view filters() {
    static const char* value = std::getenv("LEONIDA_COMPUTE_INPUTS");
    return value ? value : "";
}
inline bool enabled() { return !filters().empty(); }
inline bool matches(const char* name) {
    if (!name) return false;
    auto parts = filters();
    while (!parts.empty()) {
        const auto end = parts.find(',');
        const auto part = parts.substr(0, end);
        if (!part.empty() && std::string_view(name).find(part) != std::string_view::npos) return true;
        if (end == parts.npos) break;
        parts.remove_prefix(end + 1);
    }
    return false;
}

// Each decimal pair selects a table root dword and a 32-byte descriptor index.
inline const auto& slots() {
    static const auto pairs = [] {
        std::vector<std::pair<unsigned, unsigned>> result;
        const char* p = std::getenv("LEONIDA_COMPUTE_INPUT_SLOTS");
        if (!p) return result;
        const char* end = p + std::strlen(p);
        while (p != end && result.size() < 256) {
            unsigned root{}, slot{};
            auto r = std::from_chars(p, end, root);
            if (r.ec != std::errc{} || r.ptr == end || *r.ptr != '/') break;
            auto s = std::from_chars(r.ptr + 1, end, slot);
            if (s.ec != std::errc{} || (s.ptr != end && *s.ptr != ',')) break;
            if (root < 32 && slot < (1u << 21)) result.emplace_back(root, slot);
            p = s.ptr == end ? end : s.ptr + 1;
        }
        return result;
    }();
    return pairs;
}

inline constexpr const char* writers[] = {"unobserved", "CreateConstantBufferView", "CreateShaderResourceView",
    "CreateUnorderedAccessView", "CopyDescriptors", "slot70", "slot71"};
inline std::uint8_t writer_code(std::string_view name) {
    for (std::uint8_t i = 1; i < std::size(writers); ++i)
        if (name == writers[i]) return i;
    return 0;
}

inline bool read_cpu(std::uint64_t address, void* out, std::size_t size) {
    SIZE_T read{};
    return address && ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address),
                                        out, size, &read) && read == size;
}

struct Backing {
    const char* kind;
    std::uint64_t title{}, host{}, available{};
};

inline bool read_backing(Backing b, std::uint64_t offset, void* out, std::size_t size) {
    if (!b.host || offset > b.available || size > b.available - offset) return false;
    // The device owns the arena mapping for its lifetime. Linux cross-process reads can reject a
    // device mapping even though the current process can read it directly.
    if (std::string_view(b.kind) == "arena" || std::string_view(b.kind) == "arena-shadow" ||
        std::string_view(b.kind) == "window-shadow") {
        std::memcpy(out, reinterpret_cast<const void*>(b.host + offset), size);
        return true;
    }
    return read_cpu(b.host + offset, out, size);
}

template<class State>
Backing backing(State& state, std::uint64_t base) {
    auto& aliases = leonida::title_va_aliases();
    std::uint64_t source{}, room{};
    const bool alias = aliases.source_of_vsharp(base, source, room);
    if (base < state.arena.size) {
        std::lock_guard lock(state.arena.mutex);
        auto next = state.arena.free.upper_bound(base);
        if (next != state.arena.free.begin()) {
            const auto [at, size] = *std::prev(next);
            if (base - at < size) return {"arena-free"};
        }
        const auto end = next == state.arena.free.end() ? state.arena.size : next->first;
        return {alias ? "arena-shadow" : "arena", source, reinterpret_cast<std::uint64_t>(state.arena.mapped) + base,
                alias ? std::min<std::uint64_t>(room, end - base) : end - base};
    }
    if (alias) {
        std::uint64_t host{}, bytes{};
        {
            std::lock_guard lock(state.title_window_mutex);
            if (state.external_windows.lookup(state.arena.address + base, host, bytes))
                return {"window-shadow", source, host, std::min(room, bytes)};
        }
        if (!state.context->diagnostic_buffer(source, host, bytes)) return {"alias-only", source};
        return {host == source ? "import" : host ? "external-shadow" : "external-buffer", source, host,
                std::min(room, bytes)};
    }
    std::uint64_t lo{}, hi{};
    if (state.geometry.cap()) {
        std::uint64_t mapped{}, size{}, source{};
        bool populated{};
        if (state.geometry.lookup(state.arena.address + base, mapped, size, &source, &populated))
            return {populated ? "geometry" : "geometry-pending", source, mapped, size};
    }
    if (aliases.title_memory(base, lo, hi)) return {"untranslated-title", base, 0, hi - base};
    const auto normalized = (base ^ (1ull << 47)) - (1ull << 47);
    std::uint64_t host{}, bytes{};
    if (state.context->diagnostic_buffer(normalized, host, bytes)) return {"registered-buffer", 0, host, bytes};
    return {"unknown"};
}

template<class State>
Backing descriptor(State& state, std::uint64_t seq, std::size_t list, std::size_t cmd,
                const std::array<std::uint32_t, 32>& roots, unsigned reg, unsigned slot) {
    const auto at = std::uint64_t(roots[reg]) + std::uint32_t(slot * 32u);
    std::array<std::uint32_t, 8> raw{}, gpu{};
    const char* origin = "no-heap";
    std::uint64_t index = 0, title_at = 0;
    bool table = false, have_gpu = false;
    {
        std::lock_guard lock(state.probe_mutex);
        for (const auto& heap : state.probes) {
            const auto delta = at - heap->shader_va;
            if (delta / 32 >= heap->slots || delta % 32) continue;
            index = delta / 32;
            title_at = heap->console_va + delta;
            table = read_cpu(title_at, raw.data(), sizeof(raw));
            if (!table) break;
            const auto h = State::probe_hash(reinterpret_cast<const std::uint8_t*>(raw.data()));
            origin = heap->api[index] == h && !heap->writer.empty() ? writers[heap->writer[index]] :
                     heap->api[index] ? "changed-since-API" : "no-API-write-recorded";
            if (heap->shadowed && at >= state.arena.size) {
                gpu = raw;
                have_gpu = true;
            }
            break;
        }
    }
    if (state.geometry.cap() && table && at >= state.arena.size) {
        std::lock_guard lock(state.console_heap_mutex);
        for (const auto& heap : state.console_heaps) {
            const auto delta = at - heap->shader_va;
            if (heap->copy && delta <= heap->bytes && sizeof(gpu) <= heap->bytes - delta) {
                std::memcpy(gpu.data(), heap->copy + delta, sizeof(gpu));
                have_gpu = true;
                break;
            }
        }
    }
    if (!have_gpu) have_gpu = leonida::fsdraw::arena_words(at, gpu.data(), gpu.size());
    const auto prefix = "[rt] compute-input: execute=%llu list=%zu cmd=%zu s%u/%u ";
    std::fprintf(stderr, prefix, (unsigned long long)seq, list, cmd, reg, slot);
    if (!have_gpu) {
        std::fprintf(stderr, "table=0x%llx no GPU-visible bytes\n", (unsigned long long)at);
        return {"missing-table"};
    }
    std::fprintf(stderr, "heap-slot=%llu table-title=0x%llx writer=%s words=[%s] ",
        (unsigned long long)index, (unsigned long long)title_at, origin,
        leonida::fsdraw::hex_words(gpu.data(), gpu.size()).c_str());
    if (table && raw != gpu)
        std::fprintf(stderr, "title-words=[%s] ", leonida::fsdraw::hex_words(raw.data(), raw.size()).c_str());
    if (gpu[3] >> 30) {
        std::fprintf(stderr, "texture key=%s\n", leonida::fsdraw::has_tsharp(gpu.data()) ? "registered" : "absent");
        return {"texture"};
    }
    if (!(gpu[0] | gpu[1] | gpu[2] | gpu[3])) { std::fputs("null\n", stderr); return {"null"}; }
    const auto base = std::uint64_t(gpu[0]) | (std::uint64_t(gpu[1] & 0xffff) << 32);
    const auto stride = (gpu[1] >> 16) & 0x3fffu;
    const auto span = std::uint64_t(gpu[2]) * (stride ? stride : 1);
    auto b = backing(state, base);
    const auto signed_base = (base ^ (1ull << 47)) - (1ull << 47);
    std::fprintf(stderr, "V#=0x%llx bda=0x%llx stride=%u records=%u storage=%s title=0x%llx",
        (unsigned long long)base, (unsigned long long)(state.arena.address + signed_base), stride, gpu[2],
        b.kind, (unsigned long long)b.title);
    std::array<std::uint32_t, 16> data{};
    const auto n = std::min<std::uint64_t>({span, b.available, sizeof(data)}) / 4;
    if (n && read_backing(b, 0, data.data(), 4 * n))
        std::fprintf(stderr, " cpu-sample=[%s]", leonida::fsdraw::hex_words(data.data(), n).c_str());
    if (std::string_view(b.kind) == "untranslated-title") {
        std::uint64_t created_at{};
        const bool from_cbv = leonida::fsdraw::cbv_origin(base, created_at);
        const auto source = leonida::title_gpu_views::to_cpu(base);
        const auto aliased = leonida::title_va_aliases().shader_va(source);
        MEMORY_BASIC_INFORMATION region{};
        VirtualQuery(reinterpret_cast<const void*>(source), &region, sizeof(region));
        std::fprintf(stderr, " source=0x%llx source-alias=0x%llx state=0x%lx protect=0x%lx",
                     (unsigned long long)source, (unsigned long long)aliased, region.State, region.Protect);
        if (from_cbv) std::fprintf(stderr, " recorded-CBV-location=0x%llx", (unsigned long long)created_at);
        const auto size = std::min<std::uint64_t>(span, sizeof(data)) / 4;
        if (size && region.State == MEM_COMMIT && !(region.Protect & (PAGE_NOACCESS | PAGE_GUARD)) &&
            read_cpu(source, data.data(), size * 4))
            std::fprintf(stderr, " source-only=[%s]", leonida::fsdraw::hex_words(data.data(), size).c_str());
    }
    std::fputc('\n', stderr);
    b.available = std::min(b.available, span);
    return b;
}

// The two measured culling kernels share this record format. Read a bounded CPU snapshot
// of the actual int offsets for each dispatched group. A prior GPU writer can still be in flight.
template<class State>
void culling_records(State& state, std::uint64_t seq, std::size_t list, std::size_t cmd,
                     const std::array<std::uint32_t, 32>& roots, unsigned groups, Backing indices, Backing records) {
    auto word = [](Backing b, std::uint64_t off, unsigned& out) {
        return read_backing(b, off, &out, sizeof(out));
    };
    std::set<unsigned> indices_seen;
    for (unsigned group = 0; group < std::min(groups, 256u); ++group) {
        unsigned record{};
        if (!word(indices, 4ull * group, record)) break;
        record &= ~3u;
        unsigned branches[2]{}, vb{}, count{};
        const bool valid = word(records, record, branches[0]) && word(records, record + 4ull, branches[1]) &&
                           word(records, record + 0x28ull, vb);
        std::fprintf(stderr, "[rt] compute-record: execute=%llu cmd=%zu group=%u offset=0x%x accessible=%u",
                     (unsigned long long)seq, cmd, group, record, valid);
        if (valid) {
            std::fprintf(stderr, " vertex-slot=%u", vb);
            unsigned byte_offset{}, first{}, count_instances{};
            if (word(records, record + 0x2cull, byte_offset) && word(records, record + 0x3cull, first) &&
                word(records, record + 0x40ull, count_instances))
                std::fprintf(stderr, " byte-offset=%u first=%u instances=%u", byte_offset, first, count_instances);
            indices_seen.insert(vb);
            for (unsigned branch : branches) {
                std::uint64_t node = branch & ~3u;
                unsigned indices_pair[2]{};
                if (word(records, node + 4, count)) std::fprintf(stderr, " node=0x%llx count4=%u", (unsigned long long)node, count);
                if (word(records, node + 0x374, count)) std::fprintf(stderr, " count374=%u", count);
                if (word(records, node + 0x39c, indices_pair[0]) && word(records, node + 0x3a0, indices_pair[1])) {
                    indices_seen.insert(indices_pair[0]);
                    indices_seen.insert(indices_pair[1]);
                }
            }
        }
        std::fputc('\n', stderr);
    }
    std::vector<std::uint32_t> alternatives;
    {
        std::lock_guard lock(state.probe_mutex);
        for (const auto& heap : state.probes)
            if (heap->shadowed && heap->shader_va <= UINT32_MAX && roots[7] != heap->shader_va)
                alternatives.push_back(static_cast<std::uint32_t>(heap->shader_va));
    }
    for (unsigned index : indices_seen) {
        descriptor(state, seq, list, cmd, roots, 7, index);
        for (auto heap : alternatives) {
            std::fprintf(stderr, "[rt] compute-record: comparison only, heap=0x%x index=%u\n", heap, index);
            auto expected = roots;
            expected[7] = heap;
            descriptor(state, seq, list, cmd, expected, 7, index);
        }
    }
}

template<class State>
void inspect(State& state, std::span<const leonida::CommandList* const> lists, std::uint64_t seq, const char* queue) {
    for (std::size_t l = 0; l < lists.size(); ++l) {
        const auto& cmds = lists[l]->commands();
        for (std::size_t i = 0; i < cmds.size(); ++i) {
            const auto* d = std::get_if<leonida::Dispatch>(&cmds[i]);
            const auto* ind = std::get_if<leonida::ExecuteIndirect>(&cmds[i]);
            if (!d && (!ind || ind->kind != leonida::IndirectKind::dispatch)) continue;
            const char* name = state.context->diagnostic_compute_name(d ? d->pipeline : ind->pipeline);
            if (!matches(name)) continue;
            const auto& roots = d ? d->user_data : ind->user_data;
            std::fprintf(stderr, "[rt] compute-input: execute=%llu list=%zu cmd=%zu queue=%s name=%s pso=%p %s groups=%u,%u,%u "
                         "heap-base=0x%llx roots=[%s]\n", (unsigned long long)seq, l, i, queue, name, reinterpret_cast<void*>(d ? d->pipeline : ind->pipeline),
                         d ? "direct" : "indirect", d ? d->x : 0, d ? d->y : 0, d ? d->z : 0,
                         (unsigned long long)state.arena.address, leonida::fsdraw::hex_words(roots.data(), roots.size()).c_str());
            Backing indices{}, records{};
            for (const auto& [reg, slot] : slots()) {
                const auto b = descriptor(state, seq, l, i, roots, reg, slot);
                if (reg == 5 && slot == 0) indices = b;
                if (reg == 4 && slot == 86) records = b;
            }
            if (d && (std::string_view(name).ends_with("lib_cs_01405") || std::string_view(name).ends_with("lib_cs_01410")))
                culling_records(state, seq, l, i, roots, d->x, indices, records);
        }
    }
}
} // namespace rt::compute_probe
