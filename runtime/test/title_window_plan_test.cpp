// The mapping a title address gets, without a GPU: src/title_window_plan.h (a window's range and what fits the
// LEONIDA_TITLE_WINDOWS_MB budget; one mapping per readable run of a stream pool) and src/title_va_alias.h with a
// resolver that runs out of budget (no raw address may come back) and with a pool's two addresses (src/title_page_watch.h,
// title_gpu_views) sharing one mapping. The title's stream pool is a 16-24 GiB reservation whose GPU-only alias and CPU view name the same
// frames, and a title address that gets no copy must not reach a shader as itself (heap_base + it is no allocation).
#include "../src/title_page_watch.h"
#include "../src/title_va_alias.h"
#include "../src/title_window_plan.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

constexpr std::uint64_t kAlign = 65536, kMiB = 1ull << 20, kArenaCap = 1ull << 32;

// A resolver that makes a window when the budget has room and answers "no backing" when it does not, as the device's
// resolve_title_memory does (d3d12_vulkan_device.cpp).
struct Budget {
    std::uint64_t cap = 64 * kMiB;
    std::uint64_t used = 0;
    std::uint64_t next = 0x10000;  // synthetic arena offsets
    std::uint64_t windows = 0, refusals = 0;
    leonida::TitleVaAliases* table = nullptr;
};

bool budget_resolver(void* context, std::uint64_t va, std::uint64_t bytes, std::uint64_t& out) {
    auto& budget = *static_cast<Budget*>(context);
    leonida::TitleWindowRange range{};
    if (!leonida::title_window_range(va, bytes, 0, UINT64_MAX, kAlign, 16 * kMiB, budget.cap, budget.used, range)) {
        ++budget.refusals;
        out = 0;  // title memory, no backing: the view is written with a zero-size range
        return true;
    }
    budget.used += range.size;
    ++budget.windows;
    const std::uint64_t shader = budget.next;
    budget.next += range.size;
    budget.table->add(range.start, range.size, shader);
    out = shader + (va - range.start);
    return true;
}

int check_window_budget() {
    Budget budget;
    leonida::TitleVaAliases table;
    budget.table = &table;
    table.set_hooks(&budget, &budget_resolver, nullptr);
    constexpr std::uint64_t title = 0x5800e0000ull;  // a stream-pool alias on the console, past the arena
    std::uint64_t backed = 0, zeroed = 0;
    for (unsigned i = 0; i < 800; ++i) {
        const std::uint64_t va = title + std::uint64_t(i) * 0x20000;
        const std::uint64_t to = table.resolve(va, 0x20000, leonida::TitleVaAliases::vertex_view);
        // Never the raw title address: a window (an offset below the arena's range), or 0 for a zero-size view.
        EXPECT(to == 0 || to < kArenaCap);
        EXPECT(to != va);
        if (to) ++backed;
        else ++zeroed;
    }
    EXPECT(backed != 0 && zeroed != 0);
    EXPECT(budget.windows == budget.cap / 0x20000);  // 512 aligned 128 KiB windows fill 64 MiB exactly
    EXPECT(budget.used == budget.cap);
    EXPECT(budget.refusals >= zeroed);
    // A window that does not fit is refused outright: no range and no charge.
    leonida::TitleWindowRange range{};
    EXPECT(!leonida::title_window_range(title, 0x20000, 0, UINT64_MAX, kAlign, 16 * kMiB, budget.cap, budget.cap,
                                        range));
    EXPECT(leonida::title_window_range(title, 0x10000, 0, UINT64_MAX, kAlign, 16 * kMiB, budget.cap,
                                       budget.cap - 0x10000, range));
    EXPECT(range.size == 0x10000);
    // A span that ends before the request backs nothing.
    EXPECT(!leonida::title_window_range(title, 0x20000, 0, title, kAlign, 16 * kMiB, budget.cap, 0, range));
    table.clear_hooks(&budget);
    std::printf("title window budget: %llu windows backed 800 requests, %llu zero-size views, %llu KiB of a %llu MiB "
                "budget, no raw address\n",
                static_cast<unsigned long long>(budget.windows), static_cast<unsigned long long>(zeroed),
                static_cast<unsigned long long>(budget.used >> 10), static_cast<unsigned long long>(budget.cap >> 20));
    return 0;
}

// One stream pool's pages: an RW CPU view (the pages the title commits and the host reads) and a GPU-only alias of
// the same size, both uncommitted at the alias, as the title's xmem reservations are (protect 0x801).
int check_stream_pool() {
    constexpr std::uint64_t pool = 1 * kMiB;
    void* cpu_pages = VirtualAlloc(nullptr, pool, MEM_TOP_DOWN | MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    void* gpu_alias = VirtualAlloc(nullptr, pool, MEM_TOP_DOWN | MEM_RESERVE, PAGE_NOACCESS);
    EXPECT(cpu_pages && gpu_alias);
    const auto cpu = reinterpret_cast<std::uintptr_t>(cpu_pages);
    const auto gpu = reinterpret_cast<std::uintptr_t>(gpu_alias);
    EXPECT(!leonida::title_gpu_views::note_reservation(cpu, pool, false));  // remembered as the CPU twin
    EXPECT(leonida::title_gpu_views::note_reservation(gpu, pool, true));    // paired: true

    // The pair is found from either address, and rule B sends the GPU-only pages to the CPU view the host reads.
    leonida::title_gpu_views::Range view{};
    EXPECT(leonida::title_gpu_views::view_at(gpu + 0x20000, 0x20000, view) && view.gpu == gpu && view.cpu == cpu);
    EXPECT(leonida::title_gpu_views::view_at(cpu + 0x20000, 0x20000, view) && view.gpu == gpu && view.cpu == cpu);
    EXPECT(leonida::title_gpu_views::to_host_view(gpu + 0x20000, 0x20000) == cpu + 0x20000);
    EXPECT(leonida::title_gpu_views::to_host_view(cpu + 0x20000, 0x20000) == cpu + 0x20000);
    EXPECT(leonida::title_gpu_views::other_view(gpu + 0x20000) == cpu + 0x20000);

    // One mapping for the readable run: two vertex-buffer requests inside it compute the same extent, so they cost
    // one window, not one 128 KiB window each.
    const std::uint64_t readable_end = cpu + pool;
    leonida::TitleWindowRange first{}, second{};
    EXPECT(leonida::title_pool_extent(cpu, cpu + pool, cpu + 0x20000, readable_end, kAlign, 1 * kMiB, first));
    EXPECT(leonida::title_pool_extent(cpu, cpu + pool, cpu + 0x28000, readable_end, kAlign, 1 * kMiB, second));
    EXPECT(first.start == cpu + 0x20000 && first.size == 0xE0000);
    EXPECT(second.start == first.start && second.size == first.size);  // one mapping for both requests
    // The extent stops at the first page the title has not committed.
    leonida::TitleWindowRange partial{};
    EXPECT(leonida::title_pool_extent(cpu, cpu + pool, cpu + 0x20000, cpu + 0x30000, kAlign, 1 * kMiB, partial));
    EXPECT(partial.start == cpu + 0x20000 && partial.size == 0x10000);
    EXPECT(!leonida::title_pool_extent(cpu, cpu + pool, cpu + 0x20000, cpu + 0x20000, kAlign, 1 * kMiB, partial));

    // The mapping the device makes: one shader VA for the extent, entered for both addresses of the pair.
    leonida::TitleVaAliases table;
    constexpr std::uint64_t shader = 0x3000000;
    const auto twin_start = static_cast<std::uint64_t>(leonida::title_gpu_views::other_view(first.start));
    EXPECT(table.add_twin(first.start, twin_start, first.size, shader));
    for (std::uint64_t off = 0; off < pool; off += 0x40000) {
        EXPECT(table.shader_va(first.start + off) == shader + off);      // the CPU view
        EXPECT(table.shader_va(twin_start + off) == shader + off);       // the GPU-only alias: the same backing
        EXPECT(table.resolve(first.start + off, 0x20000, leonida::TitleVaAliases::vertex_view) == shader + off);
        EXPECT(table.resolve(twin_start + off, 0x20000, leonida::TitleVaAliases::vertex_view) == shader + off);
    }
    // One mapping, two ranges: 8 lookups on both sides called no resolver and made no second mapping.
    EXPECT(table.translated(leonida::TitleVaAliases::vertex_view) == 8);
    EXPECT(table.size() == 2);

    leonida::title_gpu_views::state().views.clear();
    leonida::title_gpu_views::state().count.store(0);
    VirtualFree(gpu_alias, 0, MEM_RELEASE);
    VirtualFree(cpu_pages, 0, MEM_RELEASE);
    std::printf("stream pool: alias 0x%llx and CPU view 0x%llx share one %llu KiB mapping; 8 lookups, no second "
                "mapping\n",
                static_cast<unsigned long long>(gpu), static_cast<unsigned long long>(cpu),
                static_cast<unsigned long long>(first.size >> 10));
    return 0;
}

}  // namespace

int main() {
    if (check_window_budget()) return 1;
    if (check_stream_pool()) return 1;
    std::printf("title_window_plan: a refused window gives no raw base, and a stream pool's two views share one "
                "mapping\n");
    return 0;
}
