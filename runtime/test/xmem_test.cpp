// Checks the registered xmem.dll shim against the measured Xbox ABI.
//
// Every function is resolved through rt::shims::find, so the test exercises the same shim table and calling
// convention the runtime installs for xmem.dll imports instead of a private copy of the implementation. The checks
// cover what the measured call sites use: 64-byte-aligned allocations, XMemFree, working-set statistics with the
// size at +8, and XMemVirtualQuery's MEMORY_BASIC_INFORMATION prefix inside a zeroed 64-byte output.
// XMemTransferMemory follows the call shape in Microsoft's public samples: TRUE for any direction and size, with
// nothing committed, and one trace event per distinct (type, size). XMemVirtualAlloc accepts attribute words and console
// graphics protection bits and commits 2 MiB guest pages; the placeholder flags pass through; physical pages keep their
// contents across unmap and remap, alias when mapped twice, and scatter-map per address.
#include <windows.h>
#include <malloc.h>
#include <psapi.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "shims.h"
#include "startup_trace.h"
#include "xmem.h"
#include "../src/title_page_watch.h"

// The process heap as test_heap_lock_faults needs it (the same gate as d3d12_existing_heap_test's): an allocation that
// waits while a thread holds the heap's lock. The NT heap's low-fragmentation front end serves most small sizes
// without that lock, so a real deadlock waits for an allocation its back end serves, which a short test can't count
// on. Here, while the gate is on, every operator new and delete of the program (xmem.cpp's included) takes
// g_heap_gate shared, and the test's "title" threads that hold the heap's lock hold the gate exclusively: any
// allocation xmem makes while it holds a lock its fault handler waits for then deadlocks at once.
namespace {
SRWLOCK g_heap_gate = SRWLOCK_INIT;
std::atomic<bool> g_heap_gate_on{false};
struct HeapGate {
    const bool on = g_heap_gate_on.load(std::memory_order_acquire);
    HeapGate() {
        if (on) AcquireSRWLockShared(&g_heap_gate);
    }
    ~HeapGate() {
        if (on) ReleaseSRWLockShared(&g_heap_gate);
    }
};
}  // namespace

void* operator new(std::size_t bytes) {
    HeapGate gate;
    if (void* p = std::malloc(bytes ? bytes : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept {
    HeapGate gate;
    std::free(p);
}
void operator delete(void* p, std::size_t) noexcept { operator delete(p); }

namespace {

using XMemAllocFn = void*(__stdcall*)(size_t, uint64_t);
using XMemFreeFn = void(__stdcall*)(void*, uint64_t);
using XMemVirtualQueryFn = SIZE_T(__stdcall*)(HANDLE, const void*, void*);
// BOOL XMemGetWorkingSetStatistics(UINT32 kind, stats*): the FSR DLL passes kind 0x20000001 at every call site.
using XMemGetWorkingSetStatisticsFn = BOOL(__stdcall*)(uint32_t, void*);
using XMemVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using XMemGetWriteWatchFn = UINT(WINAPI*)(DWORD, void*, SIZE_T, void**, ULONG_PTR*, ULONG*);
using XMemResetWriteWatchFn = UINT(WINAPI*)(void*, SIZE_T);
// BOOL XMemTransferMemory(XMEM_TRANSFER_TYPE, SIZE_T): the shape Microsoft's public NetRumble and NativeAOT-Xbox
// samples call (see runtime/shims/xmem.cpp).
using XMemTransferMemoryFn = BOOL(WINAPI*)(uint32_t, SIZE_T);
// The shapes measured at a title's call sites (see runtime/shims/xmem.cpp).
using XMemIsToolsMemAvailableFn = BOOL(WINAPI*)();
using XMemAllocatePhysicalPagesFn = BOOL(WINAPI*)(uint64_t, uint64_t*, uint64_t*);
using XMemFreePhysicalPagesFn = BOOL(WINAPI*)(uint64_t, const uint64_t*);
using XMemMapPhysicalPagesFn = void*(WINAPI*)(void*, uint64_t, const uint64_t*);
using XMemMapPhysicalPagesScatterFn = BOOL(WINAPI*)(void* const*, uint32_t, const uint64_t*);
constexpr uint32_t kMeasuredKind = 0x20000001;
constexpr DWORD kXmemLargePages = 0x20000000;
constexpr DWORD kXmem64kPages = 0x20400000;
constexpr SIZE_T kPage64k = 0x10000;
constexpr SIZE_T kPage2m = 0x200000;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

bool aligned_64(const void* p) { return p && (reinterpret_cast<uintptr_t>(p) & 63u) == 0; }

void test_alloc(XMemAllocFn alloc, XMemFreeFn free_mem) {
    void* const zero = alloc(0, 0);
    check(zero != nullptr, "XMemAlloc(0, 0) returns non-null");
    check(aligned_64(zero), "XMemAlloc(0, 0) returns a 64-byte aligned block");

    bool zero_readable = false;
    if (zero) {
        auto* bytes = static_cast<uint8_t*>(zero);
        bytes[0] = 0xa5;
        zero_readable = bytes[0] == 0xa5;
    }
    check(zero_readable, "XMemAlloc(0, 0) block is writable and readable");

    void* const big = alloc(4096, 0x5e50000);
    check(big != nullptr, "XMemAlloc(4096, 0x5e50000) returns non-null");
    check(aligned_64(big), "XMemAlloc(4096, 0x5e50000) returns a 64-byte aligned block");

    bool pattern_ok = big != nullptr;
    if (big) {
        auto* bytes = static_cast<uint8_t*>(big);
        for (size_t i = 0; i < 4096; ++i) bytes[i] = static_cast<uint8_t>((i * 37u + 11u) & 0xffu);
        for (size_t i = 0; i < 4096; ++i) {
            if (bytes[i] != static_cast<uint8_t>((i * 37u + 11u) & 0xffu)) {
                pattern_ok = false;
                break;
            }
        }
    }
    check(pattern_ok, "XMemAlloc(4096, 0x5e50000) block holds a write/read pattern");

    free_mem(zero, 0);
    free_mem(big, 0x5e50000);
    check(true, "XMemFree accepts both blocks (no crash)");
}

void test_working_set(XMemGetWorkingSetStatisticsFn get_stats) {
    char saved_env[64]{};
    const DWORD saved_n = GetEnvironmentVariableA("LEONIDA_TITLE_MEMORY_MB", saved_env, sizeof(saved_env));
    SetEnvironmentVariableA("LEONIDA_TITLE_MEMORY_MB", nullptr);

    alignas(64) uint8_t stats[64];
    std::memset(stats, 0xa5, sizeof(stats));
    // The pointer is the second argument: the first (0x20000001 here, as the FSR DLL passes it) must never be written
    // through, which is what the one-argument form did.
    const BOOL ok = get_stats(kMeasuredKind, stats);
    check(ok != FALSE, "XMemGetWorkingSetStatistics(kind, 64-byte out) returns TRUE");

    uint64_t total_memory = 0;
    std::memcpy(&total_memory, stats + 0, sizeof(total_memory));
    check(ok != FALSE && total_memory != 0, "XMemGetWorkingSetStatistics writes a non-zero total memory at +0");

    MEMORYSTATUSEX ms{sizeof(ms)};
    if (GlobalMemoryStatusEx(&ms) && ms.ullTotalPhys != 0) {
        check(ok != FALSE && total_memory == ms.ullTotalPhys,
              "XMemGetWorkingSetStatistics total memory (+0) matches GlobalMemoryStatusEx ullTotalPhys");
    } else {
        check(ok != FALSE && total_memory == 12ULL * 1024 * 1024 * 1024,
              "XMemGetWorkingSetStatistics total memory (+0) falls back to 12 GiB title limit");
    }

    uint64_t working_set = 0;
    std::memcpy(&working_set, stats + 8, sizeof(working_set));
    check(ok != FALSE && working_set != 0, "XMemGetWorkingSetStatistics writes a non-zero size at +8");

    check(ok != FALSE && total_memory >= working_set,
          "XMemGetWorkingSetStatistics total memory (+0) >= working set (+8)");

    bool tail_untouched = true;
    for (size_t i = 16; i < sizeof(stats); ++i) {
        if (stats[i] != 0xa5) {
            tail_untouched = false;
            break;
        }
    }
    check(ok != FALSE && tail_untouched, "XMemGetWorkingSetStatistics leaves bytes 16..63 untouched");

    // LEONIDA_TITLE_MEMORY_MB override and clamping (matches XSystemGetAppMemoryUsageLimit).
    auto query_total = [&]() -> uint64_t {
        alignas(64) uint8_t buf[64]{};
        if (!get_stats(kMeasuredKind, buf)) return 0;
        uint64_t out = 0;
        std::memcpy(&out, buf, sizeof(out));
        return out;
    };
    SetEnvironmentVariableA("LEONIDA_TITLE_MEMORY_MB", "6144");
    check(query_total() == 6144ULL * 1024ULL * 1024ULL,
          "XMemGetWorkingSetStatistics honors LEONIDA_TITLE_MEMORY_MB=6144");
    SetEnvironmentVariableA("LEONIDA_TITLE_MEMORY_MB", "512");
    check(query_total() == 2048ULL * 1024ULL * 1024ULL,
          "XMemGetWorkingSetStatistics clamps LEONIDA_TITLE_MEMORY_MB below 2048 MiB");
    SetEnvironmentVariableA("LEONIDA_TITLE_MEMORY_MB", "131072");
    check(query_total() == 65536ULL * 1024ULL * 1024ULL,
          "XMemGetWorkingSetStatistics clamps LEONIDA_TITLE_MEMORY_MB above 65536 MiB");
    SetEnvironmentVariableA("LEONIDA_TITLE_MEMORY_MB", "invalid");
    check(query_total() == total_memory,
          "XMemGetWorkingSetStatistics ignores invalid LEONIDA_TITLE_MEMORY_MB");
    SetEnvironmentVariableA("LEONIDA_TITLE_MEMORY_MB", "-4096");
    check(query_total() == total_memory,
          "XMemGetWorkingSetStatistics ignores negative LEONIDA_TITLE_MEMORY_MB");
    SetEnvironmentVariableA("LEONIDA_TITLE_MEMORY_MB", (saved_n > 0 && saved_n < sizeof(saved_env)) ? saved_env : nullptr);

    check(get_stats(kMeasuredKind, nullptr) == FALSE,
          "XMemGetWorkingSetStatistics(kind, nullptr) returns FALSE without crashing");
}

void test_virtual_query(XMemVirtualQueryFn query) {
    uint8_t local = 0;
    const void* const local_address = &local;

    alignas(64) uint8_t out[64];
    std::memset(out, 0xa5, sizeof(out));
    const SIZE_T written = query(GetCurrentProcess(), local_address, out);
    check(written != 0, "XMemVirtualQuery(current process, stack local) returns non-zero");

    MEMORY_BASIC_INFORMATION info{};
    std::memcpy(&info, out, sizeof(info));
    const uintptr_t base = reinterpret_cast<uintptr_t>(info.BaseAddress);
    const uintptr_t address = reinterpret_cast<uintptr_t>(local_address);
    const bool contains_local =
        base != 0 && info.RegionSize != 0 && address >= base && address - base < info.RegionSize;
    check(written != 0 && contains_local,
          "XMemVirtualQuery fills MEMORY_BASIC_INFORMATION with a region containing the local");

    bool tail_zero = true;
    for (size_t i = sizeof(info); i < sizeof(out); ++i) {
        if (out[i] != 0) {
            tail_zero = false;
            break;
        }
    }
    check(written != 0 && tail_zero, "XMemVirtualQuery leaves the 64-byte output tail zero");

    alignas(64) uint8_t bad_out[64];
    std::memset(bad_out, 0xa5, sizeof(bad_out));
    // Address 1 is queryable (it falls in the free region starting at 0), so use a non-canonical address that
    // VirtualQueryEx rejects.
    const void* const unmappable = reinterpret_cast<const void*>(~static_cast<uintptr_t>(0));
    const SIZE_T bad = query(GetCurrentProcess(), unmappable, bad_out);
    check(bad == 0, "XMemVirtualQuery on an invalid address returns 0 without crashing");
}

void test_virtual_alloc(XMemVirtualAllocFn virtual_alloc, XMemGetWriteWatchFn get_write_watch,
                       XMemResetWriteWatchFn reset_write_watch) {
    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    const SIZE_T page_size = system_info.dwPageSize;
    check(page_size != 0, "GetSystemInfo reports a non-zero page size");
    if (page_size == 0) return;

    const SIZE_T region_size = page_size * 3;
    void* const base = virtual_alloc(nullptr, region_size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, 0,
                                     PAGE_READWRITE);
    check(base != nullptr, "XMemVirtualAlloc creates a synthetic write-watch region");
    if (!base) return;

    void* watched[4]{};
    ULONG_PTR count = 4;
    ULONG granularity = 0;
    UINT status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    check(status == NO_ERROR && count == 0, "new write-watch region starts with no dirty pages");
    check(status == NO_ERROR && granularity == page_size, "write-watch reports the host page granularity");

    auto* const bytes = static_cast<volatile uint8_t*>(base);
    bytes[0] = 0xa1;
    bytes[page_size * 2] = 0xb2;

    count = 4;
    status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    bool saw_first_page = false;
    bool saw_third_page = false;
    bool all_addresses_are_page_aligned = count <= 4;
    for (ULONG_PTR i = 0; i < count && i < 4; ++i) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(watched[i]);
        const uintptr_t start = reinterpret_cast<uintptr_t>(base);
        all_addresses_are_page_aligned &= address >= start && (address - start) % page_size == 0;
        saw_first_page |= address == start;
        saw_third_page |= address == start + page_size * 2;
    }
    check(status == NO_ERROR && count == 2 && saw_first_page && saw_third_page,
          "write-watch reports each written page by its relative address");
    check(all_addresses_are_page_aligned, "write-watch addresses are page aligned and within the region");

    count = 4;
    status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    check(status == NO_ERROR && count == 2, "query-only write-watch leaves dirty pages recorded");

    status = reset_write_watch(base, region_size);
    check(status == NO_ERROR, "XMemResetWriteWatch clears the synthetic region");
    count = 4;
    status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    check(status == NO_ERROR && count == 0, "reset write-watch query returns no dirty pages");

    bytes[page_size] = 0xc3;
    count = 4;
    status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    check(status == NO_ERROR && count == 1 &&
              reinterpret_cast<uintptr_t>(watched[0]) == reinterpret_cast<uintptr_t>(base) + page_size,
          "write-watch records writes again after reset");

    check(VirtualFree(base, 0, MEM_RELEASE), "release the synthetic write-watch region");

    // 2 MiB guest pages: the region is 2 MiB-aligned, rounded up to whole 2 MiB pages and committed.
    void* const large_page = virtual_alloc(nullptr, kPage2m + page_size, MEM_RESERVE | MEM_COMMIT | kXmemLargePages,
                                           0x183, PAGE_READWRITE | PAGE_GRAPHICS_READWRITE | PAGE_GRAPHICS_COHERENT);
    check(large_page != nullptr, "XMemVirtualAlloc commits 2 MiB guest pages");
    if (large_page) {
        check((reinterpret_cast<uintptr_t>(large_page) & (kPage2m - 1)) == 0, "a 2 MiB-page region is 2 MiB-aligned");
        MEMORY_BASIC_INFORMATION info{};
        VirtualQuery(large_page, &info, sizeof(info));
        check(info.State == MEM_COMMIT && info.RegionSize >= 2 * kPage2m && info.Protect == PAGE_READWRITE,
              "a 2 MiB-page region is rounded up to whole pages, committed read-write without the graphics bits");
        static_cast<volatile uint8_t*>(large_page)[2 * kPage2m - 1] = 0x77;
        check(VirtualFree(large_page, 0, MEM_RELEASE), "release the 2 MiB-page region");
    }

    // The attribute word is accepted (every measured call passes a nonzero one) and the protection loses its graphics
    // bits, in the 64 KiB page mode as well.
    void* const attributed = virtual_alloc(nullptr, kPage64k, MEM_RESERVE | MEM_COMMIT | kXmem64kPages, 0x181,
                                           PAGE_READWRITE | PAGE_GRAPHICS_READWRITE);
    check(attributed != nullptr, "XMemVirtualAlloc accepts a nonzero attribute word and console protection bits");
    if (attributed) {
        MEMORY_BASIC_INFORMATION info{};
        VirtualQuery(attributed, &info, sizeof(info));
        check(info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE,
              "the region is committed PAGE_READWRITE (graphics bits dropped)");
        static_cast<volatile uint8_t*>(attributed)[0] = 0x42;
        check(VirtualFree(attributed, 0, MEM_RELEASE), "release the attributed region");
    }

    // The retail executable's GPU pools: the attribute word is recorded, and the PAGE_GRAPHICS_* bits are dropped.
    MEMORY_BASIC_INFORMATION mbi{};
    void* const pool = virtual_alloc(nullptr, SIZE_T{8} << 30, MEM_RESERVE | 0x20400000, 0x21, PAGE_READWRITE);
    check(pool != nullptr, "attributes 0x21 with 64 KiB pages reserves 8 GiB");
    if (pool) {
        check(VirtualQuery(pool, &mbi, sizeof mbi) && mbi.State == MEM_RESERVE, "the 8 GiB pool is reserve-only");
        check(VirtualFree(pool, 0, MEM_RELEASE), "release the 8 GiB pool");
    }

    void* const noaccess = virtual_alloc(nullptr, 0x10000, MEM_RESERVE | 0x20400000, 0x3, 0x801);
    check(noaccess != nullptr, "attributes 0x3 with protect PAGE_GRAPHICS_NOACCESS|PAGE_NOACCESS succeeds");
    if (noaccess) {
        check(VirtualQuery(noaccess, &mbi, sizeof mbi) && mbi.AllocationProtect == PAGE_NOACCESS,
              "protect 0x801 becomes PAGE_NOACCESS");
        check(VirtualFree(noaccess, 0, MEM_RELEASE), "release the PAGE_NOACCESS pool");
    }

    void* const gpu = virtual_alloc(nullptr, 0x10000, MEM_RESERVE | MEM_COMMIT, 0x183, 0x22004);
    check(gpu != nullptr, "attributes 0x183 with protect 0x22004 succeeds");
    if (gpu) {
        check(VirtualQuery(gpu, &mbi, sizeof mbi) && mbi.Protect == PAGE_READWRITE, "protect 0x22004 becomes PAGE_READWRITE");
        static_cast<volatile uint8_t*>(gpu)[0] = 0x5a;
        check(static_cast<volatile uint8_t*>(gpu)[0] == 0x5a, "the 0x183 allocation is writable");
        check(VirtualFree(gpu, 0, MEM_RELEASE), "release the 0x183 allocation");
    }

    SetLastError(ERROR_SUCCESS);
    void* const replace_without_address =
        virtual_alloc(nullptr, kPage64k, MEM_RESERVE | MEM_REPLACE_PLACEHOLDER | kXmem64kPages, 0, PAGE_READWRITE);
    check(replace_without_address == nullptr && GetLastError() == ERROR_INVALID_PARAMETER,
          "MEM_REPLACE_PLACEHOLDER without an address is rejected");
}

// XMemAllocDefault / XMemFreeDefault share XMemAlloc's heap: a block from either is freed by either.
void test_default_alloc(XMemAllocFn alloc, XMemFreeFn free_mem, XMemAllocFn alloc_default,
                        XMemFreeFn free_default) {
    void* const a = alloc_default(100, 0x5e50000);
    check(aligned_64(a), "XMemAllocDefault returns a 64-byte aligned block");
    void* const zero = alloc_default(0, 0);
    check(zero != nullptr, "XMemAllocDefault(0) returns non-null");
    void* const b = alloc(100, 0);
    if (a) std::memset(a, 0x11, 100);
    if (b) std::memset(b, 0x22, 100);
    free_mem(a, 0x5e50000);
    free_default(b, 0);
    free_default(zero, 0);
    check(true, "XMemFree releases an XMemAllocDefault block and XMemFreeDefault an XMemAlloc block");
}

uint32_t read_u32(const void* base, size_t offset) {
    uint32_t value = 0;
    std::memcpy(&value, static_cast<const uint8_t*>(base) + offset, sizeof(value));
    return value;
}

void fill_page(void* page, uint32_t tag) {
    auto* words = static_cast<uint32_t*>(page);
    for (size_t i = 0; i < kPage64k / sizeof(uint32_t); i += 1024) words[i] = tag + static_cast<uint32_t>(i);
}

bool page_holds(const void* page, uint32_t tag) {
    for (size_t i = 0; i < kPage64k / sizeof(uint32_t); i += 1024) {
        if (read_u32(page, i * sizeof(uint32_t)) != tag + static_cast<uint32_t>(i)) return false;
    }
    return true;
}

bool reserved_only(const void* address) {
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(address, &info, sizeof(info)) && info.State == MEM_RESERVE;
}

// The address is inside a section view (whatever state the host reports for its pages), not a bare reservation.
bool in_view(const void* address) {
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(address, &info, sizeof(info)) && info.Type == MEM_MAPPED;
}

// Physical pages are views of a reserved section and are committed on first touch. Windows then reports the page as
// MEM_COMMIT with its protection. Wine's VirtualQuery keeps reporting MEM_RESERVE and no protection for a view page
// that has been written and reads back, so a check of the reported state and protection has nothing to look at
// there. True for exactly that answer on that host; the caller says what it skips.
bool touched_view_reported_reserved(const MEMORY_BASIC_INFORMATION& info) {
    static const bool wine = [] {
        const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
        return ntdll && GetProcAddress(ntdll, "wine_get_version") != nullptr;
    }();
    return wine && info.Type == MEM_MAPPED && info.State == MEM_RESERVE && info.Protect == 0;
}

struct PhysicalApi {
    XMemVirtualAllocFn virtual_alloc;
    XMemAllocatePhysicalPagesFn allocate;
    XMemFreePhysicalPagesFn free_pages;
    XMemMapPhysicalPagesFn map;
    XMemMapPhysicalPagesScatterFn scatter;
};

// Physical pages in a 64 KiB-page window, as the measured callers use them: reserve a window, allocate pages, map them,
// remap them elsewhere and in another order, unmap, free.
void test_physical_pages(const PhysicalApi& api) {
    constexpr uint64_t kPages = 4;
    auto* const window = static_cast<uint8_t*>(api.virtual_alloc(
        nullptr, kPages * kPage64k, MEM_RESERVE | kXmem64kPages, 0x23, PAGE_READWRITE | PAGE_GRAPHICS_READWRITE));
    check(window != nullptr, "XMemVirtualAlloc reserves a 64 KiB-page window");
    auto* const second = static_cast<uint8_t*>(api.virtual_alloc(
        nullptr, kPages * kPage64k, MEM_RESERVE | kXmem64kPages, 0x21, PAGE_READWRITE));
    check(second != nullptr, "XMemVirtualAlloc reserves a second window");
    if (!window || !second) return;
    check(reserved_only(window), "the window is reserved, not committed");

    uint64_t pages[kPages]{};
    uint64_t count = kPages;
    check(api.allocate(0x280, &count, pages) == TRUE && count == kPages,
          "XMemAllocatePhysicalPages allocates every page asked for and keeps the count");
    bool numbers_ok = true;
    for (uint64_t i = 0; i < kPages; ++i) {
        numbers_ok &= pages[i] != 0 && pages[i] % 16 == 0 && pages[i] < (uint64_t{1} << 25);
        for (uint64_t j = 0; j < i; ++j) numbers_ok &= pages[i] != pages[j];
    }
    check(numbers_ok, "page numbers are distinct non-zero multiples of 16 below 2^25");

    auto generation = leonida::title_memory_map::generation();
    check(api.map(window, kPages, pages) == window, "XMemMapPhysicalPages returns the address it mapped");
    check(leonida::title_memory_map::generation() != generation, "mapping physical pages invalidates shadows");
    for (uint64_t i = 0; i < kPages; ++i) fill_page(window + i * kPage64k, 0x1000u * static_cast<uint32_t>(i + 1));
    {
        MEMORY_BASIC_INFORMATION info{};
        VirtualQuery(window, &info, sizeof(info));
        if (touched_view_reported_reserved(info)) {
            std::printf("SKIP the mapped page's reported state and protection: this host's VirtualQuery shows a "
                        "written view page as reserved (Wine)\n");
            check(info.Type == MEM_MAPPED && info.AllocationProtect == PAGE_READWRITE &&
                      page_holds(window, 0x1000u),
                  "a mapped page is a read-write view that holds what was written");
        } else {
            check(info.State == MEM_COMMIT && info.Type == MEM_MAPPED && info.Protect == PAGE_READWRITE,
                  "a mapped page is a read-write view with the window's protection");
        }
        check(info.AllocationBase == window && info.RegionSize == kPages * kPage64k,
              "consecutive pages of one allocation are mapped as one view");
    }

    // A page unmapped or remapped inside that view splits it; its neighbours keep their pages.
    void* const middle = window + kPage64k;
    generation = leonida::title_memory_map::generation();
    check(api.scatter(&middle, 1, nullptr) == TRUE && reserved_only(window + kPage64k),
          "a page inside a view unmaps on its own");
    check(page_holds(window, 0x1000u) && page_holds(window + 2 * kPage64k, 0x3000u) &&
              page_holds(window + 3 * kPage64k, 0x4000u),
          "the pages around it stay mapped with their contents");
    check(leonida::title_memory_map::generation() != generation, "scatter unmapping invalidates shadows");
    check(api.map(middle, 1, &pages[1]) == middle && page_holds(window + kPage64k, 0x2000u),
          "a page maps again into the gap");

    check(api.map(window, kPages, nullptr) == window, "XMemMapPhysicalPages with no page array unmaps");
    check(reserved_only(window) && reserved_only(window + 3 * kPage64k), "unmapped pages are reserved again");
    // The split above left the range as several host placeholders; the whole run maps again over them.
    check(api.map(window, kPages, pages) == window && page_holds(window, 0x1000u) &&
              page_holds(window + 3 * kPage64k, 0x4000u),
          "a run maps again over a range an earlier partial unmap split");
    check(api.map(window, kPages, nullptr) == window && reserved_only(window), "and unmaps again");

    // Remap in reverse order into the other window: each page keeps its contents.
    uint64_t reversed[kPages];
    for (uint64_t i = 0; i < kPages; ++i) reversed[i] = pages[kPages - 1 - i];
    check(api.map(second, kPages, reversed) == second, "the pages map into another window");
    bool kept = true;
    for (uint64_t i = 0; i < kPages; ++i) {
        kept &= page_holds(second + i * kPage64k, 0x1000u * static_cast<uint32_t>(kPages - i));
    }
    check(kept, "a physical page keeps its contents across unmap and remap");

    // One page mapped at two addresses is one memory.
    check(api.map(window, 1, &pages[0]) == window, "a page already mapped elsewhere maps again");
    static_cast<volatile uint32_t*>(static_cast<void*>(window))[1] = 0xfeedf00d;
    check(read_u32(second + 3 * kPage64k, 4) == 0xfeedf00d, "both mappings of a page see the same memory");

    // Mapping over a mapped page replaces it; a 0 entry unmaps one page.
    const uint64_t replace[2] = {pages[1], 0};
    check(api.map(window, 2, replace) == window, "mapping over mapped pages replaces them");
    check(page_holds(window, 0x2000u) && reserved_only(window + kPage64k),
          "the first page now shows page 1 and a 0 entry left the second unmapped");

    // Scatter: entry i maps page i at address i, out of order; 0 unmaps.
    void* const addresses[3] = {window + 3 * kPage64k, window + 2 * kPage64k, window};
    const uint64_t scattered[3] = {pages[2], pages[3], 0};
    check(api.scatter(addresses, 3, scattered) == TRUE, "XMemMapPhysicalPagesScatter maps each page at its address");
    check(page_holds(window + 3 * kPage64k, 0x3000u) && page_holds(window + 2 * kPage64k, 0x4000u) &&
              reserved_only(window),
          "scattered pages land at their own addresses and a 0 entry unmaps");
    check(api.scatter(addresses, 3, nullptr) == TRUE && reserved_only(window + 3 * kPage64k),
          "XMemMapPhysicalPagesScatter with no page array unmaps");

    uint8_t outside = 0;
    SetLastError(ERROR_SUCCESS);
    check(api.map(reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(&outside) + 0xffff) & ~uintptr_t{0xffff}), 1,
                  pages) == nullptr,
          "mapping outside any window fails");
    check(api.map(window + 1, 1, pages) == nullptr && GetLastError() == ERROR_INVALID_PARAMETER,
          "mapping at an address that is not 64 KiB-aligned fails");
    const uint64_t bogus = 0x1fffff0;  // the highest page number, far above the few this test allocates
    check(api.map(window, 1, &bogus) == nullptr, "mapping a page number that was never allocated fails");

    check(api.map(second, kPages, nullptr) == second, "unmap the second window");
    check(api.free_pages(kPages, pages) == TRUE, "XMemFreePhysicalPages frees the pages");
    check(api.free_pages(1, pages) == FALSE, "freeing a page twice fails");

    // Freed numbers are reused.
    uint64_t again[kPages]{};
    count = kPages;
    check(api.allocate(0x180, &count, again) == TRUE && count == kPages, "pages can be allocated again");
    bool reused = true;
    for (uint64_t i = 0; i < kPages; ++i) {
        bool found = false;
        for (uint64_t j = 0; j < kPages; ++j) found |= again[i] == pages[j];
        reused &= found;
    }
    check(reused, "freed page numbers are reused");
    check(api.map(window, 1, again) == window && read_u32(window, 0) == 0, "a fresh page maps and reads as zero");
    api.map(window, 1, nullptr);
    api.free_pages(kPages, again);

    uint64_t zero_count = 0;
    check(api.allocate(0x280, &zero_count, nullptr) == TRUE, "allocating zero pages succeeds");
    check(api.allocate(0x280, nullptr, pages) == FALSE, "a null count fails");
    // The windows were split into one placeholder per page; the process exit releases them.
}

// rt::xmem::copy_page_mappings_fn, behind console queue slot 31 (CopyPageMappingsBatchX): a destination page maps the
// physical page that is mapped at the source page, so both addresses are one memory, as two XMemMapPhysicalPages calls
// with the same page would make them.
void test_copy_page_mappings(const PhysicalApi& api) {
    const auto copy = rt::xmem::copy_page_mappings_fn;
    check(copy != nullptr, "xmem.cpp sets the copy_page_mappings hook");
    if (!copy) return;
    constexpr uint64_t kPool = 4, kHeap = 8;
    auto* const pool = static_cast<uint8_t*>(api.virtual_alloc(
        nullptr, kPool * kPage64k, MEM_RESERVE | kXmem64kPages, 0x23, PAGE_READWRITE | PAGE_GRAPHICS_READWRITE));
    auto* const heap = static_cast<uint8_t*>(api.virtual_alloc(
        nullptr, kHeap * kPage64k, MEM_RESERVE | kXmem64kPages, 0x23, PAGE_READWRITE | PAGE_GRAPHICS_READWRITE));
    check(pool && heap, "XMemVirtualAlloc reserves a pool window and a heap window");
    if (!pool || !heap) return;
    uint64_t pages[kPool]{};
    uint64_t count = kPool;
    check(api.allocate(0x280, &count, pages) == TRUE && count == kPool, "pages for the pool");
    // Pool pages 0..2 mapped, pool page 3 left unmapped.
    check(api.map(pool, 3, pages) == pool, "the pool maps three pages");
    for (uint64_t i = 0; i < 3; ++i) fill_page(pool + i * kPage64k, 0x7100u + static_cast<uint32_t>(i));

    // A run: heap pages 0 and 1 show pool pages 1 and 2, and are the same memory.
    check(copy(heap, pool + kPage64k, 2, false), "a run of pool pages maps into the heap");
    check(page_holds(heap, 0x7101u) && page_holds(heap + kPage64k, 0x7102u), "the heap pages show pool pages 1 and 2");
    static_cast<volatile uint32_t*>(static_cast<void*>(heap + kPage64k))[1] = 0xc0ffee31;
    check(read_u32(pool + 2 * kPage64k, 4) == 0xc0ffee31, "a write through the heap is seen in the pool: one memory");

    // One page repeated.
    check(copy(heap + 2 * kPage64k, pool, 3, true), "one pool page maps at three heap pages");
    check(page_holds(heap + 2 * kPage64k, 0x7100u) && page_holds(heap + 4 * kPage64k, 0x7100u),
          "every repeated page shows pool page 0");

    // A pool page with nothing mapped leaves its destination unmapped; so does the null source.
    check(copy(heap + 5 * kPage64k, pool + 2 * kPage64k, 2, false), "a run that ends in an unmapped pool page maps");
    check(page_holds(heap + 5 * kPage64k, 0x7102u) && reserved_only(heap + 6 * kPage64k),
          "the unmapped pool page leaves its heap page unmapped");
    check(copy(heap, nullptr, 2, false), "a null source unmaps");
    check(reserved_only(heap) && reserved_only(heap + kPage64k), "the heap pages are reserved again");
    check(page_holds(pool + kPage64k, 0x7101u), "unmapping the heap leaves the pool's own mapping");

    // Overlapping source and destination: every source page is read before any destination page changes.
    check(copy(pool + kPage64k, pool, 2, false), "a run maps over an overlapping range");
    check(page_holds(pool + kPage64k, 0x7100u) && page_holds(pool + 2 * kPage64k, 0x7101u),
          "the overlapping run shows the pages from before it");

    uint8_t outside = 0;
    const auto outside_page =
        reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(&outside) + 0xffff) & ~uintptr_t{0xffff});
    check(!copy(outside_page, pool, 1, false), "a destination outside any window fails");
    SetLastError(ERROR_SUCCESS);
    check(!copy(heap + 1, pool, 1, false) && GetLastError() == ERROR_INVALID_PARAMETER,
          "a destination that is not 64 KiB-aligned fails");
    check(!copy(heap, pool + 1, 1, false) && GetLastError() == ERROR_INVALID_PARAMETER,
          "a source that is not 64 KiB-aligned fails");

    api.map(heap, kHeap, nullptr);
    api.map(pool, kPool, nullptr);
    check(api.free_pages(kPool, pages) == TRUE, "the pool's pages are freed");
}

bool access_faults(volatile uint32_t* address, bool write);

// The same hook on a GPU-only destination: the title reserves a CPU-mappable range and a GPU-only one of the same size
// right after it (title_gpu_views), and the D3D12 layer reads the GPU-only range's memory at the CPU range's address.
// A batch copy onto the GPU-only range maps the same pages at the CPU range too, so that memory becomes readable there.
void test_copy_page_mappings_cpu_view(const PhysicalApi& api) {
    const auto copy = rt::xmem::copy_page_mappings_fn;
    if (!copy) return;  // test_copy_page_mappings reports it
    constexpr uint64_t kPool = 2, kPair = 3;
    auto* const pool = static_cast<uint8_t*>(api.virtual_alloc(
        nullptr, kPool * kPage64k, MEM_RESERVE | kXmem64kPages, 0x23, PAGE_READWRITE | PAGE_GRAPHICS_READWRITE));
    auto* const other = static_cast<uint8_t*>(api.virtual_alloc(
        nullptr, kPage64k, MEM_RESERVE | kXmem64kPages, 0x23, PAGE_READWRITE | PAGE_GRAPHICS_READWRITE));
    // The pair: the CPU-mappable range, then the GPU-only one (PAGE_GRAPHICS_NOACCESS | PAGE_NOACCESS) of its size.
    auto* const cpu = static_cast<uint8_t*>(api.virtual_alloc(
        nullptr, kPair * kPage64k, MEM_RESERVE | kXmem64kPages, 0x23, PAGE_READWRITE | PAGE_GRAPHICS_READWRITE));
    auto* const gpu = static_cast<uint8_t*>(
        api.virtual_alloc(nullptr, kPair * kPage64k, MEM_RESERVE | kXmem64kPages, 0x3, 0x801));
    check(pool && other && cpu && gpu, "XMemVirtualAlloc reserves a pool, a window and a CPU + GPU-only pair");
    if (!pool || !other || !cpu || !gpu) return;
    check(leonida::title_gpu_views::to_cpu(reinterpret_cast<uintptr_t>(gpu)) == reinterpret_cast<uintptr_t>(cpu),
          "the GPU-only range reads at the CPU range reserved just before it");
    uint64_t pages[kPool]{};
    uint64_t count = kPool;
    check(api.allocate(0x280, &count, pages) == TRUE && count == kPool, "pages for the pool");
    check(api.map(pool, kPool, pages) == pool, "the pool maps its pages");
    for (uint64_t i = 0; i < kPool; ++i) fill_page(pool + i * kPage64k, 0x7300u + static_cast<uint32_t>(i));

    // A run onto the GPU-only range: its CPU range shows the same bytes. Each page is read under a guard first, so a
    // page left reserved fails its check instead of faulting the test (Wine's VirtualQuery does not report a view
    // mapped into a placeholder as committed, so the state is not asked for).
    const auto readable = [](const uint8_t* page) {
        return !access_faults(reinterpret_cast<volatile uint32_t*>(const_cast<uint8_t*>(page)), false);
    };
    check(copy(gpu, pool, kPool, false), "a run of pool pages maps onto the GPU-only range");
    check(readable(gpu) && page_holds(gpu, 0x7300u), "the GPU-only range shows the pool pages");
    check(readable(cpu) && readable(cpu + kPage64k), "the paired CPU range is readable after the copy");
    check(readable(cpu) && readable(cpu + kPage64k) && page_holds(cpu, 0x7300u) && page_holds(cpu + kPage64k, 0x7301u),
          "the paired CPU range shows the same bytes as the pool");
    check(!readable(cpu + 2 * kPage64k), "a CPU page past the copied run stays unmapped");
    if (readable(cpu + kPage64k)) {
        static_cast<volatile uint32_t*>(static_cast<void*>(cpu + kPage64k))[1] = 0xc0ffee32;
        check(read_u32(pool + kPage64k, 4) == 0xc0ffee32 && read_u32(gpu + kPage64k, 4) == 0xc0ffee32,
              "a write at the CPU range is seen in the pool and the GPU-only range: one memory");
    }

    // A GPU-only source whose page is mapped only at its CPU range is found there.
    check(api.map(cpu + 2 * kPage64k, 1, &pages[0]) == cpu + 2 * kPage64k, "a page maps at the CPU range alone");
    check(copy(other, gpu + 2 * kPage64k, 1, false), "a copy from the GPU-only range maps");
    check(readable(other) && page_holds(other, 0x7300u),
          "a GPU-only source page unmapped there is looked up at its CPU range");

    // A null source unmaps both ranges.
    check(copy(gpu, nullptr, kPool, false), "a null source unmaps the GPU-only range");
    check(!readable(gpu) && !readable(cpu) && !readable(cpu + kPage64k), "the paired CPU range is unmapped with it");

    api.map(other, 1, nullptr);
    api.map(cpu, kPair, nullptr);
    api.map(pool, kPool, nullptr);
    check(api.free_pages(kPool, pages) == TRUE, "the pool's pages are freed");
}

bool access_faults(volatile uint32_t* address, bool write) {
    __try {
        if (write) {
            *address = 1;
        } else {
            const uint32_t value = *address;
            (void)value;
        }
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER
                                                                  : EXCEPTION_CONTINUE_SEARCH) {
        return true;
    }
    return false;
}

int64_t commit_growth(const PERFORMANCE_INFORMATION& from, const PERFORMANCE_INFORMATION& to) {
    return (static_cast<int64_t>(to.CommitTotal) - static_cast<int64_t>(from.CommitTotal)) *
           static_cast<int64_t>(to.PageSize);
}

void test_physical_pages_commit_on_touch(const PhysicalApi& api) {
    PERFORMANCE_INFORMATION before{};
    if (!GetPerformanceInfo(&before, sizeof(before))) {
        check(false, "GetPerformanceInfo reads the system commit charge");
        return;
    }
    const uint64_t free_commit = static_cast<uint64_t>(before.CommitLimit - before.CommitTotal) * before.PageSize;
    const uint64_t count = (free_commit + (uint64_t{1} << 30)) / kPage64k;
    if (count >= (uint64_t{1} << 20)) {
        std::printf("SKIP physical pages past the free commit: %llu MiB of commit is free\n",
                    static_cast<unsigned long long>(free_commit >> 20));
        return;
    }
    std::vector<uint64_t> pages(static_cast<size_t>(count));
    uint64_t got = count;
    check(api.allocate(0x380, &got, pages.data()) == TRUE && got == count,
          "XMemAllocatePhysicalPages gives 1 GiB more pages than the system has commit free");
    if (got != count) return;
    PERFORMANCE_INFORMATION allocated{};
    GetPerformanceInfo(&allocated, sizeof(allocated));
    check(commit_growth(before, allocated) < static_cast<int64_t>(count * kPage64k / 8),
          "allocating those pages does not charge them");

    constexpr uint64_t kMapped = 1024;
    constexpr uint64_t kTouched = 512;
    auto* const window = static_cast<uint8_t*>(
        api.virtual_alloc(nullptr, kMapped * kPage64k, MEM_RESERVE | kXmem64kPages, 0x21, PAGE_READWRITE));
    auto* const second = static_cast<uint8_t*>(
        api.virtual_alloc(nullptr, 2 * kPage64k, MEM_RESERVE | kXmem64kPages, 0x21, PAGE_READWRITE));
    check(window != nullptr && second != nullptr, "XMemVirtualAlloc reserves windows for them");
    if (!window || !second) {
        api.free_pages(count, pages.data());
        return;
    }
    check(api.map(window, kMapped, pages.data()) == window, "1,024 of them map into one window");
    PERFORMANCE_INFORMATION mapped{};
    GetPerformanceInfo(&mapped, sizeof(mapped));
    check(commit_growth(allocated, mapped) < static_cast<int64_t>(kMapped * kPage64k / 2),
          "mapping them does not charge them");

    for (uint64_t i = 0; i < kTouched; ++i) {
        static_cast<volatile uint32_t*>(static_cast<void*>(window + i * kPage64k))[0] =
            0xc0de0000u | static_cast<uint32_t>(i);
    }
    PERFORMANCE_INFORMATION touched{};
    GetPerformanceInfo(&touched, sizeof(touched));
    const int64_t touched_growth = commit_growth(mapped, touched);
    check(touched_growth >= static_cast<int64_t>(kTouched * kPage64k / 2) &&
              touched_growth < static_cast<int64_t>(kMapped * kPage64k),
          "writing to 512 of them charges about those 512 pages");
    bool written = true;
    for (uint64_t i = 0; i < kTouched; ++i) written &= read_u32(window + i * kPage64k, 0) == (0xc0de0000u | i);
    check(written, "each touched page keeps what was written to it");
    check(read_u32(window + kTouched * kPage64k, 0) == 0, "an untouched page reads as zero");

    check(api.map(second, 2, pages.data()) == second, "two of the pages map into a second window");
    check(read_u32(second, 0) == 0xc0de0000u && read_u32(second + kPage64k, 0) == 0xc0de0001u,
          "the second window sees what was written through the first");
    static_cast<volatile uint32_t*>(static_cast<void*>(second))[1] = 0x5ca1ab1e;
    check(read_u32(window, 4) == 0x5ca1ab1e, "and the first window sees what is written through the second");

    DWORD old = 0;
    check(VirtualProtect(window, kPage64k, PAGE_READONLY, &old) != FALSE &&
              access_faults(static_cast<volatile uint32_t*>(static_cast<void*>(window)), true),
          "a write to a committed read-only page still faults to the next handler");
    VirtualProtect(window, kPage64k, PAGE_READWRITE, &old);
    check(VirtualProtect(window + 4096, 4096, PAGE_READONLY, &old) != FALSE &&
              access_faults(static_cast<volatile uint32_t*>(static_cast<void*>(window + 4096)), true),
          "a write to a read-only 4 KiB part of a read-write page still faults to the next handler");
    check(!access_faults(static_cast<volatile uint32_t*>(static_cast<void*>(window + 8192)), true),
          "the rest of that page stays writable");
    VirtualProtect(window + 4096, 4096, PAGE_READWRITE, &old);
    void* const elsewhere = VirtualAlloc(nullptr, kPage64k, MEM_RESERVE, PAGE_READWRITE);
    check(elsewhere != nullptr && access_faults(static_cast<volatile uint32_t*>(elsewhere), false),
          "a fault outside the mapped pages still reaches the next handler");
    if (elsewhere) VirtualFree(elsewhere, 0, MEM_RELEASE);

    uint8_t* const target = window + (kTouched + 1) * kPage64k;
    check(rt::xmem::commit_physical_range(target, kPage64k), "commit_physical_range commits an untouched page");
    uint32_t local = 0;
    check(rt::xmem::commit_physical_range(&local, sizeof(local)), "commit_physical_range skips memory it does not own");
    wchar_t directory[MAX_PATH]{};
    wchar_t path[MAX_PATH]{};
    bool read_ok = false;
    if (GetTempPathW(MAX_PATH, directory) && GetTempFileNameW(directory, L"xpg", 0, path)) {
        std::vector<uint32_t> pattern(kPage64k / sizeof(uint32_t));
        for (size_t i = 0; i < pattern.size(); ++i) pattern[i] = 0x7e570000u + static_cast<uint32_t>(i);
        const HANDLE out =
            CreateFileW(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, nullptr);
        DWORD bytes = 0;
        const bool wrote = out != INVALID_HANDLE_VALUE &&
                           WriteFile(out, pattern.data(), static_cast<DWORD>(kPage64k), &bytes, nullptr) &&
                           bytes == kPage64k;
        if (out != INVALID_HANDLE_VALUE) CloseHandle(out);
        const HANDLE in = wrote ? CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr)
                                : INVALID_HANDLE_VALUE;
        if (in != INVALID_HANDLE_VALUE) {
            read_ok = ReadFile(in, target, static_cast<DWORD>(kPage64k), &bytes, nullptr) && bytes == kPage64k &&
                      std::memcmp(target, pattern.data(), kPage64k) == 0;
            CloseHandle(in);
        }
        DeleteFileW(path);
    }
    check(read_ok, "ReadFile fills a page that commit_physical_range committed");

    check(api.map(second, 2, nullptr) == second && api.map(window, kMapped, nullptr) == window, "they unmap");
    check(api.free_pages(count, pages.data()) == TRUE, "XMemFreePhysicalPages frees them");
}

void test_commit_physical_pages(const PhysicalApi& api) {
    constexpr uint64_t kPages = 3;
    uint64_t pages[kPages]{};
    uint64_t count = kPages;
    check(api.allocate(0x280, &count, pages) == TRUE && count == kPages, "three physical pages for commit by number");
    if (count != kPages) return;
    check(rt::xmem::commit_physical_pages(pages, 0), "committing no pages succeeds");
    check(!rt::xmem::commit_physical_pages(nullptr, 1), "committing from a null page array fails");
    const uint64_t unknown = 0x1fffff0;
    check(!rt::xmem::commit_physical_pages(&unknown, 1), "committing a page number that was never allocated fails");
    check(rt::xmem::commit_physical_pages(pages, 2), "commit_physical_pages commits two pages that were never mapped");

    auto* const window = static_cast<uint8_t*>(
        api.virtual_alloc(nullptr, kPages * kPage64k, MEM_RESERVE | kXmem64kPages, 0x21, PAGE_READWRITE));
    check(window != nullptr, "XMemVirtualAlloc reserves a window for them");
    if (window) {
        check(api.map(window, kPages, pages) == window, "the three pages map");
        MEMORY_BASIC_INFORMATION info[kPages]{};
        bool queried = true;
        for (uint64_t i = 0; i < kPages; ++i)
            queried &= VirtualQuery(window + i * kPage64k, &info[i], sizeof(info[i])) == sizeof(info[i]);
        check(queried && info[0].State == MEM_COMMIT && info[1].State == MEM_COMMIT && info[2].State == MEM_RESERVE,
              "the two committed by number are committed in the window and the third is not yet");
        check(read_u32(window, 0) == 0 && read_u32(window + kPage64k, 0) == 0,
              "pages committed by number read as zero");
        api.map(window, kPages, nullptr);
    }
    check(api.free_pages(kPages, pages) == TRUE, "XMemFreePhysicalPages frees them");
}

void test_first_touch_during_remap(const PhysicalApi& api) {
    constexpr uint64_t kSlice = 16;
    constexpr uint64_t kSlices = 256;
    std::vector<uint64_t> pages(kSlice * kSlices);
    uint64_t count = pages.size();
    check(api.allocate(0x280, &count, pages.data()) == TRUE && count == pages.size(),
          "4,096 physical pages for the remap race");
    if (count != pages.size()) return;
    auto* const window = static_cast<uint8_t*>(
        api.virtual_alloc(nullptr, kSlice * kPage64k, MEM_RESERVE | kXmem64kPages, 0x21, PAGE_READWRITE));
    check(window != nullptr, "XMemVirtualAlloc reserves a window for the remap race");
    if (!window) {
        api.free_pages(count, pages.data());
        return;
    }
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> touches{0};
    std::atomic<uint64_t> unmapped_faults{0};
    std::atomic<int> started_writers{0};
    std::vector<std::thread> writers;
    for (int t = 0; t < 4; ++t) {
        writers.emplace_back([&, t] {
            started_writers.fetch_add(1);
            uint64_t i = static_cast<uint64_t>(t);
            while (!stop.load(std::memory_order_relaxed)) {
                auto* const word =
                    static_cast<volatile uint32_t*>(static_cast<void*>(window + (i % kSlice) * kPage64k)) + t;
                if (access_faults(word, true)) {
                    unmapped_faults.fetch_add(1, std::memory_order_relaxed);
                } else {
                    touches.fetch_add(1, std::memory_order_relaxed);
                }
                ++i;
            }
        });
    }
    // The remaps can finish before a writer is scheduled on a fast host, so wait for all four to run first.
    while (started_writers.load() < 4) Sleep(0);
    const ULONGLONG started = GetTickCount64();
    bool remapped = true;
    for (uint64_t s = 0; s < kSlices; ++s) {
        remapped &= api.map(window, kSlice, pages.data() + s * kSlice) == window;
        Sleep(0);
        remapped &= api.map(window, kSlice, nullptr) == window;
    }
    remapped &= api.map(window, kSlice, pages.data() + (kSlices - 1) * kSlice) == window;
    // The last slice stays mapped: give the writers up to a second to land a write in it before stopping them.
    for (const ULONGLONG until = GetTickCount64() + 1000; touches.load() == 0 && GetTickCount64() < until;) Sleep(1);
    stop.store(true);
    for (auto& writer : writers) writer.join();
    const ULONGLONG elapsed = GetTickCount64() - started;
    check(remapped, "256 slices map and unmap while 4 threads write into the window");
    check(elapsed < 30000, "first-touch faults during map and unmap finish without hanging");
    check(touches.load() > 0, "writes land while the pages are mapped");
    std::printf("remap race: %llu writes landed, %llu hit an unmapped page, %llu ms\n",
                static_cast<unsigned long long>(touches.load()),
                static_cast<unsigned long long>(unmapped_faults.load()), static_cast<unsigned long long>(elapsed));
    static_cast<volatile uint32_t*>(static_cast<void*>(window))[8] = 0x600dcafe;
    check(read_u32(window, 32) == 0x600dcafe, "the last slice is still mapped and writable");
    api.map(window, kSlice, nullptr);
    check(api.free_pages(count, pages.data()) == TRUE, "XMemFreePhysicalPages frees the remap race pages");
}

// Splitting a view briefly unmaps the view's other pages; a thread reading one of them meanwhile must not fault.
// Unmapping page k of the view [k, end) splits it every time, and the last page of each view is the one being read.
void test_split_under_reader(const PhysicalApi& api) {
    constexpr uint64_t kPages = 2048;  // two capped views
    auto* const window = static_cast<uint8_t*>(
        api.virtual_alloc(nullptr, kPages * kPage64k, MEM_RESERVE | kXmem64kPages, 0x23, PAGE_READWRITE));
    std::vector<uint64_t> pages(kPages);
    uint64_t count = kPages;
    check(window && api.allocate(0x280, &count, pages.data()) == TRUE && count == kPages &&
              api.map(window, kPages, pages.data()) == window,
          "a 2,048-page run maps for the split test");
    if (!window || count != kPages) return;
    uint8_t* const read_pages[2] = {window + 1023 * kPage64k, window + 2047 * kPage64k};
    fill_page(read_pages[0], 0x11000u);
    fill_page(read_pages[1], 0x11000u);
    std::atomic<bool> done{false};
    std::atomic<uint64_t> reads{0}, wrong{0};
    std::thread reader([&] {
        for (uint64_t i = 0; !done.load(std::memory_order_relaxed); ++i) {
            wrong += read_u32(read_pages[i & 1], 0) != 0x11000u;
            ++reads;
        }
    });
    bool unmapped = true;
    for (uint64_t k = 0; k < kPages - 1; ++k) {
        if (k == 1023) continue;
        void* const page = window + k * kPage64k;
        unmapped &= api.scatter(&page, 1, nullptr) == TRUE;
    }
    done = true;
    reader.join();
    check(unmapped, "2,046 pages unmap one at a time, each splitting the view a reader is in");
    check(reads.load() > 0 && wrong.load() == 0, "the reader of the views' last pages never faults or misreads");
    // The last split was of [2046, 2048): a fault there is retried, one on another mapped page is not.
    api.map(window, kPages, nullptr);
    api.free_pages(kPages, pages.data());
}

// A run longer than one host view is mapped as capped views, so unmapping one page inside it only remaps the rest of
// that view; every other page keeps its contents.
void test_long_run(const PhysicalApi& api) {
    constexpr uint64_t kPages = 1100;  // more than the 1024-page view cap
    auto* const window = static_cast<uint8_t*>(
        api.virtual_alloc(nullptr, kPages * kPage64k, MEM_RESERVE | kXmem64kPages, 0x23, PAGE_READWRITE));
    std::vector<uint64_t> pages(kPages);
    uint64_t count = kPages;
    check(window && api.allocate(0x280, &count, pages.data()) == TRUE, "a long run's window and pages are allocated");
    if (!window || count != kPages) return;
    check(api.map(window, kPages, pages.data()) == window, "a run longer than the view cap maps");
    // The pages are committed on first touch, so an untouched view reads MEM_RESERVE on Windows (Wine reports
    // MEM_COMMIT); either way the first region is one mapped view of exactly the cap.
    MEMORY_BASIC_INFORMATION info{};
    VirtualQuery(window, &info, sizeof(info));
    check(info.Type == MEM_MAPPED && (info.State == MEM_COMMIT || info.State == MEM_RESERVE) &&
              info.RegionSize == 1024 * kPage64k,
          "it is mapped as views of at most 1024 pages");
    for (uint64_t i = 0; i < kPages; ++i) fill_page(window + i * kPage64k, static_cast<uint32_t>(i + 1));
    void* const hole = window + 10 * kPage64k;
    check(api.scatter(&hole, 1, nullptr) == TRUE && reserved_only(hole), "a page inside the first view unmaps");
    bool kept = true;
    for (uint64_t i = 0; i < kPages; ++i) {
        if (i != 10) kept &= page_holds(window + i * kPage64k, static_cast<uint32_t>(i + 1));
    }
    check(kept, "every other page of the run keeps its contents");
    // Several pages across the boundary of two capped views unmap together; the pages on both sides stay.
    check(api.map(window + 1000 * kPage64k, 50, nullptr) == window + 1000 * kPage64k &&
              reserved_only(window + 1000 * kPage64k) && reserved_only(window + 1049 * kPage64k),
          "a multi-page range across two views unmaps");
    check(page_holds(window + 999 * kPage64k, 1000u) && page_holds(window + 1050 * kPage64k, 1051u),
          "the pages on both sides of it keep their contents");
    // Mapping other pages over the middle of a live run replaces just those pages.
    check(api.map(window + 500 * kPage64k, 3, &pages[600]) == window + 500 * kPage64k &&
              page_holds(window + 500 * kPage64k, 601u) && page_holds(window + 502 * kPage64k, 603u),
          "pages mapped over the middle of a live run replace it there");
    check(page_holds(window + 499 * kPage64k, 500u) && page_holds(window + 503 * kPage64k, 504u),
          "and the run's pages around them keep their contents");
    check(api.map(window, kPages, nullptr) == window && reserved_only(window) &&
              reserved_only(window + (kPages - 1) * kPage64k),
          "the whole run unmaps");
    check(api.free_pages(kPages, pages.data()) == TRUE, "the long run's pages are freed");
    check(api.map(window, 1, pages.data()) == nullptr && reserved_only(window), "a freed page does not map again");
    // Pages freed while still mapped keep their memory, and splitting their view later still maps the rest again.
    count = 4;
    check(api.allocate(0x280, &count, pages.data()) == TRUE && count == 4 && api.map(window, 4, pages.data()) == window,
          "four pages map as one view");
    if (count != 4) return;
    for (uint64_t i = 0; i < 4; ++i) fill_page(window + i * kPage64k, static_cast<uint32_t>(i + 1));
    check(api.free_pages(4, pages.data()) == TRUE, "they are freed while mapped");
    void* const second = window + kPage64k;
    check(api.scatter(&second, 1, nullptr) == TRUE && in_view(window) && !in_view(second) &&
              in_view(window + 2 * kPage64k) && page_holds(window, 1u) &&
              page_holds(window + 2 * kPage64k, 3u) && page_holds(window + 3 * kPage64k, 4u),
          "unmapping one of them afterwards keeps the others' memory");
    api.map(window, 4, nullptr);
}

// The placeholder flags pass through: reserve a placeholder, carve windows and committed memory out of it.
void test_placeholders(const PhysicalApi& api) {
    auto* const base = static_cast<uint8_t*>(
        api.virtual_alloc(nullptr, 6 * kPage64k, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, 7, PAGE_NOACCESS));
    check(base != nullptr, "XMemVirtualAlloc reserves a MEM_RESERVE_PLACEHOLDER region");
    if (!base) return;

    void* const window = api.virtual_alloc(base + kPage64k, 2 * kPage64k,
                                           MEM_RESERVE | MEM_REPLACE_PLACEHOLDER | kXmem64kPages, 0x27,
                                           PAGE_READWRITE | PAGE_GRAPHICS_READONLY);
    check(window == base + kPage64k, "a reserve-only MEM_REPLACE_PLACEHOLDER returns the address it replaced");

    auto* const committed = static_cast<uint8_t*>(
        api.virtual_alloc(base + 4 * kPage64k, kPage64k,
                          MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | kXmem64kPages, 0x27, PAGE_READWRITE));
    check(committed == base + 4 * kPage64k, "a committing MEM_REPLACE_PLACEHOLDER returns the address it replaced");
    if (committed) {
        committed[0] = 0x5a;
        check(committed[0] == 0x5a, "the committed replacement is writable");
    }

    uint64_t pages[2]{};
    uint64_t count = 2;
    check(api.allocate(0x380, &count, pages) == TRUE && count == 2, "allocate two pages for the carved window");
    check(api.map(window, 2, pages) == window, "pages map into a window carved out of a placeholder");
    if (window) {
        fill_page(static_cast<uint8_t*>(window) + kPage64k, 0xabc0000u);
        check(page_holds(static_cast<uint8_t*>(window) + kPage64k, 0xabc0000u), "the carved window holds data");
    }
    check(api.map(window, 2, nullptr) == window, "unmap the carved window");
    check(api.free_pages(2, pages) == TRUE, "free the carved window's pages");
    check(reserved_only(base), "the rest of the placeholder is still reserved");
}

// Three of the retail executable's five physical-page windows carry 0x400 next to PAGE_READWRITE. That is
// PAGE_WRITECOMBINE on the PC, outside the PAGE_GRAPHICS_* range, and MapViewOfFile3 refuses a view with it, so the
// mapped pages take the window's protection without the cache modifier.
void test_write_combine_window(const PhysicalApi& api) {
    auto* const window = static_cast<uint8_t*>(
        api.virtual_alloc(nullptr, kPage64k, MEM_RESERVE | kXmem64kPages, 0x27,
                          PAGE_READWRITE | PAGE_WRITECOMBINE | PAGE_GRAPHICS_READWRITE));
    check(window != nullptr, "XMemVirtualAlloc reserves a window with PAGE_WRITECOMBINE in its protection");
    if (!window) return;

    uint64_t page = 0;
    uint64_t count = 1;
    check(api.allocate(0x280, &count, &page) == TRUE && count == 1, "allocate a page for the write-combine window");
    SetLastError(ERROR_SUCCESS);
    const bool mapped = api.map(window, 1, &page) == window;
    check(mapped, "a page maps into a window reserved with PAGE_WRITECOMBINE");
    if (mapped) {
        // Physical pages are committed on first touch, so the view is written before it is queried.
        *static_cast<volatile unsigned char*>(window) = 1;
        MEMORY_BASIC_INFORMATION info{};
        VirtualQuery(window, &info, sizeof(info));
        if (touched_view_reported_reserved(info))
            std::printf("SKIP the write-combine window page's reported state and protection: this host's "
                        "VirtualQuery shows a written view page as reserved (Wine)\n");
        else
            check(info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE,
                  "the page mapped there is a plain read-write view");
        fill_page(window, 0x77000000u);
        check(page_holds(window, 0x77000000u), "the page in the write-combine window holds data");
    }
    check(api.map(window, 1, nullptr) == window, "unmap the write-combine window");
    check(api.free_pages(1, &page) == TRUE, "free the write-combine window's page");
}

// Counts the access violations in [g_counted_begin, g_counted_end). Added after xmem's handler with First = 1, so it
// runs before it and sees every fault there, the ones xmem then commits included.
std::atomic<uintptr_t> g_counted_begin{0}, g_counted_end{0};
std::atomic<uint64_t> g_counted_faults{0};

LONG CALLBACK count_window_faults(EXCEPTION_POINTERS* info) {
    const EXCEPTION_RECORD* record = info ? info->ExceptionRecord : nullptr;
    if (record && record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && record->NumberParameters >= 2) {
        const auto at = static_cast<uintptr_t>(record->ExceptionInformation[1]);
        if (at >= g_counted_begin.load(std::memory_order_relaxed) && at < g_counted_end.load(std::memory_order_relaxed))
            g_counted_faults.fetch_add(1, std::memory_order_relaxed);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

// About `us` microseconds of spinning, without a call that could allocate.
void spin_for(LONGLONG us) {
    LARGE_INTEGER now{}, frequency{};
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&now);
    const LONGLONG until = now.QuadPart + frequency.QuadPart * us / 1000000;
    while (QueryPerformanceCounter(&now) && now.QuadPart < until) YieldProcessor();
}

// Title threads take access violations while they hold the process heap's lock: XMemAlloc is _aligned_malloc, so
// HeapAlloc and HeapFree write block headers on title pages, which may be physical pages waiting for their first touch
// or pages the D3D12 layer's shadows guard. xmem's handler waits for g_fault_lock (shared) at every access violation,
// its own or not, so whoever holds that lock exclusively (a map or an unmap of physical pages) must never wait for the
// process heap meanwhile: it would wait for the faulting thread, which waits for it. While
//   - threads take the process heap's lock (HeapLock, as HeapAlloc does inside, and the heap gate above) and, holding
//     it, _aligned_malloc and _aligned_free and write a word into every page of a window whose pages are mapped fresh
//     all the time (a first-touch commit, or a fault on a page that is being unmapped) and into a read-only page (a
//     fault that isn't xmem's, as on a guarded shadow page);
//   - threads allocate physical pages, map them into their part of the window (as a run, or page by page through the
//     scatter call), and unmap and free them, some freed while still mapped so that the unmap drops the last
//     reference to their section;
// a watchdog ends the process (exit code 3) when nothing got anywhere for 30 s: a deadlock.
void test_heap_lock_faults(const PhysicalApi& api) {
    if (const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll"); ntdll && GetProcAddress(ntdll, "wine_get_version")) {
        // Under Wine a holder of the fault lock still reaches the process heap through the host's own view calls (or
        // Wine's fault dispatch needs it): the test deadlocks there with the fix in. The Wine side is an open item.
        std::printf("SKIP heap-lock faults under Wine: a fault-lock holder still waits for the process heap there\n");
        return;
    }
    constexpr uint64_t kMappers = 2, kSlice = 4, kPages = kMappers * kSlice;
    constexpr unsigned kTouchers = 3;
    auto* const window = static_cast<uint8_t*>(
        api.virtual_alloc(nullptr, kPages * kPage64k, MEM_RESERVE | kXmem64kPages, 0x21, PAGE_READWRITE));
    auto* const read_only = static_cast<uint8_t*>(VirtualAlloc(nullptr, 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_READONLY));
    check(window != nullptr && read_only != nullptr, "a window and a read-only page for the heap-lock faults");
    if (!window || !read_only) return;
    g_counted_begin = reinterpret_cast<uintptr_t>(window);
    g_counted_end = reinterpret_cast<uintptr_t>(window) + kPages * kPage64k;
    g_counted_faults = 0;
    // xmem's handler is in by now (the first XMemAllocatePhysicalPages put it in), so this one runs first.
    void* const counter = AddVectoredExceptionHandler(1, &count_window_faults);
    check(counter != nullptr, "a fault counter in front of xmem's handler");
    if (!counter) return;

    std::atomic<uint64_t> progress{0};  // bumped by every round of every thread
    std::atomic<bool> stop{false}, finished{false}, failed{false};
    std::thread watchdog([&] {
        uint64_t seen = progress.load();
        ULONGLONG since = GetTickCount64();
        while (!finished.load()) {
            Sleep(100);
            const uint64_t now = progress.load();
            if (now != seen) {
                seen = now;
                since = GetTickCount64();
            } else if (GetTickCount64() - since > 30000) {
                // No heap here: the process heap may be what is stuck.
                static const char message[] =
                    "xmem heap-lock faults: no progress for 30 s: deadlock (a holder of the lock xmem's fault handler "
                    "waits for is waiting for the process heap)\n";
                DWORD wrote = 0;
                WriteFile(GetStdHandle(STD_ERROR_HANDLE), message, DWORD(sizeof(message) - 1), &wrote, nullptr);
                TerminateProcess(GetCurrentProcess(), 3);
            }
        }
    });
    std::atomic<uint64_t> rounds{0}, missed{0}, read_only_faults{0}, maps{0};
    std::vector<std::thread> threads;
    threads.reserve(kTouchers + kMappers);
    g_heap_gate_on = true;
    for (unsigned t = 0; t < kTouchers; ++t)
        threads.emplace_back([&, t] {
            const HANDLE heap = GetProcessHeap();
            for (unsigned round = 0; !stop.load(std::memory_order_relaxed); ++round) {
                uint64_t round_missed = 0;
                bool round_read_only = false;
                AcquireSRWLockExclusive(&g_heap_gate);
                HeapLock(heap);
                void* block = _aligned_malloc(64 + (round % 32) * 160, 16);
                for (uint64_t p = 0; p < kPages; ++p) {
                    auto* const word =
                        static_cast<volatile uint32_t*>(static_cast<void*>(window + p * kPage64k)) + t;
                    round_missed += access_faults(word, true);
                }
                round_read_only = access_faults(static_cast<volatile uint32_t*>(static_cast<void*>(read_only)), true);
                _aligned_free(block);
                HeapUnlock(heap);
                ReleaseSRWLockExclusive(&g_heap_gate);
                missed.fetch_add(round_missed, std::memory_order_relaxed);
                read_only_faults.fetch_add(round_read_only, std::memory_order_relaxed);
                rounds.fetch_add(1, std::memory_order_relaxed);
                progress.fetch_add(1, std::memory_order_relaxed);
                spin_for(50);  // the other threads get to allocate
            }
        });
    for (uint64_t m = 0; m < kMappers; ++m)
        threads.emplace_back([&, m] {
            uint8_t* const slice = window + m * kSlice * kPage64k;
            uint64_t pages[kSlice]{};
            void* addresses[kSlice]{};
            for (uint64_t k = 0; k < kSlice; ++k) addresses[k] = slice + k * kPage64k;
            for (uint64_t round = 0; !stop.load(std::memory_order_relaxed); ++round) {
                uint64_t count = kSlice;
                if (api.allocate(0x280, &count, pages) != TRUE || count != kSlice) {
                    failed = true;
                    break;
                }
                bool ok = round % 2 == 0 ? api.map(slice, kSlice, pages) == slice
                                         : api.scatter(addresses, static_cast<uint32_t>(kSlice), pages) == TRUE;
                spin_for(100);  // the touchers get to fault the fresh pages in
                // Freed while mapped: the unmap then drops the last reference to the section.
                const bool free_first = round % 3 == 0;
                if (free_first) ok &= api.free_pages(kSlice, pages) == TRUE;
                ok &= api.map(slice, kSlice, nullptr) == slice;
                if (!free_first) ok &= api.free_pages(kSlice, pages) == TRUE;
                if (!ok) {
                    failed = true;
                    break;
                }
                maps.fetch_add(1, std::memory_order_relaxed);
                progress.fetch_add(1, std::memory_order_relaxed);
            }
        });
    for (const ULONGLONG until = GetTickCount64() + 3000; !failed.load() && GetTickCount64() < until;) Sleep(10);
    stop = true;
    for (std::thread& thread : threads) thread.join();
    g_heap_gate_on = false;
    finished = true;
    watchdog.join();
    RemoveVectoredExceptionHandler(counter);
    g_counted_end = 0;

    const uint64_t window_faults = g_counted_faults.load();
    // A fault in the window that the toucher's own handler didn't catch is one xmem's handler resolved: a first touch
    // (or a retry of one another toucher committed meanwhile).
    const uint64_t committed = window_faults > missed.load() ? window_faults - missed.load() : 0;
    check(!failed.load(), "physical pages allocate, map (run and scatter), unmap and free while faults hold the heap");
    check(maps.load() > 0 && rounds.load() > 0, "the mappers and the faulting threads both made progress");
    check(committed > 0, "threads holding the process heap's lock took first-touch faults xmem committed");
    check(read_only_faults.load() == rounds.load(), "every write to the read-only page faulted through xmem's handler");
    std::printf("heap-lock faults: %llu mappings, %llu rounds holding the process heap's lock, %llu first touches "
                "committed, %llu faults on unmapped pages: no deadlock\n",
                static_cast<unsigned long long>(maps.load()), static_cast<unsigned long long>(rounds.load()),
                static_cast<unsigned long long>(committed), static_cast<unsigned long long>(missed.load()));
    VirtualFree(read_only, 0, MEM_RELEASE);
}

// gta6rt --trace: one "xmem_attributes" event per distinct raw attribute word and API, none with the trace off.
void test_attribute_trace(XMemAllocFn alloc, XMemFreeFn free_mem, XMemVirtualAllocFn virtual_alloc) {
    wchar_t temp[MAX_PATH];
    GetTempPathW(MAX_PATH, temp);
    const std::wstring path =
        std::wstring(temp) + L"xmem_trace_test_" + std::to_wstring(GetCurrentProcessId()) + L".jsonl";
    DeleteFileW(path.c_str());

    void* untraced = alloc(64, 0x1111);  // trace not open yet: nothing to write, nothing to crash
    check(untraced != nullptr, "XMemAlloc with the trace off still allocates");
    free_mem(untraced, 0x1111);

    check(rt::startup_trace::open(path.c_str()), "open the startup trace file");
    void* a = alloc(4096, 0x5e50000);
    void* b = alloc(8192, 0x5e50000);  // same word again: no second event
    void* c = alloc(16, 0x99);
    free_mem(a, 0x5e50000);
    free_mem(b, 0x5e50000);  // same word again: no second event
    free_mem(c, 0x99);
    free_mem(nullptr, 0x77);  // a free-only word is recorded too
    void* const v = virtual_alloc ? virtual_alloc(nullptr, 0x10000, MEM_RESERVE | MEM_COMMIT, 0x183, PAGE_READWRITE)
                                  : nullptr;
    if (v) VirtualFree(v, 0, MEM_RELEASE);
    rt::startup_trace::close();

    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    const std::string log = text.str();
    auto count = [&](const std::string& needle) {
        int n = 0;
        for (size_t at = log.find(needle); at != std::string::npos; at = log.find(needle, at + 1)) ++n;
        return n;
    };
    check(count("\"kind\":\"xmem_attributes\"") == 6, "six events: two XMemAlloc words, three XMemFree words and one XMemVirtualAlloc word");
    check(count("\"api\":\"XMemVirtualAlloc\",\"attributes\":\"0x183\",\"size\":65536") == 1,
          "XMemVirtualAlloc 0x183 is recorded once, with its size");
    check(count("\"api\":\"XMemAlloc\",\"attributes\":\"0x5e50000\",\"size\":4096") == 1,
          "XMemAlloc 0x5e50000 is recorded once, with the size of the first call");
    check(count("\"api\":\"XMemAlloc\",\"attributes\":\"0x99\",\"size\":16") == 1, "XMemAlloc 0x99 is recorded once");
    check(count("\"api\":\"XMemFree\",\"attributes\":\"0x5e50000\"") == 1, "XMemFree 0x5e50000 is recorded once");
    check(count("\"api\":\"XMemFree\",\"attributes\":\"0x77\"") == 1, "XMemFree 0x77 is recorded");
    check(count("\"attributes\":\"0x1111\"") == 0, "nothing was recorded while the trace was off");
    check(count("\"caller\":\"") == 6, "every event names its caller");
    DeleteFileW(path.c_str());
}

// XMemTransferMemory moves budget between the console's title and system partitions; a PC has neither, so the shim
// succeeds for every direction and size and moves (allocates) nothing. The XMEM_TRANSFER_TYPE values are not public,
// so the directions here are just the first few enum values and one far outside any plausible range.
void test_transfer_memory(XMemTransferMemoryFn transfer) {
    constexpr SIZE_T kMiB = 1024 * 1024;
    const uint32_t types[] = {0u, 1u, 2u, 0xffffffffu};
    // 2 MiB and 40 MiB are the sizes the samples pass ("Must be multiples of 2MB sizes"); 0, an odd size and SIZE_MAX
    // are the boundaries.
    const SIZE_T sizes[] = {0, 2 * kMiB, 40 * kMiB, 2 * kMiB + 1, (std::numeric_limits<SIZE_T>::max)()};

    PROCESS_MEMORY_COUNTERS_EX before{};
    const bool have_before = GetProcessMemoryInfo(GetCurrentProcess(),
                                                  reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&before), sizeof(before));
    bool all_true = true;
    bool last_error_kept = true;
    for (const uint32_t type : types) {
        for (const SIZE_T size : sizes) {
            SetLastError(0x2bad);
            const BOOL result = transfer(type, size);
            if (result != TRUE) {
                all_true = false;
                std::printf("  XMemTransferMemory(%u, %zu) -> %d\n", type, static_cast<size_t>(size), result);
            }
            last_error_kept &= GetLastError() == 0x2bad;
        }
    }
    check(all_true, "XMemTransferMemory returns TRUE (1) for every direction and size, 0 and SIZE_MAX included");
    check(last_error_kept, "XMemTransferMemory leaves the thread's last error alone");

    PROCESS_MEMORY_COUNTERS_EX after{};
    const bool have_after = GetProcessMemoryInfo(GetCurrentProcess(),
                                                 reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&after), sizeof(after));
    // 20 calls asked to move up to 40 MiB each; nothing may be committed for them. 1 MiB of slack covers the CRT's
    // own buffers (the one-time stderr line).
    check(have_before && have_after && after.PrivateUsage < before.PrivateUsage + kMiB,
          "XMemTransferMemory commits no memory for the transferred sizes");

    // A bool-returning declaration reads only AL: TRUE must be non-zero there as well.
    const auto as_bool = reinterpret_cast<bool(WINAPI*)(uint32_t, SIZE_T)>(reinterpret_cast<void*>(transfer));
    check(as_bool(0, 2 * kMiB), "XMemTransferMemory's result also reads as true through a bool return");

    // The enum is 32 bits: garbage in the upper half of RCX must not change anything.
    const auto wide = reinterpret_cast<BOOL(WINAPI*)(uint64_t, SIZE_T)>(reinterpret_cast<void*>(transfer));
    check(wide(0xdead000000000000ull, 2 * kMiB) == TRUE, "XMemTransferMemory ignores the upper half of RCX");
}

// gta6rt --trace: one "xmem_transfer" event per distinct (type, size), none with the trace off.
void test_transfer_trace(XMemTransferMemoryFn transfer) {
    wchar_t temp[MAX_PATH];
    GetTempPathW(MAX_PATH, temp);
    const std::wstring path =
        std::wstring(temp) + L"xmem_transfer_trace_test_" + std::to_wstring(GetCurrentProcessId()) + L".jsonl";
    DeleteFileW(path.c_str());

    check(transfer(7, 12345) == TRUE, "XMemTransferMemory with the trace off still returns TRUE");

    check(rt::startup_trace::open(path.c_str()), "open the startup trace file");
    transfer(0, 2097152);
    transfer(0, 2097152);  // same pair again: no second event
    transfer(0, 41943040);
    transfer(1, 2097152);
    rt::startup_trace::close();

    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    const std::string log = text.str();
    auto count = [&](const std::string& needle) {
        int n = 0;
        for (size_t at = log.find(needle); at != std::string::npos; at = log.find(needle, at + 1)) ++n;
        return n;
    };
    check(count("\"kind\":\"xmem_transfer\"") == 3, "three xmem_transfer events: one per distinct (type, size)");
    check(count("\"api\":\"XMemTransferMemory\",\"type\":0,\"size\":2097152") == 1,
          "type 0 / 2 MiB is recorded once although it was called twice");
    check(count("\"api\":\"XMemTransferMemory\",\"type\":0,\"size\":41943040") == 1, "type 0 / 40 MiB is recorded");
    check(count("\"api\":\"XMemTransferMemory\",\"type\":1,\"size\":2097152") == 1, "type 1 / 2 MiB is recorded");
    check(count("\"size\":12345") == 0, "nothing was recorded while the trace was off");
    check(count("\"caller\":\"") == 3, "every transfer event names its caller");
    DeleteFileW(path.c_str());
}

using HostVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, DWORD);
using HostVirtualFreeFn = BOOL(WINAPI*)(void*, SIZE_T, DWORD);
using HostVirtualProtectFn = BOOL(WINAPI*)(void*, SIZE_T, DWORD, PDWORD);

// RT-217: a title's kernel32!VirtualAlloc carries the console's MEM_GRAPHICS / MEM_TITLE type bits, which the host
// rejects. The registered shim has to drop them and still allocate.
void test_title_virtual_alloc() {
    constexpr DWORD kMemGraphics = 0x10000000;
    constexpr DWORD kMemTitle = 0x40000000;
    const char* const dlls[] = {"kernel32.dll", "kernelbase.dll", "api-ms-win-core-memory-l1-1-0.dll"};
    for (const char* dll : dlls) {
        const auto alloc = reinterpret_cast<HostVirtualAllocFn>(rt::shims::find(dll, "VirtualAlloc"));
        const auto free = reinterpret_cast<HostVirtualFreeFn>(rt::shims::find(dll, "VirtualFree"));
        const auto protect = reinterpret_cast<HostVirtualProtectFn>(rt::shims::find(dll, "VirtualProtect"));
        char what[160];
        std::snprintf(what, sizeof what, "registry exposes %s's virtual memory shims", dll);
        check(alloc && free && protect, what);
        if (!alloc || !free || !protect) continue;

        auto generation = leonida::title_memory_map::generation();
        void* const gfx = alloc(nullptr, 0x10000, MEM_RESERVE | MEM_COMMIT, 0x2004);  // PAGE_GRAPHICS_READWRITE
        check(gfx != nullptr, "title VirtualAlloc with protect 0x2004 succeeds");
        if (gfx) {
            check(leonida::title_memory_map::generation() != generation, "title VirtualAlloc invalidates shadows");
            MEMORY_BASIC_INFORMATION mbi{};
            check(VirtualQuery(gfx, &mbi, sizeof mbi) && mbi.Protect == PAGE_READWRITE,
                  "protect 0x2004 commits PAGE_READWRITE");
            DWORD old = 0;
            generation = leonida::title_memory_map::generation();
            check(protect(gfx, 0x1000, PAGE_READONLY, &old) && old == PAGE_READWRITE,
                  "title VirtualProtect changes protection and returns the previous protection");
            check(leonida::title_memory_map::generation() != generation, "title VirtualProtect invalidates shadows");
            generation = leonida::title_memory_map::generation();
            check(free(gfx, 0x1000, MEM_DECOMMIT), "title VirtualFree decommits a page");
            check(leonida::title_memory_map::generation() != generation, "title VirtualFree invalidates shadows");
            check(free(gfx, 0, MEM_RELEASE), "release the graphics allocation");
        }
        for (const DWORD bits : {kMemGraphics, kMemTitle, kMemGraphics | kMemTitle, DWORD{0}}) {
            void* const p = alloc(nullptr, 0x10000, MEM_RESERVE | MEM_COMMIT | bits, PAGE_READWRITE);
            std::snprintf(what, sizeof what, "%s!VirtualAlloc with type bits %#lx returns memory", dll,
                          static_cast<unsigned long>(bits));
            check(p != nullptr, what);
            if (!p) continue;
            static_cast<volatile uint8_t*>(p)[0] = 0x5a;
            check(static_cast<volatile uint8_t*>(p)[0] == 0x5a, "the allocation is committed and writable");
            check(VirtualFree(p, 0, MEM_RELEASE), "release the allocation");
        }
    }
}

// The title's ordinary reservations are write-watched for the D3D12 layer's shadows (leonida::title_write_watch):
// XMemVirtualAlloc's plain, 2 MiB and 64 KiB committed regions and kernel32!VirtualAlloc's reservations, each recorded
// until the title releases it. Placeholders and physical-page windows are not (a section view can't be), nor is a
// region the title asked to be write-watched itself (its bits are the title's). With LEONIDA_WRITE_WATCH=0 (the
// xmem_no_write_watch test) nothing is.
void test_write_watched_reservations(XMemVirtualAllocFn virtual_alloc) {
    const bool on = leonida::title_write_watch::enabled();
    std::printf("xmem_test: write watching %s\n", on ? "on" : "off (LEONIDA_WRITE_WATCH=0)");
    const auto kernel_watching = [](void* base, SIZE_T size) {
        void* found[16]{};
        ULONG_PTR count = 16;
        ULONG granularity = 0;
        return GetWriteWatch(0, base, size, found, &count, &granularity) == 0;
    };
    const auto title_free = reinterpret_cast<HostVirtualFreeFn>(rt::shims::find("kernel32.dll", "VirtualFree"));
    const auto title_alloc = reinterpret_cast<HostVirtualAllocFn>(rt::shims::find("kernel32.dll", "VirtualAlloc"));
    check(title_free && title_alloc, "registry exposes kernel32.dll's VirtualAlloc and VirtualFree shims");
    if (!title_free || !title_alloc) return;
    struct Case {
        const char* what;
        DWORD type;
        SIZE_T size;
        bool watched;
    };
    const Case cases[] = {
        {"plain reserve+commit", MEM_RESERVE | MEM_COMMIT, 0x3000, true},
        {"commit at no address", MEM_COMMIT, 0x3000, true},
        {"reserve only", MEM_RESERVE, 0x20000, true},
        {"2 MiB pages", kXmemLargePages | MEM_RESERVE | MEM_COMMIT, kPage2m, true},
        {"64 KiB pages committed", kXmem64kPages | MEM_RESERVE | MEM_COMMIT, kPage64k * 2, true},
        {"64 KiB physical-page window", kXmem64kPages | MEM_RESERVE, kPage64k * 2, false},
        {"placeholder", MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, kPage64k * 2, false},
        {"the title's own write watch", MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, 0x3000, false},
    };
    for (const Case& c : cases) {
        void* const base = virtual_alloc(nullptr, c.size, c.type, 0, PAGE_READWRITE);
        char what[192];
        std::snprintf(what, sizeof what, "XMemVirtualAlloc %s succeeds", c.what);
        check(base != nullptr, what);
        if (!base) continue;
        const auto at = reinterpret_cast<uintptr_t>(base);
        const bool want = on && c.watched;
        std::snprintf(what, sizeof what, "XMemVirtualAlloc %s: %s for the shadows", c.what,
                      want ? "recorded write-watched" : "not recorded");
        check(leonida::title_write_watch::contains(at) == want &&
                  leonida::title_write_watch::contains(at + c.size - 1) == want,
              what);
        if (want) {
            std::snprintf(what, sizeof what, "XMemVirtualAlloc %s: the kernel write-watches it", c.what);
            check(kernel_watching(base, c.size), what);
        }
        if ((c.type & MEM_RESERVE_PLACEHOLDER) || ((c.type & kXmem64kPages) == kXmem64kPages && !(c.type & MEM_COMMIT))) {
            VirtualFree(base, 0, MEM_RELEASE);
            continue;
        }
        std::snprintf(what, sizeof what, "title VirtualFree releases the region (%s) and forgets it", c.what);
        check(title_free(base, 0, MEM_RELEASE) && !leonida::title_write_watch::contains(at), what);
    }
    // kernel32!VirtualAlloc as the title calls it: a reservation is recorded, a commit inside it adds nothing.
    void* const reserved = title_alloc(nullptr, 0x40000, MEM_RESERVE, PAGE_READWRITE);
    check(reserved != nullptr, "title VirtualAlloc reserves");
    if (!reserved) return;
    const auto at = reinterpret_cast<uintptr_t>(reserved);
    const auto generation = leonida::title_write_watch::generation();
    check(title_alloc(reserved, 0x1000, MEM_COMMIT, PAGE_READWRITE) == reserved, "title VirtualAlloc commits in it");
    check(leonida::title_write_watch::generation() == generation, "a commit inside a reservation records nothing new");
    check(leonida::title_write_watch::contains(at + 0x3ffff) == on,
          on ? "title VirtualAlloc's reservation is recorded write-watched" : "title VirtualAlloc's is not recorded");
    if (on) {
        check(kernel_watching(reserved, 0x40000), "the kernel write-watches title VirtualAlloc's reservation");
        static_cast<volatile uint8_t*>(reserved)[0x10] = 1;
        void* found[4]{};
        ULONG_PTR count = 4;
        ULONG granularity = 0;
        check(GetWriteWatch(0, reserved, 0x40000, found, &count, &granularity) == 0 && count == 1 &&
                  found[0] == reserved,
              "a write to it is in the kernel's answer");
    } else {
        check(!kernel_watching(reserved, 0x40000), "LEONIDA_WRITE_WATCH=0: the kernel does not write-watch it");
    }
    check(title_free(reserved, 0, MEM_RELEASE) && !leonida::title_write_watch::contains(at),
          "title VirtualFree releases it and forgets it");
}

// A reservation the title releases and reserves again at the same address is a new one to the shadows' journal
// (leonida::title_write_watch::ask): a reader that asked over the old one is told all of the new one's pages as
// written, though nothing wrote them (a new reservation's kernel bits are clear, and a MEM_RESERVE | MEM_COMMIT in one
// call marks nothing). Through kernel32!VirtualAlloc and XMemVirtualAlloc; a page decommitted and committed again
// inside a reservation is told as written too. And the title's XMemGetWriteWatch and XMemResetWriteWatch refuse a
// reservation write-watched for the shadows (on the console it is not write-watched, and a reset would clear the bits
// the shadows ask about): the journal still sees the write.
void test_write_watch_rereserve(XMemVirtualAllocFn virtual_alloc, XMemGetWriteWatchFn get_write_watch,
                                XMemResetWriteWatchFn reset_write_watch) {
    if (!leonida::title_write_watch::enabled()) return;
    namespace ww = leonida::title_write_watch;
    const auto title_free = reinterpret_cast<HostVirtualFreeFn>(rt::shims::find("kernel32.dll", "VirtualFree"));
    const auto title_alloc = reinterpret_cast<HostVirtualAllocFn>(rt::shims::find("kernel32.dll", "VirtualAlloc"));
    if (!title_free || !title_alloc) return;
    constexpr SIZE_T kSize = 0x10000;
    // The pages of [base, base + kSize) the reader is told were written.
    const auto told = [](std::uint64_t reader, void* base) {
        const auto from = reinterpret_cast<uintptr_t>(base);
        const std::vector<std::pair<uintptr_t, uintptr_t>> runs{{from, from + kSize}};
        size_t pages = 0;
        ww::ask(
            reader, runs, [&](uintptr_t a, uintptr_t b) { pages += (b - a) / ww::kPage; }, [](uintptr_t, uintptr_t) {});
        return pages;
    };
    struct Way {
        const char* what;
        std::function<void*(void*)> reserve;  // MEM_RESERVE | MEM_COMMIT at the address (null: anywhere)
    };
    const Way ways[] = {
        {"kernel32!VirtualAlloc",
         [&](void* at) { return title_alloc(at, kSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); }},
        {"XMemVirtualAlloc", [&](void* at) { return virtual_alloc(at, kSize, MEM_RESERVE | MEM_COMMIT, 0, PAGE_READWRITE); }},
    };
    char what[192];
    for (const Way& way : ways) {
        void* base = nullptr;
        std::uint64_t reader = 0;
        bool told_once = true, same_address = false;
        // Another thread may take the range between the release and the new reservation: tried again elsewhere.
        for (int attempt = 0; attempt < 8 && !same_address; ++attempt) {
            base = way.reserve(nullptr);
            if (!base) break;
            reader = ww::open_reader();
            static_cast<volatile uint8_t*>(base)[0x10] = 1;
            told_once = told_once && told(reader, base) == 1 && told(reader, base) == 0;
            void* const again = title_free(base, 0, MEM_RELEASE) ? way.reserve(base) : nullptr;
            same_address = again == base;
            if (!same_address) {
                ww::close_reader(reader);
                if (again) title_free(again, 0, MEM_RELEASE);
            }
        }
        std::snprintf(what, sizeof what, "%s: a write to a new reservation is told once", way.what);
        check(told_once, what);
        std::snprintf(what, sizeof what, "%s: released and reserved again at the same address", way.what);
        check(same_address, what);
        if (!same_address) continue;
        std::snprintf(what, sizeof what, "%s: the new reservation's pages are all told as written", way.what);
        check(told(reader, base) == kSize / ww::kPage, what);
        check(told(reader, base) == 0, "and only once");
        std::snprintf(what, sizeof what, "%s: a page decommitted and committed again is told as written", way.what);
        check(title_free(static_cast<uint8_t*>(base) + 0x3000, 0x1000, MEM_DECOMMIT) &&
                  title_alloc(static_cast<uint8_t*>(base) + 0x3000, 0x1000, MEM_COMMIT, PAGE_READWRITE) &&
                  told(reader, base) == 1,
              what);
        // The title's own write-watch calls leave the shadows' bits alone.
        static_cast<volatile uint8_t*>(base)[0x5010] = 1;
        void* found[16]{};
        ULONG_PTR count = 16;
        ULONG granularity = 0;
        std::snprintf(what, sizeof what, "%s: XMemResetWriteWatch and XMemGetWriteWatch refuse the shadows' reservation",
                      way.what);
        check(reset_write_watch(base, kSize) != 0 &&
                  get_write_watch(WRITE_WATCH_FLAG_RESET, base, kSize, found, &count, &granularity) != 0,
              what);
        check(told(reader, base) == 1, "the write is still told to the shadows");
        ww::close_reader(reader);
        title_free(base, 0, MEM_RELEASE);
    }
}

void print_result() {
    if (g_failures == 0) {
        std::printf("xmem_test: OK (%d checks)\n", g_checks);
    } else {
        std::printf("xmem_test: FAILED (%d checks, %d failure%s)\n", g_checks, g_failures,
                    g_failures == 1 ? "" : "s");
    }
}

}  // namespace

int main() {
    std::printf("xmem_test: registered xmem.dll shim ABI\n");

    const auto alloc = reinterpret_cast<XMemAllocFn>(rt::shims::find("xmem.dll", "XMemAlloc"));
    const auto free_mem = reinterpret_cast<XMemFreeFn>(rt::shims::find("xmem.dll", "XMemFree"));
    const auto query = reinterpret_cast<XMemVirtualQueryFn>(rt::shims::find("xmem.dll", "XMemVirtualQuery"));
    const auto get_stats =
        reinterpret_cast<XMemGetWorkingSetStatisticsFn>(rt::shims::find("xmem.dll", "XMemGetWorkingSetStatistics"));
    const auto virtual_alloc =
        reinterpret_cast<XMemVirtualAllocFn>(rt::shims::find("xmem.dll", "XMemVirtualAlloc"));
    const auto get_write_watch =
        reinterpret_cast<XMemGetWriteWatchFn>(rt::shims::find("xmem.dll", "XMemGetWriteWatch"));
    const auto reset_write_watch =
        reinterpret_cast<XMemResetWriteWatchFn>(rt::shims::find("xmem.dll", "XMemResetWriteWatch"));
    const auto transfer =
        reinterpret_cast<XMemTransferMemoryFn>(rt::shims::find("xmem.dll", "XMemTransferMemory"));

    check(alloc != nullptr, "registry exposes xmem.dll!XMemAlloc");
    check(free_mem != nullptr, "registry exposes xmem.dll!XMemFree");
    check(query != nullptr, "registry exposes xmem.dll!XMemVirtualQuery");
    check(get_stats != nullptr, "registry exposes xmem.dll!XMemGetWorkingSetStatistics");
    check(virtual_alloc != nullptr, "registry exposes xmem.dll!XMemVirtualAlloc");
    check(get_write_watch != nullptr, "registry exposes xmem.dll!XMemGetWriteWatch");
    check(reset_write_watch != nullptr, "registry exposes xmem.dll!XMemResetWriteWatch");
    check(transfer != nullptr, "registry exposes xmem.dll!XMemTransferMemory");

    const auto alloc_default = reinterpret_cast<XMemAllocFn>(rt::shims::find("xmem.dll", "XMemAllocDefault"));
    const auto free_default = reinterpret_cast<XMemFreeFn>(rt::shims::find("xmem.dll", "XMemFreeDefault"));
    const auto tools_mem =
        reinterpret_cast<XMemIsToolsMemAvailableFn>(rt::shims::find("xmem.dll", "XMemIsToolsMemAvailable"));
    const PhysicalApi physical{
        virtual_alloc,
        reinterpret_cast<XMemAllocatePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemAllocatePhysicalPages")),
        reinterpret_cast<XMemFreePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemFreePhysicalPages")),
        reinterpret_cast<XMemMapPhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemMapPhysicalPages")),
        reinterpret_cast<XMemMapPhysicalPagesScatterFn>(rt::shims::find("xmem.dll", "XMemMapPhysicalPagesScatter")),
    };
    check(alloc_default != nullptr, "registry exposes xmem.dll!XMemAllocDefault");
    check(free_default != nullptr, "registry exposes xmem.dll!XMemFreeDefault");
    check(tools_mem != nullptr, "registry exposes xmem.dll!XMemIsToolsMemAvailable");
    check(physical.allocate != nullptr, "registry exposes xmem.dll!XMemAllocatePhysicalPages");
    check(physical.free_pages != nullptr, "registry exposes xmem.dll!XMemFreePhysicalPages");
    check(physical.map != nullptr, "registry exposes xmem.dll!XMemMapPhysicalPages");
    check(physical.scatter != nullptr, "registry exposes xmem.dll!XMemMapPhysicalPagesScatter");

    if (!alloc || !free_mem || !query || !get_stats) {
        print_result();
        return 1;
    }

    test_alloc(alloc, free_mem);
    test_working_set(get_stats);
    test_virtual_query(query);
    if (virtual_alloc && get_write_watch && reset_write_watch) {
        test_virtual_alloc(virtual_alloc, get_write_watch, reset_write_watch);
    }
    test_attribute_trace(alloc, free_mem, virtual_alloc);
    if (transfer) {
        test_transfer_memory(transfer);
        test_transfer_trace(transfer);
    }

    test_title_virtual_alloc();
    if (virtual_alloc) test_write_watched_reservations(virtual_alloc);
    if (virtual_alloc && get_write_watch && reset_write_watch)
        test_write_watch_rereserve(virtual_alloc, get_write_watch, reset_write_watch);

    if (alloc_default && free_default) test_default_alloc(alloc, free_mem, alloc_default, free_default);
    if (tools_mem) check(tools_mem() == FALSE, "XMemIsToolsMemAvailable returns FALSE (no devkit tools memory)");
    if (virtual_alloc && physical.allocate && physical.free_pages && physical.map && physical.scatter) {
        test_physical_pages(physical);
        test_copy_page_mappings(physical);
        test_copy_page_mappings_cpu_view(physical);
        test_physical_pages_commit_on_touch(physical);
        test_commit_physical_pages(physical);
        test_first_touch_during_remap(physical);
        test_long_run(physical);
        test_split_under_reader(physical);
        test_placeholders(physical);
        test_write_combine_window(physical);
        test_heap_lock_faults(physical);
    }

    print_result();
    return g_failures ? 1 : 0;
}
