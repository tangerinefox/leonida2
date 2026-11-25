// xmem.dll: the GDK's tracked allocator. Mapped onto the CRT's aligned heap and VirtualQueryEx.
//
// The measured ABI (docs from the clean-room spec) shows four imports:
//   XMemAlloc(size, attributes)
//   XMemFree(ptr, attributes)
//   XMemVirtualQuery(process, address, out64)
//   XMemGetWorkingSetStatistics(kind, out)   (kind 0x20000001 at all seven FSR call sites)
//
// ASSUMPTION: the attribute word is not decoded by the measured call sites (one passes 0x5e50000). Until a call
// site shows the alignment/type fields, every block is 64-byte aligned (covers cache lines and SIMD) and a size of
// zero is rounded up to one byte so the result is a stable, non-null pointer that XMemFree can release.
//
// The default allocators, the physical-page functions and XMemIsToolsMemAvailable follow argument shapes measured
// locally at a title's call sites (each function says which). The page functions are built on the Windows placeholder
// APIs (VirtualAlloc2, MapViewOfFile3), whose flags the title's own XMemVirtualAlloc calls already use.
#include <windows.h>
#include <intrin.h>
#include <malloc.h>
#include <psapi.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shims.h"
#include "startup_trace.h"
#include "xmem.h"
#include "../src/dstorage_trace.h"
#include "../src/title_page_watch.h"
#include "../src/title_geometry_watch.h"
#include "../src/title_va_alias.h"

namespace {

constexpr size_t kAlign = 64;
constexpr size_t kNoSize = static_cast<size_t>(-1);  // XMemFree has no size field

std::string caller_field(const void* caller) {
    return ",\"caller\":" + rt::startup_trace::quote(rt::startup_trace::where(reinterpret_cast<uint64_t>(caller)));
}

// gta6rt --trace FILE: the first XMemAlloc, XMemFree, XMemAllocDefault, XMemFreeDefault and XMemVirtualAlloc call for
// each distinct raw attribute word go to the startup trace as an "xmem_attributes" event (api, the word, the size for an
// allocation, the caller as module+RVA). The bits of the attribute word are not known yet; this records them from a
// real run without logging every allocation, which would fill the file. Off: one relaxed load per call.
void note_attributes(const char* api, uint64_t attributes, size_t size, const void* caller) {
    if (!rt::startup_trace::enabled()) return;
    static std::mutex mutex;
    static std::set<std::pair<std::string, uint64_t>> seen;
    {
        const std::lock_guard lock(mutex);
        if (!seen.insert({api, attributes}).second) return;
    }
    std::string fields = std::string("\"api\":\"") + api +
                         "\",\"attributes\":" + rt::startup_trace::quote(rt::startup_trace::hex(attributes));
    if (size != kNoSize) fields += ",\"size\":" + std::to_string(size);
    fields += caller_field(caller);
    rt::startup_trace::event("xmem_attributes", fields);
}

void* allocate_block(size_t size) { return _aligned_malloc(size ? size : 1, kAlign); }

// Every block and range the title gets here is recorded as the title's memory (src/title_va_alias.h): a shader
// address into it can be given a GPU copy, and an address anywhere else cannot.
void* title_block(size_t size) {
    void* p = allocate_block(size);
    if (p) leonida::title_memory_map::changed();
    leonida::title_va_aliases().add_title_memory(reinterpret_cast<uintptr_t>(p), size ? size : 1);
    return p;
}

void free_title_block(void* p) {
    if (p) leonida::title_va_aliases().remove_title_memory(reinterpret_cast<uintptr_t>(p));
    _aligned_free(p);
    if (p) leonida::title_memory_map::changed();
}

void* __stdcall XMemAlloc(size_t size, uint64_t attributes) {
    note_attributes("XMemAlloc", attributes, size, _ReturnAddress());
    return title_block(size);
}

void __stdcall XMemFree(void* p, uint64_t attributes) {
    note_attributes("XMemFree", attributes, kNoSize, _ReturnAddress());
    free_title_block(p);
}

// XMemAllocDefault / XMemFreeDefault: the GDK's own allocator behind XMemAlloc and XMemFree, which a title that hooks
// XMemAlloc forwards to. The measured title reaches each through one wrapper that is a bare tail jump to the import,
// so they take exactly XMemAlloc's and XMemFree's arguments. They share the heap: a block from either allocator may be
// released by either free.
void* __stdcall XMemAllocDefault(size_t size, uint64_t attributes) {
    note_attributes("XMemAllocDefault", attributes, size, _ReturnAddress());
    return title_block(size);
}

void __stdcall XMemFreeDefault(void* p, uint64_t attributes) {
    note_attributes("XMemFreeDefault", attributes, kNoSize, _ReturnAddress());
    free_title_block(p);
}

// BOOL XMemIsToolsMemAvailable(void). The measured title calls it from one helper, which six out-of-memory paths reach
// only after an allocation has failed; the helper breaks into the debugger when the result is zero. Tools memory is
// the extra memory of a development kit. A PC has none, so the answer is FALSE, and the title's own breakpoint marks
// where it ran out of memory.
BOOL WINAPI XMemIsToolsMemAvailable() {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[rt] xmem: XMemIsToolsMemAvailable -> FALSE (no devkit tools memory; logged once)\n");
    }
    return FALSE;
}

// Measured call sites pass a zeroed 64-byte structure. Only the MEMORY_BASIC_INFORMATION-compatible prefix is
// known: callers read at least the base and size fields. Fill that prefix from VirtualQueryEx and zero the rest.
constexpr size_t kVirtualQueryOutSize = 64;
static_assert(sizeof(MEMORY_BASIC_INFORMATION) <= kVirtualQueryOutSize,
              "the 64-byte XMemVirtualQuery output must hold MEMORY_BASIC_INFORMATION");

// Returns the number of bytes written (the VirtualQuery convention) or 0 on failure. The exact return type and the
// meaning of every output field are not fully known; callers in the package only test for success.
SIZE_T __stdcall XMemVirtualQuery(HANDLE process, const void* address, void* out) {
    if (!out) return 0;
    std::memset(out, 0, kVirtualQueryOutSize);
    if (!address) return 0;

    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] xmem: XMemVirtualQuery: fields after MEMORY_BASIC_INFORMATION are unknown; "
                     "the 64-byte output is zero-filled there\n");
    }

    // A null handle means the current process; GetCurrentProcess() is the (HANDLE)-1 pseudo handle.
    HANDLE target = process ? process : GetCurrentProcess();
    MEMORY_BASIC_INFORMATION info{};
    SIZE_T written = VirtualQueryEx(target, address, &info, sizeof(info));
    if (!written) return 0;

    std::memcpy(out, &info, written < kVirtualQueryOutSize ? written : kVirtualQueryOutSize);
    return written;
}

// BOOL XMemGetWorkingSetStatistics(UINT32 kind, XMEM_WORKING_SET_STATISTICS* stats). Measured at all seven call sites
// in amd_fidelityfx_upscaler_xbox.dll (1.0.2.1; calls at RVAs 0x26b3, 0x2ea1, 0x39a9, 0x7452, 0x74d2, 0x7568,
// 0x75b1): ECX is the constant 0x20000001 (its meaning is not known) and RDX points at a stack buffer (0x80 bytes
// where the frame shows it). The caller reads a 64-bit size at +8 and treats a zero return as "no data". Taking the
// pointer from the first argument wrote to address 0x20000001.
// The measured FSR sites establish argument shape, kind and the 64-bit working-set read at +8; +0's title-total memory
// limit interpretation is inferred from other Series implementations (xcl, xbox-emulator) and title engine telemetry
// sizing memory pools (RT-220). Note that if GlobalMemoryStatusEx reports 0 physical memory, XMem falls back to the
// 12 GiB title limit, whereas XSystemGetAppMemoryUsageLimit remains 0.
BOOL __stdcall XMemGetWorkingSetStatistics(uint32_t /*kind*/, void* stats) {
    if (!stats) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            std::fprintf(stderr, "[rt] xmem: XMemGetWorkingSetStatistics(nullptr) -> FALSE\n");
        }
        return FALSE;
    }

    PROCESS_MEMORY_COUNTERS pmc{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return FALSE;

    MEMORYSTATUSEX ms{sizeof(ms)};
    uint64_t total_memory = 12ULL * 1024 * 1024 * 1024;
    if (GlobalMemoryStatusEx(&ms) && ms.ullTotalPhys != 0) {
        total_memory = ms.ullTotalPhys;
    }
    // Optional override (matching XSystemGetAppMemoryUsageLimit) so memory-constrained hosts can cap title pools.
    char env[32]{};
    const DWORD env_n = GetEnvironmentVariableA("LEONIDA_TITLE_MEMORY_MB", env, sizeof(env));
    if (env_n > 0 && env_n < sizeof(env) && env[0] != '-' && env[0] != '+') {
        errno = 0;
        char* end = nullptr;
        unsigned long long mib = std::strtoull(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && mib > 0) {
            if (mib < 2048ULL) mib = 2048ULL;
            if (mib > 65536ULL) mib = 65536ULL;
            total_memory = static_cast<uint64_t>(mib) * 1024ULL * 1024ULL;
        }
    }

    uint8_t* bytes = static_cast<uint8_t*>(stats);
    std::memset(bytes, 0, 16);
    std::memcpy(bytes + 0, &total_memory, sizeof(total_memory));
    uint64_t working_set = pmc.WorkingSetSize;
    std::memcpy(bytes + 8, &working_set, sizeof(working_set));
    return TRUE;
}

// The placeholder APIs (Windows 10 1803 and later) come from kernelbase.dll at run time, so neither gta6rt nor the
// tests link onecore.lib. A host without them fails the calls that need them with ERROR_NOT_SUPPORTED.
using VirtualAlloc2Fn = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
using MapViewOfFile3Fn = PVOID(WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*,
                                        ULONG);
using UnmapViewOfFile2Fn = BOOL(WINAPI*)(HANDLE, PVOID, ULONG);

struct PlaceholderApi {
    VirtualAlloc2Fn virtual_alloc2 = nullptr;
    MapViewOfFile3Fn map_view3 = nullptr;
    UnmapViewOfFile2Fn unmap_view2 = nullptr;
    bool available() const { return virtual_alloc2 && map_view3 && unmap_view2; }
};

const PlaceholderApi& placeholder_api() {
    static const PlaceholderApi api = [] {
        PlaceholderApi resolved;
        if (const HMODULE kernelbase = GetModuleHandleW(L"kernelbase.dll")) {
            resolved.virtual_alloc2 = reinterpret_cast<VirtualAlloc2Fn>(GetProcAddress(kernelbase, "VirtualAlloc2"));
            resolved.map_view3 = reinterpret_cast<MapViewOfFile3Fn>(GetProcAddress(kernelbase, "MapViewOfFile3"));
            resolved.unmap_view2 =
                reinterpret_cast<UnmapViewOfFile2Fn>(GetProcAddress(kernelbase, "UnmapViewOfFile2"));
        }
        return resolved;
    }();
    return api;
}

// XMem allocation-type page bits select the guest page granularity and must not be forwarded as Win32 flags.
constexpr DWORD kXmemLargePages = 0x20000000;  // 2 MiB pages
constexpr DWORD kXmem64kPages = 0x20400000;    // 64 KiB pages
constexpr DWORD kXmemPageModeMask = kXmem64kPages;
constexpr SIZE_T kXmemPageSize64k = 0x10000;
constexpr SIZE_T kXmemPageSize2m = 0x200000;
constexpr DWORD kXmemHostAllocationTypeMask = MEM_COMMIT | MEM_RESERVE | MEM_REPLACE_PLACEHOLDER |
                                              MEM_RESERVE_PLACEHOLDER | MEM_RESET | MEM_TOP_DOWN | MEM_WRITE_WATCH |
                                              MEM_RESET_UNDO;

bool round_up_size(SIZE_T value, SIZE_T alignment, SIZE_T& rounded) {
    const SIZE_T max = (std::numeric_limits<SIZE_T>::max)();
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 || value > max - (alignment - 1)) return false;
    rounded = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

// The console's PAGE_GRAPHICS_* protection bits (winnt.h: PAGE_GRAPHICS_NOACCESS 0x800 .. PAGE_GRAPHICS_NOCACHE
// 0x40000). The host refuses them with ERROR_INVALID_PARAMETER, and the D3D12 layer reads guest memory through the
// CPU mapping, so only the CPU protection is applied. The retail executable passes 0x801 and 0x22004.
constexpr DWORD kPageGraphicsMask = 0x0007F800;

DWORD host_protect(DWORD p) {
    const DWORD cpu = p & ~kPageGraphicsMask;
    return cpu ? cpu : PAGE_NOACCESS;
}

// Physical pages and the windows they are mapped into. One mutex covers both tables and every map and unmap.
//
// A physical-page window is a reserve-only XMemVirtualAlloc in a guest page mode, or a reserve-only replacement of
// part of a placeholder. On the host it is a placeholder, the only kind of reservation a section view can later
// replace, and its protection is kept here: the map calls take none, so a mapped page gets its window's protection, as
// AWE's MapUserPhysicalPages does.
//
// ASSUMPTION: a title does not commit memory inside such a window with kernel32!VirtualAlloc. A placeholder refuses a
// plain commit; the measured windows are only ever passed to the map calls.
struct Window {
    SIZE_T size;
    DWORD protect;
};

struct Section {
    HANDLE handle;
    void* scratch;
    explicit Section(HANDLE h) : handle(h), scratch(MapViewOfFile(h, FILE_MAP_WRITE, 0, 0, 0)) {}
    Section(const Section&) = delete;
    Section& operator=(const Section&) = delete;
    ~Section() {
        if (scratch) UnmapViewOfFile(scratch);
        CloseHandle(handle);
    }
};

struct PhysicalPage {
    std::shared_ptr<Section> section;  // a mapped view holds its own reference to the section
    uint64_t offset;
};

constexpr uint64_t kPhysicalPageSize = 0x10000;
// Page numbers are 4 KiB frame numbers of 64 KiB pages: the measured title stores them packed and assumes each is a
// non-zero multiple of 16 below 2^25, with 0 for an entry it unmaps. So a page number is slot * 16 with slot in
// [1, 2^21), and freed slots are reused so a long run never runs out.
constexpr uint64_t kFramesPerPage = kPhysicalPageSize / 0x1000;
constexpr uint64_t kPageSlotLimit = uint64_t{1} << 21;

std::mutex g_phys_mutex;
std::map<uintptr_t, Window> g_windows;              // base -> window
std::unordered_map<uint64_t, PhysicalPage> g_pages;  // page number -> backing
std::vector<uint64_t> g_free_slots;
uint64_t g_next_slot = 1;
// A mapped run: `count` pages of one section at consecutive offsets, mapped as one view at its base. One view per run
// rather than per page: under Wine each view costs a placeholder split and a section map whose time grows with the
// number of views, and the retail title maps about 100,000 pages in one call.
struct View {
    uint64_t count;
    std::shared_ptr<Section> section;  // kept for remapping part of the run after its pages are freed
    uint64_t offset;                   // section offset of the first page
    DWORD protect;
};
leonida::host_heap::map<uintptr_t, View> g_views;  // view base -> view; host_heap: see g_fault_lock

// The views (g_views) as the fault handler reads them, which is all it reads: physical_page_fault takes it shared at
// every access violation in the process, xmem's own or not, and so does commit_physical_range. A title thread can
// take an access violation while it holds the process heap's lock (XMemAlloc is _aligned_malloc, so HeapAlloc and
// HeapFree write block headers on the title's pages: a page waiting for its first touch, or one a D3D12 shadow
// guards), and then waits here. So a thread that holds it exclusively must never wait for the process heap: it would
// wait for the faulting thread, which waits for it. Everything allocated or freed under it comes from
// leonida::host_heap (g_views, g_mapped, g_page_addresses, the sections, title_page_aliases), nothing under it logs,
// and nothing under it reads title memory, which could fault into the handler on the same thread.
SRWLOCK g_fault_lock = SRWLOCK_INIT;

struct FaultLockExclusive {
    FaultLockExclusive() { AcquireSRWLockExclusive(&g_fault_lock); }
    ~FaultLockExclusive() { ReleaseSRWLockExclusive(&g_fault_lock); }
    FaultLockExclusive(const FaultLockExclusive&) = delete;
    FaultLockExclusive& operator=(const FaultLockExclusive&) = delete;
};

struct FaultLockShared {
    FaultLockShared() { AcquireSRWLockShared(&g_fault_lock); }
    ~FaultLockShared() { ReleaseSRWLockShared(&g_fault_lock); }
    FaultLockShared(const FaultLockShared&) = delete;
    FaultLockShared& operator=(const FaultLockShared&) = delete;

};

// With g_phys_mutex held; changed under g_fault_lock too (unmap_range), so they allocate from host_heap.
template <class K, class V>
using HostUnorderedMap =
    std::unordered_map<K, V, std::hash<K>, std::equal_to<K>, leonida::host_heap::Allocator<std::pair<const K, V>>>;
using AddressSet = std::set<uintptr_t, std::less<uintptr_t>, leonida::host_heap::Allocator<uintptr_t>>;
HostUnorderedMap<uintptr_t, uint64_t> g_mapped;            // mapped 64 KiB address -> page number
HostUnorderedMap<uint64_t, AddressSet> g_page_addresses;  // page number -> the addresses it is mapped at

// With g_phys_mutex held. Keeps leonida::title_page_aliases in step: a page mapped at two or more addresses is
// aliased at each of them (the D3D12 layer's slot-133 shadows copy those pages at every submission instead of
// waiting for a write fault, which a write through another address doesn't raise).
void note_mapping(uint64_t page, uintptr_t address, bool mapped) {
    auto& addresses = g_page_addresses[page];
    if (mapped) {
        addresses.insert(address);
    } else {
        addresses.erase(address);
        leonida::title_page_aliases::set(address, false);
    }
    using namespace leonida::geometry_writes;
    if (state().count.load(std::memory_order_acquire)) {
        notify(address, address + kPhysicalPageSize, Event::mapping);
        for (auto other : addresses)
            if (other != address) notify(other, other + kPhysicalPageSize, Event::mapping);
    }
    const bool aliased = addresses.size() > 1;
    for (const uintptr_t at : addresses) leonida::title_page_aliases::set(at, aliased);
    if (addresses.empty()) g_page_addresses.erase(page);
}

// With g_phys_mutex held. A window replaces every window it overlaps: the title released those regions before
// the host could hand the addresses out again.
void record_window(void* base, SIZE_T size, DWORD protect) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(base);
    const uintptr_t end = start + size;
    auto it = g_windows.lower_bound(start);
    if (it != g_windows.begin()) {
        const auto prev = std::prev(it);
        if (prev->first + prev->second.size > start) it = prev;
    }
    while (it != g_windows.end() && it->first < end) it = g_windows.erase(it);
    g_windows[start] = Window{size, protect};
}

// With g_phys_mutex held: the window holding [address, address + bytes), if one does.
const Window* find_window(uintptr_t address, uint64_t bytes) {
    auto it = g_windows.upper_bound(address);
    if (it == g_windows.begin()) return nullptr;
    --it;
    const uint64_t offset = address - it->first;
    if (offset >= it->second.size || bytes > it->second.size - offset) return nullptr;
    return &it->second;
}

// [address, address + size) as one placeholder of its own: split it out of the placeholder that holds it. Fails when
// the range is not inside one placeholder.
bool isolate_placeholder(void* address, SIZE_T size) {
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(address, &info, sizeof(info)) || info.State != MEM_RESERVE) {
        SetLastError(ERROR_INVALID_ADDRESS);
        return false;
    }
    if (info.AllocationBase == address && info.RegionSize == size) return true;
    if (!VirtualFree(address, size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) return false;
    leonida::title_memory_map::changed();
    return true;
}

PVOID reserve_placeholder(void* address, SIZE_T size, SIZE_T alignment) {
    MEM_ADDRESS_REQUIREMENTS requirements{};
    MEM_EXTENDED_PARAMETER parameter{};
    ULONG parameter_count = 0;
    if (!address && alignment > kXmemPageSize64k) {
        requirements.Alignment = alignment;
        parameter.Type = MemExtendedParameterAddressRequirements;
        parameter.Pointer = &requirements;
        parameter_count = 1;
    }
    return placeholder_api().virtual_alloc2(GetCurrentProcess(), address, size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                            PAGE_NOACCESS, parameter_count ? &parameter : nullptr, parameter_count);
}

// A new ordinary reservation of the title's (MEM_RESERVE, not a placeholder, not physical, not large pages) is made
// with MEM_WRITE_WATCH and recorded (leonida::title_write_watch): the D3D12 layer's shadows of these pages then ask the
// kernel which pages were written instead of making them read-only and taking a fault at the title's first write.
// `allocate(type)` makes the call. A host that refuses the flag gets the call without it, and the pages are guarded
// as before; so are they with LEONIDA_WRITE_WATCH=0. A reservation the title asked to be write-watched itself is not
// recorded: its bits are the title's (XMemGetWriteWatch).
template <class Allocate>
void* reserve_watched(DWORD type, const void* address, SIZE_T size, Allocate&& allocate) {
    constexpr DWORD kNotWatchable = MEM_WRITE_WATCH | MEM_PHYSICAL | MEM_LARGE_PAGES | MEM_RESERVE_PLACEHOLDER |
                                    MEM_REPLACE_PLACEHOLDER | MEM_RESET | MEM_RESET_UNDO;
    // A commit at no given address reserves too.
    const bool reserves = (type & MEM_RESERVE) || (!address && (type & MEM_COMMIT));
    const bool watch = reserves && !(type & kNotWatchable) && leonida::title_write_watch::enabled();
    void* result = allocate(watch ? type | MEM_RESERVE | MEM_WRITE_WATCH : type);
    if (!watch) return result;
    if (!result) return allocate(type);
    // The reservation is [result, end of the requested range): VirtualAlloc rounds a given address down to 64 KiB.
    const uintptr_t base = reinterpret_cast<uintptr_t>(result);
    const uintptr_t end = (address ? reinterpret_cast<uintptr_t>(address) : base) + size;
    leonida::title_write_watch::add(base, end - base);
    return result;
}

// After a successful call that changes pages of an existing reservation without a write the kernel would see (a
// commit, which may bring back decommitted pages as zeros; MEM_RESET; a decommit with `decommit`): if it is
// write-watched, the shadows take [address, address + size) as written (size 0: to the reservation's end).
void note_unwatched_change(const void* address, SIZE_T size, DWORD type, bool decommit) {
    if (!address || !(decommit || (type & (MEM_COMMIT | MEM_RESET | MEM_RESET_UNDO))) || (type & MEM_RESERVE)) return;
    const auto from = reinterpret_cast<uintptr_t>(address);
    const uintptr_t to = size ? from + size : leonida::title_write_watch::end_of(from);
    if (to > from) {
        leonida::title_write_watch::mark(from, to);
        leonida::geometry_writes::notify(from, to, decommit ? leonida::geometry_writes::Event::replaced
                                                            : leonida::geometry_writes::Event::mapping);
    }
}

// PVOID XMemVirtualAlloc(address, size, allocationType, attributes, protect). All 23 calls of the measured title pass a
// nonzero attribute word; its bits are not decoded and nothing on the host depends on them, so it is accepted and only
// recorded (xmem_attributes). Its types cover committed 2 MiB and 64 KiB pages, reserve-only 64 KiB windows for physical pages, one
// MEM_RESERVE_PLACEHOLDER reservation and five MEM_REPLACE_PLACEHOLDER windows carved out of it, each then filled with
// XMemMapPhysicalPages. The guest page size sets the alignment and rounding of the region; the host's own pages stay
// 4 KiB (host large pages need SeLockMemoryPrivilege). A 2 MiB region the host places is reserved 2 MiB-aligned through
// VirtualAlloc2.
void* xmem_virtual_alloc(void* address, SIZE_T size, DWORD allocation_type, ULONGLONG attributes, DWORD protect);

void* WINAPI XMemVirtualAlloc(void* address, SIZE_T size, DWORD allocation_type, ULONGLONG attributes, DWORD protect) {
    note_attributes("XMemVirtualAlloc", attributes, size, _ReturnAddress());
    void* result = xmem_virtual_alloc(address, size, allocation_type, attributes, protect);
    // The title's memory (src/title_va_alias.h). Never forgotten: a released range is no longer committed, which
    // the copy also checks.
    if (result) {
        leonida::title_memory_map::changed();
        leonida::title_va_aliases().add_title_memory(reinterpret_cast<uintptr_t>(result), size);
        note_unwatched_change(address, size, allocation_type, false);
        if (allocation_type & MEM_RESERVE) {
            const auto from = reinterpret_cast<uintptr_t>(result);
            leonida::geometry_writes::notify(from, from + size, leonida::geometry_writes::Event::replaced);
        }
    }
    // A reservation: a GPU-only one (no CPU access, PAGE_GRAPHICS_* bits) is paired with the CPU-mappable one of the
    // same size made just before it (src/title_page_watch.h, title_gpu_views).
    const DWORD host_type = allocation_type & kXmemHostAllocationTypeMask;
    if (result && host_type && !(host_type & MEM_COMMIT)) {
        const bool gpu_only = (protect & kPageGraphicsMask) && host_protect(protect) == PAGE_NOACCESS;
        if (leonida::title_gpu_views::note_reservation(reinterpret_cast<uintptr_t>(result), size, gpu_only))
            std::fprintf(stderr, "[rt] xmem: GPU-only view %p (%llu MiB, protect 0x%lx) reads at the CPU view %p\n",
                         result, static_cast<unsigned long long>(size >> 20), static_cast<unsigned long>(protect),
                         reinterpret_cast<void*>(leonida::title_gpu_views::to_cpu(reinterpret_cast<uintptr_t>(result))));
        else if (gpu_only && size >= (64u << 20))
            std::fprintf(stderr, "[rt] xmem: GPU-only reservation %p (%llu MiB, protect 0x%lx) has no CPU view\n", result,
                         static_cast<unsigned long long>(size >> 20), static_cast<unsigned long>(protect));
    }
    return result;
}

void* xmem_virtual_alloc(void* address, SIZE_T size, DWORD allocation_type, ULONGLONG attributes, DWORD protect) {
    if (size == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    const DWORD page_mode = allocation_type & kXmemPageModeMask;
    if (page_mode != 0 && page_mode != kXmem64kPages && page_mode != kXmemLargePages) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    const DWORD unsupported = allocation_type & ~(kXmemHostAllocationTypeMask | kXmemPageModeMask);
    if (unsupported != 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    DWORD host_type = allocation_type & kXmemHostAllocationTypeMask;
    if (host_type == 0) host_type = MEM_RESERVE | MEM_COMMIT;
    const DWORD host_prot = host_protect(protect);

    const bool commit = (host_type & MEM_COMMIT) != 0;
    const bool new_placeholder = (host_type & MEM_RESERVE_PLACEHOLDER) != 0;
    const bool replace = (host_type & MEM_REPLACE_PLACEHOLDER) != 0;
    const bool window = page_mode != 0 && !commit && !new_placeholder;
    if ((new_placeholder || replace || window || page_mode == kXmemLargePages) && !placeholder_api().available()) {
        SetLastError(ERROR_NOT_SUPPORTED);
        return nullptr;
    }

    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    SIZE_T page_granularity = system_info.dwPageSize;
    SIZE_T base_granularity = (host_type & MEM_RESERVE) ? system_info.dwAllocationGranularity : system_info.dwPageSize;
    if (page_mode == kXmem64kPages) page_granularity = base_granularity = kXmemPageSize64k;
    if (page_mode == kXmemLargePages) page_granularity = base_granularity = kXmemPageSize2m;
    if (new_placeholder || replace) {
        // Placeholders are split and replaced on allocation-granularity boundaries only.
        if (page_granularity < system_info.dwAllocationGranularity) page_granularity = system_info.dwAllocationGranularity;
        if (base_granularity < system_info.dwAllocationGranularity) base_granularity = system_info.dwAllocationGranularity;
    }
    if (page_granularity == 0 || (page_granularity & (page_granularity - 1)) != 0 || base_granularity == 0 ||
        (base_granularity & (base_granularity - 1)) != 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    if (address != nullptr) {
        const uintptr_t requested = reinterpret_cast<uintptr_t>(address);
        const uintptr_t max = (std::numeric_limits<uintptr_t>::max)();
        if (size > max - requested) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        const uintptr_t requested_end = requested + size;
        const uintptr_t base = requested & ~(static_cast<uintptr_t>(base_granularity) - 1);
        if (requested_end > max - (static_cast<uintptr_t>(page_granularity) - 1)) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        const uintptr_t end = (requested_end + static_cast<uintptr_t>(page_granularity) - 1) &
                              ~(static_cast<uintptr_t>(page_granularity) - 1);
        address = reinterpret_cast<void*>(base);
        size = static_cast<SIZE_T>(end - base);
    } else if (replace) {
        SetLastError(ERROR_INVALID_PARAMETER);  // a replacement names the placeholder it replaces
        return nullptr;
    } else if (page_granularity > system_info.dwPageSize && !round_up_size(size, page_granularity, size)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    if (replace) {
        const std::lock_guard lock(g_phys_mutex);
        if (!isolate_placeholder(address, size)) return nullptr;
        if (window) {
            // A reserve-only replacement stays a placeholder: the physical pages are mapped into it.
            record_window(address, size, host_prot);
            return address;
        }
        return placeholder_api().virtual_alloc2(GetCurrentProcess(), address, size, host_type | MEM_RESERVE, host_prot,
                                                nullptr, 0);
    }
    if (new_placeholder || window) {
        void* const result = reserve_placeholder(address, size, base_granularity);
        if (result && window) {
            const std::lock_guard lock(g_phys_mutex);
            record_window(result, size, host_prot);
        }
        return result;
    }
    if (page_mode == kXmemLargePages && address == nullptr) {
        MEM_ADDRESS_REQUIREMENTS requirements{};
        requirements.Alignment = kXmemPageSize2m;
        MEM_EXTENDED_PARAMETER parameter{};
        parameter.Type = MemExtendedParameterAddressRequirements;
        parameter.Pointer = &requirements;
        return reserve_watched(host_type | MEM_RESERVE, nullptr, size, [&](DWORD type) {
            return placeholder_api().virtual_alloc2(GetCurrentProcess(), nullptr, size, type, host_prot, &parameter, 1);
        });
    }
    return reserve_watched(host_type, address, size,
                           [&](DWORD type) { return VirtualAlloc(address, size, type, host_prot); });
}

// With g_phys_mutex held. Maps `view` at the 64 KiB-aligned `address`, which holds no mapped page. MEM_REPLACE_PLACEHOLDER
// needs the view to fill one placeholder exactly, and earlier partial unmaps can leave several host placeholders in one
// run, so the run is mapped as one view per placeholder it spans. Views are capped at kMaxViewPages so that unmapping a
// page inside a long run only remaps the rest of one capped view, not the whole run.
constexpr uint64_t kMaxViewPages = 1024;  // 64 MiB


bool map_view(uintptr_t address, const View& view) {
    for (uint64_t mapped = 0; mapped < view.count;) {
        const uintptr_t base = address + mapped * kPhysicalPageSize;
        void* const target = reinterpret_cast<void*>(base);
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(target, &info, sizeof(info)) || info.State != MEM_RESERVE ||
            info.RegionSize < kPhysicalPageSize) {
            SetLastError(ERROR_INVALID_ADDRESS);
            return false;
        }
        const uint64_t available = info.RegionSize / kPhysicalPageSize;
        const uint64_t remaining = view.count - mapped;
        const uint64_t count = (std::min)({available, remaining, kMaxViewPages});
        const SIZE_T size = static_cast<SIZE_T>(count * kPhysicalPageSize);
        const uint64_t offset = view.offset + mapped * kPhysicalPageSize;
        if (!isolate_placeholder(target, size)) return false;
        if (!placeholder_api().map_view3(view.section->handle, GetCurrentProcess(), target, offset, size,
                                         MEM_REPLACE_PLACEHOLDER, view.protect, nullptr, 0)) {
            return false;
        }
        leonida::title_memory_map::changed();
        g_views.emplace(base, View{count, view.section, offset, view.protect});
        mapped += count;
    }
    return true;
}

// With g_phys_mutex held. Unmaps every page in [start, end); the addresses become placeholders again. A view that
// reaches outside the range is unmapped whole, and its pages outside the range are mapped again, so neighbours keep
// their pages.
// There is no rollback: if mapping a remainder again fails, those neighbouring pages stay unmapped and the caller only
// sees the failure.
bool unmap_range(uintptr_t start, uintptr_t end) {
    const FaultLockExclusive lock;  // faults on neighbouring pages wait through the split and remap
    auto it = g_views.upper_bound(start);
    if (it != g_views.begin()) {
        const auto prev = std::prev(it);
        if (prev->first + prev->second.count * kPhysicalPageSize > start) it = prev;
    }
    while (it != g_views.end() && it->first < end) {
        const uintptr_t base = it->first;
        const View view = it->second;
        const uintptr_t view_end = base + view.count * kPhysicalPageSize;
        if (!placeholder_api().unmap_view2(GetCurrentProcess(), reinterpret_cast<void*>(base),
                                           MEM_PRESERVE_PLACEHOLDER)) {
            return false;
        }
        leonida::title_memory_map::changed();
        leonida::geometry_writes::notify(base, view_end, leonida::geometry_writes::Event::replaced);
        it = g_views.erase(it);
        // The pages of [start, end) this view held are unmapped now; the ones outside it are mapped again below.
        for (uintptr_t at = (std::max)(base, start); at < (std::min)(view_end, end); at += kPhysicalPageSize) {
            if (const auto mapped = g_mapped.find(at); mapped != g_mapped.end()) {
                note_mapping(mapped->second, at, false);
                g_mapped.erase(mapped);
            }
        }
        if (base < start) {
            View left = view;
            left.count = (start - base) / kPhysicalPageSize;
            if (!map_view(base, left)) return false;
        }
        if (view_end > end) {
            const uint64_t skipped = (end - base) / kPhysicalPageSize;
            View right = view;
            right.count = view.count - skipped;
            right.offset += skipped * kPhysicalPageSize;
            return map_view(end, right);  // the last view the range touches
        }
    }
    return true;
}

bool unmap_page(uintptr_t address) { return unmap_range(address, address + kPhysicalPageSize); }

// With g_phys_mutex held. Maps physical page `page` (0: unmap) at the 64 KiB-aligned `address`, replacing what was
// mapped there. The caller has found the window.
bool map_page(uintptr_t address, uint64_t page, DWORD protect) {
    if (!unmap_page(address)) return false;
    if (page == 0) return true;
    const auto it = g_pages.find(page);
    if (it == g_pages.end()) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    {
        const FaultLockExclusive lock;
        if (!map_view(address, View{1, it->second.section, it->second.offset, protect})) return false;
    }
    g_mapped[address] = page;
    note_mapping(page, address, true);
    return true;
}

enum class PageCommit { not_ours, committed_before, committed_now, failed };

PageCommit commit_mapped_page(uintptr_t address, uintptr_t accessed, DWORD& protect) {
    auto it = g_views.upper_bound(address);
    if (it == g_views.begin()) return PageCommit::not_ours;
    --it;
    const uintptr_t within = address - it->first;
    if (within / kPhysicalPageSize >= it->second.count || !it->second.section->scratch) return PageCommit::not_ours;
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(reinterpret_cast<void*>(accessed), &info, sizeof(info)) || info.Type != MEM_MAPPED) {
        return PageCommit::not_ours;
    }
    if (info.State == MEM_COMMIT) {
        protect = info.Protect;
        return PageCommit::committed_before;
    }
    void* const backing = static_cast<uint8_t*>(it->second.section->scratch) + it->second.offset + within;
    if (!VirtualAlloc(backing, kPhysicalPageSize, MEM_COMMIT, PAGE_READWRITE)) return PageCommit::failed;
    leonida::geometry_writes::notify(address, address + kPhysicalPageSize, leonida::geometry_writes::Event::mapping);
    leonida::title_memory_map::changed();
    return PageCommit::committed_now;
}

bool access_allowed(DWORD protect, ULONG_PTR access) {
    if (protect & PAGE_GUARD) return false;
    switch (protect & 0xff) {
    case PAGE_EXECUTE_READWRITE:
    case PAGE_EXECUTE_WRITECOPY:
        return true;
    case PAGE_READWRITE:
    case PAGE_WRITECOPY:
        return access != 8;
    case PAGE_EXECUTE_READ:
        return access != 1;
    case PAGE_READONLY:
        return access == 0;
    case PAGE_EXECUTE:
        return access == 8;
    default:
        return false;
    }
}

LONG CALLBACK physical_page_fault(EXCEPTION_POINTERS* info) {
    thread_local bool busy = false;
    const EXCEPTION_RECORD* record = info ? info->ExceptionRecord : nullptr;
    if (busy || !record || record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    busy = true;
    const uintptr_t accessed = static_cast<uintptr_t>(record->ExceptionInformation[1]);
    const uintptr_t address = accessed & ~static_cast<uintptr_t>(kPhysicalPageSize - 1);
    DWORD protect = 0;
    PageCommit result;
    {
        const FaultLockShared lock;
        result = commit_mapped_page(address, accessed, protect);
    }
    busy = false;
    if (result == PageCommit::committed_now) return EXCEPTION_CONTINUE_EXECUTION;
    if (result == PageCommit::committed_before && access_allowed(protect, record->ExceptionInformation[0])) {
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

bool install_physical_page_fault_handler() {
    static const bool installed = AddVectoredExceptionHandler(1, &physical_page_fault) != nullptr;
    return installed;
}

// Two of the measured windows are reserved PAGE_NOACCESS for the CPU and the GPU alike; pages mapped there are still
// GPU memory, which the D3D12 layer reads on the CPU.
DWORD view_protect(const Window& window) {
    // A view takes no cache modifier. Three of the measured windows carry PAGE_WRITECOMBINE (0x400) next to
    // PAGE_READWRITE: Wine's MapViewOfFile3 refuses it, and PAGE_NOCACHE, with ERROR_INVALID_PARAMETER, and on Windows
    // the view comes out write-combined, which an import of those pages as a GPU buffer refuses (#7218). PAGE_GUARD
    // is an access alarm, not a cache modifier, so it is left as the window has it.
    const DWORD protect = window.protect & ~static_cast<DWORD>(PAGE_WRITECOMBINE | PAGE_NOCACHE);
    return protect == PAGE_NOACCESS ? PAGE_READWRITE : protect;
}

bool aligned_page_address(uintptr_t address) { return (address & (kPhysicalPageSize - 1)) == 0; }

// gta6rt --trace FILE: the first XMemAllocatePhysicalPages for each distinct attribute word goes to the startup trace
// as an "xmem_physical_pages" event with the page count and the caller.
void note_physical_attributes(uint64_t attributes, uint64_t count, const void* caller) {
    if (!rt::startup_trace::enabled()) return;
    static std::mutex mutex;
    static std::set<uint64_t> seen;
    {
        const std::lock_guard lock(mutex);
        if (!seen.insert(attributes).second) return;
    }
    rt::startup_trace::event("xmem_physical_pages",
                             "\"api\":\"XMemAllocatePhysicalPages\",\"attributes\":" +
                                 rt::startup_trace::quote(rt::startup_trace::hex(attributes)) +
                                 ",\"pages\":" + std::to_string(count) + caller_field(caller));
}

// BOOL XMemAllocatePhysicalPages(attributes, UINT64* pageCount, UINT64* pageArray). Measured at four call sites: RCX is
// a constant attribute word, RDX points at the number of 64 KiB pages (every caller computes it as the byte size
// rounded up and shifted right by 16), R8 at an array that receives one page number per page. The result is tested as
// a 32-bit BOOL. After success every caller compares the count with the one it asked for and frees the pages if they
// differ, so the count is in/out.
//
// Each call is backed by one pagefile-backed section of count * 64 KiB: committed (charged) on touch, given physical
// memory as it is touched, zero-filled. Pages keep their contents while unmapped or mapped elsewhere, and one page can
// be mapped at several addresses. ASSUMPTION: the attribute word (memory type, perhaps the page size) is not decoded;
// every page is 64 KiB, which all four callers' arithmetic agrees with.
BOOL WINAPI XMemAllocatePhysicalPages(uint64_t attributes, uint64_t* page_count, uint64_t* page_array) {
    note_physical_attributes(attributes, page_count ? *page_count : 0, _ReturnAddress());
    if (!page_count || (*page_count && !page_array)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const uint64_t count = *page_count;
    if (count == 0) return TRUE;

    const std::lock_guard lock(g_phys_mutex);
    if (count > g_free_slots.size() + (kPageSlotLimit - g_next_slot)) {
        *page_count = 0;
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return FALSE;
    }
    if (!install_physical_page_fault_handler()) {
        *page_count = 0;
        SetLastError(ERROR_NOT_SUPPORTED);
        return FALSE;
    }
    const uint64_t bytes = count * kPhysicalPageSize;  // below 2^37: count is below 2^21
    const HANDLE handle = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE,
                                             static_cast<DWORD>(bytes >> 32), static_cast<DWORD>(bytes), nullptr);
    if (!handle) {
        *page_count = 0;
        return FALSE;
    }
    // From host_heap: the unmap that drops the last reference frees it under g_fault_lock.
    const auto section = std::allocate_shared<Section>(leonida::host_heap::Allocator<Section>{}, handle);
    if (!section->scratch) {
        *page_count = 0;
        return FALSE;
    }
    for (uint64_t i = 0; i < count; ++i) {
        uint64_t slot;
        if (!g_free_slots.empty()) {
            slot = g_free_slots.back();
            g_free_slots.pop_back();
        } else {
            slot = g_next_slot++;
        }
        const uint64_t page = slot * kFramesPerPage;
        g_pages[page] = PhysicalPage{section, i * kPhysicalPageSize};
        page_array[i] = page;
    }
    return TRUE;
}

// BOOL XMemFreePhysicalPages(UINT64 pageCount, UINT64* pageArray). Measured at three call sites: RCX is the count (the
// one XMemAllocatePhysicalPages returned, or the sum of several allocations), RDX the array. The callers unmap the
// pages first; a page that is still mapped keeps its memory until it is unmapped. FALSE, with the entries before it
// freed, for a number that is not a live page.
BOOL WINAPI XMemFreePhysicalPages(uint64_t page_count, const uint64_t* page_array) {
    if (page_count && !page_array) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const std::lock_guard lock(g_phys_mutex);
    for (uint64_t i = 0; i < page_count; ++i) {
        const auto it = g_pages.find(page_array[i]);
        if (it == g_pages.end()) {
            if (i != 0) leonida::title_memory_map::changed();
            SetLastError(ERROR_INVALID_PARAMETER);
            return FALSE;
        }
        g_pages.erase(it);
        g_free_slots.push_back(page_array[i] / kFramesPerPage);
    }
    if (page_count != 0) leonida::title_memory_map::changed();
    return TRUE;
}

// PVOID XMemMapPhysicalPages(PVOID address, UINT64 pageCount, UINT64* pageArray). Measured at twelve call sites: RCX
// is an address in a window XMemVirtualAlloc reserved, RDX a 64-bit count, R8 the pages to map there one after another,
// or null to unmap that many (four calls at shutdown). Callers test the 64-bit result for zero, so it is the address on
// success. A page number of 0 in the array unmaps that one page.
//
// Consecutive pages of one allocation are mapped as one view (MapViewOfFile3 into a placeholder of their size); a
// page remapped or unmapped later splits its view, and its neighbours stay mapped. The range is unmapped first, so a
// page number that is not a live page fails with the pages before it mapped and the rest of the range unmapped.
void* WINAPI XMemMapPhysicalPages(void* address, uint64_t page_count, const uint64_t* page_array) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(address);
    if (!aligned_page_address(start) || page_count > (std::numeric_limits<uintptr_t>::max)() / kPhysicalPageSize ||
        start > (std::numeric_limits<uintptr_t>::max)() - page_count * kPhysicalPageSize) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    if (!placeholder_api().available()) {
        SetLastError(ERROR_NOT_SUPPORTED);
        return nullptr;
    }
    const std::lock_guard lock(g_phys_mutex);
    const uintptr_t end = start + page_count * kPhysicalPageSize;
    if (!page_array) return unmap_range(start, end) ? address : nullptr;
    const Window* const window = find_window(start, page_count * kPhysicalPageSize);
    if (!window) {
        SetLastError(ERROR_INVALID_ADDRESS);
        return nullptr;
    }
    const DWORD protect = view_protect(*window);
    if (!unmap_range(start, end)) return nullptr;
    for (uint64_t i = 0; i < page_count;) {
        if (page_array[i] == 0) {
            ++i;
            continue;
        }
        const auto it = g_pages.find(page_array[i]);
        if (it == g_pages.end()) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        View view{1, it->second.section, it->second.offset, protect};
        while (i + view.count < page_count) {
            const auto next = g_pages.find(page_array[i + view.count]);
            if (next == g_pages.end() || next->second.section != view.section ||
                next->second.offset != view.offset + view.count * kPhysicalPageSize) {
                break;
            }
            ++view.count;
        }
        {
            const FaultLockExclusive view_lock;
            if (!map_view(start + i * kPhysicalPageSize, view)) return nullptr;
        }
        // After the fault lock: page_array is the title's memory, and a read of it that faults would wait for the
        // lock on this thread. g_mapped and the aliases are g_phys_mutex's; the fault handler reads neither.
        for (uint64_t k = 0; k < view.count; ++k) {
            const uintptr_t at = start + (i + k) * kPhysicalPageSize;
            g_mapped[at] = page_array[i + k];
            note_mapping(page_array[i + k], at, true);
        }
        i += view.count;
    }
    return address;
}

// BOOL XMemMapPhysicalPagesScatter(PVOID* addresses, UINT32 pageCount, UINT64* pageArray). Measured at two call sites:
// RCX is an array of addresses, EDX a 32-bit count, R8 the page for each address (an entry of 0, or a null array,
// unmaps). The result is tested as a 32-bit BOOL. Like MapUserPhysicalPagesScatter: entry i maps pageArray[i] at
// addresses[i].
BOOL WINAPI XMemMapPhysicalPagesScatter(void* const* addresses, uint32_t page_count, const uint64_t* page_array) {
    if (page_count && !addresses) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    if (!placeholder_api().available()) {
        SetLastError(ERROR_NOT_SUPPORTED);
        return FALSE;
    }
    const std::lock_guard lock(g_phys_mutex);
    for (uint32_t i = 0; i < page_count; ++i) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(addresses[i]);
        if (!aligned_page_address(address)) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return FALSE;
        }
        const uint64_t page = page_array ? page_array[i] : 0;
        if (page == 0) {
            if (!unmap_page(address)) return FALSE;
            continue;
        }
        const Window* const window = find_window(address, kPhysicalPageSize);
        if (!window) {
            SetLastError(ERROR_INVALID_ADDRESS);
            return FALSE;
        }
        if (!map_page(address, page, view_protect(*window))) return FALSE;
    }
    return TRUE;
}

// rt::xmem::copy_page_mappings_fn (xmem.h), for the console queue's CopyPageMappingsBatchX: each destination page gets
// a view of the physical page mapped at its source page, so both addresses are one memory, as two XMemMapPhysicalPages
// calls with the same page make them. Every source page is looked up before any destination page changes, so a source
// range that overlaps the destination maps what was there before the call. A destination in a GPU-only range is mapped
// at its paired CPU range too (title_gpu_views), and a source page missing at a GPU-only address is looked up there.
bool copy_page_mappings(void* destination, const void* source, uint64_t pages, bool repeat_first) {
    const uintptr_t to = reinterpret_cast<uintptr_t>(destination);
    const uintptr_t from = reinterpret_cast<uintptr_t>(source);
    constexpr uintptr_t kMax = (std::numeric_limits<uintptr_t>::max)();
    if (!aligned_page_address(to) || !aligned_page_address(from) || pages > kMax / kPhysicalPageSize ||
        to > kMax - pages * kPhysicalPageSize || (!repeat_first && from > kMax - pages * kPhysicalPageSize)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    if (!placeholder_api().available()) {
        SetLastError(ERROR_NOT_SUPPORTED);
        return false;
    }
    const std::lock_guard lock(g_phys_mutex);
    // The destination is usually a GPU-only reservation, which the D3D12 layer reads at its paired CPU view
    // (title_gpu_views::to_cpu); that view gets the same pages, or it would stay reserved and never readable.
    const uintptr_t cpu_to = leonida::title_gpu_views::to_cpu(to, pages * kPhysicalPageSize);
    const Window* const cpu_window = cpu_to != to ? find_window(cpu_to, pages * kPhysicalPageSize) : nullptr;
    if (!source) {
        for (uint64_t i = 0; i < pages; ++i) {
            if (!unmap_page(to + i * kPhysicalPageSize)) return false;
            if (cpu_window && !unmap_page(cpu_to + i * kPhysicalPageSize)) return false;
        }
        return true;
    }
    const Window* const window = find_window(to, pages * kPhysicalPageSize);
    if (!window) {
        SetLastError(ERROR_INVALID_ADDRESS);
        return false;
    }
    std::vector<uint64_t> found(static_cast<size_t>(pages), 0);
    for (uint64_t i = 0; i < pages; ++i) {
        const uintptr_t src_va = from + (repeat_first ? 0 : i) * kPhysicalPageSize;
        auto it = g_mapped.find(src_va);
        if (it == g_mapped.end()) {  // a GPU-only source whose pages were mapped at its CPU view
            const uintptr_t cpu_src = leonida::title_gpu_views::to_cpu(src_va, kPhysicalPageSize);
            if (cpu_src != src_va) it = g_mapped.find(cpu_src);
        }
        if (it != g_mapped.end()) found[static_cast<size_t>(i)] = it->second;
    }
    const DWORD protect = view_protect(*window);
    const DWORD cpu_protect = cpu_window ? view_protect(*cpu_window) : 0;
    for (uint64_t i = 0; i < pages; ++i) {
        const uint64_t page = found[static_cast<size_t>(i)];
        if (!map_page(to + i * kPhysicalPageSize, page, protect)) return false;
        if (cpu_window && !map_page(cpu_to + i * kPhysicalPageSize, page, cpu_protect)) return false;
    }
    return true;
}
[[maybe_unused]] const bool g_copy_page_mappings_set =(rt::xmem::copy_page_mappings_fn = &copy_page_mappings, true);

// The title's write watching, on reservations it made with MEM_WRITE_WATCH. One xmem write-watched for the shadows
// (reserve_watched) is refused as the host refuses a region that is not write-watched ((UINT)-1,
// ERROR_INVALID_PARAMETER), as the console would: it isn't write-watched there, and a reset here would clear the bits
// the shadows ask about.
bool shadows_write_watch(const void* base_address) {
    if (!leonida::title_write_watch::contains(reinterpret_cast<uintptr_t>(base_address))) return false;
    SetLastError(ERROR_INVALID_PARAMETER);
    return true;
}

UINT WINAPI XMemGetWriteWatch(DWORD flags, void* base_address, SIZE_T region_size, void** addresses, ULONG_PTR* count,
                              ULONG* granularity) {
    if (shadows_write_watch(base_address)) return UINT(-1);
    return GetWriteWatch(flags, base_address, region_size, addresses, count, granularity);
}

UINT WINAPI XMemResetWriteWatch(void* base_address, SIZE_T region_size) {
    if (shadows_write_watch(base_address)) return UINT(-1);
    return ResetWriteWatch(base_address, region_size);
}

// BOOL XMemTransferMemory(XMEM_TRANSFER_TYPE type, SIZE_T size). Its reference page and the XMEM_TRANSFER_TYPE enum
// page (gaming/gdk/docs/reference/system/xmem/{functions/xmemtransfermemory,enums/xmem_transfer_type}) are NDA
// topics on Microsoft Learn, so the shape comes from public code built against the GDK's xmem.h: Microsoft's
// XBOX-MonoGame-NetRumble sample (console/NetRumble.Bootstrap/Main.cpp) and FNA-XNA/NativeAOT-Xbox (Bootstrap/Main.cpp)
// both call `if (!XMemTransferMemory(XMemTransferTitleToSystem, additionalKernelPoolMemory))` with a `const size_t`,
// commented "Transfer additional memory to kernel pool" and "Must be multiples of 2MB sizes". So: two arguments, the
// enum (32 bits) in ECX and the size in RDX, and a zero return is failure. The public XLaunchNewGame page adds that
// "Memory transferred to the system using XMemTransferMemory (NDA topic) is persisted", and the game-save debugging
// page suggests it for E_OUTOFMEMORY "to transfer some memory to the system partition".
//
// On the console it moves budget between the title's partition and the system's. A PC has one pool and no budget
// split, so there is nothing to move: every direction and size returns TRUE and nothing is allocated or reserved. Not
// known (the pages are NDA): the numeric XMEM_TRANSFER_TYPE values, and whether the console refuses or rounds a size
// that is not a multiple of 2 MiB. Refusing would stop a title at startup (both samples return -1), while succeeding
// changes nothing here. TRUE is 1 in EAX, so it also reads as true if the real return type is a one-byte bool.
//
// gta6rt --trace FILE: the first call for each distinct (type, size) pair goes to the startup trace as an
// "xmem_transfer" event with the caller as module+RVA, so a run with the executable records the values it passes.
BOOL WINAPI XMemTransferMemory(uint32_t type, SIZE_T size) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] xmem: XMemTransferMemory(type %u, %zu bytes): a PC has no title/system memory partition; "
                     "returning TRUE without moving anything (logged once)\n",
                     type, static_cast<size_t>(size));
    }
    if (rt::startup_trace::enabled()) {
        static std::mutex mutex;
        static std::set<std::pair<uint32_t, uint64_t>> seen;
        bool first = false;
        {
            const std::lock_guard lock(mutex);
            first = seen.insert({type, static_cast<uint64_t>(size)}).second;
        }
        if (first) {
            rt::startup_trace::event(
                "xmem_transfer",
                "\"api\":\"XMemTransferMemory\",\"type\":" + std::to_string(type) +
                    ",\"size\":" + std::to_string(static_cast<uint64_t>(size)) + ",\"caller\":" +
                    rt::startup_trace::quote(
                        rt::startup_trace::where(reinterpret_cast<uint64_t>(_ReturnAddress()))));
        }
    }
    return TRUE;
}

// kernel32!VirtualAlloc as a title calls it. The console's allocation type has two bits of its own, MEM_GRAPHICS
// (0x10000000, memory the GPU reads) and MEM_TITLE (0x40000000, the title's own heap). On a PC the host rejects both
// with ERROR_INVALID_PARAMETER (measured on Windows 11 26200: VirtualAlloc(NULL, 64 KiB, MEM_COMMIT | MEM_RESERVE |
// 0x10000000, PAGE_READWRITE) returns NULL; so does 0x40000000), so an engine allocating its heaps this way gets
// NULL at boot. Both are hints about where the memory lives; the host has no such split, so they are dropped and
// the rest is passed through. The PAGE_GRAPHICS_* protection bits are dropped (host_protect). Not done: whether
// graphics memory has to come from an arena the D3D12 layer can see (#313).
constexpr DWORD kMemGraphics = 0x10000000;
constexpr DWORD kMemTitle = 0x40000000;

// A new reservation is write-watched for the D3D12 layer's shadows (reserve_watched).
LPVOID WINAPI TitleVirtualAlloc(LPVOID address, SIZE_T size, DWORD allocation_type, DWORD protect) {
    const DWORD type = allocation_type & ~(kMemGraphics | kMemTitle);
    const DWORD host_prot = host_protect(protect);
    LPVOID result =
        reserve_watched(type, address, size, [&](DWORD t) { return VirtualAlloc(address, size, t, host_prot); });
    if (result) {
        leonida::title_memory_map::changed();
        leonida::title_va_aliases().add_title_memory(reinterpret_cast<uintptr_t>(result), size);
        note_unwatched_change(address, size, type, false);
        if (type & MEM_RESERVE) {
            const auto from = reinterpret_cast<uintptr_t>(result);
            leonida::geometry_writes::notify(from, from + size, leonida::geometry_writes::Event::replaced);
        }
    }
    return result;
}

BOOL WINAPI TitleVirtualFree(LPVOID address, SIZE_T size, DWORD free_type) {
    const BOOL result = VirtualFree(address, size, free_type);
    if (result) leonida::title_memory_map::changed();
    if (result && (free_type & MEM_DECOMMIT)) note_unwatched_change(address, size, 0, true);
    if (result && (free_type & MEM_RELEASE) && !(free_type & MEM_PRESERVE_PLACEHOLDER))
        leonida::title_write_watch::remove(reinterpret_cast<uintptr_t>(address));
    return result;
}

BOOL WINAPI TitleVirtualProtect(LPVOID address, SIZE_T size, DWORD protect, PDWORD old) {
    const BOOL result = VirtualProtect(address, size, protect, old);
    if (result) {
        leonida::title_memory_map::changed();
        const auto start = reinterpret_cast<uintptr_t>(address);
        leonida::geometry_writes::notify(start, start + size, leonida::geometry_writes::Event::replaced);
    }
    return result;
}

}  // namespace

namespace rt::xmem {

void geometry_aliases(std::uintptr_t va, std::vector<std::uintptr_t>& addresses) {
    std::lock_guard lock(g_phys_mutex);
    const auto mapped = g_mapped.find(va);
    if (mapped == g_mapped.end()) return;
    const auto peers = g_page_addresses.find(mapped->second);
    if (peers != g_page_addresses.end())
        for (auto other : peers->second) if (other != va) addresses.push_back(other);
}

bool commit_physical_range(const void* address, size_t size) {
    if (size == 0) return true;
    const uintptr_t start = reinterpret_cast<uintptr_t>(address);
    if (start > (std::numeric_limits<uintptr_t>::max)() - (size - 1)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    const uintptr_t mask = ~static_cast<uintptr_t>(kPhysicalPageSize - 1);
    const uintptr_t first = start & mask;
    const uintptr_t pages = (((start + (size - 1)) & mask) - first) / kPhysicalPageSize + 1;
    const FaultLockShared lock;
    for (uintptr_t i = 0; i < pages; ++i) {
        DWORD protect = 0;
        const uintptr_t page = first + i * kPhysicalPageSize;
        if (commit_mapped_page(page, page, protect) == PageCommit::failed) return false;
    }
    return true;
}

static bool commit_pages_locked(const uint64_t* page_array, size_t count) {
    if (count == 0) return true;
    if (!page_array) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    size_t first = 0;
    while (first < count) {
        const auto it = g_pages.find(page_array[first]);
        if (it == g_pages.end() || !it->second.section->scratch) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return false;
        }
        size_t last = first + 1;
        while (last < count) {
            const auto next = g_pages.find(page_array[last]);
            if (next == g_pages.end() || next->second.section != it->second.section ||
                next->second.offset != it->second.offset + (last - first) * kPhysicalPageSize) {
                break;
            }
            ++last;
        }
        void* const backing = static_cast<uint8_t*>(it->second.section->scratch) + it->second.offset;
        if (!VirtualAlloc(backing, (last - first) * kPhysicalPageSize, MEM_COMMIT, PAGE_READWRITE)) return false;
        leonida::title_memory_map::changed();
        first = last;
    }
    return true;
}

bool commit_physical_pages(const uint64_t* page_array, size_t count) {
    const std::lock_guard lock(g_phys_mutex);
    return commit_pages_locked(page_array, count);
}

bool transfer_physical(std::span<const uint64_t> pages, uint16_t offset, std::span<uint8_t> staging, bool write) {
    if (staging.empty()) return true;
    if (pages.size() != 1 + (staging.size() - 1 + offset) / kPhysicalPageSize) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    const std::lock_guard lock(g_phys_mutex);
    if (!commit_pages_locked(pages.data(), pages.size())) return false;
    for (const auto page : pages) {
        const auto& backing = g_pages.at(page);
        auto* ptr = static_cast<uint8_t*>(backing.section->scratch) + backing.offset + offset;
        const size_t bytes = (std::min)(staging.size(), size_t(kPhysicalPageSize - offset));
        if (write) {
            std::memcpy(ptr, staging.data(), bytes);
            if (auto mapped = g_page_addresses.find(page); mapped != g_page_addresses.end()) {
                for (uintptr_t address : mapped->second) {
                    leonida::geometry_writes::notify(address + offset, address + offset + bytes, leonida::geometry_writes::Event::write);
                    leonida::title_texel_writes::note(reinterpret_cast<void*>(address + offset), bytes);
                    if (leonida::dstorage::trace_enabled())
                        std::fprintf(stderr, "[rt] ds-scatter cpu=%llx bytes=%zu page=%llu\n",
                                     static_cast<unsigned long long>(address + offset), bytes,
                                     static_cast<unsigned long long>(page));
                }
            }
        } else {
            std::memcpy(staging.data(), ptr, bytes);
        }
        staging = staging.subspan(bytes);
        offset = 0;
    }
    return true;
}

}

#define TITLE_VIRTUAL_MEMORY(dll)                                                               \
    {dll, "VirtualAlloc", reinterpret_cast<void*>(&TitleVirtualAlloc)},                          \
    {dll, "VirtualFree", reinterpret_cast<void*>(&TitleVirtualFree)},                            \
    {dll, "VirtualProtect", reinterpret_cast<void*>(&TitleVirtualProtect)}

// The classic libraries and the memory api sets a title's imports name.
SHIM_TABLE(title_memory,
    TITLE_VIRTUAL_MEMORY("kernel32.dll"),
    TITLE_VIRTUAL_MEMORY("kernelbase.dll"),
    TITLE_VIRTUAL_MEMORY("api-ms-win-core-memory-l1-1-0.dll"),
    TITLE_VIRTUAL_MEMORY("api-ms-win-core-memory-l1-1-1.dll"),
    TITLE_VIRTUAL_MEMORY("api-ms-win-core-memory-l1-1-2.dll"),
    TITLE_VIRTUAL_MEMORY("api-ms-win-core-memory-l1-1-3.dll"),
    TITLE_VIRTUAL_MEMORY("api-ms-win-core-memory-l1-1-4.dll"),
    TITLE_VIRTUAL_MEMORY("api-ms-win-core-memory-l1-1-5.dll"),
    TITLE_VIRTUAL_MEMORY("api-ms-win-core-memory-l1-1-6.dll"),
);

SHIM_TABLE(xmem,
    {"xmem.dll", "XMemAlloc", reinterpret_cast<void*>(&XMemAlloc)},
    {"xmem.dll", "XMemFree", reinterpret_cast<void*>(&XMemFree)},
    {"xmem.dll", "XMemVirtualQuery", reinterpret_cast<void*>(&XMemVirtualQuery)},
    {"xmem.dll", "XMemGetWorkingSetStatistics", reinterpret_cast<void*>(&XMemGetWorkingSetStatistics)},
    {"xmem.dll", "XMemVirtualAlloc", reinterpret_cast<void*>(&XMemVirtualAlloc)},
    {"xmem.dll", "XMemGetWriteWatch", reinterpret_cast<void*>(&XMemGetWriteWatch)},
    {"xmem.dll", "XMemResetWriteWatch", reinterpret_cast<void*>(&XMemResetWriteWatch)},
    {"xmem.dll", "XMemTransferMemory", reinterpret_cast<void*>(&XMemTransferMemory)},
    {"xmem.dll", "XMemAllocDefault", reinterpret_cast<void*>(&XMemAllocDefault)},
    {"xmem.dll", "XMemFreeDefault", reinterpret_cast<void*>(&XMemFreeDefault)},
    {"xmem.dll", "XMemIsToolsMemAvailable", reinterpret_cast<void*>(&XMemIsToolsMemAvailable)},
    {"xmem.dll", "XMemAllocatePhysicalPages", reinterpret_cast<void*>(&XMemAllocatePhysicalPages)},
    {"xmem.dll", "XMemFreePhysicalPages", reinterpret_cast<void*>(&XMemFreePhysicalPages)},
    {"xmem.dll", "XMemMapPhysicalPages", reinterpret_cast<void*>(&XMemMapPhysicalPages)},
    {"xmem.dll", "XMemMapPhysicalPagesScatter", reinterpret_cast<void*>(&XMemMapPhysicalPagesScatter)},
);
