#pragma once
// What the D3D12 layer's slot-133 shadow buffers (ShadowPages in d3d12_vulkan_device.cpp) need from the rest of the
// runtime to see every write to the title's pages.
//
// A shadow notices the title's writes by asking the kernel which pages were written, where xmem reserved them with
// MEM_WRITE_WATCH (title_write_watch, below), and elsewhere by making the buffer's pages read-only between
// submissions: a write faults, the layer's handler makes the page writable again and the page is copied at the next
// submission. Two kinds of write don't fault there, and are reported here instead:
//
// 1. Aliases (title_page_aliases). xmem maps one physical page at as many VAs as the title asks
//    (XMemMapPhysicalPages): a write through one VA changes what the others read without touching them. xmem
//    reports each 64 KiB page mapped at more than one address; a shadow copies those at every submission.
//
// 2. Kernel writes (kernel_writes). A read-only page makes ReadFile fail with ERROR_NOACCESS instead of faulting
//    into a user-mode handler. The file shims (read_file and the rest, below) pin the destination before the kernel
//    writes it: the layer makes its pages writable and doesn't protect them again until the pin ends, when the
//    call returns or, for an overlapped read, once its OVERLAPPED says it completed. A read that still fails with
//    ERROR_NOACCESS is retried once after the destination is opened again and xmem has committed the physical pages
//    mapped there that wait for their first touch (they are reserved until then, and a kernel write does not fault
//    them in either: measured on Windows 11, the title's 258 MiB effect-database read failed this way forever).
//
//    Any other host code that has the kernel write title memory (a DirectStorage queue that ReadFiles straight into
//    a request's destination, for one) does the same: kernel_writes::read_file instead of ::ReadFile, or
//    begin_write(destination, bytes, overlapped) before the call and end_write(id, pending) after it. A user-mode
//    copy into title memory needs nothing: it faults like the title's own writes.
//
// Successful title memory map changes also invalidate the shadows' cached protection state. The shadows' own
// protection changes stay local to their trackers.
//
// The shadows themselves are brought up to date in one place, DeviceState::refresh_shadows in
// d3d12_vulkan_device.cpp, which VulkanQueue::execute_now calls (VulkanContext::set_before_execute) before every
// submission of command lists.
//
// Header-only, so the D3D12 layer, xmem and the file shims share one state in a process without linking each other.
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

// The runtime's own heap (host_heap) for the state the D3D12 layer's write-fault handler waits behind. A thread can take
// a write fault on a guarded title page while it holds the process heap's lock: XMemAlloc is _aligned_malloc, so
// HeapAlloc and HeapFree write block headers on the title's pages. Its handler then waits for the tracker's lock
// (ShadowPages::reopen) and the list of trackers (g_shadow_pages_lock). So a thread that holds either of those, or any
// lock such a thread waits for (the record and journal of title_write_watch, title_page_aliases, kernel_writes' pins),
// must never wait for the process heap: it would wait for the faulting thread, which waits for it. Their containers
// allocate here instead, from a heap the title never touches and on whose lock no thread faults.
namespace leonida::host_heap {

inline HANDLE handle() {
    static const HANDLE heap = [] {
        const HANDLE created = HeapCreate(0, 0, 0);
        if (!created)  // the process heap brings back the deadlock this heap avoids: write watching stays off
            std::fprintf(stderr, "[rt] host heap: HeapCreate failed (%lu); write watching is off\n", GetLastError());
        return created ? created : GetProcessHeap();
    }();
    return heap;
}

template <class T>
struct Allocator {
    using value_type = T;
    Allocator() noexcept { handle(); }  // created before any lock is taken
    template <class U>
    Allocator(const Allocator<U>&) noexcept {}
    T* allocate(std::size_t n) {
        if (n > std::size_t(-1) / sizeof(T)) throw std::bad_array_new_length();
        if (void* p = HeapAlloc(handle(), 0, n * sizeof(T))) return static_cast<T*>(p);
        throw std::bad_alloc();
    }
    void deallocate(T* p, std::size_t) noexcept { HeapFree(handle(), 0, p); }
    template <class U>
    bool operator==(const Allocator<U>&) const noexcept {
        return true;
    }
};

template <class T>
using vector = std::vector<T, Allocator<T>>;
template <class K, class V>
using map = std::map<K, V, std::less<K>, Allocator<std::pair<const K, V>>>;

}  // namespace leonida::host_heap

namespace leonida::title_memory_map {

inline std::atomic<std::uint64_t> map_generation{0};

inline std::uint64_t generation() { return map_generation.load(std::memory_order_acquire); }
inline void changed() { map_generation.fetch_add(1, std::memory_order_release); }

}  // namespace leonida::title_memory_map

namespace leonida::title_page_aliases {

constexpr std::uintptr_t kPageSize = 0x10000;

struct State {
    std::shared_mutex mutex;
    // 64 KiB-aligned VAs
    std::unordered_set<std::uintptr_t, std::hash<std::uintptr_t>, std::equal_to<std::uintptr_t>,
                       host_heap::Allocator<std::uintptr_t>>
        aliased;
    std::atomic<std::uint64_t> generation{0};
};

inline State& state() {
    static State s;
    return s;
}

// The 64 KiB page at `va` (rounded down) shows a page that is also mapped elsewhere (true), or no longer does.
inline void set(std::uintptr_t va, bool aliased) {
    State& s = state();
    va &= ~(kPageSize - 1);
    std::unique_lock lock(s.mutex);
    const bool changed = aliased ? s.aliased.insert(va).second : s.aliased.erase(va) != 0;
    if (changed) s.generation.fetch_add(1, std::memory_order_release);
}

// Bumped by every change of set().
inline std::uint64_t generation() { return state().generation.load(std::memory_order_acquire); }

// Calls `f(va)` for each aliased 64 KiB page overlapping [begin, end).
template<class F>
void for_each_in(std::uintptr_t begin, std::uintptr_t end, F&& f) {
    State& s = state();
    std::shared_lock lock(s.mutex);
    if (s.aliased.empty()) return;
    for (std::uintptr_t va = begin & ~(kPageSize - 1); va < end; va += kPageSize)
        if (s.aliased.count(va)) f(va);
}

}  // namespace leonida::title_page_aliases

// 3. GPU-only views (title_gpu_views). The title reserves some memory twice: a range the CPU maps physical pages into
//    and a GPU-only range of the same size (PAGE_NOACCESS | PAGE_GRAPHICS_NOACCESS) that the console's GPU page
//    tables map to the same pages, and it hands the GPU-only address to the GPU (the texture stream pool: slot-133
//    placements, slot-70 views, DirectStorage destinations; gfxres notes section 5). Nothing maps the GPU-only range
//    for the host CPU, so the D3D12 layer reads such memory at the CPU range's address: to_cpu. xmem pairs each
//    GPU-only reservation with the CPU-mappable reservation of the same size made just before it, as
//    five-star-recomp does (docs/gpu-resources.md "Addresses", XMemReservation.CpuView; GPL-3.0-or-later).
namespace leonida::title_gpu_views {

struct View {
    std::uintptr_t gpu, cpu, size;
};

struct State {
    std::shared_mutex mutex;
    std::vector<View> views;
    std::atomic<std::size_t> count{0};
    std::uintptr_t last_cpu = 0, last_cpu_size = 0;  // the last CPU-mappable reservation
};

inline State& state() {
    static State s;
    return s;
}

// xmem: a reservation the title made, GPU-only or not. Returns true when it paired a GPU-only one.
inline bool note_reservation(std::uintptr_t base, std::uintptr_t size, bool gpu_only) {
    State& s = state();
    std::unique_lock lock(s.mutex);
    if (!gpu_only) {
        s.last_cpu = base;
        s.last_cpu_size = size;
        return false;
    }
    if (!s.last_cpu || s.last_cpu_size != size) return false;
    s.views.push_back({base, s.last_cpu, size});
    s.count.store(s.views.size(), std::memory_order_release);
    s.last_cpu = s.last_cpu_size = 0;
    return true;
}

// Where the CPU reads the memory a GPU address names: the paired CPU range for a GPU-only view, else the address.
inline std::uintptr_t to_cpu(std::uintptr_t va, std::uintptr_t bytes = 1) {
    State& s = state();
    if (!s.count.load(std::memory_order_acquire)) return va;
    std::shared_lock lock(s.mutex);
    for (const View& v : s.views)
        if (va >= v.gpu && va - v.gpu < v.size && bytes <= v.size - (va - v.gpu)) return va - v.gpu + v.cpu;
    return va;
}

// The other address of memory in a paired range: the CPU view's address for a GPU-only one and the GPU-only
// address for a CPU view's; 0 outside every pair.
inline std::uintptr_t other_view(std::uintptr_t va) {
    State& s = state();
    if (!s.count.load(std::memory_order_acquire)) return 0;
    std::shared_lock lock(s.mutex);
    for (const View& v : s.views) {
        if (va >= v.gpu && va - v.gpu < v.size) return va - v.gpu + v.cpu;
        if (va >= v.cpu && va - v.cpu < v.size) return va - v.cpu + v.gpu;
    }
    return 0;
}

}  // namespace leonida::title_gpu_views

// 4. Texel writes (title_texel_writes). The title writes texels into its own texture memory through XG's
//    CopyIntoSubresource (xg_xs.cpp) and through the pointer Map gives it; the D3D12 layer fills a texture's image
//    from that memory again when told (Device::refresh_title_textures drains these ranges before a submission), so a
//    small update (a glyph in a font cache) is not left to a hash that samples the bytes.
namespace leonida::title_texel_writes {

struct State {
    std::mutex mutex;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> ranges;  // [begin, end)
    bool overflow = false;                                          // too many to keep: check everything
    std::atomic<bool> pending{false};
};

inline State& state() {
    static State s;
    return s;
}

inline void note(const void* begin, std::size_t bytes) {
    if (!begin || !bytes) return;
    State& s = state();
    std::lock_guard lock(s.mutex);
    const auto at = reinterpret_cast<std::uintptr_t>(begin);
    if (s.ranges.size() < 4096) s.ranges.emplace_back(at, at + bytes);
    else s.overflow = true;
    s.pending.store(true, std::memory_order_release);
}

// The ranges noted since the last call; `overflow` says some were dropped.
inline std::vector<std::pair<std::uintptr_t, std::uintptr_t>> take(bool& overflow) {
    State& s = state();
    overflow = false;
    if (!s.pending.load(std::memory_order_acquire)) return {};
    std::lock_guard lock(s.mutex);
    overflow = std::exchange(s.overflow, false);
    s.pending.store(false, std::memory_order_relaxed);
    return std::exchange(s.ranges, {});
}

}  // namespace leonida::title_texel_writes

// 5. XG layouts by shape (title_xg_layouts). A texture the title locks and writes with the CPU through Map is laid out
//    as the XG computer the title made for it says (its lock computes the layout with XGCreateTextureComputer right
//    before it maps, gfxres notes section 5), which a slot-133 description (Layout 0) does not tell: measured, the
//    2048x2048 and 1024x1024 A8 glyph caches are LINEAR (0x100) while XG's default for that description is 4KB_S.
//    xg_xs records the Layout word of every computer by the texture's shape; the D3D12 layer takes the latest one of
//    a mapped texture's shape when no slot-70 view has given it a layout.
namespace leonida::title_xg_layouts {

struct Shape {
    std::uint32_t dimension, format, height, depth, mips;
    std::uint64_t width;
    bool operator<(const Shape& o) const {
        return std::tie(dimension, format, width, height, depth, mips) <
               std::tie(o.dimension, o.format, o.width, o.height, o.depth, o.mips);
    }
};

struct State {
    std::mutex mutex;
    std::map<Shape, std::uint32_t> layouts;
};

inline State& state() {
    static State s;
    return s;
}

inline void note(const Shape& shape, std::uint32_t layout) {
    State& s = state();
    std::lock_guard lock(s.mutex);
    if (s.layouts.size() < 65536 || s.layouts.count(shape)) s.layouts[shape] = layout;
}

// The Layout word of the latest computer of this shape, or false.
inline bool find(const Shape& shape, std::uint32_t& layout) {
    State& s = state();
    std::lock_guard lock(s.mutex);
    const auto it = s.layouts.find(shape);
    if (it == s.layouts.end()) return false;
    layout = it->second;
    return true;
}

}  // namespace leonida::title_xg_layouts

// 6. Write-watched title memory (title_write_watch). xmem reserves the title's ordinary memory (the XMemVirtualAlloc
//    and VirtualAlloc reservations that are not placeholders or physical-page windows) with MEM_WRITE_WATCH, and the
//    kernel keeps a dirty bit per 4 KiB page there: GetWriteWatch with WRITE_WATCH_FLAG_RESET returns the pages
//    written since and clears them. A shadow over such pages asks the kernel instead of making them read-only: no
//    write fault, no VirtualProtect, and a kernel write (ReadFile) into them neither fails nor needs opening first.
//    Sections (xmem's physical-page views) can't be write-watched, and keep the read-only guard.
//
//    Several shadows can cover one page (a console descriptor heap's mirror next to a slot-133 buffer, two devices),
//    and one asking clears the kernel's bits for all of them. So what each asking found is kept in a journal: per
//    page, the number of the last asking that found it written. A reader (a shadow's tracker) was written to since its
//    own last asking where a page's number is above it. As VIctory does (crates/victory-gpu/src/watch.rs,
//    GPL-3.0-or-later). A reservation is numbered the same way when it is recorded: a reader whose last asking is
//    older takes all of its pages as written. A reservation the title releases and reserves again at the same address
//    is then new to the shadows, though its pages are write-watched both before and after and the new one's kernel
//    bits are clear (what the old one's pages held is gone: they read as zeros now).
//
//    LEONIDA_WRITE_WATCH=0 reserves without MEM_WRITE_WATCH: every shadow keeps the read-only guard, as before. Under
//    Wine that is the default (=1 turns write watching on).
namespace leonida::title_write_watch {

constexpr std::uintptr_t kPage = 0x1000;  // the kernel's unit
// The journal is pruned of what every reader has seen once it doubles. Past kMaxPages it forgets askings more than
// kKeep old even if a reader hasn't seen them: that reader then takes all of its pages as written, once.
constexpr std::size_t kMaxPages = std::size_t(1) << 18;
constexpr std::uint64_t kKeep = std::uint64_t(1) << 16;

// On by default on Windows; off under Wine, whose write watching was not measured here (LEONIDA_WRITE_WATCH=1 turns it
// on there). Off with LEONIDA_WRITE_WATCH=0.
inline bool enabled() {
    static const bool on = [] {
        char value[8] = {};
        std::size_t length = 0;
        if (host_heap::handle() == GetProcessHeap()) return false;  // no private heap: see host_heap
        if (getenv_s(&length, value, sizeof(value), "LEONIDA_WRITE_WATCH") == 0 && length > 1) return value[0] != '0';
        const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
        return !(ntdll && GetProcAddress(ntdll, "wine_get_version"));
    }();
    return on;
}

// A reservation made with MEM_WRITE_WATCH: its end, and the number it was recorded with (an asking's, see ask).
struct Region {
    std::uintptr_t end;
    std::uint64_t born;
};

// The containers allocate from host_heap: the D3D12 layer takes these locks while it holds a tracker's lock, which its
// write-fault handler waits for, and a faulting thread may hold the process heap's lock.
struct State {
    std::shared_mutex mutex;                         // regions
    host_heap::map<std::uintptr_t, Region> regions;  // base -> the reservation at base
    std::atomic<std::uint64_t> generation{0};        // bumped by every change of regions
    std::atomic<std::size_t> count{0};               // regions.size(), read without the lock

    // The last number given to an asking (or to a recorded reservation, which takes one without the journal's lock).
    std::atomic<std::uint64_t> asked{0};
    std::mutex journal;       // everything below
    std::uint64_t floor = 0;  // askings up to this one were forgotten before every reader saw them
    host_heap::map<std::uintptr_t, std::uint64_t> written;  // page address -> the last asking that found it written
    std::size_t pruned_size = 0;                            // written.size() after the last pruning
    std::uint64_t pruned_at = 0;                            // the asking of the last pruning
    host_heap::map<std::uint64_t, std::uint64_t> readers;   // reader -> its last asking
    std::uint64_t next_reader = 1;
    host_heap::vector<void*> found;            // room for the kernel's answer
    host_heap::vector<std::uint8_t> refusals;  // per run of an asking: the kernel refused it
};

inline State& state() {
    static State s;
    return s;
}

// xmem: [base, base + size) was reserved with MEM_WRITE_WATCH for the shadows. A reservation the title asked to be
// write-watched itself is not added: its bits are the title's. Every reader takes its pages as written at its next
// asking (they are not the pages it copied, even at the same address).
inline void add(std::uintptr_t base, std::uintptr_t size) {
    if (!size) return;
    State& s = state();
    const std::uintptr_t end = base + ((size + kPage - 1) & ~(kPage - 1));
    std::unique_lock lock(s.mutex);
    // A reservation released without remove() overlapping the new one is gone.
    auto it = s.regions.lower_bound(base);
    if (it != s.regions.begin() && std::prev(it)->second.end > base) --it;
    while (it != s.regions.end() && it->first < end) it = s.regions.erase(it);
    // Numbered under the lock: an asking that finds the reservation took its own number later (see ask).
    s.regions[base] = Region{end, s.asked.fetch_add(1, std::memory_order_acq_rel) + 1};
    s.count.store(s.regions.size(), std::memory_order_release);
    s.generation.fetch_add(1, std::memory_order_release);
}

// xmem: the reservation holding `address` was released.
inline void remove(std::uintptr_t address) {
    State& s = state();
    if (!s.count.load(std::memory_order_acquire)) return;
    std::unique_lock lock(s.mutex);
    auto it = s.regions.upper_bound(address);
    if (it == s.regions.begin() || (--it)->second.end <= address) return;
    s.regions.erase(it);
    s.count.store(s.regions.size(), std::memory_order_release);
    s.generation.fetch_add(1, std::memory_order_release);
}

inline std::uint64_t generation() { return state().generation.load(std::memory_order_acquire); }

// Whether `address` is in a write-watched reservation.
inline bool contains(std::uintptr_t address) {
    State& s = state();
    if (!s.count.load(std::memory_order_acquire)) return false;
    std::shared_lock lock(s.mutex);
    auto it = s.regions.upper_bound(address);
    return it != s.regions.begin() && (--it)->second.end > address;
}

// The end of the write-watched reservation holding `address`, or 0.
inline std::uintptr_t end_of(std::uintptr_t address) {
    State& s = state();
    if (!s.count.load(std::memory_order_acquire)) return 0;
    std::shared_lock lock(s.mutex);
    auto it = s.regions.upper_bound(address);
    return it != s.regions.begin() && (--it)->second.end > address ? it->second.end : 0;
}

// Calls `f(from, to)` for each write-watched part of [begin, end): each inside one reservation.
template <class F>
void for_each_in(std::uintptr_t begin, std::uintptr_t end, F&& f) {
    State& s = state();
    if (!s.count.load(std::memory_order_acquire)) return;
    std::shared_lock lock(s.mutex);
    auto it = s.regions.upper_bound(begin);
    if (it != s.regions.begin() && std::prev(it)->second.end > begin) --it;
    for (; it != s.regions.end() && it->first < end; ++it) {
        const std::uintptr_t from = (std::max)(it->first, begin), to = (std::min)(it->second.end, end);
        if (from < to) f(from, to);
    }
}

// xmem: [from, to) of a write-watched reservation changed with no write the kernel sees (decommitted, committed again,
// reset): every reader takes its pages as written at its next asking. Past kKeep pages, every reader takes all of its
// write-watched pages as written instead.
inline void mark(std::uintptr_t from, std::uintptr_t to) {
    from &= ~(kPage - 1);
    to = (to + kPage - 1) & ~(kPage - 1);
    if (from >= to || !contains(from)) return;
    State& s = state();
    std::lock_guard lock(s.journal);
    const std::uint64_t now = s.asked.fetch_add(1, std::memory_order_acq_rel) + 1;
    if ((to - from) / kPage > kKeep) {
        s.floor = now;
        return;
    }
    for (std::uintptr_t p = from; p < to; p += kPage) s.written[p] = now;
}

// A new reader, which has seen everything asked so far (its pages are not known yet: it copies them all first).
inline std::uint64_t open_reader() {
    State& s = state();
    std::lock_guard lock(s.journal);
    const std::uint64_t id = s.next_reader++;
    s.readers[id] = s.asked.load(std::memory_order_acquire);
    return id;
}

inline void close_reader(std::uint64_t reader) {
    State& s = state();
    std::lock_guard lock(s.journal);
    s.readers.erase(reader);
}

// `reader` takes over what `other` has not seen yet (a tracker absorbing another's pages).
inline void merge_reader(std::uint64_t reader, std::uint64_t other) {
    State& s = state();
    std::lock_guard lock(s.journal);
    const auto a = s.readers.find(reader), b = s.readers.find(other);
    if (a != s.readers.end() && b != s.readers.end()) a->second = (std::min)(a->second, b->second);
}

// When the reservation holding `address` was recorded (see ask), or 0 when no recorded one holds it.
inline std::uint64_t born(std::uintptr_t address) {
    State& s = state();
    if (!s.count.load(std::memory_order_acquire)) return 0;
    std::shared_lock lock(s.mutex);
    auto it = s.regions.upper_bound(address);
    return it != s.regions.begin() && (--it)->second.end > address ? it->second.born : 0;
}

// One asking for `reader`: the kernel's written pages of each run (4 KiB aligned, each inside one reservation from
// for_each_in) are taken and cleared, then `dirty(from, to)` gets the pages of the runs written since the reader's
// last asking, the kernel's answer now and the earlier askings of other readers alike, and all of a run whose
// reservation was recorded since then (or is not recorded: reserved again, not added yet). `refused(from, to)` gets
// a run the kernel refused (no longer write-watched: released), for which dirty is not called. Returns the asking's
// number. `runs` is a vector of (from, to) pairs.
template <class Runs, class Dirty, class Refused>
std::uint64_t ask(std::uint64_t reader, const Runs& runs, Dirty&& dirty, Refused&& refused) {
    State& s = state();
    std::lock_guard lock(s.journal);
    const auto me = s.readers.find(reader);
    const std::uint64_t since = me != s.readers.end() ? me->second : 0;
    const std::uint64_t now = s.asked.fetch_add(1, std::memory_order_acq_rel) + 1;
    s.refusals.assign(runs.size(), 0);
    for (std::size_t r = 0; r < runs.size(); ++r) {
        const auto [from, to] = runs[r];
        ULONG_PTR count = (to - from) / kPage;
        if (s.found.size() < count) s.found.resize(count);
        ULONG granularity = 0;
        if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, reinterpret_cast<void*>(from), to - from, s.found.data(), &count,
                          &granularity) != 0) {
            s.refusals[r] = 1;
            continue;
        }
        for (ULONG_PTR k = 0; k < count; ++k)
            s.written[reinterpret_cast<std::uintptr_t>(s.found[k]) & ~(kPage - 1)] = now;
    }
    for (std::size_t r = 0; r < runs.size(); ++r) {
        const auto [from, to] = runs[r];
        if (s.refusals[r]) {
            refused(from, to);
            continue;
        }
        // Forgotten before this reader saw it, or a reservation new to it: all of it. Looked up after `now` was
        // taken, so a reservation recorded later has a higher number and is all written at the next asking too.
        const std::uint64_t reserved = born(from);
        if (since < s.floor || !reserved || reserved > since) {
            dirty(from, to);
            continue;
        }
        std::uintptr_t run_from = 0, run_to = 0;
        for (auto it = s.written.lower_bound(from); it != s.written.end() && it->first < to; ++it) {
            if (it->second <= since) continue;
            if (it->first != run_to) {
                if (run_from != run_to) dirty(run_from, run_to);
                run_from = it->first;
            }
            run_to = it->first + kPage;
        }
        if (run_from != run_to) dirty(run_from, run_to);
    }
    if (me != s.readers.end()) me->second = now;
    // What every reader has seen goes, once the journal doubled or every kPruneEvery askings (a pruning that dropped
    // little, readers being behind then, must not stop the next); past kMaxPages, what is more than kKeep askings old
    // goes too.
    constexpr std::uint64_t kPruneEvery = 256;
    if (s.written.size() > 4096 && (s.written.size() > 2 * s.pruned_size || now - s.pruned_at >= kPruneEvery)) {
        s.pruned_at = now;
        std::uint64_t seen = now;
        for (const auto& [id, last] : s.readers) seen = (std::min)(seen, last);
        if (s.written.size() > kMaxPages && now > kKeep && seen < now - kKeep) {
            seen = now - kKeep;
            s.floor = (std::max)(s.floor, seen);
        }
        std::erase_if(s.written, [seen](const auto& entry) { return entry.second <= seen; });
        s.pruned_size = s.written.size();
    }
    return now;
}

}  // namespace leonida::title_write_watch

namespace leonida::kernel_writes {

// The layer's side: makes every watched page in [begin, end) writable. Set once the first shadow exists.
using Opener = void (*)(std::uintptr_t begin, std::uintptr_t end);

// xmem's side: commits the physical pages mapped in [address, address + bytes) that are still waiting for their first
// touch (rt::xmem::commit_physical_range; pages xmem does not map are left as they are). gta6rt sets it at start.
using Committer = bool (*)(const void* address, std::size_t bytes);

struct Pin {
    std::uintptr_t begin, end;
    const OVERLAPPED* overlapped;  // null: ends when the call returns
    std::uint64_t id;
};

struct State {
    std::atomic<Opener> opener{nullptr};
    std::atomic<Committer> committer{nullptr};
    std::mutex mutex;
    host_heap::vector<Pin> pins;  // host_heap: see there
    std::uint64_t next_id = 1;
    std::atomic<std::size_t> count{0};  // pins.size(), read without the lock
};

inline State& state() {
    static State s;
    return s;
}

inline void set_opener(Opener opener) { state().opener.store(opener, std::memory_order_release); }
inline void set_committer(Committer committer) { state().committer.store(committer, std::memory_order_release); }

// Whether an overlapped request is still in flight. Its OVERLAPPED is the title's and may be gone by now: a read that
// faults counts as completed (the title doesn't free an OVERLAPPED it is still waiting on).
inline bool overlapped_pending(const OVERLAPPED* overlapped) {
    __try {
        return overlapped->Internal == static_cast<ULONG_PTR>(STATUS_PENDING);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// The pinned ranges overlapping [begin, end), for the layer's refresh. Overlapped pins whose request completed are
// dropped first.
template<class F>
void for_each_pin(std::uintptr_t begin, std::uintptr_t end, F&& f) {
    State& s = state();
    if (!s.count.load(std::memory_order_acquire)) return;
    std::lock_guard lock(s.mutex);
    std::erase_if(s.pins, [](const Pin& pin) { return pin.overlapped && !overlapped_pending(pin.overlapped); });
    s.count.store(s.pins.size(), std::memory_order_release);
    for (const Pin& pin : s.pins)
        if (pin.begin < end && begin < pin.end) f((std::max)(pin.begin, begin), (std::min)(pin.end, end));
}

// Before the kernel writes [buffer, buffer + bytes): pins the range (until end_write(), or until `overlapped` completes
// when it is not null) and opens it. Returns the pin's id, 0 when nothing was pinned.
inline std::uint64_t begin_write(const void* buffer, std::size_t bytes, const OVERLAPPED* overlapped) {
    State& s = state();
    const Opener opener = s.opener.load(std::memory_order_acquire);
    if (!opener || !buffer || !bytes) return 0;
    const auto begin = reinterpret_cast<std::uintptr_t>(buffer);
    std::uint64_t id = 0;
    {
        std::lock_guard lock(s.mutex);
        id = s.next_id++;
        s.pins.push_back({begin, begin + bytes, overlapped, id});
        s.count.store(s.pins.size(), std::memory_order_release);
    }
    // After the pin: a refresh that runs from here on leaves the range writable.
    opener(begin, begin + bytes);
    return id;
}

// The call is over: a synchronous pin ends. An overlapped pin whose request is still pending stays until it completes.
inline void end_write(std::uint64_t id, bool pending) {
    if (!id) return;
    State& s = state();
    std::lock_guard lock(s.mutex);
    std::erase_if(s.pins, [id, pending](const Pin& pin) { return pin.id == id && !(pending && pin.overlapped); });
    s.count.store(s.pins.size(), std::memory_order_release);
}

// Readies [buffer, buffer + bytes) for a retry after ERROR_NOACCESS: commits xmem's pages that wait for their first
// touch (the kernel's write does not fault them in) and opens watched pages again. false: neither side is set.
inline bool reopen(const void* buffer, std::size_t bytes) {
    if (!buffer || !bytes) return false;
    const State& s = state();
    const Committer committer = s.committer.load(std::memory_order_acquire);
    const Opener opener = s.opener.load(std::memory_order_acquire);
    if (committer) committer(buffer, bytes);
    if (opener) {
        const auto begin = reinterpret_cast<std::uintptr_t>(buffer);
        opener(begin, begin + bytes);
    }
    return committer || opener;
}

// The file shims' reads into title memory (file_system.cpp): ReadFile, ReadFileEx and ReadFileScatter with their
// destination pinned and opened first, and retried once if it fails with ERROR_NOACCESS anyway (after committing
// first-touch pages: the title retries such a read forever, `0x314200`, so the retry has to change something).
inline BOOL read_file(HANDLE file, LPVOID buffer, DWORD bytes, LPDWORD read, LPOVERLAPPED overlapped) {
    const std::uint64_t pin = begin_write(buffer, bytes, overlapped);
    BOOL ok = ::ReadFile(file, buffer, bytes, read, overlapped);
    if (!ok && GetLastError() == ERROR_NOACCESS && reopen(buffer, bytes))
        ok = ::ReadFile(file, buffer, bytes, read, overlapped);
    const DWORD error = GetLastError();
    end_write(pin, !ok && error == ERROR_IO_PENDING);
    SetLastError(error);
    return ok;
}

inline BOOL read_file_ex(HANDLE file, LPVOID buffer, DWORD bytes, LPOVERLAPPED overlapped,
                         LPOVERLAPPED_COMPLETION_ROUTINE routine) {
    const std::uint64_t pin = begin_write(buffer, bytes, overlapped);
    BOOL ok = ::ReadFileEx(file, buffer, bytes, overlapped, routine);
    if (!ok && GetLastError() == ERROR_NOACCESS && reopen(buffer, bytes))
        ok = ::ReadFileEx(file, buffer, bytes, overlapped, routine);
    const DWORD error = GetLastError();
    end_write(pin, bool(ok));  // queued: pending until the OVERLAPPED says otherwise
    SetLastError(error);
    return ok;
}

// Each segment is one system page.
inline BOOL read_file_scatter(HANDLE file, FILE_SEGMENT_ELEMENT segments[], DWORD bytes, LPDWORD reserved,
                              LPOVERLAPPED overlapped) {
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    std::vector<std::uint64_t> pins;
    const DWORD count = segments ? (bytes + info.dwPageSize - 1) / info.dwPageSize : 0;
    for (DWORD i = 0; i < count; ++i)
        pins.push_back(begin_write(segments[i].Buffer, info.dwPageSize, overlapped));
    BOOL ok = ::ReadFileScatter(file, segments, bytes, reserved, overlapped);
    if (!ok && GetLastError() == ERROR_NOACCESS && count) {
        for (DWORD i = 0; i < count; ++i) reopen(segments[i].Buffer, info.dwPageSize);
        ok = ::ReadFileScatter(file, segments, bytes, reserved, overlapped);
    }
    const DWORD error = GetLastError();
    for (const std::uint64_t pin : pins) end_write(pin, bool(ok) || error == ERROR_IO_PENDING);
    SetLastError(error);
    return ok;
}

}  // namespace leonida::kernel_writes
