#include <windows.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <cstring>
#include <vector>

#include "shims.h"
#include "xg_xs_layout.h"
#include "../src/title_texture_memory.h"
#include "../src/title_texture_refresh.h"

namespace {

using namespace rt::xg_xs;
using CreateFn = HRESULT(__stdcall*)(const void*, void**);
using SuggestFn = HRESULT(__stdcall*)(const void*, uint32_t, uint32_t*);
using RefFn = ULONG(__stdcall*)(void*);
using LayoutFn = HRESULT(__stdcall*)(void*, ResourceLayout*);
using SizeFn = uint64_t(__stdcall*)(void*);
using OffsetFn = uint64_t(__stdcall*)(void*, uint32_t, uint32_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t*);
using IntoFn = HRESULT(__stdcall*)(void*, void*, uint32_t, uint32_t, const void*, uint32_t, uint32_t);
using FromFn = HRESULT(__stdcall*)(void*, void*, uint32_t, uint32_t, uint32_t, const void*, uint32_t, uint32_t);
struct Tiling { uint32_t width; uint16_t height, depth; uint32_t start; };
using TilingFn = HRESULT(__stdcall*)(void*, uint32_t*, uint32_t*, uint32_t*, uint32_t*, uint32_t, Tiling*);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("FAIL %s\n", what);
    }
}

template<class Fn>
Fn slot(void* computer, unsigned index) {
    return reinterpret_cast<Fn>((*static_cast<void* const* const*>(computer))[index]);
}

bool make(CreateFn create, const ResourceDesc& d, void*& computer, const char* what) {
    const bool success = create(&d, &computer) == S_OK && computer;
    check(success, what);
    return success;
}

void check_addresses(const Texture& t) {
    const ElementAddress address(t);
    std::vector<bool> seen(t.layout.size / t.format.bytes);
    for (uint32_t slice = 0; slice < t.array; ++slice) {
        for (uint32_t level = 0; level < t.layout.mips; ++level) {
            const auto& mip = t.layout.plane[0].mips[level];
            const auto tail = address.tail_origin(mip);
            for (uint32_t y = 0; y < mip.height_elements; ++y) {
                for (uint32_t x = 0; x < mip.width_elements; ++x) {
                    const auto off = address.offset(mip, tail, x, y, slice);
                    if (off % t.format.bytes || off / t.format.bytes >= seen.size() || seen[off / t.format.bytes]) {
                        check(false, "every mip/array element has a distinct aligned address inside the resource");
                        return;
                    }
                    seen[off / t.format.bytes] = true;
                }
            }
        }
    }
    check(true, "every mip/array element has a distinct aligned address inside the resource");
}

// copy_elements (row + column addressing) reads every element where offset() says it is.
void check_detile(const Texture& t) {
    const ElementAddress address(t);
    std::vector<uint8_t> tiled(t.layout.size);
    for (size_t i = 0; i < tiled.size(); ++i) tiled[i] = uint8_t(i * 2654435761u >> 13);
    std::vector<uint64_t> columns;
    for (uint32_t slice = 0; slice < t.array; ++slice) {
        for (uint32_t level = 0; level < t.layout.mips; ++level) {
            const auto& mip = t.layout.plane[0].mips[level];
            const uint32_t pitch = mip.width_elements * t.format.bytes + 4;
            std::vector<uint8_t> linear(size_t(pitch) * mip.height_elements);
            copy_elements<false>(t, level, slice, tiled.data(), linear.data(), pitch, 0, columns);
            const auto tail = address.tail_origin(mip);
            for (uint32_t y = 0; y < mip.height_elements; ++y) {
                for (uint32_t x = 0; x < mip.width_elements; ++x) {
                    const uint64_t at = address.offset(mip, tail, x, y, slice);
                    if (std::memcmp(linear.data() + size_t(y) * pitch + size_t(x) * t.format.bytes, tiled.data() + at,
                                    t.format.bytes)) {
                        check(false, "copy_elements reads each element at offset()");
                        return;
                    }
                }
            }
        }
    }
    check(true, "copy_elements reads each element at offset()");
}

}  // namespace

int main() {
    {
        struct Entry { bool mapped, stale; int id; };
        Entry first{false, false, 1}, mapped{true, false, 2}, stale{false, true, 3};
        Entry last{false, false, 4}, added{false, false, 5};
        std::map<std::uint64_t, Entry*> entries{{10, &first}, {20, &mapped}, {30, &stale}, {40, &last}};
        std::uint64_t cursor = 0;
        std::vector<int> polled;
        const auto visit = [&](Entry& t) { polled.push_back(t.id); };
        leonida::poll_title_textures(entries, cursor, visit, [&] { return !polled.empty(); });
        check(polled == std::vector<int>{1}, "budget stops after a background texture");
        entries.erase(10);
        entries[50] = &added;
        leonida::poll_title_textures(entries, cursor, visit, [&] { return polled.size() == 2; });
        check(polled == std::vector<int>{1, 4}, "poll resumes after deletion and skips priority textures");
        polled.clear();
        leonida::poll_title_textures(entries, cursor, visit, [] { return false; });
        check(polled == std::vector<int>{5, 4}, "unlimited poll wraps once and includes new textures");
        cursor = 20;
        polled.clear();
        for (unsigned pass = 0; pass < 3; ++pass)
            leonida::poll_title_textures(entries, cursor, visit, [] { return true; });
        check(polled == std::vector<int>{4}, "an expired wall budget advances past priority entries to the next texture");

    }

    {
        leonida::TitleWriteRanges writes{{20, 30}, {10, 20}, {15, 17}, {60, 75}, {60, 67}, {90, 90}};
        const auto original = writes;
        leonida::merge_title_writes(writes);
        check(writes == leonida::TitleWriteRanges{{10, 30}, {60, 75}}, "coalesce overlapping and adjacent writes");
        bool same = true;
        for (unsigned begin = 0; begin < 100; ++begin) {
            for (unsigned end = begin; end < 100; ++end) {
                const bool overlap = std::any_of(original.begin(), original.end(), [&](auto r) {
                    return begin < end && r.first < r.second && r.first < end && r.second > begin;
                });
                same &= leonida::title_write_overlaps(writes, begin, end) == overlap;
            }
        }
        check(same, "indexed writes find the same overlaps as scanning each original notification");
    }

    SYSTEM_INFO sys{};
    GetSystemInfo(&sys);
    const auto page = sys.dwPageSize;
    leonida::TextureReadability residency;
    unsigned probes = 0;
    bool committed = false;
    const auto probe = [&] { ++probes; return committed; };
    for (unsigned pass = 0; pass < 100; ++pass) residency.check(8, true, probe);
    check(probes == 1 && residency.pending(), "unchanged reserved texture is probed only once in 100 passes");
    check(!residency.check(9, true, probe) && probes == 2, "a new map generation retries the failed footprint");
    committed = true;
    residency.retry();  // overlapping DirectStorage or XG write
    check(residency.check(9, true, probe) && probes == 3, "write notification retries at the same generation");
    committed = false;
    check(!residency.check(9, true, probe) && probes == 4, "previous success is rechecked before the next read");
    check(!residency.check(9, false, probe) && probes == 5, "rollback disables the failed-probe cache");
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE, 0, 65536, nullptr);
    if (!section) return 1;
    void* view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 65536);
    if (!view) return 1;
    const auto placed = reinterpret_cast<std::uintptr_t>(view);
    const auto first_touch = [](const void* at, std::size_t bytes) {
        return VirtualAlloc(const_cast<void*>(at), bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr;
    };
    check(!leonida::texture_memory_readable(placed, 65536), "background polling leaves mapped first-touch pages reserved");
    check(!leonida::texture_memory_readable(placed, 65536, [](const void*, std::size_t) { return true; }),
          "a section outside xmem stays deferred when the commit hook leaves it reserved");
    check(leonida::texture_memory_readable(placed, 65536, first_touch), "persistent Map can populate a physical section before checking it");
    UnmapViewOfFile(view);
    CloseHandle(section);
    void* reservation = VirtualAlloc(nullptr, 2 * page, MEM_RESERVE, PAGE_NOACCESS);
    if (!reservation) return 1;
    const auto address = reinterpret_cast<std::uintptr_t>(reservation);
    check(!leonida::texture_memory_readable(address, page), "untouched reserved texture is deferred");
    check(!leonida::texture_memory_readable(address, page, first_touch), "a persistent Map cannot commit a bare reservation");
    leonida::TextureReadability persistent_map;
    const auto probe_mapped = [&] { return leonida::texture_memory_readable(address, page); };
    check(!persistent_map.check(11, false, probe_mapped), "persistent Map defers a reserved texture before decoder writes");
    if (!VirtualAlloc(reservation, page, MEM_COMMIT, PAGE_READWRITE)) return 1;
    auto* pixels = static_cast<std::uint8_t*>(reservation);
    pixels[page - 1] = 93;
    check(persistent_map.check(11, false, probe_mapped) && pixels[page - 1] == 93,
          "persistent Map sees decoder writes after a failed probe at the same generation");
    check(leonida::texture_memory_readable(address + 3, page - 3) &&
          !leonida::texture_memory_readable(address + 3, page), "range crossing into a reserved mip is deferred");
    if (!VirtualAlloc(reinterpret_cast<void*>(address + page), page, MEM_COMMIT, PAGE_READONLY)) return 1;
    check(leonida::texture_memory_readable(address, 2 * page), "later stream commit completes the footprint");
    DWORD previous = 0;
    if (!VirtualProtect(reservation, page, PAGE_READWRITE | PAGE_GUARD, &previous)) return 1;
    check(!leonida::texture_memory_readable(address, 2 * page), "guarded texture page remains untouched");
    VirtualFree(reservation, page, MEM_DECOMMIT);
    check(!leonida::texture_memory_readable(address, page), "previously readable page is rechecked after decommit");
    VirtualFree(reservation, 0, MEM_RELEASE);

    auto create = reinterpret_cast<CreateFn>(rt::shims::find("xg_xs.dll", "XGCreateTextureComputer"));
    auto suggest = reinterpret_cast<SuggestFn>(rt::shims::find("xg_xs.dll", "XGSuggestSwizzleMode"));
    check(create && suggest && rt::shims::owns_dll("xg_xs.dll"), "XG exports are registered");
    if (!create || !suggest) return 1;

    ResourceDesc d{3, 0, 256, 128, 1, 1, DXGI_FORMAT_R8_UNORM, 1, 0, 0, 0};
    check(suggest(&d, 0, &d.layout) == S_OK && d.layout == 0x105, "2D SRV selects SW_4KB_S with aliased output");
    uint32_t mode = 0xdeadbeef;
    check(suggest(&d, 1, &mode) == E_NOTIMPL && mode == 0xdeadbeef, "unknown selector leaves output unchanged");
    check(create(&d, nullptr) == E_POINTER, "null computer output is rejected");
    void* computer = nullptr;
    if (!make(create, d, computer, "create 2D SW_4KB_S computer")) return 1;

    ResourceLayout l;
    std::memset(&l, 0xa5, sizeof(l));
    auto get_layout = slot<LayoutFn>(computer, 2);
    auto get_offset = slot<OffsetFn>(computer, 6);
    auto copy_into = slot<IntoFn>(computer, 8);
    check(get_layout(computer, &l) == S_OK, "slot 2 fills the Series layout");
    check(l.size == 32768 && l.alignment == 4096 && l.planes == 1 && l.mips == 1 && l.dimension == 3,
          "256x128 R8 uses eight 64x64 SW_4KB_S blocks");
    check(l.plane[0].mips[0].pitch_bytes == 256 && l.plane[0].mips[0].offset == 0,
          "slot 2 reports pitch and offset");
    const std::array<PlaneLayout, 3> unused{};
    check(std::memcmp(&l.plane[1], unused.data(), sizeof(unused)) == 0 && !l.reserved,
          "slot 2 clears all three unused planes and the trailing reserved word");
    check(slot<SizeFn>(computer, 3)(computer) == 32768 && slot<SizeFn>(computer, 4)(computer) == 4096,
          "size and alignment agree with slot 2");
    uint32_t bit = 99;
    check(get_offset(computer, 0, 0, 0, 1, 0, 0, &bit) == 16 && bit == 0 &&
          get_offset(computer, 0, 0, 64, 0, 0, 0, nullptr) == 4096,
          "SW_4KB_S addresses Y0 at bit 4 and X6 in the second tile");
    std::vector<uint8_t> tiled(l.size, 0xcd);
    constexpr unsigned pitch = 260;
    std::vector<uint8_t> linear(pitch * 128, 0);
    linear[64] = 17;
    linear[pitch] = 29;
    check(copy_into(computer, tiled.data(), 0, 0, linear.data(), pitch, 0) == S_OK &&
          tiled[4096] == 17 && tiled[16] == 29, "slot 8 writes expected tiled offsets from padded linear rows");
    check(copy_into(computer, tiled.data(), 0, 1, linear.data(), pitch, 0) == E_INVALIDARG,
          "slot 8 rejects an out-of-range subresource");
    check(copy_into(computer, tiled.data(), 0, 0, linear.data(), 255, 0) == E_INVALIDARG,
          "slot 8 rejects a short row pitch");
    check(slot<TilingFn>(computer, 10)(computer, nullptr, nullptr, nullptr, nullptr, 0, nullptr) == E_FAIL,
          "slot 10 requires a 64 KiB mode");
    check(slot<RefFn>(computer, 0)(computer) == 2 && slot<RefFn>(computer, 1)(computer) == 1 &&
          slot<RefFn>(computer, 1)(computer) == 0, "release deletes the computer at refcount zero");

    // The VIctory 256x256 BC7 SW_64KB_D fixture has one 64 KiB standard
    // mip and a tail block before it. Exercise it as a two-element array.
    d = {3, 0, 256, 256, 2, 0, DXGI_FORMAT_BC7_UNORM, 1, 0, 0x10a, 0};
    if (!make(create, d, computer, "create BC7 mipmapped array computer")) return 1;
    check(slot<LayoutFn>(computer, 2)(computer, &l) == S_OK, "get BC7 layout");
    const auto& p = l.plane[0];
    check(l.size == 262144 && l.mips == 9 && p.slice_bytes == 131072 &&
          p.standard_mips == 1 && p.tail_mips == 8 && p.element_bytes == 16,
          "BC7 array uses a standard block and a tail block per slice");
    check(p.mips[0].offset == 65536 && p.mips[0].pitch_bytes == 1024 &&
          p.mips[1].offset == 32768 && p.mips[1].in_tail,
          "BC7 standard mip and first tail offsets match fixture");
    uint32_t tiles = 0, packed[3]{}, shape[3]{}, count = 18;
    std::array<Tiling, 18> tilings{};
    tilings[1].start = 0xdeadbeef;
    check(slot<TilingFn>(computer, 10)(computer, &tiles, packed, shape, &count, 0, tilings.data()) == S_OK &&
          tiles == 4 && packed[0] == 0x0801 && packed[1] == 1 && packed[2] == 0 &&
          shape[0] == 256 && shape[1] == 256 && shape[2] == 1 && count == 2 &&
          tilings[0].start == 1 && tilings[9].start == 3 && tilings[1].start == 0xdeadbeef,
          "slot 10 reports standard tiles, texture-unit shape and packed tail");
    tiled.assign(l.size, 0);
    std::array<uint8_t, 16> block{}, back{};
    block[0] = 77;
    const unsigned last_mip_in_slice_1 = 17;
    check(slot<IntoFn>(computer, 8)(computer, tiled.data(), 0, last_mip_in_slice_1, block.data(), 16, 0) == S_OK &&
          tiled[p.slice_bytes + p.mips[8].offset] == 77 &&
          slot<FromFn>(computer, 9)(computer, back.data(), 0, last_mip_in_slice_1, 0, tiled.data(), 16, 0) == S_OK &&
          back == block, "slots 8 and 9 copy a BC block in the second slice's tail");
    slot<RefFn>(computer, 1)(computer);

    // The addrlib padded mip sizes use ceil division even for odd extents.
    Texture texture;
    d = {3, 0, 30, 20, 1, 0, DXGI_FORMAT_R8G8B8A8_UNORM, 1, 0, 1, 0};
    check(build_texture(d, texture) && texture.layout.size == 10240 &&
          texture.layout.plane[0].mips[0].offset == 5120 && texture.layout.plane[0].mips[0].pitch_bytes == 256,
          "row-major chain aligns each physical pitch to 256 B");

    for (const uint32_t swizzle : {0x105u, 0x109u, 0x10au}) {
        for (DXGI_FORMAT format : {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM,
                                   DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC7_UNORM}) {
            d = {3, 0, 259, 175, 2, 0, uint32_t(format), 1, 0, swizzle, 0};
            if (build_texture(d, texture)) check_addresses(texture), check_detile(texture);
            else check(false, "create odd-sized tiled mip chain");
        }
    }

    d = {4, 0, 9, 2, 2, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 1, 0, 0x105, 0};
    if (!make(create, d, computer, "create single-mip 3D SW_4KB_S computer")) return 1;
    check(slot<LayoutFn>(computer, 2)(computer, &l) == S_OK &&
          l.size == 8192 && l.plane[0].mips[0].padded_depth == 8 && l.plane[0].mips[0].tile_depth == 8,
          "3D 4KB_S blocks for RGBA8 are 8x16x8");
    check(slot<OffsetFn>(computer, 6)(computer, 0, 0, 0, 0, 1, 0, nullptr) == 16 &&
          slot<OffsetFn>(computer, 6)(computer, 0, 0, 8, 0, 0, 0, nullptr) == 4096,
          "3D 4KB_S selects Z0 at bit 4 and X3 in the second tile");
    slot<RefFn>(computer, 1)(computer);

    // GFX10 thick 4 KiB blocks: each 16-deep slab has one tail block, one
    // level-1 block and four level-0 blocks. Crossing z=16 skips the whole slab.
    ResourceDesc volume{4, 0, 32, 32, 32, 6, DXGI_FORMAT_R8_UNORM, 1, 0, 0x105, 0};
    if (!build_texture(volume, texture)) return 1;
    const ElementAddress thick(texture);
    const auto& levels = texture.layout.plane[0].mips;
    check(texture.layout.size == 49152 && levels[2].in_tail && levels[2].depth == 8 &&
          thick.offset(levels[0], {}, 0, 0, 0) == 8192 &&
          thick.offset(levels[0], {}, 0, 0, 16) == 32768 &&
          thick.offset(levels[2], thick.tail_origin(levels[2]), 0, 0, 0) == 2048,
          "3D mip chain uses per-slab macro offsets and a thick tail origin");
    for (DXGI_FORMAT format : {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM,
                               DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT,
                               DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC5_UNORM}) {
        volume = {4, 0, 61, 35, 67, 7, uint32_t(format), 1, 0, 0x105, 0};
        if (!build_texture(volume, texture)) return 1;
        const ElementAddress addr(texture);
        std::vector<uint8_t> used(texture.layout.size);
        bool distinct = true, matches = true;
        for (unsigned level = 0; level < texture.layout.mips; ++level) {
            const auto& mip = texture.layout.plane[0].mips[level];
            const auto tail = addr.tail_origin(mip);
            const unsigned pitch = mip.width_elements * texture.format.bytes + 8;
            const unsigned slice_pitch = pitch * mip.height_elements + 16;
            std::vector<uint8_t> source(slice_pitch * mip.depth), back(source.size()), tiled(texture.layout.size);
            std::vector<uint64_t> columns;
            for (unsigned z = 0; z < mip.depth; ++z)
                for (unsigned y = 0; y < mip.height_elements; ++y)
                    for (unsigned x = 0; x < mip.width_elements; ++x) {
                        const auto at = addr.offset(mip, tail, x, y, z);
                        if (at % texture.format.bytes || at > used.size() - texture.format.bytes || used[at]) {
                            distinct = false;
                            continue;
                        }
                        used[at] = 1;
                        source[z * slice_pitch + y * pitch + x * texture.format.bytes] = uint8_t(1 + (x+y+z)%251);
                    }
            if (!distinct) break;
            copy_elements<true>(texture, level, 0, source.data(), tiled.data(), pitch, slice_pitch, columns);
            copy_elements<false>(texture, level, 0, tiled.data(), back.data(), pitch, slice_pitch, columns);
            matches &= back == source;
        }
        check(distinct && matches, "odd volume mips and z slices are disjoint and preserve padded linear pitches");
    }

    d.samples = 4;
    check(create(&d, &computer) == E_NOTIMPL && computer, "unsupported texture keeps a safe fallback object");
    check(slot<LayoutFn>(computer, 2)(computer, &l) == E_NOTIMPL && l.size == 8192 &&
          slot<RefFn>(computer, 1)(computer) == 1, "fallback leaves caller outputs untouched and is safe to release");
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
