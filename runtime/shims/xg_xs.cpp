// xg_xs.dll: Xbox Series texture layout and CPU copies. The object has an
// Xbox vtable (AddRef and Release first, with no QueryInterface).
#include <windows.h>
#include <intrin.h>

#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>

#include "shims.h"
#include "xg_xs_layout.h"
#include "../src/title_page_watch.h"

namespace {

using namespace rt::xg_xs;
constexpr size_t kSlots = 32;

struct Computer {
    const void* const* vtable;
    std::atomic<ULONG> refs{1};
    Texture texture;
};

void trace(std::atomic<unsigned>& calls, const char* name, const void* caller, const char* format, ...) {
    static const bool enabled = [] {
        char value[2]{};
        return GetEnvironmentVariableA("LEONIDA_XG_TRACE", value, sizeof(value)) == 1 && value[0] == '1';
    }();
    if (!enabled) return;
    const unsigned call = calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (call > 1024) return;
    char details[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(details, sizeof(details), format, args);
    va_end(args);
    std::fprintf(stderr, "[rt] xg_xs: LOCAL DIAGNOSTIC: %s #%u caller=%s %s\n",
                 name, call, rt::shims::module_offset(reinterpret_cast<uint64_t>(caller)).c_str(), details);
}

void trace_desc(std::atomic<unsigned>& calls, const char* name, const void* caller,
                const ResourceDesc* desc, const void* out, uint32_t options,
                HRESULT hr, uint32_t mode, uint64_t bytes) {
    if (!desc) {
        trace(calls, name, caller, "desc=null out=%p options=%#x hr=0x%08x", out, options, uint32_t(hr));
        return;
    }
    trace(calls, name, caller,
          "dim=%u %llux%ux%u mips=%u fmt=%u samples=%u quality=%u align=%llu layout=%#x flags=%#x "
          "options=%#x out=%p hr=0x%08x mode=%#x bytes=%llu",
          desc->dimension, static_cast<unsigned long long>(desc->width), desc->height, unsigned(desc->depth_or_array),
          unsigned(desc->mips), desc->format, desc->samples, desc->quality, static_cast<unsigned long long>(desc->alignment),
          desc->layout, desc->flags, options, out, uint32_t(hr), mode, static_cast<unsigned long long>(bytes));
}

ULONG __stdcall add_ref(Computer* self) {
    static std::atomic<unsigned> calls{0};
    const ULONG refs = self->texture.layout.planes ? self->refs.fetch_add(1, std::memory_order_relaxed) + 1 : 1;
    trace(calls, "slot 0", _ReturnAddress(), "this=%p refs=%lu", self, refs);
    return refs;
}

ULONG __stdcall release(Computer* self) {
    static std::atomic<unsigned> calls{0};
    const ULONG refs = self->texture.layout.planes ? self->refs.fetch_sub(1, std::memory_order_acq_rel) - 1 : 1;
    trace(calls, "slot 1", _ReturnAddress(), "this=%p refs=%lu", self, refs);
    if (!refs) delete self;
    return refs;
}

HRESULT __stdcall get_layout(Computer* self, ResourceLayout* out) {
    static std::atomic<unsigned> calls{0};
    const auto& layout = self->texture.layout;
    const HRESULT hr = !layout.planes ? E_NOTIMPL : !out ? E_POINTER : S_OK;
    trace(calls, "slot 2", _ReturnAddress(), "this=%p out=%p hr=0x%08x bytes=%llu align=%llu mips=%u planes=%u",
          self, out, uint32_t(hr), static_cast<unsigned long long>(layout.size),
          static_cast<unsigned long long>(layout.alignment), layout.mips, layout.planes);
    if (SUCCEEDED(hr)) {
        *out = {layout.size, layout.alignment, layout.mips, layout.planes, {layout.plane[0]}, layout.dimension, 0};
    }
    return hr;
}

uint64_t __stdcall get_size(Computer* self) {
    static std::atomic<unsigned> calls{0};
    const uint64_t bytes = self->texture.layout.size;
    trace(calls, "slot 3", _ReturnAddress(), "this=%p bytes=%llu", self, static_cast<unsigned long long>(bytes));
    return bytes;
}

uint64_t __stdcall get_alignment(Computer* self) {
    static std::atomic<unsigned> calls{0};
    const uint64_t bytes = self->texture.layout.alignment;
    trace(calls, "slot 4", _ReturnAddress(), "this=%p bytes=%llu", self, static_cast<unsigned long long>(bytes));
    return bytes;
}

uint64_t __stdcall get_mip_offset(Computer* self, uint32_t plane, uint32_t level) {
    static std::atomic<unsigned> calls{0};
    const auto& layout = self->texture.layout;
    const uint64_t offset = plane < layout.planes && level < layout.mips ? layout.plane[plane].mips[level].offset : 0;
    trace(calls, "slot 5", _ReturnAddress(), "this=%p plane=%u mip=%u offset=%llu",
          self, plane, level, static_cast<unsigned long long>(offset));
    return offset;
}

uint64_t __stdcall get_element_offset(Computer* self, uint32_t plane, uint32_t level,
                                      uint64_t x, uint32_t y, uint32_t z, uint32_t sample, uint32_t* bit) {
    static std::atomic<unsigned> calls{0};
    const auto& t = self->texture;
    uint64_t offset = 0;
    if (bit) *bit = 0;
    if (plane < t.layout.planes && level < t.layout.mips && sample == 0) {
        const auto& mip = t.layout.plane[plane].mips[level];
        if (x < mip.width_elements && y < mip.height_elements && z < mip.depth) {
            const ElementAddress address(t);
            offset = address.offset(mip, address.tail_origin(mip), uint32_t(x), y, z);
        }
    }
    trace(calls, "slot 6", _ReturnAddress(), "this=%p plane=%u mip=%u x=%llu y=%u z=%u sample=%u bit=%p offset=%llu",
          self, plane, level, static_cast<unsigned long long>(x), y, z, sample, bit,
          static_cast<unsigned long long>(offset));
    return offset;
}

template<bool Into>
HRESULT copy_subresource(const Texture& t, uint32_t plane, uint32_t subresource, uint32_t sample,
                         const void* src, void* dst, uint32_t row_pitch, uint32_t slice_pitch) {
    const auto& layout = t.layout;
    if (!layout.planes) return E_NOTIMPL;
    if (!src || !dst) return E_POINTER;
    if (plane >= layout.planes || subresource >= layout.mips * t.array || sample) return E_INVALIDARG;
    const auto& mip = layout.plane[plane].mips[subresource % layout.mips];
    const uint32_t slice = subresource / layout.mips;
    const uint32_t row_bytes = mip.width_elements * t.format.bytes;
    if (row_pitch < row_bytes) return E_INVALIDARG;
    const uint32_t depth = t.layout.dimension == kTexture3D ? mip.depth : 1;
    const uint64_t one_slice = uint64_t(row_pitch) * (mip.height_elements - 1) + row_bytes;
    if (depth > 1 && slice_pitch < one_slice) return E_INVALIDARG;
    const uint64_t linear_size = uint64_t(slice_pitch) * (depth - 1) + one_slice;
    const auto source = static_cast<const std::byte*>(src);
    auto destination = static_cast<std::byte*>(dst);
    if (leonida::gfx10::tiling_detail::overlap(
            {source, Into ? linear_size : layout.size}, {destination, Into ? layout.size : linear_size}))
        return E_INVALIDARG;
    std::vector<uint64_t> columns;
    copy_elements<Into>(t, subresource % layout.mips, slice, source, destination, row_pitch, slice_pitch, columns);
    return S_OK;
}

HRESULT __stdcall copy_into(Computer* self, void* dst, uint32_t plane, uint32_t sub, const void* src,
                            uint32_t row, uint32_t slice) {
    static std::atomic<unsigned> calls{0};
    const HRESULT hr = copy_subresource<true>(self->texture, plane, sub, 0, src, dst, row, slice);
    // The texture's image on the PC GPU is filled from this memory: tell the D3D12 layer it changed.
    if (SUCCEEDED(hr)) leonida::title_texel_writes::note(dst, size_t(self->texture.layout.size));
    trace(calls, "slot 8", _ReturnAddress(), "this=%p dst=%p plane=%u sub=%u src=%p row=%u slice=%u hr=0x%08x",
          self, dst, plane, sub, src, row, slice, uint32_t(hr));
    return hr;
}

HRESULT __stdcall copy_from(Computer* self, void* dst, uint32_t plane, uint32_t sub, uint32_t sample,
                            const void* src, uint32_t row, uint32_t slice) {
    static std::atomic<unsigned> calls{0};
    const HRESULT hr = copy_subresource<false>(self->texture, plane, sub, sample, src, dst, row, slice);
    trace(calls, "slot 9", _ReturnAddress(), "this=%p dst=%p plane=%u sub=%u sample=%u src=%p row=%u slice=%u hr=0x%08x",
          self, dst, plane, sub, sample, src, row, slice, uint32_t(hr));
    return hr;
}

struct PackedMips {
    uint8_t standard, packed;
    uint16_t reserved;
    uint32_t tiles, start;
};
struct TileShape { uint32_t width, height, depth; };
struct SubresourceTiling { uint32_t width; uint16_t height, depth; uint32_t start; };
static_assert(sizeof(PackedMips) == 12 && sizeof(TileShape) == 12 && sizeof(SubresourceTiling) == 12);

HRESULT __stdcall get_tiling(Computer* self, uint32_t* total_tiles, PackedMips* packed, TileShape* shape,
                             uint32_t* count, uint32_t first, SubresourceTiling* tilings) {
    static std::atomic<unsigned> calls{0};
    const auto& t = self->texture;
    const auto& layout = t.layout;
    const auto& p = layout.plane[0];
    const unsigned requested = count ? *count : 0;
    const bool supported = layout.alignment == kTileBytes && layout.dimension == kTexture2D;
    const HRESULT hr = !layout.planes ? E_NOTIMPL : !supported ? E_FAIL : S_OK;
    const uint32_t tiles = supported ? uint32_t(layout.size / kTileBytes) : 0;
    unsigned written = 0;
    if (SUCCEEDED(hr)) {
        if (total_tiles) *total_tiles = tiles;
        if (packed) *packed = {uint8_t(p.standard_mips), uint8_t(p.tail_mips), 0, p.tail_mips ? 1u : 0u, 0};
        if (shape) *shape = {t.tile_width * t.format.side, t.tile_height * t.format.side, 1};
        const uint32_t total_subresources = t.array * layout.mips;
        if (count && first < total_subresources) {
            const uint32_t end = std::min(requested, total_subresources - first);
            for (uint32_t i = 0; i < end; ++i) {
                const uint32_t sub = first + i;
                const uint32_t level = sub % layout.mips;
                if (level >= p.standard_mips) continue;
                ++written;
                if (!tilings) continue;
                const auto& mip = p.mips[level];
                const uint64_t start_bytes = (sub / layout.mips) * p.slice_bytes + mip.offset;
                tilings[i] = {mip.padded_width / t.tile_width, uint16_t(mip.padded_height / t.tile_height), 1,
                              uint32_t(start_bytes / kTileBytes)};
            }
        }
        if (count) *count = written;
    }
    trace(calls, "slot 10", _ReturnAddress(),
          "this=%p total=%p packed=%p shape=%p count=%p first=%u out=%p requested=%u written=%u "
          "hr=0x%08x tiles=%u standard=%u tail=%u tile=%ux%u",
          self, total_tiles, packed, shape, count, first, tilings, requested, written, uint32_t(hr),
          tiles, p.standard_mips, p.tail_mips, t.tile_width * t.format.side, t.tile_height * t.format.side);
    return hr;
}

template<size_t Slot>
HRESULT __stdcall unimplemented_slot(Computer* self) {
    static std::atomic<unsigned> calls{0};
    trace(calls, "unknown slot", _ReturnAddress(), "this=%p slot=%zu hr=0x80004001", self, Slot);
    return E_NOTIMPL;
}

template<size_t... Slot>
auto make_vtable(std::index_sequence<Slot...>) {
    std::array<void*, sizeof...(Slot)> vtable{reinterpret_cast<void*>(&unimplemented_slot<Slot>)...};
    vtable[0] = reinterpret_cast<void*>(&add_ref);
    vtable[1] = reinterpret_cast<void*>(&release);
    vtable[2] = reinterpret_cast<void*>(&get_layout);
    vtable[3] = reinterpret_cast<void*>(&get_size);
    vtable[4] = reinterpret_cast<void*>(&get_alignment);
    vtable[5] = reinterpret_cast<void*>(&get_mip_offset);
    vtable[6] = reinterpret_cast<void*>(&get_element_offset);
    vtable[8] = reinterpret_cast<void*>(&copy_into);
    vtable[9] = reinterpret_cast<void*>(&copy_from);
    vtable[10] = reinterpret_cast<void*>(&get_tiling);
    return vtable;
}
const auto kVtable = make_vtable(std::make_index_sequence<kSlots>{});
Computer unavailable{kVtable.data(), {1}, {}};

HRESULT __stdcall XGCreateTextureComputer(const void* desc, void** out) {
    static std::atomic<unsigned> calls{0};
    ResourceDesc d{};
    if (desc) std::memcpy(&d, desc, sizeof(d));
    Texture t;
    HRESULT hr = E_NOTIMPL;
    if (!out) hr = E_POINTER;
    else {
        // Two measured callers dereference the computer even after a failed creation.
        *out = &unavailable;
        if (!desc) hr = E_INVALIDARG;
        else if (build_texture(d, t)) {
            auto* computer = new (std::nothrow) Computer{kVtable.data(), {1}, t};
            hr = computer ? S_OK : E_OUTOFMEMORY;
            if (computer) *out = computer;
            // How the title lays out a texture of this shape (title_xg_layouts): what its CPU writes use.
            leonida::title_xg_layouts::note({d.dimension, d.format, d.height, d.depth_or_array, d.mips, d.width},
                                            d.layout);
        }
    }
    trace_desc(calls, "create", _ReturnAddress(), desc ? &d : nullptr, out, 0, hr, t.mode, t.layout.size);
    // LOCAL DIAGNOSTIC: LEONIDA_XG_TRACE_FORMAT=<DXGI format> logs the first 256 computers of that format whatever
    // LEONIDA_XG_TRACE's cap, with the layout the title asked for (which tiling its CPU writes use).
    static const uint32_t only = [] {
        char value[8]{};
        return GetEnvironmentVariableA("LEONIDA_XG_TRACE_FORMAT", value, sizeof(value)) ? uint32_t(atoi(value)) : 0u;
    }();
    static std::atomic<unsigned> only_logged{0};
    if (only && desc && d.format == only && only_logged.fetch_add(1, std::memory_order_relaxed) < 256)
        std::fprintf(stderr,
                     "[rt] xg_xs: LOCAL DIAGNOSTIC: computer for format %u: dim=%u %llux%ux%u mips=%u layout=%#x "
                     "flags=%#x -> mode %u, %llu bytes, hr 0x%08x, caller %s\n",
                     d.format, d.dimension, static_cast<unsigned long long>(d.width), d.height,
                     unsigned(d.depth_or_array), unsigned(d.mips), d.layout, d.flags, t.mode,
                     static_cast<unsigned long long>(t.layout.size), uint32_t(hr),
                     rt::shims::module_offset(reinterpret_cast<uint64_t>(_ReturnAddress())).c_str());
    return hr;
}

HRESULT __stdcall XGSuggestSwizzleMode(const void* desc, uint32_t options, uint32_t* out) {
    static std::atomic<unsigned> calls{0};
    ResourceDesc d{};
    if (desc) std::memcpy(&d, desc, sizeof(d));
    HRESULT hr = E_INVALIDARG;
    uint32_t mode = 0;
    if (desc && out && d.dimension >= kTexture1D && d.dimension <= kTexture3D && d.width) {
        hr = E_NOTIMPL;
        if (!options) {
            mode = kSwizzled | select_mode(d);
            *out = mode;
            hr = S_OK;
        }
    }
    trace_desc(calls, "suggest", _ReturnAddress(), desc ? &d : nullptr, out, options, hr, mode, 0);
    return hr;
}

}  // namespace

SHIM_TABLE(xg_xs,
    {"xg_xs.dll", "XGCreateTextureComputer", reinterpret_cast<void*>(&XGCreateTextureComputer)},
    {"xg_xs.dll", "XGSuggestSwizzleMode", reinterpret_cast<void*>(&XGSuggestSwizzleMode)},
);
