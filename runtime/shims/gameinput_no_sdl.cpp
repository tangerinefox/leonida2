// gameinput.dll: GameInputCreate without SDL3.
//
// runtime/shims/gameinput.cpp serves the real thing through SDL3 and is only compiled when SDL3 is available. A build
// without SDL3 therefore left gameinput.dll unserved, the loader fell through to the host's gameinput.dll (wine ships
// one), and that host answers GameInputCreate with a null interface while the title ignores the HRESULT: the title's
// lazy singleton stores the null and calls AddRef/Release through it, which is a page fault at address 0.
//
// This file keeps the object identity honest without any input backend: one static object, slot 0/1/2 as
// QueryInterface/AddRef/Release, E_NOTIMPL for every other slot, so a caller that only holds and releases the
// interface survives and one that asks for devices gets a clean refusal instead of a null dereference.
#include <windows.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <utility>

#include "shims.h"

namespace {

constexpr size_t kSlots = 64;  // comfortably covers the public v0 IGameInput vtable

struct GameInputStub {
    const void* const* vtable;
};

std::atomic<uint32_t> g_refs{1};

HRESULT __stdcall stub_qi(GameInputStub* self, const IID& riid, void** out);
ULONG __stdcall stub_add_ref(GameInputStub*);
ULONG __stdcall stub_release(GameInputStub*);

template <size_t Slot>
HRESULT __stdcall stub_method(GameInputStub*) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] gameinput: slot %zu is not served without SDL3; E_NOTIMPL (logged once)\n", Slot);
    return E_NOTIMPL;
}

template <size_t... Slot>
constexpr auto make_vtable(std::index_sequence<Slot...>) {
    return std::array<void*, sizeof...(Slot)>{
        (Slot == 0   ? reinterpret_cast<void*>(&stub_qi)
         : Slot == 1 ? reinterpret_cast<void*>(&stub_add_ref)
         : Slot == 2 ? reinterpret_cast<void*>(&stub_release)
                     : reinterpret_cast<void*>(&stub_method<Slot>))...};
}

const auto kVtable = make_vtable(std::make_index_sequence<kSlots>{});
GameInputStub g_gameinput{kVtable.data()};

HRESULT __stdcall stub_qi(GameInputStub*, const IID& riid, void** out) {
    if (!out) return E_POINTER;
    // An interface pointer is handed back for any IID: the measured call site holds this object only to AddRef and
    // Release it, and the other v0 interfaces answer E_NOTIMPL on their slots rather than faulting on a null.
    (void)riid;
    *out = &g_gameinput;
    g_refs.fetch_add(1);
    return S_OK;
}

ULONG __stdcall stub_add_ref(GameInputStub*) { return g_refs.fetch_add(1) + 1; }

ULONG __stdcall stub_release(GameInputStub*) { return g_refs.fetch_sub(1) - 1; }

HRESULT WINAPI GameInputCreate(void** out) {
    if (!out) return E_POINTER;
    g_refs.fetch_add(1);
    *out = &g_gameinput;
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] gameinput: served without SDL3 (no devices; E_NOTIMPL for every other slot)\n");
    return S_OK;
}

}  // namespace

SHIM_TABLE(gameinput_no_sdl,
    {"gameinput.dll", "GameInputCreate", reinterpret_cast<void*>(&GameInputCreate)},
    // The retail executable imports this one by ordinal, so the registry key is "#1"; without the alias the loader
    // fell through to the host's gameinput.dll, which handed the title a null interface.
    {"gameinput.dll", "#1", reinterpret_cast<void*>(&GameInputCreate)},
);
