// gameinput.dll!#1 without SDL3: the retail executable imports GameInputCreate by ordinal, and a build without SDL3
// still has to hand back an interface instead of letting the loader fall through to the host's gameinput.dll, which
// answers with a null pointer that the title dereferences.
#include <windows.h>

#include <cstdint>
#include <cstdio>

#include "shims.h"

namespace {
int g_failed = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok   " : "FAIL ", what);
    if (!ok) ++g_failed;
}
}  // namespace

int main() {
    auto create = reinterpret_cast<HRESULT(WINAPI*)(void**)>(rt::shims::find("gameinput.dll", "#1"));
    auto named = reinterpret_cast<HRESULT(WINAPI*)(void**)>(rt::shims::find("gameinput.dll", "GameInputCreate"));
    check(create != nullptr, "the ordinal import gameinput.dll!#1 is served");
    check(named != nullptr, "the name GameInputCreate is served");
    if (!create) return 1;

    void* object = nullptr;
    check(create(&object) == S_OK, "GameInputCreate returns S_OK");
    check(object != nullptr, "GameInputCreate hands back a non-null interface");
    if (!object) return 1;

    auto* vtable = *static_cast<void* const* const*>(object);
    auto add_ref = reinterpret_cast<ULONG(__stdcall*)(void*)>(vtable[1]);
    auto release = reinterpret_cast<ULONG(__stdcall*)(void*)>(vtable[2]);
    check(add_ref(object) >= 1, "slot 1 (AddRef) is callable and counts");
    check(release(object) >= 0, "slot 2 (Release) is callable and does not underflow");

    // A device query answers a clean refusal through the same object rather than faulting.
    auto get_devices = reinterpret_cast<HRESULT(__stdcall*)(void*, uint32_t, void**)>(vtable[9]);
    void* devices = reinterpret_cast<void*>(0x1);
    check(get_devices(object, 0, &devices) == E_NOTIMPL, "an unserved slot returns E_NOTIMPL");
    check(devices == reinterpret_cast<void*>(0x1), "an unserved slot writes nothing to its out-parameter");

    check(create(nullptr) == E_POINTER, "a null out parameter is E_POINTER");
    std::printf("%d failure(s)\n", g_failed);
    return g_failed ? 1 : 0;
}
