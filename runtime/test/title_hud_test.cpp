// shims/title_hud.h: LEONIDA_TITLE_HUD's counters reach the title bar of a window owned by another thread, and a
// window that stops pumping messages does not hold the HUD thread. A real Win32 window; no GPU needed.
#include "title_hud.h"

#include <windows.h>

#include <cstdio>
#include <string>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

static std::wstring title_of(HWND window) {
    wchar_t text[512] = {};
    InternalGetWindowText(window, text, 512);  // no message: the window's own thread may be busy
    return text;
}

// Pumps this thread's messages for `ms`, presenting every 16 ms when `present` is set.
static void pump(DWORD ms, rt::d3d12_xs::TitleHud* present) {
    const ULONGLONG end = GetTickCount64() + ms;
    while (GetTickCount64() < end) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (present) present->note_present();
        Sleep(16);
    }
}

int main() {
    WNDCLASSW cls{};
    cls.lpfnWndProc = DefWindowProcW;
    cls.hInstance = GetModuleHandleW(nullptr);
    cls.lpszClassName = L"LeonidaTitleHudTest";
    CHECK(RegisterClassW(&cls));
    HWND window = CreateWindowExW(WS_EX_NOACTIVATE, cls.lpszClassName, L"before", WS_OVERLAPPEDWINDOW, 0, 0, 64, 64,
                                  nullptr, nullptr, cls.hInstance, nullptr);
    CHECK(window);
    {
        rt::d3d12_xs::TitleHud hud(window);
        // Presents at ~60 Hz for 1.5 s: the title shows a frame rate and the counters.
        pump(1500, &hud);
        const std::wstring text = title_of(window);
        std::fwprintf(stderr, L"title: %ls\n", text.c_str());
        CHECK(text.rfind(L"Leonida | ", 0) == 0);
        CHECK(text.find(L" fps ") != std::wstring::npos && text.find(L" ms (worst ") != std::wstring::npos);
        CHECK(text.find(L"RAM ") != std::wstring::npos && text.find(L"VRAM ") != std::wstring::npos);
        CHECK(text.find(L"PSO -") != std::wstring::npos);  // no LEONIDA_MEM_PROBE: no pipeline counters
        const int fps = _wtoi(text.c_str() + 10);
        CHECK(fps >= 10 && fps <= 120);  // a loose band: Sleep(16) on a loaded machine is not 60 Hz

        // Presents stop while the window still pumps: the title says how long since the last frame.
        pump(1300, nullptr);
        const std::wstring stalled = title_of(window);
        std::fwprintf(stderr, L"title: %ls\n", stalled.c_str());
        CHECK(stalled.find(L"0 fps, no frame for ") != std::wstring::npos);

        // The window's thread stops pumping: the HUD's SendMessageTimeout gives up and the HUD thread is not stuck,
        // so the destructor below returns.
        Sleep(1200);
    }
    const ULONGLONG before = GetTickCount64();
    {
        rt::d3d12_xs::TitleHud quick(window);
    }  // stops at once, without waiting a whole interval
    CHECK(GetTickCount64() - before < 1000);
    DestroyWindow(window);
    std::printf("title_hud: counters in the title bar of another thread's window; a busy window does not hold it\n");
    return 0;
}
