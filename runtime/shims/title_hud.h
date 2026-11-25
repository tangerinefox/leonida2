#pragma once
// LEONIDA_TITLE_HUD=1 shows live counters in the title bar of the window the title presents in:
//   frame rate and frame time from PresentX (average and worst frame of the last half second), or how long
//   since the last frame when there was none;
//   the process's private memory and its GPU memory (DXGI QueryVideoMemoryInfo, local and non-local segments);
//   graphics and compute pipelines created, and the seconds spent in vkCreate*Pipelines in the last interval,
//   which is where an inline compile shows up as a hitch (needs LEONIDA_MEM_PROBE, src/mem_probe.h; else "-");
//   time since the first present.
// The text is set from a thread of its own with SendMessageTimeout, so a busy or hung window thread never holds
// the present path.

#include <windows.h>
#include <dxgi1_4.h>
#include <psapi.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <thread>

#include "../src/mem_probe.h"

namespace rt::d3d12_xs {

class TitleHud {
public:
    static bool enabled() {
        static const bool on = [] {
            char v[2] = {};
            return GetEnvironmentVariableA("LEONIDA_TITLE_HUD", v, sizeof v) == 1 && v[0] != '0';
        }();
        return on;
    }

    explicit TitleHud(HWND window) : window_(window) {
        QueryPerformanceFrequency(&frequency_);
        stop_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (stop_) thread_ = std::thread([this] { run(); });
    }
    TitleHud(const TitleHud&) = delete;
    TitleHud& operator=(const TitleHud&) = delete;
    ~TitleHud() {
        if (stop_) SetEvent(stop_);
        if (thread_.joinable()) thread_.join();
        if (stop_) CloseHandle(stop_);
    }

    // On every successful PresentX.
    void note_present() {
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        const std::int64_t previous = last_present_.exchange(now.QuadPart, std::memory_order_relaxed);
        if (previous) {
            const std::int64_t gap = now.QuadPart - previous;
            std::int64_t worst = worst_gap_.load(std::memory_order_relaxed);
            while (gap > worst && !worst_gap_.compare_exchange_weak(worst, gap, std::memory_order_relaxed)) {}
        } else {
            first_present_.store(now.QuadPart, std::memory_order_relaxed);
        }
        presents_.fetch_add(1, std::memory_order_relaxed);
    }

private:
    static constexpr DWORD interval_ms = 500;

    void run() {
        Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter = gpu_adapter();
        std::uint64_t last_presents = 0, last_compile_ns = 0;
        LARGE_INTEGER last;
        QueryPerformanceCounter(&last);
        while (WaitForSingleObject(stop_, interval_ms) == WAIT_TIMEOUT) {
            if (!IsWindow(window_)) return;
            LARGE_INTEGER now;
            QueryPerformanceCounter(&now);
            const double seconds = double(now.QuadPart - last.QuadPart) / double(frequency_.QuadPart);
            last = now;
            const std::uint64_t presents = presents_.load(std::memory_order_relaxed);
            const std::uint64_t frames = presents - last_presents;
            last_presents = presents;
            const double worst_ms =
                double(worst_gap_.exchange(0, std::memory_order_relaxed)) * 1000.0 / double(frequency_.QuadPart);
            const double fps = seconds > 0 ? double(frames) / seconds : 0.0;
            const double frame_ms = frames ? seconds * 1000.0 / double(frames) : 0.0;

            wchar_t memory[96] = L"RAM -";
            PROCESS_MEMORY_COUNTERS_EX counters{};
            counters.cb = sizeof counters;
            if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                                     sizeof counters))
                std::swprintf(memory, 96, L"RAM %.1f GB (ws %.1f)", gb(counters.PrivateUsage),
                              gb(counters.WorkingSetSize));

            wchar_t video[96] = L"VRAM -";
            DXGI_QUERY_VIDEO_MEMORY_INFO local{}, shared{};
            if (adapter && SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)) &&
                SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &shared)))
                std::swprintf(video, 96, L"VRAM %.1f/%.1f GB +%.1f shared", gb(local.CurrentUsage),
                              gb(local.Budget), gb(shared.CurrentUsage));

            wchar_t pipelines[128] = L"PSO -";
            auto& probe = leonida::mem_probe::state();
            if (probe.installed) {
                std::uint64_t graphics, compute, ns;
                {
                    std::lock_guard lock(probe.mutex);
                    graphics = probe.graphics_made;
                    compute = probe.compute_made;
                    ns = probe.graphics_ns + probe.compute_ns;
                }
                const double compiling = double(ns - last_compile_ns) / 1e9;
                last_compile_ns = ns;
                std::swprintf(pipelines, 128, L"PSO %llug %lluc, compiling %.1f s", (unsigned long long)graphics,
                              (unsigned long long)compute, compiling);
            }

            const std::int64_t first = first_present_.load(std::memory_order_relaxed);
            const unsigned up = first ? unsigned((now.QuadPart - first) / frequency_.QuadPart) : 0;

            // No present in this interval: how long since the last one, which tells a hang from a slow frame.
            wchar_t frame[96];
            const std::int64_t last_present = last_present_.load(std::memory_order_relaxed);
            if (frames || !last_present)
                std::swprintf(frame, 96, L"%.0f fps %.1f ms (worst %.0f)", fps, frame_ms, worst_ms);
            else
                std::swprintf(frame, 96, L"0 fps, no frame for %.1f s",
                              double(now.QuadPart - last_present) / double(frequency_.QuadPart));
            wchar_t text[512];
            std::swprintf(text, 512, L"Leonida | %ls | %ls | %ls | %ls | %u:%02u", frame, memory, video, pipelines,
                          up / 60, up % 60);
            DWORD_PTR ignored = 0;
            SendMessageTimeoutW(window_, WM_SETTEXT, 0, reinterpret_cast<LPARAM>(text), SMTO_ABORTIFHUNG,
                                interval_ms, &ignored);
        }
    }

    static double gb(std::uint64_t bytes) { return double(bytes) / double(1ull << 30); }

    // The hardware adapter with the most dedicated memory: the one LEONIDA_VK_DEVICE picks on a machine with a
    // discrete GPU. The host's own dxgi.dll, not the title's shim.
    static Microsoft::WRL::ComPtr<IDXGIAdapter3> gpu_adapter() {
        using Create = HRESULT(WINAPI*)(REFIID, void**);
        HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
        auto create = dxgi ? reinterpret_cast<Create>(GetProcAddress(dxgi, "CreateDXGIFactory1")) : nullptr;
        Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
        Microsoft::WRL::ComPtr<IDXGIAdapter3> best;
        if (!create || FAILED(create(IID_PPV_ARGS(&factory)))) return best;
        SIZE_T most = 0;
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
            Microsoft::WRL::ComPtr<IDXGIAdapter3> three;
            if (desc.DedicatedVideoMemory > most && SUCCEEDED(adapter.As(&three))) {
                most = desc.DedicatedVideoMemory;
                best = three;
            }
        }
        return best;
    }

    HWND window_;
    HANDLE stop_ = nullptr;
    std::thread thread_;
    LARGE_INTEGER frequency_{};
    std::atomic<std::int64_t> last_present_{0}, first_present_{0}, worst_gap_{0};
    std::atomic<std::uint64_t> presents_{0};
};

}  // namespace rt::d3d12_xs
