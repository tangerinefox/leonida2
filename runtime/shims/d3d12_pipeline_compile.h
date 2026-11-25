// Timing and scheduling for Vulkan pipeline creation on the D3D12 device.
#pragma once

#include "../src/vulkan_replay.h"
#include <windows.h>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include <string>

namespace rt::d3d12 {
inline bool trace_pipeline_compile() {
    static const bool on = [] {
        wchar_t v[2]{};
        return GetEnvironmentVariableW(L"LEONIDA_TRACE_PIPELINE_COMPILE", v, 2) == 1 && v[0] == L'1';
    }();
    return on;
}

inline bool async_replay_compile() {
    static const bool on = [] {
        wchar_t v[2]{};
        return GetEnvironmentVariableW(L"LEONIDA_ASYNC_PIPELINES", v, 2) == 1 && v[0] == L'1';
    }();
    return on && !leonida::strict_replay();
}

// Absolute monotonic timestamps let trace readers distinguish time in the ICD from time queued for the worker.
using CompileClock = std::chrono::steady_clock;
inline long long compile_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(CompileClock::now().time_since_epoch()).count();
}

inline long long compile_wall_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

template<class Build> VkPipeline replay_pipeline(const char* trigger, Build&& get) {
    if (!trace_pipeline_compile()) return get();
    const auto start = compile_us();
    auto handle = get();
    auto elapsed = compile_us() - start;
    if (elapsed >= 1000)
        std::fprintf(stderr, "[rt] pipeline-use trigger=%s tid=%lu us=%lld ready=%d end_ms=%lld\n",
                     trigger, GetCurrentThreadId(), elapsed, int(handle != VK_NULL_HANDLE), compile_wall_ms());
    return handle;
}

struct CompileTrace {
    VkPipelineCreationFeedback feedback{};
    VkPipelineCreationFeedbackCreateInfo chain{VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
                                               nullptr, &feedback, 0, nullptr};
    long long started = 0;
    const char* key = "-";

    template<class Info> void begin(Info& info, bool supported, const char* trigger, const char* shaders,
                                    std::uint32_t variant, std::size_t bytes, const std::string& description_key) {
        if (!trace_pipeline_compile()) return;
        if (!description_key.empty()) key = description_key.c_str();
        if (supported) {
            chain.pNext = info.pNext;
            info.pNext = &chain;
        }
        std::fprintf(stderr, "[rt] pipeline-compile-begin trigger=%s tid=%lu bytes=%zu variant=%u begin_ms=%lld key=%s shaders=%s\n",
                     trigger, GetCurrentThreadId(), bytes, variant, compile_wall_ms(), key, shaders);
        started = compile_us();
    }

    void end(const char* trigger, const char* shaders, std::uint32_t variant, std::size_t bytes, VkResult result) const {
        if (!started) return;
        const auto ended = compile_us();
        const int hit = !(feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) ? -1 :
                        int((feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) != 0);
        std::fprintf(stderr, "[rt] pipeline-compile trigger=%s tid=%lu start_us=%lld us=%lld hit=%d bytes=%zu "
                             "variant=%u result=%d end_ms=%lld key=%s shaders=%s\n", trigger, GetCurrentThreadId(), started,
                             ended - started, hit, bytes, variant, int(result), compile_wall_ms(), key, shaders);
    }
};


// Demanded work has priority zero, speculative work is ordered by module size, and
// an idle checkpoint runs last. Jobs own their object until they finish. No Vulkan
// call runs with the queue lock held. The worker outlives individual D3D12 devices.
class PipelineCompiler {
public:
    static PipelineCompiler& instance() {
        static auto* pool = new PipelineCompiler;
        return *pool;
    }
    void submit(std::size_t priority, std::function<void()> work) {
        std::lock_guard lock(mutex_);
        if (!started_) {
            std::thread([this] { run(); }).detach();
            started_ = true;
        }
        jobs_.emplace(priority, std::move(work));
        wake_.notify_one();
    }
    bool has_compile_waiter() {
        std::lock_guard lock(mutex_);
        return !jobs_.empty() && jobs_.begin()->first != SIZE_MAX;
    }
    // Returns when no job is queued or running. A job owns its pipeline state, and through it the device, until it
    // finishes; a caller that counts references (rt::d3d12_xs::device_proxy_ref_count) waits here first.
    void wait_idle() {
        std::unique_lock wait(mutex_);
        idle_.wait(wait, [this] { return jobs_.empty() && !running_; });
    }
private:
    void run() {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock wait(mutex_);
                wake_.wait(wait, [this] { return !jobs_.empty(); });
                auto first = jobs_.begin();
                job = std::move(first->second);
                jobs_.erase(first);
                running_ = true;
            }
            job();
            job = nullptr;  // what the job captured goes before the pool reports itself idle
            {
                std::lock_guard done(mutex_);
                running_ = false;
            }
            idle_.notify_all();
        }
    }
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable idle_;
    bool running_ = false;
    std::multimap<std::size_t, std::function<void()>> jobs_;
    bool started_ = false;
};
}  // namespace rt::d3d12
