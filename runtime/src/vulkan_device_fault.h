#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <cstring>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace leonida {

struct DeviceFaultExtensions {
    bool fault = false;
    bool checkpoints = false;
    bool diagnostics_config = false;
};

// What a device asks for. LEONIDA_DEVICE_FAULT=1 wants all of it; LEONIDA_AFTERMATH=1 alone (vulkan_aftermath.h) only
// needs the NVIDIA diagnostics config, whose flags turn on the Aftermath features of the GPU crash dump.
struct DeviceFaultRequest {
    bool fault = true;               // VK_EXT_device_fault and VK_NV_device_diagnostic_checkpoints
    bool diagnostics_config = true;  // VK_NV_device_diagnostics_config
    VkDeviceDiagnosticsConfigFlagsNV extra_flags = 0;  // added to the default config flags below
};

// Owns the feature structures until vkCreateDevice returns.
struct DeviceFaultFeatures {
    VkPhysicalDeviceFaultFeaturesEXT fault{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};
    VkPhysicalDeviceDiagnosticsConfigFeaturesNV diagnostics{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES_NV};
    VkDeviceDiagnosticsConfigCreateInfoNV config{VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV};

    DeviceFaultExtensions enable(VkPhysicalDevice physical, PFN_vkGetPhysicalDeviceFeatures2 query,
                                 std::span<const VkExtensionProperties> advertised,
                                 std::vector<const char*>& extensions, VkDeviceCreateInfo& device,
                                 DeviceFaultRequest request = {}) {
        const auto has = [&](const char* name) {
            return std::any_of(advertised.begin(), advertised.end(), [name](const auto& ext) {
                return std::strcmp(ext.extensionName, name) == 0;
            });
        };
        const bool has_fault = request.fault && has(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
        const bool has_config = request.diagnostics_config && has(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME);
        if (has_fault || has_config) {
            diagnostics.pNext = nullptr;
            fault.pNext = has_config ? &diagnostics : nullptr;
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            features.pNext = has_fault ? static_cast<void*>(&fault) : static_cast<void*>(&diagnostics);
            query(physical, &features);
        }
        const DeviceFaultExtensions enabled{has_fault && fault.deviceFault == VK_TRUE,
                                            request.fault && has(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME),
                                            has_config && diagnostics.diagnosticsConfig == VK_TRUE};
        if (enabled.fault) {
            extensions.push_back(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
            fault.pNext = const_cast<void*>(device.pNext);
            device.pNext = &fault;
        }
        if (enabled.checkpoints) extensions.push_back(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
        if (enabled.diagnostics_config) {
            extensions.push_back(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME);
            diagnostics.pNext = const_cast<void*>(device.pNext);
            config.pNext = &diagnostics;
            config.flags = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV |
                           VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV |
                           VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV | request.extra_flags;
            device.pNext = &config;
        }
        return enabled;
    }
};

enum class CheckpointKind : std::uint8_t { draw = 1, dispatch, copy, resolve, clear };

struct CheckpointMarker {
    static constexpr std::uint32_t max_list = 0xff;
    static constexpr std::uint32_t max_command = 0x1ffff;
    static constexpr std::uint64_t tag = 0xa;

    std::uint32_t submission = 0;
    std::uint32_t list = 0;
    std::uint32_t command = 0;
    CheckpointKind kind{};

    // Vulkan treats the value as an opaque pointer. The runtime never dereferences it.
    static constexpr std::uintptr_t encode(std::uint32_t submission, std::size_t list, std::size_t command,
                                           CheckpointKind kind) {
        if (sizeof(void*) < 8 || !submission || list > max_list || command > max_command) return 0;
        return static_cast<std::uintptr_t>((tag << 60) | (std::uint64_t{submission} << 28) |
                                         (std::uint64_t{list} << 20) | (std::uint64_t{command} << 3) |
                                         static_cast<std::uint8_t>(kind));
    }

    static constexpr CheckpointMarker decode(std::uintptr_t marker) {
        const std::uint64_t bits = marker;
        const auto kind = static_cast<CheckpointKind>(bits & 7);
        if ((bits >> 60) != tag || kind < CheckpointKind::draw || kind > CheckpointKind::clear) return {};
        return {static_cast<std::uint32_t>(bits >> 28), static_cast<std::uint32_t>((bits >> 20) & max_list),
                static_cast<std::uint32_t>((bits >> 3) & max_command), kind};
    }

    const char* kind_name() const {
        switch (kind) {
        case CheckpointKind::draw: return "draw";
        case CheckpointKind::dispatch: return "dispatch";
        case CheckpointKind::copy: return "copy";
        case CheckpointKind::resolve: return "resolve";
        case CheckpointKind::clear: return "clear";
        default: return "unknown";
        }
    }
};

using CheckpointShaders = std::array<const char*, 3>;  // Immutable ShaderLibrary names, valid until context teardown.

struct CheckpointPipeline {
    VkPipeline handle = VK_NULL_HANDLE;
    std::uintptr_t id = 0;
    CheckpointShaders shaders{};
};

struct CheckpointList {
    struct ShaderChange {
        std::uint32_t command;
        CheckpointPipeline pipeline;
    };
    std::vector<ShaderChange> shaders;
};

// CPU record of work handed to vkQueueSubmit. A driver checkpoint can identify which
// entry it reached; on drivers without checkpoints these remain submitted candidates.
struct CheckpointDispatch {
    std::uint64_t execute_call;
    VkQueue queue;
    const char* queue_kind;
    std::size_t list, command;
    const char* shader;
    std::array<std::uint32_t, 3> groups;
    std::uintptr_t argument_buffer, count_buffer;
    std::uint64_t argument_offset, count_offset;
    std::uint32_t max_count, stride;
    std::array<std::uint32_t, 32> roots;
    bool indirect;
};

struct CheckpointSubmission {
    std::uint32_t sequence = 0;
    std::uint64_t execute_call = 0;
    VkQueue queue = VK_NULL_HANDLE;
    const char* queue_kind = nullptr;
    std::size_t list_count = 0;
    std::vector<CheckpointList> lists;
    std::vector<CheckpointDispatch> dispatches;
    std::size_t dispatch_count = 0;
    static constexpr std::size_t dispatch_limit = 64;

    void append_dispatch(CheckpointDispatch d) {
        if (dispatches.size() < dispatch_limit) dispatches.push_back(d);
        else dispatches[dispatch_count % dispatch_limit] = d;
        ++dispatch_count;
    }
};

struct CheckpointRecorder {
    PFN_vkCmdSetCheckpointNV set;
    CheckpointSubmission& submission;
    const std::unordered_map<std::uintptr_t, CheckpointShaders>& pipeline_names;
    std::size_t list;

    void record(VkCommandBuffer buffer, std::size_t command, CheckpointKind kind,
                VkPipeline pipeline = VK_NULL_HANDLE, std::uintptr_t id = 0) const {
        if (!set) return;
        const auto marker = CheckpointMarker::encode(submission.sequence, list, command, kind);
        if (!marker) return;
        if (kind == CheckpointKind::draw || kind == CheckpointKind::dispatch) {
            auto& shaders = submission.lists[list].shaders;
            if (shaders.empty() || shaders.back().pipeline.handle != pipeline || shaders.back().pipeline.id != id) {
                const auto names = pipeline_names.find(id);
                shaders.push_back({static_cast<std::uint32_t>(command),
                                   {pipeline, id, names != pipeline_names.end() ? names->second : CheckpointShaders{}}});
            }
        }
        set(buffer, reinterpret_cast<const void*>(marker));
    }

    void dispatch(VkCommandBuffer buffer, std::size_t command, VkPipeline pipeline, std::uintptr_t id,
                  const std::array<std::uint32_t, 32>& roots, std::array<std::uint32_t, 3> groups,
                  bool indirect = false, std::uintptr_t args = 0, std::uint64_t offset = 0,
                  std::uintptr_t count = 0, std::uint64_t count_offset = 0,
                  std::uint32_t max_count = 0, std::uint32_t stride = 0) const {
        const auto names = pipeline_names.find(id);
        submission.append_dispatch({submission.execute_call, submission.queue, submission.queue_kind, list, command,
            names != pipeline_names.end() ? names->second[0] : nullptr, groups, args, count,
            offset, count_offset, max_count, stride, roots, indirect});
        record(buffer, command, CheckpointKind::dispatch, pipeline, id);
    }
};

// A D3D fence may stay unsignalled by design. Only queue retirement tickets represent
// work that actually reached vkQueueSubmit.
class SubmissionProgressWatch {
public:
    using Clock = std::chrono::steady_clock;
    enum class State { idle, pending, submit_lock_busy };

    bool sample(State state, std::uint64_t retired, Clock::time_point now) {
        if (state == State::idle || state != state_ || retired != retired_) {
            state_ = state;
            retired_ = retired;
            since_ = now;
            reported_ = false;
            return false;
        }
        if (reported_ || now - since_ < std::chrono::seconds(15)) return false;
        reported_ = true;
        return true;
    }

private:
    State state_ = State::idle;
    std::uint64_t retired_ = 0;
    Clock::time_point since_{};
    bool reported_ = false;
};

// Runs on every VK_ERROR_DEVICE_LOST that DeviceFaultDiagnostics::observe or VulkanContext::mark_device_lost sees,
// with or without LEONIDA_DEVICE_FAULT. Set before the device exists by the Aftermath integration
// (vulkan_aftermath.h), which blocks there until its GPU crash dump is written.
inline std::atomic<void (*)() noexcept> device_lost_hook{nullptr};

inline void run_device_lost_hook() noexcept {
    if (const auto hook = device_lost_hook.load(std::memory_order_acquire)) hook();
}

class DeviceFaultDiagnostics {
public:
    DeviceFaultDiagnostics(PFN_vkGetDeviceProcAddr get, VkDevice device, DeviceFaultExtensions enabled)
        : device_(device) {
#define LEONIDA_FAULT_LOAD(name) name = reinterpret_cast<PFN_##name>(get(device, #name))
        if (enabled.fault) LEONIDA_FAULT_LOAD(vkGetDeviceFaultInfoEXT);
        if (enabled.checkpoints) {
            LEONIDA_FAULT_LOAD(vkCmdSetCheckpointNV);
            LEONIDA_FAULT_LOAD(vkGetQueueCheckpointDataNV);
        }
#undef LEONIDA_FAULT_LOAD
        std::fprintf(stderr, "[rt] device-fault: enabled (fault info %s, checkpoints %s, NVIDIA diagnostics config %s)\n",
                     vkGetDeviceFaultInfoEXT ? "on" : "unavailable", checkpoint_command() ? "on" : "unavailable",
                     enabled.diagnostics_config ? "on" : "unavailable");
    }

    PFN_vkCmdSetCheckpointNV checkpoint_command() const {
        return sizeof(void*) >= 8 && vkGetQueueCheckpointDataNV ? vkCmdSetCheckpointNV : nullptr;
    }

    void add_queue(VkQueue queue) {
        if (!queue || !vkGetQueueCheckpointDataNV) return;
        std::lock_guard lock(mutex_);
        if (std::find(queues_.begin(), queues_.end(), queue) == queues_.end()) queues_.push_back(queue);
    }

    std::uint64_t next_execute_call() { return ++execute_calls_; }

    // Accessed under the context's resource lock. Reports read the names copied into the ring instead.
    std::unordered_map<std::uintptr_t, CheckpointShaders> pipeline_names;

    // Called under the context's tracker lock, in replay order.
    CheckpointSubmission begin_submission(std::uint64_t call, VkQueue queue, const char* kind, std::size_t lists) {
        if (!++sequence_) ++sequence_;  // 0 is reserved for invalid markers
        return {sequence_, call, queue, kind, lists,
                std::vector<CheckpointList>(std::min(lists, std::size_t{CheckpointMarker::max_list} + 1))};
    }

    void publish(CheckpointSubmission&& submission) {
        std::lock_guard lock(mutex_);
        for (std::size_t i = submission.dispatch_count - submission.dispatches.size(); i < submission.dispatch_count; ++i)
            dispatches_[dispatch_count_++ % dispatches_.size()] = submission.dispatches[i % submission.dispatch_limit];
        submission.dispatches.clear();
        recent_[submission.sequence % recent_.size()] = std::move(submission);
    }

    static VkResult observe(DeviceFaultDiagnostics* diagnostics, VkResult result, const char* operation) noexcept {
        if (result == VK_ERROR_DEVICE_LOST) {
            if (diagnostics) diagnostics->report(operation);
            run_device_lost_hook();
        }
        return result;
    }

    void report_stall(const char* reason, std::uint64_t submitted, std::uint64_t retired) {
        std::fprintf(stderr, "[rt] device-stall: %s for 15 s; submitted ticket %" PRIu64
                             ", last observed retired ticket %" PRIu64 "\n", reason, submitted, retired);
        // NV checkpoints and EXT fault data require a lost device. The CPU ring is valid at any time.
        dump_dispatches();
        // LEONIDA_STALL_CHECKPOINTS=1 (diagnostic): ask for the checkpoints anyway. A hang (Xid 109) the driver
        // recovers from without losing the device otherwise names no program; outside the lost state the call is
        // out of the extension's contract, which the NVIDIA driver has answered so far.
        static const bool stall_checkpoints = [] {
            const char* v = std::getenv("LEONIDA_STALL_CHECKPOINTS");
            return v && std::strcmp(v, "1") == 0;
        }();
        if (!stall_checkpoints || !vkGetQueueCheckpointDataNV) return;
        try {
            std::lock_guard lock(mutex_);
            for (VkQueue queue : queues_) report_queue(queue);
        } catch (...) {
            std::fprintf(stderr, "[rt] device-fault: checkpoint collection failed\n");
        }
    }

private:
    void dump_dispatches() {
        std::lock_guard lock(mutex_);
        const auto n = std::min(dispatch_count_, dispatches_.size());
        for (std::size_t i = dispatch_count_ - n; i < dispatch_count_; ++i) {
            const auto& d = dispatches_[i % dispatches_.size()];
            std::fprintf(stderr, "[rt] device-fault: submitted dispatch seq=%zu execute=%" PRIu64
                         " queue=%p (%s) list=%zu command=%zu shader=%s ", i + 1, d.execute_call,
                         static_cast<void*>(d.queue), d.queue_kind, d.list, d.command, d.shader ? d.shader : "-");
            if (d.indirect)
                std::fprintf(stderr, "indirect args=0x%" PRIxPTR "+0x%" PRIx64 " count=0x%" PRIxPTR
                             "+0x%" PRIx64 " max=%u stride=%u", d.argument_buffer, d.argument_offset,
                             d.count_buffer, d.count_offset, d.max_count, d.stride);
            else std::fprintf(stderr, "groups=%u,%u,%u", d.groups[0], d.groups[1], d.groups[2]);
            if (include_roots_) {
                std::fputs(" roots=", stderr);
                for (auto word : d.roots) std::fprintf(stderr, "%08x ", word);
            }
            std::fputc('\n', stderr);
        }
    }

public:
    // LOCAL DIAGNOSTIC: run once after a loss is reported (the compute input probe rechecks the descriptors it saw).
    void set_lost_hook(std::function<void()> hook) {
        std::lock_guard lock(mutex_);
        lost_hook_ = std::move(hook);
    }

private:
    std::function<void()> lost_hook_;

    void report(const char* operation) noexcept {
        if (reported_.exchange(true, std::memory_order_relaxed)) return;
        std::fprintf(stderr, "[rt] device-fault: VK_ERROR_DEVICE_LOST from %s\n", operation);
        dump_dispatches();
        try {
            std::function<void()> hook;
            {
                std::lock_guard lock(mutex_);
                hook = lost_hook_;
            }
            if (hook) hook();
        } catch (...) {
        }
        try {
            if (vkGetDeviceFaultInfoEXT) report_fault();
        } catch (...) {
            std::fprintf(stderr, "[rt] device-fault: fault info collection failed\n");
        }
        try {
            std::lock_guard lock(mutex_);
            for (VkQueue queue : queues_) report_queue(queue);
        } catch (...) {
            std::fprintf(stderr, "[rt] device-fault: checkpoint collection failed\n");
        }
    }

    void report_fault() const {
        VkDeviceFaultCountsEXT counts{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
        VkResult result = vkGetDeviceFaultInfoEXT(device_, &counts, nullptr);
        if (result != VK_SUCCESS) {
            std::fprintf(stderr, "[rt] device-fault: vkGetDeviceFaultInfoEXT counts failed (%d)\n", int(result));
            return;
        }
        std::fprintf(stderr, "[rt] device-fault: vendor binary size %" PRIu64 " bytes\n", counts.vendorBinarySize);
        std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
        std::vector<VkDeviceFaultVendorInfoEXT> vendors(counts.vendorInfoCount);
        std::vector<std::byte> binary;
        // Keep collecting addresses and vendor records if the crash binary cannot be allocated.
        try {
            if (counts.vendorBinarySize <= binary.max_size())
                binary.resize(static_cast<std::size_t>(counts.vendorBinarySize));
        } catch (...) {
        }
        counts.vendorBinarySize = binary.size();
        VkDeviceFaultInfoEXT fault{VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
        fault.pAddressInfos = addresses.data();
        fault.pVendorInfos = vendors.data();
        fault.pVendorBinaryData = binary.data();
        result = vkGetDeviceFaultInfoEXT(device_, &counts, &fault);
        if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
            std::fprintf(stderr, "[rt] device-fault: vkGetDeviceFaultInfoEXT data failed (%d)\n", int(result));
            return;
        }
        std::fprintf(stderr, "[rt] device-fault: description: %.*s%s\n", int(sizeof(fault.description)),
                     fault.description, result == VK_INCOMPLETE ? " (partial report)" : "");
        for (std::size_t i = 0; i < counts.addressInfoCount; ++i) {
            const auto& address = addresses[i];
            std::fprintf(stderr, "[rt] device-fault: address[%zu] type=%d address=0x%" PRIx64
                                 " precision=%" PRIu64 "\n",
                         i, int(address.addressType), address.reportedAddress, address.addressPrecision);
        }
        for (std::size_t i = 0; i < counts.vendorInfoCount; ++i) {
            const auto& vendor = vendors[i];
            std::fprintf(stderr, "[rt] device-fault: vendor[%zu] %.*s code=0x%" PRIx64 " data=0x%" PRIx64 "\n",
                         i, int(sizeof(vendor.description)), vendor.description, vendor.vendorFaultCode,
                         vendor.vendorFaultData);
        }
    }

    void report_queue(VkQueue queue) const {
        std::uint32_t count = 0;
        vkGetQueueCheckpointDataNV(queue, &count, nullptr);
        std::vector<VkCheckpointDataNV> data(count, {VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
        if (count) vkGetQueueCheckpointDataNV(queue, &count, data.data());
        std::fprintf(stderr, "[rt] device-fault: queue %p checkpoints=%u\n", static_cast<void*>(queue), count);
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto& checkpoint = data[i];
            const auto stage = static_cast<std::uint64_t>(checkpoint.stage);
            const auto marker = CheckpointMarker::decode(reinterpret_cast<std::uintptr_t>(checkpoint.pCheckpointMarker));
            if (!marker.submission) {
                std::fprintf(stderr, "[rt] device-fault: stage=0x%" PRIx64 " marker=%p (unrecognized)\n",
                             stage, checkpoint.pCheckpointMarker);
                continue;
            }
            std::fprintf(stderr, "[rt] device-fault: stage=0x%" PRIx64 " submission=%u list=%u command=%u kind=%s\n",
                         stage, marker.submission, marker.list, marker.command, marker.kind_name());
            const auto& entry = recent_[marker.submission % recent_.size()];
            if (entry.sequence != marker.submission || entry.queue != queue || marker.list >= entry.lists.size()) {
                std::fprintf(stderr, "[rt] device-fault: submission metadata unavailable\n");
                continue;
            }
            std::fprintf(stderr, "[rt] device-fault: ExecuteCommandLists #%" PRIu64 " queue=%p (%s) lists=%zu\n",
                         entry.execute_call, static_cast<void*>(entry.queue), entry.queue_kind, entry.list_count);
            const auto& shaders = entry.lists[marker.list].shaders;
            const auto next = std::upper_bound(shaders.begin(), shaders.end(), marker.command,
                                               [](auto command, const auto& change) { return command < change.command; });
            if (next != shaders.begin()) {
                const auto& pipeline = (next - 1)->pipeline;
                std::fprintf(stderr, "[rt] device-fault: last shader pipeline=0x%" PRIx64 " shaders=%s %s %s\n",
                             pipeline_handle(pipeline.handle), pipeline.shaders[0] ? pipeline.shaders[0] : "-",
                             pipeline.shaders[1] ? pipeline.shaders[1] : "-",
                             pipeline.shaders[2] ? pipeline.shaders[2] : "-");
            }
        }
    }

    template<typename Handle>
    static std::uint64_t pipeline_handle(Handle handle) {
        if constexpr (VK_USE_64_BIT_PTR_DEFINES) return reinterpret_cast<std::uintptr_t>(handle);
        else return handle;
    }

    VkDevice device_;
    PFN_vkGetDeviceFaultInfoEXT vkGetDeviceFaultInfoEXT = nullptr;
    PFN_vkCmdSetCheckpointNV vkCmdSetCheckpointNV = nullptr;
    PFN_vkGetQueueCheckpointDataNV vkGetQueueCheckpointDataNV = nullptr;
    std::atomic<bool> reported_{false};
    std::atomic<std::uint64_t> execute_calls_{0};
    std::uint32_t sequence_ = 0;
    std::mutex mutex_;  // leaf lock, guards queues_ and recent_
    std::vector<VkQueue> queues_;
    const bool include_roots_ = [] {
        const char* value = std::getenv("LEONIDA_DEVICE_FAULT_ROOTS");
        return value && !std::strcmp(value, "1");
    }();
    std::array<CheckpointDispatch, CheckpointSubmission::dispatch_limit> dispatches_{};
    std::size_t dispatch_count_ = 0;
    std::array<CheckpointSubmission, 64> recent_;
};

}  // namespace leonida
