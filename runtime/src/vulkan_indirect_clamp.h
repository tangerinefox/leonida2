// GPU-side clamping of D3D12 ExecuteIndirect arguments, for the replay (vulkan_replay.h).
//
// The title's own compute passes write most of these arguments, and a pass the replay drops (an async pipeline that
// is not built yet, a stub pipeline, LEONIDA_SKIP_COMPUTE) or mistranslates leaves whatever the memory held: a stale
// or garbage record makes a dispatch of billions of groups or a draw of four billion vertices run until the host
// resets the display driver (TDR, device loss). So, just before the indirect command, one tiny compute pass copies
// the active records into device-local scratch with their values bounded - 2^24 vertices or indices, 2^26 vertex
// instances, 65 535 groups per dimension, 2^20 groups per dispatch, and a 2^27-unit work budget per command
// list (the shader is indirect_clamp.comp) - and the indirect command reads that scratch. The application's argument
// and count buffers are left untouched, so values within the limits replay bit-identically.
//
// One pass covers all four argument kinds the layer can emit (draw, draw indexed, dispatch, mesh dispatch) and the
// counted variants, whose count is clamped to max_command_count as well (a *Count draw then reads a count the scratch
// really holds).
//
// LEONIDA_INDIRECT_CLAMP=0 disables it: the indirect command reads the application's argument buffer, as before.
//
// The pass is recorded before anything reads the arguments, and before the drawing scope an indirect draw opens
// around itself (vkCmdCopyBuffer and a memory barrier with a transfer access are not allowed inside a
// dynamic-rendering scope, VUID-vkCmdCopyBuffer-renderpass / VUID-vkCmdPipelineBarrier-None-09586). That is where
// `vulkan_replay.h` emits it for the legacy ExecuteIndirect commands, and where a stateful indirect draw - the
// workspace branch that records a Draw with the arguments it reads, opening its own rendering scope - emits it too:
// the same prepare/emit pair, from `Prepared` or its own prepared draw, before vkCmdBeginRendering. Nothing here
// depends on that branch.
//
// Scratch comes from a per-submission pool (vulkan_scratch.h): handed out while a list is prepared, retired with the
// submission's timeline ticket, reused once the GPU is past it - so a list with many indirect calls allocates no
// buffer per call. The only other state per call is a 16-byte counter slot in one persistent host-visible buffer,
// read (and logged rate-limited) once the submission that used it has completed. The counters are diagnostics: a call
// that finds no free slot is still clamped, without its counters.
#pragma once
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "submit_phases.h"
#include "indirect_clamp_spv.h"
#include "vulkan_indirect_dispatch_count.h"
#include "vulkan_aftermath.h"
#include "vulkan_memory.h"
#include "vulkan_scratch.h"

namespace leonida {

// Argument kinds, numbered as leonida::IndirectKind (command_list.h): the shader reads them.
inline constexpr std::uint32_t indirect_clamp_draw = 0;
inline constexpr std::uint32_t indirect_clamp_draw_indexed = 1;
inline constexpr std::uint32_t indirect_clamp_dispatch = 2;
inline constexpr std::uint32_t indirect_clamp_mesh = 3;

// What the pass bounds, as indirect_clamp.comp states them.
inline constexpr std::uint32_t indirect_clamp_max_vertices = 1u << 24;
inline constexpr std::uint32_t indirect_clamp_max_vertex_instances = 1u << 26;
inline constexpr std::uint32_t indirect_clamp_max_group_dimension = 65535u;
inline constexpr std::uint32_t indirect_clamp_max_groups = 1u << 20;
inline constexpr std::uint32_t indirect_clamp_budget = 1u << 27;

// The largest scratch allocation accepted for one ExecuteIndirect call.
inline constexpr std::uint64_t indirect_clamp_max_scratch_bytes = std::uint64_t(64) << 20;

// Read once: LEONIDA_INDIRECT_CLAMP=0 leaves the application's argument buffer to the indirect command.
inline bool indirect_clamp_enabled() {
    static const bool enabled = [] {
#ifdef _WIN32
        char value[3]{};
        std::size_t length = 0;
        if (getenv_s(&length, value, sizeof value, "LEONIDA_INDIRECT_CLAMP") != 0 || !length) return true;
        return !(length == 2 && value[0] == '0' && !value[1]);
#else
        const char* value = std::getenv("LEONIDA_INDIRECT_CLAMP");
        return !value || value[0] != '0' || value[1];
#endif
    }();
    return enabled;
}

// The record size of each kind, as Vulkan's indirect commands define it and D3D12's argument structs match.
inline constexpr std::uint32_t indirect_clamp_record_size(std::uint32_t kind) {
    return kind == indirect_clamp_draw ? 4u * 4u : kind == indirect_clamp_draw_indexed ? 5u * 4u : 3u * 4u;
}

// Where a clamped ExecuteIndirect call's arguments are, and what its clamp pass needs to record it.
struct ClampedIndirect {
    // The command the replay then emits: the clamped records, and for a counted draw the clamped count word.
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkBuffer count_buffer = VK_NULL_HANDLE;
    VkDeviceSize count_offset = 0;
    std::uint32_t stride = 0;   // bytes per clamped record: 3, 4 or 5 words by kind
    std::uint32_t records = 0;  // max_command_count
    bool counted = false;

    // The clamp pass itself (emit_indirect_clamp).
    VkDeviceAddress arguments = 0;  // the application's first record
    VkDeviceAddress count = 0;      // the application's count word, 0 when uncounted
    VkDeviceAddress records_out = 0;
    VkDeviceAddress count_out = 0;
    VkDeviceAddress stats = 0;
    VkDeviceAddress list_budget = 0;
    void share_list_budget(VkDeviceAddress& first) {
        if (!first) first = list_budget;
        list_budget = first;
    }
    VkBuffer stats_buffer = VK_NULL_HANDLE;  // the host-visible counter slot the pass copies its counters to
    VkDeviceSize stats_offset = 0;
    VkDeviceSize stats_scratch_offset = 0;  // where in `buffer` the pass's own counters are
    std::uint32_t kind = 0;
    std::uint32_t stride_words = 0;
    std::uint32_t max_records = 0;
};

// One device's clamp: the pass's pipeline (made on the first call), the per-submission record pool, and the counter
// slots. Single-threaded per list, as the replay's preparation is: the caller holds the context's tracker lock for
// prepare and emit, and the batch destructor only touches the slot bookkeeping under `mutex_`.
class IndirectArgumentClamp {
private:
    struct Slot {
        std::uint32_t area;
        std::uint32_t index;
    };
    static constexpr std::uint32_t no_slot = UINT32_MAX;  // Slot::area of a call clamped without counters

public:
    using Allocate = ScratchBuffers::Allocate;

    IndirectArgumentClamp(PFN_vkGetDeviceProcAddr get, VkDevice device, const VkPhysicalDeviceMemoryProperties& memory,
                          const VkPhysicalDeviceLimits& limits, Allocate allocate,
                          DeviceFaultDiagnostics* diagnostics = nullptr);
    ~IndirectArgumentClamp();
    IndirectArgumentClamp(const IndirectArgumentClamp&) = delete;
    IndirectArgumentClamp& operator=(const IndirectArgumentClamp&) = delete;

    // False when the pass cannot run on this device (no device addresses, no compute): the replay then keeps the
    // application's buffer.
    bool loaded() const { return scratch_.loaded() && limits_.maxComputeWorkGroupCount[0] != 0; }

    // The record pool's life cycle, as the copy scratch's (vulkan_scratch.h).
    void collect() { scratch_.collect(); }
    void retire(VkSemaphore timeline, std::uint64_t ticket) { scratch_.retire(timeline, ticket); }
    void forget(VkSemaphore timeline) { scratch_.forget(timeline); }

    // What the pass found since the process started. `records_clamped` counts the records whose values the pass
    // changed (the ones it emptied included).
    struct Counters {
        std::uint64_t calls = 0;
        std::uint64_t records = 0;
        std::uint64_t records_clamped = 0;
        std::uint64_t counts_clamped = 0;
        std::uint64_t over_budget = 0;
        std::uint64_t zeroed = 0;
        std::uint64_t unclamped_calls = 0;  // the fallbacks: calls the pool or the limits left to the replay
        std::uint64_t uncounted_calls = 0;  // clamped with no counter slot free: their counters are not read back
    };

    // One submission's clamp state: the counter slots its calls claimed, read and released together once that
    // submission has completed. Retained by the command-buffer slot until its timeline ticket does.
    class Batch {
    public:
        explicit Batch(IndirectArgumentClamp& owner) : owner_(owner) {}
        ~Batch() { owner_.release(slots_, true); }
        Batch(const Batch&) = delete;
        Batch& operator=(const Batch&) = delete;

        // Reserve scratch for one command. A host-visible counter slot is optional.
        bool prepare(VkBuffer arguments, VkDeviceSize argument_offset, VkBuffer count, VkDeviceSize count_offset,
                     std::uint32_t kind, std::uint32_t stride, std::uint32_t max_count, ClampedIndirect& out) {
            // LOCAL FIX (post-load device loss): a submission after the story load carries thousands of
            // ExecuteIndirect calls, far more than the counter slots, and each call that found none used to fall back
            // to the application's buffer - unclamped, over arguments that skipped compute never wrote, while one
            // submission blocked in the driver for 10 s and the device was lost. Out of slots, the call is clamped all
            // the same; only its counters are not read back.
            Slot slot{};
            const bool counted = owner_.claim_slot(slot);
            if (!counted) {
                owner_.note_uncounted();
                slot = Slot{no_slot, 0};
            }
            if (!owner_.prepare(arguments, argument_offset, count, count_offset, kind, stride, max_count, slot, out)) {
                if (counted) owner_.release_one(slot);
                return false;
            }
            if (counted) slots_.push_back(slot);
            return true;
        }

        std::size_t calls() const { return slots_.size(); }

    private:
        IndirectArgumentClamp& owner_;
        std::vector<Slot> slots_;
    };

    std::unique_ptr<Batch> batch() { return std::make_unique<Batch>(*this); }

    // Records one clamp pass: its counters filled, the dispatch, and the counters copied back to their slot. Emitted
    // before the indirect command that reads `job`, and before any rendering scope that command runs in.
    void emit(VkCommandBuffer cb, const ClampedIndirect& job);

    Counters counters() const {
        std::lock_guard lock(mutex_);
        return counters_;
    }

private:
    // A host-visible buffer of counter slots: what the pass's counters are copied into, 16 bytes each.
    struct HostArea {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        std::uint8_t* host = nullptr;
        std::uint32_t slots = 0;
        static constexpr std::uint32_t slot_bytes = 32;  // 16 bytes of counters, 16 of padding
    };
    static constexpr std::uint32_t slots_per_area = 64;
    // 4,096 slots (128 KiB of host memory): enough for a submission's ExecuteIndirect calls after the load, and the
    // calls past them are still clamped (no_slot).
    static constexpr std::uint32_t max_areas = 64;

    template<typename T> T fn(const char* name) const { return reinterpret_cast<T>(get_(device_, name)); }
    bool make_pipeline();
    bool make_area();
    bool claim_slot(Slot& out);
    void release_one(Slot slot);
    void release(std::vector<Slot>& slots, bool read);
    bool prepare(VkBuffer arguments, VkDeviceSize argument_offset, VkBuffer count, VkDeviceSize count_offset,
                 std::uint32_t kind, std::uint32_t stride, std::uint32_t max_count, Slot slot, ClampedIndirect& out);
    void fallback(const char* reason);
    void note_uncounted();

    PFN_vkGetDeviceProcAddr get_;
    VkDevice device_;
    VkPhysicalDeviceMemoryProperties memory_{};
    VkPhysicalDeviceLimits limits_{};
    DeviceFaultDiagnostics* diagnostics_ = nullptr;
    ScratchBuffers scratch_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    mutable std::mutex mutex_;  // the slot bookkeeping and the counters
    std::vector<HostArea> areas_;
    std::vector<Slot> free_slots_;
    Counters counters_{};
    std::uint64_t next_report_frame_ = 0;
};

namespace detail {

// One clamp pass's push-constant block. The shader's block is the same: five device addresses, then the six words.
struct IndirectClampPush {
    VkDeviceAddress arguments;
    VkDeviceAddress count;
    VkDeviceAddress records_out;
    VkDeviceAddress count_out;
    VkDeviceAddress stats;
    VkDeviceAddress list_budget;
    std::uint32_t kind;
    std::uint32_t stride_words;
    std::uint32_t records;
    std::uint32_t first_record;
    std::uint32_t max_records;
    std::uint32_t flags;
};
static_assert(sizeof(IndirectClampPush) == 72);

inline constexpr VkBufferUsageFlags indirect_clamp_scratch_usage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

}  // namespace detail

inline IndirectArgumentClamp::IndirectArgumentClamp(PFN_vkGetDeviceProcAddr get, VkDevice device,
                                                    const VkPhysicalDeviceMemoryProperties& memory,
                                                    const VkPhysicalDeviceLimits& limits, Allocate allocate,
                                                    DeviceFaultDiagnostics* diagnostics)
    : get_(get), device_(device), memory_(memory), limits_(limits), diagnostics_(diagnostics),
      scratch_(get, device, std::move(allocate), diagnostics, detail::indirect_clamp_scratch_usage) {}

inline IndirectArgumentClamp::~IndirectArgumentClamp() {
    const auto destroy_pipeline = fn<PFN_vkDestroyPipeline>("vkDestroyPipeline");
    const auto destroy_layout = fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout");
    if (pipeline_ && destroy_pipeline) destroy_pipeline(device_, pipeline_, nullptr);
    if (layout_ && destroy_layout) destroy_layout(device_, layout_, nullptr);
    const auto destroy_buffer = fn<PFN_vkDestroyBuffer>("vkDestroyBuffer");
    const auto free_memory = fn<PFN_vkFreeMemory>("vkFreeMemory");
    for (auto& area : areas_) {
        if (area.buffer && destroy_buffer) destroy_buffer(device_, area.buffer, nullptr);
        if (area.memory && free_memory) free_memory(device_, area.memory, nullptr);
    }
}

inline bool IndirectArgumentClamp::make_pipeline() {
    if (pipeline_) return true;
    for (const char* name : {"vkCreateBuffer", "vkDestroyBuffer", "vkGetBufferMemoryRequirements",
             "vkAllocateMemory", "vkFreeMemory", "vkBindBufferMemory", "vkMapMemory", "vkUnmapMemory",
             "vkGetBufferDeviceAddress", "vkCreateShaderModule", "vkDestroyShaderModule", "vkCreatePipelineLayout",
             "vkDestroyPipelineLayout", "vkCreateComputePipelines", "vkDestroyPipeline", "vkCmdPipelineBarrier",
             "vkCmdBindPipeline", "vkCmdPushConstants", "vkCmdDispatch", "vkCmdFillBuffer", "vkCmdCopyBuffer"})
        if (!get_(device_, name)) return false;
    VkShaderModule module = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader.codeSize = sizeof(indirect_clamp_spirv);
    shader.pCode = indirect_clamp_spirv;
    const aftermath::ShaderName aftermath_name("(runtime) indirect_clamp");
    bool ok = fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(device_, &shader, nullptr, &module) == VK_SUCCESS;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(detail::IndirectClampPush)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;
    ok = ok && fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(device_, &layout_info, nullptr, &layout) ==
                  VK_SUCCESS;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = layout;
    ok = ok && fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(device_, VK_NULL_HANDLE, 1, &info, nullptr,
                                                                           &pipeline) == VK_SUCCESS;
    if (module) fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device_, module, nullptr);
    if (!ok) {
        if (pipeline) fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device_, pipeline, nullptr);
        if (layout) fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device_, layout, nullptr);
        return false;
    }
    layout_ = layout;
    pipeline_ = pipeline;
    return true;
}

// One host-visible buffer of counter slots: a coherent mapping the pass's counters are copied into.
inline bool IndirectArgumentClamp::make_area() {
    if (areas_.size() >= max_areas) return false;
    const auto create_buffer = fn<PFN_vkCreateBuffer>("vkCreateBuffer");
    const auto requirements_fn = fn<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements");
    const auto bind_fn = fn<PFN_vkBindBufferMemory>("vkBindBufferMemory");
    const auto allocate_fn = fn<PFN_vkAllocateMemory>("vkAllocateMemory");
    const auto map_fn = fn<PFN_vkMapMemory>("vkMapMemory");
    if (!create_buffer || !requirements_fn || !bind_fn || !allocate_fn || !map_fn || !memory_.memoryTypeCount)
        return false;
    HostArea area{};
    area.slots = slots_per_area;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = VkDeviceSize(area.slots) * HostArea::slot_bytes;
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (create_buffer(device_, &info, nullptr, &area.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{};
    requirements_fn(device_, area.buffer, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    bool bound = false;
    for (const std::uint32_t type : memory_type_order(memory_, requirements.memoryTypeBits,
                                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                      0, requirements.size)) {
        allocation.memoryTypeIndex = type;
        if (allocate_fn(device_, &allocation, nullptr, &area.memory) != VK_SUCCESS) {
            area.memory = VK_NULL_HANDLE;
            continue;
        }
        bound = bind_fn(device_, area.buffer, area.memory, 0) == VK_SUCCESS;
        if (bound) break;
    }
    if (!bound || map_fn(device_, area.memory, 0, info.size, 0, reinterpret_cast<void**>(&area.host)) != VK_SUCCESS) {
        const auto destroy_buffer = fn<PFN_vkDestroyBuffer>("vkDestroyBuffer");
        const auto free_memory = fn<PFN_vkFreeMemory>("vkFreeMemory");
        if (area.buffer && destroy_buffer) destroy_buffer(device_, area.buffer, nullptr);
        if (area.memory && free_memory) free_memory(device_, area.memory, nullptr);
        return false;
    }
    std::memset(area.host, 0, size_t(info.size));
    for (std::uint32_t i = 0; i < area.slots; ++i) free_slots_.push_back({std::uint32_t(areas_.size()), i});
    areas_.push_back(area);
    return true;
}

inline bool IndirectArgumentClamp::claim_slot(Slot& out) {
    std::lock_guard lock(mutex_);
    if (free_slots_.empty() && !make_area()) return false;
    out = free_slots_.back();
    free_slots_.pop_back();
    return true;
}

inline void IndirectArgumentClamp::release_one(Slot slot) {
    std::lock_guard lock(mutex_);
    free_slots_.push_back(slot);
}

// Reads the counters a submission's clamp passes wrote, adds them to the process totals, and hands the slots back.
// Called when the submission's timeline ticket has been reached, so the GPU is past the copies.
inline void IndirectArgumentClamp::release(std::vector<Slot>& slots, bool read) {
    Counters delta{};
    const std::size_t calls = slots.size();
    {
        std::lock_guard lock(mutex_);
        for (const Slot& slot : slots) {
            if (read && slot.area < areas_.size() && slot.index < areas_[slot.area].slots) {
                const HostArea& area = areas_[slot.area];
                const std::uint32_t* words =
                    reinterpret_cast<const std::uint32_t*>(area.host + std::size_t(slot.index) * HostArea::slot_bytes);
                delta.records_clamped += words[0];
                delta.counts_clamped += words[1];
                delta.over_budget += words[2];
                delta.zeroed += words[3];
            }
            free_slots_.push_back(slot);
        }
        counters_.records_clamped += delta.records_clamped;
        counters_.counts_clamped += delta.counts_clamped;
        counters_.over_budget += delta.over_budget;
        counters_.zeroed += delta.zeroed;
        const auto frame = submit_phases::totals().present_serial.load(std::memory_order_relaxed);
        if (read && calls && frame >= next_report_frame_) {
            next_report_frame_ = frame + 60;
            std::fprintf(stderr, "[rt] indirect clamp: frame %llu, observed %llu records clamped, %llu zeroed, %llu over list budget\n",
                         static_cast<unsigned long long>(frame), static_cast<unsigned long long>(counters_.records_clamped),
                         static_cast<unsigned long long>(counters_.zeroed), static_cast<unsigned long long>(counters_.over_budget));
        }
    }
    slots.clear();
}

// A call clamped with no counter slot free: logged at the 1st, 2nd, 4th... such call.
inline void IndirectArgumentClamp::note_uncounted() {
    std::uint64_t total = 0;
    {
        std::lock_guard lock(mutex_);
        total = ++counters_.uncounted_calls;
    }
    if (total & (total - 1)) return;
    std::fprintf(stderr, "[rt] indirect clamp: %llu execute-indirect call(s) clamped without counters (no counter slot "
                         "free); the clamp still bounds them\n", static_cast<unsigned long long>(total));
}

// A call the pass leaves to the replay: logged at the 1st, 2nd, 4th... fallback.
inline void IndirectArgumentClamp::fallback(const char* reason) {
    std::uint64_t total = 0;
    {
        std::lock_guard lock(mutex_);
        total = ++counters_.unclamped_calls;
    }
    if (total & (total - 1)) return;
    std::fprintf(stderr, "[rt] indirect clamp: %llu execute-indirect call(s) not clamped (%s); the replay refuses the "
                         "indirect command\n", static_cast<unsigned long long>(total), reason);
}

inline bool IndirectArgumentClamp::prepare(VkBuffer arguments, VkDeviceSize argument_offset, VkBuffer count,
                                           VkDeviceSize count_offset, std::uint32_t kind, std::uint32_t stride,
                                           std::uint32_t max_count, Slot slot, ClampedIndirect& out) {
    out = {};
    const std::uint32_t record_size = indirect_clamp_record_size(kind);
    if (kind > indirect_clamp_mesh || !arguments || !max_count || stride < record_size || stride % 4) {
        fallback("an argument stride this pass does not handle");
        return false;
    }
    const VkDeviceSize records_bytes = VkDeviceSize(max_count) * record_size;
    // The clamped records start the slice (the indirect command reads them there), then the clamped count word and
    // the pass's counters, each on a cache line of its own.
    const VkDeviceSize count_scratch = (records_bytes + 63) / 64 * 64;
    const VkDeviceSize stats_scratch = count_scratch + 64;
    if (stats_scratch + 64 > indirect_clamp_max_scratch_bytes) {
        fallback("an argument batch past the scratch cap");
        return false;
    }
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceAddress address = 0;
    if (!scratch_.get(stats_scratch + 64, buffer, offset, address)) {
        fallback("no scratch");
        return false;
    }
    const auto arguments_address = [&](VkBuffer b) {
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = b;
        return fn<PFN_vkGetBufferDeviceAddress>("vkGetBufferDeviceAddress")(device_, &info);
    };
    if (!make_pipeline()) {
        fallback("no compute pipeline");
        return false;
    }
    const VkDeviceAddress arguments_va = arguments_address(arguments);
    const VkDeviceAddress count_va = count ? arguments_address(count) : 0;
    if (!arguments_va || (count && !count_va) || argument_offset > UINT64_MAX - arguments_va ||
        (count && count_offset > UINT64_MAX - count_va)) {
        fallback("an argument buffer without a device address");
        return false;
    }
    out.kind = kind;
    out.stride_words = stride / 4;
    out.max_records = max_count;
    out.records = max_count;
    out.stride = record_size;
    out.counted = count != VK_NULL_HANDLE;
    out.buffer = buffer;
    out.offset = offset;  // the clamped records, compact
    out.count_buffer = count ? buffer : VK_NULL_HANDLE;
    out.count_offset = count ? offset + count_scratch : 0;
    out.arguments = arguments_va + argument_offset;
    out.count = count_va ? count_va + count_offset : 0;
    out.records_out = address;
    out.count_out = address + count_scratch;
    out.stats = address + stats_scratch;
    out.list_budget = out.stats + 4 * sizeof(std::uint32_t);
    out.stats_scratch_offset = offset + stats_scratch;
    out.stats_buffer = slot.area == no_slot ? VK_NULL_HANDLE : areas_[slot.area].buffer;  // no_slot: no copy-back
    out.stats_offset = slot.area == no_slot ? 0 : VkDeviceSize(slot.index) * HostArea::slot_bytes;
    {
        std::lock_guard lock(mutex_);
        ++counters_.calls;
        counters_.records += max_count;
    }
    return true;
}

inline void IndirectArgumentClamp::emit(VkCommandBuffer cb, const ClampedIndirect& job) {
    const auto barrier = fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    const auto bind = fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline");
    const auto push = fn<PFN_vkCmdPushConstants>("vkCmdPushConstants");
    const auto dispatch = fn<PFN_vkCmdDispatch>("vkCmdDispatch");
    // The counters start at zero: the budget's compare-and-swap and the record counts accumulate into them. A
    // transfer write, ordered to the compute stage by the barrier below, like the arguments' own writes.
    fn<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(cb, job.buffer, job.stats_scratch_offset, job.list_budget == job.stats + 16 ? 20 : 16, 0);
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;  // every earlier write: the compute that wrote these arguments
    before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0,
            nullptr);
    bind(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    const VkDeviceSize dispatch_limit = VkDeviceSize(limits_.maxComputeWorkGroupCount[0]) * 64;
    // Bound per-dispatch indexing to 32 bits, even when the D3D12 stride is very large.
    const std::uint32_t index_limit = (UINT32_MAX - 4u) / std::max<std::uint32_t>(1u, job.stride_words);
    const VkDeviceSize chunk_limit = std::max<VkDeviceSize>(1, std::min(dispatch_limit, VkDeviceSize(index_limit)));
    const std::uint32_t tags = job.kind <= indirect_clamp_draw_indexed ? (job.kind == indirect_clamp_draw ? 4u : 5u)
                                                                       : 3u;
    for (VkDeviceSize first = 0; first < job.max_records;) {
        const std::uint32_t records = std::uint32_t(std::min<VkDeviceSize>(chunk_limit, job.max_records - first));
        // Use local word offsets within each chunk. Only the count comparison needs the global record index.
        const detail::IndirectClampPush constants{job.arguments + first * job.stride_words * 4,
                                                  job.count,
                                                  job.records_out + first * tags * 4,
                                                  job.count_out,
                                                  job.stats,
                                                  job.list_budget,
                                                  job.kind,
                                                  job.stride_words,
                                                  records,
                                                  std::uint32_t(first),
                                                  job.max_records,
                                                  (job.kind == indirect_clamp_dispatch ? indirect_group_budget() << 1 : 0u) |
                                                      (job.counted ? 1u : 0u)};
        push(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
        dispatch(cb, (records - 1u) / 64u + 1u, 1, 1);
        first += records;
    }
    if (job.stats_buffer) {  // none for a call clamped without a counter slot
        VkMemoryBarrier after_copy{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after_copy.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        after_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &after_copy, 0, nullptr,
                0, nullptr);
        VkBufferCopy copy{job.stats_scratch_offset, job.stats_offset, 4 * sizeof(std::uint32_t)};
        fn<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(cb, job.buffer, job.stats_buffer, 1, &copy);
    }
    // The indirect command reads the records, and nothing else may read them while it does.
    VkMemoryBarrier done{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    done.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    done.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
    barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &done, 0, nullptr, 0,
            nullptr);
}
}  // namespace leonida
