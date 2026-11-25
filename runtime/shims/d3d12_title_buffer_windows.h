#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <vulkan/vulkan.h>

namespace rt::d3d12 {

// Storage for shader-addressed mirrors when the main arena has no usable offset.
// A window lasts for the lifetime of the device, so these blocks only grow.
class TitleBufferWindows {
    struct Block {
        VkBuffer buffer{};
        VkDeviceMemory memory{};
        VkDeviceAddress address{};
        VkDeviceSize capacity{}, used{};
        std::uint8_t* mapped{};
    };
    std::vector<Block> blocks_;
    VkDeviceSize reserved_{};
    // A new block's allocation failed at this time: none is tried again for kRetryAfter. After the story load the
    // driver refused every 4 MiB block (VK_ERROR_OUT_OF_HOST_MEMORY), and each V# that wanted a copy tried again,
    // through DeviceState::allocate's sleeping retries under the device's memory lock: one submission spent 28-35 s
    // there (speed-8, speed-9). A view refused meanwhile goes without a copy, as one past the cap does.
    static constexpr auto kRetryAfter = std::chrono::seconds(2);
    std::chrono::steady_clock::time_point failed_at_{};
    bool failed_ = false;
    bool local_refused_ = false;  // device-local blocks refused once: host memory from then on
    std::uint64_t quick_refusals_ = 0;

public:
    VkDeviceSize reserved() const { return reserved_; }

    template<class Device>
    bool allocate(Device& d, VkDeviceSize bytes, VkDeviceSize& offset, std::uint8_t*& mapped,
                  bool device_local = false, VkDeviceSize cap = UINT64_MAX, bool* at_cap = nullptr) {
        const VkDeviceSize alignment = device_local ? 16 : 65536;
        constexpr VkDeviceSize block_size = 4ull << 20;
        auto take = [&](Block& b) {
            const auto start = (b.used + alignment - 1) & ~(alignment - 1);
            if (start > b.capacity || bytes > b.capacity - start) return false;
            mapped = b.mapped + start;
            offset = b.address - d.arena.address + start;
            b.used = start + bytes;
            return true;
        };
        if (!blocks_.empty() && take(blocks_.back())) return true;
        // A large view may start a new slab while smaller views still fit earlier tails.
        // Use those tails even after the budget can no longer accommodate another slab.
        if (device_local && blocks_.size() > 1)
            for (std::size_t i = blocks_.size() - 1; i != 0; --i)
                if (take(blocks_[i - 1])) return true;
        auto& vk = d.context->vk;
        if (!vk.vkGetBufferDeviceAddress) return false;
        if (failed_ && std::chrono::steady_clock::now() - failed_at_ < kRetryAfter) {
            ++quick_refusals_;
            return false;
        }
        const auto failed = [&] {
            failed_ = true;
            failed_at_ = std::chrono::steady_clock::now();
            return false;
        };
        Block b{};
        b.capacity = std::max(bytes, block_size);
        if (b.capacity > cap - reserved_) {
            if (at_cap) *at_cap = true;
            return false;
        }
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = b.capacity;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        d.share(info);
        if (vk.vkCreateBuffer(d.info.device, &info, nullptr, &b.buffer) != VK_SUCCESS) return failed();
        VkMemoryRequirements req{};
        vk.vkGetBufferMemoryRequirements(d.info.device, b.buffer, &req);
        if (req.size > cap - reserved_) {
            if (at_cap) *at_cap = true;
            vk.vkDestroyBuffer(d.info.device, b.buffer, nullptr);
            return false;
        }
        // Device-local (Resizable BAR) when asked and while the driver gives it; host memory otherwise. After the
        // story load the driver refused every 4 MiB device-local, host-visible block with a device address
        // (VK_ERROR_OUT_OF_HOST_MEMORY, speed-8/-9) while host memory still served: once that happens, later blocks
        // go straight to host memory instead of through DeviceState::allocate's sleeping retries first.
        VkMemoryPropertyFlags chosen = 0;
        const bool want_local = device_local && !local_refused_;
        b.memory = d.allocate(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              want_local ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : 0, &chosen, true);
        if (b.memory && want_local && !(chosen & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            local_refused_ = true;
            std::fprintf(stderr, "[rt] d3d12: title buffer windows: device-local memory refused a %llu KiB block; "
                                 "this and later blocks are in host memory\n",
                         static_cast<unsigned long long>(req.size >> 10));
        }
        void* host{};
        if (b.memory && vk.vkBindBufferMemory(d.info.device, b.buffer, b.memory, 0) == VK_SUCCESS &&
            vk.vkMapMemory(d.info.device, b.memory, 0, VK_WHOLE_SIZE, 0, &host) == VK_SUCCESS) {
            VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            address.buffer = b.buffer;
            b.address = vk.vkGetBufferDeviceAddress(d.info.device, &address);
            const auto biased = b.address - d.arena.address;
            // A positive 47-bit offset works with both generations of the translated V# decoder.
            if (b.address && biased < (1ull << 47) && b.capacity <= (1ull << 47) - biased) {
                b.mapped = static_cast<std::uint8_t*>(host);
                blocks_.push_back(b);
                reserved_ += req.size;
                if (blocks_.size() == 1 || !(blocks_.size() & (blocks_.size() - 1)))
                    std::fprintf(stderr, "[rt] d3d12: title buffer shadows outside the arena: %zu blocks, "
                                         "new block %llu KiB at shader VA 0x%llx\n", blocks_.size(),
                                 static_cast<unsigned long long>(b.capacity >> 10),
                                 static_cast<unsigned long long>(biased));
                return take(blocks_.back());
            }
            vk.vkUnmapMemory(d.info.device, b.memory);
        }
        vk.vkDestroyBuffer(d.info.device, b.buffer, nullptr);
        if (b.memory) vk.vkFreeMemory(d.info.device, b.memory, nullptr);
        return failed();
    }
    // Allocations refused at once because a new block failed less than kRetryAfter ago.
    std::uint64_t quick_refusals() const { return quick_refusals_; }

    bool lookup(VkDeviceAddress address, std::uint64_t& host, std::uint64_t& remaining) const {
        for (const auto& b : blocks_) {
            const auto delta = address - b.address;
            if (delta < b.used) {
                host = reinterpret_cast<std::uintptr_t>(b.mapped) + delta;
                remaining = b.used - delta;
                return true;
            }
        }
        return false;
    }

    // Device idle. No tracker writes into the blocks after this point.
    template<class Device>
    void release(Device& d) {
        for (auto& b : blocks_) {
            d.context->vk.vkUnmapMemory(d.info.device, b.memory);
            d.context->vk.vkDestroyBuffer(d.info.device, b.buffer, nullptr);
            d.context->vk.vkFreeMemory(d.info.device, b.memory, nullptr);
        }
        blocks_.clear();
        reserved_ = 0;
    }
};
}  // namespace rt::d3d12
