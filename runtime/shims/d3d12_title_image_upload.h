#pragma once

#include "xg_xs_layout.h"
#include "../src/vulkan_backend.h"

namespace rt::d3d12 {

// One upload buffer and command buffer per device, used under the title-texture registry lock.
// A whole image (including arrays and compressed mip tails) is detiled into the mapped buffer.
// The transfer never re-enters the D3D12 queue's pre-submit callbacks.
class TitleImageUpload {
public:
    VkDeviceSize capacity() const { return capacity_; }

    template<class Device>
    void release(Device& d) {
        auto& vk = d.context->vk;
        release_buffer(d);
        if (pool_) vk.vkDestroyCommandPool(d.info.device, pool_, nullptr);
        if (ready_) vk.vkDestroySemaphore(d.info.device, ready_, nullptr);
        pool_ = VK_NULL_HANDLE;
        command_ = VK_NULL_HANDLE;
        ready_ = VK_NULL_HANDLE;
    }

    template<class Device>
    bool write(Device& d, leonida::ResourceId id, const leonida::ResourceBinding& image,
               const xg_xs::Texture& layout, const std::uint8_t* tiled, std::vector<std::uint64_t>& columns,
               bool inspect, bool& nonzero) {
        const auto block = leonida::copy_block_format(image, VK_IMAGE_ASPECT_COLOR_BIT);
        const auto mips = layout.layout.mips;
        const auto layers = layout.array;
        if (!block.bytes || block.bytes != layout.format.bytes || !mips || mips > image.mip_levels ||
            !layers || layers > image.array_layers) return false;

        regions_.clear();
        VkDeviceSize bytes = 0;
        for (unsigned layer = 0; layer < layers; ++layer) {
            for (unsigned level = 0; level < mips; ++level) {
                const auto& mip = layout.layout.plane[0].mips[level];
                VkBufferImageCopy copy{};
                copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, layer, 1};
                copy.imageExtent = {std::max(1u, image.width >> level), std::max(1u, image.height >> level),
                                    image.volume ? std::max(1u, image.depth >> level) : 1u};
                if (mip.width_elements != (copy.imageExtent.width + block.block - 1) / block.block ||
                    mip.height_elements != (copy.imageExtent.height + block.block - 1) / block.block ||
                    (image.volume && mip.depth != copy.imageExtent.depth)) return false;
                // Vulkan requires offsets aligned to four bytes and to the texel block size.
                const auto align = 4u * block.bytes;
                copy.bufferOffset = bytes + (align - bytes % align) % align;
                const VkDeviceSize row = VkDeviceSize(mip.width_elements) * block.bytes;
                const VkDeviceSize slice = row * mip.height_elements;
                if (slice > UINT32_MAX || (UINT64_MAX - copy.bufferOffset) / copy.imageExtent.depth < slice) return false;
                bytes = copy.bufferOffset + slice * copy.imageExtent.depth;
                regions_.push_back(copy);
            }
        }
        if (!prepare(d, bytes)) return false;
        for (const auto& copy : regions_) {
            const auto level = copy.imageSubresource.mipLevel;
            const auto& mip = layout.layout.plane[0].mips[level];
            const auto row = mip.width_elements * block.bytes, slice = row * mip.height_elements;
            auto* dst = mapped_ + copy.bufferOffset;
            xg_xs::copy_elements<false>(layout, level, copy.imageSubresource.baseArrayLayer, tiled, dst, row, slice, columns);
            if (inspect && !nonzero)
                nonzero = std::any_of(dst, dst + VkDeviceSize(slice) * copy.imageExtent.depth,
                                      [](std::uint8_t c) { return c != 0; });
        }

        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, layers};
        const auto result = d.context->write_staged_title_image(id, range, [&](const leonida::ReplayFunctions& replay,
                                                                               const leonida::LayoutPieces& pieces) {
            auto& vk = d.context->vk;
            VkResult r = vk.vkResetCommandBuffer(command_, 0);
            if (r != VK_SUCCESS) return r;
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if ((r = vk.vkBeginCommandBuffer(command_, &begin)) != VK_SUCCESS) return r;
            // All texels of every mip/layer named in range are overwritten. Prior reads already finished.
            leonida::detail::image_barrier(replay, command_, image, range, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            vk.vkCmdCopyBufferToImage(command_, buffer_, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      static_cast<std::uint32_t>(regions_.size()), regions_.data());
            for (const auto& piece : pieces)
                leonida::detail::image_barrier(replay, command_, image, piece.range, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    piece.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : piece.layout,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            if ((r = vk.vkEndCommandBuffer(command_)) != VK_SUCCESS) return r;
            return submit(d);
        });
        return result == VK_SUCCESS;
    }

private:
    template<class Device>
    void release_buffer(Device& d) {
        auto& vk = d.context->vk;
        if (buffer_) vk.vkDestroyBuffer(d.info.device, buffer_, nullptr);
        if (memory_) vk.vkFreeMemory(d.info.device, memory_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        memory_ = VK_NULL_HANDLE;
        mapped_ = nullptr;
        capacity_ = 0;
    }

    template<class Device>
    bool prepare(Device& d, VkDeviceSize size) {
        auto& vk = d.context->vk;
        const auto dev = d.info.device;
        if (!pool_) {
            VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool.queueFamilyIndex = d.info.queue_family;
            pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            if (vk.vkCreateCommandPool(dev, &pool, nullptr, &pool_) != VK_SUCCESS) return false;
        }
        if (!command_) {
            VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            alloc.commandPool = pool_;
            alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            alloc.commandBufferCount = 1;
            if (vk.vkAllocateCommandBuffers(dev, &alloc, &command_) != VK_SUCCESS) return false;
        }
        if (!ready_) {
            VkSemaphoreTypeCreateInfo timeline{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &timeline};
            if (vk.vkCreateSemaphore(dev, &info, nullptr, &ready_) != VK_SUCCESS) return false;
            for (const auto* qs : {&d.info.command_queues, &d.info.compute_queues, &d.info.copy_queues})
                for (auto q : *qs)
                    if (q && q != d.info.queue && std::find(consumers_.begin(), consumers_.end(), q) == consumers_.end())
                        consumers_.push_back(q);
        }
        if (size <= capacity_) return true;
        release_buffer(d);
        VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer.size = size;
        buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vk.vkCreateBuffer(dev, &buffer, nullptr, &buffer_) != VK_SUCCESS) return false;
        VkMemoryRequirements req{};
        vk.vkGetBufferMemoryRequirements(dev, buffer_, &req);
        memory_ = d.allocate(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
        void* map = nullptr;
        if (!memory_ || vk.vkBindBufferMemory(dev, buffer_, memory_, 0) != VK_SUCCESS ||
            vk.vkMapMemory(dev, memory_, 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS) {
            release_buffer(d);
            return false;
        }
        mapped_ = static_cast<std::uint8_t*>(map);
        capacity_ = size;
        return true;
    }

    template<class Device>
    VkResult submit(Device& d) {
        auto& vk = d.context->vk;
        ++value_;
        VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        timeline.signalSemaphoreValueCount = 1;
        timeline.pSignalSemaphoreValues = &value_;
        VkSubmitInfo transfer{VK_STRUCTURE_TYPE_SUBMIT_INFO, &timeline};
        transfer.commandBufferCount = 1;
        transfer.pCommandBuffers = &command_;
        transfer.signalSemaphoreCount = 1;
        transfer.pSignalSemaphores = &ready_;
        auto r = vk.vkQueueSubmit(d.info.queue, 1, &transfer, VK_NULL_HANDLE);
        if (r != VK_SUCCESS) return r;
        // The registry may reuse these mapped bytes or retire the image as soon as write() returns.
        VkSemaphoreWaitInfo done{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        done.semaphoreCount = 1;
        done.pSemaphores = &ready_;
        done.pValues = &value_;
        if ((r = vk.vkWaitSemaphores(d.info.device, &done, UINT64_MAX)) != VK_SUCCESS) return r;
        // GPU writes become visible to later submissions on every other queue. The primary queue has the image barrier.
        VkTimelineSemaphoreSubmitInfo acquire{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        acquire.waitSemaphoreValueCount = 1;
        acquire.pWaitSemaphoreValues = &value_;
        VkPipelineStageFlags stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo wait{VK_STRUCTURE_TYPE_SUBMIT_INFO, &acquire};
        wait.waitSemaphoreCount = 1;
        wait.pWaitSemaphores = &ready_;
        wait.pWaitDstStageMask = &stages;
        for (auto q : consumers_)
            if ((r = vk.vkQueueSubmit(q, 1, &wait, VK_NULL_HANDLE)) != VK_SUCCESS) return r;
        return VK_SUCCESS;
    }

    VkBuffer buffer_{};
    VkDeviceMemory memory_{};
    std::uint8_t* mapped_ = nullptr;
    VkDeviceSize capacity_ = 0;
    VkCommandPool pool_{};
    VkCommandBuffer command_{};
    VkSemaphore ready_{};
    std::uint64_t value_ = 0;
    std::vector<VkQueue> consumers_;
    std::vector<VkBufferImageCopy> regions_;
};

}  // namespace rt::d3d12
