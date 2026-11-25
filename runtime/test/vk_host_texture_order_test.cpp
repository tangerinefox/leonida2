// Title texture uploads must keep a queue submission from racing the host copy.
#include "../src/vulkan_backend.h"

#include <cstdio>
#include <thread>

namespace {
unsigned waits = 0;
VkResult wait_result = VK_SUCCESS;
VkResult VKAPI_CALL idle(VkDevice) { ++waits; return wait_result; }
PFN_vkVoidFunction VKAPI_CALL no_proc(VkDevice, const char*) { return nullptr; }
}

int main() {
    leonida::VulkanContext ctx(no_proc, VK_NULL_HANDLE, false);
    ctx.vk.vkDeviceWaitIdle = idle;
    leonida::ResourceBinding binding{};
    binding.image = reinterpret_cast<VkImage>(1);
    binding.mip_levels = 1;
    ctx.add_resource(1, binding, VK_IMAGE_LAYOUT_GENERAL);
    ctx.add_resource(2, binding, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    unsigned copies = 0;
    const auto copy = [&] {
        ++copies;
        bool raced = false;
        std::thread contender([&] {
            if (ctx.submit_mutex().try_lock()) {
                raced = true;
                ctx.submit_mutex().unlock();
            }
        });
        contender.join();
        return !raced;
    };
    ctx.note_image_work_locked();
    if (ctx.write_host_texture(2, 0, copy) != VK_NOT_READY || copies || waits) return 1;
    if (ctx.write_host_texture(1, 0, copy) != VK_SUCCESS || copies != 1 || waits != 1) return 1;
    if (ctx.write_host_texture(1, 0, copy) != VK_SUCCESS || copies != 2 || waits != 1) return 1;
    ctx.note_image_work_locked();
    wait_result = VK_ERROR_DEVICE_LOST;
    if (ctx.write_host_texture(1, 0, copy) != VK_ERROR_DEVICE_LOST || copies != 2 || waits != 2) return 1;
    const auto stats = ctx.host_texture_stats();
    if (stats.copies != 2 || stats.waits != 2) return 1;
    std::puts("ordered texture upload: layout, submit exclusion, wait reuse, wait failure passed");
}
