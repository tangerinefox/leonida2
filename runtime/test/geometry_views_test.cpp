// rt::d3d12::GeometryViews (runtime/shims/d3d12_geometry_views.h): device-local copies of the title's geometry, kept
// up to date from 64 KiB tiles of the title's pages that are watched for writes. A fake Vulkan device gives the copies
// host memory. No game data, no GPU.
//   1. a new view is copied whole;
//   2. a write into one tile of a 1 MiB view (16 tiles) copies that tile's 64 KiB again, not the view;
//   3. a write into a tile two views share is read once and copied into both;
//   4. two written tiles of one view are two tile copies (LEONIDA_GEOMETRY_WHOLE_VIEWS=1, whole views, fails 2-4).
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../shims/d3d12_geometry_views.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

// The Vulkan calls TitleBufferWindows::allocate makes, over host memory.
struct FakeBuffer {
    VkDeviceSize size = 0;
    void* memory = nullptr;
};
VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice, const VkBufferCreateInfo* info, const VkAllocationCallbacks*,
                                             VkBuffer* out) {
    auto* b = new FakeBuffer{info->size};
    *out = reinterpret_cast<VkBuffer>(b);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL requirements(VkDevice, VkBuffer buffer, VkMemoryRequirements* out) {
    *out = {};
    out->size = reinterpret_cast<FakeBuffer*>(buffer)->size;
    out->alignment = 65536;
    out->memoryTypeBits = 1;
}
VKAPI_ATTR VkResult VKAPI_CALL bind(VkDevice, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize) {
    reinterpret_cast<FakeBuffer*>(buffer)->memory = memory;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL map(VkDevice, VkDeviceMemory memory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags,
                                   void** out) {
    *out = memory;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL unmap(VkDevice, VkDeviceMemory) {}
VKAPI_ATTR VkDeviceAddress VKAPI_CALL address(VkDevice, const VkBufferDeviceAddressInfo* info) {
    return VkDeviceAddress(reinterpret_cast<std::uintptr_t>(reinterpret_cast<FakeBuffer*>(info->buffer)->memory));
}
VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
    delete reinterpret_cast<FakeBuffer*>(buffer);
}
VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*) {
    VirtualFree(memory, 0, MEM_RELEASE);
}

struct FakeDevice {
    struct {
        PFN_vkCreateBuffer vkCreateBuffer = create_buffer;
        PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = requirements;
        PFN_vkBindBufferMemory vkBindBufferMemory = bind;
        PFN_vkMapMemory vkMapMemory = map;
        PFN_vkUnmapMemory vkUnmapMemory = unmap;
        PFN_vkGetBufferDeviceAddress vkGetBufferDeviceAddress = address;
        PFN_vkDestroyBuffer vkDestroyBuffer = destroy_buffer;
        PFN_vkFreeMemory vkFreeMemory = free_memory;
    } vk_;
    struct {
        decltype(vk_)& vk;
    } context_{vk_};
    decltype(context_)* context = &context_;
    struct {
        VkDevice device = VK_NULL_HANDLE;
    } info;
    struct {
        VkDeviceAddress address = 0;
    } arena;
    void share(VkBufferCreateInfo&) {}
    VkDeviceMemory allocate(const VkMemoryRequirements& r, VkMemoryPropertyFlags required,
                            VkMemoryPropertyFlags preferred, VkMemoryPropertyFlags* chosen, bool) {
        if (chosen) *chosen = required | preferred;
        return reinterpret_cast<VkDeviceMemory>(VirtualAlloc(nullptr, r.size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    }
};

// What the views copy, through the same readability check the device makes.
std::vector<std::size_t> g_copies;
rt::d3d12::GeometryCopy copy(std::uint8_t* to, const std::uint8_t* from, std::size_t size) {
    g_copies.push_back(size);
    std::memcpy(to, from, size);
    return rt::d3d12::GeometryCopy::copied;
}

// The title writes into a watched tile: reopened (as the write-fault handler does), then written.
void title_write(std::uintptr_t at, std::uint8_t value) {
    namespace watch = leonida::geometry_writes;
    watch::notify(at, at + 1, watch::Event::fault);
    *reinterpret_cast<std::uint8_t*>(at) = value;
}

}  // namespace

int main() {
    _putenv_s("LEONIDA_GEOMETRY_MB", "64");
    constexpr std::size_t kTile = 65536, kView = 16 * kTile;
    auto* title = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 2 * kView, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!title) return 1;
    for (std::size_t i = 0; i < 2 * kView; ++i) title[i] = std::uint8_t(i * 7);
    const auto base = reinterpret_cast<std::uintptr_t>(title);

    FakeDevice device;
    rt::d3d12::GeometryViews views;
    g_copies.clear();
    const std::uint64_t va = views.resolve(device, base, kView, copy);
    check(va != 0 && g_copies.size() == 1 && g_copies[0] == kView, "a new 1 MiB view is copied whole");
    std::uint64_t host = 0, bytes = 0;
    bool populated = false;
    check(views.lookup(device.arena.address + va, host, bytes, nullptr, &populated) && populated && bytes == kView,
          "the copy is found by its device address");
    auto* copy_bytes = reinterpret_cast<std::uint8_t*>(host);
    check(!std::memcmp(copy_bytes, title, kView), "the copy holds the title's bytes");

    // 2. One write in tile 5.
    g_copies.clear();
    views.refresh(copy);
    check(g_copies.empty(), "a refresh with nothing written copies nothing");
    title_write(base + 5 * kTile + 123, 0xab);
    views.refresh(copy);
    check(g_copies.size() == 1 && g_copies[0] == kTile, "one write copies its 64 KiB tile again, not the 1 MiB view");
    check(copy_bytes[5 * kTile + 123] == 0xab && !std::memcmp(copy_bytes, title, kView),
          "the copy has the new byte and the rest unchanged");

    // 3. A second view over tiles 8..23 shares tiles 8..15 with the first: a write in tile 10 reaches both.
    const std::uint64_t va2 = views.resolve(device, base + 8 * kTile, kView, copy);
    check(va2 != 0, "a second view over half of the first one's tiles");
    std::uint64_t host2 = 0;
    views.lookup(device.arena.address + va2, host2, bytes);
    g_copies.clear();
    title_write(base + 10 * kTile + 9, 0xcd);
    views.refresh(copy);
    check(g_copies.size() == 1 && g_copies[0] == kTile,
          "a write in a shared tile reads that tile from the title's pages once for both views");
    check(copy_bytes[10 * kTile + 9] == 0xcd && reinterpret_cast<std::uint8_t*>(host2)[2 * kTile + 9] == 0xcd,
          "both copies see it");

    // 4. Writes in two tiles of one view: two tile copies.
    g_copies.clear();
    title_write(base + 1 * kTile, 1);
    title_write(base + 3 * kTile + kTile - 1, 2);
    views.refresh(copy);
    check(g_copies.size() == 2 && g_copies[0] == kTile && g_copies[1] == kTile, "two written tiles, two tile copies");
    check(!std::memcmp(copy_bytes, title, kView), "the copy matches the title's pages");

    views.release(device);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
