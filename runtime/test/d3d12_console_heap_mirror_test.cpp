// The host-RAM mirror of a console descriptor heap's GPU copy (src/console_heap_copy.h) through a heap's life, with
// LEONIDA_CONSOLE_RELEASE_FILTER=1 (the release sweep reads the mirror to pick the slots to fix): the first refresh,
// slots the title rewrites and zeroes after it, and a released buffer whose V#s the sweep must find and null. After
// each step the GPU copy holds what it should and the mirror agrees with it byte for byte. Run twice by ctest: with
// the mirror, and with LEONIDA_HEAP_MIRROR=0 (no mirror; the sweep reads the copy, as before it), where the copy must
// come out the same. Nothing dispatches through a descriptor. No game data.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

constexpr UINT kDescriptor = 32;
constexpr UINT kViewBytes = 256;
constexpr UINT kBufferBytes = 64 * 1024;

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

leonida::gfx10::BufferWords raw_vsharp(std::uint64_t address) {
    leonida::gfx10::Buffer b{};
    b.address = address;
    b.num_records = kViewBytes;
    b.format = leonida::gfx10::format_from_vulkan(VK_FORMAT_R32_UINT);
    b.oob_select = leonida::gfx10::OobSelect::raw;
    return leonida::gfx10::encode(b);
}

std::uint64_t base_of(const std::uint32_t* w) { return std::uint64_t(w[0]) | (std::uint64_t(w[1] & 0xFFFFu) << 32); }
bool zero_size(const std::uint32_t* w) { return base_of(w) == 0 && w[2] == 0; }

struct Heap {
    static constexpr UINT slots = 4096;
    static constexpr std::uint64_t bytes = std::uint64_t(slots) * kDescriptor;
};

// What a slot of the copy is expected to hold: the buffer it names, or nothing.
enum class Want { a, b, none, zero };

int check(ID3D12Device* device, std::uint64_t shader_va, std::uint64_t title, const Want* want, std::uint64_t a_va,
          std::uint64_t b_va, bool mirrored, const char* step) {
    static std::uint32_t copy[Heap::slots][kDescriptor / 4];
    EXPECT(rt::d3d12::read_arena(device, shader_va, copy, sizeof(copy)));
    for (UINT s = 0; s < Heap::slots; ++s) {
        const std::uint32_t* w = copy[s];
        bool ok = true;
        switch (want[s]) {
        case Want::a: ok = base_of(w) >= a_va && base_of(w) < a_va + kBufferBytes && w[2] == kViewBytes; break;
        case Want::b: ok = base_of(w) >= b_va && base_of(w) < b_va + kBufferBytes && w[2] == kViewBytes; break;
        case Want::none: ok = zero_size(w); break;
        case Want::zero: for (int i = 0; i < 8; ++i) ok = ok && !w[i]; break;
        }
        if (!ok) {
            std::fprintf(stderr, "%s: slot %u holds %08x %08x %08x %08x, wanted %d\n", step, s, w[0], w[1], w[2], w[3],
                         int(want[s]));
            return 1;
        }
    }
    const long long differ = rt::d3d12::console_heap_mirror_mismatches(device, title);
    if (mirrored) EXPECT(differ == 0);
    else EXPECT(differ == -1);
    std::printf("%s: copy as expected, %s\n", step, mirrored ? "mirror agrees with it" : "no mirror");
    return 0;
}

int run(ID3D12Device* device, bool mirrored) {
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    UINT64 fence_value = 0;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
    EXPECT(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    const auto submit = [&]() -> int {  // every submission refreshes the console heaps first
        queue->ExecuteCommandLists(1, lists);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, ++fence_value)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr)));
        return 0;
    };

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const auto desc = buffer_desc(kBufferBytes);
    Ref<ID3D12Resource> a, b;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                     D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&a.p))));
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                     D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&b.p))));
    const std::uint64_t a_va = a->GetGPUVirtualAddress(), b_va = b->GetGPUVirtualAddress();
    EXPECT(a_va && b_va);

    auto* title = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, Heap::bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    EXPECT(title);
    static Want want[Heap::slots];
    const auto put = [&](UINT slot, std::uint64_t va, Want w) {
        const auto words = raw_vsharp(va);
        std::memcpy(title + slot * kDescriptor, words.data(), sizeof(words));
        want[slot] = w;
    };
    for (UINT s = 0; s < Heap::slots; ++s) want[s] = Want::zero;
    // Before the heap exists: slots 0-1023 name A, 1024-2047 B, the rest empty but for one data slot.
    for (UINT s = 0; s < 1024; ++s) put(s, a_va + (s % 256) * kViewBytes, Want::a);
    for (UINT s = 1024; s < 2048; ++s) put(s, b_va + (s % 256) * kViewBytes, Want::b);
    const std::uint32_t data[8] = {0x12345000u, 0x00007777u, 0x00000400u, 0x40000000u, 1, 2, 3, 4};
    std::memcpy(title + 3000 * kDescriptor, data, sizeof(data));
    want[3000] = Want::none;  // its descriptor half null in the copy (fix_console_slot)

    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, Heap::slots,
                                               D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    Ref<ID3D12DescriptorHeap> heap;
    const auto title_va = reinterpret_cast<std::uint64_t>(title);
    EXPECT(SUCCEEDED(rt::d3d12::create_console_descriptor_heap(device, &heap_desc, title_va, Heap::bytes,
                                                               IID_PPV_ARGS(&heap.p))));
    D3D12_DESCRIPTOR_HEAP_DESC got{};
    D3D12_GPU_VIRTUAL_ADDRESS shader_va = 0;
    EXPECT(rt::d3d12::descriptor_heap_info(heap.p, got, shader_va));
    if (check(device, shader_va, title_va, want, a_va, b_va, mirrored, "first refresh")) return 1;

    // The title rewrites slots after the heap is made (the write watch finds them): a slot of B now names A, new V#s
    // past what it filled, and a page of A's slots zeroed (128 slots).
    put(1500, a_va + 0x400, Want::a);
    for (UINT s = 2048; s < 2112; ++s) put(s, b_va + (s % 256) * kViewBytes, Want::b);
    std::memset(title + 128 * kDescriptor, 0, 4096);
    for (UINT s = 128; s < 256; ++s) want[s] = Want::zero;
    if (submit()) return 1;
    if (check(device, shader_va, title_va, want, a_va, b_va, mirrored, "title writes")) return 1;

    // A goes: the release sweep (filtered: only the V#s naming A) nulls the slots that name it, and only those. Its
    // memory is given back once its GPU work retires, so a few submissions may pass first.
    a.reset();
    for (UINT s = 0; s < Heap::slots; ++s)
        if (want[s] == Want::a) want[s] = Want::none;
    bool nulled = false;
    for (int pass = 0; pass < 32 && !nulled; ++pass) {
        if (submit()) return 1;
        std::uint32_t w[8];
        EXPECT(rt::d3d12::read_arena(device, shader_va + 1500 * kDescriptor, w, sizeof(w)));
        nulled = zero_size(w);
    }
    EXPECT(nulled);
    if (check(device, shader_va, title_va, want, a_va, b_va, mirrored, "A released")) return 1;

    heap.reset();
    VirtualFree(title, 0, MEM_RELEASE);
    return 0;
}

}  // namespace

int main() {
    char value[4] = {};
    const bool mirrored = !(GetEnvironmentVariableA("LEONIDA_HEAP_MIRROR", value, sizeof(value)) == 1 && value[0] == '0');
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(status == vktest::DeviceStatus::ready);
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.subgroup = d.subgroup;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        EXPECT(rt::d3d12::device_arena_size(device.p) > 0);
        if (run(device.p, mirrored)) return 1;
    }
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::puts("ok");
    return 0;
}
