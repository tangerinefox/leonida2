// Directly indexed descriptor heaps for graphics draws, through the Xbox-ABI proxy over the Vulkan device.
//
// A root signature with CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED and/or SAMPLER_HEAP_DIRECTLY_INDEXED gives the console's
// vertex and pixel shaders the bound heaps' 32-bit shader VAs in the user-data dwords right after the explicit root
// arguments (from s8, one dword per table): resource heap first, then sampler heap. The vertex and pixel shaders here
// (test/glsl/vs_user_data_window.vert, ps_user_data_window.frag) both read s10..s13 and write what they saw to two
// UINT render targets, one pixel per draw:
//   0. three constants (s8..s10), both flags:        s10 = constant, s11 = resource heap, s12 = sampler heap;
//   1. the same signature without the flags:         s11..s13 stay 0;
//   2. a table (s8) and three constants (s9..s11):    s12 = resource heap, s13 = sampler heap;
//   3. three constants, the sampler flag alone:      s11 = sampler heap;
//   4. as 0 after SetDescriptorHeaps with the resource heap alone: the sampler heap's dword is 0 for that draw only
//      (each draw snapshots the heaps, as a dispatch does).
// The recorded Draw commands are checked too. With --off (ctest d3d12_graphics_direct_heaps_off sets
// LEONIDA_GRAPHICS_DIRECT_HEAPS=0) no draw gets a heap dword. No game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <variant>
#include <vector>

#if !defined(LEONIDA_TEST_VS_SPV) || !defined(LEONIDA_TEST_PS_SPV)
#error "LEONIDA_TEST_VS_SPV and LEONIDA_TEST_PS_SPV must name the compiled test shaders"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: CPU/GPU handle in RAX

int main(int argc, char** argv) {
    const bool off = argc == 2 && std::strcmp(argv[1], "--off") == 0;
    if (off) {
        const char* v = std::getenv("LEONIDA_GRAPHICS_DIRECT_HEAPS");
        EXPECT(v && std::strcmp(v, "0") == 0);
    }
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);

    const auto vs_program = fssm_program({0x56530011, 0xbf810000});
    const auto ps_program = fssm_program({0x50530011, 0xbf810000});
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);
    const testfiles::ScratchDir scratch("leonida_graphics_direct_heaps_test");
    const auto& library = scratch.path();
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_SPV), library / "vs_user_data_window.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_SPV), library / "ps_user_data_window.spv"));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_user_data_window\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"ps_user_data_window\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0, 1]}}";
    }

    constexpr UINT draws = 5;
    constexpr UINT pixel_bytes = 16;  // R32G32B32A32_UINT
    std::array<std::array<std::uint32_t, 4>, draws> vs_seen{}, ps_seen{}, want{};
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
        info.shader_dir = library.string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p), /*console_lists=*/false)));
        rt::d3d12::set_device_outer(inner, device.p);

        // The shader-visible heaps whose VAs the draws receive.
        const D3D12_DESCRIPTOR_HEAP_DESC resource_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC sampler_desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 8,
                                                      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> resources, samplers;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&resource_desc, IID_PPV_ARGS(&resources.p))));
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&sampler_desc, IID_PPV_ARGS(&samplers.p))));
        const std::uint64_t res = reinterpret_cast<XboxHeapHandleFn>(slot(resources.p, 10))(resources.p);
        const std::uint64_t smp = reinterpret_cast<XboxHeapHandleFn>(slot(samplers.p, 10))(samplers.p);
        EXPECT(res >= 0x10000 && res < (1ull << 32) && smp >= 0x10000 && smp < (1ull << 32) && res != smp);

        // Root signatures: three constants (s8..s10), or a table (s8) and three constants (s9..s11).
        D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER three[1] = {};
        three[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        three[0].Constants = {0, 0, 3};
        D3D12_ROOT_PARAMETER table_first[2] = {};
        table_first[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        table_first[0].DescriptorTable = {1, &range};
        table_first[1] = three[0];
        constexpr auto both = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED |
                              D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
        const auto make_root = [&](const D3D12_ROOT_PARAMETER* params, UINT count, D3D12_ROOT_SIGNATURE_FLAGS flags,
                                   Ref<ID3D12RootSignature>& out) {
            D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
            desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
            desc.Desc_1_0 = {count, params, 0, nullptr, flags};
            const auto blob = serialize_root_signature(desc);
            return !blob.empty() &&
                   SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&out.p)));
        };
        Ref<ID3D12RootSignature> three_both, three_none, table_both, three_sampler;
        EXPECT(make_root(three, 1, both, three_both));
        EXPECT(make_root(three, 1, D3D12_ROOT_SIGNATURE_FLAG_NONE, three_none));
        EXPECT(make_root(table_first, 2, both, table_both));
        EXPECT(make_root(three, 1, D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED, three_sampler));

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.VS = {vs_container.data(), vs_container.size()};
        pso_desc.PS = {ps_container.data(), ps_container.size()};
        pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.BlendState.RenderTarget[1].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.SampleMask = 0xffffffff;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = 2;
        pso_desc.RTVFormats[0] = pso_desc.RTVFormats[1] = DXGI_FORMAT_R32G32B32A32_UINT;
        pso_desc.SampleDesc.Count = 1;
        const auto make_pso = [&](ID3D12RootSignature* root, Ref<ID3D12PipelineState>& out) {
            auto desc = pso_desc;
            desc.pRootSignature = root;
            return SUCCEEDED(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&out.p)));
        };
        Ref<ID3D12PipelineState> pso_three_both, pso_three_none, pso_table_both, pso_three_sampler;
        EXPECT(make_pso(three_both.p, pso_three_both));
        EXPECT(make_pso(three_none.p, pso_three_none));
        EXPECT(make_pso(table_both.p, pso_table_both));
        EXPECT(make_pso(three_sampler.p, pso_three_sampler));

        // Two 5x1 UINT targets (what the VS saw, what the PS saw) and their RTVs.
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC rt_desc{};
        rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rt_desc.Width = draws;
        rt_desc.Height = 1;
        rt_desc.DepthOrArraySize = 1;
        rt_desc.MipLevels = 1;
        rt_desc.Format = DXGI_FORMAT_R32G32B32A32_UINT;
        rt_desc.SampleDesc.Count = 1;
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        std::array<Ref<ID3D12Resource>, 2> targets;
        std::array<Ref<ID3D12DescriptorHeap>, 2> rtv_heaps;
        std::array<D3D12_CPU_DESCRIPTOR_HANDLE, 2> rtvs{};
        const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                                                  0};
        for (std::size_t t = 0; t < targets.size(); ++t) {
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                             D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                             IID_PPV_ARGS(&targets[t].p))));
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heaps[t].p))));
            rtvs[t] = {SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heaps[t].p, 9))(rtv_heaps[t].p))};
            device->CreateRenderTargetView(targets[t].p, nullptr, rtvs[t]);
        }
        Ref<ID3D12Resource> readback;
        const auto readback_desc = buffer_desc(1024);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p,
                                                   pso_three_both.p, IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        const FLOAT seven[4] = {7, 7, 7, 7};  // a pixel no draw reached reads 7s
        for (const auto& rtv : rtvs) list->ClearRenderTargetView(rtv, seven, 0, nullptr);
        list->OMSetRenderTargets(2, rtvs.data(), FALSE, nullptr);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_RECT scissor{0, 0, LONG(draws), 1};
        list->RSSetScissorRects(1, &scissor);
        ID3D12DescriptorHeap* both_heaps[] = {resources.p, samplers.p};
        list->SetDescriptorHeaps(2, both_heaps);
        const std::uint32_t r = off ? 0 : std::uint32_t(res), s = off ? 0 : std::uint32_t(smp);
        const UINT a[3] = {0xA0, 0xA1, 0xA2}, b[3] = {0xB0, 0xB1, 0xB2};
        const auto draw_at = [&](UINT pixel, ID3D12PipelineState* pso, ID3D12RootSignature* root, bool table) {
            const D3D12_VIEWPORT viewport{float(pixel), 0, 1, 1, 0, 1};
            list->RSSetViewports(1, &viewport);
            list->SetPipelineState(pso);
            list->SetGraphicsRootSignature(root);
            if (table) {
                list->SetGraphicsRootDescriptorTable(0, {res});
                list->SetGraphicsRoot32BitConstants(1, 3, b, 0);
            } else {
                list->SetGraphicsRoot32BitConstants(0, 3, a, 0);
            }
            list->DrawInstanced(3, 1, 0, 0);
        };
        draw_at(0, pso_three_both.p, three_both.p, false);
        want[0] = {0xA2, r, s, 0};
        draw_at(1, pso_three_none.p, three_none.p, false);
        want[1] = {0xA2, 0, 0, 0};
        draw_at(2, pso_table_both.p, table_both.p, true);
        want[2] = {0xB1, 0xB2, r, s};
        draw_at(3, pso_three_sampler.p, three_sampler.p, false);
        want[3] = {0xA2, s, 0, 0};
        list->SetDescriptorHeaps(1, &resources.p);  // the sampler heap is unbound for the next draw
        draw_at(4, pso_three_both.p, three_both.p, false);
        want[4] = {0xA2, r, 0, 0};

        for (std::size_t t = 0; t < targets.size(); ++t) {
            const auto to_copy = transition(targets[t].p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                            D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &to_copy);
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = readback.p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint = {512 * t, {DXGI_FORMAT_R32G32B32A32_UINT, draws, 1, 1, 256}};
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = targets[t].p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        EXPECT(list->Close() == S_OK);

        // The recorded draws carry the same words, snapshotted per draw.
        std::vector<leonida::Command> commands;
        EXPECT(rt::d3d12::recorded_commands(list.p, commands));
        UINT recorded = 0;
        for (const auto& command : commands) {
            const auto* draw = std::get_if<leonida::Draw>(&command);
            if (!draw) continue;
            EXPECT(recorded < draws);
            const auto& w = want[recorded++];
            for (UINT k = 0; k < 4; ++k) EXPECT(draw->user_data[10 + k] == w[k]);
            for (UINT k = 14; k < draw->user_data.size(); ++k) EXPECT(draw->user_data[k] == 0);
        }
        EXPECT(recorded == draws);

        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        const auto* bytes = static_cast<const std::uint8_t*>(mapped);
        for (UINT pixel = 0; pixel < draws; ++pixel) {
            std::memcpy(vs_seen[pixel].data(), bytes + pixel * pixel_bytes, pixel_bytes);
            std::memcpy(ps_seen[pixel].data(), bytes + 512 + pixel * pixel_bytes, pixel_bytes);
        }
        readback->Unmap(0, nullptr);

        for (auto* p : {&pso_three_both, &pso_three_none, &pso_table_both, &pso_three_sampler}) p->reset();
        for (auto* p : {&three_both, &three_none, &table_both, &three_sampler}) p->reset();
        for (auto& t : targets) t.reset();
        for (auto& h : rtv_heaps) h.reset();
        readback.reset();
        resources.reset();
        samplers.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    unsigned mismatches = 0;
    for (UINT pixel = 0; pixel < draws; ++pixel) {
        for (const auto* seen : {&vs_seen, &ps_seen}) {
            const auto& got = (*seen)[pixel];
            if (got == want[pixel]) continue;
            ++mismatches;
            std::fprintf(stderr, "draw %u %s s10..s13: got %08x %08x %08x %08x want %08x %08x %08x %08x\n", pixel,
                         seen == &vs_seen ? "VS" : "PS", got[0], got[1], got[2], got[3], want[pixel][0],
                         want[pixel][1], want[pixel][2], want[pixel][3]);
        }
    }
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("graphics direct heaps%s: %u draws, the VS and PS read s10..s13 as recorded (heap dwords after the "
                "explicit arguments at s11/s12 or s12/s13, sampler alone at s11, none without the flags, per-draw "
                "snapshot)\n",
                off ? " (LEONIDA_GRAPHICS_DIRECT_HEAPS=0: none supplied)" : "", draws);
    return 0;
}
