#pragma once

// Source: adapted from enx/leonida, commit 5c175db901f8486bcdefd03746b578b657e3151f on the original author's git
// mirror (https://git.notrexed.dev/enx/leonida), a fork of this project. Licence: GPL-3.0-or-later as a fork of this
// project. That server could not be reached on 2026-11-25, when this file was merged, so the source and its licence
// could not be read there; the maintainer accepted the file because the source is a fork of this same
// GPL-3.0-or-later project (the README links the server as the original author's mirror) and because
// tools/native_pso_hashes.py is adapted from the same commit. Changed, as the pull request that brought it describes
// it (the source could not be compared): the state and stage resolution moved to runtime/src/native_pso.h, and the
// stage bytecode handed on is read from the user's own carved DXBC containers, not from the description.
//
// Xbox native graphics pipeline state, adapted from enx/leonida's slot-57 implementation.
// The state and stage resolution are in native_pso.h. All stage bytecode handed to D3D12
// comes from the user's carved DXBC containers and is translated by the normal Vulkan PSO path.
#include "../src/native_pso.h"
#include "d3d12_xs_pso_stream.h"
#include "../src/shader_library.h"

#include <d3d12.h>
#include <climits>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>

namespace rt::d3d12_xs {
namespace native_pipeline_detail {
namespace np = leonida::native_pso;

inline void fill_fixed_state(const np::FixedState& s, D3D12_GRAPHICS_PIPELINE_STATE_DESC& pc) {
    for (auto& rt : pc.BlendState.RenderTarget) rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pc.SampleMask = s.sample_mask ? s.sample_mask : UINT_MAX;
    pc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pc.RasterizerState.DepthClipEnable = TRUE;
    pc.DSVFormat = static_cast<DXGI_FORMAT>(s.dsv_format);
    pc.DepthStencilState.DepthEnable = pc.DSVFormat != DXGI_FORMAT_UNKNOWN;
    pc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    pc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    pc.PrimitiveTopologyType = s.topology_type ? static_cast<D3D12_PRIMITIVE_TOPOLOGY_TYPE>(s.topology_type)
                                               : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pc.NumRenderTargets = s.rt_count;
    for (unsigned i = 0; i < s.rt_count; ++i) pc.RTVFormats[i] = static_cast<DXGI_FORMAT>(s.rtv_formats[i]);
    pc.SampleDesc.Count = s.sample_count;
}

template <D3D12_PIPELINE_STATE_SUBOBJECT_TYPE T, typename P>
struct alignas(void*) Subobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = T;
    P value{};
};

struct MeshStream {
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*> root;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE> ms;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE> ps;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, D3D12_BLEND_DESC> blend;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, UINT> sample_mask;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC> raster;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, D3D12_DEPTH_STENCIL_DESC> depth;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY> rtvs;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT> dsv;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, DXGI_SAMPLE_DESC> sample;
};

inline HRESULT create_mesh(ID3D12Device* real, const D3D12_GRAPHICS_PIPELINE_STATE_DESC& pc,
                           D3D12_SHADER_BYTECODE ms, const IID& riid, void** out) {
    ID3D12Device2* device2 = nullptr;
    if (FAILED(real->QueryInterface(__uuidof(ID3D12Device2), reinterpret_cast<void**>(&device2))) || !device2)
        return E_NOTIMPL;
    MeshStream s;
    s.root.value = pc.pRootSignature;
    s.ms.value = ms;
    s.ps.value = pc.PS;
    s.blend.value = pc.BlendState;
    s.sample_mask.value = pc.SampleMask;
    s.raster.value = pc.RasterizerState;
    s.depth.value = pc.DepthStencilState;
    s.rtvs.value.NumRenderTargets = pc.NumRenderTargets;
    for (unsigned i = 0; i < 8; ++i) s.rtvs.value.RTFormats[i] = pc.RTVFormats[i];
    s.dsv.value = pc.DSVFormat;
    s.sample.value = pc.SampleDesc;
    const D3D12_PIPELINE_STATE_STREAM_DESC stream{sizeof(s), &s};
    const HRESULT hr = device2->CreatePipelineState(&stream, riid, out);
    device2->Release();
    return hr;
}

// A pipeline state stream: each subobject is a type word, then its payload at its own alignment, padded to 8 bytes
// (d3dx12's alignas(void*) subobjects).
class Stream {
public:
    template <typename P>
    void add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type, const P& payload) {
        const std::size_t at = size_;
        const std::size_t offset = (at + sizeof(type) + alignof(P) - 1) & ~(alignof(P) - 1);
        size_ = (offset + sizeof(P) + 7) & ~std::size_t(7);
        words_.resize(size_ / 8);
        auto* bytes = reinterpret_cast<unsigned char*>(words_.data());
        std::memcpy(bytes + at, &type, sizeof(type));
        std::memcpy(bytes + offset, &payload, sizeof(P));
    }
    D3D12_PIPELINE_STATE_STREAM_DESC desc() { return {size_, words_.data()}; }

private:
    std::vector<std::uint64_t> words_;
    std::size_t size_ = 0;
};

// The stream of a graphics description; an MS stage makes a mesh pipeline (no input layout or topology).
inline void build_stream(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& pc, D3D12_SHADER_BYTECODE ms, Stream& s) {
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, pc.pRootSignature);
    const bool mesh = ms.pShaderBytecode != nullptr;
    if (mesh) s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms);
    if (pc.VS.pShaderBytecode) s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, pc.VS);
    if (pc.GS.pShaderBytecode) s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, pc.GS);
    if (pc.PS.pShaderBytecode) s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, pc.PS);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, pc.BlendState);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, pc.SampleMask);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, pc.RasterizerState);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, pc.DepthStencilState);
    if (!mesh) {
        s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, pc.InputLayout);
        s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, pc.PrimitiveTopologyType);
    }
    D3D12_RT_FORMAT_ARRAY rtvs{};
    rtvs.NumRenderTargets = pc.NumRenderTargets;
    for (unsigned i = 0; i < 8; ++i) rtvs.RTFormats[i] = pc.RTVFormats[i];
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, rtvs);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, pc.DSVFormat);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, pc.SampleDesc);
}


}  // namespace native_pipeline_detail

// One instance per Vulkan device. The catalog is immutable once loaded; repeated pipeline
// descriptions can share the source containers without sharing mutable COM pipeline objects.
class NativeGraphicsPipelines {
public:
    template<class Create>
    HRESULT create(const std::filesystem::path& folder, const leonida::native_pso::GraphicsDesc& desc,
                   void** out, Create&& create_pipeline) {
        using namespace native_pipeline_detail;
        *out = nullptr;
        try {
            std::call_once(load_, [&] {
                std::ifstream file(folder / L"native_hashes.txt", std::ios::binary);
                if (!file) {
                    std::fprintf(stderr, "[rt] d3d12_xs: slot 57 needs native_hashes.txt from the shaders stage\n");
                    return;
                }
                const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
                std::string error;
                if (!catalog_.parse(text, &error)) std::fprintf(stderr, "[rt] d3d12_xs: slot 57: %s\n", error.c_str());
            });
            const auto plan = np::plan(desc, catalog_);
            if (plan.status == np::PlanStatus::malformed) return E_INVALIDARG;
            if (plan.status != np::PlanStatus::ok) return E_NOTIMPL;

            D3D12_GRAPHICS_PIPELINE_STATE_DESC pc{};
            pc.pRootSignature = static_cast<ID3D12RootSignature*>(const_cast<void*>(desc.root_signature));
            fill_fixed_state(plan.state, pc);
            std::vector<D3D12_INPUT_ELEMENT_DESC> elements;
            for (const auto& e : plan.state.elements) {
                elements.push_back({e.name.c_str(), e.semantic_index, static_cast<DXGI_FORMAT>(e.format), e.input_slot,
                                    e.aligned_byte_offset, static_cast<D3D12_INPUT_CLASSIFICATION>(e.classification),
                                    e.step_rate});
            }
            pc.InputLayout = {elements.data(), static_cast<UINT>(elements.size())};

            std::array<Container, np::kStageBlocks> code;
            D3D12_SHADER_BYTECODE mesh{};
            for (const auto& stage : plan.stages) {
                auto& bytes = code[stage.block];
                bytes = container(folder, *stage.shader);
                if (!bytes) return E_INVALIDARG;
                const D3D12_SHADER_BYTECODE view{bytes->data(), bytes->size()};
                if (stage.shader->stage == "vs") pc.VS = view;
                else if (stage.shader->stage == "ps") pc.PS = view;
                else if (stage.shader->stage == "gs") pc.GS = view;
                else mesh = view;  // plan() admits only VS/PS/GS or MS/PS.
            }
            return create_pipeline(pc, mesh, code);
        } catch (const std::bad_alloc&) {
            return E_OUTOFMEMORY;
        } catch (const std::exception&) {
            return E_INVALIDARG;
        }
    }

private:
    using Container = std::shared_ptr<const std::vector<std::uint8_t>>;
    Container container(const std::filesystem::path& folder, const leonida::native_pso::ShaderRef& ref) {
        {
            std::lock_guard lock(mutex_);
            if (const auto it = containers_.find(ref.digest); it != containers_.end()) return it->second;
        }
        // A catalog names files inside the shader workspace. Checking only its components lets a
        // reader with access to the two shader directories work without opening their ancestors.
        if (ref.path.find_first_of("\\:\r\n\t") != std::string::npos || ref.path.find('\0') != std::string::npos)
            return {};
        const auto path = std::filesystem::u8path(ref.path);
        if (path.has_root_path() || path.extension() != ".dxbc") return {};
        const auto workspace = folder.parent_path();
        const auto resolved = (folder / path).lexically_normal();
        const auto relative = resolved.lexically_relative(workspace);
        if (relative.empty() || *relative.begin() == "..") return {};
        auto component = workspace;
        for (const auto& part : relative) {
            component /= part;
            const DWORD attrs = GetFileAttributesW(component.c_str());
            if (attrs == INVALID_FILE_ATTRIBUTES || (attrs & FILE_ATTRIBUTE_REPARSE_POINT)) return {};
        }
        std::ifstream in(resolved, std::ios::binary | std::ios::ate);
        const std::streamoff size = in ? static_cast<std::streamoff>(in.tellg()) : 0;
        if (size <= 0 || size > (64 << 20)) return {};
        auto result = std::make_shared<std::vector<std::uint8_t>>(static_cast<std::size_t>(size));
        in.seekg(0);
        if (!in.read(reinterpret_cast<char*>(result->data()), size)) return {};
        const auto hash = leonida::dxbc_part(*result, "HASH");
        if (hash.size() < 20 || std::memcmp(hash.data() + 4, ref.digest.data(), ref.digest.size())) return {};
        std::lock_guard lock(mutex_);
        if (const auto it = containers_.find(ref.digest); it != containers_.end()) return it->second;
        constexpr std::size_t budget = std::size_t{256} << 20;
        if (result->size() <= budget - cached_bytes_) {
            containers_.emplace(ref.digest, result);
            cached_bytes_ += result->size();
        }
        return result;
    }

    std::once_flag load_;
    leonida::native_pso::HashMap catalog_;
    std::mutex mutex_;
    std::map<leonida::native_pso::Digest, Container> containers_;
    std::size_t cached_bytes_ = 0;
};

}  // namespace rt::d3d12_xs
