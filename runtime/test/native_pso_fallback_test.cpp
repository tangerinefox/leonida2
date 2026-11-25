// The slot 57 factory's blend fallback (shims/d3d12_xs_native_pso_fallback.h): only a pipeline that blends a
// B4G4R4A4 target is retried, with blending off on those targets alone. Any other E_NOTIMPL refusal keeps the blend.
#include "../shims/d3d12_xs_native_pso_fallback.h"

#include <cstdio>
#include <cstring>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

D3D12_GRAPHICS_PIPELINE_STATE_DESC two_targets(DXGI_FORMAT first, DXGI_FORMAT second) {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.NumRenderTargets = 2;
    desc.RTVFormats[0] = first;
    desc.RTVFormats[1] = second;
    for (auto& rt : desc.BlendState.RenderTarget) {
        rt.BlendEnable = TRUE;
        rt.SrcBlend = D3D12_BLEND_SRC_ALPHA;
        rt.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        rt.BlendOp = D3D12_BLEND_OP_ADD;
        rt.SrcBlendAlpha = D3D12_BLEND_ONE;
        rt.DestBlendAlpha = D3D12_BLEND_ZERO;
        rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    }
    desc.BlendState.IndependentBlendEnable = FALSE;
    return desc;
}

}  // namespace

int main() {
    using rt::d3d12_xs::native_pipeline_detail::without_stand_in_blend;

    {
        auto desc = two_targets(DXGI_FORMAT_B4G4R4A4_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM);
        expect(without_stand_in_blend(desc), "a blended B4G4R4A4 target is the stand-in's refusal");
        expect(!desc.BlendState.RenderTarget[0].BlendEnable, "blending goes off on the B4G4R4A4 target");
        expect(desc.BlendState.RenderTarget[1].BlendEnable, "the other target keeps its blend");
        expect(desc.BlendState.IndependentBlendEnable, "the targets now differ, so blending is per target");
    }
    {
        // No B4G4R4A4 target: an E_NOTIMPL here is some other refusal (a rect list, mesh limits, a dual-source output
        // that does not fit Location 0 Index 1), and the pipeline must not be rebuilt without its blend.
        const auto before = two_targets(DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT);
        auto desc = before;
        expect(!without_stand_in_blend(desc), "no B4G4R4A4 target, no retry");
        expect(std::memcmp(&desc, &before, sizeof(desc)) == 0, "the description is left as it was");
    }
    {
        auto before = two_targets(DXGI_FORMAT_B4G4R4A4_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM);
        before.BlendState.RenderTarget[0].BlendEnable = FALSE;
        auto desc = before;
        expect(!without_stand_in_blend(desc), "a B4G4R4A4 target that does not blend, no retry");
        expect(std::memcmp(&desc, &before, sizeof(desc)) == 0, "the description is left as it was");
    }
    {
        auto desc = two_targets(DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B4G4R4A4_UNORM);
        desc.NumRenderTargets = 1;  // the B4G4R4A4 format past the target count is not a target
        expect(!without_stand_in_blend(desc), "a format past NumRenderTargets is not looked at");
    }
    if (failures) return 1;
    std::printf("native_pso_fallback: ok\n");
    return 0;
}
