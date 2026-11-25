# Offline first-run UI on Windows

This integration combines the retail frame-loop runtime with the shader and rendering changes needed to display the first-run controller prompts. The visual milestone is a solid background with legible prompts at the lower right. Gameplay and the 3D world are outside the verified result.

## Build

Use an x64 Developer PowerShell with MSVC, CMake and Ninja. Install the Vulkan SDK and put the SPIR-V tools and an AMDGPU-capable `llvm-mc` on `PATH`. The shader tool setup is described in [testing.md](testing.md). Supply an extracted Xbox 1.0.2.1 package and its matching executable from your own files.

Run the commands from the checkout. Set `$game` to the local package and `$scratch` to a writable directory outside both the package and the checkout. These commands support a fresh shader workspace. SDL3 is fetched during configuration if there is no installed static SDL3 package.

```powershell
$game = Read-Host 'Path to the extracted Xbox 1.0.2.1 package'
$scratch = Read-Host 'Path for the build, shader workspace and port'
$game = (Resolve-Path -LiteralPath $game).Path
$build = Join-Path $scratch 'build'
$work = Join-Path $scratch 'work'
$port = Join-Path $scratch 'port'
cmake -S runtime -B $build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLEONIDA_REQUIRE_VULKAN=ON -DLEONIDA_D3D12_SHIM_TESTS=ON -DLEONIDA_SDL3_FETCH=ON
cmake --build $build --target gta6rt
python -I make_game.py --game $game --work $work --out $port --runtime-build $build --stages check,shaders,runtime
```

Keep the generated shader workspace. The port reads the shader directory recorded in `shader_path.txt`. Native graphics pipelines also use the source containers in the sibling `dxbc` directory. The build indexes the base, library, FidelityFX, BVH and executable-embedded programs.

## Launch

Run the title offline. The Winsock shim permits loopback and the WinHTTP shim refuses outbound requests. The hardware verification used a Windows AppContainer with zero network capabilities. [Boot runbook](boot.md)

```powershell
& (Join-Path $port 'gta6rt.exe') --first-run-ui (Join-Path $game 'game_scarlett_master.exe')
```

The profile selects Vulkan, a 2 GiB graphics address arena, on-demand pipelines, console root-argument layout, frame pacing and the keyboard gamepad in its focused mode (`LEONIDA_KEYBOARD_PAD=1`: keys count only while the title window has focus, so typing into another program presses nothing on a consent page; scripted presses still run). GPU completion labels (`LEONIDA_GPU_LABELS`) are not part of it: the list values land the title's list writes, and two writers of one address could rewind a counter. Explicit environment settings take precedence; a variable that is set but empty does not count. Speech synthesis through the host's own runtime is not part of the profile: `LEONIDA_LOCAL_SPEECH=1` turns it on, and without it the silent voice answers. The switches are listed in [`runtime/README.md`](../runtime/README.md). The package directory is the working directory. Guest output goes under `first-run-ui/overlay` beside the runtime, and sparse physical-memory backing uses `first-run-ui/xmem` on that volume. Use `--overlay DIR` or the absolute directory in `LEONIDA_XMEM_BACKING_DIR` to choose other writable locations.

## Contributions

The frame-loop stack comes from [#321](https://github.com/ubiquitarianism/leonida/pull/321). The archive and shader catalog, shader translation, depth and texture support, presentation window and rectangle-list changes retain their upstream commits. Native and derived pipeline support incorporates the work of enx, TomBox, LooperSalty and KkelebekK. Four patches of the published [graphics runtime patches](https://github.com/Sigtryggur-Mani-Gudmundsson/graphics-runtime-patches/tree/d9a0bc1b070d8a36c81b182bc67edd7d24b41bf2) (GPL-3.0, author smg, committed as "Codex") are in this integration almost as published, about 390 lines: `graphics-root-layout` (the switch between the two packings of graphics root arguments, `runtime/shims/d3d12_com.h`), `descriptor-table-refresh` (the title's descriptors pulled into a bound table, and the table scan in the submitting device, `runtime/shims/d3d12_xs_device.cpp` and `runtime/src/vulkan_backend.h`), `descriptor-write-through` (descriptors the layer writes also stored at the title's address, `runtime/shims/d3d12_xs_device.cpp` and `.h`) and `sampler-table-refresh` (title-backed sampler tables, `runtime/shims/d3d12_vulkan_device.cpp`), with their tests in `runtime/test/d3d12_title_va_test.cpp` and `runtime/test/d3d12_m2_textured_test.cpp`. Each region names its patch in a comment.
