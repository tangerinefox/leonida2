# Linux (Proton/Wine) and second-machine test report

This page describes how the runtime is built and booted on Linux through Proton's Wine, what a healthy boot looks
like there, and what was observed on the machines that tested this branch (one Linux, two native Windows). It holds
counts, timings and settings only: no game data, no traces.

Status in one line: on Linux/Proton and on native Windows the retail 1.0.2.1 package launches, shows its launch
movie, first-run pages and landing prompt, runs the story load to its end with the loading slideshow drawing its
artwork, and enters the world phase. A first world frame has not been verified yet on any of the three machines; what
stops it is listed under [Where it stops](#where-it-stops).

## Test machines

| | Linux | Windows A | Windows B |
|---|---|---|---|
| OS | CachyOS (Arch), kernel 7.3 | Windows 11 | Windows 11 |
| Run through | Proton Experimental (Wine 11) | native | native |
| CPU | Ryzen 5 5600X | Core i9-13900K | Ryzen 7 9800X3D |
| GPU, driver | RTX 4070 12 GB, NVIDIA 620.73 | RTX 4070 Ti 12 GB, NVIDIA 630.20 | RTX 5070 |
| RAM | 16 GB (+32 GB swap file) | 64 GB | not reported yet |
| Toolchain | clang-cl 23 + xwin SDK, Ninja, ccache | MSVC 2022, SDK 26100 | - |
| Results here | yes, ~80 boots on 2026-11-24 | yes | joining; none yet |

Windows A reaches every stage three to five times faster than the Linux machine (initialisation ~70 s against
~180 s, story load from ~110 s against ~240-300 s), so it is the better machine for iterating on the world phase;
the Linux machine is the one that shows what 16 GB of RAM allows.

## Building on Linux

The runtime is a Windows executable. On Linux it is cross-built with clang-cl against an [xwin](https://github.com/Jake-Shadle/xwin)
copy of the Windows SDK:

```sh
scripts/linux-boot/configure-worktree.sh <checkout>      # XWIN_DIR=<sdk>, SDL3_SRC=<SDL3 tree> for an offline configure
cmake --build <checkout>/runtime/build/boot --target gta6rt -j6
```

`gta6rt_tls_host.dll` is built with the `gta6rt` target and must stay next to the executable. Unit tests are Windows
executables too and run under Wine (`scripts/run_wine_tests.py`, or directly with the same `wine` binary and a
prefix of their own: sharing the game's prefix with tests breaks its window).

## Booting on Linux

`scripts/linux-boot/boot.sh` is the plain launcher and `run-boot.sh` wraps it as a `systemd --user` unit with a
memory sampler, a GPU lock, a build lock (no compile while the game runs) and a digest after the boot. They assume
the layout the comments at their top describe (`/opt/leonida` for traces, overlay and caches; the package under
`/opt/gta6`); adjust the paths at the top to yours. What matters:

- **Wine:** Proton Experimental's `files/bin/wine` with its own prefix (`WINEPREFIX`), `WINEDLLOVERRIDES="mscoree,mshtml="`.
- **No network:** the title runs inside `unshare -rn` (new user and network namespace). Nothing is online.
- **Backend:** `LEONIDA_D3D12=vulkan`, `LEONIDA_SHADERS=<translated library>`.
- **NVIDIA shader cache:** `__GL_SHADER_DISK_CACHE_SIZE` raised (the library is ~19 GB of SPIR-V) and
  `__GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1`.
- **Input:** a pad through GameInput/SDL3, or the keyboard pad (`LEONIDA_KEYBOARD_PAD=always`), or a script:
  `LEONIDA_INPUT_SCRIPT=a@210,a@240` presses A at those seconds (any length since this branch; it was cut at 63
  characters before). The script also works in a build without SDL3, where it drives one scripted pad
  ([docs/gameinput.md](gameinput.md#scripted-input-without-sdl3)); pads and the keyboard pad need SDL3.
- **Window:** frames are presented in the title's own window; `LEONIDA_PRESENT_WINDOW=own|hidden` chooses otherwise.

Three setup mistakes each give a blank or empty picture without an error: the executable's embedded shaders missing
from the library (`exe/index.json`), a build without SDL3 and without `LEONIDA_INPUT_SCRIPT` (no pad, so nothing
gets past the first prompt; set a script, or build with SDL3 for a pad or the keyboard pad), and a BCPack predictor
file that was never generated (`tools/build_dstorage_history.py`; without it the loading art and most BC7 textures
stay empty).

## Running the game under Wine with one command

`scripts/run_game_wine.sh` makes one bounded, silent run of the retail game under Wine and prints counts only. It
complements `scripts/run_wine_session.sh` (the runtime's unit tests under Wine) and the `systemd --user` boot
wrappers described above: it needs no fixed paths, and its output is a summary that can go into an issue.

```sh
scripts/run_game_wine.sh --build <build folder or checkout> --exe <game executable> --game-dir <package folder> \
    --shaders <translated library> --out <new run folder> --device 1002:744c --seconds 600
```

`--help` lists every input. Each path comes from an argument or from `LEONIDA_GAME_EXE`, `LEONIDA_GAME_DIR` and
`LEONIDA_SHADERS`; `WINE` names another Wine (Proton's `files/bin/wine`), and `WINEPREFIX` (or `--prefix`) a prefix to
reuse instead of a new one in the run folder.

What it needs:

- **Wine** (the `wine` on `PATH`, or `WINE`), `timeout`, `flock`, `setsid` and Python 3.
- **A Vulkan driver** for the GPU (Mesa's RADV was tested). `--device VID:DID` pins it through
  `MESA_VK_DEVICE_SELECT`; without it the script warns that Mesa picks the device.
- **A display.** By default the run is headless: a private `kwin_wayland --virtual --xwayland` session under
  `dbus-run-session`, with its own `XDG_RUNTIME_DIR` in the run folder, so nothing appears on the desktop. Its
  rootless Xwayland offers DRI3, which RADV needs to present; Xvfb has none, so without `kwin_wayland` the script
  says what is missing instead of falling back. `--visible` opens the window on the current `DISPLAY` instead.
- **A runtime build**: a folder with `gta6rt.exe` and `gta6rt_tls_host.dll`, or a checkout, whose `runtime/` it
  cross-builds first with clang-cl and xwin (`XWIN_DIR`, see [linux.md](linux.md#the-runtime-under-wine-or-proton)).

What it does:

- **Sound off:** Wine's audio drivers are disabled through `WINEDLLOVERRIDES`, PulseAudio, PipeWire and ALSA point
  at nothing, and `LEONIDA_AUDIO=silent` makes the runtime serve its silent output endpoint
  ([runtime_imports.md](runtime_imports.md)); the summary says whether it was served.
- **Limits:** a hard time limit (`--seconds`, then SIGTERM and SIGKILL 10 s later), a lock so one run happens at a
  time (`--lock`, by default in `XDG_RUNTIME_DIR`), and teardown of Wine and the whole session on exit, time-out or
  signal.
- **Traces:** `gta6rt --run` with `--trace` and `--trace-d3d12`, a fresh `--overlay` and save folder in the run
  folder. Every other `LEONIDA_*` variable reaches the runtime unchanged, so `LEONIDA_INPUT_SCRIPT=a@210,a@240`
  presses A as above. Arguments after `--` go to `gta6rt`.
- **Summary:** `tools/game_run_summary.py` prints the run time, how it ended, exceptions by code, D3D12 calls by
  outcome, presents, the most frequent failing calls and the `[rt]` lines by category with numbers masked. It never
  prints a log line. `--screenshot-every S` captures the game window into the run folder (ImageMagick's `import` and
  `xprop`).

The run folder's `stderr.log`, `boot.jsonl` and screenshots can show or quote what the title prints, so they stay on
your machine; share the summary.

## What a healthy boot looks like (Linux machine)

| Time | Stage |
|---|---|
| 0-10 s | package splash |
| ~60-90 s | launch movie, then the landing prompt appears while initialisation finishes |
| ~180 s | initialisation done; A is accepted |
| ~240-300 s | story load starts: loading slideshow with its artwork |
| ~540-630 s | story load ends (about 38,000 streamed images registered), world phase begins |

On Windows A the same stages are at ~70 s, ~110 s and ~190 s.

## What this branch changes to get there

Each item was a stop on at least one of the machines. Tests named in the pull request cover them.

- **Presentation:** frames go to the title's own window (they went to a window that was never shown).
- **DirectStorage:** reads pin write-watched destination pages, page-list scatter/gather, BCPack and unshuffle, a
  locally generated predictor file, bounded staging.
- **Story-load allocation stall:** one device allocation per streamed texture exhausted the driver's CPU-visible
  mappings near 16,000 images; small streamed images now share pooled pages.
- **Images on plain device-local memory:** host-copy usage excluded it on NVIDIA and 4.8 GB of images spilled into
  host memory; streamed images are now filled through staging.
- **Deferred backing:** placed and streamed images get private memory at their first write or GPU copy. VRAM at the
  end of the load went from 11.8 GB to ~5 GB on Linux; on Windows A host private memory went from 22-24 GB to 11.9 GB.
- **Enhanced barrier into `LAYOUT_UNDEFINED`:** accepted; it failed `Close`, which the title treats as fatal.
- **Bindless table:** a table rebuild that cannot allocate no longer refuses the whole submission; larger image
  arrays; a hash finalizer for the descriptor table (runtime and translator halves, off until a library carries it).
- **Translator:** loop bounds on every emitted loop (5,128 compute modules have loops), the eight FidelityFX programs
  that used an instruction form the disassembler skips (all 29 translate), buffer bounds modes per view kind.
- **Directly indexed descriptor heaps:** the heap offset is supplied in the user-data dword after the explicit root
  arguments for compute and for VS/PS/GS draws. Mesh stages are not covered yet.
- **Title buffers at raw addresses:** a window outside the arena when title reservations fragment it; opt-in
  persistent device-local copies of streamed buffer views.
- **Native Windows:** write faults on shared pages retried under the tracker lock, one tracker per run of shared
  pages, write-watched title memory, a fault-lock that never waits for the process heap.
- **Diagnostics:** device-loss report with the last dispatches and NVIDIA checkpoints, a probe for the descriptors a
  compute dispatch receives, memory probe, press log. All off by default.

## Where it stops

- **Streamed memory that reaches its destination late or not at all.** The title calls queue slot 31
  (`CopyPageMappingsBatchX`) about 300 times per boot. With #968 this branch maps each batch's pool pages at the
  destination and at the paired CPU view the layer reads (`docs/d3d12_vulkan.md`, section 4.9), so most streamed
  views find readable memory; before it, tens of thousands of them waited for memory that never became readable. Two gaps
  remain: the mappings change when the call is made, not in queue order, and a run with the CPU-view fix still had
  views waiting after 420 s, apparently for ranges no mapping call had reached yet.
- **Culling compute at world start** reads buffers named by descriptors in the directly indexed heap; where the
  memory behind them is not mapped, those reads leave mapped memory and the device is lost (NVIDIA Xid 13/109 on
  Linux).
- **Mesh stages** do not get the heap offset yet.
- **16 GB of RAM:** at the end of the load the process holds ~14.6 GB (RAM plus swap) on the Linux machine after the
  cuts above. zram made this worse (compressed pages still occupy RAM); a swap file on disk is the better choice.
- **Initialisation takes ~180 s under Proton** against ~70 s natively. Not investigated.
- **Under Wine** a thread that takes an access violation while holding the process heap lock can still deadlock
  with xmem's fault handler; the test for it is skipped there.

## Reporting a result

Say the machine (OS, GPU and driver, RAM), the commit, the environment switches that differ from the defaults, and
counts: presents, the last stage reached with its time, the first refusal or device-loss line. Keep logs and traces
on the machine that produced them.
