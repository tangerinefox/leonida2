#!/bin/bash
# usage: configure-worktree.sh <worktree>
# Configures a Windows build of the runtime on Linux with clang-cl and an xwin SDK (XWIN_DIR, default
# ~/.local/share/xwin-sdk). SDL3_SRC=<an SDL3 source tree> makes the configure work offline.
set -eu; W=$(realpath "$1")
cmake -S "$W/runtime" -B "$W/runtime/build/boot" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$W/runtime/cmake/clang-cl-xwin.cmake" -DXWIN_DIR="${XWIN_DIR:-$HOME/.local/share/xwin-sdk}" \
  -DLEONIDA_BUILD_WINDOWS_RUNTIME=ON -DLEONIDA_SDL3=ON \
  ${SDL3_SRC:+-DFETCHCONTENT_SOURCE_DIR_SDL3="$SDL3_SRC"}
