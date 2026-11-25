#!/bin/bash
# Boot the packaged port under Wine with the network off (unshare -rn: user + network namespace, no root).
# usage: boot.sh dry-run|init|run [seconds]     (run defaults to 150 s, then the title is stopped)
# Traces stay local in /opt/leonida/traces: they can quote game data, never post them.
set -u
MODE=${1:?dry-run|init|run}; SECS=${2:-150}
OUT=/opt/leonida
REPO=${LEONIDA_REPO:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}   # the checkout this script is in
STAMP=$(date +%H%M%S)
T=$OUT/traces; mkdir -p "$T" "$OUT/overlay" "$OUT/cache/nv"

WINE=${WINE:-wine}   # e.g. WINE=".../Proton - Experimental/files/bin/wine" WINEPREFIX=~/.local/share/leonida-proton/pfx
export WINEPREFIX=${WINEPREFIX:-$HOME/.local/share/leonida-wine} WINEDEBUG=${WINEDEBUG:--all} WINEDLLOVERRIDES="mscoree,mshtml="
export LEONIDA_D3D12=vulkan
export LEONIDA_SHADERS=$OUT/work/shaders/spirv
export LEONIDA_ARENA_MB=${LEONIDA_ARENA_MB:-2048}
export LEONIDA_PIPELINE_CACHE_BYTES=${LEONIDA_PIPELINE_CACHE_BYTES:-4294967296}
export LEONIDA_STARTUP_CACHE_CHECKPOINTS=1
export __GL_SHADER_DISK_CACHE=1 __GL_SHADER_DISK_CACHE_PATH=$OUT/cache/nv __GL_SHADER_DISK_CACHE_SIZE=21474836480
export __GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1
[ -n "${SKIP_INT3:-}" ] && export LEONIDA_SKIP_INT3=1

GAME_WIN='Z:\opt\gta6\GTA6.Xbox.v1.0.2.1'
EXE=/opt/gta6/GTA6.Xbox.v1.0.2.1/game_scarlett_master.exe
TRACE_WIN="Z:\\opt\\leonida\\traces\\boot-$MODE-$STAMP.jsonl"
D3D=$T/d3d12-$MODE-$STAMP.log
GTA6RT=${GTA6RT:-$OUT/gta6rt.exe}; [ -f "$GTA6RT" ] || GTA6RT=$REPO/runtime/build/boot/gta6rt.exe   # GTA6RT=<your build>/gta6rt.exe boots a private build without installing it

args=(--"$MODE" "$EXE" --game-dir "$GAME_WIN" --overlay 'Z:\opt\leonida\overlay' --trace "$TRACE_WIN")
[ "$MODE" = run ] && args+=(--trace-d3d12)
echo "[boot] $MODE  gta6rt=$GTA6RT  trace=$T/boot-$MODE-$STAMP.jsonl  d3d12=$D3D"
# unshare: new user + network namespace, so the title has no network at all
if [ "$MODE" = run ]; then
  timeout --signal=INT --kill-after=20 "$SECS" unshare -rn "$WINE" "$GTA6RT" "${args[@]}" > "$T/stdout-$MODE-$STAMP.log" 2> "$D3D"
else
  unshare -rn "$WINE" "$GTA6RT" "${args[@]}" > "$T/stdout-$MODE-$STAMP.log" 2> "$D3D"
fi
rc=$?; echo "[boot] exit=$rc"
sleep 0
python3 -I "$REPO/tools/boot_trace.py" "$T/boot-$MODE-$STAMP.jsonl" --d3d12 "$D3D" > "$T/summary-$MODE-$STAMP.txt" 2>&1
echo "[boot] summary: $T/summary-$MODE-$STAMP.txt"
exit $rc
