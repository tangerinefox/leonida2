#!/bin/bash
# Runs inside the leonida-boot-* unit (see run-boot.sh). Waits for memory, takes the locks, samples resources, boots.
set -u
WORK=/opt/leonida/work/$BOOT_AGENT; N=$BOOT_N; LOG=$WORK/boot-$N.log; RES=$WORK/boot-$N.resources.tsv
META=$WORK/boot-$N.meta
: > "$LOG"
echo "start_epoch=$(date +%s)" > "$META"
echo "unit=leonida-boot-$BOOT_AGENT-$N gta6rt=$GTA6RT secs=$BOOT_SECS" >> "$META"
env | grep -E '^(LEONIDA_|WINEDEBUG=)' | sort >> "$META"

mem_free_mb() {  # RAM available plus a third of free zram swap (zram pages cost RAM, roughly 3:1)
  awk '/^MemAvailable:/{a=$2} /^SwapFree:/{s=$2} END{printf "%d", (a + s/3)/1024}' /proc/meminfo
}
NEED=${LEONIDA_BOOT_MIN_FREE_MB:-6500}
swap_pct() { awk '/^SwapTotal:/{t=$2} /^SwapFree:/{f=$2} END{printf "%d", t ? (t-f)*100/t : 0}' /proc/meminfo; }

exec 8>/opt/leonida/.gpu.lock; flock 8                   # one boot at a time
echo "[run-boot] $(date +%T) got the GPU lock" >> "$LOG"
for i in $(seq 1 40); do                                 # up to 10 min for memory headroom
  f=$(mem_free_mb); sp=$(swap_pct); [ "$f" -ge "$NEED" ] && [ "$sp" -lt 70 ] && break
  echo "[run-boot] $(date +%T) waiting for memory: ${f} MB free (need ${NEED}), swap ${sp}% (need < 70%)" >> "$LOG"; sleep 15
done
echo "[run-boot] $(date +%T) memory before boot: $(mem_free_mb) MB free, swap $(swap_pct)%" >> "$LOG"
exec 9>/opt/leonida/.build.lock; flock 9                 # no compiles while the game runs
echo "[run-boot] $(date +%T) got the build lock; booting" >> "$LOG"
echo "boot_epoch=$(date +%s)" >> "$META"

CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
( printf 'time\tmem_avail_mb\tswap_used_mb\tunit_mem_mb\tunit_swap_mb\tgpu_mem_mb\tgpu_util\n'
  while :; do
    read -r ma su < <(awk '/^MemAvailable:/{a=$2} /^SwapTotal:/{t=$2} /^SwapFree:/{f=$2} END{print int(a/1024), int((t-f)/1024)}' /proc/meminfo)
    um=$(( $(cat "$CG/memory.current" 2>/dev/null || echo 0) / 1048576 )); us=$(( $(cat "$CG/memory.swap.current" 2>/dev/null || echo 0) / 1048576 ))
    g=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr ',' '\t')
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%T)" "$ma" "$su" "$um" "$us" "${g:-?	?}"
    sleep 2
  done ) > "$RES" 2>/dev/null &
SAMPLER=$!

/opt/leonida/boot.sh run "$BOOT_SECS" >> "$LOG" 2>&1
rc=$?
kill $SAMPLER 2>/dev/null
echo "[run-boot] $(date +%T) boot.sh exit $rc" >> "$LOG"
# boot.sh stops the title with timeout(1) at the time limit: 124 means it ran the whole time, which is success here.
[ "$rc" = 124 ] && { echo "[run-boot] time limit reached" >> "$LOG"; rc=0; }
exit $rc
