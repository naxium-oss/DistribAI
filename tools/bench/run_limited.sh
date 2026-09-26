#!/usr/bin/env bash
# run_limited.sh -- MANDATORY resource-limited execution wrapper.
#
# Any benchmark or training command run for this mission MUST go through
# this wrapper (or the in-process rlimits in the bench harness). It:
#   1. Aborts up-front if free RAM is below BENCH_MIN_FREE_RAM_MB (default 2048).
#   2. Applies hard ulimits to the WHOLE process tree:
#        RLIMIT_AS   = BENCH_MEM_MB      (default 4096 MiB, exact)
#        RLIMIT_CPU  = BENCH_CPU_SEC     (default 300 s, hard == soft)
#        RLIMIT_NPROC= 256, RLIMIT_NOFILE= 256, RLIMIT_FSIZE= 512 MiB
#        RLIMIT_CORE = 0
#   3. Sets torch/CPU thread caps (BENCH_THREADS, default 4) via OMP/MKL.
#   4. Reports peak child RSS via /usr/bin/time and exits non-zero if the
#      child died from a limit violation.
#
# Usage: tools/bench/run_limited.sh <command...>
set -u

MIN_FREE_MB="${BENCH_MIN_FREE_RAM_MB:-2048}"
MEM_MB="${BENCH_MEM_MB:-4096}"
CPU_SEC="${BENCH_CPU_SEC:-300}"
THREADS="${BENCH_THREADS:-4}"

# NPROC is per-UID and counts THREADS too: size to current task usage
# (ps -L) + headroom so torch's thread pool can still spawn.
UID_TASKS=$(ps -u "$(id -u)" -L --no-headers 2>/dev/null | wc -l)
NPROC="${BENCH_NPROC:-0}"
if [ "${NPROC}" -le 0 ]; then NPROC=$((UID_TASKS + 96)); fi

FREE_MB=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)
echo "[run_limited] available RAM ${FREE_MB} MiB (min ${MIN_FREE_MB}); limits: AS=${MEM_MB}MiB CPU=${CPU_SEC}s threads=${THREADS} nproc=${NPROC} (uid tasks: ${UID_TASKS})"
if [ "${FREE_MB}" -lt "${MIN_FREE_MB}" ]; then
  echo "[run_limited] ABORT: insufficient free RAM (< ${MIN_FREE_MB} MiB)." >&2
  exit 2
fi

export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}"

ulimit -f $((512 * 1024)) 2>/dev/null || true   # file size: 512 MiB
ulimit -u "${NPROC}" 2>/dev/null || true        # processes/threads (uid-wide, dynamic)
ulimit -n 256 2>/dev/null || true               # open files
ulimit -c 0 2>/dev/null || true                 # no cores
ulimit -t "${CPU_SEC}" 2>/dev/null || true      # CPU seconds (hard==soft for children)

# RLIMIT_AS: ulimit -v is in KiB, exact.
ulimit -v $((MEM_MB * 1024)) || { echo "[run_limited] ABORT: cannot set RLIMIT_AS." >&2; exit 2; }

if command -v /usr/bin/time >/dev/null 2>&1; then
  /usr/bin/time -v "$@" 2> >(grep -E "Maximum resident|Elapsed|Command (exited|being run)" >&2)
else
  "$@"
fi
status=$?
case ${status} in
  0) echo "[run_limited] OK (exit 0)" ;;
  137|134|139) echo "[run_limited] CHILD DIED (exit ${status}) -- likely limit violation (OOM/SIGKILL/SIGSEGV)." >&2 ;;
  *) echo "[run_limited] child exit ${status}" >&2 ;;
esac
exit ${status}
