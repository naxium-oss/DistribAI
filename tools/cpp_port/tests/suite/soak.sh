#!/usr/bin/env bash
# soak.sh (category: run) - 10-minute continuous train+sandbox soak under
# the mandatory limits. Wraps suite_run (which itself loops train bursts +
# sandbox spawns while census-ing FDs/threads/RSS and spawn-cost drift).
#   bash tools/cpp_port/tests/suite/soak.sh [SECONDS]   # default 600
set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 1

ZIG=".toolchain/zig-x86_64-linux-0.16.0/zig c++"
B=build/cpp_port
SECS="${1:-600}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export ZIG_LOCAL_CACHE_DIR="$TMP/zig-local"
export ZIG_GLOBAL_CACHE_DIR="$TMP/zig-global"

mkdir -p "$B"
$ZIG -std=c++17 -O2 -Itools/cpp_port tools/cpp_port/tests/suite/categories/suite_run.cpp \
  tools/cpp_port/core/workspace.cpp -o "$B/suite_run" 2>/dev/null || { echo "BUILD FAIL: suite_run"; exit 1; }

echo "soak: ${SECS}s continuous train+sandbox under run_limited"
tools/bench/run_limited.sh "$B/suite_run" --seconds "$SECS"
