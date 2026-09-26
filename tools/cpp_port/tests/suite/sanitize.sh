#!/usr/bin/env bash
# sanitize.sh (suite category: sanitize) - ASan/TSan/UBSan on core+sandbox.
# Self-skips with an explicit note when the toolchain cannot run sanitizers
# (this box's zig has no ASan runtime for the target); the orchestrator gates
# port-check on a CLEAN run, and a skipped run reports SKIP, not PASS.
set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 1

ZIG=".toolchain/zig-x86_64-linux-0.16.0/zig c++"
B=build/cpp_port
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export ZIG_LOCAL_CACHE_DIR="$TMP/zig-local"
export ZIG_GLOBAL_CACHE_DIR="$TMP/zig-global"

SAN="-fsanitize=address,undefined -fno-omit-frame-pointer"
FAIL=0
RAN=0

build_and_run() { # <bin_suffix> <category> <sources...>
  local name="$1"; local cat="$2"; shift 2
  echo "-- sanitize: $name ($cat) --"
  if ! $ZIG -std=c++17 -O1 -g $SAN -Itools/cpp_port "$@" -o "$B/suite_san_$name" \
       2>"$TMP/san_build_err"; then
    if grep -qE "cannot find|unknown argument|not supported|no such file.*sanit|undefined symbol: __asan|undefined symbol: __ubsan|undefined symbol: __tsan" "$TMP/san_build_err"; then
      echo "SKIP: sanitizer runtime unavailable for this toolchain"
      sed -n 1,3p "$TMP/san_build_err"
      return 2
    fi
    echo "BUILD FAIL: $name"; sed -n 1,8p "$TMP/san_build_err"; return 1
  fi
  RAN=$((RAN+1))
  local log="$TMP/san_$name.log"
  if ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 \
     tools/bench/run_limited.sh "$B/suite_san_$name" --category "$cat" > "$log" 2>&1; then
    strings "$log" | grep -E "CATEGORY" || true
    if grep -qiE "ERROR: (Address|Leak)Sanitizer|runtime error:|SUMMARY: .*Sanitizer" "$log"; then
      echo "SANITIZER FINDINGS in $name:"; grep -iE "ERROR|runtime error" "$log" | head -5
      return 1
    fi
    return 0
  fi
  echo "SANITIZED RUN FAILED: $name"; strings "$log" | grep -E "FAIL|ERROR|runtime error" | head -8
  return 1
}

CORE_ARGS="tools/cpp_port/tests/suite/categories/suite_core.cpp tools/cpp_port/core/workspace.cpp tools/cpp_port/core/envelope.cpp tools/cpp_port/abi/dai_abi.cpp"
SANDBOX_ARGS="tools/cpp_port/tests/suite/categories/suite_sandbox.cpp tools/cpp_port/core/workspace.cpp tools/cpp_port/core/envelope.cpp"
for cat in edge safety works architecture; do
  build_and_run core "$cat" $CORE_ARGS
  rc=$?
  [ $rc = 1 ] && FAIL=$((FAIL+1))
  [ $rc = 2 ] && { echo "sanitize: SKIPPED (toolchain)"; exit 3; }
done
for cat in security adversarial redteam; do
  build_and_run sandbox "$cat" $SANDBOX_ARGS
  rc=$?
  [ $rc = 1 ] && FAIL=$((FAIL+1))
  [ $rc = 2 ] && { echo "sanitize: SKIPPED (toolchain)"; exit 3; }
done

if [ $FAIL = 0 ]; then
  echo "SANITIZE: ALL GREEN ($RAN sanitized runs)"
  exit 0
fi
echo "SANITIZE: $FAIL FAILURES"
exit 1
