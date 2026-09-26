#!/usr/bin/env bash
# run_suite.sh - master orchestrator for the massive test suite.
#   --quick       fast subset (skips long soak + differential)
#   --only CAT    one category: parity|safety|security|edge|adversarial|redteam|works|architecture|property|grid|store|json|run|differential|sanitize
# Every run goes through tools/bench/run_limited.sh.
set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 1

ZIG=".toolchain/zig-x86_64-linux-0.16.0/zig c++"
B=build/cpp_port
LIM=tools/bench/run_limited.sh
S=tools/cpp_port/tests/suite/categories
REPORT=runtime/baselines/suite_report.json
QUICK=0
ONLY=""
for a in "$@"; do
  case "$a" in
    --quick) QUICK=1 ;;
    --only) ONLY="set" ;;  # value taken below
    --only=*) ONLY="${a#--only=}" ;;
    *) if [ "$ONLY" = "set" ]; then ONLY="$a"; fi ;;
  esac
done

mkdir -p build/cpp_port runtime/baselines
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export ZIG_LOCAL_CACHE_DIR="$TMP/zig-local"
export ZIG_GLOBAL_CACHE_DIR="$TMP/zig-global"

zbuild() {
  local out="$1"; shift
  local attempt
  for attempt in 1 2 3; do
    if $ZIG "$@" -o "$out" 2>"$TMP/build_err"; then return 0; fi
    grep -q "CacheCheckFailed" "$TMP/build_err" || break
    sleep 1
  done
  echo "BUILD FAIL: $out"; sed -n 1,8p "$TMP/build_err"; return 1
}

echo "== building suite binaries =="
CORE_SRCS="tools/cpp_port/core/workspace.cpp tools/cpp_port/core/envelope.cpp"
zbuild $B/suite_core      -std=c++17 -O2 -Itools/cpp_port $S/suite_core.cpp      $CORE_SRCS tools/cpp_port/abi/dai_abi.cpp || exit 1
zbuild $B/suite_sandbox   -std=c++17 -O2 -Itools/cpp_port $S/suite_sandbox.cpp   $CORE_SRCS || exit 1
zbuild $B/suite_property  -std=c++17 -O2 -Itools/cpp_port -Itools/cpp_port/tests/suite/categories $S/suite_property.cpp $CORE_SRCS || exit 1
zbuild $B/suite_run       -std=c++17 -O2 -Itools/cpp_port $S/suite_run.cpp        tools/cpp_port/core/workspace.cpp || exit 1
zbuild $B/suite_grid      -std=c++17 -O2 -Itools/cpp_port -lpthread $S/suite_grid.cpp tools/cpp_port/core/envelope.cpp || exit 1
zbuild $B/suite_json      -std=c++17 -O2 -Itools/cpp_port $S/suite_json.cpp || exit 1
zbuild $B/golden_cases    -std=c++17 -O2 -Itools/cpp_port $S/parity/golden_cases.cpp tools/cpp_port/core/workspace.cpp || exit 1
# The store category needs SQLite; skip its build (and its run) without headers.
HAVE_SQLITE=0
if [ -f /usr/include/sqlite3.h ]; then
  HAVE_SQLITE=1
  zbuild $B/suite_store -std=c++17 -O2 -Itools/cpp_port $S/suite_store.cpp \
    tools/cpp_port/grid/store.cpp tools/cpp_port/core/envelope.cpp -lsqlite3 -lpthread || exit 1
fi

want() { [ -z "$ONLY" ] && return 0; [ "$1" = "$ONLY" ] && return 0; return 1; }
declare -a RESULTS=()
FAILS=0

record() { RESULTS+=("$1"); [ "$2" != "0" ] && FAILS=$((FAILS+1)); }

run_cat() { # run_cat <name> <bin> <args...>
  local name="$1"; shift
  echo "-- $name --"
  $LIM "$@" 2>&1 | tee "$TMP/$name.log" | strings | grep -E "FAIL|CATEGORY" || true
  local fails
  fails=$(grep -c "FAIL$" "$TMP/$name.log" 2>/dev/null || echo 1)
  if $LIM "$@" >/dev/null 2>&1; then record "$name: PASS" 0; else record "$name: FAIL" 1; fi
}

want parity      && { echo "-- parity (golden 50) --"; \
  if $LIM python3 $S/parity/golden_cases.py 2>&1 | tee "$TMP/parity.log" | tail -3; then record "parity: PASS" 0; else record "parity: FAIL" 1; fi; }
want safety      && run_cat safety $B/suite_core --category safety
want security    && run_cat security $B/suite_sandbox --category security
want edge        && run_cat edge $B/suite_core --category edge
want adversarial && run_cat adversarial $B/suite_sandbox --category adversarial
want redteam     && run_cat redteam $B/suite_sandbox --category redteam
want works       && run_cat works $B/suite_core --category works
want architecture && run_cat architecture $B/suite_core --category architecture
want property    && run_cat property $B/suite_property
want grid        && run_cat grid $B/suite_grid
want json        && run_cat json $B/suite_json
want store && {
  if [ "$HAVE_SQLITE" = "1" ]; then
    run_cat store $B/suite_store
  else
    echo "-- store --"
    echo "store: SKIPPED (install libsqlite3-dev for the coordinator store)"
    record "store: SKIPPED" 0
  fi
}
want run && {
  if [ "$QUICK" = "1" ]; then
    run_cat run $B/suite_run --seconds 10
  else
    # SOAK_SECS env knob (default 600 = the 10-minute mission soak)
    run_cat run $B/suite_run --seconds "${SOAK_SECS:-600}"
  fi
}
want differential && {
  echo "-- differential --"
  if [ -x ".venv/bin/python" ] && [ "$QUICK" != "1" ]; then
    if $LIM python3 tools/cpp_port/tests/suite/differential_suite.py \
        2>&1 | tee "$TMP/diff.log" | tail -2; then record "differential: PASS" 0
    elif [ $? = 3 ]; then record "differential: SKIPPED" 0
    else record "differential: FAIL" 1; fi
  else
    record "differential: SKIPPED (no .venv or --quick)" 0
  fi
}
want sanitize && {
  echo "-- sanitize --"
  if bash tools/cpp_port/tests/suite/sanitize.sh 2>&1 | tee "$TMP/sanitize.log" | tail -3; then
    record "sanitize: PASS" 0
  else
    record "sanitize: FAIL" 1
  fi
}

# aggregate report
{
  echo "{"
  echo "  \"captured_at\": \"$(date -Iseconds)\","
  echo "  \"mode\": \"$([ $QUICK = 1 ] && echo quick || echo full)$([ -n "$ONLY" ] && echo " only=$ONLY")\","
  first=1
  for r in "${RESULTS[@]:-}"; do
    [ -z "$r" ] && continue
    [ $first = 1 ] && first=0 || echo ","
    name="${r%%: *}"
    status="${r#*: }"
    printf "  \"%s\": \"%s\"" "$name" "$status"
  done
  echo ""
  echo "}"
} > "$REPORT"

echo
echo "==================== SUITE SUMMARY ===================="
for r in "${RESULTS[@]:-}"; do [ -n "$r" ] && echo "  $r"; done
echo "report -> $REPORT"
if [ "$FAILS" = "0" ]; then echo "RUN_SUITE: ALL GREEN"; exit 0; else echo "RUN_SUITE: $FAILS CATEGORY FAILURES"; exit 1; fi
