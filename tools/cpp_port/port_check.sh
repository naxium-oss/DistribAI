#!/usr/bin/env bash
# port_check.sh (O8) - one-shot build + gates + bench report for the C++ port.
# Regenerates runtime/baselines/cpp_port_check_report.md and exits nonzero on
# any gate failure. All runs go through tools/bench/run_limited.sh.
#   tools/cpp_port/port_check.sh [--quick]   # or: make port-check [ARGS=--quick]
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

ZIG=".toolchain/zig-x86_64-linux-0.16.0/zig c++"
B=build/cpp_port
LIM=tools/bench/run_limited.sh
REPORT=runtime/baselines/cpp_port_check_report.md
QUICK="${1:-}"
STEPS=1000
[ "$QUICK" = "--quick" ] && STEPS=200

mkdir -p build/cpp_port runtime/baselines
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# Private per-run zig cache: the shared-box default cache intermittently
# fails with CacheCheckFailed under concurrent access from other users.
export ZIG_LOCAL_CACHE_DIR="$TMP/zig-local"
export ZIG_GLOBAL_CACHE_DIR="$TMP/zig-global"
fail=0

zbuild() { # zbuild <out> <zig args...>  (paths relative to repo root)
  local out="$1"; shift
  local attempt
  for attempt in 1 2 3; do
    if $ZIG "$@" -o "$out" 2>"$TMP/build_err"; then return 0; fi
    if ! grep -q "CacheCheckFailed" "$TMP/build_err"; then break; fi
    sleep 1  # transient shared-cache fault: retry
  done
  echo "BUILD FAIL: $out"
  sed -n 1,5p "$TMP/build_err"
  fail=1
  return 1
}

echo "== port-check: building =="
zbuild $B/distribai_train -std=c++17 -O2 -I. tools/cpp_port/apps/train.cpp tools/cpp_port/gpu/gpu_shim.cpp
zbuild $B/distribai_bench -std=c++17 -O2 -I. tools/cpp_port/apps/bench.cpp tools/cpp_port/gpu/gpu_shim.cpp
zbuild $B/test_golden     -std=c++17 -O2 -I. tools/cpp_port/tests/test_golden.cpp tools/cpp_port/core/workspace.cpp
zbuild $B/test_parity     -std=c++17 -O2 -I. tools/cpp_port/tests/test_parity.cpp
zbuild $B/distribai_dualrun -std=c++17 -O2 -I. tools/cpp_port/apps/dualrun.cpp tools/cpp_port/core/workspace.cpp
zbuild $B/test_features   -std=c++17 -O2 -I. tools/cpp_port/tests/test_features.cpp tools/cpp_port/core/workspace.cpp tools/cpp_port/core/envelope.cpp
zbuild $B/dai_job         -std=c++17 -O2 -I. tools/cpp_port/apps/dai_job.cpp tools/cpp_port/core/workspace.cpp tools/cpp_port/core/envelope.cpp
zbuild $B/libdai_core.so  -std=c++17 -O2 -I. -shared -fPIC tools/cpp_port/abi/dai_abi.cpp tools/cpp_port/core/workspace.cpp tools/cpp_port/core/envelope.cpp
cc tools/cpp_port/tests/abi_consumer.c -Itools/cpp_port/abi -o $B/abi_consumer -ldl 2>/dev/null \
  || { echo "BUILD FAIL: abi_consumer"; fail=1; }

echo "== port-check: gates =="
$LIM $B/test_golden   2>&1 | tee "$TMP/golden.log"  | tail -2
grep -q "ALL MUST-MATCH CONTRACTS GREEN" "$TMP/golden.log" || fail=1
$LIM $B/test_parity   2>&1 | tee "$TMP/parity.log"  | tail -1
grep -q "ALL PARITY TESTS PASSED" "$TMP/parity.log" || fail=1
$LIM $B/test_features 2>&1 | tee "$TMP/feat.log"    | tail -1
grep -q "ALL FEATURE TESTS PASSED" "$TMP/feat.log" || fail=1
$LIM $B/abi_consumer  2>&1 | tee "$TMP/abi.log"     | tail -1
grep -q "ABI CONSUMER TEST PASSED" "$TMP/abi.log" || fail=1

echo "== port-check: bench ($STEPS steps) =="
$LIM $B/distribai_bench 2>&1 | tee "$TMP/bench.log" | grep -E "train 1K|sandbox 1K|A\)|B\)"
grep -q "ok=1" "$TMP/bench.log" || fail=1

# e2e grad-path cycle (O5 evidence) + 100K model under tight RLIMIT_AS (O7)
zbuild $TMP/prof13 -std=c++17 -O2 -Itools/cpp_port build/prof/prof13.cpp tools/cpp_port/core/workspace.cpp tools/cpp_port/core/envelope.cpp \
  && $LIM $TMP/prof13 2>&1 | grep "A)" | tee "$TMP/grad.log"
zbuild $TMP/prof14 -std=c++17 -O2 -Itools/cpp_port build/prof/prof14.cpp tools/cpp_port/core/workspace.cpp \
  && $LIM $TMP/prof14 2>&1 | grep "mem_cap" | tee "$TMP/mem.log"
[ -s "$TMP/mem.log" ] && grep -q "ok=1" "$TMP/mem.log" || fail=1

# Differential parity-vs-Python gate (fixed corpus; needs .venv with torch).
# This is the full-parity check: must-match tolerances from parity_matrix.md.
if [ -x ".venv/bin/python" ]; then
  echo "== port-check: differential parity gate (python vs c++, 10 seeds) =="
  if $LIM .venv/bin/python tools/cpp_port/tests/suite/differential_suite.py 2>&1 \
      | tee "$TMP/diff.log" | tail -2; then :; else fail=1; fi
  grep -q "DIFFERENTIAL GATE: PASS" "$TMP/diff.log" || fail=1
else
  echo "differential: SKIPPED (no .venv)"
fi

# LibTorch live train path (the ONLY train path): needs g++ (libstdc++ ABI,
# C++20) plus torch in .venv; skipped cleanly when either is missing.
# Prefer the shared .venv; fall back to python3 (CI installs torch there).
if [ -x .venv/bin/python ]; then TORCH_PY=.venv/bin/python; else TORCH_PY=$(command -v python3 || true); fi
TORCH_DIR=$( [ -n "$TORCH_PY" ] && "$TORCH_PY" -c "import torch,os;print(os.path.dirname(torch.__file__))" 2>/dev/null )
if command -v g++ >/dev/null 2>&1 && [ -n "${TORCH_DIR:-}" ]; then
  echo "== port-check: LibTorch live train path =="
  if g++ -std=c++20 -O2 -Itools/cpp_port \
      -isystem "$TORCH_DIR/include" -isystem "$TORCH_DIR/include/torch/csrc/api/include" \
      tools/cpp_port/torch/torch_train.cpp tools/cpp_port/core/envelope.cpp \
      -o "$B/distribai_torch_train" \
      -L"$TORCH_DIR/lib" -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_DIR/lib" 2>"$TMP/torch_build.log"; then
    :
  else
    echo "BUILD FAIL: distribai_torch_train"; sed -n 1,8p "$TMP/torch_build.log"; fail=1
  fi
  $LIM "$TORCH_PY" tools/cpp_port/tests/torch_parity.py 2>&1 | tee "$TMP/torch.log" | tail -3
  grep -q "TORCH PARITY: PASS" "$TMP/torch.log" || fail=1
else
  echo "torch path: SKIPPED (needs g++ and .venv torch)"
fi

# Translator: Python trainer -> LibTorch job, including fail-closed paths.
# Needs torch AND pytest; skipped cleanly when the python env lacks either.
if [ -n "$TORCH_PY" ] && [ -n "${TORCH_DIR:-}" ] \
    && "$TORCH_PY" -c "import pytest" 2>/dev/null \
    && [ -d tools/trainer_translate/tests ]; then
  echo "== port-check: trainer translator (python -> LibTorch) =="
  if "$TORCH_PY" -m pytest tools/trainer_translate/tests -q -p no:cacheprovider \
      > "$TMP/translate.log" 2>&1; then
    tail -1 "$TMP/translate.log" | tee "$TMP/translate_summary.log"
  else
    echo "TRANSLATOR TEST FAILURES:"; tail -20 "$TMP/translate.log"; fail=1
  fi
else
  echo "translator: SKIPPED (no .venv python)"
fi

# gridlink: expose/join links and free tunnel providers (Cloudflare quick
# tunnels, optional ngrok). No torch needed; runs whenever pytest is present.
if [ -n "$TORCH_PY" ] && "$TORCH_PY" -c "import pytest" 2>/dev/null \
    && [ -d tools/gridlink/tests ]; then
  echo "== port-check: gridlink (expose / join / providers) =="
  if "$TORCH_PY" -m pytest tools/gridlink/tests -q -p no:cacheprovider \
      > "$TMP/gridlink.log" 2>&1; then
    tail -1 "$TMP/gridlink.log" | tee "$TMP/gridlink_summary.log"
  else
    echo "GRIDLINK TEST FAILURES:"; tail -20 "$TMP/gridlink.log"; fail=1
  fi
else
  echo "gridlink: SKIPPED (no pytest)"
fi

# Native grid: coordinator (SQLite) + worker + one job through the whole path.
# Needs libsqlite3-dev; skipped cleanly without it.
if [ -f /usr/include/sqlite3.h ]; then
  echo "== port-check: native grid (coordinator + workers) =="
  if make -C tools/cpp_port grid > "$TMP/grid_build.log" 2>&1; then
    if $LIM bash tools/cpp_port/tests/grid_e2e.sh > "$TMP/grid.log" 2>&1; then
      grep -c "   PASS" "$TMP/grid.log" | sed 's/^/grid checks passed: /' | tee "$TMP/grid_summary.log"
    else
      echo "GRID GATE FAILURES:"; grep -E "FAIL|GRID GATE" "$TMP/grid.log" | head -10; fail=1
    fi
    # Edge cases and regressions against the same real binaries.
    if $LIM bash tools/cpp_port/tests/grid_edge.sh > "$TMP/grid_edge.log" 2>&1; then
      grep -c "   PASS" "$TMP/grid_edge.log" | sed 's/^/grid edge checks passed: /' \
        | tee "$TMP/grid_edge_summary.log"
      grep -E "GRID EDGE GATE" "$TMP/grid_edge.log" >> "$TMP/grid_edge_summary.log"
    else
      echo "GRID EDGE GATE FAILURES:"
      grep -E "FAIL|GRID EDGE GATE" "$TMP/grid_edge.log" | head -10
      fail=1
    fi
    if $LIM bash tools/cpp_port/tests/grid_regression.sh > "$TMP/grid_reg.log" 2>&1; then
      grep -E "GRID REGRESSION GATE" "$TMP/grid_reg.log" | tee "$TMP/grid_reg_summary.log"
    else
      echo "GRID REGRESSION GATE FAILURES:"
      grep -E "FAIL|GRID REGRESSION GATE" "$TMP/grid_reg.log" | head -10
      fail=1
    fi
    # Authorization and the session model.
    if $LIM bash tools/cpp_port/tests/grid_security.sh > "$TMP/grid_sec.log" 2>&1; then
      grep -c "   PASS" "$TMP/grid_sec.log" | sed 's/^/grid security checks passed: /' \
        | tee "$TMP/grid_sec_summary.log"
      grep -E "GRID SECURITY GATE" "$TMP/grid_sec.log" >> "$TMP/grid_sec_summary.log"
    else
      echo "GRID SECURITY GATE FAILURES:"
      grep -E "FAIL|GRID SECURITY GATE" "$TMP/grid_sec.log" | head -10
      fail=1
    fi
    # Load: many workers, many jobs, reconciled ledger.
    if $LIM bash tools/cpp_port/tests/grid_soak.sh > "$TMP/grid_soak.log" 2>&1; then
      grep -c "   PASS" "$TMP/grid_soak.log" | sed 's/^/grid soak checks passed: /' \
        | tee "$TMP/grid_soak_summary.log"
      grep -E "GRID SOAK GATE" "$TMP/grid_soak.log" >> "$TMP/grid_soak_summary.log"
    else
      echo "GRID SOAK GATE FAILURES:"
      grep -E "FAIL|GRID SOAK GATE" "$TMP/grid_soak.log" | head -10
      fail=1
    fi
    # One operator's whole day, including a cancel and a coordinator restart.
    if $LIM bash tools/cpp_port/tests/real_usage.sh > "$TMP/usage.log" 2>&1; then
      grep -c "   PASS" "$TMP/usage.log" | sed 's/^/real usage checks passed: /' \
        | tee "$TMP/usage_summary.log"
      grep -E "REAL USAGE GATE" "$TMP/usage.log" >> "$TMP/usage_summary.log"
    else
      echo "REAL USAGE GATE FAILURES:"
      grep -E "FAIL|REAL USAGE GATE" "$TMP/usage.log" | head -10
      fail=1
    fi
    # The product slice: a translated PyTorch script runs on the real LibTorch
    # trainer across two workers, plus the trainer's own edge cases. Both skip
    # themselves without torch.
    if [ -x "$B/distribai_torch_train" ]; then
      if $LIM bash tools/cpp_port/tests/grid_e2e_torch.sh > "$TMP/grid_torch.log" 2>&1; then
        grep -E "GRID TORCH GATE" "$TMP/grid_torch.log" | tee "$TMP/grid_torch_summary.log"
      else
        echo "GRID TORCH GATE FAILURES:"
        grep -E "FAIL|GRID TORCH GATE" "$TMP/grid_torch.log" | head -10
        fail=1
      fi
      if $LIM bash tools/cpp_port/tests/trainer_edge.sh > "$TMP/trainer_edge.log" 2>&1; then
        grep -E "TRAINER EDGE GATE" "$TMP/trainer_edge.log" \
          | tee "$TMP/trainer_edge_summary.log"
      else
        echo "TRAINER EDGE GATE FAILURES:"
        grep -E "FAIL|TRAINER EDGE GATE" "$TMP/trainer_edge.log" | head -10
        fail=1
      fi
    else
      echo "grid torch slice: SKIPPED (build distribai_torch_train first)"
    fi
  else
    echo "BUILD FAIL: native grid"; tail -5 "$TMP/grid_build.log"; fail=1
  fi
else
  echo "grid: SKIPPED (install libsqlite3-dev to build the coordinator)"
fi

# Massive test suite (suite categories). port-check gates on the sanitizer
# category being CLEAN (or explicitly skipped when the toolchain lacks
# sanitizer runtimes) and the quick suite subset being green. Full run:
# bash tools/cpp_port/tests/suite/run_suite.sh (10-min soak + differential).
echo "== port-check: test suite (quick subset + sanitizers) =="
if bash tools/cpp_port/tests/suite/run_suite.sh --quick > "$TMP/suite.log" 2>&1; then
  echo "suite quick subset: ALL GREEN"
  grep -E "SANITIZE: " "$TMP/suite.log" | tail -1 | tee "$TMP/sanitize.log"
else
  echo "SUITE FAILURES:"
  grep -E "FAILURES|FAIL\b" "$TMP/suite.log" | head -10
  sed -n '/SUITE SUMMARY/,$p' "$TMP/suite.log" | head -18
  fail=1
fi

{
  echo "# C++ Port Check Report"
  echo
  echo "Generated by tools/cpp_port/port_check.sh$QUICK under tools/bench/run_limited.sh."
  echo
  echo '```'
  grep -hE "contract checks|GREEN|PASSED" "$TMP/golden.log" "$TMP/parity.log" "$TMP/feat.log" "$TMP/abi.log"
  grep -E "train 1K|sandbox 1K" "$TMP/bench.log"
  cat "$TMP/grad.log" "$TMP/mem.log" 2>/dev/null
  grep -E "suite quick|SANITIZE:|DIFFERENTIAL" "$TMP/suite.log" "$TMP/diff.log" 2>/dev/null
  grep -E "^  (json|store|grid): " "$TMP/suite.log" 2>/dev/null | head -5
  grep -E "TORCH PARITY|passed|grid checks|grid edge|grid security|grid soak|real usage|GATE" \
    "$TMP/torch.log" "$TMP/translate_summary.log" "$TMP/gridlink_summary.log" \
    "$TMP/grid_summary.log" "$TMP/grid_torch_summary.log" "$TMP/grid_edge_summary.log" \
    "$TMP/grid_reg_summary.log" "$TMP/grid_sec_summary.log" "$TMP/grid_soak_summary.log" \
    "$TMP/usage_summary.log" "$TMP/trainer_edge_summary.log" 2>/dev/null
  echo '```'
  echo
  echo "Overall: $([ $fail -eq 0 ] && echo ALL GATES GREEN || echo FAILURES PRESENT)"
} > "$REPORT"

echo
echo "report -> $REPORT"
[ $fail -eq 0 ] && echo "PORT-CHECK: ALL GATES GREEN" || { echo "PORT-CHECK: FAILURES"; exit 1; }
