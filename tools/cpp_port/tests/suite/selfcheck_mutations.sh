#!/usr/bin/env bash
# selfcheck_mutations.sh - proves the suite catches real bugs (red/green).
# Injects three bugs one at a time (stashed copies, never lost), expects the
# targeted category to FAIL for each, restores the originals, and expects
# green. Exit 0 only if every mutation was caught AND the restored tree is
# green. Full transcript: runtime/baselines/mutation_log.txt
set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 1

S=tools/cpp_port/tests/suite/categories
B=build/cpp_port
ZIG=".toolchain/zig-x86_64-linux-0.16.0/zig c++"
LOG=runtime/baselines/mutation_log.txt
LIM=tools/bench/run_limited.sh

mkdir -p runtime/baselines
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export ZIG_LOCAL_CACHE_DIR="$TMP/zig-local"
export ZIG_GLOBAL_CACHE_DIR="$TMP/zig-global"

log() { echo "$@" | tee -a "$LOG"; }
: > "$LOG"

zbuild() {
  $ZIG -std=c++17 -O2 -Itools/cpp_port tools/cpp_port/tests/suite/categories/"$1" \
    tools/cpp_port/core/workspace.cpp tools/cpp_port/core/envelope.cpp \
    tools/cpp_port/abi/dai_abi.cpp -o "$B/$2" 2>"$TMP/berr" || { echo "BUILD FAIL"; sed -n 1,5p "$TMP/berr"; return 1; }
}

category_result() { # <bin> <cat> -> prints PASS|FAIL
  if $LIM "$B/$1" --category "$2" > "$TMP/out.log" 2>&1; then echo PASS; else echo FAIL; fi
}

failures=0

# --- Mutation 1 (core): AdamW bias-correction off by one ----------------
cp tools/cpp_port/core/tensor.hpp "$TMP/tensor.hpp.bak"
log "MUTATION 1: AdamW t_ pre-increment removed (bias correction off by one)"
python3 - <<'EOF'
import re
p = 'tools/cpp_port/core/tensor.hpp'
s = open(p).read()
# move ++t_ after the bc1 computation so t_ used is one less
s = s.replace("""  void step() {
    ++t_;
    // Step-level constants hoisted""", """  void step() {
    // Step-level constants hoisted""", 1)
s = s.replace("""    const float om_b1 = 1.0f - b1_;
    const float om_b2 = 1.0f - b2_;
    // The wd_ != 0 check""", """    ++t_;
    const float om_b1 = 1.0f - b1_;
    const float om_b2 = 1.0f - b2_;
    // The wd_ != 0 check""", 1)
open(p, 'w').write(s)
EOF
if zbuild suite_core.cpp suite_core && [ "$(category_result suite_core edge)" = "FAIL" ]; then
  log "  caught by: edge category (adamw first-step magnitude / determinism)  -> RED OK"
else
  log "  NOT CAUGHT -> mutation test FAILED"; failures=$((failures+1))
fi
cp "$TMP/tensor.hpp.bak" tools/cpp_port/core/tensor.hpp

# --- Mutation 2 (sandbox): drop RLIMIT_NPROC application ----------------
cp tools/cpp_port/sandbox/sandbox.hpp "$TMP/sandbox.hpp.bak"
log "MUTATION 2: sandbox drops RLIMIT_NPROC from applied limits"
python3 - <<'EOF'
p = 'tools/cpp_port/sandbox/sandbox.hpp'
s = open(p).read()
s = s.replace("""  rl.rlim_cur = nproc; rl.rlim_max = nproc;
  setrlimit(RLIMIT_NPROC, &rl);
""", "", 1)
open(p, 'w').write(s)
EOF
if zbuild suite_sandbox.cpp suite_sandbox && [ "$(category_result suite_sandbox security)" = "FAIL" ]; then
  log "  caught by: security category (nproc cap enforced inside own-system child)  -> RED OK"
else
  log "  NOT CAUGHT -> mutation test FAILED"; failures=$((failures+1))
fi
cp "$TMP/sandbox.hpp.bak" tools/cpp_port/sandbox/sandbox.hpp

# --- Mutation 3 (envelope): CRC verification skipped --------------------
cp tools/cpp_port/core/envelope.cpp "$TMP/envelope.cpp.bak"
log "MUTATION 3: envelope decode accepts any CRC (verification disabled)"
python3 - <<'EOF'
p = 'tools/cpp_port/core/envelope.cpp'
s = open(p).read()
# neutralize the check while keeping the code compiling: the computed CRC is
# compared against itself, so every corruption verifies
count = s.count('crc32_ieee(')
assert count >= 2, count
s = s.replace('crc32_ieee(', 'crc32_ieee_neutralized(')
# add a pass-through shim so the build stays valid
s = s.replace('uint32_t crc32_ieee_neutralized(', 'uint32_t crc32_ieee_neutralized_real_unused(', 1)
shim = '''static uint32_t crc32_ieee_neutralized(const uint8_t* data, size_t n, uint32_t crc = 0xFFFFFFFFu) {
  (void)data; (void)n; (void)crc;
  return 0u;  // mutation: "CRC always matches"
}
'''
# insert shim right after the first include block
s = s.replace('namespace distribai {', shim + '\nnamespace distribai {', 1)
open(p, 'w').write(s)
EOF
if zbuild suite_sandbox.cpp suite_sandbox && [ "$(category_result suite_sandbox adversarial)" = "FAIL" ]; then
  log "  caught by: adversarial category (single-byte corruption must be rejected)  -> RED OK"
else
  log "  NOT CAUGHT -> mutation test FAILED"; failures=$((failures+1))
fi
cp "$TMP/envelope.cpp.bak" tools/cpp_port/core/envelope.cpp

# --- restore + green check ----------------------------------------------
log "RESTORED originals; verifying green..."
zbuild suite_core.cpp suite_core && zbuild suite_sandbox.cpp suite_sandbox || failures=$((failures+1))
for c in "suite_core edge" "suite_core safety" "suite_sandbox security" "suite_sandbox adversarial" "suite_sandbox redteam"; do
  set -- $c
  r=$(category_result "$1" "$2")
  log "  $1 $2: $r"
  [ "$r" = "PASS" ] || failures=$((failures+1))
done

if [ $failures = 0 ]; then
  log "MUTATION SELFCHECK: ALL 3 BUGS CAUGHT, TREE RESTORED GREEN"
  exit 0
fi
log "MUTATION SELFCHECK: $failures PROBLEMS"
exit 1
