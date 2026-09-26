#!/usr/bin/env bash
# demo_features.sh - end-to-end demo of the four new capabilities.
# Run from repo root: tools/cpp_port/demos/demo_features.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
ZIG=".toolchain/zig-x86_64-linux-0.16.0/zig c++"
# build commands run via zig c++ (name contains a space, so eval or array)
zbuild() { $ZIG "$@"; }
B=build/cpp_port
LIM="tools/bench/run_limited.sh"

echo "== 0. build =="
$ZIG tools/cpp_port/apps/dai_job.cpp tools/cpp_port/core/envelope.cpp -std=c++17 -O2 -o $B/dai_job 2>/dev/null
$ZIG tools/cpp_port/tests/test_features.cpp tools/cpp_port/core/envelope.cpp -std=c++17 -O2 -o $B/test_features 2>/dev/null
$ZIG -shared tools/cpp_port/abi/dai_abi.cpp tools/cpp_port/core/envelope.cpp -std=c++17 -O2 -fPIC -o $B/libdai_core.so 2>/dev/null
cc tools/cpp_port/tests/abi_consumer.c -Itools/cpp_port/abi -o /tmp/abi_consumer -ldl
echo "built: dai_job, test_features, libdai_core.so"

echo
echo "== 1. envelope schema + checkpoint + multi-model feature tests =="
$LIM $B/test_features | tail -2

echo
echo "== 2. job manifest CLI (Feature 3) =="
cat > /tmp/demo_job.json <<'EOF'
{
  "job_id": "demo-001",
  "models": [
    {"name": "alpha", "seed": 42, "steps": 200},
    {"name": "beta",  "seed": 43, "steps": 200},
    {"name": "gamma", "seed": 44, "steps": 200}
  ],
  "sandbox_count": 2,
  "rlimits": {"mem_mb": 64, "cpu_sec": 120},
  "aggregate": "trimmed_mean"
}
EOF
echo "-- dry run --"
$LIM $B/dai_job --manifest /tmp/demo_job.json --dry-run
echo "-- live run (3 models, 2 concurrent sandboxes) --"
$LIM $B/dai_job --manifest /tmp/demo_job.json | tee /tmp/demo_job_result.json

echo
echo "== 3. checkpoint/resume bit-exact proof (Feature 2, inside feature tests) =="
echo "(covered by test_features above: resumed losses + weights match uncrashed run exactly)"

echo
echo "== 4. C ABI dlopen consumer (Feature 4) =="
/tmp/abi_consumer $B/libdai_core.so | tail -4

echo
echo "ALL FEATURE DEMOS COMPLETE"
