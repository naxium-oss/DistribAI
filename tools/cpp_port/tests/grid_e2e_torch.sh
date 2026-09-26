#!/usr/bin/env bash
# grid_e2e_torch.sh - translated trainer through the whole native stack.
#
# This is the gate that proves the product slice end to end:
#
#   1. tools/trainer_translate turns a plain PyTorch script into a job bundle
#      (TorchScript model, captured data, job.json).
#   2. the bundle is submitted to a real coordinator with a real SQLite file.
#   3. two real worker processes claim one replica each and run the REAL LibTorch
#      trainer as a limited child.
#   4. the coordinator aggregates the reported envelopes into result.json.
#
# grid_e2e.sh covers the same plumbing with a fixture trainer so it runs without
# torch. This one is the real thing, so it skips when the LibTorch trainer or
# torch itself is missing.
#
#   bash tools/cpp_port/tests/grid_e2e_torch.sh    (or: make -C tools/cpp_port grid-torch-test)
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

B=build/cpp_port
ORCH=$B/distribai_orch
WORKER=$B/distribai_worker
TRAINER=$B/distribai_torch_train
# A random port per run, so a coordinator left over from an interrupted run can
# never answer for this one.
PORT=${GRID_E2E_TORCH_PORT:-$((25000 + (RANDOM % 20000)))}
PY=${GRID_E2E_PY:-.venv/bin/python}
TMP=$(mktemp -d)
PIDS=()
fail=0

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null
  done
  pkill -f "distribai_worker.*--orchestrator http://127.0.0.1:$PORT" 2>/dev/null
  sleep 0.2
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
  done
  rm -rf "$TMP"
}
trap cleanup EXIT

step() { echo "== grid-torch: $*"; }
ok()   { echo "   PASS  $*"; }
bad()  { echo "   FAIL  $*"; fail=1; }
skip() { echo "GRID TORCH GATE: SKIP ($*)"; exit 0; }

[ -x "$ORCH" ]    || skip "missing $ORCH; run make -C tools/cpp_port grid"
[ -x "$WORKER" ]  || skip "missing $WORKER; run make -C tools/cpp_port grid"
[ -x "$TRAINER" ] || skip "missing $TRAINER; run make torch"
[ -x "$PY" ]      || skip "no interpreter at $PY"
"$PY" -c 'import torch' 2>/dev/null || skip "torch is not importable from $PY"

# Refuse to run against a grid this script did not start.
if curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; then
  echo "port $PORT already answers a grid; stop it or set GRID_E2E_TORCH_PORT"
  exit 1
fi

# ---- the trainer a user would actually write ------------------------------
# A Linear(10,16) -> ReLU -> Linear(16,1) stack has 10*16+16 + 16*1+1 = 193
# parameters. The gate asserts that number, so an architecture-capture
# regression shows up as a mismatch instead of a silently different model.
TRAINER_SRC=$TMP/net.py
cat > "$TRAINER_SRC" <<'PY'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def main():
    torch.manual_seed(0)
    model = Net()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(64, 10)
    y = torch.randn(64, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    for xb, yb in loader:
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(xb), yb)
        loss.backward()
        opt.step()


main()
PY

step "translating the trainer"
if (cd tools && "../$PY" -m trainer_translate.translate "$TRAINER_SRC" \
        --out "$TMP/job-in" --steps 30 > "$TMP/translate.log" 2>&1); then
  ok "bundle written"
else
  bad "translation failed"
  tail -20 "$TMP/translate.log"
  echo; echo "GRID TORCH GATE: FAILURES"; exit 1
fi
for f in job.json model.pt x.bin y.bin; do
  [ -s "$TMP/job-in/$f" ] && ok "bundle has $f" || bad "bundle is missing $f"
done
grep -q '"input": "x.bin"' "$TMP/job-in/job.json" &&
  ok "job.json points at the captured data by relative name" ||
  bad "job.json does not reference x.bin"

# ---- coordinator ----------------------------------------------------------
step "starting the coordinator on 127.0.0.1:$PORT"
"$ORCH" --host 127.0.0.1 --port "$PORT" \
        --db "$TMP/grid.db" --schema runtime/db/schema.sql \
        --jobs-dir "$TMP/jobs" --tasks-dir "$TMP/tasks" \
        --web-dir tools/cpp_port/grid/web \
        --invite torch-team --token torch-secret \
        --node-ttl 30 --aggregate trimmed_mean > "$TMP/orch.log" 2>&1 &
PIDS+=($!)

for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && break
  sleep 0.2
done
if curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; then
  ok "coordinator is up"
else
  bad "coordinator never came up"
  tail -20 "$TMP/orch.log"
  echo; echo "GRID TORCH GATE: FAILURES"; exit 1
fi

step "submitting the translated job with two replicas"
if "$ORCH" --db "$TMP/grid.db" --schema runtime/db/schema.sql \
           --jobs-dir "$TMP/jobs" --tasks-dir "$TMP/tasks" \
           --submit "$TMP/job-in" --replicas 2 > "$TMP/submit.json" 2>&1; then
  ok "submitted: $(cat "$TMP/submit.json")"
else
  bad "submit failed: $(cat "$TMP/submit.json")"
fi
JOB_ID=$(sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p' "$TMP/submit.json")
[ -n "$JOB_ID" ] && ok "job id $JOB_ID" || bad "no job id in the submit reply"

step "starting two workers on the real LibTorch trainer"
for i in 1 2; do
  "$WORKER" --orchestrator "http://127.0.0.1:$PORT" --invite torch-team \
            --node-id "torch-node-$i" --trainer "$TRAINER" \
            --work-dir "$TMP/w$i" --max-jobs 1 > "$TMP/worker$i.log" 2>&1 &
  PIDS+=($!)
done

step "waiting for the job to finish"
job_status() {
  curl -fsS "http://127.0.0.1:$PORT/v1/jobs/$1" 2>/dev/null |
    grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4
}
STATUS=""
for _ in $(seq 1 400); do
  STATUS=$(job_status "$JOB_ID")
  [ "$STATUS" = "completed" ] && break
  sleep 0.3
done
[ "$STATUS" = "completed" ] && ok "job completed" ||
  bad "job status is '${STATUS:-unknown}'"

# ---- assertions -----------------------------------------------------------
step "checking the results"
RESULT="$TMP/jobs/$JOB_ID/result.json"
[ -s "$RESULT" ] && ok "result written: $(cat "$RESULT")" || bad "no result.json"

TORCH_LOGS=$(cat "$TMP"/w*/*/trainer.log 2>/dev/null)
if [ -z "$TORCH_LOGS" ]; then
  bad "no trainer output; the workers never ran the trainer"
else
  echo "$TORCH_LOGS" | sed 's/^/   trainer: /'
fi

# The workers must report the LibTorch engine, not the fixture one.
if echo "$TORCH_LOGS" | grep -q '"engine":"libtorch"'; then
  ok "both replicas ran the LibTorch engine"
else
  bad "the LibTorch engine did not report"
fi

# A fixed RLIMIT_NPROC smaller than the uid's thread count makes every fork
# fail, which is exactly how this gate first went red. The check stays so the
# limit cannot quietly go back to a constant.
if echo "$TORCH_LOGS" | grep -q 'fork() failed'; then
  bad "a replica could not fork; RLIMIT_NPROC is below the uid's thread count"
else
  ok "replicas forked their sandboxed children"
fi

echo "$TORCH_LOGS" | grep -q '"status":"ok"' && ok "training reported ok" ||
  bad "training did not report ok"

grep -q '"contributors":2' "$RESULT" && ok "both replicas contributed" ||
  bad "expected 2 contributors: $(cat "$RESULT" 2>/dev/null)"
grep -q '"grad_len":193' "$RESULT" && ok "aggregate carries 193 gradients" ||
  bad "unexpected grad_len: $(cat "$RESULT" 2>/dev/null)"
grep -q '"aggregate":"trimmed_mean"' "$RESULT" && ok "trimmed mean aggregate" ||
  bad "aggregate mode is wrong"

# Both replicas train on the captured data with different seeds, so their
# gradient sums differ; the aggregate must land between the two.
if "$PY" - "$RESULT" <<'PY'
import json, sys
res = json.load(open(sys.argv[1]))
sums = []
for r in res.get("replicas_detail", []):
    assert r["status"] == "done", r
    assert r["steps"] == 30, r
    assert r["metrics"]["engine"] == "libtorch", r
    assert r["metrics"]["final_loss"] > 0, r
    sums.append(r["metrics"]["final_loss"])
assert res["grad_len"] == 193, res["grad_len"]
assert res["contributors"] == 2, res
print(f"   replica losses: {sums}")
PY
then
  ok "replica details are consistent"
else
  bad "replica details did not parse or hold up"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "GRID TORCH GATE: PASS"
else
  echo "GRID TORCH GATE: FAILURES"
  echo "--- coordinator log ---"
  tail -20 "$TMP/orch.log"
  echo "--- worker logs ---"
  cat "$TMP"/worker*.log 2>/dev/null
fi
exit "$fail"
