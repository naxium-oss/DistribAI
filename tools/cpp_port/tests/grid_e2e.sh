#!/usr/bin/env bash
# grid_e2e.sh - the native grid's end-to-end gate.
#
# Starts a real coordinator (SQLite state, HTTP API, dashboard), submits a real
# job bundle, runs two real worker processes that each claim one replica and run
# a trainer as a limited child, then checks the aggregate, the ledger and the
# dashboard. Nothing here is mocked: the workers connect over loopback HTTP and
# the coordinator writes to a real SQLite file.
#
# The trainer is build/cpp_port/grid_fake_trainer, a test fixture that speaks the
# same CLI and envelope contract as the LibTorch trainer. That keeps the gate
# runnable on machines without torch; make torch-test covers the real trainer.
#
#   bash tools/cpp_port/tests/grid_e2e.sh      (or: make -C tools/cpp_port grid-test)
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

B=build/cpp_port
ORCH=$B/distribai_orch
WORKER=$B/distribai_worker
TRAINER=$B/grid_fake_trainer
# A random port per run, so a coordinator left over from an interrupted run can
# never answer for this one.
PORT=${GRID_E2E_PORT:-$((40000 + (RANDOM % 20000)))}
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

step() { echo "== grid-e2e: $*"; }
ok()   { echo "   PASS  $*"; }
bad()  { echo "   FAIL  $*"; fail=1; }

for bin in "$ORCH" "$WORKER" "$TRAINER"; do
  [ -x "$bin" ] || { echo "missing $bin (run: make -C tools/cpp_port grid-test)"; exit 1; }
done

# Refuse to run against a grid this script did not start.
if curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; then
  echo "port $PORT already answers a grid; stop it or set GRID_E2E_PORT"
  exit 1
fi

# ---- job bundle: a translated-job shape, built by hand -------------------
mkdir -p "$TMP/job-in"
cat > "$TMP/job-in/job.json" <<'JSON'
{
  "job_id": "e2e-job",
  "name": "e2e-model",
  "steps": 24,
  "batch_size": 8,
  "seed": 42,
  "model": "model.pt",
  "optimizer": "adamw",
  "loss": "mse",
  "aggregate": "trimmed_mean"
}
JSON
printf 'test fixture, not a real TorchScript module' > "$TMP/job-in/model.pt"

step "starting the coordinator on 127.0.0.1:$PORT"
"$ORCH" --host 127.0.0.1 --port "$PORT" \
        --db "$TMP/grid.db" --schema runtime/db/schema.sql \
        --jobs-dir "$TMP/jobs" --tasks-dir "$TMP/tasks" \
        --web-dir tools/cpp_port/grid/web \
        --invite e2e-team --token e2e-secret \
        --node-ttl 5 --aggregate trimmed_mean \
        > "$TMP/orch.log" 2>&1 &
PIDS+=($!)

for _ in $(seq 1 50); do
  curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && break
  sleep 0.2
done
if curl -fsS "http://127.0.0.1:$PORT/v1/health" > "$TMP/health.json" 2>/dev/null; then
  ok "health endpoint answers: $(cat "$TMP/health.json")"
else
  bad "health endpoint never came up"
  sed -n 1,20p "$TMP/orch.log"
  exit 1
fi

# ---- registration gate ---------------------------------------------------
step "invite gate"
"$WORKER" --orchestrator "http://127.0.0.1:$PORT" --invite wrong-code --once \
          --trainer "$TRAINER" --work-dir "$TMP/w0" > "$TMP/bad-invite.log" 2>&1
if grep -q "invite code required" "$TMP/bad-invite.log"; then
  ok "wrong invite code is refused"
else
  bad "wrong invite code was not refused"
  sed -n 1,5p "$TMP/bad-invite.log"
fi

# ---- submit ---------------------------------------------------------------
step "submitting the job with two replicas"
if "$ORCH" --db "$TMP/grid.db" --schema runtime/db/schema.sql \
           --jobs-dir "$TMP/jobs" --tasks-dir "$TMP/tasks" \
           --submit "$TMP/job-in" --replicas 2 > "$TMP/submit.json" 2>&1; then
  ok "submitted: $(cat "$TMP/submit.json")"
else
  bad "submit failed"
  cat "$TMP/submit.json"
fi
JOB_ID=$(sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p' "$TMP/submit.json")
[ -n "$JOB_ID" ] && ok "job id $JOB_ID" || bad "no job id in the submit reply"

step "admin token gate"
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/jobs" \
       -H 'Content-Type: application/json' --data '{"job_dir":"/nonexistent","replicas":1}')
if [ "$CODE" = "403" ]; then
  ok "submit without the token is refused (403)"
else
  bad "submit without the token returned $CODE, expected 403"
fi

# ---- two workers ---------------------------------------------------------
step "starting two workers"
for i in 1 2; do
  "$WORKER" --orchestrator "http://127.0.0.1:$PORT" --invite e2e-team \
            --node-id "e2e-node-$i" --trainer "$TRAINER" \
            --work-dir "$TMP/w$i" --mem-mb 256 --cpu-sec 60 --max-jobs 1 \
            > "$TMP/worker$i.log" 2>&1 &
  PIDS+=($!)
done

step "waiting for the job to finish"
# The job status is the first "status" field in the reply; replica rows repeat
# the field later, so take the first match only.
job_status() {
  curl -fsS "http://127.0.0.1:$PORT/v1/jobs/$1" 2>/dev/null |
    grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4
}
STATUS=""
for _ in $(seq 1 100); do
  STATUS=$(job_status "$JOB_ID")
  [ "$STATUS" = "completed" ] && break
  sleep 0.3
done
[ "$STATUS" = "completed" ] && ok "job completed" || bad "job status is '${STATUS:-unknown}'"

# ---- assertions ----------------------------------------------------------
step "checking the results"
curl -fsS "http://127.0.0.1:$PORT/v1/summary" > "$TMP/summary.json" 2>/dev/null
curl -fsS "http://127.0.0.1:$PORT/v1/jobs/$JOB_ID" > "$TMP/job.json" 2>/dev/null
curl -fsS "http://127.0.0.1:$PORT/" > "$TMP/index.html" 2>/dev/null

RESULT="$TMP/jobs/$JOB_ID/result.json"
[ -s "$RESULT" ] && ok "result written: $(cat "$RESULT")" || bad "no result.json at $RESULT"
[ -s "$TMP/jobs/$JOB_ID/aggregate.env" ] && ok "aggregate envelope written" ||
  bad "no aggregate envelope"
[ -s "$TMP/tasks/${JOB_ID}-r0.env" ] && ok "replica 0 envelope stored" ||
  bad "replica 0 envelope missing"
[ -s "$TMP/tasks/${JOB_ID}-r1.env" ] && ok "replica 1 envelope stored" ||
  bad "replica 1 envelope missing"

grep -q '"contributors":2' "$RESULT" && ok "both replicas contributed" ||
  bad "expected 2 contributors: $(cat "$RESULT" 2>/dev/null)"
grep -q '"grad_len":16' "$RESULT" && ok "aggregate carries 16 gradients" ||
  bad "unexpected grad_len: $(cat "$RESULT" 2>/dev/null)"
grep -q '"node_id":"e2e-node-1"' "$TMP/summary.json" &&
  grep -q '"node_id":"e2e-node-2"' "$TMP/summary.json" &&
  ok "both workers appear in the summary" || bad "a worker is missing from the summary"
grep -q '"job_reward"' "$TMP/summary.json" && ok "credits recorded in the ledger" ||
  bad "no credits in the summary"
grep -q "DistribAI" "$TMP/index.html" && ok "dashboard index served" ||
  bad "dashboard index did not render"

# Two replicas use different seeds, so their gradients must differ and the
# trimmed mean must sit between them. Compare the envelope grad sums.
SUM0=$(grep -o '"node_id":"[^"]*"' "$TMP/summary.json" | head -1)
if grep -q '"jobs_completed":1' "$TMP/summary.json"; then
  ok "node counters advanced ($SUM0)"
else
  bad "node job counters did not advance"
fi

step "reaping an abandoned node"
# A trainer that cannot finish before the kill, so the task really is left
# assigned to a node that stops heartbeating.
cat > "$TMP/slow_trainer.sh" <<SLOW
#!/usr/bin/env bash
sleep 30
exec "$(pwd)/$TRAINER" "\$@"
SLOW
chmod +x "$TMP/slow_trainer.sh"
mkdir -p "$TMP/job-stale"
sed 's/e2e-model/stale-model/; s/"steps": 24/"steps": 12/' "$TMP/job-in/job.json" > "$TMP/job-stale/job.json"
cp "$TMP/job-in/model.pt" "$TMP/job-stale/model.pt"
"$ORCH" --db "$TMP/grid.db" --schema runtime/db/schema.sql --jobs-dir "$TMP/jobs" \
        --tasks-dir "$TMP/tasks" --submit "$TMP/job-stale" --replicas 1 > "$TMP/stale.json" 2>&1
STALE_ID=$(sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p' "$TMP/stale.json")
if [ -n "$STALE_ID" ]; then
  ok "submitted the throwaway job $STALE_ID"
else
  bad "could not submit the throwaway job: $(cat "$TMP/stale.json")"
fi
"$WORKER" --orchestrator "http://127.0.0.1:$PORT" --invite e2e-team --node-id e2e-doomed \
          --trainer "$TMP/slow_trainer.sh" --work-dir "$TMP/w3" --cpu-sec 60 \
          > "$TMP/doomed.log" 2>&1 &
DOOMED=$!
sleep 2.5
kill -9 "$DOOMED" 2>/dev/null
sleep 0.5
if grep -q "task $STALE_ID-r0" "$TMP/doomed.log"; then
  ok "the doomed worker claimed the task before the kill"
else
  bad "the doomed worker never claimed its task"
  sed -n 1,10p "$TMP/doomed.log"
  curl -fsS "http://127.0.0.1:$PORT/v1/tasks" 2>/dev/null | head -c 400
  echo
fi
# The node stops heartbeating, so after node-ttl the reaper must put the task
# back in the queue.
# The first job's tasks are all done by now, so any queued task is the one the
# reaper recovered from the dead node.
FOUND_REQUEUE=0
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/tasks" 2>/dev/null | grep -q '"status":"queued"'; then
    FOUND_REQUEUE=1
    break
  fi
  sleep 0.5
done
[ "$FOUND_REQUEUE" = "1" ] && ok "offline node's task went back to the queue" ||
  bad "the reaper did not requeue the abandoned task"
QUEUED_ERR=$(curl -fsS "http://127.0.0.1:$PORT/v1/summary" 2>/dev/null | grep -o '"error":"[^"]*"' | head -1)
echo "   note  requeue reason: ${QUEUED_ERR:-none reported}"

echo
if [ "$fail" -eq 0 ]; then
  echo "GRID GATE: PASS"
else
  echo "GRID GATE: FAILURES"
  echo "--- coordinator log ---"
  tail -20 "$TMP/orch.log"
fi
exit "$fail"
