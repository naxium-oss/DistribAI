#!/usr/bin/env bash
# grid_regression.sh - regressions for bugs that actually happened.
#
# Each block names the symptom it refuses to let come back, and drives the real
# binaries. Keep the list short and specific: a regression check that does not
# map to a past failure belongs in grid_edge.sh or grid_e2e.sh instead.
#
#   1. a forked trainer died with "fork() failed" under the worker's default
#      RLIMIT_NPROC, because the limit was a constant below the uid's thread
#      count
#   2. the coordinator hung on shutdown: close() does not wake a thread blocked
#      in accept()
#   3. a worker that started before its coordinator never recovered
#   4. two workers sharing a node id knocked each other out
#   5. a job cancelled while queued still ran its replicas
#   6. credits were paid per reported result instead of per completed replica
#   7. an abandoned node's task stayed assigned forever
#
#   bash tools/cpp_port/tests/grid_regression.sh   (or: make -C tools/cpp_port grid-regression-test)
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

B=build/cpp_port
ORCH=$B/distribai_orch
WORKER=$B/distribai_worker
TRAINER=$B/grid_fake_trainer
# A random port per run, so a coordinator left over from an interrupted run can
# never answer for this one.
PORT=${GRID_REGRESSION_PORT:-$((30000 + (RANDOM % 20000)))}
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

step() { echo "== grid-regression: $*"; }
ok()   { echo "   PASS  $*"; }
bad()  { echo "   FAIL  $*"; fail=1; }

for bin in "$ORCH" "$WORKER" "$TRAINER"; do
  [ -x "$bin" ] || { echo "missing $bin (run: make -C tools/cpp_port grid-test)"; exit 1; }
done

# Refuse to run against a grid this script did not start.
for candidate in "$PORT" "$((PORT + 1))" "$((PORT + 2))" "$((PORT + 3))"; do
  if curl -fsS "http://127.0.0.1:$candidate/v1/health" >/dev/null 2>&1; then
    echo "port $candidate already answers a grid; stop it or set GRID_REGRESSION_PORT"
    exit 1
  fi
done

start_orch() { # start_orch <port> <log>
  "$ORCH" --host 127.0.0.1 --port "$1" --db "$TMP/grid.db" \
          --schema runtime/db/schema.sql --jobs-dir "$TMP/jobs" --tasks-dir "$TMP/tasks" \
          --web-dir tools/cpp_port/grid/web --invite reg-team --token reg-secret \
          --node-ttl "${2:-60}" > "$3" 2>&1 &
  PIDS+=($!)
}

wait_health() { # wait_health <port>
  for _ in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:$1/v1/health" >/dev/null 2>&1 && return 0
    sleep 0.2
  done
  return 1
}

submit() { # submit <dir> <replicas>
  "$ORCH" --db "$TMP/grid.db" --schema runtime/db/schema.sql --jobs-dir "$TMP/jobs" \
          --tasks-dir "$TMP/tasks" --submit "$1" --replicas "$2" 2>&1
}

job_status() { # job_status <port> <job_id>
  curl -fsS "http://127.0.0.1:$1/v1/jobs/$2" 2>/dev/null |
    grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4
}

# Two bundles: one for the ordinary runs, one to cancel.
for name in plain cancelled; do
  mkdir -p "$TMP/bundle-$name"
  cat > "$TMP/bundle-$name/job.json" <<JSON
{
  "job_id": "reg-$name",
  "name": "reg-$name",
  "steps": 16,
  "batch_size": 8,
  "seed": 42,
  "model": "model.pt",
  "optimizer": "adamw",
  "loss": "mse",
  "aggregate": "mean"
}
JSON
  printf 'fixture' > "$TMP/bundle-$name/model.pt"
done

# ---------------------------------------------------------------------------
step "regression 1: a trainer that forks works under the default limits"
# The old default passed RLIMIT_NPROC=512 to the child. On a box whose uid
# already had more tasks than that, every fork inside the trainer failed, and
# the replica came back as "fork() failed".
cat > "$TMP/forking_trainer.sh" <<FORK
#!/usr/bin/env bash
# Spawn a batch of children, then hand over to the real trainer.
for _ in \$(seq 1 32); do (sleep 0.05) & done
wait
exec "$(pwd)/$TRAINER" "\$@"
FORK
chmod +x "$TMP/forking_trainer.sh"

start_orch "$PORT" 60 "$TMP/orch.log"
wait_health "$PORT" && ok "coordinator is up" || { bad "coordinator never started"; exit 1; }
JOB=$(submit "$TMP/bundle-plain" 1 | sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p')
"$WORKER" --orchestrator "http://127.0.0.1:$PORT" --invite reg-team --node-id reg-fork \
          --trainer "$TMP/forking_trainer.sh" --work-dir "$TMP/wfork" --once \
          > "$TMP/fork.log" 2>&1
FORK_STATUS=$(grep -o '"status":"[^"]*"' "$TMP/wfork"/*/trainer.log 2>/dev/null | head -1 | cut -d'"' -f4)
if [ "$FORK_STATUS" = "ok" ]; then
  ok "32 forked children plus training all succeeded under the default limits"
else
  bad "the forking trainer reported '$FORK_STATUS': $(grep -o '"error":"[^"]*"' "$TMP/wfork"/*/trainer.log 2>/dev/null | head -1)"
fi
grep -q 'fork() failed' "$TMP/fork.log" 2>/dev/null && bad "the worker still reports fork() failures" ||
  ok "no fork() failure anywhere in the worker log"

# The child must see a limit above the uid's existing task count, not a constant.
cat > "$TMP/probe_trainer.sh" <<PROBE
#!/usr/bin/env bash
ulimit -u > "$TMP/ulimit_child.txt"
exec "$(pwd)/$TRAINER" "\$@"
PROBE
chmod +x "$TMP/probe_trainer.sh"
UID_TASKS=$(ps -u "$(id -u)" -L 2>/dev/null | wc -l)
submit "$TMP/bundle-plain" 1 > "$TMP/probe_submit.json" 2>&1
"$WORKER" --orchestrator "http://127.0.0.1:$PORT" --invite reg-team --node-id reg-probe \
          --trainer "$TMP/probe_trainer.sh" --work-dir "$TMP/wprobe" --once \
          > "$TMP/probe.log" 2>&1
CHILD_LIMIT=$(cat "$TMP/ulimit_child.txt" 2>/dev/null || echo 0)
if [ "${CHILD_LIMIT:-0}" -gt "$UID_TASKS" ]; then
  ok "the child limit ($CHILD_LIMIT) sits above the uid's task count ($UID_TASKS)"
else
  bad "the child limit ($CHILD_LIMIT) is at or below the uid's task count ($UID_TASKS)"
fi

# ---------------------------------------------------------------------------
step "regression 2: the coordinator stops promptly on a signal"
start_orch "$((PORT + 1))" 60 "$TMP/orch_sig.log"
wait_health "$((PORT + 1))" && ok "second coordinator is up" || bad "second coordinator never started"
SIG_PID=${PIDS[-1]}
kill -TERM "$SIG_PID" 2>/dev/null
STOPPED=0
for _ in $(seq 1 30); do
  if ! kill -0 "$SIG_PID" 2>/dev/null; then STOPPED=1; break; fi
  sleep 0.1
done
[ "$STOPPED" = "1" ] && ok "SIGTERM stops the coordinator within 3s" ||
  bad "the coordinator ignored SIGTERM (accept() was not woken)"
grep -q "stopped" "$TMP/orch_sig.log" && ok "shutdown announces itself" ||
  bad "shutdown printed nothing"

# ---------------------------------------------------------------------------
step "regression 3: a worker that starts first still joins"
# Point a worker at the port before the coordinator exists, then bring the
# coordinator up: the worker has to keep retrying registration.
"$WORKER" --orchestrator "http://127.0.0.1:$((PORT + 2))" --invite reg-team \
          --node-id reg-late --trainer "$TRAINER" --work-dir "$TMP/wlate" --once \
          > "$TMP/late.log" 2>&1 &
LATE_PID=$!
PIDS+=($LATE_PID)
sleep 1
start_orch "$((PORT + 2))" 60 "$TMP/orch_late.log"
wait_health "$((PORT + 2))" && ok "the coordinator came up late" || bad "the late coordinator failed"
submit "$TMP/bundle-plain" 1 > "$TMP/late_submit.json" 2>&1
LATE_JOB=$(sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p' "$TMP/late_submit.json")
LATE_STATUS=""
for _ in $(seq 1 100); do
  LATE_STATUS=$(job_status "$((PORT + 2))" "$LATE_JOB")
  [ "$LATE_STATUS" = "completed" ] && break
  sleep 0.3
done
grep -q "registered as reg-late" "$TMP/late.log" &&
  ok "the early worker registered once the coordinator appeared" ||
  bad "the early worker never registered: $(head -3 "$TMP/late.log")"
[ "$LATE_STATUS" = "completed" ] &&
  ok "the job submitted after the late start completed" ||
  bad "the late-start job ended as '${LATE_STATUS:-unknown}'"

# ---------------------------------------------------------------------------
step "regression 4: two workers sharing a node id both keep working"
submit "$TMP/bundle-plain" 2 > "$TMP/dup_submit.json" 2>&1
DUP_JOB=$(sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p' "$TMP/dup_submit.json")
for i in 1 2; do
  "$WORKER" --orchestrator "http://127.0.0.1:$((PORT + 2))" --invite reg-team \
            --node-id reg-twin --trainer "$TRAINER" --work-dir "$TMP/wtwin$i" --max-jobs 3 \
            > "$TMP/twin$i.log" 2>&1 &
  PIDS+=($!)
done
DUP_STATUS=""
for _ in $(seq 1 200); do
  DUP_STATUS=$(job_status "$((PORT + 2))" "$DUP_JOB")
  [ "$DUP_STATUS" = "completed" ] && break
  sleep 0.3
done
[ "$DUP_STATUS" = "completed" ] && ok "both replicas completed despite the shared node id" ||
  bad "the shared-node-id job ended as '${DUP_STATUS:-unknown}'"
grep -q "session expired\|registered as reg-twin" "$TMP/twin2.log" &&
  ok "the second worker handled the token rotation" ||
  bad "the second worker never recovered: $(head -3 "$TMP/twin2.log")"

# ---------------------------------------------------------------------------
step "regression 5: a cancelled job never runs"
CANCEL_JOB=$(submit "$TMP/bundle-cancelled" 2 | sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p')
curl -fsS -X POST -H 'X-Grid-Token: reg-secret' \
  "http://127.0.0.1:$((PORT + 2))/v1/jobs/$CANCEL_JOB/cancel" > "$TMP/cancel.json" 2>/dev/null
[ "$(job_status "$((PORT + 2))" "$CANCEL_JOB")" = "cancelled" ] &&
  ok "the queued job is cancelled" || bad "the job did not cancel"
"$WORKER" --orchestrator "http://127.0.0.1:$((PORT + 2))" --invite reg-team \
          --node-id reg-nocancel --trainer "$TRAINER" --work-dir "$TMP/wcancel" --once \
          > "$TMP/cancel_worker.log" 2>&1
sleep 0.5
if find "$TMP/tasks" -name "$CANCEL_JOB*" 2>/dev/null | grep -q .; then
  bad "the cancelled job still produced task envelopes"
else
  ok "no task envelope was written for the cancelled job"
fi
grep -q "$CANCEL_JOB" "$TMP/cancel_worker.log" 2>/dev/null &&
  bad "a worker claimed a task from the cancelled job" ||
  ok "no worker claimed work from the cancelled job"

# ---------------------------------------------------------------------------
step "regression 6: credits are paid once per completed replica"
# The ledger must hold one reward per contributing node for the two-replica
# job, and the aggregate must be the mean of the two contributions.
curl -fsS "http://127.0.0.1:$((PORT + 2))/v1/summary" > "$TMP/summary.json" 2>/dev/null
REWARDS=$(grep -o '"type":"job_reward"' "$TMP/summary.json" | wc -l)
if [ "$REWARDS" -ge 2 ]; then
  ok "the ledger holds a reward per contributing node ($REWARDS entries)"
else
  bad "expected at least 2 reward entries, found $REWARDS"
fi
curl -fsS "http://127.0.0.1:$((PORT + 2))/v1/jobs/$DUP_JOB" > "$TMP/dupjob.json" 2>/dev/null
FAILED_REPLICAS=$(grep -o '"status":"failed"' "$TMP/dupjob.json" | wc -l)
[ "$FAILED_REPLICAS" = "0" ] && ok "no replica of the two-replica job failed" ||
  bad "$FAILED_REPLICAS replica(s) failed in the two-replica job"
DUP_RESULT="$TMP/jobs/$DUP_JOB/result.json"
if grep -q '"contributors":2' "$DUP_RESULT" 2>/dev/null; then
  ok "the aggregate counts both replicas"
else
  bad "the aggregate did not count both replicas: $(cat "$DUP_RESULT" 2>/dev/null | head -c 200)"
fi
if grep -q '"aggregate":"mean"' "$DUP_RESULT" 2>/dev/null; then
  ok "the aggregate honours the job's declared method"
else
  bad "the aggregate method was not recorded"
fi
grep -q '"grad_len":16' "$DUP_RESULT" 2>/dev/null && ok "the aggregate carries 16 gradients" ||
  bad "unexpected aggregate gradient count"

# ---------------------------------------------------------------------------
step "regression 7: an abandoned node's task goes back to the queue"
start_orch "$((PORT + 3))" 2 "$TMP/orch_ttl.log"
wait_health "$((PORT + 3))" && ok "third coordinator is up (node-ttl 2s)" ||
  bad "third coordinator never started"
TTL_JOB=$(submit "$TMP/bundle-plain" 1 | sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p')
# A trainer that sleeps long enough for the node to look dead.
cat > "$TMP/slow_trainer.sh" <<SLOW
#!/usr/bin/env bash
sleep 30
exec "$(pwd)/$TRAINER" "\$@"
SLOW
chmod +x "$TMP/slow_trainer.sh"
"$WORKER" --orchestrator "http://127.0.0.1:$((PORT + 3))" --invite reg-team \
          --node-id reg-doomed --trainer "$TMP/slow_trainer.sh" --work-dir "$TMP/wdoomed" \
          > "$TMP/doomed.log" 2>&1 &
DOOMED=$!
sleep 2
kill -9 "$DOOMED" 2>/dev/null
grep -q "task $TTL_JOB-r0" "$TMP/doomed.log" && ok "the doomed worker claimed the task" ||
  bad "the doomed worker never claimed its task"
REQUEUED=0
for _ in $(seq 1 80); do
  if curl -fsS "http://127.0.0.1:$((PORT + 3))/v1/tasks" 2>/dev/null | grep -q '"status":"queued"'; then
    REQUEUED=1
    break
  fi
  sleep 0.5
done
[ "$REQUEUED" = "1" ] && ok "the abandoned task returned to the queue" ||
  bad "the abandoned task stayed assigned"
# And a fresh worker finishes it.
"$WORKER" --orchestrator "http://127.0.0.1:$((PORT + 3))" --invite reg-team \
          --node-id reg-takeover --trainer "$TRAINER" --work-dir "$TMP/wtakeover" --once \
          > "$TMP/takeover.log" 2>&1
TTL_STATUS=""
for _ in $(seq 1 60); do
  TTL_STATUS=$(job_status "$((PORT + 3))" "$TTL_JOB")
  [ "$TTL_STATUS" = "completed" ] && break
  sleep 0.3
done
[ "$TTL_STATUS" = "completed" ] && ok "a second worker completed the recovered task" ||
  bad "the recovered task ended as '${TTL_STATUS:-unknown}'"

echo
if [ "$fail" -eq 0 ]; then
  echo "GRID REGRESSION GATE: PASS"
else
  echo "GRID REGRESSION GATE: FAILURES"
  echo "--- coordinator log ---"
  tail -20 "$TMP/orch.log"
fi
exit "$fail"
