#!/usr/bin/env bash
# real_usage.sh - one operator's day against the native grid.
#
# grid_e2e.sh and grid_soak.sh check a single happy path and raw load. This gate
# walks the whole lifecycle an operator actually goes through, on real binaries:
#
#   1. bring a coordinator up, register a worker, run a two-replica job
#   2. add a second worker and run more work
#   3. cancel a job while a worker is training it
#   4. stop the coordinator and start it again on the same database
#   5. submit more work after the restart
#   6. read the operator surfaces: dashboard, static asset, summary JSON,
#      job detail, aggregate envelope, leaderboard
#
# Nothing is mocked: real coordinator, real workers, real SQLite, real HTTP.
# The trainer is the fixture that speaks the LibTorch envelope contract.
#
#   bash tools/cpp_port/tests/real_usage.sh   (or: make -C tools/cpp_port usage-test)
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

B=build/cpp_port
ORCH=$B/distribai_orch
WORKER=$B/distribai_worker
TRAINER=$B/grid_fake_trainer
PORT=${GRID_USAGE_PORT:-$((30000 + (RANDOM % 20000)))}
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

step() { echo "== real-usage: $*"; }
ok()   { echo "   PASS  $*"; }
bad()  { echo "   FAIL  $*"; fail=1; }

for bin in "$ORCH" "$WORKER" "$TRAINER"; do
  [ -x "$bin" ] || { echo "missing $bin (run: make -C tools/cpp_port grid-test)"; exit 1; }
done
if curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; then
  echo "port $PORT already answers a grid; stop it or set GRID_USAGE_PORT"
  exit 1
fi

job_status() { # job_status <job_id>
  curl -fsS "http://127.0.0.1:$PORT/v1/jobs/$1" 2>/dev/null |
    grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4
}
wait_status() { # wait_status <job_id> <status> <tries>
  local i
  for i in $(seq 1 "$3"); do
    [ "$(job_status "$1")" = "$2" ] && return 0
    sleep 0.25
  done
  return 1
}
nodes_online() {
  curl -fsS "http://127.0.0.1:$PORT/v1/health" 2>/dev/null |
    grep -o '"nodes_online":[0-9]*' | cut -d: -f2
}
start_orch() { # start_orch <log>
  "$ORCH" --host 127.0.0.1 --port "$PORT" \
          --db "$TMP/usage.db" --schema runtime/db/schema.sql \
          --jobs-dir "$TMP/jobs" --tasks-dir "$TMP/tasks" \
          --web-dir tools/cpp_port/grid/web \
          --invite usage-team --token usage-secret \
          --node-ttl 30 --aggregate trimmed_mean > "$1" 2>&1 &
  ORCH_PID=$!
  PIDS+=("$ORCH_PID")
}
start_worker() { # start_worker <node-id> <trainer> <work-dir> [log]
  "$WORKER" --orchestrator "http://127.0.0.1:$PORT" --invite usage-team \
            --node-id "$1" --trainer "$2" --work-dir "$3" \
            --mem-mb 256 --cpu-sec 60 --max-jobs 50 > "${4:-$TMP/worker-$1.log}" 2>&1 &
  PIDS+=($!)
}
wait_health() {
  local i
  for i in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && return 0
    sleep 0.2
  done
  return 1
}
stop_worker() { # stop_worker <node-id>
  pkill -f "distribai_worker.*--node-id $1" 2>/dev/null
}
submit_job() { # submit_job <dir> <replicas> -> prints the job id
  "$ORCH" --db "$TMP/usage.db" --schema runtime/db/schema.sql \
          --jobs-dir "$TMP/jobs" --tasks-dir "$TMP/tasks" \
          --submit "$1" --replicas "$2" 2>&1 |
    sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p'
}
mk_bundle() { # mk_bundle <dir> <id> <steps>
  mkdir -p "$1"
  cat > "$1/job.json" <<JSON
{
  "job_id": "$2",
  "name": "$2",
  "steps": $3,
  "batch_size": 4,
  "seed": 7,
  "model": "model.pt",
  "optimizer": "adamw",
  "loss": "mse",
  "aggregate": "mean"
}
JSON
  printf 'fixture' > "$1/model.pt"
}

# ---------------------------------------------------------------------------
step "bringing the grid up"
start_orch "$TMP/orch.log"
wait_health && ok "coordinator is up on 127.0.0.1:$PORT" ||
  { bad "coordinator never started"; sed -n 1,20p "$TMP/orch.log"; exit 1; }

start_worker usage-a "$TRAINER" "$TMP/wa"
for _ in $(seq 1 60); do
  [ "$(nodes_online)" -ge 1 ] && break
  sleep 0.2
done
[ "$(nodes_online)" -ge 1 ] && ok "the first worker registered" || bad "worker A never registered"

# ---------------------------------------------------------------------------
step "running the first job"
mk_bundle "$TMP/job1" "usage-job-1" 12
JOB1=$(submit_job "$TMP/job1" 2)
[ -n "$JOB1" ] && ok "job 1 accepted as $JOB1" || bad "job 1 was not accepted"
wait_status "$JOB1" completed 200 && ok "job 1 completed" ||
  bad "job 1 ended as '$(job_status "$JOB1")'"

RESULT1="$TMP/jobs/$JOB1/result.json"
[ -s "$RESULT1" ] && ok "job 1 wrote result.json" || bad "no result.json for job 1"
grep -q '"contributors":2' "$RESULT1" && ok "job 1 counted both replicas" ||
  bad "job 1 did not count both replicas"
DONE_ROWS=$(grep -o '"status":"done"' "$RESULT1" 2>/dev/null | wc -l | tr -d ' ')
[ "$DONE_ROWS" = "2" ] && ok "job 1 detail lists two done replicas" ||
  bad "job 1 detail lists $DONE_ROWS done replicas"
head -c 4 "$TMP/jobs/$JOB1/aggregate.env" 2>/dev/null | grep -q "NEID" &&
  ok "the aggregate envelope carries its magic" ||
  bad "the aggregate envelope is missing or malformed"

curl -fsS "http://127.0.0.1:$PORT/v1/summary" > "$TMP/summary1.json" 2>/dev/null
grep -q '"type":"job_reward"' "$TMP/summary1.json" &&
  ok "the ledger recorded the rewards" || bad "no job_reward in the ledger"
A_COMPLETED=$(grep -o "\"node_id\":\"usage-a\"[^}]*\"jobs_completed\":[0-9]*" "$TMP/summary1.json" |
                grep -o '"jobs_completed":[0-9]*' | head -1 | cut -d: -f2)
[ "${A_COMPLETED:-0}" -ge 2 ] && ok "worker A counted ${A_COMPLETED:-0} completed tasks" ||
  bad "worker A counted ${A_COMPLETED:-0} completed tasks"

# ---------------------------------------------------------------------------
step "a second worker joins"
start_worker usage-b "$TRAINER" "$TMP/wb"
for _ in $(seq 1 60); do
  [ "$(nodes_online)" -ge 2 ] && break
  sleep 0.2
done
[ "$(nodes_online)" -ge 2 ] && ok "both workers are online" ||
  bad "only $(nodes_online) worker is online"

mk_bundle "$TMP/job2" "usage-job-2" 10
JOB2=$(submit_job "$TMP/job2" 4)
wait_status "$JOB2" completed 300 && ok "a wider job completed" ||
  bad "job 2 ended as '$(job_status "$JOB2")'"
grep -q '"contributors":4' "$TMP/jobs/$JOB2/result.json" 2>/dev/null &&
  ok "the wider job counted all four replicas" ||
  bad "job 2 counted the wrong number of replicas"

# ---------------------------------------------------------------------------
step "cancelling a job while it runs"
# Park the fast workers first, so the slow worker below is the only candidate
# for the next task and the cancel lands while training is in flight.
stop_worker usage-a
stop_worker usage-b
sleep 1
cat > "$TMP/slow_trainer.sh" <<SLOW
#!/usr/bin/env bash
sleep 3
exec "$(pwd)/$TRAINER" "\$@"
SLOW
chmod +x "$TMP/slow_trainer.sh"
start_worker usage-cancel "$TMP/slow_trainer.sh" "$TMP/wc"
for _ in $(seq 1 60); do
  [ "$(nodes_online)" -ge 1 ] && break
  sleep 0.2
done

mk_bundle "$TMP/job-cancel" "usage-cancel-1" 20
CANCEL_JOB=$(submit_job "$TMP/job-cancel" 1)
[ -n "$CANCEL_JOB" ] && ok "the cancel candidate was accepted" ||
  bad "the cancel candidate was not accepted"
wait_status "$CANCEL_JOB" running 100 && ok "a worker is training the candidate" ||
  bad "the candidate never reached running"

CANCEL_CODE=$(curl -s -o "$TMP/cancel.json" -w '%{http_code}' -X POST \
                -H 'X-Grid-Token: usage-secret' \
                "http://127.0.0.1:$PORT/v1/jobs/$CANCEL_JOB/cancel")
[ "$CANCEL_CODE" = "200" ] && ok "the cancel was accepted" || bad "cancel returned $CANCEL_CODE"
[ "$(job_status "$CANCEL_JOB")" = "cancelled" ] && ok "the job reads cancelled" ||
  bad "the job reads '$(job_status "$CANCEL_JOB")'"
[ -s "$TMP/jobs/$CANCEL_JOB/result.json" ] && bad "a cancelled job wrote a result" ||
  ok "a cancelled job writes no result"
sleep 4
[ "$(job_status "$CANCEL_JOB")" = "cancelled" ] &&
  ok "the cancelled job stays cancelled after the trainer finishes" ||
  bad "the cancelled job reopened as '$(job_status "$CANCEL_JOB")'"
stop_worker usage-cancel

# ---------------------------------------------------------------------------
step "restarting the coordinator on the same database"
BEFORE=$(curl -fsS "http://127.0.0.1:$PORT/v1/summary" 2>/dev/null |
           grep -o '"jobs_done":[0-9]*' | cut -d: -f2)
kill -TERM "$ORCH_PID" 2>/dev/null
STOPPED=0
for _ in $(seq 1 50); do
  if ! kill -0 "$ORCH_PID" 2>/dev/null; then STOPPED=1; break; fi
  sleep 0.1
done
[ "$STOPPED" = "1" ] && ok "the coordinator stopped on SIGTERM" ||
  bad "the coordinator ignored SIGTERM"

start_orch "$TMP/orch2.log"
wait_health && ok "the coordinator came back on the same port" ||
  { bad "the coordinator did not restart"; sed -n 1,20p "$TMP/orch2.log"; }

AFTER=$(curl -fsS "http://127.0.0.1:$PORT/v1/summary" 2>/dev/null |
          grep -o '"jobs_done":[0-9]*' | cut -d: -f2)
[ -n "$BEFORE" ] && [ "$BEFORE" = "$AFTER" ] &&
  ok "the completed count survived the restart ($AFTER)" ||
  bad "the completed count changed from '${BEFORE:-?}' to '${AFTER:-?}'"
curl -fsS "http://127.0.0.1:$PORT/v1/jobs/$JOB1" 2>/dev/null |
  grep -q '"status":"completed"' &&
  ok "a job from before the restart is still readable" ||
  bad "job 1 was lost across the restart"

# ---------------------------------------------------------------------------
step "running more work after the restart"
start_worker usage-d "$TRAINER" "$TMP/wd"
for _ in $(seq 1 60); do
  [ "$(nodes_online)" -ge 1 ] && break
  sleep 0.2
done
[ "$(nodes_online)" -ge 1 ] && ok "a worker rejoined the restarted grid" ||
  bad "no worker rejoined after the restart"
mk_bundle "$TMP/job4" "usage-job-4" 8
JOB4=$(submit_job "$TMP/job4" 1)
wait_status "$JOB4" completed 200 && ok "work submitted after the restart ran" ||
  bad "post-restart job ended as '$(job_status "$JOB4")'"

# ---------------------------------------------------------------------------
step "reading the operator surfaces"
curl -fsS "http://127.0.0.1:$PORT/" > "$TMP/index.html" 2>/dev/null
grep -q "DistribAI" "$TMP/index.html" && ok "the dashboard index is served" ||
  bad "the dashboard index did not render"
ASSET_CODE=$(curl -s -o "$TMP/app.js" -w '%{http_code}' "http://127.0.0.1:$PORT/assets/app.js")
[ "$ASSET_CODE" = "200" ] && [ -s "$TMP/app.js" ] &&
  ok "a static asset is served (200)" || bad "static asset returned $ASSET_CODE"

curl -fsS "http://127.0.0.1:$PORT/v1/summary" > "$TMP/summary2.json" 2>/dev/null
if command -v python3 >/dev/null 2>&1; then
  if python3 -m json.tool "$TMP/summary2.json" >/dev/null 2>&1; then
    ok "the summary is valid JSON"
  else
    bad "the summary is not valid JSON"
  fi
else
  ok "skipped the JSON validity check (no python3)"
fi
curl -fsS "http://127.0.0.1:$PORT/v1/jobs/$JOB1" > "$TMP/job1.json" 2>/dev/null
if command -v python3 >/dev/null 2>&1; then
  # The detail embeds result.json, which repeats the replica rows, so count the
  # arrays rather than grepping for task ids.
  DETAIL=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d.get("replicas_detail") or []))' \
             "$TMP/job1.json" 2>/dev/null)
  REPFIELD=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("replicas"))' \
               "$TMP/job1.json" 2>/dev/null)
  [ "$REPFIELD" = "2" ] && ok "the job detail reports two replicas" ||
    bad "the job detail reports '${REPFIELD:-?}' replicas"
  [ "$DETAIL" = "2" ] && ok "the job detail expands both replica rows" ||
    bad "the job detail expands ${DETAIL:-0} replica rows"
else
  ok "skipped the replica-detail check (no python3)"
fi
grep -q '"node_id":"usage-[a-z]*","credits"' "$TMP/summary2.json" &&
  ok "the leaderboard names a contributing node" || bad "the leaderboard is empty"
grep -q '"jobs_done":3' "$TMP/summary2.json" &&
  ok "three jobs completed over the session" ||
  bad "the session did not record three completed jobs"

echo
if [ "$fail" -eq 0 ]; then
  echo "REAL USAGE GATE: PASS"
else
  echo "REAL USAGE GATE: FAILURES"
  echo "--- coordinator log ---"
  tail -20 "$TMP/orch.log"
fi
exit "$fail"
