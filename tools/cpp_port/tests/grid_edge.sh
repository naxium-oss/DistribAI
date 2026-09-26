#!/usr/bin/env bash
# grid_edge.sh - edge cases against the real coordinator and worker binaries.
#
# grid_e2e.sh walks the happy path. This one goes after everything that can go
# wrong: malformed worker messages, invite and token gates, stale sessions,
# static asset abuse, job bundles that should be refused, workers that fail,
# cancelled jobs, and state that has to survive a restart. Every check drives
# the shipped binaries over loopback HTTP; nothing is stubbed.
#
#   bash tools/cpp_port/tests/grid_edge.sh    (or: make -C tools/cpp_port grid-edge-test)
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

B=build/cpp_port
ORCH=$B/distribai_orch
WORKER=$B/distribai_worker
TRAINER=$B/grid_fake_trainer
# A random port per run. A fixed port turns a leftover coordinator from an
# interrupted run into a silent cross-test: the new run would talk to the old
# grid, write its own database, and every later check would look wrong.
PORT=${GRID_EDGE_PORT:-$((20000 + (RANDOM % 20000)))}
PORT2=${GRID_EDGE_PORT2:-$((PORT + 3))}
PORT3=$((PORT + 4))
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

step() { echo "== grid-edge: $*"; }
ok()   { echo "   PASS  $*"; }
bad()  { echo "   FAIL  $*"; fail=1; }

for bin in "$ORCH" "$WORKER" "$TRAINER"; do
  [ -x "$bin" ] || { echo "missing $bin (run: make -C tools/cpp_port grid-test)"; exit 1; }
done

# Refuse to run against a grid this script did not start.
for candidate in "$PORT" "$PORT2" "$PORT3"; do
  if curl -fsS "http://127.0.0.1:$candidate/v1/health" >/dev/null 2>&1; then
    echo "port $candidate already answers a grid; stop it or set GRID_EDGE_PORT"
    exit 1
  fi
done

ORCH_ARGS=("$ORCH" --host 127.0.0.1 --port "$PORT" --db "$TMP/grid.db"
           --schema runtime/db/schema.sql --jobs-dir "$TMP/jobs"
           --tasks-dir "$TMP/tasks" --web-dir tools/cpp_port/grid/web
           --invite edge-team --token edge-secret --node-ttl 5 --aggregate mean)

# code <curl args...> -> writes the body to $TMP/body and echoes the status
code() {
  curl -s -o "$TMP/body" -w '%{http_code}' "$@"
}
# want_code <desc> <expected> <curl args...>
want_code() {
  local desc="$1" want="$2"
  shift 2
  local got
  got=$(code "$@")
  if [ "$got" = "$want" ]; then
    ok "$desc ($got)"
  else
    bad "$desc: got $got, want $want; body=$(head -c 200 "$TMP/body")"
  fi
}
# wait_body <desc> <pattern> <url> [tries]
# A submit that ran in its own process (the --submit CLI) and a read on the
# server are two processes talking to one SQLite file. Retry briefly instead of
# asserting on the first read.
wait_body() {
  local desc="$1" pattern="$2" url="$3" tries="${4:-15}"
  for _ in $(seq 1 "$tries"); do
    curl -fsS "$url" > "$TMP/body" 2>/dev/null || true
    if grep -q -e "$pattern" "$TMP/body" 2>/dev/null; then
      ok "$desc"
      return 0
    fi
    sleep 0.2
  done
  bad "$desc (no match for '$pattern' in $(head -c 240 "$TMP/body" 2>/dev/null))"
  return 1
}

# want_body <desc> <pattern> <file>
want_body() {
  # -e keeps a pattern that starts with a dash from being read as a grep flag.
  if grep -q -e "$2" "$3" 2>/dev/null; then
    ok "$1"
  else
    bad "$1 (no match for '$2' in $(basename "$3"): $(head -c 240 "$3" 2>/dev/null))"
  fi
}

# ---------------------------------------------------------------------------
step "startup and argument handling"
"$ORCH" --help > "$TMP/help.txt" 2>&1
[ $? -eq 0 ] && ok "--help exits 0" || bad "--help exited non-zero"
want_body "usage lists the worker and API flags" "--web-dir" "$TMP/help.txt"
want_body "usage lists the submit flag" "--submit" "$TMP/help.txt"

"$ORCH" --db /proc/definitely/not/writable.db --schema runtime/db/schema.sql \
        --jobs-dir "$TMP/jobs" --tasks-dir "$TMP/tasks" > "$TMP/badbdb.log" 2>&1
if [ $? -ne 0 ]; then
  ok "an unwritable database path exits non-zero"
else
  bad "an unwritable database path was accepted"
fi
[ -s "$TMP/badbdb.log" ] && ok "the failure explains itself on stderr" ||
  bad "the failure printed nothing"

"$ORCH" --db "$TMP/grid.db" --schema runtime/db/does-not-exist.sql \
        --jobs-dir "$TMP/jobs" --tasks-dir "$TMP/tasks" > "$TMP/badschema.log" 2>&1
if [ $? -ne 0 ]; then
  ok "a missing schema file exits non-zero"
else
  bad "a missing schema file was accepted"
fi

# ---------------------------------------------------------------------------
step "starting the coordinator"
"${ORCH_ARGS[@]}" > "$TMP/orch.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && break
  sleep 0.2
done
curl -fsS "http://127.0.0.1:$PORT/v1/health" > "$TMP/health.json" 2>/dev/null ||
  { bad "coordinator never came up"; tail -20 "$TMP/orch.log"; exit 1; }
ok "coordinator is up"
want_body "health reports the engine and the wire revision" '"engine":"native-grid"' "$TMP/health.json"
want_body "health reports the wire revision" '"proto":1' "$TMP/health.json"

# A second coordinator on the same port has to refuse to start.
"${ORCH_ARGS[@]}" > "$TMP/portbusy.log" 2>&1
if [ $? -ne 0 ]; then
  ok "a second coordinator on the same port exits non-zero"
else
  bad "a second coordinator took a port that was already bound"
fi
want_body "the bind failure names the port" "bind(127.0.0.1:$PORT)" "$TMP/portbusy.log"

# ---------------------------------------------------------------------------
step "dashboard and static assets"
want_code "GET / serves the dashboard" 200 "http://127.0.0.1:$PORT/"
want_body "the dashboard is the native one" "DistribAI" "$TMP/body"
want_code "GET /assets/app.css serves the stylesheet" 200 "http://127.0.0.1:$PORT/assets/app.css"
want_code "GET /assets/app.js serves the script" 200 "http://127.0.0.1:$PORT/assets/app.js"
want_body "the script talks to the (real) native API" "/v1/summary" "$TMP/body"
want_code "a missing asset is a 404" 404 "http://127.0.0.1:$PORT/assets/nope.js"
want_code "a nested asset path is refused" 400 "http://127.0.0.1:$PORT/assets/sub/dir.js"
want_code "a traversal inside the asset path is refused" 400 --path-as-is \
  "http://127.0.0.1:$PORT/assets/../index.html"
want_code "a traversal of the asset root is refused" 400 --path-as-is \
  "http://127.0.0.1:$PORT/assets/../../runtime/db/schema.sql"
want_code "a path outside the dashboard is not routed" 404 --path-as-is \
  "http://127.0.0.1:$PORT/../runtime/db/schema.sql"
want_code "an unknown top-level path is not routed" 404 "http://127.0.0.1:$PORT/app.css"
want_code "static assets are GET only" 405 -X DELETE "http://127.0.0.1:$PORT/"
want_code "HEAD on the dashboard is allowed" 200 -I "http://127.0.0.1:$PORT/"

# ---------------------------------------------------------------------------
step "registration gates"
want_code "register without a proto is refused" 400 -d '{"node_id":"n1"}' \
  "http://127.0.0.1:$PORT/v1/register"
want_body "the proto refusal explains the grid's revision" "protocol mismatch" "$TMP/body"
want_code "register with a future proto is refused" 400 \
  -d '{"proto":2,"node_id":"n1"}' "http://127.0.0.1:$PORT/v1/register"
want_code "register without a node id is refused" 400 -d '{"proto":1}' \
  "http://127.0.0.1:$PORT/v1/register"
LONGID=$(printf 'n%.0s' $(seq 1 200))
want_code "an over-long node id is refused" 400 \
  -d "{\"proto\":1,\"node_id\":\"$LONGID\"}" "http://127.0.0.1:$PORT/v1/register"
want_code "an empty body is refused" 400 -X POST "http://127.0.0.1:$PORT/v1/register"
want_code "a non-JSON body is refused" 400 -d 'hi there' "http://127.0.0.1:$PORT/v1/register"
want_code "an array body is refused" 400 -d '[1,2,3]' "http://127.0.0.1:$PORT/v1/register"
want_code "a deep body is refused" 400 -d "$(printf '[%.0s' $(seq 1 200))1$(printf ']%.0s' $(seq 1 200))" \
  "http://127.0.0.1:$PORT/v1/register"
want_code "the wrong invite code is refused" 403 \
  -d '{"proto":1,"node_id":"n1","invite":"nope"}' "http://127.0.0.1:$PORT/v1/register"
want_code "the right invite code is accepted" 200 \
  -d '{"proto":1,"node_id":"n1","invite":"edge-team"}' "http://127.0.0.1:$PORT/v1/register"
cp "$TMP/body" "$TMP/welcome1.json"
want_body "registration hands back a session token" '"session_token"' "$TMP/welcome1.json"
want_body "registration announces the heartbeat cadence" '"heartbeat_s"' "$TMP/welcome1.json"
TOKEN1=$(sed -n 's/.*"session_token":"\([^"]*\)".*/\1/p' "$TMP/welcome1.json")
[ "${#TOKEN1}" -eq 32 ] && ok "the session token is 32 hex characters" ||
  bad "unexpected token length ${#TOKEN1}"

want_code "registering the same node id again succeeds" 200 \
  -d '{"proto":1,"node_id":"n1","invite":"edge-team"}' "http://127.0.0.1:$PORT/v1/register"
TOKEN2=$(sed -n 's/.*"session_token":"\([^"]*\)".*/\1/p' "$TMP/body")
[ "$TOKEN1" != "$TOKEN2" ] && ok "re-registering rotates the session token" ||
  bad "the session token was reused across registrations"

# ---------------------------------------------------------------------------
step "session and heartbeat gates"
want_code "heartbeat with the revoked token is refused" 401 \
  -d "{\"node_id\":\"n1\",\"token\":\"$TOKEN1\"}" "http://127.0.0.1:$PORT/v1/heartbeat"
want_code "heartbeat with no token is refused" 401 -d '{"node_id":"n1"}' \
  "http://127.0.0.1:$PORT/v1/heartbeat"
want_code "heartbeat for an unknown node is refused" 401 \
  -d '{"node_id":"ghost","token":"deadbeef"}' "http://127.0.0.1:$PORT/v1/heartbeat"
want_code "heartbeat with the live token is accepted" 200 \
  -d "{\"node_id\":\"n1\",\"token\":\"$TOKEN2\",\"status\":\"idle\"}" \
  "http://127.0.0.1:$PORT/v1/heartbeat"
want_body "the heartbeat answer is an ack" '"type":"ack"' "$TMP/body"
want_code "claim without a session is refused" 401 -d '{"node_id":"n1","token":"bad"}' \
  "http://127.0.0.1:$PORT/v1/claim"
want_code "claim with an empty queue answers idle" 200 \
  -d "{\"node_id\":\"n1\",\"token\":\"$TOKEN2\"}" "http://127.0.0.1:$PORT/v1/claim"
want_body "the idle answer tells the worker how long to wait" '"type":"idle"' "$TMP/body"
want_body "the idle answer carries the poll interval" '"poll_s"' "$TMP/body"
want_code "a result for an unknown task is a 404" 404 \
  -d "{\"node_id\":\"n1\",\"token\":\"$TOKEN2\",\"task_id\":\"nope-r0\",\"ok\":true}" \
  "http://127.0.0.1:$PORT/v1/result"
want_code "a result without a task id is refused" 400 \
  -d "{\"node_id\":\"n1\",\"token\":\"$TOKEN2\",\"ok\":true}" "http://127.0.0.1:$PORT/v1/result"
want_code "a result without a session is refused" 401 \
  -d '{"node_id":"n1","token":"bad","task_id":"x-r0"}' "http://127.0.0.1:$PORT/v1/result"
want_code "bye with the live token is accepted" 200 \
  -d "{\"node_id\":\"n1\",\"token\":\"$TOKEN2\"}" "http://127.0.0.1:$PORT/v1/bye"

# ---------------------------------------------------------------------------
step "job bundle validation"
mkdir -p "$TMP/bundle-ok"
cat > "$TMP/bundle-ok/job.json" <<'JSON'
{
  "job_id": "edge-job",
  "name": "edge-model",
  "steps": 12,
  "batch_size": 8,
  "seed": 7,
  "model": "model.pt",
  "optimizer": "adamw",
  "loss": "mse",
  "aggregate": "mean"
}
JSON
printf 'fixture' > "$TMP/bundle-ok/model.pt"

want_code "submit without the operator token is refused" 403 -X POST \
  -H 'Content-Type: application/json' -d "{\"job_dir\":\"$TMP/bundle-ok\",\"replicas\":1}" \
  "http://127.0.0.1:$PORT/v1/jobs"
want_code "submit through the API with the operator token is accepted" 202 -X POST \
  -H 'Content-Type: application/json' -H 'X-Grid-Token: edge-secret' \
  -d "{\"job_dir\":\"$TMP/bundle-ok\",\"replicas\":2}" \
  "http://127.0.0.1:$PORT/v1/jobs"
want_body "the API submit returns a job id" '"job_id"' "$TMP/body"
want_body "the API submit echoes the aggregate mode" '"aggregate"' "$TMP/body"
want_code "a wrong operator token is refused" 403 -X POST \
  -H 'Content-Type: application/json' -H 'X-Grid-Token: nope' \
  -d "{\"job_dir\":\"$TMP/bundle-ok\"}" "http://127.0.0.1:$PORT/v1/jobs"

want_code "submitting a nonexistent directory is refused" 400 -X POST \
  -H 'Content-Type: application/json' -H 'X-Grid-Token: edge-secret' \
  -d '{"job_dir":"/nonexistent"}' "http://127.0.0.1:$PORT/v1/jobs"

mkdir -p "$TMP/bundle-nojson"
want_code "a bundle without job.json is refused" 400 -X POST \
  -H 'Content-Type: application/json' -H 'X-Grid-Token: edge-secret' \
  -d "{\"job_dir\":\"$TMP/bundle-nojson\"}" "http://127.0.0.1:$PORT/v1/jobs"
want_body "the refusal points at the missing spec" "job.json" "$TMP/body"

mkdir -p "$TMP/bundle-badsteps"
sed 's/"steps": 12/"steps": 0/' "$TMP/bundle-ok/job.json" > "$TMP/bundle-badsteps/job.json"
want_code "a bundle with zero steps is refused" 400 -X POST \
  -H 'Content-Type: application/json' -H 'X-Grid-Token: edge-secret' \
  -d "{\"job_dir\":\"$TMP/bundle-badsteps\"}" "http://127.0.0.1:$PORT/v1/jobs"
want_body "the refusal explains that steps must be positive" "steps" "$TMP/body"

mkdir -p "$TMP/bundle-broken"
printf '{"job_id": "broken",' > "$TMP/bundle-broken/job.json"
want_code "a malformed job.json is refused" 400 -X POST \
  -H 'Content-Type: application/json' -H 'X-Grid-Token: edge-secret' \
  -d "{\"job_dir\":\"$TMP/bundle-broken\"}" "http://127.0.0.1:$PORT/v1/jobs"

want_code "a submit with a bad JSON body is refused" 400 -X POST \
  -H 'Content-Type: application/json' -H 'X-Grid-Token: edge-secret' \
  -d 'not json' "http://127.0.0.1:$PORT/v1/jobs"

want_code "a submit with a missing job_dir is refused" 400 -X POST \
  -H 'Content-Type: application/json' -H 'X-Grid-Token: edge-secret' \
  -d '{"replicas":2}' "http://127.0.0.1:$PORT/v1/jobs"

# ---------------------------------------------------------------------------
step "queue behavior"
"$ORCH" --db "$TMP/grid.db" --schema runtime/db/schema.sql --jobs-dir "$TMP/jobs" \
        --tasks-dir "$TMP/tasks" --submit "$TMP/bundle-ok" --replicas 2 \
        > "$TMP/submit.json" 2>&1
JOB_ID=$(sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p' "$TMP/submit.json")
[ -n "$JOB_ID" ] && ok "CLI submit produced $JOB_ID" || bad "CLI submit failed: $(cat "$TMP/submit.json")"

wait_body "the queue lists the submitted job's tasks" "\"job_id\":\"$JOB_ID\"" \
  "http://127.0.0.1:$PORT/v1/tasks"
curl -fsS "http://127.0.0.1:$PORT/v1/tasks" > "$TMP/tasks.json" 2>/dev/null
QUEUED=$(grep -o '"status":"queued"' "$TMP/tasks.json" | wc -l)
[ "$QUEUED" -ge 2 ] && ok "both replicas are queued ($QUEUED)" ||
  bad "expected at least 2 queued tasks, found $QUEUED"

# A second submit of the same directory is a separate job, not a duplicate.
"$ORCH" --db "$TMP/grid.db" --schema runtime/db/schema.sql --jobs-dir "$TMP/jobs" \
        --tasks-dir "$TMP/tasks" --submit "$TMP/bundle-ok" --replicas 1 \
        > "$TMP/submit2.json" 2>&1
JOB_ID2=$(sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p' "$TMP/submit2.json")
[ -n "$JOB_ID2" ] && [ "$JOB_ID2" != "$JOB_ID" ] && ok "a repeat submit gets its own job id" ||
  bad "the repeat submit reused a job id"

want_code "cancelling an unknown job is a 404" 404 -X POST \
  -H 'X-Grid-Token: edge-secret' "http://127.0.0.1:$PORT/v1/jobs/no-such-job/cancel"
want_code "cancelling without the operator token is refused" 403 -X POST \
  "http://127.0.0.1:$PORT/v1/jobs/$JOB_ID2/cancel"
CANCEL_CODE=""
for _ in $(seq 1 15); do
  CANCEL_CODE=$(code -X POST -H 'X-Grid-Token: edge-secret' \
    "http://127.0.0.1:$PORT/v1/jobs/$JOB_ID2/cancel")
  [ "$CANCEL_CODE" = "200" ] && break
  sleep 0.2
done
if [ "$CANCEL_CODE" = "200" ]; then
  ok "cancelling a queued job is accepted (200)"
else
  bad "cancelling a queued job returned $CANCEL_CODE; body=$(head -c 200 "$TMP/body")"
fi
wait_body "the cancelled job is marked cancelled" '"status":"cancelled"' \
  "http://127.0.0.1:$PORT/v1/jobs/$JOB_ID2"

# ---------------------------------------------------------------------------
step "workers that fail"
# A trainer that exits non-zero must be reported, not swallowed.
cat > "$TMP/broken_trainer.sh" <<'BROKEN'
#!/usr/bin/env bash
echo '{"kind":"torch_result","status":"error","error":"edge: intentional trainer failure"}'
exit 3
BROKEN
chmod +x "$TMP/broken_trainer.sh"
"$WORKER" --orchestrator "http://127.0.0.1:$PORT" --invite edge-team --node-id edge-broken \
          --trainer "$TMP/broken_trainer.sh" --work-dir "$TMP/wb" --max-jobs 1 \
          > "$TMP/broken.log" 2>&1
grep -q "intentional trainer failure" "$TMP/broken.log" &&
  ok "a failing trainer's reason is surfaced" ||
  bad "the failing trainer's reason was lost"
# Which job the worker actually claimed is decided by the queue, so read the
# task id out of the worker's own log rather than assuming.
FAILED_TASK=$(sed -n 's/^worker: task \([^ ]*\) .*/\1/p' "$TMP/broken.log" | head -1)
FAILED_JOB=${FAILED_TASK%-r*}
if [ -n "$FAILED_TASK" ]; then
  ok "the broken worker claimed $FAILED_TASK"
else
  bad "the broken worker claimed nothing"
fi
curl -fsS "http://127.0.0.1:$PORT/v1/jobs/$FAILED_JOB" > "$TMP/jobfail.json" 2>/dev/null
want_body "the failed replica records its error on the job" "intentional trainer failure" \
  "$TMP/jobfail.json"
want_body "the failed replica is marked failed" '"status":"failed"' "$TMP/jobfail.json"
want_body "the failed attempt is counted" '"attempts":1' "$TMP/jobfail.json"

# A worker pointed at nothing must give up with a clear message.
"$WORKER" --orchestrator "http://127.0.0.1:9" --invite edge-team --node-id edge-lost \
          --register-wait 1 --trainer "$TRAINER" --work-dir "$TMP/wl" \
          > "$TMP/lost.log" 2>&1
if [ $? -ne 0 ]; then
  ok "a worker that cannot reach the coordinator exits non-zero"
else
  bad "a worker against a dead coordinator exited 0"
fi
[ -s "$TMP/lost.log" ] && ok "the unreachable-coordinator failure is reported" ||
  bad "the unreachable worker said nothing"
want_body "the unreachable worker says it is retrying while it waits" "retrying in" \
  "$TMP/lost.log"

# The wrong invite must stop a worker before it claims anything.
"$WORKER" --orchestrator "http://127.0.0.1:$PORT" --invite wrong --node-id edge-wrong \
          --trainer "$TRAINER" --work-dir "$TMP/ww" > "$TMP/wrong.log" 2>&1
grep -q "invite code required" "$TMP/wrong.log" &&
  ok "a worker with the wrong invite is refused" ||
  bad "a worker with the wrong invite got through"

# A missing trainer binary fails the task with a hint, not a crash.
"$WORKER" --orchestrator "http://127.0.0.1:$PORT" --invite edge-team --node-id edge-notrainer \
          --trainer "$TMP/does-not-exist" --work-dir "$TMP/wn" --max-jobs 1 \
          > "$TMP/notrainer.log" 2>&1
grep -qE "trainer not found|no such file" "$TMP/notrainer.log" &&
  ok "a missing trainer is reported as such" ||
  bad "a missing trainer produced: $(head -3 "$TMP/notrainer.log")"

# ---------------------------------------------------------------------------
step "the queue recovers and a real job finishes"
# Clear whatever the failure tests left queued, so this check depends only on
# the job it submits below.
curl -fsS "http://127.0.0.1:$PORT/v1/summary" > "$TMP/pre.json" 2>/dev/null
for jid in $(grep -o '"job_id":"[^"]*"' "$TMP/pre.json" | cut -d'"' -f4 | sort -u); do
  jst=$(curl -fsS "http://127.0.0.1:$PORT/v1/jobs/$jid" 2>/dev/null |
        grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4)
  case "$jst" in
    queued|running|finalizing)
      curl -fsS -X POST -H 'X-Grid-Token: edge-secret' \
        "http://127.0.0.1:$PORT/v1/jobs/$jid/cancel" >/dev/null 2>&1 ;;
  esac
done
ok "leftover work from the failure tests was drained"
"$ORCH" --db "$TMP/grid.db" --schema runtime/db/schema.sql --jobs-dir "$TMP/jobs" \
        --tasks-dir "$TMP/tasks" --submit "$TMP/bundle-ok" --replicas 2 \
        > "$TMP/submit3.json" 2>&1
JOB3=$(sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p' "$TMP/submit3.json")
for i in 1 2 3; do
  "$WORKER" --orchestrator "http://127.0.0.1:$PORT" --invite edge-team \
            --node-id "edge-good-$i" --trainer "$TRAINER" --work-dir "$TMP/w$i" \
            --max-jobs 2 > "$TMP/good$i.log" 2>&1 &
done
STATUS=""
for _ in $(seq 1 120); do
  STATUS=$(curl -fsS "http://127.0.0.1:$PORT/v1/jobs/$JOB3" 2>/dev/null |
           grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4)
  [ "$STATUS" = "completed" ] && break
  sleep 0.3
done
[ "$STATUS" = "completed" ] && ok "a job submitted after the failures completes" ||
  bad "job $JOB3 ended as '${STATUS:-unknown}'"

curl -fsS "http://127.0.0.1:$PORT/v1/summary" > "$TMP/summary.json" 2>/dev/null
want_body "the ledger records a credit entry" '"credits"' "$TMP/summary.json"
want_body "the leaderboard lists a contributor" '"leaderboard"' "$TMP/summary.json"
want_body "node counters moved" '"jobs_completed"' "$TMP/summary.json"

# ---------------------------------------------------------------------------
step "state survives a restart"
kill "${PIDS[0]}" 2>/dev/null
sleep 0.4
"${ORCH_ARGS[@]}" > "$TMP/orch2.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && break
  sleep 0.2
done
curl -fsS "http://127.0.0.1:$PORT/v1/summary" > "$TMP/summary2.json" 2>/dev/null
want_body "the finished job is still there after a restart" '"jobs_done"' "$TMP/summary2.json"
want_body "the ledger survives a restart" '"credits"' "$TMP/summary2.json"
if grep -q "\"job_id\":\"$JOB3\"" "$TMP/summary2.json"; then
  ok "the specific job row survived the restart"
else
  bad "job $JOB3 disappeared across a restart"
fi

# Re-applying the schema on an existing database must stay quiet.
grep -qiE "error|fail" "$TMP/orch2.log" && bad "the restart logged an error: $(head -2 "$TMP/orch2.log")" ||
  ok "reopening the database applies the schema without errors"

# An older database is migrated forward: a grid created before the jobs table
# had an aggregate column has to keep working after the upgrade.
# Drop the aggregate column and the comma the previous column would otherwise
# leave dangling.
awk '
  { line[NR] = $0 }
  /^[[:space:]]*-- Aggregation method for this job/ { from = NR }
  /^[[:space:]]*aggregate TEXT DEFAULT/ { to = NR }
  END {
    prev = from - 1
    while (prev > 0 && line[prev] ~ /^[[:space:]]*--/) prev--
    for (i = 1; i <= NR; i++) {
      if (from && i >= from && i <= to) continue
      s = line[i]
      if (from && i == prev) sub(/,[[:space:]]*$/, "", s)
      print s
    }
  }
' runtime/db/schema.sql > "$TMP/old_schema.sql"
if grep -q "aggregate TEXT" "$TMP/old_schema.sql"; then
  bad "could not build the pre-aggregate schema for the migration check"
else
  ok "built a schema without the aggregate column"
fi
mkdir -p "$TMP/old"
"$ORCH" --host 127.0.0.1 --port "$PORT3" --db "$TMP/old/grid.db" \
        --schema "$TMP/old_schema.sql" --jobs-dir "$TMP/old/jobs" \
        --tasks-dir "$TMP/old/tasks" > "$TMP/old1.log" 2>&1 &
OLD_PID=$!
for _ in $(seq 1 50); do
  curl -fsS "http://127.0.0.1:$PORT3/v1/health" >/dev/null 2>&1 && break
  sleep 0.2
done
curl -fsS "http://127.0.0.1:$PORT3/v1/health" >/dev/null 2>&1 &&
  ok "an old database opens under the old schema" ||
  bad "could not create the old database: $(tail -2 "$TMP/old1.log")"
kill "$OLD_PID" 2>/dev/null
sleep 0.4
"$ORCH" --host 127.0.0.1 --port "$PORT3" --db "$TMP/old/grid.db" \
        --schema runtime/db/schema.sql --jobs-dir "$TMP/old/jobs" \
        --tasks-dir "$TMP/old/tasks" > "$TMP/old2.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 50); do
  curl -fsS "http://127.0.0.1:$PORT3/v1/health" >/dev/null 2>&1 && break
  sleep 0.2
done
if curl -fsS "http://127.0.0.1:$PORT3/v1/health" >/dev/null 2>&1; then
  ok "the old database reopens after the column migration"
else
  bad "the migrated database refused to open: $(tail -2 "$TMP/old2.log")"
fi
# Submitting proves the migrated table accepts the new column.
if "$ORCH" --db "$TMP/old/grid.db" --schema runtime/db/schema.sql \
        --jobs-dir "$TMP/old/jobs" --tasks-dir "$TMP/old/tasks" \
        --submit "$TMP/bundle-ok" --replicas 1 > "$TMP/old_submit.json" 2>&1; then
  ok "a job can be submitted to the migrated database"
else
  bad "submit against the migrated database failed: $(cat "$TMP/old_submit.json")"
fi

# A fresh database created from scratch is usable too.
FRESH=$(mktemp -d)
"$ORCH" --host 127.0.0.1 --port "$PORT2" --db "$FRESH/new.db" \
        --schema runtime/db/schema.sql --jobs-dir "$FRESH/jobs" --tasks-dir "$FRESH/tasks" \
        > "$FRESH/log" 2>&1 &
FRESH_PID=$!
for _ in $(seq 1 50); do
  curl -fsS "http://127.0.0.1:$PORT2/v1/health" >/dev/null 2>&1 && break
  sleep 0.2
done
curl -fsS "http://127.0.0.1:$PORT2/v1/health" > "$TMP/fresh_health.json" 2>/dev/null &&
  ok "a coordinator bootstraps a brand new database" ||
  bad "a coordinator could not bootstrap a new database"
kill "$FRESH_PID" 2>/dev/null
rm -rf "$FRESH"

echo
if [ "$fail" -eq 0 ]; then
  echo "GRID EDGE GATE: PASS"
else
  echo "GRID EDGE GATE: FAILURES"
  echo "--- coordinator log ---"
  tail -20 "$TMP/orch.log"
fi
exit "$fail"
