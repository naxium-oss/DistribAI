#!/usr/bin/env bash
# grid_security.sh - who may do what on the native grid.
#
# The coordinator faces a tunnel, so every endpoint has to decide for itself
# whether an anonymous caller, a registered stranger, or the operator is asking.
# grid_edge.sh checks malformed input; this gate checks authorization and the
# session model, against the real binaries:
#
#   * the admin token gates submit and cancel, and a query param does not
#   * the invite gates registration
#   * body shape gates: non-JSON, arrays, oversized ids, deep nesting
#   * a worker may only act on its own session and its own assigned task
#   * tokens never leak into any response
#   * a hostile node id is stored as data, not executed
#
#   bash tools/cpp_port/tests/grid_security.sh  (or: make -C tools/cpp_port grid-security-test)
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

B=build/cpp_port
ORCH=$B/distribai_orch
WORKER=$B/distribai_worker
TRAINER=$B/grid_fake_trainer
PORT=${GRID_SECURITY_PORT:-$((10000 + (RANDOM % 9000)))}
INVITE=sec-team
TOKEN=sec-secret
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

step() { echo "== grid-security: $*"; }
ok()   { echo "   PASS  $*"; }
bad()  { echo "   FAIL  $*"; fail=1; }
check_code() { # check_code <expected> <label> <actual>
  if [ "$3" = "$1" ]; then ok "$2 ($3)"; else bad "$2: got $3, expected $1"; fi
}

[ -x "$ORCH" ] || { echo "missing $ORCH (run: make -C tools/cpp_port grid-test)"; exit 1; }
if curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; then
  echo "port $PORT already answers a grid; stop it or set GRID_SECURITY_PORT"
  exit 1
fi

post() { # post <path> <json-body> [extra curl args...]
  local path=$1 body=$2
  shift 2
  curl -s -o "$TMP/body.json" -w '%{http_code}' -X POST \
       -H 'Content-Type: application/json' "$@" --data "$body" \
       "http://127.0.0.1:$PORT$path"
}
post_file() { # post_file <path> <file> [extra curl args...]
  local path=$1 file=$2
  shift 2
  curl -s -o "$TMP/body.json" -w '%{http_code}' -X POST \
       -H 'Content-Type: application/json' "$@" --data-binary "@$file" \
       "http://127.0.0.1:$PORT$path"
}
getcode() { # getcode <path> [extra curl args...]
  local path=$1
  shift
  curl -s -o "$TMP/body.out" -w '%{http_code}' "$@" "http://127.0.0.1:$PORT$path"
}
register() { # register <json> [extra curl args...] -> prints the body
  local body=$1
  shift
  post /v1/register "$body" "$@" >/dev/null
  cat "$TMP/body.json"
}
token_of() { printf '%s' "$1" | sed -n 's/.*"session_token":"\([^"]*\)".*/\1/p'; }
mk_bundle() { # mk_bundle <dir> <job-id>
  mkdir -p "$1"
  cat > "$1/job.json" <<JSON
{
  "job_id": "$2",
  "name": "$2",
  "steps": 6,
  "batch_size": 2,
  "seed": 1,
  "model": "model.pt",
  "optimizer": "adamw",
  "loss": "mse",
  "aggregate": "mean"
}
JSON
  printf 'fixture' > "$1/model.pt"
}

step "starting the coordinator"
"$ORCH" --host 127.0.0.1 --port "$PORT" \
        --db "$TMP/grid.db" --schema runtime/db/schema.sql \
        --jobs-dir "$TMP/jobs" --tasks-dir "$TMP/tasks" \
        --web-dir tools/cpp_port/grid/web \
        --invite "$INVITE" --token "$TOKEN" \
        --node-ttl 60 --aggregate mean > "$TMP/orch.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && break
  sleep 0.2
done
curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && ok "coordinator is up" ||
  { bad "coordinator never started"; sed -n 1,20p "$TMP/orch.log"; exit 1; }

# Three bundles, because a job id is a primary key and each submit below needs
# a job the grid has not seen yet.
mk_bundle "$TMP/bundle" sec-job
mk_bundle "$TMP/bundle2" sec-job-2
mk_bundle "$TMP/bundle3" sec-job-3

# ---------------------------------------------------------------------------
step "the admin token gates the operator writes"
CODE=$(post /v1/jobs "{\"job_dir\":\"$TMP/bundle\",\"replicas\":1}")
check_code 403 "submit without a token is refused" "$CODE"
CODE=$(post /v1/jobs "{\"job_dir\":\"$TMP/bundle\",\"replicas\":1}" -H 'X-Grid-Token: wrong')
check_code 403 "submit with a wrong token is refused" "$CODE"
CODE=$(post /v1/jobs "{\"job_dir\":\"$TMP/bundle\",\"replicas\":1}" -H 'x-grid-token: wrong')
check_code 403 "submit with a wrong lowercase header is refused" "$CODE"
CODE=$(post "/v1/jobs?token=$TOKEN" "{\"job_dir\":\"$TMP/bundle\",\"replicas\":1}")
check_code 403 "a token in the query string does not authorize" "$CODE"
CODE=$(post /v1/jobs '{"job_dir":"/nonexistent","replicas":1}' -H "X-Grid-Token: $TOKEN")
check_code 400 "a bad job_dir is refused once authorized" "$CODE"
CODE=$(post /v1/jobs "{\"job_dir\":\"$TMP/bundle\",\"replicas\":1}" -H "x-grid-token: $TOKEN")
check_code 202 "submit with the right token is accepted" "$CODE"
JOB=$(sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p' "$TMP/body.json")
[ -n "$JOB" ] && ok "the submitted job id is $JOB" || bad "no job id came back"

CODE=$(getcode /v1/health)
check_code 200 "health is open to anonymous callers" "$CODE"
CODE=$(getcode /v1/summary)
check_code 200 "summary is open to anonymous callers" "$CODE"
CODE=$(getcode /v1/nodes)
check_code 200 "the node list is open to anonymous callers" "$CODE"
CODE=$(getcode /v1/tasks)
check_code 200 "the task list is open to anonymous callers" "$CODE"
CODE=$(post /v1/jobs '{"job_dir":"x",' -H "X-Grid-Token: $TOKEN")
check_code 400 "a malformed submit body is refused" "$CODE"

step "the invite gates registration"
CODE=$(post /v1/register '{"proto":1,"node_id":"no-invite"}')
check_code 403 "registration without an invite is refused" "$CODE"
CODE=$(post /v1/register '{"proto":1,"node_id":"bad-invite","invite":"guess"}')
check_code 403 "registration with a wrong invite is refused" "$CODE"
CODE=$(post /v1/register '{"proto":99,"node_id":"bad-proto","invite":"sec-team"}')
check_code 400 "a protocol mismatch is refused" "$CODE"
CODE=$(post /v1/register '{"proto":1,"node_id":"","invite":"sec-team"}')
check_code 400 "an empty node id is refused" "$CODE"
LONGID=$(printf 'n%.0s' $(seq 1 200))
CODE=$(post /v1/register "{\"proto\":1,\"node_id\":\"$LONGID\",\"invite\":\"$INVITE\"}")
check_code 400 "an over-long node id is refused" "$CODE"
CODE=$(post /v1/register 'not json at all' )
check_code 400 "a non-JSON registration body is refused" "$CODE"
CODE=$(post /v1/register '[1,2,3]')
check_code 400 "a JSON array registration body is refused" "$CODE"
DEEP=""
for _ in $(seq 1 200); do DEEP="$DEEP{\"a\":"; done
DEEP="${DEEP}1"
for _ in $(seq 1 200); do DEEP="$DEEP}"; done
CODE=$(post /v1/register "$DEEP")
check_code 400 "a deeply nested registration body is refused" "$CODE"

A_BODY=$(register "{\"proto\":1,\"node_id\":\"sec-a\",\"invite\":\"$INVITE\"}")
A_TOK=$(token_of "$A_BODY")
[ -n "$A_TOK" ] && ok "the first worker registered with a session token" ||
  bad "the first worker got no session token"
B_BODY=$(register "{\"proto\":1,\"node_id\":\"sec-b\",\"invite\":\"$INVITE\"}")
B_TOK=$(token_of "$B_BODY")
[ -n "$B_TOK" ] && ok "the second worker registered" || bad "the second worker failed"

step "a session token is bound to its node"
CODE=$(post /v1/heartbeat "{\"node_id\":\"sec-a\",\"token\":\"bogus\",\"status\":\"idle\"}")
check_code 401 "heartbeat with a bogus token is refused" "$CODE"
CODE=$(post /v1/heartbeat '{"node_id":"sec-a","status":"idle"}')
check_code 401 "heartbeat without a token is refused" "$CODE"
CODE=$(post /v1/heartbeat "{\"node_id\":\"sec-a\",\"token\":\"$A_TOK\",\"status\":\"idle\"}")
check_code 200 "heartbeat with the right token is accepted" "$CODE"
CODE=$(post /v1/heartbeat "{\"node_id\":\"sec-b\",\"token\":\"$A_TOK\",\"status\":\"idle\"}")
check_code 401 "another node's session token is refused" "$CODE"
CODE=$(post /v1/claim "{\"node_id\":\"sec-a\",\"token\":\"bogus\"}")
check_code 401 "claim with a bogus token is refused" "$CODE"
CODE=$(post /v1/result "{\"node_id\":\"sec-a\",\"token\":\"bogus\",\"task_id\":\"x\",\"ok\":false}")
check_code 401 "result with a bogus token is refused" "$CODE"
CODE=$(post /v1/bye "{\"node_id\":\"sec-a\",\"token\":\"bogus\"}")
check_code 401 "bye with a bogus token is refused" "$CODE"

# Rotating a token must retire the old one.
OLD=$A_TOK
ROT_BODY=$(register "{\"proto\":1,\"node_id\":\"sec-a\",\"invite\":\"$INVITE\"}")
A_TOK=$(token_of "$ROT_BODY")
CODE=$(post /v1/heartbeat "{\"node_id\":\"sec-a\",\"token\":\"$OLD\",\"status\":\"idle\"}")
check_code 401 "the rotated-out token is refused" "$CODE"
CODE=$(post /v1/heartbeat "{\"node_id\":\"sec-a\",\"token\":\"$A_TOK\",\"status\":\"idle\"}")
check_code 200 "the fresh token still works" "$CODE"

step "a worker may only report its own task"
CLAIM=$(post /v1/claim "{\"node_id\":\"sec-a\",\"token\":\"$A_TOK\"}" >/dev/null; cat "$TMP/body.json")
TASK=$(printf '%s' "$CLAIM" | sed -n 's/.*"task_id":"\([^"]*\)".*/\1/p')
[ "$TASK" = "$JOB-r0" ] && ok "the first worker claimed $TASK" ||
  bad "expected $JOB-r0, the claim returned '${TASK:-nothing}'"
CODE=$(post /v1/result \
        "{\"node_id\":\"sec-b\",\"token\":\"$B_TOK\",\"task_id\":\"$TASK\",\"ok\":false,\"error\":\"spoof\"}")
check_code 403 "another node cannot fail the claimed task" "$CODE"
grep -q "assigned to another node" "$TMP/body.json" &&
  ok "the refusal names the reason" || bad "the refusal message is vague"
CODE=$(post /v1/result \
        "{\"node_id\":\"sec-b\",\"token\":\"$B_TOK\",\"task_id\":\"$TASK\",\"ok\":true,\"envelope_b64\":\"x\"}")
check_code 403 "another node cannot push an envelope into the task" "$CODE"
CODE=$(post /v1/result \
        "{\"node_id\":\"sec-a\",\"token\":\"$A_TOK\",\"task_id\":\"$TASK\",\"ok\":false,\"error\":\"probe\"}")
check_code 200 "the assignee may report on its own task" "$CODE"
CODE=$(post /v1/result \
        "{\"node_id\":\"sec-a\",\"token\":\"$A_TOK\",\"task_id\":\"ghost-r0\",\"ok\":false}")
check_code 404 "a result for an unknown task is refused" "$CODE"
CODE=$(post /v1/result "{\"node_id\":\"sec-a\",\"token\":\"$A_TOK\",\"ok\":false}")
check_code 400 "a result with no task id is refused" "$CODE"

step "an authorized envelope still has to be readable"
post /v1/register "{\"proto\":1,\"node_id\":\"sec-c\",\"invite\":\"$INVITE\"}" >/dev/null
C_TOK=$(token_of "$(cat "$TMP/body.json")")
CODE=$(post /v1/jobs "{\"job_dir\":\"$TMP/bundle2\",\"replicas\":1}" -H "X-Grid-Token: $TOKEN")
check_code 202 "a second job is accepted for the envelope check" "$CODE"
JOB2=$(sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p' "$TMP/body.json")
post /v1/claim "{\"node_id\":\"sec-c\",\"token\":\"$C_TOK\"}" >/dev/null
TASK2=$(sed -n 's/.*"task_id":"\([^"]*\)".*/\1/p' "$TMP/body.json")
[ -n "$TASK2" ] && ok "the third worker claimed $TASK2" || bad "the third worker claimed nothing"
CODE=$(post /v1/result \
        "{\"node_id\":\"sec-c\",\"token\":\"$C_TOK\",\"task_id\":\"$TASK2\",\"ok\":true,\"envelope_b64\":\"not base64!!\"}")
check_code 400 "a garbage envelope is refused" "$CODE"

step "a hostile node id is stored as data"
cat > "$TMP/inject.json" <<'JSON'
{"proto":1,"node_id":"x'); DROP TABLE jobs;--","invite":"sec-team"}
JSON
CODE=$(post_file /v1/register "$TMP/inject.json")
check_code 200 "a SQL-shaped node id registers as a literal" "$CODE"
INJ_TOK=$(token_of "$(cat "$TMP/body.json")")
[ -n "$INJ_TOK" ] && ok "the hostile id got a session" || bad "the hostile id got no session"
cat > "$TMP/quote.json" <<'JSON'
{"proto":1,"node_id":"nod\"e\\x","invite":"sec-team"}
JSON
CODE=$(post_file /v1/register "$TMP/quote.json")
check_code 200 "a quote-and-backslash node id registers" "$CODE"
curl -fsS "http://127.0.0.1:$PORT/v1/nodes" > "$TMP/nodes.json" 2>/dev/null
if command -v python3 >/dev/null 2>&1; then
  if python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
rows = d.get("nodes") if isinstance(d, dict) else d
ids = [r.get("node_id") if isinstance(r, dict) else r for r in (rows or [])]
sys.exit(0 if any(i and "DROP TABLE" in i for i in ids) else 1)
' "$TMP/nodes.json"; then
    ok "the hostile id round-trips through the node list"
  else
    bad "the hostile id did not survive the node list"
  fi
else
  ok "skipped the node-list check (no python3)"
fi
CODE=$(post /v1/jobs "{\"job_dir\":\"$TMP/bundle3\",\"replicas\":1}" -H "X-Grid-Token: $TOKEN")
check_code 202 "the jobs table still accepts work after the hostile id" "$CODE"

step "tokens do not leak"
for f in health summary nodes tasks; do
  curl -fsS "http://127.0.0.1:$PORT/v1/$f" > "$TMP/leak-$f.json" 2>/dev/null
done
curl -fsS "http://127.0.0.1:$PORT/" > "$TMP/leak-index.html" 2>/dev/null
LEAKED=0
for f in "$TMP"/leak-*; do
  if grep -q "$TOKEN" "$f" 2>/dev/null; then LEAKED=1; echo "   note  $f holds the admin token"; fi
  if grep -q "$A_TOK" "$f" 2>/dev/null; then LEAKED=1; echo "   note  $f holds a session token"; fi
done
[ "$LEAKED" = "0" ] && ok "no response carries a token" || bad "a token leaked into a response"

step "abuse does not take the coordinator down"
head -c 200000 /dev/zero | tr '\0' 'x' > "$TMP/junk.txt"
CODE=$(post_file /v1/register "$TMP/junk.txt")
check_code 400 "a 200 KiB junk body is refused" "$CODE"
CODE=$(getcode "/v1/jobs")
check_code 404 "a GET on the submit endpoint has no route" "$CODE"
CODE=$(getcode "/v1/health" -X DELETE)
check_code 404 "a DELETE on health has no route" "$CODE"
CODE=$(getcode "/assets/../../etc/passwd" --path-as-is)
if [ "$CODE" = "400" ] || [ "$CODE" = "404" ]; then
  ok "a traversal asset path is refused ($CODE)"
else
  bad "a traversal asset path returned $CODE"
fi
CODE=$(getcode /v1/health)
check_code 200 "the coordinator is still healthy after the abuse" "$CODE"

echo
if [ "$fail" -eq 0 ]; then
  echo "GRID SECURITY GATE: PASS"
else
  echo "GRID SECURITY GATE: FAILURES"
  echo "--- coordinator log ---"
  tail -20 "$TMP/orch.log"
fi
exit "$fail"
