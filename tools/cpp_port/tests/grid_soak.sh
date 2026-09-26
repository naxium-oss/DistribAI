#!/usr/bin/env bash
# grid_soak.sh - a load gate: many workers, many jobs, one coordinator.
#
# grid_e2e.sh proves one job works. This proves the coordinator holds up when
# work arrives in a burst from a crowd: a batch of jobs submitted back to back,
# more workers than a single job needs, every task claimed exactly once, and a
# ledger that adds up at the end. Nothing is mocked: real coordinator, real
# worker processes, real SQLite, real HTTP over loopback, and the fixture
# trainer behind the same envelope contract as LibTorch.
#
# The size is tunable, so a slow box can shrink it and CI can keep the default:
#   GRID_SOAK_WORKERS (default 6), GRID_SOAK_JOBS (default 12),
#   GRID_SOAK_REPLICAS (default 2)
#
#   bash tools/cpp_port/tests/grid_soak.sh   (or: make -C tools/cpp_port grid-soak-test)
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

B=build/cpp_port
ORCH=$B/distribai_orch
WORKER=$B/distribai_worker
TRAINER=$B/grid_fake_trainer
WORKERS=${GRID_SOAK_WORKERS:-6}
JOBS=${GRID_SOAK_JOBS:-12}
REPLICAS=${GRID_SOAK_REPLICAS:-2}
STEPS=8
PORT=${GRID_SOAK_PORT:-$((20000 + (RANDOM % 9000)))}
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

step() { echo "== grid-soak: $*"; }
ok()   { echo "   PASS  $*"; }
bad()  { echo "   FAIL  $*"; fail=1; }

for bin in "$ORCH" "$WORKER" "$TRAINER"; do
  [ -x "$bin" ] || { echo "missing $bin (run: make -C tools/cpp_port grid-test)"; exit 1; }
done

# Refuse to run against a grid this script did not start.
if curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; then
  echo "port $PORT already answers a grid; stop it or set GRID_SOAK_PORT"
  exit 1
fi

job_status() { # job_status <job_id>
  curl -fsS "http://127.0.0.1:$PORT/v1/jobs/$1" 2>/dev/null |
    grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4
}
count_field() { # count_field <file> <field>
  grep -o "\"$2\"" "$1" 2>/dev/null | wc -l | tr -d ' '
}

step "starting the coordinator on 127.0.0.1:$PORT"
"$ORCH" --host 127.0.0.1 --port "$PORT" \
        --db "$TMP/grid.db" --schema runtime/db/schema.sql \
        --jobs-dir "$TMP/jobs" --tasks-dir "$TMP/tasks" \
        --web-dir tools/cpp_port/grid/web \
        --invite soak-team --token soak-secret \
        --node-ttl 60 --aggregate mean \
        > "$TMP/orch.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && break
  sleep 0.2
done
if curl -fsS "http://127.0.0.1:$PORT/v1/health" > "$TMP/health.json" 2>/dev/null; then
  ok "coordinator is up"
else
  bad "coordinator never came up"
  sed -n 1,20p "$TMP/orch.log"
  exit 1
fi

step "starting $WORKERS workers"
for i in $(seq 1 "$WORKERS"); do
  "$WORKER" --orchestrator "http://127.0.0.1:$PORT" --invite soak-team \
            --node-id "soak-node-$i" --trainer "$TRAINER" \
            --work-dir "$TMP/w$i" --mem-mb 256 --cpu-sec 60 --max-jobs 100 \
            > "$TMP/worker$i.log" 2>&1 &
  PIDS+=($!)
done
REGISTERED=0
for _ in $(seq 1 100); do
  ONLINE=$(curl -fsS "http://127.0.0.1:$PORT/v1/health" 2>/dev/null |
             grep -o '"nodes_online":[0-9]*' | cut -d: -f2)
  if [ "${ONLINE:-0}" -ge "$WORKERS" ]; then REGISTERED=1; break; fi
  sleep 0.2
done
[ "$REGISTERED" = "1" ] && ok "all $WORKERS workers registered" ||
  bad "only ${ONLINE:-0} of $WORKERS workers registered"

step "submitting $JOBS jobs with $REPLICAS replicas each"
IDS=()
for n in $(seq 1 "$JOBS"); do
  dir="$TMP/bundle-$n"
  mkdir -p "$dir"
  cat > "$dir/job.json" <<JSON
{
  "job_id": "soak-$n",
  "name": "soak-model-$n",
  "steps": $STEPS,
  "batch_size": 4,
  "seed": $((100 + n)),
  "model": "model.pt",
  "optimizer": "adamw",
  "loss": "mse",
  "aggregate": "mean"
}
JSON
  printf 'fixture' > "$dir/model.pt"
  out=$("$ORCH" --db "$TMP/grid.db" --schema runtime/db/schema.sql \
                 --jobs-dir "$TMP/jobs" --tasks-dir "$TMP/tasks" \
                 --submit "$dir" --replicas "$REPLICAS" 2>&1)
  id=$(printf '%s' "$out" | sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p')
  if [ -n "$id" ]; then IDS+=("$id"); else bad "submit $n failed: $out"; fi
done
[ "${#IDS[@]}" = "$JOBS" ] && ok "all $JOBS jobs accepted" ||
  bad "only ${#IDS[@]} of $JOBS jobs were accepted"

step "waiting for every job to finish"
DONE=0
for _ in $(seq 1 1200); do
  DONE=0
  for id in "${IDS[@]}"; do
    [ "$(job_status "$id")" = "completed" ] && DONE=$((DONE + 1))
  done
  [ "$DONE" = "$JOBS" ] && break
  sleep 0.25
done
[ "$DONE" = "$JOBS" ] && ok "all $JOBS jobs completed" || bad "$DONE of $JOBS jobs completed"

# Give the coordinator a moment to write the last result and credits.
sleep 0.5
curl -fsS "http://127.0.0.1:$PORT/v1/summary" > "$TMP/summary.json" 2>/dev/null

step "checking every result"
GOOD_RESULTS=0
for id in "${IDS[@]}"; do
  RESULT="$TMP/jobs/$id/result.json"
  if [ -s "$RESULT" ] && grep -q "\"contributors\":$REPLICAS" "$RESULT" &&
     grep -q '"status":"completed"' "$RESULT" && [ -s "$TMP/jobs/$id/aggregate.env" ]; then
    GOOD_RESULTS=$((GOOD_RESULTS + 1))
  fi
done
[ "$GOOD_RESULTS" = "$JOBS" ] && ok "all $JOBS results carry $REPLICAS contributors" ||
  bad "only $GOOD_RESULTS of $JOBS results look complete"

grep -q '"aggregate":"mean"' "$TMP/jobs/${IDS[0]}/result.json" &&
  ok "the per-job aggregate method is recorded" ||
  bad "the aggregate method is missing from the result"

step "checking the queue drained"
QUEUED=$(grep -o '"tasks_queued":[0-9]*' "$TMP/summary.json" | cut -d: -f2)
ASSIGNED=$(grep -o '"status":"assigned"' "$TMP/summary.json" | wc -l | tr -d ' ')
[ "${QUEUED:-1}" = "0" ] && ok "no tasks are left queued" || bad "$QUEUED tasks are still queued"
[ "$ASSIGNED" = "0" ] && ok "no tasks are left assigned" || bad "$ASSIGNED tasks are still assigned"

step "checking the ledger and the counters"
REWARDS=$(count_field "$TMP/summary.json" "job_reward")
EXPECTED=$((JOBS * REPLICAS))
# The summary shows the most recent 25 ledger rows; the leaderboard below is
# what has to add up no matter how much work ran.
if [ "$EXPECTED" -le 25 ]; then
  [ "$REWARDS" = "$EXPECTED" ] && ok "$EXPECTED job rewards recorded" ||
    bad "expected $EXPECTED rewards, found $REWARDS"
else
  [ "$REWARDS" = "25" ] && ok "the ledger window shows its 25-row cap" ||
    bad "expected the 25-row ledger window, found $REWARDS"
fi

BOARD=$(grep -o '"node_id":"soak-node-[0-9]*","credits"' "$TMP/summary.json" | wc -l | tr -d ' ')
[ "$BOARD" = "$WORKERS" ] && ok "every one of the $WORKERS workers earned credits" ||
  bad "the leaderboard holds $BOARD of $WORKERS workers"

TOTAL_CREDITS=$(grep -o '"credits":[0-9.]*' "$TMP/summary.json" |
                  sed 's/.*://' | awk '{s+=$1} END {printf "%.6f", s+0}')
EXPECTED_CREDITS=$(awk -v j="$JOBS" -v r="$REPLICAS" -v s="$STEPS" \
                       'BEGIN {printf "%.6f", j*r*s/1000}')
if awk -v a="$TOTAL_CREDITS" -v b="$EXPECTED_CREDITS" \
       'BEGIN {d=a-b; if (d<0) d=-d; exit !(d <= 0.0001)}'; then
  ok "credits total $TOTAL_CREDITS matches the expected $EXPECTED_CREDITS"
else
  bad "credits total $TOTAL_CREDITS does not match the expected $EXPECTED_CREDITS"
fi

JOBS_DONE=$(grep -o '"jobs_done":[0-9]*' "$TMP/summary.json" | cut -d: -f2)
[ "${JOBS_DONE:-0}" = "$JOBS" ] && ok "the summary counts $JOBS completed jobs" ||
  bad "the summary counts ${JOBS_DONE:-0} completed jobs"

step "checking the workers stayed healthy"
BAD_LOGS=0
for i in $(seq 1 "$WORKERS"); do
  if grep -q "fork() failed\|could not be reported\|transport error" "$TMP/worker$i.log"; then
    BAD_LOGS=$((BAD_LOGS + 1))
    echo "   note  worker$i log: $(grep -m1 'fork() failed\|could not be reported' "$TMP/worker$i.log")"
  fi
done
[ "$BAD_LOGS" = "0" ] && ok "no worker reported a fork or report failure" ||
  bad "$BAD_LOGS worker logs hold errors"

if curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; then
  ok "the coordinator is still healthy after the burst"
else
  bad "the coordinator stopped answering"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "GRID SOAK GATE: PASS"
else
  echo "GRID SOAK GATE: FAILURES"
  echo "--- coordinator log ---"
  tail -20 "$TMP/orch.log"
fi
exit "$fail"
