#!/usr/bin/env bash
# screenshots.sh - regenerate the dashboard images the README shows.
#
# The pictures in docs/assets are screenshots of the real dashboard reading a
# real coordinator, not mockups. This script starts a coordinator, submits a few
# jobs, brings up five workers (one of them on a trainer that holds its replica
# open, so the page shows a job in flight), then captures the page at desktop and
# phone widths with headless Chrome.
#
#   bash tools/cpp_port/grid/screenshots.sh    (or: make screenshots)
#
# Needs google-chrome or chromium on PATH. Without one it says so and stops.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

B=build/cpp_port
OUT=${SCREENSHOT_DIR:-docs/assets}
PORT=${SCREENSHOT_PORT:-$((52000 + (RANDOM % 2000)))}
TMP=$(mktemp -d)
PIDS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null
  done
  pkill -f "distribai_worker.*--orchestrator http://127.0.0.1:$PORT" 2>/dev/null
  sleep 0.3
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
  done
  rm -rf "$TMP"
}
trap cleanup EXIT

CHROME=""
for candidate in google-chrome google-chrome-stable chromium chromium-browser; do
  if command -v "$candidate" >/dev/null 2>&1; then
    CHROME=$(command -v "$candidate")
    break
  fi
done
[ -n "$CHROME" ] || { echo "no chrome/chromium on PATH; install one to capture screenshots"; exit 1; }

for bin in "$B/distribai_orch" "$B/distribai_worker" "$B/grid_fake_trainer"; do
  [ -x "$bin" ] || { echo "missing $bin (run: make -C tools/cpp_port grid-test)"; exit 1; }
done

if curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; then
  echo "port $PORT already answers a grid; set SCREENSHOT_PORT"
  exit 1
fi

mkdir -p "$OUT" "$TMP/bundle"
printf 'fixture' > "$TMP/bundle/model.pt"

write_bundle() { # write_bundle <name> <steps> <aggregate>
  cat > "$TMP/bundle/job.json" <<JSON
{
  "job_id": "demo",
  "name": "$1",
  "steps": $2,
  "batch_size": 64,
  "seed": 42,
  "model": "model.pt",
  "optimizer": "adamw",
  "loss": "mse",
  "aggregate": "$3",
  "translation": {"tool": "tools/trainer_translate", "source_script": "train.py"}
}
JSON
}

submit() { # submit <replicas>
  "$B/distribai_orch" --db "$TMP/grid.db" --schema runtime/db/schema.sql \
    --jobs-dir "$TMP/jobs" --tasks-dir "$TMP/tasks" --submit "$TMP/bundle" \
    --replicas "$1" > /dev/null 2>&1
}

echo "== starting a coordinator on 127.0.0.1:$PORT"
"$B/distribai_orch" --host 127.0.0.1 --port "$PORT" --db "$TMP/grid.db" \
  --schema runtime/db/schema.sql --jobs-dir "$TMP/jobs" --tasks-dir "$TMP/tasks" \
  --web-dir tools/cpp_port/grid/web --invite demo-team --token demo-secret \
  --aggregate trimmed_mean > "$TMP/orch.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && break
  sleep 0.2
done

echo "== submitting work"
write_bundle "mnist-mlp" 300 "trimmed_mean"
submit 3
write_bundle "vit-tiny" 420 "mean"
submit 2

# node-3 runs a trainer that sleeps, so one replica stays in flight and the
# dashboard has something to show under "running".
cat > "$TMP/slow_trainer.sh" <<CAP
#!/usr/bin/env bash
sleep 600
exec "$(pwd)/$B/grid_fake_trainer" "\$@"
CAP
chmod +x "$TMP/slow_trainer.sh"

echo "== starting five workers"
for i in 1 2 3 4 5; do
  trainer="$B/grid_fake_trainer"
  [ "$i" = 3 ] && trainer="$TMP/slow_trainer.sh"
  "$B/distribai_worker" --orchestrator "http://127.0.0.1:$PORT" --invite demo-team \
    --node-id "node-$i" --trainer "$trainer" --work-dir "$TMP/w$i" \
    > "$TMP/w$i.log" 2>&1 &
  PIDS+=($!)
done
sleep 3

write_bundle "llama-tiny" 5000 "median"
submit 3
sleep 2

capture() { # capture <file> <width> <height>
  echo "== capturing $1"
  "$CHROME" --headless=new --disable-gpu --no-sandbox --disable-dev-shm-usage \
    --hide-scrollbars --force-device-scale-factor=2 --window-size="$2,$3" \
    --virtual-time-budget=5000 --screenshot="$1" "http://127.0.0.1:$PORT/" \
    > "$TMP/chrome.log" 2>&1 || {
      echo "chrome failed:"; tail -3 "$TMP/chrome.log"; return 1
    }
  [ -s "$1" ] || { echo "$1 is empty"; return 1; }
}

capture "$OUT/dashboard-desktop.png" 1440 1220
capture "$OUT/dashboard-phone.png" 430 1500

curl -fsS "http://127.0.0.1:$PORT/v1/summary" > "$TMP/summary.json" 2>/dev/null
echo "== captured state"
head -c 300 "$TMP/summary.json"
echo
ls -la "$OUT"/*.png
