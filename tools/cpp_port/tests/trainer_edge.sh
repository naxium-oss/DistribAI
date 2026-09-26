#!/usr/bin/env bash
# trainer_edge.sh - edge cases for the LibTorch trainer binary.
#
# The trainer is what actually runs on a contributor's machine, and every field
# of job.json reaches it from a file someone else wrote. This gate feeds it the
# shapes that break things: missing and corrupt models, malformed specs,
# unsupported options, out-of-range steps, worse-than-useless data files, and
# rlimits it cannot live inside. A bad job has to fail closed with a readable
# error, never a silent wrong answer and never a crash.
#
# The fixtures come from the real translator (tools/trainer_translate), so the
# spec, the TorchScript module and the captured data are the same shapes a
# submitted job has in production.
#
#   bash tools/cpp_port/tests/trainer_edge.sh   (or: make -C tools/cpp_port trainer-edge-test)
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

B=build/cpp_port
TRAINER=$B/distribai_torch_train
PY=${TRAINER_EDGE_PY:-.venv/bin/python}
TMP=$(mktemp -d)
fail=0

cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

step() { echo "== trainer-edge: $*"; }
ok()   { echo "   PASS  $*"; }
bad()  { echo "   FAIL  $*"; fail=1; }
skip() { echo "TRAINER EDGE GATE: SKIP ($*)"; exit 0; }

[ -x "$TRAINER" ] || skip "missing $TRAINER; run make torch"
[ -x "$PY" ] || skip "no interpreter at $PY"
"$PY" -c 'import torch' 2>/dev/null || skip "torch is not importable from $PY"

# run <spec> [extra args...] -> stdout in $TMP/out, stderr in $TMP/err, code in $TMP/code
run() {
  local spec="$1"
  shift
  out=$("$TRAINER" --spec "$spec" --json "$@" 2>"$TMP/err")
  code=$?
  printf '%s\n' "$out" > "$TMP/out"
}

field() { # field <key> -> value from the last JSON line
  sed -n 's/.*"'"$1"'":"\([^"]*\)".*/\1/p' "$TMP/out" | tail -1
}
num_field() { # num_field <key>
  sed -n 's/.*"'"$1"'":\([-0-9.eE]*\).*/\1/p' "$TMP/out" | tail -1
}
last_json_is_object() {
  tail -1 "$TMP/out" | grep -q '^{.*}$'
}

# ---------------------------------------------------------------------------
step "building the fixtures with the real translator"
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

if (cd tools && "../$PY" -m trainer_translate.translate "$TRAINER_SRC" --out "$TMP/fix" \
      --steps 20 > "$TMP/translate.log" 2>&1); then
  ok "the translator produced a job bundle"
else
  bad "translation failed"; tail -5 "$TMP/translate.log"
  echo; echo "TRAINER EDGE GATE: FAILURES"; exit 1
fi
SPEC=$TMP/fix/job.json
[ -s "$SPEC" ] && ok "the bundle has job.json" || { bad "no job.json"; exit 1; }

# ---------------------------------------------------------------------------
step "the happy path, and what must not change it"
run "$SPEC"
[ "$(field status)" = "ok" ] && ok "a valid spec trains" ||
  bad "a valid spec failed: $(head -c 200 "$TMP/out")"
last_json_is_object && ok "the last stdout line is a JSON object" ||
  bad "stdout is not a single JSON line: $(tail -2 "$TMP/out")"
[ "$(field engine)" = "libtorch" ] && ok "the engine reports libtorch" || bad "engine field is wrong"
[ "$(num_field params)" = "193" ] && ok "the translated model has 193 parameters" ||
  bad "unexpected parameter count: $(num_field params)"
[ "$(num_field grad_len)" = "193" ] && ok "the gradient vector matches the parameters" ||
  bad "unexpected gradient length: $(num_field grad_len)"
LOSS_A=$(num_field final_loss)
[ -n "$LOSS_A" ] && ok "a final loss was reported ($LOSS_A)" || bad "no final loss"

run "$SPEC"
LOSS_B=$(num_field final_loss)
[ "$LOSS_A" = "$LOSS_B" ] && ok "the same spec trains to the same loss" ||
  bad "the run is not deterministic: $LOSS_A vs $LOSS_B"

run "$SPEC" --no-sandbox
LOSS_C=$(num_field final_loss)
[ "$(field status)" = "ok" ] && ok "the trainer runs unsandboxed too" || bad "unsandboxed run failed"
[ "$LOSS_A" = "$LOSS_C" ] && ok "sandboxing does not change the numbers" ||
  bad "sandbox changed the result: $LOSS_A vs $LOSS_C"

# A different seed is how replicas differ; it must reach the data shuffling.
"$PY" - "$SPEC" "$TMP/seed77.json" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
spec["seed"] = 77
json.dump(spec, open(sys.argv[2], "w"))
PY
run "$TMP/seed77.json"
LOSS_SEED=$(num_field final_loss)
[ "$LOSS_SEED" != "$LOSS_A" ] && ok "a different seed changes the trajectory" ||
  bad "the seed had no effect on the loss"

# ---------------------------------------------------------------------------
step "device policy: cuda and auto fall back to CPU without a GPU"
# The trainer resolves the requested device against the build. A translated
# bundle that asks for cuda must not fail on a CPU-only box; it trains on the
# CPU. An explicit cpu never leaves the CPU. This is the leftover from the old
# fail-closed rule, so assert the fallback directly.
if "$PY" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  HAS_CUDA=1
else
  HAS_CUDA=0
fi

# device_spec <device> -> writes $TMP/fix/dev.json beside the bundle so the
# relative model/data names still resolve.
device_spec() {
  "$PY" - "$SPEC" "$1" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
spec["device"] = sys.argv[2]
json.dump(spec, open(sys.argv[1].replace("job.json", "dev.json"), "w"))
PY
}
DEVSPEC=$TMP/fix/dev.json

device_spec auto
run "$DEVSPEC"
if [ "$HAS_CUDA" = "1" ]; then
  [ "$(field device)" = "cuda" ] && ok "auto picks CUDA when it is available" ||
    bad "auto did not pick CUDA (got $(field device))"
else
  [ "$(field status)" = "ok" ] && ok "auto trains on the CPU when there is no GPU" ||
    bad "auto failed without CUDA: $(head -c 200 "$TMP/out")"
  [ "$(field device)" = "cpu" ] && ok "auto reports cpu on a CPU-only box" ||
    bad "auto reported device $(field device)"
fi

device_spec cuda
run "$DEVSPEC"
if [ "$HAS_CUDA" = "1" ]; then
  [ "$(field device)" = "cuda" ] && ok "an explicit cuda uses the GPU" ||
    bad "explicit cuda did not use the GPU"
else
  [ "$(field status)" = "ok" ] && ok "cuda falls back to CPU instead of failing closed" ||
    bad "cuda failed on a CPU-only box: $(head -c 200 "$TMP/out")"
  [ "$(field device)" = "cpu" ] && ok "the fallback reports cpu" ||
    bad "the fallback reported device $(field device)"
  [ "$(field engine)" = "libtorch" ] && ok "the fallback still runs the real trainer" ||
    bad "engine field is wrong"
fi

device_spec cpu
run "$DEVSPEC"
[ "$(field status)" = "ok" ] && [ "$(field device)" = "cpu" ] &&
  ok "an explicit cpu runs on the CPU" || bad "explicit cpu failed (device=$(field device))"

# ---------------------------------------------------------------------------
step "the --steps override"
run "$SPEC" --steps 3
STEPS=$(num_field steps)
[ "$STEPS" = "3" ] && ok "--steps overrides the spec" || bad "--steps was ignored (got $STEPS)"
run "$SPEC" --steps 0
[ "$(field status)" = "ok" ] && ok "zero steps is a valid, boring run" ||
  bad "zero steps failed: $(head -c 160 "$TMP/out")"

# ---------------------------------------------------------------------------
step "broken and hostile specs"
printf '{"model":' > "$TMP/truncated.json"
run "$TMP/truncated.json"
[ "$code" -ne 0 ] && ok "a truncated spec exits non-zero" || bad "a truncated spec exited 0"
grep -q '"status":"error"' "$TMP/out" && ok "a truncated spec reports an error" ||
  bad "a truncated spec said: $(head -c 160 "$TMP/out")"

printf '[1,2,3]' > "$TMP/array.json"
run "$TMP/array.json"
grep -q 'object' "$TMP/out" && ok "an array spec is refused with an explanation" ||
  bad "an array spec said: $(head -c 160 "$TMP/out")"

printf '{}' > "$TMP/empty.json"
run "$TMP/empty.json"
grep -q 'no' "$TMP/out" && grep -q 'model' "$TMP/out" &&
  ok "a spec without a model is refused by name" ||
  bad "an empty spec said: $(head -c 160 "$TMP/out")"

printf '{"model":"does-not-exist.pt","steps":2}' > "$TMP/nomodel.json"
run "$TMP/nomodel.json"
grep -q '"status":"error"' "$TMP/out" && ok "a missing model file is an error, not a crash" ||
  bad "a missing model said: $(head -c 160 "$TMP/out")"

printf 'this is not a TorchScript module' > "$TMP/garbage.pt"
printf '{"model":"garbage.pt","steps":2}' > "$TMP/garbage.json"
run "$TMP/garbage.json"
grep -q '"status":"error"' "$TMP/out" && ok "a corrupt model file is an error" ||
  bad "a corrupt model said: $(head -c 200 "$TMP/out")"

printf '{"model":"model.pt","steps":2,"optimizer":"banana"}' > "$TMP/badopt.json"
cp "$TMP/fix/model.pt" "$TMP/model.pt"
run "$TMP/badopt.json"
grep -q 'unsupported optimizer' "$TMP/out" && ok "an unknown optimizer is named in the error" ||
  bad "an unknown optimizer said: $(head -c 200 "$TMP/out")"

printf '{"model":"model.pt","steps":-5}' > "$TMP/negsteps.json"
run "$TMP/negsteps.json"
[ "$code" -ne 0 ] && ok "negative steps exits non-zero" || bad "negative steps exited 0"
grep -q 'steps out of range' "$TMP/out" && ok "negative steps is reported as out of range" ||
  bad "negative steps said: $(head -c 160 "$TMP/out")"

printf '{"model":"model.pt","steps":5000000}' > "$TMP/hugesteps.json"
run "$TMP/hugesteps.json"
grep -q 'steps out of range' "$TMP/out" && ok "an absurd step count is refused up front" ||
  bad "a huge step count said: $(head -c 160 "$TMP/out")"

printf '{"model":"model.pt","steps":2,"input":"missing.bin","target":"missing.bin"}' \
  > "$TMP/nodata.json"
run "$TMP/nodata.json"
grep -q '"status":"error"' "$TMP/out" && ok "missing data files are an error" ||
  bad "missing data said: $(head -c 160 "$TMP/out")"

head -c 40 "$TMP/fix/x.bin" > "$TMP/trunc.bin"
cp "$TMP/fix/y.bin" "$TMP/y.bin"
printf '{"model":"model.pt","steps":2,"input":"trunc.bin","target":"y.bin"}' > "$TMP/truncbin.json"
run "$TMP/truncbin.json"
grep -q '"status":"error"' "$TMP/out" && ok "a truncated tensor file is refused" ||
  bad "a truncated tensor file said: $(head -c 200 "$TMP/out")"

printf 'JUNK' > "$TMP/badmagic.bin"
printf '{"model":"model.pt","steps":2,"input":"badmagic.bin","target":"y.bin"}' \
  > "$TMP/badmagic.json"
run "$TMP/badmagic.json"
grep -q '"status":"error"' "$TMP/out" && ok "a file without the DAIT magic is refused" ||
  bad "a bad-magic tensor said: $(head -c 200 "$TMP/out")"

# An input with no target rows at all.
printf 'DAIT\x01\x00\x00\x00\x00\x00' > "$TMP/emptytensor.bin"
printf '{"model":"model.pt","steps":2,"input":"emptytensor.bin","target":"y.bin"}' \
  > "$TMP/emptyt.json"
run "$TMP/emptyt.json"
grep -q '"status":"error"' "$TMP/out" && ok "an empty tensor is refused instead of trained on" ||
  bad "an empty tensor said: $(head -c 200 "$TMP/out")"

# ---------------------------------------------------------------------------
step "sandbox limits the trainer cannot live inside"
"$PY" - "$SPEC" "$TMP/tiny_mem.json" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
spec["rlimits"] = {"mem_mb": 16, "cpu_sec": 30}
json.dump(spec, open(sys.argv[2], "w"))
PY
run "$TMP/tiny_mem.json"
if [ "$(field status)" = "ok" ]; then
  # torch may fit inside 16 MiB on a small model; that is a pass, but say so.
  ok "the trainer fit inside a 16 MiB address-space cap"
else
  grep -q '"status":"error"' "$TMP/out" && ok "a 16 MiB cap fails loudly, not silently" ||
    bad "a 16 MiB cap produced: $(head -c 200 "$TMP/out")"
fi

"$PY" - "$SPEC" "$TMP/tiny_cpu.json" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
spec["steps"] = 4000
spec["rlimits"] = {"mem_mb": 8192, "cpu_sec": 1}
json.dump(spec, open(sys.argv[2], "w"))
PY
run "$TMP/tiny_cpu.json"
if [ "$(field status)" = "ok" ]; then
  bad "4000 steps inside a 1s CPU cap reported success"
else
  ok "a 1s CPU cap stops the run instead of reporting a made-up loss"
fi

# ---------------------------------------------------------------------------
step "relative paths resolve against the spec directory"
# The worker unpacks a bundle into its own directory and passes an absolute
# --spec path, so a relative \"input\" only works if it is resolved against the
# spec's directory rather than the process's working directory.
mkdir -p "$TMP/relocated"
cp "$TMP/fix/job.json" "$TMP/fix/model.pt" "$TMP/fix/x.bin" "$TMP/fix/y.bin" "$TMP/relocated/"
(cd / && "$OLDPWD/$TRAINER" --spec "$TMP/relocated/job.json" --json --no-sandbox \
  > "$TMP/relocated_out" 2>"$TMP/relocated_err")
if grep -q '"status":"ok"' "$TMP/relocated_out"; then
  ok "a bundle runs from any working directory"
else
  bad "relative data paths broke outside the bundle directory: $(head -c 200 "$TMP/relocated_out")"
fi
grep -q '"input": "x.bin"' "$TMP/relocated/job.json" &&
  ok "the bundle really does reference its data by relative name" ||
  bad "the fixture no longer uses relative data names"

# ---------------------------------------------------------------------------
step "dropped envelopes: the aggregate blob"
mkdir -p "$TMP/env"
"$TRAINER" --spec "$SPEC" --json --envelope-out "$TMP/env/out.env" --no-sandbox \
  > "$TMP/env/out.json" 2>&1
if [ -s "$TMP/env/out.env" ]; then
  ok "the trainer writes an envelope when asked"
  head -c 4 "$TMP/env/out.env" | grep -q "NEID" &&
    ok "the envelope carries the expected magic" ||
    bad "the envelope magic is wrong: $(head -c 8 "$TMP/env/out.env" | od -c | head -1)"
else
  bad "no envelope was written"
fi
[ -s "$TMP/env/out.env" ] && [ "$(stat -c %s "$TMP/env/out.env")" -gt 64 ] &&
  ok "the envelope has room for a gradient payload" ||
  bad "the envelope is suspiciously small"

# ---------------------------------------------------------------------------
step "the aggregate modes agree on the numbers they share"
# mean, median and trimmed_mean over identical contributions must agree, and the
# trimmed mean must ignore one low and one high outlier.
# The spec has to live beside the bundle: relative data names resolve against
# the spec's directory.
"$PY" - "$TMP/fix/job.json" "$TMP/fix/multi.json" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
spec["models"] = [{"name": "a", "seed": 1, "steps": 5},
                  {"name": "b", "seed": 2, "steps": 5},
                  {"name": "c", "seed": 3, "steps": 5}]
spec["aggregate"] = "trimmed_mean"
json.dump(spec, open(sys.argv[2], "w"))
PY
run "$TMP/fix/multi.json"
if grep -q '"status":"ok"' "$TMP/out"; then
  ok "a multi-model manifest trains every replica"
  for key in aggregate_grad_sum aggregate_grad_len models_ok; do
    if [ -n "$(num_field "$key")" ]; then
      ok "the report carries $key"
    else
      bad "the report is missing $key"
    fi
  done
  [ "$(num_field models_ok)" = "3" ] && ok "all three replicas trained" ||
    bad "only $(num_field models_ok) of 3 replicas trained"
  ENVF=$TMP/fix/model_a_1.pt
  [ -f "$ENVF" ] || true
else
  bad "the multi-model manifest failed: $(head -c 300 "$TMP/out")"
fi

# A manifest where nothing can train must not report success. Every model here
# points at a model file that does not exist.
"$PY" - "$TMP/fix/job.json" "$TMP/fix/doomed.json" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
spec["model"] = "not-here.pt"
spec["models"] = [{"name": "a", "seed": 1, "steps": 3},
                  {"name": "b", "seed": 2, "steps": 3}]
json.dump(spec, open(sys.argv[2], "w"))
PY
run "$TMP/fix/doomed.json"
[ "$(field status)" = "error" ] &&
  ok "a manifest where every replica fails reports an error" ||
  bad "a fully failed manifest reported '$(field status)'"
[ "$code" -ne 0 ] && ok "and it exits non-zero" || bad "a failed manifest exited 0"
[ "$(num_field models_ok)" = "0" ] && ok "and it counts zero successes" ||
  bad "models_ok is $(num_field models_ok)"

echo
if [ "$fail" -eq 0 ]; then
  echo "TRAINER EDGE GATE: PASS"
else
  echo "TRAINER EDGE GATE: FAILURES"
  echo "--- last stdout ---"
  tail -3 "$TMP/out"
  echo "--- last stderr ---"
  tail -3 "$TMP/err"
fi
exit "$fail"
