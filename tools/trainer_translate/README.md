# trainer_translate: Python trainer to LibTorch job

**LibTorch is the only live train path.** The grid runs
`build/cpp_port/distribai_torch_train`; the frozen Python stack under
[`legacy/`](../../legacy/README.md) never runs jobs. This tool bridges the two:
it turns a user's Python training script, whatever the architecture, into a
C++/LibTorch job the grid can run.

## Usage

```bash
# from the repo root, with the shared .venv (needs torch)
PYTHONPATH=tools .venv/bin/python -m trainer_translate.translate \
    trainer.py --out jobs/my-job --steps 200

# run the emitted job
tools/bench/run_limited.sh build/cpp_port/distribai_torch_train \
    --spec jobs/my-job/job.json --json
```

Or via make:

```bash
make translate ARGS="trainer.py --out jobs/my-job --steps 200"
make translate-test        # fail-closed behaviour tests
```

### Options

| Flag | Meaning |
| --- | --- |
| `--out DIR` | job directory to create (must be empty unless `--force`) |
| `--steps N` | optimizer steps the job runs (default 200) |
| `--device auto\|cpu\|cuda` | requested device (default `auto`); `auto` and `cuda` prefer CUDA and fall back to CPU when it is unavailable, `cpu` stays on the CPU |
| `--input-shape` / `--target-shape` | override shapes, e.g. `3,32,32` |
| `--max-batches N` | how many loader batches to snapshot into `x.bin`/`y.bin` |
| `--no-sandbox` | skip `tools/bench/run_limited.sh` while capturing |
| `--json` | machine-readable result on stdout |

## How it works

1. **Capture** (`capture.py`) imports the trainer as `__main__` in a
   resource-limited child, wrapping `nn.Module.__init__`, `Optimizer.__init__`,
   `Optimizer.step`, `nn.modules.loss._Loss.__init__` and `DataLoader.__init__`
   to observe the real objects the script builds. Training is stopped after one
   optimizer step (`DISTRIBAI_TRANSLATE_STEPS`), so the whole job never runs.
2. **Script** the top-level model to TorchScript (`model.pt`). This is how
   *any* architecture is carried: the C++ runner loads the scripted module and
   drives it with `torch::optim`.
3. **Emit** `job.json` (optimizer/lr/loss/shapes/steps) plus an `x.bin`/`y.bin`
   data snapshot in the port's tiny `DAIT` tensor format, and `notes.md`.

Nothing is written to `--out` unless every stage succeeds.

## Framework support

Trainers written on **Megatron-LM**, **torchtitan** or **slime** normally own
their own process orchestration: they call
`torch.distributed.init_process_group` and expect a multi-rank world. The grid
owns orchestration instead, so a plain distributed trainer is refused (see
below). For those three frameworks the translator installs a single-rank shim:
`init_process_group` and the collectives become identities, the script builds
its model on one rank, and the emitted job carries a `framework` block:

```json
"framework": {
  "key": "megatron",
  "display": "Megatron-LM",
  "mode": "single_rank",
  "detected_via": "distributed init"
}
```

The framework is recognized from the modules the trainer imported, not from a
comment; source markers only count when the package is installed. Everything
after the shim (model, optimizer, loss, shapes, TorchScript) still goes through
the normal fail-closed path, so an adapter never emits a job it cannot run. When
a framework construct is unsupported, the notice names the framework and its
single-rank mapping. See `frameworks.py`.

The three adapters and their limits:

| Framework | Single-rank mapping |
| --- | --- |
| Megatron-LM | the `megatron.training` model with tensor/pipeline parallel size 1 |
| torchtitan | the `TrainSpec` model without DTensor or FSDP wrappers |
| slime | the policy model/optimizer/loss; the SGLang rollout half has no equivalent here |

## Fail-closed behaviour

If translation is impossible, the tool emits **no job**, exits `3`, and prints
an operator notice. Two cases:

* **unsupported**: the trainer is valid Python but uses a feature the
  translator (or the LibTorch runner) does not support. The notice lists the
  missing feature(s) and asks your **admin / a developer / an AI agent to add
  support for it here** (`tools/trainer_translate/` or
  `tools/cpp_port/torch/`). PRs are suggested so everyone gets the feature; a
  GitHub issue will be patched, but it takes a while.
* **needs_bugfix**: the script itself raised while being inspected, so it needs a
  bug fix before it can be translated. The notice includes the traceback.

## Data transport (`DAIT`)

Python's `torch.save` zip is not readable by `torch::load` in this LibTorch
build, so the translator and runner share a 20-line format:

```
"DAIT" | u32 version=1 | u8 dtype(0=f32,1=i64) | u8 ndim | ndim*u64 dims | data
```

little-endian, row-major, contiguous. See `read_tensor_bin` in
[`torch_train.cpp`](../cpp_port/torch/torch_train.cpp).
