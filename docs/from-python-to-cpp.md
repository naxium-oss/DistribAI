# From Python to C++

The project moved. The C++ port under [`tools/cpp_port`](../tools/cpp_port) is the
product, and the Python distributed stack under [`legacy/`](../legacy) is
deprecated. This page says what that means in practice and how to map an old
reference to its replacement.

## What "deprecated" means here

| Allowed on `legacy/` | Not allowed |
| --- | --- |
| Bug fixes that keep it building and passing | New features |
| Security fixes | New deployments |
| Keeping the tests green | New docs or guides (write them at the repo root instead) |
| Reading it to port a behaviour | Dispatching training jobs through it |

The Python train path is parked behind
[`legacy/services_python/train_path_guard.py`](../legacy/services_python/train_path_guard.py).
A dispatch attempt raises `LegacyTrainPathDisabledError` with an explanation. The
only bypass is `DISTRIBAI_ALLOW_LEGACY_TRAIN=1`, which exists so someone
debugging the frozen stack can still run it, and is not a supported configuration.

The stack stays in the tree for two reasons: forks depend on it, and its Node
dashboards and gRPC control plane are the reference for anything not yet
reimplemented. It has no schedule. Nothing new will be added to it.

## Capability map

| Capability | Python (legacy) | C++ (current) |
| --- | --- | --- |
| Training engine | `legacy/services_python/distributed_trainer.py`, torch | `tools/cpp_port/torch/torch_train.cpp`, LibTorch |
| Engine without torch | n/a | `tools/cpp_port/core/` parity MLP plus generic MLP |
| Control plane | `services_python/orchestrator_grpc.py`, gRPC | `tools/cpp_port/grid/orchestrator.cpp`, HTTP |
| Worker | `worker/src/daemon/run.py` | `tools/cpp_port/grid/worker.cpp` |
| Sandbox | `worker/src/sandbox/sandbox.py` | `tools/cpp_port/sandbox/sandbox.hpp` |
| State | SQLite through `services_python/db/` | `tools/cpp_port/grid/store.cpp`, same schema |
| Dashboard | `legacy/client/` and `worker/src/dashboard/` (Node) | `tools/cpp_port/grid/web/` (static, served by the coordinator) |
| Expose and join | `legacy/docs/guides/contributor-join-kit.md` | `gridlink expose` and `gridlink join`, native engine by default |
| Credits | `services_python` ledger | `credit_ledger` rows written by the coordinator |
| Python trainer support | run it directly | `tools/trainer_translate/` traces it into a job |

## Reference translation

| Old reference | Now |
| --- | --- |
| `python -m services_python.orchestrator_grpc` | `build/cpp_port/distribai_orch` |
| `python -m worker.src.daemon.run --orchestrator H:P` | `build/cpp_port/distribai_worker --orchestrator http://H:P` |
| `ORCHESTRATOR_URL`, `GRPC_USE_TLS`, `GRPC_TLS_CA` | `--orchestrator https://host` plus the system trust store |
| `DISTRIBAI_INVITE_CODE` | `--invite CODE` |
| `DISTRIBAI_EPHEMERAL=1` | `--work-dir` on scratch storage |
| `STATE_DIR` | `--work-dir` |
| `cd legacy && node client/server.js` | `http://127.0.0.1:50061/` |
| `python grid.py submit ...` | `distribai_orch --submit JOBS/DIR --replicas N` |
| gRPC proto in `legacy/proto/` | JSON over HTTP, see [the protocol table](../tools/cpp_port/grid/README.md) |

## When translation refuses a trainer

`trainer_translate` fails closed and explains itself. Three outcomes:

| Notice | Meaning | What to do |
| --- | --- | --- |
| Unsupported construct | The port cannot express something your script does | Add support under `tools/trainer_translate/` or `tools/cpp_port/torch/`, or simplify the script. Send a PR so the next person gets it |
| Script error | Your trainer is broken on its own terms | Fix the script; the notice names the failing call |
| Internal error | The translator broke | A bug in this repo. Report it with the command you ran |

In all three cases no job directory is written. A half-translated job would be
worse than a refusal, because it would train something other than what you
submitted.

## Porting a behaviour that is still Python only

1. Find the behaviour in `legacy/`. Its tests under `legacy/tests/` describe the
   contract better than the code does.
2. Decide where it belongs: engine maths in `tools/cpp_port/core/`, training in
   `torch/`, control plane in `grid/`, sandbox or rlimits in `sandbox/`.
3. Write the C++ version next to its neighbours and keep the naming style of that
   directory.
4. Add a gate, not just an example. If the behaviour feeds the wire format, it
   belongs in the must-match rows in `tests/test_golden.cpp`.
5. Add a row to the capability table above. That table is how the next reader
   learns whether a Python path is a reference or a leftover.

## Common questions

**Can I keep running the Python orchestrator?** It still builds and runs, and
nothing stops you. It is deprecated, it has no new features, and the C++ path
scales further: the coordinator starts in milliseconds, holds state in SQLite,
and its worker runs a trainer under rlimits with no interpreter between them.

**Why is the Python worker still in the tree if jobs never run there?** Its
sandbox and its dashboard are the reference implementations for the port. The
sandbox contract (which rlimits, which namespaces, how the child hands back a
result) was ported line by line, and the golden tests assert the match.

**Do I need Python at all to train?** Only to translate a trainer, because that
step runs `torch` on your script to capture the architecture. If you write the
job directory by hand (`job.json` plus a TorchScript module), you can run the
whole grid with no Python installed.

**What about the Node dashboards?** They still work if you install their
dependencies, and they are the reference for anything the new dashboard does not
show yet. The new dashboard needs no Node, no build step and no CDN; it is three
files served by the coordinator.
