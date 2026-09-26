# DistribAI docs

The project is C++ now. These pages describe the system as it is, and the
[legacy Python stack](#legacy) only as something being retired.

| Page | Read it when |
| --- | --- |
| [Quickstart](quickstart.md) | you want a job running on your own machine today |
| [Architecture](architecture.md) | you need to know what talks to what, or where state lives |
| [Operations](operations.md) | you run a grid people connect to |
| [Testing](testing.md) | you changed something and want to know what proves it |
| [From Python to C++](from-python-to-cpp.md) | you hit a Python-era script, doc or path |

## The short version

Three things run in production:

- **`distribai_orch`** coordinates. It keeps the node registry, the job queue
  and the credit ledger in SQLite, serves the worker API, the operator API and
  the [dashboard](../tools/cpp_port/grid/web) from one HTTP port, and aggregates
  the gradients that come back from workers.
- **`distribai_worker`** runs work. It registers, claims one replica at a time,
  runs the trainer as a limited child, and reports a GradReport envelope.
- **`distribai_torch_train`** trains. It loads a TorchScript module plus a job
  spec and trains with LibTorch inside the native sandbox.

Everything else is either a build input (the port's engine, sandbox, envelope
and ABI) or a thin Python utility: `tools/trainer_translate` turns a Python
trainer into a job directory, and `tools/gridlink` publishes the grid and prints
a join link.

```text
Python trainer  --trainer_translate-->  job directory  --submit-->  coordinator
                                                                        |
                                              worker (anywhere) <--claim--
                                                                        |
                                    GradReport envelope --> aggregate --> result
```

## Legacy

`legacy/` holds the original Python distributed stack: the gRPC orchestrator, the
worker daemon, the Node dashboards, their tests and their docs. It still builds
and its tests still pass, because a fork somewhere depends on it. It is
**deprecated**: bug fixes only, no new features, and no new deployments. The
Python train path inside it is parked, and dispatching a job through it raises
unless you set `DISTRIBAI_ALLOW_LEGACY_TRAIN=1`.

See [From Python to C++](from-python-to-cpp.md) for the mapping and the policy.
