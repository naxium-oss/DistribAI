# Testing

Every gate runs through `tools/bench/run_limited.sh`, which caps address space,
CPU time, threads and process count before the real work starts. A gate that
ignores its budget fails instead of taking the box down.

## The gates

| Gate | Command | What it proves |
| --- | --- | --- |
| Golden contracts | `make test-golden` | 29 must-match behaviours: tensor math, envelope framing, sandbox limits, COW isolation, RNG separation |
| Port parity | `make test` | the C++ engine matches the recorded parity vectors |
| Differential | `.venv/bin/python tools/cpp_port/tests/suite/differential_suite.py` | Python and C++ agree on the same fixed corpus, ten seeds |
| LibTorch parity | `make torch-test` | Python torch and C++ LibTorch train the same model on the same data to about 1e-10 |
| Translator | `make translate-test` | a supported trainer produces a runnable job, and every unsupported or buggy input fails closed |
| Grid end to end | `make grid-test` | a real coordinator and two real workers complete a job, aggregate it, pay credits, and recover a task from a dead node |
| Grid edge cases | `make grid-edge-test` | 98 checks over the real binaries: malformed worker messages, invite and token gates, stale sessions, static asset abuse, refused bundles, failing workers, cancelled jobs, restarts, and a database upgraded from the previous schema |
| Grid regressions | `make grid-regression-test` | the grid bugs that actually happened, each named by symptom: forked trainers under the default process limit, shutdown that hung in `accept()`, a worker starting before its coordinator, two workers sharing a node id, cancelled work that ran anyway, credits paid twice, and abandoned tasks |
| Grid security | `make grid-security-test` | who may do what: the admin token gates submit and cancel, a query param does not authorize, the invite gates registration, a session is bound to its node and retired on rotation, a worker can only report the task it was given, tokens never leak into a response, and a SQL-shaped node id is stored as data |
| Grid soak | `make grid-soak-test` | load: six workers and twelve jobs submitted in one burst, every task claimed once, the queue drains to zero, and the ledger and leaderboard reconcile to the expected credit total |
| Real usage | `make usage-test` | one operator's day on real binaries: run a job, add a worker, cancel a job while it trains, stop and restart the coordinator on the same database, run more work, then read the dashboard, the summary, the job detail and the aggregate envelope |
| Grid wire layer | `bash tools/cpp_port/tests/suite/run_suite.sh --only grid` | 141 checks over base64, the JSON reader, the protocol constants, HTTP request parsing and a live loopback server |
| JSON category | `bash tools/cpp_port/tests/suite/run_suite.sh --only json` or `make suite-json` | 190 checks over the JSON reader and writer: every string escape, unicode and surrogate pairs, number shapes and overflow, the 64-level depth guard, trailing data, typed accessors, and the escape round trip |
| Store category | `bash tools/cpp_port/tests/suite/run_suite.sh --only store` or `make suite-store` | 218 checks over the SQLite store: registration and token rotation, expiry and requeue, claim order and priority, step-based progress, the single-winner finalization claim, the hash-chained ledger, a schema migrated from the previous version, and data surviving a reopen |
| Translated job on real workers | `make grid-torch-test` | a plain PyTorch script goes through the translator, is submitted, trains on two workers with the real LibTorch trainer, and aggregates to the expected 193 gradients |
| Trainer edge cases | `make trainer-edge-test` | the trainer under bad specs, missing and corrupt models, unsupported options, out-of-range steps, broken tensor files, tight rlimits, relocated bundles, and a manifest where every replica fails |
| Tunnel and links | `make gridlink-test` | link parsing, provider detection, expose results, join plans for both engines |
| Massive suite | `make suite-quick` | parity, safety, security, edge, adversarial, redteam, works, architecture, property, grid, soak and sanitizer categories |
| Static checks | `make lint typecheck` | ruff on the Python surface, mypy on the port tooling |

`make port-check` runs the build, every gate above that the machine can support,
and writes `runtime/baselines/cpp_port_check_report.md`. It skips cleanly what it
cannot run: no `g++` means no LibTorch parity, no `sqlite3.h` means no grid gate.

## Run everything that applies

```bash
make grid grid-test            # control plane, fastest meaningful signal
make grid-edge-test            # what happens when input is wrong
make grid-regression-test      # what must never come back
make grid-security-test        # who may do what
make grid-soak-test            # many workers, many jobs, one burst
make usage-test                # run, cancel, restart, resume, read the surfaces
make torch torch-test          # live train path
make grid-torch-test           # translated script through real workers
make trainer-edge-test         # trainer edge cases
translate-test gridlink-test    # the Python tools
make test port-check           # engine gates plus one consolidated report
make suite-quick               # the wide suite
make suite-json suite-store    # the JSON and SQLite categories on their own
make grid-all                  # every grid and trainer gate in one go
```

`make ci` chains lint, typecheck, the engine gates, port-check, the quick suite,
mutation checks and the Python tool tests. It is the same set CI runs.

## What each test layer is for

**Goldens and parity** catch a change in numeric behaviour. Their tolerances are
must-match for anything that feeds the wire format, because a drifting gradient
would quietly corrupt every aggregate downstream.

**The differential suite** runs the same corpus in Python and C++ and diffs the
envelopes. It is the check that would notice a semantic change the C++ unit tests
agree with each other about.

**LibTorch parity** does the same for the live train path: same architecture, same
inputs, same seeds, compared step for step. This is the gate that keeps a
translated job honest.

**The translator tests** describe failure, not success. Each unsupported
construct has a case asserting that nothing is written and the notice names the
missing feature.

**The grid gate** is the integration test for the control plane. It uses
`tests/grid_fake_trainer.cpp`, a fixture that speaks the trainer's CLI and
envelope contract without torch, so CI can run it without downloading a
framework. The gate covers: health, the invite gate, submission, the admin token
gate, two workers claiming two replicas, envelope transport, aggregation,
credits, the dashboard, and requeue after a worker is killed mid-task.

**The grid edge gate** sends the coordinator what a broken client, an impatient
operator or a hostile stranger would: bodies that are not JSON, invites that are
wrong, sessions that were rotated, asset paths that climb out of the dashboard
directory, bundles with no spec, zero steps, or a broken model, a trainer that
exits non-zero, a worker pointed at a dead port, and a database written by the
previous schema version. Every one of those has to produce a specific refusal or
a specific recovery, never a hang and never a silent success.

**The grid regression gate** exists because each check in it already failed
once. It is deliberately named after the symptom rather than the fix, so a
reader can tell what would break in production if it went red again.

**The trainer edge gate** builds its fixtures with the real translator and then
attacks the trainer with them: truncated specs, array specs, corrupt TorchScript,
unknown optimizers, negative and absurd step counts, missing and truncated tensor
files, an empty tensor, a 16 MiB address-space cap, a 1 second CPU cap, and a
bundle moved to another directory. It also pins that sandboxed and unsandboxed
runs agree and that a manifest where every replica fails reports an error.

**The grid security gate** asks the question the other gates do not: not whether
the coordinator behaves, but whether a stranger can steer it. A worker on the
other end of a tunnel holds a session token and nothing else, so the gate pins
what that token buys: its own heartbeats, its own claims, its own results. A
second worker with a valid session still cannot fail or finish the first one's
task, and the operator token gates every write. It also checks that neither token
appears in any response and that a node id written to look like SQL is stored and
returned as data.

**The soak gate** is the load check. One job proves the path works; twelve jobs
across six workers prove the claims and the ledger stay consistent when work
arrives at once. The interesting assertions are the reconciliations: every task
claimed exactly once, nothing left queued or assigned, one reward per contributing
replica, and a credit total that matches the arithmetic.

**The real usage gate** walks the lifecycle end to end, because the bugs that hurt
hide in the transitions. It cancels a job while a worker is training it and checks
the job stays cancelled after the trainer finishes, then stops the coordinator
mid-session and starts it again on the same SQLite file. If the restart loses a
job, a count or a token, this gate goes red.

All of the grid scripts pick their ports at random and refuse to start if a grid
is already answering there, so a coordinator left behind by an interrupted run can
never quietly answer for a new one. Each also stops its own workers on exit, and
the socket is closed with a shutdown before the listener, so a stop is immediate
rather than waiting on a blocked `accept()`.

**The sanitizer category** is where use-after-free and data races in the port
show up. `make asan` and `make tsan` run the subset directly when you are
chasing one.

## CI

| Job | Covers |
| --- | --- |
| `cpp-port` | port build, goldens, quick suite, soak and differential on main, full port-check, report artifact |
| `grid` | SQLite headers, coordinator and worker build, end-to-end gate, edge gate, regression gate, security gate, soak gate, real usage gate, and the grid, JSON and store suite categories |
| `libtorch` | CPU torch install, trainer build, parity, translator tests, translated job through real workers, trainer edge cases |
| `gridlink` | ruff on the Python tools surface, link and join tests, provider readiness |

The frozen Python stack has no CI job. Its `legacy-*` make targets (lint,
typecheck, unit and security tests) are manual only, run when you are fixing the
stack; they are not a gate on ordinary changes. See
[From Python to C++](from-python-to-cpp.md).

## Adding a test

- Engine, sandbox, envelope or ABI behaviour: a case in the relevant
  `tools/cpp_port/tests/` file, or a new category under
  `tools/cpp_port/tests/suite/categories/`.
- Control plane happy path or recovery: extend `tools/cpp_port/tests/grid_e2e.sh`.
  Prefer a real second worker over a mocked client, because the interesting
  failures live in the protocol and the store.
- Control plane refusals and malformed input: `tools/cpp_port/tests/grid_edge.sh`.
  It expects a specific status code and a readable message, not just "an error".
- A bug you just fixed: `tools/cpp_port/tests/grid_regression.sh`, named after the
  symptom, with a comment saying what used to happen.
- Who may call what: `tools/cpp_port/tests/grid_security.sh`. Assert the status
  code and the reason, and check that tokens stay out of the response.
- Load or concurrency: `tools/cpp_port/tests/grid_soak.sh`, and make the size
  tunable through an environment variable so a slow box can shrink it.
- Lifecycle behaviour across a stop and restart: `tools/cpp_port/tests/real_usage.sh`.
- Wire-format code without a process: a category under
  `tools/cpp_port/tests/suite/categories/`. `suite_grid.cpp` covers base64, JSON
  and HTTP this way, including a live server on loopback.
- Store logic without a network: `tools/cpp_port/tests/suite/categories/suite_store.cpp`.
  It drives the real SQLite file, so a claim, a requeue or a migration is tested
  where it lives rather than through the coordinator.
- The JSON reader and writer: `tools/cpp_port/tests/suite/categories/suite_json.cpp`.
  It is header only, so it needs no torch and no link step.
- The trainer: `tools/cpp_port/tests/trainer_edge.sh`. Fixtures come from the real
  translator, so the shapes match production.
- Python tool behaviour: `tools/<tool>/tests/`, plain pytest.
- Data to compare across runtimes: regenerate with
  `tools/cpp_port/tests/gen_golden.py` and commit both the generator output and
  the generator change, so the vector is reproducible.

## Reports

`runtime/baselines/` holds evidence rather than decoration:

| File | Source |
| --- | --- |
| `cpp_port_check_report.md` | the last `make port-check` |
| `cpp_parity_report.md`, `parity_matrix.md` | parity tolerances and results |
| `cpp_improve_plan.md` | the optimisation pass and its acceptance numbers |

Read the report after a gate run. A green exit code with a skipped section is
worth noticing: a skip means the machine could not run that check, not that the
check passed.
