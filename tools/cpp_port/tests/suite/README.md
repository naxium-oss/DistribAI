# Massive Test Suite (tools/cpp_port/tests/suite)

Ten test categories over the C++ port in one orchestrator. Feature freeze:
the core only changes when a test forces a fix (three did - see
`SUITE_MUTATIONS.md`).

## How to run

```bash
# everything: 10 categories + sanitizers + differential (soak ~10 min)
bash tools/cpp_port/tests/suite/run_suite.sh

# quick subset (fast categories + sanitized runs; ~2 min)
bash tools/cpp_port/tests/suite/run_suite.sh --quick

# one category at a time
bash tools/cpp_port/tests/suite/run_suite.sh --only=parity
bash tools/cpp_port/tests/suite/run_suite.sh --only=security
# categories: parity safety security edge adversarial redteam works
#             architecture property run differential sanitize

# the 10-minute soak alone
bash tools/cpp_port/tests/suite/soak.sh            # default 600s
bash tools/cpp_port/tests/suite/soak.sh 60         # or any length

# the red/green proof (inject 3 bugs, watch them caught, restore)
bash tools/cpp_port/tests/suite/selfcheck_mutations.sh

# differential corpus alone (needs .venv)
python3 tools/cpp_port/tests/suite/differential_suite.py
```

`port_check.sh` / `make port-check` runs the quick subset + sanitizers and
gates on them (sanitizers explicitly SKIPPED - never silently PASSed - when
the toolchain lacks runtimes).

## Categories

| # | Category | Binary/Script | What it proves |
|---|---|---|---|
| 1 | **parity** | `categories/parity/golden_cases.{cpp,py}` | Golden-50 case matrix (seeds x sizes 1/1K/10K/100K x steps 1-300 x AdamW grid x sandbox 1x/2x). Same-seed contracts: identical digest, bit-equal loss trajectories, stable init bytes, grad_len==params, finite loss; seed separation. |
| 2 | **safety** | `suite_core --category safety` | Misuse fails loud, never silent corruption: partial optimizer coverage detectable+isolated, wrong-size AdamW import rejected as no-op, missing/corrupt checkpoints error cleanly, truncated POD fails the sandbox. |
| 3 | **security** | `suite_sandbox --category security` | Isolation integrity: no env inheritance into children, uid rewired in user-ns, mount-ns degradation explicit, NOFILE/NPROC caps enforced *inside* the child (getrlimit-visible + fail-loud), parent cwd untouched. |
| 4 | **edge** | `suite_core --category edge` | Extreme-but-legal inputs: tiny lr finite, huge-lr chaos bit-deterministic (NaNs included), 3000-step 100k horizon finite, dead-ReLU zero grad rows, constant targets converge, checkpoint resume bit-exact at odd splits {0,1,7,199,200}, one-step determinism at all sizes. |
| 5 | **adversarial** | `suite_sandbox --category adversarial` | Hostile inputs: every single-byte corruption and truncation of an envelope rejected, hostile dai_job manifests (traversal/dup/absurd/garbage) never crash, 40 rapid spawn/kill cycles leak no FDs, no-namespace mode still enforces CPU rlimit. |
| 6 | **redteam** | `suite_sandbox --category redteam` | Active attacks: FSIZE enforcement proof (SIGXFSZ kill classified, never mislabeled timeout), fork COW privacy against deliberate parent-heap writes, AS cap installed + bounded outcomes, pipe-flood contained by deadline, inflated checkpoint payload_len rejected. |
| 7 | **works** | `suite_core --category works` | Does-it-work end to end: train->checkpoint->resume->grad-envelope->decode pipeline reaches identical weights, 2 concurrent sandboxes agree bit-exactly, C ABI job build/run/results smoke. |
| 8 | **architecture** | `suite_core --category architecture` | Structural contracts: envelope version/tag/CRC policy, checkpoint retention keep=3, thread-scratch isolation (concurrent==serial), ABI version stable, TrainPod size sane, fast path deterministic. |
| 9 | **property** | `suite_property` | Invariants + fuzz: RNG range/skip/export-import/draw-order, AdamW algebra (decay direction, bias-correction magnitude, monotone descent, bounded huge-grad updates, zero-grad fixpoint), shape misuse trips asserts, 20k-case envelope byte fuzz with zero crashes. |
| 10 | **run** | `suite_run` / `soak.sh` | 10-min soak: FD/thread census flat, RSS ceiling, spawn-cost drift bounded, zero failed iterations across tens of thousands of train+spawn cycles. |
| + | **differential** | `differential_suite.py` | Python harness vs C++ over the fixed 10-seed corpus via compare.py; must-match tolerances from parity_matrix.md; writes `runtime/baselines/diff_corpus_report.json`. |
| + | **sanitize** | `sanitize.sh` | ASan+UBSan builds of core+sandbox categories (7 runs), gate on zero findings; explicit SKIP when toolchain lacks runtimes. |

## Conventions

* Exit 0 = category green. Any failing check fails the category.
* Every category prints `CATEGORY <name>: <passed>/<total> checks` and a
  JSON tail line; the orchestrator aggregates into
  `runtime/baselines/suite_report.json`.
* "Fail loud": misuse must produce a nonzero exit or an explicit error
  field - never silence, never a vacuous pass. Checks that passed vacuously
  were rewritten (see SUITE_MUTATIONS.md).

## Results (latest full pass)

* All 10 C++ categories green; sanitizers 7/7 clean.
* Soak: 79,421 train bursts + 79,421 sandbox spawns, 0 failures, no leaks.
* Differential: 10/10 corpus seeds OK (9 tight, 1 explained relu-flip).
* Mutation selfcheck: 3/3 injected bugs caught, tree restored green.
