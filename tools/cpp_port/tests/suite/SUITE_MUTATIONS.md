# Red/Green Mutation Log

Proof the massive suite catches real bugs. Each mutation is injected,
the suite must go RED for the right reason, then the original code is
restored and the suite must return GREEN. Reproduce any time with:

```bash
bash tools/cpp_port/tests/suite/selfcheck_mutations.sh   # exit 0 = proof
```

Full transcript of the latest run: `runtime/baselines/mutation_log.txt`.

## Injected mutations (selfcheck_mutations.sh)

| # | Mutation | Expected catcher | Result |
|---|---|---|---|
| 1 | AdamW `t_` increment moved after bias-correction computation (off-by-one in `bc1`/`bc2`) | `edge`: finiteness + first-step magnitude checks | RED (3 checks fail) -> restored GREEN |
| 2 | sandbox `apply_limits_in_child` drops `setrlimit(RLIMIT_NPROC)` | `security`: child observes `getrlimit(NPROC)==24` and spawn-past-cap fails loudly | RED -> restored GREEN |
| 3 | envelope decode compares CRC against a constant (verification disabled) | `adversarial`: every single-byte corruption must be rejected | RED -> restored GREEN |

## Real bugs the suite caught during development (before any injection)

1. **Checkpoint retention data loss** (`core/checkpoint.hpp::retain`).
   Retention sorted checkpoints by 1-second `st_mtime`; same-second saves
   tie and the unstable sort could delete files *newer* than the cutoff,
   including the checkpoint `save()` had just returned. Caught by `edge`
   checkpoint-resume failing intermittently (2 of 6 runs) with "cannot
   open .../<id>.ckp". Fix: nanosecond `st_mtim` ordering. 8/8 clean after.

2. **Sandbox limit-kill misclassification** (`sandbox.hpp` reaper).
   Children killed by a limit signal *after* the parent's first select
   return (e.g. SIGXFSZ mid-file-write) hit the EOF path and were reported
   `timed_out=1, signal=0, limit_killed=0`: limit_killed was effectively
   dead code for children that die before or mid-write. Caught by the
   `redteam` FSIZE check. Fix: outcome classes ordered full-pod -> signal
   death -> deadline-kill -> exited-without-pod, with post-kill status
   classification (a child already dying of SIGXFSZ that loses the race
   with the reaper's SIGKILL is still a limit kill).

3. **Reaper regression caught by the golden gate during rework**: the first
   fix classified "child alive at reap time" as a timeout, which killed
   children that had *just delivered a full pod but not yet exited* - the
   golden COW/rng-separation/trivial-pod tests all went red. Root cause:
   aliveness is expected while data sits in the pipe; only deadline expiry
   may kill. Fixed by checking pod delivery first. (This is the suite
   ecosystem working: new sandbox code was gated by the old suite.)

4. **Environment-variable leakage into sandbox children** (child path).
   `fork()` inherits `environ`, so ambient secrets were visible to every
   sandbox child - the Python subprocess sandbox scrubs them, the C++ port
   did not. Caught by `security` planting `DAI_SUITE_SECRET_MARKER`.
   Fix: child-side environ scrub (PATH/TMPDIR kept), blanked in place.

5. **Stale test premise (hardware drift, caught on this box)**: the golden
   CPU-hog test used a bounded 10-billion-iteration loop that now *completes
   within the 1s CPU window* on this machine, making "hog is killed"
   hardware-dependent. Fixed to an unbounded hog so the soft CPU limit is
   always crossed.

## Vacuous-pass cleanups (tests corrected to be falsifiable)

- The old golden 3.2b AS-limit test only asserted "no hang"; it passed even
  when the allocation succeeded. The redteam AS check asserts the cap via
  `getrlimit` inside the child plus bounded outcomes.
- The original thread-spawn NPROC check passed vacuously: the 64 MiB AS cap
  made thread spawn fail with `bad_alloc` before NPROC could bind. The
  security check raises mem_mb so NPROC is the binding constraint, requires
  the child to report the cap AND fail loudly.
- `std::system()` returns raw wait statuses; the manifest adversarial check
  initially misread exit codes as signal kills. Fixed to decode with
  `WIFEXITED`/`WEXITSTATUS`.
