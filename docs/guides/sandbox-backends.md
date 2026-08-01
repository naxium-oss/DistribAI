# Sandbox backends

Script jobs run through `worker/src/sandbox/backends/build_sandbox()` after the tarball is unpacked in `ScriptRunner`.

## Selection order

1. `DISTRIBAI_SANDBOX_BACKEND` env override (`docker` | `nsjail` | `subprocess`)
2. `docker` on PATH and daemon reachable
3. `nsjail` on PATH (Linux)
4. `subprocess` fallback (Windows and dev VMs)

Per-job override: `hyperparams.sandbox_backend` or env `DISTRIBAI_SANDBOX_BACKEND`.

## Network policy

| Policy | Meaning |
|--------|---------|
| `open` | Unrestricted egress (default when not denied) |
| `none` | No outbound networking (Docker/nsjail enforce; subprocess warns) |
| `restricted` | Reserved slot for HF/S3 whitelist (v1.3) |

Set via `hyperparams.network_policy`, `DISTRIBAI_NETWORK_POLICY`, or legacy `deny_egress` / `DISTRIBAI_DENY_EGRESS`.

## Results

Successful runs include `backend_used` in the task result dict (e.g. `subprocess`, `docker`).

## Operator guidance

- **Production isolation:** install Docker or nsjail; do not rely on subprocess alone on untrusted code paths.
- **pip install:** blocked when egress is denied; bake dependencies into the tarball or use `.site-packages` in the package.

See also [beta-preprod-checklist.md](../runbooks/beta-preprod-checklist.md).
