# Security Policy

DistribAI is a distributed AI compute network that runs user-supplied
training scripts on volunteer GPUs coordinated by a central
orchestrator. This document describes the threat model, what DistribAI
defends against today, and what is explicitly out of scope.

## Reporting a vulnerability

Email **security@distribai.io** (do not open a public issue) with:

- A description of the vulnerability and impact.
- Reproduction steps (proof-of-concept code is welcome but not
  required).
- The git SHA you reproduced against.

Expect an acknowledgement within 72 hours and a patch timeline within
14 days. Coordinated disclosure preferred. See the OSS Disclose Triage
ladder if you need a public CVE assigned.

## Threat model

DistribAI is operated under the following assumptions. Anything outside
these assumptions is out of scope and will not be treated as a
vulnerability.

### Trusted

- **Orchestrator operator.** The machine running
  `services_python.orchestrator_grpc` is fully trusted. If the operator
  is malicious, every contributor loses.
- **The Python interpreter, OS, and hardware** on every node and on the
  orchestrator. DistribAI does not defend against rootkits, malicious
  BIOS, supply-chain compromise of pip/conda, or quantum-broken TLS.
- **JWT_SECRET and SIGNING_KEY persistence.** These secrets must be
  backed up. Losing them invalidates every JWT and breaks the
  credit-ledger HMAC chain. See `runtime/secrets/`.

### Semi-trusted

- **Worker nodes.** Treated as potentially malicious for the
  *aggregation* path -- gradients are filtered through Multi-Krum /
  trimmed-mean / FoolsGold / SignGuard / bucketed Krum before
  inclusion (see `worker/src/daemon/byzantine_detector/`). Trusted for
  their own resource accounting and for not attacking each other
  laterally (no node-to-node networking is initiated by DistribAI).

- **Organizations submitting jobs.** Authenticated via admin JWT
  (`required_kind="admin"` on POST /admin/jobs since v1.1) but their
  *script content* is treated as semi-trusted: it runs on contributor
  nodes inside `ScriptRunner` with env whitelisting + RLIMITs + a
  default-off `pip install` gate. It is NOT in a true sandbox -- a
  truly trustless training market needs gVisor / nsjail / Docker
  wrapping and is on the v2.0 roadmap.

### Untrusted (attack surface)

- **Network attackers** on the gRPC port (50051) and admin port (8766).
- **Anyone with a worker registration** who attempts to claim credits
  for fake work, claim someone else's node_id, or submit poisoned
  gradients.
- **Anyone who can place content in a job's tarball** (today: any
  authenticated org). Tar traversal is rejected; pip installs and
  arbitrary Python execution are scoped down per above.

## What v1.1 hardening covers

The `hardening/v1.1-security-and-quickwins` branch closes the
highest-severity gaps in v1.0.0. Concretely:

| Issue | v1.0.0 behaviour | v1.1 behaviour |
|---|---|---|
| Admin API auth | `_authenticate_request` never called from POST/DELETE handlers; any TCP reach to :8766 = full control | Every `/admin/*` handler in `jobs.py / credits.py / votes.py / nodes.py` calls `self.node_service._authenticate_request(req, required_kind="admin")` |
| JWT_SECRET lifecycle | regenerated via `secrets.token_urlsafe(32)` on every restart if env var unset; all node JWTs silently invalidated | persisted to `<DISTRIBAI_SECRETS_DIR>/JWT_SECRET` (mode 0600); reloaded on restart |
| SIGNING_KEY lifecycle | same; broke Merkle ledger HMAC chain on each restart | same fix as JWT_SECRET |
| gRPC RegisterSession | `register.jwt_token` field silently discarded; any TCP client could claim any node_id | JWT validated against claimed `node_id` (`kind=node`); empty JWT rejected unless `DISTRIBAI_GRPC_ALLOW_BOOTSTRAP=1` and node is unknown |
| `/v1/nodes/register` | issued JWTs with zero PoW check (bypassed the "enhanced" PoW path) | returns 403 unless `DISTRIBAI_ALLOW_INSECURE_REGISTER=1` |
| ScriptRunner env | inherited `os.environ.copy()`; leaked JWT_SECRET, AWS keys, etc. into user scripts | env whitelist (PATH, CUDA, OMP, HF cache); no daemon secrets |
| ScriptRunner resource cap | timeout only | RLIMITs on POSIX: memory, CPU time, file size, processes, FDs; `prctl(PR_SET_NO_NEW_PRIVS)` |
| ScriptRunner pip-install | always ran from user's `requirements.txt` | default off; `DISTRIBAI_ALLOW_SCRIPT_PIP_INSTALL=1` to opt in; `--only-binary=:all:` to refuse sdists by default |
| JSON parse errors | `NameError` (missing `json` import in jobs/credits/votes/nodes) | proper 400 response |
| Committed secrets | 617 gradient blobs, populated SQLite DB, `auth.json` files in git | all purged via `git rm --cached`; comprehensive `.gitignore` added |

## What v1.2 hardening covers

| Issue | v1.1 behaviour | v1.2 behaviour |
|---|---|---|
| Default transport | `GRPC_USE_TLS=false`; plaintext on `:50051` | `GRPC_USE_TLS=true` by default. Dev mode auto-issues self-signed certs to `runtime/secrets/tls/` on first boot; `DISTRIBAI_ENV=production` refuses to start without real certs. |
| Missing-cert behaviour | silently downgraded to insecure (`logger.warning` only) | fail-closed in production with a pointer to `scripts/dev/gen_tls_certs.py`; dev auto-generates so first-boot UX still works |
| Mutual TLS | not supported | set `GRPC_MTLS_CA=<path>` and the orchestrator requires every worker to present a client cert signed by that CA. Issue them with `python scripts/dev/gen_tls_certs.py --mtls --node-id <id>` |
| Admin REST TLS | always plaintext (loopback-only) | auto-enabled whenever `ADMIN_HOST != 127.0.0.1`; refuses non-loopback plaintext in production |
| Cert generation tooling | operators wrote their own openssl invocations | `python scripts/dev/gen_tls_certs.py` -- RSA 4096 or ECDSA P-256, SAN with hostname + localhost + 127.0.0.1, optional CA + mTLS client certs, key files mode 0600 |
| Worker channel | reads `GRPC_TLS_CA` only when `GRPC_USE_TLS=true` | reads `DISTRIBAI_GRPC_USE_TLS` (default true), optional `DISTRIBAI_GRPC_CA` for pinning, optional `DISTRIBAI_GRPC_CLIENT_CERT` / `DISTRIBAI_GRPC_CLIENT_KEY` for mTLS |
| User-script execution | hardened subprocess in daemon's Python interpreter | real container / namespace isolation via Docker / nsjail / hardened subprocess backends (auto-selected). See below. |

### TLS operator guide

1. **Generate certs** on the orchestrator host:
   ```bash
   python scripts/dev/gen_tls_certs.py --hostname your-grid.example.com --ca
   ```
   Writes `runtime/secrets/tls/{ca,server}.{crt,key}` (key files mode 0600).

2. **Distribute the CA cert** (`runtime/secrets/tls/ca.crt`) to each
   contributor node out-of-band -- e.g. via your config-management tool
   or the `DistribAI` worker installer. The orchestrator never needs to
   ship `ca.key`; that stays on the orchestrator host.

3. **Flip the orchestrator into production mode:**
   ```bash
   export DISTRIBAI_ENV=production
   ```
   Now `services_python.orchestrator_grpc` refuses to boot if TLS is
   disabled or certs are missing.

4. **Workers point at the pinned CA:**
   ```bash
   export DISTRIBAI_GRPC_CA=/etc/distribai/ca.crt
   ```
   For mutual TLS, additionally issue and distribute per-node client certs:
   ```bash
   # On the orchestrator:
   python scripts/dev/gen_tls_certs.py --mtls --node-id worker-42
   export GRPC_MTLS_CA=runtime/secrets/tls/ca.crt   # on orchestrator
   # Ship runtime/secrets/tls/worker-42.{crt,key} to worker-42, then:
   export DISTRIBAI_GRPC_CLIENT_CERT=/etc/distribai/worker.crt
   export DISTRIBAI_GRPC_CLIENT_KEY=/etc/distribai/worker.key
   ```

### Sandbox backends (v1.2)

v1.2 ships real container-level isolation. `ScriptRunner` no longer
executes the user-supplied `run.py` directly inside the daemon's
Python interpreter; it delegates to a pluggable backend selected at
startup:

| Backend | Isolation | Selected when |
|---|---|---|
| `DockerSandbox`     | ephemeral container, `--read-only` rootfs + `/tmp` tmpfs, `--cap-drop=ALL`, `--network=none` by default, `--memory` / `--pids-limit` / `--cpus` hard caps, `--user=1000:1000`, `--security-opt=no-new-privileges` | `docker` CLI on PATH AND `docker info` succeeds (default on Linux + macOS + Windows-with-Docker-Desktop) |
| `NsjailSandbox`     | mount + network + user + pid + ipc + uts namespaces, RLIMITs, seccomp allow list, `keep_caps: false`, `no_new_privs: true` | Linux only, `nsjail` on PATH, Docker not reachable |
| `SubprocessSandbox` | v1.1 hardened-subprocess: POSIX RLIMITs + env whitelist + `prctl(PR_SET_NO_NEW_PRIVS)` | fallback (Windows-without-Docker; dev hosts; CI without privileged containers) |

Force one explicitly with `DISTRIBAI_SANDBOX_BACKEND={docker|nsjail|subprocess}`.
Customize the container image with `DISTRIBAI_SANDBOX_IMAGE`
(default `python:3.11-slim`). Opt into GPU passthrough with
`DISTRIBAI_SANDBOX_GPU=1` (requires the host to have the NVIDIA
container runtime installed).

Jobs declare their network needs via `hyperparams["network"]`, which
maps to a `NetworkPolicy` enforced by the backend:

| Policy | Docker | nsjail | subprocess |
|---|---|---|---|
| `NONE` (default)   | `--network=none` | `clone_newnet: true`  | warning logged; cannot enforce |
| `RESTRICTED`       | `--network=bridge` (per-org egress ACLs queued for v1.3) | netns kept open | warning logged |
| `OPEN`             | `--network=bridge` | netns kept open | no warning |

If the chosen backend cannot enforce the requested policy, the
worker logs a warning rather than silently degrading -- the operator
sees in the worker log whether real isolation is in effect.

## What is explicitly NOT covered (yet)

- **Automated cert rotation / ACME.** Certs are issued by the
  helper script, valid for 365 days by default. No OCSP stapling,
  no Let's Encrypt, no in-band rotation -- operators must regenerate
  and restart. Roadmap for v1.3+.
- **Differential privacy on gradients.** Full uncompressed gradients
  land on S3 with 1-hour pre-signed URLs that any worker can read.
  DP-SGD (arxiv:1607.00133) + Secure Aggregation (arxiv:1611.04482)
  are research-roadmap items.
- **Verifiable compute.** Multi-Krum catches bad gradients but cannot
  prove a node actually ran the work. TOPLOC-style attestation
  (INTELLECT-2, arxiv:2505.07291) and zkML are roadmap items.
- **Single-orchestrator SPOF.** No leader election, replication, or
  failover. Operators must accept this until v2.0.
- **Bandwidth assumptions.** Server-side aggregation requires every
  gradient to round-trip S3; for a 1B-param fp32 model that is 4 GB
  per node per step. Consumer ISPs cannot sustain this. v1.1 adds
  FoolsGold / SignGuard / Bucketing for *robustness*; bandwidth
  reduction (DiLoCo, DeMo) is queued for v1.2.

## Operator security checklist

1. **Set persistent secrets** before first boot:
   ```bash
   export JWT_SECRET=$(openssl rand -base64 48)
   export SIGNING_KEY=$(openssl rand -base64 48)
   ```
   Or let the orchestrator persist them to `runtime/secrets/` on first
   boot and back that directory up.
2. **Generate TLS certs and switch to production mode** (TLS is on by
   default in v1.2 -- production mode refuses to boot without real certs):
   ```bash
   python scripts/dev/gen_tls_certs.py --hostname your-grid.example.com --ca
   export DISTRIBAI_ENV=production
   ```
   See the "TLS operator guide" above for the four-step rollout
   (generate, distribute CA, flip prod, point workers at the CA).
3. **Bind admin to loopback** (default) and front it with an
   authenticating reverse proxy (Caddy, nginx, Cloudflare Access) if
   you need remote admin access. Do NOT set `ADMIN_HOST=0.0.0.0`
   without a proxy.
4. **Mint a long-lived admin JWT** and store it in your secret manager:
   ```bash
   python scripts/dev/mint_admin_jwt.py --subject ops --ttl-hours 720
   ```
5. **Audit the bootstrap admin JWT** at `runtime/secrets/bootstrap_admin_jwt`
   and rotate it after onboarding (it has 7-day TTL).
6. **Never set `DISTRIBAI_ALLOW_INSECURE_REGISTER=1`** on a public
   network. It exists for air-gapped dev clusters.
7. **Never set `DISTRIBAI_ALLOW_SCRIPT_PIP_INSTALL=1`** unless you fully
   trust every job-submitting org.
8. **Confirm the chosen sandbox backend** at worker startup. The
   daemon logs `ScriptRunner using sandbox backend: <name>` once on
   init. On a public deployment this should read `docker` or
   `nsjail`; if it reads `subprocess`, install Docker on the host or
   force the backend via `DISTRIBAI_SANDBOX_BACKEND=docker`.

## Reporting hall of fame

Open. First disclosure earns a permanent credit here.
