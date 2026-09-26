# gridlink: expose the grid, join from anywhere, free

Two halves of one thing:

- **expose** (operator): one command publishes the coordinator through a free
  tunnel and prints a shareable **join link**.
- **join** (contributor): paste that link on any machine and start a worker.

**Local bind is the default.** Nothing is public unless `--provider` is passed.
**No paid plan and no custom domain are needed.** The recommended path is a free
Cloudflare Quick Tunnel that needs no account at all.

```
operator                                contributor
────────                                ───────────
gridlink expose --provider cloudflare   gridlink join "distribai://join?…"
        │                                        │
        └── free trycloudflare URL ──────────────┘
             (no account, no domain)         local GPU · Colab · Kaggle · Molab · VPS
```

## Engines

A link names the engine that answers it.

| Engine | Stack | Default port | Wire |
| --- | --- | --- | --- |
| `native` (default) | C++ coordinator and worker from [`tools/cpp_port/grid`](../cpp_port/grid/README.md) | 50061 | JSON over HTTP |
| `legacy` | the deprecated Python gRPC control plane under `legacy/` | 50051 | gRPC |

The native coordinator serves the worker API, the operator API and the dashboard
from one port and speaks plain HTTP, so a quick tunnel proxies it with no extra
flags. That is why `native` is the default. Pass `--engine legacy` when you are
joining a grid that still runs the Python stack.

---

## One quick setup command (operator)

```bash
# Cloudflare Quick Tunnel: free, no account, no domain (recommended)
PYTHONPATH=tools .venv/bin/python -m gridlink expose \
    --provider cloudflare --port 50061 --name my-grid --invite team-alpha

# ...which prints:
#   distribai://join?v=1&engine=native&provider=cloudflare
#       &host=random-words.trycloudflare.com&port=443&tls=1&invite=team-alpha&name=my-grid
```

`cloudflared` is located on `PATH`, via `CLOUDFLARED_BIN`, or auto-downloaded
into `tools/gridlink/bin/` on first use (add `--download` to force it). For an
HTTP origin the tunnel needs no special flags; for a legacy gRPC (h2c) origin
gridlink adds `--http2-origin`, and `--no-http2-origin` turns that off.

The coordinator keeps its normal loopback bind. Only `cloudflared` talks to it,
from the same machine, over `http://127.0.0.1:50061`.

### ngrok (optional, free tier)

ngrok works too, but its free tier needs an authtoken, so it is opt-in and never
the default:

```bash
export NGROK_AUTHTOKEN=...      # free account: https://dashboard.ngrok.com
PYTHONPATH=tools .venv/bin/python -m gridlink expose --provider ngrok --port 50061
```

If `NGROK_AUTHTOKEN` is unset, `gridlink providers` says so and points you back
at the account-free Cloudflare path.

### Private and LAN, no tunnel

```bash
# local bind only, nothing exposed (default)
python -m gridlink expose --provider local --port 50061

# LAN: advertise your machine's address instead of loopback
python -m gridlink expose --provider local --link-host 192.168.1.20 --port 50061
# or build a link without touching anything at all:
python -m gridlink link --host 192.168.1.20 --port 50061
```

---

## Join (contributor)

```bash
# plan only: prints the env and the command it would run
python -m gridlink join "distribai://join?v=1&engine=native&host=…&port=443&tls=1"

# start the worker
python -m gridlink join "<link>" --exec

# a bare host:port works too, and defaults to the native engine
python -m gridlink join 192.168.1.20:50061
```

For the native engine the plan is the C++ worker:

```bash
build/cpp_port/distribai_worker --orchestrator https://host:443 --invite CODE
```

Build it once with `make grid`. The worker needs the LibTorch trainer
(`make torch`) for jobs that arrive, and nothing else.

What `join` puts in the environment, for either engine:

| Env | Value |
| --- | --- |
| `ORCHESTRATOR_URL` | the coordinator URL from the link (`http(s)://` for native, `host:port` for legacy) |
| `GRPC_USE_TLS` | legacy only, from the link |
| `DISTRIBAI_INVITE_CODE` | from the link, when present |
| `ADMIN_URL` | from the link, when present |
| `DISTRIBAI_EPHEMERAL` | `1` with `--ephemeral` (Colab, Kaggle, Molab) |
| `STATE_DIR` | with `--state-dir`, and `--work-dir` for the native worker |

With `--engine legacy` the command is the frozen worker entrypoint
(`python -m worker.src.daemon.run`), unchanged from before.

---

## Free compute in a lot of places

The grid is not just for one machine. The same join link works on:

| Where | Template | Notes |
| --- | --- | --- |
| Local GPU or workstation | `gridlink join <link> --exec` | persistent worker |
| Google Colab (free GPU) | [`notebooks/colab_join.ipynb`](notebooks/colab_join.ipynb) | burst worker |
| Kaggle (free GPU) | [`notebooks/kaggle_join.ipynb`](notebooks/kaggle_join.ipynb) | burst worker |
| Molab (free GPU) | [`notebooks/molab_join.ipynb`](notebooks/molab_join.ipynb) | burst worker |
| Any VPS or burst server | `gridlink join <link> --exec` | persistent or ephemeral |

Each notebook has one cell to paste the link, one to clone and build, and one to
run the worker. Link parsing uses only the standard library, so it works before
anything is installed.

Burst workers join while the session runs and leave when it ends. Keep job
slices short and checkpoint to storage the operator can read.

---

## CLI reference

```bash
make grid-providers
make grid-expose ARGS="--provider cloudflare --port 50061 --name my-grid --invite team-alpha"
make grid-join   LINK="distribai://join?v=1&engine=native&provider=cloudflare&host=…&port=443&tls=1" ARGS="--exec"
make gridlink-test
```

The link goes in `LINK`, not `ARGS`, so its `&` query separators reach the CLI
intact. `ARGS` carries ordinary flags. Direct module use:

```
gridlink providers                                   # which providers are ready
gridlink expose  --provider {local,cloudflare,ngrok} [--engine native|legacy]
                 [--port N] [--invite C] [--name N] [--bind-host H]
                 [--link-host H] [--download] [--no-http2-origin] [--once] [--json]
gridlink join    [link|host:port] [--engine native|legacy] [--orchestrator H:P]
                 [--node-id N] [--worker-bin PATH] [--ephemeral] [--invite C]
                 [--admin-url U] [--state-dir D] [--exec] [--json]
gridlink link    --host H [--engine native|legacy] [--port N] [--tls]
                 [--invite C] [--name N] [--json]
```

`python -m gridlink.providers` prints a one-line-per-provider readiness report.

---

## Security notes

- Default is **local bind only**. Exposing is always an explicit choice.
- Tunnel links are TLS over port 443 with a public certificate. The native
  worker verifies it against the system trust store; the legacy path additionally
  reads `GRPC_TLS_CA` for a private orchestrator CA.
- ngrok's authtoken is read from `NGROK_AUTHTOKEN` (env), never argv.
- Set `--invite` when you expose, and share it with the link. Tunnel URLs are
  public, so the invite is what gates registration.
- Operator writes on the native coordinator need `X-Grid-Token`; without a
  configured token only loopback callers may submit or cancel.
- The tunnel proxies the coordinator's own port, which also serves the read API
  and the dashboard. That is the point, and it is a reason to keep the invite and
  the token strong.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `cloudflared not found` | run with `--download`, install it, or set `CLOUDFLARED_BIN` |
| `did not come up within Ns` | check outbound network, then re-run `gridlink expose` |
| `NGROK_AUTHTOKEN not set` | use `--provider cloudflare`, which needs no account |
| Worker registers but never gets work | no job is queued. Submit one, then check `/v1/health` |
| `invite code required` | pass the same `--invite` used when exposing |
| TLS handshake failure on a tunnel link | the worker build has no OpenSSL headers, or the clock is far off |
| Joining a Python grid by mistake | add `--engine legacy` to `gridlink join` |
