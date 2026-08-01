# TLS and mTLS

Orchestrator gRPC TLS is configured through **`services_python/grpc_tls.py`**. Both `orchestrator_grpc.serve()` and tests use `configure_grpc_server()`.

## Defaults (v1.2)

| Variable | Default | Notes |
|----------|---------|-------|
| `GRPC_USE_TLS` | `true` | Plaintext only when `false` and `DISTRIBAI_ENV` ≠ `production` |
| `GRPC_TLS_CERT` | `runtime/secrets/tls/server.crt` | Auto-generated in dev if missing |
| `GRPC_TLS_KEY` | `runtime/secrets/tls/server.key` | Auto-generated in dev if missing |
| `GRPC_BIND_HOST` | `0.0.0.0` | Listen address for gRPC |
| `GRPC_MTLS_CA` | unset | When set, requires worker client certificates |
| `GRPC_TLS_REQUIRE_CLIENT_CERT` | `false` | Alternative mTLS path via `GRPC_TLS_CA` |

Generate dev certs:

```bash
python scripts/dev/gen_tls_certs.py --hostname localhost --ca
```

## Admin HTTP

Admin TLS uses `_setup_admin_tls()` in `orchestrator_grpc.py` (loopback may stay HTTP in dev).

Admin **Bearer** auth uses `DISTRIBAI_ADMIN_SECRET` (not JWT). Handler-level admin routes also accept admin JWT or the shared secret when auth is enforced.

## Worker clients

Workers connecting with TLS need matching trust material via env documented in [environment-reference.md](environment-reference.md).
