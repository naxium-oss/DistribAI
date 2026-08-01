# gRPC API (NodeService)

Canonical contract: [`proto/distribai.proto`](../../proto/distribai.proto).

Human-readable overview (message shapes, auth, examples): [README.md](./README.md#gRPC-API).

## Quick reference

| Item | Value |
|------|--------|
| Service | `NodeService.StreamSession` (bidirectional stream) |
| Default port | `50051` (override with orchestrator gRPC bind env) |
| Auth | `RegisterSession.jwt_token` on first client message |
| TLS | See [`services_python/grpc_tls.py`](../../services_python/grpc_tls.py) and `.env.example` |

## Regenerating Python stubs

When `proto/distribai.proto` changes:

```bash
python -m grpc_tools.protoc \
  -I proto \
  --python_out=worker/src/distribai_proto \
  --grpc_python_out=worker/src/distribai_proto \
  proto/distribai.proto
```

Worker imports live under `worker/src/distribai_proto/`.

## Related docs

- [REST admin endpoints](./endpoints.md)
- [Deployment / ports](../runbooks/ports.md)
- [Five-minute onboarding](../guides/five-minute-onboarding.md)
