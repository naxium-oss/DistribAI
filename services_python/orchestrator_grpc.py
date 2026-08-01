"""DistribAI orchestrator process entry (gRPC + admin HTTP).

Composition root that wires modular pieces:

- constants.py -- shared configuration knobs
- scheduler.py -- assignment / queue timing
- grpc_service.py -- bidirectional stream handlers
- admin_api/ -- REST handlers for operators and v1 clients
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import grpc
import jwt
from botocore.config import Config
from dotenv import load_dotenv

from services_python.admin_api import (
    CreditsHandler,
    HealthHandler,
    JobsHandler,
    LedgerHandler,
    MultipliersHandler,
    NodesHandler,
    SybilHandler,
    V1Handler,
    VotesHandler,
)
from services_python.admin_auth import (
    admin_auth_middleware,
    log_production_security_warnings,
    validate_production_startup,
)
from services_python.constants import (
    DEFAULT_ADMIN_HOST,
    DEFAULT_ADMIN_PORT,
    DEFAULT_GRPC_PORT,
    JWT_ALGORITHM,
    JWT_SECRET,
    S3_DEFAULT_REGION,
    SIGNING_KEY,
)
from services_python.correlation_id import correlation_id_middleware
from services_python.credit_multipliers import CreditMultiplierEngine
from services_python.credit_transfers import CreditTransferManager
from services_python.db_manager import DBManager
from services_python.distribai_registry_sync import DistribAIRegistrySync
from services_python.grpc_service import GrpcServiceHandler
from services_python.grpo_coordinator import get_grpo_coordinator
from services_python.poc_challenge import PoCChallengeManager
from services_python.rate_limiter import create_rate_limiter, rate_limit_middleware
from services_python.rebenchmark_triggers import RebenchmarkTriggerManager
from services_python.scheduler import TaskScheduler
from services_python.sse_limits import SseByteBudget, admin_sse_limiter
from services_python.sybil_detector import SybilDetector
from worker.src.daemon.credit_ledger import CreditLedger
from worker.src.daemon.voting_system import VotingSystem
from worker.src.distribai_proto import distribai_pb2_grpc

try:
    from services_python.job_submission import JobSubmissionHandler, create_distributor

    JOB_SUBMISSION_AVAILABLE = True
except ImportError:
    JOB_SUBMISSION_AVAILABLE = False

# Optional auto-update subsystem
try:
    from services_python.auto_update import UpdateService

    AUTO_UPDATE_AVAILABLE = True
except ImportError:
    AUTO_UPDATE_AVAILABLE = False

from services_python.monitoring import health_checker, metrics_collector, profiler

logger = logging.getLogger(__name__)
load_dotenv()


def _safe_import_aiohttp_web():
    """Resolve aiohttp.web or raise a clear ImportError."""
    try:
        from aiohttp import web

        return web
    except ImportError as e:
        logger.error("aiohttp.web module not available: %s", e)
        raise


def _safe_import_aiohttp_cors():
    """Resolve aiohttp_cors or raise a clear ImportError."""
    try:
        import aiohttp_cors

        return aiohttp_cors
    except ImportError as e:
        logger.error("aiohttp_cors module not available: %s", e)
        raise


# Placeholder; RateLimiterAppKey imported when building the app
AppKeyRateLimiter = None

# Eagerly bind web/cors helpers used below
web = _safe_import_aiohttp_web()
aiohttp_cors = _safe_import_aiohttp_cors()


class LazyRobustAggregator:
    """Defer Torch Byzantine aggregation until gradients actually arrive."""

    def __init__(self) -> None:
        self._impl = None

    def _get_impl(self):
        if self._impl is None:
            from worker.src.daemon.byzantine_detector import AggregationMethod, RobustAggregator

            self._impl = RobustAggregator(method=AggregationMethod.TRIMMED_MEAN)
        return self._impl

    def detect_anomalies(self, updates):
        return self._get_impl().detect_anomalies(updates)

    def aggregate(self, updates):
        return self._get_impl().aggregate(updates)


@dataclass
class OrchestratorRuntime:
    """Holds live gRPC server, admin AppRunner, and NodeService.

    Owns start/stop lifecycle for the three control-plane pieces.

    Attributes:
        server: Active ``grpc.aio.Server``.
        admin_runner: aiohttp ``AppRunner`` for admin REST.
        node_service: Session/state hub for worker connections.
    """

    server: grpc.aio.Server
    admin_runner: web.AppRunner
    node_service: NodeService

    async def wait(self) -> None:
        """Block until the gRPC server shuts down."""
        await self.server.wait_for_termination()

    async def stop(self) -> None:
        """Graceful teardown: node service, gRPC, then admin runner."""
        await self.node_service.close()
        await self.server.stop(grace=5)
        await self.admin_runner.cleanup()


class NodeService:
    """Central state hub for workers, credits, and scheduling.

    Coordinates:
    - bidirectional gRPC (``GrpcServiceHandler``)
    - admin REST (``admin_api`` handlers)
    - queue timing (``TaskScheduler``)

    Attributes:
        db: Persistent SQLite facade.
        connected_nodes: node_id -> outbound message queue.
        pending_assignments: node_id -> active task id.
        log_lines: ring buffer of recent log text.
        credit_ledger: signed in-memory credit chain.
        voting_system: priority / governance votes.
        byzantine_detector: lazy robust gradient aggregator.
    """

    def __init__(self, db: DBManager) -> None:
        self.db = db
        self.connected_nodes: dict[str, asyncio.Queue] = {}
        self.pending_assignments: dict[str, str] = {}
        self.script_packages: dict[str, bytes] = {}
        self.log_lines: deque[str] = deque(maxlen=1000)
        self._closed = False

        # Optional S3 client when bucket env is set
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.s3_client = None
        if self.bucket_name:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION", S3_DEFAULT_REGION),
                config=Config(signature_version="s3v4"),
            )

        # Control-plane helpers (rate limit, PoC, multipliers, Sybil)
        self.rate_limiter = create_rate_limiter()
        self.poc_challenge = PoCChallengeManager()
        self.credit_multipliers = CreditMultiplierEngine()
        self.rebenchmark = RebenchmarkTriggerManager()
        self.sybil_detector = SybilDetector()
        self.node_metadata: dict[str, dict] = {}

        # Keep ML/Torch off the critical path during control-plane boot.
        checkpoint_path = Path(__file__).resolve().parent.parent / "runtime" / "checkpoints"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.ml_state = None

        # Voting + signed ledger (+ SQL replay)
        self.voting_system = VotingSystem(signing_key=SIGNING_KEY)
        self.credit_ledger = CreditLedger(signing_key=SIGNING_KEY)
        self._replay_signed_ledger_from_sql()
        self.credit_transfers = CreditTransferManager(db, credit_ledger=self.credit_ledger)
        self.byzantine_detector = LazyRobustAggregator()
        self.gradient_compressor = None

        # DiLoCo coordinator is created on first use
        self.diloco_coordinator = None  # lazy: from .diloco import DiLoCoCoordinator

        # Shared GRPO job coordinator singleton
        self.grpo_coordinator = get_grpo_coordinator()

        # Background task scheduler tied to this service
        self.scheduler = TaskScheduler(db, self)

        # Stream RPC implementation bound to this service
        self.grpc_handler = GrpcServiceHandler(self)

        # Kick scheduler loop; distributor starts later if enabled
        self.scheduler_task = asyncio.create_task(self.scheduler.start())
        self._distributor_task: asyncio.Task | None = None

        # Metrics: start the collector and periodically record fleet snapshots
        # so /admin/metrics and the JSON summaries stop returning "No metrics
        # available". Interval is configurable for tests.
        self._metrics_interval = float(os.getenv("DISTRIBAI_METRICS_INTERVAL_SECONDS", "15"))
        self._metrics_task: asyncio.Task | None = asyncio.create_task(self._metrics_loop())

    def record_credit_earn(
        self,
        node_id: str,
        amount: float,
        job_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write an earn into SQL balances and the signed in-memory chain."""
        meta = dict(metadata or {})
        if job_id:
            meta.setdefault("job_id", job_id)
        amount_f = float(amount)
        if amount_f > 0:
            self.credit_ledger.credit(node_id, amount_f, job_id, meta)
        self.db.add_credits(node_id, amount_f, "earn", meta)

    def _replay_signed_ledger_from_sql(self) -> None:
        """Replay SQL credit_ledger rows into the in-process signed chain after boot."""
        rows = self.db.iter_credit_ledger_rows()
        if not rows:
            return
        for row in rows:
            meta: dict[str, Any] = {}
            tx_hash = row.get("tx_hash")
            prev_hash = row.get("prev_hash")
            if tx_hash:
                meta["tx_hash"] = tx_hash
            if prev_hash:
                meta["prev_hash"] = prev_hash
            self.credit_ledger.append_record(
                str(row["node_id"]),
                str(row["tx_type"]),
                float(row["amount"]),
                meta,
            )
        if not self.credit_ledger.verify_chain_integrity():
            logger.warning(
                "Signed ledger replay from SQL failed chain verification (%s rows)",
                len(rows),
            )

    def ledger_parity_summary(self) -> dict[str, Any]:
        """Probe SQL vs in-memory credit drift after restarts."""
        drift: list[dict[str, Any]] = []
        for node_id, info in self.db.list_all_credits().items():
            sql_balance = float(info.get("balance", 0))
            memory_balance = self.credit_ledger.get_balance(node_id)
            if abs(memory_balance - sql_balance) > 1e-6:
                drift.append(
                    {
                        "node_id": node_id,
                        "sql_balance": sql_balance,
                        "memory_balance": memory_balance,
                    }
                )
        return {
            "chain_ok": self.credit_ledger.verify_chain_integrity(),
            "memory_records": self.credit_ledger.size(),
            "drift_count": len(drift),
            "drift_sample": drift[:10],
            "signing_key_from_env": bool(os.getenv("SIGNING_KEY", "").strip()),
        }

    async def _metrics_loop(self) -> None:
        """Sample host + orchestrator metrics on an interval for observability."""
        await metrics_collector.start(interval_seconds=self._metrics_interval)
        while not self._closed:
            try:
                await metrics_collector.collect_system_metrics()
                total_credits = await asyncio.to_thread(self._total_credits_distributed)
                await metrics_collector.record_orchestrator_snapshot(
                    connected_nodes=len(self.connected_nodes),
                    active_jobs=len(self.pending_assignments),
                    queued_jobs=await asyncio.to_thread(self.db.get_queue_depth),
                    total_credits_distributed=total_credits,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Metrics snapshot failed")
            try:
                await asyncio.sleep(self._metrics_interval)
            except asyncio.CancelledError:
                raise

    def _total_credits_distributed(self) -> float:
        """Sum lifetime credits across nodes (best-effort for the metrics loop)."""
        try:
            return sum(float(info.get("lifetime") or 0) for info in self.db.list_all_credits().values())
        except Exception:
            logger.exception("Failed to total lifetime credits for metrics")
            return 0.0

    async def close(self) -> None:
        """Idempotent close: cancel distributor, metrics, and stop the scheduler."""
        if self._closed:
            return
        self._closed = True
        dt = self._distributor_task
        if dt and not dt.done():
            dt.cancel()
            try:
                await dt
            except asyncio.CancelledError:
                pass
        self._distributor_task = None
        mt = self._metrics_task
        if mt and not mt.done():
            mt.cancel()
            try:
                await mt
            except asyncio.CancelledError:
                pass
        self._metrics_task = None
        await metrics_collector.stop()
        await self.scheduler.stop()

    def _issue_jwt(self, subject: str, kind: str = "node", expires_in: int = 21600) -> str:
        """Mint a signed JWT for node/admin auth.

        Args:
            subject: Usually the node_id (JWT ``sub``).
            kind: Token class such as node or admin.
            expires_in: Lifetime in seconds from now.

        Returns:
            Encoded compact JWT string.
        """
        import time as time_module

        now = int(time_module.time())
        return jwt.encode(
            {
                "sub": subject,
                "kind": kind,
                "iat": now,
                "exp": now + expires_in,
            },
            JWT_SECRET,
            algorithm=JWT_ALGORITHM,
        )

    def verify_jwt(
        self, token: str, expected_subject: str | None = None, kind: str | None = None
    ) -> dict[str, Any] | None:
        """Validate signature/exp and optionally subject/kind.

        Args:
            token: Compact JWT to decode.
            expected_subject: If set, require matching ``sub``.
            kind: If set, require matching ``kind`` claim.

        Returns:
            Claims dict on success, otherwise None.
        """
        import jwt as jwt_module

        if not token:
            return None
        try:
            claims = jwt_module.decode(
                token,
                JWT_SECRET,
                algorithms=[JWT_ALGORITHM],
                options={"verify_exp": True},
            )
        except jwt_module.InvalidTokenError as exc:
            logger.warning("JWT verification failed: %s", exc)
            return None

        if expected_subject and claims.get("sub") != expected_subject:
            logger.warning(
                "JWT subject mismatch: expected=%s got=%s", expected_subject, claims.get("sub")
            )
            return None

        if kind and claims.get("kind") != kind:
            logger.warning("JWT kind mismatch: expected=%s got=%s", kind, claims.get("kind"))
            return None

        return claims

    def generate_presigned_url(
        self, key: str, operation: str = "get_object", expiration: int = 3600
    ) -> str | None:
        """Build a time-limited S3 URL when the bucket client is configured.

        Args:
            key: Object key inside the configured bucket.
            operation: Client method name (get_object / put_object).
            expiration: Seconds until the URL expires.

        Returns:
            Presigned URL string, or None when S3 is unavailable.
        """
        from services_python.harness_policy import skip_presigned_s3_urls

        if not self.s3_client or not self.bucket_name:
            return None
        if skip_presigned_s3_urls():
            logger.debug(
                "Skipping presigned S3 URL for %s (fast-test harness with mock credentials)",
                key,
            )
            return None
        try:
            return self.s3_client.generate_presigned_url(
                ClientMethod=operation,
                Params={"Bucket": self.bucket_name, "Key": key},
                ExpiresIn=expiration,
            )
        except Exception as exc:
            logger.error("Error generating pre-signed URL: %s", exc)
            return None

    def _authenticate_request(
        self, req: web.Request, required_kind: str | None = None
    ) -> dict[str, Any]:
        """Require Bearer admin secret or JWT (loopback relaxes in dev).

        Args:
            req: Incoming aiohttp request.
            required_kind: Optional JWT kind constraint.

        Returns:
            Claims dict used by handlers.

        Raises:
            web.HTTPUnauthorized: Missing/invalid credentials when enforced.
        """
        import hmac

        from services_python.admin_auth import admin_auth_enforced, resolve_admin_secret

        auth_header = req.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            if not admin_auth_enforced():
                return {"sub": "loopback-dev", "kind": required_kind or "admin"}
            raise web.HTTPUnauthorized(
                text=json.dumps({"error": "missing bearer token"}),
                content_type="application/json",
            )
        token = auth_header.split(" ", 1)[1]
        secret = resolve_admin_secret()
        if secret and hmac.compare_digest(token, secret):
            return {"sub": "admin-bearer", "kind": required_kind or "admin"}
        claims = self.verify_jwt(token, kind=required_kind)
        if not claims:
            raise web.HTTPUnauthorized(
                text=json.dumps({"error": "invalid token"}),
                content_type="application/json",
            )
        return claims

    def _safe_json(self, payload: str | dict[str, Any] | None) -> dict[str, Any]:
        """Coerce str/dict JSON into a dict; empty on failure.

        Args:
            payload: Raw JSON text or already-parsed mapping.

        Returns:
            Dict result, or {} when empty/invalid.
        """
        if payload in (None, ""):
            return {}
        if isinstance(payload, dict):
            return payload
        try:
            return json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return {}

    async def broadcast_control(
        self,
        action: str,
        target_id: str,
        node_ids: list[str] | None = None,
    ) -> None:
        """Best-effort ControlMessage fan-out to live worker queues."""
        from worker.src.distribai_proto import distribai_pb2

        msg = distribai_pb2.ServerMessage(
            control=distribai_pb2.ControlMessage(
                action=action,
                target_id=target_id,
            )
        )
        targets = list(node_ids) if node_ids else list(self.connected_nodes.keys())
        for nid in targets:
            queue = self.connected_nodes.get(nid)
            if queue is not None:
                await queue.put(msg)


# ---------------------------------------------------------------------------
# Process-wide runtime registry keyed by (grpc_port, admin_port)
# ---------------------------------------------------------------------------

_RUNTIMES: dict[tuple[str, str], OrchestratorRuntime] = {}


def _db_path_for_ports(grpc_port: str, admin_port: str) -> str:
    """Pick SQLite path under runtime/db (or DISTRIBAI_DB_DIR) from ports."""
    override_dir = os.getenv("DISTRIBAI_DB_DIR", "").strip()
    base_dir = Path(override_dir) if override_dir else Path(__file__).resolve().parent.parent / "runtime" / "db"
    base_dir.mkdir(parents=True, exist_ok=True)
    if grpc_port == DEFAULT_GRPC_PORT and admin_port == DEFAULT_ADMIN_PORT:
        return str(base_dir / "distribai.db")
    return str(base_dir / f"distribai-{grpc_port}-{admin_port}.db")


# TLS helpers: prefer TLS; fail closed when DISTRIBAI_ENV=production
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CERT_PATH = _REPO_ROOT / "runtime" / "secrets" / "tls" / "server.crt"
_DEFAULT_KEY_PATH = _REPO_ROOT / "runtime" / "secrets" / "tls" / "server.key"


def _is_production() -> bool:
    """Whether DISTRIBAI_ENV names production (default development)."""
    return os.getenv("DISTRIBAI_ENV", "development").lower() == "production"


def _setup_grpc_tls(server: grpc.aio.Server, grpc_port: str) -> bool:
    """Attach gRPC server credentials via grpc_tls helpers."""
    from services_python.grpc_tls import configure_grpc_server

    return configure_grpc_server(server, grpc_port, grpc_listen="[::]")


def _setup_admin_tls(admin_host: str):
    """Optional SSLContext for the admin HTTP listener."""
    import ssl

    explicit = os.getenv("ADMIN_USE_TLS")
    is_loopback = admin_host in ("127.0.0.1", "::1", "localhost")
    if explicit is not None:
        enabled = explicit.lower() == "true"
    else:
        enabled = not is_loopback

    if not enabled:
        if not is_loopback and _is_production():
            raise RuntimeError(
                f"Admin API bound to non-loopback {admin_host} but ADMIN_USE_TLS=false in production. "
                "Plaintext admin on an exposed port is never OK."
            )
        return None

    cert_path = Path(os.getenv("ADMIN_TLS_CERT", os.getenv("GRPC_TLS_CERT", str(_DEFAULT_CERT_PATH))))
    key_path = Path(os.getenv("ADMIN_TLS_KEY", os.getenv("GRPC_TLS_KEY", str(_DEFAULT_KEY_PATH))))

    if not cert_path.exists() or not key_path.exists():
        if _is_production():
            raise RuntimeError(
                f"Admin TLS enabled but cert ({cert_path}) or key ({key_path}) missing. "
                f"Run: python scripts/dev/gen_tls_certs.py --hostname <your-host>"
            )
        logger.warning(
            "Admin TLS certs missing at %s / %s; auto-generating self-signed for dev.",
            cert_path,
            key_path,
        )
        from scripts.dev.gen_tls_certs import auto_generate_dev_cert

        auto_generate_dev_cert(cert_path, key_path)

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(str(cert_path), str(key_path))
    logger.info("Admin REST API TLS enabled (host=%s, cert=%s).", admin_host, cert_path)
    return ssl_context


def _make_admin_app(node_service: NodeService) -> web.Application:
    """Assemble the aiohttp admin app, CORS, handlers, and routes.

    Args:
        node_service: Shared orchestrator state for handlers.

    Returns:
        Fully routed ``web.Application``.
    """
    app = web.Application(middlewares=[correlation_id_middleware, admin_auth_middleware, rate_limit_middleware])

    from services_python.correlation_id import ensure_correlation_logging

    ensure_correlation_logging()

    # Store rate limiter under AppKey (aiohttp typed storage)
    from services_python.rate_limiter import RateLimiterAppKey

    app[RateLimiterAppKey] = node_service.rate_limiter

    # CORS allow-list from policy (localhost dashboards when env unset)
    from services_python.cors_policy import cors_origins_list

    cors_origins = cors_origins_list()
    cors = aiohttp_cors.setup(
        app,
        defaults={
            origin: aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            )
            for origin in cors_origins
        },
    )

    # Construct per-resource admin handlers
    health_handler = HealthHandler(node_service.db, node_service)
    jobs_handler = JobsHandler(node_service.db, node_service)
    nodes_handler = NodesHandler(node_service.db, node_service)
    credits_handler = CreditsHandler(
        node_service.db, node_service.credit_ledger, node_service.credit_transfers, node_service
    )
    votes_handler = VotesHandler(node_service.db, node_service.voting_system, node_service)
    ledger_handler = LedgerHandler(node_service.credit_ledger)
    multipliers_handler = MultipliersHandler(node_service.credit_multipliers, node_service)
    sybil_handler = SybilHandler(node_service.sybil_detector)
    v1_handler = V1Handler(node_service.db, node_service)
    repo_root = Path(__file__).resolve().parent.parent

    def _client_ip(request: web.Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For", "").strip()
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.remote or "unknown"

    async def admin_stream(req: web.Request) -> web.StreamResponse:
        client_ip = _client_ip(req)
        limiter = admin_sse_limiter()
        if not await limiter.try_acquire(client_ip):
            return web.json_response(
                {
                    "error": "too_many_sse_connections",
                    "message": "Admin SSE connection limit reached",
                },
                status=429,
            )

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        byte_budget = SseByteBudget()
        try:
            await response.prepare(req)
            try:
                while True:
                    queue_depth = await asyncio.to_thread(node_service.db.get_queue_depth)
                    grpo_jobs = node_service.grpo_coordinator.list_jobs()
                    payload = {
                        "type": "snapshot",
                        "ts": int(time.time() * 1000),
                        "nodes": len(node_service.connected_nodes),
                        "queue_depth": queue_depth,
                        "logs": list(node_service.log_lines)[-25:],
                        "grpo": {
                            "active_jobs": sum(1 for j in grpo_jobs if j["status"] == "running"),
                            "jobs": grpo_jobs,
                        },
                    }
                    chunk = f"data: {json.dumps(payload)}\n\n".encode()
                    await byte_budget.wait_for(len(chunk))
                    await response.write(chunk)
                    await asyncio.sleep(2)
            except ConnectionResetError:
                pass
            except asyncio.CancelledError:
                raise
            return response
        finally:
            await limiter.release(client_ip)

    async def distribai_registry_sync(req: web.Request) -> web.Response:
        """Sync and return the native DistribAI model registry payload."""
        sync = DistribAIRegistrySync(orchestrator_db=node_service.db)
        configs = await asyncio.to_thread(sync.sync)
        return web.json_response(
            {
                "ok": True,
                "synced_count": len(configs),
                "models": sorted(configs.keys()),
                "registry": configs,
            }
        )

    async def public_release_publish(req: web.Request) -> web.Response:
        try:
            body = await req.json() if req.can_read_body else {}
        except json.JSONDecodeError:
            body = {}
        script = repo_root / "scripts" / "publish_public_grid.py"
        if not script.exists():
            return web.json_response({"error": "publish script missing"}, status=500)
        args = [sys.executable, str(script)]
        if body.get("push", True) is False:
            args.append("--no-push")
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                args,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return web.json_response({"error": "publish timed out"}, status=504)
        return web.json_response(
            {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-8000:],
                "stderr": proc.stderr[-8000:],
            },
            status=200 if proc.returncode == 0 else 500,
        )

    async def docs_list(req: web.Request) -> web.Response:
        docs_root = repo_root / "docs"
        docs = []
        readme = repo_root / "README.md"
        if readme.exists():
            docs.append({"path": "README.md", "title": "README"})
        if docs_root.exists():
            for path in sorted(docs_root.rglob("*.md")):
                rel = path.relative_to(repo_root).as_posix()
                docs.append({"path": rel, "title": path.stem.replace("-", " ").title()})
        return web.json_response(docs)

    async def docs_read(req: web.Request) -> web.Response:
        rel_path = req.query.get("path", "")
        if not rel_path:
            return web.json_response({"error": "path required"}, status=400)
        target = (repo_root / rel_path).resolve()
        docs_root = (repo_root / "docs").resolve()
        if target != (repo_root / "README.md").resolve() and not (
            target == docs_root or docs_root in target.parents
        ):
            return web.json_response({"error": "path outside docs"}, status=403)
        if not target.exists() or target.suffix.lower() != ".md":
            return web.json_response({"error": "doc not found"}, status=404)
        return web.json_response(
            {
                "path": target.relative_to(repo_root).as_posix(),
                "content": target.read_text(encoding="utf-8", errors="replace"),
            }
        )

    async def logs_handler(req: web.Request) -> web.Response:
        return web.json_response(
            {"logs": list(node_service.log_lines)[-min(int(req.query.get("n", 100)), 1000) :]}
        )

    async def metrics_system_handler(req: web.Request) -> web.Response:
        return web.json_response(metrics_collector.get_system_metrics_summary())

    async def metrics_orchestrator_handler(req: web.Request) -> web.Response:
        return web.json_response(metrics_collector.get_orchestrator_summary())

    async def metrics_node_handler(req: web.Request) -> web.Response:
        return web.json_response(metrics_collector.get_node_metrics(req.match_info["node_id"]))

    async def metrics_prometheus_handler(req: web.Request) -> web.Response:
        """Prometheus text-format exposition of orchestrator metrics."""
        body = metrics_collector.render_prometheus()
        return web.Response(
            text=body,
            content_type="text/plain",
            charset="utf-8",
        )

    async def diloco_register_handler(req: web.Request) -> web.Response:
        """Register a job's canonical weights + outer-step hyperparameters.

        Required before workers may stream ``diloco_pseudo_gradient`` for the
        job. Body: ``{initial_weights: {name: nested-list}, outer_lr?,
        outer_momentum?, H?, min_workers?}``.
        """
        node_service._authenticate_request(req, required_kind="admin")
        job_id = req.match_info.get("job_id", "")
        if not job_id:
            return web.json_response({"error": "missing job_id"}, status=400)
        try:
            body = await req.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)
        if not isinstance(body, dict) or not isinstance(body.get("initial_weights"), dict):
            return web.json_response({"error": "initial_weights object required"}, status=400)
        weights = node_service.grpc_handler._json_to_numpy_weights(body["initial_weights"])
        if not weights:
            return web.json_response({"error": "no numeric weight tensors provided"}, status=400)
        coordinator = node_service.grpc_handler._ensure_diloco_coordinator()

        def _clamp(name: str, default: float, low: float, high: float) -> float:
            try:
                return max(low, min(high, float(body.get(name, default))))
            except (TypeError, ValueError):
                return default

        await coordinator.register_job(
            job_id,
            weights,
            outer_lr=_clamp("outer_lr", 0.7, 1e-4, 10.0),
            outer_momentum=_clamp("outer_momentum", 0.9, 0.0, 0.9999),
            H=int(_clamp("H", 500, 1, 1_000_000)),
            min_workers=int(_clamp("min_workers", 2, 1, 4096)),
        )
        return web.json_response({"ok": True, "job_id": job_id, "round_id": 0})

    async def diloco_status_handler(req: web.Request) -> web.Response:
        """Report a DiLoCo job's current outer round, or 404 when unregistered."""
        node_service._authenticate_request(req, required_kind="admin")
        job_id = req.match_info.get("job_id", "")
        coordinator = getattr(node_service, "diloco_coordinator", None)
        if coordinator is None or not coordinator.has_job(job_id):
            return web.json_response({"error": "DiLoCo job not found"}, status=404)
        return web.json_response({"ok": True, "job_id": job_id, "round_id": coordinator.round_id(job_id)})

    async def health_detailed_handler(req: web.Request) -> web.Response:
        return web.json_response(await health_checker.run_all_checks())

    async def operator_status_handler(req: web.Request) -> web.Response:
        """Non-secret public-release banner fields for dashboards."""
        from services_python.admin_auth import admin_auth_enforced
        from services_python.cors_policy import cors_is_permissive
        from services_python.registration_policy import registration_requires_poc

        admin_host = os.getenv("ADMIN_HOST", DEFAULT_ADMIN_HOST)
        ledger_parity = node_service.ledger_parity_summary()
        return web.json_response(
            {
                "ok": True,
                "admin_host": admin_host,
                "admin_auth_enforced": admin_auth_enforced(),
                "registration_requires_poc": registration_requires_poc(),
                "grpc_tls": os.getenv("GRPC_USE_TLS", "false").lower() == "true",
                "cors_permissive": cors_is_permissive(),
                "signing_key_from_env": bool(os.getenv("SIGNING_KEY", "").strip()),
                "jwt_secret_from_env": bool(os.getenv("JWT_SECRET", "").strip()),
                "ledger_chain_ok": ledger_parity["chain_ok"],
                "ledger_memory_records": ledger_parity["memory_records"],
                "ledger_sql_memory_drift_count": ledger_parity["drift_count"],
            }
        )

    async def stats_handler(req: web.Request) -> web.Response:
        """Aggregate fleet/queue/credit stats for the operator dashboard."""
        active_nodes = len(node_service.connected_nodes)
        queued_jobs = node_service.db.get_queue_depth()
        running_jobs = len(node_service.pending_assignments)
        connected_ids = frozenset(node_service.connected_nodes.keys())

        def _compute_dashboard_totals() -> tuple[float, float, float]:
            credits_map = node_service.db.list_all_credits()
            total_lifetime = sum(info["lifetime"] for info in credits_map.values())
            total_explicit_tflops = 0.0
            total_benchmark_score = 0.0
            for n in node_service.db.get_all_nodes():
                if n["node_id"] not in connected_ids:
                    continue
                bench = n.get("benchmark")
                if not isinstance(bench, dict):
                    continue
                added_tflops = False
                for key in ("tflops", "total_tflops", "gpu_tflops"):
                    raw = bench.get(key)
                    if isinstance(raw, (int, float)) and raw == raw:  # not NaN
                        total_explicit_tflops += float(raw)
                        added_tflops = True
                        break
                if not added_tflops:
                    score = bench.get("overall_score")
                    if isinstance(score, (int, float)) and score == score:
                        total_benchmark_score += float(score)
            return total_lifetime, total_explicit_tflops, total_benchmark_score

        total_credits, total_tflops, total_benchmark_score = await asyncio.to_thread(
            _compute_dashboard_totals
        )

        grpo_jobs = node_service.grpo_coordinator.list_jobs()
        grpo_active = sum(1 for j in grpo_jobs if j["status"] == "running")
        grpo_total_rounds = sum(j.get("current_round", 0) for j in grpo_jobs)

        return web.json_response(
            {
                "ok": True,
                "active_nodes": active_nodes,
                "running_jobs": running_jobs,
                "queued_jobs": queued_jobs,
                "credits_distributed": total_credits,
                "total_tflops": total_tflops,
                "total_benchmark_score": total_benchmark_score,
                "grpo": {
                    "active_jobs": grpo_active,
                    "total_rounds": grpo_total_rounds,
                    "jobs": grpo_jobs,
                },
            }
        )

    async def profiles_handler(req: web.Request) -> web.Response:
        return web.json_response({"profiles": profiler.get_all_profiles()})

    async def _handle_update_discovery(req: web.Request) -> web.Response:
        """Respond to worker update-discovery probes."""
        update_url = os.getenv("GITHUB_UPDATE_URL", "")
        exe_file = os.getenv("GITHUB_EXE_FILE", "")
        app_file = os.getenv("GITHUB_APP_FILE", "")

        response_data = {
            "update_url": update_url,
            "exe_file": exe_file,
            "app_file": app_file,
            "current_version": "1.0.0",
            "auto_update_enabled": bool(update_url),
        }

        # Probe UpdateService when the optional module and URL are present
        if AUTO_UPDATE_AVAILABLE and update_url:
            try:
                update_service = UpdateService(update_url, "1.0.0")
                update_info = update_service.check_for_updates()
                response_data.update(
                    {
                        "latest_version": update_info.get("latest_version")
                        or update_info.get("version")
                        or response_data.get("current_version", "1.0.0"),
                        "update_available": update_info.get("update_available", False),
                        "update_notes": update_info.get("notes", ""),
                    }
                )
                if update_info.get("update_available"):
                    response_data.update(
                        {
                            "download_url": update_info.get("download_url"),
                            "size_mb": update_info.get("size_mb"),
                            "hash": update_info.get("hash"),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to check for updates: {e}")
                response_data["update_check_error"] = str(e)

        return web.json_response(response_data)

    # ── GRPO Admin Endpoints ────────────────────────────────────────────
    async def grpo_list_jobs(req: web.Request) -> web.Response:
        """Enumerate active GRPO jobs for operators."""
        jobs = node_service.grpo_coordinator.list_jobs()
        return web.json_response({"ok": True, "grpo_jobs": jobs})

    async def grpo_get_job(req: web.Request) -> web.Response:
        """Detail payload for one GRPO job id."""
        job_id = req.match_info.get("job_id")
        status_data = node_service.grpo_coordinator.get_status(job_id)
        if status_data is None:
            return web.json_response({"error": "GRPO job not found"}, status=404)
        return web.json_response({"ok": True, "job": status_data})

    async def grpo_cancel_job(req: web.Request) -> web.Response:
        """Request cancellation of a running GRPO job."""
        job_id = req.match_info.get("job_id")
        job = node_service.grpo_coordinator.get_job(job_id)
        if job is None:
            return web.json_response({"error": "GRPO job not found"}, status=404)
        node_service.grpo_coordinator.fail_job(job_id, reason="cancelled by admin")
        return web.json_response({"ok": True, "status": "cancelled"})

    async def rebenchmark_trigger(req: web.Request) -> web.Response:
        """Enqueue a live benchmark against one node or the whole fleet."""
        try:
            body = await req.json() if req.can_read_body else {}
        except (json.JSONDecodeError, ValueError):
            return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"ok": False, "error": "request body must be an object"}, status=400)

        requested_node = str(body.get("node_id", "")).strip()
        if requested_node:
            if requested_node not in node_service.connected_nodes:
                return web.json_response(
                    {"ok": False, "error": "node is not connected", "node_id": requested_node},
                    status=404,
                )
            node_ids = [requested_node]
        else:
            node_ids = sorted(node_service.connected_nodes)

        for node_id in node_ids:
            node_service.rebenchmark.schedule_rebenchmark(node_id)
        await node_service.broadcast_control("benchmark", "", node_ids=node_ids)
        return web.json_response(
            {
                "ok": True,
                "scheduled": len(node_ids),
                "node_ids": node_ids,
                "message": "No connected nodes available" if not node_ids else "Benchmark queued",
            }
        )

    async def trust_submitters_list(req: web.Request) -> web.Response:
        rows = await asyncio.to_thread(node_service.db.list_trusted_submitters)
        return web.json_response({"ok": True, "submitters": rows})

    async def trust_submitter_add(req: web.Request) -> web.Response:
        node_id = req.match_info.get("node_id", "")
        try:
            await asyncio.to_thread(node_service.db.add_trusted_submitter, node_id)
        except ValueError:
            return web.json_response({"ok": False, "error": "invalid node_id"}, status=400)
        return web.json_response({"ok": True, "node_id": node_id})

    async def trust_submitter_remove(req: web.Request) -> web.Response:
        node_id = req.match_info.get("node_id", "")
        try:
            removed = await asyncio.to_thread(node_service.db.remove_trusted_submitter, node_id)
        except ValueError:
            return web.json_response({"ok": False, "error": "invalid node_id"}, status=400)
        return web.json_response({"ok": True, "removed": removed})

    if JOB_SUBMISSION_AVAILABLE:
        _job_submission_handler = JobSubmissionHandler(db=node_service.db)

    # Route table
    routes = [
        # Liveness
        ("GET", "/admin/health", health_handler.get),
        ("GET", "/admin/stats", stats_handler),
        ("GET", "/admin/logs", logs_handler),
        ("GET", "/admin/stream", admin_stream),
        # Operator control
        ("POST", "/api/admin/rebenchmark/trigger", rebenchmark_trigger),
        ("POST", "/api/admin/distribai/registry/sync", distribai_registry_sync),
        ("POST", "/api/admin/public-release/publish", public_release_publish),
        ("GET", "/api/docs/list", docs_list),
        ("GET", "/api/docs/read", docs_read),
        ("GET", "/api/operator/status", operator_status_handler),
        # Fleet nodes
        ("GET", "/admin/nodes", nodes_handler.list),
        ("POST", "/admin/nodes/sync-all", nodes_handler.sync_all),
        ("POST", "/admin/sync-all", nodes_handler.sync_all),
        ("POST", "/admin/nodes/{node_id}/contributing", nodes_handler.set_contributing),
        ("POST", "/admin/nodes/{node_id}/disconnect", nodes_handler.disconnect),
        ("GET", "/admin/nodes/paginated", nodes_handler.list_paginated),
        # Jobs — static segments before {job_id} (paginated is not an id)
        ("GET", "/admin/jobs", jobs_handler.list),
        ("GET", "/admin/jobs/compare", jobs_handler.compare),
        ("GET", "/admin/jobs/paginated", jobs_handler.list_paginated),
        ("POST", "/admin/jobs/estimate", jobs_handler.estimate_cost),
        ("POST", "/admin/jobs/recalculate-priorities", jobs_handler.recalculate_priorities),
        ("POST", "/admin/recalculate-priorities", jobs_handler.recalculate_priorities),
        ("POST", "/admin/clear-completed", jobs_handler.clear_completed),
        ("DELETE", "/admin/jobs/completed", jobs_handler.clear_completed),
        ("GET", "/admin/jobs/{job_id}", jobs_handler.get),
        ("GET", "/admin/jobs/{job_id}/artifacts", jobs_handler.artifacts),
        ("POST", "/admin/jobs", jobs_handler.create),
        ("POST", "/admin/jobs/{job_id}/retry", jobs_handler.retry),
        ("POST", "/admin/jobs/{job_id}/cancel", jobs_handler.cancel),
        ("DELETE", "/admin/jobs/{job_id}", jobs_handler.cancel),
        # Credits — paginated before {node_id}
        ("GET", "/admin/credits", credits_handler.list),
        ("GET", "/admin/credits/paginated", credits_handler.list_paginated),
        ("GET", "/admin/credits/{node_id}", credits_handler.get),
        ("GET", "/admin/transfers/stats", credits_handler.get_transfer_stats),
        # Voting
        ("GET", "/admin/votes", votes_handler.list),
        ("GET", "/admin/votes/{vote_id}", votes_handler.get),
        ("POST", "/admin/votes", votes_handler.create),
        ("POST", "/admin/votes/{vote_id}/cast", votes_handler.cast),
        # Credit ledger
        ("GET", "/admin/ledger/root", ledger_handler.get_root),
        ("GET", "/admin/ledger/verify/{index}", ledger_handler.verify_record),
# Multipliers and Sybil
         ("GET", "/admin/multipliers/stats", multipliers_handler.get_stats),
         ("POST", "/admin/multipliers/surge", multipliers_handler.trigger_surge),
         ("GET", "/admin/sybil/stats", sybil_handler.get_stats),
        ("GET", "/admin/sybil/nodes/{node_id}", sybil_handler.get_node_report),
        # Org job-submission API — one shared handler so allowlist state
        # (add_allowed_org / DISTRIBAI_ALLOWED_ORGS) is consistent across
        # every route instead of resetting per-route.
        *(
            [
                ("POST", "/jobs/submit", _job_submission_handler.submit_job),
                ("GET", "/jobs", _job_submission_handler.list_jobs),
                ("GET", "/jobs/{job_id}", _job_submission_handler.get_job_status),
                ("POST", "/jobs/{job_id}/cancel", _job_submission_handler.cancel_job),
                ("GET", "/jobs/queue", _job_submission_handler.get_queue_status),
            ]
            if JOB_SUBMISSION_AVAILABLE
            else []
        ),
        # GRPO admin
        ("GET", "/admin/grpo/jobs", grpo_list_jobs),
        ("GET", "/admin/grpo/jobs/{job_id}", grpo_get_job),
        ("POST", "/admin/grpo/jobs/{job_id}/cancel", grpo_cancel_job),
        # Versioned /v1
        ("POST", "/v1/nodes/register", v1_handler.register_node),
        ("POST", "/v1/jobs", v1_handler.create_job),
        ("GET", "/v1/jobs/{job_id}", v1_handler.get_job),
        ("GET", "/v1/queue", v1_handler.get_queue),
        ("POST", "/v1/votes", votes_handler.vote_v1),
        ("GET", "/v1/votes", votes_handler.list_v1),
        ("GET", "/v1/credits/balance", credits_handler.get_balance_v1),
        ("POST", "/v1/credits/transfer", credits_handler.transfer_v1),
        ("GET", "/v1/credits/transfers", credits_handler.get_transfer_history_v1),
        ("GET", "/v1/credits/multipliers", credits_handler.get_multiplier_status_v1),
        ("POST", "/v1/credits/surge-opt-in", credits_handler.set_surge_opt_in_v1),
        ("POST", "/v1/nodes/challenge", v1_handler.request_challenge),
        ("POST", "/v1/nodes/challenge/verify", v1_handler.verify_challenge),
        ("POST", "/v1/nodes/register-enhanced", v1_handler.register_node_enhanced),
        ("POST", "/v1/nodes/benchmark-status", v1_handler.get_benchmark_status),
        ("POST", "/v1/nodes/benchmark", v1_handler.submit_benchmark),
        ("GET", "/admin/rebenchmark/stats", v1_handler.get_rebenchmark_stats),
        ("GET", "/admin/trust/submitters", trust_submitters_list),
        ("POST", "/admin/trust/submitters/{node_id}", trust_submitter_add),
        ("DELETE", "/admin/trust/submitters/{node_id}", trust_submitter_remove),
        # DiLoCo outer-step coordination
        ("POST", "/admin/diloco/{job_id}/register", diloco_register_handler),
        ("GET", "/admin/diloco/{job_id}/status", diloco_status_handler),
        # Metrics / health probes
        ("GET", "/admin/metrics", metrics_prometheus_handler),
        ("GET", "/admin/metrics/system", metrics_system_handler),
        ("GET", "/admin/metrics/orchestrator", metrics_orchestrator_handler),
        ("GET", "/admin/metrics/node/{node_id}", metrics_node_handler),
        ("GET", "/admin/health/detailed", health_detailed_handler),
        ("GET", "/admin/profiles", profiles_handler),
        # Worker update discovery
        ("GET", "/admin/update-url", _handle_update_discovery),
    ]

    for method, path, handler in routes:
        if method == "GET":
            route = app.router.add_get(path, handler)
        elif method == "POST":
            route = app.router.add_post(path, handler)
        elif method == "DELETE":
            route = app.router.add_delete(path, handler)
        else:
            continue
        cors.add(route)

    return app


async def serve(block: bool = False) -> OrchestratorRuntime:
    """Boot gRPC + admin HTTP; optionally block until shutdown.

    Args:
        block: When True, wait until the runtime terminates.

    Returns:
        OrchestratorRuntime for the (grpc_port, admin_port) pair.
    """
    logging.basicConfig(level=logging.INFO)
    log_production_security_warnings()
    validate_production_startup()

    grpc_port = os.getenv("GRPC_PORT", DEFAULT_GRPC_PORT)
    admin_port = os.getenv("ADMIN_PORT", DEFAULT_ADMIN_PORT)
    runtime_key = (grpc_port, admin_port)

    if runtime_key in _RUNTIMES:
        runtime = _RUNTIMES[runtime_key]
        if block:
            await runtime.wait()
        return runtime

    # Open SQLite for this port pair
    db_path = _db_path_for_ports(grpc_port, admin_port)
    schema_path = str(Path(__file__).resolve().parent.parent / "runtime" / "db" / "schema.sql")
    db = DBManager(db_path, schema_path)

    # Construct NodeService over that DB
    node_service = NodeService(db)

    # Build and start the gRPC server
    server = grpc.aio.server()
    distribai_pb2_grpc.add_NodeServiceServicer_to_server(node_service.grpc_handler, server)

    from services_python.grpc_tls import configure_grpc_server

    configure_grpc_server(server, grpc_port)

    await server.start()

    # Build admin app and AppRunner
    admin_app = _make_admin_app(node_service)
    runner = web.AppRunner(admin_app)
    await runner.setup()

    admin_host = os.getenv("ADMIN_HOST", DEFAULT_ADMIN_HOST)
    site = web.TCPSite(runner, admin_host, int(admin_port))
    await site.start()

    # Optional org job distributor background task
    if JOB_SUBMISSION_AVAILABLE:
        distributor = create_distributor(node_service)
        node_service._distributor_task = asyncio.create_task(distributor.distribute_loop())
        logger.info("Job distributor started")

    # Register runtime under (grpc, admin) ports
    runtime = OrchestratorRuntime(server=server, admin_runner=runner, node_service=node_service)
    _RUNTIMES[runtime_key] = runtime

    logger.info("gRPC server on port %s; admin API on %s:%s", grpc_port, admin_host, admin_port)

    if block:
        try:
            await runtime.wait()
        finally:
            await runtime.stop()
            _RUNTIMES.pop(runtime_key, None)

    return runtime


async def _main() -> None:
    """CLI entry: configure logging and run the orchestrator."""
    await serve(block=True)


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
