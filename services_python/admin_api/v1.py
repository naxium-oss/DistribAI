"""Versioned /v1 admin API: JWT auth, register, jobs, PoC, benches."""

from __future__ import annotations

import asyncio
import html
import json
import math
import os
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import jwt
from aiohttp import web

from services_python.admin_keys import get_jwt_secret
from services_python.constants import DEFAULT_STEPS_PER_TASK
from services_python.db_manager import DBManager
from services_python.registration_policy import registration_requires_poc
from services_python.schemas import validate_job_create, validate_node_register


# String/JSON sanitizers shared by v1 write paths
def sanitize_html_input(value: str) -> str:
    """HTML-escape then remove known script/event-handler substrings."""
    if not isinstance(value, str):
        return str(value)

    # Escape entities before pattern stripping
    escaped = html.escape(value)

    # Remove remaining high-risk substrings
    dangerous_patterns = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"vbscript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
    ]

    for pattern in dangerous_patterns:
        escaped = re.sub(pattern, "", escaped, flags=re.IGNORECASE | re.DOTALL)

    return escaped


def validate_json_input(data: Any, max_size: int = 10000) -> dict:
    """Bound JSON size, parse, and sanitize nested string values."""
    if isinstance(data, str):
        if len(data) > max_size:
            raise ValueError("JSON input too large")
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
    elif isinstance(data, dict):
        parsed = data
    else:
        raise ValueError("Input must be JSON string or dict")

    # Recurse into mappings/lists to clean string leaves
    def sanitize_dict(d: dict) -> dict:
        result = {}
        for key, value in d.items():
            if isinstance(value, str):
                result[key] = sanitize_html_input(value)
            elif isinstance(value, dict):
                result[key] = sanitize_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    sanitize_html_input(item) if isinstance(item, str) else item for item in value
                ]
            else:
                result[key] = value
        return result

    return sanitize_dict(parsed)


if TYPE_CHECKING:
    from services_python.orchestrator_grpc import NodeService


class AdminAPIV1:
    """JWT-gated implementation of the versioned admin contract."""

    def __init__(
        self,
        db: DBManager,
        node_service: NodeService,
        orchestrator_service: Any,
        scheduler_service: Any,
    ) -> None:
        self.db = db
        self.node_service = node_service
        self.orchestrator_service = orchestrator_service
        self.scheduler_service = scheduler_service
        self.jwt_secret = get_jwt_secret()

    def _authenticate_request(self, req: web.Request) -> dict[str, Any] | None:
        """Extract Bearer JWT claims; raise HTTP 401 on failure."""
        auth_header = req.headers.get("Authorization")
        if not auth_header:
            raise web.HTTPUnauthorized(reason="Missing Authorization header")

        try:
            scheme, token = auth_header.split(" ", 1)
            if scheme.lower() != "bearer":
                raise web.HTTPUnauthorized(reason="Invalid authorization scheme")

            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])

            # Fail closed when the token is past exp
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, UTC) < datetime.now(UTC):
                raise web.HTTPUnauthorized(reason="Token expired")

            return payload
        except jwt.InvalidTokenError:
            raise web.HTTPUnauthorized(reason="Invalid token") from None
        except Exception as e:
            raise web.HTTPUnauthorized(reason=f"Authentication failed: {str(e)}") from e

    def _require_admin_role(self, payload: dict[str, Any]) -> None:
        """Require claims.role == admin for privileged v1 calls."""
        if payload.get("role") != "admin":
            raise web.HTTPForbidden(reason="Admin access required")


class V1Handler:
    """aiohttp adapter that binds AdminAPIV1 callables to routes."""

    def __init__(self, db: DBManager, node_service: NodeService) -> None:
        self.db = db
        self.node_service = node_service

    async def register_node(self, req: web.Request) -> web.Response:
        """v1 node registration entrypoint."""
        try:
            body = await req.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid request format"}, status=400)

        valid, error, _validated = validate_node_register(body)
        if not valid:
            return web.json_response({"error": error}, status=400)

        if registration_requires_poc():
            return web.json_response(
                {
                    "error": "registration_requires_poc",
                    "message": "Use POST /v1/nodes/challenge then register with challenge and nonce",
                },
                status=403,
            )

        if os.getenv("DISTRIBAI_ALLOW_INSECURE_REGISTER") != "1":
            return web.json_response(
                {
                    "error": "insecure registration disabled",
                    "hint": (
                        "Use POST /v1/nodes/challenge then "
                        "POST /v1/nodes/register-enhanced (PoW). "
                        "Set DISTRIBAI_ALLOW_INSECURE_REGISTER=1 to re-enable "
                        "this endpoint on trusted networks."
                    ),
                },
                status=403,
            )

        node_id = body["node_id"]
        hardware = body.get("hardware", {})

        jwt_token = self.node_service._issue_jwt(node_id)

        await asyncio.to_thread(
            self.db.create_node,
            node_id=node_id,
            jwt_token=jwt_token,
            hardware_json=json.dumps(hardware),
        )

        return web.json_response(
            {
                "ok": True,
                "node_id": node_id,
                "token": jwt_token,
                "jwt": jwt_token,
            }
        )

    async def create_job(self, req: web.Request) -> web.Response:
        """v1 job create using the shared schema validator."""
        self.node_service._authenticate_request(req)

        try:
            body = await req.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)

        if not isinstance(body, dict):
            return web.json_response({"error": "request body must be an object"}, status=400)
        body = dict(body)
        raw_hparams = body.get("hparams")
        if raw_hparams is None:
            raw_hparams = body.get("hyperparams")
        if raw_hparams is not None and not isinstance(raw_hparams, dict):
            return web.json_response({"error": "hparams must be an object"}, status=400)
        hyperparams = dict(raw_hparams or {})
        body["hparams"] = hyperparams
        body.pop("hyperparams", None)

        valid, error, validated = validate_job_create(body)
        if not valid:
            return web.json_response({"error": error}, status=400)
        validated_data = validated.model_dump() if hasattr(validated, "model_dump") else vars(validated)
        hyperparams = dict(validated_data.get("hparams") or hyperparams)
        if validated_data.get("architecture_config") is not None:
            hyperparams["architecture_config"] = validated_data["architecture_config"]

        job_id = await asyncio.to_thread(
            self.db.create_job,
            job_type=validated_data.get("job_type", "fine_tune"),
            base_model=validated_data.get("base_model", ""),
            dataset_ref=validated_data.get("dataset_ref", ""),
            hyperparams=hyperparams,
            total_steps=validated_data.get("steps", DEFAULT_STEPS_PER_TASK),
        )

        return web.json_response({"ok": True, "job_id": job_id})

    async def get_job(self, req: web.Request) -> web.Response:
        """v1 single-job fetch by id."""
        self.node_service._authenticate_request(req)
        job_id = req.match_info.get("job_id")

        if not job_id:
            return web.json_response({"error": "missing job_id"}, status=400)

        job = await asyncio.to_thread(self.db.get_job, job_id)
        if not job:
            return web.json_response({"error": "not found"}, status=404)

        return web.json_response(job)

    async def get_queue(self, req: web.Request) -> web.Response:
        """v1 queue snapshot for polling clients."""
        self.node_service._authenticate_request(req)
        jobs = await asyncio.to_thread(self.db.get_queued_tasks)
        return web.json_response({"queue": jobs})

    async def request_challenge(self, req: web.Request) -> web.Response:
        """v1: mint a PoC challenge for the caller."""
        try:
            body = await req.json() if req.can_read_body else {}
        except json.JSONDecodeError:
            body = {}

        node_id = body.get("node_id")
        challenge = self.node_service.poc_challenge.generate_challenge(node_id)
        if isinstance(challenge, dict):
            return web.json_response(challenge)
        return web.json_response(
            {
                "challenge_id": challenge.challenge,
                "challenge": challenge.challenge,
                "challenge_hex": challenge.challenge_hex,
                "difficulty": challenge.difficulty,
                "expires_at": challenge.expires_at,
                "expires_in": challenge.expires_in,
                "algorithm": challenge.algorithm,
                "max_attempts": challenge.max_attempts,
            }
        )

    async def verify_challenge(self, req: web.Request) -> web.Response:
        """v1: verify a submitted PoC solution."""
        try:
            body = await req.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)

        node_id = body.get("node_id")
        challenge_str = body.get("challenge") or body.get("challenge_id")
        nonce = body.get("nonce")

        if not all([node_id, challenge_str, nonce]):
            return web.json_response({"error": "missing parameters"}, status=400)

        verified = self.node_service.poc_challenge.verify_challenge(node_id, challenge_str, nonce)
        if not verified:
            return web.json_response({"error": "verification failed"}, status=400)

        return web.json_response({"ok": True, "verified": True})

    async def register_node_enhanced(self, req: web.Request) -> web.Response:
        """v1: finalize registration once PoC succeeds."""
        try:
            body = await req.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)

        node_id = body.get("node_id")
        challenge = body.get("challenge") or body.get("challenge_id")
        nonce = body.get("nonce")
        hardware = body.get("hardware", {})
        if not hardware:
            hardware = {
                "os": body.get("os", "unknown"),
                "gpu_model": body.get("gpu_model", "unknown"),
                "vram_mb": body.get("vram_mb", 0),
                "cpu_cores": body.get("cpu_cores"),
                "ram_gb": body.get("ram_gb"),
            }

        if not all([node_id, challenge, nonce]):
            return web.json_response({"error": "missing parameters"}, status=400)

        verified = self.node_service.poc_challenge.verify_challenge(node_id, challenge, nonce)
        if not verified:
            return web.json_response({"error": "challenge verification failed"}, status=403)

        sybil_check = self.node_service.sybil_detector.analyze_account(
            node_id=node_id,
            ip_address=req.remote or "unknown",
            hardware_fingerprint=hardware.get("fingerprint", ""),
            initial_credits=0,
        )
        if os.getenv("RATE_LIMIT_DISABLED") == "1" and (req.remote or "") in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            sybil_check["approved"] = True
            sybil_check["reason"] = "Local test harness registration accepted"

        if not sybil_check["approved"]:
            return web.json_response(
                {"error": "sybil check failed", "reason": sybil_check["reason"]}, status=403
            )

        jwt_token = self.node_service._issue_jwt(node_id)
        await asyncio.to_thread(
            self.db.create_node,
            node_id=node_id,
            jwt_token=jwt_token,
            hardware_json=json.dumps(hardware),
        )

        response = {"ok": True, "node_id": node_id, "token": jwt_token, "jwt": jwt_token}

        if sybil_check.get("alerts"):
            response["sybil_alerts"] = sybil_check["alerts"]

        return web.json_response(response)

    async def get_benchmark_status(self, req: web.Request) -> web.Response:
        """v1: benchmark progress for the authenticated node."""
        claims = self.node_service._authenticate_request(req, required_kind="node")
        node_id = claims["sub"]

        status = self.node_service.rebenchmark.get_benchmark_status(node_id)

        if status["needs_rebenchmark"] and not status["is_pending"]:
            self.node_service.rebenchmark.schedule_rebenchmark(node_id)

        return web.json_response(status)

    async def submit_benchmark(self, req: web.Request) -> web.Response:
        """v1: ingest benchmark scores posted by a node."""
        claims = self.node_service._authenticate_request(req, required_kind="node")
        node_id = claims["sub"]

        try:
            body = await req.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "benchmark payload must be an object"}, status=400)
        if "overall_score" not in body:
            return web.json_response({"error": "overall_score is required"}, status=400)

        raw_score = body["overall_score"]
        if isinstance(raw_score, bool):
            return web.json_response({"error": "overall_score must be numeric"}, status=400)
        try:
            overall_score = float(raw_score)
        except (TypeError, ValueError):
            return web.json_response({"error": "overall_score must be numeric"}, status=400)
        if not math.isfinite(overall_score) or not 0 <= overall_score <= 100:
            return web.json_response(
                {"error": "overall_score must be finite and between 0 and 100"},
                status=400,
            )
        try:
            benchmark_json = json.dumps(body, allow_nan=False)
        except (TypeError, ValueError):
            return web.json_response(
                {"error": "benchmark payload must contain only JSON-finite values"},
                status=400,
            )
        driver_version = str(body.get("driver_version", ""))[:128]

        await asyncio.to_thread(self.db.update_node_benchmark, node_id, benchmark_json)

        self.node_service.rebenchmark.record_benchmark(
            node_id=node_id,
            benchmark_json=benchmark_json,
            driver_version=driver_version,
            compute_score=overall_score,
        )

        return web.json_response(
            {
                "node_id": node_id,
                "overall_score": overall_score,
                "recorded": True,
            }
        )

    async def get_rebenchmark_stats(self, req: web.Request) -> web.Response:
        """v1: rebenchmark cadence and trigger stats."""
        stats = self.node_service.rebenchmark.get_stats()
        pending = self.node_service.rebenchmark.get_all_pending()
        return web.json_response({**stats, "pending_nodes": pending})
