"""gRPC streaming service handlers for DistribAI orchestrator.

This module contains the gRPC streaming logic, separated from the admin API
to improve modularity and maintainability.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any

import grpc

from services_python.blob_loader import load_json_blob
from services_python.constants import MAX_TASK_CREDITS_REPORTED
from services_python.database import validate_node_id as validate_registered_node_id
from services_python.db_manager import DBManager
from services_python.registration_policy import registration_requires_poc
from worker.src.distribai_proto import distribai_pb2, distribai_pb2_grpc

if TYPE_CHECKING:
    import torch

    from services_python.orchestrator_grpc import NodeService

logger = logging.getLogger(__name__)


class GrpcServiceHandler(distribai_pb2_grpc.NodeServiceServicer):
    """Handler for gRPC streaming requests from worker nodes."""

    def __init__(self, node_service: NodeService) -> None:
        self.node_service = node_service
        self.db: DBManager = node_service.db

    async def StreamSession(self, request_iterator, context):
        """Handle bidirectional streaming with worker nodes."""
        node_id_ref = {"id": None}
        out_queue: asyncio.Queue = asyncio.Queue()

        async def incoming_handler() -> None:
            """Process incoming messages from the worker."""
            try:
                async for msg in request_iterator:
                    if msg.HasField("register"):
                        await self._handle_register(msg.register, out_queue, node_id_ref)
                    elif msg.HasField("heartbeat"):
                        await self._handle_heartbeat(msg.heartbeat, node_id_ref["id"])
                    elif msg.HasField("result"):
                        await self._handle_result(msg.result, node_id_ref["id"])
                    elif msg.HasField("progress"):
                        await self._handle_progress(msg.progress, node_id_ref["id"])
                    elif msg.HasField("grpo_reward_report"):
                        await self._handle_grpo_reward_report(msg.grpo_reward_report, node_id_ref["id"])
            except grpc.aio.AioRpcError:
                logger.debug(
                    "Stream incoming ended with gRPC client disconnect (node=%s)",
                    node_id_ref["id"],
                )

        async def outgoing_handler() -> None:
            """Send outgoing messages to the worker."""
            try:
                while True:
                    msg = await out_queue.get()
                    if msg is None:
                        break
                    yield msg
            except asyncio.CancelledError:
                logger.debug("Stream outgoing cancelled for node=%s", node_id_ref["id"])
                raise

        handler_task = asyncio.create_task(incoming_handler())
        try:
            async for msg in outgoing_handler():
                yield msg
        finally:
            handler_task.cancel()
            if node_id_ref["id"]:
                self.node_service.connected_nodes.pop(node_id_ref["id"], None)

    async def _handle_register(
        self,
        register: distribai_pb2.RegisterSession,
        out_queue: asyncio.Queue,
        node_id_ref: dict,
    ) -> None:
        """Handle node registration with JWT validation and optional PoC."""
        node_id = register.node_id

        try:
            validate_registered_node_id(node_id)
        except ValueError:
            logger.warning(
                "gRPC register rejected: invalid node_id format (len=%s)",
                len(node_id) if node_id else 0,
            )
            await out_queue.put(
                distribai_pb2.ServerMessage(
                    register_ack=distribai_pb2.RegisterAck(
                        session_token="",
                        server_version=getattr(self.node_service, "version", "dev"),
                    )
                )
            )
            return

        if registration_requires_poc():
            session_token: str | None = None
            if register.jwt_token:
                claims = self.node_service.verify_jwt(
                    register.jwt_token,
                    expected_subject=node_id,
                    kind="node",
                )
                if claims:
                    session_token = register.jwt_token
            elif register.challenge_id and register.nonce:
                verified = self.node_service.poc_challenge.verify_challenge(
                    node_id, register.challenge_id, register.nonce
                )
                if not verified:
                    logger.warning(
                        "gRPC register rejected: invalid PoC (node_id=%s)",
                        node_id,
                    )
                    await out_queue.put(
                        distribai_pb2.ServerMessage(
                            register_ack=distribai_pb2.RegisterAck(
                                session_token="",
                                server_version=getattr(
                                    self.node_service, "version", "dev"
                                ),
                            )
                        )
                    )
                    return
            else:
                logger.warning(
                    "gRPC register rejected: registration_requires_poc (node_id=%s)",
                    node_id,
                )
                await out_queue.put(
                    distribai_pb2.ServerMessage(
                        register_ack=distribai_pb2.RegisterAck(
                            session_token="",
                            server_version=getattr(self.node_service, "version", "dev"),
                        )
                    )
                )
                return

            node_id_ref["id"] = node_id
            self.node_service.connected_nodes[node_id] = out_queue

            try:
                hardware = json.loads(register.hardware_json) if register.hardware_json else {}
            except json.JSONDecodeError:
                hardware = {}

            jwt_token = session_token or self.node_service._issue_jwt(node_id)
            await asyncio.to_thread(
                self.db.register_node,
                node_id,
                jwt_token,
                json.dumps(hardware),
                int(time.time()),
                register.benchmark_json or None,
                jwt_token,
            )

            ack = distribai_pb2.ServerMessage(
                register_ack=distribai_pb2.RegisterAck(
                    session_token=jwt_token,
                    server_version=getattr(self.node_service, "version", "dev"),
                )
            )
            await out_queue.put(ack)
            return

        presented = (register.jwt_token or "").strip()
        existing_jwt = await asyncio.to_thread(self.db.get_node_jwt, node_id)
        allow_bootstrap = os.getenv("DISTRIBAI_GRPC_ALLOW_BOOTSTRAP", "0") == "1"

        if presented:
            claims = self.node_service.verify_jwt(
                presented, expected_subject=node_id, kind="node"
            )
            if not claims:
                logger.warning("gRPC register rejected: invalid JWT for node_id=%s", node_id)
                return
        elif existing_jwt:
            logger.warning(
                "gRPC register rejected: node_id=%s already registered "
                "but no JWT presented in RegisterSession",
                node_id,
            )
            return
        elif not allow_bootstrap:
            logger.warning(
                "gRPC register rejected: bootstrap registration via gRPC is "
                "disabled (node_id=%s). Set DISTRIBAI_GRPC_ALLOW_BOOTSTRAP=1 on trusted networks.",
                node_id,
            )
            return

        node_id_ref["id"] = node_id
        self.node_service.connected_nodes[node_id] = out_queue

        try:
            hardware = json.loads(register.hardware_json) if register.hardware_json else {}
        except json.JSONDecodeError:
            hardware = {}

        jwt_token = self.node_service._issue_jwt(node_id)

        await asyncio.to_thread(
            self.db.create_node,
            node_id,
            jwt_token,
            json.dumps(hardware),
        )
        await asyncio.to_thread(
            self.db.update_node_hardware,
            node_id,
            json.dumps(hardware),
            register.benchmark_json or None,
        )
        await asyncio.to_thread(self.db.update_node_jwt, node_id, jwt_token)

        ack = distribai_pb2.ServerMessage(
            register_ack=distribai_pb2.RegisterAck(
                session_token=jwt_token,
                server_version=getattr(self.node_service, "version", "dev"),
            )
        )
        await out_queue.put(ack)

    async def _handle_heartbeat(
        self,
        heartbeat: distribai_pb2.Heartbeat,
        node_id: str | None,
    ) -> None:
        """Handle node heartbeat."""
        if not node_id:
            return

        await asyncio.to_thread(
            self.db.update_heartbeat,
            node_id,
            heartbeat.seq,
            heartbeat.gpu_util,
            heartbeat.vram_free_mb,
            heartbeat.task_id if heartbeat.HasField("task_id") else None,
        )

    async def _handle_result(
        self,
        result: distribai_pb2.TaskResult,
        session_node_id: str | None,
    ) -> None:
        """Handle task result from worker."""
        if not session_node_id:
            logger.warning(
                "Ignoring task result before registration: task=%s node=%s",
                result.task_id,
                result.node_id,
            )
            return
        if result.node_id != session_node_id:
            logger.warning(
                "Ignoring task result node_id mismatch: session=%s wire=%s task=%s",
                session_node_id,
                result.node_id,
                result.task_id,
            )
            return

        self.node_service.log_lines.append(
            f"[{result.node_id}] Task {result.task_id} Result: {result.status}"
        )

        output = self.node_service._safe_json(result.output_json)
        gradient_blob_url = result.gradient_blob_url or output.get("gradient_blob_url", "")

        await asyncio.to_thread(
            self.db.update_task_result,
            result.task_id,
            result.node_id,
            result.status,
            result.output_json,
            gradient_blob_url,
        )

        if result.status == "success":
            raw_credits = float(output.get("credits_earned", 10))
            credits_earned = min(max(raw_credits, 0.0), MAX_TASK_CREDITS_REPORTED)
            if credits_earned > 0:
                await asyncio.to_thread(
                    self.node_service.record_credit_earn,
                    session_node_id,
                    credits_earned,
                    result.job_id,
                    {"task_id": result.task_id},
                )

        await self._check_and_aggregate(result.job_id)

        if self.node_service.pending_assignments.get(session_node_id) == result.task_id:
            self.node_service.pending_assignments.pop(session_node_id, None)

    async def _handle_progress(
        self,
        progress: distribai_pb2.TaskProgress,
        session_node_id: str | None,
    ) -> None:
        """Handle progress update from worker."""
        if not session_node_id:
            return
        if progress.node_id and progress.node_id != session_node_id:
            logger.warning(
                "Ignoring progress node_id mismatch: session=%s wire=%s task=%s",
                session_node_id,
                progress.node_id,
                progress.task_id,
            )
            return

        await asyncio.to_thread(
            self.db.update_task_progress,
            progress.task_id,
            progress.step,
            progress.loss,
            progress.ts,
        )

    async def _handle_grpo_reward_report(
        self,
        report: distribai_pb2.GrpoRewardReport,
        session_node_id: str | None,
    ) -> None:
        """Handle GRPO reward report from a worker.

        Validates the session, then forwards to the GrpoCoordinator.
        """
        if not session_node_id:
            logger.warning("Ignoring GRPO reward report before registration")
            return
        if report.worker_id != session_node_id:
            logger.warning(
                "Ignoring GRPO reward report worker_id mismatch: session=%s wire=%s",
                session_node_id,
                report.worker_id,
            )
            return

        coord = self.node_service.grpo_coordinator
        coord.handle_reward_report(
            job_id=report.job_id,
            worker_id=report.worker_id,
            round_id=report.round_id,
            candidate_rewards=list(report.candidate_rewards),
            candidate_texts_json=report.candidate_texts_json or None,
        )

        # If all workers have reported, finalise the round automatically
        if coord.all_workers_reported(report.job_id, report.round_id):
            logger.info(
                "[GrpcServiceHandler] All workers reported for job=%s round=%d, finalising",
                report.job_id,
                report.round_id,
            )
            # The caller (DistributedTrainer / JobDistributor) is responsible
            # for computing the new weights and calling finalise_round().
            # We emit a log line so the dashboard can alert operators.
            self.node_service.log_lines.append(
                f"[GRPO] job={report.job_id} round={report.round_id} "
                f"all {len(state.worker_ids) if (state := coord.get_job(report.job_id)) else '?'} "
                f"workers reported"
            )

    async def _check_and_aggregate(self, job_id: str) -> None:
        """Check if enough results exist for aggregation."""
        results = await asyncio.to_thread(self.db.get_job_results, job_id)
        if len(results) < 3:
            return

        pending = [r for r in results if r["status"] == "success" and r.get("gradient_blob_url")]
        if len(pending) < 3:
            return

        # Load gradients
        node_gradients: dict[str, dict] = {}
        for r in pending:
            payload = await self._load_gradient_payload(r["gradient_blob_url"])
            if payload:
                node_gradients[r["node_id"]] = payload

        if len(node_gradients) < 3:
            return

        # Detect Byzantine gradients
        byzantine_detected, selected_payload = await self._detect_byzantine_gradients(
            node_gradients
        )

        if selected_payload:
            await asyncio.to_thread(
                self.db.update_job_aggregate,
                job_id,
                selected_payload,
            )
            try:
                await self.node_service.broadcast_control(
                    "bft_aggregate_ready",
                    job_id,
                    list(node_gradients.keys()),
                )
            except (ConnectionError, RuntimeError):
                logger.exception("broadcast_control after aggregate failed for job %s", job_id)

    async def _detect_byzantine_gradients(
        self, node_gradients: dict[str, dict]
    ) -> tuple[bool, dict | None]:
        """Detect and filter Byzantine gradients."""

        pure_result = self._detect_byzantine_gradients_pure_python(node_gradients)
        if pure_result is not None:
            return pure_result

        try:
            gradient_tensors = {}
            tensor_specs: list[tuple[str, tuple[int, ...], int]] | None = None
            for node_id, payload in node_gradients.items():
                vector, spec = self._gradient_payload_to_tensor(payload)
                if vector.numel() == 0:
                    continue
                if tensor_specs is None:
                    tensor_specs = spec
                if spec == tensor_specs:
                    gradient_tensors[node_id] = vector

            if len(gradient_tensors) < 3:
                return False, list(node_gradients.values())[0] if node_gradients else None

            scores = self.node_service.byzantine_detector.detect_anomalies(gradient_tensors)
            byzantine_nodes = [s.node_id for s in scores if s.is_byzantine]

            clean_tensors = {
                nid: tensor
                for nid, tensor in gradient_tensors.items()
                if nid not in byzantine_nodes
            }

            if not clean_tensors:
                return True, list(node_gradients.values())[0]

            aggregate = self.node_service.byzantine_detector.aggregate(clean_tensors)
            return len(byzantine_nodes) > 0, {
                "method": "robust_bft_aggregate",
                "byzantine_nodes": byzantine_nodes,
                "source_nodes": list(clean_tensors.keys()),
                "weights": aggregate.detach().cpu().tolist(),
                "parameters": self._tensor_to_gradient_payload(aggregate, tensor_specs or []),
            }

        except (TypeError, ValueError, RuntimeError):
            return False, list(node_gradients.values())[0] if node_gradients else None

    def _detect_byzantine_gradients_pure_python(
        self,
        node_gradients: dict[str, dict],
    ) -> tuple[bool, dict | None] | None:
        """Fast robust aggregation for JSON-native gradient payloads."""
        if len(node_gradients) < 3:
            return False, list(node_gradients.values())[0] if node_gradients else None

        vectors: dict[str, list[float]] = {}
        specs: list[tuple[str, Any]] | None = None
        for node_id, payload in node_gradients.items():
            vector, spec = self._gradient_payload_to_vector(payload)
            if not vector:
                continue
            if specs is None:
                specs = spec
            if spec == specs:
                vectors[node_id] = vector

        if len(vectors) < 3 or specs is None:
            return None

        length = len(next(iter(vectors.values())))
        if any(len(vector) != length for vector in vectors.values()):
            return None

        means = [
            sum(vector[i] for vector in vectors.values()) / len(vectors) for i in range(length)
        ]
        distances = {
            node_id: sum((value - means[i]) ** 2 for i, value in enumerate(vector)) ** 0.5
            for node_id, vector in vectors.items()
        }
        if not distances:
            return None

        sorted_distances = sorted(distances.values())
        median_distance = sorted_distances[len(sorted_distances) // 2]
        threshold = max(median_distance * 3.0, 1e-6)
        byzantine_nodes = [
            node_id
            for node_id, distance in distances.items()
            if distance > threshold and len(vectors) > 3
        ]
        clean = {
            node_id: vector for node_id, vector in vectors.items() if node_id not in byzantine_nodes
        } or vectors
        aggregate = [
            sum(vector[i] for vector in clean.values()) / len(clean) for i in range(length)
        ]

        return bool(byzantine_nodes), {
            "method": "robust_bft_aggregate",
            "byzantine_nodes": byzantine_nodes,
            "source_nodes": list(clean.keys()),
            "weights": aggregate,
            "parameters": self._vector_to_gradient_payload(aggregate, specs),
        }

    def _gradient_payload_to_vector(
        self,
        payload: dict[str, Any],
    ) -> tuple[list[float], list[tuple[str, Any]]]:
        if payload.get("compression") == "dgc" and isinstance(payload.get("raw_fallback"), dict):
            payload = payload["raw_fallback"]

        vector: list[float] = []
        spec: list[tuple[str, Any]] = []
        for name in sorted(payload):
            value = payload[name]
            flattened, shape = self._flatten_numeric(value)
            if flattened:
                vector.extend(flattened)
                spec.append((name, shape))
        return vector, spec

    def _flatten_numeric(self, value: Any) -> tuple[list[float], Any]:
        if isinstance(value, bool):
            return [], None
        if isinstance(value, (int, float)):
            return [float(value)], None
        if isinstance(value, list):
            flattened: list[float] = []
            shapes = []
            for item in value:
                part, shape = self._flatten_numeric(item)
                flattened.extend(part)
                shapes.append(shape)
            return flattened, shapes
        return [], None

    def _vector_to_gradient_payload(
        self,
        vector: list[float],
        spec: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        offset = 0
        payload: dict[str, Any] = {}
        for name, shape in spec:
            size = self._shape_size(shape)
            payload[name] = self._unflatten_numeric(vector[offset : offset + size], shape)
            offset += size
        return payload

    def _shape_size(self, shape: Any) -> int:
        if shape is None:
            return 1
        if isinstance(shape, list):
            return sum(self._shape_size(item) for item in shape)
        return 0

    def _unflatten_numeric(self, values: list[float], shape: Any) -> Any:
        if shape is None:
            return values[0] if values else 0.0
        result = []
        offset = 0
        for child_shape in shape:
            size = self._shape_size(child_shape)
            result.append(self._unflatten_numeric(values[offset : offset + size], child_shape))
            offset += size
        return result

    def _gradient_payload_to_tensor(
        self, payload: dict[str, Any]
    ) -> tuple[torch.Tensor, list[tuple[str, tuple[int, ...], int]]]:
        import torch

        if payload.get("compression") == "dgc" and isinstance(payload.get("compressed"), dict):
            return self._compressed_gradient_payload_to_tensor(payload["compressed"])
        if isinstance(payload.get("raw_fallback"), dict):
            payload = payload["raw_fallback"]

        if isinstance(payload.get("weights"), list):
            tensor = torch.tensor(payload["weights"], dtype=torch.float32).flatten()
            return tensor, [("weights", tuple(tensor.shape), int(tensor.numel()))]

        chunks = []
        spec: list[tuple[str, tuple[int, ...], int]] = []
        for name in sorted(payload):
            value = payload[name]
            if not isinstance(value, (int, float, list)):
                continue
            tensor = torch.tensor(value, dtype=torch.float32)
            if tensor.numel() == 0:
                continue
            chunks.append(tensor.flatten())
            spec.append((name, tuple(tensor.shape), int(tensor.numel())))
        if not chunks:
            return torch.tensor([], dtype=torch.float32), []
        return torch.cat(chunks), spec

    def _compressed_gradient_payload_to_tensor(
        self, compressed: dict[str, Any]
    ) -> tuple[torch.Tensor, list[tuple[str, tuple[int, ...], int]]]:
        import torch

        chunks = []
        spec: list[tuple[str, tuple[int, ...], int]] = []
        for name in sorted(compressed):
            data = compressed[name]
            if not isinstance(data, dict) or data.get("method") != "dgc":
                continue
            shape = tuple(int(part) for part in data.get("shape", []))
            size = 1
            for dim in shape:
                size *= dim
            if size <= 0:
                continue
            tensor = torch.zeros(size, dtype=torch.float32)
            indices = data.get("indices", [])
            values = data.get("values", [])
            if len(indices) != len(values):
                continue
            if indices:
                tensor[torch.tensor(indices, dtype=torch.long)] = torch.tensor(
                    values, dtype=torch.float32
                )
            chunks.append(tensor)
            spec.append((name, shape, size))
        if not chunks:
            return torch.tensor([], dtype=torch.float32), []
        return torch.cat(chunks), spec

    def _tensor_to_gradient_payload(
        self, tensor: torch.Tensor, spec: list[tuple[str, tuple[int, ...], int]]
    ) -> dict[str, Any]:
        offset = 0
        payload: dict[str, Any] = {}
        for name, shape, size in spec:
            chunk = tensor[offset : offset + size]
            payload[name] = chunk.reshape(shape).detach().cpu().tolist()
            offset += size
        return payload

    async def _load_gradient_payload(self, gradient_blob_url: str) -> dict[str, Any] | None:
        """Load gradient payload from URL."""
        s3_client = getattr(self.node_service, "s3_client", None)
        return await load_json_blob(gradient_blob_url, s3_client=s3_client)

    def _safe_json(self, payload: str | dict[str, Any] | None) -> dict[str, Any]:
        """Safely parse JSON payload."""
        if payload in (None, ""):
            return {}
        if isinstance(payload, dict):
            return payload
        try:
            return json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return {}
