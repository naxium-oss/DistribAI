#!/usr/bin/env python3
"""
DistribAI Server GUI Launcher - Production Version

Provides a desktop GUI for the orchestrator server using PyWebView.
"""

import argparse
import asyncio
import json
import os
import secrets
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import webview

sys.path.insert(0, str(Path(__file__).parent))

from datetime import UTC

from services_python.constants import (
    DEFAULT_ADMIN_HOST,
    DEFAULT_ADMIN_PORT,
    DEFAULT_GRPC_PORT,
    S3_DEFAULT_REGION,
)

try:
    from services_python.job_submission import (
        JobSubmissionHandler,
        create_distributor,
        job_queue,
        validate_script_file,
    )

    JOB_SUBMISSION_AVAILABLE = True
except ImportError:
    JOB_SUBMISSION_AVAILABLE = False


class ServerAPI:
    """JavaScript API bridge for the Server GUI."""

    def __init__(self):
        self.window: webview.Window | None = None
        self.server_running: bool = False
        self.runtime: Any = None
        self.env_path = Path(__file__).parent.parent / ".env"
        self.start_time: float = 0.0
        self._server_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def set_window(self, window: webview.Window):
        self.window = window
        self._ensure_env_file()
        self._start_monitoring()

    def _start_monitoring(self):
        def monitor_loop():
            while not self._stop_event.is_set():
                if self.window and self.server_running:
                    try:
                        status = self._get_full_status()
                        self._emit_event("status_update", status)
                    except (RuntimeError, ValueError):
                        pass
                time.sleep(5)

        threading.Thread(target=monitor_loop, daemon=True).start()

    def _emit_event(self, event_type: str, data: dict):
        if self.window:
            try:
                js_code = f"if(window.__distribai_events){{window.__distribai_events.emit('{event_type}', {json.dumps(data)})}}"
                self.window.evaluate_js(js_code)
            except (RuntimeError, ValueError):
                pass

    def _get_full_status(self) -> dict:
        connected_nodes = 0
        active_jobs = 0

        if self.runtime and hasattr(self.runtime, "node_service"):
            node_service = self.runtime.node_service
            connected_nodes = len(node_service.connected_nodes)
            active_jobs = len(node_service.pending_assignments)

        uptime = time.time() - self.start_time if self.start_time > 0 else 0

        return {
            "running": self.server_running,
            "grpc_port": self._get_setting("GRPC_PORT", DEFAULT_GRPC_PORT),
            "admin_port": self._get_setting("ADMIN_PORT", DEFAULT_ADMIN_PORT),
            "admin_host": self._get_setting("ADMIN_HOST", DEFAULT_ADMIN_HOST),
            "uptime_seconds": int(uptime),
            "connected_nodes": connected_nodes,
            "active_jobs": active_jobs,
        }

    def _ensure_env_file(self):
        if not self.env_path.exists():
            default_env = f"""# DistribAI Server Configuration

GRPC_PORT={DEFAULT_GRPC_PORT}
ADMIN_PORT={DEFAULT_ADMIN_PORT}
ADMIN_HOST={DEFAULT_ADMIN_HOST}

JWT_SECRET={secrets.token_urlsafe(32)}
SIGNING_KEY={secrets.token_urlsafe(32)}

AWS_REGION={S3_DEFAULT_REGION}
CORS_ALLOWED_ORIGINS=*
GITHUB_UPDATE_URL=https://github.com/your-org/distribai-releases/releases/latest/download
"""
            self.env_path.write_text(default_env)

    def _get_setting(self, key: str, default: str = "") -> str:
        settings = self.get_settings()
        return settings.get(key, default)

    def get_server_status(self) -> dict:
        return self._get_full_status()

    def start_server(self) -> dict:
        if self.server_running:
            return {"success": False, "message": "Server already running"}

        try:
            self._load_env_to_os()

            def run_server():
                try:
                    from services_python.orchestrator_grpc import serve

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    self.runtime = loop.run_until_complete(serve(block=False))

                    while not self._stop_event.is_set():
                        loop.run_until_complete(asyncio.sleep(1))

                    if self.runtime:
                        loop.run_until_complete(self.runtime.stop())

                except Exception as e:
                    print(f"[ServerGUI] Server error: {e}")
                    self.server_running = False

            self._server_thread = threading.Thread(target=run_server, daemon=True)
            self._server_thread.start()

            time.sleep(2)

            self.server_running = True
            self.start_time = time.time()

            return {"success": True, "message": "Server started"}

        except Exception as e:
            return {"success": False, "message": str(e)}

    def stop_server(self) -> dict:
        if not self.server_running:
            return {"success": False, "message": "Server not running"}

        try:
            self._stop_event.set()
            self.server_running = False

            if self.runtime and hasattr(self.runtime, "stop"):
                asyncio = __import__("asyncio")
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.runtime.stop())

            return {"success": True, "message": "Server stopped"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def restart_server(self) -> dict:
        self.stop_server()
        time.sleep(2)
        self._stop_event.clear()
        return self.start_server()

    def _load_env_to_os(self):
        if self.env_path.exists():
            with open(self.env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key] = value

    def get_settings(self) -> dict:
        settings = {}

        if self.env_path.exists():
            with open(self.env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        settings[key] = value

        defaults = {
            "GRPC_PORT": DEFAULT_GRPC_PORT,
            "ADMIN_PORT": DEFAULT_ADMIN_PORT,
            "ADMIN_HOST": DEFAULT_ADMIN_HOST,
            "AWS_REGION": S3_DEFAULT_REGION,
            "CORS_ALLOWED_ORIGINS": "*",
        }

        for key, default in defaults.items():
            if key not in settings:
                settings[key] = default

        return settings

    def save_settings(self, settings: dict) -> dict:
        try:
            lines = []
            if self.env_path.exists():
                lines = self.env_path.read_text().split("\n")

            new_lines = []
            updated_keys = set()

            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key = stripped.split("=", 1)[0]
                    if key in settings:
                        new_lines.append(f"{key}={settings[key]}")
                        updated_keys.add(key)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)

            for key, value in settings.items():
                if key not in updated_keys:
                    new_lines.append(f"{key}={value}")

            self.env_path.write_text("\n".join(new_lines))

            return {"success": True, "message": "Settings saved"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def generate_secrets(self) -> dict:
        return {
            "jwt_secret": secrets.token_urlsafe(32),
            "signing_key": secrets.token_urlsafe(32),
        }

    def test_s3_connection(self, settings: dict) -> dict:
        try:
            import boto3

            s3 = boto3.client(
                "s3",
                aws_access_key_id=settings.get("AWS_ACCESS_KEY_ID", ""),
                aws_secret_access_key=settings.get("AWS_SECRET_ACCESS_KEY", ""),
                region_name=settings.get("AWS_REGION", S3_DEFAULT_REGION),
            )

            response = s3.list_buckets()

            bucket_name = settings.get("S3_BUCKET_NAME", "")
            if bucket_name:
                bucket_names = [b["Name"] for b in response.get("Buckets", [])]
                if bucket_name in bucket_names:
                    return {
                        "success": True,
                        "message": f"Connected to S3, bucket '{bucket_name}' found",
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Connected to S3, but bucket '{bucket_name}' not found",
                    }

            return {"success": True, "message": "Connected to S3"}

        except Exception as e:
            return {"success": False, "message": str(e)}

    def test_postgres_connection(self, connection_string: str) -> dict:
        try:
            if not connection_string.startswith("postgresql://"):
                return {"success": False, "message": "Invalid connection string format"}

            return {
                "success": True,
                "message": "Connection string format valid (will be tested on server start)",
            }
        except Exception as e:
            return {"success": False, "message": str(e)}

    def get_nodes(self) -> list:
        if not self.runtime or not hasattr(self.runtime, "node_service"):
            return []

        try:
            node_service = self.runtime.node_service
            nodes = []

            for node_id in node_service.connected_nodes:
                nodes.append(
                    {
                        "id": node_id,
                        "status": "online",
                        "gpu": "Unknown",
                        "credits": 0.0,
                    }
                )

            return nodes
        except (AttributeError, TypeError):
            return []

    def get_jobs(self) -> list:
        if not self.runtime or not hasattr(self.runtime, "node_service"):
            return []

        try:
            node_service = self.runtime.node_service
            jobs = []

            for job_id, _assignment in node_service.pending_assignments.items():
                jobs.append(
                    {
                        "id": job_id,
                        "name": f"Job {job_id[:8]}",
                        "status": "running",
                        "priority": "P1",
                    }
                )

            return jobs
        except (AttributeError, TypeError):
            return []

    def get_ledger_summary(self) -> dict:
        if not self.runtime or not hasattr(self.runtime, "node_service"):
            return {"total_records": 0, "total_credits": 0.0, "last_batch": None}

        try:
            ledger = self.runtime.node_service.credit_ledger
            return {
                "total_records": ledger.size(),
                "total_credits": sum(
                    r.data.get("amount", 0) for r in ledger.records if hasattr(r, "data")
                ),
                "last_batch": ledger.get_root_hash().hex()[:16] if ledger.get_root_hash() else None,
            }
        except (AttributeError, TypeError):
            return {"total_records": 0, "total_credits": 0.0, "last_batch": None}

    def check_for_updates(self) -> dict:
        try:
            settings = self.get_settings()
            update_url = settings.get("GITHUB_UPDATE_URL", "")
            if not update_url:
                return {
                    "update_available": False,
                    "current_version": "1.0.0",
                    "latest_version": "1.0.0",
                }

            version_url = f"{update_url}/version.json"

            req = urllib.request.Request(version_url, headers={"User-Agent": "DistribAI-Server/1.0"})
            with urllib.request.urlopen(req, timeout=10) as response:
                remote_info = json.loads(response.read())

            current = "1.0.0"
            latest = remote_info.get("version", "1.0.0")

            update_available = self._version_compare(latest, current) > 0

            return {
                "update_available": update_available,
                "current_version": current,
                "latest_version": latest,
                "download_url": remote_info.get("download_url", ""),
                "download_size_mb": remote_info.get("size_mb", 0),
                "release_notes": remote_info.get("notes", ""),
            }
        except Exception as e:
            return {
                "update_available": False,
                "current_version": "1.0.0",
                "latest_version": "1.0.0",
                "error": str(e),
            }

    def _version_compare(self, v1: str, v2: str) -> int:
        parts1 = [int(x) for x in v1.split(".")]
        parts2 = [int(x) for x in v2.split(".")]

        for i in range(max(len(parts1), len(parts2))):
            p1 = parts1[i] if i < len(parts1) else 0
            p2 = parts2[i] if i < len(parts2) else 0
            if p1 > p2:
                return 1
            elif p1 < p2:
                return -1
        return 0

    def get_update_hosting_url(self) -> str:
        settings = self.get_settings()
        return settings.get("GITHUB_UPDATE_URL", "")

    # =========================================================================
    # Job Queue API
    # =========================================================================

    def get_queue_status(self) -> dict:
        """Get job queue status."""
        if JOB_SUBMISSION_AVAILABLE:
            return job_queue.get_status()
        return {"pending": 0, "running": 0, "completed": 0, "pending_jobs": []}

    def get_job_submissions(self) -> list:
        """Get recent job submissions."""
        if not JOB_SUBMISSION_AVAILABLE:
            return []

        jobs = []
        # Pending jobs
        for job in list(job_queue._pending):
            jobs.append(
                {
                    "job_id": job.job_id,
                    "name": job.name,
                    "org_id": job.org_id,
                    "type": job.job_type.value,
                    "priority": job.priority.name,
                    "status": job.status,
                    "created_at": job.created_at.isoformat(),
                }
            )

        # Running jobs
        for job in list(job_queue._running.values()):
            jobs.append(
                {
                    "job_id": job.job_id,
                    "name": job.name,
                    "org_id": job.org_id,
                    "type": job.job_type.value,
                    "priority": job.priority.name,
                    "status": job.status,
                    "created_at": job.created_at.isoformat(),
                }
            )

        return jobs[:50]  # Return top 50

    # =========================================================================
    # Job Creation & AI Assistant API
    # =========================================================================

    def get_dataset_formats(self) -> list[dict]:
        """Get available dataset formats."""
        return [
            {"id": "alpaca", "name": "Alpaca", "description": "Instruction-response pairs"},
            {"id": "sharegpt", "name": "ShareGPT", "description": "Conversations with GPT"},
            {"id": "dolly", "name": "Dolly", "description": "Databricks Dolly format"},
            {"id": "oasst1", "name": "OpenAssistant", "description": "OpenAssistant conversations"},
            {"id": "lima", "name": "LIMA", "description": "Less Is More for Alignment"},
            {"id": "gpt4all", "name": "GPT4All", "description": "Nomic GPT4All format"},
            {"id": "llama2", "name": "Llama 2 Chat", "description": "Meta Llama 2 format"},
            {"id": "chatml", "name": "ChatML", "description": "OpenAI ChatML format"},
            {"id": "vicuna", "name": "Vicuna", "description": "Vicuna conversation format"},
            {"id": "koala", "name": "Koala", "description": "Koala training format"},
            {"id": "openorca", "name": "OpenOrca", "description": "OpenOrca FLAN format"},
            {"id": "camelai", "name": "CamelAI", "description": "CamelAI role-playing"},
            {"id": "garbage", "name": "Garbage (test)", "description": "Test dataset"},
            {"id": "auto", "name": "Auto-detect", "description": "Automatically detect format"},
        ]

    def get_training_phases(self) -> list[dict]:
        """Get available training phases for DistribAI."""
        return [
            {"id": "pretrain", "name": "Pre-training", "description": "Train from scratch"},
            {"id": "sft", "name": "Supervised Fine-tuning", "description": "Instruction tuning"},
            {"id": "rl", "name": "Reinforcement Learning", "description": "RLHF / PPO"},
            {"id": "distill", "name": "Distillation", "description": "Knowledge distillation"},
            {"id": "spin", "name": "SPIN", "description": "Self-play fine-tuning"},
            {"id": "dpo", "name": "DPO", "description": "Direct Preference Optimization"},
        ]

    def get_distribai_models(self) -> list[dict]:
        """Return the same native model registry used by worker executors."""
        from worker.src.compute.distribai_models import DistribAIModelWrapper

        descriptions = {
            "distribai-tiny": "Fast development and CPU-friendly profile",
            "distribai-small": "Efficient general-purpose profile",
            "distribai-base": "Balanced production profile",
            "distribai-medium": "Higher-capacity production profile",
            "distribai-large": "Large-capacity profile for capable nodes",
            "distribai-xl": "Extra-large profile for high-memory nodes",
            "distribai-lstm-small": "Multi-layer LSTM language model profile",
            "distribai-resnet-tiny": "Residual 1D convolution LM profile",
            "distribai-hybrid-small": "Alternating attention + GRU hybrid profile",
            "distribai-dense-tiny": "Deep residual FFN tower (no attention)",
        }
        return [
            {
                "id": model_id,
                "name": model_id.replace("distribai-", "DistribAI ").title(),
                "description": descriptions.get(model_id, "Native DistribAI model profile"),
                "architecture": config.get("architecture", "decoder_transformer"),
                "config": dict(config),
            }
            for model_id, config in sorted(DistribAIModelWrapper.MODEL_CONFIGS.items())
        ]

    def check_dependencies(self, requirements: list[str]) -> dict:
        """Check dependencies for safety issues."""
        try:
            from dependency_checker import check_requirements_list

            return check_requirements_list(requirements)
        except Exception as e:
            return {
                "error": str(e),
                "can_proceed": True,  # Fail open if checker fails
            }

    def generate_ai_prompt(self, params: dict) -> dict:
        """Generate AI assistant prompt for script creation.

        Args:
            params: Dict with keys like job_type, training_phase, model, etc.

        Returns:
            Dict with prompt text and copy button content
        """
        prompts = {
            "claude_code": self._generate_claude_prompt(params),
            "cursor": self._generate_cursor_prompt(params),
            "naxi": self._generate_naxi_prompt(params),
        }

        return {
            "prompts": prompts,
            "recommended": "claude_code",
        }

    def _generate_claude_prompt(self, params: dict) -> str:
        """Generate prompt for Claude Code."""
        return f"""I need a DistribAI training script for {params.get("model", "small")} model.

JOB DETAILS:
- Training Phase: {params.get("training_phase", "sft")}
- Dataset Format: {params.get("dataset_format", "alpaca")}
- Dataset: {params.get("dataset_ref", "your_dataset.jsonl")}
- Total Steps: {params.get("total_steps", 1000)}
- Hyperparameters: LR={params.get("lr", 5e-5)}, Batch Size={params.get("batch_size", 8)}

REQUIREMENTS:
1. Use DistribAI environment variables for distributed training:
   - RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
   - DISTRIBAI_JOB_ID, DISTRIBAI_NODE_ID

2. Save checkpoints to: checkpoint_rank_{{RANK}}.pt

3. Log progress every 100 steps

4. Handle graceful shutdown on cancel signal

5. Load config from config.json

Please generate a complete run.py script."""

    def _generate_cursor_prompt(self, params: dict) -> str:
        """Generate prompt for Cursor."""
        return f"""Create a DistribAI distributed training script:

Model: {params.get("model", "small")}
Phase: {params.get("training_phase", "sft")}
Dataset: {params.get("dataset_format", "alpaca")} format from {params.get("dataset_ref", "dataset.jsonl")}
Steps: {params.get("total_steps", 1000)}

Must include:
- Distributed setup with torch.distributed
- Checkpoint saving (checkpoint_rank_X.pt)
- Progress logging
- Config loading from JSON
- Environment variable handling (RANK, WORLD_SIZE, etc.)

Output: Complete run.py script."""

    def _generate_naxi_prompt(self, params: dict) -> str:
        """Generate prompt for Naxi."""
        return f"""DistribAI training script for {params.get("training_phase", "sft")}:

Model: {params.get("model", "small")}
Dataset: {params.get("dataset_format", "alpaca")} | {params.get("dataset_ref", "data.jsonl")}
Steps: {params.get("total_steps", 1000)}

Use env vars: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, DISTRIBAI_JOB_ID
Save checkpoints: checkpoint_rank_{{RANK}}.pt
Load config: config.json

Generate run.py."""

    def create_job(self, job_data: dict) -> dict:
        """Create a new job via GUI.

        Args:
            job_data: Dict with job configuration

        Returns:
            Dict with success status and job_id or error
        """
        if not JOB_SUBMISSION_AVAILABLE:
            return {"success": False, "error": "Job submission not available"}

        try:
            script_content = job_data.get("script_content")
            script_path = job_data.get("script_path")
            has_script_content = isinstance(script_content, str) and bool(script_content.strip())
            has_script_path = isinstance(script_path, str) and bool(script_path.strip())
            if not has_script_content and not has_script_path:
                return {
                    "success": False,
                    "error": "script_content or script_path is required; generated scripts are not supported",
                }

            if has_script_content:
                from services_python.script_validation import validate_submitted_script

                err_codes, hints = validate_submitted_script(script_content)
                if err_codes:
                    return {
                        "success": False,
                        "error": "submitted script failed validation",
                        "validation_errors": err_codes,
                        "suggestions": hints,
                    }
            else:
                workspace = os.getenv("DISTRIBAI_SCRIPT_WORKSPACE", "").strip()
                if not workspace:
                    return {
                        "success": False,
                        "error": "script_path is disabled; provide script_content or configure DISTRIBAI_SCRIPT_WORKSPACE",
                    }
                try:
                    requested = Path(script_path).resolve()
                    allowed_root = Path(workspace).resolve()
                    requested.relative_to(allowed_root)
                except (OSError, ValueError):
                    return {
                        "success": False,
                        "error": "script_path must be inside DISTRIBAI_SCRIPT_WORKSPACE",
                    }
                path_errors, path_hints = validate_script_file(script_path)
                if path_errors:
                    return {
                        "success": False,
                        "error": "submitted script failed validation",
                        "validation_errors": path_errors,
                        "suggestions": path_hints,
                    }

            # Validate dependencies first
            requirements = job_data.get("requirements", [])
            dep_check = self.check_dependencies(requirements)

            if not dep_check.get("can_proceed", True):
                blockers = dep_check.get("blocked", [])
                return {
                    "success": False,
                    "error": f"Blocked dependencies: {[b.package_name for b in blockers]}",
                    "dependency_check": dep_check,
                }

            # Create job submission
            import uuid
            from datetime import datetime

            from job_submission import JobPriority, JobSubmission, JobType

            # Parse job type
            job_type_str = job_data.get("job_type", "train")
            try:
                job_type = JobType(job_type_str)
            except ValueError:
                job_type = JobType.TRAIN

            # Parse priority
            priority_str = job_data.get("priority", "NORMAL")
            try:
                priority = JobPriority[priority_str]
            except KeyError:
                priority = JobPriority.NORMAL

            job = JobSubmission(
                job_id=f"job-{uuid.uuid4().hex[:8]}",
                org_id=job_data.get("org_id", "gui-user"),
                job_type=job_type,
                priority=priority,
                name=job_data.get("name", f"Job {job_type.value}"),
                description=job_data.get("description", ""),
                script_path=script_path,
                script_content=script_content,
                requirements=requirements,
                base_model=job_data.get("base_model"),
                dataset_ref=job_data.get("dataset_ref"),
                dataset_format=job_data.get("dataset_format", "auto"),
                hyperparams=job_data.get("hyperparams", {}),
                total_steps=job_data.get("total_steps", 1000),
                trainer_type=job_data.get("trainer_type", "distribai"),
                training_phase=job_data.get("training_phase", "sft"),
                distributed_mode=job_data.get("distributed_mode", True),
                gradient_sync_steps=job_data.get("gradient_sync_steps", 100),
                checkpoint_steps=job_data.get("checkpoint_steps", 500),
                max_retries=job_data.get("max_retries", 3),
                submitted_by="gui",
                created_at=datetime.now(UTC),
            )

            # Submit to queue
            import asyncio

            job_id = asyncio.run(job_queue.submit(job))

            return {
                "success": True,
                "job_id": job_id,
                "dependency_check": dep_check,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def cancel_job(self, job_id: str) -> dict:
        """Cancel a running job."""
        if not JOB_SUBMISSION_AVAILABLE:
            return {"success": False, "error": "Job submission not available"}

        try:
            import asyncio

            success = asyncio.run(job_queue.cancel(job_id))

            # Also trigger emergency cancel for distributed jobs
            if success and self.runtime:
                try:
                    from distributed_trainer import get_distributed_trainer

                    trainer = get_distributed_trainer(self.runtime.node_service)
                    asyncio.run(trainer.emergency_cancel(job_id))
                except Exception as e:
                    print(f"[GUI] Emergency cancel warning: {e}")

            return {"success": success}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Admin Key Management API
    # =========================================================================

    def get_admin_requests(self) -> list:
        """Get pending admin key requests."""
        try:
            import asyncio

            from admin_keys import get_admin_key_manager

            manager = get_admin_key_manager()
            return asyncio.run(manager.get_pending_requests())
        except Exception as e:
            print(f"[GUI] Failed to get admin requests: {e}")
            return []

    def approve_admin_request(self, request_id: int, admin_name: str) -> dict:
        """Approve an admin key request."""
        try:
            import asyncio

            from admin_keys import get_admin_key_manager

            manager = get_admin_key_manager()
            node_id = asyncio.run(manager.approve_request(request_id, admin_name))

            if node_id:
                return {"success": True, "node_id": node_id}
            else:
                return {"success": False, "error": "Request not found or already processed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def reject_admin_request(self, request_id: int) -> dict:
        """Reject an admin key request."""
        try:
            import asyncio

            from admin_keys import get_admin_key_manager

            manager = get_admin_key_manager()
            success = asyncio.run(manager.reject_request(request_id))
            return {"success": success}
        except Exception as e:
            return {"success": False, "error": str(e)}


def get_html_content() -> str:
    dashboard_path = (
        Path(__file__).parent.parent / "worker" / "src" / "dashboard" / "static" / "index.html"
    )
    if dashboard_path.exists():
        return str(dashboard_path)

    return Path(__file__).parent / "server_dashboard.html"


def main():
    parser = argparse.ArgumentParser(description="DistribAI Server GUI")
    parser.add_argument("--windowed", action="store_true", help="Force windowed mode")
    parser.parse_args()

    api = ServerAPI()

    html = get_html_content()

    if not html.exists() if isinstance(html, Path) else False:
        html = ""

    window = webview.create_window(
        "DistribAI Server",
        html if html and not html.startswith("<") else f"file://{html}" if html else "",
        js_api=api,
        width=1400,
        height=900,
        min_size=(900, 600),
        resizable=True,
        text_select=True,
        confirm_close=True,
    )

    api.set_window(window)

    webview.start(debug=False, http_server=True if html else False)


if __name__ == "__main__":
    main()
