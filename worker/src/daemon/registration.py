"""
Registration Manager for DistribAI Worker

Handles automated node registration including hardware detection,
PoC challenge solving, and JWT token acquisition.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import platform
import re
import secrets
import time
from typing import Any

import psutil

logger = logging.getLogger(__name__)


def _sanitize_log_message(msg: str) -> str:
    """Sanitize log messages by redacting sensitive information."""
    if not msg:
        return ""
    jwt_pattern = r"ey[a-zA-Z0-9_-]+\.ey[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"
    msg = re.sub(jwt_pattern, "[REDACTED_JWT]", msg)
    msg = re.sub(r"(Authorization:?\s*)(\S+)", r"\1[REDACTED]", msg, flags=re.IGNORECASE)
    msg = re.sub(
        r"(password|secret|token|key)(\s*[:=]\s*)(\S+)", r"\1\2[REDACTED]", msg, flags=re.IGNORECASE
    )
    sanitized = "".join(c for c in msg if c.isprintable() or c in "\n\r\t")
    return sanitized[:1000] if len(sanitized) > 1000 else sanitized


class RedactingFilter(logging.Filter):
    """Logging filter that redacts sensitive information."""

    def filter(self, record):
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        record.msg = _sanitize_log_message(message)
        record.args = ()
        return True


logger.addFilter(RedactingFilter())


class RegistrationManager:
    """
    Handles automated node registration for DistribAI workers.

    Manages the registration process including:
    - Hardware detection and reporting
    - Proof-of-Work challenge solving
    - JWT token acquisition
    - Retry logic on failures

    Attributes:
        admin_url: URL of the orchestrator admin API
        node_id: Unique identifier for this worker node

    Example:
        reg_mgr = RegistrationManager(
            orchestrator_admin_url="http://localhost:8766",
            node_id="worker-001"
        )
        jwt_token = await reg_mgr.register()
    """

    def __init__(self, orchestrator_admin_url: str, node_id: str):
        """
        Initialize the registration manager.

        Args:
            orchestrator_admin_url: URL of the orchestrator admin API
            node_id: Unique identifier for this worker node

        Example:
            >>> reg_mgr = RegistrationManager(
            ...     orchestrator_admin_url="http://localhost:8766",
            ...     node_id="worker-001"
            ... )
        """
        self.admin_url = orchestrator_admin_url
        self.node_id = node_id

    def detect_hardware(self) -> dict[str, Any]:
        """
        Detect and report hardware specifications.

        Returns:
            Dictionary with hardware information including:
                - node_id: Node identifier
                - os: Operating system
                - cpu_count: Physical CPU cores
                - cpu_threads: Logical CPU threads
                - ram_total_gb: Total RAM in GB
                - gpu_model: GPU model name
                - vram_mb: GPU VRAM in MB

        Example:
            >>> hw = reg_mgr.detect_hardware()
            >>> print(f"CPU cores: {hw['cpu_count']}")
            >>> print(f"RAM: {hw['ram_total_gb']} GB")
        """
        hardware = {
            "node_id": self.node_id,
            "os": platform.system(),
            "os_release": platform.release(),
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "gpu_model": "CPU-Only",
            "vram_mb": 0,
            "ts": int(time.time()),
        }
        try:
            import torch

            if torch.cuda.is_available():
                hardware["gpu_model"] = torch.cuda.get_device_name(0)
                hardware["vram_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024**2)
                hardware["cuda_version"] = torch.version.cuda
        except ImportError:
            pass
        return hardware

    async def solve_sha256_poc(
        self, challenge_hex: str, difficulty: int, max_attempts: int = 1000000
    ) -> str | None:
        start = time.time()
        attempts = 0
        while attempts < max_attempts:
            nonce = secrets.token_hex(16)
            data = f"{challenge_hex}{nonce}".encode()
            hash_result = hashlib.sha256(data).hexdigest()
            hash_int = int(hash_result, 16)
            leading_zeros = 256 - hash_int.bit_length()
            if leading_zeros >= difficulty:
                duration = time.time() - start
                logger.info(f"Solved PoC in {duration:.2f}s after {attempts} attempts")
                return nonce
            attempts += 1
            if attempts % 100000 == 0:
                await asyncio.sleep(0)
        return None

    async def register(self) -> str | None:
        """
        Perform the full registration flow.
        Returns the JWT token if successful.
        """
        import aiohttp

        async with aiohttp.ClientSession() as session:
            logger.info(f"Requesting PoC challenge from {self.admin_url}")
            try:
                async with session.post(f"{self.admin_url}/v1/nodes/challenge") as resp:
                    if resp.status != 200:
                        logger.error(f"Failed to get challenge: {await resp.text()}")
                        return None
                    challenge = await resp.json()
            except Exception as e:
                logger.error(f"Connection error to orchestrator: {e}")
                return None
            challenge_id = challenge["challenge_id"]
            challenge_hex = challenge["challenge_hex"]
            difficulty = challenge["difficulty"]
            logger.info(f"Solving PoC challenge (difficulty={difficulty})...")
            nonce = await self.solve_sha256_poc(challenge_hex, difficulty)
            if not nonce:
                logger.error("Failed to solve PoC challenge")
                return None
            hardware = self.detect_hardware()
            reg_payload = {
                "node_id": self.node_id,
                "challenge_id": challenge_id,
                "nonce": nonce,
                "os": hardware["os"],
                "gpu_model": hardware["gpu_model"],
                "vram_mb": hardware["vram_mb"],
                "cpu_cores": hardware["cpu_count"],
                "ram_gb": hardware["ram_total_gb"],
            }
            logger.info(f"Submitting registration for {self.node_id}")
            async with session.post(
                f"{self.admin_url}/v1/nodes/register-enhanced", json=reg_payload
            ) as resp:
                text = await resp.text()
                if resp.status not in (200, 201):
                    logger.error(f"Registration failed ({resp.status}): {text}")
                    return None
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    logger.error(f"Registration invalid JSON: {text[:500]}")
                    return None
                return data.get("jwt")
