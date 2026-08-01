"""Constants for DistribAI services.

This module contains all configuration constants used across the
orchestrator and related services.
"""

from __future__ import annotations

import logging
import os
import secrets
import stat
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)


def _secrets_dir() -> Path:
    """Return the on-disk directory where persistent server secrets live.

    Resolution order:
      1. ``DISTRIBAI_SECRETS_DIR`` env var (operators can point at a mounted
         secrets volume / KMS-backed tmpfs).
      2. ``<repo>/runtime/secrets/`` — gitignored, created with 0700.
    """
    override = os.getenv("DISTRIBAI_SECRETS_DIR")
    if override:
        path = Path(override)
    else:
        path = Path(__file__).resolve().parent.parent / "runtime" / "secrets"
    path.mkdir(parents=True, exist_ok=True)
    try:  # Windows ignores chmod, but on POSIX this matters.
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def _get_or_generate_secret(env_var: str, default_length: int = 32) -> str:
    """Return the secret for ``env_var``, persisting it across restarts.

    The previous implementation regenerated a random secret every process
    start whenever the env var was unset, which silently invalidated every
    issued JWT and broke the Merkle ledger's HMAC chain on each restart.

    New behaviour:
      1. If the env var is set, use it verbatim (operator-provided / KMS).
      2. Else read ``<secrets_dir>/<env_var>``, if present.
      3. Else generate a fresh token, atomically write it to disk with mode
         0600, log a one-line warning so operators know to back it up, and
         return it.

    Set ``DISTRIBAI_DISABLE_SECRET_PERSISTENCE=1`` to opt out (tests/CI).
    """
    value = os.getenv(env_var)
    if value:
        return value

    if os.getenv("DISTRIBAI_DISABLE_SECRET_PERSISTENCE") == "1":
        return secrets.token_urlsafe(default_length)

    secret_path = _secrets_dir() / env_var
    if secret_path.exists():
        try:
            existing = secret_path.read_text(encoding="utf-8").strip()
            if existing:
                return existing
        except OSError as exc:
            logger.warning("Failed to read persisted %s (%s); regenerating.", env_var, exc)

    new_secret = secrets.token_urlsafe(default_length)
    tmp_path = secret_path.with_suffix(".tmp")
    try:
        tmp_path.write_text(new_secret, encoding="utf-8")
        try:
            os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
        except OSError:
            pass
        os.replace(tmp_path, secret_path)
        logger.warning(
            "Generated and persisted new %s at %s. "
            "Back this up — losing it invalidates every issued JWT and breaks "
            "the credit-ledger HMAC chain.",
            env_var,
            secret_path,
        )
    except OSError as exc:
        logger.error(
            "Could not persist %s to %s (%s). "
            "Falling back to ephemeral secret — JWTs will be invalidated on restart!",
            env_var,
            secret_path,
            exc,
        )
    return new_secret


# =============================================================================
# Security Constants
# =============================================================================

JWT_SECRET: Final[str] = _get_or_generate_secret("JWT_SECRET")
JWT_ALGORITHM: Final[str] = "HS256"
SIGNING_KEY: Final[str] = _get_or_generate_secret("SIGNING_KEY")

DEFAULT_NODE_JWT_TTL_SECONDS: Final[int] = 6 * 60 * 60  # 6 hours

# =============================================================================
# Task & Scheduling Constants
# =============================================================================

DEFAULT_TASK_DEADLINE_SECONDS: Final[int] = 10 * 60  # 10 minutes
DEFAULT_STEPS_PER_TASK: Final[int] = 25
DEFAULT_REQUEUE_HEARTBEAT_TIMEOUT_SECONDS: Final[int] = 30

# =============================================================================
# Rate Limiting Constants
# =============================================================================

DEFAULT_RATE_LIMIT: Final[float] = 10.0  # requests per second
DEFAULT_RATE_LIMIT_CAPACITY: Final[float] = 20.0  # burst capacity
DEFAULT_RATE_LIMIT_BLOCK_DURATION: Final[float] = 60.0  # seconds

# =============================================================================
# S3 / Storage Constants
# =============================================================================

S3_PRESIGNED_URL_EXPIRATION: Final[int] = 3600  # 1 hour
S3_DEFAULT_REGION: Final[str] = "us-east-1"

# =============================================================================
# Network Constants
# =============================================================================

DEFAULT_GRPC_PORT: Final[str] = "50051"
DEFAULT_ADMIN_PORT: Final[str] = "8766"
DEFAULT_ADMIN_HOST: Final[str] = "127.0.0.1"

# =============================================================================
# Byzantine Detection Constants
# =============================================================================

DEFAULT_BYZANTINE_FRACTION: Final[float] = 0.2
DEFAULT_REPUTATION_THRESHOLD: Final[float] = 0.3
GRADIENT_HISTORY_SIZE: Final[int] = 100

# =============================================================================
# Credit & Staking Constants
# =============================================================================

DEFAULT_STAKING_APR: Final[float] = 0.05  # 5% annual yield
DEFAULT_STAKING_LOCK_DAYS: Final[int] = 30

SLASHING_BYZANTINE: Final[float] = 0.5  # 50% slash
SLASHING_DOUBLE_VOTE: Final[float] = 0.3  # 30% slash
SLASHING_SYBIL: Final[float] = 1.0  # 100% slash
SLASHING_CHECKPOINT: Final[float] = 0.1  # 10% slash
SLASHING_DOWNTIME: Final[float] = 0.05  # 5% slash

# =============================================================================
# Voting Constants
# =============================================================================

MAX_VOTES_PER_HOUR: Final[int] = 1000
MAX_VOTES_PER_JOB: Final[int] = 500
DEFAULT_VOTE_DURATION_HOURS: Final[int] = 24
DEFAULT_CREDIT_PRICE_PER_VOTE: Final[int] = 10
MAX_TASK_CREDITS_REPORTED: Final[float] = 100.0

# =============================================================================
# Pagination Constants
# =============================================================================

DEFAULT_PAGE_SIZE: Final[int] = 20
MAX_PAGE_SIZE: Final[int] = 100

# =============================================================================
# PoC Challenge Constants
# =============================================================================

POC_CHALLENGE_DIFFICULTY: Final[int] = 12  # bits
POC_CHALLENGE_EXPIRY_SECONDS: Final[int] = 300  # 5 minutes
POC_CHALLENGE_LENGTH: Final[int] = 6  # hex digits

# =============================================================================
# Sybil Detection Constants
# =============================================================================

SYBIL_ACCOUNT_AGE_THRESHOLD_DAYS: Final[int] = 7
SYBIL_HIGH_VALUE_JOB_THRESHOLD: Final[int] = 1000
SYBIL_MAX_ACCOUNTS_PER_IP: Final[int] = 5
SYBIL_MAX_ACCOUNTS_PER_HARDWARE: Final[int] = 3
SYBIL_RAPID_CREATION_WINDOW_SECONDS: Final[int] = 3600  # 1 hour
SYBIL_RAPID_CREATION_THRESHOLD: Final[int] = 10
SYBIL_COORDINATED_VOTE_WINDOW_SECONDS: Final[int] = 300  # 5 minutes
