"""Admin key management system for DistribAI.

Provides encrypted admin tokens for nodes with request/approval flow.
Uses Fernet (AES-128-CBC) encryption for secure token exchange.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import secrets
from datetime import UTC, datetime, timedelta

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

import time

logger = logging.getLogger(__name__)

import jwt

from .database import get_database


class AdminKeyManager:
    """Manages encrypted admin keys for node authentication.

    Each node gets a unique encrypted key that must be validated
    for admin operations. Keys expire after a set duration.
    """

    def __init__(self):
        if not CRYPTO_AVAILABLE:
            raise ImportError("Cryptography library is required")

        self.db = get_database()
        self._key_cache: dict[str, tuple[str, datetime]] = {}  # node_id -> (key, expiry)
        self._master_key = self._get_or_create_master_key()
        self._lock = asyncio.Lock()

    def _get_or_create_master_key(self) -> bytes:
        """Get or create the master encryption key.

        The master key is derived from a secret stored in environment variable
        or generated once and persisted in the database.
        """
        secret = os.environ.get("DISTRIBAI_ADMIN_SECRET")

        if not secret:
            secret = secrets.token_urlsafe(32)
            secret_path = os.path.expanduser("~/.distribai/.admin_secret")
            os.makedirs(os.path.dirname(secret_path), exist_ok=True)
            if os.path.exists(secret_path):
                with open(secret_path) as f:
                    secret = f.read().strip()
            else:
                with open(secret_path, "w") as f:
                    f.write(secret)
                os.chmod(secret_path, 0o600)  # Owner read/write only

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"distribai_admin_salt",  # In production, use unique salt per installation
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(secret.encode()))
        return key

    async def request_admin_key(
        self,
        node_id: str,
        username: str | None = None,
    ) -> int:
        """Request an admin key for a node.

        Creates a pending request in the database. Orchestrator admin
        must approve this request via the GUI.

        Args:
            node_id: The requesting node's ID
            username: Optional username for identification

        Returns:
            Request ID for tracking
        """
        existing = await self.db.get_admin_key(node_id)
        if existing:
            expires_at = existing.get("expires_at")
            if expires_at:
                expiry = datetime.fromisoformat(expires_at)
                if expiry > datetime.now(UTC):
                    return 0  # Already has valid key

        request_id = await self.db.request_admin_key(node_id, username)

        print(f"[AdminKey] Admin key request #{request_id} from node {node_id} (user: {username})")

        return request_id

    async def approve_request(
        self,
        request_id: int,
        approved_by: str,
        expires_hours: int = 168,  # 7 days default
    ) -> str | None:
        """Approve an admin key request.

        Args:
            request_id: The request to approve
            approved_by: Username of admin approving
            expires_hours: Hours until key expires

        Returns:
            Node ID that was approved, or None if failed
        """
        async with self._lock:
            node_id = await self.db.approve_admin_request(request_id, approved_by)

            if not node_id:
                return None

            encrypted_key, expiry = await self._generate_encrypted_key(
                node_id,
                expires_hours,
            )

            await self.db.create_admin_key(
                node_id=node_id,
                encrypted_key=encrypted_key,
                expires_at=expiry,
            )

            self._key_cache[node_id] = (encrypted_key, expiry)

            print(f"[AdminKey] Approved request #{request_id} for node {node_id}, expires {expiry}")

            return node_id

    async def reject_request(self, request_id: int) -> bool:
        """Reject an admin key request.

        Args:
            request_id: The request to reject

        Returns:
            True if rejected successfully
        """
        return await self.db.reject_admin_request(request_id)

    async def get_pending_requests(self) -> list[dict]:
        """Get all pending admin key requests."""
        return await self.db.get_pending_admin_requests()

    async def validate_admin_token(
        self,
        node_id: str,
        encrypted_token: str,
    ) -> bool:
        """Validate an admin token from a node.

        Args:
            node_id: The node presenting the token
            encrypted_token: The encrypted token to validate

        Returns:
            True if token is valid and not expired
        """
        if node_id in self._key_cache:
            cached_key, expiry = self._key_cache[node_id]
            if expiry > datetime.now(UTC):
                if hmac.compare_digest(cached_key, encrypted_token):
                    return True

        stored = await self.db.get_admin_key(node_id)
        if not stored:
            return False

        expires_at = stored.get("expires_at")
        if expires_at:
            expiry = datetime.fromisoformat(expires_at)
            if expiry <= datetime.now(UTC):
                return False

        stored_key = stored.get("encrypted_key")
        if not stored_key:
            return False

        if hmac.compare_digest(stored_key, encrypted_token):
            self._key_cache[node_id] = (stored_key, expiry or datetime.now(UTC))
            return True

        return False

    async def revoke_admin_key(self, node_id: str) -> bool:
        """Revoke an admin key.

        Args:
            node_id: The node to revoke

        Returns:
            True if revoked successfully
        """
        async with self._lock:
            if node_id in self._key_cache:
                del self._key_cache[node_id]

            return await self.db.revoke_admin_key(node_id)

    async def get_encrypted_key(self, node_id: str) -> str | None:
        """Get the encrypted key for a node (to send to node).

        Args:
            node_id: The node to get key for

        Returns:
            Encrypted key string, or None if not found/expired
        """
        if node_id in self._key_cache:
            key, expiry = self._key_cache[node_id]
            if expiry > datetime.now(UTC):
                return key

        stored = await self.db.get_admin_key(node_id)
        if not stored:
            return None

        expires_at = stored.get("expires_at")
        if expires_at:
            expiry = datetime.fromisoformat(expires_at)
            if expiry <= datetime.now(UTC):
                return None
            key = stored.get("encrypted_key")
            self._key_cache[node_id] = (key, expiry)
            return key

        return stored.get("encrypted_key")

    async def _generate_encrypted_key(
        self,
        node_id: str,
        expires_hours: int,
    ) -> tuple[str, datetime]:
        """Generate an encrypted key for a node.

        Args:
            node_id: The node to generate key for
            expires_hours: Hours until key expires

        Returns:
            Tuple of (encrypted_key, expiry_datetime)
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError("Cryptography library is required")

        expiry = datetime.now(UTC) + timedelta(hours=expires_hours)

        payload = f"{node_id}:{expiry.isoformat()}:{secrets.token_hex(16)}"

        f = Fernet(self._master_key)
        encrypted = f.encrypt(payload.encode())

        return encrypted.decode(), expiry

    async def refresh_key(
        self,
        node_id: str,
        expires_hours: int = 168,
    ) -> str | None:
        """Refresh an existing admin key.

        Args:
            node_id: The node to refresh
            expires_hours: New expiry duration

        Returns:
            New encrypted key, or None if node has no existing key
        """
        existing = await self.db.get_admin_key(node_id)
        if not existing:
            return None

        async with self._lock:
            encrypted_key, expiry = await self._generate_encrypted_key(
                node_id,
                expires_hours,
            )

            await self.db.create_admin_key(
                node_id=node_id,
                encrypted_key=encrypted_key,
                expires_at=expiry,
            )

            self._key_cache[node_id] = (encrypted_key, expiry)

            return encrypted_key


_manager: AdminKeyManager | None = None


def get_admin_key_manager() -> AdminKeyManager:
    """Get or create global admin key manager."""
    global _manager
    if _manager is None:
        _manager = AdminKeyManager()
    return _manager


# Import here to avoid circular dependency issues
import base64

# JWT Security Functions for OWASP Compliance
# Global secret cache for consistency within the session
_jwt_secret_cache = None


def get_jwt_secret() -> str:
    """Get JWT secret from environment or generate a secure one."""
    global _jwt_secret_cache

    if _jwt_secret_cache:
        return _jwt_secret_cache

    secret = os.environ.get("JWT_SECRET")
    if not secret:
        secret = secrets.token_urlsafe(32)
        logger.warning(
            "JWT_SECRET not set; using an auto-generated secret for this process only"
        )

    _jwt_secret_cache = secret
    return secret


def generate_jwt_token(payload: dict) -> str:
    """Generate a JWT token with strong cryptographic signing.

    Args:
        payload: Dictionary containing token claims

    Returns:
        JWT token string
    """
    secret = get_jwt_secret()

    token_payload = {
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,  # 1 hour expiry
        **payload,
    }

    token = jwt.encode(token_payload, secret, algorithm="HS256")

    # Handle both PyJWT versions
    if isinstance(token, bytes):
        token = token.decode("utf-8")

    return token


def verify_jwt_token(token: str) -> dict | None:
    """Verify and decode a JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded payload if valid, None otherwise
    """
    try:
        secret = get_jwt_secret()
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        logger.info("JWT verification failed: token expired")
        return None
    except jwt.InvalidTokenError:
        logger.info("JWT verification failed: invalid token")
        return None


# Security constants for testing
ADMIN_KEYS = {}


def validate_password_strength(password: str) -> bool:
    """Validate password strength according to security policy.

    Args:
        password: Password to validate

    Returns:
        True if password meets strength requirements
    """
    import re

    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False
    return True
