"""
OAuth 2.0 Provider with PKCE for DistribAI

Implements Authorization Code Flow with PKCE for secure authentication.
Supports authorization code grant, refresh tokens, and client credentials.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GrantType(Enum):
    """
    OAuth 2.0 grant types.

    Attributes:
        AUTHORIZATION_CODE: Authorization code grant
        REFRESH_TOKEN: Refresh token grant
        CLIENT_CREDENTIALS: Client credentials grant

    Example:
        grant = GrantType.AUTHORIZATION_CODE
        print(f"Grant type: {grant.value}")
    """

    AUTHORIZATION_CODE = "authorization_code"
    REFRESH_TOKEN = "refresh_token"
    CLIENT_CREDENTIALS = "client_credentials"


class TokenType(Enum):
    """
    OAuth 2.0 token types.

    Attributes:
        BEARER: Bearer token type
        MAC: MAC token type

    Example:
        token_type = TokenType.BEARER
    """

    BEARER = "Bearer"
    MAC = "MAC"


@dataclass
class OAuthClient:
    """Registered OAuth 2.0 client (see server operator guide for setup)."""

    client_id: str
    client_secret: str
    redirect_uris: list[str]
    name: str
    description: str
    allowed_scopes: list[str]
    is_confidential: bool = True
    created_at: float = field(default_factory=time.time)


@dataclass
class AuthorizationCode:
    """Short-lived OAuth authorization code with optional PKCE metadata."""

    code: str
    client_id: str
    user_id: str
    redirect_uri: str
    scope: str
    code_challenge: str
    code_challenge_method: str = "S256"
    expires_at: float = field(default_factory=lambda: time.time() + 600)
    used: bool = False


@dataclass
class AccessToken:
    """Issued OAuth access token (and optional refresh token)."""

    token: str
    token_type: TokenType
    expires_at: float
    scope: str
    client_id: str
    user_id: str
    refresh_token: str | None = None


@dataclass
class PKCEParameters:
    """
    PKCE (Proof Key for Code Exchange) parameters.

    Used to enhance security for public clients by preventing
    authorization code interception attacks.

    Attributes:
        code_challenge: SHA-256 hash of the code verifier
        code_challenge_method: Method used (S256 for SHA-256)

    Example:
        verifier = PKCEParameters.generate_verifier()
        challenge = PKCEParameters.create_challenge(verifier)
        pkce = PKCEParameters(code_challenge=challenge)
    """

    code_challenge: str
    code_challenge_method: str = "S256"

    @staticmethod
    def generate_verifier() -> str:
        """
        Generate a random PKCE code verifier.

        Returns:
            Random URL-safe string for use as code verifier

        Example:
            >>> verifier = PKCEParameters.generate_verifier()
            >>> print(f"Verifier: {verifier}")
        """
        return secrets.token_urlsafe(64)

    @staticmethod
    def create_challenge(verifier: str) -> str:
        """
        Create PKCE code challenge from verifier.

        Args:
            verifier: Code verifier string

        Returns:
            SHA-256 hash of the verifier

        Example:
            >>> challenge = PKCEParameters.create_challenge(verifier)
            >>> print(f"Challenge: {challenge}")
        """
        return hashlib.sha256(verifier.encode()).digest().hex()


class OAuthProvider:
    """
    OAuth 2.0 Provider implementing authorization flows.

    Supports:
    - Authorization Code Flow with PKCE
    - Token refresh
    - Scope-based access control
    - Token introspection

    Attributes:
        db: Database manager for persistent storage
        clients: Registered OAuth clients
        authorization_codes: Active authorization codes
        access_tokens: Active access tokens
        refresh_tokens: Refresh token to access token mapping
        access_token_ttl: Access token time-to-live (default 3600 seconds)
        refresh_token_ttl: Refresh token time-to-live (default 30 days)
        authorization_code_ttl: Authorization code time-to-live (default 600 seconds)

    Example:
        provider = OAuthProvider(db_manager)
        provider.register_client(client)
        auth_code = provider.create_authorization_code(...)
        tokens = provider.exchange_code_for_token(...)
    """

    def __init__(self, db_manager=None):
        """
        Initialize the OAuth provider.

        Args:
            db_manager: Optional database manager for persistent storage

        Example:
            >>> provider = OAuthProvider(db_manager=db)
        """
        self.db = db_manager
        self.clients: dict[str, OAuthClient] = {}
        self.authorization_codes: dict[str, AuthorizationCode] = {}
        self.access_tokens: dict[str, AccessToken] = {}
        self.refresh_tokens: dict[str, str] = {}
        self.access_token_ttl = 3600
        self.refresh_token_ttl = 86400 * 30
        self.authorization_code_ttl = 600
        self._load_default_clients()

    def _load_default_clients(self) -> None:
        """
        Load default OAuth clients.

        Registers pre-configured clients for the CLI, web dashboard, and worker.
        """
        self.clients["distribai-cli"] = OAuthClient(
            client_id="distribai-cli",
            client_secret="",
            redirect_uris=["http://localhost:8765/callback"],
            name="DistribAI CLI",
            description="Command-line interface for DistribAI",
            allowed_scopes=["jobs:submit", "jobs:read", "nodes:read", "credits:read"],
            is_confidential=False,
        )
        self.clients["distribai-web"] = OAuthClient(
            client_id="distribai-web",
            client_secret=secrets.token_urlsafe(32),
            redirect_uris=["https://app.distribai.io/callback"],
            name="DistribAI Web Dashboard",
            description="Web dashboard for DistribAI",
            allowed_scopes=[
                "jobs:submit",
                "jobs:read",
                "jobs:write",
                "nodes:read",
                "credits:read",
                "credits:write",
                "admin",
            ],
            is_confidential=True,
        )
        self.clients["distribai-worker"] = OAuthClient(
            client_id="distribai-worker",
            client_secret=secrets.token_urlsafe(32),
            redirect_uris=["urn:ietf:wg:oauth:2.0:oob"],
            name="DistribAI Worker Node",
            description="Worker node machine authentication",
            allowed_scopes=[
                "worker:register",
                "worker:heartbeat",
                "jobs:execute",
                "gradients:submit",
            ],
            is_confidential=True,
        )

    def register_client(
        self,
        name: str,
        description: str,
        redirect_uris: list[str],
        allowed_scopes: list[str],
        is_confidential: bool = True,
    ) -> OAuthClient:
        """
        Register a new OAuth client.

        Args:
            name: Client application name
            description: Client description
            redirect_uris: List of allowed redirect URIs
            allowed_scopes: List of allowed OAuth scopes
            is_confidential: Whether client is confidential (has secret)

        Returns:
            Registered OAuthClient instance

        Example:
            >>> client = provider.register_client(
            ...     name="My App",
            ...     description="Sample application",
            ...     redirect_uris=["http://localhost:8080/callback"],
            ...     allowed_scopes=["read", "write"],
            ...     is_confidential=True
            ... )
            >>> print(f"Client ID: {client.client_id}")
        """
        client_id = f"distribai-{secrets.token_urlsafe(16)}"
        client_secret = secrets.token_urlsafe(32) if is_confidential else ""
        client = OAuthClient(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uris=redirect_uris,
            name=name,
            description=description,
            allowed_scopes=allowed_scopes,
            is_confidential=is_confidential,
        )
        self.clients[client_id] = client
        return client

    def create_authorization_request(
        self,
        client_id: str,
        redirect_uri: str,
        scope: str,
        code_challenge: str,
        code_challenge_method: str = "S256",
        state: str | None = None,
    ) -> dict[str, Any]:
        """
        Create authorization request (Step 1 of Authorization Code flow with PKCE).
        Returns authorization URL parameters.
        """
        client = self.clients.get(client_id)
        if not client:
            return {"error": "invalid_client", "error_description": "Unknown client"}
        if redirect_uri not in client.redirect_uris:
            return {"error": "invalid_request", "error_description": "Invalid redirect_uri"}
        requested_scopes = scope.split()
        invalid_scopes = [s for s in requested_scopes if s not in client.allowed_scopes]
        if invalid_scopes:
            return {
                "error": "invalid_scope",
                "error_description": f"Invalid scopes: {invalid_scopes}",
            }
        if not code_challenge:
            return {
                "error": "invalid_request",
                "error_description": "code_challenge required (PKCE)",
            }
        if code_challenge_method not in ("S256", "plain"):
            return {
                "error": "invalid_request",
                "error_description": "code_challenge_method must be S256 or plain",
            }
        return {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
            "state": state,
        }

    def authorize(
        self,
        client_id: str,
        user_id: str,
        redirect_uri: str,
        scope: str,
        code_challenge: str,
        code_challenge_method: str = "S256",
    ) -> str:
        """
        Create authorization code after user consent.
        Returns authorization code to be exchanged for tokens.
        """
        code = secrets.token_urlsafe(32)
        auth_code = AuthorizationCode(
            code=code,
            client_id=client_id,
            user_id=user_id,
            redirect_uri=redirect_uri,
            scope=scope,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            expires_at=time.time() + self.authorization_code_ttl,
        )
        self.authorization_codes[code] = auth_code
        return code

    def exchange_code(
        self,
        code: str,
        client_id: str,
        client_secret: str,
        code_verifier: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """
        Exchange authorization code for access token (PKCE verification).
        """
        client = self.clients.get(client_id)
        if not client:
            return {"error": "invalid_client"}
        if client.is_confidential:
            if not hmac.compare_digest(client.client_secret.encode(), client_secret.encode()):
                return {"error": "invalid_client", "error_description": "Invalid client_secret"}
        auth_code = self.authorization_codes.get(code)
        if not auth_code:
            return {"error": "invalid_grant", "error_description": "Invalid authorization code"}
        if auth_code.used:
            return {
                "error": "invalid_grant",
                "error_description": "Authorization code already used",
            }
        if time.time() > auth_code.expires_at:
            return {"error": "invalid_grant", "error_description": "Authorization code expired"}
        if auth_code.client_id != client_id:
            return {"error": "invalid_grant", "error_description": "Client ID mismatch"}
        if auth_code.redirect_uri != redirect_uri:
            return {"error": "invalid_grant", "error_description": "Redirect URI mismatch"}
        if auth_code.code_challenge_method == "S256":
            expected_challenge = hashlib.sha256(code_verifier.encode()).digest().hex()
            if expected_challenge != auth_code.code_challenge:
                return {"error": "invalid_grant", "error_description": "PKCE verification failed"}
        else:
            if code_verifier != auth_code.code_challenge:
                return {"error": "invalid_grant", "error_description": "PKCE verification failed"}
        auth_code.used = True
        return self._generate_tokens(
            client_id=client_id,
            user_id=auth_code.user_id,
            scope=auth_code.scope,
        )

    def _generate_tokens(self, client_id: str, user_id: str, scope: str) -> dict[str, Any]:
        now = time.time()
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)
        token = AccessToken(
            token=access_token,
            token_type=TokenType.BEARER,
            expires_at=now + self.access_token_ttl,
            scope=scope,
            client_id=client_id,
            user_id=user_id,
            refresh_token=refresh_token,
        )
        self.access_tokens[access_token] = token
        self.refresh_tokens[refresh_token] = access_token
        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": self.access_token_ttl,
            "refresh_token": refresh_token,
            "scope": scope,
        }

    def refresh_access_token(
        self, refresh_token: str, client_id: str, client_secret: str
    ) -> dict[str, Any]:
        client = self.clients.get(client_id)
        if not client:
            return {"error": "invalid_client"}
        if client.is_confidential:
            if not hmac.compare_digest(client.client_secret.encode(), client_secret.encode()):
                return {"error": "invalid_client"}
        old_access_token = self.refresh_tokens.get(refresh_token)
        if not old_access_token:
            return {"error": "invalid_grant", "error_description": "Invalid refresh token"}
        old_token = self.access_tokens.get(old_access_token)
        if not old_token:
            return {"error": "invalid_grant", "error_description": "Token not found"}
        self._revoke_token(old_access_token)
        return self._generate_tokens(
            client_id=client_id,
            user_id=old_token.user_id,
            scope=old_token.scope,
        )

    def _revoke_token(self, access_token: str):
        token = self.access_tokens.pop(access_token, None)
        if token and token.refresh_token:
            self.refresh_tokens.pop(token.refresh_token, None)

    def introspect_token(self, token: str) -> dict[str, Any]:
        access_token = self.access_tokens.get(token)
        if not access_token:
            return {"active": False}
        if time.time() > access_token.expires_at:
            return {"active": False}
        return {
            "active": True,
            "scope": access_token.scope,
            "client_id": access_token.client_id,
            "username": access_token.user_id,
            "token_type": access_token.token_type.value,
            "exp": int(access_token.expires_at),
            "iat": int(access_token.expires_at - self.access_token_ttl),
        }

    def validate_token(
        self, token: str, required_scope: str | None = None
    ) -> dict[str, Any] | None:
        introspection = self.introspect_token(token)
        if not introspection.get("active"):
            return None
        if required_scope:
            token_scopes = set(introspection.get("scope", "").split())
            if required_scope not in token_scopes:
                return None
        return introspection

    def cleanup_expired(self):
        now = time.time()
        expired_codes = [
            code
            for code, auth_code in self.authorization_codes.items()
            if now > auth_code.expires_at
        ]
        for code in expired_codes:
            del self.authorization_codes[code]
        expired_tokens = [
            token
            for token, access_token in self.access_tokens.items()
            if now > access_token.expires_at
        ]
        for token in expired_tokens:
            self._revoke_token(token)
