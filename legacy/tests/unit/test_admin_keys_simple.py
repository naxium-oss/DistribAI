"""Simple tests for admin_keys module that match the actual implementation."""

import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest

# Import the module under test
import services_python.admin_keys as admin_keys_module


class TestAdminKeyManagerSimple:
    """Simple tests for AdminKeyManager that match actual implementation."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database."""
        db = Mock()
        db.get_admin_secret = Mock(return_value=None)
        db.store_admin_secret = Mock(return_value=None)
        db.get_admin_key = AsyncMock(return_value=None)
        db.store_admin_key = AsyncMock(return_value=None)
        return db

    @pytest.fixture
    def manager(self, mock_db):
        """Create an AdminKeyManager instance with mocked database."""
        with patch("services_python.admin_keys.get_database", return_value=mock_db):
            return admin_keys_module.AdminKeyManager()

    def test_module_import(self):
        """Test that the module can be imported."""
        assert admin_keys_module is not None
        assert hasattr(admin_keys_module, "AdminKeyManager")
        assert hasattr(admin_keys_module, "get_admin_key_manager")

    def test_init_without_crypto(self):
        """Test initialization without crypto available."""
        with patch("services_python.admin_keys.CRYPTO_AVAILABLE", False):
            with pytest.raises(ImportError, match="Cryptography library is required"):
                admin_keys_module.AdminKeyManager()

    def test_get_or_create_master_key_from_env(self, mock_db):
        """Test master key from environment variable."""
        test_secret = "test_secret_key_123"
        with patch.dict(os.environ, {"DISTRIBAI_ADMIN_SECRET": test_secret}):
            with patch("services_python.admin_keys.get_database", return_value=mock_db):
                manager = admin_keys_module.AdminKeyManager()
                assert manager._master_key is not None

    def test_get_or_create_master_key_from_db(self, mock_db):
        """Test master key from database."""
        mock_db.get_admin_secret.return_value = "stored_secret"
        with patch("services_python.admin_keys.get_database", return_value=mock_db):
            manager = admin_keys_module.AdminKeyManager()
            assert manager._master_key is not None

    def test_get_or_create_master_key_generate_new(self, mock_db):
        """Test master key generation when none exists."""
        # Mock the file system operations
        with (
            patch("os.path.exists", return_value=False),
            patch("builtins.open", mock_open()),
            patch("os.chmod"),
            patch.dict(os.environ, {}, clear=True),
        ):
            manager = admin_keys_module.AdminKeyManager()
            assert manager._master_key is not None

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_request_admin_key_success(self, manager, mock_db):
        """Test successful admin key request."""
        node_id = "test_node_123"

        # Mock database to return no existing key
        mock_db.get_admin_key = AsyncMock(return_value=None)
        mock_db.store_admin_key = AsyncMock(return_value=1)
        mock_db.request_admin_key = AsyncMock(return_value=123)

        result = await manager.request_admin_key(node_id)

        assert isinstance(result, int)
        assert result > 0

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_request_admin_key_existing(self, manager, mock_db):
        """Test admin key request when key already exists."""
        node_id = "test_node_123"

        # Mock database to return existing valid key
        future_time = datetime.now(UTC) + timedelta(hours=24)
        mock_db.get_admin_key.return_value = {"expires_at": future_time.isoformat()}

        result = await manager.request_admin_key(node_id)

        assert result == 0  # Already has valid key

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_approve_request_success(self, manager, mock_db):
        """Test successful request approval."""
        request_id = 123
        node_id = "test_node_123"

        # Mock database methods
        mock_db.get_admin_request = AsyncMock(
            return_value={"node_id": node_id, "username": "test_user"}
        )
        mock_db.store_admin_key = AsyncMock(return_value=None)
        mock_db.delete_admin_request = AsyncMock(return_value=True)
        mock_db.approve_admin_request = AsyncMock(return_value=node_id)
        mock_db.create_admin_key = AsyncMock(return_value=None)

        result = await manager.approve_request(request_id, approved_by="admin_user")

        assert result == node_id

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_reject_request_success(self, manager, mock_db):
        """Test successful request rejection."""
        request_id = 123

        mock_db.delete_admin_request = AsyncMock(return_value=True)
        mock_db.reject_admin_request = AsyncMock(return_value=True)

        result = await manager.reject_request(request_id)

        assert result is True

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_get_pending_requests(self, manager, mock_db):
        """Test getting pending requests."""
        mock_db.get_admin_requests = AsyncMock(
            return_value=[
                {"request_id": 1, "node_id": "node1"},
                {"request_id": 2, "node_id": "node2"},
            ]
        )
        mock_db.get_pending_admin_requests = AsyncMock(
            return_value=[
                {"request_id": 1, "node_id": "node1"},
                {"request_id": 2, "node_id": "node2"},
            ]
        )

        result = await manager.get_pending_requests()

        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_validate_admin_token_success(self, manager, mock_db):
        """Test successful admin token validation."""
        node_id = "test_node_123"
        encrypted_token = "valid_token"

        # Mock the database to return a valid key
        future_time = datetime.now(UTC) + timedelta(hours=24)
        mock_db.get_admin_key.return_value = {
            "encrypted_key": encrypted_token,
            "expires_at": future_time.isoformat(),
        }

        result = await manager.validate_admin_token(node_id, encrypted_token)

        assert result is True

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_validate_admin_token_invalid(self, manager, mock_db):
        """Test admin token validation with invalid token."""
        node_id = "test_node_123"
        encrypted_token = "invalid_token"

        # Mock the database to return None (no key found)
        mock_db.get_admin_key.return_value = None

        result = await manager.validate_admin_token(node_id, encrypted_token)

        assert result is False

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_validate_admin_token_expired(self, manager, mock_db):
        """Test admin token validation with expired token."""
        node_id = "test_node_123"
        encrypted_token = "expired_token"

        # Mock the database to return an expired key
        expired_time = datetime.now(UTC) - timedelta(hours=1)
        mock_db.get_admin_key.return_value = {
            "encrypted_key": encrypted_token,
            "expires_at": expired_time.isoformat(),
        }

        result = await manager.validate_admin_token(node_id, encrypted_token)

        assert result is False

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_revoke_admin_key_success(self, manager, mock_db):
        """Test successful admin key revocation."""
        node_id = "test_node_123"

        mock_db.delete_admin_key = AsyncMock(return_value=True)
        mock_db.revoke_admin_key = AsyncMock(return_value=True)

        result = await manager.revoke_admin_key(node_id)

        assert result is True
        # Check that key was removed from cache
        assert node_id not in manager._key_cache

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_get_encrypted_key_success(self, manager, mock_db):
        """Test getting encrypted key."""
        node_id = "test_node_123"
        encrypted_key = "encrypted_data"

        mock_db.get_admin_key.return_value = {
            "encrypted_key": encrypted_key,
            "expires_at": (datetime.now(UTC) + timedelta(hours=24)).isoformat(),
        }

        result = await manager.get_encrypted_key(node_id)

        assert result == encrypted_key

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_get_encrypted_key_not_found(self, manager, mock_db):
        """Test getting encrypted key when not found."""
        node_id = "test_node_123"

        mock_db.get_admin_key.return_value = None

        result = await manager.get_encrypted_key(node_id)

        assert result is None

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_refresh_key_success(self, manager, mock_db):
        """Test key refresh."""
        node_id = "test_node_123"

        # Mock database methods - simulate existing key
        mock_db.get_admin_key = AsyncMock(
            return_value={
                "encrypted_key": "old_encrypted_key",
                "expires_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),  # Expired
            }
        )
        mock_db.store_admin_key = AsyncMock(return_value=None)
        mock_db.create_admin_key = AsyncMock(return_value=None)

        result = await manager.refresh_key(node_id)

        assert result is not None
        assert isinstance(result, str)

    def test_get_admin_key_manager(self):
        """Test getting admin key manager singleton."""
        manager = admin_keys_module.get_admin_key_manager()
        assert manager is not None
        assert isinstance(manager, admin_keys_module.AdminKeyManager)

    def test_get_jwt_secret(self):
        """Test getting JWT secret."""
        secret = admin_keys_module.get_jwt_secret()
        assert secret is not None
        assert isinstance(secret, str)
        assert len(secret) > 0

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    def test_generate_jwt_token(self):
        """Test JWT token generation."""
        payload = {"node_id": "test_node", "exp": datetime.now(UTC) + timedelta(hours=24)}
        token = admin_keys_module.generate_jwt_token(payload)
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    def test_verify_jwt_token_success(self):
        """Test JWT token verification success."""
        payload = {"node_id": "test_node", "exp": datetime.now(UTC) + timedelta(hours=24)}
        token = admin_keys_module.generate_jwt_token(payload)

        result = admin_keys_module.verify_jwt_token(token)

        assert result is not None
        assert result["node_id"] == "test_node"

    def test_verify_jwt_token_invalid(self):
        """Test JWT token verification with invalid token."""
        result = admin_keys_module.verify_jwt_token("invalid_token")
        assert result is None

    def test_validate_password_strength(self):
        """Test password strength validation."""
        # Strong password
        assert admin_keys_module.validate_password_strength("StrongP@ssw0rd!") is True

        # Weak passwords
        assert admin_keys_module.validate_password_strength("weak") is False
        assert admin_keys_module.validate_password_strength("") is False
        assert admin_keys_module.validate_password_strength("123") is False
