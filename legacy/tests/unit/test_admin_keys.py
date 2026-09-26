"""Tests for admin_keys module."""

import os
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, mock_open, patch

import pytest

# Import the module under test
import services_python.admin_keys as admin_keys_module


class TestAdminKeyManager:
    """Test cases for AdminKeyManager class."""

    def test_module_import(self):
        """Test that the module can be imported."""
        assert hasattr(admin_keys_module, "AdminKeyManager")
        assert hasattr(admin_keys_module, "CRYPTO_AVAILABLE")

    @pytest.fixture
    def mock_db(self):
        """Mock database fixture."""
        return Mock()

    @pytest.fixture
    def manager(self, mock_db):
        """Create AdminKeyManager instance with mocked dependencies."""
        with patch("services_python.admin_keys.get_database", return_value=mock_db):
            return admin_keys_module.AdminKeyManager()

    def test_init_without_crypto(self, mock_db):
        """Test initialization when crypto is not available."""
        with patch("services_python.admin_keys.CRYPTO_AVAILABLE", False):
            with patch("services_python.admin_keys.get_database", return_value=mock_db):
                with pytest.raises(ImportError, match="Cryptography library is required"):
                    admin_keys_module.AdminKeyManager()

    def test_get_or_create_master_key_from_env(self, mock_db):
        """Test master key creation from environment variable."""
        test_secret = "test_secret_key_for_admin"
        with patch.dict(os.environ, {"DISTRIBAI_ADMIN_SECRET": test_secret}):
            with patch("services_python.admin_keys.get_database", return_value=mock_db):
                manager = admin_keys_module.AdminKeyManager()
                # Should not raise an exception
                assert manager._master_key is not None

    def test_get_or_create_master_key_from_db(self, mock_db):
        """Test master key retrieval from database."""
        # Mock database response
        mock_db.get_admin_secret.return_value = "stored_secret_key"

        with patch.dict(os.environ, {}, clear=True):
            with patch("services_python.admin_keys.get_database", return_value=mock_db):
                manager = admin_keys_module.AdminKeyManager()
                assert manager._master_key is not None

    def test_get_or_create_master_key_generate_new(self, mock_db):
        """Test master key generation when none exists."""
        # Mock the file system operations
        with (
            patch("os.path.exists", return_value=False),
            patch("builtins.open", mock_open()) as mock_file,
            patch("os.chmod"),
            patch.dict(os.environ, {}, clear=True),
        ):
            manager = admin_keys_module.AdminKeyManager()
            assert manager._master_key is not None
            # Check that the secret file was written
            mock_file.assert_called_once()

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_generate_admin_key_success(self, manager):
        """Test successful admin key generation."""
        node_id = "test_node_123"

        # Mock the database methods - _generate_encrypted_key doesn't store in DB
        with patch.object(manager.db, "get_admin_key", return_value=None):
            encrypted_key, expiry = await manager._generate_encrypted_key(
                node_id, expires_hours=168
            )

            assert encrypted_key is not None
            assert isinstance(encrypted_key, str)
            assert isinstance(expiry, datetime)
            # Verify the key is encrypted (not plain text)
            assert node_id not in encrypted_key

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_generate_admin_key_without_crypto(self, manager):
        """Test admin key generation without crypto available."""
        with patch("services_python.admin_keys.CRYPTO_AVAILABLE", False):
            with pytest.raises(ImportError, match="Cryptography library is required"):
                await manager._generate_encrypted_key("test_node", expires_hours=168)

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_validate_admin_key_success(self, manager):
        """Test successful admin key validation."""
        node_id = "test_node_123"
        token = "valid_token"

        # Mock the database to return a valid key
        async def mock_get_admin_key(node_id):
            future_time = datetime.now(UTC) + timedelta(hours=1)
            return {"encrypted_key": token, "expires_at": future_time.isoformat()}

        with patch.object(manager.db, "get_admin_key", side_effect=mock_get_admin_key):
            result = await manager.validate_admin_token(node_id, token)

            assert result is True

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_validate_admin_key_invalid_node(self, manager):
        """Test admin key validation with wrong node ID."""

        # Mock the database to return a valid key for a different node
        async def mock_get_admin_key(requested_node_id):
            if requested_node_id == "wrong_node":
                return None  # No key found for wrong node
            future_time = datetime.now(UTC) + timedelta(hours=1)
            return {"encrypted_key": "valid_token", "expires_at": future_time.isoformat()}

        with patch.object(manager.db, "get_admin_key", side_effect=mock_get_admin_key):
            # This will fail because no key is found for wrong_node
            result = await manager.validate_admin_token("wrong_node", "valid_token")

            assert result is False

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_validate_admin_key_expired(self, manager):
        """Test admin key validation with expired token."""
        node_id = "test_node_123"
        token = "expired_token"

        # Mock the database to return an expired key
        expired_time = datetime.now(UTC) - timedelta(hours=1)

        async def mock_get_admin_key(node_id):
            return {"encrypted_key": token, "expires_at": expired_time.isoformat()}

        with patch.object(manager.db, "get_admin_key", side_effect=mock_get_admin_key):
            result = await manager.validate_admin_token(node_id, token)

            assert result is False

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_validate_admin_key_invalid_token(self, manager):
        """Test admin key validation with invalid token."""
        node_id = "test_node_123"
        token = "invalid_token"

        # Mock the database to return None (no key found)
        async def mock_get_admin_key(node_id):
            return None

        with patch.object(manager.db, "get_admin_key", side_effect=mock_get_admin_key):
            result = await manager.validate_admin_token(node_id, token)

            assert result is False

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_generate_encrypted_key_encryption(self, manager):
        """Test that _generate_encrypted_key properly encrypts data."""
        node_id = "test_node_123"

        # Test the actual encryption method
        encrypted_key, expiry = await manager._generate_encrypted_key(node_id, expires_hours=24)

        assert encrypted_key is not None
        assert isinstance(encrypted_key, str)
        assert isinstance(expiry, datetime)
        # Verify the key is encrypted (not plain text)
        assert node_id not in encrypted_key
        # Verify it's not the original plaintext
        assert len(encrypted_key) > len(node_id)
        # Fernet tokens are URL-safe base64, so they should only contain valid characters
        import re

        assert re.match(r"^[A-Za-z0-9_-]+=*$", encrypted_key), (
            "Encrypted key should contain only valid base64url characters"
        )

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_validate_admin_token_integration(self, manager):
        """Test full admin token validation integration."""
        node_id = "test_node_123"

        # Generate a real encrypted key
        encrypted_key, expiry = await manager._generate_encrypted_key(node_id, expires_hours=24)

        # Mock the database to return this key
        async def mock_get_admin_key(requested_node_id):
            if requested_node_id == node_id:
                return {"encrypted_key": encrypted_key, "expires_at": expiry.isoformat()}
            return None

        with patch.object(manager.db, "get_admin_key", side_effect=mock_get_admin_key):
            # Test validation with the correct token
            result = await manager.validate_admin_token(node_id, encrypted_key)
            assert result is True

            # Test validation with wrong token
            result = await manager.validate_admin_token(node_id, "wrong_token")
            assert result is False

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_validate_admin_token_expired_integration(self, manager):
        """Test admin token validation with expired token."""
        node_id = "test_node_123"

        # Generate an expired key
        encrypted_key, expiry = await manager._generate_encrypted_key(node_id, expires_hours=1)

        # Mock the database to return an expired key
        expired_time = datetime.now(UTC) - timedelta(hours=2)  # Expired 2 hours ago

        async def mock_get_admin_key(requested_node_id):
            if requested_node_id == node_id:
                return {"encrypted_key": encrypted_key, "expires_at": expired_time.isoformat()}
            return None

        with patch.object(manager.db, "get_admin_key", side_effect=mock_get_admin_key):
            # Test validation with expired token
            result = await manager.validate_admin_token(node_id, encrypted_key)
            assert result is False

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_revoke_admin_key(self, manager):
        """Test admin key revocation."""
        node_id = "test_node_123"

        # Add a key to cache first
        manager._key_cache[node_id] = ("test_key", datetime.now(UTC) + timedelta(hours=1))

        # Mock the database method
        async def mock_revoke_admin_key(node_id):
            return True

        with patch.object(manager.db, "revoke_admin_key", side_effect=mock_revoke_admin_key):
            result = await manager.revoke_admin_key(node_id)

            assert result is True
            assert node_id not in manager._key_cache

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_key_cache_expiry_behavior(self, manager):
        """Test that expired keys in cache are properly handled."""
        node_id = "test_node_123"

        # Add an expired key to cache
        expired_time = datetime.now(UTC) - timedelta(hours=1)
        manager._key_cache[node_id] = ("test_key", expired_time)

        # Mock database to return no key (simulating expired state)
        async def mock_get_admin_key(requested_node_id):
            return None

        with patch.object(manager.db, "get_admin_key", side_effect=mock_get_admin_key):
            # Validation should fail due to no valid key in database
            result = await manager.validate_admin_token(node_id, "test_key")
            assert result is False

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    def test_key_cache_management(self, manager):
        """Test key cache management functionality."""
        current_time = datetime.now(UTC)

        # Add some keys to cache
        node1 = "node1"
        node2 = "node2"

        manager._key_cache[node1] = ("key1", current_time + timedelta(hours=1))
        manager._key_cache[node2] = ("key2", current_time + timedelta(hours=2))

        # Test cache contents
        assert len(manager._key_cache) == 2
        assert node1 in manager._key_cache
        assert node2 in manager._key_cache
        assert all(isinstance(expiry, datetime) for _, expiry in manager._key_cache.values())

        # Test cache clearing
        manager._key_cache.clear()
        assert len(manager._key_cache) == 0

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_key_cache_ttl_integration(self, manager):
        """Test that cached keys respect TTL through actual validation."""
        node_id = "test_node"

        # Generate a real encrypted key with 1 hour expiry
        encrypted_key, expiry = await manager._generate_encrypted_key(node_id, expires_hours=1)

        # Mock the database to return this key
        async def mock_get_admin_key(requested_node_id):
            if requested_node_id == node_id:
                return {"encrypted_key": encrypted_key, "expires_at": expiry.isoformat()}
            return None

        with patch.object(manager.db, "get_admin_key", side_effect=mock_get_admin_key):
            # Test validation with valid token
            result = await manager.validate_admin_token(node_id, encrypted_key)
            assert result is True

            # Verify cache entry has correct expiry
            assert node_id in manager._key_cache
            key, cache_expiry = manager._key_cache[node_id]
            assert cache_expiry > datetime.now(UTC)
            assert cache_expiry <= datetime.now(UTC) + timedelta(hours=24)

    @pytest.mark.skipif(not admin_keys_module.CRYPTO_AVAILABLE, reason="Cryptography not available")
    @pytest.mark.asyncio
    async def test_concurrent_key_generation(self, manager):
        """Test concurrent key generation through database calls."""
        node_id = "test_node"

        # Generate a real encrypted key
        encrypted_key, expiry = await manager._generate_encrypted_key(node_id, expires_hours=24)

        # Mock the database to return the same key
        call_count = 0

        async def mock_get_admin_key(requested_node_id):
            nonlocal call_count
            call_count += 1
            if requested_node_id == node_id:
                return {"encrypted_key": encrypted_key, "expires_at": expiry.isoformat()}
            return None

        with patch.object(manager.db, "get_admin_key", side_effect=mock_get_admin_key):
            # Test validation multiple times
            result1 = await manager.validate_admin_token(node_id, encrypted_key)
            result2 = await manager.validate_admin_token(node_id, encrypted_key)

            # Both should succeed
            assert result1 is True
            assert result2 is True
            # Database should be called once (second call hits cache)
            assert call_count == 1
            # Verify cache is populated
            assert node_id in manager._key_cache
