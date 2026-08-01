"""Unit tests for admin_api.credits module."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from services_python.admin_api.credits import CreditsHandler
from services_python.db_manager import DBManager


class TestCreditsHandler:
    """Test cases for CreditsHandler."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database manager."""
        db = MagicMock(spec=DBManager)
        return db

    @pytest.fixture
    def mock_node_service(self):
        """Create a mock node service."""
        node_service = MagicMock()
        node_service._authenticate_request = MagicMock()
        return node_service

    @pytest.fixture
    def mock_credit_ledger(self):
        """Create a mock credit ledger."""
        return MagicMock()

    @pytest.fixture
    def mock_credit_transfers(self):
        """Create a mock credit transfers manager."""
        return MagicMock()

    @pytest.fixture
    def credits_handler(
        self, mock_db, mock_node_service, mock_credit_ledger, mock_credit_transfers
    ):
        """Create a CreditsHandler instance."""
        return CreditsHandler(mock_db, mock_credit_ledger, mock_credit_transfers, mock_node_service)

    def test_init(self, mock_db, mock_node_service, mock_credit_ledger, mock_credit_transfers):
        """Test CreditsHandler initialization."""
        handler = CreditsHandler(
            mock_db, mock_credit_ledger, mock_credit_transfers, mock_node_service
        )
        assert handler.db == mock_db
        assert handler.credit_ledger == mock_credit_ledger
        assert handler.credit_transfers == mock_credit_transfers
        assert handler.node_service == mock_node_service

    @pytest.mark.asyncio
    async def test_list_credits(self, credits_handler, mock_db):
        """Test listing all credits."""
        # Mock database response
        mock_credits = {
            "node1": {"balance": 1000, "lifetime": 2000},
            "node2": {"balance": 500, "lifetime": 1000},
        }
        mock_db.list_all_credits.return_value = mock_credits

        # Call the handler
        response = await credits_handler.list(MagicMock())

        # Verify response
        assert response.status == 200
        # For aiohttp Response, we need to check the text attribute
        import json

        data = json.loads(response.text)
        assert data["credits"] == mock_credits

        # Verify database was called
        mock_db.list_all_credits.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_credits_success(self, credits_handler, mock_db):
        """Test successful credits retrieval."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = "test_node_id"

        # Mock database response
        expected_credits = {"balance": 1000.0, "lifetime": 2000.0, "votes_cast": 5}
        mock_db.get_node_credits.return_value = expected_credits

        # Call the handler
        response = await credits_handler.get(mock_request)

        # Verify response
        assert response.status == 200
        import json

        data = json.loads(response.text)
        assert data["node_id"] == "test_node_id"
        assert data["balance"] == 1000.0

        # Verify database was called
        mock_db.get_node_credits.assert_called_once_with("test_node_id")

    @pytest.mark.asyncio
    async def test_get_credits_missing_node_id(self, credits_handler):
        """Test credits retrieval with missing node_id."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = None

        # Call the handler
        response = await credits_handler.get(mock_request)

        # Verify response
        assert response.status == 400
        import json

        data = json.loads(response.text)
        assert data["error"] == "missing node_id"

    @pytest.mark.asyncio
    async def test_get_credits_not_found(self, credits_handler, mock_db):
        """Test credits retrieval when node not found."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = "nonexistent_node"

        # Mock database response - node not found
        mock_db.get_node_credits.return_value = None

        # Call the handler
        response = await credits_handler.get(mock_request)

        # Verify response - should return default values when not found
        assert response.status == 200
        import json

        data = json.loads(response.text)
        assert data["node_id"] == "nonexistent_node"
        assert data["balance"] == 0
        assert data["lifetime"] == 0
        assert data["votes_cast"] == 0

    @pytest.mark.asyncio
    async def test_get_credits_database_error(self, credits_handler, mock_db):
        """Test credits retrieval with database error."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = "test_node"

        # Mock database error
        mock_db.get_node_credits.side_effect = Exception("Database error")

        # Call the handler - should raise exception
        with pytest.raises(Exception, match="Database error"):
            await credits_handler.get(mock_request)

    @pytest.mark.asyncio
    async def test_get_transfer_stats(self, credits_handler, mock_credit_transfers):
        """Test getting transfer statistics."""
        # Mock transfer stats
        mock_stats = {"total_transfers": 100, "total_volume": 10000.0}
        mock_credit_transfers.get_stats.return_value = mock_stats

        # Call the handler
        response = await credits_handler.get_transfer_stats(MagicMock())

        # Verify response
        assert response.status == 200
        import json

        data = json.loads(response.text)
        assert data == mock_stats

        # Verify transfer manager was called
        mock_credit_transfers.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_paginated(self, credits_handler, mock_db):
        """Test list credits with pagination."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.query = {"page": "1", "per_page": "10", "sort": "node_id"}

        # Mock database response
        mock_credits = {
            "node1": {
                "balance": 100.0,
                "lifetime": 200.0,
                "votes_cast": 5,
                "created_ts": "2023-01-01T00:00:00Z",
            },
            "node2": {
                "balance": 150.0,
                "lifetime": 300.0,
                "votes_cast": 3,
                "created_ts": "2023-01-02T00:00:00Z",
            },
        }
        mock_db.list_all_credits.return_value = mock_credits

        # Call the handler
        response = await credits_handler.list_paginated(mock_request)

        # Verify response
        assert response.status == 200
        import json

        data = json.loads(response.text)
        assert "data" in data
        assert "pagination" in data
        assert len(data["data"]) == 2

        # Verify database was called
        mock_db.list_all_credits.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_balance_v1(self, credits_handler, mock_node_service, mock_db):
        """Test V1 API get balance (SQL totals match admin /admin/credits/{node})."""
        mock_request = MagicMock()
        mock_claims = {"sub": "test_node"}
        mock_node_service._authenticate_request.return_value = mock_claims

        mock_db.get_node_credits.return_value = {
            "balance": 1500.0,
            "lifetime": 2200.0,
            "votes_cast": 0.0,
        }

        response = await credits_handler.get_balance_v1(mock_request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["node_id"] == "test_node"
        assert data["confirmed"] == 1500.0
        assert data["pending"] == 0.0
        assert data["lifetime_earned"] == 2200.0

        mock_node_service._authenticate_request.assert_called_once_with(
            mock_request, required_kind="node"
        )
        mock_db.get_node_credits.assert_called_once_with("test_node")

    @pytest.mark.asyncio
    async def test_transfer_v1_success(
        self, credits_handler, mock_node_service, mock_credit_transfers
    ):
        """Test V1 API successful transfer."""
        # Setup mock request and authentication
        mock_request = MagicMock()
        mock_request.json = AsyncMock(
            return_value={"recipient": "recipient_node", "amount": 100.0, "memo": "test transfer"}
        )
        mock_claims = {"sub": "sender_node"}
        mock_node_service._authenticate_request.return_value = mock_claims

        # Mock transfer result
        mock_result = {"success": True, "transfer_id": "tx123"}
        mock_credit_transfers.transfer = AsyncMock(return_value=mock_result)

        # Call the handler
        response = await credits_handler.transfer_v1(mock_request)

        # Verify response
        assert response.status == 200
        import json

        data = json.loads(response.text)
        assert data == mock_result

        # Verify transfer was called
        mock_credit_transfers.transfer.assert_called_once_with(
            "sender_node", "recipient_node", 100.0, "test transfer"
        )

    @pytest.mark.asyncio
    async def test_transfer_v1_invalid_json(self, credits_handler, mock_node_service):
        """Test V1 API transfer with invalid JSON."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))

        # Call the handler
        response = await credits_handler.transfer_v1(mock_request)

        # Verify response
        assert response.status == 400
        data = json.loads(response.text)
        assert data["error"] == "invalid JSON"

    @pytest.mark.asyncio
    async def test_transfer_v1_invalid_parameters(self, credits_handler, mock_node_service):
        """Test V1 API transfer with invalid parameters."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={"recipient": "", "amount": -100.0})
        mock_node_service._authenticate_request.return_value = {"sub": "sender_node"}

        # Call the handler
        response = await credits_handler.transfer_v1(mock_request)

        # Verify response
        assert response.status == 400
        import json

        data = json.loads(response.text)
        assert data["error"] == "invalid parameters"
