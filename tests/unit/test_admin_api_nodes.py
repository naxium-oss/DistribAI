"""Unit tests for admin_api.nodes module."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from services_python.admin_api.nodes import NodesHandler
from services_python.db_manager import DBManager


class TestNodesHandler:
    """Test cases for NodesHandler."""

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
    def nodes_handler(self, mock_db, mock_node_service):
        """Create a NodesHandler instance."""
        return NodesHandler(mock_db, mock_node_service)

    def test_init(self, mock_db, mock_node_service):
        """Test NodesHandler initialization."""
        handler = NodesHandler(mock_db, mock_node_service)
        assert handler.db == mock_db
        assert handler.node_service == mock_node_service

    @pytest.mark.asyncio
    async def test_list_nodes_success(self, nodes_handler, mock_db):
        """Test successful node listing."""
        # Setup mock request
        mock_request = MagicMock()

        # Mock database response
        expected_nodes = [
            {"node_id": "node-1", "status": "online", "last_heartbeat": "2023-01-01T00:00:00Z"},
            {"node_id": "node-2", "status": "offline", "last_heartbeat": "2023-01-01T01:00:00Z"},
        ]
        mock_db.get_all_nodes.return_value = expected_nodes

        # Call the handler
        response = await nodes_handler.list(mock_request)

        # Verify response
        assert response.status == 200
        data = json.loads(response.text)
        assert "nodes" in data
        assert len(data["nodes"]) == 2

        # Verify database was called
        mock_db.get_all_nodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_nodes_empty(self, nodes_handler, mock_db):
        """Test node listing when no nodes exist."""
        # Setup mock request
        mock_request = MagicMock()

        # Mock database response - no nodes
        mock_db.get_all_nodes.return_value = []

        # Call the handler
        response = await nodes_handler.list(mock_request)

        # Verify response
        assert response.status == 200
        data = json.loads(response.text)
        assert data["nodes"] == []

    @pytest.mark.asyncio
    async def test_list_nodes_database_error(self, nodes_handler, mock_db):
        """Test node listing with database error."""
        # Setup mock request
        mock_request = MagicMock()

        # Mock database error
        mock_db.get_all_nodes.side_effect = Exception("Database error")

        # Call the handler - should raise exception
        with pytest.raises(Exception, match="Database error"):
            await nodes_handler.list(mock_request)

    @pytest.mark.asyncio
    async def test_set_contributing_success(self, nodes_handler, mock_db):
        """Test successful node contributing status update."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = "node-1"
        mock_request.json = AsyncMock(return_value={"contributing": True})

        # Mock database response
        mock_db.set_node_contributing = MagicMock()

        # Call the handler
        response = await nodes_handler.set_contributing(mock_request)

        # Verify response
        assert response.status == 200
        data = json.loads(response.text)
        assert data["ok"] is True
        assert data["node_id"] == "node-1"
        assert data["contributing"] is True

        # Verify database was called
        mock_db.set_node_contributing.assert_called_once_with("node-1", True)

    @pytest.mark.asyncio
    async def test_set_contributing_missing_node_id(self, nodes_handler):
        """Test node contributing status update with missing node_id."""
        # Setup mock request without node_id
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = None

        # Call the handler
        response = await nodes_handler.set_contributing(mock_request)

        # Verify response
        assert response.status == 400
        data = json.loads(response.text)
        assert data["error"] == "missing node_id"

    @pytest.mark.asyncio
    async def test_set_contributing_invalid_json(self, nodes_handler):
        """Test node contributing status update with invalid JSON."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = "node-1"
        mock_request.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))

        # Call the handler
        response = await nodes_handler.set_contributing(mock_request)

        # Verify response
        assert response.status == 400
        data = json.loads(response.text)
        assert data["error"] == "invalid JSON"

    @pytest.mark.asyncio
    async def test_set_contributing_database_error(self, nodes_handler, mock_db):
        """Test node contributing status update with database error."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = "node-1"
        mock_request.json = AsyncMock(return_value={"contributing": True})

        # Mock database error
        mock_db.set_node_contributing.side_effect = Exception("Database error")

        # Call the handler - should raise exception
        with pytest.raises(Exception, match="Database error"):
            await nodes_handler.set_contributing(mock_request)

    @pytest.mark.asyncio
    async def test_list_paginated_success(self, nodes_handler, mock_db):
        """Test successful paginated node listing."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.query = {}

        # Mock database response
        expected_nodes = [
            {"node_id": "node-1", "status": "online", "last_heartbeat_ts": "2023-01-01T00:00:00Z"},
            {"node_id": "node-2", "status": "offline", "last_heartbeat_ts": "2023-01-01T01:00:00Z"},
        ]
        mock_db.get_all_nodes.return_value = expected_nodes

        # Call the handler
        response = await nodes_handler.list_paginated(mock_request)

        # Verify response
        assert response.status == 200
        data = json.loads(response.text)
        assert "data" in data
        assert "pagination" in data

        # Verify database was called
        mock_db.get_all_nodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_paginated_with_sorting(self, nodes_handler, mock_db):
        """Test paginated node listing with sorting parameters."""
        # Setup mock request with sorting
        mock_request = MagicMock()
        mock_request.query = {"sort": "status", "order": "desc"}

        # Mock database response
        expected_nodes = [
            {"node_id": "node-1", "status": "online"},
            {"node_id": "node-2", "status": "offline"},
        ]
        mock_db.get_all_nodes.return_value = expected_nodes

        # Call the handler
        response = await nodes_handler.list_paginated(mock_request)

        # Verify response
        assert response.status == 200
        data = json.loads(response.text)
        assert "data" in data
        assert "pagination" in data

        # Verify database was called
        mock_db.get_all_nodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_paginated_database_error(self, nodes_handler, mock_db):
        """Test paginated node listing with database error."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.query = {}

        # Mock database error
        mock_db.get_all_nodes.side_effect = Exception("Database error")

        # Call the handler - should raise exception
        with pytest.raises(Exception, match="Database error"):
            await nodes_handler.list_paginated(mock_request)

    @pytest.mark.asyncio
    async def test_set_contributing_default_value(self, nodes_handler, mock_db):
        """Test node contributing status update with default contributing value."""
        # Setup mock request without contributing field
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = "node-1"
        mock_request.json = AsyncMock(return_value={})  # Empty body

        # Mock database response
        mock_db.set_node_contributing = MagicMock()

        # Call the handler
        response = await nodes_handler.set_contributing(mock_request)

        # Verify response - should default to True
        assert response.status == 200
        data = json.loads(response.text)
        assert data["ok"] is True
        assert data["node_id"] == "node-1"
        assert data["contributing"] is True  # Default value

        # Verify database was called
        mock_db.set_node_contributing.assert_called_once_with("node-1", True)

    @pytest.mark.asyncio
    async def test_set_contributing_false_value(self, nodes_handler, mock_db):
        """Test node contributing status update set to False."""
        # Setup mock request with contributing=False
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = "node-1"
        mock_request.json = AsyncMock(return_value={"contributing": False})

        # Mock database response
        mock_db.set_node_contributing = MagicMock()

        # Call the handler
        response = await nodes_handler.set_contributing(mock_request)

        # Verify response
        assert response.status == 200
        data = json.loads(response.text)
        assert data["ok"] is True
        assert data["node_id"] == "node-1"
        assert data["contributing"] is False

        # Verify database was called
        mock_db.set_node_contributing.assert_called_once_with("node-1", False)

    @pytest.mark.asyncio
    async def test_list_paginated_sorting_type_error(self, nodes_handler, mock_db):
        """Test paginated node listing with sorting type error."""
        # Setup mock request with invalid sort field
        mock_request = MagicMock()
        mock_request.query = {"page": "1", "per_page": "10", "sort": "invalid_field"}

        # Mock database response with nodes that have different data types
        expected_nodes = [
            {"node_id": "node1", "invalid_field": "string_value"},
            {"node_id": "node2", "invalid_field": 123},
        ]
        mock_db.get_all_nodes.return_value = expected_nodes

        # Call the handler - should handle TypeError gracefully
        response = await nodes_handler.list_paginated(mock_request)

        # Verify response is still successful despite sorting error
        assert response.status == 200
        data = json.loads(response.text)
        assert "data" in data
        assert len(data["data"]) == 2

    @pytest.mark.asyncio
    async def test_list_paginated_sorting_key_error(self, nodes_handler, mock_db):
        """Test paginated node listing with sorting key error."""
        # Setup mock request with non-existent sort field
        mock_request = MagicMock()
        mock_request.query = {"page": "1", "per_page": "10", "sort": "nonexistent_field"}

        # Mock database response
        expected_nodes = [
            {"node_id": "node1", "status": "online"},
            {"node_id": "node2", "status": "offline"},
        ]
        mock_db.get_all_nodes.return_value = expected_nodes

        # Call the handler - should handle KeyError gracefully
        response = await nodes_handler.list_paginated(mock_request)

        # Verify response is still successful despite missing field
        assert response.status == 200
        data = json.loads(response.text)
        assert "data" in data
        assert len(data["data"]) == 2
