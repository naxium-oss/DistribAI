"""Tests for server GUI module."""

import pytest

try:
    import webview

    HAS_WEBVIEW = True
except ImportError:
    webview = None
    HAS_WEBVIEW = False

try:
    from services_python.server_gui import ServerAPI

    HAS_SERVER_GUI = True
except ImportError:
    HAS_SERVER_GUI = False
    ServerAPI = None


@pytest.mark.skipif(not HAS_SERVER_GUI, reason="server_gui not available")
def test_server_gui_import():
    """Test server GUI module imports."""
    assert ServerAPI is not None


@pytest.mark.skipif(not HAS_SERVER_GUI, reason="server_gui not available")
def test_server_api_creation():
    """Test ServerAPI can be instantiated."""
    api = ServerAPI()
    assert api is not None


@pytest.mark.skipif(not HAS_SERVER_GUI, reason="server_gui not available")
def test_server_api_attributes():
    """Test ServerAPI has expected attributes."""
    api = ServerAPI()
    assert hasattr(api, "window")
    assert hasattr(api, "server_running")
    assert hasattr(api, "start_time")


@pytest.mark.skipif(not HAS_SERVER_GUI, reason="server_gui not available")
def test_server_api_methods():
    """Test ServerAPI has expected methods."""
    api = ServerAPI()
    assert hasattr(api, "set_window")
    assert hasattr(api, "_get_full_status")
    assert hasattr(api, "_emit_event")
