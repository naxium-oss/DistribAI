"""
Test that no mock data is served by the dashboard or orchestrator.
This ensures real backend integration is active.
"""

import pytest

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False

from pathlib import Path


def _client_dashboard_sources() -> str:
    """Combined contributor dashboard server + extracted route/lib modules."""
    root = Path(__file__).parent.parent.parent / "client"
    parts = [root / "server.js"]
    for subdir in ("routes", "lib"):
        d = root / subdir
        if d.is_dir():
            parts.extend(sorted(d.glob("*.js")))
    return "".join(p.read_text(encoding="utf-8") for p in parts if p.is_file())


class TestNoMockData:
    """Verify that no mock/preview data is being served."""

    def test_server_js_no_mock_routes(self):
        """Verify server.js doesn't contain mock route handlers."""
        content = _client_dashboard_sources()

        assert "if (GUI_PREVIEW_MOCKS)" not in content, (
            "GUI_PREVIEW_MOCKS conditional routes should be removed"
        )
        assert "preview-node" not in content, "Mock node names should not exist"
        assert "preview-merkle-root" not in content, "Mock merkle root should not exist"

    def test_examples_do_not_emit_fabricated_results(self):
        """Contributor examples must execute real work or fail explicitly."""
        root = Path(__file__).parent.parent.parent
        inference = (root / "examples" / "inference_template.py").read_text(encoding="utf-8")
        training = (root / "examples" / "train_template.py").read_text(encoding="utf-8")
        golden = (root / "examples" / "golden_template" / "run.py").read_text(encoding="utf-8")
        for content in (inference, training, golden):
            assert "Placeholder results" not in content
            assert "Generated text for:" not in content
            assert "pytorch-stub" not in content
        assert "AutoModelForCausalLM" in inference
        assert "torch.save" in training
        assert "MSELoss" in golden

    def test_server_js_has_real_endpoints(self):
        """Verify server.js has real API endpoints for hardware info."""
        content = _client_dashboard_sources()

        # Check that new real endpoints exist
        assert "/api/system/info" in content, "Hardware info endpoint should exist"
        assert "/api/settings/resources" in content, "Resource settings endpoint should exist"
        assert "/api/regions" in content, "Regions endpoint should exist"
        assert "/api/status" in content, "Status endpoint should exist"

    def test_executor_has_resource_limits(self):
        """Verify executor.py has resource limit enforcement."""
        executor_py = (
            Path(__file__).parent.parent.parent / "worker" / "src" / "daemon" / "executor.py"
        )
        content = executor_py.read_text(encoding="utf-8")

        # Check for resource limit functions
        assert "get_resource_limits" in content, "Resource limits function should exist"
        assert "apply_cpu_limit" in content, "CPU limit function should exist"
        assert "apply_gpu_limit" in content, "GPU limit function should exist"
        assert "ResourceMonitor" in content, "Resource monitor class should exist"

    def test_preview_script_no_mock_flag(self):
        """Verify preview_gui.py doesn't set GUI_PREVIEW_MOCKS."""
        preview_py = Path(__file__).parent.parent.parent / "scripts" / "dev" / "preview_gui.py"
        content = preview_py.read_text(encoding="utf-8")

        # Should NOT contain the old mock flag
        assert "GUI_PREVIEW_MOCKS" not in content, (
            "GUI_PREVIEW_MOCKS should be removed from preview script"
        )

    def test_html_uses_real_data_fetching(self):
        """Verify index preview sources fetch real data from API endpoints."""
        node_dir = (
            Path(__file__).parent.parent.parent
            / "worker"
            / "src"
            / "dashboard"
            / "static"
            / "node"
        )
        content = "".join(
            (node_dir / name).read_text(encoding="utf-8")
            for name in (
                "index.html",
                "index-preview.js",
                "index-security.js",
                "index-dev-panel.js",
            )
            if (node_dir / name).is_file()
        )

        # Check for real data fetching (these are the key indicators of real backend integration)
        assert "fetch('/api/system/info')" in content, "Should fetch real system info"
        assert "fetch('/api/status')" in content, "Should fetch real status"
        assert "fetch('/api/settings/resources')" in content, "Should fetch resource settings"
        assert "checkDashboardStatus" in content, "Should have orchestrator status checker"


class TestRealDataIntegration:
    """Test that real data is being fetched from endpoints."""

    def test_system_info_endpoint_structure(self):
        """Verify /api/system/info returns expected structure."""
        content = _client_dashboard_sources()

        # Check the endpoint returns cpu, memory, gpu info
        assert "cpu.brand" in content or "cpu: {" in content, "Should return CPU info"
        assert "gpu.model" in content or "gpu:" in content, "Should return GPU info"
        assert "totalGB" in content or "memory: { total" in content, "Should return memory info"

    def test_status_endpoint_structure(self):
        """Verify /api/status returns orchestrator connection status."""
        content = _client_dashboard_sources()

        # Check for orchestrator status in the response
        assert "orchestrator" in content.lower(), "Status should include orchestrator info"
        assert (
            "orchAdminUrl" in content or "orchestratorUrl" in content.lower() or "url:" in content
        ), "Should include orchestrator URL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
