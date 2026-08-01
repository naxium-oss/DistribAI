"""Textual TUI for DistribAI: a terminal dashboard over the admin HTTP API.

Covers the operational core of the GUI dashboards (health, fleet, jobs,
credits, logs, local node settings) for headless boxes and terminal-first
operators. The browser dashboards remain the fully-featured surface — see
README.md's "CLI & TUI" section.

Run with ``distribai tui`` / ``distribai-tui`` / ``python -m scripts.cli.tui``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.containers import Horizontal, Vertical  # noqa: E402
from textual.screen import ModalScreen  # noqa: E402
from textual.widgets import (  # noqa: E402
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Log,
    Static,
    TabbedContent,
    TabPane,
)

from scripts.cli.api_client import AdminAPIClient  # noqa: E402

# tui.py reuses distribai_cli's local-config helpers (NodeManager) so the flat
# CLI and the TUI never diverge on how desktop.json is read/written; that
# module in turn lazily imports this one for the `distribai tui` subcommand,
# which is why the dependency is one-directional and safe here.
from scripts.cli.distribai_cli import NodeManager  # noqa: E402
from scripts.cli.identity import ensure_identity  # noqa: E402
from scripts.cli.process_manager import (  # noqa: E402
    ManagedProcess,
    orchestrator_argv,
    orchestrator_env,
)

REFRESH_SECONDS = 5.0


class NewJobModal(ModalScreen[dict | None]):
    """Small form to submit a job without leaving the TUI."""

    DEFAULT_CSS = """
    NewJobModal {
        align: center middle;
    }
    #new-job-box {
        width: 60;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #new-job-box Input {
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="new-job-box"):
            yield Label("Submit a new job")
            yield Input(placeholder="Model name (e.g. distribai-small)", id="model")
            yield Input(placeholder="Steps (e.g. 100)", id="steps", value="100")
            yield Input(placeholder="Batch size (e.g. 32)", id="batch_size", value="32")
            with Horizontal():
                yield Button("Submit", variant="primary", id="submit")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        model = self.query_one("#model", Input).value.strip() or "distribai-small"
        try:
            steps = int(self.query_one("#steps", Input).value.strip() or "100")
            batch_size = int(self.query_one("#batch_size", Input).value.strip() or "32")
        except ValueError:
            steps, batch_size = 100, 32
        self.dismiss({"model_name": model, "steps": steps, "batch_size": batch_size})


class DistribAITUI(App[None]):
    """Terminal dashboard: Overview / Nodes / Jobs / Credits / Settings / Logs."""

    TITLE = "DistribAI"
    SUB_TITLE = "terminal dashboard"

    CSS = """
    .metric-row {
        height: auto;
        padding: 1 2;
    }
    .metric {
        width: 1fr;
        border: round $accent;
        padding: 1 2;
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_all", "Refresh"),
        Binding("n", "new_job", "New job"),
        Binding("c", "cancel_job", "Cancel job"),
        Binding("s", "toggle_orchestrator", "Start/stop orchestrator"),
    ]

    def __init__(self, admin_url: str | None = None) -> None:
        super().__init__()
        self.client = AdminAPIClient(admin_url)
        self.node_manager = NodeManager()
        self._job_ids_by_row: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="overview"):
            with TabPane("Overview", id="overview"):
                with Horizontal(classes="metric-row"):
                    yield Static("", id="metric-health", classes="metric")
                    yield Static("", id="metric-nodes", classes="metric")
                    yield Static("", id="metric-jobs", classes="metric")
                yield Static("", id="overview-detail")
            with TabPane("Nodes", id="nodes"):
                yield DataTable(id="nodes-table")
            with TabPane("Jobs", id="jobs"):
                yield DataTable(id="jobs-table")
            with TabPane("Credits", id="credits"):
                yield DataTable(id="credits-table")
            with TabPane("Settings", id="settings"):
                yield Static("", id="settings-detail")
            with TabPane("Logs", id="logs"):
                yield Log(id="logs-view", highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#nodes-table", DataTable).add_columns(
            "Node ID", "Status", "Online", "Credits", "Hardware"
        )
        self.query_one("#jobs-table", DataTable).add_columns(
            "Job ID", "Status", "Model", "Steps"
        )
        self.query_one("#credits-table", DataTable).add_columns("Node ID", "Balance", "Lifetime")
        self.refresh_all()
        self.set_interval(REFRESH_SECONDS, self.refresh_all)

    def action_refresh_all(self) -> None:
        self.refresh_all()

    def refresh_all(self) -> None:
        self._refresh_overview()
        self._refresh_nodes()
        self._refresh_jobs()
        self._refresh_credits()
        self._refresh_settings()
        self._refresh_logs()

    def _refresh_overview(self) -> None:
        health = self.client.health()
        reachable = "error" not in health
        proc = ManagedProcess("orchestrator").status()
        health_widget = self.query_one("#metric-health", Static)
        if reachable:
            health_widget.update("[b green]Admin API reachable[/]")
        else:
            health_widget.update(f"[b red]Admin API unreachable[/]\n{health.get('error', '')}")

        self.query_one("#metric-nodes", Static).update(
            f"[b]Active nodes[/]\n{health.get('active_nodes', '—') if reachable else '—'}"
        )
        self.query_one("#metric-jobs", Static).update(
            f"[b]Queued / running jobs[/]\n"
            f"{health.get('queued_jobs', '—') if reachable else '—'} / "
            f"{health.get('running_jobs', '—') if reachable else '—'}"
        )
        proc_line = (
            f"[b green]orchestrator process running (pid {proc['pid']})[/]"
            if proc["running"]
            else "[dim]orchestrator process not started from this machine[/]"
        )
        self.query_one("#overview-detail", Static).update(
            f"Admin URL: {self.client.base_url}\n{proc_line}\n\n"
            "[dim]r[/] refresh   [dim]n[/] new job (Jobs tab)   "
            "[dim]c[/] cancel selected job (Jobs tab)   [dim]q[/] quit"
        )

    def _refresh_nodes(self) -> None:
        table = self.query_one("#nodes-table", DataTable)
        table.clear()
        for node in self.client.list_nodes():
            table.add_row(
                str(node.get("node_id", "N/A")),
                str(node.get("status", "unknown")),
                "yes" if node.get("online") else "no",
                f"{float(node.get('credits', 0) or 0):.2f}",
                str(node.get("hardware_summary") or ""),
            )

    def _refresh_jobs(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        table.clear()
        self._job_ids_by_row = []
        for job in self.client.list_jobs():
            job_id = str(job.get("job_id", "N/A"))
            table.add_row(
                job_id,
                str(job.get("status", "unknown")),
                str(job.get("model_name", "N/A")),
                str(job.get("steps", 0)),
            )
            self._job_ids_by_row.append(job_id)

    def _refresh_credits(self) -> None:
        table = self.query_one("#credits-table", DataTable)
        table.clear()
        for node_id, info in self.client.list_credits().items():
            table.add_row(
                node_id,
                f"{float(info.get('balance', 0) or 0):.2f}",
                f"{float(info.get('lifetime', 0) or 0):.2f}",
            )

    def _refresh_settings(self) -> None:
        config = self.node_manager.get_config()
        identity, changed = ensure_identity(config)
        if changed:
            self.node_manager.set_config(identity)
        node_proc = ManagedProcess("node").status()
        lines = [
            f"[b]Org ID:[/]  {identity.get('org_id')}",
            f"[b]Node ID:[/] {identity.get('node_id')}",
            f"[b]Node name:[/] {identity.get('node_name', 'Not set')}",
            f"[b]Region:[/] {identity.get('region', 'Not set')}",
            f"[b]CPU / GPU / RAM caps:[/] "
            f"{identity.get('cpuPercent', 50)}% / {identity.get('gpuPercent', 50)}% / "
            f"{identity.get('ramPercent', 50)}%",
            "",
            "[b green]worker daemon running[/]"
            if node_proc["running"]
            else "[dim]worker daemon not started from this machine[/]",
            "",
            "[dim]Edit resource caps: distribai node set-resources CPU GPU RAM[/]",
        ]
        self.query_one("#settings-detail", Static).update("\n".join(lines))

    def _refresh_logs(self) -> None:
        log_widget = self.query_one("#logs-view", Log)
        log_widget.clear()
        for line in self.client.tail_logs(200):
            log_widget.write_line(line)

    def action_new_job(self) -> None:
        def handle_result(result: dict | None) -> None:
            if not result:
                return
            response = self.client.post("/admin/jobs", result)
            if "error" in response:
                self.notify(f"Job submission failed: {response['error']}", severity="error")
            else:
                self.notify(f"Job created: {response.get('job_id', 'unknown')}")
                self._refresh_jobs()

        self.push_screen(NewJobModal(), handle_result)

    def action_toggle_orchestrator(self) -> None:
        proc = ManagedProcess("orchestrator")
        info = proc.status()
        if info["running"]:
            result = proc.stop()
            if result.get("ok"):
                self.notify(f"Orchestrator stopped (was pid {result.get('pid')})")
            else:
                self.notify(result.get("error", "stop failed"), severity="error")
        else:
            result = proc.start(orchestrator_argv(None, None), env=orchestrator_env(None, None))
            if result.get("ok", True):
                self.notify(f"Orchestrator started (pid {result.get('pid')})")
            else:
                self.notify(result.get("error", "start failed"), severity="error")
        self._refresh_overview()

    def action_cancel_job(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        if table.cursor_row is None or not self._job_ids_by_row:
            self.notify("Select a job row first", severity="warning")
            return
        try:
            job_id = self._job_ids_by_row[table.cursor_row]
        except IndexError:
            return
        response = self.client.delete(f"/admin/jobs/{job_id}")
        if "error" in response:
            self.notify(f"Cancel failed: {response['error']}", severity="error")
        else:
            self.notify(f"Cancelled {job_id}")
            self._refresh_jobs()


def run_tui(admin_url: str | None = None) -> None:
    DistribAITUI(admin_url).run()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="distribai-tui", description="DistribAI terminal dashboard")
    parser.add_argument(
        "--admin-url",
        default=None,
        help="Orchestrator admin URL (default ORCHESTRATOR_ADMIN_URL or http://127.0.0.1:8766)",
    )
    args = parser.parse_args(argv)
    run_tui(args.admin_url)


if __name__ == "__main__":
    main()
