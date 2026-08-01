"""Surface why queued work is stalled (fleet-wide and per-job blockers)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from services_python.db_manager import DBManager
    from services_python.orchestrator_grpc import NodeService


def build_fleet_summary(node_service: NodeService, db: DBManager) -> dict[str, Any]:
    """Describe connected / idle / busy / offline workers for assignment diagnostics."""
    nodes = {node["node_id"]: node for node in db.get_all_nodes()}
    connected = set(node_service.connected_nodes)

    idle: list[str] = []
    busy: list[str] = []
    non_contributing: list[str] = []
    for node_id in connected:
        node = nodes.get(node_id)
        if node is not None and node.get("contributing", True) is False:
            non_contributing.append(node_id)
            continue
        if node_id in node_service.pending_assignments:
            busy.append(node_id)
        else:
            idle.append(node_id)

    offline = [nid for nid in nodes if nid not in connected]
    queue_depth = db.get_queue_depth()

    return {
        "queue_depth": queue_depth,
        "connected_count": len(connected),
        "idle_count": len(idle),
        "busy_count": len(busy),
        "offline_registered_count": len(offline),
        "non_contributing_count": len(non_contributing),
        "idle_node_ids": idle[:20],
        "busy_node_ids": busy[:20],
        "offline_node_ids": offline[:20],
    }


def _fleet_level_blockers(summary: dict[str, Any]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    if summary["queue_depth"] <= 0:
        return blockers

    if summary["connected_count"] == 0:
        blockers.append(
            {
                "code": "no_workers_connected",
                "message": "No workers are connected on gRPC; tasks cannot be assigned.",
            }
        )
        if summary["offline_registered_count"] > 0:
            blockers.append(
                {
                    "code": "workers_offline",
                    "message": (
                        f"{summary['offline_registered_count']} node(s) registered in SQLite "
                        "but not connected to the orchestrator."
                    ),
                }
            )
        return blockers

    if summary["idle_count"] == 0:
        if summary["busy_count"] > 0:
            blockers.append(
                {
                    "code": "all_workers_busy",
                    "message": "Every connected contributing worker already has a pending assignment.",
                }
            )
        if summary["non_contributing_count"] > 0:
            blockers.append(
                {
                    "code": "workers_not_contributing",
                    "message": (
                        f"{summary['non_contributing_count']} connected node(s) are marked "
                        "non-contributing and will not receive tasks."
                    ),
                }
            )

    return blockers


def diagnose_job_blockers(
    node_service: NodeService,
    db: DBManager,
    job: dict[str, Any],
    *,
    fleet_summary: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """List readable blockers for a job or task still waiting in the queue."""
    status = (job.get("status") or "").lower()
    if status not in ("queued", "pending"):
        return []

    summary = fleet_summary or build_fleet_summary(node_service, db)
    blockers = list(_fleet_level_blockers(summary))

    submitter = job.get("submitter_id")
    if submitter:
        trusted = {row["node_id"] for row in db.list_trusted_submitters()}
        if trusted and submitter not in trusted:
            blockers.append(
                {
                    "code": "submitter_not_trusted",
                    "message": (
                        f"Submitter '{submitter}' is not in the trusted submitters list "
                        "(trust gate may delay or block scheduling)."
                    ),
                }
            )

    if not blockers and summary["queue_depth"] > 0 and summary["idle_count"] > 0:
        blockers.append(
            {
                "code": "scheduler_pending",
                "message": "Workers are idle; the scheduler should assign this job shortly.",
            }
        )

    return blockers


def enrich_jobs_with_queue_hints(
    node_service: NodeService,
    db: DBManager,
    jobs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach ``queue_blockers`` on each job and return the shared fleet summary."""
    summary = build_fleet_summary(node_service, db)
    enriched: list[dict[str, Any]] = []
    for job in jobs:
        row = dict(job)
        blockers = diagnose_job_blockers(node_service, db, row, fleet_summary=summary)
        if blockers:
            row["queue_blockers"] = blockers
        enriched.append(row)
    return enriched, summary
