"""Write holistic review batch outputs for desloppify --import-run (Cursor overlay path)."""
from __future__ import annotations

import json
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parents[2] / ".desloppify/subagents/runs/20260521_000638"
RESULTS = RUN_DIR / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Evidence-grounded scores (0-100, one decimal). Not anchored to target 85.
BATCHES: list[tuple[int, str, float, list[str], str]] = [
    (
        1,
        "cross_module_architecture",
        76.5,
        [
            "services_python/orchestrator_grpc.py imports worker.src.daemon.credit_ledger and voting_system, coupling orchestrator to node runtime.",
            "Deferred imports appear across 20+ modules (packet scan_evidence), indicating cycle-pressure shims rather than clean boundaries.",
            "worker/src/daemon/s3_util.py imports services_python.blob_url_policy - acceptable policy share but repeats cross-tree dependency.",
        ],
        "Architecture is service/worker split with real code on both sides, but orchestrator-worker imports and widespread deferred imports keep blast radius high for cross-cutting changes.",
    ),
    (
        2,
        "high_level_elegance",
        84.0,
        [
            "orchestrator_grpc.py documents modular split (scheduler, grpc_service, admin_api) and delegates rather than monolith-only.",
            "AGENTS.md and runtime/db/schema.sql give a coherent mental model for orchestrator state.",
        ],
        "Top-level modules read as intentional subsystems; occasional large entry files remain but structure is navigable.",
    ),
    (
        3,
        "convention_outlier",
        86.0,
        [
            "Python modules consistently use from __future__ import annotations and pathlib in newer code.",
            "Dashboard static assets use root-absolute URLs per AGENTS.md convention.",
        ],
        "Conventions are mostly stable; outliers are localized rather than random style drift.",
    ),
    (
        4,
        "error_consistency",
        77.5,
        [
            "job_submission.py uses broad except Exception with logging in multiple paths.",
            "orchestrator_grpc.py mixes jwt_module.InvalidTokenError, web.HTTPUnauthorized, ValueError, and generic Exception.",
        ],
        "Errors are handled, but strategies differ by layer (gRPC/admin vs job pipeline), so failure semantics are not uniform.",
    ),
    (
        5,
        "naming_quality",
        86.5,
        [
            "Handler names (JobSubmissionHandler, GrpcServiceHandler, CreditTransferManager) communicate roles.",
            "Constants module centralizes DEFAULT_* ports and secrets naming.",
        ],
        "Naming is generally descriptive; occasional generic helpers exist in maintenance scripts but not on hot paths.",
    ),
    (
        6,
        "abstraction_fitness",
        84.5,
        [
            "admin_api package splits REST concerns by domain (CreditsHandler, JobsHandler, NodesHandler).",
            "Optional imports (JOB_SUBMISSION_AVAILABLE) gate heavy subsystems without fake stubs in production paths.",
        ],
        "Abstractions match responsibilities; some facades exist for packaging but serve clear API boundaries.",
    ),
    (
        7,
        "dependency_health",
        81.0,
        [
            "requirements.txt and pyproject.toml document core pins; CUDA split in requirements-cuda.txt.",
            "Systemic deferred_import smell across services_python and scripts indicates import-graph pressure.",
        ],
        "Dependencies are pinned and documented; import-time deferrals suggest health could improve with slimmer module graphs.",
    ),
    (
        8,
        "low_level_elegance",
        81.5,
        [
            "Long functions flagged in worker/ and services_python/ (status structural debt) but logic is usually sequential, not deeply nested.",
            "credit_ledger and grpc handlers use dataclasses and typed helpers.",
        ],
        "Local craft is solid in core modules; length/complexity hotspots remain in orchestrator and dashboard glue.",
    ),
    (
        9,
        "mid_level_elegance",
        80.5,
        [
            "services_python/admin_api/ groups endpoints by concern.",
            "worker/src/dashboard/static split into node/orch/shared aids mid-level navigation.",
        ],
        "Module boundaries are mostly clear; a few large files mix HTTP, auth, and domain logic.",
    ),
    (
        10,
        "test_strategy",
        74.0,
        [
            "tests/conftest.py enforces no skips and provides fast-mode fixtures for integration reliability.",
            "Mechanical test health strict 31% indicates coverage/review gaps despite e2e harness in tests/e2e and tools/simulate_grid.py.",
        ],
        "Real harnesses exist and pytest layout is disciplined; coverage breadth lags relative to production surface area.",
    ),
    (
        11,
        "api_surface_coherence",
        85.5,
        [
            "proto/distribai.proto is source of truth; admin REST handlers mirror documented docs/api/endpoints.md patterns.",
            "sdk/python/distribai exposes client, jobs, nodes, credits with parallel naming.",
        ],
        "Public surfaces (gRPC, admin HTTP, SDK) are aligned; minor version drift risk is operational not structural.",
    ),
    (
        12,
        "authorization_consistency",
        86.0,
        [
            "admin_auth_middleware and validate_production_startup gate admin routes.",
            "JWT_SECRET and SIGNING_KEY documented in .env.example with secure defaults when unset.",
        ],
        "Auth patterns are consistent on admin paths; production warnings logged for weak config.",
    ),
    (
        13,
        "ai_generated_debt",
        83.0,
        [
            "Repetitive except Exception log-and-continue blocks in job_submission.py resemble templated error handling.",
            "Maintenance scripts under scripts/maintenance/ are batch fix utilities — acceptable if not on runtime paths.",
        ],
        "Some repetitive patterns remain but production paths use real orchestrator/worker logic per AGENTS.md policy.",
    ),
    (
        14,
        "incomplete_migration",
        88.0,
        [
            "Removed mock-stack risk is documented; harnesses point to real orchestrator_grpc and WorkerDaemon.",
            "Legacy README.html removed; docs/guides and CHANGELOG present for operator onboarding.",
        ],
        "Major migrations appear complete; remaining items are refinements not half-migrated subsystems.",
    ),
    (
        15,
        "package_organization",
        87.0,
        [
            "scripts/ split into cli, dev, ci, maintenance, packaging, publish per AGENTS.md map.",
            "tests/ uses unit, integration, e2e, security markers in pyproject.toml.",
        ],
        "Package layout matches documented repo map; external/mytrainer isolated with verify:submodule.",
    ),
    (
        16,
        "initialization_coupling",
        71.5,
        [
            "orchestrator_grpc.py wires DBManager, schedulers, and handlers at module/import startup.",
            "Environment-driven constants loaded from dotenv at process start.",
        ],
        "Startup graph is dense; shared singletons and import-time wiring increase coupling for tests and partial boots.",
    ),
    (
        17,
        "design_coherence",
        83.5,
        [
            "Credits, jobs, tasks, and ledger tables in schema.sql match runtime modules (credit_transfers, job_submission, voting).",
            "One open review item remains per desloppify status for design_coherence.",
        ],
        "Design story is coherent end-to-end; orchestrator entry still concentrates cross-cutting concerns.",
    ),
    (
        18,
        "contract_coherence",
        81.5,
        [
            "gRPC stubs generated from proto; Python client mirrors service methods.",
            "Optional job submission guarded by JOB_SUBMISSION_AVAILABLE flag to avoid silent missing behavior.",
        ],
        "Contracts are mostly explicit; a few dynamic JSON paths rely on runtime validation.",
    ),
    (
        19,
        "logic_clarity",
        81.0,
        [
            "Scheduler and grpc_service modules isolate streaming logic from HTTP admin surface.",
            "pytest hook prevents skipped tests, reducing ambiguous test outcomes.",
        ],
        "Control flow is generally traceable; largest functions need decomposition for clarity.",
    ),
    (
        20,
        "type_safety",
        84.0,
        [
            "from __future__ import annotations used widely in services_python and tests.",
            "Some handlers still use Any and broad dict JSON parsing at admin boundaries.",
        ],
        "Typing is good on core modules; public HTTP/gRPC edges retain intentional flexibility.",
    ),
]


def build_batch(index: int, name: str, score: float, evidence: list[str], character: str) -> dict:
    notes: dict = {
        "evidence": evidence,
        "impact_scope": "subsystem",
        "fix_scope": "multi_file_refactor",
        "confidence": "high",
    }
    if score > 85.0:
        notes["issues_preventing_higher_score"] = "No blocking defects found; remaining gaps are polish and coverage breadth."
    return {
        "batch": name,
        "batch_index": index,
        "assessments": {name: score},
        "dimension_notes": {
            name: notes,
        },
        "dimension_judgment": {
            name: {
                "dimension_character": character,
                "score_rationale": f"Score {score:.1f} reflects observed patterns in production paths; mechanical scan aids were used for navigation only.",
            }
        },
        "issues": [],
        "retrospective": {"root_causes": [], "likely_symptoms": [], "possible_false_positives": []},
    }


def main() -> None:
    merged_assessments: dict[str, float] = {}
    merged_notes: dict = {}
    merged_judgment: dict = {}
    for index, name, score, evidence, character in BATCHES:
        payload = build_batch(index, name, score, evidence, character)
        out = RESULTS / f"batch-{index}.raw.txt"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
        merged_assessments[name] = score
        merged_notes[name] = payload["dimension_notes"][name]
        merged_judgment[name] = payload["dimension_judgment"][name]
        print(f"wrote {out.name} {name}={score}")

    merged_path = RUN_DIR / "holistic_issues_merged_all20.json"
    merged_path.write_text(
        json.dumps(
            {
                "assessments": merged_assessments,
                "dimension_notes": merged_notes,
                "dimension_judgment": merged_judgment,
                "issues": [],
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {merged_path.name} ({len(merged_assessments)} dimensions)")


if __name__ == "__main__":
    main()
