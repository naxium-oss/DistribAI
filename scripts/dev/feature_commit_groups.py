"""Split repository changes into logical feature commits (paths are repo-relative)."""

from __future__ import annotations

COMMIT_GROUPS: list[tuple[str, list[str]]] = [
    (
        "chore: gitignore, env template, and tracked artifact cleanup",
        [
            ".gitignore",
            ".env.example",
            ".env.template",
            ".coverage",
            "README.html",
            ".claude/skills/desloppify/",
            "subjective_assessment.json",
            "runtime/db/distribai.db",
        ],
    ),
    (
        "refactor: reorganize scripts into ci/cli/dev/maintenance/packaging/publish",
        [
            "scripts/ci/",
            "scripts/cli/",
            "scripts/dev/",
            "scripts/maintenance/",
            "scripts/packaging/",
            "scripts/publish/",
            "scripts/run_pytest_phase.cjs",
            "scripts/distribai_cli.py",
            "scripts/preview_gui.py",
            "scripts/preview_orchestrator.py",
            "scripts/submit_job.py",
            "scripts/bundle.py",
            "scripts/publish_public_grid.py",
        ],
    ),
    (
        "ci: GitHub workflows and production gate scripts",
        [
            ".github/workflows/",
        ],
    ),
    (
        "feat(security): admin authentication and registration policy",
        [
            "services_python/admin_auth.py",
            "services_python/registration_policy.py",
            "tests/unit/test_admin_auth.py",
            "tests/unit/test_registration_policy.py",
            "tests/unit/test_v1_registration_gate.py",
            "tests/integration/test_admin_auth_integration.py",
            "tests/integration/test_poc_registration_flow.py",
            "tests/integration/test_registration_lockdown_integration.py",
            "tests/unit/test_grpc_register_jwt_poc.py",
            "tests/unit/test_grpc_registration_gate.py",
        ],
    ),
    (
        "feat(security): blob URL allowlist policy",
        [
            "services_python/blob_url_policy.py",
            "tests/unit/test_blob_url_policy.py",
            "tests/unit/test_blob_url_policy_fuzz.py",
            "tests/unit/test_worker_blob_allowlist.py",
            "tests/unit/test_grpc_gradient_allowlist.py",
        ],
    ),
    (
        "feat(security): CORS policy",
        [
            "services_python/cors_policy.py",
            "tests/unit/test_cors_policy.py",
        ],
    ),
    (
        "feat(security): gRPC TLS support",
        [
            "services_python/grpc_tls.py",
            "tests/unit/test_grpc_tls.py",
        ],
    ),
    (
        "feat(security): script validation and sandbox serialization",
        [
            "services_python/script_validation.py",
            "worker/src/sandbox/serialization.py",
            "tests/unit/test_script_validation.py",
            "tests/unit/test_script_importers.py",
            "tests/integration/test_script_runner_package.py",
        ],
    ),
    (
        "feat(security): SSE limits and rate protection",
        [
            "services_python/sse_limits.py",
            "tests/unit/test_sse_limits.py",
        ],
    ),
    (
        "feat(orchestrator): preflight, failure codes, and queue diagnostics",
        [
            "services_python/preflight.py",
            "services_python/job_failure_codes.py",
            "services_python/queue_diagnostics.py",
            "tests/unit/test_preflight_and_failure_codes.py",
            "tests/unit/test_queue_diagnostics.py",
        ],
    ),
    (
        "feat(orchestrator): memory manager, platform utils, bundle store, auto-update",
        [
            "services_python/memory_manager.py",
            "services_python/platform_utils.py",
            "services_python/bundle_store.py",
            "services_python/auto_update.py",
            "tests/unit/test_memory_manager.py",
            "tests/unit/test_platform_utils.py",
            "tests/unit/test_bundle_store.py",
            "tests/unit/test_auto_update_service.py",
        ],
    ),
    (
        "feat(admin-api): admin HTTP API routes and registry",
        [
            "services_python/admin_api/",
            "scripts/ci/admin_route_manifest.txt",
            "tests/unit/test_admin_api_credits.py",
            "tests/unit/test_admin_api_jobs.py",
            "tests/unit/test_admin_api_nodes.py",
            "tests/unit/test_admin_route_registry.py",
            "tests/unit/test_admin_health_job_submission.py",
        ],
    ),
    (
        "feat(orchestrator): core service layer updates",
        [
            "services_python/__init__.py",
            "services_python/admin_keys.py",
            "services_python/constants.py",
            "services_python/credit_multipliers.py",
            "services_python/credit_transfers.py",
            "services_python/database.py",
            "services_python/db_manager.py",
            "services_python/dependency_checker.py",
            "services_python/distributed_trainer.py",
            "services_python/grpc_service.py",
            "services_python/job_submission.py",
            "services_python/migrations.py",
            "services_python/monitoring.py",
            "services_python/mytrainer_sync.py",
            "services_python/oauth_provider.py",
            "services_python/orchestrator_grpc.py",
            "services_python/pagination.py",
            "services_python/poc_challenge.py",
            "services_python/rate_limiter.py",
            "services_python/rebenchmark_triggers.py",
            "services_python/scheduler.py",
            "services_python/schemas.py",
            "services_python/server_gui.py",
            "services_python/sybil_detector.py",
        ],
    ),
    (
        "feat(worker): daemon, executor, benchmarks, and OOM guard",
        [
            "worker/",
        ],
    ),
    (
        "feat(dashboard): operator static pages, shared assets, and maintenance scripts",
        [
            "worker/src/dashboard/",
            "scripts/maintenance/dashboard_timeout_patches.py",
            "scripts/maintenance/dashboard_xss_patches.py",
            "scripts/maintenance/node_index_timeout_patches.py",
            "scripts/maintenance/orchestrator_html_tail_patches.py",
            "scripts/maintenance/orchestrator_html_tail_patches_supplement.py",
            "scripts/maintenance/wire_node_operator_banner.py",
            "tests/unit/test_dashboard_static_audit.py",
            "tests/unit/test_operator_status.py",
        ],
    ),
    (
        "feat(client): Node dashboard and orchestrator Express servers",
        [
            "client/",
        ],
    ),
    (
        "feat(sdk): Python client package updates",
        [
            "sdk/python/",
        ],
    ),
    (
        "feat(proto): gRPC proto definitions and generated stubs",
        [
            "proto/",
        ],
    ),
    (
        "build: Python and Node dependency pins",
        [
            "pyproject.toml",
            "requirements.txt",
            "requirements-cuda.txt",
            "setup.py",
            "package.json",
            "package-lock.json",
            "playwright.config.js",
        ],
    ),
    (
        "build: PyInstaller bundle entrypoint",
        [
            "build.py",
        ],
    ),
    (
        "docs: README, AGENTS, guides, runbooks, and API reference",
        [
            "README.md",
            "AGENTS.md",
            "docs/",
        ],
    ),
    (
        "docs: CHANGELOG, TODO backlog, and scorecard",
        [
            "CHANGELOG.md",
            "TODO.md",
            "docs/assets/scorecard.png",
            "docs/assets/scorecard-secondary.png",
        ],
    ),
    (
        "examples: job recipes and golden template",
        [
            "examples/",
        ],
    ),
    (
        "test: fast test harness and unit test suite",
        [
            "tests/fast_env.py",
            "tests/conftest.py",
            "tests/unit/",
        ],
    ),
    (
        "test: integration, e2e, chaos, and performance suites",
        [
            "tests/integration/",
            "tests/e2e/",
            "tests/chaos/",
            "tests/performance/",
        ],
    ),
    (
        "test: security, OWASP, and Playwright UI coverage",
        [
            "tests/security/",
            "tests/playwright/",
        ],
    ),
    (
        "chore: specs, tools, and runtime schema",
        [
            "specs/",
            "tools/",
            "runtime/db/schema.sql",
        ],
    ),
    (
        "chore: cursor agents, rules, and review tooling",
        [
            ".cursor/",
        ],
    ),
]

# Never stage these paths (local/runtime noise).
EXCLUDE_PREFIXES = (
    "node_modules/",
    "dist/",
    "test-results/",
    "runtime/smoke/",
    "runtime/gui-test-node/",
    "runtime/triage_",
    "runtime/pen-orch.err",
    "bandit-report.json",
    ".env",
    "~/",
    ".desloppify/",
)
