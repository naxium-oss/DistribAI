"""Regression tests for DistribAI v1.1 security hardening.

Covers the two security commits that landed on hardening/v1.1-tests:

* 5714956 sec(admin)     -- admin JWT required on /admin/*; JWT_SECRET +
                            SIGNING_KEY persisted to disk; mint_admin_jwt CLI;
                            bootstrap admin JWT written on first start.
* b3b110c sec(grpc,v1)   -- gRPC RegisterSession validates jwt_token; the
                            simple /v1/nodes/register path returns 403 unless
                            DISTRIBAI_ALLOW_INSECURE_REGISTER=1.

Test sections (correspond to A-F in the task brief):

  A. Secret persistence (services_python/constants.py)
  B. Admin JWT auth on /admin/jobs, /admin/credits, /admin/votes, /admin/nodes
  C. Missing-json regression -- malformed JSON returns 400, not 500 NameError
  D. gRPC RegisterSession JWT enforcement
  E. /v1/nodes/register insecure-by-default disable
  F. scripts/mint_admin_jwt.py CLI smoke
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import jwt
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = str(REPO_ROOT / "runtime" / "db" / "schema.sql")
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Protobuf availability probe.
#
# gencode 6.31.1 in the worktree wants protobuf runtime >=6 and grpcio>=1.80,
# but a stale env may still have 5.x. If imports fail, skip section D rather
# than producing a misleading regression.
# ---------------------------------------------------------------------------

try:
    from worker.src.distribai_proto import distribai_pb2  # type: ignore

    PROTOBUF_AVAILABLE = True
    PROTOBUF_ERROR: str | None = None
except Exception as _exc:  # pragma: no cover - environmental
    PROTOBUF_AVAILABLE = False
    PROTOBUF_ERROR = str(_exc)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fresh_secrets_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point DISTRIBAI_SECRETS_DIR at a clean tmpdir and clear inline overrides.

    Returns the tmpdir path (already a Path)."""
    monkeypatch.setenv("DISTRIBAI_SECRETS_DIR", str(tmp_path))
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("SIGNING_KEY", raising=False)
    monkeypatch.delenv("DISTRIBAI_DISABLE_SECRET_PERSISTENCE", raising=False)
    return tmp_path


def _reload_constants():
    """Reimport services_python.constants under current env state."""
    import importlib

    import services_python.constants as _const

    importlib.reload(_const)
    return _const


# ---------------------------------------------------------------------------
# SECTION A -- secret persistence
# ---------------------------------------------------------------------------


def _run_subprocess_get_secret(secrets_dir: Path, env_extra: dict | None = None) -> str:
    """Run a clean subprocess that imports constants and prints JWT_SECRET."""
    env = {**os.environ, "DISTRIBAI_SECRETS_DIR": str(secrets_dir)}
    # Wipe any inline JWT_SECRET / SIGNING_KEY that may be set in the parent
    # so the child genuinely exercises the disk-read path.
    env.pop("JWT_SECRET", None)
    env.pop("SIGNING_KEY", None)
    env.pop("DISTRIBAI_DISABLE_SECRET_PERSISTENCE", None)
    if env_extra:
        env.update(env_extra)
    cp = subprocess.run(
        [
            PYTHON,
            "-c",
            "from services_python.constants import _get_or_generate_secret; "
            "print(_get_or_generate_secret('JWT_SECRET'))",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert cp.returncode == 0, f"subprocess failed: {cp.stderr}"
    return cp.stdout.strip().splitlines()[-1]


def test_secret_persists_across_subprocess_invocations(tmp_path: Path):
    """Two cold processes with the same DISTRIBAI_SECRETS_DIR see the same JWT_SECRET."""
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()

    first = _run_subprocess_get_secret(secrets_dir)
    second = _run_subprocess_get_secret(secrets_dir)

    assert first == second, "JWT_SECRET should persist across process restarts"
    assert (secrets_dir / "JWT_SECRET").exists(), "secret should be written to disk"


def test_env_var_overrides_disk(tmp_path: Path):
    """An explicit JWT_SECRET env var beats whatever is on disk."""
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    # Prime disk with one value.
    on_disk = _run_subprocess_get_secret(secrets_dir)
    # Now ask for it again with an explicit override; should return override.
    overridden = _run_subprocess_get_secret(secrets_dir, env_extra={"JWT_SECRET": "env-wins-12345"})
    assert overridden == "env-wins-12345"
    assert overridden != on_disk


def test_disable_secret_persistence_returns_fresh_each_time(tmp_path: Path):
    """DISTRIBAI_DISABLE_SECRET_PERSISTENCE=1 -> ephemeral secret, no file write."""
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    extra = {"DISTRIBAI_DISABLE_SECRET_PERSISTENCE": "1"}
    first = _run_subprocess_get_secret(secrets_dir, env_extra=extra)
    second = _run_subprocess_get_secret(secrets_dir, env_extra=extra)
    assert first != second, "with persistence disabled, each call must be fresh"
    assert not (secrets_dir / "JWT_SECRET").exists(), "no file should be written"


def test_secret_file_mode_is_0600(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The persisted secret file must be readable only by its owner (POSIX)."""
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    _fresh_secrets_env(monkeypatch, secrets_dir)

    const = _reload_constants()
    const._get_or_generate_secret("JWT_SECRET")

    secret_file = secrets_dir / "JWT_SECRET"
    assert secret_file.exists()
    if sys.platform == "win32":
        # Windows ACLs differ from POSIX mode bits; chmod is best-effort there.
        return
    mode = secret_file.stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


# ---------------------------------------------------------------------------
# SECTION B + C -- admin auth + missing-json regression
# ---------------------------------------------------------------------------


class _StubNodeService:
    """Minimal NodeService stub: real JWT logic, mock subsystems.

    We intentionally avoid instantiating the real NodeService because it spins
    up a scheduler task and a number of stateful subsystems that are unrelated
    to the auth gates we are regression-testing.
    """

    def __init__(self, db):
        # Lazy import keeps this file importable even if heavy deps fail later.
        from services_python.constants import (
            JWT_ALGORITHM,
            JWT_SECRET,
        )

        self.db = db
        self.connected_nodes: dict = {}
        self._jwt_secret = JWT_SECRET
        self._jwt_alg = JWT_ALGORITHM
        # Mocked subsystems used by some handler code paths.
        self.credit_multipliers = MagicMock()

    def _issue_jwt(self, subject: str, kind: str = "node", expires_in: int = 3600) -> str:
        now = int(time.time())
        return jwt.encode(
            {"sub": subject, "kind": kind, "iat": now, "exp": now + expires_in},
            self._jwt_secret,
            algorithm=self._jwt_alg,
        )

    def verify_jwt(self, token, expected_subject=None, kind=None):
        if not token:
            return None
        try:
            claims = jwt.decode(token, self._jwt_secret, algorithms=[self._jwt_alg])
        except jwt.InvalidTokenError:
            return None
        if expected_subject and claims.get("sub") != expected_subject:
            return None
        if kind and claims.get("kind") != kind:
            return None
        return claims

    def _authenticate_request(self, req, required_kind=None):
        auth = req.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise web.HTTPUnauthorized(
                text=json.dumps({"error": "missing bearer token"}),
                content_type="application/json",
            )
        token = auth.split(" ", 1)[1]
        claims = self.verify_jwt(token, kind=required_kind)
        if not claims:
            raise web.HTTPUnauthorized(
                text=json.dumps({"error": "invalid token"}),
                content_type="application/json",
            )
        return claims


@pytest.fixture
def secrets_dir(tmp_path, monkeypatch):
    """A per-test secrets dir + reloaded constants module."""
    d = tmp_path / "secrets"
    d.mkdir()
    _fresh_secrets_env(monkeypatch, d)
    _reload_constants()
    # Also reload modules that captured JWT_SECRET at import time so they pick
    # up the freshly generated value for this test.
    import importlib

    import services_python.orchestrator_grpc as _og  # noqa: F401

    importlib.reload(_og)
    return d


@pytest.fixture
def db(tmp_path):
    from services_python.db_manager import DBManager

    db_path = str(tmp_path / "test.db")
    return DBManager(db_path, SCHEMA_PATH)


@pytest.fixture
def node_service(secrets_dir, db):
    return _StubNodeService(db)


def _build_admin_app(node_service):
    """Wire only the admin routes we are regression-testing.

    Deliberately omits the rate_limit_middleware (irrelevant) and only
    constructs the handlers that the spec calls out, so the test stays
    self-contained.
    """
    from services_python.admin_api.credits import CreditsHandler
    from services_python.admin_api.jobs import JobsHandler
    from services_python.admin_api.nodes import NodesHandler
    from services_python.admin_api.votes import VotesHandler

    jobs_h = JobsHandler(node_service.db, node_service)
    nodes_h = NodesHandler(node_service.db, node_service)
    credits_h = CreditsHandler(
        node_service.db,
        MagicMock(),  # credit_ledger
        MagicMock(),  # credit_transfers
        node_service,
    )
    voting_system_mock = MagicMock()
    voting_system_mock.get_active_votes.return_value = []
    voting_system_mock.get_vote_status.return_value = None
    voting_system_mock.create_vote.return_value = "fake-vote-id"
    voting_system_mock.cast_vote.return_value = True
    votes_h = VotesHandler(node_service.db, voting_system_mock, node_service)

    app = web.Application()
    app.router.add_get("/admin/jobs", jobs_h.list)
    app.router.add_post("/admin/jobs", jobs_h.create)
    app.router.add_delete("/admin/jobs/{job_id}", jobs_h.cancel)

    app.router.add_get("/admin/credits", credits_h.list)
    app.router.add_get("/admin/credits/{node_id}", credits_h.get)

    app.router.add_get("/admin/votes", votes_h.list)
    app.router.add_post("/admin/votes", votes_h.create)
    app.router.add_post("/admin/votes/{vote_id}/cast", votes_h.cast)

    app.router.add_get("/admin/nodes", nodes_h.list)
    app.router.add_post("/admin/nodes/{node_id}/contributing", nodes_h.set_contributing)

    return app


def _admin_h(node_service) -> dict:
    return {"Authorization": f"Bearer {node_service._issue_jwt('admin-test', kind='admin')}"}


def _node_h(node_service) -> dict:
    return {"Authorization": f"Bearer {node_service._issue_jwt('node-test', kind='node')}"}


# Each row: (method, path, body) for endpoints that require admin auth.
ADMIN_ENDPOINTS = [
    ("GET", "/admin/jobs", None),
    ("POST", "/admin/jobs", {"base_model": "distribai-small"}),
    ("DELETE", "/admin/jobs/some-id", None),
    ("GET", "/admin/credits", None),
    ("GET", "/admin/credits/some-node", None),
    ("GET", "/admin/votes", None),
    (
        "POST",
        "/admin/votes",
        {
            "proposer": "p",
            "title": "t",
            "vote_type": "job_priority",
            # Note: VotesHandler.create internally rejects with 400 due to
            # schema validation -- that's fine, it proves auth passed.
            "job_id": "abcd",
            "credits": 1,
        },
    ),
    ("POST", "/admin/votes/vote123/cast", {"voter_id": "v", "option": "yes"}),
    ("GET", "/admin/nodes", None),
    ("POST", "/admin/nodes/some-node/contributing", {"contributing": False}),
]


async def _request(client, method, path, headers=None, json_body=None, raw_body=None):
    kwargs = {"headers": headers or {}}
    if json_body is not None:
        kwargs["json"] = json_body
    elif raw_body is not None:
        kwargs["data"] = raw_body
        kwargs.setdefault("headers", {})["Content-Type"] = "application/json"

    method = method.upper()
    if method == "GET":
        return await client.get(path, **kwargs)
    if method == "POST":
        return await client.post(path, **kwargs)
    if method == "DELETE":
        return await client.delete(path, **kwargs)
    raise AssertionError(f"unsupported method {method}")


@pytest.mark.parametrize("method,path,body", ADMIN_ENDPOINTS)
async def test_admin_endpoint_rejects_missing_auth(node_service, method, path, body):
    app = _build_admin_app(node_service)
    async with TestClient(TestServer(app)) as client:
        resp = await _request(client, method, path, json_body=body)
        assert resp.status == 401, f"{method} {path} should be 401 without auth, got {resp.status}"


@pytest.mark.parametrize("method,path,body", ADMIN_ENDPOINTS)
async def test_admin_endpoint_rejects_node_kind(node_service, method, path, body):
    app = _build_admin_app(node_service)
    async with TestClient(TestServer(app)) as client:
        resp = await _request(client, method, path, headers=_node_h(node_service), json_body=body)
        assert resp.status == 401, (
            f"{method} {path} should reject node-kind token (got {resp.status})"
        )


@pytest.mark.parametrize("method,path,body", ADMIN_ENDPOINTS)
async def test_admin_endpoint_accepts_admin_kind(node_service, method, path, body):
    app = _build_admin_app(node_service)
    async with TestClient(TestServer(app)) as client:
        resp = await _request(client, method, path, headers=_admin_h(node_service), json_body=body)
        # The auth gate passed if we did NOT get 401 (other 4xx/2xx all imply
        # the handler body executed). We additionally assert NOT 500 to catch
        # any NameError-style regressions.
        assert resp.status != 401, f"{method} {path} returned 401 with admin token"
        assert resp.status < 500, (
            f"{method} {path} returned {resp.status} -- handler crashed with admin token"
        )


# Section C -- malformed JSON should yield 400, not 500 (regression for
# missing `import json` in handler modules).
MALFORMED_JSON_TARGETS = [
    ("POST", "/admin/jobs"),
    ("POST", "/admin/votes"),
    ("POST", "/admin/votes/vote123/cast"),
    ("POST", "/admin/nodes/some-node/contributing"),
]


@pytest.mark.parametrize("method,path", MALFORMED_JSON_TARGETS)
async def test_malformed_json_returns_400_not_500(node_service, method, path):
    app = _build_admin_app(node_service)
    async with TestClient(TestServer(app)) as client:
        resp = await _request(
            client,
            method,
            path,
            headers=_admin_h(node_service),
            raw_body=b"{this is not valid json",
        )
        assert resp.status == 400, (
            f"{method} {path} returned {resp.status} for malformed JSON; "
            "should be 400. A 500 here typically means a missing `import json`."
        )


# ---------------------------------------------------------------------------
# SECTION D -- gRPC RegisterSession JWT enforcement
# ---------------------------------------------------------------------------

pytestmark_grpc = pytest.mark.skipif(
    not PROTOBUF_AVAILABLE,
    reason=f"protobuf/grpcio version mismatch: {PROTOBUF_ERROR}",
)


@pytest.fixture
def grpc_handler(node_service):
    """Build a GrpcServiceHandler bound to our stub NodeService + real DB."""
    if not PROTOBUF_AVAILABLE:
        pytest.skip(f"protobuf unavailable: {PROTOBUF_ERROR}")
    from services_python.grpc_service import GrpcServiceHandler

    # Patch in a stand-in for the missing connected_nodes attr on the stub.
    node_service.connected_nodes = {}
    return GrpcServiceHandler(node_service)


def _make_register_msg(node_id: str, jwt_token: str = "") -> distribai_pb2.RegisterSession:
    return distribai_pb2.RegisterSession(
        node_id=node_id,
        jwt_token=jwt_token,
        hardware_json="{}",
        benchmark_json="",
    )


@pytestmark_grpc
async def test_grpc_register_empty_jwt_new_node_no_bootstrap_rejected(
    grpc_handler, db, node_service, monkeypatch
):
    monkeypatch.delenv("DISTRIBAI_GRPC_ALLOW_BOOTSTRAP", raising=False)
    q: asyncio.Queue = asyncio.Queue()
    ref = {"id": None}

    msg = _make_register_msg("brand-new-node-1", jwt_token="")
    await grpc_handler._handle_register(msg, q, ref)

    assert ref["id"] is None, "node should not be registered"
    assert q.empty(), "no ack should be enqueued"
    assert db.get_node_jwt("brand-new-node-1") is None, "no JWT should be persisted"


@pytestmark_grpc
async def test_grpc_register_empty_jwt_new_node_with_bootstrap_accepted(
    grpc_handler, db, monkeypatch
):
    """First-time bootstrap path: registration is accepted and a JWT issued.

    Note: there is a follow-up regression worth investigating -- on the
    bootstrap path the issued JWT is NOT actually persisted because
    db.update_node_hardware/update_node_jwt are UPDATE-only and the row
    has not been created. This test only asserts what _handle_register
    is documented to do (accept + issue + ack); the persistence gap is
    flagged separately.
    """
    monkeypatch.setenv("DISTRIBAI_GRPC_ALLOW_BOOTSTRAP", "1")
    q: asyncio.Queue = asyncio.Queue()
    ref = {"id": None}

    msg = _make_register_msg("bootstrap-node-2", jwt_token="")
    await grpc_handler._handle_register(msg, q, ref)

    assert ref["id"] == "bootstrap-node-2", "bootstrap-mode registration should succeed"
    assert not q.empty(), "ack should be enqueued"
    ack = await q.get()
    assert ack.register_ack.session_token, "fresh session_token should be in the ack"


@pytestmark_grpc
async def test_grpc_register_empty_jwt_existing_node_rejected(
    grpc_handler, db, node_service, monkeypatch
):
    # Seed the DB so the node already exists with a JWT.
    db.create_node(
        node_id="existing-node-3",
        jwt_token="prior-jwt",
        hardware_json="{}",
    )
    monkeypatch.delenv("DISTRIBAI_GRPC_ALLOW_BOOTSTRAP", raising=False)

    q: asyncio.Queue = asyncio.Queue()
    ref = {"id": None}

    msg = _make_register_msg("existing-node-3", jwt_token="")
    await grpc_handler._handle_register(msg, q, ref)

    assert ref["id"] is None, "existing node with empty JWT must be rejected"
    assert q.empty()


@pytestmark_grpc
async def test_grpc_register_bad_jwt_wrong_subject_rejected(
    grpc_handler, db, node_service, monkeypatch
):
    monkeypatch.delenv("DISTRIBAI_GRPC_ALLOW_BOOTSTRAP", raising=False)
    # JWT is for sub=other-node, but claim is for node_id=victim-node-4.
    bad_token = node_service._issue_jwt("other-node", kind="node")

    q: asyncio.Queue = asyncio.Queue()
    ref = {"id": None}

    msg = _make_register_msg("victim-node-4", jwt_token=bad_token)
    await grpc_handler._handle_register(msg, q, ref)

    assert ref["id"] is None, "JWT subject mismatch must be rejected"
    assert q.empty()


@pytestmark_grpc
async def test_grpc_register_good_jwt_accepted(grpc_handler, db, node_service):
    """Valid JWT (kind=node, sub matches node_id) is accepted and rotated.

    The DB row already exists (we pre-seed it because update_node_jwt is
    UPDATE-only). We then assert the ack contains a fresh session_token
    and that the persisted JWT was rotated to the new value.
    """
    # Seed an existing row so update_node_jwt has something to update.
    db.create_node(node_id="good-node-5", jwt_token="initial-seed", hardware_json="{}")
    good_token = node_service._issue_jwt("good-node-5", kind="node")

    q: asyncio.Queue = asyncio.Queue()
    ref = {"id": None}

    msg = _make_register_msg("good-node-5", jwt_token=good_token)
    await grpc_handler._handle_register(msg, q, ref)

    assert ref["id"] == "good-node-5"
    assert not q.empty()
    ack = await q.get()
    new_token = ack.register_ack.session_token
    assert new_token, "ack must contain a fresh session_token"
    # The newly issued session JWT should be persisted back to the DB,
    # replacing the seeded "initial-seed" value. (May happen to equal
    # `good_token` if iat falls on the same second; what matters is that
    # the seed value is gone.)
    persisted = db.get_node_jwt("good-node-5")
    assert persisted == new_token, "fresh session JWT must overwrite the seed"
    assert persisted != "initial-seed", "seeded JWT must have been replaced"


# ---------------------------------------------------------------------------
# SECTION E -- /v1/nodes/register insecure-by-default
# ---------------------------------------------------------------------------


def _v1_app(db, node_service):
    from services_python.admin_api.v1 import V1Handler

    handler = V1Handler(db, node_service)
    app = web.Application()
    app.router.add_post("/v1/nodes/register", handler.register_node)
    return app


async def test_v1_register_returns_403_by_default(db, node_service, monkeypatch):
    monkeypatch.delenv("DISTRIBAI_ALLOW_INSECURE_REGISTER", raising=False)
    app = _v1_app(db, node_service)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/v1/nodes/register",
            json={"node_id": "node-e1", "public_key": "k", "os": "linux"},
        )
        assert resp.status == 403
        body = await resp.json()
        assert body.get("error") == "insecure registration disabled"
        assert "hint" in body


async def test_v1_register_works_when_explicitly_enabled(db, node_service, monkeypatch):
    monkeypatch.setenv("DISTRIBAI_ALLOW_INSECURE_REGISTER", "1")
    app = _v1_app(db, node_service)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/v1/nodes/register",
            json={"node_id": "node-e2", "public_key": "k", "os": "linux"},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body.get("ok") is True
        assert body.get("token"), "JWT should be returned"


# ---------------------------------------------------------------------------
# SECTION F -- mint_admin_jwt.py CLI smoke
# ---------------------------------------------------------------------------


def test_mint_admin_jwt_cli(tmp_path: Path):
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    script = REPO_ROOT / "scripts" / "dev" / "mint_admin_jwt.py"
    assert script.exists()

    env = {**os.environ, "DISTRIBAI_SECRETS_DIR": str(secrets_dir)}
    env.pop("JWT_SECRET", None)
    env.pop("DISTRIBAI_DISABLE_SECRET_PERSISTENCE", None)

    cp = subprocess.run(
        [PYTHON, str(script), "--subject", "test", "--ttl-hours", "1", "--quiet"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert cp.returncode == 0, f"mint failed: {cp.stderr}"
    token = cp.stdout.strip().splitlines()[-1]
    assert token, "token must be printed"

    secret_path = secrets_dir / "JWT_SECRET"
    assert secret_path.exists(), "minting should have persisted JWT_SECRET to disk"
    secret = secret_path.read_text(encoding="utf-8").strip()

    claims = jwt.decode(token, secret, algorithms=["HS256"])
    assert claims["kind"] == "admin"
    assert claims["sub"] == "test"
    assert claims["exp"] - claims["iat"] == 3600, "TTL must be exactly 1 hour"
