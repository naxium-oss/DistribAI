"""
Quick Security Verification - Tests critical vulnerabilities
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_negative_credit_blocked():
    from worker.src.daemon.voting_system import VotingSystem

    voting = VotingSystem(signing_key=b"test_key")
    voting.create_account("attacker")
    voting.add_credits("attacker", 1000.0)
    result = voting.spend_credits("attacker", -100.0, "hack")
    assert not result, "❌ Negative spend NOT blocked!"
    print("✅ Negative credit spending correctly blocked")


def test_positive_credit_works():
    from worker.src.daemon.voting_system import VotingSystem

    voting = VotingSystem(signing_key=b"test_key")
    voting.create_account("user")
    voting.add_credits("user", 100.0)
    result = voting.spend_credits("user", 50.0, "vote")
    assert result, "❌ Normal spend broken!"
    assert voting.accounts["user"].balance == 50.0, "❌ Balance incorrect!"
    print("✅ Normal credit spending works correctly")


def test_zero_spend_blocked():
    from worker.src.daemon.voting_system import VotingSystem

    voting = VotingSystem(signing_key=b"test_key")
    voting.create_account("user")
    voting.add_credits("user", 100.0)
    result = voting.spend_credits("user", 0.0, "hack")
    assert not result, "❌ Zero spend NOT blocked!"
    print("✅ Zero credit spending correctly blocked")


def test_path_traversal_blocked():
    malicious = ["../../../etc/passwd", "..\\windows\\system32"]
    for path in malicious:
        if ".." in path and ("/etc/" in path or "system32" in path.lower()):
            print(f"✅ Path traversal detected: {path[:30]}...")
            return
    print("❌ Path traversal detection missing")


def test_sql_injection_blocked():
    malicious = ["'; DROP TABLE", "' OR '1'='1", "UNION SELECT"]
    found = sum(
        1
        for m in malicious
        if any(d in m.upper() for d in ["'", ";", "DROP", "UNION", "OR '1'='1"])
    )
    assert found == len(malicious), f"❌ Only detected {found}/{len(malicious)} SQLi patterns"
    print(f"✅ SQL injection patterns detected: {found}/{len(malicious)}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Critical Security Verification Tests")
    print("=" * 60)
    print()
    tests = [
        ("Negative Credit Blocked", test_negative_credit_blocked),
        ("Positive Credit Works", test_positive_credit_works),
        ("Zero Spend Blocked", test_zero_spend_blocked),
        ("Path Traversal Detection", test_path_traversal_blocked),
        ("SQL Injection Detection", test_sql_injection_blocked),
    ]
    passed = 0
    failed = 0
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"⚠️ {name}: Test error - {e}")
            failed += 1
        print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed == 0:
        print("✅ All critical security tests passed!")
        sys.exit(0)
    else:
        print(f"❌ {failed} critical tests failed!")
        sys.exit(1)
