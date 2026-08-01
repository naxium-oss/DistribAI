"""Extended tests for PoC challenge module."""

import pytest


def test_poc_challenge_manager_creation():
    """Test PoCChallengeManager creation."""
    try:
        from services_python.poc_challenge import PoCChallengeManager
    except ImportError:
        pytest.skip("PoCChallengeManager not available")
        return

    poc = PoCChallengeManager(difficulty=4)
    assert poc.difficulty == 4


def test_poc_challenge_generate():
    """Test challenge generation."""
    try:
        from services_python.poc_challenge import PoCChallengeManager
    except ImportError:
        pytest.skip("PoCChallengeManager not available")
        return

    poc = PoCChallengeManager(difficulty=2)
    challenge = poc.generate_challenge("node-1")

    assert challenge is not None
    assert hasattr(challenge, "challenge") or isinstance(challenge, (str, bytes))


def test_poc_challenge_solve():
    """Test challenge solving."""
    try:
        from services_python.poc_challenge import PoCChallengeManager
    except ImportError:
        pytest.skip("PoCChallengeManager not available")
        return

    poc = PoCChallengeManager(difficulty=1)
    challenge_obj = poc.generate_challenge("node-1")

    # Extract challenge string
    challenge = (
        challenge_obj.challenge if hasattr(challenge_obj, "challenge") else str(challenge_obj)
    )

    # Solve the challenge
    nonce = poc.solve_challenge(challenge, max_attempts=1000)
    assert nonce is not None


def test_poc_challenge_verify():
    """Test challenge verification."""
    try:
        from services_python.poc_challenge import PoCChallengeManager
    except ImportError:
        pytest.skip("PoCChallengeManager not available")
        return

    poc = PoCChallengeManager(difficulty=1)
    challenge_obj = poc.generate_challenge("node-1")

    # Extract challenge string
    challenge = (
        challenge_obj.challenge if hasattr(challenge_obj, "challenge") else str(challenge_obj)
    )

    # Solve and verify
    nonce = poc.solve_challenge(challenge, max_attempts=1000)
    is_valid = poc.verify_challenge("node-1", challenge, nonce)
    assert is_valid is True


def test_poc_challenge_difficulty_levels():
    """Test different difficulty levels."""
    try:
        from services_python.poc_challenge import PoCChallengeManager
    except ImportError:
        pytest.skip("PoCChallengeManager not available")
        return

    for difficulty in [1, 2, 3]:
        poc = PoCChallengeManager(difficulty=difficulty)
        assert poc.difficulty == difficulty

        challenge = poc.generate_challenge("node-1")
        assert challenge is not None
