"""
Unit tests for voting system
"""

import pytest


def test_voting_system_import():
    from worker.src.daemon.voting_system import Vote, VoteType, VotingSystem

    assert VotingSystem is not None
    assert Vote is not None
    assert VoteType is not None


def test_voting_system_creation():
    from worker.src.daemon.voting_system import VotingSystem

    system = VotingSystem(signing_key=b"test-key-123")
    assert system is not None
    assert system.credit_price_per_vote == 10


def test_create_account():
    from worker.src.daemon.voting_system import VotingSystem

    system = VotingSystem(signing_key=b"test-key-123")
    account = system.create_account("user123")
    assert account.contributor_id == "user123"
    assert account.balance == 0.0
    assert account.credits_earned == 0.0


def test_add_credits():
    from worker.src.daemon.voting_system import VotingSystem

    system = VotingSystem(signing_key=b"test-key-123")
    account = system.create_account("user123")
    result = system.add_credits("user123", 100.0, job_id="job1")
    assert result is True
    assert account.balance == 100.0
    assert account.credits_earned == 100.0


def test_spend_credits():
    from worker.src.daemon.voting_system import VotingSystem

    system = VotingSystem(signing_key=b"test-key-123")
    system.create_account("user123")
    system.add_credits("user123", 100.0)
    result = system.spend_credits("user123", 30.0, reason="vote")
    assert result is True
    account = system.accounts["user123"]
    assert account.balance == 70.0
    assert account.credits_spent == 30.0


def test_spend_insufficient_credits():
    from worker.src.daemon.voting_system import VotingSystem

    system = VotingSystem(signing_key=b"test-key-123")
    system.create_account("user123")
    system.add_credits("user123", 10.0)
    result = system.spend_credits("user123", 30.0)
    assert result is False


def test_create_vote():
    from worker.src.daemon.voting_system import VoteType, VotingSystem

    system = VotingSystem(signing_key=b"test-key-123")
    system.create_account("user123")
    system.add_credits("user123", 200.0)
    vote_id = system.create_vote(
        proposer="user123",
        vote_type=VoteType.JOB_PRIORITY,
        title="Test Vote",
        description="This is a test vote",
        options=["option1", "option2"],
        credits_required=100,
    )
    assert vote_id in system.votes
    vote = system.votes[vote_id]
    assert vote.title == "Test Vote"
    assert vote.proposer == "user123"


def test_create_vote_insufficient_credits():
    from worker.src.daemon.voting_system import VoteType, VotingSystem

    system = VotingSystem(signing_key=b"test-key-123")
    system.create_account("user123")
    system.add_credits("user123", 50.0)
    with pytest.raises(ValueError):
        system.create_vote(
            proposer="user123",
            vote_type=VoteType.JOB_PRIORITY,
            title="Test Vote",
            description="This is a test vote",
            options=["option1"],
            credits_required=100,
        )


def test_cast_vote():
    from worker.src.daemon.voting_system import VoteType, VotingSystem

    system = VotingSystem(signing_key=b"test-key-123")
    system.create_account("proposer")
    system.add_credits("proposer", 200.0)
    system.create_account("voter1")
    system.add_credits("voter1", 100.0)
    vote_id = system.create_vote(
        proposer="proposer",
        vote_type=VoteType.JOB_PRIORITY,
        title="Test Vote",
        description="Test",
        options=["yes", "no"],
        credits_required=100,
    )
    result = system.cast_vote("voter1", vote_id, "yes")
    assert result is True
    vote = system.votes[vote_id]
    assert "voter1" in vote.votes_cast
    assert vote.votes_cast["voter1"] == "yes"


def test_quorum_config():
    from worker.src.daemon.voting_system import VoteType, VotingSystem

    system = VotingSystem(signing_key=b"test-key-123")
    for vote_type in VoteType:
        assert vote_type in system.quorum_configs
        config = system.quorum_configs[vote_type]
        assert config.min_participation > 0
        assert config.quorum_required > 0


def test_credit_account_delegation():
    from worker.src.daemon.voting_system import VotingSystem

    system = VotingSystem(signing_key=b"test-key-123")
    account = system.create_account("user123")
    assert account.delegated_to is None
    assert account.delegates == set()
    account.delegated_to = "delegate456"
    assert account.delegated_to == "delegate456"
