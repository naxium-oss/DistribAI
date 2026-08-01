"""Extended tests for credit ledger module."""

import pytest


def test_credit_ledger_import():
    """Test credit ledger module imports."""
    from worker.src.daemon.credit_ledger import CreditEntry, CreditLedger, MerkleNode

    assert CreditLedger is not None
    assert CreditEntry is not None
    assert MerkleNode is not None


def test_merkle_node_creation():
    """Test MerkleNode dataclass creation."""
    import hashlib

    from worker.src.daemon.credit_ledger import MerkleNode

    node = MerkleNode(hash=hashlib.sha256(b"test").digest())
    assert node is not None
    assert node.left is None
    assert node.right is None
    assert len(node.hash) == 32  # SHA256 hash length


def test_credit_entry_creation():
    """Test CreditEntry dataclass creation."""
    from worker.src.daemon.credit_ledger import CreditEntry

    entry = CreditEntry(
        contributor_id="node-123",
        job_id="job-456",
        amount=100.0,
        timestamp=1234567890.0,
        metadata={"task": "training"},
    )
    assert entry.contributor_id == "node-123"
    assert entry.job_id == "job-456"
    assert entry.amount == 100.0
    assert entry.metadata == {"task": "training"}


def test_credit_ledger_initialization():
    """Test CreditLedger initialization."""
    from worker.src.daemon.credit_ledger import CreditLedger

    ledger = CreditLedger(signing_key=b"test-signing-key-32-bytes-long")
    assert ledger is not None
    assert len(ledger.records) == 0
    assert ledger.batch_size == 100


def test_credit_ledger_auto_signing_key(monkeypatch):
    """Test CreditLedger auto-generates signing key if None and env unset."""
    monkeypatch.delenv("SIGNING_KEY", raising=False)
    from worker.src.daemon.credit_ledger import CreditLedger

    ledger = CreditLedger(signing_key=None)
    assert ledger is not None
    assert ledger.signing_key is not None
    assert len(ledger.signing_key) > 0


def test_credit_ledger_uses_env_signing_key(monkeypatch):
    """Test CreditLedger reads SIGNING_KEY from environment when set."""
    monkeypatch.setenv("SIGNING_KEY", "shared-orchestrator-key")
    from worker.src.daemon.credit_ledger import CreditLedger

    ledger = CreditLedger(signing_key=None)
    assert ledger.signing_key == b"shared-orchestrator-key"


def test_credit_ledger_methods_exist():
    """Test that expected methods exist on CreditLedger."""
    from worker.src.daemon.credit_ledger import CreditLedger

    ledger = CreditLedger(signing_key=b"test-key-123")

    # Check expected methods
    expected_methods = [
        "add_credit",
        "get_balance",
        "get_signed_head",
        "verify_chain_integrity",
    ]

    for method in expected_methods:
        assert hasattr(ledger, method), f"Missing method: {method}"


def test_ledger_record_dataclass():
    """Test LedgerRecord dataclass."""
    import hashlib

    from worker.src.daemon.credit_ledger import LedgerRecord

    record = LedgerRecord(
        index=0,
        timestamp=1234567890.0,
        data=b"test data",
        prev_hash=hashlib.sha256(b"prev").digest(),
        hash=hashlib.sha256(b"test").digest(),
    )
    assert record.index == 0
    assert record.timestamp == 1234567890.0
    assert record.data == b"test data"


def test_signed_head_dataclass():
    """Test SignedHead dataclass."""
    import hashlib

    from worker.src.daemon.credit_ledger import SignedHead

    head = SignedHead(
        batch_index=0,
        root_hash=hashlib.sha256(b"root").digest(),
        signature=hashlib.sha256(b"sig").digest(),
        timestamp=1234567890.0,
        size=10,
    )
    assert head.batch_index == 0
    assert head.size == 10
    assert len(head.root_hash) == 32


def test_merkle_tree_creation():
    """Test MerkleTree creation."""
    import hashlib

    from worker.src.daemon.credit_ledger import MerkleTree

    data_hashes = [
        hashlib.sha256(b"data1").digest(),
        hashlib.sha256(b"data2").digest(),
    ]

    tree = MerkleTree(data_hashes)
    assert tree is not None
    root = tree.root_hash()
    assert root is not None
    assert len(root) == 32


def test_merkle_tree_single_item():
    """Test MerkleTree with single item."""
    import hashlib

    from worker.src.daemon.credit_ledger import MerkleTree

    data_hashes = [hashlib.sha256(b"single").digest()]
    tree = MerkleTree(data_hashes)
    assert tree is not None
    assert tree.root_hash() == data_hashes[0]


def test_enhanced_credit_ledger_import():
    """Test EnhancedCreditLedger import."""
    try:
        from worker.src.daemon.credit_ledger import EnhancedCreditLedger

        assert EnhancedCreditLedger is not None
    except ImportError:
        pytest.skip("EnhancedCreditLedger not available")


def test_stake_info_dataclass():
    """Test StakeInfo dataclass."""
    try:
        from worker.src.daemon.credit_ledger import StakeInfo

        stake = StakeInfo(
            contributor_id="node-123",
            amount=1000.0,
            staked_at=1234567890.0,
            lock_period_days=30,
            unlocks_at=1234567890.0 + 30 * 24 * 3600,
            purpose="security",
        )
        assert stake.contributor_id == "node-123"
        assert stake.amount == 1000.0
        assert stake.purpose == "security"
    except ImportError:
        pytest.skip("StakeInfo not available")
