from worker.src.daemon.credit_ledger import CreditEntry, CreditLedger, LedgerRecord


def _key32() -> bytes:
    return b"x" * 32


def test_credit_ledger_import():
    assert CreditLedger is not None
    assert CreditEntry is not None


def test_ledger_creation():
    ledger = CreditLedger(signing_key=_key32())
    assert ledger.size() == 0
    assert ledger.get_root_hash() == b""


def test_add_credit_and_total():
    ledger = CreditLedger(signing_key=_key32(), batch_size=2)
    i0 = ledger.add_credit("node1", "job1", 100.0)
    assert i0 == 0
    i1 = ledger.add_credit("node1", "job2", 50.0)
    assert i1 == 1
    assert ledger.get_total_credits("node1") == 150.0


def test_append_record_chain():
    ledger = CreditLedger(signing_key=_key32(), batch_size=10)
    ledger.append_record("node1", "contribution", 100.0, job_id="j1")
    ledger.append_record("node1", "vote", -20.0, job_id="j2")
    assert ledger.size() == 2
    assert ledger.records[1].prev_hash == ledger.records[0].hash


def test_verify_chain_integrity_unfinalized():
    ledger = CreditLedger(signing_key=_key32(), batch_size=100)
    ledger.add_credit("n1", "j1", 10.0)
    assert ledger.verify_chain_integrity() is True


def test_verify_chain_integrity_after_finalize():
    ledger = CreditLedger(signing_key=_key32(), batch_size=2)
    ledger.add_credit("n1", "j1", 10.0)
    ledger.add_credit("n2", "j2", 5.0)
    ledger.force_finalize()
    assert ledger.verify_chain_integrity() is True


def test_get_credit_history():
    ledger = CreditLedger(signing_key=_key32(), batch_size=10)
    ledger.add_credit("node1", "j1", 100.0)
    hist = ledger.get_credit_history("node1")
    assert len(hist) == 1
    assert hist[0]["job_id"] == "j1"
    assert hist[0]["amount"] == 100.0


def test_get_record():
    ledger = CreditLedger(signing_key=_key32(), batch_size=10)
    ledger.add_credit("n", "j", 1.0)
    rec = ledger.get_record(0)
    assert rec is not None
    assert isinstance(rec, LedgerRecord)


def test_empty_ledger_verify():
    ledger = CreditLedger(signing_key=_key32())
    assert ledger.verify_chain_integrity() is True
    assert ledger.get_total_credits("any") == 0.0


def test_ledger_record_hash_unique():
    """Test that record hashes are unique."""
    ledger = CreditLedger(signing_key=_key32(), batch_size=10)
    ledger.add_credit("node1", "job1", 100.0)
    ledger.add_credit("node1", "job2", 50.0)

    rec1 = ledger.get_record(0)
    rec2 = ledger.get_record(1)

    assert rec1.hash != rec2.hash


def test_ledger_signature_verification():
    """Test signature verification."""
    ledger = CreditLedger(signing_key=_key32(), batch_size=2)
    ledger.add_credit("node1", "job1", 100.0)
    ledger.add_credit("node2", "job2", 50.0)

    # After batch_size records, should be signed
    assert ledger.signature is not None


def test_ledger_size():
    """Test size method."""
    ledger = CreditLedger(signing_key=_key32(), batch_size=10)
    assert ledger.size() == 0

    ledger.add_credit("node1", "job1", 100.0)
    assert ledger.size() == 1

    ledger.add_credit("node2", "job2", 50.0)
    assert ledger.size() == 2


def test_ledger_root_hash():
    """Test get_root_hash method."""
    ledger = CreditLedger(signing_key=_key32(), batch_size=10)

    # Empty ledger has empty root hash
    assert ledger.get_root_hash() == b""

    ledger.add_credit("node1", "job1", 100.0)
    # After adding records, should have non-empty root hash
    root_hash = ledger.get_root_hash()
    assert isinstance(root_hash, bytes)
