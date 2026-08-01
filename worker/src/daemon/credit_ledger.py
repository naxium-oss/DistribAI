"""
Credit Ledger for DistribAI

Implements a hash-chained append-only ledger for credit tracking with
Merkle tree inclusion proofs, consistency proofs, and batch operations.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import struct
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MerkleNode:
    """
    Node in a Merkle tree.

    Attributes:
        left: Left child node
        right: Right child node
        hash: Hash value of this node

    Example:
        node = MerkleNode(hash=b"abc123")
    """

    left: MerkleNode | None = None
    right: MerkleNode | None = None
    hash: bytes = field(default_factory=lambda: b"")


@dataclass
class CreditEntry:
    """
    Credit transaction entry.

    Attributes:
        contributor_id: Node or user identifier
        job_id: Job identifier
        amount: Credit amount
        timestamp: Transaction timestamp
        metadata: Additional transaction metadata

    Example:
        entry = CreditEntry(
            contributor_id="node-001",
            job_id="job-123",
            amount=50.0
        )
    """

    contributor_id: str
    job_id: str
    amount: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class LedgerRecord:
    """
    Record in the credit ledger.

    Attributes:
        index: Record index in the ledger
        timestamp: Record timestamp
        data: Serialized record data
        prev_hash: Hash of previous record
        hash: Hash of this record
        batch_index: Batch index for grouping

    Example:
        record = LedgerRecord(
            index=0,
            timestamp=time.time(),
            data=b"entry_data",
            prev_hash=b"",
            hash=b"hash123"
        )
    """

    index: int
    timestamp: float
    data: bytes
    prev_hash: bytes
    hash: bytes
    batch_index: int = 0


@dataclass
class SignedHead:
    """
    Signed ledger head for tamper detection.

    Attributes:
        root_hash: Merkle tree root hash
        signature: Cryptographic signature
        timestamp: Head timestamp
        size: Number of records
        batch_index: Current batch index

    Example:
        head = SignedHead(
            root_hash=b"root123",
            signature=b"sig456",
            timestamp=time.time(),
            size=100
        )
    """

    root_hash: bytes
    signature: bytes
    timestamp: float
    size: int
    batch_index: int = 0


@dataclass
class ConsistencyProof:
    """
    Proof of ledger consistency between two states.

    Attributes:
        old_root: Old Merkle root hash
        new_root: New Merkle root hash
        proof: Proof path
        old_size: Old ledger size
        new_size: New ledger size

    Example:
        proof = ConsistencyProof(
            old_root=b"old123",
            new_root=b"new456",
            proof=[b"path1", b"path2"],
            old_size=100,
            new_size=200
        )
    """

    old_root: bytes
    new_root: bytes
    proof: list[bytes]
    old_size: int
    new_size: int


class MerkleTree:
    """
    Merkle tree for cryptographic proofs of data inclusion.

    Provides efficient verification that a specific piece of data is
    included in a dataset without revealing the entire dataset.

    Attributes:
        root: Root node of the Merkle tree
        leaves: Leaf nodes containing data hashes

    Example:
        tree = MerkleTree(data_hashes=[b"hash1", b"hash2", b"hash3"])
        root_hash = tree.root_hash()
        proof = tree.get_proof(index=0)
        verified = MerkleTree.verify_proof(root_hash, b"hash1", proof, 0)
    """

    def __init__(self, data_hashes: list[bytes]):
        """
        Initialize the Merkle tree from data hashes.

        Args:
            data_hashes: List of hash values for the data

        Example:
            >>> tree = MerkleTree(data_hashes=[b"hash1", b"hash2"])
        """
        self.root = self._build_tree([MerkleNode(hash=h) for h in data_hashes])
        self.leaves = [MerkleNode(hash=h) for h in data_hashes]

    def _build_tree(self, nodes: list[MerkleNode]) -> MerkleNode | None:
        """
        Build the Merkle tree from leaf nodes.

        Args:
            nodes: List of nodes at current level

        Returns:
            Root node of the tree
        """
        if len(nodes) == 0:
            return None
        if len(nodes) == 1:
            return nodes[0]
        next_level = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i + 1] if i + 1 < len(nodes) else left
            combined = left.hash + right.hash
            hash_val = hashlib.sha256(combined).digest()
            next_level.append(MerkleNode(left=left, right=right, hash=hash_val))
        return self._build_tree(next_level)

    def root_hash(self) -> bytes:
        """
        Get the root hash of the Merkle tree.

        Returns:
            Root hash bytes

        Example:
            >>> root = tree.root_hash()
            >>> print(f"Root hash: {root.hex()}")
        """
        return self.root.hash if self.root else b""

    def get_proof(self, index: int) -> list[bytes]:
        """
        Generate an inclusion proof for a leaf at the given index.

        Args:
            index: Index of the leaf in the tree

        Returns:
            List of hashes forming the proof path

        Example:
            >>> proof = tree.get_proof(index=0)
            >>> print(f"Proof length: {len(proof)}")
        """
        if index < 0 or index >= len(self.leaves):
            return []
        proof = []
        node = self.leaves[index]
        current_index = index
        while node != self.root:
            parent = self._find_parent(self.root, node)
            if parent is None:
                break
            if parent.left == node:
                proof.append(parent.right.hash)
                current_index = current_index // 2
            else:
                proof.append(parent.left.hash)
                current_index = (current_index - 1) // 2
            node = parent
        return proof

    def _find_parent(self, root: MerkleNode | None, node: MerkleNode) -> MerkleNode | None:
        """
        Find the parent of a node in the tree.

        Args:
            root: Root node to search from
            node: Node to find parent for

        Returns:
            Parent node or None if not found
        """
        if root is None or root == node:
            return None
        if root.left == node or root.right == node:
            return root
        left_parent = self._find_parent(root.left, node)
        if left_parent:
            return left_parent
        return self._find_parent(root.right, node)

    @staticmethod
    def verify_proof(root_hash: bytes, data: bytes, proof: list[bytes], index: int) -> bool:
        """
        Verify an inclusion proof against a root hash.

        Args:
            root_hash: Expected root hash
            data: Data to verify
            proof: Proof path hashes
            index: Index of the data in the original set

        Returns:
            True if proof is valid, False otherwise

        Example:
            >>> is_valid = MerkleTree.verify_proof(
            ...     root_hash=b"root123",
            ...     data=b"my_data",
            ...     proof=[b"sibling1", b"sibling2"],
            ...     index=0
            ... )
        """
        computed_hash = hashlib.sha256(data).digest()
        for i, sibling_hash in enumerate(proof):
            if (index >> i) & 1 == 0:
                combined = computed_hash + sibling_hash
            else:
                combined = sibling_hash + computed_hash
            computed_hash = hashlib.sha256(combined).digest()
        return computed_hash == root_hash


class CreditLedger:
    """
    Append-only credit ledger with cryptographic integrity.

    Provides tamper-evident storage for credit transactions using
    hash chaining, Merkle trees, and digital signatures.

    Attributes:
        records: List of ledger records
        root_hash: Current Merkle root hash
        signature: Digital signature of the current head
        signed_heads: History of signed heads
        batch_size: Number of records per batch
        current_batch: Current batch of records
        batch_index: Current batch index
        signing_key: Key for signing operations

    Example:
        ledger = CreditLedger(signing_key=b"secret_key")
        ledger.append(b"transaction_data")
        head = ledger.get_signed_head()
    """

    def __init__(self, signing_key: bytes | None = None, batch_size: int = 100):
        """
        Initialize the credit ledger.

        Args:
            signing_key: Key for signing root hashes (HMAC-SHA256)
            batch_size: Number of records per batch for efficiency

        Example:
            >>> ledger = CreditLedger(signing_key=b"my_secret_key")
        """
        self.records: list[LedgerRecord] = []
        self.root_hash: bytes = b""
        self.signature: bytes = b""
        self.signed_heads: list[SignedHead] = []
        self.batch_size = batch_size
        self.current_batch: list[LedgerRecord] = []
        self.batch_index = 0
        import threading

        self._lock = threading.Lock()
        if signing_key is None:
            env_key = os.getenv("SIGNING_KEY", "").strip()
            if env_key:
                signing_key = env_key
            else:
                try:
                    from services_python.constants import SIGNING_KEY

                    signing_key = SIGNING_KEY
                except ImportError:
                    signing_key = secrets.token_urlsafe(32)
                if not env_key:
                    logger.warning(
                        "SIGNING_KEY is not set; using process signing material. "
                        "Set SIGNING_KEY to match the orchestrator for ledger continuity."
                    )
        self.signing_key = signing_key.encode() if isinstance(signing_key, str) else signing_key

    def _compute_record_hash(
        self, index: int, timestamp: float, data: bytes, prev_hash: bytes
    ) -> bytes:
        """
        Compute the hash for a ledger record.

        Args:
            index: Record index
            timestamp: Record timestamp
            data: Record data
            prev_hash: Hash of previous record

        Returns:
            SHA-256 hash of the record
        """
        payload = b"".join(
            [
                index.to_bytes(8, byteorder="big", signed=False),
                struct.pack(">d", float(timestamp)),
                len(data).to_bytes(8, byteorder="big", signed=False),
                data,
                prev_hash,
            ]
        )
        return hashlib.sha256(payload).digest()

    def append(self, data: bytes) -> int:
        """
        Append data to the ledger.

        Args:
            data: Data to append

        Returns:
            Index of the appended record

        Example:
            >>> index = ledger.append(b"transaction_data")
            >>> print(f"Record index: {index}")
        """
        with self._lock:
            prev_hash = self.records[-1].hash if self.records else b""
            index = len(self.records)
            timestamp = time.time()
            record_hash = self._compute_record_hash(index, timestamp, data, prev_hash)
            record = LedgerRecord(
                index=index,
                timestamp=timestamp,
                data=data,
                prev_hash=prev_hash,
                hash=record_hash,
                batch_index=self.batch_index,
            )
            self.records.append(record)
            self.current_batch.append(record)
            if len(self.current_batch) >= self.batch_size:
                self._finalize_batch()
            return index

    def _finalize_batch(self) -> None:
        """
        Finalize the current batch of records.

        Computes the Merkle tree root hash, signs it, and stores
        the signed head. Called automatically when batch size is reached.
        """
        if not self.current_batch:
            return
        data_hashes = [r.hash for r in self.records]
        tree = MerkleTree(data_hashes)
        self.root_hash = tree.root_hash()
        self.signature = self._sign_root_hash(self.root_hash)
        signed_head = SignedHead(
            root_hash=self.root_hash,
            signature=self.signature,
            timestamp=time.time(),
            size=len(self.records),
            batch_index=self.batch_index,
        )
        self.signed_heads.append(signed_head)
        self.current_batch.clear()
        self.batch_index += 1

    def force_finalize(self) -> None:
        """
        Force finalization of the current batch.

        Called to finalize a batch even if it hasn't reached the batch size.

        Example:
            >>> ledger.force_finalize()
        """
        with self._lock:
            if self.current_batch:
                self._finalize_batch()

    def append_batch(self, data_list: list[bytes]) -> list[int]:
        """
        Append multiple records in a batch for efficiency.

        Thread-safe batch append operation.

        Args:
            data_list: List of data to append

        Returns:
            List of indices where records were appended

        Example:
            >>> indices = ledger.append_batch([b"data1", b"data2", b"data3"])
            >>> print(f"Appended at indices: {indices}")
        """
        indices = []
        for data in data_list:
            indices.append(self.append(data))
        self.force_finalize()
        return indices

    def add_credit(
        self, contributor_id: str, job_id: str, amount: float, metadata: dict | None = None
    ) -> int:
        """
        Add a credit entry to the ledger.

        Args:
            contributor_id: Node or user identifier
            job_id: Job identifier
            amount: Credit amount (must be non-negative)
            metadata: Optional metadata dictionary

        Returns:
            Index of the appended record

        Raises:
            ValueError: If amount is negative

        Example:
            >>> index = ledger.add_credit(
            ...     contributor_id="node-001",
            ...     job_id="job-123",
            ...     amount=50.0
            ... )
        """
        if amount < 0:
            raise ValueError("credit amount must be non-negative")
        entry = CreditEntry(
            contributor_id=contributor_id, job_id=job_id, amount=amount, metadata=metadata or {}
        )
        data = json.dumps(
            {
                "contributor_id": entry.contributor_id,
                "job_id": entry.job_id,
                "amount": entry.amount,
                "timestamp": entry.timestamp,
                "metadata": entry.metadata,
            }
        ).encode()
        return self.append(data)

    def credit(
        self, contributor_id: str, amount: float, job_id: str = "", metadata: dict | None = None
    ) -> int:
        """Append a signed credit transaction from live worker results."""
        return self.add_credit(contributor_id, job_id, amount, metadata)

    def get_balance(self, contributor_id: str) -> float:
        """Return confirmed balance for a contributor from signed ledger records."""
        balance = 0.0
        for record in self.records:
            try:
                data = json.loads(record.data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if data.get("contributor_id") == contributor_id:
                balance += float(data.get("amount", 0.0))
        return balance

    def append_record(
        self,
        node_id: str,
        tx_type: str,
        amount: float,
        metadata: dict | None = None,
        job_id: str = "",
    ) -> int:
        """
        Append a generic record to the ledger.

        Args:
            node_id: Node identifier
            tx_type: Transaction type
            amount: Transaction amount
            metadata: Optional metadata
            job_id: Optional job identifier

        Returns:
            Index of the appended record

        Example:
            >>> index = ledger.append_record(
            ...     node_id="node-001",
            ...     tx_type="reward",
            ...     amount=100.0
            ... )
        """
        data = json.dumps(
            {
                "contributor_id": node_id,
                "job_id": job_id,
                "tx_type": tx_type,
                "amount": amount,
                "timestamp": time.time(),
                "metadata": metadata or {},
            },
            sort_keys=True,
        ).encode("utf-8")
        return self.append(data)

    def get_proof(self, index: int) -> list[bytes]:
        """
        Get an inclusion proof for a record at the given index.

        Args:
            index: Record index

        Returns:
            List of hashes forming the proof path

        Example:
            >>> proof = ledger.get_proof(index=0)
            >>> print(f"Proof: {proof}")
        """
        if index < 0 or index >= len(self.records):
            return []
        data_hashes = [r.hash for r in self.records]
        tree = MerkleTree(data_hashes)
        return tree.get_proof(index)

    def verify_record(self, index: int, data: bytes) -> bool:
        """
        Verify a record against the current root hash.

        Args:
            index: Record index to verify
            data: Expected record data

        Returns:
            True if record is valid, False otherwise

        Example:
            >>> is_valid = ledger.verify_record(index=0, data=b"expected_data")
            >>> print(f"Valid: {is_valid}")
        """
        if index < 0 or index >= len(self.records):
            return False
        proof = self.get_proof(index)
        return MerkleTree.verify_proof(self.root_hash, data, proof, index)

    def verify_credit(self, index: int, entry: CreditEntry) -> bool:
        """
        Verify a credit entry against the ledger.

        Args:
            index: Record index to verify
            entry: Credit entry to verify

        Returns:
            True if entry is valid at the given index

        Example:
            >>> entry = CreditEntry(contributor_id="node-001", job_id="job-123", amount=50.0)
            >>> is_valid = ledger.verify_credit(index=0, entry=entry)
        """
        data = json.dumps(
            {
                "contributor_id": entry.contributor_id,
                "job_id": entry.job_id,
                "amount": entry.amount,
                "timestamp": entry.timestamp,
                "metadata": entry.metadata,
            }
        ).encode()
        return self.verify_record(index, data)

    def get_total_credits(self, contributor_id: str) -> float:
        """
        Get total credits for a contributor.

        Args:
            contributor_id: Node or user identifier

        Returns:
            Total credit amount for the contributor

        Example:
            >>> total = ledger.get_total_credits("node-001")
            >>> print(f"Total credits: {total}")
        """
        total = 0.0
        for record in self.records:
            try:
                entry_data = json.loads(record.data.decode())
                if entry_data.get("contributor_id") == contributor_id:
                    total += entry_data.get("amount", 0)
            except (json.JSONDecodeError, KeyError):
                continue
        return total

    def get_consistency_proof(self, old_size: int) -> ConsistencyProof | None:
        """
        Generate a consistency proof showing the ledger hasn't been tampered with.

        Args:
            old_size: Size of the ledger at the previous checkpoint

        Returns:
            Consistency proof or None if old_size is invalid

        Example:
            >>> proof = ledger.get_consistency_proof(old_size=100)
            >>> if proof:
            ...     print(f"Proof generated")
        """
        if old_size < 0 or old_size > len(self.records):
            return None
        if old_size == len(self.records):
            return ConsistencyProof(
                old_root=self.root_hash,
                new_root=self.root_hash,
                proof=[],
                old_size=old_size,
                new_size=len(self.records),
            )
        old_data_hashes = [r.hash for r in self.records[:old_size]]
        old_tree = MerkleTree(old_data_hashes)
        new_data_hashes = [r.hash for r in self.records]
        new_tree = MerkleTree(new_data_hashes)
        proof = self.get_proof(old_size - 1) if old_size > 0 else []
        return ConsistencyProof(
            old_root=old_tree.root_hash(),
            new_root=new_tree.root_hash(),
            proof=proof,
            old_size=old_size,
            new_size=len(self.records),
        )

    def verify_consistency_proof(self, proof: ConsistencyProof) -> bool:
        """
        Verify a consistency proof.

        Args:
            proof: Consistency proof to verify

        Returns:
            True if proof is valid

        Example:
            >>> is_valid = ledger.verify_consistency_proof(proof)
            >>> print(f"Proof valid: {is_valid}")
        """
        current_hashes = [r.hash for r in self.records]
        current_tree = MerkleTree(current_hashes)
        return current_tree.root_hash() == proof.new_root

    def get_credit_history(self, contributor_id: str) -> list[dict]:
        """
        Get credit history for a contributor.

        Args:
            contributor_id: Node or user identifier

        Returns:
            List of credit transactions with proofs

        Example:
            >>> history = ledger.get_credit_history("node-001")
            >>> for tx in history:
            ...     print(f"Amount: {tx['amount']}")
        """
        history = []
        for record in self.records:
            try:
                entry_data = json.loads(record.data.decode())
                if entry_data.get("contributor_id") == contributor_id:
                    history.append(
                        {
                            "index": record.index,
                            "timestamp": record.timestamp,
                            "job_id": entry_data.get("job_id"),
                            "amount": entry_data.get("amount"),
                            "proof": self.get_proof(record.index),
                            "batch_index": record.batch_index,
                        }
                    )
            except (json.JSONDecodeError, KeyError):
                continue
        return history

    def verify_chain_integrity(self) -> bool:
        """
        Verify the integrity of the entire ledger chain.

        When no batch has been finalized yet, root_hash and signature may be
        empty while records still form a valid hash chain. Only the per-record
        links are verified in that case. After force_finalize (or when a full
        batch closes), Merkle root and HMAC signature must match the chain.

        Returns:
            True if the chain is valid, False otherwise

        Example:
            >>> is_valid = ledger.verify_chain_integrity()
            >>> print(f"Chain valid: {is_valid}")
        """
        if not self.records:
            return True
        for i, record in enumerate(self.records):
            if i > 0:
                if record.prev_hash != self.records[i - 1].hash:
                    return False
            expected_hash = self._compute_record_hash(
                record.index, record.timestamp, record.data, record.prev_hash
            )
            if record.hash != expected_hash:
                return False
        data_hashes = [r.hash for r in self.records]
        if not data_hashes:
            return True
        tree = MerkleTree(data_hashes)
        computed_root = tree.root_hash()
        if self.root_hash and self.signature:
            if computed_root != self.root_hash:
                return False
            if not self._verify_signature(self.root_hash, self.signature):
                return False
        return True

    def get_signed_head(self, batch_index: int = -1) -> SignedHead | None:
        """
        Get a signed head at a specific batch index.

        Args:
            batch_index: Batch index (-1 for latest)

        Returns:
            Signed head or None if not found

        Example:
            >>> head = ledger.get_signed_head(batch_index=-1)
            >>> print(f"Root hash: {head.root_hash.hex()}")
        """
        if batch_index == -1:
            return self.signed_heads[-1] if self.signed_heads else None
        for head in self.signed_heads:
            if head.batch_index == batch_index:
                return head
        return None

    def get_all_signed_heads(self) -> list[SignedHead]:
        """
        Get all signed heads from the ledger.

        Returns:
            List of all signed heads

        Example:
            >>> heads = ledger.get_all_signed_heads()
            >>> print(f"Total heads: {len(heads)}")
        """
        return self.signed_heads.copy()

    def _sign_root_hash(self, root_hash: bytes) -> bytes:
        """
        Sign a root hash using HMAC-SHA256.

        Args:
            root_hash: Root hash to sign

        Returns:
            HMAC signature
        """
        return hmac.new(self.signing_key, root_hash, hashlib.sha256).digest()

    def _verify_signature(self, root_hash: bytes, signature: bytes) -> bool:
        """
        Verify a signature against a root hash.

        Args:
            root_hash: Root hash
            signature: Signature to verify

        Returns:
            True if signature is valid
        """
        expected = self._sign_root_hash(root_hash)
        return hmac.compare_digest(expected, signature)

    def get_record(self, index: int) -> LedgerRecord | None:
        """
        Get a record by index.

        Args:
            index: Record index

        Returns:
            Ledger record or None if not found

        Example:
            >>> record = ledger.get_record(index=0)
            >>> if record:
            ...     print(f"Record: {record.index}")
        """
        if 0 <= index < len(self.records):
            return self.records[index]
        return None

    def size(self) -> int:
        """
        Get the number of records in the ledger.

        Returns:
            Number of records

        Example:
            >>> count = ledger.size()
            >>> print(f"Total records: {count}")
        """
        return len(self.records)

    def get_root_hash(self) -> bytes:
        """
        Get the current Merkle root hash.

        Returns:
            Root hash bytes

        Example:
            >>> root = ledger.get_root_hash()
            >>> print(f"Root: {root.hex()}")
        """
        return self.root_hash


@dataclass
class StakeInfo:
    """Information about a staked amount."""

    contributor_id: str
    amount: float
    staked_at: float
    lock_period_days: int
    unlocks_at: float
    purpose: str  # "security", "governance", "validation"
    slashed_amount: float = 0.0


class EnhancedCreditLedger(CreditLedger):
    """
    Enhanced credit ledger with staking and slashing capabilities.

    Extends the base CreditLedger with:
    - Staking: Lock credits for network security/governance
    - Slashing: Penalize misbehavior by burning staked credits
    - Vesting: Time-locked credit releases
    - Yield: Interest on staked credits

    Attributes:
        stakes: Dictionary of active stakes by contributor
        vesting_schedules: Pending credit releases
        slashing_conditions: Rules for automatic slashing
        annual_yield_rate: Interest rate for staked credits

    Example:
        ledger = EnhancedCreditLedger(signing_key=b"secret")
        ledger.stake("node-001", 1000.0, lock_days=30, purpose="security")
        ledger.slash("node-001", 0.5, reason="byzantine_behavior")  # Slash 50%
    """

    # Slashing conditions and penalties
    SLASHING_RULES = {
        "byzantine_behavior": 0.5,  # 50% slash for confirmed Byzantine
        "double_voting": 0.3,  # 30% slash for voting fraud
        " Sybil_attack": 1.0,  # 100% slash for Sybil
        "checkpoint_failure": 0.1,  # 10% slash for missed checkpoints
        "downtime_excessive": 0.05,  # 5% slash for >90% downtime
    }

    def __init__(
        self,
        signing_key: bytes | None = None,
        batch_size: int = 100,
        annual_yield_rate: float = 0.05,  # 5% APR
    ):
        super().__init__(signing_key, batch_size)
        self.stakes: dict[str, list[StakeInfo]] = {}
        self.vesting_schedules: dict[str, list[dict]] = {}
        self.annual_yield_rate = annual_yield_rate
        self.slashing_history: list[dict] = []
        self.total_staked: float = 0.0
        self.total_slashed: float = 0.0

    def stake(
        self,
        contributor_id: str,
        amount: float,
        lock_period_days: int = 30,
        purpose: str = "security",
    ) -> StakeInfo:
        """
        Stake credits for network participation.

        Args:
            contributor_id: Node or user identifier
            amount: Amount to stake
            lock_period_days: Days until unlock
            purpose: Staking purpose (security/governance/validation)

        Returns:
            Stake information

        Raises:
            ValueError: If insufficient balance

        Example:
            >>> stake = ledger.stake("node-001", 1000.0, lock_days=30)
            >>> print(f"Staked until {stake.unlocks_at}")
        """
        available = self.get_balance(contributor_id) - self.get_staked_amount(contributor_id)
        if amount > available:
            raise ValueError(f"Insufficient balance. Available: {available}, Requested: {amount}")

        now = time.time()
        unlocks_at = now + (lock_period_days * 24 * 3600)

        stake_info = StakeInfo(
            contributor_id=contributor_id,
            amount=amount,
            staked_at=now,
            lock_period_days=lock_period_days,
            unlocks_at=unlocks_at,
            purpose=purpose,
        )

        if contributor_id not in self.stakes:
            self.stakes[contributor_id] = []
        self.stakes[contributor_id].append(stake_info)
        self.total_staked += amount

        # Record to ledger
        stake_record = {
            "type": "stake",
            "contributor_id": contributor_id,
            "amount": amount,
            "lock_period_days": lock_period_days,
            "purpose": purpose,
            "unlocks_at": unlocks_at,
        }
        self.append(json.dumps(stake_record).encode())

        logger.info(f"Staked {amount} credits for {contributor_id} until {unlocks_at}")
        return stake_info

    def unstake(self, contributor_id: str, stake_index: int | None = None) -> float:
        """
        Unstake credits after lock period expires.

        Args:
            contributor_id: Node or user identifier
            stake_index: Specific stake to unstake (None = oldest unlocked)

        Returns:
            Amount unstaked

        Raises:
            ValueError: If no unlocked stakes available

        Example:
            >>> amount = ledger.unstake("node-001")
            >>> print(f"Unstaked {amount} credits")
        """
        if contributor_id not in self.stakes or not self.stakes[contributor_id]:
            raise ValueError(f"No stakes found for {contributor_id}")

        now = time.time()

        if stake_index is not None:
            stake = self.stakes[contributor_id][stake_index]
            if now < stake.unlocks_at:
                raise ValueError(f"Stake still locked until {stake.unlocks_at}")
            unstaked = stake.amount
            self.stakes[contributor_id].pop(stake_index)
        else:
            # Find oldest unlocked stake
            for i, stake in enumerate(self.stakes[contributor_id]):
                if now >= stake.unlocks_at:
                    unstaked = stake.amount - stake.slashed_amount
                    self.stakes[contributor_id].pop(i)
                    break
            else:
                raise ValueError("No unlocked stakes available")

        self.total_staked -= unstaked

        # Calculate and add yield
        yield_amount = self._calculate_yield(unstaked, stake.staked_at, now)
        total_return = unstaked + yield_amount

        # Credit back to balance
        self.credit(contributor_id, yield_amount, job_id="staking_yield")

        # Record to ledger
        unstake_record = {
            "type": "unstake",
            "contributor_id": contributor_id,
            "amount": unstaked,
            "yield": yield_amount,
            "total_return": total_return,
        }
        self.append(json.dumps(unstake_record).encode())

        logger.info(f"Unstaked {unstaked} + {yield_amount} yield for {contributor_id}")
        return total_return

    def slash(
        self,
        contributor_id: str,
        fraction: float | None = None,
        reason: str = "byzantine_behavior",
    ) -> float:
        """
        Slash (burn) a portion of staked credits for misbehavior.

        Args:
            contributor_id: Node to slash
            fraction: Portion to slash (0.0-1.0), or None for rule-based
            reason: Slashing reason (must be in SLASHING_RULES)

        Returns:
            Amount slashed

        Example:
            >>> slashed = ledger.slash("node-001", reason="byzantine_behavior")
            >>> print(f"Slashed {slashed} credits from malicious node")
        """
        if contributor_id not in self.stakes:
            return 0.0

        # Determine slash fraction
        if fraction is None:
            fraction = self.SLASHING_RULES.get(reason, 0.1)

        fraction = max(0.0, min(1.0, fraction))

        total_slashed = 0.0
        for stake in self.stakes[contributor_id]:
            slash_amount = stake.amount * fraction
            stake.slashed_amount += slash_amount
            stake.amount -= slash_amount
            total_slashed += slash_amount

        self.total_slashed += total_slashed
        self.total_staked -= total_slashed

        # Record slashing
        slash_record = {
            "type": "slash",
            "contributor_id": contributor_id,
            "amount": total_slashed,
            "fraction": fraction,
            "reason": reason,
            "timestamp": time.time(),
        }
        self.slashing_history.append(slash_record)
        self.append(json.dumps(slash_record).encode())

        logger.warning(f"Slashed {total_slashed} credits from {contributor_id} for {reason}")
        return total_slashed

    def create_vesting_schedule(
        self,
        contributor_id: str,
        total_amount: float,
        vesting_months: int = 12,
        cliff_months: int = 3,
    ) -> dict:
        """
        Create a vesting schedule for credits.

        Args:
            contributor_id: Beneficiary
            total_amount: Total credits to vest
            vesting_months: Total vesting period
            cliff_months: Months before any vesting

        Returns:
            Vesting schedule details

        Example:
            >>> schedule = ledger.create_vesting_schedule("node-001", 12000.0)
            >>> # Credits vest 1000/month after 3-month cliff
        """
        now = time.time()
        monthly_amount = total_amount / vesting_months

        schedule = {
            "contributor_id": contributor_id,
            "total_amount": total_amount,
            "vested_amount": 0.0,
            "monthly_amount": monthly_amount,
            "vesting_months": vesting_months,
            "cliff_months": cliff_months,
            "created_at": now,
            "releases": [],
        }

        # Generate release schedule
        for month in range(vesting_months):
            release_time = now + ((cliff_months + month + 1) * 30 * 24 * 3600)
            schedule["releases"].append(
                {
                    "month": month + 1,
                    "amount": monthly_amount,
                    "release_time": release_time,
                    "released": False,
                }
            )

        if contributor_id not in self.vesting_schedules:
            self.vesting_schedules[contributor_id] = []
        self.vesting_schedules[contributor_id].append(schedule)

        # Lock the credits
        self.stake(
            contributor_id, total_amount, lock_period_days=vesting_months * 30, purpose="vesting"
        )

        return schedule

    def process_vesting_releases(self) -> list[dict]:
        """
        Process due vesting releases.

        Returns:
            List of releases processed

        Example:
            >>> releases = ledger.process_vesting_releases()
            >>> for r in releases:
            ...     print(f"Released {r['amount']} to {r['contributor_id']}")
        """
        now = time.time()
        processed = []

        for contributor_id, schedules in self.vesting_schedules.items():
            for schedule in schedules:
                for release in schedule["releases"]:
                    if not release["released"] and now >= release["release_time"]:
                        # Credit the released amount
                        self.credit(contributor_id, release["amount"], job_id="vesting_release")
                        release["released"] = True
                        schedule["vested_amount"] += release["amount"]

                        processed.append(
                            {
                                "contributor_id": contributor_id,
                                "amount": release["amount"],
                                "month": release["month"],
                            }
                        )

                        logger.info(f"Vesting release: {release['amount']} to {contributor_id}")

        return processed

    def get_staked_amount(self, contributor_id: str) -> float:
        """Get total staked amount for a contributor."""
        if contributor_id not in self.stakes:
            return 0.0
        return sum(s.amount for s in self.stakes[contributor_id])

    def get_available_balance(self, contributor_id: str) -> float:
        """Get balance excluding staked amounts."""
        total = self.get_balance(contributor_id)
        staked = self.get_staked_amount(contributor_id)
        return max(0.0, total - staked)

    def _calculate_yield(self, amount: float, start_time: float, end_time: float) -> float:
        """Calculate staking yield based on duration."""
        duration_days = (end_time - start_time) / (24 * 3600)
        daily_rate = self.annual_yield_rate / 365
        return amount * daily_rate * duration_days

    def get_staking_stats(self, contributor_id: str | None = None) -> dict:
        """Get staking statistics."""
        if contributor_id:
            staked = self.get_staked_amount(contributor_id)
            stakes = self.stakes.get(contributor_id, [])
            return {
                "contributor_id": contributor_id,
                "total_staked": staked,
                "active_stakes": len(stakes),
                "average_lock_days": sum(s.lock_period_days for s in stakes) / len(stakes)
                if stakes
                else 0,
                "next_unlock": min((s.unlocks_at for s in stakes), default=None),
                "estimated_annual_yield": staked * self.annual_yield_rate,
            }

        return {
            "total_staked_network": self.total_staked,
            "total_slashed": self.total_slashed,
            "staking_participants": len(self.stakes),
            "average_annual_yield": self.annual_yield_rate,
            "total_slashing_events": len(self.slashing_history),
        }

    def get_slashed_history(self, contributor_id: str | None = None) -> list[dict]:
        """Get slashing history."""
        if contributor_id:
            return [s for s in self.slashing_history if s["contributor_id"] == contributor_id]
        return self.slashing_history.copy()
