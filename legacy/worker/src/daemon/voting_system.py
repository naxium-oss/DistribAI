"""
Voting System for DistribAI Governance

Implements a democratic voting system with quorum, multi-signature,
weighted voting, delegation, and reputation-based governance.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class VoteStatus(Enum):
    """
    Status of a voting proposal.

    Attributes:
        PENDING: Vote is currently open
        APPROVED: Vote passed
        REJECTED: Vote failed
        EXPIRED: Vote expired without reaching quorum
        QUORUM_NOT_MET: Quorum not reached

    Example:
        status = VoteStatus.APPROVED
        print(f"Status: {status.value}")
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    QUORUM_NOT_MET = "quorum_not_met"


class VoteType(Enum):
    """
    Types of voting proposals.

    Attributes:
        JOB_PRIORITY: Job priority voting
        SYSTEM_CONFIG: System configuration changes
        GOVERNANCE: Governance proposals
        PARAMETER_CHANGE: Parameter adjustments
        TRUST_ANCHOR: Trust anchor selection

    Example:
        vote_type = VoteType.GOVERNANCE
    """

    JOB_PRIORITY = "job_priority"
    SYSTEM_CONFIG = "system_config"
    GOVERNANCE = "governance"
    PARAMETER_CHANGE = "parameter_change"
    TRUST_ANCHOR = "trust_anchor"


@dataclass
class VoteSignature:
    """
    Digital signature for a vote.

    Attributes:
        voter_id: Voter identifier
        signature: Cryptographic signature
        timestamp: Signature timestamp

    Example:
        sig = VoteSignature(voter_id="node-001", signature=b"sig123", timestamp=time.time())
    """

    voter_id: str
    signature: bytes
    timestamp: float


@dataclass
class Vote:
    """
    Voting proposal with all metadata and results.

    Attributes:
        id: Unique vote identifier
        vote_type: Type of vote
        title: Vote title
        description: Vote description
        proposer: Proposer identifier
        options: List of voting options
        created_at: Creation timestamp
        expires_at: Expiration timestamp
        status: Current vote status
        votes_cast: Dictionary of voter to option mappings
        signatures: List of vote signatures
        credits_required: Minimum credits to propose
        min_participation: Minimum participation rate
        quorum_required: Quorum threshold
        weighted_voting: Whether voting is credit-weighted
        delegation_chain: Vote delegations
        winning_option: The option that won

    Example:
        vote = Vote(
            id="vote-123",
            vote_type=VoteType.GOVERNANCE,
            title="Update Configuration",
            description="Proposal to update system configuration",
            proposer="node-001",
            options=["yes", "no", "abstain"]
        )
    """

    id: str
    vote_type: VoteType
    title: str
    description: str
    proposer: str
    options: list[str]
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    status: VoteStatus = VoteStatus.PENDING
    votes_cast: dict[str, str] = field(default_factory=dict)
    signatures: list[VoteSignature] = field(default_factory=list)
    credits_required: int = 100
    min_participation: float = 0.1
    quorum_required: float = 0.5
    weighted_voting: bool = True
    delegation_chain: dict[str, str] = field(default_factory=dict)
    winning_option: str | None = None


@dataclass
class CreditAccount:
    """
    Credit account for voting power.

    Attributes:
        contributor_id: Contributor identifier
        balance: Credit balance

    Example:
        account = CreditAccount(contributor_id="node-001", balance=1000.0)
    """

    contributor_id: str
    balance: float = 0.0
    credits_earned: float = 0.0
    credits_spent: float = 0.0
    votes_cast: int = 0
    reputation_score: float = 0.0
    delegated_to: str | None = None
    delegates: set[str] = field(default_factory=set)


@dataclass
class QuorumConfig:
    """
    Configuration for quorum requirements per vote type.

    Attributes:
        vote_type: Type of vote this config applies to
        min_participation: Minimum participation rate required
        quorum_required: Quorum threshold
        approval_threshold: Approval threshold for passing
        weighted_voting: Whether voting is credit-weighted
        min_voters: Minimum number of voters required

    Example:
        config = QuorumConfig(
            vote_type=VoteType.GOVERNANCE,
            min_participation=0.3,
            quorum_required=0.7,
            approval_threshold=0.75
        )
    """

    vote_type: VoteType
    min_participation: float = 0.1
    quorum_required: float = 0.5
    approval_threshold: float = 0.5
    weighted_voting: bool = True
    min_voters: int = 3


class VotingSystem:
    """
    Manages the voting system for governance.

    Handles proposal creation, voting, quorum enforcement, and
    result calculation with support for weighted voting and delegation.

    Attributes:
        votes: Dictionary of active and completed votes
        accounts: Credit accounts for voting power
        credit_price_per_vote: Credits required per vote
        vote_duration_hours: Default vote duration
        signing_key: Key for signing operations
        quorum_configs: Quorum configurations per vote type

    Example:
        system = VotingSystem(signing_key=b"secret_key")
        vote = system.create_proposal(...)
        system.cast_vote(vote_id, voter_id, option="yes")
    """

    def __init__(self, signing_key: bytes | None = None):
        """
        Initialize the voting system.

        Args:
            signing_key: Key for vote signature security

        Raises:
            ValueError: If signing_key is not provided

        Example:
            >>> system = VotingSystem(signing_key=b"my_secret_key")
        """
        self.votes: dict[str, Vote] = {}
        self.accounts: dict[str, CreditAccount] = {}
        self.credit_price_per_vote = 10
        self.vote_duration_hours = 24
        if signing_key is None:
            raise ValueError("signing_key is required for vote signature security")
        self.signing_key = signing_key.encode() if isinstance(signing_key, str) else signing_key
        self.quorum_configs: dict[VoteType, QuorumConfig] = {
            VoteType.JOB_PRIORITY: QuorumConfig(VoteType.JOB_PRIORITY, 0.05, 0.3, 0.5, True, 3),
            VoteType.SYSTEM_CONFIG: QuorumConfig(VoteType.SYSTEM_CONFIG, 0.2, 0.6, 0.67, True, 5),
            VoteType.GOVERNANCE: QuorumConfig(VoteType.GOVERNANCE, 0.3, 0.7, 0.75, True, 7),
            VoteType.PARAMETER_CHANGE: QuorumConfig(
                VoteType.PARAMETER_CHANGE, 0.1, 0.5, 0.67, True, 5
            ),
            VoteType.TRUST_ANCHOR: QuorumConfig(VoteType.TRUST_ANCHOR, 0.5, 0.8, 0.9, True, 10),
        }

    def create_account(self, contributor_id: str) -> CreditAccount:
        """
        Create a credit account for a contributor.

        Args:
            contributor_id: Unique contributor identifier

        Returns:
            Created account

        Example:
            >>> account = system.create_account(contributor_id="node-001")
        """
        if contributor_id not in self.accounts:
            self.accounts[contributor_id] = CreditAccount(contributor_id=contributor_id)
        return self.accounts[contributor_id]

    def add_credits(self, contributor_id: str, amount: float, job_id: str = "") -> bool:
        """
        Add credits to a contributor's account.

        Args:
            contributor_id: Contributor identifier
            amount: Amount of credits to add
            job_id: Job that earned these credits (for audit trail)

        Returns:
            True if successful

        Example:
            >>> success = system.add_credits("node-001", amount=100.0, job_id="job-123")
        """
        account = self.create_account(contributor_id)
        account.balance += amount
        account.credits_earned += amount
        return True

    def spend_credits(self, contributor_id: str, amount: float, reason: str = "") -> bool:
        """
        Spend credits from a contributor's account.

        Args:
            contributor_id: Contributor identifier
            amount: Amount of credits to spend (must be positive)
            reason: Reason for spending (for audit trail)

        Returns:
            True if successful, False if insufficient balance

        Example:
            >>> success = system.spend_credits("node-001", amount=50.0, reason="voting")
        """
        if amount <= 0:
            return False
        account = self.accounts.get(contributor_id)
        if not account or account.balance < amount:
            return False
        account.balance -= amount
        account.credits_spent += amount
        return True

    def create_vote(
        self,
        proposer: str,
        vote_type: VoteType,
        title: str,
        description: str,
        options: list[str],
        credits_required: int = 100,
        duration_hours: int = 24,
    ) -> str:
        """
        Create a new vote proposal.

        Args:
            proposer: Contributor proposing the vote
            vote_type: Type of vote
            title: Vote title
            description: Vote description
            options: List of voting options
            credits_required: Credits required to propose
            duration_hours: Duration of vote in hours

        Returns:
            Vote ID

        Raises:
            ValueError: If proposer has insufficient credits

        Example:
            >>> vote_id = system.create_vote(
            ...     proposer="node-001",
            ...     vote_type=VoteType.GOVERNANCE,
            ...     title="Update Configuration",
            ...     description="Proposal to update system configuration",
            ...     options=["yes", "no", "abstain"]
            ... )
        """
        if not self.spend_credits(proposer, credits_required, f"Vote proposal: {title}"):
            raise ValueError(f"Insufficient credits to propose vote (requires {credits_required})")
        vote_id = hashlib.sha256(f"{proposer}{title}{time.time()}".encode()).hexdigest()[:16]
        vote = Vote(
            id=vote_id,
            vote_type=vote_type,
            title=title,
            description=description,
            proposer=proposer,
            options=options,
            expires_at=time.time() + (duration_hours * 3600),
            credits_required=credits_required,
        )
        self.votes[vote_id] = vote
        return vote_id

    def cast_vote(self, voter_id: str, vote_id: str, option: str) -> bool:
        """
        Cast a vote on a proposal.

        Args:
            voter_id: Contributor casting the vote
            vote_id: Vote proposal ID
            option: Selected option

        Returns:
            True if vote was cast successfully

        Example:
            >>> success = system.cast_vote("node-001", "vote-123", option="yes")
        """
        vote = self.votes.get(vote_id)
        if not vote:
            return False
        if vote.status != VoteStatus.PENDING:
            return False
        if vote.expires_at and time.time() > vote.expires_at:
            vote.status = VoteStatus.EXPIRED
            return False
        if option not in vote.options:
            return False
        if voter_id in vote.votes_cast:
            return False
        if not self.spend_credits(voter_id, self.credit_price_per_vote, f"Vote on: {vote.title}"):
            return False
        vote.votes_cast[voter_id] = option
        account = self.create_account(voter_id)
        account.votes_cast += 1
        return True

    def tally_vote(self, vote_id: str) -> dict[str, int]:
        """
        Tally votes for a proposal.

        Args:
            vote_id: Vote proposal ID

        Returns:
            Dictionary of option -> vote count

        Example:
            >>> counts = system.tally_vote("vote-123")
            >>> print(f"Yes votes: {counts['yes']}")
        """
        vote = self.votes.get(vote_id)
        if not vote:
            return {}
        tally = dict.fromkeys(vote.options, 0)
        for option in vote.votes_cast.values():
            if option in tally:
                tally[option] += 1
        return tally

    def finalize_vote(self, vote_id: str) -> str | None:
        """
        Finalize a vote and determine the outcome with quorum checking.

        Args:
            vote_id: Vote proposal ID

        Returns:
            Winning option or None if vote is still pending

        Example:
            >>> winner = system.finalize_vote("vote-123")
            >>> if winner:
            ...     print(f"Winner: {winner}")
        """
        vote = self.votes.get(vote_id)
        if not vote:
            return None
        if vote.status != VoteStatus.PENDING:
            if vote.status == VoteStatus.APPROVED:
                return vote.winning_option
            return None
        if vote.expires_at and time.time() > vote.expires_at:
            vote.status = VoteStatus.EXPIRED
            return None
        config = self.quorum_configs.get(vote.vote_type)
        if not config:
            config = QuorumConfig(vote.vote_type)
        if len(vote.votes_cast) < config.min_voters:
            return None
        total_voters = len(self.accounts)
        votes_cast = len(vote.votes_cast)
        if total_voters > 0 and votes_cast / total_voters < config.quorum_required:
            vote.status = VoteStatus.QUORUM_NOT_MET
            return None
        if config.weighted_voting:
            tally = self._tally_weighted(vote_id)
        else:
            tally = self._tally_simple(vote_id)
        if not tally:
            return None
        total_weight = sum(tally.values())
        if total_weight == 0:
            return None
        winner = max(tally, key=tally.get)
        winner_ratio = tally[winner] / total_weight
        if winner_ratio >= config.approval_threshold:
            vote.status = VoteStatus.APPROVED
            vote.winning_option = winner
        else:
            vote.status = VoteStatus.REJECTED
        return winner

    def _tally_simple(self, vote_id: str) -> dict[str, int]:
        """
        Tally votes using simple one-person-one-vote counting.

        Args:
            vote_id: Vote proposal ID

        Returns:
            Dictionary of option -> vote count
        """
        vote = self.votes.get(vote_id)
        if not vote:
            return {}
        tally = dict.fromkeys(vote.options, 0)
        for option in vote.votes_cast.values():
            if option in tally:
                tally[option] += 1
        return tally

    def _tally_weighted(self, vote_id: str) -> dict[str, float]:
        """
        Tally votes using credit-weighted counting with delegation support.

        Args:
            vote_id: Vote proposal ID

        Returns:
            Dictionary of option -> weighted vote total
        """
        vote = self.votes.get(vote_id)
        if not vote:
            return {}
        tally = dict.fromkeys(vote.options, 0.0)
        for voter_id, option in vote.votes_cast.items():
            if option in tally:
                account = self.accounts.get(voter_id)
                weight = account.balance if account else 1.0
                if account:
                    for delegate_id in account.delegates:
                        delegate_account = self.accounts.get(delegate_id)
                        if delegate_account:
                            weight += delegate_account.balance
                tally[option] += weight
        return tally

    def delegate_vote(self, voter_id: str, delegate_to: str) -> bool:
        """
        Delegate voting power to another contributor with cycle detection.

        Args:
            voter_id: Voter delegating their power
            delegate_to: Recipient of delegation

        Returns:
            True if delegation successful

        Example:
            >>> success = system.delegate_vote("node-001", delegate_to="node-002")
        """
        voter_account = self.accounts.get(voter_id)
        delegate_account = self.accounts.get(delegate_to)
        if not voter_account or not delegate_account:
            return False
        if voter_id == delegate_to:
            return False
        if self._would_create_cycle(voter_id, delegate_to):
            return False
        if voter_account.delegated_to:
            old_delegate = self.accounts.get(voter_account.delegated_to)
            if old_delegate:
                old_delegate.delegates.discard(voter_id)
        voter_account.delegated_to = delegate_to
        delegate_account.delegates.add(voter_id)
        return True

    def _would_create_cycle(self, voter_id: str, delegate_to: str) -> bool:
        """
        Detect if creating a delegation would create a cycle.

        Args:
            voter_id: Voter delegating their power
            delegate_to: Recipient of delegation

        Returns:
            True if cycle would be created
        """
        visited = set()
        current = delegate_to
        while current:
            if current == voter_id:
                return True
            if current in visited:
                return True
            visited.add(current)
            account = self.accounts.get(current)
            if not account or not account.delegated_to:
                break
            current = account.delegated_to
        return False

    def revoke_delegation(self, voter_id: str) -> bool:
        """
        Revoke a voting delegation.

        Args:
            voter_id: Voter revoking their delegation

        Returns:
            True if revocation successful

        Example:
            >>> success = system.revoke_delegation("node-001")
        """
        voter_account = self.accounts.get(voter_id)
        if not voter_account or not voter_account.delegated_to:
            return False
        delegate_account = self.accounts.get(voter_account.delegated_to)
        if delegate_account:
            delegate_account.delegates.discard(voter_id)
        voter_account.delegated_to = None
        return True

    def sign_vote(self, vote_id: str, voter_id: str) -> VoteSignature | None:
        """
        Create a signature for a vote.

        Args:
            vote_id: Vote to sign
            voter_id: Voter signing

        Returns:
            Vote signature or None

        Example:
            >>> sig = system.sign_vote("vote-123", "node-001")
        """
        vote = self.votes.get(vote_id)
        if not vote:
            return None
        timestamp = time.time()
        data = f"{vote_id}:{voter_id}:{vote.winning_option or ''}:{int(timestamp)}".encode()
        signature = hmac.new(self.signing_key, data, hashlib.sha256).digest()
        vote_sig = VoteSignature(voter_id=voter_id, signature=signature, timestamp=timestamp)
        vote.signatures.append(vote_sig)
        return vote_sig

    def verify_signature(self, vote_id: str, signature: VoteSignature) -> bool:
        """
        Verify a vote signature.

        Args:
            vote_id: Vote proposal ID
            signature: Signature to verify

        Returns:
            True if signature is valid

        Example:
            >>> is_valid = system.verify_signature("vote-123", vote_sig)
        """
        vote = self.votes.get(vote_id)
        if not vote:
            return False
        data = f"{vote_id}:{signature.voter_id}:{vote.winning_option or ''}:{int(signature.timestamp)}".encode()
        expected = hmac.new(self.signing_key, data, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature.signature)

    def update_reputation(self, contributor_id: str, delta: float) -> bool:
        """
        Update reputation score for a contributor.

        Args:
            contributor_id: Contributor identifier
            delta: Reputation change (clamped to [0.0, 1.0])

        Returns:
            True if successful

        Example:
            >>> success = system.update_reputation("node-001", delta=0.1)
        """
        account = self.accounts.get(contributor_id)
        if not account:
            return False
        account.reputation_score = max(0.0, min(1.0, account.reputation_score + delta))
        return True

    def get_vote_status(self, vote_id: str) -> dict | None:
        """
        Get detailed status of a vote.

        Args:
            vote_id: Vote proposal ID

        Returns:
            Vote status dictionary or None

        Example:
            >>> status = system.get_vote_status("vote-123")
            >>> print(f"Status: {status['status']}")
        """
        vote = self.votes.get(vote_id)
        if not vote:
            return None
        tally = self.tally_vote(vote_id)
        winner = vote.winning_option
        if vote.status == VoteStatus.PENDING and vote.expires_at and time.time() > vote.expires_at:
            self.finalize_vote(vote_id)
            winner = vote.winning_option
        elif vote.status == VoteStatus.APPROVED:
            winner = vote.winning_option
        return {
            "id": vote.id,
            "type": vote.vote_type.value,
            "title": vote.title,
            "description": vote.description,
            "proposer": vote.proposer,
            "options": vote.options,
            "created_at": vote.created_at,
            "expires_at": vote.expires_at,
            "status": vote.status.value,
            "votes_cast": len(vote.votes_cast),
            "tally": tally,
            "winner": winner,
        }

    def get_active_votes(self) -> list[dict]:
        """
        Get all active (pending) votes.

        Returns:
            List of vote status dictionaries

        Example:
            >>> active = system.get_active_votes()
            >>> print(f"Active votes: {len(active)}")
        """
        return [
            self.get_vote_status(vote_id)
            for vote_id, vote in self.votes.items()
            if vote.status == VoteStatus.PENDING
        ]

    def get_credit_balance(self, contributor_id: str) -> float:
        """
        Get credit balance for a contributor.

        Args:
            contributor_id: Contributor identifier

        Returns:
            Credit balance

        Example:
            >>> balance = system.get_credit_balance("node-001")
            >>> print(f"Balance: {balance}")
        """
        account = self.accounts.get(contributor_id)
        return account.balance if account else 0.0

    def get_account_info(self, contributor_id: str) -> dict | None:
        """
        Get detailed account information.

        Args:
            contributor_id: Contributor identifier

        Returns:
            Account information dictionary or None

        Example:
            >>> info = system.get_account_info("node-001")
            >>> print(f"Balance: {info['balance']}")
        """
        account = self.accounts.get(contributor_id)
        if not account:
            return None
        return {
            "contributor_id": account.contributor_id,
            "balance": account.balance,
            "credits_earned": account.credits_earned,
            "credits_spent": account.credits_spent,
            "votes_cast": account.votes_cast,
        }

    def expire_votes(self) -> int:
        """
        Expire all votes that have passed their expiration time.
        Returns:
            Number of votes expired
        """
        expired_count = 0
        current_time = time.time()
        for vote in self.votes.values():
            if (
                vote.status == VoteStatus.PENDING
                and vote.expires_at
                and current_time > vote.expires_at
            ):
                vote.status = VoteStatus.EXPIRED
                expired_count += 1
        return expired_count

    def cleanup_expired_votes(self, max_age_hours: int = 168) -> int:
        """
        Remove expired votes older than max_age_hours.
        Args:
            max_age_hours: Maximum age in hours (default: 7 days)
        Returns:
            Number of votes removed
        """
        current_time = time.time()
        to_remove = []
        for vote_id, vote in self.votes.items():
            age_hours = (current_time - vote.created_at) / 3600
            if vote.status == VoteStatus.EXPIRED and age_hours > max_age_hours:
                to_remove.append(vote_id)
        for vote_id in to_remove:
            del self.votes[vote_id]
        return len(to_remove)


@dataclass
class QuadraticVote:
    """
    Quadratic voting record.

    In quadratic voting, the cost of votes increases quadratically:
    cost = votes^2

    This allows small contributors to have more voice per credit,
    reducing whale dominance.
    """

    voter_id: str
    vote_id: str
    votes_cast: int
    credits_spent: float
    option: str
    timestamp: float = field(default_factory=time.time)


class QuadraticVotingSystem(VotingSystem):
    """
    Enhanced voting system with quadratic voting support.

    Implements quadratic voting as researched in RFC 008:
    - Cost = votes^2 credits
    - Reduces whale dominance
    - More democratic for small contributors

    Also adds:
    - Vote velocity caps (prevent spam)
    - Automated execution for passed proposals
    - Proposal discussion threads
    - Vote receipts with proofs
    """

    # Vote velocity limits (RFC 008 Phase 3)
    MAX_VOTES_PER_HOUR = 1000
    MAX_VOTES_PER_JOB = 500

    def __init__(self, signing_key: bytes | None = None, enable_quadratic: bool = False):
        super().__init__(signing_key)
        self.enable_quadratic = enable_quadratic
        self.quadratic_votes: dict[str, list[QuadraticVote]] = {}
        self.vote_history: dict[str, list[dict]] = {}  # For velocity tracking
        self.proposal_discussions: dict[str, list[dict]] = {}
        self.executed_proposals: list[dict] = []
        self.vote_receipts: dict[str, dict] = {}

    def calculate_quadratic_cost(self, votes: int) -> int:
        """
        Calculate cost for quadratic voting.

        Args:
            votes: Number of votes to cast

        Returns:
            Credits required (votes^2)

        Example:
            >>> cost = system.calculate_quadratic_cost(5)
            >>> print(f"Cost for 5 votes: {cost} credits")  # 25 credits
        """
        return votes * votes

    def cast_quadratic_vote(
        self,
        voter_id: str,
        vote_id: str,
        option: str,
        votes: int,
    ) -> dict | None:
        """
        Cast a quadratic vote.

        Args:
            voter_id: Contributor casting vote
            vote_id: Vote proposal ID
            option: Selected option
            votes: Number of votes (cost = votes^2)

        Returns:
            Vote receipt or None if failed

        Example:
            >>> receipt = system.cast_quadratic_vote("node-001", "vote-123", "yes", votes=10)
            >>> print(f"Spent {receipt['credits_spent']} credits")
        """
        if not self.enable_quadratic:
            # Fall back to regular voting
            success = self.cast_vote(voter_id, vote_id, option)
            return (
                {"success": success, "credits_spent": self.credit_price_per_vote}
                if success
                else None
            )

        # Check velocity limits
        if not self._check_vote_velocity(voter_id, vote_id):
            logger.warning(f"Vote velocity limit exceeded for {voter_id}")
            return None

        # Calculate cost
        cost = self.calculate_quadratic_cost(votes)

        # Spend credits
        if not self.spend_credits(voter_id, cost, f"Quadratic vote on {vote_id}"):
            return None

        # Record quadratic vote
        qv = QuadraticVote(
            voter_id=voter_id,
            vote_id=vote_id,
            votes_cast=votes,
            credits_spent=cost,
            option=option,
        )

        if vote_id not in self.quadratic_votes:
            self.quadratic_votes[vote_id] = []
        self.quadratic_votes[vote_id].append(qv)

        # Also record in base system (as single vote)
        vote = self.votes.get(vote_id)
        if vote:
            vote.votes_cast[voter_id] = option

        # Track for velocity
        self._record_vote_activity(voter_id, vote_id, cost)

        # Generate receipt
        receipt = {
            "voter_id": voter_id,
            "vote_id": vote_id,
            "option": option,
            "votes": votes,
            "credits_spent": cost,
            "timestamp": time.time(),
            "receipt_id": hashlib.sha256(f"{voter_id}{vote_id}{time.time()}".encode()).hexdigest()[
                :16
            ],
        }

        # Sign receipt
        receipt_data = json.dumps(receipt, sort_keys=True).encode()
        receipt["signature"] = hmac.new(self.signing_key, receipt_data, hashlib.sha256).hexdigest()[
            :32
        ]

        self.vote_receipts[receipt["receipt_id"]] = receipt

        return receipt

    def _check_vote_velocity(self, voter_id: str, vote_id: str) -> bool:
        """Check if voter is within velocity limits."""
        now = time.time()

        if voter_id not in self.vote_history:
            self.vote_history[voter_id] = []

        history = self.vote_history[voter_id]

        # Clean old entries
        hour_ago = now - 3600
        recent_votes = [h for h in history if h["timestamp"] > hour_ago]

        # Check hourly limit
        if len(recent_votes) >= self.MAX_VOTES_PER_HOUR:
            return False

        # Check per-job limit
        job_votes = sum(1 for h in recent_votes if h.get("vote_id") == vote_id)
        if job_votes >= self.MAX_VOTES_PER_JOB:
            return False

        return True

    def _record_vote_activity(self, voter_id: str, vote_id: str, credits: float) -> None:
        """Record vote for velocity tracking."""
        if voter_id not in self.vote_history:
            self.vote_history[voter_id] = []

        self.vote_history[voter_id].append(
            {
                "vote_id": vote_id,
                "timestamp": time.time(),
                "credits": credits,
            }
        )

    def tally_quadratic_votes(self, vote_id: str) -> dict[str, float]:
        """
        Tally votes using quadratic counting.

        In quadratic voting, the "voice credits" are the square root
        of votes spent (since cost = votes^2).

        Returns:
            Dictionary of option -> voice credits
        """
        if vote_id not in self.quadratic_votes:
            return {}

        tally: dict[str, float] = {}

        for qv in self.quadratic_votes[vote_id]:
            # Voice credits = sqrt(votes_cast) = sqrt(sqrt(credits_spent))
            voice_credits = qv.votes_cast  # votes already represents voice

            if qv.option not in tally:
                tally[qv.option] = 0.0
            tally[qv.option] += voice_credits

        return tally

    def add_proposal_comment(
        self,
        vote_id: str,
        author_id: str,
        comment: str,
        reply_to: str | None = None,
    ) -> str:
        """
        Add a comment to a proposal discussion.

        Args:
            vote_id: Vote proposal ID
            author_id: Comment author
            comment: Comment text
            reply_to: Comment ID being replied to (for threading)

        Returns:
            Comment ID

        Example:
            >>> comment_id = system.add_proposal_comment("vote-123", "node-001", "Great idea!")
        """
        if vote_id not in self.proposal_discussions:
            self.proposal_discussions[vote_id] = []

        comment_id = hashlib.sha256(f"{author_id}{comment}{time.time()}".encode()).hexdigest()[:16]

        comment_data = {
            "id": comment_id,
            "author": author_id,
            "content": comment,
            "timestamp": time.time(),
            "reply_to": reply_to,
            "upvotes": 0,
            "downvotes": 0,
        }

        self.proposal_discussions[vote_id].append(comment_data)

        return comment_id

    def get_proposal_discussion(self, vote_id: str) -> list[dict]:
        """Get all comments for a proposal."""
        return self.proposal_discussions.get(vote_id, [])

    def execute_proposal(self, vote_id: str, executor_callback: callable) -> dict | None:
        """
        Execute an approved proposal.

        Args:
            vote_id: Vote to execute
            executor_callback: Function to execute the proposal action

        Returns:
            Execution result or None if not approved

        Example:
            >>> result = system.execute_proposal("vote-123", lambda: update_config())
        """
        vote = self.votes.get(vote_id)
        if not vote or vote.status != VoteStatus.APPROVED:
            return None

        try:
            result = executor_callback()

            execution_record = {
                "vote_id": vote_id,
                "executed_at": time.time(),
                "result": result,
                "status": "success",
            }
            self.executed_proposals.append(execution_record)

            return execution_record

        except Exception as e:
            execution_record = {
                "vote_id": vote_id,
                "executed_at": time.time(),
                "error": str(e),
                "status": "failed",
            }
            self.executed_proposals.append(execution_record)
            return execution_record

    def verify_vote_receipt(self, receipt_id: str) -> dict | None:
        """
        Verify a vote receipt.

        Args:
            receipt_id: Receipt to verify

        Returns:
            Receipt data if valid, None if invalid
        """
        receipt = self.vote_receipts.get(receipt_id)
        if not receipt:
            return None

        # Verify signature
        receipt_copy = receipt.copy()
        stored_sig = receipt_copy.pop("signature")
        receipt_data = json.dumps(receipt_copy, sort_keys=True).encode()
        expected_sig = hmac.new(self.signing_key, receipt_data, hashlib.sha256).hexdigest()[:32]

        if not hmac.compare_digest(expected_sig.encode(), stored_sig.encode()):
            return None

        return receipt

    def get_quadratic_stats(self, vote_id: str) -> dict:
        """Get quadratic voting statistics for a vote."""
        if vote_id not in self.quadratic_votes:
            return {}

        qvotes = self.quadratic_votes[vote_id]

        total_votes = sum(qv.votes_cast for qv in qvotes)
        total_credits = sum(qv.credits_spent for qv in qvotes)
        unique_voters = len({qv.voter_id for qv in qvotes})

        # Calculate Gini coefficient (measure of inequality)
        credits_per_voter: dict[str, float] = {}
        for qv in qvotes:
            credits_per_voter[qv.voter_id] = (
                credits_per_voter.get(qv.voter_id, 0) + qv.credits_spent
            )

        credit_values = sorted(credits_per_voter.values())
        n = len(credit_values)

        if n == 0:
            gini = 0.0
        else:
            cumsum = 0
            for i, val in enumerate(credit_values):
                cumsum += (2 * (i + 1) - n - 1) * val
            gini = cumsum / (n * sum(credit_values)) if sum(credit_values) > 0 else 0.0

        return {
            "total_quadratic_votes": total_votes,
            "total_credits_spent": total_credits,
            "unique_voters": unique_voters,
            "avg_credits_per_voter": total_credits / unique_voters if unique_voters > 0 else 0,
            "gini_coefficient": gini,  # Lower = more democratic
            "options": self.tally_quadratic_votes(vote_id),
        }
