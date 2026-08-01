"""
Proof-of-Computation (PoC) Challenge System (Production Implementation)
Implements PoC challenges for node registration as specified in README §6.1.
Challenges are strengthened to 6 hex digits to prevent spam registrations.
Flow:
1. Node requests challenge via POST /v1/nodes/challenge
2. Server generates random 6-digit hex challenge + computes expected answer
3. Node computes proof (SHA-256 of challenge + nonce that produces required prefix)
4. Server verifies proof before issuing JWT
"""

import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Challenge:
    """
    Represents a Proof-of-Computation challenge.

    Attributes:
        challenge_id: Unique identifier for the challenge
        challenge_hex: Hex string to solve
        difficulty: Number of leading zero bits required
        created_at: Creation timestamp
        expires_at: Expiration timestamp
        solved: Whether challenge has been solved
        attempts: Number of solution attempts
        expected_nonce: Expected nonce for validation

    Example:
        challenge = Challenge(
            challenge_id="ch-123",
            challenge_hex="abc123",
            difficulty=12,
            created_at=time.time(),
            expires_at=time.time() + 300
        )
    """

    challenge_id: str
    challenge_hex: str
    difficulty: int
    created_at: float
    expires_at: float
    solved: bool = False
    attempts: int = 0
    expected_nonce: str = field(default_factory=lambda: secrets.token_hex(8))

    def verify(self, nonce: str) -> bool:
        """
        Verify a proof-of-computation solution.

        The proof is valid if SHA-256(challenge_hex + nonce) has
        the required number of leading zero bits.

        Args:
            nonce: Nonce value to verify

        Returns:
            True if the proof is valid, False otherwise

        Example:
            >>> challenge = Challenge(...)
            >>> is_valid = challenge.verify("deadbeef")
            >>> print(f"Proof valid: {is_valid}")
        """
        data = f"{self.challenge_hex}{nonce}".encode()
        hash_result = hashlib.sha256(data).hexdigest()
        hash_int = int(hash_result, 16)
        leading_zeros = 256 - hash_int.bit_length()
        return leading_zeros >= self.difficulty


@dataclass(frozen=True)
class ChallengeIssued:
    """REST-facing challenge envelope for node registration."""

    challenge: str
    challenge_hex: str
    difficulty: int
    expires_at: float
    expires_in: int
    algorithm: str = "SHA-256"
    max_attempts: int = 100


class PoCChallengeManager:
    """
    Manages Proof-of-Computation challenges for node registration.

    Challenges prevent spam registrations by requiring nodes to solve
    a computational puzzle before receiving a JWT token.

    Attributes:
        difficulty: Number of leading zero bits required (default 12)
        challenges: Dictionary of active challenges by ID
        challenge_expiry: Time before challenge expires (default 300 seconds)
        challenge_length: Length of hex challenge string (default 6)
        max_attempts: Maximum solution attempts per challenge (default 100)

    Example:
        manager = PoCChallengeManager(difficulty=12)
        challenge = manager.create_challenge("node-123")
        solution = manager.solve_challenge(challenge.challenge_hex)
        verified = manager.verify_solution(challenge.challenge_id, solution)
    """

    DEFAULT_DIFFICULTY = 12
    CHALLENGE_EXPIRY_SECONDS = 300
    CHALLENGE_LENGTH = 6
    MAX_ATTEMPTS = 100

    def __init__(self, difficulty: int = DEFAULT_DIFFICULTY):
        """
        Initialize the challenge manager.

        Args:
            difficulty: Number of leading zero bits required in solution

        Example:
            >>> manager = PoCChallengeManager(difficulty=10)
        """
        self.difficulty = difficulty
        self.challenges: dict[str, Challenge] = {}
        self.completed_challenges: set[str] = set()

    def generate_challenge(self, node_id: str | None = None) -> dict[str, str] | ChallengeIssued:
        """
        Generate a new Proof-of-Computation challenge.

        Creates a random hex challenge that the client must solve by finding
        a nonce that produces a hash with the required number of leading zero bits.

        Returns:
            Dictionary with challenge data:
                - challenge_id: Unique challenge identifier
                - challenge_hex: Hex string to solve
                - difficulty: Required leading zero bits
                - difficulty_description: Human-readable description
                - expires_in: Seconds until expiration
                - algorithm: Hash algorithm used (SHA-256)
                - max_attempts: Maximum allowed attempts

        Example:
            >>> challenge = manager.generate_challenge()
            >>> print(f"Challenge: {challenge['challenge_hex']}")
            >>> print(f"Difficulty: {challenge['difficulty']} bits")
        """
        challenge_id = secrets.token_urlsafe(16)
        challenge_hex = secrets.token_hex(3)[: self.CHALLENGE_LENGTH]
        now = time.time()
        challenge = Challenge(
            challenge_id=challenge_id,
            challenge_hex=challenge_hex,
            difficulty=self.difficulty,
            created_at=now,
            expires_at=now + self.CHALLENGE_EXPIRY_SECONDS,
        )
        self.challenges[challenge_id] = challenge
        logger.info("Generated PoC challenge %s... (hex: %s)", challenge_id[:16], challenge_hex)
        payload = {
            "challenge_id": challenge_id,
            "challenge_hex": challenge_hex,
            "difficulty": self.difficulty,
            "difficulty_description": f"Find nonce such that SHA256({challenge_hex} + nonce) has {self.difficulty} leading zero bits",
            "expires_in": self.CHALLENGE_EXPIRY_SECONDS,
            "algorithm": "SHA-256",
            "max_attempts": self.MAX_ATTEMPTS,
        }
        if node_id is None:
            return payload
        return ChallengeIssued(
            challenge=challenge_id,
            challenge_hex=challenge_hex,
            difficulty=self.difficulty,
            expires_at=challenge.expires_at,
            expires_in=self.CHALLENGE_EXPIRY_SECONDS,
            max_attempts=self.MAX_ATTEMPTS,
        )

    def verify_solution(self, challenge_id: str, nonce: str) -> tuple[bool, str | None]:
        """
        Verify a Proof-of-Computation solution.

        Validates that the provided nonce produces a hash with the required
        number of leading zero bits when combined with the challenge hex.

        Args:
            challenge_id: The challenge identifier
            nonce: The nonce value provided by the client

        Returns:
            Tuple of (is_valid, error_message):
                - is_valid: True if solution is correct
                - error_message: Error description if invalid, None otherwise

        Example:
            >>> is_valid, error = manager.verify_solution(challenge_id, "deadbeef")
            >>> if is_valid:
            ...     print("Solution accepted")
            >>> else:
            ...     print(f"Error: {error}")
        """
        if challenge_id in self.completed_challenges:
            return False, "Challenge already used"
        challenge = self.challenges.get(challenge_id)
        if not challenge:
            return False, "Challenge not found or expired"
        if time.time() > challenge.expires_at:
            del self.challenges[challenge_id]
            return False, "Challenge expired"
        challenge.attempts += 1
        if challenge.attempts > self.MAX_ATTEMPTS:
            del self.challenges[challenge_id]
            return False, "Too many attempts"
        if challenge.verify(nonce):
            challenge.solved = True
            self.completed_challenges.add(challenge_id)
            del self.challenges[challenge_id]
            logger.info("PoC challenge %s... verified successfully", challenge_id[:16])
            return True, None
        return False, "Invalid proof"

    def verify_challenge(self, node_id: str, challenge: str, nonce: str) -> bool:
        """Verify a node registration challenge."""
        valid, _error = self.verify_solution(challenge, nonce)
        return valid

    def solve_challenge(self, challenge_id: str, max_attempts: int = 10000) -> str:
        """
        Solve a PoC challenge by brute force (for testing purposes).

        This method iteratively tests random nonces until finding one that
        produces a hash with the required number of leading zero bits.

        Args:
            challenge_id: The challenge identifier to solve
            max_attempts: Maximum number of nonce attempts before giving up

        Returns:
            Valid nonce that solves the challenge

        Raises:
            ValueError: If challenge not found
            RuntimeError: If solution not found within max attempts

        Example:
            >>> nonce = manager.solve_challenge(challenge_id, max_attempts=50000)
            >>> print(f"Found solution: {nonce}")
        """
        challenge = self.challenges.get(challenge_id)
        if not challenge:
            raise ValueError("Challenge not found")
        for _i in range(max_attempts):
            nonce = secrets.token_hex(8)
            if challenge.verify(nonce):
                return nonce
        raise RuntimeError(f"Could not find solution after {max_attempts} attempts")

    def cleanup_expired(self) -> int:
        """
        Clean up expired challenges from memory.

        Removes challenges that have exceeded their expiration time and
        trims the completed challenges set to prevent memory leaks.

        Returns:
            Number of expired challenges cleaned up

        Example:
            >>> cleaned = manager.cleanup_expired()
            >>> print(f"Cleaned {cleaned} expired challenges")
        """
        now = time.time()
        expired = [cid for cid, ch in self.challenges.items() if now > ch.expires_at]
        for cid in expired:
            del self.challenges[cid]
        if len(self.completed_challenges) > 10000:
            self.completed_challenges = set(list(self.completed_challenges)[-5000:])
        return len(expired)

    def get_stats(self) -> dict:
        """
        Get statistics about the challenge manager.

        Returns:
            Dictionary with:
                - active_challenges: Number of currently active challenges
                - completed_challenges: Number of successfully solved challenges
                - difficulty: Current difficulty setting
                - max_attempts: Maximum allowed attempts per challenge

        Example:
            >>> stats = manager.get_stats()
            >>> print(f"Active challenges: {stats['active_challenges']}")
        """
        return {
            "active_challenges": len(self.challenges),
            "completed_challenges": len(self.completed_challenges),
            "difficulty": self.difficulty,
            "expiry_seconds": self.CHALLENGE_EXPIRY_SECONDS,
        }


def solve_challenge(challenge_hex: str, difficulty: int, max_attempts: int = 1000000) -> str | None:
    """
    Solve a PoC challenge (client-side implementation for testing).
    Args:
        challenge_hex: The challenge hex string
        difficulty: Required leading zero bits
        max_attempts: Maximum attempts before giving up
    Returns:
        Valid nonce or None if not found
    """
    attempts = 0
    while attempts < max_attempts:
        nonce = secrets.token_hex(16)
        data = f"{challenge_hex}{nonce}".encode()
        hash_result = hashlib.sha256(data).hexdigest()
        hash_int = int(hash_result, 16)
        leading_zeros = 256 - hash_int.bit_length()
        if leading_zeros >= difficulty:
            return nonce
        attempts += 1
    return None
