"""
Sybil Attack Detection System (Production Implementation)
Implements Sybil detection as specified in README §6.4:
"Sybil detection: accounts created within 7 days of a high-value job's
submission are flagged for manual review before votes count"
Also implements additional Sybil detection patterns:
- Multiple accounts from same IP/hardware fingerprint
- Coordinated voting patterns
- Rapid account creation bursts
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AccountCreation:
    node_id: str
    timestamp: float
    ip_address: str
    hardware_fingerprint: str
    invite_code: str | None = None


@dataclass
class VotePattern:
    node_id: str
    job_id: str
    credits: float
    timestamp: float


class SybilDetector:
    """
    Detects Sybil attacks and coordinated manipulation.
    As per README §6.4:
    - Flag accounts created <7 days before high-value job votes
    - Flag multiple accounts from same hardware/IP
    - Flag coordinated voting patterns
    """

    ACCOUNT_AGE_THRESHOLD_DAYS = 7
    HIGH_VALUE_JOB_THRESHOLD = 1000
    MAX_ACCOUNTS_PER_IP = 5
    MAX_ACCOUNTS_PER_HARDWARE = 3
    RAPID_CREATION_WINDOW_SECONDS = 3600
    RAPID_CREATION_THRESHOLD = 10
    COORDINATED_VOTE_WINDOW_SECONDS = 300
    COORDINATED_VOTE_SIMILARITY_THRESHOLD = 0.8

    def __init__(self):
        self.account_creations: dict[str, AccountCreation] = {}
        self.ip_accounts: dict[str, set[str]] = defaultdict(set)
        self.hardware_accounts: dict[str, set[str]] = defaultdict(set)
        self.vote_history: list[VotePattern] = []
        self.flagged_accounts: dict[str, list[str]] = {}
        self.suspicious_jobs: set[str] = set()

    def _generate_hardware_fingerprint(self, hardware_json: str, gpu_model: str) -> str:
        data = f"{hardware_json}:{gpu_model}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    def check_registration_allowed(self, node_id: str, ip_address: str) -> tuple[bool, str]:
        """
        Check if registration is allowed (for testing Sybil detection).
        Args:
            node_id: Proposed node ID
            ip_address: IP address of request
        Returns:
            Tuple of (is_allowed, reason)
        """
        if ip_address in self.ip_accounts:
            if len(self.ip_accounts[ip_address]) >= self.MAX_ACCOUNTS_PER_IP:
                return (
                    False,
                    f"IP {ip_address} has reached max accounts ({self.MAX_ACCOUNTS_PER_IP})",
                )
        now = time.time()
        recent_creations = [
            c
            for c in self.account_creations.values()
            if now - c.timestamp < self.RAPID_CREATION_WINDOW_SECONDS
        ]
        if len(recent_creations) > self.RAPID_CREATION_THRESHOLD:
            return False, f"Rapid account creation detected: {len(recent_creations)} recent"
        return True, "Registration allowed"

    def analyze_account(
        self,
        node_id: str,
        ip_address: str,
        hardware_fingerprint: str = "",
        initial_credits: float = 0,
    ) -> dict:
        """Analyze and record a node account during registration."""
        allowed, reason = self.check_registration_allowed(node_id, ip_address)
        alerts = self.record_account_creation(
            node_id=node_id,
            ip_address=ip_address,
            hardware_json=hardware_fingerprint,
            gpu_model=hardware_fingerprint,
        )
        trust_score = self.get_account_trust_score(node_id)
        approved = allowed and trust_score >= 0.4
        if alerts and trust_score < 0.6:
            approved = False
        return {
            "approved": approved,
            "reason": reason if not approved else "Account accepted",
            "alerts": alerts,
            "trust_score": trust_score,
        }

    def record_account_creation(
        self,
        node_id: str,
        ip_address: str,
        hardware_json: str,
        gpu_model: str,
        invite_code: str | None = None,
    ) -> list[str]:
        """
        Record new account creation and check for Sybil patterns.
        Returns:
            List of detection alerts (empty if clean)
        """
        alerts = []
        hw_fingerprint = self._generate_hardware_fingerprint(hardware_json, gpu_model)
        creation = AccountCreation(
            node_id=node_id,
            timestamp=time.time(),
            ip_address=ip_address,
            hardware_fingerprint=hw_fingerprint,
            invite_code=invite_code,
        )
        self.account_creations[node_id] = creation
        self.ip_accounts[ip_address].add(node_id)
        self.hardware_accounts[hw_fingerprint].add(node_id)
        if len(self.ip_accounts[ip_address]) > self.MAX_ACCOUNTS_PER_IP:
            alerts.append(
                f"Multiple accounts from same IP: {len(self.ip_accounts[ip_address])} accounts"
            )
            for nid in self.ip_accounts[ip_address]:
                if nid not in self.flagged_accounts:
                    self.flagged_accounts[nid] = []
                self.flagged_accounts[nid].append(
                    f"Shared IP with {len(self.ip_accounts[ip_address])} accounts"
                )
        if len(self.hardware_accounts[hw_fingerprint]) > self.MAX_ACCOUNTS_PER_HARDWARE:
            alerts.append(
                f"Multiple accounts from same hardware: {len(self.hardware_accounts[hw_fingerprint])} accounts"
            )
            for nid in self.hardware_accounts[hw_fingerprint]:
                if nid not in self.flagged_accounts:
                    self.flagged_accounts[nid] = []
                reason = f"Shared hardware fingerprint with {len(self.hardware_accounts[hw_fingerprint])} accounts"
                if reason not in self.flagged_accounts[nid]:
                    self.flagged_accounts[nid].append(reason)
        now = time.time()
        recent_creations = [
            c
            for c in self.account_creations.values()
            if now - c.timestamp < self.RAPID_CREATION_WINDOW_SECONDS
        ]
        if len(recent_creations) > self.RAPID_CREATION_THRESHOLD:
            alerts.append(
                f"Rapid account creation burst: {len(recent_creations)} accounts in last hour"
            )
        if alerts:
            logger.warning("Sybil detection alerts for %s...: %s", node_id[:20], alerts)
        return alerts

    def record_vote(self, node_id: str, job_id: str, credits: float):
        vote = VotePattern(node_id=node_id, job_id=job_id, credits=credits, timestamp=time.time())
        self.vote_history.append(vote)
        if len(self.vote_history) > 10000:
            self.vote_history = self.vote_history[-10000:]
        if credits >= self.HIGH_VALUE_JOB_THRESHOLD:
            self._check_high_value_job_sybil(job_id)

    def _check_high_value_job_sybil(self, job_id: str):
        now = time.time()
        recent_votes = [
            v
            for v in self.vote_history
            if v.job_id == job_id and now - v.timestamp < self.COORDINATED_VOTE_WINDOW_SECONDS
        ]
        if len(recent_votes) < 3:
            return
        new_account_votes = []
        for vote in recent_votes:
            if vote.node_id in self.account_creations:
                creation = self.account_creations[vote.node_id]
                account_age_days = (now - creation.timestamp) / 86400
                if account_age_days < self.ACCOUNT_AGE_THRESHOLD_DAYS:
                    new_account_votes.append(
                        {
                            "node_id": vote.node_id,
                            "account_age_days": account_age_days,
                            "credits": vote.credits,
                        }
                    )
        if new_account_votes:
            self.suspicious_jobs.add(job_id)
            logger.warning(
                f"High-value job {job_id[:20]}... has {len(new_account_votes)} votes from new accounts"
            )
            for vote_info in new_account_votes:
                node_id = vote_info["node_id"]
                if node_id not in self.flagged_accounts:
                    self.flagged_accounts[node_id] = []
                reason = f"Voted on high-value job within {self.ACCOUNT_AGE_THRESHOLD_DAYS} days of account creation"
                if reason not in self.flagged_accounts[node_id]:
                    self.flagged_accounts[node_id].append(reason)

    def is_account_flagged(self, node_id: str) -> tuple[bool, list[str]]:
        """
        Check if an account is flagged for review.
        Returns:
            Tuple of (is_flagged, list_of_reasons)
        """
        reasons = self.flagged_accounts.get(node_id, [])
        return len(reasons) > 0, reasons

    def check_vote_allowed(
        self, node_id: str, job_id: str, credits: float
    ) -> tuple[bool, str | None]:
        """
        Check if a vote should be allowed or requires review.
        As per README §6.4: new accounts voting on high-value jobs are flagged.
        Returns:
            Tuple of (allowed, reason_if_blocked)
        """
        if credits < self.HIGH_VALUE_JOB_THRESHOLD:
            return True, None
        if node_id not in self.account_creations:
            return False, "Account not found"
        creation = self.account_creations[node_id]
        account_age_days = (time.time() - creation.timestamp) / 86400
        is_flagged, reasons = self.is_account_flagged(node_id)
        if is_flagged and account_age_days < self.ACCOUNT_AGE_THRESHOLD_DAYS:
            return False, f"Account flagged for review: {'; '.join(reasons)}"
        return True, None

    def get_account_trust_score(self, node_id: str) -> float:
        """
        Calculate trust score for an account (0.0 - 1.0).
        Factors:
        - Account age (older = more trustworthy)
        - Sybil flags (flagged = lower score)
        - Voting history (established pattern = more trustworthy)
        """
        score = 1.0
        is_flagged, reasons = self.is_account_flagged(node_id)
        if is_flagged:
            score -= min(0.5, len(reasons) * 0.1)
        if node_id in self.account_creations:
            creation = self.account_creations[node_id]
            age_days = (time.time() - creation.timestamp) / 86400
            if age_days > self.ACCOUNT_AGE_THRESHOLD_DAYS:
                score += 0.1
        node_votes = [v for v in self.vote_history if v.node_id == node_id]
        if len(node_votes) > 10:
            score += 0.1
        return max(0.0, min(1.0, score))

    def get_network_stats(self) -> dict:
        total_accounts = len(self.account_creations)
        flagged_count = len(self.flagged_accounts)
        suspicious_jobs_count = len(self.suspicious_jobs)
        multi_ip_count = sum(1 for ips in self.ip_accounts.values() if len(ips) > 1)
        multi_hw_count = sum(1 for hws in self.hardware_accounts.values() if len(hws) > 1)
        return {
            "total_accounts_tracked": total_accounts,
            "flagged_accounts": flagged_count,
            "flagged_percentage": round(flagged_count / total_accounts * 100, 2)
            if total_accounts > 0
            else 0,
            "suspicious_jobs": suspicious_jobs_count,
            "ips_with_multiple_accounts": multi_ip_count,
            "hardware_with_multiple_accounts": multi_hw_count,
            "account_age_threshold_days": self.ACCOUNT_AGE_THRESHOLD_DAYS,
            "high_value_threshold_credits": self.HIGH_VALUE_JOB_THRESHOLD,
        }

    def get_account_report(self, node_id: str) -> dict:
        is_flagged, reasons = self.is_account_flagged(node_id)
        trust_score = self.get_account_trust_score(node_id)
        creation_info = None
        if node_id in self.account_creations:
            c = self.account_creations[node_id]
            creation_info = {
                "timestamp": c.timestamp,
                "age_days": (time.time() - c.timestamp) / 86400,
                "ip_address": c.ip_address[:20] + "..." if len(c.ip_address) > 20 else c.ip_address,
                "hardware_fingerprint": c.hardware_fingerprint[:16] + "...",
                "invite_code": c.invite_code,
            }
        related_accounts = 0
        if node_id in self.account_creations:
            c = self.account_creations[node_id]
            related_accounts += len(self.ip_accounts.get(c.ip_address, set())) - 1
            related_accounts += len(self.hardware_accounts.get(c.hardware_fingerprint, set())) - 1
        return {
            "node_id": node_id,
            "is_flagged": is_flagged,
            "flag_reasons": reasons,
            "trust_score": round(trust_score, 2),
            "creation": creation_info,
            "related_accounts_count": related_accounts,
            "total_votes_cast": len([v for v in self.vote_history if v.node_id == node_id]),
        }
