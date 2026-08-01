"""Vote transaction mixin."""

import secrets
import time
from typing import Any


class VotesMixin:
    """Mixin for DBManager."""

    def record_vote(self, job_id: str, voter_id: str, credits: int) -> dict[str, Any]:
        now = int(time.time())
        vote_id = f"vote_{secrets.token_urlsafe(10)}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO vote_transactions (vote_id, job_id, voter_id, credits, created_ts)
                VALUES (?, ?, ?, ?, ?)
                """,
                (vote_id, job_id, voter_id, int(credits), now),
            )
            conn.execute(
                """
                UPDATE jobs
                SET total_votes = COALESCE(total_votes, 0) + ?, updated_ts = ?
                WHERE job_id = ?
                """,
                (int(credits), now, job_id),
            )
        self.refresh_queue_positions()
        return {
            "vote_id": vote_id,
            "job_id": job_id,
            "voter_id": voter_id,
            "credits": int(credits),
            "created_ts": now,
        }

    def get_votes(self, job_id: str | None = None) -> list[dict[str, Any]]:
        conn = self._get_conn()
        if job_id:
            cur = conn.execute(
                """
                SELECT vote_id, job_id, voter_id, credits, created_ts
                FROM vote_transactions
                WHERE job_id = ?
                ORDER BY created_ts DESC
                """,
                (job_id,),
            )
        else:
            cur = conn.execute(
                """
                SELECT vote_id, job_id, voter_id, credits, created_ts
                FROM vote_transactions
                ORDER BY created_ts DESC
                """
            )
        return [
            {
                "vote_id": row["vote_id"],
                "job_id": row["job_id"],
                "voter_id": row["voter_id"],
                "credits": row["credits"],
                "created_ts": row["created_ts"],
            }
            for row in cur.fetchall()
        ]

