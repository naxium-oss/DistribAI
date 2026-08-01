"""GRPO (Group Relative Policy Optimization) Coordinator for DistribAI.

Orchestrator-side coordinator that manages distributed GRPO training rounds.
Workers generate N candidate responses per prompt, compute local reward scores,
and report them to the coordinator. The coordinator normalises advantages across
the entire worker group so per-candidate advantage reflects relative quality
within the global group. Workers then apply the GRPO clipped-surrogate + KL
penalty update with the normalised advantage.

Architecture:

    Orchestrator (GrpoCoordinator)
      │  ┌─ group 1: worker_A, worker_B  (share prompts, normalise together)
      │  └─ group 2: worker_C, worker_D, worker_E
      │
      ├── round_start(weights_blob, prompts_blob, config)  → all workers
      ├── collect reward_reports(worker_id, rewards[N*group_size]) ← workers
      │   └── normalise advantages across group
      └── round_complete(new_weights, advantages[N*group_size]) → all workers

Bandwidth per round: one weight blob down + N*group_size rewards up (scalars)
+ advantages array (scalars) + new weight blob down.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GrpoRound:
    """State for a single GRPO round."""

    round_id: int
    weights_blob_url: str
    prompts_json_url: str
    config: dict[str, Any]
    deadline_ts: float

    # Per-worker reward buffers
    worker_rewards: dict[str, list[float]] = field(default_factory=dict)
    worker_texts: dict[str, list[str]] = field(default_factory=dict)
    worker_report_ts: dict[str, float] = field(default_factory=dict)

    # Computed after aggregation
    advantages: list[float] | None = None
    candidates_per_worker: int = 0
    num_prompts: int = 0
    mean_reward: float = 0.0


@dataclass
class GrpoJobState:
    """Tracks all GRPO state for a single job."""

    job_id: str
    worker_ids: list[str]
    config: dict[str, Any]
    current_round: int = 0
    total_rounds: int = 0
    rounds: dict[int, GrpoRound] = field(default_factory=dict)
    status: str = "running"  # running, paused, completed, failed

    # Callbacks for pushing messages to workers
    send_to_worker: Callable[[str, Any], None] | None = None

    # Total steps counter (for reporting)
    total_steps_completed: int = 0


def _default_grpo_config() -> dict[str, Any]:
    return {
        "group_size": 4,
        "kl_coef": 0.1,
        "clip_eps": 0.2,
        "reward_scale": 1.0,
        "prompts_per_step": 2,
        "max_gen_tokens": 512,
        "gen_temperature": 0.9,
        "gen_top_k": 40,
    }


class GrpoCoordinator:
    """Manages distributed GRPO training across a group of workers.

    Usage::

        coord = GrpoCoordinator()
        state = coord.start_job(job_id, worker_ids, grpo_config, send_callback)
        # After each round:
        coord.handle_reward_report(job_id, worker_id, rewards, texts)
        # When ready to advance:
        coord.finalise_round(job_id)
    """

    def __init__(self) -> None:
        self._jobs: dict[str, GrpoJobState] = {}

    def start_job(
        self,
        job_id: str,
        worker_ids: list[str],
        grpo_config: dict[str, Any] | None = None,
        send_to_worker: Callable[[str, Any], None] | None = None,
    ) -> GrpoJobState:
        """Register a new GRPO job and begin round 1.

        Args:
            job_id: Unique job identifier.
            worker_ids: List of worker node IDs in the GRPO group.
            grpo_config: GRPO hyperparameter dict (merged with defaults).
            send_to_worker: Callable ``(worker_id, ServerMessage)`` for
                pushing messages to connected workers.

        Returns:
            The newly created GrpoJobState.
        """
        cfg = _default_grpo_config()
        if grpo_config:
            cfg.update(grpo_config)

        state = GrpoJobState(
            job_id=job_id,
            worker_ids=list(worker_ids),
            config=cfg,
            total_rounds=0,
            send_to_worker=send_to_worker,
        )
        self._jobs[job_id] = state
        logger.info(
            "[GrpoCoordinator] Started job=%s workers=%s config=%s",
            job_id,
            worker_ids,
            cfg,
        )
        return state

    def get_job(self, job_id: str) -> GrpoJobState | None:
        """Return the GrpoJobState for *job_id*, or None."""
        return self._jobs.get(job_id)

    def start_round(
        self,
        job_id: str,
        weights_blob_url: str,
        prompts_json_url: str,
    ) -> GrpoRound | None:
        """Begin a new GRPO round for *job_id*.

        Sends ``GrpoRoundStart`` to every worker in the group.

        Args:
            job_id: Job identifier.
            weights_blob_url: S3/HTTP URL of the current policy weights blob.
            prompts_json_url: S3/HTTP URL of the JSON prompt batch.

        Returns:
            The GrpoRound, or None if the job is unknown.
        """
        state = self._jobs.get(job_id)
        if state is None:
            logger.error("[GrpoCoordinator] Unknown job=%s", job_id)
            return None

        round_id = state.current_round + 1
        deadline_ts = time.time() + 300  # 5 min default

        grpo_round = GrpoRound(
            round_id=round_id,
            weights_blob_url=weights_blob_url,
            prompts_json_url=prompts_json_url,
            config=state.config,
            deadline_ts=deadline_ts,
            candidates_per_worker=state.config.get("group_size", 4)
            * state.config.get("prompts_per_step", 2),
        )

        state.rounds[round_id] = grpo_round
        state.current_round = round_id

        # Build protobuf message
        from worker.src.distribai_proto import distribai_pb2

        msg = distribai_pb2.ServerMessage(
            grpo_round_start=distribai_pb2.GrpoRoundStart(
                job_id=job_id,
                round_id=round_id,
                weights_blob_url=weights_blob_url,
                prompts_json_url=prompts_json_url,
                config=distribai_pb2.GrpoConfig(
                    group_size=state.config.get("group_size", 4),
                    kl_coef=state.config.get("kl_coef", 0.1),
                    clip_eps=state.config.get("clip_eps", 0.2),
                    reward_scale=state.config.get("reward_scale", 1.0),
                    ref_model_url=state.config.get("ref_model_url", ""),
                    prompts_per_step=state.config.get("prompts_per_step", 2),
                    max_gen_tokens=state.config.get("max_gen_tokens", 512),
                    gen_temperature=state.config.get("gen_temperature", 0.9),
                    gen_top_k=state.config.get("gen_top_k", 40),
                ),
                deadline_ts=int(deadline_ts),
            )
        )

        # Push to all workers
        if state.send_to_worker:
            for wid in state.worker_ids:
                try:
                    state.send_to_worker(wid, msg)
                except Exception as exc:
                    logger.warning(
                        "[GrpoCoordinator] Failed to send round_start to %s: %s",
                        wid,
                        exc,
                    )

        logger.info(
            "[GrpoCoordinator] job=%s round=%d started, %d workers, prompts=%s",
            job_id,
            round_id,
            len(state.worker_ids),
            prompts_json_url,
        )
        return grpo_round

    def handle_reward_report(
        self,
        job_id: str,
        worker_id: str,
        round_id: int,
        candidate_rewards: list[float],
        candidate_texts_json: str | None = None,
    ) -> None:
        """Process a ``GrpoRewardReport`` from a worker.

        Buffers the per-candidate rewards. Once all workers have reported
        (or timeout), call :meth:`finalise_round` to normalise advantages.

        Args:
            job_id: Job identifier.
            worker_id: Reporting worker.
            round_id: Round the worker is reporting for.
            candidate_rewards: Per-candidate reward scalars.
            candidate_texts_json: Optional JSON array of generated texts.
        """
        state = self._jobs.get(job_id)
        if state is None:
            logger.warning("[GrpoCoordinator] Unknown job=%s, dropping report", job_id)
            return

        grpo_round = state.rounds.get(round_id)
        if grpo_round is None:
            logger.warning(
                "[GrpoCoordinator] Unknown round=%d for job=%s, dropping",
                round_id,
                job_id,
            )
            return

        grpo_round.worker_rewards[worker_id] = list(candidate_rewards)
        grpo_round.worker_report_ts[worker_id] = time.time()

        if candidate_texts_json:
            try:
                texts = json.loads(candidate_texts_json)
                if isinstance(texts, list):
                    grpo_round.worker_texts[worker_id] = texts
            except (json.JSONDecodeError, TypeError):
                logger.debug("[GrpoCoordinator] Invalid candidate_texts_json from %s", worker_id)

        logger.debug(
            "[GrpoCoordinator] job=%s round=%d reward report from %s: %d rewards",
            job_id,
            round_id,
            worker_id,
            len(candidate_rewards),
        )

    def _count_reported_workers(self, state: GrpoJobState, grpo_round: GrpoRound) -> int:
        return sum(
            1 for wid in state.worker_ids if wid in grpo_round.worker_rewards
        )

    def all_workers_reported(self, job_id: str, round_id: int) -> bool:
        """Check if all workers have submitted reward reports for *round_id*."""
        state = self._jobs.get(job_id)
        if state is None:
            return False
        grpo_round = state.rounds.get(round_id)
        if grpo_round is None:
            return False
        return self._count_reported_workers(state, grpo_round) >= len(state.worker_ids)

    def finalise_round(
        self,
        job_id: str,
        round_id: int,
        new_weights_blob_url: str,
    ) -> GrpoRound | None:
        """Normalise advantages across the group and broadcast ``GrpoRoundComplete``.

        This computes group-normalised advantages::

            all_rewards = concat(worker_rewards for all workers)
            mean = mean(all_rewards)
            std = std(all_rewards) + 1e-8
            advantages = (all_rewards - mean) / std

        Then sends every worker its slice of the advantages plus the new
        canonical weights URL so it can apply the GRPO update.

        Args:
            job_id: Job identifier.
            round_id: Completed round ID.
            new_weights_blob_url: S3/HTTP URL of the post-update policy weights.

        Returns:
            The finalised GrpoRound, or None if job/round is unknown.
        """
        state = self._jobs.get(job_id)
        if state is None:
            logger.error("[GrpoCoordinator] Unknown job=%s", job_id)
            return None

        grpo_round = state.rounds.get(round_id)
        if grpo_round is None:
            logger.error("[GrpoCoordinator] Unknown round=%d for job=%s", round_id, job_id)
            return None

        reported = self._count_reported_workers(state, grpo_round)
        if reported < len(state.worker_ids):
            missing = sorted(set(state.worker_ids) - set(grpo_round.worker_rewards))
            logger.warning(
                "[GrpoCoordinator] job=%s round=%d finalising with %d/%d workers, "
                "missing=%s",
                job_id,
                round_id,
                reported,
                len(state.worker_ids),
                missing,
            )

        # Collect all rewards from all workers
        all_rewards: list[float] = []
        per_worker_slices: dict[str, list[float]] = {}
        for wid in state.worker_ids:
            rewards = grpo_round.worker_rewards.get(wid, [])
            per_worker_slices[wid] = rewards
            all_rewards.extend(rewards)

        # Normalise advantages across the group
        arr = np.array(all_rewards, dtype=np.float64)
        mean = float(np.mean(arr)) if len(arr) > 0 else 0.0
        std = float(np.std(arr)) + 1e-8
        if len(arr) <= 1 or np.isnan(std) or np.isinf(std):
            advantages_arr = np.zeros_like(arr)
        else:
            advantages_arr = (arr - mean) / std
        grpo_round.mean_reward = mean

        # Build a map: worker_id -> list[float] of normalised advantages
        worker_advantages: dict[str, list[float]] = {}
        idx = 0
        for wid in state.worker_ids:
            count = len(per_worker_slices.get(wid, []))
            worker_advantages[wid] = advantages_arr[idx : idx + count].tolist()
            idx += count

        grpo_round.advantages = advantages_arr.tolist()
        grpo_round.num_prompts = max(
            1, len(all_rewards) // max(state.config.get("group_size", 4), 1)
        )

        # Update state
        state.total_steps_completed += grpo_round.num_prompts

        from worker.src.distribai_proto import distribai_pb2

        # Send round complete to each worker with its advantage slice
        if state.send_to_worker:
            for wid in state.worker_ids:
                adv_slice = worker_advantages.get(wid, [])
                msg = distribai_pb2.ServerMessage(
                    grpo_round_complete=distribai_pb2.GrpoRoundComplete(
                        job_id=job_id,
                        round_id=round_id,
                        new_weights_blob_url=new_weights_blob_url,
                        advantages_json=json.dumps(adv_slice),
                        prompts_completed=grpo_round.num_prompts,
                        workers_contributed=reported,
                        mean_reward=mean,
                    )
                )
                try:
                    state.send_to_worker(wid, msg)
                except Exception as exc:
                    logger.warning(
                        "[GrpoCoordinator] Failed to send round_complete to %s: %s",
                        wid,
                        exc,
                    )

        logger.info(
            "[GrpoCoordinator] job=%s round=%d complete: %d workers, "
            "mean_reward=%.4f, %d prompts, new_weights=%s",
            job_id,
            round_id,
            reported,
            mean,
            grpo_round.num_prompts,
            new_weights_blob_url,
        )

        # Mark job as complete if target steps reached
        total_steps = state.config.get("target_steps", 0)
        if total_steps > 0 and state.total_steps_completed >= total_steps:
            state.status = "completed"
            logger.info("[GrpoCoordinator] job=%s completed (%d steps)", job_id, total_steps)

        return grpo_round

    def fail_job(self, job_id: str, reason: str = "") -> None:
        """Mark a GRPO job as failed and notify all workers."""
        state = self._jobs.get(job_id)
        if state is None:
            return
        state.status = "failed"
        logger.warning("[GrpoCoordinator] job=%s failed: %s", job_id, reason)

        # Send control message to cancel workers
        from worker.src.distribai_proto import distribai_pb2

        if state.send_to_worker:
            msg = distribai_pb2.ServerMessage(
                control=distribai_pb2.ControlMessage(
                    action="cancel_job",
                    target_id=job_id,
                )
            )
            for wid in state.worker_ids:
                try:
                    state.send_to_worker(wid, msg)
                except Exception:
                    pass

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        """Return a status dict for dashboard / API consumption."""
        state = self._jobs.get(job_id)
        if state is None:
            return None

        round_data: dict[int, dict[str, Any]] = {}
        for rid, rnd in state.rounds.items():
            round_data[rid] = {
                "round_id": rid,
                "workers_reported": list(rnd.worker_rewards.keys()),
                "mean_reward": rnd.mean_reward,
                "num_prompts": rnd.num_prompts,
                "candidates_per_worker": rnd.candidates_per_worker,
            }

        return {
            "job_id": state.job_id,
            "status": state.status,
            "current_round": state.current_round,
            "workers": list(state.worker_ids),
            "config": state.config,
            "total_steps_completed": state.total_steps_completed,
            "rounds": round_data,
        }

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all active GRPO jobs (for metrics endpoints)."""
        return [
            {
                "job_id": s.job_id,
                "status": s.status,
                "current_round": s.current_round,
                "workers": len(s.worker_ids),
                "steps": s.total_steps_completed,
            }
            for s in self._jobs.values()
        ]


# Global singleton for the orchestrator process
_grpo_coordinator: GrpoCoordinator | None = None


def get_grpo_coordinator() -> GrpoCoordinator:
    """Return the process-global GrpoCoordinator singleton."""
    global _grpo_coordinator
    if _grpo_coordinator is None:
        _grpo_coordinator = GrpoCoordinator()
    return _grpo_coordinator
