"""Integration test for the GRPO coordinator's round lifecycle.

Tests :class:`GrpoCoordinator` directly with a ``send_to_worker``
callback that captures protobuf messages instead of using real gRPC
streams or blob storage. Across multiple scenarios we verify:

* job start / round start / reward report / finalise round flow
* advantage normalisation across the worker group
* proper round_id tracking
* partial-worker timeout handling doesn't break the round
* cancellation removes state
* coordinator singleton lifecycle

This is the integration that catches wiring regressions between
:class:`GrpoCoordinator`, its reward-report contract, and the
``send_to_worker`` callback pattern.
"""

from __future__ import annotations

import json

import numpy as np

from services_python.grpo_coordinator import GrpoCoordinator, get_grpo_coordinator


def _capture_send(messages: list) -> callable:
    """Return a ``send_to_worker`` callback that appends to *messages*."""
    def _send(worker_id: str, msg) -> None:
        messages.append((worker_id, msg))
    return _send


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_coord_with_capture() -> tuple[GrpoCoordinator, list, str, callable]:
    coord = GrpoCoordinator()
    msgs: list = []
    send_cb = _capture_send(msgs)
    job_id = "integration-job"
    state = coord.start_job(
        job_id,
        worker_ids=["w-a", "w-b"],
        grpo_config={
            "group_size": 4,
            "kl_coef": 0.1,
            "clip_eps": 0.2,
            "prompts_per_step": 2,
        },
        send_to_worker=send_cb,
    )
    assert state is not None
    return coord, msgs, job_id, send_cb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGrpoCoordinatorIntegration:
    """In-process coordinator tests with captured protobuf messages."""

    def test_start_job(self) -> None:
        coord, msgs, job_id, _send_cb = _make_coord_with_capture()
        assert coord.get_job(job_id) is not None
        jobs = coord.list_jobs()
        assert any(j["job_id"] == job_id for j in jobs)
        assert msgs == []

    def test_start_round_broadcasts_to_all_workers(self) -> None:
        coord, msgs, job_id, _send_cb = _make_coord_with_capture()
        grpo_round = coord.start_round(job_id, "w://blob/1", "p://prompts/1")
        assert grpo_round is not None
        assert grpo_round.round_id == 1
        # 2 messages sent (one per worker)
        assert len(msgs) == 2
        for wid, msg in msgs:
            assert wid in ("w-a", "w-b")
            assert msg.HasField("grpo_round_start")
            rs = msg.grpo_round_start
            assert rs.job_id == job_id
            assert rs.round_id == 1
            assert rs.weights_blob_url == "w://blob/1"
            assert rs.prompts_json_url == "p://prompts/1"
            assert rs.config.group_size == 4

    def test_handle_reward_report_and_finalise(self) -> None:
        coord, msgs, job_id, _send_cb = _make_coord_with_capture()
        round_id = 1
        coord.start_round(job_id, "w://blob/1", "p://prompts/1")
        msgs.clear()  # ignore round_start messages

        # Both workers report 8 rewards each (4 candidates * 2 prompts)
        coord.handle_reward_report(job_id, "w-a", round_id, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        coord.handle_reward_report(job_id, "w-b", round_id, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

        assert coord.all_workers_reported(job_id, round_id)

        result = coord.finalise_round(job_id, round_id, "w://blob/new")
        assert result is not None
        assert result.round_id == 1
        assert result.mean_reward is not None
        assert result.advantages is not None

        # Should have sent 2 round_complete messages
        assert len(msgs) == 2
        for wid, msg in msgs:
            assert wid in ("w-a", "w-b")
            assert msg.HasField("grpo_round_complete")
            rc = msg.grpo_round_complete
            assert rc.job_id == job_id
            assert rc.round_id == 1
            assert rc.new_weights_blob_url == "w://blob/new"

    def test_advantages_normalised_across_group(self) -> None:
        coord, msgs, job_id, _send_cb = _make_coord_with_capture()
        round_id = 1
        coord.start_round(job_id, "w://blob/1", "p://prompts/1")
        msgs.clear()

        # w-a: lower rewards, w-b: higher rewards
        coord.handle_reward_report(job_id, "w-a", round_id, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        coord.handle_reward_report(job_id, "w-b", round_id, [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0])

        result = coord.finalise_round(job_id, round_id, "w://blob/new")
        assert result is not None

        # Overall mean should be 5.0
        assert abs(result.mean_reward - 5.0) < 1e-6

        # Overall advantages mean close to 0
        all_adv = result.advantages
        assert all_adv is not None
        assert abs(float(np.mean(all_adv))) < 1e-6

        # w-a advantages should be negative, w-b positive
        for wid, msg in msgs:
            rc = msg.grpo_round_complete
            adv = json.loads(rc.advantages_json)
            if wid == "w-a":
                assert all(a < 0 for a in adv), f"w-a should have negative advantages: {adv}"
            elif wid == "w-b":
                assert all(a > 0 for a in adv), f"w-b should have positive advantages: {adv}"

    def test_partial_report_then_finalise(self) -> None:
        """Test that finalise_round can complete with workers still pending."""
        coord, msgs, job_id, _send_cb = _make_coord_with_capture()
        round_id = 1
        coord.start_round(job_id, "w://blob", "p://prompts")
        msgs.clear()

        # Only w-a reports
        coord.handle_reward_report(job_id, "w-a", round_id, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        assert not coord.all_workers_reported(job_id, round_id)

        # But finalise should still succeed (logs a warning but completes)
        result = coord.finalise_round(job_id, round_id, "w://blob/new")
        assert result is not None
        assert result.round_id == 1

        # Worker count = 2, reported = 1. Advantages for w-b will be empty.
        # The method should handle this gracefully.
        assert len(msgs) == 2  # still broadcasts to both

    def test_cancel_job_sets_status_to_failed(self) -> None:
        coord, msgs, job_id, _send_cb = _make_coord_with_capture()
        assert any(j["job_id"] == job_id for j in coord.list_jobs())

        coord.fail_job(job_id)
        status = coord.get_status(job_id)
        assert status is not None
        assert status["status"] == "failed"

    def test_singleton_shared(self) -> None:
        c1 = get_grpo_coordinator()
        c2 = get_grpo_coordinator()
        assert c1 is c2

    def test_start_round_unknown_job_returns_none(self) -> None:
        coord = GrpoCoordinator()
        result = coord.start_round("nonexistent", "w://x", "p://x")
        assert result is None

    def test_handle_report_unknown_job_does_not_raise(self) -> None:
        coord = GrpoCoordinator()
        coord.handle_reward_report("ghost", "w-a", 1, [1.0, 2.0])  # should not raise

    def test_finalise_unknown_job_returns_none(self) -> None:
        coord = GrpoCoordinator()
        result = coord.finalise_round("ghost", 1, "w://x")
        assert result is None

    def test_constant_rewards_advantages_are_zero(self) -> None:
        """When all rewards are equal, advantages should be ~0."""
        coord, msgs, job_id, _send_cb = _make_coord_with_capture()
        round_id = 1
        coord.start_round(job_id, "w://blob", "p://prompts")
        msgs.clear()

        coord.handle_reward_report(job_id, "w-a", round_id, [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        coord.handle_reward_report(job_id, "w-b", round_id, [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

        result = coord.finalise_round(job_id, round_id, "w://blob/new")
        assert result is not None
        assert result.advantages is not None
        for a in result.advantages:
            assert abs(a) < 1e-6

    def test_round_id_increments(self) -> None:
        coord, msgs, job_id, _send_cb = _make_coord_with_capture()
        r1 = coord.start_round(job_id, "w://1", "p://1")
        r2 = coord.start_round(job_id, "w://2", "p://2")
        assert r1 is not None
        assert r2 is not None
        assert r1.round_id == 1
        assert r2.round_id == 2
        assert r2.weights_blob_url == "w://2"
