"""Unit tests for the GrpoCoordinator (orchestrator-side GRPO coordinator)."""

import json

import pytest

from services_python.grpo_coordinator import GrpoCoordinator, get_grpo_coordinator


@pytest.fixture
def coord():
    """Fresh GrpoCoordinator for each test."""
    return GrpoCoordinator()


@pytest.fixture
def sent_messages():
    """Collector for messages sent to workers."""
    msgs: dict[str, list] = {}

    def _send(worker_id: str, msg) -> None:
        msgs.setdefault(worker_id, []).append(msg)

    return msgs, _send


class TestGrpoCoordinator:
    def test_start_job(self, coord):
        """Starting a GRPO job creates a GrpoJobState with the right config."""
        state = coord.start_job(
            job_id="test-job-1",
            worker_ids=["worker-a", "worker-b", "worker-c"],
            grpo_config={"group_size": 8, "kl_coef": 0.05},
        )
        assert state.job_id == "test-job-1"
        assert state.worker_ids == ["worker-a", "worker-b", "worker-c"]
        assert state.config["group_size"] == 8
        assert state.config["kl_coef"] == 0.05
        assert state.config["clip_eps"] == 0.2  # default preserved
        assert state.status == "running"
        assert state.current_round == 0

    def test_start_job_default_config(self, coord):
        """Starting without grpo_config uses defaults."""
        state = coord.start_job("test-job-2", ["w1", "w2"])
        assert state.config["group_size"] == 4
        assert state.config["kl_coef"] == 0.1
        assert state.config["clip_eps"] == 0.2
        assert state.config["prompts_per_step"] == 2
        assert state.config["max_gen_tokens"] == 512

    def test_get_job(self, coord):
        """get_job returns None for unknown jobs and the state for known ones."""
        assert coord.get_job("nonexistent") is None
        state = coord.start_job("test-job-3", ["w1"])
        assert coord.get_job("test-job-3") is state

    def test_start_round(self, coord, sent_messages):
        """start_round sends GrpoRoundStart to all workers."""
        msgs, send_fn = sent_messages
        state = coord.start_job(
            "test-round-job", ["w1", "w2"],
            send_to_worker=send_fn,
        )

        result = coord.start_round(
            job_id="test-round-job",
            weights_blob_url="s3://weights/round1.pt",
            prompts_json_url="s3://prompts/batch1.json",
        )

        assert result is not None
        assert result.round_id == 1
        assert result.weights_blob_url == "s3://weights/round1.pt"
        assert result.prompts_json_url == "s3://prompts/batch1.json"
        assert state.current_round == 1

        # Both workers should have received a message
        assert len(msgs["w1"]) == 1
        assert len(msgs["w2"]) == 1
        # Messages should be ServerMessage protobufs with grpo_round_start
        msg1 = msgs["w1"][0]
        assert msg1.WhichOneof("payload") == "grpo_round_start"
        assert msg1.grpo_round_start.round_id == 1
        assert msg1.grpo_round_start.weights_blob_url == "s3://weights/round1.pt"

    def test_start_round_unknown_job(self, coord):
        """start_round on unknown job returns None."""
        result = coord.start_round("nonexistent", "url", "url")
        assert result is None

    def test_handle_reward_report(self, coord):
        """handle_reward_report buffers worker rewards correctly."""
        state = coord.start_job("test-rr-job", ["w1", "w2", "w3"])
        coord.start_round("test-rr-job", "wurl", "purl")

        coord.handle_reward_report("test-rr-job", "w1", 1, [0.5, 1.0, 1.5, 2.0])
        coord.handle_reward_report("test-rr-job", "w2", 1, [0.8, 1.2, 1.8, 2.2])
        coord.handle_reward_report("test-rr-job", "w3", 1, [0.3, 0.9, 1.1, 1.9])

        grpo_round = state.rounds[1]
        assert "w1" in grpo_round.worker_rewards
        assert "w2" in grpo_round.worker_rewards
        assert "w3" in grpo_round.worker_rewards
        assert len(grpo_round.worker_rewards["w1"]) == 4
        assert coord.all_workers_reported("test-rr-job", 1) is True

    def test_handle_reward_report_unknown_job(self, coord):
        """handle_reward_report silently drops reports for unknown jobs."""
        coord.handle_reward_report("ghost-job", "w1", 1, [1.0])
        # No error, no state created
        assert coord.get_job("ghost-job") is None

    def test_all_workers_reported(self, coord):
        """all_workers_reported returns True only when every worker has reported."""
        coord.start_job("test-all-report", ["w1", "w2", "w3"])
        coord.start_round("test-all-report", "wurl", "purl")

        assert coord.all_workers_reported("test-all-report", 1) is False

        coord.handle_reward_report("test-all-report", "w1", 1, [1.0, 2.0])
        assert coord.all_workers_reported("test-all-report", 1) is False

        coord.handle_reward_report("test-all-report", "w2", 1, [1.5, 2.5])
        assert coord.all_workers_reported("test-all-report", 1) is False

        coord.handle_reward_report("test-all-report", "w3", 1, [0.5, 3.0])
        assert coord.all_workers_reported("test-all-report", 1) is True

    def test_finalise_round_normalises_advantages(self, coord, sent_messages):
        """finalise_round computes group-normalised advantages correctly."""
        msgs, send_fn = sent_messages
        coord.start_job(
            "test-adv-job", ["w1", "w2"],
            send_to_worker=send_fn,
        )
        coord.start_round("test-adv-job", "wurl", "purl")

        coord.handle_reward_report("test-adv-job", "w1", 1, [1.0, 3.0])
        coord.handle_reward_report("test-adv-job", "w2", 1, [5.0, 7.0])

        result = coord.finalise_round("test-adv-job", 1, "s3://weights/new.pt")

        assert result is not None
        assert result.round_id == 1
        assert len(result.advantages) == 4  # 2 workers * 2 candiates = 4

        # Advantages should be normalised (zero-mean-ish, unit-variance-ish)
        import statistics
        mean = statistics.mean(result.advantages)
        stdev = statistics.stdev(result.advantages)
        assert abs(mean) < 0.5  # roughly zero mean
        assert 0.5 < stdev < 2.0  # roughly unit variance

        # Both workers should receive GrpoRoundComplete with their advantage slice
        msg_w1 = msgs["w1"][1]  # second message (after round_start)
        assert msg_w1.WhichOneof("payload") == "grpo_round_complete"
        msg_w2 = msgs["w2"][1]
        assert msg_w2.WhichOneof("payload") == "grpo_round_complete"

        # Each should have 2 advantages (2 candidates per worker)
        adv_w1 = json.loads(msg_w1.grpo_round_complete.advantages_json)
        adv_w2 = json.loads(msg_w2.grpo_round_complete.advantages_json)
        assert len(adv_w1) == 2
        assert len(adv_w2) == 2

        # The sum of all advantages across both workers should be ~0
        all_adv = adv_w1 + adv_w2
        assert abs(sum(all_adv)) < 0.01

    def test_finalise_round_partial_report(self, coord, sent_messages):
        """finalise_round works with partial reports (some workers missing)."""
        msgs, send_fn = sent_messages
        coord.start_job(
            "test-partial-job", ["w1", "w2", "w3"],
            send_to_worker=send_fn,
        )
        coord.start_round("test-partial-job", "wurl", "purl")

        coord.handle_reward_report("test-partial-job", "w1", 1, [2.0, 4.0])
        # w2 and w3 haven't reported

        result = coord.finalise_round("test-partial-job", 1, "s3://weights/new.pt")
        assert result is not None
        assert len(result.advantages) == 2  # only w1's rewards

    def test_finalise_round_unknown_job(self, coord):
        """finalise_round returns None for unknown job."""
        result = coord.finalise_round("ghost", 1, "url")
        assert result is None

    def test_fail_job(self, coord, sent_messages):
        """fail_job marks the job as failed and sends cancel to all workers."""
        msgs, send_fn = sent_messages
        coord.start_job("test-fail", ["w1", "w2"], send_to_worker=send_fn)
        coord.fail_job("test-fail", reason="test failure")

        state = coord.get_job("test-fail")
        assert state is not None
        assert state.status == "failed"

        # Both workers should get a cancel_job control message
        cancel_msgs_w1 = [m for m in msgs.get("w1", [])
                          if m.WhichOneof("payload") == "control"]
        cancel_msgs_w2 = [m for m in msgs.get("w2", [])
                          if m.WhichOneof("payload") == "control"]
        assert len(cancel_msgs_w1) >= 1
        assert len(cancel_msgs_w2) >= 1

    def test_get_status(self, coord):
        """get_status returns structured status for dashboard."""
        coord.start_job("test-status-job", ["w1", "w2"], grpo_config={"group_size": 6})
        coord.start_round("test-status-job", "wurl", "purl")
        coord.handle_reward_report("test-status-job", "w1", 1, [1.0, 2.0])
        coord.handle_reward_report("test-status-job", "w2", 1, [3.0, 4.0])
        coord.finalise_round("test-status-job", 1, "new_url")

        status = coord.get_status("test-status-job")
        assert status is not None
        assert status["job_id"] == "test-status-job"
        assert status["status"] == "running"
        assert status["current_round"] == 1
        assert len(status["workers"]) == 2
        assert status["config"]["group_size"] == 6
        assert 1 in status["rounds"]
        assert status["rounds"][1]["workers_reported"] == ["w1", "w2"]

    def test_get_status_unknown_job(self, coord):
        """get_status returns None for unknown job."""
        assert coord.get_status("ghost") is None

    def test_list_jobs(self, coord):
        """list_jobs returns all active GRPO jobs."""
        assert coord.list_jobs() == []
        coord.start_job("job-a", ["w1"])
        coord.start_job("job-b", ["w2", "w3"])
        jobs = coord.list_jobs()
        assert len(jobs) == 2
        job_ids = {j["job_id"] for j in jobs}
        assert job_ids == {"job-a", "job-b"}

    def test_singleton(self):
        """get_grpo_coordinator returns the same instance."""
        c1 = get_grpo_coordinator()
        c2 = get_grpo_coordinator()
        assert c1 is c2

    def test_reward_report_with_texts(self, coord):
        """handle_reward_report can optionally store candidate texts."""
        state = coord.start_job("texts-job", ["w1"])
        coord.start_round("texts-job", "wurl", "purl")

        texts = json.dumps(["candidate 1 text", "candidate 2 text"])
        coord.handle_reward_report(
            "texts-job", "w1", 1,
            [0.5, 1.5],
            candidate_texts_json=texts,
        )

        grpo_round = state.rounds[1]
        assert grpo_round.worker_texts["w1"] == ["candidate 1 text", "candidate 2 text"]

    def test_finalise_round_increments_steps(self, coord):
        """finalise_round increments total_steps_completed."""
        state = coord.start_job("steps-job", ["w1"])
        coord.start_round("steps-job", "wurl", "purl")
        coord.handle_reward_report("steps-job", "w1", 1, [1.0, 2.0, 3.0, 4.0])
        coord.finalise_round("steps-job", 1, "new_url")

        # 4 rewards / group_size=4 = 1 prompt
        assert state.total_steps_completed == 1

        coord.start_round("steps-job", "wurl2", "purl2")
        coord.handle_reward_report("steps-job", "w1", 2, [1.0, 2.0, 3.0, 4.0])
        coord.finalise_round("steps-job", 2, "new_url2")

        assert state.total_steps_completed == 2

    def test_finalise_round_auto_complete(self, coord):
        """Job auto-completes when target_steps is reached."""
        state = coord.start_job(
            "auto-complete", ["w1"],
            grpo_config={"group_size": 4, "target_steps": 2},
        )

        for _ in range(2):
            coord.start_round("auto-complete", "wurl", "purl")
            coord.handle_reward_report("auto-complete", "w1", state.current_round,
                                        [1.0] * 4)
            coord.finalise_round("auto-complete", state.current_round, "new_url")

        assert state.status == "completed"
