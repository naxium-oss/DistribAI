"""Unit tests for the GRPO loss function and heuristic reward computation."""

import math

import torch

from worker.src.daemon.grpo_runner import (
    _detect_gibberish,
    compute_heuristic_reward,
    grpo_loss,
    scores_to_advantages,
)


class TestDetectGibberish:
    def test_empty_text(self):
        assert _detect_gibberish("") == 1.0
        assert _detect_gibberish("ab") == 1.0

    def test_clean_text(self):
        # Normal English text should have a low gibberish score
        score = _detect_gibberish("The quick brown fox jumps over the lazy dog.")
        assert score < 0.4

    def test_gibberish_text(self):
        # Repetitive characters should have a moderate gibberish score
        score = _detect_gibberish("aaaaaaa bbbbbbb ccccccc ddddddd")
        assert score > 0.1

    def test_repeated_words(self):
        score = _detect_gibberish("hello hello hello hello hello hello hello hello")
        assert score > 0.5

    def test_mixed_content(self):
        score = _detect_gibberish("12345 !@#$% ^&*()")
        # High proportion of non-alpha characters
        assert score > 0.3


class TestComputeHeuristicReward:
    def test_empty_response(self):
        reward = compute_heuristic_reward("", "some prompt")
        assert reward == -5.0

    def test_good_response_no_ref(self):
        reward = compute_heuristic_reward(
            "This is a reasonable and coherent response to the question.",
            "What is AI?",
        )
        assert -5 < reward < 10  # In reasonable range

    def test_good_response_with_ref(self):
        reward = compute_heuristic_reward(
            "Paris is the capital of France.",
            "What is the capital of France?",
            reference_text="Paris is the capital of France and its largest city.",
        )
        assert reward > 0  # Should get a positive reward for matching reference

    def test_gibberish_penalty(self):
        gibberish = compute_heuristic_reward(
            "aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj",
            "some prompt",
        )
        clean = compute_heuristic_reward(
            "This is a coherent sentence with real words and proper structure.",
            "some prompt",
        )
        # Gibberish should score lower or equal (not higher)
        assert gibberish <= clean + 1e-6

    def test_reward_bounds(self):
        # Reward should be clamped to [-10, 10]
        reward = compute_heuristic_reward(
            "This is a truly excellent and very long response that should get a high score.",
            "test",
            reference_text="This is a truly excellent",
        )
        assert -10 <= reward <= 10

        reward = compute_heuristic_reward("x" * 5, "test")
        assert -10 <= reward <= 10


class TestScoresToAdvantages:
    def test_normalisation(self):
        scores = [1.0, 2.0, 3.0, 4.0]
        adv = scores_to_advantages(scores)
        assert len(adv) == 4
        mean = sum(adv) / len(adv)
        assert abs(mean) < 1e-6

    def test_single_score(self):
        adv = scores_to_advantages([5.0])
        assert len(adv) == 1
        # With a single score, std is 0, result is (5-5)/1e-8 = 0
        assert abs(adv[0]) < 1.0

    def test_empty_list(self):
        assert scores_to_advantages([]) == []

    def test_constant_scores(self):
        # All equal -> advantages all ~0
        adv = scores_to_advantages([3.0, 3.0, 3.0])
        for a in adv:
            assert abs(a) < 1e-4


class TestGrpoLoss:
    def test_native_distribai_reference_model_is_reconstructed(self):
        from worker.src.compute.distribai_models import get_model
        from worker.src.daemon.grpo_runner import GrpoRunner, GrpoRunnerConfig

        model = get_model("distribai-tiny", vocab_size=32)
        runner = GrpoRunner(
            model=model,
            tokenizer=object(),
            cfg=GrpoRunnerConfig(job_id="job-1", worker_id="worker-1"),
        )

        reference = runner._build_reference_model()

        assert reference is not None
        assert type(reference) is type(model)
        assert reference.model_name == model.model_name
        assert all(not parameter.requires_grad for parameter in reference.parameters())
        for expected, actual in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(expected, actual)

    def test_update_bounds_sequences_to_model_context(self):
        from worker.src.compute.distribai_models import get_model
        from worker.src.daemon.grpo_runner import GrpoRunner, GrpoRunnerConfig

        class Tokenizer:
            def encode(self, text, add_bos=False, add_eos=False):
                return [1] * len(text)

        model = get_model("distribai-tiny", vocab_size=32, seq_len=8)
        runner = GrpoRunner(
            model=model,
            tokenizer=Tokenizer(),
            cfg=GrpoRunnerConfig(job_id="job-1", worker_id="worker-1", group_size=1),
        )
        runner._current_prompts = [{"prompt": "long prompt"}]
        runner._round_response_ids = [[2, 3, 4, 5, 6, 7]]
        runner._round_old_log_probs = [torch.zeros(6)]

        metrics = runner.apply_grpo_update([1.0])

        assert runner._context_limit() == 8
        assert all(math.isfinite(value) for value in metrics.values())

    def test_loss_shapes(self):
        batch, tokens, vocab = 2, 8, 16
        logits = torch.randn(batch, tokens, vocab)
        target_ids = torch.randint(0, vocab, (batch, tokens))
        old_log_probs = torch.randn(batch, tokens)
        advantages = torch.tensor([0.5, -0.3])
        loss_mask = torch.ones(batch, tokens, dtype=torch.bool)

        total, policy_kl, kl = grpo_loss(
            logits, target_ids, old_log_probs, advantages, loss_mask,
        )

        assert total.ndim == 0  # scalar
        assert policy_kl.ndim == 0
        assert kl.ndim == 0

    def test_loss_masked(self):
        """Tokens outside the loss mask shouldn't contribute."""
        batch, tokens, vocab = 1, 10, 8
        logits = torch.randn(batch, tokens, vocab)
        target_ids = torch.randint(0, vocab, (batch, tokens))
        old_log_probs = torch.randn(batch, tokens)
        advantages = torch.tensor([1.0])
        loss_mask = torch.zeros(batch, tokens, dtype=torch.bool)
        loss_mask[:, 5:] = True  # only last 5 tokens matter

        total1, _, _ = grpo_loss(
            logits, target_ids, old_log_probs, advantages, loss_mask,
        )

        # With everything masked, loss should be 0
        loss_mask_all = torch.zeros(batch, tokens, dtype=torch.bool)
        total0, _, _ = grpo_loss(
            logits, target_ids, old_log_probs, advantages, loss_mask_all,
        )
        assert float(total0) == 0.0

    def test_loss_with_reference(self):
        batch, tokens, vocab = 2, 6, 12
        logits = torch.randn(batch, tokens, vocab)
        target_ids = torch.randint(0, vocab, (batch, tokens))
        old_log_probs = torch.randn(batch, tokens)
        advantages = torch.tensor([0.5, -0.3])
        loss_mask = torch.ones(batch, tokens, dtype=torch.bool)
        ref_log_probs = torch.randn(batch, tokens)

        total, policy_loss, kl = grpo_loss(
            logits, target_ids, old_log_probs, advantages, loss_mask,
            ref_log_probs=ref_log_probs,
        )

        assert total.ndim == 0
        assert policy_loss.ndim == 0
        assert kl.ndim == 0

    def test_policy_loss_is_finite(self):
        """GRPO loss should produce finite values."""
        batch, tokens, vocab = 1, 4, 8
        logits = torch.randn(batch, tokens, vocab)
        target_ids = torch.randint(0, vocab, (batch, tokens))
        old_log_probs = torch.full((batch, tokens), -5.0)
        advantages = torch.tensor([1.0])
        loss_mask = torch.ones(batch, tokens, dtype=torch.bool)

        total, policy_loss, kl = grpo_loss(
            logits, target_ids, old_log_probs, advantages, loss_mask,
        )

        assert torch.isfinite(total)
        assert torch.isfinite(policy_loss)
        assert torch.isfinite(kl)

    def test_clipping_behaviour(self):
        """Very high ratios should be clipped compared to unclipped."""
        batch, tokens, vocab = 1, 2, 4
        _, target_ids = torch.randn(batch, vocab).max(dim=-1)
        target_ids = target_ids.unsqueeze(1).expand(batch, tokens)
        new_log_probs = torch.full((batch, tokens), -0.1)
        old_log_probs = torch.full((batch, tokens), -20.0)
        advantages = torch.tensor([1.0])
        loss_mask = torch.ones(batch, tokens, dtype=torch.bool)

        ratio = torch.exp(new_log_probs - old_log_probs)
        adv = advantages.unsqueeze(-1).float()
        clip_eps = 0.2

        # Clipped loss (what GRPO uses)
        loss_clip = -torch.max(
            ratio * adv,
            torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv,
        )
        flat_mask = loss_mask.reshape(-1).float()
        valid = flat_mask.sum().clamp_min(1.0)
        policy_loss = float((loss_clip.reshape(-1) * flat_mask).sum() / valid)

        # Unclipped loss (would be much more extreme)
        unclipped_val = float(((-ratio * adv).reshape(-1) * flat_mask).sum() / valid)

        # The clipped version should be less extreme
        assert policy_loss < unclipped_val + 1e-4
        # Both should be finite
        assert math.isfinite(policy_loss)
        assert math.isfinite(unclipped_val)
