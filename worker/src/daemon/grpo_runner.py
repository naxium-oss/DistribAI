"""Worker-side GRPO (Group Relative Policy Optimization) runner.

On the worker, the GRPO runner is responsible for:

  1. At round start (``GrpoRoundStart``), download the current policy weights
     and prompt batch from the orchestrator.
  2. For each prompt, generate ``group_size`` candidate responses with
     temperature sampling.
  3. Score each candidate with a reward model or built-in heuristic.
  4. Upload the per-candidate reward scalars to the orchestrator.
  5. On round complete, receive normalised advantages and apply the GRPO
     clipped-surrogate + KL penalty update.
  6. Download new weights and begin the next round.

Reference: https://arxiv.org/abs/2402.03300 (DeepSeekMath GRPO)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)


def _detect_gibberish(text: str) -> float:
    """Heuristic gibberish detection. Returns 0.0 (clean) to 1.0 (pure gibberish)."""
    if not text or len(text) < 5:
        return 1.0
    chars = list(text.lower())
    if len(chars) >= 3:
        trigrams = [tuple(chars[i: i + 3]) for i in range(len(chars) - 2)]
        unique_ratio = len(set(trigrams)) / len(trigrams) if trigrams else 1.0
        char_rep_score = 1.0 - unique_ratio
    else:
        char_rep_score = 0.5
    words = text.split()
    if len(words) >= 3:
        word_unique_ratio = len({w.lower() for w in words}) / len(words)
        word_rep_score = 1.0 - word_unique_ratio
    else:
        word_rep_score = 0.5
    if words:
        alpha_words = [
            w for w in words
            if len(w) <= 15 and sum(c.isalpha() for c in w) / max(len(w), 1) > 0.7
        ]
        alpha_ratio = len(alpha_words) / len(words)
        nonsense_score = 1.0 - alpha_ratio
    else:
        nonsense_score = 1.0
    return 0.3 * char_rep_score + 0.3 * word_rep_score + 0.4 * nonsense_score


def compute_heuristic_reward(
    generated_text: str,
    prompt: str,
    reference_text: str | None = None,
) -> float:
    """Compute a heuristic reward score for a generated response.

    Uses gibberish detection, length norms, and optional reference overlap.
    Returns a scalar in approximately [-5, 10].
    """
    if not generated_text or not generated_text.strip():
        return -5.0

    gibberish = _detect_gibberish(generated_text)
    if gibberish > 0.7:
        gibberish_penalty = -4.0 * gibberish
    elif gibberish > 0.4:
        gibberish_penalty = -2.0 * gibberish
    else:
        gibberish_penalty = 0.0

    # Length reward: prefer mid-length responses
    word_count = len(generated_text.split())
    if word_count < 5:
        len_reward = -2.0
    elif word_count < 20:
        len_reward = 0.0
    elif word_count < 200:
        len_reward = 1.0
    elif word_count < 500:
        len_reward = 0.5
    else:
        len_reward = -1.0

    # Reference overlap (if reference is provided, e.g. from dataset)
    ref_reward = 0.0
    if reference_text:
        gen_set = set(generated_text.lower().split())
        ref_set = set(reference_text.lower().split())
        if gen_set and ref_set:
            jaccard = len(gen_set & ref_set) / len(gen_set | ref_set)
            ref_reward = jaccard * 5.0

    total = gibberish_penalty + len_reward + ref_reward
    return max(-10.0, min(10.0, total))


def scores_to_advantages(scores: list[float]) -> list[float]:
    """Normalise scores to zero-mean unit-variance advantages."""
    if not scores:
        return []
    if len(scores) == 1:
        return [0.0]
    arr = torch.tensor(scores, dtype=torch.float32)
    mean = arr.mean()
    std = arr.std().clamp_min(1e-8)
    if torch.isnan(std) or torch.isinf(std):
        return [0.0] * len(scores)
    return ((arr - mean) / std).tolist()


def grpo_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.1,
    ref_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute GRPO clipped-surrogate + KL penalty loss.

    Args:
        logits: Model output logits (B, T, V).
        target_ids: Target token IDs (B, T).
        old_log_probs: Log probabilities of the behaviour policy (B, T).
        advantages: Per-token advantage scalar (B,) broadcastable to (B, 1).
        loss_mask: Boolean mask for valid tokens (B, T).
        clip_eps: PPO/GRPO clip epsilon.
        kl_coef: KL penalty coefficient.
        ref_log_probs: Optional reference model log probabilities for KL.

    Returns:
        Tuple of (total_loss, policy_loss, kl_loss) scalars.
    """
    flat_mask = loss_mask.reshape(-1).float()
    valid_count = flat_mask.sum().clamp_min(1.0)

    new_log_probs = (
        F.log_softmax(logits.float(), dim=-1)
        .gather(-1, target_ids.unsqueeze(-1))
        .squeeze(-1)
    )

    # Policy ratio
    old_lp = old_log_probs.to(new_log_probs.device)
    ratio = torch.exp(new_log_probs - old_lp)
    adv = advantages.to(ratio.device).unsqueeze(-1)

    # Clipped surrogate (GRPO uses MAX, not MIN)
    loss_clip = -torch.max(
        ratio * adv,
        torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv,
    )
    policy_loss = (loss_clip.reshape(-1) * flat_mask).sum() / valid_count

    # KL penalty
    if ref_log_probs is not None:
        ref_lp = ref_log_probs.to(new_log_probs.device)
        kl_per_tok = torch.exp(new_log_probs) * (new_log_probs - ref_lp)
        kl_loss = (kl_coef * (kl_per_tok.reshape(-1) * flat_mask).sum() / valid_count)
    else:
        # Approximate KL from the ratio
        kl_approx = (ratio - 1) - (new_log_probs - old_lp)
        kl_loss = kl_coef * (kl_approx.reshape(-1) * flat_mask).sum() / valid_count

    total_loss = policy_loss + kl_loss
    return total_loss, policy_loss.detach(), kl_loss.detach()


@dataclass
class GrpoRunnerConfig:
    """Per-job GRPO configuration on the worker."""

    job_id: str
    worker_id: str
    group_size: int = 4
    kl_coef: float = 0.1
    clip_eps: float = 0.2
    reward_scale: float = 1.0
    prompts_per_step: int = 2
    max_gen_tokens: int = 512
    gen_temperature: float = 0.9
    gen_top_k: int = 40

    # Reference model URL for KL computation (optional)
    ref_model_url: str = ""

    # Optimizer settings
    lr: float = 5e-5
    min_lr: float = 5e-6
    warmup_steps: int = 10
    max_grad_norm: float = 1.0

    hyperparams: dict[str, Any] = field(default_factory=dict)


class GrpoRunner:
    """Worker-side GRPO loop driver.

    Owns the local policy model and reference model. The orchestrator pushes
    ``round_start`` / ``round_complete`` events and the runner generates
    candidates, scores them, and applies the GRPO update.

    Attributes:
        model: The local policy model being trained.
        ref_model: Frozen reference model for KL computation (optional).
        cfg: Per-job GRPO configuration.
        device: Torch device for training.
        round_id: Current GRPO round.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        cfg: GrpoRunnerConfig,
        device: torch.device | str = "cpu",
        send_reward_report: Callable[[str, int, list[float], str | None], None] | None = None,
    ) -> None:
        """Initialize the GRPO runner.

        Args:
            model: Policy model to optimise.
            tokenizer: Tokenizer for encoding/decoding text.
            cfg: GRPO configuration.
            device: Torch device.
            send_reward_report: Async callback ``(job_id, round_id, rewards, texts_json)``.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = torch.device(device)
        self.model.to(self.device)
        self.send_reward_report = send_reward_report

        # Reference model (frozen copy of the policy at round start)
        self.ref_model: nn.Module | None = None

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

        # Round state
        self.round_id: int = 0
        self._current_prompts: list[dict[str, Any]] = []
        self._round_old_log_probs: list[torch.Tensor] = []
        self._round_response_ids: list[list[int]] = []
        self._round_prompt_ids: list[list[int]] = []

    def _context_limit(self) -> int:
        """Return the maximum token context supported by the policy model."""
        decoder = getattr(self.model, "model", self.model)
        position_embedding = getattr(decoder, "position_embedding", None)
        return max(1, int(getattr(position_embedding, "num_embeddings", 2048)))

    # -------------------------------------------------------------- public

    def on_round_start(
        self,
        round_id: int,
        weights_blob_url: str,
        prompts_json_url: str,
        grpo_config: dict[str, Any] | None = None,
    ) -> None:
        """Handle ``GrpoRoundStart`` from the orchestrator.

        Downloads the policy weights, builds a frozen reference model,
        loads the prompt batch, and prepares for candidate generation.
        """
        logger.info(
            "GRPO round %d starting (job=%s worker=%s, weights=%s, prompts=%s)",
            round_id,
            self.cfg.job_id,
            self.cfg.worker_id,
            weights_blob_url,
            prompts_json_url,
        )

        # Update config from round overrides
        if grpo_config:
            for key in ("group_size", "kl_coef", "clip_eps", "reward_scale",
                         "prompts_per_step", "max_gen_tokens", "gen_temperature", "gen_top_k"):
                if key in grpo_config:
                    setattr(self.cfg, key, grpo_config[key])

        # Download and load weights
        self._load_weights(weights_blob_url)

        # Build frozen reference model from current weights
        self.ref_model = self._build_reference_model()

        # Load prompts
        self._current_prompts = self._load_prompts(prompts_json_url)

        self.round_id = round_id
        logger.info(
            "GRPO round %d: loaded %d prompts, group_size=%d",
            round_id,
            len(self._current_prompts),
            self.cfg.group_size,
        )

    def generate_candidates_and_report(self) -> dict[str, Any]:
        """Generate candidates for all prompts, score them, and report rewards.

        This is the main work method. It:
        1. For each prompt, generates ``group_size`` candidates.
        2. Scores each candidate.
        3. Packages and sends the reward report to the orchestrator.

        Returns:
            Dict with keys: num_prompts, num_candidates, mean_reward, candidate_texts.
        """
        if not self._current_prompts:
            logger.warning("GRPO round %d: no prompts to generate candidates for", self.round_id)
            return {"num_prompts": 0, "num_candidates": 0, "mean_reward": 0.0, "candidate_texts": []}

        self.model.eval()
        all_rewards: list[float] = []
        all_texts: list[str] = []

        for prompt_data in self._current_prompts:
            prompt_text = prompt_data.get("prompt", "")
            reference_text = prompt_data.get("reference", "") or prompt_data.get("target", "")

            # Save old log probs for policy ratio computation
            prompt_ids = self._encode(prompt_text)
            self._round_prompt_ids.append(prompt_ids)

            for _ in range(self.cfg.group_size):
                response_text, response_ids, old_log_probs = self._generate_response(prompt_text)

                # Score with heuristic (and optionally with a reward model)
                reward = compute_heuristic_reward(
                    generated_text=response_text,
                    prompt=prompt_text,
                    reference_text=reference_text or None,
                )
                reward *= self.cfg.reward_scale
                all_rewards.append(reward)
                all_texts.append(response_text)

                # Store the sampled tokens and behaviour-policy log probs for the update.
                self._round_response_ids.append(response_ids)
                self._round_old_log_probs.append(old_log_probs.detach())

        mean_reward = sum(all_rewards) / max(len(all_rewards), 1)

        # Send reward report to orchestrator
        if self.send_reward_report:
            texts_json = json.dumps(all_texts)
            self.send_reward_report(
                self.cfg.job_id,
                self.round_id,
                all_rewards,
                texts_json,
            )

        logger.info(
            "GRPO round %d: generated %d candidates across %d prompts, mean_reward=%.4f",
            self.round_id,
            len(all_rewards),
            len(self._current_prompts),
            mean_reward,
        )

        return {
            "num_prompts": len(self._current_prompts),
            "num_candidates": len(all_rewards),
            "mean_reward": mean_reward,
            "candidate_texts": all_texts,
        }

    def apply_grpo_update(
        self,
        advantages: list[float],
    ) -> dict[str, float]:
        """Apply the GRPO update using normalised advantages from the orchestrator.

        Args:
            advantages: Per-candidate normalised advantages from orchestrator
                (length = num_prompts * group_size).

        Returns:
            Dict with keys: policy_loss, kl_loss, total_loss, grad_norm.
        """
        if not advantages:
            logger.warning("GRPO round %d: empty advantages, skipping update", self.round_id)
            return {"policy_loss": 0.0, "kl_loss": 0.0, "total_loss": 0.0, "grad_norm": 0.0}

        was_training = self.model.training
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = torch.tensor(0.0, device=self.device)
        total_policy_loss = 0.0
        total_kl_loss = 0.0
        num_train = 0

        candidate_count = min(
            len(advantages),
            len(self._round_response_ids),
            len(self._round_old_log_probs),
        )
        if candidate_count != len(advantages):
            logger.warning(
                "GRPO round %d: truncating %d advantages to %d available candidates",
                self.round_id,
                len(advantages),
                candidate_count,
            )

        adv_idx = 0
        for prompt_data in self._current_prompts:
            prompt_ids = self._encode(prompt_data.get("prompt", ""))
            for _ in range(self.cfg.group_size):
                if adv_idx >= candidate_count:
                    break
                response_ids = self._round_response_ids[adv_idx]
                old_lp = self._round_old_log_probs[adv_idx].to(self.device).reshape(-1)
                adv = torch.tensor([advantages[adv_idx]], device=self.device)

                # Recompute each response token with the same rolling context
                # used during generation. This keeps policy/reference logits
                # aligned even when the prompt plus response exceeds context.
                if response_ids and old_lp.numel():
                    token_count = min(len(response_ids), old_lp.numel())
                    response_ids = response_ids[:token_count]
                    response_logits_list = []
                    reference_logits_list = []
                    if self.ref_model is not None:
                        self.ref_model.eval()
                    for token_index, _token_id in enumerate(response_ids):
                        prefix = prompt_ids + response_ids[:token_index]
                        prefix = prefix[-self._context_limit():] or [0]
                        input_ids = torch.tensor(
                            prefix, dtype=torch.long, device=self.device
                        ).unsqueeze(0)
                        with torch.autocast(device_type=self.device.type, enabled=False):
                            logits, *_ = self._model_forward(self.model, input_ids)
                        response_logits_list.append(logits[:, -1:, :])
                        if self.ref_model is not None:
                            with torch.no_grad():
                                ref_logits, *_ = self._model_forward(
                                    self.ref_model, input_ids
                                )
                            reference_logits_list.append(ref_logits[:, -1:, :])
                    response_logits = torch.cat(response_logits_list, dim=1)
                    target_ids = torch.tensor(
                        response_ids, dtype=torch.long, device=self.device
                    ).unsqueeze(0)
                    old_targets = old_lp[:token_count].unsqueeze(0)
                    ref_targets = None
                    if reference_logits_list:
                        ref_logits = torch.cat(reference_logits_list, dim=1)
                        ref_targets = (
                            F.log_softmax(ref_logits.float(), dim=-1)
                            .gather(-1, target_ids.unsqueeze(-1))
                            .squeeze(-1)
                        )
                    loss_mask = torch.ones_like(
                        target_ids, dtype=torch.bool, device=self.device
                    )
                    ploss, p_loss, k_loss = grpo_loss(
                        response_logits,
                        target_ids,
                        old_targets,
                        adv,
                        loss_mask,
                        clip_eps=self.cfg.clip_eps,
                        kl_coef=self.cfg.kl_coef,
                        ref_log_probs=ref_targets,
                    )
                    total_loss = total_loss + ploss / max(candidate_count, 1)
                    total_policy_loss += float(p_loss)
                    total_kl_loss += float(k_loss)
                    num_train += 1
                adv_idx += 1

        self.model.train(was_training)
        if num_train > 0 and total_loss.requires_grad:
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm
            ).item()
            self.optimizer.step()

            logger.info(
                "GRPO round %d: update applied, policy_loss=%.4f kl_loss=%.4f grad_norm=%.4f",
                self.round_id,
                total_policy_loss / num_train,
                total_kl_loss / num_train,
                grad_norm,
            )

            return {
                "policy_loss": total_policy_loss / max(num_train, 1),
                "kl_loss": total_kl_loss / max(num_train, 1),
                "total_loss": float(total_loss.item()),
                "grad_norm": grad_norm,
            }

        return {"policy_loss": 0.0, "kl_loss": 0.0, "total_loss": 0.0, "grad_norm": 0.0}

    def on_round_complete(
        self,
        round_id: int,
        new_weights_blob_url: str,
        advantages: list[float],
    ) -> dict[str, Any]:
        """Handle ``GrpoRoundComplete``: apply GRPO update and load new weights.

        Args:
            round_id: Completed round ID.
            new_weights_blob_url: URL of the updated policy weights.
            advantages: Normalised advantages from the orchestrator.

        Returns:
            Dict with training metrics.
        """
        logger.info(
            "GRPO round %d complete (job=%s, fetching new weights %s, %d advantages)",
            round_id,
            self.cfg.job_id,
            new_weights_blob_url,
            len(advantages),
        )

        # Apply GRPO update with advantages
        metrics = self.apply_grpo_update(advantages)

        # Load the new canonical weights
        self._load_weights(new_weights_blob_url)

        # Clear round state
        self._round_old_log_probs.clear()
        self._round_response_ids.clear()
        self._round_prompt_ids.clear()
        self._current_prompts.clear()
        self.ref_model = None

        return metrics

    # -------------------------------------------------------------- helpers

    def _encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if hasattr(self.tokenizer, "encode"):
            return self.tokenizer.encode(text, add_bos=True, add_eos=False)
        # Fallback: byte encoding
        return list(text.encode("utf-8"))

    def _decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(ids, skip_special=False)
        return bytes(ids).decode("utf-8", errors="replace")

    def _generate_response(
        self,
        prompt: str,
    ) -> tuple[str, list[int], torch.Tensor]:
        """Generate a single candidate response.

        Returns:
            Tuple of (response_text, response_token_ids, log_probs_tensor).
        """
        prompt_ids = self._encode(prompt)
        x = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        generated_ids: list[int] = []
        log_probs_list: list[torch.Tensor] = []
        was_training = self.model.training
        self.model.eval()

        try:
            with torch.no_grad():
                for _ in range(self.cfg.max_gen_tokens):
                    ctx = x[:, -self._context_limit():]  # Truncate to max context
                    logits, *_ = self._model_forward(self.model, ctx)
                    next_logits = logits[:, -1, :] / max(self.cfg.gen_temperature, 1e-6)

                    if self.cfg.gen_top_k > 0:
                        v, idx = torch.topk(
                            next_logits, k=min(self.cfg.gen_top_k, next_logits.shape[-1])
                        )
                        probs = torch.softmax(v, dim=-1)
                        next_id = idx.gather(-1, torch.multinomial(probs, 1))
                        log_prob = torch.log_softmax(next_logits, dim=-1).gather(-1, next_id)
                    else:
                        probs = torch.softmax(next_logits, dim=-1)
                        next_id = torch.multinomial(probs, 1)
                        log_prob = torch.log(probs.gather(-1, next_id) + 1e-10)

                    generated_ids.append(int(next_id.item()))
                    log_probs_list.append(log_prob)
                    x = torch.cat([x, next_id], dim=1)

                    # Check for EOS
                    if hasattr(self.tokenizer, "eos_id"):
                        if next_id.item() == self.tokenizer.eos_id:
                            break
        finally:
            self.model.train(was_training)

        log_probs_cat = (
            torch.cat(log_probs_list) if log_probs_list else torch.zeros(1, device=self.device)
        )
        response_text = self._decode(generated_ids)
        return response_text, generated_ids, log_probs_cat

    def _model_forward(self, model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass through the model.

        Returns (logits, ...) tuple. Handles different model output formats.
        """
        out = model(x)
        if isinstance(out, (tuple, list)):
            return out
        return (out,)

    def _load_weights(self, url: str) -> None:
        """Load policy weights from a blob URL."""
        if not url:
            logger.warning("GRPO round %d: empty weights URL, skipping load", self.round_id)
            return

        try:
            from services_python.blob_loader import load_json_blob

            raw = load_json_blob(url)
            if raw is None:
                logger.error("GRPO round %d: failed to download weights from %s", self.round_id, url)
                return

            state_dict: dict[str, torch.Tensor] = {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, list):
                        state_dict[k] = torch.tensor(v, device=self.device)
                    elif isinstance(v, torch.Tensor):
                        state_dict[k] = v.to(self.device)
                    else:
                        logger.debug("GRPO: skipping non-tensor weight %s (%s)", k, type(v))

            if state_dict:
                self.model.load_state_dict(state_dict, strict=False)
                logger.info("GRPO round %d: loaded %d weight tensors", self.round_id, len(state_dict))
        except Exception as exc:
            logger.error("GRPO round %d: failed to load weights: %s", self.round_id, exc, exc_info=True)

    def _load_prompts(self, url: str) -> list[dict[str, Any]]:
        """Load prompt batch from a JSON URL."""
        if not url:
            logger.warning("GRPO round %d: empty prompts URL", self.round_id)
            return []

        try:
            from services_python.blob_loader import load_json_blob

            raw = load_json_blob(url)
            if raw is None:
                logger.error("GRPO round %d: failed to download prompts from %s", self.round_id, url)
                return []

            if isinstance(raw, dict):
                prompts = raw.get("prompts", raw.get("data", raw.get("rows", [])))
            elif isinstance(raw, list):
                prompts = raw
            else:
                prompts = []

            result: list[dict[str, Any]] = []
            for p in prompts[: self.cfg.prompts_per_step]:
                if isinstance(p, str):
                    result.append({"prompt": p})
                elif isinstance(p, dict):
                    result.append(p)
                else:
                    result.append({"prompt": str(p)})
            return result
        except Exception as exc:
            logger.error("GRPO round %d: failed to load prompts: %s", self.round_id, exc, exc_info=True)
            return []

    def _build_reference_model(self) -> nn.Module | None:
        """Create a frozen copy of the current policy as reference model."""
        try:
            ref = type(self.model)(**self._get_model_args(self.model))
            ref.load_state_dict(self.model.state_dict())
            ref.to(self.device)
            ref.eval()
            for param in ref.parameters():
                param.requires_grad = False
            logger.info("GRPO round %d: reference model created", self.round_id)
            return ref
        except Exception as exc:
            logger.warning("GRPO round %d: could not build reference model: %s", self.round_id, exc)
            return None

    def _get_model_args(self, model: nn.Module) -> dict[str, Any]:
        """Extract constructor args from a model (best-effort).

        Native DistribAI wrappers expose their public profile name while the
        actual vocab/context sizes live on the inner decoder. Preserve those
        values so the frozen GRPO reference has an identical architecture.
        """
        if hasattr(model, "model_name") and hasattr(model, "model"):
            decoder = model.model
            embedding = getattr(decoder, "embedding", None)
            position_embedding = getattr(decoder, "position_embedding", None)
            config = getattr(model, "config", None)
            if config is not None:
                raw = dict(vars(config))
                # ModelConfig may carry training-only knobs; architecture validation
                # accepts only the declarative family payload keys.
                allowed = {
                    "version",
                    "family",
                    "architecture",
                    "dim",
                    "n_unique_layers",
                    "n_logical_layers",
                    "n_heads",
                    "n_kv_heads",
                    "ffn_dim",
                    "seq_len",
                    "sliding_window",
                    "engram_dim",
                    "mhc_expansion",
                    "num_experts",
                    "top_k",
                    "conv_kernel",
                    "gru_layers",
                    "dropout",
                }
                custom = {key: value for key, value in raw.items() if key in allowed}
                if "family" not in custom and "architecture" in custom:
                    custom["family"] = custom["architecture"]
                return {
                    "model_name": model.model_name,
                    "vocab_size": getattr(embedding, "num_embeddings", 256),
                    "seq_len": getattr(position_embedding, "num_embeddings", 2048),
                    "custom_config": custom,
                }
            return {
                "model_name": model.model_name,
                "vocab_size": getattr(embedding, "num_embeddings", 256),
                "seq_len": getattr(position_embedding, "num_embeddings", 2048),
            }

        args: dict[str, Any] = {}
        source = model.config if hasattr(model, "config") else model
        for attr in ("vocab_size", "dim", "n_heads", "n_layers", "ffn_dim", "dropout"):
            if hasattr(source, attr):
                args[attr] = getattr(source, attr)
        return args
