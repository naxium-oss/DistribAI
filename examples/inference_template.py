#!/usr/bin/env python3
"""Run real text generation for a DistribAI inference job.

Provide ``config.json`` with a Hugging Face ``base_model`` and either
``input.json`` (``{"prompts": [...]}``) or ``input.txt`` beside this file.
The worker must have ``transformers`` and ``torch`` installed and be able to
load the configured model from its local cache or the configured model hub.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent


def _load_json(filename: str) -> dict[str, Any]:
    path = ROOT / filename
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{filename} must contain a JSON object")
    return value


def load_config() -> dict[str, Any]:
    """Load model configuration from ``config.json``."""
    return _load_json("config.json")


def load_hyperparams() -> dict[str, Any]:
    """Load generation parameters from ``hyperparams.json``."""
    return _load_json("hyperparams.json")


def load_input_data() -> list[str]:
    """Load prompts from an operator-provided JSON or text input file."""
    input_json = ROOT / "input.json"
    input_txt = ROOT / "input.txt"
    if input_json.exists():
        with input_json.open(encoding="utf-8") as handle:
            data = json.load(handle)
        prompts = data.get("prompts") if isinstance(data, dict) else None
        if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
            raise ValueError("input.json must contain a string list named 'prompts'")
        return [prompt for prompt in prompts if prompt.strip()]
    if input_txt.exists():
        return [line.strip() for line in input_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise FileNotFoundError("Provide input.json or input.txt beside inference_template.py")


def generate(model_name: str, prompts: list[str], params: dict[str, Any]) -> list[dict[str, str]]:
    """Generate completions with the configured local/remote Transformers model."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Inference requires torch and transformers; install the worker extras") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    max_tokens = int(params.get("max_tokens", 100))
    temperature = float(params.get("temperature", 0.7))
    top_p = float(params.get("top_p", 0.9))
    results: list[dict[str, str]] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "completion": completion})
    return results


def main() -> None:
    config = load_config()
    params = load_hyperparams()
    model_name = str(config.get("base_model", "")).strip()
    if not model_name:
        raise ValueError("config.json must define a non-empty 'base_model'")
    prompts = load_input_data()
    if not prompts:
        raise ValueError("Input data contains no prompts")

    task_id = os.getenv("DISTRIBAI_TASK_ID", "unknown")
    job_id = os.getenv("DISTRIBAI_JOB_ID", "unknown")
    results = generate(model_name, prompts, params)
    output = {
        "status": "completed",
        "task_id": task_id,
        "job_id": job_id,
        "model": model_name,
        "results": results,
        "num_prompts": len(prompts),
    }
    (ROOT / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    (ROOT / "completions.txt").write_text(
        "\n".join(f"Prompt: {item['prompt']}\nCompletion: {item['completion']}" for item in results),
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Inference failed: {exc}")
        raise SystemExit(1) from exc
