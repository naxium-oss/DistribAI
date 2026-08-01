"""Ollama integration for distributed inference and benchmarking.

Handles model downloading, caching, and distributed execution.
Supports both broadcast (ensemble) and shard (throughput) modes.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OllamaConfig:
    """Configuration for Ollama inference."""

    model_name: str  # e.g., "llama3.1", "qwen2.5", "gemma2"
    mode: str  # "broadcast" or "shard"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: str | None = None

    # Cache settings
    cache_dir: Path = Path.home() / ".distribai" / "ollama-models"

    # Benchmark settings
    benchmark_mode: bool = False
    benchmark_dataset: str | None = None  # mmlu, humaneval, gsm8k


class OllamaRunner:
    """Manages Ollama model execution on nodes."""

    def __init__(self, work_dir: Path | None = None):
        self.work_dir = work_dir or Path.home() / ".distribai" / "ollama-jobs"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_available = self._check_ollama_installed()

    def _check_ollama_installed(self) -> bool:
        """Check if Ollama CLI is available."""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    async def ensure_model(self, model_name: str, node_config: dict) -> bool:
        """Ensure a model is downloaded and cached locally.

        Args:
            model_name: Name of the Ollama model
            node_config: Node configuration dict with storage preferences

        Returns:
            True if model is available
        """
        if not self.ollama_available:
            print("[Ollama] Ollama CLI is not installed or not responding")
            return False

        cache_dir = self._get_cache_dir(node_config)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Check if model already exists in cache
        model_path = cache_dir / model_name / "model.bin"
        if model_path.exists():
            print(f"[Ollama] Model {model_name} found in cache")
            return True

        # Ollama owns its model store; ask the CLI instead of guessing at a
        # cache filename. This also validates models installed outside our cache.
        try:
            result = subprocess.run(
                ["ollama", "show", model_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return True
        except (subprocess.SubprocessError, OSError):
            return False

        # Check storage preference
        storage_mode = node_config.get("storage_mode", "smart")

        if storage_mode == "never":
            print("[Ollama] Model caching disabled; using the Ollama-managed model store")

        if storage_mode == "ask":
            print("[Ollama] Storage mode 'ask' is non-interactive; proceeding with download")

        # Download model
        print(f"[Ollama] Downloading model {model_name}...")
        print("[Ollama] This may take several minutes depending on model size")

        try:
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout for large models
            )

            if result.returncode == 0:
                print(f"[Ollama] Model {model_name} downloaded successfully")
                return True
            else:
                print(f"[Ollama] Failed to download model: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("[Ollama] Model download timed out")
            return False
        except Exception as e:
            print(f"[Ollama] Error downloading model: {e}")
            return False

    async def run_inference(
        self,
        config: OllamaConfig,
        prompts: list[str],
        node_config: dict,
    ) -> dict:
        """Run inference on prompts.

        Args:
            config: Ollama configuration
            prompts: List of prompts to process
            node_config: Node configuration

        Returns:
            Dict with results and metadata
        """
        if not await self.ensure_model(config.model_name, node_config):
            return {
                "status": "failed",
                "error": f"Failed to load model {config.model_name}",
            }

        results = []

        for i, prompt in enumerate(prompts):
            try:
                # Build Ollama command
                cmd = [
                    "ollama",
                    "run",
                    config.model_name,
                    "--format",
                    "json",
                ]

                # Create prompt with system prompt if provided
                full_prompt = prompt
                if config.system_prompt:
                    full_prompt = f"System: {config.system_prompt}\n\nUser: {prompt}\n\nAssistant:"

                # Run inference
                result = subprocess.run(
                    cmd,
                    input=full_prompt,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout per prompt
                )

                if result.returncode == 0:
                    # Parse output
                    try:
                        response = json.loads(result.stdout)
                    except json.JSONDecodeError:
                        response = {"response": result.stdout.strip()}

                    results.append(
                        {
                            "prompt_index": i,
                            "status": "success",
                            "response": response,
                        }
                    )
                else:
                    results.append(
                        {
                            "prompt_index": i,
                            "status": "error",
                            "error": result.stderr,
                        }
                    )

            except subprocess.TimeoutExpired:
                results.append(
                    {
                        "prompt_index": i,
                        "status": "error",
                        "error": "Inference timeout",
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "prompt_index": i,
                        "status": "error",
                        "error": str(e),
                    }
                )

        return {
            "status": "completed",
            "model": config.model_name,
            "mode": config.mode,
            "total_prompts": len(prompts),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "results": results,
        }

    async def run_benchmark(
        self,
        config: OllamaConfig,
        node_config: dict,
    ) -> dict:
        """Run benchmark on model.

        Supports MMLU, HumanEval, GSM8K benchmarks.

        Args:
            config: Ollama configuration with benchmark settings
            node_config: Node configuration

        Returns:
            Dict with benchmark results
        """
        benchmark = config.benchmark_dataset or "mmlu"

        print(f"[Ollama] Running {benchmark} benchmark on {config.model_name}")

        # Load benchmark data supplied by the operator.
        benchmark_data = self._load_benchmark_data(benchmark, node_config)
        if not benchmark_data:
            return {
                "status": "failed",
                "error": f"No readable dataset supplied for benchmark: {benchmark}",
            }

        # Run inference on benchmark questions
        results = []
        correct = 0
        evaluated = 0
        total = len(benchmark_data)

        for item in benchmark_data:
            prompt = item["prompt"]
            expected = str(item.get("answer", "")).strip()
            if not expected:
                results.append({
                    "prompt": prompt,
                    "expected": None,
                    "actual": None,
                    "correct": None,
                    "status": "unevaluable",
                })
                continue

            result = await self.run_inference(
                config,
                [prompt],
                node_config,
            )

            if result["status"] == "completed" and result["results"]:
                response = result["results"][0].get("response", {})
                actual = response.get("response", "").strip().upper()

                # Check if correct (exact match or contains answer letter)
                is_correct = expected.upper() in actual or actual == expected.upper()

                evaluated += 1
                if is_correct:
                    correct += 1

                results.append(
                    {
                        "prompt": prompt,
                        "expected": expected,
                        "actual": actual,
                        "correct": is_correct,
                    }
                )

        accuracy = correct / evaluated if evaluated > 0 else None

        return {
            "status": "completed",
            "benchmark": benchmark,
            "model": config.model_name,
            "total_questions": total,
            "correct": correct,
            "evaluated": evaluated,
            "accuracy": accuracy,
            "results": results,
        }

    def _get_cache_dir(self, node_config: dict) -> Path:
        """Get cache directory based on node config."""
        custom_path = node_config.get("ollama_cache_path")
        if custom_path:
            return Path(custom_path)
        return Path.home() / ".distribai" / "ollama-models"

    def _load_benchmark_data(self, benchmark: str, node_config: dict) -> list[dict]:
        """Load benchmark records from an operator-provided JSON or JSONL file.

        The runner deliberately does not ship hidden sample questions: benchmark
        accuracy must describe the dataset selected by the operator. Each record
        must contain ``prompt`` and may contain ``answer``.
        """
        dataset_path = node_config.get("benchmark_data_path")
        if not dataset_path and Path(benchmark).is_file():
            dataset_path = benchmark
        if not dataset_path:
            return []

        path = Path(dataset_path).expanduser()
        if not path.is_file():
            return []
        try:
            if path.suffix.lower() == ".jsonl":
                records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                records = payload.get("records", payload) if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                return []
            return [
                record for record in records
                if isinstance(record, dict) and isinstance(record.get("prompt"), str)
            ]
        except (OSError, json.JSONDecodeError):
            return []

    def get_cached_models(self, node_config: dict) -> list[str]:
        """Get list of cached models."""
        cache_dir = self._get_cache_dir(node_config)
        if not cache_dir.exists():
            return []

        models = []
        for model_dir in cache_dir.iterdir():
            if model_dir.is_dir():
                models.append(model_dir.name)

        return sorted(models)

    def clear_cache(self, node_config: dict, model_name: str | None = None) -> bool:
        """Clear model cache.

        Args:
            node_config: Node configuration
            model_name: Specific model to clear, or None for all

        Returns:
            True if cache was cleared
        """
        cache_dir = self._get_cache_dir(node_config)

        try:
            if model_name:
                model_path = cache_dir / model_name
                if model_path.exists():
                    import shutil

                    shutil.rmtree(model_path)
                    print(f"[Ollama] Cleared cache for {model_name}")
            else:
                # Clear all
                if cache_dir.exists():
                    import shutil

                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    print("[Ollama] Cleared all model cache")

            return True
        except Exception as e:
            print(f"[Ollama] Failed to clear cache: {e}")
            return False


# Global instance
_runner: OllamaRunner | None = None


def get_ollama_runner() -> OllamaRunner:
    """Get or create global Ollama runner."""
    global _runner
    if _runner is None:
        _runner = OllamaRunner()
    return _runner
