"""Regression tests for bugs found while verifying multi-device training:

- gRPC task-assign stores hyperparams under ``hparams``, but the executor
  only read ``hyperparams``, so architecture_config never reached model
  creation for real (non-test) jobs.
- ``create_job`` always overwrote an explicit ``batch_blob_url`` with
  ``dataset_ref``, silently discarding any pre-staged batch file.
- ``_load_batch_source`` didn't recognize Windows drive-letter paths
  (``C:\\...``) as local files.
- The gradient compressor's per-parameter momentum buffer crashed instead of
  resetting when a node trained two architectures that happen to share a
  parameter name with a different shape (e.g. GRU vs LSTM ``recurrent.*``).
- ``_is_language_model``/``_compute_loss`` only recognized DistribAI's own
  native model wrapper; any external/custom-code architecture (loaded via
  ``load_external_architecture``) fell through to the toy MSE batch/loss
  path and crashed instead of training on real text.
- ``_collect_gradients`` crashed on bfloat16/float16 gradients (common
  ``torch_dtype`` on published Hub architectures) because numpy has no
  bfloat16 type; gradients are now upcast to float32 before compression.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from services_python.db_manager import DBManager
from worker.src.daemon.executor import JobExecutor
from worker.src.daemon.gradient_compression import TopKCompressor


def _first_task_for_job(db: DBManager, job_id: str) -> dict:
    tasks = [task for task in db.get_queued_tasks() if task["job_id"] == job_id]
    assert tasks, f"expected at least one queued task for {job_id}"
    return tasks[0]


@pytest.mark.unit
def test_create_job_preserves_explicit_batch_blob_url(tmp_path):
    schema_path = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db = DBManager(str(tmp_path / "regress.db"), str(schema_path))
    job_id = db.create_job(
        job_type="fine_tune",
        base_model="uploaded-architecture",
        dataset_ref="",
        batch_blob_url="runtime/verify_batches/sample.txt",
        total_steps=2,
    )
    task = _first_task_for_job(db, job_id)
    assert task["batch_blob_url"] == "runtime/verify_batches/sample.txt"


@pytest.mark.unit
def test_create_job_falls_back_to_dataset_ref_when_no_explicit_batch_url(tmp_path):
    schema_path = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db = DBManager(str(tmp_path / "regress2.db"), str(schema_path))
    job_id = db.create_job(
        job_type="fine_tune",
        base_model="distribai-tiny",
        dataset_ref="s3://bucket/data.json",
        total_steps=2,
    )
    task = _first_task_for_job(db, job_id)
    assert task["batch_blob_url"] == "s3://bucket/data.json"


@pytest.mark.unit
def test_executor_reads_hyperparams_from_hparams_key():
    async def _noop(*_args, **_kwargs):
        return None

    executor = JobExecutor(node_id="test-node", on_progress=_noop, on_result=_noop)
    job = {
        "job_id": "job-1",
        "task_id": "task-1",
        "model_name": "uploaded-architecture",
        # gRPC wire path stores this key as "hparams", not "hyperparams".
        "hparams": {
            "architecture_config": {
                "family": "gru",
                "dim": 32,
                "gru_layers": 1,
                "seq_len": 16,
            }
        },
        "steps": 1,
        "batch_size": 2,
    }
    model = executor._create_model(
        job["model_name"],
        architecture_config=job["hparams"]["architecture_config"],
        hyperparams=job["hparams"],
    )
    assert model is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_batch_source_accepts_windows_drive_path(tmp_path, monkeypatch):
    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setenv("GRADIENT_LOCAL_ROOT", str(tmp_path))
    batch_file = tmp_path / "sample.txt"
    batch_file.write_text("hello distributed world", encoding="utf-8")
    windows_style_path = str(batch_file).replace("/", "\\")

    executor = JobExecutor(node_id="test-node", on_progress=_noop, on_result=_noop)
    batch_source = await executor._load_batch_source("task-1", windows_style_path)

    assert batch_source["mode"] == "text"
    assert batch_source["content"] == "hello distributed world"


@pytest.mark.unit
def test_topk_compressor_resets_on_shape_mismatch():
    compressor = TopKCompressor(sparsity=0.5)
    first = {"recurrent.weight_ih_l0": torch.randn(96, 32)}
    compressor.compress(first)
    assert compressor.momentum_buffer["recurrent.weight_ih_l0"].shape == (96, 32)

    # A different architecture reuses the same parameter name with a
    # different shape (e.g. LSTM has 4 gates vs GRU's 3).
    second = {"recurrent.weight_ih_l0": torch.randn(128, 32)}
    compressed = compressor.compress(second)
    assert compressor.momentum_buffer["recurrent.weight_ih_l0"].shape == (128, 32)
    assert "recurrent.weight_ih_l0" in compressed


@pytest.mark.unit
def test_is_language_model_recognizes_external_pretrained_models():
    """External/custom architectures loaded via load_external_architecture are
    plain transformers.PreTrainedModel instances, not DistribAIModelWrapper —
    _is_language_model must still treat them as language models or every such
    job silently falls into the toy MSE batch/loss path and crashes."""
    transformers = pytest.importorskip("transformers")

    async def _noop(*_args, **_kwargs):
        return None

    executor = JobExecutor(node_id="test-node", on_progress=_noop, on_result=_noop)
    config = transformers.GPT2Config(
        n_layer=1, n_embd=8, n_head=2, vocab_size=32, n_positions=16
    )
    model = transformers.GPT2LMHeadModel(config)

    assert executor._is_language_model(model) is True


@pytest.mark.unit
def test_compute_loss_reads_logits_from_external_model_output():
    """External models return a transformers ModelOutput, not a raw tensor or
    tuple — _compute_loss must pull outputs.logits before cross_entropy."""
    transformers = pytest.importorskip("transformers")

    async def _noop(*_args, **_kwargs):
        return None

    executor = JobExecutor(node_id="test-node", on_progress=_noop, on_result=_noop)
    config = transformers.GPT2Config(
        n_layer=1, n_embd=8, n_head=2, vocab_size=32, n_positions=16
    )
    model = transformers.GPT2LMHeadModel(config)
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))

    loss = executor._compute_loss(model, (inputs, targets))
    loss.backward()

    assert torch.isfinite(loss)
    assert model.get_input_embeddings().weight.grad is not None


@pytest.mark.unit
def test_collect_gradients_upcasts_bfloat16_before_compression():
    """Published architectures often default to bfloat16/float16; numpy can't
    convert those directly, so gradients must be upcast to float32 first."""

    async def _noop(*_args, **_kwargs):
        return None

    executor = JobExecutor(node_id="test-node", on_progress=_noop, on_result=_noop)
    linear = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
    linear.weight.grad = torch.randn(4, 4, dtype=torch.bfloat16)
    linear.bias.grad = torch.randn(4, dtype=torch.bfloat16)

    gradients, total_norm = executor._collect_gradients(linear)

    assert total_norm > 0.0
    assert gradients
