"""Sandboxed inspector for a user Python trainer.

Run as a module::

    python -m trainer_translate.capture --trainer trainer.py --out jobdir

It imports the trainer as ``__main__`` (so ``if __name__ == "__main__":``
blocks run), watches torch for the objects that matter (modules, optimizers,
loss functions and data loaders), and stops training after
``DISTRIBAI_TRANSLATE_STEPS`` optimizer steps (default 1) instead of running the
whole job.

The captured architecture is scripted to TorchScript (``model.pt``) and the
first few batches are materialized to ``x.pt`` / ``y.pt``. Everything the
translator cannot handle is reported as a structured failure. The harness
never fabricates a job. ``capture.json`` is always written.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from typing import Any

from .frameworks import detect_framework, install_single_rank

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


class StopTrainingError(Exception):
    """Raised from the patched optimizer.step once we have seen enough."""


def _param_count(module: Any) -> int:
    try:
        return int(sum(p.numel() for p in module.parameters()))
    except Exception:
        return 0


def _shape(t: Any) -> list[int] | None:
    try:
        import torch

        if isinstance(t, torch.Tensor):
            return [int(d) for d in t.shape]
    except Exception:
        return None
    return None


def _write(out_dir: str, payload: dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "capture.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _fail(out_dir: str, status: str, stage: str, **kw: Any) -> int:
    payload = {"ok": False, "status": status, "stage": stage}
    payload.update(kw)
    _write(out_dir, payload)
    return 1


def _read_source(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def _framework_block(state: dict[str, Any]) -> dict[str, Any] | None:
    """JSON-safe provenance for the framework adapter, or None."""
    spec = state.get("framework")
    if spec is None:
        return None
    return {
        "key": spec.key,
        "display": spec.display,
        "mode": "single_rank",
        "detected_via": state.get("framework_source", "distributed init"),
    }


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def capture(
    trainer_path: str,
    out_dir: str,
    max_batches: int,
    input_shape_override: list[int] | None = None,
    target_shape_override: list[int] | None = None,
) -> int:
    import torch
    import torch.nn as nn
    import torch.utils.data as tud

    state: dict[str, Any] = {
        "modules": [],
        "optimizers": [],
        "losses": [],
        "loaders": [],
        "steps": 0,
        "distributed": False,
        "framework": None,
        "framework_source": None,
        "framework_adapted": False,
    }
    max_steps = int(os.environ.get("DISTRIBAI_TRANSLATE_STEPS", "1"))
    unpatchers: list[Any] = []
    warnings: list[str] = []
    trainer_source = _read_source(trainer_path)

    # --- wrap nn.Module construction so we see every layer the script builds ---
    orig_module_init = nn.Module.__init__

    def module_init(self, *a, **k):
        orig_module_init(self, *a, **k)
        state["modules"].append(self)

    nn.Module.__init__ = module_init
    unpatchers.append(lambda: setattr(nn.Module, "__init__", orig_module_init))

    # --- optimizers: record each instance and stop after N steps ---
    orig_opt_init = torch.optim.Optimizer.__init__

    def opt_init(self, params, defaults):
        orig_opt_init(self, params, defaults)
        state["optimizers"].append(self)

    torch.optim.Optimizer.__init__ = opt_init
    unpatchers.append(lambda: setattr(torch.optim.Optimizer, "__init__", orig_opt_init))

    orig_step = torch.optim.Optimizer.step

    def step(self, *a, **k):
        state["steps"] += 1
        if state["steps"] >= max_steps:
            raise StopTrainingError()
        return orig_step(self, *a, **k)

    torch.optim.Optimizer.step = step
    unpatchers.append(lambda: setattr(torch.optim.Optimizer, "step", orig_step))

    # --- loss functions (nn.*Loss is a Module, filter later by class name) ---
    orig_loss_init = nn.modules.loss._Loss.__init__

    def loss_init(self, *a, **k):
        orig_loss_init(self, *a, **k)
        state["losses"].append(type(self).__name__)

    nn.modules.loss._Loss.__init__ = loss_init
    unpatchers.append(lambda: setattr(nn.modules.loss._Loss, "__init__", orig_loss_init))

    # --- data loaders: record batch shape/size without draining the dataset ---
    orig_dl_init = tud.DataLoader.__init__

    def dl_init(self, *a, **k):
        orig_dl_init(self, *a, **k)
        info: dict[str, Any] = {
            "batch_size": getattr(self, "batch_size", None),
            "_loader": self,
        }
        try:
            info["dataset_len"] = len(self.dataset)  # type: ignore[arg-type]
        except Exception:
            info["dataset_len"] = None
        try:
            first = next(iter(self))
            tensors = first if isinstance(first, (list, tuple)) else [first]
            info["batch_shapes"] = [_shape(t) for t in tensors]
        except Exception as exc:  # a loader that cannot yield is not translatable
            info["batch_shapes"] = None
            info["peek_error"] = f"{type(exc).__name__}: {exc}"
        state["loaders"].append(info)

    tud.DataLoader.__init__ = dl_init
    unpatchers.append(lambda: setattr(tud.DataLoader, "__init__", orig_dl_init))

    # --- frameworks that bring their own process group ---------------------
    # Megatron-LM, torchtitan and slime always initialize torch.distributed. The
    # grid owns orchestration, so a trainer that does this is rejected by
    # default. When we recognize one of those frameworks, install a single-rank
    # shim up front so the model still gets built, and tag the job with the
    # framework instead of refusing it.
    def adapt_framework(spec: Any, via: str) -> None:
        state["framework"] = spec
        state["framework_source"] = via
        if state["framework_adapted"]:
            return
        state["framework_adapted"] = True
        unpatchers.extend(install_single_rank(torch))
        warnings.append(
            f"Detected {spec.display}; the trainer was adapted to a single rank for "
            "capture (the grid owns orchestration)."
        )

    detected = detect_framework(source=trainer_source)
    if detected is not None:
        adapt_framework(detected, "source import")

    # --- distributed init: reject a plain process group, adapt a framework ---
    if hasattr(torch, "distributed") and hasattr(torch.distributed, "init_process_group"):
        orig_init_pg = torch.distributed.init_process_group

        def init_pg(*a, **k):
            state["distributed"] = True
            spec = state["framework"] or detect_framework(
                loaded_modules=sys.modules.keys(), source=trainer_source
            )
            if spec is None:
                raise StopTrainingError()
            adapt_framework(spec, "distributed init")
            return None

        torch.distributed.init_process_group = init_pg
        unpatchers.append(lambda: setattr(torch.distributed, "init_process_group", orig_init_pg))

    # --- run the trainer as __main__ ---
    run_error: BaseException | None = None
    trainer_dir = os.path.dirname(os.path.abspath(trainer_path)) or "."
    try:
        spec = importlib.util.spec_from_file_location(
            "__main__", trainer_path, submodule_search_locations=[trainer_dir]
        )
        if spec is None or spec.loader is None:
            return _fail(
                out_dir, "unsupported", "import", missing_features=["module could not be loaded"]
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules["__main__"] = module
        sys.argv = [trainer_path]
        if trainer_dir not in sys.path:
            sys.path.insert(0, trainer_dir)
        spec.loader.exec_module(module)
    except StopTrainingError:
        pass  # expected: the patched optimizer.step bailed us out
    except SystemExit:
        pass  # a __main__ block called sys.exit(): treat it as script completion
    except BaseException as exc:  # noqa: BLE001 - reported verbatim, fail-closed
        run_error = exc
    finally:
        for undo in unpatchers:
            try:
                undo()
            except Exception:
                pass

    framework_block = _framework_block(state)

    def fail(status: str, stage: str, **kw: Any) -> int:
        kw.setdefault("framework", framework_block)
        return _fail(out_dir, status, stage, **kw)

    if state["distributed"] and state["framework"] is None:
        return fail(
            "unsupported",
            "capture",
            missing_features=["torch.distributed process groups"],
            detail="The grid owns process orchestration; trainers must not bring their own "
            "torch.distributed process group. Trainers on Megatron-LM, torchtitan or "
            "slime are recognized and adapted to a single rank instead.",
        )

    if run_error is not None:
        # The script raised. That is most likely a bug in the script itself.
        _write(
            out_dir,
            {
                "ok": False,
                "status": "needs_bugfix",
                "stage": "capture",
                "bug_summary": f"{type(run_error).__name__}: {run_error}",
                "traceback": "".join(
                    traceback.format_exception(type(run_error), run_error, run_error.__traceback__)
                ),
                "framework": framework_block,
            },
        )
        return 1

    # --- pick the top-level model: uncaptured parents with the most params ---
    modules = [m for m in state["modules"] if _param_count(m) > 0]
    nested_ids: set[int] = set()
    for m in modules:
        try:
            for sub in m.modules():
                if sub is not m:
                    nested_ids.add(id(sub))
        except Exception:
            continue
    candidates = [m for m in modules if id(m) not in nested_ids and hasattr(m, "forward")]
    if not candidates:
        return fail(
            "unsupported",
            "analyze",
            missing_features=["an nn.Module with trainable parameters and a forward()"],
        )
    model = max(candidates, key=_param_count)

    # --- script the architecture to TorchScript ---
    try:
        model = model.eval().cpu()
        scripted = torch.jit.script(model)
        model_path = os.path.join(out_dir, "model.pt")
        scripted.save(model_path)
    except Exception as exc:  # noqa: BLE001
        return fail(
            "unsupported",
            "script",
            missing_features=[f"TorchScript-scriptable model ({type(model).__name__})"],
            detail=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )

    # --- optimizer ---
    if not state["optimizers"]:
        return fail(
            "unsupported",
            "analyze",
            missing_features=["a torch.optim optimizer"],
            detail="No optimizer was constructed while inspecting the trainer.",
        )
    opt = state["optimizers"][0]
    opt_name = type(opt).__name__.lower()
    known = {"sgd": "sgd", "adam": "adam", "adamw": "adamw"}
    if opt_name not in known:
        return fail(
            "unsupported",
            "analyze",
            missing_features=[f"optimizer {type(opt).__name__}"],
        )
    defaults = getattr(opt, "defaults", {}) or {}

    # --- loss ---
    loss_map = {
        "MSELoss": "mse",
        "L1Loss": "l1",
        "CrossEntropyLoss": "cross_entropy",
    }
    loss_name = state["losses"][0] if state["losses"] else None
    if loss_name is None:
        warnings.append("No nn.*Loss instance found; defaulting to MSE loss.")
        loss = "mse"
    elif loss_name not in loss_map:
        return fail(
            "unsupported",
            "analyze",
            missing_features=[f"loss function {loss_name}"],
        )
    else:
        loss = loss_map[loss_name]
    target_dtype = "long" if loss == "cross_entropy" else "float"

    # --- shapes: prefer the first real batch, fall back to the module topology ---
    input_shape: list[int] | None = None
    target_shape: list[int] | None = None
    batch_size = 64
    if state["loaders"]:
        loader = state["loaders"][0]
        if loader.get("batch_size"):
            batch_size = int(loader["batch_size"])
        batch_shapes = loader.get("batch_shapes")
        if batch_shapes and len(batch_shapes) >= 2 and batch_shapes[0] and batch_shapes[1]:
            input_shape = batch_shapes[0][1:]
            target_shape = batch_shapes[1][1:]
        elif loader.get("peek_error"):
            warnings.append(f"DataLoader could not yield a batch: {loader['peek_error']}")
    inferred_in, inferred_out = _infer_shapes(model)
    # An explicit caller override wins over both the batch and the topology,
    # because the topology only knows channel/feature dims for Conv/Linear.
    if input_shape_override:
        input_shape = list(input_shape_override)
        warnings.append(f"input shape supplied by the caller: {input_shape}")
    elif input_shape is None:
        input_shape = inferred_in
    if target_shape_override:
        target_shape = list(target_shape_override)
        warnings.append(f"target shape supplied by the caller: {target_shape}")
    elif target_shape is None:
        target_shape = inferred_out
    if not input_shape:
        return fail(
            "unsupported",
            "analyze",
            missing_features=["input shape (no DataLoader batch and no Linear/Conv input dim)"],
            detail="Pass --input-shape, or use a DataLoader whose first batch is a pair of tensors.",
        )
    if not target_shape:
        return fail(
            "unsupported",
            "analyze",
            missing_features=["target shape (no DataLoader batch and no output dim)"],
        )

    # --- materialize the first batches to tensor files (data snapshot) ---
    data_note = "deterministic synthetic data"
    if state["loaders"]:
        saved = _save_batches(state["loaders"][0], out_dir, max_batches)
        if saved is None:
            warnings.append(
                "Could not materialize the DataLoader batches; the job will use "
                "deterministic synthetic data instead."
            )
        else:
            data_note = saved

    return _finish(
        out_dir,
        model=model,
        scripted_params=_param_count(model),
        model_path="model.pt",
        optimizer=known[opt_name],
        defaults=defaults,
        loss=loss,
        target_dtype=target_dtype,
        input_shape=input_shape,
        target_shape=target_shape,
        batch_size=batch_size,
        steps_observed=state["steps"],
        warnings=warnings,
        data_note=data_note,
        framework=framework_block,
    )


def _infer_shapes(model: Any) -> tuple[list[int] | None, list[int] | None]:
    """Best-effort shapes from the module topology (Linear/Conv2d only)."""
    import torch.nn as nn

    in_shape = out_shape = None
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if in_shape is None:
                in_shape = [int(m.in_features)]
            out_shape = [int(m.out_features)]
        elif isinstance(m, nn.Conv2d) and in_shape is None:
            in_shape = [int(m.in_channels)]  # spatial dims unknown without a batch
    return in_shape, out_shape


def _write_tensor_bin(path: str, t: Any) -> bool:
    """Write one tensor in the port's DAIT format (see torch_train.cpp)."""
    import struct

    import torch

    if not isinstance(t, torch.Tensor):
        return False
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.float32:
        dtype = 0
    elif t.dtype == torch.int64:
        dtype = 1
    else:
        return False
    dims = [int(d) for d in t.shape]
    if len(dims) > 255:
        return False
    with open(path, "wb") as fh:
        fh.write(b"DAIT")
        fh.write(struct.pack("<I", 1))
        fh.write(bytes([dtype, len(dims)]))
        for d in dims:
            fh.write(struct.pack("<Q", d))
        fh.write(t.numpy().tobytes())
    return True


def _save_batches(loader_info: dict[str, Any], out_dir: str, max_batches: int) -> str | None:
    """Materialize up to ``max_batches`` loader batches into x.bin / y.bin."""
    loader = loader_info.get("_loader")
    if loader is None:
        return None
    import torch

    xs, ys = [], []
    try:
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                return None
            x, y = batch[0], batch[1]
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                return None
            xs.append(x.detach().cpu())
            ys.append(y.detach().cpu())
    except Exception:
        return None
    if not xs:
        return None
    ok = _write_tensor_bin(os.path.join(out_dir, "x.bin"), torch.cat(xs, dim=0))
    ok = ok and _write_tensor_bin(os.path.join(out_dir, "y.bin"), torch.cat(ys, dim=0))
    if not ok:
        return None
    return f"materialized {len(xs)} batch(es) -> x.bin/y.bin"


def _finish(out_dir: str, **kw: Any) -> int:
    model = kw.pop("model")
    payload = {
        "ok": True,
        "stage": "emit",
        "model_class": type(model).__name__,
        **kw,
    }
    _write(out_dir, payload)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Capture a Python trainer for LibTorch translation")
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-batches", type=int, default=8)
    ap.add_argument("--input-shape", default="")
    ap.add_argument("--target-shape", default="")
    args = ap.parse_args(argv)

    def _dims(s: str) -> list[int] | None:
        s = (s or "").strip()
        if not s:
            return None
        try:
            return [int(x) for x in s.replace("x", ",").split(",") if x.strip()]
        except ValueError:
            return None

    trainer = os.path.abspath(args.trainer)
    try:
        return capture(
            trainer,
            os.path.abspath(args.out),
            args.max_batches,
            input_shape_override=_dims(args.input_shape),
            target_shape_override=_dims(args.target_shape),
        )
    except Exception as exc:  # noqa: BLE001 - last-resort fail-closed
        return _fail(
            os.path.abspath(args.out),
            "internal",
            "capture",
            detail=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )


if __name__ == "__main__":
    raise SystemExit(main())
