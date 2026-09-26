"""Framework adapters for trainers built on Megatron-LM, torchtitan or slime.

Those three frameworks normally own process orchestration. Each one calls
``torch.distributed.init_process_group``, wraps the model in tensor, pipeline or
FSDP parallel layers, and expects a multi-rank world. The grid owns orchestration
instead and runs one replica per worker, so a trainer that brings its own
process group is rejected by default (see ``capture.py``).

An adapter lets such a trainer through the capture stage on a single rank:

  * it recognizes the framework from the modules the trainer imported (or from
    the source, when the package is actually installed); and
  * it replaces the distributed bootstrap and the collectives with single-rank
    no-ops, so the script builds its model on one rank instead of stalling or
    failing on a world that does not exist.

Everything after that (model selection, TorchScript scripting, optimizer and
loss mapping, shapes) still goes through the normal fail-closed path, so an
adapter never fabricates a job it cannot actually run. When a framework
construct cannot be translated, the job is refused and the operator notice names
the framework.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrameworkSpec:
    """Recognized training framework and how it maps onto a grid job."""

    key: str
    display: str
    # Top-level import names that identify the framework once it is loaded.
    module_roots: tuple[str, ...]
    # Conservative source markers, only trusted when the package is installed.
    source_markers: tuple[str, ...]
    # What the framework calls single-rank execution, for the operator notice.
    single_rank: str


FRAMEWORKS: tuple[FrameworkSpec, ...] = (
    FrameworkSpec(
        key="megatron",
        display="Megatron-LM",
        module_roots=("megatron",),
        source_markers=("import megatron", "from megatron", "megatron.core", "megatron.training"),
        single_rank=(
            "Megatron-LM builds its model through megatron.training with "
            "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1; the "
            "translated job is that model on one rank, not the full parallel run."
        ),
    ),
    FrameworkSpec(
        key="torchtitan",
        display="torchtitan",
        module_roots=("torchtitan",),
        source_markers=("import torchtitan", "from torchtitan", "torchtitan."),
        single_rank=(
            "torchtitan drives training from a TrainSpec and a parallelism config; "
            "a translatable trainer must expose the model the same way a "
            "single-rank TrainSpec would, without DTensor or FSDP wrappers."
        ),
    ),
    FrameworkSpec(
        key="slime",
        display="slime",
        module_roots=("slime",),
        source_markers=("import slime", "from slime", "slime."),
        single_rank=(
            "slime orchestrates Megatron-LM training plus an SGLang rollout; only "
            "the policy model, optimizer and loss can become a grid job, and the "
            "rollout half has no single-rank equivalent here."
        ),
    ),
)

_BY_KEY = {spec.key: spec for spec in FRAMEWORKS}


def _spec_for_key(key: str) -> FrameworkSpec | None:
    return _BY_KEY.get(key)


def _package_available(spec: FrameworkSpec) -> bool:
    """True when at least one of the framework's module roots can be imported."""
    for root in spec.module_roots:
        try:
            if importlib.util.find_spec(root) is not None:
                return True
        except (ImportError, ValueError):
            continue
    return False


def detect_framework(loaded_modules: Iterable[str] = (), source: str = "") -> FrameworkSpec | None:
    """Identify a supported framework from loaded modules or the trainer source.

    Loaded modules are authoritative: if the trainer imported ``megatron``, the
    framework is known regardless of what the source text says. Source markers
    are a weaker signal and are only trusted when the package is actually
    installed, so a comment that merely mentions a framework cannot change how a
    plain distributed script is handled.
    """
    roots = set()
    for name in loaded_modules:
        if name:
            roots.add(name.split(".", 1)[0])
    for spec in FRAMEWORKS:
        if any(root in roots for root in spec.module_roots):
            return spec

    text = source or ""
    if text:
        for spec in FRAMEWORKS:
            if any(marker in text for marker in spec.source_markers) and _package_available(spec):
                return spec
    return None


def framework_hint(key: str) -> str:
    """One-sentence guidance for a framework key, or an empty string."""
    spec = _spec_for_key(key)
    return spec.single_rank if spec else ""


class _SingleRankGroup:
    """Stand-in for a process group with exactly one member."""

    def size(self) -> int:
        return 1

    def rank(self) -> int:
        return 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "SingleRankGroup()"


def _first_tensor(args: tuple[Any, ...]) -> Any:
    for a in args:
        if hasattr(a, "numel") and hasattr(a, "shape"):
            return a
    return None


def _identity_collective(name: str) -> Callable[..., Any]:
    """A collective on one rank is the identity, so return the tensor unchanged."""

    def fn(*args: Any, **kwargs: Any) -> Any:
        if name in {"barrier", "send", "recv"}:
            return None
        if name == "all_gather" and len(args) >= 2 and hasattr(args[0], "__iter__"):
            tensor_list, tensor = args[0], args[1]
            try:
                for t in tensor_list:
                    t.copy_(tensor)
            except Exception:  # noqa: BLE001 - best effort, capture continues
                pass
            return None
        if name in {"all_gather_into_tensor", "reduce_scatter_tensor"} and args:
            output = args[0]
            if name == "all_gather_into_tensor" and len(args) >= 2:
                try:
                    output.copy_(args[1])
                except Exception:  # noqa: BLE001
                    pass
            return output
        return _first_tensor(args)

    return fn


# Collectives that are identities with a single member. Anything not listed here
# is left untouched, and a trainer that needs it fails closed at scripting time.
_COLLECTIVES = (
    "barrier",
    "broadcast",
    "all_reduce",
    "all_gather",
    "all_gather_into_tensor",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "all_to_all",
    "send",
    "recv",
    "gather",
    "scatter",
)


def install_single_rank(torch_mod: Any) -> list[Callable[[], None]]:
    """Patch distributed bootstrap and collectives to single-rank no-ops.

    Returns a list of undo callables. ``torch.distributed.init_process_group`` is
    deliberately left alone: the capture harness owns that call so it can still
    reject a plain distributed trainer that is not on a supported framework.
    """
    dist = getattr(torch_mod, "distributed", None)
    if dist is None:
        return []

    unpatchers: list[Callable[[], None]] = []

    def patch(name: str, replacement: Any) -> None:
        if not hasattr(dist, name):
            return
        original = getattr(dist, name)

        def undo(_dist: Any = dist, _name: str = name, _orig: Any = original) -> None:
            setattr(_dist, _name, _orig)

        setattr(dist, name, replacement)
        unpatchers.append(undo)

    patch("is_initialized", lambda: True)
    patch("is_available", lambda: True)
    patch("get_rank", lambda *a, **k: 0)
    patch("get_world_size", lambda *a, **k: 1)
    patch("get_local_rank", lambda *a, **k: 0)
    patch("get_backend", lambda *a, **k: "gloo")
    patch("get_global_rank", lambda group=None, rank=0: rank)
    patch("new_group", lambda *a, **k: _SingleRankGroup())
    for name in _COLLECTIVES:
        patch(name, _identity_collective(name))

    return unpatchers


__all__ = [
    "FRAMEWORKS",
    "FrameworkSpec",
    "detect_framework",
    "framework_hint",
    "install_single_rank",
]
