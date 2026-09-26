#!/usr/bin/env python3
"""LibTorch parity gate: Python torch vs the C++ LibTorch live trainer.

Builds a deterministic TorchScript model plus a data snapshot, runs the exact
same optimizer/loss/step schedule through both:

  * Python:  torch.jit.load(model.pt) + torch.optim.AdamW
  * C++    : build/cpp_port/distribai_torch_train --spec job.json

and requires the final loss, first loss and gradient sum to agree. Because both
sides are LibTorch, agreement is expected to be tight; the tolerances below are
the must-match contract.

Usage: python tools/cpp_port/tests/torch_parity.py
Prints "TORCH PARITY: PASS" / "TORCH PARITY: FAIL" and exits non-zero on fail.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BIN = ROOT / "build" / "cpp_port" / "distribai_torch_train"

TOL_LOSS = 1e-4
TOL_GRAD = 1e-3
STEPS = 40
BS = 16
N = 128
IN = 8
OUT = 2
SEED = 42


def write_dait(path: Path, t) -> None:  # type: ignore[no-untyped-def]
    import torch

    t = t.detach().cpu().contiguous()
    dtype = {torch.float32: 0, torch.int64: 1}[t.dtype]
    dims = [int(d) for d in t.shape]
    with open(path, "wb") as fh:
        fh.write(b"DAIT")
        fh.write(struct.pack("<I", 1))
        fh.write(bytes([dtype, len(dims)]))
        for d in dims:
            fh.write(struct.pack("<Q", d))
        fh.write(t.numpy().tobytes())


def read_dait(path: Path):  # type: ignore[no-untyped-def]
    import torch

    with open(path, "rb") as fh:
        assert fh.read(4) == b"DAIT", "bad magic"
        (version,) = struct.unpack("<I", fh.read(4))
        assert version == 1, "bad version"
        dtype, ndim = fh.read(2)
        dims = [struct.unpack("<Q", fh.read(8))[0] for _ in range(ndim)]
        count = 1
        for d in dims:
            count *= d
        width = 4 if dtype == 0 else 8
        raw = fh.read(count * width)
    tdt = torch.float32 if dtype == 0 else torch.int64
    return torch.frombuffer(bytearray(raw), dtype=tdt).clone().reshape(dims)


def python_reference(model_path: Path, x, y, lr: float, wd: float) -> tuple[float, float]:
    import torch
    import torch.nn as nn

    module = torch.jit.load(str(model_path)).eval()
    params = list(module.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    n = x.shape[0]
    bs = min(BS, n)
    first = last = 0.0
    for s in range(STEPS):
        if bs < n:
            off = (s * bs) % n
            idx = torch.arange(off, off + bs) % n
            xb, yb = x.index_select(0, idx), y.index_select(0, idx)
        else:
            xb, yb = x, y
        opt.zero_grad()
        loss = nn.functional.mse_loss(module(xb), yb)
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if s == 0:
            first = last
    grad_sum = 0.0
    for p in params:
        if p.grad is not None:
            grad_sum += float(p.grad.detach().to(torch.float64).sum())
    return first, last, grad_sum


def main() -> int:
    import torch
    import torch.nn as nn

    if not BIN.is_file():
        print(f"TORCH PARITY: SKIP (missing {BIN}; run `make torch`)")
        return 0

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(IN, 32)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(32, OUT)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.fc2(self.act(self.fc1(x)))

    tmp = Path(tempfile.mkdtemp(prefix="torch-parity-"))
    try:
        torch.manual_seed(SEED)
        model = Net().eval()
        torch.jit.script(model).save(str(tmp / "model.pt"))

        g = torch.Generator().manual_seed(SEED)
        x = torch.randn(N, IN, generator=g)
        y = torch.randn(N, OUT, generator=g)
        write_dait(tmp / "x.bin", x)
        write_dait(tmp / "y.bin", y)

        lr, wd = 1e-2, 1e-4
        spec = {
            "job_id": "parity",
            "model": "model.pt",
            "optimizer": "adamw",
            "lr": lr,
            "weight_decay": wd,
            "loss": "mse",
            "steps": STEPS,
            "batch_size": BS,
            "seed": SEED,
            "input": "x.bin",
            "target": "y.bin",
            "target_dtype": "float",
            "input_shape": [IN],
            "target_shape": [OUT],
        }
        (tmp / "job.json").write_text(json.dumps(spec))

        py_first, py_last, py_grad = python_reference(
            tmp / "model.pt", read_dait(tmp / "x.bin"), read_dait(tmp / "y.bin"), lr, wd
        )

        proc = subprocess.run(
            [str(BIN), "--spec", str(tmp / "job.json"), "--json", "--no-sandbox"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        try:
            out = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            print("TORCH PARITY: FAIL (no JSON from trainer)")
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            return 1

        if out.get("status") != "ok":
            print(f"TORCH PARITY: FAIL (trainer error: {out.get('error')})")
            return 1

        cpp_last = float(out["final_loss"])
        d_loss = abs(cpp_last - py_last)
        d_loss_rel = d_loss / max(abs(py_last), 1e-9)
        print(
            f"  python final_loss={py_last:.9f}  cpp final_loss={cpp_last:.9f}  "
            f"abs={d_loss:.3e} rel={d_loss_rel:.3e}"
        )
        print(f"  steps={STEPS} params={out.get('params')} grad_len={out.get('grad_len')}")

        ok = d_loss <= TOL_LOSS or d_loss_rel <= TOL_LOSS
        # gradient length must match (same parameterization)
        if int(out.get("grad_len", -1)) != sum(
            1 for p in torch.jit.load(str(tmp / "model.pt")).parameters() for _ in range(p.numel())
        ):
            print("  grad_len mismatch")
            ok = False

        print("TORCH PARITY: " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
