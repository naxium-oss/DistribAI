# Golden training template

Real PyTorch regression smoke job for DistribAI:

```bash
python -m scripts.cli.distribai_cli submit ./examples/golden_template --steps 3
```

Requires PyTorch and `run.py` at the top level. The script trains a small linear model, reports measured losses, and outputs `results.json` when the sandbox sets `DISTRIBAI_RESULTS_PATH`.
