#!/usr/bin/env bash
set -euo pipefail

# Build the DistribAI Node package for the current platform
# Usage: bash scripts/packaging/build_node_exe.sh

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PLATFORM="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
NAME="DistribAI-Node-${PLATFORM}-${ARCH}"

echo "============================================"
echo " Building DistribAI Node ($NAME)"
echo "============================================"

echo ""
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt -q
pip install pyinstaller pyinstaller-hooks-contrib -q

echo ""
echo "[2/3] Building package..."
pyinstaller \
  --name "$NAME" \
  --clean --noconfirm \
  --onedir \
  --console \
  --hidden-import "worker.src.daemon.run" \
  --hidden-import "worker.src.daemon.scheduler_config" \
  --hidden-import "worker.src.daemon.job_executor" \
  --hidden-import "worker.src.daemon.byzantine_detector" \
  --hidden-import "worker.src.daemon.credit_ledger" \
  --hidden-import "worker.src.daemon.voting_system" \
  --hidden-import "worker.src.daemon.gradient_compression" \
  --hidden-import "worker.src.daemon.ml_core" \
  --hidden-import "worker.src.distribai_proto" \
  --hidden-import "grpc" \
  --hidden-import "grpc.aio" \
  --hidden-import "torch" \
  --hidden-import "torch.cuda" \
  --hidden-import "numpy" \
  --hidden-import "psutil" \
  --hidden-import "aiohttp" \
  --collect-all "torch" \
  --add-data "worker/src/dashboard/static:static" \
  worker/src/daemon/gui_launcher.py

echo ""
echo "[3/3] Done!"
echo "Output: dist/$NAME/"
