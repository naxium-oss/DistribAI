@echo off
REM Build DistribAI Node Windows EXE locally
REM Run from repo root on Windows with Python 3.11+

echo ============================================
echo  Building DistribAI Node Windows EXE
echo ============================================

echo.
echo [1/3] Installing dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install pyinstaller pyinstaller-hooks-contrib

echo.
echo [2/3] Building EXE...
pyinstaller ^
  --name "DistribAI-Node" ^
  --clean --noconfirm ^
  --onedir ^
  --console ^
  --hidden-import "worker.src.daemon.run" ^
  --hidden-import "worker.src.daemon.scheduler_config" ^
  --hidden-import "worker.src.daemon.job_executor" ^
  --hidden-import "worker.src.daemon.byzantine_detector" ^
  --hidden-import "worker.src.daemon.credit_ledger" ^
  --hidden-import "worker.src.daemon.voting_system" ^
  --hidden-import "worker.src.daemon.gradient_compression" ^
  --hidden-import "worker.src.daemon.ml_core" ^
  --hidden-import "worker.src.distribai_proto" ^
  --hidden-import "grpc" ^
  --hidden-import "grpc.aio" ^
  --hidden-import "torch" ^
  --hidden-import "torch.cuda" ^
  --hidden-import "numpy" ^
  --hidden-import "psutil" ^
  --hidden-import "aiohttp" ^
  --hidden-import "pywebview" ^
  --collect-all "torch" ^
  --add-data "worker/src/dashboard/static;static" ^
  worker/src/daemon/gui_launcher.py

echo.
echo [3/3] Build complete!
echo Output: dist/DistribAI-Node/
echo.
echo To build portable single-file EXE instead:
echo   pyinstaller --onefile --name "DistribAI-Node-Portable" ... (same flags)
echo.
pause
