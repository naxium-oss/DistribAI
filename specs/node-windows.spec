"""
PyInstaller spec for DistribAI Node (Windows)

Build with:
    pyinstaller node-windows.spec --clean --noconfirm
"""

import os
from pathlib import Path

from PyInstaller.building.build_main import COLLECT, EXE, PYZ, Analysis

project_root = os.path.abspath(os.path.join(SPECPATH, '..'))
services_python = os.path.join(project_root, 'services_python')
worker = os.path.join(project_root, 'worker')

# ---------------------------------------------------------------------------
# Collect native DLLs that PyInstaller cannot statically detect:
#   clr_loader -> ClrLoader.dll (native .NET bridge, loaded via cffi.dlopen)
#   pythonnet  -> Python.Runtime.dll (.NET assembly, loaded at runtime)
# ---------------------------------------------------------------------------
_binaries = []

# Resolve the venv / site-packages path that contains these packages.
# We use importlib to find them at build time.
try:
    import clr_loader.ffi

    _clr_ffi_dir = Path(clr_loader.ffi.__file__).parent
    _dll_root = _clr_ffi_dir / "dlls"
    for _arch_dir in _dll_root.iterdir():
        if _arch_dir.is_dir():
            for _dll in _arch_dir.glob("*.dll"):
                _binaries.append((str(_dll), "clr_loader/ffi/dlls"))
            for _pdb in _arch_dir.glob("*.pdb"):
                _binaries.append((str(_pdb), "clr_loader/ffi/dlls"))
except (ImportError, Exception):
    pass

try:
    import pythonnet

    _pynet_dir = Path(pythonnet.__file__).parent
    for _dll in _pynet_dir.rglob("*.dll"):
        _binaries.append((str(_dll), "pythonnet/runtime"))
except (ImportError, Exception):
    pass

a = Analysis(
    [os.path.join(worker, 'src', 'daemon', 'gui_launcher.py')],
    pathex=[
        project_root,
        services_python,
        worker,
        os.path.join(worker, 'src'),
    ],
    binaries=_binaries,
    datas=[
        (os.path.join(worker, 'src', 'dashboard', 'static'), 'static'),
        (os.path.join(project_root, 'runtime', 'secrets', 'tls', 'ca.crt'),
         os.path.join('static', 'tls', 'ca.crt')),
    ],
    hiddenimports=[
        'worker.src.daemon.run',
        'worker.src.daemon.scheduler_config',
        'worker.src.daemon.job_executor',
        'worker.src.daemon.byzantine_detector',
        'worker.src.daemon.credit_ledger',
        'worker.src.daemon.voting_system',
        'worker.src.daemon.gradient_compression',
        'worker.src.daemon.ml_core',
        'worker.src.distribai_proto',
        'worker.src.daemon._node_defaults',

        'grpc',
        'grpc.aio',

        'torch',
        'torch.cuda',
        'torch.nn',
        'torch.nn.functional',
        'torch.optim',

        'pywebview',
        'webview',
        'webview.http',

        # ---- pythonnet / clr_loader / cffi (pywebview GUI stack on Windows) ----
        'pythonnet',
        'clr_loader',
        'clr_loader.ffi',
        'clr_loader.ffi.netfx',
        'clr_loader.netfx',
        'clr_loader.types',
        'cffi',
        'cffi.api',
        'cffi.cparser',
        'cffi.model',
        'cffi.ffi',
        'cffi.library',
        # -----------------------------------------------------------------------
        'numpy',
        'psutil',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        os.path.join(worker, 'src', 'daemon', '_node_defaults.py'),
    ],
    excludes=[
        'matplotlib',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'wx',
        'sphinx',
        'sphinx_rtd_theme',
        'alabaster',
        'sphinxcontrib',
        'docutils',
        'test',
        'tests',
        'testing',
        'unittest',
        'unittest.mock',
        'pytest',
        '_pytest',
        'nose',
        'nose2',
        'coverage',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

a.binaries = list(dict(a.binaries).items())

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DistribAI-Node',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(project_root, 'assets', 'icon-node.ico') if os.path.exists(os.path.join(project_root, 'assets', 'icon-node.ico')) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DistribAI-Node-Windows'
)
