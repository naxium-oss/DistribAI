"""
PyInstaller spec for DistribAI Node (Windows)

Build with:
    pyinstaller specs/node-windows.spec --clean --noconfirm
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

_datas = [
    (os.path.join(worker, 'src', 'dashboard', 'static'), 'static'),
]
_tls_ca = os.path.join(project_root, 'runtime', 'secrets', 'tls', 'ca.crt')
if os.path.isfile(_tls_ca):
    _datas.append((_tls_ca, os.path.join('static', 'tls')))

_icon = os.path.join(project_root, 'assets', 'icon-node.ico')
if not os.path.isfile(_icon):
    _icon = None

a = Analysis(
    [os.path.join(worker, 'src', 'daemon', 'gui_launcher.py')],
    pathex=[
        project_root,
        services_python,
        worker,
        os.path.join(worker, 'src'),
    ],
    binaries=_binaries,
    datas=_datas,
    hiddenimports=[
        'worker.src.daemon.run',
        'worker.src.daemon.scheduler_config',
        'worker.src.daemon.executor',
        'worker.src.daemon.byzantine_detector',
        'worker.src.daemon.credit_ledger',
        'worker.src.daemon.voting_system',
        'worker.src.daemon.gradient_compression',
        'worker.src.daemon.ml_core',
        'worker.src.distribai_proto',
        'worker.src.daemon._node_defaults',
        'worker.src.compute.distribai_models',
        'worker.src.compute.external_arch',

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

_seen_bins: set[str] = set()
_deduped_bins = []
for entry in a.binaries:
    key = entry[0]
    if key in _seen_bins:
        continue
    _seen_bins.add(key)
    _deduped_bins.append(entry)
a.binaries = _deduped_bins

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DistribAI-Node',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DistribAI-Node-Windows',
)
