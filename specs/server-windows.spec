# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for DistribAI Server (Windows)

Build with:
    pyinstaller server-windows.spec --clean --noconfirm
"""

from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
from PyInstaller.building.api import BUNDLE
from PyInstaller.building.osx import BUNDLE as OSX_BUNDLE
import os
import sys
from pathlib import Path

# Paths
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

# Analysis
a = Analysis(
    [os.path.join(services_python, 'server_gui.py')],
    pathex=[
        project_root,
        services_python,
        worker,
        os.path.join(worker, 'src'),
    ],
    binaries=_binaries,
    datas=[
        (os.path.join(worker, 'src', 'dashboard', 'static'), 'static'),
        (os.path.join(project_root, 'docs'), 'docs'),
    ],
    hiddenimports=[
        # Core modules
        'services_python.orchestrator_grpc',
        'services_python.grpc_service',
        'services_python.scheduler',
        'services_python.constants',
        'services_python.admin_api',
        'services_python.admin_api.health',
        'services_python.admin_api.jobs',
        'services_python.admin_api.nodes',
        'services_python.admin_api.credits',
        'services_python.admin_api.votes',
        'services_python.admin_api.v1',
        'services_python.admin_api.ledger',
        'services_python.admin_api.multipliers',
        'services_python.admin_api.sybil',

        # Worker modules
        'worker.src.daemon.byzantine_detector',
        'worker.src.daemon.credit_ledger',
        'worker.src.daemon.voting_system',
        'worker.src.daemon.scheduler_config',

        # gRPC
        'grpc',
        'grpc.aio',
        'grpc.experimental',
        'grpc._cython',
        'grpc._cython.cygrpc',

        # Web framework
        'aiohttp',
        'aiohttp.web',
        'aiohttp.web_request',
        'aiohttp.web_response',
        'aiohttp.web_middlewares',
        'aiohttp_cors',
        'aiofiles',

        # ML
        'torch',
        'torch.cuda',
        'torch.nn',
        'torch.nn.functional',
        'torch.optim',
        'torch.utils.data',

        # AWS
        'boto3',
        'botocore',
        'botocore.config',

        # Security
        'jwt',
        'cryptography',
        'cryptography.hazmat',
        'cryptography.hazmat.primitives',

        # Database
        'aiosqlite',
        'redis',
        'redis.asyncio',

        # Validation
        'pydantic',
        'pydantic.deprecated',

        # GUI
        'pywebview',
        'webview',
        'webview.http',
        'webview.util',

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

        # Utilities
        'dotenv',
        'psutil',
        'numpy',
        'structlog',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'matplotlib.backends',
        'matplotlib.pyplot',
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt6',
        'PySide2',
        'PySide6',
        'wx',
        'wxPython',
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
        'pylint',
        'flake8',
        'black',
        'mypy',
        'bandit',
        'safety',
        'pip_audit',
        'wheel',
        'setuptools',
        'pkg_resources',
        'distribute',
        'distutils',
        'ensurepip',
        'idlelib',
        'turtle',
        'turtledemo',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Remove duplicate entries
a.binaries = list(dict(a.binaries).items())

# Create PYZ (compressed Python archive)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Windows executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DistribAI-Server',
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
    icon=os.path.join(project_root, 'assets', 'icon-server.ico') if os.path.exists(os.path.join(project_root, 'assets', 'icon-server.ico')) else None,
    version=os.path.join(project_root, 'assets', 'version.txt') if os.path.exists(os.path.join(project_root, 'assets', 'version.txt')) else None,
    manifest=os.path.join(project_root, 'assets', 'manifest.xml') if os.path.exists(os.path.join(project_root, 'assets', 'manifest.xml')) else None,
)

# Collect everything into a directory
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DistribAI-Server-Windows'
)
