# Packaging Guide

Complete guide for building DistribAI standalone executables.

## Prerequisites

- Python 3.11+
- PyInstaller: `pip install pyinstaller`
- Windows: NSIS (for installer creation)
- macOS: Xcode command line tools
- Linux: `appimagetool` (optional, for AppImage)

## Quick Build

```bash
python scripts/packaging/setup_wizard.py
```

This launches the interactive wizard that:
1. Detects your platform and CUDA availability
2. Installs correct dependencies
3. Builds the packages

## Manual Build

**PyInstaller bundles (server/node desktop):** use the interactive wizard or build-only mode:

```bash
python scripts/packaging/setup_wizard.py              # interactive wizard
python scripts/packaging/setup_wizard.py --build-only # non-interactive PyInstaller build (see specs/*.spec)
```

**Python packages (wheels/sdist):** `build.py` wraps setuptools — it does **not** accept `--target` or `--platform`:

```bash
python build.py all     # wheel + sdist
python build.py verify  # smoke-check build artifacts
python build.py --help  # full command list
```

Legacy docs referring to `python build.py --target server` are outdated; use `scripts/packaging/setup_wizard.py` + `specs/` instead.

### Build Server (PyInstaller)

```bash
python scripts/packaging/setup_wizard.py --build-only
# or: pyinstaller specs/server-windows.spec  (see specs/ for platform variants)
```

### Build Node (PyInstaller)

```bash
python scripts/packaging/setup_wizard.py --build-only
# or: pyinstaller specs/node-windows.spec
```

### Build Python wheels only

```bash
python build.py all
```

## Platform-Specific Instructions

PyInstaller specs live under [`specs/`](../../specs/). Use `python scripts/packaging/setup_wizard.py` (interactive) or `python scripts/packaging/setup_wizard.py --build-only` on each platform. For Python wheels/sdist only, use `python build.py all`.

### Windows

1. Install NSIS from https://nsis.sourceforge.io/ (optional, for installers)
2. Add NSIS to PATH if creating installers
3. Build:
   ```bash
   python scripts/packaging/setup_wizard.py --build-only
   # or: pyinstaller specs/server-windows.spec
   #     pyinstaller specs/node-windows.spec
   ```

Typical output under `dist/` (names vary by spec):
- `dist/DistribAI-Server-Windows/` (portable folder)
- `dist/DistribAI-Node-Windows/` (portable folder)

### macOS

1. Ensure Xcode tools installed: `xcode-select --install`
2. Build:
   ```bash
   python scripts/packaging/setup_wizard.py --build-only
   # or: pyinstaller specs/server-macos.spec
   ```

Output:
- `dist/DistribAI-Server-macos.app` (app bundle)
- `dist/DistribAI-Node-macos.app` (app bundle)

To create DMG for distribution:
```bash
hdiutil create -volname "DistribAI" -srcfolder dist/DistribAI-Server-macos.app -ov -format UDZO DistribAI-Server.dmg
```

### Linux

Build:
```bash
python scripts/packaging/setup_wizard.py --build-only
# or: pyinstaller specs/server-linux.spec
```

Output:
- `dist/DistribAI-Server-Linux/` (portable folder)
- `dist/DistribAI-Node-Linux/` (portable folder)

Optional AppImage (after portable build):
```bash
wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x appimagetool-x86_64.AppImage
# package the dist folder per your AppImage recipe
```

## Build Outputs

### Public releases (contributors only)

For grids where the orchestrator stays private:

- Publish **node** artifacts only via GitHub Releases (`python scripts/packaging/bundle.py node`)
- Do **not** ship admin/server binaries in the public release channel
- Point contributors to [contributor-join-kit.md](contributor-join-kit.md) and `requirements-worker.txt`
- Verify the public mirror excludes `services_python/` ([`publish_public_grid.py`](../../scripts/publish/publish_public_grid.py))

Operator (boss) builds admin locally or in private CI:

```bash
python scripts/packaging/bundle.py admin   # private — not for public releases
python scripts/packaging/bundle.py node    # public contributor binary
python scripts/packaging/bundle.py cli     # admin CLI + TUI, onefile (no Python needed)
```

Contributors can also `pip install -r requirements-worker.txt` from the public repo instead of using a binary.

### CLI Package

`python scripts/packaging/bundle.py cli` produces a onefile `distribai-cli`/`distribai-cli.exe` binary
bundling the flat CLI and the Textual TUI (`distribai-cli tui`) — see README.md's
["CLI & TUI"](../../README.md#cli--tui) section for the full command reference. This is the
right artifact for admins who want fleet control (`nodes`, `job`, `credits`, `orchestrator`) from a
terminal without a Python install. It talks to the same admin HTTP API as the GUI dashboards, so it
never needs orchestrator/server source — safe to hand out alongside node binaries.

### Server Package

Contents:
- `DistribAI-Server.exe` (Windows) or `DistribAI-Server` (Unix)
- `_internal/` - Python runtime and dependencies
- `static/` - Dashboard web files
- `.env` - Configuration file (created on first run)

### Node Package

Contents:
- `DistribAI-Node.exe` (Windows) or `DistribAI-Node` (Unix)
- `_internal/` - Python runtime and dependencies
- `static/` - Dashboard web files
- `.distribai/` - User data directory (created at runtime)

## Customization

### Using Custom Spec Files

Edit spec files in `specs/` directory:
- `specs/server-windows.spec`
- `specs/node-windows.spec`

Then build:
```bash
pyinstaller specs/server-windows.spec --clean
```

### Reducing Package Size

Exclude unnecessary modules in spec files:
```python
excludes = [
    'matplotlib',
    'tkinter',
    'PyQt5',
    'unittest',
    'pytest',
]
```

### Single File vs Directory

PyInstaller one-file vs one-folder mode is controlled in each `specs/*.spec` file (`EXE(..., onefile=True)` vs `COLLECT`). Edit the spec, then:

```bash
pyinstaller specs/server-windows.spec --clean
```

For Python wheels only (not desktop bundles): `python build.py all`.

## Troubleshooting

### "ModuleNotFoundError" after build

Add to spec file hidden imports:
```python
hiddenimports=['missing_module'],
```

### "DLL load failed" on Windows

Install Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### Large file size

- Use `--exclude-module` to remove unused packages
- Use UPX compression (enabled by default)
- Consider single-file only for distribution, not development

### macOS "app is damaged"

Code sign or disable Gatekeeper:
```bash
xattr -cr DistribAI-Server-macos.app
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build Packages

on: [push, pull_request]

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt pyinstaller
      - run: python scripts/packaging/setup_wizard.py --build-only
      - uses: actions/upload-artifact@v3
        with:
          name: windows-packages
          path: dist/

  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt pyinstaller
      - run: python scripts/packaging/setup_wizard.py --build-only
      - uses: actions/upload-artifact@v3
        with:
          name: macos-packages
          path: dist/

  build-linux:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt pyinstaller
      - run: python scripts/packaging/setup_wizard.py --build-only
      - uses: actions/upload-artifact@v3
        with:
          name: linux-packages
          path: dist/
```
