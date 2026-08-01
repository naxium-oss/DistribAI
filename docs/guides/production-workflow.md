# Production Workflow

Complete end-to-end workflow for building and distributing DistribAI.

## Phase 1: Environment Setup

### 1.1 Install Python Dependencies

```bash
# Clone repository
git clone https://github.com/naxium-oss/DistribAI.git
cd distribai

# Run setup wizard (recommended)
python scripts/packaging/setup_wizard.py

# Or manual install
pip install -r requirements.txt  # For CPU
pip install -r requirements-cuda.txt  # For CUDA
```

### 1.2 Verify Setup

```bash
python tools/verify_setup.py
```

Should show all components ready.

## Phase 2: Build Packages

### 2.1 PyInstaller bundles (desktop)

```bash
python scripts/packaging/setup_wizard.py              # interactive wizard (recommended)
python scripts/packaging/setup_wizard.py --build-only # non-interactive; uses specs/*.spec
```

### 2.2 Python wheels / sdist

```bash
python build.py all
python build.py verify
```

### 2.3 Verify Builds

Check `dist/` after PyInstaller (layout depends on platform spec), for example:
```
dist/
├── DistribAI-Server-Windows/
├── DistribAI-Server-Windows-Setup.exe
├── DistribAI-Node-Windows/
├── DistribAI-Node-Windows-Setup.exe
├── DistribAI-Server-macos.app
├── DistribAI-Node-macos.app
├── DistribAI-Server-Linux/
└── DistribAI-Node-Linux/
```

## Phase 3: Test Locally

### 3.1 Test Server

```bash
# Windows
dist\DistribAI-Server-Windows\DistribAI-Server.exe

# macOS
open dist/DistribAI-Server-macos.app

# Linux
./dist/DistribAI-Server-Linux/DistribAI-Server
```

Verify:
- Settings panel opens
- Can save configuration
- Server starts/stops
- Dashboard accessible at http://localhost:8766

### 3.2 Test Node

```bash
# Windows
dist\DistribAI-Node-Windows\DistribAI-Node.exe

# macOS
open dist/DistribAI-Node-macos.app

# Linux
./dist/DistribAI-Node-Linux/DistribAI-Node
```

Verify:
- Connects to localhost:50051
- Hardware detection works
- Schedule can be set
- Auto-start toggle works

### 3.3 Integration Test

1. Start Server
2. Start Node
3. Verify connection in Server dashboard
4. Create test job via Server GUI
5. Verify Node receives and processes job

## Phase 4: Prepare Distribution

### 4.1 Calculate Package Hashes

```bash
# Windows PowerShell
Get-FileHash dist/DistribAI-Node-Windows-Setup.exe -Algorithm SHA256

# macOS/Linux
sha256sum dist/DistribAI-Node-Windows-Setup.exe
sha256sum dist/DistribAI-Node-macOS.zip
sha256sum dist/DistribAI-Node-Linux.tar.gz
```

### 4.2 Create version.json

```json
{
  "version": "1.0.0",
  "packages": {
    "windows": {
      "url": "https://your-server.com/DistribAI-Node-Windows-Setup.exe",
      "size_mb": 450,
      "hash": "sha256:YOUR_HASH_HERE"
    },
    "macos": {
      "url": "https://your-server.com/DistribAI-Node-macOS.zip",
      "size_mb": 480,
      "hash": "sha256:YOUR_HASH_HERE"
    },
    "linux": {
      "url": "https://your-server.com/DistribAI-Node-Linux.tar.gz",
      "size_mb": 460,
      "hash": "sha256:YOUR_HASH_HERE"
    }
  },
  "mandatory": false,
  "release_notes": "Initial production release"
}
```

### 4.3 Setup Update Hosting

**Option A: GitHub Releases**
1. Create public repo `yourname/distribai-releases`
2. Upload packages to Release
3. Host version.json in repo or use GitHub Pages

**Option B: Self-hosted**
1. Upload packages to web server
2. Upload version.json
3. Configure CORS headers

**Option C: S3 + CloudFront**
1. Upload to S3 bucket
2. Enable static website hosting
3. (Optional) CloudFront distribution

## Phase 5: Deploy Server

### 5.1 Server Configuration

Edit `.env` or use Settings GUI:
```bash
GRPC_PORT=50051
ADMIN_PORT=8766
ADMIN_HOST=0.0.0.0
JWT_SECRET=your-secret
SIGNING_KEY=your-signing-key
GITHUB_UPDATE_URL=https://your-server.com/releases/latest/download
```

### 5.2 Start Production Server

```bash
# Using package
./DistribAI-Server

# Or using source
python -m services_python.server_gui
```

### 5.3 Verify Server Health

```bash
curl http://localhost:8766/admin/health
```

Expected: `{"ok": true, ...}`

### 5.4 Open Firewall

```bash
# UFW (Ubuntu)
ufw allow 50051/tcp
ufw allow 8766/tcp

# iptables
iptables -A INPUT -p tcp --dport 50051 -j ACCEPT
iptables -A INPUT -p tcp --dport 8766 -j ACCEPT

# Windows Firewall
netsh advfirewall firewall add rule name="DistribAI gRPC" dir=in action=allow protocol=tcp localport=50051
netsh advfirewall firewall add rule name="DistribAI Admin" dir=in action=allow protocol=tcp localport=8766
```

## Phase 6: Distribute to Users

### 6.1 Create Distribution Package

**For Windows Users:**
```bash
zip -r distribai-windows.zip \
  DistribAI-Node-Windows-Setup.exe \
  README.txt
```

**For macOS Users:**
```bash
zip -r distribai-macos.zip \
  DistribAI-Node-macos.app \
  README.txt
```

**For Linux Users:**
```bash
tar czvf distribai-linux.tar.gz \
  DistribAI-Node-Linux/ \
  README.txt
```

### 6.2 Distribution Channels

**Direct Download:**
- Host on your server
- Provide direct links

**GitHub Releases:**
- Create release with packages
- Users download from Releases page

**Package Managers (Optional):**
- Homebrew (macOS)
- Chocolatey (Windows)
- APT/YUM (Linux)

### 6.3 User Instructions

Include with distribution:

```markdown
# DistribAI Node

## Quick Start

1. Install the package
2. Launch DistribAI Node
3. Enter server address: YOUR_SERVER:50051
4. Set your contribution schedule
5. Done!

## Server Address
YOUR_SERVER:50051

## Support
- **Primary support, bug reports, and feature requests**: https://github.com/naxium-oss/DistribAI/issues
- **Documentation**: see the `docs/` directory in this repository
- **Community discussion**: optional peer discussion only; file support requests in GitHub Issues
```

## Phase 7: Monitor and Maintain

### 7.1 Monitor Server

**Dashboard:**
- Check node count
- Monitor job queue
- Watch credit distribution

**Logs:**
```bash
# Server logs
tail -f ~/.distribai/logs/server.log

# Node logs (from user machines)
~/.distribai/logs/node.log
```

**Metrics (if enabled):**
```bash
curl http://localhost:9090/metrics
```

### 7.2 Update Packages

**When releasing new version:**

1. Update `version` in `pyproject.toml` (and any component version strings)
2. Rebuild packages: `python scripts/packaging/setup_wizard.py --build-only` (or `python build.py all` for wheels only)
3. Test locally
4. Calculate new hashes
5. Update `version.json`
6. Upload to hosting
7. Nodes auto-detect update

### 7.3 Handle Issues

**Node Connection Issues:**
- Check firewall
- Verify server running
- Check node logs

**Byzantine Detection Triggers:**
- Review flagged nodes
- Ban confirmed malicious actors
- Adjust thresholds if needed

**Ledger Issues:**
- Run integrity check
- Restore from backup if needed
- Verify Merkle root

## Automation (CI/CD)

### GitHub Actions Workflow

```yaml
name: Build and Release

on:
  push:
    tags: ['v*']

jobs:
  build:
    strategy:
      matrix:
        os: [windows-latest, macos-latest, ubuntu-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt pyinstaller
      
      - name: Build packages
        run: python scripts/packaging/setup_wizard.py --build-only
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: packages-${{ matrix.os }}
          path: dist/

  release:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Download all artifacts
        uses: actions/download-artifact@v3
      
      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            packages-*/*
```

## Checklist

### Pre-Release
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version numbers bumped
- [ ] Changelog prepared
- [ ] Packages built for all platforms
- [ ] Installers tested
- [ ] Hashes calculated
- [ ] version.json created

### Release
- [ ] Upload to hosting
- [ ] Verify download URLs
- [ ] Test update mechanism
- [ ] Announce to users
- [ ] Monitor adoption

### Post-Release
- [ ] Monitor error rates
- [ ] Track node adoption
- [ ] Collect feedback
- [ ] Plan next version

## Troubleshooting Common Issues

### Build Fails
- Check PyInstaller installed
- Verify all imports resolvable
- Check for missing hidden imports in spec files

### Package Won't Start
- Check dependencies bundled
- Verify executable permissions (Unix)
- Check for missing DLLs (Windows)

### Auto-Update Not Working
- Verify version.json accessible from browser
- Check CORS headers
- Verify HTTPS (required)
- Check hash format (sha256: prefix)

### Nodes Can't Connect
- Verify server port open
- Check firewall rules
- Test with telnet/netcat
- Check server logs

## Resources

- **Packaging Guide**: `docs/guides/packaging.md`
- **Node User Guide**: `docs/guides/node-user-guide.md`
- **Server Operator Guide**: `docs/guides/server-operator-guide.md`
- **Update Hosting Guide**: `docs/guides/update-hosting.md`
