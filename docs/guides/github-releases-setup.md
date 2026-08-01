# GitHub Releases Setup Guide for DistribAI Admin

## Overview

This guide walks you through setting up GitHub releases for the DistribAI auto-update system. This is **CRITICAL** for production deployment.

## Prerequisites

- GitHub account with admin access to your releases repository
- Built DistribAI Node executables from `setup.py`
- Git installed locally

## Step 1: Create Releases Repository

1. **Create a new public repository** on GitHub:
   - Name: `distribai-releases` (or your preferred name)
   - Description: "DistribAI Node Application Releases"
   - **IMPORTANT**: Make it PUBLIC so nodes can download updates
   - **DO NOT** upload source code - only release binaries

2. **Clone the repository locally**:
```bash
git clone https://github.com/your-org/distribai-releases.git
cd distribai-releases
```

## Step 2: Prepare Release Files

1. **Build the Node packages** (if not already done):
```bash
cd /path/to/GRID-PROJECT
python setup.py
```

2. **Locate your built files** in `dist/`:
   - `DistribAI-Node-Windows.exe` (or similar)
   - `DistribAI-Node-Linux.AppImage` (or similar)
   - `DistribAI-Node-macOS.app.zip` (or similar)

## Step 3: Create Your First Release

### Method A: Using GitHub Web Interface (Recommended)

1. **Go to your repository** on GitHub
2. **Click "Releases"** → "Create a new release"
3. **Create a new tag**:
   - Tag version: `v1.0.0`
   - Release title: `DistribAI Node v1.0.0`
   - Release notes: "Initial production release"

4. **Upload your binaries**:
   - Drag and drop your `.exe`, `.AppImage`, and `.app.zip` files
   - GitHub will automatically generate download URLs

### Method B: Using GitHub CLI

```bash
# Install GitHub CLI if not present
gh auth login

# Create release with files
gh release create v1.0.0 \
  dist/DistribAI-Node-Windows.exe \
  dist/DistribAI-Node-Linux.AppImage \
  dist/DistribAI-Node-macOS.app.zip \
  --title "DistribAI Node v1.0.0" \
  --notes "Initial production release"
```

## Step 4: Create version.json

1. **Generate SHA-256 hashes** for your files:
```bash
# On Windows
certutil -hashfile dist/DistribAI-Node-Windows.exe SHA256

# On Linux/macOS
sha256sum dist/DistribAI-Node-Windows.exe
```

2. **Create version.json** with the following format:
```json
{
  "version": "1.0.0",
  "download_url": "https://github.com/your-org/distribai-releases/releases/download/v1.0.0/",
  "size_mb": 85,
  "notes": "Initial production release with auto-update support",
  "hash": "a1b2c3d4e5f6...",
  "release_date": "2025-01-10",
  "platform": "windows"
}
```

3. **Upload version.json** to your release:
   - Either add it as another asset to the release
   - Or upload it to the repository root

## Step 5: Configure Your .env

Update your `.env` file with the release information:

```bash
# Update these values in your .env
GITHUB_UPDATE_URL=https://github.com/your-org/distribai-releases/releases/latest/download
GITHUB_EXE_FILE=DistribAI-Node-Windows.exe
GITHUB_APP_FILE=DistribAI-Node-macOS.app.zip
```

## Step 6: Test Auto-Update

1. **Start your orchestrator** with the new config:
```bash
python -m services_python.orchestrator_grpc
```

2. **Verify the update endpoint**:
```bash
curl http://localhost:8766/admin/update-url
```

Should return:
```json
{
  "update_url": "https://github.com/your-org/distribai-releases/releases/latest/download",
  "exe_file": "DistribAI-Node-Windows.exe",
  "app_file": "DistribAI-Node-macOS.app.zip"
}
```

3. **Test version.json access**:
```bash
curl https://github.com/your-org/distribai-releases/releases/latest/download/version.json
```

## Step 7: Update Process for Future Releases

When you need to release updates:

1. **Build new versions**:
```bash
python setup.py
```

2. **Create new release** with incremented version:
```bash
gh release create v1.0.1 \
  dist/DistribAI-Node-Windows.exe \
  dist/DistribAI-Node-Linux.AppImage \
  dist/DistribAI-Node-macOS.app.zip \
  --title "DistribAI Node v1.0.1" \
  --notes "Bug fixes and performance improvements"
```

3. **Update version.json** with new version and hash
4. **Upload version.json** to the new release

## Security Considerations

✅ **SAFE**: Repository only contains compiled binaries, not source code
✅ **SAFE**: Nodes only download from your official releases
✅ **SAFE**: SHA-256 hash verification prevents tampering
✅ **SAFE**: Auto-update only works for Node apps, not Admin/Orch

❌ **NEVER**: Upload source code to releases repository
❌ **NEVER**: Include Admin/Orch binaries in releases
❌ **NEVER**: Use untrusted download URLs

## Troubleshooting

### Nodes Can't Download Updates

1. **Check repository is public**: Private repos won't work
2. **Verify URLs are correct**: Test download URLs in browser
3. **Check version.json format**: JSON must be valid
4. **Verify file hashes**: Mismatched hashes block updates

### Release Not Found

1. **Check tag format**: Use `v1.0.0` format, not `1.0.0`
2. **Verify release is published**: Draft releases won't work
3. **Check file names**: Must match `.env` configuration

### Hash Verification Failed

1. **Regenerate hashes**: Use correct SHA-256 for your platform
2. **Check file integrity**: Files may be corrupted
3. **Update version.json**: Ensure correct hash in JSON

## Quick Reference

**Essential URLs**:
- Repository: `https://github.com/your-org/distribai-releases`
- Downloads: `https://github.com/your-org/distribai-releases/releases/latest/download/`
- API: `https://api.github.com/repos/your-org/distribai-releases/releases/latest`

**Required Files**:
- `DistribAI-Node-Windows.exe`
- `DistribAI-Node-Linux.AppImage` 
- `DistribAI-Node-macOS.app.zip`
- `version.json`

**Environment Variables**:
```bash
GITHUB_UPDATE_URL=https://github.com/your-org/distribai-releases/releases/latest/download
GITHUB_EXE_FILE=DistribAI-Node-Windows.exe
GITHUB_APP_FILE=DistribAI-Node-macOS.app.zip
```

---

**⚠️ CRITICAL**: This setup is REQUIRED for production deployment. Without it, nodes cannot auto-update and your deployment will fail.