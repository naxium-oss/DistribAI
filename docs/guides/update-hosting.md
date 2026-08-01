# Update Hosting Guide

How to host DistribAI packages for auto-update distribution.

## Overview

Nodes periodically check for updates from a configured URL. When new versions are available, they download and install automatically (with user consent).

**Requirements:**
- Public URL accessible by nodes
- `version.json` file with version info
- Package files (exe, app, tar.gz)

## Hosting Options

### Option 1: GitHub Releases (Recommended)

Free, reliable, easy to use.

**Setup:**

1. Create public repository (e.g., `yourname/distribai-releases`)
2. Go to Releases → Draft new release
3. Tag version: `v1.0.0`
4. Upload packages:
   - `DistribAI-Node-Windows.exe`
   - `DistribAI-Node-macOS.zip`
   - `DistribAI-Node-Linux.tar.gz`
5. Publish release

**version.json:**

Create file in repo root:
```json
{
  "version": "1.0.0",
  "packages": {
    "windows": {
      "url": "https://github.com/yourname/distribai-releases/releases/download/v1.0.0/DistribAI-Node-Windows.exe",
      "size_mb": 450,
      "hash": "sha256:abc123..."
    },
    "macos": {
      "url": "https://github.com/yourname/distribai-releases/releases/download/v1.0.0/DistribAI-Node-macOS.zip",
      "size_mb": 480,
      "hash": "sha256:def456..."
    },
    "linux": {
      "url": "https://github.com/yourname/distribai-releases/releases/download/v1.0.0/DistribAI-Node-Linux.tar.gz",
      "size_mb": 460,
      "hash": "sha256:ghi789..."
    }
  },
  "mandatory": false,
  "release_notes": "Bug fixes and performance improvements."
}
```

**Server Configuration:**

In Server Settings → Update Hosting:
```
https://raw.githubusercontent.com/yourname/distribai-releases/main/version.json
```

Or use GitHub Pages:
```
https://yourname.github.io/distribai-releases/version.json
```

### Option 2: Self-hosted Web Server

More control, custom domain.

**Directory Structure:**
```
/var/www/distribai-updates/
├── version.json
├── DistribAI-Node-Windows.exe
├── DistribAI-Node-macOS.zip
└── DistribAI-Node-Linux.tar.gz
```

**Nginx Config:**
```nginx
server {
    listen 80;
    server_name updates.yourgrid.com;
    root /var/www/distribai-updates;
    
    location / {
        autoindex on;
        add_header Access-Control-Allow-Origin *;
    }
    
    location ~ \.json$ {
        add_header Content-Type application/json;
        add_header Cache-Control no-cache;
    }
}
```

**version.json:**
```json
{
  "version": "1.0.0",
  "packages": {
    "windows": {
      "url": "https://updates.yourgrid.com/DistribAI-Node-Windows.exe",
      "size_mb": 450,
      "hash": "sha256:abc123..."
    },
    "macos": {
      "url": "https://updates.yourgrid.com/DistribAI-Node-macOS.zip",
      "size_mb": 480,
      "hash": "sha256:def456..."
    },
    "linux": {
      "url": "https://updates.yourgrid.com/DistribAI-Node-Linux.tar.gz",
      "size_mb": 460,
      "hash": "sha256:ghi789..."
    }
  },
  "mandatory": false,
  "release_notes": "Bug fixes and performance improvements."
}
```

**Server Configuration:**

Settings → Update Hosting:
```
https://updates.yourgrid.com
```

### Option 3: S3/Cloud Storage

CDN distribution, global availability.

**AWS S3:**

1. Create bucket: `distribai-updates`
2. Enable public read:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Principal": "*",
       "Action": "s3:GetObject",
       "Resource": "arn:aws:s3:::distribai-updates/*"
     }]
   }
   ```
3. Upload packages
4. Enable static website hosting

**CloudFront (Optional CDN):**

1. Create distribution
2. Origin: S3 bucket
3. TTL: 1 hour (for version.json), 1 day (for packages)

**version.json:**
```json
{
  "version": "1.0.0",
  "packages": {
    "windows": {
      "url": "https://your-cdn.cloudfront.net/DistribAI-Node-Windows.exe",
      "size_mb": 450,
      "hash": "sha256:abc123..."
    },
    "macos": {
      "url": "https://your-cdn.cloudfront.net/DistribAI-Node-macOS.zip",
      "size_mb": 480,
      "hash": "sha256:def456..."
    },
    "linux": {
      "url": "https://your-cdn.cloudfront.net/DistribAI-Node-Linux.tar.gz",
      "size_mb": 460,
      "hash": "sha256:ghi789..."
    }
  },
  "mandatory": false,
  "release_notes": "Bug fixes and performance improvements."
}
```

## version.json Format

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Semantic version (e.g., "1.0.0") |
| `packages` | object | Platform-specific packages |

### Package Object

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Direct download URL |
| `size_mb` | number | Size for user notification |
| `hash` | string | SHA256 for verification (optional) |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `mandatory` | boolean | Force update (default: false) |
| `release_notes` | string | Changelog shown to users |
| `minimum_version` | string | Require nodes to be on at least this version |

## Generating Hashes

### Windows (PowerShell)
```powershell
Get-FileHash DistribAI-Node-Windows.exe -Algorithm SHA256
```

### macOS/Linux
```bash
sha256sum DistribAI-Node-Windows.exe
```

## Release Process

### 1. Build Packages

```bash
python setup.py --build-only
```

### 2. Test Locally

Install and run each package before release.

### 3. Calculate Hashes

```bash
sha256sum dist/DistribAI-Node-*
```

### 4. Update version.json

Edit with new version, URLs, hashes, notes.

### 5. Upload

**GitHub:**
```bash
# Create release via web UI or gh CLI
gh release create v1.1.0 dist/*
```

**S3:**
```bash
aws s3 cp dist/DistribAI-Node-Windows.exe s3://distribai-updates/
aws s3 cp version.json s3://distribai-updates/
```

**Self-hosted:**
```bash
scp dist/* your-server:/var/www/distribai-updates/
```

### 6. Notify Users

Nodes check automatically on schedule. No action needed.

For urgent updates, notify users via:
- Server dashboard announcement
- Email notification
- Set `mandatory: true` in version.json

## Node Update Behavior

### Check Schedule

Nodes check for updates:
- On startup
- Every 24 hours (configurable)
- When manually triggered in GUI

### Update Flow

1. Node queries `version.json` from update URL
2. Compares local version to remote
3. If newer version available:
   - Shows notification to user
   - Displays size and release notes
4. User clicks "Download"
5. Download in background with progress
6. SHA256 verification (if hash provided)
7. User clicks "Install and Restart"
8. App updates and relaunches

### Mandatory Updates

Set `mandatory: true` in version.json:
- Shows modal dialog (can't dismiss)
- Countdown to forced install
- Use for critical security fixes

## Troubleshooting

### "Update check failed"

Node logs show:
```
[Updater] Failed to fetch version.json: HTTP 404
```

**Fix:**
1. Verify URL accessible from browser
2. Check CORS headers (for web-hosted JSON)
3. Ensure proper Content-Type

### "Hash verification failed"

Package corrupted or hash mismatch.

**Fix:**
1. Re-calculate hash
2. Update version.json
3. Re-upload package

### "Download interrupted"

Partial download, network issue.

**Fix:**
- App auto-resumes partial downloads
- Or clears and restarts on retry
- Check Content-Length header is correct

### "Install failed"

Platform-specific issue.

**Windows:**
- Run installer with `/SILENT` flag
- Check Windows Defender exclusion

**macOS:**
- Code sign or ad-hoc sign: `codesign --force --deep --sign - DistribAI-Node.app`
- Notarize for distribution outside App Store

**Linux:**
- Check AppImage dependencies
- Verify executable permissions

## Security Considerations

### HTTPS Required

Always use HTTPS for update URLs to prevent MITM attacks.

### Hash Verification

Always provide SHA256 hashes. Nodes verify before install.

### Code Signing

**Windows:**
- Sign with EV certificate for SmartScreen
- Or standard certificate

**macOS:**
- Developer ID certificate required for distribution
- Notarization required for Catalina+

**Linux:**
- GPG sign AppImage or packages
- Distribute public key for verification

### Rollback Protection

Nodes only update forward (higher version numbers). Prevents downgrade attacks.

## Analytics (Optional)

Track update adoption:

```json
{
  "version": "1.1.0",
  "packages": {...},
  "analytics_url": "https://your-analytics.com/update?v=1.1.0&platform={platform}"
}
```

Nodes ping analytics URL when checking for updates.

## Example Workflows

### CI/CD Automated Releases

```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    tags: ['v*']

jobs:
  build-and-release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build all packages
        run: |
          pip install -r requirements.txt
          python setup.py --build-only
      
      - name: Calculate hashes
        run: |
          echo "WINDOWS_HASH=$(sha256sum dist/DistribAI-Node-Windows.exe | cut -d' ' -f1)" >> $GITHUB_ENV
          echo "MACOS_HASH=$(sha256sum dist/DistribAI-Node-macOS.zip | cut -d' ' -f1)" >> $GITHUB_ENV
          echo "LINUX_HASH=$(sha256sum dist/DistribAI-Node-Linux.tar.gz | cut -d' ' -f1)" >> $GITHUB_ENV
      
      - name: Create version.json
        run: |
          cat > version.json <<EOF
          {
            "version": "${GITHUB_REF#refs/tags/v}",
            "packages": {
              "windows": {"url": "...", "size_mb": 450, "hash": "sha256:${WINDOWS_HASH}"},
              "macos": {"url": "...", "size_mb": 480, "hash": "sha256:${MACOS_HASH}"},
              "linux": {"url": "...", "size_mb": 460, "hash": "sha256:${LINUX_HASH}"}
            }
          }
          EOF
      
      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          files: dist/*
```

This automates the entire release process on every version tag.
