# Node User Guide

Guide for contributors running DistribAI Node.

## Installation

### Option 1: Pre-built Package

1. Download from your grid operator's update URL
2. Run the installer (Windows) or copy the app (macOS/Linux)
3. Launch DistribAI Node

### Option 2: Build from Source

```bash
git clone https://github.com/naxium-oss/DistribAI.git
cd distribai
pip install -r requirements.txt
python -m worker.src.daemon.run
```

## First Run

### 1. Server Connection

Enter the server address provided by your grid operator:
- Format: `server.example.com:50051` or `192.168.1.100:50051`
- Click "Connect"

### 2. Hardware Detection

The app automatically detects your GPU:
- **GPU detected**: Shows CUDA version and VRAM
- **No GPU**: Falls back to CPU mode with warning

### 3. CUDA Setup (if needed)

If you have an NVIDIA GPU but CUDA isn't detected:
1. Click "Download CUDA Runtime" (~815 MB)
2. Confirm download size
3. Wait for installation
4. Click "Rescan for CUDA"

### 4. Set Compute Donation

Choose how much of your GPU to donate:

**Percentage Slider:**
- Drag to set percentage (0-100%)
- Default: 90%
- Shows actual VRAM: "90% (10.8GB of 12GB)"

**Why not 100%?**
- Leaves room for your own work
- Prevents system lag
- Recommended: 75-90% for daily use

### 5. Create Schedule

Set when to contribute:

**Example Schedule 1 - "Work Hours":**
- Start: 9:00 AM
- End: 5:00 PM
- Days: Monday - Friday
- GPU: 25% (minimal impact while working)

**Example Schedule 2 - "Night Mining":**
- Start: 11:00 PM
- End: 6:00 AM
- Days: All days
- GPU: 90% (full power while sleeping)

**Timezone:** Automatically detected from your system.

## Dashboard

### Status Indicators

| Indicator | Meaning |
|-----------|---------|
| 🟢 Connected | Connected to server, ready |
| 🔴 Disconnected | Not connected |
| 🟡 Contributing | Currently processing a task |
| ⚪ Idle | Connected but no active task |

### Stats Displayed

- **Credits Earned**: Total from completed tasks
- **Jobs Completed**: Number of micro-tasks finished
- **Uptime**: Time since connection
- **Current Task**: What you're working on now
- **GPU Usage**: Current VRAM and compute %

## Scheduling

### Managing Schedules

1. Click "Schedules" tab
2. Click "Add Schedule" to create new
3. Click existing schedule to edit
4. Toggle switch to enable/disable

### Schedule Conflicts

If multiple schedules overlap, the **highest GPU % wins**.

Example:
- Schedule A: 9AM-5PM @ 25%
- Schedule B: 12PM-1PM @ 90%

Result: 12PM-1PM uses 90%, rest of 9-5 uses 25%.

### Auto-Join/Leave

- **Auto-join**: Connect at schedule start time
- **Auto-leave**: Disconnect at schedule end time

Both can be toggled independently.

## Per-Job Compute Boost

Increase donation for specific jobs:

1. Go to "Jobs" tab
2. See active jobs on the grid
3. Click job name
4. Set "Extra Compute" (e.g., +20%)
5. Your donation becomes: base% + extra%

Use case: Support a specific AI model you care about.

## Settings

### General

- **Auto-start on boot**: Launch app on login
- **Minimize to tray**: Keep running in background
- **Dark/Light theme**: UI appearance

### Hardware

- **GPU Selection**: If multiple GPUs
- **VRAM Limit**: Cap max VRAM usage
- **CPU Threads**: For CPU-only mode
- **Rescan CUDA**: Detect new hardware

### Connection

- **Server Address**: Change grid operator
- **Reconnect Delay**: Seconds between retries
- **Heartbeat Interval**: Keep-alive frequency

### Updates

- **Check on launch**: Auto-check for updates
- **Auto-download**: Download in background
- **Update URL**: Usually auto-discovered from server

## Auto-Start on Boot

### Windows

Settings → General → Auto-start:
- Creates shortcut in Startup folder
- Runs minimized to system tray

### macOS

Settings → General → Auto-start:
- Creates LaunchAgent
- Runs in background

### Linux

Settings → General → Auto-start:
- Creates systemd user service
- Runs in background

## Troubleshooting

### "Cannot connect to server"

1. Check internet connection
2. Verify server address format: `host:port`
3. Ask operator if server is running
4. Check firewall (port 50051)

### "CUDA not detected"

1. Install NVIDIA drivers: https://www.nvidia.com/drivers
2. Restart app
3. Click "Resets" → "Rescan for CUDA"
4. Download CUDA runtime if offered

### "Out of memory"

1. Reduce GPU percentage in Settings
2. Close other GPU apps (games, other ML)
3. Set VRAM limit lower

### "No tasks assigned"

1. Check schedule is currently active
2. Verify server has active jobs (ask operator)
3. Ensure GPU percentage > 0

### "Earnings not showing"

Credits are awarded after task completion:
- Training tasks: 5-30 minutes each
- Check "Jobs Completed" counter
- Credits appear in batches

### App won't start

Windows:
- Install Visual C++ Redistributable
- Run as Administrator once

macOS:
- Check Security & Privacy settings
- Right-click app → Open

Linux:
- Check dependencies: `ldd DistribAI-Node`
- Install missing libraries

## Security

### What the App Can Access

- **GPU**: For compute only
- **Network**: To server only (port 50051)
- **Files**: Only `.distribai/` directory in home
- **System**: No admin rights needed (except auto-start)

### What the App Cannot Do

- Access personal files
- Browse internet
- Install other software
- Access microphone/camera

### Privacy

- Node ID is anonymous (random string)
- No personal data collected
- Only hardware specs sent to server
- Gradients are encrypted in transit

## Earning Credits

### How Credits Work

1. Complete training tasks
2. Submit valid gradients
3. Pass Byzantine checks
4. Credits added to ledger

### Credit Multipliers

Base rate × Multipliers:
- **Reliability**: Consistent uptime bonus
- **Surge**: High-demand periods
- **Hardware**: Faster GPUs earn more per minute

### Using Credits

Spend credits to:
- Vote on job priorities
- Boost your own jobs (if operator allows)
- Trade (if secondary market exists)

## FAQ

**Q: Does this slow down my computer?**
A: Only during scheduled contribution times. Set lower % for work hours.

**Q: Can I use my GPU while contributing?**
A: Yes, but performance will be reduced by your donation percentage.

**Q: Is it safe?**
A: Yes. Code is open source, no personal data collected, sandboxed execution.

**Q: How much can I earn?**
A: Depends on GPU power and contribution time. Track record shows RTX 4070 earns ~50 credits/hour at 90%.

**Q: What if I lose internet?**
A: Tasks pause and resume. Progress not lost for most task types.

**Q: Can I run multiple nodes?**
A: Yes, one per GPU. Each gets unique node ID.

## Support

- **Issues (primary support)**: https://github.com/naxium-oss/DistribAI/issues
