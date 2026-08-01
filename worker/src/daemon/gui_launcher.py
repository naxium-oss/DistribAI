#!/usr/bin/env python3
"""
DistribAI Node GUI Launcher - Production Version

Provides a desktop GUI for the node using PyWebView.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

try:
    import webview  # type: ignore[import-not-found]
except ImportError:  # headless / Linux servers without GTK / minimal PyInstaller bundle
    webview = None  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "services_python"))

from worker.src.daemon.daemon import WorkerDaemon
from worker.src.daemon.scheduler_config import (
    ComputeProfile,
    TimeSlot,
    detect_local_timezone,
    get_schedule_manager,
)


class NodeAPI:
    """JavaScript API bridge for the Node GUI."""

    def __init__(self):
        self.window: webview.Window | None = None
        self.daemon: WorkerDaemon | None = None
        self.server_address: str = ""
        self.connected: bool = False
        self.server_locked: bool = os.getenv("DISTRIBAI_LOCK_SERVER", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self.auto_start_on_launch: bool = os.getenv(
            "DISTRIBAI_NODE_AUTOSTART", ""
        ).lower() in ("1", "true", "yes")
        self.total_vram_gb: float = 0.0
        self.has_cuda: bool = False
        self.cuda_downloaded: bool = False
        self.node_id: str = ""
        self.start_time: float = 0.0
        self.credits_earned: float = 0.0
        self.jobs_completed: int = 0
        self.current_task: dict | None = None
        self._daemon_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def set_window(self, window):
        self.window = window
        self._detect_hardware()
        self.node_id = self._get_node_id()
        self._start_status_updater()

    def _start_status_updater(self):
        def update_loop():
            while not self._stop_event.is_set():
                if self.window and self.daemon:
                    try:
                        status = self._get_full_status()
                        self._emit_event("status_update", status)
                    except (RuntimeError, ValueError):
                        pass
                time.sleep(5)

        threading.Thread(target=update_loop, daemon=True).start()

    def _emit_event(self, event_type: str, data: dict):
        if self.window and webview is not None:
            try:
                js_code = f"if(window.__distribai_events){{window.__distribai_events.emit('{event_type}', {json.dumps(data)})}}"
                self.window.evaluate_js(js_code)
            except (RuntimeError, ValueError):
                pass

    def _detect_hardware(self):
        try:
            import torch

            self.has_cuda = torch.cuda.is_available()
            if self.has_cuda:
                self.total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            self.has_cuda = False

    def _get_node_id(self) -> str:
        config_dir = Path.home() / ".distribai"
        config_dir.mkdir(exist_ok=True)
        node_id_file = config_dir / "node_id"
        if node_id_file.exists():
            return node_id_file.read_text().strip()
        import uuid

        node_id = f"node-{uuid.uuid4().hex[:8]}"
        node_id_file.write_text(node_id)
        return node_id

    def _get_full_status(self) -> dict:
        manager = get_schedule_manager()
        should_contribute = manager.should_be_contributing()

        uptime = time.time() - self.start_time if self.start_time > 0 else 0

        return {
            "connected": self.connected,
            "server_address": self.server_address,
            "node_id": self.node_id,
            "uptime_seconds": int(uptime),
            "credits_earned": self.credits_earned,
            "jobs_completed": self.jobs_completed,
            "current_task": self.current_task,
            "should_contribute": should_contribute,
            "has_cuda": self.has_cuda,
            "next_change_seconds": manager.get_time_until_change(),
        }

    def get_connection_status(self) -> dict:
        return self._get_full_status()

    def connect(self, server_address: str) -> dict:
        if self.server_locked and server_address and server_address != self.server_address:
            return {
                "success": False,
                "message": (
                    "Server address is locked in this build. "
                    f"Already connected to {self.server_address}."
                ),
            }
        if self.server_locked and not server_address and self.server_address:
            server_address = self.server_address
        self.server_address = server_address

        try:
            host, port_str = server_address.rsplit(":", 1)
            port = int(port_str)
        except ValueError:
            return {"success": False, "message": "Invalid server address format. Use host:port"}

        try:
            self.daemon = WorkerDaemon(
                orchestrator_url=f"{host}:{port}",
                node_id=self.node_id,
            )

            def run_daemon():
                try:
                    asyncio = __import__("asyncio")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.daemon.run())
                except Exception as e:
                    print(f"[NodeDaemon] Error: {e}")

            self._daemon_thread = threading.Thread(target=run_daemon, daemon=True)
            self._daemon_thread.start()

            self.connected = True
            self.start_time = time.time()

            self._discover_update_url()

            return {"success": True, "message": f"Connected to {server_address}"}

        except Exception as e:
            return {"success": False, "message": f"Connection failed: {str(e)}"}

    def disconnect(self) -> dict:
        if self.server_locked:
            return {
                "success": False,
                "message": "Disconnect is disabled in this build. Close the window to stop contributing.",
            }
        if self.daemon:
            try:
                self.daemon.stop()
            except (RuntimeError, OSError):
                pass
            self.daemon = None

        self.connected = False
        self._stop_event.set()
        return {"success": True, "message": "Disconnected"}

    def _discover_update_url(self):
        try:
            host = (
                self.server_address.split(":")[0]
                if ":" in self.server_address
                else self.server_address
            )
            admin_port = os.getenv("ADMIN_PORT", "8766")
            url = f"http://{host}:{admin_port}/admin/update-url"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read())
                self.update_url = data.get("update_url", "")
        except Exception:
            self.update_url = ""

    def get_hardware_info(self) -> dict:
        info = {
            "has_cuda": self.has_cuda,
            "cuda_downloaded": self.cuda_downloaded,
            "total_vram_gb": self.total_vram_gb,
            "gpu_name": "Unknown",
            "cuda_version": None,
        }

        if self.has_cuda:
            try:
                import torch

                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["cuda_version"] = torch.version.cuda
            except (RuntimeError, AttributeError):
                pass

        return info

    def rescan_cuda(self) -> dict:
        self._detect_hardware()
        return self.get_hardware_info()

    def get_cuda_download_size(self) -> dict:
        components = {
            "cuda_runtime": 85,
            "cudnn": 450,
            "cublas": 280,
            "total": 815,
        }
        return {"size_mb": components["total"], "components": components}

    def download_cuda(self) -> dict:
        """CUDA cannot be installed reliably from the node GUI (driver + toolkit).

        Users should install the NVIDIA driver and a CUDA-enabled PyTorch build from
        https://pytorch.org/get-started/locally/ — do not download arbitrary wheels as ``zip``.
        """
        return {
            "success": False,
            "message": (
                "CUDA is provided by your NVIDIA driver and your PyTorch install. "
                "Install drivers from NVIDIA and reinstall PyTorch with the matching CUDA "
                "variant from https://pytorch.org/get-started/locally/"
            ),
        }

    def get_schedule(self) -> dict:
        manager = get_schedule_manager()
        return manager.to_display_dict(self.total_vram_gb or 8.0)

    def save_schedule(self, schedule_data: dict) -> dict:
        try:
            manager = get_schedule_manager()

            manager.timezone_offset_hours = schedule_data.get(
                "timezone_offset_hours", detect_local_timezone()
            )
            manager.auto_join = schedule_data.get("auto_join", False)
            manager.auto_leave = schedule_data.get("auto_leave", True)

            default_data = schedule_data.get("default_profile", {})
            manager.default_profile = ComputeProfile(
                gpu_percent=default_data.get("gpu_percent", 0),
                vram_limit_gb=default_data.get("vram_limit_gb"),
                cpu_percent=default_data.get("cpu_percent", 50),
            )

            manager.time_slots = []
            for slot_data in schedule_data.get("time_slots", []):
                profile_data = slot_data.get("compute_profile", {})
                profile = ComputeProfile(
                    gpu_percent=profile_data.get("gpu_percent", 90),
                    vram_limit_gb=profile_data.get("vram_limit_gb"),
                    cpu_percent=profile_data.get("cpu_percent", 50),
                )

                from datetime import time as dt_time

                start_str = slot_data.get("start_time", "09:00")
                end_str = slot_data.get("end_time", "17:00")

                start_h, start_m = map(int, start_str.split(":"))
                end_h, end_m = map(int, end_str.split(":"))

                slot = TimeSlot.from_local(
                    start_time=dt_time(start_h, start_m),
                    end_time=dt_time(end_h, end_m),
                    timezone_offset_hours=manager.timezone_offset_hours,
                    compute_profile=profile,
                    days_of_week=slot_data.get("days_of_week", list(range(7))),
                    label=slot_data.get("label", "Schedule"),
                )
                manager.add_slot(slot)

            manager.save()
            return {"success": True, "message": "Schedule saved"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def get_timezone_offset(self) -> dict:
        return {"offset_hours": detect_local_timezone()}

    def get_compute_display(self, gpu_percent: float) -> str:
        profile = ComputeProfile(gpu_percent=gpu_percent)
        vram = self.total_vram_gb if self.total_vram_gb > 0 else 8.0
        return profile.format_gpu_display(vram)

    def get_job_list(self) -> list:
        if not self.connected or not self.daemon:
            return []
        jobs = self.daemon.snapshot_current_jobs_for_gui()
        if jobs:
            return jobs
        return [{"id": "idle", "name": "Idle — waiting for assignment", "priority": "normal"}]

    def get_status(self) -> dict:
        return self._get_full_status()

    def get_auto_start_status(self) -> dict:
        enabled = self._is_auto_start_enabled()
        return {
            "enabled": enabled,
            "platform": platform.system(),
            "can_toggle": True,
        }

    def set_auto_start(self, enabled: bool) -> dict:
        try:
            system = platform.system()

            if system == "Windows":
                self._set_windows_auto_start(enabled)
            elif system == "Darwin":
                self._set_macos_auto_start(enabled)
            elif system == "Linux":
                self._set_linux_auto_start(enabled)

            return {"success": True, "enabled": enabled}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _is_auto_start_enabled(self) -> bool:
        system = platform.system()

        if system == "Windows":
            startup_dir = (
                Path.home()
                / "AppData"
                / "Roaming"
                / "Microsoft"
                / "Windows"
                / "Start Menu"
                / "Programs"
                / "Startup"
            )
            return (startup_dir / "DistribAI-Node.lnk").exists()

        elif system == "Darwin":
            launch_agents = Path.home() / "Library" / "LaunchAgents"
            return (launch_agents / "io.distribai.node.plist").exists()

        elif system == "Linux":
            systemd_user = Path.home() / ".config" / "systemd" / "user"
            return (systemd_user / "distribai-node.service").exists()

        return False

    def _set_windows_auto_start(self, enabled: bool):
        startup_dir = (
            Path.home()
            / "AppData"
            / "Roaming"
            / "Microsoft"
            / "Windows"
            / "Start Menu"
            / "Programs"
            / "Startup"
        )
        startup_dir.mkdir(parents=True, exist_ok=True)
        shortcut_path = startup_dir / "DistribAI-Node.lnk"

        if enabled:
            exe_path = sys.executable
            if getattr(sys, "frozen", False):
                exe_path = sys.executable
            else:
                exe_path = str(Path(__file__).parent / "gui_launcher.py")

            import win32com.client

            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(str(shortcut_path))
            shortcut.Targetpath = exe_path
            shortcut.WorkingDirectory = str(Path(__file__).parent.parent.parent)
            shortcut.save()
        else:
            if shortcut_path.exists():
                shortcut_path.unlink()

    def _set_macos_auto_start(self, enabled: bool):
        launch_agents = Path.home() / "Library" / "LaunchAgents"
        launch_agents.mkdir(parents=True, exist_ok=True)
        plist_path = launch_agents / "io.distribai.node.plist"

        if enabled:
            exe_path = (
                sys.executable
                if getattr(sys, "frozen", False)
                else str(Path(__file__).parent / "gui_launcher.py")
            )

            plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>io.distribai.node</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exe_path}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>"""
            plist_path.write_text(plist_content)
            subprocess.run(["launchctl", "load", str(plist_path)], check=False)
        else:
            if plist_path.exists():
                subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
                plist_path.unlink()

    def _set_linux_auto_start(self, enabled: bool):
        systemd_user = Path.home() / ".config" / "systemd" / "user"
        systemd_user.mkdir(parents=True, exist_ok=True)
        service_path = systemd_user / "distribai-node.service"

        if enabled:
            exe_path = (
                sys.executable
                if getattr(sys, "frozen", False)
                else str(Path(__file__).parent / "gui_launcher.py")
            )

            service_content = f"""[Unit]
Description=DistribAI Node
After=network.target

[Service]
Type=simple
ExecStart={exe_path}
Restart=always
RestartSec=10

[Install]
WantedBy=default.target"""
            service_path.write_text(service_content)
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
            subprocess.run(["systemctl", "--user", "enable", "distribai-node"], check=False)
            subprocess.run(["systemctl", "--user", "start", "distribai-node"], check=False)
        else:
            if service_path.exists():
                subprocess.run(["systemctl", "--user", "stop", "distribai-node"], check=False)
                subprocess.run(["systemctl", "--user", "disable", "distribai-node"], check=False)
                service_path.unlink()
                subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)

    def check_for_updates(self) -> dict:
        try:
            if not self.update_url:
                return {
                    "update_available": False,
                    "current_version": "1.0.0",
                    "latest_version": "1.0.0",
                }

            version_url = f"{self.update_url}/version.json"
            with urllib.request.urlopen(version_url, timeout=10) as response:
                remote_info = json.loads(response.read())

            current = "1.0.0"
            latest = remote_info.get("version", "1.0.0")

            update_available = self._version_compare(latest, current) > 0

            return {
                "update_available": update_available,
                "current_version": current,
                "latest_version": latest,
                "download_url": remote_info.get("download_url", ""),
                "download_size_mb": remote_info.get("size_mb", 0),
                "release_notes": remote_info.get("notes", ""),
            }
        except Exception as e:
            return {
                "update_available": False,
                "current_version": "1.0.0",
                "latest_version": "1.0.0",
                "error": str(e),
            }

    def _version_compare(self, v1: str, v2: str) -> int:
        parts1 = [int(x) for x in v1.split(".")]
        parts2 = [int(x) for x in v2.split(".")]

        for i in range(max(len(parts1), len(parts2))):
            p1 = parts1[i] if i < len(parts1) else 0
            p2 = parts2[i] if i < len(parts2) else 0
            if p1 > p2:
                return 1
            elif p1 < p2:
                return -1
        return 0

    def download_update(self, download_url: str) -> dict:
        try:
            download_dir = Path.home() / ".distribai" / "updates"
            download_dir.mkdir(parents=True, exist_ok=True)

            ext = (
                ".exe"
                if platform.system() == "Windows"
                else ".zip"
                if platform.system() == "Darwin"
                else ".tar.gz"
            )
            download_path = download_dir / f"update{ext}"

            def progress_callback(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(int(downloaded * 100 / total_size), 100)
                self._emit_event("update_download_progress", {"progress": percent})

            urllib.request.urlretrieve(download_url, download_path, reporthook=progress_callback)

            expected_hash = self._get_expected_hash()
            if expected_hash:
                actual_hash = self._hash_file(download_path)
                if actual_hash != expected_hash:
                    download_path.unlink()
                    return {"success": False, "message": "Hash verification failed"}

            return {"success": True, "download_path": str(download_path)}

        except Exception as e:
            return {"success": False, "message": str(e)}

    def _get_expected_hash(self) -> str:
        try:
            hash_url = f"{self.update_url}/version.json"
            with urllib.request.urlopen(hash_url, timeout=10) as response:
                info = json.loads(response.read())
            return info.get("hash", "")
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            return ""

    def _hash_file(self, path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def install_update(self, download_path: str) -> dict:
        try:
            if platform.system() == "Windows":
                subprocess.Popen([str(download_path), "/SILENT"])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(download_path)])
            else:
                subprocess.Popen(["bash", str(download_path)])

            self._stop_event.set()
            sys.exit(0)

        except Exception as e:
            return {"success": False, "message": str(e)}


def get_html_content() -> str:
    dashboard_path = Path(__file__).parent.parent / "dashboard" / "static" / "node" / "index.html"
    if dashboard_path.exists():
        return str(dashboard_path)
    return ""


def _maybe_register_autostart(api: "NodeAPI") -> None:
    if not api.auto_start_on_launch:
        return
    sentinel = Path.home() / ".distribai" / ".autostart_registered"
    if sentinel.is_file():
        return
    try:
        if api.server_locked:
            api.set_auto_start(True)
    except (RuntimeError, OSError, ValueError):
        return
    try:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("registered", encoding="utf-8")
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(description="DistribAI Node GUI")
    parser.add_argument("--server", default="localhost:50051", help="Default server address")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without opening a window (headless / CI / minimal bundle).",
    )
    args = parser.parse_args()
    headless = args.no_gui or webview is None or os.getenv("DISTRIBAI_HEADLESS", "").lower() in (
        "1",
        "true",
        "yes",
    )

    api = NodeAPI()
    if api.server_locked:
        api.server_address = os.getenv("ORCHESTRATOR_URL", args.server)
    else:
        api.server_address = args.server

    html = get_html_content()

    if not headless:
        window = webview.create_window(
            "DistribAI Node",
            html if html and not html.startswith("<") else f"file://{html}" if html else "",
            js_api=api,
            width=1200,
            height=800,
            min_size=(800, 600),
            resizable=True,
            text_select=True,
            confirm_close=True,
        )
        api.set_window(window)

    if api.server_locked and api.server_address:
        try:
            api.connect(api.server_address)
        except (RuntimeError, ValueError, OSError):
            pass

    _maybe_register_autostart(api)

    if headless:
        print(f"[DistribAI] running headless, connected to {api.server_address or '(unset)'}")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n[DistribAI] shutting down (Ctrl-C)")
        return

    webview.start(debug=False, http_server=True if html else False)


if __name__ == "__main__":
    main()
