"""Live server monitor: GPU/RAM/CPU usage + llama-server /metrics throughput.

ServerMonitorThread polls system stats (psutil + Windows GPU perf counters)
and, while a server is running, its Prometheus /metrics endpoint (the launch
command includes --metrics). ServerMonitorPanel renders the numbers.
"""

from __future__ import annotations

import os
import re
import subprocess

try:
    import psutil
except ImportError:  # psutil is in requirements-gui.txt; degrade gracefully
    psutil = None

try:
    import requests
except ImportError:
    requests = None

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QGroupBox, QLabel, QVBoxLayout

_GPU_COUNTER_PS = (
    "$ci=[System.Globalization.CultureInfo]::InvariantCulture; "
    "(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage',"
    "'\\GPU Engine(*)\\Utilization Percentage' -ErrorAction SilentlyContinue)"
    ".CounterSamples | ForEach-Object { $_.Path + '|' + $_.CookedValue.ToString($ci) }"
)

# adapters are identified by their full LUID; phys_N alone collides across cards
_VRAM_RE = re.compile(r"gpu adapter memory\((luid_.+?_phys_\d+)\)\\dedicated usage", re.IGNORECASE)
_UTIL_RE = re.compile(r"gpu engine\(pid_\d+_(luid_.+?_phys_\d+)_engtype_\w+\)\\utilization percentage", re.IGNORECASE)


def _query_gpu_counters(timeout: float = 6.0) -> list[dict]:
    """Per-adapter VRAM usage (GB) and load (%) via Windows perf counters."""
    if os.name != "nt":
        return []
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _GPU_COUNTER_PS],
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=creationflags,
        )
    except Exception:
        return []

    vram_by_luid: dict[str, float] = {}
    util_by_luid: dict[str, float] = {}
    for line in result.stdout.splitlines():
        if "|" not in line:
            continue
        path, _, raw_value = line.rpartition("|")
        try:
            value = float(raw_value.strip().replace(",", "."))
        except ValueError:
            continue

        vram_match = _VRAM_RE.search(path)
        if vram_match:
            luid = vram_match.group(1).lower()
            vram_by_luid[luid] = vram_by_luid.get(luid, 0.0) + value
            continue
        util_match = _UTIL_RE.search(path)
        if util_match and value > 0:
            luid = util_match.group(1).lower()
            util_by_luid[luid] = util_by_luid.get(luid, 0.0) + value

    gpus = []
    for luid in sorted(set(vram_by_luid) | set(util_by_luid)):
        vram_gb = vram_by_luid.get(luid, 0.0) / (1024 ** 3)
        if vram_gb < 0.05 and util_by_luid.get(luid, 0.0) <= 0:
            continue  # ghost adapters (e.g. remote display) with no usage
        gpus.append({
            "index": len(gpus),
            "vram_gb": vram_gb,
            "util_pct": min(100.0, util_by_luid.get(luid, 0.0)),
        })
    return gpus


def _parse_prometheus(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("llamacpp:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{", 1)[0]
        try:
            metrics[name] = float(parts[-1])
        except ValueError:
            continue
    return metrics


class ServerMonitorThread(QThread):
    """Polls system + server stats every few seconds and emits one dict."""

    stats_ready = pyqtSignal(dict)

    POLL_INTERVAL_MS = 3000

    def __init__(self, parent=None):
        super().__init__(parent)
        # written from the GUI thread, read here; assignment is atomic enough
        self._base_url: str = ""
        self._api_key: str = ""
        self._server_pid: int | None = None

    def set_server(self, base_url: str, api_key: str = "", pid: int | None = None) -> None:
        self._api_key = api_key
        self._server_pid = pid
        self._base_url = base_url.rstrip("/")

    def clear_server(self) -> None:
        self._base_url = ""
        self._server_pid = None

    def stop(self) -> None:
        self.requestInterruption()
        self.wait(2000)

    def run(self):
        while not self.isInterruptionRequested():
            stats: dict = {"server": None, "gpus": [], "system": {}}

            if psutil is not None:
                try:
                    memory = psutil.virtual_memory()
                    stats["system"] = {
                        "ram_used_gb": (memory.total - memory.available) / (1024 ** 3),
                        "ram_total_gb": memory.total / (1024 ** 3),
                        "cpu_pct": psutil.cpu_percent(interval=None),
                    }
                    if self._server_pid:
                        try:
                            proc = psutil.Process(self._server_pid)
                            stats["system"]["proc_rss_gb"] = proc.memory_info().rss / (1024 ** 3)
                        except Exception:
                            pass
                except Exception:
                    pass

            stats["gpus"] = _query_gpu_counters()

            base_url = self._base_url
            if base_url and requests is not None:
                try:
                    headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
                    response = requests.get(f"{base_url}/metrics", headers=headers, timeout=1.5)
                    if response.status_code == 200:
                        metrics = _parse_prometheus(response.text)

                        def ratio(numerator: str, denominator: str) -> float:
                            den = metrics.get(denominator, 0.0)
                            return metrics.get(numerator, 0.0) / den if den > 0 else 0.0

                        stats["server"] = {
                            "prompt_tps_avg": metrics.get("llamacpp:prompt_tokens_seconds", 0.0)
                            or ratio("llamacpp:prompt_tokens_total", "llamacpp:prompt_seconds_total"),
                            "decode_tps_avg": metrics.get("llamacpp:predicted_tokens_seconds", 0.0)
                            or ratio("llamacpp:tokens_predicted_total", "llamacpp:tokens_predicted_seconds_total"),
                            "prompt_tokens_total": metrics.get("llamacpp:prompt_tokens_total", 0.0),
                            "predicted_tokens_total": metrics.get("llamacpp:tokens_predicted_total", 0.0),
                            "kv_pct": metrics.get("llamacpp:kv_cache_usage_ratio", 0.0) * 100.0,
                            "busy": int(metrics.get("llamacpp:requests_processing", 0.0)),
                            "deferred": int(metrics.get("llamacpp:requests_deferred", 0.0)),
                        }
                except Exception:
                    stats["server"] = {"error": True}

            self.stats_ready.emit(stats)

            # sleep in small slices so stop() stays responsive
            for _ in range(self.POLL_INTERVAL_MS // 200):
                if self.isInterruptionRequested():
                    return
                self.msleep(200)


class ServerMonitorPanel(QGroupBox):
    """Compact live readout: server throughput, GPU, RAM/CPU."""

    def __init__(self, parent=None):
        super().__init__("Live Monitor", parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setSpacing(2)

        self.server_line = QLabel("Server: stopped")
        self.speed_line = QLabel("Prompt: — · Decode: —")
        self.gpu_line = QLabel("GPU: —")
        self.system_line = QLabel("RAM: — · CPU: —")
        for label in (self.server_line, self.speed_line, self.gpu_line, self.system_line):
            label.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 11px;")
            layout.addWidget(label)

    def update_stats(self, stats: dict) -> None:
        server = stats.get("server")
        if server is None:
            self.server_line.setText("Server: stopped")
            self.speed_line.setText("Prompt: — · Decode: —")
        elif server.get("error"):
            self.server_line.setText("Server: metrics unavailable")
            self.speed_line.setText("Prompt: — · Decode: —")
        else:
            busy = server.get("busy", 0)
            deferred = server.get("deferred", 0)
            queue_text = f", queue {deferred}" if deferred else ""
            self.server_line.setText(
                f"Server: ● running ({busy} req{queue_text}) · KV cache {server.get('kv_pct', 0.0):.0f}%"
            )
            prompt_total = int(server.get("prompt_tokens_total", 0))
            decode_total = int(server.get("predicted_tokens_total", 0))
            self.speed_line.setText(
                f"Prompt avg: {server.get('prompt_tps_avg', 0.0):.1f} tok/s ({prompt_total} tok) · "
                f"Decode avg: {server.get('decode_tps_avg', 0.0):.2f} tok/s ({decode_total} tok)"
            )

        gpus = stats.get("gpus") or []
        if gpus:
            self.gpu_line.setText("   ".join(
                f"GPU{gpu['index']}: {gpu['util_pct']:.0f}% · {gpu['vram_gb']:.1f} GB VRAM"
                for gpu in gpus
            ))
        else:
            self.gpu_line.setText("GPU: —")

        system = stats.get("system") or {}
        if system:
            proc_text = f"server {system['proc_rss_gb']:.1f} GB · " if "proc_rss_gb" in system else ""
            self.system_line.setText(
                f"RAM: {proc_text}sys {system.get('ram_used_gb', 0.0):.1f}/{system.get('ram_total_gb', 0.0):.0f} GB · "
                f"CPU: {system.get('cpu_pct', 0.0):.0f}%"
            )
        else:
            self.system_line.setText("RAM: — · CPU: —")
