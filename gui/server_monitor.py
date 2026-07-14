"""Live server monitor: GPU/RAM/CPU usage + llama-server /metrics throughput.

ServerMonitorThread polls system stats (psutil + Windows GPU perf counters)
and, while a server is running, its Prometheus /metrics endpoint (the launch
command includes --metrics). ServerMonitorPanel renders progress bars.
"""

from __future__ import annotations

import os
import re
import subprocess
import time

try:
    import psutil
except ImportError:  # psutil is in requirements-gui.txt; degrade gracefully
    psutil = None

try:
    import requests
except ImportError:
    requests = None

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

# NB: "Utilization Percentage" is a rate counter — a single sample always reads
# 0, so the engine query takes two samples 1s apart and reports the second one.
_GPU_COUNTER_PS = (
    "$ci=[System.Globalization.CultureInfo]::InvariantCulture; "
    "(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue)"
    ".CounterSamples | ForEach-Object { $_.Path + '|' + $_.CookedValue.ToString($ci) }; "
    "$u = Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -SampleInterval 1 -MaxSamples 2 -ErrorAction SilentlyContinue; "
    "if ($u) { $u[-1].CounterSamples | ForEach-Object { $_.Path + '|' + $_.CookedValue.ToString($ci) } }"
)

# dedicated VRAM size (bytes) + adapter name per display adapter, from the
# driver registry keys; one "bytes|name" line per adapter
_GPU_TOTALS_PS = (
    "Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\"
    "{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' -Name 'HardwareInformation.qwMemorySize' "
    "-ErrorAction SilentlyContinue | ForEach-Object { "
    "$desc = (Get-ItemProperty $_.PSPath -Name DriverDesc -ErrorAction SilentlyContinue).DriverDesc; "
    "$_.'HardwareInformation.qwMemorySize'.ToString() + '|' + $desc }"
)

# adapters are identified by their full LUID; phys_N alone collides across cards
_VRAM_RE = re.compile(r"gpu adapter memory\((luid_.+?_phys_\d+)\)\\dedicated usage", re.IGNORECASE)
_UTIL_RE = re.compile(r"gpu engine\(pid_\d+_(luid_.+?_phys_\d+)_engtype_\w+\)\\utilization percentage", re.IGNORECASE)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _run_powershell(script: str, timeout: float) -> str:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=_NO_WINDOW,
    )
    return result.stdout


def _query_gpu_totals(timeout: float = 8.0) -> tuple[list[float], list[str]]:
    """Total dedicated VRAM in GB and adapter name per adapter, best effort."""
    if os.name != "nt":
        return [], []
    try:
        output = _run_powershell(_GPU_TOTALS_PS, timeout)
    except Exception:
        return [], []
    totals: list[float] = []
    names: list[str] = []
    for line in output.splitlines():
        size_text, _, name = line.strip().partition("|")
        if size_text.isdigit() and int(size_text) > 0:
            totals.append(int(size_text) / (1024 ** 3))
            names.append(name.strip())
    return totals, names


def _short_gpu_name(name: str) -> str:
    """'AMD Radeon RX 9070 XT' -> 'RX 9070 XT' — fits the monitor row label."""
    for noise in ("AMD ", "NVIDIA ", "Intel(R) ", "Radeon(TM) ", "Radeon ", "GeForce ", "Graphics "):
        name = name.replace(noise, "")
    return name.strip()


def _query_gpu_counters(timeout: float = 12.0) -> list[dict]:
    """Per-adapter VRAM usage (GB) and load (%) via Windows perf counters."""
    if os.name != "nt":
        return []
    try:
        output = _run_powershell(_GPU_COUNTER_PS, timeout)
    except Exception:
        return []

    vram_by_luid: dict[str, float] = {}
    util_by_luid: dict[str, float] = {}
    for line in output.splitlines():
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
    for luid in sorted(vram_by_luid):
        vram = vram_by_luid.get(luid, 0.0)
        util = util_by_luid.get(luid, 0.0)
        # ghost adapters (basic render etc.) report exactly 0 committed bytes;
        # a real idle card still holds a few MB of driver allocations
        if vram <= 0.0 and util <= 0.0:
            continue
        gpus.append({
            "index": len(gpus),
            "vram_gb": vram / (1024 ** 3),
            "util_pct": min(100.0, util),
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

    # the GPU query itself takes ~2-3s (two-sample rate counters), so the
    # effective refresh period is roughly POLL_INTERVAL_MS + 3s
    POLL_INTERVAL_MS = 2000

    def __init__(self, parent=None):
        super().__init__(parent)
        # written from the GUI thread, read here; assignment is atomic enough
        self._base_url: str = ""
        self._api_key: str = ""
        self._server_pid: int | None = None
        self._ctx_tokens: int = 0
        self._prev_totals: tuple[float, float, float] | None = None  # (t, prompt, decode)
        self._gpu_totals: list[float] | None = None
        self._gpu_names: list[str] = []

    def set_server(self, base_url: str, api_key: str = "", pid: int | None = None, ctx_tokens: int = 0) -> None:
        self._api_key = api_key
        self._server_pid = pid
        self._ctx_tokens = ctx_tokens
        self._prev_totals = None
        self._base_url = base_url.rstrip("/")

    def clear_server(self) -> None:
        self._base_url = ""
        self._server_pid = None
        self._prev_totals = None

    def stop(self) -> None:
        self.requestInterruption()
        self.wait(2000)

    def _collect_server_stats(self) -> dict | None:
        base_url = self._base_url
        if not base_url or requests is None:
            return None
        try:
            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
            response = requests.get(f"{base_url}/metrics", headers=headers, timeout=1.5)
            if response.status_code != 200:
                return {"error": True}
            metrics = _parse_prometheus(response.text)
        except Exception:
            return {"error": True}

        def ratio(numerator: str, denominator: str) -> float:
            den = metrics.get(denominator, 0.0)
            return metrics.get(numerator, 0.0) / den if den > 0 else 0.0

        prompt_total = metrics.get("llamacpp:prompt_tokens_total", 0.0)
        decode_total = metrics.get("llamacpp:tokens_predicted_total", 0.0)

        # instantaneous throughput from counter deltas between ticks
        now = time.monotonic()
        prompt_now = decode_now = 0.0
        if self._prev_totals is not None:
            prev_time, prev_prompt, prev_decode = self._prev_totals
            elapsed = now - prev_time
            if elapsed > 0:
                prompt_now = max(0.0, prompt_total - prev_prompt) / elapsed
                decode_now = max(0.0, decode_total - prev_decode) / elapsed
        self._prev_totals = (now, prompt_total, decode_total)

        return {
            "prompt_tps_avg": metrics.get("llamacpp:prompt_tokens_seconds", 0.0)
            or ratio("llamacpp:prompt_tokens_total", "llamacpp:prompt_seconds_total"),
            "decode_tps_avg": metrics.get("llamacpp:predicted_tokens_seconds", 0.0)
            or ratio("llamacpp:tokens_predicted_total", "llamacpp:tokens_predicted_seconds_total"),
            "prompt_tps_now": prompt_now,
            "decode_tps_now": decode_now,
            "prompt_tokens_total": prompt_total,
            "predicted_tokens_total": decode_total,
            # this server build exports no KV-cache metrics; n_tokens_max is the
            # high watermark of occupied context tokens
            "tokens_peak": int(metrics.get("llamacpp:n_tokens_max", 0.0)),
            "ctx_tokens": self._ctx_tokens,
            "busy": int(metrics.get("llamacpp:requests_processing", 0.0)),
            "deferred": int(metrics.get("llamacpp:requests_deferred", 0.0)),
        }

    def run(self):
        if self._gpu_totals is None:
            self._gpu_totals, self._gpu_names = _query_gpu_totals()

        while not self.isInterruptionRequested():
            stats: dict = {
                "server": None,
                "gpus": [],
                "system": {},
                "gpu_totals": self._gpu_totals,
                "gpu_names": self._gpu_names,
            }

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
            stats["server"] = self._collect_server_stats()

            self.stats_ready.emit(stats)

            # sleep in small slices so stop() stays responsive
            for _ in range(self.POLL_INTERVAL_MS // 200):
                if self.isInterruptionRequested():
                    return
                self.msleep(200)


_BAR_STYLE = """
QProgressBar {
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    background-color: #202020;
    text-align: center;
    font-size: 10px;
    color: #dddddd;
    min-height: 14px;
    max-height: 14px;
}
QProgressBar::chunk { background-color: %s; border-radius: 2px; }
"""

_CHUNK_OK = "#2e7d32"
_CHUNK_WARN = "#b26a00"
_CHUNK_HOT = "#b71c1c"


def _make_bar() -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(True)
    bar.setStyleSheet(_BAR_STYLE % _CHUNK_OK)
    return bar


def _set_bar(bar: QProgressBar, pct: float, text: str) -> None:
    pct = max(0.0, min(100.0, pct))
    bar.setValue(int(round(pct)))
    bar.setFormat(text)
    chunk = _CHUNK_OK if pct < 75 else _CHUNK_WARN if pct < 92 else _CHUNK_HOT
    bar.setStyleSheet(_BAR_STYLE % chunk)


class _GpuRow(QWidget):
    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.name_label = QLabel(f"GPU{index}")
        self.name_label.setFixedWidth(92)
        layout.addWidget(self.name_label)
        self.util_bar = _make_bar()
        self.util_bar.setToolTip("GPU load (all engines)")
        layout.addWidget(self.util_bar, 2)
        self.vram_bar = _make_bar()
        self.vram_bar.setToolTip("Dedicated VRAM used / total")
        layout.addWidget(self.vram_bar, 3)
        self._named = False

    def set_name(self, index: int, full_name: str) -> None:
        if self._named or not full_name:
            return
        short = _short_gpu_name(full_name)
        if len(short) > 13:
            short = short[:12] + "…"
        self.name_label.setText(short or f"GPU{index}")
        self.name_label.setToolTip(f"GPU{index}: {full_name}")
        self._named = True


class ServerMonitorPanel(QGroupBox):
    """Live readout: server throughput, per-GPU load/VRAM, RAM/CPU bars."""

    def __init__(self, parent=None):
        super().__init__("Live Monitor", parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 6)
        root.setSpacing(4)

        top = QGridLayout()
        top.setHorizontalSpacing(8)
        top.setVerticalSpacing(2)

        self.server_state = QLabel("● stopped")
        self.server_state.setStyleSheet("color: #888888; font-weight: bold;")
        top.addWidget(QLabel("Server:"), 0, 0)
        top.addWidget(self.server_state, 0, 1)

        self.ctx_bar = _make_bar()
        self.ctx_bar.setToolTip("Peak occupied context tokens (llamacpp:n_tokens_max) vs configured ctx")
        top.addWidget(QLabel("Ctx:"), 0, 2)
        top.addWidget(self.ctx_bar, 0, 3)

        self.prompt_label = QLabel("—")
        self.decode_label = QLabel("—")
        for value_label in (self.prompt_label, self.decode_label):
            value_label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        top.addWidget(QLabel("Prompt:"), 1, 0)
        top.addWidget(self.prompt_label, 1, 1)
        top.addWidget(QLabel("Decode:"), 1, 2)
        top.addWidget(self.decode_label, 1, 3)
        top.setColumnStretch(1, 2)
        top.setColumnStretch(3, 3)
        root.addLayout(top)

        self._gpu_rows: list[_GpuRow] = []
        self._gpu_container = QVBoxLayout()
        self._gpu_container.setSpacing(2)
        root.addLayout(self._gpu_container)

        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        ram_name = QLabel("RAM")
        ram_name.setFixedWidth(92)
        bottom.addWidget(ram_name)
        self.ram_bar = _make_bar()
        bottom.addWidget(self.ram_bar, 3)
        cpu_name = QLabel("CPU")
        bottom.addWidget(cpu_name)
        self.cpu_bar = _make_bar()
        bottom.addWidget(self.cpu_bar, 2)
        root.addLayout(bottom)

    def _gpu_row(self, index: int) -> _GpuRow:
        while len(self._gpu_rows) <= index:
            row = _GpuRow(len(self._gpu_rows))
            self._gpu_rows.append(row)
            self._gpu_container.addWidget(row)
        return self._gpu_rows[index]

    def update_stats(self, stats: dict) -> None:
        server = stats.get("server")
        if server is None:
            self.server_state.setText("● stopped")
            self.server_state.setStyleSheet("color: #888888; font-weight: bold;")
            self.prompt_label.setText("—")
            self.decode_label.setText("—")
            _set_bar(self.ctx_bar, 0, "—")
        elif server.get("error"):
            self.server_state.setText("● no metrics")
            self.server_state.setStyleSheet("color: #d7a72d; font-weight: bold;")
            self.prompt_label.setText("—")
            self.decode_label.setText("—")
            _set_bar(self.ctx_bar, 0, "—")
        else:
            busy = server.get("busy", 0)
            deferred = server.get("deferred", 0)
            queue_text = f" +{deferred} queued" if deferred else ""
            self.server_state.setText(f"● running · {busy} req{queue_text}")
            self.server_state.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.prompt_label.setText(
                f"{server.get('prompt_tps_now', 0.0):7.1f} tok/s (avg {server.get('prompt_tps_avg', 0.0):.1f})"
            )
            self.decode_label.setText(
                f"{server.get('decode_tps_now', 0.0):6.2f} tok/s (avg {server.get('decode_tps_avg', 0.0):.2f})"
            )
            tokens_peak = server.get("tokens_peak", 0)
            ctx_tokens = server.get("ctx_tokens", 0)
            if ctx_tokens > 0:
                _set_bar(self.ctx_bar, tokens_peak / ctx_tokens * 100.0, f"peak {tokens_peak} / {ctx_tokens} tok")
            else:
                _set_bar(self.ctx_bar, 0, f"peak {tokens_peak} tok" if tokens_peak else "—")

        gpu_totals = stats.get("gpu_totals") or []
        gpu_names = stats.get("gpu_names") or []
        for gpu in stats.get("gpus") or []:
            row = self._gpu_row(gpu["index"])
            if gpu_names:
                row.set_name(gpu["index"], gpu_names[min(gpu["index"], len(gpu_names) - 1)])
            util = gpu.get("util_pct", 0.0)
            _set_bar(row.util_bar, util, f"{util:.0f}%")
            used = gpu.get("vram_gb", 0.0)
            # registry can hold stale duplicate entries; clamp the index
            total = gpu_totals[min(gpu["index"], len(gpu_totals) - 1)] if gpu_totals else 0.0
            if total > 0:
                _set_bar(row.vram_bar, used / total * 100.0, f"{used:.1f} / {total:.0f} GB")
            else:
                _set_bar(row.vram_bar, 0, f"{used:.1f} GB")

        system = stats.get("system") or {}
        ram_total = system.get("ram_total_gb", 0.0)
        if ram_total > 0:
            ram_used = system.get("ram_used_gb", 0.0)
            proc_text = f" · srv {system['proc_rss_gb']:.1f}" if "proc_rss_gb" in system else ""
            _set_bar(self.ram_bar, ram_used / ram_total * 100.0, f"{ram_used:.1f} / {ram_total:.0f} GB{proc_text}")
            cpu_pct = system.get("cpu_pct", 0.0)
            _set_bar(self.cpu_bar, cpu_pct, f"{cpu_pct:.0f}%")
