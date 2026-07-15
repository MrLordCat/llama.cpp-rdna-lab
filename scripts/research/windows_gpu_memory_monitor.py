#!/usr/bin/env python3
"""Measure per-process WDDM dedicated/shared memory during a benchmark."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import psutil


COUNTER_SCRIPT = r"""
$ci=[System.Globalization.CultureInfo]::InvariantCulture
$c=Get-Counter @(
    '\GPU Process Memory(*)\Dedicated Usage',
    '\GPU Process Memory(*)\Shared Usage'
) -ErrorAction SilentlyContinue
if ($c) {
    $c.CounterSamples | ForEach-Object {
        $_.Path + '|' + $_.CookedValue.ToString($ci)
    }
}
"""

COUNTER_RE = re.compile(
    r"gpu process memory\(pid_(\d+)_(luid_.+?_phys_\d+)\)\\(dedicated|shared) usage",
    re.IGNORECASE,
)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def find_process(name: str, deadline: float) -> psutil.Process:
    wanted = name.lower()
    while time.monotonic() < deadline:
        matches: list[psutil.Process] = []
        for process in psutil.process_iter(["name", "create_time"]):
            try:
                if (process.info["name"] or "").lower() == wanted:
                    matches.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if matches:
            return max(matches, key=lambda process: process.info.get("create_time") or 0.0)
        time.sleep(0.25)
    raise TimeoutError(f"process {name!r} did not appear before the monitor deadline")


def query_wddm(pid: int, timeout: float = 8.0) -> tuple[dict[str, float], dict[str, float]]:
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", COUNTER_SCRIPT],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            creationflags=NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return {}, {}

    dedicated: dict[str, float] = {}
    shared: dict[str, float] = {}
    for line in result.stdout.splitlines():
        path, separator, raw_value = line.rpartition("|")
        if not separator:
            continue
        match = COUNTER_RE.search(path)
        if not match or int(match.group(1)) != pid:
            continue
        try:
            value = float(raw_value.strip().replace(",", "."))
        except ValueError:
            continue
        target = dedicated if match.group(3).lower() == "dedicated" else shared
        target[match.group(2).lower()] = value
    return dedicated, shared


def current_phase(server_log: Path | None) -> str:
    if server_log is None or not server_log.exists():
        return "startup"
    try:
        with server_log.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 16_384), os.SEEK_SET)
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return "startup"
    if "prompt processing done" in tail:
        return "decode"
    if "new prompt" in tail:
        return "prefill"
    return "startup"


def update_peaks(peaks: dict[str, float], values: dict[str, float]) -> None:
    for key, value in values.items():
        peaks[key] = max(peaks.get(key, 0.0), value)


def gib(value: float) -> float:
    return round(value / (1024**3), 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-name", default="llama-server.exe")
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--wait-seconds", type=float, default=330.0)
    parser.add_argument("--poll-seconds", type=float, default=0.75)
    parser.add_argument("--max-runtime-seconds", type=float, default=900.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    process = find_process(args.process_name, time.monotonic() + args.wait_seconds)
    create_time = process.create_time()
    pid = process.pid
    started = time.monotonic()
    samples = 0
    counter_timeouts = 0
    dedicated_peaks: dict[str, float] = {}
    shared_peaks: dict[str, float] = {}
    phase_peaks: dict[str, dict[str, float]] = {}
    private_peak = 0.0
    working_peak = 0.0

    while True:
        try:
            try:
                process.wait(timeout=0)
            except psutil.TimeoutExpired:
                pass
            else:
                break
            if process.create_time() != create_time:
                break
            memory = process.memory_info()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break

        if time.monotonic() - started >= args.max_runtime_seconds:
            break

        working = float(memory.rss)
        private = float(getattr(memory, "private", memory.vms))
        working_peak = max(working_peak, working)
        private_peak = max(private_peak, private)

        dedicated, shared = query_wddm(pid)
        if not dedicated and not shared:
            counter_timeouts += 1
        update_peaks(dedicated_peaks, dedicated)
        update_peaks(shared_peaks, shared)

        phase = current_phase(args.server_log)
        phase_values = phase_peaks.setdefault(
            phase,
            {"dedicated": 0.0, "shared": 0.0, "private": 0.0, "working": 0.0},
        )
        phase_values["dedicated"] = max(phase_values["dedicated"], sum(dedicated.values()))
        phase_values["shared"] = max(phase_values["shared"], sum(shared.values()))
        phase_values["private"] = max(phase_values["private"], private)
        phase_values["working"] = max(phase_values["working"], working)
        samples += 1
        time.sleep(max(0.1, args.poll_seconds))

    output = {
        "pid": pid,
        "elapsed_s": round(time.monotonic() - started, 1),
        "samples": samples,
        "empty_counter_samples": counter_timeouts,
        "dedicated_peak_by_luid_gib": {key: gib(value) for key, value in sorted(dedicated_peaks.items())},
        "shared_peak_by_luid_gib": {key: gib(value) for key, value in sorted(shared_peaks.items())},
        "private_peak_gib": gib(private_peak),
        "working_peak_gib": gib(working_peak),
        "phases": {
            phase: {f"{key}_gib": gib(value) for key, value in values.items()}
            for phase, values in sorted(phase_peaks.items())
        },
    }
    rendered = json.dumps(output, separators=(",", ":"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
