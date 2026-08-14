#!/usr/bin/env python3
"""Summarize default-off CUDA/HIP graph state and host timing traces."""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


STATE_RE = re.compile(
    r"ggml_backend_cuda_graph_compute: dev=(?P<dev>\d+) key=(?P<key>\S+) .*?"
    r"first_name=(?P<first_name>.*?) first_ne=\((?P<first_ne>[^)]*)\) "
    r"last_op=\S+ last_name=(?P<last_name>.*?) last_ne=\((?P<last_ne>[^)]*)\) "
    r"uid=(?P<uid>\d+) nodes=(?P<nodes>\d+) enabled=(?P<enabled>[01]) "
    r"compatible=(?P<compatible>[01]) props_changed=(?P<props>[01]) "
    r"warmup_before=(?P<warmup_before>[01]) warmup_after=(?P<warmup_after>[01]) "
    r"instance_before=(?P<instance_before>[01]) use=(?P<use>[01]) update=(?P<update>[01])"
)

HOST_RE = re.compile(
    r"GGML_TRACE_CUDA_GRAPH_HOST_TIMING: dev=(?P<dev>\d+) key=(?P<key>\S+) "
    r"nodes=(?P<nodes>\d+) uid=(?P<uid>\d+) use=(?P<use>[01]) "
    r"update=(?P<update>[01]) device_ms=(?P<device>[0-9.]+) "
    r"key_ms=(?P<key_ms>[0-9.]+) enabled_ms=(?P<enabled_ms>[0-9.]+) "
    r"compat_ms=(?P<compat_ms>[0-9.]+) props_ms=(?P<props_ms>[0-9.]+) "
    r"eval_ms=(?P<eval_ms>[0-9.]+) total_ms=(?P<total_ms>[0-9.]+)"
)

DEVICE_RE = re.compile(
    r"GGML_TRACE_CUDA_GRAPH_DEVICE_TIMING: dev=(?P<dev>\d+) key=(?P<key>\S+) "
    r"nodes=(?P<nodes>\d+) uid=(?P<uid>\d+) use=(?P<use>[01]) "
    r"update=(?P<update>[01]) device_ms=(?P<device_ms>-?[0-9.]+) flush=(?P<flush>[01])"
)


def iter_lines(paths: list[str]) -> Iterable[str]:
    for raw_path in paths:
        if raw_path == "-":
            yield from sys.stdin
            continue
        with Path(raw_path).open("r", encoding="utf-8", errors="replace") as source:
            yield from source


def parse_shape(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in raw.split(","))
    except ValueError:
        return ()


def first_rows_label(shape: tuple[int, ...]) -> tuple[str, str]:
    if len(shape) < 2:
        return "-", "unknown"
    rows = shape[1]
    if rows == 1:
        bucket = "N=1?"
    elif 2 <= rows <= 4:
        bucket = "N=2-4?"
    elif rows > 4:
        bucket = "N>4?"
    else:
        bucket = "unknown"
    return str(rows), bucket


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


def escape(value: str) -> str:
    return value.replace("|", "\\|") or "-"


def summarize(paths: list[str]) -> str:
    groups: dict[tuple[int, str, int], dict[str, object]] = defaultdict(
        lambda: {
            "state_calls": 0,
            "host_calls": 0,
            "replays": 0,
            "updates": 0,
            "props": 0,
            "incompatible": 0,
            "first_name": "",
            "last_name": "",
            "first_ne": (),
            "last_ne": (),
            "host_total_ms": [],
            "host_eval_ms": [],
            "host_direct_ms": [],
            "host_replay_ms": [],
            "device_ms": [],
        }
    )

    unmatched_state = 0
    unmatched_host = 0
    unmatched_device = 0
    invalid_device = 0
    for line in iter_lines(paths):
        if "ggml_backend_cuda_graph_compute: dev=" in line:
            match = STATE_RE.search(line)
            if not match:
                unmatched_state += 1
                continue
            key = (int(match["dev"]), match["key"], int(match["nodes"]))
            item = groups[key]
            item["state_calls"] = int(item["state_calls"]) + 1
            item["replays"] = int(item["replays"]) + int(match["use"])
            item["updates"] = int(item["updates"]) + int(match["update"])
            item["props"] = int(item["props"]) + int(match["props"])
            item["incompatible"] = int(item["incompatible"]) + (1 - int(match["compatible"]))
            item["first_name"] = match["first_name"]
            item["last_name"] = match["last_name"]
            item["first_ne"] = parse_shape(match["first_ne"])
            item["last_ne"] = parse_shape(match["last_ne"])
            continue

        if "GGML_TRACE_CUDA_GRAPH_HOST_TIMING:" in line:
            match = HOST_RE.search(line)
            if not match:
                unmatched_host += 1
                continue
            key = (int(match["dev"]), match["key"], int(match["nodes"]))
            item = groups[key]
            item["host_calls"] = int(item["host_calls"]) + 1
            total_ms = float(match["total_ms"])
            item["host_total_ms"].append(total_ms)  # type: ignore[union-attr]
            item["host_eval_ms"].append(float(match["eval_ms"]))  # type: ignore[union-attr]
            host_class = "host_replay_ms" if match["use"] == "1" else "host_direct_ms"
            item[host_class].append(total_ms)  # type: ignore[union-attr]
            continue

        if "GGML_TRACE_CUDA_GRAPH_DEVICE_TIMING:" in line:
            match = DEVICE_RE.search(line)
            if not match:
                unmatched_device += 1
                continue
            key = (int(match["dev"]), match["key"], int(match["nodes"]))
            device_ms = float(match["device_ms"])
            if device_ms < 0.0:
                invalid_device += 1
                continue
            groups[key]["device_ms"].append(device_ms)  # type: ignore[union-attr]

    state_calls = sum(int(item["state_calls"]) for item in groups.values())
    host_calls = sum(int(item["host_calls"]) for item in groups.values())
    replays = sum(int(item["replays"]) for item in groups.values())
    updates = sum(int(item["updates"]) for item in groups.values())
    props = sum(int(item["props"]) for item in groups.values())
    incompatible = sum(int(item["incompatible"]) for item in groups.values())
    host_total = sum(sum(item["host_total_ms"]) for item in groups.values())  # type: ignore[arg-type]
    host_direct = sum(sum(item["host_direct_ms"]) for item in groups.values())  # type: ignore[arg-type]
    host_replay = sum(sum(item["host_replay_ms"]) for item in groups.values())  # type: ignore[arg-type]
    device_calls = sum(len(item["device_ms"]) for item in groups.values())  # type: ignore[arg-type]
    device_total = sum(sum(item["device_ms"]) for item in groups.values())  # type: ignore[arg-type]

    output = [
        "# ROCm HIP graph trace summary",
        "",
        "> `N` buckets are a first-tensor `ne[1]` heuristic and require phase-trace confirmation.",
        "> Host tracing is diagnostic; do not use a traced run as an unbracketed TPS claim.",
        "",
        f"- graph groups: `{len(groups)}`",
        f"- state calls: `{state_calls}`; replay calls: `{replays}`; direct/capture calls: `{state_calls - replays}`",
        f"- updates: `{updates}`; property changes: `{props}`; incompatible calls: `{incompatible}`",
        f"- host timing calls: `{host_calls}`; summed host time: `{host_total:.3f} ms` "
        f"(direct `{host_direct:.3f}`, replay `{host_replay:.3f}`)",
        f"- device timing calls: `{device_calls}`; summed per-device graph time: `{device_total:.3f} ms`",
        f"- invalid negative device intervals excluded: `{invalid_device}`",
        f"- unmatched state/host/device lines: `{unmatched_state}/{unmatched_host}/{unmatched_device}`",
        "",
        "| dev | key | nodes | first rows | bucket | first -> last | calls | replay | update | props | host p50 ms | host p95 ms | host total ms | device p50 ms | device p95 ms | device total ms |",
        "| ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    ranked = sorted(
        groups.items(),
        key=lambda pair: sum(pair[1]["host_total_ms"]),  # type: ignore[arg-type]
        reverse=True,
    )
    for (device, graph_key, nodes), item in ranked:
        totals = item["host_total_ms"]  # type: ignore[assignment]
        device_times = item["device_ms"]  # type: ignore[assignment]
        rows, bucket = first_rows_label(item["first_ne"])  # type: ignore[arg-type]
        names = f"{escape(str(item['first_name']))} -> {escape(str(item['last_name']))}"
        output.append(
            f"| {device} | `{graph_key}` | {nodes} | {rows} | {bucket} | {names} | "
            f"{item['state_calls']} | {item['replays']} | {item['updates']} | {item['props']} | "
            f"{percentile(totals, 0.50):.3f} | {percentile(totals, 0.95):.3f} | {sum(totals):.3f} | "
            f"{percentile(device_times, 0.50):.3f} | {percentile(device_times, 0.95):.3f} | {sum(device_times):.3f} |"
        )

    return "\n".join(output) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="server log paths, or - for stdin")
    parser.add_argument("--output", type=Path, help="optional Markdown output path")
    args = parser.parse_args()

    report = summarize(args.logs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
