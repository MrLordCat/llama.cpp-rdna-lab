#!/usr/bin/env python3
"""Run the D095 R6 KV precision scout on the two canonical quick prompts."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from agent_workload_bench import (  # noqa: E402
    TASKS_QUICK,
    apply_real_context_prefix,
    build_repo_snapshot_prefix,
)

PREFIXES = {
    "KV_SCOUT_CAPTURE": "capture",
    "KV_SCOUT_TENSOR": "tensor",
    "KV_SCOUT_LOGIT": "logit",
}

CSV_FIELDS = [
    "record",
    "task",
    "layer",
    "tensor",
    "method",
    "block",
    "bpv",
    "n",
    "samples",
    "mse",
    "mae",
    "cosine_error",
    "max_abs_error",
    "saturation_rate",
    "zero_rate",
    "subnormal_rate",
    "key_stride",
    "d",
    "kv_heads",
    "q_heads",
    "prompt_tokens",
    "k_tokens",
    "v_tokens",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scout-bin",
        default="build-vulkan/bin/llama-kv-precision-scout.exe",
    )
    parser.add_argument("--model", default="models/Qwen3.6-27B-Q4_K_M.gguf")
    parser.add_argument("--label", default="d095-r6-kv-precision")
    parser.add_argument("--task-ids", default="triage_diff,review_bug")
    parser.add_argument("--snapshot-chars", type=int, default=24576)
    parser.add_argument("--ctx-size", type=int, default=49152)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--max-layer", type=int, default=11)
    parser.add_argument("--key-stride", type=int, default=4)
    parser.add_argument("--gate-method", default="bs_e4m3")
    parser.add_argument("--gate-block", type=int, default=32)
    parser.add_argument("--gate-reduction", type=float, default=0.25)
    parser.add_argument("--gate-max-metadata", type=float, default=0.04)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--out-dir", default="build_logs/agent-workload")
    return parser.parse_args()


def ensure_no_gpu_server() -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        process_text = result.stdout.lower()
    else:
        result = subprocess.run(
            ["ps", "-eo", "comm,args"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        process_text = result.stdout.lower()
    if "llama-server" in process_text or "agent_workload_bench" in process_text:
        raise RuntimeError("refusing to start while llama-server/benchmark is active")


def parse_records(output: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line in output.splitlines():
        prefix = line.split(maxsplit=1)[0] if line else ""
        if prefix not in PREFIXES:
            continue
        record: dict[str, str] = {"record": PREFIXES[prefix]}
        for token in line.split()[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            record[key] = value
        records.append(record)
    return records


def method_key(record: dict[str, str]) -> tuple[str, int]:
    return record.get("method", ""), int(record.get("block", "0"))


def weighted_logit_mse(records: list[dict[str, str]]) -> dict[tuple[str, int], float]:
    totals: dict[tuple[str, int], tuple[float, int]] = {}
    for record in records:
        if record.get("record") != "logit":
            continue
        key = method_key(record)
        samples = int(record.get("samples", "0"))
        mse = float(record.get("mse", "nan"))
        current_sum, current_n = totals.get(key, (0.0, 0))
        totals[key] = current_sum + mse*samples, current_n + samples
    return {
        key: total/count
        for key, (total, count) in totals.items()
        if count > 0
    }


def fmt(value: str | float | None, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}g}"


def write_markdown(
    path: Path,
    args: argparse.Namespace,
    records: list[dict[str, str]],
    snapshot_chars: int,
    snapshot_files: int,
) -> None:
    captures = [record for record in records if record.get("record") == "capture"]
    logits = [record for record in records if record.get("record") == "logit"]
    tensors = [record for record in records if record.get("record") == "tensor"]
    weighted = weighted_logit_mse(records)
    raw = weighted.get(("raw_e4m3", 0))
    gate_key = (args.gate_method, args.gate_block)
    candidate = weighted.get(gate_key)
    reduction = None
    if raw is not None and raw > 0.0 and candidate is not None:
        reduction = 1.0 - candidate/raw

    pair_ratios: list[float] = []
    raw_by_pair = {
        (r.get("task"), r.get("layer")): float(r["mse"])
        for r in logits
        if method_key(r) == ("raw_e4m3", 0)
    }
    for record in logits:
        if method_key(record) != gate_key:
            continue
        raw_value = raw_by_pair.get((record.get("task"), record.get("layer")))
        if raw_value is not None and raw_value > 0.0:
            pair_ratios.append(float(record["mse"])/raw_value)

    candidate_record = next((record for record in logits if method_key(record) == gate_key), None)
    candidate_bpv = float(candidate_record["bpv"]) if candidate_record else float("nan")
    metadata_fraction = max(0.0, candidate_bpv - 1.0)
    gate_pass = (
        reduction is not None
        and reduction >= args.gate_reduction
        and metadata_fraction <= args.gate_max_metadata
        and pair_ratios
        and max(pair_ratios) <= 1.0
    )

    lines = [
        "# D095 R6 KV precision scout",
        "",
        f"- Label: `{args.label}`",
        f"- Snapshot: {snapshot_chars} chars from {snapshot_files} files",
        f"- Capture: full-attention layers through `{args.max_layer}`, key stride `{args.key_stride}`",
        f"- Gate candidate: `{args.gate_method}` block `{args.gate_block}`",
        f"- Candidate metadata: `{metadata_fraction:.3%}` of payload",
        "",
        "## Capture integrity",
        "",
        "| Task | Layer | D | KV heads | Q heads | Prompt | K | V |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in captures:
        lines.append(
            f"| {record.get('task')} | {record.get('layer')} | {record.get('d')} | "
            f"{record.get('kv_heads')} | {record.get('q_heads')} | {record.get('prompt_tokens')} | "
            f"{record.get('k_tokens')} | {record.get('v_tokens')} |"
        )

    lines.extend([
        "",
        "## Attention-logit error",
        "",
        "| Task | Layer | Method | Block | B/value | MSE | Cosine error | Max abs | Samples |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for record in logits:
        lines.append(
            f"| {record.get('task')} | {record.get('layer')} | {record.get('method')} | "
            f"{record.get('block')} | {fmt(record.get('bpv'), 6)} | {fmt(record.get('mse'))} | "
            f"{fmt(record.get('cosine_error'))} | {fmt(record.get('max_abs_error'))} | "
            f"{record.get('samples')} |"
        )

    lines.extend([
        "",
        "## Tensor reconstruction MSE",
        "",
        "| Task | Layer | Tensor | Method | Block | MSE | Zero rate | Saturation |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ])
    for record in tensors:
        lines.append(
            f"| {record.get('task')} | {record.get('layer')} | {record.get('tensor')} | "
            f"{record.get('method')} | {record.get('block')} | {fmt(record.get('mse'))} | "
            f"{fmt(record.get('zero_rate'))} | {fmt(record.get('saturation_rate'))} |"
        )

    lines.extend([
        "",
        "## Prebuild gate",
        "",
        f"- Weighted raw-E4M3 logit MSE: `{fmt(raw)}`",
        f"- Weighted candidate logit MSE: `{fmt(candidate)}`",
        f"- Candidate reduction: `{reduction:.2%}`" if reduction is not None else "- Candidate reduction: unavailable",
        f"- Worst per-task/layer candidate/raw ratio: `{max(pair_ratios):.4f}`" if pair_ratios else "- Pairwise ratio: unavailable",
        f"- Screening verdict: **{'PASS' if gate_pass else 'FAIL'}**",
        "",
        f"PASS means >={args.gate_reduction:.0%} weighted logit-MSE reduction, "
        f"<={args.gate_max_metadata:.1%} scale metadata, and no captured task/layer regression. "
        "It authorizes only a default-off format prototype, not a speed or quality claim.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    scout_bin = (ROOT / args.scout_bin).resolve()
    model = (ROOT / args.model).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    if not scout_bin.exists():
        raise FileNotFoundError(f"scout binary not found: {scout_bin}")
    if not model.exists():
        raise FileNotFoundError(f"model not found: {model}")

    ensure_no_gpu_server()
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {item.strip() for item in args.task_ids.split(",") if item.strip()}
    tasks = [dict(task) for task in TASKS_QUICK if task["id"] in wanted]
    if {task["id"] for task in tasks} != wanted:
        missing = sorted(wanted - {task["id"] for task in tasks})
        raise ValueError(f"unknown quick task ids: {missing}")

    prefix, snapshot_chars, snapshot_files = build_repo_snapshot_prefix(ROOT, args.snapshot_chars)
    tasks = apply_real_context_prefix(tasks, prefix)
    all_records: list[dict[str, str]] = []

    csv_path = out_dir / f"{args.label}.csv"
    markdown_path = out_dir / f"{args.label}.md"
    if args.report_only:
        if not csv_path.exists():
            raise FileNotFoundError(f"report-only CSV not found: {csv_path}")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            all_records = list(csv.DictReader(handle))
        write_markdown(markdown_path, args, all_records, snapshot_chars, snapshot_files)
        print(f"Wrote {markdown_path} from {csv_path}")
        return 0

    for task in tasks:
        prompt_path = out_dir / f"{args.label}-{task['id']}.prompt.txt"
        log_path = out_dir / f"{args.label}-{task['id']}.log"
        prompt_path.write_text(task["prompt"], encoding="utf-8")
        env = os.environ.copy()
        env["KV_SCOUT_LABEL"] = task["id"]
        env["KV_SCOUT_MAX_LAYER"] = str(args.max_layer)
        env["KV_SCOUT_KEY_STRIDE"] = str(args.key_stride)
        env["GGML_VK_FA_F8_P5"] = "1"
        env["LLAMA_VK_MTP_KV_LAST_F16"] = "0"
        command = [
            str(scout_bin),
            "-m", str(model),
            "-f", str(prompt_path),
            "-c", str(args.ctx_size),
            "-b", str(args.batch_size),
            "-ub", str(args.ubatch_size),
            "-ngl", "999",
            "--flash-attn", "on",
            "--cache-type-k", "f8_e4m3",
            "--cache-type-v", "f8_e4m3",
            "-dev", "Vulkan1,Vulkan0",
            "-sm", "layer",
            "-ts", "1,1",
            "--no-mmap",
            "-fit", "off",
        ]
        print(f"[{task['id']}] capturing Q/K/V from {len(task['prompt'])} prompt chars")
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        combined = result.stdout + "\n--- stderr ---\n" + result.stderr
        log_path.write_text(
            "COMMAND: " + subprocess.list2cmdline(command) + "\n\n" + combined,
            encoding="utf-8",
        )
        records = parse_records(result.stdout)
        if result.returncode != 0:
            tail = "\n".join(combined.splitlines()[-40:])
            raise RuntimeError(f"scout failed for {task['id']} (see {log_path}):\n{tail}")
        if not records:
            raise RuntimeError(f"scout produced no records for {task['id']} (see {log_path})")
        all_records.extend(records)
        print(f"[{task['id']}] captured {len(records)} metric rows")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in all_records:
            writer.writerow(record)

    write_markdown(markdown_path, args, all_records, snapshot_chars, snapshot_files)
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())