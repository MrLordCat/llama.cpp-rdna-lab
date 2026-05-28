#!/usr/bin/env python3
"""Estimate potential Q4-MetaComp memory savings from GGUF tensor stats.

This script keeps the Q4 payload assumption (4 bits/weight) and estimates
possible savings by compressing metadata overhead in Q4 tensors.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GGUF_PY = ROOT / "gguf-py"
if str(GGUF_PY) not in sys.path:
    sys.path.insert(0, str(GGUF_PY))

from gguf.gguf_reader import GGUFReader  # type: ignore


Q4_TYPES = {
    "Q4_0",
    "Q4_1",
    "Q4_K",
    "Q4_K_M",
    "Q4_K_S",
}


@dataclass
class TensorEstimate:
    name: str
    ttype: str
    n_elements: int
    n_bytes: int
    bpw_now: float
    payload_bpw: float
    meta_bpw: float
    predicted_bpw: float
    predicted_bytes: int
    saved_bytes: int


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument(
        "--meta-save-frac",
        type=float,
        default=0.60,
        help="Fraction of metadata bpw removable in Q4 tensors (0..1)",
    )
    parser.add_argument(
        "--label",
        default=f"q4metacomp-estimate-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Output label",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "build_logs" / "agent-workload"),
        help="Directory for outputs",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many top tensors by potential savings to include",
    )
    return parser.parse_args(argv)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def gib(bytes_count: int) -> float:
    return bytes_count / float(1024**3)


def estimate_tensor(meta_save_frac: float, tensor) -> TensorEstimate:
    n_elements = int(tensor.n_elements)
    n_bytes = int(tensor.n_bytes)
    ttype = tensor.tensor_type.name
    bpw_now = (n_bytes * 8.0 / n_elements) if n_elements else 0.0
    payload_bpw = 4.0 if ttype in Q4_TYPES else bpw_now
    meta_bpw = max(0.0, bpw_now - payload_bpw)
    predicted_bpw = bpw_now
    if ttype in Q4_TYPES:
        predicted_bpw = payload_bpw + meta_bpw * (1.0 - meta_save_frac)
    predicted_bytes = int(math.ceil(predicted_bpw * n_elements / 8.0)) if n_elements else n_bytes
    saved_bytes = max(0, n_bytes - predicted_bytes)
    return TensorEstimate(
        name=str(tensor.name),
        ttype=ttype,
        n_elements=n_elements,
        n_bytes=n_bytes,
        bpw_now=bpw_now,
        payload_bpw=payload_bpw,
        meta_bpw=meta_bpw,
        predicted_bpw=predicted_bpw,
        predicted_bytes=predicted_bytes,
        saved_bytes=saved_bytes,
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    meta_save_frac = clamp01(float(args.meta_save_frac))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = GGUFReader(str(model_path))
    estimates = [estimate_tensor(meta_save_frac, t) for t in reader.tensors]

    total_bytes = sum(t.n_bytes for t in estimates)
    predicted_total_bytes = sum(t.predicted_bytes for t in estimates)
    saved_total_bytes = total_bytes - predicted_total_bytes

    q4_estimates = [t for t in estimates if t.ttype in Q4_TYPES]
    q4_total_bytes = sum(t.n_bytes for t in q4_estimates)
    q4_pred_bytes = sum(t.predicted_bytes for t in q4_estimates)
    q4_saved_bytes = q4_total_bytes - q4_pred_bytes

    sorted_savers = sorted(q4_estimates, key=lambda t: t.saved_bytes, reverse=True)
    top_n = max(1, int(args.top_n))
    top_rows = sorted_savers[:top_n]

    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    summary = {
        "timestamp": ts,
        "label": args.label,
        "model": str(model_path),
        "meta_save_frac": meta_save_frac,
        "tensor_count": len(estimates),
        "q4_tensor_count": len(q4_estimates),
        "total_bytes": total_bytes,
        "predicted_total_bytes": predicted_total_bytes,
        "saved_total_bytes": saved_total_bytes,
        "saved_total_percent": (100.0 * saved_total_bytes / total_bytes) if total_bytes else 0.0,
        "q4_total_bytes": q4_total_bytes,
        "q4_predicted_bytes": q4_pred_bytes,
        "q4_saved_bytes": q4_saved_bytes,
        "q4_saved_percent": (100.0 * q4_saved_bytes / q4_total_bytes) if q4_total_bytes else 0.0,
        "top_saved_tensors": [
            {
                "name": t.name,
                "type": t.ttype,
                "saved_bytes": t.saved_bytes,
                "saved_mib": t.saved_bytes / (1024**2),
                "bpw_now": t.bpw_now,
                "bpw_predicted": t.predicted_bpw,
            }
            for t in top_rows
        ],
    }

    json_path = out_dir / f"{args.label}.q4_metacomp_estimate.json"
    md_path = out_dir / f"{args.label}.q4_metacomp_estimate.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Q4 MetaComp Estimate: {args.label}",
        "",
        f"- timestamp: {ts}",
        f"- model: {model_path.as_posix()}",
        f"- meta_save_frac: {meta_save_frac:.2f}",
        "",
        "## Summary",
        "",
        f"- total model bytes: {total_bytes} ({gib(total_bytes):.3f} GiB)",
        f"- predicted total bytes: {predicted_total_bytes} ({gib(predicted_total_bytes):.3f} GiB)",
        f"- predicted saved total: {saved_total_bytes} ({gib(saved_total_bytes):.3f} GiB, {summary['saved_total_percent']:.2f}%)",
        "",
        f"- Q4-only bytes: {q4_total_bytes} ({gib(q4_total_bytes):.3f} GiB)",
        f"- predicted Q4-only bytes: {q4_pred_bytes} ({gib(q4_pred_bytes):.3f} GiB)",
        f"- predicted Q4-only savings: {q4_saved_bytes} ({gib(q4_saved_bytes):.3f} GiB, {summary['q4_saved_percent']:.2f}%)",
        "",
        "## Top Tensor Savings (Q4)",
        "",
        "| Tensor | Type | Saved MiB | BPW now | BPW predicted |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for t in top_rows:
        lines.append(
            f"| {t.name} | {t.ttype} | {t.saved_bytes / (1024**2):.2f} | {t.bpw_now:.4f} | {t.predicted_bpw:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path.as_posix()}")
    print(f"Wrote {md_path.as_posix()}")
    print(
        "Predicted savings: "
        f"total={saved_total_bytes} bytes ({summary['saved_total_percent']:.2f}%), "
        f"q4={q4_saved_bytes} bytes ({summary['q4_saved_percent']:.2f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
