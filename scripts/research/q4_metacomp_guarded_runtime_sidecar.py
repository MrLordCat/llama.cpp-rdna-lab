#!/usr/bin/env python3
"""Build a guarded runtime sidecar for H54-B value-aware Q4 prototype.

This script is fail-closed by design:
- only tensors marked as selected in guarded manifest are considered,
- every tensor that fails validation/dequant/codebook build is marked fallback,
- runtime defaults are not changed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GGUF_PY = ROOT / "gguf-py"
if str(GGUF_PY) not in sys.path:
    sys.path.insert(0, str(GGUF_PY))

from gguf.gguf_reader import GGUFReader  # type: ignore

from q4_c2_value_aware_gate import (  # type: ignore
    BLOCK_SIZES,
    ELEMENTS_PER_BLOCK,
    Q4_TYPE_IDS,
    Q4_TYPE_NAMES,
    compute_entropy,
    dequant_q4_tensor,
    lloyd_max_1d,
    unpack_nibbles,
    quantize_with_codebook,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Path to Q4 GGUF model")
    p.add_argument("--guarded-prototype-json", required=True, help="Path to q4_metacomp_guarded_prototype.json")
    p.add_argument(
        "--max-elements-per-tensor",
        type=int,
        default=131072,
        help="Element cap per tensor for sidecar codebook building",
    )
    p.add_argument(
        "--max-selected-tensors",
        type=int,
        default=0,
        help="Optional cap on number of selected tensors (0 means all)",
    )
    p.add_argument(
        "--nlevels",
        type=int,
        default=16,
        help="Codebook levels (must stay 16 for Q4 guarded prototype)",
    )
    p.add_argument(
        "--label",
        default=f"q4metacomp-guarded-runtime-sidecar-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Output label",
    )
    p.add_argument(
        "--out-dir",
        default=str(ROOT / "build_logs" / "agent-workload"),
        help="Output directory",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    model_path = Path(args.model)
    guarded_path = Path(args.guarded_prototype_json)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not guarded_path.exists():
        raise FileNotFoundError(f"Guarded prototype JSON not found: {guarded_path}")

    if args.nlevels != 16:
        raise ValueError("Only nlevels=16 is supported for Q4 guarded runtime sidecar")

    guarded = json.loads(guarded_path.read_text(encoding="utf-8"))
    decision_rows = guarded.get("tensor_decisions", [])
    selected_names = [r["name"] for r in decision_rows if bool(r.get("selected", False))]
    fallback_names = [r["name"] for r in decision_rows if not bool(r.get("selected", False))]

    if args.max_selected_tensors > 0:
        selected_names = selected_names[: args.max_selected_tensors]

    reader = GGUFReader(str(model_path))
    tensor_map = {str(t.name): t for t in reader.tensors}

    selected_rows: list[dict] = []
    fallback_rows: list[dict] = []

    total_selected_elements = 0
    total_selected_entropy_orig = 0.0
    total_selected_entropy_new = 0.0

    for name in selected_names:
        t = tensor_map.get(name)
        if t is None:
            fallback_rows.append({"name": name, "reason": "missing_in_model"})
            continue

        ttype = int(t.tensor_type)
        if ttype not in Q4_TYPE_IDS:
            fallback_rows.append({"name": name, "reason": "not_q4_tensor"})
            continue

        raw_data = t.data
        if hasattr(raw_data, "flatten"):
            flat = raw_data.flatten()
        elif hasattr(raw_data, "tobytes"):
            flat = np.frombuffer(raw_data.tobytes(), dtype=np.uint8)
        else:
            flat = np.asarray(raw_data, dtype=np.uint8).flatten()

        try:
            fp32_values, _ = dequant_q4_tensor(
                flat,
                ttype,
                int(t.n_elements),
                max_elements=args.max_elements_per_tensor,
            )
        except Exception as exc:  # fail-closed
            fallback_rows.append({"name": name, "reason": f"dequant_failed:{type(exc).__name__}"})
            continue

        if fp32_values.size == 0:
            fallback_rows.append({"name": name, "reason": "empty_sample"})
            continue

        codebook, boundaries = lloyd_max_1d(fp32_values, nlevels=args.nlevels)
        new_indices = quantize_with_codebook(fp32_values, codebook)

        # Baseline entropy for the same sampled scope using original Q4 nibble indices.
        block_size = BLOCK_SIZES[ttype]
        elements_per_block = ELEMENTS_PER_BLOCK[ttype]
        nblocks = max(1, fp32_values.size // elements_per_block)

        all_orig_indices: list[int] = []
        for b in range(nblocks):
            block_data = flat[b * block_size : (b + 1) * block_size]
            if ttype == 2:  # Q4_0
                qs = block_data[2:18]
            elif ttype == 3:  # Q4_1
                qs = block_data[4:20]
            else:  # Q4_K family
                qs = block_data[16:144]

            nibbles = unpack_nibbles(np.asarray(qs, dtype=np.uint8))
            all_orig_indices.extend(nibbles[:elements_per_block].tolist())

        orig_indices = np.asarray(all_orig_indices[: fp32_values.size], dtype=np.int32)

        entropy_orig = float(compute_entropy(orig_indices, 16))
        entropy_new = float(compute_entropy(new_indices, 16))

        total_selected_elements += int(fp32_values.size)
        total_selected_entropy_orig += entropy_orig * fp32_values.size
        total_selected_entropy_new += entropy_new * fp32_values.size

        selected_rows.append(
            {
                "name": name,
                "type": Q4_TYPE_NAMES.get(ttype, f"Q4_TYPE_{ttype}"),
                "elements_total": int(t.n_elements),
                "elements_sampled": int(fp32_values.size),
                "entropy_original_bpw_sample": entropy_orig,
                "entropy_new_bpw_sample": entropy_new,
                "entropy_delta_bpw_sample": entropy_new - entropy_orig,
                "codebook_fp32": [float(x) for x in codebook.tolist()],
                "boundaries_fp32": [float(x) for x in boundaries.tolist()],
                "status": "selected",
            }
        )

    # Preserve explicit non-selected manifest rows as deterministic fallback entries.
    for name in fallback_names:
        fallback_rows.append({"name": name, "reason": "manifest_not_selected"})

    entropy_orig_weighted = (
        total_selected_entropy_orig / total_selected_elements if total_selected_elements > 0 else 0.0
    )
    entropy_new_weighted = (
        total_selected_entropy_new / total_selected_elements if total_selected_elements > 0 else 0.0
    )

    payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "label": args.label,
        "model": str(model_path),
        "guarded_prototype_json": str(guarded_path),
        "mode": "guarded_runtime_sidecar",
        "fail_closed": True,
        "contracts": {
            "nlevels": args.nlevels,
            "max_elements_per_tensor": int(args.max_elements_per_tensor),
            "max_selected_tensors": int(args.max_selected_tensors),
        },
        "summary": {
            "selected_rows": len(selected_rows),
            "fallback_rows": len(fallback_rows),
            "selected_elements_sampled": int(total_selected_elements),
            "selected_entropy_original_bpw_sample": float(entropy_orig_weighted),
            "selected_entropy_new_bpw_sample": float(entropy_new_weighted),
            "selected_entropy_delta_bpw_sample": float(entropy_new_weighted - entropy_orig_weighted),
        },
        "runtime_gate": {
            "enabled_env": "LLAMA_Q4_METACOMP_ENABLE=1",
            "sidecar_env": "LLAMA_Q4_METACOMP_SIDECAR=<path>",
            "rollback": "unset LLAMA_Q4_METACOMP_ENABLE LLAMA_Q4_METACOMP_SIDECAR",
            "note": "Runtime integration is opt-in only; missing/invalid rows must fallback to legacy Q4 path",
        },
        "selected": selected_rows,
        "fallback": fallback_rows,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.label}.q4_metacomp_guarded_runtime_sidecar.json"
    md_path = out_dir / f"{args.label}.q4_metacomp_guarded_runtime_sidecar.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Q4 MetaComp Guarded Runtime Sidecar: {args.label}",
        "",
        f"- model: {model_path.as_posix()}",
        f"- guarded_prototype_json: {guarded_path.as_posix()}",
        f"- selected rows: {len(selected_rows)}",
        f"- fallback rows: {len(fallback_rows)}",
        f"- sampled selected elements: {total_selected_elements}",
        f"- selected entropy (orig sample): {entropy_orig_weighted:.6f} bpw",
        f"- selected entropy (new sample): {entropy_new_weighted:.6f} bpw",
        f"- selected entropy delta (sample): {entropy_new_weighted - entropy_orig_weighted:+.6f} bpw",
        "",
        "## Runtime Gate",
        "",
        "- enable: `LLAMA_Q4_METACOMP_ENABLE=1`",
        "- sidecar: `LLAMA_Q4_METACOMP_SIDECAR=<path-to-json>`",
        "- rollback: `unset LLAMA_Q4_METACOMP_ENABLE LLAMA_Q4_METACOMP_SIDECAR`",
        "",
        "Fail-closed policy: invalid/missing/non-selected tensor rows must use legacy Q4 path.",
    ]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path.as_posix()}")
    print(f"Wrote {md_path.as_posix()}")
    print(
        "Guarded runtime sidecar summary: "
        f"selected={len(selected_rows)}, fallback={len(fallback_rows)}, "
        f"entropy_delta={entropy_new_weighted - entropy_orig_weighted:+.6f} bpw"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
