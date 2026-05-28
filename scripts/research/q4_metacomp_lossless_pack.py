#!/usr/bin/env python3
"""Build and verify a lossless packed payload for selected Q4 tensors.

Format goals:
- lossless by construction (exact byte restoration),
- fail-closed metadata (hashes, sizes, offsets),
- optional reversible preprocess before compression.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GGUF_PY = ROOT / "gguf-py"
if str(GGUF_PY) not in sys.path:
    sys.path.insert(0, str(GGUF_PY))

from gguf.gguf_reader import GGUFReader  # type: ignore

Q4_TYPE_IDS = {2, 3, 12}
Q4_TYPE_NAMES = {
    2: "Q4_0",
    3: "Q4_1",
    12: "Q4_K",
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Path to GGUF model")
    p.add_argument("--runtime-sidecar-json", required=True, help="Path to q4 metacomp runtime sidecar JSON")
    p.add_argument("--label", required=True, help="Output artifact label")
    p.add_argument(
        "--max-selected-tensors",
        type=int,
        default=0,
        help="Optional cap for selected tensor count (0 = all)",
    )
    p.add_argument(
        "--out-dir",
        default=str(ROOT / "build_logs" / "agent-workload"),
        help="Output directory",
    )
    p.add_argument(
        "--zlib-level",
        type=int,
        default=9,
        help="zlib compression level (0..9)",
    )
    p.add_argument(
        "--preprocess",
        choices=["none", "nibble_split", "bitplane8"],
        default="none",
        help="Optional reversible pre-process before compression",
    )
    p.add_argument(
        "--target-free-gib",
        type=float,
        default=0.0,
        help="Optional free-space target in GiB for current packed scope",
    )
    p.add_argument(
        "--fail-if-below-target",
        action="store_true",
        help="Exit with code 2 if target-free-gib is set and not reached",
    )
    p.add_argument(
        "--live-log",
        action="store_true",
        help="Print live per-tensor progress during packing",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Emit live log every N packed tensors (default: 1)",
    )
    return p.parse_args(argv)


def _fmt_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    idx = 0
    while x >= 1024.0 and idx < len(units) - 1:
        x /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(x)} {units[idx]}"
    return f"{x:.2f} {units[idx]}"


def _tensor_raw_bytes(tensor: object) -> bytes:
    data = getattr(tensor, "data", None)
    if data is None:
        raise RuntimeError("tensor has no data field")
    if hasattr(data, "tobytes"):
        return data.tobytes()
    return np.asarray(data, dtype=np.uint8).tobytes()


def _load_selected_names(sidecar_path: Path, max_selected_tensors: int) -> list[str]:
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    selected = payload.get("selected", [])
    names = [str(row["name"]) for row in selected if isinstance(row, dict) and "name" in row]
    if max_selected_tensors > 0:
        names = names[:max_selected_tensors]
    return names


def _pack_nibbles(values: np.ndarray) -> bytes:
    if values.size % 2 != 0:
        values = np.concatenate([values, np.zeros(1, dtype=np.uint8)])
    lo = values[0::2] & 0x0F
    hi = values[1::2] & 0x0F
    return (lo | (hi << 4)).astype(np.uint8).tobytes()


def _unpack_nibbles(buf: bytes, n_values: int) -> np.ndarray:
    x = np.frombuffer(buf, dtype=np.uint8)
    out = np.empty(x.size * 2, dtype=np.uint8)
    out[0::2] = x & 0x0F
    out[1::2] = (x >> 4) & 0x0F
    return out[:n_values]


def _preprocess_encode(raw: bytes, mode: str) -> bytes:
    if mode == "none":
        return raw

    x = np.frombuffer(raw, dtype=np.uint8)

    if mode == "nibble_split":
        lows = x & 0x0F
        highs = (x >> 4) & 0x0F
        return _pack_nibbles(lows) + _pack_nibbles(highs)

    if mode == "bitplane8":
        bits = np.unpackbits(x, bitorder="little").reshape(-1, 8)
        planes = [np.packbits(bits[:, i], bitorder="little") for i in range(8)]
        return b"".join(p.tobytes() for p in planes)

    raise ValueError(f"Unsupported preprocess mode: {mode}")


def _preprocess_decode(encoded: bytes, mode: str, raw_size: int) -> bytes:
    if mode == "none":
        return encoded

    if mode == "nibble_split":
        n = raw_size
        half = (n + 1) // 2
        lows_packed = encoded[:half]
        highs_packed = encoded[half : half * 2]
        lows = _unpack_nibbles(lows_packed, n)
        highs = _unpack_nibbles(highs_packed, n)
        out = (lows | (highs << 4)).astype(np.uint8)
        return out.tobytes()

    if mode == "bitplane8":
        n = raw_size
        plane_bytes = (n + 7) // 8
        required = 8 * plane_bytes
        if len(encoded) < required:
            raise RuntimeError("bitplane8 decode: encoded payload is too short")
        planes = []
        for i in range(8):
            part = encoded[i * plane_bytes : (i + 1) * plane_bytes]
            bits = np.unpackbits(np.frombuffer(part, dtype=np.uint8), bitorder="little")[:n]
            planes.append(bits)
        bits2d = np.stack(planes, axis=1)
        out = np.packbits(bits2d.reshape(-1), bitorder="little")[:n]
        return out.tobytes()

    raise ValueError(f"Unsupported preprocess mode: {mode}")


def _verify_blob(blob: bytes, entries: list[dict], preprocess: str) -> None:
    for row in entries:
        off = int(row["blob_offset"])
        sz = int(row["packed_size"])
        part = blob[off : off + sz]

        pre = zlib.decompress(part)
        raw = _preprocess_decode(pre, preprocess, int(row["raw_size"]))

        if len(raw) != int(row["raw_size"]):
            raise RuntimeError(f"verify size mismatch for {row['name']}")

        crc = zlib.crc32(raw) & 0xFFFFFFFF
        if crc != int(row["raw_crc32"]):
            raise RuntimeError(f"verify crc mismatch for {row['name']}")

        sha = hashlib.sha256(raw).hexdigest()
        if sha != str(row["raw_sha256"]):
            raise RuntimeError(f"verify sha mismatch for {row['name']}")


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    model_path = Path(args.model)
    sidecar_path = Path(args.runtime_sidecar_json)
    out_dir = Path(args.out_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Sidecar JSON not found: {sidecar_path}")
    if args.zlib_level < 0 or args.zlib_level > 9:
        raise ValueError("zlib-level must be in range 0..9")

    selected_names = _load_selected_names(sidecar_path, args.max_selected_tensors)
    if not selected_names:
        raise RuntimeError("No selected tensor names in sidecar")

    reader = GGUFReader(str(model_path))
    tensor_map = {str(t.name): t for t in reader.tensors}

    entries: list[dict] = []
    blob = bytearray()

    total_raw = 0
    total_packed = 0
    skipped_non_q4 = 0
    skipped_missing = 0

    total_requested = len(selected_names)
    for idx_name, name in enumerate(selected_names, start=1):
        tensor = tensor_map.get(name)
        if tensor is None:
            skipped_missing += 1
            if args.live_log:
                print(
                    f"[lossless-pack] {idx_name}/{total_requested} skip missing: {name}",
                    flush=True,
                )
            continue

        ttype = int(getattr(tensor, "tensor_type"))
        if ttype not in Q4_TYPE_IDS:
            skipped_non_q4 += 1
            if args.live_log:
                print(
                    f"[lossless-pack] {idx_name}/{total_requested} skip non-q4: {name} type={ttype}",
                    flush=True,
                )
            continue

        raw = _tensor_raw_bytes(tensor)
        pre = _preprocess_encode(raw, args.preprocess)
        packed = zlib.compress(pre, level=args.zlib_level)

        off = len(blob)
        blob.extend(packed)

        entries.append(
            {
                "name": name,
                "tensor_type_id": ttype,
                "tensor_type_name": Q4_TYPE_NAMES.get(ttype, f"Q4_TYPE_{ttype}"),
                "n_elements": int(getattr(tensor, "n_elements")),
                "raw_size": len(raw),
                "packed_size": len(packed),
                "blob_offset": off,
                "raw_crc32": int(zlib.crc32(raw) & 0xFFFFFFFF),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )

        total_raw += len(raw)
        total_packed += len(packed)

        if args.live_log and (len(entries) % max(1, args.log_every) == 0):
            ratio_now = float(total_packed) / float(total_raw) if total_raw > 0 else 0.0
            saved_now = total_raw - total_packed
            print(
                "[lossless-pack] "
                f"{idx_name}/{total_requested} packed={len(entries)} "
                f"raw={_fmt_bytes(total_raw)} packed_bytes={_fmt_bytes(total_packed)} "
                f"saved={_fmt_bytes(saved_now)} ratio={ratio_now:.6f} current={name}",
                flush=True,
            )

    if not entries:
        raise RuntimeError("No Q4 entries were packed (all missing/non-q4)")

    _verify_blob(bytes(blob), entries, args.preprocess)

    out_dir.mkdir(parents=True, exist_ok=True)
    blob_path = out_dir / f"{args.label}.q4_metacomp_lossless_pack.bin"
    json_path = out_dir / f"{args.label}.q4_metacomp_lossless_pack.json"
    md_path = out_dir / f"{args.label}.q4_metacomp_lossless_pack.md"

    blob_path.write_bytes(bytes(blob))

    ratio = float(total_packed) / float(total_raw) if total_raw > 0 else 0.0
    saved_bytes = total_raw - total_packed
    saved_mib = saved_bytes / (1024.0 * 1024.0)

    target_bytes = int(args.target_free_gib * (1024.0 ** 3))
    target_ok = True if target_bytes <= 0 else (saved_bytes >= target_bytes)

    payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "label": args.label,
        "mode": "q4_metacomp_lossless_pack_v1",
        "model": str(model_path),
        "runtime_sidecar_json": str(sidecar_path),
        "codec": {
            "name": "zlib",
            "level": int(args.zlib_level),
            "preprocess": args.preprocess,
        },
        "contracts": {
            "lossless": True,
            "verify": "packed->decompress->inverse_preprocess exact raw bytes by crc32+sha256",
            "fallback": "missing/non-q4 entries are skipped; runtime must fail-closed per-entry",
        },
        "summary": {
            "selected_requested": len(selected_names),
            "entries_packed": len(entries),
            "skipped_missing": skipped_missing,
            "skipped_non_q4": skipped_non_q4,
            "total_raw_bytes": total_raw,
            "total_packed_bytes": total_packed,
            "packed_to_raw_ratio": ratio,
            "bytes_saved": saved_bytes,
            "target_free_bytes": target_bytes,
            "target_reached": target_ok,
        },
        "blob_file": os.path.basename(blob_path),
        "entries": entries,
    }

    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Q4 MetaComp Lossless Pack: {args.label}",
        "",
        f"- model: {model_path.as_posix()}",
        f"- sidecar: {sidecar_path.as_posix()}",
        f"- codec: zlib level {args.zlib_level}",
        f"- preprocess: {args.preprocess}",
        f"- selected requested: {len(selected_names)}",
        f"- packed entries: {len(entries)}",
        f"- skipped missing: {skipped_missing}",
        f"- skipped non-q4: {skipped_non_q4}",
        f"- total raw bytes: {total_raw}",
        f"- total packed bytes: {total_packed}",
        f"- packed/raw ratio: {ratio:.6f}",
        f"- bytes saved: {saved_bytes} ({saved_mib:.2f} MiB)",
    ]
    if target_bytes > 0:
        lines.extend(
            [
                f"- target free bytes: {target_bytes} ({args.target_free_gib:.2f} GiB)",
                f"- target reached: {'yes' if target_ok else 'no'}",
            ]
        )
    lines.extend(
        [
            "",
            "Lossless contract: verified by per-entry crc32 + sha256 after decompress + inverse preprocess.",
        ]
    )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {blob_path.as_posix()}")
    print(f"Wrote {json_path.as_posix()}")
    print(f"Wrote {md_path.as_posix()}")
    print(
        "Lossless pack summary: "
        f"entries={len(entries)}, ratio={ratio:.6f}, saved_bytes={saved_bytes}, preprocess={args.preprocess}"
    )
    if target_bytes > 0:
        print(f"Target check: free_target_bytes={target_bytes}, reached={'yes' if target_ok else 'no'}")

    if args.fail_if_below_target and target_bytes > 0 and not target_ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
