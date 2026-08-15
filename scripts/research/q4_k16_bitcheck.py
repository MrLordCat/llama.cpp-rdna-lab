"""Битовая сверка C++ порта Q4_K16 против прототипа (quants.py).

Сценарии:
  random:  случайные блоки (seed) с/без imatrix; все три конфига.
  dump:    .npz от prototype/dump_blocks.py (реальный тензор модели).

Сравнивает побайтно: d, dmin (u16 f16), ls, lm (uint8 значения, НЕ битовые
поля), qs. Плюс декант: max-abs ошибка против прототипа (должна быть 0 —
обе стороны декодируют одинаково).

Использование:
  python scripts/research/q4_k16_bitcheck.py random [--blocks 64] [--seed 42]
  python scripts/research/q4_k16_bitcheck.py dump PATH.npz
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "subProject_q4" / "prototype"))

import quants as q  # noqa: E402

HARNESS = ROOT / "scripts" / "research" / "q4_k16_harness.exe"

CFGS = {
    "b77": dict(sc_bits=7, min_bits=7, super_block=512),
    "b76": dict(sc_bits=7, min_bits=6, super_block=512),
    "e55": dict(sc_bits=5, min_bits=5, super_block=512),
}


def run_harness(x: np.ndarray, cfg: str, qw: np.ndarray | None, tmp: Path) -> tuple[np.ndarray, np.ndarray]:
    xf = (tmp / f"x_{cfg}.f32")
    xf.write_bytes(x.astype(np.float32).tobytes())
    args = [str(HARNESS), str(xf), cfg, str(tmp / f"out_{cfg}.bin")]
    if qw is not None:
        qf = tmp / f"qw_{cfg}.f32"
        qf.write_bytes(qw.astype(np.float32).tobytes())
        args.append(str(qf))
    subprocess.run(args, check=True)
    raw = (tmp / f"out_{cfg}.bin").read_bytes()
    # per block: d(2) dmin(2) ls(32) lm(32) qs(256) = 324 bytes
    n = len(raw) // 324
    data = np.frombuffer(raw, dtype=np.uint8).reshape(n, 324)
    d = data[:, 0:2].copy().view(np.uint16).reshape(n)
    dmin = data[:, 2:4].copy().view(np.uint16).reshape(n)
    ls = data[:, 4:36]
    lm = data[:, 36:68]
    qs = data[:, 68:324]
    deq = np.fromfile(tmp / f"out_{cfg}.bin.deq", dtype=np.float32).reshape(n, 512)
    return {"d": d, "dmin": dmin, "scales": ls, "mins": lm, "qs": qs}, deq


def compare(cpp: dict, py: dict, x: np.ndarray, cfg: str, imatrix: bool) -> list[str]:
    errs = []
    tag = f"{cfg}{' (imatrix)' if imatrix else ''}"
    for field in ("d", "dmin", "scales", "mins", "qs"):
        a = np.asarray(cpp[field])
        b = np.asarray(py[field])
        if a.shape != b.shape:
            errs.append(f"{tag}: {field}: shape mismatch {a.shape} vs {b.shape}")
            continue
        if field == "dmin":
            # знак нуля: numpy max(-0.0) = -0.0 (f16 0x8000), C даёт +0.0 — значения равны
            a = np.where(a == 0x8000, np.uint16(0), a)
            b = np.where(b == 0x8000, np.uint16(0), b)
        n_diff = int((a != b).sum())
        if n_diff:
            idx = np.argwhere(a != b)[:5]
            errs.append(f"{tag}: {field}: {n_diff}/{a.size} differ, first at {idx.tolist()}")
    deq_cpp, deq_py = cpp.get("deq"), py.get("deq")
    if deq_cpp is not None and deq_py is not None:
        err = float(np.max(np.abs(deq_cpp - deq_py)))
        if err > 0:
            errs.append(f"{tag}: dequant max-abs diff {err:.6e}")
    return errs


def check_random(blocks: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    fails = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for cfg in CFGS:
            x = rng.standard_normal((blocks, 512)).astype(np.float32)
            x = (x * rng.uniform(0.02, 0.4, (blocks, 1))).astype(np.float32)
            # несколько вырожденных/константных блоков
            x[0] = 0.0
            x[1] = 1.0
            if blocks > 2:
                x[2, ::17] = 0.123
            # без imatrix (exact = последовательные f32-суммы, как C)
            py = q.quantize_q4_k16(x, exact=True, **CFGS[cfg])
            cpp, deq_cpp = run_harness(x, cfg, None, tmp)
            py["deq"] = q.dequantize_q4_k16(py)
            cpp["deq"] = deq_cpp
            fails += compare(cpp, py, x, cfg, False)
            # с imatrix: ПОКОЛОНОЧНЫЕ веса (одна строка, тайлится на все строки),
            # как в llama-quantize (imatrix.gguf хранит n_per_row значений на тензор)
            qw = rng.random((1, 512)).astype(np.float32) + 0.5
            qw_tiled = np.tile(qw, (blocks, 1)).astype(np.float32)
            py = q.quantize_q4_k16(x, quant_weights=qw_tiled, exact=True, **CFGS[cfg])
            cpp, deq_cpp = run_harness(x, cfg, qw_tiled, tmp)
            py["deq"] = q.dequantize_q4_k16(py)
            cpp["deq"] = deq_cpp
            fails += compare(cpp, py, x, cfg, True)
    if fails:
        print("\n".join(fails))
        return 1
    print("OK: все конфиги и оба режима (с/без imatrix) совпали байт-в-байт")
    return 0


def check_dump(npz_path: Path) -> int:
    """Сверка C++ против эталонного дампа dump_blocks.py (f32-exact прототип)."""
    name = npz_path.name
    cfg = None
    for c in CFGS:
        if f"_{c}_" in name or name.startswith(c) or f"ref_{c}" in name:
            cfg = c
            break
    if cfg is None:
        # старые имена вида ref_b77_blk0_ffngate.npz
        for c in CFGS:
            if name.startswith(f"ref_{c}"):
                cfg = c
                break
    if cfg is None:
        print(f"не удалось определить конфиг из имени {name} (ожидаю b77/b76/e55)")
        return 2

    data = np.load(npz_path)
    x = data["x"].astype(np.float32)
    print(f"дамп {name}: конфиг {cfg}, блоков {x.shape[0]}")
    with tempfile.TemporaryDirectory() as td:
        cpp, deq_cpp = run_harness(x, cfg, None, Path(td))
        py = {
            "d": data["d"], "dmin": data["dmin"], "scales": data["scales"],
            "mins": data["mins"], "qs": data["qs"],
        }
        cpp_no_deq = {k: v for k, v in cpp.items() if k != "deq"}
        errs = compare(cpp_no_deq, py, x, cfg, False)
        if errs:
            print("\n".join(errs))
            return 1
        print(f"OK: {cfg} совпал байт-в-байт с эталоном ({x.shape[0]} блоков)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    rnd = sub.add_parser("random")
    rnd.add_argument("--blocks", type=int, default=64)
    rnd.add_argument("--seed", type=int, default=42)
    dmp = sub.add_parser("dump")
    dmp.add_argument("npz")
    args = ap.parse_args()

    if not HARNESS.exists():
        print(f"harness не собран: {HARNESS}")
        print("собери: g++ -std=c++17 -O2 -I ggml/include -I ggml/src -o scripts/research/q4_k16_harness.exe scripts/research/q4_k16_harness.cpp build-cpu/ggml/src/ggml-base.a")
        return 2

    if args.mode == "random":
        return check_random(args.blocks, args.seed)
    return check_dump(Path(args.npz))


if __name__ == "__main__":
    sys.exit(main())
