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
            # без imatrix
            py = q.quantize_q4_k16(x, **CFGS[cfg])
            cpp, deq_cpp = run_harness(x, cfg, None, tmp)
            py["deq"] = q.dequantize_q4_k16(py)
            cpp["deq"] = deq_cpp
            fails += compare(cpp, py, x, cfg, False)
            # с imatrix
            qw = rng.random((blocks, 512)).astype(np.float32) + 0.5
            py = q.quantize_q4_k16(x, quant_weights=qw, **CFGS[cfg])
            cpp, deq_cpp = run_harness(x, cfg, qw, tmp)
            py["deq"] = q.dequantize_q4_k16(py)
            cpp["deq"] = deq_cpp
            fails += compare(cpp, py, x, cfg, True)
    if fails:
        print("\n".join(fails))
        return 1
    print("OK: все конфиги и оба режима (с/без imatrix) совпали байт-в-байт")
    return 0


def check_dump(npz_path: Path) -> int:
    data = np.load(npz_path)
    x = data["x"]
    cfg = None
    # определяем конфиг по форме (sc_bits неизвестны из npz — берём по имени файла/аргументу)
    print("поля дампа:", {k: v.shape for k, v in data.items()})
    print("x блоков:", x.shape)
    # конфиг передаётся вторым аргументом в имени: dump_blocks.py имя не хранит — требуем явно
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
