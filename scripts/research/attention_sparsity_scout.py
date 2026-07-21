#!/usr/bin/env python3
"""Attention sparsity scout driver (P003 T5a gate).

Runs the CPU-only `llama-attn-sparsity-scout` tool on a real prompt and reports,
per attention layer and globally, what fraction of the valid K/V positions
carries 75/90/95/99% of the post-softmax attention mass.

Gate: a sparse-FlashAttention prototype is only worth building if >75% of the
mass sits in <25% of the K/V blocks, i.e. the global frac75 mean is below 0.25.

This drives the CPU build so it never touches the GPU discovery/driver path.
Flash Attention is forced off inside the tool so the softmax node is exposed.

Example:
    python scripts/research/attention_sparsity_scout.py \
        --model models/Qwen3.6-27B-Q3_K_S.gguf \
        --prompt-file some_long_prompt.txt \
        --ctx 8192 --batch 2048 --ubatch 2048 --query-stride 8
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def default_bin() -> Path:
    name = "llama-attn-sparsity-scout"
    if os.name == "nt":
        name += ".exe"
    return REPO_ROOT / "build-cpu" / "bin" / name


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attention sparsity scout (P003 T5a gate)")
    p.add_argument("--bin", type=Path, default=default_bin(),
                   help="path to llama-attn-sparsity-scout (default: build-cpu bin)")
    p.add_argument("--model", "-m", type=Path, required=True, help="GGUF model path")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt-file", "-f", type=Path, help="prompt text file")
    src.add_argument("--prompt", "-p", type=str, help="inline prompt text")
    p.add_argument("--max-chars", type=int, default=0,
                   help="truncate the prompt file to the first N characters (0 = full)")
    p.add_argument("--ctx", "-c", type=int, default=8192)
    # batch must be >= prompt token count (the whole prompt is one logical batch);
    # ubatch bounds the per-node attention matrix size.
    p.add_argument("--batch", "-b", type=int, default=8192)
    p.add_argument("--ubatch", "-ub", type=int, default=1024)
    p.add_argument("--threads", "-t", type=int, default=0, help="0 = tool default")
    p.add_argument("--query-stride", type=int, default=8,
                   help="sample every Nth query position per layer (cost control)")
    p.add_argument("--block-size", type=int, default=32,
                   help="K/V block size for the block-max selector recovery test")
    p.add_argument("--tile", type=int, default=0,
                   help="query-tile size for gather viability test (0 = disabled, e.g. 32 for coopmat)")
    p.add_argument("--csv", type=Path, default=None, help="optional CSV output path")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="extra args passed verbatim to the scout tool")
    return p.parse_args(argv)


def build_command(args: argparse.Namespace, prompt_path: Path | None) -> list[str]:
    cmd: list[str] = [
        str(args.bin),
        "-m", str(args.model),
        "-c", str(args.ctx),
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "-fa", "off",
    ]
    if args.threads > 0:
        cmd += ["-t", str(args.threads)]
    if prompt_path is not None:
        cmd += ["-f", str(prompt_path)]
    else:
        cmd += ["-p", args.prompt]
    cmd += list(args.extra)
    return cmd


def parse_rows(stdout: str) -> tuple[list[dict], dict | None, dict, dict | None]:
    layers: list[dict] = []
    glob: dict | None = None
    gather: dict | None = None
    header: dict = {}
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("ATTN_SPARSITY_HEADER"):
            header = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
        elif line.startswith("ATTN_GATHER_GLOBAL"):
            gather = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
        elif line.startswith("ATTN_SPARSITY_GLOBAL"):
            glob = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
        elif line.startswith("ATTN_SPARSITY "):
            layers.append(dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv))
    return layers, glob, header, gather


def write_csv(path: Path, layers: list[dict]) -> None:
    cols = ["layer", "rows", "valid_mean", "frac75", "frac90", "frac95", "frac99",
            "bm06", "bm12", "bm25", "bm50"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for row in layers:
            fh.write(",".join(str(row.get(c, "")) for c in cols) + "\n")


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if not args.bin.exists():
        print(f"error: scout binary not found: {args.bin}\n"
              f"build it first: cmake --build build-cpu -j 4 --target llama-attn-sparsity-scout",
              file=sys.stderr)
        return 2
    if not args.model.exists():
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 2

    prompt_path: Path | None = args.prompt_file
    tmp_path: Path | None = None
    if prompt_path is not None and args.max_chars > 0:
        text = prompt_path.read_text(encoding="utf-8", errors="replace")[: args.max_chars]
        tmp_path = REPO_ROOT / "build_logs" / "agent-workload" / "attn-scout-prompt.txt"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(text, encoding="utf-8")
        prompt_path = tmp_path

    env = dict(os.environ)
    env["ATTN_SCOUT_QUERY_STRIDE"] = str(max(1, args.query_stride))
    env["ATTN_SCOUT_BLOCK_SIZE"] = str(max(1, args.block_size))
    if args.tile > 0:
        env["ATTN_SCOUT_TILE"] = str(max(0, args.tile))
    # The CPU build links the MinGW (Strawberry gcc) runtime dynamically; put its
    # runtime DLLs first on PATH so the correct libstdc++/libgcc/libgomp load.
    for cand in (r"C:\Strawberry\c\bin", r"C:\Strawberry\perl\bin"):
        if os.path.isdir(cand):
            env["PATH"] = cand + os.pathsep + env.get("PATH", "")

    cmd = build_command(args, prompt_path)
    print("running:", " ".join(cmd), file=sys.stderr)

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        print(f"\nscout tool exited with code {proc.returncode}", file=sys.stderr)
        return proc.returncode

    layers, glob, header, gather = parse_rows(proc.stdout)
    if glob is None:
        sys.stderr.write(proc.stderr[-2000:])
        print("\nerror: no ATTN_SPARSITY_GLOBAL line found in tool output", file=sys.stderr)
        return 1

    print(f"\nmodel        : {args.model.name}")
    print(f"prompt tokens: {header.get('n_prompt', '?')}")
    print(f"attn layers  : {header.get('attn_layers', '?')}  query_stride={header.get('query_stride', '?')}"
          f"  block_size={header.get('block_size', '?')}")
    print("\nmass concentration (frac of valid keys holding X% of mass)"
          "  |  block-max selector recovery (mass kept at block budget)")
    print(f"{'layer':>5} {'valid':>8} {'f75':>7} {'f90':>7} {'f95':>7} {'f99':>7} "
          f"| {'bm06':>7} {'bm12':>7} {'bm25':>7} {'bm50':>7}")
    for row in layers:
        print(f"{row.get('layer', '?'):>5} {float(row.get('valid_mean', 0)):>8.0f} "
              f"{float(row.get('frac75', 0)):>7.3f} {float(row.get('frac90', 0)):>7.3f} "
              f"{float(row.get('frac95', 0)):>7.3f} {float(row.get('frac99', 0)):>7.3f} "
              f"| {float(row.get('bm06', 0)):>7.3f} {float(row.get('bm12', 0)):>7.3f} "
              f"{float(row.get('bm25', 0)):>7.3f} {float(row.get('bm50', 0)):>7.3f}")

    g75 = float(glob.get("frac75", 1.0))
    gate = glob.get("gate_75in25", "FAIL")
    bm25 = float(glob.get("bm25", 0.0))
    or25 = float(glob.get("or25", 0.0))
    sel_gate = glob.get("gate_bm25_99", "FAIL")
    print("\n--- global ---")
    print(f"valid_mean={float(glob.get('valid_mean', 0)):.0f}  "
          f"frac75={g75:.4f} frac90={float(glob.get('frac90', 0)):.4f} "
          f"frac95={float(glob.get('frac95', 0)):.4f} frac99={float(glob.get('frac99', 0)):.4f}")
    print(f"block-max recovery: bm06={float(glob.get('bm06', 0)):.4f} "
          f"bm12={float(glob.get('bm12', 0)):.4f} bm25={bm25:.4f} bm50={float(glob.get('bm50', 0)):.4f} "
          f"(oracle@25%={or25:.4f})")
    print(f"\nT5a gate 1 (>75% mass in <25% of K/V blocks): {gate}")
    print(f"T5a gate 2 (block-max top-25% recovers >=99% mass): {sel_gate}")

    # --- gather viability (T5b coopmat-FA gate) ---
    if gather is not None:
        tile = int(gather.get("tile", 0))
        pq25 = float(gather.get("pq25", 0))
        tu25 = float(gather.get("tu25", 0))
        union_penalty = float(gather.get("union_penalty25", 0))
        pq_gate = gather.get("gate_pq25_99", "?")
        tu_gate = gather.get("gate_tu25_99", "?")
        print(f"\n--- gather viability (tile={tile}) ---")
        print(f"per-query key-level recovery: pq06={float(gather.get('pq06',0)):.4f} "
              f"pq12={float(gather.get('pq12',0)):.4f} pq25={pq25:.4f} "
              f"pq50={float(gather.get('pq50',0)):.4f}  gate_pq25_99={pq_gate}")
        if tile > 0:
            print(f"tile-union shared-key recovery:  tu06={float(gather.get('tu06',0)):.4f} "
                  f"tu12={float(gather.get('tu12',0)):.4f} tu25={tu25:.4f} "
                  f"tu50={float(gather.get('tu50',0)):.4f}  gate_tu25_99={tu_gate}")
            print(f"union penalty at 25% budget: {union_penalty:+.4f} "
                  f"({abs(union_penalty)*100:.1f}% mass lost when sharing a key set across the tile)")
            if pq_gate == "PASS" and tu_gate == "PASS":
                print("=> gather-FA with coopmat tile is viable: per-query AND tile-union both pass 99% at 25% key budget")
            elif pq_gate == "PASS":
                print(f"=> per-query key select is viable (pq25={pq25:.3f}) but tile-union "
                      f"drops to tu25={tu25:.3f}. Gather-FA needs per-query key sets "
                      f"(breaks coopmat) or a larger budget.")
            else:
                print(f"=> even per-query key-level recovery fails at 25% budget (pq25={pq25:.3f}). "
                      f"Gather-FA not viable at this granularity.")
        else:
            if pq_gate == "PASS":
                print("=> per-query key-level recovery passes at 25% budget. Gather-FA is viable in principle; "
                      "tile-union test pending (rerun with --tile).")
            else:
                print(f"=> per-query key-level recovery fails at 25% budget (pq25={pq25:.3f}). "
                      "Gather-FA not viable.")

    if gate == "PASS" and sel_gate == "PASS":
        print("=> headroom AND a cheap block-max selector both hold at this context; "
              "sparse-FA prototype is justified. Next: confirm the trend holds at long context.")
    elif gate == "PASS":
        print(f"=> headroom exists but the cheap block-max selector recovers only {bm25:.1%} at 25% budget "
              f"(oracle {or25:.1%}). Need a better selector or larger budget before a shader.")
    else:
        print("=> attention mass is too diffuse; reject the sparse-FA family cheaply here.")

    if args.csv is not None:
        write_csv(args.csv, layers)
        print(f"\nwrote per-layer CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
