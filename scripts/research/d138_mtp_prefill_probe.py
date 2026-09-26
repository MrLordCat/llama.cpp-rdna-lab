"""D138: inspect the MTP prompt path without running a benchmark.

What it does:
  1. starts llama-server (ROCm build) with a fixed prompt config,
  2. asks for 48 greedy tokens (temp 0, seed 42) with return_tokens,
  3. prints the generated token ids so the MTP arm can be compared against
     the spec=none arm (output correctness, not speed),
  4. prints the per-batch spec-phase timings from the server log, which are
     what the prompt path looks like from the inside:
         LLAMA_SPEC_SERVER_PHASE_TIMING=1
         LLAMA_SPEC_SERVER_PHASE_TIMING_MAX_ROWS=8192

Usage:
  python scripts/research/d138_mtp_prefill_probe.py --spec mtp   --port 64311
  python scripts/research/d138_mtp_prefill_probe.py --spec none  --port 64312

The two runs are meant to be diffed by hand: same prompt, same seed, so the
token ids must match; the phase lines explain where the prompt time went.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FILLER = (
    "Memory bandwidth is the binding constraint when a matrix-vector kernel "
    "streams quantized weights for a single token. The kernel reads every "
    "weight byte once per token, so the achievable rate is set by how evenly "
    "the reads fill the memory controllers, not by arithmetic. "
)


def make_prompt(target_chars: int = 7000, unfinished: bool = False) -> str:
    body = (FILLER * (target_chars // len(FILLER) + 1))[:target_chars]
    if unfinished:
        # A prompt that cannot be "finished": the model has no choice but to
        # continue, so a first-token mismatch between arms is a real mismatch
        # and not a coin flip between EOS and text.
        return body.rstrip()[:-len(" evenly")] if body.rstrip().endswith("evenly") else body.rstrip()
    return body + "\n\nSummarize the paragraph above in one short sentence.\n"


def wait_health(port: int, proc: subprocess.Popen, timeout: float = 600.0) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1.0)
    return False


def post_json(port: int, path: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def stop_server(proc: subprocess.Popen) -> None:
    """Same contract as bench2: short graceful window, then make sure it is gone."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            except (OSError, ValueError):
                proc.terminate()
        else:
            proc.terminate()
        proc.wait(timeout=15)
        return
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        pass
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True, check=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", choices=["none", "mtp"], default="mtp")
    ap.add_argument("--spec-n", type=int, default=3)
    ap.add_argument("--port", type=int, default=64311)
    ap.add_argument("--level-ctx", type=int, default=8192)
    ap.add_argument("--prompt-chars", type=int, default=7000)
    ap.add_argument("--n-predict", type=int, default=48)
    ap.add_argument("--unfinished", action="store_true",
                    help="send a prompt that is cut mid-sentence, so the model must continue")
    ap.add_argument("--server-bin", default="build-rocm72/bin/llama-server.exe")
    ap.add_argument("--model", default="models/Qwen3.8-27B-UD-Q4_K_M.gguf")
    ap.add_argument("--out-dir", default="build_logs/d138-mtp-probe")
    ap.add_argument("--keep-server-env", action="store_true",
                    help="do not add the phase-timing env vars")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VAL",
                    help="extra environment for the server process")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.spec}-{int(time.time())}.log"

    cmd = [
        str(ROOT / args.server_bin), "-m", str(ROOT / args.model),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--flash-attn", "on", "-np", "1",
        "-c", str(args.level_ctx), "-b", "8192", "-ub", "1024",
        "--cache-type-k", "f8_e4m3", "--cache-type-v", "f8_e4m3",
        "-ngl", "999", "--seed", "42", "--no-warmup",
        "--cache-ram", "0", "--ctx-checkpoints", "0",
        "-dev", "ROCm1,ROCm0", "-sm", "layer", "-ts", "1,1", "--fit", "off",
    ]
    if args.spec == "mtp":
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(args.spec_n)]
    else:
        cmd += ["--spec-type", "none"]

    env = dict(os.environ)
    env["PATH"] = r"C:\Program Files\AMD\ROCm\7.2\bin" + os.pathsep + env.get("PATH", "")
    if not args.keep_server_env:
        env["LLAMA_SPEC_SERVER_PHASE_TIMING"] = "1"
        env["LLAMA_SPEC_SERVER_PHASE_TIMING_MAX_ROWS"] = "8192"
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    print(f"[probe] spec={args.spec} port={args.port} log={log_path}")
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
                                env=env, creationflags=creationflags)
        try:
            if not wait_health(args.port, proc):
                print("[probe] server did not become healthy")
                return 1
            print("[probe] server healthy, sending completion")
            t0 = time.time()
            res = post_json(args.port, "/completion", {
                "prompt": make_prompt(args.prompt_chars, unfinished=args.unfinished),
                "n_predict": args.n_predict,
                "temperature": 0.0,
                "seed": 42,
                "top_k": 1,
                "cache_prompt": False,
                "return_tokens": True,
            })
            wall = time.time() - t0
        finally:
            stop_server(proc)

    tim = res.get("timings", {})
    toks = res.get("tokens", []) or []
    content = (res.get("content") or "").strip()
    meta = {k: v for k, v in res.items() if k not in ("content", "tokens", "timings", "prompt")}
    print(f"[probe] meta={json.dumps(meta, ensure_ascii=False)[:300]}")
    print(f"[probe] prompt_n={tim.get('prompt_n')} prompt_ms={tim.get('prompt_ms')} "
          f"predicted_n={tim.get('predicted_n')} wall={wall:.1f}s")
    print(f"[probe] unique_token_ids={len(set(toks))} first_ids={toks[:16]}")
    print(f"[probe] content[:160]={content[:160]!r}")

    print("\n[probe] spec phase lines (prompt batches are the ones with rows > 16):")
    phases = []
    for line in open(log_path, encoding="utf-8", errors="replace"):
        if "spec phase=target" in line:
            phases.append(line.strip())
    for p in phases:
        print("  " + p)
    if not phases:
        print("  (none - the timing env did not reach the server?)")

    print("\n[probe] prompt/eval summary:")
    for line in open(log_path, encoding="utf-8", errors="replace"):
        s = line.strip()
        if s.startswith("prompt eval time") or s.startswith("eval time"):
            print("  " + s)

    print(f"\n[probe] tokens json: {json.dumps(toks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
