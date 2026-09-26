"""D138/P1 fidelity check for the GQA-columns FA prototype.

Runs llama-server twice on the same lane (GQA mode off and on), sends the same
deterministic completion request to both, and stores the raw token ids so the
two runs can be compared exactly. Tokens, not text: terminal encoding cannot
mask a numeric break.

Usage: python build_logs/tmp_p1_fidelity.py <gqa_env 0|1> <out_json>
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "build-rocm72" / "bin" / "llama-server.exe"
MODEL = ROOT / "models" / "Qwen3.8-27B-UD-Q4_K_M.gguf"
PORT = 8123
HOST = "127.0.0.1"

PROMPT = (
    (
        "The memory subsystem of the accelerator was measured while a long-context decode loop was running. "
        "Each layer issues matrix-vector products whose weights stream from device memory. "
        "The counters show that the weight stream, not the arithmetic units, sets the pace. "
    ) * 120
    + "\nThe capital of France is"
)

def wait_health(timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=3) as r:
                if r.status == 200 and json.loads(r.read().decode()).get("status") == "ok":
                    return True
        except Exception:
            time.sleep(1.0)
    return False


def main() -> int:
    gqa = sys.argv[1]
    out = Path(sys.argv[2])
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 8192
    env = dict(os.environ)
    env["PATH"] = r"C:\Program Files\AMD\ROCm\7.2\bin;" + env["PATH"]
    if gqa == "1":
        env.pop("GGML_ROCM_FATTN_F8_GQA_COLS", None)
    else:
        env["GGML_ROCM_FATTN_F8_GQA_COLS"] = "0"

    log = out.with_suffix(".server.log")
    cmd = [
        str(SERVER), "-m", str(MODEL),
        "--host", HOST, "--port", str(PORT),
        "-c", str(ctx), "-ngl", "99", "-dev", "ROCm1,ROCm0", "-sm", "layer", "-ts", "1,1",
        "-fit", "off", "--flash-attn", "on", "-ctk", "f8_e4m3", "-ctv", "f8_e4m3",
        "-b", "8192", "-ub", "1024", "--no-warmup", "--cache-ram", "0", "--ctx-checkpoints", "0",
        "--seed", "42",
    ]
    with open(log, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    prompt = (
        "The memory subsystem of the accelerator was measured while a long-context decode loop was running. "
        "Each layer issues matrix-vector products whose weights stream from device memory. "
        "The counters show that the weight stream, not the arithmetic units, sets the pace. "
    ) * reps + "\nThe capital of France is"
    try:
        if not wait_health():
            print("server did not become healthy", flush=True)
            return 2
        body = json.dumps({
            "prompt": prompt, "n_predict": 48, "temperature": 0.0, "top_k": 1,
            "seed": 42, "cache_prompt": False, "return_tokens": True,
            "repeat_penalty": 1.0,
        }).encode()
        req = urllib.request.Request(
            f"http://{HOST}:{PORT}/completion", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read().decode())
    finally:
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.wait(timeout=120)
        except Exception:
            proc.kill()
            proc.wait(timeout=60)

    out.write_text(json.dumps({
        "gqa": gqa,
        "tokens": resp.get("tokens"),
        "content": resp.get("content"),
        "timings": resp.get("timings"),
    }, indent=1), encoding="utf-8")
    print(f"gqa={gqa} tokens={len(resp.get('tokens') or [])} "
          f"prompt_tps={resp['timings']['prompt_per_second']:.1f} "
          f"gen_tps={resp['timings']['predicted_per_second']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
