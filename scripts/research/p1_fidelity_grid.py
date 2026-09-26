"""D138 sanity grid: does the f8-native FA path degrade long-context output?

Usage: python build_logs/tmp_p1_grid.py <label> <kvt> <kv_kEnv> <reps>
  label   - name for the log/json
  kvt     - f8_e4m3 | f16
  kv_kEnv - comma separated ENV=VAL overrides for the server process
  reps    - how many times the filler sentence is repeated
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
PORT = 8125


def wait_health(timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200 and json.loads(r.read().decode()).get("status") == "ok":
                    return True
        except Exception:
            time.sleep(1.0)
    return False


def main() -> int:
    label, kvt, overrides, reps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    server = Path(sys.argv[5]) if len(sys.argv) > 5 else SERVER
    fa = sys.argv[6] if len(sys.argv) > 6 else "on"
    devs = sys.argv[7] if len(sys.argv) > 7 else "ROCm1,ROCm0"
    env = dict(os.environ)
    env["PATH"] = r"C:\Program Files\AMD\ROCm\7.2\bin;" + env["PATH"]
    for item in [o for o in overrides.split(",") if o and o != "-"]:
        if item.startswith("!"):
            env.pop(item[1:], None)
            continue
        k, v = item.split("=")
        env[k] = v

    filler = (
        "The memory subsystem of the accelerator was measured while a long-context decode loop was running. "
        "Each layer issues matrix-vector products whose weights stream from device memory. "
        "The counters show that the weight stream, not the arithmetic units, sets the pace. "
    ) * reps
    prompt = filler + "\nThe capital of France is"

    log = ROOT / "build_logs" / f"p1-grid-{label}.server.log"
    cmd = [
        str(server), "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
        "-c", "8192", "-ngl", "99", "-dev", devs, "-sm", "layer", "-ts", "1,1",
        "-fit", "off", "--flash-attn", fa, "-ctk", kvt, "-ctv", kvt,
        "-b", "8192", "-ub", "1024", "--no-warmup", "--cache-ram", "0", "--ctx-checkpoints", "0",
        "--seed", "42",
    ]
    with open(log, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        if not wait_health():
            print(f"{label}: server not healthy")
            return 2
        body = json.dumps({"prompt": prompt, "n_predict": 24, "temperature": 0.0, "top_k": 1,
                           "seed": 42, "cache_prompt": False, "return_tokens": True,
                           "repeat_penalty": 1.0}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            resp = json.loads(r.read().decode())
    finally:
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.wait(timeout=120)
        except Exception:
            proc.kill()

    toks = resp.get("tokens") or []
    unique = len(set(toks))
    print(f"{label:28s} kvt={kvt:8s} reps={reps:3d} ntok={len(toks):3d} uniq={unique:3d} "
          f"ptps={resp['timings']['prompt_per_second']:7.1f} gtps={resp['timings']['predicted_per_second']:6.2f} "
          f"| {resp.get('content','')[:70]!r}")
    (ROOT / "build_logs" / f"p1-grid-{label}.json").write_text(
        json.dumps({"label": label, "kvt": kvt, "overrides": overrides, "reps": reps,
                    "server": str(server),
                    "tokens": toks, "content": resp.get("content"),
                    "timings": resp.get("timings")}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
