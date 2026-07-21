#!/usr/bin/env python3
"""Reproduce long-context checkpoint overhead in a single llama-server slot.

The workload deliberately replaces the final marker between requests. This
forces a hybrid/recurrent model to restore the latest usable checkpoint, then
process a small suffix at an already large context position. The same generated
text and request sequence can be used before and after runtime changes.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "build_logs" / "checkpoint-bench"


def parse_args() -> argparse.Namespace:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(
        description="Reproduce checkpoint restore/save overhead at a long context",
    )
    parser.add_argument("--label", default=f"checkpoint-longctx-{stamp}")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--server-bin",
        default=str(ROOT / "build-rocm-full" / "bin" / "llama-server.exe"),
    )
    parser.add_argument(
        "--model",
        default=str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
    )
    parser.add_argument(
        "--mmproj",
        default=str(ROOT / "models" / "mmproj-F16.gguf"),
        help="empty string disables the multimodal projector",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ctx-size", type=int, default=131072)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--ubatch-size", type=int, default=1024)
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q8_0")
    parser.add_argument("--devices", default="ROCm1,ROCm0")
    parser.add_argument("--tensor-split", default="27,37")
    parser.add_argument("--ctx-checkpoints", type=int, default=4)
    parser.add_argument("--checkpoint-min-step", type=int, default=0)
    staging_group = parser.add_mutually_exclusive_group()
    staging_group.add_argument(
        "--pinned-staging",
        dest="pinned_staging",
        action="store_true",
        help="use reusable backend-pinned host memory for checkpoint transfers",
    )
    staging_group.add_argument(
        "--no-pinned-staging",
        dest="pinned_staging",
        action="store_false",
        help="force pageable checkpoint memory for a controlled comparison",
    )
    parser.set_defaults(pinned_staging=None)
    parser.add_argument("--spec-type", default="draft-mtp")
    parser.add_argument("--spec-draft-n-max", type=int, default=2)
    parser.add_argument("--spec-prefill-window", type=int)
    parser.add_argument("--spec-prefill-sparse-stride", type=int)
    parser.add_argument("--spec-prefill-sparse-chunk", type=int)
    parser.add_argument("--root-tokens", type=int, default=57000)
    parser.add_argument(
        "--delta-tokens",
        default="3072,1280,4096",
        help="comma-separated raw-content token targets for successive branches",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--endpoint",
        choices=("completion", "chat"),
        default="completion",
        help="completion gives an exact append-only LCP; chat also covers template behavior",
    )
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--soft-stop-timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rocm-bin",
        default="",
        help="ROCm bin directory; auto-detected under Program Files when omitted",
    )
    parser.add_argument("--server-extra", default="")
    return parser.parse_args()


def http_json(method: str, url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:2000]}") from exc


def choose_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def wait_ready(base_url: str, timeout: float, proc: subprocess.Popen[bytes], log_path: Path) -> None:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        return_code = proc.poll()
        if return_code is not None:
            log_tail = ""
            if log_path.exists():
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                f"server exited before readiness with code {return_code}\n{log_tail}"
            )
        for endpoint in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(base_url + endpoint, timeout=5) as response:
                    if response.status == 200:
                        return
            except Exception as exc:  # noqa: BLE001 - readiness is best effort
                last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"server readiness timed out after {timeout:.0f}s: {last_error}")


def running_server_pids() -> list[str]:
    if os.name != "nt":
        return []
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: list[str] = []
    for line in result.stdout.splitlines():
        match = re.match(r'^"llama-server\.exe","(\d+)"', line.strip(), re.IGNORECASE)
        if match:
            pids.append(match.group(1))
    return pids


def stop_server(proc: subprocess.Popen[bytes], log_handle: Any, timeout: float) -> bool:
    if proc.poll() is not None:
        log_handle.close()
        return True
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: graceful stop signal failed: {exc}", flush=True)
    try:
        proc.wait(timeout=timeout)
        log_handle.close()
        return True
    except subprocess.TimeoutExpired:
        print(
            f"WARNING: llama-server pid={proc.pid} did not exit after {timeout:.0f}s; "
            "leaving it alive because hard termination is disabled.",
            flush=True,
        )
        return False


def make_source_blob(min_chars: int, seed: int) -> str:
    lines: list[str] = []
    total = 0
    state = seed & 0xFFFFFFFF
    index = 0
    while total < min_chars:
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        line = (
            f"namespace checkpoint_probe_{index % 97} {{ constexpr unsigned value_{index} = "
            f"0x{state:08x}u; constexpr const char * label_{index} = \"phase-{index % 211:03d}\"; }}\n"
        )
        lines.append(line)
        total += len(line)
        index += 1
    return "".join(lines)


def token_count(base_url: str, text: str) -> int:
    result = http_json("POST", base_url + "/tokenize", {"content": text}, timeout=120.0)
    tokens = result.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError("/tokenize response did not contain a token list")
    return len(tokens)


def fit_text_to_tokens(base_url: str, source: str, target: int) -> tuple[str, int]:
    if target <= 0:
        return "", 0
    source_tokens = token_count(base_url, source)
    if source_tokens < target:
        raise RuntimeError(f"generated source is too small: {source_tokens} < {target} tokens")

    low = 1
    high = len(source)
    best_text = source
    best_tokens = source_tokens
    for _ in range(20):
        if low > high:
            break
        mid = (low + high) // 2
        candidate = source[:mid]
        count = token_count(base_url, candidate)
        if abs(count - target) < abs(best_tokens - target):
            best_text = candidate
            best_tokens = count
        if count < target:
            low = mid + 1
        elif count > target:
            high = mid - 1
        else:
            return candidate, count
    return best_text, best_tokens


def parse_phase_lines(text: str, phase: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if f"checkpoint phase={phase} " in line]
    values = [
        {key: float(value) for key, value in re.findall(r"([a-z_]+)=([0-9.]+)", line)}
        for line in lines
    ]
    return {
        f"checkpoint_{phase}_count": len(lines),
        f"checkpoint_{phase}_ms": round(sum(item.get("total", 0.0) for item in values), 3),
        f"checkpoint_{phase}_sync_ms": round(sum(item.get("sync", 0.0) for item in values), 3),
        f"checkpoint_{phase}_transfer_ms": round(sum(item.get("transfer", 0.0) for item in values), 3),
        f"checkpoint_{phase}_alloc_ms": round(sum(item.get("alloc", 0.0) for item in values), 3),
        f"checkpoint_{phase}_bytes": int(sum(item.get("bytes", 0.0) for item in values)),
    }


def parse_request_log(text: str) -> dict[str, Any]:
    prompt_matches = list(
        re.finditer(
            r"prompt eval time =\s+([0-9.]+) ms /\s+(\d+) tokens .*?([0-9.]+) tokens per second",
            text,
        )
    )
    eval_matches = list(
        re.finditer(
            r"(?m)^\s*eval time =\s+([0-9.]+) ms /\s+(\d+) tokens .*?([0-9.]+) tokens per second",
            text,
        )
    )
    acceptance_matches = list(
        re.finditer(
            r"draft acceptance rate =\s*([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\)",
            text,
        )
    )
    result: dict[str, Any] = {
        "prompt_eval_ms": 0.0,
        "prompt_eval_tokens": 0,
        "prompt_eval_tps": 0.0,
        "decode_ms": 0.0,
        "decode_tokens": 0,
        "decode_tps": 0.0,
        "draft_acceptance": 0.0,
        "draft_accepted": 0,
        "draft_generated": 0,
        "checkpoints_created": text.count("created context checkpoint"),
        "checkpoints_restored": text.count("restored context checkpoint"),
        "full_reprocess": "forcing full prompt re-processing" in text,
        "speculative_batch_failures": text.count("failed to process speculative batch"),
    }
    if prompt_matches:
        match = prompt_matches[-1]
        result.update(
            {
                "prompt_eval_ms": float(match.group(1)),
                "prompt_eval_tokens": int(match.group(2)),
                "prompt_eval_tps": float(match.group(3)),
            }
        )
    if eval_matches:
        match = eval_matches[-1]
        result.update(
            {
                "decode_ms": float(match.group(1)),
                "decode_tokens": int(match.group(2)),
                "decode_tps": float(match.group(3)),
            }
        )
    if acceptance_matches:
        match = acceptance_matches[-1]
        result.update(
            {
                "draft_acceptance": float(match.group(1)),
                "draft_accepted": int(match.group(2)),
                "draft_generated": int(match.group(3)),
            }
        )
    result.update(parse_phase_lines(text, "save"))
    result.update(parse_phase_lines(text, "load"))
    return result


def read_request_log(path: Path, offset: int, timeout: float = 30.0) -> str:
    deadline = time.time() + timeout
    text = ""
    while time.time() < deadline:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            text = handle.read()
        if "prompt eval time =" in text:
            return text
        time.sleep(0.2)
    return text


def run_request(
    base_url: str,
    content: str,
    max_tokens: int,
    timeout: float,
    endpoint: str,
) -> tuple[dict[str, Any], float]:
    if endpoint == "chat":
        url = base_url + "/v1/chat/completions"
        payload = {
            "model": "local-model",
            "messages": [
                {
                    "role": "system",
                    "content": "Inspect the supplied source snapshot. Reply with exactly OK.",
                },
                {"role": "user", "content": content},
            ],
            "cache_prompt": True,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    else:
        url = base_url + "/completion"
        payload = {
            "prompt": content,
            "cache_prompt": True,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    started = time.perf_counter()
    response = http_json("POST", url, payload, timeout)
    return response, time.perf_counter() - started


def resolve_rocm_bin(explicit: str) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.is_dir() else None

    roots = [
        Path(os.environ.get("ROCM_PATH", "")) if os.environ.get("ROCM_PATH") else None,
        Path(os.environ.get("HIP_PATH", "")) if os.environ.get("HIP_PATH") else None,
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "AMD" / "ROCm",
    ]
    candidates: list[Path] = []
    for root in roots:
        if root is None or not root.exists():
            continue
        if (root / "bin").is_dir():
            candidates.append(root / "bin")
        for child in root.iterdir() if root.is_dir() else []:
            if child.is_dir() and (child / "bin").is_dir():
                candidates.append(child / "bin")

    for candidate in sorted(candidates, reverse=True):
        if any(candidate.glob("amdhip64*.dll")):
            return candidate
    return None


def start_server(args: argparse.Namespace, port: int, log_path: Path) -> tuple[subprocess.Popen[bytes], Any, list[str]]:
    cmd = [
        str(Path(args.server_bin)),
        "-m",
        str(Path(args.model)),
        "--host",
        args.host,
        "--port",
        str(port),
        "-c",
        str(args.ctx_size),
        "-t",
        "8",
        "--threads-http",
        "8",
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--parallel",
        "1",
        "-ngl",
        "999",
        "--cache-type-k",
        args.cache_type_k,
        "--cache-type-v",
        args.cache_type_v,
        "--metrics",
        "--cache-ram",
        "0",
        "--ctx-checkpoints",
        str(args.ctx_checkpoints),
        "--checkpoint-every-n-tokens",
        "-1",
        "--checkpoint-min-step",
        str(args.checkpoint_min_step),
        "--flash-attn",
        "on",
        "--no-warmup",
        "--seed",
        str(args.seed),
        "-dev",
        args.devices,
        "-sm",
        "layer",
        "-ts",
        args.tensor_split,
        "--spec-type",
        args.spec_type,
    ]
    if args.spec_type != "none":
        cmd.extend(["--spec-draft-n-max", str(args.spec_draft_n_max)])
    if args.mmproj:
        cmd.extend(["--mmproj", str(Path(args.mmproj))])
    if args.server_extra:
        import shlex

        cmd.extend(shlex.split(args.server_extra, posix=(os.name != "nt")))

    env = os.environ.copy()
    env["LLAMA_CHECKPOINT_TIMING"] = "1"
    if args.pinned_staging is not None:
        env["LLAMA_CHECKPOINT_PINNED_STAGING"] = "1" if args.pinned_staging else "0"
    if args.spec_prefill_window is not None:
        env["LLAMA_SPEC_PREFILL_WINDOW"] = str(args.spec_prefill_window)
    if args.spec_prefill_sparse_stride is not None:
        env["LLAMA_SPEC_PREFILL_SPARSE_STRIDE"] = str(args.spec_prefill_sparse_stride)
    if args.spec_prefill_sparse_chunk is not None:
        env["LLAMA_SPEC_PREFILL_SPARSE_CHUNK"] = str(args.spec_prefill_sparse_chunk)
    rocm_bin = resolve_rocm_bin(args.rocm_bin)
    if rocm_bin is not None:
        env["PATH"] = str(rocm_bin) + os.pathsep + env.get("PATH", "")
        rocm_root = rocm_bin.parent
        env.setdefault("ROCM_PATH", str(rocm_root))
        env.setdefault("HIP_PATH", str(rocm_root))
    log_handle = log_path.open("wb")
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    return proc, log_handle, cmd


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "stage",
        "target_content_tokens",
        "actual_content_tokens",
        "api_prompt_tokens",
        "wall_ms",
        "prompt_eval_tokens",
        "prompt_eval_ms",
        "prompt_eval_tps",
        "checkpoints_restored",
        "checkpoints_created",
        "checkpoint_load_count",
        "checkpoint_load_ms",
        "checkpoint_load_sync_ms",
        "checkpoint_load_transfer_ms",
        "checkpoint_save_count",
        "checkpoint_save_ms",
        "checkpoint_save_sync_ms",
        "checkpoint_save_alloc_ms",
        "checkpoint_save_transfer_ms",
        "full_reprocess",
        "speculative_batch_failures",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    server_bin = Path(args.server_bin)
    model = Path(args.model)
    if not server_bin.exists():
        raise FileNotFoundError(f"server binary not found: {server_bin}")
    if not model.exists():
        raise FileNotFoundError(f"model not found: {model}")
    if args.mmproj and not Path(args.mmproj).exists():
        raise FileNotFoundError(f"mmproj not found: {args.mmproj}")
    existing = running_server_pids()
    if existing:
        raise RuntimeError(f"llama-server already running: {', '.join(existing)}")

    delta_targets = [int(part.strip()) for part in args.delta_tokens.split(",") if part.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.label}.server.log"
    json_path = out_dir / f"{args.label}.json"
    csv_path = out_dir / f"{args.label}.csv"
    port = args.port or choose_port(args.host)
    base_url = f"http://{args.host}:{port}"

    proc, log_handle, cmd = start_server(args, port, log_path)
    stopped = False
    rows: list[dict[str, Any]] = []
    try:
        print(f"Starting server pid={proc.pid} on {base_url}", flush=True)
        wait_ready(base_url, args.startup_timeout, proc, log_path)
        print("Server ready; constructing token-stable workload", flush=True)

        root_source = make_source_blob(max(700000, args.root_tokens * 14), 0xC0FFEE)
        root_text, root_actual = fit_text_to_tokens(base_url, root_source, args.root_tokens)
        deltas: list[tuple[str, int]] = []
        for index, target in enumerate(delta_targets):
            source = make_source_blob(max(120000, target * 18), 0xABC000 + index)
            deltas.append(fit_text_to_tokens(base_url, source, target))

        stages: list[tuple[str, str, int, int]] = []
        prefix = root_text
        # Each request extends the previous user content but deliberately omits
        # the previous generated answer. The LCP therefore ends at the prompt
        # boundary and exercises the N-4 recurrent checkpoint used by agents.
        stages.append(("root", prefix, args.root_tokens, root_actual))
        cumulative_target = args.root_tokens
        cumulative_actual = root_actual
        for index, ((delta_text, delta_actual), delta_target) in enumerate(zip(deltas, delta_targets), start=1):
            prefix += "\n" + delta_text
            cumulative_target += delta_target
            cumulative_actual += delta_actual
            stages.append(
                (
                    f"delta-{index}",
                    prefix,
                    cumulative_target,
                    cumulative_actual,
                )
            )

        for stage, content, target_tokens, actual_tokens in stages:
            offset = log_path.stat().st_size
            print(
                f"Running {stage}: content_tokens={actual_tokens} target={target_tokens}",
                flush=True,
            )
            response, wall_s = run_request(
                base_url, content, args.max_tokens, args.request_timeout, args.endpoint
            )
            request_log = read_request_log(log_path, offset)
            metrics = parse_request_log(request_log)
            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            response_timings = response.get("timings", {}) if isinstance(response, dict) else {}
            row = {
                "stage": stage,
                "target_content_tokens": target_tokens,
                "actual_content_tokens": actual_tokens,
                "api_prompt_tokens": int(
                    usage.get("prompt_tokens") or response_timings.get("prompt_n") or 0
                ),
                "completion_tokens": int(
                    usage.get("completion_tokens") or response_timings.get("predicted_n") or 0
                ),
                "wall_ms": round(wall_s * 1000.0, 3),
                **metrics,
            }
            rows.append(row)
            print(
                f"  prompt={row['prompt_eval_tokens']} tokens, {row['prompt_eval_tps']:.2f} tok/s, "
                f"decode={row['decode_tps']:.2f} tok/s, acceptance={row['draft_acceptance']:.2%}, "
                f"restore={row['checkpoint_load_ms']:.2f} ms, save={row['checkpoint_save_ms']:.2f} ms, "
                f"created={row['checkpoints_created']}",
                flush=True,
            )

        result = {
            "label": args.label,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "server_command": cmd,
            "server_log": str(log_path),
            "rows": rows,
        }
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_csv(csv_path, rows)
        print(f"Wrote {json_path}", flush=True)
        print(f"Wrote {csv_path}", flush=True)
        return 0
    finally:
        stopped = stop_server(proc, log_handle, args.soft_stop_timeout)
        if not stopped:
            print("Server remains alive; no hard kill was attempted.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
