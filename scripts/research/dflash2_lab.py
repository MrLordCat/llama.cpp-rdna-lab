#!/usr/bin/env python3
"""Run a reproducible DFlash2 parity, stability, and concurrency matrix.

The tool can attach to an existing OpenAI-compatible llama-server or own the
full server lifecycle. Owned Windows servers are placed in a new process group
and stopped with CTRL_BREAK so GPU teardown remains graceful.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import dflash2_report


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "build_logs" / "dflash2-lab"
DEFAULT_PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "Water freezes at what temperature in Celsius?",
    "Name the largest planet in the Solar System.",
]


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sanitize_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    return cleaned or "dflash2"


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError("prompts JSON must be a non-empty array of non-empty strings")
    return value


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if payload is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"unexpected JSON response from {url}")
    return value


class CompletionClient:
    def __init__(
        self,
        base_url: str,
        max_tokens: int,
        temperature: float,
        seed: int,
        timeout: float,
        include_text: bool,
    ) -> None:
        self.endpoint = base_url.rstrip("/") + "/v1/completions"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.include_text = include_text

    def __call__(self, prompt: str, n_max: int) -> dict[str, Any]:
        body = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "speculative.n_max": n_max,
        }
        started = time.perf_counter()
        value = request_json(self.endpoint, body, self.timeout)
        wall_s = time.perf_counter() - started
        choices = value.get("choices", [])
        if not choices or not isinstance(choices[0], dict):
            raise RuntimeError(f"completion response contains no choice: {value}")
        text = str(choices[0].get("text", ""))
        sample: dict[str, Any] = {
            "text_sha256": text_sha256(text),
            "text_bytes": len(text.encode("utf-8")),
            "wall_s": wall_s,
            "usage": value.get("usage", {}),
            "timings": value.get("timings", {}),
            "finish_reason": choices[0].get("finish_reason"),
        }
        if self.include_text:
            sample["text"] = text
        return sample


def run_parallel(
    client: Callable[[str, int], dict[str, Any]], prompts: list[str], n_max: int, workers: int
) -> list[dict[str, Any]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda prompt: client(prompt, n_max), prompts))


def run_matrix(
    client: Callable[[str, int], dict[str, Any]],
    prompts: list[str],
    *,
    parallel: int,
    spec_n_max: int,
    serial_repeats: int,
    target_waves: int,
    spec_waves: int,
    identical_waves: int,
) -> dict[str, Any]:
    phases: dict[str, dict[str, Any]] = {}

    phases["serial_target"] = {
        "n_max": 0,
        "waves": [[client(prompt, 0) for prompt in prompts] for _ in range(serial_repeats)],
    }
    phases["serial_spec"] = {
        "n_max": spec_n_max,
        "waves": [[client(prompt, spec_n_max) for prompt in prompts] for _ in range(serial_repeats)],
    }
    phases["heterogeneous_target"] = {
        "n_max": 0,
        "waves": [run_parallel(client, prompts, 0, parallel) for _ in range(target_waves)],
    }
    phases["heterogeneous_spec"] = {
        "n_max": spec_n_max,
        "waves": [run_parallel(client, prompts, spec_n_max, parallel) for _ in range(spec_waves)],
    }

    identical_prompts = [prompts[0]] * parallel
    phases["identical_target"] = {
        "n_max": 0,
        "prompt": prompts[0],
        "waves": [run_parallel(client, identical_prompts, 0, parallel) for _ in range(identical_waves)],
    }
    phases["identical_spec"] = {
        "n_max": spec_n_max,
        "prompt": prompts[0],
        "waves": [run_parallel(client, identical_prompts, spec_n_max, parallel) for _ in range(identical_waves)],
    }
    return {"phases": phases}


def existing_llama_server_processes() -> str:
    if os.name == "nt":
        command = [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "Get-Process llama-server -ErrorAction SilentlyContinue | "
            "Select-Object Id,ProcessName,Path | ConvertTo-Json -Compress",
        ]
    else:
        command = ["pgrep", "-a", "llama-server"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def parse_env(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--env requires KEY=VALUE, got: {value}")
        key, item = value.split("=", 1)
        if not key:
            raise ValueError(f"--env has an empty key: {value}")
        result[key] = item
    return result


def parse_positive_ints(value: str) -> list[int]:
    if not value.strip():
        return []
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise ValueError("boundary token values must be positive integers")
    return result


def wait_ready(base_url: str, process: subprocess.Popen[str], timeout: float, log_path: Path) -> None:
    deadline = time.monotonic() + timeout
    health_url = base_url.rstrip("/") + "/health"
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = ""
            if log_path.exists():
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
            raise RuntimeError(f"llama-server exited during startup with {process.returncode}\n{tail}")
        try:
            request_json(health_url, None, 2.0)
            return
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = str(error)
            time.sleep(0.25)
    raise TimeoutError(f"server readiness timed out after {timeout}s: {last_error}")


def stop_server(process: subprocess.Popen[str], timeout: float) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"llama-server did not exit after the graceful signal within {timeout}s; "
            "it was deliberately not hard-killed for GPU driver safety"
        ) from error


def build_server_command(args: argparse.Namespace, port: int) -> list[str]:
    """Build the owned llama-server command without starting a GPU process."""
    command = [
        str(args.server_bin),
        "-m", str(args.model),
        "-md", str(args.draft_model),
        "--spec-type", "draft-dflash",
        "--spec-draft-n-max", str(args.spec_n_max),
        "-dev", args.devices,
        "-sm", args.split_mode,
        "-c", str(args.ctx_size),
        "-ngl", str(args.gpu_layers),
        "-np", str(args.parallel),
        "--cache-ram", "0",
        "--port", str(port),
    ]
    if args.draft_devices:
        command.extend(["-devd", args.draft_devices])
    if args.server_extra:
        command.extend(shlex.split(args.server_extra, posix=True))
    return command


def start_server(
    args: argparse.Namespace,
    port: int,
    log_path: Path,
    env_overrides: dict[str, str],
) -> tuple[subprocess.Popen[str], Any]:
    existing = existing_llama_server_processes()
    if existing:
        raise RuntimeError(
            "refusing to start a second llama-server; attach with --url or stop the existing process first:\n"
            + existing
        )

    command = build_server_command(args, port)

    environment = os.environ.copy()
    environment.update(env_overrides)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
    except Exception:
        log_file.close()
        raise
    return process, log_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DFlash2 parity/stability lab")
    parser.add_argument("--url", default="http://127.0.0.1:8089", help="attach URL when --server-bin is omitted")
    parser.add_argument("--server-bin", type=Path, help="own a llama-server process instead of attaching")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--devices", default="Vulkan1,Vulkan0")
    parser.add_argument("--draft-devices", default="")
    parser.add_argument("--split-mode", default="layer")
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--server-extra", default="")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--port", type=int, default=0, help="owned-server port; 0 selects a free port")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--spec-n-max", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--serial-repeats", type=int, default=2)
    parser.add_argument("--target-waves", type=int, default=3)
    parser.add_argument("--spec-waves", type=int, default=5)
    parser.add_argument("--identical-waves", type=int, default=2)
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--shutdown-timeout", type=float, default=90.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="vulkan-np4")
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--boundary-tokens",
        default="",
        help="comma-separated max_tokens values that must return full, bit-exact target/spec output",
    )
    parser.add_argument("--require-serial-parity", action="store_true")
    parser.add_argument("--require-identical-slot-stability", action="store_true")
    parser.add_argument("--require-boundary-parity", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    env_overrides = parse_env(args.env)
    boundary_tokens = parse_positive_ints(args.boundary_tokens)
    if args.parallel <= 0 or args.spec_n_max <= 0 or args.max_tokens <= 0:
        raise SystemExit("ERROR: parallel, spec-n-max, and max-tokens must be positive")
    if args.server_bin and (args.model is None or args.draft_model is None):
        raise SystemExit("ERROR: --server-bin requires --model and --draft-model")
    if args.quick:
        args.serial_repeats = 2
        args.target_waves = 1
        args.spec_waves = 2
        args.identical_waves = 1

    prompts = load_prompts(args.prompts_file)
    if args.prompts_file is None and args.parallel <= len(prompts):
        prompts = prompts[:args.parallel]
    if len(prompts) != args.parallel:
        raise SystemExit(
            f"ERROR: prompts count ({len(prompts)}) must equal --parallel ({args.parallel}); "
            "use --prompts-file for another slot count"
        )

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{timestamp}-{sanitize_label(args.label)}"
    output_dir = args.output_dir.resolve()
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    owned_log_path = output_dir / f"{stem}.server.log"

    process: subprocess.Popen[str] | None = None
    log_file: Any = None
    base_url = args.url.rstrip("/")
    try:
        if args.server_bin:
            port = args.port or find_free_port()
            base_url = f"http://127.0.0.1:{port}"
            process, log_file = start_server(args, port, owned_log_path, env_overrides)
            wait_ready(base_url, process, args.startup_timeout, owned_log_path)

        client = CompletionClient(
            base_url,
            args.max_tokens,
            args.temperature,
            args.seed,
            args.request_timeout,
            args.include_text,
        )
        matrix = run_matrix(
            client,
            prompts,
            parallel=args.parallel,
            spec_n_max=args.spec_n_max,
            serial_repeats=args.serial_repeats,
            target_waves=args.target_waves,
            spec_waves=args.spec_waves,
            identical_waves=args.identical_waves,
        )
        if boundary_tokens:
            boundary_cases = []
            original_max_tokens = client.max_tokens
            try:
                for max_tokens in boundary_tokens:
                    client.max_tokens = max_tokens
                    boundary_cases.append({
                        "max_tokens": max_tokens,
                        "target": client(prompts[0], 0),
                        "spec": client(prompts[0], args.spec_n_max),
                    })
            finally:
                client.max_tokens = original_max_tokens
            matrix["phases"]["max_token_boundaries"] = {
                "n_max": args.spec_n_max,
                "cases": boundary_cases,
            }
    finally:
        if process is not None:
            try:
                stop_server(process, args.shutdown_timeout)
            finally:
                if log_file is not None:
                    log_file.close()

    payload: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": timestamp,
        "prompts": prompts,
        "configuration": {
            "url": base_url,
            "parallel": args.parallel,
            "spec_n_max": args.spec_n_max,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "serial_repeats": args.serial_repeats,
            "target_waves": args.target_waves,
            "spec_waves": args.spec_waves,
            "identical_waves": args.identical_waves,
            "boundary_tokens": boundary_tokens,
            "server_bin": str(args.server_bin) if args.server_bin else None,
            "model": str(args.model) if args.model else None,
            "draft_model": str(args.draft_model) if args.draft_model else None,
            "devices": args.devices if args.server_bin else None,
            "draft_devices": args.draft_devices if args.server_bin else None,
            "split_mode": args.split_mode if args.server_bin else None,
            "ctx_size": args.ctx_size if args.server_bin else None,
            "gpu_layers": args.gpu_layers if args.server_bin else None,
            "server_extra": args.server_extra if args.server_bin else None,
            "env_overrides": env_overrides if args.server_bin else None,
        },
        **matrix,
    }
    analysis = dflash2_report.analyze_payload(payload)
    payload["analysis"] = analysis

    log_stats = None
    if args.server_bin and owned_log_path.exists():
        log_stats = dflash2_report.parse_server_log(owned_log_path)
        payload["server_log"] = log_stats

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(
        dflash2_report.render_markdown(payload, analysis, log_stats), encoding="utf-8"
    )

    print(f"json={json_path}")
    print(f"report={markdown_path}")
    if args.server_bin:
        print(f"server_log={owned_log_path}")
    for finding in analysis["findings"]:
        print(f"{finding['severity']}:{finding['code']}: {finding['detail']}")

    summary = analysis["summary"]
    if args.require_serial_parity and not all(summary["serial_spec_matches_target"]):
        return 2
    if args.require_identical_slot_stability and summary["identical_spec_distinct"] > 1:
        return 3
    if args.require_boundary_parity and not all(item["passed"] for item in summary["max_token_boundaries"]):
        return 4
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; owned server cleanup was requested.", file=sys.stderr)
        raise SystemExit(130)