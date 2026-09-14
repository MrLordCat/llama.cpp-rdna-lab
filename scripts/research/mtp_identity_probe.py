#!/usr/bin/env python3
"""Deterministic MTP identity probe for llama-server.

Speculative decoding is exact: with greedy sampling (temperature 0) and a
fixed seed, a server with MTP enabled must emit byte-identical text to the
same server with MTP disabled. Any divergence is a real bug, not noise.

This probe starts one server configuration, sends two identical greedy
requests (plus an optional chat/tools request), and reports a content hash,
a periodicity metric for repetition loops, and the server-side spec lines
found in the log. A ladder mode runs several configurations back to back and
reports which lanes diverge from the spec=none control.

Usage:
  python scripts/research/mtp_identity_probe.py --ladder --server-bin ... \
      --model ... --mmproj ... --out /tmp/mtp_probe

Each lane is started and stopped sequentially; the model server is never run
concurrently with another one. Logs are written next to --out.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_LIBPATH = "/home/chris/rocm/lib"

# The corpus is fixed for the whole run; its hashes are recorded so a lane
# comparison is only trusted when every lane saw the same bytes.
CORPUS_FILES = (
    "common/speculative.cpp",
    "common/sampling.cpp",
    "src/llama-kv-cache.cpp",
    "README.md",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "file path"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the entries of a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "directory path"}},
                "required": ["path"],
            },
        },
    },
]

CHAT_SYSTEM = (
    "You are a coding agent inside a repository. Use the provided tools to answer. "
    "Call a tool when you need information; never answer from memory."
)
CHAT_USER = (
    "Find where the speculative decoding accept loop lives and tell me the file name. "
    "Start by listing the common/ directory."
)

COMPLETION_SUFFIX = (
    "\n\n# The repository snapshot above ends here.\n"
    "# Task: in one short paragraph, describe what the function "
    "common_sampler_sample_and_accept_n does and why the draft tokens are compared "
    "with the sampled tokens. Keep it factual.\n# Answer:\n"
)

LANES = (
    # name, spec args, extra env, cache types
    ("none", [], {}, None),
    ("mtp", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"], {}, None),
    ("mtp-nohybrid", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
     {"LLAMA_VK_MTP_KV_LAST_F16": "0"}, None),
    # A pure f16 KV lane is not runnable at ctx=151552 on 2x16 GiB; it is left
    # out on purpose rather than reported as a failure.
    ("mtp-nohandoff", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
     {"LLAMA_MTP_DEVICE_HANDOFF": "0"}, None),
    ("mtp-nosparse", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
     {"LLAMA_SPEC_PREFILL_WINDOW": "0", "LLAMA_MTP_DEFER_SPARSE_PREFILL": "0"}, None),
    # The fork defaults the ROCm pipeline scheduler to one copy buffer
    # (ggml/src/ggml-backend.cpp, GGML_SCHED_PIPELINE_COPIES); 2 restores the
    # upstream parallel default and is the suspected nondeterminism source.
    ("none-copies2", [], {"GGML_SCHED_PIPELINE_COPIES": "2"}, None),
    ("mtp-copies2", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
     {"GGML_SCHED_PIPELINE_COPIES": "2"}, None),
)


def corpus(repo: Path, ctx_tokens: int) -> str:
    """Deterministic natural-text prompt of roughly ctx_tokens tokens."""
    budget_chars = max(4096, ctx_tokens * 4)
    parts: list[str] = []
    size = 0
    while size < budget_chars:
        for name in CORPUS_FILES:
            path = repo / name
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            parts.append(text)
            size += len(text)
            if size >= budget_chars:
                break
    return "".join(parts)[:budget_chars]


def corpus_hash(repo: Path) -> dict[str, str]:
    out = {}
    for name in CORPUS_FILES:
        path = repo / name
        if path.is_file():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return out


def http_json(url: str, payload: dict | None, timeout: float = 600.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready(base: str, proc: subprocess.Popen, timeout_s: float = 420.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        with contextlib.suppress(urllib.error.URLError, OSError, ValueError):
            if http_json(base + "/health", None, timeout=3) is not None:
                return True
        time.sleep(0.5)
    return False


def periodicity(text: str) -> dict:
    """Shortest exact period in the tail of the text, in characters."""
    if len(text) < 64:
        return {"period": 0, "repeats": 0}
    tail = text[-1024:]
    for p in range(4, 257):
        if len(tail) < 3 * p:
            break
        if all(tail[i] == tail[i - p] for i in range(p, len(tail))):
            return {"period": p, "repeats": len(tail) // p}
    return {"period": 0, "repeats": 0}


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def run_completion(base: str, prompt: str, n_predict: int, tag: str) -> dict:
    body = {
        "prompt": prompt,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "seed": 0,
        "n_predict": n_predict,
        "cache_prompt": True,
        "stream": False,
    }
    t0 = time.time()
    result = http_json(base + "/completion", body)
    text = result.get("content", "")
    return {
        "probe": tag,
        "hash": digest_text(text),
        "chars": len(text),
        "prompt_tokens": result.get("tokens_evaluated"),
        "gen_tokens": result.get("tokens_predicted"),
        "prompt_tps": round(result.get("timings", {}).get("prompt_per_second", 0.0), 2),
        "gen_tps": round(result.get("timings", {}).get("predicted_per_second", 0.0), 2),
        "wall_s": round(time.time() - t0, 2),
        "periodicity": periodicity(text),
        "head": text[:160],
        "tail": text[-160:],
    }


def run_chat_tools(base: str, max_tokens: int, tag: str, enable_thinking: bool) -> dict:
    body = {
        "model": "probe",
        "messages": [
            {"role": "system", "content": CHAT_SYSTEM},
            {"role": "user", "content": CHAT_USER},
        ],
        "tools": TOOLS,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    t0 = time.time()
    result = http_json(base + "/v1/chat/completions", body)
    message = result["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    # Tool-call ids are random per response; the identity check must ignore them.
    signature = ";".join(
        f"{c.get('function', {}).get('name')}({c.get('function', {}).get('arguments')})"
        for c in calls
    )
    stable = json.dumps(
        {
            "content": message.get("content") or "",
            "reasoning": message.get("reasoning_content") or "",
            "calls": [sig for sig in signature.split(";") if sig],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return {
        "probe": tag,
        "hash": digest_text(stable),
        "finish_reason": result["choices"][0].get("finish_reason"),
        "tool_signature": signature[:300],
        "content": (message.get("content") or "")[:200],
        "wall_s": round(time.time() - t0, 2),
    }


def agent_loop(base: str, context: str, turns: int, enable_thinking: bool,
               max_tokens: int, tag: str, temperature: float = 0.0) -> dict:
    """Deterministic multi-turn tool loop, the shape an agent extension drives.

    Tool results are scripted so every lane sees the same conversation; the
    per-turn assistant output is hashed. A loop shows up as a repeated tool
    signature, a divergence as a different hash at the same turn.
    """
    messages = [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": context + "\n\n" + CHAT_USER},
    ]
    turns_out = []
    signatures = []
    for turn in range(turns):
        body = {
            "model": "probe",
            "messages": messages,
            "tools": TOOLS,
            "temperature": temperature,
            "top_k": 20 if temperature > 0 else 1,
            "top_p": 0.95 if temperature > 0 else 1.0,
            "min_p": 0.05 if temperature > 0 else 0.0,
            "seed": turn,
            "max_tokens": max_tokens,
            "stream": False,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        t0 = time.time()
        result = http_json(base + "/v1/chat/completions", body)
        choice = result["choices"][0]
        message = choice["message"]
        calls = message.get("tool_calls") or []
        signature = ";".join(
            f"{c.get('function', {}).get('name')}({c.get('function', {}).get('arguments')})"
            for c in calls
        )
        stable = json.dumps(
            {
                "content": message.get("content") or "",
                "reasoning": message.get("reasoning_content") or "",
                "calls": [sig for sig in signature.split(";") if sig],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        turns_out.append({
            "turn": turn,
            "hash": digest_text(stable),
            "finish_reason": choice.get("finish_reason"),
            "signature": signature[:200],
            "call_ids": [c.get("id") for c in calls],
            "chars": len(stable),
            "wall_s": round(time.time() - t0, 2),
            "prompt_tokens": result.get("usage", {}).get("prompt_tokens"),
        })
        signatures.append(signature)
        if not calls:
            break
        # Feed the assistant turn back with deterministic call ids: the server
        # generates random ids, and echoing them would make every lane see a
        # different prompt, which would mask a real divergence.
        stable_calls = []
        for index, call in enumerate(calls):
            stable_calls.append({
                "id": f"call_{turn}_{index}",
                "type": "function",
                "function": call.get("function", {}),
            })
        messages.append({"role": "assistant", "content": message.get("content") or "",
                         "tool_calls": stable_calls})
        for call in stable_calls:
            name = call.get("function", {}).get("name")
            args = call.get("function", {}).get("arguments") or "{}"
            if name == "list_dir":
                payload = "CMakeLists.txt\narg.cpp\nchat.cpp\ncommon.cpp\nsampling.cpp\nspeculative.cpp"
            elif name == "read_file":
                payload = "// scripted tool result: file contents unavailable in probe"
            else:
                payload = "{}"
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": f"{payload}\n(args={args[:120]})"})
    repeated = 0
    if signatures:
        last = signatures[-1]
        for sig in reversed(signatures):
            if sig and sig == last:
                repeated += 1
            else:
                break
    # Longest run of one identical non-empty signature anywhere in the loop;
    # >=3 consecutive identical calls is what agent hosts treat as a loop.
    longest = 0
    run = 0
    prev = None
    for sig in signatures:
        if sig and sig == prev:
            run += 1
        else:
            run = 1
        prev = sig
        longest = max(longest, run)
    return {"probe": tag, "turns": turns_out, "repeated_tail": repeated,
            "longest_repeat": longest if signatures else 0,
            "unique_signatures": len({s for s in signatures if s})}


def spec_lines(log_path: Path) -> list[str]:
    if not log_path.is_file():
        return []
    keep = re.compile(
        r"hybrid KV cache|draft acceptance|speculative draft context|MTP draft prefill|"
        r"spec token trace|failed to trim|device handoff|sparse"
    )
    lines = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if keep.search(line):
            lines.append(line.strip())
    return lines[-14:]


def port_is_free(port: int) -> bool:
    """Guard against probing whatever happens to listen on the probe port.

    wait_ready() only polls /health, so a stale server from an earlier run would
    answer and silently invalidate a lane.
    """
    import socket

    with socket.socket() as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def start_server(args: argparse.Namespace, lane_env: dict, spec: list[str], cache: tuple | None,
                 log_path: Path) -> subprocess.Popen:
    cmd = [
        args.server_bin,
        "-m", args.model,
        "--host", "127.0.0.1", "--port", str(args.port),
        "-c", str(args.ctx),
        "--batch-size", str(args.batch), "--ubatch-size", str(args.ubatch),
        "--parallel", "1", "-ngl", "999", "--metrics",
        "-t", "5", "--threads-http", "12", "--kv-unified",
        "--cache-ram", "8192", "--ctx-checkpoints", "8",
        "--checkpoint-every-n-tokens", "4096",
        "--flash-attn", "on",
        "-dev", args.dev, "-ts", args.ts,
    ]
    if args.mmproj:
        cmd += ["--mmproj", args.mmproj]
    ck, cv = cache or (args.cache_type_k, args.cache_type_v)
    cmd += ["--cache-type-k", ck, "--cache-type-v", cv]
    cmd += spec
    cmd += list(args.extra_arg or [])

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = DEFAULT_LIBPATH + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    env.update({k: str(v) for k, v in lane_env.items()})
    log = log_path.open("wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=args.cwd)
    proc._probe_log = log  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError):
        proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        sys.stderr.write("server ignored SIGTERM; not forcing a kill\n")
    log = getattr(proc, "_probe_log", None)
    if log is not None:
        with contextlib.suppress(Exception):
            log.close()


def check_lane(args: argparse.Namespace, name: str, spec: list[str], lane_env: dict,
               cache: tuple | None, prompt: str, repo: Path) -> dict:
    if args.attach:
        base = args.attach.rstrip("/")
        record: dict = {"lane": name, "attach": base, "env": lane_env, "spec": spec}
        record["completion_a"] = run_completion(base, prompt, args.n_predict, "completion-a")
        record["completion_b"] = run_completion(base, prompt, args.n_predict, "completion-b")
        record["tools"] = run_chat_tools(base, args.tools_max_tokens, "tools", args.enable_thinking)
        if args.agent:
            record["agent"] = agent_loop(base, prompt, args.turns, args.enable_thinking,
                                         args.tools_max_tokens, "agent", args.agent_temp)
        return record

    log_path = Path(f"{args.out}-{name}.log")
    if not port_is_free(args.port):
        record: dict = {"lane": name, "error": f"port {args.port} is already in use"}
        sys.stderr.write(f"lane {name}: port {args.port} busy, refusing to run\n")
        return record
    proc = start_server(args, lane_env, spec, cache, log_path)
    base = f"http://127.0.0.1:{args.port}"
    record: dict = {"lane": name, "env": lane_env, "spec": spec, "cache": cache or
                    (args.cache_type_k, args.cache_type_v)}
    try:
        if not wait_ready(base, proc, args.startup_timeout):
            record["error"] = "server did not become ready"
            return record
        record["completion_a"] = run_completion(base, prompt, args.n_predict, "completion-a")
        record["completion_b"] = run_completion(base, prompt, args.n_predict, "completion-b")
        record["tools"] = run_chat_tools(base, args.tools_max_tokens, "tools", args.enable_thinking)
        if args.agent:
            record["agent"] = agent_loop(base, prompt, args.turns, args.enable_thinking,
                                         args.tools_max_tokens, "agent", args.agent_temp)
    finally:
        stop_server(proc)
    record["spec_log"] = spec_lines(log_path)
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-bin", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--mmproj", default=None)
    ap.add_argument("--cwd", default=None, help="working directory for the server")
    ap.add_argument("--repo", default=None, help="repository root for the prompt corpus")
    ap.add_argument("--dev", default="ROCm1,ROCm0")
    ap.add_argument("--ts", default="61,39")
    ap.add_argument("--ctx", type=int, default=151552)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--ubatch", type=int, default=1024)
    ap.add_argument("--cache-type-k", default="f8_e4m3")
    ap.add_argument("--cache-type-v", default="f8_e4m3")
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--tools-max-tokens", type=int, default=256)
    ap.add_argument("--agent", action="store_true",
                    help="also run the deterministic multi-turn tool loop")
    ap.add_argument("--turn-log", action="store_true",
                    help="print every agent turn (id, args, text) for both lanes")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--agent-temp", type=float, default=0.0,
                    help="temperature for the agent loop; 0 keeps it deterministic")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="send enable_thinking=true on the tools probe")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--startup-timeout", type=float, default=420.0)
    ap.add_argument("--out", default="/tmp/mtp_identity")
    ap.add_argument("--extra-arg", action="append", default=None,
                    help="extra server argument; repeat for several (e.g. --extra-arg '--flash-attn off')")
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--lane", default=None, help="run one lane by name")
    ap.add_argument("--lanes", default=None,
                    help="comma separated lane names, e.g. none,mtp")
    ap.add_argument("--attach", default=None,
                    help="probe an already running server (URL) instead of starting one")
    args = ap.parse_args()

    if args.attach:
        if args.lane is None:
            args.lane = "none"
    elif not args.server_bin or not args.model:
        ap.error("--server-bin and --model are required unless --attach is used")

    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[2]
    if args.server_bin:
        args.cwd = args.cwd or str(Path(args.server_bin).resolve().parent)
    prompt = corpus(repo, args.prompt_tokens) + COMPLETION_SUFFIX
    prompt_hash = digest_text(prompt)
    sys.stderr.write(f"prompt chars={len(prompt)} hash={prompt_hash}\n")

    if args.lane:
        lanes = [lane for lane in LANES if lane[0] == args.lane]
        if not lanes:
            sys.stderr.write(f"unknown lane {args.lane}\n")
            return 2
    elif args.lanes:
        wanted = [name.strip() for name in args.lanes.split(",") if name.strip()]
        lanes = []
        seen: dict[str, int] = {}
        missing = []
        for name in wanted:
            match = next((lane for lane in LANES if lane[0] == name), None)
            if match is None:
                missing.append(name)
                continue
            seen[name] = seen.get(name, 0) + 1
            label = name if seen[name] == 1 else f"{name}#{seen[name]}"
            lanes.append((label, match[1], match[2], match[3]))
        if missing:
            sys.stderr.write(f"unknown lanes: {', '.join(missing)}\n")
            return 2
    elif args.ladder:
        lanes = list(LANES)
    else:
        lanes = [LANES[1]]

    results = []
    for name, spec, env, cache in lanes:
        sys.stderr.write(f"== lane {name} ==\n")
        results.append(check_lane(args, name, spec, env, cache, prompt, repo))

    report = {
        "corpus": corpus_hash(repo),
        "prompt_chars": len(prompt),
        "prompt_hash": prompt_hash,
        "results": results,
    }
    Path(args.out + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Repeat lanes (none,none,mtp,mtp) are the null test: identical labels
    # after the first occurrence must produce identical hashes or the lane
    # itself is not reproducible across server processes.
    control = next((r for r in results if r["lane"].split("#")[0] == "none"), None)
    summary = {}
    for r in results:
        entry = {
            "completion_a": r.get("completion_a", {}).get("hash"),
            "completion_b": r.get("completion_b", {}).get("hash"),
            "tools": r.get("tools", {}).get("hash"),
            "tool_signature": r.get("tools", {}).get("tool_signature"),
            "period_a": r.get("completion_a", {}).get("periodicity"),
            "gen_tps_a": r.get("completion_a", {}).get("gen_tps"),
            "prompt_tps_a": r.get("completion_a", {}).get("prompt_tps"),
            "error": r.get("error"),
        }
        if r.get("agent"):
            entry["agent_turns"] = [t["hash"] for t in r["agent"]["turns"]]
            entry["agent_repeated_tail"] = r["agent"]["repeated_tail"]
            entry["agent_longest_repeat"] = r["agent"]["longest_repeat"]
            entry["agent_unique_signatures"] = r["agent"]["unique_signatures"]
        summary[r["lane"]] = entry
    print(json.dumps(summary, indent=2))

    if control is not None:
        print("\nverdict vs lane 'none':")
        for r in results:
            if r is control or r.get("error"):
                continue
            same = (
                r["completion_a"]["hash"] == control["completion_a"]["hash"]
                and r["completion_b"]["hash"] == control["completion_b"]["hash"]
                and r["tools"]["hash"] == control["tools"]["hash"]
            )
            print(f"  {r['lane']:14s} {'IDENTICAL' if same else 'DIVERGED'}")
            if not same:
                for probe in ("completion_a", "completion_b", "tools"):
                    if r[probe]["hash"] != control[probe]["hash"]:
                        print(f"    {probe}: lane={r[probe]['hash']} control={control[probe]['hash']}")
                        if probe.startswith("completion"):
                            print(f"      lane head: {r[probe]['head']!r}")
                            print(f"      ctrl head: {control[probe]['head']!r}")
                        else:
                            print(f"      lane call: {r[probe]['tool_signature']!r}")
                            print(f"      ctrl call: {control[probe]['tool_signature']!r}")
            if r.get("agent") and control.get("agent"):
                lane_turns = [t["hash"] for t in r["agent"]["turns"]]
                ctrl_turns = [t["hash"] for t in control["agent"]["turns"]]
                first = next((i for i, (a, b) in enumerate(zip(lane_turns, ctrl_turns)) if a != b), None)
                if len(lane_turns) != len(ctrl_turns) or first is not None:
                    print(f"    agent loop: lane turns={len(lane_turns)} control turns={len(ctrl_turns)}"
                          + (f" first divergent turn={first}" if first is not None else ""))
                    limit = len(lane_turns) if args.turn_log else 8
                    for t in r["agent"]["turns"][:limit]:
                        print(f"      lane turn {t['turn']}: {t['signature'][:90]!r} hash={t['hash']}")
                    for t in control["agent"]["turns"][:limit]:
                        print(f"      ctrl turn {t['turn']}: {t['signature'][:90]!r} hash={t['hash']}")
                    if args.turn_log:
                        for t in r["agent"]["turns"][:limit]:
                            print(f"      lane ids {t['turn']}: {t['call_ids']} tokens={t.get('prompt_tokens')}")
                        for t in control["agent"]["turns"][:limit]:
                            print(f"      ctrl ids {t['turn']}: {t['call_ids']} tokens={t.get('prompt_tokens')}")
                else:
                    print(f"    agent loop: IDENTICAL over {len(lane_turns)} turns "
                          f"(repeated tail={r['agent']['repeated_tail']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
