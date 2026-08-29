# BENCH 2.0 — Plan (agreed 2026-08-28)

## 1. Goal

A universal benchmark tool independent of backend (ROCm / Vulkan / CPU / RPC),
environment and goal. Two scenario types:
1. **Single level** — one big prompt + decode (tokens).
2. **Agent session** — 10 turns in one growing context (KV accumulates),
   decode degradation with context size is visible.

The run level is a parameter (`--level 0..5` / `--session-level 1..3`), no
separate `suite/single` modes: one level, a list, or a range.

## 2. Single levels

| Level | Context | Prompt (tokens) | Decode (tokens) | Time estimate |
|---|---|---|---|---|
| L0 smoke | 8K    | ~4K  | 64  | ~15 s |
| L1 test  | 16K   | ~8K  | 128 | ~40 s |
| L2       | 49K   | ~31K | 256 | ~3 min |
| L3       | 98K   | ~65K | 256 | ~7 min |
| L4       | 131K  | ~95K | 256 | ~10-15 min |
| L5       | 200K  | ~190K| 256 | 20+ min (rare, may not fit in 2x16GB — pre-check) |

Defaults (hardware profile, overridable):
- batch **8192** / ubatch **1024** (best configuration on this rig);
- `-dev ROCm1,ROCm0 -sm layer -ts 1,1` — ROCm; `-dev Vulkan1,Vulkan0 -sm layer -ts 1,1` — Vulkan;
- KV q8_0/q8_0, flash-attn, `-fit off`, no-warmup, `-c 0`, fixed seed.

## 3. Agent sessions (10 turns)

| Session | Context | Input/turn | Decode/turn | Load |
|---|---|---|---|---|
| SL1 light  | 32K  | ~1K tok | 128 | de-anon/test |
| SL2 medium | 98K  | ~2K tok | 256 | standard |
| SL3 heavy  | 131K (opt. 200K) | ~4K tok | 512 | max + MTP |

Each turn: a short "agent" request, KV grows; context persists between turns
(one session = consecutive answers on one context).

## 4. Metrics

Single level:
- prefill TPS, decode TPS, TTFT (ms), prefill_ms, total_ms, aggregate TPS;
- for MTP: acceptance_ratio, draft_tps, effective_decode_tps.

Session:
- per-turn: TTFT, decode TPS, context length, input tokens, decode tokens, wall ms;
- averages/min/max over turns, decode TPS degradation slope (turn 1 -> 10),
  session aggregate TPS.

## 5. Data formats

**Run directory** — `build_logs/bench/<RUN_NAME>/`:

| File | Format | Contents |
|---|---|---|
| `run.json` | JSON | full effective config: run_name, type, level, timestamp, backend, binary/commit, model, ctx/batch/ubatch, KV, spec, env (redacted), server flags, seed |
| `<RUN_NAME>.jsonl` | JSONL | line events: server start/ready, prefill (start/end, tokens, tps), decode (tps, time), final summary record; sessions — one event per turn |
| `metrics.csv` | CSV | one row = one scenario (level or session): run_name, type, level, backend, model, ctx, prompt_tokens, decoded_tokens, prefill_tps, decode_tps, ttft_ms, total_ms, aggregate_tps, mtp_acc, eff_decode_tps, session_turns, path |
| `summary.md` | Markdown | human-readable report: metrics table + config + conclusions |
| `server.log` | text | raw server log for diagnostics |
| `artifacts/` | opt. | `responses.jsonl`, `timing*.jsonl` |

**Session** — same directory plus:
- `session_turns.csv` — one row per turn: turn, ctx_len, input_tokens, decode_tokens, ttft_ms, decode_tps, wall_ms;
- `metrics.csv` — session aggregates (aggregate, avg decode tps, slope),
  `summary.md` — turns table.

**Global indexes** (search by run name):
- `build_logs/bench/index.csv` — one row per run: run_name, type, level, timestamp,
  backend, model, key metrics, path;
- `build_logs/bench/index.md` — same summary for reading.

Search: `bench2 find --name <pattern> --type session --level 3 --backend rocm`.

The legacy benchmark archive is kept AS-IS (BENCH_HISTORY.*, BENCH_RUNS.*,
BENCH_*.md, per-run jsonl/csv) — no conversion, to avoid data loss.

## 6. CLI (sketch)

```text
python scripts/bench2.py run --run-name q38-rocm-l2-a --level 2
python scripts/bench2.py run --run-name d094-vk-session-sl2 --session-level 2
python scripts/bench2.py run --run-name recheck-r3 --level 0,2 --runs 3
python scripts/bench2.py run --run-name smoke --level 0-1 --backend vk
python scripts/bench2.py find --name "l2" --type single --backend rocm
python scripts/bench2.py list --recent 20
```

Common options:
- `--server-bin <path>` (start + graceful teardown) or `--attach <url>`;
- `--model <path>`, `--backend auto|rocm|vk|cpu`, `--profile <hardware-name>`;
- `--level 0..5` / `--session-level 1..3`, `--runs N`;
- `--context-source synthetic|repo-snapshot|file:<path>`;
- `--batch-size/--ubatch-size/--kv/--spec/--flash-attn/--gpu-layers/--parallel/--dev ...` — overrides;
- `--results-dir` (default `build_logs/bench`), `--run-name` (required for run);
- GPU-free precheck, TERM clean, driver-rule enforcement (`-fit off`, no `hipMemGetInfo`).

## 7. Configs

JSON in `configs/bench/`:
- `hardware.profiles.json` — hardware presets (batch/ubatch/dev/sm/ts/ctx limits);
- `levels.json` — level table (ctx, prompt_tokens, decode_tokens);
- `sessions.json` — session presets (turns, input/decode, ctx);
- `server.defaults.json` — common defaults (KV, flash-attn, fit off, seed, no-warmup).

CLI overrides take precedence over configs.

## 8. Legacy bench archive

- Move `build_logs/agent-workload/*` -> `build_logs/archive/agent-workload-legacy-2026-08/`
  (+ archive README with date and purpose).
- Old `scripts/agent_workload_bench.py` -> `scripts/legacy/`; keep a thin
  compatibility wrapper at the old path (GUI/`tool_call_workload_bench.py`/
  `large_context_*` must not break in this step; GUI switch — later, separate decision).
- Dependent scripts: fix imports; do not change old JSONL/CSV formats.

## 9. Implementation steps

1. [x] Configs `configs/bench/*.json` + `scripts/bench2.py` skeleton (CLI, run-name, indexes).
2. [x] Synthetic context generator (deterministic seed) for L0-L5 + `repo-snapshot`/`file`.
3. [x] Single scenario: server start/attach, HTTPS/live log, metrics, run.json/jsonl/csv/md output.
4. [x] Session scenario: N turns, growing KV, per-turn metrics, degradation slope.
5. [x] `find/list` over index.csv + human-readable summaries.
6. [x] Legacy data archive + legacy wrapper + import fixes.
7. [x] Smoke validation: L0 (ROCm) ok, L0 (Vulkan) ok, SL1 ok (2026-08-28);
      agreed full fork-vs-stock L0-L4 runs done (2026-08-28).
8. [x] Documentation: `docs/local/BENCH2.md` (+ memory update).

## 10. Open questions (resolve before/during implementation)

- [ ] L5 (200K ctx + ~190K prompt tokens) on 2x16GB: pre-check VRAM/load;
      otherwise L5 = "experimental, with warning".
- [ ] MTP: acceptance metrics from `draft_n/accepted` — one format for ROCm/VK.
- [x] Synthetic prompt composition (otherwise L2/L3 not comparable with old
      repo-snapshot results) — resolution: synthetic by default for all levels,
      L2 can use `repo-snapshot` via `--context-source`.
- [ ] Cold/adjacent control: A/B benchmarks always interleaved (thermal drift
      rule), record "after N run session" in summary.
- [x] GPU-free precheck (tasklist + Get-Counter) and rejection when a server is
      active — implemented (tasklist + preflight).
