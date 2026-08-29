# bench2 — universal benchmark tool (v2, 2026-08-28)

A single CLI tool for any backend (ROCm / Vulkan / CPU / RPC), any environment
and any goal. It replaces **v1** (`scripts/agent_workload_bench.py`, archived in
`build_logs/archive/agent-workload-legacy-2026-08/`).

## Quick start

```bash
# Three commands cover 95% of use cases:
python scripts/bench2.py run --level 1              # L1 (16K), auto binary/model/name, live log
python scripts/bench2.py run --level 2 --runs 3     # L2 (49K) three times, auto name search
python scripts/bench2.py run --session-level 2      # agent session SL2 (98K, 10 turns)

# Search/list:
python scripts/bench2.py find --name recheck --type single
python scripts/bench2.py list --recent 10
```

Levels accept lists and ranges: `--level 0,2`, `--level 1-3`.
Run name is auto-generated when `--run-name` is omitted, e.g.
`rocm-l2-20260828-0934`.

---

## 1. Scenarios

### Single levels (`--level N`): one big prompt + decode

| Level | ctx | Prompt (tokens) | Decode | Purpose |
|---|---|---|---|---|
| L0 smoke | 8K | ~4K | 64 | instant launch check |
| L1 test | 16K | ~8K | 128 | short test runs |
| L2 | 49K | ~31.7K | 256 | standard working lane |
| L3 | 98K | ~66K | 256 | large |
| L4 | 131K | ~97K | 256 | max for 2x16GB |
| L5 | 200K | ~194K | 256 | rare, may not fit — pre-check |

### Agent sessions (`--session-level N`): 10 turns, growing KV

| Session | ctx | Input/turn | Decode/turn | Purpose |
|---|---|---|---|---|
| SL1 light | 32K | ~1K | 128 | de-anon/test |
| SL2 medium | 98K | ~2K | 256 | standard |
| SL3 heavy | 131K | ~4K | 512 | heavy + MTP |

Each turn is a new agent request; context is preserved (`cache_prompt`), KV is
reused; per-turn metrics + decode degradation slope (tok/s per turn).

## 2. What is automated

- Server lookup: `--server-bin` or **auto-search** by backend
  (`build-rocm|build-vulkan|build-cpu/bin/llama-server.exe`);
  runtime PATH is prepended automatically (ROCm 7.1 bin / Strawberry MinGW).
- Model lookup: `--model` or **auto** (falls back to
  `models/Qwen3.8-27B-Q4_K_M.gguf`, otherwise first `models/*.gguf`).
- Run name (if not set), free port, server start/graceful stop, preflight:
  rejects if a live `llama-server.exe` exists.
- **Live log by default**: prefill progress (parsed from `server.log`) and
  every level/turn result printed to console.

## 3. Data

Run directory `build_logs/bench/<RUN_NAME>/`:

| File | Contents |
|---|---|
| `run.json` | effective config (backend, commit, server flags, seed, redacted env) |
| `<RUN_NAME>.jsonl` | line events (server, level/turn start/done, summary) |
| `metrics.csv` | one row per measurement (single level or session) |
| `summary.md` | human-readable table |
| `session_turns.csv` | sessions only: one row per turn (turn, ctx, prompt, cache_n, decode, tps, wall) |
| `server.log` | raw server log |

Global indexes: `build_logs/bench/index.csv` (one row per run) and `index.md`.

### Key metrics

- `prefill_tps` — of **new** prefill tokens per second (for KV-reuse sessions
  `cache_n` is subtracted);
- `decode_tps`, `ttft_ms` (prefill latencies), `total_ms`, `aggregate_tps`;
- sessions: `decode_slope` (decode degradation over 10 turns), `session_turns`.

## 4. CLI options (run)

```text
--run-name NAME        name (auto if empty)
--level 0|1|2|3|4|5    single levels (list/range), default 1
--session-level 1|2|3  agent sessions
--runs N               repeats per scenario
--server-bin PATH  |  --attach http://host:port
--backend auto|rocm|vk|cpu   (auto from binary name)
--model PATH, --profile NAME
--context-source synthetic|repo-snapshot|file  --context-file PATH
--batch-size 8192 --ubatch-size 1024   (rdna-lab profile defaults)
--kv-k q8_0 --kv-v q8_0 --spec none|mtp --spec-n 2
--flash-attn/--no-flash-attn, --gpu-layers, --parallel
--dev, --sm, --ts, --fit, --seed, --temperature, --top-p
--server-extra "raw args", --api-extra '{"chat_format": 0}', --results-dir,
--health-timeout, --fail-fast
```

## 5. Configs (`configs/bench/`)

| File | Contents |
|---|---|
| `hardware.profiles.json` | hardware profiles: `default_batch 8192`, `default_ubatch 1024`, `dev/sm/ts` for `rocm/vk/cpu` |
| `levels.json` | level table + context sources |
| `sessions.json` | agent session presets |
| `server.defaults.json` | llama-server defaults (KV q8_0, FA on, `fit off`, `cache_ram 0`, `ctx_checkpoints 0`, seed 42) |

CLI flags always override configs; `--profile generic` disables hardware specifics.

## 6. Important rules/limitations

- Uses **`/v1/chat/completions`**, not raw `/completion`: Qwen thinking models
  without chat tags emit `|im_end|>` as the first token and the server stops
  (diagnosed 2026-08-28).
- `--cache-ram 0 --ctx-checkpoints 0` are on by default for cold measurements.
- L2/L3+ use a synthetic context; `repo-snapshot` is capped at ~52K tokens and
  is only useful up to L2.
- Preflight rejects an active `llama-server.exe`.
- Driver rules (AGENTS.md): `-fit off`, no `hipMemGetInfo`, graceful stop.

## 7. Status (2026-08-28)

- [x] framework + configs + CLI + live log;
- [x] single L0/L1 tested (ROCm): L0 3995 tok -> 64 decode (23.9 tps, prefill 1387), L1 7901 -> 128 (23.9 tps, prefill 1418);
- [x] session SL1 (10 turns, KV-reuse): avg decode 24.1 tps, slope -0.097;
- [x] v1 archive (14 GB) in `build_logs/archive/agent-workload-legacy-2026-08/`;
- [x] v1 wrapper `scripts/agent_workload_bench.py` -> legacy, tests 6/6;
- [x] full fork-vs-stock L0-L4 on Vulkan and ROCm (2026-08-28,
      see `docs/research/D136_STOCK_VULKAN_DECODE_EDGE.md`);
- [ ] GUI switch to bench2 (per user: "later").
