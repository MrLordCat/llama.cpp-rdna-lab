# E187 Route-Chain L1 Baseline Lock

## Metadata

- Experiment ID: E187
- Date: 2026-05-23
- Owner: Copilot
- Branch/Commit: master (post-E186 docs audit)
- Target lane: L1 ROCm H39 active route-chain baseline

## Purpose

Зафиксировать clean baseline для guided route-chain цикла, чтобы все следующие stack A/B сравнивались с одной точкой и без env-паразитов.

## Lane Contract

- backend: `build-rocm-vec/bin/llama-server.exe`
- model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- `ctx=12288`, `batch=6144`, `ubatch=2048`
- KV: `q4_0/q4_0`
- `spec=none`, `--cache-ram 0 --ctx-checkpoints 0`
- tasks: `quick/triage_diff`
- `--max-tokens 128`
- no reuse, no v2-prime, thinking enabled

## Method

1. Readiness snapshot + verify no background `llama-server`.
2. Run baseline `r1` in clean env (unset trace/selector overrides).
3. Run baseline `r3` in clean env.
4. Run resource snapshot with `GGML_TRACE_MMVQ_RESOURCES=1`.

## Results

### Runtime

- `e187-l1-baseline-r1`: aggregate `11.8389 TPS`
- `e187-l1-baseline-r3`: aggregate `12.4256 TPS`, median `12.59`, stdev `0.2612`
- `e187-l1-baseline-resources-r1`: aggregate `12.07 TPS`

### Server summary (r3)

- prompt eval TPS mean: `1222.9433`
- decode eval TPS mean: `30.49`
- prompt eval ms mean: `6069.3967`
- decode eval ms mean: `4197.9733`
- prompt tokens mean: `7413`

Interpretation: на этом контракте decode остается существенной частью wall-time; route-chain для H39 должен фокусироваться на MMVQ decode path, не на broad runtime toggles.

### Resource snapshot highlights

Из `GGML_TRACE_MMVQ_RESOURCES`:

- `type=11/q3_K` (типичные записи в run): `regs=70`, `occupancy_pct=100.00`, `block=(32,1,1)`
- `type=12/q4_K`: `regs=42`, `occupancy_pct=100.00`, `block=(32,1,1)`

Это baseline ресурсный отпечаток для сравнения следующих кандидатов. Любой candidate, который ухудшает доминирующий bucket при росте регистров/падении occupancy, должен отбрасываться до r3.

## Decision

- Keep baseline lock for subsequent stack tests.
- Use `e187-l1-baseline-r3` as control reference for ближайший route-chain candidate.

## Artifacts

- `build_logs/agent-workload/e187-l1-baseline-r1.diagnostics.md`
- `build_logs/agent-workload/e187-l1-baseline-r3.diagnostics.md`
- `build_logs/agent-workload/e187-l1-baseline-resources-r1.diagnostics.md`
- `build_logs/agent-workload/e187-l1-baseline-resources-r1.server.log`
- `build_logs/agent-workload/e187-l1-baseline-r3.server.log`
