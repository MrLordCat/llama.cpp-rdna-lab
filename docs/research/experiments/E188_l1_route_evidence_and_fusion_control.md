# E188 L1 Route Evidence and Fusion Negative Control

## Metadata

- Experiment ID: E188
- Date: 2026-05-23
- Owner: Copilot
- Target lane: L1 ROCm (`ctx=12288,b=6144,ub=2048,q4/q4,spec=none,no-reuse`)

## Goal

До первого stack-test зафиксировать route evidence на текущем цикле и проверить quick negative control для fusion без изменения кода.

## Runs

1. `e187-l1-baseline-synctrace-r1`
   - env: `GGML_TRACE_MMVQ_TIMING=1`, `GGML_TRACE_MMVQ_TIMING_SYNC=1`
   - preset: `--trace-preset kernel-full`

2. `e187-l1-disablefusion-r1`
   - env: `GGML_CUDA_DISABLE_FUSION=1`

3. `e188-l1-disablefusion-r3`
   - same env/contract, `runs=3`

## Key Route Evidence

Из synctrace server log:

- Активный Q3_K decode path присутствует как `route=qwen-hot`, `ncols_dst=1`, `small_k=1`, `nwarps=2`.
- Видны и direct, и fused Q3_K участки (`fusion=0/1`), включая формы `ncols_x=5120` и `ncols_x=17408`.
- Fused FFN route активен на поздних слоях (`ffn_gate/ffn_out` с `q3_K`).

Диагностическая оговорка:

- при `*_TIMING_SYNC=1` trace-run замедляется (aggregate `10.76 TPS`), поэтому это route evidence, не speed baseline.

## Fusion negative control

- baseline r1 (`e187-l1-baseline-r1`): aggregate `11.8389`, decode `30.50 tok/s`
- disable-fusion r1 (`e187-l1-disablefusion-r1`): aggregate `11.9276`, decode `30.08 tok/s`
- baseline r3 (`e187-l1-baseline-r3`): aggregate `12.4256`, median `12.59`, decode mean `30.49 tok/s`
- disable-fusion r3 (`e188-l1-disablefusion-r3`): aggregate `12.1909`, median `12.37`, decode mean `30.0167 tok/s`

Интерпретация:

- r1 был шумовым и не должен был использоваться как route claim,
- парный r3 показал устойчивый регресс против baseline (`-1.89%` aggregate, хуже decode mean),
- fusion-path в текущем L1 цикле считать полезным, а disable-fusion использовать только как negative control.

## Decision

- Keep as gate evidence with resolved verdict: disable-fusion is negative on paired `r3`.
- Для любых fusion-related выводов в этом цикле использовать только `r3`-пары.
- Следующий шаг: guided stack-test candidate с обязательной схемой `baseline r3 -> resource/timing trace -> candidate r3 -> post-check`.

## Artifacts

- `build_logs/agent-workload/e187-l1-baseline-synctrace-r1.server.log`
- `build_logs/agent-workload/e187-l1-baseline-synctrace-r1.diagnostics.md`
- `build_logs/agent-workload/e187-l1-disablefusion-r1.diagnostics.md`
- `build_logs/agent-workload/e188-l1-disablefusion-r3.diagnostics.md`
