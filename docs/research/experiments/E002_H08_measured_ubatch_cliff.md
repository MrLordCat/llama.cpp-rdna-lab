# E002: H08 Boundary-Aware Chunk Contract - Measured UBatch Cliff

Date: 2026-05-12
Owner: Copilot
Stage: Measured analysis on existing artifacts

## Objective

Подтвердить количественно, что boundary/cliff на ubatch не является шумом, а устойчивым режимным срывом на active lane.

## Compared Artifacts

- Baseline: `gdn-tail64-scan-ub824-r1`
- Candidate: `gdn-tail64-trace-ub832-r2`
- Lane: prompt-heavy, ctx=12288, b=6144, spec=none, q4_0/q4_0

## Tool Command

```bash
python scripts/research/bench_pair_compare.py \
  --baseline-name ub824 \
  --baseline-csv build_logs/agent-workload/gdn-tail64-scan-ub824-r1.csv \
  --baseline-log build_logs/agent-workload/gdn-tail64-scan-ub824-r1.server.log \
  --candidate-name ub832 \
  --candidate-csv build_logs/agent-workload/gdn-tail64-trace-ub832-r2.csv \
  --candidate-log build_logs/agent-workload/gdn-tail64-trace-ub832-r2.server.log
```

## Measured Results

- aggregate TPS: 10.1707 -> 3.5955 (speedup 0.3535x)
- delta: -6.5752 TPS
- prompt_eval_tps mean: 1076.1850 -> 290.3000 (0.2697x)
- decode_eval_tps mean: 27.7300 -> 20.6750 (0.7456x)

## Interpretation

1. Срыв почти полностью prefill-доминированный (prompt_eval падает сильнее, чем decode).
2. Это подтверждает только класс проблемы H08 (опасные ubatch/chunk boundaries), но не фиксирует актуальную границу.
3. Наблюдение устойчиво и слишком велико для объяснения обычным run-to-run шумом, однако конкретная пара 824/832 больше не должна считаться текущим target.

## Current Correction

Date: 2026-05-12

The current autotune evidence supersedes the old 824/832 action target. On the current best-search lane with ctx=32768, q4_0/q4_0, spec=ngram-mod, repo-snapshot prompt size around 6971 tokens:

| Artifact | batch | ubatch | TPS | Errors |
| --- | ---: | ---: | ---: | ---: |
| `gui-autotune-Qwen3.6-27B-Q3_K_S-20260512-072210-cfg01` | 2048 | 448 | 9.7253 | 0 |
| `gui-autotune-Qwen3.6-27B-Q3_K_S-20260512-072210-cfg02` | 2048 | 464 | 9.8614 | 0 |
| `gui-autotune-Qwen3.6-27B-Q3_K_S-20260512-072210-cfg03` | 2048 | 480 | 9.9656 | 0 |
| `gui-autotune-Qwen3.6-27B-Q3_K_S-20260512-072210-cfg04` | 2048 | 496 | 0.0000 | 1 |
| `gui-autotune-Qwen3.6-27B-Q3_K_S-20260512-073021-cfg01` | 2560 | 480 | 9.9751 | 0 |
| `gui-autotune-Qwen3.6-27B-Q3_K_S-20260512-073021-cfg02` | 2560 | 485 | 9.9616 | 0 |
| `gui-autotune-Qwen3.6-27B-Q3_K_S-20260512-073021-cfg03` | 2560 | 490 | 0.0000 | 1 |
| `gui-autotune-Qwen3.6-27B-Q3_K_S-20260512-073021-cfg04` | 2560 | 495 | 0.0000 | 1 |

Interpretation of the correction:

- Current maximum is `ubatch=480`; `ubatch=485` is already slightly lower, and `ubatch>=490` fails the benchmark in the current artifacts.
- The old 824/832 pair should be treated as historical evidence that cliffs exist, not as the current optimization boundary.
- Future H08 work must target the current 480/490 neighborhood and compare against the current autotune/history best, not against the old 824/832 lane.

## Decision

Decision: Keep H08 as a boundary/cliff class, but supersede the 824/832 target with the current 480/490 boundary.
Confidence: High for the existence of ubatch cliffs; medium for mechanism until the 480/490 failure is instrumented directly.

## Next Step

- Для H08 проводить изменения только с проверкой current best boundary pairs: 480/485/490/495 (и adjacent values при необходимости).
- При каждой итерации фиксировать prompt_eval_tps отдельно от wall TPS.
