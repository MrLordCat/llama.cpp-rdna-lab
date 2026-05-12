# E006: Full Retest + Scientific Audit of Research Track

Date: 2026-05-12
Owner: Copilot
Stage: reproducibility retest + methodology audit

## Objective

1. Перепроверить все завершенные исследования E001-E005.
2. Дать честную оценку научной корректности текущего процесса.
3. Проверить, применимы ли гипотезы к реальному коду репозитория.

## Retest Scope

Retest выполнен как воспроизведение всех зафиксированных research-команд на тех же артефактах (CSV/server.log) с сохранением свежих результатов в:

- build_logs/agent-workload/retests-20260512/

Это ретест воспроизводимости выводов из артефактов, а не новый долгий lane-бенч с генерацией новых CSV.

## Retest Results

| Experiment | Status | Retest Result |
| --- | --- | --- |
| E001 | PASS | Formula sanity OK; required acceptance и speedup-оценки совпали (например D=24: 0.0129 для 1.20x, 0.0323 для 1.30x; projected TPS 16.2255) |
| E002 | PASS, historical boundary only | Cliff class подтвержден на старой паре 824/832: 10.1707 -> 3.5955 TPS (0.3535x), prompt_eval 0.2697x, decode_eval 0.7456x. Current action target superseded to ubatch 480/490. |
| E003 | PASS | Case A: gen_drafts=0 (прирост не speculative-driven); Case B: observed 1.1006x, local=0.4167, coverage=0.01435, effective=0.00598 |
| E004 | PASS | Coverage-aware лучше naive в sparse-draft кейсе (ошибка 0.012662 vs 0.316176), control-кейс ведет себя ожидаемо |
| E005 | PASS | Batch-проверка совпала: coverage-aware wins 5/6, tie 1/6; large implied MTP overhead preserved |

## Scientific Method Audit

### What is done correctly

1. Есть явный hypothesis backlog и последовательный цикл "модель -> измерение -> cross-check".
2. Используются control-like кейсы (например no-draft case в E004).
3. Разделяются local acceptance, coverage, effective acceptance, что устраняет типичную интерпретационную ошибку.
4. Результаты фиксируются с командами и артефактами, что поддерживает воспроизводимость.

### Main scientific risks (current)

1. Низкая статистическая мощность для части claims: многие сравнения построены на 2 task rows в CSV.
2. Нет системного интервала неопределенности (CI/bootstrapping) в experiment notes.
3. Часть аналитики использует post-hoc backsolve (например implied acceptance/overhead), это диагностично, но не каузальное доказательство.
4. Cross-mode обобщение пока ограничено: ngram и MTP показывают разную динамику overhead.

### Verdict on scientific quality

Текущий процесс **научно корректен как engineering R&D pipeline**, но для уровня "строгого доказательного бенч-исследования" нужно усилить статистику и явно фиксировать uncertainty.

## Hypothesis Recheck (as of E006)

| Hypothesis | Current Evidence | Scientific Status |
| --- | --- | --- |
| H01 adaptive ngram length | measured evidence отсутствует | plausible, untested |
| H02 dynamic draft length | E001 analytic gate positive | plausible, not yet measured |
| H03 ngram/mtp router | measured evidence отсутствует | plausible, untested |
| H04 early reject verify | measured evidence отсутствует | plausible, untested |
| H05 flash-attn tile retarget | measured evidence в этом цикле нет | plausible, pending kernel-level tests |
| H06 QKV/RoPE fusion | measured evidence отсутствует | plausible, untested |
| H07 KV layout locality | measured evidence отсутствует | plausible, untested |
| H08 chunk contract alignment | E002 strong measured cliff support; current autotune shows best at ubatch 480 and failure at 490+ | supported as boundary/cliff class, current target 480/490 |
| H09 coverage-aware acceptance | E004+E005 reproduce better fit | supported for low-coverage speculative modeling |
| H10 overhead-aware model | E005 reveals large MTP implied overhead | supported as next modeling need |

## Applicability To Code (Theory -> Implementation Paths)

Hypotheses are applicable to existing code paths:

1. Speculative policy/metrics layer:
   - common/speculative.cpp
   - common/speculative.h
   - common/ngram-mod.cpp

2. MTP integration points:
   - src/llama-context.cpp

3. UBatch and chunk-contract path (H08):
   - src/llama-context.cpp
   - ggml/src/ggml-cuda/gated_delta_net.cu

4. Kernel/perf paths (H05/H06/H07 family):
   - ggml/src/ggml-cuda/fattn.cu
   - ggml/src/ggml-cuda/mmq.cuh

Conclusion: теоретические гипотезы имеют реальные точки внедрения в runtime/kernel коде; research трек применим к коду практическим образом.

## Recommended Next Step

Для повышения научной строгости перед новыми speed claims:

1. Для каждого high-impact case выполнить минимум 3 cold reruns.
2. Добавить CI/dispersion блок в experiment notes.
3. Для H10 построить mode-aware overhead model и проверить на расширенном наборе кейсов.
