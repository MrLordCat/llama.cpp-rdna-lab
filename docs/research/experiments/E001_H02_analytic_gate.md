# E001: H02 Dynamic Draft Length - Analytic Gate

Date: 2026-05-12
Owner: Copilot
Stage: Analytic gate (before code patch)

## Target

Проверить, есть ли математический коридор для прироста wall TPS при динамической длине draft (без изменения качества и без глубоких kernel-правок).

## Baseline

Согласно активному lane в BENCHMARKS:

- ctx=12288, prompt-heavy repo-snapshot, no-reuse
- baseline aggregate TPS ~= 9.85

## Assumptions For Analytic Model

- prefill_share p = 0.70
- speculative overhead o = 0.08
- prefill speedup S_prefill = 1.20 (умеренный сценарий)
- decode kernel speedup S_decode = 1.00 (без доп. kernel-оптимизации)
- draft length D = 24 (база для оценки)

## Executed Tools

1. Formula sanity checks:

```bash
python scripts/research/formula_sanity_checks.py --samples 3000 --seed 9070
```

Output: `OK: formula sanity checks passed (samples=3000, seed=9070)`

1. Required acceptance corridor:

```bash
python scripts/research/required_acceptance.py \
  --target-wall 1.10,1.20,1.30 \
  --draft-len 16,24,32,48 \
  --prefill-share 0.70 \
  --prefill-speedup 1.20 \
  --decode-kernel-speedup 1.00 \
  --spec-overhead 0.08
```

Key rows:

- target 1.20x: required acceptance for D=24 is 0.0129
- target 1.30x: required acceptance for D=24 is 0.0323

1. Moderate scenario estimate:

```bash
python scripts/research/speedup_model.py \
  --baseline-tps 9.85 \
  --prefill-share 0.70 \
  --draft-len 24 \
  --accept-rate 0.55 \
  --spec-overhead 0.08 \
  --flash-prefill-speedup 1.20 \
  --decode-kernel-speedup 1.00 \
  --sweep-accept 0.35,0.45,0.55,0.65 \
  --sweep-flash 1.10,1.20,1.30
```

Main estimate:

- combined wall speedup ~= 1.6473x
- projected TPS ~= 16.2255

## Interpretation

- Аналитический барьер для H02 открыт: даже умеренные параметры дают заметный потенциал.
- Чисто математически идея правдоподобна, но это не benchmark proof.
- Следующий шаг: microbench + lane benchmark с жёсткой проверкой стабильности.

## Decision

Decision: Proceed to microbench implementation planning.
Confidence: Medium (analytic only).

## Next Step Draft

- Introduce simple adaptive policy for draft length based on rolling acceptance proxy.
- Keep hard caps and hysteresis to avoid oscillation.
- Compare against static D baseline on prompt-heavy ctx=12288 lane.

## Runtime Prototype Check

Date: 2026-05-12
Branch/Commit: 96decb6af + local guarded prototype

Implementation:

- Added a guarded `LLAMA_NGRAM_MOD_DYNAMIC_DRAFT=1` prototype in `common/speculative.cpp`.
- The prototype adapted `ngram-mod` effective `n_max` from acceptance feedback while keeping the default path disabled.
- `common_speculative_accept()` was allowed to receive `n_accepted=0` so failed draft rounds could feed the controller.

Comparison target:

- Historical autotune best config, not a fresh sweep: ctx=32768, batch=2560, ubatch=480, kv=q4_0/q4_0, spec=ngram-mod, extra=base.
- The old best row was `gui-autotune-Qwen3.6-27B-Q3_K_S-20260512-073021-cfg01` at 9.9751 TPS with `task.n_tokens=6971`.
- The valid cold-first A/B uses `--real-context-mode repo-snapshot --real-context-chars 21872` and no `--v2-prime-pass`.
- A prior primed A/B (`e001-best-ub480-repo21872-baseline` vs `e001-best-ub480-repo21872-dynamic`) is superseded because the unmeasured prime pass is not a real first-request scenario.

Commands:

```bash
python scripts/agent_workload_bench.py --label e001-best-ub480-repo21872-noprime-baseline --history-version v2 --build-id bld-20260508113803-2d592989 --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks v2-mini --runs 1 --ctx-size 32768 --batch-size 2560 --ubatch-size 480 --cache-type-k q4_0 --cache-type-v q4_0 --max-tokens 120 --real-context-mode repo-snapshot --real-context-chars 21872 --no-reuse --allow-ctx-above-16k --background-server-policy fail --task-fail-timeout 0 --server-extra "--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-max 64"
```

```bash
LLAMA_NGRAM_MOD_DYNAMIC_DRAFT=1 python scripts/agent_workload_bench.py --label e001-best-ub480-repo21872-noprime-dynamic --history-version v2 --build-id bld-20260508113803-2d592989 --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks v2-mini --runs 1 --ctx-size 32768 --batch-size 2560 --ubatch-size 480 --cache-type-k q4_0 --cache-type-v q4_0 --max-tokens 120 --real-context-mode repo-snapshot --real-context-chars 21872 --no-reuse --allow-ctx-above-16k --background-server-policy fail --task-fail-timeout 0 --server-extra "--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-max 64"
```

Measured result:

| Run | TPS | Prompt tokens | Prompt eval TPS mean | Decode eval TPS mean | Draft coverage |
| --- | ---: | ---: | ---: | ---: | --- |
| Static baseline | 9.9270 | 7125 | 962.450 | 25.790 | 0 generated drafts |
| Dynamic prototype | 9.9056 | 7125 | 958.840 | 25.810 | 0 generated drafts |

Artifacts:

- `build_logs/agent-workload/e001-best-ub480-repo21872-noprime-baseline.*`
- `build_logs/agent-workload/e001-best-ub480-repo21872-noprime-dynamic.*`
- `build_logs/agent-workload/gui-autotune-Qwen3.6-27B-Q3_K_S-20260512-073021-autotune-summary.csv`
- `build_logs/agent-workload/gui-autotune-Qwen3.6-27B-Q3_K_S-20260512-073021-cfg01.server.log`

Interpretation:

- The no-prime A/B result is a slight regression (-0.22%) and does not beat the current autotune-best lane.
- The dynamic policy had no signal to act on in the real cold-first run: both logs show `#gen drafts = 0`, `#acc drafts = 0`, `#gen tokens = 0`, `#acc tokens = 0`.
- The theory failed here because it assumed non-zero speculative coverage. Without draft coverage, changing draft length cannot improve throughput; the bottleneck remains normal prefill/decode work.
- The primed result is useful only as steady-state diagnostics: it creates artificial coverage by running the same task once unmeasured, which is not the first-request user scenario.

Runtime decision:

- Decision: Revert prototype from runtime. Keep H02 only as a secondary idea that first requires a mechanism to create real cold-first draft coverage.
- Confidence: Medium for this best-config lane; single-run result, but coverage is exactly zero, so the mechanism cannot express a win.
