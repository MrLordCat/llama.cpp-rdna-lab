# E004: Coverage-Aware Speculative Formula Validation

Date: 2026-05-12
Owner: Copilot
Stage: measured-vs-model validation

## Objective

Проверить, какая формула лучше описывает observed speedup в speculative кейсе:

- naive: использует local acceptance как глобальный параметр `a`
- coverage-aware: использует `a_eff = coverage * local_acceptance`

## Inputs

### Case B (ngram-prime, sparse drafts)

- observed baseline TPS: 9.8511
- observed candidate TPS: 10.8425
- observed speedup: 1.100639x
- prefill_share: 0.70
- prefill_speedup: 1.0071
- decode_kernel_speedup: 1.1788
- draft_len: 60
- local_acceptance: 75/180 = 0.416667
- coverage: 3/209 = 0.014354
- effective_acceptance: 0.005981

### Case A (ngram-mod, no drafts)

- observed baseline TPS: 9.4264
- observed candidate TPS: 9.8146
- observed speedup: 1.041182x
- prefill_share: 0.70
- prefill_speedup: 1.0646
- decode_kernel_speedup: 0.9996
- draft_len: 24
- local_acceptance: 0.0
- coverage: 0.0
- effective_acceptance: 0.0

## Tool Command

```bash
python scripts/research/spec_model_compare.py \
  --observed-baseline-tps 9.8511 \
  --observed-candidate-tps 10.8425 \
  --prefill-share 0.70 \
  --prefill-speedup 1.0071 \
  --decode-kernel-speedup 1.1788 \
  --draft-len 60 \
  --local-acceptance 0.4166666666666667 \
  --coverage 0.014354066985645933 \
  --spec-overhead 0.08

python scripts/research/spec_model_compare.py \
  --observed-baseline-tps 9.4264 \
  --observed-candidate-tps 9.8146 \
  --prefill-share 0.70 \
  --prefill-speedup 1.0646 \
  --decode-kernel-speedup 0.9996 \
  --draft-len 24 \
  --local-acceptance 0.0 \
  --coverage 0.0 \
  --spec-overhead 0.08
```

## Measured-vs-Model Result

Case B:

- naive projection: 1.416815x
- coverage-aware projection: 1.113301x
- observed: 1.100639x
- abs error naive: 0.316176
- abs error coverage-aware: 0.012662
- implied acceptance for exact fit: 0.004871

Case A:

- naive projection: 1.018689x
- coverage-aware projection: 1.018689x
- observed: 1.041182x
- abs error naive: 0.022493
- abs error coverage-aware: 0.022493
- implied acceptance for exact fit: 0.003044

## Interpretation

1. В sparse-draft кейсе (Case B) naive модель системно переоценивает эффект.
2. Coverage-aware модель в Case B существенно ближе к measured результату.
3. В no-draft кейсе (Case A) обе модели сходятся, что ожидаемо при coverage=0.
4. Для практики прогнозов в этом репозитории acceptance надо учитывать как effective acceptance, а не только local token ratio.

## Decision

Decision: adopt coverage-aware acceptance for speculative research projections.
Confidence: Medium-high (clear gain in sparse-draft case + neutral behavior in no-draft control), pending more traces.

## Next Step

- Повторить эту проверку на других speculative runs (ngram/simple/map/mtp где возможно) и оценить устойчивость вывода.
