# E005: Spec Model Batch Validation (Low vs High Coverage)

Date: 2026-05-12
Owner: Copilot
Stage: measured-vs-model validation (multi-case)

## Objective

Проверить устойчивость вывода из E004 на серии кейсов:

- sparse-coverage ngram
- high-coverage MTP

Сравниваются две версии модели:

- naive: использует local acceptance как глобальный коэффициент
- coverage-aware: использует effective acceptance = coverage * local_acceptance

Дополнительно оценивается required speculative overhead для совпадения с observed wall speedup.

## Cases

Source manifest:

- docs/research/experiments/E005_spec_model_cases.json

Cases included:

1. ngram_prime_sparse_postrebuild
2. specngram_lowcov_exp_b4096_ub512
3. mtp_n1_highcov_c12288
4. mtp_n2_highcov_c12288
5. mtp_n3_highcov_c12288
6. mtp_n2_highcov_c12288_ub256

## Tool Commands

```bash
python scripts/research/spec_model_batch_compare.py \
  --cases-json docs/research/experiments/E005_spec_model_cases.json \
  > build_logs/agent-workload/e005-spec-model-batch.txt

python scripts/research/spec_model_batch_compare.py \
  --cases-json docs/research/experiments/E005_spec_model_cases.json \
  --json > build_logs/agent-workload/e005-spec-model-batch.json

python scripts/research/required_spec_overhead.py \
  --batch-json build_logs/agent-workload/e005-spec-model-batch.json \
  --acceptance-mode effective \
  > build_logs/agent-workload/e005-required-overhead-effective.txt
```

## Measured-vs-Model Results

| id | observed | naive | coverage-aware | err naive | err cov | better |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ngram_prime_sparse_postrebuild | 1.100641 | 1.416874 | 1.113340 | 0.316233 | 0.012699 | coverage-aware |
| specngram_lowcov_exp_b4096_ub512 | 0.998800 | 1.346673 | 0.984610 | 0.347874 | 0.014190 | coverage-aware |
| mtp_n1_highcov_c12288 | 0.538537 | 0.671789 | 0.671789 | 0.133252 | 0.133252 | tie |
| mtp_n2_highcov_c12288 | 0.601650 | 0.956872 | 0.953700 | 0.355222 | 0.352050 | coverage-aware |
| mtp_n3_highcov_c12288 | 0.572592 | 1.026286 | 1.018304 | 0.453694 | 0.445712 | coverage-aware |
| mtp_n2_highcov_c12288_ub256 | 0.565025 | 0.786504 | 0.783712 | 0.221479 | 0.218687 | coverage-aware |

Summary:

- coverage-aware wins: 5/6
- naive wins: 0/6
- ties: 1/6

## Required Overhead Backsolve (effective acceptance)

| id | required overhead o |
| --- | ---: |
| ngram_prime_sparse_postrebuild | 0.135091 |
| specngram_lowcov_exp_b4096_ub512 | 0.031026 |
| mtp_n1_highcov_c12288 | 0.678427 |
| mtp_n2_highcov_c12288 | 2.218596 |
| mtp_n3_highcov_c12288 | 3.564481 |
| mtp_n2_highcov_c12288_ub256 | 1.713936 |

## Interpretation

1. Coverage-aware acceptance устойчиво лучше naive для low-coverage speculative кейсов.
2. Для MTP observed speedup во всех выбранных кейсах ниже 1.0x, несмотря на высокую local/effective acceptance.
3. Для согласования формулы с observed MTP нужны очень большие implied overhead (до ~3.56), что указывает на отсутствие в модели важного overhead-компонента.

## Decision

Decision: keep H09 (coverage-aware acceptance) and open next modeling step for overhead-aware speculative model.
Confidence: High for sparse-coverage conclusion, medium for cross-mode generalization until overhead model is expanded.

## Artifacts

- build_logs/agent-workload/e005-spec-model-batch.txt
- build_logs/agent-workload/e005-spec-model-batch.json
- build_logs/agent-workload/e005-required-overhead-effective.txt
