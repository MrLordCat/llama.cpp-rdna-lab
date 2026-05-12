# E003: ngram Formula vs Observed Cross-Check

Date: 2026-05-12
Owner: Copilot
Stage: Measured + analytic cross-check

## Objective

Честно проверить, согласуются ли наблюдаемые приросты `ngram-mod` с простой формулой speculative decoding, и какие переменные в практике важны.

## Case A: ngram active flag, but no generated drafts

Compared artifacts:
- baseline: `scan16k-vec-b6144-ub512-none`
- candidate: `scan16k-vec-b6144-ub512-ngrammod-noprime-postrebuild`

Measured via tool:

```bash
python scripts/research/bench_pair_compare.py \
  --baseline-name none \
  --baseline-csv build_logs/agent-workload/scan16k-vec-b6144-ub512-none.csv \
  --baseline-log build_logs/agent-workload/scan16k-vec-b6144-ub512-none.server.log \
  --candidate-name ngram-mod \
  --candidate-csv build_logs/agent-workload/scan16k-vec-b6144-ub512-ngrammod-noprime-postrebuild.csv \
  --candidate-log build_logs/agent-workload/scan16k-vec-b6144-ub512-ngrammod-noprime-postrebuild.server.log
```

Result:
- observed speedup: 1.0412x
- prompt_eval_speedup: 1.0646x
- decode_eval_speedup: 0.9996x

Spec stats check:

```bash
python scripts/research/spec_log_stats.py --log build_logs/agent-workload/scan16k-vec-b6144-ub512-ngrammod-noprime-postrebuild.server.log --json
```

Result:
- `gen_drafts=0`, `acc_tokens=0`

Conclusion (Case A): прирост объясняется prefill variance, не speculative acceptance.

## Case B: ngram with real draft activity

Compared artifacts:
- baseline: `postrebuild-vec-b6144-ub512-none`
- candidate: `postrebuild-vec-b6144-ub512-ngram-prime`

Measured via tool:

```bash
python scripts/research/bench_pair_compare.py \
  --baseline-name postrebuild-none \
  --baseline-csv build_logs/agent-workload/postrebuild-vec-b6144-ub512-none.csv \
  --baseline-log build_logs/agent-workload/postrebuild-vec-b6144-ub512-none.server.log \
  --candidate-name postrebuild-ngram-prime \
  --candidate-csv build_logs/agent-workload/postrebuild-vec-b6144-ub512-ngram-prime.csv \
  --candidate-log build_logs/agent-workload/postrebuild-vec-b6144-ub512-ngram-prime.server.log
```

Result:
- observed speedup: 1.1006x
- prompt_eval_speedup: 1.0071x
- decode_eval_speedup: 1.1788x

Spec stats:

```bash
python scripts/research/spec_log_stats.py --log build_logs/agent-workload/postrebuild-vec-b6144-ub512-ngram-prime.server.log --json
python scripts/research/spec_effective_acceptance.py --log build_logs/agent-workload/postrebuild-vec-b6144-ub512-ngram-prime.server.log --json
```

Extracted:
- local token acceptance: 75/180 = 0.4167
- coverage: calls_accumulate/calls_generate = 3/209 = 0.01435
- effective acceptance: 0.00598

## Formula Cross-Checks

Naive usage (local acceptance as global `a`) over-predicts speedup strongly.
Coverage-aware acceptance is much closer.

Checks executed:

```bash
python scripts/research/formula_vs_observed.py --observed-baseline-tps 9.8511 --observed-candidate-tps 10.8425 --prefill-share 0.70 --prefill-speedup 1.0071 --decode-kernel-speedup 1.1788 --draft-len 60 --accept-rate 0.4167 --spec-overhead 0.08
python scripts/research/formula_vs_observed.py --observed-baseline-tps 9.8511 --observed-candidate-tps 10.8425 --prefill-share 0.70 --prefill-speedup 1.0071 --decode-kernel-speedup 1.1788 --draft-len 60 --accept-rate 0.004871 --spec-overhead 0.08
```

Key insight:
- local acceptance is not equal to global acceptance variable in simple formula.
- practical modeling should use coverage-weighted acceptance.

## Decision

Decision: introduce coverage-aware acceptance modeling in research protocol.
Confidence: Medium-High (measured + cross-checked, but still small sample count).

## Next Step

- Add explicit coverage-aware term in hypothesis formulas for speculative projections.
- For future ngram claims, always report both local acceptance and effective acceptance.
