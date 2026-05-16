# E026 FATTN and ngram Current-Lane Probe

## Metadata

- Experiment ID: E026
- Date: 2026-05-16
- Owner: Codex
- Target lane: `Qwen3.6-27B-Q3_K_S`, `tasks=quick`, `task_ids=review_bug,patch_sim`, `ctx=12288`, `batch=6144`, `ubatch=192`, `q4_0/q4_0`, no-reuse, thinking on
- Baseline: `c01-e015-rdna4-y64w4-r3-retest-20260516 = 9.4111 TPS`
- Hypotheses: H05, H09/H10, H01/H02

## FATTN Trace

- Trace artifact: `build_logs/agent-workload/e026-current-fattn-trace-r1.server.log`
- Path artifact: `build_logs/agent-workload/e026-current-fattn-path-r1.server.log`
- Sync CUDA_NODE total: `24758.198 ms`
- `FLASH_ATTN_EXT forward`: `638.004 ms`
- Share: about `2.58%` of sync CUDA_NODE time
- Dominant FATTN shape:
  - `ne=(256,24,192,1)`: `607.121 ms`, count `1216`, avg `0.4993 ms`
- Dispatch/path signal:
  - WMMA config logs show `D=256`, `q_rows=192`, `selected_cols=16`
  - this is the current WMMA F16 path, not an obvious VEC-threshold miss on the active prompt shape

### FATTN Decision

- No immediate FATTN code probe.
- Reason: current wall ceiling is small. Even a local `10%` FATTN win would be roughly `0.25-0.30%` wall on this lane.
- FATTN should be revisited only if the lane changes to a longer-context/FATTN-heavy profile, or if a very low-risk selector knob exists.

## ngram Results

| Candidate | TPS | Delta vs baseline | Statistics verdict | Notes |
| --- | ---: | ---: | --- | --- |
| baseline spec none | 9.4111 | 0.00% | reference | current same-session baseline |
| `ngram-mod 24/48/64` | 9.7225 | +3.31% | inconclusive | high variance; gain appears on run 3 |
| `ngram-mod n_match=12` | 9.4153 | +0.04% | inconclusive | lower acceptance, no useful speedup |
| `ngram-simple` | 9.3882 | -0.24% | inconclusive/negative leaning | generated zero drafts |

Run split for `ngram-mod 24/48/64`:

| Repeat | Baseline TPS | ngram TPS | Delta |
| --- | ---: | ---: | ---: |
| 1 | 9.4380 | 9.4061 | -0.34% |
| 2 | 9.4081 | 9.4068 | -0.01% |
| 3 | 9.3872 | 10.4229 | +11.03% |

Spec stats for `ngram-mod 24/48/64`:

- cumulative generated drafts: `5`
- accepted drafts: `5`
- generated draft tokens: `311`
- accepted draft tokens: `126`
- local acceptance: `0.4051`
- coverage: `0.0167`
- effective acceptance: `0.00675`
- last acceptance line: `63/123 = 0.5122`

## Formula Cross-Check

- With prompt/prefill share about `0.684`, draft length `64`, and measured effective acceptance `0.006752`, the simple model overpredicts the observed speedup:
  - observed wall speedup: `1.0331x`
  - projected wall speedup: `1.1041x`
  - implied acceptance for observed speedup: `0.00179`
- Interpretation: the aggregate `+3.31%` is not a stable cold-first effect; it is dominated by late repeated-task draft hits and benchmark variance.

## Decision

- FATTN: no current-lane code work; low ceiling.
- ngram-simple: reject for this lane.
- ngram-mod `n_match=12`: reject/neutral.
- ngram-mod `24/48/64`: keep as an opt-in steady/repeated-task mode candidate, not as cold-first default.
- Next possible ngram work:
  - measure with a workload that intentionally contains repeated edit/review patterns,
  - or add instrumentation/controls for cache warm state before changing defaults.

## Artifacts

- `build_logs/agent-workload/e026-current-fattn-trace-r1.server.log`
- `build_logs/agent-workload/e026-current-fattn-path-r1.server.log`
- `build_logs/agent-workload/e026-current-ngram-mod-r3.csv`
- `build_logs/agent-workload/e026-current-ngram-mod-match12-r3.csv`
- `build_logs/agent-workload/e026-current-ngram-simple-r3.csv`
