# E025 Current Environment Retests

## Metadata

- Experiment ID: E025
- Date: 2026-05-16
- Owner: Codex
- Target lane: `Qwen3.6-27B-Q3_K_S`, `tasks=quick`, `task_ids=review_bug,patch_sim`, `ctx=12288`, `batch=6144`, `ubatch=192`, `q4_0/q4_0`, `spec=none`, no-reuse, thinking on

## Reason

- A fresh retest of the kept E015 point showed the local environment is currently slower than the original `9.6080 TPS` reference.
- Before reviving older negative knobs, compare them against the new same-session baseline.

## Baseline

- Old reference: `c01-e015-rdna4-y64w4-r3 = 9.6080 TPS`
- Current retest: `c01-e015-rdna4-y64w4-r3-retest-20260516 = 9.4111 TPS`
- Delta: `-2.05%`
- Decision stats vs old reference:
  - bootstrap 95% CI: `[-0.2261, -0.1695]` TPS
  - verdict: `negative`

## Retest Results

| Candidate | TPS | Delta vs current baseline | Verdict |
| --- | ---: | ---: | --- |
| current baseline | 9.4111 | 0.00% | reference |
| `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11=144` | 9.3837 | -0.29% | negative |
| `GGML_CUDA_FORCE_MMQ_RUNTIME=1` | 9.3746 | -0.39% | negative |
| `GGML_GDN_FAST_EXP=1` | 9.3625 | -0.52% | negative |
| `GGML_GDN_CHUNK_SIZE=192` | 9.3485 | -0.67% | negative |
| `GGML_GDN_CHUNK_SIZE=128` | 9.3522 | -0.63% | negative |

## Decision

- No no-code candidate beats the current environment baseline.
- Keep default E015 code path.
- For future comparisons, use the current same-session baseline `9.4111 TPS` unless the machine is rebooted/cooled and E015 returns closer to `9.6080 TPS`.
- Code-reverted candidates such as E020 compact half-scale are not covered by this no-code retest and require a separate restore/rebuild cycle.

## Artifacts

- `build_logs/agent-workload/c01-e015-rdna4-y64w4-r3-retest-20260516.csv`
- `build_logs/agent-workload/c01-retest-current-streamk144-r3.csv`
- `build_logs/agent-workload/c01-retest-current-force-mmq-r3.csv`
- `build_logs/agent-workload/c01-retest-current-gdn-fast-exp-r3.csv`
- `build_logs/agent-workload/c01-retest-current-gdn-chunk192-r3.csv`
- `build_logs/agent-workload/c01-retest-current-gdn-chunk128-r3.csv`
