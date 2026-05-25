# E244 ROCm Q3_K multi-chunk GEMM gate

## Metadata

- Experiment ID: E244
- Date: 2026-05-25
- Owner: Codex
- Branch/Commit: master after `e5c4d5dff`
- Target lane: ROCm cold-first L1 route model for Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: a larger graph route that processes multiple prompt chunks for the same hot Q3_K weight while the weight is live may reduce repeated `src0` conversion and improve rocBLAS GEMM efficiency by using a wider `n` dimension.
- Mechanism: scout rocBLAS GEMM timing for the same hot shapes as either separate prompt chunks (`3 x n=2048 + 1 x n=1345`) or one combined full-prompt shape (`n=7489`).
- Why now: E228/E241 show the +20% cold target needs a structural Q3_K route. E203 rejected cheap same-step transient reuse because the repeated events are far apart in the current graph order, so the remaining version is a heavier graph-scheduling or multi-chunk route.

## Math / Theory

- Assumptions:
  - E228 robust Q3_K split: total `3783.195 ms`, `src0_convert_ms=508.206`, `gemm_ms=2906.403`.
  - Prompt chunk pattern is three full `n=2048` chunks plus one `n=1345` tail.
  - A perfect weight-major route could convert a repeated `src0` once for the prompt instead of four times, but only if it avoids extending fp16 residency across the whole model.
- Expected speedup corridor:
  - GEMM coalescing must beat separate chunks by enough to justify graph complexity.
  - Conversion reuse plus GEMM coalescing must project beyond a low-single-digit wall gain before runtime implementation.
- Failure conditions:
  - combined `n=7489` GEMM is tied or slower than separate chunks;
  - projected wall ceiling remains far below the current +20% target;
  - naive full-prompt intermediates exceed the memory/residency corridor already exposed by `ubatch=3072/4096` gates.

## Implementation Plan

1. Minimal code surface to change:
   - none for this gate; use existing standalone `rocm_rocblas_solution_scout.exe` with `--max-solutions 0` to time default rocBLAS GEMM.
2. Guard rails:
   - this is not solution-index plumbing;
   - no runtime graph changes unless the standalone route ceiling is strong.
3. Rollback path:
   - diagnostic-only.

## Benchmark Plan

- Baseline command:
  - run default rocBLAS GEMM scout for `n=2048` and `n=1345` on hot shapes.
- Candidate command:
  - run the same scout for `n=6144` and/or `n=7489`.
- Number of runs:
  - standalone `warmup=4`, `iters=10`.
- Artifacts path:
  - `build_logs/agent-workload/e244-rocblas-multichunk-*.csv`

## Metrics

- default rocBLAS GEMM average milliseconds
- separate chunk synthetic total: `3 * n2048 + n1345`
- combined chunk total: `n7489`
- local time reduction
- projected route ceiling

## Result

- Outcome: keep as a stack/design clue, not an immediate implementation.
- Delta:
  - main FFN `17408x5120`: separate `11.7008 ms`, combined `10.9411 ms`, `-6.49%` time;
  - reverse FFN `5120x17408`: separate `11.4727 ms`, combined `10.7176 ms`, `-6.58%` time;
  - `10240x5120`: separate `7.0267 ms`, combined `6.6638 ms`, `-5.16%` time;
  - `6144x5120`: separate `4.3125 ms`, combined `4.2446 ms`, `-1.57%` time.
- Confidence: medium for the standalone rocBLAS shape signal; low for runtime conversion because graph order and memory residency are the hard part.
- Recommendation: do not implement a naive full-ubatch graph route. Preserve this as a possible Q3_K stack component only if the implementation can stream/recycle intermediates and avoid full-prompt f32 outputs staying resident.

## Measured Data

| Shape `(m,k)` | `n=2048` | `n=1345` | Separate `3x2048+1345` | `n=7489` | Time delta | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `17408,5120` | `3.1652 ms` | `2.2052 ms` | `11.7008 ms` | `10.9411 ms` | `-6.49%` | `1.0694x` |
| `5120,17408` | `3.0894 ms` | `2.2045 ms` | `11.4727 ms` | `10.7176 ms` | `-6.58%` | `1.0705x` |
| `10240,5120` | `1.8924 ms` | `1.3495 ms` | `7.0267 ms` | `6.6638 ms` | `-5.16%` | `1.0545x` |
| `6144,5120` | `1.1587 ms` | `0.8364 ms` | `4.3125 ms` | `4.2446 ms` | `-1.57%` | `1.0160x` |

Full-chunk-only check:

| Shape `(m,k)` | Separate `3x2048` | `n=6144` | Time delta | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `17408,5120` | `9.4956 ms` | `8.9905 ms` | `-5.32%` | `1.0562x` |
| `5120,17408` | `9.2682 ms` | `8.9456 ms` | `-3.48%` | `1.0361x` |

Approximate route ceiling:

- If all robust Q3_K conversion were reused across four chunks, max `src0` savings is roughly `3/4 * 508.206 = 381 ms`.
- If GEMM coalescing saved about `5.5%` across the `2906.403 ms` robust GEMM bucket, that adds roughly `160 ms`.
- Combined optimistic route saving is therefore about `540 ms` on an E228 traced total near `7.68 s`, or about `+7%` wall before overhead.
- That is meaningful as a stack component, but it is not enough for the current standalone `+20%` cold target.

Memory/residency warning:

- A single `17408 x 7489` f32 output is about `497 MiB`.
- Gate/up together would approach `1 GiB` before GLU output, and a naive full-prompt graph would also increase other activations.
- This matches the warning from E240/E224: larger physical `ubatch` shapes can fit poorly or timeout even when individual GEMMs look better.

## Notes

- Surprises:
  - rocBLAS does get a measurable efficiency lift from wider `n`, but the gain is moderate rather than breakthrough-sized.
- Follow-up action:
  - only revisit as a graph-scheduling design that converts each hot Q3_K weight once and streams chunk outputs through consumers;
  - do not try to get this by simply raising `ubatch` or by keeping full-prompt FFN intermediates resident.
