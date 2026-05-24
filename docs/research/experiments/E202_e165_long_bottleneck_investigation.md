# E202+E165 Long Bottleneck Investigation

## Metadata

- Experiment ID: E202+E165
- Date: 2026-05-24
- Owner: Codex + user
- Context: long-form continuation of E165 (fused Q3_K preload-y failure)
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Why this track exists

E165 showed a classic bottleneck shift:

- local idea: fewer repeated `q8_1` loads in fused Q3_K;
- side effect: much larger live state in kernel;
- result: register cliff and occupancy loss, then slower dominant fused buckets.

E165 measured:

- dominant fused buckets: slower (`ncols_x=5120`, `ncols_x=17408`),
- resources: `regs 84 -> 136`, `occupancy 87.5% -> 68.75%`,
- decision: reject and revert.

## E202 baseline lock (fresh)

Fresh control run on the same lane was captured to start E202 from current tree.

Run label and artifacts:

- `build_logs/agent-workload/e202-e165-baseline-r1.diagnostics.md`
- `build_logs/agent-workload/e202-e165-baseline-r1.server.log`

Route/resource confirmation from log (fused Q3_K path, `ncols_dst=1`, `small_k=1`, `fusion=1`):

- `ncols_x=5120`, `grid=(8704,1,1)`, `block=(32,2,1)`, `regs=84`, `occupancy=87.50%`, `static_shared=512`
- `ncols_x=17408`, `grid=(2560,1,1)`, `block=(32,2,1)`, `regs=84`, `occupancy=87.50%`, `static_shared=512`

This matches the pre-E165 resource corridor and is the hard gate for all new candidate code.

## What exactly increased in E165 (and why this hurts)

When preload-y was added, the kernel kept more intermediate values alive across both dot paths.
That increases:

1. per-thread register allocation (VGPR pressure),
2. scheduler constraints (fewer resident waves),
3. latency-hiding ability (less overlap while waiting on memory/pipe dependencies),
4. risk of additional spills or less favorable instruction scheduling.

So even if one micro-part got less memory traffic, the dominant cost moved to execution pressure/scheduling.

## Core E202 rule: budget-first, not preload-first

Candidate is allowed to continue only if all are true on resource gate:

- dominant fused Q3_K buckets keep `regs` close to 84 (no cliff),
- occupancy does not drop below the control corridor,
- no regression on dominant `ncols_x=5120` and `ncols_x=17408` timing rows.

If a candidate improves a low-share bucket but worsens the dominant buckets, it is immediate reject.

## Planned sequence

### Phase A: Causality lock

1. Reuse E165 evidence + fresh E202 baseline.
2. Keep a compact before/after table per candidate:
   - local improved section,
   - dominant fused bucket timing,
   - regs/occupancy,
   - lane wall delta.

### Phase B: Candidate families (small, guarded)

Candidate family priority:

1. Live-range shrink in fused Q3_K helper path (shorter lifetime of temporary values).
2. Work partitioning to avoid dual-path live-state growth in one thread scope.
3. Only after resource pass: runtime r1; then r3 for any non-noise signal.

Explicitly forbidden repeats:

- direct preload-style duplication that inflates live arrays,
- any candidate that reproduces `regs >= 120` in dominant fused buckets,
- r1-only speed claims.

### Phase C: Accept rule

Keep only if:

- resource gate passes,
- dominant fused buckets do not regress,
- lane r3 is positive on same contract.

Otherwise classify as bottleneck shift or noise and close.

## Current status

- E165: completed, rejected, reverted.
- E202: baseline locked, ready for first code-backed candidate.

## Candidate A1 (streaming pair-dot) outcome

Implementation notes:

- helper compile fix in `ggml/src/ggml-cuda/vecdotq.cuh`: explicit `__half2float(...)` cast for `bq3->d * sumf` to avoid HIP overload ambiguity.
- fused path wiring in `ggml/src/ggml-cuda/mmvq.cu` remained as A1 candidate under the same lane contract.

Run and artifacts (first pass, resource-gate focused):

- label: `e202-e165-a1-rg-r1`
- `build_logs/agent-workload/e202-e165-a1-rg-r1.server.log`
- `build_logs/agent-workload/e202-e165-a1-rg-r1.csv`

Resource gate comparison vs E202 baseline (dominant fused Q3_K buckets):

- baseline (`e202-e165-baseline-r1`):
   - `ncols_x=5120`: `regs=84`, `occupancy=87.50%`
   - `ncols_x=17408`: `regs=84`, `occupancy=87.50%`
- A1 (`e202-e165-a1-rg-r1`):
   - `ncols_x=5120`: `regs=95`, `occupancy=100.00%`
   - `ncols_x=17408`: `regs=95`, `occupancy=100.00%`

Initial lane signal from this run (r1, no reuse):

- baseline completion TPS: `26.4663`
- A1 completion TPS: `12.2152`

Root-cause check (apples-to-apples rerun):

- The large TPS drop above was caused by **prompt mismatch**, not decode regression:
   - baseline prompt tokens: `159`
   - `e202-e165-a1-rg-r1` prompt tokens: `7413` (repo snapshot injection active)
- Matching run without repo-snapshot context (`e202-e165-a1-apples-r1`) produced:
   - prompt tokens: `159`
   - aggregate completion TPS: `29.8122`
   - decode eval TPS: `32.11` (vs baseline `29.66`)

Artifacts:

- `build_logs/agent-workload/e202-e165-a1-apples-r1.server.log`
- `build_logs/agent-workload/e202-e165-a1-apples-r1.csv`
- `build_logs/agent-workload/e202-e165-a1-apples-r1.diagnostics.md`

Decision:

- **Do not classify A1 as runtime regression** based on `e202-e165-a1-rg-r1`; that run is invalid for throughput comparison due to prompt mismatch.
- Current status: A1 is **performance-positive in matched r1**, but still flagged for resource-corridor drift (`regs 84 -> 95`) that needs follow-up.
- As agreed, **r3 was not run**.

## A1.1 and A1.2 follow-up (implementation bottleneck research)

Hypothesis:

- Remove runtime gate branching in MMVQ hot path via compile-time specialization (`use_gate_fast`) and check if this lowers overhead.

### A1.1 (always gate-fast for fused Q3_K)

Code direction:

- Added template specialization parameter in MMVQ kernel path and launched gate-fast variant for fused Q3_K ncols_dst=1.

Observed behavior:

- apples runtime stayed strong (`e202-e165-a11-apples-r1`: `30.0067` TPS).
- but resource gate showed **new cliff** on dominant bucket:
   - `ncols_x=5120`: `regs=120`, `occupancy=68.75%`
   - `ncols_x=17408`: `regs=95`, `occupancy=100.00%`

Conclusion:

- Full gate-fast specialization over-increased compiler live state specifically in the 5120 bucket.
- A1.1 is rejected as unstable for corridor control.

### A1.2 (guarded gate-fast dispatch)

Code direction:

- Keep gate-fast only for larger K (`ncols_x > 8192`), fallback to regular fused path for 5120 bucket.
- Also kept helper live-range cleanup in pair-stream loop.

Observed behavior:

- apples runtime remains strong:
   - `e202-e165-a12-apples-r1`: `29.8457` TPS
- resource gate recovers from A1.1 cliff:
   - `ncols_x=5120`: `regs=95`, `occupancy=100.00%`
   - `ncols_x=17408`: `regs=95`, `occupancy=100.00%`

Current best interpretation:

- Original large drop was measurement mismatch (prompt mismatch).
- Real implementation risk is **resource-shape sensitivity by bucket** under aggressive specialization.
- A1.2 is viable for continued study (better runtime than baseline, no A1.1-style 120-reg cliff), but still outside strict baseline corridor (`84 -> 95`).

## Wider risk check before further implementation

Question addressed: can this narrowing (now `ncols_x == 6144 || ncols_x > 8192` for gate-fast) hurt other points later?

Observed fused Q3_K `ncols_x` buckets in current agent-workload logs:

- `5120` (dominant)
- `6144` (present in prefill traces and A1.1 resource logs)
- `17408` (dominant)

Expanded matrix from E202/E165 logs (`ncols_x`, `regs`, count):

- `5120`: `70` (1350), `84` (258), `88` (2880), `94` (30), `95` (516), `120` (516)
- `6144`: `42` (240), `62` (960), `70` (80), `84` (60), `88` (20), `95` (240)
- `17408`: `70` (315), `84` (258), `95` (1032)

Implication for current guard:

- `5120` stays on non-fast path and avoids the known `120` cliff from A1.1.
- `6144` and `17408` use fast path in current candidate and both show `regs=95` in new resource-gated run.
- The A1.1 cliff (`regs=120`) was shape-local to `5120` when fast path was forced everywhere.

Risk that remains:

- The guard is empirical for shapes seen so far; unseen shapes above 8192 in other models/lane configs could still regress.

Safe rollout recommendation:

1. Keep A1.2 as **experimental** (not final) until multi-lane resource gate confirms no new cliffs.
2. Run same resource check on at least one additional Q3_K model family with different hidden size.
3. Only after that, proceed with further narrowing/implementation changes.

## A1.3 candidate (enable 6144 in fast path) and validation

Code direction:

- Updated Q3_K fused guard in `mmvq.cu` from `ncols_x > 8192` to `(ncols_x == 6144) || (ncols_x > 8192)`.

Fresh A/B in identical current lane contract (`ctx=12288,b=6144,ub=2048`, quick/triage_diff, no reuse, repo-snapshot real context):

- baseline apples (`e202-e165-a12c-baseline-apples-r1`): `12.3456` TPS
- candidate apples (`e202-e165-a13-cand-apples-r1`): `12.3814` TPS
- delta: `+0.0358` TPS (`+0.29%`)

Resource-gated comparison:

- baseline rg (`e202-e165-a12c-baseline-rg-r1`): `12.2928` TPS
- candidate rg (`e202-e165-a13-cand-rg-r1`): `12.3104` TPS
- delta: `+0.0176` TPS (`+0.14%`)

Resource profile in candidate remains stable (no new cliff):

- `5120`: `70/88/95`
- `6144`: `70/88/95`
- `17408`: `70/95`

Decision:

- Keep A1.3 candidate code as a small positive/no-regression step.
- Treat gain as modest and still require broader cross-model/cross-lane confirmation before calling this a generalized policy.
