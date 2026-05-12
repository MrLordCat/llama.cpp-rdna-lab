# P7 RDNA4 GDN Kernel Hotpath Rework

## Objective

Raise prompt-heavy throughput by reducing device-side cost in RDNA4 `gated_delta_net` non-KDA prefill hotpath.

This pass upgrades P7 from "plausible" to a falsifiable theory with explicit yes/no conditions.

## Scope

- Primary lane: `ctx=12288`, `b=6144`, no-reuse, repo-snapshot workload.
- Secondary safety lane: `ctx=16384` with same protocol.
- Runtime/kernel track only.

## Code anchors (mechanism)

- `ggml/src/ggml-cuda/gated_delta_net.cu:111-209`
	- Inner token loop does per-token state math + reductions.
	- non-KDA branch uses `expf` and repeated warp reductions.
- `ggml/src/ggml-cuda/gated_delta_net.cu:235-367`
	- RDNA4 chunked prefill routing and chunk launch loop (`use_chunked_prefill`, `chunk_size`, per-chunk launches).
- `ggml/src/ggml-cuda/gated_delta_net.cu:273-289`
	- Contract trace (`LLAMA_TRACE_DELTA_NET_CONTRACT`) for route verification.

## Evidence corpus (verifiable)

1. New manual run, shape-score, `ub300`:
	 - aggregate TPS `8.5199`
	 - prompt `826.90 tok/s`, decode `27.65 tok/s`
	 - source: `build_logs/agent-workload/p1-manual-20260511-202428-shape-ub300-r1.diagnostics.md`

2. New manual run, shape-score, `ub512`:
	 - aggregate TPS `4.2357`
	 - prompt `336.24 tok/s`, decode `27.18 tok/s`
	 - source: `build_logs/agent-workload/p1-manual-20260511-202458-shape-ub512-r1.diagnostics.md`

3. Baseline boundary (no shape-score lock):
	 - `ub192`: prompt `826.35`, decode `27.63`, TPS `8.5052`
	 - `ub194`: prompt `594.10`, decode `27.78`, TPS `6.7088`
	 - sources:
		 - `build_logs/agent-workload/p1-gate-20260511-174248-base-ub192-r1.diagnostics.md`
		 - `build_logs/agent-workload/p1-gate-20260511-174521-base-ub194-r1.diagnostics.md`

4. Shape-score near-boundary control:
	 - `ub192` vs `ub194` both stay near prompt `~829 tok/s` and decode `~27.76 tok/s`
	 - sources:
		 - `build_logs/agent-workload/p1-gate-20260511-174248-shape-ub192-r1.diagnostics.md`
		 - `build_logs/agent-workload/p1-gate-20260511-174248-shape-ub194-r1.diagnostics.md`

5. P6 route parity proof at hot `n_tokens=192`:
	 - baseline and adaptive both route to `chunk_size=96`, `n_chunks=2`
	 - sources:
		 - `build_logs/agent-workload/p6-base-trace-only-r1.server.log`
		 - `build_logs/agent-workload/p6-impl-smoke-r2.server.log`

## What this proves

- Large regressions (`ub194` base, `ub512` shape-score) are prefill-side phenomena:
	- prompt collapses heavily;
	- decode remains in tight corridor (`~27 tok/s`).
- Route-level knobs alone are not sufficient:
	- when route remains equivalent (P6 parity at `192/96`), speed does not move.
- Therefore, unresolved lever is inside prefill compute path, not decode selector and not host scheduling.

## Formal speedup model (Amdahl framing)

Let:

- `T = T_prefill + T_decode + T_other`
- `S_prefill = T_prefill / T`
- `S_gdn = T_gdn / T` (unknown directly, but `T_gdn <= T_prefill`)

If P7 accelerates only GDN by factor `r`, total speedup upper bound is:

`Speedup_total <= 1 / ((1 - S_gdn) + S_gdn / r)`.

Observed shares:

- `ub300`: `S_prefill ~= 9710.91 / (9710.91 + 4339.59) ~= 0.691`
- `ub512`: `S_prefill ~= 23881.52 / (23881.52 + 4414.23) ~= 0.844`

Implication:

- P7 headroom is higher in prefill-dominant regimes like `ub512`.
- In mixed regimes (`ub192/300`), measurable gain still requires meaningful GDN share and non-trivial kernel acceleration.

## P7 falsifiable theory

P7 gives speedup if and only if all three hold:

1. Workload repeatedly executes non-KDA chunked prefill hotpath.
2. GDN share in prompt path is material (not a tiny subcomponent).
3. Kernel rewrite reduces per-token math/state cost without occupancy collapse.

P7 is rejected for a lane if any one fails (especially #2 or #3).

## Parameter transfer (will theory hold with other params?)

Yes, but conditionally:

- Likely positive:
	- higher `S_prefill` lanes (longer prefill windows, larger effective token chunks);
	- lanes where prompt dominates wall time.
- Likely weak/zero:
	- decode-dominant lanes;
	- lanes where route is already in stable fast corridor and GDN share is small.

## Verdict

Verdict: GO for implementation as primary kernel track.

Confidence:

- High that P7 is the right class of lever.
- Medium on exact gain magnitude (depends on measured `S_gdn` after instrumentation).

## Execution blueprint

P7-A: observability first

- Add low-overhead per-launch counters/timers for non-KDA path.
- Keep all probes env-gated.

P7-B: arithmetic and register pressure cleanup

- Reduce repeated transcendental and redundant scalar work where mathematically safe.
- Keep KDA and non-KDA variants isolated.

P7-C: state-traffic cleanup

- Minimize redundant loads/stores in non-`keep_intermediates` mode.
- Preserve state handoff semantics across chunks.

P7-D: optional geometry retune

- Only after B/C are measured and stable.

## Performance gates

- Gate 1 (r1 screen): `>= +6%` aggregate TPS on active lane.
- Gate 2 (promotion): `>= +10%` aggregate TPS and prompt-eval uplift `>= +12%`.
- Gate 3 (stability): pass `runs=3` confirmation.

## Risks

- Numerical drift from aggressive math rewrites.
- Register pressure and occupancy regressions.
- Overfitting to one shape band.

## Rollback criteria

- Any correctness regression, hang, or reproducible throughput loss above noise.

## Implementation pass (2026-05-11)

Status: implemented (first pass), built, and benchmarked with final r1 run on target lane.

Code changes applied in `ggml/src/ggml-cuda/gated_delta_net.cu`:

- non-KDA scalar `g` handling optimized from per-lane recomputation to warp-broadcast usage;
- pointer-walk token loop (fewer repeated address recomputations);
- KDA path avoids duplicate `expf` calls within the same token step;
- optional guarded fast-exp path (`GGML_GDN_FAST_EXP=1`);
- optional guarded timing trace (`GGML_TRACE_GDN_TIMING=1`).

Build validation:

- `cmake --build build-rocm-vec --target ggml-hip -j 16` passed.

Benchmark results (`ctx=12288`, `b=6144`, `ub=512`, q4_0/q4_0, shape-score, no-reuse):

- baseline pre-P7: `4.2357` TPS
	- `build_logs/agent-workload/p1-manual-20260511-202458-shape-ub512-r1.diagnostics.md`
- P7 impl r1: `4.1856` TPS
	- `build_logs/agent-workload/p7-impl2-20260511-204029-shape-ub512-r1.diagnostics.md`
- P7 impl + fast-exp guard r1: `4.1966` TPS
	- `build_logs/agent-workload/p7-fastexp-20260511-203910-shape-ub512-r1.diagnostics.md`

Outcome of this pass:

- No measurable uplift versus baseline on the target `ub512` lane in r1.
- Prefill remains dominant bottleneck; decode stays near-flat.
- P7 remains relevant as direction, but this concrete first implementation pass does not pass performance gate.

## Implementation pass2 (2026-05-11)

Status: pass2 kernel micro-optimizations implemented and benchmarked; no direct uplift on `ub512` without context-level change.

Code changes in `ggml/src/ggml-cuda/gated_delta_net.cu`:

- added `__restrict__` qualifiers on hot kernel pointers;
- switched core shard accumulations and state updates to FMA-style math (`fmaf`) in both non-KDA and KDA branches.

Build validation:

- `cmake --build build-rocm-vec --target ggml-hip -j 16` passed.

Benchmark results (`ctx=12288`, `b=6144`, shape-score, no-reuse):

- `ub512` after pass2: `4.2009` TPS
	- `build_logs/agent-workload/p7-pass2-20260511-205249-shape-ub512-r1.diagnostics.md`
- `ub192` guard after pass2: `8.5249` TPS
	- `build_logs/agent-workload/p7-pass2-20260511-205327-shape-ub192-r1.diagnostics.md`

Result: pass2 kernel arithmetic cleanup is effectively neutral on the problematic `ub512` lane.

## Root-cause isolation update (2026-05-11)

Observed paradox (shape-score enabled):

- `ub192` and `ub512` show identical planner chosen/target histograms (`chosen=192` dominant).
- GDN contract `n_tokens` histograms are identical.
- FATTN shape histograms for hot `Q1=192` are identical.
- MMQ selector traces are identical (`ncols_max=192`, `mmq_x_best=96`, `ntiles_x_best=2`).

But wall TPS remains split:

- `ub192`: ~`8.52`
- `ub512`: ~`4.20`

Key evidence from normalized server-log diff:

- context memory footprint is much larger when launching with `-ub 512`, despite effective split to 192;
- the log shows context memory jump from roughly `185 MiB` to `495 MiB` in the compared runs.

Interpretation:

- the regression is dominated by physical context `n_ubatch` inflation, not by GDN/FATTN/MMQ route differences in the traced hotpath.

## Pass2.1 experimental runtime cap (2026-05-11, superseded)

To validate the hypothesis, an opt-in runtime guard was temporarily tested in `src/llama-context.cpp`:

- new env flag: `LLAMA_UBATCH_SHAPE_CONTEXT_CAP=1`;
- active only when `LLAMA_UBATCH_SPLIT_POLICY=shape-score` and `LLAMA_UBATCH_SHAPE_PREFERRED>0`;
- caps physical context `n_ubatch` to `preferred` at context creation.
- this guard was later removed because it was a workaround, not the root cause fix.

Validation runs:

- `ub512` without cap (same binary): `4.19` TPS
	- `build_logs/agent-workload/p7-pass2-postctx-20260511-205925-shape-ub512-r1.diagnostics.md`
- `ub512` with cap enabled: `8.5270` TPS
	- `build_logs/agent-workload/p7-pass2-cap-20260511-205849-shape-ub512-r1.diagnostics.md`
- `ub192` with cap enabled remains in corridor (`~8.53` TPS)
	- `build_logs/agent-workload/p7-pass2-cap-20260511-205746-shape-ub192-r1.diagnostics.md`

Conclusion after pass2.1 and the follow-up root-cause pass:

- P7 kernel passes alone did not recover `ub512`.
- The cap proved the cliff was tied to scheduler reserve/layout pressure, not to the individual GDN/FATTN/MMQ hotpath.
- The final fix is output-aware PP graph reservation in `llama_context::sched_reserve()`, which leaves requested `-ub` intact and reserves PP outputs according to actual decode outputs. Direct `ub490` and `ub512` now stay in the fast band without shape-score/context-cap.
