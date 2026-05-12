# P4 Validation Protocol And Gates

## Objective

Define a strict, reproducible protocol so planning-to-implementation transition is safe and performance claims are trustworthy.

## Scope

This protocol applies to all points in this folder:

- P1 prefill shape route blocker
- P2 MMVQ decode and linker blocker
- P3 HIP build pressure and TU split
- P5 long-run server stability and throughput

## Code study map

- scripts/agent_workload_bench.py: lane and mode contract arguments (`--tasks`, `--real-context-mode`, `--no-reuse`, `--no-disable-thinking`), active-lane context policy guard (`ctx <= 16384` unless explicit override), runtime failure surfacing (`row["error"]`, exit code `2`, `RuntimeError` path), diagnostics artifacts (`.diagnostics.json/.md`) with prompt/decode metrics and warning/error extraction.
- scripts/repo_snapshot_context_bench.py: same context policy gate (`--allow-ctx-above-16k` for archival runs only).
- BENCHMARKS.md: active-lane policy (`ctx <= 16k`, no-reuse, repo-snapshot incoming context) and reproducibility examples (`runs=3`, stdev reporting, prompt/decode deltas).
- docs/rocm-deep-plan/P3_hip_build_pressure_and_tu_split.md: profile-level runtime closure evidence (default/reduced pass, mmvq-focused timeout on active lane).

## Verified observations for this pass (2026-05-11)

- Active context policy is enforced in scripts, not just documented: both benchmark runners block `ctx > 16384` unless `--allow-ctx-above-16k` is explicitly provided.
- Cold prompt-heavy lane contract is already encoded in practical usage and docs (`v2-review`, repo-snapshot context, no-reuse, thinking enabled).
- Diagnostics already provide enough signal for gate-level decisions: aggregate TPS, error count, prompt/decode timing, prompt/decode TPS, and warning/error lines from server logs.
- Existing data shows low-noise corridors where strict gates are meaningful (example: `p1-confirm-...-base-ub192-r3` stdev `0.0009`).
- Existing data also shows profile eligibility risk that must be codified as a gate: `mmvq-focused` can run short quick lane but fails active lane completion by timeout in prompt-heavy path.

## Benchmark baseline contract

Primary lane contract:

- model: Qwen3.6-27B-Q3_K_S.gguf
- ctx: 12288
- batch/ubatch reference: 6144/192
- kv: q4_0/q4_0
- tasks: v2-review
- no-reuse and thinking enabled

Comparison policy:

- Do not compare across mixed context or mixed reuse modes.
- Use runs=1 for rapid iteration, runs=3 only for borderline confirmation.
- Keep labels explicit about mode and toggles.
- P5 may use a dedicated long-run scenario contract (ctx=65536); artifact, gate, and reporting structure from this protocol still apply.

## Required artifacts per experiment

- build_logs/agent-workload/[label].csv
- build_logs/agent-workload/[label].jsonl
- build_logs/agent-workload/[label].server.log
- optional diagnostics markdown/json if generated

## Quality gates

Gate A: build reliability

- Configure succeeds with expected mode flags.
- Build finishes without linker or unresolved-symbol failures.

Gate B: correctness

- No runtime crash/hang in first prompt batch.
- No request timeout in benchmark task execution for target lane.
- No API/output integrity issues in benchmark script flow.

Gate C: performance

- Improvement must be above noise floor.
- Breakthrough target for active lane: at least +10 percent wall TPS over ub192 reference.
- Report prompt and decode metrics separately.
- Promotion threshold for non-breakthrough improvements: treat changes below both `1.0%` and `0.10 TPS` as noise; require `runs=3` confirmation for any claimed keep decision.

Gate D: rollback safety

- Any reproducible regression above noise triggers rollback candidate.
- Any hang or non-deterministic build failure blocks promotion.

Gate E: profile eligibility

- Only profiles that pass active-lane completion with `errors=0` can be promoted for performance claims.
- Profiles marked as debug/build-only are allowed for smoke A/B but are excluded from production-lane claims.

## Command templates

Reference benchmark command:

python scripts/agent_workload_bench.py --label [label] --server-bin [build]/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks v2-review --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 192 --cache-type-k q4_0 --cache-type-v q4_0 --server-extra "--spec-type none" --real-context-mode repo-snapshot --real-context-chars 21872 --no-reuse --no-v2-prime-pass --no-disable-thinking --max-tokens 120

## Reporting template for each implementation step

1. Change summary
2. Exact files touched
3. Build result
4. Prompt metrics delta
5. Decode metrics delta
6. Aggregate TPS delta
7. Decision: keep or rollback

## Theoretical confirmation matrix

| Claim | Status | Evidence |
| --- | --- | --- |
| Context policy can be enforced as a hard gate in tooling | Confirmed | `scripts/agent_workload_bench.py`, `scripts/repo_snapshot_context_bench.py` (`ctx > 16384` guard) |
| Protocol can separate correctness and performance failures | Confirmed | `errors` field + diagnostics output in `agent_workload_bench.py`; timeout/hang artifacts in `build_logs/agent-workload/*` |
| Existing protocol text (before this pass) fully prevented profile-level false promotion | Rejected | P3 closure: `mmvq-focused` passed build and quick lane but timed out on active lane |
| Adding explicit profile-eligibility + noise-floor thresholds closes that gap | Confirmed in theory | Gate E + tightened Gate C/B in this document |
| P4 by itself should produce direct TPS uplift | Rejected | P4 is a validation/protocol phase, not a kernel/runtime speed phase |

## Theoretical verdict (go/no-go)

Verdict: GO for scoped implementation.

Rationale:

- P4 now has explicit, testable gates for build, correctness, performance, rollback safety, and profile eligibility.
- The protocol directly addresses the most recent failure pattern (profile passes smoke but fails active lane).
- The phase has low regression risk because initial implementation scope is validation tooling/process, not runtime-kernel behavior.

## Implementation scope decision

Do implement P4, but with strict scope:

- P4-A (required): apply this protocol to upcoming point implementations and report each change with the required artifact bundle and gate outcomes.
- P4-B (optional, next step): add an opt-in strict gate mode in `scripts/agent_workload_bench.py` that can fail CI/local run when Gate B/C/E conditions are not met.
- P4-C (out of scope): no direct kernel/runtime optimization changes in this phase.

## Exit criteria for planning stage

- Every point document has explicit build, correctness, and performance gates.
- Noise floor and promotion thresholds are explicit.
- Profile eligibility is explicit (production-lane vs debug/build-only profiles).
- Dependencies between points are clear.
- Implementation order is approved.

Only after that, code modifications and tests begin.
