# P4 Validation Protocol And Gates

## Objective

Define a strict, reproducible protocol so planning-to-implementation transition is safe and performance claims are trustworthy.

## Scope

This protocol applies to all points in this folder:

- P1 prefill shape route blocker
- P2 MMVQ decode and linker blocker
- P3 HIP build pressure and TU split
- P5 long-run server stability and throughput

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
- No API/output integrity issues in benchmark script flow.

Gate C: performance

- Improvement must be above noise floor.
- Breakthrough target for active lane: at least +10 percent wall TPS over ub192 reference.
- Report prompt and decode metrics separately.

Gate D: rollback safety

- Any reproducible regression above noise triggers rollback candidate.
- Any hang or non-deterministic build failure blocks promotion.

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

## Exit criteria for planning stage

- Every point document has explicit build, correctness, and performance gates.
- Dependencies between points are clear.
- Implementation order is approved.

Only after that, code modifications and tests begin.
