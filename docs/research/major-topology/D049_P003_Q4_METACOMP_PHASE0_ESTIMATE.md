# D049 - P003 Q4 MetaComp Phase 0 estimate gate

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: measured estimator gate (no runtime kernel edits)

## Scope

Phase 0 objective from D048: estimate realistic VRAM/file savings for a
Q4-preserving algorithm by compressing Q4 metadata overhead, without changing
4-bit payload semantics.

Estimator script added:

- `scripts/research/q4_metacomp_estimator.py`

Run:

- model: `models/Qwen3.6-27B-Q4_K_S.gguf`
- label: `q4metacomp-estimate-qwen36-27b-q4ks-r1`
- assumption: `meta_save_frac=0.60` (remove 60% of metadata bpw on Q4 tensors)

## Baseline context captured before D049

- Full offload fit check on practical lane fails for this Q4 model:
  `q4fit-vulkan130k-big-c152k-b512-ub256-r1` (server exits before ready).
- Working fit-auto practical TPS baseline:
  `q4fitauto-vulkan130k-big-c152k-b512-ub256-r1` = `0.1178 TPS`, prompt
  `427.86`, decode `4.15`.
- Quality baseline for this track:
  `q4metacomp-bfcl-default8-r1` = `8/8`.

## Estimator Results

From `q4metacomp-estimate-qwen36-27b-q4ks-r1.q4_metacomp_estimate.md`:

- total model bytes: `16110362624` (`15.004 GiB`)
- predicted total bytes: `15225921536` (`14.180 GiB`)
- predicted total savings: `884441088` bytes (`0.824 GiB`, `5.49%`)

Q4-only portion:

- Q4-only bytes: `13266616320` (`12.355 GiB`)
- predicted Q4-only bytes: `12382175232` (`11.532 GiB`)
- predicted Q4-only savings: `884441088` bytes (`0.824 GiB`, `6.67%`)

Top individual saver in this model is `token_embd.weight` (`~45.47 MiB` under
this assumption); many FFN Q4 tensors contribute around `~3.19 MiB` each.

## Decision

1. Keep P003 open and advance to implementation.
2. The estimate is large enough to justify Phase 1 converter work (`~0.82 GiB`
   projected), but still below the full-offload deficit observed in fit logs,
   so additional runtime memory measures are likely needed together with
   metadata compaction.
3. Continue with a conservative integration strategy:
   - Phase 1 converter + roundtrip checker,
   - Phase 2 correctness-only gated runtime path,
   - then fused backend work.

## Artifacts

- `scripts/research/q4_metacomp_estimator.py`
- `build_logs/agent-workload/q4metacomp-estimate-qwen36-27b-q4ks-r1.q4_metacomp_estimate.json`
- `build_logs/agent-workload/q4metacomp-estimate-qwen36-27b-q4ks-r1.q4_metacomp_estimate.md`
- `build_logs/agent-workload/q4fit-vulkan130k-big-c152k-b512-ub256-r1.server.log`
- `build_logs/agent-workload/q4fitauto-vulkan130k-big-c152k-b512-ub256-r1.diagnostics.md`
- `build_logs/agent-workload/q4metacomp-bfcl-default8-r1.bfcl_lite.summary.md`
