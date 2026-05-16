# E024 H08 C05 GDN Chunk128 Probe

## Metadata

- Experiment ID: E024
- Date: 2026-05-16
- Owner: Codex
- Target lane: `Qwen3.6-27B-Q3_K_S`, `tasks=quick`, `task_ids=review_bug,patch_sim`, `ctx=12288`, `batch=6144`, `ubatch=192`, `q4_0/q4_0`, `spec=none`, no-reuse
- Hypothesis ID: H08

## Hypothesis

- Statement: For the active C05 prompt route with `n_tokens=192`, changing GDN chunking from `96+96` to `128+64` might slightly improve scheduling or per-launch efficiency while keeping the same number of launches.
- Why this was still worth checking:
  - E022 rejected `chunk_size=192`, but not `chunk_size=128`.
  - Current C01 trace shows GDN at about `1467.855 ms`, with prompt GDN around `1174.486 ms`.

## Math / Theory

- Default for `n_tokens=192`: `chunk_size=96`, two launches of `96+96`.
- Candidate: `chunk_size=128`, two launches of `128+64`.
- Launch count is unchanged, so the only plausible win is from better occupancy/cache behavior in one larger chunk.
- Ceiling:
  - GDN is about `6.6%` of sync CUDA_NODE time.
  - A local `10%` GDN win would be roughly `0.6-0.7%` wall-time, so the runtime gate must be strict.

## Result

- Baseline/reference: `c01-e015-rdna4-y64w4-r3 = 9.6080 TPS`
- Candidate: `c01-e024-gdn-chunk128-r1 = 9.43 TPS`
- Delta: about `-1.85%`
- Prompt eval:
  - `851.23 tok/s` on `review_bug`
  - `849.15 tok/s` on `patch_sim`
- Decode eval:
  - `29.94 tok/s` on `review_bug`
  - `29.90 tok/s` on `patch_sim`

## Decision

- `reject`
- Reason: cheap screen is below the current best by more than the expected ceiling.
- Trace: skipped because the non-trace r1 gate failed.
- Code state: no code changes.
- Artifact:
  - `build_logs/agent-workload/c01-e024-gdn-chunk128-r1.csv`
