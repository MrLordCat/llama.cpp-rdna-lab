# E140 Vulkan 64k Q3_K Matmul Split-K Gate

## Metadata

- Experiment ID: E140
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E139 (`9f5959604`)
- Target lane: Vulkan 64k, `Qwen3.6-27B-Q3_K_S`, q4/q4 KV, FlashAttention on, `b8192/ub1024`, `--no-mmap`, `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`, `spec=none`, no reuse

## Hypothesis

- Statement: force the existing Vulkan matmul split-K route for the long-K Q3_K hot shape (`m=5120,n=1024,k=17408`) and test whether shorter K loops plus more dispatch workgroups can beat the direct unsplit Q3_K route.
- Mechanism: unlike E139, this keeps the direct Q3_K matmul shader and avoids a fp16 predequant temp. Split-K writes partial output to `prealloc_split_k`, synchronizes, then runs `pipeline_matmul_split_k_reduce`.
- Why now: E133 shows `m=5120,n=1024,k=17408` is the second-largest parsed Q3_K bucket. E139 rejected predequant, so remaining route-level Q3_K ideas must stay direct or single-dispatch where possible. Split-K is an existing route worth closing before designing a new shader branch.

## Math / Theory

- Hot reverse-shape output temp for split2:
  - `5120 * 1024 * 4 * 2 = 41943040 B`, about `40.0 MiB`.
- This is much smaller than E139's `~170 MiB` fp16 temp, but it still adds a sync and reduce dispatch.
- Expected upside is low because `m=5120,n=1024` already has many workgroups: approximately `ceil(5120/128) * ceil(1024/128) = 320` large-tile groups before split-K, so the route is not obviously parallelism-starved.
- Failure condition: if extra partial writes/reduce/sync dominate, split-K should be rejected as another multi-dispatch anti-pattern for this lane.

## Implementation Plan

1. Add a temporary default-off host env gate in `ggml_vk_mul_mat_q_f16`.
2. Scope to `src0=Q3_K`, large dimensions, and `k>=17000` by default.
3. Force `split_k >= 2` only when the env is set.
4. Run a short pp7488 gate; skip full 64k real-server run unless it is positive.
5. Revert the code if negative.

## Benchmark Plan

- Baseline command: `llama-bench -p 7488 -n 0 -r 1 --no-warmup -b 8192 -ub 1024 -ctk q4_0 -ctv q4_0 -ngl 999 -fa 1 -mmp 0`.
- Candidate command: same, with `GGML_VK_Q3K_FORCE_SPLIT_K=2`.
- Number of runs: `1` for route gate.
- Artifacts path: `build_logs/agent-workload/e140-vulkan-q3k-splitk-*`.

## Metrics

- prompt eval tok/s on pp gate;
- route activation trace;
- decision whether split-K belongs in future Q3_K route work.

## Result

- Outcome: tie/slight regression; code reverted.
- Delta:
  - baseline direct route: `968.74 tok/s`;
  - forced split-K2 for `k>=17000`: `966.21 tok/s` (`-0.26%`);
  - forced split-K4 for `k>=17000`: `964.46 tok/s` (`-0.44%`).
- Confidence: enough to reject promotion. The result is close to noise, but it is not positive and the mechanism adds temp output, sync, and reduce work.
- Recommendation: do not pursue existing matmul split-K for the hot reverse Q3_K shape. The route is not catastrophic like E138/E139, but it does not move the lane and confirms the shape is not parallelism-starved.

## Notes

- A negative result here should not reject all new Q3_K shaders. It only rejects the existing multi-dispatch split-K topology for the active hot shape.
- Route activation was confirmed:
  - baseline already had a small default split-K route for `m=1024,n=320,k=5120`, unrelated to the hot reverse shape;
  - candidate added `matmul split-k route: ... m=5120|n=1024|k=17408|split_k=2/4|q3k_split_k_forced=1`.
- The analytical expectation matched the result: the hot reverse shape already has about `320` large-tile workgroups before split-K, so adding more K partitions mostly adds partial-output traffic and reduce overhead.
- Build verification after revert:
  - `cmake --build build-vulkan --config Release --target llama-bench -j 8`
  - `cmake --build build-vulkan --config Release --target llama-server -j 8`
- Artifacts:
  - `build_logs/agent-workload/e140-vulkan-q3k-splitk-baseline-pp7488.log`
  - `build_logs/agent-workload/e140-vulkan-q3k-splitk2-k17000-pp7488.log`
  - `build_logs/agent-workload/e140-vulkan-q3k-splitk4-k17000-pp7488.log`
