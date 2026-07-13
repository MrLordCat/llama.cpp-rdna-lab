# E279 Vulkan Batched Recurrent Checkpoint

## Metadata

- Experiment ID: E279
- Date: 2026-07-11
- Owner: Codex
- Branch/Commit: `research/cherry-pick-upstream`
- Target lane: dual RX 9070 XT Vulkan, Qwen3.6-27B, reused 98k context with incremental prompt tails

## Hypothesis

- Statement: Recurrent checkpoint serialization contributes removable fixed overhead to incremental prompts, but must be separated from pending graph work and long-KV attention.
- Mechanism: A `149.626 MiB` checkpoint queues about 98 R/S tensor reads through synchronous `ggml_backend_tensor_get()` calls. Batching reads per Vulkan backend should replace per-tensor submits/synchronizations with one transfer submit and synchronization per GPU.
- Why now: The live server log shows 26-40 token tails collapsing toward decode throughput while all model layers, KV, and recurrent state remain GPU-resident.

## Math / Theory

- Assumptions: Qwen3.6 has 49 recurrent layers and each contributes one R and one S tensor read; both GPUs support the existing Vulkan transfer queue.
- Expected speedup corridor: reduce the isolated incremental checkpoint transfer by `10-25%`; total small-tail prompt latency should improve by several percent when the KV is not yet dominant.
- Failure conditions: checkpoint bytes differ, rollback output changes, transfer remains within noise of the synchronous control, or Host/staging residency grows without a useful wall-time reduction.

## Implementation Plan

1. Add an optional backend batch tensor-read interface with a synchronous fallback.
2. Implement Vulkan batching with one staging buffer and one synchronization per backend.
3. Group `llama_io_write_buffer` tensor reads by the owning context backend.
4. Add env-gated checkpoint timing around size, allocation, and state serialization.
5. Add GUI controls for prompt-cache RAM, checkpoint count, and checkpoint interval.

Guard rails: default state format and byte layout remain unchanged; non-Vulkan backends keep the existing fallback; timing is disabled by default.

Rollback path: remove the Vulkan batch callback or set `LLAMA_CHECKPOINT_BATCH_READ=0` to force the existing sequential route.

## Benchmark Plan

- Baseline command: current Vulkan server with `LLAMA_CHECKPOINT_BATCH_READ=0`, fixed long prefix, then 40/256/1024-token incremental tails.
- Candidate command: identical command with the default batched route.
- Number of runs: one diagnostic gate, then three only if the wall delta is promising.
- Artifacts path: `build_logs/agent-workload/e279-*`

## Metrics

- checkpoint size and serialization wall time
- prompt processing wall time and prompt tok/s
- output equality after checkpoint rollback
- process working set/private bytes
- errors and server stability

## Result

- Outcome: keep
- Delta: on a reused 7k prefix with a 15-token incremental tail, sequential checkpoint transfer took `33.362 ms` and the full prompt took `180.553 ms` (`83.08 tok/s`). Batched Vulkan transfer took `27.376 ms`, and the prompt took `165.787 ms` (`90.48 tok/s`). This is `-17.9%` checkpoint transfer time, `-8.2%` prompt wall, and `+8.9%` prompt TPS.
- Confidence: high for correctness and activation, medium for the exact speed delta. The trace reports 96 tensors grouped as 50/46 across the two Vulkan backends. A deterministic branch rollback produced identical output on sequential and batched paths.
- Recommendation: keep the batched callback and env rollback `LLAMA_CHECKPOINT_BATCH_READ=0`. Keep GUI defaults `--cache-ram 0 --ctx-checkpoints 4 --checkpoint-every-n-tokens -1` for the local single-slot workflow. Continue H63 for the dominant 98k small-N long-KV attention cost.

## Notes

- `--cache-ram 0` reduces the separate idle prompt cache but does not remove recurrent checkpoints.
- Reducing `--ctx-checkpoints` lowers retained RAM but does not remove the per-request checkpoint transfer.
- The first long-prefill trace attributed `757-790 ms` to checkpoint save because the first backend read also synchronized pending 1024-token prompt compute. Batched and sequential paths tied there, proving that value was not raw checkpoint transfer.
- Artifacts: `e279-incremental-sync-r2.server.log`, `e279-incremental-batch-r2.server.log`, `e279-rollback-sync-r1.server.log`, and `e279-rollback-batch-r1.server.log`.
