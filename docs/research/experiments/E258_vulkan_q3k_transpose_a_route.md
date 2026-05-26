# E258 Vulkan Q3_K Transpose-A Route

## Metadata

- Experiment ID: E258
- Date: 2026-05-26
- Owner: Copilot
- Branch/Commit: local dirty tree after E257
- Target lane: Vulkan, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=7168`, `ubatch=1024`, q4/q4 KV, FlashAttention on, `spec=none`, cold/no-reuse/no-prime, thinking on

## Hypothesis

- Statement: a backend-private Q3_K transpose-A layout can improve the heavy-prompt Vulkan route by changing the A-side block access pattern for the dominant dense FFN matmuls.
- Mechanism: upload Q3_K weight blocks in `[k_block,row]` order instead of `[row,k_block]`, then use a dedicated `*_transa` matmul shader. For a fixed K tile, the shader reads the BM rows from a contiguous block run instead of striding by the whole K row.
- Why now: E257 raised the no-code lane to `7.0319 TPS`, but the perf trace is still Q3_K dominated (`MUL_MAT q3_K = 82.71%`). E078/E099 rejected current MMQ/Q8_1/int-dot, and E143/E146/E147 rejected nearby tile/storage-lite routes. A layout/topology route is the remaining high-share path.

## Math / Theory

- Assumptions: active E257 best is `7.0319 TPS`; +20% target is about `8.44 TPS`; the prebuild gate says Q3_K alone needs roughly `+20.81%` local hotspot gain.
- Expected speedup corridor: small wins below `+5%` wall are not enough; a useful result needs either a direct wall jump or a clear Q3_K local proxy gain that can stack with a second topology change.
- Failure conditions: route does not activate, output corrupts, upload path bypasses repack, route falls back to normal `matmul_q3_k_f32_f16acc_aligned_l`, or same-lane cold TPS does not beat E257.

## Implementation Plan

1. Add `GGML_VK_Q3K_TRANSPOSE_A=1` as a default-off env gate.
2. Repack only full Q3_K `set_tensor` uploads and track tensors that were actually repacked.
3. Generate and select Q3_K `matmul_q3_k_f32_transa*` pipelines only for tracked tensors.
4. Teach the Q3_K matvec path to read tracked transposed tensors so decode remains correct.
5. Rollback path: unset `GGML_VK_Q3K_TRANSPOSE_A` or revert the E258 code if correctness/perf fails.

## Benchmark Plan

- Build: `cmake --build build-vulkan --target llama-server llama-bench --config Release -j 8`.
- Route smoke: `GGML_VK_Q3K_TRANSPOSE_A=1 GGML_VK_MATMUL_ROUTE_TRACE=1` with a short Vulkan server/bench run.
- Candidate command: E257 cold lane plus `GGML_VK_Q3K_TRANSPOSE_A=1`.
- Number of runs: `--runs 1` for the first gate; raise only if the result is borderline or promising.
- Artifacts path: `build_logs/agent-workload/e258-*`.

## Metrics

- aggregate completion TPS (wall)
- prompt eval TPS
- decode eval TPS
- route trace pipeline names
- error rate / visible output sanity

## Result

- Outcome: regression after successful route activation.
- Baseline: `e258-vulkan12k-control-postbuild-r1` measured `6.9827 TPS`, prompt `994.75 tok/s`, decode `40.09 tok/s`, errors `0`.
- Candidate: `e258-vulkan12k-q3k-transa-r1` measured `6.9053 TPS`, prompt `1008.34 tok/s`, decode `35.76 tok/s`, errors `0`.
- Delta: wall `-1.11%` vs paired control. Prompt eval improved `+1.37%`, but decode eval regressed `-10.80%`, so task wall lost.
- Route proof: server route trace selected `matmul_q3_k_f32_transa_f16acc_aligned_l` with `q3k_transa=1` on the hot Q3_K shapes, including `m=17408,n=1024,k=5120` and `m=5120,n=1024,k=17408`.
- Confidence: high for rejection. The candidate was active and correctness-clean, and a paired no-env control was taken on the same build.
- Recommendation: reject and remove the source prototype. Do not keep `GGML_VK_Q3K_TRANSPOSE_A` as an opt-in route; it shifts cost from prefill into decode/matvec and does not move the cold wall target.

## Notes

- The implementation was Q3_K-only and default-off. After the negative A/B, the runtime/shader source changes were reverted.
- SPIR-V opcode summaries for the transposed and baseline Q3_K aligned f16acc shaders were nearly identical (`OpCooperativeMatrixLoadKHR=2`, `OpCooperativeMatrixMulAddKHR=1`, `OpControlBarrier=6` for both), so the loss is attributed to layout/route behavior, especially decode, rather than a large shader-complexity accident.
- Device-memory headroom was not the limiter: the candidate projected `12164 MiB` of device use against `15222 MiB` free.
