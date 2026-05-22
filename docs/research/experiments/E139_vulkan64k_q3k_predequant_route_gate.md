# E139 Vulkan 64k Q3_K Predequant Route Gate

## Metadata

- Experiment ID: E139
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E138 (`a1179a579`)
- Target lane: Vulkan 64k, `Qwen3.6-27B-Q3_K_S`, q4/q4 KV, FlashAttention on, `b8192/ub1024`, `--no-mmap`, `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`, `spec=none`, no reuse

## Hypothesis

- Statement: force hot large Q3_K Vulkan matmuls through the existing backend predequant route (`Q3_K -> fp16 prealloc_x -> f16 matmul`) to test whether a backend-private repack/predequant branch has enough structural ceiling to replace direct `matmul_q3_k_f32_f16acc_aligned_l`.
- Mechanism: direct Q3_K matmul repeats A-side dequant work across N tiles. A predequant route removes quant unpack/dequant from the matmul shader and shifts the main work to f16 matmul.
- Why now: E137 rejected dual-N reuse inside `mul_mm.comp` because accumulator pressure regressed performance. A separate route branch can test A-side work reduction without perturbing the default Q3_K shader fingerprint.

## Math / Theory

- Active 64k Vulkan best: `1.3406 TPS`.
- ROCm 64k same-lane target: `1.5545 TPS`.
- Required wall speedup: `1.1596x`.
- E134 Q3_K share proxy: `0.5228`, so Q3_K alone needs about `1.357x` local speedup to close the full lane.
- Prebuild gate with an optimistic `20%` local Q3_K gain projects only `1.0955x` total, so this route probably needs to stack with FA or another Q3_K branch.
- Hot shape fp16 temp size:
  - `m=17408,k=5120`: `17408 * 5120 * 2 = 178257920 B`, about `170.0 MiB`.
  - `m=5120,k=17408`: same fp16 temp size.
- Main failure condition: existing fallback inserts a dequant dispatch plus `ggml_vk_sync_buffers()` before matmul. If sync/temp traffic dominates, this route is an anti-pattern even if it reduces in-shader Q3_K arithmetic.

## Implementation Plan

1. Add a default-off host env gate in `ggml_vk_mul_mat_q_f16`.
2. Scope it to `src0=Q3_K` and large matmul dimensions.
3. Force direct Q3_K pipeline selection to fall through to the existing `qx_needs_dequant` path.
4. Trace route activation with `GGML_VK_MATMUL_ROUTE_TRACE=1`.
5. Revert the code if the short prompt gate regresses or proves sync/temp overhead dominates.

## Benchmark Plan

- Baseline command: short Vulkan prompt gate on the clean route using `llama-bench` `-p 7488`, plus a real-server prompt-screen if pp gate is not catastrophic.
- Candidate command: same, with `GGML_VK_Q3K_FORCE_PREDEQUANT=1`.
- Number of runs: `1` for gate; escalate only if candidate is near or above baseline.
- Artifacts path: `build_logs/agent-workload/e139-vulkan-q3k-predequant-*`.

## Metrics

- prompt eval tok/s on pp gate;
- route trace: direct Q3_K vs f16 fallback;
- fallback temp/sync evidence;
- aggregate/full lane only if the gate is positive.

## Result

- Outcome: regression; code reverted.
- Delta:
  - baseline direct Q3_K route: `969.61 tok/s`;
  - force all large Q3_K shapes: `743.65 tok/s` (`-23.30%`);
  - force only `m>=17000` gate/up shape: `832.27 tok/s` (`-14.16%`);
  - force only `k>=17000` reverse shape: `929.40 tok/s` (`-4.15%`);
  - reverse-shape stats run: `927.89 tok/s`.
- Confidence: high enough to reject the existing per-node predequant fallback before a full 64k real-server run.
- Recommendation: do not promote or keep this env gate. Future Q3_K repack/layout work must avoid the existing `prealloc_x` fp16 temp plus `ggml_vk_sync_buffers()` route; it needs either a direct/single-dispatch shader or a graph-safe persistent layout that does not serialize every hot matmul.

## Notes

- This experiment is intentionally a route-level gate. A negative result should not kill all Q3_K repack/layout ideas; it specifically rejects the existing per-node fp16 temp fallback as the implementation strategy.
- Route activation was confirmed:
  - baseline hot shapes used `matmul_q3_k_f32_f16acc_aligned_l`, `qx_dequant=0`;
  - candidate hot shapes used `matmul_f16_f32_f16acc_aligned_l`, `qx_dequant=1`.
- The f16 fallback pipeline itself is not resource-heavy: `77 VGPR`, `44 SGPR`, `22528 B LDS`, `0 scratch`. The regression therefore points at route overhead rather than f16 matmul register pressure: a `~170 MiB` fp16 temp for each top hot shape, dequant dispatch, sync boundary, and extra global write/read traffic.
- Build verification:
  - `cmake --build build-vulkan --config Release --target llama-bench -j 8`
  - `cmake --build build-vulkan --config Release --target llama-server -j 8`
- Analytical gates run:
  - `python scripts\research\formula_sanity_checks.py`
  - `python scripts\research\vulkan_q3k_prebuild_gate.py --candidate "...predequant..." --baseline-pp 974.92 --goal-total-speedup 1.1596 --target-share 0.5228 --local-gain-pct 20`
  - `python scripts\research\required_acceptance.py --target-wall 1.1596 --draft-len 4 --prefill-share 0.9387 --prefill-speedup 1.172 --spec-overhead 0.0`
  - `python scripts\research\speedup_model.py --baseline-tps 1.3406 --prefill-share 0.9387 --flash-prefill-speedup 1.172 --draft-len 4 --accept-rate 0.0 --spec-overhead 0.0`
- Artifacts:
  - `build_logs/agent-workload/e139-vulkan-q3k-predequant-baseline-pp7488.log`
  - `build_logs/agent-workload/e139-vulkan-q3k-predequant-candidate-pp7488.log`
  - `build_logs/agent-workload/e139-vulkan-q3k-predequant-m17000-pp7488.log`
  - `build_logs/agent-workload/e139-vulkan-q3k-predequant-k17000-pp7488.log`
  - `build_logs/agent-workload/e139-vulkan-q3k-predequant-k17000-stats-pp7488.log`
