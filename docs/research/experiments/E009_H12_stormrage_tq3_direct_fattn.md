# E009 - H12 Stormrage TQ3 Direct FlashAttention Recon

## Metadata

- Experiment ID: E009
- Date: 2026-05-13
- Owner: Copilot
- Branch/Commit: master @ 9ef08998a
- Target lane: Qwen3.6-27B prompt-heavy ROCm/RDNA4; secondary KV-cache lane from Stormrage benchmark shape

## Hypothesis

- Statement: The local `tq3_0`/`turbo3` KV lane is slow partly because K/V are dequantized in the attention graph before FlashAttention; adapting Stormrage's direct compressed-KV FlashAttention/WHT idea to local real TurboKV types may recover enough overhead to make the low-VRAM lane useful.
- Mechanism: Store K/V in rotated compressed `TKV2/3/4`; pre-rotate Q with the matching WHT; compute KQ directly from compressed K in FATTN vec kernels; inverse-rotate the attention output when V is compressed. This avoids full K/V dequant graph nodes on every attention call.
- Why now: Stormrage has a working deeper TurboKV design (`GGML_TYPE_TURBO2_0/3_0/4_0`, `GGML_OP_TURBO_WHT`, direct FATTN vec-dot instances), while the local implementation currently logs that `TQ3_0` with FlashAttention dequants K/V in the graph.

## Math / Theory

- Assumptions:
  - WHT/sign rotation is orthogonal, so `<Q, K> = <R(Q), R(K)>` for KQ.
  - Weighted sums over rotated V can be inverse-rotated after attention to recover the original-space output.
  - The first implemented prototype uses local `TKV2/3/4` block size 128 because it matches the Stormrage TurboKV route and avoids conflicting with existing local `TQ3_0` semantics.
- Expected speedup corridor:
  - Projected only: +5% to +40% for compressed TurboKV lanes if graph dequant/cast is a meaningful share of time.
  - No claim yet that it beats `q4_0/q4_0`; first goal is to close the observed `turbo3` gap.
- Failure conditions:
  - FATTN compressed vec-dot cost exceeds saved dequant bandwidth.
  - Output equivalence drifts because Q/V rotation order or scaling does not match `cpy-utils.cuh` and `convert.cu`.
  - Extra template instances make ROCm builds too expensive for default profiles.

## Implementation Plan

1. Minimal code surface to change:
  - Add guarded WHT graph op for the local 128-wide TKV route.
  - Add direct FATTN vec-dot support for `GGML_TYPE_TKV2_0`, `GGML_TYPE_TKV3_0`, and `GGML_TYPE_TKV4_0` same-type K/V combinations.
  - Add limited HIP template instances for D=128 and D=256.
  - Update graph path to avoid TKV graph dequant when the direct path is enabled.
2. Guard rails:
  - Runtime env flag `GGML_TKV_DIRECT_FATTN=0` as explicit fallback switch.
   - Direct local TKV path is default-enabled for eligible same-type K/V lanes with FlashAttention.
   - Compare logits/output against current `tq3_0` dequant path on a tiny deterministic prompt.
3. Rollback path:
   - Disable the guard and remove added template instances/op dispatch.

## Benchmark Plan

- Baseline command: repeat current `turbo3/tq3_0` Stormrage-shape and prompt-heavy lanes with the existing dequant path.
- Candidate command: same commands with guarded direct-FATTN path enabled.
- Number of runs: 1 run for exploration; 3 runs only if the delta is borderline or promising.
- Artifacts path: `build_logs/agent-workload/e009-*`.

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate
- prefill/decode split
- FATTN/dequant graph node timing if instrumentation is enabled
- output/logit equivalence vs current dequant path

## Result

- Outcome: TKV FlashAttention implemented, built, and enabled by default for eligible local TKV K/V combinations. Current default is hybrid for Turbo4: direct decode with F16 dequant + WMMA prefill; `GGML_TKV_DIRECT_PREFILL=1` forces full direct prefill for experiments.
- Delta (smoke): `pp64/tg8`, `turbo4_0/turbo4_0` improved from `186.69/17.09 tok/s` fallback to `227.88/24.82 tok/s`; direct `turbo3_0/turbo3_0` measured `221.67/24.60 tok/s`; direct `turbo2_0/turbo2_0` measured `225.50/25.52 tok/s`.
- Delta (active lane vs q4, corrected best-shape): `v2-review` lane `ctx=12288`, `b=6144`, `ub=1024`, no-reuse, repo-snapshot context, 3 runs: `q4_0/q4_0 = 11.15 TPS`; `turbo4_0/turbo4_0` hybrid default = `10.02 TPS` (`-10.1%`). Full direct prefill probe with `GGML_TKV_DIRECT_PREFILL=1` measured `7.70 TPS`.
- Delta (diagnostic ub192): `q4_0/q4_0 = 9.01 TPS`; direct `turbo4_0 = 6.68 TPS`, direct `turbo3_0 = 6.25 TPS`, direct `turbo2_0 = 6.71 TPS`; fallback `turbo4_0` with `GGML_TKV_DIRECT_FATTN=0` = `3.10 TPS`.
- Confidence: medium for runtime speed behavior (smoke + active-lane A/B). Full logit-level equivalence is still pending.
- Recommendation: keep the hybrid default for Turbo4 and continue kernel/runtime tuning before claiming TurboKV throughput parity or advantage over q4_0 on active lane.

## Notes

- 2026-05-13: Phase 1 correctness path started under the local `TKV` names: `GGML_TYPE_TKV2_0`, `GGML_TYPE_TKV3_0`, `GGML_TYPE_TKV4_0`. This is not the direct-FATTN speed path yet; it makes `turbo2/3/4` real KV cache formats first. `cmake --build build-rocm-vec --target llama-bench --config Release -j` completed successfully.
- 2026-05-13: Phase 2 direct path added: `GGML_OP_TURBO_WHT`, CUDA/HIP WHT backend op, direct TKV FATTN vec helpers, same-type D=128/D=256 instances. Runtime now default-enables direct path for eligible TKV K/V and uses `GGML_TKV_DIRECT_FATTN=0` for fallback.
- 2026-05-13 smoke commands used `--no-warmup -r 1 -p 64 -n 8 -b 128 -ub 128 -fa 1 -fitt 2048 -fitc 4096` on `models/Qwen3.6-27B-Q3_K_S.gguf` with `HSA_OVERRIDE_GFX_VERSION` unset.
- 2026-05-13 active-lane A/B (`v2-review`, no-reuse, repo-snapshot) stored in `build_logs/agent-workload/e009-q4-vs-turbokv-v2review-20260513.md`.
- 2026-05-13 corrected `ub=1024` Turbo4 vs q4 A/B stored in `build_logs/agent-workload/e009-q4-vs-turbo4-ub1024-v2review-20260513.md`.
- Stormrage full TurboKV port is larger because its `GGML_TYPE_TURBO2_0/3_0/4_0` IDs conflict with local `Q1_0/TBQ3_0/TBQ4_0/TQ3_0` layout.
- Stormrage RDNA2 MMQ double-buffer path appears less relevant to the current dense RDNA4 target and should be treated as a separate MoE/IQ4_XS experiment if revisited.
