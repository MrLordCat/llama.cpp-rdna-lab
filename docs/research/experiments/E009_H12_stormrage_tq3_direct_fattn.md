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

- Outcome: TKV FlashAttention implemented, built, and enabled by default for eligible local TKV K/V combinations, including explicit mixed TKV/Q8 selections. Current default is hybrid for Turbo4: direct decode with F16 dequant + WMMA prefill; `GGML_TKV_DIRECT_PREFILL=1` forces full direct prefill for experiments.
- Delta (smoke): `pp64/tg8`, `turbo4_0/turbo4_0` improved from `186.69/17.09 tok/s` fallback to `227.88/24.82 tok/s`; direct `turbo3_0/turbo3_0` measured `221.67/24.60 tok/s`; direct `turbo2_0/turbo2_0` measured `225.50/25.52 tok/s`.
- Delta (active lane vs q4, corrected best-shape): `v2-review` lane `ctx=12288`, `b=6144`, `ub=1024`, no-reuse, repo-snapshot context, 3 runs: `q4_0/q4_0 = 11.15 TPS`; `turbo4_0/turbo4_0` hybrid default = `10.02 TPS` (`-10.1%`). Full direct prefill probe with `GGML_TKV_DIRECT_PREFILL=1` measured `7.70 TPS`.
- Delta (follow-up set_rows + mixed route): specialized `TKV4 set_rows` improved same-type `turbo4_0/turbo4_0` to `10.38 TPS` (`-7.1%` vs `q4_0=11.17`). Mixed `turbo4_0/q8_0` direct decode with F16 prefill measured `10.60 TPS` over 3 runs (`-5.1%` vs q4, `303 MiB` KV); reverse `q8_0/turbo4_0` smoke measured `10.26 TPS` over 1 run.
- Negative control: mixed `turbo4_0/q8_0` with `GGML_TKV_DIRECT_FATTN=0` completed through F16 fallback at `4.51 TPS` r1, confirming the fallback path is safe but not performance-competitive.
- Stormrage-shape recheck: local current `run_rdna2_bench.sh` shape (`p=512,2048,4096`, `n=128`, `b=256`, `ub=128`, `ctk=turbo4`, `ctv=turbo2`, `fa=1`, `fit-target=2048`, `fitc=4096`, `r=3`) measured dense27B `TKV4/TKV2 = 636.45/608.08/554.85 pp, 20.49 tg128` and MoE35B `TKV4/TKV2 = 1143.86/1064.55/992.07 pp, 56.71 tg128` on RX 9070 XT. Same-shape local `q4_0/q4_0` stayed faster (`795.66/787.07/776.22 pp, 28.59 tg128` dense; `1318.83/1275.92/1239.98 pp, 102.76 tg128` MoE). Stormrage README numbers are RX 6800 XT/RDNA2 and not direct pass/fail targets.
- Extra Stormrage-shape `b=1024,ub=1024`: dense27B measured `q4_0/q4_0 = 1079.38/1244.60/1225.79 pp, 28.85 tg128`, `TKV4/TKV4 = 1006.08/1172.52/1135.15 pp, 20.95 tg128`, and `TKV4/TKV2 = 997.35/1168.99/1133.96 pp, 20.78 tg128`. MoE35B measured `q4_0/q4_0 = 2807.61/3549.80/3500.76 pp, 102.50 tg128` and `TKV4/TKV2 = 2590.18/3290.59/3182.46 pp, 56.28 tg128`. Large `ubatch` is useful for MoE prefill, but q4 remains the local speed baseline.
- MoE accelerator assessment: Stormrage `RDNA2_MATMUL_OPT_V1` in external `ggml/src/ggml-cuda/mmq.cuh` is compile/env gated and runtime-limited to `GGML_CUDA_CC_IS_RDNA2(cc)`. It should be treated as a separate RDNA4 MoE/MMQ experiment with opt-in gating and dense negative control; not a direct carry-over from the TurboKV storage/FATTN work.
- Delta (diagnostic ub192): `q4_0/q4_0 = 9.01 TPS`; direct `turbo4_0 = 6.68 TPS`, direct `turbo3_0 = 6.25 TPS`, direct `turbo2_0 = 6.71 TPS`; fallback `turbo4_0` with `GGML_TKV_DIRECT_FATTN=0` = `3.10 TPS`.
- Confidence: medium for runtime speed behavior (smoke + active-lane A/B). Full logit-level equivalence is still pending.
- Recommendation: keep the hybrid default for Turbo4, keep mixed `turbo4_0/q8_0` as explicit opt-in, and continue kernel/runtime tuning before claiming TurboKV throughput parity or advantage over q4_0 on active lane.

## Notes

- 2026-05-13: Phase 1 correctness path started under the local `TKV` names: `GGML_TYPE_TKV2_0`, `GGML_TYPE_TKV3_0`, `GGML_TYPE_TKV4_0`. This is not the direct-FATTN speed path yet; it makes `turbo2/3/4` real KV cache formats first. `cmake --build build-rocm-vec --target llama-bench --config Release -j` completed successfully.
- 2026-05-13: Phase 2 direct path added: `GGML_OP_TURBO_WHT`, CUDA/HIP WHT backend op, direct TKV FATTN vec helpers, same-type D=128/D=256 instances. Runtime now default-enables direct path for eligible TKV K/V and uses `GGML_TKV_DIRECT_FATTN=0` for fallback.
- 2026-05-13 smoke commands used `--no-warmup -r 1 -p 64 -n 8 -b 128 -ub 128 -fa 1 -fitt 2048 -fitc 4096` on `models/Qwen3.6-27B-Q3_K_S.gguf` with `HSA_OVERRIDE_GFX_VERSION` unset.
- 2026-05-13 active-lane A/B (`v2-review`, no-reuse, repo-snapshot) stored in `build_logs/agent-workload/e009-q4-vs-turbokv-v2review-20260513.md`.
- 2026-05-13 corrected `ub=1024` Turbo4 vs q4 A/B stored in `build_logs/agent-workload/e009-q4-vs-turbo4-ub1024-v2review-20260513.md`.
- 2026-05-13 specialized `TKV4 set_rows` follow-up stored in `build_logs/agent-workload/e013-tkv4setrows-finalstable-*.{jsonl,csv,diagnostics.md,server.log}`.
- 2026-05-13 mixed TKV/Q8 route follow-up stored in `build_logs/agent-workload/e015-mixedroute-*.{jsonl,csv,diagnostics.md,server.log}`.
- 2026-05-13 Stormrage-shape recheck stored in `build_logs/agent-workload/stormrage-shape-current-*-20260513.jsonl`.
- Stormrage full TurboKV port is larger because its `GGML_TYPE_TURBO2_0/3_0/4_0` IDs conflict with local `Q1_0/TBQ3_0/TBQ4_0/TQ3_0` layout.
- Stormrage RDNA2 MoE LDS double-buffer path remains the only clearly unported performance idea, but it targets gfx1030 MoE prefill and should be treated as a separate MoE/IQ4_XS experiment if revisited on RDNA4.
