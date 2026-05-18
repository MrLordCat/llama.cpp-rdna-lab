# E059 External RDNA4 llama.cpp Research

## Metadata

- Experiment ID: E059
- Date: 2026-05-18
- Owner: Codex
- Branch/Commit: local `master`, research-only, no code change
- Target lane: Qwen3.6-27B-Q3_K_S cold-first prefill lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on.

## Hypothesis

- Statement: external RDNA4/llama.cpp work may reveal a non-local acceleration point after E053-E058 exhausted simple local Q3_K conversion/runtime probes.
- Mechanism: compare upstream RDNA4 selector work, independent gfx12 WMMA/MMQ projects, ROCm library tuning, and Vulkan/RDNA4 reports against the exact local hot lane before writing more kernel code.
- Why now: local search plateaued after E054 proved Q3_K `src0` is conversion/store, while half2/unroll4 and runtime knobs failed the new small-gain policy.

## External Findings

### Upstream llama.cpp RDNA4 MMQ selector work

- `ggml-org/llama.cpp#18537` merged RDNA/RDNA3 MMQ switching changes and originally left RDNA4 as `return true` for MMQ, based on a comment that RDNA4 was worse on rocBLAS for the tested cases.
- `ggml-org/llama.cpp#18816` is still open and narrows RDNA4 routing instead of keeping unconditional MMQ:
  - `Q3_K`, `Q2_K`, `Q4_K`, `Q5_K`, selected IQ types: MMQ only for `ne11 <= 256`.
  - `Q6_K` and `IQ2_S`: MMQ only for `ne11 <= 128`.
  - `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `MXFP4`: MMQ always.
  - `Q8_0`, `IQ4_NL`, `IQ4_XS`: MMQ for `ne11 <= 512`.
  - It also adds an RDNA4 MMVQ gate for some IQ types, capping them at `ncols_dst <= 4`.
- Reviewer evidence in #18816 is mixed: RX 9060 XT `Q6_K` default settings showed severe regressions at `n_ubatch=256/512`, while the author's R9700 data showed different behavior. The PR is not merged and should not be treated as a safe default.
- Local relevance: our current lane is `Q3_K` with `ne11=2048`. Both #18816 and local E050 reject MMQ for that exact large shape. E050 measured the target `type=11,nrows_x=6144,ncols_max=2048` as `2529.35 ms` on MMQ versus `1839.27 ms` on the current cuBLAS split path, a `+37.52%` local regression.

### Local selector discrepancy

- Local `ggml/src/ggml-cuda/mmq.cu` already has an RDNA4-specific selector, but it differs from #18816:
  - local keeps `n_experts >= 64 -> true`, while #18816's final diff removed that special case for RDNA4;
  - local caps `Q4_0/Q4_1/Q5_0/Q5_1` at `ne11 <= 256`, while #18816 routes them to MMQ unconditionally;
  - local caps `Q2_K/Q3_K/Q4_K/Q5_K/Q6_K` at `ne11 <= 192`, while #18816 splits them into `Q6_K <= 128` and `Q2/Q3/Q4/Q5_K <= 256`;
  - local default is `ne11 <= 128`, while #18816 has `Q8_0/IQ4_* <= 512` and `default false`.
- This is actionable only as a narrow selector audit, not as a current-lane Q3_K `ne11=2048` MMQ retry.

### hipfire gfx12/R9700 WMMA and paging work

- `Kaden-Schutt/hipfire#45` reports gfx1201/R9700 WMMA work with large prefill gains in that engine, including roughly `+27.8%` to `+42.4%` prefill for 27B scenarios and later HFQ4 MMQ auto-dispatch at `batch_size >= 256`.
- The useful technical lesson is that gfx11 -> gfx12 WMMA is not a macro swap: A/B vector width, K packing, and K dimension thread mapping changed. That makes it a design reference, not a drop-in llama.cpp patch.
- `Kaden-Schutt/hipfire#77` is MoE/NVMe paging oriented. The hot-path lesson is still useful: avoid `hipDeviceSynchronize` in repeated scheduling paths; same-stream `hipMemcpyAsync` can serialize without a global sync. Current E054 Q3_K conversion is not dominated by that issue.
- Local relevance: hipfire suggests a high-ceiling gfx12 quantized prefill design may exist, but local E050 shows llama.cpp's current Q3_K MMQ is slower than dequant+GEMM at `ne11=2048`. Any new direct kernel must be materially different from the current MMQ path.

### TileKernels, AITER, FP8, and MoE leads

- `kmbandy/llama.cpp#6` discusses DeepSeek TileKernels FP8/MoE ROCm/HIP ideas: FP8 KV, MoE routing, fused top-k/expand/reduce, and wave32 assumptions.
- ROCm AITER discussions show gfx1201 support activity, but not a ready dense Q3_K llama.cpp patch.
- Local relevance: these are future tracks for MoE/FP8/AWQ or MTP-enabled models, not the current dense Qwen3.6-27B Q3_K_S lane.

### Vulkan/RADV and ZINC reports

- `ggml-org/llama.cpp#22898` reports RDNA4 Vulkan/RADV long-context generation collapse; `RADV_PERFTEST=nogttspill` changed persistent collapse into temporary slowdown. ROCm text-only remained stable around the reported 26.9-27 tok/s in that issue.
- ZINC's Vulkan engine discussion targets fewer CPU/GPU round trips and GPU-side MoE scheduling. It is architecturally interesting, but not a proven speed source for the current ROCm lane.
- Local relevance: useful fallback/debug information for Vulkan, not a ROCm Q3_K prefill optimization.

### hipBLASLt and Stream-K tuning

- ROCm docs expose offline hipBLASLt tuning via `HIPBLASLT_LOG_MASK=32`, `HIPBLASLT_TUNING_FILE`, `HIPBLASLT_TUNING_OVERRIDE_FILE`, and workspace controls.
- Stream-K knobs (`TENSILE_SOLUTION_SELECTION_METHOD=2`, dynamic/fixed grid controls) are meant for uneven GEMM utilization.
- Local relevance: E048/E052/E058 already tested simple env gates. E058 saw only `+0.42%` aggregate for hipBLASLt with weaker median evidence, so this remains watchlist-only. The only remaining version worth testing would be exact-shape offline tuning, and only after confirming it touches the current large GEMMs.

### ROCm profiler counters

- ROCm/rocprof reports for gfx1201 show some counters returning zero. Treat ROCm counters as advisory only for this hardware.
- Local relevance: continue using wall TPS, route traces, and synchronized timing instrumentation for gates; do not make keep/revert decisions from suspect counters.

## Math / Theory

- E053/E054 still set the local ceiling: Q3_K conversion/store is the largest concrete measured target after route/runtime candidates failed.
- E050 closes the tempting large-prefill MMQ route: a same-shape `37.52%` local regression means upstream selector talk cannot justify another broad MMQ run on `Q3_K ne11=2048`.
- Selector parity has a smaller ceiling on the current lane because #18816 changes Q3_K only around `ne11=193..256`, not `2048`. It may matter for other local lanes or if a trace shows base `Q4_0/Q5_0/Q8_0/IQ4_*` shapes that local currently routes differently.
- A new gfx12 direct quantized prefill kernel has the highest theoretical ceiling, but it must beat both current dequant+GEMM and the existing llama.cpp MMQ path. That requires a design gate before code.

## Ranked Leads

1. Keep P1 on Q3_K conversion/layout, not MMQ routing. Internet search did not reveal a proven llama.cpp/RDNA4 fix for dense Q3_K large-prefill conversion.
2. Add H28: RDNA4 selector parity audit against #18816. First step is route/log analysis and a tiny llama-bench-style matrix, not `GGML_CUDA_FORCE_MMQ_RUNTIME=1`.
3. Add H29: gfx12 WMMA direct quantized prefill design study inspired by hipfire. Treat as high-risk/high-effort until an analytic gate shows a path to beating E049/E050 timings.
4. Keep H21 hipBLASLt as watchlist-only. Exact-shape offline tuning is the only unresolved library route idea, but the simple env path is not strong enough.
5. Keep TileKernels/AITER/FP8/MoE/ZINC leads outside the current dense Q3_K target unless the benchmark lane changes.

## Validation Plan

1. For H28, parse current trace/cublas/MMQ logs and list shapes where local RDNA4 selector differs from #18816. Ignore Q3_K `ne11=2048` because E050 already measured it negative.
2. If an affected shape has non-trivial wall share, run a narrow env-gated or patch-gated r1 screen for only that type/shape. Promote to r3 only for `>=0.5%` aggregate or clear target timing improvement under the small-gain policy.
3. For H29, write a design gate before code: current timings to beat are E049 target cuBLAS split `1839.27 ms` and E050 MMQ `2529.35 ms` for `type=11,nrows_x=6144,ncols_max=2048`. Reject any direct-kernel idea that does not plausibly change packing/K mapping enough to beat these.
4. Do not rerun broad MMQ force or more generic hipBLASLt/Stream-K env sweeps.

## Result

- Outcome: research completed, no code-speed claim.
- Delta: no TPS delta; all conclusions are source review plus local evidence reconciliation.
- Confidence: high that broad Q3_K `ne11=2048` MMQ remains closed; medium that selector parity may find small stackable wins on other shapes; low-to-medium that a new gfx12 direct quantized prefill kernel is worth implementing without a stronger design gate.
- Recommendation: proceed with H28 as the cheap next validation, then return to Q3_K conversion/layout or H29 design only if H28 finds an activated shape.

## Notes

- Surprise: #18816 does not actually support unconditional RDNA4 MMQ; its final selector still routes current-lane `Q3_K ne11=2048` away from MMQ.
- Follow-up action: create a small selector-diff trace audit before any benchmark run.
