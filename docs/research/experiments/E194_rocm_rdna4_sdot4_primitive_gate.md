# E194 ROCm RDNA4 Sdot4 Primitive Gate

## Metadata

- Experiment ID: E194
- Date: 2026-05-23
- Owner: Codex
- Hypothesis ID: H39
- Branch/Commit: temporary local patch after `624a8d163`
- Target lane: L1/H39 ROCm, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on, `quick/triage_diff`, `max_tokens=128`

## Hypothesis

- Statement: On RDNA4, using the signed `__builtin_amdgcn_sdot4` backend for `ggml_cuda_dp4a` may improve the quantized Q3_K/Q4_K matvec route versus the current `__builtin_amdgcn_sudot4(true, true, ...)` path.
- Mechanism: H39 is dominated by Q3_K `mul_mat_vec_q` direct/fused kernels. Those kernels call `ggml_cuda_dp4a` for every packed int8 dot. Changing this primitive affects the whole quant matvec branch at once, unlike prior nearby warp/tile/layout probes.
- Why now: E178-E180 and E190 rejected local selector/layout/y-reuse probes. A primitive-level gate is a larger branch and can be rejected cheaply by compile + one same-lane runtime gate.

## Math / Theory

- Assumptions:
  - E190 paired control r3: `12.9580 TPS`, decode `31.4433 tok/s`.
  - E151/E177 show Q3_K matvec remains a large decode route share.
  - If the compiler lowers `sudot4(true,true)` and `sdot4` identically on gfx1201, runtime should tie.
- Expected speedup corridor:
  - `+1%..+4%` wall if RDNA4 signed dot codegen is materially better for Q3_K.
  - Compile failure or neutral/negative r1 is enough to reject.
- Failure conditions:
  - `sdot4` unavailable or incorrectly lowered on gfx1201,
  - output corruption,
  - prompt/decode regression from changed codegen or instruction scheduling.

## Implementation Plan

1. Minimal code surface to change: temporary `ggml/src/ggml-cuda/common.cuh` patch in `ggml_cuda_dp4a`.
2. Guard rails: no default claim until build + same-lane runtime + live-output sanity if positive.
3. Rollback path: revert the temporary primitive patch if compile or runtime gate fails.

## Benchmark Plan

- Build gate: `cmake --build build-rocm-vec --config Release -j`.
- Baseline command: E193 same-session control or a fresh `e194-l1-control-r1` if build changes require it.
- Candidate command: `e194-l1-sdot4-r1`.
- Number of runs: `r1` gate, `r3` only if positive.
- Artifacts path: `build_logs/agent-workload/e194-l1-*.{csv,jsonl,server.log,diagnostics.md}`.

## Metrics

- aggregate completion TPS (wall)
- prompt eval tok/s and ms
- decode eval tok/s and ms
- error rate
- output sanity if promoted beyond r1

## Result

- Outcome: reject at build gate; no runtime benchmark.
- Delta: no TPS delta. The temporary source patch fails HIP compilation for `gfx1201`.
- Confidence: high for current toolchain/build contract. ROCm clang 7.1 reports that `__builtin_amdgcn_sdot4` needs target feature `dot1-insts` when compiling the local `gfx1201` build.
- Recommendation: keep the existing RDNA4 `__builtin_amdgcn_sudot4(true, true, ...)` primitive. Do not replace it with `sdot4` in source unless a separate toolchain-feature gate proves `dot1-insts` is supported and enabled for this target, followed by a normal correctness/runtime A/B.

## Build Gate

Temporary patch tested:

```diff
-#elif defined(RDNA3) || defined(RDNA4)
+#elif defined(RDNA4)
+    c = __builtin_amdgcn_sdot4(a, b, c, false);
+#elif defined(RDNA3)
     c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);
```

Build command:

```powershell
cmake --build build-rocm-vec --config Release -j 2 2>&1 |
  Tee-Object -FilePath build_logs\agent-workload\e194-sdot4-build.log
```

Key failure:

```text
common.cuh:673:9: error: '__builtin_amdgcn_sdot4' needs target feature dot1-insts
579 warnings and 1 error generated when compiling for gfx1201.
```

Artifact:

- `build_logs/agent-workload/e194-sdot4-build.log`

Rollback:

- The temporary `common.cuh` patch was reverted.
- `cmake --build build-rocm-vec --config Release -j` passed after the revert.

## Notes

- Surprises: the primitive-level idea is blocked before runtime; the compiler distinguishes the signed dot intrinsic feature from the current `gfx1201` build flags, while the existing `sudot4(true,true)` path remains buildable.
- Follow-up action: if this branch is revisited, treat it as a compiler/target-feature experiment first. It should not be part of the active H39 route-body queue until the feature gate is proven and output sanity can be checked.
