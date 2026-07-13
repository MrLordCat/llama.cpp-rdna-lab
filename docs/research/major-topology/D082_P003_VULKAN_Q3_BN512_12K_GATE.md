# D082 P003 Vulkan Q3 BN512 12k Gate

## Status

- Rejected and removed from runtime source.
- Active lane: non-MTP `Qwen3.6-27B-Q3_K_S.gguf`, Vulkan dual layer split,
  `ctx=12288`, about 7.4k prompt tokens, `b=1024`, `ub=1024`, q8/q8 KV,
  `-ts 5,6`, no reuse, no prime, no mmap.
- Default `bn256` route remains unchanged.

## Measured Baseline

- Production-like `max_tokens=16`: `1821.13 prompt tok/s`.
- Resource control `max_tokens=1`: `1840.55 prompt tok/s`.
- Target: `2000 prompt tok/s`.
- Required total speedup from the production-like baseline: `1.0982x`.

## 12k Trace

Parsed prompt GPU time:

- Q3_K matmul: `67.79%`.
- FlashAttention: `8.33%`.
- Q4_K matmul: `5.67%`.
- Two hot Q3_K shapes: `48.80%` of parsed prompt time.

Required local speedup:

- all Q3_K: about `1.152x`;
- the two hot Q3_K shapes alone: about `1.224x`.

FA cannot close this lane alone: eliminating all measured FA time projects only
about `1986 prompt tok/s`.

## Candidate

Add `GGML_VK_AMD_LARGE_MATMUL_VARIANT=bn512` as an opt-in extension of the
accepted AMD large Q3_K matmul route.

Current `bn256`:

- prepared workgroup: 512 threads;
- measured resources: 82 VGPR, 44 SGPR, 31,744 B LDS, zero scratch;
- four N workgroups for `N=1024`.

Projected `bn512`:

- prepared workgroup: 1024 threads;
- projected LDS: 54,272 B;
- two N workgroups for `N=1024`;
- half the repeated Q3 A-dequant and K-loop barrier rounds;
- unchanged aggregate B load proxy.

Unlike the rejected E137 `NITER=2` route, this does not add a second accumulator
set per invocation or modify the default shader. It expands the cooperative
workgroup and shares the A tile across more warps.

## Result

- The prepared BN512 layout needs about 54 KiB LDS, above the current Vulkan
  device's usable 32 KiB workgroup limit for this route.
- Runtime therefore retained the medium Q3 pipeline; the candidate did not
  become the active hot-shape route.
- Measured prompt speed was `950.35 tok/s`, far below the `1821.13 tok/s`
  production-like control.
- The prototype was removed. Reopen only with a different dataflow that stays
  within the device LDS limit; changing the tile label alone is not sufficient.

## Gates

1. Static scout must report valid layout and load mapping. **Passed.**
2. Shader must compile with at most 100 VGPR, no scratch, and no more than
   56 KiB LDS.
3. Route trace must prove the hot Q3_K shapes select the candidate pipeline.
   **Failed.**
4. A short deterministic response must complete without invalid output.
5. First `max_tokens=16` gate must beat `1821.13` by at least 3%; otherwise
   reject without r3.
6. Promote only after a repeated cold-first result reaches or approaches 2000.

## Rollback

The runtime branches were removed. The static scout row remains as a rejection
record so the same over-LDS design is not repeated.
