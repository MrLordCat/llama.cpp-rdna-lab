# E293 ROCm RDNA4 rocWMMA FlashAttention Restore

## Metadata

- Date: 2026-07-14
- Target: Qwen3.6-27B Q3_K_S ROCm prompt evaluation on dual RX 9070 XT
- Production rollback: configure with `-DGGML_HIP_ROCWMMA_FATTN=OFF`

## Root Cause

The current `build-rocm-full` and `build-rocm-vec` caches had
`GGML_HIP_ROCWMMA_FATTN=OFF` after the Windows reinstall. Qwen3.6 uses
`D=256` attention heads. With rocWMMA disabled, the full RDNA4 selector cannot
take its `Q0 <= 128` native MMA branch and falls through to the generic tile
kernel for prompt evaluation.

The older Qwen reduced experiment corridor explicitly configured rocWMMA, but
that requirement was not carried into fresh production build defaults. The
Windows ROCm SDK also does not expose the vendored headers through its normal
include path, so merely toggling the GUI option was insufficient without a
manual include-directory flag.

## Implementation

- Enabled `GGML_HIP_ROCWMMA_FATTN` by default for HIP builds.
- Added automatic discovery of versioned headers under
  `third_party/rocwmma/*/library/include`, after explicit and ROCm SDK paths.
- Made the GUI ROCm build option default-on while preserving an explicit OFF
  rollback.
- Kept the full production FlashAttention source matrix; the reduced profile
  was used only as the first isolated proof.

## Route Proof

A low-overhead configuration trace on the full production build recorded 192
rocWMMA launches for an 11,584-token prompt:

- 176 launches: `D=256`, `q_rows=1024`, `selected_cols=16`;
- 16 launches: `D=256`, `q_rows=320`, `selected_cols=16`.

All traced Q/K/V pointers were 128-byte aligned. The full and reduced profiles
selected the same kernel geometry.

## Performance

Common settings: dual layer split `ROCm1,ROCm0`, `-ts 1,1`, q8_0/q8_0 KV,
`b8192/ub1024`, FlashAttention on, no cache reuse, no prime pass, spec none.

### 11,561-token prompt, 16 output tokens, r3

| Metric | Generic tile control | Full rocWMMA | Delta |
| --- | ---: | ---: | ---: |
| Prompt TPS | 1713.61 | 1930.26 | +12.64% |
| Decode TPS | 28.02 | 30.71 | +9.60% |
| Aggregate TPS | 2.1696 | 2.4403 | +12.48% |
| Mean server time | 7330.71 ms | 6524.40 ms | -11.00% |

### 30,075-token prompt, 16 output tokens, r1

| Metric | Generic tile control | Full rocWMMA | Delta |
| --- | ---: | ---: | ---: |
| Prompt TPS | 1369.24 | 1761.34 | +28.64% |
| Decode TPS | 28.05 | 27.72 | -1.18% (single-run noise) |
| Aggregate TPS | 0.7082 | 0.9031 | +27.52% |
| Server time | 22535.10 ms | 17652.29 ms | -21.67% |

### 53,523-token prompt, 16 output tokens, r1, thinking on

| Metric | Generic tile control | Full rocWMMA | Delta |
| --- | ---: | ---: | ---: |
| Prompt TPS | 1091.68 | 1557.94 | +42.71% |
| Decode TPS | 22.30 | 22.47 | +0.76% |
| Aggregate TPS | 0.3210 | 0.4551 | +41.78% |
| Server time | 49745.56 ms | 35067.14 ms | -29.51% |
| External request wall | 49.85 s | 35.16 s | -29.47% |

The increasing prompt gain with context length is consistent with
FlashAttention carrying more of the wall time as the KV sequence grows.

## Validation

- Reduced rocWMMA profile built and served the model successfully.
- Full production rocWMMA profile built and selected the same D256/cols16
  route.
- Focused Qwen-shaped `FLASH_ATTN_EXT` correctness passed `1/1` for D256,
  GQA 6, q8_0/q8_0 KV, masked prefill, and F32 accumulation.
- Focused Q3_K `MUL_MAT` correctness remained `11/11`.
- Three clean short-prompt runs completed without warnings or errors.
- Two matched 30,075-token production runs completed without errors.
- Two matched `ctx=131072`, 53,523-token production runs completed without
  errors and with thinking enabled.
- The full rocWMMA `ggml-hip.dll` is slightly smaller than the tile control
  (`87,544,320` vs `88,190,976` bytes), so this promotion does not explain the
  previously observed extra ROCm VRAM use through binary growth.

Primary artifacts:

- `e293-rocm-tile-control-12k-r3.*`;
- `e293-rocm-full-rocwmma-12k-r3.*`;
- `e293-rocm-full-rocwmma-route.*`;
- `e293-prod-tile-control-30k-r1b.*`;
- `e293-prod-rocwmma-30k-r1.*`;
- `e293-prod-{tile-control,rocwmma}-60k-r1.*`.

## Decision

Promote rocWMMA FlashAttention as the HIP production default. This closes a
fresh-build configuration regression and materially improves the user's main
long-prompt workload without trading away decode throughput. Keep
`GGML_HIP_ROCWMMA_FATTN=OFF` as the build-time rollback for unsupported or
regressing ROCm configurations.
