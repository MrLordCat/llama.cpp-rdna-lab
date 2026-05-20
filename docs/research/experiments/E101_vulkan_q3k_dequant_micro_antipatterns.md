# E101 Vulkan Q3_K Dequant Micro Anti-Patterns

## Metadata

- Experiment ID: E101
- Date: 2026-05-20
- Owner: Copilot
- Type: narrow shader micro-probe rejection note
- Hypothesis: H31
- Target lane: Vulkan Q3_K pp7488 gate, `b=5120`, `ub=1024`, `q4_0/q4_0`, FlashAttention on

## Hypothesis

The Q3_K dequant helper in `mul_mm_funcs.glsl` still contains power-of-two division/modulo and an `int8_t` scale round-trip. Rewriting these into cheaper-looking GLSL expressions might reduce hot dequant ALU.

## Probes

### Shift/mask index math

Changed Q3_K power-of-two `/` and `%` operations to explicit shifts and masks.

Static SPIR-V effect for `matmul_q3_k_f32_aligned_f16acc_cm1.spv`:

| Op | Before | Candidate |
| --- | ---: | ---: |
| `OpUDiv` | `28` | `20` |
| `OpUMod` | `12` | `6` |
| `OpShiftRightLogical` | `4` | `12` |
| `OpBitwiseAnd` | `4` | `10` |

Measured pp7488: `927.51 tok/s`, below the current same-shape baseline (`~959-985 tok/s` depending on control run). Rejected and reverted.

### Scale int simplification

Changed:

```glsl
const int8_t us = int8_t(bits);
const float dl = float(data_a[ib].d) * float(us - 32);
```

to direct `int(bits) - 32` arithmetic.

Measured pp7488: `929.30 tok/s`, with unchanged driver resources (`113 VGPR / 45 SGPR / 20480 B LDS / scratch 0`). Rejected and reverted.

## Decision

Do not rewrite Q3_K helper arithmetic purely because SPIR-V opcode counts look cleaner. On AMD proprietary LLPC for RX 9070 XT, the original division/modulo and `int8_t us` patterns are faster for this shader. Future Q3_K dequant candidates need a stronger mechanism than expression-level cleanup: fewer dequant pairs, less LDS traffic, different work distribution, or a measured route/resource change.

## Artifacts

- `build_logs/agent-workload/e100-q3k-shiftmask-spirv-summary.md`
- `build_logs/agent-workload/e100-q3k-shiftmask-pp7488-r1.txt`
- `build_logs/agent-workload/e101-q3k-scale-int-spirv-summary.md`
- `build_logs/agent-workload/e101-q3k-scale-int-pp7488-r1.txt`