# D101: ROCm full-FP8 long-KV phase ceiling

Date: 2026-08-14

Status: complete negative program. No production source candidate is accepted.

## Objective

Identify which device-side portion of the current gfx1201 D256 full-FP8
FlashAttention route owns a recoverable long-context decode ceiling after D100
closed HIP graph templates, host submission, generic wave-count, narrow-tile,
and KV-slice scheduling.

D101 compares the already correctness-qualified D099 rollback routes before
editing kernel dataflow:

1. portable/reference FP8 K/V conversion plus the WMMA f16 body;
2. native FP8 KQ with the portable f16 V leg;
3. full native FP8 KQ plus native P*V.

The comparison is phase attribution, not a proposal to change the FP8 default.

## Locked lane

- model: `models/Qwen3.6-27B-Q4_K_M.gguf`;
- Windows ROCm/HIP 7.1, `gfx1201`, Ninja release build;
- `-dev ROCm1,ROCm0 -sm layer -ts 1,1`, default output on ROCm0;
- `ctx=49152,b=8192,ub=1024`, one slot, FlashAttention on;
- `f8_e4m3/f8_e4m3`, cold/no-reuse/no-prime/no-warmup, `-fit off`;
- `triage_diff`, seed 42, 128 output tokens, `spec=none`;
- 98K confirmation only after a 49K phase delta exceeds 3%;
- no synchronized per-node timing and no hardware discovery during a run.

D100's adjacent full-native anchors are about `1814-1825 prompt tok/s` and
`22.9 decode tok/s`. They are context only; D101 uses a same-binary bracket.

## Route controls

| Route | Environment |
| --- | --- |
| full native KQ+V | both rollback variables unset |
| native KQ only | `GGML_ROCM_FATTN_F8_NATIVE_V=0` |
| portable reference | `GGML_ROCM_FATTN_F8_NATIVE_KQ=0` |

`GGML_TRACE_FATTN_PATH=1` is permitted for route proof because it only logs
dispatch selection and does not synchronize the stream.

## Gate ladder

### G0: 49K phase bracket

Run full-open -> KQ-only -> reference -> full-close on one binary. Preserve
prompt/decode/aggregate TPS, exact route evidence, output completion length,
placement, and graceful shutdown.

Interpret only center/bracketed deltas:

- KQ-only faster than full by at least 3% admits native-V dataflow work;
- reference faster than KQ-only by at least 3% admits native-KQ dataflow work;
- full native best or all differences below 3% closes nearby phase rollback
  work and requires a new common-kernel/data-movement mechanism.

### G1: 98K scaling

Run only the G0 winner and the adjacent full-native control at 98K with 64
output tokens. The phase must retain at least a 3% decode ceiling and preserve
prompt throughput within 2% before source work begins.

### G2: bounded prototype

Select one instruction/dataflow mechanism inside the admitted phase. Do not
repeat D100 wave-count, cols-per-block, or parallel-KV-slice sweeps. A kept
prototype requires focused deterministic correctness, route proof, a 49K
A-B-A pass, and 98K confirmation.

Selected first prototype: the PB8 result-combine kernel currently recomputes
the same KQ maximum, eight exponential scales and softmax denominator in all
256 D256 threads. A default-off exact-route variant may compute those scalar
values once in shared memory, then retain the existing per-D numerator order.
This changes combine dataflow without changing the main KQ/P*V body, tile
count, or launch geometry.

## Safety and closure

- Never force `LLAMA_OUTPUT_DEVICE=ROCm1`.
- Never use peer copy or `hipMemGetInfo` in this program.
- Never hard-kill a server during load, prompt evaluation, or decode.
- Negative prototypes are removed; default-off measurement tools may remain
  only when documented and behavior-neutral.
- Finish each source candidate with the ROCm build and `git diff --check`.

## Results

### G0: current-binary phase bracket

The exact 49K lane produced:

| Route | Prompt tok/s | Decode tok/s | Aggregate TPS | Decode vs full center |
| --- | ---: | ---: | ---: | ---: |
| full native open | 1791.30 | 22.69 | 5.61 | |
| native KQ only | 1745.31 | 19.60 | 5.29 | -14.1% |
| portable reference | 1694.78 | 17.31 | 5.01 | -24.1% |
| full native close | 1798.24 | 22.94 | 5.64 | |

The full-native center is `1794.77/22.815/5.625`. Native KQ gains about
`13.2%` decode over the portable reference, and native P*V adds about `16.4%`
over KQ-only. Both D098 phases are material wins on the current production
lane. There is no rollback-route winner to confirm at 98K, so G1 correctly did
not run.

### G2: shared scalar combine prototype

The default-off candidate computed the PB8 KQ maximum, exponential scales and
softmax denominator once per D256 result block instead of identically in all
256 lanes. Focused ROCm0 native-V correctness passed `2/2`; route tracing
proved `combine_shared_scales=1` on both devices with unchanged PB8 geometry.

The same-binary 49K A-B-A was neutral:

| Route | Prompt tok/s | Decode tok/s | Aggregate TPS |
| --- | ---: | ---: | ---: |
| control open | 1809.05 | 22.91 | 5.66 |
| shared scales | 1806.45 | 22.91 | 5.65 |
| control close | 1809.24 | 22.82 | 5.66 |

Against the `22.865` control center, the candidate is only `+0.20%` decode
and slightly negative in prompt/aggregate throughput. The duplicated scalar
combine arithmetic is therefore not on the token critical path. The source,
environment gate and trace field were removed; 98K confirmation was not
admitted.

## Closure

D101 closes portable/KQ-only rollback work and combine-scalar optimization.
The current full-native KQ+V route remains the production FP8 owner. A
successor needs measurement or a dataflow change inside the main fused
long-KV body itself; graph submission, launch topology, phase rollback, and
result-combine scalar work are now fenced by D100/D101 evidence.
