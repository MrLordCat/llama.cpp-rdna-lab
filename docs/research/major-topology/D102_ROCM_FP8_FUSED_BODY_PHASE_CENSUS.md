# D102: ROCm full-FP8 fused-body phase census

Date: 2026-08-14

Status: complete. The census is measured and the gate closed without a
prototype.

## Results

### G0: 49K phase census

The exact 49K lane with the default-off census instantiation reported for
both devices, 192 blocks each:

| Phase | Share |
| --- | ---: |
| P*V WMMA + fp32 VKQ LDS store | 55.4% |
| KQ WMMA over the long KV | 28.3% |
| softmax + packed P -> E4M3 requant | 14.3% |
| rescaled VKQ merge | 2.0% |

- dev=0: `kq=28.1% softmax=14.4% pv=55.6% merge=2.0%`
- dev=1: `kq=28.5% softmax=14.3% pv=55.2% merge=2.0%`

The P*V phase is the largest at about 55%, but no phase owns the 60%
required by the gate, so the fused body is jointly MMA/LDS/bandwidth bound
and D102 closes without a prototype.

### Census-side interpretation

Both KQ and P*V loop over the same 60 KV chunks and read the same 128 KiB
K/V tile per chunk, and both issue the same number of fp8 x fp8 MMAs per
chunk. P*V costing 2x KQ therefore comes from the extra per-chunk work: the
fp32 VKQ-part store/reload through LDS (~135 KiB each way) plus the rescale
merge. D098's bisect shows the fp32 store roundtrip is structural, not a
bandwidth limit: LDS is not the bottleneck, registers are (the f16 store
variant lost 6.5% because the conversion copies cost +32 VGPR on a
spill-free 154-156 VGPR kernel). The lever for a successor is instruction
and register pressure in the P*V phase, not LDS bytes.

### Decode-share math

The D101 bracket gives the native V phase (softmax+requant + P*V + merge =
71.7% of the kernel) a decode value of about 14.6% (full-native 22.94 versus
KQ-only 19.60). That implies the FA kernel owns roughly 20% of the 49K
decode token. Consequently:

- a 10% P*V-phase reduction is worth about `10% x 55.4% x 20% = 1.1%`
  decode — below the `3%` gate and inside adjacent-run noise;
- clearing the 49K gate needs a ~30% P*V-phase reduction;
- at 98K the FA share roughly doubles with the KV, so the same prototype is
  worth about twice as much there. A P*V candidate should therefore be
  validated on the 98K lane.

### Measurement validity

- The four clock64() windows cover only the chunk loop; the end-of-kernel
  pinned-mirror writes and the system fence sit after the last marker, so
  the phase shares are not affected by the mirror mechanism.
- The census run is distorted by design: decode `19.70 tok/s` versus the
  adjacent `22.82-22.94` controls, because 192 blocks write their deltas to
  pinned host memory and fence at every decode step. Cycle shares are
  relative and are never used as a TPS claim.
- Prompt `1725.67 tok/s` versus the earlier `~1809-1818` controls runs on the
  unchanged production instantiation and is attributed to session thermal
  drift after seven heavy 49K prefills in one session, not to the census.
- Focused ROCm0 FP8 correctness passed `2/2` with the census path and `2/2`
  on the production path. The production instantiation is byte-identical to
  the pre-D102 one: the census variant is a separate template selected only
  under the environment gate and `Q->ne[1] <= 4`.

### G1: bounded prototype

Not admitted: the largest phase (`55.4%`) is below the 60% gate, and the
remaining phases each own less than a third of the kernel.

## Implementation notes

Three output mechanisms were tried; only the last is safe on this machine:

1. `hipMemcpyFromSymbol` in an `atexit` handler: hangs forever after
   `ggml_backend_free` tore the context down; the process becomes unkillable
   and write-locks `ggml-hip.dll`.
2. Synchronous/async host symbol copy inside a graph capture: capture is
   poisoned (`operation failed due to a previous error during capture`) or
   fails outright (`invalid device symbol`) — device-symbol host lookup is
   broken on this HIP/Windows setup.
3. Kept: the census kernel writes its four deltas directly into pinned host
   memory through a device pointer installed by a tiny init kernel captured
   in the same graph. The exit handler only reads plain host memory; no HIP
   call ever happens after teardown.

## Closure

D102 delivers the first intra-kernel phase census of the full-native D256
decode body. The next candidate boundary is the instruction/register cost
inside the P*V phase (fp32 accumulator materialization and merge), but the
phase does not dominate the kernel alone, and the decode-share math says a
49K prototype would need a ~30% phase cut to clear the 3% gate. The 98K lane
is the more sensitive venue.

## Objective

D100 closed graph/host submission and launch topology; D101 closed phase
rollback and result-combine scalar work. The remaining long-context decode
ceiling lives inside the fused D256 full-native FP8 kernel itself. D102 first
measures where that kernel spends its N=1 time before any dataflow edit.

The census targets four phases inside the production eight-wave body:

1. KQ WMMA tile loop over the long KV (K f8 reads + fp8 x fp8 MMA);
2. softmax + packed P -> E4M3 requantization;
3. P*V WMMA tile loop plus the fp32 VKQ part store to LDS;
4. rescaled VKQ merge into the shared accumulator and final store.

## Measurement design

- Template-bounded kernel variant: the accounting exists only in a
  `phase_census` instantiation of `flash_attn_ext_f16` selected when
  `GGML_ROCM_FATTN_PHASE_CENSUS=1` and `Q->ne[1] <= 4` on the D256
  full-native KQ+V route. Prefill and every other route keep the unchanged
  production instantiation.
- Each block accumulates per-phase `clock64()` deltas across its KV-chunk
  loop and writes four values into pinned host memory through a device
  pointer installed by a tiny init kernel captured in the same graph.
- The host prints per-device phase fractions once at process exit, after the
  graceful server teardown drained the stream. No per-token or per-node
  synchronization is ever issued.
- Raw cycle sums are relative shares inside one kernel, not absolute wall
  time. They are never used as a TPS claim.

## Locked lane

- model `models/Qwen3.6-27B-Q4_K_M.gguf`, `f8_e4m3/f8_e4m3`;
- `ctx=49152,b=8192,ub=1024`, one slot, FlashAttention on;
- `-dev ROCm1,ROCm0 -sm layer -ts 1,1`, default output on ROCm0;
- `triage_diff`, seed 42, 128 output tokens, `spec=none`;
- cold/no-reuse/no-prime/no-warmup, `-fit off`.

## Gate ladder

### G0: 49K phase census

Run the exact lane with the census environment. Gate requirements:

- focused ROCm0 FP8 correctness passes with and without the census path;
- the census instantiation builds and the report appears for both devices;
- the production instantiation is proven unchanged (identical binary path when
  the environment is unset).

Interpret block-level shares per device. A phase admits prototype work when it
owns at least 60% of the kernel and has a modeled reduction larger than the
phase share multiplied by the `3%` decode ceiling; otherwise the fused body is
treated as jointly bandwidth/MMA bound and D102 closes without a prototype.

### G1: bounded prototype

One instruction/dataflow change inside the admitted phase only. Focused
correctness, exact-route proof, a same-binary 49K A-B-A with the `>=3%` decode
gate, then a 98K confirmation. Wave count, cols-per-block, parallel-KV
slices, rollback routes and combine scalars are fenced by D100/D101 and may
not be repeated.

## Safety and closure

- The census path must never change output semantics: it is timing reads plus
  a trailing write of cycle deltas only.
- Never use synchronized per-node traces for absolute wall claims.
- No forceful server termination during load, prompt evaluation, or decode.
- Negative prototypes are removed; the census remains default-off and
  behavior-neutral if kept.
- Finish with the ROCm build, `git diff --check`, and canonical history.
