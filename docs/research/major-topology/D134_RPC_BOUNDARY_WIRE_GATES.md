# D134 - RPC boundary wire gates

Status: Q8 R2 gate positive and short PPL-neutral; F8 R2 prompt-negative (2026-08-27)

## Scope

Continue RPC-only work after D133. The priority lane is the three-device
topology with the remote RTX 3080 first and both local Vulkan GPUs after it:
`RPC0,Vulkan0,Vulkan1`. Local dual-Vulkan is a control only and is not a
target for source optimization in this program.

## Locked R2 lane

- model: `Qwen3.8-27B-Q4_K_M.gguf`;
- context/batch: `12288`, `8192/1024`;
- KV: `q8_0/q8_0`, flash attention enabled;
- MTP: `draft-mtp`, `spec-draft-n-max=4`;
- output: `128` tokens, seed `42`, no warmup;
- prompt: `repo-snapshot`, `24576` requested chars;
- fitting/cache: `-fit off`, `--cache-ram 0`, `--ctx-checkpoints 0`;
- devices: `-dev RPC0,Vulkan0,Vulkan1 -sm layer -ts 0.8,1,1.4`;
- RPC endpoint: `192.168.1.60:50052`, pre-P2 protocol-compatible server;
- one benchmark at a time, no diagnostic environment in speed controls.

The clean R2 control is `d133-r2-rpc3080-dualvulkan-14k`: aggregate
`13.0091`, prompt `1217.015`, decode `39.875` TPS, acceptance `175/314`
(`55.73%`).

## Hypothesis

H82 remains the narrow candidate: the F32 `l_out-*` activation crosses the
RPC boundary as F16 by default, so the client sends about 10 MiB for a
1024-token ubatch. The D133 trace found 14 large `l_out-16` GETs with
`385.2 ms` average client-side wait while the remote handler itself averaged
`19.0 ms`. This is sufficient evidence to measure wire reduction before
touching protocol scheduling or reopening D132/P2.

## Measurement sequence

1. Repeat the clean R2 F16 control to establish short-lane variance.
2. Run the existing opt-in `GGML_RPC_ACT_Q8_0=1` route on the identical R2
   lane. This changes only intermediate activation transport; final logits
   remain F16.
3. If Q8 is positive and acceptance remains stable, repeat it before any
   promotion and apply a separate quality/PPL gate. Measure F8 only after Q8
   has a reproducible signal.
4. Keep `GGML_RPC_TIMELINE`, scheduler timing and server tracing out of speed
   controls; use one separate diagnostic run for operation counts and wire
   sizes.

The first gate is not a production claim: a candidate must clear short-lane
noise (target at least 3% prompt improvement on repeated R2 controls) without
an aggregate/decode regression or an acceptance collapse. No runtime source
change is authorized by this note; a negative candidate stays opt-in or is
rejected.

## Measured R2 gates (2026-08-27)

| Variant | Label | Aggregate TPS | Prompt TPS | Decode TPS | Acceptance |
| --- | --- | ---: | ---: | ---: | --- |
| F16 control | `d134-r2-f16-control-r1` | 12.9731 | 1219.3950 | 39.1850 | 85/168 + 90/146 |
| Q8 wire | `d134-r2-q8-wire-r1` | 14.4432 | 1296.1050 | 48.1750 | 96/121 + 94/128 |
| Q8 wire repeat | `d134-r2-q8-wire-r2` | 14.3348 | 1278.7850 | 48.1850 | 96/121 + 94/128 |
| F8 wire | `d134-r2-f8-wire-r1` | 13.0656 | 1167.9500 | 44.1600 | 89/150 + 94/128 |

The two Q8 runs average `1287.445` prompt TPS, `48.18` decode TPS and
`14.389` aggregate TPS. Against the F16 control this is `+5.68%` prompt,
`+21.88%` decode and `+10.76%` aggregate. The result is reproducible enough
to keep Q8 as the leading R2 opt-in candidate; it still requires a separate
quality/PPL gate before any default or production claim. The F8 run is
`-4.12%` on prompt and only `+0.57%` aggregate versus F16, so it is rejected
for the current prompt-focused lane and is not repeated.

## Quality gate (2026-08-27)

The first F16 PPL attempt with `batch=8192` hit a Vulkan pinned-memory buffer
limit before scoring (`Requested buffer size exceeds device buffer size limit`).
This was a runner configuration limit, not a model or RPC failure. Both the
control and candidate were then run with identical safe settings
`batch=1024/ubatch=1024`, `n_ctx=8192`, two WikiText-2 chunks and the same R2
RPC topology:

| Variant | PPL |
| --- | ---: |
| F16 control (`d134-r2-f16-ppl-b1024`) | `4.9183 +/- 0.12282` |
| Q8 wire (`d134-r2-q8-ppl-b1024`) | `4.9194 +/- 0.12287` |

The Q8 delta is `+0.0011` PPL (`+0.02%`), well inside the short-gate error
bar. This clears the two-chunk smoke quality gate, but is not a full quality
or production claim; keep the route opt-in until a longer quality check is
needed for release evidence.

Existing scheduling opt-ins were also checked around the Q8 route on the same
short R2 lane. The one-run results were below the Q8-only center
(`14.389` aggregate / `1287.445` prompt / `48.18` decode TPS):

- Q8 + `LLAMA_RPC_RUN_AHEAD=1`: `13.84 / 1238.59 / 46.37`;
- Q8 + `GGML_RPC_ASYNC_GRAPH=1`: `14.03 / 1229.90 / 49.28`;
- Q8 + both existing scheduling opt-ins: `13.71 / 1223.71 / 45.98`.

These knobs add overhead on this 14K smoke and remain non-default; they do
not change the current RPC wire conclusion.

The Q8 diagnostic (`d134-r2-q8-wire-diag`) recorded 88 `GET_TENSOR` calls
versus 103 in the F16 diagnostic. The 14 full `l_out-16` reads remained, but
their client-side wait fell from `5393.4 ms` (`385.2 ms` each) to
`5248.2 ms` (`374.9 ms` each). The source wire contract changes each full
activation from 10,485,760 F16 bytes to 5,570,560 block-Q8 bytes. The modest
large-GET wait reduction and the TPS gain should not be treated as a pure
bandwidth proof: command ordering and remote graph timing remain mixed into
the client wall time.

## Current evidence and next decision

D133 shows R2 is materially faster than the one-local-GPU RPC topology but
still slower than local dual Vulkan. D134 now gives a measured direction:
continue RPC-only work with Q8 activation wire; the short PPL gate is neutral,
while async/run-ahead scheduling is negative on this lane. Next measure the
receive/protocol path around the smaller payload. Do not reopen D132/P2 or
move into local Vulkan kernel work from this result.