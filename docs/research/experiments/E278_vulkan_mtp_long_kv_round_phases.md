# E278 Vulkan MTP Long-KV Round Phases

## Metadata

- Experiment ID: E278
- Date: 2026-07-11
- Backend: dual RX 9070 XT Vulkan
- Model: `Qwen3.6-27B-Q3_K_S_mtp.gguf`

## Observation

At `ctx=49152` with 38,870 prompt tokens, Vulkan spec-none reaches `27.13`
decode tok/s while MTP n3 reaches only `18.02`, despite `76.9%` acceptance.
The same current Vulkan binary reaches `40.95` decode tok/s with n3 on a
159-token prompt, so the failure depends on actual KV length rather than only
the four-column verification shape.

## Existing Trace

Synchronized long-KV n3 trace (`e278-vulkan-49k-n3-sync-trace-r1`):

- steady target N=4 ubatch: about `44.8 ms`;
- draft-context N=4 catch-up: about `1.9 ms`;
- three single-row draft decodes: about `10.2 ms` total;
- observed full decode: `887.6 ms` for 16 tokens, roughly four rounds.

The visible llama-decode phases explain only about `57 ms` per steady round,
while observed wall time is roughly `220 ms` per round. The missing center is
therefore outside the timed graph calls.

## Hypothesis And Gate

Instrument the existing server flow behind `LLAMA_SPEC_SERVER_PHASE_TIMING=1`:

1. draft generation in `server_slot::update_batch()`;
2. target `llama_decode()` return and `common_speculative_process()`;
3. verify sample/accept;
4. target and draft `seq_rm()` rollback.

This is diagnostic only. Reject any optimization proposal until these phases
sum to the observed round wall time. No forced device query, executable probe,
or Vulkan staging script is permitted during the trace sequence.

## Root Cause And Fix

Server phase timing closed the missing interval:

- first long-KV target verify N=4: `630.6 ms`;
- subsequent N=4 verifies: `47-49 ms` including NextN processing;
- draft: about `11 ms/round`;
- verify sampling: below `1 ms/round`;
- rollback: about `0.05 ms/round`.

The GUI autotune hardcoded `max_tokens=16` and disabled model warmup, so the
single cold N=4 graph dominated the reported decode rate. N2 has the same
problem at 16 tokens (`18.38 tok/s`) despite reaching `34.06 tok/s` in the
existing 64-token long-prompt run. Ordinary warmup did not help because
`common_init_from_params()` warmed only the BOS/EOS batch (one or two rows),
not the speculative verification shape.

Implemented fix: GUI autotune now uses `max_tokens=128` for every spec mode.
This keeps separately launched none/MTP runs comparable and amortizes the
one-time backend graph transition without hiding it from request wall time.

Both the ordinary warmup and an experimental post-MTP exact-N4 warmup failed
to survive the long PP prefill/TG transition; the latter was reverted. Keeping
PP scheduler buffers reduced the first verify from about `630` to `453-494 ms`
but did not remove it and slightly reduced prompt throughput, so it remains a
diagnostic env control rather than a default.

Final same-prompt 128-token A/B:

| Mode | Prompt tok/s | Decode tok/s | Aggregate TPS |
| --- | ---: | ---: | ---: |
| Vulkan none | `1449.41` | `29.15` | `4.10` |
| Vulkan MTP n3 | `1401.10` | `38.78` | `4.12` |

MTP therefore improves decode by `1.33x`; the remaining `3.3%` prompt tax and
prompt-heavy workload explain the small aggregate delta.

After reverting the ineffective warmup prototype and rebuilding the final
Vulkan source, `e278-vulkan-49k-n3-final-r1` independently confirmed
`1399.85` prompt / `38.61` decode tok/s and `4.12` aggregate TPS.

The server phase trace remains env-gated by
`LLAMA_SPEC_SERVER_PHASE_TIMING=1` and has no default runtime logging cost.
