# C05 - GATED_DELTA_NET forward

## Current cost snapshot

- Center: `CUDA_NODE op=GATED_DELTA_NET kind=forward`
- current C01 sum_ms: `1467.855`
- current C01 count: `3120`
- current C01 avg_ms: `0.470`
- prompt-phase current C01 sum_ms: `1174.486`
- Priority: `P5`

Source trace:
- legacy decode trace: `build_logs/agent-workload/decode-trace-current-ctx12288-ub192-r1.server.log`
- current C01 trace: `build_logs/agent-workload/focus-c01-current-hotspots-r1.server.log`

## Planned trace steps

1. Separate decode-relevant chunks from prefill-dominant chunks.
2. Collect chunk-size and route traces for current decode lane.
3. Re-evaluate only after C01/C02 results are stabilized.

## Current C01 scouting (2026-05-14)

- Diagnostic label: `c05-gdn-current-trace-r1`
- Route:
  - dominant prompt calls: `n_tokens=192`, `chunked_prefill=1`, `chunk_size=96`, `fast_exp=0`
  - internal chunk histogram: `96` chunks dominate; tails `90/91` are tiny
- Chunk timing:
  - prompt `n_tokens=192`: `7296` chunks, sum about `1398.350 ms`, mean `0.1917 ms`
  - decode `n_tokens<=4`: `432` chunks, sum about `19.613 ms`
- Probe results:
  - `GGML_GDN_FAST_EXP=1`: `c05-gdn-fast-exp-r1 = 9.59 TPS`, below current best `9.6080`
  - `GGML_GDN_CHUNK_SIZE=192`: `c05-gdn-chunk192-r1 = 9.58 TPS`, below current best
  - `GGML_GDN_CHUNK_SIZE=128`: `c01-e024-gdn-chunk128-r1 = 9.43 TPS`, below current best
- Decision:
  - no C05 default change
  - do not repeat `fast_exp`, `chunk=128`, or `chunk=192` on this lane unless another code change shifts the GDN route/cost

## E024 chunk128 closure (2026-05-16)

- Theory: for `n_tokens=192`, `chunk=128` keeps two launches like the default `chunk=96`, but changes the token loops from `96+96` to `128+64`.
- Expected ceiling was small: GDN is about `6.6%` of current sync CUDA_NODE time, so even a local `10%` win would be only about `0.6-0.7%` wall.
- Result: `9.6080 -> 9.43 TPS`.
- Decision: reject; no trace and no code changes.
