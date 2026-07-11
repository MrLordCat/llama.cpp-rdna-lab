# E277 ROCm MTP Deferred Accepted-Prefix Catch-Up

## Metadata

- Experiment ID: E277
- Date: 2026-07-11
- Owner: Codex
- Target lane: dual RX 9070 XT ROCm, Qwen3.6-27B-Q3_K_S MTP n3

## Hypothesis

The MTP `process()` path currently catch-up decodes every target verification
row into the draft context before acceptance is known. For n3 this is normally
four rows: the already sampled input plus three draft tokens. After sampling,
the server truncates rejected rows. Deferring catch-up until `accept()` allows
the draft context to decode only row 0 plus the accepted draft prefix.

## Measured Gate

`h60-rocm-n3-phase-process-r1`, 128 output tokens and 46 draft rounds:

- total decode: `3336.42 ms`;
- MTP `process`: `1594.63 ms` (`34.67 ms/round`);
- MTP `draft`: `561.27 ms` (`12.20 ms/round`);
- verify sampling: about `0.21-0.95 ms/round`;
- accepted-prefix mean length including row 0: `2.76` versus four rows
  currently decoded.

The row-count ceiling is about `4 / 2.76 = 1.45x` for catch-up itself. This is
not a whole-decode projection because small-row kernels have fixed costs.

## Prototype Contract

- Opt in with `LLAMA_MTP_DEFER_ACCEPT_CATCHUP=1`.
- Restrict to one sequence, non-shared, single-head MTP verification batches
  with two or more rows, all logits requested, contiguous positions, and a
  valid previous hidden row.
- Keep prompt processing and all unsupported shapes on immediate catch-up.
- Preserve target hidden rows, tokens, positions, and the preceding hidden row
  until acceptance; decode exactly `n_accepted + 1` rows in `accept()`.
- On any gate miss, retain the current path.

## Benchmark Gate

1. Build and run a deterministic short output/acceptance smoke.
2. Compare env-off and env-on at `ctx=12288,b=8192,ub=1024,q8/q8,n3`,
   `max_tokens=256`, thinking on, cold-first, no reuse/no prime.
3. Reject on errors, stale KV, material acceptance loss, or no decode gain.
4. Run 131k long-prompt confirmation only after a clean short-lane win.

## Result

- Outcome: rejected; prototype source reverted.
- Neighboring r1: immediate `38.49` decode tok/s versus deferred `38.65`
  (`+0.42%`, noise-level), with the same `81/136 = 59.56%` acceptance.
- Instrumented deferred run: `process=1574.90 ms`, `draft=547.60 ms`,
  `accept=38.11 ms` over 46 rounds. The immediate trace measured
  `process=1594.63 ms`, `draft=561.27 ms`, `accept=0.03 ms`.
- Interpretation: reading target NextN embeddings synchronizes the target
  decode, so most time attributed to `process()` is required target verify
  completion rather than the draft-context catch-up body. Moving catch-up only
  shifts its enqueue/synchronization between phases and does not remove a
  meaningful wall-time center.
- Kept diagnostic: generic speculative stats now include `process` time as
  `dur(b,p,g,a)`. No deferred-catch-up runtime knob remains.
- Artifacts: `h60-rocm-n3-phase-process-r1.*`,
  `e277-rocm-n3-immediate-128-r1.*`, `e277-rocm-n3-deferred-128-r1.*`, and
  `e277-rocm-n3-deferred-phase-r1.*`.
