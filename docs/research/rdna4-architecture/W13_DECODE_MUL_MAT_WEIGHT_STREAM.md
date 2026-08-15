# W13: decode MUL_MAT/MMVQ weight-stream audit (source-only)

Date: 2026-08-14
Status: in progress (source audit + candidate plan; GPU runs deferred - GPUs are
busy with user work, no launches until the next signal)

## Question

W12 census: MUL_MAT >= 50% of a 49K decode token, FA ~10-20%, GDN ~13.5%.
Where exactly does the MUL_MAT decode traffic go, and which weight-stream
levers can physically clear the whole-lane `>=3%` decode gate?

## Audit facts (source-only, mmvq.cu / vecdotq.cuh)

Lane: dual-ROCm 49K, Q4_K_M weights, decode M<=4 tokens, K=5120, N in
{5120 attn/out, 17408 FFN gate/up, 5120 FFN down}.

### Dispatch (decode)

- Quantized weights + `ne11 <= 8` -> `mul_mat_vec_q` (MMVQ), one launch per
  (op, token). No MMQ in decode; hipBLAS is off the hot path (W10/W11).
- RDNA4 table (`get_mmvq_mmid_max_batch_rdna4`): Q4_K max batch = 4, so
  `ncols_dst` 1..4 all reach MMVQ.

### Kernel geometry for Q4_K, ncols_dst == 1 (the spec-none decode stream)

- `calc_nwarps(RDNA4)`: ncols_dst == 1 -> **8 warps** (256 threads, wave32);
  every ncols_dst > 1 -> **1 warp** (32 threads).
- `should_use_small_k` (auto policy, mmvq.cu:1120-1190): the Qwen-hot RDNA4
  branch overrides the generic trigger: for ncols_dst == 1,
  `use = type != Q6_K`, i.e. **Q3_K and Q4_K always take small_k**,
  Q6_K never. small_k -> `rows_per_block = nwarps = 8` (mmvq.cu:467).
- Result: block = (32, 8) = 8 warps x 8 weight rows; each warp owns one row;
  grid.x = nrows/8 (640 CTAs for N=5120, 2176 for N=17408), grid.y = tokens.
- K-loop: 20 Q4_K blocks per row (5120/256). `blocks_per_iter = vdr*nwarps*32/qi`
  = 2*8*32/64 = 8 -> **3 iterations** (8/8/4). Each thread computes one
  Q4_K block per iteration.
- Weight reads per thread-block-iteration (`vec_dot_q4_K_q8_1`,
  vecdotq.cuh:1044): 2x int32 quant chunks, 4x uint16 scales, 1x half2 dm
  (~20 bytes of weights per block) plus the Q8_1 activation block. Scalar
  int/half loads, no explicit cache-policy modifiers.
- The QWEN small-K env toggles listed in W10
  (`GGML_MMVQ_QWEN_FORCE_SMALL_K` / `_DISABLE_SMALL_K`) **no longer exist**:
  the winner was consolidated into the auto policy during phase-3 debt
  cleanup. The only remaining Qwen decode switches are `GGML_MMVQ_RDNA4_Q3K_MAX_BATCH`,
  `GGML_MMVQ_Q3K_RDNA4_VK16`, `GGML_MMVQ_Q3K_DISABLE_PAIRDOT` and the
  default-off traces (`GGML_TRACE_MMVQ_*`).

### Traffic model (per token, per GPU, 49K)

- Weight stream: ~16.5 GB model / 2 GPUs = ~8.2 GB re-read per token ->
  ~183 GB/s at 22.2 tps (Qwen3.8 rebaseline RC-49K-f8 decode).
- KV stream (W09): ~3 GB per GPU per token -> ~66 GB/s.
- Total ~250-280 GB/s vs 644 GB/s peak = ~40-45% of peak BW: the decode
  stream is latency/occupancy-bound as much as BW-bound.
- 493 MUL_MAT launches per token (W12): at D100's measured ~0.6 ms total
  host launch cost, launch overhead is ~1.4% - not the lever.

## Candidates (priority order)

1. **C1 - small_k A/B for Q4_K (cheap code-switch A-B-A)**. The 8-rows-per-CTA
   policy was chosen on Qwen3.6 lanes; the Qwen3.8 rebaseline never re-tested
   it. Opposite point = 1 row/CTA with 8 warps covering one row (generic
   geometry), or nwarps 4/2 midpoints. Temporary env gate, same-binary A-B-A
   on the locked 49K lane, 98K confirmation only if >= 3% decode.
   Expected: single-digit % either way; the 8-row layout has the best DRAM
   row locality (8 adjacent 144 B rows = ~1.2 KB contiguous), so C1 may
   confirm the current policy rather than beat it.
2. **C2 - draft-batch weight reuse for MTP decode (structural, big)**. If the
   MTP decode drives each draft through its own ncols_dst==1 launch, the
   weight stream is re-read per draft token. Batching 2-4 draft tokens into
   ncols_dst 2..4 launches reads the weights once per draft group. But the
   RDNA4 table gives ncols>1 only 1 warp/32 threads, so a batch path also
   needs an nwarps retune (today unmeasured for ncols 2..4). Open question
   to verify in llama-graph: does MTP decode batch drafts or not? This is a
   graph-level change - discuss with the user before coding.
3. **C3 - GDN per-token cost audit (next W-item)**. W12: GATED_DELTA_NET =
   13.5% of a decode token at 49K (48 nodes/token) - the chunk recurrence
   runs per token, not only prefill. Source audit of the decode-time chunk
   state update is the next cheapest research step (C05 is a 2026-05 12K
   record, not a 49K decode audit).
4. **C4 - cache-policy hints on weight/KV streams (blocked, same as H80)**.
   The W09/H80 toolchain blocker applies verbatim: ROCm 7.1 clang drops
   `__builtin_nontemporal_load`, and llvm-mc accepts no cache_policy modifier.
   A weight-stream `glc`/`slc` hint cannot be expressed today. The useful
   hint (KV streaming so it does not evict the weight stream) remains
   unexpressible until a toolchain change.
5. **Demoted (no expectation)**. FA-side micro-optimizations: FA is 10-20%
   of the token, so even a 20% FA win clears only 2-4% whole-lane, at the
   edge of the gate (W12 finding 3).

## C1 verdict (2026-08-15)

small_k=0 (1 row/CTA, 8 warps per row) versus the auto policy (8 rows/CTA,
1 warp per row), spec=none, f8 KV, Qwen3.8:

- 49K: candidate ahead of control in 3/3 interleaved A-B-A pairs, mean decode
  22.41 vs 21.52 tps (+4.1%; +2.7% excluding one anomalous control task).
  Prompt throughput unaffected (as expected: MMVQ decode-only).
- 98K (128 out): cand 19.43 vs ctrl 19.85 (-2.1%) - noise, not confirmed.
- 98K (512 out, second A-B-A): cand 20.13 vs ctrl 19.83 (+1.5%) - noise.
- Correctness: coherent smoke at ctx=98304 with sk=0, clean teardown.

VERDICT: keep the gate as OPT-IN (default = current auto policy). The 49K
win does not clear the track's 98K-confirmation fence, so it cannot become
the default. Protocol note added: context-dependent MMVQ wins must pass BOTH
49K and 98K gates; a 49K-only win stays opt-in/diagnostic.

## C1 mechanism (GGML_TRACE_MMVQ_RESOURCES, decode stream)

Trace comparison (49K, 96 out, sync mode; med_ms includes a constant
per-node sync overhead, relative deltas are the signal):

| group (ncols_dst=1) | sk=1 (auto) | sk=0 | |
| --- | --- | --- | --- |
| q4_K K=5120 fusion=1 (FFN gate/up) | occ 50%, waves 32, regs 76, shared 14336 B, 0.282 ms | occ 100%, waves 64, regs 36, shared 1792 B, 0.264 ms | sk=0 -6.4% |
| q4_K K=5120 fusion=0 (attn/down) | occ 100%, regs 61, shared 7168 B, 0.0645 ms | occ 100%, regs 24, shared 896 B, 0.0565 ms | sk=0 -12.4% |
| q4_K K=17408 fusion=1 (FFN down) | 0.079 ms (grid 640) | 0.0815 ms (grid 5120) | sk=0 +3% |

Mechanism: the auto 8-rows-per-CTA layout costs 76 registers and 14 KB
shared on the fused FFN gate/up path, capping occupancy at 50% (32
waves/SM). The 1-row layout halves registers (36) and cuts shared 8x
(1792 B), restoring 100% occupancy - the decode stream is latency-bound
(W13 traffic model), so occupancy directly buys decode TPS. The only
regression is the K=17408 down projection (+3%, launch-count dominated),
which is a minority of the MUL_MAT time. On 98K the same absolute gain
shrinks in percent because the FA share of a token grows with KV - which
is exactly why the 98K A-B-A landed in noise. No further value-guessing is
needed: occupancy is monotone in the 1..8 row range, so sk=0 is the
predicted optimum; a 4-row midpoint would only split the difference.

Next step (C1b, later): reduce the fused-FFN register pressure of the
8-row layout itself (76 regs come from 8 row accumulators x fusion), which
would recover occupancy without the K=17408 launch-count cost.

## C1b results (2026-08-15): staged x/gate reduce

The trace showed the 8-row fused limit was SHARED, not registers:
`max_blocks_per_sm = 4` at 14336 B shared (65536/14336 = 4.6), so the
registers were never the binding constraint. C1b halves the shared footprint
by reusing the x-partial buffer for the gate partials (two extra barriers,
zero extra registers): 14336 -> 7168 B, `max_blocks_per_sm` 4 -> 8,
occupancy 50% -> 100%, grid unchanged (no K=17408 launch-count cost).
Gate `GGML_MMVQ_RDNA4_QWEN_STAGED_REDUCE` (default 1).

A-B-A, spec=none, f8 KV, Qwen3.8:
- 49K: ctrl(legacy) 19.72/21.27 vs cand(staged) 21.67 -> +5.7% over the
  control interpolation, both candidate tasks above all controls.
- 98K (512 out): ctrl 19.64/19.46 vs cand 19.91 -> +1.8% - stable positive
  but below the 3% gate, same dilution as C1 (FA share grows with KV).
- Prompt throughput neutral on both lanes.

VERDICT: pending user choice. Staged reduce is the only C1-family variant
with zero negative rows: 49K +5.7%, 98K +1.8% (4/4 candidate tasks above
adjacent controls), no K=17408 regression, no register change. It does not
pass the strict dual-lane >=3% protocol, so either it stays opt-in (strict)
or is promoted as a default with the documented 49K-only edge (user
call). If promoted, the legacy dual-buffer branch is removed.

## Measurement plan (when GPUs are free again)

1. Fresh adjacent baseline: dual-ROCm 49K, Qwen3.8-27B-Q4_K_M, f8_e4m3 KV,
   spec=none, 128 out (`q38-rb-rc-49k-f8-none` contract; decode 22.20 tps in
   the rebaseline). Adjacent A-B-A per the thermal-drift rule (7b3aa487):
   interleave control/candidate runs in one fresh session, never long series.
2. C1 A-B-A: control, candidate, control on the 49K lane; 98K confirmation
   only for a >= 3% decode win.
3. Rejected candidates: revert code, document in RESULTS, keep the trace
   infrastructure default-off.

## Artifacts

- Source: `ggml/src/ggml-cuda/mmvq.cu` (dispatch, tables, small_k policy),
  `ggml/src/ggml-cuda/vecdotq.cuh` (vec_dot_q4_K_q8_1).
- This doc; README progress table.
