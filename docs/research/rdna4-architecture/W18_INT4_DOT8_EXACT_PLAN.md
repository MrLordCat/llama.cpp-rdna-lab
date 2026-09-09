# W18: Exact INT4 dot8 for Q4_K MMVQ (no-quality-loss plan)

Date: 2026-09-08

Scope: reduce compute load of the decode-hot `mul_mat_vec_q` Q4_K kernels
without changing any arithmetic result (bitwise-safe integer equivalence).
This follows W19 (decode bottleneck diagnosis): MMVQ q4_K fused = 45.9% / all
MMVQ = 86.7% of GPU kernel time, and the kernel is compute/issue-bound
(SQ_BUSY_CYCLES close to total cycles), not bandwidth-bound (377-381 GB/s = 59%
of peak with 100% occupancy; W16 prefetch did not help).

## Why this is possible without quality loss

Q4_K decode currently computes, per 32-element group, the exact integer dot
`sum_i w_i * a_i` where:

- `w_i` = 4-bit unsigned weights (0..15), stored packed as 4-bit nibbles,
  then byte-expanded for the current `v_dot4_i32_iu8` DP4A;
- `a_i` = activation bytes from `block_q8_1` (raw unsigned bytes 0..255).

On RDNA4 `ggml_cuda_dp4a` lowers to `__builtin_amdgcn_sudot4` =
`v_dot4_i32_iu8` (4 MAC/instruction). gfx1201 also has:

- `v_dot8_i32_i4` (8 x signed4, i32 acc) - `dot1-insts`;
- `v_dot8_i32_iu4` (8 x mixed signed4) - `dot8-insts`;
- `v_dot8_u32_u4` (8 x unsigned4) - `dot7-insts` (assembler-verified on gfx1201).

Packed nibble decomposition of an unsigned byte is exact:

```
a = lo + 16 * hi,   lo = a & 0xF,  hi = (a >> 4) & 0xF
```

Therefore, with 8 sequential weights packed into one u32 (`w` nibbles) and the
corresponding 8 activations packed into `lo`/`hi` u32s:

```
sum8_i w_i * a_i
  = sum8_i w_i * lo_i  +  16 * sum8_i w_i * hi_i
  = v_dot8_u32_u4(w, lo) + 16 * v_dot8_u32_u4(w, hi)
```

Both dot8 results are i32 exact (max 8*15*15 = 1800 per dot; 16x mul remains in
i32), and the accumulation is exact - the result is bit-identical to the
current integer stage. Quality is untouched because the float scaling chain
(`d8`, `sc`, `dmin`/`m`, `ds`) stays the same.

Instruction count (per MAC): DP4A = 4 MAC/instr; dot8 = 8 MAC/instr. So the
productive dot (`dot1`) halves its instruction count, at the price of a
nibble-extract/pack pass for the activation bytes (which are 8-bit in group
size 32 and reused across all weight rows in the same K window).

## Measured (2026-09-08) - dot8 IPC, exact decomposition and the real lever

- Host emulator: exact `lo + 16*hi` decomposition of q8_1 bytes into two
  `dot8_u32_u4` (weights u4 x act u4) matches the current DP4A reference
  **bit-exactly for both dot1 and dot2** (200k random groups, 0 fails).
- GPU microbenchmark (8 independent accumulators, gfx1201): `v_dot4_i32_iu8`
  = 744 Gi/s (2978 GMAC/s), `v_dot8_u32_u4` = 745 Gi/s (5963 GMAC/s)
  -> **same instruction rate, 2x MAC/instruction** (ratio 1.00x / 2.00x).
- **Decision**: dot8 is NOT a win for 8-bit activations - the exact
  decomposition needs `dot8(w,lo) + 16*dot8(w,hi)` = 2 instructions for the
  same 8 MACs that 2x DP4A already do in 2 instructions. The 2x MAC rate
  cannot be exploited without losing quality (4-bit activations) or doubling
  instructions (lo+hi), which cancels it out.
- **Dot2-hoist prototype MEASURED - no gain (REJECTED)**. Implemented
  `dot2` (q8_1 activation sum, used only for the exact `dmin` correction)
  precomputed once per `(kby, kqs)` in `mul_mat_vec_q` and reused across the
  `rows_per_cuda_block=8` rows and gate path - removes ~50% of the DP4A
  instructions in the Q4_K dot loop with **bit-identical output** (smoke
  `content equal: True`, temp 0).
  A-B-A on L1 (7901 prompt / 128 decode, f8 KV, spec none, ROCm1,ROCm0,
  no-warmup, seed 42):
  | run | decode tok/s | prefill tok/s |
  | --- | --- | --- |
  | A (base, hoist off) | 24.1759 | 1908.93 |
  | B (hoist on) | 24.2095 | 1905.35 |
  | A2 (base) | 24.2057 | 1899.49 |
  -> B vs interpolated control (24.1908) = **+0.08% decode**, prefill -0.15%
  (noise). **Verdict: no gain.**
  Artifacts: `build_logs/bench/w18-dot2hoist/{w18-base-a,w18-cand-b,w18-base-a2}-*`;
  binary snapshot `/tmp/w18-snap/`; env `GGML_MMVQ_Q4K_DOT2_HOIST` (default off).
- **Interpretation (updates W19)**: removing ~50% of DP4A has no effect on
  decode time -> the MMVQ kernel is NOT DP4A/compute-latency-bound; SQ_BUSY
  near 100% also counts memory-issue cycles. The limiter is the memory system
  (weight stream arrival / issue of loads) - consistent with W16 lesson
  ("must change memory system or geometry, not add instructions") and with
  W13's 40-45% of peak BW. Next candidates: weight-stream geometry/coalescing
  (e.g. wider per-thread K windows, L2 residency/preload channel), not
  arithmetic.

## Implementation plan (staged, no-loss guarantee)

0. **Hoist dot2 (REJECTED after measurement)** - prototype + A-B-A done,
   no gain (+0.08% decode), bit-identical. Code reverted to HEAD
   (e2... via clean rebuild); env-gate removed with the revert.
1. **Weight-stream geometry prototype (next)** - change *memory system*:
   - measure real per-kernel BW + cache-line efficiency with a minimal SPM
     group (GL2C_EA_RDREQ_{32,64,128,256}B, GL2C_HIT/MISS) - not yet
     successfully collected (needs 2-4 counters, previous group errored);
   - if 32B/64B requests dominate, increase per-thread K-window so each
     weight load is 128B (L2 cache-line) - or use vector loads to force
     full-line fetches;
   - validate via same bit-exact smoke + A-B-A.
2. **Register-pressure check**: hoisting adds `s0/s1` (2 SGPR/VMEM) but
   removed dot2 DP4A; verify occupancy stays 100% via
   `GGML_TRACE_MMVQ_RESOURCES` (only if the geometry idea survives).
3. **A/B in production**: env-gated again, smoke byte-identical, then bench2
   L2 A-B-A decode (Q4_K_M 27B, ctx 49152, f8 KV, spec none, ROCm1,ROCm0).
   Acceptance > +1.5% decode with no prefill regression.
4. (closed) dot8 exact decomposition and dot2 hoist - no gain without
   quality loss; keep documented as closed.

## Expected outcome

- Hoist dot2: measured practically zero -> next candidates are memory-
  system/geometry changes, not instruction reduction.
- Any future compute-side change that only removes arithmetic must be
  micro-validated (this prototype shows arithmetic is not the limiter).
