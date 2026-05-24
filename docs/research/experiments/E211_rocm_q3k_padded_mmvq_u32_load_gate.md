# E211 ROCm Q3_K Padded MMVQ U32 Load Gate

## Metadata

- Experiment ID: E211
- Date: 2026-05-24
- Owner: Codex
- Branch/Commit: master after `e15f2923d`
- Target lane: H43/H39 ROCm Q3_K padded-storage decode path, `build-rocm-vec`

## Hypothesis

- Statement: after H43 gives ROCm Q3_K a real 112-byte physical stride, the padded MMVQ helpers can safely use 32-bit packed loads for `qs`/`hmask` instead of reconstructing each 32-bit word from two 16-bit loads.
- Mechanism: change only the padded-storage Q3_K MMVQ helpers (`block_q3_K_padded`) to use `get_int_b4` where the padded layout guarantees 4-byte block alignment. The raw Q3_K route remains unchanged.
- Why this is not the rejected path: E199 rejected storage-blind packed-load rewrites while ROCm still used raw 110-byte blocks. This gate is storage-aware and only active under `GGML_CUDA_Q3K_PADDED_STORAGE=1`.

## Math / Theory

- Assumptions:
  - decode-side Q3_K MMVQ remains a meaningful share of the short decode lane;
  - if the load reduction matters, synchronized MMVQ point time should move before any wall run;
  - if point-ms is neutral, this is another helper-level dead end and must be reverted.
- Expected speedup corridor:
  - small: likely sub-1% wall unless the dominant fused buckets move clearly.
- Failure conditions:
  - build fails;
  - padded Q3_K `MUL_MAT` smoke fails;
  - MMVQ point timing is neutral/slower;
  - output sanity regresses.

## Implementation Plan

1. Baseline point trace with current padded route:
   - `GGML_CUDA_Q3K_PADDED_STORAGE=1`
   - `GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1`
   - `GGML_CUDA_DISABLE_GRAPHS=1`
   - `GGML_TRACE_MMVQ_TIMING=1`
   - `GGML_TRACE_MMVQ_TIMING_SYNC=1`
   - `GGML_TRACE_MMVQ_TIMING_PRE_SYNC=1`
   - `GGML_TRACE_MMVQ_RESOURCES=1`
2. Patch only padded Q3_K MMVQ helper loads.
3. Rebuild `build-rocm-vec`.
4. Repeat the same point trace and compare robust MMVQ timing rows.
5. Wall A/B only if point-ms moves enough to matter.

## Benchmark Plan

- Baseline command: short decode-focused `agent_workload_bench.py`, `quick/triage_diff`, `max_tokens=32`, no reuse, no real-context, graphs disabled.
- Candidate command: same after patch.
- Number of runs: point `r1`; wall only on point win.
- Artifacts path: `build_logs/agent-workload/e211-rocm-q3k-padded-mmvq-u32-*`.

## Metrics

- Q3_K MMVQ synchronized `total_ms`, robust sum excluding startup outliers
- resource telemetry (`regs`, occupancy)
- wall/decode TPS only if promoted

## Result

- Outcome: rejected and reverted before wall A/B.
- Delta:
  - build passed and padded Q3_K `MUL_MAT` smokes passed after the temporary patch;
  - point trace rows matched (`9281` Q3_K MMVQ timing rows on both sides);
  - baseline robust MMVQ sum (`total_ms < 10`) was `1363.700 ms`;
  - candidate robust MMVQ sum was `1472.573 ms`, a `+108.873 ms` regression (`+7.98%` slower);
  - trace context wall also moved the wrong way: `10.97 -> 10.24 TPS`;
  - dominant fused bucket `ncols_dst=1,fusion=1,ncols_x=5120,gridx=8704` regressed `467.365 -> 509.874 ms`, while registers rose `94 -> 97` and occupancy fell `100.00% -> 87.50%`;
  - dominant fused bucket `ncols_dst=1,fusion=1,ncols_x=17408,gridx=2560` regressed `301.544 -> 327.206 ms` with the same register/occupancy cliff.
- Confidence: high for rejecting this exact helper. The candidate changed only padded Q3_K loads, route counts matched, and the point gate showed a large enough regression that no wall confirmation is useful.
- Recommendation: do not keep the patch. The likely cause is increased live state/register pressure: replacing two 16-bit loads with one 32-bit load looked cheaper, but ROCm codegen raised VGPR enough to reduce occupancy on the fused hot buckets.

## Notes

- This is deliberately a cheap gate. It must not become a new helper-polishing loop if point timing does not move.
- The temporary code patch was reverted and `build-rocm-vec` was rebuilt back to the prior helper.
- Artifacts:
  - `build_logs/agent-workload/e211-rocm-q3k-padded-mmvq-u32-baseline-point-r1.server.log`
  - `build_logs/agent-workload/e211-rocm-q3k-padded-mmvq-u32-baseline-point-r1.diagnostics.md`
  - `build_logs/agent-workload/e211-rocm-q3k-padded-mmvq-u32-candidate-point-r1.server.log`
  - `build_logs/agent-workload/e211-rocm-q3k-padded-mmvq-u32-candidate-point-r1.diagnostics.md`
