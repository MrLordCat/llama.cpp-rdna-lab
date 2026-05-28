# D067 P003 H54-B value-aware quantization fast gate

Date: 2026-05-28
Owner: Copilot/perf workspace
Mode: theory-only (fast analytical gate)

## Context

After D065, H54-A (rotation/permutation family) was rejected by permutation invariance
of Shannon entropy. The next branch is H54-B: change quantization itself via
value-aware codebooks (Lloyd-Max style) and re-check payload entropy corridor.

## Hypothesis

H54-B: value-aware 16-level scalar quantization can shift index distributions enough
to move effective payload entropy below the C2 corridor upper bound (`3.77 bpw`).

## Method

Script:

- `scripts/research/q4_c2_value_aware_gate.py`

Run:

- `python scripts/research/q4_c2_value_aware_gate.py --model models/Qwen3.6-27B-Q4_K_S.gguf --sample 24 --max-elements-per-tensor 262144 > build_logs/agent-workload/q4c2-value-aware-qwen36-27b-q4ks-fast-r1.log`

Procedure:

1. Parse Q4 tensors from GGUF.
2. Decode baseline nibble indices and compute original entropy.
3. Dequantize sampled blocks to float values.
4. Build Lloyd-Max 16-level codebook per tensor sample.
5. Re-quantize with the learned codebook and compute new entropy.

## Results

From `build_logs/agent-workload/q4c2-value-aware-qwen36-27b-q4ks-fast-r1.log`:

- Tensors analyzed: `24`
- Total elements analyzed: `6,291,456`
- Original entropy: `3.870042 bpw`
- New entropy: `3.267969 bpw`
- Delta: `-0.602073 bpw`
- Feasible tensors (`< 3.77 bpw`): `24/24`

Gate verdict:

- **PASS (fast gate)**: sampled evidence is strongly below corridor upper bound.

## Interpretation

This is the first positive signal after H45-H53 and H54-A rejections.
Unlike rotation/permutation routes, H54-B changes quantization mapping itself,
so entropy reduction is not blocked by permutation invariance.

## Constraints and caveats

- This is sampled fast-gate evidence, not full unsampled corpus closure.
- MSE was logged as an absolute value only; no normalized quality envelope gate
  has been enforced yet.
- No runtime/decode complexity budget has been re-scored yet for H54-B.

## Decision

- Keep H54-B open and advance to full analytical definition gate.

## Next

1. Run full unsampled H54-B analytical pass (or broad representative pass with
   strict convergence criteria) and record corpus-wide entropy statistics.
2. Add quality-side gate: normalized error delta vs baseline quantizer.
3. Build a first decode-complexity proxy for value-aware codebook lookup path.
4. Re-enter Ck-4/Ck-5 style feasibility gate with updated H54-B data.
