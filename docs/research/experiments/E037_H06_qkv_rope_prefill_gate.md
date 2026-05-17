# E037 H06 QKV+RoPE Prefill Gate

## Metadata

- Experiment ID: E037
- Date: 2026-05-17
- Owner: Copilot
- Type: implementation gate (code-anchored, no kernel changes yet)
- Target lane: C01 (`ctx=12288`, `b=6144`, `ub=192`, no-reuse)

## Hypothesis

- Statement: fusing Q/K/V projection and RoPE-adjacent transforms in the prefill hot path can reduce launch count and intermediate memory traffic.
- Why now: most low-risk MMQ selector probes are closed, while attention-adjacent nodes still keep a measurable share.

## Code Anchors (verified)

- `src/llama-graph.cpp`
  - helper `ggml_mul_mat_aux(...)` (used for K/V rotation matrix application).
  - `build_attn(...)` overloads for `llm_graph_input_attn_kv`, `llm_graph_input_attn_k`, `llm_graph_input_attn_kv_iswa`:
    - separate `q_cur/k_cur/v_cur` transforms,
    - separate `ggml_build_forward_expand(...)` of q/v/k,
    - separate KV cache writes via `cpy_k` and `cpy_v`.
- `src/llama-kv-cache.cpp`
  - `build_input_k_rot(...)`, `build_input_v_rot(...)`, and `build_rope_shift(...)` show current rotation/rope dataflow is not fused with QKV projection path.

## Measured Gate Snapshot

Using `build_logs/agent-workload/decode-trace-current-ctx12288-ub192-r1.server.log`:

- total `GGML_TRACE_CUDA_NODE_TIMING`: `3303.800 ms`
- attention/QKV/RoPE-related node-name slice (regex gate): `575.093 ms` (`17.41%`)

Ceiling estimate (CUDA_NODE-local):

- 10% slice gain -> `~+1.74%`
- 20% slice gain -> `~+3.48%`

This is high enough to justify a guarded prototype.

## Minimal Safe Prototype Plan

1. Add an env-gated graph-path experiment flag in graph build path only (default off).
2. Keep existing unfused path as strict fallback.
3. First prototype scope:
   - target only `build_attn(...)` path with `self_k_rot/self_v_rot` active,
   - avoid touching SWA and cross-attention branches in v1.
4. Verify graph correctness/equivalence first (shape/type and deterministic output checks), then lane benchmarks.

## Acceptance Criteria

- Runtime gate:
  - r1 screen: non-negative and target-hotspot-positive,
  - r3 confirm: `>= +1.5%` aggregate TPS on C01 clean lane.
- Trace gate:
  - reduced attention-adjacent node sum,
  - no new regressions in MMQ q3 bucket.
- Safety:
  - default path unchanged when env is off.

## Decision

- Verdict: `proceed to guarded prototype`
- Next step: implement env-gated graph-path fusion attempt in `src/llama-graph.cpp` and run paired r1 + trace.
