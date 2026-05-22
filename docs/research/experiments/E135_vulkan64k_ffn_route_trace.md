# E135 Vulkan 64k FFN Route Trace

## Metadata

- Experiment ID: E135
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E134
- Hypothesis ID: H38 / H31
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan `ctx=65536`, q4/q4 KV, FlashAttention on, real server repo-snapshot, no reuse

## Hypothesis

- Statement: before implementing a large Vulkan FFN route, prove that the active 64k graph exposes the dense `MUL_MAT + MUL_MAT + GLU` pattern as one matchable branch.
- Mechanism: Qwen dense FFN builds sibling gate/up `MUL_MAT` nodes and consumes them through `SWIGLU`. If Vulkan can detect this branch in `ggml_backend_vk_graph_compute(...)`, a future route can operate on the pair instead of treating the two large Q3_K matmuls and the GLU as unrelated nodes.
- Why now: E134 showed that single micro-probes are below the required ceiling. A complex FFN branch needs graph-pattern proof before shader/resource work.

## Code Change

Added default-off Vulkan diagnostic tracing:

```powershell
$env:GGML_VK_FFN_ROUTE_TRACE = "1"
```

The trace is behavior-neutral unless the env var is set. It checks:

- adjacent `MUL_MAT`, `MUL_MAT`, `GLU` subgraph fusion eligibility through `ggml_can_fuse_subgraph`;
- gate/up tensors are the actual GLU inputs, including swapped order;
- same Q weight type, shape, stride, activation source, and optional ids tensor;
- supported GLU ops: `SWIGLU`, `GEGLU`, `SWIGLU_OAI`;
- non-swapped GLU parameter.

## Real Server Trace

Command:

```powershell
$env:GGML_VK_FFN_ROUTE_TRACE='1'
python scripts\repo_snapshot_context_bench.py `
  --label-prefix e134-vulkan64k-ffn-route-trace `
  --server-bin build-vulkan\bin\llama-server.exe `
  --model models\Qwen3.6-27B-Q3_K_S.gguf `
  --ctx-values 65536 `
  --allow-ctx-above-16k `
  --batch-size 8192 `
  --ubatch-size 1024 `
  --cache-type-k q4_0 `
  --cache-type-v q4_0 `
  --gpu-layers 999 `
  --max-tokens 1 `
  --base-ctx 65536 `
  --base-char-budget 152000 `
  --min-char-budget 152000 `
  --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0 --no-mmap --flash-attn on" `
  --request-timeout 1200
Remove-Item Env:\GGML_VK_FFN_ROUTE_TRACE
```

Result:

| Metric | Value |
| --- | ---: |
| Prompt chars | `152000` |
| Prompt tokens | `57611` |
| Completion tokens | `1` |
| Prompt eval | `660.89 tok/s` |
| Server prompt time | `87172.42 ms` |
| Result | real server completed |

Trace evidence:

```text
ggml_vulkan: ffn_route_trace candidates=64 prefill=63 q3=64
ggml_vulkan: ffn_route_trace q3_K glu=SWIGLU m=17408 n=1024 k=5120 count=63
```

There is also one non-prefill/tail candidate in each graph (`n=0` or `n=1/2/267`, depending on split). The important result is stable: the active 64k prefill graph exposes `63 x q3_K SWIGLU` dense FFN gate/up candidates with the E133 hot shape `m=17408,n=1024,k=5120`.

## Interpretation

- The graph hook for a whole FFN route is real on the target server lane.
- The candidate covers the gate/up half of the dense FFN block, not the down projection. E134 says this route alone still needs an unrealistic `2.234x` local speedup to close the full Vulkan-vs-ROCm 64k gap, so it must be treated as one component of a Q3_K/FA route stack.
- A launch-only fusion is still rejected. The trace only proves matchability; a useful implementation must reuse the activation/B tile, reduce memory traffic, or change the Q3_K layout enough to move the local Q3_K time.
- The next gate is resource proof. A dual-A/same-B Q3_K SwiGLU shader likely doubles accumulators; if that breaks coopmat/no-scratch or raises VGPR too far above the current `113 VGPR / 20480 B LDS` Q3_K route, switch to a backend-private Q3_K repack/layout branch.

## Result

- Outcome: keep diagnostic trace; no speed claim.
- Decision: proceed to resource-gated design for the dense FFN Q3_K route, while keeping FA long-KV redesign as the co-primary branch.
- Confidence: high that the graph route exists; medium/low that dual-A fusion is resource-safe without a smaller tile profile.

## Artifacts

- `build_logs/agent-workload/e134-vulkan64k-ffn-route-trace-ctx64k.server.log`
- `build_logs/agent-workload/e134-vulkan64k-ffn-route-trace-repo-summary.md`
- `docs/research/experiments/E135_vulkan64k_ffn_route_trace.md`
