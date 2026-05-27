# D007 P002 Vulkan FFN Block Route Gate

Date: 2026-05-26

Status: scout gate complete; strict adjacent 4-node whole-FFN fusion is blocked, but a non-adjacent dependency scan recovers the full Q3_K FFN block surface.

## Lane

- Backend: Vulkan, `build-vulkan`.
- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Contract: `ctx=131072,batch=512,ubatch=256,q4_0/q4_0,FlashAttention,spec=none,--no-mmap`.
- Workload: `quick:triage_diff`, `real-context-chars=24576`, cold-first, no reuse, no v2 prime, thinking on.
- Mode: `max_tokens=1` diagnostic trace only. No speed claim.

## Reason

D005 kept a shape-gated split-K win for FFN down, and D006 closed output-layer
residency relief as diagnostic-only. The remaining plausible Vulkan route is a
Q3_K/FFN topology that stacks with D005. Before writing a shader, the workflow
requires proof that the runtime graph exposes a matchable whole block rather
than only separate gate/up and down matmuls.

Historical E135 proved a `MUL_MAT + MUL_MAT + GLU` gate/up surface at 64k. D007
extends that trace for the active 130k lane to ask the stricter question: is
`MUL_MAT + MUL_MAT + GLU + MUL_MAT` adjacent and fusable for the whole dense
FFN block?

## Code Change

`GGML_VK_FFN_ROUTE_TRACE=1` in `ggml/src/ggml-vulkan/ggml-vulkan.cpp` now also prints:

- `blocks=<n>` strict whole-block matches;
- `scan_blocks=<n>` dependency-scanned whole-block matches where down is found by `down->src[1] == GLU`, not by adjacency;
- `ffn_block_trace` shapes for gate/up and down;
- `ffn_scan_block_trace` shapes and graph gaps for non-adjacent blocks;
- reject counters for missing next node, non-matmul next node, source mismatch,
  type mismatch, and `ggml_can_fuse_subgraph` rejection;
- histogram of the next op when the node after GLU is not a down matmul, plus
   source information for rejected next-node `VIEW`s.

The trace is default-off and behavior-neutral when the env var is unset.

Build validation:

```powershell
cmake --build build-vulkan --target llama-server --config Release -j 8
```

Result: passed after the trace extension.

## Trace Evidence

Final trace label: `d007-vulkan-130k-ffn-scanblock-trace-r1`.

Server residency context:

| Metric | Value |
| --- | ---: |
| Vulkan model buffer | `11434.19 MiB` |
| Vulkan host model buffer | `521.00 MiB` |
| Vulkan compute buffer | `228.27 MiB` |
| Vulkan host compute buffer | `138.27 MiB` |
| Graph splits | `2` |
| Prompt eval | `873.78 tok/s` diagnostic-only |

Representative trace lines repeated across the prompt graphs:

```text
ggml_vulkan: ffn_route_trace candidates=64 blocks=17 view_blocks=0 scan_blocks=64 prefill=63 q3=64
ggml_vulkan: ffn_route_trace q3_K glu=SWIGLU m=17408 n=256 k=5120 count=63
ggml_vulkan: ffn_block_trace q3_K glu=SWIGLU gate_up_m=17408 gate_up_n=256 gate_up_k=5120 down_m=5120 down_n=256 down_k=17408 count=16
ggml_vulkan: ffn_scan_block_trace q3_K glu=SWIGLU gap=3 gate_up_m=17408 gate_up_n=256 gate_up_k=5120 down_m=5120 down_n=256 down_k=17408 count=16
ggml_vulkan: ffn_scan_block_trace q3_K glu=SWIGLU gap=4 gate_up_m=17408 gate_up_n=256 gate_up_k=5120 down_m=5120 down_n=256 down_k=17408 count=47
ggml_vulkan: ffn_view_block_trace rejects view_hazard=0 fuse_reject=0 struct_reject=47
ggml_vulkan: ffn_block_trace rejects missing_next=0 next_not_mul_mat=47 src_mismatch=0 type_mismatch=0 fuse_reject=0
ggml_vulkan: ffn_block_trace reject_next_op VIEW count=47
ggml_vulkan: ffn_block_trace reject_next_view src=NONE src_is_glu=0 view_ne=30720x1 glu_ne=17408x256 count=47
```

Interpretation of the counts:

| Surface | Count | Meaning |
| --- | ---: | --- |
| Gate/up + GLU Q3_K candidates | `64` | all dense FFN layers expose the historical gate/up trace surface |
| Prefill gate/up + GLU candidates | `63` | active prompt chunks use `n=256` for 63 layers plus a tail/empty row |
| Strict whole-block candidates | `17` | only `16` prefill blocks plus one tail/empty row are immediately `MUL_MAT,MUL_MAT,GLU,MUL_MAT` |
| Simple `VIEW`-aware block candidates | `0` | the next `VIEW` is not a view of the GLU activation |
| Non-adjacent scanned block candidates | `64` | all Q3_K FFN blocks can be recovered by following the down matmul dependency |
| Rejected because next op is `VIEW` | `47` | most layers have an unrelated `VIEW` node between GLU and down in topological order |
| Source/type/fuse rejects | `0` | the blocker is adjacency/view structure, not Q type mismatch or subgraph use count |

## Decision

Do not start a shader prototype for a strict adjacent 4-node whole-FFN route. It
would cover only about a quarter of the prefill block surface before considering
implementation overhead.

The simple `VIEW`-aware matcher was also the wrong model: the rejected next node
is `src=NONE src_is_glu=0 view_ne=30720x1`, so it is not the GLU activation on a
no-data view path. The graph is merely non-adjacent for most layers. A dependency
scan that looks for the down matmul by `src[1] == GLU` recovers `64/64` Q3_K
blocks and passes the subgraph-use gate for the non-adjacent node set.

Therefore the next valid candidate is not a strict adjacency shader. It is a
non-adjacent whole-FFN design that explicitly handles the intermediate GLU
activation, the unrelated topological node at `gap=4`, and the existing D005
split-K down route.

D007 keeps the trace extension because it is default-off and is now the cheapest
guard against implementing a low-coverage FFN fusion. This is a source-level
major-topology gate, not a benchmark tuning branch.

## Next Source Gate

1. Recompute the local speedup ceiling using `scan_blocks=64` coverage, with the
   post-D005 dense FFN share and the retained D005 down split-K behavior.
2. Write the non-adjacent whole-FFN route design: node scheduling contract,
   intermediate activation residency, memory traffic, temporary buffer size, and
   rollback/env guard.
3. Add a tiny correctness scout for prompt and decode graph invariants before any
   shader/runtime prototype.
4. Only after the design gate, build a default-off prototype and compare
   SPIR-V/resource deltas before server A/B.

Stop condition: if the non-adjacent design cannot reduce memory traffic or route
launch/work enough to clear the post-D005 ceiling, abandon whole-block FFN fusion
and move back to an all-Q3 layout or narrower route with an honest ceiling model.