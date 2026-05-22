# E164 ROCm Q3_K Direct RPB4 Probe

## Metadata

- Experiment ID: E164
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E163 resource trace, temporary code reverted
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: The direct Q3_K decode branch might improve if packed at `rows_per_block=4`, while fused Q3_K stays at the E151 `rows_per_block=2`.
- Mechanism: Direct Q3_K is about one third of parsed Q3_K MMVQ time after E151. Larger row packing halves direct grid count and may reuse the same activation block across more rows.
- Risk: Fewer blocks can reduce grid-level parallelism and latency hiding even if reported occupancy improves.

## Method

Temporary code changed only RDNA4 Q3_K direct small-k packing:

- fused Q3_K: kept `rows_per_block=2`;
- direct Q3_K: changed to `rows_per_block=4`.

Bench command used the active H39 decode gate:

```powershell
python scripts\agent_workload_bench.py --label e164-rocm-decode-q4-direct-rpb4-r1 --out-dir build_logs\agent-workload --tasks quick --task-ids triage_diff --runs 1 --server-bin build-rocm-vec\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --gpu-layers=-1 --parallel 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --no-disable-thinking --max-tokens 128 --no-reuse --no-v2-prime-pass --server-extra "--spec-type none" --write-diagnostics
```

## Results

| Stack | Aggregate TPS | Decode eval | Decision |
| --- | ---: | ---: | --- |
| E164 direct rpb4 r1 | `28.6520` | `30.96 tok/s` | reject |
| post-revert clean r1 | `28.8295` | `31.18 tok/s` | recovery sanity |

Resource comparison against E163:

| Direct bucket | E163 clean avg | E164 rpb4 avg | Resource change |
| --- | ---: | ---: | --- |
| `ncols_x=5120`, main grid | `0.156 ms` | `0.160 ms` | regs `88 -> 45`, occupancy `87.5% -> 100%`, grid halved |
| `ncols_x=5120`, mid grid | `0.124 ms` | `0.128 ms` | same resource pattern |
| `ncols_x=5120`, small grid | `0.084 ms` | `0.093 ms` | same resource pattern |

## Decision

- Reject and revert.
- The initial expectation was wrong because resource relief was not the limiter. Larger packing improved reported occupancy but removed too much grid-level parallelism/latency hiding.
- Workflow correction: for MMVQ row packing, treat `max_blocks_per_sm` as secondary. Require shape-level timing improvement, because higher occupancy can still be slower when the grid gets smaller.

## Artifacts

- `build_logs/agent-workload/e164-rocm-decode-q4-direct-rpb4-r1.diagnostics.md`
- `build_logs/agent-workload/e164-rocm-decode-q4-direct-rpb4-resources-r1.server.log`
- `build_logs/agent-workload/e164-rocm-decode-q4-postrevert-clean-r1.diagnostics.md`
