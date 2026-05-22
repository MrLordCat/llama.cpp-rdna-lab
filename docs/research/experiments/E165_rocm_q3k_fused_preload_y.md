# E165 ROCm Q3_K Fused Preload-Y Probe

## Metadata

- Experiment ID: E165
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E163 resource trace, temporary code reverted
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: Fused Q3_K FFN MMVQ can improve by loading the shared `q8_1` activation block once for the up and gate dot products.
- Mechanism: The generic fused loop calls `vec_dot_q3_K_q8_1` for `vx` and `vgate` separately with the same `y` block. A Q3_K-only fused path can preload `u`/`d8` once and reuse it for both X matrices.
- Risk: The preloaded path may increase live values and register pressure enough to lose occupancy.

## Analytical Gate

E163 showed the fused Q3_K buckets dominate parsed Q3_K MMVQ time:

- fused `ncols_x=5120`: `341.640 ms`;
- fused `ncols_x=17408`: `211.049 ms`;
- direct Q3_K buckets are smaller.

With decode wall share about `92.9%` on this short decode gate, even a `2%` decode-kernel win would project to about `1.0186x` wall speedup, while `5%` decode-kernel win projects to about `1.0463x`. That was enough ceiling to justify a build probe.

## Method

Temporary code:

- added Q3_K MMVQ helpers to preload `q8_1` values;
- used them only inside `has_fusion && type == GGML_TYPE_Q3_K && use_gate`;
- left direct and non-Q3_K routes unchanged.

Bench command used active H39:

```powershell
python scripts\agent_workload_bench.py --label e165-rocm-decode-q4-q3-fused-preload-y-r1 --out-dir build_logs\agent-workload --tasks quick --task-ids triage_diff --runs 1 --server-bin build-rocm-vec\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --gpu-layers=-1 --parallel 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --no-disable-thinking --max-tokens 128 --no-reuse --no-v2-prime-pass --server-extra "--spec-type none" --write-diagnostics
```

## Results

| Stack | Aggregate TPS | Decode eval | Decision |
| --- | ---: | ---: | --- |
| post-revert clean r1 | `28.8295` | `31.18 tok/s` | control |
| E165 preload-y r1 | `29.1718` | `31.26 tok/s` | noise; not enough |

Resource trace explained the weak runtime signal:

| Fused bucket | E163 clean | E165 preload-y | Change |
| --- | ---: | ---: | --- |
| `ncols_x=5120`, `grid.x=8704` | `0.355 ms`, `84 regs`, `87.5% occ` | `0.376 ms`, `136 regs`, `68.75% occ` | slower |
| `ncols_x=17408`, `grid.x=2560` | `0.219 ms`, `84 regs`, `87.5% occ` | `0.223 ms`, `136 regs`, `68.75% occ` | slower |
| `ncols_x=6144`, `grid.x=2560` | `0.126 ms`, `84 regs`, `87.5% occ` | `0.123 ms`, `136 regs`, `68.75% occ` | small local win, low share |

## Decision

- Reject and revert.
- The expected memory-load saving was real in concept, but the implementation kept too many live values. Register pressure rose `84 -> 136`, dropping occupancy from `87.5%` to `68.75%` and slowing the dominant fused bucket.
- Workflow correction: fused Q3_K work must check register count before any r3 promotion. Avoid preloading schemes that duplicate live arrays unless they keep regs near the E163 baseline.

## Artifacts

- `build_logs/agent-workload/e165-rocm-decode-q4-q3-fused-preload-y-r1.diagnostics.md`
- `build_logs/agent-workload/e165-rocm-decode-q4-q3-fused-preload-y-resources-r1.server.log`
