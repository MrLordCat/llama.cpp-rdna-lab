# E166 ROCm Q3_K Fused Pair-Dot Probe

## Metadata

- Experiment ID: E166
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E163 resource trace, temporary code reverted
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: The fused Q3_K FFN MMVQ path can improve if the X and gate dots are computed by one paired helper.
- Mechanism: The current fused loop calls the Q3_K MMVQ dot twice with the same `q8_1` activation block. A paired helper can reuse activation loads while decoding the two Q3_K weight blocks side by side.
- Risk: Interleaving two Q3_K dots may increase register pressure and instruction scheduling pressure enough to offset load reuse.

## Analytical Gate

E163 showed fused Q3_K MMVQ remains the main residual decode bucket:

- fused `ncols_x=5120`, `grid.x=8704`: `341.640 ms`, `34.99%` of parsed MMVQ time;
- fused `ncols_x=17408`, `grid.x=2560`: `211.049 ms`, `21.61%`;
- fused `ncols_x=6144`, `grid.x=2560`: `28.421 ms`, `2.91%`.

The ceiling was high enough to test only if the paired helper stayed close to the E163 register baseline (`84 regs`).

## Method

Temporary code:

- added `vec_dot_q3_K_q8_1_pair_mmvq`;
- used it only for `has_fusion && type == GGML_TYPE_Q3_K && use_gate`;
- left direct Q3_K and all non-Q3_K MMVQ routes unchanged.

Candidate command:

```powershell
python scripts\agent_workload_bench.py --label e166-rocm-decode-q4-q3-fused-pair-dot-r3 --out-dir build_logs\agent-workload --tasks quick --task-ids triage_diff --runs 3 --server-bin build-rocm-vec\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --gpu-layers=-1 --parallel 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --no-disable-thinking --max-tokens 128 --no-reuse --no-v2-prime-pass --server-extra "--spec-type none" --write-diagnostics
```

Resource trace command used the same lane with `--max-tokens 16` and `GGML_TRACE_MMVQ_RESOURCES=1`.

## Results

| Stack | Aggregate TPS | Decode eval | Decision |
| --- | ---: | ---: | --- |
| E151 promoted best | `30.3145` | `32.2467 tok/s` | baseline |
| E166 paired dot r1 | `29.9253` | `32.14 tok/s` | promising but not enough |
| E166 paired dot r3 | `29.84` | `31.11 / 32.10 / 31.98 tok/s` | below best |

Resource trace:

| Fused bucket | E163 clean | E166 paired dot | Change |
| --- | ---: | ---: | --- |
| `ncols_x=5120`, `grid.x=8704` | `0.355 ms`, `84 regs`, `87.5% occ` | `0.362 ms`, `95 regs`, `100% occ` | slower |
| `ncols_x=17408`, `grid.x=2560` | `0.219 ms`, `84 regs`, `87.5% occ` | `0.218 ms`, `95 regs`, `100% occ` | tie/noise |
| `ncols_x=6144`, `grid.x=2560` | `0.126 ms`, `84 regs`, `87.5% occ` | `0.123 ms`, `95 regs`, `100% occ` | small local win, low share |
| Total parsed MMVQ trace | `1075.567 ms` | `1081.870 ms` | slower |

## Decision

- Reject and revert.
- The hypothesis was plausible because it reused `q8_1` activation loads, but the paired helper raised registers `84 -> 95` and did not improve the dominant `5120 -> 8704` fused bucket.
- Workflow correction: do not promote Q3_K fused pair/interleave helpers on occupancy alone. The gate must require shape-level improvement in the highest-share fused bucket, especially `ncols_x=5120`, `grid.x=8704`.

## Artifacts

- `build_logs/agent-workload/e166-rocm-decode-q4-q3-fused-pair-dot-r1.diagnostics.md`
- `build_logs/agent-workload/e166-rocm-decode-q4-q3-fused-pair-dot-r3.diagnostics.md`
- `build_logs/agent-workload/e166-rocm-decode-q4-q3-fused-pair-dot-resources-r1.server.log`
