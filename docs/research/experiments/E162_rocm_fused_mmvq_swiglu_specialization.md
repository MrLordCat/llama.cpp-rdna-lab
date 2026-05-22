# E162 ROCm Fused MMVQ SWIGLU Specialization Gate

## Metadata

- Experiment ID: E162
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after `1cf415f0f`, temporary code reverted
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: The fused FFN MMVQ route can improve by specializing the common
  Qwen path `gate + SWIGLU` with no bias at compile time.
- Mechanism: The existing `mul_mat_vec_q<..., has_fusion=true>` keeps `use_gate`,
  bias flags, and `active_glu` as runtime checks. A template path for
  `fusion.gate != nullptr`, no bias, and `GGML_GLU_OP_SWIGLU` could remove
  branches from the hot Q3_K decode loop.
- Risk: The specialization may increase code/register pressure or change
  compiler scheduling enough to lose occupancy, even if it removes branches.

## Method

Temporary code:

- added `fast_swiglu_no_bias` as a template parameter to `mul_mat_vec_q`;
- routed `fusion.gate != nullptr && fusion.x_bias == nullptr &&
  fusion.gate_bias == nullptr && fusion.glu_op == GGML_GLU_OP_SWIGLU` to the
  specialized kernel;
- left generic fusion as fallback.

Built with:

```powershell
cmake --build build-rocm-vec --config Release -j
```

Bench command shape:

```powershell
python scripts\agent_workload_bench.py --label e162-rocm-decode-q4-fast-swiglu-r1 --out-dir build_logs\agent-workload --tasks quick --task-ids triage_diff --runs 1 --server-bin build-rocm-vec\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --gpu-layers=-1 --parallel 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --no-disable-thinking --max-tokens 128 --no-reuse --no-v2-prime-pass --server-extra "--spec-type none" --write-diagnostics
```

## Result

| Run | Aggregate TPS | Decode eval |
| --- | ---: | ---: |
| E151 promoted r3 reference | `30.3145` | `32.2467 tok/s` |
| E161 clean post-revert r3 reference | `30.1073` | `32.03 tok/s` |
| E162 fast SWIGLU r1 | `28.0887` | `30.33 tok/s` |

The candidate was bad enough on r1 to stop without r3 confirmation.

## Decision

- Reject and revert.
- Branch elimination is not a sufficient reason to specialize this fused kernel;
  the likely cost is extra code/register pressure or worse compiler scheduling.
- Future fused-MMVQ work should start from resource/instruction evidence, not
  from generic runtime-branch removal.

## Artifacts

- `build_logs/agent-workload/e162-rocm-decode-q4-fast-swiglu-r1.diagnostics.md`
- `build_logs/agent-workload/e162-rocm-decode-q4-fast-swiglu-r1.server.log`
