# E150 ROCm Decode Fusion Gate

## Metadata

- Experiment ID: E150
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master worktree, no runtime code changes
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: If the fused ROCm Q3_K MMVQ path is a bad route, disabling graph fusion should improve short decode.
- Mechanism: E149 route-delta showed ROCm decode-only Q3_K time is dominated by `mul_mat_vec_q_fused` (`63.78%` of parsed Q3_K matvec time). A no-code fusion-disable gate can tell whether the next branch should optimize the fused kernel or avoid it.
- Risk: r1 short gates are noisy; this is a direction gate, not a promoted speed claim.

## Method

Before the run:

- confirmed no background `llama-server`;
- cleared `HSA_OVERRIDE_GFX_VERSION`;
- cleared trace env variables;
- used `--runs 1`, `--max-tokens 128`, `triage_diff`;
- kept `--no-disable-thinking`, `--no-reuse`, `--no-v2-prime-pass`, and `--spec-type none`.

Control:

```powershell
python scripts\agent_workload_bench.py --label e150-rocm-decode-q4-clean-r1 --out-dir build_logs\agent-workload --tasks quick --task-ids triage_diff --runs 1 --server-bin build-rocm-vec\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --gpu-layers=-1 --parallel 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --no-disable-thinking --max-tokens 128 --no-reuse --no-v2-prime-pass --server-extra "--spec-type none" --write-diagnostics
```

Candidate:

```powershell
$env:GGML_CUDA_DISABLE_FUSION = "1"
python scripts\agent_workload_bench.py --label e150-rocm-decode-q4-disablefusion-r1 --out-dir build_logs\agent-workload --tasks quick --task-ids triage_diff --runs 1 --server-bin build-rocm-vec\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --gpu-layers=-1 --parallel 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --no-disable-thinking --max-tokens 128 --no-reuse --no-v2-prime-pass --server-extra "--spec-type none" --write-diagnostics
```

## Result

| Run | Aggregate TPS | Decode eval | Prompt eval |
| --- | ---: | ---: | ---: |
| clean | `28.1374` | `30.08 tok/s` | `578.25 tok/s` |
| `GGML_CUDA_DISABLE_FUSION=1` | `26.6406` | `28.61 tok/s` | `509.97 tok/s` |

Delta:

- aggregate: `-5.32%`;
- decode eval: `-4.89%`;
- prompt eval in this short gate also regressed.

## Decision

- Reject disabling ROCm fusion.
- Keep H39 focused on optimizing the existing fused Q3_K MMVQ path rather than removing it.
- The next code audit should inspect `ggml/src/ggml-cuda/mmvq.cu` around:
  - `calc_nwarps(...)`: RDNA4 Q3_K `ncols_dst=1` currently returns `1` warp;
  - `should_use_small_k(...)`: Qwen-hot RDNA4 types force `small_k=true`, but Q3_K still has `nwarps=1`, so `rows_per_block` stays `1`;
  - `mul_mat_vec_q<..., has_fusion=true>`: fused FFN gate/up computes two Q3_K vec-dots per row and applies GLU in the same kernel.

## Artifacts

- `build_logs/agent-workload/e150-rocm-decode-q4-clean-r1.diagnostics.md`
- `build_logs/agent-workload/e150-rocm-decode-q4-disablefusion-r1.diagnostics.md`
- `build_logs/agent-workload/e150-rocm-decode-q4-clean-r1.server.log`
- `build_logs/agent-workload/e150-rocm-decode-q4-disablefusion-r1.server.log`
