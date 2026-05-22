# E151 ROCm RDNA4 Q3_K MMVQ nwarps=2 Decode Gate

## Metadata

- Experiment ID: E151
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master worktree after `1f05aebcb`, candidate in `ggml/src/ggml-cuda/mmvq.cu`
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: RDNA4 `Q3_K` decode should use `nwarps=2` for `mul_mat_vec_q` when `ncols_dst=1`, rather than the current implicit `nwarps=1`.
- Mechanism: E149 showed ROCm short decode is dominated by Q3_K MMVQ, especially the fused FFN gate/up route. E150 showed disabling fusion regresses, so the target is the quality of the fused/direct Q3_K MMVQ kernel. Current code forced Qwen-hot `small_k=true`, but because `calc_nwarps(...)` returned `1` for RDNA4 `Q3_K`, `calc_rows_per_block(...)` still stayed at `1`; returning `2` lets the small-k branch process two rows per block with two warps.
- Prior signal: historical E013 kept a Q3_K `nwarps=2` decode policy and rejected `nwarps=4`; the current tree had drifted away from that exact Q3_K branch.
- Risk: This is a local RDNA4/Qwen/Q3_K policy. It should not be generalized to IQ types or non-RDNA4 devices without separate A/B.

## Change

In `ggml/src/ggml-cuda/mmvq.cu`, under `MMVQ_PARAMETERS_RDNA4` and `ncols_dst == 1`, return `2` for `GGML_TYPE_Q3_K`.

Other RDNA4 simple vec-dot types keep the existing `nwarps=8` branch. IQ2/IQ3-style complex vec-dot types remain outside the higher-warp whitelist.

## Method

Before the run:

- confirmed no background `llama-server`;
- cleared `HSA_OVERRIDE_GFX_VERSION`;
- cleared fusion/graph/trace override variables;
- rebuilt `build-rocm-vec` with `cmake --build build-rocm-vec --config Release -j`;
- used `--runs 1` for the first gate and `--runs 3` for confirmation;
- kept `--no-disable-thinking`, `--no-reuse`, `--no-v2-prime-pass`, and `--spec-type none`.

Candidate command shape:

```powershell
python scripts\agent_workload_bench.py --label e151-rocm-decode-q4-q3warps2-r3 --out-dir build_logs\agent-workload --tasks quick --task-ids triage_diff --runs 3 --server-bin build-rocm-vec\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --gpu-layers=-1 --parallel 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --no-disable-thinking --max-tokens 128 --no-reuse --no-v2-prime-pass --server-extra "--spec-type none" --write-diagnostics
```

Clean post-rebuild control:

```powershell
python scripts\agent_workload_bench.py --label e151-rocm-decode-q4-cleanpost-r3 --out-dir build_logs\agent-workload --tasks quick --task-ids triage_diff --runs 3 --server-bin build-rocm-vec\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --gpu-layers=-1 --parallel 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --no-disable-thinking --max-tokens 128 --no-reuse --no-v2-prime-pass --server-extra "--spec-type none" --write-diagnostics
```

## Result

| Run | Aggregate TPS | Decode eval | Prompt eval |
| --- | ---: | ---: | ---: |
| clean post-rebuild r3 | `28.1123` | `29.77 tok/s` | `711.73 tok/s` |
| Q3_K `nwarps=2` r1 | `29.1618` | `31.52 tok/s` | `513.66 tok/s` |
| Q3_K `nwarps=2` r3 | `30.3145` | `32.2467 tok/s` | `713.8533 tok/s` |

Confirmed delta versus the same-session clean post-rebuild r3:

- aggregate completion TPS: `+7.83%`;
- decode eval: `+8.32%`;
- prompt eval is essentially unchanged in the r3 comparison, which matches the intended short-decode target.

The remaining Vulkan q4 decode comparator from E116 is still around
`40.8683 tok/s`; after E151, the ROCm short-decode gap falls from about
`1.37x` to about `1.27x`.

## Real Server Sanity

The promoted candidate was checked through the real `llama-server` route, not a synthetic kernel-only bench:

```powershell
python scripts\agent_workload_bench.py --label e151-rocm-decode-q4-q3warps2-live-sanity-r1 --out-dir build_logs\agent-workload --tasks quick --task-ids triage_diff --runs 1 --server-bin build-rocm-vec\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --gpu-layers=-1 --parallel 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --no-disable-thinking --max-tokens 64 --no-reuse --no-v2-prime-pass --server-extra "--spec-type none" --write-diagnostics
```

Sanity result:

- errors: `0`;
- decode eval: `30.55 tok/s` on the short sanity run;
- response preview starts with normal `Thinking Process:` triage text;
- no repeated-symbol or `wm32-wn32`-style corruption observed.

This live sanity run is recorded as correctness evidence, not as the speed claim.

## Decision

- Keep the RDNA4 `Q3_K/ncols_dst=1` `nwarps=2` policy.
- This is the first H39 code win and closes part of the ROCm/Vulkan decode gap, but it does not finish the parity target.
- Next H39 work should collect a post-E151 route/timing trace and inspect the residual split between fused FFN Q3_K MMVQ, direct Q3_K MMVQ, FlashAttention, and norm/rope. Larger changes should target a whole Q3_K decode route branch, not a single nearby selector toggle.

## Artifacts

- `build_logs/agent-workload/e151-rocm-decode-q4-cleanpost-r3.diagnostics.md`
- `build_logs/agent-workload/e151-rocm-decode-q4-q3warps2-r1.diagnostics.md`
- `build_logs/agent-workload/e151-rocm-decode-q4-q3warps2-r3.diagnostics.md`
- `build_logs/agent-workload/e151-rocm-decode-q4-q3warps2-live-sanity-r1.diagnostics.md`
- `build_logs/agent-workload/e151-rocm-decode-q4-cleanpost-r3.jsonl`
- `build_logs/agent-workload/e151-rocm-decode-q4-q3warps2-r3.jsonl`
- `build_logs/agent-workload/e151-rocm-decode-q4-q3warps2-live-sanity-r1.jsonl`
