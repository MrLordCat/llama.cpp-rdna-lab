# E148 Vulkan 64k FA Analytic Causal Mask Route

## Metadata

- Experiment ID: E148
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E147 (`c4f60ecbb`)
- Hypothesis ID: H38 / H05
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan `ctx=65536`, q4/q4 KV, FlashAttention on, no reuse

## Hypothesis

- Statement: the active Vulkan q4/q4 FlashAttention route may improve if full 1024-token prefill chunks use an analytic causal mask instead of the current mask-opt prepass.
- Mechanism: E131 proved mask-opt itself is useful, but the current route computes mask-opt through a separate `fa_mask_opt` dispatch plus `ggml_vk_sync_buffers()` before every FA node. For Qwen text prefill with one causal sequence, no SWA, and no ALiBi, the mask tile class is derivable from `N`, `KV`, `Br`, `Bc`, and the row tile. A shader branch can keep all-zero/all--inf/mixed behavior while removing the prepass and mask-buffer reads for full chunks.
- Why now: Br/Bc, f16acc, SHMEM staging, split-k, KV dtype, larger-Br, and D-split were all rejected. This is a route-body change that keeps the same coopmat1 q4/q4 FA route and changes how the long-KV mask path is represented.

## Math / Theory

- Assumptions:
  - apply only to main full chunks where `N=1024`, `KV` is a multiple of `1024`, `gqa_ratio=1`, no sinks, no ALiBI/max-bias, and no logit softcap;
  - keep the default mask-opt route for warmup, tail, multi-stream, SWA, arbitrary masks, and all default runs unless an env flag is set;
  - causal query base for eligible chunks is `KV - N`.
- Expected speedup corridor:
  - E128 parsed FA share is about `42.25%`;
  - eligible full chunks are expected to be almost all long-KV FA time except the small warmup/tail rows;
  - a `1.05x` local win on eligible FA projects about `1.37 TPS`, `1.10x` projects about `1.39 TPS`, and `1.20x` projects about `1.44 TPS`.
- Failure conditions:
  - if the extra causal branch increases shader VGPR/instruction pressure enough to offset prepass removal;
  - if pp7488 is not representative of full 64k because it has a short tail (`N=320,KV=7680`);
  - if the mask is not pure causal text despite Qwen's current `n_swa=0` and `f_max_alibi_bias=0`;
  - if removing the host prepass exposes an implicit synchronization/order dependency.

## Implementation Plan

1. Minimal code surface to change:
   - add a research script that counts eligible FA tiles and mask prepass work from the E128 perf log;
   - add a default-off Vulkan env branch `GGML_VK_FA_ANALYTIC_CAUSAL_MASK=1`;
   - add a cm1 shader branch that computes all-zero/all--inf/mixed causal mask classes analytically.
2. Guard rails:
   - default route unchanged when the env var is absent;
   - only enable the branch for the active full-chunk cm1 q4/q4 conditions above;
   - route trace must show the new flag and still report coopmat1 q4/q4.
3. Rollback path:
   - revert the host env branch and shader flag if pp/resource gate regresses or output sanity fails.

## Benchmark Plan

- Baseline command: q4/q4 pp7488, FlashAttention on, `b8192/ub1024`, `-mmp 0`, route trace and FA pipeline stats.
- Candidate command: same plus `GGML_VK_FA_ANALYTIC_CAUSAL_MASK=1`.
- Number of runs: `r=1` for first pp gate; full 64k max-token-1 only if the pp gate is positive.
- Artifacts path: `build_logs/agent-workload/e148-*`.

## Metrics

- prompt eval tok/s
- FA route trace (`flags`, analytic causal mask enabled, `Br/Bc`, path)
- driver pipeline stats (VGPR/SGPR/LDS/scratch)
- full 64k real-server prompt eval if candidate survives
- real server output sanity before any promotion

## Result

- Outcome: reject/revert runtime prototype; keep diagnostic gate script and artifacts.
- Delta:
  - analytic gate: `99.71%` of E128 parsed FA time is eligible full-chunk FA; mask-opt prepass work is about `1,634,304` workgroups and a `51072.00 MiB` fp16 mask-cell read proxy across the E128 trace;
  - first pp/resource gate with pipeline stats: baseline `946.63 tok/s`, candidate `951.78 tok/s`; candidate resource report `84 VGPR / 65 SGPR / 26112 B LDS / 0 scratch` for the analytic full-chunk pipeline, with the tail falling back to the normal `98 VGPR / 76 SGPR` route;
  - repeated no-stats pp gate: baseline `971.41 tok/s`, candidate `972.21 tok/s`, only `+0.08%`.
- Confidence: medium-high for rejecting this exact implementation as a promotion candidate; medium for the broader conclusion because pp7488 is shorter than the full 64k tail, but the same mechanism should have shown more than noise if mask prepass/sync were the dominant long-KV cost.
- Recommendation: do not keep the runtime env branch. Future FA work should target the in-shader K/V dequant + softmax/PV loop or a more structural KV traversal change; removing mask-opt prepass alone is too low-ceiling.

## Notes

- This is not a repeat of E131 mask-opt disable. The candidate keeps mask skip semantics and tries to remove the prepass/sync and arbitrary-mask loads only when the mask is analytically causal.
- The initial suspicion was reasonable because the route trace has `use_mask_opt=1` for all main chunks and host code calls `fa_mask_opt` plus `ggml_vk_sync_buffers()` per FA node. The measured near-tie shows that those costs are not the limiter at pp7488; the long-KV loop remains dominated by the main FA shader body.
- Workflow correction: for FA, a prepass-elimination idea needs either direct per-dispatch timing that isolates the prepass or a pp gate clearly above noise before spending a full 64k server run.

## Artifacts

- `build_logs/agent-workload/e148-vulkan-fa-causal-mask-gate.md`
- `build_logs/agent-workload/e148-vulkan-fa-causal-mask-speedup-model.md`
- `build_logs/agent-workload/e148-vulkan-fa-causal-mask-baseline-pp7488.log`
- `build_logs/agent-workload/e148-vulkan-fa-causal-mask-candidate-pp7488.log`
- `build_logs/agent-workload/e148-vulkan-fa-causal-mask-baseline2-pp7488.log`
- `build_logs/agent-workload/e148-vulkan-fa-causal-mask-candidate2-pp7488.log`
