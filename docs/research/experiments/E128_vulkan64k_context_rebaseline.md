# E128 Vulkan 64k Context Rebaseline

## Metadata

- Experiment ID: E128
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master, local working tree after driver update
- Hypothesis ID: H38
- Target lane: Qwen3.6-27B-Q3_K_S, real `llama-server` repo-snapshot request, `ctx=65536`, q4/q4 KV, FlashAttention on, full offload, `spec=none`, thinking on, no reuse, `max_tokens=120`

## Hypothesis

- Statement: Vulkan's post-driver decode advantage may not transfer to full 64k contexts because long-context prefill can dominate wall time.
- Mechanism: at about 57k prompt tokens, wall time is mostly large Q3_K matmul plus q4 FlashAttention over a long KV range. Decode is still fast, but it is too small a share to offset weaker prompt eval.
- Why now: user reported 64k context feeling much slower and asked to focus on Vulkan because it is the practical fast backend for decode/session routes.

## Benchmark Contract

- Real server only: `scripts\repo_snapshot_context_bench.py`, not synthetic token-only microbench.
- Prompt calibration:
  - old large prompt overflowed `ctx=65536` at `466999` prompt tokens.
  - `base-char-budget 50000` still overflowed at `74200` prompt tokens.
  - final lane used `base-char-budget 38000`, `152000` prompt chars, `57409` prompt tokens.
- Cold/full-context flags:
  - `--cache-ram 0 --ctx-checkpoints 0`
  - no v2 priming
  - `--cache-type-k q4_0 --cache-type-v q4_0`
  - q4 V cache requires FlashAttention, so `--flash-attn off` is an invalid negative control.
- Main binaries:
  - Vulkan: `build-vulkan\bin\llama-server.exe`
  - ROCm comparison: `build-rocm-vec\bin\llama-server.exe`

## Metrics

| Route | b/ub | Wall TPS | Prompt Eval TPS | Decode Eval TPS | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Vulkan q4 baseline | 2048/512 | `1.2896` | `640.63` | `36.04` | baseline |
| ROCm q4 comparison | 2048/512 | `1.5545` | `799.09` | `22.83` | confirms Vulkan 64k wall loss is prefill-side |
| Vulkan q4 shape | 4096/1024 | `1.3106` | `651.59` | `35.86` | small keepable shape improvement |
| Vulkan q4 graphics queue + `--no-mmap` | 4096/1024 | `1.3375` | `665.00` | `36.54` | keep as route/profile stack |
| Vulkan q4 graphics queue + `--no-mmap` confirm | 8192/1024 | `1.3406` | `666.62` | `36.58` | best safe no-code route |
| Vulkan q4 graphics queue + `--no-mmap` | 8192/2048 | `1.3088` | `650.22` | `36.58` | reject larger ubatch |
| Vulkan q8/q8 KV | 4096/1024 | `0.2008` | `96.62` | `35.53` | reject; q8 KV destroys prefill/residency |
| Vulkan q4 FA scalar-only probe | 8192/1024 | `1.0526` | `520.12` | `34.19` | reject and revert code probe |

Best safe Vulkan stack:

```powershell
$env:GGML_VK_ALLOW_GRAPHICS_QUEUE = "1"
python scripts\repo_snapshot_context_bench.py --label-prefix e128-vulkan64k-c152k-b8192-ub1024-q4-graphicsq-nommap-confirm-none-noreuse --server-bin build-vulkan\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --contexts 65536 --base-char-budget 38000 --scale-context-chars --max-tokens 120 --gpu-layers 999 --batch-size 8192 --ubatch-size 1024 --cache-type-k q4_0 --cache-type-v q4_0 --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0 --no-mmap"
```

Delta:

- Best Vulkan vs first Vulkan 64k baseline: `1.2896 -> 1.3406 TPS`, `+3.95%`; prompt eval `640.63 -> 666.62`, `+4.06%`.
- Best Vulkan vs ROCm 64k comparison: `1.3406` vs `1.5545 TPS`, about `-13.8%`; prompt eval `666.62` vs `799.09`, about `-16.6%`.
- Decode still favors Vulkan strongly: best Vulkan `36.58 tok/s` vs ROCm `22.83 tok/s`.

## Route Trace

Vulkan route trace on `b=4096,ub=1024,q4` confirms the active long-prefill kernels:

- Q3_K matmul: `matmul_q3_k_f32_f16acc_aligned_l`
- Q4_K matmul: `matmul_q4_k_f32_f16acc_aligned_l`
- Q8 route is not active: `q8_candidate_empty=1`, `q8_mmp_found=0`
- q4 KV stays on `FLASH_ATTN_EXT`; q4 V cache cannot run with FlashAttention disabled.

Perf logger aggregate from the `max_tokens=1` trace:

| Op | Total ms | Share |
| --- | ---: | ---: |
| `MUL_MAT q3_K` | `42684.45` | `47.79%` |
| `FLASH_ATTN_EXT` | `33965.16` | `38.03%` |
| `MUL_MAT q4_K` | `2824.24` | `3.16%` |
| `CONCAT` | `2272.02` | `2.54%` |
| `GATED_DELTA_NET` | `1841.17` | `2.06%` |
| `GLU` | `1271.25` | `1.42%` |
| `RMS_NORM_MUL` | `1158.70` | `1.30%` |

Interpretation: this is not a decode/speculative bottleneck. On the 64k cold lane, roughly `85.8%` of traced Vulkan time is only two families: Q3_K matmul and q4 FlashAttention.

## Negative Controls

- `--flash-attn off`: invalid with q4 V cache; server init fails with `V cache quantization requires flash_attn`.
- q8/q8 KV: huge prefill regression and not enough practical VRAM headroom for the 64k lane.
- `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1`, `GGML_VK_DISABLE_FUSION=1`, `GGML_VK_DISABLE_ASYNC=1`: noise or small regressions.
- `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`, `GGML_VK_ENABLE_MEMORY_PRIORITY=1`: small signals only; did not beat graphics queue + `--no-mmap` stack.
- FA scalar-only code probe: targeted the right family but regressed `-21.48%`; reverted.

## Result

- Outcome: keep the no-code Vulkan 64k profile as an opt-in improvement, but do not claim Vulkan has closed the 64k full-context gap to ROCm.
- Practical current Vulkan 64k profile:
  - q4/q4 KV, FlashAttention on
  - `b=8192`, `ub=1024`
  - `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`
  - `--no-mmap`
  - `--spec-type none`
  - no reuse for cold-first measurements
- Next useful code work:
  - H31/H38 Q3_K large-prefill kernel work, only after prebuild/static route gates.
  - H05/H38 q4 FlashAttention long-KV tuning, because FA becomes a first-class 64k bottleneck.
  - Avoid speculative/ngram work for this lane until prefill is no longer dominant.

## Artifacts

- `build_logs/agent-workload/e128-vulkan64k-c152k-b2048-ub512-q4-none-noreuse-repo-summary.md`
- `build_logs/agent-workload/e128-rocm64k-c152k-b2048-ub512-q4-none-noreuse-repo-summary.md`
- `build_logs/agent-workload/e128-vulkan64k-c152k-b4096-ub1024-q4-none-noreuse-repo-summary.md`
- `build_logs/agent-workload/e128-vulkan64k-c152k-b4096-ub1024-q4-graphicsq-nommap-none-noreuse-repo-summary.md`
- `build_logs/agent-workload/e128-vulkan64k-c152k-b8192-ub1024-q4-graphicsq-nommap-confirm-none-noreuse-repo-summary.md`
- `build_logs/agent-workload/e128-vulkan64k-c152k-b4096-ub1024-q4-perf1-ctx64k.server.log`
- `build_logs/agent-workload/e128-vulkan64k-c152k-b4096-ub1024-q4-trace8-ctx64k.server.log`
- `build_logs/agent-workload/e128-vulkan64k-c152k-b4096-ub1024-q8-none-noreuse-repo-summary.md`
- `build_logs/agent-workload/e128-vulkan64k-c152k-b8192-ub1024-q4-graphicsq-nommap-fascalar-none-noreuse-repo-summary.md`
