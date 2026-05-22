# E149 ROCm Decode Parity Audit

## Metadata

- Experiment ID: E149
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master worktree, no runtime code changes
- Target lane: Qwen3.6-27B-Q3_K_S decode-focused ROCm vs Vulkan, RX 9070 XT / `gfx1201`, driver `32.0.31007.5012`

## Hypothesis

- Statement: ROCm decode can close part of the current gap to Vulkan, but the next useful work must target the measured decode route families, not generic "Vulkan has fusion" claims.
- Mechanism: E116 shows Vulkan q4/f16 decode around `40-41 tok/s`, while ROCm q4 is around `29.6 tok/s`. The gap is large enough that single low-share fusions cannot close it; the likely high-ceiling routes are Q3_K direct decode/MMVQ-MMQ and, for 64k, FlashAttention/KV.
- Why now: the user switched focus from Vulkan 64k prefill to catching ROCm decode up to Vulkan.

## Audit Findings

The provided `VULKAN_ROCM_DECODE_ANALYSIS.md` is useful, but several claims need correction before implementation:

- ROCm already has graph-level fusion: `RMS_NORM+MUL`, `RMS_NORM+MUL+ADD`, `ROPE+VIEW+SET_ROWS`, `UNARY+MUL`, SSM fusion, FFN `MUL_MAT_VEC` fusion, and HIP/CUDA graph capture.
- The missing Vulkan-only route is narrower: `RMS_NORM+MUL+ROPE(_VIEW_SET_ROWS)`. It is not yet proven active for the local Qwen M-RoPE route; existing Vulkan perf logs show `RMS_NORM_MUL`, not `RMS_NORM_MUL_ROPE`.
- The source-size argument is overstated. Current local counts are about `15.1k` lines for `ggml-vulkan.cpp`, `15.8k` lines of Vulkan shader source, and `33.2k` lines across `ggml-cuda/*.cu/*.cuh`.
- Launch/pipeline overhead is not proven as the root cause because ROCm has CUDA/HIP graph capture and update logic. A fresh decode trace must record whether capture is active for the current lane.

## Math / Theory

Known decode-focused measurements:

| Route | Aggregate TPS | Decode eval |
| --- | ---: | ---: |
| ROCm q4, E116 r1 | `29.1685` | `29.625 tok/s` |
| Vulkan q4, E116 r3 | `39.8801` | `40.8683 tok/s` |
| Vulkan f16, E116 r3 | `40.2753` | `41.2283 tok/s` |

Using Vulkan q4 as the first parity target, ROCm needs about `1.38x` decode-eval speedup.

Old ROCm C01 decode trace, used only as a ceiling proxy until a fresh post-driver trace exists:

| Route group | Sum | Share of parsed CUDA-node time |
| --- | ---: | ---: |
| `MUL_MAT forward+fused` | `2044.258 ms` | `61.9%` |
| `RMS_NORM+ROPE+SET_ROWS` | `277.737 ms` | `8.4%` |
| parsed CUDA-node total | `3303.800 ms` | `100%` |

Implications:

- If only matmul/direct-Q3 work moves, it needs about `1.80x` local speedup to reach Vulkan q4 decode.
- If matmul plus norm/rope/set_rows move together, they still need about `1.64x` local speedup.
- A standalone `RMS_NORM+MUL+ROPE` port cannot close the gap unless a fresh trace shows much larger rope/norm share than the old C01 data.

Fresh E149 post-driver diagnostic trace (`triage_diff`, `max_tokens=128`,
trace env enabled; use route shares, not wall TPS, because tracing adds overhead):

| Route group | Sum | Share of parsed CUDA-node time |
| --- | ---: | ---: |
| `MUL_MAT forward` | `276.50 ms` | `77.84%` |
| `FLASH_ATTN_EXT forward` | `18.77 ms` | `5.29%` |
| `ADD forward` | `16.54 ms` | `4.66%` |
| `RMS_NORM fused` | `4.47 ms` | `1.26%` |
| `ROPE forward` | `4.24 ms` | `1.19%` |
| `SET_ROWS forward` | `2.89 ms` | `0.81%` |

Trace caveat: the ROCm node logger records graph update/capture paths, not every
subsequent graph replay equally. It is reliable for route structure and first
ceiling, but clean server timing remains authoritative for speed.

When the parsed rows are split into graph sections and the initial section is
excluded, the route remains matmul-led but FA becomes visible:

| Route group | Share of parsed post-initial sections |
| --- | ---: |
| `MUL_MAT forward+fused` | `68.57%` |
| `FLASH_ATTN_EXT forward` | `17.43%` |
| `RMS_NORM+ROPE+SET_ROWS` | `3.27%` |

Sync companion trace (`e149-rocm-decode-q4-synctrace-mt16-r1`,
`GGML_CUDA_DISABLE_GRAPHS=1`, `GGML_TRACE_CUDA_NODE_TIMING_SYNC=1`,
`max_tokens=16`) intentionally slows the server (`5.1090 TPS`,
`6.15 tok/s decode eval`) and is diagnostic only. With the first graph section
excluded, it gives a less enqueue-biased per-node view:

| ROCm sync route group | Sum | Share of parsed post-initial sections |
| --- | ---: | ---: |
| `MUL_MAT forward` | `988.83 ms` | `34.45%` |
| `MUL_MAT fused` | `555.60 ms` | `19.35%` |
| `RMS_NORM fused` | `259.68 ms` | `9.05%` |
| `GET_ROWS` | `131.16 ms` | `4.57%` |
| `UNARY fused` | `130.22 ms` | `4.54%` |
| `ADD` | `123.68 ms` | `4.31%` |
| `CPY` | `121.12 ms` | `4.22%` |
| `L2_NORM` | `114.02 ms` | `3.97%` |
| `GATED_DELTA_NET` | `104.28 ms` | `3.63%` |
| `FLASH_ATTN_EXT` | `41.98 ms` | `1.46%` |
| `ROPE` | `40.57 ms` | `1.41%` |
| `SET_ROWS` | `39.33 ms` | `1.37%` |

Top joined Q3_K route/shape rows in that sync trace:

| ROCm route / shape | Sum | Count | Avg |
| --- | ---: | ---: | ---: |
| `mul_mat_vec_q_direct`, `src0_ne=(5120,10240)`, `dst_ne=(10240,1)` | `120.77 ms` | `720` | `0.17 ms` |
| `mul_mat_q_direct`, `src0_ne=(5120,17408)`, `dst_ne=(17408,159)` | `118.41 ms` | `126` | `0.94 ms` |
| `mul_mat_vec_q_direct`, `src0_ne=(5120,6144)`, `dst_ne=(6144,1)` | `97.72 ms` | `720` | `0.14 ms` |
| `mul_mat_q_direct`, `src0_ne=(17408,5120)`, `dst_ne=(5120,159)` | `62.42 ms` | `63` | `0.99 ms` |
| `mul_mat_vec_q_direct`, `src0_ne=(5120,1024)`, `dst_ne=(1024,1)` | `50.29 ms` | `480` | `0.10 ms` |
| `mul_mat_vec_q_direct`, `src0_ne=(5120,12288)`, `dst_ne=(12288,1)` | `44.41 ms` | `240` | `0.19 ms` |
| `mul_mat_vec_q_direct`, `src0_ne=(5120,17408)`, `dst_ne=(17408,2)` | `35.28 ms` | `126` | `0.28 ms` |
| `mul_mat_q_direct`, `src0_ne=(5120,10240)`, `dst_ne=(10240,159)` | `35.09 ms` | `48` | `0.73 ms` |

This sync companion moderates the non-sync `77.84%` matmul number, but it does
not change the decision. `MUL_MAT forward+fused` is still `53.80%` of parsed
post-initial sync time, while `RMS_NORM+ROPE+SET_ROWS` is `11.83%`. A standalone
norm/rope fusion therefore has a measurable cleanup ceiling, but it cannot close
the `1.38x` Vulkan decode gap by itself; the first code branch still needs to
move the Q3_K direct matmul families.

Reusable shape-delta tooling was added as
`scripts/research/rocm_vulkan_decode_route_delta.py`. The first decode-only Q3_K
comparison used ROCm sync sections after the initial two sections and Vulkan
decode sections with `Total time` between `20000` and `40000 us`:

```powershell
python scripts\research\rocm_vulkan_decode_route_delta.py --rocm-log build_logs\agent-workload\e149-rocm-decode-q4-synctrace-mt16-r1.server.log --vulkan-log build_logs\agent-workload\e149-vulkan-decode-q4-perflog-r1.server.log --rocm-skip-sections 2 --vulkan-min-section-us 20000 --vulkan-max-section-us 40000 --qtype q3_K --top 20
```

Decode-only Q3_K bucket split:

| Backend bucket | Share of parsed Q3_K matmul time |
| --- | ---: |
| ROCm `mul_mat_vec_q_fused q3_K->f32` | `63.78%` |
| ROCm `mul_mat_vec_q_direct q3_K->f32` | `36.22%` |
| Vulkan `MUL_MAT_VEC q3_K` | `72.33%` |
| Vulkan `MUL_MAT_ADD_VEC q3_K` | `27.67%` |

Top normalized Q3_K shapes:

| QType / shape | ROCm share | Vulkan share |
| --- | ---: | ---: |
| `q3_K m=17408 n=1 k=5120` | `36.07%` | `48.32%` |
| `q3_K m=5120 n=1 k=17408` | `24.33%` | `25.47%` |
| `q3_K m=10240 n=1 k=5120` | `13.88%` | `10.99%` |
| `q3_K m=6144 n=1 k=5120` | `11.23%` | `7.32%` |
| `q3_K m=1024 n=1 k=5120` | `5.78%` | `1.25%` |
| `q3_K m=12288 n=1 k=5120` | `5.10%` | `4.30%` |
| `q3_K m=5120 n=1 k=6144` | `3.61%` | `2.34%` |

This is the strongest current planning signal: the first ROCm code audit should
start inside the fused MMVQ decode path for FFN gate/up/down shapes, then compare
the non-fused direct shapes. Vulkan's `MUL_MAT_ADD_VEC` advantage is shape-aligned
with the ROCm fused bucket, so "fusion exists" is true, but "the fused Q3_K decode
kernel is competitive enough" is still unproven.

Fresh route counts from the non-sync route trace:

| Route | Count |
| --- | ---: |
| `mul_mat_vec_q_direct, q3_K` | `929` |
| `mul_mat_vec_f_direct, f32` | `560` |
| `cublas_backend, f32` | `400` |
| `mul_mat_q_direct, q3_K` | `349` |
| `mul_mat_vec_q_direct, q4_K` | `240` |

This strengthens the audit conclusion: the short-decode parity target is
matmul/direct-Q3 dominated. `RMS_NORM+ROPE+SET_ROWS` is only about `3.26%` of
the parsed non-sync trace and `11.83%` of the sync companion, so it remains a
secondary cleanup candidate rather than the first parity branch.

Vulkan q4 perf comparator (`GGML_VK_PERF_LOGGER=1`, same short decode gate)
also slows wall timing, but its per-token sections show the target route:

| Vulkan decode route group | Share of parsed decode sections |
| --- | ---: |
| Q3_K `MUL_MAT_VEC` | `50.67%` |
| Q3_K `MUL_MAT_ADD_VEC` | `19.38%` |
| Q6_K `MUL_MAT_VEC` | `5.99%` |
| Q4_K `MUL_MAT_VEC` | `5.57%` |
| `RMS_NORM_MUL` | `3.79%` |
| `FLASH_ATTN_EXT` | `1.11%` |
| `ROPE+SET_ROWS` | `0.60%` |

This says the transferable advantage to study is not broad Vulkan fusion. It is
the Q3_K/QK direct decode kernel family, plus a secondary check on why ROCm FA
shows up more strongly in parsed post-initial sections.

Graph-disable diagnostic, sequential rerun:

| Run | Aggregate TPS | Decode eval |
| --- | ---: | ---: |
| clean sequential | `27.1129` | `29.15 tok/s` |
| `GGML_CUDA_DISABLE_GRAPHS=1` sequential | `27.2063` | `29.28 tok/s` |

This is an r1 diagnostic only, but it does not support pipeline/graph launch
overhead as the first H39 lever on this short-decode lane.

Protocol sanity:

```powershell
python scripts\research\formula_sanity_checks.py
python scripts\research\required_acceptance.py --target-wall 1.3797 --draft-len 4 --prefill-share 0.0 --prefill-speedup 1.0 --decode-kernel-speedup 1.0 --spec-overhead 0.0
```

The no-overhead speculative control says draft length 4 would need `0.1266` full-coverage acceptance to reach the same wall target, but E116 showed `ngram-mod 12/16/32` regressed in decode-only mode. This audit is therefore a backend-route plan, not a speculative plan.

## Implementation Plan

1. Evidence refresh, no code:
   - Stop/avoid background `llama-server`.
   - Re-run ROCm q4 and Vulkan q4 decode-focused gates with the same E116 lane, `runs=3` for the final baseline pair.
   - Collect ROCm trace with `GGML_TRACE_CUDA_NODE_TIMING=1`, `GGML_TRACE_CUDA_MUL_MAT_ROUTE=1`, `GGML_TRACE_MMVQ_SMALL_K=1`, and, for focused runs, `GGML_TRACE_MMVQ_TIMING=1`.
   - Collect Vulkan perf route with `GGML_VK_PERF_LOGGER=1` and no experimental corrupt tile knobs.
   - Record whether ROCm graph capture is active or disabled.
2. Build a route-delta table:
   - Compare ROCm vs Vulkan by op family, tensor shape, quant type, and phase.
   - Split short decode (`ctx=12288`, short prompt, long generation) from long-context decode (`ctx=65536`) because the latter can become FA/KV dominated.
3. Candidate A, highest ceiling: ROCm Q3_K direct decode/MMVQ-MMQ specialization.
   - Start with the shape-delta target: `mul_mat_vec_q_fused` Q3_K `m=17408,n=1,k=5120` and `m=5120,n=1,k=17408`, then non-fused direct shapes `m=10240/6144/12288,n=1,k=5120`.
   - Revalidate the current post-driver Q3_K `ncols_dst=1` policy before citing historical E013 `nwarps=2`.
   - If compile pressure blocks MMVQ edits, split or isolate the Q3_K/Q4_K/Q6_K MMVQ translation-unit surface before changing math.
   - Only test changes that move observed Q3_K buckets; avoid broad selector toggles already rejected in H35/C01.
4. Candidate B, secondary: missing ROCm fused `RMS_NORM+MUL+ROPE(+VIEW+SET_ROWS)`.
   - Proceed only if the fresh trace proves the exact pattern is active and rope/set_rows share is large enough for at least a `>=2%` wall ceiling.
   - Treat this as a launch/memory cleanup component, not the main parity route.
5. Candidate C, separate long-context branch: ROCm 64k decode FA/KV route.
   - Compare ROCm 64k decode trace against Vulkan E128 (`36.58 tok/s`) and ROCm E128 (`22.83 tok/s`).
   - If FA dominates, do not apply short-decode MMVQ conclusions to 64k without a separate trace.

## Benchmark Plan

Baseline ROCm q4 command family, after ensuring no background server:

```powershell
python scripts\agent_workload_bench.py --label e149-rocm-decode-q4-r3 --out-dir build_logs/agent-workload --tasks quick --task-ids triage_diff,review_bug --runs 3 --server-bin build-rocm-vec\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --gpu-layers=-1 --parallel 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --no-disable-thinking --max-tokens 512 --no-reuse --no-v2-prime-pass --server-extra "--spec-type none" --write-diagnostics
```

Vulkan comparator:

```powershell
$env:GGML_VK_ALLOW_GRAPHICS_QUEUE = "1"
python scripts\agent_workload_bench.py --label e149-vulkan-decode-q4-r3 --out-dir build_logs/agent-workload --tasks quick --task-ids triage_diff,review_bug --runs 3 --server-bin build-vulkan\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --gpu-layers=-1 --parallel 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --no-disable-thinking --max-tokens 512 --no-reuse --no-v2-prime-pass --server-extra "--spec-type none" --write-diagnostics
```

## Result

- Outcome: open H39 and proceed to Q3_K direct decode/MMVQ-MMQ investigation; no runtime code change.
- Delta: no new speed claim. E116 remains the clean decode baseline; E149 trace and graph-disable runs are diagnostics.
- Confidence: high that the original document over-attributes the gap to missing fusion; high that short-decode is Q3_K matvec dominated in both ROCm and Vulkan diagnostics; medium on exact cross-backend shares because trace instrumentation differs.
- Recommendation: do not start with standalone `RMS_NORM+MUL+ROPE` fusion. First inspect and model the ROCm fused Q3_K MMVQ decode path for the FFN gate/up/down shapes, then design a Q3_K-specific MMVQ branch if the code/resource audit gives a plausible local ceiling. Keep ROCm FA as a secondary decode check because it appears larger than Vulkan FA in parsed diagnostic sections.

## Notes

- Invalid controls: `e149-rocm-decode-q4-clean-r1` and
  `e149-rocm-decode-q4-disablegraphs-r1` were accidentally launched in
  parallel and both collapsed to about `2.25 TPS`; do not use those numbers.
- Valid diagnostic artifacts:
  - `build_logs/agent-workload/e149-rocm-decode-q4-trace-r1.server.log`
  - `build_logs/agent-workload/e149-rocm-decode-q4-trace-r1.diagnostics.md`
  - `build_logs/agent-workload/e149-rocm-decode-q4-synctrace-mt16-r1.server.log`
  - `build_logs/agent-workload/e149-rocm-decode-q4-synctrace-mt16-r1.diagnostics.md`
  - `build_logs/agent-workload/e149-rocm-decode-q4-cleanseq-r1.diagnostics.md`
  - `build_logs/agent-workload/e149-rocm-decode-q4-disablegraphsseq-r1.diagnostics.md`
  - `build_logs/agent-workload/e149-vulkan-decode-q4-perflog-r1.server.log`
  - `build_logs/agent-workload/e149-vulkan-decode-q4-perflog-r1.diagnostics.md`
  - `build_logs/agent-workload/e149-rocm-vulkan-decode-route-delta-q3k.md`
- Future breakthrough candidates must include a real `llama-server` output sanity check before promotion, following E118.
