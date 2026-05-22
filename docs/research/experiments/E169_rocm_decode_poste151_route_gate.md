# E169 ROCm Decode Post-E151 Route Gate

## Metadata

- Experiment ID: E169
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E168 rejection
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: After E151 and the rejected E162-E168 MMVQ probes, the next useful ROCm decode work should be selected from a fresh route-share gate rather than more local Q3_K vec-dot edits.
- Mechanism: E149/E152 showed Q3_K matvec dominance, but E151 reduced the worst Q3_K policy issue. A new sync trace can show whether secondary fusion routes such as RMS/ROPE/SET_ROWS now have enough share to justify a complex fused kernel, or whether Q3_K still dominates so strongly that only a new Q3_K topology can matter.
- Risk: Trace-heavy sync runs are not clean speed baselines. Use them only for relative route share and next-candidate selection.

## Benchmark Plan

- Run active H39 with `--max-tokens 16`, graph disabled, CUDA node timing sync, route logs, and MMVQ timing/resources disabled unless needed.
- Parse op/kind timing shares from `GGML_TRACE_CUDA_NODE_TIMING`.
- Use this as a planning gate, not a speed claim.

## Result

Decode-only parse after `prompt processing done`:

| Route group | Count | Time | Share |
| --- | ---: | ---: | ---: |
| `MUL_MAT` all | `8015` | `1593.711 ms` | `54.20%` |
| `MUL_MAT` `q3_K` fused | `2147` | `593.299 ms` | `20.18%` |
| `MUL_MAT` `q3_K` forward/direct | `2524` | `578.345 ms` | `19.67%` |
| `MUL_MAT` `f32` | `2560` | `278.180 ms` | `9.46%` |
| `MUL_MAT` `q4_K` | `768` | `115.325 ms` | `3.92%` |
| `RMS_NORM` fused | `3344` | `263.444 ms` | `8.96%` |
| `ROPE` forward | `512` | `40.101 ms` | `1.36%` |
| `SET_ROWS` forward | `512` | `39.773 ms` | `1.35%` |
| `FLASH_ATTN_EXT` forward | `256` | `45.086 ms` | `1.53%` |

`RMS_NORM + ROPE + SET_ROWS` is about `11.67%` of traced decode time. Even a large `30%` local win there projects to only about `3.6%` wall-speed gain, useful but not enough to close the remaining Vulkan gap. Q3_K remains the primary parity target: fused + direct Q3_K is `39.84%` of traced decode time.

## Decision

- Keep as a planning gate.
- Do not spend the next main branch on launch-only or standalone RMS/ROPE cleanup unless it is stacked with Q3_K work or has a very cheap prototype.
- Next high-ceiling ROCm decode work needs a new Q3_K topology that covers both fused FFN and direct attention/QKV/down buckets. The rejected E162-E168 probes show that local register/occupancy tweaks around the existing MMVQ dot loop are no longer enough.

## Artifacts

- `build_logs/agent-workload/e169-rocm-decode-q4-poste151-opshare-r1.server.log`
- `build_logs/agent-workload/e169-rocm-decode-q4-poste151-opshare-r1.diagnostics.md`
