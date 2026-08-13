# P002 Vulkan 130k Evidence Pack

Status: historical evidence pack; P002 closed and parked on 2026-08-13.
D005's retained runtime behavior is unchanged, but the D028 `2.4 TPS` research
target and this evidence queue are no longer active. Signed-nibble runtime,
D003 larger-ubatch recovery and D006 output-placement relief remain rejected.
D007 retains the non-adjacent Q3_K FFN graph-surface evidence.

## Lane

- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Contract: `ctx=131072`, `batch=512`, `q4_0/q4_0`, FlashAttention on, `--spec-type none --no-mmap`; Vulkan current best uses `ubatch=256`.
- Workload: `quick:triage_diff`, `real-context-chars=24576`, `max_tokens=1` for diagnostic traces.
- Baseline projection anchor: P002 Vulkan cold-first `1.7898 TPS` from `d005-vulkan-default-splitk-confirm3` (`max_tokens=16`, current default cold check). The earlier `1.6654 TPS` current check and `1.6249 TPS` 3-run confirmation remain comparison anchors, and the old E265 `ub128` anchor remains historical.

## Artifacts

- Evidence pack: `build_logs/agent-workload/vscode-vulkan130k-routepack-r1.vulkan-evidence.md`.
- Route/perf trace: `build_logs/agent-workload/vscode-vulkan130k-routepack-r1.server.log`.
- Q3 resource trace: `build_logs/agent-workload/vscode-vulkan130k-q3stats-r1.server.log`.
- Corrected route ceiling: `build_logs/agent-workload/d004-vulkan130k-route-ceiling-corrected.md`.
- FFN gate/up model: `build_logs/agent-workload/d004-vulkan130k-ffn-gateup-route-model.md`.
- D005 split-K route: `build_logs/agent-workload/d005-vulkan-ffndown-splitk3-route-ceiling.md`, `build_logs/agent-workload/d005-vulkan-default-splitk-confirm3.diagnostics.md`.
- D006 output/residency scout: `build_logs/agent-workload/d006-vulkan-130k-d005-default-powerplus-r1.diagnostics.md`, `build_logs/agent-workload/d006-vulkan-130k-no-output-ub512-powerplus-r1.diagnostics.md`, `build_logs/agent-workload/d006-vulkan-130k-output-host-gpudev-ub512-r1.diagnostics.md`.
- D007 FFN block route gate: `build_logs/agent-workload/d007-vulkan-130k-ffn-block-nextop-trace-r1.diagnostics.md`, `build_logs/agent-workload/d007-vulkan-130k-ffn-block-nextop-trace-r1.server.log`, `build_logs/agent-workload/d007-vulkan-130k-ffn-scanblock-trace-r1.diagnostics.md`, `build_logs/agent-workload/d007-vulkan-130k-ffn-scanblock-trace-r1.server.log`.
- SPIR-V inputs: `build-vulkan/ggml/src/ggml-vulkan/vulkan-shaders.spv/matmul_q3_k_f32_aligned_f16acc_cm1.spv`, `build-vulkan/ggml/src/ggml-vulkan/vulkan-shaders.spv/flash_attn_f32_f16_q4_0_f16acc_cm1.spv`.

## Findings

Parsed Vulkan perf time is dominated by Q3_K prompt matmul:

| Bucket | Parsed share | Total ms |
| --- | ---: | ---: |
| `MUL_MAT q3_K` | 79.41% | 7941.70 |
| `FLASH_ATTN_EXT` | 7.24% | 724.12 |
| `MUL_MAT f32` | 6.41% | 640.94 |
| `MUL_MAT q4_K` | 4.91% | 491.27 |
| `GLU` | 1.70% | 169.83 |

Top hot shapes:

| Shape | Parsed share | Total ms |
| --- | ---: | ---: |
| `MUL_MAT q3_K m=17408 n=128 k=5120` | 39.33% | 3933.65 |
| `MUL_MAT q3_K m=5120 n=128 k=17408` | 18.35% | 1835.36 |
| `MUL_MAT q3_K m=10240 n=128 k=5120` | 9.27% | 926.85 |
| `MUL_MAT f32 m=48 n=128 k=5120` | 5.96% | 596.13 |
| `MUL_MAT q3_K m=6144 n=128 k=5120` | 5.94% | 594.31 |

The Q3 pipeline resource query reported zero scratch for the aligned large and
small Q3_K pipelines. The driver did not expose a richer VGPR/SGPR breakdown in
the captured log, so SPIR-V opcode diffs remain the primary static signal for a
shader-body prototype.

## Ceiling Sketch

With the pre-D005 P002 Vulkan cold baseline `1.6249 TPS`, the original diagnostic Amdahl estimate said:

- A 1.20x local speedup of all `MUL_MAT q3_K` projects to about `1.8728 TPS`.
- A 1.50x local speedup of all `MUL_MAT q3_K` projects to about `2.2100 TPS`.
- A 2.00x local speedup of only `FLASH_ATTN_EXT` projects to about `1.6860 TPS`.
- A 2.00x local speedup of the top `17408 x 128 x 5120` Q3_K shape projects to about `2.0226 TPS`.

With the current D005 `1.7898 TPS` default and the current `ub256` routepack,
the active target is `+11.7%` wall speedup to reach `2.0 TPS`. The post-D005
ceiling model puts all Q3_K at about `78.63%` of parsed prompt time, so an
all-Q3 local speedup of roughly `1.154x` is enough to reach the `2 TPS` gate.
These are route gates, not speed claims.

The D004 correction fixed the FFN family matcher in
`scripts/research/vulkan_route_ceiling.py`: the old helper only recognized the
historical `n=1024` FFN shape, while the active 130k trace uses `n=256`.
Corrected P002 route shares are:

| Route | Parsed share | Required local speedup for `2 TPS` |
| --- | ---: | ---: |
| Dense FFN gate/up Q3_K | `34.91%` | `1.920x` |
| Dense FFN down Q3_K | `24.61%` | `3.123x` |
| Dense FFN gate/up + down Q3_K | `59.52%` | `1.391x` |
| All Q3_K MUL_MAT | `80.50%` | `1.262x` |
| All Q3_K MUL_MAT + FA | `88.10%` | `1.234x` |

D004 also reran the dense FFN gate/up model for `17408x256x5120`. A simple
dual-A/same-B route has a base-tile local ceiling of only `1.417x` after
unchanged A-side Q3_K work is included, projecting to about `1.114x` wall on
the active lane. That makes gate/up-only fusion too weak as the next standalone
source prototype.

## 2026-05-26 Update

Measured no-code recenter on the same cold P002 lane:

| Variant | Runs | TPS | Prompt tok/s | Decode tok/s | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `p002-vulkan-ub128-confirm3` | 3 | `1.5635` | `811.02` | `42.41` | old control |
| `p002-vulkan-ub256-confirm3` | 3 | `1.6249` | `843.60` | `42.34` | pre-D005 control |
| `d005-vulkan-default-splitk-confirm3` | 3 | `1.7898` | `934.81` | `43.59` | current default |

Rejected nearby checks: `ub64` `0.97 TPS`, `b1024/ub128` `1.57 TPS`, `threads16` `1.57 TPS`. S001 signed-nibble runtime did not survive: all-Q3 failed fit, and `hot5` completed at `1.5186 TPS` versus a same-session `1.5798 TPS` control.

Follow-up ubatch sweep after the confirmation keeps `ub256` as the only nearby
positive point: `ub192` fell to `1.4011 TPS`, while `ub320`, `ub384`, and `ub512`
hit a severe prompt cliff (`0.3277`, `0.3550`, `0.3040 TPS`). The likely next
shape work is explaining this `256 -> 320` cliff, not promoting a larger ubatch.

Current recenter after the latest driver/workflow updates:

| Variant | Runs | TPS | Prompt tok/s | Decode tok/s | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `p002-vulkan-ub256-current-r1` | 1 | `1.6654` | `866.47` | `43.42` | current Vulkan anchor |
| `d005-vulkan-default-splitk-confirm3` | 3 | `1.7898` | `934.81` | `43.59` | current default after D005 |

D003 then tested the cliff path. Existing-route changes did not repair it:
forcing Q3_K `n>256` to the medium pipeline gave `141.33 tok/s` on `ub384`,
and splitting `n>256` into `n<=256` dispatches gave `204.42 tok/s` versus
`211.55 tok/s` control. `GGML_VK_ENABLE_MEMORY_PRIORITY=1` recovered `ub320`
prompt speed (`174.21 -> 808.73 tok/s`), but the full wall run was only
`1.5562 TPS`, below `ub256`. Keep it as diagnostic evidence, not a default.

D005 then changed the actual default route for the FFN down projection. The
targeted shape is Q3_K `m=5120,n>=128,k=17408`, where split-K 3 improved the
point trace from `2188.84 ms` to `1626.31 ms` for dense FFN down. Split-K 2 was
positive but smaller (`1817.54 ms`), while split-K 4 hit a reduce/temp overhead
cliff (`6050.53 ms`) and is rejected. The full cold lane confirmed the wall
gain: paired r3 control `1.6679 TPS`, opt-in split-K 3 `1.7774 TPS`, and the
no-env default route `1.7898 TPS`. Prompt eval moved `867.95 -> 934.81 tok/s`;
decode stayed tied. `GGML_VK_Q3K_FFN_DOWN_SPLIT_K=0` or `1` is the rollback.

D006 checked whether the remaining 130k instability was a simple output-layer
residency issue. After the GPU power limit was restored to `+10%`, a fresh D005
default/full-output run still cliffed at `0.3252 TPS`, prompt `163.67 tok/s`,
decode `41.65 tok/s`, with `11434.19 MiB` Vulkan model buffer and `2` graph
splits. Moving the output layer out of device-local VRAM recovered prompt eval:

| Variant | Label | B/UB | TPS | Prompt tok/s | Decode tok/s | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| D005 default after power restore | `d006-vulkan-130k-d005-default-powerplus-r1` | `512/256` | `0.3252` | `163.67` | `41.65` | residency cliff evidence |
| `LLAMA_NO_OUTPUT_OFFLOAD` | `d006-vulkan-130k-no-output-ub512-powerplus-r1` | `512/512` | `1.7551` | `955.88` | `22.13` | diagnostic only |
| `LLAMA_OUTPUT_HOST_GPU_DEV` | `d006-vulkan-130k-output-host-gpudev-ub512-r1` | `512/512` | `1.7769` | `968.32` | `22.28` | diagnostic only |

Both output-relief paths moved model residency to about `10430.07 MiB` Vulkan
and `1515.64 MiB` host, but decode dropped to about half of the D005 default
decode rate. Earlier one-run output-relief variants reached up to `1.8068 TPS`,
but they were not r3-confirmed and still carried the same decode penalty.
Therefore D006 is evidence that residency/lifetime matters; it is not a new
baseline and not a route to `2 TPS` by itself.

D007 extended `GGML_VK_FFN_ROUTE_TRACE` to classify whole dense-FFN block
coverage before writing shader code. The active 130k graph still exposes all
gate/up+GLU candidates, but a strict adjacent `MUL_MAT,MUL_MAT,GLU,MUL_MAT`
whole-block matcher covers only `16/63` prefill layers; the other `47` are
blocked because the node after GLU is an unrelated `VIEW`, not down `MUL_MAT`.
The simple `VIEW`-aware path recovered `0` candidates because that `VIEW` is not
sourced from GLU (`src=NONE src_is_glu=0`). A non-adjacent dependency scan that
finds down by `down->src[1] == GLU` recovers all Q3_K FFN blocks:

| Trace | Count / Result |
| --- | ---: |
| Gate/up + GLU Q3_K candidates | `64` |
| Prefill gate/up + GLU candidates | `63` |
| Strict whole-block prefill matches | `16` |
| Simple `VIEW`-aware recovered matches | `0` |
| Non-adjacent dependency-scanned block matches | `64` |
| Scanned prefill gaps | `16` at `gap=3`, `47` at `gap=4` |
| Rejects: next op is `VIEW` | `47` |
| Source/type/fuse rejects | `0` |

This blocks an immediate strict-adjacent whole-FFN shader prototype, but it keeps
the broader whole-FFN candidate alive. The next source gate is a non-adjacent
whole-FFN design/ceiling update, not another ubatch/flag sweep.

These are ranking estimates only. Any speed claim still requires paired cold
P002 A/B with `max_tokens=16`, no reuse, and no v2 prime pass.

## Decision

For the P002 130k quick lane, the first Vulkan topology design should target the
Q3_K prompt route, especially dense FFN gate/up and down projections. FA work is
still relevant for long-KV behavior, but this pack does not justify making FA the
first rewrite candidate for the current cold quick baseline.

diagnostic only. D007 blocks a naive strict whole-FFN fusion until the `VIEW`
D004 narrows that further: do not build a gate/up-only fusion prototype as the
next step. D005 takes the low-risk down-projection split-K win and shifts the
remaining target. D006 closes the output-placement/ubatch relief branch as
diagnostic only. D007 blocks a naive strict-adjacent whole-FFN fusion, but a
non-adjacent dependency scan recovers all Q3_K FFN blocks for a design gate. At
the kept `1.7898 TPS` baseline, reaching `2 TPS` needs
about `1.154x` local on all Q3_K or `1.137x` local on Q3_K+FA in the post-D005
trace.

Next gate before source edits:

1. Recompute the ceiling with `scan_blocks=64` non-adjacent whole-block coverage.
2. Write a design note for a whole-FFN/Q3_K route with node scheduling, intermediate activation residency, memory traffic, output/logits residency behavior, D005 split-K interaction, and SPIR-V deltas.
3. Produce a tiny correctness scout for prompt and decode path invariants.
4. Build a default-off prototype and compare SPIR-V/resource deltas before server A/B.
5. Run paired cold P002 A/B on the same lane and update this note with measured results.
