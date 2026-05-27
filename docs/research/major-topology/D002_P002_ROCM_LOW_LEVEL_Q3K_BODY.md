# D002 P002 ROCm Low-Level Q3_K Body Gate

Status: open; S002A/S002B/S002D and D013-D023 route gates are rejected as
standalone runtime candidates. The only current source change is a correctness
fix for Q3_K padded partial slices plus a default-off diagnostic trace.

## Intent

The user asked whether moving selected hot blocks to a lower-level programming
layer can unlock more speed. The answer for this fork is yes, but only if the
lower layer expresses a route that the current C++ runtime plus rocBLAS path
cannot express. This design gate scopes that idea to one measurable target:
ROCm/RDNA4 Q3_K prompt matmul on the dense 130k Qwen lane.

This is not a rewrite of llama.cpp, the GUI, or the scheduler in another
language. The C++ runtime stays the control plane. The candidate low-level body
is a narrowly-scoped HIP/RDNA4 kernel or code-object route behind an opt-in gate.

## Active Lane

- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Context: `ctx=131072`, cold-first, repo-snapshot real context.
- Workload: `quick:triage_diff`, `real-context-chars=24576`, `max_tokens=16`.
- Reuse/prime: off/off; thinking on.
- KV/spec: `q4_0/q4_0`, `--spec-type none`.
- Current baselines: Vulkan D012 stack `2.0013 TPS` r3 after q3quad/GLU plus
  `--no-mmap` (D005 split-K anchor `1.7898 TPS`); ROCm `b512/ub128` `1.5200 TPS`
  r3 (`p002-rocm-ub128-current-confirm3`). The older ROCm `1.3984 TPS` scout is
  now historical because the same current lane recentered higher without a route
  change.

The first goal is not to beat all Vulkan behavior. The first gate is to prove a
ROCm low-level route has enough local speedup to justify runtime integration.
With the current `1.5200 TPS -> 2.0 TPS` target, the Amdahl gate is about
`2.04x` local speedup if only the measured `~0.47` Q3_K MMQ share moves, or
about `1.43x` local speedup if a broader `~0.80` all-Q3/FFN share moves.

## Why ROCm First

Vulkan already has a specialized shader route and a fresh no-code recenter at
`ub256`. ROCm remains behind on the same 130k quick lane. Prior H35/H42 evidence
focused on Q3_K staging plus library GEMM, but the fresh P002 `ub=128` trace below
shows the active 130k route is MMQ/direct rather than the old cublas split route.
A lower-level body is relevant only if it reduces real Q3_K MMQ work or changes
the dataflow, not if it merely wraps the same library call or selector choice.

## Prior Rejection Fence

Do not repeat these as the new low-level experiment:

- E245 direct Q3FlashMatmul P0/P1/P2/P3 bodies: correctness passed, but point
  timing was far slower than `Q3_K -> fp16 -> rocBLAS`.
- E246 streaming Q3_K dequant plus rocBLAS: standalone points were positive, but
  the real-server runtime route regressed or timed out once safe stream/lifetime
  dependencies were enforced.
- E249 hipBLASLt grouped GEMM: ROCm 7.1 Windows exposed no grouped algorithms
  for the active contracts.
- Existing-MMQ selector forcing, row chunking, fp16-output-with-convert,
  graph-level F16 activation casts, broad persistent fp16 Q3_K cache, broad src1
  reuse stacks, GLU-only CUDA/HIP fast paths, dense Q3_K staging, padded b4
  loads, Q3FlashMatmul active-shape promotion, wider-N scalar Q3Flash tile
  variants, dual-Y MMQ dataflow, ROCm compute-vbuffer single-chunk mode, and
  multi-row WMMA Q3Flash tile reuse, upstream-stock rollback, and streaming
  dequant+rocBLAS chunk sizing are already rejected or unstable.

## Candidate Shape

Before writing a kernel, refresh ROCm P002 route evidence. The old H42 work used
short-context `n=2048` shapes, while P002 ROCm starts at physical `ub=128`. The
candidate must be selected from the new 130k trace, not inherited from the 12k
lane by habit.

Expected Qwen-family forms to check first:

- `m=17408, n=128, k=5120` gate/up FFN family;
- `m=5120, n=128, k=17408` down/reverse family;
- `m=10240, n=128, k=5120` and `m=6144, n=128, k=5120` secondary FFN forms.

## Low-Level Ladder

Use the lowest layer that answers the next question, not the lowest possible
layer immediately:

1. HIP C++ scout with RDNA4 WMMA builtins and explicit resource/timing output.
2. HIP inline assembly or hand-written GCN only if the HIP scout is close enough
   to rocBLAS point timing and the remaining blocker is instruction selection or
   register allocation.
3. Runtime integration only after standalone point timing and correctness pass.

Inline GCN is therefore a phase-2 tool, not the starting point. If HIP C++ is
`0.3x` of rocBLAS again, assembly will not rescue the route without a different
dataflow.

## Gate Model

Use `scripts/research/rocm_lowlevel_route_gate.py` after the ROCm P002 trace has
a reliable touched-route share. For a rough first pass, the original "catch
current Vulkan" gate from the old ROCm scout (`1.3984 TPS`) was:

| Touched ROCm wall share | Required local speedup |
| ---: | ---: |
| `0.47` | `1.4216x` |
| `0.50` | `1.3866x` |
| `0.60` | `1.3026x` |
| `0.70` | `1.2486x` |
| `0.80` | `1.2110x` |

These are not speed claims. They are coding thresholds. A candidate with only a
`1.05x` point win on a 60-70% route is not worth a risky runtime branch.

After the 2026-05-26 ROCm recenter, the project target is stricter: reach
`2.0 TPS` from the current ROCm `1.5200 TPS` baseline. That requires much larger
local gains:

| Touched ROCm wall share | Required local speedup to reach `2.0 TPS` |
| ---: | ---: |
| `0.47` | `2.0435x` |
| `0.50` | `1.9231x` |
| `0.60` | `1.6667x` |
| `0.70` | `1.5217x` |
| `0.80` | `1.4286x` |

Implication: isolated load-width or selector tweaks are now even less likely to
matter. D002 needs a body/topology/dataflow change, or it should stay as a
documented scout lane while the main 2 TPS search moves to a larger touched
share.

## 2026-05-26 Trace Notes

Diagnostic traces used `max_tokens=1` to keep the run cheap; they are route and
resource evidence, not TPS claims.

Artifacts:

- `build_logs/agent-workload/d002-rocm130k-q3k-splittrace-r1.server.log`
- `build_logs/agent-workload/d002-rocm130k-q3k-splittrace-min1-r1.server.log`
- `build_logs/agent-workload/d002-rocm130k-q3k-mmqtrace-r1.server.log`
- `build_logs/agent-workload/d002-rocm130k-q3k-mmqtrace-r1.mmq-summary.md`

Result:

- `GGML_TRACE_CUBLAS_Q3K_ROUTE` with `min_ncols=1` produced no Q3_K rows on the
  P002 ROCm `ub=128` lane. The visible rows were `f32/f32/f32`, so the old H42
  cublas split assumption does not describe this lane.
- `GGML_TRACE_MMQ_TIMING=1`, `GGML_TRACE_MMQ_TIMING_SYNC=1`, and
  `GGML_TRACE_MMQ_RESOURCES=1` captured `25011` MMQ timing rows.
- MMQ total: `6101.061 ms` of a `12.23 s` diagnostic wall (`49.89%`).
- Q3_K within MMQ: `5754.612 ms`, `21987` rows, `94.32%` of MMQ and `47.05%`
  of diagnostic wall.
- Q4_K within MMQ: `346.449 ms`, `5.68%` of MMQ.

Top P002 ROCm Q3_K MMQ shapes:

| Shape | Count | Total ms | MMQ share | Resources |
| --- | ---: | ---: | ---: | --- |
| `q3_K nrows=17408 ncols=128` | `7812` | `1813.080` | `29.72%` | `mmq=128x64`, `regs=183`, `LDS=40448`, `occ=6.25%` |
| `q3_K nrows=5120 ncols=128` | `4898` | `1568.555` | `25.71%` | `mmq=128x64`, `regs=183`, `LDS=40448`, `occ=6.25%` |
| `q3_K nrows=6144 ncols=128` | `2976` | `1316.417` | `21.58%` | `mmq=128x64`, `regs=183`, `LDS=40448`, `occ=6.25%` |
| `q3_K nrows=10240 ncols=128` | `2976` | `504.781` | `8.27%` | `mmq=128x64`, `regs=183`, `LDS=40448`, `occ=6.25%` |

Gate implication: if the actionable touched share is only the measured Q3_K MMQ
wall share (`~0.47`), reaching `2.0 TPS` from the current ROCm `1.5200 TPS`
baseline requires a local `~2.04x` improvement. A `1.15x` local body on that
share would project only about `+6.5%` to the ROCm lane; a `~1.43x` local target
is sufficient only if the candidate genuinely broadens the touched route toward
`~0.80` of the wall.

## S002A Padded 32-Bit Load Scout

Added standalone scout:

```bash
hipcc -std=c++17 -O3 --offload-arch=gfx1201 \
  scripts/research/rocm_q3k_mmq_unpack_scout.cpp \
  -o build_logs/agent-workload/rocm_q3k_mmq_unpack_scout.exe
```

Purpose: test whether the padded Q3_K path can replace current two-16-bit
`get_int_b2` loads with aligned 32-bit loads for `hmask`, `qs`, and `scales`.
This is lower-level than a selector toggle but smaller than a runtime kernel
rewrite.

Artifacts:

- `scripts/research/rocm_q3k_mmq_unpack_scout.cpp`
- `build_logs/agent-workload/d002-rocm-q3k-mmq-unpack-scout.md`

Result:

| Blocks | Global-output b4/b2 | Shared-output b4/b2 | Correctness |
| ---: | ---: | ---: | --- |
| `262144` | `1.3458x` | `1.0708x` | clean |
| `1048576` | `1.0383x` | `1.0003x` | clean |

Decision: reject S002A as a standalone runtime patch. The aligned 32-bit load
variant is correct, but the LDS-like measurement does not show a robust local
speedup, and it is far below the D002 `~1.42x` local gate. Do not add a runtime
`get_int_b4` Q3_K loader unless it is part of a larger body/dataflow change.

## S002B RDNA4 Stream-K Threshold Gate

Probe: force the existing MMQ stream-K threshold to the active `ncols=128` lane
with `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11=1`. This is not a new body, so it was
treated as a cheap point-level gate only.

Artifacts:

- `build_logs/agent-workload/d002-rocm130k-q3k-streamk-min1-r1.server.log`
- `build_logs/agent-workload/d002-rocm130k-q3k-streamk-min1-r1.mmq-summary.md`
- `build_logs/agent-workload/d002-rocm130k-streamk-min1-full-r1.diagnostics.md`
- `build_logs/agent-workload/d002-rocm130k-control-neighbor-r1.diagnostics.md`
- `build_logs/agent-workload/p002-rocm-ub128-current-confirm3.diagnostics.md`

Point result:

| Scope | Prior trace | Stream-K min1 trace | Delta |
| --- | ---: | ---: | ---: |
| Q3_K MMQ total | `5754.612 ms` | `5571.978 ms` | `-3.17%` |
| Top `17408x128` shape | `1813.080 ms` | `1739.697 ms` | `-4.05%` |
| Top `5120x128` shape | `1568.555 ms` | `1514.647 ms` | `-3.44%` |

Wall A/B:

| Route | Label | TPS | Prompt tok/s | Decode tok/s |
| --- | --- | ---: | ---: | ---: |
| Stream-K min1 | `d002-rocm130k-streamk-min1-full-r1` | `1.5196` | `802.05` | `28.64` |
| Neighbor control | `d002-rocm130k-control-neighbor-r1` | `1.5206` | `802.81` | `28.50` |
| Current confirm baseline | `p002-rocm-ub128-current-confirm3` | `1.5200` | `801.71` | `29.07` |

Decision: reject S002B as a speed route. The point movement is far below the
new `2.0 TPS` gate, and the full wall run ties/slightly loses to the neighboring
control. The higher ROCm number versus the original `1.3984 TPS` scout is a
baseline recenter on the current tree/system state, not an effect of this knob.

## S002C ROCm `ubatch=256` Shape Gate

Probe: recheck the nearby physical `ubatch=256` ROCm shape on the current
`build-rocm-vec` binary after the workflow/driver recenter. This was a single
shape gate, not a broad ubatch sweep.

Artifact:

- `build_logs/agent-workload/d004-rocm-ub256-current-r1.server.log`

Result:

- The server exited before readiness with
  `ggml/src/ggml-cuda/ggml-cuda.cu:1017: GGML_ASSERT(size % sizeof(block_q3_K) == 0) failed`.
- The assert is in `ggml_cuda_q3k_padded_storage_nblocks_from_raw_slice`, after
  the model had loaded and before the benchmark request could run.
- Treat `ubatch=256` as a correctness/storage gate failure for the current
  ROCm 130k lane, not as a speed candidate.

Decision: keep ROCm P002 baseline at `b512/ub128`. Do not continue ROCm
`ubatch=256` tuning until the padded-storage/view/copy slice has a dedicated
correctness investigation.

## S002D ROCm `ubatch=256` Raw-Storage Escape Gate

Probe: test whether the S002C assert was only a padded-storage implementation
problem by disabling the padded Q3_K storage path:

```powershell
GGML_CUDA_Q3K_PADDED_STORAGE=0
GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=0
```

This is a negative/escape gate, not a proposed default. The goal was to see if
raw storage can make the nearby `ubatch=256` shape runnable and fast enough to
justify a full wall A/B.

Artifacts:

- `build_logs/agent-workload/d002-rocm-ub256-paddedoff-max1-r1.server.log`
- `build_logs/agent-workload/d002-rocm-ub256-paddedoff-max1-r1.diagnostics.md`
- `build_logs/agent-workload/d002-rocm-ub256-paddedoff-max1-r1.csv`

Result:

- The server loaded successfully and avoided the S002C assert.
- The first `max_tokens=1` request hit the `90 s` hard task timeout before
  prompt processing completed.
- The log reached only `6144 / 7970` prompt tokens by the timeout, versus the
  current ROCm `ub128` lane at about `802 prompt tok/s`.

Decision: reject raw-storage `ub256` as a speed escape. This confirms the
correctness branch cannot be bypassed by simply disabling padded storage; raw
Q3_K storage at this shape is far too slow for the active 130k lane. Keep ROCm
at `ub128`, and reopen `ub256` only through a padded-storage/view/copy fix that
preserves the fast padded MMQ path.

## D013 Q3_K Padded Partial-Slice Fix And Route Gates

After S002C/S002D, the padded-storage copy helpers in
`ggml/src/ggml-cuda/ggml-cuda.cu` were changed to support partial raw
`block_q3_K` slices against physically padded Q3_K storage. The change fixes
model-load/view/copy correctness for shapes that do not pass whole raw blocks
through the backend copy API. It is not a speed claim.

Artifacts:

- `build_logs/agent-workload/d013-rocm130k-paddedfix-ub128-control-r1.diagnostics.md`
- `build_logs/agent-workload/d013-rocm130k-ub256-paddedfix-full-r1.diagnostics.md`
- `build_logs/agent-workload/d013-rocm130k-q3k-cublas-threshold0-max1-r1.diagnostics.md`
- `build_logs/agent-workload/d013-rocm130k-nommap-control-r1.diagnostics.md`
- `build_logs/agent-workload/d013-rocm130k-q3k-src1quant-presync-r1.server.log`
- `build_logs/agent-workload/d013-rocm130k-y32w2-max1-r1.diagnostics.md`
- `build_logs/agent-workload/d013-rocm130k-y32w2-postrevert-max1-r1.diagnostics.md`

Results:

| Probe | Result | Decision |
| --- | ---: | --- |
| `ub128` after padded partial-slice fix | `1.5074 TPS`, prompt `796.48 tok/s` | Neutral enough for correctness fix; no speed claim |
| `ub256` after padded partial-slice fix | `1.4138 TPS`, prompt `744.40 tok/s`, decode `28.59 tok/s` | Reject `ub256` as speed route |
| Force Q3_K `n=128` away from MMQ to cublas/dequant path | prompt `429.99 tok/s` on `max_tokens=1` | Reject; temporary threshold knob reverted |
| ROCm `--no-mmap` | `1.5028 TPS`, prompt `793.91 tok/s`, decode `28.65 tok/s` | Reject as ROCm P002 speed route |
| Q3_K `src1 -> q8_1` quant pre-sync trace | pure quant sync `793.596 ms`; previous-work pre-sync `5599.930 ms`; `21987` rows | Reject activation quant reuse/fusion as primary 2 TPS route; ceiling is about 8% of baseline prompt |
| RDNA4 MMQ `y32/w2` temporary build | prompt `662.48 tok/s` on `max_tokens=1` | Reject and revert; post-revert prompt sanity returned to `792.88 tok/s` |

The `GGML_TRACE_MMQ_SRC1_QUANT_TIMING` diagnostic remains default-off because it
answered a real design question: quantization of `src1` is measurable but too
small to bridge `1.52 -> 2.0 TPS`. With `GGML_TRACE_MMQ_SRC1_QUANT_TIMING_SYNC=1`
and `GGML_TRACE_MMQ_SRC1_QUANT_TIMING_PRE_SYNC=1`, `pre_sync_ms` mostly captures
the previous MMQ body while `sync_ms` isolates the quantization launch.

Decision: keep the padded partial-slice correctness fix and the default-off
trace. Do not promote `ub256`, Q3_K cublas/dequant fallback, ROCm `--no-mmap`,
activation quant reuse, or `y32/w2` geometry. The next ROCm 2 TPS candidate must
change Q3_K body/dataflow or touch a larger route share than isolated source-1
quantization or launch geometry.

## D014-D018 MMQ/GLU Micro-Route Gates

After D013, five smaller route probes checked whether the remaining ROCm gap was
recoverable through adjacent GLU work, current-MMQ loader changes, or a direct
Q3FlashMatmul promotion for the active `n=128` hot shapes. They all failed the
2 TPS gate and were reverted where they changed runtime code.

Artifacts:

- `build_logs/agent-workload/d014-rocm130k-glu-contig-max1-r1.diagnostics.md`
- `build_logs/agent-workload/d014-rocm130k-glu-contig-full-r1.diagnostics.md`
- `build_logs/agent-workload/d015-rocm130k-dense-q3k-staging-max1-r1.diagnostics.md`
- `build_logs/agent-workload/d016-rocm130k-q3k-padded-b4loads-max1-r1.diagnostics.md`
- `build_logs/agent-workload/d017-rocm-q3flash-p002-17408x128x5120.csv`
- `build_logs/agent-workload/d017-rocm-q3flash-p002-5120x128x17408.csv`
- `build_logs/agent-workload/d018-rocm130k-q3k-dualy-max1-r1.diagnostics.md`

Results:

| Probe | Result | Decision |
| --- | ---: | --- |
| D014 CUDA/HIP GLU contiguous split fast path | `max_tokens=1` prompt `795.92 tok/s`; full run `1.5059 TPS`, prompt `795.62`, decode `28.64` | Reject/revert; GLU-only work is below the required touched share |
| D015 dense Q3_K staging in current MMQ body | prompt `684.45 tok/s` | Reject/revert; staging increases work and loses badly |
| D016 padded Q3_K b4 loads in current MMQ body | prompt `760.23 tok/s` | Reject/revert; wider loads raise enough pressure/overhead to lose |
| D017 Q3FlashMatmul active `n=128` scout | `17408x128x5120` pipeline `2.0677 ms` vs baseline `2.2030 ms` (`1.0654x`); `5120x128x17408` pipeline `1.8126 ms` vs baseline `1.6496 ms` (`0.9101x`) | Reject runtime promotion; point win is mixed and far below the `2.0 TPS` local gate |
| D018 dual-Y MMQ dataflow | prompt `735.68 tok/s` | Reject/revert; barrier/work reuse attempt regressed prompt |

Fusion note: CUDA/HIP already has `gate + up + GLU` fusion in
`ggml_cuda_try_fuse`, but the implemented route dispatches only to vector paths
(`ggml_cuda_mul_mat_vec_f/q`) and rejects `ncols_dst != 1`. The active 130k
prefill lane uses MMQ with `ncols=128`, so enabling the existing fusion is not a
valid speed route. A real FFN-level route would need new MMQ/prefill fusion or a
different topology, not a flag flip.

Decision: extend the rejection fence. Do not repeat isolated GLU-contiguity,
dense staging, padded b4-load, Q3Flash active-shape promotion, or dual-Y MMQ
probes as the next ROCm 130k route. Simple loader/barrier/body reshuffles have
not approached the required local speedup; the next candidate must change a
larger FFN/Q3_K dataflow or prove a new compressed-GEMM topology before runtime
integration.

## D019 Wider-N Direct Q3Flash Scout

Probe: extend `scripts/research/rocm_q3flashmatmul_scout.cpp` with scalar
`16x32x256` and `16x64x256` direct Q3_K tiles. This answered the first allowed
post-D018 scout question: whether reusing a dequantized Q3 tile across more of
the active `n=128` prompt columns can rescue the direct compressed-matmul route
without touching the runtime.

Artifacts:

- `build_logs/agent-workload/d019-rocm-q3flash-wide-small-r1.csv`
- `build_logs/agent-workload/d019-rocm-q3flash-wide-p002-17408x128x5120.csv`
- `build_logs/agent-workload/d019-rocm-q3flash-wide-p002-5120x128x17408.csv`

Correctness gate:

- Small `128x128x512` shape passed against in-process `Q3_K -> f16 -> rocBLAS`:
  `wide32_max_abs=0`, `wide64_max_abs=0`, `rmse=0`.

Point result:

| Shape | rocBLAS baseline ms | P0 `16x16` | Wide32 `16x32` | Wide64 `16x64` | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `17408x128x5120` | `1.5857` | `6.6182 ms` (`0.2396x`) | `5.8322 ms` (`0.2719x`) | `6.9938 ms` (`0.2267x`) | Reject |
| `5120x128x17408` | `1.4849` | `6.7539 ms` (`0.2199x`) | `5.9205 ms` (`0.2508x`) | `7.0936 ms` (`0.2093x`) | Reject |

Fresh same-binary pipeline timings were also negative in this run
(`0.8108x` and `0.7306x` vs the rocBLAS baseline), reinforcing the earlier
D017 decision not to promote the Q3Flash/runtime path for P002.

Decision: reject D019 as a runtime route. Wider-N scalar tiling reduces some
repeated Q3 unpack versus P0, but it remains about `4x` slower than the local
`dequant+rocBLAS` reference and therefore cannot plausibly beat the current
P002 MMQ/direct route or the `~2.04x` local gate at `~0.47` touched share. Keep
the scout variants only as diagnostic harness code; the next route must preserve
matrix-core occupancy or broaden to FFN-level dataflow rather than scalar direct
Q3Flash tiling.

## D020 Compute VBuffer Single-Chunk Control

Probe: run the same P002 ROCm 130k cold lane with
`GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1`. This is the allocator/layout
negative control requested by the RDNA4 workflow before drawing conclusions from
body or selector experiments.

Artifact:

- `build_logs/agent-workload/d020-rocm130k-vbuffer-singlechunk-r1.diagnostics.md`

Wall result:

| Route | Label | TPS | Prompt tok/s | Decode tok/s |
| --- | --- | ---: | ---: | ---: |
| Current ROCm baseline | `p002-rocm-ub128-current-confirm3` | `1.5200` | `801.71` | `29.07` |
| VBuffer single chunk | `d020-rocm130k-vbuffer-singlechunk-r1` | `1.5067` | `798.48` | `28.78` |

Decision: reject single-chunk vbuffer mode as a speed route for the active 130k
ROCm lane. It is slightly slower than baseline and does not explain the current
gap to 2 TPS. Keep it only as a causality/rollback control for future allocator
work; the next performance route must still move Q3_K/FFN dataflow or prove a
matrix-core-preserving compressed GEMM body.

## D021 Multi-Row WMMA Q3Flash Scout

Probe: extend `scripts/research/rocm_q3flashmatmul_scout.cpp` with a P4
`64x64x128` multi-row WMMA tile. P4 keeps the matrix-core route from P3 but
loads one B tile into shared memory and reuses it across four M waves, testing
whether the direct compressed-GEMM route was mainly losing to repeated B traffic
per 16-row tile.

Artifacts:

- `build_logs/agent-workload/d021-rocm-q3flash-p4-small-r1.csv`
- `build_logs/agent-workload/d021-rocm-q3flash-p4-p002-17408x128x5120.csv`
- `build_logs/agent-workload/d021-rocm-q3flash-p4-p002-5120x128x17408.csv`
- `build_logs/agent-workload/d021-rocm-q3flash-p4-p002-10240x128x5120.csv`
- `build_logs/agent-workload/d021-rocm-q3flash-p4-p002-6144x128x5120.csv`

Correctness gate:

- Small `128x128x512` shape passed against in-process `Q3_K -> f16 -> rocBLAS`:
  P4 `max_abs=0`, `max_rel=0.0004`, `rmse=0`.

Point result:

| Shape | rocBLAS baseline ms | P3 k-stage WMMA | P4 multi-row WMMA | Pipeline reference | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `17408x128x5120` | `1.5909` | `3.7747 ms` (`0.4215x`) | `3.0377 ms` (`0.5237x`) | `1.9077 ms` (`0.8339x`) | Reject |
| `5120x128x17408` | `1.4678` | `3.5298 ms` (`0.4158x`) | `3.5702 ms` (`0.4111x`) | `2.1184 ms` (`0.6929x`) | Reject |
| `10240x128x5120` | `0.9128` | `1.9643 ms` (`0.4647x`) | `1.8321 ms` (`0.4982x`) | `1.2352 ms` (`0.7390x`) | Reject |
| `6144x128x5120` | `0.4830` | `1.2158 ms` (`0.3973x`) | `1.1746 ms` (`0.4112x`) | `0.9167 ms` (`0.5269x`) | Reject |

Decision: reject D021 as a runtime route. The multi-row B-reuse shape proves a
better direct-WMMA dataflow than P3 on the gate/up shape, but it still remains
about `1.9x` slower than the local `dequant+rocBLAS` point baseline and does not
improve the down/reverse shape; secondary `10240`/`6144` forms stayed below
`0.50x`. This is not close to the `~2.04x` local gate; keep P4 only as
diagnostic evidence. Future ROCm work must either find a much stronger
compressed-GEMM topology than per-block Q3Flash/WMMAs or broaden to an FFN-level
route that changes more of the wall share.

## D022 Upstream-Stock ROCm Control

Probe: run the imported upstream-stock ROCm binary from
`build-rocm-upstream-stock/bin/llama-server.exe` on the exact P002 130k cold
lane. This checks whether the local fork carries a hidden ROCm regression or
whether the active fork-specific ROCm/runtime changes are part of the current
baseline.

Artifact:

- `build_logs/agent-workload/d022-rocm130k-upstream-stock-r1.diagnostics.md`

Wall result:

| Route | Label | TPS | Prompt tok/s | Decode tok/s |
| --- | --- | ---: | ---: | ---: |
| Current fork ROCm baseline | `p002-rocm-ub128-current-confirm3` | `1.5200` | `801.71` | `29.07` |
| Upstream-stock ROCm import | `d022-rocm130k-upstream-stock-r1` | `0.5720` | `294.40` | `21.96` |

Decision: reject upstream-stock rollback as a speed route. The stock build is
about `62%` slower on the same lane, so the current fork is not carrying the
missing-regression explanation for ROCm. Continue from the local ROCm baseline;
do not spend the next cycle bisecting toward the old stock import unless a newer
upstream commit is explicitly being evaluated for sync.

## D023 Streaming Dequant+rocBLAS Chunk Sweep

Probe: use the existing `rocm_q3flashmatmul_scout` streaming pipeline mode to
test whether larger per-chunk `Q3_K -> fp16 -> rocBLAS` staging can beat the
local in-process full-dequant baseline on P002 `n=128` hot shapes. This closes
the remaining point-level library-staging angle before moving away from Q3Flash
style scouts.

Artifacts:

- `build_logs/agent-workload/d023-rocm-q3-pipeline-chunk512-17408x128x5120.csv`
- `build_logs/agent-workload/d023-rocm-q3-pipeline-chunk1024-17408x128x5120.csv`
- `build_logs/agent-workload/d023-rocm-q3-pipeline-chunk4096-17408x128x5120.csv`
- `build_logs/agent-workload/d023-rocm-q3-pipeline-chunk8192-17408x128x5120.csv`
- `build_logs/agent-workload/d023-rocm-q3-pipeline-chunk4096-5120x128x17408.csv`
- `build_logs/agent-workload/d023-rocm-q3-pipeline-chunk8192-5120x128x17408.csv`

Point result:

| Shape | Best chunk | Baseline ms | Pipeline ms | Speedup vs baseline | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `17408x128x5120` | `8192` | `1.6031` | `1.7697` | `0.9058x` | Reject |
| `5120x128x17408` | `8192` | `1.4914` | `1.5974` | `0.9336x` | Reject |

Decision: reject streaming dequant+rocBLAS chunk sizing as a P002 route. The
large chunk sweep approaches the local full-dequant baseline but still loses on
both hot shapes, and prior runtime staging work already showed synchronization
and lifetime risks. Do not build another runtime cublas/dequant wrapper for the
active 130k ROCm lane unless a new point proof first exceeds the local baseline
by a meaningful margin.

## D024 FFN Pair+SwiGLU WMMA Scout

Probe: add `scripts/research/rocm_ffn_pairglu_scout.cpp`, a standalone HIP
scout for the broad FFN direction. It computes `gate` and `up` Q3_K projections
in one WMMA tile body, shares the B/activation tile between both projections,
and applies SwiGLU in the epilogue. The comparator is deliberately strict but
local: two `Q3_K -> f16 -> rocBLAS` GEMMs followed by a separate SwiGLU kernel.
This answers whether shared-B FFN pairing can beat even a simple library pair
before any runtime integration or down-projection fusion is considered.

Artifacts:

- `scripts/research/rocm_ffn_pairglu_scout.cpp`
- `build_logs/agent-workload/rocm_ffn_pairglu_scout.exe`
- `build_logs/agent-workload/d024-rocm-ffn-pairglu-small-r1.csv`
- `build_logs/agent-workload/d024-rocm-ffn-pairglu-p002-17408x128x5120.csv`
- `build_logs/agent-workload/d024-rocm-ffn-pairglu-p002-5120x128x17408.csv`

Correctness gate:

- Small `128x128x512` shape passed against the paired rocBLAS+SwiGLU baseline:
  `pair64_max_abs=0`, `pair64_max_rel=0.0025`, `pair64_rmse=0`.
- The reverse hot shape also checked all `655360` output elements with
  `pair64_max_abs=0`, `pair64_rmse=0`.

Point result:

| Shape | Paired rocBLAS+SwiGLU baseline | Pair64 shared-B WMMA | Speedup | Decision |
| --- | ---: | ---: | ---: | --- |
| `128x128x512` | `0.0546 ms` | `0.0918 ms` | `0.5951x` | Correct but slow |
| `17408x128x5120` | `3.6452 ms` | `6.8151 ms` | `0.5349x` | Reject |
| `5120x128x17408` | `3.0370 ms` | `8.2056 ms` | `0.3701x` | Reject |

Decision: reject the shared-B FFN pair+SwiGLU WMMA scout as a runtime route.
The design broadens beyond isolated GLU and avoids repeating the activation tile
for gate/up, but the double Q3_K unpack plus two accumulator streams outweigh
the saved B traffic. Because it loses by `1.9x-2.7x` against even the local
paired library baseline, it cannot plausibly beat the current P002 MMQ/direct
route or project toward `2 TPS`. Do not promote this pair-only fused FFN body;
future FFN work must either include a stronger down-projection streaming design
with a new resource model or switch to a different compressed-GEMM/layout proof.

## D025 Full-FFN Streaming Design Gate

Probe: add `scripts/research/rocm_ffn_streaming_gate.py` to quantify the next
obvious extension after D024: fusing `gate/up + SwiGLU + down` so the hidden
activation does not become a global intermediate. This is an analytical gate,
not a TPS claim. It checks the two unavoidable implementation choices for a
single-kernel or tightly streamed FFN design on the active P002 shape.

Artifact:

- `build_logs/agent-workload/d025-rocm-ffn-streaming-gate.md`

Inputs:

- Hidden width `17408`, output width `5120`, `ncols=128`.
- D024 paired rocBLAS+SwiGLU gate/up point: `3.6452 ms`.
- D023/D024 down-like point reference: `1.4914 ms`.
- Current materialized lower-bound point model: `5.1366 ms`.

Design result:

| Route model | Lower-bound local time | Speedup vs materialized point model | Blocker |
| --- | ---: | ---: | --- |
| D024 pair-only fused gate/up+SwiGLU plus unchanged down | `8.3065 ms` | `0.6184x` | Pair body is already slower than separate rocBLAS pair |
| Full streaming without hidden materialization, recompute per `64` down rows | `293.1074 ms` | `0.0175x` | Requires `80` gate/up recomputes |
| Hidden-tile partial output accumulation, `hidden_tile=128` | bandwidth-only | n/a | Adds `680.0 MiB` partial output read/write traffic per layer |

Memory sketch:

- Minimum SwiGLU hidden materialization is `8.50 MiB` per layer.
- Separate gate/up/SwiGLU intermediates are `25.50 MiB` per layer.
- One down output tensor is `2.50 MiB`.

Decision: reject naive full-FFN streaming as the next implementation route. A
design that avoids hidden materialization needs cross-down-row sharing of the
hidden tile. Without grid-wide sharing it either recomputes gate/up for every
down-row tile or writes partial outputs many times. The recompute lower bound is
orders of magnitude slower than the materialized point model, and partial output
traffic reaches hundreds of MiB per layer before counting Q3_K work. Do not code
a whole-FFN HIP runtime route unless a new design first explains how it shares
hidden tiles across down-row tiles without global hidden materialization,
multi-pass output traffic, or massive gate/up recomputation.

## D026 Persistent Q3_K Layout Memory Gate

Probe: add `scripts/research/rocm_q3_layout_memory_gate.py` and run it against
`models/Qwen3.6-27B-Q3_K_S.gguf` to quantify persistent/predecoded Q3_K layout
ideas before coding a new HIP path. This is a residency gate, not a TPS claim.

Artifact:

- `build_logs/agent-workload/d026-rocm-q3-layout-memory-gate.md`

Actual model Q3_K footprint:

| Group | Tensors | Elements | GGUF GiB | Runtime padded GiB |
| --- | ---: | ---: | ---: | ---: |
| FFN Q3_K | `192` | `17,112,760,320` | `6.848` | `6.973` |
| Other Q3_K | `161` | `6,975,651,840` | `2.792` | `2.842` |
| All Q3_K | `353` | `24,088,412,160` | `9.640` | `9.815` |

Persistent layout expansion versus runtime padded Q3_K:

| Layout | FFN-only extra | All-Q3 extra | Expansion |
| --- | ---: | ---: | ---: |
| Compact signed-nibble + int8 scales, raw `146 B/block` | `+2.117 GiB` | `+2.980 GiB` | `1.304x` |
| Compact signed-nibble + int8 scales, aligned `160 B/block` | `+2.988 GiB` | `+4.206 GiB` | `1.429x` |
| MMA-ready int8 values + fp16 scales, `288 B/block` | `+10.957 GiB` | `+15.423 GiB` | `2.571x` |
| MMA-ready int8 values + fp32 scales, `320 B/block` | `+12.949 GiB` | `+18.228 GiB` | `2.857x` |

Decision: reject persistent MMA-ready expanded Q3_K as a 130k ROCm route. Even
FFN-only int8+fp16 expansion adds about `11 GiB`, which is incompatible with the
already spill-sensitive 16 GB lane unless a separate memory residency plan
exists. Compact signed-nibble variants are not fully rejected, but they still
preserve nibble unpack work and only reduce hmask/scale packing overhead; they
need a point kernel with enough local speedup to justify the extra `2.1-4.2 GiB`
before any runtime integration.

## D027 Compact Signed-Nibble Q3_K Layout Scout

Probe: add `scripts/research/rocm_q3k_compact_layout_scout.cpp` to test the only
D026 layout variant that was not immediately too large: a `160 B/block` compact
signed-nibble layout with unpacked int8 scales. The scout compares raw padded
Q3_K unpack/shared-tile work against compact layout unpack/shared-tile work, with
identical output format (`64` packed int32 q groups plus `16` float scale values
per block). This is a local unpack/tile point measurement, not a TPS claim.

Artifacts:

- `build_logs/agent-workload/rocm_q3k_compact_layout_scout.exe`
- `build_logs/agent-workload/d027-rocm-q3k-compact-layout-scout-r1.txt`
- `build_logs/agent-workload/d027-rocm-q3k-compact-layout-scout-large-r1.txt`

Results:

| Sample | Raw unpack | Compact unpack | Compact vs raw | Raw shared | Compact shared | Compact shared vs raw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `262144` blocks | `0.1975 ms` | `0.2111 ms` | `0.9353x` | `0.0793 ms` | `0.0901 ms` | `0.8810x` |
| `1048576` blocks | `0.8297 ms` | `0.9408 ms` | `0.8819x` | `0.3382 ms` | `0.4868 ms` | `0.6947x` |

Correctness:

- `qs_mismatches=0`, `df_mismatches=0`, `max_abs=0` on both measured samples
  after fixing the harness host-side scale packing to avoid cross-byte borrow.

Decision: reject compact signed-nibble persistent layout as a ROCm speed route.
It costs extra residency from D026 (`+2.99 GiB` FFN-only and `+4.21 GiB` all-Q3
for the aligned layout) and still loses in the isolated unpack/shared-tile point
test. The expected hmask/scale simplification is not enough to overcome the
larger global footprint or simpler raw padded path. Do not promote compact
signed-nibble Q3_K without a different compute body that uses the layout for
more than unpack simplification.

## Required Scout

The next full body scout should be a new mode or sibling of
`scripts/research/rocm_q3flashmatmul_scout.cpp`, not a runtime patch. It must:

- compare against the current P002 MMQ/direct timing for the same shape; if the
  scout intentionally switches back to a library-GEMM staging route, also compare
  against in-process `Q3_K -> fp16 -> rocBLAS`;
- report correctness (`max_abs`, `rmse`) before timing is trusted;
- measure at least one P002 `n=128` hot shape and one small correctness shape;
- report local speedup and Amdahl projection with the touched-route share;
- fail closed if output differs or if point timing is below the gate.

The candidate body must change the dataflow relative to E245. Acceptable first
directions are:

- persistent Q3_K tile dequant with enough N reuse to avoid repeating A unpack;
- tile-major schedule that keeps Q3_K decode close to WMMA/MFMA feed without a
  large fp16 global staging buffer;
- a split body where only a bounded tile is dequantized and consumed before it
  reaches global memory;
- FFN-level prefill fusion only if it first solves D025's hidden-sharing problem
  without global hidden materialization, multi-pass output traffic, or massive
  gate/up recomputation.
- Q3_K layout work only if it changes the compute body beyond D027's compact
  unpack simplification and first proves a local win large enough to offset the
  D026 residency cost.

Not acceptable as D002 first code:

- direct scalar dot loops;
- wider-N scalar direct Q3Flash tile variants without a matrix-core or FFN-level
  dataflow change;
- P4-style per-block Q3Flash/WMMA tile reuse unless the standalone point result
  first beats the `Q3_K -> fp16 -> rocBLAS` baseline;
- pair-only gate/up+SwiGLU shared-B WMMA without a down-projection streaming
  mechanism, because D024 lost to the paired rocBLAS+SwiGLU point baseline;
- naive whole-FFN streaming that recomputes gate/up per down-row tile or writes
  global partial outputs for every hidden tile, because D025 fails the resource
  model before code;
- persistent MMA-ready expanded Q3_K layouts, because D026 shows `+11 GiB`
  FFN-only and `+15 GiB` all-Q3 extra residency for int8+fp16 expansion;
- compact signed-nibble persistent Q3_K layout used only to simplify unpack,
  because D027 is slower than raw padded Q3_K in both global and shared-tile
  unpack tests while adding residency;
- another current-MMQ selector override;
- another library grouped/batched GEMM wrapper;
- broad runtime route before standalone point proof;
- a persistent full-weight fp16 cache.

## Promotion Rules

Strong pass:

- correctness clean on small and hot shapes;
- `>=1.15x` point speedup on a top P002 ROCm Q3_K shape;
- Amdahl projection `>=2%` on the 130k ROCm quick lane;
- no extra persistent VRAM allocation that breaks the 130k residency envelope.

Weak pass:

- `1.05x-1.15x` point speedup with a clear stack path. Keep as a scout result,
  do not integrate runtime yet.

Reject:

- any correctness mismatch;
- point timing below rocBLAS baseline;
- route that needs broad fp16 residency or unsafe async lifetime to win;
- runtime speed claim without a same-lane cold/no-reuse/no-prime A/B.

## First Commands

After ROCm route trace is refreshed, model the coding threshold:

```bash
python scripts/research/rocm_lowlevel_route_gate.py \
  --baseline-tps 1.5200 \
  --target-tps 2.0 \
  --shares 0.47,0.50,0.60,0.70,0.80 \
  --local-speedups 1.10,1.20,1.30,1.40,1.50,2.00
```

Then write the HIP scout only for the highest-share P002 ROCm Q3_K shape.

## Decision

Open D002 as the low-level-language experiment, but keep it gated. The next
useful step is a standalone HIP/topology scout that is materially different from
the rejected current-MMQ loader/barrier variants and the mixed D017 Q3Flash
active-shape route. Do not rewrite broader runtime code until the scout beats the
current same-lane point timing and projects to a real 130k lane gain.
