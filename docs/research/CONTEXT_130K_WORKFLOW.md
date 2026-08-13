# 130k Context Workflow

Status update (2026-07-20): D089 promotes Qwen3.6-27B Q4_K_M and the safe
dual-ROCm 49K lane to the project baseline. This document remains the
model-scoped Q3_K_S 130K workflow and a residency-stress reference; it is no
longer the generic project entry lane.

Closure update (2026-08-13): D002 and D028 are closed and P002 is parked.
Commands and targets below are historical/reopen-only; D096 owns the active
primary FP8/platform-tail queue.

This was the active dense Qwen performance lane as of 2026-05-27.

Important correction: `ctx=131072` only sets capacity. It does not mean the
benchmark actually filled the prompt close to 130k. The D012/D005 quick rows
below used only about `8k` prompt tokens, so they are valid route/config smoke
and regression checks, but they are not sufficient as the practical long-prompt
baseline for live agent work.

## Lane Contract

- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- Context: `ctx=131072` (~130k)
- Backend baselines: Vulkan and ROCm, measured separately and sequentially
- Quick smoke workload: `quick`, `triage_diff`, `max_tokens=16`
- Quick smoke real context: `--real-context-mode repo-snapshot --real-context-chars 24576` (current snapshot: `23531` chars, `7904` prompt tokens)
- Headline practical workload: same `ctx=131072`, but with a real large prompt around `57k-62k` prompt tokens (`--real-context-chars≈152000` in the repo-snapshot harness, or equivalent live GUI/API requests)
- Cache/reuse: cold-first only, `--no-reuse --no-v2-prime-pass`
- Thinking: enabled, `--no-disable-thinking`
- KV: `q4_0/q4_0`
- Starting shape: `batch=512`; Vulkan achieved best `ubatch=256`, ROCm current baseline `ubatch=128`
- Practical constraint when this lane is reopened: do not run `ubatch > 256`
  because VRAM headroom is already limited.
- Speculation: off, `--spec-type none`
- Vulkan residency knob: `--no-mmap` is part of the historical D005/D012 lane; `mmap=true` can fall into a severe 130k prefill slow path.
- Historical target contract: D012 solved the quick-smoke `2 TPS` target; any
  explicit reopen must report the real-big-prompt prompt-eval/decode split and
  first clear the D027/D033 rejection fences.

## Quick Smoke Baseline

Measured 2026-05-26 on dense `Qwen3.6-27B-Q3_K_S`, `ctx=131072`, q4_0/q4_0,
cold/no-reuse/no-prime, thinking on:

| Backend | Label | Batch/UBatch | Wall | Aggregate TPS | Prompt tok/s | Decode tok/s | Prompt tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vulkan achieved target | `d012-vulkan-130k-glu-fast-q3quad-bn256-lowtile3-confirm3` | `512/256` | `~7.99s` | `2.0013` | `1053.11` | `42.72` | `7970` |
| Vulkan D005 anchor | `d005-vulkan-default-splitk-confirm3` | `512/256` | `~8.94s` | `1.7898` | `934.81` | `43.59` | `7970` |
| Vulkan pre-D005 control | `p002-vulkan-ub256-confirm3` | `512/256` | `~9.85s` | `1.6249` | `843.60` | `42.34` | `7947` |
| Vulkan old control | `p002-vulkan-ub128-confirm3` | `512/128` | `~10.23s` | `1.5635` | `811.02` | `42.41` | `7947` |
| ROCm | `p002-rocm-ub128-current-confirm3` | `512/128` | `~10.52s` | `1.5200` | `801.71` | `29.07` | `7970` |
| ROCm old control | `scout-rocm130k-quick-c24k-b512-ub128-r1` | `512/128` | `11.44s` | `1.3984` | `725.21` | `31.44` | `7904` |

These rows validate the command shape and route at 130k capacity, but the prompt
payload is only about `8k` tokens. They should not be quoted as the real
large-prompt user-experience baseline.

Shape evidence: Vulkan `b512/ub256` is validated on the 24k-char 130k quick
smoke lane. D005 keeps split-K 3 for the guarded Q3_K FFN-down shape, and D012
reaches the 2 TPS quick target with the q3quad/GLU opt-in stack.
The retired Vulkan target was `2.4 TPS`, which D028 models as `1.1992x` required
wall speedup over D012 and about `1277 tok/s` prompt eval if decode and overhead
stay flat. A gate/up-only FFN route is below the new bar (`1.908x` local needed);
the next candidate should touch the whole dense FFN Q3_K route (`1.387x` local
needed) or enough of all-Q3 prefill (`1.260x` local needed). D029 rejects the
obvious whole-FFN variants: activation-only fusion saves only `2.09 GiB` hidden
write/read traffic across the active prefill graph while needing about
`1166 ms` dense-FFN savings, and naive streaming is blocked by recompute or
`83.67 GiB` partial-output traffic. D030 then rejects the nearby all-Q3 old
families: extending the current q3quad/tile stack is already in D012 and still
needs about `1175 ms` more all-Q3 point savings, scale-only helpers were
negative, signed-nibble-only storage lost at runtime, Q8_1/int-dot is strongly
negative, and broad expanded layouts exceed the 130k residency budget. D031
additionally rejects compact Q3S/signed-nibble plus predecoded-scale
layout-body work: the optimistic static ceiling is far below the needed
`~1175 ms` all-Q3 point savings, runtime evidence is negative, and residency
cost is too high. The next Vulkan route must be a true Q3_K compute body or
compressed-dot route with point/resource proof. D032 allows a Q3+FA stack only
after Q3 reaches roughly `1.18-1.20x` local: FA `1.5x` still leaves `1.1987x`
Q3 local needed, and FA `2.0x` still leaves `1.1702x` Q3 local needed, so
FA-only work cannot carry the `2.4 TPS` target.
D033 rejects q3-octa/`LOAD_VEC_A=8` as a near-repeat of the measured negative
E087 family (`-1.50%`), so do not spend a shader build on wider per-invocation
Q3_K dequant unless the topology changes more than load width.
Vulkan `b512/ub64` regressed to `0.97 TPS`, `b512/ub192` to `1.4011 TPS`, and
`ub>=320` hit a severe prefill cliff (`0.30-0.36 TPS`).
`b1024/ub128` tied or slightly regressed, and the older heavy Vulkan
`b2048/ub512` still remains a rejected 32k-char scout shape. ROCm `b1024/ub256`
timed out on the 32k-char probe, while the same-lane `b512/ub128` confirm
recentered to `1.5200 TPS`; keep ROCm at `ubatch=128` until a new same-lane
route gate moves point timing first.

GUI/autotune note: the active Vulkan 130k contract is `--spec-type none --no-mmap`.
The incomplete GUI run `gui-autotune-Qwen3.6-27B-Q3_K_S-20260526-161645` used
`mmap=true`; its `b512/ub256` config fell to `188.11 prompt tok/s`, so it is a
residency mismatch, not evidence that `ub256` lost to `ub192` on the D005 lane.
The post-fix GUI-equivalent check `d005-gui-nommap-check-r1` used the D005
contract and recovered the fast path (`1.6857 TPS`, `881.26 prompt tok/s`).

D006 output/residency note: after the GPU power limit was restored to `+10%`, a
fresh full-output D005/default check still cliffed
(`d006-vulkan-130k-d005-default-powerplus-r1`: `0.3252 TPS`, `163.67 prompt
tok/s`, `41.65 decode tok/s`). Keeping the output layer off device-local VRAM
recovered prompt eval (`LLAMA_NO_OUTPUT_OFFLOAD`: `1.7551 TPS`, `955.88 prompt
tok/s`; `LLAMA_OUTPUT_HOST_GPU_DEV`: `1.7769 TPS`, `968.32 prompt tok/s`) but
decode fell to about `22 tok/s`. Treat these knobs as default-off diagnostics,
not launch defaults or a new baseline. The next Vulkan work should be a
major-topology source design with an explicit residency model, not another
ubatch/queue/output-placement sweep.

D007 source/topology gate: `GGML_VK_FFN_ROUTE_TRACE=1` now reports strict and
non-adjacent whole-FFN block coverage. On
`d007-vulkan-130k-ffn-scanblock-trace-r1`, the active graph exposed `64` Q3_K
gate/up+GLU candidates and `63` prefill candidates. Strict adjacent matching
covers only `16` prefill blocks because the other `47` have an unrelated
next-op `VIEW` (`src=NONE`, not a GLU view). The dependency scan recovers
`scan_blocks=64`, with prefill gaps `16` at `gap=3` and `47` at `gap=4`. Do not
start a naive adjacent fused whole-FFN shader; the next Vulkan gate is a
non-adjacent whole-FFN ceiling/design/correctness scout.

ROCm D002 diagnostic trace (`max_tokens=1`, not a TPS claim) shows P002
`ub=128` Q3_K routes through MMQ/direct rather than the old cublas split trace:
Q3_K MMQ is `5754.612 ms`, `94.32%` of MMQ and about `47.05%` of diagnostic
wall. The first low-level padded 32-bit load scout was correct but rejected as a
standalone runtime patch because the LDS-like mode did not show a robust gain.
Continue D002 only with a body/dataflow change, not selector forcing,
load-width-only edits, allocator/vbuffer toggles, or scalar direct Q3Flash tile widening. The cheap stream-K threshold probe
`GGML_MMQ_RDNA4_STREAM_K_MIN_NE11=1` is rejected: point-level Q3_K MMQ moved
only about `-3.17%`, and wall A/B tied/lost (`1.5196 TPS` candidate versus
`1.5206 TPS` neighbor control). D019 wider-N scalar Q3Flash improved over P0
but remained only `0.2719x` / `0.2508x` of the in-process rocBLAS point
baseline on the two P002 hot shapes. D020
`GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1` measured `1.5067 TPS`, prompt
`798.48 tok/s`, decode `28.78 tok/s`, so the allocator single-chunk control is
also not the missing route. D021 P4 multi-row WMMA Q3Flash improved the gate/up
direct-WMMA point from `0.4215x` to `0.5237x` versus local rocBLAS, but still
lost badly and did not improve down/reverse (`0.4111x`). The next ROCm route
needs a much stronger compressed-GEMM topology or a broader FFN-level dataflow.
D022 upstream-stock rollback control measured only `0.5720 TPS`, prompt
`294.40 tok/s`, decode `21.96 tok/s`, so the current fork baseline is not a
local regression relative to the imported stock build. D023 streaming
`Q3_K -> fp16 -> rocBLAS` chunk sweep got close but still lost to the local
point baseline (`0.9058x` gate/up, `0.9336x` down at chunk `8192`), so another
runtime cublas/dequant wrapper is not the next route. D024 pair-only
`gate/up + SwiGLU` shared-B WMMA scout was correctness-clean but much slower
than paired rocBLAS+SwiGLU (`0.5349x` on `17408x128x5120`, `0.3701x` on
`5120x128x17408`), so broad FFN work still needs a down-projection streaming
mechanism or a different compressed-GEMM/layout proof rather than pair-only
fusion. D025 rejects naive whole-FFN streaming before code: without a new
cross-down-row hidden-sharing mechanism, it either recomputes gate/up for every
down-row tile (`0.0175x` lower-bound speedup) or adds about `680 MiB` partial
output read/write traffic per layer at hidden tile `128`. D026 rejects
persistent MMA-ready expanded Q3_K layouts on residency: FFN-only int8+fp16
expansion adds about `+10.96 GiB`, all-Q3 adds `+15.42 GiB`; compact nibble
layouts remain only a point-kernel idea until they prove enough speedup to pay
for `+2.1-4.2 GiB` extra residency. D027 then rejects the compact nibble route
when used only for unpack simplification: the aligned `160 B/block` layout is
correctness-clean but slower than raw padded Q3_K (`0.8819x` global unpack and
`0.6947x` shared-tile unpack on `1048576` blocks).

P002 closure note: do not continue either backend by default unless the user
explicitly reopens the program and a new design first clears the D002/D013-D027
and D028-D033 fences. D012 and D028 are historical baseline/target artifacts,
not the active project queue.

## Real-Big-Prompt Lane

The practical lane is now a large prompt at `ctx=131072`, not just a large
context window. Use this lane for user-visible speed claims and optimization
decisions:

- prompt scale: about `57k-62k` prompt tokens;
- harness target: `--real-context-mode repo-snapshot --real-context-chars 152000`;
- live GUI/API target: requests whose server log reports `task.n_tokens` or
	`prompt_tokens` around `60k`;
- expected current symptom under reduced GPU power limit: prompt eval can be
	around `~200 tok/s`, and decode can fall to roughly `~15 tok/s`; treat up to
	about `10%` baseline loss from the power limit as normal when comparing runs;
- record whether prompt cache/checkpoints are enabled. They are a practical
	session feature, but first-request and cold/no-reuse claims must be labeled
	separately.

Do not compare a candidate measured on the `8k` quick smoke lane against a
`60k` prompt run. They stress different residency and RAM/PCIe behavior.

Practical big-prompt backend checkpoint (`real-context-chars=152000`,
`task_prompt_tokens=56425`, repeated/steady with reuse on):

| Backend | Label | Batch/UBatch | Aggregate TPS | Prompt tok/s | Decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Vulkan | `p002-vulkan130k-big-c152k-b512-ub256-r1` | `512/256` | `0.1758` | `626.06` | `21.60` |
| ROCm | `p002-rocm130k-big-c152k-b512-ub128-r1` | `512/128` | `0.1023` | `363.81` | `13.82` |

For this practical checkpoint, Vulkan leads ROCm by about `+71.85%` wall TPS
on the same prompt scale and lane contract.

D041 no-reuse control on the same practical Vulkan lane (`b512/ub256`,
`q4_0/q4_0`, `--spec-type none --no-mmap`) shows checkpoint removal is not the
primary limiter:

| Mode | Label | max_tokens | Aggregate TPS | Prompt tok/s | Decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Reuse on (D040) | `p002-vulkan130k-big-c152k-b512-ub256-r1` | `16` | `0.1758` | `626.06` | `21.60` |
| No-reuse (D041) | `d041-vulkan130k-big-c152k-noreuse-mt16-b512-ub256-r1` | `16` | `0.1766` | `628.88` | `21.46` |
| No-reuse decode sanity | `d041-vulkan130k-big-c152k-noreuse-mt64-b512-ub256-r1` | `64` | `0.6911` | `631.73` | `20.09` |

No-reuse vs reuse at `max_tokens=16` is effectively flat (`+0.46%` wall), and
decode stays around `~20-21 tok/s`. Next Vulkan-only work should target true
long-prompt Q3_K/FFN/body scaling, not session cache/checkpoint semantics.

Additional D042 gate (`first4` direct host-KV under the same no-reuse lane)
also stayed flat on wall and regressed decode:

- `d042-vulkan130k-big-c152k-first4-noreuse-mt16-b512-ub256-r1`: `0.1768 TPS`, prompt `629.61`, decode `20.93`
- vs D041 no-reuse last3 baseline: wall `+0.11%`, decode `-2.47%`

Treat host-KV placement sweeps as closed for this practical lane unless a new
lifetime/migration mechanism changes the expected decode tax.

D043 lowtile3 gate on the same no-reuse lane shows a small positive shift and
is currently the best practical no-reuse mt16 point in this cycle:

- `d043-vulkan130k-big-c152k-lowtile3-noreuse-mt16-b512-ub256-r1`: `0.1787 TPS`, prompt `636.09`, decode `21.53`
- vs D041 no-reuse baseline: wall `+1.19%`, prompt `+1.15%`, decode `+0.33%`

This is still far from practical targets (`prompt 900`, `decode 30`), so keep
lowtile3 as a minor knob and continue toward larger Q3_K/FFN/body route gains.

D044 attempted `ubatch=512` (`d044-vulkan130k-big-c152k-lowtile3-noreuse-mt16-b512-ub512-r1`)
but was aborted after clear severe prefill slowdown and no diagnostics output;
it is closed as out-of-lane for this practical profile.

D045 tested disabling the AMD bn256 default on top of D043 lowtile3, same lane:

- `d045-vulkan130k-big-c152k-lowtile3-nobn256-noreuse-mt16-b512-ub256-r1`:
	`0.1669 TPS`, prompt `593.85`, decode `21.66`
- vs D043: wall `-6.60%`, prompt `-6.64%`, decode `+0.60%`

Reject `GGML_VK_DISABLE_AMD_BN256_DEFAULT=1` for this lane. Keep bn256
auto-default behavior enabled.

D046 tested `batch=640` (keeping `ubatch=256` and lowtile3):

- `d046-vulkan130k-big-c152k-lowtile3-noreuse-mt16-b640-ub256-r1`:
	`0.1640 TPS`, prompt `583.47`, decode `21.92`
- vs D043: wall `-8.23%`, prompt `-8.27%`, decode `+1.81%`

Reject `batch=640` for this practical lane and keep `batch=512` as active shape
while route-level work continues.

D047 tested disabling Q3 quad dequant on the same practical lane:

- `d047-vulkan130k-big-c152k-lowtile3-noq3quad-noreuse-mt16-b512-ub256-r1`:
	`0.1748 TPS`, prompt `622.40`, decode `21.36`
- vs D043: wall `-2.18%`, prompt `-2.15%`, decode `-0.79%`

Reject `GGML_VK_Q3K_QUAD_DEQUANT=0` for this lane; keep q3quad dequant enabled.

`ubatch > 256` is now explicitly out-of-lane for this practical runbook due to
VRAM pressure. Keep follow-up candidates at `ubatch=256` or below.

`ubatch > 256` is now explicitly out-of-lane for this practical runbook due to
VRAM pressure. Keep follow-up candidates at `ubatch=256` or below.

## RAM-Spill Rule

At `ctx=131072`, RX 9070 XT 16 GB should not be assumed VRAM-resident. A large
part of KV/context/working set may live in system RAM or move over PCIe. Treat
this as the target constraint, not as a benchmark anomaly.

Every 130k run should preserve diagnostics that answer:

- did startup fit cleanly or rely on mmap/host memory behavior;
- did Vulkan `--no-mmap` or queue selection change wall/prompt/decode split;
- did ROCm allocator/residency messages change between runs;
- did prompt eval or decode eval dominate the wall;
- was any trace/selector environment accidentally left enabled.

## Baseline Tasks

Use VS Code tasks first:

- `bench: vulkan q3 130k big prompt baseline` for the practical headline lane
- `bench: vulkan q3 130k baseline` for quick route/config smoke only
- `bench: rocm q3 130k baseline`

Run them one at a time. Do not run Vulkan and ROCm 130k baselines in parallel.
The quick tasks check for an existing `llama-server`, keep the run short, and
write diagnostics. The big-prompt task is intentionally slower and should be
used for headline prompt/decode claims.

## Claim Policy

- New practical 130k speed claims compare against a same-backend real-big-prompt baseline. Quick-smoke results are route/regression evidence only unless explicitly labeled as such.
- Compare candidates only against the matching backend baseline with the same ctx, batch, ubatch, KV, task, max-token budget, real-context settings, reuse state, and thinking mode.
- Also match or disclose GPU power-limit state. During the current lowered power-limit period, about `10%` lower speed is acceptable baseline drift.
- Old `ctx=12288`, `32768`, `65536`, and sentinel `131072` rows are historical references. Tiny-prompt sentinel128 runs are not valid 130k real-context baselines.
- If a candidate changes residency knobs such as mmap, cache RAM, allocator chunking, or queue selection, report startup/residency diagnostics with the TPS result.
- Output-layer placement changes must also report decode eval separately; prompt recovery alone is not a speed claim on this lane.
