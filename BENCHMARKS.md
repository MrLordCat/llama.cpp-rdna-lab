# Benchmarks

Главный локальный benchmark для этой ветки:

```powershell
python scripts\agent_workload_bench.py
```

Он запускает короткую симуляцию агентной работы через OpenAI-compatible `llama-server`: triage diff, code review, ROCm log diagnosis и маленький patch simulation. По умолчанию инструмент ищет ROCm server binary в:

```text
build-rocm\bin\llama-server.exe
build-rocm\bin\Release\llama-server.exe
```

и модель в:

```text
models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
models\Qwen3.6-27B-Q3_K_S.gguf
models\Qwen3.5-9B-Q6_K.gguf
```

Результаты пишутся в:

```text
build_logs\agent-workload\<label>.csv
build_logs\agent-workload\<label>.jsonl
build_logs\agent-workload\<label>.server.log
```

По умолчанию runner выбирает свободный порт сам. Для уже запущенного сервера укажи `--no-start --port 8080`.

Для активной 130k performance lane runner по умолчанию держит thinking включённым. Для явного отключения thinking укажи `--disable-thinking`.

## Active 130k baseline plan (2026-05-27)

Current global performance target is dense `Qwen3.6-27B-Q3_K_S` at
`ctx=131072` (~130k), not the older 12k cold route. Vulkan reached the old
`2 TPS` target with D012; the active target is now Vulkan `2.4 TPS` on the same
lane. ROCm is paused at the D013-D027 rejection fence unless a stronger
compressed-GEMM/FFN dataflow proof appears. The measured same-lane baselines are:

- `ctx=131072`, `batch=512`, `q4_0/q4_0`, FlashAttention on. Vulkan current best uses `ubatch=256`; ROCm current baseline uses `ubatch=128`.
- `quick:triage_diff`, `max_tokens=16`, repo-snapshot real context with `--real-context-chars 24576`.
- cold-first: `--no-reuse --no-v2-prime-pass`, thinking enabled, `--spec-type none`.
- Vulkan starting route uses `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` and `--no-mmap`; ROCm uses the native HIP 7.1 `gfx1201` path.

Measured short-baseline results:

| Backend | Label | Wall | TPS | Prompt tok/s | Decode tok/s | Prompt tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Vulkan | `p002-vulkan-ub256-confirm3` | `~9.85s` | `1.6249` | `843.60` | `42.34` | `7947` |
| Vulkan current check | `p002-vulkan-ub256-current-r1` | `9.61s` | `1.6654` | `866.47` | `43.42` | `7970` |
| Vulkan current default | `d005-vulkan-default-splitk-confirm3` | `~8.94s` | `1.7898` | `934.81` | `43.59` | `7970` |
| Vulkan current opt-in stack | `d012-vulkan-130k-glu-fast-q3quad-bn256-lowtile3-confirm3` | `~7.94s` | `2.0013` | `1053.11` | `42.72` | `7970` |
| Vulkan old control | `p002-vulkan-ub128-confirm3` | `~10.23s` | `1.5635` | `811.02` | `42.41` | `7947` |
| ROCm | `p002-rocm-ub128-current-confirm3` | `~10.52s` | `1.5200` | `801.71` | `29.07` | `7970` |
| ROCm old control | `scout-rocm130k-quick-c24k-b512-ub128-r1` | `11.44s` | `1.3984` | `725.21` | `31.44` | `7904` |

Vulkan `b512/ub256` was confirmed with 3 cold/no-reuse/no-prime runs and improves
the same-lane control by about `+3.93%` (`1.5635 -> 1.6249 TPS`) through prompt
eval (`811.02 -> 843.60 tok/s`) while decode stays tied. Rejected as default
starting shapes: Vulkan `b512/ub64` (`0.9724 TPS`), `b512/ub192` (`1.4011 TPS`),
`b512/ub320` (`0.3277 TPS`), `b512/ub384` (`0.3550 TPS`), `b512/ub512`
(`0.3040 TPS`), `b1024/ub128` (`1.5703 TPS`), `threads16` (`1.5697 TPS`), old
Vulkan `b2048/ub512` (`70.92s` on the 32k-char scout), and ROCm `b1024/ub256`
timeout on the 32k-char scout. ROCm `b512/ub128` was rechecked on the current
tree and recentered to `1.5200 TPS`; do not treat the older `1.3984 TPS` scout
as the active comparator.

The signed-nibble Q3_K layout route is not kept as runtime code: all-Q3 failed
the 130k fit check, while a narrow `hot5` runtime prototype completed but
regressed (`1.5186 TPS` vs `1.5798 TPS` same-session control). Keep only the
static scout/tooling artifact unless a new zero-overhead layout mechanism appears.

D002 ROCm stream-K threshold forcing is also rejected as a speed route:
`GGML_MMQ_RDNA4_STREAM_K_MIN_NE11=1` moved traced Q3_K MMQ time only about
`-3.17%` and did not convert to wall gain (`1.5196 TPS` candidate versus
`1.5206 TPS` neighbor control). A single ROCm `ubatch=256` recheck on
`build-rocm-vec` failed before readiness with
`ggml-cuda.cu:1017 GGML_ASSERT(size % sizeof(block_q3_K) == 0)`, so ROCm stays
at `ubatch=128` until the padded-storage/view/copy correctness path is handled.
Disabling padded Q3_K storage avoided that assert but timed out on the first
`max_tokens=1` request after only `6144 / 7970` prompt tokens, so raw-storage
`ub256` is also rejected as a speed escape.

D003 Vulkan larger-ubatch recovery is not a speed route. `ub320` and `ub384`
fall into a prompt cliff under the active `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`
Vulkan lane. Existing-route fixes failed: forcing Q3_K `n>256` to the medium
pipeline and splitting logical `n>256` into `n<=256` dispatches both regressed
`ub384`. `GGML_VK_ENABLE_MEMORY_PRIORITY=1` recovered `ub320` prompt speed
(`174.21 -> 808.73 tok/s` in the max-token-1 prompt gate), but the full
`max_tokens=16` run reached only `1.5562 TPS`, below current `ub256`
`1.6654 TPS`; the same knob on `ub256` tied at `1.6646 TPS`. Keep memory
priority as diagnostic/recovery evidence only.

D004 corrected the Vulkan route ceiling helper for the active `n=256` trace.
The dense FFN route is now the clearest 2 TPS target: gate/up Q3_K is `34.91%`
of parsed prompt time, down Q3_K is `24.61%`, dense FFN gate/up+down is
`59.52%`, and all Q3_K MUL_MAT is `80.50%`. Reaching `2 TPS` requires about
`1.391x` local speedup on dense FFN, or `1.262x` local speedup on all Q3_K.
The old gate/up-only dual-A fusion idea remains too weak by itself: at
`17408x256x5120` it projects only `1.114x` wall from the model ceiling. Next
Vulkan source work should target a broader FFN/Q3_K route, not a simple
gate/up-only shader.

D005 is the first kept 130k code speedup. Vulkan now defaults Q3_K FFN-down
`m=5120,n>=128,k=17408` to split-K 3. Point timing moved dense FFN down
`2188.84 -> 1626.31 ms` (`-25.7%`) and parsed total `8893.65 -> 8245.34 ms`.
The paired wall confirmation improved `1.6679 -> 1.7898 TPS` (`+7.31%`) and
prompt eval `867.95 -> 934.81 tok/s`; decode stayed tied. `GGML_VK_Q3K_FFN_DOWN_SPLIT_K=0`
or `1` disables the new default, and values `2..8` force a split count for
future probes. Split-K 4 is rejected (`28214.79 ms` parsed trace), so do not
increase the split count blindly.

D006 checked the current Vulkan 130k residency/output branch after the GPU power
limit was restored to `+10%`. A fresh D005 default/full-output run
`d006-vulkan-130k-d005-default-powerplus-r1` still hit the residency cliff
(`0.3252 TPS`, prompt `163.67 tok/s`, decode `41.65 tok/s`, `11434.19 MiB`
Vulkan model buffer, `2` graph splits). Moving the output layer out of
device-local VRAM recovered prompt eval but cut decode roughly in half:

| Variant | Label | B/UB | TPS | Prompt tok/s | Decode tok/s | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| CPU output layer | `d006-vulkan-130k-no-output-ub512-powerplus-r1` | `512/512` | `1.7551` | `955.88` | `22.13` | diagnostic only |
| Host output, GPU device | `d006-vulkan-130k-output-host-gpudev-ub512-r1` | `512/512` | `1.7769` | `968.32` | `22.28` | diagnostic only |

Do not promote `LLAMA_NO_OUTPUT_OFFLOAD` or `LLAMA_OUTPUT_HOST_GPU_DEV` as
launch defaults. They show that the output layer participates in the 130k
residency problem, but they are not a 2 TPS path and are not better than the
D005 r3 baseline. Further Vulkan work should follow the major-topology workflow
with a source/topology design note; do not continue ubatch, queue, or output
placement sweeps as the next step.

D007 moved the Vulkan work back to source-level topology gates. A default-off
`GGML_VK_FFN_ROUTE_TRACE=1` diagnostic now checks strict and non-adjacent
whole-FFN block coverage in addition to the historical gate/up+GLU surface. The
130k trace `d007-vulkan-130k-ffn-scanblock-trace-r1` showed `64` Q3_K
gate/up+GLU candidates and `63` prefill candidates. Strict adjacent
`MUL_MAT,MUL_MAT,GLU,MUL_MAT` matching only covers `16` prefill blocks; the
other `47` have an unrelated next-op `VIEW` (`src=NONE`, not a GLU view). The
dependency scan recovers `scan_blocks=64`, with the recovered prefill blocks at
`gap=3` for `16` layers and `gap=4` for `47` layers. This is not a speed claim.
It blocks a naive adjacent whole-FFN shader but keeps a non-adjacent whole-FFN
route alive for the next ceiling/design/correctness gate.

D012 is the current target-clearing Vulkan 130k checkpoint. It stacks D005 with
the env-gated `bn256` large-matmul variant, low-tile split-K `3`, a separate
compile-time Q3_K quad-dequant pipeline, and the default GLU contiguous split
fast path. The confirmed command uses `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`,
`GGML_VK_AMD_LARGE_MATMUL_VARIANT=bn256`, `GGML_VK_QK_LOW_TILE_SPLIT_K=3`,
`GGML_VK_Q3K_QUAD_DEQUANT=1`, plus `--no-mmap`. The r3 confirmation
`d012-vulkan-130k-glu-fast-q3quad-bn256-lowtile3-confirm3` reached `2.0013 TPS`,
prompt `1053.1067 tok/s`, decode `42.7233 tok/s`, errors `0`, improving the
D005 anchor by `+11.82%` wall TPS. Treat this as an opt-in measured stack, not a
plain default Vulkan claim, until q3quad/bn256/lowtile promotion hardening is
done. Rejected neighbors: `m=10240,k=5120` q3quad inclusion (`0.3783 TPS` full
cliff), vector-return q3quad (`1.9476 TPS`), `lowtile2` r3 (`1.9926 TPS`),
`lowtile4` (`1.9877 TPS`), down split-K `6` (`1.9943 TPS`), disabling MMVQ
(`1.9805 TPS` with decode collapse), and running without graphics queue
(`~0.37 TPS` class cliff).

D028 retargets Vulkan from the solved `2 TPS` goal to `2.4 TPS` on the same D012
lane. The target gate calculates `1.1992x` required wall speedup over D012 and
about `1277.25` prompt tok/s if decode and overhead stay flat. Gate/up-only FFN
is too small for the new target (`1.908x` local needed), and down-only is worse
(`3.077x`). The next Vulkan work should target whole dense FFN (`1.387x` local)
or all-Q3 prefill (`1.260x` local), starting from
`docs/research/major-topology/D028_P002_VULKAN_2P4_TARGET_GATE.md`.

D029 closes the obvious whole-FFN follow-up before code. D007's non-adjacent
whole-block graph surface is real, but activation-only fusion can save at most
`2.09 GiB` of hidden write/read traffic across the active prefill graph while the
target needs about `1166 ms` dense-FFN point savings. Naive full streaming is
also blocked: recomputing gate/up per `64` down rows has a `222214 ms` lower
bound, while hidden-tile partial output accumulation adds `83.67 GiB` of output
traffic. Reopen whole-FFN only if it reduces Q3_K matmul work itself; otherwise
move to a broader all-Q3 body/layout proof.

D030 performs that all-Q3 proof and closes the nearby old families. The D012
all-Q3 point is `5691.67 ms`; the `2.4 TPS` target needs it around `4517.10 ms`,
or `1174.57 ms` faster. D008->D009 q3quad saved only `184.81 ms` and is already
in the baseline. Scale-only helpers were negative, signed-nibble-only storage
lost at runtime (`1.5186` vs `1.5798 TPS`) and failed broad 130k fit, Q8_1/int-dot
was strongly negative, and expanded layouts are too large for 16 GB. The next
candidate needs a new Q3_K compute body or layout-body pair, not another
storage-only/tile-neighbor probe.

D031 checks compact Q3S/signed-nibble plus predecoded-scale layout-body work and
rejects it as the next `2.4 TPS` route. The old S001 static scout reduced op
count only `1491 -> 1375`; even an optimistic linear all-Q3 upper bound is about
`442.87 ms`, far below the required `1174.57 ms`. The runtime signed-nibble hot5
probe was negative (`1.5186` vs `1.5798 TPS`), and D026 residency cost is still
`+2.980 GiB` all-Q3 raw or `+4.206 GiB` aligned. D032 must be a true Q3_K compute
body or compressed-dot route that reduces matrix work itself.

D032 quantifies Q3+FA stacking. The D010 full-trace FA point is only `693.77 ms`,
so FA-only cannot carry `2.4 TPS`: even FA `1.5x` leaves `943.31 ms` Q3 savings
needed (`1.1987x` local), and FA `2.0x` leaves `827.68 ms` Q3 savings needed
(`1.1702x` local). A stack is plausible only after a true Q3_K body reaches
about `1.18-1.20x` local in point/static evidence.

D033 rejects a q3-octa/`LOAD_VEC_A=8` successor before build. The prebuild gate
matched E087 corrected Q3_K `LOAD_VEC_A=8`, which measured `-1.50%` (`947.44`
pp7488 r1 vs E086 `961.82`), and the `+20%` estimate still projected below the
target-closing gate. Wider per-invocation dequant is not the next body route.

D034 rechecked the current 130k Vulkan slow pocket. Fresh D012 full-server
controls in this session fell to `~0.36-0.39 TPS`, but direct `llama-bench`
remained fast (`pp4096 1066.39 tok/s`) and the same full-server shape at
`ctx=65536` reached `1.9212 TPS`, so the blocker is 130k server residency/
paging rather than raw Q3_K shader selection. Memory-priority/pageable probes
did not recover. Backend-host KV placement recovered prompt but paid decode back:
the best diagnostic stack, `d034-vulkan-130k-kvhost14-fulltile-lowtile2-ub512-r1`,
reached `1.9826 TPS`, prompt `1078.72 tok/s`, decode `36.98 tok/s`, still below
D012 `2.0013 TPS` and far below the `2.4 TPS` target. A Q3_K aligned full-tile
store prototype improved direct `pp4096` only `1066.39 -> 1085.72 tok/s` and was
below the prebuild gate. D034 code prototypes were reverted; keep the artifacts
as residency evidence only and do not use the `0.37 TPS` slow-pocket controls as
a baseline for speed claims.

GUI/autotune note: the incomplete run
`gui-autotune-Qwen3.6-27B-Q3_K_S-20260526-161645` is not a valid `ub192` vs
`ub256` comparison for D005 because it launched with `mmap = true`. Its
`b512/ub256` config fell to `188.11` prompt tok/s, while the D005 lane uses
`--no-mmap` and confirms `934.81` prompt tok/s. GUI 130k Vulkan bench/autotune
now injects `--no-mmap` so the UI follows the active lane contract. A
post-fix GUI-equivalent check `d005-gui-nommap-check-r1` returned to the fast
path (`1.6857 TPS`, `881.26` prompt tok/s, `40.86` decode tok/s, `7983` prompt
tokens), so the prior `ub256` slowdown was a residency/mmap mismatch.

At 130k, RX 9070 XT 16 GB is expected to spill KV/context/working set into
system RAM. Baseline notes must preserve diagnostics/server logs and report
startup/residency behavior alongside TPS. Old `ctx=12288`, `32768`, `65536`, and
sentinel `131072` rows are historical references only; especially tiny-prompt
sentinel128 runs are not valid 130k real-context baselines.

## Archived ROCm 10 TPS cold-target route gates (2026-05-25)

Historical short-context target: reach `10 TPS` on a cold/no-reuse 12k run. Two
high-ceiling routes were gated after E248:

Practical result: E255 confirms `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` clears the
cold 12k target with the same no-reuse benchmark contract: r3 aggregate
`22.1407 TPS`, prompt mean `2179.07 ms`, decode mean `678.62 ms`, errors `0`.
This is now available as the first-match GUI preset
`Qwen3.6-35B-A3B-UD-IQ3_XXS (RX 9070 XT cold 12k)`. It is a practical local
Qwen3.6 profile, not an apples-to-apples speedup of the dense 27B-Q3 model.

| Route | Result | Decision |
| --- | --- | --- |
| E249 `hipBLASLt` grouped FFN GEMM | Scout built, but ROCm 7.1 Windows exposed no grouped algorithms for the hot `17408x5120@n2048` pair, including f32-output and f16/fast16 variants | reject runtime integration |
| E250 local Q3 MTP GGUF | `Qwen3.6-27B-Q3_K_S_mtp.gguf` hard-timed out before prompt eval completion on exact `b6144/ub2048` control and smaller `b4096/ub512` MTP probe | reject as cold 10 TPS route |
| E251 Q4 MTP fit-auto | Full-offload Q4 MTP still failed fit; omitting `-ngl` allowed a run but only reached `1.17 TPS` | reject as pragmatic cold escape |
| E252 f32-output fast16 hipBLAS compute | Opt-in `HIPBLAS_COMPUTE_32F_FAST_16F` reached first Q3_K GEMM then failed with `CUBLAS_STATUS_NOT_SUPPORTED`; prototype knob reverted | reject compute-contract route |
| E253 E248 reuse + `batch=8192` stack | `batch=8192` control was only `7.2756 TPS`; src1 reuse collapsed to `2.4701 TPS`, and guarded revalidation still collapsed on both `8192` and `6144` | reject stack; do not recommend src1 reuse opt-in |

Conclusion: archived `hipBLASLt` grouping, local Q3/Q4 MTP detours, and the
f32-output fast16 hipBLAS compute contract are not valid paths to the `10 TPS`
cold 12k target. E253 also removes E248 src1 reuse from practical launch/autotune
recommendations. These results are historical references for the current 130k
work, not current baselines.

## ROCm 12k cold-first gate after ngram GUI profile (2026-05-25)

GUI/server, Bench/Autotune, agent autotune defaults, large-context helper
scripts, and bundled ngram presets now use the measured repeated-session profile
`ngram-mod 12/16/32`, but E227 confirms this is not a cold-first speed route.
The cold reference remains E226 same-task no-reuse r3:
`7.8890 TPS`, prompt mean `5978.04 ms`, decode `30.45 tok/s`.

| Route | Aggregate TPS | Prompt ms mean | Decode tok/s mean | Decision |
| --- | ---: | ---: | ---: | --- |
| ROCm cold baseline, `spec=none` | `7.8890` | `5978.04` | `30.45` | baseline |
| ROCm `ngram-mod 12/16/32` | `7.8987` | `5995.01` | `30.845` | cold tie |
| ROCm `ngram-mod 24/48/64` | `7.8976` | `5996.79` | `30.85` | cold tie |
| Vulkan q4/q4 `spec=none` | `6.7552` | `7854.255` | `40.56` | reject for cold 12k |
| Vulkan q4/q4 graphics queue + `--no-mmap` | `7.0101` | `7522.98` | `40.88` | still below ROCm |

Conclusion: keep `12/16/32` for repeated/session workflows where prompt reuse
exposes draftable spans, but the cold +20% target still requires structural
ROCm route work. From the E226 cold r3 baseline the +20% target is about
`9.47 TPS`.

E241 recentered the immediate current-snapshot cold control after the GUI/default
alignment: `7.6932 TPS`, prompt `6204.02 ms`, prompt eval `1207.12 tok/s`,
decode `30.74 tok/s`, errors `0`. For the next same-snapshot r1 gates, the
current +20% cold target is about `9.23 TPS`; with decode unchanged, the route
model says a prefill-only solution needs roughly `1.30x` local prefill speed.

## ROCm 12k cold Q3_K route recenter (2026-05-25)

E228 re-ran point-level traces on the current H43-default ROCm build and updated
driver. These are trace diagnostics, not wall speed claims. The Q3_K split trace
completed without errors, but sync/detail tracing lowered wall TPS to `7.0592`.

Robust Q3_K cuBLAS split, excluding one long `sum_ms >= 20` outlier:

| Scope | Calls | Total ms | src0 convert | src1 | GEMM | GEMM share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q3_K cuBLAS robust rows | `1395` | `3783.195` | `508.206` | `363.521` | `2906.403` | `76.82%` |
| `17408x5120@2048` | `378` | `1398.159` | `177.540` | `84.195` | `1131.560` | `80.9%` |
| `5120x17408@2048` | `189` | `786.054` | `89.144` | `98.414` | `598.457` | `76.1%` |

Full kernel trace agrees: `MUL_MAT/q3_K` is `4159.323 ms`, or `54.16%` of
traced total; `GATED_DELTA_NET/f32` is second at `998.759 ms` (`13.01%`).
rocBLAS logging shows the dominant f16/f16->f32 GEMM_EX shapes are emitted with
`solution_index 0`. The next cold-first ROCm candidate should therefore be a
GEMM-side route/body/solution-index probe for the dominant Q3_K families, not
another fp16 staging cache or current-MMQ selector salvage.

E229 added `scripts/research/rocm_rocblas_solution_scout.cpp` and checked exact
rocBLAS solution indices. The only confirmed local win on a large repeated
family was `(5120,2048,17408)` with solution `60017`: `3.1253 -> 2.5927 ms`
(`0.8296x`). The dominant `(17408,2048,5120)` family rejected solution override
on confirm (`3.3551 -> 3.3833 ms`), and rocBLAS metric/flag gates regressed.
Projected confirmed savings are only about `121 ms` on the trace before runtime
overhead, so no runtime solution-index route was added.

E230 checked the second hotspot, `GATED_DELTA_NET`. Larger GDN chunk sizes moved
sync point timing from `1017.705 ms` down to `813.862 ms` (`GGML_GDN_CHUNK_SIZE=4096`,
about `-20%` local), but no-trace cold wall did not convert: paired r1 control
was `7.8474 TPS`, while chunk `4096` was `7.7753 TPS` with prompt mean worsening
from `6036.77` to `6118.88 ms`. Conclusion: keep current GDN default; the chunk
route is a bottleneck-shift example, not a cold-first TPS improvement.

E242 closed another FFN library-scheduling branch before runtime integration:
two concurrent rocBLAS streams did not improve the dominant `17408x5120@n2048`
gate/up pair (`6.9594 -> 7.0088 ms`, `1.007x` slower). The `n1345` tail improved
locally (`4.4483 -> 3.8136 ms`), but that bucket is too small to justify graph
pairing complexity. Continue only with a real Q3_K body/layout/topology change.

E243 rechecked GDN block geometry after the driver/code changes. A temporary
`GGML_GDN_NUM_WARPS=2` point probe moved synchronized GDN total only
`1002.233 -> 987.097 ms` (`-1.5%` local) with trace-context wall tied
(`7.4809 -> 7.4870`). The probe was reverted and `build-rocm-vec` rebuilt.
Conclusion: GDN warp-count/block-geometry tuning is below the current cold +20%
target; keep focusing on Q3_K route body/layout/topology.

E244 checked a heavier graph-scheduling idea: combine prompt chunks for the same
hot Q3_K weight into one wider rocBLAS GEMM. Standalone scout results were real
but moderate: `17408x5120` separate `3*n2048+n1345` took `11.7008 ms` versus
`10.9411 ms` for `n=7489` (`1.0694x`), and `5120x17408` took `11.4727 ->
10.7176 ms` (`1.0705x`). The `n=6144` check removed the tail effect and still
showed only `1.0562x` / `1.0361x` on the two main full-chunk shapes. Even with
optimistic src0 conversion reuse, the route ceiling is about `1.0756x` wall
before overhead (`~8.28 TPS` from the E241 baseline, below the `9.23 TPS` +20%
target). Keep this as a possible stack component only for a residency-safe
streaming design; do not implement a naive full-prompt FFN route that keeps
hundreds of MiB of f32 intermediates resident.

## Vulkan 64k full-context rebaseline (2026-05-21)

Фокус переключён на Vulkan `ctx=65536`: пользователь заметил, что длинный контекст ощущается сильно медленнее, несмотря на быстрый Vulkan decode. Проверка была сделана через реальный `llama-server` request в `scripts\repo_snapshot_context_bench.py`, не через синтетический bench. Финальная калибровка prompt: `152000` chars, `57409` prompt tokens, `120` completion tokens, Qwen3.6-27B-Q3_K_S, q4/q4 KV, FlashAttention on, full offload, `spec=none`, thinking on, no reuse (`--cache-ram 0 --ctx-checkpoints 0`).

| Backend / route | b/ub | Wall TPS | Prompt eval TPS | Decode eval TPS | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Vulkan baseline | 2048/512 | `1.2896` | `640.63` | `36.04` | baseline |
| ROCm comparison | 2048/512 | `1.5545` | `799.09` | `22.83` | Vulkan loses full 64k wall on prefill |
| Vulkan shape | 4096/1024 | `1.3106` | `651.59` | `35.86` | `+1.63%` wall vs Vulkan baseline |
| Vulkan `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` + `--no-mmap` | 4096/1024 | `1.3375` | `665.00` | `36.54` | keep as stack |
| Vulkan confirm, graphics queue + `--no-mmap` | 8192/1024 | `1.3406` | `666.62` | `36.58` | best safe Vulkan 64k route, `+3.95%` vs baseline |
| Vulkan q8/q8 KV | 4096/1024 | `0.2008` | `96.62` | `35.53` | reject; prefill/residency collapse |
| Vulkan FA scalar code probe | 8192/1024 | `1.0526` | `520.12` | `34.19` | reject/revert; `-21.48%` |

Итог: подозрение подтвердилось частично. Vulkan остаётся намного быстрее в decode (`36.58` vs ROCm `22.83` tok/s), но на полном 64k cold-context wall проигрывает ROCm примерно `13.8%`, потому что prompt eval ниже (`666.62` vs `799.09` tok/s). Лучший безопасный no-code профиль для Vulkan 64k сейчас:

```powershell
$env:GGML_VK_ALLOW_GRAPHICS_QUEUE = "1"
# server-extra: "--spec-type none --cache-ram 0 --ctx-checkpoints 0 --no-mmap"
# batch/ubatch: 8192/1024, q4_0/q4_0, flash-attn on
```

Perf trace для Vulkan 64k показывает, почему это сдерживает TPS: `MUL_MAT q3_K` занимает `47.79%`, `FLASH_ATTN_EXT` ещё `38.03%`; вместе это около `85.8%` traced time. Значит следующий кодовый фокус не ngram/spec, а Q3_K large-prefill kernel path и q4 FlashAttention long-KV route.

E131 добавил default-off `GGML_VK_FA_ROUTE_TRACE=1` и снял живой 64k route trace. Активная FA-ветка: `flash_attn_f32_f16_aligned_f32accq4_0`, `coopmat1`, `q4_0/q4_0`, `Br=16,Bc=64,D_split=8,row_split=4`, `workgroup_size=256`, `subgroup_size=64`. Основные prefill chunks идут как `N=1024`, `KV=1024..57344`, `split_k=1`, `use_mask_opt=1`; tail chunk `N=178,KV=57600`. Две быстрые гипотезы закрыты: отключение mask-opt упало до `639.44` prompt tok/s, а forced FA f16acc на полном `max_tokens=120` дал `1.3380 TPS` против best `1.3406`. Вывод: не повторять mask/f16acc toggles; FA требует более глубокого shader/resource анализа, а Q3_K matmul остаётся равноправной первой целью.

E132 добавил FA resource fingerprint: main 64k coopmat1 route использует `98 VGPR`, `76 SGPR`, `26112 B LDS`, `0 scratch`; warmup GQA route `101/69/26112/0`. Попытка forced SHMEM staging не осталась на coopmat1: support gate откатил FA в scalar route (`Bc=32`, `192 VGPR`, `6144 B LDS`) и prompt упал до `520.18` tok/s. Значит staging не кандидат на текущем driver/resource limit; любой будущий FA-кандидат должен route trace-ом доказать, что не ушёл в scalar fallback.

E133 добавил shape-level разбор E128 perf log через `scripts/research/vulkan_perf_shape_summary.py`. По распарсенным hot rows `MUL_MAT q3_K` занимает `42684.45 ms`, `FLASH_ATTN_EXT` `33965.16 ms`, а две главные Q3_K формы дают `31628.56 ms`: `m=17408,n=1024,k=5120` (`20338.69 ms`) и `m=5120,n=1024,k=17408` (`11289.87 ms`). Значит следующий Q3_K-кандидат должен явно двигать эти feed-forward up/gate и down shapes, а следующий FA-кандидат должен показывать per-KV tail improvement (`KV=45k..57k`), не только общий prompt TPS.

E134 добавил route-ceiling gate через `scripts/research/vulkan_route_ceiling.py`. Он подтвердил, что отдельная FFN gate/up fusion-ветка не закроет 64k gap сама по себе: для этого ей нужен `2.234x` local speedup на `24.91%` parsed share. Вся Q3_K `MUL_MAT` ветка требует `1.357x`, вся FA ветка `1.494x`, а общий Q3_K+FA core требует только `1.172x` local speedup. Вывод: следующий Vulkan 64k план должен быть комплексным route stack, начиная с Q3_K large-prefill route detection/resource proof и затем FA long-KV redesign, а не новым одиночным флагом.

E135 добавил default-off `GGML_VK_FFN_ROUTE_TRACE=1` и проверил dense FFN graph hook на реальном 64k `llama-server` прогоне (`max_tokens=1`, no reuse). Trace подтвердил, что активный prefill graph содержит `63 x q3_K SWIGLU` кандидатов формы `m=17408,n=1024,k=5120`, то есть Vulkan может матчить цельную ветку `MUL_MAT + MUL_MAT + GLU` для gate/up FFN. Это не speed claim: следующий gate — resource proof для dual-A/same-B Q3_K SwiGLU или переход к Q3_K repack/layout, если регистры/coopmat не проходят.

E136 добавил модель `scripts/research/vulkan_ffn_route_model.py`. Она показывает, что dual-A/same-B FFN fusion без уменьшения A-side Q3_K work не закрывает 64k gap: для base tile dual-A LDS `29696 B`, accumulators `16 -> 32`, local ceiling с A proxy `1.417x`, projected wall `1.4466 TPS` против ROCm target `1.5545`. Вывод: FFN fusion остаётся возможным stack-компонентом, но следующий соло Q3_K маршрут должен уменьшать повторный A-dequant across N-blocks или идти в Q3_K repack/layout; второй главный путь остаётся FA long-KV.

Артефакты:
- `build_logs/agent-workload/e128-vulkan64k-c152k-b2048-ub512-q4-none-noreuse-repo-summary.md`
- `build_logs/agent-workload/e128-rocm64k-c152k-b2048-ub512-q4-none-noreuse-repo-summary.md`
- `build_logs/agent-workload/e128-vulkan64k-c152k-b8192-ub1024-q4-graphicsq-nommap-confirm-none-noreuse-repo-summary.md`
- `build_logs/agent-workload/e128-vulkan64k-c152k-b4096-ub1024-q4-perf1-ctx64k.server.log`
- `build_logs/agent-workload/e128-vulkan64k-c152k-b4096-ub1024-q4-trace8-ctx64k.server.log`
- `build_logs/agent-workload/e131-vulkan64k-fa-route-trace-c152k-b8192-ub1024-q4-ctx64k.server.log`
- `build_logs/agent-workload/e131-vulkan64k-fa-force-f16acc-c152k-b8192-ub1024-q4-confirm120-repo-summary.md`
- `build_logs/agent-workload/e132-vulkan64k-fa-pipeline-stats-c152k-b8192-ub1024-q4-ctx64k.server.log`
- `build_logs/agent-workload/e132-vulkan64k-fa-shmem-staging-c152k-b8192-ub1024-q4-screen-repo-summary.md`
- `build_logs/agent-workload/e133-vulkan64k-perf-shape-summary.md`
- `build_logs/agent-workload/e134-vulkan64k-route-ceiling.md`
- `build_logs/agent-workload/e134-vulkan64k-ffn-route-trace-repo-summary.md`
- `build_logs/agent-workload/e136-vulkan64k-ffn-route-model.md`
- `docs/research/experiments/E128_vulkan64k_context_rebaseline.md`
- `docs/research/experiments/E131_vulkan64k_fa_route_trace_and_gates.md`
- `docs/research/experiments/E132_vulkan64k_fa_resource_and_shmem_gate.md`
- `docs/research/experiments/E133_vulkan64k_perf_shape_summary.md`
- `docs/research/experiments/E134_vulkan64k_complex_route_gate.md`
- `docs/research/experiments/E135_vulkan64k_ffn_route_trace.md`
- `docs/research/experiments/E136_vulkan64k_ffn_route_model.md`

## ROCm decode parity first code win (2026-05-22)

Фокус H39: догонять Vulkan decode на ROCm для Qwen3.6-27B-Q3_K_S, не смешивая
это с 64k Vulkan prefill lane. После аудита E149/E150 первый сохранённый кодовый
win: RDNA4 `Q3_K/ncols_dst=1` в MMVQ использует `nwarps=2`.

Lane: `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on,
`--spec-type none`, no reuse, thinking on, `max_tokens=128`, real
`llama-server` via `scripts\agent_workload_bench.py`.

| Route | Aggregate TPS | Decode eval TPS | Prompt eval TPS | Result |
| --- | ---: | ---: | ---: | --- |
| Clean post-rebuild r3 | `28.1123` | `29.77` | `711.73` | baseline |
| RDNA4 Q3_K MMVQ `nwarps=2` r3 | `30.3145` | `32.2467` | `713.8533` | keep, decode `+8.32%` |

Live server sanity was also run with `max_tokens=64`: errors `0`, decode
`30.55 tok/s`, response preview was normal `Thinking Process:` text without
repeated-symbol corruption. Remaining E116 Vulkan q4 comparator is still around
`40.8683 tok/s`, so ROCm decode parity remains open at about `1.27x`.

### E201: Q3_K padded storage opt-in follow-up (2026-05-24)

H43 moved from pure correctness to a small measured opt-in signal. `GGML_CUDA_Q3K_PADDED_STORAGE=1`
stores non-split Q3_K tensors as backend-private `112`-byte blocks, uses padded-aware
dequant/cublas and dense MMVQ accessors, disables MMQ under the knob, and fail-closes
Q3_K `CPY` until partial/raw-copy semantics are covered. Default behavior is unchanged.

Same lane as above except `max_tokens=120`:

| Route | Runs | Aggregate TPS | Prompt eval TPS | Decode eval TPS | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| control | 3 | `12.0761` | `1259.06` | `29.9933` | baseline |
| `GGML_CUDA_Q3K_PADDED_STORAGE=1` | 3 | `12.1572` | `1260.0033` | `30.4333` | opt-in only, wall `+0.67%`, decode `+1.47%` |

Trace point after restoring padded-aware MMVQ recovered the earlier decode collapse and moved
the hot cublas bucket `17408x5120@2048` from `1424.346` to `1393.866 ms`.
With `GGML_CUDA_DISABLE_GRAPHS=1`, MMVQ point timing also showed a real local Q3_K decode
gain: the fused `nx=17408` bucket moved `76.090 -> 72.740 ms`, while Q4_K/Q6_K buckets
were unchanged/noise.
Do not promote this as default yet: split buffers, partial views, MMQ/prefill, and MoE are
not covered. Treat it as a valid H43 foundation for the next structural route-body slice.

Follow-up E201-P2a adds a second opt-in gate, `GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1`.
With both storage gates enabled, Q3_K MMQ reads the padded physical blocks directly instead
of falling back to cuBLAS on small-prompt MMQ shapes. Default behavior remains unchanged.

Point gate on a short real-server lane (`quick/triage_diff`, no real-context, `max_tokens=32`,
`GGML_CUDA_DISABLE_GRAPHS=1`) moved Q3_K MMQ `ncols_max=159/mmq_x=80` from
`252.526` to `231.453 ms` (`+8.34%` local), with route proof `q3k_padded=1`.

No-trace wall confirmation on the decode-biased short lane (`max_tokens=256`):

| Route | Runs | Aggregate TPS | Prompt eval TPS | Decode eval TPS | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| control | 3 | `30.2390` | `708.8467` | `31.1767` | baseline |
| padded storage + MMQ | 3 | `30.9884` | `735.3667` | `31.9167` | opt-in only, wall `+2.48%`, decode `+2.37%` |

Active prompt-heavy sanity (`repo-snapshot`, `7422` prompt tokens, `max_tokens=120`) also
stayed positive in r1: `11.8483 -> 12.0795 TPS`, prompt `1209.88 -> 1232.65 tok/s`,
decode `30.41 -> 30.93 tok/s`. Candidate JSONL response previews were normal
`Thinking Process:` text, with `errors=0`.

This is the first H43 route slice with a confirmed short-lane wall win, but it is still
not default-ready: split buffers, partial views, broader storage/copy semantics, and MoE
remain uncovered, and the large `ncols=2048` prefill cublas body still needs H42-style
route-body work.

Artifacts:
- `docs/research/experiments/E149_rocm_decode_parity_audit.md`
- `docs/research/experiments/E150_rocm_decode_fusion_gate.md`
- `docs/research/experiments/E151_rocm_q3k_mmvq_warps2_decode.md`
- `docs/research/experiments/E152_rocm_poste151_residual_trace.md`
- `build_logs/agent-workload/e151-rocm-decode-q4-cleanpost-r3.diagnostics.md`
- `build_logs/agent-workload/e151-rocm-decode-q4-q3warps2-r3.diagnostics.md`
- `build_logs/agent-workload/e151-rocm-decode-q4-q3warps2-live-sanity-r1.diagnostics.md`
- `build_logs/agent-workload/e152-rocm-vulkan-decode-route-delta-q3k.md`
- `docs/research/experiments/E201_rocm_q3k_padded_storage_p1.md`
- `build_logs/agent-workload/e201-rocm-q3k-padded-storage-p1.md`
- `build_logs/agent-workload/e201-rocm-q3k-padded-storage-mmq-p2.md`

### E223: Q3_K padded HIP default rollout (2026-05-24)

После E219-E222 safety/hardening H43 переведён в HIP default-on с явным opt-out:

- default (без env): `GGML_CUDA_Q3K_PADDED_STORAGE=1`, `GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1` на HIP;
- fallback: `GGML_CUDA_Q3K_PADDED_STORAGE=0 GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=0`.

Correctness gate (`test-backend-ops`, Q3_K `MUL_MAT` + `MUL_MAT_ID`) прошёл во всех трёх режимах:

- no-env default: `13/13`;
- explicit-off: `13/13`;
- explicit-on: `13/13`.

Контрольные A/B бенчи (runs=1) после rollout:

| Lane | Control (explicit-off) | Candidate (default/no-env) | Delta |
| --- | ---: | ---: | ---: |
| 12k prompt-heavy (`ctx=12288,b=4096,ub=1024`, `quick/triage_diff`, `max_tokens=64`) | `7.20` | `7.25` | `+0.69%` |
| 32k control (`ctx=32768,b=5120,ub=1024`, `v2-mini/v2_write_function`, `max_tokens=120`) | `11.03` | `11.07` | `+0.36%` |

Решение: keep default-on rollout для HIP с сохранением явного opt-out env. Это закрывает H43 default-policy цель; дальнейший speed-up трек для больших prefill shape остаётся в H42.

Артефакты:
- `docs/research/experiments/E223_rocm_q3k_padded_default_on_rollout.md`
- `build_logs/agent-workload/e223-rocm-q3k-noenv-broad-smoke.txt`
- `build_logs/agent-workload/e223-rocm-q3k-explicit-off-broad-smoke.txt`
- `build_logs/agent-workload/e223-rocm-q3k-explicit-on-broad-smoke.txt`
- `build_logs/agent-workload/e223-rocm12k-defaultoff-control-r1.diagnostics.md`
- `build_logs/agent-workload/e223-rocm12k-defaulton-candidate-r1.diagnostics.md`
- `build_logs/agent-workload/e223-rocm32k-defaultoff-control-r1.diagnostics.md`
- `build_logs/agent-workload/e223-rocm32k-defaulton-candidate-r1.diagnostics.md`

### E226: ROCm repeated/session route after H43 default-on (2026-05-25)

This is a practical GUI/agent-session result, not a cold-first kernel claim. Same
ROCm lane as E224/E223: `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV,
FlashAttention on, thinking on, `max_tokens=64`, real repo-snapshot prompt,
`build-rocm-vec/bin/llama-server.exe`.

| Route | Runs | Aggregate TPS | Prompt eval mean | Decode eval mean | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| cold-control, reuse off, `spec=none` | 3 | `7.8890` | `5978.04 ms` | `30.45 tok/s` | baseline |
| prompt cache/checkpoints, `spec=none` | 3 | `13.5774` | `2590.05 ms` | `30.605 tok/s` | `+72.11%` vs cold |
| prompt cache/checkpoints + `ngram-mod 12/16/32` | 3 | `14.1202` | `2606.59 ms` | `35.0967 tok/s` | keep, `+78.99%` vs cold, `+4.00%` vs reuse-only |

The main win is checkpoint restore of the shared repo prompt prefix: after the
first request, the server restored a `5437`-token context checkpoint and reduced
repeated prompt work to about `2033-2052` tokens. `ngram-mod` added a smaller
decode-side lift after reuse exposed draftable repeated spans; final stats were
`196` generated draft tokens and `155` accepted tokens (`0.7908` local
acceptance). Two accidentally parallel control runs timed out and are marked
invalid in E226; heavy ROCm real-server benchmarks must be run sequentially.

Artifacts:
- `docs/research/experiments/E226_rocm_session_reuse_post_h43.md`
- `build_logs/agent-workload/e226-rocm12k-cold-two-task-specnone-r3-seq.diagnostics.md`
- `build_logs/agent-workload/e226-rocm12k-session-reuse-specnone-r3.diagnostics.md`
- `build_logs/agent-workload/e226-rocm12k-session-reuse-ngram12-16-32-r3-seq.diagnostics.md`

## H03 ngram+MTP chain smoke (2026-05-19)

Добавлен экспериментальный `--spec-type ngram-mtp`: в одном server run сначала пробуется `ngram-mod`, затем MTP fallback. Это opt-in режим для проверки совместимости двух speculative источников, не новый default.

Мини-smoke: `Qwen3.6-27B-Q4_K_S`, `build-rocm-vec/bin/llama-server.exe`, `ctx=12288`, `b=4096`, `ub=512`, `q4_0/q4_0`, `max_tokens=64`, `runs=1`, no reuse, no v2 prime, thinking on.

| Spec mode | Aggregate TPS | Draft stats |
| --- | ---: | --- |
| `ngram-mod` | `10.91` | ngram generated `0` draft tokens on `triage_diff` |
| `mtp` | `13.53` | MTP accepted `46/48`, acceptance `0.95833` |
| `ngram-mtp` | `13.54` | ngram generated `0`; MTP fallback accepted `46/48`, acceptance `0.95833` |

Итог: режим жизнеспособен и корректно включает MTP fallback после ngram miss. На этом prompt ngram coverage нулевая, поэтому результат равен чистому MTP в пределах шума. Оставляем как experimental opt-in и ищем ngram-friendly/longer-session проверку перед любым default claim.

Артефакты:
- `build_logs/agent-workload/hybrid-spec-ngram-mtp-fallback-smoke-autotune-summary.csv`
- `build_logs/agent-workload/hybrid-spec-ngram-mtp-fallback-smoke-cfg03.server.log`
- `docs/research/experiments/E060_H03_ngram_mtp_chain_smoke.md`

## Vulkan vs ROCm mini A/B (2026-05-19)

Собран Vulkan server: `build-vulkan/bin/llama-server.exe`, Release/Ninja, `GGML_VULKAN=ON`, Vulkan SDK `1.4.313.1`, AMD driver API `1.4.344`. Важно для запуска этой MinGW-сборки: `C:\Strawberry\c\bin` должен быть в `PATH` перед MSYS2 `/mingw64/bin`, иначе Windows может подхватить несовместимые runtime DLL и `llama-server.exe --help` завершается кодом `127`.

Оба backend прошли full-offload sanity: `65/65` слоёв на GPU, `q4_0/q4_0`, `flash-attn=on`, `spec=none`, no reuse, thinking on, `ctx=12288`, `b=4096`, `ub=512`, модель `Qwen3.6-27B-Q3_K_S.gguf`.

### Prompt-heavy mini (`repo-snapshot`)

`tasks=quick`, `task=triage_diff`, `runs=1`, 7489 prompt tokens, 64 generated tokens.

| Backend | Wall TPS | Prompt eval TPS | Decode eval TPS | Result |
| --- | ---: | ---: | ---: | --- |
| ROCm `build-rocm-vec` | `6.3327` | `960.26` | `28.32` | baseline |
| Vulkan `build-vulkan` | `4.2206` | `573.93` | `30.85` | `-33.4%` wall; decode `+8.9%` |

Итог для текущего target lane: Vulkan не заменяет ROCm. Decode чуть быстрее, но prefill сильно медленнее, а cold prompt-heavy wall time проигрывает.

### Decode-biased sanity

Тот же task без `repo-snapshot`, 159 prompt tokens, 128 generated tokens.

| Backend | Wall TPS | Prompt eval TPS | Decode eval TPS | Result |
| --- | ---: | ---: | ---: | --- |
| ROCm `build-rocm-vec` | `27.9781` | `776.05` | `29.42` | baseline |
| Vulkan `build-vulkan` | `35.2850` | `518.83` | `38.81` | `+26.1%` wall; decode `+31.9%` |

Итог: наблюдение “Vulkan быстрее” подтверждается для decode-heavy формы, но не для нашего активного prompt-heavy сценария. Оставляем ROCm default, Vulkan использовать как opt-in/backend comparison для decode-heavy профилей.

Артефакты:
- `build_logs/agent-workload/e061-rocm-mini-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e061-vulkan-mini-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e061-rocm-decode-mini-q3ks.diagnostics.md`
- `build_logs/agent-workload/e061-vulkan-decode-mini-q3ks.diagnostics.md`
- `docs/research/experiments/E061_vulkan_rocm_mini_ab.md`

## Vulkan prefill research follow-up (2026-05-19)

Почему Vulkan медленнее на prompt-heavy lane: это не общий проигрыш backend, а конкретно prefill/K-quant path. Локальный Vulkan на Windows AMD proprietary driver показывает сильный decode, но prompt eval остаётся ниже ROCm. В коде Vulkan large matmul tile для AMD proprietary driver отключён, `Q3_K/Q6_K` MMVQ выключен из-за 2-byte alignment concerns, а upstream сейчас обсуждает сразу несколько repack/transpose/alignment PR.

Ключевые external leads:
- `ggml-org/llama.cpp#20934`: внешний RX 7900 XTX отчёт совпадает с нашей картиной - Vulkan быстрее в tg, ROCm быстрее в pp.
- `ggml-org/llama.cpp#22970`: open PR transposes K-quant A-matrix layout; reported RDNA4 prompt gains `+4%..+11%`, Q6_K microshape `+15.2%`.
- `ggml-org/llama.cpp#22951` и `#21024`: Q3_K/Q6_K alignment/repack work; потенциально важно для `Q3_K_S`, но результаты зависят от устройства.
- `ggml-org/llama.cpp#23106`: large `MUL_MAT_ID` tile на AMD был отключён намеренно из-за regression risk; это не главный dense prefill path.

Локальный no-code A/B на том же prompt-heavy task (`triage_diff`, `repo-snapshot`, 7489 prompt tokens, 64 generated):

| Backend / env | Wall TPS | Prompt eval TPS | Decode eval TPS | Итог |
| --- | ---: | ---: | ---: | --- |
| ROCm E061 baseline | `6.3327` | `960.26` | `28.32` | baseline |
| Vulkan E061 initial | `4.2206` | `573.93` | `30.85` | initial |
| Vulkan default rerun | `4.5539` | `607.78` | `38.32` | same-session control |
| Vulkan `GGML_VK_FORCE_MMVQ=1` | `4.6383` | `619.79` | `38.20` | small prefill gain |
| Vulkan `GGML_VK_DISABLE_MMVQ=1` | `4.7172` | `639.81` | `35.15` | best Vulkan prompt-heavy probe |

`GGML_VK_DISABLE_MMVQ=1` даёт примерно `+3.6%` к wall против same-session Vulkan rerun и `+5.3%` к prompt eval, но всё ещё проигрывает ROCm примерно `25.5%` wall и `33.4%` prompt eval. Decode-biased sanity при этом слегка ниже default (`34.67` против `35.2850` wall TPS), поэтому это не универсальный default.

Матрица `batch/ubatch` с `GGML_VK_DISABLE_MMVQ=1` не нашла лучшего Vulkan prefill shape: `b=4096,ub=512` остался лучшим из проверенных (`pp4096=632.96`, `pp8192=609.12`).

Итог: Vulkan можно ускорить флагом для prompt-heavy opt-in сравнений, но не до уровня ROCm. Для настоящего кода следующий разумный шаг - guarded/minimal port K-quant transpose/repack/alignment идеи (`#22970` или более узкий Q3_K/Q6_K probe), с correctness test и тем же E061/E062 benchmark contract.

Артефакты:
- `build_logs/agent-workload/e062-vulkan-default-rerun-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e062-vulkan-disable-mmvq-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e062-vulkan-disable-mmvq-b-ub-matrix.md`
- `docs/research/experiments/E062_vulkan_prefill_research.md`

## Vulkan prefill code probes to ROCm level (2026-05-19)

После E062 были проверены три code-level кандидата на той же cold prompt-heavy lane (`triage_diff`, repo-snapshot, 7489 prompt tokens, 64 generated, `ctx=12288`, `q4_0/q4_0`, `flash-attn=on`, `spec=none`, no reuse, thinking on).

### E063: K-quant transpose-A

Upstream `#22970` был применён как opt-in `GGML_VK_TRANSPOSE_A=1`, затем откатан после A/B. Для текущей `Q3_K_S` модели результат отрицательный: full workload `4.3765` wall TPS против E062 best `4.7172`. Причина ожидаемая: patch в основном покрывает Q4_K/Q5_K/Q6_K transpose pipelines, а активный bottleneck здесь Q3_K.

### E064: AMD proprietary large matmul tile

Добавлен guarded knob `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`. Он включает large matmul tile path и AMD tuned `l_warptile` даже на AMD proprietary Vulkan driver, где upstream default оставляет large tile выключенным.

| Vulkan config | Runs | b/ub | Wall TPS | Prompt eval TPS | Decode eval TPS |
| --- | ---: | --- | ---: | ---: | ---: |
| E062 `GGML_VK_DISABLE_MMVQ=1` | 1 | 4096/512 | `4.7172` | `639.81` | `35.15` |
| `GGML_VK_FORCE_AMD_LARGE_MATMUL=1` | 1 | 4096/512 | `5.6963` | `786.43` | `38.30` |
| `GGML_VK_FORCE_AMD_LARGE_MATMUL=1 GGML_VK_DISABLE_MMVQ=1` | 1 | 4096/512 | `6.2619` | `885.69` | `37.10` |
| same combo | 3 | 4096/512 | `6.18` | n/a | n/a |

E064 nearly reached ROCm but did not reliably exceed it. The large-tile path stays opt-in because upstream disabled it for AMD proprietary driver regression risk on other devices.

### E065: Q3_K/Q6_K alignment plus large tile

Applied upstream `#22951`: Vulkan-specific padded device size for Q3_K/Q6_K, padded tensor offset accounting, adjusted shader layout/loads, and re-enabled Q3_K/Q6_K MMVQ eligibility. Combined with `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`, the best confirmed shape is `b=4096,ub=1024` with default MMVQ.

| Backend / config | Runs | b/ub | Wall TPS | Prompt eval TPS | Decode eval TPS | Result |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| ROCm E061 baseline | 1 | 4096/512 | `6.3327` | `960.26` | `28.32` | historical reference |
| ROCm same-session control | 3 | 4096/1024 | `7.3868` aggregate / `7.49` median | `1173.2367` | `28.62` | current fair target |
| Vulkan E061 initial | 1 | 4096/512 | `4.2206` | `573.93` | `30.85` | initial Vulkan |
| Vulkan E065 `GGML_VK_FORCE_AMD_LARGE_MATMUL=1` | 3 | 4096/1024 | `6.4180` aggregate / `6.38` median | `897.63` | `40.35` | `+1.35%` vs E061; `-13.1%` vs fresh ROCm |

Итог: E065 впервые превысил старый E061 ROCm reference и сильно поднял Vulkan относительно E064, но свежий same-session ROCm `b4096/ub1024` r3 остаётся впереди. Vulkan decode заметно быстрее (`40.35` vs `28.62`), но prompt eval всё ещё ниже ROCm (`897.63` vs `1173.2367`), поэтому активная cold prompt-heavy цель ещё не достигнута.

Рекомендуемый E065 validation profile:

```powershell
$env:GGML_VK_FORCE_AMD_LARGE_MATMUL = "1"
python scripts\agent_workload_bench.py --tasks quick --task-ids triage_diff --runs 3 --server-bin build-vulkan\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --gpu-layers=-1 --parallel 1 --ctx-size 12288 --batch-size 4096 --ubatch-size 1024 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --no-disable-thinking --max-tokens 64 --real-context-mode repo-snapshot --no-reuse --no-v2-prime-pass --server-extra "--spec-type none"
```

Следующий шаг перед promotion: дополнительные Vulkan prefill изменения, decode-biased sanity after E065, and a second prompt-heavy task. Until then, treat this as an opt-in RDNA4/Vulkan acceleration profile, not a universal Vulkan default.

### E066: chunked GATED_DELTA_NET probe

Upstream `#20377` chunked GDN idea was tested as a temporary env-gated prototype (`GGML_VK_GDN_CHUNKED=1`) on top of E065. It built and ran, but regressed the active lane, so the code was reverted.

| Vulkan config | Runs | b/ub | Wall TPS | Prompt eval TPS | Decode eval TPS | Result |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| E065 large tile + Q3_K alignment | 3 | 4096/1024 | `6.4180` | `897.63` | `40.35` | reference |
| E066 `GGML_VK_GDN_CHUNKED=1` | 1 | 4096/1024 | `5.4760` | `745.49` | `40.33` | `-14.7%`; reject/revert |

Итог: chunked GDN is not a useful lever for this Qwen3.6-27B prompt-heavy Vulkan lane right now. Keep the E065 path, continue searching in the remaining prefill bottleneck.

### E067: Q3_K packed32 matmul load probe

E067 used `GGML_VK_PERF_LOGGER=1` to profile the E065 path. The trace showed that prompt chunks are dominated by large Q3_K `MUL_MAT`, especially shapes such as `m=17408,n=1024,k=5120` and `m=5120,n=1024,k=17408`. A narrow shader probe changed the non-coopmat2 Q3_K `mul_mm_funcs.glsl` branch to use padded 32-bit loads for scales, hmask, and quants.

The cheap pp7488 gate regressed from restored E065 `875.25 tok/s` to `836.22 tok/s`, so the shader change was reverted. Wider loads were not enough to offset extra shift/register pressure.

### E068: AMD large matmul WN tile tuning reaches ROCm level

E068 kept the E064/E065 guarded large matmul path but added an experimental runtime selector:

```powershell
$env:GGML_VK_FORCE_AMD_LARGE_MATMUL = "1"
$env:GGML_VK_AMD_LARGE_MATMUL_VARIANT = "wm32-wn32"
```

This only affects the opt-in AMD large matmul path. Default Vulkan behavior is unchanged.

Key pp7488 gates (`b=4096,ub=1024`, `q4_0/q4_0`, FlashAttention on):

| Vulkan config | pp7488 tok/s |
| --- | ---: |
| restored E065 default | `875.25` |
| `block128` | `900.32` |
| `wn32` | `981.28` |
| `wn16` | `1039.53` |
| `wm32-wn32` | `1035.80` |

Confirmed active-lane result:

| Backend / config | Runs | b/ub | Wall TPS | Prompt eval TPS | Decode eval TPS | Result |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| ROCm same-session control | 3 | 4096/1024 | `7.3868` aggregate / `7.49` median | `1173.2367` | `28.62` | current fair target |
| Vulkan E065 large + Q3_K align | 3 | 4096/1024 | `6.4180` aggregate / `6.38` median | `897.63` | `40.35` | previous Vulkan best |
| Vulkan E068 `wm32-wn32` | 3 | 4096/1024 | `7.6446` aggregate / `7.58` median | `1110.0867` | `40.40` | `+3.5%` aggregate vs ROCm; `+19.1%` vs E065 |

Follow-up correction: later validation rejected the old `wm32-wn32` win as a corrupt/undercovered route, not an accepted profile. Keep this entry only as historical context; do not use `wm32-wn32` as a Vulkan opt-in baseline.

### E069: decode-focused MMVQ probe

After E068, a decode-biased run with the same `wm32-wn32` profile reached `39.1935` wall TPS and `40.75` decode eval TPS at 256 generated tokens. `GGML_VK_PERF_LOGGER=1` is too intrusive for speed claims, but it clearly places the remaining decode cost in Q3_K MMVQ, not in FlashAttention or GDN:

| Decode hot center | Approx per-token total |
| --- | ---: |
| `MUL_MAT_VEC q3_K m=17408 n=1 k=5120` | `8.7-9.1 ms` |
| `MUL_MAT_ADD MUL_MAT_VEC q3_K m=5120 n=1 k=17408` | `4.7-4.9 ms` |
| `MUL_MAT_VEC q6_K m=248320 n=1 k=5120` | `1.66-1.68 ms` |
| `GATED_DELTA_NET` | `0.32-0.34 ms` |
| `FLASH_ATTN_EXT` | `0.24-0.27 ms` |

Cheap knobs did not expose a keep candidate: `GGML_VK_FORCE_MMVQ=1` was neutral (`37.86` vs `37.76` r1), while `GGML_VK_DISABLE_MMVQ=1` and `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` regressed to `34.25` and `33.92`. Temporary code probes for large DMMV workgroups and integer K-quant rows-per-workgroup also regressed (`33.16-35.56`). A Q3_K packed32 scale-load shader rewrite was stable but only noise-positive (`37.96` r3 vs baseline `37.91`, baseline median slightly higher), so it was reverted.

Итог: pure decode has a real hotspot and therefore future potential, but E069 found no safe small implementation to keep. The next decode work should be deeper Q3_K MMVQ specialization rather than route forcing or simple scale-load repacking.

Артефакты:
- `docs/research/experiments/E063_vulkan_transpose_a_probe.md`
- `docs/research/experiments/E064_vulkan_amd_large_matmul_probe.md`
- `docs/research/experiments/E065_vulkan_q3k_alignment_rocm_level.md`
- `build_logs/agent-workload/e065-vulkan-q3k-align-large-mmvq-default-b4096-ub1024-ctx12288-q3ks-r3.diagnostics.md`
- `build_logs/agent-workload/e065-rocm-control-b4096-ub1024-ctx12288-q3ks-r3.diagnostics.md`
- `docs/research/experiments/E066_vulkan_gdn_chunked_probe.md`
- `build_logs/agent-workload/e066-vulkan-gdnchunk-large-b4096-ub1024-ctx12288-q3ks.diagnostics.md`
- `docs/research/experiments/E067_vulkan_q3k_packed32_matmul_probe.md`
- `docs/research/experiments/E068_vulkan_amd_large_matmul_tile_tuning.md`
- `build_logs/agent-workload/e068-vulkan-large-wm32-wn32-b4096-ub1024-ctx12288-q3ks-r3.diagnostics.md`
- `docs/research/experiments/E069_vulkan_decode_mmvq_probe.md`
- `build_logs/agent-workload/e069-vulkan-decode-q3scale-packed32-128-r3.diagnostics.md`

### E075: GUI 32k Vulkan env promotion

Fresh GUI autotune A/B on `Qwen3.6-27B-Q3_K_S.gguf` used the 32k lane: `ctx=32768`, `b=5120`, `ub=1024`, `q4_0/q4_0`, `spec=ngram-mod`, `repo-snapshot chars=21872`, no reuse, no v2 prime, thinking on, 120 generated tokens. The initial GUI result showed ROCm ahead because Vulkan was launched without the E068 runtime env.

| Backend / config | Aggregate TPS | Prompt eval TPS | Decode eval TPS | Result |
| --- | ---: | ---: | ---: | --- |
| ROCm GUI fresh control | `11.0606` | `1155.52` | `28.83` | current GUI control |
| Vulkan GUI baseline | `8.1791` | `659.49` | `40.05` | missing runtime env |
| Vulkan `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`, `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wm32-wn32` r1 | `12.1414` | `1121.21` | `40.01` | `+48.4%` vs Vulkan baseline; `+9.8%` vs ROCm wall |
| Vulkan same env r3 | `12.6420` | `1163.96` | `42.22` | `+54.6%` vs Vulkan baseline; `+14.3%` vs ROCm wall |

Conclusion: large-context Vulkan prefill did drop when the GUI ran default Vulkan, and the previously discovered `wm32-wn32` RDNA4/Vulkan profile restores 32k prefill to ROCm level in throughput-only benchmarks. Follow-up real generation tests rejected the profile for default GUI use: with `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wm32-wn32`, Vulkan returned all-slash output in both thinking chat and raw completion, while Vulkan without the variant, Vulkan with only `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`, and ROCm control produced normal text. GUI Vulkan server/autotune now applies only `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`; future Vulkan work should treat `wm32-wn32` as a correctness bug until logits/output equivalence is fixed.

Artifacts:
- `build_logs/agent-workload/gui-autotune-Qwen3.6-27B-Q3_K_S-20260519-233738-cfg01.server.log`
- `build_logs/agent-workload/gui-autotune-Qwen3.6-27B-Q3_K_S-20260519-233825-cfg01.server.log`
- `build_logs/agent-workload/e075-vulkan32k-gui-baseline-r1.diagnostics.md`
- `build_logs/agent-workload/e075-vulkan32k-gui-wm32wn32-r1.diagnostics.md`
- `build_logs/agent-workload/e075-vulkan32k-gui-wm32wn32-r3.diagnostics.md`
- `docs/research/experiments/E075_vulkan_32k_gui_prefill_runtime_profile.md`

### E076: valid Vulkan 32k follow-up

After E075 rejected the corrupt `wm32-wn32` profile, the next pass checked external leads and same-session no-code gates on the safe Vulkan GUI profile (`GGML_VK_FORCE_AMD_LARGE_MATMUL=1` only). Current tree already contains the useful `#23056` Q3_K/Q6_K MMVQ block-load and 32-bit subtract work, while `#22970` remains mostly Q4_K/Q5_K/Q6_K and was already rejected for this Q3_K_S lane in E063.

Same lane as E075: `ctx=32768`, `b=5120`, `ub=1024`, `q4_0/q4_0`, `spec=ngram-mod`, `repo-snapshot chars=21872`, no reuse, no v2 prime, thinking on, 120 generated tokens.

| Vulkan config | Aggregate TPS | Prompt eval TPS | Decode eval TPS | Result |
| --- | ---: | ---: | ---: | --- |
| Safe force-only base | `9.8493` | `907.02` | `32.59` | baseline |
| `wm128-wn32` | `9.6075` | `870.91` | `32.97` | reject |
| `block128-wm128` | `9.0091` | `792.98` | `33.32` | reject |
| `block128-bn64` | `7.5254` | `624.78` | `33.10` | reject |
| `GGML_VK_DISABLE_MMVQ=1` | `9.3738` | `876.90` | `29.92` | reject |
| `b4096/ub1024` | `9.8353` | `905.08` | `32.58` | tie/reject |
| `b6144/ub1024` | `9.7029` | `883.55` | `32.97` | reject |
| `q8_0/q8_0` KV | `9.1102` | `865.69` | `28.12` | reject |
| `f16/f16` KV | `8.8361` | `937.74` | `22.40` | reject: prompt up, decode down |
| Post-guard safe force-only base | `9.8389` | `900.12` | `33.08` | no regression |
| Post-guard `wn16` | `4.7196` | `781.57` | `7.71` | reject |

The backend now validates/prepares all manual AMD large-matmul variants, not just exact `wm32-wn32`, so an invalid coopmat geometry cannot silently benchmark by skipping output work. The safe profile did not regress after this guard. Current conclusion: honest Vulkan 32k remains below the fresh ROCm control (`11.0606 TPS`, `1155.52 tok/s` prompt) because of prefill, not because of batch/ubatch/KV/env selection. Next Vulkan work should be source-level Q3_K prefill profiling/code, with real generation smoke before any speed claim.

Artifacts:
- `build_logs/agent-workload/e076-vulkan32k-valid-base-r1.diagnostics.md`
- `build_logs/agent-workload/e076-vulkan32k-valid-wm128wn32-r1.diagnostics.md`
- `build_logs/agent-workload/e076-vulkan32k-shape-b4096ub1024-r1.diagnostics.md`
- `build_logs/agent-workload/e076-vulkan32k-kvf16-r1.diagnostics.md`
- `build_logs/agent-workload/e076-vulkan32k-postguard-base-r1.diagnostics.md`
- `docs/research/experiments/E076_vulkan_32k_valid_prefill_followup.md`

### E078-E102: valid Q3_K `mul_mm.comp` prefill work and tile validation

After the corrupt `wm32-wn32` route was rejected, H31 moved to the active Vulkan Q3_K coopmat route. Tracing showed the 12k prompt-heavy path uses `matmul_q3_k_f32_f16acc_aligned_l` from `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp`, not the MMQ-only path. A fixed `llama-bench pp7488` gate is now used for shader probes because it keeps prompt tokens stable while matching the large-prefill hotspot.

Current controls for the 12k lane (`ctx=12288`, `b=4096`, `ub=1024`, `q4_0/q4_0`, FlashAttention on, no reuse, thinking on, `triage_diff`, 64 generated tokens):

| Backend / config | Aggregate TPS | Prompt eval TPS | Decode eval TPS | Result |
| --- | ---: | ---: | ---: | --- |
| Vulkan force-large control before E082/E086 | `6.4679` | `905.64` | `40.36` | baseline |
| ROCm same-lane control | `7.1936` | `1129.76` | `28.63` | target |
| Vulkan E086 corrected Q3_K loadvec4 | `6.6277` | `934.80` | `40.13` | `+2.47%` aggregate, `+3.22%` prompt eval vs Vulkan control |

Fixed pp7488 gates:

| Config | pp7488 tok/s | Result |
| --- | ---: | --- |
| E082 baseline before stride18 | `908.23` | reference |
| E082 Q3_K stride18 | `922.62 ± 2.45` | kept, `+1.58%` |
| E086 Q3_K stride18 + corrected `LOAD_VEC_A=4` | `961.82 ± 25.60` | kept, `+4.25%` vs E082 |
| E091 E086 + `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wn48` | `972.31 ± 1.97` | downgraded by E093; static `WN=48` invalid for `BN=128`, not a profile |
| ROCm control | `1097.66` | remaining raw-prefill target |

Rejected probes in this phase: f16 dequant arithmetic, unsigned scale arithmetic, coopmat strides 16/19/22, E082 tile variants `wn32`, `wn16`, `wm128-wn32`, corrected `LOAD_VEC_A=8`, pair-scale helper, packed32 pair helper, Q8_1/MMQ/Q8-int-dot Q3 route, aligned-store specialization, Q3_K shift/mask and scale-int arithmetic rewrites, `bm256`/`bn256*` large tiles, and the old generator-only Q3_K `LOAD_VEC_A=4` probe. The kept E086 path differs from that old probe: it maps each 4-value load index to two Q3_K pair indices and writes both shared-memory slots.

Active Q3_K pipeline stats after E086/E102: `matmul_q3_k_f32_f16acc_aligned_l` uses `113 VGPR`, `45 SGPR`, `20480 B LDS`, `0 scratch` (`118 VGPR` before E086). E093/E097 static warptile scout downgraded E091's `wn48` result because the requested `WN=48` layout is invalid for the current `BN=128` route (`128 % 48 != 0`) and should runtime-fallback to base. Accepted H31 source baseline remains E082 stride18 + E086 corrected `LOAD_VEC_A=4`: fixed pp7488 `961.82 ± 25.60`, workload r1 `6.6277`, prompt eval `934.80`. E102 makes this fast AMD large-matmul route the no-env default on the local eligible AMD proprietary coopmat device; the no-env pp7488 recheck after cleanup was `983.48`, while `GGML_VK_DISABLE_AMD_LARGE_MATMUL=1` dropped to `708.19`.

E257 refreshed the dense `Qwen3.6-27B-Q3_K_S` Vulkan 12k cold lane after refocusing away from A3B/MoE profiles. The previous Vulkan q4 control shape (`ctx=12288,b=6144,ub=2048`, spec off, no reuse, no prime) confirmed at `6.6895 TPS` r3 with `947.36 tok/s` prompt eval. A focused shape gate found that Vulkan prefers `ub=1024` here, with `b7168/ub1024` confirmed at `7.0319 TPS` r3, `999.22 tok/s` prompt eval, and `40.93 tok/s` decode; lower `ub=768/512` follow-ups stayed below it. That is a measured `+5.12%` wall and `+5.47%` prompt improvement over the previous Vulkan control, so the GUI exact `Qwen3.6-27B-Q3_K_S.gguf` preset now uses `ctx=12288,b=7168,ub=1024,q4_0/spec=none` for the active Vulkan cold profile. The E257 intrusive trace still puts Q3_K `MUL_MAT` at `82.71%` of parsed time and FlashAttention at `9.60%`; future source changes still need a real Q3_K topology, not another nearby helper/tile retune.

E258 tested a larger Q3_K topology change: backend-private transposed Q3_K storage with a dedicated `matmul_q3_k_f32_transa_f16acc_aligned_l` route. The route activated on the hot Q3_K shapes and prompt eval rose slightly (`994.75 -> 1008.34 tok/s` vs paired control), but decode fell hard (`40.09 -> 35.76 tok/s`) and wall regressed (`6.9827 -> 6.9053 TPS`). The source prototype was reverted. E259 then closed two practical no-code follow-ups on the E257 shape: `batch=7680` lost (`6.9795 TPS` r1), while f16/f16 KV looked good in r1 (`7.1417`) but confirmed as a noise-level tie in r3 (`7.0543 TPS`, prompt `1008.37`, decode `40.00`). Do not change the 12k dense 27B Vulkan preset to f16 KV; q4/q4 `b7168/ub1024` remains the current default.

E260/E261 closed the older Vulkan transfer gates on the exact E257 shape. `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` measured `6.8663 TPS`, graphics queue plus `--no-mmap` measured `6.8743 TPS`, and `batch=8192/ub1024` measured `6.7312 TPS`; all three lost prompt throughput against E257. The older 64k graphics-queue/no-mmap lesson does not transfer to this dense 27B 12k lane, and a single `7489/7489` prompt chunk at `b8192` is slower than the current `b7168` shape. The broad f16-disable/f32acc-style pivot is also rejected: `GGML_VK_DISABLE_F16=1` measured only `5.2700 TPS` with prompt eval `710.49 tok/s`.

E264 tested a source-level graph topology gate: cast Q3_K FFN `src1` activations to F16 so Vulkan could use the existing f16-src1 matmul route. It lost decisively. Paired control measured `7.1320 TPS`, gate/up F16 src1 measured `5.9323 TPS`, and down-only F16 src1 measured `6.8000 TPS`. The prototype was reverted; per-layer activation casts are not the missing Vulkan Q3_K route.

12k workload validation for E091 (`GGML_VK_FORCE_AMD_LARGE_MATMUL=1`, `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wn48`) reached `6.7981` aggregate TPS r3, prompt eval `962.8567 tok/s`, decode `40.0967 tok/s`, prompt tokens `7489`, errors `0`, but this is now a suspect measurement rather than an accepted improvement. Do not use `wn48` as an opt-in profile unless a later backend log proves a different active valid prepared tile. Future tile/env variants must pass `python scripts/research/vulkan_warptile_static_scout.py` before benchmark claims.

After E093/E097 correction, the no-build gate estimates that matching ROCm pp7488 from the accepted E086/E102 baseline requires a large local speedup in the active Q3_K hotspot. This is too large for repeated helper-only or neighboring-stride probes unless the gate identifies a new high-share mechanism.

The updated static scout also closes the naive BK-depth idea before build: E097 corrected the model to local runtime constants (`subgroup=64`, coopmat `16x16x16`) and base Q3 LDS `20480 B`, matching driver stats. `BK=64` halves K-block/barrier rounds (`160 -> 80`) but leaves full-K dequant/B traffic unchanged and raises Q3 shared memory to `34816 B`; `BK=16` lowers shared memory but doubles K-block/barrier rounds. Treat BK variants as `needs-resource-proof`, not as benchmark candidates by default. E098 then measured the larger `bm256`/`bn256*` family: `bn256=947.12`, `bm256=909.59`, `bn256-wn128=921.84`, `bn256-wm128=940.21` pp7488 vs base around `983.21`, so all env branches were removed.

Fresh 32k spec-none controls after E100 are consistent with the updated no-env fast path: Vulkan reached `10.5230` aggregate TPS / `993.94 tok/s` prompt eval / `32.93 tok/s` decode, while ROCm reached `10.8879` / `1132.44` / `28.49` on the same lane. This is a control checkpoint, not a new candidate; the aggregate gap is down to about `-3.35%`, but the 32k gap remains prompt/prefill-side even though Vulkan decode is faster.

E095 added a Vulkan feature snapshot gate. On this system, `glslc` supports `coopmat`, `coopmat2`, `integer_dot`, and `bfloat16` feature tests, but `vulkaninfo` reports runtime `VK_KHR_cooperative_matrix=yes` and `VK_NV_cooperative_matrix2=no` for the AMD proprietary `26.3.1 (LLPC)` driver. Do not chase `mul_mm_cm2`/NV coopmat2 as the AMD acceleration route for this lane unless the driver capability changes.

E096 added a SPIR-V opcode summary gate for generated Vulkan shaders. The active KHR coopmat Q3_K binary (`matmul_q3_k_f32_aligned_f16acc_cm1.spv`) contains `OpCooperativeMatrixLoadKHR=2`, `OpCooperativeMatrixMulAddKHR=1`, `OpCooperativeMatrixStoreKHR=3`, and `OpControlBarrier=6`; the plain Q3_K binary has no cooperative-matrix ops. Use this only as a route/mechanism fingerprint, not a speed claim.

E099 kept `GGML_VK_MATMUL_ROUTE_TRACE=1` as a default-off diagnostic and rejected the Q8_1/int-dot route: temporary `matmul_q3_k_q8_1_l` creation did switch routes, but pp256 fell to `225.08` with `143 VGPR / 43 SGPR / 28672 B LDS`. E100 rejected aligned-store cleanup despite cleaner SPIR-V (`StoreKHR 3 -> 2`, barriers `6 -> 4`) because same-session 32k workload was `10.4974` vs baseline `10.5230`. E101 rejected Q3_K shift/mask and scale-int arithmetic rewrites (`927.51` and `929.30` pp7488). Do not retry these families without a new structural mechanism.

Artifacts:
- `docs/research/experiments/E078_vulkan12k_q3k_prefill_route_trace.md`
- `docs/research/experiments/E082_vulkan12k_q3k_coopmat_stride18_probe.md`
- `docs/research/experiments/E085_vulkan12k_q3k_stride18_tile_scout.md`
- `docs/research/experiments/E086_vulkan12k_q3k_correct_loadvec4_probe.md`
- `docs/research/experiments/E091_vulkan12k_q3k_e086_tile_rescout.md`
- `docs/research/experiments/E093_vulkan_warptile_static_scout.md`
- `docs/research/experiments/E094_vulkan_rocm_32k_specnone_controls.md`
- `docs/research/experiments/E095_vulkan_feature_snapshot_tooling.md`
- `docs/research/experiments/E096_spirv_opcode_summary_tooling.md`
- `docs/research/experiments/E097_vulkan_warptile_static_model_correction.md`
- `docs/research/experiments/E098_vulkan_q3k_bn256_large_tile_probe.md`
- `docs/research/experiments/E099_vulkan_q8_route_trace_and_negative_control.md`
- `docs/research/experiments/E100_vulkan_current_controls_store_and_shape_gates.md`
- `docs/research/experiments/E101_vulkan_q3k_dequant_micro_antipatterns.md`
- `docs/research/experiments/E102_vulkan_amd_auto_large_matmul_default.md`
- `docs/research/experiments/E257_vulkan12k_27b_prefill_rebaseline.md`
- `docs/research/experiments/E258_vulkan_q3k_transpose_a_route.md`
- `docs/research/experiments/E259_vulkan12k_practical_kv_shape_gates.md`
- `docs/research/experiments/E260_vulkan12k_queue_mmap_batch_gates.md`
- `docs/research/experiments/E264_vulkan12k_ffn_f16_src1_gate.md`
- `build_logs/agent-workload/e086-vulkan-q3-correct-loadvec4-pp7488-r3.md`
- `build_logs/agent-workload/e086-vulkan12k-q3-correct-loadvec4-r1.diagnostics.md`
- `build_logs/agent-workload/e086-vulkan-q3-correct-loadvec4-pipeline-stats.log`
- `build_logs/agent-workload/e091-vulkan-q3-e086-wn48-pp7488-r3.md`
- `build_logs/agent-workload/e091-vulkan12k-q3-e086-wn48-r3.diagnostics.md`
- `build_logs/agent-workload/e092-vulkan-q3k-prebuild-gate-smoke.md`
- `build_logs/agent-workload/e093-vulkan-warptile-static-scout.md`
- `build_logs/agent-workload/vscode-vulkan32k-control-r1.diagnostics.md`
- `build_logs/agent-workload/vscode-rocm32k-control-r1.diagnostics.md`
- `build_logs/agent-workload/e095-vulkan-feature-snapshot.md`
- `build_logs/agent-workload/e096-spirv-op-summary-q3k.md`
- `build_logs/agent-workload/e097-warptile-static-scout-corrected.md`
- `build_logs/agent-workload/e098-pipeline-stats-bn256-q3k-pp7488.txt`
- `build_logs/agent-workload/e099-force-intdot-q8-route-p256.txt`
- `build_logs/agent-workload/e100-vulkan32k-store-baseline-r1.diagnostics.md`
- `build_logs/agent-workload/e100-rocm32k-current-control-r1.diagnostics.md`
- `build_logs/agent-workload/e101-q3k-scale-int-pp7488-r1.txt`
- `build_logs/agent-workload/e102-vulkan-auto-large-noenv-pp7488-r1.txt`
- `build_logs/agent-workload/e102-vulkan32k-auto-large-noenv-r1.diagnostics.md`
- `build_logs/agent-workload/e257-vulkan12k-shape-b7168-ub1024-r3.diagnostics.md`
- `build_logs/agent-workload/e258-vulkan12k-q3k-transa-r1.diagnostics.md`
- `build_logs/agent-workload/e258-vulkan12k-control-postbuild-r1.diagnostics.md`
- `build_logs/agent-workload/e259-vulkan12k-kvf16-b7168-ub1024-r3.diagnostics.md`
- `build_logs/agent-workload/e260-vulkan12k-graphicsq-b7168-ub1024-r1.diagnostics.md`
- `build_logs/agent-workload/e260-vulkan12k-graphicsq-nommap-b7168-ub1024-r1.diagnostics.md`
- `build_logs/agent-workload/e261-vulkan12k-b8192-ub1024-r1.diagnostics.md`
- `build_logs/agent-workload/e263-vulkan12k-disable-f16-b7168-ub1024-r1.diagnostics.md`
- `build_logs/agent-workload/e264-vulkan12k-ffn-f16src1-gate-r1.diagnostics.md`
- `build_logs/agent-workload/e264-vulkan12k-ffn-f16src1-down-r1.diagnostics.md`

## TurboKV direct FlashAttention smoke (2026-05-13)

Это короткий технический smoke для guarded prototype `GGML_TKV_DIRECT_FATTN=1`, а не финальный target-lane speed claim. Полный артефакт с командами и числами: `build_logs/agent-workload/e009-tkv-direct-fattn-smoke-20260513.md`.

Профиль:

- `build-rocm-vec/bin/llama-bench.exe`
- `models/Qwen3.6-27B-Q3_K_S.gguf`
- `-p 64 -n 8 -b 128 -ub 128 -fa 1 -fitt 2048 -fitc 4096 -r 1 --no-warmup`
- `HSA_OVERRIDE_GFX_VERSION` unset

Результаты:

| KV cache | Path | pp64 tok/s | tg8 tok/s |
| --- | --- | ---: | ---: |
| q4_0/q4_0 | baseline before direct prototype | `224.10` | `26.81` |
| turbo4_0/turbo4_0 | graph dequant fallback | `186.69` | `17.09` |
| turbo4_0/turbo4_0 | `GGML_TKV_DIRECT_FATTN=1` | `227.88` | `24.82` |
| turbo3_0/turbo3_0 | `GGML_TKV_DIRECT_FATTN=1` | `221.67` | `24.60` |
| turbo2_0/turbo2_0 | `GGML_TKV_DIRECT_FATTN=1` | `225.50` | `25.52` |

Итог: direct path убирает основной penalty graph-dequant на маленьком smoke-lane и возвращает TKV decode близко к `q4_0/q4_0`. Перед включением по умолчанию нужны deterministic equivalence и полноценный prompt-heavy A/B.

## TurboKV vs q4 on active lane (2026-05-13)

Главное сравнение для `turbo4` нужно вести на том же best-shape, что и `q4`: `v2-review`, `ctx=12288`, `b=6144`, `ub=1024`, `repo-snapshot chars=21872`, no-reuse, thinking on, `spec=none`, модель `Qwen3.6-27B-Q3_K_S.gguf`.

Подробный артефакт с командами и файлами: `build_logs/agent-workload/e009-q4-vs-turbo4-ub1024-v2review-20260513.md`.

| KV cache | Mode | Runs | Aggregate TPS | Delta vs q4 |
| --- | --- | ---: | ---: | ---: |
| q4_0/q4_0 | baseline | `3` | `11.15` | baseline |
| turbo4_0/turbo4_0 | hybrid default (direct decode, F16 prefill) | `3` | `10.02` | `-10.1%` |
| turbo4_0/turbo4_0 | full direct prefill (`GGML_TKV_DIRECT_PREFILL=1`) | `1` | `7.70` | `-30.9%` |

Breakdown confirmed by server timings:

| KV cache | Prompt eval TPS mean | Decode eval TPS mean |
| --- | ---: | ---: |
| q4_0/q4_0 | `1149.47` | `27.85` |
| turbo4_0/turbo4_0 hybrid | `1013.22` | `25.80` |

Итог для текущего этапа: правильный `ub=1024` резко сокращает разрыв `turbo4` к `q4` с прежних `~26%` до `~10%`. Full-direct prefill пока хуже; текущий лучший путь для качества/скорости — `turbo4` hybrid: prefill через F16 dequant + WMMA, decode через direct TKV.

### Follow-up: specialized TKV4 set_rows kernel (2026-05-13)

После внедрения отдельного `TKV4` kernel path в `ggml/src/ggml-cuda/set-rows.cu` (вместо generic quant path для `GGML_TYPE_TKV4_0`) повторный A/B на том же lane дал дополнительное сокращение разрыва к `q4`.

Артефакты:
- `build_logs/agent-workload/e013-tkv4setrows-finalstable-q4-ub1024-r3.*`
- `build_logs/agent-workload/e013-tkv4setrows-finalstable-turbo4-ub1024-r3.*`

| KV cache | Mode | Runs | Aggregate TPS | Delta vs q4 |
| --- | --- | ---: | ---: | ---: |
| q4_0/q4_0 | baseline | `3` | `11.17` | baseline |
| turbo4_0/turbo4_0 | hybrid default + specialized TKV4 set_rows | `3` | `10.38` | `-7.1%` |

Дополнительно проверялись stage-2/stage-3 идеи (warp-level pack/reduction в set_rows и sign LUT для WHT), но воспроизводимого выигрыша поверх stage-1 не показали, поэтому откатаны для сохранения стабильного минимального diff.

### Follow-up: mixed TKV/Q8 direct FATTN route (2026-05-13)

Следующая идея из shadow/storage route - разрешить direct decode для mixed K/V, где одна сторона остаётся `TKV`, а другая `q8_0`. В hybrid prefill обе стороны при необходимости приводятся к F16, поэтому large-ubatch prefill остаётся на стабильном WMMA пути; direct compressed path используется на decode.

Артефакты:
- `build_logs/agent-workload/e015-mixedroute-control-turbo4-turbo4-ub1024-r3.*`
- `build_logs/agent-workload/e015-mixedroute-prefillfix-turbo4-q8v-ub1024-r3.*`
- `build_logs/agent-workload/e015-mixedroute-prefillfix-q8k-turbo4v-ub1024-r1.*`
- `build_logs/agent-workload/e015-mixedroute-directoff-turbo4-q8v-ub1024-r1.*`

| KV cache | Mode | Runs | KV size | Aggregate TPS | Delta vs q4 |
| --- | --- | ---: | ---: | ---: | ---: |
| q4_0/q4_0 | baseline from set_rows A/B | `3` | `216 MiB` | `11.17` | baseline |
| turbo4_0/turbo4_0 | same-build control | `3` | `198 MiB` | `10.36` | `-7.3%` |
| turbo4_0/q8_0 | mixed direct decode, F16 prefill | `3` | `303 MiB` | `10.60` | `-5.1%` |
| q8_0/turbo4_0 | mixed direct decode, F16 prefill smoke | `1` | `303 MiB` | `10.26` | `-8.1%` |

Итог: mixed `turbo4_0/q8_0` больше не падает на prefill и даёт небольшой speed-up относительно `turbo4_0/turbo4_0` control (`+2.3%`), но требует больше KV памяти (`303 MiB` против `198 MiB`) и всё ещё не обгоняет q4. Оставляем как явный opt-in режим для проверки более точного V cache, не как default recommendation.

Negative control: `GGML_TKV_DIRECT_FATTN=0` для `turbo4_0/q8_0` теперь корректно уходит в F16 fallback и завершает lane (`4.51 TPS` r1), но этот путь не конкурентен и нужен только как guard/debug switch.

### Stormrage benchmark shape recheck (2026-05-13)

Повторён benchmark shape из `Stormrage34/llama.cpp-turboquant-hip/scripts/run_rdna2_bench.sh` на текущей локальной сборке: `p=512,2048,4096`, `n=128`, `b=256`, `ub=128`, `ctk=turbo4`, `ctv=turbo2`, `fa=1`, `mmp=0`, `t=8`, `ngl=99`, `fit-target=2048`, `fitc=4096`, `r=3`. Для контроля также снят `q4_0/q4_0` тем же shape.

Важное ограничение: внешние числа из Stormrage README сняты на RX 6800 XT / RDNA2 (`gfx1030`) и для их MoE IQ4_XS/RDNA2 accelerator path. Локальные числа ниже сняты на RX 9070 XT / RDNA4 (`gfx1201`), ROCm 7.1, с локальными моделями `Qwen3.6-35B-A3B-UD-IQ3_XXS` и `Qwen3.6-27B-Q3_K_S`. Это operational-сравнение одного benchmark shape, не строгое apples-to-apples.

| Source / GPU | Model / KV | pp512 | pp2048 | pp4096 | tg128 |
| --- | --- | ---: | ---: | ---: | ---: |
| Stormrage README, RX 6800 XT | MoE baseline | `~480` | n/a | n/a | `~57` |
| Stormrage README, RX 6800 XT | MoE stable RDNA2 | `~540` | n/a | n/a | `~55` |
| Stormrage README, RX 6800 XT | MoE + RDNA2_MATMUL_OPT_V1 | `~1772 +/- 6` | n/a | n/a | `~52 +/- 7` |
| Stormrage README, RX 6800 XT | Dense 27B summary | `~480` | n/a | n/a | `~27` |
| Local RX 9070 XT | MoE35B `q4_0/q4_0` | `1318.83` | `1275.92` | `1239.98` | `102.76` |
| Local RX 9070 XT | MoE35B `turbo4_0/turbo2_0` | `1143.86` | `1064.55` | `992.07` | `56.71` |
| Local RX 9070 XT | Dense27B `q4_0/q4_0` | `795.66` | `787.07` | `776.22` | `28.59` |
| Local RX 9070 XT | Dense27B `turbo4_0/turbo2_0` | `636.45` | `608.08` | `554.85` | `20.49` |

Артефакты локального повторения:
- `build_logs/agent-workload/stormrage-shape-current-moe35b-q4-q4-20260513.jsonl`
- `build_logs/agent-workload/stormrage-shape-current-moe35b-turbo4-turbo2-20260513.jsonl`
- `build_logs/agent-workload/stormrage-shape-current-dense27b-q4-q4-20260513.jsonl`
- `build_logs/agent-workload/stormrage-shape-current-dense27b-turbo4-turbo2-20260513.jsonl`

Вывод: Stormrage `turbo4/turbo2` shape теперь воспроизводится на локальных реальных `TKV4/TKV2`, но на наших моделях и RDNA4 он не даёт speed advantage над `q4_0/q4_0`. Главный внешний выигрыш Stormrage остаётся связан с RDNA2 MoE-specific accelerator (`RDNA2_MATMUL_OPT_V1`), а не с общим dense/TurboKV path.

Extra `b=1024,ub=1024` recheck: по просьбе снят тот же Stormrage shape, но с раскрытым большим microbatch (`b=1024`, `ub=1024`; при исходном `b=256` значение `ub=1024` фактически не проверяет 1024-token microbatch). На RX 9070 XT большой `ubatch` резко поднимает MoE prefill, включая TurboKV, но `q4_0/q4_0` всё ещё быстрее в том же shape.

| Local RX 9070 XT | KV | pp512 | pp2048 | pp4096 | tg128 |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense27B Q3_K_S | `q4_0/q4_0` | `1079.38` | `1244.60` | `1225.79` | `28.85` |
| Dense27B Q3_K_S | `turbo4_0/turbo4_0` | `1006.08` | `1172.52` | `1135.15` | `20.95` |
| Dense27B Q3_K_S | `turbo4_0/turbo2_0` | `997.35` | `1168.99` | `1133.96` | `20.78` |
| MoE35B IQ3_XXS | `q4_0/q4_0` | `2807.61` | `3549.80` | `3500.76` | `102.50` |
| MoE35B IQ3_XXS | `turbo4_0/turbo2_0` | `2590.18` | `3290.59` | `3182.46` | `56.28` |

Артефакты extra run:
- `build_logs/agent-workload/stormrage-extra-ub1024-dense27b-q4-q4-20260513.jsonl`
- `build_logs/agent-workload/stormrage-extra-ub1024-dense27b-turbo4-turbo4-20260513.jsonl`
- `build_logs/agent-workload/stormrage-extra-ub1024-dense27b-turbo4-turbo2-20260513.jsonl`
- `build_logs/agent-workload/stormrage-extra-ub1024-moe35b-q4-q4-20260513.jsonl`
- `build_logs/agent-workload/stormrage-extra-ub1024-moe35b-turbo4-turbo2-20260513.jsonl`

MoE accelerator portability note: Stormrage `RDNA2_MATMUL_OPT_V1` is gated by compile flag, env var and `GGML_CUDA_CC_IS_RDNA2(cc)` in their `ggml/src/ggml-cuda/mmq.cuh`. It uses an RDNA2-tuned LDS double-buffer/padding path for MoE prefill, so it should not be blindly enabled on RDNA4 (`gfx1201`). If revisited, treat it as a separate guarded RDNA4 MoE/MMQ experiment with q4/TKV A/B and dense negative control; it is not a direct TurboKV storage-port follow-up.

Первичный underfilled A/B на `ub=192` сохранён только как диагностический trace direct/fallback, не как главный speed claim:

Подробный артефакт: `build_logs/agent-workload/e009-q4-vs-turbokv-v2review-20260513.md`.

| KV cache | Mode | Aggregate TPS | Delta vs q4 |
| --- | --- | ---: | ---: |
| q4_0/q4_0 | baseline | `9.01` | baseline |
| turbo4_0/turbo4_0 | direct (default) | `6.68` | `-25.9%` |
| turbo3_0/turbo3_0 | direct (default) | `6.25` | `-30.6%` |
| turbo2_0/turbo2_0 | direct (default) | `6.71` | `-25.5%` |
| turbo4_0/turbo4_0 | fallback (`GGML_TKV_DIRECT_FATTN=0`) | `3.10` | `-65.6%` |

## Надёжность замеров

### Активная политика контекста (2026-05-10)

- Для текущего performance-трека запускать benchmark только при `ctx <= 16384`.
- Запуски выше 16k считаются архивными/исследовательскими и не используются для текущих KPI.
- В `scripts/agent_workload_bench.py` и `scripts/repo_snapshot_context_bench.py` это ограничение включено по умолчанию; обход только явным флагом `--allow-ctx-above-16k`.

### Новый автономный чекпоинт (2026-05-10, lane <16k)

Профиль:

- `tasks=v2-mini`, `runs=1`, `ctx=12288`
- incoming prompt: `--real-context-mode repo-snapshot --real-context-chars 21872`
- no-reuse: `--cache-ram 0 --ctx-checkpoints 0`
- `q4_0/q4_0`, `spec=none`

Подтверждённый baseline после пересборки `build-rocm-vec`:

| Label | Build | Batch | UBatch | Aggregate TPS |
| --- | --- | ---: | ---: | ---: |
| `postrebuild-vec-b6144-ub512-none` | `build-rocm-vec` | `6144` | `512` | `9.85` |
| `postrebuild-vec-b6144-ub512-none-r2` | `build-rocm-vec` | `6144` | `512` | `9.84` |

Наблюдения из этого цикла:

- `ub=640` даёт резкий cliff на `build-rocm-wmma` (примерно `3.67-3.69 TPS`), поэтому активный safe corridor остаётся `ub=512`.
- `spec=ngram-mod` без prime (`--no-v2-prime-pass`) почти равен `spec=none`; всплеск при включённом prime не считать cold-first прогрессом.
- KV-типы `f16/bf16` на этом lane дают сильную регрессию (`~3.7-3.8 TPS`), `q4_0` остаётся лучшим.

### Shape-score paradox + context-cap probe (2026-05-11, superseded)

На lane `ctx=12288`, `b=6144`, `q4_0/q4_0`, no-reuse с shape-score:

- `ub=192` стабильно в быстром коридоре (`~8.52 TPS`);
- `ub=512` при тех же split-параметрах падает до `~4.19-4.24 TPS`.

Трассы показали, что для `ub192` и `ub512` совпадают:

- planner chosen/target histogram (`chosen=192`);
- GDN `n_tokens` histogram;
- FATTN hot-shape и MMQ selector route.

Бывший экспериментальный runtime-рычаг, использованный только как диагностический discriminator:

- env `LLAMA_UBATCH_SHAPE_CONTEXT_CAP=1`;
- при `LLAMA_UBATCH_SPLIT_POLICY=shape-score` и `LLAMA_UBATCH_SHAPE_PREFERRED=192` физический context `n_ubatch` капается до preferred.
- после root-cause проверки этот guard удалён из runtime: финальный фикс не меняет requested `-ub` и не использует shape-score/preferred cap.

Проверка на том же бинаре:

| Label | UBatch arg | Context cap | Aggregate TPS | Prompt eval | Decode eval |
| --- | ---: | --- | ---: | ---: | ---: |
| `p7-pass2-postctx-20260511-205925-shape-ub512-r1` | `512` | off | `4.19` | `332.79 tok/s` | `27.26 tok/s` |
| `p7-pass2-cap-20260511-205849-shape-ub512-r1` | `512` | on (`n_ubatch 512 -> 192`) | `8.53` | `827.82 tok/s` | `27.81 tok/s` |
| `p7-pass2-cap-20260511-205746-shape-ub192-r1` | `192` | on | `8.54` | `~828 tok/s` | `~27.8 tok/s` |

Вывод на этом этапе был неполным: context-cap доказал связь cliff с reserve/layout, но сам по себе был workaround, а не финальным решением.

### PP reserve outputs root cause (2026-05-12)

Финальная причина `ub489 -> ub490+` cliff на RDNA4/ROCm оказалась в reserve-time PP graph layout: обычный server decode резервировал PP graph как будто нужны logits/outputs для всех `n_tokens`, хотя на этом lane фактически нужен один output. Это раздувало compute buffer и переводило full graph в медленный layout при `ub490+`.

Финальный фикс: `llama_context::sched_reserve()` резервирует PP graph по фактическому числу decode outputs; all-output/encoder режимы оставляют полный reserve. Это не cap/guard и не меняет requested `-ub`.

Clean validation после удаления diagnostic probes, без `LLAMA_PP_RESERVE_SEQ_OUTPUTS`, без `LLAMA_UBATCH_SPLIT_POLICY`, без `LLAMA_UBATCH_SHAPE_PREFERRED`, без `LLAMA_UBATCH_SHAPE_CONTEXT_CAP`:

| Label | UBatch arg | Auto reserve log | Wall | Prompt eval |
| --- | ---: | --- | ---: | ---: |
| `e010-ub490-final-ppout` | `490` | `PP reserve outputs 490 -> 1` | `7.41s` | `966.26 tok/s` |
| `e010-ub512-clean-ppout` | `512` | `PP reserve outputs 512 -> 1` | `7.32s` | `979.33 tok/s` |

До output-aware reserve direct `ub490/491/512` были в slow band около `24-25s` wall и `~280-300 tok/s` prompt eval; с финальным reserve прямые `ub490+` остаются в fast band без обхода через меньший ubatch.

Чтобы исключить искажения от фонового `llama-server`, запускать benchmark с жёсткой проверкой:

```powershell
python scripts\agent_workload_bench.py --background-server-policy fail
```

Если процесс уже занят, runner завершится с ошибкой и покажет PID.

Для снижения методологического шума и анализа cold-vs-warm поведения:

```powershell
python scripts\agent_workload_bench.py `
  --background-server-policy fail `
  --server-seed 42 `
  --no-disable-thinking `
  --stats-ignore-first-run
```

- `--server-seed 42` фиксирует seed на стороне `llama-server` и уменьшает run-to-run случайность sampling path;
- `--no-disable-thinking` принудительно оставляет thinking включённым (обязательный режим для performance benchmark в этом форке);
- `--stats-ignore-first-run` печатает отдельные warm-only метрики (без run #1), чтобы не смешивать cold старт и рабочую фазу.

### Политика метрик (cold-first, 2026-05-09)

Для v2/v2-mini в этой ветке основной KPI фиксируется как **cold-first throughput**:

- измерение: первый измеряемый проход при `--runs 1`;
- для cold-замера отключать priming pass: `--no-v2-prime-pass`;
- warm/prime метрики считать диагностическими и публиковать отдельно, без подмены headline-числа.

Почему так:

- в агентном использовании с большим и меняющимся контекстом cold-фаза сильно влияет на реальный UX;
- warm-only числа показывают потенциал steady-state, но могут завышать ожидаемую скорость для «первого ответа»;
- ускорение cold-path почти автоматически улучшает и последующую warm-фазу.

Рекомендуемый формат отчёта:

- `Cold first-turn TPS` (headline);
- `Warm steady-state TPS` (secondary);
- `Session aggregate TPS` (смешанный показатель для серии запросов).

### Batch 4096 / UBatch 512 with stabilized method (2026-05-09)

Новый контрольный 5-run с фиксированным seed, thinking ON и warm-only статистикой:

- `build-rocm-wmma/bin/llama-server.exe`
- `Qwen3.6-27B-Q3_K_S.gguf`
- `-c 65536 -b 4096 -ub 512 -np 1 --flash-attn on`
- `--cache-type-k q4_0 --cache-type-v q4_0`
- `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`
- `--server-seed 42 --no-disable-thinking --stats-ignore-first-run`

Результат (`sprint14-b512-newmethod-thinkon-5run`):

- Aggregate completion TPS: `37.57`
- Mean task TPS: `38.90`
- Task TPS stdev: `6.5194`
- Warm-only aggregate TPS: `41.61`
- Warm-only task TPS stdev: `3.0439`

Итог: цель `>=35 TPS` для `b=4096/ub=512` подтверждена на обновлённой методике, при этом warm-only дисперсия существенно ниже.

## V2-mini simple workflow (27B only, 2026-05-09)

Цикл выполнен строго на `Qwen3.6-27B-Q3_K_S.gguf` с коротким набором задач:

- `--tasks v2-mini` (`v2_code_review` + `v2_write_function`)
- `--runs 1`
- `-c 65536 -b 4096 -ub 512 -np 1 --flash-attn on`
- `--cache-type-k q4_0 --cache-type-v q4_0`
- `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`
- `--background-server-policy fail`

Команды запускались через `build-rocm-exp/bin/llama-server.exe`.

Результаты по шагам:

| Label | Изменение | Aggregate TPS | Действие |
|---|---|---:|---|
| `wf-27b-baseline-exp-r1` | baseline | `25.98` | baseline |
| `wf-27b-varA-fattn-vec2-r1` | RDNA4 FATTN: quantized VEC порог `<=4 -> <=2` | `25.87` | **rollback (regress)** |
| `wf-27b-varB-mmq-routing-r1` | RDNA4 MMQ routing: убрать always-MMQ, ввести `ne11/type` эвристику | `26.58` | **keep (profit)** |
| `wf-27b-varC-streamk-r1` | MMQ stream-k: enable for RDNA4 при `ne11 >= 256` | `26.90` | **keep (profit)** |
| `wf-27b-varD-mmq-q45-384-r1` | RDNA4 MMQ routing: расширить окно Q4/Q5 `ne11 <= 256 -> <= 384` | `26.79` | **rollback (regress)** |
| `wf-27b-varE-mmq-k224-r1` | RDNA4 MMQ routing: расширить окно QK `ne11 <= 192 -> <= 224` | `26.72` | **rollback (regress)** |

Итог по циклу: финальная комбинация (B + C) дала `+0.92 TPS` к baseline v2-mini на 27B в этой сессии.

## Large Context Autotune (32K+)

Новый режим автоподбора параметров для длинного контекста:

```powershell
python scripts\agent_workload_bench.py `
  --autotune `
  --label rocm-autotune-32k `
  --server-bin build-rocm\bin\llama-server.exe `
  --model models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf `
  --background-server-policy fail `
  --autotune-min-ctx 32768 `
  --autotune-ctx-values 32768,49152,65536 `
  --autotune-batch-values 1024,2048,4096 `
  --autotune-ubatch-values 1024,2048,4096 `
  --autotune-kv-values q8_0,q4_0 `
  --autotune-spec-values none,ngram-mod `
  --autotune-update-preset `
  --autotune-preset-file gui\model_presets.json
```

Что делает режим:

- прогоняет grid конфигураций только для контекста `>= 32768`;
- сохраняет обычные `.csv/.jsonl` для каждой конфигурации;
- пишет summary: `<label>-autotune-summary.csv` и `.json`;
- печатает `BEST: ...` по aggregate completion TPS;
- при `--autotune-update-preset` обновляет `gui/model_presets.json` для выбранной модели.

## Large Context Reality Check (2026-05-10)

Статус изменён: активная long-context оптимизация временно остановлена как primary lane.

Почему:

- `sentinel128` дал ложное ощущение, что `128k` почти не хуже `64k`, потому что там prompt был всего `489/410` токенов.
- новый repo-snapshot workload загрузил действительно длинный prompt и показал, что проблема начинается уже на `64k`.

Зафиксированный reference:

| Workload | ctx64k | ctx128k | Комментарий |
| --- | ---: | ---: | --- |
| `sentinel128-qwen36q3` | `26.5825 TPS` | `26.0672 TPS` | Короткий sentinel, не годится как главный real-world сигнал |
| `repo-real-64k128k` | `2.3128 TPS` | `0.8167 TPS` | Реальный repo snapshot prompt, корректный long-prefill сигнал |

Исторический вывод на 2026-05-10:

- не делать новые 128k прогоны по умолчанию;
- не использовать 64k как стартовую «главную» точку оптимизации;
- тогдашний performance lane: prompt-heavy стартовая точка ниже `16k`.

### Archived Primary Goal (2026-05-10)

- Тогдашняя стартовая точка: `ctx=12288` в prompt-heavy no-reuse режиме.
- Тогдашний уровень: `~9.24 TPS`.
- Историческая цель: `25-27 TPS` на стартовой точке.
- Способ достижения: поиск и верификация изменений в кодовой базе llama.cpp/ggml (prefill/runtime path), не только параметрический тюнинг запуска.

### Agent Workload: prompt-heavy mode (incoming context fix)

Проблема: стандартный `scripts/agent_workload_bench.py` в `v2-mini` режиме часто оставался decode-heavy и имел слишком маленький входящий prompt для real-scenario выводов.

Решение:

- добавлен режим `--real-context-mode repo-snapshot`;
- в каждый task prompt инжектится большой `repo snapshot` префикс;
- добавлен ctx-aware safe cap, чтобы избегать `HTTP 400` от переполнения контекста:
  - `--real-context-safe-fill` (historical default `0.70`; 130k fallback when explicit chars are set to `0`),
  - `--real-context-reserve-tokens` (default `2048`),
  - `--real-context-chars-per-token` (default `3.4`).

Исторический запуск для реального входящего контекста без prompt-cache reuse:

```powershell
python scripts\agent_workload_bench.py `
  --label ctxwall-real-noreuse-c32768 `
  --server-bin build-rocm-compare\bin\llama-server.exe `
  --model models\Qwen3.6-27B-Q3_K_S.gguf `
  --tasks v2-mini --runs 1 `
  --ctx-size 32768 -b 2048 -ub 512 `
  --cache-type-k q4_0 --cache-type-v q4_0 `
  --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0" `
  --max-tokens 120 `
  --real-context-mode repo-snapshot --real-context-chars 180000
```

Первые точки нового ctx sweep (prompt-heavy, no-reuse):

| Label | ctx | Avg prompt tokens | Aggregate TPS |
| --- | ---: | ---: | ---: |
| `ctxwall-real-noreuse-c12288` | `12288` | `~8050` | `9.2389` |
| `ctxwall-real-noreuse-c16384` | `16384` | `~11550` | `6.8685` |
| `ctxwall-real-noreuse-c24576` | `24576` | `~19382` | `4.1962` |
| `ctxwall-real-noreuse-c32768` | `32768` | `~26860` | `2.8934` |

Вывод: на реалистичном большом входящем prompt'е стена начинается намного раньше, чем показывал старый decode-heavy режим.

Это исторический reference-коридор. На 2026-05-26 активные speed claims требуют fresh 130k Vulkan/ROCm baseline.

### Archived: 64K real-scenario single-ctx sanity (`repo_snapshot_context_bench.py`)

Эта секция сохранена как исторический reference. Активные speed claims теперь принимаются только по 130k lane или явно помеченным отдельным lanes.

Скрипт `scripts/repo_snapshot_context_bench.py` теперь по умолчанию смотрит на одиночный 130k профиль; `64k`/`128k` значения нужно передавать явно как исторические probes.

Первый 64k-only A/B на `build-rocm-compare`, `b=2048`, `ub=512`, `q4_0/q4_0`, prompt `62610` токенов, completion `120` токенов:

| Label | Spec | Wall TPS | Prompt eval | Decode eval | Вывод |
| --- | --- | ---: | ---: | ---: | --- |
| `repo-64k-single-none-ctx64k` | `none` | `0.9101` | `514.72 tok/s` | `11.89 tok/s` | текущий 64k baseline для real-scenario single-ctx |
| `repo-64k-single-ngram-ctx64k` | `ngram-mod` | `0.8955` | `506.02 tok/s` | `11.83 tok/s` | немного хуже baseline |

Практический вывод:

- для prompt-heavy `64k` repo snapshot workload `ngram-mod` пока не окупает свой overhead;
- ближайший safe baseline для новых 64k real-scenario исследований - `spec=none`;
- если возвращаться к speculative на этом lane, то только после новой гипотезы или kernel-level улучшения, а не по инерции от коротких decode-heavy benchmark'ов.

Дополнительный 64k-only check по `ubatch` на том же workload:

| Label | Batch | UBatch | Spec | Wall TPS | Prompt eval | Decode eval | Вывод |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `repo-64k-single-none-ctx64k` | `2048` | `512` | `none` | `0.9101` | `514.72 tok/s` | `11.89 tok/s` | текущий baseline |
| `repo-64k-single-none-ub256-ctx64k` | `2048` | `256` | `none` | `0.7263` | `405.13 tok/s` | `11.86 tok/s` | сильная регрессия по prefill |

Это означает, что для реального `64k` bottleneck сейчас чувствителен прежде всего к prompt processing throughput, и уменьшение `ubatch` до `256` здесь вредно, даже если в отдельных synthetic рассуждениях такой шаг казался безопасным.

Ещё три быстрых 64k-only проверки на том же repo snapshot lane:

| Label | Build | Batch | UBatch | Extra | Wall TPS | Prompt eval | Decode eval | Вывод |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `repo-64k-single-none-ctx64k` | `build-rocm-compare` | `2048` | `512` | none | `0.9101` | `514.72 tok/s` | `11.89 tok/s` | текущий local baseline |
| `repo-64k-single-none-b4096-ctx64k` | `build-rocm-compare` | `4096` | `512` | none | `0.8834` | `501.10 tok/s` | `11.79 tok/s` | `b=4096` не помогает, слегка хуже baseline |
| `repo-64k-single-none-nocache-ctx64k` | `build-rocm-compare` | `2048` | `512` | `--cache-ram 0 --ctx-checkpoints 0` | `0.8671` | `490.85 tok/s` | `11.86 tok/s` | отключение prompt cache/checkpoints не помогло |
| `repo-64k-exp-none-ctx64k` | `build-rocm-exp` | `2048` | `512` | none | `0.8985` | `510.48 tok/s` | `11.77 tok/s` | соседний ROCm build тоже не даёт прорыва |

Вывод по этому micro-screen:

- быстрые server-level рычаги на новом `64k` real-scenario lane почти исчерпаны;
- bottleneck остаётся в prefill/prompt path, а не в speculative или decode-path настройках;
- следующий полезный уровень исследования: kernel/path selection и runtime поведение на длинном prompt, а не новые перестановки `spec/cache/batch` вокруг того же бинаря.

Отдельно была проверена kernel-level probe в `ggml/src/ggml-cuda/gated_delta_net.cu`: принудительный `chunk_size=96` для RDNA4 chunked prefill вместо текущего adaptive `96/128`.

| Label | Variant | Wall TPS | Prompt eval | Decode eval | Решение |
| --- | --- | ---: | ---: | ---: | --- |
| `repo-64k-single-none-ctx64k` | baseline adaptive chunk | `0.9101` | `514.72 tok/s` | `11.89 tok/s` | baseline |
| `repo-64k-chunk96-none-ctx64k` | forced `chunk_size=96` | `0.8791` | `499.60 tok/s` | `11.81 tok/s` | rollback |
| `repo-64k-revert-check-none-ctx64k` | baseline rebuilt after rollback | `0.8957` | `510.31 tok/s` | `11.70 tok/s` | corridor restored |

Вывод: старая идея из quick-agent sweep не переносится напрямую на реальный repo-snapshot `64k` lane; фиксированный `chunk_size=96` ухудшает prefill и не подходит как следующий шаг.

## Archived: Separate Real-World Large Context Bench (120K + 160K)

Для регулярной оценки ожидаемой скорости в реальных агентных сценариях добавлен отдельный сценарный раннер:

- `scripts/large_context_realworld_bench.py`
- запускает одинаковый workload в двух практичных точках контекста: `122880` (120K) и `163840` (160K);
- строит итоговую сводку сравнения, чтобы быстро видеть деградацию скорости на растущем контексте;
- использует текущий `scripts/agent_workload_bench.py` как движок, поэтому метрики и формат логов полностью совместимы.

### Почему 120K/160K?

- **120K**: минимум для реальных длинных агентных диалогов + документы + контекст из других чатов.
- **160K**: расширенный сценарий, где важно видеть point-of-no-return по скорости.
- Диапазон позволяет оценить насколько критична масштабируемость и где находятся узкие места.

Рекомендуемый запуск:

```powershell
python scripts\large_context_realworld_bench.py `
  --label-prefix realctx-120k-qwen27b `
  --server-bin build-rocm-exp\bin\llama-server.exe `
  --model models\Qwen3.6-27B-Q3_K_S.gguf `
  --tasks v2-mini `
  --runs 1 `
  --batch-size 4096 `
  --ubatch-size 512 `
  --cache-type-k q4_0 `
  --cache-type-v q4_0 `
  --spec-profile ngram-mod `
  --background-server-policy fail
```

Что получаем после запуска:

- `build_logs/agent-workload/<label-prefix>-ctx120k.csv`
- `build_logs/agent-workload/<label-prefix>-ctx160k.csv`
- `build_logs/agent-workload/<label-prefix>-largectx-summary.csv` (TPS decay от 120K к 160K)
- `build_logs/agent-workload/<label-prefix>-largectx-summary.md`

Где смотреть итог:

- headline метрика: `aggregate_tps` из `*-largectx-summary.*`;
- сравнение `160K vs 120K` уже посчитано (ratio % показывает потерю скорости);
- если нужна стабильность вместо быстрых итераций, повышай `--runs` до `3`.

### Archived Research Target: 120K Large Context Optimization

Архивный baseline на `ctx=131072, ubatch=512, q4_0 KV, ngram-mod`:
- **Prefix (PP)**: ~215 TPS
- **Generation (TG)**: ~8.5 TPS ← **узкое место**
- **Spec acceptance rate**: ~18% (низко)

Гипотезы для исследования:

1. **MMQ/FATTN kernel threshold** — может быть на большом контексте срабатывает неоптимальный path (VEC vs TILE).
2. **Speculative decoding overshoot** — ngram-mod с large ctx может генерировать слишком много draft токенов, замедляя verification.
3. **KV cache bandwidth** — даже q4_0 может быть узким местом при 120K+ tokens × 24 heads × 256 dims.
4. **ROCm kernel occupancy** — RDNA4 может недополучать work при малых batch/ubatch на большом контексте.

Архивный research workflow:

```bash
# Step 1: historical baseline на 120K
python scripts/large_context_realworld_bench.py --label-prefix baseline-120k ...

# Step 2: попробовать более консервативный speculative (ngram-simple или none)
python scripts/large_context_realworld_bench.py --label-prefix nostep-120k --spec-profile none ...

# Step 3: попробовать меньший ubatch (256 вместо 512)
python scripts/large_context_realworld_bench.py --label-prefix ub256-120k --ubatch-size 256 ...

# Step 4: попробовать больший batch (6144 вместо 4096)
python scripts/large_context_realworld_bench.py --label-prefix b6144-120k --batch-size 6144 ...

# Сравнить результаты и выбрать best по aggregate_tps
```

Примечание 2026-05-10: этот workflow сохранён только как исторический след. Новые performance-итерации вести на `ctx=65536`.

## GUI Automation API (E2E)

GUI теперь поднимает локальный HTTP API для автоматизации действий и проверки результата end-to-end.

- Base URL: `http://127.0.0.1:8765`
- Port можно переопределить через `LLAMA_GUI_API_PORT`.

### Endpoints

- `GET /api/ping` — health check.
- `GET /api/state` — текущее состояние GUI-параметров (модель, контекст, batch, kv и т.д.).
- `POST /api/autotune` — запуск автотюна из GUI.
- `POST /api/apply-preset` — применение model preset в Launch Server.
- `POST /api/scenario/autotune-apply` — сценарий: autotune одной модели + apply preset.

### Пример сценария autotune + apply preset

```powershell
python - << 'PY'
import json, urllib.request

payload = {
  "model_path": "models/Qwen3.5-9B-Q6_K.gguf",
  "wait": True,
  "timeout_sec": 1200,
  "sweep_mode": "smoke"
}

req = urllib.request.Request(
  "http://127.0.0.1:8765/api/scenario/autotune-apply",
  data=json.dumps(payload).encode("utf-8"),
  headers={"Content-Type": "application/json"},
  method="POST",
)

with urllib.request.urlopen(req, timeout=1800) as resp:
    print(resp.read().decode("utf-8"))
PY
```

Если `ok=true`, в ответе будет:

- блок `autotune.result.best` с лучшей конфигурацией;
- пути к `*-autotune-summary.csv/json`;
- блок `preset.result` с применёнными значениями (`context`, `batch`, `kv`, ...);
- `state` с текущим состоянием GUI после применения пресета.

## Current Clean Snapshot

Актуальный clean snapshot на текущем ROCm build `5facfaea9` был снят через `build\bin\llama-server.exe`.

| Model | Mode | Key args | Aggregate completion TPS |
| --- | --- | --- | ---: |
| `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` | baseline | `-np 1 -c 32768 -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0` | `37.454` |
| `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` | `ngram-mod` | baseline + `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64` | `41.007` |
| `Qwen3.6-27B-Q3_K_S.gguf` | baseline | `-np 1 -c 32768 -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0` | `12.055` |
| `Qwen3.6-27B-Q3_K_S.gguf` | `ngram-mod` | baseline + `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64` | `13.547` |

Вывод на текущем билде: `ngram-mod` уже поддерживается и даёт прирост примерно `+9.5%` на 35B A3B и `+12.4%` на 27B Q3_K_S для короткой coding-agent симуляции.

Старые baseline CSV (`rocm-baseline-qwen36-*.csv`) стоит считать noisy, потому что часть прошлых замеров выполнялась при параллельной игровой нагрузке.

## RDNA4 Gated Delta Net Chunked Prefill (2026-05-08)

Экспериментальная kernel-ветка для `gated_delta_net` (chunked prefill на RDNA4) была проверена по строгому протоколу `3 runs` на quick-agent workload.

Параметры прогона:

- `build-rocm-vec/bin/llama-server.exe`
- `Qwen3.6-27B-Q3_K_S.gguf`
- `--spec-type ngram-mod`
- `-c 65536 -b 4096 -np 1 --flash-attn on`
- `--cache-type-k q4_0 --cache-type-v q4_0`

| Label | UBatch | Runs | Aggregate completion TPS | Mean task TPS | Median task TPS | Stdev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sprint4-gdn-chunk-ub256` | `256` | `3` | `31.7809` | `33.86` | `30.73` | `10.2471` |
| `sprint4-gdn-chunk-ub128` | `128` | `3` | `31.9844` | `33.39` | `36.37` | `6.6477` |

Вывод:

- Оба 3-проходных прогона выше ранее используемого ориентира `~29 TPS`.
- Зафиксирован новый практический коридор aggregate throughput: `~31.8-32.0 TPS` для этой модели и профиля.

Артефакты:

- `build_logs/agent-workload/sprint4-gdn-chunk-ub256.csv`
- `build_logs/agent-workload/sprint4-gdn-chunk-ub256.jsonl`
- `build_logs/agent-workload/sprint4-gdn-chunk-ub128.csv`
- `build_logs/agent-workload/sprint4-gdn-chunk-ub128.jsonl`

## RDNA4 Gated Delta Net Chunk Size Sweep (2026-05-08)

Проверен локальный A/B по `chunk_size` в `ggml/src/ggml-cuda/gated_delta_net.cu` при одинаковом quick-agent профиле и `Qwen3.6-27B-Q3_K_S.gguf`.

Параметры прогона:

- `build-rocm-vec/bin/llama-server.exe`
- `--spec-type ngram-mod`
- `-c 65536 -b 4096 -ub 256 -np 1 --flash-attn on`
- `--cache-type-k q4_0 --cache-type-v q4_0`

| Label | Chunk size | UBatch | Launches | Runs | Aggregate TPS | Mean TPS | Median TPS | Stdev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sprint5-gdn-chunk96-ub256` | `96` | `256` | `~3` | `3` | `33.17` | `35.57` | `32.87` | `10.47` |
| `sprint5-gdn-chunk96-ub256-r2` | `96` | `256` | `~3` | `3` | `31.86` | `33.47` | `29.86` | `7.65` |
| `sprint5-gdn-chunk64-control-ub256` | `64` | `256` | `4` | `3` | `30.76` | `31.52` | `30.17` | `5.11` |
| `sprint5-gdn-chunk64-control-ub256-r2` | `64` | `256` | `4` | `3` | `28.63` | `29.26` | `27.12` | `4.87` |
| `sprint5-gdn-chunk128-ub256` | `128` | `256` | `2` | `3` | `28.86` | `29.44` | `27.14` | `4.65` |
| `sprint5-gdn-chunk96-ub128` | `96` | `128` | `~2` | `3` | `28.53` | — | — | — |
| `sprint5-gdn-chunk96-ub512` | `96` | `512` | `~6` | `3` | `31.71` | `32.90` | `30.56` | `6.50` |
| `sprint5-gdn-chunk128-ub512` | `128` | `512` | `4` | `3` | `31.32` | `32.52` | `28.69` | `6.48` |

**Замечания по sweep ub × chunk_size:**

- `ub=512` НЕ регрессирует к ~20 TPS — ранее наблюдавшийся провал был при других условиях.
- chunk=128 на ub=256 (2 запуска) хуже chunk=96 (3 запуска): вероятно, увеличенный внутренний цикл (128 итераций vs 96) создаёт большее регистровое давление или является шумом (stdev ~5 TPS делает 3-run сравнение ненадёжным).
- Для ub=512 chunk=96 и chunk=128 дают одинаковый результат (~31.3-31.7 TPS) — разница в пределах погрешности.
- ub=256 чуть выше ub=512 при chunk=96 (~32.5 vs ~31.7 TPS), но разница незначительная при данной дисперсии.

**Теоретический предел chunk_size:**

$$\text{launches} = \left\lceil \frac{n\_tokens}{chunk\_size} \right\rceil$$

Снижение launch overhead даёт выгоду, пока:
- Каждый запуск меньше L1/L2 cache рабочего набора
- Отсутствует регистровое давление (spilling)
- Ядро остаётся memory-bandwidth-bound, а не compute-bound

Для ub=256: оптимум при chunk≈96 (3 launches). Переход к chunk=128 (2 launches) не даёт выигрыша — вероятно, внутренний цикл достигает предела.

Вывод: **chunk_size=96 — текущий confirmed optimal** для RDNA4 + Qwen3.6-27B на ub=256.

Артефакты:

- `build_logs/agent-workload/sprint5-gdn-chunk96-ub128.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-gdn-chunk96-ub256.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-gdn-chunk96-ub256-r2.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-gdn-chunk64-control-ub256.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-gdn-chunk64-control-ub256-r2.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-gdn-chunk128-ub256.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-gdn-chunk96-ub512.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-gdn-chunk128-ub512.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-adaptive-chunk-ub256.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-adaptive-chunk-ub512.{csv,jsonl}`

## RDNA4 Adaptive Chunk — Финальный результат (2026-05-08)

По итогам sweep реализован адаптивный `chunk_size` в `gated_delta_net.cu`:

```cpp
// n_tokens > 256 → chunk=128 (4 launches), иначе chunk=96 (3 launches)
const int64_t chunk_size = (n_tokens > 256) ? 128 : 96;
```

Верификационные прогоны (3 runs каждый):

| Label | UBatch | Effective chunk | Aggregate TPS |
| --- | ---: | ---: | ---: |
| `sprint5-adaptive-chunk-ub256` | `256` | `96` | `30.53` |
| `sprint5-adaptive-chunk-ub512` | `512` | `128` | **`33.86`** |

- `ub=512` с адаптивным chunk показал **33.86 TPS** — лучший результат за всю sprint5 сессию.
- `ub=256` в рамках нормальной дисперсии (~30-33 TPS, stdev ~5).
- Прежде ub≥256 деградировало до ~20 TPS из-за FATTN kernel switch — эта проблема устранена через chunked prefill.

Итоговый диапазон TPS для Qwen3.6-27B-Q3_K_S на RX 9070 XT (ROCm/gfx1201):

| Параметр | До sprint5 | После sprint5 |
|---|---:|---:|
| max ub без регресса | 128 | 512+ |
| типичный TPS (ub=256) | ~29 TPS | ~31-33 TPS |
| типичный TPS (ub=512) | ~20 TPS | ~31-34 TPS |

## RDNA4 FATTN Routing Tuning (2026-05-08, Sprint7)

Цель: проверить, можно ли получить стабильный выигрыш на фокусном профиле `ub=512` за счёт более раннего перехода из `TILE` в `MMA_F16` для RDNA4 в quantized KV path.

Изменение в `ggml/src/ggml-cuda/fattn.cu` (ветка `amd_wmma_available && RDNA4`):

```cpp
// было
if (Q->ne[1] * gqa_ratio_eff <= 8) return BEST_FATTN_KERNEL_TILE;

// стало
if (Q->ne[1] * gqa_ratio_eff <= 4) return BEST_FATTN_KERNEL_TILE;
```

Идея: сдвинуть crossover в сторону `MMA_F16` для более широкого диапазона эффективных батчей.

Профиль сравнения (одинаковый для всех запусков):

- `Qwen3.6-27B-Q3_K_S.gguf`
- `-c 65536 -b 4096 -ub 512 -np 1 --flash-attn on`
- `--cache-type-k q4_0 --cache-type-v q4_0`
- `--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-max 64`
- `build-rocm-vec/bin/llama-server.exe`

| Label | Variant | Runs | Aggregate TPS | Mean TPS | Median TPS | Stdev |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `sprint7-baseline5-ub512-ngram` | baseline (`tile<=8`) | `5` | `33.25` | `34.68` | `32.80` | `7.36` |
| `sprint7-tile4-5run-ub512-ngram` | patched (`tile<=4`) | `5` | `35.68` | `37.49` | `37.31` | `8.16` |
| `sprint7-tile4-5run-ub512-ngram-r2` | patched confirm | `5` | `33.96` | `36.12` | `33.00` | `9.55` |

Вывод:

- Патч показывает устойчивое преимущество над baseline в обоих 5-run замерах.
- Прирост по aggregate TPS:
  - run1: `35.68 - 33.25 = +2.43` TPS (`+7.3%`)
  - run2: `33.96 - 33.25 = +0.71` TPS (`+2.1%`)
- Порог `>32 TPS` устойчиво выполнен, а лучший подтверждённый результат цикла — `35.68 TPS`.

Артефакты:

- `build_logs/agent-workload/sprint7-baseline5-ub512-ngram.{csv,jsonl,server.log}`
- `build_logs/agent-workload/sprint7-tile4-5run-ub512-ngram.{csv,jsonl,server.log}`
- `build_logs/agent-workload/sprint7-tile4-5run-ub512-ngram-r2.{csv,jsonl,server.log}`

## Batch 4096 / UBatch 512 Repro Check (2026-05-09)

Запрос: подтвердить целевой уровень `>=35 TPS` именно на профиле `b=4096, ub=512` для long-context agent workflow.

Условия прогона:

- `Qwen3.6-27B-Q3_K_S.gguf`
- `-c 65536 -b 4096 -ub 512 -np 1 --flash-attn on`
- `--cache-type-k q4_0 --cache-type-v q4_0`
- `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`
- `build-rocm-wmma/bin/llama-server.exe`
- `scripts/agent_workload_bench.py --runs 5 --background-server-policy fail`

Результаты sprint14 (сегодня):

| Label | Aggregate TPS | Mean task TPS | Median task TPS | Stdev |
| --- | ---: | ---: | ---: | ---: |
| `sprint14-b512-target35-5run` | `30.25` | `31.15` | `29.34` | `5.55` |
| `sprint14-b512-target35-5run-r2` | `34.98` | `36.53` | `35.18` | `7.58` |
| `sprint14-b512-target35-5run-r3` | `32.90` | `34.34` | `32.55` | `7.25` |
| `sprint14-b512-target35-5run-r4` | `31.39` | `32.18` | `30.81` | `5.28` |

Ранее подтвержденные попадания `>=35 TPS` на том же профиле:

| Label | Build | Aggregate TPS |
| --- | --- | ---: |
| `sprint13-wmma-5run-r2` | `build-rocm-wmma` | `36.53` |
| `sprint7-tile4-5run-ub512-ngram` | `build-rocm-vec` | `35.68` |
| `sprint9-tile4-warmup-ub512-5run` | `build-rocm-vec` | `35.15` |

Вывод:

- Цель `35+ TPS` для `b=4096/ub=512` **достижима**, но имеет заметную run-to-run вариативность.
- Для стабильного daily-профиля на `build-rocm-clean` сейчас практичнее `ub=256` (средний 5-run `35.69 TPS`).
- Для приоритета именно `ub=512` нужно продолжать работу над снижением дисперсии (warmup discipline, thermal/load control, kernel-path stability).

Артефакты sprint14:

- `build_logs/agent-workload/sprint14-b512-target35-5run.{csv,jsonl,server.log}`
- `build_logs/agent-workload/sprint14-b512-target35-5run-r2.{csv,jsonl,server.log}`
- `build_logs/agent-workload/sprint14-b512-target35-5run-r3.{csv,jsonl,server.log}`
- `build_logs/agent-workload/sprint14-b512-target35-5run-r4.{csv,jsonl,server.log}`

### Stdev Investigation (2026-05-09)

Цель: выяснить, почему на `b=4096/ub=512` выросла дисперсия (`stdev`).

Ключевые наблюдения:

- В server log для нестабильных прогонов сильно гуляет `draft acceptance rate` и число speculative draft tokens.
- Пример:
  - низкий прогон `sprint14-b512-target35-5run`: итог `#gen tokens = 954`, `#acc tokens = 461`;
  - более быстрый прогон `sprint14-b512-target35-5run-r2`: итог `#gen tokens = 1500`, `#acc tokens = 918`.
- Это указывает, что заметная часть дисперсии идёт из speculative path (`ngram-mod`), а не из prompt prefill.

Контрольный тест без speculative (`--spec-type none`) на том же профиле:

| Label | Aggregate TPS | Mean task TPS | Median task TPS | Stdev |
| --- | ---: | ---: | ---: | ---: |
| `sprint14-b512-specnone-5run` | `27.54` | `27.54` | `27.66` | **`0.28`** |

Вывод: без speculative дисперсия почти исчезает, но throughput заметно ниже.

Быстрый стабилизационный A/B (3-run, warmup on) не дал снижения stdev:

| Label | Config | Aggregate TPS | Stdev |
| --- | --- | ---: | ---: |
| `sprint14-stab-warmup-default-3run` | ngram 24/48/64 | `31.77` | `5.99` |
| `sprint14-stab-warmup-n32-3run` | ngram 32/48/64 | `32.65` | `6.76` |
| `sprint14-stab-warmup-min32max48-3run` | ngram 24/32/48 | `32.79` | `9.18` |

Практический итог:

- Высокий stdev на `ub=512` в первую очередь связан с нестабильным speculative acceptance.
- Для стабильного daily-профиля приоритет остаётся у `ub=256`.
- Для `ub=512` следующая работа должна быть направлена на стабилизацию speculative acceptance, а не только на peak TPS.

## UBatch=256 Optimization Discovery (2026-05-09)

**Critical finding**: При систематическом тестировании разных ubatch размеров выявлено, что **ubatch=256 даёт значительное преимущество** на этом профиле и GPU.

### Methodology

Compared 5-run baseline warm-cache runs с одинаковыми параметрами:

- `Qwen3.6-27B-Q3_K_S.gguf`
- `-c 65536 -b 4096 -np 1 --flash-attn on`
- `--cache-type-k q4_0 --cache-type-v q4_0`
- `--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-max 64`
- `build-rocm-clean/bin/llama-server.exe` (master commit 8c7db71f1)

| UBatch | Runs | Aggregate TPS | Mean task TPS | Median task TPS | Stdev |
| --- | ---: | ---: | ---: | ---: | ---: |
| `256` | `5` | **`35.45`** | `37.20` | `39.37` | `7.95` |
| `256` (r2) | `5` | **`35.93`** | `37.76` | `37.67` | `8.40` |
| `224` | `5` | `33.80` | `35.24` | `34.82` | `7.18` |
| `512` | `5` | `31.05` | `32.08` | `27.84` | `6.23` |

**Average ub=256**: `(35.45 + 35.93) / 2 = **35.69 TPS**` — **+14.7% vs ub=512 baseline**

### Why ub=256?

Гипотезы:

1. **Memory hierarchy alignment**: ub=256 (32 KB uBatch state per thread block) может оптимально вписываться в GPU L1/L2 cache на gfx1201.
2. **GDN chunking**: Адаптивный chunk_size=96 (from sprint5-adaptive-chunk) работает наилучше именно с ub=256 как базовой единицей.
3. **FATTN kernel dispatch**: VEC/TILE/MMA crossover точки оптимальны для ub=256 при данной длине контекста.

### Single-run cold-cache behavior

Интересно, что на single-run (cold cache) нет заметного преимущества:

| UBatch | Single-run TPS |
| --- | ---: |
| `256` | `27.00` |
| `192` | `27.10` |
| `224` | `25.88` |
| `320` | `26.81` |
| `384` | `26.97` |
| `512` | `25.14` |
| `768` | `19.84` |

**Вывод**: Преимущество ub=256 проявляется только при **прогреве кэша** в серии запусков. Single-run benchmarks **не отражают реальной производительности** для этого профиля.

### Artifacts

- `build_logs/agent-workload/baseline-clean-5run-ub256.{csv,jsonl,server.log}`
- `build_logs/agent-workload/baseline-clean-5run-ub256-r2.{csv,jsonl,server.log}`
- `build_logs/agent-workload/baseline-clean-5run-ub512.{csv,jsonl,server.log}` (для сравнения)

### Recommendation

**Обновить все Qwen3.6-27B профили** в `gui/model_presets.json` с `ubatch: 512` → `ubatch: 256`.

Цель: **Стабильно достичь 35+ TPS** на RX 9070 XT при агентной рабочей нагрузке.

## Baseline ROCm

```powershell
python scripts\agent_workload_bench.py `
  --label rocm-baseline `
  --server-bin build-rocm\bin\llama-server.exe `
  --model models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf `
  --ctx-size 32768 `
  --batch-size 2048 `
  --ubatch-size 2048 `
  --cache-type-k q8_0 `
  --cache-type-v q8_0 `
  --max-tokens 160
```

## Existing Server

Если GUI уже запустил сервер:

```powershell
python scripts\agent_workload_bench.py --no-start --port 8080 --label gui-server-baseline
```

## MTP Branch Test

Только после того, как `llama-server --help` показывает `mtp`:

```powershell
python scripts\agent_workload_bench.py `
  --label rocm-mtp-draft3 `
  --server-bin build-rocm-mtp\bin\llama-server.exe `
  --model models\Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf `
  --server-extra "--spec-type mtp --spec-draft-n-max 3" `
  --ctx-size 32768 `
  --cache-type-k q8_0 `
  --cache-type-v q8_0
```

MTP benchmark должен быть text-only: не добавлять `--mmproj`.

## ngram-mod Coding-Agent Test

Для текущего master без MTP:

```powershell
python scripts\agent_workload_bench.py `
  --label rocm-ngram-mod `
  --server-extra "--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-draft-n-min 48 --spec-draft-n-max 64"
```

Для текущего parser актуальны и новые long-form имена флагов:

```powershell
python scripts\agent_workload_bench.py `
  --label rocm-ngram-mod `
  --server-extra "--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64"
```

На текущем билде предпочтительнее использовать именно `--spec-ngram-mod-*`, чтобы не путать их с draft-model speculative decoding.

## Глоссарий метрик

| Метрика | Тип | Пояснение |
|---------|-----|-----------|
| `wall_s` | секунды | Астрономическое (настенное) время от отправки запроса до получения последнего токена. Включает все задержки: сеть, prompt processing, generation. Главная метрика скорости для агентной задачи. |
| `completion_tokens` | шт. | Количество токенов, сгенерированных моделью (не считая prompt). Зависит от задачи и stop-sequence, у нас лимитируется `--max-tokens`. |
| `completion_tps_wall` | тк/с | Throughput генерации: `completion_tokens / wall_s`. Основная агрегированная метрика в CSV. Чем выше — тем лучше. |
| `prompt_tokens` | шт. | Число токенов в контексте (системный промпт + вопрос). Влияет на prefill latency. |
| `ttft_s` | секунды | Time-To-First-Token — latency до первого сгенерированного токена. Отражает скорость prompt processing (PP). |
| `tg_tps` | тк/с | Token Generation speed из server log — чистая скорость генерации без prefill. Отличается от `completion_tps_wall`: wall учитывает TTFT, tg_tps — нет. |
| `pp_tps` | тк/с | Prompt Processing speed из server log — скорость обработки контекста (prefill). |
| `spec_accept_rate` | % | Процент принятых speculative токенов (для MTP/ngram). 100% = все драфтные токены приняты, 0% = ни одного. Реальный прирост TPS зависит от acceptance rate. |
| `error` | строка | Непустое поле означает сбой запроса (HTTP error, timeout, empty response). |

> **Важно для агентного workflow**: если MTP/ngram повышает `tg_tps`, но увеличивает `ttft_s` (более долгий prefill), итоговый `wall_s` может не улучшиться. Смотреть нужно именно на `completion_tps_wall` и `wall_s`.

## Что сравнивать

Смотреть в CSV:

- `wall_s` по каждой задаче;
- `completion_tokens`;
- `completion_tps_wall`;
- ошибки запуска/ответа.

Смотреть в server log:

- prompt processing tok/s;
- generation tok/s;
- speculative draft acceptance rate;
- ROCm/HIP warnings;
- VRAM/memory allocation failures.

Для нашего workflow важен не только TG. Если MTP ускоряет generation, но сильно режет prompt processing, агентная задача может стать медленнее.

Смежный roadmap по следующим аппаратно-ориентированным оптимизациям вынесен в `ROCM_ACCELERATION_PLAN.md`.

---

## Методика V2 — Реалистичный Agentic-Flow Benchmark (2026-05-09)

### Мотивация

Задачи `TASKS_QUICK/FULL` (v1) специально коротки (`max_tokens=160`, "keep it brief"), что создаёт искусственно высокий TPS (многократные короткие burst генерации с частым ngram accept). Реальный агентный флоу — длинные ответы (400–600 токенов), разнообразные промпты с низким ngram acceptance. Поэтому v1 и ручной чат показывают разные числа.

### V2 Task Set (`--tasks v2`)

По умолчанию v2 теперь запускает компактный набор для быстрых итераций:
- включены: `v2_code_review`, `v2_write_function`;
- отключены: `v2_debug_trace`, `v2_refactor_plan`, `v2_perf_analysis`.

Полный набор включается только для ретеста после заметного speed breakthrough:
- добавить флаг `--v2-include-heavy`.

| ID | Название | Целевая длина ответа |
|----|----------|---------------------|
| `v2_code_review` | Полный code review модуля build_manager | ~400–500 токенов |
| `v2_write_function` | Написать класс BuildRegistry | ~450–550 токенов |
| `v2_debug_trace` | Диагностика crash-лога ROCm сервера | ~350–450 токенов |
| `v2_refactor_plan` | План рефакторинга монолитного GUI | ~400–500 токенов |
| `v2_perf_analysis` | Анализ performance bottleneck | ~400–500 токенов |

### Ключевые отличия от V1

| Параметр | V1 (quick) | V2 |
|----------|------------|-----|
| `--max-tokens` | 160 | 500 (автоматически) |
| Формулировка задач | "keep it brief / under 140 words" | Развёрнутые, без ограничений длины |
| `--history-version` | v1 → `BENCH_HISTORY.csv` | v2 → `BENCH_HISTORY_V2.csv` |
| Соответствие реальному чату | Оптимистичная оценка | Репрезентативная оценка |

### Команда V2 Baseline

```powershell
python scripts\agent_workload_bench.py `
  --label v2-baseline-rocm-ub512 `
  --tasks v2 `
  --runs 3 `
  --server-seed 42 `
  --no-disable-thinking `
  --stats-ignore-first-run `
  --server-bin build-rocm-vec\bin\llama-server.exe `
  --model models\Qwen3.6-27B-Q3_K_S.gguf `
  --ctx-size 65536 `
  --batch-size 4096 `
  --ubatch-size 512 `
  --cache-type-k q4_0 `
  --cache-type-v q4_0 `
  --flash-attn `
  --server-extra "--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64 --spec-ngram-mod-n-match 24"
```

Для полного ретеста с тяжёлыми задачами:

```powershell
python scripts\agent_workload_bench.py `
  --label v2-baseline-rocm-ub512-heavy `
  --tasks v2 `
  --v2-include-heavy `
  --runs 3 `
  --server-seed 42 `
  --no-disable-thinking `
  --stats-ignore-first-run `
  --server-bin build-rocm-vec\bin\llama-server.exe `
  --model models\Qwen3.6-27B-Q3_K_S.gguf `
  --ctx-size 65536 `
  --batch-size 4096 `
  --ubatch-size 512 `
  --cache-type-k q4_0 `
  --cache-type-v q4_0 `
  --flash-attn `
  --server-extra "--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64 --spec-ngram-mod-n-match 24"
```

История результатов хранится отдельно: `build_logs/agent-workload/BENCH_HISTORY_V2.csv` и `BENCH_HISTORY_V2.md`.

### V2 Baseline Results

| Label | Build | Runs | Aggregate TPS | Mean TPS | Median TPS | Stdev | Warm-only TPS | Warm stdev | max_tokens |
|-------|-------|------|--------------|----------|------------|-------|--------------|------------|------------|
| v2-baseline-rocm-ub512 | build-rocm-vec | 3×5 | 27.77 | 27.78 | 27.97 | 0.47 | 28.07 | 0.19 | 500 |

**Вывод:** v2 baseline = **~28 TPS** при 500-токенных ответах — это точно совпадает с тем, что наблюдается в ручном чате (28–30 TPS). Очень низкий stdev (0.47) показывает, что при длинных ответах генерация устойчива. V1 (~33-37 TPS) был оптимистичен из-за многократных коротких burst (160 токенов).

### V2 A/B: `build-rocm-clean` vs `build-rocm-vec` (ub=512, ngram-mod)

| Label | Build | Runs | Aggregate TPS | Mean TPS | Median TPS | Stdev | Warm-only TPS | Warm stdev |
|-------|-------|------|--------------|----------|------------|-------|--------------|------------|
| `v2-baseline-rocm-ub512` | `build-rocm-vec` | 3x5 | `27.77` | `27.78` | `27.97` | `0.47` | `28.07` | `0.19` |
| `v2-clean-ub512` | `build-rocm-clean` | 3x5 | `27.72` | `27.72` | `27.80` | `0.35` | `27.92` | `0.17` |

Разница по aggregate: `+0.06 TPS` в пользу `build-rocm-vec` (меньше порога `0.5 TPS`).

**Вывод:** на реалистичной v2 нагрузке патчи `tile<=4 + chunk=96` не дают значимого выигрыша по throughput.

### V2 A/B: `spec-type none` vs `ngram-mod` (ub=512, build-rocm-vec)

| Label | Spec mode | Runs | Aggregate TPS | Mean TPS | Median TPS | Stdev | Warm-only TPS | Warm stdev |
|-------|-----------|------|--------------|----------|------------|-------|--------------|------------|
| `v2-baseline-rocm-ub512` | `ngram-mod 48/64/24` | 3x5 | `27.77` | `27.78` | `27.97` | `0.47` | `28.07` | `0.19` |
| `v2-rocm-vec-specnone-ub512` | `none` | 3x5 | `27.78` | `27.78` | `27.92` | `0.33` | `27.99` | `0.06` |

Разница по aggregate: `~0.00 TPS` (в пределах шума).

**Вывод:** для v2-кодовых промптов `ngram-mod` практически не ускоряет, но и не штрафует throughput; заметный эффект в основном на variance (без speculative stdev ниже).

### V2 A/B: `ubatch 256` vs `ubatch 512` (build-rocm-vec, ngram-mod)

| Label | ubatch | Runs | Aggregate TPS | Mean TPS | Median TPS | Stdev |
|-------|--------|------|--------------|----------|------------|-------|
| `v2-baseline-rocm-ub512` | `512` | 3x5 | `27.77` | `27.78` | `27.97` | `0.47` |
| `v2-rocm-vec-ub256-ngram-r1` | `256` | 1x5 | `27.52` | `27.52` | `27.35` | `0.32` |

Разница по aggregate: `-0.25 TPS` при переходе на `ub=256`.

**Вывод:** на текущем профиле длинных ответов `ub=512` остаётся предпочтительным.

### Политика прогонов для V2 (обновлено)

- Для быстрых итераций/скрининга использовать `--runs 1` (экономия времени, stdev на v2 обычно низкий).
- Повторять `--runs 3` только для финального подтверждения спорных/пограничных изменений (например, дельта в диапазоне `0.2-0.5 TPS`).

### Research Phase R35-01 (2026-05-09): старт long-run к цели 35 TPS

Цель фазы: найти конфиг/билд, который сможет вывести v2-профиль к `35 TPS`.

#### Скрининг готовых ROCm билдов (`runs=1`, v2, `b=4096`, `ub=512`, `ngram-mod`)

| Label | Build | Aggregate TPS |
|-------|-------|--------------|
| `v2-scan-rocm-exp-ub512-r1` | `build-rocm-exp` | `27.37` |
| `v2-scan-rocm-wmma-ub512-r1` | `build-rocm-wmma` | `27.34` |
| `v2-scan-build-bin-ub512-r1` | `build` | `27.33` |
| `v2-scan-rocm-clean-ub512-r1` | `build-rocm-clean` | `27.26` |
| `v2-scan-rocm-vec-ub512-r1` | `build-rocm-vec` | `27.26` |
| `v2-scan-rocm-a-check-ub512-r1` | `build-rocm-a-check` | `27.20` |

Промежуточный лидер: `build-rocm-exp` (`27.37 TPS`).

#### Свип параметров на лидере `build-rocm-exp` (`runs=1`)

| Label | Конфиг | Aggregate TPS |
|-------|--------|--------------|
| `v2-scan-exp-b4096-ub512-p1-specnone-r1` | `b=4096, ub=512, p=1, spec=none` | `27.24` |
| `v2-scan-exp-b8192-ub512-p1-specngram-r1` | `b=8192, ub=512, p=1, spec=ngram` | `27.21` |
| `v2-scan-exp-b4096-ub512-p1-specngram-r1` | `b=4096, ub=512, p=1, spec=ngram` | `27.20` |
| `v2-scan-exp-b4096-ub512-p2-specngram-r1` | `b=4096, ub=512, p=2, spec=ngram` | `25.70` |
| `v2-scan-exp-b8192-ub1024-p1-specngram-r1` | `b=8192, ub=1024, p=1, spec=ngram` | `20.18` |

Вывод по свипу:
- `ub=1024` и `parallel=2` в этом профиле явно вредят throughput.
- `spec none` и `ngram-mod` дают почти одинаковую скорость на v2-кодовых задачах.
- На текущем железе/модели v2-профиль упирается в ~`27.2-27.4 TPS`.

#### Статус чекпоинта

- Целевой чекпоинт `35 TPS` на v2-профиле **не достигнут** (текущий максимум в этой фазе: `27.37 TPS`).
- Для дальнейшего роста нужен следующий виток: новые кодовые kernel-правки + свежая ROCm сборка с корректной toolchain-настройкой.

#### Новый ROCm контур `build-rocm-r35-c` (`GGML_CUDA_FA_ALL_QUANTS=ON`, `GGML_OPENMP=OFF`)

| Label | Spec mode | Aggregate TPS |
|-------|-----------|--------------|
| `v2-scan-rocm-r35-c-ub512-r1` | `ngram-mod 48/64/24` | `26.83` |
| `v2-scan-rocm-r35-c-specnone-r1` | `none` | `27.30` |

Вывод:
- `GGML_CUDA_FA_ALL_QUANTS=ON` сам по себе не помог на текущем v2 профиле.
- Без speculative новый контур близок к обычному уровню, но всё равно не обгоняет `build-rocm-exp`.
- Этот билд не выглядит перспективным для дальнейшего разгона к `35 TPS`.

### Research Phase R35-02 (2026-05-09): kernel micro-optimizations (ROCm, runs=1)

Цель фазы: проверить быстрые low-risk правки в ядрах без смены модели/режима и оценить, дают ли они выход за потолок `~27.4 TPS` на v2.

#### Эксперимент A: `ggml/src/ggml-cuda/gated_delta_net.cu`

Гипотеза:
- уменьшить стоимость `expf` в fused GDN (замена на fast intrinsic + кэширование `exp(g)` в `KDA` ветке) может ускорить decode/prefill.

Результаты (`build-rocm-exp`, `b=4096`, `ub=512`, `np=1`):

| Label | Spec mode | Aggregate TPS |
|-------|-----------|--------------|
| `v2-r35-gdn-expfast-ub512-r1` | `ngram-mod 48/64/24` | `27.42` |
| `v2-r35-gdn-expfast-specnone-ub512-r1` | `none` | `27.29` |

Промежуточный вывод:
- метрики остались в шумовом коридоре относительно текущего потолка `27.2-27.4`;
- устойчивого прироста не подтверждено.

#### Эксперимент B: `ggml/src/ggml-cuda/fattn.cu` (RDNA4 selector threshold)

Гипотеза:
- расширить окно выбора VEC/TILE (`<=8` вместо `<=4`) в RDNA4 ветке и ускорить decode на малом эффективном батче.

Результаты:

| Label | Spec mode | Aggregate TPS |
|-------|-----------|--------------|
| `v2-r35-fattn8-gdnexp-ub512-r1` | `ngram-mod 48/64/24` | `27.36` |
| `v2-r35-fattn8-gdnexp-specnone-ub512-r1` | `none` | `27.06` |

Вывод:
- изменение порога ухудшило non-spec профиль и не дало выигрыша с `ngram-mod`.
- правка откатана.

#### Rollback-check после отката обеих правок

| Label | Spec mode | Aggregate TPS |
|-------|-----------|--------------|
| `v2-r35-rollback-check-ub512-r1` | `ngram-mod 48/64/24` | `27.42` |

Финал фазы R35-02:
- обе kernel-гипотезы не дали подтверждённого роста TPS;
- дерево возвращено к baseline-поведению;
- целевой чекпоинт `35 TPS` для v2 остаётся недостигнутым.

### Research Phase R35-03 (2026-05-09): draft-model path + kernel pass

Цель фазы: проверить «дорогой» путь ускорения через draft model (non-MTP), затем сделать kernel/runtime pass по самому слабому месту из логов.

#### Что использовалось как draft model path

- target model: `models/Qwen3.6-27B-Q3_K_S.gguf`;
- draft model: `models/Qwen3.5-9B-Q6_K.gguf`;
- режим: `--model-draft ... --spec-draft-n-max 12 --spec-draft-n-min 0 --spec-draft-p-min 0.75`.

#### Сравнение baseline режимов на compact v2 (`runs=1`, 2 задачи)

| Label | Spec mode | Aggregate TPS |
|-------|-----------|--------------|
| `v2-r35-combo-draft-sweep-r1-cfg01` | `none` | `27.44` |
| `v2-r35-combo-draft-sweep-r1-cfg02` | `ngram-mod 48/64/24` | `27.41` |

#### Draft-model результат

По `v2-r35-combo-draft-only-r1b-cfg01.server.log`:
- `prompt eval`: ~`1.22-1.28 ms/token` (нормально);
- `eval`: ~`308-321 ms/token` (`~3.11-3.24 tok/s`) — критический провал;
- `draft acceptance rate`: высокий (`~0.79-0.86`), но это не помогает;
- `statistics draft ... dur(g)`: `~140-288 s` — узкое место именно генерация draft model.

Вывод:
- на текущем локальном draft (`Qwen3.5-9B-Q6_K`) speculative через draft model радикально медленнее baseline (`~3.2 tok/s` vs `~27.4 TPS`);
- bottleneck не в acceptance, а в стоимости самого draft decode.

#### Kernel/runtime pass по узкому месту

Была проверена runtime-гипотеза снижения стоимости draft-контекста (batch sizing в `tools/server/server-context.cpp`), но воспроизводимого ускорения не получено.

Итог:
- runtime-патч откатан;
- кодовая база возвращена к baseline-поведению;
- для продолжения draft-ветки нужен существенно более лёгкий draft GGUF (уровня ~0.5B-1.5B), иначе этот путь не конкурентен.

### Research Phase R35-04 (2026-05-09): kernel-only возврат (без draft-model)

Цель: вернуться к чистой kernel-only ветке и проверить более агрессивный selector-твик в RDNA4 FlashAttention.

Изменение:
- файл: `ggml/src/ggml-cuda/fattn.cu`;
- ветка `amd_wmma_available && RDNA4`;
- для non-quantized single-query decode (`Q->ne[1] == 1`) добавлен ранний выбор `BEST_FATTN_KERNEL_VEC` при `gqa_ratio_eff <= 2`.

#### Результаты compact v2 (`runs=1`, 2 задачи)

| Label | Spec mode | Aggregate TPS |
|-------|-----------|--------------|
| `v2-konly-baseline-r1` | `ngram-mod 48/64/24` | `27.04` |
| `v2-konly-fattnvec-r1-ngram` | `ngram-mod 48/64/24` | `27.47` |
| `v2-konly-fattnvec-r2-ngram` | `ngram-mod 48/64/24` | `27.50` |
| `v2-konly-fattnvec-r1-none` | `none` | `27.40` |

Вывод фазы:
- патч не даёт прорыва, но показывает небольшой стабильный плюс относительно локального baseline-прогона;
- целевой порог `35 TPS` всё ещё далеко, нужен следующий цикл более глубоких kernel-изменений (не только selector tuning).

### Research Phase R35-05 (2026-05-09): deep FATTN softmax/fixup exp-path (kernel-only)

Цель: сделать более глубокий pass по вычислительным блокам FATTN (не selector), сфокусированный на softmax/fixup hot-path.

Изменение (экспериментальное, затем откат):
- `ggml/src/ggml-cuda/fattn-vec.cuh`: замена `expf` -> `__expf` в softmax-обновлении `KQ_max_scale`, `KQ_reg`, sink-path и финальном merge-scale;
- `ggml/src/ggml-cuda/fattn-tile.cuh`: замена `expf` -> `__expf` в KQ softmax (`KQ_max_scale`, `val`);
- `ggml/src/ggml-cuda/fattn-common.cuh`: замена `expf` -> `__expf` в stream-k fixup/combine scaling.

#### Результаты compact v2 (`runs=1`, 2 задачи)

| Label | Spec mode | Aggregate TPS |
|-------|-----------|--------------|
| `v2-konly-fattnvec-r2-ngram` | `ngram-mod 48/64/24` | `27.50` |
| `v2-konly-deepfattnexp-r1-ngram` | `ngram-mod 48/64/24` | `27.46` |
| `v2-konly-deepfattnexp-r1-none` | `none` | `26.74` |

Вывод фазы:
- deep exp-path замена не дала прироста в `ngram-mod` и дала заметный регресс в `spec none`;
- патч признан неуспешным и полностью откатан;
- рабочее состояние оставлено на kernel-only ветке с сохранённым улучшением из R35-04.

### Research Phase R35-06 (2026-05-09): serving-param exhaustive screen (no rebuild)

Цель: проверить 3 serving-level гипотезы из deep research документа, не требующие пересборки.
Базовый уровень в этой фазе: `build-rocm-exp`, `ctx=65536`, `b=4096`, `ub=512`, `kv=q4_0/q4_0`, `ngram-mod 48/24/64`, `np=1` → **~27.4–27.5 TPS**.

| Label | Гипотеза | ctx | ub | kv_k | kv_v | Aggregate TPS | Δ vs baseline |
|-------|----------|-----|-----|------|------|--------------|--------------|
| `v2-h1-ctx32k-ub512-r1` | Меньше KV IO: ctx=32K | 32768 | 512 | q4_0 | q4_0 | **27.55** | +0.1 (нейтр.) |
| `v2-h2-ctx65k-ub128-r1` | VEC path: ub=128 | 65536 | 128 | q4_0 | q4_0 | **25.61** | -1.9 (регрессия) |
| `v2-h3-kv-q8-ub512-r1` | Qwen KV qual: q8_0/q8_0 | 65536 | 512 | q8_0 | q8_0 | **26.74** | -0.7 (регрессия) |

#### Выводы фазы

- **H1 (ctx=32K)**: нейтрально. v2-задачи укладываются в 32K, реальный использованный KV-размер не меняется — bandwidth не является ограничивающим фактором для данной нагрузки.
- **H2 (ub=128)**: регрессия −1.9 TPS. «Гарантированный VEC path» хуже ub=512: при ngram-mod verification batches часто >128 токенов, что создаёт overhead из дополнительных kernel launches; меньший batching снижает GPU utilization.
- **H3 (q8_0 KV)**: регрессия −0.7 TPS. Удвоенная KV bandwidth → чуть медленнее, несмотря на более высокое качество кэша. Вывод противоположен гипотезе: q4_0 KV предпочтительнее на данной нагрузке.

#### Общий вывод по serving-param exploration

Пространство serving-параметров в текущем v2-профиле исчерпано:
- `ub`: 128 → регрессия; 256 → -0.25; **512 → оптимум**; 1024 → обрыв (-7 TPS TILE switch)
- `ctx`: 32K ≈ 65K → оба одинаковы (нагрузка не использует полный ctx)
- `kv type`: **q4_0 оптимум**; q8_0 → -0.7 TPS
- `parallel`: **p=1 оптимум**; p=2 → -1.7 TPS
- `spec`: ngram-mod ≈ none (для v2 кодовых задач acceptance rate низкий)

Потолок ~27.5 TPS является compute-bound ограничением линейных слоёв модели (weight loading / MMQ), не KV bandwidth и не selector kernel.
Для прорыва требуется: более лёгкая модель (IQ2/IQ3_XS), более быстрый MMQ kernel (RDNA4 MFMA tuning), или MTP с подходящей GGUF.

### Research Phase R35-07 (2026-05-09): MMQ RDNA4 cap (`x_max=96`) + rebuild

Цель: проверить RDNA4-специфичный MMQ тюнинг после полного rebuild ROCm контура.

Изменение:
- файл: `ggml/src/ggml-cuda/mmq.cuh`;
- функция: `get_mmq_x_max_host(const int cc)`;
- для `GGML_CUDA_CC_IS_RDNA4(cc)` установлен экспериментальный cap: `return 96` (вместо общего пути до `128`).

Сборка:
- после зависания терминала были обнаружены «осиротевшие» процессы `cmake/ninja/clang++`; они остановлены принудительно;
- rebuild выполнен командой `cmake --build build-rocm-exp --target llama-server -j 4`;
- новый бинарь: `build-rocm-exp/bin/llama-server.exe`.

#### Результат A/B (`runs=1`, compact v2, ngram-mod 48/24/64)

| Label | Конфиг | Aggregate TPS |
|-------|--------|--------------|
| baseline corridor | `ctx=65536, b=4096, ub=512, q4_0/q4_0` | `~27.4-27.5` |
| `v2-r35-mmqx96-r1` | `MMQ RDNA4 x_max=96` | **`25.77`** |

Вывод:
- текущий MMQ cap `x_max=96` для RDNA4 даёт **существенную регрессию** (`~ -1.7 TPS`);
- гипотеза не подтверждена, вариант не подходит для дальнейшего использования в baseline.

## Пост-мортем: почему патчи не пробили потолок ~27 TPS (cold-first)

Ниже сводный анализ по фазам R35-01..R35-07 для v2/v2-mini профиля.

1. Упор в compute-bound линейных слоёв (MMQ/weight loading), а не в KV/selector мелочи.

- Это подтверждено serving-перебором: `ctx 32K ~= 65K`, `q8_0 KV` даёт регрессию, `ub=512` остаётся лучшим для cold-first.
- Следствие: параметры, которые в основном двигают KV bandwidth, почти не меняют потолок.

2. Спекулятивный путь (`ngram-mod`) в cold-first v2 не даёт стабильного ускорения.

- В v2-кодовых задачах `spec none ~= ngram-mod` по aggregate.
- Большие warm-числа возникают на повторном проходе одинаковых задач (прогретый speculative context), но это другой режим, не cold-first headline.

3. Большинство проверенных патчей были «локально-микро», а не в главном bottleneck.

- GDN fast-exp, FATTN threshold widening, deep `expf -> __expf` не дали устойчивого выигрыша и/или дали регрессию в non-spec.
- Логика: даже если локально ускоряется отдельный участок, вклад в общий wall-time decode недостаточен для заметного роста aggregate TPS.

4. Часть направлений уже исчерпана и показала регресс заранее.

- `parallel=2`, `ub=1024`, RDNA4 MMQ `x_max=96`, расширения отдельных MMQ окон и draft-path с 9B draft-моделью — все дали отрицательный результат.
- Это указывает на структурный потолок текущей пары: модель 27B Q3_K_S + текущие kernel policies на RX 9070 XT.

5. Draft-model ветка упёрлась в стоимость самого draft decode.

- При высоком acceptance итоговый TPS всё равно резко падает из-за дорогой генерации draft-моделью.
- Значит bottleneck был не в acceptance, а в latency draft model per token.

Практический вывод:

- текущая ветка оптимизаций упёрлась в устойчивый cold-first коридор около `27.2-27.5 TPS`;
- для реального прорыва выше потолка нужны не микро-твики selector/exp, а более крупные изменения: новый MMQ/MFMA путь под RDNA4, более лёгкая target/draft модель, либо полноценный MTP-путь с совместимым MTP GGUF.

## Аудит окружения: что уже проверено вне build-патчей (2026-05-09)

Цель этого блока: зафиксировать, какие внешние ограничения уже видны по хосту и runtime, чтобы не переоценивать очередные kernel-правки.

### 1. Ключевые ROCm билды собраны почти в одинаковом базовом контуре

Проверенные CMakeCache для `build-rocm-exp`, `build-rocm-wmma`, `build-rocm-r35-c` показывают общий фундамент:

- `ROCm 7.1` (`clang/clang++` из `C:/Program Files/AMD/ROCm/7.1/bin`);
- `AMDGPU_TARGETS=gfx1201`;
- `Release` + `Ninja`;
- `GGML_HIP=ON`, `GGML_HIP_MMQ_MFMA=ON`, `GGML_HIP_NO_VMM=ON`.

Вывод: cold-first потолок нельзя объяснить тем, что один из основных билдов случайно собран «не под ту архитектуру» или на другом toolchain.

### 2. Runtime-путь у разных билдов почти одинаков

По server logs для `build-rocm-exp`, `build-rocm-wmma`, `build`:

- везде `offloaded 65/65 layers to GPU`;
- везде `graph nodes = 3849`, `graph splits = 2`;
- везде decode для cold-first задач держится около `~28.2-28.6 tok/s` на уровне отдельных задач;
- включение `rocWMMA FATTN` не дало отдельного качественного скачка.

Вывод: разные локальные бинарники в текущем workload в основном проходят через одинаковый практический runtime-контур.

### 3. На хосте уже видны платформенные ограничения Windows ROCm

Проверено на машине:

- Windows power plan: `Balanced`;
- GPU driver: `AMD Radeon RX 9070 XT`, driver `32.0.23033.1002` от `2026-03-09`;
- активная HIP runtime DLL: `C:/Windows/System32/amdhip64_7.dll`;
- `amdhip64_7.dll` из `System32` и из `C:/Program Files/AMD/ROCm/7.1/bin` имеют одинаковую version string `10.0.3665.0`, но разные размеры и разные SHA256;
- рядом с `llama-server.exe` в локальных `build*/bin` нет копий ROCm DLL, а текущие launcher paths в основном только prepend'ят `PATH`/`HIP_PATH`;
- `hipInfo`: `gfx1201`, `32 CU`, `clockRate 2460 MHz`, `memoryClockRate 1259 MHz`;
- `hipInfo`: `isLargeBar = 0`, `concurrentKernels = 1`, `cooperativeLaunch = 0`;
- server logs: `VMM: no`, `ROCm : NO_VMM = 1`.

Интерпретация:

- build менялся, но платформа исполнения оставалась одной и той же;
- на Windows это делает загрузку HIP runtime из `System32` фактическим default-path для текущих билдов, поэтому простой prepend `PATH` не гарантирует использование DLL из ROCm SDK;
- это делает рассинхрон `compiler/toolchain` vs `runtime DLL` правдоподобным кандидатом на скрытый performance ceiling;
- отсутствие VMM и Large BAR не доказывает текущий bottleneck само по себе, но указывает на менее гибкий runtime-контур, чем хотелось бы для агрессивного разгона;
- по коду backend `NO_VMM` в первую очередь отключает VMM memory pool / virtual-memory allocation path, а не переписывает основные MMQ/FATTN kernels; поэтому сам по себе `NO_VMM` скорее объясняет ограничения среды/allocator path, чем весь `~27 TPS` потолок;
- свойства `concurrentKernels = 1` и `cooperativeLaunch = 0` пока не выглядят главной причиной: в коде проекта нет явной логики, которая бы строила текущий decode hot path вокруг этих capability flags;
- `Balanced` power plan и Windows ROCm stack остаются валидными внешними кандидатами для A/B, прежде чем делать ещё 10 микро-патчей в kernel-слое.

### 4. Что это значит для цели `35 TPS cold-first`

На текущем наборе фактов наиболее вероятна такая картина:

- главный потолок формируется сочетанием `model size + quant format + RX 9070 XT + Windows ROCm runtime`;
- многие kernel-патчи не попадают в основной wall-time, потому что runtime и workload остаются почти неизменными;
- дальнейший поиск нужно вести не только в коде, а в платформенных A/B:
  - power plan `Balanced` vs `High performance`;
  - BIOS/driver проверка ReBAR / Smart Access Memory;
  - чистый runtime A/B Windows vs Linux на том же железе;
  - проверка, не вносит ли заметный штраф системная HIP DLL/driver pair.

Текущий рабочий вывод: до смены хотя бы одного существенного внешнего фактора вероятность получить `35 TPS cold-first` только build-патчами выглядит низкой.

### 5. Быстрый A/B: app-local `amdhip64_7.dll` рядом с `llama-server.exe`

Был проверен прямой эксперимент: принудительно положить `amdhip64_7.dll` из `C:/Program Files/AMD/ROCm/7.1/bin` рядом с `build-rocm-exp/bin/llama-server.exe`, чтобы уйти от implicit загрузки HIP runtime из `System32`.

Результат:

- `v2mini-local-hipdll-r1` (`v2-mini`, cold-first, `ctx=65536`, `b=4096`, `ub=512`, `q4_0/q4_0`, `ngram-mod`) дал **`26.21 TPS`**;
- это хуже обычного cold-first коридора `~27.4-27.5 TPS`.

Вывод:

- простая подкладка только `amdhip64_7.dll` рядом с бинарём **не является выигрышным путём**;
- более того, такой частичный override может создавать смешанный runtime-контур, поэтому он не подтверждает гипотезу «локальная DLL = автоматически быстрее»;
- практическое решение: этот путь считать **проверенным и регрессивным**, не держать его как активную оптимизацию.

## Продолжение cold-first цикла (2026-05-09, вечер)

Профиль для всех сравнений ниже:

- `--tasks v2-mini --runs 1 --no-v2-prime-pass`
- `-c 65536 -b 4096`
- `--cache-type-k q4_0 --cache-type-v q4_0`
- `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`

### 1. Бинарный A/B (build-rocm-exp vs свежий build-rocm-compare)

Для честного сравнения был поднят отдельный каталог `build-rocm-compare` с ROCm clang 7.1 и Ninja.

Ключевой технический момент:

- при первом configure OpenMP подцепил `C:/Strawberry/c/lib/libgomp.dll.a` и ломал линковку (`__kmpc_*`);
- после выравнивания с рабочим профилем (`-DGGML_OPENMP=OFF`) сборка прошла и тест стал валидным.

Результат:

- `v2mini-buildrocmcompare-r1`: **`26.65 TPS`**
- против `v2mini-postrollback-streamk-r1`: **`26.72 TPS`**
- дополнительная проверка `build-rocm-vec`: `v2mini-buildrocmvec-r1` = **`25.45 TPS`**

Вывод: отдельный свежий ROCm-билд не дал прироста; код/ядро остаются в том же практическом коридоре.

### 2. Runtime sweep по `ubatch`

На `build-rocm-exp`:

| Label | UBatch | Aggregate TPS | Вывод |
| --- | ---: | ---: | --- |
| `v2mini-ub256-r1` | `256` | `25.50` | regress |
| `v2mini-ub384-r1` | `384` | `25.63` | regress |
| `v2mini-ub640-r1` | `640` | `26.65` | около baseline, без выигрыша |

Вывод: для текущего cold-first профиля `ub=512` остаётся практическим опорным значением.

### 3. Runtime sweep по потокам (host feeder)

Проверка `--threads 16 --threads-batch 16`:

| Label | UBatch | Aggregate TPS | Вывод |
| --- | ---: | ---: | --- |
| `v2mini-threads16-r1` | `512` | `26.78` | небольшой плюс в шуме |
| `v2mini-threads16-ub512-r1` | `512` | `26.67` | без подтверждения плюса |
| `v2mini-threads16-ub640-r1` | `640` | `26.62` | без выигрыша |

Итог: устойчивого выигрыша от 16/16 потоков не подтверждено.

### 4. MTP-путь с локальным MTP GGUF

Проверен запуск:

- модель: `models/Qwen3.6-27B-IQ3_M-mtp.gguf`
- `--spec-type mtp`

Результат:

- `v2mini-mtp-main-r1`: **`4.00 TPS`** (сильный regress)

Диагностика по server log:

- MTP действительно активирован (`set_mtp: MTP draft head registered`);
- acceptance высокий (`#acc drafts` высокий), но `dur(g)` (generation stage) огромный;
- wall-time уходит в MTP generation path, поэтому общая скорость резко ниже базового ngram-mod.

Практический вывод:

- в текущем локальном сочетании модель/квант/железо MTP-путь **непригоден** как ускорение;
- держим его как проверенный тупик до появления более лёгкого/лучше совместимого MTP-конфига.

### 5. Альтернативные ngram-режимы (без изменений кода)

Проверены на том же профиле:

| Label | Spec type | Aggregate TPS | Наблюдение |
| --- | --- | ---: | --- |
| `v2mini-ngramsimple-r1` | `ngram-simple` | `26.67` | около baseline |
| `v2mini-ngramk4v-r1` | `ngram-map-k4v` | `26.56` | небольшой regress |

По server logs:

- `ngram-simple`: drafts есть, но мало (`#gen drafts = 5`, `#acc tokens = 30` суммарно);
- `ngram-map-k4v`: drafts почти не активируются (`#gen drafts = 1`, `#acc tokens = 6`);
- основная decode-скорость остаётся близкой к `~27.7-27.9 tok/s` на задачу, поэтому общий aggregate почти не меняется.

Итог: переключение между ngram-режимами само по себе не даёт прорыва для текущего cold-first workload.

### 6. MMQ host-policy pass для Q3_K-heavy decode

Текущая модель `Qwen3.6-27B-Q3_K_S.gguf` по server log почти целиком упирается в `q3_K` тензоры (`353` tensors), поэтому следующим шагом был проверен более структурный MMQ-тюнинг в `ggml/src/ggml-cuda/mmq.cuh`.

#### A. RDNA4 `granularity=16`

Изменение:

- для RDNA4 в MMQ зафиксирован более мелкий `granularity=16` вместо перехода на `32` при `mmq_x >= 128`.

Результаты:

| Label | Spec mode | Aggregate TPS |
| --- | --- | ---: |
| `v2mini-mmq-gran16-r1` | `ngram-mod 48/64/24` | `26.89` |
| `v2mini-mmq-gran16-r2` | `ngram-mod 48/64/24` | `26.94` |
| `v2mini-mmq-gran16-r3` | `ngram-mod 48/64/24` | `26.88` |
| `v2mini-mmq-gran16-specnone-r1` | `none` | `26.89` |
| `v2mini-mmq-gran16-specnone-r2` | `none` | `26.94` |
| `v2mini-mmq-gran16-specnone-r3` | `none` | `26.92` |

Интерпретация:

- прирост небольшой, но воспроизвёлся и с `ngram-mod`, и с `spec none`;
- три независимых cold-first прогона с `ngram-mod` дали combined aggregate `26.9053 TPS`, то есть патч выглядит повторяемым;
- три независимых cold-first прогона с `spec none` дали combined aggregate `26.9184 TPS`, то есть `spec none` на этом профиле как минимум не хуже `ngram-mod`;
- это похоже на маленький decode/MMQ gain, а не на speculative-шум;
- патч оставлен как **текущий лучший малый кандидат** в рабочем дереве.

Практический вывод по runtime mode:

- для обычного `Qwen3.6-27B-Q3_K_S` на `ctx=65536, b=4096, ub=512` после MMQ `gran16` режим `spec none` выглядит наиболее консервативным default;
- разница против `ngram-mod` минимальна, но `spec none` чуть лучше по combined 3-run aggregate и не зависит от speculative counters.

#### B. Дополнительный RDNA4 bundle: `y=64` + `4 warps`

Поверх `gran16` был проверен ещё один более агрессивный MMQ host-policy bundle:

- `get_mmq_y_* = 64` для RDNA4;
- `mmq_get_nwarps_* = 4` для RDNA4.

Результаты:

| Label | Spec mode | Aggregate TPS |
| --- | --- | ---: |
| `v2mini-mmq-gran16-y64-r1` | `ngram-mod 48/64/24` | `26.91` |
| `v2mini-mmq-gran16-y64-specnone-r1` | `none` | `26.86` |

Вывод:

- отдельной ценности поверх `gran16` этот слой не показал;
- improvement в `ngram-mod` слишком мал, а на `spec none` он уже не подтверждается;
- bundle `y=64 + 4 warps` **откачен**, чтобы оставить в дереве только более чистый `gran16`-патч.

#### C. RDNA4 selector: `always-MMQ` поверх `gran16`

На том же 65K cold-first профиле был отдельно проверен старый сильный кандидат с 32K-сессии: принудительный `always-MMQ` для RDNA4 в `ggml_cuda_should_use_mmq()`.

Результат:

| Label | Spec mode | Aggregate TPS |
| --- | --- | ---: |
| `v2mini-mmq-gran16-always-r1` | `ngram-mod 48/64/24` | `26.01` |

Вывод:

- на 65K cold-first этот путь даёт сильный regress;
- старый выигрыш `always-MMQ` на 32K не переносится напрямую на текущий профиль;
- правка **откачена**, в рабочем дереве оставлен только `gran16`.

#### D. Мягкий MMQ cap: `x_max=112` поверх `gran16`

После подтверждения `gran16` был проверен более мягкий соседний cap для RDNA4:

- `get_mmq_x_max_{host,device} = 112` вместо `128`.

Результат:

| Label | Spec mode | Aggregate TPS |
| --- | --- | ---: |
| `v2mini-mmq-gran16-x112-r1` | `ngram-mod 48/64/24` | `26.80` |

Вывод:

- мягкий cap `112` всё равно хуже подтверждённого `gran16` baseline;
- этот путь **откачен**.

### UBatch cliff study for prompt-heavy `v2-mini` (2026-05-10)

Профиль:

- `--tasks v2-mini --runs 1 --no-v2-prime-pass`
- `--ctx-size 12288 -b 6144`
- `--cache-type-k q4_0 --cache-type-v q4_0`
- `--spec-type none --cache-ram 0 --ctx-checkpoints 0`
- repo-snapshot prompt lane

Последние точки на `build-rocm-vec`:

| Label | UBatch | Aggregate TPS | Статус |
| --- | ---: | ---: | --- |
| `trace-vec-b6144-ub784-none` | `784` | `10.0734` | fast |
| `trace-vec-b6144-ub800-none-r2` | `800` | `9.9074` | fast |
| `trace-vec-b6144-ub832-none-r2` | `832` | `3.6181` | cliff |

Что показал трассировочный лог:

- в `ggml/src/ggml-cuda/gated_delta_net.cu` RDNA4 prefill идёт через chunked path при `n_tokens >= 128`;
- при `n_tokens > 256` launcher выбирает `chunk_size = 128`, иначе `96`;
- для fast точек `784/800` trace показывает final chunk `16/32`, а для slow точки `832` final chunk становится `64`;
- это выглядит как локальный tail-chunk threshold в RDNA4 `Gated Delta Net` prefill, а не как общий убыток от самого `ubatch`.

Рабочая гипотеза на следующий цикл:

- избегать конфигураций, где `n_tokens % 128 == 64` в этом lane;
- проверить ещё несколько точек вокруг границы (`848`, `864`, `880`) и посмотреть, сохраняется ли провал именно на tail `64`;
- если гипотеза подтвердится, рассмотреть alignment-aware `ubatch` policy или локальную правку chunking в RDNA4 prefill.

### Short lane: `v2-review` (only `v2_code_review`) for low-noise prompt-eval checks (2026-05-10)

Чтобы сократить длительность прогона и уменьшить шум от смешивания нескольких задач,
в `scripts/agent_workload_bench.py` добавлен режим:

- `--tasks v2-review` (только `v2_code_review`).

Быстрый шаблон запуска:

```powershell
python scripts/agent_workload_bench.py --label promptfocus-v2review-<tag> --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks v2-review --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size <UB> --cache-type-k q4_0 --cache-type-v q4_0 --server-extra "--spec-type none" --real-context-mode repo-snapshot --real-context-chars 21872 --no-reuse --background-server-policy ignore --no-v2-prime-pass --no-disable-thinking --max-tokens 120
```

Подтверждение на `runs=3` (одинаковый lane, `spec=none`, no-reuse):

| Label | UBatch | Aggregate TPS | prompt_eval_tps mean | prompt_eval_ms mean |
| --- | ---: | ---: | ---: | ---: |
| `promptfocus-v2review-ub128-r3` | `128` | `7.8540` | `736.65` | `10900.92` |
| `promptfocus-v2review-ub256-r3` | `256` | `8.0202` | `758.95` | `10582.62` |
| `promptfocus-v2review-ub512-r1` | `512` | `3.7187` | `288.93` | `27791.89` |
| `promptfocus-v2review-ub824-r3` | `824` | `3.8020` | `297.17` | `27025.85` |
| `promptfocus-v2review-ub832-r3` | `832` | `3.6944` | `287.32` | `27949.02` |

Вывод:

- в этом конкретном prompt-heavy lane высокие `ubatch` сейчас вредят prefill: `128/256` существенно быстрее `512+`;
- даже в коротком однотасковом режиме сохраняется просадка `ub832` против `ub824`;
- фокус оптимизации остаётся на prefill/prompt-eval path;
- для быстрых A/B итераций по prompt-eval использовать `v2-review` как дефолтный short lane.

Трассировочные подтверждения (текущий `build-rocm-vec`):

- `GGML_TRACE_GDN_PATH=1`:
  - `ub256`: `launch_gated_delta_net ... n_tokens=256 ... chunk_size=96`;
  - `ub512`: `launch_gated_delta_net ... n_tokens=512 ... chunk_size=128`.
- `GGML_TRACE_FATTN_SELECTED=1`:
  - `ub256`: много вызовов `Q1=256`, `selected=wmma_f16`;
  - `ub512`: вызовы с `Q1=512`, также `selected=wmma_f16`, но итоговый prompt eval резко хуже.

Проверка гипотезы «виноват только chunk_size=128»:

- принудительный override `GGML_GDN_CHUNK_SIZE=96` при `ub512` (`promptfocus-v2review-ub512-ch96-r1`) не восстановил скорость (`aggregate ~3.71 TPS`).
- значит, деградация связана не только с `chunk_size`, а с более широким kernel-route/shape поведением prefill при больших `n_tokens`.

### Narrow-band ubatch sweep (`<=256`) on `v2-review` (2026-05-10)

После подтверждения, что выше `ub=256` в этом lane смысла нет, был сделан короткий sweep только по low-ubatch зоне (`runs=1`):

| Label | UBatch | Aggregate TPS |
| --- | ---: | ---: |
| `promptfocus-v2review-ub176-r1-micro` | `176` | `8.07` |
| `promptfocus-v2review-ub184-r1-micro` | `184` | `8.23` |
| `promptfocus-v2review-ub192-r1-micro` | `192` | `8.47` |
| `promptfocus-v2review-ub200-r1-micro` | `200` | `6.81` |
| `promptfocus-v2review-ub208-r1-micro` | `208` | `6.92` |
| `promptfocus-v2review-ub216-r1-micro2` | `216` | `7.14` |
| `promptfocus-v2review-ub224-r1-micro2` | `224` | `7.26` |
| `promptfocus-v2review-ub232-r1-micro2` | `232` | `7.42` |
| `promptfocus-v2review-ub240-r1-micro2` | `240` | `7.55` |
| `promptfocus-v2review-ub248-r1-micro2` | `248` | `7.72` |
| `promptfocus-v2review-ub256-r1-micro2` | `256` | `7.86` |

Тонкий sweep вокруг пика (`188..198`) показал резкую границу:

| Label | UBatch | Aggregate TPS |
| --- | ---: | ---: |
| `promptfocus-v2review-ub190-r1-fine` | `190` | `8.38` |
| `promptfocus-v2review-ub192-r1-fine` | `192` | `8.46` |
| `promptfocus-v2review-ub194-r1-fine` | `194` | `6.67` |
| `promptfocus-v2review-ub196-r1-fine` | `196` | `6.72` |
| `promptfocus-v2review-ub198-r1-fine` | `198` | `6.75` |

Трассировочный A/B (`GGML_TRACE_GDN_PATH=1 GGML_TRACE_FATTN_SELECTED=1`):

- `ub192` (`promptfocus-v2review-ub192-trace-r1`): `aggregate 8.47`, `prompt_eval_tps 821.08`, `decode_eval_tps 27.53`.
- `ub194` (`promptfocus-v2review-ub194-trace-r1`): `aggregate 6.66`, `prompt_eval_tps 591.25`, `decode_eval_tps 27.51`.
- decode почти неизменен; просадка целиком в prefill.
- histogram `n_tokens` в GDN trace:
  - `ub192`: `{192, 158, 2, 1}`;
  - `ub194`: `{194, 140, 130, 2, 1}`.

Проверка гипотезы `GDN chunk_size` на лучшей точке (`ub192`):

- `GGML_GDN_CHUNK_SIZE={64,80,96,128}` дали `8.46-8.47 TPS` (разница < 1%).
- по правилу трека это **не прогресс**, гипотеза закрыта без дополнительных re-check.
- `LLAMA_FUSED_GDN_CH=0` / отключение chunked prefill на `ub192` не является рабочим обходом: `nonmtp-ub192-gdnch-off-20260511-r1` завис на первом prompt batch (`6144/8030`) сразу после `prompt processing progress`, поэтому эксперимент откатан и помечен как no-go.

Дополнительные no-go проверки на той же точке (`ctx=12288`, `b=6144` unless noted, `ub192`, `spec=none`, no-reuse, `runs=1`):

| Label | Проверка | Aggregate TPS | Вывод |
| --- | --- | ---: | --- |
| `nonmtp-ub192-nographs-noreuse-20260511-r1` | `GGML_CUDA_DISABLE_GRAPHS=1` | `8.53` | <1% к лучшему `8.47`, не считается |
| `nonmtp-ub192-offloadmin1-noreuse-20260511-r1` | `GGML_OP_OFFLOAD_MIN_BATCH=1` | `8.47` | нет прироста |
| `nonmtp-ub192-nocudafusion-noreuse-20260511-r1` | `GGML_CUDA_DISABLE_FUSION=1` | `8.38` | регрессия |
| `nonmtp-ub192-t16tb16-noreuse-20260511-r1` | `--threads 16 --threads-batch 16` | `8.47` | CPU threads не bottleneck |
| `nonmtp-ub192-backendsampling-noreuse-20260511-r1` | `--backend-sampling` | `8.43` | backend sampling не окупает overhead |
| `nonmtp-ub192-b3072-noreuse-20260511-r1` | `b=3072` | `8.44` | хуже `b=6144` |
| `nonmtp-ub192-b2048-noreuse-20260511-r1` | `b=2048` | `8.45` | хуже `b=6144` |
| `nonmtp-compare-ub192-noreuse-20260511-r1` | `build-rocm-compare` | `8.10` | готовая compare-сборка хуже |
| `nonmtp-exp-ub192-noreuse-20260511-r1` | `build-rocm-exp` | `8.09` | готовая exp-сборка хуже |
| `nonmtp-ub192-ngrammod-noreuse-20260511-r1` | `--spec-type ngram-mod` | `8.46` | ngram-mod сгенерировал `0` draft tokens, ускорения нет |
| `nonmtp-ub192-kvq8-noreuse-20260511-r1` | `--cache-type-k q8_0 --cache-type-v q8_0` | `8.41` | prefill тот же, decode хуже (`26.99 tok/s`) |

Первые shape-planner и outer-batch проверки после добавления `LLAMA_UBATCH_SPLIT_POLICY=tail-avoid`:

| Label | Изменение | Aggregate TPS | Вывод |
| --- | --- | ---: | --- |
| `nonmtp-shapeplan-ub256-pref192-noreuse-20260511-r1` | `-ub 256`, `LLAMA_UBATCH_SHAPE_PREFERRED=192` | `8.44` | planner реально дал chunks `192...192,158`, восстановив `ub256` с прежних `7.86`, но peak `ub192` не побил |
| `nonmtp-ub192-b8192-noreuse-20260511-r1` | `b=8192`, `ub=192` | `8.47` | один outer prompt batch вместо `6144+1886` почти не меняет wall; boundary не bottleneck |

### P1 shape-score boundary gate (`v2-review`, 2026-05-11)

После внедрения `shape-score` planner (`src/llama-batch.cpp`) был выполнен полный gate на active lane:

- `ctx=12288`, `b=6144`, `q4_0/q4_0`, `spec=none`, no-reuse, `--no-disable-thinking`.

Screening (`runs=1`):

| Label | UBatch | Policy | Aggregate TPS |
| --- | ---: | --- | ---: |
| `p1-gate-20260511-174248-base-ub192-r1` | `192` | off | `8.51` |
| `p1-gate-20260511-174248-shape-ub190-r1` | `190` | shape-score | `8.43` |
| `p1-gate-20260511-174248-shape-ub192-r1` | `192` | shape-score | `8.54` |
| `p1-gate-20260511-174248-shape-ub194-r1` | `194` | shape-score | `8.53` |
| `p1-gate-20260511-174248-shape-ub196-r1` | `196` | shape-score | `8.54` |
| `p1-gate-20260511-174521-base-ub194-r1` | `194` | off | `6.71` |

Confirmation (`runs=3`):

| Label | UBatch | Policy | Aggregate TPS | TPS stdev |
| --- | ---: | --- | ---: | ---: |
| `p1-confirm-20260511-174606-base-ub194-r3` | `194` | off | `6.83` | `0.0663` |
| `p1-confirm-20260511-174606-shape-ub194-r3` | `194` | shape-score | `8.52` | `0.0064` |
| `p1-confirm-20260511-174606-base-ub192-r3` | `192` | off | `8.51` | `0.0009` |

Итоговые дельты (по diagnostics + CSV):

- shape-score `ub194` vs baseline `ub194`:
  - aggregate TPS: `+24.73%`
  - prompt_eval_ms: `-26.42%`
  - decode_eval_ms: `+0.10%` (в пределах шума)
- shape-score `ub194` vs baseline `ub192`:
  - aggregate TPS: `+0.08%`
  - prompt_eval_ms: `-0.11%`
  - decode_eval_ms: `-0.10%`

Verdict:

- boundary cliff на `ub194` воспроизводимо снят под `shape-score` без decode-regression;
- throughput `ub194` возвращён в corridor `ub192` класса;
- изменение оставлено в дереве как env-guarded policy.

Timing trace после добавления `LLAMA_UBATCH_TIMING`:

- `nonmtp-ub192-timing-noreuse-20260511-r1`: async trace, `8.44 TPS`; build/alloc/input overhead на prompt chunks меньше `~1.5 ms`, но `compute_call` асинхронный и не показывает полную GPU стоимость.
- `nonmtp-ub192-timing-sync32-noreuse-20260511-r1`: diagnostic-only (`LLAMA_UBATCH_TIMING_SYNC=1`, `max_tokens=32`, TPS не сравнивать). Средние sync timings: prompt `n_tokens=192` стоит `~232-240 ms` total на chunk, decode `n_tokens=1` стоит `~36 ms` на token. Host-side graph overhead не является bottleneck; следующий реальный рычаг — GDN/FATTN/MMQ device kernels или model-graph reshape вокруг них.

Reduced HIP/FlashAttention build corridor после `amdgcn-link` blocker:

| Label | Проверка | Aggregate TPS | Вывод |
| --- | --- | ---: | --- |
| `nonmtp-fa-reduced-ub192-noreuse-20260511-r1` | `build-rocm-fa-reduced`, `GGML_HIP_QWEN_FA_REDUCED=ON`, `GGML_OPENMP=OFF` | `8.46` | reduced dispatcher проходит активную Qwen/RDNA4 lane, но сам по себе не ускоряет |
| `nonmtp-fa-reduced-forcevec-ub192-mt32-20260511-r1` | `GGML_QWEN_FA_REDUCED_FORCE=vec`, diagnostic `max_tokens=32` | diagnostic only | prompt eval упал `820 -> 580 tok/s`, force-vec для `Q1=192` закрыт |
| `nonmtp-fa-reduced-forcewmma-ub192-mt32-20260511-r1` | `GGML_QWEN_FA_REDUCED_FORCE=wmma_f16`, diagnostic `max_tokens=32` | diagnostic only | prompt `823 tok/s`, decode `27.51 tok/s`; tiny decode через WMMA не лучше baseline |

Вывод по reduced mode:

- `GGML_HIP_QWEN_FA_REDUCED=ON` решает практический build blocker для дальнейших FATTN/GDN A/B патчей: heavy `fattn.cu`, tile/MMA dispatcher и template instances исключены, вместо них используется host-only reduced dispatcher.
- Reduced dispatcher имеет ручку `GGML_QWEN_FA_REDUCED_FORCE=vec|wmma_f16` для smoke A/B FATTN selector без тяжелого `fattn.cu` relink.
- Fresh reduced build на Windows/ROCm потребовал `-DGGML_OPENMP=OFF`, иначе link `ggml-cpu.dll` падает на `__kmpc_*` symbols.
- Результаты из этого build можно использовать для smoke/A-B проверки kernel hypotheses, но финальные speed claims лучше подтверждать на обычном ROCm build после переноса удачной правки.

MMQ/MMVQ follow-up:

- `GGML_TRACE_MMQ_PATH=1` на reduced build (`nonmtp-fa-reduced-mmqtrace-ub192-mt8-20260511-r1`) дал `16674` MMQ route lines, все в prefill: `type=11/Q3_K ncols=192 xbest=96 tiles=2`, `type=12/Q4_K ncols=192 xbest=96 tiles=2`, плюс tail `ncols=158 xbest=80 tiles=2`.
- Decode не идёт через MMQ trace; активный decode matvec path — `mmvq.cu`.
- Попытка добавить MMVQ trace/Q3_K nwarps knob упёрлась в `amdgcn-link command failed due to signal` на `mmvq.cu`.
- Попытка ограничить MMVQ switch до Qwen tensor types (`q3_K/q4_K/q6_K`) тоже не прошла: source-specific `mmvq.cu` compile всё равно падал в `amdgcn-link`. Эксперимент откатан, чтобы не оставлять несобираемый source state.

P2 Stage A+B+C+D (MMVQ dispatch split + observability/tuning scaffold, 2026-05-11):

- Stage A: публичные host entrypoints вынесены из `mmvq.cu` в новый `mmvq-dispatch.cu`.
- Stage B: type switch (`ggml_cuda_mmvq_switch_type`) перенесён в lightweight `mmvq-dispatch.cu`, а `mmvq.cu` экспортирует per-type entrypoints.
- Stage C: type routing разделён на `mmvq-kernels-qwen.cu` (`Q3_K/Q4_K/Q6_K`) и `mmvq-kernels-rest.cu` (остальные типы).
- Stage D: добавлены env-gated MMVQ observability/tuning hooks:
  - `GGML_TRACE_MMVQ_PATH=1` (route trace `qwen-hot/rest` с type и shape полями)
  - `GGML_TRACE_MMVQ_SMALL_K=1` (small_k decision trace)
  - `GGML_MMVQ_QWEN_FORCE_SMALL_K=1` / `GGML_MMVQ_QWEN_DISABLE_SMALL_K=1` (RDNA4 Qwen-hot override, default unchanged)
- Normal ROCm gate (`build-rocm-vec`, target `llama-server`) прошёл после переконфигурации.
- Reduced ROCm gate (`build-rocm-fa-reduced`, `GGML_HIP_QWEN_FA_REDUCED=ON`, `GGML_OPENMP=OFF`) также прошёл.
- Повторные инкрементальные touch+rebuild циклы (`mmvq.cu`, `mmvq-dispatch.cu`, `mmvq-kernels-qwen.cu`, `mmvq-kernels-rest.cu`) прошли без `amdgcn-link ... signal`.
- Runtime smoke на активной lane:
  - `p2-stageA-smoke-20260511-181905-ub192-r1`: `8.54 TPS`
  - `p2-stageB-smoke-20260511-182335-ub192-r1`: `8.54 TPS`
  - `p2-stageC-smoke-20260511-182726-ub192-r1`: `8.54 TPS`
  - `p2-stageC-reduced-smoke-20260511-183047-ub192-r1`: `8.54 TPS`
  - `p2-active-lane-posthooks-20260511-184542-ub192-r1`: `8.55 TPS`
  - `p2-reduced-posthooks-20260511-184624-ub192-r1`: `8.55 TPS`
  - Все результаты остаются в `ub192` corridor, явной default-regression на scaffold этапе не видно.

Stage D diagnostics:

- Route trace sample (`p2-trace-route-20260511-183846-ub192-r1`) подтвердил рабочий MMVQ маршрутный лог: `qwen-hot=1077`, `rest=0` (для этой Qwen lane).
- Force trace sample (`p2-trace-force-smallk-20260511-184219-ub192-r1`) подтвердил, что `GGML_MMVQ_QWEN_FORCE_SMALL_K=1` реально переключает `small_k=1` в Qwen-hot вызовах (`680` lines; baseline trace had `680` lines with `small_k=0`).
- Decode-biased lane (`ctx=12288`, no-reuse, no real-context, `max_tokens=256`):
  - runs=1: base `26.84`, force `27.09`, disable `26.88` TPS.
  - runs=3 confirm: base `26.8355` vs force `27.0066` TPS (`+0.64%`), decode_eval_tps `28.6767 -> 28.8767` (`+0.70%`).
  - эффект умеренный; default policy не менялась.

P3 theory fanout check (dry-run explain, 2026-05-11):

- Команда для всех проверок: `cmake --build <build-dir> --target llama-server -- -d explain -n`.
- `touch fattn.cu`:
  - normal (`build-rocm-vec`): rebuild `fattn.cu.obj` + relink chain (`7` steps).
  - reduced (`build-rocm-fa-reduced`): `ninja: no work to do`.
- `touch mmvq.cu`:
  - normal (`build-rocm-vec`): rebuild `mmvq.cu.obj` + relink chain (`7` steps).
  - reduced (`build-rocm-fa-reduced`): rebuild `mmvq.cu.obj` + relink chain (`7` steps).
- Теоретический вывод: reduced corridor уже снимает FATTN-side build pressure, но не снимает MMVQ-side pressure; MMVQ-focused corridor остаётся предметом P3 implementation.

Artifacts:

- `build_logs/agent-workload/p3-dryrun-normal-fattn.txt`
- `build_logs/agent-workload/p3-dryrun-reduced-fattn.txt`
- `build_logs/agent-workload/p3-dryrun-normal-mmvq.txt`
- `build_logs/agent-workload/p3-dryrun-reduced-mmvq.txt`

P3 implementation build gates (2026-05-11):

- Build-system implementation landed for:
  - centralized HIP source bundle assembly,
  - `GGML_HIP_EXPERIMENT_PROFILE` (`default`, `qwen-fa-reduced`, `mmvq-focused`),
  - Windows HIP compiler fail-fast guard (`clang++/hipcc` required).
- Configure gates passed for all three profiles:
  - `build-rocm-vec` (`default`)
  - `build-rocm-fa-reduced` (`qwen-fa-reduced`)
  - `build-rocm-mmvq-focused` (`mmvq-focused`)
- Build gate passed for all three profiles: `llama-server` linked successfully.
- Guard test passed: intentional bad configure with Strawberry/GNU now fails early with:
  - `GGML_HIP on Windows requires ROCm clang++ or hipcc as CMAKE_CXX_COMPILER`.

Artifacts:

- `build_logs/agent-workload/p3-implementation-build-gates.txt`
- `build_logs/agent-workload/p3-guard-bad-config.txt`

P3 runtime closure checks (2026-05-11):

- Active lane (`v2-review`, `repo-snapshot chars=21872`, `ctx=12288`, `b/ub=6144/192`, no-reuse):
  - `p3-close-default-20260511-r1`: **8.54 TPS** (pass)
  - `p3-close-reduced-20260511-r1`: **8.55 TPS** (pass)
  - `p3-close-mmvq-focused-20260511-r2`: **request timeout** (`TimeoutError('timed out')`), server log stops at prompt progress `6144/8030`.
- Short decode-biased sanity (`tasks=quick`, no real-context, same ctx/b/ub, `max_tokens=64`):
  - `p3-close-default-quick-20260511-r1`: **26.57 TPS**
  - `p3-close-reduced-quick-20260511-r1`: **26.59 TPS**
  - `p3-close-mmvq-focused-quick-20260511-r1`: **17.56 TPS** (major regression vs default/reduced in short lane)
- Additional check (`p3-close-mmvq-focused-sanity-20260511-r1`): with `--flash-attn off` and KV `q4_0/q4_0`, context init fails with `V cache quantization requires flash_attn`.

P3 closure interpretation:

- P3 is closed for build-pressure workflow objective.
- `mmvq-focused` is kept as a narrow debug/build profile only and is not promoted to active prompt-heavy runtime lane.

Artifacts:

- `build_logs/agent-workload/p3-close-default-20260511-r1.csv`
- `build_logs/agent-workload/p3-close-reduced-20260511-r1.csv`
- `build_logs/agent-workload/p3-close-mmvq-focused-20260511-r2.csv`
- `build_logs/agent-workload/p3-close-default-quick-20260511-r1.csv`
- `build_logs/agent-workload/p3-close-reduced-quick-20260511-r1.csv`
- `build_logs/agent-workload/p3-close-mmvq-focused-quick-20260511-r1.csv`
- `build_logs/agent-workload/p3-close-mmvq-focused-20260511-r2.diagnostics.md`
- `build_logs/agent-workload/p3-close-mmvq-focused-sanity-20260511-r1.server.log`

---

## Dual TG/PP Compute Scheduler (2026-05-10) ✅

### Проблема

При `ubatch=512` `ggml_backend_sched` выделял compute buffer **495 MiB** (под PP-граф максимального размера). Во время decode (1 токен за шаг) GPU тратил cache bandwidth на весь этот буфер, хотя реально нужно только ~7 MiB.

Результат: `ub=512` давал **~19-25 TPS** против **~25 TPS** при `ub=128`.

### Решение

Реализован dual TG/PP compute scheduler в `src/llama-context.cpp`:

- **PP scheduler** (`sched`): стандартный, sized для полного ubatch — 495 MiB. Используется при prefill (`n_tokens > 1`).
- **TG scheduler** (`sched_tg`): новый, sized для 1-токенного графа — **6.95 MiB**. Используется при decode (`n_tokens == 1`).
- Переключение происходит автоматически в `process_ubatch()` при смене режима.
- Оба scheduler'а имеют отдельный кэш графа (`gf_res_prev` / `gf_res_prev_tg`).

**Изменённые файлы:**
- `src/llama-context.h` — поля `sched_tg`, `sched_is_tg`, `gf_res_prev_tg`
- `src/llama-context.cpp` — `sched_reserve()`, `process_ubatch()`, `synchronize()`, destructor
- `gui/model_presets.json` — пресет обновлён: `ubatch_size: 128 → 512`
- `gui/llama_gui.py` — ROCm fallback поиск по `build-rocm-wmma`, `build-rocm-vec`

### Результаты (`build-rocm-wmma`, `Qwen3.6-27B-Q3_K_S.gguf`, `ctx=65536`, `q4_0 KV`, `ngram-mod`)

| Label | Dual-sched | UBatch | Wall TPS | Runs |
|---|:---:|---:|---:|---:|
| `nodual-ub128-wmma` | нет | 128 | 25.17 | 1 |
| `nodual-ub512-wmma` | нет | 512 | 19.58 | 1 |
| `dual-sched-ub512-wmma` | **да** | 512 | **32.16** | 3 |
| `gui-dual-sched-ub512-wmma` | **да** | 512 | **29.97** | 3 (через GUI live) |

**+64% к старому `ub=512`**, **+27% к лучшему `ub=128` без dual-sched.**

TG compute buffer: 6.95 MiB вместо 495 MiB → GPU cache pressure в decode фазе снята.

### Воспроизведение через GUI

1. Launch Server → Backend: **ROCm** → подхватит `build-rocm-wmma` автоматически
2. Apply Preset → `Qwen3.6-27B-Q3_K_S.gguf` → `ub=512, b=4096, ctx=65536, q4_0, ngram-mod`
3. Start Server → ожидаемый decode: **~30 TPS**

## Dual TG/PP Compute Scheduler (2026-05-10)

### Мотивация

При больших `ubatch_size` (например, 512) `ggml_backend_sched` выделял compute buffer под максимальный PP-граф (495 MiB для `ub=512`). В режиме TG (decode, n_tokens=1) этот буфер оставался занятым, создавая GPU memory pressure при каждом шаге decode.

Гипотеза: отдельный TG-scheduler с маленьким compute buffer (TG-граф из 1 токена) снимет это давление и приблизит `ub=512` к `ub=128` по TG TPS.

### Реализация

Добавлены поля в `llama_context`:
- `sched_tg` — второй `ggml_backend_sched_ptr` с TG-буфером (~7 MiB для Qwen3.6-27B)
- `sched_is_tg` — флаг активного scheduler
- `gf_res_prev_tg` — кэш TG-графа (чтобы сохранить graph reuse в decode фазе)

Переключение происходит в `process_ubatch()`: при `ubatch.n_tokens == 1` → TG-scheduler, иначе → PP-scheduler. При смене режима оба scheduler синхронизируются.

Файлы:
- `src/llama-context.h` — объявления полей
- `src/llama-context.cpp` — `sched_reserve()`, `process_ubatch()`, `synchronize()`, деструктор

### Результаты (ctx=65536, b=4096, Qwen3.6-27B-Q3_K_S, q4_0/q4_0, ngram-mod, tasks=quick)

| Build | Dual-sched | UBatch | Wall TPS | Runs |
|---|:---:|---:|---:|---:|
| `build-rocm-wmma` | нет | 128 | 25.17 | 1 |
| `build-rocm-wmma` | нет | 512 | 19.58 | 1 |
| `build-rocm-vec` | **да** | 128 | 25.39 | 1 |
| `build-rocm-vec` | **да** | 512 | 24.53 | 1 |
| `build-rocm-wmma` | **да** | 512 | **32.16** | 3 |

Итог:
- Разрыв `ub=128 vs ub=512` на `build-rocm-vec` сократился с ~5.6 TPS → **0.86 TPS**.
- `build-rocm-wmma + dual-sched + ub=512` = **32.16 wall TPS** (+64% к baseline ub=512 того же билда).
- TG compute buffer: 6.95 MiB (TG) vs 495 MiB (PP) — подтверждён.

### Overhead

Переключение scheduler происходит один раз (PP→TG) после prefill. Остальные decode-шаги не вызывают swap. Overhead измеримо мал (< 1 мс на переключение).

## RDNA4 Graph-Opt Hang (2026-05-10)

Проблема:

- На Windows + ROCm (RX 9070 XT / `gfx1201`) запуск с `GGML_CUDA_GRAPH_OPT=1` стабильно зависал в начале первого запроса (после prefill/checkpoint, до первого ответа).
- Симптом в server log: остановка около `begin: ngram_mod occupancy ...`.

Диагностика:

- Добавлена временная instrumentation в `ggml_backend_cuda_graph_optimize()`.
- Лог показал, что зависание происходит в graph-opt path до стабильного compute/reply цикла.

Фикс:

- В `ggml/src/ggml-cuda/ggml-cuda.cu` добавлен guard:
  - для `GGML_CUDA_CC_IS_RDNA4(cc)` graph optimizer отключается по умолчанию;
  - override доступен через `GGML_CUDA_ALLOW_RDNA4_GRAPH_OPT=1` (только для ручных экспериментов).

Результат после фикса (тот же workload, `ctx=65536`, `b=4096`, `ub=512`, `q4_0/q4_0`):

| Label | Env/Spec | Status | Wall TPS |
|---|---|---|---:|
| `graphopt-on-smoke` | `GGML_CUDA_GRAPH_OPT=1`, `spec=ngram-mod` | hang | — |
| `graphsafe-off-specnone-r1` | `GGML_CUDA_DISABLE_GRAPHS=1`, `spec=none` | stable | `24.61` |
| `graphopt-rdna4-guard-r1` | `GGML_CUDA_GRAPH_OPT=1` + RDNA4 guard, `spec=none` | stable | `24.59` |
| `graphopt-rdna4-guard-ngram-r1` | `GGML_CUDA_GRAPH_OPT=1` + RDNA4 guard, `spec=ngram-mod` | stable | `24.64` |

Вывод:

- На текущем RDNA4/ROCm пути использовать graph-opt без guard нельзя (deadlock-risk).
- Безопасный baseline: оставить guard включённым, или задавать `GGML_CUDA_DISABLE_GRAPHS=1` для диагностических прогонов.

## RDNA4 ROCm Native UBatch Cliff Fix (2026-05-12)

Проблема:

- На RX 9070 XT / ROCm `Qwen3.6-27B-Q3_K_S` уходил в slow pocket при полном native PP reserve: `ctx=32768, ub=904/1024` и также `ctx=16384, ub=900`.
- Full trace показал одинаковые graph/node counts и те же FATTN/GDN/MMQ route classes, но широкое замедление memory-heavy ops: GLU/RMS_NORM/ADD/SSM_CONV и часть MUL_MAT/FATTN. Это не был single-kernel selector bug.
- A/B подтвердил причину: один крупный ROCm compute vbuffer allocation попадает в плохой residency/placement pocket. Простое смещение base offset не помогало; разбиение compute vbuffer на backend chunks помогало при сохранении полного `PP reserve`.

Фикс:

- В `ggml/src/ggml-alloc.c` для ROCm graph allocator добавлен default max compute vbuffer chunk size `256 MiB`.
- `ggml_dyn_tallocr` теперь может создавать несколько backend buffer chunks для одного virtual compute buffer; model/KV offload и requested ubatch не уменьшаются.
- Override для экспериментов: `GGML_COMPUTE_VBUFFER_MAX_CHUNK_SIZE=<bytes>`.
- Контрольное отключение default ROCm chunking: `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1`.

Результаты (`build-rocm-vec`, `ctx=32768`, `b=5120`, `q4_0/q4_0`, `ngram-mod`, no-reuse, repo-snapshot context):

| Label | PP reserve | ROCm0 compute | Prompt eval |
| --- | ---: | ---: | ---: |
| `native-singlechunk-ctx32768-ub904-mt1-r1` | `904 -> 1` | single chunk | `23524.85 ms / 302.87 tok/s` |
| `native-defaultchunk-ctx32768-ub904-mt1-r1` | `904 -> 1` | `374.84 MiB` | `6862.92 ms / 1038.19 tok/s` |
| `native-final-ctx32768-ub1024-mt1-r1` | `1024 -> 1` | `424.53 MiB` | `6392.54 ms / 1114.58 tok/s` |
| `native-defaultchunk-ctx16384-ub900-mt1-r1` | `900 -> 1` | `281.54 MiB` | `6798.72 ms / 1047.99 tok/s` |

Practical run:

| Label | PP reserve | Prompt eval | Decode | Total |
| --- | ---: | ---: | ---: | ---: |
| `native-defaultchunk-ctx32768-ub1024-mt120-r1` | `1024 -> 1` | `6394.28 ms / 1114.28 tok/s` | `120 tok / 25.03 tok/s` | `11188.80 ms` |

Вывод:

- Старый guard/cap до `ub=900` больше не нужен для native `ub1024` path.
- Контроль `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1` возвращает slow result, что подтверждает allocator/residency root cause.

## RDNA4 ROCm Q4_K_S Full-Offload + MMQ Selector (2026-05-19)

Контекст: пользовательская модель `models/Qwen3.6-27B-Q4_K_S.gguf` оказалась значительно медленнее текущего Q3 профиля. Это не просто Q3 с другим bpw: Q4 файл имеет 65 блоков, NextN/MTP слой и Q4_K/Q5_K-heavy tensor mix.

### Диагностика

На активной prompt-heavy lane (`ctx=12288`, `b=4096`, `ub=1024`, `q4_0/q4_0`, FlashAttention, thinking ON, no-reuse) default auto-fit сначала уводил слои на CPU:

| Label | Extra args | Offload | Wall | Prompt eval | Decode eval |
| --- | --- | ---: | ---: | ---: | ---: |
| `e070-rocm-q4ks-baseline-ctx12288-r1` | default fit | 60/66 | `122.23s` | `64.39 tok/s` | `10.89 tok/s` |
| `e070-rocm-q4ks-fitt0-ctx12288-r1` | `-fitt 0` | 64/66 | `111.43s` | `70.59 tok/s` | `12.07 tok/s` |
| `e070-rocm-q4ks-fitoff-smoke-ctx12288-r1` | `-fit off` | 66/66 | smoke | `69.72 tok/s` | n/a |

`-fit off` доказал, что CPU offload был первой проблемой, но не всей причиной: даже полный offload оставался медленным. `llama-bench` показал route issue в Q4_K/Q5_K: forced MMQ был намного быстрее старого default dequant+hipBLAS path.

| Model | Route | pp512 | pp1024 |
| --- | --- | ---: | ---: |
| Q4_K_S old default | old RDNA4 gate (`ne11<=192`) | `57.30 tok/s` | `72.17 tok/s` |
| Q4_K_S forced MMQ | `GGML_CUDA_FORCE_MMQ_RUNTIME=1` | `250.22 tok/s` | `328.25 tok/s` |

### Изменение

В `ggml/src/ggml-cuda/mmq.cu` RDNA4 selector теперь расширяет MMQ gate только для `Q4_K`/`Q5_K` до `ne11<=1024`. Остальные K-типы (`Q2_K`, `Q3_K`, `Q6_K`) остаются на старом `ne11<=192`. Для A/B и rollback добавлен override:

```text
GGML_MMQ_RDNA4_Q4K_MAX_NE11=<int>
```

Отрицательный контроль `GGML_MMQ_RDNA4_Q4K_MAX_NE11=192` возвращает старую скорость (`58.12 tok/s` pp512), что подтверждает причинность.

### Результат

| Label | Mode | Wall TPS | Wall | Prompt eval | Decode eval | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `e070-rocm-q4ks-defaultmmq-fitoff-clean-ctx12288-r1` | `spec=none`, `-fit off` | `2.25` | `28.44s` | `330.42 tok/s` | `11.15 tok/s` | current Q4 default recommendation |
| `e070-rocm-q4ks-defaultmmq-mtp-fitoff-clean-ctx12288-r1` | MTP, `-fit off` | `1.63` | `39.17s` | `219.37 tok/s` | `12.80 tok/s` | acceptance `41/63`, slower overall |
| Vulkan E068 Q4 check | `wm32-wn32` opt-in | `1.66` | `38.60s` | `262.35 tok/s` | `6.39 tok/s` | slower than ROCm Q4 after fix |

Q3 negative control after the patch stayed healthy: `Qwen3.6-27B-Q3_K_S` pp512 measured `502.03 tok/s`; Q3_K route is intentionally unchanged.

### GUI recommendation

`gui/model_presets.json` now sets the Q4 preset to the practical RX 9070 XT path:

- `ctx=12288`
- `batch=4096`, `ubatch=1024`
- `gpu_layers=-1`, FlashAttention ON
- `q4_0/q4_0` KV
- thinking ON
- `extra_args`: `--spec-type none` + `-fit off`

MTP remains opt-in for Q4. It improves accepted decode tokens in this test but adds enough prompt/MTP overhead that prompt-heavy wall time regresses.

## ROCm Q3_K 12k Route Refresh and Repeated-Session Gain (2026-05-20)

Контекст: после полной карты маршрутов был обновлён активный `Qwen3.6-27B-Q3_K_S` профиль на RX 9070 XT / ROCm:

- `ctx=12288`
- `batch=6144`, `ubatch=2048`
- KV `q4_0/q4_0`
- `spec=none`
- tasks `triage_diff,review_bug`
- thinking ON

Cold-first baseline остаётся отдельной метрикой: `--no-reuse --cache-ram 0 --ctx-checkpoints 0`.

| Label | Mode | Aggregate TPS | Notes |
| --- | --- | ---: | --- |
| `e106-rocm-q3k-control-r1` | cold-first, no reuse | `11.8464` | fresh same-lane control |
| `e107-rocm-q3k-ngrammod-r1` | cold-first, ngram-mod 24/48/64 | `11.7838` | generated zero drafts; reject |
| `e107-rocm-q3k-ngrammod-m12-r1` | cold-first, ngram-mod 12/16/32 | `11.3471` | effective acceptance `0.001428`; reject |
| `e107-rocm-q3k-ngramsimple-n8m16-r1` | cold-first, ngram-simple | `11.2810` | effective acceptance `0.004908`; reject |
| `e108-rocm-gdn-control-r1` | cold-first post-build control | `11.7604` | GDN probe baseline |
| `e108-rocm-gdn-warps2-r1` | cold-first, temporary `num_warps=2` | `11.7408` | reject/reverted |
| `e108-rocm-gdn-warps1-r1` | cold-first, temporary `num_warps=1` | `11.7258` | reject/reverted |
| `e109-vulkan12k-q3k-q4kv-r1` | Vulkan same-lane fallback | `0.0000` | full offload loaded, first task timed out |
| `e110-rocm-q3k-fitoff-r1` | cold-first, `-fit off` | `11.7557` | tie; do not transfer Q4 fit-off rule to Q3 |
| `e111-rocm-q3k-reuse-steady-r1` | repeated/session, reuse enabled | `14.6132` | prompt cache/checkpoints enabled |
| `e111-rocm-q3k-reuse-steady-r3` | repeated/session, reuse enabled | `17.7984` | confirmed route; after-first tasks about `20.00 TPS` |
| `e112-rocm-q3k-reuse-ngram244864-r3` | repeated/session, reuse + ngram-mod | `18.7194` | stacked opt-in route; after-first tasks about `21.40 TPS` |

E111 is the useful practical gain from this cycle, but it is **not** a cold-first kernel/default speedup. Server logs show the mechanism:

- prompt cache enabled with `8192 MiB` limit;
- first task creates checkpoints around `5370` and `7418` tokens;
- later tasks select the slot by LCP similarity (`sim_best=0.982-0.984`);
- later tasks restore the `5370`-token checkpoint and reprocess only about `2033-2052` prompt tokens instead of the full `7403-7422`.

Вывод:

- Для GUI/agent sessions keep prompt cache/checkpoints enabled: practical repeated throughput is now around `17.8 TPS` aggregate and `~20 TPS` after the first shared-prefix task.
- Optional stacked session route: add `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64` on top of cache/checkpoints. In E112 this reached `18.72 TPS` aggregate and about `21.40 TPS` after the first task, with `102/126` accepted draft tokens in two bursts.
- For cold-first kernel work keep using no-reuse controls; the current cold-first ceiling remains around `11.75-11.85 TPS` on this 12k q4-KV lane.
- Speculative cold-first projections must include coverage/effective acceptance. In E107 local acceptance alone was misleading because coverage was almost zero.
- GDN block-geometry changes now require resource/occupancy proof before coding; `num_warps=1/2` and chunk-size style probes are closed.

## Driver 32.0.31007.5012 Rebaseline (2026-05-21)

Контекст: после обновления AMD video driver все baseline были обновлены. Новый driver fingerprint:

- GPU: AMD Radeon RX 9070 XT
- Driver: `32.0.31007.5012`
- Driver date: `2026-05-12`

Same active lane:

- `Qwen3.6-27B-Q3_K_S`
- ROCm `build-rocm-vec`
- `ctx=12288`, `batch=6144`, `ubatch=2048`
- KV `q4_0/q4_0`
- tasks `triage_diff,review_bug`
- thinking ON

| Label | Mode | Aggregate TPS | Notes |
| --- | --- | ---: | --- |
| `e113-driver5012-rocm-cold-specnone-r3` | cold-first, no reuse, `spec=none` | `11.9858` | new cold-first baseline |
| `e113-driver5012-rocm-reuse-specnone-r3` | repeated/session, reuse, `spec=none` | `17.8934` | after-first mean `20.2012 TPS` |
| `e113-driver5012-rocm-reuse-ngram244864-r3` | reuse + ngram-mod 24/48/64 | `17.7270` | noisy negative r3 |
| `e113-driver5012-rocm-reuse-ngram244864-r3b` | reuse + ngram-mod 24/48/64 | `18.4637` | noisy positive r3b |
| `e113-driver5012-rocm-reuse-ngram121632-r3` | reuse + ngram-mod 12/16/32 | `19.0148` | after-first mean `23.1681 TPS` |
| `e113-driver5012-rocm-reuse-ngram121632-r3b` | reuse + ngram-mod 12/16/32 | `19.5051` | after-first mean `23.9038 TPS` |
| `e113-driver5012-rocm-reuse-ngramsimple-n8m16-r3` | reuse + ngram-simple n8/m16 | `15.3491` | reject |
| `e114-driver5012-rocm-reuse-ngram081632-r3` | reuse + ngram-mod 8/16/32 | `14.2479` | reject; local acceptance collapsed |

Prompt-only pp7488 sanity:

| Label | Backend | pp7488 tok/s | Notes |
| --- | --- | ---: | --- |
| `e113-driver5012-vulkan-pp7488-r3` | Vulkan | `900.22 +/- 151.13` | first post-driver run, likely shader/pipeline cache noise |
| `e113-driver5012-vulkan-pp7488-r3b` | Vulkan | `962.41 +/- 33.93` | warmed repeat |
| `e113-driver5012-rocm-pp7488-r3` | ROCm | `1159.49 +/- 73.80` | ROCm still ahead on prompt |

Вывод:

- Cold-first baseline changed modestly: use `11.9858 TPS` for new no-reuse kernel/default comparisons.
- Practical repeated/session best is now prompt cache/checkpoints + `ngram-mod 12/16/32`: best r3 `19.5051 TPS`, after-first mean `23.9038 TPS`.
- Why shorter ngram won: it increased effective acceptance to `0.035028` (`320/484` accepted draft tokens), enough to speed repeated decode. The first cold task is slower, so this remains session-only.
- Why even shorter match failed: `ngram-mod 8/16/32` generated more bad drafts (`74/544` accepted), effective acceptance fell to `0.004251`, and decode collapsed to `20.33 tok/s`.
- `ngram-simple` is rejected: despite drafts, decode fell to `22.70 tok/s`, so its speculative overhead/verify pattern is worse than reuse-only.
- Vulkan did not become a replacement for ROCm after the driver update; warmed Vulkan pp7488 is still about `17%` behind ROCm.

## Driver 5012 Decode Route and Live-Server Sanity (2026-05-21)

Context: after the driver update, decode-heavy routing was tested separately from prompt-heavy cold-first routing. This is not a replacement for the ROCm prompt-heavy default; it is a long-generation/decode profile.

| Label | Backend / KV | Mode | Aggregate TPS | Decode eval | Notes |
| --- | --- | --- | ---: | ---: | --- |
| `e116-driver5012-decode-rocm-q4-specnone-r1` | ROCm q4 | short-prompt decode gate | `29.1685` | `29.625 tok/s` | decode control |
| `e116-driver5012-decode-vulkan-q4-specnone-r3` | Vulkan q4 | short-prompt decode gate | `39.8801` | `40.8683 tok/s` | confirmed r3 |
| `e116-driver5012-decode-vulkan-f16-specnone-r3` | Vulkan f16 | short-prompt decode gate | `40.2753` | `41.2283 tok/s` | current decode-heavy route |
| `e118-quality-vulkan-f16-specnone-mt512-r1` | Vulkan f16 | real-context server run, 512 output tokens | `28.7575` | `39.855 tok/s` | prompt + decode mixed, real server logs |
| `e119-realctx512-rocm-q4-specnone-r3` | ROCm q4 | real-context r3, 512 output tokens | `24.9524` | `28.4483 tok/s` | warm-only `25.82 TPS` |
| `e119-realctx512-vulkan-f16-specnone-r3` | Vulkan f16 | real-context r3, 512 output tokens | `32.0298` | `39.4483 tok/s` | warm-only `33.89 TPS` |
| `e120-realctx512-vulkan-q4-specnone-r3` | Vulkan q4 | real-context r3, 512 output tokens | `32.1668` | `40.1350 tok/s` | warm-only `33.96 TPS`, KV `216 MiB` |
| `e121-realctx256-rocm-q4-specnone-r3` | ROCm q4 | real-context r3, 256 output tokens | `22.3563` | `28.7067 tok/s` | warm-only `23.88 TPS` |
| `e121-realctx256-vulkan-q4-specnone-r3` | Vulkan q4 | real-context r3, 256 output tokens | `26.6050` | `40.0600 tok/s` | warm-only `29.43 TPS` |
| `e122-realctx128-rocm-q4-specnone-r3` | ROCm q4 | real-context r3, 128 output tokens | `18.3480` | `28.8350 tok/s` | warm-only `20.55 TPS`, cold-only `15.11 TPS` |
| `e122-realctx128-vulkan-q4-specnone-r3` | Vulkan q4 | real-context r3, 128 output tokens | `19.6365` | `40.2400 tok/s` | warm-only `23.15 TPS`, cold-only `15.06 TPS` |

Live-server correctness:

- User manually verified the Vulkan route in a real server/client flow: normal thinking/answers, no corrupted symbol output, no slash spam.
- Direct ROCm server sanity with unrestricted reasoning produced coherent `reasoning_content` but no final `content` before `max_tokens=1024`; this is a reasoning-budget/API extraction behavior, not a backend corruption signal.
- Direct ROCm server sanity with `--reasoning-budget 256` produced a normal final answer (`finish_reason=stop`, `436` completion tokens, decode `29.52 tok/s`).

Workflow update: every future large speedup needs a lightweight live-server smoke before promotion. The check is simple: run the actual target backend, ask a normal prompt, and reject the route if it produces repeated punctuation/symbols or broken reasoning like the old `wm32-wn32` Vulkan bug.

E119/E120 result: Vulkan is now the confirmed long-answer/repeated-session backend for this model on driver `32.0.31007.5012`. E120 makes the practical route **Vulkan q4 KV**, not f16: it matched/slightly beat f16 (`32.1668` vs `32.0298 TPS`) while reducing KV from `768 MiB` to `216 MiB`. Against ROCm q4, the long-answer gain is `+28.91%` aggregate and `+31.53%` warm-only. This does not replace the ROCm q4 cold prompt-heavy default, because ROCm still wins prompt eval (`1152.56` vs Vulkan q4 `894.05 tok/s` in E119/E120).

E121 boundary result: Vulkan q4 already wins at 256 generated tokens (`26.6050` vs ROCm `22.3563`, `+19.00%`; warm-only `29.43` vs `23.88`, `+23.24%`). The route split is now: ROCm q4 for cold/prompt-heavy and short generation; Vulkan q4 for medium/long repeated-session answers.

E122 boundary result: at 128 generated tokens, repeated/session Vulkan q4 still wins (`19.6365` vs `18.3480`, `+7.02%`; warm-only `23.15` vs `20.55`, `+12.65%`), but cold-only is a tie/slightly ROCm (`15.06` Vulkan vs `15.11` ROCm). This locks the policy: choose by scenario, not by a single global backend.

ROCm prefill allocator scout (E123): changing compute vbuffer chunking did not improve the cold prompt-heavy lane after driver `32.0.31007.5012`. Against E113 cold `11.9858 TPS`, `128 MiB` measured `11.8684`, `64 MiB` measured `11.8013`, and `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1` measured `11.7784`. Keep default chunking; continue prefill work in H35 route/kernel space.

Vulkan speculative stack scout (E124): do not add `ngram-mod 12/16/32` to the Vulkan q4 session route. On the E122 128-token lane it regressed from `19.6365` to `14.3229 TPS` (`0.7294x`), with effective acceptance only `0.004844`. Vulkan q4 session route stays `spec=none`; ROCm keeps the separate E113 ngram session opt-in.

## Vulkan CPU 0-Offload Route (E125, 2026-05-21)

Context: CPU fallback inside the Vulkan build, launched with `--gpu-layers 0`.
This is not pure CPU unless `--no-op-offload` is also used. The benchmark was
intentionally small because Qwen3.6-27B-Q3_K_S CPU fallback is slow:
`ctx=4096`, `batch=512`, `ubatch=128`, q4 KV, FlashAttention on, `spec=none`,
no reuse, thinking on, `max_tokens=32`, task `review_bug`.

| Label | Route | Aggregate TPS | Prompt eval | Decode eval | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `e125-cpu0offload-default32-r3` | `-ngl 0`, mmap, op-offload on | `1.7703` | `32.5033 tok/s` | `2.3267 tok/s` | baseline |
| `e125-cpu0offload-t6-32-r3` | `--threads 6 --threads-batch 6` | `1.7995` | `30.6367 tok/s` | `2.4267 tok/s` | small/tie |
| `e125-cpu0offload-noopoff32-r3` | `--no-op-offload` | `0.8900` | about `6.18 tok/s` | about `2.47 tok/s` | reject |
| `e125-cpu0offload-f16kv32-r1` | f16/f16 KV | `1.7617` | `27.69 tok/s` | `2.45 tok/s` | reject |
| `e125-cpu0offload-mlock32-r1` | `--mlock` | `1.7196` | `27.81 tok/s` | `2.36 tok/s` | reject |
| `e125-cpu0offload-nommap32-r3` | `--no-mmap` | `1.8815` | `33.9133 tok/s` | `2.4900 tok/s` | keep |
| `e125-cpu0offload-nommap-t6-32-r3` | `--no-mmap --threads 6 --threads-batch 6` | `1.8931` | `31.4133 tok/s` | `2.5767 tok/s` | optional decode-skew route |

Partial-offload scout with `--no-mmap`:

| Label | GPU layers | Aggregate TPS |
| --- | ---: | ---: |
| `e125-vulkan-hybrid-ngl8-nommap32-r1` | 8 | `2.11` |
| `e125-vulkan-hybrid-ngl16-nommap32-r1` | 16 | `2.32` |
| `e125-vulkan-hybrid-ngl32-nommap32-r1` | 32 | `3.46` |
| `e125-vulkan-hybrid-ngl48-nommap32-r1` | 48 | `6.03` |
| `e125-vulkan-full-ngl65-nommap32-r1` | 65 | `28.93` |

Route findings:

- Keep `--no-mmap` for practical Vulkan `-ngl 0` CPU fallback. It improved r3
  wall TPS from `1.7703` to `1.8815` (`+6.28%`) while preserving the same
  Vulkan op-offload scheduler route.
- Do not use `--no-op-offload` for speed. It collapses graph splits but drops
  prompt eval by about 4-5x.
- q4 V cache requires FlashAttention; `--no-flash-attn` failed at init with
  `V cache quantization requires flash_attn`.
- The real code bottleneck is CPU Q3_K matvec: `GGML_TYPE_Q3_K` routes to
  `ggml_vec_dot_q3_K_q8_K`, `.nrows = 1`, and Q3_K has no current x86 repack
  route in `ggml/src/ggml-cpu/repack.cpp`.
- Next CPU-code work should isolate Q3_K x86 vec-dot changes and/or add a
  Q3_K repack/interleaved route. The current local `quants.c` micro-change was
  present in the measured binary and is not yet a clean-vs-candidate claim.

Recommended practical CPU fallback command additions:

```powershell
--gpu-layers 0 --no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on --spec-type none
```

CPU Q3_K follow-up probes:

- E126 isolated the current local `quants.c` Q3_K mask/shuffle preload change
  against a clean worktree. Aggregate moved `1.8067 -> 1.8611 TPS`, but decode
  did not improve (`2.4833 -> 2.4800 tok/s`), so this is not a promoted CPU
  decode speedup.
- E127 tested a next-block Q3/Q8 prefetch inside `ggml_vec_dot_q3_K_q8_K` on a
  64-token gate. It measured only `2.0716 -> 2.0950 TPS` r1 and decode
  `2.42 -> 2.44 tok/s`, so the patch was reverted. This points away from
  simple mask/prefetch tweaks and toward Q3_K repack/interleaved matvec work.

## Vulkan 64k Q3_K Route Gate (E137, 2026-05-22)

Short pp gate for the complex Q3_K route idea "reuse one A/dequant tile across
two adjacent N-blocks":

| Build state | Variant | pp7488 | Pipeline resources | Decision |
| --- | --- | ---: | --- | --- |
| temporary probe source | default with generic B guard | `858.83` | `95 VGPR / 45 SGPR / 20480 B LDS / 0 scratch` | reject source shape |
| temporary probe source | `GGML_VK_AMD_LARGE_MATMUL_VARIANT=niter2` | `855.29` | `120 VGPR / 45 SGPR / 20480 B LDS / 0 scratch` | reject |
| clean restored source | accepted default | `974.92` | `113 VGPR / 45 SGPR / 20480 B LDS / 0 scratch` | restored |

The temporary patch was reverted and no full 64k server A/B was run. The result
is useful because it separates the route class from this concrete implementation:
reducing repeated A-side Q3_K work remains attractive, but doing it by doubling
accumulator live state inside current `mul_mm.comp` loses to VGPR/occupancy and
shader fingerprint risk. Next Q3_K work should prefer backend-private Q3_K
repack/layout or a separate shape-specific shader that does not perturb the
accepted default route.

## Vulkan 64k FlashAttention Split-K Gate (E138, 2026-05-22)

Real-server full prompt screen with `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`,
`--no-mmap`, `b8192/ub1024`, q4/q4 KV, FlashAttention on:

| Route | Prompt tokens | Elapsed | Prompt eval | Decision |
| --- | ---: | ---: | ---: | --- |
| default FA main chunks | `57518` | `86.3639 s` | `666.87 tok/s` | baseline |
| forced existing FA split-k2 from `KV>=8192` | `57518` | `597.4568 s` | `96.29 tok/s` | reject/revert |

Route trace confirmed the candidate used `split_k=2` and `split_kv=KV/2` for
main long-KV chunks. The failure is structural: the existing split-k route adds
temporary output/L/M writes, `ggml_vk_sync_buffers`, and a split-k reduce
dispatch for each FA node. Future FA long-KV work must stay in a single dispatch
or first redesign that reduce topology.

## Vulkan 64k Q3_K Predequant Route Gate (E139, 2026-05-22)

Short pp gate for the complex Q3_K route idea "use existing backend
predequant, then f16 matmul" on the current Vulkan `b8192/ub1024`, q4/q4,
FlashAttention-on lane:

| Route | pp7488 | Route activation | Decision |
| --- | ---: | --- | --- |
| direct Q3_K baseline | `969.61` | `matmul_q3_k_f32_f16acc_aligned_l`, `qx_dequant=0` | baseline |
| force all large Q3_K predequant | `743.65` | `matmul_f16_f32_f16acc_aligned_l`, `qx_dequant=1` | reject |
| force only `m>=17000` | `832.27` | predequant only for `m=17408,k=5120` | reject |
| force only `k>=17000` | `929.40` | predequant only for `m=5120,k=17408` | reject |

Pipeline stats for the f16 fallback route were not the problem by themselves:
`77 VGPR / 44 SGPR / 22528 B LDS / 0 scratch`. The regression points at the
route topology: each top hot shape writes a roughly `170 MiB` fp16 temp,
synchronizes, then rereads it for f16 matmul. The temporary env-gated code was
reverted and `llama-bench`/`llama-server` were rebuilt clean.

## Vulkan 64k Q3_K Matmul Split-K Gate (E140, 2026-05-22)

Short pp gate for forcing existing Vulkan matmul split-K on the hot reverse
Q3_K shape `m=5120,n=1024,k=17408`:

| Route | pp7488 | Decision |
| --- | ---: | --- |
| direct Q3_K baseline | `968.74` | baseline |
| forced split-K2 for `k>=17000` | `966.21` | reject |
| forced split-K4 for `k>=17000` | `964.46` | reject |

Route trace confirmed the forced split on the intended hot shape. This is not a
catastrophic branch, but it is not positive: the shape already exposes about
`320` large-tile workgroups before split-K, so adding K partitions mostly adds
partial-output traffic, sync, and reduce overhead. The temporary code was
reverted and the Vulkan bench/server binaries were rebuilt clean.

## Vulkan 64k KV Dtype Route Gate (E141, 2026-05-22)

Short pp gate for the complex FA-route question "is q4 KV dequant the missing
large lever?":

| KV route | pp7488 | Fit / decision |
| --- | ---: | --- |
| q4_0/q4_0 | `970.03 tok/s` | baseline |
| f16/f16 | `996.00 tok/s` | only `+2.68%` pp upper bound; 64k server fit failed |
| q8_0/q8_0 | `940.03 tok/s` | reject |

The f16 64k server never became ready because the memory fitter projected
`16183 MiB` Vulkan device use against `15221 MiB` free and needed to reduce by
`1986 MiB`. Using the E134 FA share as a rough proxy, the f16 pp result implies
only about `1.067x` local FA speedup, far below the `1.494x` local speedup FA
would need to close the 64k gap alone. Keep q4/q4 KV for H38; future FA work
should optimize the single-dispatch q4 coopmat1 route directly rather than
building an f16 KV cache/dequant route that does not fit.

## Vulkan 64k FA Br32/Bc32 Route Gate (E142, 2026-05-22)

Short pp/resource gate for a structural FA route that doubles query rows per
workgroup while cutting `Bc` to stay within the 32 KiB LDS budget:

| FA route | pp7488 | Pipeline resources | Decision |
| --- | ---: | --- | --- |
| default `Br16/Bc64`, f32acc | `971.09 tok/s` | `98 VGPR / 76 SGPR / 26112 B LDS / 0 scratch` | baseline |
| `Br32/Bc32`, f32acc | `896.97 tok/s` | `133 VGPR / 83 SGPR / 27136 B LDS / 0 scratch` | reject |
| `Br32/Bc32`, f16acc | `922.22 tok/s` | `134 VGPR / 83 SGPR / 25088 B LDS / 0 scratch` | reject |

Route trace confirmed the candidate stayed on coopmat1 q4/q4 with
`Br=32,Bc=32,row_split=2,workgroup_size=128`. The failure is not split/reduce or
fallback; it is live-state pressure inside the single-dispatch shader. Doubling
rows increases `Of/Lf/Mf/mask` state enough to raise VGPR sharply, and f16acc
does not fix it. The temporary env-gated code was reverted and Vulkan
`llama-bench`/`llama-server` were rebuilt clean.

## Vulkan 64k Q3_K Large-N Warptile Gate (E143, 2026-05-22)

Complex route-family gate for the idea "reduce repeated A-side Q3_K work by
making each large matmul cover more N columns per workgroup". Static scout
first rejected plain `BN192` as unsafe for the current A-load map, then tested
valid `BN192/WN96` and `BN256` variants under `GGML_VK_AMD_LARGE_MATMUL_VARIANT`.

| Q3_K route | pp7488 | Pipeline resources | Decision |
| --- | ---: | --- | --- |
| default `BN128/WN64` | `974.19` | `113 VGPR / 45 SGPR / 20480 B LDS / 0 scratch` | baseline |
| `bn192-wn96` | `760.78` | `139 VGPR / 48 SGPR / 25088 B LDS / 0 scratch` | reject |
| `bn192-wm128-wn96` | `137.71` | `171 VGPR / 54 SGPR / 24064 B LDS / 784 B scratch` | reject |
| `bn256-wn128` | `659.02` | `165 VGPR / 58 SGPR / 29696 B LDS / 0 scratch` | reject |
| `bn256-wm128` | `660.97` | `165 VGPR / 43 SGPR / 29696 B LDS / 0 scratch` | reject |

The static model was directionally useful but incomplete: `BN192/BN256` really
reduce N-tile count and the A-dequant proxy, yet the current `mul_mm.comp`
topology turns that into much higher accumulator/live-state pressure. `WN96`,
`WN128`, and `WM128` variants all lose to VGPR/LDS/scratch and lower occupancy.
The temporary env-gated branches were reverted and Vulkan `llama-bench` /
`llama-server` were rebuilt clean. Next Q3_K work should not be another larger
N-tile retune; it needs a backend-private Q3_K layout or separate
shape-specific shader that reduces repeated A-side work without growing live
fragments this way.

## Vulkan 64k Q3_K BK16 Route Gate (E144, 2026-05-22)

Resource-direction gate for the opposite route idea: shrink the Q3_K K tile
from `BK=32` to `BK=16`, accepting twice as many K-loop/barrier rounds in
exchange for much lower LDS and register pressure. `BK64` was rejected at the
static gate because Q3_K shader LDS would be `36864 B`, above the 32 KiB device
budget.

| Q3_K route | pp7488 | Pipeline resources | Decision |
| --- | ---: | --- | --- |
| default `BK32` | `972.77` | `113 VGPR / 45 SGPR / 20480 B LDS / 0 scratch` | baseline |
| `bk16` | `587.52` | `70 VGPR / 46 SGPR / 12288 B LDS / 0 scratch` | reject |

This is a useful negative control: the `BK16` pipeline is genuinely lighter,
but prompt throughput drops `-39.60%`. The current Q3_K route is not primarily
saved by reducing LDS/VGPR if that doubles K-loop and barrier cadence. The
temporary env-gated branch was reverted and clean Vulkan `llama-bench` /
`llama-server` were rebuilt.

## Vulkan 64k FA D-Split Route Gate (E145, 2026-05-22)

FlashAttention resource-distribution gate for the active q4/q4 coopmat1 route.
The default `Br16/Bc64,row_split=4,D_split=8` was compared against
`D_split=4` and `D_split=16` with the same `pp7488`, `b8192/ub1024`,
q4/q4 KV lane. Route trace confirmed all candidates stayed on
`flash_attn_f32_f16_aligned_f32accq4_0`.

| FA route | pp7488 | Pipeline resources | Decision |
| --- | ---: | --- | --- |
| default `D_split=8` | `978.88` | `98 VGPR / 76 SGPR / 26112 B LDS / 0 scratch` | baseline |
| `D_split=4` | `953.24` | `98 VGPR / 76 SGPR / 26112 B LDS / 0 scratch` | reject |
| `D_split=16` | `951.54` | `98 VGPR / 76 SGPR / 26112 B LDS / 0 scratch` | reject |

This closes the simple `D_split` branch. The model was plausible because
`D_split=16` halves the live output vector state while doubling score-column
state, and `D_split=4` does the inverse. In practice the driver reports the
same resource fingerprint and both directions regress `-2.6%` to `-2.8%`.
Future FA work needs a shader-body change or per-KV-tail timing proof; merely
redistributing the head dimension inside the current cm1 shader is not enough.

## Vulkan 64k Q3_K BM256 Route Gate (E146, 2026-05-22)

Route-level Q3_K test in the opposite direction from E143. Instead of growing
`BN` to reduce A-side repetition, `BM256` keeps `BN=128,BK=32` and halves the
M-block count. Static scout predicted about `2x` lower B reload/workgroup
proxy on hot 64k shapes while leaving A-pair dequant work roughly unchanged.

| Q3_K route | pp7488 | Pipeline resources | Decision |
| --- | ---: | --- | --- |
| default `BM128/BN128/BK32` | `972.84` | `113 VGPR / 45 SGPR / 20480 B LDS / 0 scratch` | baseline |
| `BM256/BN128/BK32` | `916.62` | `94 VGPR / 45 SGPR / 31744 B LDS / 0 scratch` | reject |

The route did reduce VGPR and kept scratch at zero, but LDS rose to `31744 B`,
almost the whole 32 KiB device budget, and prompt throughput fell `-5.78%`.
That corrects the workflow: B/workgroup proxy reduction is not enough unless
the candidate also preserves occupancy/residency. Future Q3_K work should not
repeat larger M/N tiles in current `mul_mm.comp`; it needs a different layout
or shape-specific shader that reduces repeated A work without near-limit LDS.

## Vulkan 64k Q3_K Layout Route Gate (E147, 2026-05-22)

Design gate for the larger Q3_K layout/repack branch. The real GGUF tensor map
shows broad persistent fp16/int8 layouts are not viable for the 16 GiB 64k lane:
FFN `gate/up/down` already occupy `6.85 GiB` as Q3_K, while alternate fp16
would add `25.03 GiB` and int8 would add `9.09 GiB`. A signed-nibble layout is
the only broad memory-plausible option (`+1.12 GiB` for FFN, `+1.58 GiB` for
all Q3_K), but it only attacks bit unpack, not scale math or coopmat work.

Static SPIR-V counts also lowered confidence in a signed-nibble implementation:
current Q3_K aligned cm1 has `247 OpLoad`, `84 OpStore`, `81 OpIMul`,
`28 OpUDiv`, `13 OpUMod`, and a small number of bit ops; f16 is lighter but not
by a target-closing margin, and Q4_K is not obviously lighter than Q3_K. E147
therefore rejects persistent fp16/int8 and single-accumulator sequential-N
analytically, and defers signed-nibble until a stronger instruction/resource
proof exists. Next complex Vulkan work should move to FA long-KV shader-body
work or a new Q3_K topology that removes A-pair count without fp16 temp,
accumulator blowup, near-limit LDS, or extra reduce.

## Vulkan 64k FA Analytic Causal Mask Gate (E148, 2026-05-22)

Route-body probe for the active q4/q4 FlashAttention path. The idea was to keep
mask-opt semantics but remove the separate `fa_mask_opt` dispatch/sync for full
1024-token causal text chunks by deriving all-zero/all--inf/mixed mask tiles in
the cm1 shader.

Analytic gate from the E128 perf trace:

| Metric | Value |
| --- | ---: |
| Parsed FA rows | `58` |
| FA total | `33965.16 ms` |
| Eligible full-chunk FA time | `33865.25 ms` (`99.71%` of FA) |
| Eligible FA tiles | `26148864` |
| All-zero tiles | `98.14%` |
| All--inf skipped tiles | `1.64%` |
| Mixed boundary tiles | `0.22%` |
| Mask-opt prepass read proxy | `51072.00 MiB` fp16 mask cells |

Measured pp7488 gate after a temporary env-gated prototype:

| Route | pp7488 | Pipeline resources | Decision |
| --- | ---: | --- | --- |
| same-build baseline | `971.41` | normal full/tail route | baseline |
| analytic causal full chunks | `972.21` | full chunks `84 VGPR / 65 SGPR / 26112 B LDS / 0 scratch`; tail normal `98/76` | reject/tie |

The mechanism was plausible, but the measured delta was only `+0.08%`, inside
noise and far below the threshold for a 64k server run. The runtime prototype
was reverted. The useful conclusion is causal: mask-opt prepass/sync is not the
main FA limiter; future FA work should target the main q4 K/V dequant,
softmax/PV loop, or a more structural long-KV traversal that changes the shader
body cost rather than only removing the prepass.
