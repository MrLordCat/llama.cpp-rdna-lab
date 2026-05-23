# Route-Chain Follow-up Plan

Дата: 2026-05-23
Основание: [docs/research/NEUTRAL_SMALL_PLUS_AUDIT.md](docs/research/NEUTRAL_SMALL_PLUS_AUDIT.md), особенно Phase 3 route-chain pass.

## Цель

Проверять не одиночный «микро-плюс», а цепочку узких мест по одному lane contract:

1. фиксируем текущий bottleneck #1,
2. ускоряем его,
3. сразу меряем, куда переехал bottleneck #2,
4. ускоряем #2,
5. принимаем только суммарный wall gain.

## Lane contracts для follow-up

### L1 (ROCm H39 active)

- backend: ROCm (`build-rocm-vec/bin/llama-server.exe`)
- model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- `ctx=12288`, `batch=6144`, `ubatch=2048`
- KV: `q4_0/q4_0`
- `spec=none`, no reuse, thinking on
- task: `quick/triage_diff`

### L2 (Vulkan long-context diagnostic)

- backend: Vulkan (`build-vulkan/bin/llama-server.exe`)
- model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- `ctx=65536`, `batch=8192`, `ubatch=1024`
- KV: `q4_0/q4_0`
- `spec=none`, no reuse

## Track A: H39 Q3_K micro -> route-chain

### A0. Baseline lock

- собрать clean r3 baseline на L1;
- зафиксировать fused/direct split и top buckets.

### A1. Candidate gate (до runtime)

- resource gate: regs/occupancy/LDS;
- reject, если доминирующий fused bucket ухудшается уже на resource/timing trace.

### A2. Runtime gate

- сначала r1, затем обязательный r3;
- если r1 plus и r3 tie/minus, классифицировать как noise и не продвигать.

### A3. Bottleneck-shift capture

- для каждого кандидата заполнить:
  - what improved,
  - what regressed,
  - new bottleneck id.

## Track B: Vulkan route validity -> route-chain

### B0. Validity first

- static scout + route trace + pipeline stats до speed claim;
- без route validity не запускать «победный» benchmark.

### B1. Candidate type restriction

- не брать одиночные toggle/nearby-tile кандидаты;
- брать только route-body candidates (Q3_K main path или FA main loop).

### B2. Accept rule

- pp-screen плюс должен повториться на lane-level run;
- если pp плюс и lane tie/minus, это not-keep и признак смещения узкости.

## Track C: Stack test (главная проверка пользовательской гипотезы)

### C0. Two-step stack on same lane

- stack-1: патч под bottleneck #1;
- stack-2: патч под новый bottleneck #2 после stack-1.

### C1. Acceptance

- keep только если stack-2 > stack-1 > baseline по r3;
- если stack-1 вырос, а stack-2 не растет, значит либо #2 выбран неверно, либо #1 выигрыш был шумом.

## Минимальные артефакты на каждый цикл

- diagnostics r1/r3;
- trace/log с route/resource подтверждением;
- короткий markdown с таблицей:
  - baseline,
  - candidate/stack,
  - delta,
  - bottleneck before/after,
  - decision.

## Stop rules

- три подряд цикла без устойчивого r3 прироста на том же lane -> stop и пересборка гипотезы;
- любой route-validity fail -> immediate reject без broad sweep;
- любой candidate с ресурсным cliff (резкий рост regs/LDS/scratch) -> reject до r3.

## Execution Progress (2026-05-23)

- Done: L1 baseline lock (`E187`): r1/r3/resource snapshot.
- Done: route evidence + fusion control (`E188`):
  - synctrace confirms active Q3_K direct/fused route (`ncols_dst=1`, `small_k=1`, `nwarps=2`),
  - paired `r3` for `GGML_CUDA_DISABLE_FUSION=1` is negative (`-1.89%`), so fusion-disable remains control-only.
- Done: first guided route-body design gate (`E189`):
  - local decode-only improvements below `+2%` are too small for this L1 wall mix;
  - the only currently plausible Q3_K follow-up is a streaming fused pair-dot candidate, and only if resource gate avoids the E165 register/occupancy cliff.
- Done: code-backed pair-dot probe (`E190`):
  - resource gate was locally positive (`615.144 -> 552.412 ms` on the dominant fused Q3_K bucket, regs `84 -> 95`, occupancy `87.5% -> 100%`);
  - paired real-context r3 rejected it (`12.9580 -> 12.8560 TPS`, decode `31.4433 -> 31.3267 tok/s`);
  - conclusion: y-reuse inside the existing fused MMVQ loop is not the current limiting cost; local bucket wins can be eaten by graph/runtime and prompt/decode shifts.
- Done: bottleneck recapture (`E191`/`E192`):
  - real-context trace with prompt tokens `7489` shows parsed `MUL_MAT forward` is dominated by `cublas_backend|q3_K` (`3891.530 ms`, `78.70%`), while MMVQ is only `148.009 ms` on the short diagnostic run;
  - current Q3_K cuBLAS split is `5213.358 ms`: `src0_convert_ms=1637.070`, `src1_ms=364.309`, `gemm_ms=3203.883`;
  - repeated staging remains real (`1396` Q3_K route rows, `698` unique keys, all repeated, max `4` calls per key), but E104/E105 already reject persistent fp16 cache and existing-MMQ forcing.
- Done: cheap follow-up gates (`E193`/`E194`):
  - `--no-mmap` is negative on the full-offload ROCm L1 lane (`12.7743 -> 12.6940 TPS`), so E125's CPU fallback residency lesson does not transfer here;
  - replacing RDNA4 `sudot4(true,true)` with `sdot4` fails the ROCm clang 7.1 `gfx1201` build gate because `sdot4` needs target feature `dot1-insts`; the temporary patch was reverted and the ROCm build was restored.
- Done: H39 static-fusion gate (`E195`):
  - static Q3_K SWIGLU/no-bias specialization lowered one fused bucket's resource profile (`ncols_x=5120` regs `84 -> 46`, occupancy `87.5% -> 100%`) but did not lower fused Q3_K timing (`580.240 -> 580.369 ms`);
  - clean r1 regressed (`31.9110 -> 30.6142 TPS`, decode `32.45 -> 31.11 tok/s`), so the patch was reverted and build restored.
- Done: decode-heavy recapture (`E196`):
  - current clean ROCm r3 is `31.9233 TPS` / `32.3833 tok/s` decode;
  - current clean Vulkan r3 is `40.8007 TPS` / `41.795 tok/s` decode, so ROCm still needs about `1.278x` decode speedup for parity;
  - fresh ROCm Q3_K route split remains `mul_mat_vec_q_fused 56.95%`, `mul_mat_vec_q_direct 31.33%`, `mul_mat_q_direct 11.72%`;
  - fresh Vulkan perf-log remains `MUL_MAT_VEC q3_K 72.32%` and `MUL_MAT_ADD_VEC q3_K 27.68%`;
  - top aligned shapes are still `m=17408,n=1,k=5120`, `m=5120,n=1,k=17408`, `m=10240,n=1,k=5120`, `m=6144,n=1,k=5120`.
- Next guided step:
  - do not repeat pair/preload-style shared-`q8_1` helpers without a fresh measured load-pressure signal;
  - do not repeat `--no-mmap` or primitive `sdot4` source swaps unless a new lower-level residency/toolchain feature gate changes the premise;
  - do not pursue static branch-removal/no-bias fusion micro-specializations without instruction-level proof; lower VGPR alone is not enough on this route;
  - for practical real-context wall, choose a larger H35 route: non-persistent fused/direct Q3_K x F16 or graph scheduling that avoids repeated source staging without broad fp16 residency;
  - for pure H39 decode parity, E196 reopens Q3_K route-body work only if it changes topology toward the Vulkan q8_1 matvec family without the known rejected failure modes: lower grid width, VDR2 register cliff, pair-dot live-state growth, or static fusion branch removal;
  - keep the strict sequence `baseline r3 -> resource/timing trace -> candidate r3 -> post-check`.
