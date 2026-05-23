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
- Next guided step:
  - do not repeat pair/preload-style shared-`q8_1` helpers without a fresh measured load-pressure signal;
  - choose a larger Q3_K route change: launch topology, graph/fusion policy, or a new specialized fused route for the next measured top bucket;
  - before another patch, capture the post-E190 top bucket and verify whether the bottleneck moved to prompt/prefill, graph scheduling, direct Q3_K, or non-MMVQ ops;
  - keep the strict sequence `baseline r3 -> resource/timing trace -> candidate r3 -> post-check`.
