# Large Context Research Status

## 2026-05-10 Reset

Этот документ больше не задаёт активную цель на `120k+`.

Новая позиция проекта:

- `64k` и `128k` больше не являются активным optimisation lane.
- Новый active lane: prompt-heavy стартовая точка ниже `16k` (текущий reference `ctx=12288`) с целью `25-27 TPS`.
- Причина смены курса: synthetic sentinel benchmark скрывал стоимость длинного prefill, потому что использовал prompt всего `489/410` токенов.

## Почему старый 128k вывод оказался нерепрезентативным

Два разных режима дали принципиально разные результаты:

1. `sentinel128-qwen36q3-largectx-summary.md`
  - `ctx64k`: `26.5825 TPS`
  - `ctx128k`: `26.0672 TPS`
  - вывод казался почти нейтральным, но benchmark был коротким decode-heavy sentinel.

2. `repo-real-64k128k-repo-summary.md`
  - `ctx64k`: prompt `62407` токенов, `2.3128 TPS`
  - `ctx128k`: prompt `113400` токенов, `0.8167 TPS`
  - реальное сравнение показало ratio всего `35.31%`.

Следствие: прежние 120k/160k планы оптимизации были построены на слишком лёгком workload и больше не должны использоваться как активный roadmap.

## Активная цель

Оптимизировать wall-time и throughput для `Qwen3.6-27B-Q3_K_S` на ROCm в реальном prompt-heavy коридоре:

- стартовая точка ниже `16k` (текущий reference `ctx=12288`)
- реальные большие входящие prompt'ы с `--real-context-mode repo-snapshot`
- целевая метрика на стартовой точке: `25-27 TPS`
- приоритет метрики: aggregate completion TPS by wall-time в no-reuse режиме

## Что считать reference, а что archived

- `scripts/repo_snapshot_context_bench.py`: reference tool для проверки реалистичного long-context scaling.
- `scripts/large_context_realworld_bench.py`: historical experiment tool; не использовать как главный аргумент о производительности без проверки prompt size.
- 120k/160k hypothesis tables ниже считаются архивными и не формируют текущий план работ.

## Архив: предыдущая гипотеза 120K+

Ниже сохранён старый материал как исторический контекст, но он больше не является действующим execution plan.

## Bottleneck Analysis

1. **Speculative Decoding Overshoot**
   - ngram-mod generates 128 draft tokens but only accepts ~23 (18% rate)
   - Verification overhead may exceed generation gains
   - Action: test `spec-profile=none` to quantify speculative cost

2. **FATTN Kernel Switching**
   - On large ctx, attention may switch from VEC to TILE kernel (known performance cliff)
   - Investigate FATTN threshold adjustment for RDNA4 with ctx=120K+

3. **KV Cache Bandwidth**
   - q4_0 KV: 1152 MiB per 65K tokens
   - At 120K: ~2.1 GB, approaching L3 cache limits
   - Consider: q6_K or mixed q4_0/f16 strategy (if profile allows)

4. **ROCm Kernel Occupancy**
   - Small ubatch on large seq_len may under-saturate GPU compute
   - Action: test ubatch=256, ubatch=1024 variants

5. **Prompt Caching Overhead**
   - 1005.90 ms prompt save/restore on second request (10x slower than first)
   - May indicate inefficient checkpoint serialization
   - Action: profile with `--no-warmup` and fresh slot allocation

## Experiment Plan

### Phase 1: Speculative Mode Sweep (3 runs each)

| Label | Spec Mode | Expected | Reason |
|-------|-----------|----------|--------|
| `spec-none-120k` | none | ~15-18 TPS | baseline without spec overhead |
| `spec-ngram-mod-120k` | ngram-mod | ~8-10 TPS | current config (acceptance too low) |
| `spec-ngram-simple-120k` | ngram-simple | ~10-12 TPS | fewer draft tokens, lower overhead |

```bash
# None (baseline generation without speculative)
python scripts/large_context_realworld_bench.py \
  --label-prefix phase1-spec-none \
  --ctx-values 122880 \
  --spec-profile none \
  --runs 3

# Ngram-mod (current)
python scripts/large_context_realworld_bench.py \
  --label-prefix phase1-spec-ngram \
  --ctx-values 122880 \
  --spec-profile ngram-mod \
  --runs 3

# Ngram-simple (fewer drafts)
python scripts/large_context_realworld_bench.py \
  --label-prefix phase1-spec-ngram-simple \
  --ctx-values 122880 \
  --spec-profile custom \
  --server-extra "--spec-type ngram-simple" \
  --runs 3
```

**Decision Point**: If `spec-none` is significantly faster (>15 TPS), speculative is the problem.

### Phase 2: Batch Parameter Tuning (if Phase 1 shows improvement window)

| Label | Batch | UBatch | Expected | Reason |
|-------|-------|--------|----------|--------|
| `ub256-120k` | 4096 | 256 | ? | smaller ubatch = more kernel launches but more stable |
| `ub768-120k` | 4096 | 768 | ? | intermediate |
| `ub1024-120k` | 4096 | 1024 | ? | aggressive ubatch (may hit kernel limit) |
| `b6144-120k` | 6144 | 512 | ? | larger batch = more throughput, higher latency |

### Phase 3: KV Cache Strategy (if bandwidth is confirmed bottleneck)

- Test q6_K (slower load, better quality) vs q4_0 (current)
- Mixed approach: f16 for recent tokens, q4_0 for older

### Phase 4: FATTN Tuning (kernel-level)

- Adjust FATTN VEC→TILE threshold for RDNA4 with large ctx
- Profile with `--verbose` to capture kernel selection logs

## Execution Log

### 2026-05-10 Session Start

- Created large_context_realworld_bench.py with 120K/160K focus
- Updated BENCHMARKS.md with realistic scenario description
- Logged baseline: PP=215 TPS, TG=8.5 TPS, spec_acceptance=18%
- **Next**: Run Phase 1 spec sweep to confirm speculative overhead

---

## Quick Command Reference

**Set up baseline (1 run for quick iteration)**:
```bash
python scripts/large_context_realworld_bench.py \
  --label-prefix test \
  --ctx-values 122880 \
  --runs 1 \
  --spec-profile none
```

**Full experiment (3 runs for confidence)**:
```bash
python scripts/large_context_realworld_bench.py \
  --label-prefix full-exp \
  --ctx-values 122880,163840 \
  --runs 3 \
  --spec-profile none
```

**Parse results**:
- Check `build_logs/agent-workload/<label>-largectx-summary.md`
- Compare `aggregate_tps` column across variants
- Look for >20% improvement to justify a configuration change
