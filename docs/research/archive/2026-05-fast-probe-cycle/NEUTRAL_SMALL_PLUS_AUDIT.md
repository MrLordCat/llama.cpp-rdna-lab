# Ревизия нейтральных и малоположительных экспериментов

Дата: 2026-05-23

## Что именно отобрано

Источник: строки из [docs/research/RESULTS_LOG.md](docs/research/RESULTS_LOG.md) по 22 ID, где результат был:

- tie/noise, или
- небольшой плюс без устойчивого подтверждения/промоута в default, или
- локальный выигрыш в hotspot без итогового устойчивого выигрыша по lane.

Список ID: E010, E011, E012, E032, E044, E044-R2, E048, E055, E058, E060, E069, E091, E110, E120, E127, E148, E157, E158, E160, E161, E165, E180.

## Краткий список

| ID | Краткий итог | Статус |
| --- | --- | --- |
| E010 | мелкие MoE плюсы по части точек, смешанный профиль | iterate/no-go для текущей раскладки |
| E011 | +0.2% шум, trace-регресс, целевой gate не активировался | reject |
| E012 | +1.12% и hotspot-плюс, но оставлено только как knob | keep-as-knob |
| E032 | +1.32% trace-only, целевой F32 SSM путь не включился | reject |
| E044 | лучший вариант +0.11% шум | reject |
| E044-R2 | 0.00% tie | reject |
| E048 | +0.10% шум | reject |
| E055 | +0.78% lane, но локально слишком малый выигрыш convert | reject |
| E058 | +0.42% на одном варианте, медиана хуже | reject |
| E060 | +0.07% к MTP (почти tie) | opt-in only |
| E069 | +0.13% в лучшем code probe, медиана не лучше | reject |
| E091 | +1.09%/+2.57%, но позже признано invalid layout риск | downgraded/no promotion |
| E110 | tie/noise | reject |
| E120 | +0.43% tie-break плюс большой выигрыш по VRAM | keep route choice |
| E127 | +1.13% r1, внутри шума | reject |
| E148 | +0.08% noise/tie | reject |
| E157 | +0.82% на C01, в active lane шум | hold/reject default |
| E158 | +0.10% на C01, в active lane хуже | superseded/reject default |
| E160 | r1 +1.23%, r3 ушел в минус | reject |
| E161 | +0.28% decode шум, альтернативный стек резко хуже | reject |
| E165 | r1 плюс, но доминирующий bucket стал медленнее | reject |
| E180 | r1 плюс, r3 минус | reject |

## Детальный старт-анализа: почему «не дожали»

Ниже не просто констатация reject, а первичная причинность по маршруту вычислений.

### 1) Сняли локальную узкость, но выигрыш съела новая узкость (shift bottleneck)

- E165: идея preload для fused Q3_K уменьшала повторные загрузки, но подняла регистры 84 -> 136 и уронила occupancy 87.5% -> 68.75%. В итоге главные fused buckets замедлились. Узкость переехала из memory-load в register-pressure/scheduling.
- E055: небольшой плюс lane и небольшое улучшение src0_convert, но вклад слишком мал относительно полного пути. Узкость осталась в суммарном маршруте (GEMM и другие части prefill/decode), а не в store-пакетировании.
- E148: убрали часть mask prepass/sync идеи для FA, но основной long-KV cost остался в теле FA шейдера (K/V dequant + softmax/PV). Узкость переехала не туда, где ожидали.

### 2) Маршрут не активировался или геометрия была невалидна

- E011: целевой gate для нужных shape не активировался на данном lane.
- E032: F32 SSM route не переключился (остался cublas_backend), поэтому измеряли почти тот же путь.
- E091: первоначально выглядело положительно, но позже E093 показал invalid static layout риск для wn48 при BN=128.

### 3) Эффект на уровне шума и/или слабая статистическая опора

- E044, E044-R2, E048, E058, E110, E127, E148, E157, E158, E160, E161, E180.
- Общий паттерн: r1-плюс не удерживается на r3, либо медиана не улучшается, либо активный lane не подтверждает C01-локальный выигрыш.

### 4) Локальный плюс признан полезным, но только как узкий knob или route-choice

- E012: правильный пример «малый плюс, но не мусор». Hotspot улучшился, aggregate чуть вырос, оставлено как knob, не как default.
- E120: небольшой плюс по TPS, но крупная практическая польза по памяти (минус 552 MiB KV). Это не «шум», а корректный tie-break по системной устойчивости.
- E060: +0.07% к MTP, поэтому только opt-in fallback chain без default claim.

## Кластеризация для следующей глубокой итерации

### Кластер A: H39 Q3_K micro-ядро (E157, E158, E160, E161, E165, E180)

Вероятная причина безрезультатности:

- микро-оптимизации конкурируют за один и тот же бюджет регистров/occupancy,
- часть выигрыша видна только в C01 срезе и пропадает на active H39,
- r1-эффекты не переживают r3.

Что проверять глубже:

- строгий pre-gate по regs/occupancy до runtime,
- сравнение не только aggregate TPS, но и доли fused/direct внутри Q3_K,
- отдельные A/B на одинаковом route split после E151.

### Кластер B: runtime/env toggles (E044, E044-R2, E048, E058, E110)

Вероятная причина безрезультатности:

- это low-ceiling рычаги, уже близко к локальному насыщению,
- переключатели меняют backend-политику, но не меняют доминирующее ядро.

Что проверять глубже:

- прекращать новые broad env sweeps без route-level доказательства,
- использовать их только как negative controls для подтверждения причинности.

### Кластер C: Vulkan route illusions (E069, E091, E148)

Вероятная причина безрезультатности:

- измеряли близкие к активному пути микроформы, но не меняли главный cost center,
- часть «плюса» была связана с невалидной геометрией или route ambiguity.

Что проверять глубже:

- сначала route validity/scout/pipeline gate,
- затем только route-level изменения в главном Q3_K или FA теле.

## Первые выводы по вашей аналогии «дорог и узких мест»

Аналогия полностью подтверждается:

1. В нескольких кейсах локальный ремонт дороги сделан (E055, E148), но поток уперся в следующую узкость дальше по трассе.
2. В ряде кейсов расширение участка добавило «штраф на развязке» (E165: регистры/occupancy), и суммарный трафик стал хуже.
3. Устойчивый прогресс появляется только там, где одновременно контролируются:
   - валидность активного маршрута,
   - локальная доля в общем wall-time,
   - ресурсный профиль (regs/LDS/occupancy),
   - подтверждение r3 на том же lane contract.

## Что делать дальше (следующий шаг глубокого разбора)

Приоритетный порядок:

1. Кластер A (H39 Q3_K micro) как главный кандидат на «недожатость».
2. Кластер C (Vulkan route illusions) как главный риск ложноположительных микро-побед.
3. Кластер B оставить как контрольный, не как направление ускорения.

Отдельно: E012 и E120 не считать провалами. Это полезные точечные решения (knob и route tie-break по памяти).

## Phase 2: карта «узкость до/после»

| ID | Что ускоряли | Что стало новой узкостью | Вывод |
| --- | --- | --- | --- |
| E165 | повторное чтение `q8_1` в fused Q3_K | register pressure и падение occupancy | локальная оптимизация памяти проиграла scheduler path |
| E055 | Q3_K fp16 conversion store path | остальной prefill маршрут (GEMM + нецелевые участки) | локальный выигрыш слишком мал по доле маршрута |
| E148 | FA mask prepass/sync | main FA shader body (dequant + softmax/PV) | сняли вторичный overhead, не главный cost center |
| E160/E180 | микро-layout/арифметика | статистическая нестабильность и lane-shape drift риска | r1 плюс без r3 подтверждения ненадежен |
| E091 | tile-вариант route | route validity/layout correctness | сначала валидность маршрута, потом speed claim |

## Что похоже на «не дожали», а что нет

Похоже на недожатость (можно вернуться при правильном протоколе):

- E012: есть и runtime, и hotspot сигнал; это валидный кандидат на дальнейший controlled squeeze.
- E055: есть слабый локальный signal в convert path; может дать эффект только в составе более крупного route-stack, не как одиночный патч.
- E165: идея полезная по направлению, но текущая реализация уперлась в регистры; нужен redesign с budget-first, а не «еще preload».

С высокой вероятностью не недожали (а просто низкий ceiling/не тот маршрут):

- E044, E044-R2, E048, E058, E110: env/runtime toggles без смены доминирующего ядра.
- E148: prepass removal без изменения main FA compute body.
- E127: простая prefetch-идея на CPU Q3_K при ограничении другим участком.

Требует строгой валидации маршрута перед любыми повторными циклами:

- E091, E069: риск route-иллюзий и ложноположительных микро-улучшений.

## Протокол «ускоряем весь маршрут, а не один участок»

Перед кодом:

1. Зафиксировать lane contract (ctx/batch/ubatch/KV/spec/reuse/thinking/max_tokens/tasks).
2. Построить текущую долю маршрута: top-op share + top-shape share.
3. Для кандидата оценить required local gain из доли маршрута; если ceiling ниже порога, не кодить.

Во время кода:

1. Запускать resource gate до runtime gate (regs/LDS/occupancy/route validity).
2. Не принимать r1-плюс без r3 подтверждения на том же lane.
3. Для каждого плюса фиксировать, какой следующий bottleneck вырос после патча.

После кода:

1. Если локальный hotspot улучшился, а wall не вырос, явно помечать «bottleneck shifted».
2. Следующий эксперимент выбирать по новому bottleneck, а не по старой гипотезе.
3. Обновлять этот файл отдельной строкой в карте «узкость до/после».

## Конкретный план следующего углубления

Шаг 1: H39 micro-route replay на активном lane

- Цель: доказать, что проблема E157/E158/E160/E161/E165/E180 в budget/route, а не в «плохой удаче».
- Минимум: route split fused/direct + resource snapshot + r3 pair.
- Критерий keep: устойчивый r3 плюс и отсутствие деградации доминирующего bucket.

Шаг 2: Vulkan route validity first

- Цель: исключить повтор E091/E069/E148 класса.
- Минимум: static scout + route trace + pipeline stats до benchmark claim.
- Критерий keep: валидный route и рост на основной lane-метрике, не только на pp micro-screen.

Шаг 3: stack testing вместо single-knob testing

- Цель: проверить гипотезу пользователя о последовательных узкостях.
- Минимум: A/B не только для одного патча, а для последовательности из 2-3 патчей по цепочке bottleneck.
- Критерий keep: суммарный wall gain на lane при стабильном качестве/валидности маршрута.

## Phase 3: глубокий route-chain pass по каждому ID

Правило этого pass: если локальный счетчик стал лучше, но lane TPS не вырос, это не финальный reject само по себе. Нужно записать, куда переехала узкость, и только потом решить: есть ли следующий маршрут для кода или это низкопотолочный участок.

| ID | Локальный сигнал | Что съело или ограничило выигрыш | Следующая узкость / маршрут | Решение после глубокого pass |
| --- | --- | --- | --- | --- |
| E010 | `pp512 +0.64%`, `pp2048 +0.29%`, `tg128 +4.26% r1` на MoE35B | Trace shows `rdna4_staging=0`: фактический staging-route не активировался, результат в основном fallback/noise; позднее E034 показал scoped staging `-2%..-3%` | MoE route split: routed IQ experts already go through direct quant route, shared `q6_K`/F32 pieces and decode remain outside this staging fix | Не брать как H39-кандидат. Доводить только в отдельном MoE lane через новый LDS-footprint route или shared-expert route, не повторяя текущий staging knob |
| E011 | `+0.2%` wall-like noise | Target condition did not activate: trace had `ncols_max=192`, while candidate gate targeted the wrong shape | Route-validity gate, not optimization | Закрыт. Повторять только если свежий trace снова покажет exact target shape |
| E012 | Aggregate small plus and hotspot time improved with `skmin=144` | Wall gain small; knob changes StreamK policy but not the dominant active H39 Q3_K decode route | Useful as narrow C01/pre-H39 knob, not current decode bottleneck | Keep-as-knob. Can be stack-tested only after current L1 route has a real r3 baseline pair |
| E032 | Trace-only `+1.32%` | Intended F32 SSM route did not flip from `cublas_backend`; measured deltas are other-route noise | Need actual F32 MMF support for this shape before speed work | Closed for current code. No route activation, no next bottleneck claim |
| E044 | Tiny allocator/runtime deltas around tie | Broad runtime knobs changed environment, not dominant kernels; MMF-F32 and no-batched F32 variants tied/regressed | Kernel-local route, not allocator/global env | Closed as control. Use only as negative control when suspecting allocator/residency |
| E044-R2 | `0.00%` tie | Disabling RDNA4 batched F32 cublas did not change wall materially | F32 backend is not the current limiter | Closed |
| E048 | hipBLASLt `+0.10%` aggregate / `+0.04%` prompt | Large GEMM library route did not materially improve despite high MUL_MAT share | Q3_K staging/conversion or custom fused route, not library toggle | Closed for toggles; keep as evidence to avoid more hipBLASLt sweeps |
| E055 | Full lane r1 `+0.78%`; Q3_K convert local `-1.58%`; hot shape only `-0.41%` | Local conversion improvement far below required `~25%` local gate; rest of prefill/GEMM route dominates | Structural Q3_K conversion/layout route, not store packing | Candidate family closed. Reopen only with a design that removes much more convert work |
| E058 | hipBLASLt r3 best `+0.42%` aggregate | Median below control and control had a slow outlier; no stable prompt/decode win | Same as E048: not a route-body fix | Closed/watchlist only; no promotion |
| E060 | `ngram-mtp +0.07%` vs MTP | Spec chain viability test; no meaningful wall gain and MTP route is not active/default compatible in this repo | Needs model-supported MTP plus coverage/effective acceptance proof | Opt-in only, not current TPS route |
| E069 | Vulkan Q3_K packed32 scale probe `+0.13%` aggregate, median not better | Cheap MMVQ knobs changed small pieces; Q3_K decode remains dominant | Deeper Q3_K MMVQ specialization, not workgroup forcing/packed scale loads | Good direction, bad probes. Future Vulkan decode work must be new route-body specialization |
| E091 | `wn48` looked positive (`+1.09% pp`, `+2.57% workload`) | E093 static scout flags layout invalid for active `BN=128`; speed likely route ambiguity/fallback artifact | Valid warptile/topology proof before benchmark | Do not use. Any tile resurrection must pass static scout and route log first |
| E110 | Fit-off exactly tied | Q4_K_S offload lesson did not transfer to Q3_K_S; route unchanged | Kernel-level Q3_K work or session route | Closed |
| E120 | Vulkan q4 KV tied/slightly faster and saves `552 MiB` KV | Not a raw speed breakthrough, but memory headroom reduces future 64k/session pressure | Practical route choice for Vulkan long-answer/session | Keep. This is a valid tie-break, not a failed speed probe |
| E127 | CPU prefetch `+1.13%` aggregate r1 but decode `+0.02 tok/s` | Hardware prefetch likely already covers regular streams; dequant/scale arithmetic remains | CPU Q3_K repack/interleaving, not simple prefetch | Closed for current prefetch patch |
| E148 | Analytic FA mask candidate `+0.08%` noise with lower VGPR in full-chunk pipeline | Mask prepass/sync was not the limiter; main q4 K/V dequant + softmax/PV body remains | FA shader-body or long-KV traversal redesign | Good causal correction. Do not repeat mask-only variants |
| E157 | C01 bit-ops `+0.81%` decode | Active H39 only `32.03 -> 32.12 tok/s`, within noise | Q3_K MMVQ arithmetic is not enough alone | Hold as possible micro ingredient, not standalone |
| E158 | C01 `rows_per_block=1 +0.08%` decode | Active H39 regressed to `30.05 tok/s`, losing E151 policy win | Keep E151 `rows_per_block=2`; next route must preserve row batching | Closed/reject default |
| E160 | r1 `+1.23%`, r3 `-0.10%` | One-run noise; modulo-hoist did not survive paired r3 | Resource/timing gate before runtime claims | Closed |
| E161 | Active-lane audit: bit-ops tied, rpb1 bad | Confirms E157/E158 were C01-shape artifacts | Preserve E151/E152 state and choose larger Q3_K route | Baseline correction kept |
| E165 | Some low-share fused bucket got faster | Dominant fused bucket slowed; regs `84 -> 136`, occupancy `87.5% -> 68.75%` | Budget-first fused redesign: reduce duplicate live arrays, keep regs near E163/E151 | Reopen only as new design, not more preload |
| E180 | q8x4 layout slightly improved decode micro-rate | Prompt/prefill regressed enough for net wall `0.9964x`; isolated transient q8 layout below ceiling | Needs to be part of larger route-stack, not alone | Closed as single patch; possible ingredient only after new bottleneck proof |

## Практический приоритет после pass

1. Не начинать с E010 для текущей H39 цели: это MoE35B route, а не dense Qwen3.6-27B decode parity. Если пользователь отдельно вернет MoE, правильный next route там: новый RDNA4 MoE LDS layout с меньшим shared footprint или отдельная shared-expert `q6_K` route, после E034 negative control.
2. Для текущего ROCm/Vulkan decode route первый реальный кандидат остается Q3_K route-body redesign: E069/E157-E165/E180 показывают, что микроправки вокруг существующей геометрии исчерпаны.
3. Следующая кодовая попытка должна быть stack-style: `baseline r3 -> resource/timing trace -> candidate r3 -> new bottleneck row`. Если hotspot time упал, но wall не вырос, следующий patch выбирается по новому top bucket, а не по старой гипотезе.
