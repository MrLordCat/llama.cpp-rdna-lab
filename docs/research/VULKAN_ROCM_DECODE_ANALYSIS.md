# Vulkan vs ROCm Decode Performance: Глубокий Анализ

Дата: 2026-05-24

## 0. Codex Audit / Status Check

Статус документа: **полезный как стартовая гипотеза, но не достаточный как
план разработки ROCm decode parity без правок**.

Что подтверждается текущими локальными данными:

- Decode gap реален. После driver `32.0.31007.5012` E116 измерил ROCm q4
  decode-focused route на `29.1685 TPS` / `29.625 tok/s decode eval`, а
  Vulkan q4/f16 route на `39.8801-40.2753 TPS` / `40.8683-41.2283 tok/s`.
- Vulkan advantage особенно важен для decode-heavy и длинной генерации. Для
  prompt-heavy/cold-first ROCm всё ещё остаётся сильнее из-за prefill.

Что нужно исправить в исходной интерпретации:

- ROCm backend **не является backend без fusion**. В текущем `ggml-cuda` уже
  есть `RMS_NORM+MUL`, `RMS_NORM+MUL+ADD`, `ROPE+VIEW+SET_ROWS`,
  `UNARY+MUL`, SSM/activation fusion, FFN `MUL_MAT_VEC` fusion и HIP/CUDA graph
  capture. Поэтому тезис "каждый op отдельный kernel launch" неверен.
- У Vulkan действительно есть более широкая generic fusion surface, включая
  `RMS_NORM_MUL_ROPE(_VIEW_SET_ROWS)`, но по текущим логам активного Qwen route
  виден в основном `RMS_NORM_MUL`, а не доказанная активная
  `RMS_NORM_MUL_ROPE` ветка. Для Qwen M-RoPE/позиционного route это нужно
  подтверждать fresh trace, а не считать автоматически.
- Line-count аргумент нужно считать аккуратнее: один `ggml-vulkan.cpp` около
  `15.1k` строк плюс Vulkan shaders около `15.8k`, но `ggml-cuda/*.cu/*.cuh`
  суммарно около `33.2k` строк. Vulkan codebase больше по pipeline/shader
  специализациям, но не "ROCm в 3x меньше" если учитывать весь HIP/CUDA backend.
- Pipeline caching/dispatch overhead не доказан главным фактором: ROCm имеет
  graph capture/update path. Перед работой над launch overhead нужно сначала
  доказать, что decode graph реально не capture/replay friendly на текущей
  lane.

Первичный ceiling из старого ROCm decode trace показывает, почему "просто
добавить fusion" не закрывает gap:

- Old C01 decode trace: `MUL_MAT forward+fused` примерно `61.9%` CUDA-node
  time; `RMS_NORM+ROPE+SET_ROWS` вместе только `8.4%`.
- Чтобы ROCm q4 decode eval `29.625 tok/s` догнал Vulkan q4 `40.868 tok/s`,
  нужен общий speedup около `1.38x`.
- Если ускорять только matmul/direct-Q3 route, ему нужен примерно `1.80x`
  local speedup. Если ускорять matmul + norm/rope/set_rows вместе, всё равно
  нужен примерно `1.64x` local speedup.
- E149 fresh post-driver diagnostic traces усилили этот вывод. Non-sync route
  trace показал `MUL_MAT forward` `77.84%` parsed CUDA-node time,
  `FLASH_ATTN_EXT` `5.29%`, а `RMS_NORM+ROPE+SET_ROWS` вместе только около
  `3.26%`; route counts указывают на Q3_K direct decode:
  `mul_mat_vec_q_direct,q3_K` `929` hits и `mul_mat_q_direct,q3_K` `349` hits.
  Более дорогой sync companion (`max_tokens=16`, graph disabled, first section
  excluded) дал менее enqueue-biased split: `MUL_MAT forward+fused` `53.80%`,
  `RMS_NORM+ROPE+SET_ROWS` `11.83%`, `FLASH_ATTN_EXT` `1.46%`. Это повышает
  ceiling для norm/rope cleanup относительно non-sync trace, но всё равно не
  делает standalone fusion достаточной для `1.38x` gap.
- E149 Vulkan q4 perf comparator показывает похожую картину с другой
  реализацией: в decode sections Q3_K `MUL_MAT_VEC` `50.67%`,
  Q3_K `MUL_MAT_ADD_VEC` `19.38%`, `RMS_NORM_MUL` `3.79%`,
  `FLASH_ATTN_EXT` `1.11%`, `ROPE+SET_ROWS` `0.60%`. То есть переносимое
  преимущество нужно искать в Q3_K/QK direct decode kernel family, а не в
  широком тезисе "у Vulkan есть fusion".
- Первый E149 route-delta parser ещё сильнее сузил цель: если брать ROCm
  decode-only Q3_K sections, `mul_mat_vec_q_fused` занимает `63.78%` Q3_K
  matvec time, direct `mul_mat_vec_q_direct` `36.22%`. Главные совпадающие
  формы с Vulkan: `m=17408,n=1,k=5120`, `m=5120,n=1,k=17408`,
  `m=10240,n=1,k=5120`, `m=6144,n=1,k=5120`. Значит следующий кодовый аудит
  должен начинаться с fused MMVQ Q3_K FFN gate/up/down route, а не с
  generic RMS/RoPE fusion.
- E150 проверил обратную гипотезу: отключение ROCm fusion через
  `GGML_CUDA_DISABLE_FUSION=1` не помогает, а регрессирует clean short decode
  `30.08 -> 28.61 tok/s`. Значит fusion не "лишняя"; отставание нужно искать
  внутри качества fused Q3_K MMVQ kernel/resource policy.
- E151 дал первый подтверждённый ROCm code win: RDNA4 `Q3_K/ncols_dst=1`
  в MMVQ возвращён на `nwarps=2`. На current-tree r3 gate clean post-rebuild
  был `28.1123 TPS` / `29.77 tok/s` decode, candidate стал `30.3145 TPS` /
  `32.2467 tok/s` decode (`+8.32%` decode). Живой `llama-server` sanity
  прошёл нормально: ответ начинался с обычного `Thinking Process:`, без
  повторяющихся символов или `wm32-wn32`-style corruption.
- E196 обновил decode-heavy baseline после серии отрицательных Q3_K probes:
  current clean ROCm r3 `31.9233 TPS` / `32.3833 tok/s`, current clean Vulkan
  r3 `40.8007 TPS` / `41.795 tok/s`. Разрыв остаётся `+27.81%` aggregate в
  пользу Vulkan, то есть ROCm нужен примерно `1.278x` decode speedup. Свежий
  route delta снова указывает на Q3_K route body: ROCm Q3_K split
  `mul_mat_vec_q_fused 56.95%`, `mul_mat_vec_q_direct 31.33%`,
  `mul_mat_q_direct 11.72%`; Vulkan split `MUL_MAT_VEC q3_K 72.32%`,
  `MUL_MAT_ADD_VEC q3_K 27.68%`. Следующий ROCm-кандидат должен менять
  Q3_K topology, а не generic fusion/graph/static-branch/occupancy-only policy.
- E197 проверил самый прямой topology перенос "Vulkan-like wider subgroup":
  `Q3_K/ncols_dst=1/small_k=1` был env-gated на wave64 `block=(64,1,1)` при
  сохранении `rows_per_block=2`. Route activation сработал и build прошёл, но
  hot buckets стали чуть медленнее: fused `ncols_x=5120`
  `676.110 -> 681.567 ms`, direct `5120` `554.893 -> 557.519 ms`, fused
  `17408` `415.253 -> 418.187 ms`; clean r1 также не вырос
  (`31.6788 TPS`, decode `32.365 tok/s`). Это закрывает простую гипотезу
  "ROCm проигрывает из-за shared cross-warp reduction"; текущий `(32,2,1)`
  small-k route, похоже, выигрывает от своего K-split/latency-hiding schedule.
  Следующие ROCm parity кандидаты должны уменьшать реальную Q3_K работу/трафик
  или менять layout, а не только форму редукции.
- E198 проверил более близкий к Vulkan route-level механизм: cache уже
  квантизованного q8 activation внутри одного HIP graph compute. Гипотеза была
  частично верна: trace показал `303` hits / `777` misses (`28.1%`) на
  повторных `attn_norm-*` и `attn_post_norm-*`. Но clean same-binary A/B не
  дал прироста: baseline r1 `31.9368 TPS`, cache r1 `31.8573 TPS`, decode
  `32.50 -> 32.405 tok/s`. Значит standalone q8 activation reuse не является
  главным Vulkan преимуществом: buffers малы (`~11.5-39.2 KB`), HIP graph
  capture уже убирает host-launch стоимость, а Q3_K dot/dequant body остаётся
  основным лимитером. Этот код был reverted.
- E199 проверил следующий Vulkan-like layout механизм до runtime-кода. У
  Vulkan `ggml_vk_device_type_size()` фактически хранит Q3_K/Q6_K device blocks
  с `+2` bytes padding (`110 -> 112` для Q3_K), поэтому packed32 route не
  является только shader helper. ROCm CUDA/HIP сейчас копирует raw GGUF bytes,
  а все `block_q3_K *` kernels предполагают `110`-byte stride. Замена всей
  Q3_K storage на padded layout стоила бы только `+179.47 MiB` (`1.818%`) для
  этой модели, но transient repack, duplicate padded copy и локальная
  `vecdotq.cuh` 32-bit load rewrite отвергнуты аналитически. Если переносить
  это преимущество в ROCm, нужен отдельный backend-private storage branch с
  padded set/get/view offsets и full Q3_K kernel correctness audit.
- E200 разложил этот storage branch на проверяемый cut. Минимальная корректная
  ветка должна покрыть не только MMVQ decode, но и non-split CUDA buffer
  set/get, Q3_K dequant/getrows, MMVQ, MMQ, затем split/async/copy/view
  offsets. Current-tree cheap smoke через `test-backend-ops` проходит для
  Q3_K `MUL_MAT` (`m=16,n=1,k=256` и `m=1,n=64,k=256` на `ROCm0`), поэтому
  следующий padded-storage код должен сначала сохранить этот correctness gate
  под env knob, а уже потом идти к real-server speed.
- Sequential graph-disable diagnostic не показал пользы от launch/graph
  гипотезы на short-decode gate: clean `27.1129 TPS` / `29.15 tok/s decode`
  против `GGML_CUDA_DISABLE_GRAPHS=1` `27.2063 TPS` / `29.28 tok/s`.

Вывод аудита: главный ROCm decode parity трек должен начинаться с **fresh
post-driver route trace и Q3_K direct decode/MMVQ-MMQ route proof**. E151 уже
подтвердил первую маленькую часть этого направления и снизил short-decode gap
к Vulkan q4 примерно с `1.37x` до `1.27x`, но parity ещё не закрыт.
`RMS_NORM+MUL+ROPE` fusion остаётся измеряемым secondary candidate. Для 64k
decode отдельно нужен FA/KV trace, потому что там разрыв ROCm/Vulkan может быть
уже attention/long-KV, а не short-decode matvec.

Важно для чтения остального файла: sections ниже оставлены как исходная
гипотеза/контекст. Утверждения вроде "ROCm не использует fusion в decode path"
и "fusion - главный фактор" считаются **superseded by audit** до тех пор, пока
новый trace не докажет конкретную недостающую fusion-ветку с достаточным wall
ceiling.

## Резюме

Vulkan backend показывает **~30-40 TPS decode** против ROCm **~28-30 TPS decode** на одной и той же машине (RX 9070 XT, Qwen3.6-27B-Q3_K_S, ctx=12288, full offload). Это **~30-40% преимущество Vulkan в decode TPS**, что кажется контринтуитивным — ведь ROCm/HIP должен быть "родным" для AMD GPU.

**Ключевой вывод:** Да, возможно перенести часть Vulkan-оптимизаций в ROCm, но не всё. Некоторые преимущества Vulkan фундаментально связаны с архитектурой API.

---

## 1. Эмпирические Данные

### Prompt-heavy lane (ctx=12288, b=4096, ub=512, q4_0/q4_0, spec=none)

| Backend | Wall TPS | Prompt eval TPS | Decode eval TPS |
|---------|----------|-----------------|-----------------|
| ROCm | 6.33 | 960.26 | **28.32** |
| Vulkan | 4.22 | 573.93 | **30.85** |

Vulkan decode: **+8.9%** vs ROCm, но prefill: **-40%** (573 vs 960 tok/s)

### Decode-biased lane (159 prompt tokens, 128 generated)

| Backend | Wall TPS | Prompt eval TPS | Decode eval TPS |
|---------|----------|-----------------|-----------------|
| ROCm | 27.98 | 776.05 | **29.42** |
| Vulkan | 35.29 | 518.83 | **38.81** |

Vulkan decode: **+31.9%**, wall: **+26.1%**

### 64k context lane (ctx=65536, b=8192, ub=1024)

| Backend | Wall TPS | Prompt eval TPS | Decode eval TPS |
|---------|----------|-----------------|-----------------|
| Vulkan | 1.34 | 666.62 | **36.58** |
| ROCm | 1.55 | 799.09 | **22.83** |

Vulkan decode: **+60.2%** vs ROCm! На большом контексте разрыв ещё больше.

---

## 2. Архитектурные Различия Backend

### Масштаб кода

| Backend | Строк кода | Шейдеров |
|---------|-----------|----------|
| Vulkan | ~17,500 строк (ggml-vulkan.cpp) | 150+ .comp/.glsl файлов |
| ROCm/CUDA | ~6,200 строк (ggml-cuda.cu) | Кernels inline в .cuh/.cu |

Vulkan backend почти в **3x больше** — это не случайность. Это результат лет отдельной разработки с фокусом на exhaustive optimizations.

### Pipeline Caching

**Vulkan:** Все pipelines предварительно компилируются и кэшируются. В `vk_device_struct` хранятся десятки заранее созданных pipeline объектов:
- `pipeline_matmul_f32`, `pipeline_matmul_f16`, `pipeline_matmul_bf16`
- `pipeline_dequant_mul_mat_mat[GGML_TYPE_COUNT]`
- `pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_COUNT]`
- `pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_COUNT]`
- `pipeline_dequant_mul_mat_vec_f32_f32[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT][8]`
- `pipeline_dequant_mul_mat_vec_q8_1_f32[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT][8]`
- И т.д. — каждый тип квантизации × каждый acc type × каждый batch size

На decode **нет overhead pipeline compilation** — pipeline уже готов, просто dispatch.

**ROCm:** Использует template instantiation в compile time (C++ templates в .cuh файлах). Кernels генерируются через макросы, но нет такого же exhaustive pipeline caching.

**Вывод:** Vulkan имеет преимущество в zero-overhead dispatch, но ROCm может это приблизить.

---

## 3. Kernel Fusion: Главное Преимущество Vulkan

### Vulkan Fusion

В Vulkan реализована агрессивная kernel fusion для decode path:

**rms_norm + mul + rope fusion:**
```cpp
// ggml-vulkan.cpp, ggml_vk_rms_norm()
if (ctx->num_additional_fused_ops > 0) {
    // fused rms_norm + mul
    ggml_tensor *mul = cgraph->nodes[node_idx + 1];
    ggml_tensor *other_src = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    dst = mul;
    src0 = cgraph->nodes[node_idx]->src[0];
    src1 = other_src;
}
// more than one fused op means rms_norm+mul+rope
if (ctx->num_additional_fused_ops > 1) {
    // Full fusion: rms_norm → mul → rope → [set_rows]
}
```

Vulkan имеет dedicated fused pipelines:
- `pipeline_rms_norm_mul_f32`
- `pipeline_rms_norm_mul_partials_f32`
- `pipeline_rms_norm_mul_rope_f32_f32`
- `pipeline_rms_norm_mul_rope_f32_f16`
- `pipeline_add_rms[2][2][2]`
- `pipeline_multi_add[MAX_FUSED_ADDS]`
- `pipeline_multi_add_rms[MAX_FUSED_ADDS]`

**Для decode это критически важно:** каждый слой имеет `rms_norm → mul → rope → mul → add → mul → glu → rms_norm → attention`.
Vulkan fuse-ит `rms_norm + mul + rope` в один kernel, избегая:
- 2 extra kernel launches
- 2 extra global memory reads/writes
- 2 extra pipeline switches

### ROCm Fusion

ROCm имеет `ggml_cuda_should_fuse_mul_mat()` для FFN fusion (mul_mat + mul_mat + glu), но:
- Нет dedicated fused rms_norm+mul+rope pipeline
- Fusion более ограниченный
- Каждый op — отдельный kernel launch

**Вывод:** Это **главный фактор** decode advantage Vulkan. На 65 слоях Qwen3.6, каждый слой имеет ~6-8 ops. Если Vulkan fuse-ит 3 ops в 1 kernel, экономия = ~20-25% kernel launches и memory traffic.

**Можно ли перенести?** Да, но это большая работа:
1. Создать fused rms_norm+mul+rope kernel для ROCm
2. Интегрировать в graph scheduling
3. Добавить conditional dispatch (fuse только когда shapes подходят)

---

## 4. Cooperative Matrices vs Wavefront MMA

### Vulkan Cooperative Matrices

Vulkan использует `VK_KHR_cooperative_matrix` extension для matmul:
- Dedicated hardware для matrix multiply
- Tile-based computation в LDS/shared memory
- Automatic register/LDS allocation через pipeline executable properties
- Multiple tile configurations (16x16x16, 32x32x32, etc.)

Для AMD RDNA4 Vulkan detect-ит архитектуру и выбирает оптимальные tile sizes.

### ROCm MMQ/MMVQ

ROCm использует:
- **MMQ (Matrix-Vector Quantized):** Для больших batch sizes, tile-based dequantize+mul
- **MMVQ (Matrix-Vector Quantized):** Для decode (batch=1), vec_dot-based approach
- **rocWMMA:** Для FlashAttention (vendored rocWMMA 2.0.0)

Для decode Q3_K используется `mmvq` path с `vec_dot_q3_K_q8_1` kernel.

**Ключевое различие:** Vulkan dequant_mul_mat_vec может использовать cooperative matrix tile для dequantize+mul fusion внутри одного kernel. ROCm mmvq делает dequantize и mul как separate operations внутри того же kernel, но без coop mat hardware acceleration.

**Можно ли перенести?** Частично:
- ROCm уже имеет rocWMMA для FlashAttention
- Для MMVQ можно добавить matrix-tile approach вместо vec_dot
- Но cooperative matrix API — это Vulkan-specific, ROCm использует rocWMMA/hipWMMA

---

## 5. Scratch Buffers и Memory Management

### Vulkan Scratch System

```cpp
// ggml-vulkan.cpp
struct vk_context {
    vk_buffer prealloc_x;
    vk_buffer prealloc_y;
    vk_buffer prealloc_z;
    vk_buffer prealloc_w;
    size_t prealloc_size_x;
    size_t prealloc_size_y;
    size_t prealloc_size_z;
    size_t prealloc_size_w;
};
```

Vulkan pre-allocates scratch buffers перед graph execution. Кernels могут использовать их для intermediates без extra allocations.

### ROCm Scratch

```cpp
// ggml-cuda.cu
struct ggml_backend_cuda_context {
    ggml_cuda_pool * pool;
    // VMM (Virtual Memory Management) для dynamic allocation
    // Либо legacy pool с fixed chunks
};
```

ROCm использует pool allocator, который может требовать reallocation при growth.

**Вывод:** Vulkan scratch system более эффективна для decode, где shapes предсказуемы. ROCm может это улучшить.

---

## 6. Descriptor Set vs Stream Binding

### Vulkan Descriptor Sets

Vulkan использует descriptor set layout + binding. На decode каждый kernel dispatch требует:
- Descriptor set update (buffer bindings)
- Pipeline bind
- Push constants set
- Dispatch

Но Vulkan pipeline caching делает bind очень быстрым (~микросекунды).

### ROCm Streams

ROCm использует CUDA/HIP streams. Каждый kernel launch:
- Stream select
- Kernel config (grid/block dims)
- Argument setup
- Launch

HIP kernel launch overhead выше чем Vulkan pipeline bind, но разрыв не такой большой как кажется.

**Вывод:** Это не главный фактор decode difference.

---

## 7. Что Можно Перенести в ROCm

### High Impact (feasible)

1. **Kernel Fusion:** fused rms_norm+mul+rope kernel — это 30-40% decode advantage
   - Создать `rms_norm_mul_rope` kernel в `ggml-cuda/`
   - Добавить detection в graph scheduling
   - Оценка: 2-3 недели работы

2. **Scratch Buffer Pre-allocation:** Pre-allocate decode scratch buffers
   - Изменить pool allocator для fixed decode shapes
   - Оценка: 1 неделя

3. **MMVQ Tile-Based Approach:** Перевести Q3_K decode на tile-based matmul вместо vec_dot
   - Использовать rocWMMA-style tile для small N dimensions
   - Оценка: 3-4 недели

### Medium Impact (possible)

4. **Pipeline Caching:** Cached kernel configs для common decode shapes
   - Pre-compute grid/block dims для typical decode shapes
   - Оценка: 1-2 недели

5. **Descriptor Optimization:** Reduce argument setup overhead
   - Batch kernel configs
   - Оценка: 1 неделя

### Low Impact / Not Feasible

6. **Cooperative Matrix for MMVQ:** Vulkan-specific API, ROCm использует другой подход
   - rocWMMA есть только для large matmul, не для small decode tiles
   - Не feasible без изменений в rocWMMA/ROCm SDK

7. **Shader Compilation Model:** Vulkan pre-compiles всё, ROCm compile-time templates
   - Архитектурное различие, не легко изменить

---

## 8. Реалистичная Оценка

### Текущий Decode Gap
- ROCm: 28-30 TPS
- Vulkan: 38-40 TPS
- Gap: ~10-12 TPS

### Потенциальные Улучшения

| Оптимизация | Ожидаемый Gain | Сложность |
|------------|---------------|-----------|
| Kernel Fusion (rms_norm+mul+rope) | +3-5 TPS | Высокая |
| Scratch Pre-allocation | +1-2 TPS | Средняя |
| MMVQ Tile-Based | +2-4 TPS | Очень высокая |
| Pipeline Caching | +0.5-1 TPS | Низкая |
| Descriptor Optimization | +0.5 TPS | Низкая |

**Максимальный realistic gain:** ~7-13 TPS
**Новый ROCm decode ceiling:** ~35-43 TPS

### Но Важный Trade-off

Vulkan **проигрывает prefill** (-40% prompt eval TPS). Если мы переносим Vulkan-оптимизации в ROCm:
- Decode может улучшиться
- Но prefill может ухудшиться (Vulkan prefill медленнее из-за других ограничений)

Для prompt-heavy workload (наш основной сценарий), wall TPS определяется prefill, не decode. Даже если ROCm decode станет 40 TPS, wall TPS может не улучшиться если prefill остаётся bottleneck.

---

## 9. Рекомендации

### Для Decode-Focused Workloads

Если цель — чистый decode throughput (chat с короткими prompt, long generation):
1. Рассмотреть Vulkan как primary backend
2. Использовать `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` для AMD driver
3. Использовать `--no-mmap` для better prefill

### Для Prompt-Heavy Workloads (наш основной case)

ROCm остаётся лучшим выбором, но можно:
1. Добавить selective kernel fusion только для ops где это помогает decode
2. Улучшить scratch allocation
3. Не трогать prefill path

### Для Balanced Approach

Гибридный подход: использовать ROCm для prefill (где он быстрее) и Vulkan-style optimizations для decode. Но это требует dual-backend scheduling — сложная задача.

---

## 10. Заключение

**Можно ли догнать Vulkan decode в ROCm?**

**Да, частично.** Реалистичная цель: довести ROCm decode до ~35-38 TPS (vs Vulkan 38-40).

**Ключевые шаги:**
1. Kernel fusion для decode-critical ops (rms_norm+mul+rope)
2. Scratch buffer optimization
3. MMVQ tile-based approach

**Что не переносится:**
- Cooperative matrix API (Vulkan-specific)
- Shader compilation model (architectural difference)
- Full pipeline caching system (~5000 строк кода)

**Финальная оценка:** Полностью догнать Vulkan decode (40+ TPS) на ROCm без фундаментальных изменений в ggml-cuda architecture нереалистично. Но gain в 5-8 TPS достижим и стоит усилий для decode-heavy workload.

---

## 11. Upstream Evidence: Kernel Fusion — Доказанный Cross-Backend Паттерн

### 11.1. Fusion Существует Практически Во Всех Backends

Upstream поиск по `ggml-org/llama.cpp` подтверждает: kernel fusion — это не случайность Vulkan, а общепринятый architectural pattern.

**Metal:**
```cpp
// ggml-metal-ops.cpp
case GGML_OP_MUL_MAT_ID:
{
    n_fuse = ggml_metal_op_mul_mat_id(ctx, idx);
} break;
```
Metal имеет `n_fuse` system где `n_fuse = 1, 2, 3...` определяет глубину fusion:
- `n_fuse = 1`: norm only
- `n_fuse = 2`: norm + mul
- `n_fuse = 3`: norm + mul + add

Dedicated fused pipelines:
- `kernel_norm_f32`
- `kernel_norm_mul_f32`
- `kernel_norm_mul_add_f32`
- `kernel_norm_mul_add_glu_f32`

**WebGPU:**
```cpp
// ggml-webgpu.cpp
static std::optional<webgpu_encoded_op> ggml_webgpu_rms_norm(
    webgpu_context & ctx,
    ggml_tensor *    rn_src,
    ggml_tensor *    rn_dst,
    ggml_tensor *    mul_src0,
    ggml_tensor *    mul_src1,
    ggml_tensor *    dst) {
    // Fused rms_norm + mul pipeline
    // Handles broadcast, inplace, overlap cases
}
```
WebGPU имеет explicit `rms_norm_mul` fusion с handling различных memory layout scenarios (inplace, overlap, src_overlap).

**CANN (Huawei Ascend):**
```cpp
// ggml-cann.cpp
// ACLNN ops: aclnn_mul, aclnn_add
// Fusion для rope:
void ggml_cann_rope(...) {
    aclnn_mul(ctx, acl_src.get(), acl_cos_reshape_tensor.get());
    aclnn_mul(ctx, acl_input_roll_mul_scale_tensor.get(), acl_sin_reshape_tensor.get());
    aclnn_add(ctx, acl_src.get(), acl_input_roll_mul_scale_tensor.get(), acl_dst.get());
}
```
CANN fuse-ит rope formula operations в один ACLNN graph.

**CPU (Spacemit RISC-V):**
```cpp
// spacemit/ime.cpp
case GGML_OP_RMS_NORM:
    spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
case GGML_OP_NORM:
    spacemit_kernels::rvv::forward_norm_f32(params, op);
```
Даже CPU backend имеет optimized norm implementations.

### 11.2. Vulkan Fusion Implementation

```cpp
// ggml-vulkan.cpp
ggml_vk_rms_norm() {
    if (ctx->num_additional_fused_ops > 0) {
        // Fused rms_norm + mul
        ggml_tensor *mul = cgraph->nodes[node_idx + 1];
        // ...
    }
    if (ctx->num_additional_fused_ops > 1) {
        // Full fusion: rms_norm + mul + rope
    }
}
```

Dedicated fused pipelines (из `vulkan-shaders-gen.cpp`):
- `rms_norm_mul_rope_f32_f32` — f32 compute, f32 rope params
- `rms_norm_mul_rope_f32_f16` — f32 compute, f16 rope params
- `rms_norm_mul_f32`
- `rms_norm_mul_partials_f32`
- `pipeline_multi_add[MAX_FUSED_ADDS]`
- `pipeline_multi_add_rms[MAX_FUSED_ADDS]`

Shader generation использует defines для variants:
```cpp
string_to_spv("rms_norm_mul_rope_f32_f32", "rms_norm.comp", 
    merge_maps(base_dict, {
        {"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"},
        {"ROPE_D_TYPE", "float"}, {"RMS_NORM_ROPE_FUSION", "1"}
    }));
```

### 11.3. ROCm/CUDA: Fusion Отсутствует в Decode Path

```cpp
// ggml-cuda.cu, ggml_cuda_mul_mat()
if (!split && use_mul_mat_vec_q) {
    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
} else if (!split && use_mul_mat_q) {
    ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
}
```
**Нет fusion.** Каждый op — отдельный kernel launch:
- `rms_norm` → отдельный kernel (`ggml_cuda_norm.cu`)
- `mul` → отдельный kernel (`ggml_cuda_mul_mat_vec_q`)
- `rope` → отдельный kernel (`ggml_cuda_rope.cu`)
- `add` → отдельный kernel

Для Qwen3.6-27B (65 layers), каждый decode step имеет ~6-8 ops per layer.
**Total kernel launches per decode step: ~400-500.**

Vulkan fuse-ит `rms_norm + mul + rope` в 1 kernel, сокращая launches на ~20-25%.

### 11.4. Что Это Значит Для Porting

**Fusion — proven pattern.** Это не "Vulkan magic", а общепринятый подход:
- Metal его использует ✅
- WebGPU его использует ✅
- CANN его использует ✅
- Vulkan его использует ✅
- **ROCm НЕ использует для decode ❌**

**Вывод:** Porting fusion в ROCm — это не "изобретать велосипед", а "доводить до parity с другими backends". Это значительно укрепляет argument что:
1. Fusion **feasible** для ROCm (уже сделано в 3 других backends)
2. Fusion **worth it** (все serious backends его делают)
3. Fusion **проверено** (Metal/WebGPU/Vulkan показывают gains)

### 11.5. Upstream Issues о Decode Performance

Search по `ggml-org/llama.cpp` issues для "vulkan decode performance" вернул **0 direct matches**. Это значит:
- Никто не репортовал "vulkan decode faster than rocm" как issue
- Gap существует, но не документирован upstream
- **Это opportunity:** кто первый implement fusion для ROCm decode, получит measurable gain

Search для "cooperative matrix mmvq" также вернул 0 matches — coop mat не используется в MMVQ path ни в одном backend.

---

## 12. Обновлённая Реалистичная Оценка (с Upstream Context)

### Текущий Decode Gap
- ROCm: 28-30 TPS
- Vulkan: 38-40 TPS
- Gap: ~10-12 TPS

### Потенциальные Улучшения (Updated)

| Оптимизация | Ожидаемый Gain | Сложность | Upstream Proof |
|------------|---------------|-----------|----------------|
| Kernel Fusion (rms_norm+mul+rope) | +3-5 TPS | Высокая | ✅ Metal/WebGPU/Vulkan все имеют fusion |
| Scratch Pre-allocation | +1-2 TPS | Средняя | ✅ Vulkan pre-allocates, ROCm pool growth overhead |
| MMVQ Tile-Based | +2-4 TPS | Очень высокая | ❌ No upstream proof для small-tile decode |
| Pipeline Caching | +0.5-1 TPS | Низкая | ✅ Vulkan proves zero-overhead dispatch works |
| Descriptor Optimization | +0.5 TPS | Низкая | ⚠️ Minor factor, не главный |

**Максимальный realistic gain (high-confidence):** +4-6 TPS (fusion + scratch)
**Максимальный optimistic gain:** +7-13 TPS (включая MMVQ tile rewrite)
**Новый ROCm decode ceiling:** ~34-36 TPS (high-confidence) до ~41 TPS (optimistic)

### Ключевой Insight

**Fusion — highest ROI оптимизация.** Это:
- Proven pattern (4 backends уже сделали)
- Measurable gain (Vulkan +30-40% decode TPS)
- Feasible для ROCm (HIP kernel model supports fusion)
- **Не ломает prefill** (fusion только для decode shapes)

---

## 13. Conclusion (Updated)

**Можно ли догнать Vulkan decode в ROCm?**

**Да, частично, и upstream evidence это подтверждает.**

**Железное теоретическое подтверждение:**

1. **Kernel fusion — proven cross-backend pattern.** Metal, WebGPU, CANN, Vulkan все implement fusion. ROCm — outlier. Porting fusion не "эксперимент", а "доводим до parity".

2. **Fusion даёт measurable gain.** Vulkan decode +30-40% vs ROCm, и fusion — главный фактор (20-25% fewer kernel launches, less global memory traffic).

3. **Fusion feasible для ROCm.** HIP kernel model идентичен CUDA, и CUDA backend уже имеет FFN fusion (`ggml_cuda_should_fuse_mul_mat`). Добавить rms_norm+mul+rope fusion — extension существующего pattern, не новый concept.

4. **Scratch pre-allocation feasible.** Vulkan proves fixed pre-allocated buffers work better than pool allocator для predictable decode shapes.

5. **Cooperative matrix НЕ портируемо.** Vulkan-specific API, ROCm использует rocWMMA только для FA. Это architectural limit.

**Реалистичная цель:**
- High-confidence: ROCm decode 34-36 TPS (fusion + scratch)
- Optimistic: 37-40 TPS (включая MMVQ tile rewrite)
- Full parity с Vulkan (40+ TPS): требует fundamental rewrite decode path

**Рекомендуемый план:**
1. **Phase 1 (2-3 недели):** Implement rms_norm+mul+rope fusion для ROCm decode
   - Создать `rms_norm_mul_rope` kernel в `ggml-cuda/rope.cu` или новый `ggml-cuda/fusion.cu`
   - Добавить detection в `ggml_cuda_multiplex_backend` graph scheduling
   - Использовать WebGPU/Metal как reference implementation
   
2. **Phase 2 (1-2 недели):** Scratch buffer pre-allocation для decode
   - Pre-allocate fixed buffers для common decode shapes
   - Изменить pool allocator для decode-specific path

3. **Validate:** A/B benchmark Vulkan vs ROCm на том же lane
   - Цель: сократить gap с 10-12 TPS до 3-5 TPS

**Что не стоит делать:**
- Попытка port cooperative matrix (Vulkan-specific, не feasible)
- Full shader compilation model rewrite (architectural difference, high effort/low gain)
- Touching prefill path (ROCm уже быстрее Vulkan в prefill, не ломать)
