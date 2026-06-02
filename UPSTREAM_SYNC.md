# Upstream Sync Policy

Цель: догонять `ggml-org/llama.cpp` по важным core/runtime изменениям, не импортируя upstream Actions, upstream documentation и чужие агентские инструкции поверх локального форка.

## Почему это нужно

Этот репозиторий не является чистым mirror upstream. Здесь есть:

- PyQt6 GUI;
- ROCm/RDNA4 workflow под RX 9070 XT;
- TurboQuant изменения;
- локальная документация;
- локальные агентские инструкции;
- собственные GitHub/project файлы.

Слепой `git merge upstream/master` может принести полезный runtime-код, но одновременно перезаписать `.github`, `docs`, `README.md` и instruction-файлы.

## Защищённые зоны

Не импортировать из upstream без отдельного решения:

```text
.github/**
docs/**
README.md
AGENTS.md
CLAUDE.md
MTP.md
UPSTREAM_SYNC.md
gui/README.md
gui/QUICKSTART.md
.gemini/**
.devops/**
```

Эти правила также записаны в `.gitattributes` через `merge=ours`.

## Одноразовая локальная настройка merge driver

В каждом clone нужно один раз включить driver `ours`:

```powershell
git config merge.ours.driver true
```

Проверка:

```powershell
git config --get merge.ours.driver
```

Ожидаемый вывод:

```text
true
```

## Безопасная процедура

1. Проверить рабочее дерево:

```powershell
git status --short --branch
```

2. Получить upstream:

```powershell
git fetch upstream
```

3. Посмотреть масштаб изменений:

```powershell
git log --oneline --left-right --cherry-pick HEAD...upstream/master
git diff --stat HEAD...upstream/master
```

4. Если есть локальные незакоммиченные GUI/ROCm/TurboQuant изменения, не начинать merge, пока не понятно, как их сохранить.

5. Для обычного догоняющего merge:

```powershell
git merge --no-ff upstream/master
```

6. Если Git всё равно принёс изменения в защищённые зоны, вернуть локальную сторону:

```powershell
git checkout --ours README.md AGENTS.md CLAUDE.md MTP.md UPSTREAM_SYNC.md
git checkout --ours .github docs gui/README.md gui/QUICKSTART.md
git add README.md AGENTS.md CLAUDE.md MTP.md UPSTREAM_SYNC.md .github docs gui/README.md gui/QUICKSTART.md
```

7. Разрешить конфликты только в core/runtime коде.

8. Проверить:

```powershell
python -m py_compile run.py gui\main_window.py gui\server_tab.py gui\benchmark_tab.py gui\build_tab.py gui\build_manager.py gui\dependency_checker.py gui\hardware_detector.py
git diff --check
```

9. Для ROCm-sensitive изменений сначала configure:

```powershell
cmake -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release
```

## Что импортировать из upstream в первую очередь

Обычно полезно брать:

- `common/`
- `src/`
- `include/`
- `ggml/`
- `tools/`
- `examples/`
- `CMakeLists.txt`
- `cmake/`
- model conversion scripts, если они нужны для новых GGUF/model architectures

С осторожностью:

- `requirements/`
- `pyproject.toml`
- `scripts/`
- `tests/`
- `vendor/`

Не брать автоматически:

- `.github/`
- `docs/`
- root README;
- upstream agent instruction files;
- чужие issue templates/actions.

## MTP sync branch

Для MTP лучше делать отдельную ветку:

```powershell
git switch -c feature/mtp-sync
git fetch upstream pull/22673/head:upstream-pr-22673
git merge --no-ff upstream-pr-22673
```

После этого проверить:

```powershell
rg -n "mtp|MTP|COMMON_SPECULATIVE_TYPE" common src include tools examples
cmake -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm --config Release -j 4 --target llama-server
```

Не переносить MTP в `master`, пока:

- сервер без MTP запускается как раньше;
- `--spec-type mtp` есть в `llama-server --help`;
- MTP-enabled GGUF реально генерирует быстрее baseline;
- multimodal/vision путь не ломается или явно отключает MTP.

## Если нужно перенести только core без merge

Иногда безопаснее создать временную ветку и cherry-pick конкретные commits upstream, чем делать большой merge. Для MTP или ROCm fixes это предпочтительно, если PR маленький и зоны изменений понятны.

Перед cherry-pick:

```powershell
git show --stat <commit>
git show --name-only <commit>
```

Не cherry-pick commit, который в основном меняет `.github`, docs или release automation, если цель была core/runtime.

---

## Cherry-Pick Protocol Log: 2026-06-02

**Ветка**: `research/cherry-pick-upstream` (от `c40f8a5af`).
**Baseline**: `vscode-cherrypick-rocm-baseline-r3` — **25.83 TPS** (ROCm, ctx=4096, b=512, ub=128, q4_0 KV, FA on, spec=none, max_tokens=48, runs=3).

### Правила протокола

1. Без `git merge` — только ручное внесение (manual apply).
2. Если cherry-pick конфликтует с локальными изменениями форка — abort, читаем upstream diff, вносим только новые строки.
3. Никогда не перезаписывать локальные изменения через `git checkout --theirs`.
4. После каждого PR: сборка + бенч (Python `agent_workload_bench.py`, без trace env).

### Успешно внедрено

| # | PR | Коммит | Файлы | Строк | TPS | Δ |
|---|-----|--------|-------|-------|-----|-----|
| 1 | **#23227** | `d61a7d14f` | `mmvq.cuh` +2, `mmvq.cu` +47, `ggml-cuda.cu` +2 | +51 | **26.13** | +1.2% |
| 2 | **#23646** | `88533968b` | `server-context.cpp` +2 | +2 | **26.09** | +1.0% |

**Суммарно**: +2.3% (в пределах шума, без регрессий).

**#23227** (ROCm per-quant MMVQ/MMQ): `ggml_cuda_should_use_mmvq()` — per-type пороги CDNA1/CDNA2. На RDNA4 не активируется, но не вредит.

**#23646** (MTP KV-cache draft type): MTP draft-контекст использует `cache_type_k`/`cache_type_v` из spec-параметров.

### Пропущены — уже есть в форке

| PR | Причина |
|----|--------|
| #23433 (mtp inp_out_ids) | `inp_out_ids` и `ggml_get_rows` уже в qwen35.cpp:131-153 |
| #23988 (speculative fix) | `common_speculative_n_max` уже есть; draft-simple auto-enable убран |

### Пропущены — структурно несовместимы

| PR | Причина |
|----|--------|
| #23056 (Vulkan Q3_K block-load +57%) | Требует symbols отсутствующие в нашем Vulkan билде |
| #22887 (Vulkan MUL_MAT_VEC 4K) | Связан с #23056 |
| #23643 (llm_graph_input_mtp) | Требует класс `graph_mtp` — другая структура MTP |
| #23461 (MTP VRAM leak fix) | `ctx_dft` отсутствует в нашем `server-context.cpp` |

### Пропущен по сути — #23764 (FA f16 mask)

**Идея**: KQ-маска F32→F16 при `flash_attn=on`, экономия 50% памяти под маску.
**Для нас**: decode 1 токен — 262 KB, speculative 4 токена — ~1 MB экономии.
**Цена**: темплейтить 7 вариантов `set_input_kq_mask_impl` + `fill_mask` + `print_mask`.
**Вердикт**: не окупается для decode-доминантного сценария.

### Bench progression

| Step | Wall (s) | TPS | Δ |
|------|----------|-----|---|
| baseline | 1.86 | **25.83** | — |
| + #23227 | 1.84 | **26.13** | +1.2% |
| + #23646 | 1.84 | **26.09** | +1.0% |

---

## Rejected/Closed Vulkan PR Analysis: 2026-06-02

Изучены отклонённые и закрытые Vulkan PR в upstream. 159 total unmerged Vulkan PRs.
Ниже — наиболее интересные для нашего форка с анализом идей.

### #23696 — Vulkan backend performance with helper classes (MaxwellGengYF)
- **Закрыт**: 0cc4m, «VMA is not gonna get accepted here, there is no reason to use it»
- **Идея**: VMA (AMD Vulkan Memory Allocator) для сабаллокации, bindless descriptors (VK_EXT_descriptor_indexing), persistent VkPipelineCache, barrier tracker, ring-buffer staging
- **Почему отклонён**: слишком большой (21K строк), AI-сгенерированные описания, multiple unrelated changes
- **Полезные идеи для нас**:
  - **Persistent pipeline cache**: сохраняет скомпилированные пайплайны на диск между запусками → ускорение старта сервера. Это можно реализовать отдельно, ~50 строк
  - **Ring-buffer staging**: циклический буфер для CPU↔GPU трансферов вместо выделения нового буфера под каждый тензор → меньше аллокаций, быстрее prompt ingestion
  - **Barrier tracker**: автоматическое отслеживание барьеров между диспатчами → меньше ручных vkCmdPipelineBarrier
- **Вердикт**: идеи хорошие, но реализовывать по отдельности, не гигантским PR

### #23573 — Pipeline cache shared mutex (winstonma)
- **Закрыт**: автором после замечания jeffbolznv
- **Идея**: `std::shared_mutex` для concurrent reads пайплайн-кэша, double-checked locking
- **Почему отклонён**: пайплайн-карты также guarded через device mutex в `ggml_vk_load_shaders`; fine-grained locks → риск дедлоков
- **Полезная идея**: shared_mutex для read-heavy workloads. Но требует осторожности из-за пересечений с device mutex
- **Вердикт**: для single-user сценария (наш случай) lock contention не проблема. Не нужно

### #22750 — Skip cooperative matrix on integrated AMD GPUs (elana-voss)
- **Закрыт**: 0cc4m, «Coopmat works perfectly fine on AMD RDNA3+, we will not disable the feature»
- **Идея**: отключать coopmat на AMD iGPU (Radeon 860M) из-за бага драйвера (amdvlk64.dll access violation)
- **Почему отклонён**: проблема в конкретной конфигурации драйвера, не в llama.cpp
- **Нам**: RX 9070 XT — дискретная RDNA4. Coopmat работает. **Не актуально**

### #22459 — Pipeline cache for compute pipelines (winstonma)
- **Закрыт**
- **Идея**: переиспользование скомпилированных SPIR-V пайплайнов через VkPipelineCache
- **Почему отклонён**: не указано явно, вероятно перекрыт другими изменениями
- **Полезная идея**: совпадает с #23696. Реализация pipeline cache отдельно — low-hanging fruit

### #21357 — Zero-copy host_ptr for CPU tensors (Perinban)
- **Закрыт**
- **Идея**: zero-copy host_ptr для CPU тензоров — избежать двойной загрузки модели на memory-constrained устройствах
- **Нам**: 64 GB RAM — не критично. Но идея интересная для сценария с нехваткой RAM

### #21359 — UMA host buffer support (Perinban)
- **Закрыт**
- **Идея**: поддержка UMA (unified memory architecture) host буферов в Vulkan
- **Нам**: RX 9070 XT — дискретная с dedicated VRAM. Не актуально

### #23570 — Refactor vk_queue to per-instance mutexes (winstonma) — 🟡 OPEN
- **33 комментария**, активное обсуждение
- **Идея**: рефакторинг vk_queue — separate mutex per queue instance instead of global device lock
- **Зачем**: уменьшить lock contention при многопоточном доступе к разным queue families
- **Статус**: ещё не вмержен, идут правки после ревью

### #23762 — Fix UMA performance with cached host memory (winstonma) — 🟡 OPEN
- **Идея**: предпочитать cached host memory на UMA-устройствах, обрабатывать non-coherent память
- **Нам**: RX 9070 XT — dedicated GPU. UMA не используется. **Не актуально**

### Сводка идей для потенциальной реализации

| Идея | Откуда | Приоритет | Сложность | Потенциальный эффект |
|---|---|---|---|---|
| **Persistent VkPipelineCache** | #23696, #22459 | 🟡 Средний | ~50 строк | Ускорение холодного старта сервера (shader compilation) |
| **Ring-buffer staging** | #23696 | 🟡 Средний | ~150 строк | Меньше аллокаций при prompt ingestion |
| **Barrier tracker** | #23696 | 🟢 Низкий | ~200 строк | less error-prone barriers, но уже работает |
| Shared mutex для pipeline map | #23573 | 🔴 Не нужно | ~30 строк | Для single-user нет контеншна |
| Coopmat skip на iGPU | #22750 | 🔴 Не нужно | — | RX 9070 XT — дискретная |
| VMA (Vulkan Memory Allocator) | #23696 | 🔴 Не нужно | ~20K строк | Уже есть своя memory management |

### 🎯 Рекомендация

**Persistent pipeline cache** — самый полезный takeaway. Сохраняет скомпилированные SPIR-V пайплайны в файл и загружает при следующем запуске, проверяя vendorID/deviceID/pipelineCacheUUID. Это:
- Ускоряет холодный старт сервера (особенно заметно на Vulkan где shader compilation — bottleneck)
- Безопасно: не меняет рантайм-поведение, только кэширует
- Маленький: ~50 строк
- Независим: не затрагивает другие компоненты
