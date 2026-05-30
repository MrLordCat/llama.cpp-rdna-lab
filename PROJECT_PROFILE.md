# Project Profile

Дата профиля: 2026-05-26.

## Назначение форка

`llama.cpp-with-GUI` — персональный форк `ggml-org/llama.cpp` для локального inference на машине владельца. Главный сценарий: быстро запускать и тестировать Qwen/GGUF модели через GUI, собирать llama.cpp под AMD ROCm/Vulkan, пробовать TurboQuant и аккуратно подтягивать upstream runtime fixes без импорта чужой документации и CI-шума.

## Текущий performance target

- Активный target: dense `Qwen3.6-27B-Q3_K_S` на длинном контексте `ctx=131072` (~130k), cold-first, repo-snapshot real context, no-reuse/no-prime, thinking enabled.
- Первое обязательное действие новой фазы: получить свежие baseline для Vulkan и ROCm на одинаковом 130k lane, затем сравнивать любые ускорения только с ними.
- Ключевое ограничение: RX 9070 XT имеет 16 GB VRAM, поэтому при `ctx=131072` значимая часть KV/context/working set может уходить в system RAM. RAM-spill/residency/PCIe поведение теперь является частью целевой задачи.
- Старые `ctx=12288`, `32768`, `65536` и sentinel `128k` результаты остаются историческими reference; особенно старые sentinel128 с tiny prompt не считать 130k baseline.

## Текущий research workflow (главный)

- Основной research-контур ведется через `docs/research/major-topology/`.
- Перед новыми кодовыми прототипами обязательно сначала оформить или обновить major-topology заметку (`P`/`D`/`S`) и пройти gate-пакет.
- `docs/research/experiments/` использовать как вторичный/исторический слой и для узких измерительных записей, а не как главный вход в новую фазу.
- Любые speed claims по active lane принимаются только после измерений и записи результата в major-topology note + `docs/research/RESULTS_LOG.md`.

## Машина

| Компонент | Значение |
| --- | --- |
| OS | Windows 11 Pro build 26200, 64-bit |
| Motherboard | Gigabyte B550 GAMING X V2 |
| CPU | AMD Ryzen 7 5800X3D, 8 cores / 16 threads |
| RAM | 64 GB |
| GPU | AMD Radeon RX 9070 XT |
| ROCm target | `gfx1201` |
| GPU driver | 32.0.23033.1002 |
| HIP SDK | `C:\Program Files\AMD\ROCm\7.1` |
| Python | 3.13.4 |
| CMake | 3.29.2 |
| Ninja | 1.13.0 |
| Git | 2.49.0.windows.1 |

## Локальные модели

Текущая папка `models/` содержит локальные большие GGUF, которые не должны попадать в git:

| Модель | Размер | Назначение |
| --- | ---: | --- |
| `bge-m3-Q8_0.gguf` | ~605 MB | embeddings |
| `Qwen3.5-9B-Q6_K.gguf` | ~6.9 GB | быстрый Qwen text/VLM pair |
| `Qwen3.6-27B-Q3_K_S.gguf` | ~11.5 GB | основной dense Qwen3.6 для 16 GB VRAM |
| `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` | ~12.3 GB | MoE Qwen3.6 A3B для RX 9070 XT |
| `mmproj-F16.gguf` | ~876 MB | generic VLM projector |
| `Qwen3.5-9B.mmproj-F16.gguf` | ~876 MB | Qwen3.5 VLM projector |
| `Qwen3.6-35B-A3B.mmproj-F16.gguf` | ~858 MB | Qwen3.6 VLM projector |

## Git remotes

| Remote | URL | Роль |
| --- | --- | --- |
| `origin` | `https://github.com/MrLordCat/llama.cpp-with-GUI.git` | основной fork |
| `upstream` | `https://github.com/ggml-org/llama.cpp.git` | источник llama.cpp |
| `animehacker` | `https://github.com/animehacker/llama-turboquant.git` | TurboQuant reference |
| `turboquant` | `https://github.com/elusznik/llama.cpp.git` | альтернативный TurboQuant remote |

## Локальная ценность форка

- GUI запускает `llama-server` и `llama-cli`, управляет build-директориями, моделями и системной диагностикой.
- ROCm path адаптирован под Windows + HIP SDK 7.1 + Ninja.
- Добавлены GUI-настройки KV cache, TurboQuant типов, multimodal/mmproj и export build log.
- В `gui/model_presets.json` уже есть персональные Qwen3/Qwen3.5/Qwen3.6 presets.
- Синхронизация upstream должна сохранять локальные docs/actions/instructions.

## Что считается core upstream

Можно подтягивать регулярно:

- `common/`
- `src/`
- `include/`
- `ggml/`
- `tools/`
- `examples/`
- `cmake/`
- `CMakeLists.txt`
- conversion scripts, когда нужны новые модели

## Что считается локальным слоем

Не перетирать из upstream автоматически:

- `gui/`
- `README.md`
- `PROJECT_PROFILE.md`
- `AGENTS.md`
- `CLAUDE.md`
- `MTP.md`
- `QWEN_SPEED_RESEARCH.md`
- `ROCM_ACCELERATION_PLAN.md`
- `UPSTREAM_SYNC.md`
- `.github/**`
- `docs/**`

## Практичные defaults для RX 9070 XT

Для Qwen3.6 dense 27B active 130k research profile (2026-05-26):

```text
backend=Vulkan and ROCm, measured separately
-ngl 999
--flash-attn on
-np 1
-c 131072
-b 512
-ub 256    # Vulkan current best; use 128 for ROCm until rechecked
--cache-type-k q4_0
--cache-type-v q4_0
--spec-type none
--no-reuse
--no-v2-prime-pass
--real-context-mode repo-snapshot
--real-context-chars 24576
--max-tokens 16
--tasks quick --task-ids triage_diff
request timeout >= 180s, startup timeout >= 900s, task hard timeout >= 120s

Optional experimental:
--spec-type mtp --spec-draft-n-max 3 (MTP support already in codebase, awaiting MTP-enabled GGUF)
```

На 130k нельзя интерпретировать slowdown как простой kernel regression без проверки residency: сохранять diagnostics/server log, startup messages, mmap/no-mmap settings, RAM pressure and prompt/decode split. Текущий короткий baseline: Vulkan `1.7898 TPS` r3 на `b512/ub256` после D005 split-K с `--no-mmap`; ROCm `1.5200 TPS` r3 на `b512/ub128`, оба с `real-context-chars=24576`.

Для исторического 32k prompt-heavy ROCm/Qwen3.6-27B lane после native ubatch cliff fix (2026-05-12):

```text
backend=ROCm (RX 9070 XT / gfx1201)
-c 32768
-b 5120
-ub 1024
--cache-type-k q4_0
--cache-type-v q4_0
--spec-type ngram-mod
no reuse / no v2 prime for cold-first claims
```

Важно: `-ub 1024` теперь должен идти нативно (`PP reserve outputs 1024 -> 1`), без cap до `900`. Если cliff возвращается, сначала проверять ROCm compute vbuffer chunking и negative control `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1`.

Если нужна короткая sanity-проверка вместо active benchmark:

```text
-c 16384
--cache-type-k q4_0
--cache-type-v q4_0
```

Для MTP branch:

```text
--spec-type mtp
--spec-draft-n-max 3
-np 1
без --mmproj
без ctx_shift/cache_reuse
```

Для coding-agent workloads с большим количеством повторов:

```text
--spec-type ngram-mod
--spec-ngram-mod-n-match 24
--spec-draft-n-min 48
--spec-draft-n-max 64
```

Актуальный roadmap аппаратно-ориентированных ускорений и следующих optimizations вынесен в `ROCM_ACCELERATION_PLAN.md`.
