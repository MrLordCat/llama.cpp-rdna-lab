# Project Profile

Дата профиля: 2026-08-04.

## Назначение форка

`llama.cpp-rdna-lab` — специализированный форк `ggml-org/llama.cpp` для
локального inference, RDNA performance research и приложения **RDNA LLM
Studio**. Главный сценарий: быстро запускать и тестировать Qwen/GGUF модели,
собирать llama.cpp под AMD ROCm/Vulkan и аккуратно подтягивать upstream runtime
fixes без импорта нерелевантных backend, converter и CI-компонентов.

## Текущий performance target

- Primary model: dense `Qwen3.8-27B-Q4_K_M.gguf` (17.1 GiB, rebased from Qwen3.6 2026-08-14). Базовая безопасная дорожка — dual-ROCm `ctx=49152,b=8192,ub=1024,q8_0/q8_0`, cold/no-reuse/no-warmup.
- Любой MTP результат сравнивать с соседним `spec=none` запуском той же Q4 модели. Production agent profile использует MTP n3, когда длина ответа окупает небольшой prefill tax.
- `ctx=98304` — проверенный extended Q4 lane с one-copy ROCm scheduler. `ctx=131072` остаётся residency stress и требует отдельного placement/backend контроля.
- Q3_K_S остаётся secondary моделью для максимального VRAM/context headroom, vision и исторической Q3 kernel research программы. Q3 и Q4 TPS не объединять в один baseline.

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
| GPU driver | 32.0.31035.1003 |
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
| `Qwen3.6-27B-Q4_K_M.gguf` | ~15.9 GiB | previous primary dense Qwen3.6 (до 2026-08-14), MTP-enabled |
| `Qwen3.8-27B-Q4_K_M.gguf` | ~17.1 GiB | **primary** dense Qwen3.8 (2026-08), same qwen35 family, MTP-enabled; rebaseline 2026-08-14 |
| `Qwen3.6-27B-Q3_K_S.gguf` | ~11.5 GB | secondary headroom/vision/Q3 research model |
| `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` | ~12.3 GB | MoE Qwen3.6 A3B для RX 9070 XT |
| `mmproj-F16.gguf` | ~876 MB | generic VLM projector |
| `Qwen3.5-9B.mmproj-F16.gguf` | ~876 MB | Qwen3.5 VLM projector |
| `Qwen3.6-35B-A3B.mmproj-F16.gguf` | ~858 MB | Qwen3.6 VLM projector |

## Git remotes

| Remote | URL | Роль |
| --- | --- | --- |
| `origin` | `https://github.com/MrLordCat/llama.cpp-rdna-lab.git` | основной fork |
| `upstream` | `https://github.com/ggml-org/llama.cpp.git` | источник llama.cpp |

## Локальная ценность форка

- GUI запускает `llama-server` и `llama-cli`, управляет build-директориями, моделями и системной диагностикой.
- ROCm path адаптирован под Windows + HIP SDK 7.1 + Ninja.
- Добавлены GUI-настройки KV cache, multimodal/mmproj и export build log.
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
- converter scripts не импортируются; для редкой конвертации используется отдельный upstream checkout

## Что считается локальным слоем

Не перетирать из upstream автоматически:

- `gui/`
- `README.md`
- `PROJECT_PROFILE.md`
- `AGENTS.md`
- `MTP.md`
- `QWEN_SPEED_RESEARCH.md`
- `UPSTREAM_SYNC.md`
- `.github/**`
- `docs/**`

## Практичные defaults для RX 9070 XT

Для primary Qwen3.6-27B Q4_K_M profile (2026-07-20):

```text
backend=ROCm
model=Qwen3.6-27B-Q4_K_M.gguf
-dev ROCm1,ROCm0 -sm layer -ts 1,1
-ngl 999
--flash-attn on
-np 1
-c 49152
-b 8192
-ub 1024
--cache-type-k q8_0
--cache-type-v q8_0
--no-warmup
--cache-ram 0
--ctx-checkpoints 0
-fit off

Choose exactly one speculation mode:
--spec-type none                              # adjacent performance control
--spec-type draft-mtp --spec-draft-n-max 3  # production agent profile
```

Текущий Q4 baseline на prompt `29561`, output `128`: spec-none
`1778.59/21.98` prompt/decode tok/s; MTP n3 `1731.71/39.58`, aggregate
`6.2802 TPS`, acceptance `74.36%`. На 98K one-copy scheduler проверен до
`1493.21` prompt tok/s. Для 131K сохранять residency diagnostics; старый
memory-aware ROCm split `27,37` и Vulkan — отдельные stress-профили, а не
автоматический safe default. На текущем драйвере Vulkan Q4_K_M использует
восстановленный default `wn32`; детали и rollback gate зафиксированы в D093.

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

Актуальная очередь аппаратно-ориентированных ускорений ведётся в `docs/research/major-topology/README.md`.
