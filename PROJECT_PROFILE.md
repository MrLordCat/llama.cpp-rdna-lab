# Project Profile

Дата профиля: 2026-05-07.

## Назначение форка

`llama.cpp-with-GUI` — персональный форк `ggml-org/llama.cpp` для локального inference на машине владельца. Главный сценарий: быстро запускать и тестировать Qwen/GGUF модели через GUI, собирать llama.cpp под AMD ROCm/Vulkan, пробовать TurboQuant и аккуратно подтягивать upstream runtime fixes без импорта чужой документации и CI-шума.

## Текущий performance target

- Активный target: `Qwen3.6-27B-Q3_K_S` на ROCm в prompt-heavy no-reuse workload со стартовой точкой ниже `16k`.
- Текущий reference стартовой точки: `ctx=12288` с входящим prompt около `~8k` токенов и throughput `~9.24 TPS`.
- Цель на текущую фазу: `25-27 TPS` на стартовой точке (`ctx=12288` или ближайший контекст, где модель остаётся в prompt-heavy режиме).
- `128k` и старый `64k` lane остаются только архивными reference-профилями и не являются активной целью.

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

Для Qwen3.6 text active research profile (rocWMMA enabled, 2026-05-10):

```text
backend=ROCm (HIP SDK 7.1 + rocWMMA 2.0.0 + RDNA4 MMA configs)
-ngl 999
--flash-attn on
-np 1
-c 12288 (current стартовая точка prompt-heavy lane)
-b 4096
-ub 512 (current working 64k corridor for Qwen3.6-27B speed research)
--cache-type-k q4_0 (slightly better than q8_0)
--cache-type-v q4_0
--spec-type none (current baseline for prompt-heavy no-reuse lane)

Optional experimental:
--spec-type mtp --spec-draft-n-max 3 (MTP support already in codebase, awaiting MTP-enabled GGUF)
```

Если не хватает VRAM:

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
