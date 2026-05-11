# MTP Recheck 2026-05-11

Цель: перепроверить локальный MTP-путь на текущей ROCm/RDNA4 сборке, отделить факты от старых предположений и зафиксировать, что именно нужно улучшать дальше.

## Короткий вывод

MTP в текущем локальном сочетании `Qwen3.6-27B-IQ3_M-mtp`, RX 9070 XT, ROCm 7.1 и llama.cpp runtime включается корректно, имеет высокий acceptance rate, но всё ещё медленнее non-MTP.

Главная причина не в плохом acceptance. На новом recheck `draft acceptance` был 87-97%, но MTP draft generation стоил слишком дорого: примерно 62-85 ms на draft token против примерно 40 ms на обычный non-MTP target token. Пока draft token дороже target token, MTP не может ускорять wall throughput даже при хорошем acceptance.

Лучший быстрый результат recheck:

- non-MTP: 19.48 completion TPS wall
- MTP `--spec-draft-n-max 2`: 11.72 completion TPS wall
- MTP остаётся примерно на 40% медленнее baseline на этом lane.

## Что перепроверили

### Runtime

Текущий `build-rocm-wmma/bin/llama-server.exe` поддерживает MTP:

- `--spec-type [none|mtp|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod]`
- `--spec-draft-n-max N`, default `16`
- draft KV flags есть: `--spec-draft-type-k`, `--spec-draft-type-v`

Важно: старый `MTP.md` в корне уже частично устарел, потому что описывает состояние, где локальный runtime ещё не поддерживал `mtp`. Новый актуальный статус: runtime поддерживает MTP, но performance-negative на проверенной модели.

### Модель

Локальная модель для теста:

- `models/Qwen3.6-27B-IQ3_M-mtp (1).gguf`
- server log определяет её как `Qwen3.6 27B`
- file type: `IQ3_S mix - 3.66 bpw`
- GGUF содержит `qwen35.nextn_predict_layers`
- есть NextN tensors, например `blk.64.nextn.eh_proj.weight`, `blk.64.nextn.enorm.weight`, `blk.64.nextn.hnorm.weight`, `blk.64.nextn.shared_head_norm.weight`
- MTP head загружается как partial load: used 18 of 866 tensors

MTP действительно активируется:

- `set_mtp: MTP draft head registered`
- отдельный MTP context создаётся и получает собственный KV/cache/scheduler

### External 40 t/s claim

Из открытой Reddit-страницы с заголовком `Managed to get 40 t/s on Qwen 27B (MTP) with an RX 6800 XT` извлечены только ограниченные технические детали:

- автор пишет про generation около 40 t/s на `Qwen 2.5 27B (IQ4_XS)` при `32k context`;
- ссылка на fork: `https://github.com/Stormrage34/llama.cpp-turboquant-hip`;
- в комментарии автор пишет: `With ncmoe 16 i get on 35b IQ4 MTP 55 tp/s`.

Точного reproduce command, prompt, metric definition и server log в посте нет. Поэтому это полезный источник гипотез, но не сопоставимый benchmark. Особенно важно: там говорится про generation t/s, а наши основные цифры ниже — `completion_tps_wall` на агентном workload.

## Mini-sweep 2026-05-11

Общая методика:

- server: `build-rocm-wmma/bin/llama-server.exe`
- model: `models/Qwen3.6-27B-IQ3_M-mtp (1).gguf`
- task: `v2-review`
- runs: `1`
- ctx: `12288`
- batch: `1024`
- KV: `q4_0/q4_0`
- flash-attn: `on`
- GPU layers: `99`
- thinking: enabled via `--no-disable-thinking`
- prompt cache reuse: no repeated task within run
- actual completion tokens: 500 for the v2 review task

### Results

| label | ubatch | spec | draft max | wall TPS | prompt tok/s | eval tok/s | draft accepted/generated | acceptance | MTP dur(g) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mtp-recheck-wmma-none-v2review-c12288` | 1024 | none | - | 19.48 | 86.08 | 25.05 | - | - | - |
| `mtp-recheck-wmma-mtp-n1-v2review-c12288` | 1024 | mtp | 1 | 10.49 | 73.14 | 12.21 | 246/253 | 97.23% | 21.441 s |
| `mtp-recheck-wmma-mtp-n2-v2review-c12288` | 1024 | mtp | 2 | 11.72 | 81.57 | 13.64 | 326/346 | 94.22% | 23.090 s |
| `mtp-recheck-wmma-mtp-n3-v2review-c12288` | 1024 | mtp | 3 | 11.15 | 80.87 | 12.90 | 361/414 | 87.20% | 25.627 s |
| `mtp-recheck-wmma-none-v2review-c12288-ub256` | 256 | none | - | 18.96 | 69.48 | 25.90 | - | - | - |
| `mtp-recheck-wmma-mtp-n2-v2review-c12288-ub256` | 256 | mtp | 2 | 10.71 | 51.22 | 13.48 | 325/348 | 93.39% | 23.311 s |

Artifacts:

- `build_logs/agent-workload/mtp-recheck-wmma-none-v2review-c12288.*`
- `build_logs/agent-workload/mtp-recheck-wmma-mtp-n1-v2review-c12288.*`
- `build_logs/agent-workload/mtp-recheck-wmma-mtp-n2-v2review-c12288.*`
- `build_logs/agent-workload/mtp-recheck-wmma-mtp-n3-v2review-c12288.*`
- `build_logs/agent-workload/mtp-recheck-wmma-none-v2review-c12288-ub256.*`
- `build_logs/agent-workload/mtp-recheck-wmma-mtp-n2-v2review-c12288-ub256.*`

## Что это значит

### 1. Acceptance хороший, но это не спасает

На лучшем MTP режиме `n=2` accepted/generated = `326/346`, то есть acceptance около 94%. Это обычно должно быть достаточно хорошо для ускорения. Но MTP всё равно падает до 11.72 TPS против 19.48 TPS.

Причина: стоимость генерации draft tokens слишком высокая.

Примерная цена draft generation:

- n=1: `21.441 s / 253 draft tokens` = ~84.7 ms per draft token
- n=2: `23.090 s / 346 draft tokens` = ~66.7 ms per draft token
- n=3: `25.627 s / 414 draft tokens` = ~61.9 ms per draft token

Для сравнения, non-MTP target decode на том же lane:

- `19957.40 ms / 500 tokens` = ~39.9 ms per token

То есть MTP draft token сейчас дороже обычного target token. В такой экономике MTP не может быть ускорением.

### 2. `--spec-draft-n-max 16` был плохим дефолтом, но не единственной причиной

Старые MTP прогоны с default depth могли быть чрезмерно дорогими. Recheck с `n=1/2/3` это проверил:

- да, `n=2` лучше `n=1` и `n=3`;
- нет, даже лучший малый depth не выходит в плюс.

Значит, проблема глубже, чем просто слишком длинная draft chain.

### 3. `ubatch=256` не лечит MTP

`ubatch=256` уменьшает compute buffer:

- target ROCm0 compute buffer: 990.00 MiB -> 247.50 MiB
- MTP ROCm0 compute buffer: 990.00 MiB -> 247.50 MiB

Но MTP n=2 ухудшается:

- ub=1024: 11.72 TPS
- ub=256: 10.71 TPS

Это значит, что bottleneck не только в reserved compute buffer или VRAM pressure.

### 4. Dual-scheduler помогает общему decode, но не исправляет MTP

`build-rocm-wmma` содержит dual TG/PP scheduler:

- target TG compute buffer: 6.72 MiB
- MTP TG compute buffer: 0.97 MiB

Это уже сильно лучше старого положения, где token-generation scheduler не был отделён. Но MTP всё равно медленный, значит основной overhead сидит в самой MTP draft generation path, а не только в размере TG buffer.

### 5. Самый подозрительный участок кода

Текущая MTP implementation делает дорогие операции между target context и MTP context:

- target prefill hook синхронизирует target context;
- копирует hidden states через `ggml_backend_tensor_get`;
- вызывает `llama_decode(ctx_mtp, hook_batch)` для MTP context;
- draft loop для каждого шага снова делает synchronize/tensor_get/llama_decode/sample.

Ключевые места:

- `src/llama-context.cpp`: `llama_context::handle_mtp_for_ubatch`
- `common/speculative.cpp`: `common_speculative_state_mtp::draft`

Пока hidden-state handoff идёт через host-visible copy/sync и отдельный context decode, MTP на ROCm легко становится дороже, чем обычный target decode.

## Что могло быть сделано не так раньше

1. Старое прямое сравнение `v2mini-buildrocmcompare-r1` vs `v2mini-mtp-main-r1` смешивало разные модели: non-MTP был на `Qwen3.6-27B-Q3_K_S.gguf`, MTP на `Qwen3.6-27B-IQ3_M-mtp.gguf`. Это годилось как practical comparison, но не как чистый A/B.
2. Более корректный smoke `post-change-mtp-smoke` уже сравнивал одну и ту же MTP GGUF с spec none vs mtp, и там MTP тоже проиграл: 12.45 TPS vs 7.69 TPS.
3. Старые MTP smoke-логи были сняты до проверки dual-scheduler build. Новый recheck на `build-rocm-wmma` подтвердил: dual-scheduler улучшает базовый decode, но MTP всё равно проигрывает.
4. Default `--spec-draft-n-max 16` слишком агрессивен для этой модели/реализации. Но даже `1/2/3` не дают ускорения.
5. External headline `40 t/s` нельзя напрямую сравнивать с нашим `completion_tps_wall`: там нет команды, нет prompt, нет wall/prompt split, и, судя по тексту, речь про generation TPS на другой модели/кванте.

## Что улучшать дальше

### Приоритет 1: инструментировать MTP draft generation

Нужна разбивка `dur(g)` внутри MTP draft loop:

- time in `llama_synchronize(ctx_tgt/ctx_mtp)`
- time in `ggml_backend_tensor_get`
- time in `llama_decode(ctx_mtp, batch)`
- time in sampling/logits readback

Цель: понять, draft token дорогой из-за sync/copy, MTP decode graph, output projection или sampling.

### Приоритет 2: убрать host roundtrip hidden states

Самая перспективная кодовая оптимизация: передавать hidden state target -> MTP head без CPU-visible `ggml_backend_tensor_get` на каждый draft step. Идеально — общий graph/device-side handoff или fused MTP head path внутри target decode.

### Приоритет 3: проверить, почему MTP head token дороже target token

Нужно отдельно измерить MTP context decode для одного шага. Подозрения:

- MTP context всё равно тянет дорогой output projection по большому vocab;
- MTP head в GGUF частично q8_0/f32 и не настолько лёгкий;
- отдельный context/scheduler ломает locality;
- ROCm sync overhead доминирует на коротких single-token graph;
- Gated Delta Net / Qwen3.6 hybrid path даёт неочевидную цену даже для MTP layer.

### Приоритет 4: проверить другой MTP GGUF/quant

Текущая модель `IQ3_S mix - 3.66 bpw` с MTP head не похожа на внешний claim `IQ4_XS`. Нужен отдельный тест:

- Qwen3.6 27B MTP IQ4/UD quant;
- если доступно, точно тот GGUF/quant, на котором community получает 40+ generation TPS;
- фиксировать не только TPS, но и `dur(g)`, prompt/eval split, acceptance.

### Приоритет 5: оставить GUI MTP guarded

MTP не должен становиться default или recommended. В GUI/autotune его можно показывать только когда:

- имя модели содержит `mtp`/`nextn` или metadata подтверждает NextN tensors;
- server help содержит `--spec-type mtp`;
- результат autotune реально лучше `spec=none`.

## Ближайший рабочий план

1. Не тратить время на большие `n_max` значения. Текущий best среди малых depth: `--spec-draft-n-max 2`, но он всё равно negative.
2. Instrumentation `common_speculative_state_mtp::draft` по фазам sync/get/decode/sample добавлен в `common/speculative.cpp`.
3. После успешной ROCm link-сборки прогнать только два режима:
   - `spec=none`
   - `spec=mtp --spec-draft-n-max 2`
4. Если `tensor_get/sync` занимает значимую долю `dur(g)`, проектировать device-side handoff.
5. Если почти всё время в `llama_decode(ctx_mtp)`, смотреть MTP graph/output projection и возможность более лёгкой MTP head execution path.

## Follow-up instrumentation

Добавлена строка статистики:

```text
statistics mtp detail: #calls(get,decode) = ... ..., dur(sync,get,decode,sample) = ... ms
```

Что она должна показать в следующем MTP run:

- долю времени на `llama_synchronize(ctx_tgt/ctx_mtp)`;
- долю времени на `ggml_backend_tensor_get` hidden state;
- долю времени на `llama_decode(ctx_mtp, batch)`;
- долю времени на sample/accept.

Validation status:

- `get_errors` for `common/speculative.cpp`: clean;
- `git diff --check`: clean;
- direct clang++ `-fsyntax-only` for `common/speculative.cpp` using the existing `compile_commands.json` command: clean;
- full `build-rocm-wmma`/main `llama-server` link is currently blocked before app link by unrelated HIP backend `amdgcn-link command failed due to signal` while compiling/linking `ggml-cuda/fattn.cu` or `ggml-cuda/mmq.cu`;
- existing `build-cpu` cannot regenerate because its CMake cache still references removed `C:/Program Files/CMake/share/cmake-4.0` modules.

## Commands used

Baseline:

```powershell
python scripts/agent_workload_bench.py `
  --label mtp-recheck-wmma-none-v2review-c12288 `
  --server-bin build-rocm-wmma/bin/llama-server.exe `
  --model "models/Qwen3.6-27B-IQ3_M-mtp (1).gguf" `
  --tasks v2-review --runs 1 `
  --ctx-size 12288 --batch-size 1024 --ubatch-size 1024 `
  --cache-type-k q4_0 --cache-type-v q4_0 `
  --gpu-layers 99 --parallel 1 `
  --startup-timeout 240 --request-timeout 240 `
  --background-server-policy fail --artifact-mode full `
  --no-disable-thinking --no-v2-prime-pass
```

Best MTP recheck:

```powershell
python scripts/agent_workload_bench.py `
  --label mtp-recheck-wmma-mtp-n2-v2review-c12288 `
  --server-bin build-rocm-wmma/bin/llama-server.exe `
  --model "models/Qwen3.6-27B-IQ3_M-mtp (1).gguf" `
  --tasks v2-review --runs 1 `
  --ctx-size 12288 --batch-size 1024 --ubatch-size 1024 `
  --cache-type-k q4_0 --cache-type-v q4_0 `
  --gpu-layers 99 --parallel 1 `
  --startup-timeout 240 --request-timeout 240 `
  --background-server-policy fail --artifact-mode full `
  --no-disable-thinking --no-v2-prime-pass `
  --server-extra "--spec-type mtp --spec-draft-n-max 2"
```