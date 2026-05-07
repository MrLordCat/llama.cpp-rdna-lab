# Benchmarks

Главный локальный benchmark для этой ветки:

```powershell
python scripts\agent_workload_bench.py
```

Он запускает короткую симуляцию агентной работы через OpenAI-compatible `llama-server`: triage diff, code review, ROCm log diagnosis и маленький patch simulation. По умолчанию инструмент ищет ROCm server binary в:

```text
build-rocm\bin\llama-server.exe
build-rocm\bin\Release\llama-server.exe
```

и модель в:

```text
models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
models\Qwen3.6-27B-Q3_K_S.gguf
models\Qwen3.5-9B-Q6_K.gguf
```

Результаты пишутся в:

```text
build_logs\agent-workload\<label>.csv
build_logs\agent-workload\<label>.jsonl
build_logs\agent-workload\<label>.server.log
```

По умолчанию runner выбирает свободный порт сам. Для уже запущенного сервера укажи `--no-start --port 8080`.

Для коротких агентных ответов runner по умолчанию добавляет `--chat-template-kwargs {"enable_thinking":false,"preserve_thinking":false}`. Это можно отключить флагом `--no-disable-thinking`.

## Baseline ROCm

```powershell
python scripts\agent_workload_bench.py `
  --label rocm-baseline `
  --server-bin build-rocm\bin\llama-server.exe `
  --model models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf `
  --ctx-size 32768 `
  --batch-size 2048 `
  --ubatch-size 2048 `
  --cache-type-k q8_0 `
  --cache-type-v q8_0 `
  --max-tokens 160
```

## Existing Server

Если GUI уже запустил сервер:

```powershell
python scripts\agent_workload_bench.py --no-start --port 8080 --label gui-server-baseline
```

## MTP Branch Test

Только после того, как `llama-server --help` показывает `mtp`:

```powershell
python scripts\agent_workload_bench.py `
  --label rocm-mtp-draft3 `
  --server-bin build-rocm-mtp\bin\llama-server.exe `
  --model models\Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf `
  --server-extra "--spec-type mtp --spec-draft-n-max 3" `
  --ctx-size 32768 `
  --cache-type-k q8_0 `
  --cache-type-v q8_0
```

MTP benchmark должен быть text-only: не добавлять `--mmproj`.

## ngram-mod Coding-Agent Test

Для текущего master без MTP:

```powershell
python scripts\agent_workload_bench.py `
  --label rocm-ngram-mod `
  --server-extra "--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-draft-n-min 48 --spec-draft-n-max 64"
```

## Что сравнивать

Смотреть в CSV:

- `wall_s` по каждой задаче;
- `completion_tokens`;
- `completion_tps_wall`;
- ошибки запуска/ответа.

Смотреть в server log:

- prompt processing tok/s;
- generation tok/s;
- speculative draft acceptance rate;
- ROCm/HIP warnings;
- VRAM/memory allocation failures.

Для нашего workflow важен не только TG. Если MTP ускоряет generation, но сильно режет prompt processing, агентная задача может стать медленнее.
