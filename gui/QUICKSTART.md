# Quick Start

1. Запустите `python run.py` из корня репозитория.
2. В Build & Setup выберите `Vulkan`, `ROCm/HIP` или `CPU`.
3. Выполните Configure, затем Build.
4. В Launch Server выберите модель и собранный build.
5. Для первого запуска используйте `Spec: None`; MTP включайте только для
   MTP-enabled GGUF.
6. Для vision выберите совместимый `mmproj-*.gguf`.

Для этой dual-RX 9070 XT системы основной Vulkan порядок устройств:
`Vulkan1,Vulkan0`. Актуальные batch, ubatch и split берите из autotune history.

Для Qwen3.6-27B Q4_K_M базовый ROCm KV — `q8_0`, MTP n3. На Vulkan preset
использует native P5 `f8_e4m3`, MTP n2 и FlashAttention. Имя FP8 не означает,
что raw E4M3 точнее block-scaled q8_0: на длинном MTP сервер автоматически
держит последние 8 KV-слоёв в f16, а при `ctx >= 98304` — последние 12
(D097). Явный `LLAMA_VK_MTP_KV_LAST_F16` переопределяет эту политику.
