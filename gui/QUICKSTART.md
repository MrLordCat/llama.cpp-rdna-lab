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
