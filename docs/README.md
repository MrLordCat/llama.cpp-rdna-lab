# Local Docs Index

Эта папка в основном содержит upstream документацию `ggml-org/llama.cpp`. В этом форке она сохранена как технический справочник, но не является главным входом в проект.

Главные локальные документы находятся в корне:

- `README.md` — обзор форка.
- `PROJECT_PROFILE.md` — железо, окружение, модели и локальные правила.
- `AGENTS.md` — инструкции для AI-агентов.
- `UPSTREAM_SYNC.md` — как догонять upstream без импорта чужих docs/actions.
- `MTP.md` — заметки по Multi-Token Prediction.
- `QWEN_SPEED_RESEARCH.md` — план ускорений Qwen и внедрения MTP.
- `docs/research/README.md` — research hub для новых гипотез ускорения (после ngram/FlashAttention).

При merge из upstream папка `docs/**` защищена через `.gitattributes` (`merge=ours`). Новые сведения из upstream docs переносить сюда вручную только если они реально нужны этому форку.
