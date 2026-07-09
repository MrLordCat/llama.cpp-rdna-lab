# HANDOFF: апстрим-порт spec-стека (для code-агента)

Кратко: что сделано, что в полёте, на что осторожно. Полный контекст:
`docs/research/UPSTREAM_SYNC_JULY2026.md` + `docs/research/SPEC_DECODING_STATUS.md`.

## Состояние

- **S1 ЗАКОММИЧЕН (`43c9ad999`)** — nextn-транспорт по образцу upstream:
  `cparams.embeddings_nextn(+masked)`, `res->t_h_nextn` (пост-final-norm, контракт
  #24025) в qwen35-графе (unmasked-режим отключает in-loop out_ids-редукцию),
  `embd_nextn` в общем output-буфере, извлечение в decode рядом с логитами
  (async, без лишних синков), API `llama_set/get_embeddings_nextn(_ith)`.
  Дефолтный путь топологически не изменён. Ничем пока не потребляется.
- **S2 НЕ ЗАВЕРШЁН** — план: виртуал `process(const llama_batch&)` (+`set_phase`)
  в базовом `common_speculative_state`; сервер зовёт process после каждого
  успешного `llama_decode` (batch_view!); state_mtp кормит ctx_mtp из
  `llama_get_embeddings_nextn_ith(ctx_tgt, i)` (masked: только строки с
  logits!=0, индекс = позиция в batch_view; unmasked: dense по позиции);
  k=0-кондиционирование драфта — из последней обработанной строки (stash) +
  строки verify-батча для случая частичного акцепта (h строки с pos = pos_max);
  фазы через server-gating (LLAMA_SPEC_PREFILL_WINDOW): 0=bulk (nextn off),
  1=хвост промпта (unmasked), 2=генерация (masked). После этого старый хук
  (`handle_mtp_for_ubatch`) удаляется.
- **ВАЖНО (пост-norm консистентность):** t_h_nextn = ПОСТ-norm. Если state
  кормит ctx_mtp пост-norm строками, то и AR-цепочка (k>0) должна быть
  пост-norm: в `qwen35_mtp.cpp` перенести `res->t_mtp_out` ПОСЛЕ
  shared_head_norm (сейчас pre-norm). Иначе смешение контрактов уронит
  acceptance. Если пост-norm даст регресс vs наши 75-81% на pre-norm —
  A/B: вернуть capture в qwen35.cpp на pre-norm И t_mtp_out на pre-norm
  (пара всегда согласована).

## Грабли (проверено на своей шкуре)

1. Отложенный/staged хук БЕЗ переноса extraction в output-пайплайн — НЕ работает
   (эксперимент дал 15.25→10.2 tok/s: flush-синки дороже; см. память/логи).
   Выигрыш апстрима именно в том, что копия nextn едет тем же стримом, что и
   логиты, и синк один.
2. `accept()` обязан видеть fed-KV ДО pos_max/seq_rm — иначе порча MTP-KV
   (acceptance 75%→53%).
3. Измерения: перед сравнением — 1-мин baseline sanity (~29.6 на чистом GPU,
   PL−5%); проверить сторонних потребителей GPU
   (`Get-Counter '\GPU Process Memory(*)\Dedicated Usage'`) — LoL дважды
   портил цифры. GPU2 падает от жёстких teardown'ов — таймаут поднят до 60с
   (`faeb7f43b`), больше не hard-kill'ить сервера раньше.
4. cuda-graph key fix (`41da2ef74`+`c49e0282e`) — verify теперь реплеится
   (150→57мс/verify). Не откатывать.

## Валидация S2 (когда соберётся)

Полоса: ctx4096 b512/ub128 triage_diff max128 temp0.2 no-thinking,
`-dev ROCm1 -sm none` (основной GPU теперь ROCm1!), `--spec-type draft-mtp
--spec-draft-n-max 2`. Референсы (чистый GPU): baseline 29.9; старый хук
n2 = 15.25 (1-GPU) / 24.19 (2-GPU, layer split), acceptance 72.8%.
Цель: 1-GPU выше 20+, 2-GPU выше baseline.
