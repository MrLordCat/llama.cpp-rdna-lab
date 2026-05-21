# E118 Real Server Output Sanity

## Metadata

- Experiment ID: E118
- Date: 2026-05-21
- Owner: Codex + user live check
- Branch/Commit: master @ 054bccb00 plus E115-E117 local notes
- Driver: AMD `32.0.31007.5012`
- Target lane: Qwen3.6-27B-Q3_K_S, real `llama-server`, thinking on

## Hypothesis

- Statement: a throughput breakthrough is only usable if the real server still produces coherent thinking/answers.
- Mechanism: prior Vulkan `wm32-wn32` work looked promising in speed but produced corrupted slash/symbol output. Decode-route speed gates therefore need a lightweight live-server correctness smoke.
- Why now: E116 found a strong Vulkan decode route (`~40 tok/s`). The user explicitly checked the real server and confirmed the model output is normal.

## Benchmark / Sanity Plan

1. Start the actual `llama-server.exe`, not a synthetic parser.
2. Send a normal coding prompt through `/v1/chat/completions`.
3. Check for:
   - coherent reasoning/answer text;
   - no repeated punctuation/symbol spam;
   - no broken output like the old `wm32-wn32` route;
   - sensible `finish_reason` when a reasoning budget is used.

## Result

- Outcome: keep E116 Vulkan f16 as a real decode-heavy route.
- User live-server verdict: model thinks and answers normally; no symbol spam; the `~40 tok/s` Vulkan decode result is treated as real.
- Direct ROCm sanity artifact:
  - unrestricted reasoning, `max_tokens=1024`: `content_chars=0`, `reasoning_chars=4008`, `finish_reason=length`; text is coherent reasoning, not corruption.
  - `--reasoning-budget 256`, `max_tokens=1024`: normal final answer, `content_chars=712`, `reasoning_chars=1055`, `finish_reason=stop`, `436` completion tokens, decode `29.52 tok/s`.
- Interpretation: an empty `content` field during unrestricted thinking is a reasoning-budget/API extraction issue, not evidence of broken logits or bad backend output. Correctness smoke should look for corruption/repetition and final-answer viability with a bounded reasoning budget.

## Workflow Update

- For every future large speedup, add a live-server smoke before promotion:
  - run the actual target backend/server;
  - ask a normal prompt;
  - reject immediately if output repeats slashes/symbols or reasoning becomes incoherent;
  - if unrestricted thinking consumes the whole limit, repeat with a small `--reasoning-budget` instead of calling the backend broken.
- Keep performance claims separate:
  - `decode route`: use server log decode eval tok/s and long-generation TPS;
  - `prompt-heavy route`: use cold-first/reuse lane metrics;
  - `correctness smoke`: manual/live-server answer sanity.

## Artifacts

- `build_logs/agent-workload/e116-driver5012-decode-vulkan-f16-specnone-r3.diagnostics.md`
- `build_logs/agent-workload/e118-realserver-rocm-q4-specnone-mt1024.response.json`
- `build_logs/agent-workload/e118-realserver-rocm-q4-specnone-reason256-mt1024.response.json`
- `build_logs/agent-workload/e118-realserver-rocm-q4-specnone-reason256-mt1024.server.log`
