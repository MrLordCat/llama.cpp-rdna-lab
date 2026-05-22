# E184 - DFlash Runtime End-to-End Smoke

## Metadata

- Experiment ID: E184
- Date: 2026-05-22
- Owner: Copilot
- Branch/Commit: local working tree (no commit yet)
- Target lane: runtime correctness validation (no TPS claim)

## Scope

Validate that the integrated DFlash path is not only parse/build-complete, but also runs through the live server HTTP path.

## Validation Steps

1. Start server with DFlash + draft model:
   - `build-rocm-vec/bin/llama-server.exe`
   - `--model models/Qwen3.5-9B-Q6_K.gguf`
   - `--spec-type dflash`
   - `--spec-draft-model models/Qwen3.5-9B-Q6_K.gguf`
   - `--host 127.0.0.1 --port 8114`
2. Verify startup log reports:
   - DFlash backend helper hooks detected;
   - speculative decoding context initialized;
   - `server is listening on http://127.0.0.1:8114`.
3. Call HTTP endpoints:
   - `GET /health` -> `{"status":"ok"}`
   - `POST /completion` with short deterministic prompt.
4. Confirm generation response contains:
   - `"speculative.type":"dflash"`
   - non-empty generated `content`.
5. Stop server and verify no lingering `llama-server.exe` process.

## Result

- Outcome: Keep.
- Delta: no TPS claim (runtime correctness gate).
- Notes:
  - DFlash end-to-end request path is operational with draft model.
  - Parser hardening from E183 now rejects DFlash without draft model before runtime.
