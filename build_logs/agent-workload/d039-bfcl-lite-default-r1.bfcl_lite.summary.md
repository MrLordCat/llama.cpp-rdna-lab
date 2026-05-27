# BFCL Lite Pilot: d039-bfcl-lite-default-r1

This is a small BFCL v4 pilot over public JSONL cases, not an official leaderboard score.

- BFCL data: `C:\Users\Chris\Documents\GitHub\bfcl-eval-work\wheel\bfcl_eval\data`
- Endpoint: `http://127.0.0.1:8088/v1`
- Model field: `qwen3.6-27b-q3ks-local`
- Mode: `tool_choice=auto,chat_template_enable_thinking=unset`
- Temperature: `0.001`; max tokens: `384`; seed: `42`
- Overall: `8/8` (`100.00%`)

| Category | Pass |
| --- | ---: |
| `irrelevance` | `2/2` |
| `multiple` | `2/2` |
| `parallel` | `2/2` |
| `simple_python` | `2/2` |
