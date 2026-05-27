# BFCL Lite Pilot: d039-bfcl-lite-default25-r1

This is a small BFCL v4 pilot over public JSONL cases, not an official leaderboard score.

- BFCL data: `C:\Users\Chris\Documents\GitHub\bfcl-eval-work\wheel\bfcl_eval\data`
- Endpoint: `http://127.0.0.1:8088/v1`
- Model field: `qwen3.6-27b-q3ks-local`
- Mode: `tool_choice=auto,chat_template_enable_thinking=unset`
- Temperature: `0.001`; max tokens: `384`; seed: `42`
- Overall: `24/25` (`96.00%`)

| Category | Pass |
| --- | ---: |
| `irrelevance` | `5/5` |
| `multiple` | `5/5` |
| `parallel` | `4/5` |
| `parallel_multiple` | `5/5` |
| `simple_python` | `5/5` |

| ID | Error | Actual Calls |
| --- | --- | ---: |
| `parallel_1` | `wrong_call_count` | `1` |
