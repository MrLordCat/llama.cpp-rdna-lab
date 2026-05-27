# D039 P002 BFCL Q3 Tool-Call Pilot

Date: 2026-05-27

Status: measured BFCL-lite pilot. This is a public benchmark data point for
tool-call quality, not an official BFCL leaderboard score and not a TPS result.

## Trigger

D038 promoted the scoped server-side tool-call thinking guard to default-on for
OpenAI-compatible chat requests with tools. The next question was whether that
improvement transfers beyond the local D038 workload into public-style function
calling data.

## BFCL Setup

The official `bfcl-eval` package could not be installed in the active Python
3.13 environment because the package pins `numpy==1.26.4`, which has no Python
3.13 wheel and attempted a local source build. To keep the experiment moving, a
small local harness was added:

- Harness: `scripts/research/bfcl_lite_pilot.py`.
- Public data source: `bfcl-eval==2026.3.23` wheel, extracted outside the repo.
- Data used: BFCL v4 JSONL question files plus `possible_answer` files.
- Scoring: strict single-turn AST-style subset: function name, call count,
  argument keys, types, and accepted values; `irrelevance` passes only with no
  tool calls.

This is sufficient for a local pilot and A/B signal, but it is not a replacement
for the full official runner because it does not run BFCL multi-turn execution,
memory, web-search, or the official aggregate leaderboard scripts.

## Lane

Local server:

- Server: `build-vulkan/bin/llama-server.exe`.
- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Shape: `ctx=131072,batch=512,ubatch=256,q4_0/q4_0,FlashAttention`.
- Extra: `--spec-type none --no-mmap`, seed `42`.
- Runtime: D036 direct host-KV last3 default, graph splits `2`.
- Guard: default-on unless the request explicitly sets
  `chat_template_kwargs.enable_thinking=true`.

## Measurements

Smoke pilot:

| Label | Mode | Cases | Result |
| --- | --- | ---: | ---: |
| `d039-bfcl-lite-default-r1` | default guard | 8 | `8/8` |

Expanded pilot, 5 cases each from `simple_python`, `multiple`, `parallel`,
`parallel_multiple`, and `irrelevance`:

| Label | Mode | Overall | simple | multiple | parallel | parallel_multiple | irrelevance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `d039-bfcl-lite-default25-r1` | default guard | `24/25` | `5/5` | `5/5` | `4/5` | `5/5` | `5/5` |
| `d039-bfcl-lite-thinktrue25-r1` | explicit `enable_thinking=true` | `16/25` | `5/5` | `3/5` | `1/5` | `2/5` | `5/5` |

The explicit-thinking run failed mostly by producing no tool calls before the
`max_tokens=384` cap. This independently supports the D038 conclusion: on this
q3/q4 model, hidden thinking often prevents the transition into structured tool
calls.

The remaining default-guard miss is `parallel_1`. The prompt asks for two
independent calls to the same function, once with `d_time=4` and once with
`d_time=10`; the model emitted only the first call:

```json
[{"calculate_em_force":{"area":2,"b_field":5,"d_time":4}}]
```

This is a different failure from D038's no-tool-call and final-grounding modes:
the model enters tool mode, but under-covers a repeated parallel call.

## Decision

Keep the D038 default guard. It transfers to public BFCL-style single-turn tool
data: `24/25` default versus `16/25` when explicit thinking bypasses the guard.
Do not report this as a leaderboard score or speed result.

Next code work should target a narrower fallback/retry layer:

- retry after no-tool-call responses that hit the output cap while tools are
  present;
- retry or repair repeated parallel-call undercoverage when one call is emitted
  but the request clearly asks for multiple independent invocations;
- keep explicit `enable_thinking=true` respected, and avoid globally disabling
  thinking outside tool-call turns.

## Artifacts

- `scripts/research/bfcl_lite_pilot.py`
- `build_logs/agent-workload/d039-bfcl-lite-default-r1.bfcl_lite.summary.md`
- `build_logs/agent-workload/d039-bfcl-lite-default25-r1.bfcl_lite.summary.md`
- `build_logs/agent-workload/d039-bfcl-lite-thinktrue25-r1.bfcl_lite.summary.md`
- `build_logs/agent-workload/BFCL_LITE_RUNS.csv`
- `build_logs/agent-workload/BFCL_LITE_RECENT.md`