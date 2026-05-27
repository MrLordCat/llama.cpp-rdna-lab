# D038 P002 Q4 Tool-Call Thinking Guard

Date: 2026-05-27

Status: implemented, built, measured, and promoted to the default server-side
guard for tool-call requests. This is a quality/agent-behavior compensation
route, not a TPS route.

## Trigger

D037 showed that q8/q8 KV is a VRAM/residency problem at `ctx=131072` on the RX
9070 XT 16 GB lane. q8/q8 fits only with direct host-KV relief and falls to
about `0.36 TPS`, so it cannot become the default speed profile. The next
question was whether q4/q4 can stay as the fast default while llama.cpp gives
tool-calling models a safer runtime path.

## Baseline Failure

The D038 q4/q4 baseline uses the same active Vulkan 130k lane:

- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Server: `build-vulkan/bin/llama-server.exe`.
- Shape: `ctx=131072,batch=512,ubatch=256,q4_0/q4_0,FlashAttention`.
- Extra: `--spec-type none --no-mmap`, cold no-reuse, seed `42`.

Two repeats produced the same result:

| Label | Pass Rate | Mean Score | Invalid JSON | Unexpected Tools |
| --- | ---: | ---: | ---: | ---: |
| `d038-toolcall-q4-baseline-r1` | 2/4 | 0.8958 | 0 | 0 |
| `d038-toolcall-q4-baseline-r2` | 2/4 | 0.8958 | 0 | 0 |

The failures were not malformed JSON. The model could call tools, but with
thinking enabled it spent too much of the response budget in hidden reasoning
and lost required final grounded facts after the tool loop. The shorter smoke
repeat also caught a separate failure: the model reasoned about calling tools
but emitted no `tool_calls` before hitting `max_tokens=256`.

## Implementation

`tools/server/server-common.cpp` now applies the guard by default. To disable it
for compatibility testing, set:

```bash
LLAMA_SERVER_TOOL_CALL_THINKING_GUARD=0
```

By default, OpenAI-compatible chat requests with non-empty `tools` and
`tool_choice != none` are rendered with `enable_thinking=false` in the chat
template. The guard is scoped and request-overridable:

- It does not affect normal chat/completion requests without tools.
- It does not override an explicit per-request
  `chat_template_kwargs.enable_thinking` value.
- It can be disabled with `LLAMA_SERVER_TOOL_CALL_THINKING_GUARD=0`, `false`, or
  `off` for A/B and compatibility checks.

This is intentionally narrower than `--reasoning off`: it gives tool-call turns
a structured-output path while preserving thinking for ordinary complex work by
default.

## Measurements

Diagnostic controls:

- `--reasoning-budget 0` did not fix the smoke failure; the sampler closed the
  thinking block, but the model still emitted text planning instead of tool
  calls.
- `--reasoning off` fixed the main no-tool-call mode and improved full D038 to
  `3/4`, showing that the fault was the thinking-to-tool transition rather than
  KV placement.

Code guard result before default promotion:

| Label | Guard | Tasks | Pass Rate | Mean Score | Invalid JSON | Unexpected Tools |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `d038-toolcall-q4-baseline-r2` | off | `q4-weakness` | 2/4 | 0.8958 | 0 | 0 |
| `d038-toolcall-q4-thinkguard-r2` | on | `q4-weakness` | 4/4 | 1.0000 | 0 | 0 |
| `d038-toolcall-smoke-seed42-r2` | off | `smoke` | 0/2 | 0.3833 | 0 | 0 |
| `d038-toolcall-smoke-thinkguard-m384-r2` | on | `smoke` | 2/2 | 1.0000 | 0 | 0 |
| `d038-toolcall-smoke-default-m384-r1` | default | `smoke` | 2/2 | 1.0000 | 0 | 0 |

The guard fixes both observed D038 classes under the same seed: missing tool
emission in smoke and lost final grounding after tool results in the full
q4-weakness workload.

The post-promotion smoke `d038-toolcall-smoke-default-m384-r1` was run with
`LLAMA_SERVER_TOOL_CALL_THINKING_GUARD` unset and confirms the new default-on
path.

## Decision

Promote the guard to the default for server tool-call requests. Keep an explicit
env rollback (`LLAMA_SERVER_TOOL_CALL_THINKING_GUARD=0`) because some complex
agent workflows may still want thinking while planning before tool use. The next
step is a broader public-style agent benchmark and a smarter fallback that can
preserve thinking first, then retry with the guard after no-tool-call or length
failures.
