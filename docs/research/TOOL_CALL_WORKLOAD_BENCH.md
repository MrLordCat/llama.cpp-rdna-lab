# Tool Calling Workload Benchmark

Date: 2026-05-27

Purpose: reproduce long-context q4 tool-use weaknesses before attempting
runtime/code compensation in llama.cpp. This is a quality/agent-behavior bench,
not a TPS-only lane.

## Why This Exists

D037 showed that q8/q8 KV at `ctx=131072` is a residency/VRAM problem on the RX
9070 XT 16 GB lane. q8/q8 can fit only with direct host-KV relief, but prompt
speed falls to about `185-188 tok/s`; mixed q4/q8 is worse because it creates
`34` graph splits without a usable mixed-KV FlashAttention route.

The next question is therefore different: can q4 remain the fast default while
llama.cpp/server-side code compensates for q4's tool-calling weaknesses? Before
changing runtime code, we need a benchmark that makes those weaknesses visible.

## Script

`scripts/tool_call_workload_bench.py`

It starts or reuses `llama-server`, injects the same repo-snapshot style context
as `scripts/agent_workload_bench.py`, sends OpenAI-compatible `tools`, executes
mock tool results, and records multi-turn tool-loop metrics.

Artifacts are written to `build_logs/agent-workload/`:

- `<label>.toolcalls.jsonl`
- `<label>.toolcalls.csv`
- `<label>.toolcalls.summary.md`
- `<label>.server.log` when the script starts the server
- `TOOL_CALL_BENCH_RUNS.csv`
- `TOOL_CALL_BENCH_RECENT.md`

## Metrics

Primary behavioral metrics:

- `pass_rate`: strict task-level pass/fail.
- `mean_score`: partial-credit score across validation checks.
- `invalid_json_args`: tool calls whose `function.arguments` was not a JSON object.
- `unexpected_tool_calls`: model called a decoy or unavailable tool.
- `mean_turns`: number of assistant/tool turns needed.
- `mean_max_parallel`: largest tool batch per task; lower values expose failure
  to batch independent tool calls.

Secondary timing metrics:

- `aggregate_completion_tps_wall`: completion tokens divided by wall time across
  the tool loop.
- server-log prompt/decode diagnostics, when available.

Do not use this bench as a replacement for the dense 130k TPS baseline. Use it
to compare tool-use behavior at the same server shape.

## Task Set

Default `--tasks q4-weakness` includes:

| ID | Weakness Surface |
| --- | --- |
| `tc_context_parallel` | long-context independent tool batching, source lookup, final grounded synthesis |
| `tc_bench_compare_args` | exact long benchmark labels, JSON argument fidelity, q4-vs-q8 decision |
| `tc_error_recovery` | recovery from a wrong path via `file_search`, then correct file read |
| `tc_tool_restraint` | avoid decoy destructive/heavy tools when only benchmark planning was requested |

`--tasks smoke` runs only `tc_bench_compare_args` and `tc_tool_restraint`.

List tasks without starting a server:

```bash
python scripts/tool_call_workload_bench.py --list-tasks --tasks q4-weakness --real-context-mode off
```

## Canonical Q4 Baseline Command

Use this as the first q4 tool-calling baseline on the current Vulkan default:

```bash
unset HSA_OVERRIDE_GFX_VERSION GGML_TRACE_FATTN_SELECTED GGML_TRACE_FATTN_TIMING GGML_TRACE_FATTN_TIMING_SYNC GGML_TRACE_FATTN_TIMING_PRE_SYNC GGML_TRACE_CUBLAS_SPLIT_TIMING GGML_TRACE_CUBLAS_Q3K_ROUTE GGML_TRACE_MMVQ_TIMING GGML_TRACE_MMVQ_RESOURCES GGML_VK_PIPELINE_STATS GGML_VK_FA_ROUTE_TRACE GGML_VK_FFN_ROUTE_TRACE GGML_CUDA_FORCE_MMQ_RUNTIME GGML_VK_AMD_LARGE_MATMUL_VARIANT GGML_VK_QK_LOW_TILE_SPLIT_K GGML_VK_Q3K_QUAD_DEQUANT GGML_VK_DISABLE_AMD_BN256_DEFAULT GGML_VK_DISABLE_QK_LOW_TILE_DEFAULT GGML_VK_DISABLE_Q3K_QUAD_DEQUANT LLAMA_DISABLE_VK_KV_HOST_AUTO LLAMA_VK_KV_HOST_LAYERS LLAMA_VK_KV_HOST_DIRECT LLAMA_VK_KV_HOST_POSITION LLAMA_VK_KV_HOST_MODE LLAMA_VK_KV_HOST_AUTO_Q8
PATH="/c/Strawberry/c/bin:$PATH" GGML_VK_ALLOW_GRAPHICS_QUEUE=1 python scripts/tool_call_workload_bench.py \
  --server-bin build-vulkan/bin/llama-server.exe \
  --model models/Qwen3.6-27B-Q3_K_S.gguf \
  --label d038-toolcall-q4-baseline-r1 \
  --ctx-size 131072 --batch-size 512 --ubatch-size 256 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --gpu-layers 999 --flash-attn --parallel 1 \
  --max-tokens 384 --tasks q4-weakness \
  --real-context-mode repo-snapshot --real-context-chars 24576 \
  --no-reuse \
  --request-timeout 180 --startup-timeout 900 --task-hard-timeout 180 \
  --background-server-policy fail \
  --server-extra "--spec-type none --no-mmap"
```

Shorter actual command:

```bash
PATH="/c/Strawberry/c/bin:$PATH" GGML_VK_ALLOW_GRAPHICS_QUEUE=1 python scripts/tool_call_workload_bench.py \
  --server-bin build-vulkan/bin/llama-server.exe \
  --model models/Qwen3.6-27B-Q3_K_S.gguf \
  --label d038-toolcall-q4-baseline-r1 \
  --ctx-size 131072 --batch-size 512 --ubatch-size 256 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --gpu-layers 999 --flash-attn --parallel 1 \
  --max-tokens 384 --tasks q4-weakness \
  --real-context-mode repo-snapshot --real-context-chars 24576 \
  --no-reuse --request-timeout 180 --startup-timeout 900 --task-hard-timeout 180 \
  --background-server-policy fail \
  --server-extra "--spec-type none --no-mmap"
```

## Observed Q4 Baseline

The first two q4/q4 baseline repeats used the same server seed (`--seed 42`),
same cold no-reuse launch, and the same `repo-snapshot` context. Both runs
produced the same pass/fail pattern:

| Label | Pass Rate | Mean Score | Invalid Args | Unexpected Tools | Mean Max Parallel | Wall TPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `d038-toolcall-q4-baseline-r1` | 2/4 | 0.8958 | 0 | 0 | 2.00 | 15.1501 |
| `d038-toolcall-q4-baseline-r2` | 2/4 | 0.8958 | 0 | 0 | 2.00 | 15.1011 |

Failures are not JSON syntax failures. The model can call tools, recover from a
bad path, and avoid destructive decoys, but it loses required final-grounding
facts after the tool loop:

- `tc_context_parallel`: uses the requested parallel source lookups, but the
  final answer omits the q8 opt-in env and mixed-KV rejection decision.
- `tc_bench_compare_args`: looks up both exact labels and compares the right
  metrics, but does not complete the final "not default" decision.

The shorter smoke repeat (`d038-toolcall-smoke-seed42-r1/r2`) also catches a
separate failure class with the same seed: the model reasons about the required
tool calls but emits no `tool_calls` before hitting `max_tokens=256`.

Use these rows as the current q4 quality baseline before testing q8 or
llama.cpp/server-side compensation.

## Tool-Call Thinking Guard

D038 added a default server-side guard for tool-call requests. Disable it only
for compatibility A/B with:

```bash
LLAMA_SERVER_TOOL_CALL_THINKING_GUARD=0
```

By default, `/v1/chat/completions` requests with non-empty `tools` and
`tool_choice != none` render the chat template with `enable_thinking=false`,
unless the request explicitly sets `chat_template_kwargs.enable_thinking`. This
does not change normal non-tool requests and is not the same as globally running
`--reasoning off`.

Measured on the same q4/q4 Vulkan 130k lane:

| Label | Guard | Tasks | Pass Rate | Mean Score | Invalid Args | Unexpected Tools |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `d038-toolcall-q4-baseline-r2` | off | `q4-weakness` | 2/4 | 0.8958 | 0 | 0 |
| `d038-toolcall-q4-thinkguard-r2` | on | `q4-weakness` | 4/4 | 1.0000 | 0 | 0 |
| `d038-toolcall-smoke-seed42-r2` | off | `smoke` | 0/2 | 0.3833 | 0 | 0 |
| `d038-toolcall-smoke-thinkguard-m384-r2` | on | `smoke` | 2/2 | 1.0000 | 0 | 0 |
| `d038-toolcall-smoke-default-m384-r1` | default | `smoke` | 2/2 | 1.0000 | 0 | 0 |

Use the guard as the default q4/q3 tool-call reliability compensation. Keep the
disable env for A/B and compatibility checks, because some complex planning-heavy
tool workflows may still benefit from request-explicit thinking.

## Q8 Comparison Probe

Only use q8/q8 as the D037 opt-in stability comparison. It is expected to be
slow, so compare quality metrics first and timing second:

```bash
PATH="/c/Strawberry/c/bin:$PATH" GGML_VK_ALLOW_GRAPHICS_QUEUE=1 LLAMA_VK_KV_HOST_AUTO_Q8=1 python scripts/tool_call_workload_bench.py \
  --server-bin build-vulkan/bin/llama-server.exe \
  --model models/Qwen3.6-27B-Q3_K_S.gguf \
  --label d038-toolcall-q8-autoq8-r1 \
  --ctx-size 131072 --batch-size 512 --ubatch-size 256 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --gpu-layers 999 --flash-attn --parallel 1 \
  --max-tokens 384 --tasks q4-weakness \
  --real-context-mode repo-snapshot --real-context-chars 24576 \
  --no-reuse --request-timeout 180 --startup-timeout 900 --task-hard-timeout 240 \
  --background-server-policy fail \
  --server-extra "--spec-type none --no-mmap"
```

## How To Use Results

The first useful comparison is not whether q8 is faster; D037 already says it
is not. The useful comparison is whether q8 materially improves:

- exact tool selection;
- JSON argument validity;
- recovery from tool errors;
- final answer grounding after tool results;
- independent-call batching.

If q8 improves those metrics while q4 fails, future code work should target the
observed q4 failure class. Candidate compensation areas include server-side tool
grammar/repair, stricter chat-template tool-call framing, retry-on-invalid JSON,
and optional tool-call constrained decoding. Any runtime change must be measured
against this same bench before being promoted.