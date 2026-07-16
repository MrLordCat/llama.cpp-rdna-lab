# E315: ROCm long-context MTP sparse history

Date: 2026-07-15

## Goal

Recover MTP acceptance on long prompts without paying the prompt-eval cost of
prefilling the complete MTP context.

## Root cause

The old 256-token MTP prefill window left the draft attention cache with too
little target history. This was not a ROCm numerical defect: exact-prefix
traces gave equal Vulkan and ROCm acceptance when both backends saw the same
target tokens and MTP history.

Full MTP history restored acceptance, but the original process graph evaluated
the complete MTP layer for every prompt row. The retained implementation:

- builds a KV-only Qwen MTP process graph and skips attention output, FFN,
  final norm, and LM head work when no logits are requested;
- keeps selected target hidden states on the backend;
- preallocates staging before prompt evaluation;
- defers sparse draft prefill until the next selected block;
- uses the existing event-ordered ROCm cross-context handoff instead of a full
  target scheduler drain;
- captures 4096 rows every 32768 prompt positions plus the latest 256 rows.

The sparse policy is the HIP default. The environment variables remain as
rollback and research controls:

- `LLAMA_SPEC_PREFILL_WINDOW` (default `256`);
- `LLAMA_SPEC_PREFILL_SPARSE_STRIDE` (HIP default `32768`);
- `LLAMA_SPEC_PREFILL_SPARSE_CHUNK` (HIP default `4096`);
- `LLAMA_MTP_KV_ONLY_PROCESS=0` disables the KV-only process graph;
- `LLAMA_MTP_ASYNC_DEVICE_HANDOFF=0` restores the full source synchronize;
- `LLAMA_MTP_DEFER_SPARSE_PREFILL=0` disables deferred sparse prefill;
- `LLAMA_MTP_PREALLOC_DEVICE_STAGING=0` restores lazy staging allocation.

## Results

Hardware and route: dual RX 9070 XT, `ROCm1,ROCm0`, layer split `1,1`, Qwen3.6
27B Q3_K_S MTP GGUF, q8/q8 KV, `b8192/ub1024`, deterministic 128-token
generation, no reuse, no prime.

### 29,563-token prompt

| Route | Prompt tok/s | Decode tok/s | Acceptance |
| --- | ---: | ---: | ---: |
| No MTP | 1787.94 | 25.21 | - |
| Sparse MTP n3, 4096-row anchor | 1721.97 | 42.02 | 75.86% |

The MTP route loses 3.7% prompt throughput and gains 66.7% decode throughput.

### 41,114-token prompt

| Route | Wall TPS | Prompt tok/s | Decode tok/s | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| No MTP | 4.30 | 1670.27 | 25.34 | - |
| Sparse MTP n3, 1024-row anchor | 4.34 | 1603.73 | 33.87 | 63.08% |
| Sparse MTP n3, 4096-row anchor | 4.36 | 1597.23 | 35.92 | 68.55% |
| Full KV-only MTP history | 4.12 | 1488.77 | 37.54 | 74.36% |

The 4096-row sparse policy retains most of the full-history acceptance while
limiting prompt loss to 4.4%. Full history is rejected as the default because
its 10.9% prompt loss makes total wall time worse on this prompt-heavy lane.

Increasing only the recent window to 512, 1024, or 4096 did not materially
improve acceptance. Adding a 16k midpoint anchor also failed. Wider 4k anchor
blocks, rather than more frequent sparse points, were the useful compromise.

## Vulkan side finding

Vulkan device handoff kept the unmasked NextN output active over the whole
prompt, even while MTP processing was window-disabled. On the matched 29,563
prompt this reduced prompt eval to about 852 tok/s. Vulkan now defaults to the
host handoff, which enables NextN only for the active window:

| Vulkan route | Prompt tok/s | Decode tok/s | Acceptance |
| --- | ---: | ---: | ---: |
| No MTP | 1556.89 | 35.45 | - |
| MTP n3 after handoff fix | 1508.01 | 45.20 | 52.38% |

This restores prompt eval to within 3.1% of baseline while improving decode by
27.5%. ROCm keeps device handoff because its event-ordered path is beneficial.

## Decision

Keep the HIP sparse-history defaults and the backend-specific handoff policy.
For long agent prompts, MTP now has a bounded 3.7-4.4% prompt cost and a
41.7-66.7% decode gain. Do not use full-history prefill as the default.

Key artifacts use the prefixes `e318-`, `e319-`, `e320-`, and `e321-` in
`build_logs/agent-workload`.

## 2026-07-16 correctness follow-up

E338 found an adjacent-active-batch edge case in deferred sparse prefill. A
pending sparse block could survive a true-to-true capture gate transition and
be overwritten by the final recent-window batch. The implementation now
flushes the pending block before every next active capture and checks failed
draft-memory trims. A 72,295-token MTP n3 validation completed with no warning,
74.14% acceptance, and 32.53 decode tok/s. See
[E338: ROCm dual-GPU long-context scheduler residency](E338_rocm_dual_long_context_scheduler_residency.md).
