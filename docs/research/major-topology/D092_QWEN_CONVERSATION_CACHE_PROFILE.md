# D092 Qwen conversation-cache profile

## Goal

Provide one explicit server option for high-hit, append-only Qwen3.6 chat
sessions without pretending that the hybrid Gated DeltaNet state is an SWA KV
cache.

## Existing evidence

- Qwen3.6 has `n_swa=0`; `--swa-full` is therefore disabled for this model.
- E111 measured `0.982-0.984` shared-prefix similarity and restored a recurrent
  checkpoint instead of reprocessing the full prompt.
- E346 validated exact append-only reuse at a 57k-token root. The recurrent
  checkpoint is about 150 MiB for the target context, so retaining a bounded
  set of snapshots in host RAM is practical; retaining per-token recurrent
  states is not.

## Design

Add an opt-in `--conversation-cache` server profile that:

- forces request prompt caching on, even when a client sends
  `"cache_prompt": false`;
- retains at least 32 recurrent/SWA rollback checkpoints;
- creates ordinary long-prefill checkpoints at intervals no larger than 8192
  tokens;
- reports the reused-token ratio for every completed prompt;
- leaves the separate idle-prompt RAM cache under `--cache-ram`, because an
  active single-slot chat does not need a second full state copy.

Expose the same profile as one checkbox in the GUI. Advanced cache arguments
placed after the profile may override the profile's numeric defaults.

## Correctness and limits

The mode accelerates token-identical common prefixes. It cannot guarantee a
95% hit when a client rewrites the system prompt, tool schema, chat template,
or compacted history near the beginning of the request. The returned timing
field `cache_n` and the new server log line remain the source of truth.

## Validation

- argument-parser coverage for the new profile;
- Python compile check for the GUI;
- CPU `llama-server` build;
- `git diff --check`;
- no GPU benchmark is required for the configuration-only profile. A future
  live A/B must keep one append-only chat payload and compare
  `cache_n / (cache_n + prompt_n)` after the cold first request.