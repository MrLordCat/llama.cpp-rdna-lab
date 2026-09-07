# W15: MTP draft-batch weight-stream audit (C2)

Date: 2026-09-07

Scope: C2 from W13 - "if the MTP decode drives each draft through its own
`ncols_dst==1` launch, the weight stream is re-read per draft token; batching
2-4 draft tokens into `ncols_dst 2..4` would read the weights once per draft
group." Source audit only; no GPU launches. Verdict: **CLOSED-REJECTED**
(premise corrected at two levels).

## Facts from source

### Draft context construction (`common/speculative.cpp:2715-2760`)

- For MTP (`--spec-type mtp`), `ctx_dft = llama_init_from_model(model_tgt,
  cparams)` with `cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP`: a SECOND context
  on the SAME target model, not a separate drafter model.
- Graph for that context (`src/models/qwen35.cpp:575-808`, `graph_mtp`):
  `il = n_layer - nextn_predict_layers + offset`; with
  `nextn_predict_layers = 1` (this UD GGUF) it executes **ONE transformer
  block + the NextN head** (eh_proj, attn, FFN, shared-head norm, LM head).
  It is NOT the full 65-layer model.
- The MTP block is the last block (`il = 64` of 0-64), excluded from the
  recurrent set (`n_main = n_layer - nextn_predict_layers`, qwen35.cpp:26-30),
  so it carries full-attention tensors (`wq/wk/wv/wo` + FFN).

### Draft generation (`draft()`, speculative.cpp:1830-2030)

- Loop `i = 0, 1, ...` while any sequence is drafting; each iteration calls
  `decode_hidden_batch(ctx_dft, ...)` with a batch of **one token per
  sequence** (`common_batch_add(batch, id, n_past + i + 1, {seq_id}, true)`).
- Production lane is `-np 1` -> `n_seq == 1` -> **exactly 1 token per draft
  step** -> all MMVQ inside `graph_mtp` run `ncols_dst == 1`
  (wq/wk/wv/wo, FFN gate/up/down, LM head).
- **Auto-regressive dependency**: draft step i+1 must sample from step i's
  logits before decoding. Draft positions within one sequence therefore
  CANNOT be batched into one `llama_decode` - this kills the C2 mechanism for
  the single-slot lane. Batching across sequences already exists (the loop
  adds one token per `drafting` seq_id into the same batch).

### Verification (`tools/server/server-context.cpp:651-690`)

- The target batch is built as `sampled` + ALL `spec_draft` tokens in one
  `common_batch_add` sequence (`spec_i_batch`), then decoded together:
  `n_tokens = 1 + n_draft` -> the full-model verify is already batched
  (`ncols_dst 2..4` MMVQ for the whole 65-block model), so verify reads the
  weight stream ONCE per draft group, not per draft token.

## Corrected cost picture (per MTP verify cycle, n_seq=1, n_max=2)

1. Draft: 2 sequential single-token decodes of the **MTP block** (one
   transformer block + head, ~340M params, Q4 ~170 MB - about 2.6x the 64 MB
   L2, so it cannot be L2-resident). Weight stream ~2 x 170 MB per cycle.
2. Verify: 1 batched decode of the full model with `ncols=3` (sampled + 2
   drafts) - weight stream ~17 GB read ONCE (already optimal for the
   workload; a `ncols=1` per-draft verify would have been 2-3x worse).
3. Waste is therefore small relative to the original premise: per draft
   step the re-read is only the 1-block MTP subgraph, not the full model, and
   the dominating full-model stream is already batched.

## Verdict

- The C2 premise ("MTP drives each draft through its own full-model launch")
  is FALSE: full-model verify is batched; only the MTP-layer draft steps are
  per-token, and those are semantically dependent (AR) plus only 1/65 of the
  model.
- A "batch draft tokens into ncols 2..4" change is not applicable:
  within-sequence batching is impossible (AR), across-sequence batching
  already implemented, and the residual lever (MTP block L2 residency,
  170 MB > 64 MB) is closed.
- C2 is closed. The remaining MTP levers are acceptance/tuning, not
  weight-stream geometry (the `+24.97%` decode / `97-99%` acceptance MTP n2
  row in RESULTS_LOG is the production config and is unchanged).

## Measurement pointer (optional, no code change)

If a future session wants hard evidence:
`LLAMA_SPEC_SERVER_PHASE_TIMING=1` (server phase timing: draft vs verify wall)
plus `GGML_TRACE_MMVQ_RESOURCES=1` on the 49K f8 MTP lane to confirm
`ncols_dst` of the verify MMVQ groups. Not required for the verdict: the
graph structure is definitive.
