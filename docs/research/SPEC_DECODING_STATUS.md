# Speculative Decoding Status — MTP & DFlash

Status doc for the two custom speculative-decoding paths in this fork, targeting
Qwen3.6-27B on 2× RX 9070 XT (ROCm/HIP, Windows). Last updated 2026-07-07.

Rig: 2× RX 9070 XT (gfx1201, 16 GB each = 32 GB), ROCm 7.1, Windows 11.
Target model: `models/Qwen3.6-27B-Q3_K_S.gguf` (dense qwen35 arch, 64 layers).
Baseline decode (spec=none): ~28–40 tok/s single-GPU; prompt-eval ~800–1100 tok/s.

---

## 0. TL;DR

| Path   | State | Acceptance | Decode | Verdict |
|--------|-------|-----------:|-------:|---------|
| **MTP**    | works after multi-GPU fix | draft head runs; acceptance modest | ~faster than none but **not the expected big win** | needs tuning to reach ~2× |
| **DFlash** | Phase 1 functional end-to-end | ~7.5% (≈1 token/block) | ~5 tok/s (**slower** than baseline) | not yet a speedup; Phase 2 needed |

Both compile+link+run on dual-GPU. Neither yet delivers the target ~2× speedup.

---

## 1. Critical dependency fixed: multi-GPU HIP peer-copy corruption

**Root cause (found 2026-07-07):** on Windows + HIP/RDNA4, GPU-to-GPU peer copies
**silently corrupt cross-device tensors**. This broke generation on multi-GPU
("mangled thoughts", MTP garbage / ~0% acceptance) while single-GPU worked.

**Fix** (`ggml/src/ggml-cuda/ggml-cuda.cu`, commit `facd9d8ba`):
`ggml_cuda_peer_copy_enabled()` returns `false` by default on Win+HIP → uses
host-staged cross-device transfers. Opt back in with `GGML_ROCM_ENABLE_PEER_COPY=1`.

**Implication:** earlier dual-GPU MTP "0% greedy acceptance" was largely THIS
corruption, not only the MTP hidden-state bug. Re-measure MTP on the fixed build.

Also note: RX 9070 XT #2 hit Windows driver **code 43** during this work; cleared
by a full ROCm driver reinstall. When only 1 GPU is visible (`found 1 ROCm
devices`) suspect a driver/code-43 state.

---

## 2. MTP (Multi-Token Prediction / NextN)

### What it is
Qwen3.6 ships a `-mtp` GGUF (`Qwen3.6-27B-Q3_K_S_mtp.gguf`, `nextn_predict_layers=1`)
with an extra NextN head. Runtime: `--spec-type mtp --spec-draft-n-max N`. The head
is a separate context (`ctx_type=MTP`) built from the target model; `llama_set_mtp`
installs a post-decode hook that captures the target's hidden state and prefills the
MTP context, then an AR loop drafts N tokens.

Key files: `common/speculative.cpp` (`common_speculative_state_mtp`),
`src/llama-context.cpp` (`set_mtp`, `handle_mtp_for_ubatch`), `src/models/qwen35_mtp.cpp`
(the NextN head graph), `src/models/qwen35.cpp` (hidden-state capture point).

### Fixes already landed
- **Pre-norm hidden state** (`c6caa5dba`): the NextN head applies its own `hnorm`
  to the input, so it must receive the PRE-(final-)norm hidden (raw last-layer
  residual). Capturing post-`output_norm` double-normalizes → near-random drafts.
  `src/models/qwen35.cpp` / `qwen35moe.cpp` capture `t_h_pre_norm` before the final norm.
- **Multi-GPU peer-copy** (see §1) — the big one for dual-GPU correctness.
- MTP dual-GPU follow-ups by a parallel effort (context/mtp.h/server-context.cpp).

### Current problem (the reason for this doc's next section)
MTP **runs** but the decode speedup is **smaller than expected** — nowhere near ~2×.
The draft/verify loop works but acceptance and/or per-step overhead cap the gain.
See §4 for the plan to reach ~2×.

### How to bench MTP
```
python scripts/agent_workload_bench.py --label mtp-check \
  --model models/Qwen3.6-27B-Q3_K_S_mtp.gguf --ctx-size 4096 \
  --batch-size 512 --ubatch-size 128 --max-tokens 256 --temperature 0.2 \
  --tasks quick --runs 1 --disable-thinking \
  --server-extra "--spec-type mtp --spec-draft-n-max 3"
```
The harness parses `statistics mtp` acceptance into diagnostics.md.
NOTE: agent-workload tasks emit near-empty output with thinking ON — use
`--disable-thinking` + high `--max-tokens` so decode is actually exercised.

---

## 3. DFlash (block-diffusion speculative decoding)

### What it is
Ported from `../beellama.cpp` (`Anbeeld/beellama.cpp` @ `c6dfa39e3`). A separate
**trained drafter** GGUF (`models/Qwen3.6-27B-DFlash-Q8_0.gguf`, arch `dflash-draft`,
5 layers, `block_size=16`, `target_layer_ids=[1,16,31,46,61]`, `n_target_features=25600`)
predicts a whole block of tokens conditioned on the target's captured hidden states.

Runtime: `--spec-type dflash --spec-draft-model models/Qwen3.6-27B-DFlash-Q8_0.gguf`.
Because the drafter is a separate ~1.85 GB model, it needs its own device — run
`--spec-draft-device ROCm1` and pin the target with `-dev ROCm0` (both don't fit on one 16 GB card).

### Architecture (as implemented, Phase 1 CPU-safe)
1. **Target hidden capture** (`src/llama-context.cpp`): an eval callback
   (`cb_eval = dflash_eval_callback`) intercepts the target's `l_out-<il>` layer
   activations for `target_layer_ids` and appends them into `layer_hiddens` buffers.
   Buffers accumulate the committed context; the DFlash state truncates them to the
   committed length each `draft()`.
2. **Cross window** (`common/speculative.cpp`, `common_speculative_state_dflash`):
   `build_cross()` packs the last `cross_window` (512) captured tokens as
   `v_embd[token*n_feat + layer*n_embd + e]` and sets the drafter's cross via
   `llama_dflash_set_cross`.
3. **Drafter graph** (`src/models/dflash_draft.cpp`, ~1220 lines): the block queries
   attend in ONE fused attention over `[cross-window keys (ctx_len) + block keys]`
   (non-causal within the block). `fused_target = norm(dflash_fc(target_hidden))` is
   the cross K/V source. No persistent self-KV — the drafter is stateless per block.
4. **Draft**: decode `[id_last, mask×N]` at absolute positions; read the greedy
   `ggml_argmax` (`llama_get_logits_argmax`) for positions 1..N as the drafts.

Phase-1 simplifications vs bee (deferred to Phase 2): greedy argmax only (bee has
`ggml_argmax_ext`/`ggml_topk_ext` with temperature+Gumbel); GPU cross-ring + GPU
tape + multi-slot + server prefill-flush integration omitted (CPU host path only).

### What works
Loads + runs end-to-end on dual-GPU: `dflash: state ready`, drafts generated and
accepted, no crash. 22 commits (`b04f5465c..21bf6247e`), tree clean.

### Current problems (blockers to a speedup)
1. **Block acceptance stuck ~7.5% — exactly ~1 token/block.** The first block token
   (conditioned on the real `id_last`) is reliably accepted; the 14 mask-conditioned
   positions are ~never accepted. Tried absolute positions, sliding cross window, and
   committed-context accumulation — **none moved it.** So the target verifies a
   16-token block but advances ~2 → ~8× wasted target compute.
   - Ruled out: hidden-state form (drafter self-projects+norms `target_hidden`; `l_out`
     residual in `target_layer_ids` order is correct — NOT an MTP-style norm bug);
     drafter statelessness (architecturally there is no self-KV to maintain).
   - Leading hypotheses: (a) **block diffusion needs ITERATIVE denoising** — multiple
     drafter passes per block, re-masking low-confidence positions (a single pass only
     nails the first token); (b) a subtle position/mask detail in `set_input`.
2. **Target verify decode ~200 ms/tok (~6× slow).** The `cb_eval` hidden-capture
   callback runs per graph node and **disables CUDA/HIP graph capture + fusion**, so
   even correct drafting can't beat baseline. Needs **graph-embedded capture** (the
   target graph copies `l_out` into persistent buffers, no eval callback) — bee's Phase 2.
3. Minor: `--spec-draft-n-max 4` errors "Invalid input batch" (default 16 works) —
   a batch-sizing edge case in `draft()`.

### Next DFlash steps (Phase 2)
- Graph-embedded hidden capture (restore CUDA graphs; kill the ~6× verify penalty).
- Iterative block-diffusion denoise to raise block acceptance beyond 1.
- Diagnostic first: dump a few draft token pieces vs the target's actual next tokens
  to classify plausible-but-rejected (→ iteration/threshold) vs garbage (→ pos/mask).

---

## 4. Reaching ~2× — how speculative decoding actually wins (for the MTP push)

(To be expanded in the MTP-speedup investigation. Placeholder outline:)

Speculative decoding speedup ≈ `(1 + accepted_per_step) / (1 + draft_overhead_ratio)`.
To approach 2× you need BOTH:
- **High acceptance** (mean accepted draft length per verify ≥ ~1.5–2 for a 2× target),
- **Cheap drafting + cheap verify overhead** (draft model much smaller/faster than target;
  verify batch amortized; no per-step host stalls; graphs/fusion intact).

Common techniques others use (EAGLE/EAGLE-2/-3, Medusa, MTP/NextN, lookahead):
- tree/branched drafting to raise accepted-length,
- self-speculation via a cheap head (MTP/EAGLE) to keep draft cost tiny,
- keeping the verify on the GPU with graphs (avoid callback/host sync per token),
- tuning `n_draft` to the acceptance curve (too large wastes verify compute).

Our MTP gap analysis + concrete tuning follow in the investigation below.
