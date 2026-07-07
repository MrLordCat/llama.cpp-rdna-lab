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

## 4. Reaching ~2× — measured MTP gap analysis (2026-07-07, dual-GPU, peer-copy fixed)

### Measurement (Qwen3.6-27B-Q3_K_S_mtp, ctx 4096, --disable-thinking, max-tokens 256)
| Config | decode tok/s | acceptance |
|--------|-------------:|-----------:|
| baseline (none) | **25.6** | — |
| MTP (n_max=3)   | **20.3** (SLOWER) | **55%** |

So the earlier "MTP broken" was the multi-GPU peer-copy corruption — **acceptance is
now a healthy 55%**. Yet MTP is *slower* than baseline. The loss is pure **draft
overhead**, not acceptance.

### Where the time goes (per `statistics mtp detail`, task 1: 222 tokens)
`dur(sync, get, decode, sample) = 0.3, 27.8, 348, 642 ms` (draft total ≈ 1019 ms).
- **`sample` = 642 ms dominates.** `common_sampler_sample(smpl, ctx_mtp)` for each
  draft token reads the **full-vocab logits row (248320 floats ≈ 1 MB) GPU→host**,
  then argmaxes on CPU. 252 draft tokens ⇒ ~250 MB of transfers + a sync per token.
- `decode` = 348 ms — the MTP head (1 nextn layer) forward per draft token.
- Target verify: MTP eval ≈ 50 ms/tok vs baseline 39 ms/tok (+11 ms/tok).

### Theory: why 55% acceptance still loses
Expected accepted run length with per-token accept p≈0.55, n_max=3:
`0.55 + 0.55² + 0.55³ ≈ 1.02` drafts + 1 resample ≈ **~2.0 tokens per verify**.
That *should* give ~1.5–1.7× **if** each verify+draft ≈ 1–1.3 baseline decodes. But the
per-round draft overhead (esp. the 1 MB logits transfer × n_max) makes each round cost
> 2 baseline decodes ⇒ net slower.

### How others avoid this (EAGLE/EAGLE-2/-3, Medusa, MTP/NextN, lookahead)
1. **Never transfer full-vocab logits to draft.** Draft sampling (usually greedy /
   top-k=1 for the draft) is done **on the GPU** — compute argmax/top-k in the graph and
   copy back only the token id(s) (4 bytes), not 1 MB. This is exactly what our DFlash
   drafter already does via `ggml_argmax` → `llama_get_logits_argmax`.
2. **Keep draft + verify on the GPU with graphs/fusion intact** — no per-token host sync.
3. **Tree / branched drafting** (EAGLE-2/-3, Medusa heads) to raise accepted length per
   verify beyond a single linear chain.
4. **Tune `n_draft`** to the acceptance curve — with p≈0.55 the marginal accept prob at
   depth k is 0.55ᵏ, so depth 3–5 is the useful range; deeper just wastes verify compute.

### Concrete plan for our MTP (ordered by expected payoff)
1. **GPU argmax for the MTP draft (biggest lever).** Add `ggml_argmax` to the qwen35_mtp
   head graph and read the token id via a tiny transfer, replacing the full-logits
   `common_sampler_sample` in `common_speculative_state_mtp::draft()` for the top-k=1
   path. Kills the ~642 ms `sample` term. (Reuse the DFlash `t_logits_argmax` /
   `llama_get_logits_argmax` plumbing already in the tree.)
2. **Confirm the target verify keeps CUDA/HIP graphs** (the MTP hidden-capture hook runs
   post-decode, not via cb_eval, so graphs *should* survive — verify with a graph trace).
3. **Sweep `n_max` (2..5)** at the measured 55% acceptance to find the throughput peak.
4. Later: tree drafting on the MTP head top-k for higher accepted length.

### Update: implemented GPU argmax + n_max sweep (2026-07-07) — the REAL ROCm bottleneck

Cross-checked against Unsloth's Qwen3.6 MTP guide (unsloth.ai/docs/models/qwen3.6#mtp-guide):
they quote **1.4–2.2×** with `--spec-type draft-mtp --spec-draft-n-max 2`, acceptance
**83% @ 2 draft tokens, 50% @ 4** (on RTX 6000: 27B ≈ 160 tok/s). Our acceptance curve
matches theirs — so acceptance is NOT our problem.

Implemented **GPU argmax for the MTP draft** (`e8d0ee0b2`): `sample` term dropped
850 ms → 0.4 ms, acceptance unchanged. But net decode barely moved — the GPU **sync
just relocated** from the logits read into the per-draft `llama_decode(ctx_mtp)`.

n_max sweep (GPU-argmax build, ctx 4096, --disable-thinking, temp 0.2):
| n_max | decode tok/s | acceptance |
|------:|-------------:|-----------:|
| baseline (none) | 25.6 | — |
| 1 | 24.2 | 81% |
| 2 | 23.0 | 66% |
| 3 | 20.3 | 55% |

**Even at the optimal n_max=1 (81% acceptance), MTP is ~5% BELOW baseline.** Root cause
is NOT the logits transfer and NOT acceptance — it is **per-MTP-context-decode dispatch
overhead on ROCm**: `statistics mtp detail` shows ~247 `ctx_mtp` decodes for 50 draft
rounds, because the target-side hidden-capture hook (`handle_mtp_for_ubatch`) runs a
`llama_decode(ctx_mtp)` prefill on **every target decode**, plus the n_max draft decodes.
Each small MTP decode costs ~4–5 ms of launch/sync overhead on ROCm (compute is ~0.6 ms).
On NVIDIA those decodes are ~0.6 ms thanks to CUDA graph replay — which is exactly why
the guide's 1.4–2.2× appears there but not here.

### Revised path to the guide's speedup ON ROCm (ordered)
1. **HIP/CUDA graph capture for the MTP-context decodes** (the single-token draft decodes
   + the hook prefill). This is the big lever: it turns ~5 ms/decode dispatch into ~0.6 ms,
   which is what makes NVIDIA hit 1.4–2.2×. Requires stable graph topology across MTP
   decodes (ggml-hip graph path; check GGML_HIP_GRAPHS covers the MTP ctx).
2. **Reduce hook cost**: the per-target-decode `handle_mtp_for_ubatch` prefill is a large
   fraction of the ~247 MTP decodes — batch it / skip redundant prefills.
3. Keep GPU argmax (done) + n_max=1–2 for this rig.

Net: acceptance is already guide-level; the remaining gap is pure ROCm kernel-dispatch
overhead on the small MTP decodes → needs HIP graphs for ctx_mtp. That is the concrete,
measured answer to "how do others get ~2× and why don't we yet."
