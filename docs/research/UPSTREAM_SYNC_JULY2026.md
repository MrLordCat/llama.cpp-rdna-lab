# Upstream llama.cpp research — May 4 → July 8, 2026

Fork diverged at merge-base `36a694c96` (2026-05-04). Upstream is **906 commits ahead**.
Key finding: upstream has landed official, cleaner implementations of BOTH features we
have been building by hand (MTP pipeline + DFlash), and their MTP architecture fixes
exactly the bottleneck we measured (the per-verify sync hook).

## 1. MTP — official upstream stack (architecturally better than ours)

- `eef59a764` **llm_graph_input_mtp** (#23643) + staging API in `llama-ext.h`:
  `llama_set_embeddings_nextn` / `llama_get_embeddings_nextn_ith`. NextN hidden states
  are extracted **batched through the normal output pipeline** (like logits/embd) —
  there is NO per-target-decode hook with `synchronize()` + host readback + extra
  ctx_mtp decode. This is precisely the fix for our measured MTP tax
  (~50 ms/round on 2-GPU, ~90 ms on 1-GPU; our fork's `handle_mtp_for_ubatch`).
- `166fe2949` (#24025) **qwen35: use post-norm hidden state for MTP** — upstream
  renamed pre_norm→nextn and switched qwen35 to post-norm consistently on both sides.
  (Our fork uses pre-norm capture + a head that applies its own hnorm — works at
  72–81% acceptance, but the *contract differs from upstream*; when porting their
  stack, take their head + their capture together, not halves.)
- `e95dae18d` remove padding + multiple D2D copies for MTP (#24086).
- `5a460dea9` remove redundant CUDA copies after **gated_delta_net** (#23940) — GDN,
  directly relevant to Qwen3.5/3.6 hybrid decode.
- Also: `b3ce5cedf` (quantize MoE+MTP), `260862b8c` (double MTP download),
  `d78952748` (Step3.5/3.7 flash mtp3), `e495d1e74`/`7d2b45b4f`/`04eb4c446` (Gemma4 MTP).

## 2. DFlash — official upstream since `d1b34251b` (#22105)

- **Much leaner than the beellama implementation we ported**: `src/models/dflash.cpp`
  is ~276 lines (bee's dflash_draft.cpp is 1064), +303 in speculative.cpp,
  llama-graph.cpp only +7 — integrated through the same staged-extraction machinery
  rather than eval callbacks / GPU rings.
- Arch string `"dflash"`; hparams key **`target_layers`** (+ SWA pattern) — NOTE: our
  drafter GGUF `Qwen3.6-27B-DFlash-Q8_0.gguf` is in the **bee format**
  (arch `dflash-draft`, keys `dflash.target_layer_ids` etc.) and will NOT load with
  the upstream loader as-is. Upstream ships conversion scripts
  (`conversion/qwen.py`) to produce drafters.
- Follow-ups: `fa72bc682` (conversion refactor), `152d337fa` (spec-draft-p-min in
  DFlash), `a4107133a` (K/V rotation input guard), `4f31eedb0` (qwen3next t_layer_inp).

## 3. Multi-GPU

- **New `docs/multi-gpu.md`** (`b9afc19cb`, #22729): `-sm row` is **deprecated**;
  new **`-sm tensor` (EXPERIMENTAL)** tensor-parallel via a meta-device — splits
  weights AND KV, minimizes token-generation latency (vs `layer` = pipeline parallel,
  which maximizes batch throughput). Caveat: "performance should be good for multiple
  NVIDIA GPUs using the CUDA backend, no guarantees otherwise" — needs a ROCm trial.
  For our decode-bound dual-GPU (layer-split 25.9 vs single 29.9) `-sm tensor` is the
  interesting new option to test after a sync.
- `3fc4e1052` **sched: reintroduce less synchronizations during split compute**
  (#20793) — directly targets multi-GPU split overhead.
- `2da668617` fix stale tensor-split params for draft models.

## 4. Strategic conclusion

The fork has been re-implementing (via beellama) what upstream landed in cleaner,
maintained form. Recommended direction, in order:

1. **Cherry-pick the upstream spec stack** (llama-ext nextn staging +
   llm_graph_input_mtp + #24025 post-norm pair + dflash.cpp + conversion) to replace
   our hook-based MTP and bee-ported DFlash. This removes both measured bottlenecks
   (hook sync; and DFlash gets the staged pipeline instead of our checkpoint-heavy
   round loop). Large but high-value; our branch is literally named
   `research/cherry-pick-upstream`.
2. Cheap standalone cherry-picks meanwhile: `3fc4e1052` (sched syncs, dual-GPU),
   `5a460dea9` (GDN copies), `e95dae18d` (MTP D2D).
3. Re-generate / convert a DFlash drafter in the upstream format (conversion/qwen.py)
   or add a compat shim for the bee-format GGUF.
4. Try `-sm tensor` on ROCm for the dual-GPU decode lane.

Keep in mind our local fixes that upstream may not have: the Windows/RDNA4 HIP
peer-copy corruption guard (`facd9d8ba`) and the cuda-graph stable key
(`41da2ef74`/`c49e0282e`) — verify whether upstream's graph cache has the same
pointer-keyed instability before dropping ours.
