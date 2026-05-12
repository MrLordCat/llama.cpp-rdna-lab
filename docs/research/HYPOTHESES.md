# Candidate Hypotheses Beyond Current ngram + FlashAttention

This file tracks candidate changes that could unlock additional efficiency.

## Core Equation Used For Screening

Approximate decode speedup from speculative decoding:

`S_spec ~= (1 + a * (D - 1)) / (1 + o)`

Coverage-aware variant (for sparse draft coverage):

`S_spec_cov ~= (1 + (c * a_local) * (D - 1)) / (1 + o)`

Where:

- a: accepted draft token ratio
- D: drafted tokens per verification step
- o: relative overhead of draft generation + bookkeeping
- c: coverage (share of verify steps where draft was present)
- a_local: accepted/attempted inside draft-enabled steps

Approximate total wall speedup:

S_total ~= 1 / (p / S_prefill + (1 - p) / (S_spec * S_decode))

Where:

- p: baseline wall-time share spent in prefill
- S_prefill: prefill speedup (for example from attention/kernel work)
- S_decode: decode kernel speedup not covered by speculation

## Hypothesis Backlog

| ID | Idea | Why It Might Work | Expected Impact | Main Risk | First Check |
| --- | --- | --- | --- | --- | --- |
| H01 | Adaptive ngram length by local entropy | Fixed n can be too short in repetitive spans and too long in noisy spans | +5% to +20% decode | extra control overhead | acceptance rate vs entropy bucket |
| H02 | Dynamic draft length policy | Keep D small when mismatch risk is high, increase D in stable segments | +3% to +15% wall | oscillation and instability | accepted tokens per verify call |
| H03 | Hybrid router: ngram or mtp per step | Different spans favor different draft methods | +5% to +25% wall | routing overhead kills gain | net TPS with router on/off |
| H04 | Early reject bound in verify stage | Fast reject without full path when mismatch is obvious | +2% to +10% wall | quality regressions if bound is unsafe | token agreement and exact output diff |
| H05 | Flash-attn tile retarget for current RDNA path | Better tile mapping can reduce memory stalls | +5% to +20% prefill | compile/runtime complexity | prefill tok/s and occupancy |
| H06 | QKV and RoPE fusion in prefill hot path | Fewer launches and less memory traffic | +3% to +15% prefill | register pressure | kernel time breakdown |
| H07 | KV cache layout tuned for decode locality | Better cache-line behavior in long decode loops | +3% to +12% decode | migration complexity | decode tok/s and memory counters |
| H08 | Chunk-size contract alignment (model op + runtime) | Avoid pathological boundaries that trigger slow paths; current bad zone is physical n_ubatch >480 | +5% to +30% at cliff zones | model-specific behavior and boundary drift | current-best ubatch boundary sweep + physical context cap |
| H09 | Coverage-aware speculative acceptance model | Local draft acceptance overestimates global speedup if draft coverage is low | improves prediction fidelity | extra instrumentation complexity | compare implied vs effective acceptance |
| H10 | Overhead-aware speculative model by mode/config | Fixed overhead term misses severe regressions in some high-coverage MTP cases | improves prediction fidelity | more parameters and overfitting risk | backsolve implied overhead across measured cases |

## Priority (Start Here)

1. H08 because existing measurements show sharp boundary cliffs, but target the current 480/490 neighborhood rather than the historical 824/832 pair.
2. H02 because it can be prototyped quickly in scheduler logic.
3. H05 because prefill IO is still dominant in prompt-heavy scenarios.
4. H09 to avoid misleading speculative projections in low-coverage runs.
5. H10 to explain cross-mode speculative regressions with measured overhead.
6. H01 as a low-risk extension of existing ngram flow.

## Evidence Snapshot (E006 Retest)

- Supported by measured evidence: H08 as a boundary/cliff class (current action target: physical n_ubatch cap at 480), H09.
- Supported as modeling-next-step: H10.
- Analytic-only so far: H02.
- Plausible but not measured yet: H01, H03, H04, H05, H06, H07.
