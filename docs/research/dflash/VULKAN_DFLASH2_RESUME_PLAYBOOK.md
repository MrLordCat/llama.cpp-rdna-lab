# Vulkan DFlash2 Resume Playbook

Status date: 2026-08-20
Branch: `dflash2`
Scope: Qwen3.8-27B target + Qwen3.8-27B-DFlash2 on 2x RX 9070 XT, Vulkan.

This is the active resume document for the current DFlash2 work. The older
`docs/research/SPEC_DECODING_STATUS.md` describes the legacy Qwen3.6 DFlash1
port and is not the runtime contract for this branch.

## Primary sources

1. Model card: <https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2>
2. Reference implementation: <https://github.com/z-lab/dflash>
3. llama.cpp DFlash2 PR: <https://github.com/ggml-org/llama.cpp/pull/27342>
4. Batched greedy divergence issue: <https://github.com/ggml-org/llama.cpp/issues/27407>
5. mtmd/cache-hole follow-up: <https://github.com/z-lab/llama.cpp-fork/pull/1>

The browser snapshots were refreshed on the status date. Treat upstream PR and
issue state as time-sensitive when resuming later.

## Source-backed invariants

- The trained block size is 8: one real token plus at most seven drafted tokens
  per verification step.
- The official acceptance-length metric is completion tokens divided by
  verification steps; it is not the same as the printed accepted/generated
  draft-token percentage.
- The model-card reference uses SGLang on one H200, block size 8, Qwen's
  recommended sampling (`temperature=1`, `top_p=0.95`, `top_k=20`), xhigh
  reasoning, and up to 4096 new tokens. Those throughput numbers are not a
  Vulkan/RDNA4 baseline.
- Upstream issue #27407 shows that greedy speculative output can differ from
  the sequential target because verification changes the target batch shape.
  `draft-simple` reproduces the class without DFlash2; DFlash2 can amplify it.
  The issue reports coherent output and unaffected acceptance, and explicitly
  does not claim a logic fix.
- Therefore bit-exact parity and runtime stability are separate gates. A stable
  speculative hash that differs from the sequential target is evidence, not by
  itself a corruption verdict.
- Upstream PR #27342 includes reports of poor Vulkan/ROCm multi-session scaling
  on dual R9700-class systems. It also contains a multi-GPU shared-vocabulary
  device-placement failure. Preserve explicit device ordering in local runs.
- z-lab fork PR #1 documents draft-cache position holes from mtmd embedding
  chunks and reused prefixes. Text-only parity tests do not cover that path.

## Local fixes retained

- DFlash encoder input uses `n_embd_inp_enc()` (five target layers fused to
  25,600 elements for this model), preventing the earlier OOB/SIGSEGV.
- Scheduler reservation includes the encoder graph, eliminating the mismatched
  Vulkan compute-buffer cleanup warning.
- Empty recurrent cells have their physical R/S storage zeroed, fixing repeated
  same-slot target-only drift from stale state.
- Server speculative `max_tokens` accounting increments only for emitted
  tokens; boundary regression cases cover 1, 2, 7, 8, 9, 15, and 16.

## Verified local behavior

- Single-slot target versus DFlash2 greedy hashes matched on the short parity
  set, including repeated same-slot calls, max-token boundaries, streaming,
  stop sequences, and chat completions.
- `LLAMA_DFLASH_UBATCH=8` completed without the old encoder crash and retained
  single-slot parity.
- Four identical concurrent target-only prompts were stable, but identical
  DFlash2 prompts produced multiple hashes in all three 2026-08-20 quick runs:
  normal graph reuse (`2`), global graph reuse disabled (`3`), and
  `LLAMA_DFLASH_UBATCH=8` (`2`). This is a failed hard gate for `np=4`.
- Four heterogeneous prompts across repeated concurrent waves also produced
  varying DFlash2 hashes. Global graph-reuse disable, target-only and
  DFlash-only reuse guards, checkpoint-cache invalidation, synchronous
  hidden-state copy, and `LLAMA_DFLASH_UBATCH=8` did not restore multi-slot
  stability. The capture-specific graph-reuse guard is therefore rejected and
  removed.
- Serial target and serial DFlash2 remained stable and bit-exact for all four
  short prompts in the same artifacts. The fault boundary is concurrent
  DFlash2 verification, not basic single-slot decoding or stale serial state.
- Dedicated `LLAMA_DFLASH_UBATCH=8` probes narrowed the concurrency boundary:
  `np=1` was fully stable and bit-exact, while `np=2` already diverged for both
  heterogeneous and identical DFlash2 requests. Target-only controls remained
  stable. Investigate the first shared multi-sequence draft/verification path.
- The owned `np=1` boundary gate passed full, bit-exact target/spec output for
  `max_tokens=1,2,7,8,9,15,16` (7/7). The upstream pytest module preloader
  currently exits on its unrelated second BERT embedding preset before the
  speculative test runs, so this local-model gate is the actionable regression
  result for this environment.
- The best short-lane placement keeps the target on `Vulkan1,Vulkan0` but pins
  the 1.05 GiB DFlash2 model to `Vulkan0` with `-devd Vulkan0`. This removes
  draft pipeline parallelism. `-devd Vulkan1` is not equivalent in this build:
  the shared `output.weight` vocabulary tensor resides on `Vulkan0`, so the
  Vulkan1-only draft fails during model initialization.

## Adjacent agent-workload result

The first matched `np=1`, 4k-context, 128-output-token quick lane completed
without request errors. Both lanes used the same target, Vulkan device order,
2048/512 batch/ubatch, f16 KV, flash attention, cold-cache policy, seed, prompt
set, and greedy sampling.

| Lane | Aggregate TPS | Decode TPS | Prompt TPS |
| --- | ---: | ---: | ---: |
| target-only | 27.0917 | 29.635 | 521.59 |
| DFlash2 (`LLAMA_DFLASH_UBATCH=8`) | 36.1684 | 41.825 | 435.665 |

DFlash2 improved aggregate completion throughput by **33.5%** and decode TPS
by **41.1%** in this narrow lane. Printed draft acceptance was 29.70%
(`169/569` accepted/generated). The saved response previews were coherent but
not bit-exact between lanes, consistent with the separate #27407 numerical
divergence class; the multi-slot hard failure above is not waived by this
single-slot speedup.

Artifacts:

- `build_logs/agent-workload/vk-dflash2-target-ctx4k-20260820.*`
- `build_logs/agent-workload/vk-dflash2-spec-ctx4k-20260820.*`
- `build_logs/dflash2-lab/20260820T152948Z-vk-np4-normal-quick.*`
- `build_logs/dflash2-lab/20260820T153207Z-vk-np4-noreuse-quick.*`
- `build_logs/dflash2-lab/20260820T153416Z-vk-np4-dflash-ubatch8-quick.*`
- `build_logs/dflash2-lab/20260820T154547Z-vk-np2-dflash-ubatch8-quick.*`
- `build_logs/dflash2-lab/20260820T154708Z-vk-np1-dflash-ubatch8-quick.*`
- `build_logs/dflash2-lab/20260820T155505Z-vk-np1-boundaries-20260820.*`

## DFlash2 versus MTP head-to-head

A later sweep used the same six-request quick lane for both speculative
implementations: Qwen3.8-27B Q4_K_M target, `Vulkan1,Vulkan0`, `np=1`, ctx
4096, batch/ubatch 2048/512, f16 KV, 128 output tokens, greedy sampling, cold
cache, and three benchmark repetitions for each promoted candidate.

| Candidate | Aggregate TPS | Decode TPS | Printed acceptance |
| --- | ---: | ---: | ---: |
| MTP n=2 | 39.53 | 47.01 | 59.83% (`411/687`) |
| DFlash2 n=3, two-device draft | 36.17 | 41.83 | 29.70% (`169/569`) |
| DFlash2 n=3, `-devd Vulkan0` | **46.29** | **55.51** | 54.74% (`468/855`) |

The promoted DFlash2 configuration beats the best MTP configuration by
**17.1% aggregate TPS** and **18.1% decode TPS** in this lane. The result is
not caused by a higher acceptance percentage than MTP; the dominant win is the
cheaper single-device draft path plus the shorter optimal draft width.

Width sweeps made the optimum explicit:

- MTP n=1/2/3: `36.46 / 39.53 / 37.71` aggregate TPS.
- DFlash2 single-device n=2/3/4/5/6/7:
  `43.69 / 46.29 / 45.52 / 44.31 / 40.29 / 38.67` aggregate TPS.

The DFlash2 n=3 strict run was stable and bit-exact at `np=1`, with all eight
`max_tokens=1,2,3,7,8,9,15,16` boundary cases passing. The short-lane result
does not waive the separate `np=2` correctness failure.

Promoted artifacts:

- `build_logs/agent-workload/vk-mtp2-ctx4k-r3-20260820.*`
- `build_logs/agent-workload/vk-dflash2-n3-vk0-ctx4k-r3-20260820.*`
- `build_logs/dflash2-lab/20260820T161924Z-vk-np1-n3-devd-vk0-boundaries.*`

## Real-context distance sweep

The promoted candidates were then measured with a 32K context, 21.6K-22.0K
actual prompt tokens from the repository snapshot, f16 KV, cold full prefill,
and the `v2_write_function` agent task. Every request reached its exact output
limit. The 2048-token table is the mean of two independently started runs in
opposite ordering.

| Candidate | End-to-end TPS | Prompt TPS | Decode TPS | Wall time | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| target-only | 23.78 | 1735.34 | 27.83 | 86.14 s | - |
| MTP n=2 | 34.06 | 1719.79 | 43.13 | 60.13 s | 56.48% |
| DFlash2 n=3, `-devd Vulkan0` | **35.74** | 1040.61 | **56.18** | **57.30 s** | 60.89% |

At 2048 output tokens DFlash2 reaches **2.019x target-only decode throughput**
and **1.503x target-only end-to-end throughput**. Against the already
speculative MTP n=2 control it is **1.303x faster in decode** and **1.049x
faster end-to-end**.

The 512-token repeated lane exposes the amortization boundary: DFlash2 decode
was 52.89 TPS versus MTP's 40.70 TPS (+29.9%), but DFlash2 prompt processing
was 1061.85 versus 1723.81 TPS. Cold end-to-end throughput therefore remained
16.81 versus 20.15 TPS (-16.6%). The measured timing components predict a
rough crossover near 1,400 generated tokens for this 22K-token cold prompt;
the 2048-token run crosses it as predicted.

All long runs completed without request errors. The first 180 saved response
characters had the same hash across both repeats and all three implementations;
each implementation also produced a stable response length. The benchmark
stores only a preview, so this is a determinism/coherent-prefix signal rather
than a full semantic-equivalence claim.

Long-lane artifacts:

- `build_logs/agent-workload/vk-{mtp2,dflash2-n3-vk0}-ctx32k-out512-real-r2-20260820.*`
- `build_logs/agent-workload/vk-{target,mtp2,dflash2-n3-vk0}-ctx32k-out2048-real-r1-20260820.*`
- `build_logs/agent-workload/vk-{target,mtp2,dflash2-n3-vk0}-ctx32k-out2048-real-r1b-20260820.*`

## New tools

### Owned lifecycle + parity/stability matrix

```powershell
python scripts/research/dflash2_lab.py `
  --server-bin build-vulkan/bin/llama-server.exe `
  --model models/Qwen3.8-27B-Q4_K_M.gguf `
  --draft-model models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf `
  --devices Vulkan1,Vulkan0 --draft-devices Vulkan0 `
  --parallel 1 --ctx-size 4096 --spec-n-max 3 --max-tokens 64 `
  --boundary-tokens 1,2,3,7,8,9,15,16 `
  --require-serial-parity --require-identical-slot-stability `
  --require-boundary-parity --label vk-np1-n3-devd-vk0 `
  --env LLAMA_DFLASH_UBATCH=8
```

The tool refuses to start while another `llama-server` process exists, chooses
a free port, waits on `/health`, captures the server log, sends CTRL_BREAK to
its owned Windows process, waits for graceful exit, and writes:

```text
build_logs/dflash2-lab/<timestamp>-<label>.json
build_logs/dflash2-lab/<timestamp>-<label>.md
build_logs/dflash2-lab/<timestamp>-<label>.server.log
```

Use `--quick` for a short diagnostic. Use `--include-text` only when full output
is needed; hashes, byte counts, usage, timings, and finish reasons are retained
by default. To attach without owning a process, omit the three model/binary
arguments and pass `--url`.

Add `--boundary-tokens 1,2,7,8,9,15,16 --require-boundary-parity` to validate
speculative output accounting in the same owned-server run. Degenerate
single-token timing sentinels are excluded from decode-TPS means.

### Offline reclassification

```powershell
python scripts/research/dflash2_report.py <artifact.json> `
  --server-log <artifact.server.log> --output <report.md>
```

This allows classification rules to improve without rerunning the GPUs.

### CPU-only tool tests

```powershell
python scripts/research/test_dflash2_tools.py
```

## Decision gates

Hard failures:

- crash, HTTP error, truncated response beyond the requested boundary;
- serial target hash changes across repeats;
- serial DFlash2 hash changes across repeats;
- identical prompts diverge across simultaneous slots;
- failed graceful server teardown.

Recorded separately, not automatically fatal:

- stable speculative output differing from sequential target (#27407 class);
- heterogeneous-only concurrent hash variation when identical-slot controls
  remain stable;
- throughput below the reference model card (different hardware/backend).

Performance promotion requires an adjacent spec-none run with the same model,
backend, context, batch/ubatch, KV types, device order, prompt set, concurrency,
sampling, cache policy, and background load.

## Next commands

1. Add an opt-in full-response field or semantic judge to the benchmark; the
  current preview hash is insufficient for full quality comparison.
2. Repeat the distance sweep on a reused multi-turn chat, where the cold
  DFlash2 prefill penalty should be paid less often.
3. Run a full multi-wave `np=2` identical-prompt matrix and retain full text for
  the divergent phase; the quick probe already establishes `np=1` pass and
  `np=2` fail.
4. Instrument per-slot draft state ownership and verification batch row/seq-id
  mapping; graph reuse and DFlash ubatch are already excluded.
5. Profile and reduce DFlash encoder prefill cost on `Vulkan0`; it is now the
  dominant short-session regression. The next decode speed target is draft
  compute/synchronization rather than wider speculation.
6. Keep `np=4` DFlash2 disabled for production-style use until identical-slot
  stability passes; target-only `np=4` and DFlash2 `np=1` remain valid controls.
