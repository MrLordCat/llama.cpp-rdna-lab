# D078 — CPU Baseline & Hotspot Analysis

Date: 2026-05-30  
Owner: Copilot/perf workspace  
Branch: `research/cpu-perf`

## Platform

| Parameter | Value |
|---|---|
| CPU | AMD Ryzen 7 5800X3D, 8C/16T, 3.4 GHz base, 4.5 GHz boost |
| L3 | 96 MB 3D V-Cache (single CCX, all cores share) |
| RAM | 64 GB DDR4-3200 dual-channel (~50 GB/s) |
| Model | Qwen3.6-27B-Q3_K_S.gguf (11.5 GB, 26.9B params) |
| Build | AVX2 + FMA + F16C + REPACK |

## Baseline (llama-bench, cold)

| Test | t=1 | t=2 | t=4 | t=8 |
|---|---|---|---|---|
| tg1 (decode) | 1.04 | 1.81 | **2.54** | 2.30 |
| pp32 (prefill) | - | - | - | 7.10 |
| pp64 | - | - | - | 7.03 |

Peak decode: **2.54 tok/s @ 4 threads**. Ditches at 8 threads (2.30) — memory-bound.

## Hotspot Analysis

### Decode (tg1)

Each token generation:
1. Q3_K matvec: read ~11.5 GB weights / 27B params per layer (stride over K dimension)
2. The working set for ONE layer: ~180 MB (5120×17408×q3_K + extras)
3. RAM bandwidth: 50 GB/s. Each layer matvec reads ~180 MB → 3.6ms/layer × 64 layers = 230ms
4. Observed: 1/2.54 = 394ms/token. Matvec ~60% of decode time.

### Why > 4 threads ditches

- 8 threads all reading from RAM via 2 memory channels
- DDR4-3200 dual-channel: 51.2 GB/s theoretical, ~45 GB/s effective
- Beyond 4 threads, memory controller saturated, threads contend for bandwidth
- 3D V-Cache helps with weight reuse across layers but doesn't increase bandwidth

## Optimization Candidates (ordered by expected gain)

| # | Candidate | Mechanism | Expected gain |
|---|---|---|---|
| 1 | **Thread affinity** (OMP_PROC_BIND=close) | Pin threads to cores 0-3, one CCX | +0-5% (less L3 thrashing) |
| 2 | **Q8_0 KV cache** (f16→q8_0, saves 2x KV bandwidth) | Decode reads K/V cache; halving bytes helps memory-bound path | +5-10% decode |
| 3 | **Q3_K repacking for GEMV** (no repack currently) | Better cache line utilization, fewer TLB misses | +10-20% decode |
| 4 | **Prefetch distance tuning** | HW prefetcher may not see far enough ahead | +5-10% |
| 5 | **Batch decode** (process 2-4 tokens together) | Amortize weight reads across tokens | +30-50% decode |

## Quick Benchmark Command

```bash
# ~10 seconds, no server startup
./build-cpu/bin/llama-bench.exe -m models/Qwen3.6-27B-Q3_K_S.gguf -p 0 -n 1 -t $THREADS
```

## CPU Mini Bench (agent_workload)

Prepared short preset command (single task, short response, 30s killer):

```bash
python scripts/agent_workload_bench.py \
	--cpu-mini \
	--label cpu-mini-r1 \
	--server-extra "--spec-type none"
```

Preset behavior:
- task set: `quick` with `triage_diff`
- `runs=1`, `max_tokens<=8`, `ctx=4096`, `-ngl 0`
- `cache-type-k/v=f16`
- `--task-hard-timeout=30` (kills request and owned server on timeout)

## MTP Potential on CPU (initial probe)

Hypothesis: large host RAM can reduce pressure from KV/context residency for CPU-oriented runs, so speculative decoding may improve wall TPS if draft overhead is low.

Observed quick A/B (same mini lane, `runs=1`, `task-hard-timeout=30`):

1. `--spec-type none`: completed, wall `10.62s`, `completion_tokens=8`, `~0.75 TPS`.
2. `--spec-type mtp --spec-draft-n-max 3`: server init failed for this model.

Failure reason from server log:

```text
GGML_ASSERT(hparams.nextn_predict_layers > 0 && "QWEN35_MTP requires nextn_predict_layers > 0") failed
```

Conclusion:
- RAM capacity alone is not enough to enable MTP acceleration.
- Current `Qwen3.6-27B-Q3_K_S.gguf` is non-MTP for `--spec-type mtp` path in this runtime.
- Practical near-term speculative path on this model is `ngram-mod`/`ngram-mtp` A/B, while true MTP speedups require an MTP-enabled GGUF.

## MTP-enabled Qwen model probe (validated)

Model used: `models/Qwen3.6-27B-Q3_K_S_mtp.gguf`

Short CPU lane (same shape, cold, `task-hard-timeout=30`):

```bash
python scripts/agent_workload_bench.py \
	--server-bin build-rocm-vec/bin/llama-server.exe \
	--model models/Qwen3.6-27B-Q3_K_S_mtp.gguf \
	--gpu-layers 0 --ctx-size 4096 --batch-size 512 --ubatch-size 128 \
	--cache-type-k f16 --cache-type-v f16 --flash-attn --parallel 1 \
	--tasks quick --task-ids triage_diff --max-tokens 32 \
	--real-context-mode off --no-reuse --runs 1 \
	--startup-timeout 240 --request-timeout 30 --task-hard-timeout 30 \
	--background-server-policy fail --server-extra "--spec-type <none|mtp>"
```

Measured:

1. `spec=none`: wall `19.10s`, completion TPS `1.68`
2. `spec=mtp --spec-draft-n-max 3`: wall `14.52s`, completion TPS `2.20`

Delta on this lane:
- wall-time speedup: `19.10 / 14.52 = 1.315x` (~`+31.5%`)
- completion TPS gain: `2.20 / 1.68 = 1.31x` (~`+31%`)

Takeaway:
- For an actual MTP-enabled GGUF, MTP gives a meaningful CPU-lane speedup even with RAM-resident workload.
- The earlier no-gain result came from running a non-MTP model with `--spec-type mtp`, which cannot initialize MTP heads.

## MTP Draft Sweep (parameter tuning)

Lane: CPU (ngl=0), ctx=4096, b=512, ub=128, f16 KV, cold, max_tokens=48, `task-hard-timeout=45`.

### Default temperature (0.2)

| Config | Wall (s) | TPS | vs baseline |
|---|---|---|---|
| `spec=none` (baseline) | 24.54 | 1.96 | -- |
| `mtp d1` | 21.14 | 2.27 | +15.8% |
| `mtp d2` | 19.62 → r3: 19.25 avg | 2.45 → r3: 2.50 | +27.5% r3 |
| `mtp d4` | 21.95 | 2.19 | +11.7% |
| `mtp d6` | 23.93 | 2.01 | +2.6% |

Peak at default temp: `d2`, acceptance 77.8% (28/36 per run, 18 draft calls).

### Deterministic (temperature=0.0)

| Config | Wall (s) | TPS | vs baseline |
|---|---|---|---|
| `mtp d2 t0` | 18.97 | 2.53 | +29.1% |
| `mtp d3 t0` | 18.80 → r3: 18.22 avg | 2.55 → r3: 2.63 | **+34.7% r3** |
| `mtp d4 t0` | 21.39 | 2.24 | +14.3% |

**Best config**: `--spec-type mtp --spec-draft-n-max 3 --temperature 0.0`
- r3 confirmation: 18.24/18.30/18.12s, TPS 2.63, stdev 0.01
- Draft acceptance: 78.6% (33/42 per run, 14 draft calls x 3 tokens)
- decode_eval_tps (server-side): 4.69
- Speedup over spec=none: **+34.7%**
- Key mechanism: deterministic decoding aligns draft and target distributions → fewer draft calls (14 vs 18 for d2) even at longer window

## MTP GPU / Mixed GPU-CPU Sweep (ROCm, RX 9070 XT)

Model: `Qwen3.6-27B-Q3_K_S_mtp.gguf`. KV: q4_0/q4_0. Mini bench: quick/triage_diff, max_tokens=48, cold.

### Full matrix (wall seconds / TPS)

| ctx | ngl | none (s/TPS) | MTP d2 (s/TPS) | MTP Δ |
|---|---:|---|---:|---:|
| 4k | 999 | 1.81 / 26.49 | 1.95 / 24.58 | **−7.3%** |
| 4k | 48 | 8.07 / 5.95 | 5.90 / 8.14 | **+36.8%** |
| 4k | 32 | 13.41 / 3.58 | 9.09 / 5.28 | **+47.5%** |
| 4k | 0 | 24.54 / 1.96 | 19.25 / 2.50 | +27.5% |
| 64k | 999 | 1.81 / 26.52 | 1.96 / 24.47 | −7.7% |
| 64k | 48 | 8.01 / 5.99 | 6.05 / 7.93 | **+32.4%** |
| 130k | 999 | 1.86 / 25.75 | 1.99 / 24.16 | −6.2% |
| 130k | 32 | — | 9.51 / 5.05 | — |

### VRAM at 130k

- gpu999 (+MTP): projected 14499 MiB, free 14791 MiB → **на грани**, −732 MiB от safety margin
- KV buffer: 2304 MiB (q4_0, 16 attention layers)
- MTP draft context overhead: ~100-200 MiB estimated

### Pattern

1. **Pure GPU (999)**: MTP hurts by 6-8% — GPU compute-bound, draft overhead > benefit
2. **Mixed GPU+CPU**: MTP helps by 32-48% — CPU memory-bound layers gain from fewer forward passes
3. **Sweet spot**: gpu32 (+47.5% MTP benefit)
4. **But absolute TPS**: gpu999 none (25.8) >> gpu32 d2 (5.0) — 5x gap

### Practical interpretation

For **short-prompt decode** (mini bench):
- gpu999 none = best (25.8 TPS)
- MTP never beats pure GPU on decode-only

For **large-prompt scenarios** (60k tokens at 130k ctx):
- gpu999 VRAM usage is borderline (14499/16304 MiB, only 1.8 GB free)
- Prompt processing generates massive attention intermediates that can overflow VRAM
- When VRAM overflows → PCIe spill → effective TPS drops to ~1-3 TPS
- In that regime, gpu48/gpu32 + MTP (5-8 TPS) may OUTPERFORM a VRAM-thrashing gpu999
- **This needs verification with real large-prompt benchmark** (repo-snapshot mode, real-context-chars=152000)

### Recommendation

1. For decode-heavy workloads: gpu999, spec=none
2. For prompt-heavy workloads at 130k: test gpu48 + MTP d2 vs gpu999 none with real large prompt
3. The mixed GPU-CPU + MTP strategy is valid when VRAM pressure forces PCIe spill
4. MTP is a **memory-bandwidth amplifier**, not a universal speedup
