# D138: ROCm Q4_K MMVQ memory-interface program (decode weight stream)

Date: 2026-09-25
Status: **steps 0-7 done** - decode budget measured end-to-end (MMVQ `80%`,
FA `16%`, other `5%`); the fused-FFN share is inside MMVQ, not a separate bucket
Branch: `master`
Owner: coordinator (single GPU lane owner)

## Question

The decode weight stream is the documented decode bottleneck on ROCm
(`W12`/`W13`/`W19`). Measured efficiency differs sharply by shape:

- fused Q4_K MMVQ: `377-381 GB/s` = `59%` of the 644 GB/s physical peak, with
  100% occupancy (`W18:4-7`);
- Q6_K output head `5120x248320`: `585.3 GB/s` = `91.4%` (`G11`,
  `docs/research/rocm72/05_GGML_GAP_MAP.md:491`);
- Vulkan cross-check on the same fused Q4_K shape: `526 GB/s` of a 620 GB/s
  achievable ceiling (`D105`), i.e. the shape is the low outlier there too, but
  at a materially higher fraction than ROCm.

Why does the same silicon stream one shape at ~91% and the fused FFN Q4_K at
~59%, and can a targeted **memory-interface** change (not arithmetic) move the
Q4_K family without any quality loss?

## Frozen lane contract (D138)

Windows ROCm 7.2 (`build-rocm72`), `Qwen3.8-27B-UD-Q4_K_M.gguf`:

- `-dev ROCm1,ROCm0 -sm layer -ts 1,1`, `-ngl 999`, one slot;
- `ctx=49152` (L2) and `ctx=98304` (L3) confirmation lane;
- `batch 8192 / ubatch 1024`; KV `f8_e4m3/f8_e4m3`; flash on;
- `spec=none`; `fit off`; `cache-ram 0`; `ctx-checkpoints 0`; no-warmup;
  `seed 42`, `temp 0.2`, `top_p 0.9`;
- L2 = 31744 prompt target / 256 decode; L3 = 66560 / 256.

Command (reproduces the `win-rc72-*` lane, 2026-09-23):

```
python scripts/bench2.py --backend rocm --model models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --level 2 --runs 1 --kv-k f8_e4m3 --kv-v f8_e4m3 --run-name <label>
```

Rules kept from the workspace: adjacent interleaved A-B-A in one fresh session
(thermal drift, `bench` rule), no GPU discovery while a server runs, graceful
stop only.

## Known dead ends (do NOT repeat in this program)

Arithmetic/instruction side is measured dead on this lane:

- dot2-hoist (`-50%` DP4A, bit-identical): `+0.08%` decode = noise (`W18`);
- exact INT4 dot8: same instruction rate, exact 8-bit decomposition needs the
  same instruction count (`W18`);
- software K-stream prefetch: `-0.64%` (`W16`);
- occupancy/shared staging (C1b): `+1.10%` once, then lever exhausted (thread
  limited, bank-conflict-optimal layout) (`W13` C1b);
- cache-policy hints: NT loads `-22.8%` decode and `hipAccessPolicyWindow`
  unsupported on gfx1201 (`H80`); no prefetch ISA on gfx1201 (`W17`);
- Q5_K/Q6_K N=1 body probes x5: neutral/negative (`G11`);
- pair-streaming (Q3_K technique) applied to Q5_K: `-0.4%` (`G11`);
- GDN two-output fusion: launch/latency-bound, no wall win (`G10`).

## Steps

**Step 0 - baseline (this session).** Fresh adjacent control on L2 after the
binary is rebuilt to `master` (`fea1c3180`, the cleanup touched `mmvq.cu`).
Artifacts and numbers: below.

**Step 1 - per-shape effective-bandwidth census (no code change).** Use the
in-fork traces (`GGML_TRACE_MMVQ_TIMING`, `GGML_TRACE_MMVQ_RESOURCES`) to rank
decode shapes by effective GB/s (weight bytes / kernel time) and identify
whether the fused gate+up / down / lm_head split matches `W19`. Purpose: pick
the exact shape the candidate must move, and give the geometry gate a
per-shape metric besides wall TPS.

**Step 2 - Candidate A: Q4_K geometry re-check (C1).** `small_k` off for Q4_K
on RDNA4 means 1 row per CTA with 8 warps covering the row (the Q6_K-like
geometry) instead of the current 8 rows per CTA / one row per warp. Prior
evidence (`W13` C1, 2026-08-15): 49K `+2.7-4.1%` (3/3 interleaved pairs),
98K noise -> not promoted, no env gate remains in tree. Needs a temporary
env gate for a same-binary A-B-A.

**Step 3 - Candidate B: per-thread K-window widening** to full 128 B L2 line
loads (`W18:97-110`), admitted only if step 1/2 show request-size waste.

### Step 3 admission test: the current pattern is already at HBM peak

`scripts/research/d138_vecdot_probe.hip` (built with the exact production
type/vec_dot sources: `-I ggml/src/ggml-cuda`, `-x hip --offload-arch=gfx1201`,
same `-O3 -ffast-math`; the kernel loop is `mul_mat_vec_q`'s non-fused
`ncols_dst == 1` RDNA4 shape, verbatim). Forms and `N`/`K` from the byte
census, buffers sized >300 MB so the 64 MB L2 cannot flatter the result, plus
an L2 churn between reps. `d138-vecdot-probe.exe`:

| form | bytes | ms | GB/s | vs HBM peak |
| --- | ---: | ---: | ---: | ---: |
| q5_K K=5120 N=87040 | 306.4 MB | `0.464` | **`660.9`** | 98% |
| q5_K K=17408 N=25600 | 306.4 MB | `0.463` | **`662.2`** | 98% |
| iq4_xs K=5120 N=87040 | 236.8 MB | `0.368` | **`643.2`** | 96% |
| iq4_xs K=17408 N=25600 | 236.8 MB | `0.359` | **`660.4`** | 98% |
| q6_K K=5120 N=248320 | 1042.9 MB | `1.636` | `637.7` | 95% |
| stream reference, 16 B loads | 268.4 MB | `0.398` | `674.3` | 100% |

Independent cross-check of the step-2b fit: the card's read peak is
`~665-675 GB/s`, measured two ways (a 1 GB q6_K form and a pure 16 B linear
stream), i.e. the `623.8 GB/s` in the fit was a couple of percent pessimistic
and **not** an artifact.

Interpretation: the production `vec_dot` memory pattern - the 4 B `ql`/`qh`
loads, the 2 B scales, the 22 B of weight per call - reaches 96-98% of HBM peak
as soon as the grid is saturated, for both the short-K (20 blocks/row) and the
long-K (68 blocks/row) hot shapes. **Step 3 is rejected: there is no
request-size waste left to recover.** The 364 GB/s effective rate inside the
decode is therefore a *scheduling* result, not a `vec_dot` result.

Second measurement from the same probe, `run_split_test` (same 306 MB read as
1..458 launches, one stream, dependent order):

| launches | rows/launch | ms | GB/s | vs 1 launch |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 87040 | `0.452` | `677.5` | - |
| 2 | 43520 | `0.476` | `644.1` | `-4.9%` |
| 8 | 10880 | `0.497` | `616.8` | `-9.0%` |
| 32 | 2720 | `0.565` | `541.9` | `-20%` |
| 128 | 680 | `0.688` | `445.5` | `-34%` |
| 458 | 190 | `1.080` | `283.7` | `-58%` |

-> `~1.4-1.9 us` of fixed cost per launch (launch + wave ramp + drain).
Applied to the real decode this is only `0.6-0.9 ms/token` for the 458 MMVQ
launches, because the decode's launches are large (10.53 GB / 458 = `23 MB`
average, and the fragmentation loss is a function of bytes-per-launch, not of
the launch count alone). Note the coincidence worth remembering: taking the
split test at face value for 458 equal-size launches reproduces the census
number (`283.7` vs the `~284 GB/s` class), which is exactly the trap - the
real decode's launches are ~34x larger than that test's.

**Step 4 - DEFERRED (blocked on host):** `GL2C_EA_RDREQ_{32,64,128,256}B` +
`GL2C_HIT/MISS` request-size PMC profile (`W19:111`). `rocprofv3` is a Linux
ROCm 10 tool; ROCProfiler on native Windows is a documented closed item
(`05_GGML_GAP_MAP.md`, "Closed or rejected without a material change"). Run it
on the Linux ROCm 10 host when that lane is free.

## Gates

- Candidate must clear decode `>= +2.0%` on L2 against an adjacent interleaved
  control, with the same sign at L3 (98K), prefill within `+/-1%` and a
  byte-identical greedy smoke;
- one changed mechanism per candidate; temporary gates reverted unless promoted;
- negative results are documented and reverted, not left enabled.

## Baseline (step 0)

Binary rebuilt to `master` `fea1c3180` first (`build-rocm72/bin/llama-server.exe`
25-Sep 14:19; the cleanup commit touched `ggml/src/ggml-cuda/mmvq.cu`).

| lane | prefill tok/s | decode tok/s | aggregate TPS | ttft ms | run dir |
| --- | ---: | ---: | ---: | ---: | --- |
| L2 `ctx=49152` (30609) | `1852.9565` | `24.1019` | `9.4324` | `16519` | `d138-base-q4k-l2--20260925T112226805832Z` |
| L3 `ctx=98304` (64287) | `1528.7130` | `20.9304` | `4.7159` | `42053` | `d138-base-q4k-l3--20260925T112329140435Z` |

Adjacent 2026-09-23 session on the same lane/model: L2 `1819.4558/23.9991`,
L3 `1525.661/20.915`. The rebuilt binary reproduces the lane within
`+0.07%` decode at 98K, so the baseline is valid after the cleanup commits.

## Step 1 - per-shape census (done)

Method: `GGML_TRACE_MMVQ_TIMING=1 GGML_TRACE_MMVQ_TIMING_SYNC=1
GGML_TRACE_MMVQ_RESOURCES=1` (+ `GGML_HIP_DISABLE_GRAPHS=1` for full coverage),
same L2 lane, 124968 parsed launch records. Parser:
`scripts/research/d138_census.py` (working tool; it moves to `scripts/research/`
when the program closes).

Share of measured MMVQ kernel time (256 decode tokens + prompt pass, isolated
launch timing):

| type | share | calls | weights read | isolated GB/s |
| --- | ---: | ---: | ---: | ---: |
| q5_K | `33.1%` | 33073 | 1161 GB | 163 |
| iq4_xs | `22.3%` | 26307 | 1034 GB | 215 |
| q4_K | `20.6%` | 26563 | 878 GB | 198 |
| q8_0 | `12.6%` | 28184 | 19 GB | 7 |
| q6_K | `7.5%` | 6510 | 402 GB | 247 |
| iq4_nl / q3_K / iq3_s | `3.9%` | 4331 | 138 GB | ~230 |
| **total** | `21.5 s` | 186k | `3.63 TB` | `169` |

Key findings:

1. **`GGML_HIP_DISABLE_GRAPHS=1` costs `2.3x` decode** (`24.10 -> 10.18 tok/s`):
   HIP graphs are a first-order decode lever on this lane, not a detail.
   Prompt eval is unaffected (`1853 -> 1842 tok/s`).
2. `lm_head` `q6_K 5120x248320` reproduces `576-580 GB/s` = `~90%` of the
   644 GB/s physical peak (`1.811 ms` per call, 248320 CTAs) - the only shape
   at the ceiling.
3. Every FFN/attention shape streams at `44-338 GB/s` *when measured in
   isolation*. On the same `17408x5120` shape, `q4_K` (small_k, 8 rows/CTA)
   measures `277 GB/s` and `q5_K` (1 row/CTA) `303 GB/s`. See step 2b: most of
   this gap is the fixed per-launch part, not steady-state bandwidth.
4. **`q8_0` micro-kernels are `12.6%` of MMVQ time for `0.5%` of bytes**:
   28184 launches, `grid=48`, `K=5120`, `93 us` each (`7 GB/s`). This launch
   class is the largest per-byte cost on the lane and is not addressed by any
   `vec_dot` change.
5. Two shapes time at `0.497 ms` regardless of type (`q5_K` and `q6_K`,
   `fus=1`, `K=6144`, `grid=5120`). Resolved in step 2b: the group is bimodal
   and the slow mode is the tail of the preceding attention kernel attributed
   by `TIMING_SYNC`, not a fused-path defect.
6. Early fit from the isolated timings was `t ~ 85-90 us + bytes / ~530 GB/s`;
   the better-conditioned fit after step 2b is `134 us + bytes / 623.8 GB/s`
   (step 2b). Wall-level conclusions stay on the interleaved A-B-A lane, never
   on this table.

Upshot for candidate selection: the UD-Q4_K_M primary model is **not** a
Q4_K-dominated lane (`q4_K` is `20.6%`), and the documented `Q4_K` geometry
question is now directly measurable per shape. `q5_K` and `iq4_xs` are larger
shares, but both already use the `1 row/CTA` geometry, so the geometry lever
applies only to `q4_K` (and `q3_K`, out of scope here).

## Step 2 - Candidate A: Q4_K 1-row/CTA geometry (REJECTED)

Probe gate (default off, kept for reproducibility):
`GGML_MMVQ_RDNA4_Q4K_ROWS1=1` forces `use = false` for `Q4_K` on the RDNA4
Qwen-hot decode path (`ggml/src/ggml-cuda/mmvq.cu`, 14 added lines).

Gate validated by trace: with the gate set, `type=12/q4_K ncols_dst=1
nwarps=8 blocks_per_row=20 small_k=0` (i.e. 1 row/CTA, 8 warps/row), versus
`small_k=1` without it. Same binary for both arms.

Interleaved A-B-A, L2 lane, same session:

| arm | decode tok/s | prefill tok/s | aggregate TPS |
| --- | ---: | ---: | ---: |
| A1 control | `24.0655` | 1848.24 | 9.4122 |
| B1 `ROWS1=1` | `24.0167` | 1847.54 | 9.4025 |
| A2 control | `24.0796` | 1847.80 | 9.4130 |

Control mean `24.0726`, spread `0.06%`; candidate is `-0.23%` -> **no gain,
rejected**. The 2026-08-15 `W13` C1 result (`+2.7-4.1%` at 49K) does not
reproduce on the current model/lane: `UD-Q4_K_M` spends only `20.6%` of MMVQ
time in `q4_K`, so even a perfect `q4_K` fix is bounded at ~2% wall here, and
the geometry change itself is neutral-to-negative.

### Byte-exact census: which types the decode actually reads (2026-09-25, supersedes the time shares below)

The per-shape times above come from `TIMING_SYNC` and are polluted (the sync
wait includes queueing and the tail of the previous kernel), so the byte side
was re-measured separately: one L2 run with `GGML_TRACE_MMVQ_TIMING=1` **and
`GGML_HIP_DISABLE_GRAPHS=1`** (outside graph capture every `mul_mat_vec_q`
launch is traced), then the grid sizes multiplied by the per-block byte cost
(`build_logs/tmp_d138_bytes_full.py`). 124482 traced launches, 272 decode
tokens (`d138-bytes-l2`):

| type | fused | GB/token | calls/token |
| --- | ---: | ---: | ---: |
| q5_K | 0 | `2.151` | 80 |
| q5_K | 1 | `2.100` | 41 |
| iq4_xs | 1 | `2.085` | 45 |
| iq4_xs | 0 | `1.700` | 52 |
| q6_K | 0 | `1.300` | 19 |
| q4_K | 0 | `0.309` | 82 |
| q4_K | 1 | `0.093` | 15 |
| iq4_nl / iq3_s / q3_K / q8_0 | both | `0.517` | 125 |
| **total** | | **`10.53`** | **458** |

With `MMVQ = 28.9 ms/token` (step 7 ablation) this is an effective
`364 GB/s` across both GPUs for the whole MMVQ phase, i.e. `~58%` of the
`624 GB/s` single-GPU ceiling of the step-2b fit.

**This changes the target.** The Q4_K MMVQ line (W18, 2026-08) was chasing a
type that is `4%` of decode bytes; the mass is `q5_K` (`40%`) and `iq4_xs`
(`36%`). Two geometry checks were run on the real mass, from the same binary:

- **C1 (Q4_K `small_k` on/off, trace-verified 588 launches flipped):**
  `24.72` vs `24.80` t/s, i.e. `0%`. Same result as the 08-15 A-B-A.
- **C2 (invert `should_use_small_k` for `q5_K` + `iq4_xs`):** control `25.77`
  / `25.85`, inverted `23.55` -> **`-8.6%`, rejected**. The existing geometry
  policy is not a leftover, it is the correct choice: 8 rows/CTA for `q5_K`
  non-fused, 1 row/CTA for `iq4_xs`/fused shapes both beat the alternative.

So the layout axis is exhausted for both the old target and the real one. What
remains is the byte path inside `vec_dot` (per-thread K window, narrow 1-byte
loads) and the launch/overlap structure - and the `DUP_ALL` probes show the
ceiling is real: duplicating the whole non-fused class (`+5.72 GB/token`) costs
`+12.37 ms`, duplicating fused on top (`+4.81 GB/token`) another `+10.66 ms`,
i.e. the extra pass moves `~457 GB/s` - the fused/non-fused classes are
bandwidth-limited, not fixed-cost-limited.

Probes kept for future A/B (all default-off): `GGML_MMVQ_RDNA4_Q4K_ROWS1`,
`GGML_MMVQ_HOT_ROWS1_INVERT` (see `mmvq.cu`), plus the older
`GGML_TRACE_MMVQ_SMALL_K` trace used above to prove the flip happened.

## Step 2b - What the census implies (supersedes the "low bandwidth" framing)

Least-squares fit over the per-shape medians (`build_logs/tmp_d138_fit.py`,
40 groups):

```
t_ms = 134 us + bytes / 623.8 GB/s
```

- The `lm_head` point lands on the fit within `0.3%` (`1.806` predicted vs
  `1.811` measured), and most large FFN shapes land within `15-25%` on the
  optimistic side. **The MMVQ family already reaches ~97% of the 644 GB/s
  physical peak once a shape is long enough to amortize the fixed part.**
- The fixed part (~`134 us` per launch) dominates short shapes: decode executes
  `~727 MMVQ launches per token` (186k launches / 256 tokens). With graphs the
  whole token still costs `42 ms`, i.e. `~58 us` per launch amortized - so
  launches, not steady-state bandwidth, set the decode bill.
- Methodological limit: `TIMING_SYNC` attributes the tail of the *previous*
  non-MMVQ kernel (e.g. `FLASH_ATTN_EXT`) to the next MMVQ launch. The fused
  `K=6144` group is bimodal (`0.14-0.19 ms` early, `~0.50 ms` in steady state,
  2414/2700 samples) precisely because that launch follows attention. Absolute
  per-shape timings from this method are therefore an upper bound; shares and
  structural conclusions (grid size -> achievable bandwidth, per-type split)
  are the usable output.

New picture: the documented `Q4_K = 377-381 GB/s = 59%` (`W18`) is a
*lane-average over a launch-granular decode*, not a kernel-ceiling failure.
The remaining levers are launch granularity / overlap and fusion, and the
`q8_0` micro-launch class (`12.6%` of MMVQ time, `0.5%` of bytes), not the
`vec_dot` instruction stream or the L2 line size.

## Step 3 - Candidate B: `q8_0` micro class (RESOLVED, no win)

The `q8_0` micro class (`grid=(48,1,1)`, `K=5120`, 96 launches/token) is
`blk.*.ssm_alpha.weight` + `blk.*.ssm_beta.weight`, Q8_0 `[5120, 48]`, 48 SSM
layers (`build_logs/tmp_d138_gguf_scan.py`). The G10 fusion route exists but
only handles **f32** `ssm_alpha/beta` (`runtime_graph.inc`, `GGML_ROCM_GDN_PAIR`),
so it cannot cover this q8_0 pair; it was also already rejected (wall `-0.4%`).

Measured with a bit-identical duplicate probe (`GGML_MMVQ_MICRO_DUP=N`; for
`ncols_dst == 1` the kernel assigns `dst`, so repeats change no result):

| arm | decode tok/s | ms/token | delta vs dup=0 |
| --- | ---: | ---: | ---: |
| dup=0 | `24.0111` | `41.647` | - |
| dup=1 (96 extra launches/token) | `23.9652` | `41.727` | `+0.080 ms` |
| dup=10 (960 extra launches/token) | `23.2041` | `43.096` | `+1.449 ms` |

Warmup deltas (96 extra launches) are cleaner: `+0.0642 ms` (dup=1) /
`+0.370 ms` (dup=10) per token -> `0.4-0.7 us` per micro launch, with visible
overlap between consecutive duplicates. **The whole class costs
`<= 0.07 ms/token` (`0.15%` of decode).** The `12.6%` share seen in step 1 was
pure `TIMING_SYNC` attribution; there is nothing to optimize here.

## Step 4 - Launch granularity and the real decode split

Graph-level traces (`GGML_TRACE_CUDA_GRAPH_HOST_TIMING` +
`GGML_TRACE_CUDA_GRAPH_DEVICE_TIMING`, graph mode on, L2 lane,
`build_logs/tmp_d138_graphs.py`):

| metric | value |
| --- | ---: |
| ggml graphs per decode token | `2` (1890 and 1766 nodes) |
| GPU ms per token inside graphs | `37.2` |
| host ms per token inside `graph_compute` | `1.55` (`0.79 + 0.76`) |
| wall ms per token | `42.3` |
| outside-graph remainder | `~5.1` |

So launch granularity is **not** the limiter: two `cudaGraphLaunch` calls per
token, `1.55 ms` host cost (`3.7%`). The earlier "727 launches x 58-77 us"
framing was a `TIMING_SYNC` artifact.

Full-MMVQ pricing via duplicate probes (`GGML_MMVQ_DUP_ALL=1`, which now also
repeats the fused launches; fused kernels assign `dst`, so repeats are
bit-identical):

| arm | decode tok/s | ms/token | delta |
| --- | ---: | ---: | ---: |
| control | `24.0111` | `41.647` | - |
| non-fused duplicated (`8.46 GB/token`) | `18.5108` | `54.023` | `+12.376 ms` |
| all MMVQ duplicated (`14.33 GB/token`) | `15.4568` | `64.696` | `+23.049 ms` |

Derived effective bandwidth inside the captured graph:

- non-fused MMVQ: `8.46 GB / 12.38 ms` = **684 GB/s** (above the 644 GB/s
  "physical peak" reference; duplicates overlap slightly, so read this as
  "at the ceiling");
- fused MMVQ: `5.87 GB / 10.67 ms` = **550 GB/s**;
- total MMVQ: `14.33 GB / 23.05 ms` = **622 GB/s**, i.e. **55% of the decode
  token**; the other `18.6 ms` (`45%`) is everything else (FA / GDN / norms /
  rope / sampling / residual GPU idle).

New picture for the next iteration:

1. the fused MMVQ path (`5.87 GB/token`, `10.7 ms`, `26%` of decode) is the
   only confirmed headroom *inside* MMVQ: closing its 550 -> 684 GB/s gap is
   worth about `2.1 ms/token` (`+5%` decode);
2. ~~the `18.6 ms` non-MMVQ half of the decode is still unmeasured at kernel
   level~~ - done in step 5: `15.6 ms` MMVQ + `5.7 ms` flash attention +
   `14.7 ms` other work, i.e. the non-MMVQ half is roughly even between
   attention and a long tail of small/fused nodes;
3. host-side and launch overhead are done (`~1.5-5 ms`, not worth more work).

Recommended next step (needs a decision, not yet started): the `14.7 ms`
"other node work" bucket is the largest single slice nobody has attacked, and
its nodes are already half-merged by the fork's fusion calls, so the useful
work there is (a) per-node GPU timing inside the captured graphs, or (b)
schema-level merging (for example RS-state `GET_ROWS`/`CPY`/`CONT` chains and
per-layer norm pairs). Item 1 (`fused`, `+5%`) is the smaller and harder of the
two: its two-tensor read and GLU epilogue are already tuned (W13 C1/C1b) and
the pair-streaming idea was already measured dead for Q5_K (`-0.4%`, `G11`).

## Step 5: naming the non-MMVQ half (ablation + device timing)

Per-node event timers inside the captured HIP graph were tried first and are
**not usable**: recording `hipEventRecord` during stream capture made the
driver fail (`invalid resource handle`, `GET_ROWS failed`, server died on the
warmup request; see run `d138-nodegpu-l1`). The probe was reverted.

Working tool instead (both default off, kept only while D138 is open):

- `GGML_CUDA_ABLATE=<op>[,<op>...]` in `runtime_compute.inc`: skips executing a
  node whose `ggml_op_name` equals a pattern (exact match) - the graph keeps its
  shape, the node keeps its stale contents, so only timing is meaningful;
- `GGML_TRACE_CUDA_GRAPH_DEVICE_TIMING=1`: existing fork gate, GPU ms per graph
  replay (`slot.use`, `slot.update` also streamed).

Cost of one class = `device_ms(control) - device_ms(ablated)`, tail medians.

### Methodology warning (cost us one wrong lead)

The first matcher used `strstr` on the op name, so `MUL` also matched
`MUL_MAT`. "Skipping elementwise work gives `+76%` decode" was that artefact:
it was silently deleting every matrix multiply. The matcher now compares op
names exactly (`MUL` no longer matches `MUL_MAT`). Any future ablation pattern
must be re-checked against the op-name table before the numbers are trusted.

Second limitation, measured with the corrected matcher: ablation only reaches
nodes that go through `ggml_cuda_compute_forward`. Nodes consumed by the
fork's built-in fusion calls in `runtime_graph.inc`
(`ggml_cuda_op_rms_norm_fused`, `ggml_cuda_op_ssm_conv`, `ggml_cuda_op_unary_mul`,
`ggml_cuda_op_softcap`, ...) never see it, and their ablations return exactly
`0.00 ms` (RMS_NORM, L2_NORM, ROPE, ADD, UNARY, SSM_CONV, GLU, SCALE, CONCAT,
GATED_DELTA_NET all measured `0.00 ms` deltas). `MUL_MAT`/`MUL_MAT_ID` and
`FLASH_ATTN_EXT` are not fused here, so those two are real numbers.

### Result: decode GPU budget in graph mode

`GGML_TRACE_CUDA_GRAPH_DEVICE_TIMING`, graph replay medians, `spec=none`:

| lane (level) | control dev0/dev1 | MMVQ ablated | FA ablated | non-MMVQ = MMVQ - control |
| --- | ---: | ---: | ---: | ---: |
| L0 `ctx=8192` | `16.30 / 14.85` = **31.16 ms** | `8.25 / 7.66` = `15.92` | n/a | **15.24 ms** |
| L2 `ctx=49152` | `18.77 / 17.25` = **36.02 ms** | `10.78 / 9.66` = `20.44` | `15.86 / 14.48` = `30.34` | **15.58 ms** |

Fusion price on the same lane (`GGML_CUDA_DISABLE_FUSION=1`):
L2 device `36.02 -> 35.45 ms` (`-0.57 ms`, fusion is worth `+0.6 ms` of GPU
time) and wall decode `23.65 -> 23.25 tok/s` (`-1.7%`); L0 device
`31.16 -> 30.84 ms`. Fusion is a net win on both lanes.

Wall-clock cross-check on L2: MMVQ ablation moves decode `23.65 -> 39.86 tok/s`
(`+68%`, `+17.6 ms/token`), FA ablation `23.65 -> 25.84 tok/s` (`+9.3%`,
`+3.6 ms/token`).

So the `18.6 ms` "everything else" from step 4 is now split, per token:

- **MMVQ: `15.6 ms` (`43%` of the 36.0 ms decode graph)**;  <- **WRONG, see step 7**
- **flash attention: `5.7 ms` (`16%`)**;
- **other node work: `14.7 ms` (`41%`)** - `SUPERSEDED` by step 7: this bucket
  was almost entirely the *fused* FFN MMVQ launches, which the op-name
  ablation could not reach before the `try_fuse` hook (step 7). The real
  "other" bucket is `1.7 ms`.

The "other" bucket is `1766 - 922 = 844` nodes per graph (`~17-24 us` per node
across both devices, L2), i.e. it is not one hot kernel but a long chain of
small kernels, a large share of which is already merged by the fork's fusion
calls. Verified not the cause: sampling (`SOFT_MAX,ARGSORT` ablation
`25.53 vs 26.06 tok/s` control at L0), norms+rope (`23.92 vs 24.02` at L2).

### Fusion is a win, not a leak (step 6, pre-answer)

`GGML_CUDA_DISABLE_FUSION=1` A/B: L0 device `31.16 -> 30.84 ms` (fusion worth
`0.32 ms`), L2 decode `24.02 -> 23.77 tok/s` (`-1.05%`). The fused path is
`550 GB/s` vs `684 GB/s` for plain MMVQ *because it streams two weight tensors
(up + gate) and runs the GLU epilogue*, not because of a dispatch defect: the
fused node still wins against two separate launches that would additionally
write and re-read the `f16`/`f32` intermediate activation. Any "fix" for the
fused path has to beat that, i.e. it must make the two-tensor read stream at
close to `684 GB/s`; the honest ceiling is the step-4 number, `~2.1 ms/token`
(`+5%` decode), and the cheap levers inside it are already spent (W13 C1/C1b,
`small_k`, rows-per-block).

## Step 7 - corrected decode budget: MMVQ is 80-91%, not 43%

Step 5 could not ablate the *fused* MMVQ launches: the fork's fusion sites
(`ggml_cuda_op_mul_mat_vec_q_fusion` and friends) call the kernels straight from
`runtime_graph.inc` and never pass through `ggml_cuda_compute_forward`, so
`GGML_CUDA_ABLATE=MUL_MAT` silently missed the up+gate+GLU site (the single
largest weight consumer, `5.87 GB/token`). Fix: a hook at the top of
`ggml_cuda_try_fuse` now refuses a fusion whose head node matches the ablation
pattern, which lets the head node fall through to `compute_forward` and be
ablated. With that hook (rebuilt `build-rocm72`):

| ablation | L0 `ctx=8192` (dev0/dev1/sum) | L2 `ctx=49152` (dev0/dev1/sum) |
| --- | ---: | ---: |
| control | `16.30 / 14.85 / 31.16 ms` | `18.77 / 17.25 / 36.02 ms` |
| `MUL_MAT` (all, incl. fused) | `1.38 / 1.29 / 2.67` | `3.67 / 3.42 / 7.09` |
| `FLASH_ATTN_EXT` | - | `15.86 / 14.48 / 30.34` |
| `MUL_MAT,FLASH_ATTN_EXT` | `0.87 / 0.87 / 1.73` | `0.85 / 0.83 / 1.67` |

Budget per decode token (graph replay, `spec=none`, tail medians):

| bucket | L0 ms (%) | L2 ms (%) |
| --- | ---: | ---: |
| **MMVQ** (`MUL_MAT`/`MUL_MAT_ID`, fused + plain) | `28.5` (`91.4%`) | **`28.9` (`80.3%`)** |
| **flash attention** | `0.9` (`3.0%`) | **`5.7` (`15.8%`)** |
| **everything else** (norms, rope, GDN, cpy/concat, sampling) | `1.7` (`5.6%`) | `1.7` (`4.6%`) |
| wall-clock decode | `37.2 ms` (`26.9 t/s`) | `42.3 ms` (`23.7 t/s`) |

The combined ablation (`1.67 ms` at L2) closes the balance within `0.3 ms`
(`28.9 + 5.7 + 1.4 = 36.0`), so the budget is measured, not inferred.

Consequences:

1. the step-5 "`14.7 ms` of small nodes" lead is **dead** - the small-node
   bucket is `1.7 ms`; no schema-level work on `GET_ROWS`/`CPY`/`CONT` chains
   can pay off;
2. the decode is a two-kernel-class problem: **MMVQ `80%` + FA `16%`**;
3. the fused FFN site (`5.87 GB`, `~10.7-12 ms`) is `37-42%` of the MMVQ
   bucket at L0 - larger than the step-4 estimate suggested, because step 4
   priced it against the *added-traffic* slope, while the ablation prices the
   whole launch.

### Group ablations inside MMVQ: unreliable, documented here so nobody repeats it

Splitting the MMVQ bucket by tensor name (`z`, `Kcur`, `Vcur`, `Qcur_full`,
`attn_output`, `linear_attn_out`, `ffn_out`) mostly does **not** work:

- small groups report plausible-looking but noisy deltas (`z` `1.76 ms`,
  `linear_attn_out` `1.77 ms`, `Qcur_full` `1.01 ms`, `attn_output` `0.60 ms`,
  `Kcur,Vcur` `0.3 ms`);
- large groups **inflate** the measured time (`ffn_out`: L0 `31.16 -> 28.40 ms`,
  but `dev0 16.30 -> 21.58 ms`; `ffn_up,ffn_gate`: L2 `dev0` bimodal
  `11.1 / 22.5 ms`), and wall decode *drops* (`26.9 -> 16.5 t/s`). Blanking
  the FFN leaves non-finite activations that flow into every later node, so
  the measurement stops being additive; the same failure mode made the earlier
  `strstr`-on-`MUL` lead in step 5 look like `+76%`.
Rule: only whole-class ablations (`MUL_MAT`, `FLASH_ATTN_EXT`) and the
single-node sampling probes are trustworthy; treat name-group deltas as hints
that must be re-measured with a live value.

### FA is where the f8-KV tax sits (measured)

Same lane, `--kv-k f16 --kv-v f16` instead of `f8_e4m3`:

| arm | dev0/dev1/sum | decode tok/s | FA ablation delta |
| --- | ---: | ---: | ---: |
| f8 control | `18.77 / 17.25 / 36.02` | `23.65` | `5.68 ms` |
| f16 control | `18.08 / 16.62 / 34.70` | `24.27` | `4.37 ms` |

The entire `1.32 ms` graph difference is the FA bucket (`1.31 ms`): f8-KV
flash attention is `30%` more expensive than f16 on this lane even though it
streams half the bytes, i.e. it is not bandwidth-bound; the f8 path pays for
in-kernel f8 handling (see `fattn.cu`, `rocm_f8_types_supported` +
`ggml_get_to_fp16_cuda`). Prefill moves the other way (`1825` f16 vs `1909`
f8 tok/s, `-4.4%`), so f16-KV is a decode/prefill trade, not a free win.

### Which FA kernel runs (traced, no guesswork)

`GGML_TRACE_FATTN_PATH=1`, L0 f8 lane, `1664` dispatches, all identical:

```
D099 F8 cc=16781825 rdna4=1 reference=0 native_kq=1 native_v=1 K_type=f8_e4m3 V_type=f8_e4m3
```

So the lane uses the **D099 native RDNA4 f8 WMMA body**
(`BEST_FATTN_KERNEL_WMMA_F16`, `cols_per_block=16`, fp32 accumulator,
`fattn-wmma-f16.cu:1077-1095`), *not* the portable tile + f16-conversion path.
The `1.31 ms` f8 tax is therefore inside the native body itself (raw f8 KQ/PV
fragments), not a conversion pass, and it is bisectable at runtime:

- `GGML_ROCM_FATTN_F8_NATIVE_KQ=0` - master off -> portable tile path with
  shared converted f16 allocation;
- `GGML_ROCM_FATTN_F8_NATIVE_V=0` - P*V phase back to portable f16;
- `GGML_ROCM_FATTN_F8_REFERENCE=1` - forced portable reference;
- `GGML_ROCM_FATTN_Q8_V_DIRECT_WMMA` - separate q8-V direct path.

That is the next measurement set (see handoff note below).

### History: this lane used to go through f16 conversion

`git log -S GGML_ROCM_FATTN_F8_NATIVE_KQ` (checked 2026-09-25):

- before `d0a3a6fa1` (D098 G3a) there was **no native f8 FA at all**: f8 K/V
  was converted to a shared f16 allocation (`ggml_get_to_fp16_cuda`) and run
  through the portable tile body;
- `d0a3a6fa1` (KQ) and `415ba6277` (PV) added the native RDNA4 fp8 WMMA bodies
  as **opt-in** (`...=1`, default off);
- `e58227730` ("perf(rocm): promote native full FP8 attention", 2026-08-13)
  inverted both gates to opt-out, so native f8 has been the RDNA4 default
  since then. Its evidence was a same-binary 49K spec-none bracket of
  **q8-KV vs f8-KV** (`1686.75/21.83` vs `1769.37/22.97`,
  `+4.9%/+5.2%`) plus MTP acceptance (`83.16%` vs `77.78%`), and it removed
  `196` VGPR spills + `788 B/thread` scratch.

So today's `5.68 ms` f8 FA is the *promoted native* body, not a conversion
pass, and the pre-promotion portable path is still reachable at runtime via
`GGML_ROCM_FATTN_F8_NATIVE_KQ=0`. Note the promotion bracket never compared
f8 against f16 KV; today's measurement (`4.37 ms` at f16) does, and that is
the gap worth re-testing.

### A-B-A: portable f8 path is not a candidate (measured 2026-09-25,
background game)

L2 lane, f8 KV, one run per arm, background game active on the display GPU
(operator-approved window; the load is recorded because the lane contract
requires it):

| arm | prefill tok/s | decode tok/s | dev0/dev1/sum device ms |
| --- | ---: | ---: | ---: |
| A1 control (native f8) | `1740.5` | `20.15` | `18.70 / 24.51 / 43.20` |
| B1 `GGML_ROCM_FATTN_F8_NATIVE_KQ=0` | `1662.5` | **`14.95`** | `25.53 / 34.43 / 59.96` |
| A2 control | `1831.1` | `20.76` | `18.74 / 22.60 / 41.34` |

Verdict: the portable path is `-27%` decode (`14.95` vs control mean `20.46`)
cold and `-51%` prefill, and it inflates the graph by `+45%`. The
`e58227730` promotion was correct; rolling it back is not an option, and the
`1.31 ms` f8-vs-f16 gap from the previous section is therefore a property of
the *native* f8 WMMA body, not something a conversion pass can recover.

### Background-game cost on this lane (same session)

Game/desktop load on the display GPU, compared against the game-free runs of
the same lane (`d138-l2gpu-ctrl`, `36.02 ms`, `23.65 tok/s`):

| metric | game-free | with game | delta |
| --- | ---: | ---: | ---: |
| decode tok/s | `23.65` | `20.15 / 20.76` | **`-13.5%`** |
| prefill tok/s | `1909` | `1740 / 1831` | `-4..9%` |
| device ms, ggml dev0 (free GPU) | `18.77` | `18.70 / 18.74` | `0%` |
| device ms, ggml dev1 (display GPU) | `17.25` | `22.60 / 24.51` | **`+31..42%`** |

The load lands almost entirely on the display device (`ROCm0` = ggml dev1
under `-dev ROCm1,ROCm0`); the compute-first device is unaffected. Any future
lane comparison must therefore be A-B-A inside one session, which is what the
workspace rule already requires.

### Phase bisect inside the native f8 body (A-B-A-C, 2026-09-25, game active)

L2 lane, single run per arm, background game (all four arms under the same
load, so the ordering is A-B-A-C and the control repeats bracket the result):

| arm | prefill tok/s | decode tok/s | dev0/dev1/sum device ms |
| --- | ---: | ---: | ---: |
| A1 f8 control (native KQ+V) | `1762.5` | `22.49` | `18.67 / 18.40 / 37.07` |
| B1 f8, `GGML_ROCM_FATTN_F8_NATIVE_V=0` | `1497.5` | `19.50` | `22.34 / 22.25 / 44.59` |
| A2 f8 control | `1799.7` | `23.00` | `18.61 / 18.38 / 36.98` |
| C1 f16 KV | `1727.5` | `20.71` | `18.06 / 23.57 / 41.63` |

Readings:

1. **Native PV is not the problem.** Disabling it costs `-14%` decode
   (`19.50` vs control mean `22.75`), `-16%` prefill and `+21%` graph time.
   Both native phases earn their place, so the remaining `1.31 ms`
   f8-vs-f16 gap is structural (fp32 accumulators + f8 operand handling),
   not a bad phase that can simply be switched off.
2. **f16-KV loses under memory pressure.** With the game running, f16 is
   *slower* than f8 (`20.71` vs `22.75`), the opposite of the game-free
   session (`24.27` vs `23.65`). The loss sits entirely on the display device
   that the game shares (`dev1 23.57` vs f8 `18.38`; `dev0` is actually
   slightly *faster* at f16, `18.06`). f16 doubles the KV stream
   (`3072` vs `1536` MiB per GPU) and that traffic is what a busy display GPU
   cannot serve. So the f16-vs-f8 decode trade is load-dependent: f16 wins on
   a quiet machine, f8 wins when the display GPU is shared.

### Isolated WMMA throughput on gfx1201: fp8 is 2x faster than f16

The whole f8-FA question is bounded by the raw MMA rate, so it was measured in
isolation (`scripts/research/d138_fp8_wmma_density.hip`, built with hipcc
`-O3 --offload-arch=gfx1201` into `build_logs/d138_wmma_density.exe`; 16x16x16
WMMA, 32-lane waves, 8 independent accumulators per lane, `clock64` around the
loop; instruction forms taken from ROCm 7.2 `rocwmma/internal/wmma_impl.hpp`):

| variant | clk/MMA (1 block) | clk/MMA (64 blocks) | TFLOPS (64 blocks) |
| --- | ---: | ---: | ---: |
| f16 operands, fp32 acc | `20.74` | `20.87` | `19.84` |
| **fp8 operands, fp32 acc** | `10.67` | `11.01` | **`39.55`** |
| f16 operands, f16 acc | `22.08` | `22.94` | `20.71` |
| fp8 operands, LDS-fed | - | `93.62` | `11.84` |
| f16 operands, LDS-fed | - | `120.94` | `11.25` |

(The last two rows feed every accumulator from shared memory each step, one
8-byte (fp8) or 16-byte (f16) lane-local fetch per MMA, which is what
`load_matrix_sync` does inside the FA kernel. Wall-clock TFLOPS in that run are
depressed because the game was sharing the device; `clk/MMA` is the stable
number, and the same run reproduced `20.87 / 11.01 / 22.94` for the
register-fed variants.)

(GPU reports `CU=32` for one RX 9070 XT, `clk=2.46 GHz`, `warp=32`.)

Readings:

1. **The fp8 MMA itself is `1.9x` cheaper than the f16 MMA at equal fp32
   accumulator precision.** So a `2x` arithmetic advantage does exist on
   gfx1201; "fp8 is only a memory format" is wrong at the instruction level.
2. **That makes the FA result a software gap, not a hardware one.** Rough
   arithmetic for the 49K lane (16 attention layers, D=256, 24 heads,
   49152 keys, KQ + PV): `~2.0e10` FLOP/token, i.e. `~0.5 ms` at the measured
   `39.5` TFLOPS fp8 rate and `~1.0 ms` at `19.8` TFLOPS f16. The kernel
   actually spends `5.68 ms` (f8) / `4.37 ms` (f16), so the f8 body runs at
   `~9%` of its own MMA ceiling versus `~23%` for the f16 body: the fp8 path
   leaves `2.5x` more headroom on the table than the f16 path it loses to.
3. Where the loss goes is therefore *outside* the MMA: mandatory fp32
   accumulators (`fatn-wmma-f16.cu:96-98`), the P->E4M3 re-quantisation into a
   second shared tile plus its barrier, the fp32 VKQ store/`/128` merge, and
   the fact that Q is stored as raw f8 so the scale is applied per KQ element
   in the softmax loop. This is the actionable list, and it is consistent with
   the D098 finding that registers (not LDS) are the scarce resource.
4. **Operand delivery, not arithmetic, is the real limiter.** Feeding the same
   MMA from LDS collapses the fp8 advantage from `1.9x` to `1.23x`
   (`93.6` vs `120.9` clk/MMA) - and even the fp8 LDS form is `8.5x` slower
   than its register form. Inside FA every MMA needs freshly loaded operands
   (`load_matrix_sync` per K/V tile step), so the kernel is operand-bound and
   fp8's arithmetic win cannot be realised there.
5. **This explains the prefill/decode asymmetry.** On prefill the K/V loads are
   amortised over `1024` Q rows, so arithmetic matters and fp8 wins
   (`+4.6%`); on decode `q_rows=1` (traced: `q_rows=1 kv=... ncols=16 nwarps=8
   grid=(1,parallel_blocks,24)`, `parallel_blocks=2..8`) the `16x16` WMMA tile
   carries a single useful Q row, so almost all work is operand fetch plus
   wasted lanes, and fp8 pays the same LDS bill as f16 while being unable to
   spend its arithmetic surplus. That is a structural property of a `16x16`
   WMMA tile at `q_rows=1`, not a coding defect, and it is the same reason the
   f16 body only reaches `~23%` of its MMA ceiling too.

### Phase accounting inside the f8 FA body (measured 2026-09-25)

Instrumentation: five `clock64` marks around the shared-memory barriers of the
`k_VKQ` loop in `flash_attn_ext_f16` (`fattn-wmma-f16.cu`), accumulated by one
thread per block and flushed with six atomics into a device symbol; the host
reports and resets the symbol under `GGML_TRACE_FATTN_PHASE=1` (graphs
disabled, because the read synchronises the device).

**Probe tax (corrected 2026-09-25, late session).** The original neutrality
check (`23.66 tok/s` with the probe vs `22.7-23.9` without, L0) was valid for
the instance it was measured on, but the probe is *not* neutral in general: it
costs per KV iteration, so the instance with the larger `parallel_blocks`
(more iterations per block) pays more. Measured against a same-hour HEAD build
on the L2 lane: prefill `1942 -> 1327 tok/s` and ctrl decode `23.87 -> 17.38 t/s`
with the probe compiled in (`-32%` / `-27%`), while the GQA-packed instance
looked unaffected. The probe now lives behind `GGML_D138_PHASE` (compile-time,
off by default) and any absolute number taken with it enabled is inflated; the
shares below are still usable as a ranking (each mark adds the same `clock64`
overhead, and the six marks are spread across the loop), but treat the PV share
as "dominant" rather than as a calibrated figure.

Share of FA time per phase (L0 decode, `f8/f8` native, dev0, six samples
spread `0.3%`):

| phase | f8/f8 body | f16 body (f16 KV) |
| --- | ---: | ---: |
| KQ tile (K load + MMA + LDS store) | `9.3%` | `21.6%` |
| softmax + P re-quantisation | `11.8%` | `24.9%` |
| P fragment load (LDS) | `1.6%` | `3.4%` |
| **V fragment load + PV MMA** | **`58.8%`** | `29.3%` |
| VKQ LDS store | `13.6%` | `16.9%` |
| output accumulation (merge) | `4.8%` | `3.5%` |

Absolute check for the f8 body (one launch: `2832968` PV cycles over
`24 heads x 8 KV-splits x 8 k_VKQ steps = 1536` tile iterations =>
`1844` cycles/tile). Eight fp8 MMAs at the measured `11` cycles each are only
`~88` cycles, i.e. **`~95%` of the PV phase is operand delivery and its
dependencies, not tensor-core work**. For comparison the KQ phase spends
`280` cycles per tile on `16` K loads plus `16` MMAs (`17.5` cycles per step),
so K loads (row-major, contiguous lane addresses) are well hidden while V loads
(`col_major` via `stride_V`, one row per lane) are not.

Consequences:

1. the f8 body is a **PV-phase kernel**; KQ, softmax and the P path together
   are only `23%`;
2. the fp8 arithmetic advantage (isolated: `1.9x` over f16) is spent exactly
   where it does not matter - the MMA is `5%` of the phase that dominates;
3. the phase split also explains the f8-vs-f16 inversion: the f16 body spreads
   `~46%` into KQ+softmax and only `29%` into PV, so it is not the "same kernel
   with bigger bytes";
4. the actionable target is V operand delivery: **V is read once per Q head**
   (24 heads over 4 KV heads = `6x` re-read of the same bytes) and the
   `col_major` fragment layout makes each lane fetch a different row. Both
   point at the same fix: process several Q heads of one KV group in one block
   so one V fetch feeds several accumulators.

### P1: GQA column packing in the f8 FA body (prototype, 2026-09-25)

Motivation: the decode shape dispatches one block per Q head with `ncols=16`
and `ne01=1`, so 15 of 16 tile columns carry a zero Q row, and the K/V
fragments of one K/V head are fetched 6 times (24 Q heads / 4 K/V heads). The
vec/tile/MMA kernels already pack Q heads of one K/V head into the columns
(`zt_Q = z_KV*gqa_ratio + n*ncols2`, `fattn-tile.cuh`, `fattn-mma-f16.cuh`);
the WMMA body was the only one without it.

Implementation (opt-in, `GGML_ROCM_FATTN_F8_GQA_COLS=1`):

- new `bool gqa_cols` template parameter on `flash_attn_ext_f16`
  (`ggml/src/ggml-cuda/fattn-wmma-f16.cu`);
- with it, `sequence = blockIdx.z / ne12`, `head = (blockIdx.z % ne12)*gqa_ratio`
  and column `j` is Q head `head + j` of the same K/V head, so `Q_f` strides by
  `nb02` and `ncols_live = min(gqa_ratio, ncols) = 6`;
- the mask loses its per-column token term (all columns are the same token),
  ALiBi slope and sinks become per-column, and the store writes head
  `head + j_VKQ` at token `ic0`;
- launch: `launch_fattn<D, 1, cols_per_block>` instead of `<D, cols_per_block, 1>`,
  which makes `ntiles_z_gqa = ceil(6/16) = 1` and `blocks_num.z = ne12 = 4`.

Result (L0 and L2, f8/f8, same lane as the rest of this note, game in the
background). **Numbers below are the corrected set**: the first pass was taken
while the phase probe was still compiled into the kernel, which slowed the
non-GQA instance (see the probe trap below) and inflated the delta by ~6x.

| level | control | gqa_cols | delta |
| --- | ---: | ---: | ---: |
| L2 (49k), first clean pass | `23.94` / `24.11` | `25.34` / `24.54` | `+2..+6%` |
| L2, later pass (heavier background load) | `20.46` | `22.01` / `22.02` | `+7.6%` |
| L0 (8k) | `24.79` / `23.60` | `25.31` / `23.93` | `+1.4..+2.1%` |
| 15k ctx, greedy server A/B | `21.88` | `23.79` | `+8.7%` |
| L2, HEAD build (no patch at all) | `23.87` | - | reference |

Prefill is unchanged (`1937.6..1957.5 tok/s` with the patch, `1942.5` for the
patched-out HEAD build; the earlier spread `1286..1327` was the probe).

So the real effect is small and honest: packing the 6 Q heads of one K/V head
into the 16 columns removes the 6x re-read of K/V, but the FA phase is only
`~16%` of the L2 graph and `~3%` of L0, which bounds the win. The FA-internal V
traffic estimate (V delivery `~59%` of the f8 body) predicted about `-8%` at L2
if fully amortised; the measured `+2..8%` means the packed instance loses part
of that elsewhere. Checked and closed during promotion review:

- `parallel_blocks` is **not** a lever: the heuristic in `launch_fattn` picks
  `16` because `ntiles_dst = 4` and `nsm*max_blocks_per_sm = 64`, i.e. exactly
  one full wave at 100% efficiency; every other value either under-fills the
  wave or adds a second one for no coverage gain;
- prefill, f16 KV (`f16` lane logs no GQA activation at all) and short contexts
  are unaffected by construction (`Q->ne[1] == 1` is the gate);
- still open: (a) MTP/verify shape (`ne01 = 2..3` tokens x 6 heads = 12-18 of 16
  columns are live, but different tokens need different KV masks, so this is a
  design task, not a flag); (b) a fresh phase split with the probe compile-time
  gated, to see where the packed instance spends what it saves; (c) the same
  packing for the f16 body (not obviously useful on this lane - f16 KV loses on
  the display GPU).

Fidelity: token-exact against the unpacked path, using the same server, prompt
and greedy sampling (`temperature 0`, `seed 42`, `return_tokens`):

- reps=120 (~2.5k tokens, ctx 8192): `48/48` tokens equal, dec  `21.12 -> 23.56 t/s`;
- reps=340 (~15k tokens, ctx 16384): `48/48` tokens equal, prefill `11009 -> 11011 ms`,
  dec `21.71 -> 24.26 t/s`.

Status: **promoted to default on 2026-09-25** (`ggml/src/ggml-cuda/fattn-wmma-f16.cu`).
The dispatch is now a shape rule - `Q->ne[1] == 1 && Q->ne[2] % K->ne[2] == 0 &&
ratio > 1 && ratio <= cols_per_block && KQV->type == F32` inside the native f8
path - and `GGML_ROCM_FATTN_F8_GQA_COLS=0` is the escape hatch for A/B and for
bisecting a future shape. Re-verified after the switch, default vs `=0`:
A-B-A at L2 `22.01 / 20.46 / 22.02 t/s`, L0 `23.93 / 23.60 t/s`, 15k greedy
`48/48` tokens equal with `23.79` vs `21.88 t/s`, and the activation log (now
behind `GGML_TRACE_FATTN_WMMA_CONFIG`) present on the f8 lane and absent on the
f16 lane.

Probe trap (cost an hour, worth remembering): the first A/B gave `+6%`/`+38%`,
which is what the P1 store-index bug and the phase probe together produced. The
probe (six `clock64()` marks per KV iteration plus a per-block `u64[6]` counter
array) was neutral when it was added to the `D=256` decode instance, but the
packed prototype made the comparison nonsense: for the same wall time the
non-GQA instance takes twice as many KV iterations (`parallel_blocks 8` vs `16`),
so it pays the probe twice and the "GQA win" absorbed the probe tax. Measured
against a freshly built HEAD binary in the same five minutes:
prefill `1942 -> 1327 tok/s` (`-32%`) and ctrl decode `23.87 -> 17.38 t/s`
(`-27%`) *because of the probe*, while the GQA instance looked fine. The probe is
now behind `GGML_D138_PHASE` (compile-time, off by default) and the phase table
earlier in this note carries the probe tax in its absolute numbers. Rule: never
compare two kernel instances when the measurement harness is asymmetric between
them, and always keep a same-hour baseline build (here:
`bench2-bins/base-head-rocm72/`, a copy of `build-rocm72/bin` from the HEAD
source, run through `--server-bin`).

Trap fixed on the way (worth remembering): the first version of the store
index dropped `j_VKQ` from the token term, i.e.
`((seq*ne01 + ic0)*ne02 + head)`. Decode still looked fine (`j_VKQ == 0`),
but prefill collapsed every column onto one token and produced the classic
degenerate output - the greedy sampler locked onto token id 0 (`!`) for the
whole completion. Two lessons: **always compare generated tokens against a
baseline build before trusting a speed number**, and the cheapest reliable
baseline is `git stash push -- <file>` + rebuild + same test (no-second-worktree
needed). The broken version gave plausible tokens/s (`21.7 t/s` at L0), which is
exactly how a silent numeric break survives a benchmark-only workflow.

### MTP lane measured on ROCm (2026-09-26): the spec=none numbers above are not the production lane

The `UD-Q4_K_M` GGUF in `models/` is **already MTP-enabled**: with
`--spec-type none` its `blk.64.*` tensors are reported as *unused* at load, and
with `--spec-type draft-mtp` they are used (0 "unused tensor" lines). So the
whole note above was measured on the non-MTP lane; the production lane is:

| lane | L0 (8k) | L2 (49k) | acceptance |
| --- | ---: | ---: | ---: |
| `spec=none` | `27.47 / 27.69` | `24.32` | - |
| `spec=mtp --spec-n 3` | `37.00 / 37.00` | **`44.28`** | L0 `36/78 = 46%`, L2 `171/251 = 68%` |

(read A-B-A: L0 `none 27.69 -> mtp 37.00 -> mtp 37.00 -> none 27.47`, so the
effect is `+35%` at L0 and `+82%` at L2; the L2 arm is a single pair and the
higher number there tracks the higher acceptance, so treat `44.28` as
acceptance-dependent rather than a property of the lane.)

**Prefill regresses with MTP and it reproduces**: L0 `1942.5 / 1915.0` (none)
vs `1709.7 / 1703.6 / 1701.6` (mtp), i.e. `-11%`; L2 `1887.6` vs `1813.4`
(`-3.9%`). MTP is documented as a decode-only accelerator, so this is a real
open item, not an expected trade: either the draft head is being evaluated on
the prefill batch or the spec path changes the prefill graph. **Not new:**
`BENCHMARKS.md:91` already records `1731.71/39.58` + `74.36%` acceptance for
`mtp n3` on ROCm, i.e. the same prefill-below-non-MTP pattern; the earlier
`BENCH_RUNS.csv` rows (`...-mtp-n2`, `...-mtp-n3`, 2026-09-23 and 2026-09-25)
are the same lane. This section is a same-session A-B-A confirmation with the
acceptance split, not a new lane discovery.

Mechanism visible in the server log (the reason prefill is slower): the MTP
arm processes the prompt in **two batches** - `prompt processing progress,
n_tokens = 3739, batch.n_tokens = 3739` followed by `n_tokens = 3739,
memory_seq_rm [3739, end)` and `prompt processing done, n_tokens = 3995,
batch.n_tokens = 256`, while the `spec=none` arm runs the whole 3995-token
prompt as one `batch.n_tokens = 3995` - so the last 256 tokens of the prompt
are re-processed through a small, launch-bound batch. That matches the shape of
the regression (`-11%` on a 4k prompt, `-3.9%` on a 30k prompt).

**The GQA column packing does not apply on the MTP lane.** The promotion rule
requires `Q->ne[1] == 1`; the dispatch trace
(`GGML_TRACE_FATTN_PATH=1` + the D099 line) shows the verify shapes are
`Q=[256,2,24,1]`, `Q=[256,3,24,1]`, `Q=[256,4,24,1]` against
`K=[256,8192,4,1]` - 2-4 tokens x 24 Q heads over 4 KV heads at `head_dim 256`
- and the `WMMA FA GQA columns` line is printed 4 times on the `spec=none` arm
and 0 times on the MTP arm in the same session. FA for the verify shapes still
runs through the same WMMA f8/f16 body (no `RDNA4 choose VEC/TILE` lines were
emitted on either arm), so the column-packing work is *reachable* from the MTP
lane but needs the per-column KV masks that P1 deliberately skipped; the
possible column budget is `2x6 = 12` (fits 16), `3x6 = 18` and `4x6 = 24`
(do not fit in one `cols_per_block == 16` tile).

### The MTP prompt path from the inside (2026-09-26, no benchmark run)

Tool: `scripts/research/d138_mtp_prefill_probe.py` (starts the server itself,
one greedy 48-token completion, prints token ids + the phase lines; two arms
are meant to be diffed by hand). The per-batch breakdown needs the phase trace
widened from its decode-only default: `LLAMA_SPEC_SERVER_PHASE_TIMING=1` plus
`LLAMA_SPEC_SERVER_PHASE_TIMING_MAX_ROWS=8192` (the second knob is new; the
trace used to be hard-limited to `n_tokens <= 16`, so prompt batches were
invisible).

Measured on a 1309-token prompt (`n_ctx 8192`, `-ub 1024`, window 256):

| arm | bulk batch | tail batch | prompt total |
| --- | --- | --- | --- |
| `spec=none` | `1309` rows, one pass | - | `1948-2040 ms` |
| `spec=mtp` | `1053` rows (`1024 + 29`), `decode 2064 ms` | `256` rows, `decode 217 ms` + `spec-process 126 ms` | `2331 ms` |
| `spec=mtp` + tail align | `1024` rows (full ubatch), `decode 1441 ms` | `272` rows, `decode 234 ms` + `spec-process 182 ms` | `2211 ms` |

Three things this pins down:

1. The tail batch is not wasted work: its 217 ms are the same 256 tokens the
   bulk pass would have carried, and they come out at `0.85 ms/token` against
   `1.4-2.0 ms/token` for the bulk pass. The extra cost is the *split itself*
   plus the `spec-process` call (drafter warm-up/hand-off, `126 ms` per 256
   rows).
2. The split lands at `n_tokens - window` with no regard for the physical
   batch, so the bulk batch ends on a partial ubatch (`1053 = 1024 + 29` here,
   `3739 = 3x1024 + 667` on the L0 bench prompt). Each partial ubatch still
   costs a full pass over the layer stack. `LLAMA_SPEC_PREFILL_TAIL_ALIGN=1`
   moves the split down to the ubatch boundary (`272`-row tail) - **opt-in,
   default off, output identical** (see below). Whether it pays is a bench
   question and is not answered here.
3. **Fidelity trap:** on a prompt that can be "finished" (ending in a complete
   sentence plus an instruction) greedy decoding diverges between the arms -
   `spec=none` returns a single `248046` (stop) token while `spec=mtp`
   continues with text. On a prompt cut mid-sentence both arms return the same
   first 16 ids (`524, 539, 33633, 13, 13453, 32288, 369, 279, 10649, 20908,
   948, 264, 5979, 62991, 9705, 22327`). So the earlier "MTP gives different
   output" reading was ULP sensitivity around a near-tie between stop and text,
   not a broken nextn path - **verify MTP output on a prompt that must be
   continued**, otherwise the comparison is meaningless.

### Bench results (2026-09-26): tail alignment rejected, the window is a real knob

Measurements now exist, so the two guesses above can be replaced by numbers.
All runs `--level 0/2`, `f8_e4m3` KV, dual ROCm layer split, `spec=mtp n3`.

**Tail alignment (`tail_start` rounded down to `n_ubatch`) - REJECTED.**

| arm (L0, 3995-token prompt) | prefill tok/s | decode t/s |
| --- | ---: | ---: |
| `mtp`, window 256 (A) | `1487.3` | `32.27` |
| `mtp`, window 256 (A2) | `1503.4` | `32.30` |
| `mtp` + align (B) | **`1375.1`** | `33.71` |

`-8.5%` prefill, outside the `1%` spread of the two control runs. Worse, on the
49k lane the aligned split produced a 913-row tail and the server **aborted**:
`spec process: failed to select the pending MTP device row` ->
`failed to process speculative batch` -> `no tokens to decode` x4 ->
`fatal error` at `server-context.cpp:3166`. So the device handoff buffer is
bounded and a tail much larger than the window is fatal, not just slow. The
flag was removed from the code (`server-context.cpp` keeps a NOTE with these
numbers instead). Root cause of the extra cost: every extracted tail row costs
`~0.4-0.5 ms` (from `process` timings), which is far more than the partial
ubatch it removes.

**Where the window's prefill price comes from (phase trace, 2026-09-26).**

Two facts from the phase trace explain the linear prefill cost, and both are
about *what the draft context is asked to do*, not about the split itself:

1. **Bulk rows are only copied, tail rows are also decoded.** `process` for the
   bulk batch is `2.4 ms / 297 rows` and `15.4 ms / 3739 rows` - that is the
   `store_nextn_device_output` device copy (`LLAMA_MTP_DEVICE_HANDOFF_TRACE`
   shows `store rows=3739 first=1 cap=4097`, i.e. the whole prompt is staged on
   device). For the tail batch the same call costs `127-216 ms`, because the
   draft context also runs `decode_hidden_batch` over those rows. So a row moved
   from bulk into the tail picks up both a less efficient target batch
   (`0.52-0.59 ms` vs `0.49 ms` per row) **and** a full draft-context decode
   (`0.42-0.50 ms` per row) - roughly `+1.0 ms` per moved row.
2. **Bulk does not get cheaper when rows are moved out of it.** Its time tracks
   the number of physical batches, not the row count: for the 3995-token prompt
   the bulk is `4` batches at every window (`3739 = 3x1024 + 667`,
   `3675 = 3x1024 + 603`, `3483 = 3x1024 + 411`, `3227 = 3x1024 + 155`) and
   measures `1840 / 1832 / 1852 / 2000 ms`. Cutting the last physical batch from
   667 to 411 rows bought nothing back, so the whole tail cost is added on top
   instead of being traded.

Together that gives `ttft ≈ A + ~0.8 ms x window` on this prompt (`n_ctx 8192`,
`-ub 1024`), which is what the sweep shows:

| window | 64 | 192 | 256 | 320 | 512 | 768 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill tok/s | `1749.9` | `1526.2`* | `1683.8` | `1615.9` | `1575.5` | `1406.7` |
| ttft ms | `2283` | `2618`* | `2372` | `2472` | `2536` | `2840` |
| decode t/s | `35.15` | `33.42`* | `35.87` | `38.21` | `35.34` | `31.92` |
| acceptance | `35/81` | `37/75` | `36/78` | `38/72` | `36/78` | `36/81` |

\* the `192` row came from a run whose series drifted (a repeat inside one
series measured `1684.5` against `1683.8/1691.0` for `256`), so treat single
series as `+/-10%` and only compare windows measured in the same series.

Consequences: (a) the window is a *linear* prefill cost, so pushing it up is
only worth it when acceptance actually pays - `320` is the only point in the
sweep that returned more decode than it spent prefill, and `512`/`768` spent
without returning; (b) the honest target for a fix is not the split point (that
was tried, see above) but the *draft-context decode of the tail* - if those rows
could be staged for the handoff without being re-decoded in the draft context,
the window would get cheaper instead of more expensive. Whether the draft
context needs its own KV over the whole window or only over the last few rows is
the open question.

**The window is a real knob, but not a free one** (window sweep, 2026-09-26).

**What the draft context is actually asked to do** (code + probe, 2026-09-26,
and a follow-up that closes a direction):

- `common_mtp_sparse_capture_pos()` (`common/speculative.cpp:96-115`) returns
  `true` for **every** row when the batch is not larger than the window
  (`n_tokens <= window`), and otherwise only for `pos % stride < chunk` (HIP
  defaults `stride 32768`, `chunk 4096`). Rows that fail it are skipped inside
  `process()`, so they are neither decoded by the draft context nor given a
  draft-KV entry.
- `server_spec_prefill_capture_pos()` (`tools/server/server-context.cpp:173-182`)
  keeps the *last `window` tokens of the prompt* unconditionally
  (`pos >= n_prompt_tokens - window`). The draft context therefore always
  receives a contiguous recent segment of at least `window` rows - that is the
  draft KV it works from, and it is exactly the `process` cost measured above
  (`window x ~0.5 ms`: `127 ms` at 256, `216 ms` at 512). The cost is linear in
  the window by construction, not by accident.
- On prompts shorter than `chunk` (4096) the sparse rule degenerates to "capture
  everything": the handoff trace shows `store rows=3739 first=1 cap=4097` for a
  3995-token prompt at every window. The *store* covers the whole prompt; only
  the tail is *decoded* in the draft context - hence bulk `process` of
  `2.4-15.4 ms` against tail `process` of `127-216 ms`.

**Follow-up probe - raising the draft context's own ubatch is a regression.**
`LLAMA_MTP_CTX_UBATCH=512` (`common/speculative.cpp:2308-2324`; HIP default is
`min(n_ubatch, 256)`) was meant to test whether the `320` acceptance peak is an
artefact of splitting the tail into draft batches:

| window | prefill tok/s (ub 256 -> 512) | decode t/s (ub 256 -> 512) | acceptance (ub 256 -> 512) |
| ---: | ---: | ---: | ---: |
| 256 | `1684` -> `1442` | `36.2` -> `29.9` | `36/78` -> `35/81` |
| 320 | `1616` -> `1268` | `38.2` -> `29.9` | `38/72` -> `38/72` |
| 512 | `1576` -> `1348` | `35.3` -> `29.7` | `36/78` -> `36/79` |

So the `320` peak is **not** a draft-batching artefact - it reproduces exactly
(`38/72`) at both ubatches, i.e. it is a property of the tail length itself -
and raising the draft ubatch costs `-17%` prefill and `-17..-22%` decode. The
HIP-clamped default is the right one; this direction is closed.

### Per-node timing: the host is not the bottleneck (2026-09-26)

First use of the per-node trace (`GGML_TRACE_CUDA_NODE_TIMING` +
`_SYNC=1`, `runtime_graph.inc:1106-1134`), L2 49k, `spec=none`,
`GGML_HIP_DISABLE_GRAPHS=1` so every node is enqueued individually
(`d138-nodet-l2`, `d138-nodet-l2b`).

Decode graphs have `1890` nodes on one device and `1766` on the other
(`3656` per token, a full 64-layer pass per device), prefill chunks have the
same node count. What the trace says:

| quantity | value |
| --- | --- |
| `pre_sync_ms` summed over a whole decode graph | **`0.1-0.3 ms`** |
| `enqueue_ms` (host) per decode graph | `6-7 ms` |
| `sync_ms` per decode graph, instrumented (`SYNC=1`) | `92 / 75 ms` |
| wall decode, same run, no graphs | `47 ms/token` (`21.2 t/s`) |

So **the GPU never waits for the host**: the summed pre-node wait is a fraction
of a millisecond per graph even with graphs disabled and nodes enqueued one by
one. Host-side enqueue/launch overhead is therefore not what turns `10.53 GB`
of MMVQ traffic into `28.9 ms` - that hypothesis is dead.

Caveat that also kills the naive next step: `_SYNC=1` puts a synchronisation
after every node, so each node reports ~`0.1 ms` regardless of its real cost
(`RMS_NORM n=627` sums to `58 ms`, `UNARY n=480` to `44 ms`, i.e. the same
~`0.09-0.1 ms` per node as `MUL_MAT n=1383`). The instrumented graph total is
`537 ms` for 3 tokens against `47 ms/token` real, so per-node `sync_ms` values
are a floor, not a measurement. What is usable from the same dump is the *node
inventory*: per token the graph runs ~`1980` non-matmul nodes (`RMS_NORM`,
`UNARY`, `ADD`, `GET_ROWS`, `L2_NORM`, `CONCAT`, `SSM_CONV`, `ROPE`, `GLU`)
against ~`1380` matmul nodes - i.e. **two thirds of the graph's nodes do not
touch weights**, and MMVQ is only `458` of them.

Where that leaves the `~12 ms` question: `28.9 ms / 458 launches = 63 us` per
MMVQ launch against `23 MB / 645 GB/s = 36 us` expected from the isolated probe,
so each launch carries ~`27 us` of something that is neither bandwidth nor
host-side launch overhead. The remaining candidates are per-launch wave
ramp-up on a `grid = (1, parallel_blocks, 24)` shape and the dependency stall
between consecutive matmuls. The dup probe already showed the marginal byte is
cheaper than the average one (`462 GB/s` for added passes vs `364 GB/s`
effective), which is the signature of a fixed per-launch cost rather than of a
bandwidth ceiling.

### MMVQ node inventory and the real byte count (2026-09-26)

The same node dump gives the exact MMVQ inventory per token - this replaces the
older estimate that was built from `grid.x x bytes/block`:

| form (real decode) | nodes / 3 tokens | avg bytes |
| --- | ---: | ---: |
| `q5_K K=6144 N=5120` | `102` forward + `96` fused | `21.6` / `48.9 MB` |
| `q4_K K=8192 N=10240` | `87` | `29.5 MB` |
| `iq4_xs K=5120 N=17408` | `63` + `60` fused | `46-47 MB` |
| `q4_K K=8192 N=6144` | `72` | `17.7 MB` |
| `q4_K K=8192 N=17408` | `54` | `50.1 MB` |
| `q5_K K=8192 N=10240` | `36` | `36.0 MB` |
| `q6_K K=8192 N=1024` | `36` | `4.3 MB` |
| `q8_0 N=48` | `288` | `0.26 MB` |

Totals per token: **`461` matmul nodes**, `13.54 GB` of `src0` tensor bytes,
average `29.4 MB` per node. Two consequences:

1. **The decode byte figure in the memory, not the bandwidth, was wrong for the
   gravestone.** The D138 census counted `10.53 GB/token` as `grid.x x
   bytes/block`; the node trace reports `13.54 GB/token` of tensor bytes for the
   same pass. Recomputing with the trace number gives
   `13.54 GB / 28.9 ms = 469 GB/s`, not `364 GB/s` - i.e. MMVQ in the graph runs
   at ~`73%` of the isolated probe peak. That matches the independent dup probe
   (`457-462 GB/s` marginal), so the ~30% gap is real and consistent.
2. **The probe cannot measure these forms as they are used.** `run_vecdot`
   repeats the kernel over one buffer with an L2 churn between reps; a
   `21.6 MB` buffer fits entirely in the `64 MB` L2, and the measurement
   collapses (it printed `-0.007 ms`). The `61 MB` form is already polluted
   (`759 GB/s`, above the HBM fit). Only the `>300 MB` forms in that table are
   honest HBM numbers. To test real node sizes the probe needs a *pool* of
   buffers read once each, not one buffer read many times.

So the question from item 6 is now narrower and better posed: not "why is a
23 MB launch 63 us instead of 36", but **why does the same kernel reach
`645-660 GB/s` alone and only `460-470 GB/s` inside the decode graph** - with
the host, the launch count and the read pattern all ruled out.

### What the graph does to MMVQ, and the byte accounting conflict (2026-09-26)

Three measurements that narrow item 6 from "the kernel is slow" to "the shape
mix is slow". All on L2 49k, f8 KV, `spec=none`, 256 decode tokens.

1. **HIP graphs are worth ~2%.** Decode `25.82 t/s` with graphs vs `25.34 t/s`
   with `GGML_HIP_DISABLE_GRAPHS=1` (`d138-graphoff-l2`), i.e. enqueueing all
   `3656` nodes one by one costs almost nothing - consistent with the summed
   `pre_sync` of `0.1 ms` per graph. Graph-vs-no-graph is not the missing 30%.
2. **The two byte countings disagree by 21%, in the same run**
   (`d138-mmvb-l2`):
   - census of real `mul_mat_vec_q` launches: `11.19 GB/token`,
     `486` launches/token (`124482 / 256`);
   - node-tensor accounting of all `MUL_MAT` nodes: `13.54 GB/token`,
     `461` nodes/token.
   The gap is nodes that reach `MUL_MAT` without the `mul_mat_vec_q` path
   (`ncols_dst > 1`, e.g. the `ncols_dst=2` starts visible in the census). So
   "MMVQ bandwidth" and "matmul-class bandwidth" are different denominators:
   `11.19 GB / 28.9 ms = 387 GB/s` for MMVQ proper, `13.54 GB / 28.9 ms =
   469 GB/s` for the whole matmul class. The earlier memory figure of
   `10.53 GB/token` is the same census on a 272-token run; the difference from
   `11.19` is run-to-run shape drift, not a method change.
3. **Inside the real graph the big launches still reach ~91% of peak.**
   `GGML_TRACE_MMVQ_TIMING_SYNC=1` (`d138-mmvsync-l2`) synchronises after every
   launch, so each node is timed alone: per-launch bandwidth p100 =
   `588 GB/s`, p90 = `278`, p50 = `70 GB/s`. The best launches are the
   `1.04 GB` `q6_K` vocab projections at `1.77 ms`. So a large MMVQ node in a
   real decode graph runs at `588 GB/s` against `645-660 GB/s` isolated - a
   `~10%` graph tax, not a `30%` one.

Consequence: the `387-469 GB/s` average is not the kernel's fault. It is the
**shape mix**: `486` launches per token, of which `197` read under `4 MB`
(`q8_0 0.67 MB x110`, `q4_K 3.74 MB x87`). Those carry `~400 MB` of the
`13.54 GB` (`3%`) but each still pays wave ramp-up and a dependency stall.
Under the sync harness they account for `26.6` of `87.6 ms` (`30%`), which is
an upper bound (the sync itself costs ~`0.1 ms` per node) - the honest open
question is now *the real in-graph cost of a small MMVQ launch*, and the
instrument for it already exists: `GGML_MMVQ_MICRO_DUP` re-runs narrow
`ncols_dst == 1` launches a set number of extra times, so the marginal cost of
the micro class can be read off the wall decode.

### The micro class is priced: not the missing 30% (2026-09-26)

`GGML_MMVQ_MICRO_DUP=N` (mmvq.cu:1158-1174) repeats bit-identical narrow launches
(`ncols_dst == 1` and `nrows <= 64`), which is exactly the class the sync
harness accused of eating a third of the MMVQ phase. Counted from the census
log (the dup condition mirrors the launch condition, including
`rows_per_block = small_k ? nwarps : 1`):

| quantity | value |
| --- | --- |
| micro launches per token | **`101.2`** (all `q8_0 48x5120`) |
| micro bytes per token | `26.4 MB` = **`0.2%`** of the matmul traffic |

A-B-C-A-D on L2, `spec=none`, f8 KV, decode t/s:

| `MICRO_DUP` | runs | decode t/s |
| ---: | --- | --- |
| `0` | `a1/a2/a3` | `22.41 / 22.30 / 22.81` |
| `1` | `b1/b2` | `23.13 / 22.71` |
| `2` | `d1` | `21.47` |
| `4` | `c1` | `21.86` |

Controls span `2.3%`, and `+101` launches (dup=1) sits *above* the control
mean while `+404` (dup=4) sits `~2.9%` below it. Fitting the only monotone
part (`0 -> 4`, i.e. `+404` launches, `+106 MB/token`) gives
`1.3 ms / 404 = ~3 us` per micro launch, so the whole micro class costs
`101 x 3 us + 26.4 MB/0.6 TB/s = **0.3 ms/token (0.7%)**`, not `26.6 ms`.

**Verdict: the "197 small launches eat 30%" hypothesis is dead.** The sync
number was an artefact of the harness itself (every launch paid a
`~0.1 ms` drain). The remaining explanation for the `469` vs `588-645 GB/s`
gap has to come from the two candidates the trace cannot separate yet:
the `ncols_dst > 1` `MUL_MAT` nodes that bypass `mul_mat_vec_q`
(`13.54 - 11.19 = 2.35 GB/token`) and inter-device/dependency stalls.

### Byte-accounting correction: `364 GB/s` was an artefact (2026-09-26)

The census of `tmp_d138_bytes_full.py` computed bytes as
`grid.x * bytes_per_row`, which is wrong for `small_k` launches: there
`rows_per_block = nwarps = block.y`, so a launch covers
`grid.x * block.y` rows. `calc_rows_per_block` (`mmvq.cu:449-470`) confirms it,
and the raw log shows it directly:

```
type=12/q4_K ncols_dst=1 small_k=1 ncols_x=5120 grid=(1280,1,1) block=(32,8,1)
```

i.e. `1280 x 8 = 10240` rows, not `1280`. q4_K is exactly the hot type that
takes `small_k`, so the old count lost a factor of `8` on it (`0.43` instead of
`3.42 GB/token`) and nobody noticed because q5_K/iq4_xs are `small_k=0`.

Corrected census for the same run (`tmp_d138_bytes_fix.py`, 256 tokens):

| quantity | value |
| --- | --- |
| MMVQ launches per token | `486` |
| bytes per token, old accounting | `11.18 GB` |
| **bytes per token, corrected** | **`14.29 GB`** |
| of which `fusion=0` | `365` launches, `8.45 GB` |
| of which `fusion=1` | `121` launches, `5.84 GB` |

That also reconciles the two methods: the node-tensor count gave `13.54 GB`,
the corrected census gives `14.29 GB` (`5%` apart), so the earlier "21%
conflict" was one bad arithmetic rule, not two different objects.

**And the class price agrees.** `GGML_MMVQ_MICRO_DUP=N` + `GGML_MMVQ_DUP_ALL=1`
duplicates every `ncols_dst == 1` launch `N` extra times, i.e. adds
`14.29 GB/token` per dup step (L2, `spec=none`, f8):

| `MICRO_DUP`+`DUP_ALL` | decode t/s | ms/token | marginal GB/s over control |
| --- | ---: | ---: | ---: |
| `0` (a1/a2) | `24.78 / 23.14` | `40.4 / 43.2` | - |
| `1+1` | `14.40` | `69.4` | **`516`** |
| `3+1` | `7.71` | `129.9` | **`486`** |

Linear in the number of dups (`27.7 ms` then `88.2 ms` for `1x` and `3x`), so
the marginal bandwidth of the MMVQ class inside the real graph is
**`~500 GB/s`**, not `364`. Both independent methods now say the same thing.

Consequences, and this is where item 6 lands:

1. The decode reads `14.29 GB/token` of MMVQ weights and runs the class at
   `~500 GB/s` = **`76-78%`** of the `645-660 GB/s` isolated peak, with the
   largest nodes reaching `588 GB/s` (`91%`).
2. The old "`58%` of peak, `12 ms` unaccounted" framing is retired: the
   remaining `~25%` is spread over the mid-size nodes (ramp-up and dependency
   stalls) plus the `121` fused launches, not concentrated anywhere a single
   fix could remove it.
3. The practical ceiling for this lane: `14.29 GB / 645 GB/s = 22.2 ms` of pure
   MMVQ plus `~10.6 ms` of non-matmul nodes = `32.8 ms/token` (`30.5 t/s`)
   against the current `39.5 ms` (`25.3 t/s`). So even a perfect MMVQ only buys
   `+20%` decode - the lane's real headroom is in *doing less work per token*
   (multi-token verification), which is the MTP result already measured.

### Sparse capture parameters: the default survives both extremes (2026-09-26)

`LLAMA_SPEC_PREFILL_SPARSE_STRIDE` / `_CHUNK` (`common/speculative.cpp:96-115`)
decide which prompt rows enter the draft KV: with the HIP defaults
(`stride 32768`, `chunk 4096`) a 49k prompt captures `pos % 32768 < 4096`, i.e.
*the first 4096 tokens plus the windowed tail*. `LLAMA_MTP_DEFER_SPARSE_PREFILL`
(default on for HIP) defers that copy (`speculative.cpp:1384-1397`), so the
rows land after the bulk pass. A-B-C-A on L2 (`spec=mtp n3`, f8), one run per
extreme, controls `a1`/`a2` agree on acceptance exactly:

| arm | prefill | decode | acceptance | drafts offered |
| --- | ---: | ---: | ---: | ---: |
| default (`32768/4096`) | `1624.9` / `1607.5` | `36.25` / `35.21` | **`171/251 = 68.1%`** | `251` |
| sparse off (`stride 0`) | `1727.1` (`+6.6%`) | `34.40` (`-5.1%`) | `169/257 = 65.8%` | `257` |
| full capture (`256/256`) | `1426.5` (`-11.5%`) | `30.36` (`-16%`) | `156/294 = 53.1%` | `294` |

Readings:

1. **The captured history is worth its price.** Dropping it costs `2.3 pp`
   acceptance; capturing everything costs `15 pp`. The default is a local
   optimum, so this lever does not hand us a free acceptance gain.
2. **The sparse rows are not free copies.** Skipping them saves `1.1 s` of
   ttft, far more than the `~17 ms` the `store` phase bills for 4096 rows -
   so the price is paid by the drafter attending over a `4096 + window` KV
   during generation, and (with `defer`) by the deferred catch-up inside the
   prompt phase.
3. **Break-even against `spec=none`-style workloads**: at 256 decoded tokens
   sparse-off is the faster arm end-to-end (`25.2 s` vs `25.9 s`); the
   `+6.6%` prefill only pays off for long generations, where the `-5%` decode
   dominates.
4. **Load caveat**: this series ran with a background game active and its
   absolute numbers sit `~19%` under the clean L2 lane (`44.3 t/s`), so treat
   the comparison as in-series only and re-run on a clean GPU before acting.

Not yet tested: intermediate points (`chunk 8192`, `stride 16384`), and
`LLAMA_MTP_DEFER_SPARSE_PREFILL=0` to separate the catch-up cost from the
attention cost.

**Clean re-run, same day, GPU idle** (`d138-sp2-l2-*`), five arms plus two
controls:

| arm | prefill | decode | acceptance | total |
| --- | ---: | ---: | ---: | ---: |
| default (`32768/4096`) a1/a2 | `1828.1` / `1810.0` | `44.35` / `43.59` | **`171/251 = 68.1%`** | `22516` / `22784` |
| sparse off (`stride 0`) | `1871.8` (`+3.1%`) | `43.43` (`-1.0%`) | `169/257 = 65.8%` | `22247` |
| `chunk 8192` | `1815.5` | `37.64` (`-14%`) | `156/294 = 53.1%` | `23660` |
| full capture (`256/256`) | `1531.1` (`-15%`) | `32.59` (`-26%`) | `156/295 = 52.9%` | `27846` |
| `defer` off | `1800.4` | `41.61` (`-5.4%`) | `166/263 = 63.1%` | `23154` |

Three things this settles:

1. **Acceptance is deterministic between sessions.** Every arm reproduced the
   loaded-run acceptance exactly (`171/251`, `169/257`, `156/29x`, `166/263`),
   so the earlier "`-5.1%` decode for sparse off" was load noise: on a clean
   GPU sparse off costs only `-1.0%` decode and *gains* `+3.1%` prefill.
2. **There is a cliff between `4096` and `8192` captured rows, and beyond it
   the capture is worthless.** `chunk 8192` lands on the same acceptance as
   full capture (`53.1%` vs `52.9%`) while capturing a quarter of the rows -
   so the extra history does not just cost more, it degrades the drafter.
   The handoff buffer's `rows_capacity` is `4097` (`src/llama-context.cpp:1459`,
   trace `store rows=... cap=...`), which is suspiciously close to the cliff
   and is the thing to check next with `LLAMA_MTP_DEVICE_HANDOFF_TRACE=1`.
3. **`defer` earns its default.** Turning it off costs `-5.4%` decode and
   `-5.0 pp` acceptance, so the deferred catch-up is not just book-keeping -
   it is what keeps the captured rows useful.

Practical outcome for the lane: keep `stride 32768 / chunk 4096 / defer on`;
`chunk 8192` and full capture are rejected. Sparse off stays an option only for
short generations (it wins `269 ms` of ttft and loses `1%` decode).



### Sampling: the bench does not run the model's recommended mode (2026-09-26)

The model carries its own recommendation in the GGUF metadata -
`general.sampling.temp = 1.0`, `general.sampling.top_p = 0.95`,
`general.sampling.top_k = 20` - which is the standard Qwen3.8 thinking-mode
set (add `min_p = 0`, `presence_penalty = 0`, `repetition_penalty = 1.0`).
`llama-arch.cpp` only maps those keys to names; **nothing reads them at
runtime**, so they never reach the sampler. What actually runs:

| parameter | bench actual | model recommends | source of the actual value |
| --- | ---: | ---: | --- |
| temperature | `0.2` | `1.0` | `bench2.py` default, sent in the request |
| top_p | `0.9` | `0.95` | `bench2.py` default, sent in the request |
| top_k | `40` | `20` | server default, `common/common.h:216` |
| min_p | `0.05` | `0.0` | server default, `common/common.h:218` |
| repeat / presence / frequency | `1.0 / 0 / 0` | same | defaults, already matching |

`bench2` has `--temperature`/`--top-p` but no `--top-k`/`--min-p`; those go
through `--server-extra "--top-k 20 --min-p 0"`. The recommended-mode runs below
used exactly that. The chat template is already thinking-on
(`server.log: chat template, thinking = 1`), so the difference is purely the
sampler.

**Measured impact (same server, same lanes, dual ROCm, f8 KV, `spec=mtp n3`):**

| lane | mode | prefill tok/s | decode t/s | acceptance |
| --- | --- | ---: | ---: | ---: |
| L2 49k, mtp | actual | `1785.9` / `1796.3` | **`43.46` / `43.60`** | **`171/251 = 68.1%`** |
| L2 49k, mtp | recommended | `1791.2` / `1794.2` | **`35.58` / `35.73`** | `152/308 = 49.4%` |
| L2 49k, none | actual | `1857.6` | `25.82` | - |
| L2 49k, none | recommended | `1856.7` | `25.65` | - |
| L0 4k, mtp | actual | `1683.4` | `36.47` | `36/78 = 46.2%` |
| L0 4k, mtp | recommended | `1687.4` | `36.17` | `36/80 = 45.0%` |

Both runs of each arm reproduced **identically** (the seed still pins the
stream even at `temp 1.0`), so these are not noise. Conclusions:

1. Sampling does not touch prefill at all (`1786-1796` vs `1791-1794` at L2) -
   expected, and it means the prefill/window results above are mode-independent.
2. Without speculation sampling is inert (`25.82` vs `25.65`, `-0.7%`): there is
   no verifier whose work could change.
3. With MTP the sampler sets the acceptance ceiling, and therefore the decode
   result: at L2 `temp 1.0/top_k 20/min_p 0` costs `-18%` decode and
   `-18.7 pp` acceptance, which cuts the MTP gain from `+68.6%` over `spec=none`
   to **`+39.0%`**. Any MTP number measured in the old regime is optimistic.
4. The size of the effect is **context-dependent**: it is ~0 at L0
   (`36.47 -> 36.17`, acceptance `46.2 -> 45.0%`) and large at L2. Do not
   generalise a sampling result from one level to another.



| window (L0) | prefill tok/s | decode t/s | acceptance |
| ---: | ---: | ---: | ---: |
| 256 | `1684.0` / `1682.5` | `36.48` / `36.51` | `36/78 = 46.2%` |
| 320 | `1645.7` / `1646.1` | **`39.56` / `39.62`** | `38/72 = 52.8%` |
| 512 | `1611.9` | `36.23` | `36/78 = 46.2%` |

Window 320 buys `+6.6 pp` acceptance and `+8.5%` decode for `-2.2%` prefill at
L0 (all four runs consistent, A-B-A). Window 512 buys nothing and only costs
prefill - so the gain is not monotonic in the window and 320 is not simply
"more context is better"; the mechanism behind the 320 point is not yet
understood and must be established before anyone hardcodes it.
On the 49k lane the same 320 window was **within noise**: decode `44.09` vs
`43.13 / 42.61` and acceptance `172/248 = 69.4%` vs the `68%` previously
recorded at window 256, prefill `1797.9` vs `1723.5 / 1792.0`. Long prompts
already carry a large window-relative KV, so the L0 gain does not transfer.

## Handoff / next bench window

GPU runs in this workspace are scheduled by the operator: the agent must ask
before starting any benchmark. Pending arms, in priority order, each cheap on
L0 (`~25 s`) and confirmed on L2 (`~40 s`):

0. ~~**GQA column packing (P1) promotion decision.**~~ - **done 2026-09-25:
   promoted to default** (shape rule, `GGML_ROCM_FATTN_F8_GQA_COLS=0` disables).
   Default vs disabled: L2 A-B-A `22.01 / 20.46 / 22.02 t/s`, L0 `23.93 / 23.60`,
   15k greedy `48/48` token-equal, `23.79` vs `21.88 t/s`; `parallel_blocks`
   reviewed and left alone (the heuristic already lands on one full wave).
   Remaining, in value order: (a) MTP/verify shape (`ne01 = 2..3` tokens x 6
   heads = 12-18 of 16 columns, needs per-column KV masks); (b) fresh phase
   split with `GGML_D138_PHASE` to see where the packed instance spends what it
   saves; (c) f16 body packing (low value on this lane);
1. ~~`GGML_ROCM_FATTN_F8_NATIVE_KQ=0` (f8, L0 + L2) vs the f8 control~~ -
   **done 2026-09-25 while a game was running: portable path is `-27%`,
   rejected** (see the A-B-A section above);
2. `GGML_ROCM_FATTN_F8_NATIVE_V=0` - phase bisect (KQ vs P*V). Given the
   A-B-A result the upside is now bounded by the `~1.3 ms` f8-vs-f16 gap and
   a portable PV leg is unlikely to be faster, so this is low priority;
   **done 2026-09-25: portable PV is `-14%` decode, rejected.**
3. f16-KV re-bracket: only interesting if a lane can afford double KV memory
   and accepts the `-4.4%` prefill (`+2.6%` decode); not a default candidate.
   New data point: with the display GPU shared, f16 also loses decode
   (`20.71` vs `22.75 tok/s`), so this trade is load-dependent;
4. ~~**MMVQ geometry axis.**~~ - **closed 2026-09-25 (C1 + C2, see the
   byte-exact census section)**: Q4_K `small_k` on/off is `0%` (588 launches
   trace-verified flipped), and inverting `should_use_small_k` for the real
   mass (`q5_K` + `iq4_xs`, `76%` of decode bytes) is `-8.6%`. The policy in
   the tree is already the right one; probes stay default-off
   (`GGML_MMVQ_RDNA4_Q4K_ROWS1`, `GGML_MMVQ_HOT_ROWS1_INVERT`);
5. **Step 3 (per-thread K-window widening) - REJECTED by measurement
   2026-09-25.** The isolated probe (`scripts/research/d138_vecdot_probe.hip`,
   real `vecdotq.cuh` code, production loop shape) reaches `643-662 GB/s` on
   >300 MB forms, i.e. `96-98%` of the card's measured `~665-675 GB/s` read
   peak, for both the 20-block and 68-block K shapes. There is no
   request-size waste to recover. What the probe *did* find is a fixed
   `1.4-1.9 us` per launch, which costs the real decode only `0.6-0.9 ms/token`
   because its launches average `23 MB`;
6. **Where the MMVQ time actually goes (updated 2026-09-26 after four
   passes; the original `10.53 GB at 364 GB/s` framing is superseded):**
   - *bytes, corrected*: the census must multiply `small_k` launches by
     `block.y` (`rows_per_block`); q4_K is the type that takes `small_k`, so the
     old rule under-counted it `8x`. Corrected: **`14.29 GB/token`** over `486`
     launches (`8.45` non-fused + `5.84` fused), which reconciles with the
     node-tensor figure (`13.54 GB`, `5%` apart) - the earlier "21% conflict"
     was an arithmetic bug, not two objects;
   - *class price (DUP_ALL probe)*: adding `14.29 GB/token` of duplicates costs
     `27.7 ms` (1 dup) and `88.2 ms` (3 dups), i.e. a **marginal `~500 GB/s`**
     for the MMVQ class inside the real graph. So the lane runs at
     `76-78%` of the `645-660 GB/s` isolated peak, and the largest nodes alone
     reach `588 GB/s` (`91%`);
   - *ceiling*: `14.29 GB / 645 GB/s = 22.2 ms` of pure MMVQ + `10.6 ms` of
     non-matmul nodes = `32.8 ms/token` (`30.5 t/s`) versus the current
     `39.5 ms` (`25.3 t/s`). **A perfect MMVQ buys only `+20%` decode** - the
     remaining headroom is in doing less work per token, i.e. MTP;
   - *retired*: `364 GB/s`, "`58%` of peak", and the search for the single
     missing `12 ms`;
   - *the host is excluded*: `pre_sync = 0.1-0.3 ms` per whole graph, and
     disabling HIP graphs costs only `+1.9%` (`25.82` vs `25.34 t/s`);
   - *the kernel is close to peak inside the graph*: per-launch sync trace
     shows the large nodes at `588 GB/s` (p100) = `91%` of the isolated
     `645-660 GB/s`;
   - *the micro class is priced and innocent*: `101.2` launches/token
     (`q8_0 48x5120`), `26.4 MB/token` (`0.2%`), `~3 us` per launch =>
     `~0.3 ms/token` (`0.7%`), not the `26.6 ms` the sync harness suggested;
   - **open**: (a) the `ncols_dst > 1` nodes on the MMQ path
     (`2.35 GB/token`) have never been timed separately, and their bandwidth
     may be the whole gap; (b) inter-node and inter-device stalls (`dev0 16.9
     + dev1 15.4 ms` = `82%` of the wall). Next instruments: `GGML_MMVQ_DUP_ALL`
     for a class price with an order-of-magnitude larger signal than
     `MICRO_DUP`, and a device-timing split of the MMQ nodes;
7. **MTP lane (measured 2026-09-26):** the production lane is
   `--spec mtp --spec-n 3` (`44.28 t/s` at L2 vs `24.32` for `spec=none`).
   Prompt-path follow-ups, in value order: (a) **`LLAMA_SPEC_PREFILL_WINDOW`**
   is a real knob at L0 - `320` gives `+6.6 pp` acceptance and `+8.5%` decode
   for `-2.2%` prefill, while `512` gives nothing (numbers above) - but the
   `320` mechanism is unexplained and the L2 lane shows no gain, so do not
   hardcode it before that is understood; (b) the tail split makes the L0
   prefill `-11..-13%` against `spec=none` and the tail cost is `~0.4-0.5 ms`
   per extracted row, so the only way forward is to make the handoff cheaper,
   not to move the split (align: rejected, `-8.5%` prefill and a fatal abort at
   913 rows); (c) verify-shape GQA packing (`Q1 = 2..4` x 24 heads over
   `cols_per_block == 16`) - **closed 2026-09-26 as not worth the kernel edit**:
   the verify graph spends `~3.1 ms` of `~24 ms` on FA (8 nodes,
   `ne=(256,24,4,1)`), the packing fits only `2 x 6 = 12` of 16 columns, and
   the FA nodes are latency-bound (read `0.1 MB` per node while costing
   `0.48 ms`), so the returned traffic is not the returned time. See the
   section above for the full table; (d) sparse capture
   parameters (`LLAMA_SPEC_PREFILL_SPARSE_STRIDE/CHUNK`) - measured
   2026-09-26 and **the default is a local optimum**: sparse off costs `-5.1%`
   decode for `+6.6%` prefill, full capture costs `-16%` decode and `-15 pp`
   acceptance (see the sparse section). Remaining sub-arms, cheap but needing a
   clean GPU: `chunk 8192`, `stride 16384`,
   `LLAMA_MTP_DEFER_SPARSE_PREFILL=0`. That series ran with a background game,
   so re-run clean before acting on it;
   (e) **read every MTP number as mode-dependent**: the recommended thinking
   sampling (`temp 1.0/top_p 0.95/top_k 20/min_p 0`) costs `-18%` decode and
   `-18.7 pp` acceptance at L2 (MTP gain drops from `+68.6%` to `+39.0%` over
   `spec=none`), while L0 shows no effect. The model ships those values as
   `general.sampling.*` metadata and nothing in the runtime reads them;
8. **Device serialisation confirmed, split axis closed** (`d138-devtiming-l2`,
   `GGML_TRACE_CUDA_GRAPH_DEVICE_TIMING`, decode graph medians): `dev0 16.9 ms`
   + `dev1 15.4 ms` = `32.3 ms` against a `39.5 ms` token (`25.31 t/s`), i.e.
   device time is `82%` of the wall and the two GPUs **do not overlap** - at
   batch = 1 with `-sm layer` every layer waits for the previous one, so the
   second card buys memory, not decode speed. Consequence: the decode ceiling
   is one card's memory bandwidth (`624 GB/s`), and only MTP-style multi-token
   verification changes that (it gives a layer more than one row of work).
   `-sm row` was measured and is **dead on this lane**: L0 prefill `25.9 tok/s`
   (vs `1600+`) and decode `1.38 tok/s` (vs `24`) - inter-device traffic per
   layer dominates. Do not retry `-sm row` here (`c3-l0-row`, `c3-l2-row`;
   the L2 arm was killed during prefill after 8192/30609 tokens);

### 7(c) verify-shape GQA packing: measured, not worth the kernel edit (2026-09-26)

Before touching the WMMA body (a `Q`-addressing / mask / store / dispatch edit
in four places), the size of the prize was measured. The ablation route is
poisoned - `GGML_CUDA_ABLATE=FLASH_ATTN_EXT` destroys acceptance
(`2/753 = 0.3%` vs `171/251 = 68.1%`), decode falls to `16.15 t/s` and the
number stops measuring FA - so the node trace was used instead
(`GGML_HIP_DISABLE_GRAPHS=1`, `GGML_TRACE_CUDA_NODE_TIMING[_SYNC]=1`,
`SKIP_GRAPHS=150`, `MAX_GRAPHS=3`, reader `scripts/research/d138_mtp_compose.py`).

The MTP decode path in `spec=mtp n3` is: drafter graphs of `9-23` nodes
(`MUL_MAT 3-8`, `FLASH_ATTN_EXT 1`) and one verify graph of **`802` nodes**:

| op | nodes | sync ms | note |
| --- | ---: | ---: | --- |
| `MUL_MAT` | `256` | `46.0` | `7.11 GB` of weights, the bulk |
| `RMS_NORM` | `107` | `9.9` | at the `~0.1 ms` sync floor entirely |
| `ADD` | `91` | `8.5` | same |
| `UNARY` | `83` | `8.2` | same |
| `GET_ROWS` | `50` | `4.6` | `327 MB` |
| `GATED_DELTA_NET` | `25` | `4.2` | SSM core |
| **`FLASH_ATTN_EXT`** | **`8`** | **`3.9`** | **`ne=(256,24,4,1)`, `~0.48 ms` each** |

Two things decide it:

1. **The prize is bounded.** After removing the sync floor (`~0.1 ms` per
   node) from the two large classes, the verify graph spends roughly
   `20 ms` on `MUL_MAT`, `3.1 ms` on FA and `1.7 ms` on `GATED_DELTA_NET`.
   So FA is the second class, but at `~12%` of the graph, and the graph is
   only part of an MTP token. Even a perfect FA would buy single-digit
   percent.
2. **The column budget does not fit the shapes that matter.** Packing needs
   `n_tokens x ratio <= cols_per_block = 16`; with `ratio = 6` only
   `n_tokens = 2` fits (`12/16`). `n3` runs a 4-token verify, and the log
   confirms the relevant nodes are `ne=(256,24,4,1)` - they cannot use the
   packed tile at all. The realistic payoff is a fraction of the `3.1 ms`
   above, for an edit to the most delicate part of the FA body.

Also worth stating plainly: the FA nodes read only `~0.1 MB` of K/V each
while costing `~0.48 ms`, i.e. three orders of magnitude off the bandwidth
that would explain them. The verify-shape FA is latency/compute-bound, and
head packing attacks operand traffic, not latency - which is consistent with
the fact that the same packing on `spec=none` (where it *is* a bandwidth
story) paid off at `+7.6%`.

**Verdict: 7(c) is closed as not worth the kernel edit.** Keep the packing
rule at `Q->ne[1] == 1`. The MTP lane's remaining lever is the number of
verify passes (acceptance), which is the sparse/window work already done.

### Next candidates, re-prioritised by the phase split

P1 (highest value, no GPU probing needed to justify): **feed one V fetch to
several Q heads of the same KV group.** V is currently re-read `6x` (24 Q heads
over 4 KV heads) and the PV phase is `59%` of the f8 body with `95%` of it in
operand delivery. Batching `gqa_ratio_eff` heads into the tile's N dimension
amortises exactly that traffic. First implementation step is a measurement, not
a rewrite: verify the GQA ratio and the KV-head grouping of the production
model, then prototype the multi-head tile on D=256 (the existing RDNA4
"small Q x GQA -> TILE/VEC" dispatch only covers D<=128, `fattn.cu:556-615`).

P2: **hide V-load latency in the PV loop** (software pipelining/prefetch of the
next `k0` fragment). Cheap to try inside the existing body, but bounded: it can
only recover hidden latency, not the `6x` duplicate traffic.

P3: check whether the `_v` scalar path or a non-WMMA design is better for
`q_rows==1` with f8 KV on RDNA4. The isolated numbers say a 16-wide tile at one
useful column is the wrong shape, but every portable alternative measured so
far (TILE at `-27%`, `NATIVE_V=0` at `-14%`) is worse.

## Artifacts

- baseline run dirs: `build_logs/bench/d138-base-q4k-l2--20260925T112226805832Z-13f5d1f78e0f4fae9a3774468465c5ff`,
  `build_logs/bench/d138-base-q4k-l3--20260925T112329140435Z-fc7ffdd7dd95441193b06f8f1b00b99d`
- census run dirs: `build_logs/bench/d138-census-mmvq-l2--20260925T112546303765Z`,
  `build_logs/bench/d138-census-nograph-l2--20260925T113054706437Z`
- probe arms: `d138-abc` (step 2), `d138-dup0/dup1/dup10` (step 3),
  `d138-dupall-l2` / `d138-dupall-fused-l2` (step 4), `d138-graph-split-l2` (step 4 traces)
- step 5 arms: `d138-dev-ctrl-l0`, `d138-dev-MUL-l0` (pre-fix matcher, kept as
  the negative example), `d138-abx-*-l0` (exact-op ablations), `d138-l2gpu-{ctrl,xmulmat,xflashattnext}`,
  `d138-nograph-nodetrace-l0` (`GGML_HIP_DISABLE_GRAPHS=1` + per-node trace),
  `d138-nodegpu-l1` (failed in-graph event timer)
- step 6 arms: `d138-nofusion-l2`, `d138-l0-nofusion-dev`
- step 7 arms: `d138-allmm-{L0,L2}` (all MMVQ ablated), `d138-mmfa-{0,2}`
  `d138-nf-*`, `d138-pj-*`, `d138-p2-*` (documented as unreliable),
  `d138-fattnpath-l0` (`GGML_TRACE_FATTN_PATH=1`, 1664 dispatches, all
  `native_kq=1 native_v=1 reference=0`)
- tools (moved to `scripts/research/` during the D138 cleanup; run from the
  repository root): `d138_census.py`, `d138_nodes.py` (per-graph op/name
  breakdown), `d138_nodetrace.py`, `d138_ablate.py`,
  `d138_ablate_summary.py`, `d138_mtp_compose.py` (MTP graph composition and
  FA shapes), `p1_fidelity_grid.py`, `p1_fidelity_ab.py`
- phase arms: `d138-phase-f8-l0`, `d138-phase2-f8-l0` (six-way split),
  `d138-phase2-f16-l0` (f16 body comparison), `d138-phase-neutral-l0`
  (neutrality check, `23.66 tok/s`)
- phase probe env: `GGML_TRACE_FATTN_PHASE=1` + `GGML_HIP_DISABLE_GRAPHS=1`
  (the symbol read synchronises the device, so graph capture must be off); the
  probe body itself now needs the compile-time `GGML_D138_PHASE`, because its
  per-iteration cost is not neutral (see the probe tax above)
- P1 arms: `p1-ctrl-a` / `p1-gqa-a` / `p1-gqa-b` / `p1-ctrl-b` (first L0 pass,
  broken index, probe compiled in), `p1b-{ctrl,gqa,ctrl}` (L0 A-B-A, fixed),
  `p1c-{ctrl,gqa,ctrl,gqa2}` (L2 A-B-A-B, fixed, probe compiled in),
  `p1d-{l2,l0}-{ctrl,gqa}` (probe still in -> poisoned), `base-head-{l0,l2}`
  (same-hour HEAD reference build), `p1e-{l2,l0}-{ctrl,gqa}`,
  `p1f-{l0-gqa,l2-gqa2,l2-ctrl2}` (**clean set, pre-promotion**),
  `p1g-{l2-ctrl-nofa,l2-gqa-nofa}` (FLASH_ATTN_EXT ablation - note that the
  ablated decode came out *slower*, so that probe is not usable for FA timing on
  this lane), `p1h-{l2-def-a,l2-off,l2-def-b,l0-f16,l0-def,l0-off}`
  (**post-promotion A-B-A set, default vs `GGML_ROCM_FATTN_F8_GQA_COLS=0`**)
- MMVQ probes: `c1-l2-{ctrl-a,rows1,ctrl-b,rows1-b,ctrl-c,rows1-c}` (Q4_K
  `small_k` flip; the `-a` arm carries the `d138-bytes-l2`-style `SMALL_K`
  trace), `c2-l2-{ctrl-a,inv,ctrl-b}` (invert for `q5_K`/`iq4_xs`, rejected),
  `d138-bytes-l2` (**byte census with graphs disabled**, 124968 traced
  launches), `d138-devtiming-l2` (**device-timing reference**, `25.31 t/s`,
  dev0 `16.9` / dev1 `15.4 ms` medians)
- split experiments (all rejected): `c3-l2-layer` (`25.92`), `c3-l2-row`
  (killed in prefill at 8192/30609), `c3-l0-row` (prefill `25.9 tok/s`,
  decode `1.38 t/s`)
- MTP lane (2026-09-26): `d138-mtp-l0-{none,n3,n3b,noneb,trace}`, `d138-mtp-l2-{none,n3}` — the GGUF is MTP-enabled, `bench2 --spec mtp --spec-n 3`
- MTP prompt-path probe (2026-09-26): `scripts/research/d138_mtp_prefill_probe.py`, logs in `build_logs/d138-mtp-probe/`
- `d138-nodetiming-l0` (`GGML_TRACE_CUDA_NODE_TIMING=1` +
  `GGML_HIP_DISABLE_GRAPHS=1` + `..._SYNC=1`, `SKIP_GRAPHS=4`, `MAX_GRAPHS=1`,
  `MIN_MS=0.0`): 14728 node rows over 168 graphs; decode graphs have
  `n_nodes=1766` (matching the device-timing key). Time columns are inflated by
  the per-node sync (graph total `159 ms` against a `37 ms` token), so only the
  structure is usable: `MUL_MAT` `241` nodes, `ADD` `84`, `FLASH_ATTN_EXT` `8`,
  `UNARY` `77`, `RMS_NORM` `102`, `GET_ROWS` `48`, `GATED_DELTA_NET` `23`,
  `ROPE` `16`, `L2_NORM` `46`, `CONCAT` `23`, `SSM_CONV` `23`, `GLU` `31`.
  Reader: `scripts/research/d138_nodes.py` (per-graph op/name breakdown)
- promotion fidelity: `build_logs/p1-prom-{off,on}.json` (~2.5k ctx) and
  `build_logs/p1-prom-len-{off,on}.json` (~15k ctx), both `48/48` equal
- P1 fidelity tools: `scripts/research/p1_fidelity_ab.py` (server A/B, token
  compare; `reps` and `ctx` arguments; `0` now means
  `GGML_ROCM_FATTN_F8_GQA_COLS=0` because the feature is default-on),
  `scripts/research/p1_fidelity_grid.py` (lane grid:
  kvt, flash-attn, devices, server path; `!VAR` in the override list unsets a
  variable)
- P1 fidelity results: `build_logs/p1-fid-{off,on}.json` (~2.5k ctx),
  `build_logs/p1-fid-len-{off,on}.json` (~15k ctx),
  `build_logs/p1-fid2-{off,on}.json` (final build, `48/48` equal),
  grid `build_logs/p1-grid-*.json`
- P1 baseline build: copied, not rebuilt: `bench2-bins/base-head-rocm72/` holds
  `build-rocm72/bin` as it was with the HEAD source (`--server-bin` points at it),
  so the reference stays runnable for the rest of the session; the earlier route
  (`git stash push -- ggml/src/ggml-cuda/fattn-wmma-f16.cu` + rebuild, grid arm
  `head-fa-f16`) still works and is what proved the output regression was mine;
  work copy `build_logs/d138-fattn-work-copy.cu`, full diff
  `build_logs/d138-work-20260925.patch` (rewritten as UTF-8 without BOM -
  `git diff > file` in PowerShell 5.1 produces UTF-16 and cannot be applied)
- standalone probes: `scripts/research/d138_fp8_wmma_density.hip` (isolated
  gfx1201 WMMA throughput: f16-fp32acc `19.84` TFLOPS, fp8-fp32acc `39.55`
  TFLOPS, f16-f16acc `20.71` TFLOPS, 16x16x16 w32) and
  `scripts/research/d138_vecdot_probe.hip`; their `.exe` builds were deleted
  with the D138 cleanup (rebuild from the `.hip` source if needed)
- probe env gates (mmvq.cu): `GGML_MMVQ_RDNA4_Q4K_ROWS1`,
  `GGML_MMVQ_MICRO_DUP=<n>`, `GGML_MMVQ_DUP_ALL=1` - **removed from the source
  on 2026-09-26 when D138 closed**, and `build-rocm72` was rebuilt so the
  binary no longer carries them. The `GGML_MMVQ_HOT_ROWS1_INVERT` gate went
  the same way. `GGML_CUDA_ABLATE` and the node/MMVQ trace gates were kept as
  general instruments (all default off)
- probe env gates (runtime_compute.inc, default off):
  `GGML_CUDA_ABLATE=<op,op>`; matching is exact for op names, substring for
  patterns containing `_`, and `name`/`name-<suffix>` for single words
- probe hook (runtime_graph.inc, active only when `GGML_CUDA_ABLATE` is set):
  a matching head node refuses its fusion so it can be ablated
- canonical history: `build_logs/bench/index.csv`

## Status: CLOSED (2026-09-26)

The program is finished. What it settled:

- **The original target (Q4_K MMVQ memory interface) is answered.** The decode
  reads `14.29 GB/token` through MMVQ at `~500 GB/s` marginal (`76-78%` of the
  isolated peak, large nodes `588 GB/s` = `91%`), the launch geometry policy in
  the tree is already the optimum, graphs/host call overhead are immaterial
  (`+1.9%`), and the perfect-kernel ceiling of the lane is `+20%` decode
  (`30.5` vs `25.3 t/s`). No single fix is left inside the MMVQ class.
- **The real lever was confirmed as MTP** (fewer work units per token):
  `44.28 t/s` at L2 vs `24.32` for `spec=none` (`+82%`), acceptance `68.1%`.
- **MTP sub-questions are closed with numbers**: tail alignment (rejected,
  `-8.5%` prefill + fatal abort at 913 rows), window `320` (L0-only, mechanism
  unknown, not hardcoded), window `512` (nothing), drafter ubatch `512`
  (`-17%`), sparse capture (default is a local optimum; `chunk 8192` is a
  cliff), verify-shape GQA packing (not worth the kernel edit).
- **Promoted and kept**: GQA head packing for the `1`-token f8 FA shape
  (`GGML_ROCM_FATTN_F8_GQA_COLS=0` is the escape hatch), and the MTP
  prefill/sparse env knobs in `tools/server/server-context.cpp`.
- **Removed with the close**: the four `mmvq.cu` probe gates, their `.exe`
  probes and the one-off scripts. Historical `tmp_d138_*.py` mentions above
  refer to those deleted one-offs (the ones worth keeping moved to
  `scripts/research/`, see the tools entry in Artifacts). `build-rocm72` was
  rebuilt after the source edit (`ninja: no work to do` on the second pass,
  `ggml-hip.dll` / `llama-server.exe` stamped `2026-09-26 13:33`), which has
  **not been smoke-tested on a GPU yet** - the first L0 run after this close
  doubles as that smoke test.
- **Out of scope by decision**: f16 KV re-bracket (VRAM budget conflicts with
  the goals) and the `320`-window mechanism (low value, no L2 gain).
- **Open issue found next to this work, not fixed here** (2026-09-26): on
  Vulkan `--spec mtp` prompt processing collapses to `~110 tok/s` on L1-L3
  (README baseline `1607 / 1580 / 1357`) while `spec=none` matches the baseline
  (`1588 / 1490 / 1275`) and decode/acceptance stay normal (`36.3-38.2 tok/s`,
  `54.9-66.7%`). The per-batch profile is a flat `~9.1 ms/token` regardless of
  batch size, i.e. the prompt is effectively serialised;
  `LLAMA_MTP_DEVICE_HANDOFF=0` restores L0 (`1318.5 tok/s`, `3029.9 ms` ttft)
  but not L1 (`108.8`), so the handoff is not the only cause. No HIP file is
  involved (`git status` shows the Vulkan tree untouched). Runs:
  `post-d138-vk-mtp`, `vk-mtp-nodevhandoff-l0`, `post-d138-vk-mtp-nohandoff`
  (stopped after L1). Diagnosis to try first: `LLAMA_SPEC_SERVER_PHASE_TIMING=1`
  + `LLAMA_SPEC_SERVER_PHASE_TIMING_MAX_ROWS=8192` on VK MTP L0-L1, then the
  same lane on the `7c49e6212` binary.

Carry-over for future performance work: decode on this lane is bounded by one
card's bandwidth until a lane gives a layer more than one row per pass, so the
next lever is acceptance/verify count, not kernel bandwidth. See
`PERFORMANCE.md` and the L1-L3 baselines in `README.md`.
