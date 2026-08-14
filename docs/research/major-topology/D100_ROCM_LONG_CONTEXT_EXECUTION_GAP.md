# D100: ROCm long-context execution gap

Date: 2026-08-13

Status: complete measured diagnosis. Default-off tracing is retained; no
runtime or kernel performance candidate is accepted.

## Objective

Explain and reduce the long-context decode gap between production Windows
ROCm/HIP and Vulkan on two RX 9070 XT GPUs without weakening D098/D099
native FP8 correctness, MTP acceptance, placement, or driver safety.

D100 first separates these possible costs:

1. HIP graph misses, updates, capture, or host launch overhead;
2. the serial two-device layer boundary and host-staged copies;
3. long-KV FlashAttention work;
4. target/draft MTP work or synchronization;
5. another measured operation family.

No broad graph or RDNA4-policy refactor starts before a current production-lane
trace establishes a recoverable wall-time ceiling above local noise.

## Locked lane

- model: `models/Qwen3.6-27B-Q4_K_M.gguf`;
- Windows ROCm/HIP 7.1, `gfx1201`, Ninja release build;
- `-dev ROCm1,ROCm0 -sm layer -ts 1,1`, default output on ROCm0;
- `b8192/ub1024`, one slot, FlashAttention on;
- `f8_e4m3/f8_e4m3`, cold/no-reuse/no-prime/no-warmup, `-fit off`;
- `triage_diff`, seed 42, 256 output tokens;
- 49K first gate and 98K long-distance confirmation;
- matched `spec=none` and MTP n2 controls. MTP n3 is a separate production
  profile and must not be substituted into an n2 bracket.

Historical anchors, requiring fresh adjacent controls:

| Context | Backend | Mode | Prompt tok/s | Decode tok/s | Acceptance |
| ---: | --- | --- | ---: | ---: | ---: |
| 49K | ROCm | FP8 MTP n2 | 1751.58 | 41.98 | 83.16% |
| 98K | ROCm | FP8 MTP n2 | 1482.86 | 35.82 | 81.25% |
| 49K | Vulkan | FP8 MTP n2 | 1679.76 | 44.51 | 67.59% |
| 98K | Vulkan | FP8 MTP n2 | 1510.95 | 41.79 | 73.79% |

## Existing graph evidence

The HIP graph path is not starting from zero:

- `GGML_HIP_GRAPHS` is enabled by default;
- graph instances use a stable shape/data fingerprint and distinguish PP/TG
  compute buffers;
- property equality, warmup, replay, update fallback, and timed LRU eviction
  already exist;
- `GGML_TRACE_CUDA_GRAPH_STATE=1` reports compatibility, property changes,
  warmup, replay, and update state;
- `GGML_TRACE_CUDA_GRAPH_HOST_TIMING=1` separates keying, compatibility,
  property checks, and graph evaluation/launch host time.

E294 confirmed replay on both GPUs. E296 measured about `0.917 ms` and
`0.632 ms` of steady HIP graph launch-path host time for the two splits;
compatibility scans were `0.035-0.041 ms`. E307's `0.42-0.49 ms` values were
individual warm FlashAttention launches inside a synchronization-distorted Q3
trace, not whole-token compute time.

Therefore stable caching and generic PP/TG templates are not assumed missing.
D100 first verifies whether Q4/FP8 target and draft graphs reuse the current
facilities at N=1 and N=2-4.

## Gate ladder

### G0: clean controls

1. Confirm no active `llama-server`.
2. Run an untraced 49K FP8 `spec=none` control.
3. Run the matched MTP n2 control.
4. Preserve prompt/decode/aggregate TPS, acceptance, placement, and graceful
   shutdown. Run 98K only after G1 selects a plausible long-distance mechanism.

### G1: non-synchronizing graph census

Run the same lane with graph-state and host-submit tracing only. Do not enable
per-node synchronization timing. Summarize per device and graph class:

- calls, keys, and node counts;
- replay, warmup, capture/update, and property-change counts;
- host submit p50/p95/total;
- target TG N=1, target verify N=2-4, and draft-context calls;
- graph launches per returned token and per accepted token;
- cross-device copy/wait counts where existing traces identify them.

If graph classes are ambiguous, add a default-off phase/class tag to the
existing trace before changing graph behavior.

### G2: ceiling decision

Graph/dispatch source work requires at least one of:

- steady target/draft graphs repeatedly miss replay or reset warmup;
- capture/update/property work consumes at least 10% of decode wall time;
- graph launch plus scheduler/boundary overhead has a modeled recoverable
  ceiling above 3% wall and a concrete reduction in submissions or waits.

A persistent N=1 or N=2-4 template is justified only for an observed unstable
class. If replay is stable and the known two-launch cost remains near 1.5 ms,
D100 pivots to long-KV FA or target/draft topology.

### G3: one bounded prototype

Select exactly one mechanism from G2:

1. eliminate an avoidable submission or host wait while preserving dependencies;
2. keep a proven target/draft graph class warm without broad lifetime changes;
3. reduce a measured long-KV FA work/traffic center;
4. reduce duplicated target/draft work proven by the phase trace.

The prototype remains default-off until focused correctness, route proof, and
one-run adjacent A/B pass. Remove a negative prototype.

### G4: confirmation

A kept change requires deterministic correctness, unchanged MTP acceptance
outside deterministic variation, no FP8 fallback or placement drift, no prompt
regression above 2%, at least 3% decode improvement (or a lower-noise paired
result), 98K confirmation, graceful shutdown, and `git diff --check`.

## RDNA4 policy follow-up

Policy consolidation is deferred until D100 keeps a mechanism. Split it into:

- kernel policy: MMQ, MMVQ, FlashAttention, small-N;
- runtime topology policy: graph reuse, copies, scheduler residency, placement.

Extraction must be behavior-neutral, compile for gfx1201 and compile-only
gfx1100, and migrate one family at a time.

## Safety

- Never set `LLAMA_OUTPUT_DEVICE=ROCm1` for the production order.
- Never enable direct peer copy from a performance trace.
- Never call `hipMemGetInfo` while a GPU workload is active.
- Never hard-kill a server during load, prompt evaluation, or decode.
- Never use synchronized per-node traces for absolute wall claims.
- Preserve D098 rollback controls and D099 portable fallback behavior.

## First command

```bash
python scripts/agent_workload_bench.py --label d100-g0-rocm49k-fp8-none-r1 --server-bin build-rocm/bin/llama-server.exe --model models/Qwen3.6-27B-Q4_K_M.gguf --cache-type-k f8_e4m3 --cache-type-v f8_e4m3 --flash-attn --no-warmup --ctx-size 49152 --batch-size 8192 --ubatch-size 1024 --max-tokens 256 --task-ids triage_diff --task-hard-timeout 900 --request-timeout 900 --startup-timeout 900 --real-context-mode repo-snapshot --real-context-chars 147456 --no-reuse --no-v2-prime-pass --server-extra "-dev ROCm1,ROCm0 -sm layer -ts 1,1 --spec-type none -fit off" --tasks quick
```

The G1 diagnostic uses the same command and a distinct label with
`GGML_TRACE_CUDA_GRAPH_STATE=1 GGML_TRACE_CUDA_GRAPH_HOST_TIMING=1`.
Do not enable `GGML_TRACE_CUDA_NODE_TIMING_SYNC` in G1.

## Activity log

### 2026-08-13: G0 deferred under background load

The first untraced 49K control reached the end of its 30,663-token prefill but
received an external stop before producing a benchmark result. The harness
completed graceful server cleanup. A game was active in the background, so
this partial run and any TPS derived from it are invalid for D100 decisions.

No additional GPU benchmark will run until the background workload is gone.
Offline work may continue on default-off trace fields, the trace summarizer,
compilation, and static review.

### 2026-08-13: G0-G2 execution ceiling

After the GPU became free, adjacent untraced 49K controls established the
current lane:

| Mode | Prompt tok/s | Decode tok/s | Aggregate TPS | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| `spec=none` | 1814.35 | 22.66 | 9.05 | n/a |
| MTP n2 | 1746.27 | 38.89 | 10.57 | 76.62% |

The graph census found stable target and draft replay, not repeated capture or
shape churn: 470 of 548 observed MTP graph calls replayed; the remaining 78
were expected direct PP/growing-shape calls. Whole-graph asynchronous HIP
event timing then separated host submission from device execution without a
per-node synchronize:

- 49K `spec=none` N=1 p50 was `18.829 ms` on ROCm1 and `19.642 ms` on ROCm0;
- 98K `spec=none` N=1 p50 rose to `21.329/22.192 ms`;
- 49K MTP N=3 target-verify p50 was `22.449/25.583 ms`;
- the ROCm0 draft/logit N=1 graph was only `2.353 ms` p50;
- steady host replay remained about `0.6 ms` per graph launch, and the two
  layer splits differed by less than one millisecond in the none lanes.

Three negative event intervals reported by the Windows HIP runtime were
explicitly rejected by the summarizer and do not enter device totals. Device
totals are sums across per-device graph intervals, not decode wall time.

This closes the graph-overhaul gate: stable graph reuse already exists, host
replay is below the `3%` admission ceiling, and the long-distance growth is in
device work. Persistent N=1/N=2-4 templates or broad graph-update changes are
not justified by this lane.

### 2026-08-13: G3 full-FP8 FlashAttention topology probes

The measured full-native D256 N=1 launch uses `ncols=16`, eight warps,
`parallel_blocks=8`, 24 output tiles, and 192 blocks: exactly three full
64-block waves per GPU at 49K. Bounded alternatives did not pass:

| Probe | Result | Decision |
| --- | --- | --- |
| 16-wave native body | `22.965 -> 22.54 tok/s` (`-1.85%`) | removed |
| 8-column rocWMMA body | compile-time input-fragment register-layout rejection | removed; no binary |
| 32 KV slices | `22.90 -> 22.59 tok/s` (`-1.35%`) | removed |
| 4 KV slices | auto `22.94`, forced `23.01`, auto `22.94` (`+0.31%`) | noise; removed |

The current eight-slice heuristic is therefore locally sound. More waves add
work; fewer slices save fixup work but lose enough occupancy to be neutral.
No experimental environment override remains.

### Closure

D100 retains only default-off observability:

- `GGML_TRACE_CUDA_GRAPH_DEVICE_TIMING=1` for asynchronous whole-graph device
  intervals;
- `GGML_TRACE_FATTN_LAUNCH_CONFIG=1` for exact FA grid/occupancy geometry;
- `scripts/research/rocm_graph_trace_summary.py` for state, host, and device
  summaries with invalid-interval filtering.

No production execution policy changes. A successor must begin from a new
device dataflow/kernel mechanism with a modeled `>=3%` ceiling; graph-template,
host-submit, generic warp-count, narrow-tile, and KV-slice sweeps are closed by
this evidence.
