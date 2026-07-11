# D078 P002 ROCm MTP Small-N DP4A MMQ Gate

Status: implemented, clean A/B passed, and promoted for RDNA4 Q3_K `N=2..4`.
`N=5` remains experimental because it did not improve the `n_max=4` lane.

## Intent

E275 moved RDNA4 Q3_K multi-column MTP verification from MMVQ to MMQ and
reached `35.58` decode tok/s in the clean dual-GPU ROCm run. E276 then showed
that removing one duplicate Q8_1 activation conversion is too small a mechanism
to explain the remaining gap. This design attacks the compute topology itself.

The current gfx1201 MMQ route always uses the RDNA4 WMMA body. Its integer
intrinsic and result tile are fixed at `16x16x16`, while Qwen3.6 MTP presents
only `N=2..5` columns. The current route therefore computes a 16-column tile
with only `12.5-31.25%` useful columns. AMD's RDNA4 WMMA guide explicitly notes
that matrices smaller than 16 must be padded.

Primary references:

- https://gpuopen.com/learn/using_matrix_core_amd_rdna4/
- https://rocm.docs.amd.com/projects/composable_kernel/en/docs-7.0.1/doxygen/html/amd__wmma_8hpp_source.html

## Lane Lock

- Model: `models/Qwen3.6-27B-Q3_K_S_mtp.gguf`.
- Backend: dual RX 9070 XT ROCm/HIP 7.1, layer split, both GPUs idle for claims.
- Context/config: `ctx=12288,b=8192,ub=1024,q8_0/q8_0`, FlashAttention.
- Speculation: `--spec-type draft-mtp --spec-draft-n-max 4`.
- Workload: `quick:triage_diff`, repo snapshot `24576` chars,
  `max_tokens=256`, thinking on, cold-first, no reuse/no prime.
- Clean comparison: E275 `35.58` decode tok/s and a new neighboring control.
- The E276 game-contended `~26.1` decode numbers are diagnostics only.

This is an opt-in speculative/session route and is not compared with the P002
spec-none 130k baseline.

## Route Evidence

- `e276-rocm-n4-trace.server.log`: warmed target verification is normally
  `N=5`, with target decode around `74-92 ms` in the synchronized trace.
- `e276-rocm-n4-mmq-components.server.log`: Q3_K gate/up MMQ calls dominate the
  relevant FFN body; GLU itself is only about `5 ms`.
- Current `mmq_get_granularity_{host,device}` returns `16` on RDNA4, and the
  selected target/draft buckets report `mmq_x_best=16`.
- The source already contains a complete Q3_K DP4A dot body and DP4A writeback,
  but compile-time `AMD_WMMA_AVAILABLE` selects the MMA loader, dot body, LDS
  layout, and writeback for every gfx1201 MMQ specialization.

## Proposed Route

Add a separate Q3_K-only RDNA4 kernel specialization rather than weakening the
global WMMA compile-time path:

1. `N<=4`: use `mmq_x=4`, four wave32 warps, DP4A Q3_K loader/dot/writeback.
2. `N=5`: start with `mmq_x=8`; keep `mmq_x=4 x 2` as a point-only alternative
   because it reloads each Q3_K weight tile for the second output-column block.
3. Preserve padded Q3_K storage handling and the existing Q8_1 activation
   format.
4. Guard with exact conditions: HIP, RDNA4, Q3_K, dense non-MoE,
   non-stream-K, `2<=N<=4`. `GGML_RDNA4_Q3K_SMALLN_DP4A=0` disables the
   route; setting it to a non-zero value also enables experimental `N=5`.
5. Keep all larger-N prompt and normal decode paths on current WMMA/MMVQ.

## Ceiling Model

`scripts/research/mtp_smalln_mmq_topology_gate.py` models padding and the
minimum effective DP4A throughput needed to break even with the 16-wide WMMA
body.

| N | Current useful columns | Candidate width | Candidate useful columns | DP4A break-even vs WMMA |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 12.50% | 4 | 50.00% | 25.00% |
| 3 | 18.75% | 4 | 75.00% | 25.00% |
| 4 | 25.00% | 4 | 100.00% | 25.00% |
| 5 | 31.25% | 8 | 62.50% | 50.00% |

The arithmetic-only ideal ceiling is `4x` at `N<=4` and `2x` at `N=5` if
DP4A had the same effective per-column throughput. That is not a speed claim.
The candidate wins only if measured effective DP4A throughput exceeds the
table's break-even after loader, barriers, and weight traffic.

## Resource Model

The model mirrors current `mmq.cuh` constants (`mmq_y=64`, Q3_K tile layouts,
four wave32 warps, 512-byte Y-tile alignment):

| Route | Dynamic LDS |
| --- | ---: |
| WMMA `x=16` | `23.56 KiB` |
| DP4A `x=4` | `18.55 KiB` |
| DP4A `x=8` | `19.06 KiB` |

The candidate lowers LDS rather than approaching the 64 KiB limit. Unknowns
that require compiled resource tracing are VGPR count, scratch, active blocks
per CU, and whether four wave32 warps remain the best DP4A workgroup.

## Rejection Analogs

- E190 rejected an MMVQ pair-dot helper. D078 differs by selecting the existing
  MMQ DP4A dataflow and reducing output-tile width; it is not another helper-only
  arithmetic rewrite.
- D013-D027 rejected large-N ROCm Q3_K selector/load changes. Their active shape
  was `N=128`; D078 is restricted to speculative `N=2..5`, where fixed-16 WMMA
  padding is the new mechanism.
- E276 rejected activation reuse as a sufficient body. D078 does not depend on
  pair fusion or persistent activation caches.

## Gate Ladder

1. Add a standalone/template point route behind an environment variable.
2. Compile only Q3_K `x=4` and `x=8` small-N variants; record VGPR/LDS/occupancy.
3. Run deterministic output comparison against WMMA at N=2,3,4,5.
4. Time the target Q3_K shapes (`m=17408/5120/6144/10240`, matching K) without
   a server lane. Reject if N<=4 is not at least `1.25x` locally or N=5 loses.
5. Only after the point gate, run one clean `max_tokens=256` control/candidate
   A/B. Promote only with unchanged acceptance/output and at least `+5%`
   end-to-end decode before r3 confirmation.

## Implementation Evidence

- The separate Q3_K DP4A loader, dot, writeback, kernel, and launcher compile
  only for `mmq_x=4/8`; the existing WMMA path remains unchanged for other
  types and shapes.
- Resource trace confirmed route activation for `N=2/4` at `mmq_x=4` with
  `18,992` bytes LDS, `226` registers, and three active blocks per CU. The
  experimental `N=5` x8 kernel used `19,520` bytes LDS and `124` registers.
- A deterministic eight-token smoke preserved the output prefix and `5/7`
  acceptance against the WMMA control. Full output can differ because DP4A
  and WMMA accumulate in a different order; lane acceptance is therefore the
  required behavioral guard.

## Measured Results

Short cold-first lane, `ctx=12288,b=8192,ub=1024,q8/q8,max_tokens=256`:

| Route | n_max | Runs | Aggregate TPS | Decode tok/s | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| WMMA control | 3 | 3 | `21.2330` | `34.9170` | `65.76%` |
| DP4A `N=2..4` | 3 | 3 | `23.4532` | `41.2505` | `64.37%` |
| spec none baseline | - | 3 | `17.61` | `25.02` | - |

The candidate improved decode by `18.14%` over the neighboring MTP control and
reached `1.65x` the same-build spec-none baseline. Aggregate wall throughput
improved `10.46%` over MTP control and about `1.33x` over spec none. The small
`1.39` percentage-point acceptance shift is not large enough to explain the
gain. A rebuilt default-path r1 reproduced `41.35` decode tok/s with no env
override, while `n_max=2` reached `39.06` and the default hybrid `n_max=4`
reached only `35.06`.

Canonical long-prompt lane, `ctx=131072,b=8192,ub=1024,q8/q8`, 56,305 prompt
tokens, 128 output tokens, cold-first:

| Route | Prompt tok/s | Decode tok/s | Aggregate TPS | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| spec none | `1088.67` | `19.02` | `2.1859` | - |
| MTP `n_max=3` | `1045.62` | `26.85` | `2.1799` | `68.55%` |
| MTP `n_max=2` | `1047.84` | `24.61` | `2.17` | see diagnostics |

At 131k, `n_max=3` still accelerates decode by `1.41x`, but the single request
is dominated by about 52-54 seconds of prefill and only 4.8-6.7 seconds of
decode. MTP does not accelerate prefill, and its roughly 4% prompt tax cancels
the decode saving for this 128-token response. The reduced long-context decode
multiplier is consistent with FlashAttention/KV work becoming a larger share;
the D078 route only accelerates Q3_K small-N matrix work.

The required speculative-model cross-check was also run. For the short lane,
using local acceptance `0.6437` and an approximate `0.96875` coverage gives
effective acceptance `0.6236`. The coverage-aware model still overpredicts the
observed `1.6487x` spec-none-to-MTP decode gain and backsolves about `0.61`
combined speculative overhead. This is a diagnostic ceiling, not an attribution
to one kernel: the remaining cost includes draft passes, attention/KV work,
scheduler synchronization, and dual-GPU transfers. The next route should trace
those centers rather than continue tuning the now-positive `N<=4` Q3_K body.

Artifacts:

- `d078-rocm-n3-control-r3.*`, `d078-rocm-n3-dp4a-r3.*`
- `d078-rocm-none-baseline-r3.*`, `d078-rocm-n3-default-r1.*`
- `d078-rocm-n4-default-hybrid-r1.*`, `d078-rocm-n2-default-r1.*`
- `d078-rocm-131k-none-r1.*`, `d078-rocm-131k-mtp-n3-r1.*`,
  `d078-rocm-131k-mtp-n2-r1.*`

## Decision And Rollback

Keep the dedicated route as the RDNA4 Q3_K default for `N=2..4`. Keep
`N=5` opt-in only; its higher DP4A break-even and x8 tile did not produce a
useful `n_max=4` gain. Set `GGML_RDNA4_Q3K_SMALLN_DP4A=0` for an immediate
runtime rollback to WMMA. No tensor format, model, or graph contract changed.
