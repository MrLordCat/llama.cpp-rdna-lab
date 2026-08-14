# D103: ROCm P*V phase instruction/register cost

Date: 2026-08-14

Status: deprioritized (2026-08-14). Superseded by the R001 RDNA4
architecture track: per-kernel micro-gates on the FA phase cannot carry the
program while ~55% of the 49K decode token remains unmeasured. The known
bounds below stay valid if the track is resumed.

## Objective

D102 measured the full-native D256 decode kernel: P*V owns `55.4%`, KQ
`28.3%`, softmax+P requant `14.3%`, merge `2.0%`. Both MMA phases read the
same 128 KiB K/V tile per chunk and issue the same number of fp8 x fp8 MMAs,
yet P*V costs twice KQ. The extra cost is the fp32 VKQ-part materialization
through LDS plus the rescale merge.

## Known bounds (from D098 bisect and D102 math)

- LDS bandwidth is not the bottleneck on gfx1201: the fp32 store roundtrip
  is structural, and the f16-store variant lost 6.5% because the conversion
  copies cost +32 VGPR on a spill-free 154-156 VGPR kernel. Registers are
  the constraint.
- Register-resident fp32 accumulators are impossible: the running VKQ sum is
  ~16 KiB per warp (64 VGPR/lane) on top of an already full register file.
- The 49K decode share of the FA kernel is roughly 20% (derived from the
  D101 full-native 22.94 versus KQ-only 19.60 bracket). A 10% P*V-phase cut
  is worth only ~1.1% decode at 49K; the 3% gate needs ~30% phase cut.
- At 98K the FA share roughly doubles with the KV, so the same prototype is
  worth about twice as much there. Candidates are validated on the 98K lane.

## Open questions (G0)

- How much of the 55.4% phase is the store/merge instruction stream versus
  the P/V fragment loads and MMA issue? The census can be extended with
  finer markers if needed.
- Can the merge be fused with the final store so the VKQ parts never round
  trip LDS in fp32 (direct fragment store into the rescaled f16 VKQ2 while
  keeping the conversion in the existing register budget)?
- Does trimming `D_padded` and vectorizing the merge loads/stores clear the
  gate at 98K even without removing the roundtrip?

## Fences

- Wave count, cols-per-block and parallel-KV slices are closed by D100.
- Phase rollback and combine scalars are closed by D101.
- The f16 VKQ store is closed by D098 (register regression).
- A prototype must pass focused correctness, exact-route proof, a same-binary
  A-B-A on the 98K lane with the `>=3%` decode gate, and 49K regression
  confirmation.
- The census tool (`GGML_ROCM_FATTN_PHASE_CENSUS`) may be reused for
  pre-prototype evidence; it is default-off and never part of a TPS claim.

## Lane

Same locked lane as D102: `Qwen3.6-27B-Q4_K_M`, f8 KV, `ctx=49152`,
`b=8192/ub=1024`, `-dev ROCm1,ROCm0 -sm layer -ts 1,1`, `triage_diff`,
seed 42, `spec=none`, cold/no-reuse/no-prime/no-warmup, `-fit off`.
98K confirmation uses `ctx=98304` with the same recipe.
