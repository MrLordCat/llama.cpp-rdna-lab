# D097: Q4_K_M Vulkan FP8 long-context acceptance recovery

Status: accepted. Context-scoped FP8 N=12 policy implemented and validated;
q8 and shorter FP8 contexts remain N=8.

## Objective

Explain and recover the `ctx=98304` MTP acceptance loss of the native P5
`f8_e4m3` KV profile without hiding prompt, decode or memory trade-offs. Keep
q8_0 as the production fallback until a matched long-context bracket proves a
better FP8 policy.

## Locked lane

- model: `models/Qwen3.6-27B-Q4_K_M.gguf`;
- backend: `Vulkan1,Vulkan0`, layer split `1,1`;
- context/batch: `98304`, `8192/1024`;
- workload: full `triage_diff`, repo snapshot 224000 chars, 57530 measured
  prompt tokens, cold/no-reuse/no-prime;
- output: 256 tokens for the diagnostic sweep;
- speculation: draft MTP, `n_max=2`;
- q8 and FP8 both use the same contiguous last-N-f16 full-attention KV policy;
- P5 is enabled only for FP8; no queue/mmap or placement change is mixed in.

## Existing evidence

The 128-token documentation refresh measured:

| KV | Last f16 | Prompt tok/s | Decode tok/s | Acceptance | Aggregate TPS |
|---|---:|---:|---:|---:|---:|
| q8_0 | 8 | 1440.02 | 37.79 | 73/106 (68.87%) | 2.9407 |
| f8_e4m3 P5 | 8 | 1465.98 | 32.40 | 64/124 (51.61%) | 2.9494 |

The loss is numerically plausible. On captured Q/K/V from two real prompts,
raw E4M3 attention-logit MSE is `0.0030764424`; q8_0 is `0.0001028896`, about
29.9x lower. E4M3 has three mantissa bits, while q8_0 has an 8-bit integer and
a scale per 32 values. FP8 range is not the same as FP8 precision.

The small MTP prompt gap has a separate mechanism. Spec-none FP8 beats q8 by
12.6% at 98K, but the MTP hybrid changes half of the 16 full-attention KV
layers to f16. FP8 therefore loses its P5 advantage on half of the FA graph;
q8 simultaneously avoids its preconvert cost on those layers.

## Hypotheses

1. The 128-token acceptance delta is partly trajectory noise, but a longer
   256-token adjacent bracket will retain a material FP8 deficit.
2. Long-KV quantization error from the eight remaining FP8 attention layers is
   the primary deficit. Increasing contiguous last-f16 depth should recover
   acceptance, although the curve may be non-monotonic for one generation.
3. A useful point exists before full f16: acceptance within 2 percentage points
   of q8, prompt at least 3% above q8, and lower main KV allocation.

## Gate ladder

1. Run q8 N=8, FP8 N=8, then a closing FP8 N=8 control at 256 output tokens.
2. Sweep FP8 N=10/12/14/16 between the N=8 controls. Record prompt, decode,
   aggregate TPS, accepted/generated drafts and main KV MiB.
3. Promote the smallest N that is reproducible and meets all three hypothesis-3
   thresholds. If none does, do not force an FP8 MTP default at 98K.
4. Only after a passing point, add a context-scoped default in `common.cpp`
   with the existing `LLAMA_VK_MTP_KV_LAST_F16` environment variable as an
   explicit override/rollback.
5. Rebuild and run adjacent q8/current-FP8/candidate controls. Output must be
   sane, no server may survive, and `git diff --check` must pass.

If depth cannot recover acceptance without erasing the prompt/memory benefit,
the next design is a default-off mixed q8/FP8 bridge or the D095 R9 K-scale
sidecar. Neither is admitted by this gate without a new resource/lifecycle
proof.

## Root cause

The longer 256-token bracket made the result deterministic: both FP8 N=8
controls accepted exactly `140/230` drafts (`60.87%`), while both q8 N=8
controls accepted `151/208` (`72.60%`). This is not thermal noise or a shader
selector regression. It is accumulated KV precision error in the eight
quantized full-attention layers that remain ahead of the f16 tail.

The format names are misleading if interpreted as precision ranks. Raw E4M3
has only three explicit mantissa bits and no local scale. q8_0 stores a signed
int8 value with one f16 scale per 32 values. On captured real K/Q blocks q8_0
therefore has about 29.9x lower attention-logit MSE than raw E4M3. FP8's
advantage is native P5 execution and one-byte payload, not higher effective
precision than block-scaled q8_0.

The MTP prompt gap has a separate cause: hybrid N=8 turns half of the 16
full-attention layers into f16 for both formats. P5 can accelerate only the
remaining FP8 layers, so its spec-none 98K prompt lead is diluted.

## Depth and bridge gates

The explicit FP8 depth sweep used the locked 57,530-token/256-output lane:

| FP8 policy | Prompt tok/s | Decode tok/s | Acceptance | Main KV MiB |
|---|---:|---:|---:|---:|
| last 8 f16, control A | 1471.04 | 37.46 | 140/230 (60.87%) | 4608 |
| last 10 f16 | 1489.03 | 35.77 | 132/245 (53.88%) | 4992 |
| last 12 f16 | 1509.36 | 42.27 | 152/206 (73.79%) | 5376 |
| last 14 f16 | 1533.79 | 43.67 | 155/199 (77.89%) | 5760 |
| last 16 f16 | 1557.41 | 45.88 | 157/194 (80.93%) | 6144 |
| last 8 f16, control B | 1468.93 | 38.04 | 140/230 (60.87%) | 4608 |

N=12 is the smallest point that restores q8-level acceptance. N=10 proves the
curve is not monotonic enough to infer from layer count without a real run.

A default-off q8 bridge was also implemented with
`LLAMA_VK_MTP_KV_Q8_BEFORE_F16=M`. M=6 leaves two early layers in FP8, uses six
q8 layers before the last-eight f16 tail, and reaches `164/181` (`90.61%`)
acceptance, `47.05` decode tok/s and 4680 MiB. It is a valid generation-heavy
research profile, but prompt throughput is only `+0.33%` over the q8 control
center, so it is not the default fix for the combined acceptance/prompt goal.

## Final adjacent bracket and decision

| Policy | Prompt tok/s | Decode tok/s | Aggregate TPS | Acceptance | Main KV MiB |
|---|---:|---:|---:|---:|---:|
| q8 N=8, control A | 1430.89 | 42.09 | 5.5039 | 151/208 (72.60%) | 4704 |
| FP8 N=12 | **1510.95** | 41.79 | **5.7618** | **152/206 (73.79%)** | 5376 |
| FP8 q8-bridge M=6 + N=8 | 1427.45 | **47.05** | 5.5687 | 164/181 (90.61%) | **4680** |
| q8 N=8, control B | 1414.53 | 41.83 | 5.4432 | 151/208 (72.60%) | 4704 |

Against the two-control q8 center, FP8 N=12 is `+6.20%` prompt, `-0.41%`
decode, `+5.27%` aggregate and `+1.19` acceptance percentage points. It costs
672 MiB (`+14.29%`) more main KV than q8; this memory cost is explicit and is
the reason the policy is scoped to long-context FP8 MTP.

`common_context_params_to_llama()` now selects last-12 f16 only when K/V are
both `f8_e4m3`, draft MTP is active and `n_ctx >= 98304`. q8 and shorter FP8
contexts keep last-8. Any explicit `LLAMA_VK_MTP_KV_LAST_F16`, including `0`,
still wins over the automatic policy.

The rebuilt automatic run logged N=12 and reproduced `152/206` acceptance at
57,656 prompt tokens (`1473.74/42.78`, aggregate `5.6569`). Separate startup
smokes proved 49K auto-N8 and 98K explicit-N8 rollback. Artifacts:
`d097-98k-*`, `d097b-98k-*`, `d097c-98k-*`,
`d097-auto-98k-f8-n12-mt256` and `d097-smoke-*`.