# E058 Small-Gain Runtime Recheck

## Metadata

- Experiment ID: E058
- Date: 2026-05-18
- Owner: Copilot
- Branch/Commit: local `master`
- Target lane: Qwen3.6-27B-Q3_K_S cold-first prefill lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on.

## Why

The acceptance policy changed after E056: a same-session r3 `~0.5-1.5%` aggregate gain can be useful as a stackable opt-in knob even if it is not a standalone default win. This means old `noise-level` runtime rejections should be rechecked when they are low-risk and easy to reproduce.

## Candidates

- `ROCBLAS_USE_HIPBLASLT=1`: E048 showed `+0.10%` r1 on the current `ubatch=2048` lane.
- `GGML_CUDA_NO_PINNED=1`: E044 showed `+0.11%` r1 on the old `ubatch=192` C01 lane.
- `ROCBLAS_USE_HIPBLASLT=1 + GGML_CUDA_NO_PINNED=1`: E044 showed `-0.11%` on the old lane, but the current large-prefill lane has a different route mix.

## Benchmark Plan

1. Run same-session default control r3.
2. Run each candidate r3 with the same lane contract.
3. Keep only as opt-in/profile recommendation if r3 shows a positive aggregate signal without decode/prompt regression large enough to erase stack value.

## Results

- Control r3: `prefill-e058-control-r3 = 11.6132 TPS` aggregate by wall time; mean task TPS `11.6187`, median `11.7103`, stdev `0.2731`.
- hipBLASLt r3: `prefill-e058-hipblaslt-r3 = 11.6618 TPS`; mean `11.6637`, median `11.6655`, stdev `0.1634`, aggregate delta `+0.42%` vs E058 control.
- no-pinned r3: `prefill-e058-nopinned-r3 = 11.5714 TPS`; mean `11.5728`, median `11.6000`, stdev `0.1402`, aggregate delta `-0.36%`.
- combo r3: `prefill-e058-hipblaslt-nopinned-r3 = 11.5915 TPS`; mean `11.5938`, median `11.6415`, stdev `0.1757`, aggregate delta `-0.19%`.
- Diagnostics:
	- control prompt/decode: `1196.73 / 29.31 tok/s`.
	- hipBLASLt prompt/decode: `1202.23 / 29.36 tok/s`.
	- no-pinned prompt/decode: `1197.78 / 28.97 tok/s`.
	- combo prompt/decode: `1201.37 / 28.97 tok/s`.
- Decision: do not promote any runtime knob. hipBLASLt remains a watchlist candidate because aggregate was `+0.42%`, but median task TPS was below control and the control run had a slow outlier. Under the small-gain policy this is not strong enough for even an opt-in recommendation.

## Audit Notes

- Rechecked now because they were cheap, no-code, and previously rejected as `noise-level`: E048 hipBLASLt, E044 no-pinned, E044 hipBLASLt+no-pinned.
- Not rechecked in this pass:
	- E018/E020/E032 were on the old closed C01 geometry or were trace-only with target/activation caveats.
	- E046/E047/E050/E051/E052/E056/E057 already had clear same-lane negative evidence.
	- E028/E029/E030 ngram work was already kept as opt-in and does not need a reject-policy recheck.