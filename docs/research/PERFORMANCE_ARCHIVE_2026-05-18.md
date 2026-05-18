# Performance R&D Archive - 2026-05-18

## Status

The current Qwen3.6/RDNA4 acceleration cycle is archived. There is no active default performance branch after E059.

This is a pause point, not a claim that no future speedup exists. Reopen only when at least one of these changes:

- upstream lands a relevant RDNA4/HIP/MMQ/MTP change with evidence for the same shape class;
- an MTP-enabled Qwen3.6 GGUF is available for measured wall-time validation;
- the benchmark lane, model, quant, or route mix changes materially;
- a new design passes a preflight gate with a plausible `>=2%` wall ceiling and is not a duplicate of a rejected experiment.

## Final Active Lane

- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- Backend/build: ROCm/HIP `build-rocm-vec`, target `gfx1201`
- Workload: `scripts/agent_workload_bench.py --tasks quick --task-ids triage_diff,review_bug`
- Shape: `ctx=12288`, `batch=6144`, `ubatch=2048`
- KV: `q4_0/q4_0`
- Speculation: `--spec-type none`
- Reuse: off (`--no-reuse`, `--cache-ram 0 --ctx-checkpoints 0`)
- Thinking: on (`--no-disable-thinking`)
- Iteration policy: `--runs 1` for screens, `--runs 3` only for promising or borderline confirmation.

Canonical command shape:

```bash
python scripts/agent_workload_bench.py --label <label> \
  --server-bin build-rocm-vec/bin/llama-server.exe \
  --model models/Qwen3.6-27B-Q3_K_S.gguf \
  --tasks quick --task-ids triage_diff,review_bug --runs 1 \
  --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 \
  --cache-type-k q4_0 --cache-type-v q4_0 --max-tokens 120 \
  --real-context-mode repo-snapshot --no-reuse \
  --background-server-policy fail --task-fail-timeout 0 \
  --no-v2-prime-pass --no-disable-thinking \
  --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0"
```

## Best Reference Points

- E045 current cold-first lane baseline: `11.6534 TPS`, prompt eval `1197.5567 tok/s`.
- E053 trace-off control: `11.7681 TPS`.
- E056 control r3: `11.6726 TPS`.
- E058 control r3: `11.6132 TPS`.

Use these as context, not as interchangeable baselines. Same-session A/B still wins over historical comparison.

## Kept Wins And Useful Defaults

- E008/H11: ROCm graph compute vbuffer chunking fixed the RDNA4 native `ub904/1024` residency cliff. Keep `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1` only as a negative control.
- E013/H15: RDNA4 Q3_K MMVQ decode path uses `nwarps=2` and remains kept.
- E015/H17: RDNA4 MMQ uses `mmq_y=64,nwarps=4` and remains kept.
- E028/E029/E030: `ngram-mod 24/48/64` is a practical opt-in session accelerator, not a no-spec cold-first default claim.
- E009/H12: TurboKV hybrid/direct decode work is kept for eligible TKV lanes, but q4 remains faster on the archived Q3_K lane.

## Closed Directions

Do not restart these without new route evidence:

- broad `GGML_CUDA_FORCE_MMQ_RUNTIME=1` on the archived Q3_K lane;
- Q3_K `ne11=2048` / `6144x5120@ncols2048` shape-specific MMQ override;
- Q3_K 128-thread dequant, simple half2 store packing, and explicit scalar unroll4;
- compute16/fp16 accumulation for the large Q3_K cuBLAS route;
- more generic hipBLASLt/Stream-K env sweeps;
- GDN chunk-size sweeps and fast-exp style knobs;
- C01 `ncols=192` selector/force-x/staging/load-fusion probes;
- graph-level QKV/RoPE concat/split fusion.

## Parked Leads

- H28: RDNA4 selector parity audit vs upstream #18816. Only test activated shapes where local routing differs and E050 does not already close the case.
- H29: gfx12 WMMA direct quantized prefill design gate inspired by hipfire/R9700 work. Do not code until the design explains how it beats both E049 cuBLAS split timing and E050 current MMQ timing.
- Q3_K conversion/layout remains the largest measured local cost, but simple store/thread-shape variants were not enough. Future work needs a materially different layout or fused strategy.
- GATED_DELTA_NET specialized hot-contract kernel remains possible, but chunk policy is closed.
- MTP remains separate and requires an MTP-enabled GGUF plus wall-time validation.

## Resume Protocol

1. Read this file first.
2. Check `git status --short --branch` and avoid reverting user changes.
3. If reopening C01, read `docs/research/decode-hotspots/C01_RESUME_PLAYBOOK.md` and `docs/research/decode-hotspots/DECODE_TRACE_CHECKLIST.md`.
4. If reopening the current prefill lane, read `docs/research/POST_C01_ACCELERATION_SCAN_2026-05-18.md`, E053-E059, and `docs/research/HYPOTHESES.md` H25-H29.
5. Re-establish a same-session control before making a speed claim.

## Final Decision

Archive the current acceleration work. Keep the documented wins and diagnostics, stop local low/medium-risk probing, and wait for a new external signal or a genuinely new high-ceiling design.