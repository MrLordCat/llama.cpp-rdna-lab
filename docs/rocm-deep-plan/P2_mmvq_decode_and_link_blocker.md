# P2 MMVQ Decode And Linker Blocker

## Objective

Remove the MMVQ experimentation blocker (`amdgcn-link` failures on Windows ROCm) and restore a safe path for decode-kernel optimization on Qwen-heavy quantized types.

Secondary objective: once build reliability is restored, improve decode efficiency without regressing active prompt-heavy lane stability.

## Code study map

- ggml/src/ggml-cuda/mmvq.cu
  - `get_mmvq_mmid_max_batch()` and per-arch batch tables.
  - `calc_nwarps()` architecture/type decision tables (including RDNA4-specific branch and type whitelist).
  - `mul_mat_vec_q<type, ncols_dst, ...>()` kernel template and launch bounds.
  - `mul_mat_vec_q_switch_ncols_dst()` and large `mul_mat_vec_q_switch_type()` fanout across many quant types.
  - `ggml_cuda_mul_mat_vec_q()` and `ggml_cuda_op_mul_mat_vec_q()` public dispatch entry points.
- ggml/src/ggml-cuda/mmvq.cuh
  - MMVQ max batch contract and public interface.
- ggml/src/ggml-cuda/mmq.cuh
  - Prefill-oriented MMQ context (for contrast; decode on this lane is MMVQ-driven).
- ggml/src/ggml-hip/CMakeLists.txt
  - HIP backend source wiring (`file(GLOB ../ggml-cuda/*.cu)`) and template-instance inclusion policy.

## What is currently known

- Decode path ownership is MMVQ, not MMQ, on the active lane; benchmark notes confirm MMQ traces are prefill-only while decode matvec path is `mmvq.cu`.
- Direct MMVQ tuning attempts are currently blocked by link instability; attempted MMVQ trace/nwarps knob changes failed with `amdgcn-link command failed due to signal` on `mmvq.cu`.
- Reducing type switch scope alone was not enough; restricting MMVQ switch to Qwen-relevant `Q3_K/Q4_K/Q6_K` still failed in link stage.
- P1 confirms this lane is still prefill-dominant in wall time; recent runs remain around ~69/31 prompt/decode share at `ctx=12288`, so decode-only wins have lower wall leverage than prefill-route fixes.
- Large `ubatch` does not automatically improve speed here; `ub=512` can be much slower than `ub=192` even when shape planner emits the same 192-sized prefill chunks.

## Theoretical validation in current environment

Environment assumptions:

- OS: Windows 11.
- GPU: RDNA4 (`gfx1201`).
- ROCm/HIP: 7.1 toolchain.
- Active lane: Qwen3.6-27B-Q3_K_S, `ctx=12288`, `b=6144`, `q4_0/q4_0`, no-reuse.

Representative metrics for leverage analysis:

- baseline class point (`ub192`):
  - `prompt_eval_ms ~ 9700`
  - `decode_eval_ms ~ 4320`
  - total `~14024 ms`

Approximate contribution:

- prompt share: ~69.2%
- decode share: ~30.8%

Implication for wall-speed targets:

- For +10% wall TPS, total time must drop by ~9.09%.
- If only decode improves, required decode-time reduction is about:
  - `0.0909 * (prompt + decode) / decode ≈ 29.5%`.

Interpretation:

- Decode-only path can still matter, but P2 is unlikely to deliver large wall gains on this lane unless decode improvement is very strong.
- Therefore P2 should be treated primarily as a build-unblock-and-control point first, speed point second.

## Potential verdict

Verdict: P2 has high value as an enablement point and moderate direct wall-speed probability on the active lane.

Expected bands:

- Build reliability value: high (unblocks further MMVQ research loop).
- Direct wall gain (active lane): low-to-moderate unless decode improves substantially.
- Decode-metric gain potential: moderate if targeted Qwen hot types can be tuned after TU decomposition.

## Explicit confirmation status (2026-05-11)

Confirmed now:

1. `mmvq.cu` is the decode-side tuning surface to target.
2. Link-stage blocker is reproducible enough to stop direct MMVQ iteration.
3. Switch narrowing to Qwen hot types alone does not clear blocker.
4. Active lane wall bottleneck remains prompt-heavy after P1.

Not yet proven:

1. Which exact portion of `mmvq.cu` contributes most to link pressure (type fanout, fusion variants, or launch variants).
2. Whether TU decomposition alone is sufficient without reducing template breadth.
3. Whether decode improvements from MMVQ tuning can cross meaningful wall thresholds on prompt-heavy lane.

Minimal next check before P2 implementation:

- Keep runtime behavior unchanged and run one build-only probe with synthetic decomposition scaffold (no tuning changes).
- Acceptance to start tuning implementation:
  - link succeeds in reduced and normal ROCm corridors;
  - no unresolved symbol regressions;
  - no runtime dispatch mismatch on smoke decode path.

## Pre-implementation theory gates

- Build-unblock gate: proposed split must reduce single-TU pressure in `mmvq.cu` without changing exported call surfaces.
- Dispatch-equivalence gate: type routing and `ncols_dst` dispatch behavior must remain bit-for-bit equivalent before any tuning knobs are added.
- Runtime-safety gate: no strategy may depend on disabling known-required fused/chunked prefill paths.
- Benefit gate: decode-focused target should be explicit (for example, >=15% decode eval TPS on decode-biased lane) before claiming wall impact.

If any gate fails, do not proceed to parameter tuning; keep P2 in unblock mode.

## Root-cause hypothesis

`mmvq.cu` combines large type fanout, multiple launch variants (`ncols_dst`, fusion/no-fusion, small_k paths), and architecture-specific tables in one heavy translation unit. Under current Windows ROCm toolchain this likely creates unstable link pressure; decomposition is needed before safe performance iteration.

## Solution strategy

- Translation-unit decomposition first: split dispatcher surface from kernel implementation clusters and isolate hot Qwen types from long-tail types.
- Dispatch boundary hardening: keep one thin front dispatcher preserving existing entry points and move only internal switch branches to per-group files.
- Controlled tuning surface: introduce opt-in tuning knobs only after build reliability passes and keep defaults unchanged unless experiment flags are enabled.
- Observability before optimization: add low-overhead counters/log hooks only where needed to validate decode kernel selection.

## Implementation progress (2026-05-11, Stage A+B+C+D)

Applied now (defaults unchanged; opt-in tuning hooks added):

- `ggml/src/ggml-cuda/mmvq.cu`
  - Keeps MMVQ kernels/shared utility logic.
  - Exports per-type dispatch entrypoints `ggml_cuda_mmvq_dispatch_type_<GGML_TYPE_...>(...)`.
- `ggml/src/ggml-cuda/mmvq-dispatch.cu` (new)
  - Contains `ggml_cuda_mul_mat_vec_q(...)` and `ggml_cuda_op_mul_mat_vec_q(...)`.
  - Contains lightweight `ggml_cuda_mmvq_switch_type(...)` (type routing moved out of heavy kernel TU).
  - Preserves existing dispatch contract while delegating to per-type entrypoints from `mmvq.cu`.
- `ggml/src/ggml-cuda/mmvq.cuh`
  - Added declaration for `ggml_cuda_mmvq_switch_type(...)`.
  - Added shared MMVQ type-list and per-type dispatch declarations for cross-TU wiring.
- `ggml/src/ggml-cuda/mmvq-kernels-qwen.cu` (new)
  - Contains Qwen-hot routing group (`Q3_K`, `Q4_K`, `Q6_K`) as a dedicated helper dispatcher.
- `ggml/src/ggml-cuda/mmvq-kernels-rest.cu` (new)
  - Contains non-Qwen routing group for the remaining MMVQ quantized types.
- `ggml/src/ggml-cuda/mmvq-dispatch.cu`
  - Added env-gated route observability via `GGML_TRACE_MMVQ_PATH=1`.
  - Route log includes type name, route group (`qwen-hot` / `rest`), `ncols_dst`, ids/fusion flags.
- `ggml/src/ggml-cuda/mmvq.cu`
  - Added env-gated small-k observability via `GGML_TRACE_MMVQ_SMALL_K=1`.
  - Added env-gated RDNA4 Qwen-hot small-k controls:
    - `GGML_MMVQ_QWEN_FORCE_SMALL_K=1`
    - `GGML_MMVQ_QWEN_DISABLE_SMALL_K=1`
  - Default behavior remains unchanged when these env vars are not set.

Stage A/B/C validation results:

- Reduced corridor ROCm gate:
  - `cmake -S . -B build-rocm-fa-reduced -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DGGML_HIP_QWEN_FA_REDUCED=ON -DGGML_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release`
  - `cmake --build build-rocm-fa-reduced --target llama-server -j 4`
  - Result: pass; reduced `llama-server` linked successfully.
- Normal ROCm configure/build gate:
  - `cmake -S . -B build-rocm-vec -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release`
  - `cmake --build build-rocm-vec --target llama-server -j 4`
  - Result: pass; `build-rocm-vec/bin/llama-server.exe` linked successfully.
- Incremental link-stability probes:
  - Stage A cycle: `touch mmvq.cu` + rebuild: pass.
  - Stage A cycle: `touch mmvq-dispatch.cu` + rebuild: pass.
  - Stage B cycle (after switch move): `touch mmvq-dispatch.cu` + rebuild: pass.
  - Stage B cycle (after switch move): `touch mmvq.cu` + rebuild: pass.
  - Stage C cycle: `touch mmvq-kernels-qwen.cu` + rebuild: pass.
  - Stage C cycle: `touch mmvq-kernels-rest.cu` + rebuild: pass.
  - Stage C cycle: `touch mmvq-dispatch.cu` + rebuild: pass.
  - Stage C cycle: `touch mmvq.cu` + rebuild: pass.
  - No `amdgcn-link ... signal` failure observed in these repeated MMVQ-touch cycles.
- Runtime smoke equivalence gate (active lane, 1 run):
  - Label `p2-stageA-smoke-20260511-181905-ub192-r1`
  - Aggregate TPS `8.54` (in expected `ub192` corridor).
  - Label `p2-stageB-smoke-20260511-182335-ub192-r1`
  - Aggregate TPS `8.54` (same corridor, no obvious dispatch regression).
  - Label `p2-stageC-smoke-20260511-182726-ub192-r1`
  - Aggregate TPS `8.54` (same corridor after qwen/rest partition).
  - Label `p2-stageC-reduced-smoke-20260511-183047-ub192-r1`
  - Aggregate TPS `8.54` (reduced build corridor smoke remains stable).

Stage D validation results:

- Post-hook active-lane regression gate (default behavior, 1 run):
  - Label `p2-active-lane-posthooks-20260511-184542-ub192-r1`
  - Aggregate TPS `8.55` (same active-lane corridor as Stage A/B/C smoke).
- Post-hook reduced-corridor smoke gate (default behavior, 1 run):
  - Label `p2-reduced-posthooks-20260511-184624-ub192-r1`
  - Aggregate TPS `8.55`.
- Route observability confirmation (`GGML_TRACE_MMVQ_PATH=1`, short diagnostic run):
  - Label `p2-trace-route-20260511-183846-ub192-r1`
  - Server log contains MMVQ route lines with `route=qwen-hot|rest` and type names.
  - Observed counts: `qwen-hot=1077`, `rest=0` on this Qwen workload sample.
- Small-k observability and override confirmation:
  - Baseline trace run shows `small_k=0` decisions for Qwen-hot decode-side calls (`680` lines in sample run).
  - Force trace run (`GGML_MMVQ_QWEN_FORCE_SMALL_K=1`) shows `small_k=1` decisions for the same call shape (`680` lines, override is active).
- Decode-biased lane A/B (ctx=12288, no-reuse, no real-context prefix, max_tokens=256):
  - Runs=1:
    - base: `p2-decode-lane-base2-20260511-184239-ub192-r1` -> `26.84 TPS`
    - force-small-k: `p2-decode-lane-force2-20260511-184304-ub192-r1` -> `27.09 TPS`
    - disable-small-k: `p2-decode-lane-disable2-20260511-184331-ub192-r1` -> `26.88 TPS`
  - Runs=3 confirmation:
    - base: `p2-decode-lane-base2-20260511-184406-ub192-r3` -> `26.8355 TPS`, `decode_eval_tps=28.6767`
    - force-small-k: `p2-decode-lane-force2-20260511-184451-ub192-r3` -> `27.0066 TPS`, `decode_eval_tps=28.8767`
    - delta force vs base: `+0.64%` aggregate TPS, `+0.70%` decode_eval_tps.

Interpretation:

- Stage A+B+C met the build-unblock and dispatch-equivalence intent for the scaffold split.
- Type-routing edits are now isolated in lightweight dispatch TUs (`mmvq-dispatch.cu`, `mmvq-kernels-qwen.cu`, `mmvq-kernels-rest.cu`), reducing the need to touch heavy `mmvq.cu` for routing-only experiments.
- Stage D adds explicit MMVQ route/small-k observability and an env-gated decode tuning probe without changing defaults.
- Decode gain from force-small-k is positive but modest; it does not satisfy the earlier aggressive decode target band by itself.

P2 blocker verdict:

- **Closed for blocker scope**: build/link instability blocker is removed for iterative MMVQ work, and observability/tuning hooks are in place.
- Default runtime policy remains unchanged; tuning controls stay opt-in via env vars.

## Follow-up after P2 closure (optional)

- If pursuing additional decode speedups, continue from this stable split with type-specific kernel tuning (for example, Q3_K/Q4_K/Q6_K launch policy variants) under env guards.
- Keep active-lane no-regression checks mandatory before any default policy changes.

## Validation plan (after implementation)

Build gates:

- Reduced corridor build sanity: `GGML_HIP_QWEN_FA_REDUCED=ON` path config and build complete.
- Normal ROCm build sanity: standard ROCm configure and `llama-server` build complete.
- Link stability: no `amdgcn-link ... signal` failure in repeated incremental rebuilds touching MMVQ files.

Runtime gates:

- Decode-biased lane: short incoming-context setup with larger completion budget to amplify decode share.
- Active lane regression gate: current `v2-review` prompt-heavy contract has no wall regression beyond noise.
- Metrics split: report prompt/decode separately, not aggregate only.

Acceptance criteria:

- Build reliability restored for MMVQ edits.
- Decode eval improves on decode-biased lane without active-lane regressions.
- Any wall claim on active lane must be reproducible and above noise.

## Risks

- TU split may duplicate logic and increase maintenance overhead.
- Dispatch wiring mistakes may silently change kernel selection.
- Build reliability improvements may not translate to runtime speed gains.
- Over-tuning for Qwen hot types may hurt other model families.

## Rollback criteria

- Any non-deterministic link failure remains after split.
- Any output mismatch versus baseline behavior.
- Any reproducible active-lane wall regression above noise.
- Any unacceptable complexity increase without measurable benefit.

## Open questions

- Which subset of MMVQ template instantiations contributes most to link pressure on this machine?
- Should type partitioning be static in source layout or selectable via CMake experiment options?
- Is it worth adding a decode-only microbench harness before touching runtime dispatch tables?
