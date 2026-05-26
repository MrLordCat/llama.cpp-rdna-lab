# E251 Q4 MTP Fit-Auto Cold Gate

## Metadata

- Experiment ID: E251
- Date: 2026-05-25
- Owner: Copilot
- Branch/Commit: local `master` after `b4f2fddf5`
- Target lane: Qwen3.6-27B cold-first 12k repo-snapshot lane, `ctx=12288`, KV `q4_0/q4_0`, `triage_diff`, no reuse, no prime, thinking on.

## Hypothesis

- Statement: if the Q4 MTP model cannot full-fit with `-ngl 999`, omitting `-ngl` may let server fit placement choose a usable GPU/CPU split while MTP still raises cold aggregate above 10 TPS.
- Mechanism: prior history had Q4 MTP cold wins, but current full-offload fit fails due VRAM projection. Auto placement might trade a small CPU offload penalty for enabling the MTP path.
- Risk: CPU layer fallback can dominate prompt/decode and erase MTP gains.

## Benchmark Plan

- Run `Qwen3.6-27B-Q4_K_S.gguf` with `--spec-type mtp --spec-draft-n-max 3`, `batch=4096`, `ubatch=512`, no reuse, no prime.
- Use `--gpu-layers -1` so `agent_workload_bench.py` omits `-ngl` and allows server fit behavior.

## Result

- Forced-default runner behavior check: omitting `--gpu-layers` still passed `-ngl 999`, so the server failed fit before readiness.
- Correct fit-auto run with `--gpu-layers -1`: `e251-q4mtp-cold-fitauto-nglminus1-b4096ub512-r1` completed in `54.61 s`, `64` completion tokens, aggregate `1.17 TPS`.

## Decision

- Reject as a cold 10 TPS route. The fit-auto CPU/GPU placement makes the Q4 MTP path far slower than the Q3 cold baseline and cannot serve as a pragmatic target escape.
- Continue H42 kernel/body work for the Q3 cold lane; do not repeat Q4 MTP fit-auto unless VRAM availability or model placement changes materially.

## Artifacts

- `build_logs/agent-workload/e251-q4mtp-cold-fitauto-b4096ub512-r1.server.log`
- `build_logs/agent-workload/e251-q4mtp-cold-fitauto-nglminus1-b4096ub512-r1.diagnostics.md`
- `build_logs/agent-workload/e251-q4mtp-cold-fitauto-nglminus1-b4096ub512-r1.server.log`
