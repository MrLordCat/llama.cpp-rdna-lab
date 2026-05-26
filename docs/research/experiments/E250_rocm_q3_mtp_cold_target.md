# E250 ROCm Q3 MTP Cold Target Gate

## Metadata

- Experiment ID: E250
- Date: 2026-05-25
- Owner: Copilot
- Hypothesis ID: H03 / MTP route
- Target lane: cold-first repo-snapshot task, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, no reuse, no prime, thinking on
- Goal: test whether the local MTP-enabled `Qwen3.6-27B-Q3_K_S_mtp.gguf` can reach the current user target of `10 TPS` cold-run.

## Hypothesis

- Statement: the current cold lane is prompt-heavy but not purely prompt-bound; with post-E248 timing around `5.95 s` prompt and `2.06 s` decode for 64 generated tokens, a high-acceptance MTP route can cross `10 TPS` if it cuts enough decode work without adding large prompt overhead.
- Mechanism: MTP verifies multiple predicted tokens per target decode step, so the decode portion can shrink even when the prompt/prefill path is unchanged.
- Why now: local `models/Qwen3.6-27B-Q3_K_S_mtp.gguf` exists, and H03/E060 already showed MTP can work on a different Qwen MTP model/config. The new target is the active 12k cold Q3 lane.

## Gate

- Control A: MTP GGUF with `--spec-type none` on the exact cold lane.
- Candidate B: same MTP GGUF with `--spec-type mtp` and a small draft cap.
- Strong pass: candidate aggregate >= `10 TPS`, no prompt blow-up, and MTP acceptance high enough to explain the wall result.
- Weak pass: candidate improves materially but stays below `10 TPS`; keep as opt-in and tune draft length.
- Reject: candidate regresses wall time or MTP overhead dominates.

## Commands

- Use `scripts/agent_workload_bench.py` with `--runs 1` for first gate.
- Use `--no-reuse --no-v2-prime-pass --no-disable-thinking --write-diagnostics`.
- Keep task/max-token/context settings identical between A and B.

## Result

- Outcome: rejected for the local `Qwen3.6-27B-Q3_K_S_mtp.gguf` cold route.
- Control A, exact active shape `b=6144,ub=2048`, `--spec-type none`:
	- hard task timeout at `30.01 s` before prompt eval completed;
	- log reached `prompt processing progress, n_tokens = 6144 / 7489` and produced no prompt/decode metrics.
- Candidate B, smaller MTP-friendly probe `b=4096,ub=512`, `--spec-type mtp --spec-draft-n-max 3`:
	- MTP head registered successfully;
	- hard task timeout at `30.00 s` before prompt eval completed;
	- log reached `prompt processing progress, n_tokens = 4096 / 7489`, then rebuilt target graph reserve (`3849` nodes) and did not finish inside the hard timeout.
- Q4 MTP sanity attempt with `Qwen3.6-27B-Q4_K_S.gguf`, `b=4096,ub=512`, `--spec-type mtp --spec-draft-n-max 3` did not start under the current VRAM/free-memory state because fit-on refused the projected `15644 MiB` device use with `-ngl 999`.

## Decision

- Reject the local Q3_MTP GGUF as a route to the `10 TPS` cold target on this lane. It is slower than the target by timeout before generation starts, so high acceptance cannot help.
- Do not promote MTP for Q3 cold-first runs. MTP remains opt-in and model/config-specific; future MTP work needs either a lighter compatible MTP GGUF or a separate startup/prompt-overhead fix before speed A/B.
