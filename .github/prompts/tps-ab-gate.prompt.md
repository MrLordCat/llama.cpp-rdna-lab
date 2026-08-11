---
description: "Run a focused TPS A/B gate for a llama.cpp-rdna-lab performance hypothesis."
agent: "tps-research"
argument-hint: "Hypothesis ID, baseline label, candidate change, lane"
---
Run a focused TPS A/B gate for the requested hypothesis.

Recommended model classes: long-context workhorse for initial synthesis; full
executor for the source patch. When subagents are available, hand the fixed
hardware lane to `bench-runner` and the completed diff to
`reviewer-validator`. Do not pin a provider model ID.

Inputs from user:

- Hypothesis / experiment ID: ${input:hypothesis}
- Candidate change: ${input:candidate}
- Lane: ${input:lane}

Required flow:

1. Read `docs/research/PERF_WORKSPACE.md` and the relevant hypothesis/experiment note.
2. Check `git status --short --branch`.
3. Confirm no background `llama-server`.
4. Run a neighboring baseline and candidate with identical lane shape.
5. Default to `--runs 1`; use `--runs 3` only if the result is borderline or promising.
6. Decide keep / iterate / revert.
7. Update experiment docs and `docs/research/RESULTS_LOG.md` if code or measured evidence changed.

Report commands, metrics, artifacts, and decision.
