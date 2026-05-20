---
description: "Update research docs from existing benchmark artifacts and experiment decisions."
agent: "research-docs"
argument-hint: "Experiment ID, artifact labels, decision"
---
Update research documentation from existing measured artifacts.

Inputs from user:

- Experiment ID: ${input:experiment}
- Artifact labels/files: ${input:artifacts}
- Decision: ${input:decision}

Required updates:

1. Experiment note under `docs/research/experiments/`.
2. `docs/research/RESULTS_LOG.md`.
3. `docs/research/HYPOTHESES.md` if priority/status changed.
4. `BENCHMARKS.md` only if the result changes defaults, presets, or future user-facing guidance.

Do not invent metrics. Mark projected values separately from measured values.
