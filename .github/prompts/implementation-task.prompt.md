---
description: "Implement one bounded change with explicit, non-overlapping file ownership."
agent: "implementation-driver"
argument-hint: "Goal, owned files, acceptance criteria, validation"
---
Implement the requested bounded task.

- Goal: ${input:goal}
- Owned files/paths: ${input:ownership}
- Acceptance criteria: ${input:acceptance}
- Required validation: ${input:validation}

Stop rather than editing outside the assigned ownership. Return the canonical
handoff from `AGENT_WORKFLOW.md`.