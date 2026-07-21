---
description: "Plan and execute a multi-part repository task using bounded subagents when they reduce a concrete risk or rate-limit cost."
argument-hint: "Goal, constraints, acceptance criteria, validation"
---
Act as the coordinating agent and follow `AGENTS.md` plus
`AGENT_WORKFLOW.md`.

- Goal: ${input:goal}
- Constraints / protected scope: ${input:constraints}
- Acceptance criteria: ${input:acceptance}
- Required validation: ${input:validation}

Required flow:

1. Inspect the dirty worktree and perform one common discovery pass.
2. Decide whether delegation materially reduces a named risk or uncertainty.
3. If useful, dispatch at most two independent read-only scouts with compact
   context and the canonical handoff format.
4. Record non-overlapping file ownership before any parallel edit.
5. Keep shared integration files with the coordinator.
6. Integrate and inspect the complete diff.
7. Use an independent reviewer for material changes and a single sequential
   owner for any hardware/GPU work.
8. Return outcome, validation evidence, remaining risk, and next action.

Select advisory model classes at dispatch time; do not pin provider IDs or
store BYOK configuration in the repository.