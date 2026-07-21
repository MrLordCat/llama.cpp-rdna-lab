---
description: "Use when work involves subagents, parallel research, independent review, multiple file owners, repository cleanup, BYOK model routing, or delegated implementation."
---
# Delegated Agent Workflow

Read `AGENTS.md` and `AGENT_WORKFLOW.md` before dispatching subagents.

- The coordinating agent owns the plan, shared files, integration, validation,
  and final answer.
- Scouts are read-only unless explicit write ownership is assigned.
- Give every agent one bounded task, exact scope, acceptance criteria, and the
  handoff format from `AGENT_WORKFLOW.md`.
- One file has one writer. Do not let parallel agents edit shared integration
  files.
- Use at most two parallel scouts or two non-overlapping writers by default.
- Hardware discovery and GPU benchmarks are always sequential and have one
  owner.
- Select BYOK models by the advisory class in `AGENT_WORKFLOW.md`; do not pin
  provider-specific IDs or store credentials in the repository.
- Stop delegation when another agent would only repeat context or cannot reduce
  a named risk.