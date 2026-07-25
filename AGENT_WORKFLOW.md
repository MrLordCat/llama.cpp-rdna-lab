# Agent and BYOK Workflow

This document defines how a coordinating agent delegates work in
`llama.cpp-with-GUI`. It applies to local and BYOK models, custom VS Code agents,
and any client that supports subagents.

## Precedence

1. `AGENTS.md` contains mandatory repository, worktree, backend, and hardware
   safety rules.
2. Task-specific `.github/instructions/*.instructions.md` files may tighten
   those rules for a domain, but never relax them.
3. This file defines delegation, ownership, review, and model routing.
4. `.github/agents/*.agent.md` and `.github/prompts/*.prompt.md` define narrow
   roles and entry points.

When instructions conflict, the earlier item wins.

## Coordinator Responsibility

The root/coordinating agent owns the plan, architecture decisions, file
ownership, integration, final validation, and user-facing answer. Delegation
does not transfer responsibility for correctness or safety.

Before delegating, the coordinator must:

- inspect the current dirty worktree;
- state the goal and acceptance criteria;
- identify protected or shared integration files;
- give each agent one bounded question or implementation scope;
- specify whether the task is read-only or may edit files;
- actively select and pass an explicit `model` for each subagent, never `auto`;
- specify the required handoff and validation.

## When to Delegate

Delegate when at least one of these is true:

- two or more independent read-only investigations can run in parallel;
- a large repository search can be reduced to a compact evidence digest;
- implementation can be divided into non-overlapping file ownership;
- an independent reviewer materially reduces regression risk;
- a fixed benchmark or validation lane should be isolated from implementation.

Do not delegate a trivial one-file change, an unclear task, tightly coupled
edits to the same symbols, or work where the next agent would only repeat
already collected context.

## Roles

| Role | Default permissions | Responsibility |
| --- | --- | --- |
| Coordinator | read, search, edit, execute | plan, decisions, shared files, integration, final answer |
| `repo-scout` | read, search | local entry points, symbols, risks, minimal change set |
| `upstream-scout` | read, search, web | upstream commits, issues, PRs, and external evidence |
| `implementation-driver` | read, search, edit, execute | one bounded implementation with explicit file ownership |
| `tps-research` | read, search, edit, execute | performance hypothesis and owned source/config patch |
| `bench-runner` | read, search, execute | the single fixed hardware benchmark lane and artifacts |
| `research-docs` | read, search, edit | prose updates from existing measured artifacts |
| `reviewer-validator` | read, search, execute | independent diff review and non-hardware validation |

An agent may operate alone for a small task, but in a delegated workflow these
role boundaries are strict.

## Concurrency and Ownership

- Normally use the coordinator plus at most two parallel read-only scouts.
- Use no more than two implementation agents, with non-overlapping paths.
- One file has one writer. Shared headers, root CMake files, manifests, lock
  files, CI, and central GUI wiring belong to the coordinator unless explicitly
  assigned to one integrator.
- Keep one available slot for review/validation when the client permits it.
- All GPU discovery, server, and benchmark work is sequential and has exactly
  one owner: `bench-runner` or the coordinator. Read-only source research may
  continue in parallel only when it cannot touch hardware or server state.
- If ownership overlaps, the dirty worktree changes unexpectedly, or a required
  destructive action appears, stop and return the blocker.

Use an ownership table before parallel edits:

```text
Agent                  Paths / resources            Result
repo-scout             read-only                    entry-point digest
implementation-driver  gui/foo.py, tests/test_foo   patch + narrow tests
reviewer-validator     read-only diff               findings + validation
```

## BYOK Model Routing

Model neutrality applies to the stored `.agent.md` definition files, not to
dispatch. Shared repository agent files must not pin a provider-specific `model`
ID, because exact names, availability, tool support, fallbacks, context limits,
and quotas belong to the user's VS Code/BYOK configuration.

Choosing the model for each subtask is mandatory. When the coordinator dispatches
a subagent it must actively pick the model and pass it explicitly:

- Always pass an explicit `model` argument to the subagent tool. Never omit it
  and never rely on `auto`; `auto` resolves to a base free model and is not an
  acceptable routing decision.
- Match the subtask to a class in the table below, then map that class to a
  concrete model exposed by the current client at dispatch time.
- If no suitable BYOK model is available, say so explicitly and let the user
  choose, instead of silently falling back to `auto`.

| Model class | Typical local choice | Use for |
| --- | --- | --- |
| Fast executor | Codex fast/mini tier | commands, deterministic checks, triage, small mechanical edits |
| Long-context workhorse | DeepSeek V4 | broad repository reading, log synthesis, first-pass research and docs |
| Full executor | full Codex model | multi-file implementation, repo-grounded debugging, integration |
| Deep reviewer | Opus 4.8 | rare high-impact architecture, conflicting evidence, security or final design review |

Escalate only when the cheaper class cannot resolve a concrete uncertainty.
Do not request multi-model consensus by default. Opus is an escalation layer,
not a routine scout or command runner.

Never store BYOK API keys, endpoints, account identifiers, quota data, or local
provider configuration in this repository.

## Standard Delegated Cycle

1. **Discover** - coordinator performs one common search and dispatches at most
   two independent scouts with the relevant excerpts.
2. **Decide** - coordinator chooses the design and records ownership.
3. **Implement** - each writer changes only its assigned scope and runs the
   narrowest relevant checks.
4. **Integrate** - coordinator resolves shared wiring and inspects the complete
   diff without reverting user or other-agent changes.
5. **Validate** - reviewer runs non-hardware checks; `bench-runner` alone owns
   any GPU measurement.
6. **Document** - measured artifacts go to `research-docs` only after the
   acceptance decision is made.
7. **Close** - coordinator reports outcome, evidence, remaining risk, and next
   action.

## VS Code Entry Points

| Scenario | Prompt / agent |
| --- | --- |
| Plan and coordinate a multi-part task | `delegated-work.prompt.md` |
| Locate local entry points before coding | `repo-discovery.prompt.md` / `repo-scout` |
| Implement an isolated owned change | `implementation-task.prompt.md` / `implementation-driver` |
| Review a completed patch | `review-change.prompt.md` / `reviewer-validator` |
| Research upstream evidence | `upstream-scout` |
| Develop a performance candidate | `tps-ab-gate.prompt.md` / `tps-research` |
| Run a fixed hardware control | `bench-control.prompt.md` / `bench-runner` |
| Document accepted measurements | `research-doc-update.prompt.md` / `research-docs` |

The coordinator may invoke these roles directly when the client exposes
subagents. If it does not, follow the same stages sequentially in the current
agent rather than pretending delegation occurred.

## Handoff Contract

Every subagent returns:

```text
Goal: bounded question or implementation target
Scope: read paths, owned paths, and resources used
Actions: searches, edits, or commands performed
Decision: conclusion and rationale
Validation: exact commands and PASS / FAIL / SKIPPED
Evidence: paths, symbols, artifact labels, or measured values
Risks: unresolved assumptions, platforms, backends, or edge cases
Next: one concrete recommended action
```

Do not claim that something works without a command and result. A reviewer lists
findings by severity with paths/lines; if there are no findings, it states the
remaining untested risk.

## Rate-Limit Economy

- Give agents a short digest instead of the entire conversation.
- Ask one question per delegation and require a compact output format.
- Reuse an existing agent for a related follow-up instead of spawning another.
- Batch independent searches; never repeat a build or benchmark without changed
  inputs.
- Stop delegating when another agent would not reduce a named risk or
  uncertainty.
- If a subagent lacks required tools, it must return that blocker immediately.
  The coordinator may supply excerpts for analysis or continue locally; it must
  not spend repeated calls asking the same tool-less agent to inspect files.

## Performance and GPU Work

Performance tasks also inherit
`.github/instructions/perf-workspace.instructions.md`.

- `tps-research` owns the hypothesis and its source/config patch.
- `bench-runner` owns authoritative baseline/candidate execution and artifacts.
- `research-docs` owns prose derived from accepted measurements.
- `reviewer-validator` checks the diff and non-hardware acceptance criteria.
- Only the owner of a probe patch may remove it, using a narrow reverse patch;
  never use destructive checkout/reset over a dirty file.

Driver reset, OOM, unexpected Shared-memory cliff, server lifecycle anomaly, or
changed background GPU load pauses the lane and returns control to the
coordinator.