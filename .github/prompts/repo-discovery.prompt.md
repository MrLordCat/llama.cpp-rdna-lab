---
description: "Run a bounded read-only repository discovery before implementation."
agent: "repo-scout"
argument-hint: "Question, relevant subsystem, expected evidence"
---
Investigate one bounded repository question.

- Question: ${input:question}
- Initial scope: ${input:scope}
- Required evidence: ${input:evidence}

Do not edit or execute. Return the handoff format from `AGENT_WORKFLOW.md` with
paths, symbols, risks, and the smallest recommended change set.