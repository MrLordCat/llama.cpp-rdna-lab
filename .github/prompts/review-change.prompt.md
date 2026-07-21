---
description: "Independently review a completed change and run safe non-hardware validation."
agent: "reviewer-validator"
argument-hint: "Goal, changed files, acceptance criteria, claimed validation"
---
Review the completed change without editing it.

- Goal: ${input:goal}
- Changed files: ${input:files}
- Acceptance criteria: ${input:acceptance}
- Claimed validation: ${input:validation}

Return findings by severity with paths/lines, exact validation results, and the
remaining untested risk. Do not run GPU discovery or benchmarks.