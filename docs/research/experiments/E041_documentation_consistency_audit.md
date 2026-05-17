# E041 Documentation Consistency Audit (C01 workflow)

## Metadata

- Experiment ID: E041
- Date: 2026-05-17
- Owner: Copilot
- Type: documentation/process audit

## Scope

Audit and align active benchmark contract across key C01 docs after E040 workflow findings.

## What was audited

Search targets:

- C01 center docs
- C01 resume/checklist docs
- benchmark methodology summary
- recent process docs

Command class used:

- text scan for `review_bug,patch_sim`, `patch_sim`, `TASKS_QUICK` references.

## Findings

1. Many references to `review_bug,patch_sim` are historical and valid for archived experiments.
2. Active contract was not clearly separated from historical records, causing ambiguity.
3. Core docs needed one explicit current-contract note.

## Updates applied

1. `BENCHMARKS.md`
   - active C01 lane wording changed to `quick triage_diff,review_bug`.
   - added note that `review_bug,patch_sim` mentions below are historical context.

2. `docs/research/decode-hotspots/C01_RESUME_PLAYBOOK.md`
   - lane contract tasks updated to `triage_diff,review_bug`.
   - resume command updated to `--task-ids triage_diff,review_bug`.
   - historical note added.

3. `docs/research/decode-hotspots/DECODE_TRACE_CHECKLIST.md`
   - added explicit section `Current quick-bench contract (2026-05-17)`.

4. `docs/research/decode-hotspots/C01_mul_mat_forward.md`
   - added contract note at top clarifying active vs historical task-set references.

## Result

- Active contract is now explicit and aligned across core workflow docs.
- Historical experiment references remain intact for reproducibility.

## Follow-up recommendation

- Keep historical experiment docs unchanged unless rewriting archived runs.
- For new experiments, use `triage_diff,review_bug` by default on `--tasks quick`.
