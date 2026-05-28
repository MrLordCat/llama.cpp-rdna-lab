# P003 Q4 C2 theory backlog (no prototype mode)

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Mode: analysis-only

## Goal

Build a prototype-ready theory for C2 without writing converter/runtime code.

## Candidate set

- H45: entropy-bounded nibble stream (EBNS)
- H46: superblock palette remap + escapes (SPRE)
- H47: tuple dictionary coding (PDNT)
- H48: layer-adaptive mixed policy (LAMC2) — rejected D059
- H49: context-conditioned entropy pages (CCEP) — negative fast gate D061, reformulate required
- H50: bounded-rANS deterministic micropages (BRDM) — negative fast gate D062, reformulate required
- H51: superblock graph remap + selective literal lanes (SGRL) — parked pre-gate
- H52: hierarchical tuple-context dictionary (HTCD) — parked pre-gate
- H53: nibble reordering within superblocks — rejected D063 (empirical delta=0, analytical 0 feasible)
- H54-A: TBQ-style Householder Q rotation for Q4 — rejected D065 (Shannon entropy permutation invariant)
- H54: quantization redesign for lower-entropy payloads
   - H54-B (value-aware quantization): fast gate positive in D067, move to full analytical gate
   - H54-C (TBQ+Q4 hybrid): pending
   - H54-D (random projection): pending

## Hard gates inherited from D050-D053

1. C2 must support effective payload corridor roughly `3.57-3.77 bpw`.
2. No quality-loss claim is allowed unless coding path is lossless at symbol
   level or has explicit bounded-error argument with recovery.
3. No performance-neutrality claim is allowed without decode complexity budget.
4. No prototype work until all theory checkpoints below are complete.

## Theory checkpoints

### Ck-1: Symbol entropy atlas

Status (2026-05-28):

- fast exploratory pass completed in D054 (`max_tensors=24`,
  `max_blocks_per_tensor=131072`),
- full unsampled pass completed in D055 (`max_tensors=0`,
  `max_blocks_per_tensor=0`),
- checkpoint status: closed.

Output:

- per-tensor and per-superblock nibble entropy distributions,
- entropy quantiles and worst-case tails,
- expected lower-bound bpw for EBNS-like routes.

Decision:

- reject H45 if practical bound cannot approach corridor after headers.

### Ck-2: Active-symbol/escape atlas

Status (2026-05-28):

- fast exploratory pass completed in D054,
- sampled evidence shows dominant `K_active=16` tails,
- full unsampled pass completed in D055,
- dominant `K_active=16` pattern persists on full corpus,
- checkpoint status: closed.

Output:

- active symbol count histogram per tensor class,
- expected escape rates for SPRE,
- net bpw with dictionary and escape side-overhead.

Decision:

- reject H46 if escape tails remove net benefit on dominant tensors.

### Ck-3: Tuple repetition atlas

Status (2026-05-28):

- fast exploratory pass completed in D056 (`L=2,3,4`, sampled scope),
- fixed-code+escape proxy on sampled scope did not reach corridor,
- full unsampled pass completed in D057,
- checkpoint status: closed,
- current fixed-code+escape tuple formulation is rejected as a primary route.

Output:

- tuple-frequency curves for configurable tuple lengths,
- top-K dictionary coverage,
- net bpw under realistic dictionary overhead.

Decision:

- reject H47 if dictionary route cannot beat SPRE/EBNS in net bpw-risk score.

Evidence:

- D056 sampled best modeled bpw: `L2=4.500003`, `L3=5.528656`, `L4=5.242850`.
- D057 full best modeled bpw: `L2=4.500000`, `L3=5.514632`, `L4=5.242693`.
- Both remain above corridor `3.57-3.77 bpw`.

### Ck-4: Decode complexity budget model

Status (2026-05-28):

- analytical budget model completed in D058,
- no standalone primary candidate survives combined compression+complexity gate,
- H45 retained as auxiliary-only input to Ck-5 mixed-policy stage,
- checkpoint status: closed.

Output:

- branch/table access estimate per decoded symbol,
- expected memory access pattern class (sequential/random/mixed),
- relative complexity score vs legacy Q4 decode.

Decision:

- reject any candidate that exceeds complexity budget without compensating bpw gain.

Evidence:

- H45: modeled `3.864885 bpw`, complexity index `1.512500`, decision `auxiliary_only`.
- H46: modeled `3.999999 bpw`, complexity index `1.257000`, decision `reject_primary`.
- H47: modeled `4.500000 bpw`, complexity index `1.704000`, decision `reject_primary`.

Artifacts:

- `docs/research/major-topology/D058_P003_Q4_CK4_DECODE_COMPLEXITY_BUDGET_GATE.md`
- `scripts/research/q4_c2_decode_budget.py`
- `build_logs/agent-workload/q4c2-decode-budget-qwen36-27b-q4ks-r1.q4_c2_decode_budget.json`

### Ck-5: Mixed-policy optimizer (analytical)

Status (2026-05-28):

- analytical mixed-policy optimizer completed in D059,
- no feasible candidate mixture satisfies corridor + complexity budget,
- checkpoint status: closed,
- prototype unlock for current P003 portfolio: denied.

Output:

- per-tensor assignment among remaining candidates,
- projected global bpw and overhead envelope,
- fallback share estimate.

Decision:

- produce ranked top-2 shortlist for future prototype authorization.

Evidence:

- Feasible policy count on searched grid: `0`.
- Best searched payload: `3.864885 bpw` (H45-only), still above corridor upper bound `3.7701`.
- Best searched point complexity: `1.512500`, above Ck-4 hard budget `1.35`.

Artifacts:

- `docs/research/major-topology/D059_P003_Q4_CK5_MIXED_POLICY_OPTIMIZER_GATE.md`
- `scripts/research/q4_c2_mixed_policy_optimizer.py`
- `build_logs/agent-workload/q4c2-mixed-policy-qwen36-27b-q4ks-r1.q4_c2_mixed_policy.json`

## Ranking metric (analysis phase)

Use weighted score, higher is better:

- `Score = 0.50 * Compression + 0.25 * RuntimeRisk + 0.25 * Robustness`

Where:

- `Compression`: normalized bpw gain toward corridor,
- `RuntimeRisk`: inverse complexity penalty,
- `Robustness`: stability under worst-case tails and fallback share.

## Exit criteria to unlock prototypes

All must be true:

1. At least one candidate or mixed policy reaches corridor in projection.
2. Lossless symbol semantics path is defined for dominant tensors.
3. Runtime complexity budget is inside acceptable envelope.
4. Top-2 candidate shortlist and fallback contract are documented.

Current status (2026-05-28): criteria are not met by the present H45-H47
portfolio after D058/D059. Phase 1 prototype planning remains blocked until a
new C2 hypothesis family is introduced and re-gated.

Reopen continuation (2026-05-28, D060): new-family admission gate completed.
H49/H50 are admitted to the next analytical queue; H51/H52 are parked
pre-gate. Prototype planning remains blocked.

H49 continuation update (2026-05-28, D061): first fast conditional-entropy
formulation is negative (`H1-Hcond ~= 0` on sampled scope). H49 is not closed,
but must be reformulated before any full unsampled run.

H50 continuation update (2026-05-28, D062): first fast bounded-rANS micropage
formulation is also negative (best modeled `3.882697 bpw`, feasible count `0`
under corridor `<=3.7701`). H50 is not closed, but this formulation is parked.

H53 nibble reorder update (2026-05-28, D063): empirical and analytical gates
both negative. Empirical test on 24 tensors showed `Hcond delta=0.000000 bpw`
(block boundaries dominate bigram count). Analytical gate tested 44 configs
(11 block sizes × 4 encoding methods), 0 feasible under corridor. Best overall
`N=32768, run_length, net=0.0096 bpw` — far below corridor.

**Conclusion:** Symbol-level compression on current Q4 layout cannot reach corridor.
`H1=3.864885 bpw` leaves only `~0.135 bpw` headroom before any overhead.
All C2 formulations (H45-H53) fail. Next direction: H54 — change quantization
itself to produce lower-entropy payloads.

H54-B update (2026-05-28, D067): sampled fast gate is positive.
On 24 tensors (`6,291,456` sampled elements), entropy moved from
`3.870042 -> 3.267969 bpw` (`delta=-0.602073`), with feasible tensors `24/24`
under corridor upper bound `3.77`.

H54-B representative update (2026-05-28, D068): broader spread sample also
confirms the signal. On 48 tensors (`12,582,912` sampled elements), entropy
moved `3.865866 -> 3.277582 bpw` (`delta=-0.588283`), feasible tensors `48/48`,
weighted `NRMSE=0.101195`.

H54-B wide update (2026-05-28, D069): all detected Q4 tensors were screened
with per-tensor cap (`131,072` elements). On 348 tensors (`45,613,056` sampled
elements), entropy moved `3.864270 -> 3.277495 bpw` (`delta=-0.586775`),
feasible tensors `348/348`, weighted `NRMSE=0.101327`.

H54-B final analytical update (2026-05-28, D070): explicit triple-contract gate
is now closed and PASS:

- entropy `3.277495 <= 3.7701`,
- weighted `NRMSE=0.101327 <= 0.115`,
- complexity index `1.127917 <= 1.35`.

Decision: `authorize_guarded_prototype` (not default rollout).

Immediate next step update: D071, D072, D073, and D074 are complete (guarded
plan + runtime-sidecar MVP + guarded runtime reader + guarded runtime
application proof). D074 confirms applied runtime behavior with measurable
residency shift (`forced_cpu=128`, Vulkan model buffer reduced, Host model
buffer increased). Next stage must move beyond residency redistribution to true
compressed storage/decode path that shrinks resident bytes without shifting the
same bytes to host RAM. Do not pursue further symbol-level C2 variants on
current Q4 layout.

D075 follow-up (2026-05-28): first lossless storage format prototype is complete
(`q4_metacomp_lossless_pack.py`) with exact byte-restoration verification.
Priority is now runtime integration for this quality-safe route: manifest/blob
reader + fail-closed per-entry decode path that preserves original Q4 payload
semantics before/inside backend compute routing. Do not re-open lossy Q4->Q3
runtime transcode route for P003 quality-safe target.

## Related

- `docs/research/major-topology/D053_P003_Q4_THEORY_PORTFOLIO_PREPROTOTYPE.md`
- `docs/research/major-topology/D052_P003_Q4_C2_REQUIREMENTS_TABLE.md`
- `docs/research/major-topology/D057_P003_Q4_CK3_TUPLE_ATLAS_FULL_GATE.md`
- `docs/research/major-topology/D058_P003_Q4_CK4_DECODE_COMPLEXITY_BUDGET_GATE.md`
- `docs/research/major-topology/D059_P003_Q4_CK5_MIXED_POLICY_OPTIMIZER_GATE.md`
- `docs/research/major-topology/D060_P003_Q4_C2_REOPEN_ADMISSION_GATE.md`
- `docs/research/HYPOTHESES.md`
