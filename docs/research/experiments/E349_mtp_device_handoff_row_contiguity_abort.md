# E349 - MTP device-handoff row contiguity aborts a long agent session

**Date:** 2026-09-14 (Linux, ROCm 10, dual RX 9070 XT)
**Status:** diagnosed, NOT fixed. Workaround available (`LLAMA_MTP_DEVICE_HANDOFF=0`).
**Build:** `build-rocm-linux`, `a771fdbd0`-era libs first seen, same on the fresh
`a6c9589f7` build.

## Symptom

A long tool-using agent session (task 6387, ~89 009 prompt tokens, `ctx 151552`,
`f8_e4m3` KV, MTP n=2, `-ts 55,45`) died with:

```
slot update_slots: id  0 | task 6387 | n_tokens = 88757, memory_seq_rm [88757, end)
slot update_slots: id  0 | task 6387 | prompt processing done, n_tokens = 89009, batch.n_tokens = 252
spec      process: non-contiguous MTP device rows: first=1 current=0 batch_row=196
srv  update_slots: failed to process speculative batch
srv  update_slots: no tokens to decode      (x4)
tools/server/server-context.cpp:3144: fatal error - please provide logs and repro in
https://github.com/ggml-org/llama.cpp/pull/20277
```

Backtrace ends in `server_context_impl::update_slots()` -> `ggml_abort()`, so the
crash is the abort, not a GPU fault or an OOM.

## Failure chain

1. `common_speculative_process()` builds the draft catch-up batch row by row in
   device-handoff mode. Staging row 0 holds the **pending** continuation captured
   by the previous decode; the rows produced by this batch start at staging row
   one, so a target row `k-1` is staging row `k`
   (`common/speculative.cpp:1735-1757`).
2. The code requires the device rows of one batch to increase by exactly one per
   added row (`common/speculative.cpp:1760-1768`). The check runs **after**
   `common_batch_add()` and `return false`s on violation, leaving the batch
   half-built and `first_added_pos`/`selected_end` already mutated.
3. The violated case is a mixture: `first=1` (the batch started at staging row 1,
   i.e. the pending row was *not* its first row) and then the pending row
   (`current=0`) appeared at `batch_row=196`. Row 0 is physically the first
   staging slot, so it can never satisfy `first + batch_row`.
4. `process_ok == false` makes `update_slots` `break` out of the batch loop with
   `batch.n_tokens == 0` (`server-context.cpp:3263-3266`).
5. The empty-batch guard then counts four consecutive empty batches and aborts
   (`server-context.cpp:3140-3146`). That guard is upstream's, and its message
   asks for a repro in upstream PR #20277 - it converts a stuck server into a
   crash on purpose. **Nothing is wrong with the guard; the fork's row mapping
   is what fails.**

## Evidence

1. **Pre-existing, not from the f16 KV tail change.** The check arrived with
   `41a8ca785 perf(mtp): keep NextN hidden states on device` (2026-07-14). The
   error string appears in probe logs written *before* the E348 rebuild of the
   same day (`/tmp/mtp_fp8_ladder2-mtp.log` and friends), and the KV tail only
   chooses tensor types in `src/llama-kv-cache.cpp`.
2. **It is the device-handoff path, not the sparse-prefill policy.** One
   occurrence per MTP lane in every probe run that had handoff on, zero in every
   lane that set `LLAMA_MTP_DEVICE_HANDOFF=0`, and it still fires with sparse
   prefill fully disabled (`LLAMA_SPEC_PREFILL_WINDOW=0` +
   `LLAMA_MTP_DEFER_SPARSE_PREFILL=0`):

   | lane env | logs | "non-contiguous" hits |
   | --- | --- | ---: |
   | default (handoff on, sparse on) | `mtp_fp8_ladder2-mtp`, `mtp_fp8_pair-mtp`, `mtp_copies-mtp-copies2`, `mtp_nulltest-*`, `mtp_agent_ladder-mtp`, `mtp_pair-mtp` | 1 each |
   | `LLAMA_SPEC_PREFILL_WINDOW=0` + `LLAMA_MTP_DEFER_SPARSE_PREFILL=0` | `mtp_fp8_ladder2-mtp-nosparse`, `mtp_agent_ladder-mtp-nosparse` | 1 each |
   | `LLAMA_VK_MTP_KV_LAST_F16=0` | `mtp_nulltest-mtp-nohybrid*`, `mtp_fp8_ladder2-mtp-nohybrid` | 1 each |
   | `LLAMA_MTP_DEVICE_HANDOFF=0` | `mtp_handoff-mtp-nohandoff*`, `mtp_agent_ladder-mtp-nohandoff`, `mtp_ladder-mtp-nohandoff` | **0** |

3. **The trigger is the ordinary prompt-cache tail trim.** The
   `n_tokens = %d, memory_seq_rm [%d, end)` line is emitted at
   `tools/server/server-context.cpp:2881` right before truncating "any tokens
   that are beyond n_past for this slot" (`:2873`). In this session the new
   request shared 88 757 tokens with the cached prefix, so the server trimmed
   from there and re-processed 252 tokens - a plain agent turn, not an exotic
   state. Every turn that diverges mid-prefix passes through it, which is why the
   probe hits the error once per server too.
4. **The probe reproduces it deterministically** (73.8K prompt + scripted turns,
   once per server) but the server then recovers, because a fresh batch can still
   be built. The user's session died because after the trim the slot had nothing
   left to build a batch from, so the empty-batch guard ran out its four
   attempts.

## Cost of the only working workaround

`LLAMA_MTP_DEVICE_HANDOFF=0` from `/tmp/mtp_handoff.json` (same 73 816-token
prompt, same contract, handoff on vs off):

| lane | turn-0 wall | turn-1 wall |
| --- | ---: | ---: |
| `none` (no MTP) | 66.82 s | 6.00 s |
| `mtp-nohandoff` | 77.39 s | 4.48 s |
| `mtp-nohandoff#2` | 77.38 s | 7.04 s |

Disabling handoff costs about **+15.8% on the long prompt** (66.8 -> 77.4 s
against the no-MTP control), which is the metric this fork cares most about, so
the workaround is for availability only.

## Fix proposal (not applied)

Move the row decision in front of `common_batch_add()` and stop treating the
pending row as fatal:

1. **Prefer the staging row when it exists.** For `k > 0` where the input row
   `k-1` is the same sequence and adjacent in position, the staging row already
   holds the hidden state of exactly that token - the same token the pending row
   refers to, because both branches require `pending_pos == batch_in.pos[k-1]`.
   Choosing `device_row = k` therefore keeps the content and the contiguity.
   Only when no staging row exists (`k == 0`, or the previous row was filtered
   out) is row 0 the only option.
2. **Skip, do not fail, when only the pending row is left** and the batch already
   has rows: `continue` with a trace line, exactly like the other "row not
   available" path. The draft then misses one position for that step - the same
   trade the sparse window already makes - while the target still verifies.
3. **Keep the hard error** for any other non-contiguity so a real invariant
   violation is not masked.

Validation harness: `scripts/research/mtp_identity_probe.py`, lane
`mtp` (handoff on, sparse on). Gate: zero `non-contiguous MTP device rows`
occurrences and byte-identical greedy outputs against the recorded values
(`mtp` lane completion hashes in `/tmp/mtp_fp8_pair.json`, `/tmp/mtp_copies.json`),
plus no decode regression.

## Artifacts

- user crash log: pasted in the session, sequence quoted above
- `/tmp/mtp_*.log` lane logs (occurrence counts above), `/tmp/mtp_handoff.json`
- code: `common/speculative.cpp:1735-1768`, `tools/server/server-context.cpp:3140-3146`,
  `tools/server/server-context.cpp:3263-3267`
