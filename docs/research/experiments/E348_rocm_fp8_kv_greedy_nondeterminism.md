# E348 ROCm fp8-KV Greedy Nondeterminism vs MTP Agent-Loop Attribution

## Trigger

User report: `Qwen3.8-27B-UD-Q4_K_M` in an agent session (llama-vscode-chat
"AI bridge") started repeating itself; a single run with MTP disabled did not
loop, so MTP was suspected. Hard requirement from the user: **the production
server always runs with fp8 KV** (`--cache-type-k f8_e4m3 --cache-type-v
f8_e4m3`), so any conclusion must hold on that lane.

Server under test: `build-rocm-linux/bin/llama-server`, `b9492-a771fdbd0`,
tree `4097e7726`; `ctx=151552`, `b8192/ub1024`, `--parallel 1 --kv-unified
-ngl 999 --metrics --cache-ram 8192 --ctx-checkpoints 8
--checkpoint-every-n-tokens 4096`, `--mmproj mmproj-F16.gguf`,
`--flash-attn on`, `-dev ROCm1,ROCm0 -ts 61,39`; KV `f8_e4m3/f8_e4m3`.

## Method

New tool `scripts/research/mtp_identity_probe.py` (added by this experiment):

- starts exactly one server per lane, sequentially, and stops it with SIGTERM;
- sends two greedy `/completion` probes on a fixed 262,378-char repo prompt
  (~74K tokens; prompt and corpus SHA-256 are recorded in the JSON report) and
  one greedy chat+tools probe;
- optional deterministic multi-turn tool loop (`--agent`): scripted tool
  results and deterministic `tool_call_id` (`call_<turn>_<idx>`) so every lane
  sees byte-identical prompt bytes - the first version echoed the server's
  random ids and produced prompt differences that masked real divergence;
- `--lanes none,none,...` repeats a lane in one run, which is the null test for
  "is this lane reproducible at all";
- all lanes default to `f8_e4m3` KV.

Greedy settings: `temperature 0`, `top_k 1`, `top_p 1`, `min_p 0`, `seed 0/0`
(agent: `seed=turn`), `cache_prompt true`. Within every single server process
the two back-to-back probes returned bit-identical text, so cross-request
determinism is not the question; only process-to-process reproducibility is.

## Results

Prompt hash `243c8e9b0766761c` and corpus hashes were identical in all four
runs. `A`/`B` are the two greedy completions, `t0`/`t1` the agent-loop turn
hashes.

| run | lane | A / B | agent t0 / t1 |
| --- | --- | --- | --- |
| pair | `none` | `6e1ecfb6` / `6e1ecfb6` | `32f6242a` / `b1bf4b38` |
| pair | `mtp` | `2dabfc07` / `2dabfc07` | `32f6242a` / `e1e2446d` |
| ladder2 | `none` | `ffaa2a2e` / `ffaa2a2e` | `32f6242a` / `33d266ce` |
| ladder2 | `mtp` | `2dabfc07` / `2dabfc07` | `32f6242a` / `e1e2446d` |
| ladder2 | `mtp-nohybrid` | `9d558bcd` / `9d558bcd` | `32f6242a` / `31110164` |
| ladder2 | `mtp-nosparse` | `0078727e` / `0078727e` | `32f6242a` / `70063a5f` |
| nulltest | `none` | `6e1ecfb6` / `6e1ecfb6` | `32f6242a` / `b1bf4b38` |
| nulltest | `none#2` | `6e1ecfb6` / `6e1ecfb6` | `32f6242a` / `7e249ed9` |
| nulltest | `mtp-nohybrid` | `9d558bcd` / `9d558bcd` | `32f6242a` / `31110164` |
| nulltest | `mtp-nohybrid#2` | `9d558bcd` / `9d558bcd` | `32f6242a` / `31110164` |
| handoff | `none` | `34af43f7` / `34af43f7` | `32f6242a` / `3f0ea467` |
| handoff | `mtp-nohandoff` | `1556c78d` / `1556c78d` | `32f6242a` / `1776a578` |
| handoff | `mtp-nohandoff#2` | `0078727e` / `0078727e` | `32f6242a` / `71940a7c` |

Findings:

1. **Rare, one-off cross-process drift exists** in `spec=none` too (three
   different greedy texts for one byte-identical prompt and one byte-identical
   command line: `6e1ecfb6` in pair/nulltest, `ffaa2a2e` in ladder2, `34af43f7`
   in handoff), and `mtp-nohandoff` once gave `1556c78d` then `0078727e` within
   a single run. **This was NOT reproduced in the dedicated clean re-run** (see
   below), where every lane returned bit-identical results across two processes.
   Treat it as an environment-sensitive event, not yet a code defect.
2. **The reproducible signal is the MTP numeric change itself**: `spec=none`
   (`6e1ecfb6`), MTP with the fork hybrid tail (`2dabfc07`) and MTP with
   `LLAMA_VK_MTP_KV_LAST_F16=0` (`9d558bcd`) each reproduce across processes and
   each differ from the others.
3. **The diffs are coherent paraphrases**, not garbage: e.g. control
   "performs speculative verification by sampling tokens from the target model
   at the specified batch indices" vs MTP "performs speculative-decoding
   verification by sampling the target model's token at each position". That is
   the signature of a last-bit logit perturbation flipping a near-tie under
   greedy decoding.
4. **MTP silently changes KV precision on this lane.** With fp8 KV and MTP the
   fork enables the hybrid tail at startup:
   `MTP + quantized KV: enabling hybrid KV cache (last 12 layers f16,
   LLAMA_VK_MTP_KV_LAST_F16=12)` (`src/llama-kv-cache.cpp:168-182`,
   `common/common.cpp:1575-1593`). The log line is absent from the
   `LLAMA_VK_MTP_KV_LAST_F16=0` lane, confirming the knob. A "MTP on vs off"
   comparison with fp8 KV therefore also changes the target's own KV precision
   for the last 12 layers - it is not a clean MTP-only A/B.
5. Lane cost on this 74K prompt: MTP prefill 971 tok/s vs 1123 tok/s
   (`-13.5%`), decode 28.4 vs 20.0 tok/s (`+42%`).
6. `--flash-attn off` cannot be used to isolate the FA kernels on a quantized
   KV lane: the server refuses to load (`server did not become ready` for both
   lanes; the CLI warns "only last value will be used" for the repeated flag).

## Dedicated clean re-run (copies hypothesis) - run matrix

`--lanes none,none,none-copies2,none-copies2,mtp-copies2,mtp-copies2`, same
74K prompt, one server process per lane, no other server running, no bind
failures:

| lane | process | greedy A/B | agent turn 1 | decode tok/s | prefill tok/s |
| --- | ---: | --- | --- | ---: | ---: |
| `none` | 1 | `6e1ecfb6` | `b1bf4b38` | 19.76 | 1124.6 |
| `none` | 2 | `6e1ecfb6` | `b1bf4b38` | 19.79 | 1119.8 |
| `none-copies2` | 1 | `6e1ecfb6` | `f1ba1a65` | 19.78 | 1146.8 |
| `none-copies2` | 2 | `6e1ecfb6` | `f1ba1a65` | 19.78 | 1146.9 |
| `mtp-copies2` | 1 | `2dabfc07` | `e1e2446d` | 25.99 | 950.4 |
| `mtp-copies2` | 2 | `2dabfc07` | `e1e2446d` | 25.85 | 980.0 |

Conclusions:

1. **`n_copies = 1` is NOT the cause of the drift.** `copies=2` reproduces bit
   for bit and returns the *same* greedy text as the `copies=1` control
   (`6e1ecfb6`); `mtp-copies2` returns the same text as the default MTP lane
   (`2dabfc07`). The scheduler-copy default does change the agent-loop numeric
   path (`f1ba1a65` vs `b1bf4b38`), so it is a real behavioural knob - but it
   does not destabilise greedy output.
2. **MTP vs no-MTP divergence is deterministic and unrelated to rolling
   drift**: `2dabfc07` (MTP, hybrid 12) vs `6e1ecfb6` (spec=none) vs
   `9d558bcd` (MTP, `LLAMA_VK_MTP_KV_LAST_F16=0`) reproduce across processes.
3. The earlier single-process outliers (`ffaa2a2e`, `34af43f7`, and
   `mtp-nohandoff` giving `1556c78d` then `0078727e`) were **not** reproduced in
   this clean run and must not be over-claimed. Their possible cause is external
   GPU state (VRAM pressure / another server or display workload present while
   that run was active), not a proven code defect. The tool now refuses to run a
   lane when the probe port is already in use, so this class of contamination
   cannot silently invalidate a lane again.

## Confirmation by the reported session (user test, same day)

Running the actual agent session with `LLAMA_VK_MTP_KV_LAST_F16=0` (MTP still on,
`--spec-type draft-mtp --spec-draft-n-max 2`, fp8 KV, ctx 151552) removed the
symptom: the model used its tools and answered normally. So the three-arm test
resolved to arm (c):

| arm | target KV | session behaviour |
| --- | --- | --- |
| `spec=none` | pure f8_e4m3 | fine |
| MTP, auto hybrid (fork default) | last 12 layers f16, rest f8 | loops / ignores tools |
| MTP, `LLAMA_VK_MTP_KV_LAST_F16=0` | pure f8_e4m3 | fine |

The defect is therefore the **hybrid KV tail**, not the speculative loop. What
the hybrid costs and what it was supposed to buy, measured on this machine at
`ctx=151552` (`/tmp/mtp_fp8_ladder2-mtp*.log`):

| lane | target KV buffers | decode | prefill |
| --- | --- | ---: | ---: |
| MTP auto hybrid (12 f16 layers) | 3552 + 4736 MiB = 8288 MiB | 28.38 | 971.6 |
| MTP, `LLAMA_VK_MTP_KV_LAST_F16=0` | 1776 + 2960 MiB = 4736 MiB | 28.24 | 1106.6 |

The tail is `1.75x` the KV bytes (12 layers at 2 B + 4 at 1 B against 16 at 1 B)
and costs `+3552 MiB` VRAM and `-13.5%` prefill. It buys the draft head
"full-precision attention context" - the stated rationale in
`src/llama-kv-cache.cpp:166-172` is that the MTP draft reads the last
transformer layer's KV, and the auto-enable in `common/common.cpp:1575-1595`
turns it on whenever MTP meets quantized KV without an explicit env value.

That rationale was established on the Vulkan lanes (D096/D097/W30, MXFP4 and Q3
prompts). On ROCm the draft head does not read the target's K/V through that
path: the fork's default is device handoff of the NextN hidden state
(`LLAMA_MTP_DEVICE_HANDOFF`, HIP default on - see D094-c8j, where the trace
shows `src=ROCm1 dst=ROCm1 same_device=1` rows and the Vulkan lane showed zero
handoff rows), and the draft context keeps its own 592 MiB cache. So on ROCm the
f16 tail is charged to the target and never read by the draft - and in exchange
it changes the target's own attention numerics mid-cache, which is what the
agent session trips over.

Also relevant: the `k_scale` satellites added by D131 R9 exist only for f8 K
layers (`src/llama-kv-cache.cpp:462-466`, `type_k_il == GGML_TYPE_F8_E4M3`), so
the hybrid creates a cache where 12 of 16 layers carry scales and 4 do not.

## Verdict

The cause is identified: **the hybrid f16 KV tail that the fork turns on by
itself when MTP meets a quantized KV cache.** It is not the speculative accept
loop, and it is not "f16 being less accurate" - it is an f16 tail changing the
target's attention numerics for the last 12 of 16 KV layers, charged at
`+3552 MiB` VRAM and `-13.5%` prefill, on a backend where the draft head does not
read target K/V through that path.

Three numerically distinct configurations, each reproducible, each with its own
greedy continuation:

| configuration | greedy output | meaning |
| --- | --- | --- |
| `spec=none`, fp8 KV | `6e1ecfb6` | control |
| MTP, hybrid tail (fork default) | `2dabfc07` | target KV changed |
| MTP, `LLAMA_VK_MTP_KV_LAST_F16=0` | `9d558bcd` | MTP path changed, KV control |

"Without MTP the agent does not loop" did not isolate MTP: it also isolated the
hybrid KV change that MTP drags in. The user's three-arm session test resolved
it in favour of arm (c) - hybrid off, MTP on.

Fix options, cheapest first:

1. **stop auto-enabling the tail on ROCm** (`common/common.cpp:1575-1595`): the
   env override and the rollback contract stay, the Vulkan lanes that measured
   the win keep it, and the ROCm lanes stop paying for a tail their draft head
   never reads. Not applied yet on purpose - the running A/B must keep meaning
   what it means, and a default change mid-experiment would silently redefine
   arm (b);
2. audit the hybrid path itself (`src/llama-kv-cache.cpp:432-447`) for the
   layer-indexing and `has_kv`/filter interactions, and for what a mid-cache
   type change does to the FA route and to the R9 `k_scale` satellites;
3. nothing to fix in `common/speculative.cpp`: acceptance and the accept loop
   are downstream of the target's logits, and the loop reproduces with
   `spec=none` too (as a different continuation, not a different mechanism).

## Next

1. Decide on (1) above and apply it with the env rollback kept, then re-run the
   three arms to confirm the symptom does not return.
2. Repeat `none,none` on `q8_0` KV: `kv_q8` is in the same auto-enable branch, so
   the same tail is applied there; compare with/without the knob.
3. Re-run the drift check whenever GPU state differs (another server running,
   display workload, background game) - the earlier outliers all came from runs
   whose environment was not controlled; a lane is only trustworthy when the
   probe port was verified free and no other model server exists.
4. Keep the prefill cost in view: the fp8 lane's whole point is prompt
   evaluation, and the tail costs `-13.5%` of it.

## Artifacts

`/tmp/mtp_fp8_pair.{json,out}`, `/tmp/mtp_fp8_ladder2.{json,out}`,
`/tmp/mtp_nulltest.{json,out}`, `/tmp/mtp_handoff.{json,out}`,
`/tmp/mtp_no_fa.{json,out}` (FA-off negative), per-lane server logs
`/tmp/mtp_*<lane>.log`.
