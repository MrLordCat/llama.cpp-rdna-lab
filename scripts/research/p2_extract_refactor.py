#!/usr/bin/env python3
"""P2 (D132): turn the ubatch result-extraction block in llama-context.cpp
into a reusable lambda that can be called after the tail phase of a P2
ubatch (which happens on the next loop iteration). Mechanical transform:

  1. the extraction block (from the '// Upstream-port: extract NextN
     embeddings' marker up to, but not including, the trailing
     'n_outputs_prev += n_outputs;' lines) is captured as-is and wrapped
     into a lambda named extract_ubatch_results(ubatch, n_outputs,
     n_outputs_prev, n_tokens_prev) - the parameter names match the
     variable names used inside the block, so the body needs no renaming;
  2. in the decode loop the block is replaced by a call to that lambda
     (the trailing counters are incremented right after the call).
The file is rewritten; review `git diff` before building."""
import sys
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "src" / "llama-context.cpp"
src = path.read_text(encoding="utf-8")

START = "    // Upstream-port: extract NextN embeddings (rides the same async stream as\n"
# lines we must leave in the loop after the lambda call
TRAIL = "        n_outputs_prev += n_outputs;\n"
INSERT_AFTER = "    int64_t n_tokens_prev  = 0; // upstream-port: dense row offset for unmasked nextn extraction\n"

i0 = src.find(START)
if i0 < 0:
    print("FAIL: start marker not found", file=sys.stderr)
    sys.exit(1)
i1 = src.find(TRAIL, i0)
if i1 < 0:
    print("FAIL: trailing counter lines not found after block", file=sys.stderr)
    sys.exit(1)

head = src[:i0]
block = src[i0:i1]
tail = src[i1:]
# find end of the two counter lines
j0 = tail.find(TRAIL)
j1 = tail.find("        n_tokens_prev  += ubatch.n_tokens;\n", j0)
if j1 < 0:
    print("FAIL: n_tokens_prev counter line not found", file=sys.stderr)
    sys.exit(1)
j1 += len("        n_tokens_prev  += ubatch.n_tokens;\n")
counters = tail[j0:j1]
after_counters = tail[j1:]

# insert the lambda before the do-loop
ik = head.find(INSERT_AFTER)
if ik < 0:
    print("FAIL: insert anchor not found", file=sys.stderr)
    sys.exit(1)
ik += len(INSERT_AFTER)

lam = (
    "    // P2 (D132): extraction can happen after the tail phase of an ubatch,\n"
    "    // which may run one loop iteration later (pipelined head/tail order).\n"
    "    auto extract_ubatch_results = [&](const llama_ubatch & ubatch, uint32_t n_outputs,\n"
    "            int64_t n_outputs_prev, int64_t n_tokens_prev) {\n"
    + block
    + "    };\n\n"
)

new_src = head[:ik] + lam + head[ik:] + "        extract_ubatch_results(ubatch, n_outputs, n_outputs_prev, n_tokens_prev);\n" + counters + after_counters

path.write_text(new_src, encoding="utf-8")
print("OK: lambda inserted, block replaced (%d chars)" % len(block))
