# Major Topology Program Board

This directory is for post-E264 architecture work. It is intentionally upstream
of normal E### benchmarking: use it to rank designs before editing backend code.

## Active Program

| Field | Value |
| --- | --- |
| Program | P001 Vulkan/RDNA4 Q3_K 12k dense prompt route |
| Model | `models/Qwen3.6-27B-Q3_K_S.gguf` |
| Backend | Vulkan on RX 9070 XT / AMD proprietary driver |
| Lane | `ctx=12288,b=7168,ub=1024,q4_0/q4_0,FlashAttention,spec=none` |
| Reuse/prime | off / off |
| Thinking | on |
| Current best | E257 r3 `7.0319 TPS`, prompt `999.22 tok/s`, decode `40.93 tok/s` |
| Current trace | E257 `MUL_MAT q3_K 82.71%`, `FLASH_ATTN_EXT 9.60%` |
| Near gates closed | E258, E259, E260, E264 |

## Current Rejection Fence

Do not reopen these without a new mechanism and a design note:

- Q3_K transpose-A as a single layout for both prompt and decode.
- f16 KV as the 12k dense default.
- `batch=7680`, `batch=8192`, graphics queue, `--no-mmap` transfer gates.
- `GGML_VK_DISABLE_F16=1` or broad f32acc/f16-disable pivots.
- Per-layer FFN activation casts to F16.
- Helper-only Q3_K arithmetic rewrites, pair-scale reuse, packed32 helper-only
  rewrites, nearby stride tweaks, and large current-tile variants already
  rejected by H31 history.

## Candidate Queue

| ID | Candidate | Status | Next required artifact |
| --- | --- | --- | --- |
| T1 | Dual-layout Q3_K storage: raw decode layout plus prefill layout for matrix routes | design-needed | VRAM/storage model and fail-closed tensor movement plan |
| T2 | Shader-native Q3_K prompt layout: compact signed/scale-expanded block for coopmat matmul | design-needed | instruction-count and SPIR-V/resource model |
| T3 | Fused dense FFN block: gate/up/swiglu/down tiled route | design-needed | memory-traffic ceiling model and partial-output plan |
| T4 | FA shader-body work for long-KV or secondary 12k share | parked | fresh FA route/share evidence |
| T5 | Vulkan/ROCm differential harness | tooling-needed | shape/resource comparison table for same lane |

## Required Evidence Pack

Before any source prototype, attach or link:

- paired lane control on the current binary;
- route trace for Q3_K and FA;
- `vulkan_perf_shape_summary.py` output for the active trace;
- relevant SPIR-V opcode/resource summaries;
- Amdahl/ceiling estimate for the touched route;
- correctness plan for prompt and decode routes;
- rollback path.

## Branch Discipline

- Use one branch/worktree per candidate topology.
- Keep negative prototypes out of `master` unless they are default-off diagnostic
  tools with a documented reason to keep them.
- If a candidate fails before lane A/B, write a design/scout rejection instead
  of creating a benchmark-shaped E### note.
