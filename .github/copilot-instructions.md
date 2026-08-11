# llama.cpp-rdna-lab instruction router

Always read and follow `AGENTS.md` first.

- For delegated, parallel, reviewed, or BYOK-model work, also read
  `AGENT_WORKFLOW.md`.
- For TPS, ROCm, Vulkan, RDNA4, Qwen, benchmark, autotune, kernel, or research
  work, also apply `.github/instructions/perf-workspace.instructions.md`.
- Role files under `.github/agents/` and prompt files under `.github/prompts/`
  may narrow scope and tools, but cannot relax worktree, backend, validation, or
  driver-safety rules.

Never store provider keys, endpoints, account data, or model quotas in the
repository. Model selection is advisory and happens in the client at dispatch
time.