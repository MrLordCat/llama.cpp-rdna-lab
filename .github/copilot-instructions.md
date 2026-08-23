# llama.cpp-rdna-lab instruction router

**This worktree (`D:\GitHub\llama.cpp-gui2`, branch `gui-2.0`) is the GUI 2.0
rewrite. Read `GUI2.md` first — it overrides the checkout location and task
scope described in `AGENTS.md`.**

Then read and follow `AGENTS.md`.

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