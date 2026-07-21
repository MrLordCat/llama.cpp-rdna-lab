---
description: "Run a fixed ROCm or Vulkan benchmark control without editing files."
agent: "bench-runner"
argument-hint: "Backend, label, lane parameters"
---
Run the requested benchmark control exactly as specified.

Recommended model class: fast executor. Escalate to a full executor only for a
reproducible anomaly or flaky lane. Do not pin a provider model ID.

Inputs from user:

- Backend: ${input:backend}
- Label: ${input:label}
- Lane/extra args: ${input:lane}

Rules:

1. Read `docs/research/PERF_WORKSPACE.md`.
2. Check no background `llama-server`.
3. Do not edit source or docs.
4. Preserve all lane parameters.
5. Report aggregate TPS, prompt tok/s, decode tok/s, errors, and artifact paths.
