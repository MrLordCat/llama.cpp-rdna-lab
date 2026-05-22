# DFlash Vendor Manifest (Bee -> llama.cpp-with-GUI)

Purpose: track exact source provenance for each DFlash-related ported chunk.

Source repo baseline:

- repo: `Anbeeld/beellama.cpp`
- branch: `main`
- baseline commit: to be pinned at implementation start

## Mandatory Port Set

| Bee source path | Local target path | Port mode | Required in phase |
| --- | --- | --- | --- |
| `common/speculative.h` | `common/speculative.h` | merge/surgical | Phase 0 |
| `common/speculative.cpp` | `common/speculative.cpp` | merge/surgical | Phase 0 |
| `common/arg.cpp` | `common/arg.cpp` | merge/surgical | Phase 0 |
| `src/models/dflash_draft.cpp` | `src/models/dflash_draft.cpp` | new file | Phase 1 |
| `src/llama-context.h` | `src/llama-context.h` | merge/surgical | Phase 1 |
| `src/llama-context.cpp` | `src/llama-context.cpp` | merge/surgical | Phase 1 |
| `src/llama-cparams.h` | `src/llama-cparams.h` | merge/surgical | Phase 1 |
| `src/llama-graph.h` | `src/llama-graph.h` | merge/surgical | Phase 1 |
| `src/llama-graph.cpp` | `src/llama-graph.cpp` | merge/surgical | Phase 1 |
| `src/llama-arch.cpp` | `src/llama-arch.cpp` | merge/surgical | Phase 1 |
| `src/llama-model.cpp` | `src/llama-model.cpp` | merge/surgical | Phase 1 |
| `include/llama.h` | `include/llama.h` | merge/surgical | Phase 1 |
| `tools/server/server-context.cpp` | `tools/server/server-context.cpp` | merge/surgical | Phase 2 |
| `tools/server/server-adaptive-dm.h` | `tools/server/server-adaptive-dm.h` | new file or inline port | Phase 3 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `ggml/src/ggml-cuda/ggml-cuda.cu` | merge/surgical | Phase 2 |
| `ggml/src/ggml-cuda/cross-ring-interleave.cu` | `ggml/src/ggml-cuda/cross-ring-interleave.cu` | merge/surgical | Phase 2 |
| `ggml/src/ggml-cuda/argmax.cu` | `ggml/src/ggml-cuda/argmax.cu` | merge/surgical | Phase 2 |

## Optional / Deferred Port Set

| Bee source path | Local target path | Why optional |
| --- | --- | --- |
| `common/download.h` / `common/download.cpp` | same | optional DFlash draft auto-discovery UX |
| `tests/test-dflash-ring.cpp` | `tests/test-dflash-ring.cpp` | useful but can be replaced by local unit test style |
| `tests/test-dflash-plumbing.cpp` | split across local tests | too broad as-is; use as checklist for targeted tests |

## Provenance Rules

1. Every DFlash implementation PR must record Bee source commit hash.
2. Every copied or heavily adapted chunk should include a short comment in PR description:
   - source path;
   - source line range;
   - adaptation summary.
3. Keep this manifest updated when source anchors move.

## License/Attribution Notes

1. If `common/int32-map.h` or suffix-tree code is imported from Bee stack, verify Apache-2.0 attribution chain and add/update license notices accordingly.
2. Do not import Bee docs/workflows into protected local paths.
