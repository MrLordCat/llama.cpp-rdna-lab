# E271 Vulkan Single/Dual No-MTP Smoke

## Metadata

- Experiment ID: E271
- Date: 2026-07-09
- Owner: Codex
- Target lane: Vulkan single/dual GPU smoke without MTP,
  `Qwen3.6-27B-Q3_K_S_mtp.gguf`

## Setup

Vulkan `llama-server` was rebuilt on the current branch after fixing a MinGW
link issue in the DFlash loader path. The Vulkan DLL staging script was also
made compatible with both Git Bash `/c/...` and WSL-style `/mnt/c/...`
Strawberry paths.

Lane:

`ctx=8192`, `b512/ub128`, `q4_0/q4_0`, FlashAttention on,
`max_tokens=256`, `temperature=0.0`, `quick:triage_diff`,
`real-context-mode off`, no reuse/no prime, thinking enabled,
`--spec-type none`.

## Results

| Variant | Label | Status | Aggregate TPS | Prompt tok/s | Decode tok/s | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Vulkan0 single | `vulkan0-single-short-mt256-none-r1` | completed | `8.2558` | `53.20` | `9.15` | graph splits `2`, sched copies `1` |
| Vulkan1 single | `vulkan1-single-short-mt256-none-r1` | crashed before ready | - | - | - | process exit `0xC0000005` after tensor load, during context initialization |
| Vulkan0,Vulkan1 dual layer | `vulkan-dual-layer-short-mt256-none-r1` | invalid after driver reset | - | - | - | backend reported no usable GPU / invalid `Vulkan0` |

## Driver Event

After the Vulkan1/dual attempts, Windows recorded fresh `LiveKernelEvent`
watchdog entries around `20:59-21:00`, including `P1: 141` and
`WATCHDOG` / `AMD_WATCHDOG` dump references. Treat this as a real GPU driver
reset/hang, not just a server process crash.

Post-reboot minimal retest confirmed the same failure mode:

1. `llama-server --list-devices` initially showed both `Vulkan0` and `Vulkan1`.
2. `GGML_VK_VISIBLE_DEVICES=0` isolated list succeeded.
3. `GGML_VK_VISIBLE_DEVICES=1` isolated list succeeded.
4. Physical Vulkan0 model smoke with `llama-bench -p 16 -n 1 -sm none -dev Vulkan0`
   completed.
5. Physical Vulkan1 model smoke was run in conservative safe mode with
   coopmat, coopmat2, bf16, integer-dot, and async disabled. It reached Vulkan
   device discovery but failed context creation.
6. Immediately after that failed context creation, Windows reported the second
   RX 9070 XT as `ConfigManagerErrorCode=31`, while Vulkan and ROCm both saw
   only one remaining usable GPU. WER recorded fresh `LiveKernelEvent P1=141`
   watchdog reports, including `WATCHDOG-20260709-2120.dmp` and
   `WATCHDOG-20260709-2121.dmp`.

This makes physical Vulkan1 model/context initialization unsafe on the current
Windows AMD driver stack, even with the conservative Vulkan feature mask. ROCm
does not show this failure mode on the current branch unless Vulkan has already
put the second GPU into the driver-error state.

## Driver Rollback Retest

After rolling the AMD display driver back to `32.0.23033.1002`
(`2026-03-09`), the same Vulkan path no longer reproduced the driver drop:

| Probe | Command shape | Status | Notes |
| --- | --- | --- | --- |
| Device list | `llama-server --list-devices` | passed | both `Vulkan0` and `Vulkan1` visible |
| Isolated Vulkan0 list | `GGML_VK_VISIBLE_DEVICES=0` | passed | one visible device |
| Isolated Vulkan1 list | `GGML_VK_VISIBLE_DEVICES=1` | passed | one visible device |
| Physical Vulkan1 9B smoke | `Qwen3.5-9B-Q6_K`, `-p 16 -n 1`, `-sm none` | passed | normal coopmat/bf16/int-dot path |
| Physical Vulkan1 27B smoke | `Qwen3.6-27B-Q3_K_S`, `-p 16 -n 1`, `-sm none` | passed | normal coopmat/bf16/int-dot path |
| Dual Vulkan 27B smoke | `-dev Vulkan0/Vulkan1 -sm layer -ts 1/1`, `-p 16 -n 1` | passed | true dual-device JSON field: `devices=Vulkan0/Vulkan1` |

Final post-test checks showed both RX 9070 XT devices still in Windows
`Status=OK`, `ConfigManagerErrorCode=0`, and Vulkan still listing both devices.
No new `WATCHDOG` / `AMD_WATCHDOG` dump appeared for the rollback retest.

Decision update: the Vulkan1 drop is strongly driver-version dependent. The
newer driver stack that produced `LiveKernelEvent P1=141` should be considered
unsafe for dual Vulkan on this machine; the rollback driver passes minimal
single and dual Vulkan context/compute smoke.

## Decision

Stop Vulkan1 and Vulkan dual-GPU testing in this session. `Vulkan0` single is
valid but much slower than ROCm on this short Q3 decode lane. Vulkan dual has no
valid performance result yet and must be re-tested only after reboot with a much
smaller smoke sequence:

1. `llama-server --list-devices`
2. `Vulkan0` single `max_tokens=1`
3. `Vulkan1` single `max_tokens=1` only if the device list is sane
4. dual only after both single-device context initializations are stable

Follow-up code added a default-off `GGML_VK_INIT_TRACE=1` init tracer around
logical device creation, queue setup, shader loading, fences, semaphores, and
command pools. Repeated cached device lookups are intentionally hidden unless
`GGML_VK_INIT_TRACE_VERBOSE=1` is also set. Use the tracer only for minimal
smoke after reboot. If the trace still reaches the dangerous second-device path,
start with a conservative Vulkan safe-mode env set before any dual test:

`GGML_VK_VISIBLE_DEVICES=0` for normal Vulkan use, or for Vulkan1 diagnosis only:
`GGML_VK_INIT_TRACE=1`, `GGML_VK_DISABLE_COOPMAT=1`,
`GGML_VK_DISABLE_COOPMAT2=1`, `GGML_VK_DISABLE_BFLOAT16=1`,
`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`, `GGML_VK_DISABLE_ASYNC=1`.

Current operational decision: with driver `32.0.23033.1002`, Vulkan1 and dual
Vulkan are allowed only through short smoke-first validation. Do not upgrade
back to the driver that produced `LiveKernelEvent P1=141` unless the goal is a
controlled driver regression test.
