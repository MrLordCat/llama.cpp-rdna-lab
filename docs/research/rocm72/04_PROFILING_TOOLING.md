# Profiling and analysis on native Windows

## Correct tool choice

ROCm 7.2 documentation often demonstrates `rocprofv3`, ROCProfiler SDK,
`rocminfo`, and Linux tracing. The HIP SDK for Windows component matrix points
to Radeon GPU Profiler instead.

Radeon GPU Profiler (RGP) 2.7 officially supports:

- HIP and OpenCL compute applications;
- Radeon RX 9000, 7000, 6000, and 5000 series;
- Windows 11.

That exactly covers the RX 9070 XT lane. RGP is paired with Radeon Developer
Service and Radeon Developer Panel (RDP), which launches/captures the target;
RGP opens and analyzes the resulting profile.

The current machine has both HIP SDK 7.1 and 7.2, but the searched AMD program
directories contain no RDP or RGP executable. The tool suite must be installed
separately before this workflow is available.

## What RGP can answer

For the decode investigation, the useful views are:

- per-event/kernel device timing;
- wavefront occupancy over time;
- instruction timing and ISA view;
- queue activity, barriers, and synchronization;
- the most expensive events in a non-frame compute profile.

These answer questions that the existing synchronized node profiler cannot:

1. Which MMVQ, GDN, FA, or copy events actually dominate device time?
2. Does splitting one token across two GPUs add idle queue gaps or only extra
   work in each graph half?
3. Are candidate kernels bandwidth-bound, instruction-bound, or limited by
   occupancy/register pressure?
4. Is the 20 KiB boundary event itself slow, or is its apparent duration a
   wait for the first graph half? Existing host traces predict the latter.

## Minimal L0 capture plan

1. Install the current AMD Radeon Developer Tool Suite containing RDS, RDP,
   and RGP. Keep the current Adrenalin/HIP versions recorded with the capture.
2. Start with the single-ROCm0 L0 control. Launch the server through RDP as a
   non-frame HIP compute application.
3. Warm up normally, then capture only a short deterministic decode window.
4. Repeat the same capture for dual `ROCm1,ROCm0 -sm layer -ts 1,1`.
5. Compare event/device time and idle gaps; do not compare instrumented TPS to
   normal wall-clock TPS.
6. Export/save profile artifacts under a new `build_logs/bench/` experiment
   directory with command, model, SDK, driver, and RGP versions.

RDP documentation says profiling reserves up to 75 MiB of video memory per
shader engine, and the shader engine with instruction tracing can reserve
300 MiB. This model is already close to the WDDM budget. Use a short L0
capture, watch residency, and do not treat the profiled run as a production
performance result.

## Radeon GPU Analyzer

Radeon GPU Analyzer (RGA) is an offline compiler/analysis tool. Its useful
outputs include ISA, VGPR pressure, LDS, and scratch usage. The current product
supports Windows 11, but its advertised modes and target coverage vary by API
and release. For this project:

- prefer RGP's captured HIP ISA view when it recognizes the kernel;
- use RGA only after confirming that the chosen mode accepts the dumped HIP
  code object for `gfx1201`;
- the existing compiler code-object/ISA extraction remains the fallback.

Offline resource reports identify a mechanism; only the L0 wall benchmark
establishes benefit.

## HIP debugging controls

`AMD_SERIALIZE_KERNEL=3` and `AMD_SERIALIZE_COPY=3` make asynchronous failures
easier to localize by forcing completion around submissions. They destroy
normal overlap and must never be used for a throughput claim. Likewise,
code-object dumping and verbose HIP/AMD logs are evidence runs, not baselines.

## Sources

- [RGP manual and support matrix](https://gpuopen.com/manuals/rgp_manual/rgp_manual-index/)
- [RDP manual and HIP capture support](https://gpuopen.com/manuals/rdp_manual/rdp_manual-index/)
- [RGP quick start](https://gpuopen.com/manuals/rgp_manual/quickstart/)
- [RGA product page](https://gpuopen.com/rga/)
- [HIP debugging](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/how-to/debugging.html)
- [Windows component support](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html)
