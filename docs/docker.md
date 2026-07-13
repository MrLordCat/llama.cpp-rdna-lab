# Docker

The repository keeps three Dockerfiles:

- `.devops/cpu.Dockerfile`;
- `.devops/vulkan.Dockerfile`;
- `.devops/rocm.Dockerfile`.

Examples:

```bash
docker build -t llama-local:cpu -f .devops/cpu.Dockerfile .
docker build -t llama-local:vulkan -f .devops/vulkan.Dockerfile .
docker build -t llama-local:rocm -f .devops/rocm.Dockerfile .
```

The primary supported environment remains native Windows. Container builds are
secondary packaging paths and must not reintroduce removed backends.
