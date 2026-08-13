#pragma once

#include "common.cuh"

#include <cstdint>

#if defined(GGML_USE_HIP)
// D098: byte-compatible finite E4M3 storage for the HIP backend. Native OCP
// E4M3 can encode exponent-15 finite values up to 448 and uses a different tie
// rule, so storage uses the established CPU/Vulkan bit contract instead.
static __device__ __forceinline__ uint8_t ggml_cuda_fp32_to_f8_e4m3(float f) {
    if (isnan(f)) {
        return 0x7F;
    }

    const uint8_t sign = f < 0.0f ? 0x80 : 0;
    f = fabsf(f);
    f = f > 240.0f ? 240.0f : f;

    const uint32_t bits = __float_as_uint(f);
    const uint32_t exp32 = (bits >> 23) & 0xFF;
    if (exp32 < 121) {
        uint32_t man = (uint32_t) llroundf(f * 512.0f);
        man = man > 7 ? 7 : man;
        return (uint8_t) (sign | man);
    }

    uint32_t exp_field = exp32 - 120;
    if (exp_field > 14) {
        return (uint8_t) (sign | (14 << 3) | 7);
    }
    uint32_t man = ((bits & 0x7FFFFF) + 0x80000) >> 20;
    exp_field += man >> 3;
    man &= 7;
    if (exp_field > 14) {
        return (uint8_t) (sign | (14 << 3) | 7);
    }
    return (uint8_t) (sign | (exp_field << 3) | man);
}

// D098 G4: the native FP8 V leg quantizes softmax P after multiplying it by
// 128. P is finite, non-negative and no larger than 128, so gfx12 OCP E4M3
// cannot enter the exponent-15 range that differs from the repository's KV
// storage contract. Convert two values with one packed hardware instruction;
// keep the portable storage converter above for all persistent cache writes.
static __device__ __forceinline__ uint16_t ggml_cuda_fp32x2_to_f8_e4m3_p(float2 f) {
#if defined(RDNA4) && defined(FP8_AVAILABLE) && HIP_VERSION >= 60500000
    return (uint16_t) __hip_cvt_float2_to_fp8x2(f, __HIP_SATFINITE, __HIP_E4M3);
#else
    return (uint16_t) ggml_cuda_fp32_to_f8_e4m3(f.x) |
           (uint16_t) ggml_cuda_fp32_to_f8_e4m3(f.y) << 8;
#endif
}

static __device__ __forceinline__ float ggml_cuda_f8_e4m3_to_fp32(uint8_t v) {
    // Exponent 15 is reserved by the repository format even though OCP E4M3
    // assigns finite values to most of this range.
    if ((v & 0x78) == 0x78) {
        return NAN;
    }

#if defined(RDNA4) && defined(FP8_AVAILABLE)
    // gfx1201 uses OCP rather than FNUZ FP8. All non-reserved bytes have the
    // same bit-level interpretation and can be decoded by the native type.
    __nv_fp8_e4m3 native;
    native.__x = v;
    return (float) native;
#else
    const uint32_t sign = (v >> 7) & 1;
    const uint32_t exp  = (v >> 3) & 0xF;
    const uint32_t man  = v & 0x7;
    const float f = exp == 0
        ? (float) man * (1.0f / 512.0f)
        : ldexpf(1.0f + (float) man / 8.0f, (int) exp - 7);
    return sign ? -f : f;
#endif
}

struct ggml_cuda_f8_e4m3 {
    uint8_t data;

    __device__ operator float() const {
        return ggml_cuda_f8_e4m3_to_fp32(data);
    }
};

static_assert(sizeof(ggml_cuda_f8_e4m3) == 1, "FP8 storage must stay byte-sized");
#endif // defined(GGML_USE_HIP)
