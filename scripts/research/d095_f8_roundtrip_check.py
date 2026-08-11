#!/usr/bin/env python3
"""D095-D2: mirror of the GLSL f32<->f8_e4m3 converters to find roundtrip breakage.

Mirrors ggml/src/ggml-vulkan/vulkan-shaders/types.glsl EXACTLY, including
the uint wrap of negative exponent fields and the round() semantics.
"""
import math

def glsl_round(x: float) -> float:
    # GLSL round(): half away from zero
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)

def fp8_e4m3_to_f32(v: int) -> float:
    sign = (v >> 7) & 0x1
    exp  = (v >> 3) & 0xF
    man  = v & 0x7
    if exp == 0:
        f = man / 512.0
    elif exp == 15:
        f = 0.0
    else:
        f = math.ldexp(1.0 + man / 8.0, exp - 7)
    return -f if sign else f

def f32_to_fp8_e4m3_fixed(f: float) -> int:
    """Mirror of the FIXED GLSL encoder (subnormal zone covers [0, 2^-6))."""
    sign = 0x80 if f < 0.0 else 0
    f = abs(f)
    if f > 448.0:
        f = 448.0
    if f < 1.0 / 64.0:
        man = int(glsl_round(f * 512.0))
        man = min(man, 7)
        return sign | man
    m, e = math.frexp(f)
    exp_field = e - 1 + 7
    exp_field = min(exp_field, 14)
    scaled = f / math.ldexp(1.0, exp_field - 7)
    man = int(glsl_round((scaled - 1.0) * 8.0))
    man = min(man, 7)
    return sign | (exp_field << 3) | man

def main():
    print(f"{'value':>12} {'enc':>6} {'dec':>12} {'err%':>9} {'abs_err':>10}")
    # grid: powers of 2 crossing all exponent boundaries + typical KV values
    vals = []
    for e in range(-20, 9):
        for m in (1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 9.0):
            vals.append(m * 2.0 ** e)
    vals += [0.001554, -0.011459, -0.059570, 0.530273, 0.01, 0.003, 0.005, 0.008,
             0.001, 0.002, 0.004, 1e-6, 3e-4, 448.0, -448.0, 240.0, 1.0, 0.5, 0.25]
    vals = sorted(set(vals))
    worst = []
    for v in vals:
        enc = f32_to_fp8_e4m3_fixed(v)
        dec = fp8_e4m3_to_f32(enc)
        rel = abs(dec - v) / max(abs(v), 1e-12) * 100
        worst.append((rel, v, enc, dec))
        flag = " <<<" if rel > 50 else ""
        print(f"{v:>12.6g} {enc:>6} {dec:>12.6g} {rel:>8.1f}% {abs(dec-v):>10.3g}{flag}")
    print("\n--- worst 15 ---")
    for rel, v, enc, dec in sorted(worst, reverse=True)[:15]:
        print(f"in={v:>12.6g} enc=0x{enc:02x} dec={dec:>12.6g} rel_err={rel:.1f}%")
    # how many of a uniform sample in [-2,2] hit the bad zone?
    import random
    random.seed(42)
    bad = 0
    huge = 0
    n = 100000
    for _ in range(n):
        v = random.uniform(-2.0, 2.0)
        dec = fp8_e4m3_to_f32(f32_to_fp8_e4m3_fixed(v))
        rel = abs(dec - v) / max(abs(v), 1e-12)
        if rel > 0.5:
            bad += 1
        if abs(dec) > 10:
            huge += 1
    print(f"\nuniform [-2,2] sample n={n}: rel_err>50%: {bad} ({bad/n*100:.2f}%), |dec|>10 (catastrophic): {huge} ({huge/n*100:.2f}%)")

if __name__ == "__main__":
    main()
