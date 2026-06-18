package dev.supirvast.vastir.core;

/**
 * A built-in floating-point math intrinsic, named semantically so {@code core} stays free of SPIR-V opcode
 * vocabulary (the lowering maps each to {@code OpDot} or a {@code GLSL.std.450} extended instruction). These
 * are the operations a shading model needs beyond the elementwise {@link BinaryOp}s — dot products, vector
 * normalization, and the transcendental/utility functions of a BRDF.
 *
 * <p>{@link #DOT} and {@link #LENGTH} reduce a vector to its scalar component type; the rest are elementwise
 * and share their (first) argument's type.
 */
public enum MathFn {
    DOT, LENGTH, NORMALIZE, CROSS, REFLECT,
    POW, SQRT, INVERSE_SQRT, ABS,
    MIN, MAX, CLAMP, MIX
}
