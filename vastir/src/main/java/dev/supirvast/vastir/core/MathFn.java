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
    // Geometric (DOT/LENGTH/DISTANCE reduce a vector to its scalar component type).
    DOT, LENGTH, DISTANCE, NORMALIZE, CROSS, REFLECT, REFRACT, FACE_FORWARD,
    // Common elementwise.
    POW, SQRT, INVERSE_SQRT, ABS, SIGN, MIN, MAX, CLAMP, MIX, STEP, SMOOTHSTEP, FMA,
    // Exponential / logarithmic.
    EXP, LOG, EXP2, LOG2,
    // Trigonometric.
    SIN, COS, TAN, ASIN, ACOS, ATAN, ATAN2, RADIANS, DEGREES,
    // Hyperbolic.
    SINH, COSH, TANH, ASINH, ACOSH, ATANH,
    // Rounding.
    ROUND, ROUND_EVEN, TRUNC, FLOOR, CEIL, FRACT
}
