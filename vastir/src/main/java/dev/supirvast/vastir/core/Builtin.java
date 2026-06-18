package dev.supirvast.vastir.core;

import dev.supirvast.vastir.type.Type;

/**
 * A graphics-pipeline built-in variable, named at the core level without SPIR-V vocabulary (the lowering maps
 * each to a SPIR-V {@code BuiltIn} decoration). Each carries a fixed type and direction:
 * {@link #POSITION} is the vertex stage's output clip-space position, {@link #VERTEX_INDEX} the vertex stage's
 * input index of the vertex being processed.
 *
 * @see Expr.BuiltinRead
 * @see Statement.BuiltinWrite
 */
public enum Builtin {

    /** {@code gl_Position} — a vertex shader output, {@code vec4}. */
    POSITION,

    /** {@code gl_VertexIndex} — a vertex shader input, {@code int}. */
    VERTEX_INDEX;

    /** The fixed value type of this built-in. */
    public Type type() {
        return switch (this) {
            case POSITION -> new Type.Vector(Type.float32(), 4);
            case VERTEX_INDEX -> Type.int32();
        };
    }

    /** Whether this built-in is an input (read by the shader) rather than an output (written by it). */
    public boolean isInput() {
        return this == VERTEX_INDEX;
    }
}
