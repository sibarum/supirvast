package dev.supirvast.vastir.core;

import dev.supirvast.vastir.type.Type;

/**
 * A graphics-pipeline built-in variable, named at the core level without SPIR-V vocabulary (the lowering maps
 * each to a SPIR-V {@code BuiltIn} decoration). Each carries a fixed type and direction:
 * {@link #POSITION} is the vertex stage's output clip-space position, {@link #VERTEX_INDEX} the vertex stage's
 * input index of the vertex being processed, {@link #FRAG_DEPTH} the fragment stage's replacement for the
 * interpolated depth.
 *
 * @see Expr.BuiltinRead
 * @see Statement.BuiltinWrite
 */
public enum Builtin {

    /** {@code gl_Position} — a vertex shader output, {@code vec4}. */
    POSITION,

    /** {@code gl_VertexIndex} — a vertex shader input, {@code int}. */
    VERTEX_INDEX,

    /**
     * {@code gl_FragDepth} — a fragment shader output, {@code float}: the depth this fragment writes in place
     * of the one interpolated from its primitive's vertices.
     *
     * <p>The built-in a ray-marched or ray-traced fragment cannot do without. Such a shader draws over a
     * fullscreen triangle whose vertices carry one fixed depth, so without this the only depth it can
     * contribute is that constant — which is not the distance to whatever it actually hit, and makes it
     * occlude a rasterised mesh at a flat plane rather than at its own geometry.
     *
     * <p>Two consequences a caller should know, neither of which this enum can enforce:
     *
     * <ul>
     *   <li><b>It costs early-z</b> for the pipeline that writes it. Writing this makes the lowering declare
     *       the {@code DepthReplacing} execution mode, which is how an implementation is told the depth test
     *       cannot be settled before the fragment shader has run.</li>
     *   <li><b>Every path must write it.</b> A fragment shader that writes it on one branch and not another
     *       leaves the value undefined on the branch that did not, and the symptom is geometry that occludes
     *       intermittently rather than anything a validator reports. The miss path of a ray-march writes the
     *       far value; it does not skip the write.</li>
     * </ul>
     *
     * <p>The value is a clip depth in the range the viewport declares ({@code minDepth}..{@code maxDepth},
     * conventionally 0 near and 1 far), not a world distance. Turning a distance into one is the caller's
     * projection convention, and is deliberately not modelled here: {@code core} would have to acquire a
     * notion of camera to do it, and two stages that disagree about that convention is a problem no amount
     * of typing in this enum could catch.
     */
    FRAG_DEPTH;

    /** The fixed value type of this built-in. */
    public Type type() {
        return switch (this) {
            case POSITION -> new Type.Vector(Type.float32(), 4);
            case VERTEX_INDEX -> Type.int32();
            case FRAG_DEPTH -> Type.float32();
        };
    }

    /** Whether this built-in is an input (read by the shader) rather than an output (written by it). */
    public boolean isInput() {
        return this == VERTEX_INDEX;
    }
}
