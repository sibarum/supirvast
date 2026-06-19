package dev.supirvast.vastir.pbr;

import dev.supirvast.vastir.type.Type;

/**
 * A surface channel of a metallic-roughness PBR material — the inputs the {@link PbrShader} lighting model
 * consumes. A material declares which channels it exposes and a surface function computes each; unspecified
 * channels fall back to a sensible default. Each channel has a fixed element {@link #type()} the surface
 * function's expression must match.
 */
public enum Channel {

    /** Base color (linear RGB). */
    ALBEDO(vec3()),
    /** Shading normal in the same space as the geometry; defaults to the interpolated geometric normal. */
    NORMAL(vec3()),
    /** 0 = dielectric, 1 = metal. */
    METALLIC(Type.float32()),
    /** Perceptual roughness in [0,1]. */
    ROUGHNESS(Type.float32()),
    /** Ambient occlusion multiplier in [0,1]. */
    AO(Type.float32()),
    /** Emitted radiance (linear RGB), added after lighting. */
    EMISSIVE(vec3()),
    /** Alpha in [0,1], written to the output's w. */
    OPACITY(Type.float32());

    private final Type type;

    Channel(Type type) {
        this.type = type;
    }

    public Type type() {
        return type;
    }

    private static Type.Vector vec3() {
        return new Type.Vector(Type.float32(), 3);
    }
}
