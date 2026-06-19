package dev.supirvast.vastir.core;

/**
 * A sampled image resource bound at a descriptor {@code set}/{@code binding} — either a 2D texture (GLSL
 * {@code sampler2D}, sampled with a {@code vec2}) or a cubemap (GLSL {@code samplerCube}, sampled with a
 * {@code vec3} direction), per {@link #kind()}. Sampled in a fragment shader with {@link Expr.SampleTexture},
 * yielding an RGBA {@code vec4}.
 */
public record Texture(String name, int set, int binding, Kind kind) {

    /** Whether the texture is a flat 2D image or a cubemap. */
    public enum Kind { TEXTURE_2D, CUBE }

    /** A 2D texture in descriptor set 0 at the given binding (the common case). */
    public Texture(String name, int binding) {
        this(name, 0, binding, Kind.TEXTURE_2D);
    }

    /** A 2D texture at the given set/binding. */
    public Texture(String name, int set, int binding) {
        this(name, set, binding, Kind.TEXTURE_2D);
    }

    /** A cubemap in descriptor set 0 at the given binding. */
    public static Texture cube(String name, int binding) {
        return new Texture(name, 0, binding, Kind.CUBE);
    }
}
