package dev.supirvast.vastir.core;

/**
 * A combined 2D image + sampler resource (a GLSL {@code sampler2D}) bound at a descriptor {@code set}/
 * {@code binding}. Sampled in a fragment shader with {@link Expr.SampleTexture}, yielding an RGBA {@code vec4}.
 * Identity is by value (set + binding), so the same physical texture can be referenced from several places.
 */
public record Texture(String name, int set, int binding) {

    /** A texture in descriptor set 0 at the given binding (the common case). */
    public Texture(String name, int binding) {
        this(name, 0, binding);
    }
}
