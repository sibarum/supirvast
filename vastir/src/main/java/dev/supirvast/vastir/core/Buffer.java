package dev.supirvast.vastir.core;

import dev.supirvast.vastir.type.Type;

/**
 * A storage buffer bound at a descriptor {@code set} and {@code binding} — a runtime-sized array of {@code element}
 * on the GPU. Read with {@link Expr.BufferLoad} and written with {@link Statement.BufferStore}, indexed per
 * invocation; this is how a data-parallel kernel consumes and produces arrays. The element type selects the
 * SPIR-V array element and stride (e.g. {@code i32} or {@code f32}).
 *
 * <p>A kernel's buffers are all in set 0. A graphics stage that has set 0 already, for its textures, puts its buffers
 * in a set of their own. A module's buffers are told apart by binding alone: two buffers in one module never share a
 * binding number, even in different sets.
 */
public record Buffer(String name, int set, int binding, Type element) {

    /** A buffer in set 0, as every kernel's are. */
    public Buffer(String name, int binding, Type element) {
        this(name, 0, binding, element);
    }

    /** A buffer of {@code i32} elements in set 0 — the common case and backward-compatible default. */
    public Buffer(String name, int binding) {
        this(name, 0, binding, Type.int32());
    }
}
