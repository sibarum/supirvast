package dev.supirvast.vastir.core;

import dev.supirvast.vastir.type.Type;

/**
 * A storage buffer bound at descriptor set 0 and the given binding — a runtime-sized array of {@code element}
 * on the GPU. Read with {@link Expr.BufferLoad} and written with {@link Statement.BufferStore}, indexed per
 * invocation; this is how a data-parallel kernel consumes and produces arrays. The element type selects the
 * SPIR-V array element and stride (e.g. {@code i32} or {@code f32}).
 */
public record Buffer(String name, int binding, Type element) {

    /** A buffer of {@code i32} elements — the common case and backward-compatible default. */
    public Buffer(String name, int binding) {
        this(name, binding, Type.int32());
    }
}
