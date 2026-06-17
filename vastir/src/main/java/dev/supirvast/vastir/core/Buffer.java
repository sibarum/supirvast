package dev.supirvast.vastir.core;

/**
 * A storage buffer of {@code i32}, bound at descriptor set 0 and the given binding — a runtime-sized array on
 * the GPU. Read with {@link Expr.BufferLoad} and written with {@link Statement.BufferStore}, indexed per
 * invocation; this is how a data-parallel kernel consumes and produces arrays.
 */
public record Buffer(String name, int binding) {
}
