package dev.supirvast.vastir.core;

import dev.supirvast.vastir.type.Type;

/**
 * A core-level function definition: a name, a signature, and a structured body.
 *
 * <p>Distinct from {@link Type.FunctionType}, which is the function's <em>signature type</em>; this is the
 * function <em>definition</em>. Per the naming convention's collision rule, the type-level concept keeps the
 * {@code Type} qualifier while this stays the unprefixed {@code Function}.
 */
public record Function(String name, Type.FunctionType signature, Region body) {
}
