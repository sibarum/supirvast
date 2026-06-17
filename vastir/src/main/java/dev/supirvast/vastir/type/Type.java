package dev.supirvast.vastir.type;

import java.util.List;

/**
 * The shared Supir-Vast type system, used by every IR level (see the IR naming convention).
 *
 * <p>Types are value objects (records), so structural equality is free — which is exactly what SPIR-V needs,
 * since a module must declare each distinct type exactly once. A lowering pass can therefore dedupe type
 * declarations with a plain {@code Map<Type, Integer>}.
 */
public sealed interface Type {

    /** The {@code void} type, used for functions that return nothing. */
    Void VOID = new Void();

    /** The boolean type. */
    Bool BOOL = new Bool();

    record Void() implements Type {}

    record Bool() implements Type {}

    /** An integer of the given bit width and signedness. */
    record Int(int width, boolean signed) implements Type {}

    /** A floating-point number of the given bit width. */
    record Float(int width) implements Type {}

    /** A vector of {@code count} components of a scalar {@code component} type. */
    record Vector(Type component, int count) implements Type {}

    /** A function signature: a return type and ordered parameter types. */
    record FunctionType(Type returnType, List<Type> parameterTypes) implements Type {
        public FunctionType {
            parameterTypes = List.copyOf(parameterTypes);
        }
    }

    // Common scalar constructors, for readability at call sites.

    static Int int32() {
        return new Int(32, true);
    }

    static Int uint32() {
        return new Int(32, false);
    }

    static Float float32() {
        return new Float(32);
    }
}
