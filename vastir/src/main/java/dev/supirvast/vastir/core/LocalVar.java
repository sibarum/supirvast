package dev.supirvast.vastir.core;

import dev.supirvast.vastir.type.Type;

/**
 * A function-local variable. Identity is by reference (not value), since two distinct variables may share a
 * name and type; lowering keys its pointer map on the instance.
 */
public final class LocalVar {

    private final String name;
    private final Type type;

    public LocalVar(String name, Type type) {
        this.name = name;
        this.type = type;
    }

    public String name() {
        return name;
    }

    public Type type() {
        return type;
    }

    /** The name, so a diagnostic that prints an expression reads as the kernel was written. */
    @Override
    public String toString() {
        return name;
    }
}
