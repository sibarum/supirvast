package dev.supirvast.vastir.core;

import dev.supirvast.vastir.type.Type;

/**
 * A user-defined stage interface variable bound to a {@code location} — a vertex output / fragment input
 * "varying", or a vertex attribute input / fragment color output. The {@code direction} selects the SPIR-V
 * storage class (Input vs Output); matching producer/consumer stages use the same {@code location} and
 * {@code type}. Read with {@link Expr.InterfaceRead} (inputs) and written with {@link Statement.InterfaceWrite}
 * (outputs).
 */
public record InterfaceVar(String name, int location, Type type, Direction direction) {

    public enum Direction { INPUT, OUTPUT }

    public static InterfaceVar input(String name, int location, Type type) {
        return new InterfaceVar(name, location, type, Direction.INPUT);
    }

    public static InterfaceVar output(String name, int location, Type type) {
        return new InterfaceVar(name, location, type, Direction.OUTPUT);
    }
}
