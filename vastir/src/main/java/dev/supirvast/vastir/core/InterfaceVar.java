package dev.supirvast.vastir.core;

import dev.supirvast.vastir.type.Type;

/**
 * A user-defined stage interface variable bound to a {@code location} — a vertex output / fragment input
 * "varying", or a vertex attribute input / fragment color output. The {@code direction} selects the SPIR-V
 * storage class (Input vs Output); matching producer/consumer stages use the same {@code location} and
 * {@code type}. Read with {@link Expr.InterfaceRead} (inputs) and written with {@link Statement.InterfaceWrite}
 * (outputs).
 *
 * <p>A {@code flat} variable is not interpolated: the fragment stage reads the provoking vertex's value exactly.
 * An integer varying must be flat (Vulkan rejects an interpolated one), and a value that names something, such as
 * an index into a buffer, wants to be, since an interpolated float is not promised to arrive whole. Both stages
 * declare it flat, as they declare the same location and type.
 */
public record InterfaceVar(String name, int location, Type type, Direction direction, boolean flat) {

    public enum Direction { INPUT, OUTPUT }

    /** Interpolated, as a varying is unless said otherwise. */
    public InterfaceVar(String name, int location, Type type, Direction direction) {
        this(name, location, type, direction, false);
    }

    public static InterfaceVar input(String name, int location, Type type) {
        return new InterfaceVar(name, location, type, Direction.INPUT);
    }

    public static InterfaceVar output(String name, int location, Type type) {
        return new InterfaceVar(name, location, type, Direction.OUTPUT);
    }

    /** This variable, not interpolated. */
    public InterfaceVar asFlat() {
        return new InterfaceVar(name, location, type, direction, true);
    }
}
