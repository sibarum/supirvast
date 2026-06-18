package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.type.Type;

/**
 * One column of a data-parallel kernel's interface — a storage buffer bound at {@code binding}, carrying
 * either input or output data laid out struct-of-arrays (one element per invocation). The {@code type} is the
 * element type (e.g. {@code i32} or {@code f32}); together with the name, binding, and direction it forms the
 * language-neutral ABI a front end marshals against. The {@code binding} must equal the column's position in
 * the kernel's column list (binding {@code i} ⇔ slot {@code i}).
 */
public record KernelColumn(String name, int binding, Type type, Direction direction) {

    /** Whether a column is read by the kernel (input) or written by it (output). */
    public enum Direction { INPUT, OUTPUT }

    public KernelColumn {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("column name must be non-blank");
        }
        if (binding < 0) {
            throw new IllegalArgumentException("column binding must be >= 0, got " + binding);
        }
        if (type == null) {
            throw new IllegalArgumentException("column element type must be set");
        }
        if (direction == null) {
            throw new IllegalArgumentException("column direction must be set");
        }
    }

    public static KernelColumn input(String name, int binding) {
        return input(name, binding, Type.int32());
    }

    public static KernelColumn output(String name, int binding) {
        return output(name, binding, Type.int32());
    }

    public static KernelColumn input(String name, int binding, Type type) {
        return new KernelColumn(name, binding, type, Direction.INPUT);
    }

    public static KernelColumn output(String name, int binding, Type type) {
        return new KernelColumn(name, binding, type, Direction.OUTPUT);
    }

    public boolean isOutput() {
        return direction == Direction.OUTPUT;
    }
}
