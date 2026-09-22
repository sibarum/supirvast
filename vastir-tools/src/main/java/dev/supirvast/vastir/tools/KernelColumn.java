package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.type.Type;

/**
 * One column of a data-parallel kernel's interface — a storage buffer bound at {@code binding}, carrying
 * either input or output data. The {@code type} is the element type (e.g. {@code i32} or {@code f32});
 * together with the name, binding, direction and length it forms the language-neutral ABI a front end
 * marshals against. The {@code binding} must equal the column's position in the kernel's column list
 * (binding {@code i} ⇔ slot {@code i}).
 *
 * <p>By default a column is laid out struct-of-arrays, one element per invocation, and a run checks it holds
 * at least that many. A column with a fixed {@code length} is instead checked against that length, whatever
 * the invocation count — which is the shape of anything many invocations reduce into: a histogram, a counter,
 * a grid that particles scatter onto. Without it, a sixteen-bin histogram over a million invocations would
 * have to be marshalled as a million-element buffer.
 *
 * @param length the column's element count, or {@link #PER_INVOCATION} for one element per invocation
 */
public record KernelColumn(String name, int binding, Type type, Direction direction, int length) {

    /** The {@code length} of a column holding one element per invocation. */
    public static final int PER_INVOCATION = 0;

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
        if (length < 0) {
            throw new IllegalArgumentException("column length must be >= 0, got " + length);
        }
    }

    /** A column with one element per invocation. */
    public KernelColumn(String name, int binding, Type type, Direction direction) {
        this(name, binding, type, direction, PER_INVOCATION);
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

    /** This column with a fixed element count, independent of how many invocations run. */
    public KernelColumn withLength(int elements) {
        if (elements <= 0) {
            throw new IllegalArgumentException("a fixed column length must be positive, got " + elements);
        }
        return new KernelColumn(name, binding, type, direction, elements);
    }

    /** The element count a run over {@code invocations} needs this column to hold. */
    public int elementsFor(int invocations) {
        return length == PER_INVOCATION ? invocations : length;
    }

    public boolean isOutput() {
        return direction == Direction.OUTPUT;
    }
}
