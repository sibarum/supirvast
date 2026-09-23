package dev.supirvast.vastir.core;

import dev.supirvast.vastir.type.Type;

/**
 * A fixed-length array in workgroup memory: one copy per workgroup, shared by every invocation in it and by
 * no invocation outside it. Read with {@link Expr.SharedLoad}, written with {@link Statement.SharedStore},
 * updated indivisibly with {@link Statement.SharedAtomicUpdate}; a {@link Statement.Barrier} is what makes one
 * invocation's writes visible to the others.
 *
 * <p>Its contents are undefined when a workgroup starts — the GPU gives it no initializer — so every element
 * a kernel reads must be written first, in the same workgroup. The CPU backend happens to start it at zero,
 * which is one of the values "undefined" allows and not a promise.
 *
 * <p>The length is fixed when the kernel is built, which is what lets a lowering check the total against the
 * device's workgroup-memory limit ({@code maxComputeSharedMemorySize}, 16 KB guaranteed) before anything runs.
 *
 * <p>Identity is by reference, as for {@link LocalVar}: two arrays with the same name, type and length are
 * still two arrays.
 */
public final class SharedArray {

    private final String name;
    private final Type element;
    private final int length;

    public SharedArray(String name, Type element, int length) {
        if (length < 1) {
            throw new IllegalArgumentException("shared array '" + name + "' needs at least one element, got "
                    + length);
        }
        elementBytes(element);   // rejects an element type with no place in workgroup memory
        this.name = name;
        this.element = element;
        this.length = length;
    }

    public String name() {
        return name;
    }

    public Type element() {
        return element;
    }

    public int length() {
        return length;
    }

    /**
     * The bytes this array occupies, as counted against the workgroup-memory limit. The layout of workgroup
     * memory is the implementation's, so this is the conservative std430 figure: a {@code vec3} counts as
     * four components and a {@code bool} as four bytes.
     */
    public long bytes() {
        return (long) elementBytes(element) * length;
    }

    private static int elementBytes(Type element) {
        return switch (element) {
            case Type.Int i -> i.width() / 8;
            case Type.Float f -> f.width() / 8;
            case Type.Bool ignored -> 4;
            case Type.Vector v -> (v.count() == 3 ? 4 : v.count()) * elementBytes(v.component());
            default -> throw new IllegalArgumentException("a shared array holds scalars or vectors, not "
                    + element);
        };
    }

    @Override
    public String toString() {
        return name + ": " + element + "[" + length + "]";
    }
}
