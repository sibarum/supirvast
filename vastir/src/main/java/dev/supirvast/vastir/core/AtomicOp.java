package dev.supirvast.vastir.core;

import dev.supirvast.vastir.type.Type;

/**
 * The read-modify-write an {@link Statement.AtomicUpdate} performs on one storage-buffer element, indivisibly
 * with respect to every other invocation touching that element.
 *
 * <p>Which operations exist depends on the element type, and the table is SPIR-V's rather than an aesthetic
 * one: a 32-bit integer supports all of them, with {@code MIN}/{@code MAX} choosing the signed or unsigned
 * instruction from the element's signedness; an {@code f32} supports {@code ADD}, {@code MIN}, {@code MAX} and
 * {@code EXCHANGE}, where the first three need a device extension and so a capability the target may refuse.
 * There is no float {@code SUB} because SPIR-V has none — add the negation.
 */
public enum AtomicOp {
    ADD, SUB, MIN, MAX, AND, OR, XOR, EXCHANGE;

    /** Whether this operation is defined on buffer elements of type {@code element}. */
    public boolean definedOn(Type element) {
        if (element instanceof Type.Int i && i.width() == 32) {
            return true;
        }
        if (element instanceof Type.Float f && f.width() == 32) {
            return this == ADD || this == MIN || this == MAX || this == EXCHANGE;
        }
        return false;
    }
}
