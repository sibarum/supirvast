package dev.supirvast.vastir.core;

import dev.supirvast.vastir.type.Type;

/**
 * The operation a {@link Statement.SubgroupArithmetic} combines lanes' values with.
 *
 * <p>SPIR-V's table: on a 32-bit integer every operation, with {@code MIN}/{@code MAX} choosing the signed or
 * unsigned instruction from the type's signedness; on an {@code f32}, {@code ADD}, {@code MUL}, {@code MIN}
 * and {@code MAX}. Narrower and wider types need a device feature ({@code shaderSubgroupExtendedTypes}) that
 * nothing here asks for yet, so they are refused.
 *
 * <p>A float {@code ADD} or {@code MUL} over a subgroup combines in an order the device chooses and the CPU
 * does not reproduce, so the two backends agree exactly only where the order cannot matter — small integers,
 * say. {@code MIN} and {@code MAX} are exact on both.
 */
public enum SubgroupOp {
    ADD, MUL, MIN, MAX, AND, OR, XOR;

    /** Whether this operation is defined on values of type {@code type}. */
    public boolean definedOn(Type type) {
        if (type instanceof Type.Int i && i.width() == 32) {
            return true;
        }
        if (type instanceof Type.Float f && f.width() == 32) {
            return this == ADD || this == MUL || this == MIN || this == MAX;
        }
        return false;
    }
}
