package dev.supirvast.vastir.core;

/**
 * Target-neutral binary operators. The lowering picks the concrete SPIR-V opcode from the operand type
 * (e.g. {@code ADD} → {@code OpIAdd} for ints, {@code OpFAdd} for floats), so core stays type-driven.
 *
 * <p>Three kinds: arithmetic/bitwise (result = operand type), comparison (result = bool), and logical
 * (bool operands → bool). Logical ops are <em>not</em> short-circuit — both operands are always evaluated,
 * matching {@code OpLogicalAnd}/{@code OpLogicalOr}.
 */
public enum BinaryOp {
    ADD(Kind.ARITHMETIC),
    SUB(Kind.ARITHMETIC),
    MUL(Kind.ARITHMETIC),
    DIV(Kind.ARITHMETIC),
    MOD(Kind.ARITHMETIC),
    BIT_AND(Kind.ARITHMETIC),
    BIT_OR(Kind.ARITHMETIC),
    BIT_XOR(Kind.ARITHMETIC),
    SHIFT_LEFT(Kind.ARITHMETIC),
    SHIFT_RIGHT(Kind.ARITHMETIC),
    LESS_THAN(Kind.COMPARISON),
    GREATER_THAN(Kind.COMPARISON),
    EQUAL(Kind.COMPARISON),
    LOGICAL_AND(Kind.LOGICAL),
    LOGICAL_OR(Kind.LOGICAL);

    private enum Kind { ARITHMETIC, COMPARISON, LOGICAL }

    private final Kind kind;

    BinaryOp(Kind kind) {
        this.kind = kind;
    }

    /** Comparisons and logical ops yield a boolean; arithmetic/bitwise yields the operand type. */
    public boolean producesBool() {
        return kind != Kind.ARITHMETIC;
    }
}
