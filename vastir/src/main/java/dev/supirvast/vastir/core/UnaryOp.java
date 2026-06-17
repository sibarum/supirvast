package dev.supirvast.vastir.core;

/** Unary operators. {@code NEGATE}/{@code NOT} return the operand type; {@code LOGICAL_NOT} returns bool. */
public enum UnaryOp {
    /** Arithmetic negation (int/float). */
    NEGATE,
    /** Bitwise complement (int). */
    NOT,
    /** Boolean negation. */
    LOGICAL_NOT
}
