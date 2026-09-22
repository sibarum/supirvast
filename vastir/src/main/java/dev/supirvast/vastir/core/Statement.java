package dev.supirvast.vastir.core;

import dev.supirvast.vastir.type.Type;

/**
 * A core-level statement. Control flow is structured: {@link If} and {@link While} carry nested {@link Region}s
 * rather than branching to labels, which is exactly what lets the lowering produce SPIR-V's structured
 * {@code OpSelectionMerge}/{@code OpLoopMerge} (and a future Truffle tree) without reconstructing a CFG.
 */
public sealed interface Statement {

    /** Returns from a {@code void} function. */
    record ReturnVoid() implements Statement {}

    /** Returns a value from a non-{@code void} function. */
    record Return(Expr value) implements Statement {}

    /**
     * Writes an int value to element 0 of the shader's output storage buffer (descriptor set 0, binding 0).
     * This is how a compute shader produces an observable result on the GPU; the CPU backend does not model
     * it (the differential harness reads CPU results via {@link Return} instead).
     */
    record StoreResult(Expr value) implements Statement {}

    /** Writes {@code value} to {@code buffer[index]} (an element of a storage buffer). */
    record BufferStore(Buffer buffer, Expr index, Expr value) implements Statement {}

    /** Writes a graphics-pipeline built-in output (e.g. {@code gl_Position}). */
    record BuiltinWrite(Builtin builtin, Expr value) implements Statement {}

    /** Writes {@code value} to a stage interface output variable at its declared {@code location}. */
    record InterfaceWrite(InterfaceVar variable, Expr value) implements Statement {}

    /** Declares a local variable and initializes it. */
    record DeclareVar(LocalVar variable, Expr initializer) implements Statement {}

    /** Stores a value into an existing local variable. */
    record Assign(LocalVar variable, Expr value) implements Statement {}

    /** Structured selection. {@code elseRegion} may be empty for an {@code if} with no {@code else}. */
    record If(Expr condition, Region thenRegion, Region elseRegion) implements Statement {}

    /** Structured pre-tested loop: {@code while (condition) { body }}. */
    record While(Expr condition, Region body) implements Statement {}

    /**
     * Atomically replaces {@code buffer[index]} with {@code op(buffer[index], value)}, and assigns the element's
     * value from before the update to {@code previous}.
     *
     * <p>A statement rather than an expression on purpose: a side effect inside an expression tree is one a
     * pass that folds, substitutes or drops dead subexpressions can duplicate or erase, and the passes that
     * rewrite trees above this IR have no reason to know which nodes are unsafe to touch.
     *
     * <p>{@code previous} is {@code null} when the old value is not wanted. When it is given, this statement
     * assigns it, and declares it if nothing else does — so a loop can reuse one variable across iterations.
     * Memory scope is the device and ordering is relaxed: atomicity per element, with no ordering between
     * elements. Anything that needs the latter needs a barrier, which this IR does not have.
     */
    record AtomicUpdate(LocalVar previous, AtomicOp op, Buffer buffer, Expr index, Expr value)
            implements Statement {
        public AtomicUpdate {
            if (!op.definedOn(buffer.element())) {
                throw new IllegalArgumentException(
                        "atomic " + op + " is not defined on buffer '" + buffer.name() + "' of " + buffer.element());
            }
            requireType(value.type(), buffer, "value");
            if (previous != null) {
                requireType(previous.type(), buffer, "previous");
            }
        }

        /** An update whose old value is not wanted. */
        public AtomicUpdate(AtomicOp op, Buffer buffer, Expr index, Expr value) {
            this(null, op, buffer, index, value);
        }
    }

    /**
     * Atomically: if {@code buffer[index] == expected}, replace it with {@code desired}. Either way, assign the
     * value found there to {@code previous} — the exchange happened exactly when {@code previous == expected}.
     *
     * <p>Integer elements only, as in SPIR-V. {@code previous} is required: a compare-exchange whose outcome is
     * not read cannot be told apart from one that failed. Declared by this statement if nothing else declares
     * it, which is what lets a retry loop reuse it. Device scope, relaxed ordering on both outcomes.
     */
    record AtomicCompareExchange(LocalVar previous, Buffer buffer, Expr index, Expr expected, Expr desired)
            implements Statement {
        public AtomicCompareExchange {
            if (!(buffer.element() instanceof Type.Int i && i.width() == 32)) {
                throw new IllegalArgumentException("atomic compare-exchange needs a 32-bit integer buffer; '"
                        + buffer.name() + "' holds " + buffer.element());
            }
            if (previous == null) {
                throw new IllegalArgumentException("atomic compare-exchange needs a variable for its outcome");
            }
            requireType(previous.type(), buffer, "previous");
            requireType(expected.type(), buffer, "expected");
            requireType(desired.type(), buffer, "desired");
        }
    }

    private static void requireType(Type type, Buffer buffer, String role) {
        if (!type.equals(buffer.element())) {
            throw new IllegalArgumentException("atomic " + role + " is " + type + ", but buffer '" + buffer.name()
                    + "' holds " + buffer.element());
        }
    }
}
