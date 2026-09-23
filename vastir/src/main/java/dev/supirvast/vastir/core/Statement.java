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
     * elements. Ordering within a workgroup is what a {@link Barrier} provides; there is none across them.
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

    /** Writes {@code value} to {@code array[index]} (an element of workgroup memory). */
    record SharedStore(SharedArray array, Expr index, Expr value) implements Statement {}

    /**
     * Waits until every invocation in the workgroup has arrived here, and makes each one's earlier writes —
     * to workgroup memory and to storage buffers — visible to the others after it. The execution barrier
     * {@code OpControlBarrier} with workgroup scope and acquire-release semantics over both kinds of memory.
     *
     * <p>Buffer memory is included deliberately. A barrier that ordered only workgroup memory would let an
     * invocation read a buffer element another in its group wrote before the barrier and see the old value,
     * where the CPU backend — which runs the invocations of a workgroup phase by phase — always sees the new
     * one; the two backends would disagree on a kernel neither rejects.
     *
     * <p>Every invocation of the workgroup must reach the same barriers the same number of times, so a barrier
     * may only sit in <em>uniform</em> control flow: under conditions every invocation in the group evaluates
     * alike, and after no return some of them took. {@link Barriers#check} enforces that from the IR's structure
     * and both lowerings call it. A kernel with a barrier runs whole workgroups; see {@link
     * Expr.InvocationCount}.
     */
    record Barrier() implements Statement {}

    /**
     * {@link AtomicUpdate} on an element of workgroup memory: atomic with respect to the other invocations of
     * the workgroup, which are the only ones that can see it. Workgroup scope, relaxed ordering.
     *
     * <p>The same table of operations as a buffer's, {@link AtomicOp#definedOn}. The float ones each need the
     * capability a buffer's would and a workgroup-memory device feature besides (float exchange included,
     * which on workgroup memory is licensed by {@code shaderSharedFloat32Atomics}); a device may license
     * float atomics on one kind of memory and not the other, so the two are budgeted apart.
     */
    record SharedAtomicUpdate(LocalVar previous, AtomicOp op, SharedArray array, Expr index, Expr value)
            implements Statement {
        public SharedAtomicUpdate {
            if (!op.definedOn(array.element())) {
                throw new IllegalArgumentException("atomic " + op + " is not defined on shared array '"
                        + array.name() + "' of " + array.element());
            }
            requireType(value.type(), array, "value");
            if (previous != null) {
                requireType(previous.type(), array, "previous");
            }
        }

        /** An update whose old value is not wanted. */
        public SharedAtomicUpdate(AtomicOp op, SharedArray array, Expr index, Expr value) {
            this(null, op, array, index, value);
        }
    }

    /**
     * {@link AtomicCompareExchange} on an element of workgroup memory: 32-bit integers only, as in SPIR-V.
     * Workgroup scope, relaxed ordering.
     */
    record SharedAtomicCompareExchange(LocalVar previous, SharedArray array, Expr index, Expr expected,
            Expr desired) implements Statement {
        public SharedAtomicCompareExchange {
            requireInt32(array);
            if (previous == null) {
                throw new IllegalArgumentException("atomic compare-exchange needs a variable for its outcome");
            }
            requireType(previous.type(), array, "previous");
            requireType(expected.type(), array, "expected");
            requireType(desired.type(), array, "desired");
        }
    }

    private static void requireInt32(SharedArray array) {
        if (!(array.element() instanceof Type.Int i && i.width() == 32)) {
            throw new IllegalArgumentException("atomic compare-exchange needs a 32-bit integer array; '"
                    + array.name() + "' holds " + array.element());
        }
    }

    private static void requireType(Type type, SharedArray array, String role) {
        if (!type.equals(array.element())) {
            throw new IllegalArgumentException("atomic " + role + " is " + type + ", but shared array '"
                    + array.name() + "' holds " + array.element());
        }
    }

    private static void requireType(Type type, Buffer buffer, String role) {
        if (!type.equals(buffer.element())) {
            throw new IllegalArgumentException("atomic " + role + " is " + type + ", but buffer '" + buffer.name()
                    + "' holds " + buffer.element());
        }
    }
}
