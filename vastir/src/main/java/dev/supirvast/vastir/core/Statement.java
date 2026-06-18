package dev.supirvast.vastir.core;

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

    /** Writes {@code value} to {@code buffer[index]} (an i32 element of a storage buffer). */
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
}
