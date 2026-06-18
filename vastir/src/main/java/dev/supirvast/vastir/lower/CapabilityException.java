package dev.supirvast.vastir.lower;

/**
 * Thrown by {@link CoreToSpirv} when a kernel requires a SPIR-V capability the {@link SpirvTarget} does not
 * permit — the witness for "this can't be lowered within that budget." Distinct from other lowering failures
 * so an orchestrator can react specifically (refuse, or fall back to another backend).
 */
public final class CapabilityException extends RuntimeException {

    public CapabilityException(String message) {
        super(message);
    }
}
