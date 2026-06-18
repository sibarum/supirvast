package dev.supirvast.vastir.lower;

import dev.supirvast.vastir.spirv.AddressingModel;
import dev.supirvast.vastir.spirv.Capability;
import dev.supirvast.vastir.spirv.MemoryModel;

import java.util.Set;

/**
 * The constraint a {@link CoreToSpirv} lowering must fit inside — chiefly the set of SPIR-V capabilities it is
 * allowed to emit. {@code CoreToSpirv} already derives the <em>minimal required</em> capabilities from the
 * types a kernel uses; a target is the <em>budget</em> those must fit within. A required capability outside
 * the budget makes the lowering fail with a witness (the capability analog of GPU-lowerability).
 *
 * <p>Two ways a target arises: a {@code restrictedTo(...)} budget a caller specifies (to refuse generating
 * features even where a device supports them — portability/ahead-of-time targeting), and a device-derived
 * profile (what the hardware actually supports). The effective target is their intersection.
 *
 * <p>{@code Shader} is always permitted (it is mandatory for compute). The addressing/memory model are fixed
 * at {@code Logical}/{@code GLSL450} today (the only combination the emitter supports); SPIR-V version
 * selection is not yet enforced (emission targets 1.6 and relies on &ge;1.4 features).
 */
public record SpirvTarget(int addressingModel, int memoryModel, Set<Capability> allowedCapabilities) {

    public SpirvTarget {
        allowedCapabilities = allowedCapabilities == null ? null : Set.copyOf(allowedCapabilities);
    }

    /** No capability restriction — emit whatever the kernel requires (the default). */
    public static SpirvTarget unconstrained() {
        return new SpirvTarget(AddressingModel.Logical.value(), MemoryModel.GLSL450.value(), null);
    }

    /** A budget allowing only the given capabilities (beyond the always-present {@code Shader}). */
    public static SpirvTarget restrictedTo(Set<Capability> capabilities) {
        return new SpirvTarget(AddressingModel.Logical.value(), MemoryModel.GLSL450.value(), capabilities);
    }

    /** Whether {@code capability} may be emitted. {@code Shader} always may; a null set allows everything. */
    public boolean allows(Capability capability) {
        return capability == Capability.Shader || allowedCapabilities == null
                || allowedCapabilities.contains(capability);
    }
}
