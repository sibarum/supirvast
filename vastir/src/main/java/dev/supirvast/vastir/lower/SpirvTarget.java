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
 * <p>Workgroup memory is budgeted the same way. A kernel's {@link dev.supirvast.vastir.core.SharedArray
 * shared arrays} have sizes fixed when it is built, so their total is known before anything runs and is
 * checked against {@code maxWorkgroupMemoryBytes} exactly as a capability is checked against the allowed set
 * — the device's {@code maxComputeSharedMemorySize}, intersected with whatever limit the caller imposes.
 *
 * <p>{@code Shader} is always permitted (it is mandatory for compute). The addressing/memory model are fixed
 * at {@code Logical}/{@code GLSL450} today (the only combination the emitter supports); SPIR-V version
 * selection is not yet enforced (emission targets 1.6 and relies on &ge;1.4 features).
 *
 * <p>So are the {@link DeviceFeature}s no capability distinguishes. A kernel needing one needs its capability
 * too, so a capability budget already refuses it; the feature set exists for a device-derived target, where
 * the capability can be present and the feature absent.
 *
 * @param maxWorkgroupMemoryBytes the most workgroup memory a module may declare, or {@link #UNLIMITED}
 * @param allowedFeatures         the device features a module may need; null allows every one
 */
public record SpirvTarget(int addressingModel, int memoryModel, Set<Capability> allowedCapabilities,
        long maxWorkgroupMemoryBytes, Set<DeviceFeature> allowedFeatures) {

    /** No limit on workgroup memory. */
    public static final long UNLIMITED = Long.MAX_VALUE;

    public SpirvTarget {
        allowedCapabilities = allowedCapabilities == null ? null : Set.copyOf(allowedCapabilities);
        allowedFeatures = allowedFeatures == null ? null : Set.copyOf(allowedFeatures);
        if (maxWorkgroupMemoryBytes < 0) {
            throw new IllegalArgumentException("workgroup memory limit must be >= 0, got " + maxWorkgroupMemoryBytes);
        }
    }

    /** A target with no limit on workgroup memory and no device feature withheld. */
    public SpirvTarget(int addressingModel, int memoryModel, Set<Capability> allowedCapabilities) {
        this(addressingModel, memoryModel, allowedCapabilities, UNLIMITED, null);
    }

    /** No capability restriction — emit whatever the kernel requires (the default). */
    public static SpirvTarget unconstrained() {
        return new SpirvTarget(AddressingModel.Logical.value(), MemoryModel.GLSL450.value(), null);
    }

    /** A budget allowing only the given capabilities (beyond the always-present {@code Shader}). */
    public static SpirvTarget restrictedTo(Set<Capability> capabilities) {
        return new SpirvTarget(AddressingModel.Logical.value(), MemoryModel.GLSL450.value(), capabilities);
    }

    /** This target, allowing at most {@code bytes} of workgroup memory. */
    public SpirvTarget withWorkgroupMemoryLimit(long bytes) {
        return new SpirvTarget(addressingModel, memoryModel, allowedCapabilities, bytes, allowedFeatures);
    }

    /** This target, allowing only {@code features} among the {@link DeviceFeature}s. */
    public SpirvTarget withFeatures(Set<DeviceFeature> features) {
        return new SpirvTarget(addressingModel, memoryModel, allowedCapabilities, maxWorkgroupMemoryBytes, features);
    }

    /** Whether {@code capability} may be emitted. {@code Shader} always may; a null set allows everything. */
    public boolean allows(Capability capability) {
        return capability == Capability.Shader || allowedCapabilities == null
                || allowedCapabilities.contains(capability);
    }

    /** Whether a module may need {@code feature}; a null set allows every one. */
    public boolean allows(DeviceFeature feature) {
        return allowedFeatures == null || allowedFeatures.contains(feature);
    }
}
