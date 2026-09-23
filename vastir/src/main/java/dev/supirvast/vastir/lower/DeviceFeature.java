package dev.supirvast.vastir.lower;

/**
 * A device feature a kernel can need that no SPIR-V capability distinguishes — so a {@link SpirvTarget}
 * budgets it separately, and a lowering that needs one outside the target fails with a {@link
 * CapabilityException} exactly as a missing capability does.
 *
 * <p>Float atomics are the case. {@code AtomicFloat32AddEXT} is one capability whether the pointer is to a
 * storage buffer or to workgroup memory, but Vulkan licenses the two with separate features
 * ({@code shaderBufferFloat32AtomicAdd}, {@code shaderSharedFloat32AtomicAdd}), and a device may have one
 * without the other. The capability says the instruction exists; these say where it may point.
 */
public enum DeviceFeature {

    /** {@code shaderBufferFloat32AtomicAdd}: float add on a storage buffer. */
    BUFFER_FLOAT32_ATOMIC_ADD,

    /** {@code shaderBufferFloat32AtomicMinMax}: float min and max on a storage buffer. */
    BUFFER_FLOAT32_ATOMIC_MIN_MAX,

    /** {@code shaderSharedFloat32Atomics}: float exchange on workgroup memory. */
    SHARED_FLOAT32_ATOMICS,

    /** {@code shaderSharedFloat32AtomicAdd}: float add on workgroup memory. */
    SHARED_FLOAT32_ATOMIC_ADD,

    /** {@code shaderSharedFloat32AtomicMinMax}: float min and max on workgroup memory. */
    SHARED_FLOAT32_ATOMIC_MIN_MAX
}
