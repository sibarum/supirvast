package dev.supirvast.vastir.tools;

/**
 * One-shot Vulkan compute execution — a thin convenience wrapper over {@link GpuContext} that opens a context,
 * builds a pipeline, dispatches once, and tears everything down. Useful for verification and tests; for
 * repeated dispatch of a fixed kernel set, hold a {@link GpuContext} (or use {@link Accelerator}) so the
 * instance/device/pipeline are built once and reused.
 *
 * <p>Headless — compute only, no surface/swapchain. Requires Vulkan 1.3 (to consume SPIR-V 1.6).
 */
public final class VulkanCompute {

    /** Whether a Vulkan 1.3 device with a compute queue is usable on this machine. */
    public boolean isAvailable() {
        return GpuContext.isAvailable();
    }

    /** Runs the shader's {@code entryPoint} with a 1×1×1 dispatch and returns {@code buffer[0]} (binding 0). */
    public int execute(byte[] spirv, String entryPoint) {
        try (GpuContext context = GpuContext.open();
                GpuContext.ResidentKernel kernel = context.build(spirv, entryPoint, 1)) {
            return context.dispatch(kernel, new int[][] {new int[1]}, 1)[0][0];
        }
    }

    /**
     * Runs a data-parallel kernel. {@code buffers[i]} is the storage buffer at binding {@code i} (inputs
     * pre-filled, outputs sized and zeroed); the kernel is dispatched with {@code groupCountX} workgroups.
     * Returns the buffers' contents read back after execution.
     */
    public int[][] executeKernel(byte[] spirv, String entryPoint, int[][] buffers, int groupCountX) {
        try (GpuContext context = GpuContext.open();
                GpuContext.ResidentKernel kernel = context.build(spirv, entryPoint, buffers.length)) {
            return context.dispatch(kernel, buffers, groupCountX);
        }
    }
}
