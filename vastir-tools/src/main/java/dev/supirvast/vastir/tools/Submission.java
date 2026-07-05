package dev.supirvast.vastir.tools;

import org.lwjgl.vulkan.VkCommandBuffer;

/**
 * A GPU dispatch in flight — the opaque handle returned by {@link KernelHandle#submitAsync} (and, beneath
 * it, {@link GpuContext#submitAsync}) and consumed once by {@link KernelHandle#await}. It carries the fence
 * to wait on and the per-dispatch device resources (command buffer, storage buffers and their memory) that
 * {@code await} reads back and frees. Several submissions can be outstanding at once — that is the whole
 * point — so the caller can launch a batch of distinct kernels and then await them all.
 *
 * <p>Not constructed directly; the fields are package-internal, filled by {@link GpuContext#submitAsync} and
 * torn down by {@link GpuContext#await}.
 */
public final class Submission {

    final GpuContext.ResidentKernel kernel;
    final VkCommandBuffer cmd;
    final long fence;
    final long[] bufferHandles;
    final long[] memoryHandles;
    final int[] bufferLengths;
    boolean awaited;

    Submission(GpuContext.ResidentKernel kernel, VkCommandBuffer cmd, long fence, long[] bufferHandles,
            long[] memoryHandles, int[] bufferLengths) {
        this.kernel = kernel;
        this.cmd = cmd;
        this.fence = fence;
        this.bufferHandles = bufferHandles;
        this.memoryHandles = memoryHandles;
        this.bufferLengths = bufferLengths;
    }
}
