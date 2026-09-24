package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.Function;

import java.util.List;

/**
 * What a front end hands to the {@link Accelerator}: a data-parallel {@code core} kernel, the
 * {@link KernelColumn columns} it reads and writes, and how wide its workgroups are on the GPU. The kernel is
 * a {@code void} function indexed per invocation (via {@code Expr.InvocationId} and {@code Buffer} access);
 * the columns name the buffers it touches, in binding order. This is the highest-level, language-neutral
 * description of a kernel — the {@link Accelerator} turns it into validated, preloaded, runnable form.
 *
 * <p>For most kernels the workgroup size is a GPU scheduling choice and never a change of meaning: the kernel
 * still runs once per invocation, and the {@link Accelerator} stops the invocations a rounded-up dispatch adds
 * past the end. It defaults to {@link #DEFAULT_WORKGROUP_SIZE}, a whole number of lanes on the common
 * hardware, where a workgroup of one leaves most of each SIMD unit idle.
 *
 * <p>A kernel that uses workgroup memory or the workgroup indices is the exception, since the size is then
 * part of what it computes. And a kernel with a barrier runs whole workgroups on both backends, because the
 * invocations past the end cannot return before a barrier the rest of their workgroup must reach. Such a
 * kernel reads the requested count as {@code Expr.InvocationCount} and bounds itself.
 *
 * <p>The subgroup size matters only to a kernel with subgroup operations or indices, and there it is part of
 * the meaning too: which invocations a reduction combines. So it is the kernel's to fix, not the device's to
 * choose. The GPU is required to run full subgroups of exactly that size, the CPU runs the same, and a device
 * that cannot is not used for the kernel. It defaults to {@link #DEFAULT_SUBGROUP_SIZE}, and the workgroup of
 * such a kernel must be a whole number of subgroups.
 */
public record KernelSpec(Function kernel, List<KernelColumn> columns, int workgroupSize, int subgroupSize) {

    /** 64: two NVIDIA warps, one AMD wavefront — full lanes on both without tuning. */
    public static final int DEFAULT_WORKGROUP_SIZE = 64;

    /** 32: an NVIDIA warp, an AMD RDNA wave32, and a size current Intel GPUs can be asked for. */
    public static final int DEFAULT_SUBGROUP_SIZE = 32;

    public KernelSpec {
        if (kernel == null) {
            throw new IllegalArgumentException("kernel must be set");
        }
        if (workgroupSize < 1) {
            throw new IllegalArgumentException("workgroup size must be >= 1, got " + workgroupSize);
        }
        if (Integer.bitCount(subgroupSize) != 1) {
            throw new IllegalArgumentException("subgroup size must be a power of two, got " + subgroupSize);
        }
        columns = List.copyOf(columns);
    }

    /** A kernel with the {@linkplain #DEFAULT_SUBGROUP_SIZE default subgroup size}. */
    public KernelSpec(Function kernel, List<KernelColumn> columns, int workgroupSize) {
        this(kernel, columns, workgroupSize, DEFAULT_SUBGROUP_SIZE);
    }

    /** A kernel with the {@linkplain #DEFAULT_WORKGROUP_SIZE default workgroup size}. */
    public KernelSpec(Function kernel, List<KernelColumn> columns) {
        this(kernel, columns, DEFAULT_WORKGROUP_SIZE);
    }

    /** This kernel with workgroups {@code size} invocations wide. */
    public KernelSpec withWorkgroupSize(int size) {
        return new KernelSpec(kernel, columns, size, subgroupSize);
    }

    /** This kernel with subgroups of {@code size} lanes. */
    public KernelSpec withSubgroupSize(int size) {
        return new KernelSpec(kernel, columns, workgroupSize, size);
    }

    /** The kernel's entry-point name, as used in the emitted SPIR-V. */
    public String entryPoint() {
        return kernel.name();
    }
}
