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
 */
public record KernelSpec(Function kernel, List<KernelColumn> columns, int workgroupSize) {

    /** 64: two NVIDIA warps, one AMD wavefront — full lanes on both without tuning. */
    public static final int DEFAULT_WORKGROUP_SIZE = 64;

    public KernelSpec {
        if (kernel == null) {
            throw new IllegalArgumentException("kernel must be set");
        }
        if (workgroupSize < 1) {
            throw new IllegalArgumentException("workgroup size must be >= 1, got " + workgroupSize);
        }
        columns = List.copyOf(columns);
    }

    /** A kernel with the {@linkplain #DEFAULT_WORKGROUP_SIZE default workgroup size}. */
    public KernelSpec(Function kernel, List<KernelColumn> columns) {
        this(kernel, columns, DEFAULT_WORKGROUP_SIZE);
    }

    /** This kernel with workgroups {@code size} invocations wide. */
    public KernelSpec withWorkgroupSize(int size) {
        return new KernelSpec(kernel, columns, size);
    }

    /** The kernel's entry-point name, as used in the emitted SPIR-V. */
    public String entryPoint() {
        return kernel.name();
    }
}
