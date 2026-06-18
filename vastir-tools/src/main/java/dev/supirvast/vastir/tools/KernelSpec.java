package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.Function;

import java.util.List;

/**
 * What a front end hands to the {@link Accelerator}: a data-parallel {@code core} kernel and the
 * struct-of-arrays {@link KernelColumn columns} it reads and writes. The kernel is a {@code void} function
 * indexed per invocation (via {@code Expr.InvocationId} and {@code Buffer} access); the columns name the
 * buffers it touches, in binding order. This is the highest-level, language-neutral description of a kernel —
 * the {@link Accelerator} turns it into validated, preloaded, runnable form.
 */
public record KernelSpec(Function kernel, List<KernelColumn> columns) {

    public KernelSpec {
        if (kernel == null) {
            throw new IllegalArgumentException("kernel must be set");
        }
        columns = List.copyOf(columns);
    }

    /** The kernel's entry-point name, as used in the emitted SPIR-V. */
    public String entryPoint() {
        return kernel.name();
    }
}
