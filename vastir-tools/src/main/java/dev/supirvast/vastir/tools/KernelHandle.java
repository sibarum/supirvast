package dev.supirvast.vastir.tools;

import com.oracle.truffle.api.CallTarget;
import dev.supirvast.vastir.type.Type;

import java.util.Arrays;
import java.util.List;

/**
 * A registered, runnable kernel — the product of a successful {@link Accelerator#register}. It holds both
 * lowerings of the one {@code core} kernel (validated SPIR-V for the GPU, a Truffle target for the CPU) and
 * orchestrates execution: {@link #run} picks a backend and marshals data; {@link #verify} runs both and checks
 * they agree. A handle only exists because lowering succeeded, so there is no invalid runnable state.
 *
 * <p>Data is struct-of-arrays: {@code columns[i]} is the buffer bound at binding {@code i} (its
 * {@link KernelColumn}), holding one i32 per invocation. {@link #run} never mutates the caller's arrays.
 */
public final class KernelHandle implements Registration {

    /** Which backend executed a run. The two are proven equivalent, so the choice never changes results. */
    public enum Backend { GPU, CPU }

    /** Outcome of {@link #verify}: whether CPU and GPU agreed (or {@code skipped} when no GPU is present). */
    public record VerificationResult(boolean verified, boolean matches, String detail) {
        public static VerificationResult skipped() {
            return new VerificationResult(false, true, "no GPU present; equivalence not checked here");
        }
    }

    private final Accelerator accelerator;
    private final KernelSpec spec;
    private final byte[] spirv;
    private final CallTarget cpuTarget;

    KernelHandle(Accelerator accelerator, KernelSpec spec, byte[] spirv, CallTarget cpuTarget) {
        this.accelerator = accelerator;
        this.spec = spec;
        this.spirv = spirv;
        this.cpuTarget = cpuTarget;
    }

    /** The kernel's interface — the columns, in binding order — for a front end to marshal against. */
    public List<KernelColumn> abi() {
        return spec.columns();
    }

    /** The validated SPIR-V module, for advanced use (cross-compile, inspection, optimization). */
    public byte[] spirv() {
        return spirv.clone();
    }

    /**
     * Runs the kernel over {@code n} invocations against the given columns, choosing the GPU when available
     * and falling back to the equivalent CPU path otherwise. Returns fresh column arrays; the caller's input
     * is left untouched.
     */
    public int[][] run(int[][] columns, int n) {
        validate(columns, n);
        return onGpu() ? runGpu(columns, n) : runCpu(columns, n);
    }

    /** The backend {@link #run} would choose right now (GPU if a pipeline was preloaded for it, else CPU). */
    public Backend preferredBackend() {
        return onGpu() ? Backend.GPU : Backend.CPU;
    }

    /** GPU only when a device is present <em>and</em> this kernel's pipeline was preloaded (caps fit). */
    private boolean onGpu() {
        return accelerator.gpuAvailable() && accelerator.hasPipeline(this);
    }

    /**
     * Runs the kernel on both backends and checks the outputs match — the differential equivalence guarantee,
     * available on demand rather than only in the test suite. Skipped (reported, not failed) when no GPU is
     * present.
     */
    public VerificationResult verify(int[][] columns, int n) {
        validate(columns, n);
        if (!onGpu()) {
            return VerificationResult.skipped();
        }
        int[][] cpu = runCpu(columns, n);
        int[][] gpu = runGpu(columns, n);
        for (int i = 0; i < cpu.length; i++) {
            if (!Arrays.equals(cpu[i], gpu[i])) {
                KernelColumn column = spec.columns().get(i);
                return new VerificationResult(true, false,
                        "column '" + column.name() + "' (binding " + column.binding() + ") differs: CPU="
                                + Arrays.toString(cpu[i]) + " GPU=" + Arrays.toString(gpu[i]));
            }
        }
        return new VerificationResult(true, true, "CPU and GPU agree on all columns");
    }

    private int[][] runGpu(int[][] columns, int n) {
        // Dispatch reads inputs and returns fresh arrays (the caller's input is never written), against the
        // pipeline preloaded at registration — no per-run lowering or device build.
        return accelerator.dispatchGpu(this, columns, n);
    }

    private int[][] runCpu(int[][] columns, int n) {
        int[][] work = copy(columns);
        for (int i = 0; i < n; i++) {
            cpuTarget.call(i, work);
        }
        return work;
    }

    private void validate(int[][] columns, int n) {
        if (columns == null) {
            throw new IllegalArgumentException("columns must not be null");
        }
        if (columns.length != spec.columns().size()) {
            throw new IllegalArgumentException("expected " + spec.columns().size() + " columns ("
                    + spec.columns().stream().map(KernelColumn::name).toList() + "), got " + columns.length);
        }
        if (n < 0) {
            throw new IllegalArgumentException("invocation count must be >= 0, got " + n);
        }
        for (int i = 0; i < columns.length; i++) {
            if (columns[i] == null) {
                throw new IllegalArgumentException("column " + i + " ('"
                        + spec.columns().get(i).name() + "') is null");
            }
            int needed = n * wordsPerElement(spec.columns().get(i).type());
            if (columns[i].length < needed) {
                throw new IllegalArgumentException("column " + i + " ('" + spec.columns().get(i).name()
                        + "') has length " + columns[i].length + " < " + needed + " words needed for " + n
                        + " elements");
            }
        }
    }

    /** 32-bit elements occupy one word on the int[] wire; 64-bit elements occupy two. */
    private static int wordsPerElement(Type element) {
        int width = element instanceof Type.Int i ? i.width() : element instanceof Type.Float f ? f.width() : 32;
        return width == 64 ? 2 : 1;
    }

    private static int[][] copy(int[][] columns) {
        int[][] out = new int[columns.length][];
        for (int i = 0; i < columns.length; i++) {
            out[i] = columns[i].clone();
        }
        return out;
    }
}
