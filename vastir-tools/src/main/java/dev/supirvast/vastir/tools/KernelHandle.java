package dev.supirvast.vastir.tools;

import dev.supirvast.vast.CpuKernel;
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
public final class KernelHandle implements Registration, AutoCloseable {

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
    private final CpuKernel cpuTarget;

    KernelHandle(Accelerator accelerator, KernelSpec spec, byte[] spirv, CpuKernel cpuTarget) {
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

    /**
     * Runs the kernel over {@code n} invocations against resident buffers, one per column in binding order,
     * updating them where they live. Nothing is copied in or out: a stepped kernel dispatches repeatedly and
     * the host {@link ResidentBuffer#read reads} only when it wants to look. On the GPU this returns without
     * waiting, and every later dispatch, read or write is ordered after it.
     *
     * <p>Where the kernel cannot run on the GPU — no device, a capability the device lacks, a released
     * pipeline — it runs on the CPU instead, over host buffers in place, or over device buffers by reading
     * them, running, and writing them back. Slower, never wrong.
     */
    public void dispatch(List<ResidentBuffer> buffers, int n) {
        validateResident(buffers, n);
        boolean allOnDevice = buffers.stream().allMatch(ResidentBuffer::onDevice);
        if (onGpu() && allOnDevice) {
            GpuContext.DeviceBuffer[] devices = new GpuContext.DeviceBuffer[buffers.size()];
            for (int i = 0; i < devices.length; i++) {
                devices[i] = buffers.get(i).device();
            }
            accelerator.dispatchResident(this, devices, n);
            return;
        }
        int[][] work = new int[buffers.size()][];
        for (int i = 0; i < work.length; i++) {
            ResidentBuffer buffer = buffers.get(i);
            work[i] = buffer.onDevice() ? buffer.read() : buffer.hostArray();
        }
        cpuTarget.dispatch(work, n);
        for (int i = 0; i < work.length; i++) {
            if (buffers.get(i).onDevice()) {
                buffers.get(i).write(work[i]);
            }
        }
    }

    /** The backend {@link #run} would choose right now (GPU if a pipeline was preloaded for it, else CPU). */
    public Backend preferredBackend() {
        return onGpu() ? Backend.GPU : Backend.CPU;
    }

    /**
     * Submits this kernel over {@code n} invocations <em>without waiting</em>, returning a {@link Submission}
     * to {@link #await}. This is the concurrency path: submit several distinct kernels, then await them all,
     * and their device execution overlaps (round-robined across the device's compute queues). Unlike
     * {@link #run} it is <b>GPU-only</b> — it throws when no resident pipeline backs this handle, since the
     * CPU fallback is synchronous; use {@link #run} for the backend-portable path, and check
     * {@link #preferredBackend()} first if unsure.
     *
     * <p>Owning-thread only, and a given handle may have only one submission in flight at a time (its single
     * descriptor set) — a second {@code submitAsync} before {@link #await} throws. Distinct handles run
     * concurrently freely. Does not mutate the caller's arrays.
     */
    public Submission submitAsync(int[][] columns, int n) {
        validate(columns, n);
        if (!onGpu()) {
            throw new IllegalStateException("submitAsync is GPU-only (no resident pipeline for this kernel); "
                    + "use run() for the backend-portable path");
        }
        return accelerator.submitGpu(this, columns, n);
    }

    /**
     * Blocks until {@code submission} (from this handle's {@link #submitAsync}) completes and returns its
     * fresh output columns, freeing the dispatch's device resources. Each submission is awaited exactly once.
     */
    public int[][] await(Submission submission) {
        return accelerator.awaitGpu(submission);
    }

    /**
     * Releases this kernel's resident GPU pipeline, reclaiming its device resources without affecting the
     * accelerator's context or any other kernel — {@link Accelerator#release(KernelHandle)} for one
     * handle. Idempotent (a no-op once released or for a CPU-only handle). After close the kernel is
     * degraded, not dead: {@link #run} still works via the equivalent CPU path. Must be called on the
     * accelerator's owning thread and not while a run of this kernel is in flight.
     */
    @Override
    public void close() {
        accelerator.release(this);
    }

    /** The accelerator that registered this kernel, and whose context its pipeline lives in. */
    Accelerator owner() {
        return accelerator;
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
        cpuTarget.dispatch(work, n);
        return work;
    }

    void validateResident(List<ResidentBuffer> buffers, int n) {
        if (buffers == null || buffers.size() != spec.columns().size()) {
            throw new IllegalArgumentException("expected " + spec.columns().size() + " resident buffers ("
                    + spec.columns().stream().map(KernelColumn::name).toList() + "), got "
                    + (buffers == null ? "null" : buffers.size()));
        }
        if (n < 0) {
            throw new IllegalArgumentException("invocation count must be >= 0, got " + n);
        }
        for (int i = 0; i < buffers.size(); i++) {
            KernelColumn column = spec.columns().get(i);
            ResidentBuffer buffer = buffers.get(i);
            if (buffer == null || buffer.isClosed()) {
                throw new IllegalArgumentException("resident buffer " + i + " ('" + column.name() + "') is "
                        + (buffer == null ? "null" : "closed"));
            }
            if (!buffer.element().equals(column.type())) {
                throw new IllegalArgumentException("resident buffer " + i + " ('" + column.name() + "') holds "
                        + buffer.element() + ", but the column is " + column.type());
            }
            if (buffer.elements() < column.elementsFor(n)) {
                throw new IllegalArgumentException("resident buffer " + i + " ('" + column.name() + "') holds "
                        + buffer.elements() + " elements < " + column.elementsFor(n) + " needed for " + n
                        + " invocations");
            }
        }
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
            KernelColumn column = spec.columns().get(i);
            int elements = column.elementsFor(n);
            int needed = elements * wordsPerElement(column.type());
            if (columns[i].length < needed) {
                throw new IllegalArgumentException("column " + i + " ('" + column.name()
                        + "') has length " + columns[i].length + " < " + needed + " words needed for " + elements
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
