package dev.supirvast.vastir.tools;

import dev.supirvast.vast.CoreToTruffle;
import dev.supirvast.vast.CpuKernel;
import dev.supirvast.vastir.core.Barriers;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.UnaryOp;
import dev.supirvast.vastir.lower.CapabilityException;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.lower.DeviceFeature;
import dev.supirvast.vastir.lower.SpirvTarget;
import dev.supirvast.vastir.spirv.Capability;
import dev.supirvast.vastir.tools.NativeTools.ValidationResult;
import dev.supirvast.vastir.type.Type;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * The high-level orchestration front door of supir-vast — the one object a front end holds to run
 * data-parallel kernels, language-neutral and reusable by any Truffle (or plain-Java) caller. It hides the
 * whole pipeline behind two verbs:
 *
 * <ul>
 *   <li>{@link #register} — validate a {@link KernelSpec}, lower it to SPIR-V (and to a CPU Truffle target),
 *       gate the SPIR-V through {@code spirv-val}, <em>preload</em> a resident GPU pipeline when a device is
 *       present, and return a runnable {@link KernelHandle} <em>or</em> a {@link Rejection} witness. The
 *       lowering <em>is</em> the proof of lowerability.</li>
 *   <li>{@link KernelHandle#run} — execute over data, dispatching against the preloaded pipeline on the GPU
 *       when present and falling back to the proven-equivalent CPU path otherwise; the choice is invisible
 *       because the two agree.</li>
 * </ul>
 *
 * <p>The GPU context (instance/device/queue) and each kernel's pipeline are built once and held, so repeated
 * runs re-marshal only data. An {@code Accelerator} owns those native resources and must be {@link #close()
 * closed} (it is {@link AutoCloseable}); without a GPU it holds nothing and closing is a no-op.
 *
 * <p>Validation is treated as living requirements: lowering failures and {@code spirv-val} rejections become
 * concrete witnesses at registration, input/ABI mismatches fail closed at {@link KernelHandle#run}, and
 * equivalence is checkable on demand via {@link KernelHandle#verify}.
 */
public final class Accelerator implements AutoCloseable {

    /**
     * What this host can do — queried, not assumed, so integration is discoverable.
     *
     * @param maxWorkgroupMemoryBytes the device's {@code maxComputeSharedMemorySize}; 0 without a device
     * @param deviceFeatures          the device features no capability distinguishes; empty without a device
     * @param deviceName              the GPU the context runs on, as its driver names it; null without one
     * @param deviceType              {@code discrete}, {@code integrated}, ...; null without a device
     */
    public record Capabilities(boolean gpuAvailable, boolean validationAvailable,
            Set<Capability> deviceCapabilities, long maxWorkgroupMemoryBytes, Set<DeviceFeature> deviceFeatures,
            String deviceName, String deviceType) {}

    private final NativeTools tools = new NativeTools();
    private final Map<KernelHandle, GpuContext.ResidentKernel> pipelines = new IdentityHashMap<>();
    private final List<ResidentBuffer> residentBuffers = new ArrayList<>();
    private final SpirvTarget budget;   // optional caller-imposed capability restriction (#2)
    private Boolean gpuAvailable;       // probed once (probing builds a Vulkan instance — not free)
    private GpuContext context;         // opened lazily on first GPU need, held for this Accelerator's life

    /** An accelerator with no capability restriction — emit whatever a kernel requires. */
    public Accelerator() {
        this(SpirvTarget.unconstrained());
    }

    /**
     * An accelerator that refuses to generate any capability outside {@code budget} — even where the device
     * would support it (portability / ahead-of-time targeting). Kernels needing more are rejected with a witness.
     */
    public Accelerator(SpirvTarget budget) {
        this.budget = budget;
    }

    /**
     * Validates and lowers a kernel, returning a runnable handle on success or a {@link Rejection} witness on
     * failure. On success with a GPU present, the kernel's pipeline is preloaded. Never throws for an
     * unacceptable kernel — the failure is data the caller can render.
     */
    public Registration register(KernelSpec spec) {
        Rejection abiError = checkAbi(spec.columns());
        if (abiError != null) {
            return abiError;
        }

        // The GPU lowers a guarded copy when workgroups are wider than one; the CPU lowers the kernel as given,
        // since it runs exactly n invocations and has no tail to stop. A kernel with a barrier is the
        // exception on both: the tail cannot return before a barrier the rest of its workgroup must reach, so
        // it runs whole workgroups everywhere and bounds itself with Expr.InvocationCount.
        int size = spec.workgroupSize();
        boolean guard = size > 1 && !Barriers.contains(spec.kernel().body());
        Function gpuKernel = guard ? guarded(spec.kernel()) : spec.kernel();
        CoreModule coreModule = new CoreModule().addEntryPoint(EntryPoint.compute(gpuKernel, size, 1, 1));
        boolean gpu = gpuAvailable();
        // Effective target = the caller's budget, narrowed to what this device supports when we have one.
        SpirvTarget target = gpu ? deviceConstrained(context()) : budget;
        byte[] spirv;
        boolean preloadable;
        try {
            spirv = new CoreToSpirv().lower(coreModule, target).toByteArray();
            preloadable = gpu;
        } catch (CapabilityException withinDeviceAndBudget) {
            if (!gpu) {
                return new Rejection("requires a capability outside the target budget",
                        withinDeviceAndBudget.getMessage());
            }
            // device ∩ budget rejected it. Is it the budget (hard refuse) or only the device (the kernel is
            // fine — register CPU-only so run() transparently falls back to the proven-equal CPU path)?
            try {
                spirv = new CoreToSpirv().lower(coreModule, budget).toByteArray();
                preloadable = false;
            } catch (CapabilityException outsideBudget) {
                return new Rejection("requires a capability outside the target budget", outsideBudget.getMessage());
            } catch (RuntimeException e) {
                return new Rejection("not lowerable to SPIR-V", String.valueOf(e.getMessage()));
            }
        } catch (RuntimeException e) {
            return new Rejection("not lowerable to SPIR-V", String.valueOf(e.getMessage()));
        }

        if (tools.isAvailable()) {
            ValidationResult validation = tools.validate(spirv);
            if (!validation.valid()) {
                return new Rejection("spirv-val rejected the kernel", validation.output());
            }
        }

        CpuKernel cpuTarget;
        try {
            List<Buffer> buffers = spec.columns().stream()
                    .map(c -> new Buffer(c.name(), c.binding(), c.type()))
                    .toList();
            cpuTarget = new CoreToTruffle().lowerDispatch(spec.kernel(), buffers, size);
        } catch (RuntimeException e) {
            return new Rejection("not lowerable to the CPU backend", String.valueOf(e.getMessage()));
        }

        KernelHandle handle = new KernelHandle(this, spec, spirv, cpuTarget);
        if (preloadable) {
            try {
                pipelines.put(handle, context().build(spirv, spec.entryPoint(), spec.columns().size(), size));
            } catch (RuntimeException e) {
                return new Rejection("GPU pipeline build failed", String.valueOf(e.getMessage()));
            }
        }
        return handle;
    }

    /** What this host can do right now, including the device's supported SPIR-V capabilities. */
    public Capabilities capabilities() {
        Set<Capability> device = gpuAvailable() ? context().capabilities() : Set.of();
        long workgroupMemory = gpuAvailable() ? context().maxWorkgroupMemoryBytes() : 0;
        Set<DeviceFeature> features = gpuAvailable() ? context().features() : Set.of();
        String name = gpuAvailable() ? context().deviceName() : null;
        String type = gpuAvailable() ? context().deviceType() : null;
        return new Capabilities(gpuAvailable(), tools.isAvailable(), device, workgroupMemory, features, name, type);
    }

    /**
     * A {@link ResidentBuffer} of {@code elements} elements of {@code element}: device-local memory when a GPU
     * is present, a host array the CPU backend runs over in place when not. Its contents are undefined on the
     * device and zero on the host until written. Closed by {@link #close} if the caller has not.
     */
    public ResidentBuffer allocate(Type element, int elements) {
        if (!isSupportedColumnType(element)) {
            throw new IllegalArgumentException("unsupported element type for a resident buffer: " + element);
        }
        if (elements < 1) {
            throw new IllegalArgumentException("a resident buffer needs at least one element, got " + elements);
        }
        ResidentBuffer buffer = gpuAvailable()
                ? ResidentBuffer.onDevice(context(), element, elements)
                : ResidentBuffer.onHost(element, elements);
        residentBuffers.add(buffer);
        return buffer;
    }

    /** Releases resident buffers, pipelines and the context. Safe to call when no GPU was ever used. */
    @Override
    public void close() {
        if (context != null) {
            context.finish();   // nothing may be destroyed under a dispatch still running
        }
        residentBuffers.forEach(ResidentBuffer::close);
        residentBuffers.clear();
        pipelines.values().forEach(GpuContext.ResidentKernel::close);
        pipelines.clear();
        if (context != null) {
            context.close();
            context = null;
        }
    }

    /**
     * Releases the resident GPU pipeline for a single {@code handle} — its shader module, pipeline,
     * layouts and descriptor pool — reclaiming that device memory without tearing down the context or
     * any other kernel. Returns whether a pipeline was actually held ({@code false} for a CPU-only
     * handle, or one already released). After release the handle is <em>degraded, not dead</em>:
     * {@link KernelHandle#run} still works, falling back to the proven-equivalent CPU path; call
     * {@link #register} again to get a resident GPU pipeline back.
     *
     * <p>Like {@link #register} and {@link KernelHandle#run}, this touches the context's native state
     * and must be called on the accelerator's owning thread, and not while a run of {@code handle} is in
     * flight (the pipeline it is executing would be destroyed underneath it).
     */
    public boolean release(KernelHandle handle) {
        GpuContext.ResidentKernel pipeline = pipelines.remove(handle);
        if (pipeline == null) {
            return false;
        }
        context().finish();   // a resident dispatch of this kernel may still be executing its pipeline
        pipeline.close();
        return true;
    }

    // --- internals -------------------------------------------------------------------------------------

    /** Dispatches {@code handle}'s preloaded pipeline against the columns; called by {@link KernelHandle}. */
    int[][] dispatchGpu(KernelHandle handle, int[][] columns, int n) {
        GpuContext.ResidentKernel pipeline = pipelines.get(handle);
        if (pipeline == null) {
            throw new IllegalStateException("no GPU pipeline for this kernel (not preloaded)");
        }
        return context().dispatch(pipeline, columns, n);
    }

    /** Submits {@code handle}'s preloaded pipeline without waiting; called by {@link KernelHandle#submitAsync}. */
    Submission submitGpu(KernelHandle handle, int[][] columns, int n) {
        GpuContext.ResidentKernel pipeline = pipelines.get(handle);
        if (pipeline == null) {
            throw new IllegalStateException("no GPU pipeline for this kernel (not preloaded)");
        }
        return context().submitAsync(pipeline, columns, n);
    }

    /** Submits {@code handle}'s pipeline against resident device buffers; called by {@link KernelHandle#dispatch}. */
    void dispatchResident(KernelHandle handle, GpuContext.DeviceBuffer[] buffers, int n) {
        GpuContext.ResidentKernel pipeline = pipelines.get(handle);
        if (pipeline == null) {
            throw new IllegalStateException("no GPU pipeline for this kernel (not preloaded)");
        }
        context().dispatchResident(pipeline, buffers, n);
    }

    /** Awaits a {@link #submitGpu} submission and reads its results; called by {@link KernelHandle#await}. */
    int[][] awaitGpu(Submission submission) {
        return context().await(submission);
    }

    boolean hasPipeline(KernelHandle handle) {
        return pipelines.containsKey(handle);
    }

    /** The caller's budget, intersected with what the device actually supports. */
    private SpirvTarget deviceConstrained(GpuContext device) {
        Set<Capability> deviceCapabilities = device.capabilities();
        Set<Capability> budgetCaps = budget.allowedCapabilities();
        Set<Capability> effective = budgetCaps == null
                ? deviceCapabilities
                : budgetCaps.stream().filter(deviceCapabilities::contains).collect(Collectors.toSet());
        Set<DeviceFeature> features = device.features().stream().filter(budget::allows).collect(Collectors.toSet());
        return SpirvTarget.restrictedTo(effective)
                .withWorkgroupMemoryLimit(Math.min(budget.maxWorkgroupMemoryBytes(), device.maxWorkgroupMemoryBytes()))
                .withFeatures(features);
    }

    boolean gpuAvailable() {
        Boolean cached = gpuAvailable;
        if (cached == null) {
            cached = GpuContext.isAvailable();
            gpuAvailable = cached;
        }
        return cached;
    }

    private GpuContext context() {
        if (context == null) {
            context = GpuContext.open();
        }
        return context;
    }

    /**
     * {@code kernel} with a first statement returning from every invocation at or past the requested count,
     * which a dispatch rounded up to whole workgroups adds. The count is {@link Expr.InvocationCount}, a push
     * constant, so one pipeline serves every {@code n}; {@link GpuContext} declares the range and sets it per
     * dispatch. A kernel handed to this class cannot already own a push constant block — the CPU backend has
     * no push constants, so such a kernel is rejected before it gets here.
     */
    private static Function guarded(Function kernel) {
        Statement stopTheTail = new Statement.If(
                new Expr.Unary(UnaryOp.LOGICAL_NOT,
                        new Expr.Binary(BinaryOp.LESS_THAN, new Expr.InvocationId(), new Expr.InvocationCount())),
                Region.of(new Statement.ReturnVoid()),
                Region.of());
        List<Statement> body = new ArrayList<>();
        body.add(stopTheTail);
        body.addAll(kernel.body().statements());
        return new Function(kernel.name(), kernel.signature(), new Region(body));
    }

    /** Columns must be bound 0..n-1 (binding == slot == int[][] index) with at least one output. */
    private static Rejection checkAbi(List<KernelColumn> columns) {
        if (columns.isEmpty()) {
            return new Rejection("empty kernel interface", "a kernel needs at least one column");
        }
        boolean anyOutput = false;
        for (int i = 0; i < columns.size(); i++) {
            KernelColumn column = columns.get(i);
            if (column.binding() != i) {
                return new Rejection("non-contiguous column bindings",
                        "column '" + column.name() + "' is at position " + i + " but binds " + column.binding()
                                + "; bindings must be 0..n-1 in order");
            }
            if (!isSupportedColumnType(column.type())) {
                return new Rejection("unsupported column element type",
                        "column '" + column.name() + "' is " + column.type() + "; the columnar marshalling "
                                + "currently carries 32- and 64-bit scalars (i32/u32/f32/i64/u64/f64)");
            }
            anyOutput |= column.isOutput();
        }
        if (!anyOutput) {
            return new Rejection("no output column", "a kernel must write at least one output column");
        }
        return null;
    }

    /** Scalar column element types the columnar wire carries: 32- and 64-bit ints and floats. */
    private static boolean isSupportedColumnType(Type type) {
        if (type instanceof Type.Int i) {
            return i.width() == 32 || i.width() == 64;
        }
        return type instanceof Type.Float f && (f.width() == 32 || f.width() == 64);
    }
}

