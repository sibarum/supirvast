package dev.supirvast.vastir.tools;

import com.oracle.truffle.api.CallTarget;
import dev.supirvast.vast.CoreToTruffle;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.lower.CapabilityException;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.lower.SpirvTarget;
import dev.supirvast.vastir.spirv.Capability;
import dev.supirvast.vastir.tools.NativeTools.ValidationResult;
import dev.supirvast.vastir.type.Type;

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

    /** What this host can do — queried, not assumed, so integration is discoverable. */
    public record Capabilities(boolean gpuAvailable, boolean validationAvailable,
            Set<Capability> deviceCapabilities) {}

    private final NativeTools tools = new NativeTools();
    private final Map<KernelHandle, GpuContext.ResidentKernel> pipelines = new IdentityHashMap<>();
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

        CoreModule coreModule = new CoreModule().addEntryPoint(EntryPoint.compute(spec.kernel(), 1, 1, 1));
        boolean gpu = gpuAvailable();
        // Effective target = the caller's budget, narrowed to what this device supports when we have one.
        SpirvTarget target = gpu ? deviceConstrained(context().capabilities()) : budget;
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

        CallTarget cpuTarget;
        try {
            List<Buffer> buffers = spec.columns().stream()
                    .map(c -> new Buffer(c.name(), c.binding(), c.type()))
                    .toList();
            cpuTarget = new CoreToTruffle().lowerKernel(spec.kernel(), buffers);
        } catch (RuntimeException e) {
            return new Rejection("not lowerable to the CPU backend", String.valueOf(e.getMessage()));
        }

        KernelHandle handle = new KernelHandle(this, spec, spirv, cpuTarget);
        if (preloadable) {
            try {
                pipelines.put(handle, context().build(spirv, spec.entryPoint(), spec.columns().size()));
            } catch (RuntimeException e) {
                return new Rejection("GPU pipeline build failed", String.valueOf(e.getMessage()));
            }
        }
        return handle;
    }

    /** What this host can do right now, including the device's supported SPIR-V capabilities. */
    public Capabilities capabilities() {
        Set<Capability> device = gpuAvailable() ? context().capabilities() : Set.of();
        return new Capabilities(gpuAvailable(), tools.isAvailable(), device);
    }

    /** Releases the resident GPU pipelines and context. Safe to call when no GPU was ever used. */
    @Override
    public void close() {
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

    boolean hasPipeline(KernelHandle handle) {
        return pipelines.containsKey(handle);
    }

    /** The caller's budget, intersected with what the device actually supports. */
    private SpirvTarget deviceConstrained(Set<Capability> deviceCapabilities) {
        Set<Capability> budgetCaps = budget.allowedCapabilities();
        Set<Capability> effective = budgetCaps == null
                ? deviceCapabilities
                : budgetCaps.stream().filter(deviceCapabilities::contains).collect(Collectors.toSet());
        return SpirvTarget.restrictedTo(effective);
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

