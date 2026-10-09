package dev.supirvast.vastir.tools;

import dev.supirvast.vast.CoreToTruffle;
import dev.supirvast.vast.CpuKernel;
import dev.supirvast.vastir.pass.Buffered;
import dev.supirvast.vastir.pass.Pass;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Runs a {@link Buffered} program's pass lists over its named buffers, on one backend chosen by the caller.
 *
 * <ul>
 *   <li>{@link #gpu}: the buffers resident on an {@link Accelerator}, each pass's kernel registered once, and each
 *       pass list recorded once as a {@link DispatchSequence}, so a run is one submission however many passes it
 *       holds. Where the accelerator has no GPU, the sequence falls back to the CPU as it always does.</li>
 *   <li>{@link #cpu}: the same passes lowered by {@link CoreToTruffle} and run over host arrays, which the passes
 *       share in place. A run on the CPU even where there is a GPU: the backend a test judges on its own, or a
 *       reference to hold the GPU's run against.</li>
 * </ul>
 *
 * <p>Every pass is registered with the workgroup and subgroup sizes given here. A kernel that needs neither
 * still needs a workgroup; give {@link #NO_SUBGROUP} for a program with no subgroup operations, and the GPU is
 * then left to choose.
 *
 * <h2>Threads</h2>
 *
 * Making a runner allocates its buffers, and {@link #prepare} registers kernels; both may happen on any thread,
 * as the accelerator allows, so the slow part of starting a program is off the owning thread. Everything else —
 * {@link #clear}, {@link #run}, reads, writes, {@link #finish} and {@link #close} — is the owning thread's.
 */
public abstract sealed class PassRunner implements AutoCloseable permits PassRunner.Gpu, PassRunner.Cpu {

    /** The subgroup size to give a program with no subgroup operations. */
    public static final int NO_SUBGROUP = 0;

    final Buffered program;
    final int workgroup;
    final int subgroup;

    private PassRunner(Buffered program, int workgroup, int subgroup) {
        if (workgroup < 1 || subgroup < 0) {
            throw new IllegalArgumentException("a workgroup of at least one, and a subgroup of zero or more, got "
                    + workgroup + " and " + subgroup);
        }
        this.program = program;
        this.workgroup = workgroup;
        this.subgroup = subgroup;
    }

    /** {@code program}'s buffers on {@code accelerator}, which the runner uses and does not close. */
    public static PassRunner gpu(Accelerator accelerator, Buffered program, int workgroup, int subgroup) {
        return new Gpu(accelerator, program, workgroup, subgroup);
    }

    /** {@code program}'s buffers as host arrays, its passes run by the CPU backend. */
    public static PassRunner cpu(Buffered program, int workgroup, int subgroup) {
        return new Cpu(program, workgroup, subgroup);
    }

    /** The program this runs. */
    public Buffered program() {
        return program;
    }

    /** Whether the buffers live on a GPU, so that {@link #resident} has something to give. */
    public abstract boolean onDevice();

    /** Registers every pass of {@code passes} ahead of its first {@link #run}. Any thread; see the class. */
    public abstract void prepare(List<Pass> passes);

    /** Sets every buffer to zero. */
    public abstract void clear();

    /**
     * Runs every pass of {@code passes}, in order. A list is recorded on its first run; pass the same list again.
     * Returns without waiting for the GPU; the {@link Completion} says when the passes have finished. On the CPU
     * they have finished when it returns.
     */
    public abstract Completion run(List<Pass> passes);

    /**
     * As {@link #run(List)}, starting once every timeline of {@code waits} has reached its value and setting every
     * timeline of {@code signals} to its value when done, on the GPU: so work on another queue can follow it. On the
     * CPU, where everything is done when this returns, the signals are made from the host and the waits are awaited
     * there first.
     */
    public abstract Completion run(List<Pass> passes, List<GpuContext.Point> waits, List<GpuContext.Point> signals);

    /** Overwrites the start of {@code buffer} with {@code words}. */
    public abstract void write(String buffer, int[] words);

    /** The whole of {@code buffer}, after everything run before. */
    public abstract int[] read(String buffer);

    /**
     * The whole of a {@linkplain dev.supirvast.vastir.pass.BufferSpec#readout readout} {@code buffer}, as the last
     * run whose {@link Completion} has been awaited left it: read where it is, with no submission and no wait.
     */
    public abstract int[] peek(String buffer);

    /** Waits for everything run so far. Nothing to wait for on the CPU. */
    public abstract void finish();

    /** {@link #write}, of floats by their bits. */
    public void write(String buffer, float[] values) {
        int[] words = new int[values.length];
        for (int k = 0; k < values.length; k++) {
            words[k] = Float.floatToRawIntBits(values[k]);
        }
        write(buffer, words);
    }

    /** {@link #read}, as floats. */
    public float[] floats(String buffer) {
        int[] words = read(buffer);
        float[] values = new float[words.length];
        for (int k = 0; k < words.length; k++) {
            values[k] = Float.intBitsToFloat(words[k]);
        }
        return values;
    }

    /**
     * The resident buffer behind a name: what a picture binds to read the state where the kernels wrote it.
     *
     * @throws IllegalStateException on a runner whose buffers are not {@link #onDevice on a device}
     */
    public abstract ResidentBuffer resident(String buffer);

    @Override
    public abstract void close();

    private void requireNamed(String buffer) {
        if (!program.buffers().containsKey(buffer)) {
            throw new IllegalArgumentException("the program has no buffer named " + buffer);
        }
    }

    static final class Gpu extends PassRunner {
        private final Accelerator accelerator;
        private final Map<String, ResidentBuffer> buffers = new LinkedHashMap<>();
        private final Map<Pass, KernelHandle> handles = Collections.synchronizedMap(new IdentityHashMap<>());
        /**
         * Every kernel registered, by its function and what it is bound to: passes that differ only in which buffers
         * of the same lengths they name share one pipeline, rather than lowering and compiling it again each.
         */
        private final Map<KernelKey, KernelHandle> compiled = Collections.synchronizedMap(new java.util.HashMap<>());
        private final Map<List<Pass>, DispatchSequence> sequences = new IdentityHashMap<>();

        Gpu(Accelerator accelerator, Buffered program, int workgroup, int subgroup) {
            super(program, workgroup, subgroup);
            this.accelerator = accelerator;
            program.buffers().forEach((name, spec) -> buffers.put(name, spec.readout()
                    ? accelerator.allocateReadout(spec.element(), spec.length())
                    : accelerator.allocate(spec.element(), spec.length())));
        }

        @Override
        public int[] peek(String buffer) {
            return resident(buffer).peek();
        }

        @Override
        public boolean onDevice() {
            return buffers.values().stream().allMatch(ResidentBuffer::onDevice);
        }

        @Override
        public void prepare(List<Pass> passes) {
            passes.forEach(this::handle);
        }

        @Override
        public void clear() {
            accelerator.clear(List.copyOf(buffers.values()));
        }

        @Override
        public Completion run(List<Pass> passes) {
            return run(passes, List.of(), List.of());
        }

        @Override
        public Completion run(List<Pass> passes, List<GpuContext.Point> waits, List<GpuContext.Point> signals) {
            return sequences.computeIfAbsent(passes, list -> {
                DispatchSequence.Builder builder = accelerator.sequence();
                for (Pass pass : list) {
                    builder.dispatch(handle(pass), pass.buffers().stream().map(buffers::get).toList(),
                            pass.invocations());
                }
                return builder.build();
            }).run(waits, signals);
        }

        private KernelHandle handle(Pass pass) {
            return handles.computeIfAbsent(pass, p -> compiled.computeIfAbsent(KernelKey.of(p, buffers), key -> {
                List<KernelColumn> columns = new ArrayList<>();
                for (int k = 0; k < p.bindings().size(); k++) {
                    var binding = p.bindings().get(k);
                    columns.add(KernelColumn.output(binding.name(), binding.binding(), binding.element())
                            .withLength(buffers.get(p.buffers().get(k)).elements()));
                }
                KernelSpec spec = new KernelSpec(p.kernel(), columns).withWorkgroupSize(workgroup);
                if (subgroup != NO_SUBGROUP) {
                    spec = spec.withSubgroupSize(subgroup);
                }
                return accelerator.register(spec).orElseThrow();
            }));
        }

        /** A kernel by its function's identity, its bindings, and the lengths of the buffers a pass binds to them. */
        private record KernelKey(Object kernel, List<?> bindings, List<Integer> lengths) {

            static KernelKey of(Pass pass, Map<String, ResidentBuffer> buffers) {
                return new KernelKey(pass.kernel(), pass.bindings(),
                        pass.buffers().stream().map(name -> buffers.get(name).elements()).toList());
            }

            @Override
            public boolean equals(Object other) {
                return other instanceof KernelKey key && key.kernel == kernel && key.bindings.equals(bindings)
                        && key.lengths.equals(lengths);
            }

            @Override
            public int hashCode() {
                return (System.identityHashCode(kernel) * 31 + bindings.hashCode()) * 31 + lengths.hashCode();
            }
        }

        @Override
        public void write(String buffer, int[] words) {
            resident(buffer).write(words);
        }

        @Override
        public int[] read(String buffer) {
            return resident(buffer).read();
        }

        @Override
        public void finish() {
            accelerator.finish();
        }

        @Override
        public ResidentBuffer resident(String buffer) {
            ResidentBuffer resident = buffers.get(buffer);
            if (resident == null) {
                throw new IllegalArgumentException("the program has no buffer named " + buffer);
            }
            return resident;
        }

        @Override
        public void close() {
            sequences.values().forEach(DispatchSequence::close);
            sequences.clear();
            compiled.values().forEach(accelerator::release);
            compiled.clear();
            handles.clear();
            buffers.values().forEach(ResidentBuffer::close);
        }
    }

    static final class Cpu extends PassRunner {
        private final Map<String, int[]> arrays = new LinkedHashMap<>();
        private final Map<Pass, CpuKernel> lowered = Collections.synchronizedMap(new IdentityHashMap<>());
        /** Every kernel lowered, by its function and bindings: passes that differ only in their buffers share one. */
        private final Map<CpuKey, CpuKernel> byKernel = Collections.synchronizedMap(new java.util.HashMap<>());

        /** A kernel by its function's identity and its bindings. */
        private record CpuKey(Object kernel, List<?> bindings) {

            @Override
            public boolean equals(Object other) {
                return other instanceof CpuKey key && key.kernel == kernel && key.bindings.equals(bindings);
            }

            @Override
            public int hashCode() {
                return System.identityHashCode(kernel) * 31 + bindings.hashCode();
            }
        }

        Cpu(Buffered program, int workgroup, int subgroup) {
            super(program, workgroup, subgroup);
            program.buffers().forEach((name, spec) -> arrays.put(name, new int[spec.length()]));
        }

        @Override
        public boolean onDevice() {
            return false;
        }

        @Override
        public void prepare(List<Pass> passes) {
            passes.forEach(this::kernel);
        }

        @Override
        public void clear() {
            arrays.values().forEach(array -> java.util.Arrays.fill(array, 0));
        }

        @Override
        public Completion run(List<Pass> passes) {
            for (Pass pass : passes) {
                kernel(pass).dispatch(pass.buffers().stream().map(arrays::get).toArray(int[][]::new),
                        pass.invocations());
            }
            return Completion.DONE;
        }

        @Override
        public Completion run(List<Pass> passes, List<GpuContext.Point> waits, List<GpuContext.Point> signals) {
            for (GpuContext.Point wait : waits) {
                wait.timeline().await(wait.value(), Long.MAX_VALUE);
            }
            Completion done = run(passes);
            for (GpuContext.Point signal : signals) {
                signal.timeline().signal(signal.value());
            }
            return done;
        }

        private CpuKernel kernel(Pass pass) {
            return lowered.computeIfAbsent(pass, p -> byKernel.computeIfAbsent(new CpuKey(p.kernel(), p.bindings()),
                    key -> subgroup == NO_SUBGROUP
                            ? new CoreToTruffle().lowerDispatch(p.kernel(), p.bindings(), workgroup)
                            : new CoreToTruffle().lowerDispatch(p.kernel(), p.bindings(), workgroup, subgroup)));
        }

        @Override
        public void write(String buffer, int[] words) {
            super.requireNamed(buffer);
            System.arraycopy(words, 0, arrays.get(buffer), 0, words.length);
        }

        @Override
        public int[] read(String buffer) {
            super.requireNamed(buffer);
            return arrays.get(buffer).clone();
        }

        @Override
        public int[] peek(String buffer) {
            return read(buffer);
        }

        @Override
        public void finish() {
        }

        @Override
        public ResidentBuffer resident(String buffer) {
            throw new IllegalStateException("a CPU runner's buffers are host arrays, not resident buffers");
        }

        @Override
        public void close() {
        }
    }
}
