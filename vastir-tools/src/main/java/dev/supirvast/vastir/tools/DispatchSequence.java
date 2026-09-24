package dev.supirvast.vastir.tools;

import java.util.ArrayList;
import java.util.List;

/**
 * Several resident dispatches, of one kernel or many, recorded once and run as one submission — for a
 * multi-pass step (a sort's count, scans and permute; a whole simulation step) that runs every frame.
 *
 * <p>Every {@link KernelHandle#dispatch} pays a fixed cost: a descriptor pool and set, a command buffer
 * recorded and a queue submission. Measured on an RTX 5070 Ti it is ~0.021 ms even for an empty kernel, which
 * is most of a short pass. A sequence pays the descriptor sets and the recording once, at {@link Builder#build},
 * and a {@link #run} is one submission of the recorded command buffer, however many dispatches it holds.
 *
 * <p>The steps run in order, each seeing everything the previous one wrote, exactly as the same dispatches
 * made one by one would. A run returns without waiting and is ordered after all resident work before it —
 * earlier runs included — and before all after, so a host {@link ResidentBuffer#read read} sees it.
 * Everything a step is fixed when it is added: its buffers and its invocation count.
 *
 * <p>Where the steps cannot all run on the GPU — no device, a kernel registered CPU-only, a buffer on the
 * host, a pipeline released since — a run makes the dispatches one at a time through {@link
 * KernelHandle#dispatch}, which runs each where it can. Slower, never wrong, and never a partial GPU run.
 *
 * <p>Owning-thread only, like everything that touches the {@link Accelerator}'s context. The buffers must
 * outlive the sequence, which must be {@link #close closed} (the accelerator closes any left open).
 */
public final class DispatchSequence implements AutoCloseable {

    /** One dispatch: a kernel over resident buffers, one per column in binding order, and a count. */
    record Step(KernelHandle handle, List<ResidentBuffer> buffers, int invocations) {}

    private final Accelerator accelerator;
    private final List<Step> steps;
    private final GpuContext.RecordedSequence recorded;   // null when the sequence runs step by step
    private boolean closed;

    DispatchSequence(Accelerator accelerator, List<Step> steps, GpuContext.RecordedSequence recorded) {
        this.accelerator = accelerator;
        this.steps = List.copyOf(steps);
        this.recorded = recorded;
    }

    /** Collects the steps of a sequence; {@link Accelerator#sequence} starts one. */
    public static final class Builder {
        private final Accelerator accelerator;
        private final List<Step> steps = new ArrayList<>();

        Builder(Accelerator accelerator) {
            this.accelerator = accelerator;
        }

        /**
         * Adds a dispatch of {@code handle} over {@code n} invocations against {@code buffers}, checked now as
         * {@link KernelHandle#dispatch} would check it.
         *
         * @throws IllegalArgumentException if the buffers do not fit the kernel's columns
         */
        public Builder dispatch(KernelHandle handle, List<ResidentBuffer> buffers, int n) {
            handle.validateResident(buffers, n);
            steps.add(new Step(handle, List.copyOf(buffers), n));
            return this;
        }

        /** Records the steps — into one command buffer where they can all run on the GPU. */
        public DispatchSequence build() {
            if (steps.isEmpty()) {
                throw new IllegalStateException("a sequence needs at least one dispatch");
            }
            return accelerator.buildSequence(steps);
        }
    }

    /** Runs every step, in order: one submission on the GPU, the dispatches one by one otherwise. */
    public void run() {
        if (closed) {
            throw new IllegalStateException("the sequence is closed");
        }
        for (Step step : steps) {
            for (ResidentBuffer buffer : step.buffers()) {
                if (buffer.isClosed()) {
                    throw new IllegalStateException("a buffer of this sequence has been closed");
                }
            }
        }
        if (recorded()) {
            accelerator.submitSequence(recorded);
            return;
        }
        for (Step step : steps) {
            step.handle().dispatch(step.buffers(), step.invocations());
        }
    }

    /**
     * Whether a run is one GPU submission, rather than the steps dispatched one at a time: recorded at build,
     * and every kernel's pipeline still there — a {@link KernelHandle#close closed} handle's is not.
     */
    public boolean recorded() {
        return recorded != null && !closed
                && steps.stream().allMatch(s -> s.handle().preferredBackend() == KernelHandle.Backend.GPU);
    }

    /** How many dispatches a run makes. */
    public int dispatches() {
        return steps.size();
    }

    /** Waits for any run in flight and frees the recorded command buffer. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        accelerator.forget(this);
        if (recorded != null) {
            recorded.close();
        }
    }
}
