package dev.supirvast.vastir.tools;

/**
 * Whether work that was submitted without waiting has finished: what {@link DispatchSequence#run} and {@link
 * PassRunner#run} return. Ask {@link #done} each frame, or {@link #await} it.
 *
 * <p>A resident submission's completion covers everything submitted before it on the same context, since resident
 * work runs on one queue, in order. Work that ran on the CPU, or one dispatch at a time because it could not all run
 * on the GPU, was finished when it returned, and its completion is {@link #DONE}.
 *
 * <p>Owning-thread only, like the context whose work it tracks: asking may retire finished work and free what it
 * held.
 */
public final class Completion {

    /** Work that had finished by the time it returned. */
    public static final Completion DONE = new Completion(null, 0);

    private final GpuContext context;
    private final long serial;

    Completion(GpuContext context, long serial) {
        this.context = context;
        this.serial = serial;
    }

    /** Whether the work has finished, without waiting. */
    public boolean done() {
        return context == null || context.done(serial);
    }

    /** Blocks until the work has finished. */
    public void await() {
        if (context != null) {
            context.await(serial);
        }
    }

    /**
     * How long the work took on the GPU, in nanoseconds, from two timestamps written around it: once it is
     * {@link #done}, and where the device can write them. Empty for work that ran on the CPU, which has no such
     * clock, and for a run so long ago that the context has since reused its timestamps.
     */
    public java.util.OptionalLong gpuNanos() {
        return context == null ? java.util.OptionalLong.empty() : context.gpuNanos(serial);
    }
}
