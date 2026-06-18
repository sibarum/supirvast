package dev.supirvast.vastir.tools;

/**
 * The outcome of {@link Accelerator#register}: either a usable {@link KernelHandle} (the kernel validated,
 * lowered, and is runnable) or a {@link Rejection} carrying the witness for why it could not be. Lowering
 * <em>is</em> the proof of lowerability — a handle exists only when every gate passed, so there is no separate
 * "is it eligible?" query and no way to run an unregistered kernel.
 */
public sealed interface Registration permits KernelHandle, Rejection {

    /** Whether registration succeeded (a {@link KernelHandle} is available). */
    default boolean succeeded() {
        return this instanceof KernelHandle;
    }

    /**
     * Returns the handle, or throws with the rejection's witness if registration failed — for call sites that
     * treat a rejection as a programming error rather than handling it.
     */
    default KernelHandle orElseThrow() {
        if (this instanceof KernelHandle handle) {
            return handle;
        }
        Rejection rejection = (Rejection) this;
        throw new IllegalStateException(
                "kernel rejected (" + rejection.reason() + "): " + rejection.detail());
    }
}
