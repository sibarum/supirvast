package dev.supirvast.vast;

import com.oracle.truffle.api.CallTarget;

/**
 * A kernel lowered for the CPU by {@link CoreToTruffle#lowerDispatch}, run a whole dispatch at a time.
 *
 * <p>How the invocations are scheduled depends on what the kernel uses, and is hidden here so a caller never
 * has to know:
 *
 * <ul>
 *   <li>A kernel with no workgroup features runs one invocation after another, exactly {@code n} of them.</li>
 *   <li>A kernel with workgroup memory or workgroup indices runs workgroup by workgroup, each with its own
 *       shared arrays. The last workgroup is short when {@code n} is not a multiple of the size, since the GPU
 *       stops the invocations past {@code n} before they touch anything.</li>
 *   <li>A kernel with a barrier runs whole workgroups — the GPU cannot stop the tail of one before a barrier
 *       every invocation must reach — and inside each, phase by phase: every invocation runs up to the next
 *       barrier before any runs past it.</li>
 * </ul>
 */
public final class CpuKernel {

    private final CallTarget target;
    private final boolean grouped;
    private final boolean wholeWorkgroups;
    private final int workgroupSize;

    CpuKernel(CallTarget target, boolean grouped, boolean wholeWorkgroups, int workgroupSize) {
        this.target = target;
        this.grouped = grouped;
        this.wholeWorkgroups = wholeWorkgroups;
        this.workgroupSize = workgroupSize;
    }

    /**
     * Runs {@code invocations} invocations over {@code buffers} (indexed by slot, as {@link
     * CoreToTruffle#lowerKernel} documents), updating them in place.
     */
    public void dispatch(int[][] buffers, int invocations) {
        if (!grouped) {
            for (int i = 0; i < invocations; i++) {
                target.call(i, buffers, invocations);
            }
            return;
        }
        int groups = (int) (((long) invocations + workgroupSize - 1) / workgroupSize);
        for (int g = 0; g < groups; g++) {
            int running = wholeWorkgroups ? workgroupSize : Math.min(workgroupSize, invocations - g * workgroupSize);
            target.call(g, buffers, invocations, running);
        }
    }

    /** Whether the kernel has a barrier, and so runs every invocation of its last workgroup. */
    public boolean runsWholeWorkgroups() {
        return wholeWorkgroups;
    }
}
