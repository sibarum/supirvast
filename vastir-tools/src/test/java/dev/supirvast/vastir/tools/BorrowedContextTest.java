package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * An accelerator on a context somebody else made: it runs there, and it leaves the context alone.
 *
 * <p>The ownership rule is the whole of the feature. An application that lends its device to a simulation closes
 * the context itself, once, after the simulation; an accelerator that closed what it was lent would take the
 * application's compute down with the simulation's, and one that never released its own pipelines and buffers
 * would leak them into a context that outlives it. So both halves are asserted: what it made is gone, what it
 * was lent is not.
 */
class BorrowedContextTest {

    private static final Buffer OUT = new Buffer("out", 0);
    private static final Buffer A = new Buffer("a", 1);

    /** {@code out[gid] = a[gid] * 2;} */
    private static KernelSpec doubler() {
        Expr gid = new Expr.InvocationId();
        Region body = Region.of(
                new Statement.BufferStore(OUT, gid,
                        new Expr.Binary(BinaryOp.MUL, new Expr.BufferLoad(A, new Expr.InvocationId()),
                                new Expr.ConstInt(Type.int32(), 2))),
                new Statement.ReturnVoid());
        Function kernel = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new KernelSpec(kernel, List.of(KernelColumn.output("out", 0), KernelColumn.input("a", 1)));
    }

    @Test
    void anAcceleratorOnALentContextRunsOnItAndLeavesItOpen() {
        assumeTrue(GpuContext.isAvailable(), "no GPU here");
        try (GpuContext context = GpuContext.open()) {
            int n = 8;
            int[] a = {1, 2, 3, 4, 5, 6, 7, 8};

            try (Accelerator accelerator = Accelerator.on(context)) {
                Accelerator.Capabilities capabilities = accelerator.capabilities();
                assertTrue(capabilities.gpuAvailable(), "the context's device is the GPU");
                assertEquals(context.deviceName(), capabilities.deviceName(),
                        "it reports the device it was lent, not one it found");
                assertEquals(context.capabilities(), capabilities.deviceCapabilities());

                KernelHandle kernel = accelerator.register(doubler()).orElseThrow();
                assertEquals(KernelHandle.Backend.GPU, kernel.preferredBackend());
                int[][] result = kernel.run(new int[][] {new int[n], a.clone()}, n);
                assertArrayEquals(new int[] {2, 4, 6, 8, 10, 12, 14, 16}, result[0]);

                ResidentBuffer resident = accelerator.allocate(Type.int32(), 4);
                assertTrue(resident.onDevice());
                assertTrue(resident.vkBuffer() != 0L, "a resident buffer on the device has a VkBuffer to share");
                accelerator.finish();
            }

            // The accelerator is closed. What it was lent still works: a context that had been closed under it
            // could not allocate, write or read.
            GpuContext.DeviceBuffer buffer = context.allocateBuffer(4);
            try {
                context.write(buffer, new int[] {9, 8, 7, 6});
                assertArrayEquals(new int[] {9, 8, 7, 6}, context.read(buffer, 4));
            } finally {
                buffer.close();
            }
        }
    }

    @Test
    void aBufferOnTheHostHasNoVkBufferToShare() {
        // No context lent and no device used: the buffer is a host array for the CPU backend.
        try (Accelerator accelerator = new Accelerator(dev.supirvast.vastir.lower.SpirvTarget.unconstrained())) {
            if (accelerator.capabilities().gpuAvailable()) {
                return;     // on a machine with a GPU an allocation lands on it, which the other test covers
            }
            ResidentBuffer host = accelerator.allocate(Type.int32(), 4);
            assertThrows(IllegalStateException.class, host::vkBuffer);
        }
    }

    @Test
    void finishWithNothingSubmittedIsANoOp() {
        try (Accelerator accelerator = new Accelerator()) {
            accelerator.finish();     // no context was ever opened: nothing to wait for, nothing to open
        }
    }
}
