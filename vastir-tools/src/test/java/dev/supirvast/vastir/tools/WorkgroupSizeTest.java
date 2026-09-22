package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A workgroup wider than one changes how a dispatch is scheduled and nothing about what it computes.
 *
 * <p>The part with teeth is the tail. A dispatch of {@code n} rounds up to whole workgroups, so the invocations
 * past {@code n} exist on the GPU and must do nothing — in particular they must not write, because the buffers
 * are sized for {@code n} and a write past the end is undefined behaviour that usually lands somewhere
 * harmless and so goes unnoticed. The test therefore counts executed invocations with an atomic, which the
 * tail cannot hide from, rather than only checking the outputs it is supposed to leave alone.
 */
class WorkgroupSizeTest {

    private static final Type.Int I32 = Type.int32();
    private static final Type.Float F32 = Type.float32();

    /** Deliberately not a multiple of any workgroup size tried. */
    private static final int N = 1000;

    /** {@code out[gid] = gid * 3; counter[0] += 1} */
    private static KernelSpec tripleAndCount() {
        Buffer out = new Buffer("out", 0, I32);
        Buffer counter = new Buffer("counter", 1, I32);
        Expr gid = new Expr.InvocationId();
        Function kernel = new Function("main", new Type.FunctionType(Type.VOID, List.of()), Region.of(
                new Statement.BufferStore(out, gid, new Expr.Binary(BinaryOp.MUL, gid, new Expr.ConstInt(I32, 3))),
                new Statement.AtomicUpdate(AtomicOp.ADD, counter, new Expr.ConstInt(I32, 0), new Expr.ConstInt(I32, 1)),
                new Statement.ReturnVoid()));
        return new KernelSpec(kernel, List.of(
                KernelColumn.output("out", 0, I32),
                KernelColumn.output("counter", 1, I32).withLength(1)));
    }

    @Test
    void everyWorkgroupSizeComputesTheSameThingAndStopsTheTail() {
        int[] expected = new int[N];
        for (int k = 0; k < N; k++) {
            expected[k] = k * 3;
        }
        try (Accelerator accelerator = new Accelerator()) {
            assumeGpu(accelerator);
            for (int size : new int[] {1, 7, 32, 64, 256}) {
                KernelHandle handle = accelerator.register(tripleAndCount().withWorkgroupSize(size)).orElseThrow();
                assertEquals(KernelHandle.Backend.GPU, handle.preferredBackend(), "size " + size);
                int[][] result = handle.run(new int[][] {new int[N], new int[1]}, N);
                assertArrayEquals(expected, result[0], "size " + size + ": outputs");
                assertEquals(N, result[1][0], "size " + size + ": " + result[1][0] + " invocations ran, not " + N
                        + " -- the tail of the last workgroup was not stopped");
            }
        }
    }

    @Test
    void theDefaultIsSixtyFour() {
        assertEquals(64, tripleAndCount().workgroupSize());
    }

    @Test
    void aWorkgroupOfZeroIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> tripleAndCount().withWorkgroupSize(0));
    }

    /**
     * Not an assertion: what the width buys, on this machine. A kernel heavy enough per invocation that the
     * GPU's own time shows through the upload and readback each run pays.
     */
    @Test
    void throughputByWorkgroupSize() {
        throughput(512);
        throughput(8192);
    }

    private static void throughput(int iterations) {
        int n = 1 << 20;
        Buffer out = new Buffer("out", 0, F32);
        LocalVar x = new LocalVar("x", F32);
        LocalVar i = new LocalVar("i", I32);
        Function kernel = new Function("main", new Type.FunctionType(Type.VOID, List.of()), Region.of(
                new Statement.DeclareVar(x, new Expr.Convert(new Expr.InvocationId(), F32)),
                new Statement.DeclareVar(i, new Expr.ConstInt(I32, 0)),
                new Statement.While(new Expr.Binary(BinaryOp.LESS_THAN, new Expr.Read(i), new Expr.ConstInt(I32, iterations)),
                        Region.of(
                                new Statement.Assign(x, new Expr.Binary(BinaryOp.ADD,
                                        new Expr.Binary(BinaryOp.MUL, new Expr.Read(x), new Expr.ConstFloat(F32, 0.999)),
                                        new Expr.ConstFloat(F32, 0.5))),
                                new Statement.Assign(i, new Expr.Binary(BinaryOp.ADD, new Expr.Read(i),
                                        new Expr.ConstInt(I32, 1))))),
                new Statement.BufferStore(out, new Expr.InvocationId(), new Expr.Read(x)),
                new Statement.ReturnVoid()));
        KernelSpec spec = new KernelSpec(kernel, List.of(KernelColumn.output("out", 0, F32)));

        try (Accelerator accelerator = new Accelerator()) {
            assumeGpu(accelerator);
            StringBuilder report = new StringBuilder("[workgroups] 2^20 invocations x " + iterations + " multiply-adds:");
            for (int size : new int[] {1, 32, 64, 256}) {
                KernelHandle handle = accelerator.register(spec.withWorkgroupSize(size)).orElseThrow();
                int[][] columns = {new int[n]};
                handle.run(columns, n);   // warm: first dispatch pays for lazy driver work
                int runs = 5;
                long start = System.nanoTime();
                for (int r = 0; r < runs; r++) {
                    handle.run(columns, n);
                }
                report.append(String.format("  size %d: %.2f ms", size, (System.nanoTime() - start) / 1e6 / runs));
                handle.close();
            }
            System.out.println(report);
        }
    }

    private static void assumeGpu(Accelerator accelerator) {
        assumeTrue(accelerator.capabilities().gpuAvailable() || Boolean.getBoolean("supirvast.requireGpu"),
                "no Vulkan device");
    }
}
