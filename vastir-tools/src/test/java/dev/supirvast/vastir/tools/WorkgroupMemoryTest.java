package dev.supirvast.vastir.tools;

import dev.supirvast.vast.CoreToTruffle;
import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.SharedArray;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.lower.SpirvTarget;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Workgroup memory and barriers, on both backends, each checked against the known answer on its own.
 *
 * <p>Every kernel here reads a slot of workgroup memory some <em>other</em> invocation wrote, because that is
 * the only thing a barrier is for: a kernel where each invocation reads back its own slot passes without one.
 * On the CPU a missing phase split shows as zeros, since the earlier invocations read before the later ones
 * write; on the GPU as a race. And {@code n} is never a multiple of the workgroup, so the last workgroup's
 * tail — which a barrier kernel runs rather than stops — is in every run.
 */
class WorkgroupMemoryTest {

    private static final Type.Int I32 = Type.int32();
    private static final Type.Float F32 = Type.float32();
    private static final int N = 1000;

    private static int input(int k) {
        return k * 7 + 3;
    }

    private static int[] inputs(int n) {
        int[] in = new int[n];
        for (int k = 0; k < n; k++) {
            in[k] = input(k);
        }
        return in;
    }

    // --- the kernels ---------------------------------------------------------------------------------------

    /**
     * Each workgroup reverses its slice: {@code out[gid] = in[group's mirror of gid]}, through a tile of
     * workgroup memory. The tail writes {@code -1} to its slots so the mirror of a tail slot is defined.
     */
    private static Function reverseWithinWorkgroup(int size) {
        Buffer in = new Buffer("in", 0, I32);
        Buffer out = new Buffer("out", 1, I32);
        SharedArray tile = new SharedArray("tile", I32, size);
        Expr mirror = sub(i(size - 1), lid());
        return kernel(
                new Statement.If(lt(gid(), count()),
                        Region.of(new Statement.SharedStore(tile, lid(), new Expr.BufferLoad(in, gid()))),
                        Region.of(new Statement.SharedStore(tile, lid(), i(-1)))),
                new Statement.Barrier(),
                new Statement.If(lt(gid(), count()),
                        Region.of(new Statement.BufferStore(out, gid(), new Expr.SharedLoad(tile, mirror))),
                        Region.of()));
    }

    /** Tree reduction: {@code sums[workgroup] = in[..]} summed over the workgroup, a barrier every round. */
    private static Function sumPerWorkgroup(int size) {
        Buffer in = new Buffer("in", 0, I32);
        Buffer sums = new Buffer("sums", 1, I32);
        SharedArray partial = new SharedArray("partial", I32, size);
        LocalVar stride = new LocalVar("stride", I32);
        return kernel(
                new Statement.If(lt(gid(), count()),
                        Region.of(new Statement.SharedStore(partial, lid(), new Expr.BufferLoad(in, gid()))),
                        Region.of(new Statement.SharedStore(partial, lid(), i(0)))),
                new Statement.Barrier(),
                new Statement.DeclareVar(stride, i(size / 2)),
                new Statement.While(lt(i(0), new Expr.Read(stride)), Region.of(
                        new Statement.If(lt(lid(), new Expr.Read(stride)),
                                Region.of(new Statement.SharedStore(partial, lid(), add(
                                        new Expr.SharedLoad(partial, lid()),
                                        new Expr.SharedLoad(partial, add(lid(), new Expr.Read(stride)))))),
                                Region.of()),
                        new Statement.Barrier(),
                        new Statement.Assign(stride, new Expr.Binary(BinaryOp.SHIFT_RIGHT, new Expr.Read(stride), i(1))))),
                new Statement.If(eq(lid(), i(0)),
                        Region.of(new Statement.BufferStore(sums, new Expr.WorkgroupId(), new Expr.SharedLoad(partial, i(0)))),
                        Region.of()));
    }

    /**
     * A histogram of {@code in[gid] & 15} counted in workgroup memory, then flushed with one global atomic per
     * bin per workgroup — the shape that trades a global atomic per invocation for one per workgroup.
     */
    private static Function histogramInWorkgroupMemory() {
        Buffer in = new Buffer("in", 0, I32);
        Buffer bins = new Buffer("bins", 1, I32);
        SharedArray local = new SharedArray("local", I32, 16);
        return kernel(
                new Statement.If(lt(lid(), i(16)), Region.of(new Statement.SharedStore(local, lid(), i(0))), Region.of()),
                new Statement.Barrier(),
                new Statement.If(lt(gid(), count()), Region.of(new Statement.SharedAtomicUpdate(AtomicOp.ADD, local,
                        new Expr.Binary(BinaryOp.BIT_AND, new Expr.BufferLoad(in, gid()), i(15)), i(1))), Region.of()),
                new Statement.Barrier(),
                new Statement.If(lt(lid(), i(16)), Region.of(new Statement.AtomicUpdate(AtomicOp.ADD, bins, lid(),
                        new Expr.SharedLoad(local, lid()))), Region.of()));
    }

    /**
     * The pre-reducing scatter: every invocation adds {@code in[gid] & 7} into bin {@code in[gid] & 15} of an
     * f32 array in workgroup memory, and the workgroup flushes each bin with one f32 atomic on the buffer. The
     * values are small integers, which f32 adds exactly in any order, so the answer is order-independent.
     */
    private static Function floatScatterInWorkgroupMemory() {
        Buffer in = new Buffer("in", 0, I32);
        Buffer sums = new Buffer("sums", 1, F32);
        SharedArray local = new SharedArray("local", F32, 16);
        Expr value = new Expr.Convert(new Expr.Binary(BinaryOp.BIT_AND, new Expr.BufferLoad(in, gid()), i(7)), F32);
        return kernel(
                new Statement.If(lt(lid(), i(16)), Region.of(new Statement.SharedStore(local, lid(),
                        new Expr.ConstFloat(F32, 0))), Region.of()),
                new Statement.Barrier(),
                new Statement.If(lt(gid(), count()), Region.of(new Statement.SharedAtomicUpdate(AtomicOp.ADD, local,
                        new Expr.Binary(BinaryOp.BIT_AND, new Expr.BufferLoad(in, gid()), i(15)), value)), Region.of()),
                new Statement.Barrier(),
                new Statement.If(lt(lid(), i(16)), Region.of(new Statement.AtomicUpdate(AtomicOp.ADD, sums, lid(),
                        new Expr.SharedLoad(local, lid()))), Region.of()));
    }

    /** {@code maxima[workgroup]} = the largest {@code f32(in[gid])} in the workgroup, by a float atomic max. */
    private static Function floatMaxPerWorkgroup() {
        Buffer in = new Buffer("in", 0, I32);
        Buffer maxima = new Buffer("maxima", 1, F32);
        SharedArray best = new SharedArray("best", F32, 1);
        return kernel(
                new Statement.If(eq(lid(), i(0)), Region.of(new Statement.SharedStore(best, i(0),
                        new Expr.ConstFloat(F32, -1))), Region.of()),
                new Statement.Barrier(),
                new Statement.If(lt(gid(), count()), Region.of(new Statement.SharedAtomicUpdate(AtomicOp.MAX, best,
                        i(0), new Expr.Convert(new Expr.BufferLoad(in, gid()), F32))), Region.of()),
                new Statement.Barrier(),
                new Statement.If(eq(lid(), i(0)), Region.of(new Statement.BufferStore(maxima, new Expr.WorkgroupId(),
                        new Expr.SharedLoad(best, i(0)))), Region.of()));
    }

    /** Exactly one invocation per workgroup wins a compare-exchange on a workgroup flag, and counts itself. */
    private static Function oneWinnerPerWorkgroup() {
        Buffer winners = new Buffer("winners", 0, I32);
        SharedArray flag = new SharedArray("flag", I32, 1);
        LocalVar old = new LocalVar("old", I32);
        return kernel(
                new Statement.If(eq(lid(), i(0)), Region.of(new Statement.SharedStore(flag, i(0), i(0))), Region.of()),
                new Statement.Barrier(),
                new Statement.SharedAtomicCompareExchange(old, flag, i(0), i(0), add(lid(), i(1))),
                new Statement.If(eq(new Expr.Read(old), i(0)),
                        Region.of(new Statement.AtomicUpdate(AtomicOp.ADD, winners, i(0), i(1))), Region.of()));
    }

    /** No barrier and no workgroup memory: {@code out[gid] = workgroup * 1000 + local}. Its tail is stopped. */
    private static Function indices() {
        Buffer out = new Buffer("out", 0, I32);
        return kernel(new Statement.BufferStore(out, gid(),
                add(new Expr.Binary(BinaryOp.MUL, new Expr.WorkgroupId(), i(1000)), lid())));
    }

    // --- the answers ---------------------------------------------------------------------------------------

    private static int[] expectedReverse(int size) {
        int[] out = new int[N];
        for (int g = 0; g < N; g++) {
            int mirror = g / size * size + (size - 1 - g % size);
            out[g] = mirror < N ? input(mirror) : -1;
        }
        return out;
    }

    private static int[] expectedSums(int size) {
        int[] sums = new int[groups(size)];
        for (int k = 0; k < N; k++) {
            sums[k / size] += input(k);
        }
        return sums;
    }

    private static int groups(int size) {
        return (N + size - 1) / size;
    }

    // --- the tests ----------------------------------------------------------------------------------------

    @Test
    void aWorkgroupReversesItsSliceThroughWorkgroupMemory() {
        for (int size : new int[] {7, 32, 64, 256}) {
            List<KernelColumn> columns = List.of(KernelColumn.input("in", 0, I32), KernelColumn.output("out", 1, I32));
            for (int[][] result : runEach(reverseWithinWorkgroup(size), columns, size,
                    new int[][] {inputs(N), new int[N]})) {
                assertArrayEquals(expectedReverse(size), result[1], "size " + size);
            }
        }
    }

    @Test
    void aTreeReductionSumsEachWorkgroup() {
        for (int size : new int[] {32, 64, 256}) {
            List<KernelColumn> columns = List.of(KernelColumn.input("in", 0, I32),
                    KernelColumn.output("sums", 1, I32).withLength(groups(size)));
            for (int[][] result : runEach(sumPerWorkgroup(size), columns, size,
                    new int[][] {inputs(N), new int[groups(size)]})) {
                assertArrayEquals(expectedSums(size), result[1], "size " + size);
            }
        }
    }

    @Test
    void atomicsOnWorkgroupMemoryCountEveryIncrement() {
        int[] expected = new int[16];
        for (int k = 0; k < N; k++) {
            expected[input(k) & 15]++;
        }
        List<KernelColumn> columns = List.of(KernelColumn.input("in", 0, I32),
                KernelColumn.output("bins", 1, I32).withLength(16));
        for (int[][] result : runEach(histogramInWorkgroupMemory(), columns, 64,
                new int[][] {inputs(N), new int[16]})) {
            assertArrayEquals(expected, result[1]);
        }
    }

    @Test
    void floatAddsPreReducedInWorkgroupMemoryFlushOncePerBin() {
        float[] expected = new float[16];
        for (int k = 0; k < N; k++) {
            expected[input(k) & 15] += input(k) & 7;
        }
        List<KernelColumn> columns = List.of(KernelColumn.input("in", 0, I32),
                KernelColumn.output("sums", 1, F32).withLength(16));
        for (int[][] result : runEach(floatScatterInWorkgroupMemory(), columns, 64,
                new int[][] {inputs(N), new int[16]}, true)) {
            assertArrayEquals(expected, floats(result[1]));
        }
    }

    @Test
    void aFloatMaxOnWorkgroupMemoryFindsEachWorkgroupsLargest() {
        int size = 64;
        float[] expected = new float[groups(size)];
        for (int g = 0; g < groups(size); g++) {
            expected[g] = input(Math.min(N, (g + 1) * size) - 1);   // input grows with k
        }
        List<KernelColumn> columns = List.of(KernelColumn.input("in", 0, I32),
                KernelColumn.output("maxima", 1, F32).withLength(groups(size)));
        for (int[][] result : runEach(floatMaxPerWorkgroup(), columns, size,
                new int[][] {inputs(N), new int[groups(size)]}, true)) {
            assertArrayEquals(expected, floats(result[1]));
        }
    }

    @Test
    void aCompareExchangeOnWorkgroupMemoryHasOneWinnerPerWorkgroup() {
        List<KernelColumn> columns = List.of(KernelColumn.output("winners", 0, I32).withLength(1));
        for (int[][] result : runEach(oneWinnerPerWorkgroup(), columns, 64, new int[][] {new int[1]})) {
            assertEquals(groups(64), result[0][0]);
        }
    }

    @Test
    void theWorkgroupIndicesAgreeAndTheTailIsStopped() {
        int size = 48;
        int[] expected = new int[N];
        for (int g = 0; g < N; g++) {
            expected[g] = g / size * 1000 + g % size;
        }
        for (int[][] result : runEach(indices(), List.of(KernelColumn.output("out", 0, I32)), size,
                new int[][] {new int[N]})) {
            assertArrayEquals(expected, result[0]);
        }
    }

    /** A barrier some invocations may skip is refused at registration, before either backend builds anything. */
    @Test
    void aBarrierInDivergentControlFlowIsRejected() {
        Function kernel = kernel(new Statement.If(lt(lid(), i(8)), Region.of(new Statement.Barrier()), Region.of()));
        try (Accelerator accelerator = new Accelerator()) {
            Registration registration = accelerator.register(
                    new KernelSpec(kernel, List.of(KernelColumn.output("out", 0, I32))));
            Rejection rejection = assertInstanceOf(Rejection.class, registration);
            assertTrue(rejection.detail().contains("not in uniform control flow"), rejection.detail());
        }
    }

    /** Over the caller's budget is a refusal, exactly as a capability outside it is. */
    @Test
    void workgroupMemoryOverTheBudgetIsRejected() {
        try (Accelerator accelerator = new Accelerator(SpirvTarget.unconstrained().withWorkgroupMemoryLimit(1024))) {
            Registration registration = accelerator.register(new KernelSpec(reverseWithinWorkgroup(512),
                    List.of(KernelColumn.input("in", 0, I32), KernelColumn.output("out", 1, I32)), 512));
            Rejection rejection = assertInstanceOf(Rejection.class, registration);
            assertTrue(rejection.detail().contains("2048 bytes of workgroup memory"), rejection.detail());
        }
    }

    /** Over the device's limit but within the budget is not a defect: the kernel runs on the CPU instead. */
    @Test
    void workgroupMemoryOverTheDeviceLimitRunsOnTheCpu() {
        try (Accelerator accelerator = new Accelerator()) {
            assumeGpu(accelerator);
            long limit = accelerator.capabilities().maxWorkgroupMemoryBytes();
            assertTrue(limit >= 16384, "Vulkan guarantees 16 KB of workgroup memory, the device reports " + limit);
            Buffer out = new Buffer("out", 0, I32);
            SharedArray huge = new SharedArray("huge", I32, (int) (limit / 4) + 1);
            Function kernel = kernel(
                    new Statement.SharedStore(huge, lid(), gid()),
                    new Statement.Barrier(),
                    new Statement.If(lt(gid(), count()), Region.of(new Statement.BufferStore(out, gid(),
                            new Expr.SharedLoad(huge, sub(i(63), lid())))), Region.of()));
            KernelHandle handle = accelerator.register(
                    new KernelSpec(kernel, List.of(KernelColumn.output("out", 0, I32)))).orElseThrow();
            assertEquals(KernelHandle.Backend.CPU, handle.preferredBackend());
            int[] result = handle.run(new int[][] {new int[N]}, N)[0];
            for (int g = 0; g < N; g++) {
                assertEquals(g / 64 * 64 + 63 - g % 64, result[g], "invocation " + g);
            }
        }
    }

    /** A barrier kernel dispatched against resident buffers, repeatedly, reads back the same answer. */
    @Test
    void aBarrierKernelDispatchesResident() {
        int size = 64;
        try (Accelerator accelerator = new Accelerator()) {
            KernelHandle handle = accelerator.register(new KernelSpec(sumPerWorkgroup(size), List.of(
                    KernelColumn.input("in", 0, I32),
                    KernelColumn.output("sums", 1, I32).withLength(groups(size))))).orElseThrow();
            ResidentBuffer in = accelerator.allocate(I32, N);
            ResidentBuffer sums = accelerator.allocate(I32, groups(size));
            in.write(inputs(N));
            for (int step = 0; step < 3; step++) {
                handle.dispatch(List.of(in, sums), N);
            }
            assertArrayEquals(expectedSums(size), sums.read());
        }
    }

    /**
     * Not an assertion: what workgroup memory buys, on this machine. Summing 2^20 integers into one counter,
     * by one global atomic per invocation against a tree reduction and one global atomic per workgroup.
     */
    @Test
    void throughputOfAReduction() {
        int n = 1 << 20;
        int size = 256;
        Buffer in = new Buffer("in", 0, I32);
        Buffer total = new Buffer("total", 1, I32);
        Function atomics = kernel(new Statement.AtomicUpdate(AtomicOp.ADD, total, i(0), new Expr.BufferLoad(in, gid())));
        SharedArray partial = new SharedArray("partial", I32, size);
        LocalVar stride = new LocalVar("stride", I32);
        Function reduction = kernel(
                new Statement.If(lt(gid(), count()),
                        Region.of(new Statement.SharedStore(partial, lid(), new Expr.BufferLoad(in, gid()))),
                        Region.of(new Statement.SharedStore(partial, lid(), i(0)))),
                new Statement.Barrier(),
                new Statement.DeclareVar(stride, i(size / 2)),
                new Statement.While(lt(i(0), new Expr.Read(stride)), Region.of(
                        new Statement.If(lt(lid(), new Expr.Read(stride)),
                                Region.of(new Statement.SharedStore(partial, lid(), add(
                                        new Expr.SharedLoad(partial, lid()),
                                        new Expr.SharedLoad(partial, add(lid(), new Expr.Read(stride)))))),
                                Region.of()),
                        new Statement.Barrier(),
                        new Statement.Assign(stride, new Expr.Binary(BinaryOp.SHIFT_RIGHT, new Expr.Read(stride), i(1))))),
                new Statement.If(eq(lid(), i(0)), Region.of(
                        new Statement.AtomicUpdate(AtomicOp.ADD, total, i(0), new Expr.SharedLoad(partial, i(0)))),
                        Region.of()));
        List<KernelColumn> columns = List.of(KernelColumn.input("in", 0, I32),
                KernelColumn.output("total", 1, I32).withLength(1));
        int[] data = new int[n];
        long expected = 0;
        for (int k = 0; k < n; k++) {
            data[k] = k & 1023;
            expected += data[k];
        }

        try (Accelerator accelerator = new Accelerator()) {
            assumeGpu(accelerator);
            ResidentBuffer input = accelerator.allocate(I32, n);
            input.write(data);
            StringBuilder report = new StringBuilder("[workgroup memory] sum of 2^20 i32:");
            for (Object[] variant : new Object[][] {{"global atomics", atomics}, {"tree reduction", reduction}}) {
                KernelHandle handle = accelerator.register(
                        new KernelSpec((Function) variant[1], columns, size)).orElseThrow();
                ResidentBuffer sum = accelerator.allocate(I32, 1);
                sum.write(new int[1]);
                handle.dispatch(List.of(input, sum), n);
                assertEquals(expected, sum.read()[0], variant[0] + " computed the wrong sum");
                int steps = 20;
                long start = System.nanoTime();
                for (int s = 0; s < steps; s++) {
                    handle.dispatch(List.of(input, sum), n);
                }
                sum.read();
                report.append(String.format("  %s: %.3f ms", variant[0], (System.nanoTime() - start) / 1e6 / steps));
                handle.close();
            }
            System.out.println(report);
        }
    }

    // --- harness ------------------------------------------------------------------------------------------

    /**
     * The kernel run on the CPU, then on the GPU when one is present and takes it — two results for the
     * caller to check against the answer, rather than only against each other.
     */
    private static List<int[][]> runEach(Function kernel, List<KernelColumn> columns, int size, int[][] data) {
        return runEach(kernel, columns, size, data, false);
    }

    /**
     * @param optionalFeatures whether the kernel needs a device feature a GPU may lack — only then is a GPU
     *                         that registers it CPU-only an answer rather than a failure
     */
    private static List<int[][]> runEach(Function kernel, List<KernelColumn> columns, int size, int[][] data,
            boolean optionalFeatures) {
        List<int[][]> results = new ArrayList<>();
        List<Buffer> buffers = columns.stream().map(c -> new Buffer(c.name(), c.binding(), c.type())).toList();
        int[][] cpu = copy(data);
        new CoreToTruffle().lowerDispatch(kernel, buffers, size).dispatch(cpu, N);
        results.add(cpu);

        try (Accelerator accelerator = new Accelerator()) {
            KernelHandle handle = accelerator.register(new KernelSpec(kernel, columns, size)).orElseThrow();
            Accelerator.Capabilities host = accelerator.capabilities();
            if (handle.preferredBackend() == KernelHandle.Backend.GPU) {
                results.add(handle.run(copy(data), N));
            } else if (!host.gpuAvailable()) {
                assertFalse(Boolean.getBoolean("supirvast.requireGpu"), "no Vulkan device, and one is required");
                System.out.println("[workgroup memory] GPU leg not run: no Vulkan device");
            } else {
                assertTrue(optionalFeatures, "registered CPU-only on a device that should run it");
                System.out.println("[workgroup memory] GPU leg not run: the device has " + host.deviceFeatures()
                        + " of the float-atomic features");
            }
        }
        return results;
    }

    private static float[] floats(int[] words) {
        float[] out = new float[words.length];
        for (int k = 0; k < words.length; k++) {
            out[k] = Float.intBitsToFloat(words[k]);
        }
        return out;
    }

    private static void assumeGpu(Accelerator accelerator) {
        assumeTrue(accelerator.capabilities().gpuAvailable() || Boolean.getBoolean("supirvast.requireGpu"),
                "no Vulkan device");
    }

    private static int[][] copy(int[][] data) {
        int[][] out = new int[data.length][];
        for (int k = 0; k < data.length; k++) {
            out[k] = data[k].clone();
        }
        return out;
    }

    private static Function kernel(Statement... statements) {
        List<Statement> body = new ArrayList<>(List.of(statements));
        body.add(new Statement.ReturnVoid());
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), new Region(body));
    }

    private static Expr gid() {
        return new Expr.InvocationId();
    }

    private static Expr lid() {
        return new Expr.LocalInvocationId();
    }

    private static Expr count() {
        return new Expr.InvocationCount();
    }

    private static Expr i(long value) {
        return new Expr.ConstInt(I32, value);
    }

    private static Expr add(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.ADD, a, b);
    }

    private static Expr sub(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.SUB, a, b);
    }

    private static Expr lt(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.LESS_THAN, a, b);
    }

    private static Expr eq(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.EQUAL, a, b);
    }
}
