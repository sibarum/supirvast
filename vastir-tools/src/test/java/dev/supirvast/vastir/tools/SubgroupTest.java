package dev.supirvast.vastir.tools;

import dev.supirvast.vast.CoreToTruffle;
import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.Statement.SubgroupArithmetic.Scan;
import dev.supirvast.vastir.core.Statement.SubgroupShuffle;
import dev.supirvast.vastir.core.Statement.SubgroupVote;
import dev.supirvast.vastir.core.SubgroupOp;
import dev.supirvast.vastir.core.UnaryOp;
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
 * Subgroup operations on both backends, each checked against the known answer on its own.
 *
 * <p>The answers are computed here from the spec's subgroup size — lane {@code local % 32} of subgroup
 * {@code local / 32} — so a GPU that grouped its lanes any other way would fail rather than merely disagree
 * with the CPU in a way nobody checks. {@code n} is never a multiple of the workgroup, so the tail lanes that a
 * subgroup kernel runs rather than stops are in every result.
 */
class SubgroupTest {

    private static final Type.Int I32 = Type.int32();
    private static final Type.Float F32 = Type.float32();
    private static final int N = 1000;
    private static final int SUBGROUP = 32;
    private static final int WORKGROUP = 64;

    /** Subtracted before the unsigned min, splitting the inputs (3..6996) into negatives and positives. */
    private static final int SHIFT = 3500;

    private static int input(int k) {
        return k * 7 + 3;
    }

    /** What lane {@code l} of the subgroup based at {@code base} holds: the input, or 0 past the end. */
    private static int held(int base, int l) {
        return base + l < N ? input(base + l) : 0;
    }

    // --- the tests ----------------------------------------------------------------------------------------

    /** {@code inclusive[g]}, {@code exclusive[g]}, {@code total[g]}: the three scans of the input by subgroup. */
    @Test
    void scansAndReductionsCombineTheirSubgroup() {
        Buffer in = new Buffer("in", 0, I32);
        Buffer inclusive = new Buffer("inclusive", 1, I32);
        Buffer exclusive = new Buffer("exclusive", 2, I32);
        Buffer total = new Buffer("total", 3, I32);
        LocalVar v = new LocalVar("v", I32);
        LocalVar a = new LocalVar("a", I32);
        LocalVar b = new LocalVar("b", I32);
        LocalVar c = new LocalVar("c", I32);
        Function kernel = kernel(
                loadOrZero(v, in),
                new Statement.SubgroupArithmetic(a, SubgroupOp.ADD, Scan.INCLUSIVE, new Expr.Read(v)),
                new Statement.SubgroupArithmetic(b, SubgroupOp.ADD, Scan.EXCLUSIVE, new Expr.Read(v)),
                new Statement.SubgroupArithmetic(c, SubgroupOp.ADD, Scan.REDUCE, new Expr.Read(v)),
                inRange(new Statement.BufferStore(inclusive, gid(), new Expr.Read(a)),
                        new Statement.BufferStore(exclusive, gid(), new Expr.Read(b)),
                        new Statement.BufferStore(total, gid(), new Expr.Read(c))));
        int[] expectedInclusive = new int[N];
        int[] expectedExclusive = new int[N];
        int[] expectedTotal = new int[N];
        for (int g = 0; g < N; g++) {
            int base = g / SUBGROUP * SUBGROUP;
            for (int l = 0; l < SUBGROUP; l++) {
                expectedTotal[g] += held(base, l);
                if (base + l < g) {
                    expectedExclusive[g] += held(base, l);
                }
            }
            expectedInclusive[g] = expectedExclusive[g] + input(g);
        }
        List<KernelColumn> columns = List.of(KernelColumn.input("in", 0, I32), KernelColumn.output("inclusive", 1, I32),
                KernelColumn.output("exclusive", 2, I32), KernelColumn.output("total", 3, I32));
        for (int[][] result : runEach(kernel, columns, new int[][] {inputs(), new int[N], new int[N], new int[N]})) {
            assertArrayEquals(expectedInclusive, result[1], "inclusive");
            assertArrayEquals(expectedExclusive, result[2], "exclusive");
            assertArrayEquals(expectedTotal, result[3], "reduce");
        }
    }

    /** A float max, exact on both backends, and an unsigned min, whose identity is the largest u32. */
    @Test
    void floatMaxAndUnsignedMinUseTheirOwnOrder() {
        Buffer in = new Buffer("in", 0, I32);
        Buffer maxima = new Buffer("maxima", 1, F32);
        Buffer minima = new Buffer("minima", 2, I32);
        LocalVar v = new LocalVar("v", I32);
        LocalVar max = new LocalVar("max", F32);
        LocalVar min = new LocalVar("min", Type.uint32());
        // Shifted so a subgroup holds negatives and positives, which order differently signed and unsigned:
        // the unsigned minimum is the smallest non-negative, where a signed one would be the most negative.
        Expr shifted = new Expr.Bitcast(add(new Expr.Read(v), i(-SHIFT)), Type.uint32());
        Function kernel = kernel(
                loadOrZero(v, in),
                new Statement.SubgroupArithmetic(max, SubgroupOp.MAX, Scan.REDUCE, new Expr.Convert(new Expr.Read(v), F32)),
                new Statement.SubgroupArithmetic(min, SubgroupOp.MIN, Scan.REDUCE, shifted),
                inRange(new Statement.BufferStore(maxima, gid(), new Expr.Read(max)),
                        new Statement.BufferStore(minima, gid(), new Expr.Bitcast(new Expr.Read(min), I32))));
        float[] expectedMax = new float[N];
        int[] expectedMin = new int[N];
        for (int g = 0; g < N; g++) {
            int base = g / SUBGROUP * SUBGROUP;
            int largest = 0;
            int smallestUnsigned = -1;
            for (int l = 0; l < SUBGROUP; l++) {
                largest = Math.max(largest, held(base, l));
                int candidate = held(base, l) - SHIFT;
                smallestUnsigned = Integer.compareUnsigned(candidate, smallestUnsigned) < 0 ? candidate : smallestUnsigned;
            }
            expectedMax[g] = largest;
            expectedMin[g] = smallestUnsigned;
        }
        List<KernelColumn> columns = List.of(KernelColumn.input("in", 0, I32), KernelColumn.output("maxima", 1, F32),
                KernelColumn.output("minima", 2, I32));
        for (int[][] result : runEach(kernel, columns, new int[][] {inputs(), new int[N], new int[N]})) {
            assertArrayEquals(expectedMax, floats(result[1]), "max");
            assertArrayEquals(expectedMin, result[2], "unsigned min");
        }
    }

    /** Each shuffle, with the up and down sources that fall outside the subgroup guarded as a kernel must. */
    @Test
    void shufflesReadTheLaneTheyName() {
        Buffer in = new Buffer("in", 0, I32);
        Buffer out = new Buffer("out", 1, I32);   // four results per invocation
        LocalVar v = new LocalVar("v", I32);
        LocalVar first = new LocalVar("first", I32);
        LocalVar pair = new LocalVar("pair", I32);
        LocalVar up = new LocalVar("up", I32);
        LocalVar down = new LocalVar("down", I32);
        Function kernel = kernel(
                loadOrZero(v, in),
                new Statement.SubgroupShuffle(first, SubgroupShuffle.Kind.INDEX, new Expr.Read(v), i(0)),
                new Statement.SubgroupShuffle(pair, SubgroupShuffle.Kind.XOR, new Expr.Read(v), i(1)),
                new Statement.SubgroupShuffle(up, SubgroupShuffle.Kind.UP, new Expr.Read(v), i(3)),
                new Statement.SubgroupShuffle(down, SubgroupShuffle.Kind.DOWN, new Expr.Read(v), i(3)),
                new Statement.If(lt(lane(), i(3)), Region.of(new Statement.Assign(up, i(-1))), Region.of()),
                new Statement.If(lt(i(SUBGROUP - 4), lane()), Region.of(new Statement.Assign(down, i(-1))), Region.of()),
                inRange(new Statement.BufferStore(out, slot(0), new Expr.Read(first)),
                        new Statement.BufferStore(out, slot(1), new Expr.Read(pair)),
                        new Statement.BufferStore(out, slot(2), new Expr.Read(up)),
                        new Statement.BufferStore(out, slot(3), new Expr.Read(down))));
        int[] expected = new int[4 * N];
        for (int g = 0; g < N; g++) {
            int base = g / SUBGROUP * SUBGROUP;
            int l = g % SUBGROUP;
            expected[4 * g] = held(base, 0);
            expected[4 * g + 1] = held(base, l ^ 1);
            expected[4 * g + 2] = l < 3 ? -1 : held(base, l - 3);
            expected[4 * g + 3] = l > SUBGROUP - 4 ? -1 : held(base, l + 3);
        }
        List<KernelColumn> columns = List.of(KernelColumn.input("in", 0, I32),
                KernelColumn.output("out", 1, I32).withLength(4 * N));
        for (int[][] result : runEach(kernel, columns, new int[][] {inputs(), new int[4 * N]})) {
            assertArrayEquals(expected, result[1]);
        }
    }

    /** Votes whose answers do not depend on the data, so a wrong grouping of lanes is what would fail them. */
    @Test
    void votesSeeTheirWholeSubgroup() {
        Buffer out = new Buffer("out", 0, I32);
        LocalVar all = new LocalVar("all", Type.BOOL);
        LocalVar any = new LocalVar("any", Type.BOOL);
        LocalVar none = new LocalVar("none", Type.BOOL);
        LocalVar same = new LocalVar("same", Type.BOOL);
        LocalVar differ = new LocalVar("differ", Type.BOOL);
        LocalVar bits = new LocalVar("bits", I32);
        Expr subgroupOfWorkgroup = new Expr.Binary(BinaryOp.DIV, new Expr.LocalInvocationId(), i(SUBGROUP));
        Function kernel = kernel(
                new Statement.SubgroupVote(all, SubgroupVote.Kind.ALL, lt(lane(), new Expr.SubgroupSize())),
                new Statement.SubgroupVote(any, SubgroupVote.Kind.ANY, eq(lane(), i(5))),
                new Statement.SubgroupVote(none, SubgroupVote.Kind.ANY, eq(lane(), i(SUBGROUP))),
                new Statement.SubgroupVote(same, SubgroupVote.Kind.ALL_EQUAL, subgroupOfWorkgroup),
                new Statement.SubgroupVote(differ, SubgroupVote.Kind.ALL_EQUAL, lane()),
                new Statement.DeclareVar(bits, i(0)),
                bit(bits, all, 1), bit(bits, any, 2), bit(bits, none, 4), bit(bits, same, 8), bit(bits, differ, 16),
                inRange(new Statement.BufferStore(out, gid(), new Expr.Read(bits))));
        int[] expected = new int[N];
        java.util.Arrays.fill(expected, 1 | 2 | 8);
        for (int[][] result : runEach(kernel, List.of(KernelColumn.output("out", 0, I32)), new int[][] {new int[N]})) {
            assertArrayEquals(expected, result[0]);
        }
    }

    /** The lane index and the size, which the device is required to give as the spec does. */
    @Test
    void theLaneIsTheLocalIdModuloTheSubgroupSize() {
        Buffer out = new Buffer("out", 0, I32);
        Function kernel = kernel(new Statement.BufferStore(out, gid(),
                add(new Expr.Binary(BinaryOp.MUL, lane(), i(1000)), new Expr.SubgroupSize())));
        int[] expected = new int[N];
        for (int g = 0; g < N; g++) {
            expected[g] = g % WORKGROUP % SUBGROUP * 1000 + SUBGROUP;
        }
        for (int[][] result : runEach(kernel, List.of(KernelColumn.output("out", 0, I32)), new int[][] {new int[N]})) {
            assertArrayEquals(expected, result[0]);
        }
    }

    /**
     * The fluid scatter's pattern: keys sorted into runs (here {@code k / 3}, so runs cross subgroup
     * boundaries), a segmented sum of each run within the subgroup, and one atomic per run per subgroup from
     * the run's last lane — instead of one per invocation.
     */
    @Test
    void aSegmentedSumMakesOneAtomicPerRunPerSubgroup() {
        int[] keys = new int[N];
        for (int k = 0; k < N; k++) {
            keys[k] = k / 3;
        }
        checkSegmentedScatter(keys);
    }

    /**
     * Nearly sorted keys, as particles are after they move: {@code A A B A A} in every five lanes, so a key
     * recurs in one subgroup with another between. Summing by key rather than by run counts the first
     * {@code A}s twice.
     */
    @Test
    void aSegmentedSumKeepsBrokenRunsApart() {
        int[] keys = new int[N];
        for (int k = 0; k < N; k++) {
            keys[k] = k / 5 * 2 + (k % 5 == 2 ? 1 : 0);
        }
        checkSegmentedScatter(keys);
    }

    /** Runs the segmented scatter over {@code keys}, checking every cell's sum and the atomics it took. */
    private static void checkSegmentedScatter(int[] keys) {
        int cells = java.util.Arrays.stream(keys).max().orElseThrow() + 1;
        int[] values = new int[N];
        int[] expected = new int[cells];
        int atomics = 0;
        for (int k = 0; k < N; k++) {
            values[k] = k & 7;
            expected[keys[k]] += values[k];
        }
        for (int k = 0; k < N; k++) {
            boolean lastOfRun = k + 1 == N || keys[k + 1] != keys[k] || (k + 1) % SUBGROUP == 0;
            atomics += lastOfRun ? 1 : 0;
        }
        List<KernelColumn> columns = List.of(KernelColumn.input("keys", 0, I32), KernelColumn.input("values", 1, I32),
                KernelColumn.output("sums", 2, I32).withLength(cells), KernelColumn.output("atomics", 3, I32).withLength(1));
        for (int[][] result : runEach(segmentedScatter(I32), columns,
                new int[][] {keys, values, new int[cells], new int[1]})) {
            assertArrayEquals(expected, result[2]);
            assertEquals(atomics, result[3][0], "atomics performed");
        }
    }

    /** A subgroup operation some lanes may skip is refused at registration, like a barrier. */
    @Test
    void aSubgroupOperationInDivergentControlFlowIsRejected() {
        LocalVar r = new LocalVar("r", I32);
        Function kernel = kernel(new Statement.If(lt(gid(), count()),
                Region.of(new Statement.SubgroupArithmetic(r, SubgroupOp.ADD, Scan.REDUCE, i(1))), Region.of()));
        try (Accelerator accelerator = new Accelerator()) {
            Rejection rejection = assertInstanceOf(Rejection.class, accelerator.register(
                    new KernelSpec(kernel, List.of(KernelColumn.output("out", 0, I32)))));
            assertTrue(rejection.detail().contains("a subgroup operation"), rejection.detail());
        }
    }

    @Test
    void aWorkgroupThatIsNotWholeSubgroupsIsRejected() {
        try (Accelerator accelerator = new Accelerator()) {
            Rejection rejection = assertInstanceOf(Rejection.class, accelerator.register(new KernelSpec(
                    lanesKernel(), List.of(KernelColumn.output("out", 0, I32)), 48)));
            assertEquals("workgroup is not a whole number of subgroups", rejection.reason());
        }
    }

    /** A size the device cannot be held to is not a defect: the kernel runs on the CPU, at the spec's size. */
    @Test
    void aSubgroupSizeTheDeviceCannotGiveRunsOnTheCpu() {
        try (Accelerator accelerator = new Accelerator()) {
            assumeGpu(accelerator);
            int size = 128;   // no current device runs 128-lane subgroups
            KernelHandle handle = accelerator.register(new KernelSpec(lanesKernel(),
                    List.of(KernelColumn.output("out", 0, I32)), 256, size)).orElseThrow();
            assertEquals(KernelHandle.Backend.CPU, handle.preferredBackend());
            int[] result = handle.run(new int[][] {new int[N]}, N)[0];
            for (int g = 0; g < N; g++) {
                assertEquals(g % 256 % size, result[g], "invocation " + g);
            }
        }
    }

    /**
     * Not an assertion: what the segmented sum buys over one global atomic per invocation, for 2^20 sorted
     * f32 values in runs of 4, 16 and 64 — the fluid scatter's particles per cell.
     */
    @Test
    void throughputOfASegmentedScatter() {
        try (Accelerator accelerator = new Accelerator()) {
            assumeGpu(accelerator);
            StringBuilder report = new StringBuilder("[subgroups] " + accelerator.capabilities().deviceName()
                    + ", 2^20 sorted f32, ms per scatter (one atomic per invocation / segmented sum):");
            for (int run : new int[] {4, 16, 64}) {
                report.append(String.format("  runs of %d: %.3f / %.3f", run,
                        scatterMs(accelerator, run, false), scatterMs(accelerator, run, true)));
            }
            System.out.println(report);
        }
    }

    private static double scatterMs(Accelerator accelerator, int run, boolean segmented) {
        int n = 1 << 20;
        int cells = n / run;
        Buffer keys = new Buffer("keys", 0, I32);
        Buffer values = new Buffer("values", 1, F32);
        Buffer sums = new Buffer("sums", 2, F32);
        Function direct = kernel(new Statement.AtomicUpdate(AtomicOp.ADD, sums, new Expr.BufferLoad(keys, gid()),
                new Expr.BufferLoad(values, gid())));
        List<KernelColumn> columns = new ArrayList<>(List.of(KernelColumn.input("keys", 0, I32),
                KernelColumn.input("values", 1, F32), KernelColumn.output("sums", 2, F32).withLength(cells)));
        if (segmented) {
            columns.add(KernelColumn.output("atomics", 3, I32).withLength(1));
        }
        int[] keyData = new int[n];
        int[] valueData = new int[n];
        float firstCell = 0;
        for (int k = 0; k < n; k++) {
            keyData[k] = k / run;
            valueData[k] = Float.floatToRawIntBits(k & 7);
            firstCell += k < run ? k & 7 : 0;
        }
        KernelHandle handle = accelerator.register(
                new KernelSpec(segmented ? segmentedScatter(F32) : direct, columns, 256)).orElseThrow();
        assertEquals(KernelHandle.Backend.GPU, handle.preferredBackend());
        ResidentBuffer keyBuffer = accelerator.allocate(I32, n);
        ResidentBuffer valueBuffer = accelerator.allocate(F32, n);
        ResidentBuffer sum = accelerator.allocate(F32, cells);
        keyBuffer.write(keyData);
        valueBuffer.write(valueData);
        List<ResidentBuffer> buffers = new ArrayList<>(List.of(keyBuffer, valueBuffer, sum));
        if (segmented) {
            buffers.add(accelerator.allocate(I32, 1));
        }
        long warm = System.nanoTime();
        while (System.nanoTime() - warm < 500_000_000L) {   // bring a laptop GPU off its idle clock
            handle.dispatch(buffers, n);
            sum.read();
        }
        sum.write(new int[cells]);
        handle.dispatch(buffers, n);
        assertEquals(firstCell, floats(sum.read())[0], (segmented ? "segmented" : "direct") + ", runs of " + run);
        int steps = 50;
        long start = System.nanoTime();
        for (int s = 0; s < steps; s++) {
            handle.dispatch(buffers, n);
        }
        sum.read();
        double ms = (System.nanoTime() - start) / 1e6 / steps;
        handle.close();
        return ms;
    }

    // --- kernels and harness ------------------------------------------------------------------------------

    /**
     * The segmented scatter over {@code element} values: one atomic per run of equal keys per subgroup, where
     * a run is a maximal stretch of neighbouring lanes with the same key. First each lane finds where its run
     * starts — a lane whose lower neighbour has another key is a start, and an inclusive max of the starts'
     * lane numbers gives each lane its own. Then a Hillis–Steele scan by shuffle-up: after the round with
     * stride {@code d}, each lane holds the sum of its run's lanes among the {@code 2d} ending at it, adding
     * the lane {@code d} below only when that lane is at or after the run's start. The run's last lane adds
     * the run once.
     *
     * <p>Keyed by run, not by key. Comparing the key {@code d} lanes below instead would be right only for
     * sorted input: with keys {@code A B A} in one subgroup, the second {@code A} would add the first's partial
     * sum, and both {@code A}s end a run, so the first would be counted twice. Particles after an advection
     * step are only nearly sorted, which is exactly that case (found by {@code vexelray-sim-fluid}, 865ec14).
     */
    private static Function segmentedScatter(Type element) {
        Buffer keys = new Buffer("keys", 0, I32);
        Buffer values = new Buffer("values", 1, element);
        Buffer sums = new Buffer("sums", 2, element);
        Buffer atomics = new Buffer("atomics", 3, I32);
        LocalVar key = new LocalVar("key", I32);
        LocalVar acc = new LocalVar("acc", element);
        LocalVar d = new LocalVar("d", I32);
        LocalVar keyBelow = new LocalVar("keyBelow", I32);
        LocalVar head = new LocalVar("head", I32);
        LocalVar runStart = new LocalVar("runStart", I32);
        LocalVar from = new LocalVar("from", element);
        LocalVar nextKey = new LocalVar("nextKey", I32);
        Expr zero = element instanceof Type.Float ? new Expr.ConstFloat(F32, 0) : i(0);
        return kernel(
                new Statement.DeclareVar(key, i(-1)),
                new Statement.DeclareVar(acc, zero),
                inRange(new Statement.Assign(key, new Expr.BufferLoad(keys, gid())),
                        new Statement.Assign(acc, new Expr.BufferLoad(values, gid()))),
                new Statement.SubgroupShuffle(keyBelow, SubgroupShuffle.Kind.UP, new Expr.Read(key), i(1)),
                new Statement.DeclareVar(head, i(0)),
                new Statement.If(new Expr.Binary(BinaryOp.LOGICAL_OR, eq(lane(), i(0)),
                                new Expr.Unary(UnaryOp.LOGICAL_NOT, eq(new Expr.Read(keyBelow), new Expr.Read(key)))),
                        Region.of(new Statement.Assign(head, lane())), Region.of()),
                new Statement.SubgroupArithmetic(runStart, SubgroupOp.MAX, Scan.INCLUSIVE, new Expr.Read(head)),
                new Statement.DeclareVar(d, i(1)),
                new Statement.While(lt(new Expr.Read(d), new Expr.SubgroupSize()), Region.of(
                        new Statement.SubgroupShuffle(from, SubgroupShuffle.Kind.UP, new Expr.Read(acc), new Expr.Read(d)),
                        new Statement.If(new Expr.Unary(UnaryOp.LOGICAL_NOT,
                                        lt(new Expr.Binary(BinaryOp.SUB, lane(), new Expr.Read(d)), new Expr.Read(runStart))),
                                Region.of(new Statement.Assign(acc,
                                        new Expr.Binary(BinaryOp.ADD, new Expr.Read(acc), new Expr.Read(from)))),
                                Region.of()),
                        new Statement.Assign(d, add(new Expr.Read(d), new Expr.Read(d))))),
                new Statement.SubgroupShuffle(nextKey, SubgroupShuffle.Kind.DOWN, new Expr.Read(key), i(1)),
                new Statement.If(new Expr.Binary(BinaryOp.LOGICAL_AND, lt(i(-1), new Expr.Read(key)),
                                new Expr.Binary(BinaryOp.LOGICAL_OR,
                                        eq(lane(), add(new Expr.SubgroupSize(), i(-1))),
                                        new Expr.Unary(UnaryOp.LOGICAL_NOT, eq(new Expr.Read(nextKey), new Expr.Read(key))))),
                        Region.of(new Statement.AtomicUpdate(AtomicOp.ADD, sums, new Expr.Read(key), new Expr.Read(acc)),
                                new Statement.AtomicUpdate(AtomicOp.ADD, atomics, i(0), i(1))),
                        Region.of()));
    }

    /** {@code out[gid] = lane}, and nothing else: the smallest kernel whose meaning is its subgroups'. */
    private static Function lanesKernel() {
        return kernel(new Statement.BufferStore(new Buffer("out", 0, I32), gid(), lane()));
    }

    /** The kernel on the CPU, then on the GPU when one is present — each result for the caller to check. */
    private static List<int[][]> runEach(Function kernel, List<KernelColumn> columns, int[][] data) {
        List<int[][]> results = new ArrayList<>();
        List<Buffer> buffers = columns.stream().map(c -> new Buffer(c.name(), c.binding(), c.type())).toList();
        int[][] cpu = copy(data);
        new CoreToTruffle().lowerDispatch(kernel, buffers, WORKGROUP, SUBGROUP).dispatch(cpu, N);
        results.add(cpu);
        try (Accelerator accelerator = new Accelerator()) {
            KernelHandle handle = accelerator.register(new KernelSpec(kernel, columns, WORKGROUP, SUBGROUP)).orElseThrow();
            if (handle.preferredBackend() == KernelHandle.Backend.GPU) {
                results.add(handle.run(copy(data), N));
            } else {
                assertFalse(accelerator.capabilities().gpuAvailable(), "registered CPU-only on "
                        + accelerator.capabilities().deviceName());
                assertFalse(Boolean.getBoolean("supirvast.requireGpu"), "no Vulkan device, and one is required");
                System.out.println("[subgroups] GPU leg not run: no Vulkan device");
            }
        }
        return results;
    }

    private static void assumeGpu(Accelerator accelerator) {
        assumeTrue(accelerator.capabilities().gpuAvailable() || Boolean.getBoolean("supirvast.requireGpu"),
                "no Vulkan device");
    }

    /** {@code v = gid < n ? in[gid] : 0} — every lane holds a value, the tail's a harmless one. */
    private static Statement loadOrZero(LocalVar v, Buffer in) {
        return new Statement.If(new Expr.ConstBool(true), Region.of(
                new Statement.DeclareVar(v, i(0)),
                inRange(new Statement.Assign(v, new Expr.BufferLoad(in, gid())))), Region.of());
    }

    private static Statement inRange(Statement... statements) {
        return new Statement.If(lt(gid(), count()), Region.of(statements), Region.of());
    }

    private static Function kernel(Statement... statements) {
        List<Statement> body = new ArrayList<>(List.of(statements));
        body.add(new Statement.ReturnVoid());
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), new Region(body));
    }

    private static int[] inputs() {
        int[] in = new int[N];
        for (int k = 0; k < N; k++) {
            in[k] = input(k);
        }
        return in;
    }

    private static int[][] copy(int[][] data) {
        int[][] out = new int[data.length][];
        for (int k = 0; k < data.length; k++) {
            out[k] = data[k].clone();
        }
        return out;
    }

    private static float[] floats(int[] words) {
        float[] out = new float[words.length];
        for (int k = 0; k < words.length; k++) {
            out[k] = Float.intBitsToFloat(words[k]);
        }
        return out;
    }

    /** {@code bits += flag ? weight : 0}, spelled without a select, which core does not have. */
    private static Statement bit(LocalVar bits, LocalVar flag, int weight) {
        return new Statement.If(new Expr.Read(flag),
                Region.of(new Statement.Assign(bits, add(new Expr.Read(bits), i(weight)))), Region.of());
    }

    private static Expr slot(int k) {
        return add(new Expr.Binary(BinaryOp.MUL, gid(), i(4)), i(k));
    }

    private static Expr gid() {
        return new Expr.InvocationId();
    }

    private static Expr lane() {
        return new Expr.SubgroupInvocationId();
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

    private static Expr lt(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.LESS_THAN, a, b);
    }

    private static Expr eq(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.EQUAL, a, b);
    }
}
