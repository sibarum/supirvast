package dev.supirvast.vastir.tools;

import com.oracle.truffle.api.CallTarget;
import dev.supirvast.vast.CoreToTruffle;
import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.UnaryOp;
import dev.supirvast.vastir.lower.CapabilityException;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.lower.SpirvTarget;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Atomic read-modify-write on storage-buffer elements, run on both backends.
 *
 * <p>Every kernel here collides on purpose — many invocations updating the same few elements — because a
 * collision is the only thing an atomic is for, and a kernel without one would pass with plain stores too.
 * Each backend is checked against the known answer on its own. The answers are ones invocation order cannot
 * change: counts, extremes, a permutation, and sums of small integers, which {@code f32} adds exactly in any
 * order.
 */
class AtomicKernelTest {

    private static final Type.Int I32 = Type.int32();
    private static final Type.Int U32 = Type.uint32();
    private static final Type.Float F32 = Type.float32();
    private static final int N = 1024;

    private static Expr gid() {
        return new Expr.InvocationId();
    }

    private static Expr i(long value) {
        return new Expr.ConstInt(I32, value);
    }

    private static Function kernel(Statement... statements) {
        List<Statement> body = new ArrayList<>(List.of(statements));
        body.add(new Statement.ReturnVoid());
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), new Region(body));
    }

    /** {@code bins[gid & 15] += 1}: sixteen elements, 64 increments each, all racing. */
    @Test
    void histogramCountsEveryIncrement() {
        Buffer bins = new Buffer("bins", 0, I32);
        Function kernel = kernel(new Statement.AtomicUpdate(AtomicOp.ADD, bins,
                new Expr.Binary(BinaryOp.BIT_AND, gid(), i(15)), i(1)));
        int[] expected = new int[16];
        Arrays.fill(expected, N / 16);

        for (int[][] result : runEach(kernel, List.of(KernelColumn.output("bins", 0, I32).withLength(16)),
                new int[][] {new int[16]})) {
            assertArrayEquals(expected, result[0]);
        }
    }

    /**
     * {@code slot = counter[0]++; slots[slot] = gid}: the old value is a ticket, which is what an append
     * needs. Every invocation must get a distinct one, so the slots hold a permutation of the invocations.
     */
    @Test
    void theOldValueHandsOutDistinctTickets() {
        Buffer counter = new Buffer("counter", 0, I32);
        Buffer slots = new Buffer("slots", 1, I32);
        LocalVar ticket = new LocalVar("ticket", I32);
        Function kernel = kernel(
                new Statement.AtomicUpdate(ticket, AtomicOp.ADD, counter, i(0), i(1)),
                new Statement.BufferStore(slots, new Expr.Read(ticket), gid()));
        int[] everyInvocation = new int[N];
        Arrays.setAll(everyInvocation, k -> k);

        for (int[][] result : runEach(kernel, List.of(KernelColumn.output("counter", 0, I32).withLength(1),
                KernelColumn.output("slots", 1, I32)), new int[][] {new int[1], new int[N]})) {
            assertEquals(N, result[0][0], "the counter must have been bumped once per invocation");
            int[] sorted = result[1].clone();
            Arrays.sort(sorted);
            assertArrayEquals(everyInvocation, sorted, "every invocation must have drawn a distinct ticket");
        }
    }

    /** Signed min and max, over values straddling zero. */
    @Test
    void signedMinAndMaxFindTheExtremes() {
        Buffer out = new Buffer("out", 0, I32);
        Buffer values = new Buffer("values", 1, I32);
        Function kernel = kernel(
                new Statement.AtomicUpdate(AtomicOp.MIN, out, i(0), new Expr.BufferLoad(values, gid())),
                new Statement.AtomicUpdate(AtomicOp.MAX, out, i(1), new Expr.BufferLoad(values, gid())));
        int[] input = new int[N];
        Arrays.setAll(input, k -> (k * 7919) % 2001 - 1000);

        for (int[][] result : runEach(kernel, List.of(KernelColumn.output("out", 0, I32).withLength(2),
                KernelColumn.input("values", 1, I32)),
                new int[][] {{Integer.MAX_VALUE, Integer.MIN_VALUE}, input})) {
            assertEquals(Arrays.stream(input).min().orElseThrow(), result[0][0]);
            assertEquals(Arrays.stream(input).max().orElseThrow(), result[0][1]);
        }
    }

    /**
     * Unsigned min and max, where the answer differs from the signed one: {@code 0xFFFFFFFF} is the largest
     * unsigned value and would be {@code -1} read as signed.
     */
    @Test
    void unsignedMinAndMaxCompareUnsigned() {
        Buffer out = new Buffer("out", 0, U32);
        Buffer values = new Buffer("values", 1, U32);
        Function kernel = kernel(
                new Statement.AtomicUpdate(AtomicOp.MIN, out, i(0), new Expr.BufferLoad(values, gid())),
                new Statement.AtomicUpdate(AtomicOp.MAX, out, i(1), new Expr.BufferLoad(values, gid())));
        int[] input = new int[N];
        Arrays.setAll(input, k -> k + 5);
        input[N / 2] = 0xFFFFFFFF;

        for (int[][] result : runEach(kernel, List.of(KernelColumn.output("out", 0, U32).withLength(2),
                KernelColumn.input("values", 1, U32)), new int[][] {{0xFFFFFFFF, 0}, input})) {
            assertEquals(5, result[0][0], "unsigned min");
            assertEquals(0xFFFFFFFF, result[0][1], "unsigned max: 0xFFFFFFFF, which a signed max would lose");
        }
    }

    /** And, or and xor, each against one shared word. */
    @Test
    void bitwiseUpdatesCombineEveryInvocation() {
        Buffer out = new Buffer("out", 0, I32);
        Expr bit = new Expr.Binary(BinaryOp.SHIFT_LEFT, i(1), new Expr.Binary(BinaryOp.BIT_AND, gid(), i(31)));
        Function kernel = kernel(
                new Statement.AtomicUpdate(AtomicOp.OR, out, i(0), bit),
                new Statement.AtomicUpdate(AtomicOp.AND, out, i(1), new Expr.Unary(UnaryOp.NOT, bit)),
                new Statement.AtomicUpdate(AtomicOp.XOR, out, i(2), bit));

        for (int[][] result : runEach(kernel, List.of(KernelColumn.output("out", 0, I32).withLength(3)),
                new int[][] {{0, -1, 0}})) {
            assertEquals(-1, result[0][0], "or: every bit set by some invocation");
            assertEquals(0, result[0][1], "and: every bit cleared by some invocation");
            assertEquals(0, result[0][2], "xor: each bit toggled N/32 times, an even number");
        }
    }

    /**
     * A maximum built from compare-exchange alone, as a retry loop: read, and if the value is still smaller,
     * try to swap; the swap took exactly when the old value is the one that was read. {@code previous} is
     * reused across iterations, which is the case the "declared only if nothing else declares it" rule is for.
     */
    @Test
    void compareExchangeRetryLoopReachesTheMaximum() {
        Buffer out = new Buffer("out", 0, I32);
        Buffer values = new Buffer("values", 1, I32);
        LocalVar value = new LocalVar("value", I32);
        LocalVar current = new LocalVar("current", I32);
        LocalVar previous = new LocalVar("previous", I32);
        LocalVar done = new LocalVar("done", Type.BOOL);
        Expr notSmaller = new Expr.Unary(UnaryOp.LOGICAL_NOT,
                new Expr.Binary(BinaryOp.LESS_THAN, new Expr.Read(current), new Expr.Read(value)));
        Function kernel = kernel(
                new Statement.DeclareVar(value, new Expr.BufferLoad(values, gid())),
                new Statement.DeclareVar(previous, i(0)),
                new Statement.DeclareVar(current, i(0)),
                new Statement.DeclareVar(done, new Expr.ConstBool(false)),
                new Statement.While(new Expr.Unary(UnaryOp.LOGICAL_NOT, new Expr.Read(done)), Region.of(
                        new Statement.Assign(current, new Expr.BufferLoad(out, i(0))),
                        new Statement.If(notSmaller,
                                Region.of(new Statement.Assign(done, new Expr.ConstBool(true))),
                                Region.of(
                                        new Statement.AtomicCompareExchange(previous, out, i(0),
                                                new Expr.Read(current), new Expr.Read(value)),
                                        new Statement.Assign(done, new Expr.Binary(BinaryOp.EQUAL,
                                                new Expr.Read(previous), new Expr.Read(current))))))));
        int[] input = new int[N];
        Arrays.setAll(input, k -> (k * 7919) % 100_003);

        for (int[][] result : runEach(kernel, List.of(KernelColumn.output("out", 0, I32).withLength(1),
                KernelColumn.input("values", 1, I32)), new int[][] {{Integer.MIN_VALUE}, input})) {
            assertEquals(Arrays.stream(input).max().orElseThrow(), result[0][0]);
        }
    }

    /** Exchange hands back what was there: summing the old values of a chain of swaps recovers them all. */
    @Test
    void exchangeReturnsWhatItReplaced() {
        Buffer cell = new Buffer("cell", 0, I32);
        Buffer seen = new Buffer("seen", 1, I32);
        LocalVar old = new LocalVar("old", I32);
        Function kernel = kernel(
                new Statement.AtomicUpdate(old, AtomicOp.EXCHANGE, cell, i(0), gid()),
                new Statement.BufferStore(seen, gid(), new Expr.Read(old)));

        for (int[][] result : runEach(kernel, List.of(KernelColumn.output("cell", 0, I32).withLength(1),
                KernelColumn.output("seen", 1, I32)), new int[][] {{-1}, new int[N]})) {
            // The values in play are -1 and every gid; exactly one of them is left in the cell, and every
            // other one was handed back to exactly one invocation.
            int[] all = Arrays.copyOf(result[1], N + 1);
            all[N] = result[0][0];
            Arrays.sort(all);
            int[] expected = new int[N + 1];
            Arrays.setAll(expected, k -> k - 1);
            assertArrayEquals(expected, all);
        }
    }

    /** {@code sum[gid & 7] += (gid & 3)} in f32 — small integers, so the sum is exact in any order. */
    @Test
    void floatAddAccumulates() {
        Buffer sum = new Buffer("sum", 0, F32);
        Function kernel = kernel(new Statement.AtomicUpdate(AtomicOp.ADD, sum,
                new Expr.Binary(BinaryOp.BIT_AND, gid(), i(7)),
                new Expr.Convert(new Expr.Binary(BinaryOp.BIT_AND, gid(), i(3)), F32)));
        float[] expected = new float[8];
        for (int k = 0; k < N; k++) {
            expected[k & 7] += k & 3;
        }

        for (int[][] result : runEach(kernel, List.of(KernelColumn.output("sum", 0, F32).withLength(8)),
                new int[][] {new int[8]})) {
            for (int k = 0; k < 8; k++) {
                assertEquals(expected[k], Float.intBitsToFloat(result[0][k]), "sum[" + k + "]");
            }
        }
    }

    /** Float min and max, gated on their own extension. */
    @Test
    void floatMinAndMaxFindTheExtremes() {
        Buffer out = new Buffer("out", 0, F32);
        Buffer values = new Buffer("values", 1, F32);
        Function kernel = kernel(
                new Statement.AtomicUpdate(AtomicOp.MIN, out, i(0), new Expr.BufferLoad(values, gid())),
                new Statement.AtomicUpdate(AtomicOp.MAX, out, i(1), new Expr.BufferLoad(values, gid())));
        int[] input = new int[N];
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (int k = 0; k < N; k++) {
            float v = ((k * 7919) % 2001 - 1000) / 8f;
            input[k] = Float.floatToRawIntBits(v);
            min = Math.min(min, v);
            max = Math.max(max, v);
        }

        for (int[][] result : runEach(kernel, List.of(KernelColumn.output("out", 0, F32).withLength(2),
                KernelColumn.input("values", 1, F32)), new int[][] {
                        {Float.floatToRawIntBits(Float.POSITIVE_INFINITY),
                                Float.floatToRawIntBits(Float.NEGATIVE_INFINITY)}, input})) {
            assertEquals(min, Float.intBitsToFloat(result[0][0]), "min");
            assertEquals(max, Float.intBitsToFloat(result[0][1]), "max");
        }
    }

    /** An atomic writes, so the buffer it targets must not be promised read-only. */
    @Test
    void anAtomicTargetIsNotDecoratedNonWritable() {
        Buffer bins = new Buffer("bins", 0, I32);
        byte[] spirv = lower(kernel(new Statement.AtomicUpdate(AtomicOp.ADD, bins, i(0), i(1))),
                SpirvTarget.unconstrained());
        assertFalse(decoratesNonWritable(spirv),
                "the only access to 'bins' is an atomic add, which writes — NonWritable would be a lie");
    }

    /** A float atomic add under a budget without its capability is refused with a witness, not emitted. */
    @Test
    void floatAddOutsideTheBudgetIsRefused() {
        Buffer sum = new Buffer("sum", 0, F32);
        Function kernel = kernel(new Statement.AtomicUpdate(AtomicOp.ADD, sum, i(0), new Expr.ConstFloat(F32, 1)));
        CapabilityException refused = assertThrows(CapabilityException.class,
                () -> lower(kernel, SpirvTarget.restrictedTo(Set.of())));
        assertTrue(refused.getMessage().contains("AtomicFloat32AddEXT"), refused.getMessage());
    }

    /** SPIR-V's table, enforced where the statement is built rather than where it is lowered. */
    @Test
    void operationsTheElementTypeLacksAreRejectedAtConstruction() {
        Buffer floats = new Buffer("floats", 0, F32);
        Buffer longs = new Buffer("longs", 1, Type.int64());
        Buffer ints = new Buffer("ints", 2, I32);
        assertThrows(IllegalArgumentException.class, () -> new Statement.AtomicUpdate(AtomicOp.SUB, floats,
                i(0), new Expr.ConstFloat(F32, 1)), "SPIR-V has no float atomic subtract");
        assertThrows(IllegalArgumentException.class, () -> new Statement.AtomicUpdate(AtomicOp.ADD, longs,
                i(0), new Expr.ConstInt(Type.int64(), 1)), "64-bit atomics are not supported yet");
        assertThrows(IllegalArgumentException.class, () -> new Statement.AtomicUpdate(AtomicOp.ADD, ints,
                i(0), new Expr.ConstFloat(F32, 1)), "the value must have the element's type");
        assertThrows(IllegalArgumentException.class, () -> new Statement.AtomicCompareExchange(
                new LocalVar("p", F32), floats, i(0), new Expr.ConstFloat(F32, 0), new Expr.ConstFloat(F32, 1)),
                "compare-exchange is integer-only");
    }

    // --- running -----------------------------------------------------------------------------------------

    /**
     * Runs the kernel on the CPU, and on the GPU when one is present and the kernel's capabilities fit it,
     * and returns each result to be checked on its own. Registration through {@link Accelerator} is also the
     * {@code spirv-val} gate, so a kernel that lowers to invalid SPIR-V fails here either way.
     */
    private static List<int[][]> runEach(Function kernel, List<KernelColumn> columns, int[][] data) {
        List<int[][]> results = new ArrayList<>();
        List<Buffer> buffers = columns.stream().map(c -> new Buffer(c.name(), c.binding(), c.type())).toList();
        CallTarget cpu = new CoreToTruffle().lowerKernel(kernel, buffers);
        int[][] cpuData = copy(data);
        for (int k = 0; k < N; k++) {
            cpu.call(k, cpuData);
        }
        results.add(cpuData);

        try (Accelerator accelerator = new Accelerator()) {
            KernelHandle handle = accelerator.register(new KernelSpec(kernel, columns)).orElseThrow();
            if (handle.preferredBackend() == KernelHandle.Backend.GPU) {
                results.add(handle.run(copy(data), N));
            } else {
                System.out.println("[atomics] GPU leg not run: " + (accelerator.capabilities().gpuAvailable()
                        ? "the device lacks a capability this kernel needs" : "no Vulkan device"));
            }
        }
        return results;
    }

    private static int[][] copy(int[][] data) {
        int[][] out = new int[data.length][];
        for (int k = 0; k < data.length; k++) {
            out[k] = data[k].clone();
        }
        return out;
    }

    private static byte[] lower(Function kernel, SpirvTarget target) {
        return new CoreToSpirv().lower(new CoreModule().addEntryPoint(EntryPoint.compute(kernel, 1, 1, 1)), target)
                .toByteArray();
    }

    /** {@code OpDecorate <id> NonWritable}, found in the words rather than in cross-compiled source. */
    private static boolean decoratesNonWritable(byte[] spirv) {
        for (int word = 5; word < spirv.length / 4; ) {
            int instruction = wordAt(spirv, word);
            int count = instruction >>> 16;
            if ((instruction & 0xFFFF) == 71 && count >= 3 && wordAt(spirv, word + 2) == 24) {
                return true;
            }
            word += Math.max(count, 1);
        }
        return false;
    }

    private static int wordAt(byte[] bytes, int word) {
        int b = word * 4;
        return (bytes[b] & 0xFF) | (bytes[b + 1] & 0xFF) << 8 | (bytes[b + 2] & 0xFF) << 16 | (bytes[b + 3] & 0xFF) << 24;
    }

    @Test
    void theGpuLegRunsOnThisMachineWhenRequired() {
        assumeTrue(Boolean.getBoolean("supirvast.requireGpu"), "only meaningful with -Dsupirvast.requireGpu");
        try (Accelerator accelerator = new Accelerator()) {
            assertTrue(accelerator.capabilities().gpuAvailable(), "no Vulkan device, so no GPU leg ran");
        }
    }
}
