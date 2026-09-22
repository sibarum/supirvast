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

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Buffers that stay where the kernel runs, dispatched against repeatedly with nothing copied in between.
 *
 * <p>The ordering is what is under test as much as the data. Dispatches are submitted without waiting, so
 * ping-ponging two buffers a hundred times is a hundred command buffers in flight, each of which must see the
 * previous one's writes and must not overwrite a buffer the previous one is still reading. A missing barrier
 * does not fail loudly; it produces a number that is off by a few steps, sometimes. So the answer here is one
 * that every step contributes to.
 */
class ResidentBufferTest {

    private static final Type.Int I32 = Type.int32();
    private static final Type.Float F32 = Type.float32();

    /** Deliberately not a multiple of the default workgroup size. */
    private static final int N = 1000;

    /** {@code out[gid] = in[gid] + 1} */
    private static KernelSpec increment() {
        Buffer out = new Buffer("out", 0, I32);
        Buffer in = new Buffer("in", 1, I32);
        Expr gid = new Expr.InvocationId();
        Function kernel = new Function("main", new Type.FunctionType(Type.VOID, List.of()), Region.of(
                new Statement.BufferStore(out, gid,
                        new Expr.Binary(BinaryOp.ADD, new Expr.BufferLoad(in, gid), new Expr.ConstInt(I32, 1))),
                new Statement.ReturnVoid()));
        return new KernelSpec(kernel, List.of(KernelColumn.output("out", 0, I32), KernelColumn.input("in", 1, I32)));
    }

    /** {@code bins[gid & 15] += 1} */
    private static KernelSpec histogram() {
        Buffer bins = new Buffer("bins", 0, I32);
        Function kernel = new Function("main", new Type.FunctionType(Type.VOID, List.of()), Region.of(
                new Statement.AtomicUpdate(AtomicOp.ADD, bins,
                        new Expr.Binary(BinaryOp.BIT_AND, new Expr.InvocationId(), new Expr.ConstInt(I32, 15)),
                        new Expr.ConstInt(I32, 1)),
                new Statement.ReturnVoid()));
        return new KernelSpec(kernel, List.of(KernelColumn.output("bins", 0, I32).withLength(16)));
    }

    private static int[] ramp(int n) {
        int[] values = new int[n];
        Arrays.setAll(values, k -> k * 7);
        return values;
    }

    private static int[] plus(int[] values, int amount) {
        return Arrays.stream(values).map(v -> v + amount).toArray();
    }

    @Test
    void aHundredStepsPingPongWithoutAWait() {
        try (Accelerator accelerator = new Accelerator()) {
            assumeGpu(accelerator);
            KernelHandle step = accelerator.register(increment()).orElseThrow();
            ResidentBuffer a = accelerator.allocate(I32, N);
            ResidentBuffer b = accelerator.allocate(I32, N);
            assertTrue(a.onDevice() && b.onDevice());
            a.write(ramp(N));
            for (int k = 0; k < 100; k++) {
                // even steps read a and write b, odd steps the reverse
                step.dispatch(k % 2 == 0 ? List.of(b, a) : List.of(a, b), N);
            }
            assertArrayEquals(plus(ramp(N), 100), a.read(), "after an even number of steps the result is in a");
        }
    }

    @Test
    void aResidentBufferKeepsWhatEarlierDispatchesLeft() {
        try (Accelerator accelerator = new Accelerator()) {
            assumeGpu(accelerator);
            KernelHandle count = accelerator.register(histogram()).orElseThrow();
            ResidentBuffer bins = accelerator.allocate(I32, 16);
            bins.write(new int[16]);
            for (int k = 0; k < 10; k++) {
                count.dispatch(List.of(bins), 1024);
            }
            int[] expected = new int[16];
            Arrays.fill(expected, 10 * 1024 / 16);
            assertArrayEquals(expected, bins.read());
        }
    }

    /** A released pipeline sends dispatch to the CPU, which reads device buffers, runs, and writes them back. */
    @Test
    void theCpuFallbackRunsOverDeviceBuffers() {
        try (Accelerator accelerator = new Accelerator()) {
            assumeGpu(accelerator);
            KernelHandle step = accelerator.register(increment()).orElseThrow();
            ResidentBuffer a = accelerator.allocate(I32, N);
            ResidentBuffer b = accelerator.allocate(I32, N);
            a.write(ramp(N));
            step.dispatch(List.of(b, a), N);   // on the GPU
            step.close();
            assertEquals(KernelHandle.Backend.CPU, step.preferredBackend());
            step.dispatch(List.of(a, b), N);   // on the CPU, over the same device buffers
            assertArrayEquals(plus(ramp(N), 2), a.read(), "one step on each backend");
        }
    }

    /** With no device, a resident buffer is a host array the CPU runs over in place. */
    @Test
    void hostBuffersAreUpdatedInPlace() {
        try (Accelerator accelerator = new Accelerator()) {
            KernelHandle step = accelerator.register(increment()).orElseThrow();
            step.close();   // force the CPU path even where there is a device
            ResidentBuffer a = ResidentBuffer.onHost(I32, N);
            ResidentBuffer b = ResidentBuffer.onHost(I32, N);
            a.write(ramp(N));
            for (int k = 0; k < 4; k++) {
                step.dispatch(k % 2 == 0 ? List.of(b, a) : List.of(a, b), N);
            }
            assertArrayEquals(plus(ramp(N), 4), a.read());
        }
    }

    @Test
    void mismatchedBuffersAreRefused() {
        try (Accelerator accelerator = new Accelerator()) {
            KernelHandle step = accelerator.register(increment()).orElseThrow();
            ResidentBuffer ints = ResidentBuffer.onHost(I32, N);
            ResidentBuffer floats = ResidentBuffer.onHost(F32, N);
            ResidentBuffer shortOne = ResidentBuffer.onHost(I32, N - 1);
            assertThrows(IllegalArgumentException.class, () -> step.dispatch(List.of(ints), N), "one buffer short");
            assertThrows(IllegalArgumentException.class, () -> step.dispatch(List.of(floats, ints), N), "wrong type");
            assertThrows(IllegalArgumentException.class, () -> step.dispatch(List.of(ints, shortOne), N), "too short");
            ints.close();
            assertThrows(IllegalArgumentException.class, () -> step.dispatch(List.of(ints, shortOne), 1), "closed");
        }
    }

    /**
     * Not an assertion: stepping a 4 MB field a hundred times, by round trips through {@link KernelHandle#run}
     * against resident dispatches and one read at the end.
     */
    @Test
    void steppingThroughputResidentAgainstRoundTrips() {
        int n = 1 << 20;
        int steps = 100;
        Buffer out = new Buffer("out", 0, F32);
        Buffer in = new Buffer("in", 1, F32);
        LocalVar x = new LocalVar("x", F32);
        Expr gid = new Expr.InvocationId();
        Function kernel = new Function("main", new Type.FunctionType(Type.VOID, List.of()), Region.of(
                new Statement.DeclareVar(x, new Expr.BufferLoad(in, gid)),
                new Statement.BufferStore(out, gid, new Expr.Binary(BinaryOp.ADD,
                        new Expr.Binary(BinaryOp.MUL, new Expr.Read(x), new Expr.ConstFloat(F32, 0.5)),
                        new Expr.ConstFloat(F32, 1))),
                new Statement.ReturnVoid()));
        KernelSpec spec = new KernelSpec(kernel, List.of(KernelColumn.output("out", 0, F32),
                KernelColumn.input("in", 1, F32)));

        try (Accelerator accelerator = new Accelerator()) {
            assumeGpu(accelerator);
            KernelHandle step = accelerator.register(spec).orElseThrow();

            int[] field = new int[n];
            step.run(new int[][] {new int[n], field}, n);   // warm
            long start = System.nanoTime();
            for (int k = 0; k < steps; k++) {
                field = step.run(new int[][] {new int[n], field}, n)[0];
            }
            double roundTrips = (System.nanoTime() - start) / 1e6;

            ResidentBuffer a = accelerator.allocate(F32, n);
            ResidentBuffer b = accelerator.allocate(F32, n);
            a.write(new int[n]);
            step.dispatch(List.of(b, a), n);   // warm
            a.read();
            start = System.nanoTime();
            for (int k = 0; k < steps; k++) {
                step.dispatch(k % 2 == 0 ? List.of(b, a) : List.of(a, b), n);
            }
            int[] result = a.read();
            double resident = (System.nanoTime() - start) / 1e6;

            System.out.printf("[resident] 2^20 f32, %d steps: round trips %.1f ms (%.2f ms/step), "
                            + "resident %.1f ms (%.3f ms/step) including one read%n",
                    steps, roundTrips, roundTrips / steps, resident, resident / steps);
            // Both converge on the fixed point of x = x/2 + 1.
            assertEquals(2f, Float.intBitsToFloat(result[n / 2]), 1e-3f);
        }
    }

    private static void assumeGpu(Accelerator accelerator) {
        assumeTrue(accelerator.capabilities().gpuAvailable() || Boolean.getBoolean("supirvast.requireGpu"),
                "no Vulkan device");
    }
}
