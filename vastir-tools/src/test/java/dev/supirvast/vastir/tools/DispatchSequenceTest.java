package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Dispatches recorded once and run as one submission, against the same dispatches made one at a time.
 *
 * <p>What a sequence promises is order: within a run each step sees the last one's writes, and each run sees
 * the last run's. So every kernel here reads what another invocation of the step before wrote, where a
 * missing barrier shows as an answer from the wrong step, and runs are repeated with no read between.
 */
class DispatchSequenceTest {

    private static final Type.Int I32 = Type.int32();

    /** Not a multiple of any workgroup size, so every step's tail is stopped and the next reads its edge. */
    private static final int N = 1000;

    /** {@code to[i] = from[i] * 3 + 1}: integer, wrapping identically on both backends. */
    private static KernelSpec affine() {
        Buffer from = new Buffer("from", 0, I32);
        Buffer to = new Buffer("to", 1, I32);
        return spec(kernel(new Statement.BufferStore(to, gid(),
                        add(new Expr.Binary(BinaryOp.MUL, new Expr.BufferLoad(from, gid()), i(3)), i(1)))),
                KernelColumn.input("from", 0, I32), KernelColumn.output("to", 1, I32));
    }

    /** {@code to[i] = from[n - 1 - i]}: every invocation reads one the last step's other end wrote. */
    private static KernelSpec reverse() {
        Buffer from = new Buffer("from", 0, I32);
        Buffer to = new Buffer("to", 1, I32);
        Expr mirror = add(new Expr.Binary(BinaryOp.SUB, new Expr.InvocationCount(), gid()), i(-1));
        return spec(kernel(new Statement.BufferStore(to, gid(), new Expr.BufferLoad(from, mirror))),
                KernelColumn.input("from", 0, I32), KernelColumn.output("to", 1, I32));
    }

    /** {@code counter[0] += 1} per invocation. */
    private static KernelSpec count() {
        Buffer counter = new Buffer("counter", 0, I32);
        return spec(kernel(new Statement.AtomicUpdate(AtomicOp.ADD, counter, i(0), i(1))),
                KernelColumn.output("counter", 0, I32).withLength(1));
    }

    private static int[] start() {
        int[] a = new int[N];
        for (int k = 0; k < N; k++) {
            a[k] = k * 7 - 500;
        }
        return a;
    }

    /** The host's answer to {@code runs} runs of [affine a→b, reverse b→c, affine c→a]. */
    private static int[] expected(int runs) {
        int[] a = start();
        for (int r = 0; r < runs; r++) {
            int[] b = new int[N];
            int[] c = new int[N];
            for (int k = 0; k < N; k++) {
                b[k] = a[k] * 3 + 1;
            }
            for (int k = 0; k < N; k++) {
                c[k] = b[N - 1 - k];
            }
            for (int k = 0; k < N; k++) {
                a[k] = c[k] * 3 + 1;
            }
        }
        return a;
    }

    @Test
    void aRunIsItsDispatchesInOrderAndRunsFollowEachOther() {
        try (Accelerator accelerator = new Accelerator()) {
            KernelHandle affine = accelerator.register(affine()).orElseThrow();
            KernelHandle reverse = accelerator.register(reverse()).orElseThrow();
            ResidentBuffer a = accelerator.allocate(I32, N);
            ResidentBuffer b = accelerator.allocate(I32, N);
            ResidentBuffer c = accelerator.allocate(I32, N);
            a.write(start());
            try (DispatchSequence step = accelerator.sequence()
                    .dispatch(affine, List.of(a, b), N)
                    .dispatch(reverse, List.of(b, c), N)
                    .dispatch(affine, List.of(c, a), N)
                    .build()) {
                assertEquals(accelerator.capabilities().gpuAvailable(), step.recorded());
                assertEquals(3, step.dispatches());
                int runs = 40;
                for (int r = 0; r < runs; r++) {
                    step.run();   // no read between: the order across runs is the sequence's to keep
                }
                assertArrayEquals(expected(runs), a.read());
            }
        }
    }

    /** Runs that each add to the same counter, several steps per run: a lost run or step is a lost count. */
    @Test
    void everyStepOfEveryRunHappens() {
        try (Accelerator accelerator = new Accelerator()) {
            KernelHandle count = accelerator.register(count()).orElseThrow();
            ResidentBuffer counter = accelerator.allocate(I32, 1);
            counter.write(new int[1]);
            try (DispatchSequence sequence = accelerator.sequence()
                    .dispatch(count, List.of(counter), N)
                    .dispatch(count, List.of(counter), 1)
                    .dispatch(count, List.of(counter), N)
                    .build()) {
                for (int r = 0; r < 100; r++) {   // beyond the 64 runs that may be in flight at once
                    sequence.run();
                }
            }
            assertEquals(100 * (2 * N + 1), counter.read()[0]);
        }
    }

    /**
     * A step the GPU cannot take — here a kernel whose subgroup size no device offers — makes the sequence run
     * its dispatches one at a time, and the answer and its order are the same.
     */
    @Test
    void aStepThatCannotRunOnTheGpuMakesTheSequenceRunStepByStep() {
        try (Accelerator accelerator = new Accelerator()) {
            assumeGpu(accelerator);
            KernelHandle affine = accelerator.register(affine()).orElseThrow();
            KernelHandle cpuOnly = accelerator.register(
                    new KernelSpec(subgroupTouchingReverse(), reverse().columns(), 128, 128)).orElseThrow();
            assertEquals(KernelHandle.Backend.CPU, cpuOnly.preferredBackend());
            ResidentBuffer a = accelerator.allocate(I32, N);
            ResidentBuffer b = accelerator.allocate(I32, N);
            ResidentBuffer c = accelerator.allocate(I32, N);
            a.write(start());
            try (DispatchSequence step = accelerator.sequence()
                    .dispatch(affine, List.of(a, b), N)
                    .dispatch(cpuOnly, List.of(b, c), N)
                    .dispatch(affine, List.of(c, a), N)
                    .build()) {
                assertFalse(step.recorded());
                for (int r = 0; r < 3; r++) {
                    step.run();
                }
                assertArrayEquals(expected(3), a.read());
            }
        }
    }

    /** A pipeline released after the sequence was recorded: the next run goes step by step, still right. */
    @Test
    void aReleasedPipelineTurnsTheRecordingOff() {
        try (Accelerator accelerator = new Accelerator()) {
            assumeGpu(accelerator);
            KernelHandle affine = accelerator.register(affine()).orElseThrow();
            KernelHandle reverse = accelerator.register(reverse()).orElseThrow();
            ResidentBuffer a = accelerator.allocate(I32, N);
            ResidentBuffer b = accelerator.allocate(I32, N);
            ResidentBuffer c = accelerator.allocate(I32, N);
            a.write(start());
            try (DispatchSequence step = accelerator.sequence()
                    .dispatch(affine, List.of(a, b), N)
                    .dispatch(reverse, List.of(b, c), N)
                    .dispatch(affine, List.of(c, a), N)
                    .build()) {
                step.run();
                assertTrue(step.recorded());
                reverse.close();
                assertFalse(step.recorded());
                step.run();
                assertArrayEquals(expected(2), a.read());
            }
        }
    }

    /** Host buffers, as a machine without a GPU has: the sequence is the dispatches, one by one, in place. */
    @Test
    void hostBuffersRunOnTheCpu() {
        try (Accelerator accelerator = new Accelerator()) {
            KernelHandle affine = accelerator.register(affine()).orElseThrow();
            KernelHandle reverse = accelerator.register(reverse()).orElseThrow();
            ResidentBuffer a = ResidentBuffer.onHost(I32, N);
            ResidentBuffer b = ResidentBuffer.onHost(I32, N);
            ResidentBuffer c = ResidentBuffer.onHost(I32, N);
            a.write(start());
            try (DispatchSequence step = accelerator.sequence()
                    .dispatch(affine, List.of(a, b), N)
                    .dispatch(reverse, List.of(b, c), N)
                    .dispatch(affine, List.of(c, a), N)
                    .build()) {
                assertFalse(step.recorded());
                step.run();
                step.run();
                assertArrayEquals(expected(2), a.read());
            }
        }
    }

    @Test
    void theRefusals() {
        try (Accelerator accelerator = new Accelerator(); Accelerator other = new Accelerator()) {
            KernelHandle affine = accelerator.register(affine()).orElseThrow();
            KernelHandle foreign = other.register(affine()).orElseThrow();
            ResidentBuffer a = accelerator.allocate(I32, N);
            ResidentBuffer b = accelerator.allocate(I32, N);
            assertThrows(IllegalStateException.class, () -> accelerator.sequence().build(), "empty");
            assertThrows(IllegalArgumentException.class,
                    () -> accelerator.sequence().dispatch(affine, List.of(a), N), "one buffer short");
            assertThrows(IllegalArgumentException.class,
                    () -> accelerator.sequence().dispatch(affine, List.of(a, b), N + 1), "buffers too small");
            assertThrows(IllegalArgumentException.class,
                    () -> accelerator.sequence().dispatch(foreign, List.of(a, b), N).build(), "another accelerator");

            DispatchSequence sequence = accelerator.sequence().dispatch(affine, List.of(a, b), N).build();
            b.close();
            assertThrows(IllegalStateException.class, sequence::run, "a closed buffer");
            sequence.close();
            assertThrows(IllegalStateException.class, sequence::run, "a closed sequence");
        }
    }

    // --- harness ------------------------------------------------------------------------------------------

    /** {@link #reverse}, reading the lane so its meaning is its subgroups' and the size the device's to give. */
    private static Function subgroupTouchingReverse() {
        Buffer from = new Buffer("from", 0, I32);
        Buffer to = new Buffer("to", 1, I32);
        Expr mirror = add(new Expr.Binary(BinaryOp.SUB, new Expr.InvocationCount(), gid()), i(-1));
        Expr zeroFromTheLane = new Expr.Binary(BinaryOp.MUL, new Expr.SubgroupInvocationId(), i(0));
        return kernel(new Statement.BufferStore(to, gid(), add(new Expr.BufferLoad(from, mirror), zeroFromTheLane)));
    }

    private static KernelSpec spec(Function kernel, KernelColumn... columns) {
        return new KernelSpec(kernel, List.of(columns));
    }

    private static void assumeGpu(Accelerator accelerator) {
        assumeTrue(accelerator.capabilities().gpuAvailable() || Boolean.getBoolean("supirvast.requireGpu"),
                "no Vulkan device");
    }

    private static Function kernel(Statement... statements) {
        List<Statement> body = new ArrayList<>(List.of(statements));
        body.add(new Statement.ReturnVoid());
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), new Region(body));
    }

    private static Expr gid() {
        return new Expr.InvocationId();
    }

    private static Expr i(long value) {
        return new Expr.ConstInt(I32, value);
    }

    private static Expr add(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.ADD, a, b);
    }
}
