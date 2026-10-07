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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A program's kernels and buffers made on workers while the owning thread goes on dispatching, then handed to it.
 *
 * <p>That is what an application sharing its device with a window has to do: registration lowers, validates and has the
 * driver compile, which is too long for the thread that draws, so it happens elsewhere at the same time as that thread's
 * own work on the same device. What is asserted is that nothing is lost or crossed in the hand-over — every handle runs,
 * on the GPU, against the buffers it was given — and that the owning thread's work beside it is undisturbed.
 */
class OffThreadSetupTest {

    private static final Type.Int I32 = Type.int32();
    private static final int N = 1000;
    private static final int WORKERS = 4;

    /** {@code to[i] = from[i] * factor + 1}: a different kernel for each factor, so no two workers lower the same one. */
    private static KernelSpec affine(int factor) {
        Buffer from = new Buffer("from", 0, I32);
        Buffer to = new Buffer("to", 1, I32);
        return new KernelSpec(kernel(new Statement.BufferStore(to, gid(),
                        add(new Expr.Binary(BinaryOp.MUL, new Expr.BufferLoad(from, gid()), i(factor)), i(1)))),
                List.of(KernelColumn.input("from", 0, I32), KernelColumn.output("to", 1, I32)));
    }

    /** {@code counter[0] += 1} per invocation: the owning thread's work while the workers set up. */
    private static KernelSpec count() {
        Buffer counter = new Buffer("counter", 0, I32);
        return new KernelSpec(kernel(new Statement.AtomicUpdate(AtomicOp.ADD, counter, i(0), i(1))),
                List.of(KernelColumn.output("counter", 0, I32).withLength(1)));
    }

    /** What a worker hands over: one kernel and the two buffers it runs between. */
    private record Piece(int factor, KernelHandle kernel, ResidentBuffer from, ResidentBuffer to) {}

    @Test
    void kernelsAndBuffersMadeOnWorkersRunOnTheOwningThread() throws Exception {
        assumeTrue(GpuContext.isAvailable(), "no GPU here");
        try (GpuContext context = GpuContext.open();
             Accelerator busy = Accelerator.on(context);
             Accelerator program = Accelerator.on(context)) {
            // The owning thread's own work on the device, kept going the whole time the workers set up.
            KernelHandle count = busy.register(count()).orElseThrow();
            ResidentBuffer counter = busy.allocate(I32, 1);
            counter.write(new int[1]);

            ExecutorService workers = Executors.newFixedThreadPool(WORKERS);
            AtomicBoolean ready = new AtomicBoolean();
            try {
                List<Future<Piece>> pieces = new ArrayList<>();
                for (int w = 0; w < WORKERS; w++) {
                    int factor = w + 2;
                    pieces.add(workers.submit(() -> new Piece(factor,
                            program.register(affine(factor)).orElseThrow(),
                            program.allocate(I32, N), program.allocate(I32, N))));
                }
                workers.submit(() -> {
                    for (Future<Piece> piece : pieces) {
                        piece.get();
                    }
                    ready.set(true);
                    return null;
                });
                int dispatches = 0;
                while (!ready.get()) {
                    count.dispatch(List.of(counter), N);
                    dispatches++;
                }
                assertEquals(dispatches * N, counter.read()[0], "the owning thread's work beside the setup is whole");

                List<Piece> handed = new ArrayList<>();
                for (Future<Piece> piece : pieces) {
                    handed.add(piece.get());
                }
                int[] start = new int[N];
                for (int k = 0; k < N; k++) {
                    start[k] = k * 7 - 500;
                }
                DispatchSequence.Builder builder = program.sequence();
                for (Piece piece : handed) {
                    assertEquals(KernelHandle.Backend.GPU, piece.kernel().preferredBackend(),
                            "a pipeline built on a worker is there for the owning thread");
                    piece.from().write(start);
                    builder.dispatch(piece.kernel(), List.of(piece.from(), piece.to()), N);
                }
                try (DispatchSequence sequence = builder.build()) {
                    assertTrue(sequence.recorded(), "every step of it is on the GPU");
                    sequence.run();
                }
                for (Piece piece : handed) {
                    int[] expected = new int[N];
                    for (int k = 0; k < N; k++) {
                        expected[k] = start[k] * piece.factor() + 1;
                    }
                    assertArrayEquals(expected, piece.to().read(), "factor " + piece.factor());
                }
            } finally {
                workers.shutdownNow();
            }
        }
    }

    /** Several buffers holding data, cleared in one go after a dispatch that wrote one of them: all zero, after it. */
    @Test
    void clearZeroesEveryBufferAfterTheWorkBeforeIt() {
        try (Accelerator accelerator = new Accelerator()) {
            KernelHandle affine = accelerator.register(affine(3)).orElseThrow();
            ResidentBuffer from = accelerator.allocate(I32, N);
            ResidentBuffer to = accelerator.allocate(I32, N);
            ResidentBuffer other = accelerator.allocate(I32, 7);
            int[] ones = new int[N];
            java.util.Arrays.fill(ones, 1);
            from.write(ones);
            other.write(new int[] {1, 2, 3, 4, 5, 6, 7});
            affine.dispatch(List.of(from, to), N);   // writes `to`, and is not waited for

            accelerator.clear(List.of(from, to, other));

            assertArrayEquals(new int[N], from.read());
            assertArrayEquals(new int[N], to.read(), "the clear comes after the dispatch that wrote it");
            assertArrayEquals(new int[7], other.read());
        }
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
