package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.build.Body;
import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.pass.BufferSpec;
import dev.supirvast.vastir.pass.Buffered;
import dev.supirvast.vastir.pass.Pass;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static dev.supirvast.vastir.build.Body.I32;
import static dev.supirvast.vastir.build.Body.eq;
import static dev.supirvast.vastir.build.Body.gt;
import static dev.supirvast.vastir.build.Body.i;
import static dev.supirvast.vastir.build.Body.load;
import static dev.supirvast.vastir.build.Body.lt;
import static dev.supirvast.vastir.build.Body.v;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A readout: a count a kernel writes, read by the host where it is once the run that wrote it is done, with no
 * submission of its own — what a loop deciding whether to run again needs after every run.
 */
class ReadoutTest {

    private static final int N = 1 << 14;
    private static final int WORKGROUP = 64;
    private static final int RUNS = 200;

    private static final Buffer VALUES = new Buffer("values", 0, I32);
    private static final Buffer COUNT = new Buffer("count", 1, I32);

    /** {@code count = [0, run]}, by the first invocation: what the count pass then adds to. */
    private static final Function RESET = kernel("reset", b -> {
        LocalVar k = b.let("k", new Expr.InvocationId());
        b.when(eq(v(k), i(0)), first -> {
            first.store(COUNT, i(0), i(0));
            first.store(COUNT, i(1), load(VALUES, i(0)));
        });
    });

    /** {@code count[0]} = how many values are positive. */
    private static final Function POSITIVE = kernel("positive", b -> {
        LocalVar k = b.let("k", new Expr.InvocationId());
        b.when(lt(v(k), new Expr.InvocationCount()), t -> t.when(gt(load(VALUES, v(k)), i(0)),
                yes -> yes.atomic(AtomicOp.ADD, COUNT, i(0), i(1))));
    });

    private static final class Program implements Buffered {
        final List<Pass> step = List.of(
                new Pass("reset", RESET, List.of(VALUES, COUNT), List.of("values", "count"), 1),
                new Pass("positive", POSITIVE, List.of(VALUES, COUNT), List.of("values", "count"), N));
        private final Map<String, BufferSpec> buffers = new LinkedHashMap<>();

        Program() {
            buffers.put("values", new BufferSpec("values", I32, N));
            buffers.put("count", BufferSpec.readout("count", I32, 2));
        }

        @Override
        public Map<String, BufferSpec> buffers() {
            return buffers;
        }
    }

    @Test
    void theCpuPeeksTheArray() {
        Program program = new Program();
        try (PassRunner runner = PassRunner.cpu(program, WORKGROUP, PassRunner.NO_SUBGROUP)) {
            runner.write("values", values(3));
            runner.run(program.step).await();
            assertArrayEquals(new int[] {positive(3), 3}, runner.peek("count"));
        }
    }

    @Test
    void eachRunIsReadWhereItIsOnceItIsDone() {
        Program program = new Program();
        try (Accelerator accelerator = new Accelerator()) {
            assumeTrue(accelerator.capabilities().gpuAvailable(), "no Vulkan device");
            try (PassRunner runner = PassRunner.gpu(accelerator, program, WORKGROUP, PassRunner.NO_SUBGROUP)) {
                runner.prepare(program.step);
                runner.clear();
                long peeking = 0;
                long reading = 0;
                for (int run = 1; run <= RUNS; run++) {
                    runner.write("values", values(run));
                    long start = System.nanoTime();
                    runner.run(program.step).await();
                    int[] peeked = runner.peek("count");
                    peeking += System.nanoTime() - start;
                    assertArrayEquals(new int[] {positive(run), run}, peeked, "run " + run);

                    start = System.nanoTime();
                    runner.run(program.step);
                    int[] read = runner.read("count");
                    reading += System.nanoTime() - start;
                    assertArrayEquals(peeked, read, "a peek and a read of the same run agree");
                }
                System.out.printf("[readout] %s: run and peek %.1f us, run and read %.1f us%n",
                        accelerator.capabilities().deviceName(), peeking / 1e3 / RUNS, reading / 1e3 / RUNS);
            }
        }
    }

    @Test
    void aBufferThatIsNotAReadoutIsNotPeeked() {
        Program program = new Program();
        try (Accelerator accelerator = new Accelerator()) {
            assumeTrue(accelerator.capabilities().gpuAvailable(), "no Vulkan device");
            try (PassRunner runner = PassRunner.gpu(accelerator, program, WORKGROUP, PassRunner.NO_SUBGROUP)) {
                assertThrows(IllegalStateException.class, () -> runner.peek("values"));
            }
        }
    }

    /** Values whose first is {@code run} and of which a share that changes with it are positive. */
    private static int[] values(int run) {
        int[] values = new int[N];
        values[0] = run;
        for (int k = 1; k < N; k++) {
            values[k] = (k * 31 + run * 17) % 101 - 50;
        }
        return values;
    }

    private static int positive(int run) {
        int count = 0;
        for (int value : values(run)) {
            if (value > 0) {
                count++;
            }
        }
        return count;
    }

    private static Function kernel(String name, java.util.function.Consumer<Body> writer) {
        Body b = new Body();
        writer.accept(b);
        return new Function(name, new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }
}
