package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.build.Body;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.pass.BufferSpec;
import dev.supirvast.vastir.pass.Buffered;
import dev.supirvast.vastir.pass.Pass;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static dev.supirvast.vastir.build.Body.I32;
import static dev.supirvast.vastir.build.Body.add;
import static dev.supirvast.vastir.build.Body.i;
import static dev.supirvast.vastir.build.Body.load;
import static dev.supirvast.vastir.build.Body.lt;
import static dev.supirvast.vastir.build.Body.mod;
import static dev.supirvast.vastir.build.Body.mul;
import static dev.supirvast.vastir.build.Body.v;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Work submitted without waiting says when it has finished: a run's {@link Completion} is done once the run is,
 * and so are the completions of every run before it.
 */
class CompletionTest {

    private static final int N = 1 << 16;
    private static final int LOOPS = 2000;
    private static final int WORKGROUP = 64;

    private static final Buffer DATA = new Buffer("data", 0, I32);

    /** Every word stirred {@link #LOOPS} times: long enough on any GPU that a run is not over by the time it returns. */
    private static final Function STIR = kernel(b -> {
        LocalVar k = b.let("k", new Expr.InvocationId());
        b.when(lt(v(k), new Expr.InvocationCount()), t -> {
            LocalVar x = t.let("x", load(DATA, v(k)));
            LocalVar n = t.let("n", i(0));
            t.loop(lt(v(n), i(LOOPS)), body -> {
                body.set(x, mod(add(mul(v(x), i(31)), i(7)), i(1_000_003)));
                body.set(n, add(v(n), i(1)));
            });
            t.store(DATA, v(k), v(x));
        });
    });

    private static final class Program implements Buffered {
        final List<Pass> step = List.of(new Pass("stir", STIR, List.of(DATA), List.of("data"), N));

        @Override
        public Map<String, BufferSpec> buffers() {
            return Map.of("data", new BufferSpec("data", I32, N));
        }
    }

    @Test
    void aRunIsDoneOnceItHasFinishedAndSoAreTheRunsBeforeIt() {
        Program program = new Program();
        try (Accelerator accelerator = new Accelerator()) {
            assumeTrue(accelerator.capabilities().gpuAvailable(), "no Vulkan device");
            try (PassRunner runner = PassRunner.gpu(accelerator, program, WORKGROUP, PassRunner.NO_SUBGROUP)) {
                runner.prepare(program.step);
                runner.clear();
                Completion first = runner.run(program.step);
                Completion second = runner.run(program.step);
                Completion third = runner.run(program.step);
                third.await();
                assertTrue(third.done(), "awaited, and not done");
                assertTrue(first.done() && second.done(), "a later run finished before an earlier one");
                assertArrayEquals(expected(3), runner.read("data"));
            }
        }
    }

    /** Polling, never waiting, still sees a run finish: what a worker checking once a frame relies on. */
    @Test
    void pollingAloneSeesARunFinish() {
        Program program = new Program();
        try (Accelerator accelerator = new Accelerator()) {
            assumeTrue(accelerator.capabilities().gpuAvailable(), "no Vulkan device");
            try (PassRunner runner = PassRunner.gpu(accelerator, program, WORKGROUP, PassRunner.NO_SUBGROUP)) {
                runner.prepare(program.step);
                runner.clear();
                Completion run = runner.run(program.step);
                long deadline = System.nanoTime() + 10_000_000_000L;
                while (!run.done()) {
                    assertTrue(System.nanoTime() < deadline, "a run did not finish in ten seconds of polling");
                    Thread.onSpinWait();
                }
                assertArrayEquals(expected(1), runner.read("data"));
            }
        }
    }

    @Test
    void workOnTheCpuIsDoneWhenItReturns() {
        Program program = new Program();
        try (PassRunner runner = PassRunner.cpu(program, WORKGROUP, PassRunner.NO_SUBGROUP)) {
            assertSame(Completion.DONE, runner.run(program.step));
            assertTrue(Completion.DONE.done());
        }
    }

    /** An accelerator may choose its own device, whatever the system property says. */
    @Test
    void anAcceleratorChoosesItsOwnDevice() {
        int opened = 0;
        for (String selector : List.of("discrete", "integrated")) {
            Accelerator accelerator;
            try {
                accelerator = Accelerator.onDevice(selector);
            } catch (IllegalStateException none) {
                continue;   // this machine has no such device, or no Vulkan at all
            }
            try (accelerator) {
                assertEquals(selector, accelerator.capabilities().deviceType(), "asked for " + selector);
                System.out.println("[completion] " + selector + ": " + accelerator.capabilities().deviceName());
                opened++;
                Program program = new Program();
                try (PassRunner runner = PassRunner.gpu(accelerator, program, WORKGROUP, PassRunner.NO_SUBGROUP)) {
                    runner.clear();
                    runner.run(program.step).await();
                    assertArrayEquals(expected(1), runner.read("data"));
                }
            }
        }
        assumeTrue(opened > 0, "no discrete or integrated GPU");
    }

    /**
     * A run says how long it took on the GPU, on each GPU there is: a time, once it is done, no longer than the
     * wall clock saw, and four times as long for four times the work.
     */
    @Test
    void aRunSaysHowLongItTookOnTheGpu() {
        int timed = 0;
        for (String selector : List.of("discrete", "integrated")) {
            Accelerator accelerator;
            try {
                accelerator = Accelerator.onDevice(selector);
            } catch (IllegalStateException none) {
                continue;
            }
            try (accelerator) {
                Program program = new Program();
                List<Pass> four = List.of(program.step.get(0), program.step.get(0), program.step.get(0),
                        program.step.get(0));
                try (PassRunner runner = PassRunner.gpu(accelerator, program, WORKGROUP, PassRunner.NO_SUBGROUP)) {
                    runner.prepare(program.step);
                    runner.clear();
                    runner.run(program.step).await();   // first use, out of the comparison
                    long start = System.nanoTime();
                    Completion once = runner.run(program.step);
                    once.await();
                    long wall = System.nanoTime() - start;
                    Completion fourTimes = runner.run(four);
                    fourTimes.await();
                    long one = once.gpuNanos().orElseThrow();
                    long four4 = fourTimes.gpuNanos().orElseThrow();
                    System.out.printf("[completion] %s: one run %.3f ms on the GPU, %.3f ms on the wall; four %.3f ms%n",
                            accelerator.capabilities().deviceName(), one / 1e6, wall / 1e6, four4 / 1e6);
                    assertTrue(one > 0 && one <= wall, "the GPU's time is not within the wall's: " + one + " ns");
                    double ratio = (double) four4 / one;
                    assertTrue(ratio > 2.5 && ratio < 6, "four times the work took " + ratio + " times as long");
                    timed++;
                }
            }
        }
        assumeTrue(timed > 0, "no GPU");
        assertTrue(Completion.DONE.gpuNanos().isEmpty(), "work on the CPU has no GPU time");
    }

    /** The host's answer to {@code runs} runs over a buffer of zeros. */
    private static int[] expected(int runs) {
        int[] data = new int[N];
        for (int r = 0; r < runs; r++) {
            for (int k = 0; k < N; k++) {
                int x = data[k];
                for (int n = 0; n < LOOPS; n++) {
                    x = (x * 31 + 7) % 1_000_003;
                }
                data[k] = x;
            }
        }
        return data;
    }

    private static Function kernel(java.util.function.Consumer<Body> writer) {
        Body b = new Body();
        writer.accept(b);
        return new Function("stir", new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }
}
