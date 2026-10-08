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

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static dev.supirvast.vastir.build.Body.I32;
import static dev.supirvast.vastir.build.Body.add;
import static dev.supirvast.vastir.build.Body.i;
import static dev.supirvast.vastir.build.Body.load;
import static dev.supirvast.vastir.build.Body.lt;
import static dev.supirvast.vastir.build.Body.mul;
import static dev.supirvast.vastir.build.Body.sub;
import static dev.supirvast.vastir.build.Body.v;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A program of named buffers, run on each backend: the same answer from both, a list run twice running twice,
 * and one kernel bound to different buffers in different passes.
 */
class PassRunnerTest {

    /** Not a multiple of the workgroup, so each pass's tail is stopped by its bounds check. */
    private static final int N = 1000;
    private static final int WORKGROUP = 64;

    private static final Buffer FROM = new Buffer("from", 0, I32);
    private static final Buffer TO = new Buffer("to", 1, I32);
    private static final List<Buffer> BINDINGS = List.of(FROM, TO);

    /** {@code to[i] = from[i] * 3 + 1}. */
    private static final Function AFFINE = kernel("affine", b -> {
        LocalVar k = b.let("k", new Expr.InvocationId());
        b.when(lt(v(k), new Expr.InvocationCount()), t -> t.store(TO, v(k), add(mul(load(FROM, v(k)), i(3)), i(1))));
    });

    /** {@code to[i] = from[n − 1 − i]}: every invocation reads what the other end of the last pass wrote. */
    private static final Function REVERSE = kernel("reverse", b -> {
        LocalVar k = b.let("k", new Expr.InvocationId());
        b.when(lt(v(k), new Expr.InvocationCount()), t ->
                t.store(TO, v(k), load(FROM, sub(sub(new Expr.InvocationCount(), v(k)), i(1)))));
    });

    /** a → b by affine, b → c reversed, c → a by affine: the second affine is the first kernel on other buffers. */
    private static final class Program implements Buffered {
        final List<Pass> step = List.of(
                new Pass("affine", AFFINE, BINDINGS, List.of("a", "b"), N),
                new Pass("reverse", REVERSE, BINDINGS, List.of("b", "c"), N),
                new Pass("affine again", AFFINE, BINDINGS, List.of("c", "a"), N));
        private final Map<String, BufferSpec> buffers = new LinkedHashMap<>();

        Program() {
            for (String name : List.of("a", "b", "c")) {
                buffers.put(name, new BufferSpec(name, I32, N));
            }
        }

        @Override
        public Map<String, BufferSpec> buffers() {
            return buffers;
        }
    }

    @Test
    void theCpuRunsTheProgram() {
        Program program = new Program();
        try (PassRunner runner = PassRunner.cpu(program, WORKGROUP, PassRunner.NO_SUBGROUP)) {
            check(runner, program);
        }
    }

    @Test
    void theGpuRunsTheProgram() {
        Program program = new Program();
        try (Accelerator accelerator = new Accelerator()) {
            assumeTrue(accelerator.capabilities().gpuAvailable(), "no Vulkan device");
            try (PassRunner runner = PassRunner.gpu(accelerator, program, WORKGROUP, PassRunner.NO_SUBGROUP)) {
                runner.prepare(program.step);
                runner.clear();
                check(runner, program);
            }
        }
    }

    @Test
    void aNameTheProgramDoesNotHaveIsRefused() {
        try (PassRunner runner = PassRunner.cpu(new Program(), WORKGROUP, PassRunner.NO_SUBGROUP)) {
            assertThrows(IllegalArgumentException.class, () -> runner.read("d"));
        }
    }

    private static void check(PassRunner runner, Program program) {
        int[] a = new int[N];
        for (int k = 0; k < N; k++) {
            a[k] = k * 7 - 500;
        }
        runner.write("a", a);
        runner.run(program.step);
        runner.run(program.step);
        assertArrayEquals(expected(a, 2), runner.read("a"));
    }

    /** The host's answer to {@code runs} runs of the program. */
    private static int[] expected(int[] start, int runs) {
        int[] a = start.clone();
        for (int r = 0; r < runs; r++) {
            int[] c = new int[N];
            for (int k = 0; k < N; k++) {
                c[k] = a[N - 1 - k] * 3 + 1;
            }
            for (int k = 0; k < N; k++) {
                a[k] = c[k] * 3 + 1;
            }
        }
        return a;
    }

    private static Function kernel(String name, java.util.function.Consumer<Body> writer) {
        Body b = new Body();
        writer.accept(b);
        return new Function(name, new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }
}
