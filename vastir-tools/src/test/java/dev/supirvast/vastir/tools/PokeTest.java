package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.build.Body;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static dev.supirvast.vastir.build.Body.I32;
import static dev.supirvast.vastir.build.Body.add;
import static dev.supirvast.vastir.build.Body.i;
import static dev.supirvast.vastir.build.Body.load;
import static dev.supirvast.vastir.build.Body.lt;
import static dev.supirvast.vastir.build.Body.v;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A poke: what the host fills before every run, written into a readout where it is, with no staging buffer and no
 * submission of its own — so the thread that fills it need not be one allowed to submit.
 */
class PokeTest {

    private static final int N = 1 << 14;
    private static final int WORKGROUP = 64;
    private static final int RUNS = 50;

    private static final Buffer IN = new Buffer("in", 0, I32);
    private static final Buffer OUT = new Buffer("out", 1, I32);

    /** {@code out[k] = in[k] + 1}. */
    private static final Function PLUS_ONE = kernel("plusOne", b -> {
        LocalVar k = b.let("k", new Expr.InvocationId());
        b.when(lt(v(k), new Expr.InvocationCount()), t -> t.store(OUT, v(k), add(load(IN, v(k)), i(1))));
    });

    @Test
    void eachRunSeesWhatWasPokedBeforeIt() {
        try (Accelerator accelerator = new Accelerator()) {
            assumeTrue(accelerator.capabilities().gpuAvailable(), "no Vulkan device");
            KernelHandle plusOne = (KernelHandle) accelerator.register(new KernelSpec(PLUS_ONE, List.of(
                    KernelColumn.input("in", 0, I32).withLength(N),
                    KernelColumn.output("out", 1, I32).withLength(N))).withWorkgroupSize(WORKGROUP));
            ResidentBuffer in = accelerator.allocateReadout(I32, N);
            ResidentBuffer out = accelerator.allocate(I32, N);
            DispatchSequence run = accelerator.sequence().dispatch(plusOne, List.of(in, out), N).build();
            for (int r = 1; r <= RUNS; r++) {
                int[] values = values(r);
                in.poke(values);
                run.run().await();   // awaited before the next poke, which would otherwise race this run's read
                int[] expected = new int[N];
                for (int k = 0; k < N; k++) {
                    expected[k] = values[k] + 1;
                }
                assertArrayEquals(expected, out.read(), "run " + r);
            }
        }
    }

    @Test
    void aBufferThatIsNotAReadoutIsNotPoked() {
        try (Accelerator accelerator = new Accelerator()) {
            assumeTrue(accelerator.capabilities().gpuAvailable(), "no Vulkan device");
            ResidentBuffer local = accelerator.allocate(I32, N);
            assertThrows(IllegalStateException.class, () -> local.poke(new int[N]));
        }
    }

    @Test
    void onTheCpuItIsTheArray() {
        ResidentBuffer host = ResidentBuffer.onHost(I32, 4);
        host.poke(new int[] {1, 2, 3});
        assertArrayEquals(new int[] {1, 2, 3, 0}, host.peek());
    }

    private static int[] values(int run) {
        int[] values = new int[N];
        for (int k = 0; k < N; k++) {
            values[k] = k * 7 + run * 1000;
        }
        return values;
    }

    private static Function kernel(String name, java.util.function.Consumer<Body> writer) {
        Body b = new Body();
        writer.accept(b);
        return new Function(name, new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }
}
