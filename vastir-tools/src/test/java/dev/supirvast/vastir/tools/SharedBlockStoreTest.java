package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.build.Body;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.pass.BufferSpec;
import dev.supirvast.vastir.pass.Buffered;
import dev.supirvast.vastir.pass.Pass;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static dev.supirvast.vastir.build.Body.F32;
import static dev.supirvast.vastir.build.Body.f;
import static dev.supirvast.vastir.build.Body.i;
import static dev.supirvast.vastir.build.Body.load;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A store, then a load of the same word, while a read-only buffer of the same element type has been loaded
 * first: the load must see the store.
 *
 * <p>On an NVIDIA GeForce RTX 5070 Ti Laptop GPU, driver 610.78, it does not. The kernel
 *
 * <pre>
 *   ignored = r[0];      // r is only loaded, so it is decorated NonWritable
 *   x[0] = 7;
 *   x[0] = x[0];
 * </pre>
 *
 * <p>leaves {@code x[0]} as it was: the load reads memory from before the store, as though {@code x} could not
 * be written either. The CPU backend gives 7. The module passes {@code spirv-val}. The two buffers are two
 * variables of the one {@code Block} struct type {@code CoreToSpirv} declares per element type, and
 * {@code NonWritable} is on {@code r}'s variable, as SPIR-V allows; the driver acts as if it were on the type.
 * glslang, which gives every block its own struct type, never emits this shape.
 *
 * <p>Each of these makes it pass, and so is what the case needs: the read-only buffer loaded after the store
 * rather than before; {@code r} given another element type, so another struct type; {@code r} stored to as
 * well, so nothing is {@code NonWritable}; and, tried as a change to the lowering, a struct type of its own for
 * each {@code NonWritable} buffer. It was found in vexelray-sim-rigid's walls pass, where a sphere's position
 * stored in one branch and reloaded after it came back as the position before, and a sphere fell through the
 * floor.
 */
class SharedBlockStoreTest {

    /** See {@link DifferentialHarnessTest}: same flag, same reason. */
    private static final boolean REQUIRE_GPU = Boolean.getBoolean("supirvast.requireGpu");

    private static final Buffer X = new Buffer("x", 0, F32);
    private static final Buffer R = new Buffer("r", 1, F32);
    private static final List<Buffer> BINDINGS = List.of(X, R);

    private static final float BEFORE = 0.5f;
    private static final float STORED = 7f;

    /** {@code r[0]} into a local the kernel never reads, before or after the store to {@code x}. */
    private static Function kernel(boolean readOnlyFirst) {
        Body b = new Body();
        if (readOnlyFirst) {
            b.let("ignored", load(R, i(0)));
        }
        b.store(X, i(0), f(STORED));
        if (!readOnlyFirst) {
            b.let("ignored", load(R, i(0)));
        }
        b.store(X, i(0), load(X, i(0)));
        return new Function("sharedBlockStore", new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }

    private static final class Program implements Buffered {
        private final Map<String, BufferSpec> buffers = new LinkedHashMap<>();

        Program() {
            buffers.put("x", new BufferSpec("x", F32, 1));
            buffers.put("r", new BufferSpec("r", F32, 1));
        }

        @Override
        public Map<String, BufferSpec> buffers() {
            return buffers;
        }
    }

    private static float run(PassRunner runner, Function kernel) {
        List<Pass> passes = List.of(new Pass("sharedBlockStore", kernel, BINDINGS, List.of("x", "r"), 1));
        runner.prepare(passes);
        runner.clear();
        runner.write("x", new float[] {BEFORE});
        runner.write("r", new float[] {0.1f});
        runner.run(passes);
        runner.finish();
        return runner.floats("x")[0];
    }

    private static float onGpu(Function kernel) {
        Program program = new Program();
        try (Accelerator accelerator = new Accelerator()) {
            if (!accelerator.capabilities().gpuAvailable()) {
                if (REQUIRE_GPU) {
                    fail("-Dsupirvast.requireGpu=true but no Vulkan device was available");
                }
                assumeTrue(false, "no Vulkan device");
            }
            try (PassRunner runner = PassRunner.gpu(accelerator, program, 1, PassRunner.NO_SUBGROUP)) {
                return run(runner, kernel);
            }
        }
    }

    private static float onCpu(Function kernel) {
        try (PassRunner runner = PassRunner.cpu(new Program(), 1, PassRunner.NO_SUBGROUP)) {
            return run(runner, kernel);
        }
    }

    @Test
    void theCpuSeesTheStore() {
        assertEquals(STORED, onCpu(kernel(true)));
    }

    /** The failing case. */
    @Test
    void aLoadAfterAStoreSeesItWhenAReadOnlyBufferOfTheSameTypeWasLoadedFirst() {
        assertEquals(STORED, onGpu(kernel(true)),
                "the GPU's load of x read memory from before its store: a NonWritable buffer of the same "
                        + "Block struct type was loaded first");
    }

    /** The same statements with the read-only load moved after the store, which the GPU runs right. */
    @Test
    void aLoadAfterAStoreSeesItWhenTheReadOnlyBufferIsLoadedAfter() {
        assertEquals(STORED, onGpu(kernel(false)));
    }
}
