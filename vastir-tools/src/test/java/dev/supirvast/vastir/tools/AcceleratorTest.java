package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.lower.SpirvTarget;
import dev.supirvast.vastir.spirv.Capability;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The orchestration front door, exercised end-to-end. One {@code core} kernel is described as a
 * {@link KernelSpec}, {@link Accelerator#register registered} (validate → lower → {@code spirv-val} → preload),
 * then {@link KernelHandle#run run} (auto backend, CPU fallback) and {@link KernelHandle#verify verified}
 * (CPU==GPU on demand) — all without the caller touching SPIR-V, Vulkan, or the lowering passes. Other cases
 * show a {@link Rejection} witness and pipeline reuse across many runs against a held GPU context.
 */
class AcceleratorTest {

    private static final Buffer OUT = new Buffer("out", 0);
    private static final Buffer A = new Buffer("a", 1);
    private static final Buffer B = new Buffer("b", 2);

    /** {@code out[gid] = a[gid] + b[gid];} */
    private static Function vectorAdd() {
        Expr gid = new Expr.InvocationId();
        Region body = Region.of(
                new Statement.BufferStore(OUT, gid,
                        new Expr.Binary(BinaryOp.ADD,
                                new Expr.BufferLoad(A, new Expr.InvocationId()),
                                new Expr.BufferLoad(B, new Expr.InvocationId()))),
                new Statement.ReturnVoid());
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
    }

    private static KernelSpec vectorAddSpec() {
        return new KernelSpec(vectorAdd(), List.of(
                KernelColumn.output("out", 0),
                KernelColumn.input("a", 1),
                KernelColumn.input("b", 2)));
    }

    /** {@code out[gid] = a[gid] * 2;} — a second, distinct kernel to coexist in one context. */
    private static KernelSpec doubleSpec() {
        Expr gid = new Expr.InvocationId();
        Region body = Region.of(
                new Statement.BufferStore(OUT, gid,
                        new Expr.Binary(BinaryOp.MUL,
                                new Expr.BufferLoad(A, new Expr.InvocationId()),
                                new Expr.ConstInt(Type.int32(), 2))),
                new Statement.ReturnVoid());
        Function kernel = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new KernelSpec(kernel, List.of(KernelColumn.output("out", 0), KernelColumn.input("a", 1)));
    }

    @Test
    void registersRunsAndVerifiesThroughTheFacade() {
        try (Accelerator accelerator = new Accelerator()) {
            KernelHandle kernel = accelerator.register(vectorAddSpec()).orElseThrow();

            int n = 8;
            int[] a = {0, 1, 2, 3, 4, 5, 6, 7};
            int[] b = {10, 20, 30, 40, 50, 60, 70, 80};
            int[] expected = new int[n];
            for (int i = 0; i < n; i++) {
                expected[i] = a[i] + b[i];
            }

            int[][] columns = {new int[n], a.clone(), b.clone()};
            int[][] result = kernel.run(columns, n);
            assertArrayEquals(expected, result[0], "vector add via the orchestrator");
            assertArrayEquals(new int[n], columns[0], "run must not mutate the caller's columns");

            KernelHandle.VerificationResult verification = kernel.verify(columns, n);
            if (accelerator.capabilities().gpuAvailable()) {
                assertEquals(KernelHandle.Backend.GPU, kernel.preferredBackend());
                assertTrue(verification.verified() && verification.matches(),
                        () -> "CPU and GPU disagreed: " + verification.detail());
            } else {
                assertFalse(verification.verified(), "no GPU here, so equivalence is reported skipped");
            }
        }
    }

    @Test
    void preloadsPipelinesAndReusesThemAcrossRuns() {
        int n = 8;
        int[] a = {0, 1, 2, 3, 4, 5, 6, 7};
        int[] b = {10, 20, 30, 40, 50, 60, 70, 80};
        int[] sum = new int[n];
        int[] doubled = new int[n];
        for (int i = 0; i < n; i++) {
            sum[i] = a[i] + b[i];
            doubled[i] = a[i] * 2;
        }

        try (Accelerator accelerator = new Accelerator()) {
            // Two distinct kernels preloaded into one held context.
            KernelHandle add = accelerator.register(vectorAddSpec()).orElseThrow();
            KernelHandle twice = accelerator.register(doubleSpec()).orElseThrow();

            // Each pipeline is built once and dispatched against repeatedly — every run must be correct.
            for (int round = 0; round < 3; round++) {
                assertArrayEquals(sum, add.run(new int[][] {new int[n], a.clone(), b.clone()}, n)[0],
                        "vector-add run " + round);
                assertArrayEquals(doubled, twice.run(new int[][] {new int[n], a.clone()}, n)[0],
                        "double run " + round);
            }
        }
    }

    @Test
    void releaseFreesOnePipelineButLeavesTheKernelRunnableOnCpu() {
        int n = 8;
        int[] a = {0, 1, 2, 3, 4, 5, 6, 7};
        int[] b = {10, 20, 30, 40, 50, 60, 70, 80};
        int[] sum = new int[n];
        for (int i = 0; i < n; i++) {
            sum[i] = a[i] + b[i];
        }

        try (Accelerator accelerator = new Accelerator()) {
            KernelHandle add = accelerator.register(vectorAddSpec()).orElseThrow();
            boolean onGpu = accelerator.capabilities().gpuAvailable();
            assertEquals(onGpu, accelerator.hasPipeline(add), "a resident pipeline exists iff a GPU preloaded it");

            // Per-handle release reclaims just this kernel's pipeline — true only when one was held.
            assertEquals(onGpu, accelerator.release(add), "release frees a pipeline only when one was held");
            assertFalse(accelerator.hasPipeline(add), "after release no resident pipeline remains for this handle");
            assertFalse(accelerator.release(add), "release is idempotent — nothing left to free");
            add.close();  // the handle-side alias is also a safe no-op once released

            // Degraded, not dead: a released kernel still computes, via the proven-equivalent CPU path.
            assertEquals(KernelHandle.Backend.CPU, add.preferredBackend(), "no pipeline ⇒ run falls back to CPU");
            assertArrayEquals(sum, add.run(new int[][] {new int[n], a.clone(), b.clone()}, n)[0],
                    "a released kernel still runs correctly on the CPU fallback");

            // Re-registering restores a resident GPU pipeline where a device is present.
            KernelHandle again = accelerator.register(vectorAddSpec()).orElseThrow();
            assertEquals(onGpu, accelerator.hasPipeline(again), "re-register rebuilds the pipeline on a GPU");
        }
    }

    @Test
    void submitAsyncRunsTwoKernelsConcurrentlyAndAwaitsBoth() {
        int n = 8;
        int[] a = {0, 1, 2, 3, 4, 5, 6, 7};
        int[] b = {10, 20, 30, 40, 50, 60, 70, 80};
        int[] sum = new int[n];
        int[] doubled = new int[n];
        for (int i = 0; i < n; i++) {
            sum[i] = a[i] + b[i];
            doubled[i] = a[i] * 2;
        }

        try (Accelerator accelerator = new Accelerator()) {
            assumeTrue(accelerator.capabilities().gpuAvailable(), "submitAsync is the GPU concurrency path");
            KernelHandle add = accelerator.register(vectorAddSpec()).orElseThrow();
            KernelHandle twice = accelerator.register(doubleSpec()).orElseThrow();

            // Launch both without blocking — they are now in flight together, on (round-robined) queues.
            Submission addRun = add.submitAsync(new int[][] {new int[n], a.clone(), b.clone()}, n);
            Submission twiceRun = twice.submitAsync(new int[][] {new int[n], a.clone()}, n);

            // The SAME pipeline can also have several dispatches in flight at once — each submission gets
            // its own descriptor set and buffers, so a second submitAsync on `add` is fine (not corrupt).
            Submission addRun2 = add.submitAsync(new int[][] {new int[n], b.clone(), a.clone()}, n);

            // Await all three; results are correct regardless of completion order.
            assertArrayEquals(sum, add.await(addRun)[0], "concurrent vector-add");
            assertArrayEquals(doubled, twice.await(twiceRun)[0], "concurrent double");
            assertArrayEquals(sum, add.await(addRun2)[0], "a second concurrent dispatch of the same pipeline");
        }
    }

    @Test
    void submitAsyncIsRejectedWithoutAGpu() {
        try (Accelerator accelerator = new Accelerator()) {
            assumeFalse(accelerator.capabilities().gpuAvailable(), "this checks the no-GPU guard");
            KernelHandle add = accelerator.register(vectorAddSpec()).orElseThrow();
            assertThrows(IllegalStateException.class,
                    () -> add.submitAsync(new int[][] {new int[8], new int[8], new int[8]}, 8),
                    "submitAsync must refuse the CPU-only path and point at run()");
        }
    }

    @Test
    void runsFloatColumnsThroughTheFacade() {
        Type.Float f32 = Type.float32();
        Buffer out = new Buffer("out", 0, f32);
        Buffer a = new Buffer("a", 1, f32);
        Expr gid = new Expr.InvocationId();
        Function kernel = new Function("main", new Type.FunctionType(Type.VOID, List.of()),
                Region.of(
                        new Statement.BufferStore(out, gid, new Expr.Binary(BinaryOp.MUL,
                                new Expr.BufferLoad(a, new Expr.InvocationId()), new Expr.ConstFloat(f32, 1.5))),
                        new Statement.ReturnVoid()));
        KernelSpec spec = new KernelSpec(kernel, List.of(
                KernelColumn.output("out", 0, f32), KernelColumn.input("a", 1, f32)));

        try (Accelerator accelerator = new Accelerator()) {
            KernelHandle handle = accelerator.register(spec).orElseThrow();
            assertEquals(f32, handle.abi().get(0).type(), "the ABI is self-describing: column 0 is f32");

            // Columns are carried as raw 32-bit words; the caller packs/unpacks float bits (the marshalling job).
            int n = 6;
            float[] a32 = {0.0f, 1.1f, 2.5f, 3.3f, 4.0f, 5.7f};
            int[] aBits = new int[n];
            int[] expectedBits = new int[n];
            for (int i = 0; i < n; i++) {
                aBits[i] = Float.floatToRawIntBits(a32[i]);
                expectedBits[i] = Float.floatToRawIntBits(a32[i] * 1.5f);
            }

            int[][] columns = {new int[n], aBits.clone()};
            int[][] result = handle.run(columns, n);
            assertArrayEquals(expectedBits, result[0], "f32 map via the orchestrator (raw bits)");

            KernelHandle.VerificationResult verification = handle.verify(columns, n);
            if (accelerator.capabilities().gpuAvailable()) {
                assertTrue(verification.verified() && verification.matches(),
                        () -> "CPU and GPU disagreed: " + verification.detail());
            }
        }
    }

    /** {@code out[gid] = (int)((long)a[gid] * (long)a[gid]);} — uses i64, so it requires the Int64 capability. */
    private static KernelSpec int64Spec() {
        Buffer out = new Buffer("out", 0);
        Buffer a = new Buffer("a", 1);
        Expr gid = new Expr.InvocationId();
        Expr wide = new Expr.Convert(new Expr.BufferLoad(a, gid), Type.int64());
        Expr square = new Expr.Convert(new Expr.Binary(BinaryOp.MUL, wide, wide), Type.int32());
        Function kernel = new Function("main", new Type.FunctionType(Type.VOID, List.of()),
                Region.of(new Statement.BufferStore(out, gid, square), new Statement.ReturnVoid()));
        return new KernelSpec(kernel, List.of(KernelColumn.output("out", 0), KernelColumn.input("a", 1)));
    }

    @Test
    void derivesCapabilitiesFromTheDeviceAndPreloadsWithinThem() {
        try (Accelerator accelerator = new Accelerator()) {
            Accelerator.Capabilities caps = accelerator.capabilities();
            assumeTrue(caps.gpuAvailable(), "needs a GPU");
            assertTrue(caps.deviceCapabilities().contains(Capability.Shader), "every device supports Shader");
            assumeTrue(caps.deviceCapabilities().contains(Capability.Int64), "needs shaderInt64 for this check");

            // Int64 is in the device profile, so this kernel preloads and runs on the GPU (features enabled).
            KernelHandle kernel = accelerator.register(int64Spec()).orElseThrow();
            assertEquals(KernelHandle.Backend.GPU, kernel.preferredBackend());

            int n = 4;
            int[] a = {2, 3, 100_000, 5};
            int[] expected = new int[n];
            for (int i = 0; i < n; i++) {
                expected[i] = (int) ((long) a[i] * (long) a[i]);
            }
            assertArrayEquals(expected, kernel.run(new int[][] {new int[n], a.clone()}, n)[0],
                    "i64 square runs on the GPU with shaderInt64 enabled");
        }
    }

    @Test
    void runs64BitAndHeterogeneousColumns() {
        Type.Int i64 = Type.int64();
        Buffer out = new Buffer("out", 0, i64);
        Buffer a = new Buffer("a", 1, Type.int32());
        Buffer b = new Buffer("b", 2, i64);
        Expr gid = new Expr.InvocationId();
        // out[gid] = (i64)a[gid] + b[gid] — a struct {a:i32, b:i64} stream as SoA, with a 64-bit output column.
        Expr sum = new Expr.Binary(BinaryOp.ADD,
                new Expr.Convert(new Expr.BufferLoad(a, gid), i64),
                new Expr.BufferLoad(b, new Expr.InvocationId()));
        Function kernel = new Function("main", new Type.FunctionType(Type.VOID, List.of()),
                Region.of(new Statement.BufferStore(out, gid, sum), new Statement.ReturnVoid()));
        KernelSpec spec = new KernelSpec(kernel, List.of(
                KernelColumn.output("out", 0, i64),
                KernelColumn.input("a", 1, Type.int32()),
                KernelColumn.input("b", 2, i64)));

        int n = 6;
        int[] aVals = {1, 2, 3, -1, 1_000_000, -5};
        long[] bVals = {10L, 20L, 5_000_000_000L, 7L, -3L, Long.MAX_VALUE - 1};
        long[] expected = new long[n];
        for (int i = 0; i < n; i++) {
            expected[i] = (long) aVals[i] + bVals[i];
        }

        try (Accelerator accelerator = new Accelerator()) {
            KernelHandle handle = accelerator.register(spec).orElseThrow();
            int[][] columns = {new int[2 * n], aVals.clone(), packLongs(bVals)};
            int[][] result = handle.run(columns, n);
            assertArrayEquals(expected, unpackLongs(result[0]), "i64 = (i64)i32 + i64 over SoA columns");

            KernelHandle.VerificationResult verification = handle.verify(columns, n);
            if (accelerator.capabilities().gpuAvailable()) {
                assertTrue(verification.verified() && verification.matches(),
                        () -> "CPU and GPU disagreed: " + verification.detail());
            }
        }
    }

    /** 64-bit columns ride the int[] wire as two words per element (low word first). */
    private static int[] packLongs(long[] values) {
        int[] words = new int[values.length * 2];
        for (int i = 0; i < values.length; i++) {
            words[2 * i] = (int) values[i];
            words[2 * i + 1] = (int) (values[i] >>> 32);
        }
        return words;
    }

    private static long[] unpackLongs(int[] words) {
        long[] values = new long[words.length / 2];
        for (int i = 0; i < values.length; i++) {
            values[i] = (words[2 * i] & 0xFFFFFFFFL) | ((long) words[2 * i + 1] << 32);
        }
        return values;
    }

    @Test
    void refusesACapabilityOutsideTheBudget() {
        // A budget that allows no optional capabilities — Int64 is outside it, so the i64 kernel is refused.
        try (Accelerator restricted = new Accelerator(SpirvTarget.restrictedTo(Set.of()))) {
            Registration registration = restricted.register(int64Spec());
            Rejection rejection = assertInstanceOf(Rejection.class, registration,
                    "an Int64 kernel must be refused under a budget that forbids Int64");
            assertEquals("requires a capability outside the target budget", rejection.reason());
            assertTrue(rejection.detail().contains("Int64"), () -> "witness should name Int64: " + rejection.detail());
        }
        // Unconstrained, the same kernel registers fine.
        try (Accelerator open = new Accelerator()) {
            assertTrue(open.register(int64Spec()).succeeded(), "Int64 kernel registers with no budget");
        }
    }

    @Test
    void rejectsAKernelWithNoOutputColumnWithAWitness() {
        try (Accelerator accelerator = new Accelerator()) {
            KernelSpec noOutput = new KernelSpec(vectorAdd(), List.of(
                    KernelColumn.input("a", 0),
                    KernelColumn.input("b", 1)));

            Registration registration = accelerator.register(noOutput);
            Rejection rejection = assertInstanceOf(Rejection.class, registration,
                    "a kernel that writes nothing must be rejected, not handed back as runnable");
            assertEquals("no output column", rejection.reason());
        }
    }
}
