package dev.supirvast.vastir.tools;

import com.oracle.truffle.api.CallTarget;
import dev.supirvast.vast.CoreToTruffle;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * 32-bit float fidelity, end to end. The kernel reads a float (carried through an i32 buffer as its raw bits),
 * cubes it with two chained {@code f32} multiplies, and writes the result's bits back. Each multiply rounds to
 * f32 exactly as the GPU does — so this fails if the CPU backend computes in {@code double} (the old behavior),
 * because the extra precision kept across the two multiplies would diverge from the device on the round-trip.
 *
 * <p>Results are compared as raw int bits, so agreement is bit-exact. Only correctly-rounded ops are used
 * (Vulkan guarantees 0-ULP {@code OpFMul}); no division or fused multiply-add, which carry looser precision.
 * Exercises {@link Expr.Bitcast} across {@code f32}↔{@code i32} (a real re-encode on the CPU, not an identity).
 */
class FloatKernelTest {

    private static final Type.Int I32 = Type.int32();
    private static final Type.Float F32 = Type.float32();

    // Buffers ordered so slot index == binding == position in the int[][] passed to both backends.
    private static final Buffer OUT = new Buffer("out", 0);
    private static final Buffer A = new Buffer("a", 1);
    private static final List<Buffer> BUFFERS = List.of(OUT, A);

    /** {@code float f = bitcast<float>(a[gid]); float sq = f*f; out[gid] = bitcast<int>(sq*f);} */
    private static Function cubeKernel() {
        LocalVar f = new LocalVar("f", F32);
        LocalVar sq = new LocalVar("sq", F32);
        Expr gid = new Expr.InvocationId();

        Region body = Region.of(
                new Statement.DeclareVar(f, new Expr.Bitcast(new Expr.BufferLoad(A, gid), F32)),
                new Statement.DeclareVar(sq, new Expr.Binary(BinaryOp.MUL, new Expr.Read(f), new Expr.Read(f))),
                new Statement.BufferStore(OUT, gid,
                        new Expr.Bitcast(new Expr.Binary(BinaryOp.MUL, new Expr.Read(sq), new Expr.Read(f)), I32)),
                new Statement.ReturnVoid());
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
    }

    @Test
    void floatCubeAgreesOnCpuAndGpu() {
        float[] values = {1.1f, 1.0000001f, 3.14159f, 0.1f, 1.7f, 2.3f, 0.7f, 1.3f};
        int n = values.length;
        int[] a = new int[n];
        int[] expected = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = Float.floatToRawIntBits(values[i]);
            float f = Float.intBitsToFloat(a[i]);
            float sq = f * f;
            expected[i] = Float.floatToRawIntBits(sq * f); // stepwise f32 rounding, matching the kernel
        }

        Function kernel = cubeKernel();
        byte[] spirv = new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.compute(kernel, 1, 1, 1)))
                .toByteArray();

        NativeTools tools = new NativeTools();
        if (tools.isAvailable()) {
            NativeTools.ValidationResult validation = tools.validate(spirv);
            assertTrue(validation.valid(), () -> "spirv-val rejected the float kernel:\n" + validation.output());
        }

        // CPU: run the Truffle AST once per invocation over shared buffers.
        CallTarget cpu = new CoreToTruffle().lowerKernel(kernel, BUFFERS);
        int[] cpuOut = new int[n];
        int[][] cpuBuffers = {cpuOut, a.clone()};
        for (int i = 0; i < n; i++) {
            cpu.call(i, cpuBuffers);
        }
        assertArrayEquals(expected, cpuOut, "CPU f32 cube (raw bits)");

        // GPU: one Vulkan dispatch of n workgroups.
        VulkanCompute vulkan = new VulkanCompute();
        assumeTrue(vulkan.isAvailable(), "no Vulkan compute device available");
        int[][] gpuBuffers = {new int[n], a.clone()};
        int[] gpuOut = vulkan.executeKernel(spirv, "main", gpuBuffers, n)[0];

        assertArrayEquals(expected, gpuOut, "GPU f32 cube (raw bits)");
        assertArrayEquals(cpuOut, gpuOut, "CPU and GPU must agree bit-for-bit");
    }
}
