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
 * 64-bit float (f64) fidelity, end to end. The kernel widens an i32 to {@code f64}, multiplies by {@code 0.1}
 * (not exactly representable, so the result carries real f64 rounding), then bitcasts the double to its raw
 * 64 bits and splits them into two i32 halves to escape through the i32-only buffers. The halves are compared
 * as raw bits, so f64 agreement is exact.
 *
 * <p>Exercises the {@code Float64} capability, two-word f64 {@code OpConstant}, {@code OpConvertSToF} (i32→f64),
 * a correctly-rounded f64 multiply, {@code OpBitcast} f64↔u64, an unsigned 64-bit logical shift, and narrowing
 * conversions — CPU (Truffle) vs GPU (Vulkan) vs a Java {@code double} reference.
 */
class Float64KernelTest {

    private static final Type.Int I32 = Type.int32();
    private static final Type.Int U64 = Type.uint64();
    private static final Type.Float F64 = Type.float64();

    // Buffers ordered so slot index == binding == position in the int[][] passed to both backends.
    private static final Buffer OUT_LO = new Buffer("outLo", 0);
    private static final Buffer OUT_HI = new Buffer("outHi", 1);
    private static final Buffer A = new Buffer("a", 2);
    private static final List<Buffer> BUFFERS = List.of(OUT_LO, OUT_HI, A);

    private static Function float64Kernel() {
        LocalVar bits = new LocalVar("bits", U64);
        Expr gid = new Expr.InvocationId();

        Expr r = new Expr.Binary(BinaryOp.MUL,
                new Expr.Convert(new Expr.BufferLoad(A, gid), F64),
                new Expr.ConstFloat(F64, 0.1));

        Region body = Region.of(
                new Statement.DeclareVar(bits, new Expr.Bitcast(r, U64)),
                new Statement.BufferStore(OUT_LO, gid, new Expr.Convert(new Expr.Read(bits), I32)),
                new Statement.BufferStore(OUT_HI, gid, new Expr.Convert(
                        new Expr.Binary(BinaryOp.SHIFT_RIGHT, new Expr.Read(bits), new Expr.ConstInt(U64, 32)),
                        I32)),
                new Statement.ReturnVoid());
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
    }

    @Test
    void float64MathAgreesOnCpuAndGpu() {
        int[] a = {1, 3, 7, 10, 42, 100, 999, 1234};
        int n = a.length;
        int[] expectedLo = new int[n];
        int[] expectedHi = new int[n];
        for (int i = 0; i < n; i++) {
            double r = (double) a[i] * 0.1;
            long bits = Double.doubleToRawLongBits(r);
            expectedLo[i] = (int) bits;
            expectedHi[i] = (int) (bits >>> 32);
        }

        Function kernel = float64Kernel();
        byte[] spirv = new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.compute(kernel, 1, 1, 1)))
                .toByteArray();

        NativeTools tools = new NativeTools();
        if (tools.isAvailable()) {
            NativeTools.ValidationResult validation = tools.validate(spirv);
            assertTrue(validation.valid(), () -> "spirv-val rejected the f64 kernel:\n" + validation.output());
        }

        CallTarget cpu = new CoreToTruffle().lowerKernel(kernel, BUFFERS);
        int[] cpuLo = new int[n];
        int[] cpuHi = new int[n];
        int[][] cpuBuffers = {cpuLo, cpuHi, a.clone()};
        for (int i = 0; i < n; i++) {
            cpu.call(i, cpuBuffers);
        }
        assertArrayEquals(expectedLo, cpuLo, "CPU f64 low word");
        assertArrayEquals(expectedHi, cpuHi, "CPU f64 high word");

        VulkanCompute vulkan = new VulkanCompute();
        assumeTrue(vulkan.isAvailable(), "no Vulkan compute device available");
        int[][] gpuBuffers = {new int[n], new int[n], a.clone()};
        int[][] gpuOut = vulkan.executeKernel(spirv, "main", gpuBuffers, n);

        assertArrayEquals(expectedLo, gpuOut[0], "GPU f64 low word");
        assertArrayEquals(expectedHi, gpuOut[1], "GPU f64 high word");
        assertArrayEquals(cpuLo, gpuOut[0], "CPU and GPU must agree (low)");
        assertArrayEquals(cpuHi, gpuOut[1], "CPU and GPU must agree (high)");
    }
}
