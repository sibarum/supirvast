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
 * int↔float conversions, end to end: {@code out[gid] = (int)((float)a[gid] * 1.5f)}. Exercises
 * {@code OpConvertSToF} (widen a signed i32 to f32), a correctly-rounded f32 multiply, and
 * {@code OpConvertFToS} (round toward zero back to i32). Values include negatives and ones whose ×1.5 is
 * fractional, so the round-toward-zero truncation is genuinely tested. CPU (Truffle) vs GPU (Vulkan) vs a
 * Java reference; all values stay in range (out-of-range float→int is undefined in SPIR-V).
 */
class ConversionKernelTest {

    private static final Type.Int I32 = Type.int32();
    private static final Type.Float F32 = Type.float32();

    // Buffers ordered so slot index == binding == position in the int[][] passed to both backends.
    private static final Buffer OUT = new Buffer("out", 0);
    private static final Buffer A = new Buffer("a", 1);
    private static final List<Buffer> BUFFERS = List.of(OUT, A);

    private static Function conversionKernel() {
        LocalVar f = new LocalVar("f", F32);
        Expr gid = new Expr.InvocationId();

        Region body = Region.of(
                new Statement.DeclareVar(f, new Expr.Convert(new Expr.BufferLoad(A, gid), F32)),
                new Statement.BufferStore(OUT, gid, new Expr.Convert(
                        new Expr.Binary(BinaryOp.MUL, new Expr.Read(f), new Expr.ConstFloat(F32, 1.5)), I32)),
                new Statement.ReturnVoid());
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
    }

    @Test
    void conversionsAgreeOnCpuAndGpu() {
        int[] a = {0, 1, 3, 7, 10, 25, -3, -8};
        int n = a.length;
        int[] expected = new int[n];
        for (int i = 0; i < n; i++) {
            expected[i] = (int) ((float) a[i] * 1.5f);
        }

        Function kernel = conversionKernel();
        byte[] spirv = new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.compute(kernel, 1, 1, 1)))
                .toByteArray();

        NativeTools tools = new NativeTools();
        if (tools.isAvailable()) {
            NativeTools.ValidationResult validation = tools.validate(spirv);
            assertTrue(validation.valid(), () -> "spirv-val rejected the conversion kernel:\n" + validation.output());
        }

        CallTarget cpu = new CoreToTruffle().lowerKernel(kernel, BUFFERS);
        int[] cpuOut = new int[n];
        int[][] cpuBuffers = {cpuOut, a.clone()};
        for (int i = 0; i < n; i++) {
            cpu.call(i, cpuBuffers);
        }
        assertArrayEquals(expected, cpuOut, "CPU int<->float conversions");

        VulkanCompute vulkan = new VulkanCompute();
        assumeTrue(vulkan.isAvailable(), "no Vulkan compute device available");
        int[][] gpuBuffers = {new int[n], a.clone()};
        int[] gpuOut = vulkan.executeKernel(spirv, "main", gpuBuffers, n)[0];

        assertArrayEquals(expected, gpuOut, "GPU int<->float conversions");
        assertArrayEquals(cpuOut, gpuOut, "CPU and GPU must agree");
    }
}
