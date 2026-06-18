package dev.supirvast.vastir.tools;

import com.oracle.truffle.api.CallTarget;
import dev.supirvast.vast.CoreToTruffle;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
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
 * Narrow integers (i8/i16), end to end. The kernel narrows an i32 buffer element to {@code i8} (resp.
 * {@code i16}), does arithmetic that overflows the narrow width — {@code (i8)a * 3} and {@code (i16)a + 30000}
 * — then widens the wrapped result back to i32 to store. The inputs are chosen so both results wrap, which is
 * the whole point: an i8 op must keep only 8 bits, not 32.
 *
 * <p>Exercises the {@code Int8}/{@code Int16} capabilities, narrow-width {@code OpConstant}s, narrowing/widening
 * {@code OpSConvert}, and narrow-width {@code OpIMul}/{@code OpIAdd} — CPU (Truffle, where results are
 * re-truncated to the declared width) vs GPU (Vulkan) vs a Java {@code byte}/{@code short} reference.
 */
class NarrowIntKernelTest {

    private static final Type.Int I32 = Type.int32();
    private static final Type.Int I8 = Type.int8();
    private static final Type.Int I16 = Type.int16();

    // Buffers ordered so slot index == binding == position in the int[][] passed to both backends.
    private static final Buffer OUT8 = new Buffer("out8", 0);
    private static final Buffer OUT16 = new Buffer("out16", 1);
    private static final Buffer A = new Buffer("a", 2);
    private static final List<Buffer> BUFFERS = List.of(OUT8, OUT16, A);

    private static Function narrowKernel() {
        Expr gid = new Expr.InvocationId();

        // (i32)( (i8)a[gid] * (i8)3 )
        Expr i8Mul = new Expr.Convert(
                new Expr.Binary(BinaryOp.MUL,
                        new Expr.Convert(new Expr.BufferLoad(A, gid), I8),
                        new Expr.ConstInt(I8, 3)),
                I32);
        // (i32)( (i16)a[gid] + (i16)30000 )
        Expr i16Add = new Expr.Convert(
                new Expr.Binary(BinaryOp.ADD,
                        new Expr.Convert(new Expr.BufferLoad(A, gid), I16),
                        new Expr.ConstInt(I16, 30000)),
                I32);

        Region body = Region.of(
                new Statement.BufferStore(OUT8, gid, i8Mul),
                new Statement.BufferStore(OUT16, gid, i16Add),
                new Statement.ReturnVoid());
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
    }

    @Test
    void narrowIntArithmeticAgreesOnCpuAndGpu() {
        int[] a = {100, 50, 127, 200, 1000, 10000, 20000, 32000};
        int n = a.length;
        int[] expected8 = new int[n];
        int[] expected16 = new int[n];
        for (int i = 0; i < n; i++) {
            expected8[i] = (byte) ((byte) a[i] * 3);
            expected16[i] = (short) ((short) a[i] + 30000);
        }

        Function kernel = narrowKernel();
        byte[] spirv = new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.compute(kernel, 1, 1, 1)))
                .toByteArray();

        NativeTools tools = new NativeTools();
        if (tools.isAvailable()) {
            NativeTools.ValidationResult validation = tools.validate(spirv);
            assertTrue(validation.valid(), () -> "spirv-val rejected the narrow-int kernel:\n" + validation.output());
        }

        CallTarget cpu = new CoreToTruffle().lowerKernel(kernel, BUFFERS);
        int[] cpu8 = new int[n];
        int[] cpu16 = new int[n];
        int[][] cpuBuffers = {cpu8, cpu16, a.clone()};
        for (int i = 0; i < n; i++) {
            cpu.call(i, cpuBuffers);
        }
        assertArrayEquals(expected8, cpu8, "CPU i8 multiply (wrapped)");
        assertArrayEquals(expected16, cpu16, "CPU i16 add (wrapped)");

        VulkanCompute vulkan = new VulkanCompute();
        assumeTrue(vulkan.isAvailable(), "no Vulkan compute device available");
        int[][] gpuBuffers = {new int[n], new int[n], a.clone()};
        int[][] gpuOut = vulkan.executeKernel(spirv, "main", gpuBuffers, n);

        assertArrayEquals(expected8, gpuOut[0], "GPU i8 multiply (wrapped)");
        assertArrayEquals(expected16, gpuOut[1], "GPU i16 add (wrapped)");
        assertArrayEquals(cpu8, gpuOut[0], "CPU and GPU must agree (i8)");
        assertArrayEquals(cpu16, gpuOut[1], "CPU and GPU must agree (i16)");
    }
}
