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
 * 64-bit integers, end to end. The kernel widens two i32 buffer elements to {@code i64}, multiplies them (a
 * product that overflows 32 bits), and produces two results that each need real 64-bit math:
 * <ul>
 *   <li>{@code out0[i] = (int)(prod % 1_000_000_007)} — signed i64 multiply + remainder, then narrow; and</li>
 *   <li>{@code out1[i] = (int)((bitcast<u64>(prod) | 0x8000_0000_0000_0000) / 7)} — an <em>unsigned</em> i64
 *       divide of a value with the top bit set, which diverges from a signed divide.</li>
 * </ul>
 * Exercises {@link Expr.Convert} (i32↔i64, sign- and zero-handling), 64-bit {@code OpConstant}s, the
 * {@code Int64} capability, and unsigned 64-bit ops — compared CPU (Truffle) vs GPU (Vulkan) vs a Java
 * {@code long} reference. Buffers stay i32; the 64-bit values live only inside the kernel.
 */
class Int64KernelTest {

    private static final Type.Int I32 = Type.int32();
    private static final Type.Int I64 = Type.int64();
    private static final Type.Int U64 = Type.uint64();
    private static final long PRIME = 1_000_000_007L;
    private static final long TOP_BIT = 0x8000_0000_0000_0000L;

    // Buffers ordered so slot index == binding == position in the int[][] passed to both backends.
    private static final Buffer OUT0 = new Buffer("out0", 0);
    private static final Buffer OUT1 = new Buffer("out1", 1);
    private static final Buffer A = new Buffer("a", 2);
    private static final Buffer B = new Buffer("b", 3);
    private static final List<Buffer> BUFFERS = List.of(OUT0, OUT1, A, B);

    private static Function int64Kernel() {
        LocalVar prod = new LocalVar("prod", I64);
        Expr gid = new Expr.InvocationId();

        Expr la = new Expr.Convert(new Expr.BufferLoad(A, gid), I64);
        Expr lb = new Expr.Convert(new Expr.BufferLoad(B, gid), I64);

        Expr signedMod = new Expr.Binary(BinaryOp.MOD, new Expr.Read(prod), new Expr.ConstInt(I64, PRIME));

        Expr big = new Expr.Binary(BinaryOp.BIT_OR,
                new Expr.Bitcast(new Expr.Read(prod), U64),
                new Expr.ConstInt(U64, TOP_BIT));
        Expr unsignedDiv = new Expr.Binary(BinaryOp.DIV, big, new Expr.ConstInt(U64, 7));

        Region body = Region.of(
                new Statement.DeclareVar(prod, new Expr.Binary(BinaryOp.MUL, la, lb)),
                new Statement.BufferStore(OUT0, gid, new Expr.Convert(signedMod, I32)),
                new Statement.BufferStore(OUT1, gid, new Expr.Convert(unsignedDiv, I32)),
                new Statement.ReturnVoid());
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
    }

    @Test
    void int64MathAgreesOnCpuAndGpu() {
        int[] a = {1_000_000_000, 999_999_999, 123_456_789, 2_000_000_000,
                1_518_500_249, 65_536, 2_000_000_000, 1};
        int[] b = {1_000_000_000, 999_999_999, 987_654_321, 1_500_000_000,
                1_518_500_249, 65_537, 999_999_999, 2_000_000_000};
        int n = a.length;
        int[] expected0 = new int[n];
        int[] expected1 = new int[n];
        for (int i = 0; i < n; i++) {
            long prod = (long) a[i] * (long) b[i];
            expected0[i] = (int) (prod % PRIME);
            expected1[i] = (int) Long.divideUnsigned(prod | TOP_BIT, 7L);
        }

        Function kernel = int64Kernel();
        byte[] spirv = new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.compute(kernel, 1, 1, 1)))
                .toByteArray();

        NativeTools tools = new NativeTools();
        if (tools.isAvailable()) {
            NativeTools.ValidationResult validation = tools.validate(spirv);
            assertTrue(validation.valid(), () -> "spirv-val rejected the i64 kernel:\n" + validation.output());
        }

        // CPU: run the Truffle AST once per invocation over shared buffers.
        CallTarget cpu = new CoreToTruffle().lowerKernel(kernel, BUFFERS);
        int[] cpuOut0 = new int[n];
        int[] cpuOut1 = new int[n];
        int[][] cpuBuffers = {cpuOut0, cpuOut1, a.clone(), b.clone()};
        for (int i = 0; i < n; i++) {
            cpu.call(i, cpuBuffers);
        }
        assertArrayEquals(expected0, cpuOut0, "CPU signed i64 mul+mod");
        assertArrayEquals(expected1, cpuOut1, "CPU unsigned i64 divide");

        // GPU: one Vulkan dispatch of n workgroups.
        VulkanCompute vulkan = new VulkanCompute();
        assumeTrue(vulkan.isAvailable(), "no Vulkan compute device available");
        int[][] gpuBuffers = {new int[n], new int[n], a.clone(), b.clone()};
        int[][] gpuOut = vulkan.executeKernel(spirv, "main", gpuBuffers, n);

        assertArrayEquals(expected0, gpuOut[0], "GPU signed i64 mul+mod");
        assertArrayEquals(expected1, gpuOut[1], "GPU unsigned i64 divide");
        assertArrayEquals(cpuOut0, gpuOut[0], "CPU and GPU must agree (signed)");
        assertArrayEquals(cpuOut1, gpuOut[1], "CPU and GPU must agree (unsigned)");
    }
}
