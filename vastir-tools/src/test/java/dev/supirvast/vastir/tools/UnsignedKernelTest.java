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
 * Unsigned semantics, end to end. The same {@code core} kernel loads an i32 buffer element, {@link Expr.Bitcast
 * bitcasts} it to {@code uint32}, and runs the four operators whose result depends on signedness — unsigned
 * division, remainder, logical right shift, and unsigned ordering — before bitcasting back to i32 to store.
 *
 * <p>The inputs all have the high bit set, so every one of those operators diverges from its signed counterpart;
 * a kernel that accidentally lowered to {@code OpSDiv}/{@code OpShiftRightArithmetic}/{@code OpSGreaterThan}
 * (or a CPU backend using Java's signed {@code /}, {@code >>}, {@code <}) would produce different numbers and
 * fail. CPU (Truffle) and GPU (Vulkan) are both compared against an unsigned Java reference.
 */
class UnsignedKernelTest {

    private static final Type.Int I32 = Type.int32();
    private static final Type.Int U32 = Type.uint32();

    // Buffers ordered so slot index == binding == position in the int[][] passed to both backends.
    private static final Buffer OUT = new Buffer("out", 0);
    private static final Buffer A = new Buffer("a", 1);
    private static final List<Buffer> BUFFERS = List.of(OUT, A);

    /** The unsigned reference, mirrored exactly by the core IR below. */
    private static int reference(int a) {
        int q = Integer.divideUnsigned(a, 7);
        int r = Integer.remainderUnsigned(a, 7);
        int sh = a >>> 1;
        int big = Integer.compareUnsigned(a, 0x40000000) > 0 ? 1 : 0;
        return q + r + sh + big;
    }

    private static Expr u(long v) {
        return new Expr.ConstInt(U32, v);
    }

    /**
     * {@code uint ua = bitcast<uint>(a[gid]); uint res = ua/7u + ua%7u + (ua >> 1u);
     * if (ua > 0x40000000u) res += 1u; out[gid] = bitcast<int>(res);}
     */
    private static Function unsignedKernel() {
        LocalVar ua = new LocalVar("ua", U32);
        LocalVar res = new LocalVar("res", U32);
        Expr gid = new Expr.InvocationId();

        Expr q = new Expr.Binary(BinaryOp.DIV, new Expr.Read(ua), u(7));
        Expr r = new Expr.Binary(BinaryOp.MOD, new Expr.Read(ua), u(7));
        Expr sh = new Expr.Binary(BinaryOp.SHIFT_RIGHT, new Expr.Read(ua), u(1));
        Expr sum = new Expr.Binary(BinaryOp.ADD, new Expr.Binary(BinaryOp.ADD, q, r), sh);
        Expr big = new Expr.Binary(BinaryOp.GREATER_THAN, new Expr.Read(ua), u(0x40000000L));

        Region body = Region.of(
                new Statement.DeclareVar(ua, new Expr.Bitcast(new Expr.BufferLoad(A, gid), U32)),
                new Statement.DeclareVar(res, sum),
                new Statement.If(big,
                        Region.of(new Statement.Assign(res,
                                new Expr.Binary(BinaryOp.ADD, new Expr.Read(res), u(1)))),
                        Region.of()),
                new Statement.BufferStore(OUT, new Expr.InvocationId(),
                        new Expr.Bitcast(new Expr.Read(res), I32)),
                new Statement.ReturnVoid());
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
    }

    @Test
    void unsignedOperatorsAgreeOnCpuAndGpu() {
        // All high-bit-set, so signed and unsigned interpretations diverge for every operator under test.
        int[] a = {
                0x80000000, 0xFFFFFFFF, 0x90000005, 0x40000001,
                0xC0000000, 0xAAAAAAAB, 0x80000007, 0xFFFFFFFE,
        };
        int n = a.length;
        int[] expected = new int[n];
        for (int i = 0; i < n; i++) {
            expected[i] = reference(a[i]);
        }

        Function kernel = unsignedKernel();
        byte[] spirv = new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.compute(kernel, 1, 1, 1)))
                .toByteArray();

        NativeTools tools = new NativeTools();
        if (tools.isAvailable()) {
            NativeTools.ValidationResult validation = tools.validate(spirv);
            assertTrue(validation.valid(), () -> "spirv-val rejected the unsigned kernel:\n" + validation.output());
        }

        // CPU: run the Truffle AST once per invocation over shared buffers.
        CallTarget cpu = new CoreToTruffle().lowerKernel(kernel, BUFFERS);
        int[] cpuOut = new int[n];
        int[][] cpuBuffers = {cpuOut, a.clone()};
        for (int i = 0; i < n; i++) {
            cpu.call(i, cpuBuffers);
        }
        assertArrayEquals(expected, cpuOut, "CPU unsigned operators");

        // GPU: one Vulkan dispatch of n workgroups.
        VulkanCompute vulkan = new VulkanCompute();
        assumeTrue(vulkan.isAvailable(), "no Vulkan compute device available");
        int[][] gpuBuffers = {new int[n], a.clone()};
        int[] gpuOut = vulkan.executeKernel(spirv, "main", gpuBuffers, n)[0];

        assertArrayEquals(expected, gpuOut, "GPU unsigned operators");
        assertArrayEquals(cpuOut, gpuOut, "CPU and GPU must agree");
    }
}
