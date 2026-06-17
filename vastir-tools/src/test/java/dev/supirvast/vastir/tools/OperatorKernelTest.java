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
import dev.supirvast.vastir.core.UnaryOp;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Exercises the expanded operator set — div/mod, bitwise, shifts, unary negate/not, and logical and/not —
 * in one kernel, compared CPU vs GPU vs an identical Java reference.
 */
class OperatorKernelTest {

    private static final Type.Int I32 = Type.int32();
    private static final Buffer OUT = new Buffer("out", 0);

    /** The reference computation, mirrored exactly by the core IR below. */
    private static int reference(int g) {
        int t1 = (g * 7) / 3;
        int t2 = t1 % 5;
        int t3 = (g << 1) ^ (g & 3);
        int t4 = t2 | t3;
        int t5 = ~t4;
        int t6 = t5 >> 1;
        int base = -t6;
        return ((g > 1) && !(g > 5)) ? base : base - 1;
    }

    private static Expr i(long v) {
        return new Expr.ConstInt(I32, v);
    }

    private static Expr bin(BinaryOp op, Expr a, Expr b) {
        return new Expr.Binary(op, a, b);
    }

    private static Function operatorKernel() {
        LocalVar base = new LocalVar("base", I32);
        LocalVar r = new LocalVar("r", I32);
        Expr g = new Expr.InvocationId();

        Expr t1 = bin(BinaryOp.DIV, bin(BinaryOp.MUL, g, i(7)), i(3));
        Expr t2 = bin(BinaryOp.MOD, t1, i(5));
        Expr t3 = bin(BinaryOp.BIT_XOR, bin(BinaryOp.SHIFT_LEFT, g, i(1)), bin(BinaryOp.BIT_AND, g, i(3)));
        Expr t4 = bin(BinaryOp.BIT_OR, t2, t3);
        Expr t5 = new Expr.Unary(UnaryOp.NOT, t4);
        Expr t6 = bin(BinaryOp.SHIFT_RIGHT, t5, i(1));
        Expr baseExpr = new Expr.Unary(UnaryOp.NEGATE, t6);

        Expr condition = bin(BinaryOp.LOGICAL_AND,
                bin(BinaryOp.GREATER_THAN, g, i(1)),
                new Expr.Unary(UnaryOp.LOGICAL_NOT, bin(BinaryOp.GREATER_THAN, g, i(5))));

        Region body = Region.of(
                new Statement.DeclareVar(base, baseExpr),
                new Statement.DeclareVar(r, i(0)),
                new Statement.If(condition,
                        Region.of(new Statement.Assign(r, new Expr.Read(base))),
                        Region.of(new Statement.Assign(r, bin(BinaryOp.SUB, new Expr.Read(base), i(1))))),
                new Statement.BufferStore(OUT, new Expr.InvocationId(), new Expr.Read(r)),
                new Statement.ReturnVoid());
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
    }

    @Test
    void operatorsAgreeOnCpuAndGpu() {
        int n = 8;
        int[] expected = new int[n];
        for (int i = 0; i < n; i++) {
            expected[i] = reference(i);
        }

        Function kernel = operatorKernel();
        byte[] spirv = new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.compute(kernel, 1, 1, 1)))
                .toByteArray();

        NativeTools tools = new NativeTools();
        if (tools.isAvailable()) {
            NativeTools.ValidationResult validation = tools.validate(spirv);
            assertTrue(validation.valid(), () -> "spirv-val rejected the operator kernel:\n" + validation.output());
        }

        CallTarget cpu = new CoreToTruffle().lowerKernel(kernel, List.of(OUT));
        int[] cpuOut = new int[n];
        for (int i = 0; i < n; i++) {
            cpu.call(i, new int[][] {cpuOut});
        }
        assertArrayEquals(expected, cpuOut, "CPU operators");

        VulkanCompute vulkan = new VulkanCompute();
        assumeTrue(vulkan.isAvailable(), "no Vulkan compute device available");
        int[] gpuOut = vulkan.executeKernel(spirv, "main", new int[][] {new int[n]}, n)[0];

        assertArrayEquals(expected, gpuOut, "GPU operators");
        assertArrayEquals(cpuOut, gpuOut, "CPU and GPU must agree");
    }
}
