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
 * Vector math, run on CPU and GPU and compared. Each invocation builds {@code ivec4(gid,gid,gid,gid)}, adds
 * {@code ivec4(1,2,3,4)} componentwise, then sums the components: {@code out[i] = 4*i + 10}.
 */
class VectorKernelTest {

    private static final Buffer OUT = new Buffer("out", 0);

    private static Expr i(long v) {
        return new Expr.ConstInt(Type.int32(), v);
    }

    private static Function vectorKernel() {
        Type.Vector ivec4 = new Type.Vector(Type.int32(), 4);
        Expr gid = new Expr.InvocationId();
        Expr base = new Expr.VectorConstruct(ivec4, List.of(gid, gid, gid, gid));
        Expr offset = new Expr.VectorConstruct(ivec4, List.of(i(1), i(2), i(3), i(4)));
        Expr sum = new Expr.Binary(BinaryOp.ADD, base, offset); // componentwise -> ivec4

        Expr reduced = new Expr.Binary(BinaryOp.ADD,
                new Expr.Binary(BinaryOp.ADD, new Expr.VectorExtract(sum, 0), new Expr.VectorExtract(sum, 1)),
                new Expr.Binary(BinaryOp.ADD, new Expr.VectorExtract(sum, 2), new Expr.VectorExtract(sum, 3)));

        Region body = Region.of(
                new Statement.BufferStore(OUT, new Expr.InvocationId(), reduced),
                new Statement.ReturnVoid());
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
    }

    @Test
    void vectorMathAgreesOnCpuAndGpu() {
        int n = 8;
        int[] expected = new int[n];
        for (int i = 0; i < n; i++) {
            expected[i] = 4 * i + 10;
        }

        Function kernel = vectorKernel();
        byte[] spirv = new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.compute(kernel, 1, 1, 1)))
                .toByteArray();

        NativeTools tools = new NativeTools();
        if (tools.isAvailable()) {
            NativeTools.ValidationResult validation = tools.validate(spirv);
            assertTrue(validation.valid(), () -> "spirv-val rejected the vector kernel:\n" + validation.output());
        }

        CallTarget cpu = new CoreToTruffle().lowerKernel(kernel, List.of(OUT));
        int[] cpuOut = new int[n];
        for (int i = 0; i < n; i++) {
            cpu.call(i, new int[][] {cpuOut});
        }
        assertArrayEquals(expected, cpuOut, "CPU vector kernel");

        VulkanCompute vulkan = new VulkanCompute();
        assumeTrue(vulkan.isAvailable(), "no Vulkan compute device available");
        int[] gpuOut = vulkan.executeKernel(spirv, "main", new int[][] {new int[n]}, n)[0];

        assertArrayEquals(expected, gpuOut, "GPU vector kernel");
        assertArrayEquals(cpuOut, gpuOut, "CPU and GPU must agree");
    }
}
