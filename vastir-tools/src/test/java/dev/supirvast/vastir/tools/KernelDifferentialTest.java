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
 * The first data-parallel differential: {@code out[i] = a[i] + b[i]} authored once in {@code core}, run on the
 * CPU (Truffle, one call per invocation) and on the GPU (Vulkan, an N-workgroup dispatch), with the output
 * arrays compared element-wise.
 */
class KernelDifferentialTest {

    // Buffers ordered so slot index == binding == position in the int[][] passed to both backends.
    private static final Buffer OUT = new Buffer("out", 0);
    private static final Buffer A = new Buffer("a", 1);
    private static final Buffer B = new Buffer("b", 2);
    private static final List<Buffer> BUFFERS = List.of(OUT, A, B);

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

    @Test
    void vectorAddAgreesOnCpuAndGpu() {
        int n = 8;
        int[] a = {0, 1, 2, 3, 4, 5, 6, 7};
        int[] b = {10, 20, 30, 40, 50, 60, 70, 80};
        int[] expected = new int[n];
        for (int i = 0; i < n; i++) {
            expected[i] = a[i] + b[i];
        }

        Function kernel = vectorAdd();
        byte[] spirv = new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.compute(kernel, 1, 1, 1)))
                .toByteArray();

        NativeTools tools = new NativeTools();
        if (tools.isAvailable()) {
            NativeTools.ValidationResult validation = tools.validate(spirv);
            assertTrue(validation.valid(), () -> "spirv-val rejected the kernel:\n" + validation.output());
        }

        // CPU: run the Truffle AST once per invocation over shared buffers.
        CallTarget cpu = new CoreToTruffle().lowerKernel(kernel, BUFFERS);
        int[] cpuOut = new int[n];
        int[][] cpuBuffers = {cpuOut, a.clone(), b.clone()};
        for (int i = 0; i < n; i++) {
            cpu.call(i, cpuBuffers);
        }
        assertArrayEquals(expected, cpuOut, "CPU kernel result");

        // GPU: one Vulkan dispatch of n workgroups.
        VulkanCompute vulkan = new VulkanCompute();
        assumeTrue(vulkan.isAvailable(), "no Vulkan compute device available");
        int[][] gpuBuffers = {new int[n], a.clone(), b.clone()};
        int[] gpuOut = vulkan.executeKernel(spirv, "main", gpuBuffers, n)[0];

        assertArrayEquals(expected, gpuOut, "GPU kernel result");
        assertArrayEquals(cpuOut, gpuOut, "CPU and GPU must agree");
    }
}
