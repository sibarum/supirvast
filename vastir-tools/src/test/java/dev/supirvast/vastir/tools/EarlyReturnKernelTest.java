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
 * Early {@code return} inside structured regions — the guard-clause / bounds-check pattern. Each kernel returns
 * from within an {@code if} (or a loop body), which on the GPU means an {@code OpReturn} terminates a block
 * mid-construct: the lowering must drop the would-be branch to the merge and keep every structured merge/
 * continue block validly terminated (an unreachable merge gets {@code OpUnreachable}). The CPU backend already
 * unwinds via its return exception. Skipped invocations leave their output element at its initial 0.
 */
class EarlyReturnKernelTest {

    private static final Type.Int I32 = Type.int32();
    private static final Buffer OUT = new Buffer("out", 0);
    private static final List<Buffer> BUFFERS = List.of(OUT);
    private static final int N = 8;

    private static Expr i(long v) {
        return new Expr.ConstInt(I32, v);
    }

    private static Function entry(Region body) {
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
    }

    /** {@code if (gid < 4) return; out[gid] = gid*gid;} — a then-only early return, the false edge falls through. */
    @Test
    void guardReturn() {
        Expr gid = new Expr.InvocationId();
        Region body = Region.of(
                new Statement.If(new Expr.Binary(BinaryOp.LESS_THAN, gid, i(4)),
                        Region.of(new Statement.ReturnVoid()), Region.of()),
                new Statement.BufferStore(OUT, gid, new Expr.Binary(BinaryOp.MUL, gid, gid)),
                new Statement.ReturnVoid());

        int[] expected = new int[N];
        for (int g = 0; g < N; g++) {
            expected[g] = g < 4 ? 0 : g * g;
        }
        assertKernel(entry(body), expected);
    }

    /** Early return from inside a loop body: {@code while (i<10) { if (i==gid) { out[gid]=i*100; return; } i++; }} */
    @Test
    void loopReturn() {
        Expr gid = new Expr.InvocationId();
        LocalVar idx = new LocalVar("i", I32);
        Region loopBody = Region.of(
                new Statement.If(new Expr.Binary(BinaryOp.EQUAL, new Expr.Read(idx), gid),
                        Region.of(
                                new Statement.BufferStore(OUT, gid,
                                        new Expr.Binary(BinaryOp.MUL, new Expr.Read(idx), i(100))),
                                new Statement.ReturnVoid()),
                        Region.of()),
                new Statement.Assign(idx, new Expr.Binary(BinaryOp.ADD, new Expr.Read(idx), i(1))));
        Region body = Region.of(
                new Statement.DeclareVar(idx, i(0)),
                new Statement.While(new Expr.Binary(BinaryOp.LESS_THAN, new Expr.Read(idx), i(10)), loopBody),
                new Statement.BufferStore(OUT, gid, i(-1)), // only reached if the loop finishes (gid >= 10: never)
                new Statement.ReturnVoid());

        int[] expected = new int[N];
        for (int g = 0; g < N; g++) {
            expected[g] = g * 100;
        }
        assertKernel(entry(body), expected);
    }

    /** Both arms return: {@code if (gid<4) {out=1; return;} else {out=2; return;}} — the merge is unreachable. */
    @Test
    void bothBranchesReturn() {
        Expr gid = new Expr.InvocationId();
        Region body = Region.of(
                new Statement.If(new Expr.Binary(BinaryOp.LESS_THAN, gid, i(4)),
                        Region.of(new Statement.BufferStore(OUT, gid, i(1)), new Statement.ReturnVoid()),
                        Region.of(new Statement.BufferStore(OUT, gid, i(2)), new Statement.ReturnVoid())));

        int[] expected = new int[N];
        for (int g = 0; g < N; g++) {
            expected[g] = g < 4 ? 1 : 2;
        }
        assertKernel(entry(body), expected);
    }

    private static void assertKernel(Function kernel, int[] expected) {
        byte[] spirv = new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.compute(kernel, 1, 1, 1)))
                .toByteArray();

        NativeTools tools = new NativeTools();
        if (tools.isAvailable()) {
            NativeTools.ValidationResult validation = tools.validate(spirv);
            assertTrue(validation.valid(), () -> "spirv-val rejected the early-return kernel:\n" + validation.output());
        }

        CallTarget cpu = new CoreToTruffle().lowerKernel(kernel, BUFFERS);
        int[] cpuOut = new int[N];
        int[][] cpuBuffers = {cpuOut};
        for (int g = 0; g < N; g++) {
            cpu.call(g, cpuBuffers);
        }
        assertArrayEquals(expected, cpuOut, "CPU early-return result");

        VulkanCompute vulkan = new VulkanCompute();
        assumeTrue(vulkan.isAvailable(), "no Vulkan compute device available");
        int[] gpuOut = vulkan.executeKernel(spirv, "main", new int[][] {new int[N]}, N)[0];

        assertArrayEquals(expected, gpuOut, "GPU early-return result");
        assertArrayEquals(cpuOut, gpuOut, "CPU and GPU must agree");
    }
}
