package dev.supirvast.vastir.tools;

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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Demonstrates the {@code OpPhi} escape hatch. We lower locals to memory ({@code OpVariable}/{@code OpLoad}/
 * {@code OpStore}) rather than emitting {@code OpPhi} by hand — a correct, complete strategy. This test proves
 * that when SSA form is actually wanted, {@code spirv-opt}'s {@code --ssa-rewrite} (mem2reg) promotes those
 * locals to {@code OpPhi} automatically: the original has stores and no phi, the optimized module has
 * {@code OpPhi}, still validates, and computes the identical result on the GPU.
 *
 * <p>The loop's trip count depends on {@code gid}, so the optimizer can't fold the whole thing to a constant —
 * a genuine loop (hence a genuine loop-header phi) survives.
 */
class SpirvOptPhiTest {

    private static final Type.Int I32 = Type.int32();
    private static final Buffer OUT = new Buffer("out", 0);
    private static final int N = 8;

    /** {@code sum=0; for (i=0; i<gid; i++) sum+=i; out[gid]=sum;} — locals sum/i are loop-carried. */
    private static Function triangularSumKernel() {
        Expr gid = new Expr.InvocationId();
        LocalVar sum = new LocalVar("sum", I32);
        LocalVar idx = new LocalVar("i", I32);
        Region loop = Region.of(
                new Statement.Assign(sum, new Expr.Binary(BinaryOp.ADD, new Expr.Read(sum), new Expr.Read(idx))),
                new Statement.Assign(idx, new Expr.Binary(BinaryOp.ADD, new Expr.Read(idx),
                        new Expr.ConstInt(I32, 1))));
        Region body = Region.of(
                new Statement.DeclareVar(sum, new Expr.ConstInt(I32, 0)),
                new Statement.DeclareVar(idx, new Expr.ConstInt(I32, 0)),
                new Statement.While(new Expr.Binary(BinaryOp.LESS_THAN, new Expr.Read(idx), gid), loop),
                new Statement.BufferStore(OUT, gid, new Expr.Read(sum)),
                new Statement.ReturnVoid());
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
    }

    @Test
    void spirvOptPromotesMemoryLocalsToOpPhi() {
        Function kernel = triangularSumKernel();
        byte[] original = new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.compute(kernel, 1, 1, 1)))
                .toByteArray();

        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        assertTrue(tools.validate(original).valid(), "original must validate");
        String originalAsm = tools.disassemble(original);
        assertTrue(originalAsm.contains("OpStore"), "our lowering uses memory-based locals (OpStore)");
        assertFalse(originalAsm.contains("OpPhi"), "our lowering does not emit OpPhi");

        // mem2reg: promote the function-local OpVariables to SSA values + OpPhi at the loop header.
        byte[] optimized = tools.optimize(original, "--ssa-rewrite");
        assertTrue(tools.validate(optimized).valid(), "optimized module must still validate");
        assertTrue(tools.disassemble(optimized).contains("OpPhi"),
                "spirv-opt --ssa-rewrite should introduce OpPhi for the loop-carried locals");

        // The transformation must be behavior-preserving: identical results on the GPU.
        int[] expected = new int[N];
        for (int g = 0; g < N; g++) {
            expected[g] = g * (g - 1) / 2; // 0+1+...+(gid-1)
        }

        VulkanCompute vulkan = new VulkanCompute();
        assumeTrue(vulkan.isAvailable(), "no Vulkan compute device available");
        int[] before = vulkan.executeKernel(original, "main", new int[][] {new int[N]}, N)[0];
        int[] after = vulkan.executeKernel(optimized, "main", new int[][] {new int[N]}, N)[0];

        assertArrayEquals(expected, before, "original kernel result");
        assertArrayEquals(expected, after, "optimized (OpPhi) kernel result");
        assertArrayEquals(before, after, "optimization must preserve behavior");
    }
}
