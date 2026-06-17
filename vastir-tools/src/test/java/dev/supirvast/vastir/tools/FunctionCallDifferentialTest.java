package dev.supirvast.vastir.tools;

import com.oracle.truffle.api.CallTarget;
import dev.supirvast.vast.CoreToTruffle;
import dev.supirvast.vastir.core.BinaryOp;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Function calls across both backends: a helper {@code int add(int x, int y)} is invoked (via
 * {@code OpFunctionCall} on the GPU, a {@code DirectCall} on the CPU) and must produce the same result.
 */
class FunctionCallDifferentialTest {

    private static final Type.Int I32 = Type.int32();

    private static Expr i(long v) {
        return new Expr.ConstInt(I32, v);
    }

    /** {@code int add(int x, int y) { return x + y; }} */
    private static Function add() {
        return new Function("add", new Type.FunctionType(I32, List.of(I32, I32)),
                Region.of(new Statement.Return(
                        new Expr.Binary(BinaryOp.ADD, new Expr.Param(0, I32), new Expr.Param(1, I32)))));
    }

    @Test
    void callResultAgreesOnCpuAndGpu() {
        Function add = add();
        Expr call = new Expr.Call(add, List.of(i(20), i(25))); // 45

        // CPU: compute() { return add(20, 25); }
        Function compute = new Function("compute", new Type.FunctionType(I32, List.of()),
                Region.of(new Statement.Return(call)));
        int cpu = (Integer) new CoreToTruffle().lowerModule(List.of(add, compute), compute).call();
        assertEquals(45, cpu, "CPU function call");

        // GPU: void main() { result = add(20, 25); } as a compute entry point writing the SSBO.
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()),
                Region.of(new Statement.StoreResult(call), new Statement.ReturnVoid()));
        byte[] spirv = new CoreToSpirv()
                .lower(new CoreModule().addFunction(add).addEntryPoint(EntryPoint.compute(main, 1, 1, 1)))
                .toByteArray();

        NativeTools tools = new NativeTools();
        if (tools.isAvailable()) {
            NativeTools.ValidationResult validation = tools.validate(spirv);
            assertTrue(validation.valid(), () -> "spirv-val rejected the call:\n" + validation.output());
        }

        VulkanCompute vulkan = new VulkanCompute();
        assumeTrue(vulkan.isAvailable(), "no Vulkan compute device available");
        assertEquals(45, vulkan.execute(spirv, "main"), "GPU function call");
        assertEquals(cpu, vulkan.execute(spirv, "main"), "CPU and GPU must agree");
    }
}
