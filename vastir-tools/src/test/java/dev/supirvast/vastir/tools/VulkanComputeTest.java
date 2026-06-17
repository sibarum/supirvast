package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.BinaryOp;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Runs a SPIR-V compute shader on the real Vulkan device and reads back the result. */
class VulkanComputeTest {

    /** {@code int result=0,i=0; while (i<10){result=result+i; i=i+1;} buffer[0]=result;}  ==> 45 */
    private static byte[] sumToBuffer() {
        Type.Int i32 = Type.int32();
        LocalVar result = new LocalVar("result", i32);
        LocalVar i = new LocalVar("i", i32);
        Region loop = Region.of(
                new Statement.Assign(result, new Expr.Binary(BinaryOp.ADD, new Expr.Read(result), new Expr.Read(i))),
                new Statement.Assign(i, new Expr.Binary(BinaryOp.ADD, new Expr.Read(i), new Expr.ConstInt(i32, 1))));
        Function main = new Function("main", new Type.FunctionType(Type.VOID, java.util.List.of()), Region.of(
                new Statement.DeclareVar(result, new Expr.ConstInt(i32, 0)),
                new Statement.DeclareVar(i, new Expr.ConstInt(i32, 0)),
                new Statement.While(new Expr.Binary(BinaryOp.LESS_THAN, new Expr.Read(i), new Expr.ConstInt(i32, 10)), loop),
                new Statement.StoreResult(new Expr.Read(result)),
                new Statement.ReturnVoid()));
        CoreModule module = new CoreModule().addEntryPoint(EntryPoint.compute(main, 1, 1, 1));
        return new CoreToSpirv().lower(module).toByteArray();
    }

    @Test
    void executesOnGpuAndReadsBackResult() {
        VulkanCompute vulkan = new VulkanCompute();
        assumeTrue(vulkan.isAvailable(), "no Vulkan 1.3 compute device available");
        assertEquals(45, vulkan.execute(sumToBuffer(), "main"));
    }
}
