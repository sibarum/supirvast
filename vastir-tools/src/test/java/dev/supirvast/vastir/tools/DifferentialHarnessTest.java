package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Drives {@link DifferentialHarness}: the same core body runs on the CPU and produces canonical valid SPIR-V. */
class DifferentialHarnessTest {

    private static final Type.Int I32 = Type.int32();

    private static Expr i(long value) {
        return new Expr.ConstInt(I32, value);
    }

    private DifferentialHarness.Report run(String name, List<Statement> body, LocalVar result) {
        DifferentialHarness harness = new DifferentialHarness();
        assumeTrue(harness.toolsAvailable(), "native SPIR-V tools not bundled");
        DifferentialHarness.Report report = harness.run(name, body, result);
        assertTrue(report.spirvValid(), () -> "spirv-val rejected " + name + ":\n" + report.validationOutput());
        assertTrue(report.disassemblyRoundTrips(),
                () -> name + ": our SPIR-V round-trip through spirv-dis/spirv-as is unstable");
        // When a GPU is present, the same shader run on hardware must produce the CPU's value.
        if (report.gpuExecuted()) {
            assertEquals(report.cpuResult(), report.gpuResult().intValue(),
                    () -> name + ": GPU result " + report.gpuResult() + " != CPU result " + report.cpuResult());
        }
        return report;
    }

    @Test
    void loopSumAgreesAcrossBothBackends() {
        LocalVar result = new LocalVar("result", I32);
        LocalVar i = new LocalVar("i", I32);
        List<Statement> body = List.of(
                new Statement.DeclareVar(result, i(0)),
                new Statement.DeclareVar(i, i(0)),
                new Statement.While(new Expr.Binary(BinaryOp.LESS_THAN, new Expr.Read(i), i(10)), Region.of(
                        new Statement.Assign(result, new Expr.Binary(BinaryOp.ADD, new Expr.Read(result), new Expr.Read(i))),
                        new Statement.Assign(i, new Expr.Binary(BinaryOp.ADD, new Expr.Read(i), i(1))))));
        assertEquals(45, run("loop-sum", body, result).cpuResult());
    }

    @Test
    void branchSelectsAtRuntimeAndValidates() {
        LocalVar result = new LocalVar("result", I32);
        List<Statement> body = List.of(
                new Statement.DeclareVar(result, i(0)),
                new Statement.If(new Expr.Binary(BinaryOp.LESS_THAN, i(3), i(5)),
                        Region.of(new Statement.Assign(result, i(10))),
                        Region.of(new Statement.Assign(result, i(20)))));
        assertEquals(10, run("branch", body, result).cpuResult());
    }

    @Test
    void arithmeticFoldsConsistently() {
        LocalVar result = new LocalVar("result", I32);
        // (2 + 3) * 4
        List<Statement> body = List.of(new Statement.DeclareVar(result,
                new Expr.Binary(BinaryOp.MUL, new Expr.Binary(BinaryOp.ADD, i(2), i(3)), i(4))));
        assertEquals(20, run("arith", body, result).cpuResult());
    }
}
