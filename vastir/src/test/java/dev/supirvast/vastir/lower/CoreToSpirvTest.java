package dev.supirvast.vastir.lower;

import dev.supirvast.vastir.binary.Instruction;
import dev.supirvast.vastir.binary.SpirvModule;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.spirv.Op;
import dev.supirvast.vastir.spirv.Spirv;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CoreToSpirvTest {

    /**
     * {@code int acc=0, i=0; while (i<10) { acc=acc+i; i=i+1; } if (acc<5) { acc=0; }} — exercises constants,
     * local variables, arithmetic, comparison, and both structured control-flow forms.
     */
    static CoreModule controlFlowShader() {
        Type.Int i32 = Type.int32();
        LocalVar acc = new LocalVar("acc", i32);
        LocalVar i = new LocalVar("i", i32);
        Expr zero = new Expr.ConstInt(i32, 0);

        Region loopBody = Region.of(
                new Statement.Assign(acc, new Expr.Binary(BinaryOp.ADD, new Expr.Read(acc), new Expr.Read(i))),
                new Statement.Assign(i, new Expr.Binary(BinaryOp.ADD, new Expr.Read(i), new Expr.ConstInt(i32, 1))));

        Region body = Region.of(
                new Statement.DeclareVar(acc, zero),
                new Statement.DeclareVar(i, zero),
                new Statement.While(new Expr.Binary(BinaryOp.LESS_THAN, new Expr.Read(i), new Expr.ConstInt(i32, 10)), loopBody),
                new Statement.If(new Expr.Binary(BinaryOp.LESS_THAN, new Expr.Read(acc), new Expr.ConstInt(i32, 5)),
                        Region.of(new Statement.Assign(acc, zero)), Region.of()),
                new Statement.ReturnVoid());

        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreModule().addEntryPoint(EntryPoint.compute(main, 8, 1, 1));
    }

    private static long countOp(List<Instruction> instructions, Op op) {
        return instructions.stream().filter(i -> i.op() == op).count();
    }

    private static SpirvModule lowerMinimalCompute() {
        Type.FunctionType voidFn = new Type.FunctionType(Type.VOID, List.of());
        Function main = new Function("main", voidFn, Region.of(new Statement.ReturnVoid()));
        CoreModule module = new CoreModule().addEntryPoint(EntryPoint.compute(main, 1, 1, 1));
        return new CoreToSpirv().lower(module);
    }

    private static int firstIndexOf(List<Instruction> instructions, Op op) {
        for (int i = 0; i < instructions.size(); i++) {
            if (instructions.get(i).op() == op) {
                return i;
            }
        }
        return -1;
    }

    @Test
    void emitsHeaderAndExpectedOpcodes() {
        SpirvModule spirv = lowerMinimalCompute();
        assertEquals(Spirv.MAGIC_NUMBER, spirv.toWords()[0]);
        assertTrue(spirv.idBound() > 1);

        List<Op> ops = spirv.instructions().stream().map(Instruction::op).toList();
        for (Op required : List.of(Op.OpCapability, Op.OpMemoryModel, Op.OpEntryPoint, Op.OpExecutionMode,
                Op.OpTypeVoid, Op.OpTypeFunction, Op.OpFunction, Op.OpLabel, Op.OpReturn, Op.OpFunctionEnd)) {
            assertTrue(ops.contains(required), () -> "missing " + required);
        }
    }

    @Test
    void respectsSpirvModuleLayout() {
        List<Instruction> instructions = lowerMinimalCompute().instructions();
        // entry points precede type declarations, which precede function definitions.
        assertTrue(firstIndexOf(instructions, Op.OpEntryPoint) < firstIndexOf(instructions, Op.OpTypeVoid));
        assertTrue(firstIndexOf(instructions, Op.OpTypeFunction) < firstIndexOf(instructions, Op.OpFunction));
    }

    @Test
    void deduplicatesStructurallyEqualTypes() {
        // void appears as both the return type and inside the function type; it must be declared once.
        long voidDecls = lowerMinimalCompute().instructions().stream()
                .filter(i -> i.op() == Op.OpTypeVoid)
                .count();
        assertEquals(1, voidDecls);
    }

    @Test
    void lowersExpressionsAndStructuredControlFlow() {
        List<Instruction> instructions = new CoreToSpirv().lower(controlFlowShader()).instructions();
        List<Op> ops = instructions.stream().map(Instruction::op).toList();
        for (Op required : List.of(Op.OpVariable, Op.OpConstant, Op.OpLoad, Op.OpStore, Op.OpIAdd,
                Op.OpSLessThan, Op.OpSelectionMerge, Op.OpLoopMerge, Op.OpBranch, Op.OpBranchConditional)) {
            assertTrue(ops.contains(required), () -> "missing " + required);
        }
    }

    @Test
    void emitsStorageBufferForStoreResult() {
        Type.Int i32 = Type.int32();
        LocalVar result = new LocalVar("result", i32);
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), Region.of(
                new Statement.DeclareVar(result, new Expr.ConstInt(i32, 7)),
                new Statement.StoreResult(new Expr.Read(result)),
                new Statement.ReturnVoid()));
        CoreModule module = new CoreModule().addEntryPoint(EntryPoint.compute(main, 1, 1, 1));

        List<Instruction> instructions = new CoreToSpirv().lower(module).instructions();
        List<Op> ops = instructions.stream().map(Instruction::op).toList();
        for (Op required : List.of(Op.OpDecorate, Op.OpMemberDecorate, Op.OpTypeStruct,
                Op.OpVariable, Op.OpAccessChain, Op.OpStore)) {
            assertTrue(ops.contains(required), () -> "missing " + required);
        }
        // Decorations must precede the type/variable definitions they target.
        assertTrue(firstIndexOf(instructions, Op.OpDecorate) < firstIndexOf(instructions, Op.OpTypeStruct));
    }

    @Test
    void hoistsVariablesAndDeduplicatesConstants() {
        List<Instruction> instructions = new CoreToSpirv().lower(controlFlowShader()).instructions();
        // Two local variables, both hoisted.
        assertEquals(2, countOp(instructions, Op.OpVariable));
        // Distinct constant values used are {0, 1, 5, 10}; the reused 0 is declared once.
        assertEquals(4, countOp(instructions, Op.OpConstant));
    }
}
