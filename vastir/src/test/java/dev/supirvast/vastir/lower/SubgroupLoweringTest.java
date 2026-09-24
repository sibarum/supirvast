package dev.supirvast.vastir.lower;

import dev.supirvast.vastir.binary.Instruction;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.Statement.SubgroupArithmetic.Scan;
import dev.supirvast.vastir.core.Statement.SubgroupShuffle;
import dev.supirvast.vastir.core.Statement.SubgroupVote;
import dev.supirvast.vastir.core.SubgroupOp;
import dev.supirvast.vastir.core.Subgroups;
import dev.supirvast.vastir.spirv.Capability;
import dev.supirvast.vastir.spirv.Op;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What the lowering emits for subgroup operations, and what it refuses. Validity and results on a device are
 * {@code SubgroupTest}'s; this pins the capability each kind needs, so a device or budget without it refuses.
 */
class SubgroupLoweringTest {

    private static final Type.Int I32 = Type.int32();
    private static final Type.FunctionType VOID_FN = new Type.FunctionType(Type.VOID, List.of());

    private static Function kernel(Statement... statements) {
        List<Statement> body = new ArrayList<>(List.of(statements));
        body.add(new Statement.ReturnVoid());
        return new Function("main", VOID_FN, new Region(body));
    }

    private static List<Instruction> lower(Function kernel, SpirvTarget target) {
        return new CoreToSpirv().lower(new CoreModule().addEntryPoint(EntryPoint.compute(kernel, 64, 1, 1)), target)
                .instructions();
    }

    private static Set<Integer> capabilities(List<Instruction> spirv) {
        Set<Integer> caps = new HashSet<>();
        for (Instruction instruction : spirv) {
            if (instruction.op() == Op.OpCapability) {
                caps.add(instruction.toWords()[1]);
            }
        }
        return caps;
    }

    private static Statement reduce(LocalVar result) {
        return new Statement.SubgroupArithmetic(result, SubgroupOp.ADD, Scan.REDUCE, new Expr.SubgroupInvocationId());
    }

    @Test
    void eachKindAsksForItsOwnCapability() {
        LocalVar sum = new LocalVar("sum", I32);
        LocalVar up = new LocalVar("up", I32);
        LocalVar pair = new LocalVar("pair", I32);
        LocalVar all = new LocalVar("all", Type.BOOL);
        List<Instruction> spirv = lower(kernel(
                reduce(sum),
                new Statement.SubgroupShuffle(up, SubgroupShuffle.Kind.UP, new Expr.Read(sum), new Expr.ConstInt(I32, 1)),
                new Statement.SubgroupShuffle(pair, SubgroupShuffle.Kind.XOR, new Expr.Read(sum), new Expr.ConstInt(I32, 1)),
                new Statement.SubgroupVote(all, SubgroupVote.Kind.ALL, new Expr.ConstBool(true))),
                SpirvTarget.unconstrained());
        Set<Integer> caps = capabilities(spirv);
        for (Capability expected : List.of(Capability.GroupNonUniform, Capability.GroupNonUniformArithmetic,
                Capability.GroupNonUniformShuffle, Capability.GroupNonUniformShuffleRelative,
                Capability.GroupNonUniformVote)) {
            assertTrue(caps.contains(expected.value()), "missing " + expected);
        }
        assertEquals(1, spirv.stream().filter(i -> i.op() == Op.OpGroupNonUniformIAdd).count());
        assertEquals(1, spirv.stream().filter(i -> i.op() == Op.OpGroupNonUniformShuffleUp).count());
    }

    /** A kernel with nothing subgroup about it declares nothing subgroup: no capability a device might lack. */
    @Test
    void aKernelWithoutThemAsksForNone() {
        Set<Integer> caps = capabilities(lower(kernel(), SpirvTarget.unconstrained()));
        assertFalse(caps.contains(Capability.GroupNonUniform.value()));
    }

    @Test
    void aBudgetWithoutTheKindRefusesIt() {
        LocalVar sum = new LocalVar("sum", I32);
        CapabilityException refused = assertThrows(CapabilityException.class, () -> lower(kernel(reduce(sum)),
                SpirvTarget.restrictedTo(Set.of(Capability.GroupNonUniform))));
        assertTrue(refused.getMessage().contains("GroupNonUniformArithmetic"), refused.getMessage());
        assertDoesNotThrow(() -> lower(kernel(reduce(sum)), SpirvTarget.restrictedTo(
                Set.of(Capability.GroupNonUniform, Capability.GroupNonUniformArithmetic))));
    }

    @Test
    void aSubgroupOperationUnderADivergentConditionIsRefused() {
        LocalVar sum = new LocalVar("sum", I32);
        Function kernel = kernel(new Statement.If(
                new Expr.Binary(BinaryOp.LESS_THAN, new Expr.SubgroupInvocationId(),
                        new Expr.ConstInt(I32, 4)),
                Region.of(reduce(sum)), Region.of()));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> lower(kernel, SpirvTarget.unconstrained()));
        assertTrue(e.getMessage().startsWith("a subgroup operation"), e.getMessage());
    }

    /** The table SPIR-V has without the extended-types feature: 32-bit only, and no bitwise ops on floats. */
    @Test
    void theOperandTypesAreSpirvs() {
        LocalVar wide = new LocalVar("wide", Type.int64());
        LocalVar f = new LocalVar("f", Type.float32());
        assertThrows(IllegalArgumentException.class, () -> new Statement.SubgroupArithmetic(wide, SubgroupOp.ADD,
                Scan.REDUCE, new Expr.ConstInt(Type.int64(), 1)));
        assertThrows(IllegalArgumentException.class, () -> new Statement.SubgroupArithmetic(f, SubgroupOp.XOR,
                Scan.REDUCE, new Expr.ConstFloat(Type.float32(), 1)));
        assertThrows(IllegalArgumentException.class, () -> new Statement.SubgroupVote(
                new LocalVar("b", Type.BOOL), SubgroupVote.Kind.ALL, new Expr.ConstInt(I32, 1)));
        assertThrows(IllegalArgumentException.class, () -> new Statement.SubgroupShuffle(
                new LocalVar("r", I32), SubgroupShuffle.Kind.INDEX, new Expr.ConstInt(I32, 1),
                new Expr.ConstFloat(Type.float32(), 0)));
    }

    /** Reading the lane alone makes a kernel's meaning its subgroups', as a reduction does. */
    @Test
    void theLaneIndexAloneCountsAsUsingSubgroups() {
        assertTrue(Subgroups.uses(Region.of(new Statement.DeclareVar(new LocalVar("l", I32),
                new Expr.SubgroupInvocationId()))));
        assertFalse(Subgroups.uses(Region.of(new Statement.Barrier())));
    }
}
