package dev.supirvast.vastir.core;

import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Where a barrier may stand. Each rejection here is a kernel that would hang or answer wrongly on the GPU and
 * have no meaning on the CPU; each acceptance is a shape real kernels use.
 */
class BarriersTest {

    private static final Type.Int I32 = Type.int32();
    private static final Buffer DATA = new Buffer("data", 0, I32);

    private static Function kernel(Statement... body) {
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), Region.of(body));
    }

    private static Expr i(long value) {
        return new Expr.ConstInt(I32, value);
    }

    private static Expr lessThan(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.LESS_THAN, a, b);
    }

    private static void rejected(Function kernel, String because) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> Barriers.check(kernel));
        assertTrue(e.getMessage().contains(because), e.getMessage());
    }

    @Test
    void aBarrierAtTheTopIsUniform() {
        assertDoesNotThrow(() -> Barriers.check(kernel(new Statement.Barrier(), new Statement.ReturnVoid())));
    }

    /** The tree reduction: a stride halved from a constant, a barrier every round. */
    @Test
    void aLoopCountedFromConstantsIsUniform() {
        LocalVar stride = new LocalVar("stride", I32);
        Function reduction = kernel(
                new Statement.DeclareVar(stride, i(32)),
                new Statement.While(new Expr.Binary(BinaryOp.GREATER_THAN, new Expr.Read(stride), i(0)), Region.of(
                        new Statement.If(lessThan(new Expr.LocalInvocationId(), new Expr.Read(stride)),
                                Region.of(new Statement.BufferStore(DATA, new Expr.InvocationId(), i(1))),
                                Region.of()),
                        new Statement.Barrier(),
                        new Statement.Assign(stride, new Expr.Binary(BinaryOp.SHIFT_RIGHT, new Expr.Read(stride), i(1))))));
        assertDoesNotThrow(() -> Barriers.check(reduction));
    }

    @Test
    void theWorkgroupIdAndTheCountAreUniform() {
        Function kernel = kernel(new Statement.If(
                lessThan(new Expr.Binary(BinaryOp.MUL, new Expr.WorkgroupId(), i(64)), new Expr.InvocationCount()),
                Region.of(new Statement.Barrier()), Region.of()));
        assertDoesNotThrow(() -> Barriers.check(kernel));
    }

    @Test
    void aConditionOnTheInvocationIsNot() {
        rejected(kernel(new Statement.If(lessThan(new Expr.LocalInvocationId(), i(16)),
                Region.of(new Statement.Barrier()), Region.of())), "under the condition");
    }

    @Test
    void aConditionOnMemoryIsNot() {
        rejected(kernel(new Statement.While(lessThan(new Expr.BufferLoad(DATA, i(0)), i(4)),
                Region.of(new Statement.Barrier()))), "under the condition");
    }

    /** A variable assigned an invocation index once is not uniform, however it is used afterwards. */
    @Test
    void aVariableTaintedAnywhereIsNot() {
        LocalVar n = new LocalVar("n", I32);
        rejected(kernel(
                new Statement.DeclareVar(n, i(4)),
                new Statement.Assign(n, new Expr.Binary(BinaryOp.ADD, new Expr.Read(n), new Expr.InvocationId())),
                new Statement.If(lessThan(i(0), new Expr.Read(n)), Region.of(new Statement.Barrier()), Region.of())),
                "under the condition");
    }

    /** Assigned a constant, but only by the invocations one branch runs. */
    @Test
    void aVariableAssignedUnderDivergenceIsNot() {
        LocalVar n = new LocalVar("n", I32);
        rejected(kernel(
                new Statement.DeclareVar(n, i(0)),
                new Statement.If(lessThan(new Expr.InvocationId(), i(8)),
                        Region.of(new Statement.Assign(n, i(1))), Region.of()),
                new Statement.If(new Expr.Binary(BinaryOp.EQUAL, new Expr.Read(n), i(1)),
                        Region.of(new Statement.Barrier()), Region.of())),
                "under the condition");
    }

    /** The tail guard the accelerator adds: exactly why a barrier kernel is not guarded that way. */
    @Test
    void aBarrierAfterADivergentReturnIsNot() {
        rejected(kernel(
                new Statement.If(lessThan(new Expr.InvocationId(), new Expr.InvocationCount()),
                        Region.of(), Region.of(new Statement.ReturnVoid())),
                new Statement.Barrier()),
                "follows a return");
    }

    /**
     * The return is after the barrier in the text, and before it one iteration later. Reported as whichever
     * it meets first: here the counter, assigned only by the invocations that did not return.
     */
    @Test
    void aDivergentReturnLaterInALoopBodyIsBeforeItsBarrierNextTime() {
        LocalVar k = new LocalVar("k", I32);
        rejected(kernel(
                new Statement.DeclareVar(k, i(0)),
                new Statement.While(lessThan(new Expr.Read(k), i(4)), Region.of(
                        new Statement.Barrier(),
                        new Statement.If(lessThan(new Expr.LocalInvocationId(), new Expr.Read(k)),
                                Region.of(new Statement.ReturnVoid()), Region.of()),
                        new Statement.Assign(k, new Expr.Binary(BinaryOp.ADD, new Expr.Read(k), i(1)))))),
                "not in uniform control flow");
    }

    /** The same return with nothing after it in the body: now the return itself is what is reported. */
    @Test
    void aDivergentReturnAtTheEndOfALoopBodyIsReportedAsSuch() {
        rejected(kernel(new Statement.While(new Expr.ConstBool(true), Region.of(
                new Statement.Barrier(),
                new Statement.If(lessThan(new Expr.LocalInvocationId(), i(3)),
                        Region.of(new Statement.ReturnVoid()), Region.of())))),
                "follows a return");
    }

    @Test
    void aUniformReturnIsFine() {
        assertDoesNotThrow(() -> Barriers.check(kernel(
                new Statement.If(lessThan(new Expr.InvocationCount(), i(1)),
                        Region.of(new Statement.ReturnVoid()), Region.of()),
                new Statement.Barrier())));
    }

    @Test
    void containsLooksInsideRegions() {
        assertTrue(Barriers.contains(Region.of(new Statement.While(new Expr.ConstBool(false),
                Region.of(new Statement.If(new Expr.ConstBool(true), Region.of(), Region.of(new Statement.Barrier())))))));
        assertFalse(Barriers.contains(Region.of(new Statement.ReturnVoid())));
    }
}
