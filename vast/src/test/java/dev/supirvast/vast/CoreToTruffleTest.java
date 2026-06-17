package dev.supirvast.vast;

import com.oracle.truffle.api.CallTarget;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Executes core functions on the CPU via the generated Truffle AST and checks the computed results. */
class CoreToTruffleTest {

    private static final Type.Int I32 = Type.int32();

    private static Object run(Function function) {
        CallTarget target = new CoreToTruffle().lower(function);
        return target.call();
    }

    private static Function returning(String name, Expr value) {
        return new Function(name, new Type.FunctionType(value.type(), List.of()),
                Region.of(new Statement.Return(value)));
    }

    private static Expr i(long value) {
        return new Expr.ConstInt(I32, value);
    }

    @Test
    void evaluatesArithmeticExpressions() {
        // (2 + 3) * 4
        Expr expr = new Expr.Binary(BinaryOp.MUL, new Expr.Binary(BinaryOp.ADD, i(2), i(3)), i(4));
        assertEquals(20, run(returning("arith", expr)));
    }

    @Test
    void selectsBranchAtRuntime() {
        // int x; if (3 < 5) x = 10; else x = 20; return x;
        LocalVar x = new LocalVar("x", I32);
        Function f = new Function("branch", new Type.FunctionType(I32, List.of()), Region.of(
                new Statement.DeclareVar(x, i(0)),
                new Statement.If(new Expr.Binary(BinaryOp.LESS_THAN, i(3), i(5)),
                        Region.of(new Statement.Assign(x, i(10))),
                        Region.of(new Statement.Assign(x, i(20)))),
                new Statement.Return(new Expr.Read(x))));
        assertEquals(10, run(f));
    }

    @Test
    void callsAnotherFunction() {
        // int add(int x, int y) { return x + y; }   int compute() { return add(20, 25); }  ==> 45
        Function add = new Function("add", new Type.FunctionType(I32, List.of(I32, I32)),
                Region.of(new Statement.Return(
                        new Expr.Binary(BinaryOp.ADD, new Expr.Param(0, I32), new Expr.Param(1, I32)))));
        Function compute = new Function("compute", new Type.FunctionType(I32, List.of()),
                Region.of(new Statement.Return(new Expr.Call(add, List.of(i(20), i(25))))));
        assertEquals(45, new CoreToTruffle().lowerModule(List.of(add, compute), compute).call());
    }

    @Test
    void runsLoopToComputeSum() {
        // int acc=0, i=0; while (i < 10) { acc = acc + i; i = i + 1; } return acc;  ==> 45
        LocalVar acc = new LocalVar("acc", I32);
        LocalVar i = new LocalVar("i", I32);
        Region loop = Region.of(
                new Statement.Assign(acc, new Expr.Binary(BinaryOp.ADD, new Expr.Read(acc), new Expr.Read(i))),
                new Statement.Assign(i, new Expr.Binary(BinaryOp.ADD, new Expr.Read(i), i(1))));
        Function f = new Function("sum", new Type.FunctionType(I32, List.of()), Region.of(
                new Statement.DeclareVar(acc, i(0)),
                new Statement.DeclareVar(i, i(0)),
                new Statement.While(new Expr.Binary(BinaryOp.LESS_THAN, new Expr.Read(i), i(10)), loop),
                new Statement.Return(new Expr.Read(acc))));
        assertEquals(45, run(f));
    }
}
