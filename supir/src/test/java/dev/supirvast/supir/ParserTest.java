package dev.supirvast.supir;

import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Builtin;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.MathFn;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.UnaryOp;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

/** Parsing: flat-assembly forms produce the expected {@code core} records with the expected fields. */
class ParserTest {

    /** Statements of a fragment entry body, with the given lines spliced in. */
    private static List<Statement> body(String... lines) {
        String src = "fragment main {\n" + String.join("\n", lines) + "\n}";
        return Parser.parse(src).functions().getFirst().body().statements();
    }

    /** The initializer of the first statement, written as {@code x = <rhs>}. */
    private static Expr exprOf(String rhs) {
        return ((Statement.DeclareVar) body("x = " + rhs).getFirst()).initializer();
    }

    @Test
    void entryStageAndName() {
        CoreModule m = Parser.parse("vertex main {\nret\n}");
        EntryPoint ep = m.entryPoints().getFirst();
        assertEquals(ShaderStage.VERTEX, ep.stage());
        assertEquals("main", ep.function().name());
    }

    @Test
    void computeEntryCarriesWorkgroupSize() {
        CoreModule m = Parser.parse("compute main local_size(64, 2, 1) {\nret\n}");
        EntryPoint ep = m.entryPoints().getFirst();
        assertEquals(ShaderStage.COMPUTE, ep.stage());
        EntryPoint.WorkgroupSize ws = ep.workgroupSize().orElseThrow();
        assertEquals(64, ws.x());
        assertEquals(2, ws.y());
        assertEquals(1, ws.z());
    }

    @Test
    void literalsDefaultToI32AndF32() {
        Expr.ConstInt i = assertInstanceOf(Expr.ConstInt.class, exprOf("7"));
        assertEquals(Type.int32(), i.type());
        assertEquals(7L, i.value());
        Expr.ConstFloat f = assertInstanceOf(Expr.ConstFloat.class, exprOf("2.5"));
        assertEquals(Type.float32(), f.type());
        assertEquals(2.5, f.value());
    }

    @Test
    void scalarFormIsConstantForLiteralAndConvertForName() {
        assertEquals(Type.uint32(), assertInstanceOf(Expr.ConstInt.class, exprOf("u32 5")).type());
        Expr.Convert convert = assertInstanceOf(Expr.Convert.class, exprOf("f32 VertexIndex"));
        assertEquals(Type.float32(), convert.type());
        assertInstanceOf(Expr.BuiltinRead.class, convert.operand());
    }

    @Test
    void binaryUnaryAndMath() {
        assertEquals(BinaryOp.ADD, assertInstanceOf(Expr.Binary.class, exprOf("add 1, 2")).op());
        assertEquals(BinaryOp.LESS_THAN, assertInstanceOf(Expr.Binary.class, exprOf("lt 1, 2")).op());
        assertEquals(UnaryOp.NEGATE, assertInstanceOf(Expr.Unary.class, exprOf("neg 1")).op());
        // Operands must be atoms, so the vector is a prior named line.
        Statement.DeclareVar n = assertInstanceOf(Statement.DeclareVar.class,
                body("v = vec3 1.0, 2.0, 3.0", "n = normalize v").get(1));
        Expr.MathCall math = assertInstanceOf(Expr.MathCall.class, n.initializer());
        assertEquals(MathFn.NORMALIZE, math.fn());
        assertEquals(new Type.Vector(Type.float32(), 3), math.type());
    }

    @Test
    void vectorConstructDefaultsToFloatElements() {
        Expr.VectorConstruct v = assertInstanceOf(Expr.VectorConstruct.class, exprOf("vec4 1.0, 2.0, 3.0, 4.0"));
        assertEquals(new Type.Vector(Type.float32(), 4), v.type());
        assertEquals(4, v.components().size());
    }

    @Test
    void typedVectorAndGeneralVector() {
        Expr.VectorConstruct iv = assertInstanceOf(Expr.VectorConstruct.class, exprOf("vec3<i32> 1, 2, 3"));
        assertEquals(new Type.Vector(Type.int32(), 3), iv.type());
        Expr.VectorConstruct gv = assertInstanceOf(Expr.VectorConstruct.class, exprOf("vec<f32, 5> 1.0, 2.0, 3.0, 4.0, 5.0"));
        assertEquals(new Type.Vector(Type.float32(), 5), gv.type());
    }

    @Test
    void builtinReadAndWrite() {
        List<Statement> stmts = body("idx = VertexIndex", "Position = vec4 1.0, 0.0, 0.0, 1.0");
        Statement.DeclareVar read = assertInstanceOf(Statement.DeclareVar.class, stmts.get(0));
        assertInstanceOf(Expr.BuiltinRead.class, read.initializer());
        Statement.BuiltinWrite write = assertInstanceOf(Statement.BuiltinWrite.class, stmts.get(1));
        assertEquals(Builtin.POSITION, write.builtin());
    }

    @Test
    void interfaceDeclarationReadAndWrite() {
        List<Statement> stmts = body("in vColor: vec4 @loc 0", "out fragColor: vec4 @loc 0",
                "fragColor = vColor");
        Statement.InterfaceWrite w = assertInstanceOf(Statement.InterfaceWrite.class, stmts.getFirst());
        assertEquals("fragColor", w.variable().name());
        assertEquals(InterfaceVar.Direction.OUTPUT, w.variable().direction());
        Expr.InterfaceRead r = assertInstanceOf(Expr.InterfaceRead.class, w.value());
        assertEquals("vColor", r.variable().name());
        assertEquals(InterfaceVar.Direction.INPUT, r.variable().direction());
    }

    @Test
    void bufferLoadAndStore() {
        // The `buffer` line is a declaration, not a statement, so it does not appear in the statement list.
        List<Statement> stmts = body("buffer data: i32 @binding 0", "x = data[3]", "data[3] = x");
        Statement.DeclareVar load = assertInstanceOf(Statement.DeclareVar.class, stmts.get(0));
        Expr.BufferLoad bl = assertInstanceOf(Expr.BufferLoad.class, load.initializer());
        assertEquals("data", bl.buffer().name());
        Statement.BufferStore bs = assertInstanceOf(Statement.BufferStore.class, stmts.get(1));
        assertEquals("data", bs.buffer().name());
    }

    @Test
    void declareThenReassign() {
        List<Statement> stmts = body("x = 1", "x = 2");
        Statement.DeclareVar declare = assertInstanceOf(Statement.DeclareVar.class, stmts.get(0));
        Statement.Assign assign = assertInstanceOf(Statement.Assign.class, stmts.get(1));
        assertEquals(declare.variable(), assign.variable(), "reassignment must resolve to the same LocalVar");
    }

    @Test
    void typedLocalDeclaration() {
        Statement.DeclareVar dv = assertInstanceOf(Statement.DeclareVar.class, body("x: f64 = 1.0").getFirst());
        assertEquals(Type.float64(), dv.variable().type());
    }

    @Test
    void ifElseAndLoop() {
        Statement.If branch = assertInstanceOf(Statement.If.class,
                body("if true {", "ret", "} else {", "ret", "}").getFirst());
        assertEquals(1, branch.thenRegion().statements().size());
        assertEquals(1, branch.elseRegion().statements().size());

        Statement.While loop = assertInstanceOf(Statement.While.class,
                body("i = 0", "loop while lt i, 10 {", "i = add i, 1", "}").get(1));
        Expr.Binary cond = assertInstanceOf(Expr.Binary.class, loop.condition());
        assertEquals(BinaryOp.LESS_THAN, cond.op());
    }

    @Test
    void functionParamsAndCalls() {
        String src = """
                fn dbl(x: f32) -> f32 {
                  r = mul x, x
                  ret r
                }
                fragment main {
                  out color: vec4 @loc 0
                  y = call dbl, 2.0
                  color = vec4 y, y, y, 1.0
                  ret
                }""";
        CoreModule m = Parser.parse(src);
        Function dbl = m.functions().stream().filter(f -> f.name().equals("dbl")).findFirst().orElseThrow();
        Statement.DeclareVar r = assertInstanceOf(Statement.DeclareVar.class, dbl.body().statements().getFirst());
        Expr.Binary mul = assertInstanceOf(Expr.Binary.class, r.initializer());
        Expr.Param p = assertInstanceOf(Expr.Param.class, mul.lhs());
        assertEquals(0, p.index());

        Function main = m.functions().stream().filter(f -> f.name().equals("main")).findFirst().orElseThrow();
        Statement.DeclareVar y = assertInstanceOf(Statement.DeclareVar.class, main.body().statements().getFirst());
        Expr.Call call = assertInstanceOf(Expr.Call.class, y.initializer());
        assertEquals(dbl, call.callee());
        assertEquals(1, call.arguments().size());
    }
}
