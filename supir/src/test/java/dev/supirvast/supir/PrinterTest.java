package dev.supirvast.supir;

import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Builtin;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Printing specifics: nested expressions flatten to named temporaries, and the output reads as expected. */
class PrinterTest {

    private static final Type.Float F32 = Type.float32();
    private static final Type.Vector VEC4 = new Type.Vector(F32, 4);

    @Test
    void nestedExpressionFlattensToTemporaries() {
        // gl_Position = vec4(f32(VertexIndex), 0, 0, 1) — the Convert is nested inside the VectorConstruct,
        // so it must be hoisted to its own named line before the vec4.
        Expr position = new Expr.VectorConstruct(VEC4, List.of(
                new Expr.Convert(new Expr.BuiltinRead(Builtin.VERTEX_INDEX), F32),
                new Expr.ConstFloat(F32, 0), new Expr.ConstFloat(F32, 0), new Expr.ConstFloat(F32, 1)));
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()),
                Region.of(new Statement.BuiltinWrite(Builtin.POSITION, position), new Statement.ReturnVoid()));
        String text = Supir.print(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.VERTEX)));

        assertTrue(text.contains("vertex main {"), text);
        assertTrue(text.contains("t0 = convert VertexIndex, f32"), () -> "expected a hoisted convert temp:\n" + text);
        assertTrue(text.contains("Position = vec4 t0, 0.0, 0.0, 1.0"),
                () -> "expected the flattened vec4 referencing the temp:\n" + text);
    }

    @Test
    void resourceDeclarationsArePrintedFromUsage() {
        // The fragment shader references vColor (in) and fragColor (out) only inside statements; the printer
        // must recover and emit their declarations at the top of the body.
        InterfaceVar vColor = InterfaceVar.input("vColor", 0, VEC4);
        InterfaceVar fragColor = InterfaceVar.output("fragColor", 0, VEC4);
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), Region.of(
                new Statement.InterfaceWrite(fragColor, new Expr.InterfaceRead(vColor)),
                new Statement.ReturnVoid()));
        String text = Supir.print(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.FRAGMENT)));

        assertTrue(text.contains("in vColor: vec4 @loc 0"), text);
        assertTrue(text.contains("out fragColor: vec4 @loc 0"), text);
        assertTrue(text.contains("fragColor = vColor"), text);
    }

    @Test
    void floatConstantsKeepADecimalPoint() {
        // A float whose value is integral (1.0) must print with a '.' so it re-lexes as a float, not an int.
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), Region.of(
                new Statement.BuiltinWrite(Builtin.POSITION, new Expr.VectorConstruct(VEC4, List.of(
                        new Expr.ConstFloat(F32, 1), new Expr.ConstFloat(F32, 0),
                        new Expr.ConstFloat(F32, 0), new Expr.ConstFloat(F32, 1)))),
                new Statement.ReturnVoid()));
        String text = Supir.print(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.VERTEX)));
        assertTrue(text.contains("1.0") && text.contains("0.0"), () -> "floats lost their decimal point:\n" + text);
    }

    @Test
    void integerComparisonMnemonic() {
        Type.Int i32 = Type.int32();
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), Region.of(
                new Statement.StoreResult(new Expr.Binary(BinaryOp.LESS_THAN,
                        new Expr.ConstInt(i32, 1), new Expr.ConstInt(i32, 2))),
                new Statement.ReturnVoid()));
        // store_result needs an atom, so the comparison is hoisted to a temp first.
        String text = Supir.print(new CoreModule().addEntryPoint(EntryPoint.compute(main, 1, 1, 1)));
        assertTrue(text.contains("t0 = lt 1, 2"), text);
        assertTrue(text.contains("store_result t0"), text);
    }

    @Test
    void computeHeaderShowsWorkgroupSize() {
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()),
                Region.of(new Statement.ReturnVoid()));
        String text = Supir.print(new CoreModule().addEntryPoint(EntryPoint.compute(main, 8, 4, 2)));
        assertTrue(text.contains("compute main local_size(8, 4, 2) {"), text);
    }
}
