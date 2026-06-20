package dev.supirvast.supir;

import dev.supirvast.vastir.core.Builtin;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The core witnesses that Supir is a faithful, two-way textual form of the {@code core} IR:
 *
 * <ul>
 *   <li><b>Print is a normal form.</b> {@code print(parse(print(m))) == print(m)} — printing canonicalizes,
 *       so a parse/print round-trip is a fixpoint. This is the right invariant for flat assembly, which
 *       normalizes nested expressions into named temporaries rather than mirroring the AST tree.</li>
 *   <li><b>Parsed text lowers to valid SPIR-V.</b> The module a tool would hand to {@code CoreToSpirv} is
 *       well-formed (magic number, word-aligned) — checked without the native toolchain.</li>
 * </ul>
 */
class RoundTripTest {

    private static final Type.Float F32 = Type.float32();
    private static final Type.Vector VEC4 = new Type.Vector(F32, 4);

    private static Expr f(double v) {
        return new Expr.ConstFloat(F32, v);
    }

    /** print → parse → print is a fixpoint, and the parsed module lowers to valid SPIR-V. */
    private static void assertFixpoint(CoreModule module) {
        String once = Supir.print(module);
        CoreModule reparsed = Supir.parseModule(once);
        String twice = Supir.print(reparsed);
        assertEquals(once, twice, "print is not a normal form:\n--- once ---\n" + once + "\n--- twice ---\n" + twice);
        assertValidSpirv(reparsed);
    }

    @Test
    void vertexShaderRoundTrips() {
        Expr position = new Expr.VectorConstruct(VEC4, List.of(
                new Expr.Convert(new Expr.BuiltinRead(Builtin.VERTEX_INDEX), F32), f(0), f(0), f(1)));
        InterfaceVar vColor = InterfaceVar.output("vColor", 0, VEC4);
        Region body = Region.of(
                new Statement.BuiltinWrite(Builtin.POSITION, position),
                new Statement.InterfaceWrite(vColor,
                        new Expr.VectorConstruct(VEC4, List.of(f(1), f(0), f(0), f(1)))),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        assertFixpoint(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.VERTEX)));
    }

    @Test
    void fragmentShaderRoundTrips() {
        InterfaceVar vColor = InterfaceVar.input("vColor", 0, VEC4);
        InterfaceVar fragColor = InterfaceVar.output("fragColor", 0, VEC4);
        Region body = Region.of(
                new Statement.InterfaceWrite(fragColor, new Expr.InterfaceRead(vColor)),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        assertFixpoint(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.FRAGMENT)));
    }

    @Test
    void loopReductionRoundTrips() {
        // The While case: a single inline comparison condition, re-evaluated each iteration.
        Type.Int i32 = Type.int32();
        LocalVar sum = new LocalVar("sum", i32);
        LocalVar i = new LocalVar("i", i32);
        Region loop = Region.of(
                new Statement.Assign(sum, new Expr.Binary(
                        dev.supirvast.vastir.core.BinaryOp.ADD, new Expr.Read(sum), new Expr.Read(i))),
                new Statement.Assign(i, new Expr.Binary(
                        dev.supirvast.vastir.core.BinaryOp.ADD, new Expr.Read(i), new Expr.ConstInt(i32, 1))));
        Region body = Region.of(
                new Statement.DeclareVar(sum, new Expr.ConstInt(i32, 0)),
                new Statement.DeclareVar(i, new Expr.ConstInt(i32, 0)),
                new Statement.While(new Expr.Binary(
                        dev.supirvast.vastir.core.BinaryOp.LESS_THAN, new Expr.Read(i),
                        new Expr.ConstInt(i32, 10)), loop),
                new Statement.StoreResult(new Expr.Read(sum)),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        assertFixpoint(new CoreModule().addEntryPoint(EntryPoint.compute(main, 1, 1, 1)));
    }

    @Test
    void functionCallAndConvertRoundTrip() {
        // A standalone function plus an entry that calls it — exercises params, call, and the fn-before-entry
        // print order.
        Type.Int i32 = Type.int32();
        LocalVar x = new LocalVar("x", i32);
        Function dbl = new Function("dbl", new Type.FunctionType(i32, List.of(i32)), Region.of(
                new Statement.Return(new Expr.Binary(
                        dev.supirvast.vastir.core.BinaryOp.MUL, new Expr.Param(0, i32), new Expr.Param(0, i32)))));
        Region body = Region.of(
                new Statement.DeclareVar(x, new Expr.Call(dbl, List.of(new Expr.ConstInt(i32, 21)))),
                new Statement.StoreResult(new Expr.Read(x)),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        CoreModule module = new CoreModule()
                .addFunction(dbl)
                .addEntryPoint(EntryPoint.compute(main, 1, 1, 1));
        assertFixpoint(module);
    }

    @Test
    void textFirstRoundTrips() {
        // Starting from hand-written text, parse → print → parse → print is a fixpoint too.
        String source = """
                compute kernel local_size(64, 1, 1) {
                  buffer data: i32 @binding 0
                  i = invocation_id
                  v = data[i]
                  doubled = mul v, 2
                  data[i] = doubled
                  ret
                }""";
        CoreModule parsed = Supir.parseModule(source);
        assertFixpoint(parsed);
    }

    private static void assertValidSpirv(CoreModule module) {
        byte[] spirv = new CoreToSpirv().lower(module).toByteArray();
        assertTrue(spirv.length > 20 && spirv.length % 4 == 0, "implausible SPIR-V length " + spirv.length);
        int magic = (spirv[0] & 0xFF) | (spirv[1] & 0xFF) << 8 | (spirv[2] & 0xFF) << 16 | (spirv[3] & 0xFF) << 24;
        assertEquals(0x07230203, magic, "missing SPIR-V magic number");
    }
}
