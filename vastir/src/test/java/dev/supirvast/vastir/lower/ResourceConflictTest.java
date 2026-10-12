package dev.supirvast.vastir.lower;

import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.Texture;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A shader assembled from parts written apart can declare one location or binding twice. The same declaration twice
 * is one resource; two different ones at the same place are refused, naming both, rather than one silently winning.
 */
class ResourceConflictTest {

    private static final Type.Float F32 = Type.float32();
    private static final Type.Vector VEC2 = new Type.Vector(F32, 2);
    private static final Type.Vector VEC4 = new Type.Vector(F32, 4);
    private static final InterfaceVar FRAG_COLOR = InterfaceVar.output("fragColor", 0, VEC4);

    private static CoreModule fragmentWriting(Expr colour) {
        Region body = Region.of(new Statement.InterfaceWrite(FRAG_COLOR, colour), new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.FRAGMENT));
    }

    private static Expr splat(Expr x) {
        return new Expr.VectorConstruct(VEC4, List.of(x, x, x, x));
    }

    private static Expr sum(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.ADD, a, b);
    }

    @Test
    void theSameDeclarationTwiceIsOneResource() {
        Buffer a = new Buffer("styles", 1, F32);
        Buffer again = new Buffer("styles", 1, F32);
        Expr zero = new Expr.ConstInt(Type.int32(), 0);
        assertDoesNotThrow(() -> new CoreToSpirv().lower(fragmentWriting(
                splat(sum(new Expr.BufferLoad(a, zero), new Expr.BufferLoad(again, zero))))));
    }

    @Test
    void twoBuffersAtOneBindingAreRefused() {
        Buffer styles = new Buffer("styles", 1, F32);
        Buffer looks = new Buffer("looks", 1, Type.int32());
        Expr zero = new Expr.ConstInt(Type.int32(), 0);
        Expr read = sum(new Expr.BufferLoad(styles, zero), new Expr.Convert(new Expr.BufferLoad(looks, zero), F32));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new CoreToSpirv().lower(fragmentWriting(splat(read))));
        assertTrue(e.getMessage().contains("styles") && e.getMessage().contains("looks"), e::getMessage);
    }

    @Test
    void twoInputsAtOneLocationAreRefused() {
        InterfaceVar uv = InterfaceVar.input("vUv", 1, VEC2);
        InterfaceVar local = InterfaceVar.input("vLocal", 1, VEC2);
        Expr read = sum(new Expr.VectorExtract(new Expr.InterfaceRead(uv), 0),
                new Expr.VectorExtract(new Expr.InterfaceRead(local), 0));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new CoreToSpirv().lower(fragmentWriting(splat(read))));
        assertTrue(e.getMessage().contains("vUv") && e.getMessage().contains("vLocal"), e::getMessage);
    }

    @Test
    void twoTexturesAtOneBindingAreRefused() {
        Texture atlas = new Texture("uAtlas", 0, 0);
        Texture icons = new Texture("uIcons", 0, 0);
        Expr uv = new Expr.VectorConstruct(VEC2, List.of(new Expr.ConstFloat(F32, 0), new Expr.ConstFloat(F32, 0)));
        Expr read = new Expr.Binary(BinaryOp.ADD, new Expr.SampleTexture(atlas, uv), new Expr.SampleTexture(icons, uv));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new CoreToSpirv().lower(fragmentWriting(read)));
        assertTrue(e.getMessage().contains("uAtlas") && e.getMessage().contains("uIcons"), e::getMessage);
    }
}
