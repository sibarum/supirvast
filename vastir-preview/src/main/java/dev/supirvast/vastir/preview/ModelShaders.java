package dev.supirvast.vastir.preview;

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
import dev.supirvast.vastir.shader.ShaderSource;
import dev.supirvast.vastir.type.Type;

import java.util.List;

/**
 * The previewer's default model shader pair, published as build-discoverable {@link ShaderSource}s: the
 * {@code compile-shaders} goal of the supirvast-maven-plugin lowers these to SPIR-V during the build and
 * packages the binaries as classpath resources ({@code META-INF/supirvast/shaders/model.{vert,frag}.spv}),
 * so runtime consumers load them via {@code Shaders.load("model.vert")} without re-lowering.
 */
public final class ModelShaders {

    private static final Type.Float F32 = Type.float32();
    private static final Type.Vector VEC3 = new Type.Vector(F32, 3);
    private static final Type.Vector VEC4 = new Type.Vector(F32, 4);

    private ModelShaders() {
    }

    /** {@code gl_Position = vec4(position, 1.0); vNormal = normal;} */
    public static final class Vertex implements ShaderSource {
        @Override
        public String name() {
            return "model.vert";
        }

        @Override
        public CoreModule module() {
            InterfaceVar position = InterfaceVar.input("position", 0, VEC3);
            InterfaceVar normal = InterfaceVar.input("normal", 1, VEC3);
            InterfaceVar vNormal = InterfaceVar.output("vNormal", 0, VEC3);
            Expr clip = new Expr.VectorConstruct(VEC4,
                    List.of(new Expr.InterfaceRead(position), new Expr.ConstFloat(F32, 1.0)));
            Region body = Region.of(
                    new Statement.BuiltinWrite(Builtin.POSITION, clip),
                    new Statement.InterfaceWrite(vNormal, new Expr.InterfaceRead(normal)),
                    new Statement.ReturnVoid());
            Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
            return new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.VERTEX));
        }
    }

    /** {@code fragColor = vec4(vNormal * 0.5 + 0.5, 1.0);} — maps the [-1,1] normal into a [0,1] color. */
    public static final class Fragment implements ShaderSource {
        @Override
        public String name() {
            return "model.frag";
        }

        @Override
        public CoreModule module() {
            InterfaceVar vNormal = InterfaceVar.input("vNormal", 0, VEC3);
            InterfaceVar fragColor = InterfaceVar.output("fragColor", 0, VEC4);
            Expr half = new Expr.VectorConstruct(VEC3, List.of(
                    new Expr.ConstFloat(F32, 0.5), new Expr.ConstFloat(F32, 0.5),
                    new Expr.ConstFloat(F32, 0.5)));
            Expr scaled = new Expr.Binary(BinaryOp.MUL, new Expr.InterfaceRead(vNormal), half);
            Expr biased = new Expr.Binary(BinaryOp.ADD, scaled, half);
            Expr color = new Expr.VectorConstruct(VEC4,
                    List.of(biased, new Expr.ConstFloat(F32, 1.0)));
            Region body = Region.of(
                    new Statement.InterfaceWrite(fragColor, color),
                    new Statement.ReturnVoid());
            Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
            return new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.FRAGMENT));
        }
    }
}
