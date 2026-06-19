package dev.supirvast.vastir.pbr;

import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Texture;
import dev.supirvast.vastir.type.Type;

import java.util.List;

/**
 * Small expression-building conveniences for authoring surface functions and the lighting model: float/vector
 * constructors and scalar/componentwise arithmetic over {@code core} {@link Expr}s. Pairs with the
 * {@link dev.supirvast.vastir.core.Expr.MathCall} factories ({@code dot}, {@code normalize}, {@code pow}, …) for
 * the rest. The IR has no vector·scalar broadcast, so {@link #scale} / {@link #divScalar} splat the scalar to a
 * vector first.
 */
public final class Shade {

    public static final Type.Float F32 = Type.float32();
    public static final Type.Vector VEC2 = new Type.Vector(F32, 2);
    public static final Type.Vector VEC3 = new Type.Vector(F32, 3);
    public static final Type.Vector VEC4 = new Type.Vector(F32, 4);

    private Shade() {
    }

    /** A float (f32) constant. */
    public static Expr f(double value) {
        return new Expr.ConstFloat(F32, value);
    }

    /** A constant {@code vec3}. */
    public static Expr vec3(double x, double y, double z) {
        return new Expr.VectorConstruct(VEC3, List.of(f(x), f(y), f(z)));
    }

    /** A {@code vec3} from three scalar expressions. */
    public static Expr vec3(Expr x, Expr y, Expr z) {
        return new Expr.VectorConstruct(VEC3, List.of(x, y, z));
    }

    /** A {@code vec4} from a {@code vec3} and a scalar w (e.g. a color + alpha). */
    public static Expr vec4(Expr xyz, Expr w) {
        return new Expr.VectorConstruct(VEC4, List.of(xyz, w));
    }

    /** Replicates a scalar across three components. */
    public static Expr splat3(Expr scalar) {
        return new Expr.VectorConstruct(VEC3, List.of(scalar, scalar, scalar));
    }

    public static Expr add(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.ADD, a, b);
    }

    public static Expr sub(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.SUB, a, b);
    }

    public static Expr mul(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.MUL, a, b);
    }

    public static Expr div(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.DIV, a, b);
    }

    /** A {@code vec3} scaled by a scalar (splatting the scalar, since the IR has no broadcast). */
    public static Expr scale(Expr vector, Expr scalar) {
        return mul(vector, splat3(scalar));
    }

    /** A {@code vec3} divided by a scalar. */
    public static Expr divScalar(Expr vector, Expr scalar) {
        return div(vector, splat3(scalar));
    }

    /** Samples a 2D texture (descriptor set 0, given binding) at {@code uv}, yielding the RGBA {@code vec4}. */
    public static Expr sample(String name, int binding, Expr uv) {
        return new Expr.SampleTexture(new Texture(name, binding), uv);
    }

    /** Samples a cubemap (descriptor set 0, given binding) along {@code direction}, yielding the RGBA {@code vec4}. */
    public static Expr sampleCube(String name, int binding, Expr direction) {
        return new Expr.SampleTexture(Texture.cube(name, binding), direction);
    }

    /** The {@code xyz} of a {@code vec4} as a {@code vec3} (the IR has no swizzles). */
    public static Expr xyz(Expr vec4) {
        return new Expr.VectorConstruct(VEC3, List.of(
                new Expr.VectorExtract(vec4, 0), new Expr.VectorExtract(vec4, 1), new Expr.VectorExtract(vec4, 2)));
    }
}
