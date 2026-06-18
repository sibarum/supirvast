package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.UnaryOp;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Layer-A de-risk for PBR authoring: confirm the {@code core} IR can express the math a microfacet BRDF needs —
 * {@code dot} ({@code OpDot}) and the {@code GLSL.std.450} extended instructions (normalize, pow, sqrt,
 * inversesqrt, reflect, abs, min, max, clamp, mix, cross, length). A fragment shader exercising all of them
 * must validate with {@code spirv-val}, and the disassembly must show the {@code GLSL.std.450} import being used
 * via {@code OpExtInst} alongside {@code OpDot}.
 */
class MathIntrinsicsShaderTest {

    private static final Type.Float F32 = Type.float32();
    private static final Type.Vector VEC3 = new Type.Vector(F32, 3);
    private static final Type.Vector VEC4 = new Type.Vector(F32, 4);

    private static Expr f(double v) {
        return new Expr.ConstFloat(F32, v);
    }

    private static Expr vec3(double x, double y, double z) {
        return new Expr.VectorConstruct(VEC3, List.of(f(x), f(y), f(z)));
    }

    private static Expr splat3(Expr scalar) {
        return new Expr.VectorConstruct(VEC3, List.of(scalar, scalar, scalar));
    }

    private static Expr mul(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.MUL, a, b);
    }

    private static Expr add(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.ADD, a, b);
    }

    /** A fragment shader that touches every intrinsic and folds them all into the output color. */
    private static byte[] intrinsicsShader() {
        InterfaceVar vNormal = InterfaceVar.input("vNormal", 0, VEC3);
        InterfaceVar fragColor = InterfaceVar.output("fragColor", 0, VEC4);

        Expr n = Expr.MathCall.normalize(new Expr.InterfaceRead(vNormal));
        Expr l = vec3(0.577, 0.577, 0.577);
        Expr h = Expr.MathCall.normalize(add(l, vec3(0, 0, 1)));

        Expr ndotl = Expr.MathCall.max(Expr.MathCall.dot(n, l), f(0));            // DOT, MAX
        Expr ndoth = Expr.MathCall.clamp(Expr.MathCall.dot(n, h), f(0), f(1));    // DOT, CLAMP
        Expr spec = Expr.MathCall.min(Expr.MathCall.pow(ndoth, f(32)), f(1));     // POW, MIN
        Expr r = Expr.MathCall.reflect(new Expr.Unary(UnaryOp.NEGATE, l), n);     // REFLECT
        Expr extras = add(add(Expr.MathCall.length(r),                            // LENGTH
                Expr.MathCall.inverseSqrt(add(Expr.MathCall.dot(r, r), f(1)))),   // INVERSE_SQRT, DOT
                Expr.MathCall.sqrt(spec));                                        // SQRT
        Expr rim = mul(Expr.MathCall.cross(n, l),                                 // CROSS
                splat3(Expr.MathCall.abs(new Expr.Binary(BinaryOp.SUB, ndotl, f(0.5)))));   // ABS

        Expr albedo = vec3(0.8, 0.3, 0.2);
        Expr diffuse = mul(albedo, splat3(ndotl));
        Expr specColor = splat3(mul(spec, extras));
        Expr color = add(Expr.MathCall.mix(diffuse, specColor, splat3(ndoth)), rim);   // MIX

        Region body = Region.of(
                new Statement.InterfaceWrite(fragColor, new Expr.VectorConstruct(VEC4, List.of(color, f(1)))),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.FRAGMENT)))
                .toByteArray();
    }

    @Test
    void mathIntrinsicsLowerValidateAndDisassemble() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        byte[] spirv = intrinsicsShader();

        NativeTools.ValidationResult validation = tools.validate(spirv);
        assertTrue(validation.valid(),
                () -> "spirv-val rejected the math-intrinsics shader:\n" + validation.output());

        String disasm = tools.disassemble(spirv);
        assertTrue(disasm.contains("OpExtInstImport \"GLSL.std.450\""),
                () -> "expected a GLSL.std.450 import:\n" + disasm);
        assertTrue(disasm.contains("OpDot"), () -> "expected an OpDot:\n" + disasm);
        // spirv-dis renders extended instructions by their GLSL.std.450 names.
        for (String fn : List.of("Normalize", "Pow", "Reflect", "FClamp", "InverseSqrt", "Cross")) {
            assertTrue(disasm.contains(fn), () -> "expected a " + fn + " ext-instruction:\n" + disasm);
        }
    }
}
