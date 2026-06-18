package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.Builtin;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Graphics stages with interface I/O — the first non-compute shaders. A vertex shader reads
 * {@code gl_VertexIndex}, writes {@code gl_Position} (a built-in output) and a {@code location=0} colour
 * varying; a fragment shader reads that {@code location=0} input and writes a {@code location=0} colour output.
 * This exercises built-in and location-bound interface variables, the vertex/fragment execution models, and
 * vector construction with an int→float conversion.
 *
 * <p>Verification is by {@code spirv-val} + cross-compile to GLSL/HLSL/MSL — there is no CPU==GPU differential
 * here because a fragment shader needs rasterization, which the headless compute path can't drive. The CPU
 * (Truffle) backend correctly rejects these graphics-only constructs.
 */
class VertexFragmentShaderTest {

    private static final Type.Float F32 = Type.float32();
    private static final Type.Vector VEC4 = new Type.Vector(F32, 4);

    private static Expr f(double v) {
        return new Expr.ConstFloat(F32, v);
    }

    /** {@code gl_Position = vec4(float(gl_VertexIndex), 0, 0, 1); vColor = vec4(1, 0, 0, 1);} */
    private static byte[] vertexShader() {
        Expr position = new Expr.VectorConstruct(VEC4, List.of(
                new Expr.Convert(new Expr.BuiltinRead(Builtin.VERTEX_INDEX), F32), f(0), f(0), f(1)));
        InterfaceVar vColor = InterfaceVar.output("vColor", 0, VEC4);
        Region body = Region.of(
                new Statement.BuiltinWrite(Builtin.POSITION, position),
                new Statement.InterfaceWrite(vColor, new Expr.VectorConstruct(VEC4, List.of(f(1), f(0), f(0), f(1)))),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.VERTEX)))
                .toByteArray();
    }

    /** {@code fragColor = vColor;} — pass the location-0 input straight to the location-0 output. */
    private static byte[] fragmentShader() {
        InterfaceVar vColor = InterfaceVar.input("vColor", 0, VEC4);
        InterfaceVar fragColor = InterfaceVar.output("fragColor", 0, VEC4);
        Region body = Region.of(
                new Statement.InterfaceWrite(fragColor, new Expr.InterfaceRead(vColor)),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.FRAGMENT)))
                .toByteArray();
    }

    @Test
    void vertexShaderValidatesAndCrossCompiles() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        byte[] spirv = vertexShader();
        NativeTools.ValidationResult validation = tools.validate(spirv);
        assertTrue(validation.valid(), () -> "spirv-val rejected the vertex shader:\n" + validation.output());

        String glsl = tools.crossCompile(spirv, NativeTools.ShaderLanguage.GLSL);
        assertTrue(glsl.contains("gl_Position"), () -> "vertex GLSL should write gl_Position:\n" + glsl);
        assertFalse(tools.crossCompile(spirv, NativeTools.ShaderLanguage.HLSL).isBlank(), "no vertex HLSL");
        assertFalse(tools.crossCompile(spirv, NativeTools.ShaderLanguage.MSL).isBlank(), "no vertex MSL");
    }

    @Test
    void fragmentShaderValidatesAndCrossCompiles() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        byte[] spirv = fragmentShader();
        NativeTools.ValidationResult validation = tools.validate(spirv);
        assertTrue(validation.valid(), () -> "spirv-val rejected the fragment shader:\n" + validation.output());

        String glsl = tools.crossCompile(spirv, NativeTools.ShaderLanguage.GLSL);
        assertTrue(glsl.contains("location"), () -> "fragment GLSL should declare a location:\n" + glsl);
        assertFalse(tools.crossCompile(spirv, NativeTools.ShaderLanguage.HLSL).isBlank(), "no fragment HLSL");
        assertFalse(tools.crossCompile(spirv, NativeTools.ShaderLanguage.MSL).isBlank(), "no fragment MSL");
    }
}
