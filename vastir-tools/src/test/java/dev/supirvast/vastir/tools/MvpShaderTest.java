package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.PushConstants;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.Builtin;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * MVP Layer A (de-risk): confirm the {@code core} IR can express a model-view-projection transform — a
 * {@link Type.Matrix} read from a {@link PushConstants} block and multiplied by the vertex position
 * ({@link Expr.MatrixTimesVector}). The vertex shader must validate with {@code spirv-val} (a {@code Block}-
 * decorated {@code PushConstant} struct holding a {@code mat4}, {@code OpMatrixTimesVector}) and cross-compile
 * to a GLSL {@code push_constant} {@code mat4}.
 */
class MvpShaderTest {

    private static final Type.Float F32 = Type.float32();
    private static final Type.Vector VEC3 = new Type.Vector(F32, 3);
    private static final Type.Vector VEC4 = new Type.Vector(F32, 4);

    /** {@code gl_Position = mvp * vec4(position, 1);} with {@code mvp} a push-constant mat4. */
    private static byte[] mvpVertexShader() {
        PushConstants pushConstants = PushConstants.of("mvp", Type.mat4());
        InterfaceVar position = InterfaceVar.input("position", 0, VEC3);
        Expr clip = new Expr.MatrixTimesVector(pushConstants.read(0),
                new Expr.VectorConstruct(VEC4, List.of(new Expr.InterfaceRead(position), new Expr.ConstFloat(F32, 1))));
        Region body = Region.of(
                new Statement.BuiltinWrite(Builtin.POSITION, clip),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.VERTEX)))
                .toByteArray();
    }

    @Test
    void mvpTransformLowersValidatesAndCrossCompiles() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        byte[] spirv = mvpVertexShader();

        NativeTools.ValidationResult validation = tools.validate(spirv);
        assertTrue(validation.valid(),
                () -> "spirv-val rejected the MVP vertex shader:\n" + validation.output());

        String disasm = tools.disassemble(spirv);
        assertTrue(disasm.contains("OpTypeMatrix"), () -> "expected a matrix type:\n" + disasm);
        assertTrue(disasm.contains("OpMatrixTimesVector"), () -> "expected a matrix*vector:\n" + disasm);
        assertTrue(disasm.contains("PushConstant"), () -> "expected a PushConstant variable:\n" + disasm);

        // Cross-compiles to GLSL that transforms the position by the matrix (spirv-cross names the
        // push-constant block generically, so assert on the structure, not the type spelling).
        String glsl = tools.crossCompile(spirv, NativeTools.ShaderLanguage.GLSL);
        assertTrue(glsl.contains("gl_Position") && glsl.contains("vec4(_3, 1.0)"),
                () -> "expected the GLSL to transform the position:\n" + glsl);
    }
}
