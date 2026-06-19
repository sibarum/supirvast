package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.Texture;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Textures Layer A (de-risk): confirm the {@code core} IR can sample a texture. A fragment shader reads a UV
 * varying and samples a {@link Texture}, which must lower to a {@code UniformConstant} combined image+sampler
 * ({@code OpTypeImage}/{@code OpTypeSampledImage}), validate with {@code spirv-val}, and cross-compile to a GLSL
 * {@code sampler2D}.
 */
class TextureShaderTest {

    private static final Type.Float F32 = Type.float32();
    private static final Type.Vector VEC2 = new Type.Vector(F32, 2);
    private static final Type.Vector VEC3 = new Type.Vector(F32, 3);
    private static final Type.Vector VEC4 = new Type.Vector(F32, 4);

    /** {@code fragColor = texture(albedoTex, vUv);} */
    private static byte[] textureShader() {
        Texture albedo = new Texture("albedoTex", 0, 0);
        InterfaceVar uv = InterfaceVar.input("vUv", 0, VEC2);
        InterfaceVar fragColor = InterfaceVar.output("fragColor", 0, VEC4);

        Region body = Region.of(
                new Statement.InterfaceWrite(fragColor,
                        new Expr.SampleTexture(albedo, new Expr.InterfaceRead(uv))),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.FRAGMENT)))
                .toByteArray();
    }

    @Test
    void textureSampleLowersValidatesAndCrossCompiles() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        byte[] spirv = textureShader();

        NativeTools.ValidationResult validation = tools.validate(spirv);
        assertTrue(validation.valid(),
                () -> "spirv-val rejected the texture-sampling shader:\n" + validation.output());

        String disasm = tools.disassemble(spirv);
        assertTrue(disasm.contains("OpTypeImage") && disasm.contains("OpTypeSampledImage"),
                () -> "expected a sampled-image type:\n" + disasm);
        assertTrue(disasm.contains("OpImageSampleImplicitLod"),
                () -> "expected an implicit-LOD sample:\n" + disasm);
        assertTrue(disasm.contains("DescriptorSet") && disasm.contains("Binding"),
                () -> "expected the sampler decorated with set + binding:\n" + disasm);

        String glsl = tools.crossCompile(spirv, NativeTools.ShaderLanguage.GLSL);
        assertTrue(glsl.contains("sampler2D"), () -> "expected a GLSL sampler2D:\n" + glsl);
    }

    /** {@code fragColor = texture(env, vDir);} — a cubemap sampled with a direction vector. */
    private static byte[] cubeShader() {
        Texture env = Texture.cube("env", 0);
        InterfaceVar dir = InterfaceVar.input("vDir", 0, VEC3);
        InterfaceVar fragColor = InterfaceVar.output("fragColor", 0, VEC4);
        Region body = Region.of(
                new Statement.InterfaceWrite(fragColor,
                        new Expr.SampleTexture(env, new Expr.InterfaceRead(dir))),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.FRAGMENT)))
                .toByteArray();
    }

    @Test
    void cubeSampleLowersValidatesAndCrossCompiles() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        byte[] spirv = cubeShader();
        NativeTools.ValidationResult validation = tools.validate(spirv);
        assertTrue(validation.valid(),
                () -> "spirv-val rejected the cubemap-sampling shader:\n" + validation.output());

        String disasm = tools.disassemble(spirv);
        assertTrue(disasm.contains("OpTypeImage") && disasm.contains("Cube"),
                () -> "expected a cube image type:\n" + disasm);

        String glsl = tools.crossCompile(spirv, NativeTools.ShaderLanguage.GLSL);
        assertTrue(glsl.contains("samplerCube"), () -> "expected a GLSL samplerCube:\n" + glsl);
    }
}
