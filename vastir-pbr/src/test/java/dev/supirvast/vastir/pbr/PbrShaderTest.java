package dev.supirvast.vastir.pbr;

import dev.supirvast.vastir.tools.GraphicsPipelineSpec;
import dev.supirvast.vastir.tools.NativeTools;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Set;

import static dev.supirvast.vastir.pbr.Shade.f;
import static dev.supirvast.vastir.pbr.Shade.vec3;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The PBR authoring layer: a declared channel set + a surface function generate a Cook-Torrance vertex+fragment
 * pair. The generated SPIR-V must pass {@code spirv-val}; the surface/channel contract is enforced up front.
 */
class PbrShaderTest {

    @Test
    void generatesValidCookTorranceShaders() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        // A red metal: albedo set, fully metallic, fairly smooth.
        PbrShader shader = PbrShader.create(
                Set.of(Channel.ALBEDO, Channel.METALLIC, Channel.ROUGHNESS),
                inputs -> Map.of(
                        Channel.ALBEDO, vec3(0.9, 0.1, 0.1),
                        Channel.METALLIC, f(1.0),
                        Channel.ROUGHNESS, f(0.3)));
        GraphicsPipelineSpec spec = shader.spec();

        NativeTools.ValidationResult vertex = tools.validate(spec.vertexSpirv());
        assertTrue(vertex.valid(), () -> "spirv-val rejected the generated vertex shader:\n" + vertex.output());
        NativeTools.ValidationResult fragment = tools.validate(spec.fragmentSpirv());
        assertTrue(fragment.valid(),
                () -> "spirv-val rejected the generated fragment shader:\n" + fragment.output());

        // The lit fragment shader should also cross-compile (uses the GLSL.std.450 math from Layer A).
        String glsl = tools.crossCompile(spec.fragmentSpirv(), NativeTools.ShaderLanguage.GLSL);
        assertTrue(glsl.contains("void main"), () -> "fragment GLSL missing main:\n" + glsl);
    }

    @Test
    void environmentLitMaterialValidatesAndUsesACubemap() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        PbrShader shader = PbrShader.createWithEnvironment(
                Set.of(Channel.METALLIC, Channel.ROUGHNESS),
                inputs -> Map.of(Channel.METALLIC, f(1.0), Channel.ROUGHNESS, f(0.2)),
                0);
        GraphicsPipelineSpec spec = shader.spec();

        assertTrue(tools.validate(spec.vertexSpirv()).valid(), "vertex invalid");
        assertTrue(tools.validate(spec.fragmentSpirv()).valid(), "fragment invalid");
        String glsl = tools.crossCompile(spec.fragmentSpirv(), NativeTools.ShaderLanguage.GLSL);
        assertTrue(glsl.contains("samplerCube"), () -> "expected an environment samplerCube:\n" + glsl);
    }

    @Test
    void mvpMaterialValidates() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        PbrShader shader = PbrShader.create(
                        Set.of(Channel.ALBEDO, Channel.ROUGHNESS),
                        inputs -> Map.of(Channel.ALBEDO, vec3(0.8, 0.2, 0.2), Channel.ROUGHNESS, f(0.4)))
                .withMvp();
        GraphicsPipelineSpec spec = shader.spec();

        assertTrue(tools.validate(spec.vertexSpirv()).valid(), "MVP vertex shader invalid");
        assertTrue(tools.validate(spec.fragmentSpirv()).valid(), "MVP fragment shader invalid");
    }

    @Test
    void allDefaultsStillProduceAValidShader() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        // Declare nothing: every channel falls back to its default (a grey dielectric). Still a valid shader.
        PbrShader shader = PbrShader.create(Set.of(), inputs -> Map.of());
        assertTrue(tools.validate(shader.spec().fragmentSpirv()).valid());
    }

    @Test
    void rejectsSurfaceThatSetsTheWrongChannels() {
        PbrShader shader = PbrShader.create(Set.of(Channel.ALBEDO),
                inputs -> Map.of(Channel.METALLIC, f(1.0)));   // declared ALBEDO, set METALLIC
        assertThrows(IllegalArgumentException.class, shader::fragmentFunction);
    }

    @Test
    void rejectsChannelExpressionOfTheWrongType() {
        PbrShader shader = PbrShader.create(Set.of(Channel.METALLIC),
                inputs -> Map.of(Channel.METALLIC, vec3(1, 0, 0)));   // METALLIC is f32, not vec3
        assertThrows(IllegalArgumentException.class, shader::fragmentFunction);
    }
}
