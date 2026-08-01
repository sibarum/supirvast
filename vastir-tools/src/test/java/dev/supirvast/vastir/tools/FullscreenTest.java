package dev.supirvast.vastir.tools;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The fullscreen-pass primitives: a vertex shader that builds an oversized triangle from {@code gl_VertexIndex}
 * with no vertex inputs, and a constant-color fragment shader. Both must lower to {@code spirv-val}-clean SPIR-V
 * and cross-compile — this is the IR-first proof that a shader with an <em>empty</em> vertex layout is valid,
 * ahead of the windowed host that will actually present it.
 */
class FullscreenTest {

    @Test
    void triangleVertexValidatesAndCrossCompiles() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        byte[] spirv = Fullscreen.triangleVertexSpirv();
        NativeTools.ValidationResult validation = tools.validate(spirv);
        assertTrue(validation.valid(), () -> "spirv-val rejected the fullscreen vertex shader:\n" + validation.output());

        String glsl = tools.crossCompile(spirv, NativeTools.ShaderLanguage.GLSL);
        assertTrue(glsl.contains("gl_Position"), () -> "fullscreen vertex GLSL should write gl_Position:\n" + glsl);
        // SPIRV-Cross emits the GLSL spelling gl_VertexID for the Vulkan gl_VertexIndex builtin.
        assertTrue(glsl.contains("gl_VertexID"),
                () -> "fullscreen vertex GLSL should derive position from the vertex index:\n" + glsl);
    }

    @Test
    void constantColorFragmentValidatesAndCrossCompiles() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        byte[] spirv = Fullscreen.constantColorFragmentSpirv(0.1, 0.2, 0.4, 1.0);
        NativeTools.ValidationResult validation = tools.validate(spirv);
        assertTrue(validation.valid(),
                () -> "spirv-val rejected the constant-color fragment shader:\n" + validation.output());

        String glsl = tools.crossCompile(spirv, NativeTools.ShaderLanguage.GLSL);
        assertFalse(glsl.isBlank(), "constant-color fragment produced no GLSL");
    }
}
