package dev.supirvast.studio;

import org.junit.jupiter.api.Test;
import sibarum.dasum.gui.core.render.ShaderUtil;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The studio's Supir → GLSL pipeline, headless. Compiles the bundled default shader and exercises the error
 * paths, without opening a window or a GL context.
 */
class SupirShaderCompilerTest {

    private final SupirShaderCompiler compiler = new SupirShaderCompiler();

    @Test
    void bundledDefaultShaderCompilesToGlsl() {
        assumeTrue(compiler.toolsAvailable(), "native SPIR-V toolchain not bundled for this platform");

        String supir = ShaderUtil.readResource("/shaders/default.supir");
        SupirShaderCompiler.Result result = compiler.compileFragment(supir);

        assertTrue(result.ok(), () -> "default.supir failed to compile:\n" + result.error());
        String glsl = result.glsl();
        assertTrue(glsl.contains("#version 330"), () -> "expected GLSL 330 output:\n" + glsl);
        // Cross-compiled with separate-shader-objects, so the normal varying binds by location 1.
        assertTrue(glsl.contains("layout(location = 1) in"),
                () -> "expected a location-qualified varying input:\n" + glsl);
        assertTrue(glsl.contains("void main"), () -> "expected a main() entry:\n" + glsl);
    }

    @Test
    void parseErrorIsReportedNotThrown() {
        SupirShaderCompiler.Result result = compiler.compileFragment("fragment main {\n  x = bogus 1\n  ret\n}");
        assertFalse(result.ok());
        assertTrue(result.error().contains("undefined name 'bogus'"), result.error());
    }

    @Test
    void nonFragmentStageIsRejectedWithGuidance() {
        SupirShaderCompiler.Result result = compiler.compileFragment("vertex main {\n  ret\n}");
        assertFalse(result.ok());
        assertTrue(result.error().contains("expected a fragment shader"), result.error());
    }

    @Test
    void bundledDefaultVertexCompilesAndExposesMvpUniform() {
        assumeTrue(compiler.toolsAvailable(), "native SPIR-V toolchain not bundled for this platform");

        String supir = ShaderUtil.readResource("/shaders/default.vert.supir");
        SupirShaderCompiler.Result result = compiler.compileVertex(supir);

        assertTrue(result.ok(), () -> "default.vert.supir failed to compile:\n" + result.error());
        assertTrue(result.glsl().contains("#version 330"), () -> "expected GLSL 330 output:\n" + result.glsl());
        assertTrue(result.glsl().contains("gl_Position"), () -> "vertex must write gl_Position:\n" + result.glsl());
        // The push-constant MVP must be recovered so the renderer can set it each frame.
        assertNotNull(result.mvpUniform(), () -> "no MVP uniform recovered from:\n" + result.glsl());
    }

    @Test
    void matrixUniformNameParsesStructForm() {
        String glsl = "struct _20\n{\n    mat4 _m0;\n};\n\nuniform _20 _9;\n";
        assertEquals("_9._m0", SupirShaderCompiler.matrixUniformName(glsl));
    }

    @Test
    void matrixUniformNameParsesDirectForm() {
        assertEquals("uMvp", SupirShaderCompiler.matrixUniformName("uniform mat4 uMvp;"));
    }
}
