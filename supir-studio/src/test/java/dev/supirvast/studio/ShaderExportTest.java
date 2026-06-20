package dev.supirvast.studio;

import dev.supirvast.vastir.core.ShaderStage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import sibarum.dasum.gui.core.render.ShaderUtil;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** The Export pipeline, headless: every distributable form is produced, and the file writer lands them all. */
class ShaderExportTest {

    private final SupirShaderCompiler compiler = new SupirShaderCompiler();

    @Test
    void exportStageProducesEveryFormat() {
        assumeTrue(compiler.toolsAvailable(), "native SPIR-V toolchain not bundled for this platform");

        String supir = ShaderUtil.readResource("/shaders/default.supir");
        SupirShaderCompiler.Export e = compiler.exportStage(supir, ShaderStage.FRAGMENT);

        assertTrue(e.ok(), () -> "export failed:\n" + e.error());
        assertTrue(e.spirv().length > 20 && e.spirv().length % 4 == 0, "implausible SPIR-V binary");
        assertTrue(e.spirvAssembly().contains("OpEntryPoint"), () -> "not SPIR-V assembly:\n" + e.spirvAssembly());
        assertTrue(e.glsl().contains("void main"), () -> "not GLSL:\n" + e.glsl());
        assertFalse(e.hlsl().isBlank(), "empty HLSL");
        assertFalse(e.msl().isBlank(), "empty Metal");
    }

    @Test
    void writeAllFormatsLandsFiveFiles(@TempDir Path dir) throws IOException {
        assumeTrue(compiler.toolsAvailable(), "native SPIR-V toolchain not bundled for this platform");

        SupirShaderCompiler.Export e =
                compiler.exportStage(ShaderUtil.readResource("/shaders/default.supir"), ShaderStage.FRAGMENT);
        assumeTrue(e.ok(), "fragment did not compile");

        List<Path> written = ShaderExport.write(dir, "shader.frag", e, EnumSet.allOf(ExportFormat.class));

        assertEquals(5, written.size());
        for (Path p : written) {
            assertTrue(Files.size(p) > 0, () -> "empty file: " + p);
        }
        assertTrue(Files.exists(dir.resolve("shader.frag.spv")));
        assertTrue(Files.exists(dir.resolve("shader.frag.spvasm")));
        assertTrue(Files.exists(dir.resolve("shader.frag.glsl")));
        assertTrue(Files.exists(dir.resolve("shader.frag.hlsl")));
        assertTrue(Files.exists(dir.resolve("shader.frag.metal")));
    }

    @Test
    void writeSelectedFormatsOnly(@TempDir Path dir) throws IOException {
        assumeTrue(compiler.toolsAvailable(), "native SPIR-V toolchain not bundled for this platform");

        SupirShaderCompiler.Export e =
                compiler.exportStage(ShaderUtil.readResource("/shaders/default.supir"), ShaderStage.FRAGMENT);
        assumeTrue(e.ok(), "fragment did not compile");

        List<Path> written = ShaderExport.write(dir, "shader.frag", e,
                EnumSet.of(ExportFormat.GLSL, ExportFormat.SPV));

        assertEquals(2, written.size());
        assertTrue(Files.exists(dir.resolve("shader.frag.glsl")));
        assertTrue(Files.exists(dir.resolve("shader.frag.spv")));
        assertFalse(Files.exists(dir.resolve("shader.frag.hlsl")), "unselected format must not be written");
        assertFalse(Files.exists(dir.resolve("shader.frag.metal")), "unselected format must not be written");
    }
}
