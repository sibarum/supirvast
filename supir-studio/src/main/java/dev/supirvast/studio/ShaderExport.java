package dev.supirvast.studio;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Writes a compiled shader stage to disk in the chosen {@link ExportFormat}s. A
 * {@link SupirShaderCompiler.Export} becomes up to five files under a directory: {@code <base>.spv} (SPIR-V
 * binary), {@code <base>.spvasm} (SPIR-V assembly), and {@code <base>.glsl} / {@code <base>.hlsl} /
 * {@code <base>.metal} (cross-compiled).
 *
 * <p>Pure file I/O with no UI dependency, so it is unit-testable against a temp directory; the studio supplies
 * the directory (via the native folder picker), the base name (one per stage), and the selected formats.
 */
final class ShaderExport {

    private ShaderExport() {
    }

    /** Writes the selected formats for one stage under {@code dir}, returning the paths written (stable order). */
    static List<Path> write(Path dir, String baseName, SupirShaderCompiler.Export export,
                            Set<ExportFormat> formats) throws IOException {
        Files.createDirectories(dir);
        List<Path> written = new ArrayList<>();
        for (ExportFormat format : ExportFormat.values()) { // enum order, not the set's iteration order
            if (!formats.contains(format)) {
                continue;
            }
            Path file = dir.resolve(baseName + "." + format.extension());
            if (format == ExportFormat.SPV) {
                Files.write(file, export.spirv());
            } else {
                Files.writeString(file, textOf(export, format), StandardCharsets.UTF_8);
            }
            written.add(file);
        }
        return written;
    }

    private static String textOf(SupirShaderCompiler.Export export, ExportFormat format) {
        return switch (format) {
            case SPVASM -> export.spirvAssembly();
            case GLSL -> export.glsl();
            case HLSL -> export.hlsl();
            case METAL -> export.msl();
            case SPV -> throw new IllegalArgumentException("SPV is binary, not text");
        };
    }
}
