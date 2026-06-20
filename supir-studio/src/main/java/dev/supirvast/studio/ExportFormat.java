package dev.supirvast.studio;

/**
 * A distributable shader form the studio can export, with its menu label and file extension. Each maps to one
 * field of a {@link SupirShaderCompiler.Export}.
 */
enum ExportFormat {
    SPV("SPIR-V binary (.spv)", "spv"),
    SPVASM("SPIR-V assembly (.spvasm)", "spvasm"),
    GLSL("GLSL (.glsl)", "glsl"),
    HLSL("HLSL (.hlsl)", "hlsl"),
    METAL("Metal (.metal)", "metal");

    private final String label;
    private final String extension;

    ExportFormat(String label, String extension) {
        this.label = label;
        this.extension = extension;
    }

    /** Human-readable label for the export menu. */
    String label() {
        return label;
    }

    /** File extension (no dot). */
    String extension() {
        return extension;
    }
}
