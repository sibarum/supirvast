package dev.supirvast.vastir.shader;

import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.lower.SpirvTarget;

/**
 * A shader authored in the core IR that the build can discover and pre-compile to SPIR-V.
 *
 * <p>Implement this on a public concrete class with a public no-arg constructor. The
 * {@code supirvast-maven-plugin} {@code compile-shaders} goal scans the module's compiled classes,
 * instantiates every implementation, lowers {@link #module()} with {@code CoreToSpirv}, and writes the
 * binary to {@code META-INF/supirvast/shaders/<name>.spv} inside the module's output directory — so the
 * binary ships in the jar as an ordinary classpath resource.
 *
 * <p>At runtime, {@link Shaders#load(String)} resolves the pre-compiled resource by {@link #name()};
 * {@link Shaders#loadOrLower(ShaderSource)} falls back to lowering in-process only when the resource is
 * absent (e.g. running from an IDE without the plugin bound).
 */
public interface ShaderSource {

    /**
     * Stable resource name for this shader, unique within the artifact — by convention
     * {@code <base>.vert} / {@code <base>.frag} / {@code <base>.comp}. Must be a valid file-name stem:
     * the compiled binary is stored as {@code <name>.spv}.
     */
    String name();

    /** Builds the core-IR module to lower. Called once at build time (and only as a runtime fallback). */
    CoreModule module();

    /** The capability budget to lower against; unconstrained by default. */
    default SpirvTarget target() {
        return SpirvTarget.unconstrained();
    }
}
