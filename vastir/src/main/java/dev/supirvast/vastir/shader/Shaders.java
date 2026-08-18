package dev.supirvast.vastir.shader;

import dev.supirvast.vastir.lower.CoreToSpirv;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Runtime access to SPIR-V binaries pre-compiled by the {@code supirvast-maven-plugin}.
 *
 * <p>The plugin writes each {@link ShaderSource} to the classpath resource
 * {@value #RESOURCE_DIR}{@code <name>.spv} plus an {@value #INDEX_RESOURCE} listing all names. Loading a
 * pre-compiled shader here is a plain resource read — the IR is never rebuilt and {@code CoreToSpirv}
 * never runs.
 */
public final class Shaders {

    /** Classpath directory the build plugin writes compiled shaders into. */
    public static final String RESOURCE_DIR = "META-INF/supirvast/shaders/";
    /** Classpath resource listing the names of all shaders compiled into this artifact, one per line. */
    public static final String INDEX_RESOURCE = RESOURCE_DIR + "index";

    private Shaders() {
    }

    /**
     * Loads the pre-compiled SPIR-V binary for {@code name} from the classpath.
     *
     * @throws IllegalArgumentException if no pre-compiled shader with that name is on the classpath
     */
    public static byte[] load(String name) {
        return load(name, Shaders.class.getClassLoader());
    }

    /** As {@link #load(String)}, resolving against the given class loader. */
    public static byte[] load(String name, ClassLoader loader) {
        String resource = RESOURCE_DIR + name + ".spv";
        try (InputStream in = loader.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IllegalArgumentException("no pre-compiled shader '" + name + "' on the classpath "
                        + "(expected resource " + resource + "; is the supirvast-maven-plugin "
                        + "compile-shaders goal bound in the module that defines it?)");
            }
            return in.readAllBytes();
        } catch (IOException e) {
            throw new UncheckedIOException("reading " + resource, e);
        }
    }

    /** Whether a pre-compiled shader with the given name is on the classpath. */
    public static boolean isPrecompiled(String name, ClassLoader loader) {
        return loader.getResource(RESOURCE_DIR + name + ".spv") != null;
    }

    /**
     * The pre-compiled binary for {@code source} if present on the classpath, otherwise the result of
     * lowering {@code source} in-process — the fallback for running without the build plugin (IDE, tests).
     */
    public static byte[] loadOrLower(ShaderSource source) {
        ClassLoader loader = source.getClass().getClassLoader();
        if (isPrecompiled(source.name(), loader)) {
            return load(source.name(), loader);
        }
        return new CoreToSpirv().lower(source.module(), source.target()).toByteArray();
    }

    /** Names listed in the first {@value #INDEX_RESOURCE} visible to the given loader (empty if none). */
    public static List<String> names(ClassLoader loader) {
        try (InputStream in = loader.getResourceAsStream(INDEX_RESOURCE)) {
            if (in == null) {
                return List.of();
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8).lines()
                    .filter(line -> !line.isBlank())
                    .toList();
        } catch (IOException e) {
            throw new UncheckedIOException("reading " + INDEX_RESOURCE, e);
        }
    }
}
