package dev.supirvast.vastir.shader;

import dev.supirvast.vastir.lower.CoreToSpirv;

import java.io.IOException;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeMap;
import java.util.stream.Stream;

/**
 * The build-time half of shader pre-compilation: scans a directory of compiled classes for
 * {@link ShaderSource} implementations, lowers each with {@link CoreToSpirv}, and writes the binaries as
 * classpath resources under {@link Shaders#RESOURCE_DIR} (plus the {@link Shaders#INDEX_RESOURCE} index).
 *
 * <p>Invoked reflectively by the {@code supirvast-maven-plugin} inside a class loader built from the
 * project's compile classpath, so the {@code ShaderSource} interface, the project's implementations, and
 * this lowering pipeline all share one class loader — the plugin itself carries no vastir dependency and
 * cannot clash with the project's version.
 *
 * <p>Idempotent: a binary is rewritten only when its bytes change, so repeated builds don't churn
 * timestamps.
 */
public final class ShaderBuildCompiler {

    private ShaderBuildCompiler() {
    }

    /**
     * Compiles every {@link ShaderSource} found under {@code classesDir} into
     * {@code outputDir/META-INF/supirvast/shaders/}.
     *
     * @param classesDir root of the module's compiled classes to scan
     * @param outputDir  resource root to write into (normally the same directory)
     * @return the sorted shader names that were compiled
     */
    public static List<String> compile(Path classesDir, Path outputDir) throws IOException {
        ClassLoader loader = ShaderBuildCompiler.class.getClassLoader();
        TreeMap<String, ShaderSource> sources = new TreeMap<>();
        List<String> skipped = new ArrayList<>();
        for (String className : classNames(classesDir)) {
            Class<?> type;
            try {
                type = Class.forName(className, false, loader);
            } catch (ClassNotFoundException | NoClassDefFoundError e) {
                // A class we cannot even resolve: typically an optional dependency that is absent from the
                // scan classpath, so it cannot be a shader source we are meant to compile. Reported rather
                // than dropped, because it is also how a genuinely broken ShaderSource would look.
                skipped.add(className + " (" + e.getClass().getSimpleName() + ": " + e.getMessage() + ")");
                continue;
            }
            // Any other LinkageError (ExceptionInInitializerError, IncompatibleClassChangeError, a static
            // initializer that threw) means the class exists but is broken. Skipping it silently would emit
            // no .spv and leave it out of the index, turning a build error into a runtime one.
            if (!ShaderSource.class.isAssignableFrom(type) || type.isInterface()
                    || Modifier.isAbstract(type.getModifiers())) {
                continue;
            }
            ShaderSource source;
            try {
                source = (ShaderSource) type.getDeclaredConstructor().newInstance();
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException("shader source " + className
                        + " must have a public no-arg constructor", e);
            }
            ShaderSource clash = sources.put(source.name(), source);
            if (clash != null) {
                throw new IllegalStateException("shader name '" + source.name() + "' is declared by both "
                        + clash.getClass().getName() + " and " + className);
            }
        }

        if (!skipped.isEmpty()) {
            // Visible in the build log: if a shader is unexpectedly missing at runtime, this is where it went.
            System.out.println("[supirvast] skipped " + skipped.size()
                    + " unresolvable class(es) while scanning for shader sources:");
            skipped.forEach(entry -> System.out.println("[supirvast]   " + entry));
        }

        Path shaderDir = outputDir.resolve(Shaders.RESOURCE_DIR);
        List<String> names = new ArrayList<>(sources.keySet());
        if (!names.isEmpty()) {
            Files.createDirectories(shaderDir);
            for (ShaderSource source : sources.values()) {
                byte[] spirv = new CoreToSpirv().lower(source.module(), source.target()).toByteArray();
                writeIfChanged(shaderDir.resolve(source.name() + ".spv"), spirv);
            }
            writeIfChanged(shaderDir.resolve("index"),
                    (String.join("\n", names) + "\n").getBytes(StandardCharsets.UTF_8));
        }
        return names;
    }

    /** Fully-qualified names of all classes under {@code classesDir} (skipping any missing directory). */
    private static List<String> classNames(Path classesDir) throws IOException {
        if (!Files.isDirectory(classesDir)) {
            return List.of();
        }
        try (Stream<Path> files = Files.walk(classesDir)) {
            return files.filter(p -> p.getFileName().toString().endsWith(".class"))
                    .map(p -> classesDir.relativize(p).toString()
                            .replace(java.io.File.separatorChar, '.')
                            .replaceAll("\\.class$", ""))
                    .filter(name -> !name.equals("module-info") && !name.endsWith(".package-info"))
                    .toList();
        }
    }

    private static void writeIfChanged(Path file, byte[] content) throws IOException {
        if (Files.exists(file) && Arrays.equals(Files.readAllBytes(file), content)) {
            return;
        }
        Files.write(file, content);
    }
}
