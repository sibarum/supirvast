package dev.supirvast.maven;

import org.apache.maven.artifact.DependencyResolutionRequiredException;
import org.apache.maven.plugin.AbstractMojo;
import org.apache.maven.plugin.MojoExecutionException;
import org.apache.maven.plugins.annotations.LifecyclePhase;
import org.apache.maven.plugins.annotations.Mojo;
import org.apache.maven.plugins.annotations.Parameter;
import org.apache.maven.plugins.annotations.ResolutionScope;
import org.apache.maven.project.MavenProject;

import java.io.File;
import java.lang.reflect.InvocationTargetException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Path;
import java.util.List;

/**
 * Pre-compiles every {@code dev.supirvast.vastir.shader.ShaderSource} implementation in the project to
 * SPIR-V, writing {@code META-INF/supirvast/shaders/<name>.spv} (plus an {@code index}) into the output
 * directory so the binaries ship in the jar and are loadable at runtime via {@code Shaders.load(name)}
 * without re-lowering.
 *
 * <p>The heavy lifting lives in vastir's {@code ShaderBuildCompiler}, which this mojo invokes reflectively
 * inside a class loader built from the project's own compile classpath. The plugin therefore has no vastir
 * dependency of its own: the project's vastir version is the one that scans, lowers, and defines the
 * {@code ShaderSource} type — no cross-classloader type clashes, no version skew.
 */
@Mojo(name = "compile-shaders", defaultPhase = LifecyclePhase.PROCESS_CLASSES,
        requiresDependencyResolution = ResolutionScope.COMPILE, threadSafe = true)
public class CompileShadersMojo extends AbstractMojo {

    private static final String COMPILER_CLASS = "dev.supirvast.vastir.shader.ShaderBuildCompiler";

    @Parameter(defaultValue = "${project}", readonly = true, required = true)
    private MavenProject project;

    /** Directory scanned for ShaderSource classes and written to; the module's compiled-classes root. */
    @Parameter(defaultValue = "${project.build.outputDirectory}")
    private File classesDirectory;

    /** Skips shader pre-compilation entirely. */
    @Parameter(property = "supirvast.compileShaders.skip", defaultValue = "false")
    private boolean skip;

    @Override
    public void execute() throws MojoExecutionException {
        if (skip) {
            getLog().info("Shader pre-compilation skipped");
            return;
        }
        List<String> classpath;
        try {
            classpath = project.getCompileClasspathElements();
        } catch (DependencyResolutionRequiredException e) {
            throw new MojoExecutionException("resolving the compile classpath", e);
        }
        URL[] urls = classpath.stream().map(element -> {
            try {
                return new File(element).toURI().toURL();
            } catch (java.net.MalformedURLException e) {
                throw new IllegalStateException("classpath element " + element, e);
            }
        }).toArray(URL[]::new);

        // Platform parent: the project classpath must be self-contained so ShaderSource, the project's
        // implementations, and CoreToSpirv all resolve from one loader (and never from the plugin's).
        try (URLClassLoader loader = new URLClassLoader(urls, ClassLoader.getPlatformClassLoader())) {
            Class<?> compiler = loader.loadClass(COMPILER_CLASS);
            @SuppressWarnings("unchecked")
            List<String> names = (List<String>) compiler
                    .getMethod("compile", Path.class, Path.class)
                    .invoke(null, classesDirectory.toPath(), classesDirectory.toPath());
            if (names.isEmpty()) {
                getLog().info("No ShaderSource implementations found in " + classesDirectory);
            } else {
                getLog().info("Pre-compiled " + names.size() + " shader(s) to SPIR-V: "
                        + String.join(", ", names));
            }
        } catch (ClassNotFoundException e) {
            throw new MojoExecutionException(COMPILER_CLASS + " is not on the project's compile classpath — "
                    + "add a dependency on dev.supirvast:vastir", e);
        } catch (InvocationTargetException e) {
            throw new MojoExecutionException("shader pre-compilation failed: "
                    + e.getCause().getMessage(), e.getCause());
        } catch (ReflectiveOperationException | java.io.IOException e) {
            throw new MojoExecutionException("shader pre-compilation failed", e);
        }
    }
}
