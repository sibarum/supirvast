package dev.supirvast.vastir.tools;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermission;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * Runtime access to the bundled native SPIR-V toolchain.
 *
 * <p>On first use, the executables packaged for the {@linkplain Platform#current() host platform} are
 * extracted from the classpath to a temporary directory and invoked as processes. Operates on the raw SPIR-V
 * bytes produced by the vastir emitter, so the full lower → validate → cross-compile path runs from one jar.
 */
public final class NativeTools {

    /** Target high-level language for cross-compilation via SPIRV-Cross. */
    public enum ShaderLanguage { GLSL, HLSL, MSL }

    /** Result of running {@code spirv-val}: validity plus the tool's diagnostic output. */
    public record ValidationResult(boolean valid, String output) {}

    private static final String RESOURCE_BASE = "/dev/supirvast/vastir/tools/native/";

    private final Platform platform;
    private volatile Path extractedDir;

    public NativeTools() {
        this(Platform.current());
    }

    public NativeTools(Platform platform) {
        this.platform = platform;
    }

    /** Whether native tools for the host platform are bundled in this artifact. */
    public boolean isAvailable() {
        return NativeTools.class.getResource(resourcePath("spirv-val")) != null;
    }

    /** Validates a SPIR-V module with {@code spirv-val}. */
    public ValidationResult validate(byte[] spirv) {
        Path module = writeTempModule(spirv);
        try {
            Execution exec = run(List.of(toolPath("spirv-val").toString(), module.toString()));
            String output = (exec.stdout() + exec.stderr()).strip();
            return new ValidationResult(exec.exitCode() == 0, output);
        } finally {
            deleteQuietly(module);
        }
    }

    /** Disassembles a SPIR-V module to human-readable assembly with {@code spirv-dis}. */
    public String disassemble(byte[] spirv, String... extraArgs) {
        Path module = writeTempModule(spirv);
        try {
            List<String> command = new ArrayList<>();
            command.add(toolPath("spirv-dis").toString());
            command.add(module.toString());
            command.addAll(List.of(extraArgs));
            Execution exec = run(command);
            if (exec.exitCode() != 0) {
                throw new IllegalStateException("spirv-dis failed: " + exec.stderr());
            }
            return exec.stdout();
        } finally {
            deleteQuietly(module);
        }
    }

    /** Assembles SPIR-V assembly text into a binary module with {@code spirv-as}. */
    public byte[] assemble(String assembly) {
        Path source = writeTemp(assembly.getBytes(java.nio.charset.StandardCharsets.UTF_8), ".spvasm");
        Path output = newTempPath();
        try {
            Execution exec = run(List.of(
                    toolPath("spirv-as").toString(), source.toString(), "-o", output.toString()));
            if (exec.exitCode() != 0) {
                throw new IllegalStateException("spirv-as failed: " + exec.stderr());
            }
            return Files.readAllBytes(output);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to read assembled module", e);
        } finally {
            deleteQuietly(source);
            deleteQuietly(output);
        }
    }

    /**
     * Optimizes a SPIR-V module with {@code spirv-opt}, returning the rewritten binary. With no
     * {@code passes}, runs the standard {@code -O} performance recipe; otherwise runs exactly the passes given
     * (e.g. {@code "--ssa-rewrite"} to promote memory-based locals to SSA {@code OpPhi} — the {@code mem2reg}
     * transformation that makes hand-written {@code OpPhi} unnecessary).
     */
    public byte[] optimize(byte[] spirv, String... passes) {
        Path module = writeTempModule(spirv);
        Path output = newTempPath();
        try {
            List<String> command = new ArrayList<>(List.of(toolPath("spirv-opt").toString()));
            if (passes.length == 0) {
                command.add("-O");
            } else {
                command.addAll(List.of(passes));
            }
            command.add(module.toString());
            command.add("-o");
            command.add(output.toString());
            Execution exec = run(command);
            if (exec.exitCode() != 0) {
                throw new IllegalStateException("spirv-opt failed: " + exec.stderr());
            }
            return Files.readAllBytes(output);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to read optimized module", e);
        } finally {
            deleteQuietly(module);
            deleteQuietly(output);
        }
    }

    /** Cross-compiles a SPIR-V module to the given high-level language with {@code spirv-cross}. */
    public String crossCompile(byte[] spirv, ShaderLanguage language) {
        Path module = writeTempModule(spirv);
        try {
            List<String> command = new ArrayList<>(List.of(toolPath("spirv-cross").toString(), module.toString()));
            switch (language) {
                case GLSL -> { /* default output */ }
                // Shader Model 5.0 (DX11) is the baseline for modern HLSL: it supports compute/UAVs and
                // system-value semantics like SV_VertexID, which the default SM 3.0 rejects.
                case HLSL -> command.addAll(List.of("--hlsl", "--shader-model", "50"));
                case MSL -> command.add("--msl");
            }
            Execution exec = run(command);
            if (exec.exitCode() != 0) {
                throw new IllegalStateException("spirv-cross failed: " + exec.stderr());
            }
            return exec.stdout();
        } finally {
            deleteQuietly(module);
        }
    }

    // --- internals -------------------------------------------------------------------------------------

    private String resourcePath(String tool) {
        return RESOURCE_BASE + platform.dir() + "/" + platform.executableName(tool);
    }

    private Path toolPath(String tool) {
        ensureExtracted();
        Path path = extractedDir.resolve(platform.executableName(tool));
        if (!Files.isRegularFile(path)) {
            throw new IllegalStateException("native tool not bundled: " + tool + " (" + platform + ")");
        }
        return path;
    }

    private void ensureExtracted() {
        if (extractedDir != null) {
            return;
        }
        synchronized (this) {
            if (extractedDir != null) {
                return;
            }
            try {
                Path dir = Files.createTempDirectory("vast-spirv-tools");
                dir.toFile().deleteOnExit();
                for (String tool : NativeToolsInstaller.REQUIRED_TOOLS) {
                    extractTool(tool, dir);
                }
                extractedDir = dir;
            } catch (IOException e) {
                throw new UncheckedIOException("failed to extract native tools", e);
            }
        }
    }

    private void extractTool(String tool, Path dir) throws IOException {
        String resource = resourcePath(tool);
        try (InputStream in = NativeTools.class.getResourceAsStream(resource)) {
            if (in == null) {
                return; // not bundled; toolPath() reports a clear error if it is later requested
            }
            Path target = dir.resolve(platform.executableName(tool));
            Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
            makeExecutable(target);
            target.toFile().deleteOnExit();
        }
    }

    private static void makeExecutable(Path file) throws IOException {
        if (!file.getFileSystem().supportedFileAttributeViews().contains("posix")) {
            return; // Windows: .exe is already executable
        }
        Set<PosixFilePermission> perms = EnumSet.copyOf(Files.getPosixFilePermissions(file));
        perms.add(PosixFilePermission.OWNER_EXECUTE);
        perms.add(PosixFilePermission.GROUP_EXECUTE);
        perms.add(PosixFilePermission.OTHERS_EXECUTE);
        Files.setPosixFilePermissions(file, perms);
    }

    private Path writeTempModule(byte[] spirv) {
        return writeTemp(spirv, ".spv");
    }

    private Path writeTemp(byte[] content, String suffix) {
        try {
            Path file = Files.createTempFile("vast-", suffix);
            Files.write(file, content);
            return file;
        } catch (IOException e) {
            throw new UncheckedIOException("failed to write temp file", e);
        }
    }

    private Path newTempPath() {
        try {
            Path file = Files.createTempFile("vast-", ".spv");
            Files.deleteIfExists(file); // a tool will create it
            return file;
        } catch (IOException e) {
            throw new UncheckedIOException("failed to allocate temp path", e);
        }
    }

    private static Execution run(List<String> command) {
        try {
            Path out = Files.createTempFile("vast-out", ".txt");
            Path err = Files.createTempFile("vast-err", ".txt");
            try {
                Process process = new ProcessBuilder(command)
                        .redirectOutput(out.toFile())
                        .redirectError(err.toFile())
                        .start();
                if (!process.waitFor(120, TimeUnit.SECONDS)) {
                    process.destroyForcibly();
                    throw new IllegalStateException("native tool timed out: " + command.get(0));
                }
                return new Execution(process.exitValue(), Files.readString(out), Files.readString(err));
            } finally {
                deleteQuietly(out);
                deleteQuietly(err);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("failed to run native tool", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted running native tool", e);
        }
    }

    private static void deleteQuietly(Path path) {
        try {
            Files.deleteIfExists(path);
        } catch (IOException ignored) {
            // best-effort cleanup
        }
    }

    private record Execution(int exitCode, String stdout, String stderr) {}
}
