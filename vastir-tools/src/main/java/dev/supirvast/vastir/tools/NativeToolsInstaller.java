package dev.supirvast.vastir.tools;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Build-time tool that bundles the host platform's native SPIR-V toolchain into the module output.
 *
 * <p>Binaries are sourced from the pinned LunarG Vulkan SDK — the only single distribution carrying both
 * SPIRV-Tools and SPIRV-Cross, pinned and QA-tested. The SDK is downloaded once and silently installed in
 * {@code copy_only} mode (no admin, no registry) into a per-version cache under {@code ~/.supirvast}; the
 * handful of executables we need are then copied into the jar's resources. Subsequent builds reuse the cache.
 *
 * <p>Invoked from Maven at the {@code process-classes} phase: {@code <outputDir> <sdkVersion>}.
 */
public final class NativeToolsInstaller {

    /** Tools copied out of the SDK and bundled for the runtime. */
    static final List<String> REQUIRED_TOOLS =
            List.of("spirv-val", "spirv-dis", "spirv-as", "spirv-opt", "spirv-cross");

    private NativeToolsInstaller() {
    }

    public static void main(String[] args) throws Exception {
        if (Boolean.getBoolean("vast.skipNativeTools")) {
            log("skipped via -Dvast.skipNativeTools");
            return;
        }
        if (args.length < 2) {
            throw new IllegalArgumentException("usage: NativeToolsInstaller <outputDir> <sdkVersion>");
        }
        Path outputDir = Path.of(args[0]);
        String sdkVersion = args[1];
        Platform platform = Platform.current();

        if (platform != Platform.WINDOWS_X64) {
            // Host-platform-only scope; the SDK fetch is currently wired for Windows x64.
            log("host platform " + platform + " not yet supported for fetch; no binaries bundled");
            return;
        }

        Path cacheRoot = Path.of(System.getProperty("user.home"), ".supirvast", "vulkan", sdkVersion);
        Path sdkRoot = cacheRoot.resolve("sdk"); // install target; kept separate from the installer download
        Path binDir = sdkRoot.resolve("Bin");

        if (allToolsPresent(binDir, platform)) {
            log("using cached Vulkan SDK " + sdkVersion + " at " + sdkRoot);
        } else {
            downloadAndInstall(cacheRoot, sdkRoot, sdkVersion);
            if (!allToolsPresent(binDir, platform)) {
                throw new IllegalStateException("SDK install did not yield expected tools in " + binDir);
            }
        }
        bundle(binDir, outputDir, platform);
    }

    private static boolean allToolsPresent(Path binDir, Platform platform) {
        return REQUIRED_TOOLS.stream()
                .allMatch(tool -> Files.isRegularFile(binDir.resolve(platform.executableName(tool))));
    }

    private static void downloadAndInstall(Path cacheRoot, Path sdkRoot, String sdkVersion) throws Exception {
        Files.createDirectories(cacheRoot);
        Path installer = cacheRoot.resolve("vulkan_sdk.exe");

        if (!Files.isRegularFile(installer) || Files.size(installer) == 0) {
            String url = "https://sdk.lunarg.com/sdk/download/" + sdkVersion + "/windows/vulkan_sdk.exe";
            log("downloading Vulkan SDK " + sdkVersion + " (~324 MB) from " + url);
            HttpClient client = HttpClient.newBuilder()
                    .followRedirects(HttpClient.Redirect.NORMAL)
                    .connectTimeout(Duration.ofSeconds(60))
                    .build();
            HttpResponse<Path> response = client.send(
                    HttpRequest.newBuilder(URI.create(url)).GET().build(),
                    HttpResponse.BodyHandlers.ofFile(installer));
            if (response.statusCode() != 200) {
                throw new IOException("download failed: HTTP " + response.statusCode());
            }
            log("downloaded " + Files.size(installer) + " bytes");
        } else {
            log("reusing cached installer at " + installer);
        }

        // The installer refuses a non-empty --root (defaulting the overwrite prompt to "No"), so install
        // into a freshly emptied directory distinct from where the installer itself lives.
        deleteRecursively(sdkRoot);
        Files.createDirectories(sdkRoot);

        log("installing (copy_only) into " + sdkRoot);
        Process process = new ProcessBuilder(
                installer.toString(),
                "--root", sdkRoot.toString(),
                "--accept-licenses", "--default-answer", "--confirm-command", "install", "copy_only=1")
                .inheritIO()
                .start();
        if (!process.waitFor(20, TimeUnit.MINUTES)) {
            process.destroyForcibly();
            throw new IllegalStateException("Vulkan SDK installer timed out");
        }
        if (process.exitValue() != 0) {
            throw new IllegalStateException("Vulkan SDK installer exited with " + process.exitValue());
        }
    }

    private static void bundle(Path binDir, Path outputDir, Platform platform) throws IOException {
        Path dest = outputDir.resolve("dev/supirvast/vastir/tools/native").resolve(platform.dir());
        Files.createDirectories(dest);
        for (String tool : REQUIRED_TOOLS) {
            String exe = platform.executableName(tool);
            Files.copy(binDir.resolve(exe), dest.resolve(exe), StandardCopyOption.REPLACE_EXISTING);
        }
        log("bundled " + REQUIRED_TOOLS.size() + " native tools into " + dest);
    }

    private static void deleteRecursively(Path dir) throws IOException {
        if (!Files.exists(dir)) {
            return;
        }
        try (var paths = Files.walk(dir)) {
            paths.sorted(java.util.Comparator.reverseOrder()).forEach(p -> {
                try {
                    Files.delete(p);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            });
        }
    }

    private static void log(String message) {
        System.out.println("[native-tools] " + message);
    }
}
