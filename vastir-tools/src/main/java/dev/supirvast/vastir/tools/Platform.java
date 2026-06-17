package dev.supirvast.vastir.tools;

import java.util.Locale;

/** Host platform identification and the per-platform layout of bundled native binaries. */
public enum Platform {

    WINDOWS_X64("windows-x64", ".exe"),
    LINUX_X64("linux-x64", ""),
    MACOS_X64("macos-x64", ""),
    MACOS_ARM64("macos-arm64", "");

    private final String dir;
    private final String exeSuffix;

    Platform(String dir, String exeSuffix) {
        this.dir = dir;
        this.exeSuffix = exeSuffix;
    }

    /** Classifier directory under {@code native/} holding this platform's binaries. */
    public String dir() {
        return dir;
    }

    /** Executable file-name suffix for this platform ({@code .exe} on Windows, empty elsewhere). */
    public String executableName(String tool) {
        return tool + exeSuffix;
    }

    public static Platform current() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        boolean arm = arch.contains("aarch64") || arch.contains("arm");
        if (os.contains("win")) {
            return WINDOWS_X64;
        }
        if (os.contains("mac") || os.contains("darwin")) {
            return arm ? MACOS_ARM64 : MACOS_X64;
        }
        if (os.contains("nux") || os.contains("nix")) {
            return LINUX_X64;
        }
        throw new IllegalStateException("unsupported platform: " + os + " / " + arch);
    }
}
