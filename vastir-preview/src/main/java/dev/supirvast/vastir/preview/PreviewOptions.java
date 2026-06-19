package dev.supirvast.vastir.preview;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Parsed command-line options for the previewer. Kept separate from {@link PreviewApp} so the parsing is unit
 * testable without a GPU or a window.
 *
 * <pre>
 * --vert &lt;file.spv&gt;        vertex-stage SPIR-V (required)
 * --frag &lt;file.spv&gt;        fragment-stage SPIR-V (required)
 * --model &lt;file.obj&gt;       Wavefront OBJ / PLY model (required)
 * --width &lt;px&gt;             window width  (default 1280)
 * --height &lt;px&gt;            window height (default 720)
 * --screenshot &lt;png&gt;       render one frame to this PNG and exit (optional)
 * --frames &lt;n&gt;             render at most n frames then exit (optional; for non-interactive runs)
 * --texture &lt;binding&gt;=&lt;png&gt; bind a 2D texture (combined image sampler) at the binding (repeatable)
 * --cubemap &lt;binding&gt;=&lt;prefix&gt; bind a cubemap; faces are &lt;prefix&gt;_px/_nx/_py/_ny/_pz/_nz.png (repeatable)
 * --mvp                     push a rotating model-view-projection mat4 (vertex push constant, offset 0)
 * </pre>
 */
public record PreviewOptions(
        Path vert, Path frag, Path model,
        int width, int height,
        Optional<Path> screenshot, Optional<Integer> frames,
        Map<Integer, Path> textures, Map<Integer, Path> cubemaps, boolean mvp) {

    private static final int DEFAULT_WIDTH = 1280;
    private static final int DEFAULT_HEIGHT = 720;

    public PreviewOptions {
        if (vert == null || frag == null || model == null) {
            throw new IllegalArgumentException("--vert, --frag, and --model are required");
        }
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("--width/--height must be positive, got " + width + "x" + height);
        }
        frames.ifPresent(n -> {
            if (n <= 0) {
                throw new IllegalArgumentException("--frames must be positive, got " + n);
            }
        });
        textures = Map.copyOf(textures);
        cubemaps = Map.copyOf(cubemaps);
    }

    /** Parses {@code argv}, throwing {@link IllegalArgumentException} with a usage-friendly message on error. */
    public static PreviewOptions parse(String[] argv) {
        Path vert = null;
        Path frag = null;
        Path model = null;
        int width = DEFAULT_WIDTH;
        int height = DEFAULT_HEIGHT;
        Optional<Path> screenshot = Optional.empty();
        Optional<Integer> frames = Optional.empty();
        Map<Integer, Path> textures = new LinkedHashMap<>();
        Map<Integer, Path> cubemaps = new LinkedHashMap<>();
        boolean mvp = false;

        for (int i = 0; i < argv.length; i++) {
            String flag = argv[i];
            switch (flag) {
                case "--vert" -> vert = Path.of(value(argv, ++i, flag));
                case "--frag" -> frag = Path.of(value(argv, ++i, flag));
                case "--model" -> model = Path.of(value(argv, ++i, flag));
                case "--width" -> width = intValue(argv, ++i, flag);
                case "--height" -> height = intValue(argv, ++i, flag);
                case "--screenshot" -> screenshot = Optional.of(Path.of(value(argv, ++i, flag)));
                case "--frames" -> frames = Optional.of(intValue(argv, ++i, flag));
                case "--texture" -> parseBinding(value(argv, ++i, flag), textures, flag);
                case "--cubemap" -> parseBinding(value(argv, ++i, flag), cubemaps, flag);
                case "--mvp" -> mvp = true;
                default -> throw new IllegalArgumentException("unknown option: " + flag);
            }
        }
        // A 2D texture and a cubemap can't share a descriptor binding.
        for (Integer binding : cubemaps.keySet()) {
            if (textures.containsKey(binding)) {
                throw new IllegalArgumentException("binding " + binding + " is bound as both texture and cubemap");
            }
        }
        return new PreviewOptions(vert, frag, model, width, height, screenshot, frames, textures, cubemaps, mvp);
    }

    /** Parses a {@code <binding>=<path>} assignment into {@code out}. */
    private static void parseBinding(String assignment, Map<Integer, Path> out, String flag) {
        int eq = assignment.indexOf('=');
        if (eq <= 0) {
            throw new IllegalArgumentException(flag + " expects <binding>=<path>, got '" + assignment + "'");
        }
        int binding;
        try {
            binding = Integer.parseInt(assignment.substring(0, eq));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(flag + " binding must be an integer, got '"
                    + assignment.substring(0, eq) + "'");
        }
        if (out.put(binding, Path.of(assignment.substring(eq + 1))) != null) {
            throw new IllegalArgumentException(flag + " binding " + binding + " specified twice");
        }
    }

    /** One-line usage string for error reporting. */
    public static String usage() {
        return "usage: vastir-preview --vert <file.spv> --frag <file.spv> --model <file.obj|ply> "
                + "[--width <px>] [--height <px>] [--screenshot <png>] [--frames <n>] "
                + "[--texture <binding>=<png> ...] [--cubemap <binding>=<prefix> ...] [--mvp]";
    }

    private static String value(String[] argv, int index, String flag) {
        if (index >= argv.length) {
            throw new IllegalArgumentException(flag + " requires a value");
        }
        return argv[index];
    }

    private static int intValue(String[] argv, int index, String flag) {
        String raw = value(argv, index, flag);
        try {
            return Integer.parseInt(raw);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(flag + " expects an integer, got '" + raw + "'");
        }
    }
}
