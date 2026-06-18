package dev.supirvast.vastir.preview;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** CLI parsing for the previewer — pure logic, no GPU or window required. */
class PreviewOptionsTest {

    @Test
    void parsesRequiredFlagsAndDefaults() {
        PreviewOptions options = PreviewOptions.parse(new String[]{
                "--vert", "v.spv", "--frag", "f.spv", "--model", "cube.obj"});

        assertEquals(Path.of("v.spv"), options.vert());
        assertEquals(Path.of("f.spv"), options.frag());
        assertEquals(Path.of("cube.obj"), options.model());
        assertEquals(1280, options.width());
        assertEquals(720, options.height());
        assertTrue(options.screenshot().isEmpty());
        assertTrue(options.frames().isEmpty());
    }

    @Test
    void parsesOptionalFlags() {
        PreviewOptions options = PreviewOptions.parse(new String[]{
                "--vert", "v.spv", "--frag", "f.spv", "--model", "cube.obj",
                "--width", "640", "--height", "480", "--screenshot", "out.png", "--frames", "3"});

        assertEquals(640, options.width());
        assertEquals(480, options.height());
        assertEquals(Path.of("out.png"), options.screenshot().orElseThrow());
        assertEquals(3, options.frames().orElseThrow());
    }

    @Test
    void rejectsMissingRequiredFlags() {
        assertThrows(IllegalArgumentException.class,
                () -> PreviewOptions.parse(new String[]{"--vert", "v.spv"}));
    }

    @Test
    void rejectsUnknownFlagAndMissingValue() {
        assertThrows(IllegalArgumentException.class,
                () -> PreviewOptions.parse(new String[]{"--bogus"}));
        assertThrows(IllegalArgumentException.class,
                () -> PreviewOptions.parse(new String[]{
                        "--vert", "v.spv", "--frag", "f.spv", "--model", "cube.obj", "--width"}));
    }

    @Test
    void rejectsNonPositiveDimensionsAndFrames() {
        assertThrows(IllegalArgumentException.class,
                () -> PreviewOptions.parse(new String[]{
                        "--vert", "v.spv", "--frag", "f.spv", "--model", "cube.obj", "--width", "0"}));
        assertThrows(IllegalArgumentException.class,
                () -> PreviewOptions.parse(new String[]{
                        "--vert", "v.spv", "--frag", "f.spv", "--model", "cube.obj", "--frames", "-1"}));
    }
}
