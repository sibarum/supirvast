package dev.supirvast.studio;

import org.joml.Matrix4f;
import sibarum.dasum.gui.core.component.Component;
import sibarum.dasum.gui.core.layout.PixelRect;
import sibarum.dasum.gui.core.render.Batcher;
import sibarum.dasum.gui.core.render.CustomRenderers;
import sibarum.dasum.gui.core.render.ShaderUtil;
import sibarum.dasum.gui.natives.gl.Gl;
import sibarum.dasum.gui.vis.render.ViewportScope;

import static sibarum.dasum.gui.natives.gl.Gl.GL_ARRAY_BUFFER;
import static sibarum.dasum.gui.natives.gl.Gl.GL_ELEMENT_ARRAY_BUFFER;
import static sibarum.dasum.gui.natives.gl.Gl.GL_FLOAT;
import static sibarum.dasum.gui.natives.gl.Gl.GL_STATIC_DRAW;
import static sibarum.dasum.gui.natives.gl.Gl.GL_TRIANGLES;
import static sibarum.dasum.gui.natives.gl.Gl.GL_UNSIGNED_INT;

/**
 * The studio's 3D viewport renderer: it owns the model's GPU buffers, the live
 * GL program, and an orbit camera, draws the model into a
 * {@link Component.SceneView}'s rect each frame, and rebuilds the program from
 * editor text on demand.
 *
 * <p>It is registered against {@code Component.SceneView.class} in
 * {@link CustomRenderers} (see {@link #asRenderer()}). dasum-vis registers its
 * own SceneView renderer at {@code DasumVis.init()}; this one is registered
 * <em>after</em> that so it wins the (identity-keyed) slot for the studio's
 * single scene view.
 *
 * <p>The camera orbits a still model: {@link #orbit} turns mouse drag deltas
 * into yaw/pitch, {@link #zoom} dollies the view distance, both clamped to
 * sane ranges. The model matrix is identity — only the view changes — so the
 * fragment shader sees a stationary surface the user inspects from any angle.
 * Both mutators return whether the camera actually moved, so the app can keep
 * the viewport retained-mode correct: it redraws on interaction and is
 * otherwise idle (no per-frame invalidation, no time-based animation).
 *
 * <p>The render path mirrors the point-cloud renderer's contract: enter a
 * {@link ViewportScope} (which flushes the 2D batcher, scissors + retargets the
 * viewport to the rect, enables depth test, and restores all 2D GL state on
 * close), then issue plain GL 3.3 draw calls inside it.
 *
 * <p>GL resources are created lazily on the first frame — the constructor runs
 * while building the UI, before {@code Gl.load()}, so it must not touch GL.
 * {@link #close()} releases everything; calling it twice is a no-op.
 */
public final class ModelRenderer implements AutoCloseable {

    private final Geometry geometry;
    private final String vertexSource;

    private boolean glInitialised = false;
    private int vao;
    private int vbo;
    private int ebo;

    /** The live program; -1 until the first successful build. */
    private int program = -1;
    private int uModelLoc = -1;
    private int uMvpLoc = -1;

    /** Fragment source backing {@link #program}; the last source that compiled. */
    private String activeFragmentSource;

    private final Matrix4f model = new Matrix4f();
    private final Matrix4f view = new Matrix4f();
    private final Matrix4f projection = new Matrix4f();
    private final Matrix4f mvp = new Matrix4f();
    private final float[] matrixScratch = new float[16];

    // ---- orbit camera state (the model is still; the camera moves) ----

    /** Pitch clamp: just shy of straight up/down so the view never gimbal-flips. */
    private static final float PITCH_LIMIT = 1.55f;        // ~89°
    private static final float DISTANCE_MIN = 1.5f;
    private static final float DISTANCE_MAX = 20f;
    private static final float ORBIT_RAD_PER_PIXEL = 0.008f;
    private static final float ZOOM_FACTOR_PER_NOTCH = 1.15f;

    /** Azimuth about the world Y axis, radians. */
    private float yaw = 0f;
    /** Elevation, radians, clamped to {@code ±PITCH_LIMIT}. */
    private float pitch = 0.3f;
    /** Camera-to-target distance, clamped to {@code [DISTANCE_MIN, DISTANCE_MAX]}. */
    private float distance = 3.2f;

    /**
     * @param geometry         the model to draw (interleaved pos/normal/uv)
     * @param vertexSource     the fixed vertex-shader GLSL; never swapped
     * @param initialFragment  the fragment GLSL to build the first program from
     */
    public ModelRenderer(Geometry geometry, String vertexSource, String initialFragment) {
        this.geometry = geometry;
        this.vertexSource = vertexSource;
        this.activeFragmentSource = initialFragment;
    }

    /** Adapter to the framework's custom-renderer hook. */
    public CustomRenderers.Renderer asRenderer() {
        return this::render;
    }

    /**
     * Orbit the camera by a mouse-drag delta (pixels): {@code dx} turns the
     * yaw, {@code dy} the pitch. Pitch is clamped to {@code ±PITCH_LIMIT} so
     * the view can't flip over the poles.
     *
     * @return {@code true} iff the camera actually moved — the caller should
     *         {@code Invalidator.invalidate()} only then, keeping the viewport
     *         retained-mode correct (no redraw when nothing changed)
     */
    public boolean orbit(double dx, double dy) {
        float newYaw = yaw + (float) dx * ORBIT_RAD_PER_PIXEL;
        float newPitch = clamp(pitch + (float) dy * ORBIT_RAD_PER_PIXEL, -PITCH_LIMIT, PITCH_LIMIT);
        if (newYaw == yaw && newPitch == pitch) return false;
        yaw = newYaw;
        pitch = newPitch;
        return true;
    }

    /**
     * Dolly the camera in/out by a scroll-wheel delta ({@code yOff} notches;
     * positive scrolls up / zooms in). Distance is clamped to
     * {@code [DISTANCE_MIN, DISTANCE_MAX]}.
     *
     * @return {@code true} iff the distance actually changed (see {@link #orbit})
     */
    public boolean zoom(double yOff) {
        float factor = (float) Math.pow(ZOOM_FACTOR_PER_NOTCH, -yOff);
        float newDistance = clamp(distance * factor, DISTANCE_MIN, DISTANCE_MAX);
        if (newDistance == distance) return false;
        distance = newDistance;
        return true;
    }

    private static float clamp(float v, float min, float max) {
        return Math.max(min, Math.min(max, v));
    }

    /**
     * Recompile the program from {@code fragmentSource} + the fixed vertex
     * shader and, on success, swap it in as the live program (deleting the old
     * one). On failure the live program is left untouched.
     *
     * @return a {@link CompileResult} carrying success/failure and the log
     */
    public CompileResult recompile(String fragmentSource) {
        if (!glInitialised) {
            // Defer: the program is (re)built lazily inside the first render
            // pass, once a GL context + Gl.load() are guaranteed. Stash the
            // requested source so that first build uses it.
            this.activeFragmentSource = fragmentSource;
            return CompileResult.ok("Queued; compiles on first frame.");
        }
        int newProgram;
        try {
            newProgram = ShaderUtil.buildProgram(vertexSource, fragmentSource);
        } catch (RuntimeException e) {
            return CompileResult.failure(messageOf(e));
        }
        if (program != -1) Gl.glDeleteProgram(program);
        program = newProgram;
        activeFragmentSource = fragmentSource;
        cacheUniformLocations();
        return CompileResult.ok("Compiled OK.");
    }

    private void render(Component component, PixelRect rect, Batcher batcher, float[] proj) {
        if (!glInitialised) initGl();
        if (program == -1) return; // no good program yet — nothing to draw

        try (ViewportScope scope = new ViewportScope(batcher, proj, rect, true)) {
            updateMatrices(rect);

            Gl.glUseProgram(program);
            if (uModelLoc >= 0) Gl.glUniformMatrix4fv(uModelLoc, false, model.get(matrixScratch));
            if (uMvpLoc >= 0) Gl.glUniformMatrix4fv(uMvpLoc, false, mvp.get(matrixScratch));

            Gl.glBindVertexArray(vao);
            Gl.glDrawElements(GL_TRIANGLES, geometry.indexCount(), GL_UNSIGNED_INT, 0L);
            Gl.glBindVertexArray(0);
        }
    }

    private void updateMatrices(PixelRect rect) {
        // The model is still; the camera orbits it. Place the eye on a sphere
        // of radius `distance` about the origin from (yaw, pitch), then look
        // back at the origin. Pitch is pre-clamped, so cos(pitch) > 0 and the
        // up vector stays world-up without flipping.
        float cosPitch = (float) Math.cos(pitch);
        float eyeX = distance * cosPitch * (float) Math.sin(yaw);
        float eyeY = distance * (float) Math.sin(pitch);
        float eyeZ = distance * cosPitch * (float) Math.cos(yaw);

        model.identity();
        view.identity().lookAt(eyeX, eyeY, eyeZ, 0f, 0f, 0f, 0f, 1f, 0f);

        float aspect = rect.height() <= 0f ? 1f : rect.width() / rect.height();
        projection.identity().perspective((float) Math.toRadians(45.0), aspect, 0.1f, 100.0f);

        projection.mul(view, mvp).mul(model);
    }

    private void initGl() {
        vao = Gl.glGenVertexArray();
        vbo = Gl.glGenBuffer();
        ebo = Gl.glGenBuffer();

        Gl.glBindVertexArray(vao);

        Gl.glBindBuffer(GL_ARRAY_BUFFER, vbo);
        Gl.glBufferData(GL_ARRAY_BUFFER, geometry.vertices(), GL_STATIC_DRAW);

        Gl.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        Gl.glBufferData(GL_ELEMENT_ARRAY_BUFFER, geometry.indices(), GL_STATIC_DRAW);

        int strideBytes = Geometry.STRIDE_FLOATS * Float.BYTES;
        // location 0: position (vec3) at offset 0
        Gl.glEnableVertexAttribArray(0);
        Gl.glVertexAttribPointer(0, 3, GL_FLOAT, false, strideBytes, 0L);
        // location 1: normal (vec3) at offset 3 floats
        Gl.glEnableVertexAttribArray(1);
        Gl.glVertexAttribPointer(1, 3, GL_FLOAT, false, strideBytes, 3L * Float.BYTES);
        // location 2: uv (vec2) at offset 6 floats
        Gl.glEnableVertexAttribArray(2);
        Gl.glVertexAttribPointer(2, 2, GL_FLOAT, false, strideBytes, 6L * Float.BYTES);

        Gl.glBindVertexArray(0);
        Gl.glBindBuffer(GL_ARRAY_BUFFER, 0);
        Gl.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        glInitialised = true;

        // Build the initial program now that GL is ready. A failure here is
        // surfaced the same way edits are — but the bundled default must
        // compile, so a failure means a packaging bug.
        try {
            program = ShaderUtil.buildProgram(vertexSource, activeFragmentSource);
            cacheUniformLocations();
        } catch (RuntimeException e) {
            System.err.println("Initial shader build failed: " + messageOf(e));
        }
    }

    private void cacheUniformLocations() {
        uModelLoc = Gl.glGetUniformLocation(program, "uModel");
        uMvpLoc = Gl.glGetUniformLocation(program, "uMvp");
    }

    private static String messageOf(RuntimeException e) {
        String m = e.getMessage();
        return (m == null || m.isBlank()) ? e.toString() : m;
    }

    @Override
    public void close() {
        if (!glInitialised) return;
        if (program != -1) {
            Gl.glDeleteProgram(program);
            program = -1;
        }
        Gl.glDeleteBuffer(vbo);
        Gl.glDeleteBuffer(ebo);
        Gl.glDeleteVertexArray(vao);
        glInitialised = false;
    }

    /**
     * Outcome of a {@link #recompile} attempt — a success flag plus a log
     * suitable for the status line.
     */
    public record CompileResult(boolean success, String log) {
        static CompileResult ok(String log) {
            return new CompileResult(true, log);
        }

        static CompileResult failure(String log) {
            return new CompileResult(false, log);
        }
    }
}
