package dev.supirvast.studio;

import sibarum.dasum.gui.core.GlfwContext;
import sibarum.dasum.gui.core.component.AlignItems;
import sibarum.dasum.gui.core.component.Component;
import sibarum.dasum.gui.core.component.Direction;
import sibarum.dasum.gui.core.component.JustifyContent;
import sibarum.dasum.gui.core.em.Em;
import sibarum.dasum.gui.core.em.EmContext;
import sibarum.dasum.gui.core.event.EventLoop;
import sibarum.dasum.gui.core.event.Invalidator;
import sibarum.dasum.gui.core.input.FocusState;
import sibarum.dasum.gui.core.input.Handlers;
import sibarum.dasum.gui.core.input.HoverState;
import sibarum.dasum.gui.core.input.InputState;
import sibarum.dasum.gui.core.input.TextInputController;
import sibarum.dasum.gui.core.input.TextStates;
import sibarum.dasum.gui.core.input.wheel.WheelRouter;
import sibarum.dasum.gui.core.layout.HitTest;
import sibarum.dasum.gui.core.layout.LatestLayout;
import sibarum.dasum.gui.core.layout.Layout;
import sibarum.dasum.gui.core.layout.LayoutResult;
import sibarum.dasum.gui.core.layout.PixelRect;
import sibarum.dasum.gui.core.layout.Render;
import sibarum.dasum.gui.core.render.Batcher;
import sibarum.dasum.gui.core.render.Color;
import sibarum.dasum.gui.core.render.CustomRenderers;
import sibarum.dasum.gui.core.render.Projection;
import sibarum.dasum.gui.core.render.ShaderUtil;
import sibarum.dasum.gui.core.render.Texture;
import sibarum.dasum.gui.core.text.AtlasData;
import sibarum.dasum.gui.core.text.FontGroup;
import sibarum.dasum.gui.core.text.FontGroups;
import sibarum.dasum.gui.core.theme.Themed;
import sibarum.dasum.gui.core.theme.Variant;
import sibarum.dasum.gui.core.window.Window;
import sibarum.dasum.gui.natives.gl.Gl;
import sibarum.dasum.gui.natives.glfw.Glfw;
import sibarum.dasum.gui.natives.glfw.GlfwCallbacks;
import sibarum.dasum.gui.vis.DasumVis;

import java.util.List;

import static sibarum.dasum.gui.natives.gl.Gl.GL_COLOR_BUFFER_BIT;

/**
 * Phase 1 of the supir-studio live shader playground: a windowed dasumGUIshi
 * app with an editable GLSL fragment shader on the left and a 3D preview on the
 * right.
 *
 * <p>The left panel is an editable {@link Component.Text} (with a line-number
 * gutter) holding fragment-shader source plus a <strong>Compile</strong>
 * button. The right panel is a {@link Component.SceneView} whose
 * {@link ModelRenderer} draws a UV sphere with a fixed MVP vertex shader and
 * the editor's fragment shader; an orbit camera responds to mouse drag (orbit)
 * and the scroll wheel (zoom) over the viewport. A full-width status bar runs
 * along the bottom for notifications. Pressing Compile rebuilds the GL program
 * from the editor text via {@link ShaderUtil#buildProgram}; on failure the
 * compile log is shown in the status bar and the last good program is kept (the
 * app never crashes on bad GLSL).
 *
 * <p>Phase 1 feeds GLSL straight into the editor — the Supir → core → SPIR-V →
 * GLSL pipeline is wired in Phase 2 (see the module README). The loop, layout,
 * input dispatch, and font-atlas setup mirror {@code dasum-mvp-demo}'s reference
 * {@code App}, trimmed to the widgets this app actually uses.
 *
 * <p>A {@code --frames N} option renders {@code N} frames and exits cleanly,
 * for non-interactive smoke testing of the GL/FFM render path.
 */
public final class Studio {

    private static final Color FRAME_BG    = new Color(0.05f, 0.06f, 0.09f, 1f);
    private static final Color PANEL_BG     = new Color(0.10f, 0.12f, 0.15f, 1f);
    private static final Color EDITOR_BG     = new Color(0.07f, 0.09f, 0.13f, 1f);
    private static final Color VIEWPORT_BG   = new Color(0.04f, 0.05f, 0.08f, 1f);
    private static final Color LABEL_FG      = new Color(0.92f, 0.94f, 0.97f, 1f);
    private static final Color EDITOR_FG     = new Color(0.85f, 0.90f, 0.85f, 1f);
    private static final Color STATUS_FG     = new Color(0.70f, 0.78f, 0.88f, 1f);

    /**
     * Priority for the viewport-zoom wheel handler, above the router's
     * built-in DataTable handler ({@link WheelRouter#PRIORITY_DATATABLE}) —
     * matching the value dasum-vis's SceneViewController uses.
     */
    private static final int WHEEL_PRIORITY = 100;

    /** Press target for click dispatch; cleared on release (mirrors the demo). */
    private static Component pressTarget = null;

    private Studio() {}

    public static void main(String[] args) {
        int frameLimit = parseFrameLimit(args);

        String vertexSource = ShaderUtil.readResource("/shaders/model.vert");
        String fragmentSource = ShaderUtil.readResource("/shaders/default.frag");

        Geometry sphere = Geometry.uvSphere(48, 64);
        ModelRenderer modelRenderer = new ModelRenderer(sphere, vertexSource, fragmentSource);

        try (GlfwContext ctx = GlfwContext.init();
             Window window = Window.create(1280, 800, "Supir Studio — Shader Playground");
             Batcher batcher = new Batcher();
             ModelRenderer ownedRenderer = modelRenderer) {

            Gl.load();
            batcher.init();
            DasumVis.init();
            EmContext.setDpiScale(window.contentScaleX());

            // Register the studio's 3D renderer for SceneView. DasumVis.init()
            // installs the point-cloud renderer for the same component class;
            // registering afterwards overwrites it (CustomRenderers is an
            // identity map keyed by class), so our model is what draws.
            CustomRenderers.register(Component.SceneView.class, ownedRenderer.asRenderer());

            try (Texture primaryTexture = Texture.fromPngResource("/dasum/atlas/primary.png")) {
                AtlasData primaryAtlas = AtlasData.loadFromResource("/dasum/atlas/primary.json");
                FontGroups.register(FontGroup.of(FontGroups.DEFAULT, primaryAtlas, primaryTexture));

                Ui ui = buildUi(fragmentSource);
                wireCompile(ui, ownedRenderer);
                wireInput(window, ui, ownedRenderer);

                long[] renderedFrames = {0};
                EventLoop loop = new EventLoop(window, () -> {
                    renderFrame(window, ui.root(), batcher);
                    renderedFrames[0]++;
                    if (frameLimit > 0 && renderedFrames[0] >= frameLimit) {
                        window.requestClose();
                    }
                    // Retained-mode: no per-frame invalidation. The loop goes
                    // idle after the first frame and wakes only when input
                    // dirties it (camera orbit/zoom, editing, compile).
                });

                // Under a frame cap the loop would otherwise idle after the
                // first frame (it blocks in glfwWaitEvents and nothing posts a
                // wake on the main thread), never reaching the cap. For the
                // non-interactive smoke path only, re-dirty AND post an empty
                // event after each layout so glfwWaitEvents returns and renders
                // the next frame. Interactive runs stay purely event-driven.
                if (frameLimit > 0) {
                    LatestLayout.addAfterStore(() -> {
                        Invalidator.invalidate();
                        Glfw.glfwPostEmptyEvent();
                    });
                }
                loop.run();

                System.out.println("Exited cleanly; frames rendered: " + loop.renderedFrameCount());
            }
        }
    }

    /** Render one frame: clear, lay out, render the tree, flush. */
    private static void renderFrame(Window window, Component root, Batcher batcher) {
        int fbW = window.framebufferWidth();
        int fbH = window.framebufferHeight();
        float[] projection = Projection.orthoTopLeft(fbW, fbH);

        Gl.glViewport(0, 0, fbW, fbH);
        Gl.glClearColor(0.03f, 0.03f, 0.05f, 1f);
        Gl.glClear(GL_COLOR_BUFFER_BIT);

        PixelRect viewport = new PixelRect(0f, 0f, fbW, fbH);
        LayoutResult layout = Layout.compute(root, viewport);
        LatestLayout.store(root, layout);

        batcher.beginFrame(fbH);
        Render.render(root, layout, batcher, projection);
        batcher.endFrame(projection);
    }

    // ---------- UI ----------

    /**
     * The assembled UI plus the handles the wiring needs: the {@code editor}
     * to read GLSL from, the {@code compileButton} to attach the click handler
     * to, the {@code status} line to write results to, and the {@code scene}
     * to hit-test mouse input against for camera control.
     */
    private record Ui(Component root, Component.Text editor, Component compileButton,
                      Component.Text status, Component.SceneView scene) {}

    private static Ui buildUi(String initialFragment) {
        Component.Text title = new Component.Text("Fragment shader (GLSL)", Em.of(1.0f), LABEL_FG);

        Component.Text editor = new Component.Text(initialFragment, Em.of(0.85f), EDITOR_FG)
            .withEditable(true)
            .withAcceptsTab(true)
            .withLineNumbers(true)
            .withWrapWidth(Em.of(34f))
            .withClip(true)
            .withFlexGrow(1);

        Component compileButton = Themed.button("Compile", Em.of(8f), Variant.PRIMARY, 0);

        Component editorScroll = new Component.Scroll(
            null, null, Em.of(0.5f), EDITOR_BG, editor, false, 1);

        Component leftPanel = new Component.Flex(
            Em.of(38f), null, Em.of(0.6f), PANEL_BG,
            Direction.COLUMN, JustifyContent.START, AlignItems.STRETCH, Em.of(0.5f),
            List.of(title, editorScroll, compileButton),
            false, 0
        );

        Component.SceneView scene = new Component.SceneView(
            null, null, Em.ZERO, VIEWPORT_BG, true, 1);

        Component contentRow = new Component.Flex(
            null, null, Em.of(0.5f), FRAME_BG,
            Direction.ROW, JustifyContent.START, AlignItems.STRETCH, Em.of(0.5f),
            List.of(leftPanel, scene),
            false, 1
        );

        // Bottom status bar: a full-width Flex strip wrapping a single Text,
        // composed from stock Dasum components (no framework change). It spans
        // the window because the COLUMN root stretches it on the cross axis;
        // a null height lets it size to its content (one text line + padding).
        Component.Text status = new Component.Text("Ready. Drag to orbit · scroll to zoom.",
            Em.of(0.85f), STATUS_FG);
        Component statusBar = new Component.Flex(
            null, null, Em.of(0.4f), PANEL_BG,
            Direction.ROW, JustifyContent.START, AlignItems.CENTER, Em.ZERO,
            List.of(status),
            false, 0
        );

        Component root = new Component.Flex(
            null, null, Em.of(0.5f), FRAME_BG,
            Direction.COLUMN, JustifyContent.START, AlignItems.STRETCH, Em.of(0.5f),
            List.of(contentRow, statusBar),
            false, 0
        );

        return new Ui(root, editor, compileButton, status, scene);
    }

    private static void wireCompile(Ui ui, ModelRenderer renderer) {
        Handlers.onClick(ui.compileButton(), () -> {
            String source = TextStates.contentOf(ui.editor());
            ModelRenderer.CompileResult result = renderer.recompile(source);
            // The status Text's colour is fixed by its record; convey success
            // vs failure with a prefix so the single status line stays correct
            // either way (the last good program is kept on failure).
            String prefix = result.success() ? "OK — " : "ERROR — ";
            TextStates.setContent(ui.status(), prefix + result.log());
            System.out.println((result.success() ? "[compile ok] " : "[compile error] ") + result.log());
        });
    }

    // ---------- input ----------

    /** True while a left-drag that began over the scene viewport is in flight. */
    private static boolean orbiting = false;
    /** Last cursor position seen while orbiting, for per-move deltas. */
    private static double lastOrbitX = 0d, lastOrbitY = 0d;

    /**
     * Whether the point {@code (x, y)} falls inside the scene viewport's latest
     * laid-out rect. dasum-core has no per-leaf <em>drag</em> handler registry —
     * {@link Handlers} only carries click/focus/blur — so the studio hit-tests
     * the raw GLFW cursor/button callbacks against the SceneView's rect itself
     * for orbiting. Wheel zoom no longer hit-tests a raw callback: it rides the
     * framework {@link WheelRouter} and reuses this same rect test to decide
     * whether to claim the wheel. Same approach as dasum-vis's
     * {@code SceneViewController}, minus its SceneStates camera model (this
     * renderer doesn't use it) and its focus-gating (this app zooms on hover).
     */
    private static boolean overScene(Ui ui, double x, double y) {
        LayoutResult lr = LatestLayout.result();
        if (lr == null) return false;
        PixelRect rect = lr.rectOf(ui.scene());
        return rect != null && rect.contains((float) x, (float) y);
    }

    /**
     * Input wiring: focus + caret placement on click, character + editing keys
     * for the text editor, click activation for the Compile button, hover
     * tracking, and orbit/zoom camera control over the scene viewport. The
     * node-editor / table / overlay / tooltip controllers from the demo are
     * intentionally omitted — this app has none of those widgets.
     *
     * <p>Camera input is hit-tested against the SceneView rect: left-drag that
     * starts over the viewport orbits the camera (raw GLFW cursor/button
     * callbacks), and a {@link WheelRouter} handler zooms while the cursor is
     * over it — declining otherwise so the router scrolls the editor. Each only
     * redraws (via {@link Invalidator#invalidate}) when the camera actually
     * changed, so the idle viewport stays idle.
     */
    private static void wireInput(Window window, Ui ui, ModelRenderer renderer) {
        GlfwCallbacks.setKeyListener((win, key, scancode, action, mods) -> {
            InputState.setMods(mods);
            if (action != Glfw.GLFW_PRESS && action != Glfw.GLFW_REPEAT) return;
            boolean ctrl = (mods & Glfw.GLFW_MOD_CONTROL) != 0;
            boolean shift = (mods & Glfw.GLFW_MOD_SHIFT) != 0;

            if (ctrl && key == 'C' && TextInputController.onCopy(window.handle())) return;
            if (ctrl && key == 'X' && TextInputController.onCut(window.handle())) return;
            if (ctrl && key == 'V' && TextInputController.onPaste(window.handle())) return;
            if (ctrl && key == 'A' && TextInputController.onSelectAll()) return;
            if (ctrl && key == 'Z') {
                if (shift ? TextInputController.onRedo() : TextInputController.onUndo()) return;
            }
            if (ctrl && key == 'Y' && TextInputController.onRedo()) return;

            if (key == Glfw.GLFW_KEY_BACKSPACE && TextInputController.onBackspace(ctrl)) return;
            if (key == Glfw.GLFW_KEY_DELETE && TextInputController.onDelete(ctrl)) return;
            if (key == Glfw.GLFW_KEY_ENTER && TextInputController.onEnter()) return;
            if (key == Glfw.GLFW_KEY_TAB && TextInputController.onTab()) return;
            if (TextInputController.onKey(key, shift, ctrl)) return;

            if (key == Glfw.GLFW_KEY_ESCAPE && action == Glfw.GLFW_PRESS) {
                if (FocusState.focused() != null) {
                    FocusState.clear();
                } else {
                    window.requestClose();
                    Invalidator.invalidate();
                }
            }
        });

        GlfwCallbacks.setCharListener((win, codepoint) -> TextInputController.onCharInput(codepoint));

        GlfwCallbacks.setCursorPosListener((win, x, y) -> {
            InputState.updateMousePos(x, y);
            if (orbiting) {
                // Drag over the viewport orbits the camera. Redraw only when
                // the camera moved (pitch may be clamped to a no-op).
                if (renderer.orbit(x - lastOrbitX, y - lastOrbitY)) {
                    Invalidator.invalidate();
                }
                lastOrbitX = x;
                lastOrbitY = y;
                return;
            }
            LayoutResult lr = LatestLayout.result();
            Component root = LatestLayout.root();
            if (lr == null || root == null) return;
            Component hit = HitTest.test(root, lr, (float) x, (float) y);
            HoverState.update(hit);
            TextInputController.onCursorMove(hit, x, y);
        });

        // Camera zoom rides on the framework's WheelRouter instead of owning
        // the GLFW scroll callback. Window.create installs the router (it owns
        // the single scroll listener and routes the wheel to scroll containers
        // — the editor's Component.Scroll included — automatically), so this
        // app only *adds* a handler for its viewport zoom. Over the scene the
        // handler consumes the wheel (returns true) and dollies the camera;
        // anywhere else it declines (returns false) and the router's built-in
        // terminal step scrolls the editor. No more clobbering dasum's own
        // scroll dispatch by replacing the listener.
        WheelRouter.addHandler(WHEEL_PRIORITY, e -> {
            if (!overScene(ui, e.mouseXPx(), e.mouseYPx())) return false;
            // Redraw only when the distance actually changed (clamp no-op).
            if (renderer.zoom(e.rawYOff())) {
                Invalidator.invalidate();
            }
            return true;
        });

        GlfwCallbacks.setMouseButtonListener((win, button, action, mods) -> {
            InputState.setMods(mods);
            if (button != Glfw.GLFW_MOUSE_BUTTON_LEFT) return;
            boolean pressed = (action == Glfw.GLFW_PRESS);
            InputState.setLeftButtonHeld(pressed);
            boolean shift = (mods & Glfw.GLFW_MOD_SHIFT) != 0;

            LayoutResult lr = LatestLayout.result();
            Component root = LatestLayout.root();

            if (pressed) {
                Component hit = (lr != null && root != null)
                    ? HitTest.test(root, lr, (float) InputState.mouseX(), (float) InputState.mouseY())
                    : null;
                if (hit == ui.scene()) {
                    // Begin an orbit drag: the camera, not the editor, owns
                    // this gesture. Move focus onto the viewport (clears the
                    // editor caret) and arm the cursor-pos delta tracking.
                    orbiting = true;
                    lastOrbitX = InputState.mouseX();
                    lastOrbitY = InputState.mouseY();
                    pressTarget = null;
                    FocusState.set(hit);
                    return;
                }
                pressTarget = hit;
                if (hit != null) FocusState.set(hit);
                else FocusState.clear();
                TextInputController.onMouseDown(hit, InputState.mouseX(), InputState.mouseY(), shift);
            } else {
                orbiting = false;
                Component released = (lr != null && root != null)
                    ? HitTest.test(root, lr, (float) InputState.mouseX(), (float) InputState.mouseY())
                    : null;
                if (pressTarget != null && released == pressTarget) {
                    Handlers.activate(pressTarget, root);
                }
                pressTarget = null;
            }
        });
    }

    // ---------- CLI ----------

    /**
     * Parse {@code --frames N}: render exactly {@code N} frames, then exit. A
     * non-positive or absent value means "run until the window is closed".
     */
    private static int parseFrameLimit(String[] args) {
        for (int i = 0; i < args.length - 1; i++) {
            if ("--frames".equals(args[i])) {
                try {
                    return Integer.parseInt(args[i + 1]);
                } catch (NumberFormatException e) {
                    System.err.println("Ignoring non-integer --frames value: " + args[i + 1]);
                    return 0;
                }
            }
        }
        return 0;
    }
}
