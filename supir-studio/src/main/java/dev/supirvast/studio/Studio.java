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
 * <p>The left panel is an editable {@link Component.Text} holding fragment-shader
 * source, a <strong>Compile</strong> button, and a status line. The right panel
 * is a {@link Component.SceneView} whose {@link ModelRenderer} draws an
 * auto-rotating UV sphere with a fixed MVP vertex shader and the editor's
 * fragment shader. Pressing Compile rebuilds the GL program from the editor text
 * via {@link ShaderUtil#buildProgram}; on failure the compile log is shown in the
 * status line and the last good program is kept (the app never crashes on bad
 * GLSL).
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
                wireInput(window);

                long[] renderedFrames = {0};
                EventLoop loop = new EventLoop(window, () -> {
                    renderFrame(window, ui.root(), batcher);
                    renderedFrames[0]++;
                    if (frameLimit > 0 && renderedFrames[0] >= frameLimit) {
                        window.requestClose();
                    }
                    // Drive continuous animation: re-dirty so the loop renders
                    // the next frame rather than going idle (the model rotates).
                    Invalidator.invalidate();
                });
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
     * The assembled UI plus the handles the compile wiring needs: the
     * {@code editor} to read GLSL from, the {@code compileButton} to attach the
     * click handler to, and the {@code status} line to write results to.
     */
    private record Ui(Component root, Component.Text editor, Component compileButton, Component.Text status) {}

    private static Ui buildUi(String initialFragment) {
        Component.Text title = new Component.Text("Fragment shader (GLSL)", Em.of(1.0f), LABEL_FG);

        Component.Text editor = new Component.Text(initialFragment, Em.of(0.85f), EDITOR_FG)
            .withEditable(true)
            .withAcceptsTab(true)
            .withWrapWidth(Em.of(34f))
            .withClip(true)
            .withFlexGrow(1);

        Component compileButton = Themed.button("Compile", Em.of(8f), Variant.PRIMARY, 0);

        Component.Text status = new Component.Text("Ready.", Em.of(0.85f), STATUS_FG)
            .withWrapWidth(Em.of(34f));

        Component editorScroll = new Component.Scroll(
            null, null, Em.of(0.5f), EDITOR_BG, editor, false, 1);

        Component leftPanel = new Component.Flex(
            Em.of(38f), null, Em.of(0.6f), PANEL_BG,
            Direction.COLUMN, JustifyContent.START, AlignItems.STRETCH, Em.of(0.5f),
            List.of(title, editorScroll, compileButton, status),
            false, 0
        );

        Component sceneView = new Component.SceneView(
            null, null, Em.ZERO, VIEWPORT_BG, true, 1);

        Component root = new Component.Flex(
            null, null, Em.of(0.5f), FRAME_BG,
            Direction.ROW, JustifyContent.START, AlignItems.STRETCH, Em.of(0.5f),
            List.of(leftPanel, sceneView),
            false, 0
        );

        return new Ui(root, editor, compileButton, status);
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

    /**
     * Minimal input wiring for Phase 1: focus + caret placement on click,
     * character + editing keys for the text editor, click activation for the
     * Compile button, and hover tracking. The node-editor / table / overlay /
     * tooltip controllers from the demo are intentionally omitted — this app
     * has none of those widgets.
     */
    private static void wireInput(Window window) {
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
            LayoutResult lr = LatestLayout.result();
            Component root = LatestLayout.root();
            if (lr == null || root == null) return;
            Component hit = HitTest.test(root, lr, (float) x, (float) y);
            HoverState.update(hit);
            TextInputController.onCursorMove(hit, x, y);
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
                pressTarget = hit;
                if (hit != null) FocusState.set(hit);
                else FocusState.clear();
                TextInputController.onMouseDown(hit, InputState.mouseX(), InputState.mouseY(), shift);
            } else {
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
