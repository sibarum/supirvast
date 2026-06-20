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
import sibarum.dasum.gui.core.input.TabsController;
import sibarum.dasum.gui.core.input.TextInputController;
import sibarum.dasum.gui.core.input.TextStates;
import sibarum.dasum.gui.core.input.wheel.WheelRouter;
import sibarum.dasum.gui.core.layout.HitTest;
import sibarum.dasum.gui.core.layout.LatestLayout;
import sibarum.dasum.gui.core.layout.Layout;
import sibarum.dasum.gui.core.layout.LayoutResult;
import sibarum.dasum.gui.core.layout.PixelRect;
import sibarum.dasum.gui.core.layout.Render;
import sibarum.dasum.gui.core.overlay.Anchor;
import sibarum.dasum.gui.core.overlay.OverlayStack;
import sibarum.dasum.gui.core.reactive.Property;
import sibarum.dasum.gui.core.render.Batcher;
import sibarum.dasum.gui.core.render.Color;
import sibarum.dasum.gui.core.render.CustomRenderers;
import sibarum.dasum.gui.core.render.DrawCommand;
import sibarum.dasum.gui.core.render.Projection;
import sibarum.dasum.gui.core.render.ShaderUtil;
import sibarum.dasum.gui.core.render.Texture;
import sibarum.dasum.gui.core.text.AtlasData;
import sibarum.dasum.gui.core.text.FontGroup;
import sibarum.dasum.gui.core.text.FontGroups;
import sibarum.dasum.gui.core.theme.Theme;
import sibarum.dasum.gui.core.theme.Themed;
import sibarum.dasum.gui.core.theme.Variant;
import sibarum.dasum.gui.core.window.Window;
import sibarum.dasum.gui.natives.gl.Gl;
import sibarum.dasum.gui.natives.glfw.Glfw;
import sibarum.dasum.gui.natives.glfw.GlfwCallbacks;
import sibarum.dasum.gui.natives.nfd.Nfd;
import sibarum.dasum.gui.vis.DasumVis;
import dev.supirvast.vastir.core.ShaderStage;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static sibarum.dasum.gui.natives.gl.Gl.GL_COLOR_BUFFER_BIT;

/**
 * The supir-studio live shader playground: a windowed dasumGUIshi app whose left panel is a tabbed
 * <strong>Supir</strong> editor (Vertex / Fragment) and whose right panel is a 3D preview with the
 * action buttons docked at its top-right.
 *
 * <p>Each tab edits one stage's Supir source — the textual form of the core IR. <strong>Compile</strong> runs
 * the whole SupirVast pipeline per stage (Supir → core → SPIR-V → spirv-val → GLSL 330 → GL program) and links
 * the two; on any failure the message is shown in the bottom status bar and the last good program is kept.
 * <strong>Export…</strong> opens a format picker (checkboxes for SPIR-V binary/assembly, GLSL, HLSL, Metal;
 * GLSL by default) and writes both stages to a folder chosen via a native dialog.
 *
 * <p>A {@code --frames N} option renders {@code N} frames and exits cleanly, for non-interactive smoke testing.
 */
public final class Studio {

    private static final Color FRAME_BG    = new Color(0.05f, 0.06f, 0.09f, 1f);
    private static final Color PANEL_BG     = new Color(0.10f, 0.12f, 0.15f, 1f);
    private static final Color EDITOR_BG     = new Color(0.07f, 0.09f, 0.13f, 1f);
    private static final Color VIEWPORT_BG   = new Color(0.04f, 0.05f, 0.08f, 1f);
    private static final Color MENU_BG       = new Color(0.13f, 0.15f, 0.20f, 1f);
    private static final Color LABEL_FG      = new Color(0.92f, 0.94f, 0.97f, 1f);
    private static final Color EDITOR_FG     = new Color(0.85f, 0.90f, 0.85f, 1f);
    private static final Color STATUS_FG     = new Color(0.70f, 0.78f, 0.88f, 1f);

    /**
     * Priority for the viewport-zoom wheel handler, above the router's built-in DataTable handler — matching
     * the value dasum-vis's SceneViewController uses.
     */
    private static final int WHEEL_PRIORITY = 100;

    /** Press target for click dispatch; cleared on release (mirrors the demo). */
    private static Component pressTarget = null;

    private Studio() {}

    public static void main(String[] args) {
        int frameLimit = parseFrameLimit(args);

        // The editor holds Supir for both stages; the live program is GLSL. Compile the default Supir up front
        // so the first frame draws the real Supir-derived shaders. Each stage independently falls back to a
        // built-in GLSL shader if it can't be compiled (e.g. the native toolchain is missing); the two stages
        // interoperate either way because both use the same varying locations.
        String vertexSupir = ShaderUtil.readResource("/shaders/default.vert.supir");
        String fragmentSupir = ShaderUtil.readResource("/shaders/default.supir");
        String fallbackVertex = ShaderUtil.readResource("/shaders/model.vert");
        String fallbackFragment = ShaderUtil.readResource("/shaders/default.frag");

        SupirShaderCompiler compiler = new SupirShaderCompiler();
        SupirShaderCompiler.Result v = compiler.compileVertex(vertexSupir);
        SupirShaderCompiler.Result f = compiler.compileFragment(fragmentSupir);
        boolean vertexUsable = v.ok() && v.mvpUniform() != null;
        String initialVertex = vertexUsable ? v.glsl() : fallbackVertex;
        String initialFragment = f.ok() ? f.glsl() : fallbackFragment;
        String mvpUniform = vertexUsable ? v.mvpUniform() : "uMvp";
        if (!vertexUsable || !f.ok()) {
            System.err.println("Default Supir shaders did not fully compile; using built-in GLSL fallback:"
                    + (vertexUsable ? "" : "\nvertex: " + (v.ok() ? "MVP uniform not found" : v.error()))
                    + (f.ok() ? "" : "\nfragment: " + f.error()));
        }

        // Export format selection — persists across menu opens. GLSL only by default.
        Map<ExportFormat, Property<Boolean>> exportSelection = new EnumMap<>(ExportFormat.class);
        for (ExportFormat format : ExportFormat.values()) {
            exportSelection.put(format, new Property<>(format == ExportFormat.GLSL));
        }

        Geometry sphere = Geometry.uvSphere(48, 64);
        ModelRenderer modelRenderer = new ModelRenderer(sphere, initialVertex, initialFragment, mvpUniform);

        try (GlfwContext ctx = GlfwContext.init();
             Window window = Window.create(1280, 800, "Supir Studio — Shader Playground");
             Batcher batcher = new Batcher();
             ModelRenderer ownedRenderer = modelRenderer) {

            Gl.load();
            batcher.init();
            DasumVis.init();
            EmContext.setDpiScale(window.contentScaleX());

            CustomRenderers.register(Component.SceneView.class, ownedRenderer.asRenderer());

            try (Texture primaryTexture = Texture.fromPngResource("/dasum/atlas/primary.png")) {
                AtlasData primaryAtlas = AtlasData.loadFromResource("/dasum/atlas/primary.json");
                FontGroups.register(FontGroup.of(FontGroups.DEFAULT, primaryAtlas, primaryTexture));

                Ui ui = buildUi(vertexSupir, fragmentSupir);
                wireCompile(ui, ownedRenderer, compiler);
                wireExportMenu(ui, compiler, exportSelection);
                wireInput(window, ui, ownedRenderer);

                long[] renderedFrames = {0};
                EventLoop loop = new EventLoop(window, () -> {
                    renderFrame(window, ui.root(), batcher);
                    renderedFrames[0]++;
                    if (frameLimit > 0 && renderedFrames[0] >= frameLimit) {
                        window.requestClose();
                    }
                });

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

    /** Render one frame: clear, lay out (incl. overlays), render the tree, then any overlays above it. */
    private static void renderFrame(Window window, Component root, Batcher batcher) {
        int fbW = window.framebufferWidth();
        int fbH = window.framebufferHeight();
        float[] projection = Projection.orthoTopLeft(fbW, fbH);

        Gl.glViewport(0, 0, fbW, fbH);
        Gl.glClearColor(0.03f, 0.03f, 0.05f, 1f);
        Gl.glClear(GL_COLOR_BUFFER_BIT);

        PixelRect viewport = new PixelRect(0f, 0f, fbW, fbH);
        LayoutResult mainLayout = Layout.compute(root, viewport);
        // Merge overlay layouts so hit-testers and Render share one LayoutResult.
        Map<Component, PixelRect> mergedRects = new IdentityHashMap<>(mainLayout.rects());
        OverlayStack.layoutInto(mergedRects, viewport);
        LayoutResult layout = new LayoutResult(mergedRects);
        LatestLayout.store(root, layout);

        batcher.beginFrame(fbH);
        Render.render(root, layout, batcher, projection);
        // Each z-layer needs its own flush so its text doesn't race ahead of a later layer's fills.
        if (OverlayStack.isActive()) {
            batcher.flush(projection);
            if (OverlayStack.anyModal()) {
                batcher.submit(new DrawCommand.ColoredQuad(
                        viewport.x(), viewport.y(), viewport.width(), viewport.height(),
                        Theme.overlayBackdrop()));
                batcher.flush(projection);
            }
            for (OverlayStack.Overlay o : OverlayStack.active()) {
                Render.render(o.component(), layout, batcher, projection);
                batcher.flush(projection);
            }
        }
        batcher.endFrame(projection);
    }

    // ---------- UI ----------

    /**
     * The assembled UI plus the handles the wiring needs: the two stage {@code editor}s to read Supir from, the
     * {@code compileButton} / {@code exportButton} to attach click handlers to, the {@code status} line to
     * write results to, and the {@code scene} to hit-test mouse input against for camera control.
     */
    private record Ui(Component root, Component.SceneView scene, Component.Text status,
                      Component compileButton, Component exportButton,
                      Component.Text vertexEditor, Component.Text fragmentEditor) {}

    private static Component.Text editor(String initialSource) {
        return new Component.Text(initialSource, Em.of(0.85f), EDITOR_FG)
            .withEditable(true)
            .withAcceptsTab(true)
            .withLineNumbers(true)
            .withWrapWidth(Em.of(34f))
            .withClip(true)
            .withFlexGrow(1);
    }

    private static Ui buildUi(String vertexSupir, String fragmentSupir) {
        // One editor per stage; the tab switch shows one at a time. Each editor keeps its own text, so Compile
        // and Export read both regardless of which tab is active.
        Component.Text vertexEditor = editor(vertexSupir);
        Component.Text fragmentEditor = editor(fragmentSupir);

        Component vertexScroll = new Component.Scroll(null, null, Em.of(0.5f), EDITOR_BG, vertexEditor, false, 1);
        Component fragmentScroll =
            new Component.Scroll(null, null, Em.of(0.5f), EDITOR_BG, fragmentEditor, false, 1);

        // Tabs replace the old title + "switch stage" button. The header strip is the Vertex/Fragment nav.
        Property<Integer> activeTab = new Property<>(0);
        Component tabs = Themed.tabs(
            Em.of(38f), null,
            Em.of(2.2f), Em.of(1.2f), Em.of(0.4f),
            Em.of(0.95f),
            List.of(
                new Component.Tabs.TabPanel("Vertex", vertexScroll),
                new Component.Tabs.TabPanel("Fragment", fragmentScroll)
            ),
            activeTab,
            Variant.PRIMARY
        );

        // Action buttons dock at the top-right of the viewport side, freeing the editor's full height.
        Component compileButton = Themed.button("Compile", Em.of(7f), Variant.PRIMARY, 0);
        Component exportButton = Themed.button("Export…", Em.of(7f), Variant.INFO, 0);
        Component buttonRow = new Component.Flex(
            null, Em.of(2.4f), Em.of(0.2f), VIEWPORT_BG,
            Direction.ROW, JustifyContent.END, AlignItems.CENTER, Em.of(0.5f),
            List.of(compileButton, exportButton),
            false, 0
        );

        Component.SceneView scene = new Component.SceneView(null, null, Em.ZERO, VIEWPORT_BG, true, 1);

        Component rightColumn = new Component.Flex(
            null, null, Em.ZERO, FRAME_BG,
            Direction.COLUMN, JustifyContent.START, AlignItems.STRETCH, Em.of(0.4f),
            List.of(buttonRow, scene),
            false, 1
        );

        Component contentRow = new Component.Flex(
            null, null, Em.of(0.5f), FRAME_BG,
            Direction.ROW, JustifyContent.START, AlignItems.STRETCH, Em.of(0.5f),
            List.of(tabs, rightColumn),
            false, 1
        );

        // Bottom status bar: a fixed-height strip (flexGrow 0) so it stays pinned and fully visible — a
        // null-height flex measures as fill, not fit-content, which is what pushed it off the window before.
        Component.Text status = new Component.Text("Ready. Drag to orbit · scroll to zoom.",
            Em.of(0.85f), STATUS_FG);
        Component statusBar = new Component.Flex(
            null, Em.of(1.9f), Em.of(0.4f), PANEL_BG,
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

        return new Ui(root, scene, status, compileButton, exportButton, vertexEditor, fragmentEditor);
    }

    /**
     * Compile both stages from Supir and link them. The whole pipeline runs per stage; only if both succeed is
     * the live program rebuilt. On any failure the message names the offending stage; the last good program is
     * kept.
     */
    private static void wireCompile(Ui ui, ModelRenderer renderer, SupirShaderCompiler compiler) {
        Handlers.onClick(ui.compileButton(), () -> {
            SupirShaderCompiler.Result v = compiler.compileVertex(TextStates.contentOf(ui.vertexEditor()));
            if (!v.ok()) {
                fail(ui, "vertex", v.error());
                return;
            }
            if (v.mvpUniform() == null) {
                fail(ui, "vertex", "could not locate the MVP matrix uniform in the generated GLSL");
                return;
            }
            SupirShaderCompiler.Result f = compiler.compileFragment(TextStates.contentOf(ui.fragmentEditor()));
            if (!f.ok()) {
                fail(ui, "fragment", f.error());
                return;
            }

            ModelRenderer.CompileResult result = renderer.recompile(v.glsl(), f.glsl(), v.mvpUniform());
            String prefix = result.success() ? "OK — " : "ERROR — ";
            TextStates.setContent(ui.status(), prefix + result.log());
            System.out.println((result.success() ? "[compile ok] " : "[compile error] ") + result.log());
        });
    }

    // ---------- export menu ----------

    private static void wireExportMenu(Ui ui, SupirShaderCompiler compiler,
                                       Map<ExportFormat, Property<Boolean>> selection) {
        Handlers.onClick(ui.exportButton(), () -> openExportMenu(ui, compiler, selection));
    }

    /** Build and show the modal format-picker: a checkbox per format plus Save / Cancel. */
    private static void openExportMenu(Ui ui, SupirShaderCompiler compiler,
                                       Map<ExportFormat, Property<Boolean>> selection) {
        List<Component> rows = new ArrayList<>();
        rows.add(new Component.Text("Export formats", Em.of(1.0f), LABEL_FG));
        for (ExportFormat format : ExportFormat.values()) {
            Component checkbox = Themed.checkbox(Em.of(1.1f), selection.get(format), Variant.PRIMARY);
            Component label = new Component.Text(format.label(), Em.of(0.9f), LABEL_FG);
            rows.add(new Component.Flex(
                null, Em.of(1.7f), Em.ZERO, MENU_BG,
                Direction.ROW, JustifyContent.START, AlignItems.CENTER, Em.of(0.6f),
                List.of(checkbox, label),
                false, 0));
        }

        Component cancel = Themed.button("Cancel", Em.of(6f), Variant.DEFAULT, 0);
        Component save = Themed.button("Save", Em.of(6f), Variant.PRIMARY, 0);
        rows.add(new Component.Flex(
            null, Em.of(2.2f), Em.of(0.2f), MENU_BG,
            Direction.ROW, JustifyContent.END, AlignItems.CENTER, Em.of(0.5f),
            List.of(cancel, save),
            false, 0));

        Component panel = new Component.Flex(
            Em.of(22f), Em.of(17f), Em.of(0.9f), MENU_BG,
            Direction.COLUMN, JustifyContent.START, AlignItems.STRETCH, Em.of(0.5f),
            rows,
            false, 0);

        Handlers.onClick(cancel, OverlayStack::pop);
        Handlers.onClick(save, () -> {
            OverlayStack.pop();
            exportSelected(ui, compiler, selectedFormats(selection));
        });
        OverlayStack.push(new OverlayStack.Overlay(panel, Anchor.CENTER, true));
    }

    private static Set<ExportFormat> selectedFormats(Map<ExportFormat, Property<Boolean>> selection) {
        Set<ExportFormat> chosen = EnumSet.noneOf(ExportFormat.class);
        selection.forEach((format, prop) -> {
            if (Boolean.TRUE.equals(prop.get())) {
                chosen.add(format);
            }
        });
        return chosen;
    }

    /**
     * Compile both stages to every distributable form, pick a folder (native dialog), and write the selected
     * formats as {@code shader.vert.*} / {@code shader.frag.*}. Stages compile first, so a Supir error is
     * reported before the dialog opens.
     */
    private static void exportSelected(Ui ui, SupirShaderCompiler compiler, Set<ExportFormat> formats) {
        if (formats.isEmpty()) {
            TextStates.setContent(ui.status(), "Export: select at least one format.");
            return;
        }

        SupirShaderCompiler.Export v = compiler.exportStage(TextStates.contentOf(ui.vertexEditor()),
                ShaderStage.VERTEX);
        if (!v.ok()) {
            fail(ui, "vertex", v.error());
            return;
        }
        SupirShaderCompiler.Export f = compiler.exportStage(TextStates.contentOf(ui.fragmentEditor()),
                ShaderStage.FRAGMENT);
        if (!f.ok()) {
            fail(ui, "fragment", f.error());
            return;
        }

        Nfd.ensureInit();
        String folder = Nfd.pickFolder(MemorySegment.NULL, Nfd.NFD_WINDOW_HANDLE_TYPE_UNSET, null);
        if (folder == null) {
            TextStates.setContent(ui.status(), "Export cancelled.");
            return;
        }

        try {
            Path dir = Path.of(folder);
            List<Path> written = new ArrayList<>();
            written.addAll(ShaderExport.write(dir, "shader.vert", v, formats));
            written.addAll(ShaderExport.write(dir, "shader.frag", f, formats));
            TextStates.setContent(ui.status(), "OK — exported " + written.size() + " files to " + dir);
            System.out.println("[export] wrote " + written.size() + " files to " + dir);
        } catch (IOException e) {
            TextStates.setContent(ui.status(), "ERROR — export write failed: " + e.getMessage());
            System.out.println("[export error] " + e);
        }
    }

    private static void fail(Ui ui, String stage, String error) {
        TextStates.setContent(ui.status(), "ERROR (" + stage + ") — " + firstLine(error));
        System.out.println("[supir error] " + stage + ": " + error);
    }

    /** The first line of a (possibly multi-line) message — the status bar is a single line. */
    private static String firstLine(String message) {
        int nl = message.indexOf('\n');
        return nl < 0 ? message : message.substring(0, nl);
    }

    // ---------- input ----------

    /** True while a left-drag that began over the scene viewport is in flight. */
    private static boolean orbiting = false;
    /** Last cursor position seen while orbiting, for per-move deltas. */
    private static double lastOrbitX = 0d, lastOrbitY = 0d;

    /** Whether the point falls inside the scene viewport's latest laid-out rect. */
    private static boolean overScene(Ui ui, double x, double y) {
        LayoutResult lr = LatestLayout.result();
        if (lr == null) return false;
        PixelRect rect = lr.rectOf(ui.scene());
        return rect != null && rect.contains((float) x, (float) y);
    }

    /**
     * Input wiring: text editing for the focused stage editor, tab navigation, the export overlay (modal, with
     * click-outside dismissal), button clicks, and orbit/zoom camera control over the scene viewport. The
     * dispatch order mirrors the demo: tab cells (synthesized, not components) get first refusal, then the
     * overlay captures input while active, then normal hit-testing drives the editor / camera.
     */
    private static void wireInput(Window window, Ui ui, ModelRenderer renderer) {
        GlfwCallbacks.setKeyListener((win, key, scancode, action, mods) -> {
            InputState.setMods(mods);
            if (action != Glfw.GLFW_PRESS && action != Glfw.GLFW_REPEAT) return;
            boolean ctrl = (mods & Glfw.GLFW_MOD_CONTROL) != 0;
            boolean shift = (mods & Glfw.GLFW_MOD_SHIFT) != 0;

            if (key == Glfw.GLFW_KEY_ESCAPE && action == Glfw.GLFW_PRESS && OverlayStack.isActive()) {
                OverlayStack.pop();
                Invalidator.invalidate();
                return;
            }

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
            Component hitRoot = OverlayStack.activeInputRoot(root);
            Component hit = HitTest.test(hitRoot, lr, (float) x, (float) y);
            HoverState.update(hit);
            TextInputController.onCursorMove(hit, x, y);
            TabsController.onCursorMove(x, y);
        });

        WheelRouter.addHandler(WHEEL_PRIORITY, e -> {
            if (OverlayStack.isActive()) return false;
            if (!overScene(ui, e.mouseXPx(), e.mouseYPx())) return false;
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
            double mx = InputState.mouseX();
            double my = InputState.mouseY();

            if (pressed) {
                // Tab cells aren't components — give them first refusal (only when no overlay is up).
                if (!OverlayStack.isActive() && TabsController.onMouseDown(mx, my)) {
                    pressTarget = null;
                    return;
                }
                // Overlay capture: route the press through the topmost overlay; click-outside a modal pops it.
                if (OverlayStack.isActive()) {
                    LayoutResult lr = LatestLayout.result();
                    if (OverlayStack.isOutsideTopmost(lr, (float) mx, (float) my)) {
                        if (OverlayStack.anyModal()) OverlayStack.pop();
                        pressTarget = null;
                        Invalidator.invalidate();
                        return;
                    }
                    Component overlayRoot = OverlayStack.activeInputRoot(null);
                    Component hit = (lr != null && overlayRoot != null)
                        ? HitTest.test(overlayRoot, lr, (float) mx, (float) my) : null;
                    pressTarget = hit;
                    if (hit != null) FocusState.set(hit);
                    TextInputController.onMouseDown(hit, mx, my, shift);
                    return;
                }
                // Normal: hit-test the main tree for camera / editor.
                LayoutResult lr = LatestLayout.result();
                Component root = LatestLayout.root();
                Component hit = (lr != null && root != null)
                    ? HitTest.test(root, lr, (float) mx, (float) my) : null;
                if (hit == ui.scene()) {
                    orbiting = true;
                    lastOrbitX = mx;
                    lastOrbitY = my;
                    pressTarget = null;
                    FocusState.set(hit);
                    return;
                }
                pressTarget = hit;
                if (hit != null) FocusState.set(hit);
                else FocusState.clear();
                TextInputController.onMouseDown(hit, mx, my, shift);
            } else {
                orbiting = false;
                LayoutResult lr = LatestLayout.result();
                Component dispatchRoot = OverlayStack.activeInputRoot(LatestLayout.root());
                Component released = (lr != null && dispatchRoot != null)
                    ? HitTest.test(dispatchRoot, lr, (float) mx, (float) my) : null;
                if (pressTarget != null && released == pressTarget) {
                    Handlers.activate(pressTarget, dispatchRoot);
                }
                pressTarget = null;
            }
        });
    }

    // ---------- CLI ----------

    /**
     * Parse {@code --frames N}: render exactly {@code N} frames, then exit. A non-positive or absent value
     * means "run until the window is closed".
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
