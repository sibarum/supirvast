# supir-studio — live shader playground

Edit a shader in a text panel, click **Compile**, and see it on a 3D model in an OpenGL viewport — instantly.
A GUI front door to the whole SupirVast pipeline.

```
Supir source ──parse──▶ core IR ──CoreToSpirv──▶ SPIR-V ──spirv-cross──▶ GLSL 330 ──▶ OpenGL program
   (textarea)                                                                              │
                                                                                  rendered on a model
                                                                                  in a dasumGUIshi viewport
```

## Why GLSL in the middle

The UI lib (`dasumGUIshi`) is **OpenGL 3.3 core**, which can't ingest SPIR-V directly (that's GL 4.6's
`ARB_gl_spirv`). So we cross-compile **SPIR-V → GLSL `#version 330`** with `spirv-cross` — already bundled in
`vastir-tools` (`NativeTools.crossCompile(..., GLSL)`; will need a `--version 330` flag added). This also nicely
exercises the cross-compile path. `CoreToSpirv` and `spirv-cross` are pure functions, so the whole
`source → GL-ready GLSL` chain is testable headlessly before any UI.

## Build prerequisite

`dasumGUIshi` (`sibarum.dasum.gui`, JDK 25, Maven) must be installed to the local repo:

```sh
mvn -f ../dasumGUIshi/pom.xml install -DskipTests
```

Run the studio with `--enable-native-access=ALL-UNNAMED` (dasumGUIshi uses Panama FFM bindings, not LWJGL).

## dasumGUIshi integration cheatsheet (from the source)

- **Window/loop**: `GlfwContext.init()`, `Window.create(w,h,title)`, `Gl.load()`, `Batcher`, an event loop
  (see `dasum-mvp-demo` → `sibarum.dasum.gui.demo.App` — the canonical reference app).
- **Editor (textarea)**: `Component.Text` with `editable=true`, `acceptsTab=true`, a `wrapWidth`; read/write via
  `TextStates.contentOf(c)` / `TextStates.setContent(c, s)` / `TextStates.onContentChange(c, …)`.
- **Button**: `Themed.button(label, width, Variant.PRIMARY, flexGrow, () -> {…})` (or `Handlers.onClick`).
- **3D viewport**: `Component.SceneView` + `CustomRenderers.register(SceneView.class, (c, rect, batcher, proj) ->
  …)`; inside, use `try (ViewportScope scope = new ViewportScope(batcher, proj, rect, /*depth*/ true)) { … }`
  which flushes the batcher, scissors to the rect, and restores GL state on close. Issue normal GL 3.3 calls
  (VAO/VBO/`glUseProgram`/`glDrawElements`) via the `Gl` binding.
- **Shader compile helper**: `ShaderUtil.buildProgram(vertexSrc, fragmentSrc)` (compile + link GLSL); status/log
  for surfacing errors back into the UI.
- **Layout**: flexbox-style `Component.Flex` (row/column, `flexGrow`), em units.
- **Fonts**: text needs an MSDF atlas (`FontGroups.register(FontGroup.of(DEFAULT, atlas, tex))`). The demo
  generates `primary`/`icons` atlases with the `dasum-msdf-maven-plugin` (see its pom; fonts under
  `dasum-mvp-demo/fonts/`). **Phase 1 must either run that plugin here or reuse a generated atlas.**

## Plan

### Phase 1 — GUI shell + live GL loop (input = GLSL) — *next*
- [ ] Module wiring: font atlas available (msdf plugin or reuse), `mvn exec:java` with
      `--enable-native-access=ALL-UNNAMED`.
- [ ] Window with a layout: left = editable `Component.Text` (the fragment shader) + a **Compile** button +
      a status line; right = a `Component.SceneView`.
- [ ] `SceneView` custom renderer: a generated model (sphere/cube VAO+VBO), a rotating MVP (JOML), depth test,
      drawing with the current GL program. Fixed vertex shader (MVP + world normal/uv varyings); the editor
      holds the **fragment** shader (default: normal/checker coloring).
- [ ] Compile button: `ShaderUtil.buildProgram(fixedVertex, editorText)` → swap the live program; on failure,
      show the compile log in the status line (don't crash). **Checkpoint: edit GLSL → Compile → model updates.**

### Phase 2 — wire in the SupirVast pipeline (input = Supir)
- [x] Build the **`supir`** module: the textual form of the `core` IR — a flat, high-level assembly (one op
      per line, named intermediates, structured `if`/`loop`), two-way (`Supir.parseModule` text→`core`,
      `Supir.print` `core`→text), with `line:col` diagnostics. (Done — grammar in `supir/README.md`; faithful
      to lowered semantics, printing is a normal form; testable headlessly.)
- [x] `NativeTools.crossCompile` gains a GLSL version target (`crossCompile(spirv, GLSL, 330)` → `--version 330
      --no-es --separate-shader-objects`, so interstage varyings carry explicit locations on a 3.3 context).
- [x] Studio compile path: editor (Supir) → `Supir.parseModule` → `core` → `CoreToSpirv` → `spirv-val` gate →
      `spirv-cross` GLSL 330 → `ShaderUtil.buildProgram` → render. (Done — `SupirShaderCompiler`. Generated
      stages link by varying location, since spirv-cross names interface variables opaquely; `model.vert` /
      `default.frag` are GLSL fallbacks used only if a default Supir stage can't be compiled at startup.
      Parse/validation/cross-compile/link errors surface in the status line; last good program kept on failure.)

### Both stages in Supir
- [x] The vertex stage is editable Supir too. The left panel is a **Vertex / Fragment tab strip** (each tab its
      own editor); Compile builds and links both. The vertex's MVP is a push constant, which spirv-cross emits
      as an opaquely-named `uniform` struct; since the dasum GL binding can't enumerate active uniforms,
      `SupirShaderCompiler` recovers the matrix uniform's name from the generated GLSL and the renderer sets it
      each frame. The two Supir stages link to each other by varying location.

### Export
- [x] An **Export…** button opens a format picker (a checkbox per format — SPIR-V binary, SPIR-V assembly,
      GLSL, HLSL, Metal — with **GLSL** selected by default, plus Save / Cancel). On Save it writes both stages
      to a folder picked via a native dialog (NFD — the `Nfd` binding is in `dasum-natives`; its native
      `nfd.dll` is the `dasum-nfd` runtime dependency), as `shader.{vert,frag}.{spv,spvasm,glsl,hlsl,metal}` for
      the selected formats: SPIR-V binary (from `CoreToSpirv`), SPIR-V assembly (`spirv-dis`), and
      GLSL/HLSL/Metal (`spirv-cross`). Stages are compiled first, so a Supir error is reported before any
      dialog. *OpenCL C is not a spirv-cross target, so it isn't emitted — but the exported `.spv` is directly
      consumable by OpenCL 2.1+ runtimes.*

### UI
- Tabbed editor (the tab strip replaces the old stage toggle + title). Action buttons (Compile, Export…) dock
  at the top-right over the viewport, giving the editor full height. A bottom status bar (fixed height, pinned)
  reports results. The export picker is a modal overlay (`OverlayStack`); Esc or a click outside dismisses it.

### Later
- Texture/cubemap channels; hot-reload on keystroke; save/load; multiple models; a Supir snippet library;
  a "dump core IR" view (GLSL → core → `Supir.print`).

## Status
Both stages are authored in Supir, edited via Vertex/Fragment tabs. Compile runs the full pipeline per stage
(Supir → core → SPIR-V → spirv-val → GLSL 330 → GL program) and links them to drive the previewed sphere.
Export… picks formats and writes both stages out as SPIR-V / SPIR-V assembly / GLSL / HLSL / Metal. Edit the
Supir, press Compile, see it change.
