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
- [ ] Build the **`supir`** module: an S-expression parser whose forms map 1:1 to the `core` AST
      (`Expr`/`Statement`/`Function`/`Type`/`PushConstants`/`InterfaceVar`/…), with a direct `Supir → core`
      lowering and clear parse/lower error witnesses. (This is the standalone TODO item; testable headlessly.)
- [ ] `NativeTools.crossCompile` gains a GLSL version target (`--version 330`).
- [ ] Studio compile path becomes: editor (Supir) → `supir` parse → `core` → `CoreToSpirv` → `spirv-val`
      gate → `spirv-cross` GLSL 330 → `ShaderUtil.buildProgram` → render. Parse/validation errors surface in
      the status line.

### Later
- Vertex-shader editing too; texture/cubemap channels; hot-reload on keystroke; save/load; multiple models;
  a Supir snippet library.

## Status
Scaffold only (module + this plan). Phase 1 not yet implemented.
