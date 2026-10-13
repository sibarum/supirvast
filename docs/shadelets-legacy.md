# Shadelets: removing what looks made obsolete

Status: **done**, 2026-10-12. Companion to [shadelets.md](shadelets.md). Written as a plan; what each step came to is
under [Done](#done) at the end.

## The two words

A **kind** is the canvas's old way of saying what a quad is: a number written into every vertex (0 box, 1 glyph,
2 shadow, 3 border ring, 4 lit box, 5 image, 6 icon field). The shader used to branch on it. Kinds are a closed set,
fixed in the framework, and draw one thing per quad: a card with a shadow and a border was three quads.

A **look** is the new way: an element names a look in the shadelet tables, and the look names a source (where the
edge is), modifiers (move the edge) and painters (draw relative to it, front to back). Looks are open (an application
adds shadelets), stack (many effects, one quad), and are data the shader reads, not branches in it.

Today each kind is a **fixed look** (`CoreLooks`): the canvas still writes the kind into the vertex and uses it to
choose that look. That bridge is what let the shader change underneath without a pixel moving. Everything below is
taking the bridge down, now that nothing needs to cross it.

## The end state

- The canvas's primitives (a box, text, an image, an icon) each draw with a look. The canvas chooses the look by what
  is being drawn, not by a number in the vertex.
- No kind constants, no `kind` in the vertex. The vertex carries geometry and the element's record, nothing that says
  how to paint.
- Decoration (shadow, border, glow, emboss) is only ever a painter in a look. No canvas method draws one as a quad of
  its own.
- `FieldStyle` is gone: an icon is a tint and a look, at every level.

## The work, in order

Each step leaves every test green, and each says whether pictures move. A step whose pictures move is checked by eye
against the chapter baselines, as the box pass was, and the baselines re-taken when it is accepted.

### 1. Repair `vex-suite-common` (do first)

**Found while writing this, and caused by the icon work**: `vex-suite-common` does not compile against the current
`vexelray-gui`. `MaskIcon.draw` hands a `FieldStyle` to `Node.field(texture, side, spread, ...)`, which now takes a
tint; `Choice`, `Tile` and `PictureViewer` build those `FieldStyle`s, with glow and shadow. The fix is the migration
`IconView` had: a tint, and a look of `fieldFill`, `glow`, `dropShadow` when there is a glow or a shadow. That look is
built in two places then, so it should be a helper in gui-core beside `ImageField`, which `IconView` uses too. Pictures:
none should move.

Every other sibling repo that depends on vexelray or supirvast was clean-compiled, tests included, against the current
jars on 2026-10-12: Pix, calculator-vexel-demo, dxgi-bridge, mainframe, pontif-framework, text-editor-vexel-demo,
vexelray-codec, vexelray-framework, vexelray-installer, the four vexelray-sim repos and vexplore all compile.
vexelray-designer does not, for reasons unrelated to shadelets (`Shell.place` is package-private, and an atchung
`subscribe` overload it calls is gone). Each step below repeats that sweep, since a framework API change reaches past
the three repos the work is in.

### 2. Take out the renderer's old path

`TreeRenderer.drawSelf` keeps an `else` branch for a canvas with no look table, drawing shadow, fill and border as
separate kind quads. No production path takes it: the window, its offscreen capture and the headless capture all set
the table before drawing. Only gui-core's unit tests take it, because they draw into a bare `Canvas`, so they test the
path production does not use:

- `TransformTest` asserts a `KIND_SHADOW` quad;
- `FocusRingTest`, `ImageNodeTest`, `CullTest`, `OverlayNodeTest`, `PictureNodeTest` draw with no table.

Give the tests a canvas with the core look table (a small helper), assert on looks and records rather than kinds,
and delete the branch. `Canvas.drawsLooks()` then has no caller and goes. Pictures: none.

### 3. Cut `FieldStyle` down, then out

The GUI only ever passes `FieldStyle.plain(tint)`, so the glow and shadow the field kind writes into the vertex are
always zero, and `fieldFill` still carries the old inline glow and shadow arithmetic that reads them.

- `fieldFill` becomes the field's fill alone; glow and shadow are only the `glow` and `dropShadow` painters.
- `Canvas.field` takes a tint (and, for effects, a look) instead of a `FieldStyle`; the vertex's filter slot carries
  only the field's range and V per unit.
- `FieldStyle` is deleted. Its callers: the renderer (one line), `CanvasFieldTest`, `EffectsRenderTest`, and
  `vex-suite-common` (gone after step 1).

Pictures: none (the GUI's glows already come from the painters).

### 4. Move the GUI's remaining decoration onto looks

Still drawn as kind quads, from production code:

| Where | Draws | As |
| --- | --- | --- |
| `TreeRenderer` scrollbar grip (~line 525) | shadow, lit fill | 2 quads |
| `TreeRenderer` plate (~line 220) | shadow, fill, edge | 3 quads |
| `TreeRenderer` focus ring (~line 439) | a ring around the focused node | stroke quad |
| `CanvasSink.java:62` (gui-draw) | a picture's outlined box | stroke quad |
| `PanelDemo` (vexelray, test) | outlined boxes | stroke quads |

Each becomes one element with the node painters (`nodeStroke`, `nodeFill`, `nodeLit`, `nodeShadow`), as a node's box
already is. Pictures: **move**, by the same rounding the box pass did (1 or 2 levels of 255 along edges, more where a
clip cuts a shadowed shape). An eye check against the baselines.

### 5. Retire the decoration methods and kinds

Once step 4 leaves them without callers: `Canvas.shadowRoundRect`, `strokeRoundRect`, `litRoundRect`, and the kinds
behind them, `KIND_SHADOW`, `KIND_STROKE`, `KIND_LIT`, with their core painters `SHADOW`, `STROKE`, `EMBOSS` and their
looks in `CoreLooks`. A canvas that wants a shadowed box draws `box(..., look)` with the node painters.

`Canvas.push` has kind-specific rules for the canvas's own scale (it scales the `uv` parameters of exactly those three
kinds). They go with the kinds. Pictures: none.

### 6. Take `kind` out of the vertex

What is left are the primitives: box, glyph, image, field. Each canvas method knows which it draws, so it can name the
primitive's look record directly instead of writing a kind and having `CoreLooks.record(kind)` translate it:

- `CoreLooks` keeps a record per **primitive** (box, text, image, field), named, not indexed by kind.
- `CanvasVertex` drops `LOC_KIND` / `OFF_KIND` and the `KIND_*` constants: **28 floats to 27**, and every location after
  it moves down by one. `CanvasShader.Attributes`/`Varyings` lose `kind`.
- `imageKind`, the image/field switch inside `Canvas`, becomes the choice of record.
- The `uv` slot stops carrying effect parameters (no kind reads them after step 5); it is the atlas or image UV only.
- Tests that read the kind back from vertex arrays (`ImageNodeTest`, `FocusRingTest`, `CanvasFaceRunTest`,
  `CanvasFieldTest`, `CanvasImageRunTest`, `CompositionTest`) assert the element's record instead.

Pictures: none. This changes the vertex layout, so every pipeline that declares it (all of them read
`CanvasVertex.ATTRIBUTES`, so they follow) and every native-image config naming the shaders is rebuilt.

## What stays, on purpose

- **The primitives' painters**: `FILL` (box), `GLYPH`, `IMAGE_FILL`, `FIELD_FILL` and the box, MSDF and field
  sources. They are how the canvas draws its own primitives, and any application using `Canvas` directly relies on them.
- **The node painters and the effects** (`vexelray-shadelets`): the new system itself.

## Not legacy, but on the same road

- **Look lengths do not follow a pushed scale.** The old field kind scaled its glow and shadow with the canvas's scale;
  a look's lengths are fixed in the table. Step 5 removes the last kind-specific scale rules, so the fix belongs with
  it: the canvas's scale in the element's record, or lengths per instance.
- **Text has no look yet** beyond the glyph primitive: an outline or glow on text waits for an MTSDF atlas (the MSDF's
  distance holds a pixel or two from the edge) and for the text source to look elsewhere.

## Done

All six steps landed on 2026-10-12, in vexelray, vexelray-gui and vex-suite-common, with every test green in each.

1. `vex-suite-common` compiles again. The look is built in one place, `ImageField.effects(glow, shadow)` (and the static
   `ImageField.effects(modifiers, front, glow, shadow)` that `IconView` builds its outline and weight on), beside
   `ImageField` in gui-core. `MaskIcon.draw` takes a tint, a glow and a shadow.
2. `TreeRenderer` has no separate-quad path, and `Canvas.drawsLooks()` is gone. The renderer's tests draw into
   `Canvases.core(w, h)`, a canvas with the core look table, and read a box's colours and lengths from its element's
   record (`Canvases.instance`) instead of from a kind.
3. `FieldStyle` is deleted. `fieldFill` is the field's fill alone (with no glow and no shadow it was already that,
   term for term), `Canvas.field` takes a tint, and the filter slot carries `(range, v per unit, 0, 0)`.
4. The plate, the focus ring, a scrollbar's grip and `CanvasSink.outline` are each one element of node painters
   (`PanelDemo` draws its outlines the same way). The plate mixes the layers it stands on into one fill colour in Java
   rather than blending them one over another. The grip's shadow is the node shadow's: dropped half its blur, so
   1.25 px where it was 1. Against the chapter baselines, only the scrollbar grips moved, by at most 2 levels of 255
   along their edges and in their shadows; the plate and the ring are in no chapter. The baselines are re-taken.
5. `shadowRoundRect`, `strokeRoundRect`, `litRoundRect`, the `shadow`, `stroke` and `emboss` painters and their kinds
   are gone, and with them `Canvas.push`'s rule scaling their `uv` parameters. The core set is thirteen shadelets.
6. The vertex is 27 floats: `LOC_KIND`, `OFF_KIND` and the `KIND_*` constants are gone, and every location after
   `uv` moved down by one. `CoreLooks` names a record per `Primitive` (box, text, image, field); each canvas method
   names its primitive's record, and an image or a field picks between two. The `uv` slot is an atlas, image or field
   coordinate only. The native-image metadata names the shaders by glob, so it needed no change; the composed shader's
   name moved with its manifest.

Every sibling repo that depends on vexelray or supirvast was clean-built, tests included, against the new jars:
vexelray-codec, vexelray-framework, vex-suite-common, Pix, calculator-vexel-demo, dxgi-bridge, mainframe,
pontif-framework, text-editor-vexel-demo, vexelray-installer, the four vexelray-sim repos and vexplore all pass.
vexelray-designer still does not compile, for the same reasons as before (`Shell.place`, the atchung `subscribe`).

What was named above as on the same road is still open: a look's lengths do not follow a pushed scale (now true of a
node's border width and shadow lift too, as of the box pass), and text has no look beyond the glyph.
