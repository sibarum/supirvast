# Shadelets: an ubershader each application composes

Status: **design, not built.** Drafted 2026-10-12, revised the same day after a review against the code. Nothing
here exists yet; what is left to settle is listed at the end.

## The problem

VexelRay draws every 2D primitive (boxes, text, images, icons) through one ubershader, `CanvasShader` in
vexelray-canvas, authored as a vastir AST and lowered to SPIR-V at build time. Its fragment stage is one statement,
`CanvasShader.coverage(in, fragColor, aa, pixelsPerUnit)`, which branches on the vertex's `kind`:

| Kind | Distance from | What it paints |
| --- | --- | --- |
| shape | the analytic rounded box | a flat fill |
| shadow | the same box | a soft squared falloff over `uv.x` px: an elevation shadow, or a glow when tinted |
| stroke | the same box | a ring `uv.x` px wide inside the edge |
| lit | the same box, twice | fill, embossed by the box's distance a step toward a fixed light |
| image | the same box | the box's coverage times a texel of the image in set 1, through `filtered()` (nearest, sharp bilinear, kernel taps, linear light to sRGB) |
| glyph | the MSDF text atlas (set 0) | coverage from the median, by a per-vertex `screenPxRange` |
| field | a distance field in the image slot (set 1) | fill, glow and shadow, inline, composited in the one fragment |

Every kind is then multiplied by the **clip**: a rounded-box coverage at the fragment's canvas position.

So an effect is written once per kind, if at all: shadow and stroke exist for the box only, as separate quads; glow
and shadow exist for fields only, inline; text has none. Adding one means changing the framework: a new parameter in
`FieldStyle`, a new vertex attribute (`loc8` is already full, shared by the image's filter and the field's
parameters), a new branch. Every application pays for every effect, used or not, and no application can add its own.

But every kind is a **signed distance to an edge** and a function of it. Even an image is a box's distance with a
texel for its colour, and every effect (fill, weight, outline, glow, shadow, emboss, reveal, progress, a morph) is a
function of the distance, sometimes the distance somewhere else. That is the seam.

### Two hosts, not one

`coverage()` has a second caller: `PanelShader` in vexelray-technique-panel, which hangs the same geometry on a plane
in the world. It reads the same vertex format and varyings (`CanvasShader.Attributes`, `Varyings`), places the vertex
differently, and passes the two scalars per fragment where the canvas passes 1 and 1:

- `aa`, the anti-aliasing half-width in canvas units;
- `pixelsPerUnit`, the screen pixels one canvas unit covers, which keeps an MSDF's coverage sharp at any distance.

Everything below is a fragment body both hosts wrap, as `coverage()` is today. A composition yields the canvas's
shader pair and the panel's, from one list of shadelets, or the two drift apart.

## The idea

A **shadelet** is a set of vastir contributions to a host ubershader, with one of a few fixed roles. Each comes with
a `*Style` record type: the parameters an element gives it. An application chooses the shadelets it uses; the
composed ubershader is lowered to SPIR-V **at build time** by `supirvast-maven-plugin`, as `CanvasShader` already
is. So the ubershader is trimmed to what the application draws, and an application can add effects of its own
without the framework knowing them.

## Roles

```
source ──► modifier* ──► painter+ ──► composite ──► clip ──► fragment colour
```

All of a shadelet's role functions are **pure**: everything they read arrives as a parameter. A shadelet never reads
an interface variable or a builtin itself; the host reads those and passes them in. That is what lets a composition
place one shadelet in two hosts, and what lets a pure shadelet run on the CPU (see Testing).

**Sources** produce a distance: `(source inputs, aa, pixelsPerUnit) -> vec2(distance, reach)`, positive inside, in
**canvas units**, and `reach` the farthest from the edge the distance can be trusted (the analytic box: unbounded;
MSDF text: about half its pixel range; a field: its spread). A canvas unit is a screen pixel only when `aa` is 1; the
panel host is why the units are canvas units and the scalars are inputs. The sources are `BoxSource`, `MsdfSource` and
`FieldSource`.

A **morph** is a source over two shapes **of the same source**, `mix(a, b, t)`: one icon into another, one box into
another. Never across sources, and never text: a run of text is many shapes, not one, so there is no single distance
to morph. Within one source the two distances share units and reach, so the mix is a true shape-to-shape morph and
costs one more evaluation:

- Field to field: the second shape is a second cell of the **same texture**. The icon set is one atlas, so an icon
  morph is a second UV into what is already bound. Its cell offset and `t` are per-element fields of the morph's
  style. Two fields in different textures do not morph.
- Box to box: the second box's half-size and radii, evaluated in closed form.

**Modifiers** reshape the distance: `(vec2 d, style) -> vec2 d`. Weight (`d + w`: bolder or lighter), round, offset,
inset. A modifier that moves the edge outward shrinks `reach` by as much.

**Painters** turn distance into a layer: `(vec2 d, style, ...) -> vec4`, **premultiplied**. Fill, image (the fill's
colour taken from a filtered texel instead of the style), outline (the band `|d| < w`), glow, shadow (the distance
sampled offset), emboss, reveal (the cut swept from outside to the edge), progress (a clip by position or angle).
Today's shape, stroke, shadow, lit and image kinds are all the box source with one of these painters.

A painter that needs the distance elsewhere (a shadow down and to the side, a bevel toward the light, a blur at a few
taps) calls **`distanceAt(vec2 offset) -> vec2`**: the element's source and modifiers, evaluated again with the
fragment moved by `offset` canvas units. Each source says how an offset moves its inputs (for the box, `local`; for
the MSDF and the field, the UV, scaled by the source's units per texel). The second evaluation costs a second sample,
or a second closed form for the box, which the painter that asks for it pays and nobody else does.

vastir has no function values (`Expr.Call` names a fixed `Function`, and SPIR-V has no function pointers), so
`distanceAt` is not handed to the painter: the composition **generates** it, as one function that switches on the
element's source and applies its modifiers. Every painter calls that one function. The alternative, a copy of each
painter specialised per source, grows as sources times painters; the dispatcher grows as their sum, and its switch is
on a value coherent across an element.

**A painter that offsets must fade by reach.** The distance past `reach` is meaningless: an MSDF holds a true distance
only a pixel or two out, a field only to its spread, and the quad ends there. A glow, a shadow offset or a blur that
reads past it shows the quad's edge as a square. Each such painter fades its layer to nothing before
`|d| = reach`, as the field kind's glow already does (`fade`), and that is part of its contract, checked by its test.

**The composite** turns the layers into one colour. Painters run **front to back** (outline before fill before glow
before shadow), so the composite is the *under* operator: `acc += (1 - acc.a) * layer`, premultiplied. It ends by
dividing out the alpha, so the fragment's output stays **straight alpha** and the pipeline's blend state is unchanged,
as the field kind already does with its `lit` share. Moving the frame to premultiplied blending is a later and
separate change. An application can replace the composite.

**The clip** is the host's, not a shadelet's. It runs after the composite, in every composition, and no contribution
can remove or replace it.

## A shadelet is more than a function

The roles are the common case, not the whole of it. A shadelet is a set of AST **contributions**, each to a named
**injection point** in the host, so it can add whatever its effect needs wherever the shader needs it:

| Injection point | What goes there | For example |
| --- | --- | --- |
| declarations | textures, storage buffers, push-constant members, helper functions | the MSDF atlas a text source samples; a noise function a painter calls |
| reach | how far past its geometry the element's effects draw, as a Java function of its style | a glow's radius; a shadow's offset plus blur |
| vertex: outputs | values handed to the fragment stage | a morph's `t`, read once per vertex instead of per fragment |
| fragment: source / modifier / painter | the roles above | a box distance; a weight; an outline |
| fragment: composite | how the painted layers become the fragment's colour | the default front-to-back under; an application could replace it |
| fragment: epilogue | anything after the colour is decided, before the clip | a dither, a debug overlay that colours by kind |

The quad's growth is the host's, not the contribution's, and it happens **on the CPU**. A contribution says how far
its effect reaches, as a function of its style; `Canvas` knows each element's look when it emits the element, takes
the largest reach over it and emits the quad that much larger, with `local` and the UV extended to match, as it
already pads every quad by a pixel. So neither vertex stage reads a buffer or changes, and the panel, which places
the same vertices in the world, gets the growth for nothing.

The host defines the points and the order contributions at one point run in (declared order, then dependency order:
a painter that calls a helper function needs that function's declaration first). A contribution is checked when the
ubershader is composed: an injection point a contribution names must exist, and two contributions that declare one
location or binding must declare the same thing there. vastir's lowering refuses the second: two different interface
variables at one location, textures at one set and binding, or buffers at one binding, each named. A mismatch is a
build error in the application's shader build, which is the right time to find out.

The contract, the roles and the composition builder live in vexelray-canvas, in `dev.vexelray.canvas.shadelet`:
they are about a canvas's distances, which vastir knows nothing of, and the canvas is their only host, so a module of
their own would be ceremony. vastir supplies what any composed shader needs of it: the agreement checks above, flat
varyings, and buffers in a set other than 0. Shadelets are written in `vexelray-ir`'s vocabulary.

The roles are Java methods that build vastir expressions (a source's `distance`, a modifier's `modify`) or statements
(a painter's `paint`, into a layer it is handed), not vastir functions: the composition inlines each into the one
branch that runs it, so a shadelet's code is in the shader once however many looks name it. The one generated vastir
function is `distanceAt`.

This is also what keeps the core set small: a shadelet that needs a texture brings the texture with it, rather than
the host declaring every texture any shadelet might want.

## Looks and styles: how parameters reach the GPU

The constraint is the vertex. Every per-element parameter has to reach the fragment shader, and putting them in the
vertex grows it for everything (27 floats before shadelets; an application with five effects could double it).

### What stays in the vertex

Geometry is per element and per vertex, and stays where it is: the position, `local`, the box's half-size and radii,
a glyph's UV and `screenPxRange`, a field's range, the colour, the clip. Mixed text sizes and colours in one draw
depend on that. What leaves the vertex is what an effect adds; what joins it is **one number, the element's index**.
The canvas draws its quads from one vertex buffer of floats, not as instances, so the index is a float attribute the
canvas writes (exact to 2^24 elements a frame), not `gl_InstanceIndex`. The vertex stage converts it to an integer and
hands it to the fragment stage with the `Flat` decoration, so it arrives exact, not interpolated. The vertex is 28
floats, however many effects an application has and however many of their values animate.

### A look is several styles

A button is a fill, an outline and a glow: three shadelets, three `*Style` records. So an element does not have a
style; it has a **look**, an ordered list of what draws it:

- its source, and the source's style;
- its modifiers, in order, each with its style;
- its painters, front to back, each with its style.

A state change (hover, pressed, on) is a different look, not new data. Looks are few and shared: a theme defines tens
of them, not thousands. A look has at most **eight** painters; more is refused when the look table is built. Eight is
plenty for stacking stock effects: an element that needs more is a custom effect, written as one painter of the
application's own.

### Three tables, one buffer

The way ShaderLab splits material properties from per-instance ones, parameters live in tables, read-only in the
fragment stage (vastir already decorates them `NonWritable`). All three are in **one float buffer** at **set 2**,
binding 0 (`Composition.TABLES`), since set 0 is the text atlas (rebound per face) and set 1 the image (rebound per
run), and vexelray's `StorageBuffer` is one buffer with a set of its own. Indices in it are whole numbers held as
floats, exact to 2^24. The layout is in `Composition`'s javadoc:

- **The style table**: a flat array of floats. A `*Style` is a record annotated `@Style`. A processor (the
  elektroq pattern from atchung: annotate a record, a processor emits code; nothing reflects) generates its
  **layout** (which floats, in what order) and the **vastir reads** its shadelet uses, each taking the style's base
  offset into the table. The table is uploaded when the theme changes.
- **The look table**: per look, its source, modifiers and painters as (shadelet id, style offset, instance offset)
  triples. The generated `distanceAt` and the painter loop walk it. Uploaded with the style table.
- **The element records**: one per drawn element, at the offset its vertices carry: its look, and its fields marked
  `@PerInstance`: a morph's `t`, a reveal's progress, a progress fill's amount, a colour animating between two
  themes. Written when the element is emitted, which is when its animated value is known. An element with no such
  fields writes only its look, and every such element of a look can share one record: the canvas's kinds each have
  one, written with the table (`CoreLooks`), so the core set's buffer is static. Records with per-instance values are
  rewritten every frame, so they move to a buffer of their own, one per frame in flight, as the vertex buffer is
  (vexelray's `StorageBuffer` is one mapped buffer, and says the caller must wait for the draw before writing it).
  That buffer arrives with the first effect that has a per-instance field.

A shadelet reads its style through the generated accessors and does not know which buffer a field came from: whether
a field is shared or per element is a property of the style record, and changing it is an annotation, not a shader
edit.

## Composing and compiling

An application composes its ubershader the way it composes anything else: a `ShaderSource` in its own sources,
naming the shadelets it uses. `supirvast-maven-plugin` already scans a project's classes for every `ShaderSource` and
lowers each to SPIR-V into the jar, so the application's build compiles its own ubershader, with no framework
release. One composition yields four stages: the canvas's vertex and fragment and the panel's.

The **manifest** is what the shader was composed of: each shadelet's name, role, id, and the hash of the style
layout it reads. It is computed from the `Composition`, which is Java on the application's classpath, so nothing has
to ship beside the SPIR-V to say it; it also has a text form, for tools.

The framework ships a default composition, the **core set**, which every application keeps whether or not it
composes its own: the box, MSDF and field sources, and the fill and image painters (the image painter brings
`filtered()` with it). Everything a framework widget must draw to be usable is in it. Effects (outline, glow, shadow,
emboss, weight, morph) are on top.

*Open*: who owns the pipeline and vertex layout. Today `GuiApp.canvasConfig`, the canvas technique, the panel
technique and headless capture all assume the framework's one shader. They take the application's composed shader
instead, and capture uses the same one, or a capture draws differently from the window. The composed shader cannot
take the framework's names (`canvas.vert`, `canvas.frag`), which `Shaders.loadOrLower` would resolve to whichever jar
comes first.

## Unknown styles: drawn without, said once, never a crash

A framework widget defines the looks for its element (a button's hover glow, a card's lift) whether or not the
application compiled the shadelets that draw them. An application that trimmed glow still uses the framework's
button.

So a style whose shadelet is not in the application's ubershader **does nothing**, and says so once:

- Resolved when the look table is built (at theme install or change), never per frame: each `*Style` type maps to
  its shadelet's id in the manifest, and a look entry whose type has none is left out of the look. The check is off
  the frame path, where the framework does not log.
- Warned once per type through `Diagnostics.dropped`, naming the style type, the application's shader and the
  shadelet that would draw it, so "why does my glow not show" has a one-line answer.
- The element still draws: only the effect is dropped. The fill is in the core set, so a button missing its glow is
  still a button.

A **stale** shader is a different failure. An application's SPIR-V, built against an older framework, reads the
framework's shadelets' parameters at their old offsets, while the tables are written with the new: nonsense, with no
error. So the composed fragment shader's name carries the manifest's hash (`canvas.frag.<hash>`): a jar whose SPIR-V
was built against other layouts does not have the name looked for, and `Shaders.loadOrLower` lowers the composition
in-process at startup instead, which is correct, and slower, and said once through `Diagnostics.dropped` with the fix
(a rebuild).

## Testing

Each shadelet is tested on the GPU against a reference render. Where it can, it is also run on the CPU through
vast's Truffle backend, which must agree with the GPU: the same agreement check SupirVast's own tests make.

The CPU side is limited to **pure shadelets that do not sample**. `CoreToTruffle` lowers no graphics stage (it throws
on `InterfaceRead` and `InterfaceWrite`) and samples no texture (it throws on `SampleTexture`). The roles' rule that
everything arrives as a parameter is what puts the box source, weight, outline, fill, glow and the composite within
reach of it: a test builds a compute kernel around the role, its `Fragment` and `StyleView` reading set-0 buffers
the test fills (a `StyleView` takes the buffer it reads), and runs it on both backends. The MSDF and field sources, the image painter and anything that
calls `distanceAt` on them are GPU-tested only, until vast can sample a texture with the GPU sampler's filtering,
which is its own piece of work and not on this path.

## What it replaces

- `FieldStyle` (tint, glow, shadow): its three fields become the fill, glow and shadow styles.
- The shape kind's separate shadow and stroke quads: inline painters, the quad grown by the effect's reach, as a
  field already spills into its margin. One quad where there were three, but it touches every widget that draws an
  elevation shadow, so it lands behind a visual pass.
- The lit kind: the emboss painter over the box source.
- The image filter and the field's parameters sharing `loc8`: the image painter's style, and the field source's.
- The text atlas becomes MTSDF (a true distance in the alpha channel) with a wider range, or text outlines and glows
  clip at a pixel or two: plain MSDF with a range of 4 is only accurate near the edge.

## The first slice

0. ~~vastir: the `Flat` decoration, an integer varying, a buffer in a set other than 0, the agreement checks, and a
   texture sampled from a called function.~~ Done: `FlatVaryingTest`, `ResourceConflictTest`, and the supir text
   form (`@flat`, `@set`).
1. ~~The shadelet contract: the role signatures, the generated `distanceAt` dispatcher, the under composite and the
   host's clip, a composition builder that refuses two shadelets of one name.~~ Done: `dev.vexelray.canvas.shadelet`,
   `CompositionTest` (spirv-val on the core shader and on one with a styled modifier and painter).
2. ~~`@Style` and `@PerInstance`, and their processor; the tables, the manifest with its layout hashes, and the
   vertex's one element index.~~ Done: `vexelray-shadelet-codegen` (generates `<Style>Layout`: the `StyleLayout`, and
   the writers for the style table and an element's record), `LookTable`, `Manifest`, `CanvasVertex.LOC_ELEMENT`.
3. ~~Done~~: `CoreShadelets`, `CoreLooks`; every host binds the tables at set 2 (`RunRecorder`, the presenters,
   `OffscreenDraw`, both techniques, `GuiApp`/`GuiWindow` through `DeviceImages`); `CaptureDiff` and
   `capture-chapters.ps1` in vexelray-gui. 15 of 16 demo chapters pixel-identical at tolerance 0; Plot differs only by
   its plot's layout height, from a concurrent change to capture's settling, and its baseline is to be re-taken. The
   panel scene is not yet diffed. The composed fragment shader is named by its manifest's hash (`CanvasShader.Composed`).
   As planned:
   The existing kinds as sources and painters: box, MSDF and field sources; fill, image (with `filtered()`), stroke,
   shadow and emboss painters; `CanvasShader` and `PanelShader` re-expressed as the core composition, drawing exactly
   what they draw today. The check is a capture diff of the GUI demo, chapter by chapter (`Demo --capture out.png N`
   for its 16 chapters, before and after), and of a scene with a panel in the world. Nothing in the repositories
   diffs images yet, so the slice brings a small tool that does: per pixel, with a tolerance, writing a difference
   image when it fails.
4. Weight, outline and morph (field to field, box to box) as the first new effects, each with a GPU test, and a
   CPU/GPU agreement test where it is pure.
5. Unknown styles dropped and warned once; a stale shader refused at load.

Then the icon API (`IconView`, `IconButton`, a theme's icon looks) moves onto looks, and only after that do icons
roll out into the applications.

## Still to settle

Technical work items, not decisions waiting on anyone: each is settled by building or measuring it.

- Which code picks the shader the GUI draws with (the *open* paragraph under Composing and compiling): the window
  setup, the canvas, the panel and capture all load the framework's shader today, and must all take the
  application's, the same one.
- ~~Can a non-entry `Function` sample a texture?~~ It should: a `Texture` is module-scoped (name, set, binding,
  kind), `CoreToSpirv.collectInterface` scans every function's body for the textures it samples, and
  `lowerSampleTexture` loads the global variable wherever the sample is. Not yet proven on the GPU:
  `TextureShaderTest` samples only from an entry point, so the first slice starts with a test that does. The CPU side
  is settled: Truffle does not sample (see Testing).
- The cost of deciding per element which effects draw: the painter loop and the `distanceAt` switch branch the same
  way for every pixel of an element, which GPUs handle cheaply. Measured once on an integrated GPU before relying on
  it.
