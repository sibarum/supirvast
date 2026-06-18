# vastir-preview — SupirVast Shader Previewer

A small **windowed Vulkan application** that loads a vertex+fragment SPIR-V pair (exported from SupirVast)
and a 3D model from the command line, renders the model with those shaders on live hardware, and can dump a
PNG screenshot.

This is the first SupirVast path that **actually rasterizes**. Everywhere else, vertex/fragment shaders are
only proven `spirv-val`-clean and cross-compiled to GLSL/HLSL/MSL (`VertexFragmentShaderTest`); they are never
drawn. The compute path *does* run on the GPU (the CPU==GPU differential), but graphics never has. This module
closes that gap with something visual you can screenshot.

```
vastir-preview --vert myshader.vert.spv --frag myshader.frag.spv --model cube.obj
vastir-preview --vert v.spv --frag f.spv --model cube.obj --screenshot out.png   # one frame, write PNG, exit
```

## Where this fits

```
core IR ──CoreToSpirv──▶ .spv files ──┐
                                       ├──▶ vastir-preview ──▶ window / screenshot
3D model (.obj/.ply) ──────────────────┘
```

The previewer consumes **files**, not in-process objects, matching the workflow "test a shader I've written on
live hardware." Wiring `core` lowering directly into the previewer (skip the files) is a trivial later step.

## The vertex contract (v1)

The PoC fixes one vertex layout — encoded as a typed `GraphicsPipelineSpec` (in `vastir-tools`, alongside
`KernelSpec`) so it is self-describing rather than a magic convention:

| location | attribute | type   |
|----------|-----------|--------|
| 0        | position  | `vec3` |
| 1        | normal    | `vec3` |

A SupirVast author writes their vertex shader's `InterfaceVar.input(name, location, type)` against this layout;
the previewer builds the `VkPipelineVertexInputState` from the same spec. No serialization format yet — the CLI
passes raw `.spv` and the spec *is* the v1 contract.

## Non-goals (the "expand from there")

- No headless render differential / CPU rasterizer reference (out of scope by design).
- No ShaderLab DSL, materials, properties, multi-pass, or render-state authoring — `GraphicsPipelineSpec`
  stays a minimal typed record.
- No runtime MVP transform / camera / auto-rotation yet (needs `Matrix` + push constants in the `core` IR;
  see step 4). v1 renders statically with the model pre-placed in clip space.
- No textures/samplers, no UBOs, no built-in lighting model (the fragment shader does whatever the author
  wrote).
- Model formats: OBJ and PLY (ascii + binary, little/big endian). PLY's full named-channel model is parsed
  and exposed, but only position+normal are drawn (colors/UVs/custom channels are read, not yet rendered). No
  glTF/FBX, no scene graph.
- Linux/macOS (Windows-only natives, matching `vastir-tools`).

## Build / run

LWJGL core + `lwjgl-vulkan` come transitively from `vastir-tools`; this module adds `lwjgl-glfw`, `lwjgl-stb`
(+ their `natives-windows`), and pure-Java `joml`. Vulkan itself needs no natives jar — LWJGL dlopens the
system `vulkan-1` loader. Requires a Vulkan 1.3 device (to consume SPIR-V 1.6), as the compute path does.

---

## Task list

Ordered by dependency. Each step should leave the module in a runnable/verifiable state.

### 0. Module skeleton — ✅ done
- [x] Add `vastir-preview` to the parent reactor (`<modules>`).
- [x] Module `pom.xml`: depend on `vastir-tools` + `lwjgl-glfw`/`lwjgl-stb` (+ `natives-windows`) + `joml`.
- [x] This README with the task list.

### 1. Confirm the IR can express vertex attributes (de-risk first) — ✅ done
> `TODO.md` listed "vertex attribute inputs" as still-TODO under P2; this was the one place an IR fix could
> surface. Confirmed expressible with **no IR change** — `CoreToSpirv.InterfaceResources` already picks
> storage class from `InterfaceVar.Direction` (not the stage), so a vertex-stage `INPUT` lowers to a
> `Location`-decorated `Input` (a vertex attribute). Pinned by `VertexAttributeShaderTest` (vastir-tools).
- [x] `VERTEX` entry reads `InterfaceVar.input("position", 0, vec3)` + `input("normal", 1, vec3)`, writes
      `gl_Position`, passes a varying out.
- [x] `spirv-val` accepts it; disassembly shows two `Location`-decorated `Input` `vec3` variables +
      `BuiltIn Position`; GLSL shows `layout(location = 0/1) in` vertex inputs.

### 2. `GraphicsPipelineSpec` (in `vastir-tools`) — ✅ done
> The typed shader↔pipeline contract: two SPIR-V blobs + entry points + the vertex attribute layout
> (`VertexAttribute` = name/location/`Type`/offset). **Attachment formats are deliberately not in the spec** —
> color format comes from the swapchain surface at runtime, depth is the previewer's choice; neither is
> intrinsic to the shaders. Validated by `GraphicsPipelineSpecTest` (pure Java, 5 cases).
- [x] Typed record + `VertexAttribute`; fail-closed validation (word-aligned SPIR-V, contiguous
      locations 0..n-1, non-overlapping offsets, f32 scalar/vector attribute types).
- [x] `standard(vert, frag)` factory for the v1 layout (position@0 vec3, normal@1 vec3; stride 24); a
      `vertexStrideBytes()` helper for the interleaved vertex buffer.

### 3. Window + swapchain (prove a window clears to a color) — ✅ done
> Verified on the RTX 2070: `PreviewApp` renders 120 clear-color frames and exits cleanly, every `VkResult`
> checked. CLI parsing is in `PreviewOptions` (unit-tested, GPU-free). `--frames <n>` caps the loop for
> non-interactive runs; `--screenshot` is accepted but deferred to step 5 (logs a notice).
- [x] CLI parsing: `--vert`, `--frag`, `--model`, `--width`/`--height`, `--screenshot <png>`, `--frames <n>`
      (`PreviewOptions` + `PreviewOptionsTest`, 5 cases).
- [x] GLFW window (no OpenGL context: `GLFW_CLIENT_API = GLFW_NO_API`).
- [x] Vulkan instance with GLFW-required surface extensions; window surface; graphics+present queue family.
- [x] Device with `VK_KHR_swapchain`; swapchain (images created `COLOR_ATTACHMENT | TRANSFER_SRC` for
      screenshots); image views; a render pass (one color attachment) + per-image framebuffers.
- [x] Render loop (acquire/submit/present with semaphores + fence) that clears to cornflower blue and presents.
      **Checkpoint met: a colored window appears and presents.**

### 4. Model + graphics pipeline (model on screen) — ✅ done (static; MVP deferred)
> Verified on the RTX 2070: drew the sample cube (36 indices) for 120 frames, every `VkResult` clean — a
> SupirVast-authored vertex+fragment pair lowered to SPIR-V now rasterizes a depth-tested 3D model.
> `SampleAssets` generates the reproducible inputs (`cube.obj` + `model.vert/frag.spv`, authored in `core`).
- [x] Minimal OBJ loader (`ObjMesh` + `ObjMeshTest`): `v`/`vn`/`f`, fan-triangulation, flat normals when
      absent → interleaved `position+normal` (host-visible) vertex buffer + `uint32` index buffer.
- [x] `D32_SFLOAT` depth attachment added to the render pass (+ subpass dependency) + framebuffers.
- [x] Graphics pipeline from the `GraphicsPipelineSpec`: both `.spv` as shader modules,
      `VkPipelineVertexInputState` from the layout (`Type`→`VkFormat`), viewport/scissor, depth test on, one
      color attachment. (Cull = NONE: an arbitrary OBJ's winding is unknown; depth still resolves overlap.)
- [x] Bind vertex/index buffers, per-frame command recording, `vkCmdDrawIndexed`. **Checkpoint met.**
- [ ] **Deferred — runtime MVP transform + auto-rotation.** This needs a `Matrix` type, matrix×vector, and
      push-constant read in the `core` IR (none exist yet) — a separate IR chunk, not pure Vulkan. v1 renders
      statically: the vertex shader emits `gl_Position` from the model position, and `SampleAssets` generates
      the cube **pre-rotated in clip space** so several faces show. JOML is already a dependency, ready for it.

### 5. Screenshot + acceptance — ✅ done
> Verified on the RTX 2070: `--screenshot` produced a 512×512 PNG of the depth-tested cube, three faces in
> distinct normal-based colors. End-to-end proven: `.ply` model + `core`-authored shaders → SPIR-V → pipeline
> → rendered → PNG.
- [x] Copy the rendered image to a host-visible buffer (`vkCmdCopyImageToBuffer` + layout barrier); write PNG
      via `lwjgl-stb` `stbi_write_png`, swizzling BGRA→RGBA.
- [x] `--screenshot <png>`: render one frame off-screen (window hinted `GLFW_VISIBLE=false`), write, exit.
- [x] Reproducible from the committed `SampleAssets` generator — see **Reproduce** below. The generator emits
      `cube.obj`, a colored `cube.ply`, and the `core`-authored `model.vert/frag.spv`.
- [ ] Deferred (optional): keypress (e.g. `F12`) screenshot during interactive preview.

### Optional gate (nice-to-have)
- [ ] Before building the pipeline, run the loaded `.spv` through `NativeTools.validate()` and fail with a
      clear message if it's rejected — reuses the project's "validation as living requirements" ethos.

## Reproduce

From the repo root (assets are generated, not committed — the generator is the source of truth, à la the
codegen convention). Output lands under `target/samples/` (git-ignored):

```sh
# 1. generate cube.obj, cube.ply, and the core-authored vertex/fragment SPIR-V
mvn -pl vastir-preview exec:java -Dexec.mainClass=dev.supirvast.vastir.preview.SampleAssets \
    -Dexec.args="target/samples"

# 2a. interactive window (Esc/close to quit; --frames N to auto-exit)
mvn -pl vastir-preview exec:java -Dexec.mainClass=dev.supirvast.vastir.preview.PreviewApp \
    -Dexec.args="--vert target/samples/model.vert.spv --frag target/samples/model.frag.spv \
                 --model target/samples/cube.ply"

# 2b. one-shot PNG screenshot (off-screen)
mvn -pl vastir-preview exec:java -Dexec.mainClass=dev.supirvast.vastir.preview.PreviewApp \
    -Dexec.args="--vert target/samples/model.vert.spv --frag target/samples/model.frag.spv \
                 --model target/samples/cube.ply --screenshot target/samples/cube.png"
```

### Optional gate (nice-to-have)
- [ ] Before building the pipeline, run the loaded `.spv` through `NativeTools.validate()` and fail with a
      clear message if it's rejected — reuses the project's "validation as living requirements" ethos.

---

## Risks / things to watch
1. **Vertex attributes through the IR** (step 1) — the only spot an IR change might be needed.
2. **Swapchain + present-queue selection** is the bulk of the new Vulkan and is all-new vs. the headless
   compute path in `GpuContext`. Well-trodden, but budget for it.
3. **Interactive vs. one-shot screenshot** — plan both up front; same render code, different frame-loop exit.
