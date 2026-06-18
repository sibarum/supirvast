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
3D model (.obj) ──────────────────────┘
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
- No textures/samplers, no UBOs beyond an MVP push constant, no built-in lighting model (the fragment shader
  does whatever the author wrote).
- No glTF/FBX (OBJ only), no scene graph, no camera controls (auto-rotate only).
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

### 4. Model + graphics pipeline (rotating model on screen)
- [ ] Minimal OBJ loader: parse `v`/`f` (triangulate polygons), compute flat normals if absent → interleaved
      `position+normal` vertex buffer + index buffer (device-local or host-visible for the PoC).
- [ ] `D32_SFLOAT` depth attachment added to the render pass + framebuffers.
- [ ] Graphics pipeline from the `GraphicsPipelineSpec`: load both `.spv` as `VkShaderModule`s, build
      `VkPipelineVertexInputState` from the layout, viewport/scissor, back-face cull, depth test on, one
      color attachment.
- [ ] MVP `mat4` push constant (JOML `perspective` × `lookAt` × auto-rotation from `glfwGetTime`).
- [ ] Bind vertex/index buffers, `vkCmdDrawIndexed`. **Checkpoint: the model renders and rotates.**

### 5. Screenshot + acceptance
- [ ] Copy the presented image to a host-visible buffer; write PNG via `lwjgl-stb` `stbi_write_png`.
- [ ] `--screenshot <png>`: render one frame headlessly-ish (hidden/!visible window is fine), write, exit.
- [ ] Optional: keypress (e.g. `F12`) screenshot during interactive preview.
- [ ] Check in a sample vertex+fragment pair (authored in `core`, lowered to `.spv`) and a cube `.obj` so the
      end-to-end run is reproducible: `--vert … --frag … --model cube.obj --screenshot out.png` → non-blank PNG.

### Optional gate (nice-to-have)
- [ ] Before building the pipeline, run the loaded `.spv` through `NativeTools.validate()` and fail with a
      clear message if it's rejected — reuses the project's "validation as living requirements" ethos.

---

## Risks / things to watch
1. **Vertex attributes through the IR** (step 1) — the only spot an IR change might be needed.
2. **Swapchain + present-queue selection** is the bulk of the new Vulkan and is all-new vs. the headless
   compute path in `GpuContext`. Well-trodden, but budget for it.
3. **Interactive vs. one-shot screenshot** — plan both up front; same render code, different frame-loop exit.
