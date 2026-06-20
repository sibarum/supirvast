# Supir-Vast — TODO

A SPIR-V AST in Java/GraalVM: one `core` IR lowers to **validated SPIR-V** (GPU, via Vulkan + cross-compile)
and to an **executable Truffle AST** (CPU). The dual-target thesis is proven — identical results on CPU and
GPU — for a small language. This list is what turns that skeleton into something you'd write real kernels in.

Priority tiers below are ordered by leverage: **P0** unblocks real (data-parallel) programs and is the
highest-value next proof; **P1** deepens the language; **P2** broadens targets; **P3** is perf/infra.

---

## Done

- [x] Maven multi-module skeleton — `vastir-codegen`, `vastir`, `vast`, `vastir-tools` (JDK 25 / GraalVM, Truffle 25.0.1)
- [x] Grammar-driven codegen from pinned SPIRV-Headers — `Op`, `OperandKind`, 57 operand enums, `Spirv` constants
- [x] SPIR-V binary emitter (`SpirvModule`/`Instruction`) — little-endian words, well-formed header
- [x] `core` IR — shared `type` system, expressions, local vars, structured `if`/`while`, `Return`/`StoreResult`
- [x] `CoreToSpirv` — module-section lowering, type/const dedup, SSBO output; `spirv-val`-clean
- [x] `CoreToTruffle` — executable Truffle AST, runs on the CPU
- [x] Bundled native toolchain (`spirv-val`/`dis`/`as`/`opt`, `spirv-cross`) auto-fetched from pinned Vulkan SDK
- [x] `VulkanCompute` — headless LWJGL compute dispatch, runs our SPIR-V on the GPU
- [x] `DifferentialHarness` — same `core` body on CPU (Truffle) and GPU (Vulkan), **results asserted equal**

---

## P0 — Unblock real, data-parallel kernels ✅ (done 2026-06-17)

> Done: the framework now runs actual array workloads, and the differential covers real parallel data —
> `out[i] = a[i] + b[i]` authored once in `core`, identical on CPU (Truffle) and GPU (Vulkan).

- [x] Input buffers — storage buffers in `core` (`Buffer`) + `CoreToSpirv` (multiple bindings)
- [x] Array / runtime-array buffer types + dynamic indexing (`OpTypeRuntimeArray`, 2-index `OpAccessChain`)
- [x] `GlobalInvocationID` builtin exposed as `Expr.InvocationId` (uint loaded + bitcast to signed index)
- [x] N-element dispatch — `vkCmdDispatch(n,1,1)`, per-invocation indexing
- [x] `CoreToTruffle.lowerKernel` — dispatch modelled by calling the AST once per invocation over shared buffers
- [x] Differential over an array — `KernelDifferentialTest`: vector add, CPU == GPU == expected, element-wise

## P1 — Language & IR depth

- [x] Vector types + componentwise ops — `Expr.VectorConstruct`/`VectorExtract`, componentwise `Binary`
      (`OpCompositeConstruct`/`OpCompositeExtract`, vector `OpIAdd`/…); CPU `int[]`/`double[]`; CPU==GPU verified.
      *Still TODO: swizzles (multi-component), vector·scalar broadcast, `OpDot`, vector comparisons (bvec).*
- [x] Function calls + parameters (`OpFunctionCall`) on both backends; multi-function modules —
      `Expr.Param`/`Expr.Call`, `OpFunctionParameter`; CPU via `lowerModule` + lazy call targets; CPU==GPU verified.
      *Still TODO: calls from within a kernel entry; recursion is (correctly) unsupported by SPIR-V.*
- [x] More operators — div/mod, bitwise (and/or/xor), shifts, unary (negate/not/logical-not), logical and/or.
      CPU==GPU verified (`OperatorKernelTest`). *Logical and/or are non-short-circuit (match `OpLogicalAnd/Or`);
      true short-circuit would need control flow. Still TODO.*
- [x] Relax 32-bit-only — full numeric type system, every step CPU==GPU verified on real hardware:
      - *Unsigned end-to-end* — new `Expr.Bitcast` (same-width reinterpret, `OpBitcast`) lets a kernel read an
        i32 buffer element, drive unsigned `OpUDiv`/`OpUMod`/`OpShiftRightLogical`/`OpU{Less,Greater}Than`, then
        bitcast back to store; Truffle `BinaryNode` carries signedness
        (`Integer.divideUnsigned`/`remainderUnsigned`/`>>>`/`compareUnsigned`). `UnsignedKernelTest`.
      - *i64/u64* — `Type.int64()`/`uint64()`, two-word `OpConstant`, `OpCapability Int64` (emitted when a
        64-bit int type is declared). Truffle gained `Long` arithmetic (`scalarLong`, `Long.*Unsigned`).
        `Int64KernelTest`: 64-bit mul+signed-mod and unsigned divide, CPU==GPU==`long` reference.
      - *f32 fidelity* — the Truffle backend now carries f32 as Java `float` (not `double`), so each op rounds
        to 32-bit like the GPU; `f32`↔`i32` `Bitcast` is a real re-encode (`Float.intBitsToFloat`/
        `floatToRawIntBits`). `FloatKernelTest` cubes a float through i32 buffers, bit-exact CPU==GPU.
      - *Conversions* — new `Expr.Convert` (`OpSConvert`/`OpUConvert`/`OpConvert{S,U}To{F}`/`OpConvertF{To}{S,U}`/
        `OpFConvert`), opcode/extension keyed off the result type. `ConversionKernelTest`: `(int)((float)a*1.5f)`.
      - *f64* — `Type.float64()`, two-word float `OpConstant`, `OpCapability Float64`, `Bitcast` f64↔i64.
        `Float64KernelTest`: a double computed and observed bit-exactly via an i64 split through i32 buffers.
      - *Narrow i8/i16* — `Type.int8()`/`int16()`, `OpCapability Int8`/`Int16`, narrow `OpConstant`; Truffle
        `narrowInt` re-truncates each result to its declared width so wraparound matches. `NarrowIntKernelTest`:
        `(i8)a*3` and `(i16)a+30000`, both wrapping, CPU==GPU==`byte`/`short` reference.

      *Buffers remain i32-only — wider/float values live inside kernels and cross the boundary via
      `Convert`/`Bitcast`. Device features (shaderInt8/16/64, shaderFloat64) aren't explicitly enabled at
      device creation yet; the RTX 2070 runs them anyway (no validation layer). Float differentials use only
      correctly-rounded ops (+,−,×); div/FMA carry looser GPU precision.*
- [~] SSA values via `OpPhi` across `if`/`while` — **deferred (optimization, not a feature gap).** Memory-based
      locals (`OpVariable`/`OpLoad`/`OpStore`) are a complete, correct, `spirv-val`-clean lowering for every
      construct; `OpPhi` would only cut load/store traffic. Escape hatch proven: `spirv-opt --ssa-rewrite`
      (mem2reg) promotes our locals to `OpPhi` automatically — verified valid, behavior-preserving on the GPU,
      and OpPhi-bearing in `SpirvOptPhiTest`. **Revisit only on a measured perf need**, and likely alongside the
      explicit SSA mid-level IR in P3 (build demand-driven, per load-bearing constraint #4).
- [x] Early `return` inside structured regions — `CoreToSpirv` tracks per-block termination so an early
      `OpReturn`/`OpReturnValue` in an `if`/`while` branch is valid: dead code after it is skipped, the would-be
      branch-to-merge is dropped, and a merge/continue block left unreachable gets `OpUnreachable`. The
      invocation-id load was hoisted to the entry block (it dominates all blocks) to fix a dominance error when
      the first `gid` use sat inside a loop. CPU already unwinds via its return exception. `EarlyReturnKernelTest`
      (guard clause, early return from a loop, both-arms-return) — CPU==GPU.

## P2 — Targets & toolchain

- [x] Vertex/fragment stages + interface I/O (locations) — core `Builtin` (`gl_Position`/`gl_VertexIndex`) and
      `InterfaceVar` (location-bound Input/Output), with `Expr.BuiltinRead`/`InterfaceRead` and
      `Statement.BuiltinWrite`/`InterfaceWrite`. `CoreToSpirv` declares Input/Output `OpVariable`s with
      `BuiltIn`/`Location` decorations and lists them in the entry-point interface; vertex/fragment execution
      models already existed. CPU backend rejects these graphics-only constructs (no rasterizer). A vertex +
      fragment pair (`VertexFragmentShaderTest`) validates and cross-compiles to GLSL/HLSL/MSL. *Verification is
      `spirv-val` + cross-compile, not CPU==GPU — a fragment shader needs rasterization the headless compute
      path can't drive. Still TODO: vertex attribute inputs, multiple render targets, more builtins.*
- [x] Cross-compile coverage — `CrossCompileCoverageTest` cross-compiles a real data-parallel kernel to GLSL,
      HLSL, and MSL (was only trivial-GLSL before); HLSL now targets Shader Model 5.0 (`--shader-model 50`) so
      compute/UAVs and system-value semantics like `SV_VertexID` are supported.
- [x] `spirv-opt` pass integration — `NativeTools.optimize(spirv, passes…)` (default `-O`, or explicit passes
      like `--ssa-rewrite`). Proven by `SpirvOptPhiTest` (mem2reg → `OpPhi`, valid + behavior-preserving).
- [ ] Optional native textual backends (`CoreToGlsl`, …) if we ever want to bypass `spirv-cross`

## Graphics & PBR rendering — DONE (2026-06-18/19)

> A full graphics path: author shaders in `core`, render real 3D on the GPU, and a ShaderLab-style PBR
> authoring layer. Two new modules: `vastir-preview` (windowed Vulkan previewer) and `vastir-pbr` (PBR).

- [x] **`vastir-preview`** — windowed Vulkan shader previewer (GLFW + swapchain + depth): loads a vertex+
      fragment SPIR-V pair + a model from the CLI and renders it; `--screenshot` writes a PNG (`lwjgl-stb`).
      Verified by rendering on the RTX 2070. `GraphicsPipelineSpec` (vastir-tools) is the typed shader↔pipeline
      contract (vertex layout position@0 / normal@1 / uv@2).
- [x] **Vertex attributes** — vertex-stage `InterfaceVar.input` lowers to a `Location`-decorated `Input`
      (no IR change needed); models feed them via the vertex buffer.
- [x] **Model loaders** — OBJ (`v`/`vn`/`vt`/`f`) and PLY (general element/property model, ascii + binary
      little/big endian; named channels parsed/exposed) → one interleaved `Mesh`; loader dispatch by extension.
- [x] **Math intrinsics** — `Expr.MathCall` (`OpDot` + the full GLSL.std.450 float library: normalize, pow,
      sqrt, reflect, clamp, mix, trig, hyperbolic, exp/log, rounding, fma, distance, refract, …).
- [x] **Textures & samplers** — `core` `Texture` (2D + cube) + `Expr.SampleTexture` → `OpImageSample…`;
      previewer loads PNGs into images/samplers + a combined-image-sampler descriptor set (`--texture`,
      `--cubemap`); UV vertex attribute + UVs in the loaders.
- [x] **Matrices + push constants** — `Type.Matrix`/`mat4`, `PushConstants` block, `Expr.MatrixTimesVector`
      (`OpMatrixTimesVector`); previewer `--mvp` pushes a rotating model-view-projection (+ model + camera).
- [x] **`vastir-pbr`** — ShaderLab-style PBR authoring: pick `Channel`s (albedo/metallic/roughness/normal/
      AO/emissive/opacity) + a surface function → a generated **Cook-Torrance** vertex+fragment pair
      (`GraphicsPipelineSpec`). `PbrShader.create(...).withEnvironment(binding).withMvp()` composes textured
      albedo, image-based lighting (cubemap reflections), and world-space lighting under an MVP transform.
      Verified: lit/rough/metal spheres, a textured PBR sphere, an IBL metal reflecting an environment, and a
      checker-textured PBR sphere transformed by MVP — all on the RTX 2070.

> Deferred (noted): full IBL (prefiltered specular mips + irradiance + BRDF LUT); light/camera as uniforms
> (push constants now exist, so it's small); a normal matrix for non-uniform scale; multiple lights.

## Orchestration & front-ends — supir-vast as the highest-level reusable GPU/Truffle tool

> Direction: make supir-vast a Pontif-free, reusable orchestration layer that any Truffle language can target;
> its first client is the `pontif-framework` language (GPU-subset lowering). Governing ideas: the lowering *is*
> the witness for lowerability; CPU==GPU equivalence makes backend choice/fallback correctness-free; validation
> is woven in as living requirements; nothing in supir-vast may reference the front end.

- [x] `Accelerator` orchestration facade (Pontif-free) — `register(KernelSpec) → KernelHandle | Rejection(witness)`
      (validate ABI → lower SPIR-V → `spirv-val` gate → lower Truffle, each failure a concrete witness),
      `KernelHandle.run` (auto GPU/CPU + safe fallback, input untouched), `.verify` (CPU==GPU on demand),
      `.abi()`/`.spirv()`, `capabilities()`. SoA i32 columns (binding==slot). `AcceleratorTest`.
- [x] Resident GPU context + kernel registry — `GpuContext` (AutoCloseable) holds instance/device/queue once;
      `GpuContext.ResidentKernel` is a pipeline built once and dispatched against repeatedly (only buffers +
      descriptor update + command buffer are per-dispatch). `VulkanCompute` is now a thin one-shot wrapper over
      it (all 8 prior GPU tests unchanged). `Accelerator` opens one context and preloads a pipeline per
      registered kernel; `run` dispatches against it with no per-run lowering/device build. Per-call
      instance/device/pipeline rebuild is gone. `AcceleratorTest` residency case (2 kernels, dispatched 3× each).
- [x] Typed buffer columns (f32) + self-describing ABI — `Buffer`/`Expr.BufferLoad`/`KernelResources` carry an
      element `Type` (per-element-type Block + runtime array, `ArrayStride` by element size); `CoreToTruffle`
      reinterprets f32 bits on the `int[]` wire; `KernelColumn` carries the type so `KernelHandle.abi()` is
      self-describing; non-32-bit columns are rejected with a witness. f32 map (`out[i]=a[i]*1.5f`) CPU==GPU
      through the facade.
- [x] 64-bit (i64/f64) + heterogeneous columns — a 64-bit element rides the `int[]` wire as two words (low
      first); `KernelResources` already strides 8 and `GpuContext` just moves words, so only the Truffle buffer
      nodes (2-word assemble/split, element-type-aware) and the `Accelerator` (accept i64/u64/f64, validate
      words-per-element) changed. **Struct streams are SoA = multiple typed columns — already supported, no new
      primitive.** Test: `out(i64)=(i64)a(i32)+b(i64)` over heterogeneous columns, CPU==GPU. *Still TODO: narrow
      (i8/i16) columns + packed-i8 wire, bool, and a first-class ABI descriptor type.*
- [ ] The `kernel` IR level (above `core`) + feeder passes (monomorphize, defunctionalize, recursion→loop) and a
      `kernel→core` legalization (tag-encode sum types, match→if/else). `type` gains Struct + fixed Array.
- [ ] `pontif-spirv` adapter (on the Pontif side, depends on supir-vast) — `IrExpr→kernel` translation, the
      `OnGpu`-style proof/discharger registration, `RecordValue`↔columns marshalling, witness rendering. First
      milestone: a map-shaped `Iterate` over a monomorphic non-recursive Int/struct function, CPU==GPU.
- [x] Capability profiles — `SpirvTarget` (allowed-capability budget; `CoreToSpirv.lower(module, target)` gates
      required ⊆ allowed, throwing a `CapabilityException` witness). **#1 device-derived:** `GpuContext` queries
      `VkPhysicalDeviceFeatures` + Vulkan1.2 `shaderInt8` → supported `Capability` set, and now *enables* exactly
      those at device creation (fixes the latent gap where Int64/Float64 ran only because no validation layer).
      **#2 optional budget:** `new Accelerator(SpirvTarget)` refuses to generate caps outside the budget even
      where the device supports them. Effective = device ∩ budget; a kernel within budget but beyond the device
      registers CPU-only (transparent fallback); beyond budget → `Rejection`. `capabilities()` reports the device
      set. `AcceleratorTest` (budget refusal + device caps/preload). *Deferred: SPIR-V version downgrade emission,
      and DotProduct/8-bit-storage/subgroup caps until a kernel uses them.*
- [ ] Offload cost model / threshold — Pontif owns the policy, supir-vast publishes a cost hint; correctness-free.
- [ ] Streamed (vs bulk) data feeding — persistent ring buffers + async double-buffering; only if overlap/unbounded
      sources are needed (bulk-per-batch is the v1).

## P3 — Performance, quality, infra

- [ ] Truffle node specialization (`@Specialization`, typed frame slots) — remove `Object` boxing for JIT speed
- [ ] More host platforms in `vastir-tools` — Linux/macOS SDK fetch + LWJGL natives (host=Windows only today)
- [ ] CI — build + `spirv-val`, gracefully skip GPU tests where no device is present
- [ ] README + architecture doc — modules, IR naming convention, pipeline diagram
- [ ] Property/fuzz tests — random `core` programs → `spirv-val` + CPU==GPU agreement
- [ ] Explicit SSA mid-level IR (the MLIR-style split) — only if/when a concrete need appears, not speculatively

## Supir — textual form of the core IR (after the current roadmap)

- [x] `supir` — a new module holding the textual form of the `core` IR: a **flat, high-level assembly**
      (one op per line, every intermediate named, structured `if`/`loop`, no nested expression trees) chosen as
      a human-readable IR substitute / interchange format (the `.ll` / `spirv-dis`+`spirv-as` role).
      **Two-way**: `Supir.parseModule(String)` (text → `core`) and `Supir.print(CoreModule)` (`core` → text),
      with `line:col` diagnostics. Faithful to lowered semantics, not the AST tree — printing normalizes
      (subexpressions → named temps, canonical `p0`/`t0` names), so `print` is a normal form
      (`print(parse(print(m))) == print(m)`, verified by `RoundTripTest`). Grammar in `supir/README.md`.
      The infix "modernized shader language" idea belongs at a higher authoring layer that lowers *to* this IR.
- [x] Wire `supir` into supir-studio (Phase 2): editor (Supir) → `Supir.parseModule` → `core` → `CoreToSpirv`
      → spirv-val → spirv-cross GLSL 330 → `ShaderUtil.buildProgram` → render (`SupirShaderCompiler`). Added a
      GLSL version target to `NativeTools.crossCompile` (`--version 330 --no-es --separate-shader-objects`) so
      varyings carry explicit locations on a 3.3 context. Errors surface in the status bar; last good program
      kept on failure.
- [x] Both stages in Supir: a "Switch stage" toggle edits the vertex or fragment Supir in one editor; Compile
      builds and links both. The vertex's MVP push constant becomes an opaquely-named `uniform` struct; since
      the dasum GL binding can't enumerate uniforms, `SupirShaderCompiler` recovers the matrix uniform name from
      the generated GLSL and the renderer sets it each frame. The two Supir stages link by varying location.
- [x] Export…: writes both stages to a native-dialog-picked folder (NFD from `dasum-natives`) as
      `shader.{vert,frag}.{spv,spvasm,glsl,hlsl,metal}` — SPIR-V binary + assembly (`spirv-dis`) +
      GLSL/HLSL/Metal (`spirv-cross`). OpenCL C isn't a spirv-cross target; the `.spv` feeds OpenCL 2.1+
      runtimes. (`SupirShaderCompiler.exportStage` + `ShaderExport`.)

---

## Known scope limits (tracked, intentional)

Memory-based locals (no `OpPhi`, `spirv-opt --ssa-rewrite` away) · compute kernels: i32/i64/f32/f64 storage
columns, local size 1 (dispatch via workgroup count) · graphics lighting authored model-space unless `--mvp`
(no normal matrix → uniform scale only) · simplified IBL (base-level cubemap, no prefiltered mips/irradiance
convolution/BRDF LUT) · light + camera still hardcoded in shaders (not yet uniforms) · Windows-only natives.
