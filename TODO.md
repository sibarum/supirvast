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
- [x] Atomics — `Statement.AtomicUpdate` (`AtomicOp`: add, sub, min, max, and, or, xor, exchange) and
      `Statement.AtomicCompareExchange`, each assigning the old value to an optional `previous`, which the
      statement declares only if nothing else does (so a retry loop reuses one variable). **Statements, not
      expressions**: a side effect inside a tree is one a folding or dead-code pass above this IR can duplicate
      or erase. The table is SPIR-V's and is enforced at construction: every op on i32/u32 (min/max pick
      `S`/`U` by signedness), add/min/max/exchange on f32, compare-exchange integer-only. Device scope, relaxed
      semantics. Float add and float min/max are `AtomicFloat32AddEXT`/`AtomicFloat32MinMaxEXT` plus their
      `OpExtension`s — a new extensions section in the module — and go through the target budget, so a device
      without them registers the kernel CPU-only; `GpuContext` detects `VK_EXT_shader_atomic_float`/`_float2`
      and enables them. An atomic target counts as stored for the `NonWritable` derivation. The CPU backend's
      dispatch is sequential, so a plain read-modify-write is atomic by construction. Supir spells them
      `old = atomic add buf[i], v`, `atomic max buf[i], v` and `old = atomic cmpxchg buf[i], expected, desired`.
      `KernelColumn` gained a fixed `length` (`withLength`), because the shape atomics exist for — many
      invocations reducing into few elements — was unmarshallable under one-element-per-invocation.
      `AtomicKernelTest`: histogram, ticket allocation, signed/unsigned min/max, bitwise, a compare-exchange
      retry loop, exchange, f32 add and min/max — each backend checked against the order-independent answer.
      *Still TODO: 64-bit atomics (`Int64Atomics`, `shaderBufferInt64Atomics`); atomics on workgroup memory,
      which needs workgroup memory first; the graphics-stage device features (`fragmentStoresAndAtomics`).*

## P2 — Targets & toolchain

- [x] Vertex/fragment stages + interface I/O (locations) — core `Builtin` (`gl_Position`/`gl_VertexIndex`) and
      `InterfaceVar` (location-bound Input/Output), with `Expr.BuiltinRead`/`InterfaceRead` and
      `Statement.BuiltinWrite`/`InterfaceWrite`. `CoreToSpirv` declares Input/Output `OpVariable`s with
      `BuiltIn`/`Location` decorations and lists them in the entry-point interface; vertex/fragment execution
      models already existed. CPU backend rejects these graphics-only constructs (no rasterizer). A vertex +
      fragment pair (`VertexFragmentShaderTest`) validates and cross-compiles to GLSL/HLSL/MSL. *Verification is
      `spirv-val` + cross-compile, not CPU==GPU — a fragment shader needs rasterization the headless compute
      path can't drive. Still TODO: vertex attribute inputs, multiple render targets, more builtins.*
- [x] `Builtin.FRAG_DEPTH` — `gl_FragDepth` as a fragment output, `float`. The built-in a ray-marched fragment
      cannot do without: it draws over a fullscreen triangle whose vertices carry one fixed depth, so without
      this it can only contribute that constant, and occludes a rasterised mesh at a flat plane instead of at
      its own geometry. Writing it makes the lowering declare the `DepthReplacing` execution mode — required
      rather than advisory, since a `FragDepth` store without it is invalid SPIR-V, and it is also the whole
      cost of the feature: it tells the implementation the depth test cannot be settled before the fragment
      shader runs, so that pipeline loses early-z. Declared only when the module actually writes the built-in,
      so no other fragment shader pays for it. `FragDepthShaderTest` checks both directions — `spirv-val`
      accepts the writing shader and cross-compiles it to `gl_FragDepth`, and the mode is absent from a shader
      that writes none. *The clip-depth convention itself — near plane, far plane, the curve between them, and
      whether "depth" means distance to the eye or to the image plane — is deliberately not modelled here:
      `core` would have to acquire a notion of camera, and two stages disagreeing about that convention is not
      something a type in this enum could catch.*
- [x] `NonWritable` on storage buffers nothing stores to — **derived, not declared**. Without the
      `fragmentStoresAndAtomics` device feature, every storage buffer a *fragment* stage declares must carry
      this decoration (`VUID-RuntimeSpirv-NonWritable-06340`), and a ray-marched fragment reading its geometry
      out of a buffer is exactly that shape; VexelRay's `MarchSmokeTest` reported it on every pipeline it
      built. A flag on `Buffer` would have worked and would have put the decoration in the author's hands,
      where the direction that goes wrong is the dangerous one: a buffer marked read-only and then written is
      a promise to the driver that **nothing in this toolchain catches** — `spirv-val` does not object and the
      kernel computes the right answer until an implementation acts on what it was told. So `CoreToSpirv`
      collects the bindings some `Statement.BufferStore` targets and decorates every other buffer, which makes
      the decoration and the code the same fact. On the variable, not the block member: variables are already
      per buffer, where the block type is shared by every buffer of one element type and could not carry a
      decoration that differs between two of them. `NonWritableBufferTest` pins both directions, including a
      store nested inside an `if`.
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

## Workgroups — the build order (decided 2026-09-22)

Driven by `vexelray-sim-fluid`, whose particle-to-grid scatter is the first kernel that will want all of it.
In this order, and each of the last two only once a kernel is **measured** to need it:

- [x] **1. Configurable workgroup size.** `KernelSpec.workgroupSize`, default 64 (two NVIDIA warps, one AMD
      wavefront), `withWorkgroupSize` to change it. Above one, `Accelerator` lowers a guarded copy for the
      GPU — a first statement returning from every invocation at or past `n`, read from a 4-byte push
      constant so one pipeline serves every `n` — and `GpuContext` dispatches `ceil(n / size)` groups with
      the count pushed. The CPU lowers the kernel as given. `WorkgroupSizeTest` counts executed invocations
      with an atomic at sizes 1, 7, 32, 64 and 256 over `n = 1000`, so a tail that ran would be caught rather
      than landing harmlessly past the buffer. **Measured** on 2²⁰ invocations: at 8192 multiply-adds each,
      48.1 ms at size 1 against 17.7 ms at 64; at 512, 20.7 against 17.1 — because ~16 ms of every run is
      allocation, upload and readback, the kernel itself went from ~32 ms to ~1 ms. `LocalInvocationId` and
      `WorkgroupId` moved to step 2: nothing can use them before workgroup memory, and their CPU meaning
      belongs with that step's phase design.
- [x] **1½. Resident, device-local buffers.** Step 1's measurement put ~16 ms of fixed cost on every 4 MB
      run, because `GpuContext` allocated per dispatch from `HOST_VISIBLE | HOST_COHERENT` memory — reached
      across PCIe by the kernel, read back uncached by the host. Now `Accelerator.allocate(type, elements)`
      returns a `ResidentBuffer`: `DEVICE_LOCAL` memory with a GPU, a host array without, the same to a
      caller. `write`/`read` are staged copies, the readback's staging `HOST_CACHED` where the device has it.
      `KernelHandle.dispatch(buffers, n)` runs against them in place and returns without waiting: every
      resident command buffer goes on one queue and opens with a memory barrier, whose first scope is all
      earlier work in submission order, so each dispatch sees its predecessor's writes and cannot overwrite
      what its predecessor still reads. At most 64 are in flight before a dispatch waits for the oldest;
      finished ones are reclaimed lazily. A handle that cannot use the GPU dispatches on the CPU — over host
      arrays in place, over device buffers by read, run, write back. `release` and `close` wait for resident
      work before destroying a pipeline under it. `ResidentBufferTest`: a hundred ping-pong steps queued with
      no read between, an atomic histogram accumulating across dispatches, both CPU fallbacks, and the
      refusals. **Measured** on a 2²⁰-element f32 field stepped 100 times: 34.8 ms per step through `run`,
      0.35 ms per step resident, one read included. *Still per dispatch: a descriptor pool and set; caching
      them per buffer tuple is the next cut if dispatch overhead ever shows.*
- [x] **2. Workgroup memory and barriers.** `SharedArray` (fixed length, by identity) in the `Workgroup`
      storage class, read with `Expr.SharedLoad`, written with `Statement.SharedStore`, updated with
      `SharedAtomicUpdate`/`SharedAtomicCompareExchange` (workgroup scope; the buffer table, so f32 add, min,
      max and exchange too);
      `Statement.Barrier` is `OpControlBarrier` at workgroup scope, acquire-release over workgroup *and*
      buffer memory — the CPU's phases make other invocations' buffer writes visible too, so a barrier that
      ordered only workgroup memory would let the backends disagree. `Expr.LocalInvocationId`,
      `Expr.WorkgroupId` and `Expr.InvocationCount` (the requested `n`, a push constant the lowering declares
      and every compute pipeline now carries). **Uniformity** is `Barriers.check`, run by both lowerings: a
      barrier only under conditions computed from constants, the workgroup id, the count, push constants and
      locals assigned only from those in uniform flow, and after no return some invocations may have taken.
      **The tail:** a kernel with a barrier cannot stop invocations past `n` with an early return, so it runs
      whole workgroups on both backends and bounds itself with `InvocationCount`; every other kernel keeps the
      guard. **Budget:** the shared arrays' total goes against `SpirvTarget.maxWorkgroupMemoryBytes` as a
      `CapabilityException`; `GpuContext` reads `maxComputeSharedMemorySize`, so over the device is CPU-only
      and over the caller's budget is a `Rejection`. **CPU:** `CoreToTruffle.lowerDispatch` returns a
      `CpuKernel` that runs by workgroup when the kernel uses any of this. The plan splits at barriers into
      phases, each run by every live invocation before the next. Each invocation's locals persist in its own
      materialized frame, and a group-level `if`/`while` evaluates its condition for every invocation and
      throws if they disagree. Supir spells it `shared tile: i32[64]`, `tile[i]`, `barrier`,
      `local_invocation_id`, `workgroup_id`, `invocation_count`. `WorkgroupMemoryTest`: an in-workgroup
      reversal at sizes 7–256, a tree reduction, a shared-memory histogram, a one-winner compare-exchange, the
      indices, both budget outcomes, a divergent-barrier rejection and a resident barrier kernel, each checked
      against the answer on both backends at `n = 1000`. **Measured** summing 2²⁰ i32 at size 256, resident:
      0.27 ms by one global atomic per invocation, 0.49 ms by tree reduction and one global atomic per
      workgroup. The driver already coalesces same-address atomics within a subgroup, so the reduction's eight
      barriers cost more than they save. That is the case for step 3 (a subgroup add before the global
      atomic), and the reason a reduction is not automatically a win here. **Float atomics on workgroup
      memory** need both the capability a buffer's would and a Vulkan feature per kind of memory
      (`shaderSharedFloat32AtomicAdd` is not `shaderBufferFloat32AtomicAdd`). So `SpirvTarget` budgets
      `DeviceFeature`s beside capabilities, `GpuContext` detects and enables the shared and buffer ones, and
      a kernel whose feature the device lacks registers CPU-only. A pre-reducing f32 scatter (the shape
      `vexelray-sim-fluid` needs) is in `WorkgroupMemoryTest`. *Not yet: 2-D/3-D workgroups, and barriers in
      callees.*
- [ ] **The dispatch floor.** Measured by `vexelray-sim-fluid` (cfec712, `SortTest.dispatchFloor`, RTX): every
      `KernelHandle.dispatch` costs a fixed ~0.021 ms, even for an empty 256-invocation kernel, over 500
      resident dispatches and one read. It now dominates short passes. The fluid counting sort is five passes
      (count, three scans, permute), each 0.013–0.025 ms whatever its size, so ~0.1 ms of a 0.16 ms sort is
      floor. The shallow-water step (0.045 ms at 2²⁰ cells) is about half floor, and step 3's segmented
      scatter (0.032–0.037 ms) probably mostly. At 4 ppc sort + gather costs 0.20 ms against 0.095 direct;
      without the floor they would be level, and sort + gather would win above 4 ppc. The overhead is
      step 1½'s deferred note: a descriptor pool and set, a command buffer and a submission, all per
      dispatch. Two cuts: cache descriptor sets per buffer tuple, and record several dispatches, with
      barriers between them, into one command buffer submitted once. The second is what a multi-pass step (a
      sort, a whole FLIP step) wants, and needs an API for a sequence of dispatches.
- [x] **Pick the discrete GPU.** `GpuContext` took the first device with a compute queue, which on this
      machine is the Intel iGPU, not the RTX 5070 Ti — so **every measurement above was the iGPU's**. Now it
      prefers discrete, then integrated, then anything else (`DeviceSelection`), as VexelRay's renderer does.
      `-Dsupirvast.gpu=integrated|discrete|<part of a name>` overrides it, and an override matching nothing
      is an error listing the devices, never a fallback. `Capabilities` reports `deviceName`/`deviceType`,
      and the measurement tests print them. The two differ where it matters: the RTX has 48 KB of workgroup
      memory against 32 KB and shared float add, but no `VK_EXT_shader_atomic_float2`, so float min/max
      atomics now run CPU-only by default. **Measured**, same tests, iGPU → RTX:

      | | Intel iGPU | RTX 5070 Ti |
      |---|---|---|
      | resident step, 2²⁰ f32 | 0.355 ms | 0.042 ms |
      | step through `run` (upload + readback) | 35.2 ms | 35.4 ms |
      | 2²⁰ × 8192 multiply-adds, size 1 / 64, via `run` | 54.4 / 19.7 ms | 186 / 19.0 ms |
      | sum 2²⁰ i32: global atomics / tree reduction | 0.30 / 0.46 ms | 0.068 / 0.070 ms |

      Through `run` both are bound by host-visible transfers, so the devices look alike; resident, the RTX
      is 8.5× faster. A workgroup of one costs the RTX 10× (one lane of 32), so step 1's default of 64 matters
      more there. Same-address atomics are coalesced on both, and the tree reduction only draws level on the
      RTX.
- [x] **3. Subgroup operations.** `Statement.SubgroupArithmetic` (add, mul, min, max, and, or, xor; reduce,
      inclusive or exclusive scan; 32-bit ints and f32), `SubgroupShuffle` (by index, xor, up, down) and
      `SubgroupVote` (all, any, all-equal), plus `Expr.SubgroupInvocationId` and `SubgroupSize`. Statements
      rather than expressions, as atomics are: the result depends on other lanes. **The subgroup size is the
      kernel's** (`KernelSpec.subgroupSize`, default 32), since which lanes a reduction combines is part of
      what it means. The GPU pipeline requires full subgroups of exactly that size (Vulkan 1.3 size control),
      the CPU runs the same (lane = local id % size), and a device that cannot give the size runs the kernel
      CPU-only. The workgroup must be a whole number of subgroups, or registration is rejected. Each kind
      asks for its own `GroupNonUniform*` capability, derived from the device's supported operations and
      budgeted like any other. A subgroup operation is a collective point, so `Barriers.check` holds it to
      uniform control flow, a kernel with one runs whole workgroups, and on the CPU it is a phase boundary:
      every lane evaluates its operand, each subgroup is combined, every lane gets its result. Supir spells
      it `s = subgroup reduce add, v`, `u = subgroup shuffle up, v, 1`, `a = subgroup vote all, b`. A
      segmented sum keyed by cell is not a primitive. `SubgroupTest` builds it from these: each lane finds
      its run's start (an inclusive max over the lanes whose lower neighbour has another key), a Hillis–Steele
      scan by shuffle-up adds the lane `d` below only when it is at or after that start, and each run's last
      lane does one atomic. It is checked on both GPUs and the CPU, with every other operation, against known
      answers. By run, not by key: comparing the key `d` lanes down is right only for sorted input, and double
      counts `A B A` within a subgroup. Particles after advection are only nearly sorted. That was my first
      version; `vexelray-sim-fluid` found it (865ec14), and `aSegmentedSumKeepsBrokenRunsApart` now pins it.
      **Measured** scattering 2²⁰ sorted f32 into cells (ms, one atomic per invocation / segmented, the
      corrected version):

      | run length | 4 | 16 | 64 |
      |---|---|---|---|
      | RTX 5070 Ti | 0.034–0.077, either | 0.033–0.077, either | 0.083–0.087 / 0.031–0.035 |
      | Intel iGPU | 0.706 / 0.886 | 1.791 / 0.664 | 2.245 / 0.336 |

      The segmented sum is flat in run length while per-invocation atomics grow with it, which is the
      contention the fluid scatter hit. On the RTX, short runs sit at the ~0.021 ms dispatch floor (above)
      and swing ±50% between runs, so only the 64 column is a result there: 2.7×. On the iGPU, well above the
      floor, it pays from 16 per cell, and 6.7× at 64. In the fluid scatter itself (bilinear, 12 values per
      lane, 865ec14) the segmented mode is flat at 0.058 ms from 4 to 64 ppc, against 0.095–0.62 direct
      and 0.04–0.27 for a sort-dependent gather. It needs rough cell order, not a sort. **Why it was built (2026-09-24, `vexelray-sim-fluid` fc40ad8):** the FLIP scatter in cell order
      was contention-bound, and pre-reducing in workgroup memory took only 10–20% off, because every
      particle still did an atomic on its cell's slot. By a8212c5 that side found a no-atomics gather faster
      still (4 ppc: 0.05 ms against 0.095 direct), and corrected its earlier low-ppc numbers, which were ~2×
      slow from GPU idle clocks. So the scatter no longer needs this, but the gather runs short of
      parallelism in 3D or at high ppc, and its GPU sort's scans are subgroup work. *Not yet: ballot,
      broadcast, clustered and quad operations; 8-, 16- and 64-bit operands (`shaderSubgroupExtendedTypes`).*

## A CPU runtime — decided against building one yet (2026-09-22)

The CPU backend is a faithful reference, not a production runtime: every value boxed, one call per
invocation, single-threaded, no vectorisation. Whether it should become one was weighed and **deferred**:

- **Nothing forces the decision.** A CPU lowering is another backend under `core`; building it later costs
  what building it now would. What would close the option is `core` acquiring GPU-only constructs with no
  CPU meaning — so every new feature settles its CPU semantics when it lands (step 2 above is the example).
- **Not a third backend.** Every IR feature is already implemented twice. A third that must track every
  addition is the maintenance tax that retired three hand-maintained applications from `vexelray-framework`.
- **Not a replacement for Truffle either.** Only Truffle compiles at run time inside a native image, and
  VexelRay's render == sim for user-authored geometry depends on that — a surface typed at run time is
  queryable on the CPU at once. A build-time bytecode backend could not do it.
- **So if it is ever built, it is Truffle hardened:** primitive specialisations (the P3 entry below), the
  dispatch loop inside the compiled root, workgroups across platform threads. Its costs are named in
  advance: Truffle's compiler in the binary, and a first-use compile that is a frame spike unless it happens
  on a background thread with the interpreter covering.
- **What would justify it:** small volumes measured faster end-to-end on the CPU once readback is counted;
  a headless server or a GPU saturated by rendering as a real target; results needed on the CPU in the same
  frame. A build-time bytecode backend is reconsidered only if hardened Truffle is measured to fall well
  short of hand-written Java, or its binary size or warm-up proves unacceptable.

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
