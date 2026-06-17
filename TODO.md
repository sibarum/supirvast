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
- [ ] Relax 32-bit-only — unsigned semantics end-to-end, then `f64`/`i64`/narrow ints
- [ ] SSA values via `OpPhi` across `if`/`while` (drop memory round-trips where it's a win)
- [ ] Early `return` inside structured regions

## P2 — Targets & toolchain

- [ ] Vertex/fragment stages + interface I/O (locations) — exercises the GLSL/HLSL/Metal story for real shaders
- [ ] Cross-compile coverage — snapshot/verify GLSL/HLSL/Metal output via `spirv-cross`
- [ ] Optional `spirv-opt` pass integration (bundled but unused today)
- [ ] Optional native textual backends (`CoreToGlsl`, …) if we ever want to bypass `spirv-cross`

## P3 — Performance, quality, infra

- [ ] Truffle node specialization (`@Specialization`, typed frame slots) — remove `Object` boxing for JIT speed
- [ ] More host platforms in `vastir-tools` — Linux/macOS SDK fetch + LWJGL natives (host=Windows only today)
- [ ] CI — build + `spirv-val`, gracefully skip GPU tests where no device is present
- [ ] README + architecture doc — modules, IR naming convention, pipeline diagram
- [ ] Property/fuzz tests — random `core` programs → `spirv-val` + CPU==GPU agreement
- [ ] Explicit SSA mid-level IR (the MLIR-style split) — only if/when a concrete need appears, not speculatively

---

## Known scope limits (tracked, intentional)

32-bit scalars only · memory-based locals (no `OpPhi`) · i32 storage buffers only (no other element types) ·
local size fixed at 1 (dispatch via workgroup count) · no function calls · no vertex/fragment I/O ·
early-return only at top level · Windows-only natives.
