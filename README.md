# Supir-Vast

**One shader IR in Java — runs on the GPU as SPIR-V, and on the CPU as an executable Truffle AST, from the same source.**

Supir-Vast is a Java / GraalVM framework for describing GPU programs (compute kernels and shaders) as a
typed, structured intermediate representation called **`core`**. A `core` program lowers two ways:

- **to SPIR-V** — validated binary that runs on any Vulkan GPU and cross-compiles to GLSL / HLSL / Metal; and
- **to an executable Truffle AST** — the *same* program, running (and JIT-compiled by GraalVM) on the CPU.

Because both targets come from one IR, you can run a kernel on the GPU *and* on the CPU and check they agree —
the project's test suite does exactly this on real hardware, byte-for-byte on the results.

```
                                  ┌──  CoreToSpirv  ──►  SPIR-V  ──►  spirv-val ✓  ──►  Vulkan (GPU)
   core IR  (typed, structured) ──┤                         └──►  SPIRV-Cross  ──►  GLSL / HLSL / Metal
                                  └──  CoreToTruffle ──►  Truffle AST  ──►  CPU execution (Graal JIT)
```

The SPIR-V vocabulary (every opcode, enum, and operand) is **generated from Khronos's official machine-readable
SPIR-V grammar**, so the binary layer tracks the spec rather than being hand-transcribed.

---

## What it does today

Author a program once in `core`, get both backends:

```java
// out[i] = a[i] + b[i]  — a data-parallel kernel, written once
Buffer out = new Buffer("out", 0), a = new Buffer("a", 1), b = new Buffer("b", 2);
Expr gid = new Expr.InvocationId();

Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()),
    Region.of(
        new Statement.BufferStore(out, gid,
            new Expr.Binary(BinaryOp.ADD, new Expr.BufferLoad(a, gid), new Expr.BufferLoad(b, gid))),
        new Statement.ReturnVoid()));
CoreModule module = new CoreModule().addEntryPoint(EntryPoint.compute(main, 1, 1, 1));

byte[] spirv     = new CoreToSpirv().lower(module).toByteArray();      // → GPU (validated SPIR-V)
CallTarget cpu   = new CoreToTruffle().lowerKernel(main, List.of(out, a, b)); // → CPU (Truffle AST)
```

The `core` language currently supports:

- **Types:** 32-bit `int`/`uint`, `float`, `bool`, and fixed-size **vectors**.
- **Values & operators:** constants, arithmetic, division/modulo, bitwise, shifts, comparisons, logical
  and/or, unary negate/not — scalar and componentwise on vectors.
- **Structured control flow:** `if` and `while` (lowered to SPIR-V structured `OpSelectionMerge` /
  `OpLoopMerge`, not a reconstructed CFG).
- **Local variables**, **functions with parameters and calls**, and **compute kernels** with storage-buffer
  I/O and `gl_GlobalInvocationID` for data-parallel work.
- **Validation & tooling:** the official `spirv-val`, `spirv-dis`, `spirv-as`, `spirv-opt`, and `spirv-cross`
  binaries are fetched once from the pinned Vulkan SDK and bundled, so validation, disassembly, and
  cross-compilation work out of the box — no separate install.

Everything above is exercised by a **differential test harness** that lowers the same `core` program to both
backends, runs the CPU version (Truffle) and the GPU version (Vulkan compute on the local device), and asserts
the results are identical.

## What it could be used for

- **Testable GPU code.** Run a shader/kernel on the CPU as a plain JVM program — unit-test its *logic* with
  ordinary assertions, no GPU, driver, or capture tool required — then ship the identical IR to the GPU.
- **A CPU reference / fallback.** Execute the same kernel on the CPU when no suitable GPU is present, or as a
  golden reference to validate GPU output against.
- **Shader cross-compilation.** Lower to SPIR-V and cross-compile to GLSL / HLSL / Metal, generating shaders
  for multiple graphics APIs from a single description.
- **A backend for higher-level languages.** `core` is a clean target for a DSL or shading-language frontend:
  emit `core`, get SPIR-V, GPU execution, CPU execution, and cross-compiled source for free.
- **Teaching & research.** A compact, end-to-end, *validated* path from a structured IR to real SPIR-V and a
  running interpreter — useful for exploring shader compilation, progressive lowering, and GPU/CPU equivalence.

## Modules

| Module | Role |
| --- | --- |
| `vastir-codegen` | Build-time generator: emits typed Java from the pinned official SPIR-V grammar. |
| `vastir` | The `core` IR, the shared type system, and the `CoreToSpirv` lowering + SPIR-V binary emitter. No GraalVM dependency. |
| `vast` | `CoreToTruffle` — the executable Truffle AST backend (CPU). |
| `vastir-tools` | Bundled native SPIR-V toolchain, Vulkan compute execution, and the CPU-vs-GPU differential harness. |

## Status

**Experimental and early.** The architecture is proven end-to-end — generated SPIR-V vocabulary → `core` IR →
validated SPIR-V running on a real GPU, *and* the same IR running on the CPU with matching results — but the
language is intentionally small. Current limits include: compute shaders only (no vertex/fragment stages yet),
32-bit scalars, `int` storage buffers, memory-based locals (no SSA `OpPhi` yet), and a fixed workgroup size
(parallelism comes from the dispatch). See [`TODO.md`](TODO.md) for the roadmap.

## Requirements

- **JDK 25 / GraalVM** (the build and CPU JIT target).
- **Maven.** A normal `mvn install` builds everything and runs the tests.
- **First build downloads the Vulkan SDK** (pinned version) to extract the native SPIR-V tools; it is cached
  under `~/.supirvast` thereafter. Pass `-Dvast.skipNativeTools=true` to skip it (the GPU/validation tests then
  skip gracefully). Native bundling is currently wired for **Windows x64**.
- **A Vulkan-capable GPU** is optional — only the GPU-execution tests need one; they skip cleanly without it.

## Building

```bash
mvn install        # build all modules and run the test suite (incl. CPU↔GPU differential)
```

---

*The name: **VAST** is the executable Truffle **A**bstract **S**yntax **T**ree; **vastir** is the IR it shares
with the SPIR-V backend.*
