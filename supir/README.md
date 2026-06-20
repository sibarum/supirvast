# supir — the textual form of the core IR

Supir is a **flat, high-level assembly** that is the textual form of the `core` IR
(`dev.supirvast.vastir.core`). One operation per line, every intermediate named, structured `if`/`loop` blocks
for control flow, and no nested expression trees. It is faithful to the IR's *lowered semantics* rather than to
the AST's tree shape, so a tool that can't reach the in-memory `CoreModule` can read, write, and round-trip it
as text — the role `.ll` plays for LLVM bitcode, or `spirv-dis`/`spirv-as` for SPIR-V binary.

The pair is **two-way**:

```
text ──Supir.parseModule──▶ core CoreModule ──Supir.print──▶ text
                                 │
                                 └─ CoreToSpirv ──▶ SPIR-V
```

`print` is canonical (it normalizes names and flattens nested expressions to temporaries), so it is a normal
form: `print(parseModule(print(m)))` equals `print(m)`.

## Usage

```java
CoreModule module = dev.supirvast.supir.Supir.parseModule(source);
String text = dev.supirvast.supir.Supir.print(module);        // core -> text
byte[] spirv = new dev.supirvast.vastir.lower.CoreToSpirv().lower(module).toByteArray();
```

Errors throw `SupirParseException` with a `line:col` location; `exception.render(source)` adds the source line
with a caret.

## Example

A vertex shader (`gl_Position` from the vertex index, plus a colour varying). Note how the `i32 -> f32`
conversion is its own named line — nothing nests:

```
vertex main {
  out vColor: vec4 @loc 0

  t0 = convert VertexIndex, f32
  Position = vec4 t0, 0.0, 0.0, 1.0
  vColor = vec4 1.0, 0.0, 0.0, 1.0
  ret
}
```

The matching fragment shader:

```
fragment main {
  in vColor: vec4 @loc 0
  out fragColor: vec4 @loc 0

  fragColor = vColor
  ret
}
```

A compute kernel with a loop (the loop condition stays inline in the header, since it is re-evaluated each
iteration):

```
compute main local_size(1, 1, 1) {
  buffer data: i32 @binding 0

  sum = 0
  i = 0
  loop while lt i, 10 {
    v = data[i]
    sum = add sum, v
    i = add i, 1
  }
  store_result sum
  ret
}
```

## Grammar

A source file is a sequence of top-level items: entry points, standalone functions, and (optionally)
module-level resource declarations.

### Top-level items
- `vertex NAME { … }` · `fragment NAME { … }`
- `compute NAME local_size(X, Y, Z) { … }`
- `fn NAME(p0: TYPE, p1: TYPE, …) -> RETTYPE { … }` — a callable function (define before it is called)

### Resource declarations
Introduce a name; emit no instruction. Valid at module level or at the top of a body.
- `in NAME: TYPE @loc N` / `out NAME: TYPE @loc N` — stage interface (varyings, attributes, outputs)
- `buffer NAME: ELEMTYPE @binding N` — storage buffer
- `texture NAME [@set N] @binding N` — 2D sampled texture · `cubemap NAME @binding N` — cubemap
- `push NAME: TYPE` or `push { a: TYPE, b: TYPE }` — the single push-constant block

### Types
`void` `bool` · ints `i8 i16 i32 i64`, unsigned `u8 u16 u32 u64` · floats `f16 f32 f64` ·
`vec2`/`vec3`/`vec4` (float; element override `vec3<i32>`) · general `vec<ELEMTYPE, N>` ·
`mat2`/`mat3`/`mat4` (square float) · general `mat<COLUMNS, COLUMNVEC>`.

### Instructions (one per line)
- `NAME = RHS` — define a local (first use) or reassign it (later use); `NAME: TYPE = RHS` forces the type
- `OUTVAR = RHS` — write a stage output (when NAME is a declared `out`)
- `Position = RHS` — write a built-in output
- `BUF[INDEX] = RHS` — store to a buffer
- `ret` / `ret ATOM` — return · `store_result ATOM` — write the kernel result buffer
- `if COND { … }` / `if COND { … } else { … }`
- `loop while COND { … }`

### Right-hand sides
A right-hand side is a single **operation** or a bare **atom**. Operation operands are always atoms — there is
no nesting (that is the whole point; everything intermediate is named).

- atoms: `42` (i32), `1.0` (f32), `true`/`false`, a name (param / local / `in` var / push-constant member),
  `VertexIndex`, `invocation_id`, `BUF[INDEX]`
- typed constants / conversions: `u32 5` (constant), `f32 idx` (conversion); also `convert x, i64`,
  `bitcast x, u32`
- arithmetic: `add sub mul div mod` · bitwise: `band bor bxor shl shr` · compare: `lt gt eq` ·
  logical: `land lor` · unary: `neg bnot lnot`
- vectors: `vec4 a, b, c, d` (and `vec3`/`vec2`, with `<T>` for non-float); `extract v, INDEX`;
  `mtimes M, V` (matrix × vector)
- resources & calls: `sample TEX, UV` · `call FN, args…` · `pc MEMBER`
- math (GLSL.std.450): `dot length distance normalize cross reflect refract face_forward pow sqrt
  inverse_sqrt abs sign min max clamp mix step smoothstep fma exp log exp2 log2 sin cos tan asin acos atan
  atan2 radians degrees sinh cosh tanh asinh acosh atanh round round_even trunc floor ceil fract`

### Lexical
Whitespace-separated; newlines are not significant (mnemonics have fixed arity). `{ } [ ] ( ) , = : -> @ < >`
are punctuators; `;` starts a line comment. `->` is one token; `-3` is a negative number.

## Faithfulness & round-trip

Flat assembly is faithful to the **lowered semantics**, not a structural mirror of the nested AST. Printing
normalizes: subexpressions become named temporaries (`t0`, `t1`, …) and parameters/locals get canonical names
(`p0`, …). So `parse → print` is not the identity on the original *tree*; the guarantee is that **printing is a
normal form** — `print(parse(print(m))) == print(m)` — verified by `RoundTripTest`, which also lowers each
parsed module to valid SPIR-V.

One intentional limit (consistent with the project's other scope limits): a `loop while` condition is a single
inline atom or operation with atom operands. Every `While` in the codebase is a single comparison, so this is
sufficient; a deeper condition would need hoisting, which a loop header cannot do without changing semantics.

## Scope

A leaf module depending only on `vastir`; no native dependencies, fully tested headlessly. Wiring it into
supir-studio (editor → `Supir.parseModule` → `core` → spirv-cross 330 → GL) is the next step, tracked in the
studio README.
