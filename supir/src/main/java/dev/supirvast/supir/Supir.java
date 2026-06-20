package dev.supirvast.supir;

import dev.supirvast.vastir.core.CoreModule;

/**
 * Supir: the textual form of the {@code core} IR. A flat, three-address assembly that is faithful to the IR's
 * lowered semantics, so a tool that cannot reach the in-memory {@link CoreModule} can read, write, and
 * round-trip it as text (the role {@code .ll} plays for LLVM bitcode, or {@code spirv-dis}/{@code spirv-as}
 * for SPIR-V binary).
 *
 * <p>The pair is two-way: {@link #parseModule(String)} reads text into a {@link CoreModule}, {@link #print}
 * writes a {@link CoreModule} back out. Printing is canonical (it normalizes names and flattens nested
 * expressions to temporaries), so {@code print(parseModule(print(m)))} equals {@code print(m)}. See the
 * module {@code README.md} for the grammar.
 */
public final class Supir {

    private Supir() {
    }

    /**
     * Parses Supir flat-assembly source into a {@link CoreModule}, ready for {@code CoreToSpirv}.
     *
     * @throws SupirParseException on any lex, parse, or resolution error (carries the offending {@link Span})
     */
    public static CoreModule parseModule(String source) {
        return Parser.parse(source);
    }

    /** Prints a {@link CoreModule} as canonical Supir flat-assembly text. */
    public static String print(CoreModule module) {
        return Printer.print(module);
    }
}
