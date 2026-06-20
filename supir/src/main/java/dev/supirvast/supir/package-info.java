/**
 * Supir — the textual form of the {@code core} IR ({@link dev.supirvast.vastir.core}). A flat, three-address
 * assembly: one operation per line, every intermediate named, structured {@code if}/{@code loop} blocks for
 * control flow, and no nested expression trees. It is faithful to the IR's lowered semantics rather than to the
 * AST's tree shape, so it can serve as a human-readable IR substitute / interchange format for tools that
 * cannot reach the in-memory {@link dev.supirvast.vastir.core.CoreModule}.
 *
 * <p>The pair is two-way: {@link dev.supirvast.supir.Parser} reads text into a {@code CoreModule} and
 * {@link dev.supirvast.supir.Printer} writes one back out (flattening nested expressions to temporaries).
 * Printing is canonical, so {@code print} is a normalizing function. Use the
 * {@link dev.supirvast.supir.Supir} facade; see {@code README.md} for the grammar.
 */
package dev.supirvast.supir;
