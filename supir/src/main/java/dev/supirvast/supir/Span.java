package dev.supirvast.supir;

/**
 * A source span, 1-based line/column, half-open at the end column. Carried by every {@link SExpr} node and
 * every {@link SupirParseException} so diagnostics can point at the exact offending text.
 */
public record Span(int startLine, int startCol, int endLine, int endCol) {

    /** A zero-width span at a single position — used for tokens whose end is the next character. */
    public static Span point(int line, int col) {
        return new Span(line, col, line, col);
    }

    /** A span covering this span's start through {@code other}'s end (this must start no later than other). */
    public Span through(Span other) {
        return new Span(startLine, startCol, other.endLine, other.endCol);
    }

    /** {@code line:col} of the start, for prefixing diagnostic messages. */
    public String location() {
        return startLine + ":" + startCol;
    }
}
