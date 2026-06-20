package dev.supirvast.supir;

/**
 * A Supir lex/parse/lower error, carrying the {@link Span} of the offending text. {@link #getMessage()} is
 * prefixed with {@code line:col}; {@link #render(String)} additionally shows the source line with a caret,
 * which the studio can drop straight into its status bar.
 */
public final class SupirParseException extends RuntimeException {

    private final Span span;

    public SupirParseException(Span span, String message) {
        super(span.location() + ": " + message);
        this.span = span;
    }

    public Span span() {
        return span;
    }

    /**
     * The message plus the offending source line and a caret underneath the span start — e.g.
     * <pre>
     * 3:9: unknown form 'vec5'
     *   (write (vec5 1.0 2.0))
     *          ^
     * </pre>
     * Falls back to the bare message if the span's line is out of range for {@code source}.
     */
    public String render(String source) {
        String[] lines = source.split("\n", -1);
        int idx = span.startLine() - 1;
        if (idx < 0 || idx >= lines.length) {
            return getMessage();
        }
        String line = lines[idx].replace("\t", " ");
        String caret = " ".repeat(Math.max(0, span.startCol() - 1)) + "^";
        return getMessage() + "\n  " + line + "\n  " + caret;
    }
}
