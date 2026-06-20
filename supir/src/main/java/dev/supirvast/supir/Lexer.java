package dev.supirvast.supir;

import java.util.ArrayList;
import java.util.List;

/**
 * Turns Supir source text into a flat token stream for the {@link Parser}. Supir is a line-oriented,
 * three-address assembly, so the lexer is small: identifiers, int/float literals, and a handful of
 * punctuators ({@code { } [ ] ( ) , = : -> @ < >}). {@code ;} starts a line comment; newlines are ordinary
 * whitespace (structure comes from the grammar, not from line breaks). Every token carries a {@link Span}.
 */
final class Lexer {

    enum Kind {
        LBRACE, RBRACE, LBRACKET, RBRACKET, LPAREN, RPAREN,
        COMMA, EQUALS, COLON, ARROW, AT, LANGLE, RANGLE,
        IDENT, INT, FLOAT, EOF
    }

    record Token(Kind kind, String text, long intValue, double floatValue, Span span) {}

    private final String src;
    private int pos = 0;
    private int line = 1;
    private int col = 1;

    Lexer(String src) {
        this.src = src;
    }

    List<Token> tokenize() {
        List<Token> tokens = new ArrayList<>();
        Token t;
        do {
            t = next();
            tokens.add(t);
        } while (t.kind() != Kind.EOF);
        return tokens;
    }

    private Token next() {
        skipTrivia();
        if (pos >= src.length()) {
            return punct(Kind.EOF, "");
        }
        int startLine = line;
        int startCol = col;
        char c = src.charAt(pos);
        switch (c) {
            case '{' -> { advance(); return new Token(Kind.LBRACE, "{", 0, 0, span(startLine, startCol)); }
            case '}' -> { advance(); return new Token(Kind.RBRACE, "}", 0, 0, span(startLine, startCol)); }
            case '[' -> { advance(); return new Token(Kind.LBRACKET, "[", 0, 0, span(startLine, startCol)); }
            case ']' -> { advance(); return new Token(Kind.RBRACKET, "]", 0, 0, span(startLine, startCol)); }
            case '(' -> { advance(); return new Token(Kind.LPAREN, "(", 0, 0, span(startLine, startCol)); }
            case ')' -> { advance(); return new Token(Kind.RPAREN, ")", 0, 0, span(startLine, startCol)); }
            case ',' -> { advance(); return new Token(Kind.COMMA, ",", 0, 0, span(startLine, startCol)); }
            case '=' -> { advance(); return new Token(Kind.EQUALS, "=", 0, 0, span(startLine, startCol)); }
            case ':' -> { advance(); return new Token(Kind.COLON, ":", 0, 0, span(startLine, startCol)); }
            case '@' -> { advance(); return new Token(Kind.AT, "@", 0, 0, span(startLine, startCol)); }
            case '<' -> { advance(); return new Token(Kind.LANGLE, "<", 0, 0, span(startLine, startCol)); }
            case '>' -> { advance(); return new Token(Kind.RANGLE, ">", 0, 0, span(startLine, startCol)); }
            case '-' -> {
                if (pos + 1 < src.length() && src.charAt(pos + 1) == '>') {
                    advance();
                    advance();
                    return new Token(Kind.ARROW, "->", 0, 0, span(startLine, startCol));
                }
                return atom(startLine, startCol); // a leading '-' otherwise begins a negative number
            }
            default -> {
                return atom(startLine, startCol);
            }
        }
    }

    private void skipTrivia() {
        while (pos < src.length()) {
            char c = src.charAt(pos);
            if (c == ';') {
                while (pos < src.length() && src.charAt(pos) != '\n') {
                    advance();
                }
            } else if (Character.isWhitespace(c)) {
                advance();
            } else {
                return;
            }
        }
    }

    private Token atom(int startLine, int startCol) {
        int start = pos;
        while (pos < src.length() && isAtomChar(src.charAt(pos))) {
            advance();
        }
        String text = src.substring(start, pos);
        Span span = span(startLine, startCol);
        if (looksNumeric(text)) {
            boolean fractional = text.indexOf('.') >= 0 || text.indexOf('e') >= 0 || text.indexOf('E') >= 0;
            if (fractional) {
                try {
                    return new Token(Kind.FLOAT, text, 0, Double.parseDouble(text), span);
                } catch (NumberFormatException ignored) {
                    throw new SupirParseException(span, "malformed float literal '" + text + "'");
                }
            }
            try {
                return new Token(Kind.INT, text, Long.parseLong(text), 0, span);
            } catch (NumberFormatException ignored) {
                throw new SupirParseException(span, "malformed integer literal '" + text + "'");
            }
        }
        return new Token(Kind.IDENT, text, 0, 0, span);
    }

    /** Numeric if it starts with a digit, or a leading {@code -}/{@code .} immediately followed by a digit. */
    private static boolean looksNumeric(String text) {
        if (text.isEmpty()) {
            return false;
        }
        char c0 = text.charAt(0);
        if (Character.isDigit(c0)) {
            return true;
        }
        if ((c0 == '-' || c0 == '.') && text.length() > 1) {
            return Character.isDigit(text.charAt(1));
        }
        return false;
    }

    /** Atom characters: anything not whitespace, a delimiter, or a comment start. ({@code -}/{@code .} are
     *  atom chars so negative and fractional numbers stay one token; {@code <}/{@code >} are delimiters used
     *  for type parameters, so they split atoms.) */
    private static boolean isAtomChar(char c) {
        return switch (c) {
            case '{', '}', '[', ']', '(', ')', ',', '=', ':', '@', '<', '>', '"', ';' -> false;
            default -> !Character.isWhitespace(c);
        };
    }

    private Token punct(Kind kind, String text) {
        return new Token(kind, text, 0, 0, Span.point(line, col));
    }

    private Span span(int startLine, int startCol) {
        return new Span(startLine, startCol, line, col);
    }

    private void advance() {
        if (src.charAt(pos) == '\n') {
            line++;
            col = 1;
        } else {
            col++;
        }
        pos++;
    }
}
