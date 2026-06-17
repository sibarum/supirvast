package dev.supirvast.vastir.codegen;

import javax.lang.model.SourceVersion;

/** Small helpers for emitting Java source text from grammar values. */
final class JavaSource {

    private JavaSource() {
    }

    /**
     * Turns a grammar name into a legal, non-keyword Java identifier. SPIR-V names are almost all already
     * valid; the exceptions are a handful of leading-digit names (e.g. {@code "1D"}, {@code "2x2"}), which
     * get an underscore prefix. Keywords are suffixed with {@code _} defensively for future grammar bumps.
     */
    static String identifier(String name) {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("empty grammar name");
        }
        StringBuilder b = new StringBuilder(name.length() + 1);
        if (!Character.isJavaIdentifierStart(name.charAt(0))) {
            b.append('_');
        }
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            b.append(Character.isJavaIdentifierPart(c) ? c : '_');
        }
        String id = b.toString();
        return SourceVersion.isKeyword(id) ? id + "_" : id;
    }

    /** Maps a grammar operand quantifier to a {@code Quantifier} constant reference. */
    static String quantifierExpr(String quantifier) {
        if (quantifier == null) {
            return "Quantifier.REQUIRED";
        }
        return switch (quantifier) {
            case "?" -> "Quantifier.OPTIONAL";
            case "*" -> "Quantifier.VARIADIC";
            default -> throw new IllegalArgumentException("unknown quantifier: " + quantifier);
        };
    }

    /** Renders a Java string literal, or the {@code null} keyword for a null value. */
    static String stringLiteral(String s) {
        if (s == null) {
            return "null";
        }
        StringBuilder b = new StringBuilder(s.length() + 2).append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '\\' -> b.append("\\\\");
                case '"' -> b.append("\\\"");
                case '\n' -> b.append("\\n");
                case '\r' -> b.append("\\r");
                case '\t' -> b.append("\\t");
                default -> b.append(c);
            }
        }
        return b.append('"').toString();
    }
}
