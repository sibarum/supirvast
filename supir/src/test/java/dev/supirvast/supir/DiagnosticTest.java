package dev.supirvast.supir;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Error reporting: malformed input yields a {@link SupirParseException} with a useful message and location. */
class DiagnosticTest {

    private static SupirParseException error(String source) {
        return assertThrows(SupirParseException.class, () -> Supir.parseModule(source));
    }

    private static SupirParseException errorContaining(String source, String fragment) {
        SupirParseException e = error(source);
        assertTrue(e.getMessage().contains(fragment),
                () -> "expected message to contain \"" + fragment + "\" but was: " + e.getMessage());
        return e;
    }

    /** A fragment shader whose body is the (broken) line under test. */
    private static String inBody(String line) {
        return "fragment main {\n" + line + "\nret\n}";
    }

    @Test
    void noEntryPoint() {
        errorContaining("fn helper() -> void {\nret\n}", "no entry point");
    }

    @Test
    void unknownTopLevelItem() {
        errorContaining("widget main {\nret\n}", "unknown top-level item 'widget'");
    }

    @Test
    void unknownMnemonicReadsAsUndefinedName() {
        // An unknown head isn't a known mnemonic, so it falls through to atom resolution and fails there.
        errorContaining(inBody("x = frobnicate 1"), "undefined name 'frobnicate'");
    }

    @Test
    void undefinedName() {
        errorContaining(inBody("x = q"), "undefined name 'q'");
    }

    @Test
    void undefinedBuffer() {
        errorContaining(inBody("x = nosuch[0]"), "undefined buffer 'nosuch'");
    }

    @Test
    void unknownType() {
        errorContaining(inBody("out color: i7 @loc 0"), "unknown type 'i7'");
    }

    @Test
    void writingToInput() {
        errorContaining(inBody("in vIn: vec4 @loc 0\nvIn = vIn"), "cannot write to input 'vIn'");
    }

    @Test
    void duplicateLocal() {
        // A typed re-declaration of an existing local is a duplicate.
        errorContaining(inBody("x = 1\nx: f32 = 2.0"), "duplicate local 'x'");
    }

    @Test
    void unbalancedBrace() {
        errorContaining("fragment main {\nret\n", "unbalanced '{'");
    }

    @Test
    void callToUndefinedFunction() {
        errorContaining(inBody("x = call ghost"), "call to undefined function 'ghost'");
    }

    @Test
    void reportsLineAndColumn() {
        String source = """
                fragment main {
                  out color: vec4 @loc 0
                  color = q
                  ret
                }""";
        SupirParseException e = errorContaining(source, "undefined name 'q'");
        assertEquals(3, e.span().startLine());
        assertEquals(11, e.span().startCol());
    }

    @Test
    void renderShowsSourceLineWithCaret() {
        String source = inBody("x = q");
        SupirParseException e = error(source);
        String rendered = e.render(source);
        assertTrue(rendered.contains("^"), () -> "expected a caret in:\n" + rendered);
        assertTrue(rendered.contains("x = q"), () -> "expected the source line in:\n" + rendered);
    }
}
