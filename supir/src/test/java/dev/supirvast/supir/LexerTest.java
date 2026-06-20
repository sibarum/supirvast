package dev.supirvast.supir;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** Tokenization: punctuators, literal classification, the {@code ->} digraph, comments, and spans. */
class LexerTest {

    private static List<Lexer.Token> lex(String src) {
        return new Lexer(src).tokenize();
    }

    private static Lexer.Kind[] kinds(String src) {
        return lex(src).stream().map(Lexer.Token::kind).toArray(Lexer.Kind[]::new);
    }

    @Test
    void punctuatorsAndArrow() {
        assertArrayEquals(
                new Lexer.Kind[]{
                        Lexer.Kind.LPAREN, Lexer.Kind.RPAREN, Lexer.Kind.ARROW, Lexer.Kind.LBRACE,
                        Lexer.Kind.RBRACE, Lexer.Kind.LBRACKET, Lexer.Kind.RBRACKET, Lexer.Kind.COMMA,
                        Lexer.Kind.EQUALS, Lexer.Kind.COLON, Lexer.Kind.AT, Lexer.Kind.LANGLE,
                        Lexer.Kind.RANGLE, Lexer.Kind.EOF},
                kinds("()->{}[],=:@<>"));
    }

    @Test
    void classifiesIntFloatAndNegative() {
        List<Lexer.Token> t = lex("42 -7 -1.5 3.0 1e3");
        assertEquals(Lexer.Kind.INT, t.get(0).kind());
        assertEquals(42L, t.get(0).intValue());
        assertEquals(Lexer.Kind.INT, t.get(1).kind());
        assertEquals(-7L, t.get(1).intValue());
        assertEquals(Lexer.Kind.FLOAT, t.get(2).kind());
        assertEquals(-1.5, t.get(2).floatValue());
        assertEquals(Lexer.Kind.FLOAT, t.get(3).kind());
        assertEquals(Lexer.Kind.FLOAT, t.get(4).kind());
    }

    @Test
    void identifiersIncludeUnderscores() {
        List<Lexer.Token> t = lex("invocation_id local_size round_even");
        assertEquals(Lexer.Kind.IDENT, t.get(0).kind());
        assertEquals("invocation_id", t.get(0).text());
        assertEquals("round_even", t.get(2).text());
    }

    @Test
    void arrowVersusNegativeNumber() {
        // '->' is a single token; '-3' is a negative int.
        assertEquals(Lexer.Kind.ARROW, lex("->").getFirst().kind());
        assertEquals(Lexer.Kind.INT, lex("-3").getFirst().kind());
        assertEquals(-3L, lex("-3").getFirst().intValue());
    }

    @Test
    void skipsLineComments() {
        Lexer.Kind[] k = kinds("; a comment\nx = 1 ; trailing\n");
        assertArrayEquals(
                new Lexer.Kind[]{Lexer.Kind.IDENT, Lexer.Kind.EQUALS, Lexer.Kind.INT, Lexer.Kind.EOF}, k);
    }

    @Test
    void tracksLineAndColumn() {
        List<Lexer.Token> t = lex("a\n   b");
        Lexer.Token b = t.get(1);
        assertEquals(2, b.span().startLine());
        assertEquals(4, b.span().startCol());
    }

    @Test
    void splitsTypeAngleBrackets() {
        // vec3<i32> must tokenize as IDENT '<' IDENT '>' so the parser can read the element type.
        Lexer.Kind[] k = kinds("vec3<i32>");
        assertArrayEquals(
                new Lexer.Kind[]{Lexer.Kind.IDENT, Lexer.Kind.LANGLE, Lexer.Kind.IDENT, Lexer.Kind.RANGLE,
                        Lexer.Kind.EOF},
                k);
    }
}
