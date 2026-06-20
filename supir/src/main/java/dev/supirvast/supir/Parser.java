package dev.supirvast.supir;

import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Builtin;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.MathFn;
import dev.supirvast.vastir.core.PushConstants;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.Texture;
import dev.supirvast.vastir.core.UnaryOp;
import dev.supirvast.vastir.type.Type;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Parses the flat Supir assembly into a {@code core} {@link CoreModule}. The language is three-address: one
 * operation per line, every intermediate named, operands restricted to <em>atoms</em> (names and literals) so
 * there are no nested expression trees. Structured {@code if}/{@code loop} blocks carry the only control flow.
 * Mnemonics have fixed arity, which is what lets the parser stay line-break-insensitive.
 *
 * <p>This is the read half of the textual-IR pair; {@link Printer} is the write half. Errors are reported as
 * {@link SupirParseException} with the offending token's {@link Span}.
 */
final class Parser {

    private static final Map<String, BinaryOp> BINARY = Map.ofEntries(
            Map.entry("add", BinaryOp.ADD), Map.entry("sub", BinaryOp.SUB), Map.entry("mul", BinaryOp.MUL),
            Map.entry("div", BinaryOp.DIV), Map.entry("mod", BinaryOp.MOD),
            Map.entry("band", BinaryOp.BIT_AND), Map.entry("bor", BinaryOp.BIT_OR),
            Map.entry("bxor", BinaryOp.BIT_XOR), Map.entry("shl", BinaryOp.SHIFT_LEFT),
            Map.entry("shr", BinaryOp.SHIFT_RIGHT), Map.entry("lt", BinaryOp.LESS_THAN),
            Map.entry("gt", BinaryOp.GREATER_THAN), Map.entry("eq", BinaryOp.EQUAL),
            Map.entry("land", BinaryOp.LOGICAL_AND), Map.entry("lor", BinaryOp.LOGICAL_OR));

    private static final Map<String, UnaryOp> UNARY = Map.of(
            "neg", UnaryOp.NEGATE, "bnot", UnaryOp.NOT, "lnot", UnaryOp.LOGICAL_NOT);

    /** Math intrinsic mnemonic → (function, operand count). */
    private record MathOp(MathFn fn, int arity) {}

    private static final Map<String, MathOp> MATH = buildMath();

    private final List<Lexer.Token> tokens;
    private int index = 0;
    private final Scope moduleScope = new Scope();
    private final CoreModule module = new CoreModule();

    private Parser(List<Lexer.Token> tokens) {
        this.tokens = tokens;
    }

    static CoreModule parse(String source) {
        return new Parser(new Lexer(source).tokenize()).run();
    }

    private CoreModule run() {
        while (peek().kind() != Lexer.Kind.EOF) {
            topLevel();
        }
        if (module.entryPoints().isEmpty()) {
            throw new SupirParseException(peek().span(), "module declares no entry point "
                    + "(vertex/fragment/compute)");
        }
        return module;
    }

    // --- top level ------------------------------------------------------------------------------------------

    private void topLevel() {
        Lexer.Token t = peek();
        String word = identText(t);
        if (word == null) {
            throw new SupirParseException(t.span(), "expected vertex, fragment, compute, fn, or a "
                    + "resource declaration");
        }
        switch (word) {
            case "vertex", "fragment" -> graphicsEntry(word);
            case "compute" -> computeEntry();
            case "fn" -> {
                Function fn = function();
                moduleScope.defineFunction(fn.name(), fn, t.span());
                module.addFunction(fn);
            }
            case "in", "out", "buffer", "texture", "cubemap", "push" -> declaration(moduleScope);
            default -> throw new SupirParseException(t.span(), "unknown top-level item '" + word + "'");
        }
    }

    private void graphicsEntry(String stageWord) {
        advance(); // stage keyword
        String name = ident("entry name");
        ShaderStage stage = stageWord.equals("vertex") ? ShaderStage.VERTEX : ShaderStage.FRAGMENT;
        Function fn = new Function(name, new Type.FunctionType(Type.VOID, List.of()),
                block(moduleScope.child()));
        module.addEntryPoint(EntryPoint.of(fn, stage));
    }

    private void computeEntry() {
        advance(); // 'compute'
        String name = ident("entry name");
        expectIdent("local_size");
        expect(Lexer.Kind.LPAREN, "(");
        int x = intLiteral("workgroup X");
        expect(Lexer.Kind.COMMA, ",");
        int y = intLiteral("workgroup Y");
        expect(Lexer.Kind.COMMA, ",");
        int z = intLiteral("workgroup Z");
        expect(Lexer.Kind.RPAREN, ")");
        Function fn = new Function(name, new Type.FunctionType(Type.VOID, List.of()), block(moduleScope.child()));
        module.addEntryPoint(EntryPoint.compute(fn, x, y, z));
    }

    private Function function() {
        advance(); // 'fn'
        String name = ident("function name");
        Scope fnScope = moduleScope.child();
        expect(Lexer.Kind.LPAREN, "(");
        List<Type> paramTypes = new ArrayList<>();
        if (peek().kind() != Lexer.Kind.RPAREN) {
            do {
                Lexer.Token pTok = peek();
                String pName = ident("parameter name");
                expect(Lexer.Kind.COLON, ":");
                Type pType = type();
                fnScope.defineParam(pName, new Expr.Param(paramTypes.size(), pType), pTok.span());
                paramTypes.add(pType);
            } while (accept(Lexer.Kind.COMMA));
        }
        expect(Lexer.Kind.RPAREN, ")");
        expect(Lexer.Kind.ARROW, "->");
        Type returnType = type();
        Region body = block(fnScope);
        return new Function(name, new Type.FunctionType(returnType, paramTypes), body);
    }

    // --- blocks, declarations, statements -------------------------------------------------------------------

    private Region block(Scope scope) {
        expect(Lexer.Kind.LBRACE, "{");
        List<Statement> statements = new ArrayList<>();
        while (peek().kind() != Lexer.Kind.RBRACE) {
            if (peek().kind() == Lexer.Kind.EOF) {
                throw new SupirParseException(peek().span(), "unbalanced '{' — missing '}'");
            }
            String word = identText(peek());
            if (word != null && isDecl(word)) {
                declaration(scope);
            } else {
                statements.add(statement(scope));
            }
        }
        expect(Lexer.Kind.RBRACE, "}");
        return new Region(statements);
    }

    private static boolean isDecl(String word) {
        return switch (word) {
            case "in", "out", "buffer", "texture", "cubemap", "push" -> true;
            default -> false;
        };
    }

    private void declaration(Scope scope) {
        Lexer.Token start = peek();
        String word = identText(start);
        advance();
        switch (word) {
            case "in", "out" -> {
                String name = ident("interface name");
                expect(Lexer.Kind.COLON, ":");
                Type type = type();
                int location = at("loc");
                InterfaceVar var = word.equals("in")
                        ? InterfaceVar.input(name, location, type)
                        : InterfaceVar.output(name, location, type);
                scope.defineInterface(name, var, start.span());
            }
            case "buffer" -> {
                String name = ident("buffer name");
                expect(Lexer.Kind.COLON, ":");
                Type element = type();
                int binding = at("binding");
                scope.defineBuffer(name, new Buffer(name, binding, element), start.span());
            }
            case "texture" -> {
                String name = ident("texture name");
                // optional @set N, then @binding N
                int set = 0;
                int binding;
                expect(Lexer.Kind.AT, "@");
                String first = ident("set or binding");
                if (first.equals("set")) {
                    set = intLiteral("descriptor set");
                    binding = at("binding");
                } else if (first.equals("binding")) {
                    binding = intLiteral("binding");
                } else {
                    throw new SupirParseException(start.span(), "expected @set or @binding");
                }
                scope.defineTexture(name, new Texture(name, set, binding), start.span());
            }
            case "cubemap" -> {
                String name = ident("cubemap name");
                int binding = at("binding");
                scope.defineTexture(name, Texture.cube(name, binding), start.span());
            }
            case "push" -> pushConstants(scope, start);
            default -> throw new SupirParseException(start.span(), "not a declaration: " + word);
        }
    }

    private void pushConstants(Scope scope, Lexer.Token start) {
        List<PushConstants.Member> members = new ArrayList<>();
        if (accept(Lexer.Kind.LBRACE)) {
            if (peek().kind() != Lexer.Kind.RBRACE) {
                do {
                    members.add(pushMember());
                } while (accept(Lexer.Kind.COMMA));
            }
            expect(Lexer.Kind.RBRACE, "}");
        } else {
            members.add(pushMember()); // shorthand: push NAME : TYPE
        }
        scope.definePushConstants(new PushConstants(members), start.span());
    }

    private PushConstants.Member pushMember() {
        String name = ident("push-constant member name");
        expect(Lexer.Kind.COLON, ":");
        return new PushConstants.Member(name, type());
    }

    private Statement statement(Scope scope) {
        Lexer.Token t = peek();
        String word = identText(t);
        if (word == null) {
            throw new SupirParseException(t.span(), "expected a statement");
        }
        return switch (word) {
            case "ret" -> {
                advance();
                if (startsAtom()) {
                    yield new Statement.Return(atom(scope));
                }
                yield new Statement.ReturnVoid();
            }
            case "store_result" -> {
                advance();
                yield new Statement.StoreResult(atom(scope));
            }
            case "if" -> ifStatement(scope);
            case "loop" -> loopStatement(scope);
            default -> assignment(scope);
        };
    }

    /** {@code Position = …} / {@code outVar = …} / {@code buf[i] = …} / {@code name[: type] = …}. */
    private Statement assignment(Scope scope) {
        Lexer.Token lhs = peek();
        String name = ident("a statement");

        // buffer store: name[index] = value
        if (accept(Lexer.Kind.LBRACKET)) {
            Buffer buffer = scope.buffer(name);
            if (buffer == null) {
                throw new SupirParseException(lhs.span(), "undefined buffer '" + name + "'");
            }
            Expr idx = atom(scope);
            expect(Lexer.Kind.RBRACKET, "]");
            expect(Lexer.Kind.EQUALS, "=");
            return new Statement.BufferStore(buffer, idx, rhs(scope));
        }

        // typed local declaration: name : type = value
        if (accept(Lexer.Kind.COLON)) {
            Type declared = type();
            expect(Lexer.Kind.EQUALS, "=");
            Expr value = rhs(scope);
            LocalVar var = new LocalVar(name, declared);
            scope.defineLocal(name, var, lhs.span());
            return new Statement.DeclareVar(var, value);
        }

        expect(Lexer.Kind.EQUALS, "=");

        Builtin builtin = builtinOrNull(name);
        if (builtin != null) {
            return new Statement.BuiltinWrite(builtin, rhs(scope));
        }
        InterfaceVar iface = scope.iface(name);
        if (iface != null) {
            if (iface.direction() == InterfaceVar.Direction.INPUT) {
                throw new SupirParseException(lhs.span(), "cannot write to input '" + name + "'");
            }
            return new Statement.InterfaceWrite(iface, rhs(scope));
        }
        // local: reassignment if already declared, otherwise a fresh declaration (type inferred from value)
        Expr value = rhs(scope);
        LocalVar existing = scope.local(name);
        if (existing != null) {
            return new Statement.Assign(existing, value);
        }
        LocalVar var = new LocalVar(name, value.type());
        scope.defineLocal(name, var, lhs.span());
        return new Statement.DeclareVar(var, value);
    }

    private Statement ifStatement(Scope scope) {
        advance(); // 'if'
        Expr cond = condition(scope);
        Region thenRegion = block(scope.child());
        Region elseRegion = new Region(List.of());
        if (identText(peek()) != null && identText(peek()).equals("else")) {
            advance();
            elseRegion = block(scope.child());
        }
        return new Statement.If(cond, thenRegion, elseRegion);
    }

    private Statement loopStatement(Scope scope) {
        advance(); // 'loop'
        expectIdent("while");
        Expr cond = condition(scope);
        Region body = block(scope.child());
        return new Statement.While(cond, body);
    }

    /**
     * An {@code if}/{@code while} condition: a single inline atom or operation with atom operands. Kept inline
     * (not hoisted to a prior temp) so a loop condition is re-evaluated each iteration, matching
     * {@link Statement.While}'s single-expression semantics.
     */
    private Expr condition(Scope scope) {
        return rhs(scope);
    }

    // --- right-hand sides: an operation, or a bare atom -----------------------------------------------------

    private Expr rhs(Scope scope) {
        Lexer.Token t = peek();
        String word = identText(t);
        if (word != null && isOperation(word)) {
            return operation(scope);
        }
        return atom(scope);
    }

    private boolean isOperation(String word) {
        return BINARY.containsKey(word) || UNARY.containsKey(word) || MATH.containsKey(word)
                || scalarType(word) != null
                || switch (word) {
                    case "convert", "bitcast", "extract", "mtimes", "sample", "call",
                         "vec", "vec2", "vec3", "vec4", "pc" -> true;
                    default -> false;
                };
    }

    private Expr operation(Scope scope) {
        Lexer.Token t = peek();
        String op = identText(t);
        advance();

        BinaryOp binary = BINARY.get(op);
        if (binary != null) {
            Expr a = atom(scope);
            expect(Lexer.Kind.COMMA, ",");
            Expr b = atom(scope);
            return new Expr.Binary(binary, a, b);
        }
        UnaryOp unary = UNARY.get(op);
        if (unary != null) {
            return new Expr.Unary(unary, atom(scope));
        }
        MathOp math = MATH.get(op);
        if (math != null) {
            List<Expr> args = operands(scope, math.arity());
            return mathCall(math.fn(), args, t.span());
        }
        Type scalar = scalarType(op);
        if (scalar != null) {
            // polymorphic: `f32 1.0` is a constant, `f32 x` is a conversion
            if (startsLiteral()) {
                return scalarConstant(scalar);
            }
            return new Expr.Convert(atom(scope), scalar);
        }
        return switch (op) {
            case "convert" -> {
                Expr a = atom(scope);
                expect(Lexer.Kind.COMMA, ",");
                yield new Expr.Convert(a, type());
            }
            case "bitcast" -> {
                Expr a = atom(scope);
                expect(Lexer.Kind.COMMA, ",");
                yield new Expr.Bitcast(a, type());
            }
            case "extract" -> {
                Expr v = atom(scope);
                expect(Lexer.Kind.COMMA, ",");
                yield new Expr.VectorExtract(v, intLiteral("component index"));
            }
            case "mtimes" -> {
                Expr m = atom(scope);
                expect(Lexer.Kind.COMMA, ",");
                yield new Expr.MatrixTimesVector(m, atom(scope));
            }
            case "sample" -> {
                Lexer.Token texTok = peek();
                String texName = ident("texture name");
                Texture texture = scope.texture(texName);
                if (texture == null) {
                    throw new SupirParseException(texTok.span(), "undefined texture '" + texName + "'");
                }
                expect(Lexer.Kind.COMMA, ",");
                yield new Expr.SampleTexture(texture, atom(scope));
            }
            case "call" -> call(scope);
            case "vec" -> vecGeneral(scope);
            case "vec2" -> vecShorthand(scope, 2);
            case "vec3" -> vecShorthand(scope, 3);
            case "vec4" -> vecShorthand(scope, 4);
            case "pc" -> pushConstantRead(scope);
            default -> throw new SupirParseException(t.span(), "unknown operation '" + op + "'");
        };
    }

    private Expr call(Scope scope) {
        Lexer.Token nameTok = peek();
        String name = ident("function name");
        Function callee = scope.function(name);
        if (callee == null) {
            throw new SupirParseException(nameTok.span(), "call to undefined function '" + name + "'");
        }
        int arity = callee.signature().parameterTypes().size();
        List<Expr> args = new ArrayList<>();
        for (int i = 0; i < arity; i++) {
            expect(Lexer.Kind.COMMA, ",");
            args.add(atom(scope));
        }
        return new Expr.Call(callee, args);
    }

    private Expr vecGeneral(Scope scope) {
        expect(Lexer.Kind.LANGLE, "<");
        Type element = type();
        expect(Lexer.Kind.COMMA, ",");
        int count = intLiteral("component count");
        expect(Lexer.Kind.RANGLE, ">");
        return new Expr.VectorConstruct(new Type.Vector(element, count), operands(scope, count));
    }

    private Expr vecShorthand(Scope scope, int count) {
        Type element = Type.float32();
        if (accept(Lexer.Kind.LANGLE)) {
            element = type();
            expect(Lexer.Kind.RANGLE, ">");
        }
        return new Expr.VectorConstruct(new Type.Vector(element, count), operands(scope, count));
    }

    private Expr pushConstantRead(Scope scope) {
        Lexer.Token tok = peek();
        String member = ident("push-constant member");
        int idx = scope.pushConstantMember(member);
        if (idx < 0) {
            throw new SupirParseException(tok.span(), "unknown push-constant member '" + member + "'");
        }
        return new Expr.PushConstantRead(scope.pushConstants(), idx);
    }

    /** Reads {@code count} comma-separated atom operands. */
    private List<Expr> operands(Scope scope, int count) {
        List<Expr> args = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            if (i > 0) {
                expect(Lexer.Kind.COMMA, ",");
            }
            args.add(atom(scope));
        }
        return args;
    }

    // --- atoms ----------------------------------------------------------------------------------------------

    private boolean startsAtom() {
        return switch (peek().kind()) {
            case INT, FLOAT, IDENT -> true;
            default -> false;
        };
    }

    private boolean startsLiteral() {
        return peek().kind() == Lexer.Kind.INT || peek().kind() == Lexer.Kind.FLOAT;
    }

    private Expr atom(Scope scope) {
        Lexer.Token t = peek();
        switch (t.kind()) {
            case INT -> {
                advance();
                return new Expr.ConstInt(Type.int32(), t.intValue());
            }
            case FLOAT -> {
                advance();
                return new Expr.ConstFloat(Type.float32(), t.floatValue());
            }
            case IDENT -> {
                return nameAtom(scope, t);
            }
            default -> throw new SupirParseException(t.span(), "expected a value");
        }
    }

    private Expr nameAtom(Scope scope, Lexer.Token t) {
        String name = t.text();
        advance();

        // buffer load: name[index]
        if (peek().kind() == Lexer.Kind.LBRACKET) {
            Buffer buffer = scope.buffer(name);
            if (buffer == null) {
                throw new SupirParseException(t.span(), "undefined buffer '" + name + "'");
            }
            advance(); // '['
            Expr idx = atom(scope);
            expect(Lexer.Kind.RBRACKET, "]");
            return new Expr.BufferLoad(buffer, idx);
        }

        switch (name) {
            case "true" -> { return new Expr.ConstBool(true); }
            case "false" -> { return new Expr.ConstBool(false); }
            case "invocation_id" -> { return new Expr.InvocationId(); }
            default -> { /* fall through to lookups */ }
        }
        Builtin builtin = builtinOrNull(name);
        if (builtin != null) {
            return new Expr.BuiltinRead(builtin);
        }
        Expr.Param param = scope.param(name);
        if (param != null) {
            return param;
        }
        LocalVar local = scope.local(name);
        if (local != null) {
            return new Expr.Read(local);
        }
        InterfaceVar iface = scope.iface(name);
        if (iface != null) {
            return new Expr.InterfaceRead(iface);
        }
        int pcMember = scope.pushConstantMember(name);
        if (pcMember >= 0) {
            return new Expr.PushConstantRead(scope.pushConstants(), pcMember);
        }
        throw new SupirParseException(t.span(), "undefined name '" + name + "'");
    }

    // --- types ----------------------------------------------------------------------------------------------

    private Type type() {
        Lexer.Token t = peek();
        String word = identText(t);
        if (word == null) {
            throw new SupirParseException(t.span(), "expected a type");
        }
        advance();
        Type scalar = scalarType(word);
        if (scalar != null) {
            return scalar;
        }
        return switch (word) {
            case "void" -> Type.VOID;
            case "bool" -> Type.BOOL;
            case "vec2" -> vectorType(2);
            case "vec3" -> vectorType(3);
            case "vec4" -> vectorType(4);
            case "vec" -> {
                expect(Lexer.Kind.LANGLE, "<");
                Type element = type();
                expect(Lexer.Kind.COMMA, ",");
                int count = intLiteral("vector count");
                expect(Lexer.Kind.RANGLE, ">");
                yield new Type.Vector(element, count);
            }
            case "mat2" -> squareMatrix(2);
            case "mat3" -> squareMatrix(3);
            case "mat4" -> squareMatrix(4);
            case "mat" -> {
                expect(Lexer.Kind.LANGLE, "<");
                int columns = intLiteral("matrix columns");
                expect(Lexer.Kind.COMMA, ",");
                Type column = type();
                expect(Lexer.Kind.RANGLE, ">");
                if (!(column instanceof Type.Vector v)) {
                    throw new SupirParseException(t.span(), "matrix column type must be a vector");
                }
                yield new Type.Matrix(v, columns);
            }
            default -> throw new SupirParseException(t.span(), "unknown type '" + word + "'");
        };
    }

    /** A {@code vecN} shorthand whose element type is f32 unless an explicit {@code <scalar>} follows. */
    private Type vectorType(int count) {
        Type element = Type.float32();
        if (accept(Lexer.Kind.LANGLE)) {
            element = type();
            expect(Lexer.Kind.RANGLE, ">");
        }
        return new Type.Vector(element, count);
    }

    private Type squareMatrix(int n) {
        return new Type.Matrix(new Type.Vector(Type.float32(), n), n);
    }

    private static Type scalarType(String name) {
        return switch (name) {
            case "i8" -> new Type.Int(8, true);
            case "i16" -> new Type.Int(16, true);
            case "i32" -> Type.int32();
            case "i64" -> Type.int64();
            case "u8" -> new Type.Int(8, false);
            case "u16" -> new Type.Int(16, false);
            case "u32" -> Type.uint32();
            case "u64" -> Type.uint64();
            case "f16" -> new Type.Float(16);
            case "f32" -> Type.float32();
            case "f64" -> Type.float64();
            default -> null;
        };
    }

    private Expr scalarConstant(Type scalar) {
        Lexer.Token t = peek();
        if (scalar instanceof Type.Int intType) {
            if (t.kind() != Lexer.Kind.INT) {
                throw new SupirParseException(t.span(), "integer literal expected for integer type");
            }
            advance();
            return new Expr.ConstInt(intType, t.intValue());
        }
        Type.Float floatType = (Type.Float) scalar;
        double value = t.kind() == Lexer.Kind.INT ? (double) t.intValue() : t.floatValue();
        advance();
        return new Expr.ConstFloat(floatType, value);
    }

    private static Builtin builtinOrNull(String name) {
        return switch (name) {
            case "Position" -> Builtin.POSITION;
            case "VertexIndex" -> Builtin.VERTEX_INDEX;
            default -> null;
        };
    }

    private Expr mathCall(MathFn fn, List<Expr> args, Span span) {
        Type result = switch (fn) {
            case DOT, LENGTH, DISTANCE -> {
                if (!(args.getFirst().type() instanceof Type.Vector v)) {
                    throw new SupirParseException(span, fn + " expects a vector argument");
                }
                yield v.component();
            }
            default -> args.getFirst().type();
        };
        return new Expr.MathCall(fn, result, args);
    }

    // --- token helpers --------------------------------------------------------------------------------------

    /** Parses {@code @NAME N} and returns N, requiring the tag word to equal {@code expectedTag}. */
    private int at(String expectedTag) {
        expect(Lexer.Kind.AT, "@");
        Lexer.Token tagTok = peek();
        String tag = ident(expectedTag);
        if (!tag.equals(expectedTag)) {
            throw new SupirParseException(tagTok.span(), "expected @" + expectedTag + ", found @" + tag);
        }
        return intLiteral(expectedTag);
    }

    private Lexer.Token peek() {
        return tokens.get(index);
    }

    private void advance() {
        if (index < tokens.size() - 1) {
            index++;
        }
    }

    private boolean accept(Lexer.Kind kind) {
        if (peek().kind() == kind) {
            advance();
            return true;
        }
        return false;
    }

    private void expect(Lexer.Kind kind, String what) {
        if (peek().kind() != kind) {
            throw new SupirParseException(peek().span(), "expected '" + what + "'");
        }
        advance();
    }

    private void expectIdent(String word) {
        Lexer.Token t = peek();
        if (!word.equals(identText(t))) {
            throw new SupirParseException(t.span(), "expected '" + word + "'");
        }
        advance();
    }

    private String ident(String what) {
        Lexer.Token t = peek();
        if (t.kind() != Lexer.Kind.IDENT) {
            throw new SupirParseException(t.span(), "expected " + what);
        }
        advance();
        return t.text();
    }

    private int intLiteral(String what) {
        Lexer.Token t = peek();
        if (t.kind() != Lexer.Kind.INT) {
            throw new SupirParseException(t.span(), "expected an integer " + what);
        }
        advance();
        return Math.toIntExact(t.intValue());
    }

    private static String identText(Lexer.Token t) {
        return t.kind() == Lexer.Kind.IDENT ? t.text() : null;
    }

    private static Map<String, MathOp> buildMath() {
        Map<String, MathOp> m = new java.util.HashMap<>();
        // arity 1
        for (String name : List.of("length", "normalize", "abs", "sign", "sqrt", "inverse_sqrt", "exp", "log",
                "exp2", "log2", "sin", "cos", "tan", "asin", "acos", "atan", "radians", "degrees", "sinh",
                "cosh", "tanh", "asinh", "acosh", "atanh", "round", "round_even", "trunc", "floor", "ceil",
                "fract")) {
            m.put(name, new MathOp(mathFn(name), 1));
        }
        // arity 2
        for (String name : List.of("dot", "distance", "cross", "reflect", "pow", "min", "max", "atan2",
                "step")) {
            m.put(name, new MathOp(mathFn(name), 2));
        }
        // arity 3
        for (String name : List.of("clamp", "mix", "smoothstep", "fma", "face_forward", "refract")) {
            m.put(name, new MathOp(mathFn(name), 3));
        }
        return Map.copyOf(m);
    }

    private static MathFn mathFn(String mnemonic) {
        return MathFn.valueOf(mnemonic.toUpperCase(java.util.Locale.ROOT));
    }
}
