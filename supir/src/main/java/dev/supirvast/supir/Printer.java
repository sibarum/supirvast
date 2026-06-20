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
import dev.supirvast.vastir.core.PushConstants;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.Texture;
import dev.supirvast.vastir.core.UnaryOp;
import dev.supirvast.vastir.type.Type;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * Prints a {@code core} {@link CoreModule} back to Supir flat assembly — the write half of the textual-IR
 * pair (see {@link Parser}). Nested {@code Expr} trees are flattened to three-address form: each non-atomic
 * subexpression is hoisted to a freshly named temporary, so the output is faithful to the IR's lowered
 * semantics rather than to the AST's tree shape. Naming is canonical (parameters {@code p0…}, temporaries
 * {@code t0…}), which makes {@code print} a normalizing function: {@code print(parse(print(m)))} equals
 * {@code print(m)}.
 */
final class Printer {

    private final StringBuilder out = new StringBuilder();

    // Per-function state, reset by printBody.
    private final Map<LocalVar, String> localNames = new IdentityHashMap<>();
    private Set<String> used = new LinkedHashSet<>();
    private int tempCounter = 0;
    private int indent = 0;

    static String print(CoreModule module) {
        return new Printer().run(module);
    }

    private String run(CoreModule module) {
        Set<Function> entryFns = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        for (EntryPoint ep : module.entryPoints()) {
            entryFns.add(ep.function());
        }
        boolean first = true;
        // Standalone functions first, so a `call` resolves a name that is already in scope on reparse.
        for (Function fn : module.functions()) {
            if (entryFns.contains(fn)) {
                continue;
            }
            if (!first) {
                out.append('\n');
            }
            first = false;
            printFunction(fn);
        }
        for (EntryPoint ep : module.entryPoints()) {
            if (!first) {
                out.append('\n');
            }
            first = false;
            printEntry(ep);
        }
        return out.toString();
    }

    private void printEntry(EntryPoint ep) {
        Function fn = ep.function();
        ShaderStage stage = ep.stage();
        String header = switch (stage) {
            case VERTEX -> "vertex " + fn.name();
            case FRAGMENT -> "fragment " + fn.name();
            case COMPUTE -> {
                EntryPoint.WorkgroupSize ws = ep.workgroupSize().orElseThrow(
                        () -> new IllegalStateException("compute entry without a workgroup size"));
                yield "compute " + fn.name() + " local_size(" + ws.x() + ", " + ws.y() + ", " + ws.z() + ")";
            }
        };
        line(header + " {");
        printBody(fn, List.of());
        line("}");
    }

    private void printFunction(Function fn) {
        List<Type> params = fn.signature().parameterTypes();
        StringBuilder sig = new StringBuilder("fn ").append(fn.name()).append('(');
        for (int i = 0; i < params.size(); i++) {
            if (i > 0) {
                sig.append(", ");
            }
            sig.append('p').append(i).append(": ").append(typeText(params.get(i)));
        }
        sig.append(") -> ").append(typeText(fn.signature().returnType())).append(" {");
        line(sig.toString());
        printBody(fn, params);
        line("}");
    }

    private void printBody(Function fn, List<Type> params) {
        localNames.clear();
        used = new LinkedHashSet<>();
        tempCounter = 0;
        for (int i = 0; i < params.size(); i++) {
            used.add("p" + i);
        }

        indent++;
        Resources res = new Resources();
        collectRegion(fn.body(), res);
        res.names().forEach(used::add);
        printDeclarations(res);
        printRegion(fn.body());
        indent--;
    }

    // --- declarations ---------------------------------------------------------------------------------------

    private void printDeclarations(Resources res) {
        for (InterfaceVar v : res.inputs()) {
            line("in " + v.name() + ": " + typeText(v.type()) + " @loc " + v.location());
        }
        for (InterfaceVar v : res.outputs()) {
            line("out " + v.name() + ": " + typeText(v.type()) + " @loc " + v.location());
        }
        for (Buffer b : res.buffers()) {
            line("buffer " + b.name() + ": " + typeText(b.element()) + " @binding " + b.binding());
        }
        for (Texture t : res.textures()) {
            if (t.kind() == Texture.Kind.CUBE) {
                line("cubemap " + t.name() + " @binding " + t.binding());
            } else if (t.set() != 0) {
                line("texture " + t.name() + " @set " + t.set() + " @binding " + t.binding());
            } else {
                line("texture " + t.name() + " @binding " + t.binding());
            }
        }
        PushConstants pc = res.pushConstants();
        if (pc != null) {
            if (pc.members().size() == 1) {
                PushConstants.Member m = pc.members().getFirst();
                line("push " + m.name() + ": " + typeText(m.type()));
            } else {
                StringBuilder sb = new StringBuilder("push { ");
                for (int i = 0; i < pc.members().size(); i++) {
                    if (i > 0) {
                        sb.append(", ");
                    }
                    PushConstants.Member m = pc.members().get(i);
                    sb.append(m.name()).append(": ").append(typeText(m.type()));
                }
                line(sb.append(" }").toString());
            }
        }
    }

    // --- statements -----------------------------------------------------------------------------------------

    private void printRegion(Region region) {
        for (Statement s : region.statements()) {
            printStatement(s);
        }
    }

    private void printStatement(Statement s) {
        switch (s) {
            case Statement.ReturnVoid ignored -> line("ret");
            case Statement.Return r -> line("ret " + emitToAtom(r.value()));
            case Statement.StoreResult sr -> line("store_result " + emitToAtom(sr.value()));
            case Statement.BufferStore bs ->
                    line(bs.buffer().name() + "[" + emitToAtom(bs.index()) + "] = " + renderRhs(bs.value()));
            case Statement.BuiltinWrite bw -> line(builtinName(bw.builtin()) + " = " + renderRhs(bw.value()));
            case Statement.InterfaceWrite iw -> line(iw.variable().name() + " = " + renderRhs(iw.value()));
            case Statement.DeclareVar dv -> line(declareName(dv.variable()) + " = " + renderRhs(dv.initializer()));
            case Statement.Assign a -> line(localName(a.variable()) + " = " + renderRhs(a.value()));
            case Statement.If branch -> printIf(branch);
            case Statement.While loop -> printWhile(loop);
        }
    }

    private void printIf(Statement.If branch) {
        line("if " + renderRhs(branch.condition()) + " {");
        indent++;
        printRegion(branch.thenRegion());
        indent--;
        if (branch.elseRegion().statements().isEmpty()) {
            line("}");
        } else {
            line("} else {");
            indent++;
            printRegion(branch.elseRegion());
            indent--;
            line("}");
        }
    }

    private void printWhile(Statement.While loop) {
        // The condition is re-evaluated each iteration, so it must stay inline in the header — never hoisted
        // to a prior temp. Its operands must therefore already be atoms.
        line("loop while " + renderInlineCondition(loop.condition()) + " {");
        indent++;
        printRegion(loop.body());
        indent--;
        line("}");
    }

    // --- expression rendering -------------------------------------------------------------------------------

    /** Renders an expression as an assignment/condition right-hand side: an inline operation, or a bare atom. */
    private String renderRhs(Expr e) {
        return isAtom(e) ? atomText(e) : opText(e, this::emitToAtom);
    }

    /** Renders an operand position: an atom, or (for a nested operation) a hoisted temporary's name. */
    private String emitToAtom(Expr e) {
        if (isAtom(e)) {
            return atomText(e);
        }
        String temp = freshTemp();
        line(temp + " = " + opText(e, this::emitToAtom));
        return temp;
    }

    private String renderInlineCondition(Expr e) {
        return isAtom(e) ? atomText(e) : opText(e, this::shallowAtom);
    }

    private String shallowAtom(Expr e) {
        if (isAtom(e)) {
            return atomText(e);
        }
        throw new IllegalStateException("loop condition is too deeply nested to print as flat assembly "
                + "(operands must be atoms): " + e);
    }

    private static boolean isAtom(Expr e) {
        return switch (e) {
            case Expr.ConstBool ignored -> true;
            case Expr.ConstInt c -> c.type().equals(Type.int32());
            case Expr.ConstFloat c -> c.type().equals(Type.float32());
            case Expr.Read ignored -> true;
            case Expr.Param ignored -> true;
            case Expr.InterfaceRead ignored -> true;
            case Expr.BuiltinRead ignored -> true;
            case Expr.InvocationId ignored -> true;
            case Expr.PushConstantRead ignored -> true;
            default -> false;
        };
    }

    private String atomText(Expr e) {
        return switch (e) {
            case Expr.ConstBool c -> Boolean.toString(c.value());
            case Expr.ConstInt c -> Long.toString(c.value());
            case Expr.ConstFloat c -> floatText(c.value());
            case Expr.Read r -> localName(r.variable());
            case Expr.Param p -> "p" + p.index();
            case Expr.InterfaceRead ir -> ir.variable().name();
            case Expr.BuiltinRead br -> builtinName(br.builtin());
            case Expr.InvocationId ignored -> "invocation_id";
            case Expr.PushConstantRead pc -> pc.block().members().get(pc.member()).name();
            default -> throw new IllegalStateException("not an atom: " + e);
        };
    }

    private String opText(Expr e, java.util.function.Function<Expr, String> arg) {
        return switch (e) {
            case Expr.Binary b -> binaryMnemonic(b.op()) + " " + arg.apply(b.lhs()) + ", " + arg.apply(b.rhs());
            case Expr.Unary u -> unaryMnemonic(u.op()) + " " + arg.apply(u.operand());
            case Expr.Convert c -> "convert " + arg.apply(c.operand()) + ", " + typeText(c.type());
            case Expr.Bitcast c -> "bitcast " + arg.apply(c.operand()) + ", " + typeText(c.type());
            case Expr.VectorConstruct vc -> vectorHead(vc.type()) + " " + joinArgs(vc.components(), arg);
            case Expr.VectorExtract ve -> "extract " + arg.apply(ve.vector()) + ", " + ve.index();
            case Expr.MatrixTimesVector m -> "mtimes " + arg.apply(m.matrix()) + ", " + arg.apply(m.vector());
            case Expr.Call c -> {
                StringBuilder sb = new StringBuilder("call ").append(c.callee().name());
                for (Expr a : c.arguments()) {
                    sb.append(", ").append(arg.apply(a));
                }
                yield sb.toString();
            }
            case Expr.SampleTexture st -> "sample " + st.texture().name() + ", " + arg.apply(st.uv());
            case Expr.MathCall mc -> mathMnemonic(mc.fn()) + " " + joinArgs(mc.args(), arg);
            case Expr.BufferLoad bl -> bl.buffer().name() + "[" + arg.apply(bl.index()) + "]";
            case Expr.ConstInt c -> typeText(c.type()) + " " + c.value();          // non-default-width int const
            case Expr.ConstFloat c -> typeText(c.type()) + " " + floatText(c.value()); // non-f32 float const
            default -> throw new IllegalStateException("cannot render as an operation: " + e);
        };
    }

    private String joinArgs(List<Expr> args, java.util.function.Function<Expr, String> arg) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < args.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(arg.apply(args.get(i)));
        }
        return sb.toString();
    }

    // --- naming ---------------------------------------------------------------------------------------------

    private String declareName(LocalVar var) {
        String base = sanitize(var.name());
        String name = base;
        int n = 1;
        while (used.contains(name)) {
            name = base + "_" + n++;
        }
        used.add(name);
        localNames.put(var, name);
        return name;
    }

    private String localName(LocalVar var) {
        String name = localNames.get(var);
        if (name != null) {
            return name;
        }
        // Defensive: a Read of a variable never declared in this body — name it now so output stays valid.
        return declareName(var);
    }

    private String freshTemp() {
        String name;
        do {
            name = "t" + tempCounter++;
        } while (used.contains(name));
        used.add(name);
        return name;
    }

    private static String sanitize(String name) {
        if (name.isEmpty()) {
            return "v";
        }
        StringBuilder sb = new StringBuilder();
        char c0 = name.charAt(0);
        sb.append(Character.isJavaIdentifierStart(c0) ? c0 : '_');
        for (int i = 1; i < name.length(); i++) {
            char c = name.charAt(i);
            sb.append(Character.isJavaIdentifierPart(c) ? c : '_');
        }
        return sb.toString();
    }

    // --- type / enum / literal text -------------------------------------------------------------------------

    static String typeText(Type type) {
        return switch (type) {
            case Type.Void ignored -> "void";
            case Type.Bool ignored -> "bool";
            case Type.Int i -> (i.signed() ? "i" : "u") + i.width();
            case Type.Float f -> "f" + f.width();
            case Type.Vector v -> vectorTypeText(v);
            case Type.Matrix m -> matrixTypeText(m);
            case Type.FunctionType ignored -> throw new IllegalStateException("function type has no surface form");
        };
    }

    private static String vectorTypeText(Type.Vector v) {
        boolean f32 = v.component().equals(Type.float32());
        if (v.count() >= 2 && v.count() <= 4) {
            return f32 ? "vec" + v.count() : "vec" + v.count() + "<" + typeText(v.component()) + ">";
        }
        return "vec<" + typeText(v.component()) + ", " + v.count() + ">";
    }

    private static String matrixTypeText(Type.Matrix m) {
        boolean squareF32 = m.column().component().equals(Type.float32()) && m.column().count() == m.columns();
        if (squareF32 && m.columns() >= 2 && m.columns() <= 4) {
            return "mat" + m.columns();
        }
        return "mat<" + m.columns() + ", " + vectorTypeText(m.column()) + ">";
    }

    /** The vector-construction mnemonic head, matching what the parser accepts (shorthand or general). */
    private static String vectorHead(Type.Vector v) {
        return vectorTypeText(v);
    }

    private static String floatText(double value) {
        // Double.toString always yields a '.' or exponent, so the token re-lexes as a float (not an int).
        return Double.toString(value);
    }

    private static String builtinName(Builtin b) {
        return switch (b) {
            case POSITION -> "Position";
            case VERTEX_INDEX -> "VertexIndex";
        };
    }

    private static String binaryMnemonic(BinaryOp op) {
        return switch (op) {
            case ADD -> "add";
            case SUB -> "sub";
            case MUL -> "mul";
            case DIV -> "div";
            case MOD -> "mod";
            case BIT_AND -> "band";
            case BIT_OR -> "bor";
            case BIT_XOR -> "bxor";
            case SHIFT_LEFT -> "shl";
            case SHIFT_RIGHT -> "shr";
            case LESS_THAN -> "lt";
            case GREATER_THAN -> "gt";
            case EQUAL -> "eq";
            case LOGICAL_AND -> "land";
            case LOGICAL_OR -> "lor";
        };
    }

    private static String unaryMnemonic(UnaryOp op) {
        return switch (op) {
            case NEGATE -> "neg";
            case NOT -> "bnot";
            case LOGICAL_NOT -> "lnot";
        };
    }

    private static String mathMnemonic(dev.supirvast.vastir.core.MathFn fn) {
        return fn.name().toLowerCase(java.util.Locale.ROOT);
    }

    // --- output ---------------------------------------------------------------------------------------------

    private void line(String text) {
        out.append("  ".repeat(indent)).append(text).append('\n');
    }

    // --- resource collection --------------------------------------------------------------------------------

    /** Resources referenced by a function body, deduped and ordered for stable declaration output. */
    private static final class Resources {
        private final Map<Integer, InterfaceVar> inputs = new TreeMap<>();
        private final Map<Integer, InterfaceVar> outputs = new TreeMap<>();
        private final Map<Integer, Buffer> buffers = new TreeMap<>();
        private final Map<Long, Texture> textures = new TreeMap<>();
        private PushConstants pushConstants;

        void add(InterfaceVar v) {
            (v.direction() == InterfaceVar.Direction.INPUT ? inputs : outputs).putIfAbsent(v.location(), v);
        }

        void add(Buffer b) {
            buffers.putIfAbsent(b.binding(), b);
        }

        void add(Texture t) {
            textures.putIfAbsent(((long) t.set() << 32) | (t.binding() & 0xFFFFFFFFL), t);
        }

        void add(PushConstants pc) {
            pushConstants = pc;
        }

        List<InterfaceVar> inputs() {
            return new ArrayList<>(inputs.values());
        }

        List<InterfaceVar> outputs() {
            return new ArrayList<>(outputs.values());
        }

        List<Buffer> buffers() {
            return new ArrayList<>(buffers.values());
        }

        List<Texture> textures() {
            return new ArrayList<>(textures.values());
        }

        PushConstants pushConstants() {
            return pushConstants;
        }

        List<String> names() {
            List<String> names = new ArrayList<>();
            inputs.values().forEach(v -> names.add(v.name()));
            outputs.values().forEach(v -> names.add(v.name()));
            buffers.values().forEach(b -> names.add(b.name()));
            textures.values().forEach(t -> names.add(t.name()));
            if (pushConstants != null) {
                pushConstants.members().forEach(m -> names.add(m.name()));
            }
            return names;
        }
    }

    private void collectRegion(Region region, Resources res) {
        for (Statement s : region.statements()) {
            collectStatement(s, res);
        }
    }

    private void collectStatement(Statement s, Resources res) {
        switch (s) {
            case Statement.Return r -> collectExpr(r.value(), res);
            case Statement.StoreResult sr -> collectExpr(sr.value(), res);
            case Statement.BufferStore bs -> {
                res.add(bs.buffer());
                collectExpr(bs.index(), res);
                collectExpr(bs.value(), res);
            }
            case Statement.BuiltinWrite bw -> collectExpr(bw.value(), res);
            case Statement.InterfaceWrite iw -> {
                res.add(iw.variable());
                collectExpr(iw.value(), res);
            }
            case Statement.DeclareVar dv -> collectExpr(dv.initializer(), res);
            case Statement.Assign a -> collectExpr(a.value(), res);
            case Statement.If branch -> {
                collectExpr(branch.condition(), res);
                collectRegion(branch.thenRegion(), res);
                collectRegion(branch.elseRegion(), res);
            }
            case Statement.While loop -> {
                collectExpr(loop.condition(), res);
                collectRegion(loop.body(), res);
            }
            case Statement.ReturnVoid ignored -> { /* nothing to collect */ }
        }
    }

    private void collectExpr(Expr e, Resources res) {
        switch (e) {
            case Expr.InterfaceRead ir -> res.add(ir.variable());
            case Expr.PushConstantRead pc -> res.add(pc.block());
            case Expr.BufferLoad bl -> {
                res.add(bl.buffer());
                collectExpr(bl.index(), res);
            }
            case Expr.SampleTexture st -> {
                res.add(st.texture());
                collectExpr(st.uv(), res);
            }
            case Expr.Binary b -> {
                collectExpr(b.lhs(), res);
                collectExpr(b.rhs(), res);
            }
            case Expr.Unary u -> collectExpr(u.operand(), res);
            case Expr.Convert c -> collectExpr(c.operand(), res);
            case Expr.Bitcast c -> collectExpr(c.operand(), res);
            case Expr.VectorConstruct vc -> vc.components().forEach(c -> collectExpr(c, res));
            case Expr.VectorExtract ve -> collectExpr(ve.vector(), res);
            case Expr.MatrixTimesVector m -> {
                collectExpr(m.matrix(), res);
                collectExpr(m.vector(), res);
            }
            case Expr.Call c -> c.arguments().forEach(a -> collectExpr(a, res));
            case Expr.MathCall mc -> mc.args().forEach(a -> collectExpr(a, res));
            default -> { /* leaf atoms reference no resources */ }
        }
    }
}
