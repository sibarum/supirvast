package dev.supirvast.vastir.build;

import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.MathFn;
import dev.supirvast.vastir.core.PushConstants;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.SharedArray;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.SubgroupOp;
import dev.supirvast.vastir.core.UnaryOp;
import dev.supirvast.vastir.type.Type;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * A kernel body being written by hand: a list of statements, and a terse way to say the expressions in them.
 *
 * <p>Deliberately no more than that. Kernels are written the way a person would write them, so that anything that
 * generates them later has a checked shape to reproduce. It grows only by what a kernel actually needs; an
 * abstraction no kernel has asked for would be a generator arriving early.
 *
 * <pre>{@code
 * Body b = new Body();
 * LocalVar i = b.let("i", new Expr.InvocationId());
 * b.when(lt(v(i), new Expr.InvocationCount()), t -> t.store(out, v(i), add(load(a, v(i)), load(c, v(i)))));
 * Function kernel = new Function("add", new Type.FunctionType(Type.VOID, List.of()), b.finish());
 * }</pre>
 *
 * <p>Written for vexelray-sim-fluid's kernels, moved to vexelray-sim-core when a second simulation needed it,
 * and here when it was plain that nothing about it is a simulation's.
 */
public final class Body {

    public static final Type.Int I32 = Type.int32();
    public static final Type.Float F32 = Type.float32();

    private final List<Statement> statements = new ArrayList<>();
    private final Names names;

    public Body() {
        this(new Names());
    }

    private Body(Names names) {
        this.names = names;
    }

    /** Declares a new local initialised to {@code value}; the name only has to mean something to a reader. */
    public LocalVar let(String name, Expr value) {
        LocalVar variable = new LocalVar(names.fresh(name), value.type());
        statements.add(new Statement.DeclareVar(variable, value));
        return variable;
    }

    public void set(LocalVar variable, Expr value) {
        statements.add(new Statement.Assign(variable, value));
    }

    public void store(Buffer buffer, Expr index, Expr value) {
        statements.add(new Statement.BufferStore(buffer, index, value));
    }

    /** {@code buffer[index] = op(buffer[index], value)}, indivisibly, discarding the old value. */
    public void atomic(AtomicOp op, Buffer buffer, Expr index, Expr value) {
        statements.add(new Statement.AtomicUpdate(op, buffer, index, value));
    }

    /** {@link #atomic}, keeping the element's value from before the update in a new local. */
    public LocalVar fetchAtomic(String name, AtomicOp op, Buffer buffer, Expr index, Expr value) {
        LocalVar previous = new LocalVar(names.fresh(name), buffer.element());
        statements.add(new Statement.AtomicUpdate(previous, op, buffer, index, value));
        return previous;
    }

    /**
     * {@code value} as another lane of the subgroup holds it — {@code lane} lanes up or down, or lane
     * {@code lane} itself, by {@code kind}. Like a barrier, only at the top level of a kernel.
     */
    public LocalVar shuffle(String name, Statement.SubgroupShuffle.Kind kind, Expr value, Expr lane) {
        LocalVar result = new LocalVar(names.fresh(name), value.type());
        statements.add(new Statement.SubgroupShuffle(result, kind, value, lane));
        return result;
    }

    /** {@code op} over the subgroup's lanes — the whole of it, or each lane's running scan. Top level only. */
    public LocalVar subgroup(String name, SubgroupOp op, Statement.SubgroupArithmetic.Scan scan, Expr value) {
        LocalVar result = new LocalVar(names.fresh(name), value.type());
        statements.add(new Statement.SubgroupArithmetic(result, op, scan, value));
        return result;
    }

    /** Writes an element of workgroup memory. */
    public void sharedStore(SharedArray array, Expr index, Expr value) {
        statements.add(new Statement.SharedStore(array, index, value));
    }

    /** {@code array[index] = op(array[index], value)}, indivisibly within the workgroup. */
    public void sharedAtomic(AtomicOp op, SharedArray array, Expr index, Expr value) {
        statements.add(new Statement.SharedAtomicUpdate(op, array, index, value));
    }

    /**
     * Waits for the whole workgroup, and makes every write before it visible after it. Only at the top level of
     * a kernel, never in a branch: every invocation must reach it.
     */
    public void barrier() {
        statements.add(new Statement.Barrier());
    }

    /** Writes a stage output, such as a fragment's colour. */
    public void write(InterfaceVar output, Expr value) {
        statements.add(new Statement.InterfaceWrite(output, value));
    }

    /** {@code if (condition) { then }} — the statements {@code then} writes go into the branch. */
    public void when(Expr condition, Consumer<Body> then) {
        Body branch = new Body(names);
        then.accept(branch);
        statements.add(new Statement.If(condition, branch.region(), Region.of()));
    }

    /** {@code while (condition) { body }} — the condition is evaluated before every pass. */
    public void loop(Expr condition, Consumer<Body> body) {
        Body pass = new Body(names);
        body.accept(pass);
        statements.add(new Statement.While(condition, pass.region()));
    }

    public Region region() {
        return new Region(List.copyOf(statements));
    }

    /** Ends a {@code void} kernel. */
    public Region finish() {
        statements.add(new Statement.ReturnVoid());
        return region();
    }

    // --- expressions -------------------------------------------------------------------------------------

    public static Expr v(LocalVar variable) {
        return new Expr.Read(variable);
    }

    public static Expr i(int value) {
        return new Expr.ConstInt(I32, value);
    }

    public static Expr f(double value) {
        return new Expr.ConstFloat(F32, value);
    }

    /** An f32 truncated toward zero to an i32. */
    public static Expr toInt(Expr a) {
        return new Expr.Convert(a, I32);
    }

    /** An integer, to the nearest f32. */
    public static Expr toFloat(Expr a) {
        return new Expr.Convert(a, F32);
    }

    public static Expr load(Buffer buffer, Expr index) {
        return new Expr.BufferLoad(buffer, index);
    }

    public static Expr sharedLoad(SharedArray array, Expr index) {
        return new Expr.SharedLoad(array, index);
    }

    public static Expr add(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.ADD, a, b);
    }

    public static Expr sub(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.SUB, a, b);
    }

    public static Expr mul(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.MUL, a, b);
    }

    public static Expr div(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.DIV, a, b);
    }

    public static Expr mod(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.MOD, a, b);
    }

    public static Expr lt(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.LESS_THAN, a, b);
    }

    public static Expr gt(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.GREATER_THAN, a, b);
    }

    public static Expr eq(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.EQUAL, a, b);
    }

    public static Expr not(Expr a) {
        return new Expr.Unary(UnaryOp.LOGICAL_NOT, a);
    }

    /** Both, evaluating both: {@code OpLogicalAnd} does not short-circuit, so neither side may depend on the other. */
    public static Expr and(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.LOGICAL_AND, a, b);
    }

    /** Either, evaluating both, as {@link #and}. */
    public static Expr or(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.LOGICAL_OR, a, b);
    }

    public static Expr neg(Expr a) {
        return new Expr.Unary(UnaryOp.NEGATE, a);
    }

    public static Expr min(Expr a, Expr b) {
        return new Expr.MathCall(MathFn.MIN, a.type(), List.of(a, b));
    }

    public static Expr max(Expr a, Expr b) {
        return new Expr.MathCall(MathFn.MAX, a.type(), List.of(a, b));
    }

    public static Expr sqrt(Expr a) {
        return new Expr.MathCall(MathFn.SQRT, a.type(), List.of(a));
    }

    public static Expr abs(Expr a) {
        return new Expr.MathCall(MathFn.ABS, a.type(), List.of(a));
    }

    /** An f32's bits as an i32 — which, for a non-negative float, orders the same way the float does. */
    public static Expr bitsOf(Expr a) {
        return new Expr.Bitcast(a, I32);
    }

    /** The f32 whose bits an i32 holds. */
    public static Expr floatOf(Expr a) {
        return new Expr.Bitcast(a, F32);
    }

    // --- for graphics stages -----------------------------------------------------------------------------

    public static final Type.Vector VEC2 = new Type.Vector(F32, 2);
    public static final Type.Vector VEC3 = new Type.Vector(F32, 3);
    public static final Type.Vector VEC4 = new Type.Vector(F32, 4);

    /** A stage input, such as the varying a vertex stage wrote. */
    public static Expr input(InterfaceVar variable) {
        return new Expr.InterfaceRead(variable);
    }

    /** Member {@code index} of a push-constant block. */
    public static Expr pushed(PushConstants block, int index) {
        return block.read(index);
    }

    public static Expr vec3(Expr x, Expr y, Expr z) {
        return new Expr.VectorConstruct(VEC3, List.of(x, y, z));
    }

    public static Expr vec3(double x, double y, double z) {
        return vec3(f(x), f(y), f(z));
    }

    /** A scalar in every lane: the IR does not broadcast, so a scalar times a vector is spelled this way. */
    public static Expr splat3(Expr scalar) {
        return vec3(scalar, scalar, scalar);
    }

    public static Expr vec4(Expr rgb, Expr alpha) {
        return new Expr.VectorConstruct(VEC4, List.of(component(rgb, 0), component(rgb, 1), component(rgb, 2), alpha));
    }

    public static Expr component(Expr vector, int index) {
        return new Expr.VectorExtract(vector, index);
    }

    public static Expr clamp(Expr value, Expr low, Expr high) {
        return new Expr.MathCall(MathFn.CLAMP, value.type(), List.of(value, low, high));
    }

    public static Expr floor(Expr a) {
        return new Expr.MathCall(MathFn.FLOOR, a.type(), List.of(a));
    }

    /** {@code a + (b - a) · t}, componentwise. */
    public static Expr mix(Expr a, Expr b, Expr t) {
        return new Expr.MathCall(MathFn.MIX, a.type(), List.of(a, b, t));
    }

    /** Unique names across a body and its branches, so a printed kernel reads unambiguously. */
    private static final class Names {
        private int next;

        String fresh(String base) {
            return base + "_" + next++;
        }
    }
}
