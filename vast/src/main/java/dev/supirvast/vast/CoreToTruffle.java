package dev.supirvast.vast;

import com.oracle.truffle.api.CallTarget;
import com.oracle.truffle.api.TruffleLanguage;
import com.oracle.truffle.api.frame.FrameDescriptor;
import com.oracle.truffle.api.frame.FrameSlotKind;
import com.oracle.truffle.api.frame.VirtualFrame;
import com.oracle.truffle.api.nodes.ControlFlowException;
import com.oracle.truffle.api.nodes.Node;
import com.oracle.truffle.api.nodes.RootNode;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.UnaryOp;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Generates an executable Truffle AST from the core IR — the project's second backend.
 *
 * <p>The same {@link Function} that {@code CoreToSpirv} lowers to GPU SPIR-V is here turned into a
 * {@link CallTarget} that runs on the CPU (and is JIT-compiled by Graal on a GraalVM runtime). The core
 * level's structured regions and value-based expressions map directly onto a tree of {@link Node}s — exactly
 * the tree shape Truffle wants, which is why generating from the high level is natural.
 *
 * <p>Local variables live in {@link VirtualFrame} slots; structured {@code if}/{@code while} become nested
 * node trees rather than a reconstructed CFG. Values are boxed ({@link Integer}/{@link Boolean}/{@link Double})
 * for a faithful first interpreter; node specialization is a later optimization.
 */
public final class CoreToTruffle {

    /** Lowering context: local-variable frame slots, buffer slots (kernels), and callee targets (calls). */
    private record Ctx(Map<LocalVar, Integer> slots, Map<Integer, Integer> bufferSlots,
            Map<Function, CallTarget> targets) {
    }

    /** Lowers a core function to a callable Truffle target. Calling it executes the function on the CPU. */
    public CallTarget lower(Function function) {
        return lowerModule(List.of(function), function);
    }

    /**
     * Lowers a multi-function module, returning the {@code entry}'s target. Each function becomes its own
     * call target sharing one map, so {@code Expr.Call} resolves callees (looked up lazily at execution, so
     * declaration order is irrelevant). Functions read their parameters via {@code Expr.Param}.
     */
    public CallTarget lowerModule(List<Function> functions, Function entry) {
        Map<Function, CallTarget> targets = new IdentityHashMap<>();
        for (Function function : functions) {
            targets.put(function, buildTarget(function, Map.of(), targets));
        }
        return targets.get(entry);
    }

    /**
     * Lowers a data-parallel kernel. The returned target is called once per invocation as
     * {@code call(Integer invocationIndex, int[][] buffers)}, where {@code buffers} is indexed by the
     * position of each {@link Buffer} in {@code buffers} (its slot).
     */
    public CallTarget lowerKernel(Function function, List<Buffer> buffers) {
        Map<Integer, Integer> bufferSlots = new java.util.HashMap<>();
        for (int i = 0; i < buffers.size(); i++) {
            bufferSlots.put(buffers.get(i).binding(), i);
        }
        return buildTarget(function, bufferSlots, Map.of());
    }

    private CallTarget buildTarget(Function function, Map<Integer, Integer> bufferSlots,
            Map<Function, CallTarget> targets) {
        FrameDescriptor.Builder frame = FrameDescriptor.newBuilder();
        Map<LocalVar, Integer> slots = new IdentityHashMap<>();
        for (LocalVar variable : collectVariables(function.body())) {
            slots.put(variable, frame.addSlot(FrameSlotKind.Object, variable.name(), null));
        }
        StatementNode[] body = lowerRegion(function.body(), new Ctx(slots, bufferSlots, targets));
        return new ShaderRootNode(frame.build(), body).getCallTarget();
    }

    private StatementNode[] lowerRegion(Region region, Ctx ctx) {
        List<StatementNode> nodes = new ArrayList<>();
        for (Statement statement : region.statements()) {
            nodes.add(lowerStatement(statement, ctx));
        }
        return nodes.toArray(StatementNode[]::new);
    }

    private StatementNode lowerStatement(Statement statement, Ctx ctx) {
        return switch (statement) {
            case Statement.ReturnVoid ignored -> new ReturnNode(null);
            case Statement.Return r -> new ReturnNode(lowerExpr(r.value(), ctx));
            case Statement.StoreResult ignored -> throw new UnsupportedOperationException(
                    "StoreResult is GPU-only; the CPU backend observes results via Return");
            case Statement.BufferStore s -> new BufferStoreNode(ctx.bufferSlots().get(s.buffer().binding()),
                    lowerExpr(s.index(), ctx), lowerExpr(s.value(), ctx));
            case Statement.DeclareVar d -> new AssignNode(ctx.slots().get(d.variable()), lowerExpr(d.initializer(), ctx));
            case Statement.Assign a -> new AssignNode(ctx.slots().get(a.variable()), lowerExpr(a.value(), ctx));
            case Statement.If f -> new IfNode(lowerExpr(f.condition(), ctx),
                    lowerRegion(f.thenRegion(), ctx), lowerRegion(f.elseRegion(), ctx));
            case Statement.While w -> new WhileNode(lowerExpr(w.condition(), ctx), lowerRegion(w.body(), ctx));
        };
    }

    private ExprNode lowerExpr(Expr expr, Ctx ctx) {
        return switch (expr) {
            case Expr.ConstInt c -> new LiteralNode((int) c.value());
            case Expr.ConstFloat c -> new LiteralNode(c.value());
            case Expr.ConstBool c -> new LiteralNode(c.value());
            case Expr.Read r -> new ReadNode(ctx.slots().get(r.variable()));
            case Expr.InvocationId ignored -> new InvocationIdNode();
            case Expr.BufferLoad l -> new BufferLoadNode(ctx.bufferSlots().get(l.buffer().binding()),
                    lowerExpr(l.index(), ctx));
            case Expr.Binary b -> new BinaryNode(b.op(), lowerExpr(b.lhs(), ctx), lowerExpr(b.rhs(), ctx));
            case Expr.VectorConstruct vc -> new VectorConstructNode(
                    vc.components().stream().map(c -> lowerExpr(c, ctx)).toArray(ExprNode[]::new));
            case Expr.VectorExtract ve -> new VectorExtractNode(lowerExpr(ve.vector(), ctx), ve.index());
            case Expr.Unary u -> new UnaryNode(u.op(), lowerExpr(u.operand(), ctx));
            case Expr.Param p -> new ParamNode(p.index());
            case Expr.Call c -> new CallNode(ctx.targets(), c.callee(),
                    c.arguments().stream().map(a -> lowerExpr(a, ctx)).toArray(ExprNode[]::new));
        };
    }

    private static List<LocalVar> collectVariables(Region region) {
        List<LocalVar> out = new ArrayList<>();
        collectVariables(region, out);
        return out;
    }

    private static void collectVariables(Region region, List<LocalVar> out) {
        for (Statement statement : region.statements()) {
            switch (statement) {
                case Statement.DeclareVar d -> out.add(d.variable());
                case Statement.If f -> {
                    collectVariables(f.thenRegion(), out);
                    collectVariables(f.elseRegion(), out);
                }
                case Statement.While w -> collectVariables(w.body(), out);
                case Statement.Assign ignored -> { }
                case Statement.ReturnVoid ignored -> { }
                case Statement.Return ignored -> { }
                case Statement.StoreResult ignored -> { }
                case Statement.BufferStore ignored -> { }
            }
        }
    }

    // --- Truffle nodes ---------------------------------------------------------------------------------

    /** Non-local exit carrying an optional return value. */
    private static final class ReturnException extends ControlFlowException {
        final transient Object value;

        ReturnException(Object value) {
            this.value = value;
        }
    }

    private static final class ShaderRootNode extends RootNode {
        @Children private final StatementNode[] body;

        ShaderRootNode(FrameDescriptor frameDescriptor, StatementNode[] body) {
            super((TruffleLanguage<?>) null, frameDescriptor);
            this.body = body;
        }

        @Override
        public Object execute(VirtualFrame frame) {
            try {
                for (StatementNode statement : body) {
                    statement.execute(frame);
                }
            } catch (ReturnException exit) {
                return exit.value;
            }
            return null;
        }
    }

    private abstract static class StatementNode extends Node {
        abstract void execute(VirtualFrame frame);
    }

    private abstract static class ExprNode extends Node {
        abstract Object execute(VirtualFrame frame);
    }

    private static final class LiteralNode extends ExprNode {
        private final Object value;

        LiteralNode(Object value) {
            this.value = value;
        }

        @Override
        Object execute(VirtualFrame frame) {
            return value;
        }
    }

    private static final class ReadNode extends ExprNode {
        private final int slot;

        ReadNode(int slot) {
            this.slot = slot;
        }

        @Override
        Object execute(VirtualFrame frame) {
            return frame.getObject(slot);
        }
    }

    private static final class BinaryNode extends ExprNode {
        private final BinaryOp op;
        @Child private ExprNode lhs;
        @Child private ExprNode rhs;

        BinaryNode(BinaryOp op, ExprNode lhs, ExprNode rhs) {
            this.op = op;
            this.lhs = lhs;
            this.rhs = rhs;
        }

        @Override
        Object execute(VirtualFrame frame) {
            return apply(op, lhs.execute(frame), rhs.execute(frame));
        }

        private static Object apply(BinaryOp op, Object left, Object right) {
            if (left instanceof int[] l && right instanceof int[] r) { // componentwise int vector
                int[] out = new int[l.length];
                for (int k = 0; k < l.length; k++) {
                    out[k] = (Integer) scalarInt(op, l[k], r[k]);
                }
                return out;
            }
            if (left instanceof double[] l && right instanceof double[] r) { // componentwise float vector
                double[] out = new double[l.length];
                for (int k = 0; k < l.length; k++) {
                    out[k] = (Double) scalarDouble(op, l[k], r[k]);
                }
                return out;
            }
            if (left instanceof Boolean l && right instanceof Boolean r) {
                return scalarBool(op, l, r);
            }
            if (left instanceof Integer l && right instanceof Integer r) {
                return scalarInt(op, l, r);
            }
            return scalarDouble(op, ((Number) left).doubleValue(), ((Number) right).doubleValue());
        }

        private static Object scalarInt(BinaryOp op, int a, int b) {
            return switch (op) {
                case ADD -> a + b;
                case SUB -> a - b;
                case MUL -> a * b;
                case DIV -> a / b;
                case MOD -> a % b;
                case BIT_AND -> a & b;
                case BIT_OR -> a | b;
                case BIT_XOR -> a ^ b;
                case SHIFT_LEFT -> a << b;
                case SHIFT_RIGHT -> a >> b;
                case LESS_THAN -> a < b;
                case GREATER_THAN -> a > b;
                case EQUAL -> a == b;
                case LOGICAL_AND, LOGICAL_OR -> throw new IllegalStateException("logical op on int: " + op);
            };
        }

        private static Object scalarDouble(BinaryOp op, double a, double b) {
            return switch (op) {
                case ADD -> a + b;
                case SUB -> a - b;
                case MUL -> a * b;
                case DIV -> a / b;
                case MOD -> a % b;
                case LESS_THAN -> a < b;
                case GREATER_THAN -> a > b;
                case EQUAL -> a == b;
                case BIT_AND, BIT_OR, BIT_XOR, SHIFT_LEFT, SHIFT_RIGHT, LOGICAL_AND, LOGICAL_OR ->
                        throw new IllegalStateException("operator not defined on doubles: " + op);
            };
        }

        private static Object scalarBool(BinaryOp op, boolean a, boolean b) {
            return switch (op) {
                case LOGICAL_AND -> a && b;
                case LOGICAL_OR -> a || b;
                case EQUAL -> a == b;
                default -> throw new IllegalStateException("operator not defined on booleans: " + op);
            };
        }
    }

    private static final class AssignNode extends StatementNode {
        private final int slot;
        @Child private ExprNode value;

        AssignNode(int slot, ExprNode value) {
            this.slot = slot;
            this.value = value;
        }

        @Override
        void execute(VirtualFrame frame) {
            frame.setObject(slot, value.execute(frame));
        }
    }

    private static final class ReturnNode extends StatementNode {
        @Child private ExprNode value;

        ReturnNode(ExprNode value) {
            this.value = value;
        }

        @Override
        void execute(VirtualFrame frame) {
            throw new ReturnException(value == null ? null : value.execute(frame));
        }
    }

    private static final class IfNode extends StatementNode {
        @Child private ExprNode condition;
        @Children private final StatementNode[] thenBranch;
        @Children private final StatementNode[] elseBranch;

        IfNode(ExprNode condition, StatementNode[] thenBranch, StatementNode[] elseBranch) {
            this.condition = condition;
            this.thenBranch = thenBranch;
            this.elseBranch = elseBranch;
        }

        @Override
        void execute(VirtualFrame frame) {
            StatementNode[] branch = (Boolean) condition.execute(frame) ? thenBranch : elseBranch;
            for (StatementNode statement : branch) {
                statement.execute(frame);
            }
        }
    }

    private static final class WhileNode extends StatementNode {
        @Child private ExprNode condition;
        @Children private final StatementNode[] body;

        WhileNode(ExprNode condition, StatementNode[] body) {
            this.condition = condition;
            this.body = body;
        }

        @Override
        void execute(VirtualFrame frame) {
            while ((Boolean) condition.execute(frame)) {
                for (StatementNode statement : body) {
                    statement.execute(frame);
                }
            }
        }
    }

    // Kernel nodes read the per-invocation arguments: [Integer invocationIndex, int[][] buffers-by-slot].

    private static final class InvocationIdNode extends ExprNode {
        @Override
        Object execute(VirtualFrame frame) {
            return frame.getArguments()[0];
        }
    }

    private static int[] bufferAt(VirtualFrame frame, int slot) {
        return ((int[][]) frame.getArguments()[1])[slot];
    }

    private static final class BufferLoadNode extends ExprNode {
        private final int slot;
        @Child private ExprNode index;

        BufferLoadNode(int slot, ExprNode index) {
            this.slot = slot;
            this.index = index;
        }

        @Override
        Object execute(VirtualFrame frame) {
            return bufferAt(frame, slot)[(Integer) index.execute(frame)];
        }
    }

    private static final class BufferStoreNode extends StatementNode {
        private final int slot;
        @Child private ExprNode index;
        @Child private ExprNode value;

        BufferStoreNode(int slot, ExprNode index, ExprNode value) {
            this.slot = slot;
            this.index = index;
            this.value = value;
        }

        @Override
        void execute(VirtualFrame frame) {
            bufferAt(frame, slot)[(Integer) index.execute(frame)] = (Integer) value.execute(frame);
        }
    }

    // Vector values are represented as int[] (integer vectors) or double[] (float vectors).

    private static final class VectorConstructNode extends ExprNode {
        @Children private final ExprNode[] components;

        VectorConstructNode(ExprNode[] components) {
            this.components = components;
        }

        @Override
        Object execute(VirtualFrame frame) {
            Object[] values = new Object[components.length];
            boolean anyFloat = false;
            for (int i = 0; i < components.length; i++) {
                values[i] = components[i].execute(frame);
                anyFloat |= values[i] instanceof Double;
            }
            if (anyFloat) {
                double[] vector = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    vector[i] = ((Number) values[i]).doubleValue();
                }
                return vector;
            }
            int[] vector = new int[values.length];
            for (int i = 0; i < values.length; i++) {
                vector[i] = (Integer) values[i];
            }
            return vector;
        }
    }

    private static final class VectorExtractNode extends ExprNode {
        @Child private ExprNode vector;
        private final int index;

        VectorExtractNode(ExprNode vector, int index) {
            this.vector = vector;
            this.index = index;
        }

        @Override
        Object execute(VirtualFrame frame) {
            Object value = vector.execute(frame);
            return value instanceof int[] ints ? ints[index] : ((double[]) value)[index];
        }
    }

    private static final class UnaryNode extends ExprNode {
        private final UnaryOp op;
        @Child private ExprNode operand;

        UnaryNode(UnaryOp op, ExprNode operand) {
            this.op = op;
            this.operand = operand;
        }

        @Override
        Object execute(VirtualFrame frame) {
            Object value = operand.execute(frame);
            return switch (op) {
                case NEGATE -> value instanceof Integer i ? (Object) (-i) : (Object) (-((Double) value));
                case NOT -> (Object) (~((Integer) value));
                case LOGICAL_NOT -> (Object) (!((Boolean) value));
            };
        }
    }

    private static final class ParamNode extends ExprNode {
        private final int index;

        ParamNode(int index) {
            this.index = index;
        }

        @Override
        Object execute(VirtualFrame frame) {
            return frame.getArguments()[index];
        }
    }

    private static final class CallNode extends ExprNode {
        private final Map<Function, CallTarget> targets;
        private final Function callee;
        @Children private final ExprNode[] arguments;

        CallNode(Map<Function, CallTarget> targets, Function callee, ExprNode[] arguments) {
            this.targets = targets;
            this.callee = callee;
            this.arguments = arguments;
        }

        @Override
        Object execute(VirtualFrame frame) {
            Object[] values = new Object[arguments.length];
            for (int i = 0; i < arguments.length; i++) {
                values[i] = arguments[i].execute(frame);
            }
            return targets.get(callee).call(values);
        }
    }
}
