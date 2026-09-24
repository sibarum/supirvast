package dev.supirvast.vast;

import com.oracle.truffle.api.CallTarget;
import com.oracle.truffle.api.Truffle;
import com.oracle.truffle.api.TruffleLanguage;
import com.oracle.truffle.api.frame.FrameDescriptor;
import com.oracle.truffle.api.frame.FrameSlotKind;
import com.oracle.truffle.api.frame.MaterializedFrame;
import com.oracle.truffle.api.frame.VirtualFrame;
import com.oracle.truffle.api.nodes.ControlFlowException;
import com.oracle.truffle.api.nodes.Node;
import com.oracle.truffle.api.nodes.RootNode;
import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.Barriers;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.MathFn;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.SharedArray;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.SubgroupOp;
import dev.supirvast.vastir.core.UnaryOp;
import dev.supirvast.vastir.type.Type;

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

    // An invocation's frame arguments. Every kernel has the first two; a dispatch adds the count; a kernel run
    // workgroup by workgroup adds its indices within the dispatch and its workgroup's shared arrays.
    private static final int ARG_INVOCATION = 0;
    private static final int ARG_BUFFERS = 1;
    private static final int ARG_COUNT = 2;
    private static final int ARG_LOCAL = 3;
    private static final int ARG_WORKGROUP = 4;
    private static final int ARG_SHARED = 5;

    /** What an invocation's frame arguments carry, and so which expressions this lowering can support. */
    private enum Args {
        /** {@code [invocation, buffers]}, or a plain function's parameters. */
        PLAIN,
        /** Plus the dispatch's invocation count. */
        COUNTED,
        /** Plus the local invocation index, the workgroup index and the workgroup's shared arrays. */
        GROUPED
    }

    /**
     * Lowering context: local-variable frame slots, buffer slots (kernels), callee targets (calls), shared-array
     * slots (kernels run by workgroup), what the frame arguments carry, and the subgroup size (kernels run by
     * workgroup; 0 otherwise).
     */
    private record Ctx(Map<LocalVar, Integer> slots, Map<Integer, Integer> bufferSlots,
            Map<Function, CallTarget> targets, Map<SharedArray, Integer> sharedSlots, Args args, int subgroupSize) {

        Ctx(Map<LocalVar, Integer> slots, Map<Integer, Integer> bufferSlots, Map<Function, CallTarget> targets) {
            this(slots, bufferSlots, targets, Map.of(), Args.PLAIN, 0);
        }

        void require(Args needed, String what) {
            if (args.ordinal() < needed.ordinal()) {
                throw new UnsupportedOperationException(what + " needs the dispatch lowering: "
                        + "CoreToTruffle.lowerDispatch, which runs a kernel a whole dispatch at a time");
            }
        }
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
        return buildTarget(function, bufferSlots(buffers), Map.of());
    }

    /**
     * Lowers a data-parallel kernel for {@linkplain CpuKernel#dispatch whole dispatches}, in workgroups of
     * {@code workgroupSize} — the size only matters to a kernel that uses workgroup memory, workgroup indices
     * or barriers, and is otherwise how the GPU happens to schedule it. Unlike {@link #lowerKernel} this
     * supports all of those and {@link Expr.InvocationCount}.
     *
     * <p>A barrier splits the kernel into phases, and each workgroup runs one phase for every invocation before
     * starting the next: sequential invocations cannot wait for each other, but they can take turns. An
     * invocation's local variables live in its own frame, which persists across the phases, and its shared
     * arrays belong to the workgroup. That a barrier is in uniform control flow is what makes the phase
     * boundaries the same for every invocation, so {@link Barriers#check} runs first.
     *
     * <p>A subgroup operation is a phase boundary too: every lane evaluates its operand, the lanes of each
     * subgroup are combined, and every lane gets its result. Subgroups are {@link #DEFAULT_SUBGROUP_SIZE}
     * lanes; see the four-argument form.
     *
     * @throws IllegalArgumentException if a barrier or subgroup operation is not in uniform control flow
     */
    public CpuKernel lowerDispatch(Function function, List<Buffer> buffers, int workgroupSize) {
        return lowerDispatch(function, buffers, workgroupSize, DEFAULT_SUBGROUP_SIZE);
    }

    /** 32: an NVIDIA warp, an AMD RDNA wave32, and a size every current Intel GPU can be asked for. */
    public static final int DEFAULT_SUBGROUP_SIZE = 32;

    /**
     * As {@link #lowerDispatch(Function, List, int)}, with subgroups of {@code subgroupSize} lanes: lane
     * {@code local % subgroupSize} of subgroup {@code local / subgroupSize}, which is how the GPU lays out full
     * subgroups of a required size in a one-dimensional workgroup. A kernel with subgroup operations needs
     * the workgroup to be a whole number of subgroups, as the GPU does.
     *
     * @throws IllegalArgumentException if a barrier or subgroup operation is not in uniform control flow, or
     *                                  a kernel with subgroup operations has a partial subgroup
     */
    public CpuKernel lowerDispatch(Function function, List<Buffer> buffers, int workgroupSize, int subgroupSize) {
        if (workgroupSize < 1) {
            throw new IllegalArgumentException("workgroup size must be >= 1, got " + workgroupSize);
        }
        if (subgroupSize < 1) {
            throw new IllegalArgumentException("subgroup size must be >= 1, got " + subgroupSize);
        }
        Barriers.check(function);
        WorkgroupScan usage = new WorkgroupScan();
        usage.scan(function.body());
        if (usage.subgroups && workgroupSize % subgroupSize != 0) {
            throw new IllegalArgumentException("a workgroup of " + workgroupSize + " is not a whole number of "
                    + subgroupSize + "-lane subgroups, which a kernel with subgroup operations needs");
        }
        FrameDescriptor.Builder frame = FrameDescriptor.newBuilder();
        Map<LocalVar, Integer> slots = frameSlots(function, frame);
        FrameDescriptor descriptor = frame.build();
        if (!usage.grouped()) {
            Ctx ctx = new Ctx(slots, bufferSlots(buffers), Map.of(), Map.of(), Args.COUNTED, subgroupSize);
            CallTarget target = new ShaderRootNode(descriptor, lowerRegion(function.body(), ctx)).getCallTarget();
            return new CpuKernel(target, false, false, workgroupSize);
        }
        Map<SharedArray, Integer> sharedSlots = new IdentityHashMap<>();
        for (SharedArray array : usage.arrays) {
            sharedSlots.put(array, sharedSlots.size());
        }
        Ctx ctx = new Ctx(slots, bufferSlots(buffers), Map.of(), sharedSlots, Args.GROUPED, subgroupSize);
        GroupNode[] plan = lowerGroupRegion(function.body(), ctx);
        CallTarget target = new WorkgroupRootNode(descriptor, plan, usage.arrays.toArray(SharedArray[]::new),
                workgroupSize).getCallTarget();
        return new CpuKernel(target, true, usage.collective(), workgroupSize);
    }

    private static Map<Integer, Integer> bufferSlots(List<Buffer> buffers) {
        Map<Integer, Integer> bufferSlots = new java.util.HashMap<>();
        for (int i = 0; i < buffers.size(); i++) {
            bufferSlots.put(buffers.get(i).binding(), i);
        }
        return bufferSlots;
    }

    private CallTarget buildTarget(Function function, Map<Integer, Integer> bufferSlots,
            Map<Function, CallTarget> targets) {
        FrameDescriptor.Builder frame = FrameDescriptor.newBuilder();
        Map<LocalVar, Integer> slots = frameSlots(function, frame);
        StatementNode[] body = lowerRegion(function.body(), new Ctx(slots, bufferSlots, targets));
        return new ShaderRootNode(frame.build(), body).getCallTarget();
    }

    private static Map<LocalVar, Integer> frameSlots(Function function, FrameDescriptor.Builder frame) {
        Map<LocalVar, Integer> slots = new IdentityHashMap<>();
        for (LocalVar variable : collectVariables(function.body())) {
            slots.put(variable, frame.addSlot(FrameSlotKind.Object, variable.name(), null));
        }
        return slots;
    }

    /**
     * The workgroup-level plan of a region: maximal runs of barrier-free statements become {@link PhaseNode}s
     * that run every invocation through the run in turn; a barrier ends a run; an {@code if} or {@code while}
     * with a barrier inside is evaluated once for the whole group — its condition is uniform — and its regions
     * planned the same way.
     */
    private GroupNode[] lowerGroupRegion(Region region, Ctx ctx) {
        List<GroupNode> plan = new ArrayList<>();
        List<StatementNode> phase = new ArrayList<>();
        for (Statement statement : region.statements()) {
            if (!Barriers.contains(statement)) {
                phase.add(lowerStatement(statement, ctx));
                continue;
            }
            if (!phase.isEmpty()) {
                plan.add(new PhaseNode(phase.toArray(StatementNode[]::new)));
                phase.clear();
            }
            plan.add(switch (statement) {
                case Statement.Barrier ignored -> new BarrierNode();
                case Statement.SubgroupArithmetic s -> new SubgroupArithmeticNode(ctx.slots().get(s.result()),
                        ctx.subgroupSize(), lowerExpr(s.value(), ctx), s.op(), s.scan(), s.value().type());
                case Statement.SubgroupShuffle s -> new SubgroupShuffleNode(ctx.slots().get(s.result()),
                        ctx.subgroupSize(), lowerExpr(s.value(), ctx), s.kind(), lowerExpr(s.lane(), ctx));
                case Statement.SubgroupVote s -> new SubgroupVoteNode(ctx.slots().get(s.result()),
                        ctx.subgroupSize(), lowerExpr(s.value(), ctx), s.kind());
                case Statement.If f -> new GroupIfNode(lowerExpr(f.condition(), ctx),
                        lowerGroupRegion(f.thenRegion(), ctx), lowerGroupRegion(f.elseRegion(), ctx));
                case Statement.While w -> new GroupWhileNode(lowerExpr(w.condition(), ctx),
                        lowerGroupRegion(w.body(), ctx));
                default -> throw new IllegalStateException("not a barrier or a region holding one: " + statement);
            });
        }
        if (!phase.isEmpty()) {
            plan.add(new PhaseNode(phase.toArray(StatementNode[]::new)));
        }
        return plan.toArray(GroupNode[]::new);
    }

    /** The shared arrays a kernel uses, and whether it needs to be run workgroup by workgroup at all. */
    private static final class WorkgroupScan {
        final java.util.Set<SharedArray> arrays = new java.util.LinkedHashSet<>();   // identity equality
        boolean barrier;
        boolean subgroups;
        boolean indices;

        /** Whether the kernel has a point every invocation must reach, so runs whole workgroups. */
        boolean collective() {
            return barrier || subgroups;
        }

        boolean grouped() {
            return collective() || indices || !arrays.isEmpty();
        }

        void scan(Region region) {
            for (Statement statement : region.statements()) {
                switch (statement) {
                    case Statement.Barrier ignored -> barrier = true;
                    case Statement.SubgroupArithmetic s -> { subgroups = true; scan(s.value()); }
                    case Statement.SubgroupShuffle s -> { subgroups = true; scan(s.value()); scan(s.lane()); }
                    case Statement.SubgroupVote s -> { subgroups = true; scan(s.value()); }
                    case Statement.SharedStore s -> { arrays.add(s.array()); scan(s.index()); scan(s.value()); }
                    case Statement.SharedAtomicUpdate s -> { arrays.add(s.array()); scan(s.index()); scan(s.value()); }
                    case Statement.SharedAtomicCompareExchange s -> {
                        arrays.add(s.array());
                        scan(s.index());
                        scan(s.expected());
                        scan(s.desired());
                    }
                    case Statement.BufferStore s -> { scan(s.index()); scan(s.value()); }
                    case Statement.AtomicUpdate s -> { scan(s.index()); scan(s.value()); }
                    case Statement.AtomicCompareExchange s -> { scan(s.index()); scan(s.expected()); scan(s.desired()); }
                    case Statement.Return r -> scan(r.value());
                    case Statement.StoreResult s -> scan(s.value());
                    case Statement.BuiltinWrite s -> scan(s.value());
                    case Statement.InterfaceWrite s -> scan(s.value());
                    case Statement.DeclareVar d -> scan(d.initializer());
                    case Statement.Assign a -> scan(a.value());
                    case Statement.If f -> { scan(f.condition()); scan(f.thenRegion()); scan(f.elseRegion()); }
                    case Statement.While w -> { scan(w.condition()); scan(w.body()); }
                    case Statement.ReturnVoid ignored -> { }
                }
            }
        }

        void scan(Expr expr) {
            switch (expr) {
                case Expr.LocalInvocationId ignored -> indices = true;
                case Expr.WorkgroupId ignored -> indices = true;
                case Expr.SubgroupInvocationId ignored -> indices = true;
                case Expr.SubgroupSize ignored -> indices = true;
                case Expr.SharedLoad l -> { arrays.add(l.array()); scan(l.index()); }
                case Expr.BufferLoad l -> scan(l.index());
                case Expr.Binary b -> { scan(b.lhs()); scan(b.rhs()); }
                case Expr.Unary u -> scan(u.operand());
                case Expr.Bitcast b -> scan(b.operand());
                case Expr.Convert c -> scan(c.operand());
                case Expr.VectorConstruct v -> v.components().forEach(this::scan);
                case Expr.VectorExtract v -> scan(v.vector());
                case Expr.Call c -> c.arguments().forEach(this::scan);
                case Expr.MathCall m -> m.args().forEach(this::scan);
                case Expr.SampleTexture s -> scan(s.uv());
                case Expr.MatrixTimesVector m -> { scan(m.matrix()); scan(m.vector()); }
                default -> { }
            }
        }
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
            case Statement.BuiltinWrite ignored -> throw new UnsupportedOperationException(
                    "graphics built-in outputs are GPU-only; the CPU backend runs compute kernels");
            case Statement.InterfaceWrite ignored -> throw new UnsupportedOperationException(
                    "stage interface outputs are GPU-only; the CPU backend runs compute kernels");
            case Statement.BufferStore s -> new BufferStoreNode(ctx.bufferSlots().get(s.buffer().binding()),
                    s.buffer().element(), lowerExpr(s.index(), ctx), lowerExpr(s.value(), ctx));
            case Statement.AtomicUpdate s -> new AtomicUpdateNode(ctx.bufferSlots().get(s.buffer().binding()),
                    s.buffer().element(), s.op(), s.previous() == null ? -1 : ctx.slots().get(s.previous()),
                    lowerExpr(s.index(), ctx), lowerExpr(s.value(), ctx));
            case Statement.AtomicCompareExchange s -> new AtomicCompareExchangeNode(
                    ctx.bufferSlots().get(s.buffer().binding()), ctx.slots().get(s.previous()),
                    lowerExpr(s.index(), ctx), lowerExpr(s.expected(), ctx), lowerExpr(s.desired(), ctx));
            case Statement.SharedStore s -> new SharedStoreNode(sharedSlot(s.array(), ctx),
                    lowerExpr(s.index(), ctx), lowerExpr(s.value(), ctx));
            case Statement.SharedAtomicUpdate s -> new SharedAtomicUpdateNode(sharedSlot(s.array(), ctx),
                    s.array().element(), s.op(), s.previous() == null ? -1 : ctx.slots().get(s.previous()),
                    lowerExpr(s.index(), ctx), lowerExpr(s.value(), ctx));
            case Statement.SharedAtomicCompareExchange s -> new SharedAtomicCompareExchangeNode(
                    sharedSlot(s.array(), ctx), ctx.slots().get(s.previous()),
                    lowerExpr(s.index(), ctx), lowerExpr(s.expected(), ctx), lowerExpr(s.desired(), ctx));
            // Only reached outside a workgroup plan: inside one, a barrier is a phase boundary, not a node.
            case Statement.Barrier ignored -> {
                ctx.require(Args.GROUPED, "a barrier");
                throw new IllegalStateException("a barrier outside the workgroup plan");
            }
            case Statement.SubgroupArithmetic ignored -> subgroupOutsidePlan(ctx);
            case Statement.SubgroupShuffle ignored -> subgroupOutsidePlan(ctx);
            case Statement.SubgroupVote ignored -> subgroupOutsidePlan(ctx);
            case Statement.DeclareVar d -> new AssignNode(ctx.slots().get(d.variable()), lowerExpr(d.initializer(), ctx));
            case Statement.Assign a -> new AssignNode(ctx.slots().get(a.variable()), lowerExpr(a.value(), ctx));
            case Statement.If f -> new IfNode(lowerExpr(f.condition(), ctx),
                    lowerRegion(f.thenRegion(), ctx), lowerRegion(f.elseRegion(), ctx));
            case Statement.While w -> new WhileNode(lowerExpr(w.condition(), ctx), lowerRegion(w.body(), ctx));
        };
    }

    private ExprNode lowerExpr(Expr expr, Ctx ctx) {
        return switch (expr) {
            case Expr.ConstInt c -> new LiteralNode(c.type().width() == 64
                    ? (Object) c.value()
                    : (Object) narrowInt((int) c.value(), c.type().width(), !c.type().signed()));
            case Expr.ConstFloat c -> new LiteralNode(
                    c.type().width() == 64 ? (Object) c.value() : (Object) (float) c.value());
            case Expr.ConstBool c -> new LiteralNode(c.value());
            case Expr.Read r -> new ReadNode(ctx.slots().get(r.variable()));
            case Expr.InvocationId ignored -> new ArgumentNode(ARG_INVOCATION);
            case Expr.InvocationCount ignored -> {
                ctx.require(Args.COUNTED, "the invocation count");
                yield new ArgumentNode(ARG_COUNT);
            }
            case Expr.LocalInvocationId ignored -> {
                ctx.require(Args.GROUPED, "the local invocation id");
                yield new ArgumentNode(ARG_LOCAL);
            }
            case Expr.WorkgroupId ignored -> {
                ctx.require(Args.GROUPED, "the workgroup id");
                yield new ArgumentNode(ARG_WORKGROUP);
            }
            case Expr.SubgroupInvocationId ignored -> {
                ctx.require(Args.GROUPED, "the subgroup invocation id");
                yield new LaneNode(ctx.subgroupSize());
            }
            case Expr.SubgroupSize ignored -> {
                ctx.require(Args.GROUPED, "the subgroup size");
                yield new LiteralNode(ctx.subgroupSize());
            }
            case Expr.SharedLoad l -> new SharedLoadNode(sharedSlot(l.array(), ctx), lowerExpr(l.index(), ctx));
            case Expr.BufferLoad l -> new BufferLoadNode(ctx.bufferSlots().get(l.buffer().binding()),
                    l.buffer().element(), lowerExpr(l.index(), ctx));
            case Expr.BuiltinRead ignored -> throw new UnsupportedOperationException(
                    "graphics built-in inputs are GPU-only; the CPU backend runs compute kernels");
            case Expr.InterfaceRead ignored -> throw new UnsupportedOperationException(
                    "stage interface inputs are GPU-only; the CPU backend runs compute kernels");
            case Expr.Binary b -> new BinaryNode(b.op(), isUnsigned(b.lhs().type()), intWidth(b.lhs().type()),
                    lowerExpr(b.lhs(), ctx), lowerExpr(b.rhs(), ctx));
            case Expr.VectorConstruct vc -> new VectorConstructNode(
                    vc.components().stream().map(c -> lowerExpr(c, ctx)).toArray(ExprNode[]::new));
            case Expr.VectorExtract ve -> new VectorExtractNode(lowerExpr(ve.vector(), ctx), ve.index());
            case Expr.Bitcast bc -> new BitcastNode(lowerExpr(bc.operand(), ctx),
                    bitcastKind(bc.operand().type(), bc.type()));
            case Expr.Convert cv -> new ConvertNode(lowerExpr(cv.operand(), ctx),
                    element(cv.operand().type()), element(cv.type()));
            case Expr.Unary u -> new UnaryNode(u.op(), lowerExpr(u.operand(), ctx));
            case Expr.Param p -> new ParamNode(p.index());
            case Expr.Call c -> new CallNode(ctx.targets(), c.callee(),
                    c.arguments().stream().map(a -> lowerExpr(a, ctx)).toArray(ExprNode[]::new));
            case Expr.MathCall m -> new MathCallNode(m.fn(),
                    m.args().stream().map(a -> lowerExpr(a, ctx)).toArray(ExprNode[]::new));
            case Expr.SampleTexture ignored -> throw new UnsupportedOperationException(
                    "texture sampling is graphics-only — no CPU backend");
            case Expr.PushConstantRead ignored -> throw new UnsupportedOperationException(
                    "push constants are graphics-only — no CPU backend");
            case Expr.MatrixTimesVector ignored -> throw new UnsupportedOperationException(
                    "matrix math is graphics-only — no CPU backend yet");
        };
    }

    /** Like a barrier, a subgroup operation is a node of the workgroup plan, never of an invocation's body. */
    private static StatementNode subgroupOutsidePlan(Ctx ctx) {
        ctx.require(Args.GROUPED, "a subgroup operation");
        throw new IllegalStateException("a subgroup operation outside the workgroup plan");
    }

    private static int sharedSlot(SharedArray array, Ctx ctx) {
        ctx.require(Args.GROUPED, "workgroup memory");
        return ctx.sharedSlots().get(array);
    }

    /** Whether {@code type} (or a vector's component) is an unsigned integer, selecting unsigned CPU ops. */
    private static boolean isUnsigned(Type type) {
        Type element = type instanceof Type.Vector v ? v.component() : type;
        return element instanceof Type.Int i && !i.signed();
    }

    /** The scalar type, unwrapping a vector to its component (conversions act per component). */
    private static Type element(Type type) {
        return type instanceof Type.Vector v ? v.component() : type;
    }

    // Buffer columns ride an int[] wire of raw 32-bit words: 32-bit elements take one word (f32 as its bits),
    // 64-bit elements take two (low word first, little-endian), matching the SPIR-V SSBO layout.

    private static int width(Type scalar) {
        if (scalar instanceof Type.Int i) {
            return i.width();
        }
        return scalar instanceof Type.Float f ? f.width() : 32;
    }

    private static int wordsPerElement(Type element) {
        return width(element) == 64 ? 2 : 1;
    }

    /** Reads element {@code index} from a column's word array, boxing per the element type. */
    private static Object readColumnElement(int[] words, int index, Type element) {
        if (width(element) == 64) {
            long bits = (words[2 * index] & 0xFFFFFFFFL) | ((long) words[2 * index + 1] << 32);
            return element instanceof Type.Float ? (Object) Double.longBitsToDouble(bits) : (Object) bits;
        }
        int word = words[index];
        return element instanceof Type.Float ? (Object) Float.intBitsToFloat(word) : (Object) word;
    }

    /** Writes {@code value} as element {@code index} of a column's word array. */
    private static void writeColumnElement(int[] words, int index, Type element, Object value) {
        if (width(element) == 64) {
            long bits = element instanceof Type.Float
                    ? Double.doubleToRawLongBits((Double) value)
                    : (Long) value;
            words[2 * index] = (int) bits;
            words[2 * index + 1] = (int) (bits >>> 32);
        } else if (element instanceof Type.Float) {
            words[index] = Float.floatToRawIntBits((Float) value);
        } else {
            words[index] = (Integer) value;
        }
    }

    /** Integer width of {@code type} (or a vector's component); 32 for non-integers (carrier stays Integer). */
    private static int intWidth(Type type) {
        return element(type) instanceof Type.Int i ? i.width() : 32;
    }

    /**
     * Re-canonicalizes a 32-bit-carried integer to its declared width: narrow types (i8/i16) wrap and
     * sign-/zero-extend back into the {@code int} box exactly as the GPU keeps an 8/16-bit result, so chained
     * narrow arithmetic overflows identically. A no-op at 32 bits.
     */
    private static int narrowInt(int value, int width, boolean unsigned) {
        return switch (width) {
            case 8 -> unsigned ? (value & 0xFF) : (byte) value;
            case 16 -> unsigned ? (value & 0xFFFF) : (short) value;
            default -> value;
        };
    }

    /** How a {@code Bitcast} must reinterpret its operand's Java carrier (see {@link BitcastNode}). */
    private enum BitcastKind { IDENTITY, INT_TO_FLOAT32, FLOAT32_TO_INT, LONG_TO_DOUBLE, DOUBLE_TO_LONG }

    private static BitcastKind bitcastKind(Type source, Type target) {
        boolean sourceFloat = source instanceof Type.Float;
        boolean targetFloat = target instanceof Type.Float;
        if (targetFloat && source instanceof Type.Int) {
            return ((Type.Float) target).width() == 64 ? BitcastKind.LONG_TO_DOUBLE : BitcastKind.INT_TO_FLOAT32;
        }
        if (sourceFloat && target instanceof Type.Int) {
            return ((Type.Float) source).width() == 64 ? BitcastKind.DOUBLE_TO_LONG : BitcastKind.FLOAT32_TO_INT;
        }
        return BitcastKind.IDENTITY; // i32<->u32, i64<->u64: same carrier, raw bits unchanged
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
                case Statement.AtomicUpdate s -> {
                    if (s.previous() != null) {
                        addOnce(s.previous(), out);
                    }
                }
                case Statement.AtomicCompareExchange s -> addOnce(s.previous(), out);
                case Statement.SharedAtomicUpdate s -> {
                    if (s.previous() != null) {
                        addOnce(s.previous(), out);
                    }
                }
                case Statement.SharedAtomicCompareExchange s -> addOnce(s.previous(), out);
                case Statement.SubgroupArithmetic s -> addOnce(s.result(), out);
                case Statement.SubgroupShuffle s -> addOnce(s.result(), out);
                case Statement.SubgroupVote s -> addOnce(s.result(), out);
                case Statement.If f -> {
                    collectVariables(f.thenRegion(), out);
                    collectVariables(f.elseRegion(), out);
                }
                case Statement.While w -> collectVariables(w.body(), out);
                case Statement.SharedStore ignored -> { }
                case Statement.Barrier ignored -> { }
                case Statement.Assign ignored -> { }
                case Statement.ReturnVoid ignored -> { }
                case Statement.Return ignored -> { }
                case Statement.StoreResult ignored -> { }
                case Statement.BufferStore ignored -> { }
                case Statement.BuiltinWrite ignored -> { }
                case Statement.InterfaceWrite ignored -> { }
            }
        }
    }

    /** An atomic declares its {@code previous} only if nothing else does; by identity, as {@code CoreToSpirv} does. */
    private static void addOnce(LocalVar variable, List<LocalVar> out) {
        for (LocalVar existing : out) {
            if (existing == variable) {
                return;
            }
        }
        out.add(variable);
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
        private final boolean unsigned;
        private final int width; // result width for narrow-int wraparound (8/16/32; 64 uses scalarLong)
        @Child private ExprNode lhs;
        @Child private ExprNode rhs;

        BinaryNode(BinaryOp op, boolean unsigned, int width, ExprNode lhs, ExprNode rhs) {
            this.op = op;
            this.unsigned = unsigned;
            this.width = width;
            this.lhs = lhs;
            this.rhs = rhs;
        }

        @Override
        Object execute(VirtualFrame frame) {
            return apply(op, unsigned, width, lhs.execute(frame), rhs.execute(frame));
        }

        private static Object apply(BinaryOp op, boolean unsigned, int width, Object left, Object right) {
            if (left instanceof int[] l && right instanceof int[] r) { // componentwise int vector
                int[] out = new int[l.length];
                for (int k = 0; k < l.length; k++) {
                    out[k] = (Integer) scalarInt(op, unsigned, width, l[k], r[k]);
                }
                return out;
            }
            if (left instanceof float[] l && right instanceof float[] r) { // componentwise f32 vector
                float[] out = new float[l.length];
                for (int k = 0; k < l.length; k++) {
                    out[k] = (Float) scalarFloat(op, l[k], r[k]);
                }
                return out;
            }
            if (left instanceof double[] l && right instanceof double[] r) { // componentwise f64 vector
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
                return scalarInt(op, unsigned, width, l, r);
            }
            if (left instanceof Long l && right instanceof Long r) {
                return scalarLong(op, unsigned, l, r);
            }
            if (left instanceof Float l && right instanceof Float r) {
                return scalarFloat(op, l, r);
            }
            return scalarDouble(op, ((Number) left).doubleValue(), ((Number) right).doubleValue());
        }

        /**
         * Scalar integer op. The {@code unsigned} flag picks the unsigned interpretation of the operations
         * whose result depends on signedness (division, remainder, right shift, ordering comparisons),
         * matching SPIR-V's {@code OpUDiv}/{@code OpUMod}/{@code OpShiftRightLogical}/{@code OpU*Than}. The
         * 32-bit two's-complement bit pattern is shared, so add/sub/mul/bitwise/left-shift/equal are identical.
         */
        private static Object scalarInt(BinaryOp op, boolean unsigned, int width, int a, int b) {
            return switch (op) {
                case ADD -> narrowInt(a + b, width, unsigned);
                case SUB -> narrowInt(a - b, width, unsigned);
                case MUL -> narrowInt(a * b, width, unsigned);
                case DIV -> narrowInt(unsigned ? Integer.divideUnsigned(a, b) : a / b, width, unsigned);
                case MOD -> narrowInt(unsigned ? Integer.remainderUnsigned(a, b) : a % b, width, unsigned);
                case BIT_AND -> narrowInt(a & b, width, unsigned);
                case BIT_OR -> narrowInt(a | b, width, unsigned);
                case BIT_XOR -> narrowInt(a ^ b, width, unsigned);
                case SHIFT_LEFT -> narrowInt(a << b, width, unsigned);
                case SHIFT_RIGHT -> unsigned ? a >>> b : a >> b;
                case LESS_THAN -> unsigned ? Integer.compareUnsigned(a, b) < 0 : a < b;
                case GREATER_THAN -> unsigned ? Integer.compareUnsigned(a, b) > 0 : a > b;
                case EQUAL -> a == b;
                case LOGICAL_AND, LOGICAL_OR -> throw new IllegalStateException("logical op on int: " + op);
            };
        }

        /** Scalar 64-bit integer op — the {@code long} analogue of {@link #scalarInt}. */
        private static Object scalarLong(BinaryOp op, boolean unsigned, long a, long b) {
            return switch (op) {
                case ADD -> a + b;
                case SUB -> a - b;
                case MUL -> a * b;
                case DIV -> unsigned ? Long.divideUnsigned(a, b) : a / b;
                case MOD -> unsigned ? Long.remainderUnsigned(a, b) : a % b;
                case BIT_AND -> a & b;
                case BIT_OR -> a | b;
                case BIT_XOR -> a ^ b;
                case SHIFT_LEFT -> a << b;
                case SHIFT_RIGHT -> unsigned ? a >>> b : a >> b;
                case LESS_THAN -> unsigned ? Long.compareUnsigned(a, b) < 0 : a < b;
                case GREATER_THAN -> unsigned ? Long.compareUnsigned(a, b) > 0 : a > b;
                case EQUAL -> a == b;
                case LOGICAL_AND, LOGICAL_OR -> throw new IllegalStateException("logical op on long: " + op);
            };
        }

        /**
         * Scalar 32-bit float op. Crucially computed in {@code float}, not {@code double}, so each operation
         * rounds to f32 exactly as the GPU does — otherwise a chain of ops would keep extra precision the
         * GPU discards and the CPU reference would drift from the device.
         */
        private static Object scalarFloat(BinaryOp op, float a, float b) {
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
                        throw new IllegalStateException("operator not defined on floats: " + op);
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

    // Kernel nodes read the per-invocation arguments: [Integer invocationIndex, int[][] buffers-by-slot], then
    // for a dispatch [Integer count], then for a workgroup [Integer local, Integer workgroup, Object[][] shared].

    /** One of the invocation's indices, or the dispatch's count, straight from the frame arguments. */
    private static final class ArgumentNode extends ExprNode {
        private final int index;

        ArgumentNode(int index) {
            this.index = index;
        }

        @Override
        Object execute(VirtualFrame frame) {
            return frame.getArguments()[index];
        }
    }

    // --- workgroups ------------------------------------------------------------------------------------
    //
    // A kernel run by workgroup has one WorkgroupRootNode call per workgroup. It gives every invocation its own
    // materialized frame -- where its locals persist between phases -- and the workgroup fresh shared arrays,
    // then executes the plan: PhaseNodes run each live invocation through a barrier-free run of statements in
    // turn, and the group-level if/while evaluate their uniform conditions against every invocation, so a
    // condition that was not uniform after all is a thrown witness rather than a silently wrong answer.

    /** Shared arrays, one element array per slot, reached through the invocation's frame arguments. */
    private static Object[] sharedAt(VirtualFrame frame, int slot) {
        return ((Object[][]) frame.getArguments()[ARG_SHARED])[slot];
    }

    /** The invocations of one workgroup, and which of them have returned. */
    private static final class Group {
        final MaterializedFrame[] frames;
        final boolean[] returned;
        int live;

        Group(MaterializedFrame[] frames) {
            this.frames = frames;
            this.returned = new boolean[frames.length];
            this.live = frames.length;
        }

        /**
         * Whether the group proceeds into a region with a barrier in it. {@link Barriers#check} proves that no
         * barrier follows a return only some invocations took, so either all have returned or none has.
         */
        boolean proceeds() {
            if (live == 0) {
                return false;
            }
            if (live != frames.length) {
                throw new IllegalStateException((frames.length - live) + " of " + frames.length
                        + " invocations returned before a barrier the rest must reach");
            }
            return true;
        }

        /** Evaluates a uniform condition against every invocation, requiring that they agree. */
        boolean condition(ExprNode condition) {
            if (!proceeds()) {
                return false;
            }
            boolean first = (Boolean) condition.execute(frames[0]);
            for (int i = 1; i < frames.length; i++) {
                if ((Boolean) condition.execute(frames[i]) != first) {
                    throw new IllegalStateException("a condition guarding a barrier differs between invocations "
                            + "0 and " + i + " of a workgroup");
                }
            }
            return first;
        }
    }

    private static final class WorkgroupRootNode extends RootNode {
        @Children private final GroupNode[] plan;
        private final SharedArray[] arrays;
        private final int workgroupSize;

        WorkgroupRootNode(FrameDescriptor invocationFrame, GroupNode[] plan, SharedArray[] arrays, int workgroupSize) {
            super((TruffleLanguage<?>) null, invocationFrame);
            this.plan = plan;
            this.arrays = arrays;
            this.workgroupSize = workgroupSize;
        }

        /** Called as {@code (workgroup, buffers, count, running)}: one workgroup, of {@code running} invocations. */
        @Override
        public Object execute(VirtualFrame frame) {
            Object[] args = frame.getArguments();
            int workgroup = (Integer) args[0];
            int running = (Integer) args[3];
            Object[][] shared = new Object[arrays.length][];
            for (int s = 0; s < arrays.length; s++) {
                shared[s] = new Object[arrays[s].length()];
                java.util.Arrays.fill(shared[s], zero(arrays[s].element()));
            }
            MaterializedFrame[] frames = new MaterializedFrame[running];
            for (int local = 0; local < running; local++) {
                Object[] invocation = {workgroup * workgroupSize + local, args[1], args[2], local, workgroup, shared};
                frames[local] = Truffle.getRuntime().createMaterializedFrame(invocation, getFrameDescriptor());
            }
            Group group = new Group(frames);
            for (GroupNode node : plan) {
                node.execute(group);
            }
            return null;
        }

        /** What an element of workgroup memory starts as here: zero, one of the values "undefined" allows. */
        private static Object zero(Type element) {
            return switch (element) {
                case Type.Int i -> i.width() == 64 ? (Object) 0L : (Object) 0;
                case Type.Float f -> f.width() == 64 ? (Object) 0.0 : (Object) 0f;
                case Type.Bool ignored -> false;
                case Type.Vector v -> v.component() instanceof Type.Float f
                        ? (f.width() == 64 ? (Object) new double[v.count()] : (Object) new float[v.count()])
                        : (Object) new int[v.count()];
                default -> throw new IllegalStateException("no workgroup memory of " + element);
            };
        }
    }

    private abstract static class GroupNode extends Node {
        abstract void execute(Group group);
    }

    /** A barrier-free run of statements, run to its end by each live invocation before the next starts it. */
    private static final class PhaseNode extends GroupNode {
        @Children private final StatementNode[] body;

        PhaseNode(StatementNode[] body) {
            this.body = body;
        }

        @Override
        void execute(Group group) {
            for (int i = 0; i < group.frames.length; i++) {
                if (group.returned[i]) {
                    continue;
                }
                try {
                    for (StatementNode statement : body) {
                        statement.execute(group.frames[i]);
                    }
                } catch (ReturnException exit) {
                    group.returned[i] = true;
                    group.live--;
                }
            }
        }
    }

    /**
     * A barrier. The phase boundary is the barrier itself — the phase before it has run for every invocation
     * before the phase after it starts — so all that is left is to check that everyone arrived.
     */
    private static final class BarrierNode extends GroupNode {
        @Override
        void execute(Group group) {
            group.proceeds();
        }
    }

    /** The lane: the local invocation id modulo the subgroup size. */
    private static final class LaneNode extends ExprNode {
        private final int subgroupSize;

        LaneNode(int subgroupSize) {
            this.subgroupSize = subgroupSize;
        }

        @Override
        Object execute(VirtualFrame frame) {
            return (Integer) frame.getArguments()[ARG_LOCAL] % subgroupSize;
        }
    }

    /**
     * A subgroup operation, which ends a phase as a barrier does: every lane has reached it before any lane
     * gets its result. Its operand (and a shuffle's lane) is evaluated in every invocation's frame, the lanes of
     * each subgroup — {@code subgroupSize} consecutive invocations, the workgroup being a whole number of them
     * — are combined, and each invocation's result variable assigned.
     */
    private abstract static class SubgroupNode extends GroupNode {
        private final int resultSlot;
        final int subgroupSize;
        @Child private ExprNode value;
        @Child private ExprNode argument;   // a shuffle's lane or delta; null for the others

        SubgroupNode(int resultSlot, int subgroupSize, ExprNode value, ExprNode argument) {
            this.resultSlot = resultSlot;
            this.subgroupSize = subgroupSize;
            this.value = value;
            this.argument = argument;
        }

        @Override
        void execute(Group group) {
            if (!group.proceeds()) {
                return;
            }
            MaterializedFrame[] frames = group.frames;
            Object[] values = new Object[frames.length];
            int[] arguments = argument == null ? null : new int[frames.length];
            for (int i = 0; i < frames.length; i++) {
                values[i] = value.execute(frames[i]);
                if (arguments != null) {
                    arguments[i] = (Integer) argument.execute(frames[i]);
                }
            }
            Object[] results = new Object[frames.length];
            for (int base = 0; base < frames.length; base += subgroupSize) {
                combine(values, arguments, results, base);
            }
            for (int i = 0; i < frames.length; i++) {
                frames[i].setObject(resultSlot, results[i]);
            }
        }

        /** Fills {@code results[base .. base + subgroupSize)} from the same lanes of {@code values}. */
        abstract void combine(Object[] values, int[] arguments, Object[] results, int base);
    }

    /**
     * Reduce and scans, combining lanes in lane order. For a float add or multiply that order is this
     * backend's, not the device's, so the two can differ in the last bits unless the values make order moot.
     */
    private static final class SubgroupArithmeticNode extends SubgroupNode {
        private final SubgroupOp op;
        private final Statement.SubgroupArithmetic.Scan scan;
        private final boolean isFloat;
        private final boolean signed;

        SubgroupArithmeticNode(int resultSlot, int subgroupSize, ExprNode value, SubgroupOp op,
                Statement.SubgroupArithmetic.Scan scan, Type type) {
            super(resultSlot, subgroupSize, value, null);
            this.op = op;
            this.scan = scan;
            this.isFloat = type instanceof Type.Float;
            this.signed = type instanceof Type.Int i && i.signed();
        }

        @Override
        void combine(Object[] values, int[] arguments, Object[] results, int base) {
            Object acc = identity();
            for (int lane = 0; lane < subgroupSize; lane++) {
                Object before = acc;
                acc = apply(acc, values[base + lane]);
                results[base + lane] = scan == Statement.SubgroupArithmetic.Scan.EXCLUSIVE ? before : acc;
            }
            if (scan == Statement.SubgroupArithmetic.Scan.REDUCE) {
                java.util.Arrays.fill(results, base, base + subgroupSize, acc);
            }
        }

        /** The value combining with which changes nothing — what an exclusive scan gives lane 0. */
        private Object identity() {
            if (isFloat) {
                return switch (op) {
                    case ADD -> 0f;
                    case MUL -> 1f;
                    case MIN -> Float.POSITIVE_INFINITY;
                    case MAX -> Float.NEGATIVE_INFINITY;
                    case AND, OR, XOR -> throw new IllegalStateException("subgroup " + op + " on a float");
                };
            }
            return switch (op) {
                case ADD, OR, XOR -> 0;
                case MUL -> 1;
                case AND -> -1;
                case MIN -> signed ? Integer.MAX_VALUE : -1;   // -1 is the largest unsigned value
                case MAX -> signed ? Integer.MIN_VALUE : 0;
            };
        }

        private Object apply(Object left, Object right) {
            if (isFloat) {
                float a = (Float) left;
                float b = (Float) right;
                return switch (op) {
                    case ADD -> a + b;
                    case MUL -> a * b;
                    case MIN -> Math.min(a, b);
                    case MAX -> Math.max(a, b);
                    case AND, OR, XOR -> throw new IllegalStateException("subgroup " + op + " on a float");
                };
            }
            int a = (Integer) left;
            int b = (Integer) right;
            return switch (op) {
                case ADD -> a + b;
                case MUL -> a * b;
                case MIN -> signed ? Math.min(a, b) : (Integer.compareUnsigned(a, b) <= 0 ? a : b);
                case MAX -> signed ? Math.max(a, b) : (Integer.compareUnsigned(a, b) >= 0 ? a : b);
                case AND -> a & b;
                case OR -> a | b;
                case XOR -> a ^ b;
            };
        }
    }

    /** Each lane reads another's value; a source outside the subgroup gives the lane its own. */
    private static final class SubgroupShuffleNode extends SubgroupNode {
        private final Statement.SubgroupShuffle.Kind kind;

        SubgroupShuffleNode(int resultSlot, int subgroupSize, ExprNode value, Statement.SubgroupShuffle.Kind kind,
                ExprNode lane) {
            super(resultSlot, subgroupSize, value, lane);
            this.kind = kind;
        }

        @Override
        void combine(Object[] values, int[] arguments, Object[] results, int base) {
            for (int lane = 0; lane < subgroupSize; lane++) {
                int argument = arguments[base + lane];
                int source = switch (kind) {
                    case INDEX -> argument;
                    case XOR -> lane ^ argument;
                    case UP -> lane - argument;
                    case DOWN -> lane + argument;
                };
                boolean inside = source >= 0 && source < subgroupSize;
                results[base + lane] = values[base + (inside ? source : lane)];
            }
        }
    }

    private static final class SubgroupVoteNode extends SubgroupNode {
        private final Statement.SubgroupVote.Kind kind;

        SubgroupVoteNode(int resultSlot, int subgroupSize, ExprNode value, Statement.SubgroupVote.Kind kind) {
            super(resultSlot, subgroupSize, value, null);
            this.kind = kind;
        }

        @Override
        void combine(Object[] values, int[] arguments, Object[] results, int base) {
            boolean all = true;
            boolean any = false;
            boolean equal = true;
            for (int lane = 0; lane < subgroupSize; lane++) {
                Object v = values[base + lane];
                if (kind == Statement.SubgroupVote.Kind.ALL_EQUAL) {
                    equal &= same(values[base], v);
                } else {
                    all &= (Boolean) v;
                    any |= (Boolean) v;
                }
            }
            boolean result = switch (kind) {
                case ALL -> all;
                case ANY -> any;
                case ALL_EQUAL -> equal;
            };
            java.util.Arrays.fill(results, base, base + subgroupSize, result);
        }

        /** Floats compare as numbers, so 0 and -0 are equal and a NaN equals nothing, as on the device. */
        private static boolean same(Object a, Object b) {
            return a instanceof Float x && b instanceof Float y ? x.floatValue() == y.floatValue() : a.equals(b);
        }
    }

    private static final class GroupIfNode extends GroupNode {
        @Child private ExprNode condition;
        @Children private final GroupNode[] thenPlan;
        @Children private final GroupNode[] elsePlan;

        GroupIfNode(ExprNode condition, GroupNode[] thenPlan, GroupNode[] elsePlan) {
            this.condition = condition;
            this.thenPlan = thenPlan;
            this.elsePlan = elsePlan;
        }

        @Override
        void execute(Group group) {
            if (group.live == 0) {
                return;
            }
            for (GroupNode node : group.condition(condition) ? thenPlan : elsePlan) {
                node.execute(group);
            }
        }
    }

    private static final class GroupWhileNode extends GroupNode {
        @Child private ExprNode condition;
        @Children private final GroupNode[] body;

        GroupWhileNode(ExprNode condition, GroupNode[] body) {
            this.condition = condition;
            this.body = body;
        }

        @Override
        void execute(Group group) {
            while (group.condition(condition)) {
                for (GroupNode node : body) {
                    node.execute(group);
                }
            }
        }
    }

    private static final class SharedLoadNode extends ExprNode {
        private final int slot;
        @Child private ExprNode index;

        SharedLoadNode(int slot, ExprNode index) {
            this.slot = slot;
            this.index = index;
        }

        @Override
        Object execute(VirtualFrame frame) {
            return sharedAt(frame, slot)[(Integer) index.execute(frame)];
        }
    }

    private static final class SharedStoreNode extends StatementNode {
        private final int slot;
        @Child private ExprNode index;
        @Child private ExprNode value;

        SharedStoreNode(int slot, ExprNode index, ExprNode value) {
            this.slot = slot;
            this.index = index;
            this.value = value;
        }

        @Override
        void execute(VirtualFrame frame) {
            sharedAt(frame, slot)[(Integer) index.execute(frame)] = value.execute(frame);
        }
    }

    /** As {@link AtomicUpdateNode}, on workgroup memory, whose elements are already boxed values. */
    private static final class SharedAtomicUpdateNode extends StatementNode {
        private final int slot;
        private final Type element;
        private final AtomicOp op;
        private final int previousSlot; // -1 when the old value is not wanted
        @Child private ExprNode index;
        @Child private ExprNode value;

        SharedAtomicUpdateNode(int slot, Type element, AtomicOp op, int previousSlot, ExprNode index,
                ExprNode value) {
            this.slot = slot;
            this.element = element;
            this.op = op;
            this.previousSlot = previousSlot;
            this.index = index;
            this.value = value;
        }

        @Override
        void execute(VirtualFrame frame) {
            Object[] elements = sharedAt(frame, slot);
            int i = (Integer) index.execute(frame);
            Object operand = value.execute(frame);
            Object old = elements[i];
            elements[i] = element instanceof Type.Float
                    ? (Object) AtomicUpdateNode.applyFloat(op, (Float) old, (Float) operand)
                    : (Object) AtomicUpdateNode.applyInt(op, ((Type.Int) element).signed(), (Integer) old,
                            (Integer) operand);
            if (previousSlot >= 0) {
                frame.setObject(previousSlot, old);
            }
        }
    }

    private static final class SharedAtomicCompareExchangeNode extends StatementNode {
        private final int slot;
        private final int previousSlot;
        @Child private ExprNode index;
        @Child private ExprNode expected;
        @Child private ExprNode desired;

        SharedAtomicCompareExchangeNode(int slot, int previousSlot, ExprNode index, ExprNode expected,
                ExprNode desired) {
            this.slot = slot;
            this.previousSlot = previousSlot;
            this.index = index;
            this.expected = expected;
            this.desired = desired;
        }

        @Override
        void execute(VirtualFrame frame) {
            Object[] elements = sharedAt(frame, slot);
            int i = (Integer) index.execute(frame);
            int comparator = (Integer) expected.execute(frame);
            int replacement = (Integer) desired.execute(frame);
            int old = (Integer) elements[i];
            if (old == comparator) {
                elements[i] = replacement;
            }
            frame.setObject(previousSlot, old);
        }
    }

    private static int[] bufferAt(VirtualFrame frame, int slot) {
        return ((int[][]) frame.getArguments()[1])[slot];
    }

    private static final class BufferLoadNode extends ExprNode {
        private final int slot;
        private final Type element;
        @Child private ExprNode index;

        BufferLoadNode(int slot, Type element, ExprNode index) {
            this.slot = slot;
            this.element = element;
            this.index = index;
        }

        @Override
        Object execute(VirtualFrame frame) {
            return readColumnElement(bufferAt(frame, slot), (Integer) index.execute(frame), element);
        }
    }

    private static final class BufferStoreNode extends StatementNode {
        private final int slot;
        private final Type element;
        @Child private ExprNode index;
        @Child private ExprNode value;

        BufferStoreNode(int slot, Type element, ExprNode index, ExprNode value) {
            this.slot = slot;
            this.element = element;
            this.index = index;
            this.value = value;
        }

        @Override
        void execute(VirtualFrame frame) {
            writeColumnElement(bufferAt(frame, slot), (Integer) index.execute(frame), element, value.execute(frame));
        }
    }

    // Atomics. The CPU dispatch calls the target once per invocation, one after another, so a plain
    // read-modify-write on the column is atomic by construction: no two invocations are ever inside one at
    // once. A CPU dispatch that ran invocations in parallel would have to revisit these two nodes, and nothing
    // else here.

    private static final class AtomicUpdateNode extends StatementNode {
        private final int slot;
        private final Type element;
        private final AtomicOp op;
        private final int previousSlot; // -1 when the old value is not wanted
        @Child private ExprNode index;
        @Child private ExprNode value;

        AtomicUpdateNode(int slot, Type element, AtomicOp op, int previousSlot, ExprNode index, ExprNode value) {
            this.slot = slot;
            this.element = element;
            this.op = op;
            this.previousSlot = previousSlot;
            this.index = index;
            this.value = value;
        }

        @Override
        void execute(VirtualFrame frame) {
            int[] words = bufferAt(frame, slot);
            int i = (Integer) index.execute(frame);
            Object operand = value.execute(frame);
            Object old = readColumnElement(words, i, element);
            Object updated = element instanceof Type.Float
                    ? (Object) applyFloat(op, (Float) old, (Float) operand)
                    : (Object) applyInt(op, ((Type.Int) element).signed(), (Integer) old, (Integer) operand);
            writeColumnElement(words, i, element, updated);
            if (previousSlot >= 0) {
                frame.setObject(previousSlot, old);
            }
        }

        private static int applyInt(AtomicOp op, boolean signed, int old, int operand) {
            return switch (op) {
                case ADD -> old + operand;
                case SUB -> old - operand;
                case MIN -> signed ? Math.min(old, operand) : (Integer.compareUnsigned(old, operand) <= 0 ? old : operand);
                case MAX -> signed ? Math.max(old, operand) : (Integer.compareUnsigned(old, operand) >= 0 ? old : operand);
                case AND -> old & operand;
                case OR -> old | operand;
                case XOR -> old ^ operand;
                case EXCHANGE -> operand;
            };
        }

        /**
         * Float min and max follow the extension's {@code fmin}/{@code fmax} reading, where a NaN operand is
         * ignored in favour of the other. Whether a given device does the same is not something this backend
         * can promise, and the two are not required to agree.
         */
        private static float applyFloat(AtomicOp op, float old, float operand) {
            return switch (op) {
                case ADD -> old + operand;
                case MIN -> Float.isNaN(old) ? operand : Float.isNaN(operand) ? old : Math.min(old, operand);
                case MAX -> Float.isNaN(old) ? operand : Float.isNaN(operand) ? old : Math.max(old, operand);
                case EXCHANGE -> operand;
                case SUB, AND, OR, XOR -> throw new IllegalStateException("atomic " + op + " on a float");
            };
        }
    }

    private static final class AtomicCompareExchangeNode extends StatementNode {
        private final int slot;
        private final int previousSlot;
        @Child private ExprNode index;
        @Child private ExprNode expected;
        @Child private ExprNode desired;

        AtomicCompareExchangeNode(int slot, int previousSlot, ExprNode index, ExprNode expected, ExprNode desired) {
            this.slot = slot;
            this.previousSlot = previousSlot;
            this.index = index;
            this.expected = expected;
            this.desired = desired;
        }

        @Override
        void execute(VirtualFrame frame) {
            int[] words = bufferAt(frame, slot);
            int i = (Integer) index.execute(frame);
            int comparator = (Integer) expected.execute(frame);
            int replacement = (Integer) desired.execute(frame);
            int old = words[i];
            if (old == comparator) {
                words[i] = replacement;
            }
            frame.setObject(previousSlot, old);
        }
    }

    // Vector values are represented by component carrier: int[] (i32), float[] (f32), or double[] (f64).

    private static final class VectorConstructNode extends ExprNode {
        @Children private final ExprNode[] components;

        VectorConstructNode(ExprNode[] components) {
            this.components = components;
        }

        @Override
        Object execute(VirtualFrame frame) {
            Object[] values = new Object[components.length];
            boolean anyDouble = false;
            boolean anyFloat = false;
            for (int i = 0; i < components.length; i++) {
                values[i] = components[i].execute(frame);
                anyDouble |= values[i] instanceof Double;
                anyFloat |= values[i] instanceof Float;
            }
            if (anyDouble) {
                double[] vector = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    vector[i] = ((Number) values[i]).doubleValue();
                }
                return vector;
            }
            if (anyFloat) {
                float[] vector = new float[values.length];
                for (int i = 0; i < values.length; i++) {
                    vector[i] = ((Number) values[i]).floatValue();
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
            return switch (vector.execute(frame)) {
                case int[] ints -> ints[index];
                case float[] floats -> floats[index];
                case double[] doubles -> doubles[index];
                default -> throw new IllegalStateException("not a vector value");
            };
        }
    }

    /**
     * Reinterprets bits without changing the underlying value. Integer↔integer of the same width
     * ({@code i32}↔{@code uint32}, {@code i64}↔{@code uint64}) share a Java carrier, so it's an identity
     * pass-through — the surrounding operators' signedness, not the box, selects signed vs unsigned. The
     * {@code f32}↔{@code i32} cases genuinely re-encode (a boxed {@code Float} vs {@code Integer} are different
     * representations) via {@link Float#intBitsToFloat}/{@link Float#floatToRawIntBits}.
     */
    private static final class BitcastNode extends ExprNode {
        @Child private ExprNode operand;
        private final BitcastKind kind;

        BitcastNode(ExprNode operand, BitcastKind kind) {
            this.operand = operand;
            this.kind = kind;
        }

        @Override
        Object execute(VirtualFrame frame) {
            Object value = operand.execute(frame);
            return switch (kind) {
                case IDENTITY -> value;
                case INT_TO_FLOAT32 -> Float.intBitsToFloat((Integer) value);
                case FLOAT32_TO_INT -> Float.floatToRawIntBits((Float) value);
                case LONG_TO_DOUBLE -> Double.longBitsToDouble((Long) value);
                case DOUBLE_TO_LONG -> Double.doubleToRawLongBits((Double) value);
            };
        }
    }

    /**
     * Numeric conversion mirroring SPIR-V's {@code OpSConvert}/{@code OpUConvert}/{@code OpConvert*To*}/
     * {@code OpFConvert}, between the Java carriers {@code Integer} (i32), {@code Long} (i64), {@code Float}
     * (f32) and {@code Double} (f64):
     * <ul>
     *   <li>int→int — width change; on widening, sign- or zero-extend per the <em>target</em>'s signedness
     *       (the convert opcode is chosen by result type), narrowing keeps the low word;</li>
     *   <li>int→float — interpret the source per its <em>own</em> signedness, then round to the target width;</li>
     *   <li>float→int — round toward zero ({@code (long)} then narrow), matching {@code FToS}/{@code FToU} for
     *       in-range values (out-of-range is undefined in SPIR-V, so callers keep values in range);</li>
     *   <li>float→float — re-round to the target width.</li>
     * </ul>
     */
    private static final class ConvertNode extends ExprNode {
        @Child private ExprNode operand;
        private final Type source;
        private final Type target;

        ConvertNode(ExprNode operand, Type source, Type target) {
            this.operand = operand;
            this.source = source;
            this.target = target;
        }

        @Override
        Object execute(VirtualFrame frame) {
            Object value = operand.execute(frame);
            if (target instanceof Type.Float tf) {
                double d = (value instanceof Integer i && !signedInt(source))
                        ? (i & 0xFFFFFFFFL)          // unsigned i32 source: zero-extend before widening
                        : ((Number) value).doubleValue();
                return tf.width() == 64 ? (Object) d : (Object) (float) d;
            }
            int targetWidth = ((Type.Int) target).width();
            boolean targetUnsigned = !signedInt(target);
            if (value instanceof Float || value instanceof Double) {
                long truncated = (long) ((Number) value).doubleValue(); // round toward zero
                return targetWidth == 64 ? (Object) truncated
                        : (Object) narrowInt((int) truncated, targetWidth, targetUnsigned);
            }
            // int → int width change: extend by the target's signedness, then re-canonicalize narrow widths.
            long bits = value instanceof Long l
                    ? l
                    : (signedInt(target) ? (long) (Integer) value : ((Integer) value & 0xFFFFFFFFL));
            return targetWidth == 64 ? (Object) bits : (Object) narrowInt((int) bits, targetWidth, targetUnsigned);
        }

        private static boolean signedInt(Type type) {
            return !(type instanceof Type.Int i) || i.signed();
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
                case NEGATE -> switch (value) {
                    case Integer i -> (Object) (-i);
                    case Long l -> (Object) (-l);
                    case Float f -> (Object) (-f);
                    default -> (Object) (-((Double) value));
                };
                case NOT -> value instanceof Long l ? (Object) (~l) : (Object) (~((Integer) value));
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

    /**
     * GLSL.std.450 math intrinsics on the CPU — the counterpart to the SPIR-V {@code OpExtInst} the GPU emits.
     * Values ride the same carriers as everywhere else ({@code Float}/{@code Double} scalars, {@code float[]}/
     * {@code double[]} vectors); computation is done in {@code double} and packed back to {@code float} unless a
     * {@code double} operand is present. Scalars broadcast against vectors elementwise. Geometric ops
     * (dot/length/distance/normalize/cross/reflect) reduce or map as GLSL defines; unary/binary/ternary ops apply
     * per component. This makes an SDF authored once in {@code core} evaluable on the CPU (physics) as well as the
     * GPU (rendering) — the whole point of the dual backend.
     *
     * <p>Precision note: transcendentals go through {@code java.lang.Math} in {@code double} then round to the
     * carrier, so f32 results are close to but not guaranteed bit-identical to the GPU for those ops (the
     * correctly-rounded arithmetic ops still match). Fine for simulation; exact-match tests should stick to
     * +,−,× as the differential harness already does.
     */
    private static final class MathCallNode extends ExprNode {
        private final MathFn fn;
        @Children private final ExprNode[] args;

        MathCallNode(MathFn fn, ExprNode[] args) {
            this.fn = fn;
            this.args = args;
        }

        @Override
        Object execute(VirtualFrame frame) {
            Object[] v = new Object[args.length];
            boolean anyDouble = false;
            for (int i = 0; i < args.length; i++) {
                v[i] = args[i].execute(frame);
                anyDouble |= v[i] instanceof Double || v[i] instanceof double[];
            }
            return switch (fn) {
                case LENGTH -> scalar(Math.sqrt(sumSquares(v[0])), anyDouble);
                case DISTANCE -> {
                    double s = 0;
                    for (int i = 0; i < vectorLength(v[0]); i++) {
                        double d = comp(v[0], i) - comp(v[1], i);
                        s += d * d;
                    }
                    yield scalar(Math.sqrt(s), anyDouble);
                }
                case DOT -> {
                    double s = 0;
                    for (int i = 0; i < vectorLength(v[0]); i++) {
                        s += comp(v[0], i) * comp(v[1], i);
                    }
                    yield scalar(s, anyDouble);
                }
                case NORMALIZE -> {
                    int n = vectorLength(v[0]);
                    double len = Math.sqrt(sumSquares(v[0]));
                    double[] r = new double[n];
                    for (int i = 0; i < n; i++) {
                        r[i] = comp(v[0], i) / len;
                    }
                    yield pack(r, anyDouble);
                }
                case CROSS -> pack(new double[] {
                        comp(v[0], 1) * comp(v[1], 2) - comp(v[0], 2) * comp(v[1], 1),
                        comp(v[0], 2) * comp(v[1], 0) - comp(v[0], 0) * comp(v[1], 2),
                        comp(v[0], 0) * comp(v[1], 1) - comp(v[0], 1) * comp(v[1], 0)}, anyDouble);
                case REFLECT -> {
                    int n = vectorLength(v[0]);
                    double d = 0;
                    for (int i = 0; i < n; i++) {
                        d += comp(v[1], i) * comp(v[0], i);   // dot(N, I)
                    }
                    double[] r = new double[n];
                    for (int i = 0; i < n; i++) {
                        r[i] = comp(v[0], i) - 2 * d * comp(v[1], i);
                    }
                    yield pack(r, anyDouble);
                }
                default -> elementwise(v, anyDouble);
            };
        }

        /** Unary/binary/ternary ops applied per component, scalars broadcasting; scalar result if no vector arg. */
        private Object elementwise(Object[] v, boolean anyDouble) {
            int n = 0;
            for (Object o : v) {
                n = Math.max(n, o instanceof float[] f ? f.length : o instanceof double[] d ? d.length
                        : o instanceof int[] a ? a.length : 0);
            }
            if (n == 0) {
                return scalar(applyScalar(v, 0), anyDouble);
            }
            double[] r = new double[n];
            for (int i = 0; i < n; i++) {
                r[i] = applyScalar(v, i);
            }
            return pack(r, anyDouble);
        }

        private double applyScalar(Object[] v, int i) {
            double a = comp(v[0], i);
            return switch (fn) {
                case ABS -> Math.abs(a);
                case SIGN -> Math.signum(a);
                case SQRT -> Math.sqrt(a);
                case INVERSE_SQRT -> 1.0 / Math.sqrt(a);
                case FLOOR -> Math.floor(a);
                case CEIL -> Math.ceil(a);
                case TRUNC -> (double) (long) a;
                case ROUND, ROUND_EVEN -> Math.rint(a);
                case FRACT -> a - Math.floor(a);
                case SIN -> Math.sin(a);
                case COS -> Math.cos(a);
                case TAN -> Math.tan(a);
                case ASIN -> Math.asin(a);
                case ACOS -> Math.acos(a);
                case ATAN -> Math.atan(a);
                case SINH -> Math.sinh(a);
                case COSH -> Math.cosh(a);
                case TANH -> Math.tanh(a);
                case EXP -> Math.exp(a);
                case LOG -> Math.log(a);
                case EXP2 -> Math.pow(2, a);
                case LOG2 -> Math.log(a) / Math.log(2);
                case RADIANS -> Math.toRadians(a);
                case DEGREES -> Math.toDegrees(a);
                case MIN -> Math.min(a, comp(v[1], i));
                case MAX -> Math.max(a, comp(v[1], i));
                case POW -> Math.pow(a, comp(v[1], i));
                case ATAN2 -> Math.atan2(a, comp(v[1], i));
                case STEP -> comp(v[1], i) < a ? 0.0 : 1.0;               // step(edge=a, x=v1)
                case CLAMP -> Math.min(Math.max(a, comp(v[1], i)), comp(v[2], i));
                case MIX -> a + (comp(v[1], i) - a) * comp(v[2], i);
                case FMA -> a * comp(v[1], i) + comp(v[2], i);
                case SMOOTHSTEP -> {
                    double e0 = a;
                    double t = Math.min(Math.max((comp(v[2], i) - e0) / (comp(v[1], i) - e0), 0.0), 1.0);
                    yield t * t * (3.0 - 2.0 * t);
                }
                default -> throw new UnsupportedOperationException("CPU MathCall not implemented: " + fn);
            };
        }

        private static double comp(Object o, int i) {
            return switch (o) {
                case float[] f -> f[i];
                case double[] d -> d[i];
                case int[] a -> a[i];
                default -> ((Number) o).doubleValue();   // scalar broadcasts to any component
            };
        }

        private static int vectorLength(Object o) {
            return switch (o) {
                case float[] f -> f.length;
                case double[] d -> d.length;
                case int[] a -> a.length;
                default -> 1;
            };
        }

        private static double sumSquares(Object o) {
            double s = 0;
            for (int i = 0; i < vectorLength(o); i++) {
                double c = comp(o, i);
                s += c * c;
            }
            return s;
        }

        private static Object scalar(double value, boolean anyDouble) {
            return anyDouble ? (Object) value : (Object) (float) value;
        }

        private static Object pack(double[] values, boolean anyDouble) {
            if (anyDouble) {
                return values;
            }
            float[] r = new float[values.length];
            for (int i = 0; i < values.length; i++) {
                r[i] = (float) values[i];
            }
            return r;
        }
    }
}
