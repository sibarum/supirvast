package dev.supirvast.vastir.core;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;

/**
 * Where a {@link Statement.Barrier} may stand, decided from the IR's structure.
 *
 * <p>Every invocation of a workgroup must reach the same barriers the same number of times. On the GPU a
 * barrier some invocations skip is undefined behaviour — typically a hang, sometimes a wrong answer. On the
 * CPU, which runs a workgroup phase by phase, it has no meaning at all: there is no phase boundary that only
 * some invocations cross. So a barrier may only sit in <em>uniform</em> control flow, and this class proves
 * that conservatively, before either backend builds anything:
 *
 * <ul>
 *   <li>A value is uniform if it is a constant, the {@link Expr.WorkgroupId}, the {@link Expr.InvocationCount},
 *       a push constant, a parameter, a local variable only ever assigned uniform values in uniform control
 *       flow, or an operation on uniform values. Anything read from memory is not — another invocation may
 *       have written it — and neither is either invocation index.</li>
 *   <li>Control flow is uniform at the top of the function, inside an {@code if} or {@code while} whose
 *       condition is uniform and which is itself in uniform control flow, and nowhere after a return taken
 *       in non-uniform control flow, since the invocations that took it will never reach what follows.</li>
 * </ul>
 *
 * <p>Conservative in the usual direction: a barrier this rejects may be one that would in fact have been
 * reached by everyone, and the fix is to compute the condition from uniform values. Nothing it accepts can
 * diverge, because {@code core} has no {@code break}, no {@code continue} and no {@code goto} — the structure
 * the lowering sees is the control flow that runs.
 *
 * <p>A subgroup operation ({@link Statement.SubgroupArithmetic}, {@link Statement.SubgroupShuffle},
 * {@link Statement.SubgroupVote}) is a barrier for all of this. It combines every lane's value, so every lane
 * must reach it: a lane that skipped it would leave a hole in a reduction and a shuffle reading from nowhere,
 * and on the CPU it is a phase boundary exactly as a barrier is. So it stands only where a barrier may, and a
 * kernel with one runs whole workgroups. Uniform over the workgroup is stronger than a subgroup needs, and
 * cheaper to prove than the difference is worth.
 */
public final class Barriers {

    private Barriers() {
    }

    /** Whether {@code statement} is itself a collective point: a barrier or a subgroup operation. */
    public static boolean isCollective(Statement statement) {
        return switch (statement) {
            case Statement.Barrier ignored -> true;
            case Statement.SubgroupArithmetic ignored -> true;
            case Statement.SubgroupShuffle ignored -> true;
            case Statement.SubgroupVote ignored -> true;
            default -> false;
        };
    }

    /** Whether {@code region} contains a barrier or subgroup operation, at any depth. */
    public static boolean contains(Region region) {
        for (Statement statement : region.statements()) {
            if (contains(statement)) {
                return true;
            }
        }
        return false;
    }

    /** Whether {@code statement} is a barrier or subgroup operation, or contains one. */
    public static boolean contains(Statement statement) {
        return switch (statement) {
            case Statement.If f -> contains(f.thenRegion()) || contains(f.elseRegion());
            case Statement.While w -> contains(w.body());
            default -> isCollective(statement);
        };
    }

    /**
     * Checks that every barrier and subgroup operation in {@code function} is in uniform control flow.
     *
     * @throws IllegalArgumentException naming the first that is not, and why
     */
    public static void check(Function function) {
        if (!contains(function.body())) {
            return;
        }
        Analysis analysis = new Analysis();
        // Marking a variable non-uniform can make an earlier condition non-uniform, which can mark further
        // variables; iterate to the fixpoint before judging any barrier. Terminates: the set only grows.
        int before;
        do {
            before = analysis.nonUniform.size();
            analysis.run(function.body(), false);
        } while (analysis.nonUniform.size() != before);
        analysis.run(function.body(), true);
        if (analysis.violation != null) {
            throw new IllegalArgumentException(analysis.what + " in '" + function.name() + "' is not in uniform "
                    + "control flow: " + analysis.violation + ". Every invocation of a workgroup must reach the "
                    + "same barriers and subgroup operations, so one can only be under conditions computed from "
                    + "uniform values (constants, the workgroup id, the invocation count, the subgroup size, push "
                    + "constants, and variables assigned only from those)");
        }
    }

    private static final class Analysis {
        final Set<LocalVar> nonUniform = Collections.newSetFromMap(new IdentityHashMap<>());
        private boolean diverged;   // some invocations may have returned before this point
        private boolean report;
        String violation;
        String what;                // "a barrier" or "a subgroup operation", for the report

        void run(Region body, boolean report) {
            this.report = report;
            this.diverged = false;
            this.violation = null;
            walk(body, true, null);
        }

        /** Records the first collective statement found outside uniform control flow. */
        private void collective(Statement statement, boolean uniformHere, Expr why) {
            if (report && !uniformHere && violation == null) {
                what = statement instanceof Statement.Barrier ? "a barrier" : "a subgroup operation";
                violation = diverged
                        ? "it follows a return that only some invocations may take"
                        : "it is under the condition " + why + ", which can differ between invocations of a "
                                + "workgroup";
            }
        }

        /**
         * @param uniformControl whether control flow reaching this region is uniform
         * @param why            the condition that made it non-uniform, for the report (null if uniform)
         */
        private void walk(Region region, boolean uniformControl, Expr why) {
            for (Statement statement : region.statements()) {
                boolean uniformHere = uniformControl && !diverged;
                switch (statement) {
                    case Statement.Barrier b -> collective(b, uniformHere, why);
                    // A subgroup result differs between subgroups, and a shuffle's between lanes: not uniform.
                    case Statement.SubgroupArithmetic s -> {
                        collective(s, uniformHere, why);
                        markNonUniform(s.result());
                    }
                    case Statement.SubgroupShuffle s -> {
                        collective(s, uniformHere, why);
                        markNonUniform(s.result());
                    }
                    case Statement.SubgroupVote s -> {
                        collective(s, uniformHere, why);
                        markNonUniform(s.result());
                    }
                    case Statement.DeclareVar d -> assign(d.variable(), d.initializer(), uniformHere);
                    case Statement.Assign a -> assign(a.variable(), a.value(), uniformHere);
                    case Statement.AtomicUpdate s -> markNonUniform(s.previous());
                    case Statement.AtomicCompareExchange s -> markNonUniform(s.previous());
                    case Statement.SharedAtomicUpdate s -> markNonUniform(s.previous());
                    case Statement.SharedAtomicCompareExchange s -> markNonUniform(s.previous());
                    case Statement.Return ignored -> diverged |= !uniformHere;
                    case Statement.ReturnVoid ignored -> diverged |= !uniformHere;
                    case Statement.If f -> {
                        boolean inner = uniformHere && uniform(f.condition());
                        Expr reason = inner ? null : (uniformHere ? f.condition() : why);
                        walk(f.thenRegion(), inner, reason);
                        walk(f.elseRegion(), inner, reason);
                    }
                    case Statement.While w -> {
                        boolean inner = uniformHere && uniform(w.condition());
                        Expr reason = inner ? null : (uniformHere ? w.condition() : why);
                        // Twice: a return late in the body is before the barriers early in it, one iteration on.
                        walk(w.body(), inner, reason);
                        walk(w.body(), inner, reason);
                    }
                    default -> { }
                }
            }
        }

        private void assign(LocalVar variable, Expr value, boolean uniformHere) {
            if (!uniformHere || !uniform(value)) {
                nonUniform.add(variable);
            }
        }

        private void markNonUniform(LocalVar variable) {
            if (variable != null) {
                nonUniform.add(variable);
            }
        }

        private boolean uniform(Expr expr) {
            return switch (expr) {
                case Expr.ConstInt ignored -> true;
                case Expr.ConstFloat ignored -> true;
                case Expr.ConstBool ignored -> true;
                case Expr.WorkgroupId ignored -> true;
                case Expr.InvocationCount ignored -> true;
                case Expr.SubgroupSize ignored -> true;
                case Expr.SubgroupInvocationId ignored -> false;
                case Expr.PushConstantRead ignored -> true;
                case Expr.Param ignored -> true;
                case Expr.Read r -> !nonUniform.contains(r.variable());
                case Expr.Binary b -> uniform(b.lhs()) && uniform(b.rhs());
                case Expr.Unary u -> uniform(u.operand());
                case Expr.Bitcast b -> uniform(b.operand());
                case Expr.Convert c -> uniform(c.operand());
                case Expr.VectorConstruct v -> v.components().stream().allMatch(this::uniform);
                case Expr.VectorExtract v -> uniform(v.vector());
                case Expr.MathCall m -> m.args().stream().allMatch(this::uniform);
                case Expr.MatrixTimesVector m -> uniform(m.matrix()) && uniform(m.vector());
                // Per invocation by definition, or read from memory another invocation may have written.
                case Expr.InvocationId ignored -> false;
                case Expr.LocalInvocationId ignored -> false;
                case Expr.BufferLoad ignored -> false;
                case Expr.SharedLoad ignored -> false;
                case Expr.BuiltinRead ignored -> false;
                case Expr.InterfaceRead ignored -> false;
                case Expr.SampleTexture ignored -> false;
                // A callee could read memory; not worth proving otherwise until a kernel needs it.
                case Expr.Call ignored -> false;
            };
        }
    }
}
