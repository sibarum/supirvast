package dev.supirvast.vastir.core;

/**
 * Whether a kernel's meaning depends on how its invocations are grouped into subgroups — so whoever runs it
 * must fix the subgroup size rather than take the device's.
 *
 * <p>It does when it has a subgroup operation, and also when it only reads {@link Expr.SubgroupInvocationId}
 * or {@link Expr.SubgroupSize}: a lane index is as much a function of the grouping as a reduction is.
 */
public final class Subgroups {

    private Subgroups() {
    }

    /** Whether {@code region} has a subgroup operation or reads a subgroup index, at any depth. */
    public static boolean uses(Region region) {
        for (Statement statement : region.statements()) {
            boolean found = switch (statement) {
                case Statement.SubgroupArithmetic ignored -> true;
                case Statement.SubgroupShuffle ignored -> true;
                case Statement.SubgroupVote ignored -> true;
                case Statement.If f -> uses(f.condition()) || uses(f.thenRegion()) || uses(f.elseRegion());
                case Statement.While w -> uses(w.condition()) || uses(w.body());
                case Statement.Return r -> uses(r.value());
                case Statement.StoreResult s -> uses(s.value());
                case Statement.BufferStore s -> uses(s.index()) || uses(s.value());
                case Statement.SharedStore s -> uses(s.index()) || uses(s.value());
                case Statement.AtomicUpdate s -> uses(s.index()) || uses(s.value());
                case Statement.SharedAtomicUpdate s -> uses(s.index()) || uses(s.value());
                case Statement.AtomicCompareExchange s -> uses(s.index()) || uses(s.expected()) || uses(s.desired());
                case Statement.SharedAtomicCompareExchange s ->
                        uses(s.index()) || uses(s.expected()) || uses(s.desired());
                case Statement.BuiltinWrite s -> uses(s.value());
                case Statement.InterfaceWrite s -> uses(s.value());
                case Statement.DeclareVar d -> uses(d.initializer());
                case Statement.Assign a -> uses(a.value());
                case Statement.Barrier ignored -> false;
                case Statement.ReturnVoid ignored -> false;
            };
            if (found) {
                return true;
            }
        }
        return false;
    }

    private static boolean uses(Expr expr) {
        return switch (expr) {
            case Expr.SubgroupInvocationId ignored -> true;
            case Expr.SubgroupSize ignored -> true;
            case Expr.BufferLoad l -> uses(l.index());
            case Expr.SharedLoad l -> uses(l.index());
            case Expr.Binary b -> uses(b.lhs()) || uses(b.rhs());
            case Expr.Unary u -> uses(u.operand());
            case Expr.Bitcast b -> uses(b.operand());
            case Expr.Convert c -> uses(c.operand());
            case Expr.VectorConstruct v -> v.components().stream().anyMatch(Subgroups::uses);
            case Expr.VectorExtract v -> uses(v.vector());
            case Expr.Call c -> c.arguments().stream().anyMatch(Subgroups::uses);
            case Expr.MathCall m -> m.args().stream().anyMatch(Subgroups::uses);
            case Expr.SampleTexture s -> uses(s.uv());
            case Expr.MatrixTimesVector m -> uses(m.matrix()) || uses(m.vector());
            default -> false;
        };
    }
}
