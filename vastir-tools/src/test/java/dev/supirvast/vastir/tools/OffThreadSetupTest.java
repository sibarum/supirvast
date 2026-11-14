package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * A program's kernels and buffers made on workers while the owning thread goes on dispatching, then handed to it.
 *
 * <p>That is what an application sharing its device with a window has to do: registration lowers, validates and has the
 * driver compile, which is too long for the thread that draws, so it happens elsewhere at the same time as that thread's
 * own work on the same device. What is asserted is that nothing is lost or crossed in the hand-over — every handle runs,
 * on the GPU, against the buffers it was given — and that the owning thread's work beside it is undisturbed.
 */
class OffThreadSetupTest {

    private static final Type.Int I32 = Type.int32();
    private static final int N = 1000;

    /** {@code to[i] = from[i] * factor + 1}: a different kernel for each factor, so no two workers lower the same one. */
    private static KernelSpec affine(int factor) {
        Buffer from = new Buffer("from", 0, I32);
        Buffer to = new Buffer("to", 1, I32);
        return new KernelSpec(kernel(new Statement.BufferStore(to, gid(),
                        add(new Expr.Binary(BinaryOp.MUL, new Expr.BufferLoad(from, gid()), i(factor)), i(1)))),
                List.of(KernelColumn.input("from", 0, I32), KernelColumn.output("to", 1, I32)));
    }

    /** Several buffers holding data, cleared in one go after a dispatch that wrote one of them: all zero, after it. */
    @Test
    void clearZeroesEveryBufferAfterTheWorkBeforeIt() {
        try (Accelerator accelerator = new Accelerator()) {
            KernelHandle affine = accelerator.register(affine(3)).orElseThrow();
            ResidentBuffer from = accelerator.allocate(I32, N);
            ResidentBuffer to = accelerator.allocate(I32, N);
            ResidentBuffer other = accelerator.allocate(I32, 7);
            int[] ones = new int[N];
            java.util.Arrays.fill(ones, 1);
            from.write(ones);
            other.write(new int[] {1, 2, 3, 4, 5, 6, 7});
            affine.dispatch(List.of(from, to), N);   // writes `to`, and is not waited for

            accelerator.clear(List.of(from, to, other));

            assertArrayEquals(new int[N], from.read());
            assertArrayEquals(new int[N], to.read(), "the clear comes after the dispatch that wrote it");
            assertArrayEquals(new int[7], other.read());
        }
    }

    private static Function kernel(Statement... statements) {
        List<Statement> body = new ArrayList<>(List.of(statements));
        body.add(new Statement.ReturnVoid());
        return new Function("main", new Type.FunctionType(Type.VOID, List.of()), new Region(body));
    }

    private static Expr gid() {
        return new Expr.InvocationId();
    }

    private static Expr i(long value) {
        return new Expr.ConstInt(I32, value);
    }

    private static Expr add(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.ADD, a, b);
    }
}
