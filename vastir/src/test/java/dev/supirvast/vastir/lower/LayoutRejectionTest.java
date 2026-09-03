package dev.supirvast.vastir.lower;

import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.PushConstants;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.Builtin;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Layout decisions that used to be guessed at are now refused.
 *
 * <p>The dangerous property of a wrong {@code ArrayStride} or member {@code Offset} is that it produces
 * <em>valid</em> SPIR-V — {@code spirv-val} accepts it and the GPU reads the wrong memory. Nothing downstream
 * can catch it, so an element type with no defined layout has to fail here, at lowering time.
 */
class LayoutRejectionTest {

    private static final Type.Float F32 = Type.float32();

    /** A compute module writing one element of {@code buffer}, enough to force the buffer's stride decoration. */
    private static CoreModule kernelWriting(Buffer buffer) {
        Region body = Region.of(
                new Statement.BufferStore(buffer, new Expr.ConstInt(Type.int32(), 0),
                        new Expr.BufferLoad(buffer, new Expr.ConstInt(Type.int32(), 0))),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreModule().addEntryPoint(EntryPoint.compute(main, 1, 1, 1));
    }

    /**
     * A vertex shader that reads member 0 of {@code pushConstants}. The block is discovered by walking the
     * IR, so the read is what brings it into the module — and what forces its member offsets to be computed.
     */
    private static CoreModule vertexReadingMember0(PushConstants pushConstants, Type memberType) {
        Region body = Region.of(
                new Statement.DeclareVar(new LocalVar("read", memberType), pushConstants.read(0)),
                new Statement.BuiltinWrite(Builtin.POSITION,
                        new Expr.VectorConstruct(new Type.Vector(F32, 4), List.of(
                                new Expr.ConstFloat(F32, 0), new Expr.ConstFloat(F32, 0),
                                new Expr.ConstFloat(F32, 0), new Expr.ConstFloat(F32, 1)))),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.VERTEX));
    }

    /** Scalar buffer elements are the documented case and must keep working. */
    @Test
    void scalarBufferElementsStillLower() {
        assertDoesNotThrow(() -> new CoreToSpirv().lower(kernelWriting(new Buffer("i", 1, Type.int32()))));
        assertDoesNotThrow(() -> new CoreToSpirv().lower(kernelWriting(new Buffer("f", 1, F32))));
    }

    /** A vector buffer element has a well-defined std430 stride, so it lowers rather than being rejected. */
    @Test
    void vectorBufferElementsLower() {
        assertDoesNotThrow(() -> new CoreToSpirv()
                .lower(kernelWriting(new Buffer("v", 1, new Type.Vector(F32, 4)))));
    }

    /** A bool element has no defined array stride — it must be refused, not silently strided as 4 bytes. */
    @Test
    void boolBufferElementIsRejected() {
        IllegalArgumentException thrown = assertThrows(IllegalArgumentException.class,
                () -> new CoreToSpirv().lower(kernelWriting(new Buffer("b", 1, new Type.Bool()))));
        assertTrue(thrown.getMessage().contains("array stride"),
                () -> "message should name the missing stride, got: " + thrown.getMessage());
    }

    /** Likewise a bool push-constant member, which would previously have been sized at 4 bytes. */
    @Test
    void boolPushConstantMemberIsRejected() {
        IllegalArgumentException thrown = assertThrows(IllegalArgumentException.class,
                () -> new CoreToSpirv().lower(vertexReadingMember0(
                        PushConstants.of("flag", new Type.Bool()), new Type.Bool())));
        assertTrue(thrown.getMessage().contains("push-constant layout"),
                () -> "message should name the missing layout, got: " + thrown.getMessage());
    }

    /** An integer width outside {8,16,32,64} has no SPIR-V type and must not be emitted. */
    @Test
    void oddIntegerWidthIsRejected() {
        IllegalArgumentException thrown = assertThrows(IllegalArgumentException.class,
                () -> new CoreToSpirv().lower(kernelWriting(new Buffer("odd", 1, new Type.Int(7, true)))));
        assertTrue(thrown.getMessage().contains("width"),
                () -> "message should name the bad width, got: " + thrown.getMessage());
    }
}
