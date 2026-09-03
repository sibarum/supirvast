package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Builtin;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.PushConstants;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Pins the emitted std430 layout decorations by reading them back out of the official disassembly.
 *
 * <p>These are the numbers that a wrong-but-valid module gets wrong. {@code spirv-val} cannot catch a bad
 * {@code ArrayStride} or member {@code Offset} — the module is well-formed, it just describes the wrong memory
 * — so the values themselves have to be asserted.
 */
class LayoutDecorationTest {

    private static final Type.Float F32 = Type.float32();
    private static final Type.Int I32 = Type.int32();
    private static final Type.Int I64 = new Type.Int(64, true);

    private static final Pattern ARRAY_STRIDE = Pattern.compile("Decorate\\s+\\S+\\s+ArrayStride\\s+(\\d+)");
    private static final Pattern MEMBER_OFFSET =
            Pattern.compile("MemberDecorate\\s+\\S+\\s+(\\d+)\\s+Offset\\s+(\\d+)");

    private static byte[] kernelWith(Buffer buffer) {
        Region body = Region.of(
                new Statement.BufferStore(buffer, new Expr.ConstInt(I32, 0),
                        new Expr.BufferLoad(buffer, new Expr.ConstInt(I32, 0))),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.compute(main, 1, 1, 1)))
                .toByteArray();
    }

    private static byte[] vertexReading(PushConstants block, List<Type> memberTypes) {
        List<Statement> statements = new java.util.ArrayList<>();
        for (int i = 0; i < memberTypes.size(); i++) {
            statements.add(new Statement.DeclareVar(
                    new LocalVar("m" + i, memberTypes.get(i)), block.read(i)));
        }
        statements.add(new Statement.BuiltinWrite(Builtin.POSITION,
                new Expr.VectorConstruct(new Type.Vector(F32, 4), List.of(
                        new Expr.ConstFloat(F32, 0), new Expr.ConstFloat(F32, 0),
                        new Expr.ConstFloat(F32, 0), new Expr.ConstFloat(F32, 1)))));
        statements.add(new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()),
                new Region(statements));
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.VERTEX)))
                .toByteArray();
    }

    /** The single {@code ArrayStride} in the disassembly of a one-buffer kernel. */
    private static int arrayStrideOf(String disassembly) {
        Matcher m = ARRAY_STRIDE.matcher(disassembly);
        assertTrue(m.find(), () -> "no ArrayStride decoration found in:\n" + disassembly);
        return Integer.parseInt(m.group(1));
    }

    private static int memberOffsetOf(String disassembly, int member) {
        Matcher m = MEMBER_OFFSET.matcher(disassembly);
        while (m.find()) {
            if (Integer.parseInt(m.group(1)) == member) {
                return Integer.parseInt(m.group(2));
            }
        }
        throw new AssertionError("no Offset decoration for member " + member + " in:\n" + disassembly);
    }

    /** A scalar buffer strides by its own width — the pre-existing behaviour, held in place. */
    @Test
    void scalarBufferStrides() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        assertEquals(4, arrayStrideOf(tools.disassemble(kernelWith(new Buffer("i", 1, I32)))),
                "an i32 buffer strides by 4");
        assertEquals(8, arrayStrideOf(tools.disassemble(kernelWith(new Buffer("l", 1, I64)))),
                "an i64 buffer strides by 8");
    }

    /**
     * A {@code vec4} buffer element strides by 16, not by 4. This is the regression: the old code fell
     * through to {@code Integer.BYTES} and produced a valid module describing overlapping elements.
     */
    @Test
    void vectorBufferStridesByItsFullSize() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        byte[] spirv = kernelWith(new Buffer("v", 1, new Type.Vector(F32, 4)));
        assertEquals(16, arrayStrideOf(tools.disassemble(spirv)), "a vec4 buffer strides by 16");
        assertTrue(tools.validate(spirv).valid(),
                () -> "spirv-val rejected the vec4 buffer kernel:\n" + tools.validate(spirv).output());
    }

    /** A {@code vec2} strides by 8, and a {@code vec3} pads up to its 16-byte alignment. */
    @Test
    void vec2AndVec3StrideCorrectly() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        assertEquals(8, arrayStrideOf(tools.disassemble(kernelWith(
                new Buffer("v2", 1, new Type.Vector(F32, 2))))), "a vec2 buffer strides by 8");
        assertEquals(16, arrayStrideOf(tools.disassemble(kernelWith(
                new Buffer("v3", 1, new Type.Vector(F32, 3))))),
                "a vec3 element pads from 12 up to its 16-byte alignment");
    }

    /**
     * A 64-bit push-constant member occupies 8 bytes, so the member after it starts at offset 8. The old
     * {@code default -> 4} put it at 4, silently shifting every later member.
     */
    @Test
    void sixtyFourBitPushConstantMemberShiftsLaterMembersByEight() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        PushConstants block = new PushConstants(List.of(
                new PushConstants.Member("big", I64),
                new PushConstants.Member("small", I32)));
        String disassembly = tools.disassemble(vertexReading(block, List.of(I64, I32)));

        assertEquals(0, memberOffsetOf(disassembly, 0), "the first member starts at 0");
        assertEquals(8, memberOffsetOf(disassembly, 1),
                "an i64 first member is 8 bytes wide, so the i32 after it starts at 8");
    }

    /** A mat4's declared MatrixStride must equal the column stride its size was computed from. */
    @Test
    void matrixStrideMatchesItsColumnAlignment() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        PushConstants block = new PushConstants(List.of(
                new PushConstants.Member("mvp", Type.mat4()),
                new PushConstants.Member("tail", I32)));
        String disassembly = tools.disassemble(vertexReading(block, List.of(Type.mat4(), I32)));

        assertTrue(disassembly.contains("MatrixStride 16"),
                () -> "expected a 16-byte MatrixStride for a mat4:\n" + disassembly);
        assertEquals(0, memberOffsetOf(disassembly, 0), "the matrix starts at 0");
        assertEquals(64, memberOffsetOf(disassembly, 1),
                "a mat4 is 4 columns x 16 bytes, so the member after it starts at 64");
    }
}
