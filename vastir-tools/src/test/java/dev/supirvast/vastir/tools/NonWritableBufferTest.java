package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A storage buffer the module never stores to is decorated {@code NonWritable}; one it does store to is not.
 *
 * <h2>What the decoration is for</h2>
 *
 * <p>Vulkan requires it. Without the {@code fragmentStoresAndAtomics} device feature, every storage buffer a
 * <em>fragment</em> stage declares must carry {@code NonWritable}
 * ({@code VUID-RuntimeSpirv-NonWritable-06340}), and a ray-marched fragment reading its geometry out of a
 * buffer is exactly that shape. The alternative — enabling the feature — buys the same silence by promising
 * the implementation a write that never happens, and that promise costs real optimisation on tiled hardware.
 *
 * <h2>Why it is derived and not declared</h2>
 *
 * <p>A read-only flag on {@link Buffer} would put the decoration in the author's hands, and the direction
 * that goes wrong is the dangerous one: a buffer marked read-only and then written is a lie to the driver
 * that <b>nothing here catches</b>. {@code spirv-val} does not object, and the kernel goes on computing the
 * right answer until an implementation acts on what it was told. Deriving it from whether any statement
 * stores to the binding makes the decoration and the code the same fact.
 *
 * <p>So this test is two assertions and the second is the one with teeth: over-applying the decoration is
 * the failure mode that would be silent.
 */
class NonWritableBufferTest {

    private static final Type.Float F32 = Type.float32();
    private static final Type.Vector VEC4 = new Type.Vector(F32, 4);

    /** {@code OpDecorate}, and the {@code NonWritable} decoration — SPIR-V's own numbering. */
    private static final int OP_DECORATE = 71;
    private static final int NON_WRITABLE = 24;

    /** A fragment that only loads from its buffer: {@code fragColor = vec4(data[0], 0, 0, 1);} */
    private static byte[] readingFragment() {
        Buffer data = new Buffer("data", 0, F32);
        InterfaceVar fragColor = InterfaceVar.output("fragColor", 0, VEC4);
        Expr first = new Expr.BufferLoad(data, new Expr.ConstInt(Type.int32(), 0));
        Region body = Region.of(
                new Statement.InterfaceWrite(fragColor, new Expr.VectorConstruct(VEC4, List.of(
                        first, new Expr.ConstFloat(F32, 0), new Expr.ConstFloat(F32, 0),
                        new Expr.ConstFloat(F32, 1)))),
                new Statement.ReturnVoid());
        return lower(body, ShaderStage.FRAGMENT);
    }

    /**
     * A compute kernel that stores: {@code data[0] = data[0];}
     *
     * <p>A self-assignment, because what is under test is whether a {@code BufferStore} exists at all, and
     * writing something more interesting would only add ways for the test to fail for another reason.
     */
    private static byte[] writingKernel() {
        Buffer data = new Buffer("data", 0, F32);
        Expr zero = new Expr.ConstInt(Type.int32(), 0);
        Region body = Region.of(
                new Statement.BufferStore(data, zero, new Expr.BufferLoad(data, zero)),
                new Statement.ReturnVoid());
        return lower(body, ShaderStage.COMPUTE);
    }

    /** A kernel whose only store is nested inside a branch — the walk has to descend into regions. */
    private static byte[] conditionallyWritingKernel() {
        Buffer data = new Buffer("data", 0, F32);
        Expr zero = new Expr.ConstInt(Type.int32(), 0);
        Expr condition = new Expr.Binary(dev.supirvast.vastir.core.BinaryOp.LESS_THAN,
                new Expr.BufferLoad(data, zero), new Expr.ConstFloat(F32, 1));
        Region body = Region.of(
                new Statement.If(condition,
                        Region.of(new Statement.BufferStore(data, zero, new Expr.ConstFloat(F32, 2))),
                        Region.of()),
                new Statement.ReturnVoid());
        return lower(body, ShaderStage.COMPUTE);
    }

    private static byte[] lower(Region body, ShaderStage stage) {
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.of(main, stage)))
                .toByteArray();
    }

    @Test
    void aBufferThatIsOnlyLoadedIsDecoratedNonWritable() {
        assertTrue(declaresNonWritable(readingFragment()),
                "a fragment that only reads its storage buffer must decorate it NonWritable, or the pipeline "
                        + "is invalid without the fragmentStoresAndAtomics device feature");
    }

    @Test
    void aBufferThatIsStoredToIsNotDecorated() {
        assertFalse(declaresNonWritable(writingKernel()),
                "a kernel that writes its buffer must not claim the buffer is NonWritable — nothing in this "
                        + "toolchain catches that lie, which is exactly why it must not be told");
        assertFalse(declaresNonWritable(conditionallyWritingKernel()),
                "the store is inside an if, so a scan that did not descend into regions would decorate a "
                        + "buffer it can write");
    }

    /** And the writing kernel still validates and runs — the decoration is not the only thing that changed. */
    @Test
    void theWritingKernelStillValidates() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        NativeTools.ValidationResult validation = tools.validate(writingKernel());
        assertTrue(validation.valid(), () -> "spirv-val rejected the writing kernel:\n" + validation.output());
    }

    /**
     * Scan the instruction stream for {@code OpDecorate <id> NonWritable}.
     *
     * <p>Reading the words rather than the cross-compiled source: spirv-cross renders the decoration as a
     * {@code readonly} qualifier in GLSL and as nothing at all in some targets, so its absence from generated
     * source would not be evidence of its absence from the module.
     */
    private static boolean declaresNonWritable(byte[] spirv) {
        for (int word = 5; word < spirv.length / 4; ) {
            int instruction = wordAt(spirv, word);
            int wordCount = instruction >>> 16;
            int opcode = instruction & 0xFFFF;
            if (wordCount == 0) {
                return false;                       // malformed rather than looping forever
            }
            // OpDecorate is <target id> <decoration> [literals]; the decoration is the second operand.
            if (opcode == OP_DECORATE && wordCount >= 3 && wordAt(spirv, word + 2) == NON_WRITABLE) {
                return true;
            }
            word += wordCount;
        }
        return false;
    }

    private static int wordAt(byte[] spirv, int word) {
        int i = word * 4;
        return (spirv[i] & 0xFF) | ((spirv[i + 1] & 0xFF) << 8)
                | ((spirv[i + 2] & 0xFF) << 16) | ((spirv[i + 3] & 0xFF) << 24);
    }
}
