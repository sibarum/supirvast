package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.Builtin;
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
 * {@link Builtin#FRAG_DEPTH} — a fragment shader that replaces its own depth.
 *
 * <p>The built-in a ray-marched fragment cannot do without: it draws over a fullscreen triangle whose vertices
 * carry one fixed depth, so without this it can only contribute that constant and occludes a rasterised mesh
 * at a flat plane instead of at its own geometry.
 *
 * <p>Two things are checked, and the second is the one that is easy to get wrong. Writing the built-in must
 * decorate the variable {@code BuiltIn FragDepth}; it must <em>also</em> declare the {@code DepthReplacing}
 * execution mode on the entry point, without which the module is invalid SPIR-V rather than merely slow. And
 * the mode must appear only when the shader actually writes the built-in, because it is not free: it tells the
 * implementation the depth test cannot be settled before the fragment shader runs, so every fragment shader
 * that carried it needlessly would lose early-z for nothing.
 */
class FragDepthShaderTest {

    private static final Type.Float F32 = Type.float32();
    private static final Type.Vector VEC4 = new Type.Vector(F32, 4);

    /** {@code OpExecutionMode}, and the {@code DepthReplacing} mode operand — SPIR-V's own numbering. */
    private static final int OP_EXECUTION_MODE = 16;
    private static final int MODE_DEPTH_REPLACING = 12;

    private static Expr f(double v) {
        return new Expr.ConstFloat(F32, v);
    }

    /** {@code fragColor = vec4(1, 0, 0, 1); gl_FragDepth = 0.25;} */
    private static byte[] depthWritingFragment() {
        InterfaceVar fragColor = InterfaceVar.output("fragColor", 0, VEC4);
        Region body = Region.of(
                new Statement.InterfaceWrite(fragColor,
                        new Expr.VectorConstruct(VEC4, List.of(f(1), f(0), f(0), f(1)))),
                new Statement.BuiltinWrite(Builtin.FRAG_DEPTH, f(0.25)),
                new Statement.ReturnVoid());
        return lower(body);
    }

    /** The same shader with the depth write removed — the control for the execution-mode check. */
    private static byte[] plainFragment() {
        InterfaceVar fragColor = InterfaceVar.output("fragColor", 0, VEC4);
        Region body = Region.of(
                new Statement.InterfaceWrite(fragColor,
                        new Expr.VectorConstruct(VEC4, List.of(f(1), f(0), f(0), f(1)))),
                new Statement.ReturnVoid());
        return lower(body);
    }

    private static byte[] lower(Region body) {
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.FRAGMENT)))
                .toByteArray();
    }

    @Test
    void aDepthWritingFragmentValidatesAndCrossCompiles() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        byte[] spirv = depthWritingFragment();
        NativeTools.ValidationResult validation = tools.validate(spirv);
        // The assertion that earns this test. A FragDepth store with no DepthReplacing execution mode is not
        // a warning and not a slow path -- spirv-val rejects the module outright, and a driver handed it
        // behaves in whatever way it likes.
        assertTrue(validation.valid(), () -> "spirv-val rejected the depth-writing fragment:\n"
                + validation.output());

        String glsl = tools.crossCompile(spirv, NativeTools.ShaderLanguage.GLSL);
        assertTrue(glsl.contains("gl_FragDepth"),
                () -> "the fragment GLSL should write gl_FragDepth:\n" + glsl);
        assertFalse(tools.crossCompile(spirv, NativeTools.ShaderLanguage.HLSL).isBlank(), "no HLSL");
        assertFalse(tools.crossCompile(spirv, NativeTools.ShaderLanguage.MSL).isBlank(), "no MSL");
    }

    @Test
    void depthReplacingIsDeclaredExactlyWhenTheShaderWritesDepth() {
        assertTrue(declaresDepthReplacing(depthWritingFragment()),
                "a fragment that writes gl_FragDepth must declare DepthReplacing");
        assertFalse(declaresDepthReplacing(plainFragment()),
                "a fragment that writes no depth must not declare DepthReplacing — the mode costs early-z, "
                        + "so declaring it unconditionally would make every fragment shader pay for a feature "
                        + "only some of them use");
    }

    /**
     * Scan the module's instruction stream for {@code OpExecutionMode <entry> DepthReplacing}.
     *
     * <p>Reading the words rather than the cross-compiled GLSL, because spirv-cross renders the mode as a
     * layout qualifier only in some dialects and omits it where the target language implies it — so the
     * absence of a string in GLSL would not be evidence of the absence of the mode.
     */
    private static boolean declaresDepthReplacing(byte[] spirv) {
        // Header is five words; every instruction after it is {wordCount << 16 | opcode} followed by operands.
        for (int word = 5; word < spirv.length / 4; ) {
            int instruction = wordAt(spirv, word);
            int wordCount = instruction >>> 16;
            int opcode = instruction & 0xFFFF;
            if (wordCount == 0) {
                return false;                       // malformed rather than looping forever
            }
            // OpExecutionMode is <entry point id> <mode> [literals]; the mode is the instruction's second operand.
            if (opcode == OP_EXECUTION_MODE && wordCount >= 3 && wordAt(spirv, word + 2) == MODE_DEPTH_REPLACING) {
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
