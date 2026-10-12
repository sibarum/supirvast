package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Builtin;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.Texture;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * What a composed ubershader needs of vastir before any shadelet exists: an element's index carried from the vertex
 * stage to the fragment stage <em>flat</em>, as an integer, and used there to read a storage buffer; and a texture
 * sampled from a function the entry point calls, not from the entry point itself.
 */
class FlatVaryingTest {

    private static final Type.Float F32 = Type.float32();
    private static final Type.Int I32 = Type.int32();
    private static final Type.Vector VEC2 = new Type.Vector(F32, 2);
    private static final Type.Vector VEC4 = new Type.Vector(F32, 4);

    /** The index as the vertex buffer holds it (a float, as every canvas attribute is), and as a flat integer. */
    private static final InterfaceVar IN_INDEX = InterfaceVar.input("inIndex", 0, F32);
    private static final InterfaceVar IN_POS = InterfaceVar.input("inPos", 1, VEC2);
    private static final InterfaceVar V_INDEX_OUT = InterfaceVar.output("vIndex", 0, I32).asFlat();
    private static final InterfaceVar V_INDEX_IN = InterfaceVar.input("vIndex", 0, I32).asFlat();

    /** {@code gl_Position = vec4(inPos, 0, 1); vIndex = int(inIndex);} */
    private static byte[] vertex() {
        Expr pos = new Expr.InterfaceRead(IN_POS);
        Region body = Region.of(
                new Statement.BuiltinWrite(Builtin.POSITION, new Expr.VectorConstruct(VEC4, List.of(
                        new Expr.VectorExtract(pos, 0), new Expr.VectorExtract(pos, 1), f(0.0), f(1.0)))),
                new Statement.InterfaceWrite(V_INDEX_OUT, new Expr.Convert(new Expr.InterfaceRead(IN_INDEX), I32)),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.VERTEX)))
                .toByteArray();
    }

    /** {@code fragColor = vec4(styles[vIndex * 4 + 0], ..., styles[vIndex * 4 + 3]);} — a table read by element. */
    private static byte[] fragment() {
        // A set of its own: set 0 is a canvas's atlas.
        Buffer styles = new Buffer("styles", 2, 1, F32);
        InterfaceVar fragColor = InterfaceVar.output("fragColor", 0, VEC4);
        Expr base = new Expr.Binary(BinaryOp.MUL, new Expr.InterfaceRead(V_INDEX_IN), new Expr.ConstInt(I32, 4));
        List<Expr> rgba = new ArrayList<>();
        for (int k = 0; k < 4; k++) {
            rgba.add(new Expr.BufferLoad(styles, new Expr.Binary(BinaryOp.ADD, base, new Expr.ConstInt(I32, k))));
        }
        Region body = Region.of(
                new Statement.InterfaceWrite(fragColor, new Expr.VectorConstruct(VEC4, rgba)),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.FRAGMENT)))
                .toByteArray();
    }

    @Test
    void anIntegerIndexCrossesTheStagesFlat() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        for (byte[] spirv : List.of(vertex(), fragment())) {
            NativeTools.ValidationResult validation = tools.validate(spirv);
            assertTrue(validation.valid(), () -> "spirv-val rejected a stage:\n" + validation.output());
            String disasm = tools.disassemble(spirv);
            // Interface variables carry no debug names: the decoration and GLSL's qualifier are what to look for.
            assertTrue(disasm.contains(" Flat"), () -> "expected the index decorated Flat:\n" + disasm);
            String glsl = tools.crossCompile(spirv, NativeTools.ShaderLanguage.GLSL);
            assertTrue(glsl.matches("(?s).*flat (out|in) int .*"), () -> "expected a flat int varying in GLSL:\n" + glsl);
        }
        // In the SPIR-V: GLSL for OpenGL has no descriptor sets to show it in.
        String fragment = tools.disassemble(fragment());
        assertTrue(fragment.contains("DescriptorSet 2"), () -> "expected the table at set 2:\n" + fragment);
    }

    @Test
    void anInterpolatedIntegerFragmentInputIsRefusedByName() {
        InterfaceVar interpolated = InterfaceVar.input("vIndex", 0, I32);
        InterfaceVar fragColor = InterfaceVar.output("fragColor", 0, VEC4);
        Expr x = new Expr.Convert(new Expr.InterfaceRead(interpolated), F32);
        Region body = Region.of(
                new Statement.InterfaceWrite(fragColor, new Expr.VectorConstruct(VEC4, List.of(x, x, x, x))),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        CoreModule module = new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.FRAGMENT));

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new CoreToSpirv().lower(module));
        assertTrue(e.getMessage().contains("vIndex"), e::getMessage);
    }

    /**
     * {@code vec4 shade(vec2 uv, float lod) { return texture(tex, uv) * textureLod(tex, uv, lod); }}, called from
     * main: the texture is a global the called function reaches, as a shadelet's would be.
     */
    private static byte[] sampleFromCalledFunction() {
        Texture tex = new Texture("tex", 0, 0);
        InterfaceVar uv = InterfaceVar.input("vUv", 0, VEC2);
        InterfaceVar fragColor = InterfaceVar.output("fragColor", 0, VEC4);
        Expr implicit = new Expr.SampleTexture(tex, new Expr.Param(0, VEC2));
        Expr explicit = new Expr.SampleTexture(tex, new Expr.Param(0, VEC2), new Expr.Param(1, F32));
        Function shade = new Function("shade", new Type.FunctionType(VEC4, List.of(VEC2, F32)),
                Region.of(new Statement.Return(new Expr.Binary(BinaryOp.MUL, implicit, explicit))));
        Region body = Region.of(
                new Statement.InterfaceWrite(fragColor,
                        new Expr.Call(shade, List.of(new Expr.InterfaceRead(uv), f(1.0)))),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreToSpirv()
                .lower(new CoreModule().addFunction(shade).addEntryPoint(EntryPoint.of(main, ShaderStage.FRAGMENT)))
                .toByteArray();
    }

    @Test
    void aCalledFunctionSamplesATexture() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        byte[] spirv = sampleFromCalledFunction();
        NativeTools.ValidationResult validation = tools.validate(spirv);
        assertTrue(validation.valid(), () -> "spirv-val rejected sampling from a called function:\n"
                + validation.output());
        String disasm = tools.disassemble(spirv);
        assertTrue(disasm.contains("OpImageSampleImplicitLod") && disasm.contains("OpImageSampleExplicitLod")
                && disasm.contains("OpFunctionCall"), () -> "expected both samples behind a call:\n" + disasm);
        String glsl = tools.crossCompile(spirv, NativeTools.ShaderLanguage.GLSL);
        // Functions carry no debug names either: the samples must come before main, in the function it calls.
        String beforeMain = glsl.substring(0, Math.max(0, glsl.indexOf("void main")));
        assertTrue(beforeMain.contains("texture(") && beforeMain.contains("textureLod("),
                () -> "expected the samples in a called function:\n" + glsl);
    }

    private static Expr f(double v) {
        return new Expr.ConstFloat(F32, v);
    }
}
