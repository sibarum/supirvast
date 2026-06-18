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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * vastir-preview step 1 (de-risk): confirm the {@code core} IR can express <em>vertex attribute inputs</em> —
 * the model's per-vertex {@code position}/{@code normal} fed from vertex buffers — not just the
 * vertex-output / fragment-input varyings already covered by {@link VertexFragmentShaderTest}.
 *
 * <p>A vertex-stage {@link InterfaceVar#input} <em>is</em> a vertex attribute: a {@code location}-bound
 * variable in the {@code Input} storage class. {@code CoreToSpirv} already chooses storage class purely from
 * the {@link InterfaceVar.Direction} (not the stage), so this should lower without any IR change — this test
 * pins that. It mirrors the previewer's v1 vertex contract (location 0 = position {@code vec3}, location 1 =
 * normal {@code vec3}) and asserts the attributes come out as {@code Location}-decorated {@code Input}
 * variables, validate with {@code spirv-val}, and surface as real vertex inputs in cross-compiled GLSL.
 */
class VertexAttributeShaderTest {

    private static final Type.Float F32 = Type.float32();
    private static final Type.Vector VEC3 = new Type.Vector(F32, 3);
    private static final Type.Vector VEC4 = new Type.Vector(F32, 4);

    /**
     * {@code gl_Position = vec4(position, 1.0); vNormal = normal;} — reads two location-bound vertex
     * attributes, writes the position built-in, and passes the normal through as a location-0 varying.
     */
    private static byte[] vertexShaderWithAttributes() {
        InterfaceVar position = InterfaceVar.input("position", 0, VEC3);
        InterfaceVar normal = InterfaceVar.input("normal", 1, VEC3);
        InterfaceVar vNormal = InterfaceVar.output("vNormal", 0, VEC3);

        Expr clipPosition = new Expr.VectorConstruct(VEC4,
                List.of(new Expr.InterfaceRead(position), new Expr.ConstFloat(F32, 1.0)));
        Region body = Region.of(
                new Statement.BuiltinWrite(Builtin.POSITION, clipPosition),
                new Statement.InterfaceWrite(vNormal, new Expr.InterfaceRead(normal)),
                new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.VERTEX)))
                .toByteArray();
    }

    @Test
    void vertexAttributesLowerToLocationBoundInputs() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        byte[] spirv = vertexShaderWithAttributes();

        NativeTools.ValidationResult validation = tools.validate(spirv);
        assertTrue(validation.valid(),
                () -> "spirv-val rejected the vertex-attribute shader:\n" + validation.output());

        // The two attributes must be Input-storage OpVariables (a vec3 each), each decorated with its Location.
        String disasm = tools.disassemble(spirv);
        assertTrue(disasm.contains("_ptr_Input_v3float"),
                () -> "expected an Input pointer to vec3 for the vertex attributes:\n" + disasm);
        assertEquals(2, count(disasm, "OpVariable %\\S+ Input"),
                () -> "expected exactly two Input vertex-attribute variables:\n" + disasm);
        assertTrue(disasm.contains("Location 0") && disasm.contains("Location 1"),
                () -> "expected the attributes decorated Location 0 and Location 1:\n" + disasm);
        assertTrue(disasm.contains("BuiltIn Position"),
                () -> "expected gl_Position to be written:\n" + disasm);

        // And they should round-trip to GLSL as genuine vertex inputs at those locations.
        String glsl = tools.crossCompile(spirv, NativeTools.ShaderLanguage.GLSL);
        assertTrue(glsl.contains("location = 0") && glsl.contains("location = 1"),
                () -> "vertex GLSL should declare attribute inputs at locations 0 and 1:\n" + glsl);
    }

    private static int count(String haystack, String regex) {
        Matcher m = Pattern.compile(regex).matcher(haystack);
        int n = 0;
        while (m.find()) {
            n++;
        }
        return n;
    }
}
