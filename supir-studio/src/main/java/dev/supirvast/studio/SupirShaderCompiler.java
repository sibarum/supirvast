package dev.supirvast.studio;

import dev.supirvast.supir.Supir;
import dev.supirvast.supir.SupirParseException;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.tools.NativeTools;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Compiles a Supir shader stage (the editor's text) through the SupirVast pipeline:
 *
 * <pre>Supir text → core IR (supir) → SPIR-V (CoreToSpirv) → spirv-val gate → high-level shader (spirv-cross)</pre>
 *
 * <p>{@link #compileVertex}/{@link #compileFragment} produce the GLSL 330 the studio links into its live GL
 * program (cross-compiled with explicit varying locations, so the two generated stages link by location —
 * spirv-cross names interface variables opaquely). {@link #exportStage} produces every distributable form of a
 * stage — SPIR-V binary and assembly, plus GLSL/HLSL/Metal — for the Export feature.
 *
 * <p>The vertex stage's MVP matrix is a push constant, which spirv-cross emits as a struct-typed
 * {@code uniform} with an opaque name (e.g. {@code _9._m0}). Since the dasum GL binding can't enumerate active
 * uniforms, {@link #compileVertex} recovers that name from the generated GLSL so the renderer can set it.
 *
 * <p>Every failure mode comes back as a result with {@code ok=false} and a message; nothing throws to the
 * caller.
 */
final class SupirShaderCompiler {

    private final NativeTools tools = new NativeTools();

    /**
     * The GLSL 330 the studio links ({@code ok}), or an error ({@code !ok}). For a vertex stage,
     * {@code mvpUniform} is the GL name of the MVP matrix uniform to set each frame (null for the fragment).
     */
    record Result(boolean ok, String glsl, String mvpUniform, String error) {
        static Result ok(String glsl, String mvpUniform) {
            return new Result(true, glsl, mvpUniform, null);
        }

        static Result error(String message) {
            return new Result(false, null, null, message);
        }
    }

    /** Every distributable form of one stage ({@code ok}), or an error ({@code !ok}). */
    record Export(boolean ok, byte[] spirv, String spirvAssembly, String glsl, String hlsl, String msl,
                  String error) {
        static Export error(String message) {
            return new Export(false, null, null, null, null, null, message);
        }
    }

    /** Whether the native SPIR-V toolchain is bundled for this platform; the Supir path is dead without it. */
    boolean toolsAvailable() {
        return tools.isAvailable();
    }

    /** Compiles a {@code fragment} shader; the result carries no MVP uniform (the fragment has no uniforms). */
    Result compileFragment(String supirSource) {
        return compile(supirSource, ShaderStage.FRAGMENT);
    }

    /** Compiles a {@code vertex} shader; the result's {@code mvpUniform} is the matrix uniform's GL name. */
    Result compileVertex(String supirSource) {
        return compile(supirSource, ShaderStage.VERTEX);
    }

    private Result compile(String supirSource, ShaderStage expected) {
        Lowered lowered = lower(supirSource, expected);
        if (lowered.error != null) {
            return Result.error(lowered.error);
        }
        String glsl;
        try {
            glsl = tools.crossCompile(lowered.spirv, NativeTools.ShaderLanguage.GLSL, 330);
        } catch (RuntimeException e) {
            return Result.error("cross-compile to GLSL failed: " + messageOf(e));
        }
        String mvpUniform = expected == ShaderStage.VERTEX ? matrixUniformName(glsl) : null;
        return Result.ok(glsl, mvpUniform);
    }

    /**
     * Compiles a stage to every distributable form: SPIR-V binary, SPIR-V assembly, and GLSL/HLSL/Metal. The
     * GLSL here is spirv-cross's default desktop profile (a clean standalone shader), not the studio's
     * link-specific GLSL 330.
     */
    Export exportStage(String supirSource, ShaderStage expected) {
        Lowered lowered = lower(supirSource, expected);
        if (lowered.error != null) {
            return Export.error(lowered.error);
        }
        try {
            return new Export(true,
                    lowered.spirv,
                    tools.disassemble(lowered.spirv),
                    tools.crossCompile(lowered.spirv, NativeTools.ShaderLanguage.GLSL),
                    tools.crossCompile(lowered.spirv, NativeTools.ShaderLanguage.HLSL),
                    tools.crossCompile(lowered.spirv, NativeTools.ShaderLanguage.MSL),
                    null);
        } catch (RuntimeException e) {
            return Export.error("cross-compile failed: " + messageOf(e));
        }
    }

    /** Result of lowering Supir to validated SPIR-V: the bytes, or an error (exactly one is non-null). */
    private record Lowered(byte[] spirv, String error) {}

    private Lowered lower(String supirSource, ShaderStage expected) {
        CoreModule module;
        try {
            module = Supir.parseModule(supirSource);
        } catch (SupirParseException e) {
            return new Lowered(null, e.getMessage());
        } catch (RuntimeException e) {
            return new Lowered(null, "parse failed: " + messageOf(e));
        }

        String stageError = requireSingleStage(module, expected);
        if (stageError != null) {
            return new Lowered(null, stageError);
        }

        byte[] spirv;
        try {
            spirv = new CoreToSpirv().lower(module).toByteArray();
        } catch (RuntimeException e) {
            return new Lowered(null, "lowering to SPIR-V failed: " + messageOf(e));
        }

        if (!tools.isAvailable()) {
            return new Lowered(null, "native SPIR-V toolchain not bundled for this platform");
        }
        NativeTools.ValidationResult validation = tools.validate(spirv);
        if (!validation.valid()) {
            return new Lowered(null, "spirv-val rejected the shader:\n" + validation.output());
        }
        return new Lowered(spirv, null);
    }

    private static String requireSingleStage(CoreModule module, ShaderStage expected) {
        if (module.entryPoints().size() != 1) {
            return "the studio expects exactly one entry point; found " + module.entryPoints().size();
        }
        ShaderStage stage = module.entryPoints().getFirst().stage();
        if (stage != expected) {
            return "expected a " + expected.name().toLowerCase() + " shader; this is a "
                    + stage.name().toLowerCase() + " shader";
        }
        return null;
    }

    // spirv-cross emits a push-constant mat4 either directly (uniform mat4 NAME;) or, more often, wrapped in a
    // struct: `struct S { mat4 M; }; uniform S I;`, queried in GL as `I.M`.
    private static final Pattern DIRECT_MAT4 = Pattern.compile("uniform\\s+mat4\\s+(\\w+)\\s*;");
    private static final Pattern STRUCT_UNIFORM = Pattern.compile("uniform\\s+(\\w+)\\s+(\\w+)\\s*;");

    /** Recovers the GL name of the (sole) mat4 uniform from cross-compiled GLSL, or null if none is found. */
    static String matrixUniformName(String glsl) {
        Matcher direct = DIRECT_MAT4.matcher(glsl);
        if (direct.find()) {
            return direct.group(1);
        }
        Matcher u = STRUCT_UNIFORM.matcher(glsl);
        while (u.find()) {
            String structName = u.group(1);
            String instance = u.group(2);
            Matcher body = Pattern.compile("struct\\s+" + Pattern.quote(structName) + "\\s*\\{([^}]*)\\}")
                    .matcher(glsl);
            if (body.find()) {
                Matcher member = Pattern.compile("mat4\\s+(\\w+)\\s*;").matcher(body.group(1));
                if (member.find()) {
                    return instance + "." + member.group(1);
                }
            }
        }
        return null;
    }

    private static String messageOf(RuntimeException e) {
        String m = e.getMessage();
        return (m == null || m.isBlank()) ? e.toString() : m;
    }
}
