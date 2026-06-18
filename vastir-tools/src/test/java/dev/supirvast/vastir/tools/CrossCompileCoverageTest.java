package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Cross-compile coverage for all three SPIRV-Cross targets. One {@code core} kernel ({@code out[i]=a[i]+b[i]})
 * is lowered to SPIR-V and cross-compiled to GLSL, HLSL, and MSL, checking each output is non-empty and carries
 * the language's expected shape. This locks in the "author once, get GLSL/HLSL/Metal" story for a real
 * data-parallel shader (only GLSL of a trivial shader was exercised before).
 */
class CrossCompileCoverageTest {

    private static final Buffer OUT = new Buffer("out", 0);
    private static final Buffer A = new Buffer("a", 1);
    private static final Buffer B = new Buffer("b", 2);

    private static byte[] vectorAddSpirv() {
        Expr gid = new Expr.InvocationId();
        Region body = Region.of(
                new Statement.BufferStore(OUT, gid,
                        new Expr.Binary(BinaryOp.ADD,
                                new Expr.BufferLoad(A, new Expr.InvocationId()),
                                new Expr.BufferLoad(B, new Expr.InvocationId()))),
                new Statement.ReturnVoid());
        Function kernel = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        return new CoreToSpirv()
                .lower(new CoreModule().addEntryPoint(EntryPoint.compute(kernel, 1, 1, 1)))
                .toByteArray();
    }

    @Test
    void kernelCrossCompilesToAllThreeLanguages() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V toolchain not bundled for this platform");

        byte[] spirv = vectorAddSpirv();
        assertTrue(tools.validate(spirv).valid(), "kernel must validate before cross-compiling");

        String glsl = tools.crossCompile(spirv, NativeTools.ShaderLanguage.GLSL);
        assertFalse(glsl.isBlank(), "no GLSL produced");
        assertTrue(glsl.contains("#version") && glsl.contains("main("),
                () -> "GLSL missing expected shape:\n" + glsl);

        String hlsl = tools.crossCompile(spirv, NativeTools.ShaderLanguage.HLSL);
        assertFalse(hlsl.isBlank(), "no HLSL produced");
        assertTrue(hlsl.contains("numthreads") && hlsl.contains("main("),
                () -> "HLSL missing expected shape:\n" + hlsl);

        String msl = tools.crossCompile(spirv, NativeTools.ShaderLanguage.MSL);
        assertFalse(msl.isBlank(), "no MSL produced");
        assertTrue(msl.contains("metal_stdlib") && msl.contains("kernel "),
                () -> "MSL missing expected shape:\n" + msl);
    }
}
