package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.binary.SpirvModule;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
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
 * Proves the full new layer: author a shader at the core IR level, lower it through {@link CoreToSpirv}, and
 * have the bundled native toolchain validate, disassemble, and cross-compile the result. Skipped when native
 * tools are not bundled.
 */
class CoreLoweringPipelineTest {

    private static byte[] minimalComputeFromCore() {
        Type.FunctionType voidFn = new Type.FunctionType(Type.VOID, List.of());
        Function main = new Function("main", voidFn, Region.of(new Statement.ReturnVoid()));
        CoreModule module = new CoreModule().addEntryPoint(EntryPoint.compute(main, 1, 1, 1));
        SpirvModule spirv = new CoreToSpirv().lower(module);
        return spirv.toByteArray();
    }

    /** {@code int acc=0,i=0; while (i<10){acc=acc+i; i=i+1;} if (acc<5){acc=0;}} */
    private static byte[] controlFlowFromCore() {
        Type.Int i32 = Type.int32();
        LocalVar acc = new LocalVar("acc", i32);
        LocalVar i = new LocalVar("i", i32);
        Expr zero = new Expr.ConstInt(i32, 0);

        Region loopBody = Region.of(
                new Statement.Assign(acc, new Expr.Binary(BinaryOp.ADD, new Expr.Read(acc), new Expr.Read(i))),
                new Statement.Assign(i, new Expr.Binary(BinaryOp.ADD, new Expr.Read(i), new Expr.ConstInt(i32, 1))));

        Region body = Region.of(
                new Statement.DeclareVar(acc, zero),
                new Statement.DeclareVar(i, zero),
                new Statement.While(new Expr.Binary(BinaryOp.LESS_THAN, new Expr.Read(i), new Expr.ConstInt(i32, 10)), loopBody),
                new Statement.If(new Expr.Binary(BinaryOp.LESS_THAN, new Expr.Read(acc), new Expr.ConstInt(i32, 5)),
                        Region.of(new Statement.Assign(acc, zero)), Region.of()),
                new Statement.ReturnVoid());

        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), body);
        CoreModule module = new CoreModule().addEntryPoint(EntryPoint.compute(main, 8, 1, 1));
        return new CoreToSpirv().lower(module).toByteArray();
    }

    @Test
    void coreLoweringValidatesAndCrossCompiles() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V tools not bundled");

        byte[] spirv = minimalComputeFromCore();

        NativeTools.ValidationResult validation = tools.validate(spirv);
        assertTrue(validation.valid(), () -> "spirv-val rejected CoreToSpirv output:\n" + validation.output());

        assertTrue(tools.disassemble(spirv).contains("OpEntryPoint"));
        assertFalse(tools.crossCompile(spirv, NativeTools.ShaderLanguage.GLSL).isBlank());
    }

    /** {@code int result=0,i=0; while (i<10){result=result+i; i=i+1;} buffer[0]=result;} writing an SSBO. */
    private static byte[] outputBufferShader() {
        Type.Int i32 = Type.int32();
        LocalVar result = new LocalVar("result", i32);
        LocalVar i = new LocalVar("i", i32);
        Region loop = Region.of(
                new Statement.Assign(result, new Expr.Binary(BinaryOp.ADD, new Expr.Read(result), new Expr.Read(i))),
                new Statement.Assign(i, new Expr.Binary(BinaryOp.ADD, new Expr.Read(i), new Expr.ConstInt(i32, 1))));
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), Region.of(
                new Statement.DeclareVar(result, new Expr.ConstInt(i32, 0)),
                new Statement.DeclareVar(i, new Expr.ConstInt(i32, 0)),
                new Statement.While(new Expr.Binary(BinaryOp.LESS_THAN, new Expr.Read(i), new Expr.ConstInt(i32, 10)), loop),
                new Statement.StoreResult(new Expr.Read(result)),
                new Statement.ReturnVoid()));
        CoreModule module = new CoreModule().addEntryPoint(EntryPoint.compute(main, 1, 1, 1));
        return new CoreToSpirv().lower(module).toByteArray();
    }

    @Test
    void outputBufferShaderValidates() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V tools not bundled");

        byte[] spirv = outputBufferShader();
        NativeTools.ValidationResult validation = tools.validate(spirv);
        assertTrue(validation.valid(),
                () -> "spirv-val rejected SSBO-output shader:\n" + validation.output());
        // The buffer write survives cross-compilation.
        assertFalse(tools.crossCompile(spirv, NativeTools.ShaderLanguage.GLSL).isBlank());
    }

    @Test
    void structuredControlFlowValidatesAndCrossCompiles() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V tools not bundled");

        // A shader using constants, local variables, arithmetic, a comparison, and structured while + if.
        byte[] spirv = controlFlowFromCore();

        NativeTools.ValidationResult validation = tools.validate(spirv);
        assertTrue(validation.valid(),
                () -> "spirv-val rejected control-flow lowering:\n" + validation.output());

        String disassembly = tools.disassemble(spirv);
        assertTrue(disassembly.contains("OpLoopMerge"), () -> "expected a loop:\n" + disassembly);
        assertTrue(disassembly.contains("OpSelectionMerge"), () -> "expected a selection:\n" + disassembly);
        assertFalse(tools.crossCompile(spirv, NativeTools.ShaderLanguage.GLSL).isBlank());
    }
}
