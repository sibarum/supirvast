package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.binary.SpirvModule;
import dev.supirvast.vastir.spirv.AddressingModel;
import dev.supirvast.vastir.spirv.Capability;
import dev.supirvast.vastir.spirv.ExecutionMode;
import dev.supirvast.vastir.spirv.ExecutionModel;
import dev.supirvast.vastir.spirv.FunctionControl;
import dev.supirvast.vastir.spirv.MemoryModel;
import dev.supirvast.vastir.spirv.Op;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * End-to-end proof: the vastir emitter builds a minimal valid compute shader, which the bundled native
 * toolchain then validates, disassembles, and cross-compiles — all from one build. Skipped (not failed)
 * when native tools are not bundled, e.g. an offline build run with -Dvast.skipNativeTools=true.
 */
class NativeToolsPipelineTest {

    /** The smallest valid GLCompute shader: an empty {@code main} entry point. */
    private static byte[] minimalComputeShader() {
        SpirvModule m = new SpirvModule();
        int main = m.allocateId();
        int voidType = m.allocateId();
        int fnType = m.allocateId();
        int label = m.allocateId();

        m.emit(Op.OpCapability).enumValue(Capability.Shader.value());
        m.emit(Op.OpMemoryModel)
                .enumValue(AddressingModel.Logical.value())
                .enumValue(MemoryModel.GLSL450.value());
        m.emit(Op.OpEntryPoint)
                .enumValue(ExecutionModel.GLCompute.value())
                .id(main)
                .string("main");
        m.emit(Op.OpExecutionMode)
                .id(main)
                .enumValue(ExecutionMode.LocalSize.value())
                .literal(1).literal(1).literal(1);
        m.emit(Op.OpTypeVoid).id(voidType);
        m.emit(Op.OpTypeFunction).id(fnType).id(voidType);
        m.emit(Op.OpFunction)
                .id(voidType).id(main)
                .enumValue(FunctionControl.None.value())
                .id(fnType);
        m.emit(Op.OpLabel).id(label);
        m.emit(Op.OpReturn);
        m.emit(Op.OpFunctionEnd);
        return m.toByteArray();
    }

    @Test
    void emittedModuleValidatesDisassemblesAndCrossCompiles() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V tools not bundled");

        byte[] spirv = minimalComputeShader();

        NativeTools.ValidationResult validation = tools.validate(spirv);
        assertTrue(validation.valid(), () -> "spirv-val rejected emitter output:\n" + validation.output());

        String disassembly = tools.disassemble(spirv);
        assertTrue(disassembly.contains("OpEntryPoint"), () -> "unexpected disassembly:\n" + disassembly);

        String glsl = tools.crossCompile(spirv, NativeTools.ShaderLanguage.GLSL);
        assertFalse(glsl.isBlank(), "spirv-cross produced no GLSL");
    }
}
