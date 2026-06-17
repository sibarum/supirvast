package dev.supirvast.vastir.binary;

import dev.supirvast.vastir.spirv.AddressingModel;
import dev.supirvast.vastir.spirv.Capability;
import dev.supirvast.vastir.spirv.MemoryModel;
import dev.supirvast.vastir.spirv.Op;
import dev.supirvast.vastir.spirv.Spirv;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class SpirvModuleTest {

    @Test
    void headerIsWellFormed() {
        SpirvModule module = new SpirvModule();
        module.emit(Op.OpCapability).enumValue(Capability.Shader.value());
        module.emit(Op.OpMemoryModel)
                .enumValue(AddressingModel.Logical.value())
                .enumValue(MemoryModel.GLSL450.value());

        int[] words = module.toWords();
        assertEquals(Spirv.MAGIC_NUMBER, words[0]);
        assertEquals(0x00010600, words[1]); // SPIR-V 1.6 version word
        assertEquals(Spirv.VERSION_WORD, words[1]);
        assertEquals(SpirvModule.GENERATOR_MAGIC, words[2]);
        assertEquals(module.idBound(), words[3]);
        assertEquals(0, words[4]); // schema
    }

    @Test
    void instructionPacksWordCountAndOpcode() {
        Instruction cap = new Instruction(Op.OpCapability).enumValue(Capability.Shader.value());
        int[] words = cap.toWords();
        assertEquals(2, words.length);
        assertEquals((2 << 16) | Op.OpCapability.opcode(), words[0]);
        assertEquals(Capability.Shader.value(), words[1]);
    }

    @Test
    void literalStringIsNulTerminatedAndPadded() {
        // "main" is 4 bytes, so it needs a 5th NUL byte -> 2 words, NUL-padded, little-endian.
        Instruction insn = new Instruction(Op.OpName).id(1).string("main");
        int[] words = insn.toWords();
        assertEquals(4, words.length);          // opcode word + id + 2 string words
        assertEquals('m' | ('a' << 8) | ('i' << 16) | ('n' << 24), words[2]);
        assertEquals(0x00000000, words[3]);     // NUL terminator + padding
    }

    @Test
    void byteSerializationIsLittleEndianRoundTrip() {
        SpirvModule module = new SpirvModule();
        module.emit(Op.OpCapability).enumValue(Capability.Shader.value());

        int[] words = module.toWords();
        byte[] bytes = module.toByteArray();
        assertEquals(words.length * Integer.BYTES, bytes.length);

        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int[] readBack = new int[words.length];
        for (int i = 0; i < words.length; i++) {
            readBack[i] = buffer.getInt();
        }
        assertArrayEquals(words, readBack);
    }

    @Test
    void idsAreAllocatedSequentiallyAndBoundTracks() {
        SpirvModule module = new SpirvModule();
        assertEquals(1, module.allocateId());
        assertEquals(2, module.allocateId());
        assertEquals(3, module.idBound());
    }
}
