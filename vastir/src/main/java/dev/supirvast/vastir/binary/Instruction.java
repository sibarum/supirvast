package dev.supirvast.vastir.binary;

import dev.supirvast.vastir.spirv.Op;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * A single SPIR-V instruction being assembled into binary words.
 *
 * <p>Operands are appended in grammar order via the typed {@code id}/{@code literal}/{@code enumValue}/
 * {@code string} methods, then serialized as a word stream: word 0 packs {@code wordCount << 16 | opcode},
 * followed by the operand words. Encoding is independent of endianness here (words are ints); byte order is
 * applied once at module serialization.
 *
 * @see SpirvModule
 */
public final class Instruction {

    private final Op op;
    private final List<Integer> operandWords = new ArrayList<>();

    public Instruction(Op op) {
        this.op = op;
    }

    public Op op() {
        return op;
    }

    /** Appends a single {@code <id>} operand (result id, result-type id, or id reference). */
    public Instruction id(int id) {
        if (id <= 0) {
            throw new IllegalArgumentException("SPIR-V ids are positive: " + id);
        }
        operandWords.add(id);
        return this;
    }

    /** Appends a single-word literal integer. */
    public Instruction literal(int value) {
        operandWords.add(value);
        return this;
    }

    /** Appends an enum/bitmask operand by its numeric grammar value (e.g. {@code Capability.Shader.value()}). */
    public Instruction enumValue(int value) {
        operandWords.add(value);
        return this;
    }

    /**
     * Appends a SPIR-V literal string: UTF-8 bytes, NUL-terminated, then zero-padded to a whole number of
     * 32-bit words, each word packed little-endian (byte 0 in the low bits) per the SPIR-V spec.
     */
    public Instruction string(String text) {
        byte[] utf8 = text.getBytes(StandardCharsets.UTF_8);
        int wordCount = utf8.length / 4 + 1; // +1 guarantees at least one NUL terminator and padding
        for (int w = 0; w < wordCount; w++) {
            int word = 0;
            for (int b = 0; b < 4; b++) {
                int idx = w * 4 + b;
                if (idx < utf8.length) {
                    word |= (utf8[idx] & 0xFF) << (b * 8);
                }
            }
            operandWords.add(word);
        }
        return this;
    }

    /** Number of operand words appended so far (excludes the opcode word). */
    public int operandWordCount() {
        return operandWords.size();
    }

    /** Serializes this instruction to its word stream, including the leading opcode/word-count word. */
    public int[] toWords() {
        int totalWords = operandWords.size() + 1;
        if (totalWords > 0xFFFF) {
            throw new IllegalStateException(op + " exceeds the 65535-word instruction limit");
        }
        int[] words = new int[totalWords];
        words[0] = (totalWords << 16) | (op.opcode() & 0xFFFF);
        for (int i = 0; i < operandWords.size(); i++) {
            words[i + 1] = operandWords.get(i);
        }
        return words;
    }
}
