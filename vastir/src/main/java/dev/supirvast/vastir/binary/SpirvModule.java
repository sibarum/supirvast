package dev.supirvast.vastir.binary;

import dev.supirvast.vastir.spirv.Spirv;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * A SPIR-V module under construction: a monotonically allocated {@code <id>} space plus an ordered
 * instruction stream, serializable to the binary word format.
 *
 * <p>This is the low-level binary writer — the bottom of the lowering pipeline. Higher layers (the
 * structured IR) drive it; here we only guarantee a well-formed header and faithful word encoding. Semantic
 * validity (a memory model, entry points, a valid CFG) is the caller's responsibility and is checked
 * externally with {@code spirv-val}.
 */
public final class SpirvModule {

    /**
     * Generator's magic number written into the header. 0 is the reserved "unknown/unregistered" value;
     * tools may register a number in SPIRV-Headers' generator registry and set it here.
     */
    public static final int GENERATOR_MAGIC = 0;

    /** Number of words in the fixed SPIR-V header. */
    public static final int HEADER_WORDS = 5;

    private final List<Instruction> instructions = new ArrayList<>();
    private int nextId = 1; // ids are 1-based; 0 is never a valid result id

    /** Allocates and returns a fresh, unique {@code <id>}. */
    public int allocateId() {
        return nextId++;
    }

    /** The id bound written to the header: one past the largest allocated id. */
    public int idBound() {
        return nextId;
    }

    /** Appends an already-built instruction to the module's stream. */
    public SpirvModule add(Instruction instruction) {
        instructions.add(instruction);
        return this;
    }

    /** Convenience: creates, appends, and returns a new instruction for the given opcode. */
    public Instruction emit(dev.supirvast.vastir.spirv.Op op) {
        Instruction instruction = new Instruction(op);
        instructions.add(instruction);
        return instruction;
    }

    public List<Instruction> instructions() {
        return List.copyOf(instructions);
    }

    /** Assembles the full module (header + instruction stream) into its 32-bit word sequence. */
    public int[] toWords() {
        int bodyWords = 0;
        List<int[]> encoded = new ArrayList<>(instructions.size());
        for (Instruction instruction : instructions) {
            int[] words = instruction.toWords();
            encoded.add(words);
            bodyWords += words.length;
        }

        int[] module = new int[HEADER_WORDS + bodyWords];
        module[0] = Spirv.MAGIC_NUMBER;
        module[1] = Spirv.VERSION_WORD;
        module[2] = GENERATOR_MAGIC;
        module[3] = idBound();
        module[4] = 0; // schema, reserved

        int pos = HEADER_WORDS;
        for (int[] words : encoded) {
            System.arraycopy(words, 0, module, pos, words.length);
            pos += words.length;
        }
        return module;
    }

    /** Serializes the module to little-endian bytes, ready to write to a {@code .spv} file. */
    public byte[] toByteArray() {
        int[] words = toWords();
        ByteBuffer buffer = ByteBuffer.allocate(words.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int word : words) {
            buffer.putInt(word);
        }
        return buffer.array();
    }
}
