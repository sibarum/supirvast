package dev.supirvast.vastir.codegen;

import com.google.gson.annotations.SerializedName;

import java.util.List;

/**
 * Gson-mapped mirror of the official SPIR-V grammar (spirv.core.grammar.json).
 *
 * <p>Field shapes follow the grammar's JSON schema exactly; only the subset currently consumed by the
 * generators is modelled. Reserved words and snake_case keys are bridged with {@link SerializedName}.
 */
public final class Grammar {

    @SerializedName("magic_number") public String magicNumber;
    @SerializedName("major_version") public int majorVersion;
    @SerializedName("minor_version") public int minorVersion;
    public int revision;
    public List<Instruction> instructions;
    @SerializedName("operand_kinds") public List<OperandKind> operandKinds;

    public String spirvVersion() {
        return majorVersion + "." + minorVersion;
    }

    /** A single SPIR-V instruction definition. */
    public static final class Instruction {
        public String opname;
        @SerializedName("class") public String instructionClass;
        public int opcode;
        public List<Operand> operands;
        /** Earliest core version the op appears in, e.g. "1.0"; may be null for extension-only ops. */
        public String version;
        public String lastVersion;
        public List<String> capabilities;
        public List<String> extensions;
        /** Alternate opnames that share this opcode/semantics (often vendor-suffixed). */
        public List<String> aliases;
        /** True for unstable ops not yet in a ratified spec; gated out of the stable surface by default. */
        public boolean provisional;
    }

    /** One positional operand slot of an instruction. */
    public static final class Operand {
        /** References an {@link OperandKind#kind}. */
        public String kind;
        public String name;
        /** null = exactly one; "?" = optional (0..1); "*" = variadic (0..n). */
        public String quantifier;
    }

    /** A named operand kind: an enum, a literal, an id, or a composite. */
    public static final class OperandKind {
        /** One of: ValueEnum, BitEnum, Literal, Id, Composite. */
        public String category;
        public String kind;
        public List<Enumerant> enumerants;
    }

    /** One member of a ValueEnum/BitEnum operand kind. */
    public static final class Enumerant {
        public String enumerant;
        /** int for ValueEnum, hex string (e.g. "0x0002") for BitEnum. */
        public Object value;
        public List<Operand> parameters;
        public List<String> capabilities;
        public String version;
    }
}
