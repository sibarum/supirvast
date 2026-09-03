package dev.supirvast.vastir.tools;

import dev.supirvast.vast.CoreToTruffle;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.type.Type;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Differential harness over a single {@code core} computation, exercising both backends from one body.
 *
 * <p>Given a body that computes into a {@code result} variable — an integer via {@link #run}, or an
 * {@code f32} compared as raw IEEE-754 bits via {@link #runFloat32} — it:
 * <ul>
 *   <li><b>CPU:</b> wraps the body in a value-returning function, lowers it with {@link CoreToTruffle}, and
 *       executes the Truffle AST to a concrete value.</li>
 *   <li><b>GPU:</b> wraps the same body in a {@code void} compute entry point, lowers it with
 *       {@link CoreToSpirv}, validates the SPIR-V, then checks it is a fixed point of the official
 *       {@code spirv-dis}/{@code spirv-as} round-trip — i.e. disassemble → assemble → disassemble yields the
 *       same text. That cross-checks our binary encoder against Khronos's own tools.</li>
 * </ul>
 *
 * <p>It deliberately does <em>not</em> execute the SPIR-V (that needs a Vulkan device or a CPU runtime such
 * as SwiftShader). It pins that the same computation lowers cleanly to both targets, runs on the CPU, and
 * produces canonical, valid SPIR-V on the GPU side.
 */
public final class DifferentialHarness {

    /** Outcome of a differential run. {@code gpuResult} is null when no Vulkan device is available. */
    public record Report(
            String name,
            int cpuResult,
            boolean spirvValid,
            String validationOutput,
            boolean disassemblyRoundTrips,
            boolean bytesIdentical,
            Integer gpuResult) {

        public boolean gpuExecuted() {
            return gpuResult != null;
        }

        /** True when the GPU ran and produced the same value as the CPU — the real differential. */
        public boolean gpuMatchesCpu() {
            return gpuResult != null && gpuResult == cpuResult;
        }

        /**
         * Everything that could be checked, passed — <em>including</em> the GPU/CPU differential.
         *
         * <p>This is deliberately {@code false} when no GPU was available: the differential is the whole
         * point of the harness, and a run that never reached hardware has not established agreement. Use
         * {@link #lowersCleanly()} when you only mean "valid, canonical SPIR-V", and {@link #gpuExecuted()}
         * to tell "the GPU disagreed" apart from "the GPU never ran".
         */
        public boolean ok() {
            return lowersCleanly() && gpuMatchesCpu();
        }

        /**
         * The backend-independent half: the module is valid SPIR-V and is a fixed point of the official
         * disassemble/assemble round-trip. True even with no GPU present, so it is not evidence of
         * GPU/CPU agreement on its own.
         */
        public boolean lowersCleanly() {
            return spirvValid && disassemblyRoundTrips;
        }
    }

    /**
     * How a NaN result is judged when comparing the two backends.
     *
     * <p>SPIR-V leaves the bit pattern of a produced NaN implementation-defined, so a GPU and the CPU may
     * legitimately return different NaN payloads for the same computation. The policy makes that choice
     * explicit rather than letting it decide a test by accident.
     */
    public enum NanPolicy {
        /** Any NaN equals any other NaN; every non-NaN value still compared bit-exactly. The sane default. */
        CANONICAL,
        /** NaN payload bits must match too — for pinning a specific device's exact behaviour. */
        STRICT_BITS
    }

    /**
     * Outcome of an {@code f32} differential. Both {@code cpuBits} and {@code gpuBits} are the raw IEEE-754
     * words as they left each backend; {@code gpuBits} is null when no Vulkan device was available.
     */
    public record FloatReport(
            String name,
            int cpuBits,
            boolean spirvValid,
            String validationOutput,
            boolean disassemblyRoundTrips,
            boolean bytesIdentical,
            Integer gpuBits) {

        public boolean gpuExecuted() {
            return gpuBits != null;
        }

        public float cpuValue() {
            return Float.intBitsToFloat(cpuBits);
        }

        /** The GPU's value, or null if it never ran. */
        public Float gpuValue() {
            return gpuBits == null ? null : Float.intBitsToFloat(gpuBits);
        }

        public boolean cpuIsNaN() {
            return Float.isNaN(cpuValue());
        }

        public boolean gpuIsNaN() {
            return gpuBits != null && Float.isNaN(Float.intBitsToFloat(gpuBits));
        }

        /**
         * True when exactly one backend produced a NaN — always a real disagreement, never excusable as a
         * payload difference, and the case a naive {@code ==} comparison silently passes.
         */
        public boolean nanDisagreement() {
            return gpuBits != null && cpuIsNaN() != gpuIsNaN();
        }

        /** True when both sides are NaN but their payload bits differ — permitted under {@link NanPolicy#CANONICAL}. */
        public boolean nanPayloadsDiffer() {
            return cpuIsNaN() && gpuIsNaN() && cpuBits != gpuBits.intValue();
        }

        /** Whether the GPU ran and agreed with the CPU under the given NaN policy. */
        public boolean gpuMatchesCpu(NanPolicy policy) {
            if (gpuBits == null) {
                return false;
            }
            if (cpuBits == gpuBits.intValue()) {
                return true;
            }
            return policy == NanPolicy.CANONICAL && cpuIsNaN() && gpuIsNaN();
        }

        /** As {@link #ok(NanPolicy)} under {@link NanPolicy#CANONICAL}. */
        public boolean ok() {
            return ok(NanPolicy.CANONICAL);
        }

        /**
         * Everything checkable passed, including the differential. Like {@link Report#ok()} this is
         * {@code false} when no GPU was available — a run that never reached hardware proves nothing.
         */
        public boolean ok(NanPolicy policy) {
            return lowersCleanly() && gpuMatchesCpu(policy);
        }

        /** The backend-independent half: valid, canonical SPIR-V. Not evidence of GPU/CPU agreement. */
        public boolean lowersCleanly() {
            return spirvValid && disassemblyRoundTrips;
        }

        /** A human-readable comparison, with both raw words — for assertion messages. */
        public String describe() {
            if (gpuBits == null) {
                return name + ": CPU " + cpuValue() + " (bits 0x" + Integer.toHexString(cpuBits)
                        + "); GPU did not run";
            }
            return name + ": CPU " + cpuValue() + " (bits 0x" + Integer.toHexString(cpuBits)
                    + ") vs GPU " + gpuValue() + " (bits 0x" + Integer.toHexString(gpuBits) + ")";
        }
    }

    private final NativeTools tools;
    private final VulkanCompute vulkan;

    public DifferentialHarness() {
        this(new NativeTools(), new VulkanCompute());
    }

    public DifferentialHarness(NativeTools tools, VulkanCompute vulkan) {
        this.tools = tools;
        this.vulkan = vulkan;
    }

    public boolean toolsAvailable() {
        return tools.isAvailable();
    }

    public boolean gpuAvailable() {
        return vulkan.isAvailable();
    }

    /**
     * Runs the differential. {@code body} must declare and compute {@code result} (an int); the harness adds
     * the appropriate terminator for each backend.
     */
    public Report run(String name, List<Statement> body, LocalVar result) {
        Expr read = new Expr.Read(result);
        Legs legs = runBothBackends(body, read, result.type(), read);
        return new Report(name, legs.cpuWord(), legs.validation().valid(), legs.validation().output(),
                legs.roundTrips(), legs.bytesIdentical(), legs.gpuWord());
    }

    /**
     * Runs the differential over an {@code f32} {@code result}, comparing raw IEEE-754 bits.
     *
     * <p>Both backends bitcast the result to {@code i32} before it leaves the shader, so each side yields the
     * same 32-bit word and the comparison is exact — it distinguishes {@code +0.0} from {@code -0.0} and
     * treats every NaN payload as significant, neither of which {@code ==} on floats would catch. See
     * {@link FloatReport} for how NaN is then judged.
     */
    public FloatReport runFloat32(String name, List<Statement> body, LocalVar result) {
        if (!(result.type() instanceof Type.Float f) || f.width() != 32) {
            throw new IllegalArgumentException(
                    "runFloat32 requires an f32 result variable, got " + result.type());
        }
        // Bitcast on both legs: the output SSBO member is an i32, and routing the CPU leg through the same
        // Bitcast keeps the two paths symmetric instead of relying on a Java float carrier.
        Expr bits = new Expr.Bitcast(new Expr.Read(result), Type.int32());
        Legs legs = runBothBackends(body, bits, Type.int32(), bits);
        return new FloatReport(name, legs.cpuWord(), legs.validation().valid(), legs.validation().output(),
                legs.roundTrips(), legs.bytesIdentical(), legs.gpuWord());
    }

    /** The two backend legs of one differential run, before either report shape is built. */
    private record Legs(int cpuWord, NativeTools.ValidationResult validation, boolean roundTrips,
            boolean bytesIdentical, Integer gpuWord) {}

    private Legs runBothBackends(List<Statement> body, Expr cpuResult, Type cpuReturnType, Expr gpuResult) {
        int cpuWord = (Integer) new CoreToTruffle().lower(cpuFunction(body, cpuResult, cpuReturnType)).call();

        byte[] spirv = new CoreToSpirv().lower(gpuModule(body, gpuResult)).toByteArray();
        NativeTools.ValidationResult validation = tools.validate(spirv);

        // Round-trip through the official tools. We don't require byte/text identity to our own output:
        // SPIR-V ids are arbitrary, and spirv-as renumbers them, so the first disassemble→assemble pass may
        // renumber. The meaningful checks are that our binary reassembles to a still-valid module and reaches
        // a stable normal form (a second round-trip is a no-op).
        byte[] reassembled = tools.assemble(tools.disassemble(spirv, "--no-header"));
        boolean reassembledValid = tools.validate(reassembled).valid();
        String normalForm = tools.disassemble(reassembled, "--no-header");
        String secondPass = tools.disassemble(tools.assemble(normalForm), "--no-header");
        boolean roundTrips = reassembledValid && normalForm.equals(secondPass);

        // The real differential: run the same shader on the GPU and read the result back.
        Integer gpuWord = vulkan.isAvailable() ? vulkan.execute(spirv, "main") : null;

        return new Legs(cpuWord, validation, roundTrips, Arrays.equals(spirv, reassembled), gpuWord);
    }

    private static Function cpuFunction(List<Statement> body, Expr resultExpr, Type returnType) {
        List<Statement> statements = new ArrayList<>(body);
        statements.add(new Statement.Return(resultExpr));
        return new Function("compute", new Type.FunctionType(returnType, List.of()), new Region(statements));
    }

    private static CoreModule gpuModule(List<Statement> body, Expr resultExpr) {
        List<Statement> statements = new ArrayList<>(body);
        statements.add(new Statement.StoreResult(resultExpr)); // write to the output SSBO
        statements.add(new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), new Region(statements));
        return new CoreModule().addEntryPoint(EntryPoint.compute(main, 1, 1, 1));
    }
}
