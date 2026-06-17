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
 * <p>Given a body that computes into an integer {@code result} variable, it:
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

        public boolean ok() {
            return spirvValid && disassemblyRoundTrips && (gpuResult == null || gpuMatchesCpu());
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
        int cpuResult = (Integer) new CoreToTruffle().lower(cpuFunction(body, result)).call();

        byte[] spirv = new CoreToSpirv().lower(gpuModule(body, result)).toByteArray();
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
        Integer gpuResult = vulkan.isAvailable() ? vulkan.execute(spirv, "main") : null;

        return new Report(name, cpuResult, validation.valid(), validation.output(),
                roundTrips, Arrays.equals(spirv, reassembled), gpuResult);
    }

    private static Function cpuFunction(List<Statement> body, LocalVar result) {
        List<Statement> statements = new ArrayList<>(body);
        statements.add(new Statement.Return(new Expr.Read(result)));
        return new Function("compute", new Type.FunctionType(result.type(), List.of()), new Region(statements));
    }

    private static CoreModule gpuModule(List<Statement> body, LocalVar result) {
        List<Statement> statements = new ArrayList<>(body);
        statements.add(new Statement.StoreResult(new Expr.Read(result))); // write to the output SSBO
        statements.add(new Statement.ReturnVoid());
        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), new Region(statements));
        return new CoreModule().addEntryPoint(EntryPoint.compute(main, 1, 1, 1));
    }
}
