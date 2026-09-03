package dev.supirvast.vastir.tools;

import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.MathFn;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The {@code f32} half of the differential: the same float computation on both backends, compared as raw
 * IEEE-754 bits rather than with {@code ==}.
 *
 * <p>Bit comparison is the point. It separates {@code +0.0} from {@code -0.0}, refuses to let a NaN on one
 * backend match a number on the other, and surfaces a differing NaN payload instead of hiding it — all cases
 * a float {@code ==} silently passes.
 */
class FloatDifferentialTest {

    private static final Type.Float F32 = Type.float32();

    /** See {@link DifferentialHarnessTest} — same flag, same reason. */
    private static final boolean REQUIRE_GPU = Boolean.getBoolean("supirvast.requireGpu");

    private static Expr f(double value) {
        return new Expr.ConstFloat(F32, value);
    }

    private static Expr bin(BinaryOp op, Expr lhs, Expr rhs) {
        return new Expr.Binary(op, lhs, rhs);
    }

    /**
     * Runs an f32 body and asserts the two backends agree bit-for-bit (NaN payloads excepted, per
     * {@link DifferentialHarness.NanPolicy#CANONICAL}). Reports as skipped, never as a pass, when no GPU is
     * present.
     */
    private DifferentialHarness.FloatReport run(String name, List<Statement> body, LocalVar result) {
        DifferentialHarness harness = new DifferentialHarness();
        assumeTrue(harness.toolsAvailable(), "native SPIR-V tools not bundled");
        DifferentialHarness.FloatReport report = harness.runFloat32(name, body, result);
        assertTrue(report.spirvValid(),
                () -> "spirv-val rejected " + name + ":\n" + report.validationOutput());
        assertTrue(report.disassemblyRoundTrips(),
                () -> name + ": our SPIR-V round-trip through spirv-dis/spirv-as is unstable");

        if (!report.gpuExecuted()) {
            if (REQUIRE_GPU) {
                fail(name + ": -Dsupirvast.requireGpu=true but no Vulkan device was available, "
                        + "so the f32 differential never ran");
            }
            assumeTrue(false, name + ": no Vulkan device — f32 differential not run");
        }

        // A NaN on one side and a number on the other is never a payload technicality.
        assertFalse(report.nanDisagreement(),
                () -> "exactly one backend produced NaN — " + report.describe());
        assertTrue(report.gpuMatchesCpu(DifferentialHarness.NanPolicy.CANONICAL),
                () -> "f32 bits differ — " + report.describe());
        return report;
    }

    /** A plain arithmetic chain must come back with identical bits, not merely a close value. */
    @Test
    void arithmeticAgreesBitExactly() {
        LocalVar result = new LocalVar("result", F32);
        List<Statement> body = List.of(
                new Statement.DeclareVar(result, f(0.1)),
                new Statement.Assign(result, bin(BinaryOp.MUL, new Expr.Read(result), f(3.0))),
                new Statement.Assign(result, bin(BinaryOp.SUB, new Expr.Read(result), f(0.3))));
        DifferentialHarness.FloatReport report = run("f32-arithmetic", body, result);
        // 0.1f*3 - 0.3f is not 0 in binary32; pinning the exact bits is the whole point of the comparison.
        assertEquals(Float.floatToRawIntBits(0.1f * 3.0f - 0.3f), report.cpuBits(),
                () -> "CPU f32 arithmetic drifted from Java's binary32: " + report.describe());
    }

    /** {@code sqrt} is a GLSL.std.450 call on the GPU and {@code Math.sqrt} on the CPU — they must still agree. */
    @Test
    void sqrtAgreesBitExactly() {
        LocalVar result = new LocalVar("result", F32);
        List<Statement> body = List.of(
                new Statement.DeclareVar(result, new Expr.MathCall(MathFn.SQRT, F32, List.of(f(2.0)))));
        DifferentialHarness.FloatReport report = run("f32-sqrt", body, result);
        assertEquals(Float.floatToRawIntBits((float) Math.sqrt(2.0)), report.cpuBits(),
                () -> "CPU sqrt drifted from Java's binary32: " + report.describe());
    }

    /**
     * Negative zero. {@code -0.0f == 0.0f} is true in Java, so an {@code ==} based differential cannot tell
     * these apart; the bit comparison can, and must not report them as equal.
     */
    @Test
    void negativeZeroIsDistinguishedFromPositiveZero() {
        LocalVar result = new LocalVar("result", F32);
        List<Statement> body = List.of(
                new Statement.DeclareVar(result, bin(BinaryOp.MUL, f(0.0), f(-1.0))));
        DifferentialHarness.FloatReport report = run("f32-negative-zero", body, result);
        assertEquals(Float.floatToRawIntBits(-0.0f), report.cpuBits(),
                () -> "expected -0.0 bits, got " + report.describe());
        assertTrue(report.cpuBits() != Float.floatToRawIntBits(0.0f),
                "the comparison must not collapse -0.0 onto +0.0");
    }

    /** Division by zero: an infinity, which both backends must produce identically. */
    @Test
    void infinityAgreesBitExactly() {
        LocalVar result = new LocalVar("result", F32);
        List<Statement> body = List.of(
                new Statement.DeclareVar(result, bin(BinaryOp.DIV, f(1.0), f(0.0))));
        DifferentialHarness.FloatReport report = run("f32-infinity", body, result);
        assertEquals(Float.floatToRawIntBits(Float.POSITIVE_INFINITY), report.cpuBits(),
                () -> "expected +Inf bits, got " + report.describe());
    }

    /**
     * A genuine NaN, from {@code sqrt} of a negative. Both backends must agree it is NaN; the payload bits
     * are implementation-defined in SPIR-V, so a difference there is recorded rather than failed.
     */
    @Test
    void nanIsNaNOnBothBackends() {
        LocalVar result = new LocalVar("result", F32);
        List<Statement> body = List.of(
                new Statement.DeclareVar(result, new Expr.MathCall(MathFn.SQRT, F32, List.of(f(-1.0)))));
        DifferentialHarness.FloatReport report = run("f32-nan", body, result);
        assertTrue(report.cpuIsNaN(), () -> "sqrt(-1) should be NaN on the CPU: " + report.describe());
        assertTrue(report.gpuIsNaN(), () -> "sqrt(-1) should be NaN on the GPU: " + report.describe());
        if (report.nanPayloadsDiffer()) {
            // Permitted, but worth seeing: it means the two targets mint different NaNs.
            System.out.println("[differential] NaN payloads differ (allowed) — " + report.describe());
        }
    }

    /** A NaN and a number must never be judged equal, whatever the policy. */
    @Test
    void nanNeverMatchesANumber() {
        DifferentialHarness.FloatReport mixed = new DifferentialHarness.FloatReport(
                "synthetic", Float.floatToRawIntBits(Float.NaN), true, "", true, true,
                Float.floatToRawIntBits(1.0f));
        assertTrue(mixed.nanDisagreement(), "one NaN side and one numeric side is a disagreement");
        assertFalse(mixed.gpuMatchesCpu(DifferentialHarness.NanPolicy.CANONICAL));
        assertFalse(mixed.gpuMatchesCpu(DifferentialHarness.NanPolicy.STRICT_BITS));
    }

    /** The two NaN policies differ only on payload bits, and CANONICAL is the lenient one. */
    @Test
    void nanPolicyGovernsPayloadDifferences() {
        int quiet = Float.floatToRawIntBits(Float.NaN);
        DifferentialHarness.FloatReport payloads = new DifferentialHarness.FloatReport(
                "synthetic", quiet, true, "", true, true, quiet | 0x1234);
        assertTrue(payloads.nanPayloadsDiffer());
        assertFalse(payloads.nanDisagreement(), "both sides are NaN, so this is not a disagreement");
        assertTrue(payloads.gpuMatchesCpu(DifferentialHarness.NanPolicy.CANONICAL));
        assertFalse(payloads.gpuMatchesCpu(DifferentialHarness.NanPolicy.STRICT_BITS));
    }

    /** A report with no GPU leg must not read as a pass — the regression that made this suite vacuous. */
    @Test
    void absentGpuIsNotAPass() {
        DifferentialHarness.FloatReport noGpu = new DifferentialHarness.FloatReport(
                "synthetic", Float.floatToRawIntBits(1.0f), true, "", true, true, null);
        assertTrue(noGpu.lowersCleanly(), "the SPIR-V half still passed");
        assertFalse(noGpu.gpuExecuted());
        assertFalse(noGpu.ok(), "no GPU leg means no established agreement, so ok() must be false");
    }
}
