package dev.supirvast.vastir.lower;

import dev.supirvast.vastir.binary.Instruction;
import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.PushConstants;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.SharedArray;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.spirv.Capability;
import dev.supirvast.vastir.spirv.Op;
import dev.supirvast.vastir.type.Type;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What the lowering emits for workgroup memory and barriers, and what it refuses. That the emitted modules
 * are valid, and compute the right thing, is {@code WorkgroupMemoryTest}'s job, which has {@code spirv-val}
 * and a device; this pins the refusals, which need neither.
 */
class WorkgroupLoweringTest {

    private static final Type.Int I32 = Type.int32();
    private static final Type.FunctionType VOID_FN = new Type.FunctionType(Type.VOID, List.of());

    /** {@code tile[lid] = lid; barrier; tile[0] = tile[1]} */
    private static Function tileKernel(SharedArray tile) {
        return new Function("main", VOID_FN, Region.of(
                new Statement.SharedStore(tile, new Expr.LocalInvocationId(), new Expr.LocalInvocationId()),
                new Statement.Barrier(),
                new Statement.SharedStore(tile, new Expr.ConstInt(I32, 0),
                        new Expr.SharedLoad(tile, new Expr.ConstInt(I32, 1))),
                new Statement.ReturnVoid()));
    }

    private static List<Instruction> lower(Function kernel, SpirvTarget target) {
        return new CoreToSpirv().lower(new CoreModule().addEntryPoint(EntryPoint.compute(kernel, 64, 1, 1)), target)
                .instructions();
    }

    private static long count(List<Instruction> instructions, Op op) {
        return instructions.stream().filter(i -> i.op() == op).count();
    }

    @Test
    void emitsAWorkgroupArrayAndABarrier() {
        List<Instruction> spirv = lower(tileKernel(new SharedArray("tile", I32, 64)), SpirvTarget.unconstrained());
        assertEquals(1, count(spirv, Op.OpTypeArray));
        assertEquals(1, count(spirv, Op.OpControlBarrier));
        // The workgroup variable joins the builtin (LocalInvocationId) in the entry point's interface.
        Instruction entry = spirv.stream().filter(i -> i.op() == Op.OpEntryPoint).findFirst().orElseThrow();
        int[] words = entry.toWords();
        int nameWords = ("main".length() + 4) / 4;
        assertEquals(2, words.length - 3 - nameWords, "interface ids on the entry point");
    }

    @Test
    void twoArraysOfOneShapeShareATypeButNotAVariable() {
        SharedArray a = new SharedArray("a", I32, 8);
        SharedArray b = new SharedArray("b", I32, 8);
        Function kernel = new Function("main", VOID_FN, Region.of(
                new Statement.SharedStore(a, new Expr.ConstInt(I32, 0), new Expr.SharedLoad(b, new Expr.ConstInt(I32, 0))),
                new Statement.ReturnVoid()));
        List<Instruction> spirv = lower(kernel, SpirvTarget.unconstrained());
        assertEquals(1, count(spirv, Op.OpTypeArray));
        // Two workgroup variables; the rest of the OpVariables are none (no function locals, no builtins).
        assertEquals(2, count(spirv, Op.OpVariable));
    }

    @Test
    void workgroupMemoryIsCheckedAgainstTheTargetLikeACapability() {
        Function kernel = tileKernel(new SharedArray("tile", I32, 64));   // 256 bytes
        assertDoesNotThrow(() -> lower(kernel, SpirvTarget.unconstrained().withWorkgroupMemoryLimit(256)));
        CapabilityException over = assertThrows(CapabilityException.class,
                () -> lower(kernel, SpirvTarget.unconstrained().withWorkgroupMemoryLimit(255)));
        assertTrue(over.getMessage().contains("256 bytes"), over.getMessage());
    }

    @Test
    void aVec3CountsAsFourComponents() {
        assertEquals(16L * 10, new SharedArray("v", new Type.Vector(Type.float32(), 3), 10).bytes());
    }

    @Test
    void workgroupFeaturesAreComputeOnly() {
        Function fragment = new Function("frag", VOID_FN, Region.of(new Statement.Barrier(), new Statement.ReturnVoid()));
        CoreModule module = new CoreModule().addEntryPoint(EntryPoint.of(fragment, ShaderStage.FRAGMENT));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> new CoreToSpirv().lower(module));
        assertTrue(e.getMessage().contains("compute only"), e.getMessage());
    }

    @Test
    void aBarrierInACalleeIsRefused() {
        Function helper = new Function("helper", new Type.FunctionType(I32, List.of()),
                Region.of(new Statement.Barrier(), new Statement.Return(new Expr.ConstInt(I32, 0))));
        Function main = new Function("main", VOID_FN, Region.of(
                new Statement.DeclareVar(new LocalVar("x", I32),
                        new Expr.Call(helper, List.of())),
                new Statement.ReturnVoid()));
        CoreModule module = new CoreModule().addFunction(helper).addEntryPoint(EntryPoint.compute(main, 64, 1, 1));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> new CoreToSpirv().lower(module));
        assertTrue(e.getMessage().contains("only an entry point's own body"), e.getMessage());
    }

    @Test
    void aDivergentBarrierIsRefused() {
        Function kernel = new Function("main", VOID_FN, Region.of(
                new Statement.If(new Expr.Binary(BinaryOp.LESS_THAN, new Expr.InvocationId(), new Expr.ConstInt(I32, 3)),
                        Region.of(new Statement.Barrier()), Region.of()),
                new Statement.ReturnVoid()));
        assertThrows(IllegalArgumentException.class, () -> lower(kernel, SpirvTarget.unconstrained()));
    }

    /** The count is carried in the push-constant block, so the two cannot both be the module's. */
    @Test
    void theInvocationCountAndOtherPushConstantsExcludeEachOther() {
        PushConstants block = PushConstants.of("scale", I32);
        Function kernel = new Function("main", VOID_FN, Region.of(
                new Statement.If(new Expr.Binary(BinaryOp.LESS_THAN, block.read(0), new Expr.InvocationCount()),
                        Region.of(new Statement.ReturnVoid()), Region.of()),
                new Statement.ReturnVoid()));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> lower(kernel, SpirvTarget.unconstrained()));
        assertTrue(e.getMessage().contains("push constants"), e.getMessage());
    }

    private static Function sharedFloatAtomic(AtomicOp op) {
        SharedArray floats = new SharedArray("f", Type.float32(), 4);
        return new Function("main", VOID_FN, Region.of(
                new Statement.SharedAtomicUpdate(op, floats, new Expr.ConstInt(I32, 0),
                        new Expr.ConstFloat(Type.float32(), 1)),
                new Statement.ReturnVoid()));
    }

    /**
     * A float add on workgroup memory needs the capability a buffer's would, and the workgroup feature on top:
     * a device-derived target can have the first without the second, and must then refuse.
     */
    @Test
    void aFloatAtomicOnWorkgroupMemoryNeedsItsCapabilityAndItsFeature() {
        Function add = sharedFloatAtomic(AtomicOp.ADD);
        assertEquals(1, count(lower(add, SpirvTarget.unconstrained()), Op.OpAtomicFAddEXT));

        CapabilityException noCapability = assertThrows(CapabilityException.class,
                () -> lower(add, SpirvTarget.restrictedTo(Set.of())));
        assertTrue(noCapability.getMessage().contains("AtomicFloat32AddEXT"), noCapability.getMessage());

        SpirvTarget bufferOnly = SpirvTarget.restrictedTo(Set.of(Capability.AtomicFloat32AddEXT))
                .withFeatures(Set.of(DeviceFeature.BUFFER_FLOAT32_ATOMIC_ADD));
        CapabilityException noFeature = assertThrows(CapabilityException.class, () -> lower(add, bufferOnly));
        assertTrue(noFeature.getMessage().contains("SHARED_FLOAT32_ATOMIC_ADD"), noFeature.getMessage());

        assertDoesNotThrow(() -> lower(add, bufferOnly.withFeatures(Set.of(DeviceFeature.SHARED_FLOAT32_ATOMIC_ADD))));
    }

    /** Float exchange is core SPIR-V, but on workgroup memory Vulkan still asks for a feature. */
    @Test
    void aFloatExchangeOnWorkgroupMemoryNeedsOnlyItsFeature() {
        Function exchange = sharedFloatAtomic(AtomicOp.EXCHANGE);
        assertDoesNotThrow(() -> lower(exchange, SpirvTarget.restrictedTo(Set.of())));
        assertThrows(CapabilityException.class,
                () -> lower(exchange, SpirvTarget.unconstrained().withFeatures(Set.of())));
    }

    /** The buffer side of the same split: a workgroup-only license does not cover a storage buffer. */
    @Test
    void aFloatAtomicOnABufferNeedsTheBufferFeature() {
        Buffer sums = new Buffer("sums", 0, Type.float32());
        Function add = new Function("main", VOID_FN, Region.of(
                new Statement.AtomicUpdate(AtomicOp.ADD, sums, new Expr.ConstInt(I32, 0),
                        new Expr.ConstFloat(Type.float32(), 1)),
                new Statement.ReturnVoid()));
        SpirvTarget sharedOnly = SpirvTarget.unconstrained().withFeatures(Set.of(DeviceFeature.SHARED_FLOAT32_ATOMIC_ADD));
        assertThrows(CapabilityException.class, () -> lower(add, sharedOnly));
    }

    @Test
    void floatAtomicsOnWorkgroupMemoryFollowTheBufferTable() {
        SharedArray floats = new SharedArray("f", Type.float32(), 4);
        Expr zero = new Expr.ConstInt(I32, 0);
        Expr one = new Expr.ConstFloat(Type.float32(), 1);
        assertThrows(IllegalArgumentException.class,
                () -> new Statement.SharedAtomicUpdate(AtomicOp.SUB, floats, zero, one));
        assertThrows(IllegalArgumentException.class, () -> new Statement.SharedAtomicCompareExchange(
                new LocalVar("old", Type.float32()), floats, zero, one, one));
    }
}
