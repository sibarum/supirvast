package dev.supirvast.vastir.lower;

import dev.supirvast.vastir.binary.Instruction;
import dev.supirvast.vastir.binary.SpirvModule;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Builtin;
import dev.supirvast.vastir.core.MathFn;
import dev.supirvast.vastir.core.PushConstants;
import dev.supirvast.vastir.core.Texture;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.UnaryOp;
import dev.supirvast.vastir.spirv.AddressingModel;
import dev.supirvast.vastir.spirv.BuiltIn;
import dev.supirvast.vastir.spirv.Capability;
import dev.supirvast.vastir.spirv.Decoration;
import dev.supirvast.vastir.spirv.ExecutionMode;
import dev.supirvast.vastir.spirv.ExecutionModel;
import dev.supirvast.vastir.spirv.FunctionControl;
import dev.supirvast.vastir.spirv.LoopControl;
import dev.supirvast.vastir.spirv.MemoryModel;
import dev.supirvast.vastir.spirv.Op;
import dev.supirvast.vastir.spirv.SelectionControl;
import dev.supirvast.vastir.spirv.Dim;
import dev.supirvast.vastir.spirv.ImageFormat;
import dev.supirvast.vastir.spirv.StorageClass;
import dev.supirvast.vastir.type.Type;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Lowers the core IR to a {@link SpirvModule}.
 *
 * <p>Builds the module as ordered logical sections (capabilities, memory model, entry points, execution
 * modes, annotations, types/constants/globals, functions) and concatenates them at the end. That ordering is
 * mandatory in SPIR-V — in particular annotations (decorations) must precede the type/variable definitions
 * they target, which they forward-reference by id. Types and constants are deduplicated by structural
 * equality; function-local {@code OpVariable}s are hoisted to the entry block; {@code if}/{@code while} become
 * {@code OpSelectionMerge}/{@code OpLoopMerge}; a {@code StoreResult} adds an output storage buffer (SSBO).
 */
public final class CoreToSpirv {

    /** Lowers with no capability restriction (emit whatever the kernel requires). */
    public SpirvModule lower(CoreModule module) {
        return lower(module, SpirvTarget.unconstrained());
    }

    /**
     * Lowers within {@code target}'s capability budget. The minimal capabilities the kernel requires are
     * derived from its types; if any falls outside the budget, lowering fails with a witness naming the
     * offending capabilities (so the orchestrator can reject the kernel rather than emit something the target
     * can't run).
     */
    public SpirvModule lower(CoreModule module, SpirvTarget target) {
        Builder b = new Builder();

        Map<Function, Integer> functionIds = new LinkedHashMap<>();
        for (Function function : module.functions()) {
            functionIds.put(function, b.allocateId());
        }

        OutputBuffer output = usesStoreResult(module) ? new OutputBuffer(b) : null;
        List<Buffer> buffers = collectBuffers(module);
        boolean invocationId = usesInvocationId(module);
        KernelResources kernel = (!buffers.isEmpty() || invocationId)
                ? new KernelResources(b, buffers, invocationId) : null;
        InterfaceUsage iface = collectInterface(module);
        InterfaceResources interfaceResources = iface.isEmpty()
                ? null : new InterfaceResources(b, iface.builtins(), iface.variables());
        TextureResources textures = iface.textures().isEmpty()
                ? null : new TextureResources(b, iface.textures());
        PushConstantResources pushConstants = iface.pushConstants() == null
                ? null : new PushConstantResources(b, iface.pushConstants());

        b.emit(b.capabilities, Op.OpCapability).enumValue(Capability.Shader.value());
        b.emit(b.memoryModel, Op.OpMemoryModel)
                .enumValue(target.addressingModel())
                .enumValue(target.memoryModel());

        emitEntryPoints(module, b, functionIds, output, kernel, interfaceResources, textures, pushConstants);
        emitExecutionModes(module, b, functionIds);

        TypeTable types = new TypeTable(b);
        ConstantTable constants = new ConstantTable(b, types);
        if (output != null) {
            output.declare(b, types, constants);
        }
        if (kernel != null) {
            kernel.declare(b, types, constants);
        }
        if (interfaceResources != null) {
            interfaceResources.declare(b, types);
        }
        if (textures != null) {
            textures.declare(b, types);
        }
        if (pushConstants != null) {
            pushConstants.declare(b, types, constants);
        }
        prepareGlobals(module, types, constants);

        for (Function function : module.functions()) {
            new FunctionLowering(b, types, constants, output, kernel, interfaceResources, textures,
                    pushConstants, functionIds)
                    .emit(function, functionIds.get(function));
        }

        // Capabilities the kernel actually requires (derived from declared types); emit them now (they sort to
        // the top of the module via the capabilities section). A required capability outside the target's
        // budget fails the lowering with a witness — the caller decides what to do (e.g. reject, fall back).
        List<Capability> required = new ArrayList<>();
        if (types.usesInt8()) {
            required.add(Capability.Int8);
        }
        if (types.usesInt16()) {
            required.add(Capability.Int16);
        }
        if (types.usesInt64()) {
            required.add(Capability.Int64);
        }
        if (types.usesFloat64()) {
            required.add(Capability.Float64);
        }
        List<Capability> disallowed = required.stream().filter(c -> !target.allows(c)).toList();
        if (!disallowed.isEmpty()) {
            throw new CapabilityException("kernel requires capabilities outside the target profile: " + disallowed);
        }
        for (Capability capability : required) {
            b.emit(b.capabilities, Op.OpCapability).enumValue(capability.value());
        }
        return b.finish();
    }

    private void emitEntryPoints(CoreModule module, Builder b, Map<Function, Integer> functionIds,
            OutputBuffer output, KernelResources kernel, InterfaceResources interfaceResources,
            TextureResources textures, PushConstantResources pushConstants) {
        for (EntryPoint entryPoint : module.entryPoints()) {
            Instruction instruction = b.emit(b.entryPoints, Op.OpEntryPoint)
                    .enumValue(executionModel(entryPoint.stage()))
                    .id(functionIds.get(entryPoint.function()))
                    .string(entryPoint.function().name());
            // SPIR-V >= 1.4 requires every module-scope variable used by the entry point in its interface.
            if (output != null) {
                instruction.id(output.variableId());
            }
            if (kernel != null) {
                for (int interfaceVar : kernel.interfaceVariables()) {
                    instruction.id(interfaceVar);
                }
            }
            if (interfaceResources != null) {
                for (int interfaceVar : interfaceResources.interfaceVariables()) {
                    instruction.id(interfaceVar);
                }
            }
            if (textures != null) {
                for (int textureVar : textures.interfaceVariables()) {
                    instruction.id(textureVar);
                }
            }
            if (pushConstants != null) {
                instruction.id(pushConstants.variableId());
            }
        }
    }

    private void emitExecutionModes(CoreModule module, Builder b, Map<Function, Integer> functionIds) {
        for (EntryPoint entryPoint : module.entryPoints()) {
            int functionId = functionIds.get(entryPoint.function());
            switch (entryPoint.stage()) {
                case COMPUTE -> {
                    EntryPoint.WorkgroupSize wg =
                            entryPoint.workgroupSize().orElse(new EntryPoint.WorkgroupSize(1, 1, 1));
                    b.emit(b.executionModes, Op.OpExecutionMode)
                            .id(functionId)
                            .enumValue(ExecutionMode.LocalSize.value())
                            .literal(wg.x()).literal(wg.y()).literal(wg.z());
                }
                case FRAGMENT -> b.emit(b.executionModes, Op.OpExecutionMode)
                        .id(functionId)
                        .enumValue(ExecutionMode.OriginUpperLeft.value());
                case VERTEX -> { /* no execution mode required */ }
            }
        }
    }

    private boolean usesStoreResult(CoreModule module) {
        for (Function function : module.functions()) {
            if (regionUsesStoreResult(function.body())) {
                return true;
            }
        }
        return false;
    }

    private boolean regionUsesStoreResult(Region region) {
        for (Statement statement : region.statements()) {
            boolean found = switch (statement) {
                case Statement.StoreResult ignored -> true;
                case Statement.If f -> regionUsesStoreResult(f.thenRegion()) || regionUsesStoreResult(f.elseRegion());
                case Statement.While w -> regionUsesStoreResult(w.body());
                default -> false;
            };
            if (found) {
                return true;
            }
        }
        return false;
    }

    private List<Buffer> collectBuffers(CoreModule module) {
        Map<Integer, Buffer> byBinding = new LinkedHashMap<>();
        for (Function function : module.functions()) {
            collectBuffers(function.body(), byBinding);
        }
        return List.copyOf(byBinding.values());
    }

    private void collectBuffers(Region region, Map<Integer, Buffer> out) {
        for (Statement statement : region.statements()) {
            switch (statement) {
                case Statement.BufferStore s -> {
                    out.putIfAbsent(s.buffer().binding(), s.buffer());
                    collectBuffers(s.index(), out);
                    collectBuffers(s.value(), out);
                }
                case Statement.Return r -> collectBuffers(r.value(), out);
                case Statement.StoreResult s -> collectBuffers(s.value(), out);
                case Statement.BuiltinWrite s -> collectBuffers(s.value(), out);
                case Statement.InterfaceWrite s -> collectBuffers(s.value(), out);
                case Statement.DeclareVar d -> collectBuffers(d.initializer(), out);
                case Statement.Assign a -> collectBuffers(a.value(), out);
                case Statement.If f -> {
                    collectBuffers(f.condition(), out);
                    collectBuffers(f.thenRegion(), out);
                    collectBuffers(f.elseRegion(), out);
                }
                case Statement.While w -> {
                    collectBuffers(w.condition(), out);
                    collectBuffers(w.body(), out);
                }
                case Statement.ReturnVoid ignored -> { }
            }
        }
    }

    private void collectBuffers(Expr expr, Map<Integer, Buffer> out) {
        switch (expr) {
            case Expr.BufferLoad l -> {
                out.putIfAbsent(l.buffer().binding(), l.buffer());
                collectBuffers(l.index(), out);
            }
            case Expr.Binary b -> {
                collectBuffers(b.lhs(), out);
                collectBuffers(b.rhs(), out);
            }
            case Expr.VectorConstruct vc -> vc.components().forEach(c -> collectBuffers(c, out));
            case Expr.VectorExtract ve -> collectBuffers(ve.vector(), out);
            case Expr.Bitcast bc -> collectBuffers(bc.operand(), out);
            case Expr.Convert cv -> collectBuffers(cv.operand(), out);
            case Expr.Unary u -> collectBuffers(u.operand(), out);
            case Expr.Call c -> c.arguments().forEach(a -> collectBuffers(a, out));
            case Expr.MathCall mc -> mc.args().forEach(a -> collectBuffers(a, out));
            case Expr.SampleTexture s -> collectBuffers(s.uv(), out);
            case Expr.PushConstantRead ignored -> { }
            case Expr.MatrixTimesVector m -> {
                collectBuffers(m.matrix(), out);
                collectBuffers(m.vector(), out);
            }
            case Expr.ConstInt ignored -> { }
            case Expr.ConstFloat ignored -> { }
            case Expr.ConstBool ignored -> { }
            case Expr.Read ignored -> { }
            case Expr.InvocationId ignored -> { }
            case Expr.BuiltinRead ignored -> { }
            case Expr.InterfaceRead ignored -> { }
            case Expr.Param ignored -> { }
        }
    }

    private boolean usesInvocationId(CoreModule module) {
        boolean[] found = {false};
        for (Function function : module.functions()) {
            scanForInvocationId(function.body(), found);
        }
        return found[0];
    }

    private void scanForInvocationId(Region region, boolean[] found) {
        for (Statement statement : region.statements()) {
            switch (statement) {
                case Statement.BufferStore s -> { scanForInvocationId(s.index(), found); scanForInvocationId(s.value(), found); }
                case Statement.Return r -> scanForInvocationId(r.value(), found);
                case Statement.StoreResult s -> scanForInvocationId(s.value(), found);
                case Statement.BuiltinWrite s -> scanForInvocationId(s.value(), found);
                case Statement.InterfaceWrite s -> scanForInvocationId(s.value(), found);
                case Statement.DeclareVar d -> scanForInvocationId(d.initializer(), found);
                case Statement.Assign a -> scanForInvocationId(a.value(), found);
                case Statement.If f -> { scanForInvocationId(f.condition(), found); scanForInvocationId(f.thenRegion(), found); scanForInvocationId(f.elseRegion(), found); }
                case Statement.While w -> { scanForInvocationId(w.condition(), found); scanForInvocationId(w.body(), found); }
                case Statement.ReturnVoid ignored -> { }
            }
        }
    }

    private void scanForInvocationId(Expr expr, boolean[] found) {
        switch (expr) {
            case Expr.InvocationId ignored -> found[0] = true;
            case Expr.BufferLoad l -> scanForInvocationId(l.index(), found);
            case Expr.Binary b -> { scanForInvocationId(b.lhs(), found); scanForInvocationId(b.rhs(), found); }
            case Expr.VectorConstruct vc -> vc.components().forEach(c -> scanForInvocationId(c, found));
            case Expr.VectorExtract ve -> scanForInvocationId(ve.vector(), found);
            case Expr.Bitcast bc -> scanForInvocationId(bc.operand(), found);
            case Expr.Convert cv -> scanForInvocationId(cv.operand(), found);
            case Expr.Unary u -> scanForInvocationId(u.operand(), found);
            case Expr.Call c -> c.arguments().forEach(a -> scanForInvocationId(a, found));
            case Expr.MathCall mc -> mc.args().forEach(a -> scanForInvocationId(a, found));
            case Expr.SampleTexture s -> scanForInvocationId(s.uv(), found);
            case Expr.PushConstantRead ignored -> { }
            case Expr.MatrixTimesVector m -> {
                scanForInvocationId(m.matrix(), found);
                scanForInvocationId(m.vector(), found);
            }
            case Expr.ConstInt ignored -> { }
            case Expr.ConstFloat ignored -> { }
            case Expr.ConstBool ignored -> { }
            case Expr.Read ignored -> { }
            case Expr.BuiltinRead ignored -> { }
            case Expr.InterfaceRead ignored -> { }
            case Expr.Param ignored -> { }
        }
    }

    /** The graphics built-ins, interface variables, textures, and push constants a module references. */
    private record InterfaceUsage(List<Builtin> builtins, List<InterfaceVar> variables, List<Texture> textures,
            PushConstants pushConstants) {
        boolean isEmpty() {
            return builtins.isEmpty() && variables.isEmpty();
        }
    }

    /** Mutable accumulator threaded through the interface scan. */
    private static final class InterfaceScan {
        final Set<Builtin> builtins = new LinkedHashSet<>();
        final Set<InterfaceVar> variables = new LinkedHashSet<>();
        final Set<Texture> textures = new LinkedHashSet<>();
        PushConstants pushConstants;   // a module uses at most one push-constant block
    }

    private InterfaceUsage collectInterface(CoreModule module) {
        InterfaceScan scan = new InterfaceScan();
        for (Function function : module.functions()) {
            scanInterface(function.body(), scan);
        }
        return new InterfaceUsage(List.copyOf(scan.builtins), List.copyOf(scan.variables),
                List.copyOf(scan.textures), scan.pushConstants);
    }

    private void scanInterface(Region region, InterfaceScan scan) {
        for (Statement statement : region.statements()) {
            switch (statement) {
                case Statement.BuiltinWrite s -> {
                    scan.builtins.add(s.builtin());
                    scanInterface(s.value(), scan);
                }
                case Statement.InterfaceWrite s -> {
                    scan.variables.add(s.variable());
                    scanInterface(s.value(), scan);
                }
                case Statement.BufferStore s -> {
                    scanInterface(s.index(), scan);
                    scanInterface(s.value(), scan);
                }
                case Statement.Return r -> scanInterface(r.value(), scan);
                case Statement.StoreResult s -> scanInterface(s.value(), scan);
                case Statement.DeclareVar d -> scanInterface(d.initializer(), scan);
                case Statement.Assign a -> scanInterface(a.value(), scan);
                case Statement.If f -> {
                    scanInterface(f.condition(), scan);
                    scanInterface(f.thenRegion(), scan);
                    scanInterface(f.elseRegion(), scan);
                }
                case Statement.While w -> {
                    scanInterface(w.condition(), scan);
                    scanInterface(w.body(), scan);
                }
                case Statement.ReturnVoid ignored -> { }
            }
        }
    }

    private void scanInterface(Expr expr, InterfaceScan scan) {
        switch (expr) {
            case Expr.BuiltinRead r -> scan.builtins.add(r.builtin());
            case Expr.InterfaceRead r -> scan.variables.add(r.variable());
            case Expr.SampleTexture s -> {
                scan.textures.add(s.texture());
                scanInterface(s.uv(), scan);
            }
            case Expr.PushConstantRead r -> scan.pushConstants = r.block();
            case Expr.MatrixTimesVector m -> {
                scanInterface(m.matrix(), scan);
                scanInterface(m.vector(), scan);
            }
            case Expr.BufferLoad l -> scanInterface(l.index(), scan);
            case Expr.Binary b -> {
                scanInterface(b.lhs(), scan);
                scanInterface(b.rhs(), scan);
            }
            case Expr.VectorConstruct vc -> vc.components().forEach(c -> scanInterface(c, scan));
            case Expr.VectorExtract ve -> scanInterface(ve.vector(), scan);
            case Expr.Bitcast bc -> scanInterface(bc.operand(), scan);
            case Expr.Convert cv -> scanInterface(cv.operand(), scan);
            case Expr.Unary u -> scanInterface(u.operand(), scan);
            case Expr.Call c -> c.arguments().forEach(a -> scanInterface(a, scan));
            case Expr.MathCall mc -> mc.args().forEach(a -> scanInterface(a, scan));
            case Expr.ConstInt ignored -> { }
            case Expr.ConstFloat ignored -> { }
            case Expr.ConstBool ignored -> { }
            case Expr.Read ignored -> { }
            case Expr.InvocationId ignored -> { }
            case Expr.Param ignored -> { }
        }
    }

    private void prepareGlobals(CoreModule module, TypeTable types, ConstantTable constants) {
        for (Function function : module.functions()) {
            types.idOf(function.signature());
        }
        for (Function function : module.functions()) {
            prepareRegion(function.body(), types, constants);
        }
    }

    private void prepareRegion(Region region, TypeTable types, ConstantTable constants) {
        for (Statement statement : region.statements()) {
            switch (statement) {
                case Statement.ReturnVoid ignored -> { }
                case Statement.Return r -> prepareExpr(r.value(), types, constants);
                case Statement.StoreResult s -> prepareExpr(s.value(), types, constants);
                case Statement.BuiltinWrite s -> {
                    types.idOf(s.builtin().type());
                    prepareExpr(s.value(), types, constants);
                }
                case Statement.InterfaceWrite s -> {
                    types.idOf(s.variable().type());
                    prepareExpr(s.value(), types, constants);
                }
                case Statement.BufferStore s -> {
                    prepareExpr(s.index(), types, constants);
                    prepareExpr(s.value(), types, constants);
                }
                case Statement.DeclareVar d -> {
                    types.pointerType(StorageClass.Function.value(), d.variable().type());
                    prepareExpr(d.initializer(), types, constants);
                }
                case Statement.Assign a -> prepareExpr(a.value(), types, constants);
                case Statement.If f -> {
                    prepareExpr(f.condition(), types, constants);
                    prepareRegion(f.thenRegion(), types, constants);
                    prepareRegion(f.elseRegion(), types, constants);
                }
                case Statement.While w -> {
                    prepareExpr(w.condition(), types, constants);
                    prepareRegion(w.body(), types, constants);
                }
            }
        }
    }

    private void prepareExpr(Expr expr, TypeTable types, ConstantTable constants) {
        switch (expr) {
            case Expr.ConstInt c -> constants.intConst(c.type(), c.value());
            case Expr.ConstFloat c -> constants.floatConst(c.type(), c.value());
            case Expr.ConstBool c -> constants.boolConst(c.value());
            case Expr.Read r -> types.idOf(r.variable().type());
            case Expr.InvocationId ignored -> types.idOf(Type.int32());
            case Expr.BuiltinRead r -> types.idOf(r.builtin().type());
            case Expr.InterfaceRead r -> types.idOf(r.variable().type());
            case Expr.BufferLoad l -> {
                types.idOf(l.buffer().element());
                prepareExpr(l.index(), types, constants);
            }
            case Expr.Binary b -> {
                types.idOf(b.type());
                prepareExpr(b.lhs(), types, constants);
                prepareExpr(b.rhs(), types, constants);
            }
            case Expr.VectorConstruct vc -> {
                types.idOf(vc.type());
                vc.components().forEach(c -> prepareExpr(c, types, constants));
            }
            case Expr.VectorExtract ve -> {
                types.idOf(ve.type());
                prepareExpr(ve.vector(), types, constants);
            }
            case Expr.Bitcast bc -> {
                types.idOf(bc.type());
                prepareExpr(bc.operand(), types, constants);
            }
            case Expr.Convert cv -> {
                types.idOf(cv.type());
                prepareExpr(cv.operand(), types, constants);
            }
            case Expr.Unary u -> {
                types.idOf(u.type());
                prepareExpr(u.operand(), types, constants);
            }
            case Expr.Param p -> types.idOf(p.type());
            case Expr.Call c -> {
                types.idOf(c.type());
                c.arguments().forEach(a -> prepareExpr(a, types, constants));
            }
            case Expr.MathCall mc -> {
                types.idOf(mc.type());
                mc.args().forEach(a -> prepareExpr(a, types, constants));
            }
            case Expr.SampleTexture s -> {
                types.idOf(s.type());   // vec4 result
                prepareExpr(s.uv(), types, constants);
            }
            case Expr.PushConstantRead r -> {
                types.idOf(r.type());
                constants.intConst(Type.int32(), r.member());   // the access-chain member index
            }
            case Expr.MatrixTimesVector m -> {
                types.idOf(m.matrix().type());   // the matrix type (OpTypeMatrix)
                types.idOf(m.type());
                prepareExpr(m.matrix(), types, constants);
                prepareExpr(m.vector(), types, constants);
            }
        }
    }

    private static int executionModel(ShaderStage stage) {
        return switch (stage) {
            case COMPUTE -> ExecutionModel.GLCompute.value();
            case VERTEX -> ExecutionModel.Vertex.value();
            case FRAGMENT -> ExecutionModel.Fragment.value();
        };
    }

    // --- module assembly -------------------------------------------------------------------------------

    /** Accumulates instructions into ordered logical sections and concatenates them on {@link #finish()}. */
    private static final class Builder {
        final SpirvModule module = new SpirvModule();
        final List<Instruction> capabilities = new ArrayList<>();
        final List<Instruction> extInstImports = new ArrayList<>();
        final List<Instruction> memoryModel = new ArrayList<>();
        final List<Instruction> entryPoints = new ArrayList<>();
        final List<Instruction> executionModes = new ArrayList<>();
        final List<Instruction> annotations = new ArrayList<>();
        final List<Instruction> globals = new ArrayList<>(); // types, constants, global variables
        final List<Instruction> functions = new ArrayList<>();
        private int glslStd450Id = 0;   // lazily imported on first use; 0 means "not yet imported"

        int allocateId() {
            return module.allocateId();
        }

        Instruction emit(List<Instruction> section, Op op) {
            Instruction instruction = new Instruction(op);
            section.add(instruction);
            return instruction;
        }

        /** The id of the imported {@code GLSL.std.450} extended instruction set, importing it on first use. */
        int glslStd450() {
            if (glslStd450Id == 0) {
                glslStd450Id = allocateId();
                emit(extInstImports, Op.OpExtInstImport).id(glslStd450Id).string("GLSL.std.450");
            }
            return glslStd450Id;
        }

        SpirvModule finish() {
            // OpExtInstImport must follow capabilities/extensions and precede OpMemoryModel (SPIR-V layout).
            List<List<Instruction>> ordered = List.of(capabilities, extInstImports, memoryModel, entryPoints,
                    executionModes, annotations, globals, functions);
            for (List<Instruction> section : ordered) {
                section.forEach(module::add);
            }
            return module;
        }
    }

    /** The shader's output SSBO: {@code layout(set=0, binding=0) buffer { int value; }}. */
    private static final class OutputBuffer {
        private final int variableId;
        private int memberPointerType;
        private int memberIndexConst;

        OutputBuffer(Builder b) {
            this.variableId = b.allocateId(); // needed early for the entry-point interface
        }

        int variableId() {
            return variableId;
        }

        void declare(Builder b, TypeTable types, ConstantTable constants) {
            int intType = types.idOf(Type.int32());
            int blockType = b.allocateId();
            b.emit(b.globals, Op.OpTypeStruct).id(blockType).id(intType);
            int blockPointer = b.allocateId();
            b.emit(b.globals, Op.OpTypePointer).id(blockPointer)
                    .enumValue(StorageClass.StorageBuffer.value()).id(blockType);
            b.emit(b.globals, Op.OpVariable).id(blockPointer).id(variableId)
                    .enumValue(StorageClass.StorageBuffer.value());
            memberPointerType = b.allocateId();
            b.emit(b.globals, Op.OpTypePointer).id(memberPointerType)
                    .enumValue(StorageClass.StorageBuffer.value()).id(intType);
            memberIndexConst = constants.intConst(Type.int32(), 0);

            b.emit(b.annotations, Op.OpDecorate).id(blockType).enumValue(Decoration.Block.value());
            b.emit(b.annotations, Op.OpMemberDecorate)
                    .id(blockType).literal(0).enumValue(Decoration.Offset.value()).literal(0);
            b.emit(b.annotations, Op.OpDecorate)
                    .id(variableId).enumValue(Decoration.DescriptorSet.value()).literal(0);
            b.emit(b.annotations, Op.OpDecorate)
                    .id(variableId).enumValue(Decoration.Binding.value()).literal(0);
        }

        /** Emits {@code OpAccessChain} + {@code OpStore} writing {@code valueId} to element 0. */
        void store(Builder b, int valueId) {
            int pointer = b.allocateId();
            b.emit(b.functions, Op.OpAccessChain).id(memberPointerType).id(pointer)
                    .id(variableId).id(memberIndexConst);
            b.emit(b.functions, Op.OpStore).id(pointer).id(valueId);
        }
    }

    /**
     * Data-parallel kernel resources: storage buffers (runtime arrays of i32, one variable per binding, all
     * sharing one Block struct type) and the {@code GlobalInvocationId} builtin input. Buffer elements are
     * reached with a two-index {@code OpAccessChain} (struct member 0, then the dynamic array index).
     */
    private static final class KernelResources {
        private final List<Buffer> buffers;
        private final boolean useInvocationId;
        private final Map<Integer, Integer> variableByBinding = new LinkedHashMap<>();
        private int gidVariable;

        private int memberIndexConst;   // signed-int 0 (the struct member index), shared by all blocks
        // One Block/runtime-array per distinct element type; a buffer var uses the block for its element type.
        private final Map<Type, Integer> blockPointerByElement = new LinkedHashMap<>();
        private final Map<Type, Integer> memberPointerByElement = new LinkedHashMap<>();
        private int uintType;
        private int gidComponentPointer; // Input* uint
        private int uintZeroConst;       // the .x component index

        KernelResources(Builder b, List<Buffer> buffers, boolean useInvocationId) {
            this.buffers = buffers;
            this.useInvocationId = useInvocationId;
            for (Buffer buffer : buffers) {
                variableByBinding.put(buffer.binding(), b.allocateId());
            }
            if (useInvocationId) {
                gidVariable = b.allocateId();
            }
        }

        boolean usesInvocationId() {
            return useInvocationId;
        }

        List<Integer> interfaceVariables() {
            List<Integer> ids = new ArrayList<>();
            if (useInvocationId) {
                ids.add(gidVariable);
            }
            ids.addAll(variableByBinding.values());
            return ids;
        }

        void declare(Builder b, TypeTable types, ConstantTable constants) {
            memberIndexConst = constants.intConst(Type.int32(), 0);

            // A Block (struct wrapping a runtime array) + its pointer types, one per distinct element type.
            LinkedHashSet<Type> elementTypes = new LinkedHashSet<>();
            for (Buffer buffer : buffers) {
                elementTypes.add(buffer.element());
            }
            for (Type element : elementTypes) {
                int elementType = types.idOf(element);
                int runtimeArray = b.allocateId();
                b.emit(b.globals, Op.OpTypeRuntimeArray).id(runtimeArray).id(elementType);
                int blockType = b.allocateId();
                b.emit(b.globals, Op.OpTypeStruct).id(blockType).id(runtimeArray);
                int blockPointer = b.allocateId();
                b.emit(b.globals, Op.OpTypePointer).id(blockPointer)
                        .enumValue(StorageClass.StorageBuffer.value()).id(blockType);
                blockPointerByElement.put(element, blockPointer);
                memberPointerByElement.put(element, types.pointerType(StorageClass.StorageBuffer.value(), element));
                b.emit(b.annotations, Op.OpDecorate).id(runtimeArray)
                        .enumValue(Decoration.ArrayStride.value()).literal(byteSize(element));
                b.emit(b.annotations, Op.OpMemberDecorate).id(blockType).literal(0)
                        .enumValue(Decoration.Offset.value()).literal(0);
                b.emit(b.annotations, Op.OpDecorate).id(blockType).enumValue(Decoration.Block.value());
            }
            for (Buffer buffer : buffers) {
                int variable = variableByBinding.get(buffer.binding());
                b.emit(b.globals, Op.OpVariable).id(blockPointerByElement.get(buffer.element()))
                        .id(variable).enumValue(StorageClass.StorageBuffer.value());
                b.emit(b.annotations, Op.OpDecorate).id(variable)
                        .enumValue(Decoration.DescriptorSet.value()).literal(0);
                b.emit(b.annotations, Op.OpDecorate).id(variable)
                        .enumValue(Decoration.Binding.value()).literal(buffer.binding());
            }

            if (useInvocationId) {
                uintType = types.idOf(Type.uint32());
                Type.Vector uvec3 = new Type.Vector(Type.uint32(), 3);
                int gidPointer = types.pointerType(StorageClass.Input.value(), uvec3);
                gidComponentPointer = types.pointerType(StorageClass.Input.value(), Type.uint32());
                uintZeroConst = constants.intConst(Type.uint32(), 0);
                b.emit(b.globals, Op.OpVariable).id(gidPointer).id(gidVariable)
                        .enumValue(StorageClass.Input.value());
                b.emit(b.annotations, Op.OpDecorate).id(gidVariable)
                        .enumValue(Decoration.BuiltIn.value()).enumValue(BuiltIn.GlobalInvocationId.value());
            }
        }

        /** Loads gl_GlobalInvocationID.x (uint) and bitcasts it to a signed-int index. */
        int loadInvocationId(Builder b, TypeTable types) {
            int pointer = b.allocateId();
            b.emit(b.functions, Op.OpAccessChain).id(gidComponentPointer).id(pointer)
                    .id(gidVariable).id(uintZeroConst);
            int loaded = b.allocateId();
            b.emit(b.functions, Op.OpLoad).id(uintType).id(loaded).id(pointer);
            int casted = b.allocateId();
            b.emit(b.functions, Op.OpBitcast).id(types.idOf(Type.int32())).id(casted).id(loaded);
            return casted;
        }

        int loadElement(Builder b, TypeTable types, Buffer buffer, int indexId) {
            int pointer = elementPointer(b, buffer, indexId);
            int loaded = b.allocateId();
            b.emit(b.functions, Op.OpLoad).id(types.idOf(buffer.element())).id(loaded).id(pointer);
            return loaded;
        }

        void storeElement(Builder b, TypeTable types, Buffer buffer, int indexId, int valueId) {
            int pointer = elementPointer(b, buffer, indexId);
            b.emit(b.functions, Op.OpStore).id(pointer).id(valueId);
        }

        private int elementPointer(Builder b, Buffer buffer, int indexId) {
            int pointer = b.allocateId();
            b.emit(b.functions, Op.OpAccessChain).id(memberPointerByElement.get(buffer.element())).id(pointer)
                    .id(variableByBinding.get(buffer.binding())).id(memberIndexConst).id(indexId);
            return pointer;
        }

        /** Element size in bytes — the runtime-array {@code ArrayStride}. */
        private static int byteSize(Type element) {
            return switch (element) {
                case Type.Int i -> i.width() / 8;
                case Type.Float f -> f.width() / 8;
                default -> Integer.BYTES;
            };
        }
    }

    /**
     * Graphics-stage interface: built-in variables (decorated {@code BuiltIn}, e.g. {@code gl_Position} as an
     * Output, {@code gl_VertexIndex} as an Input) and user {@code location}-bound interface variables (Input or
     * Output per their direction). Each is a whole-variable {@code OpLoad}/{@code OpStore} (no access chain),
     * and all appear in the entry point's interface list.
     */
    private static final class InterfaceResources {
        private final List<Builtin> builtins;
        private final List<InterfaceVar> variables;
        private final Map<Builtin, Integer> builtinIds = new LinkedHashMap<>();
        private final Map<InterfaceVar, Integer> variableIds = new LinkedHashMap<>();

        InterfaceResources(Builder b, List<Builtin> builtins, List<InterfaceVar> variables) {
            this.builtins = builtins;
            this.variables = variables;
            for (Builtin builtin : builtins) {
                builtinIds.put(builtin, b.allocateId());
            }
            for (InterfaceVar variable : variables) {
                variableIds.put(variable, b.allocateId());
            }
        }

        int builtinVariable(Builtin builtin) {
            return builtinIds.get(builtin);
        }

        int variable(InterfaceVar variable) {
            return variableIds.get(variable);
        }

        List<Integer> interfaceVariables() {
            List<Integer> ids = new ArrayList<>(builtinIds.values());
            ids.addAll(variableIds.values());
            return ids;
        }

        void declare(Builder b, TypeTable types) {
            for (Builtin builtin : builtins) {
                int storageClass = builtin.isInput() ? StorageClass.Input.value() : StorageClass.Output.value();
                // Resolve the pointer type (which emits OpTypePointer) BEFORE emitting OpVariable, so the type
                // is defined first — `emit(OpVariable).id(pointerType(...))` would invert that order.
                int pointerType = types.pointerType(storageClass, builtin.type());
                int variableId = builtinIds.get(builtin);
                b.emit(b.globals, Op.OpVariable).id(pointerType).id(variableId).enumValue(storageClass);
                b.emit(b.annotations, Op.OpDecorate).id(variableId)
                        .enumValue(Decoration.BuiltIn.value()).enumValue(builtInValue(builtin));
            }
            for (InterfaceVar variable : variables) {
                int storageClass = variable.direction() == InterfaceVar.Direction.INPUT
                        ? StorageClass.Input.value() : StorageClass.Output.value();
                int pointerType = types.pointerType(storageClass, variable.type());
                int variableId = variableIds.get(variable);
                b.emit(b.globals, Op.OpVariable).id(pointerType).id(variableId).enumValue(storageClass);
                b.emit(b.annotations, Op.OpDecorate).id(variableId)
                        .enumValue(Decoration.Location.value()).literal(variable.location());
            }
        }

        private static int builtInValue(Builtin builtin) {
            return switch (builtin) {
                case POSITION -> BuiltIn.Position.value();
                case VERTEX_INDEX -> BuiltIn.VertexIndex.value();
            };
        }
    }

    /**
     * Combined image+sampler resources (GLSL {@code sampler2D} / {@code samplerCube}). All textures of a given
     * {@link Texture.Kind} share one {@code OpTypeImage}/{@code OpTypeSampledImage}/pointer type (sampled
     * RGBA-float, 2D or cube); each texture gets one {@code UniformConstant} variable decorated with its
     * descriptor set + binding, and is listed in the entry point's interface (SPIR-V >= 1.4).
     */
    private static final class TextureResources {
        private final List<Texture> textures;
        private final Map<Texture, Integer> variableIds = new LinkedHashMap<>();
        private final Map<Texture.Kind, Integer> sampledImageTypeByKind = new LinkedHashMap<>();

        TextureResources(Builder b, List<Texture> textures) {
            this.textures = textures;
            for (Texture texture : textures) {
                variableIds.put(texture, b.allocateId());
            }
        }

        /** The {@code OpTypeSampledImage} id for a kind, loaded before each sample. Valid after {@link #declare}. */
        int sampledImageType(Texture.Kind kind) {
            return sampledImageTypeByKind.get(kind);
        }

        int variable(Texture texture) {
            return variableIds.get(texture);
        }

        List<Integer> interfaceVariables() {
            return new ArrayList<>(variableIds.values());
        }

        void declare(Builder b, TypeTable types) {
            int floatType = types.idOf(Type.float32());
            Map<Texture.Kind, Integer> pointerTypeByKind = new LinkedHashMap<>();
            for (Texture texture : textures) {
                Texture.Kind kind = texture.kind();
                pointerTypeByKind.computeIfAbsent(kind, k -> declareKind(b, floatType, k));
                int variableId = variableIds.get(texture);
                b.emit(b.globals, Op.OpVariable).id(pointerTypeByKind.get(kind)).id(variableId)
                        .enumValue(StorageClass.UniformConstant.value());
                b.emit(b.annotations, Op.OpDecorate).id(variableId)
                        .enumValue(Decoration.DescriptorSet.value()).literal(texture.set());
                b.emit(b.annotations, Op.OpDecorate).id(variableId)
                        .enumValue(Decoration.Binding.value()).literal(texture.binding());
            }
        }

        /** Emits the shared image/sampled-image/pointer types for a kind, returning the pointer-type id. */
        private int declareKind(Builder b, int floatType, Texture.Kind kind) {
            int dim = kind == Texture.Kind.CUBE ? Dim.Cube.value() : Dim._2D.value();
            int imageType = b.allocateId();
            // OpTypeImage: sampled-type, Dim, Depth 0, Arrayed 0, MS 0, Sampled 1 (with a sampler), Unknown fmt.
            b.emit(b.globals, Op.OpTypeImage).id(imageType).id(floatType)
                    .enumValue(dim).literal(0).literal(0).literal(0).literal(1)
                    .enumValue(ImageFormat.Unknown.value());
            int sampledImageType = b.allocateId();
            b.emit(b.globals, Op.OpTypeSampledImage).id(sampledImageType).id(imageType);
            sampledImageTypeByKind.put(kind, sampledImageType);
            int pointerType = b.allocateId();
            b.emit(b.globals, Op.OpTypePointer).id(pointerType)
                    .enumValue(StorageClass.UniformConstant.value()).id(sampledImageType);
            return pointerType;
        }
    }

    /**
     * The module's push-constant block: a {@code Block}-decorated struct of the declared members in the
     * {@code PushConstant} storage class, with std430-style member offsets (matrices also carry
     * {@code ColMajor} + {@code MatrixStride}). Read via {@link Expr.PushConstantRead}.
     */
    private static final class PushConstantResources {
        private final PushConstants block;
        private final int variableId;

        PushConstantResources(Builder b, PushConstants block) {
            this.block = block;
            this.variableId = b.allocateId();
        }

        int variableId() {
            return variableId;
        }

        void declare(Builder b, TypeTable types, ConstantTable constants) {
            List<PushConstants.Member> members = block.members();
            int[] memberTypeIds = members.stream().mapToInt(m -> types.idOf(m.type())).toArray();
            int structType = b.allocateId();
            Instruction struct = b.emit(b.globals, Op.OpTypeStruct).id(structType);
            for (int memberTypeId : memberTypeIds) {
                struct.id(memberTypeId);
            }
            int pointerType = b.allocateId();
            b.emit(b.globals, Op.OpTypePointer).id(pointerType)
                    .enumValue(StorageClass.PushConstant.value()).id(structType);
            b.emit(b.globals, Op.OpVariable).id(pointerType).id(variableId)
                    .enumValue(StorageClass.PushConstant.value());

            b.emit(b.annotations, Op.OpDecorate).id(structType).enumValue(Decoration.Block.value());
            int offset = 0;
            for (int i = 0; i < members.size(); i++) {
                Type type = members.get(i).type();
                offset = align(offset, alignmentOf(type));
                if (type instanceof Type.Matrix) {
                    b.emit(b.annotations, Op.OpMemberDecorate).id(structType).literal(i)
                            .enumValue(Decoration.ColMajor.value());
                    b.emit(b.annotations, Op.OpMemberDecorate).id(structType).literal(i)
                            .enumValue(Decoration.MatrixStride.value()).literal(16);
                }
                b.emit(b.annotations, Op.OpMemberDecorate).id(structType).literal(i)
                        .enumValue(Decoration.Offset.value()).literal(offset);
                offset += sizeOf(type);
            }
        }

        private static int align(int offset, int alignment) {
            return (offset + alignment - 1) / alignment * alignment;
        }

        /** std430-ish byte size of a push-constant member (32-bit components). */
        private static int sizeOf(Type type) {
            return switch (type) {
                case Type.Matrix m -> m.columns() * 16;
                case Type.Vector v -> v.count() == 3 ? 12 : v.count() * 4;
                default -> 4;
            };
        }

        /** std430-ish alignment of a push-constant member. */
        private static int alignmentOf(Type type) {
            return switch (type) {
                case Type.Matrix ignored -> 16;
                case Type.Vector v -> v.count() == 2 ? 8 : 16;
                default -> 4;
            };
        }
    }

    // --- per-function lowering -------------------------------------------------------------------------

    private static final class FunctionLowering {

        private final Builder b;
        private final TypeTable types;
        private final ConstantTable constants;
        private final OutputBuffer output;
        private final KernelResources kernel;
        private final InterfaceResources interfaceResources;
        private final TextureResources textures;
        private final PushConstantResources pushConstants;
        private final Map<Function, Integer> functionIds;
        private final Map<LocalVar, Integer> variablePointers = new IdentityHashMap<>();
        private final List<Integer> parameterIds = new ArrayList<>();
        private int invocationIdValue; // cached per function (0 = not yet loaded)
        private boolean terminated;    // whether the current block already has a terminator (e.g. early return)

        FunctionLowering(Builder b, TypeTable types, ConstantTable constants, OutputBuffer output,
                KernelResources kernel, InterfaceResources interfaceResources, TextureResources textures,
                PushConstantResources pushConstants, Map<Function, Integer> functionIds) {
            this.b = b;
            this.types = types;
            this.constants = constants;
            this.output = output;
            this.kernel = kernel;
            this.interfaceResources = interfaceResources;
            this.textures = textures;
            this.pushConstants = pushConstants;
            this.functionIds = functionIds;
        }

        void emit(Function function, int functionId) {
            int functionTypeId = types.idOf(function.signature());
            int returnTypeId = types.idOf(function.signature().returnType());

            b.emit(b.functions, Op.OpFunction)
                    .id(returnTypeId).id(functionId)
                    .enumValue(FunctionControl.None.value()).id(functionTypeId);
            for (Type parameterType : function.signature().parameterTypes()) {
                int parameterId = b.allocateId();
                b.emit(b.functions, Op.OpFunctionParameter).id(types.idOf(parameterType)).id(parameterId);
                parameterIds.add(parameterId);
            }

            b.emit(b.functions, Op.OpLabel).id(b.allocateId()); // entry block
            for (LocalVar variable : collectVariables(function.body())) {
                int pointerType = types.pointerType(StorageClass.Function.value(), variable.type());
                int pointerId = b.allocateId();
                b.emit(b.functions, Op.OpVariable).id(pointerType).id(pointerId)
                        .enumValue(StorageClass.Function.value());
                variablePointers.put(variable, pointerId);
            }

            // Load gl_GlobalInvocationID in the entry block (it dominates all others) so the cached id is
            // usable everywhere — a lazy load at first use can land in a branch/loop that doesn't dominate
            // later uses (e.g. code after a loop that contained the first reference).
            if (kernel != null && kernel.usesInvocationId()) {
                invocationIdValue = kernel.loadInvocationId(b, types);
            }

            terminated = false;
            lowerRegion(function.body());
            if (!terminated) {
                // Fall-through off the end: void returns implicitly; a non-void body that fails to return is
                // ill-formed, but keep the block well-formed (every block needs a terminator) with Unreachable.
                b.emit(b.functions, function.signature().returnType() instanceof Type.Void
                        ? Op.OpReturn : Op.OpUnreachable);
            }
            b.emit(b.functions, Op.OpFunctionEnd);
        }

        private void lowerRegion(Region region) {
            for (Statement statement : region.statements()) {
                if (terminated) {
                    break; // statements after a return in the same block are unreachable
                }
                lowerStatement(statement);
            }
        }

        private void lowerStatement(Statement statement) {
            switch (statement) {
                case Statement.ReturnVoid ignored -> {
                    b.emit(b.functions, Op.OpReturn);
                    terminated = true;
                }
                case Statement.Return r -> {
                    // Lower the value (which emits its instructions) BEFORE emitting OpReturnValue, so the
                    // returned id is defined first — `emit(op).id(lowerExpr(...))` would invert that order.
                    int value = lowerExpr(r.value());
                    b.emit(b.functions, Op.OpReturnValue).id(value);
                    terminated = true;
                }
                case Statement.StoreResult s -> output.store(b, lowerExpr(s.value()));
                case Statement.BufferStore s -> {
                    int index = lowerExpr(s.index());
                    int value = lowerExpr(s.value());
                    kernel.storeElement(b, types, s.buffer(), index, value);
                }
                case Statement.BuiltinWrite s -> {
                    int value = lowerExpr(s.value());
                    b.emit(b.functions, Op.OpStore).id(interfaceResources.builtinVariable(s.builtin())).id(value);
                }
                case Statement.InterfaceWrite s -> {
                    int value = lowerExpr(s.value());
                    b.emit(b.functions, Op.OpStore).id(interfaceResources.variable(s.variable())).id(value);
                }
                case Statement.DeclareVar d -> store(d.variable(), lowerExpr(d.initializer()));
                case Statement.Assign a -> store(a.variable(), lowerExpr(a.value()));
                case Statement.If f -> lowerIf(f);
                case Statement.While w -> lowerWhile(w);
            }
        }

        private void store(LocalVar variable, int valueId) {
            b.emit(b.functions, Op.OpStore).id(variablePointers.get(variable)).id(valueId);
        }

        /** Loads gl_GlobalInvocationID.x once per function and reuses the (signed-int) result. */
        private int invocationId() {
            if (invocationIdValue == 0) {
                invocationIdValue = kernel.loadInvocationId(b, types);
            }
            return invocationIdValue;
        }

        private void lowerIf(Statement.If statement) {
            int condition = lowerExpr(statement.condition());
            boolean hasElse = !statement.elseRegion().statements().isEmpty();
            int thenLabel = b.allocateId();
            int mergeLabel = b.allocateId();
            int elseLabel = hasElse ? b.allocateId() : mergeLabel;

            b.emit(b.functions, Op.OpSelectionMerge).id(mergeLabel).enumValue(SelectionControl.None.value());
            b.emit(b.functions, Op.OpBranchConditional).id(condition).id(thenLabel).id(elseLabel);

            b.emit(b.functions, Op.OpLabel).id(thenLabel);
            terminated = false;
            lowerRegion(statement.thenRegion());
            if (!terminated) {
                b.emit(b.functions, Op.OpBranch).id(mergeLabel); // skipped if the branch returned early
            }
            boolean thenTerminated = terminated;

            // Without an else, the false edge of the conditional reaches the merge directly, so that path
            // always falls through.
            boolean elseTerminated = false;
            if (hasElse) {
                b.emit(b.functions, Op.OpLabel).id(elseLabel);
                terminated = false;
                lowerRegion(statement.elseRegion());
                if (!terminated) {
                    b.emit(b.functions, Op.OpBranch).id(mergeLabel);
                }
                elseTerminated = terminated;
            }

            b.emit(b.functions, Op.OpLabel).id(mergeLabel);
            if (thenTerminated && elseTerminated) {
                // Both arms returned, so the merge block has no predecessors; it must still exist (named by
                // OpSelectionMerge) and carry a terminator.
                b.emit(b.functions, Op.OpUnreachable);
                terminated = true;
            } else {
                terminated = false;
            }
        }

        private void lowerWhile(Statement.While statement) {
            int headerLabel = b.allocateId();
            int checkLabel = b.allocateId();
            int bodyLabel = b.allocateId();
            int continueLabel = b.allocateId();
            int mergeLabel = b.allocateId();

            b.emit(b.functions, Op.OpBranch).id(headerLabel);
            b.emit(b.functions, Op.OpLabel).id(headerLabel);
            b.emit(b.functions, Op.OpLoopMerge).id(mergeLabel).id(continueLabel).enumValue(LoopControl.None.value());
            b.emit(b.functions, Op.OpBranch).id(checkLabel);

            b.emit(b.functions, Op.OpLabel).id(checkLabel);
            terminated = false;
            int condition = lowerExpr(statement.condition());
            b.emit(b.functions, Op.OpBranchConditional).id(condition).id(bodyLabel).id(mergeLabel);

            b.emit(b.functions, Op.OpLabel).id(bodyLabel);
            terminated = false;
            lowerRegion(statement.body());
            if (!terminated) {
                b.emit(b.functions, Op.OpBranch).id(continueLabel); // skipped if the body returned early
            }

            b.emit(b.functions, Op.OpLabel).id(continueLabel);
            b.emit(b.functions, Op.OpBranch).id(headerLabel);

            b.emit(b.functions, Op.OpLabel).id(mergeLabel);
            terminated = false; // the loop's false edge always reaches the merge
        }

        private int lowerExpr(Expr expr) {
            return switch (expr) {
                case Expr.ConstInt c -> constants.intConst(c.type(), c.value());
                case Expr.ConstFloat c -> constants.floatConst(c.type(), c.value());
                case Expr.ConstBool c -> constants.boolConst(c.value());
                case Expr.Read r -> {
                    int resultType = types.idOf(r.variable().type());
                    int result = b.allocateId();
                    b.emit(b.functions, Op.OpLoad).id(resultType).id(result).id(variablePointers.get(r.variable()));
                    yield result;
                }
                case Expr.InvocationId ignored -> invocationId();
                case Expr.BufferLoad l -> {
                    int index = lowerExpr(l.index());
                    yield kernel.loadElement(b, types, l.buffer(), index);
                }
                case Expr.BuiltinRead r -> {
                    int resultType = types.idOf(r.builtin().type());
                    int result = b.allocateId();
                    b.emit(b.functions, Op.OpLoad).id(resultType).id(result)
                            .id(interfaceResources.builtinVariable(r.builtin()));
                    yield result;
                }
                case Expr.InterfaceRead r -> {
                    int resultType = types.idOf(r.variable().type());
                    int result = b.allocateId();
                    b.emit(b.functions, Op.OpLoad).id(resultType).id(result)
                            .id(interfaceResources.variable(r.variable()));
                    yield result;
                }
                case Expr.Binary bin -> lowerBinary(bin);
                case Expr.VectorConstruct vc -> {
                    int resultType = types.idOf(vc.type());
                    int[] componentIds = vc.components().stream().mapToInt(this::lowerExpr).toArray();
                    int result = b.allocateId();
                    Instruction instruction = b.emit(b.functions, Op.OpCompositeConstruct).id(resultType).id(result);
                    for (int componentId : componentIds) {
                        instruction.id(componentId);
                    }
                    yield result;
                }
                case Expr.VectorExtract ve -> {
                    int resultType = types.idOf(ve.type());
                    int composite = lowerExpr(ve.vector());
                    int result = b.allocateId();
                    b.emit(b.functions, Op.OpCompositeExtract)
                            .id(resultType).id(result).id(composite).literal(ve.index());
                    yield result;
                }
                case Expr.Bitcast bc -> {
                    // Lower the operand (emitting its instructions) before OpBitcast, so the source id is
                    // defined first — see the emit-order note on Statement.Return.
                    int operand = lowerExpr(bc.operand());
                    int resultType = types.idOf(bc.type());
                    int result = b.allocateId();
                    b.emit(b.functions, Op.OpBitcast).id(resultType).id(result).id(operand);
                    yield result;
                }
                case Expr.Convert cv -> {
                    int operand = lowerExpr(cv.operand());
                    int resultType = types.idOf(cv.type());
                    int result = b.allocateId();
                    b.emit(b.functions, selectConvertOp(cv.operand().type(), cv.type()))
                            .id(resultType).id(result).id(operand);
                    yield result;
                }
                case Expr.Unary u -> {
                    int operand = lowerExpr(u.operand());
                    int resultType = types.idOf(u.type());
                    int result = b.allocateId();
                    b.emit(b.functions, selectUnaryOp(u.op(), u.operand().type()))
                            .id(resultType).id(result).id(operand);
                    yield result;
                }
                case Expr.Param p -> parameterIds.get(p.index());
                case Expr.Call c -> {
                    int resultType = types.idOf(c.type());
                    int[] argumentIds = c.arguments().stream().mapToInt(this::lowerExpr).toArray();
                    int result = b.allocateId();
                    Instruction instruction = b.emit(b.functions, Op.OpFunctionCall)
                            .id(resultType).id(result).id(functionIds.get(c.callee()));
                    for (int argumentId : argumentIds) {
                        instruction.id(argumentId);
                    }
                    yield result;
                }
                case Expr.MathCall mc -> lowerMathCall(mc);
                case Expr.SampleTexture s -> lowerSampleTexture(s);
                case Expr.PushConstantRead r -> lowerPushConstantRead(r);
                case Expr.MatrixTimesVector m -> lowerMatrixTimesVector(m);
            };
        }

        /** Reads a push-constant member: access-chain into the block by member index, then load. */
        private int lowerPushConstantRead(Expr.PushConstantRead read) {
            int memberType = types.idOf(read.type());
            int pointerType = types.pointerType(StorageClass.PushConstant.value(), read.type());
            int memberIndex = constants.intConst(Type.int32(), read.member());
            int pointer = b.allocateId();
            b.emit(b.functions, Op.OpAccessChain).id(pointerType).id(pointer)
                    .id(pushConstants.variableId()).id(memberIndex);
            int result = b.allocateId();
            b.emit(b.functions, Op.OpLoad).id(memberType).id(result).id(pointer);
            return result;
        }

        private int lowerMatrixTimesVector(Expr.MatrixTimesVector mul) {
            int matrix = lowerExpr(mul.matrix());
            int vector = lowerExpr(mul.vector());
            int resultType = types.idOf(mul.type());
            int result = b.allocateId();
            b.emit(b.functions, Op.OpMatrixTimesVector).id(resultType).id(result).id(matrix).id(vector);
            return result;
        }

        /** Loads the combined image+sampler and samples it (implicit LOD) at the coordinate. */
        private int lowerSampleTexture(Expr.SampleTexture sample) {
            int coordinate = lowerExpr(sample.uv());   // lower the coordinate before emitting the load/sample
            int loaded = b.allocateId();
            b.emit(b.functions, Op.OpLoad)
                    .id(textures.sampledImageType(sample.texture().kind())).id(loaded)
                    .id(textures.variable(sample.texture()));
            int resultType = types.idOf(sample.type());
            int result = b.allocateId();
            b.emit(b.functions, Op.OpImageSampleImplicitLod).id(resultType).id(result).id(loaded).id(coordinate);
            return result;
        }

        /** Lowers a math intrinsic to {@code OpDot} or a {@code GLSL.std.450} {@code OpExtInst}. */
        private int lowerMathCall(Expr.MathCall call) {
            int resultType = types.idOf(call.type());
            // Lower the arguments first, then emit the op — emitting first would place it before its operands.
            int[] argIds = call.args().stream().mapToInt(this::lowerExpr).toArray();
            int result = b.allocateId();
            Instruction instruction = call.fn() == MathFn.DOT
                    ? b.emit(b.functions, Op.OpDot).id(resultType).id(result)
                    : b.emit(b.functions, Op.OpExtInst).id(resultType).id(result)
                            .id(b.glslStd450()).literal(glslStd450Number(call.fn()));
            for (int argId : argIds) {
                instruction.id(argId);
            }
            return result;
        }

        /** GLSL.std.450 extended-instruction numbers (float variants); {@code DOT} is the core {@code OpDot}. */
        private static int glslStd450Number(MathFn fn) {
            return switch (fn) {
                // Geometric.
                case LENGTH -> 66;
                case DISTANCE -> 67;
                case CROSS -> 68;
                case NORMALIZE -> 69;
                case FACE_FORWARD -> 70;
                case REFLECT -> 71;
                case REFRACT -> 72;
                // Common elementwise (float variants).
                case POW -> 26;
                case SQRT -> 31;
                case INVERSE_SQRT -> 32;
                case ABS -> 4;          // FAbs
                case SIGN -> 6;         // FSign
                case MIN -> 37;         // FMin
                case MAX -> 40;         // FMax
                case CLAMP -> 43;       // FClamp
                case MIX -> 46;         // FMix
                case STEP -> 48;
                case SMOOTHSTEP -> 49;  // SmoothStep
                case FMA -> 50;         // Fma
                // Exponential / logarithmic.
                case EXP -> 27;
                case LOG -> 28;
                case EXP2 -> 29;
                case LOG2 -> 30;
                // Trigonometric.
                case SIN -> 13;
                case COS -> 14;
                case TAN -> 15;
                case ASIN -> 16;
                case ACOS -> 17;
                case ATAN -> 18;
                case ATAN2 -> 25;       // Atan2
                case RADIANS -> 11;
                case DEGREES -> 12;
                // Hyperbolic.
                case SINH -> 19;
                case COSH -> 20;
                case TANH -> 21;
                case ASINH -> 22;
                case ACOSH -> 23;
                case ATANH -> 24;
                // Rounding.
                case ROUND -> 1;
                case ROUND_EVEN -> 2;   // RoundEven
                case TRUNC -> 3;
                case FLOOR -> 8;
                case CEIL -> 9;
                case FRACT -> 10;
                case DOT -> throw new IllegalArgumentException("DOT lowers to OpDot, not a GLSL.std.450 instruction");
            };
        }

        private int lowerBinary(Expr.Binary binary) {
            int lhs = lowerExpr(binary.lhs());
            int rhs = lowerExpr(binary.rhs());
            int resultType = types.idOf(binary.type());
            int result = b.allocateId();
            b.emit(b.functions, selectOp(binary.op(), binary.lhs().type()))
                    .id(resultType).id(result).id(lhs).id(rhs);
            return result;
        }

        private static Op selectOp(BinaryOp op, Type operandType) {
            // For vector operands, the opcode is chosen by the component type (the op is componentwise).
            Type element = operandType instanceof Type.Vector v ? v.component() : operandType;
            boolean isFloat = element instanceof Type.Float;
            boolean signed = !(element instanceof Type.Int i) || i.signed();
            return switch (op) {
                case ADD -> isFloat ? Op.OpFAdd : Op.OpIAdd;
                case SUB -> isFloat ? Op.OpFSub : Op.OpISub;
                case MUL -> isFloat ? Op.OpFMul : Op.OpIMul;
                case DIV -> isFloat ? Op.OpFDiv : (signed ? Op.OpSDiv : Op.OpUDiv);
                case MOD -> isFloat ? Op.OpFRem : (signed ? Op.OpSRem : Op.OpUMod);
                case BIT_AND -> Op.OpBitwiseAnd;
                case BIT_OR -> Op.OpBitwiseOr;
                case BIT_XOR -> Op.OpBitwiseXor;
                case SHIFT_LEFT -> Op.OpShiftLeftLogical;
                case SHIFT_RIGHT -> signed ? Op.OpShiftRightArithmetic : Op.OpShiftRightLogical;
                case LESS_THAN -> isFloat ? Op.OpFOrdLessThan : (signed ? Op.OpSLessThan : Op.OpULessThan);
                case GREATER_THAN ->
                        isFloat ? Op.OpFOrdGreaterThan : (signed ? Op.OpSGreaterThan : Op.OpUGreaterThan);
                case EQUAL -> isFloat ? Op.OpFOrdEqual : Op.OpIEqual;
                case LOGICAL_AND -> Op.OpLogicalAnd;
                case LOGICAL_OR -> Op.OpLogicalOr;
            };
        }

        /**
         * Numeric conversion opcode, by source and target domain:
         * <ul>
         *   <li>int→int — width change: {@code OpSConvert}/{@code OpUConvert} (SPIR-V ties the opcode, and so
         *       the sign- vs zero-extension on widening, to the <em>result</em> type's signedness; flip
         *       signedness at the same width with {@link Expr.Bitcast} instead);</li>
         *   <li>int→float — {@code OpConvertSToF}/{@code OpConvertUToF}, interpreting the source by its own
         *       signedness;</li>
         *   <li>float→int — {@code OpConvertFToS}/{@code OpConvertFToU} (round toward zero), by target sign;</li>
         *   <li>float→float — {@code OpFConvert} (width change).</li>
         * </ul>
         */
        private static Op selectConvertOp(Type sourceType, Type targetType) {
            Type source = sourceType instanceof Type.Vector v ? v.component() : sourceType;
            Type target = targetType instanceof Type.Vector v ? v.component() : targetType;
            boolean sourceFloat = source instanceof Type.Float;
            boolean targetFloat = target instanceof Type.Float;
            if (!sourceFloat && !targetFloat) {
                return signedInt(target) ? Op.OpSConvert : Op.OpUConvert;
            }
            if (!sourceFloat) {
                return signedInt(source) ? Op.OpConvertSToF : Op.OpConvertUToF;
            }
            if (!targetFloat) {
                return signedInt(target) ? Op.OpConvertFToS : Op.OpConvertFToU;
            }
            return Op.OpFConvert;
        }

        private static boolean signedInt(Type type) {
            return !(type instanceof Type.Int i) || i.signed();
        }

        private static Op selectUnaryOp(UnaryOp op, Type operandType) {
            Type element = operandType instanceof Type.Vector v ? v.component() : operandType;
            boolean isFloat = element instanceof Type.Float;
            return switch (op) {
                case NEGATE -> isFloat ? Op.OpFNegate : Op.OpSNegate;
                case NOT -> Op.OpNot;
                case LOGICAL_NOT -> Op.OpLogicalNot;
            };
        }

        private static List<LocalVar> collectVariables(Region region) {
            List<LocalVar> variables = new ArrayList<>();
            collectVariables(region, variables);
            return variables;
        }

        private static void collectVariables(Region region, List<LocalVar> out) {
            for (Statement statement : region.statements()) {
                switch (statement) {
                    case Statement.DeclareVar d -> out.add(d.variable());
                    case Statement.If f -> {
                        collectVariables(f.thenRegion(), out);
                        collectVariables(f.elseRegion(), out);
                    }
                    case Statement.While w -> collectVariables(w.body(), out);
                    case Statement.Assign ignored -> { }
                    case Statement.ReturnVoid ignored -> { }
                    case Statement.Return ignored -> { }
                    case Statement.StoreResult ignored -> { }
                    case Statement.BufferStore ignored -> { }
                    case Statement.BuiltinWrite ignored -> { }
                    case Statement.InterfaceWrite ignored -> { }
                }
            }
        }
    }

    // --- global type and constant tables ---------------------------------------------------------------

    /** Allocates and emits type declarations on demand, deduplicating by structural equality. */
    private static final class TypeTable {

        private final Builder b;
        private final Map<Type, Integer> ids = new HashMap<>();
        private final Map<PointerKey, Integer> pointerIds = new HashMap<>();
        private boolean usesInt8;
        private boolean usesInt16;
        private boolean usesInt64;
        private boolean usesFloat64;

        private record PointerKey(int storageClass, Type pointee) {}

        TypeTable(Builder b) {
            this.b = b;
        }

        /** Whether an 8-bit integer type was declared — drives the {@code Int8} capability. */
        boolean usesInt8() {
            return usesInt8;
        }

        /** Whether a 16-bit integer type was declared — drives the {@code Int16} capability. */
        boolean usesInt16() {
            return usesInt16;
        }

        /** Whether any 64-bit integer type was declared — drives the {@code Int64} capability. */
        boolean usesInt64() {
            return usesInt64;
        }

        /** Whether any 64-bit float type was declared — drives the {@code Float64} capability. */
        boolean usesFloat64() {
            return usesFloat64;
        }

        int idOf(Type type) {
            Integer cached = ids.get(type);
            if (cached != null) {
                return cached;
            }
            int id = switch (type) {
                case Type.Void ignored -> emit(Op.OpTypeVoid, i -> { });
                case Type.Bool ignored -> emit(Op.OpTypeBool, i -> { });
                case Type.Int t -> {
                    switch (t.width()) {
                        case 8 -> usesInt8 = true;
                        case 16 -> usesInt16 = true;
                        case 64 -> usesInt64 = true;
                        default -> { }
                    }
                    yield emit(Op.OpTypeInt, i -> i.literal(t.width()).literal(t.signed() ? 1 : 0));
                }
                case Type.Float t -> {
                    if (t.width() == 64) {
                        usesFloat64 = true;
                    }
                    yield emit(Op.OpTypeFloat, i -> i.literal(t.width()));
                }
                case Type.Vector t -> {
                    int componentId = idOf(t.component());
                    yield emit(Op.OpTypeVector, i -> i.id(componentId).literal(t.count()));
                }
                case Type.Matrix t -> {
                    int columnId = idOf(t.column());
                    yield emit(Op.OpTypeMatrix, i -> i.id(columnId).literal(t.columns()));
                }
                case Type.FunctionType t -> {
                    int returnId = idOf(t.returnType());
                    int[] paramIds = t.parameterTypes().stream().mapToInt(this::idOf).toArray();
                    yield emit(Op.OpTypeFunction, i -> {
                        i.id(returnId);
                        for (int paramId : paramIds) {
                            i.id(paramId);
                        }
                    });
                }
            };
            ids.put(type, id);
            return id;
        }

        int pointerType(int storageClass, Type pointee) {
            PointerKey key = new PointerKey(storageClass, pointee);
            Integer cached = pointerIds.get(key);
            if (cached != null) {
                return cached;
            }
            int pointeeId = idOf(pointee);
            int id = b.allocateId();
            b.emit(b.globals, Op.OpTypePointer).id(id).enumValue(storageClass).id(pointeeId);
            pointerIds.put(key, id);
            return id;
        }

        private int emit(Op op, java.util.function.Consumer<Instruction> operands) {
            int id = b.allocateId();
            Instruction instruction = b.emit(b.globals, op).id(id);
            operands.accept(instruction);
            return id;
        }
    }

    /** Allocates and emits module-scope constants on demand, deduplicating equal values. */
    private static final class ConstantTable {

        private final Builder b;
        private final TypeTable types;
        private final Map<Key, Integer> ids = new HashMap<>();

        private record Key(String kind, Type type, long bits) {}

        ConstantTable(Builder b, TypeTable types) {
            this.b = b;
            this.types = types;
        }

        int intConst(Type.Int type, long value) {
            return ids.computeIfAbsent(new Key("int", type, value), k -> {
                int typeId = types.idOf(type);
                int id = b.allocateId();
                // Narrow types still occupy a single 32-bit literal word (SPIR-V 2.2.1); 64-bit takes two.
                Instruction constant = b.emit(b.globals, Op.OpConstant).id(typeId).id(id).literal((int) value);
                switch (type.width()) {
                    case 8, 16, 32 -> { }
                    case 64 -> constant.literal((int) (value >>> 32)); // low word emitted above, then high
                    default -> throw new IllegalArgumentException(
                            "unsupported int constant width: " + type.width());
                }
                return id;
            });
        }

        int floatConst(Type.Float type, double value) {
            if (type.width() == 64) {
                long bits = Double.doubleToRawLongBits(value);
                return ids.computeIfAbsent(new Key("float", type, bits), k -> {
                    int typeId = types.idOf(type);
                    int id = b.allocateId();
                    // 64-bit float literals occupy two operand words, low-order first (SPIR-V 2.2.1).
                    b.emit(b.globals, Op.OpConstant).id(typeId).id(id)
                            .literal((int) bits).literal((int) (bits >>> 32));
                    return id;
                });
            }
            if (type.width() != 32) {
                throw new IllegalArgumentException("only 32- and 64-bit float constants are supported, got "
                        + type.width());
            }
            int bits = Float.floatToIntBits((float) value);
            return ids.computeIfAbsent(new Key("float", type, bits & 0xFFFFFFFFL), k -> {
                int typeId = types.idOf(type);
                int id = b.allocateId();
                b.emit(b.globals, Op.OpConstant).id(typeId).id(id).literal(bits);
                return id;
            });
        }

        int boolConst(boolean value) {
            return ids.computeIfAbsent(new Key(Boolean.toString(value), Type.BOOL, 0), k -> {
                int typeId = types.idOf(Type.BOOL);
                int id = b.allocateId();
                b.emit(b.globals, value ? Op.OpConstantTrue : Op.OpConstantFalse).id(typeId).id(id);
                return id;
            });
        }
    }
}
