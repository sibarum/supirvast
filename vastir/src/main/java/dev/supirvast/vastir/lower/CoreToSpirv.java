package dev.supirvast.vastir.lower;

import dev.supirvast.vastir.binary.Instruction;
import dev.supirvast.vastir.binary.SpirvModule;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
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
import dev.supirvast.vastir.spirv.StorageClass;
import dev.supirvast.vastir.type.Type;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

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

    public SpirvModule lower(CoreModule module) {
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

        b.emit(b.capabilities, Op.OpCapability).enumValue(Capability.Shader.value());
        b.emit(b.memoryModel, Op.OpMemoryModel)
                .enumValue(AddressingModel.Logical.value())
                .enumValue(MemoryModel.GLSL450.value());

        emitEntryPoints(module, b, functionIds, output, kernel);
        emitExecutionModes(module, b, functionIds);

        TypeTable types = new TypeTable(b);
        ConstantTable constants = new ConstantTable(b, types);
        if (output != null) {
            output.declare(b, types, constants);
        }
        if (kernel != null) {
            kernel.declare(b, types, constants);
        }
        prepareGlobals(module, types, constants);

        for (Function function : module.functions()) {
            new FunctionLowering(b, types, constants, output, kernel, functionIds)
                    .emit(function, functionIds.get(function));
        }
        return b.finish();
    }

    private void emitEntryPoints(CoreModule module, Builder b, Map<Function, Integer> functionIds,
            OutputBuffer output, KernelResources kernel) {
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
            case Expr.Unary u -> collectBuffers(u.operand(), out);
            case Expr.Call c -> c.arguments().forEach(a -> collectBuffers(a, out));
            case Expr.ConstInt ignored -> { }
            case Expr.ConstFloat ignored -> { }
            case Expr.ConstBool ignored -> { }
            case Expr.Read ignored -> { }
            case Expr.InvocationId ignored -> { }
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
            case Expr.Unary u -> scanForInvocationId(u.operand(), found);
            case Expr.Call c -> c.arguments().forEach(a -> scanForInvocationId(a, found));
            case Expr.ConstInt ignored -> { }
            case Expr.ConstFloat ignored -> { }
            case Expr.ConstBool ignored -> { }
            case Expr.Read ignored -> { }
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
            case Expr.BufferLoad l -> {
                types.idOf(Type.int32());
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
            case Expr.Unary u -> {
                types.idOf(u.type());
                prepareExpr(u.operand(), types, constants);
            }
            case Expr.Param p -> types.idOf(p.type());
            case Expr.Call c -> {
                types.idOf(c.type());
                c.arguments().forEach(a -> prepareExpr(a, types, constants));
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
        final List<Instruction> memoryModel = new ArrayList<>();
        final List<Instruction> entryPoints = new ArrayList<>();
        final List<Instruction> executionModes = new ArrayList<>();
        final List<Instruction> annotations = new ArrayList<>();
        final List<Instruction> globals = new ArrayList<>(); // types, constants, global variables
        final List<Instruction> functions = new ArrayList<>();

        int allocateId() {
            return module.allocateId();
        }

        Instruction emit(List<Instruction> section, Op op) {
            Instruction instruction = new Instruction(op);
            section.add(instruction);
            return instruction;
        }

        SpirvModule finish() {
            List<List<Instruction>> ordered =
                    List.of(capabilities, memoryModel, entryPoints, executionModes, annotations, globals, functions);
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

        private int memberPointerType;  // StorageBuffer* i32
        private int memberIndexConst;   // signed-int 0 (the struct member index)
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

        List<Integer> interfaceVariables() {
            List<Integer> ids = new ArrayList<>();
            if (useInvocationId) {
                ids.add(gidVariable);
            }
            ids.addAll(variableByBinding.values());
            return ids;
        }

        void declare(Builder b, TypeTable types, ConstantTable constants) {
            int intType = types.idOf(Type.int32());
            memberPointerType = types.pointerType(StorageClass.StorageBuffer.value(), Type.int32());
            memberIndexConst = constants.intConst(Type.int32(), 0);

            if (!buffers.isEmpty()) {
                int runtimeArray = b.allocateId();
                b.emit(b.globals, Op.OpTypeRuntimeArray).id(runtimeArray).id(intType);
                int blockType = b.allocateId();
                b.emit(b.globals, Op.OpTypeStruct).id(blockType).id(runtimeArray);
                int blockPointer = b.allocateId();
                b.emit(b.globals, Op.OpTypePointer).id(blockPointer)
                        .enumValue(StorageClass.StorageBuffer.value()).id(blockType);
                for (Buffer buffer : buffers) {
                    b.emit(b.globals, Op.OpVariable).id(blockPointer)
                            .id(variableByBinding.get(buffer.binding()))
                            .enumValue(StorageClass.StorageBuffer.value());
                }
                b.emit(b.annotations, Op.OpDecorate).id(runtimeArray)
                        .enumValue(Decoration.ArrayStride.value()).literal(Integer.BYTES);
                b.emit(b.annotations, Op.OpMemberDecorate).id(blockType).literal(0)
                        .enumValue(Decoration.Offset.value()).literal(0);
                b.emit(b.annotations, Op.OpDecorate).id(blockType).enumValue(Decoration.Block.value());
                for (Buffer buffer : buffers) {
                    int variable = variableByBinding.get(buffer.binding());
                    b.emit(b.annotations, Op.OpDecorate).id(variable)
                            .enumValue(Decoration.DescriptorSet.value()).literal(0);
                    b.emit(b.annotations, Op.OpDecorate).id(variable)
                            .enumValue(Decoration.Binding.value()).literal(buffer.binding());
                }
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
            b.emit(b.functions, Op.OpLoad).id(types.idOf(Type.int32())).id(loaded).id(pointer);
            return loaded;
        }

        void storeElement(Builder b, TypeTable types, Buffer buffer, int indexId, int valueId) {
            int pointer = elementPointer(b, buffer, indexId);
            b.emit(b.functions, Op.OpStore).id(pointer).id(valueId);
        }

        private int elementPointer(Builder b, Buffer buffer, int indexId) {
            int pointer = b.allocateId();
            b.emit(b.functions, Op.OpAccessChain).id(memberPointerType).id(pointer)
                    .id(variableByBinding.get(buffer.binding())).id(memberIndexConst).id(indexId);
            return pointer;
        }
    }

    // --- per-function lowering -------------------------------------------------------------------------

    private static final class FunctionLowering {

        private final Builder b;
        private final TypeTable types;
        private final ConstantTable constants;
        private final OutputBuffer output;
        private final KernelResources kernel;
        private final Map<Function, Integer> functionIds;
        private final Map<LocalVar, Integer> variablePointers = new IdentityHashMap<>();
        private final List<Integer> parameterIds = new ArrayList<>();
        private int invocationIdValue; // cached per function (0 = not yet loaded)

        FunctionLowering(Builder b, TypeTable types, ConstantTable constants, OutputBuffer output,
                KernelResources kernel, Map<Function, Integer> functionIds) {
            this.b = b;
            this.types = types;
            this.constants = constants;
            this.output = output;
            this.kernel = kernel;
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

            lowerRegion(function.body());
            b.emit(b.functions, Op.OpFunctionEnd);
        }

        private void lowerRegion(Region region) {
            for (Statement statement : region.statements()) {
                lowerStatement(statement);
            }
        }

        private void lowerStatement(Statement statement) {
            switch (statement) {
                case Statement.ReturnVoid ignored -> b.emit(b.functions, Op.OpReturn);
                case Statement.Return r -> {
                    // Lower the value (which emits its instructions) BEFORE emitting OpReturnValue, so the
                    // returned id is defined first — `emit(op).id(lowerExpr(...))` would invert that order.
                    int value = lowerExpr(r.value());
                    b.emit(b.functions, Op.OpReturnValue).id(value);
                }
                case Statement.StoreResult s -> output.store(b, lowerExpr(s.value()));
                case Statement.BufferStore s -> {
                    int index = lowerExpr(s.index());
                    int value = lowerExpr(s.value());
                    kernel.storeElement(b, types, s.buffer(), index, value);
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
            lowerRegion(statement.thenRegion());
            b.emit(b.functions, Op.OpBranch).id(mergeLabel);

            if (hasElse) {
                b.emit(b.functions, Op.OpLabel).id(elseLabel);
                lowerRegion(statement.elseRegion());
                b.emit(b.functions, Op.OpBranch).id(mergeLabel);
            }

            b.emit(b.functions, Op.OpLabel).id(mergeLabel);
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
            int condition = lowerExpr(statement.condition());
            b.emit(b.functions, Op.OpBranchConditional).id(condition).id(bodyLabel).id(mergeLabel);

            b.emit(b.functions, Op.OpLabel).id(bodyLabel);
            lowerRegion(statement.body());
            b.emit(b.functions, Op.OpBranch).id(continueLabel);

            b.emit(b.functions, Op.OpLabel).id(continueLabel);
            b.emit(b.functions, Op.OpBranch).id(headerLabel);

            b.emit(b.functions, Op.OpLabel).id(mergeLabel);
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

        private record PointerKey(int storageClass, Type pointee) {}

        TypeTable(Builder b) {
            this.b = b;
        }

        int idOf(Type type) {
            Integer cached = ids.get(type);
            if (cached != null) {
                return cached;
            }
            int id = switch (type) {
                case Type.Void ignored -> emit(Op.OpTypeVoid, i -> { });
                case Type.Bool ignored -> emit(Op.OpTypeBool, i -> { });
                case Type.Int t -> emit(Op.OpTypeInt, i -> i.literal(t.width()).literal(t.signed() ? 1 : 0));
                case Type.Float t -> emit(Op.OpTypeFloat, i -> i.literal(t.width()));
                case Type.Vector t -> {
                    int componentId = idOf(t.component());
                    yield emit(Op.OpTypeVector, i -> i.id(componentId).literal(t.count()));
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
            require32(type.width());
            return ids.computeIfAbsent(new Key("int", type, value), k -> {
                int typeId = types.idOf(type);
                int id = b.allocateId();
                b.emit(b.globals, Op.OpConstant).id(typeId).id(id).literal((int) value);
                return id;
            });
        }

        int floatConst(Type.Float type, double value) {
            require32(type.width());
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

        private static void require32(int width) {
            if (width != 32) {
                throw new IllegalArgumentException("only 32-bit scalar constants are supported, got " + width);
            }
        }
    }
}
