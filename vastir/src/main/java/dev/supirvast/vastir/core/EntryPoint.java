package dev.supirvast.vastir.core;

import java.util.Optional;

/**
 * A shader entry point: a {@link Function} exposed at a given {@link ShaderStage}.
 *
 * <p>Compute entry points carry a {@link WorkgroupSize}; other stages leave it empty.
 */
public record EntryPoint(Function function, ShaderStage stage, Optional<WorkgroupSize> workgroupSize) {

    /** Local workgroup dimensions for a compute entry point. */
    public record WorkgroupSize(int x, int y, int z) {}

    public static EntryPoint compute(Function function, int x, int y, int z) {
        return new EntryPoint(function, ShaderStage.COMPUTE, Optional.of(new WorkgroupSize(x, y, z)));
    }

    public static EntryPoint of(Function function, ShaderStage stage) {
        return new EntryPoint(function, stage, Optional.empty());
    }
}
