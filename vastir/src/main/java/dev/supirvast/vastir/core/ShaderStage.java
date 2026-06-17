package dev.supirvast.vastir.core;

/**
 * Target-neutral shader stage at the core level.
 *
 * <p>Deliberately a small core enum rather than the generated SPIR-V {@code ExecutionModel}: the core level
 * stays independent of SPIR-V vocabulary, and {@code CoreToSpirv} maps these to execution models.
 */
public enum ShaderStage {
    COMPUTE,
    VERTEX,
    FRAGMENT
}
