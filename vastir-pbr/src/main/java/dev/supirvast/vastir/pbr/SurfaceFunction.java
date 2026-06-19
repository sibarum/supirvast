package dev.supirvast.vastir.pbr;

import dev.supirvast.vastir.core.Expr;

import java.util.Map;

/**
 * The author-supplied "surface" of a material: given the fragment {@link SurfaceInputs}, return an expression
 * for each declared {@link Channel}. This is the ShaderLab-style surface program, written at the core-IR level
 * — set albedo/metallic/roughness/… from constants, math, or buffer reads; {@link PbrShader} wires the result
 * into the lighting model. The returned map's keys must be exactly the channels the material declared, and each
 * expression's type must match its {@link Channel#type()}.
 */
@FunctionalInterface
public interface SurfaceFunction {
    Map<Channel, Expr> evaluate(SurfaceInputs inputs);
}
